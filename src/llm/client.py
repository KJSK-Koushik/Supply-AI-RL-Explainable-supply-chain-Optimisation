"""OpenRouter chat client, built for free-tier endpoints.

Free models are rate-limited upstream and refuse requests routinely: on first
contact all three configured models returned HTTP 429, and two answered
normally on the very next call. So the retry ladder and the fall-through to the
next model are what make this usable at all, not defensive decoration.

The client never raises for an unavailable model. It returns None, and the
caller falls back to the deterministic template explainer. A demo that dies
because a free endpoint was busy is worse than one that explains itself in
plainer words.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from src.config import load_config

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _load_dotenv() -> None:
    """Read .env without adding a dependency just for this.

    python-dotenv is in requirements, but importing it here would make the
    module fail on a machine that has not installed it yet -- and the whole
    point of the fallback path is that Phase 6 degrades instead of breaking.
    """
    path = ".env"
    if not os.path.exists(path) or os.environ.get("OPENROUTER_API_KEY"):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


@dataclass
class LLMResult:
    """What came back, and from where. `text` is None when every model failed."""

    text: str | None
    model: str | None = None
    attempts: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.text)


class OpenRouterClient:
    def __init__(self, cfg: dict | None = None):
        _load_dotenv()
        self.cfg = (cfg or load_config("llm"))["provider"]
        self.api_key = os.environ.get(self.cfg["api_key_env"], "").strip()
        self.models: list[str] = list(self.cfg["models"])
        self.reasoning = set(self.cfg.get("reasoning_models", []))
        self.backoff: list[float] = list(self.cfg.get("retry_backoff_seconds", [1, 3, 8]))
        self.timeout = float(self.cfg.get("timeout_seconds", 60))

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _post(self, model: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
        body = json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode()
        req = urllib.request.Request(
            f"{self.cfg['base_url']}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                # OpenRouter attributes free-tier usage with these.
                "HTTP-Referer": "https://github.com/KJSK-Koushik/Supply-AI-RL-Explainable-supply-chain-Optimisation",
                "X-Title": "SupplyAI-RL",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.load(resp)

        choice = payload["choices"][0]
        text = (choice["message"].get("content") or "").strip()
        if not text and choice.get("finish_reason") == "length":
            # A reasoning model that spent its whole budget thinking. Report it
            # as a failure so the caller moves on rather than surfacing "".
            raise ValueError(f"{model} returned no answer (token budget exhausted)")
        return text

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 300,
    ) -> LLMResult:
        """Try each model in turn, retrying the ones that are merely busy."""
        attempts: list[str] = []
        if not self.available:
            attempts.append("no OPENROUTER_API_KEY set")
            return LLMResult(None, attempts=attempts)

        for model in self.models:
            budget = (
                int(self.cfg.get("reasoning_max_tokens", 1500))
                if model in self.reasoning
                else max_tokens
            )
            for attempt in range(len(self.backoff) + 1):
                try:
                    return LLMResult(
                        self._post(model, messages, temperature, budget),
                        model=model,
                        attempts=attempts,
                    )
                except urllib.error.HTTPError as exc:
                    attempts.append(f"{model}: HTTP {exc.code}")
                    if exc.code not in RETRYABLE_STATUS:
                        break  # 401, 404 and friends will not fix themselves
                except Exception as exc:  # timeouts, malformed payloads
                    attempts.append(f"{model}: {type(exc).__name__}")
                if attempt < len(self.backoff):
                    time.sleep(self.backoff[attempt])
        return LLMResult(None, attempts=attempts)
