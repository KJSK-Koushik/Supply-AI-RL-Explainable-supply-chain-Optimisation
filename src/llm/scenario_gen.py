"""Phase 7: have an LLM invent disruptions, then refuse to trust any of it.

The value of an LLM here is imagination, not authority. Hand-written stress
tests reflect the failures their author already thought of; a model asked for
"a plausible supply chain crisis" produces combinations nobody enumerated --
a festival spike landing while the cheap supplier is offline, a slow squeeze
rather than a single shock.

That is worth having, and it is the only thing taken from the model. Every
field is parsed, type-checked, range-clamped and bounds-checked against the
actual product and supplier counts before it reaches the simulator. Free text
never becomes executable content: the output is data that either fits the
Scenario schema or is discarded.
"""

from __future__ import annotations

import json
import re

from src.config import load_config
from src.env.scenarios import Scenario
from src.llm.client import OpenRouterClient

VALID_TYPES: tuple[str, ...] = ("demand_spike", "supplier_delay", "supplier_outage", "price_shock")

SYSTEM_PROMPT = """You design stress tests for a retail inventory simulator.

The world: 10 products (index 0-9) sold over a 180-day episode, supplied by 3 suppliers.
  supplier 0 = cheap, slow (5-8 day lead time, ships 90% of what is ordered)
  supplier 1 = fast, expensive (1-3 days, 98%)
  supplier 2 = mid-priced, unreliable (3-5 days, 80%, fails most often)

Return ONLY a JSON object, no prose and no code fences, of the form:

{"scenarios": [
  {"type": "demand_spike", "start_day": 60, "duration": 14, "products": [0, 3], "multiplier": 2.5, "label": "short reason"},
  {"type": "supplier_delay", "start_day": 70, "duration": 10, "supplier": 0, "extra_lead_days": 4, "label": "short reason"},
  {"type": "supplier_outage", "start_day": 90, "duration": 7, "supplier": 2, "label": "short reason"},
  {"type": "price_shock", "start_day": 100, "duration": 20, "cost_multiplier": 1.4, "label": "short reason"}
]}

Rules:
- "type" must be one of: demand_spike, supplier_delay, supplier_outage, price_shock.
- start_day between 20 and 170. duration between 1 and 30.
- products is a list of indices 0-9, or omit it to affect every product.
- multiplier between 0.2 and 5.0. cost_multiplier between 0.5 and 3.0.
- extra_lead_days between 0 and 20. supplier is 0, 1 or 2.
- "label" is a few words naming the real-world cause.
- Make the scenarios interact. Overlapping disruptions are the interesting case."""


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _extract_json(text: str) -> dict | None:
    """Pull the JSON object out of whatever the model wrapped it in.

    Models add code fences and preambles no matter how firmly the prompt asks
    them not to, so the parser handles it rather than discarding otherwise
    valid output over formatting.
    """
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def validate(raw: dict, n_products: int, n_suppliers: int, cfg: dict) -> list[Scenario]:
    """Turn model output into Scenarios, dropping anything that does not fit.

    Silently skipping a malformed entry is deliberate: one bad item in a list
    of five should cost that item, not the whole batch. What is never done is
    repairing a value by guessing what was meant.
    """
    clamps = cfg["scenario_generator"]["clamps"]
    dur_lo, dur_hi = clamps["duration_days"]
    mult_lo, mult_hi = clamps["demand_multiplier"]
    lead_lo, lead_hi = clamps["lead_time_extra_days"]
    out_lo, out_hi = clamps["outage_days"]
    cost_lo, cost_hi = clamps.get("cost_multiplier", [0.5, 3.0])
    day_lo, day_hi = clamps.get("start_day", [20, 170])
    episode_length = int(load_config("env")["simulation"]["episode_length"])

    items = raw.get("scenarios") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        return []

    built: list[Scenario] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind not in VALID_TYPES:
            continue
        try:
            duration = int(_clamp(float(item.get("duration", 7)), dur_lo, dur_hi))
            start = int(_clamp(float(item.get("start_day", 60)), day_lo, day_hi))
        except (TypeError, ValueError):
            continue
        # Keep the whole window inside the episode. Clamping start_day alone is
        # not enough: a model asking for day 400 for 30 days clamps to day 170,
        # which on a 180-day episode leaves 10 active days and 20 that never
        # happen -- a stress test that silently barely runs. Pull the start back
        # so the disruption lands in full, and only shorten it if it cannot.
        if start + duration > episode_length:
            start = max(day_lo, episode_length - duration)
            duration = min(duration, episode_length - start)
        if duration < 1:
            continue

        sc = Scenario(
            type=kind,  # type: ignore[arg-type]
            start_day=start,
            duration=duration,
            label=str(item.get("label", ""))[:60],
        )

        if kind == "demand_spike":
            try:
                sc.multiplier = _clamp(float(item.get("multiplier", 2.0)), mult_lo, mult_hi)
            except (TypeError, ValueError):
                continue
            products = item.get("products")
            if isinstance(products, list):
                # Out-of-range indices are dropped, not wrapped: index 14 on a
                # 10-product world is a mistake, and %10 would silently turn it
                # into a different product's crisis.
                keep = [
                    int(p) for p in products if isinstance(p, (int, float)) and 0 <= p < n_products
                ]
                sc.products = keep or None
            else:
                sc.products = None

        elif kind in ("supplier_delay", "supplier_outage"):
            supplier = item.get("supplier")
            if not isinstance(supplier, (int, float)) or not 0 <= int(supplier) < n_suppliers:
                continue
            sc.supplier = int(supplier)
            if kind == "supplier_delay":
                try:
                    sc.extra_lead_days = int(
                        _clamp(float(item.get("extra_lead_days", 3)), lead_lo, lead_hi)
                    )
                except (TypeError, ValueError):
                    continue
            else:
                sc.duration = int(_clamp(duration, out_lo, out_hi))

        elif kind == "price_shock":
            try:
                sc.cost_multiplier = _clamp(
                    float(item.get("cost_multiplier", 1.3)), cost_lo, cost_hi
                )
            except (TypeError, ValueError):
                continue

        built.append(sc)
    return built


def fallback_scenarios() -> list[Scenario]:
    """Hand-written crises, used when no key is set or every model refuses.

    Chosen to overlap: the outage lands inside the spike, which is the case
    that actually hurts and the one a single-disruption test never reaches.
    """
    from src.env.scenarios import demand_spike, price_shock, supplier_outage

    return [
        demand_spike(None, 2.5, start_day=60, duration=20, label="seasonal rush"),
        supplier_outage(0, start_day=68, duration=10, label="cheap supplier offline mid-rush"),
        price_shock(1.35, start_day=110, duration=25, label="input cost spike"),
    ]


def generate(
    n: int = 4,
    theme: str = "a realistic retail supply chain crisis",
    client: OpenRouterClient | None = None,
    cfg: dict | None = None,
    n_products: int = 10,
    n_suppliers: int = 3,
) -> dict:
    """Ask for `n` scenarios. Always returns a usable list."""
    cfg = cfg or load_config("llm")
    settings = cfg["scenario_generator"]

    client = client or OpenRouterClient(cfg)
    if not client.available:
        return {
            "scenarios": fallback_scenarios(),
            "source": "fallback",
            "reason": "no API key",
            "model": None,
        }

    result = client.complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Generate {n} interacting scenarios describing {theme}."},
        ],
        temperature=float(settings.get("temperature", 0.8)),
        max_tokens=int(settings.get("max_tokens", 500)),
    )
    if not result.ok:
        return {
            "scenarios": fallback_scenarios(),
            "source": "fallback",
            "reason": "; ".join(result.attempts[-3:]) or "no response",
            "model": None,
        }

    raw = _extract_json(result.text)
    if raw is None:
        return {
            "scenarios": fallback_scenarios(),
            "source": "fallback",
            "reason": "model output was not JSON",
            "model": result.model,
        }

    scenarios = validate(raw, n_products, n_suppliers, cfg)
    if not scenarios:
        return {
            "scenarios": fallback_scenarios(),
            "source": "fallback",
            "reason": "no scenario survived validation",
            "model": result.model,
        }
    return {"scenarios": scenarios, "source": "llm", "reason": None, "model": result.model}
