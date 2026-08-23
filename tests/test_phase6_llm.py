"""Phase 6: the LLM explainer.

Everything here runs offline with a stub client. The explainer's job is not to
be clever, it is to never put a number in front of a user that the simulator
did not produce -- and that is testable without spending a single token.

One live test exists and skips itself without a key, because free endpoints
rate-limit and a suite that fails when someone else's server is busy is a suite
people learn to ignore.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.config import load_config, resolve
from src.llm.client import LLMResult, OpenRouterClient
from src.llm.explainer import (
    build_facts,
    explain,
    template_explanation,
    ungrounded_numbers,
)

stats_path = resolve(load_config("data")["paths"]["demand_stats"])
pytestmark = pytest.mark.skipif(
    not stats_path.exists(), reason="run `python -m src.data.run_pipeline` first"
)


class StubClient:
    """Stands in for OpenRouter. `available` and the reply are dictated."""

    def __init__(self, text: str | None, available: bool = True):
        self._text = text
        self.available = available

    def complete(self, *_args, **_kwargs):
        if self._text is None:
            return LLMResult(None, attempts=["stub: forced failure"])
        return LLMResult(self._text, model="stub-model")


@pytest.fixture
def facts():
    from src.env.supply_chain_env import SupplyChainEnv

    env = SupplyChainEnv(seed=500)
    env.reset(seed=500)
    info = None
    for _ in range(30):
        action = np.full(env.n_products, 2 * env.n_suppliers, dtype=int)
        _, _, _, _, info = env.step(action)
    return build_facts(env, info, product=0)


# ------------------------------------------------------------------- facts


def test_facts_describe_the_decision_that_was_made(facts):
    for key in (
        "product_code",
        "stock_on_hand_units",
        "days_of_cover",
        "typical_daily_demand_units",
        "order_placed_units",
    ):
        assert key in facts, key
    assert facts["stock_on_hand_units"] >= 0
    assert facts["days_of_cover"] >= 0


def test_supplier_facts_appear_only_when_an_order_was_placed(facts):
    if facts["order_placed_units"] > 0:
        assert "supplier_name" in facts
    else:
        assert "supplier_name" not in facts


# -------------------------------------------------------------- grounding


def test_template_output_is_entirely_grounded(facts):
    """The fallback must pass the same check the LLM is held to.

    If it did not, every fallback would be rejected by its own guard, and the
    guard would be measuring the wrong thing.
    """
    assert ungrounded_numbers(template_explanation(facts), facts) == []


def test_invented_numbers_are_caught(facts):
    assert ungrounded_numbers("We hold 987654 units today.", facts) == ["987654"]


def test_thousands_separators_do_not_read_as_invented(facts):
    stock = facts["stock_on_hand_units"]
    assert ungrounded_numbers(f"Stock is {stock:,} units.", facts) == []


# ----------------------------------------------------------------- explain


def test_falls_back_to_template_without_a_key(facts):
    out = explain(facts, client=StubClient(None, available=False))
    assert out["source"] == "template"
    assert out["text"] == template_explanation(facts)


def test_falls_back_when_every_model_fails(facts):
    out = explain(facts, client=StubClient(None))
    assert out["source"] == "template"
    assert "stub" in out["reason"]


def test_model_output_with_an_invented_number_is_discarded(facts):
    """The whole point of Phase 6. A fluent sentence containing a figure the
    simulator never produced is worse than a plain one, because it is credible."""
    out = explain(facts, client=StubClient("We ordered 424242 units, a routine top-up."))
    assert out["source"] == "template"
    assert "424242" in out["reason"]
    assert "424242" not in out["text"]


def test_grounded_model_output_is_kept(facts):
    good = f"Stock stands at {facts['stock_on_hand_units']} units."
    out = explain(facts, client=StubClient(good))
    assert out["source"] == "llm"
    assert out["text"] == good


# --------------------------------------------------------------- live call


@pytest.mark.skipif(
    os.environ.get("SUPPLYAI_LIVE_LLM") != "1",
    reason="live call; set SUPPLYAI_LIVE_LLM=1 to run",
)
def test_live_call_returns_grounded_prose(facts):
    """Opt-in, because it is slow by design.

    Every configured model is rate-limited on the free tier, so a single call
    can walk the whole retry ladder across three models before answering. That
    is correct behaviour and it takes over a minute -- which is fine for a
    check you ask for, and intolerable in the suite you run after every edit.
    """
    client = OpenRouterClient()
    if not client.available:
        pytest.skip("no OPENROUTER_API_KEY set")
    out = explain(facts, client=client)
    assert out["text"]
    if out["source"] == "llm":
        assert ungrounded_numbers(out["text"], facts) == []
    else:
        # Free endpoints rate-limit constantly. Falling back is correct
        # behaviour, not a failure -- but it should say why.
        assert out["reason"]
