"""Phase 7: LLM-generated disruptions, and the wall between them and the simulator.

The generator's job is to be imaginative. This file's job is to prove nothing
imaginative reaches the environment unchecked. Every test here feeds the
validator output a cooperative model would never produce, because a
cooperative model is not the case worth testing.
"""

from __future__ import annotations

import os

import pytest

from src.config import load_config, resolve
from src.llm.client import LLMResult
from src.llm.scenario_gen import (
    VALID_TYPES,
    _extract_json,
    fallback_scenarios,
    generate,
    validate,
)

stats_path = resolve(load_config("data")["paths"]["demand_stats"])
pytestmark = pytest.mark.skipif(
    not stats_path.exists(), reason="run `python -m src.data.run_pipeline` first"
)

CFG = load_config("llm")
EPISODE = int(load_config("env")["simulation"]["episode_length"])


class StubClient:
    def __init__(self, text: str | None, available: bool = True):
        self._text = text
        self.available = available

    def complete(self, *_args, **_kwargs):
        if self._text is None:
            return LLMResult(None, attempts=["stub: forced failure"])
        return LLMResult(self._text, model="stub-model")


def build(payload: str):
    return validate(_extract_json(payload), n_products=10, n_suppliers=3, cfg=CFG)


# ------------------------------------------------------------- parsing


def test_code_fences_do_not_defeat_the_parser():
    fenced = (
        '```json\n{"scenarios": [{"type": "price_shock", "start_day": 50, "duration": 5}]}\n```'
    )
    assert len(build(fenced)) == 1


def test_prose_around_the_json_is_tolerated():
    chatty = (
        "Sure! Here are your scenarios:\n"
        '{"scenarios": [{"type": "price_shock", "start_day": 50, "duration": 5}]}\n'
        "Let me know if you want more."
    )
    assert len(build(chatty)) == 1


def test_unparseable_output_yields_nothing_rather_than_guessing():
    assert _extract_json("I would rather not.") is None


# ---------------------------------------------------------- rejection


def test_unknown_scenario_type_is_rejected():
    assert (
        build('{"scenarios": [{"type": "teleport_stock", "start_day": 50, "duration": 5}]}') == []
    )


def test_supplier_outside_the_real_range_is_rejected():
    payload = '{"scenarios": [{"type": "supplier_outage", "start_day": 50, "duration": 5, "supplier": 9}]}'
    assert build(payload) == []


def test_non_numeric_value_is_rejected_not_coerced():
    payload = (
        '{"scenarios": [{"type": "price_shock", "start_day": 50, "duration": 5,'
        ' "cost_multiplier": "lots"}]}'
    )
    assert build(payload) == []


def test_one_bad_item_does_not_discard_the_good_ones():
    payload = (
        '{"scenarios": ['
        '{"type": "nonsense"},'
        '{"type": "price_shock", "start_day": 50, "duration": 5, "cost_multiplier": 1.2}]}'
    )
    assert len(build(payload)) == 1


# ------------------------------------------------------------ clamping


def test_absurd_values_are_clamped_to_the_configured_range():
    payload = (
        '{"scenarios": [{"type": "demand_spike", "start_day": 60, "duration": 999,'
        ' "multiplier": 87.0}]}'
    )
    sc = build(payload)[0]
    lo, hi = CFG["scenario_generator"]["clamps"]["demand_multiplier"]
    assert lo <= sc.multiplier <= hi
    assert 1 <= sc.duration <= CFG["scenario_generator"]["clamps"]["duration_days"][1]


def test_out_of_range_products_are_dropped_not_wrapped():
    """Index 14 on a ten-product world is a mistake. Wrapping it with %10 would
    silently redirect the crisis onto product 4."""
    payload = (
        '{"scenarios": [{"type": "demand_spike", "start_day": 60, "duration": 5,'
        ' "products": [0, 3, 14], "multiplier": 2.0}]}'
    )
    assert build(payload)[0].products == [0, 3]


@pytest.mark.parametrize(
    "start,duration",
    [(400, 999), (175, 14), (170, 30), (20, 30)],
)
def test_every_disruption_lands_inside_the_episode(start, duration):
    payload = (
        f'{{"scenarios": [{{"type": "supplier_outage", "start_day": {start},'
        f' "duration": {duration}, "supplier": 1}}]}}'
    )
    built = build(payload)
    assert built, "scenario should survive, only be repositioned"
    sc = built[0]
    assert sc.start_day >= 1
    assert sc.start_day + sc.duration <= EPISODE


# ------------------------------------------------------------ fallback


def test_fallback_is_used_when_the_model_returns_prose():
    out = generate(client=StubClient("I cannot help with that."))
    assert out["source"] == "fallback"
    assert "not JSON" in out["reason"]
    assert out["scenarios"]


def test_fallback_is_used_when_nothing_survives_validation():
    out = generate(client=StubClient('{"scenarios": [{"type": "teleport_stock"}]}'))
    assert out["source"] == "fallback"
    assert "validation" in out["reason"]


def test_fallback_scenarios_are_themselves_valid_and_overlapping():
    scenarios = fallback_scenarios()
    assert scenarios
    for sc in scenarios:
        assert sc.type in VALID_TYPES
        assert sc.start_day + sc.duration <= EPISODE
    windows = [(s.start_day, s.start_day + s.duration) for s in scenarios]
    assert any(
        a[0] < b[1] and b[0] < a[1] for i, a in enumerate(windows) for b in windows[i + 1 :]
    ), "the interesting case is disruptions that overlap"


# --------------------------------------------------------- integration


def test_generated_scenarios_actually_hurt_the_simulator():
    """The point of a stress test is that it stresses something.

    A scenario set that parses cleanly but leaves profit unchanged would be
    decoration, and every test above would still pass.
    """
    import json

    from src.agents.tune_baselines import make
    from src.eval.runner import run_episode

    path = resolve("results/baselines.json")
    if not path.exists():
        pytest.skip("no tuned baseline stored")
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    policy = make(data["best_baseline"], data["policies"][data["best_baseline"]]["params"])

    calm = run_episode(policy, seed=500)["total_profit"]
    rough = run_episode(policy, seed=500, scenarios=fallback_scenarios())["total_profit"]
    assert rough < calm, f"disruption did not bite: {rough:,.0f} vs {calm:,.0f}"


@pytest.mark.skipif(
    os.environ.get("SUPPLYAI_LIVE_LLM") != "1",
    reason="live call; set SUPPLYAI_LIVE_LLM=1 to run",
)
def test_live_generation_produces_usable_scenarios():
    out = generate(n=4)
    assert out["scenarios"]
    for sc in out["scenarios"]:
        assert sc.type in VALID_TYPES
        assert sc.start_day + sc.duration <= EPISODE
