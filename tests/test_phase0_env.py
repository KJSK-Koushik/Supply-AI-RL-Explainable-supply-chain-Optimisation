"""Phase 0 smoke test: confirm every dependency imports and configs parse.

Catches install problems now rather than at Phase 4, when a broken torch
would waste a training run.
"""

from __future__ import annotations

import pytest

from src.config import load_config, resolve


def test_core_imports():
    import numpy, pandas, scipy, yaml, pydantic  # noqa: F401


def test_rl_stack_imports():
    import gymnasium, torch, stable_baselines3  # noqa: F401
    from sb3_contrib import MaskablePPO  # noqa: F401


def test_dashboard_imports():
    import streamlit, plotly, matplotlib  # noqa: F401


def test_llm_client_imports():
    import openai, tenacity, dotenv  # noqa: F401


@pytest.mark.parametrize("name", ["data", "env", "llm", "suppliers"])
def test_configs_parse(name):
    cfg = load_config(name)
    assert isinstance(cfg, dict) and cfg


def test_raw_data_present():
    """Local environment check. Raw data is gitignored (50 MB, and the source
    terms are not ours to redistribute), so on CI this skips rather than fails.
    Run `python scripts/fetch_data.py` to populate it."""
    cfg = load_config("data")
    uci = resolve(cfg["paths"]["uci_raw"])
    if not uci.exists():
        pytest.skip("raw data not present (expected on CI)")
    assert resolve(cfg["paths"]["kaggle_raw"]).exists(), "retail_store_inventory.csv missing"


def test_supplier_table_is_coherent():
    cfg = load_config("suppliers")
    sup = cfg["suppliers"]
    assert len(sup) == 3
    for s in sup:
        assert s["lead_time_min"] <= s["lead_time_mode"] <= s["lead_time_max"]
        assert 0 < s["fill_rate_mean"] <= 1
        assert s["cost_multiplier"] >= 1.0
    # The trade-off must actually exist: the fastest supplier should not also
    # be the cheapest, otherwise supplier choice is a trivial decision and the
    # RL agent has nothing to learn.
    fastest = min(sup, key=lambda s: s["lead_time_mode"])
    cheapest = min(sup, key=lambda s: s["cost_multiplier"])
    assert fastest["id"] != cheapest["id"]
