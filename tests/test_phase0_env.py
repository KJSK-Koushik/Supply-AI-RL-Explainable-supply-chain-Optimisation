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


@pytest.mark.local_env
def test_dashboard_imports():
    """Phase 8 packages. Only the laptop serves the dashboard, so remote
    training machines deselect this rather than install ~80 MB they never use.
    Kept strict where it does run: a half-installed venv should fail loudly."""
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
    Run `python scripts/fetch_data.py` to populate it.

    The two files skip independently. UCI downloads from a public URL and is
    what the simulator is calibrated from, so its absence stops everything
    downstream. The Kaggle CSV needs account credentials fetch_data.py cannot
    supply unattended, and it feeds only the dataset-screening section of the
    report -- a decision already made and written up. Absent credentials, its
    absence is the expected state, not a fault.
    """
    cfg = load_config("data")
    uci = resolve(cfg["paths"]["uci_raw"])
    if not uci.exists():
        pytest.skip("UCI raw data not present -- run scripts/fetch_data.py")
    kaggle = resolve(cfg["paths"]["kaggle_raw"])
    if not kaggle.exists():
        pytest.skip(f"{kaggle.name} not present (needs Kaggle credentials)")
    assert uci.stat().st_size > 1_000_000, "UCI file present but implausibly small"


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
