"""Phase 8: the dashboard imports and its data paths resolve.

Streamlit pages cannot be unit-tested by importing them -- the module body IS
the page and calls st.* at import. What can be checked cheaply is that the
helpers it depends on produce what the page expects, so a rename upstream
fails here rather than as a blank screen during a demo.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.config import load_config, resolve

pytestmark = pytest.mark.local_env

stats_path = resolve(load_config("data")["paths"]["demand_stats"])
if not stats_path.exists():
    pytestmark = [pytestmark, pytest.mark.skip(reason="run the data pipeline first")]


def test_dashboard_parses_and_imports_only_known_modules():
    src = Path("app/dashboard.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    for mod in imported:
        if mod.startswith("src."):
            __import__(mod)


def test_episode_summary_has_every_key_the_metrics_row_reads():
    """The five metric tiles read these by name. A rename in costs.py would
    otherwise surface as a KeyError on the first screen."""
    import json

    from src.agents.tune_baselines import make
    from src.eval.runner import run_episode

    with resolve("results/baselines.json").open(encoding="utf-8") as fh:
        data = json.load(fh)
    best = data["best_baseline"]
    summary = run_episode(make(best, data["policies"][best]["params"]), 500, collect_trace=True)
    for key in ("total_profit", "fill_rate", "ordering_cost", "stockout_penalty", "overflow_loss"):
        assert key in summary, key
    first = summary["trace"][0]
    for key in (
        "stock",
        "demand",
        "units_short",
        "order_quantities",
        "supplier_choice",
        "active_scenarios",
        "profit",
        "day",
    ):
        assert key in first, key
