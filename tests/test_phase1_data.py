"""Phase 1 checks: the calibrated outputs must be sane before the simulator
is built on top of them."""

from __future__ import annotations

import json

import pytest

from src.config import load_config, resolve
from src.data import loader

cfg = load_config("data")
stats_path = resolve(cfg["paths"]["demand_stats"])
panel_path = resolve(cfg["paths"]["daily_panel"])

pytestmark = pytest.mark.skipif(
    not stats_path.exists(), reason="run `python -m src.data.run_pipeline` first"
)


@pytest.fixture(scope="module")
def stats():
    with stats_path.open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def panel():
    # Must go through read_panel: a direct read_csv infers numeric StockCodes
    # as ints in some chunks and strings in others, splitting a single product
    # into two groups and making the panel look ragged when it is not.
    return loader.read_panel(cfg)


def test_panel_is_rectangular(panel):
    counts = panel.groupby("StockCode").size()
    assert counts.nunique() == 1, "every product must span the same calendar"


def test_no_closed_days_in_panel(panel):
    closed = set(cfg["calendar"]["closed_weekdays"])
    assert not set(panel["dow"].unique()) & closed


def test_demand_non_negative(panel):
    assert (panel["demand"] >= 0).all()


def test_prices_all_filled(panel):
    assert panel["price"].notna().all()


def test_selected_product_count(stats):
    assert len(stats["selected_products"]) == cfg["product_selection"]["n_products"]
    assert len(set(stats["selected_products"])) == len(stats["selected_products"])


def test_regimes_are_spread(stats):
    regimes = set(stats["regime_of"].values())
    assert len(regimes) >= 2, "products must span multiple demand regimes"


def test_product_stats_usable(stats):
    for code in stats["selected_products"]:
        p = stats["products"][code]
        assert p["mean_daily_demand"] >= cfg["product_selection"]["min_mean_daily_demand"]
        assert p["std_daily_demand"] > 0
        assert p["zero_day_fraction"] <= cfg["product_selection"]["max_zero_day_fraction"]
        assert p["price"] > 0


def test_seasonality_factors_are_centred(stats):
    for factors in (stats["day_of_week_factor"], stats["month_factor"]):
        vals = list(factors.values())
        assert 0.7 < sum(vals) / len(vals) < 1.3, "multiplicative factors should average ~1"


def test_holdout_exists(stats):
    for code in stats["selected_products"]:
        assert stats["products"][code]["holdout_days"] > 30
