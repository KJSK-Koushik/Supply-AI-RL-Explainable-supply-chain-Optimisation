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


# ---- the visual theme ---------------------------------------------------------


def test_card_colours_cycle_through_every_surface_without_repeating():
    """The design system cycles pink -> teal -> lavender -> peach -> ochre ->
    cream and never shows the same colour twice in a row."""
    from app import theme

    seen, previous = [], None
    for _ in range(12):
        previous = theme.next_variant(previous)
        seen.append(previous)
    assert tuple(seen[:6]) == theme.CARD_ORDER
    assert all(a != b for a, b in zip(seen, seen[1:], strict=False))


def test_card_text_flips_to_white_only_on_the_dark_surfaces():
    from app import theme

    assert theme.CARD_TEXT["pink"] == theme.CARD_TEXT["teal"] == theme.ON_DARK
    for light in ("lavender", "peach", "ochre", "cream"):
        assert theme.CARD_TEXT[light] == theme.INK


def test_card_html_escapes_what_it_is_given():
    """Card text comes from result files and policy names; none of it may
    reach the page as markup."""
    from app import theme

    html = theme.card_html("<b>label</b>", "₹1 & 2", "<script>x</script>", "teal")
    assert "<b>" not in html and "<script>" not in html
    assert "&lt;b&gt;label&lt;/b&gt;" in html and "&amp;" in html
    assert theme.CARD_FILL["teal"] in html


def test_unknown_card_variant_is_refused():
    import pytest

    from app import theme

    with pytest.raises(ValueError):
        theme.card_html("a", "b", variant="navy")


def test_plotly_layout_keeps_the_shared_axis_style_under_overrides():
    from app import theme

    layout = theme.plotly_layout(height=200, xaxis={"title": {"text": "day"}})
    assert layout["height"] == 200
    assert layout["xaxis"]["title"]["text"] == "day"
    assert layout["xaxis"]["gridcolor"] == theme.SURFACE_STRONG  # not lost to the override
    assert layout["yaxis"]["automargin"] is True
    assert list(layout["colorway"]) == list(theme.CHART_SERIES)
