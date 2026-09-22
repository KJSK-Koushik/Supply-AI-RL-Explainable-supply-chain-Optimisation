"""SupplyAI-RL decision-support dashboard (Phase 8).

    streamlit run app/dashboard.py

Framed as a shadow-mode deployment, which is the honest product story this
project supports: the tuned classical policy is the engine a buyer would
actually run today, the RL agent runs alongside as a challenger, and every
decision from either can be explained in business English. The dashboard never
executes anything -- it recommends and explains.

Everything shown comes from a real episode run on demand, or from the stored
result files. No number here is typed in.

Visual language: app/theme.py and .streamlit/config.toml.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import theme  # noqa: E402
from src.config import resolve  # noqa: E402
from src.eval.runner import EVAL_SEEDS, run_episode  # noqa: E402
from src.llm.explainer import build_facts, explain  # noqa: E402
from src.llm.scenario_gen import fallback_scenarios  # noqa: E402
from src.money import RATE, RUPEE, inr  # noqa: E402

st.set_page_config(page_title="SupplyAI-RL", page_icon="📦", layout="wide")
theme.inject_css()

LAKH = 1e5
SUPPLIERS = ["EconoSource<br>cheap, slow", "RapidTrade<br>fast, dear", "MidWay<br>unreliable"]

# Result files name runs after the experiment, not the reader. Display only:
# the result files themselves are never renamed.
PRETTY = {
    "forecast+SS": "forecast + safety stock",
    "forecast+SS (classical)": "forecast + safety stock",
    "EOQ+ROP": "EOQ + reorder point",
    "(s,S)": "(s, S) policy",
    "constant": "constant order (control)",
    "kaggle_full01 (masked)": "RL agent, tuned (2M)",
    "kaggle_full01": "RL agent, tuned (2M)",
    "kaggle_full00 (masked)": "RL agent, 2nd config (2M)",
    "kaggle_full00": "RL agent, 2nd config (2M)",
    "masked_1m (masked)": "RL agent, untuned (1M)",
    "masked_1m": "RL agent, untuned (1M)",
    "baseline_1m": "RL agent, no masking (1M)",
    "curriculum_1m": "RL agent, curriculum (1M)",
}


def pretty(name: str) -> str:
    return PRETTY.get(name, name)


# ------------------------------------------------------------------ loading


@st.cache_resource(show_spinner=False)
def load_policies() -> dict:
    """Every policy the project produced. Cached: models load once."""
    out = {}
    bl = resolve("results/baselines.json")
    if bl.exists():
        from src.agents.tune_baselines import make

        with bl.open(encoding="utf-8") as fh:
            data = json.load(fh)
        for kind, entry in data["policies"].items():
            if kind in ("random", "constant"):
                continue
            tag = "  (engine)" if kind == data["best_baseline"] else ""
            out[f"{kind}{tag}"] = ("classical", make(kind, entry["params"]))

    from src.agents.rl_policy import RLPolicy

    pretty = {
        "kaggle_full01": "RL agent, tuned (2M)  (challenger)",
        "kaggle_full00": "RL agent, 2nd config (2M)",
        "masked_1m": "RL agent, untuned (1M)",
        "curriculum_1m": "RL agent, curriculum (1M)",
    }
    for d in sorted(p for p in resolve("results/models").glob("*") if p.is_dir()):
        model = d / "best_model.zip"
        if model.exists() and d.name in pretty:
            out[pretty[d.name]] = ("agent", RLPolicy.load(str(model)))
    return out


@st.cache_data(show_spinner=False)
def run_traced(policy_name: str, seed: int, disrupted: bool) -> dict:
    kind, policy = load_policies()[policy_name]
    scenarios = fallback_scenarios() if disrupted else None
    summary = run_episode(policy, seed, scenarios=scenarios, collect_trace=True)
    # The trace holds numpy arrays; keep only what the page draws.
    trace = summary.pop("trace")
    summary["days"] = [t["day"] for t in trace]
    summary["stock_total"] = [float(np.sum(t["stock"])) for t in trace]
    summary["profit_daily"] = [float(t["profit"]) for t in trace]
    summary["demand"] = [t["demand"].tolist() for t in trace]
    summary["stock"] = [t["stock"].tolist() for t in trace]
    summary["short"] = [t["units_short"].tolist() for t in trace]
    summary["orders"] = [t["order_quantities"].tolist() for t in trace]
    summary["suppliers"] = [t["supplier_choice"].tolist() for t in trace]
    summary["active"] = [t["active_scenarios"] for t in trace]
    return summary


def load_json(path: str):
    p = resolve(path)
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as fh:
        return json.load(fh)


def crisis_shading(fig: go.Figure, days: list, active: list) -> None:
    crisis_days = [d for d, a in zip(days, active, strict=False) if a]
    if crisis_days:
        fig.add_vrect(
            x0=min(crisis_days),
            x1=max(crisis_days),
            fillcolor=theme.CORAL,
            opacity=0.10,
            line_width=0,
            annotation_text="disruption active",
            annotation_position="top left",
            annotation_font={"color": theme.MUTED, "size": 12},
        )


# --------------------------------------------------------------------- page

theme.hero(
    kicker="SupplyAI-RL · shadow mode",
    title="Order the right stock, and see why.",
    sub=(
        "The tuned classical policy is the engine. The RL agent runs alongside as a "
        "challenger. Every decision can be explained in plain English. Nothing here "
        "places an order."
    ),
    aside_title="How to read this page",
    aside=[
        (
            "Engine",
            "forecast + safety stock — the best classical policy, the one a buyer would run",
        ),
        ("Challenger", "MaskablePPO agent, tuned on Kaggle — scored, never in charge"),
        ("Explanation", "an LLM phrases the facts; every number is checked before you see it"),
        ("Money", f"rupees at 1 GBP = {RUPEE}{RATE:.0f}; the data is from a UK shop"),
    ],
)

policies = load_policies()
if not policies:
    st.error("No policies found. Run Phases 3-4 first.")
    st.stop()

with st.sidebar:
    st.header("Run an episode")
    names = list(policies)
    # Open on the engine -- the policy a buyer would actually run today -- so
    # the first screen shows the deployable system rather than the weakest rule.
    default = next((i for i, n in enumerate(names) if "(engine)" in n), 0)
    policy_name = st.selectbox("Policy", names, index=default)
    seed = st.selectbox("Scenario seed (held-out)", EVAL_SEEDS[:10], index=0)
    disrupted = st.toggle(
        "Inject a crisis", value=False, help="Seasonal rush + supplier outage + cost spike"
    )
    st.divider()
    st.caption(
        "Seeds 500-529 were never used for tuning or training. "
        "Same seed = same customers and delivery delays for every policy."
    )

tab_run, tab_explain, tab_compare, tab_robust = st.tabs(
    ["Episode", "Explain a decision", "Policy comparison", "Robustness"]
)

# ---------------------------------------------------------------- episode

with tab_run:
    with st.spinner("Simulating 180 days…"):
        ep = run_traced(policy_name, seed, disrupted)

    theme.card_row(
        [
            (
                "Profit, 180 days",
                inr(ep["total_profit"], symbol=RUPEE),
                policy_name.split("  (")[0],
            ),
            ("Fill rate", f"{ep['fill_rate']:.1%}", "share of demand served"),
            ("Ordering fees", inr(ep["ordering_cost"], symbol=RUPEE), "paid to suppliers"),
            (
                "Stockout cost",
                inr(ep["stockout_penalty"], symbol=RUPEE),
                "lost sales",
            ),
            (
                "Overflow lost",
                inr(ep["overflow_loss"], symbol=RUPEE),
                "the warehouse was full",
            ),
        ]
    )
    st.write("")

    days = ep["days"]
    left, right = st.columns([3, 2], gap="medium")

    with left:
        # Two measures of different scale get two charts on one axis each,
        # never a second y-axis: the alignment of two scales is arbitrary.
        fig = go.Figure(
            go.Scatter(
                x=days, y=ep["stock_total"], name="total stock", line=theme.line(theme.CLASSICAL)
            )
        )
        crisis_shading(fig, days, ep["active"])
        fig.update_layout(
            theme.plotly_layout(
                title={"text": "Warehouse stock, units held"},
                height=230,
                showlegend=False,
                xaxis={"title": {"text": "day"}},
            )
        )
        st.plotly_chart(fig, theme=None)

        cum = np.cumsum(ep["profit_daily"]) * RATE / LAKH
        fig = go.Figure(
            go.Scatter(x=days, y=cum, name="cumulative profit", line=theme.line(theme.AGENT))
        )
        crisis_shading(fig, days, ep["active"])
        fig.update_layout(
            theme.plotly_layout(
                title={"text": f"Cumulative profit, {RUPEE} lakh"},
                height=230,
                showlegend=False,
                xaxis={"title": {"text": "day"}},
            )
        )
        st.plotly_chart(fig, theme=None)

    with right:
        counts = np.zeros(3)
        for o, s_ in zip(ep["orders"], ep["suppliers"], strict=False):
            for q, sup in zip(o, s_, strict=False):
                if q > 0:
                    counts[int(sup)] += 1
        # One series over three nominal suppliers: one colour, values on the caps.
        fig2 = go.Figure(
            go.Bar(
                x=SUPPLIERS,
                y=counts,
                marker=theme.bar_marker(theme.CLASSICAL),
                width=0.3,
                text=[f"{c:.0f}" for c in counts],
                textposition="outside",
                textfont={"color": theme.INK},
                cliponaxis=False,
            )
        )
        fig2.update_layout(
            theme.plotly_layout(
                title={"text": f"Orders placed by supplier, {counts.sum():.0f} total"},
                height=492,
                xaxis={"tickangle": 0},
                yaxis={"title": {"text": "orders"}},
            )
        )
        st.plotly_chart(fig2, theme=None)

    st.subheader("One product, day by day")
    product = st.slider("Product index", 0, 9, 0)
    demand_p = [d[product] for d in ep["demand"]]
    stock_p = [s_[product] for s_ in ep["stock"]]
    short_p = [x[product] for x in ep["short"]]
    fig3 = go.Figure()
    fig3.add_trace(
        go.Bar(
            x=days,
            y=demand_p,
            name="daily demand",
            marker={"color": theme.CONTEXT},
            opacity=0.55,
        )
    )
    fig3.add_trace(
        go.Scatter(x=days, y=stock_p, name="stock on hand", line=theme.line(theme.CLASSICAL))
    )
    out_days = [d for d, s_ in zip(days, short_p, strict=False) if s_ > 0]
    if out_days:
        fig3.add_trace(
            go.Scatter(
                x=out_days,
                y=[0] * len(out_days),
                mode="markers",
                name="stockout day",
                marker=theme.marker(theme.STATUS["error"], "triangle-down", 10),
            )
        )
    crisis_shading(fig3, days, ep["active"])
    fig3.update_layout(
        theme.plotly_layout(
            height=340,
            bargap=0.35,
            xaxis={"title": {"text": "day"}},
            yaxis={"title": {"text": "units"}},
        )
    )
    st.plotly_chart(fig3, theme=None)

# ---------------------------------------------------------------- explain

with tab_explain:
    st.markdown(
        "The model is **never asked why** a decision was made. It receives facts the "
        "simulator computed and phrases them; every number in its output is checked "
        "against those facts, and anything invented is discarded in favour of a "
        "deterministic template."
    )
    ec1, ec2 = st.columns(2)
    ex_day = ec1.slider("Day", 15, 180, 45)
    ex_product = ec2.slider("Product", 0, 9, 0, key="ex_product")

    if st.button("Explain this decision", type="primary"):
        from src.env.supply_chain_env import SupplyChainEnv

        kind, policy = policies[policy_name]
        env = SupplyChainEnv(seed=seed, scenarios=fallback_scenarios() if disrupted else None)
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        if hasattr(policy, "reset"):
            policy.reset()
        info = None
        for _ in range(ex_day):
            act = policy.act(env, rng)
            _, _, term, trunc, info = env.step(act)
            if term or trunc:
                break
        facts = build_facts(env, info, ex_product)

        with st.spinner("Asking the model… free endpoints rate-limit, this can take a minute"):
            result = explain(facts)

        src = result["model"] if result["source"] == "llm" else "deterministic template"
        note = f"fell back because: {result['reason']}" if result["reason"] else ""
        theme.quote_card(result["text"], src, note)
        with st.expander("The facts the model was given"):
            st.json(facts)

# ---------------------------------------------------------------- compare

with tab_compare:
    cmp = load_json("results/comparison.json")
    if not cmp:
        st.info("Run `python -m src.eval.compare` first.")
    else:
        rows = []
        classical_names = set(load_json("results/baselines.json")["policies"])
        for name, v in cmp["summaries"].items():
            base = name.split(" (")[0]
            if base == "random" or "_s0" in name or "probe" in name:
                continue
            rows.append(
                {
                    "policy": pretty(name),
                    "type": "classical" if base in classical_names else "RL agent",
                    "profit": v["total_profit"],
                    "± std": v["total_profit_std"],
                    "fill rate": v["fill_rate"],
                    "ordering cost": v["ordering_cost"],
                    "stockout cost": v["stockout_penalty"],
                }
            )
        df = pd.DataFrame(rows).sort_values("profit", ascending=False).reset_index(drop=True)
        best_c = df[df.type == "classical"].iloc[0]
        best_a = df[df.type == "RL agent"].iloc[0]
        v = cmp.get("verdict") or {}
        cards = [
            ("Best classical", inr(best_c.profit, symbol=RUPEE), best_c.policy, "teal"),
            (
                "Best RL agent",
                inr(best_a.profit, symbol=RUPEE),
                f"{best_a.profit / best_c.profit:.1%} of the best classical policy",
                "pink",
            ),
        ]
        if v:
            cards.append(
                (
                    "Paired t-test, 30 seeds",
                    f"t = {v['t']:+.2f}",
                    "the gap is real" if v["significant"] else "not significant",
                    "cream",
                )
            )
        theme.card_row(cards)
        st.write("")

        # Classical and agent are two series, so both get a legend entry; the
        # categories are disjoint, so overlay keeps every bar full width.
        fig = go.Figure()
        for kind, color in (("classical", theme.CLASSICAL), ("RL agent", theme.AGENT)):
            part = df[df.type == kind]
            fig.add_trace(
                go.Bar(
                    name=kind,
                    x=part.profit * RATE / LAKH,
                    y=part.policy,
                    orientation="h",
                    marker=theme.bar_marker(color),
                    width=0.55,
                    error_x={
                        "type": "data",
                        "array": part["± std"] * RATE / LAKH,
                        "color": theme.MUTED,
                        "thickness": 1,
                        "width": 3,
                    },
                )
            )
        # Value labels sit past the whisker, not on top of it.
        for _, r in df.iterrows():
            fig.add_annotation(
                x=(r.profit + r["± std"]) * RATE / LAKH,
                y=r.policy,
                text=inr(r.profit, symbol=RUPEE),
                showarrow=False,
                xanchor="left",
                xshift=8,
                font={"color": theme.INK, "size": 12},
            )
        fig.update_layout(
            theme.plotly_layout(
                barmode="overlay",
                height=460,
                title={"text": "30 held-out seeds, identical customers for every policy"},
                xaxis={"title": {"text": f"profit per 180-day episode, {RUPEE} lakh"}},
                yaxis={
                    "categoryorder": "array",
                    "categoryarray": list(df.policy[::-1]),
                    "gridcolor": "rgba(0,0,0,0)",
                },
                margin={"r": 96},
            )
        )
        st.plotly_chart(fig, theme=None)

        shown = df.rename(
            columns={
                "profit": f"profit ({RUPEE} lakh)",
                "± std": f"± std ({RUPEE} lakh)",
                "ordering cost": f"ordering cost ({RUPEE} lakh)",
                "stockout cost": f"stockout cost ({RUPEE} lakh)",
            }
        )
        money = [c for c in shown.columns if "lakh" in c]
        shown = shown.assign(**{c: shown[c] * RATE / LAKH for c in money})
        st.dataframe(
            shown.style.format({**dict.fromkeys(money, "{:.2f}"), "fill rate": "{:.1%}"}),
            hide_index=True,
        )

# --------------------------------------------------------------- robustness

with tab_robust:
    rob = load_json("results/robustness.json")
    if not rob:
        st.info("Run `python -m src.eval.robustness` first.")
    else:
        st.markdown(
            f"Each policy scored on **{len(rob['calm_seeds'])} calm seeds** and "
            f"**{rob['n_holdout_sets']} held-out LLM-generated crises** it never trained on."
        )
        rdf = pd.DataFrame(
            [
                {
                    "policy": pretty(n),
                    "kind": "classical" if "classical" in n else "RL agent",
                    "calm": r["calm"],
                    "disrupted": r["disrupted"],
                    "drop": r["drop_pct"],
                }
                for n, r in rob["rows"].items()
            ]
        ).sort_values("disrupted", ascending=False)
        best_c = rdf[rdf.kind == "classical"].iloc[0]
        best_a = rdf[rdf.kind == "RL agent"].iloc[0]
        v = rob.get("verdict") or {}
        cards = [
            (
                "Best classical under crisis",
                inr(best_c.disrupted, symbol=RUPEE),
                f"{best_c.policy}, {best_c['drop']:+.1f}% from calm",
                "teal",
            ),
            (
                "Best RL agent under crisis",
                inr(best_a.disrupted, symbol=RUPEE),
                f"{best_a.policy}, {best_a['drop']:+.1f}% from calm",
                "pink",
            ),
        ]
        if v:
            cards.append(
                (
                    "Curriculum vs control",
                    f"t = {v['t']:+.2f}",
                    "significant" if v["significant"] else "not significant on paired crises",
                    "lavender",
                )
            )
        theme.card_row(cards)
        st.write("")

        # The disrupted bar is the story; calm is the context it is measured against.
        ticks = [p.replace(", ", ",<br>") for p in rdf.policy]  # two short lines beat a tilt
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="calm",
                x=ticks,
                y=rdf.calm * RATE / LAKH,
                marker=theme.bar_marker(theme.DEEMPHASIS),
            )
        )
        fig.add_trace(
            go.Bar(
                name="disrupted",
                x=ticks,
                y=rdf.disrupted * RATE / LAKH,
                marker=theme.bar_marker(theme.AGENT),
                text=[f"{d:+.0f}%" for d in rdf["drop"]],
                textposition="outside",
                textfont={"color": theme.INK, "size": 12},
                cliponaxis=False,
            )
        )
        fig.update_layout(
            theme.plotly_layout(
                barmode="group",
                bargap=0.55,
                height=400,
                title={"text": "Calm vs disrupted, with the drop on each disrupted bar"},
                xaxis={"tickangle": 0},
                yaxis={"title": {"text": f"profit, {RUPEE} lakh"}},
            )
        )
        st.plotly_chart(fig, theme=None)
        st.dataframe(
            rdf.assign(calm=rdf.calm * RATE / LAKH, disrupted=rdf.disrupted * RATE / LAKH)
            .rename(
                columns={"calm": f"calm ({RUPEE} lakh)", "disrupted": f"disrupted ({RUPEE} lakh)"}
            )
            .style.format(
                {
                    f"calm ({RUPEE} lakh)": "{:.2f}",
                    f"disrupted ({RUPEE} lakh)": "{:.2f}",
                    "drop": "{:+.1f}%",
                }
            ),
            hide_index=True,
        )
        if v:
            st.caption(
                f"Curriculum vs control, paired on identical (crisis, seed) pairs: "
                f"t = {v['t']:+.2f}, {'significant' if v['significant'] else 'not significant'}. "
                f"A smaller percentage drop on a lower base is not robustness."
            )

theme.footer(
    [
        (
            "SupplyAI-RL",
            ["22AIE450 Reinforcement Learning", "Explainable supply chain optimization"],
        ),
        (
            "Data",
            [
                "UCI Online Retail II, 1.07M real sales records",
                "10 products, 3 suppliers, 180-day runs",
            ],
        ),
        (
            "Money",
            [f"Shown in rupees at 1 GBP = {RUPEE}{RATE:.0f}", "Set in configs/display.yaml"],
        ),
    ]
)
