"""SupplyAI-RL decision-support dashboard (Phase 8).

    streamlit run app/dashboard.py

Framed as a shadow-mode deployment, which is the honest product story this
project supports: the tuned classical policy is the engine a buyer would
actually run today, the RL agent runs alongside as a challenger, and every
decision from either can be explained in business English. The dashboard never
executes anything -- it recommends and explains.

Everything shown comes from a real episode run on demand, or from the stored
result files. No number here is typed in.
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

from src.config import resolve  # noqa: E402
from src.eval.runner import EVAL_SEEDS, run_episode  # noqa: E402
from src.llm.explainer import build_facts, explain  # noqa: E402
from src.llm.scenario_gen import fallback_scenarios  # noqa: E402

st.set_page_config(page_title="SupplyAI-RL", page_icon="📦", layout="wide")

CLASSICAL = "#2c7a4b"
AGENT = "#c0392b"
MUTED = "#7f8c8d"


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


# --------------------------------------------------------------------- page

st.title("SupplyAI-RL — inventory decision support")
st.caption(
    "Shadow-mode deployment: the tuned classical policy is the engine, the RL "
    "agent runs alongside as a challenger, and every decision can be explained. "
    "Nothing here executes an order."
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

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Profit (180 days)", f"£{ep['total_profit']:,.0f}")
    c2.metric("Fill rate", f"{ep['fill_rate']:.1%}")
    c3.metric("Ordering fees", f"£{ep['ordering_cost']:,.0f}")
    c4.metric("Stockout cost", f"£{ep['stockout_penalty']:,.0f}")
    c5.metric("Overflow lost", f"£{ep['overflow_loss']:,.0f}")

    days = ep["days"]
    left, right = st.columns([3, 2])

    with left:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=days,
                y=ep["stock_total"],
                name="total stock",
                line={"color": "#1f4e79", "width": 2},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=days,
                y=np.cumsum(ep["profit_daily"]),
                name="cumulative profit",
                yaxis="y2",
                line={"color": CLASSICAL, "width": 2},
            )
        )
        crisis_days = [d for d, a in zip(days, ep["active"], strict=False) if a]
        if crisis_days:
            fig.add_vrect(
                x0=min(crisis_days),
                x1=max(crisis_days),
                fillcolor=AGENT,
                opacity=0.08,
                line_width=0,
                annotation_text="disruption active",
                annotation_position="top left",
            )
        fig.update_layout(
            title="Warehouse stock and cumulative profit",
            xaxis_title="day",
            yaxis_title="units held",
            yaxis2={"title": "profit (£)", "overlaying": "y", "side": "right"},
            height=380,
            margin={"l": 10, "r": 10, "t": 40, "b": 10},
            legend={"orientation": "h"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        counts = np.zeros(3)
        for o, s_ in zip(ep["orders"], ep["suppliers"], strict=False):
            for q, sup in zip(o, s_, strict=False):
                if q > 0:
                    counts[int(sup)] += 1
        fig2 = go.Figure(
            go.Bar(
                x=["EconoSource\ncheap, slow", "RapidTrade\nfast, dear", "MidWay\nunreliable"],
                y=counts,
                marker_color=["#2c7a4b", "#1f4e79", "#c0392b"],
                text=[f"{c:.0f}" for c in counts],
                textposition="outside",
            )
        )
        fig2.update_layout(
            title=f"Orders placed by supplier ({counts.sum():.0f} total)",
            height=380,
            margin={"l": 10, "r": 10, "t": 40, "b": 10},
            yaxis_title="orders",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("One product, day by day")
    product = st.slider("Product index", 0, 9, 0)
    demand_p = [d[product] for d in ep["demand"]]
    stock_p = [s_[product] for s_ in ep["stock"]]
    short_p = [x[product] for x in ep["short"]]
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=days, y=demand_p, name="daily demand", marker_color="#b8c9d6"))
    fig3.add_trace(
        go.Scatter(x=days, y=stock_p, name="stock on hand", line={"color": "#1f4e79", "width": 2})
    )
    out_days = [d for d, s_ in zip(days, short_p, strict=False) if s_ > 0]
    if out_days:
        fig3.add_trace(
            go.Scatter(
                x=out_days,
                y=[0] * len(out_days),
                mode="markers",
                name="stockout day",
                marker={"color": AGENT, "symbol": "triangle-down", "size": 9},
            )
        )
    fig3.update_layout(
        height=320,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        legend={"orientation": "h"},
        xaxis_title="day",
        yaxis_title="units",
    )
    st.plotly_chart(fig3, use_container_width=True)

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

        st.success(result["text"])
        src = "LLM: " + result["model"] if result["source"] == "llm" else "deterministic template"
        note = f"  ·  fell back because: {result['reason']}" if result["reason"] else ""
        st.caption(f"source: {src}{note}")
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
                    "policy": name,
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
        k1, k2, k3 = st.columns(3)
        k1.metric("Best classical", f"£{best_c.profit:,.0f}", best_c.policy)
        k2.metric(
            "Best RL agent",
            f"£{best_a.profit:,.0f}",
            f"{best_a.profit / best_c.profit:.1%} of classical",
        )
        v = cmp.get("verdict") or {}
        if v:
            k3.metric(
                "Paired test",
                f"t = {v['t']:+.2f}",
                "significant" if v["significant"] else "not significant",
                delta_color="off",
            )

        fig = go.Figure(
            go.Bar(
                x=df.profit,
                y=df.policy,
                orientation="h",
                marker_color=[CLASSICAL if t == "classical" else AGENT for t in df.type],
                error_x={"type": "data", "array": df["± std"], "color": MUTED},
                text=[f"{p:,.0f}" for p in df.profit],
                textposition="outside",
            )
        )
        fig.update_layout(
            height=420,
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
            yaxis={"autorange": "reversed"},
            xaxis_title="profit per 180-day episode (£)",
            title="30 held-out seeds, identical customers for every policy",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            df.style.format(
                {
                    "profit": "{:,.0f}",
                    "± std": "{:,.0f}",
                    "fill rate": "{:.1%}",
                    "ordering cost": "{:,.0f}",
                    "stockout cost": "{:,.0f}",
                }
            ),
            use_container_width=True,
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
                {"policy": n, "calm": r["calm"], "disrupted": r["disrupted"], "drop": r["drop_pct"]}
                for n, r in rob["rows"].items()
            ]
        ).sort_values("disrupted", ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="calm", x=rdf.policy, y=rdf.calm, marker_color=MUTED))
        fig.add_trace(go.Bar(name="disrupted", x=rdf.policy, y=rdf.disrupted, marker_color=AGENT))
        fig.update_layout(
            barmode="group",
            height=380,
            margin={"l": 10, "r": 10, "t": 30, "b": 10},
            yaxis_title="profit (£)",
            title="Calm vs disrupted",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            rdf.style.format({"calm": "{:,.0f}", "disrupted": "{:,.0f}", "drop": "{:+.1f}%"}),
            use_container_width=True,
            hide_index=True,
        )
        v = rob.get("verdict") or {}
        if v:
            st.caption(
                f"Curriculum vs control, paired on identical (crisis, seed) pairs: "
                f"t = {v['t']:+.2f}, {'significant' if v['significant'] else 'not significant'}. "
                f"A smaller percentage drop on a lower base is not robustness."
            )
