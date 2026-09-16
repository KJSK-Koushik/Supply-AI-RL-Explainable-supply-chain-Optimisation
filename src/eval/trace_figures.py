"""Figures that show the simulator actually running.

    python -m src.eval.trace_figures

The other figures in this project describe the data or summarise results. None
show a single episode unfolding, which is what people ask to see first: what
does a day look like, and how do two policies behave differently inside the
same 180 days?

One seed, so both policies meet identical customers and identical delivery
delays. Any divergence between the lines is a difference in behaviour, not luck.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import resolve
from src.eval.runner import run_episode
from src.money import RATE, RUPEE, inr, lakh_axis

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})

CLASSICAL = "#2c7a4b"
AGENT = "#c0392b"
MUTED = "#7f8c8d"
TRACE_SEED = 500  # the first held-out reporting seed

SUPPLIERS = ["EconoSource\n(cheap, slow)", "RapidTrade\n(fast, dear)", "MidWay\n(unreliable)"]


def _env_facts() -> tuple[int, float, str]:
    """Warm-up length, warehouse capacity, and a label for the first product."""
    from src.env.supply_chain_env import SupplyChainEnv

    env = SupplyChainEnv(seed=0)
    label = f"{env.codes[0]} {env.descriptions[0].title()[:26].strip()}"
    return env.warmup_days, float(env.capacity_total), label


def _load_policies() -> dict:
    """The best tuned baseline, and the best trained agent if one exists."""
    out = {}
    bl = resolve("results/baselines.json")
    if bl.exists():
        from src.agents.tune_baselines import make

        with bl.open(encoding="utf-8") as fh:
            data = json.load(fh)
        best = data["best_baseline"]
        out[best] = (make(best, data["policies"][best]["params"]), CLASSICAL)

    best_model = None
    for d in sorted(p for p in resolve("results/models").glob("*") if p.is_dir()):
        summary = d / "summary.json"
        if not (d / "best_model.zip").exists() or not summary.exists():
            continue
        with summary.open(encoding="utf-8") as fh:
            profit = json.load(fh).get("best_eval_profit")
        if profit is not None and np.isfinite(profit):
            if best_model is None or profit > best_model[1]:
                best_model = (d, profit)
    if best_model:
        from src.agents.rl_policy import RLPolicy

        out[best_model[0].name] = (RLPolicy.load(str(best_model[0] / "best_model.zip")), AGENT)
    return out


def _totals(trace, key) -> np.ndarray:
    return np.asarray([np.sum(t[key]) for t in trace], dtype=float)


def _panel_product(a, trace, warmup, title):
    """One product, day by day: what the stock line is reacting to."""
    days = np.arange(1, len(trace) + 1)
    demand = np.asarray([t["demand"][0] for t in trace], dtype=float)
    stock = np.asarray([t["stock"][0] for t in trace], dtype=float)
    short = np.asarray([t["units_short"][0] for t in trace], dtype=float)

    # Demand gets its own axis. Peak stock is an order of magnitude larger, so
    # on a shared scale the demand bars flatten into an unreadable strip along
    # the bottom -- and demand is the thing the stock line is responding to.
    a2 = a.twinx()
    a2.bar(days, demand, color="#b8c9d6", width=1.0)
    a2.set_ylabel("daily demand (units)", color="#5b7086", fontsize=8)
    a2.tick_params(axis="y", labelcolor="#5b7086", labelsize=8)
    a2.set_ylim(0, demand.max() * 3.2)
    a2.grid(False)

    a.set_zorder(a2.get_zorder() + 1)
    a.patch.set_visible(False)
    a.plot(days, stock, color="#1f4e79", lw=1.5)
    a.axvspan(0, warmup, color=MUTED, alpha=0.12)
    a.text(
        warmup / 2,
        stock.max(),
        "warm-up\n(not scored)",
        ha="center",
        va="top",
        fontsize=7,
        color="#5a5a66",
    )
    handles = [
        plt.Line2D([], [], color="#1f4e79", lw=1.5, label="stock on hand"),
        plt.Rectangle((0, 0), 1, 1, color="#b8c9d6", label="daily demand"),
    ]
    if (short > 0).any():
        a.plot(days[short > 0], np.zeros(int((short > 0).sum())), "v", color=AGENT, ms=5)
        handles.append(plt.Line2D([], [], marker="v", ls="", color=AGENT, label="stockout day"))
    a.set_title(title, fontweight="bold", fontsize=10)
    a.set_xlabel("day")
    a.set_ylabel("stock on hand (units)", color="#1f4e79")
    a.legend(handles=handles, fontsize=7.5, loc="upper left")


def run() -> list[str]:
    outdir = resolve("results/figures")
    outdir.mkdir(parents=True, exist_ok=True)

    policies = _load_policies()
    if not policies:
        print("      (skipping simulator trace: no tuned baseline or model found)")
        return []

    warmup, capacity, code = _env_facts()
    traces = {
        name: (run_episode(pol, TRACE_SEED, collect_trace=True), colour)
        for name, (pol, colour) in policies.items()
    }

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 7.2))
    name0 = next(iter(traces))
    trace0 = traces[name0][0]["trace"]
    days = np.arange(1, len(trace0) + 1)

    _panel_product(ax[0, 0], trace0, warmup, f"A. One product day by day  ({name0}, {code})")

    # --- B: how full the shared warehouse gets ---------------------------
    b = ax[0, 1]
    for name, (summary, colour) in traces.items():
        b.plot(days, _totals(summary["trace"], "stock"), color=colour, lw=1.4, label=name)
    b.axhline(capacity, color=MUTED, ls="--", lw=1.2)
    b.text(
        days[-1],
        capacity,
        " warehouse capacity ",
        va="bottom",
        ha="right",
        fontsize=7.5,
        color=MUTED,
        fontweight="bold",
    )
    b.set_ylim(0, capacity * 1.12)
    b.set_title("B. Total stock against the shared warehouse limit", fontweight="bold", fontsize=10)
    b.set_xlabel("day")
    b.set_ylabel("units held")
    b.legend(fontsize=7.5, loc="lower right")

    # --- C: profit accumulating ------------------------------------------
    c = ax[1, 0]
    for name, (summary, colour) in traces.items():
        # Accumulated from the end of warm-up, not from day 1. Those first days
        # exist only to clear an arbitrary opening inventory and are excluded
        # from every reported metric; including them made this panel's endpoint
        # disagree with the numbers in the tables by several thousand.
        prof = np.cumsum(_totals(summary["trace"], "profit")[warmup:]) * RATE
        c.plot(
            days[warmup:],
            prof,
            color=colour,
            lw=1.6,
            label=f"{name}: {inr(prof[-1] / RATE, symbol=RUPEE)}",
        )
    c.axhline(0, color="k", lw=0.8)
    c.set_title("C. Profit accumulating after warm-up", fontweight="bold", fontsize=10)
    c.set_xlabel("day")
    c.set_ylabel(f"cumulative profit ({RUPEE})")
    lakh_axis(c.yaxis)
    c.legend(fontsize=7.5, loc="upper left")

    # --- D: who each policy buys from -------------------------------------
    d = ax[1, 1]
    idx = np.arange(len(SUPPLIERS))
    width = 0.36
    for k, (name, (summary, colour)) in enumerate(traces.items()):
        counts = np.zeros(len(SUPPLIERS))
        for t in summary["trace"]:
            for s in t["supplier_choice"][t["order_quantities"] > 0]:
                counts[int(s)] += 1
        offset = (k - (len(traces) - 1) / 2) * width
        bars = d.bar(idx + offset, counts, width, color=colour, label=f"{name}: {counts.sum():.0f}")
        d.bar_label(bars, fmt="%d", fontsize=7, padding=2)
    d.set_xticks(idx)
    d.set_xticklabels(SUPPLIERS, fontsize=7.5)
    d.set_title("D. Orders placed with each supplier", fontweight="bold", fontsize=10)
    d.set_ylabel("orders over the episode")
    d.legend(fontsize=7.5, title="total orders placed", title_fontsize=7.5)

    fig.suptitle(
        f"Inside one episode (seed {TRACE_SEED}) - every policy meets identical "
        "customers and delivery delays",
        fontweight="bold",
    )
    fig.tight_layout()
    out = outdir / "07_simulator_trace.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return [out.name]


if __name__ == "__main__":
    for name in run():
        print("wrote", name)
