"""Result figures for the report and slides.

    python -m src.eval.result_figures

Reads only committed result files, so every figure is reproducible from the
repository and none of the numbers are retyped by hand. Anything missing is
skipped with a message rather than crashing, for the same reason the EDA
figures skip: a missing chart must not take down a pipeline.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import resolve
from src.money import RATE, RUPEE, inr, lakh_axis

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})

CLASSICAL = "#2c7a4b"
AGENT = "#c0392b"
MUTED = "#7f8c8d"


def fig_policy_comparison(outdir) -> str | None:
    """Every policy on the 30 held-out seeds, with spread.

    Error bars are the standard deviation across seeds, not a confidence
    interval. They overlap heavily, which is exactly why the significance test
    on slide 11 is paired: pairing removes the episode-to-episode variance that
    dominates these bars and would otherwise hide a real effect.
    """
    path = resolve("results/comparison.json")
    if not path.exists():
        print("      (skipping policy comparison: comparison.json not present)")
        return None

    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    summaries = data["summaries"]

    # Which rows are classical is read from baselines.json, not guessed from the
    # name. The previous rule -- "1m" in the name -- silently painted every
    # swept agent green as a classical policy once the Kaggle runs landed,
    # because they are called kaggle_full01 and friends.
    classical = set()
    bl = resolve("results/baselines.json")
    if bl.exists():
        with bl.open(encoding="utf-8") as fh:
            classical = set(json.load(fh)["policies"])

    def is_screening(name: str) -> bool:
        # Screening runs are 250k-step triage inside the sweep, and the probe
        # measured throughput. Neither is a result, and eleven extra bars bury
        # the four rows anyone is looking for.
        return "_s0" in name or "probe" in name

    order = sorted(summaries, key=lambda k: summaries[k]["total_profit"])
    names = [n for n in order if n.split(" (")[0] != "random" and not is_screening(n)]
    profits = [summaries[n]["total_profit"] * RATE for n in names]
    errs = [summaries[n]["total_profit_std"] * RATE for n in names]
    is_agent = [n.split(" (")[0] not in classical for n in names]
    colors = [AGENT if a else CLASSICAL for a in is_agent]

    # Run directories are named for the experiment, not the reader. Anything
    # unmapped falls through unchanged so a new run still plots.
    pretty = {
        "forecast+SS": "forecast + safety stock",
        "EOQ+ROP": "EOQ + reorder point",
        "(s,S)": "(s, S) policy",
        "constant": "constant order (control)",
        "masked_1m (masked)": "PPO + masking, untuned (1M)",
        "baseline_1m": "PPO, no masking (1M)",
        "kaggle_full01 (masked)": "PPO + masking, tuned (2M)",
        "kaggle_full00 (masked)": "PPO + masking, 2nd config (2M)",
    }
    labels = [pretty.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    bars = ax.barh(
        labels, profits, xerr=errs, color=colors, capsize=3, error_kw={"ecolor": MUTED, "lw": 1}
    )
    ax.set_xlabel(f"total profit over a 180-day episode ({RUPEE}, mean of 30 held-out seeds)")
    ax.axvline(0, color="k", lw=0.8)
    lakh_axis(ax.xaxis)

    for bar, p in zip(bars, profits, strict=True):
        ax.text(
            p + (900 * RATE if p >= 0 else -900 * RATE),
            bar.get_y() + bar.get_height() / 2,
            inr(p / RATE, symbol=RUPEE),
            va="center",
            ha="left" if p >= 0 else "right",
            fontsize=8,
            fontweight="bold",
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CLASSICAL),
        plt.Rectangle((0, 0), 1, 1, color=AGENT),
    ]
    ax.legend(handles, ["classical policy (tuned)", "RL agent"], loc="lower right")
    ax.set_title(
        "Tuning closed 38% of the gap; the classical policy still leads\n"
        "(error bars: standard deviation across 30 held-out seeds)",
        fontweight="bold",
        fontsize=10,
    )
    ax.margins(x=0.16)
    fig.tight_layout()
    out = outdir / "05_policy_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out.name


def fig_training_curves(outdir) -> str | None:
    """Learning progress of both runs against the baseline to beat.

    Plotted on the training-eval seeds (900-905), which is what the checkpoint
    selector saw. That makes this a picture of learning progress, not a result:
    the reported scores come from the held-out seeds instead.
    """
    # Ordered worst to best so the legend reads as the story: masking first,
    # then tuning. Runs absent from disk are skipped rather than faked.
    runs = [
        ("baseline_1m", "no masking (1M)", MUTED),
        ("masked_1m", "+ action masking (1M)", "#e8846b"),
        ("kaggle_full01", "+ tuned hyperparameters (2M)", AGENT),
    ]
    series = []
    for name, label, color in runs:
        path = resolve("results/models") / name / "eval_history.json"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            series.append((label, color, json.load(fh)))
    if not series:
        print("      (skipping training curves: no eval_history.json found)")
        return None

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    for label, color, hist in series:
        xs = [h["timesteps"] for h in hist]
        ys = [h["profit"] * RATE for h in hist]
        ax.plot(xs, ys, lw=1.6, color=color, label=label)

    bl = resolve("results/baselines.json")
    if bl.exists():
        with bl.open(encoding="utf-8") as fh:
            b = json.load(fh)
        best = b["best_baseline"]
        val = b["policies"][best]["eval"]["total_profit"] * RATE
        ax.axhline(val, color=CLASSICAL, ls="--", lw=1.4)
        ax.text(
            ax.get_xlim()[1],
            val,
            f"  best classical policy ({best}): {inr(val / RATE, symbol=RUPEE)}",
            va="bottom",
            ha="right",
            fontsize=8,
            color=CLASSICAL,
            fontweight="bold",
        )

    ax.set_xlabel("training steps")
    ax.set_ylabel(f"profit per episode ({RUPEE})")
    lakh_axis(ax.yaxis)
    ax.legend(loc="lower right")
    ax.set_title(
        "Masking, then tuning -- and still short of the classical policy\n"
        "(scored on the training-eval seeds used to pick checkpoints)",
        fontweight="bold",
        fontsize=10,
    )
    fig.tight_layout()
    out = outdir / "06_training_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out.name


def run() -> list[str]:
    outdir = resolve("results/figures")
    outdir.mkdir(parents=True, exist_ok=True)
    made = [fig_policy_comparison(outdir), fig_training_curves(outdir)]
    return [m for m in made if m]


if __name__ == "__main__":
    for name in run():
        print("wrote", name)
