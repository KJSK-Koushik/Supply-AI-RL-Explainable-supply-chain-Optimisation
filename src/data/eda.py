"""EDA figures for the report. Saves PNGs to results/figures/."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config, resolve
from src.data import loader

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def fig_dataset_comparison(cfg: dict, outdir):
    """Side-by-side: why Kaggle was rejected and UCI accepted.

    Skipped when the Kaggle CSV is absent. That file needs account credentials
    to download, so it is missing on CI and on Kaggle's own servers, and this
    figure is only the evidence for a screening decision already made and
    committed -- not something the simulator depends on. Crashing the whole
    pipeline over a missing report chart would be badly disproportionate.
    """
    kaggle_path = resolve(cfg["paths"]["kaggle_raw"])
    if not kaggle_path.exists():
        print(f"      (skipping dataset-comparison figure: {kaggle_path.name} not present)")
        return

    kag = pd.read_csv(kaggle_path, parse_dates=["Date"])
    kag["dow"] = kag["Date"].dt.dayofweek
    kag["month"] = kag["Date"].dt.month
    uci = loader.read_panel(cfg)

    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))

    k = kag.groupby("dow")["Units Sold"].mean()
    ax[0, 0].bar(range(7), k.reindex(range(7)).values, color="#c0392b")
    ax[0, 0].set_title("Kaggle: demand by weekday (flat = no signal)")
    ax[0, 0].set_xticks(range(7)); ax[0, 0].set_xticklabels(DOW_NAMES)
    ax[0, 0].set_ylim(0, k.max() * 1.6)

    u = uci.groupby("dow")["demand"].mean()
    ax[0, 1].bar(u.index, u.values, color="#27ae60")
    ax[0, 1].set_title("UCI: demand by weekday (Sat closed)")
    ax[0, 1].set_xticks(range(7)); ax[0, 1].set_xticklabels(DOW_NAMES)

    km = kag.groupby("month")["Units Sold"].mean()
    ax[1, 0].plot(km.index, km.values, "o-", color="#c0392b")
    ax[1, 0].set_title("Kaggle: demand by month (flat)")
    ax[1, 0].set_ylim(0, km.max() * 1.6); ax[1, 0].set_xlabel("month")

    um = uci.groupby("month")["demand"].mean()
    ax[1, 1].plot(um.index, um.values, "o-", color="#27ae60")
    ax[1, 1].set_title("UCI: demand by month (Christmas ramp)")
    ax[1, 1].set_xlabel("month")

    fig.suptitle("Dataset screening: flat noise vs real retail structure", fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "01_dataset_comparison.png", bbox_inches="tight")
    plt.close(fig)


def fig_selected_products(stats: dict, cfg: dict, outdir):
    """Daily demand series of the ten chosen SKUs."""
    panel = loader.read_panel(cfg)
    codes = stats["selected_products"]
    split = pd.Timestamp(stats["train_test_split_date"])

    fig, axes = plt.subplots(5, 2, figsize=(12, 11), sharex=True)
    for axis, code in zip(axes.ravel(), codes):
        g = panel[panel["StockCode"] == code].sort_values("day")
        axis.plot(g["day"], g["demand"], lw=0.6, color="#2c3e50")
        axis.axvline(split, color="#e67e22", ls="--", lw=1)
        p = stats["products"][code]
        axis.set_title(
            f"{code} - {stats['descriptions'][code][:26]}\n"
            f"mean {p['mean_daily_demand']:.0f}  CV {p['cv']:.2f}  [{stats['regime_of'][code]}]",
            fontsize=8,
        )
    fig.suptitle("Selected products - daily demand (orange = train/test split)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "02_selected_products.png", bbox_inches="tight")
    plt.close(fig)


def fig_seasonality(stats: dict, outdir):
    """The multiplicative factors the simulator will reproduce."""
    # JSON object keys are always strings, so the integer weekday/month keys
    # written by calibrate.py come back as '0', '1', ... Coerce them back.
    dow = {int(k): v for k, v in stats["day_of_week_factor"].items()}
    mon = {int(k): v for k, v in stats["month_factor"].items()}

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    ks = sorted(dow)
    ax[0].bar([DOW_NAMES[k] for k in ks], [dow[k] for k in ks], color="#2980b9")
    ax[0].axhline(1.0, color="k", ls="--", lw=0.8)
    ax[0].set_title("Day-of-week demand factor")

    ms = sorted(mon)
    ax[1].bar(ms, [mon[m] for m in ms], color="#8e44ad")
    ax[1].axhline(1.0, color="k", ls="--", lw=0.8)
    ax[1].set_title("Month demand factor")
    ax[1].set_xticks(ms)

    fig.suptitle("Calibrated seasonality fed into the simulator", fontweight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "03_seasonality_factors.png", bbox_inches="tight")
    plt.close(fig)


def fig_regime_scatter(stats: dict, outdir):
    """Demand level vs volatility for the chosen SKUs."""
    codes = stats["selected_products"]
    colors = {"steady": "#27ae60", "moderate": "#f39c12", "bursty": "#c0392b"}

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for code in codes:
        p = stats["products"][code]
        r = stats["regime_of"][code]
        ax.scatter(p["mean_daily_demand"], p["cv"], s=90, color=colors[r],
                   edgecolor="k", linewidth=0.5, zorder=3)
        ax.annotate(code, (p["mean_daily_demand"], p["cv"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    for r, c in colors.items():
        ax.scatter([], [], color=c, label=r, s=90, edgecolor="k", linewidth=0.5)
    ax.set_xlabel("mean daily demand (units)")
    ax.set_ylabel("coefficient of variation")
    ax.legend(title="regime")
    ax.set_title("Chosen SKUs span three demand regimes\n"
                 "(a single fixed reorder rule cannot serve all three well)",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "04_demand_regimes.png", bbox_inches="tight")
    plt.close(fig)


def run(cfg: dict | None = None):
    cfg = cfg or load_config("data")
    outdir = resolve(cfg["paths"]["figures"])
    outdir.mkdir(parents=True, exist_ok=True)
    with resolve(cfg["paths"]["demand_stats"]).open(encoding="utf-8") as fh:
        stats = json.load(fh)

    fig_dataset_comparison(cfg, outdir)
    fig_selected_products(stats, cfg, outdir)
    fig_seasonality(stats, outdir)
    fig_regime_scatter(stats, outdir)
    return sorted(p.name for p in outdir.glob("*.png"))


if __name__ == "__main__":
    for name in run():
        print("wrote", name)
