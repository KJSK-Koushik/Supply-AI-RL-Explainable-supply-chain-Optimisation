"""Are the demand patterns the simulator relies on stable over time?

    python -m src.data.stability

The dataset is 2009-2011 and the faculty asked, fairly, whether markets have
moved on. The simulator does not use the data's *levels* -- prices and volumes
are config -- it uses the *shape* of demand: the weekly rhythm, the seasonal
curve, and how bursty daily sales are. This asks whether those shapes are a
property of the period or of retail.

Method: split the two years of data into two disjoint one-year windows, fit
the same factors the simulator uses on each independently, and compare. If the
shape were drifting with the market, the two years would disagree. The test
cannot prove 2026 looks like 2011; it can show that the structure did not move
across the two years available, which is the evidence the data can offer.
"""

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
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

WINDOWS = {
    "year 1 (Dec 2009 - Nov 2010)": ("2009-12-01", "2010-12-01"),
    "year 2 (Dec 2010 - Nov 2011)": ("2010-12-01", "2011-12-01"),
}


def factors(sub: pd.DataFrame) -> tuple[pd.Series, pd.Series, float]:
    """Weekday factor, month factor, and variance-to-mean ratio, as calibrate.py
    computes them -- same code path, so this measures the simulator's own
    inputs rather than a proxy."""
    overall = sub["demand"].mean()
    dow = sub.groupby("dow")["demand"].mean() / overall
    month = sub.groupby("month")["demand"].mean() / overall
    per_product = sub.groupby("StockCode")["demand"].agg(["mean", "var"])
    vmr = float((per_product["var"] / per_product["mean"]).median())
    return dow, month, vmr


def run() -> dict:
    cfg = load_config("data")
    panel = loader.read_panel(cfg)
    with resolve(cfg["paths"]["demand_stats"]).open(encoding="utf-8") as fh:
        codes = json.load(fh)["selected_products"]
    sub = panel[panel["StockCode"].isin(codes)]

    fits = {}
    for name, (lo, hi) in WINDOWS.items():
        w = sub[(sub["day"] >= lo) & (sub["day"] < hi)]
        fits[name] = factors(w)

    (dow1, mon1, vmr1), (dow2, mon2, vmr2) = fits.values()
    dow_r = float(np.corrcoef(dow1.reindex(dow2.index), dow2)[0, 1])
    mon_r = float(np.corrcoef(mon1.reindex(mon2.index), mon2)[0, 1])
    dow_maxdiff = float((dow1.reindex(dow2.index) - dow2).abs().max())
    mon_maxdiff = float((mon1.reindex(mon2.index) - mon2).abs().max())
    # December is partly a window artefact: year 1's December is the first
    # days of the dataset and the retailer closes for Christmas, so it is
    # reported with and without.
    keep = [m for m in mon2.index if m != 12 and m in mon1.index]
    mon_r_no_dec = float(np.corrcoef(mon1.reindex(keep), mon2.reindex(keep))[0, 1])

    out = {
        "windows": WINDOWS,
        "weekday_factor_correlation": dow_r,
        "weekday_factor_max_abs_diff": dow_maxdiff,
        "month_factor_correlation": mon_r,
        "month_factor_correlation_excluding_december": mon_r_no_dec,
        "peak_month": {"year_1": int(mon1.idxmax()), "year_2": int(mon2.idxmax())},
        "month_factor_max_abs_diff": mon_maxdiff,
        "variance_to_mean_ratio": {"year_1": vmr1, "year_2": vmr2},
        "weekday_factors": {
            "year_1": {int(k): float(v) for k, v in dow1.items()},
            "year_2": {int(k): float(v) for k, v in dow2.items()},
        },
        "month_factors": {
            "year_1": {int(k): float(v) for k, v in mon1.items()},
            "year_2": {int(k): float(v) for k, v in mon2.items()},
        },
    }

    # ---- figure --------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    days = sorted(set(dow1.index) | set(dow2.index))
    x = np.arange(len(days))
    ax[0].bar(x - 0.2, [dow1.get(d, np.nan) for d in days], 0.4, label="year 1", color="#2c7a4b")
    ax[0].bar(x + 0.2, [dow2.get(d, np.nan) for d in days], 0.4, label="year 2", color="#c0392b")
    ax[0].axhline(1.0, color="k", lw=0.8, ls="--")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([DOW[d] for d in days])
    ax[0].set_title(f"Weekday factor, two independent years  (r = {dow_r:.3f})", fontweight="bold")
    ax[0].legend(fontsize=8)

    months = sorted(set(mon1.index) | set(mon2.index))
    ax[1].plot(months, [mon1.get(m, np.nan) for m in months], "o-", color="#2c7a4b", label="year 1")
    ax[1].plot(months, [mon2.get(m, np.nan) for m in months], "s-", color="#c0392b", label="year 2")
    ax[1].axhline(1.0, color="k", lw=0.8, ls="--")
    ax[1].set_xticks(months)
    ax[1].set_xlabel("month")
    ax[1].set_title(f"Month factor, two independent years  (r = {mon_r:.3f})", fontweight="bold")
    ax[1].legend(fontsize=8)

    # The title states what was found, not what was hoped for. The weekly
    # rhythm and the November peak repeat; the mid-year shape does not.
    fig.suptitle(
        "Weekly rhythm and the November peak repeat across two independent years; "
        "the mid-year shape does not",
        fontweight="bold",
    )
    fig.tight_layout()
    figdir = resolve(cfg["paths"]["figures"])
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / "08_pattern_stability.png", bbox_inches="tight")
    plt.close(fig)

    path = resolve("results/pattern_stability.json")
    with path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print(f"weekday factor   r = {dow_r:.3f}   max |diff| = {dow_maxdiff:.3f}")
    print(
        f"month factor     r = {mon_r:.3f}   (excl. Dec {mon_r_no_dec:.3f})   max |diff| = {mon_maxdiff:.3f}"
    )
    print(f"peak month       year 1 = {mon1.idxmax()}   year 2 = {mon2.idxmax()}")
    print(f"variance/mean    year 1 = {vmr1:.0f}   year 2 = {vmr2:.0f}")
    print(f"wrote {path.name} and 08_pattern_stability.png")
    return out


if __name__ == "__main__":
    run()
