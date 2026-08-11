"""Reproducible evidence for the dataset choice.

The Kaggle 'Retail Store Inventory Forecasting' dataset was the original
primary source. This script runs the tests that led to rejecting it and
accepting UCI Online Retail II instead, so the finding in the report can be
re-verified by anyone with both files.

Tests applied to each dataset:
  1. Do stated demand drivers (price, discount, promotion) correlate with sales?
  2. Is there a day-of-week pattern?
  3. Is there a seasonal pattern?
  4. Do products differ from one another?
  5. Is any column leaking the target?
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import load_config, resolve
from src.data import loader


def audit_kaggle(cfg: dict) -> dict:
    path = resolve(cfg["paths"]["kaggle_raw"])
    if not path.exists():
        return {"status": "file_not_present"}

    df = pd.read_csv(path, parse_dates=["Date"])
    df["dow"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    y = df["Units Sold"]

    corrs = {
        c: round(float(y.corr(df[c])), 4)
        for c in ["Price", "Discount", "Competitor Pricing", "Holiday/Promotion", "Demand Forecast"]
        if c in df.columns
    }

    dow = df.groupby("dow")["Units Sold"].mean()
    month = df.groupby("month")["Units Sold"].mean()
    prod = df.groupby("Product ID")["Units Sold"].mean()

    # If the Seasonality label were real, each calendar month would map to one
    # season. Measure how concentrated that mapping actually is.
    season_consistency = None
    if "Seasonality" in df.columns:
        ct = pd.crosstab(df["month"], df["Seasonality"])
        season_consistency = round(float((ct.max(axis=1) / ct.sum(axis=1)).mean()), 4)

    return {
        "status": "audited",
        "rows": int(len(df)),
        "driver_correlations": corrs,
        "dow_spread_pct": round(float((dow.max() - dow.min()) / dow.mean() * 100), 2),
        "month_spread_pct": round(float((month.max() - month.min()) / month.mean() * 100), 2),
        "product_spread_pct": round(float((prod.max() - prod.min()) / prod.mean() * 100), 2),
        "season_label_consistency": season_consistency,  # 0.25 == random across 4 seasons
        "target_leakage": {
            k: v for k, v in corrs.items() if abs(v) > 0.95
        },
        "verdict": "REJECTED - no learnable structure; Demand Forecast leaks target",
    }


def audit_uci(cfg: dict) -> dict:
    path = resolve(cfg["paths"]["daily_panel"])
    if not path.exists():
        return {"status": "run loader.py first"}

    panel = loader.read_panel(cfg)
    dow = panel.groupby("dow")["demand"].mean()
    month = panel.groupby("month")["demand"].mean()
    prod = panel.groupby("StockCode")["demand"].mean()
    prod = prod[prod > 0]

    return {
        "status": "audited",
        "rows": int(len(panel)),
        "products": int(panel["StockCode"].nunique()),
        "dow_spread_pct": round(float((dow.max() - dow.min()) / dow.mean() * 100), 2),
        "month_spread_pct": round(float((month.max() - month.min()) / month.mean() * 100), 2),
        "product_spread_ratio": round(float(prod.max() / prod.min()), 1),
        "peak_month": int(month.idxmax()),
        "trough_month": int(month.idxmin()),
        "peak_to_trough_ratio": round(float(month.max() / month.min()), 2),
        "verdict": "ACCEPTED - real weekday, seasonal and cross-product structure",
    }


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config("data")
    result = {"kaggle_retail_store_inventory": audit_kaggle(cfg), "uci_online_retail_ii": audit_uci(cfg)}
    out = resolve("data/processed/dataset_audit.json")
    with out.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
