"""Load and clean UCI Online Retail II into a daily per-product demand panel.

The raw file is 1,067,371 transaction lines across two Excel sheets. What the
simulator needs is a rectangular table: one row per (product, calendar day)
with the quantity sold. Getting there means removing the things that are not
retail demand -- cancellations, postage lines, negative quantities -- and then
filling in the days on which a product simply did not sell.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve


def read_panel(cfg: dict) -> pd.DataFrame:
    """Read the daily panel back from disk.

    StockCode must be forced to string. Many codes are purely numeric ('21212'),
    so pandas infers a mixed int/str column on re-read. That silently breaks
    lookups after a JSON round-trip, because JSON object keys are always
    strings while a list of codes keeps its integers.
    """
    return pd.read_csv(
        resolve(cfg["paths"]["daily_panel"]),
        parse_dates=["day"],
        dtype={"StockCode": str},
    )


def load_raw(cfg: dict) -> pd.DataFrame:
    """Read both Excel sheets, using a pickle cache because the xlsx parse
    takes roughly 40 seconds."""
    cache = resolve(cfg["paths"]["uci_cache"])
    if cache.exists():
        df = pd.read_pickle(cache)
    else:
        xlsx = resolve(cfg["paths"]["uci_raw"])
        sheets = pd.read_excel(xlsx, sheet_name=None)
        df = pd.concat(sheets.values(), ignore_index=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()
    return df


def clean(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Filter to genuine retail sales. Returns the clean frame plus an audit
    dict recording how many rows each rule removed -- that table goes straight
    into the report."""
    c = cfg["cleaning"]
    audit: dict[str, int] = {"rows_raw": len(df)}
    out = df

    if c["drop_cancellations"]:
        mask = ~out["Invoice"].astype(str).str.upper().str.startswith("C")
        audit["removed_cancellations"] = int((~mask).sum())
        out = out[mask]

    if c["drop_nonpositive_quantity"]:
        mask = out["Quantity"] > 0
        audit["removed_nonpositive_quantity"] = int((~mask).sum())
        out = out[mask]

    if c["drop_nonpositive_price"]:
        mask = out["Price"] > 0
        audit["removed_nonpositive_price"] = int((~mask).sum())
        out = out[mask]

    if c["drop_service_codes"]:
        codes = {str(s).upper() for s in c["service_codes"]}
        # Service lines are postage, bank charges, manual adjustments and so on.
        # They carry a StockCode but are not sellable stock, so an inventory
        # agent must never see them. Pure-letter codes are the same family, and
        # gift vouchers appear as GIFT_0001_20 style variants.
        mask = ~(
            out["StockCode"].isin(codes)
            | out["StockCode"].str.fullmatch(r"[A-Z]+")
            | out["StockCode"].str.startswith("GIFT")
        )
        audit["removed_service_codes"] = int((~mask).sum())
        out = out[mask]

    out = out.copy()
    out["day"] = out["InvoiceDate"].dt.normalize()
    audit["rows_clean"] = len(out)
    audit["products_clean"] = int(out["StockCode"].nunique())
    audit["date_min"] = str(out["day"].min().date())
    audit["date_max"] = str(out["day"].max().date())
    audit["trading_days"] = int(out["day"].nunique())
    return out, audit


def build_daily_panel(clean_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Aggregate transactions to one row per (product, day).

    Days with no transaction become zero-demand rows. Saturdays are dropped
    entirely rather than recorded as zeros: this retailer does not trade on
    Saturdays (5,119 units in two years against ~2M on other weekdays), so a
    Saturday zero means 'closed', not 'nobody wanted it'. Feeding those zeros
    to the demand estimator would bias every mean downwards by a seventh.
    """
    daily = (
        clean_df.groupby(["StockCode", "day"], as_index=False)
        .agg(demand=("Quantity", "sum"), price=("Price", "median"), n_orders=("Invoice", "nunique"))
    )

    calendar = pd.date_range(clean_df["day"].min(), clean_df["day"].max(), freq="D")
    closed = set(cfg["calendar"]["closed_weekdays"])
    calendar = calendar[~calendar.dayofweek.isin(closed)]

    products = daily["StockCode"].unique()
    full_index = pd.MultiIndex.from_product([products, calendar], names=["StockCode", "day"])

    panel = (
        daily.set_index(["StockCode", "day"])
        .reindex(full_index)
        .reset_index()
    )
    panel["demand"] = panel["demand"].fillna(0.0)
    panel["n_orders"] = panel["n_orders"].fillna(0.0)
    # A product's price only exists on days it sold; carry the last known price
    # forward, then backward for the leading gap, so every row has one.
    panel["price"] = panel.groupby("StockCode")["price"].ffill().bfill()

    panel["dow"] = panel["day"].dt.dayofweek
    panel["month"] = panel["day"].dt.month
    return panel


def run(cfg: dict | None = None) -> tuple[pd.DataFrame, dict]:
    cfg = cfg or load_config("data")
    raw = load_raw(cfg)
    clean_df, audit = clean(raw, cfg)
    panel = build_daily_panel(clean_df, cfg)

    audit["panel_rows"] = len(panel)
    audit["panel_days"] = int(panel["day"].nunique())

    out_path = resolve(cfg["paths"]["daily_panel"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out_path, index=False)

    audit_path = resolve(cfg["paths"]["audit_report"])
    with audit_path.open("w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)

    return panel, audit


if __name__ == "__main__":
    panel, audit = run()
    print(json.dumps(audit, indent=2))
    print(f"\npanel shape: {panel.shape}")
