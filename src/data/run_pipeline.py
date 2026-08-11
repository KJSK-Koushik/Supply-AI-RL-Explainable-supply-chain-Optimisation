"""Phase 1 entry point. Runs the whole data pipeline end to end.

    python -m src.data.run_pipeline
"""

from __future__ import annotations

import json

from src.config import load_config, resolve
from src.data import audit_datasets, calibrate, eda, loader


def main() -> None:
    cfg = load_config("data")

    print("[1/4] loading and cleaning UCI Online Retail II ...")
    panel, audit = loader.run(cfg)
    print(f"      {audit['rows_raw']:,} raw rows -> {audit['rows_clean']:,} clean rows")
    print(f"      {audit['products_clean']:,} products, {audit['trading_days']} trading days")
    print(f"      daily panel: {panel.shape[0]:,} rows")

    print("[2/4] auditing both candidate datasets ...")
    aud = audit_datasets.run(cfg)
    for name, res in aud.items():
        if res.get("status") == "audited":
            print(f"      {name}: {res['verdict']}")

    print("[3/4] calibrating simulator parameters ...")
    stats = calibrate.run(cfg)
    sc = stats["screening"]
    print(f"      screened {sc['products_considered']} products -> "
          f"{sc['eligible']} eligible -> {sc['shortlisted']} shortlisted -> "
          f"{len(stats['selected_products'])} selected")
    for reason, n in sc["rejected"].items():
        print(f"        rejected {n:5d}  {reason}")

    print("[4/4] generating EDA figures ...")
    for name in eda.run(cfg):
        print("      results/figures/" + name)

    print("\nSelected products:")
    print(f"  {'code':10s} {'description':30s} {'mean/day':>9s} {'CV':>6s} {'zero%':>7s} {'regime':>9s}")
    for code in stats["selected_products"]:
        p = stats["products"][code]
        print(f"  {code:10s} {stats['descriptions'][code][:28]:30s} "
              f"{p['mean_daily_demand']:9.1f} {p['cv']:6.2f} "
              f"{p['zero_day_fraction']*100:6.1f}% {stats['regime_of'][code]:>9s}")

    print(f"\nwrote {resolve(cfg['paths']['demand_stats'])}")


if __name__ == "__main__":
    main()
