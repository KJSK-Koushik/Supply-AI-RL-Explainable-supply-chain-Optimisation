"""Turn the real daily panel into simulator parameters.

This is the bridge between real data and the RL environment. The dataset
cannot train the agent directly -- RL needs to ask "what would have happened
if I had ordered 500 units?", which no historical record can answer. So the
data's job is to tell us what realistic demand *looks like*, and the simulator
then generates unlimited days that behave the same way.

Everything written here is derived from real UCI transactions. Supplier
attributes are the one exception and live in configs/suppliers.yaml, clearly
marked synthetic.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import load_config, resolve
from src.data import loader


def _product_profile(series: pd.Series) -> dict:
    """Summarise one product's daily demand series."""
    mean = float(series.mean())
    std = float(series.std(ddof=1))
    return {
        "mean_daily_demand": mean,
        "std_daily_demand": std,
        "cv": float(std / mean) if mean > 0 else float("inf"),
        "zero_day_fraction": float((series == 0).mean()),
        "median_daily_demand": float(series.median()),
        "p90_daily_demand": float(series.quantile(0.90)),
        "max_daily_demand": float(series.max()),
        "total_volume": float(series.sum()),
    }


def classify_regime(cv: float, bands: dict) -> str:
    """Map a coefficient of variation onto a demand regime label."""
    if cv < bands["steady_max"]:
        return "steady"
    if cv < bands["moderate_max"]:
        return "moderate"
    return "bursty"


def screen_products(panel: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Score every product and record why each one was kept or rejected.

    Screening looks only at the training period. Choosing products using the
    held-out window would leak test information into the experiment design --
    we would be picking the SKUs that happen to behave well in the data we
    later claim never to have seen.
    """
    sel = cfg["product_selection"]
    clean_cfg = cfg["cleaning"]

    split = pd.Timestamp(cfg["calendar"]["train_test_split_date"])
    panel = panel[panel["day"] < split]

    rows = []
    for code, grp in panel.groupby("StockCode"):
        s = grp.sort_values("day")["demand"]
        if s.sum() <= 0:
            continue
        # Profile the winsorised series, matching fit_patterns exactly. If
        # screening used raw CV while the reported CV was winsorised, a product
        # could be labelled 'bursty' yet display a low CV in the figures.
        prof = _product_profile(s.clip(upper=s.quantile(clean_cfg["winsorise_percentile"])))
        # One wholesale mega-order can be a product's entire history. Those
        # series are not retail demand and would teach the agent nothing.
        prof["top_day_share"] = float(s.max() / s.sum())
        prof["StockCode"] = code
        prof["price"] = float(grp["price"].median())
        rows.append(prof)

    cand = pd.DataFrame(rows)

    cand["reject_reason"] = ""
    def _reject(mask, reason):
        cand.loc[mask & (cand["reject_reason"] == ""), "reject_reason"] = reason

    _reject(cand["top_day_share"] > clean_cfg["max_single_day_share"], "single_order_dominated")
    _reject(cand["mean_daily_demand"] < sel["min_mean_daily_demand"], "too_low_volume")
    _reject(cand["zero_day_fraction"] > sel["max_zero_day_fraction"], "too_intermittent")

    eligible = cand[cand["reject_reason"] == ""].copy()
    n_eligible = int(len(eligible))
    # Cap the shortlist so selection stays among genuinely traded products.
    eligible = eligible.sort_values("total_volume", ascending=False).head(sel["candidate_pool"])

    screening = {
        "products_considered": int(len(cand)),
        "rejected": cand[cand["reject_reason"] != ""]["reject_reason"].value_counts().to_dict(),
        "eligible": n_eligible,
        "shortlisted": int(len(eligible)),
    }
    return eligible, screening


def choose_products(eligible: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Pick a spread across demand regimes rather than just the top sellers.

    If we took the ten highest-volume products they would all be steady movers
    and the agent would learn one policy that works everywhere. Mixing steady,
    moderate and bursty products is what makes the RL-vs-fixed-rule comparison
    meaningful: a single reorder point cannot serve all three well.
    """
    mix = cfg["product_selection"]["regime_mix"]
    bands = cfg["product_selection"]["regime_bands"]

    eligible = eligible.copy()
    eligible["regime"] = eligible["cv"].apply(lambda cv: classify_regime(cv, bands))

    picked = []
    for regime, n in mix.items():
        pool = eligible[eligible["regime"] == regime].sort_values("total_volume", ascending=False)
        picked.append(pool.head(n))
    chosen = pd.concat(picked, ignore_index=True)

    # If a regime was underpopulated, top up from whatever is left by volume.
    shortfall = cfg["product_selection"]["n_products"] - len(chosen)
    if shortfall > 0:
        rest = eligible[~eligible["StockCode"].isin(chosen["StockCode"])]
        chosen = pd.concat(
            [chosen, rest.sort_values("total_volume", ascending=False).head(shortfall)],
            ignore_index=True,
        )
    return chosen.head(cfg["product_selection"]["n_products"])


def fit_patterns(panel: pd.DataFrame, codes: list[str], cfg: dict) -> dict:
    """Extract the seasonal structure the simulator will reproduce."""
    sub = panel[panel["StockCode"].isin(codes)].copy()
    split = pd.Timestamp(cfg["calendar"]["train_test_split_date"])
    train = sub[sub["day"] < split]

    overall = train["demand"].mean()

    # Multiplicative factors: 1.0 means an average day.
    dow_factor = (train.groupby("dow")["demand"].mean() / overall).round(4).to_dict()
    month_factor = (train.groupby("month")["demand"].mean() / overall).round(4).to_dict()

    per_product = {}
    for code, grp in sub.groupby("StockCode"):
        grp = grp.sort_values("day")
        tr = grp[grp["day"] < split]["demand"]
        te = grp[grp["day"] >= split]["demand"]

        winsor = tr.quantile(cfg["cleaning"]["winsorise_percentile"])
        tr_w = tr.clip(upper=winsor)

        prof = _product_profile(tr_w)
        prof["raw_std_before_winsorise"] = float(tr.std(ddof=1))
        prof["winsorise_cap"] = float(winsor)
        prof["price"] = float(grp["price"].median())
        prof["holdout_mean_daily_demand"] = float(te.mean())
        prof["holdout_days"] = int(len(te))

        # Negative-binomial style overdispersion, used by the simulator's
        # demand generator. Gamma-Poisson mixture reproduces the burstiness
        # that a plain Poisson cannot.
        m, v = prof["mean_daily_demand"], prof["std_daily_demand"] ** 2
        prof["overdispersion_r"] = float(m**2 / (v - m)) if v > m else None

        per_product[code] = prof

    return {
        "day_of_week_factor": {int(k): float(v) for k, v in dow_factor.items()},
        "month_factor": {int(k): float(v) for k, v in month_factor.items()},
        "products": per_product,
    }


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config("data")
    panel = loader.read_panel(cfg)

    eligible, screening = screen_products(panel, cfg)
    chosen = choose_products(eligible, cfg)
    codes = chosen["StockCode"].tolist()

    patterns = fit_patterns(panel, codes, cfg)

    descriptions = {}
    try:
        raw = pd.read_pickle(resolve(cfg["paths"]["uci_cache"]))
        raw["StockCode"] = raw["StockCode"].astype(str).str.strip().str.upper()
        for code in codes:
            d = raw[raw["StockCode"] == code]["Description"].mode()
            descriptions[code] = str(d.iloc[0]) if len(d) else code
    except Exception:
        descriptions = {c: c for c in codes}

    stats = {
        "source": "UCI Online Retail II (real transactional data)",
        "supplier_data": "SYNTHETIC - see configs/suppliers.yaml",
        "train_test_split_date": cfg["calendar"]["train_test_split_date"],
        "closed_weekdays": cfg["calendar"]["closed_weekdays"],
        "screening": screening,
        "selected_products": codes,
        "descriptions": descriptions,
        # Derived from the same winsorised CV that appears in stats['products'],
        # so a label can never disagree with the number shown beside it.
        "regime_of": {
            c: classify_regime(patterns["products"][c]["cv"],
                               cfg["product_selection"]["regime_bands"])
            for c in codes
        },
        "regime_bands": cfg["product_selection"]["regime_bands"],
        **patterns,
    }

    out = resolve(cfg["paths"]["demand_stats"])
    with out.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    return stats


if __name__ == "__main__":
    s = run()
    print(f"selected {len(s['selected_products'])} products")
    for c in s["selected_products"]:
        p = s["products"][c]
        print(f"  {c:8s} {s['descriptions'][c][:30]:32s} "
              f"mean={p['mean_daily_demand']:7.1f} cv={p['cv']:5.2f} "
              f"regime={s['regime_of'][c]}")
