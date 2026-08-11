"""Grid-search the baseline policies, then report them on held-out seeds.

    python -m src.agents.tune_baselines

Why this file exists: comparing a carefully tuned RL agent against a
casually-configured (s,S) rule proves nothing. Every baseline here gets a real
parameter search on the same tuning seeds, and the winner of each family is
then measured on 30 held-out seeds it never saw. Those held-out numbers are the
bar the RL agent must clear in Phase 5.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

from src.agents.baselines import (
    SUPPLIER_RULES,
    ConstantPolicy,
    EOQPolicy,
    ForecastSafetyStockPolicy,
    NewsvendorPolicy,
    RandomPolicy,
    SSPolicy,
)
from src.config import resolve
from src.eval.runner import EVAL_SEEDS, TUNING_SEEDS, evaluate

SUPPLIER_NAMES = ["econo", "rapid", "midway", "urgency"]


def grid_ss():
    for s_days in [3, 5, 7, 9, 11, 14]:
        for extra in [3, 5, 8, 12, 16]:
            for rule in SUPPLIER_NAMES:
                yield {"s_days": s_days, "S_days": s_days + extra, "supplier_rule": rule}


# Grid ranges below were WIDENED after a first pass put three of the four
# tuned families on a boundary (safety_factor at its max, horizon_days and
# window at their min, z at its max). An optimum sitting on the edge of its
# range means the search wanted to go further and was stopped, which silently
# understates the baseline -- precisely the unfairness this module exists to
# prevent. Ranges are extended until every winner is interior.


def grid_eoq():
    for rop_days in [4, 6, 8, 10, 13, 16]:
        for safety in [1.0, 1.3, 1.6, 1.9, 2.2, 2.6]:
            for rule in SUPPLIER_NAMES:
                yield {"rop_days": rop_days, "safety_factor": safety, "supplier_rule": rule}


def grid_newsvendor():
    for horizon in [1, 2, 3, 5, 7, 10]:
        for service in [None, 0.85, 0.92, 0.97]:
            for rule in SUPPLIER_NAMES:
                yield {"horizon_days": horizon, "service_override": service, "supplier_rule": rule}


def grid_forecast():
    # window=1 is the hard floor -- a one-day "forecast" is simply yesterday's
    # demand, and there is no shorter history to average. If the search settles
    # there it is a real result, not a truncated grid.
    for window in [1, 2, 3, 5, 7, 14]:
        for z in [0.8, 1.28, 1.65, 2.05, 2.33, 2.6, 3.0]:
            for review in [2, 4, 7]:
                for rule in SUPPLIER_NAMES:
                    yield {"window": window, "z": z, "review_days": review, "supplier_rule": rule}


def grid_constant():
    for bucket in range(1, 6):
        for supplier in [0, 1, 2]:
            yield {"bucket": bucket, "supplier": supplier}


def make(kind: str, params: dict):
    p = dict(params)
    rule = SUPPLIER_RULES.get(p.pop("supplier_rule", "econo"))
    if kind == "(s,S)":
        return SSPolicy(p["s_days"], p["S_days"], rule)
    if kind == "EOQ+ROP":
        return EOQPolicy(p["rop_days"], rule, p["safety_factor"])
    if kind == "newsvendor":
        return NewsvendorPolicy(rule, p["horizon_days"], p["service_override"])
    if kind == "forecast+SS":
        return ForecastSafetyStockPolicy(rule, p["window"], p["z"], p["review_days"])
    if kind == "constant":
        return ConstantPolicy(p["bucket"], p["supplier"])
    raise ValueError(kind)


FAMILIES = {
    "(s,S)": grid_ss,
    "EOQ+ROP": grid_eoq,
    "newsvendor": grid_newsvendor,
    "forecast+SS": grid_forecast,
    "constant": grid_constant,
}


def _score_config(job: tuple[str, dict, list[int]]) -> tuple[dict, float]:
    """Worker for the process pool. Must be module-level to be picklable."""
    kind, params, seeds = job
    res = evaluate(make(kind, params), seeds)
    return params, res["total_profit"]


# Keys whose values are labels or hard-bounded by the environment, not points
# on a scale that could be extended. Flagging these produces false alarms:
# there is no fourth supplier to try, and no eighth order bucket.
NON_EXTENDABLE_KEYS = {"supplier", "supplier_rule", "bucket"}


def boundary_warnings(best: dict, configs: list[dict]) -> list[str]:
    """Flag any winning parameter sitting on the edge of its searched range.

    A boundary optimum means the grid, not the policy, decided the answer. The
    reported baseline is then a lower bound, and comparing RL against it would
    overstate the RL result.

    Only applies to genuinely continuous knobs. Categorical choices (which
    supplier) and choices the environment bounds (which order bucket) cannot be
    widened, so landing on their edge means nothing.
    """
    warnings = []
    for key, value in best.items():
        if key in NON_EXTENDABLE_KEYS:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        tried = sorted({c[key] for c in configs if isinstance(c.get(key), (int, float))})
        if len(tried) < 2:
            continue
        if value == tried[0]:
            warnings.append(f"{key}={value} is the MINIMUM of {tried}")
        elif value == tried[-1]:
            warnings.append(f"{key}={value} is the MAXIMUM of {tried}")
    return warnings


def tune_family(kind: str, grid_fn, seeds, jobs: int = 1) -> tuple[dict, float, int]:
    """Search one family. Returns best params, its tuning profit, grid size."""
    configs = list(grid_fn())

    if jobs <= 1:
        scored = [_score_config((kind, p, seeds)) for p in configs]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            scored = list(pool.map(_score_config, [(kind, p, seeds) for p in configs]))

    best_params, best_profit = max(scored, key=lambda x: x[1])
    return best_params, best_profit, len(configs), boundary_warnings(best_params, configs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 4) - 2),
        help="parallel worker processes",
    )
    ap.add_argument("--only", default=None, help="re-tune a single family, e.g. --only forecast+SS")
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help="skip the grid search; re-evaluate the parameters already "
        "stored in results/baselines.json on the held-out seeds",
    )
    args = ap.parse_args()

    families = FAMILIES if args.only is None else {args.only: FAMILIES[args.only]}

    # Previously tuned results, so a single-family re-run updates one entry
    # instead of silently discarding the other four.
    out_path = resolve("results/baselines.json")
    previous = {}
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fh:
            previous = json.load(fh).get("policies", {})

    if args.eval_only:
        tuned = {k: v["params"] for k, v in previous.items() if k != "random"}
        print(f"Re-evaluating {len(tuned)} stored policies on held-out seeds\n", flush=True)
        all_warnings = {}
        _report_and_save(tuned, all_warnings, previous, out_path)
        return

    # flush=True everywhere: piped stdout is block-buffered, so without it a
    # half-hour search looks identical to a hung process.
    print("Tuning baselines", flush=True)
    print(f"  tuning seeds : {len(TUNING_SEEDS)} episodes", flush=True)
    print(f"  eval seeds   : {len(EVAL_SEEDS)} episodes (held out)", flush=True)
    print(f"  workers      : {args.jobs}\n", flush=True)

    tuned = {}
    all_warnings = {}
    for kind, grid_fn in families.items():
        t0 = time.time()
        params, profit, n, warns = tune_family(kind, grid_fn, TUNING_SEEDS, jobs=args.jobs)
        dt = time.time() - t0
        tuned[kind] = params
        all_warnings[kind] = warns
        print(
            f"  {kind:14s} {n:4d} configs in {dt:6.1f}s -> tuning profit {profit:10,.0f}  {params}",
            flush=True,
        )
        for w in warns:
            print(
                f"      BOUNDARY: {w} -- widen the grid, this baseline is a lower bound", flush=True
            )

    _report_and_save(tuned, all_warnings, previous, out_path)


def _report_and_save(tuned: dict, all_warnings: dict, previous: dict, out_path) -> None:
    """Evaluate every known policy on the held-out seeds and write the report.

    `previous` carries entries from an earlier run so that re-tuning one family
    updates a single row rather than discarding the rest. Every policy is
    re-measured here, not copied forward, so all reported numbers come from the
    same code on the same seeds.
    """
    merged = {k: v["params"] for k, v in previous.items() if k != "random"}
    merged.update(tuned)

    print("\nHeld-out evaluation (30 unseen seeds)", flush=True)
    print(
        f"  {'policy':14s} {'profit':>12s} {'+/- std':>10s} {'fill':>7s} "
        f"{'holding':>9s} {'stockout':>10s} {'overflow':>9s}",
        flush=True,
    )

    report = {}
    rows = []
    for kind, params in merged.items():
        res = evaluate(make(kind, params), EVAL_SEEDS)
        report[kind] = {"params": params, "eval": res}
        rows.append((kind, res))

    rand = evaluate(RandomPolicy(), EVAL_SEEDS)
    report["random"] = {"params": {}, "eval": rand}
    rows.append(("random", rand))

    for kind, r in sorted(rows, key=lambda x: -x[1]["total_profit"]):
        print(
            f"  {kind:14s} {r['total_profit']:12,.0f} {r['total_profit_std']:10,.0f} "
            f"{r['fill_rate']:6.1%} {r['holding_cost']:9,.0f} "
            f"{r['stockout_penalty']:10,.0f} {r['overflow_loss']:9,.0f}",
            flush=True,
        )

    best = max(rows, key=lambda x: x[1]["total_profit"])
    print(f"\n  BEST BASELINE: {best[0]} at {best[1]['total_profit']:,.0f} profit", flush=True)
    print("  This is the bar the RL agent must clear in Phase 5.", flush=True)

    out = out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "tuning_seeds": TUNING_SEEDS,
                "eval_seeds": EVAL_SEEDS,
                "best_baseline": best[0],
                "boundary_warnings": all_warnings,
                "policies": report,
            },
            fh,
            indent=2,
        )
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
