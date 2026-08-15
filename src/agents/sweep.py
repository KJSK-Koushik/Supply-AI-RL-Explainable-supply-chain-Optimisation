"""Hyperparameter sweep with early pruning.

    python -m src.agents.sweep --budget 6                  # 6 configs
    python -m src.agents.sweep --budget 12 --steps 600000

Most hyperparameter settings announce themselves as hopeless early. Running
every one to completion wastes the majority of the compute, so each config gets
a short screening run first and only the survivors are trained fully
(successive halving). On the measured ~560 steps/s this roughly halves the
wall-clock cost of a search.

Designed to survive interruption: every config's result is appended to
sweep_results.json as it finishes, so a Kaggle session hitting its 12-hour
limit still leaves usable output.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time

from src.config import resolve

# Ranges revised after the first two 1M-step runs. What those showed:
#   * action masking is worth ~3x on its own (23,717 -> 72,740 best), so the
#     sweep assumes it and does not search over it
#   * the masked run plateaued near 70,000 against an 83,624 baseline, with
#     the remaining gap almost entirely stockouts (fill 93.7% vs 98.3%) and
#     supplier fragmentation (ordering fees 2.6x the baseline's)
#   * both runs oscillated throughout, which points at learning rate and
#     entropy rather than capacity
# Hence: lower learning rates to damp the oscillation, a wider entropy range
# (too little exploration is the likely cause of the supplier-consolidation
# blind spot), and larger networks to see if capacity is binding at all.
GRID = {
    "ppo.learning_rate": [5.0e-5, 1.0e-4, 3.0e-4],
    "ppo.ent_coef": [0.001, 0.005, 0.02, 0.05],
    "ppo.gamma": [0.995, 0.999],
    "ppo.n_steps": [512, 1024],
    "ppo.net_arch": ["[128,128]", "[256,256]"],
}


def configs(budget: int, seed: int = 0):
    """Random search over the grid. Random beats exhaustive grid search at
    equal budget when only a few hyperparameters actually matter."""
    import random

    rng = random.Random(seed)
    keys = list(GRID)
    all_combos = list(itertools.product(*(GRID[k] for k in keys)))
    rng.shuffle(all_combos)
    for combo in all_combos[:budget]:
        yield dict(zip(keys, combo, strict=True))


def run_one(name: str, params: dict, steps: int, extra: list[str]) -> dict:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.agents.train_ppo",
        "--name",
        name,
        "--timesteps",
        str(steps),
        "--set",
    ]
    cmd += [f"{k}={v}" for k, v in params.items()] + extra

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0

    summary_path = resolve("results/models") / name / "summary.json"
    best = None
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as fh:
            best = json.load(fh).get("best_eval_profit")

    if best is None:
        print(f"  {name}: FAILED\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}", flush=True)

    return {
        "name": name,
        "params": params,
        "steps": steps,
        "best_profit": best,
        "wall_seconds": dt,
        "ok": best is not None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=8, help="configs to screen")
    ap.add_argument("--screen-steps", type=int, default=200_000)
    ap.add_argument("--steps", type=int, default=1_000_000, help="steps for survivors")
    ap.add_argument("--keep", type=int, default=3, help="survivors to train fully")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--tag", default="sweep")
    args = ap.parse_args()

    extra = [f"run.n_envs={args.n_envs}"]
    out = resolve("results/sweep_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    print(f"SCREENING {args.budget} configs at {args.screen_steps:,} steps each", flush=True)
    for i, params in enumerate(configs(args.budget)):
        name = f"{args.tag}_s{i:02d}"
        r = run_one(name, params, args.screen_steps, extra)
        results.append(r)
        status = f"{r['best_profit']:,.0f}" if r["ok"] else "FAILED"
        print(
            f"  [{i + 1}/{args.budget}] {name}: {status}  ({r['wall_seconds'] / 60:.1f} min)",
            flush=True,
        )
        with out.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

    survivors = sorted([r for r in results if r["ok"]], key=lambda r: -r["best_profit"])[
        : args.keep
    ]

    print(f"\nFULL TRAINING for top {len(survivors)} at {args.steps:,} steps", flush=True)
    for i, s in enumerate(survivors):
        name = f"{args.tag}_full{i:02d}"
        r = run_one(name, s["params"], args.steps, extra)
        r["screened_from"] = s["name"]
        results.append(r)
        status = f"{r['best_profit']:,.0f}" if r["ok"] else "FAILED"
        print(
            f"  [{i + 1}/{len(survivors)}] {name}: {status}  ({r['wall_seconds'] / 60:.1f} min)",
            flush=True,
        )
        with out.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

    done = [r for r in results if r["ok"]]
    if done:
        best = max(done, key=lambda r: r["best_profit"])
        print(f"\nBEST: {best['name']} at {best['best_profit']:,.0f}", flush=True)
        print(f"  {best['params']}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
