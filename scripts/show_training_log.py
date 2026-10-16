"""Print saved training results as plain console tables.

    python scripts/show_training_log.py history kaggle_full01
    python scripts/show_training_log.py sweep

Both read files the training runs already wrote (``eval_history.json`` inside a
model folder, and ``results/sweep_results.json``). Nothing is recomputed or
invented here; the script only formats what was saved. Profits are shown in
rupees (lakh) with the project's 1 GBP = 105 INR display rate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.money import RATE  # noqa: E402


def lakh(gbp: float) -> float:
    return gbp * RATE / 1e5


def history(run: str, every: int = 100_000) -> None:
    path = ROOT / "results" / "models" / run / "eval_history.json"
    if not path.exists():
        raise SystemExit(f"no eval_history.json for run '{run}' ({path})")
    rows = json.loads(path.read_text(encoding="utf-8"))
    print(f"run '{run}': {len(rows)} checkpoint evaluations on the checkpoint seeds (900-905)")
    print(f"{'timesteps':>10}  {'profit (Rs lakh)':>17}  {'fill rate':>9}")
    print("-" * 42)
    best = max(rows, key=lambda r: r["profit"])
    for r in rows:
        if r["timesteps"] % every == 0 or r is rows[-1]:
            mark = "  <- best" if r is best else ""
            print(
                f"{r['timesteps']:>10,}  {lakh(r['profit']):>17.2f}  {r['fill_rate']:>8.1%}{mark}"
            )
    print("-" * 42)
    print(f"best checkpoint: {best['timesteps']:,} steps, Rs {lakh(best['profit']):.2f} lakh")


def sweep() -> None:
    path = ROOT / "results" / "sweep_results.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    print(f"hyperparameter sweep: {len(rows)} runs")
    head = f"{'run':<12} {'lr':>7} {'ent':>6} {'gamma':>6} {'steps':>9} {'best (Rs lakh)':>15} {'minutes':>8}"
    print(head)
    print("-" * len(head))
    for r in rows:
        p = r["params"]
        best = lakh(r["best_profit"])
        print(
            f"{r['name']:<12} {p['ppo.learning_rate']:>7.0e} {p['ppo.ent_coef']:>6} "
            f"{p['ppo.gamma']:>6} {r['steps']:>9,} {best:>15.2f} {r['wall_seconds'] / 60:>8.1f}"
        )


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "history" and len(sys.argv) > 2:
        history(sys.argv[2])
    elif cmd == "sweep":
        sweep()
    else:
        raise SystemExit(__doc__)
