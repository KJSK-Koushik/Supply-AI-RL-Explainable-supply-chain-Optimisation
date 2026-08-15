"""Compare every trained agent against every tuned baseline, honestly.

    python -m src.eval.compare
    python -m src.eval.compare --models results/models/masked_1m

Scores are computed on EVAL_SEEDS -- 30 episodes that neither the agents nor
the baselines were tuned on. Training-time profit (measured on seeds 900-905
while selecting checkpoints) must never be quoted as a result: those seeds
chose the model, so reporting them is the same error as quoting training
accuracy.

Every policy also faces identical demand and supplier behaviour per seed, so a
difference between two rows is a difference in skill rather than in luck.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import resolve
from src.eval.runner import EVAL_SEEDS, evaluate


def paired_comparison(a: list[float], b: list[float]) -> dict:
    """Paired test across shared seeds.

    Paired rather than independent because both policies met exactly the same
    customers on each seed. Pairing removes the episode-to-episode variance
    that dominates the raw standard deviations, and is the difference between
    "inside the noise" and a real effect.
    """
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    mean = float(d.mean())

    if n < 2:
        return {"mean_difference": mean, "t": float("nan"), "significant": False, "n": n}

    if d.std(ddof=1) == 0:
        # An identical margin on every single seed. Zero variance is the most
        # consistent result possible, not an undefined one -- treating it as
        # "not significant" would be exactly backwards. Only a zero margin
        # means no difference.
        return {
            "mean_difference": mean,
            "std_error": 0.0,
            "t": float("inf") if mean > 0 else (float("-inf") if mean < 0 else 0.0),
            "significant": mean != 0.0,
            "n": n,
        }

    se = float(d.std(ddof=1) / np.sqrt(n))
    t = mean / se
    # |t| > ~2.05 is p < 0.05 two-sided at 29 degrees of freedom.
    return {
        "mean_difference": mean,
        "std_error": se,
        "t": float(t),
        "significant": bool(abs(t) > 2.045),
        "n": n,
    }


def per_seed_profits(policy, seeds) -> list[float]:
    from src.eval.runner import run_episode

    return [run_episode(policy, s)["total_profit"] for s in seeds]


def collect_agents(model_dirs: list[Path]) -> dict:
    from src.agents.rl_policy import RLPolicy

    agents = {}
    for d in model_dirs:
        best = d / "best_model.zip"
        if not best.exists():
            continue
        label = d.name
        summary = d / "summary.json"
        if summary.exists():
            with summary.open(encoding="utf-8") as fh:
                meta = json.load(fh)
            if meta.get("action_masking"):
                label += " (masked)"
        agents[label] = RLPolicy.load(str(best))
    return agents


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None, help="model dirs; default = all")
    ap.add_argument("--out", default="results/comparison.json")
    args = ap.parse_args()

    from src.agents.tune_baselines import make

    dirs = (
        [Path(m) for m in args.models]
        if args.models
        else sorted(p for p in resolve("results/models").glob("*") if p.is_dir())
    )
    agents = collect_agents(dirs)

    bl_path = resolve("results/baselines.json")
    baselines = {}
    if bl_path.exists():
        with bl_path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        for kind, entry in data["policies"].items():
            if kind == "random":
                continue
            baselines[kind] = make(kind, entry["params"])
        best_baseline_name = data["best_baseline"]
    else:
        best_baseline_name = None

    everyone = {**baselines, **agents}
    if not everyone:
        print("nothing to compare")
        return

    print(f"Evaluating {len(everyone)} policies on {len(EVAL_SEEDS)} held-out seeds\n", flush=True)

    profits, summaries = {}, {}
    for name, pol in everyone.items():
        profits[name] = per_seed_profits(pol, EVAL_SEEDS)
        summaries[name] = evaluate(pol, EVAL_SEEDS)
        print(f"  {name:24s} {summaries[name]['total_profit']:12,.0f}", flush=True)

    ranked = sorted(summaries, key=lambda k: -summaries[k]["total_profit"])

    print(
        f"\n{'policy':24s}{'profit':>12s}{'+/-std':>9s}{'fill':>8s}{'overflow':>10s}{'ordering':>10s}"
    )
    for name in ranked:
        r = summaries[name]
        print(
            f"{name:24s}{r['total_profit']:12,.0f}{r['total_profit_std']:9,.0f}"
            f"{r['fill_rate']:7.1%}{r['overflow_loss']:10,.0f}{r['ordering_cost']:10,.0f}"
        )

    verdict = None
    if best_baseline_name and best_baseline_name in profits:
        best_agent = next((n for n in ranked if n in agents), None)
        if best_agent:
            cmp = paired_comparison(profits[best_agent], profits[best_baseline_name])
            verdict = {"agent": best_agent, "baseline": best_baseline_name, **cmp}
            direction = "BEATS" if cmp["mean_difference"] > 0 else "loses to"
            sig = "significant" if cmp["significant"] else "NOT significant (inside the noise)"
            print(
                f"\n{best_agent} {direction} {best_baseline_name} by "
                f"{abs(cmp['mean_difference']):,.0f} per episode  "
                f"(t={cmp['t']:+.2f}, {sig})"
            )

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "eval_seeds": EVAL_SEEDS,
                "summaries": summaries,
                "per_seed_profits": profits,
                "verdict": verdict,
            },
            fh,
            indent=2,
        )
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
