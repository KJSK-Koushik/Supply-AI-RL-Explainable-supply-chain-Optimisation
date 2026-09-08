"""Phase 9 result: does curriculum training make the agent less brittle?

    python -m src.eval.robustness

Scores every policy twice -- once on calm episodes, once on each held-out
disruption set -- and reports the drop. The drop is the number that matters.
An agent that earns less overall but degrades gracefully may be the better
system to deploy, and average profit alone hides that entirely.

The disruption sets used here are the holdout half of the curriculum pool. The
curriculum agent trained on the other half and has never seen these. Scoring
it on the sets it trained against would show a large improvement that means
nothing.
"""

from __future__ import annotations

import json

import numpy as np

from src.agents.curriculum import build_pool, split_pool
from src.config import resolve
from src.eval.compare import paired_comparison
from src.eval.runner import EVAL_SEEDS, run_episode

CALM_SEEDS = EVAL_SEEDS[:10]  # ten seeds per condition keeps the run tractable


def load_policies() -> dict:
    """Best tuned baseline plus every trained agent found on disk."""
    policies: dict = {}

    bl = resolve("results/baselines.json")
    if bl.exists():
        from src.agents.tune_baselines import make

        with bl.open(encoding="utf-8") as fh:
            data = json.load(fh)
        best = data["best_baseline"]
        policies[f"{best} (classical)"] = make(best, data["policies"][best]["params"])

    for d in sorted(p for p in resolve("results/models").glob("*") if p.is_dir()):
        model = d / "best_model.zip"
        if not model.exists() or d.name.endswith("probe"):
            continue
        from src.agents.rl_policy import RLPolicy

        policies[d.name] = RLPolicy.load(str(model))
    return policies


def score(policy, scenario_sets: list) -> tuple[float, float]:
    """Mean profit when calm, and mean profit across the disruption sets."""
    calm = float(np.mean([run_episode(policy, s)["total_profit"] for s in CALM_SEEDS]))
    rough = []
    for scenarios in scenario_sets:
        for seed in CALM_SEEDS[:5]:
            rough.append(run_episode(policy, seed, scenarios=scenarios)["total_profit"])
    return calm, float(np.mean(rough))


def per_set_drops(policy, scenario_sets: list) -> list[float]:
    """Profit drop on each (set, seed) pair, for the paired test.

    Paired across identical (disruption, seed) pairs: every policy faces the
    same crisis on the same customers, so a difference in drop is a difference
    in robustness rather than in which crises it happened to draw.
    """
    drops = []
    for scenarios in scenario_sets:
        for seed in CALM_SEEDS[:5]:
            calm = run_episode(policy, seed)["total_profit"]
            rough = run_episode(policy, seed, scenarios=scenarios)["total_profit"]
            drops.append(rough - calm)
    return drops


def main() -> None:
    pool = build_pool(use_llm=False)  # cached; never regenerates mid-experiment
    _train_sets, holdout = split_pool(pool)
    policies = load_policies()
    if not policies:
        print("nothing to score")
        return

    print(
        f"Scoring {len(policies)} policies on {len(CALM_SEEDS)} calm seeds and "
        f"{len(holdout)} held-out disruption sets\n",
        flush=True,
    )

    rows, drops = {}, {}
    for name, policy in policies.items():
        calm, rough = score(policy, holdout)
        rows[name] = {
            "calm": calm,
            "disrupted": rough,
            "drop": rough - calm,
            "drop_pct": (rough - calm) / calm * 100 if calm else float("nan"),
        }
        drops[name] = per_set_drops(policy, holdout)
        print(
            f"  {name:26s} calm {calm:11,.0f}   disrupted {rough:11,.0f}   "
            f"{rows[name]['drop_pct']:+6.1f}%",
            flush=True,
        )

    # The control is named, not discovered. curriculum_1m was trained with
    # masked_1m's exact hyperparameters, seed and step count, so it is the only
    # run that isolates the curriculum as the variable. Picking "some other
    # agent with 1m in its name" would have selected baseline_1m, which lacks
    # action masking, or a swept 2M-step run -- either would attribute the
    # difference to the curriculum when it came from something else.
    curriculum = "curriculum_1m" if "curriculum_1m" in rows else None
    plain = "masked_1m" if "masked_1m" in rows else None
    verdict = None
    if curriculum and plain:
        cmp = paired_comparison(drops[curriculum], drops[plain])
        verdict = {"curriculum": curriculum, "baseline_agent": plain, **cmp}
        better = cmp["mean_difference"] > 0
        sig = "significant" if cmp["significant"] else "NOT significant"
        print(
            f"\n{curriculum} loses {abs(cmp['mean_difference']):,.0f} "
            f"{'less' if better else 'more'} profit to disruption than {plain} "
            f"(t={cmp['t']:+.2f}, {sig})"
        )

    out = resolve("results/robustness.json")
    with out.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "calm_seeds": CALM_SEEDS,
                "n_holdout_sets": len(holdout),
                "rows": rows,
                "verdict": verdict,
            },
            fh,
            indent=2,
        )
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
