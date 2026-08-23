"""Phase 6 demo: explain real decisions in business English.

    python scripts/demo_explainer.py
    python scripts/demo_explainer.py --days 5 --product 3

Runs the tuned baseline policy through the simulator and explains what it did
on the last few days. Works without an API key -- the deterministic template
takes over and says so -- so the demo cannot be broken by a busy free endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Running this file directly puts scripts/ on sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import resolve  # noqa: E402
from src.env.supply_chain_env import SupplyChainEnv  # noqa: E402
from src.llm.client import OpenRouterClient  # noqa: E402
from src.llm.explainer import build_facts, explain  # noqa: E402


def _policy():
    """The best tuned baseline, or a plain top-up rule if none is stored."""
    path = resolve("results/baselines.json")
    if path.exists():
        from src.agents.tune_baselines import make

        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        best = data["best_baseline"]
        return best, make(best, data["policies"][best]["params"])
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3, help="decisions to explain")
    ap.add_argument("--product", type=int, default=0, help="product index, 0-9")
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--run-days", type=int, default=45, help="days to simulate first")
    args = ap.parse_args()

    import numpy as np

    name, policy = _policy()
    env = SupplyChainEnv(seed=args.seed)
    env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    client = OpenRouterClient()
    print(f"policy   : {name or 'fixed top-up rule'}")
    print(f"explainer: {'OpenRouter' if client.available else 'template only (no API key)'}")
    if client.available:
        print(f"models   : {', '.join(client.models)}")
    print()

    captured = []
    for day in range(args.run_days):
        if policy is not None:
            action = policy.act(env, rng)
        else:
            action = np.full(env.n_products, 2 * env.n_suppliers, dtype=int)
        _, _, terminated, truncated, info = env.step(action)
        if day >= args.run_days - args.days:
            captured.append(build_facts(env, info, args.product))
        if terminated or truncated:
            break

    for facts in captured:
        result = explain(facts, client=client)
        tag = result["model"] if result["source"] == "llm" else "template"
        print(
            f"--- day {facts['day_of_episode']} ({facts['weekday']}, {facts['month']}) [{tag}] ---"
        )
        print(result["text"])
        if result["reason"]:
            print(f"    (fell back: {result['reason']})")
        print()


if __name__ == "__main__":
    main()
