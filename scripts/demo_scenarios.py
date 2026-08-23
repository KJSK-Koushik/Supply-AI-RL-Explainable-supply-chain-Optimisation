"""Phase 7 demo: let an LLM invent a crisis, then measure what it costs.

    python scripts/demo_scenarios.py
    python scripts/demo_scenarios.py --offline        # hand-written crises
    python scripts/demo_scenarios.py --theme "a port strike and a heatwave"

Generates disruption scenarios, prints them, then scores every stored policy
under calm and disrupted conditions on the same seeds. The comparison is the
point: a scenario set that parses cleanly but costs nothing is decoration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import resolve  # noqa: E402
from src.eval.runner import evaluate  # noqa: E402
from src.llm.client import OpenRouterClient  # noqa: E402
from src.llm.scenario_gen import fallback_scenarios, generate  # noqa: E402

SEEDS = [500, 501, 502, 503, 504]


def _policies() -> dict:
    out = {}
    path = resolve("results/baselines.json")
    if path.exists():
        from src.agents.tune_baselines import make

        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        best = data["best_baseline"]
        out[f"{best} (classical)"] = make(best, data["policies"][best]["params"])

    best_model, best_profit = None, None
    for d in sorted(p for p in resolve("results/models").glob("*") if p.is_dir()):
        summary = d / "summary.json"
        if not (d / "best_model.zip").exists() or not summary.exists():
            continue
        with summary.open(encoding="utf-8") as fh:
            profit = json.load(fh).get("best_eval_profit")
        if profit is not None and (best_profit is None or profit > best_profit):
            best_model, best_profit = d, profit
    if best_model:
        from src.agents.rl_policy import RLPolicy

        out[f"{best_model.name} (RL)"] = RLPolicy.load(str(best_model / "best_model.zip"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="scenarios to request")
    ap.add_argument("--theme", default="a realistic retail supply chain crisis")
    ap.add_argument("--offline", action="store_true", help="skip the LLM entirely")
    args = ap.parse_args()

    if args.offline:
        scenarios, source, model = fallback_scenarios(), "fallback (offline)", None
    else:
        client = OpenRouterClient()
        out = generate(n=args.n, theme=args.theme, client=client)
        scenarios, source, model = out["scenarios"], out["source"], out["model"]
        if out["reason"]:
            print(f"note: {out['reason']}\n")

    print(f"scenarios from: {source}" + (f"  ({model})" if model else ""))
    for sc in scenarios:
        print(f"  - {sc.describe()}")
        if sc.label:
            print(f"      cause: {sc.label}")
    print()

    policies = _policies()
    if not policies:
        print("no stored policies to score; run Phase 3 first")
        return

    print(f"Scored on {len(SEEDS)} seeds, calm vs disrupted (GBP per episode)\n")
    print(f"{'policy':28s}{'calm':>12s}{'disrupted':>12s}{'change':>12s}{'fill':>9s}")
    for name, policy in policies.items():
        calm = evaluate(policy, SEEDS)
        rough = evaluate(policy, SEEDS, scenarios=scenarios)
        delta = rough["total_profit"] - calm["total_profit"]
        print(
            f"{name:28s}{calm['total_profit']:12,.0f}{rough['total_profit']:12,.0f}"
            f"{delta:+12,.0f}{rough['fill_rate']:9.1%}"
        )


if __name__ == "__main__":
    main()
