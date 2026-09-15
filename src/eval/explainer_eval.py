"""Measure the explainer, rather than just having built it.

    python -m src.eval.explainer_eval --n 30

Objective 3 of this project is "use an LLM to explain decisions in business
language". Building the explainer satisfies the letter of that; this measures
whether it works. Three questions, each a percentage:

  grounded   Does the explanation contain only numbers the simulator produced?
             A fluent sentence with an invented figure is worse than a plain
             one, because it is believed.
  complete   Does it state the facts a buyer needs -- what is in stock, what
             was ordered, from whom?
  accurate   Both at once. This is the figure reported as explanation accuracy.

Also reports how often the LLM was actually reachable, because free endpoints
rate-limit constantly and an evaluation that silently scored the template
fallback as the model would be measuring the wrong thing.
"""

from __future__ import annotations

import argparse
import json
import re
import time

import numpy as np

from src.config import resolve
from src.env.supply_chain_env import SupplyChainEnv
from src.llm.client import OpenRouterClient
from src.llm.explainer import build_facts, explain, template_explanation, ungrounded_numbers

SAMPLE_SEED = 500
SAMPLE_DAYS = (20, 35, 50, 65, 80, 95)


def _mentions(text: str, value) -> bool:
    """Whether a fact appears in the text, tolerating thousands separators."""
    if isinstance(value, (int, float)):
        n = int(round(value))
        return bool(re.search(rf"(?<![\d,]){n:,}(?![\d,])|(?<![\d,]){n}(?![\d,])", text))
    return str(value).lower() in text.lower()


def required_facts(facts: dict) -> dict:
    """What a buyer must be told. Deliberately minimal and unarguable."""
    req = {
        "product": facts["product_code"],
        "stock": facts["stock_on_hand_units"],
    }
    if facts["order_placed_units"] > 0:
        req["order_qty"] = facts["order_placed_units"]
        req["supplier"] = facts["supplier_name"]
    return req


def coverage(text: str, facts: dict) -> tuple[float, list[str]]:
    req = required_facts(facts)
    missing = [k for k, v in req.items() if not _mentions(text, v)]
    return 1.0 - len(missing) / len(req), missing


def sample_decisions(n: int) -> list[dict]:
    """Real decisions from the tuned baseline, spread over days and products."""
    from src.agents.tune_baselines import make

    with resolve("results/baselines.json").open(encoding="utf-8") as fh:
        data = json.load(fh)
    best = data["best_baseline"]
    policy = make(best, data["policies"][best]["params"])

    env = SupplyChainEnv(seed=SAMPLE_SEED)
    env.reset(seed=SAMPLE_SEED)
    rng = np.random.default_rng(SAMPLE_SEED)

    samples = []
    day = 0
    while len(samples) < n and day < max(SAMPLE_DAYS):
        _, _, term, trunc, info = env.step(policy.act(env, rng))
        day += 1
        if day in SAMPLE_DAYS:
            for product in range(env.n_products):
                if len(samples) >= n:
                    break
                samples.append(build_facts(env, info, product))
        if term or trunc:
            break
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default="results/explainer_eval.json")
    args = ap.parse_args()

    client = OpenRouterClient()
    samples = sample_decisions(args.n)
    print(
        f"evaluating {len(samples)} decisions  (LLM {'on' if client.available else 'OFF'})\n",
        flush=True,
    )

    rows = []
    t0 = time.time()
    for i, facts in enumerate(samples):
        result = explain(facts, client=client)
        # Grounding is checked on the raw model text where one exists. explain()
        # already discards ungrounded output, so scoring its return value would
        # measure the guard, not the model.
        llm_text = result["text"] if result["source"] == "llm" else None
        if result["source"] == "template" and result["reason"] and "ungrounded" in result["reason"]:
            grounded = False
            reached = True
        else:
            grounded = result["source"] == "llm"
            reached = result["source"] == "llm"
        text = result["text"]
        cov, missing = coverage(text, facts)
        tpl_cov, _ = coverage(template_explanation(facts), facts)
        rows.append(
            {
                "product": facts["product_code"],
                "day": facts["day_of_episode"],
                "ordered": facts["order_placed_units"] > 0,
                "source": result["source"],
                "model": result["model"],
                "reached_llm": reached,
                "grounded": grounded,
                "coverage": cov,
                "missing": missing,
                "template_coverage": tpl_cov,
                "sentences": len(re.findall(r"[.!?](?:\s|$)", text)),
                "words": len(text.split()),
                "ungrounded": ungrounded_numbers(llm_text, facts) if llm_text else [],
                "text": text,
            }
        )
        tag = (
            "LLM "
            if result["source"] == "llm"
            else ("UNGR" if not grounded and reached else "tmpl")
        )
        print(
            f"  [{i + 1:2d}/{len(samples)}] {tag}  grounded={str(grounded):5s}  "
            f"coverage={cov:.0%}  {facts['product_code']} day {facts['day_of_episode']}",
            flush=True,
        )

    reached = [r for r in rows if r["reached_llm"]]
    llm_rows = [r for r in rows if r["source"] == "llm"]
    summary = {
        "n": len(rows),
        "llm_reachability": len(reached) / len(rows),
        "grounding_rate": (sum(r["grounded"] for r in reached) / len(reached)) if reached else None,
        "completeness_llm": (
            float(np.mean([r["coverage"] for r in llm_rows])) if llm_rows else None
        ),
        "completeness_template": float(np.mean([r["template_coverage"] for r in rows])),
        # Two accuracies, and the distinction is the finding.
        #
        # raw_model_accuracy: grounded AND complete, over every output the
        # model produced -- including the ones the guard then threw away. This
        # is how good the LLM is on its own. (An ungrounded output cannot be
        # accurate whatever else it says, so scoring it False is right.)
        #
        # delivered_accuracy: the same test on what the user actually saw,
        # after the guard replaced ungrounded output with the template. This
        # is how good the system is.
        #
        # An earlier version scored only outputs that had already passed the
        # guard and called that "explanation accuracy". It could never have
        # been below the grounding rate, and reported 100% for a model that
        # invented a number one time in six.
        "raw_model_accuracy": (
            sum(r["grounded"] and r["coverage"] == 1.0 for r in reached) / len(reached)
            if reached
            else None
        ),
        "delivered_accuracy": sum(r["coverage"] == 1.0 for r in rows) / len(rows),
        "mean_sentences": float(np.mean([r["sentences"] for r in rows])),
        "mean_words": float(np.mean([r["words"] for r in rows])),
        "wall_seconds": time.time() - t0,
    }

    print("\n" + "=" * 60)
    print(f"LLM reachable            {summary['llm_reachability']:.0%}  of {len(rows)} calls")
    if summary["grounding_rate"] is not None:
        print(f"grounding rate           {summary['grounding_rate']:.0%}  (no invented numbers)")
    if summary["completeness_llm"] is not None:
        print(
            f"completeness (LLM)       {summary['completeness_llm']:.0%}  of required facts stated"
        )
    print(f"completeness (template)  {summary['completeness_template']:.0%}")
    if summary["raw_model_accuracy"] is not None:
        print(
            f"RAW MODEL ACCURACY       {summary['raw_model_accuracy']:.0%}  "
            "grounded and complete, before the guard"
        )
    print(
        f"DELIVERED ACCURACY       {summary['delivered_accuracy']:.0%}  "
        "what the user actually saw, after the guard"
    )
    print(
        f"length                   {summary['mean_sentences']:.1f} sentences, {summary['mean_words']:.0f} words"
    )

    out = resolve(args.out)
    with out.open("w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
