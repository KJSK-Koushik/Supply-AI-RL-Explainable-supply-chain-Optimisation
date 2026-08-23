"""Turn one ordering decision into business English.

The design rule for this module: the LLM is never asked *why* a decision was
made. It is given facts the simulator already computed and asked to phrase
them. The reason is that the agent's actual reason is a 20,000-parameter
policy network -- any "because" a language model supplies is a plausible story,
not the cause, and presenting it as the cause would make the explainability
claim of this project false.

So the model does presentation, and the numbers stay the simulator's. A
post-check enforces that: every number in the output must trace back to a
supplied fact, or the output is discarded and the deterministic template is
used instead.
"""

from __future__ import annotations

import re

from src.config import load_config
from src.llm.client import OpenRouterClient

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

SYSTEM_PROMPT = (
    "You are an inventory analyst writing one short note for a retail buyer. "
    "You will be given facts about a single ordering decision that has already "
    "been made. Restate those facts as two or three plain sentences of business "
    "English.\n"
    "Rules:\n"
    "- Use ONLY the numbers given. Never introduce, estimate or round a number "
    "that is not in the facts.\n"
    "- Do not speculate about causes beyond what the facts state.\n"
    "- No bullet points, no headings, no preamble. Just the sentences.\n"
    "- Write for a busy person: concrete, calm, no marketing language."
)


def build_facts(env, info: dict, product: int) -> dict:
    """Everything the explainer is allowed to talk about, for one product."""
    code = env.codes[product]
    mean = float(env.demand_gen.mean[product])
    stock = float(info["stock"][product])
    in_transit = float(info["in_transit"][product])
    ordered = float(info["order_quantities"][product])
    supplier_id = int(info["supplier_choice"][product])
    sup = env.suppliers

    recent = [float(d[product]) for d in env.demand_history[-7:]]
    recent_mean = sum(recent) / len(recent) if recent else mean
    if recent_mean > mean * 1.15:
        trend = "above its usual level"
    elif recent_mean < mean * 0.85:
        trend = "below its usual level"
    else:
        trend = "close to its usual level"

    facts = {
        "product_code": code,
        "product_name": env.descriptions[product].title().strip(),
        "day_of_episode": int(info["day"]),
        "weekday": WEEKDAYS[env.current_dow],
        "month": MONTHS[env.current_month - 1],
        "stock_on_hand_units": round(stock),
        "already_on_order_units": round(in_transit),
        "days_of_cover": round((stock + in_transit) / max(mean, 1e-6), 1),
        "typical_daily_demand_units": round(mean),
        "last_7_day_average_units": round(recent_mean),
        "recent_demand_note": trend,
        "unmet_demand_yesterday_units": round(float(info["units_short"][product])),
        "order_placed_units": round(ordered),
    }
    if ordered > 0:
        facts.update(
            {
                "supplier_name": sup.names[supplier_id],
                "supplier_lead_time_days_min": int(sup.lt_min[supplier_id]),
                "supplier_lead_time_days_max": int(sup.lt_max[supplier_id]),
                "supplier_typical_fill_rate_percent": round(
                    float(sup.fill_mean[supplier_id]) * 100
                ),
            }
        )
    return facts


def _fact_numbers(facts: dict) -> list[float]:
    out: list[float] = []
    for value in facts.values():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out.append(float(value))
        elif isinstance(value, str):
            out.extend(float(n) for n in re.findall(r"\d+(?:\.\d+)?", value))
    return out


def ungrounded_numbers(text: str, facts: dict) -> list[str]:
    """Numbers in the text that do not trace back to a supplied fact.

    Tolerant about formatting -- "1,200" and "1200" are the same number, and a
    value rounded to one decimal still matches -- but not about provenance. A
    figure the simulator never produced is exactly what this is here to catch.
    """
    allowed = _fact_numbers(facts)
    bad = []
    for token in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        value = float(token.replace(",", ""))
        tolerance = max(0.5, abs(value) * 0.01)
        if not any(abs(value - a) <= tolerance for a in allowed):
            bad.append(token)
    return bad


def template_explanation(facts: dict) -> str:
    """Deterministic fallback. Never fails, never invents anything."""
    head = (
        f"{facts['product_name']} ({facts['product_code']}): "
        f"{facts['stock_on_hand_units']:,} units in stock with "
        f"{facts['already_on_order_units']:,} already on order, "
        f"about {facts['days_of_cover']} days of cover against typical demand of "
        f"{facts['typical_daily_demand_units']:,} units a day."
    )
    middle = (
        f"Demand over the last seven days averaged "
        f"{facts['last_7_day_average_units']:,} units a day, {facts['recent_demand_note']}."
    )
    if facts["unmet_demand_yesterday_units"] > 0:
        middle += (
            f" Yesterday {facts['unmet_demand_yesterday_units']:,} units of demand went unmet."
        )
    if facts["order_placed_units"] > 0:
        tail = (
            f"Ordered {facts['order_placed_units']:,} units from "
            f"{facts['supplier_name']}, which typically delivers in "
            f"{facts['supplier_lead_time_days_min']} to "
            f"{facts['supplier_lead_time_days_max']} days and ships about "
            f"{facts['supplier_typical_fill_rate_percent']}% of what is ordered."
        )
    else:
        tail = "No order placed today; existing cover is sufficient."
    return f"{head} {middle} {tail}"


def explain(facts: dict, client: OpenRouterClient | None = None, cfg: dict | None = None) -> dict:
    """Explain one decision. Always returns something usable.

    Returns the text plus how it was produced, because "which model wrote this,
    and was it checked" is part of the result when the answer is prose.
    """
    cfg = cfg or load_config("llm")
    settings = cfg["explainer"]
    fallback = template_explanation(facts)

    client = client or OpenRouterClient(cfg)
    if not client.available:
        return {"text": fallback, "source": "template", "reason": "no API key", "model": None}

    lines = "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in facts.items())
    result = client.complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Facts about today's decision:\n{lines}"},
        ],
        temperature=float(settings.get("temperature", 0.2)),
        max_tokens=int(settings.get("max_tokens", 300)),
    )
    if not result.ok:
        return {
            "text": fallback,
            "source": "template",
            "reason": "; ".join(result.attempts[-3:]) or "no response",
            "model": None,
        }

    if settings.get("enforce_number_grounding", True):
        invented = ungrounded_numbers(result.text, facts)
        if invented:
            return {
                "text": fallback,
                "source": "template",
                "reason": f"ungrounded numbers in model output: {invented}",
                "model": result.model,
            }

    return {"text": result.text, "source": "llm", "reason": None, "model": result.model}
