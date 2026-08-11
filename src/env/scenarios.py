"""Disruption scenarios injected into a running simulation.

This module is deliberately built now, in Phase 2, rather than bolted on in
Phase 7. Phase 7 has the LLM *generate* scenarios and Phase 9 uses them as
training curriculum, so the injection mechanism has to be a first-class part of
the environment rather than a wrapper hacked around it.

A Scenario is plain validated data, never executable content. The LLM in
Phase 7 will emit JSON that is parsed into exactly this structure, checked
against the schema, and clamped to safe ranges before it can touch the
simulator. Nothing the LLM produces is ever trusted directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ScenarioType = Literal["demand_spike", "supplier_delay", "supplier_outage", "price_shock"]


@dataclass
class Scenario:
    """One disruption, active over a window of days."""

    type: ScenarioType
    start_day: int
    duration: int
    # demand_spike
    products: list[int] | None = None
    multiplier: float = 1.0
    # supplier_delay / supplier_outage
    supplier: int | None = None
    extra_lead_days: int = 0
    # price_shock
    cost_multiplier: float = 1.0
    label: str = ""

    def active_on(self, day: int) -> bool:
        return self.start_day <= day < self.start_day + self.duration

    def describe(self) -> str:
        """Plain-English summary, used by the dashboard and the explainer."""
        window = f"days {self.start_day}-{self.start_day + self.duration - 1}"
        if self.type == "demand_spike":
            who = "all products" if not self.products else f"products {self.products}"
            return f"Demand for {who} is {self.multiplier:.1f}x normal over {window}"
        if self.type == "supplier_delay":
            return f"Supplier {self.supplier} takes {self.extra_lead_days} extra days over {window}"
        if self.type == "supplier_outage":
            return f"Supplier {self.supplier} is offline for {window}"
        if self.type == "price_shock":
            return f"Purchase costs are {self.cost_multiplier:.2f}x over {window}"
        return f"{self.type} over {window}"


class ScenarioManager:
    """Holds active scenarios and reports their effect for a given day."""

    def __init__(self, n_products: int, n_suppliers: int):
        self.n_products = n_products
        self.n_suppliers = n_suppliers
        self.scenarios: list[Scenario] = []
        self._applied_outages: set[int] = set()

    def add(self, scenario: Scenario) -> None:
        self.scenarios.append(scenario)

    def clear(self) -> None:
        self.scenarios = []
        self._applied_outages = set()

    def demand_multiplier(self, day: int) -> np.ndarray:
        """Per-product demand multiplier for this day. Overlapping spikes
        compound, which is intentional -- a festival during a promotion really
        does stack."""
        mult = np.ones(self.n_products)
        for sc in self.scenarios:
            if sc.type == "demand_spike" and sc.active_on(day):
                if sc.products:
                    for p in sc.products:
                        if 0 <= p < self.n_products:
                            mult[p] *= sc.multiplier
                else:
                    mult *= sc.multiplier
        return mult

    def supplier_extra_lead(self, day: int) -> np.ndarray:
        extra = np.zeros(self.n_suppliers, dtype=int)
        for sc in self.scenarios:
            if sc.type == "supplier_delay" and sc.active_on(day) and sc.supplier is not None:
                if 0 <= sc.supplier < self.n_suppliers:
                    extra[sc.supplier] += sc.extra_lead_days
        return extra

    def cost_multiplier(self, day: int) -> float:
        mult = 1.0
        for sc in self.scenarios:
            if sc.type == "price_shock" and sc.active_on(day):
                mult *= sc.cost_multiplier
        return mult

    def outages_starting(self, day: int) -> list[tuple[int, int]]:
        """Outages that begin today, as (supplier, days). Each fires once."""
        out = []
        for i, sc in enumerate(self.scenarios):
            if (
                sc.type == "supplier_outage"
                and sc.start_day == day
                and sc.supplier is not None
                and i not in self._applied_outages
            ):
                out.append((sc.supplier, sc.duration))
                self._applied_outages.add(i)
        return out

    def active_descriptions(self, day: int) -> list[str]:
        return [sc.describe() for sc in self.scenarios if sc.active_on(day)]


# ----------------------------------------------------------------- presets


def demand_spike(products, multiplier, start_day, duration, label="") -> Scenario:
    return Scenario(
        type="demand_spike",
        products=list(products) if products else None,
        multiplier=float(multiplier),
        start_day=int(start_day),
        duration=int(duration),
        label=label or "demand spike",
    )


def supplier_delay(supplier, extra_days, start_day, duration, label="") -> Scenario:
    return Scenario(
        type="supplier_delay",
        supplier=int(supplier),
        extra_lead_days=int(extra_days),
        start_day=int(start_day),
        duration=int(duration),
        label=label or "supplier delay",
    )


def supplier_outage(supplier, start_day, duration, label="") -> Scenario:
    return Scenario(
        type="supplier_outage",
        supplier=int(supplier),
        start_day=int(start_day),
        duration=int(duration),
        label=label or "supplier outage",
    )


def price_shock(cost_multiplier, start_day, duration, label="") -> Scenario:
    return Scenario(
        type="price_shock",
        cost_multiplier=float(cost_multiplier),
        start_day=int(start_day),
        duration=int(duration),
        label=label or "price shock",
    )
