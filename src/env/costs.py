"""Profit accounting for one simulated day.

The reward function is the most consequential piece of an RL project: the
agent optimises exactly what you write here, including any mistake. Five terms:

  + revenue          units actually sold x selling price
  - purchase cost    units received x unit cost x supplier multiplier
  - ordering cost    a fixed fee per supplier used that day
  - holding cost     cost of capital tied up in shelf stock
  - stockout penalty unmet demand x lost gross profit x penalty multiplier

Two design points worth defending in the report:

*Stockout penalty exceeds lost profit* (1.5x). Running out costs more than the
missed margin -- customers defect and goodwill erodes. Without this the agent
learns that stockouts are nearly free and starves inventory, which is the
classic failure mode of naive cost-minimising inventory agents.

*Holding cost is charged on the closing balance*, so stock sitting idle is
penalised every single day. This is what stops the agent from simply ordering
enormous quantities to guarantee it never runs out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DayCosts:
    revenue: float = 0.0
    purchase: float = 0.0
    ordering: float = 0.0
    holding: float = 0.0
    stockout_penalty: float = 0.0
    # Credit, not a cost: leftover stock recovered at the end of an episode.
    salvage: float = 0.0
    # Goods paid for but refused because the warehouse was full.
    overflow_loss: float = 0.0
    overflow_units: float = 0.0

    units_sold: float = 0.0
    units_short: float = 0.0
    demand_total: float = 0.0
    stockout_products: int = 0

    def profit(self) -> float:
        return (
            self.revenue
            + self.salvage
            - self.purchase
            - self.ordering
            - self.holding
            - self.stockout_penalty
            - self.overflow_loss
        )

    def as_dict(self) -> dict:
        d = {
            "revenue": self.revenue,
            "purchase": self.purchase,
            "ordering": self.ordering,
            "holding": self.holding,
            "stockout_penalty": self.stockout_penalty,
            "salvage": self.salvage,
            "overflow_loss": self.overflow_loss,
            "overflow_units": self.overflow_units,
            "profit": self.profit(),
            "units_sold": self.units_sold,
            "units_short": self.units_short,
            "demand_total": self.demand_total,
            "stockout_products": self.stockout_products,
        }
        return d


@dataclass
class CostModel:
    """Per-product economics, derived from real prices plus synthetic margins."""

    price: np.ndarray
    unit_cost: np.ndarray
    holding_rate: float
    stockout_multiplier: float
    field_names: tuple = field(default=(), repr=False)

    @classmethod
    def from_config(cls, price: np.ndarray, econ: dict) -> CostModel:
        # Unit cost is derived from the real median selling price using an
        # assumed gross margin -- the price is real, the margin is synthetic.
        unit_cost = price * (1.0 - econ["gross_margin"])
        return cls(
            price=price,
            unit_cost=unit_cost,
            holding_rate=econ["holding_cost_per_unit_day"],
            stockout_multiplier=econ["stockout_penalty_multiplier"],
        )

    def gross_profit_per_unit(self) -> np.ndarray:
        return self.price - self.unit_cost

    def compute(
        self,
        units_sold: np.ndarray,
        units_short: np.ndarray,
        units_purchased: np.ndarray,
        supplier_multiplier: np.ndarray,
        closing_stock: np.ndarray,
        ordering_cost: float,
        demand: np.ndarray,
    ) -> DayCosts:
        revenue = float(np.sum(units_sold * self.price))
        purchase = float(np.sum(units_purchased * self.unit_cost * supplier_multiplier))
        holding = float(np.sum(closing_stock * self.unit_cost * self.holding_rate))
        penalty = float(
            np.sum(units_short * self.gross_profit_per_unit() * self.stockout_multiplier)
        )

        return DayCosts(
            revenue=revenue,
            purchase=purchase,
            ordering=float(ordering_cost),
            holding=holding,
            stockout_penalty=penalty,
            units_sold=float(units_sold.sum()),
            units_short=float(units_short.sum()),
            demand_total=float(demand.sum()),
            stockout_products=int((units_short > 0).sum()),
        )
