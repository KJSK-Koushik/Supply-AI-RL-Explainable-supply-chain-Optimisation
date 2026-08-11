"""Supplier behaviour: lead times, partial deliveries and outages.

*** All supplier parameters are synthetic (configs/suppliers.yaml). ***
No public retail dataset contains lead times or fill rates.

Three suppliers span the classic trade-off:
  0 EconoSource   cheap  (1.00x) but slow   (5-8 days), 90% fill
  1 RapidTrade    fast   (1-3 days) but dear (1.35x),   98% fill
  2 MidWay Supply middling (1.15x, 3-5 days) but flaky, 80% fill, outage-prone

None dominates. Choosing well depends on how much stock you hold, how volatile
the product is, and whether a supplier is currently down -- which is exactly
the judgement we want the RL agent to learn.

Orders are tracked in a *pipeline*: goods ordered today arrive several days
later. Inventory already ordered but not yet delivered is a real quantity a
replenishment policy must reason about, or it will double-order while a
shipment is still in transit.
"""

from __future__ import annotations

import numpy as np

from src.config import load_config


class SupplierFleet:
    """Simulates all suppliers and the in-transit order pipeline."""

    def __init__(self, cfg: dict, n_products: int, rng: np.random.Generator, max_lead: int = 25):
        self.rng = rng
        self.n_products = n_products
        sup = cfg["suppliers"]
        self.n_suppliers = len(sup)
        self.names = [s["name"] for s in sup]

        self.cost_multiplier = np.array([s["cost_multiplier"] for s in sup])
        self.lt_min = np.array([s["lead_time_min"] for s in sup])
        self.lt_max = np.array([s["lead_time_max"] for s in sup])
        self.lt_mode = np.array([s["lead_time_mode"] for s in sup], dtype=float)
        self.fill_mean = np.array([s["fill_rate_mean"] for s in sup])
        self.fill_std = np.array([s["fill_rate_std"] for s in sup])
        self.moq = np.array([s["min_order_qty"] for s in sup])
        self.fixed_cost = np.array([s["fixed_order_cost"] for s in sup])
        self.outage_prob = np.array([s["outage_prob"] for s in sup])
        self.outage_range = [s["outage_duration_range"] for s in sup]

        # Pipeline[d, p] = units of product p arriving in d days' time.
        self.max_lead = max_lead
        self.pipeline = np.zeros((max_lead + 1, n_products))

        self.outage_days_left = np.zeros(self.n_suppliers, dtype=int)
        # Scenario-injected extra delay, added on top of the natural lead time.
        self.extra_lead_days = np.zeros(self.n_suppliers, dtype=int)
        # Rolling record of realised fill rates, so the agent can observe that
        # a supplier has been under-delivering recently.
        self.recent_fill = self.fill_mean.copy()

    # ---------------------------------------------------------------- state

    def is_available(self) -> np.ndarray:
        return self.outage_days_left == 0

    def expected_lead_time(self) -> np.ndarray:
        """Mean of the triangular distribution, plus any scenario delay."""
        return (self.lt_min + self.lt_mode + self.lt_max) / 3.0 + self.extra_lead_days

    # ------------------------------------------------------------ mechanics

    def _sample_lead_time(self, s: int) -> int:
        lt = self.rng.triangular(self.lt_min[s], self.lt_mode[s], self.lt_max[s])
        lt = int(round(lt)) + int(self.extra_lead_days[s])
        return int(np.clip(lt, 1, self.max_lead))

    def _sample_fill_rate(self, s: int) -> float:
        f = self.rng.normal(self.fill_mean[s], self.fill_std[s])
        return float(np.clip(f, 0.0, 1.0))

    def place_orders(self, quantities: np.ndarray, supplier_choice: np.ndarray) -> dict:
        """Place one day's orders.

        quantities[p]      units requested of product p (already MOQ-adjusted)
        supplier_choice[p] which supplier to use for product p

        Returns per-product realised quantities and the costs incurred. An
        order to an unavailable supplier is rejected outright: nothing ships,
        and no cost is charged. The agent has to notice the outage flag in its
        observation and route around it.
        """
        shipped = np.zeros(self.n_products)
        arrival_day = np.zeros(self.n_products, dtype=int)
        rejected = np.zeros(self.n_products, dtype=bool)
        purchase_units = np.zeros(self.n_products)
        ordering_cost = 0.0

        # One fixed ordering cost per supplier used today, not per product --
        # this is what creates an incentive to consolidate orders.
        suppliers_used = set()

        for p in range(self.n_products):
            q = quantities[p]
            if q <= 0:
                continue
            s = int(supplier_choice[p])

            if self.outage_days_left[s] > 0:
                rejected[p] = True
                continue

            lead = self._sample_lead_time(s)
            fill = self._sample_fill_rate(s)
            got = np.floor(q * fill)

            self.pipeline[lead, p] += got
            shipped[p] = got
            arrival_day[p] = lead
            # You pay for what arrives, not what you asked for.
            purchase_units[p] = got
            suppliers_used.add(s)

        for s in suppliers_used:
            ordering_cost += float(self.fixed_cost[s])

        return {
            "shipped": shipped,
            "arrival_in_days": arrival_day,
            "rejected": rejected,
            "purchase_units": purchase_units,
            "ordering_cost": ordering_cost,
            "suppliers_used": suppliers_used,
        }

    def receive(self) -> np.ndarray:
        """Advance one day and return the goods arriving today."""
        arriving = self.pipeline[0].copy()
        self.pipeline = np.roll(self.pipeline, -1, axis=0)
        self.pipeline[-1] = 0.0
        return arriving

    def step_outages(self) -> None:
        """Age existing outages and randomly start new ones."""
        self.outage_days_left = np.maximum(self.outage_days_left - 1, 0)
        for s in range(self.n_suppliers):
            if self.outage_days_left[s] == 0 and self.rng.random() < self.outage_prob[s]:
                lo, hi = self.outage_range[s]
                self.outage_days_left[s] = int(self.rng.integers(lo, hi + 1))

    def force_outage(self, supplier: int, days: int) -> None:
        """Used by the scenario injector."""
        self.outage_days_left[supplier] = max(self.outage_days_left[supplier], int(days))

    def in_transit(self) -> np.ndarray:
        """Total units per product currently on the water."""
        return self.pipeline.sum(axis=0)

    def apply_moq(self, quantities: np.ndarray, supplier_choice: np.ndarray) -> np.ndarray:
        """Round each non-zero order up to that supplier's minimum order
        quantity. A supplier who will not ship fewer than 100 units forces a
        real decision: take more stock than you want, or switch supplier."""
        q = quantities.copy()
        for p in range(self.n_products):
            if q[p] > 0:
                s = int(supplier_choice[p])
                q[p] = max(q[p], self.moq[s])
        return q

    def reset(self) -> None:
        self.pipeline[:] = 0.0
        self.outage_days_left[:] = 0
        self.extra_lead_days[:] = 0
        self.recent_fill = self.fill_mean.copy()


def load_supplier_config(cfg: dict | None = None) -> dict:
    return cfg or load_config("suppliers")
