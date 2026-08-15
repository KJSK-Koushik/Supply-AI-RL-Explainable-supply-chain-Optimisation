"""The supply chain simulator, as a Gymnasium environment.

One step = one trading day. Within a day, order of operations matters and is
fixed as follows:

    1. goods ordered previously arrive and go on the shelf
    2. today's demand is realised
    3. demand is met from stock; any shortfall is a stockout
    4. the agent's order for today is placed (arrives in future days)
    5. costs are charged and the reward is computed

Step 1 precedes step 2 because stock delivered this morning can be sold this
afternoon. Step 4 comes after demand because a real buyer places today's order
knowing what today sold. Getting this order wrong is a common source of
accidental look-ahead advantage in inventory simulators.

ACTION. For each of the 10 products the agent makes ONE joint choice from 21
options = 7 order-quantity buckets (multiples of that product's own mean daily
demand: 0, 0.5x, 1x, 2x, 3x, 5x, 8x) x 3 suppliers. MultiDiscrete of 10 values.

Expressing quantity as a multiple of each product's own mean is what lets one
policy serve a product selling 169 units/day and another selling 65 --
otherwise the network would have to learn a separate scale for every product.
Keeping quantity and supplier as ONE choice rather than two both matches the
real decision and is 2.8x faster to train; see the note in __init__.

OBSERVATION. Everything a competent human buyer would look at, normalised:
stock, in-transit, days of cover, recent demand level and volatility, trend,
backlog, calendar position, and each supplier's availability, expected lead
time and recent reliability.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import load_config
from src.env.costs import CostModel, DayCosts
from src.env.demand import DemandGenerator, load_demand_stats
from src.env.scenarios import Scenario, ScenarioManager


class SupplyChainEnv(gym.Env):
    """Multi-product, multi-supplier inventory replenishment."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        env_cfg: dict | None = None,
        supplier_cfg: dict | None = None,
        stats: dict | None = None,
        scenarios: list[Scenario] | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        from src.env.suppliers import SupplierFleet  # local import keeps module load light

        self.env_cfg = env_cfg or load_config("env")
        self.supplier_cfg = supplier_cfg or load_config("suppliers")
        self.stats = stats or load_demand_stats()

        sim = self.env_cfg["simulation"]
        self.n_products = int(sim["n_products"])
        self.episode_length = int(sim["episode_length"])
        self.warmup_days = int(sim["warmup_days"])
        self.initial_cover = float(sim["initial_stock_days_of_cover"])

        self.codes = self.stats["selected_products"][: self.n_products]
        self.descriptions = [self.stats["descriptions"][c] for c in self.codes]

        self.order_buckets = np.array(self.env_cfg["action"]["order_buckets"], dtype=np.float64)
        self.n_buckets = len(self.order_buckets)

        self._seed = seed
        self.rng = np.random.default_rng(seed)

        self.demand_gen = DemandGenerator(self.stats, self.codes, self.rng)
        self.suppliers = SupplierFleet(self.supplier_cfg, self.n_products, self.rng)
        self.n_suppliers = self.suppliers.n_suppliers

        econ = self.supplier_cfg["economics"]
        self.cost_model = CostModel.from_config(self.demand_gen.price, econ)
        self.capacity_total = float(econ["warehouse_capacity_total"])
        self.overflow_written_off = bool(econ.get("overflow_is_written_off", True))
        self.backlog_enabled = bool(econ["backlog_enabled"])
        self.backlog_carryover = float(econ["backlog_carryover_fraction"])
        self.salvage_fraction = float(econ.get("terminal_salvage_fraction", 0.0))

        self.scenario_mgr = ScenarioManager(self.n_products, self.n_suppliers)
        self._initial_scenarios = scenarios or []

        self.reward_scale = float(self.env_cfg["reward"]["scale"])
        self.reward_clip = self.env_cfg["reward"]["clip"]

        # One JOINT choice per product: which (quantity bucket, supplier) pair.
        #
        # The obvious encoding is two separate choices per product, i.e.
        # MultiDiscrete([n_buckets, n_suppliers] * n_products) = 20 dimensions.
        # That was measured to be 2.8x slower: PPO builds one Categorical
        # distribution per dimension on every forward pass, so 20 dimensions
        # means 20 distributions per step, each with its own validation.
        # Collapsing to 10 joint choices halves that work.
        #
        # It is also the better model. How much to order and who to order it
        # from are not independent -- ordering 8x mean demand only makes sense
        # from a supplier who can actually deliver it -- so a joint choice
        # matches the real decision.
        self.n_joint_actions = self.n_buckets * self.n_suppliers
        self.action_space = spaces.MultiDiscrete([self.n_joint_actions] * self.n_products)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self._obs_size(),), dtype=np.float32
        )

        self.reset(seed=seed)

    # ------------------------------------------------------------ observation

    def _obs_size(self) -> int:
        per_product = 7  # stock, in-transit, cover, mean7, std7, trend, backlog
        calendar = 7 + 2  # weekday one-hot + month sin/cos
        per_supplier = 3  # available, expected lead time, recent fill
        return self.n_products * per_product + calendar + self.n_suppliers * per_supplier

    def _build_obs(self) -> np.ndarray:
        mean = np.maximum(self.demand_gen.mean, 1e-6)
        hist = (
            np.array(self.demand_history[-14:])
            if self.demand_history
            else np.zeros((1, self.n_products))
        )

        recent7 = hist[-7:].mean(axis=0) if len(hist) else np.zeros(self.n_products)
        std7 = hist[-7:].std(axis=0) if len(hist) >= 2 else np.zeros(self.n_products)
        if len(hist) >= 14:
            trend = hist[-7:].mean(axis=0) - hist[-14:-7].mean(axis=0)
        else:
            trend = np.zeros(self.n_products)

        in_transit = self.suppliers.in_transit()
        # Days of cover is the single most decision-relevant number a buyer
        # looks at: how long current stock lasts at the current sales rate.
        cover = (self.stock + in_transit) / mean

        parts = [
            self.stock / mean,
            in_transit / mean,
            np.clip(cover / 10.0, 0, 10),
            recent7 / mean,
            std7 / mean,
            trend / mean,
            self.backlog / mean,
        ]

        dow_onehot = np.zeros(7)
        dow_onehot[self.current_dow] = 1.0
        month_angle = 2 * np.pi * (self.current_month - 1) / 12.0
        calendar = np.concatenate([dow_onehot, [np.sin(month_angle), np.cos(month_angle)]])

        sup = np.concatenate(
            [
                self.suppliers.is_available().astype(np.float64),
                self.suppliers.expected_lead_time() / 10.0,
                self.suppliers.recent_fill,
            ]
        )

        obs = np.concatenate([*parts, calendar, sup])
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    # ------------------------------------------------------------------ reset

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self.rng = np.random.default_rng(seed)
            self.demand_gen.rng = self.rng
            self.suppliers.rng = self.rng

        self.day = 0
        self.suppliers.reset()
        self.scenario_mgr.clear()
        for sc in self._initial_scenarios:
            self.scenario_mgr.add(sc)

        # Start with a realistic shelf rather than an empty warehouse, so the
        # agent is not punished for an opening position it never chose.
        self.stock = self.demand_gen.mean * self.initial_cover
        self.backlog = np.zeros(self.n_products)
        self.demand_history: list[np.ndarray] = []

        start_dow = 0
        self.current_dow = start_dow
        self.current_month = 1
        self.cumulative = DayCosts()
        self.last_info: dict[str, Any] = {}

        return self._build_obs(), {}

    # ------------------------------------------------------------------- step

    def _advance_calendar(self) -> None:
        self.current_dow = (self.current_dow + 1) % 7
        # Saturday is a non-trading day in the source data, so skip it.
        if self.current_dow in self.stats["closed_weekdays"]:
            self.current_dow = (self.current_dow + 1) % 7
        # 30-day months are close enough for a 180-day episode and keep the
        # seasonal factor moving through the year.
        if self.day > 0 and self.day % 30 == 0:
            self.current_month = self.current_month % 12 + 1

    def decode_action(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split each joint choice back into a quantity and a supplier."""
        a = np.clip(np.asarray(action).ravel(), 0, self.n_joint_actions - 1).astype(int)
        bucket_idx = a // self.n_suppliers
        supplier = a % self.n_suppliers
        quantities = self.order_buckets[bucket_idx] * self.demand_gen.mean
        return np.floor(quantities), supplier

    def action_masks(self) -> np.ndarray:
        """Which joint actions are currently legal, for MaskablePPO.

        The first 1M-step run lost 25,308 to overflow -- goods ordered into a
        full warehouse, paid for, then refused at the door -- and a further
        13,671 to ordering fees, by spreading orders across all three suppliers
        nearly every day. Both are *impossible* or *pointless* actions the
        environment already knows about at decision time, so making them
        unavailable is strictly better than making the agent spend capacity
        discovering they are bad from a reward signal that arrives six days
        later, buried among nine other products' outcomes.

        Two rules:
          1. a supplier that is offline cannot be ordered from at all
          2. an order larger than the warehouse can physically accept is
             disallowed

        Returns a flat boolean array of shape (n_products * n_joint_actions,),
        which is the layout sb3-contrib expects for a MultiDiscrete space.
        """
        available = self.suppliers.is_available()
        headroom = max(
            self.capacity_total - self.stock.sum() - self.suppliers.in_transit().sum(), 0.0
        )

        # quantity[p, b] = units product p would order at bucket b
        qty = np.floor(self.order_buckets[None, :] * self.demand_gen.mean[:, None])

        bucket_of = np.arange(self.n_joint_actions) // self.n_suppliers
        supplier_of = np.arange(self.n_joint_actions) % self.n_suppliers

        mask = np.ones((self.n_products, self.n_joint_actions), dtype=bool)
        mask &= available[supplier_of][None, :]
        mask &= qty[:, bucket_of] <= headroom

        # The zero-order action must always survive, or a full warehouse or a
        # total outage would leave the agent with no legal move at all.
        zero_actions = bucket_of == 0
        mask[:, zero_actions] = True
        return mask.ravel()

    def encode_action(self, buckets: np.ndarray, suppliers: np.ndarray) -> np.ndarray:
        """Inverse of decode_action, so rule-based policies can emit the same
        joint encoding the agent uses."""
        return np.asarray(buckets, dtype=int) * self.n_suppliers + np.asarray(suppliers, dtype=int)

    def step(self, action):
        self.day += 1
        self._advance_calendar()
        day = self.day

        # Scenario effects for today.
        self.suppliers.extra_lead_days = self.scenario_mgr.supplier_extra_lead(day)
        for sup_id, dur in self.scenario_mgr.outages_starting(day):
            self.suppliers.force_outage(sup_id, dur)
        self.suppliers.step_outages()

        # 1. receive deliveries, subject to shared warehouse space
        arrivals = self.suppliers.receive()
        free_space = max(self.capacity_total - self.stock.sum(), 0.0)
        incoming = arrivals.sum()
        overflow_units = 0.0
        if incoming > free_space:
            # Not everything fits. Accept proportionally across products and
            # refuse the rest at the door -- already paid for, and lost.
            accept_ratio = free_space / incoming if incoming > 0 else 0.0
            accepted = arrivals * accept_ratio
            overflow_units = float(incoming - accepted.sum())
            arrivals = accepted
        self.stock = self.stock + arrivals

        # 2. realise demand (plus any backlog carried from yesterday)
        mult = self.scenario_mgr.demand_multiplier(day)
        demand = self.demand_gen.sample(self.current_dow, self.current_month, mult)
        effective_demand = demand + self.backlog

        # 3. serve what we can
        units_sold = np.minimum(self.stock, effective_demand)
        units_short = effective_demand - units_sold
        self.stock = self.stock - units_sold

        if self.backlog_enabled:
            self.backlog = units_short * self.backlog_carryover
        else:
            self.backlog = np.zeros(self.n_products)

        # 4. place today's order
        quantities, supplier_choice = self.decode_action(action)
        quantities = self.suppliers.apply_moq(quantities, supplier_choice)
        # Cap orders at the space physically free *today*. Deliberately NOT
        # net of in-transit stock: a buyer can over-order by forgetting what is
        # already on the water, several days of orders then land together, and
        # the excess is refused at the door. Subtracting in-transit here would
        # make that mistake impossible and remove the main thing the agent has
        # to learn about pipeline management.
        headroom = max(self.capacity_total - self.stock.sum(), 0.0)
        requested = quantities.sum()
        if requested > headroom:
            quantities = (
                np.floor(quantities * (headroom / requested)) if requested > 0 else quantities
            )
        order = self.suppliers.place_orders(quantities, supplier_choice)

        # 5. cost it
        cost_mult = self.scenario_mgr.cost_multiplier(day)
        sup_mult = self.suppliers.cost_multiplier[supplier_choice] * cost_mult
        costs = self.cost_model.compute(
            units_sold=units_sold,
            units_short=units_short,
            units_purchased=order["purchase_units"],
            supplier_multiplier=sup_mult,
            closing_stock=self.stock,
            ordering_cost=order["ordering_cost"],
            demand=demand,
        )
        if overflow_units > 0 and self.overflow_written_off:
            # Charge at average unit cost across products; the overflow was
            # split proportionally, so an average is the faithful valuation.
            avg_cost = float(self.cost_model.unit_cost.mean())
            costs.overflow_loss = overflow_units * avg_cost
            costs.overflow_units = overflow_units

        terminated = False
        truncated = day >= self.episode_length

        # On the last day, credit back unsold stock. Otherwise a 180-day
        # horizon writes off every leftover unit, which is an artefact of where
        # we stopped counting and teaches the agent to stop ordering near the
        # end of an episode -- behaviour that would not transfer to a real
        # rolling operation.
        if truncated and self.salvage_fraction > 0:
            costs.salvage = float(
                np.sum(self.stock * self.cost_model.unit_cost * self.salvage_fraction)
            )

        self.demand_history.append(demand)
        if day > self.warmup_days:
            self._accumulate(costs)

        profit = costs.profit()
        # Warm-up days still simulate but do not pay, so the agent is not
        # rewarded or punished for the arbitrary opening inventory position.
        reward = 0.0 if day <= self.warmup_days else profit * self.reward_scale
        reward = float(np.clip(reward, self.reward_clip[0], self.reward_clip[1]))

        info = {
            "day": day,
            "profit": profit,
            "costs": costs.as_dict(),
            "demand": demand.copy(),
            "units_sold": units_sold.copy(),
            "units_short": units_short.copy(),
            "stock": self.stock.copy(),
            "in_transit": self.suppliers.in_transit(),
            "backlog": self.backlog.copy(),
            "order_quantities": quantities.copy(),
            "supplier_choice": supplier_choice.copy(),
            "order_rejected": order["rejected"].copy(),
            "supplier_available": self.suppliers.is_available().copy(),
            "fill_rate": np.divide(
                order["shipped"],
                np.maximum(quantities, 1e-9),
                out=np.ones(self.n_products),
                where=quantities > 0,
            ),
            "overflow_units": overflow_units,
            "warehouse_utilisation": float(self.stock.sum() / self.capacity_total),
            "active_scenarios": self.scenario_mgr.active_descriptions(day),
            "service_level": float(
                units_sold.sum() / effective_demand.sum() if effective_demand.sum() > 0 else 1.0
            ),
        }
        self.last_info = info
        return self._build_obs(), reward, terminated, truncated, info

    def _accumulate(self, c: DayCosts) -> None:
        t = self.cumulative
        t.revenue += c.revenue
        t.purchase += c.purchase
        t.ordering += c.ordering
        t.holding += c.holding
        t.stockout_penalty += c.stockout_penalty
        t.salvage += c.salvage
        t.overflow_loss += c.overflow_loss
        t.overflow_units += c.overflow_units
        t.units_sold += c.units_sold
        t.units_short += c.units_short
        t.demand_total += c.demand_total
        t.stockout_products += c.stockout_products

    # ---------------------------------------------------------------- helpers

    def episode_summary(self) -> dict:
        """Metrics used by the Phase 5 comparison, excluding warm-up."""
        t = self.cumulative
        served = t.demand_total
        return {
            "total_profit": t.profit(),
            "revenue": t.revenue,
            "purchase_cost": t.purchase,
            "ordering_cost": t.ordering,
            "holding_cost": t.holding,
            "stockout_penalty": t.stockout_penalty,
            "salvage": t.salvage,
            "overflow_loss": t.overflow_loss,
            "overflow_units": t.overflow_units,
            "fill_rate": (t.units_sold / served) if served > 0 else 1.0,
            "units_short": t.units_short,
            "stockout_product_days": t.stockout_products,
        }

    def add_scenario(self, scenario: Scenario) -> None:
        self.scenario_mgr.add(scenario)

    def render(self):
        i = self.last_info
        if not i:
            return
        print(
            f"day {i['day']:3d} | profit {i['profit']:9.2f} | "
            f"service {i['service_level']:.1%} | stock {i['stock'].sum():8.0f} | "
            f"short {i['units_short'].sum():6.0f}"
        )
