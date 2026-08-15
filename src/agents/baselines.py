"""Classical inventory policies -- the comparison floor for the RL agent.

These are the rules real inventory systems actually use. If the RL agent cannot
beat a *properly tuned* version of these, the project has no result. Tuning
them honestly is therefore not a favour to the baselines; it is what makes the
eventual RL number mean anything.

Every policy here:

  * sees the same observation the agent sees (stock, in-transit, recent demand)
  * emits the same discrete action the agent must emit (order bucket +
    supplier), so both sides face identical action granularity
  * is parameterised in DAYS OF COVER rather than units, so one parameter set
    works across a product selling 169/day and one selling 65/day

The last point matters for fairness. If baselines were tuned per product with
free-form quantities while the agent was restricted to 7 buckets, any RL win
would just be an artefact of the action space.

A note on what these policies structurally cannot do: each one reasons about
one product at a time. None can see that the warehouse is shared, so none can
decide to hold back on cake cases because gliders need the space. That is the
coordination gap the RL agent is meant to exploit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


class Policy:
    """Base class. Subclasses implement `desired_quantity`."""

    name = "policy"

    def reset(self) -> None:
        pass

    def inventory_position(self, env) -> np.ndarray:
        """Stock on hand plus stock already on the water. Ordering against
        on-hand alone is the classic double-ordering mistake."""
        return env.stock + env.suppliers.in_transit()

    def recent_demand(self, env, window: int = 7) -> np.ndarray:
        if not env.demand_history:
            return env.demand_gen.mean.copy()
        hist = np.array(env.demand_history[-window:])
        return hist.mean(axis=0)

    def desired_quantity(self, env) -> np.ndarray:
        raise NotImplementedError

    def choose_supplier(self, env) -> np.ndarray:
        raise NotImplementedError

    def act(self, env, rng=None) -> np.ndarray:
        supplier = self.choose_supplier(env)
        want = self.desired_quantity(env)
        buckets = self.quantise(env, want)
        # Joint (bucket, supplier) encoding -- identical to what the agent
        # emits, so neither side gets a finer action space than the other.
        return env.encode_action(buckets, supplier)

    @staticmethod
    def quantise(env, want_units: np.ndarray) -> np.ndarray:
        """Snap a desired quantity in units to the nearest available bucket.

        The agent faces exactly this constraint, so applying it to the
        baselines too keeps the comparison honest.
        """
        mean = np.maximum(env.demand_gen.mean, 1e-6)
        want_multiples = np.maximum(want_units, 0.0) / mean
        # |buckets - want| for every product x bucket, then argmin per product.
        diff = np.abs(env.order_buckets[None, :] - want_multiples[:, None])
        return diff.argmin(axis=1).astype(int)


# --------------------------------------------------------------- supplier rules


@dataclass
class FixedSupplier:
    """Always use one supplier; fall back to any available one during an
    outage, because refusing to order at all would be indefensibly bad."""

    supplier: int

    def __call__(self, env) -> np.ndarray:
        s = self.supplier
        if not env.suppliers.is_available()[s]:
            available = np.flatnonzero(env.suppliers.is_available())
            if len(available):
                # Cheapest available stand-in.
                s = int(available[np.argmin(env.suppliers.cost_multiplier[available])])
        return np.full(env.n_products, s, dtype=int)


@dataclass
class UrgencySupplier:
    """Use the cheap supplier normally, the fast one when cover is critical.

    This is what a competent human buyer does, and it is the strongest
    supplier heuristic available to a per-product rule.
    """

    urgent_days: float = 4.0
    normal_supplier: int = 0
    urgent_supplier: int = 1

    def __call__(self, env) -> np.ndarray:
        mean = np.maximum(env.demand_gen.mean, 1e-6)
        cover = (env.stock + env.suppliers.in_transit()) / mean
        choice = np.where(cover < self.urgent_days, self.urgent_supplier, self.normal_supplier)

        avail = env.suppliers.is_available()
        if not avail.all():
            fallback = np.flatnonzero(avail)
            if len(fallback):
                cheap = int(fallback[np.argmin(env.suppliers.cost_multiplier[fallback])])
                choice = np.where(avail[choice], choice, cheap)
        return choice.astype(int)


# ------------------------------------------------------------------- policies


class RandomPolicy(Policy):
    """Sanity floor. Anything that cannot beat this is broken."""

    name = "random"

    def act(self, env, rng=None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        return np.array([rng.integers(0, n) for n in env.action_space.nvec])


class ConstantPolicy(Policy):
    """Order a fixed multiple of mean daily demand, every day."""

    name = "constant"

    def __init__(self, bucket: int = 2, supplier: int = 1):
        self.bucket = bucket
        self.supplier_rule = FixedSupplier(supplier)

    def choose_supplier(self, env):
        return self.supplier_rule(env)

    def act(self, env, rng=None) -> np.ndarray:
        supplier = self.choose_supplier(env)
        buckets = np.full(env.n_products, self.bucket, dtype=int)
        return env.encode_action(buckets, supplier)


class SSPolicy(Policy):
    """(s, S): when inventory position falls below s, top up to S.

    The textbook policy, and the one real ERP systems implement. Both levels
    are in days of cover. s must exceed the supplier lead time or the rule is
    structurally behind -- goods ordered at the reorder point do not arrive
    for several days, during which demand continues.
    """

    name = "(s,S)"

    def __init__(self, s_days: float, S_days: float, supplier_rule):
        self.s_days = s_days
        self.S_days = max(S_days, s_days + 0.5)
        self.supplier_rule = supplier_rule

    def choose_supplier(self, env):
        return self.supplier_rule(env)

    def desired_quantity(self, env) -> np.ndarray:
        rate = np.maximum(self.recent_demand(env), 1e-6)
        position = self.inventory_position(env)
        cover = position / rate
        target = self.S_days * rate
        return np.where(cover < self.s_days, np.maximum(target - position, 0.0), 0.0)


class EOQPolicy(Policy):
    """Economic Order Quantity with a reorder point.

    EOQ = sqrt(2 D K / h) balances fixed ordering cost against holding cost.
    It answers "how much", the reorder point answers "when". EOQ assumes
    constant, known demand -- an assumption these bursty products violate
    badly, which is exactly why it is worth including as a comparison.
    """

    name = "EOQ+ROP"

    def __init__(self, rop_days: float, supplier_rule, safety_factor: float = 1.0):
        self.rop_days = rop_days
        self.safety_factor = safety_factor
        self.supplier_rule = supplier_rule
        self._eoq: np.ndarray | None = None

    def reset(self) -> None:
        self._eoq = None

    def choose_supplier(self, env):
        return self.supplier_rule(env)

    def _compute_eoq(self, env) -> np.ndarray:
        D = env.demand_gen.mean  # units per day
        K = float(np.mean(env.suppliers.fixed_cost))
        h = env.cost_model.unit_cost * env.cost_model.holding_rate  # per unit per day
        h = np.maximum(h, 1e-9)
        return np.sqrt(2.0 * D * K / h)

    def desired_quantity(self, env) -> np.ndarray:
        if self._eoq is None:
            self._eoq = self._compute_eoq(env)
        rate = np.maximum(self.recent_demand(env), 1e-6)
        position = self.inventory_position(env)
        rop = self.rop_days * rate * self.safety_factor
        return np.where(position < rop, self._eoq, 0.0)


class NewsvendorPolicy(Policy):
    """Order up to the critical-fractile quantile of lead-time demand.

    The newsvendor ratio Cu / (Cu + Co) balances the cost of running out
    against the cost of holding too much. With a 1.5x stockout multiplier and
    cheap holding, the ratio is high, so it stocks generously.

    Lead-time demand is approximated as normal. That is a real limitation for
    these products -- their demand is heavily right-skewed -- and part of why
    a learned policy has room to do better.
    """

    name = "newsvendor"

    def __init__(
        self, supplier_rule, horizon_days: float = 7.0, service_override: float | None = None
    ):
        self.supplier_rule = supplier_rule
        self.horizon_days = horizon_days
        self.service_override = service_override

    def choose_supplier(self, env):
        return self.supplier_rule(env)

    def critical_ratio(self, env) -> np.ndarray:
        if self.service_override is not None:
            return np.full(env.n_products, self.service_override)
        cu = env.cost_model.gross_profit_per_unit() * env.cost_model.stockout_multiplier
        co = env.cost_model.unit_cost * env.cost_model.holding_rate * self.horizon_days
        return cu / np.maximum(cu + co, 1e-9)

    def desired_quantity(self, env) -> np.ndarray:
        rate = np.maximum(self.recent_demand(env), 1e-6)
        sigma = np.maximum(env.demand_gen.std, 1e-6)
        lead = float(np.mean(env.suppliers.expected_lead_time()))
        horizon = lead + self.horizon_days

        z = sps.norm.ppf(np.clip(self.critical_ratio(env), 1e-4, 1 - 1e-4))
        target = rate * horizon + z * sigma * np.sqrt(horizon)
        return np.maximum(target - self.inventory_position(env), 0.0)


class ForecastSafetyStockPolicy(Policy):
    """Moving-average forecast plus a safety stock of z * sigma * sqrt(LT).

    The most common real-world approach outside textbooks, and the closest
    classical method to what an ML forecasting pipeline would produce. It
    adapts to recent demand, unlike EOQ, but still cannot anticipate a spike
    it has not yet seen.
    """

    name = "forecast+SS"

    def __init__(self, supplier_rule, window: int = 7, z: float = 1.65, review_days: float = 3.0):
        self.supplier_rule = supplier_rule
        self.window = window
        self.z = z
        self.review_days = review_days

    def choose_supplier(self, env):
        return self.supplier_rule(env)

    def desired_quantity(self, env) -> np.ndarray:
        hist = np.array(env.demand_history[-self.window :]) if env.demand_history else None
        if hist is None or len(hist) == 0:
            forecast = env.demand_gen.mean.copy()
            sigma = env.demand_gen.std.copy()
        else:
            forecast = hist.mean(axis=0)
            sigma = hist.std(axis=0) if len(hist) >= 2 else env.demand_gen.std.copy()
        sigma = np.maximum(sigma, 1e-6)

        lead = float(np.mean(env.suppliers.expected_lead_time()))
        horizon = lead + self.review_days
        target = forecast * horizon + self.z * sigma * np.sqrt(horizon)
        return np.maximum(target - self.inventory_position(env), 0.0)


# ------------------------------------------------------------------- registry

SUPPLIER_RULES = {
    "econo": FixedSupplier(0),
    "rapid": FixedSupplier(1),
    "midway": FixedSupplier(2),
    "urgency": UrgencySupplier(),
}


def build_policy(kind: str, params: dict) -> Policy:
    """Reconstruct a policy from a tuned parameter dict."""
    p = dict(params)
    rule_name = p.pop("supplier_rule", "econo")
    rule = SUPPLIER_RULES[rule_name] if rule_name in SUPPLIER_RULES else FixedSupplier(0)

    if kind == "random":
        return RandomPolicy()
    if kind == "constant":
        return ConstantPolicy(bucket=p.get("bucket", 2), supplier=p.get("supplier", 1))
    if kind == "(s,S)":
        return SSPolicy(p["s_days"], p["S_days"], rule)
    if kind == "EOQ+ROP":
        return EOQPolicy(p["rop_days"], rule, p.get("safety_factor", 1.0))
    if kind == "newsvendor":
        return NewsvendorPolicy(rule, p.get("horizon_days", 7.0), p.get("service_override"))
    if kind == "forecast+SS":
        return ForecastSafetyStockPolicy(
            rule, p.get("window", 7), p.get("z", 1.65), p.get("review_days", 3.0)
        )
    raise ValueError(f"unknown policy kind: {kind}")
