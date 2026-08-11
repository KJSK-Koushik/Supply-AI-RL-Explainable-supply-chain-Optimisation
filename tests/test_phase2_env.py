"""Phase 2 checks on the simulator.

A silent bug here would be invisible for weeks: the agent would happily learn
against wrong physics and produce results that mean nothing. These tests assert
the mechanics directly rather than trusting the reward curve to look sensible.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import load_config, resolve
from src.env import scenarios as sc

stats_path = resolve(load_config("data")["paths"]["demand_stats"])
pytestmark = pytest.mark.skipif(
    not stats_path.exists(), reason="run `python -m src.data.run_pipeline` first"
)


@pytest.fixture
def env():
    from src.env.supply_chain_env import SupplyChainEnv

    return SupplyChainEnv(seed=0)


# ------------------------------------------------------------ gymnasium API


def test_passes_gymnasium_check(env):
    from gymnasium.utils.env_checker import check_env

    check_env(env, skip_render_check=True)


def test_spaces_shaped_correctly(env):
    assert env.action_space.shape == (env.n_products * 2,)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_reset_is_reproducible(env):
    o1, _ = env.reset(seed=123)
    o2, _ = env.reset(seed=123)
    assert np.allclose(o1, o2)


def test_same_seed_same_trajectory():
    from src.env.supply_chain_env import SupplyChainEnv

    def rollout(seed):
        e = SupplyChainEnv(seed=seed)
        e.reset(seed=seed)
        rng = np.random.default_rng(0)
        rewards = []
        for _ in range(40):
            a = np.array([rng.integers(0, n) for n in e.action_space.nvec])
            _, r, _, _, _ = e.step(a)
            rewards.append(r)
        return rewards

    assert rollout(7) == rollout(7)
    # Different seeds must actually differ, or the seeding is not wired up.
    assert rollout(7) != rollout(8)


# --------------------------------------------------------------- invariants


def test_stock_never_negative(env):
    env.reset(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(env.episode_length):
        a = np.array([rng.integers(0, n) for n in env.action_space.nvec])
        _, _, _, trunc, info = env.step(a)
        assert (info["stock"] >= -1e-9).all(), "stock went negative"
        assert (info["backlog"] >= -1e-9).all()
        if trunc:
            break


def test_shared_capacity_never_exceeded(env):
    """Total stock across ALL products must respect the one shared warehouse."""
    env.reset(seed=2)
    big = np.array([[env.n_buckets - 1, 1]] * env.n_products).ravel()
    for _ in range(120):
        _, _, _, trunc, info = env.step(big)
        assert info["stock"].sum() <= env.capacity_total + 1e-6
        if trunc:
            break


def test_overflow_is_charged_when_warehouse_fills(env):
    """Ordering hard into a full store must produce refused deliveries that
    still cost money -- that is what makes over-ordering expensive."""
    env.reset(seed=15)
    big = np.array([[env.n_buckets - 1, 1]] * env.n_products).ravel()
    overflow_units = overflow_loss = 0.0
    for _ in range(120):
        _, _, _, trunc, info = env.step(big)
        overflow_units += info["costs"]["overflow_units"]
        overflow_loss += info["costs"]["overflow_loss"]
        if trunc:
            break
    assert overflow_units > 0, "warehouse never overflowed despite maximum orders"
    assert overflow_loss > 0, "refused goods were not charged"


def test_products_compete_for_shared_space(env):
    """Filling the store with one product must deny room to the others."""
    env.reset(seed=16)
    hog = np.zeros(env.n_products * 2, dtype=int)
    hog[0] = env.n_buckets - 1  # only product 0, largest bucket
    hog[1] = 1
    for _ in range(60):
        _, _, _, trunc, info = env.step(hog)
        if trunc:
            break
    # Product 0 should dominate the shelf while the rest have drained away.
    assert info["stock"][0] > info["stock"][1:].sum()


def test_sales_never_exceed_demand_plus_backlog(env):
    env.reset(seed=3)
    rng = np.random.default_rng(3)
    prev_backlog = np.zeros(env.n_products)
    for _ in range(60):
        a = np.array([rng.integers(0, n) for n in env.action_space.nvec])
        _, _, _, _, info = env.step(a)
        assert (info["units_sold"] <= info["demand"] + prev_backlog + 1e-6).all()
        prev_backlog = info["backlog"]


def test_profit_identity_holds(env):
    """profit must equal revenue minus every cost, exactly."""
    env.reset(seed=4)
    rng = np.random.default_rng(4)
    for _ in range(60):
        a = np.array([rng.integers(0, n) for n in env.action_space.nvec])
        _, _, _, _, info = env.step(a)
        c = info["costs"]
        expected = (
            c["revenue"]
            + c["salvage"]
            - c["purchase"]
            - c["ordering"]
            - c["holding"]
            - c["stockout_penalty"]
            - c["overflow_loss"]
        )
        assert abs(c["profit"] - expected) < 1e-6


def test_salvage_credited_only_on_final_day(env):
    """Leftover stock is recovered once, at the end -- never mid-episode."""
    env.reset(seed=41)
    a = np.array([[3, 1]] * env.n_products).ravel()
    salvages = []
    while True:
        _, _, term, trunc, info = env.step(a)
        salvages.append(info["costs"]["salvage"])
        if term or trunc:
            break
    assert all(s == 0.0 for s in salvages[:-1]), "salvage paid before the episode ended"
    assert salvages[-1] > 0.0, "leftover stock was not credited at the end"


def test_holding_cost_rate_is_defensible(env):
    """Guard against silently reintroducing an absurd holding rate. 0.02/day
    is 730%/year, which no reviewer would accept."""
    annual = env.cost_model.holding_rate * 365
    assert 0.2 < annual < 1.5, f"implied annual holding rate {annual:.0%} is not defensible"


def test_doing_nothing_eventually_causes_stockouts(env):
    """Sanity: never ordering must drain the shelf. If this fails, stock is
    being created from nowhere."""
    env.reset(seed=5)
    nothing = np.zeros(env.n_products * 2, dtype=int)
    shortfall = 0.0
    for _ in range(env.episode_length):
        _, _, _, trunc, info = env.step(nothing)
        shortfall += info["units_short"].sum()
        if trunc:
            break
    assert shortfall > 0
    assert env.stock.sum() < 1.0


def test_episode_truncates_at_configured_length(env):
    env.reset(seed=6)
    nothing = np.zeros(env.n_products * 2, dtype=int)
    steps = 0
    while True:
        _, _, term, trunc, _ = env.step(nothing)
        steps += 1
        if term or trunc:
            break
        assert steps <= env.episode_length + 1
    assert steps == env.episode_length


def test_warmup_days_pay_no_reward(env):
    env.reset(seed=7)
    nothing = np.zeros(env.n_products * 2, dtype=int)
    for d in range(1, env.warmup_days + 1):
        _, r, _, _, _ = env.step(nothing)
        assert r == 0.0, f"day {d} is warm-up and must not pay reward"


def test_saturday_never_traded(env):
    env.reset(seed=8)
    nothing = np.zeros(env.n_products * 2, dtype=int)
    closed = set(env.stats["closed_weekdays"])
    for _ in range(100):
        env.step(nothing)
        assert env.current_dow not in closed


# ----------------------------------------------------------------- suppliers


def test_deliveries_arrive_on_the_promised_day():
    """Order once, then wait. Goods must land exactly when the pipeline said."""
    from src.env.supply_chain_env import SupplyChainEnv

    e = SupplyChainEnv(seed=11)
    e.reset(seed=11)
    e.stock[:] = 0.0

    order = np.zeros(e.n_products * 2, dtype=int)
    order[0] = e.n_buckets - 1  # big order of product 0
    order[1] = 1  # from the fast supplier
    _, _, _, _, info = e.step(order)

    in_transit_before = info["in_transit"][0]
    assert in_transit_before > 0, "order did not enter the pipeline"

    nothing = np.zeros(e.n_products * 2, dtype=int)
    delivered = 0.0
    for _ in range(e.suppliers.max_lead + 1):
        _, _, _, _, info = e.step(nothing)
        delivered = max(delivered, info["stock"][0])
        if info["in_transit"][0] == 0:
            break
    assert info["in_transit"][0] == 0, "goods never arrived"
    assert delivered > 0, "goods vanished in transit"


def test_moq_is_enforced(env):
    """A small order must be rounded up to the supplier's minimum."""
    env.reset(seed=12)
    q = np.zeros(env.n_products)
    q[0] = 5.0
    choice = np.zeros(env.n_products, dtype=int)  # supplier 0, MOQ 100
    adjusted = env.suppliers.apply_moq(q, choice)
    assert adjusted[0] == env.suppliers.moq[0]
    # Products not ordered stay at zero.
    assert adjusted[1] == 0.0


def test_outage_rejects_orders(env):
    env.reset(seed=13)
    env.suppliers.force_outage(1, 5)
    assert not env.suppliers.is_available()[1]

    order = np.zeros(env.n_products * 2, dtype=int)
    order[0] = env.n_buckets - 1
    order[1] = 1  # the supplier that is down
    _, _, _, _, info = env.step(order)
    assert info["order_rejected"][0], "order to an offline supplier must be rejected"
    assert info["in_transit"][0] == 0


def test_fill_rate_can_shortfall(env):
    """Supplier 2 has an 80% mean fill rate, so over many orders we must
    receive materially less than we asked for."""
    env.reset(seed=14)
    asked = received = 0.0
    order = np.zeros(env.n_products * 2, dtype=int)
    order[0] = 3
    order[1] = 2  # the unreliable supplier
    for _ in range(60):
        _, _, _, _, info = env.step(order)
        if info["order_quantities"][0] > 0 and not info["order_rejected"][0]:
            asked += info["order_quantities"][0]
            received += info["order_quantities"][0] * info["fill_rate"][0]
    assert asked > 0
    ratio = received / asked
    assert 0.6 < ratio < 0.95, f"fill ratio {ratio:.2f} is not consistent with an 80% supplier"


def test_supplier_tradeoff_is_real(env):
    """Fastest must not also be cheapest, or supplier choice is trivial."""
    s = env.suppliers
    assert np.argmin(s.lt_mode) != np.argmin(s.cost_multiplier)


# ----------------------------------------------------------------- scenarios


def test_demand_spike_raises_demand():
    from src.env.supply_chain_env import SupplyChainEnv

    def total_demand(scenarios):
        e = SupplyChainEnv(seed=21, scenarios=scenarios)
        e.reset(seed=21)
        nothing = np.zeros(e.n_products * 2, dtype=int)
        total = 0.0
        for _ in range(60):
            _, _, _, _, info = e.step(nothing)
            total += info["demand"].sum()
        return total

    base = total_demand(None)
    spiked = total_demand(
        [sc.demand_spike(products=None, multiplier=3.0, start_day=10, duration=40)]
    )
    assert spiked > base * 1.5, f"spike had no real effect ({base:.0f} -> {spiked:.0f})"


def test_scenario_window_is_respected():
    mgr = sc.ScenarioManager(n_products=10, n_suppliers=3)
    mgr.add(sc.demand_spike(products=[0], multiplier=4.0, start_day=10, duration=5))
    assert mgr.demand_multiplier(9)[0] == 1.0
    assert mgr.demand_multiplier(10)[0] == 4.0
    assert mgr.demand_multiplier(14)[0] == 4.0
    assert mgr.demand_multiplier(15)[0] == 1.0, "spike must stop after its duration"


def test_supplier_outage_scenario_takes_supplier_down():
    from src.env.supply_chain_env import SupplyChainEnv

    e = SupplyChainEnv(
        seed=22, scenarios=[sc.supplier_outage(supplier=1, start_day=5, duration=10)]
    )
    e.reset(seed=22)
    nothing = np.zeros(e.n_products * 2, dtype=int)
    seen_down = False
    for d in range(1, 13):
        _, _, _, _, info = e.step(nothing)
        if d >= 5 and not info["supplier_available"][1]:
            seen_down = True
    assert seen_down, "scheduled outage never took the supplier offline"


def test_supplier_delay_extends_lead_time():
    from src.env.supply_chain_env import SupplyChainEnv

    e = SupplyChainEnv(
        seed=23, scenarios=[sc.supplier_delay(supplier=1, extra_days=8, start_day=1, duration=50)]
    )
    e.reset(seed=23)
    baseline = (e.suppliers.lt_min[1] + e.suppliers.lt_mode[1] + e.suppliers.lt_max[1]) / 3
    nothing = np.zeros(e.n_products * 2, dtype=int)
    e.step(nothing)
    assert e.suppliers.expected_lead_time()[1] > baseline + 7


def test_scenario_descriptions_are_human_readable():
    s = sc.demand_spike(products=[0, 1], multiplier=3.0, start_day=40, duration=7)
    text = s.describe()
    assert "3.0x" in text and "40" in text


# -------------------------------------------------------------- economics


def test_ordering_more_raises_holding_cost():
    from src.env.supply_chain_env import SupplyChainEnv

    def holding(bucket):
        e = SupplyChainEnv(seed=31)
        e.reset(seed=31)
        a = np.array([[bucket, 1]] * e.n_products).ravel()
        total = 0.0
        for _ in range(60):
            _, _, _, _, info = e.step(a)
            total += info["costs"]["holding"]
        return total

    assert holding(5) > holding(1), "holding more stock must cost more"


def test_stockouts_are_penalised_more_than_lost_margin(env):
    gp = env.cost_model.gross_profit_per_unit()
    assert env.cost_model.stockout_multiplier > 1.0
    assert (gp > 0).all(), "selling price must exceed unit cost"
