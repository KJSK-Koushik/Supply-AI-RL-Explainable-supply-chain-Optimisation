"""Phase 3 checks on the baseline policies and the evaluation harness.

The point of these tests is fairness. If the baselines are quietly crippled --
given worse information, a coarser action space, or fewer tuning seeds than the
RL agent -- then any RL win in Phase 5 is manufactured. These assert that the
comparison is set up honestly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.baselines import (
    SUPPLIER_RULES,
    ConstantPolicy,
    EOQPolicy,
    ForecastSafetyStockPolicy,
    NewsvendorPolicy,
    Policy,
    RandomPolicy,
    SSPolicy,
    UrgencySupplier,
)
from src.config import load_config, resolve
from src.eval.runner import EVAL_SEEDS, TUNING_SEEDS, evaluate, run_episode

stats_path = resolve(load_config("data")["paths"]["demand_stats"])
pytestmark = pytest.mark.skipif(
    not stats_path.exists(), reason="run `python -m src.data.run_pipeline` first"
)

ECONO = SUPPLIER_RULES["econo"]

ALL_POLICIES = [
    RandomPolicy(),
    ConstantPolicy(2, 1),
    SSPolicy(7, 15, ECONO),
    EOQPolicy(8, ECONO),
    NewsvendorPolicy(ECONO),
    ForecastSafetyStockPolicy(ECONO),
]


@pytest.fixture
def env():
    from src.env.supply_chain_env import SupplyChainEnv

    return SupplyChainEnv(seed=0)


# --------------------------------------------------------------- action shape


@pytest.mark.parametrize("policy", ALL_POLICIES, ids=lambda p: p.name)
def test_action_is_valid(policy, env):
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(30):
        action = policy.act(env, rng)
        assert env.action_space.contains(np.asarray(action, dtype=np.int64)), (
            f"{policy.name} emitted an action outside the action space"
        )
        env.step(action)


@pytest.mark.parametrize("policy", ALL_POLICIES, ids=lambda p: p.name)
def test_episode_completes(policy):
    res = run_episode(policy, seed=0)
    assert res["n_episodes"] if "n_episodes" in res else True
    assert np.isfinite(res["total_profit"])


# ------------------------------------------------------------------- fairness


def test_baselines_use_same_action_granularity(env):
    """Baselines must be snapped to the same 7 buckets the agent gets.
    Free-form quantities for baselines only would rig the comparison."""
    env.reset(seed=0)
    want = np.full(env.n_products, 12345.0)  # absurd request
    buckets = Policy.quantise(env, want)
    assert buckets.max() <= env.n_buckets - 1
    assert buckets.min() >= 0


def test_quantise_picks_nearest_bucket(env):
    env.reset(seed=0)
    # Ask for exactly 2x mean daily demand; bucket index 3 is the 2.0x bucket.
    want = env.demand_gen.mean * 2.0
    assert (Policy.quantise(env, want) == 3).all()
    # Asking for nothing must map to the zero bucket.
    assert (Policy.quantise(env, np.zeros(env.n_products)) == 0).all()


def test_policies_see_in_transit_stock(env):
    """A policy ordering against on-hand stock alone would double-order."""
    env.reset(seed=0)
    p = SSPolicy(7, 15, ECONO)
    pos = p.inventory_position(env)
    assert np.allclose(pos, env.stock + env.suppliers.in_transit())


def test_tuning_and_eval_seeds_are_disjoint():
    """Parameters tuned and reported on the same seeds is the same error as
    quoting training accuracy as a test score."""
    assert not set(TUNING_SEEDS) & set(EVAL_SEEDS)
    assert len(EVAL_SEEDS) >= 30


def test_evaluation_is_deterministic_per_seed():
    p = SSPolicy(7, 15, ECONO)
    a = evaluate(p, [101, 102, 103])
    b = evaluate(p, [101, 102, 103])
    assert a["total_profit"] == b["total_profit"]


def test_all_policies_face_identical_demand():
    """Two different policies on the same seed must meet the same customers,
    or the comparison measures luck rather than skill."""
    d1 = run_episode(SSPolicy(7, 15, ECONO), seed=77, collect_trace=True)["trace"]
    d2 = run_episode(ConstantPolicy(2, 1), seed=77, collect_trace=True)["trace"]
    # Day 1 demand is drawn before either policy has influenced anything.
    assert np.allclose(d1[0]["demand"], d2[0]["demand"])


def test_evaluate_reports_spread():
    res = evaluate(SSPolicy(7, 15, ECONO), [101, 102, 103, 104])
    assert "total_profit_std" in res
    assert res["total_profit_std"] >= 0.0


# -------------------------------------------------------------- policy sanity


def test_ss_policy_orders_when_cover_is_low(env):
    env.reset(seed=0)
    env.stock[:] = 0.0
    p = SSPolicy(s_days=7, S_days=15, supplier_rule=ECONO)
    assert (p.desired_quantity(env) > 0).all(), "empty shelf must trigger an order"


def test_ss_policy_holds_when_well_stocked(env):
    env.reset(seed=0)
    env.stock[:] = env.demand_gen.mean * 40  # far above S
    p = SSPolicy(s_days=7, S_days=15, supplier_rule=ECONO)
    assert (p.desired_quantity(env) == 0).all(), "overstocked shelf must not reorder"


def test_higher_S_orders_more(env):
    env.reset(seed=0)
    env.stock[:] = 0.0
    low = SSPolicy(7, 10, ECONO).desired_quantity(env)
    high = SSPolicy(7, 20, ECONO).desired_quantity(env)
    assert (high >= low).all() and high.sum() > low.sum()


def test_eoq_quantity_matches_formula(env):
    env.reset(seed=0)
    p = EOQPolicy(8, ECONO)
    eoq = p._compute_eoq(env)
    D = env.demand_gen.mean
    K = float(np.mean(env.suppliers.fixed_cost))
    h = env.cost_model.unit_cost * env.cost_model.holding_rate
    assert np.allclose(eoq, np.sqrt(2 * D * K / h))


def test_newsvendor_ratio_favours_stocking(env):
    """With a 1.5x stockout multiplier and cheap holding, the critical ratio
    must exceed 0.5 -- running out hurts more than holding."""
    env.reset(seed=0)
    cr = NewsvendorPolicy(ECONO).critical_ratio(env)
    assert (cr > 0.5).all() and (cr < 1.0).all()


def test_higher_z_raises_safety_stock(env):
    env.reset(seed=0)
    env.stock[:] = 0.0
    lo = ForecastSafetyStockPolicy(ECONO, z=0.8).desired_quantity(env)
    hi = ForecastSafetyStockPolicy(ECONO, z=2.05).desired_quantity(env)
    assert hi.sum() > lo.sum()


# ------------------------------------------------------------ supplier rules


def test_fixed_supplier_falls_back_during_outage(env):
    env.reset(seed=0)
    env.suppliers.force_outage(0, 5)
    choice = SUPPLIER_RULES["econo"](env)
    assert (choice != 0).all(), "must not keep ordering from an offline supplier"


def test_urgency_rule_switches_to_fast_supplier_when_short(env):
    env.reset(seed=0)
    rule = UrgencySupplier(urgent_days=4.0)
    env.stock[:] = env.demand_gen.mean * 30  # comfortable
    assert (rule(env) == 0).all(), "comfortable cover should use the cheap supplier"
    env.stock[:] = 0.0  # desperate
    assert (rule(env) == 1).all(), "critical cover should use the fast supplier"


# ------------------------------------------------------------- overall shape


def test_sensible_policy_beats_random():
    """If a textbook rule cannot beat random flailing, either the reward
    function or the policy implementation is broken."""
    seeds = [201, 202, 203, 204, 205]
    good = evaluate(SSPolicy(7, 15, ECONO), seeds)["total_profit"]
    rand = evaluate(RandomPolicy(), seeds)["total_profit"]
    assert good > rand, f"(s,S) {good:,.0f} did not beat random {rand:,.0f}"


def test_sensible_policy_is_profitable():
    seeds = [201, 202, 203, 204, 205]
    assert evaluate(SSPolicy(7, 15, ECONO), seeds)["total_profit"] > 0
