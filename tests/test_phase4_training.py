"""Phase 4 checks on the training setup.

These guard the things that would silently invalidate a result: the agent
being scored differently from the baselines, evaluation seeds leaking into
training, or the action-space optimisation being undone.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.train_ppo import apply_overrides
from src.config import load_config, resolve
from src.eval.runner import EVAL_SEEDS, TUNING_SEEDS

stats_path = resolve(load_config("data")["paths"]["demand_stats"])
pytestmark = pytest.mark.skipif(
    not stats_path.exists(), reason="run `python -m src.data.run_pipeline` first"
)


def test_config_overrides_parse_types():
    cfg = apply_overrides(load_config("train"), ["ppo.learning_rate=1e-4", "run.n_envs=4"])
    assert cfg["ppo"]["learning_rate"] == 1e-4
    assert cfg["run"]["n_envs"] == 4
    assert isinstance(cfg["run"]["n_envs"], int)


def test_eval_seeds_do_not_leak():
    """The seeds watched during training must be disjoint from the final
    held-out set AND from the baseline tuning seeds. Selecting the best
    checkpoint on the reporting seeds is the same error as early-stopping on
    the test set."""
    train_eval = set(load_config("train")["eval"]["seeds"])
    assert not train_eval & set(EVAL_SEEDS), "training eval leaks the reporting seeds"
    assert not train_eval & set(TUNING_SEEDS), "training eval reuses baseline tuning seeds"


def test_action_space_stays_joint():
    """Guard the 2.8x speedup. Reverting to one dimension per (bucket,
    supplier) choice would double the number of Categorical distributions PPO
    builds per step."""
    from src.env.supply_chain_env import SupplyChainEnv

    env = SupplyChainEnv(seed=0)
    assert env.action_space.shape == (env.n_products,)
    assert env.n_joint_actions == env.n_buckets * env.n_suppliers


def test_rl_policy_uses_same_harness_as_baselines():
    """The agent must be scored through the identical evaluation path the
    baselines use, or the comparison measures two different quantities."""
    from stable_baselines3 import PPO

    from src.agents.rl_policy import RLPolicy
    from src.env.supply_chain_env import SupplyChainEnv
    from src.eval.runner import run_episode

    model = PPO("MlpPolicy", SupplyChainEnv(seed=0), device="cpu", n_steps=64, batch_size=64)
    policy = RLPolicy(model)

    res = run_episode(policy, seed=999)
    assert np.isfinite(res["total_profit"])
    # Same metric keys as any baseline, so the comparison table lines up.
    for key in ("total_profit", "fill_rate", "holding_cost", "overflow_loss"):
        assert key in res


def test_rl_policy_emits_valid_actions():
    from stable_baselines3 import PPO

    from src.agents.rl_policy import RLPolicy
    from src.env.supply_chain_env import SupplyChainEnv

    env = SupplyChainEnv(seed=0)
    env.reset(seed=0)
    model = PPO("MlpPolicy", env, device="cpu", n_steps=64, batch_size=64)
    policy = RLPolicy(model)

    for _ in range(20):
        a = policy.act(env)
        assert env.action_space.contains(np.asarray(a, dtype=np.int64))
        env.step(a)


def test_sweep_grid_is_sane():
    from src.agents.sweep import GRID, configs

    assert set(GRID) >= {"ppo.learning_rate", "ppo.ent_coef", "ppo.gamma"}
    # A high discount factor matters here: a 6-8 day lead time means today's
    # order only pays off a week later.
    assert min(GRID["ppo.gamma"]) >= 0.99
    # Entropy must be able to go both ways; a collapsed policy that never
    # orders is the classic failure mode.
    assert min(GRID["ppo.ent_coef"]) < 0.01 < max(GRID["ppo.ent_coef"])

    got = list(configs(budget=5))
    assert len(got) == 5
    assert all(set(c) == set(GRID) for c in got)


def test_baseline_bar_is_recorded():
    """Phase 5 compares against this number, so it must exist before training
    conclusions are drawn."""
    import json

    path = resolve("results/baselines.json")
    if not path.exists():
        pytest.skip("run `python -m src.agents.tune_baselines` first")
    data = json.load(path.open(encoding="utf-8"))
    best = data["policies"][data["best_baseline"]]["eval"]
    assert best["total_profit"] > 50_000
    assert best["total_profit_std"] > 0


# ------------------------------------------------------------ action masking


def test_mask_shape_and_always_has_a_legal_move():
    from src.env.supply_chain_env import SupplyChainEnv

    env = SupplyChainEnv(seed=0)
    env.reset(seed=0)
    m = env.action_masks()
    assert m.shape == (env.n_products * env.n_joint_actions,)
    assert m.dtype == bool
    per_product = m.reshape(env.n_products, env.n_joint_actions)
    assert per_product.any(axis=1).all(), "a product with no legal action would deadlock PPO"


def test_mask_blocks_offline_suppliers():
    from src.env.supply_chain_env import SupplyChainEnv

    env = SupplyChainEnv(seed=0)
    env.reset(seed=0)
    env.suppliers.force_outage(1, 5)
    m = env.action_masks().reshape(env.n_products, env.n_joint_actions)

    supplier_of = np.arange(env.n_joint_actions) % env.n_suppliers
    bucket_of = np.arange(env.n_joint_actions) // env.n_suppliers
    # Every non-zero order through the offline supplier must be unavailable.
    blocked = (supplier_of == 1) & (bucket_of > 0)
    assert not m[:, blocked].any(), "ordering from an offline supplier is still allowed"
    # Other suppliers are unaffected.
    assert m[:, (supplier_of == 0) & (bucket_of > 0)].any()


def test_mask_blocks_orders_that_cannot_fit():
    """A full warehouse must make large orders unavailable -- this is the
    25,308 overflow loss the first unmasked run suffered."""
    from src.env.supply_chain_env import SupplyChainEnv

    env = SupplyChainEnv(seed=0)
    env.reset(seed=0)
    env.stock[:] = env.capacity_total / env.n_products  # warehouse full
    m = env.action_masks().reshape(env.n_products, env.n_joint_actions)

    bucket_of = np.arange(env.n_joint_actions) // env.n_suppliers
    assert not m[:, bucket_of == env.n_buckets - 1].any(), "largest order allowed into a full store"
    assert m[:, bucket_of == 0].all(), "ordering nothing must always stay legal"


def test_mask_permits_large_orders_when_empty():
    from src.env.supply_chain_env import SupplyChainEnv

    env = SupplyChainEnv(seed=0)
    env.reset(seed=0)
    env.stock[:] = 0.0
    env.suppliers.pipeline[:] = 0.0
    m = env.action_masks().reshape(env.n_products, env.n_joint_actions)
    assert m.all(), "an empty warehouse should permit every action"


def test_maskable_agent_trains_and_scores():
    from sb3_contrib import MaskablePPO

    from src.agents.rl_policy import RLPolicy
    from src.env.supply_chain_env import SupplyChainEnv
    from src.eval.runner import run_episode

    model = MaskablePPO(
        "MlpPolicy", SupplyChainEnv(seed=0), device="cpu", n_steps=64, batch_size=64
    )
    policy = RLPolicy(model)
    assert policy._is_maskable()
    res = run_episode(policy, seed=999)
    assert np.isfinite(res["total_profit"])


# ------------------------------------------------------- honest comparison


def test_paired_comparison_detects_a_real_difference():
    from src.eval.compare import paired_comparison

    # Consistently 1,000 better on every seed: small spread, clearly real.
    a = [100.0 + i for i in range(30)]
    b = [x - 1000.0 for x in a]
    r = paired_comparison(a, b)
    assert r["mean_difference"] == pytest.approx(1000.0)
    assert r["significant"]


def test_paired_comparison_rejects_noise():
    """Two policies differing only by noise must NOT be called a win. This is
    the guard against reporting an inside-the-noise result as an improvement."""
    from src.eval.compare import paired_comparison

    rng = np.random.default_rng(0)
    a = rng.normal(80_000, 5_000, 30)
    b = rng.normal(80_000, 5_000, 30)
    assert not paired_comparison(list(a), list(b))["significant"]


def test_comparison_uses_reporting_seeds_not_training_seeds():
    """The comparison must score on EVAL_SEEDS. Scoring on the seeds that
    selected the checkpoint would inflate the agent."""
    import inspect

    from src.eval import compare

    src = inspect.getsource(compare.main)
    assert "EVAL_SEEDS" in src
    train_eval = set(load_config("train")["eval"]["seeds"])
    assert not train_eval & set(EVAL_SEEDS)
