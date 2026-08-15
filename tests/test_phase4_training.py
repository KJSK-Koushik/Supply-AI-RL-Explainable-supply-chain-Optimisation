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
