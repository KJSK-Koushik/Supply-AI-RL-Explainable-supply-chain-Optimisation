"""Fixed-seed episode runner. Shared by baselines (Phase 3) and RL (Phase 5).

Every policy must be measured on *identical* demand streams and supplier
behaviour, or the comparison is meaningless. Seeding the environment the same
way for every policy is what guarantees that: seed 101 produces the same
customers and the same delivery delays whether it is the (s,S) rule or the
trained agent facing them.
"""

from __future__ import annotations

import numpy as np

from src.env.scenarios import Scenario
from src.env.supply_chain_env import SupplyChainEnv


def run_episode(
    policy,
    seed: int,
    scenarios: list[Scenario] | None = None,
    env_cfg=None,
    supplier_cfg=None,
    stats=None,
    collect_trace: bool = False,
):
    """Run one full episode and return its summary metrics."""
    env = SupplyChainEnv(
        env_cfg=env_cfg,
        supplier_cfg=supplier_cfg,
        stats=stats,
        scenarios=scenarios,
        seed=seed,
    )
    env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()

    rng = np.random.default_rng(seed)
    trace = []
    while True:
        action = policy.act(env, rng) if hasattr(policy, "act") else policy(env, rng)
        _, _, terminated, truncated, info = env.step(action)
        if collect_trace:
            trace.append(info)
        if terminated or truncated:
            break

    summary = env.episode_summary()
    if collect_trace:
        summary["trace"] = trace
    return summary


def evaluate(policy, seeds, scenarios=None, **kwargs) -> dict:
    """Run a policy across several seeds and aggregate.

    Returns means plus standard deviations. A single episode of a stochastic
    environment tells you almost nothing; the spread is what says whether a
    difference between two policies is real.
    """
    runs = [run_episode(policy, s, scenarios, **kwargs) for s in seeds]
    keys = [k for k in runs[0] if k != "trace"]

    out = {}
    for k in keys:
        vals = np.array([r[k] for r in runs], dtype=float)
        out[k] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    out["n_episodes"] = len(runs)
    out["seeds"] = list(seeds)
    return out


# Seed discipline: parameters are chosen on TUNING_SEEDS and reported on
# EVAL_SEEDS. Tuning and reporting on the same seeds would let a policy look
# good simply by fitting those particular demand streams -- the same mistake as
# quoting training accuracy as a test result. The RL agent is held to the same
# split in Phase 5.
TUNING_SEEDS = list(range(100, 112))  # 12 episodes
EVAL_SEEDS = list(range(500, 530))  # 30 held-out episodes
