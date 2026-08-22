"""Phase 2 demo: run the simulator with simple hand-written policies.

    python scripts/demo_simulator.py

No learning here. The point is to show the mechanics working and to confirm the
environment rewards sensible behaviour -- if a naive constant-order policy did
not beat random flailing, the reward function would be broken and any RL result
built on it would be meaningless.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Running this file directly puts scripts/ on sys.path, not the project root,
# so `import src...` fails. pytest.ini fixes that for tests only, and the
# docstring above promises `python scripts/demo_simulator.py` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env.scenarios import demand_spike, supplier_outage  # noqa: E402
from src.env.supply_chain_env import SupplyChainEnv  # noqa: E402


def policy_random(env, rng):
    return np.array([rng.integers(0, n) for n in env.action_space.nvec])


def policy_never_order(env, rng):
    return np.zeros(env.n_products, dtype=int)


def policy_always_max(env, rng):
    return env.encode_action(np.full(env.n_products, env.n_buckets - 1), np.full(env.n_products, 1))


def policy_constant_1x(env, rng):
    """Order one day of mean demand, every day, from the fast supplier.
    Crude, but it roughly matches the long-run consumption rate."""
    return env.encode_action(np.full(env.n_products, 2), np.full(env.n_products, 1))


def policy_order_up_to(env, rng):
    """A hand-tuned (s,S) stand-in: top up towards a target days-of-cover.

    The target has to exceed the lead time or the rule is structurally behind
    -- goods ordered today do not arrive for 5-8 days from the cheap supplier,
    so a 5-day target guarantees permanent shortage. Phase 3 grid-searches
    these numbers properly instead of guessing them.
    """
    target_cover = 12.0
    reorder_at = 8.0
    mean = np.maximum(env.demand_gen.mean, 1e-6)
    cover = (env.stock + env.suppliers.in_transit()) / mean

    buckets = np.zeros(env.n_products, dtype=int)
    for p in range(env.n_products):
        if cover[p] < reorder_at:
            # Days of cover to make up, expressed in mean-daily-demand units,
            # snapped to the nearest available bucket.
            gap = target_cover - cover[p]
            buckets[p] = int(np.argmin(np.abs(env.order_buckets - gap)))
    # Supplier 0, EconoSource, the cheap slow one. Encoded through the env
    # rather than assembled by hand: quantity and supplier are one joint
    # choice per product now, not two separate values flattened together.
    return env.encode_action(buckets, np.zeros(env.n_products, dtype=int))


def run(policy, seed=0, scenarios=None, episodes=5):
    results = []
    for ep in range(episodes):
        env = SupplyChainEnv(seed=seed + ep, scenarios=scenarios)
        env.reset(seed=seed + ep)
        rng = np.random.default_rng(seed + ep)
        while True:
            _, _, term, trunc, _ = env.step(policy(env, rng))
            if term or trunc:
                break
        results.append(env.episode_summary())

    keys = results[0].keys()
    return {k: float(np.mean([r[k] for r in results])) for k in keys}


def show(title, rows):
    print(f"\n{title}")
    print(
        f"  {'policy':16s} {'profit':>12s} {'fill rate':>10s} {'holding':>10s} "
        f"{'stockout pen':>13s} {'short units':>12s}"
    )
    for name, r in rows:
        print(
            f"  {name:16s} {r['total_profit']:12,.0f} {r['fill_rate']:9.1%} "
            f"{r['holding_cost']:10,.0f} {r['stockout_penalty']:13,.0f} "
            f"{r['units_short']:12,.0f}"
        )


def main():
    policies = [
        ("never order", policy_never_order),
        ("random", policy_random),
        ("always max", policy_always_max),
        ("constant 1x", policy_constant_1x),
        ("order-up-to", policy_order_up_to),
    ]

    show("Normal conditions (5 episodes x 180 days, mean)", [(n, run(p)) for n, p in policies])

    disruption = [
        demand_spike(
            products=None, multiplier=3.0, start_day=60, duration=21, label="festival demand spike"
        ),
        supplier_outage(
            supplier=0, start_day=90, duration=14, label="EconoSource factory shutdown"
        ),
    ]
    show(
        "Under disruption (3x demand days 60-80, cheap supplier down days 90-103)",
        [(n, run(p, scenarios=disruption)) for n, p in policies],
    )

    print("\nSample trace (order-up-to, first 10 trading days):")
    env = SupplyChainEnv(seed=0)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(10):
        env.step(policy_order_up_to(env, rng))
        env.render()


if __name__ == "__main__":
    main()
