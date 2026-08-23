"""Phase 9: curriculum training, and the split that makes its claim meaningful.

The methodological risk in this phase is not a crash, it is a result that looks
good and means nothing. Training and evaluating on the same disruptions would
show a large improvement while measuring memorisation. Most of these tests
guard that boundary rather than the code paths.
"""

from __future__ import annotations

import pytest

from src.agents.curriculum import (
    CurriculumWrapper,
    build_pool,
    split_pool,
)
from src.config import load_config, resolve

stats_path = resolve(load_config("data")["paths"]["demand_stats"])
pytestmark = pytest.mark.skipif(
    not stats_path.exists(), reason="run `python -m src.data.run_pipeline` first"
)


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    """A throwaway cache. Never the real one.

    These tests write pools. Pointing them at results/curriculum_pool.json
    overwrote a running experiment's crises exactly once, which is once more
    than it should be possible to do by running the test suite.
    """
    return str(tmp_path_factory.mktemp("curriculum") / "pool.json")


@pytest.fixture(scope="module")
def pool(cache):
    # Offline: this must not depend on a rate-limited endpoint, and the split
    # behaviour under test is identical either way.
    return build_pool(use_llm=False, force=True, sets_per_pool=12, cache_path=cache)


def _key(group):
    return tuple(sorted((s.type, s.start_day, s.duration, s.label) for s in group))


# ------------------------------------------------------------------- split


def test_training_and_holdout_sets_do_not_overlap(pool):
    """The whole claim of Phase 9 rests on this.

    If a disruption the agent trained against also appears in the evaluation,
    the reported robustness gain measures memorisation of that crisis rather
    than the ability to handle crises.
    """
    train, holdout = split_pool(pool)
    assert train and holdout
    assert not ({_key(g) for g in train} & {_key(g) for g in holdout})


def test_split_is_deterministic(pool):
    """A training run and the evaluation that judges it are separate processes.
    If the split moved between them, the agent would be scored on sets it had
    trained against, silently."""
    assert [_key(g) for g in split_pool(pool, seed=0)[0]] == [
        _key(g) for g in split_pool(pool, seed=0)[0]
    ]


def test_every_set_lands_on_exactly_one_side(pool):
    train, holdout = split_pool(pool)
    assert len(train) + len(holdout) == len(pool)


# ------------------------------------------------------------------ pool


def test_pool_is_cached_rather_than_regenerated(pool, cache):
    """Regenerating mid-experiment would change the crises between training
    and evaluation without anything reporting that it had happened."""
    import pathlib

    again = build_pool(use_llm=False, cache_path=cache)
    assert [_key(g) for g in again] == [_key(g) for g in pool]
    assert pathlib.Path(cache).exists()


def test_sets_are_distinct(pool):
    keys = [_key(g) for g in pool]
    assert len(keys) == len(set(keys))


# --------------------------------------------------------------- wrapper


def test_wrapper_draws_a_scenario_set_on_every_reset(pool):
    from src.env.supply_chain_env import SupplyChainEnv

    train, _ = split_pool(pool)
    env = CurriculumWrapper(SupplyChainEnv(seed=0), train, seed=0)
    for i in range(6):
        env.reset(seed=i)
        assert env.current_set
        assert env.unwrapped._initial_scenarios == env.current_set


def test_wrapper_varies_the_crisis_across_episodes(pool):
    """A curriculum of one crisis is not a curriculum."""
    from src.env.supply_chain_env import SupplyChainEnv

    train, _ = split_pool(pool)
    env = CurriculumWrapper(SupplyChainEnv(seed=0), train, seed=0)
    seen = set()
    for i in range(25):
        env.reset(seed=i)
        seen.add(_key(env.current_set))
    assert len(seen) > 1


def test_wrapper_preserves_action_masking(pool):
    """Masking was worth roughly 3x. Losing it silently through a wrapper
    would be an expensive and very quiet regression."""
    from src.env.supply_chain_env import SupplyChainEnv

    train, _ = split_pool(pool)
    env = CurriculumWrapper(SupplyChainEnv(seed=0), train, seed=0)
    env.reset(seed=0)
    masks = env.action_masks()
    assert masks is not None
    assert masks.any(), "at least one action must always be legal"


def test_scenarios_are_applied_from_the_first_day(pool):
    """Assigning the set after reset would leave day one running under the
    previous episode's crisis."""
    from src.env.supply_chain_env import SupplyChainEnv

    train, _ = split_pool(pool)
    env = CurriculumWrapper(SupplyChainEnv(seed=0), train, seed=0)
    env.reset(seed=0)
    assert env.unwrapped.scenario_mgr.scenarios, "manager should hold the drawn set"
