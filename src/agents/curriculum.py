"""Phase 9: train the agent on LLM-generated crises, then test it on unseen ones.

Phase 7 measured the problem this exists to fix: under disruption the tuned
classical policy loses 13% of its profit and the RL agent loses 28%. The agent
is brittle because every episode it has ever seen was calm.

The method is a curriculum. A pool of disruption sets is generated once and
split in two. The agent trains against a random set from the training half at
the start of every episode, and is scored against the holdout half, which it
never sees during training.

That split is the whole methodology. Training and testing on the same crises
would show a large, meaningless improvement -- the agent would be memorising
particular disruptions rather than learning to handle disruption. The claim
being tested is generalisation, so the test set has to be genuinely held out.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict

import gymnasium as gym

from src.config import resolve
from src.env.scenarios import Scenario
from src.llm.scenario_gen import fallback_scenarios, generate

POOL_PATH = "results/curriculum_pool.json"

# Varied prompts, because one prompt repeated returns near-identical crises and
# a curriculum of one crisis is not a curriculum.
THEMES = [
    "a sudden viral demand surge combined with a supplier failure",
    "a slow squeeze: gradually worsening lead times and rising costs",
    "a seasonal peak that arrives earlier and harder than expected",
    "logistics disruption affecting the cheapest supplier for a long period",
    "simultaneous cost inflation and unreliable deliveries",
    "a short sharp shock followed by a second one before recovery",
]


def _to_dict(sc: Scenario) -> dict:
    return asdict(sc)


def _from_dict(d: dict) -> Scenario:
    return Scenario(**d)


def build_pool(
    n_calls: int = 4,
    sets_per_pool: int = 16,
    seed: int = 0,
    use_llm: bool = True,
    force: bool = False,
    cache_path: str | None = None,
) -> list[list[Scenario]]:
    """Generate, or reload, the pool of disruption sets.

    Cached on disk deliberately. Free endpoints are rate-limited and generation
    is slow, but the stronger reason is reproducibility: a training run and the
    evaluation that judges it must refer to the same crises, and regenerating
    would silently change the experiment between them.

    Each LLM call is expensive and each returns only a handful of scenarios, so
    the calls build a *library* and the sets are composed by sampling from it.
    A few calls then yield many distinct combinations.
    """
    # Injectable so a test can build a throwaway pool. Defaulting to the real
    # cache made the suite overwrite a live experiment's crises: training had
    # already loaded the LLM pool, the test replaced it on disk with an offline
    # one, and the evaluation would then have scored the agent against a
    # different holdout than it trained on -- with nothing reporting a problem.
    path = resolve(cache_path or POOL_PATH)
    if path.exists() and not force:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return [[_from_dict(d) for d in group] for group in data["sets"]]

    library: list[Scenario] = list(fallback_scenarios())
    sources = ["fallback"]
    if use_llm:
        for i in range(n_calls):
            out = generate(n=4, theme=THEMES[i % len(THEMES)])
            if out["source"] == "llm":
                library.extend(out["scenarios"])
                sources.append(out["model"] or "llm")
            else:
                sources.append(f"fallback ({out['reason']})")

    rng = random.Random(seed)
    sets: list[list[Scenario]] = []
    seen: set[tuple] = set()
    attempts = 0
    while len(sets) < sets_per_pool and attempts < sets_per_pool * 50:
        attempts += 1
        k = rng.randint(2, min(4, len(library)))
        picked = rng.sample(library, k)
        key = tuple(sorted((s.type, s.start_day, s.duration, s.label) for s in picked))
        if key in seen:
            continue
        seen.add(key)
        sets.append(picked)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "sources": sources,
                "library_size": len(library),
                "sets": [[_to_dict(s) for s in group] for group in sets],
            },
            fh,
            indent=2,
        )
    return sets


def split_pool(
    pool: list[list[Scenario]], holdout_fraction: float = 0.4, seed: int = 0
) -> tuple[list, list]:
    """Split into training and holdout halves.

    Shuffled with a fixed seed rather than sliced in order, because the pool is
    built by walking the themes in sequence -- an ordered slice would put whole
    themes on one side of the split and test generalisation across themes
    rather than across crises.
    """
    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)
    n_holdout = max(1, int(round(len(shuffled) * holdout_fraction)))
    return shuffled[n_holdout:], shuffled[:n_holdout]


class CurriculumWrapper(gym.Wrapper):
    """Draw a fresh disruption set at the start of every episode.

    The environment applies whatever is in `_initial_scenarios` when it resets,
    so the set is swapped in just before delegating. Assigning after reset
    would leave the first day of each episode running under the previous
    episode's crisis.
    """

    def __init__(self, env, scenario_sets: list[list[Scenario]], seed: int = 0):
        super().__init__(env)
        self.scenario_sets = scenario_sets
        self._rng = random.Random(seed)
        self.current_set: list[Scenario] = []

    def reset(self, **kwargs):
        if self.scenario_sets:
            self.current_set = list(self._rng.choice(self.scenario_sets))
            self.env.unwrapped._initial_scenarios = self.current_set
        return self.env.reset(**kwargs)

    def action_masks(self):
        # MaskablePPO looks this up on the wrapper. gym.Wrapper forwards
        # unknown attributes, but being explicit means a rename upstream fails
        # loudly instead of silently training without masking -- which was
        # worth 3x and would be an expensive thing to lose quietly.
        return self.env.unwrapped.action_masks()
