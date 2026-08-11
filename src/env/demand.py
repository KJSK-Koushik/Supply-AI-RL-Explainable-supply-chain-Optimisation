"""Daily demand generation, calibrated from real UCI Online Retail II statistics.

Real retail demand is *overdispersed*: its variance is far larger than its
mean. Product 21212 averages 168.6 units/day with a standard deviation of
191.0 -- variance is roughly 216x the mean. A Poisson process, where variance
equals the mean by definition, cannot produce that. It would generate a tidy
series hovering near 168 and the agent would learn that demand is far more
predictable than it really is, then get destroyed by the first real spike.

We use a Gamma-Poisson mixture (equivalently a negative binomial). Each day we
draw a Gamma-distributed "appetite" and then a Poisson count around it. Some
days the appetite is low, some days it is high, which reproduces the bursts
seen in the data.

On top of that sit two multiplicative calendar effects measured from the real
series: day-of-week (Thursday 1.23, Sunday 0.51) and month (January 0.69
rising to November 1.75).
"""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.config import load_config, resolve


class DemandGenerator:
    """Generates daily demand for every product from calibrated statistics."""

    def __init__(self, stats: dict, product_codes: list[str], rng: np.random.Generator):
        self.rng = rng
        self.codes = product_codes
        self.n = len(product_codes)

        self.mean = np.array([stats["products"][c]["mean_daily_demand"] for c in product_codes])
        self.std = np.array([stats["products"][c]["std_daily_demand"] for c in product_codes])
        self.price = np.array([stats["products"][c]["price"] for c in product_codes])

        # JSON object keys are always strings; the calendar factors were
        # written with integer weekday/month keys, so convert them back.
        self.dow_factor = np.ones(7)
        for k, v in stats["day_of_week_factor"].items():
            self.dow_factor[int(k)] = v
        self.month_factor = np.ones(13)  # 1-indexed; slot 0 unused
        for k, v in stats["month_factor"].items():
            self.month_factor[int(k)] = v

        # Gamma shape parameter r of the negative binomial. Var = m + m^2/r,
        # so r = m^2 / (Var - m). Small r means burstier. If the fitted
        # variance is not greater than the mean (never true for these
        # products, but guard anyway) fall back to near-Poisson behaviour.
        var = self.std**2
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(var > self.mean, self.mean**2 / (var - self.mean), 1e6)
        self.r = np.clip(r, 0.05, 1e6)

        self.zero_fraction = np.array(
            [stats["products"][c]["zero_day_fraction"] for c in product_codes]
        )

    def sample(self, dow: int, month: int, multiplier: np.ndarray | None = None) -> np.ndarray:
        """Draw one day of demand for all products.

        `multiplier` is how disruption scenarios raise or suppress demand; it
        is applied to the mean before sampling, so a spike widens the whole
        distribution rather than merely shifting it.
        """
        lam = self.mean * self.dow_factor[dow] * self.month_factor[month]
        if multiplier is not None:
            lam = lam * multiplier

        # Gamma-Poisson mixture. scale = lam/r keeps the Gamma mean at lam.
        appetite = self.rng.gamma(shape=self.r, scale=lam / self.r)
        demand = self.rng.poisson(appetite).astype(np.float64)
        return demand

    def expected(self, dow: int, month: int) -> np.ndarray:
        """Expected demand with no randomness. Baseline policies and the
        explanation module need this to reason about coverage."""
        return self.mean * self.dow_factor[dow] * self.month_factor[month]


@lru_cache(maxsize=1)
def _read_demand_stats(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data.run_pipeline` first.")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_demand_stats(cfg: dict | None = None) -> dict:
    """Calibrated statistics, parsed once and deep-copied per caller.

    A grid search builds thousands of environments; re-parsing this JSON each
    time dominated the runtime.
    """
    cfg = cfg or load_config("data")
    return copy.deepcopy(_read_demand_stats(str(resolve(cfg["paths"]["demand_stats"]))))
