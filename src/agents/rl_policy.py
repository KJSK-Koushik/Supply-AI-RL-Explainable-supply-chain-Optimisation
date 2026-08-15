"""Wrap a trained SB3 model so it looks like a baseline Policy.

This exists so the RL agent goes through the *same* evaluation harness as the
classical rules -- same seeds, same episode length, same metrics. Scoring the
agent with SB3's internal episode-reward average and the baselines with our own
harness would compare two different quantities and quietly favour whichever was
measured more generously.
"""

from __future__ import annotations

import numpy as np


class RLPolicy:
    name = "PPO"

    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def reset(self) -> None:
        pass

    def act(self, env, rng=None) -> np.ndarray:
        obs = env._build_obs()
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return np.asarray(action).ravel()

    @classmethod
    def load(cls, path: str, deterministic: bool = True) -> RLPolicy:
        from stable_baselines3 import PPO

        return cls(PPO.load(path, device="cpu"), deterministic=deterministic)
