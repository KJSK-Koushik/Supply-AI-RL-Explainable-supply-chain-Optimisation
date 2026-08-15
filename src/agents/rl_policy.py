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
        # A masked model must be given the same mask at evaluation time it saw
        # during training; without it the policy can pick an action it was
        # never trained to consider, and the reported score would not reflect
        # the trained behaviour.
        if self._is_maskable():
            action, _ = self.model.predict(
                obs, deterministic=self.deterministic, action_masks=env.action_masks()
            )
        else:
            action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return np.asarray(action).ravel()

    def _is_maskable(self) -> bool:
        return type(self.model).__name__ == "MaskablePPO"

    @classmethod
    def load(cls, path: str, deterministic: bool = True, maskable: bool | None = None) -> RLPolicy:
        """Load a saved model. `maskable` is auto-detected when not given, so
        callers do not have to remember which algorithm produced a file."""
        if maskable is None:
            try:
                from sb3_contrib import MaskablePPO

                return cls(MaskablePPO.load(path, device="cpu"), deterministic=deterministic)
            except Exception:
                maskable = False
        if maskable:
            from sb3_contrib import MaskablePPO

            return cls(MaskablePPO.load(path, device="cpu"), deterministic=deterministic)
        from stable_baselines3 import PPO

        return cls(PPO.load(path, device="cpu"), deterministic=deterministic)
