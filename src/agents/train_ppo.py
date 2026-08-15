"""Train the PPO agent.

    python -m src.agents.train_ppo --name run1
    python -m src.agents.train_ppo --name run1 --resume        # after a crash
    python -m src.agents.train_ppo --name lr3 --set ppo.learning_rate=1e-4

Everything is checkpointed. A laptop going to sleep mid-run, or a Kaggle
session hitting its time limit, costs at most the steps since the last
checkpoint rather than the whole run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.config import load_config, resolve


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """--set a.b=value, so a hyperparameter sweep needs no config files."""
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node[parts[-1]] = value
    return cfg


def make_vec_env(n_envs: int, seed: int, monitor_dir: Path | None = None):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from src.env.supply_chain_env import SupplyChainEnv

    def factory(rank: int):
        def _init():
            env = SupplyChainEnv(seed=seed + rank)
            path = str(monitor_dir / f"env{rank}") if monitor_dir else None
            return Monitor(env, filename=path)

        return _init

    fns = [factory(i) for i in range(n_envs)]
    return DummyVecEnv(fns) if n_envs == 1 else SubprocVecEnv(fns)


class HeldOutEvalCallback:
    """Score the policy on fixed seeds using the same harness the baselines
    used, so the numbers are directly comparable rather than being SB3's
    internal episode-reward average."""

    def __init__(self, seeds, every_steps, out_dir: Path, verbose=True):
        from stable_baselines3.common.callbacks import BaseCallback

        self.seeds = seeds
        self.every = every_steps
        self.out_dir = out_dir
        self.verbose = verbose
        self._Base = BaseCallback

    def build(self):
        seeds, every, out_dir, verbose = self.seeds, self.every, self.out_dir, self.verbose

        class _CB(self._Base):
            def __init__(self):
                super().__init__(verbose=0)
                self.best = -np.inf
                self.history = []
                self._next = every

            def _on_step(self) -> bool:
                if self.num_timesteps < self._next:
                    return True
                self._next += every

                from src.agents.rl_policy import RLPolicy
                from src.eval.runner import evaluate

                res = evaluate(RLPolicy(self.model), seeds)
                profit = res["total_profit"]
                self.history.append(
                    {
                        "timesteps": int(self.num_timesteps),
                        "profit": float(profit),
                        "fill_rate": float(res["fill_rate"]),
                    }
                )
                self.logger.record("eval/profit", profit)
                self.logger.record("eval/fill_rate", res["fill_rate"])

                if profit > self.best:
                    self.best = profit
                    self.model.save(str(out_dir / "best_model"))

                if verbose:
                    print(
                        f"  [{self.num_timesteps:>9,}] profit {profit:10,.0f}  "
                        f"fill {res['fill_rate']:5.1%}  best {self.best:10,.0f}",
                        flush=True,
                    )
                with (out_dir / "eval_history.json").open("w", encoding="utf-8") as fh:
                    json.dump(self.history, fh, indent=2)
                return True

        return _CB()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run1", help="run name; results/models/<name>/")
    ap.add_argument("--resume", action="store_true", help="continue from last checkpoint")
    ap.add_argument("--set", nargs="*", default=[], help="override, e.g. ppo.ent_coef=0.02")
    ap.add_argument("--timesteps", type=int, default=None)
    args = ap.parse_args()

    cfg = apply_overrides(load_config("train"), args.set)
    run_cfg, ppo_cfg = cfg["run"], dict(cfg["ppo"])
    if args.timesteps:
        run_cfg["total_timesteps"] = args.timesteps

    # Torch distribution argument validation is pure overhead once the action
    # space is known-good, and was measured at ~9% of total training time.
    if run_cfg.get("disable_torch_validation", True):
        torch.distributions.Distribution.set_default_validate_args(False)
    if run_cfg.get("torch_threads", 0):
        torch.set_num_threads(int(run_cfg["torch_threads"]))

    out_dir = resolve(cfg["paths"]["models"]) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = resolve(cfg["paths"]["logs"]) / args.name
    log_dir.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

    venv = make_vec_env(run_cfg["n_envs"], run_cfg["seed"], log_dir)

    net_arch = ppo_cfg.pop("net_arch", [128, 128])
    policy = ppo_cfg.pop("policy", "MlpPolicy")

    ckpt_path = out_dir / "checkpoint.zip"
    if args.resume and ckpt_path.exists():
        print(f"resuming from {ckpt_path}", flush=True)
        model = PPO.load(str(ckpt_path), env=venv, device=run_cfg["device"])
        done_steps = model.num_timesteps
    else:
        model = PPO(
            policy,
            venv,
            device=run_cfg["device"],
            seed=run_cfg["seed"],
            tensorboard_log=str(resolve(cfg["paths"]["tensorboard"])),
            policy_kwargs={"net_arch": net_arch},
            verbose=0,
            **ppo_cfg,
        )
        done_steps = 0

    remaining = max(run_cfg["total_timesteps"] - done_steps, 0)

    eval_cb = HeldOutEvalCallback(cfg["eval"]["seeds"], cfg["eval"]["every_steps"], out_dir).build()
    ckpt_cb = CheckpointCallback(
        save_freq=max(cfg["checkpoint"]["every_steps"] // run_cfg["n_envs"], 1),
        save_path=str(out_dir),
        name_prefix="checkpoint",
    )

    print(f"run '{args.name}': {remaining:,} steps, {run_cfg['n_envs']} envs", flush=True)
    print(
        f"  net_arch={net_arch}  lr={ppo_cfg.get('learning_rate')}  "
        f"ent_coef={ppo_cfg.get('ent_coef')}  gamma={ppo_cfg.get('gamma')}",
        flush=True,
    )

    t0 = time.time()
    model.learn(
        total_timesteps=remaining,
        callback=CallbackList([eval_cb, ckpt_cb]),
        reset_num_timesteps=not args.resume,
        tb_log_name=args.name,
        progress_bar=False,
    )
    dt = time.time() - t0

    model.save(str(out_dir / "final_model"))
    venv.close()

    summary = {
        "name": args.name,
        "timesteps": int(model.num_timesteps),
        "wall_seconds": dt,
        "steps_per_second": remaining / dt if dt > 0 else 0.0,
        "best_eval_profit": float(eval_cb.best),
        "config": cfg,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print(f"\ndone in {dt / 60:.1f} min ({remaining / max(dt, 1e-9):.0f} steps/s)", flush=True)
    print(f"best held-out profit during training: {eval_cb.best:,.0f}", flush=True)
    print(f"saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
