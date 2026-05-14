"""Run the centralized PPO baseline on a synthetic dataset.

Production: total_timesteps=100_000, device='cuda' (per Decision Gate result).
Smoke: total_timesteps=500, device='cpu'.

Bumped from 25k to 100k after the first Task 15 attempt: at 25k, 32/35 cells
collapsed to a uniform action distribution and crashed in MaskableCategorical
Simplex check. Paper 4 production uses 500k; 100k is the compromise that
clears the entropy-collapse failure mode without paying the 41-day cost of
a full-length run on every benchmark cell.

Uses `ParcelScoringPolicy` (the Paper 1-4 scorer-based policy) -- not the
generic `MaskableActorCriticPolicy`. Mirrors `train_county.train_county`
construction so the resulting baseline is comparable to Paper 4 numbers.
"""
from __future__ import annotations
import time
from pathlib import Path

from .common import build_env, init_snapshot, write_result_json


def run_ppo(
    dataset_dir,
    preset_id: str,
    seed: int,
    out_path,
    total_budget: int = 100,
    swaps_per_step: int = 5,
    total_timesteps: int = 100_000,
    device: str = "auto",
    n_steps: int = 256,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    ent_coef: float = 0.005,
) -> dict:
    import sys
    sys.path.insert(0, "D:/test")
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.monitor import Monitor
    from county_env import K_BLOCK, K_GLOBAL_COUNTY
    from parcel_scoring_policy import ParcelScoringPolicy
    from eval.metrics import extract_run_result

    env = build_env(dataset_dir, total_budget=total_budget,
                    swaps_per_step=swaps_per_step)
    monitored_env = Monitor(env)

    model = MaskablePPO(
        ParcelScoringPolicy,
        monitored_env,
        learning_rate=learning_rate,
        n_steps=n_steps, batch_size=batch_size, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=ent_coef, vf_coef=0.5, max_grad_norm=0.5,
        seed=seed, device=device, verbose=0,
        policy_kwargs=dict(
            k_parcel=K_BLOCK,
            k_global=K_GLOBAL_COUNTY,
            scorer_hiddens=[128, 64],
            value_hiddens=[64, 32],
        ),
    )

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    train_wall = time.time() - t0

    # Eval rollout: reset once, snapshot initial state, greedy rollout.
    obs, _ = env.reset(seed=seed)
    init = init_snapshot(env)
    total_reward = 0.0
    eval_t0 = time.time()
    for _ in range(env.max_steps):
        action_masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=action_masks,
                                   deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    eval_wall = time.time() - eval_t0

    result = extract_run_result(
        preset_id=preset_id, seed=seed, method="PPO-Centralized",
        env=env, init_snap=init,
        total_reward=total_reward,
        wall_seconds=train_wall + eval_wall,
        exchange_pairs_used=int(env.budget_used),
        extra={
            "total_timesteps": total_timesteps,
            "train_wall_seconds": train_wall,
            "eval_wall_seconds": eval_wall,
            "device": device,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "policy": "ParcelScoringPolicy",
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
        },
    )
    write_result_json(result, out_path)
    return result
