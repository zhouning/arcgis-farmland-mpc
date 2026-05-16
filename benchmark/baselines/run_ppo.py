"""Run the centralized PPO baseline on a synthetic dataset.

Production: total_timesteps=100_000, device='cuda' (per Decision Gate result).
Smoke: total_timesteps=500, device='cpu'.

Bumped from 25k to 100k after the first Task 15 attempt: at 25k, 32/35 cells
collapsed to a uniform action distribution and crashed in MaskableCategorical
Simplex check. Paper 4 production uses 500k; 100k is the compromise that
clears the entropy-collapse failure mode without paying the 41-day cost of
a full-length run on every benchmark cell.

Uses `StableParcelScoringPolicy` (a benchmark-only subclass of the Paper 1-4
production `ParcelScoringPolicy`) that clamps raw logits to [-50, 50] before
softmax. Without the clamp, training on synthetic 2600+ block envs produces
NaN logits within the first 1000 gradient steps. See stable_policy.py for
why we don't ship this fix back to D:/test/parcel_scoring_policy.py.

Checkpoint design (added 2026-05-16 after Colab plain_large_cons cell ran
6+ hours and got killed at the 24h VM limit before training finished):

- Save model every CKPT_EVERY env steps to <out_path.parent>/_ppo_ckpt/
  {preset_id}_seed{seed}/ppo_<num_timesteps>_steps.zip
- On resume, load the highest-numbered checkpoint and continue training
  with reset_num_timesteps=False so num_timesteps picks up where it left
- After result JSON written, delete the entire ckpt dir
"""
from __future__ import annotations
import time
from pathlib import Path

from .common import build_env, init_snapshot, write_result_json


CKPT_EVERY = 10_000  # save every 10k env steps -> 10 saves over 100k


def _latest_ppo_ckpt(ckpt_dir: Path) -> Path | None:
    """Return path of highest-numbered ppo_<N>_steps.zip, or None."""
    if not ckpt_dir.is_dir():
        return None
    candidates = []
    for p in ckpt_dir.glob("ppo_*_steps.zip"):
        try:
            n = int(p.stem.split("_")[1])
            candidates.append((n, p))
        except (ValueError, IndexError):
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


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
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CheckpointCallback
    from county_env import K_BLOCK, K_GLOBAL_COUNTY
    from .stable_policy import StableParcelScoringPolicy
    from eval.metrics import extract_run_result

    out_path = Path(out_path)
    ckpt_dir = out_path.parent / "_ppo_ckpt" / f"{preset_id}_seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(dataset_dir, total_budget=total_budget,
                    swaps_per_step=swaps_per_step)
    monitored_env = Monitor(env)

    policy_kwargs = dict(
        k_parcel=K_BLOCK,
        k_global=K_GLOBAL_COUNTY,
        scorer_hiddens=[128, 64],
        value_hiddens=[64, 32],
    )

    latest_ckpt = _latest_ppo_ckpt(ckpt_dir)
    if latest_ckpt is not None:
        print(f"[ppo] resuming from {latest_ckpt.name}", flush=True)
        model = MaskablePPO.load(
            str(latest_ckpt),
            env=monitored_env,
            device=device,
            custom_objects={"policy_class": StableParcelScoringPolicy},
        )
        steps_remaining = total_timesteps - model.num_timesteps
        print(f"[ppo] resumed at num_timesteps={model.num_timesteps}, "
              f"remaining={steps_remaining}", flush=True)
    else:
        model = MaskablePPO(
            StableParcelScoringPolicy,
            monitored_env,
            learning_rate=learning_rate,
            n_steps=n_steps, batch_size=batch_size, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=ent_coef, vf_coef=0.5, max_grad_norm=0.5,
            seed=seed, device=device, verbose=0,
            policy_kwargs=policy_kwargs,
        )
        steps_remaining = total_timesteps

    callback = CheckpointCallback(
        save_freq=CKPT_EVERY,
        save_path=str(ckpt_dir),
        name_prefix="ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    t0 = time.time()
    if steps_remaining > 0:
        model.learn(
            total_timesteps=steps_remaining,
            progress_bar=False,
            callback=callback,
            reset_num_timesteps=False if latest_ckpt is not None else True,
        )
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
            "policy": "StableParcelScoringPolicy",
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
        },
    )
    write_result_json(result, out_path)

    # Clean up checkpoint dir once result is safely on disk
    if ckpt_dir.is_dir():
        for f in ckpt_dir.iterdir():
            try:
                f.unlink()
            except OSError:
                pass
        try:
            ckpt_dir.rmdir()
        except OSError:
            pass

    return result
