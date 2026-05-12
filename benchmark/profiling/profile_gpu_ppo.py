"""Colab T4 profile: PPO 25k timesteps wall + GPU util on plain_small_cons.

Run on Colab T4 the same way as profile_gpu_mpc.py.
"""
from __future__ import annotations
import json
import sys
import time
import subprocess
from pathlib import Path

import torch
import numpy as np


def measure_ppo(dataset_dir: str, total_timesteps: int = 25_000):
    BENCH_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(BENCH_ROOT))
    sys.path.insert(0, "/content/repo")

    from synthetic_env_loader import make_synthetic_env
    from stable_baselines3 import PPO
    from sb3_contrib import MaskablePPO  # used by train_county.py
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_synthetic_env(dataset_dir, total_budget=100, swaps_per_step=5)

    model = MaskablePPO(
        MaskableActorCriticPolicy,
        env, n_steps=256, batch_size=128, learning_rate=3e-4,
        device=device, verbose=0,
    )

    dmon_log = Path("/content/gpu_dmon_ppo.log")
    dmon_proc = subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "u", "-c", "120", "-o", "T"],
        stdout=open(dmon_log, "w"))

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    elapsed = time.time() - t0
    dmon_proc.wait(timeout=30)

    util = []
    for line in dmon_log.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        toks = line.split()
        try:
            util.append(int(toks[2]))
        except (IndexError, ValueError):
            continue
    mean_util = float(np.mean(util)) if util else float("nan")

    return {
        "device": device,
        "total_timesteps": total_timesteps,
        "wall_s": elapsed,
        "steps_per_s": total_timesteps / elapsed,
        "gpu_sm_util_mean_pct": mean_util,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python profile_gpu_ppo.py <dataset_dir>")
        sys.exit(2)
    res = measure_ppo(sys.argv[1])
    out = Path(__file__).resolve().parent / "gpu_ppo_profile.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
