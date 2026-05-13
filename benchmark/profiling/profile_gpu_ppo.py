"""Colab T4 profile: PPO 25k timesteps wall + GPU util on plain_small_cons.

How to run on Colab (identical setup to profile_gpu_mpc.py):
1. Upload zip bundle (repo subset + D:/test sources), extract into /content/src
2. Generate one plain_small_cons dataset inside Colab (~30s for 800 blocks)
3. !python /content/src/benchmark/profiling/profile_gpu_ppo.py /content/data

Uses the same ParcelScoringPolicy as baselines/run_ppo.py so the wall time is
a faithful predictor of Task 15 throughput. Set TEST_SRC_ROOT in env to point
at the extracted D:/test copy.
"""
from __future__ import annotations
import json
import os
import sys
import time
import subprocess
from pathlib import Path

import torch
import numpy as np


def measure_ppo(dataset_dir: str, total_timesteps: int = 25_000):
    BENCH_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(BENCH_ROOT))

    test_src = Path(os.environ.get("TEST_SRC_ROOT", "/content/src/test"))
    sys.path.insert(0, str(test_src))

    from synthetic_env_loader import make_synthetic_env
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.monitor import Monitor
    from county_env import K_BLOCK, K_GLOBAL_COUNTY
    from parcel_scoring_policy import ParcelScoringPolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_synthetic_env(dataset_dir, total_budget=100, swaps_per_step=5)
    monitored_env = Monitor(env)

    model = MaskablePPO(
        ParcelScoringPolicy,
        monitored_env,
        learning_rate=1e-3,
        n_steps=256, batch_size=128, n_epochs=10,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5,
        seed=0, device=device, verbose=0,
        policy_kwargs=dict(
            k_parcel=K_BLOCK,
            k_global=K_GLOBAL_COUNTY,
            scorer_hiddens=[128, 64],
            value_hiddens=[64, 32],
        ),
    )

    dmon_log_path = os.environ.get("GPU_DMON_LOG_PPO", "/content/gpu_dmon_ppo.log")
    dmon_proc = None
    dmon_log_file = None
    try:
        dmon_log_file = open(dmon_log_path, "w")
        dmon_proc = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "u", "-c", "120", "-o", "T"],
            stdout=dmon_log_file,
        )
    except (FileNotFoundError, OSError):
        if dmon_log_file is not None:
            dmon_log_file.close()
        dmon_log_file = None
        dmon_proc = None

    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    elapsed = time.time() - t0
    if dmon_proc is not None:
        dmon_proc.wait(timeout=30)
    if dmon_log_file is not None:
        dmon_log_file.close()

    util = []
    if dmon_proc is not None:
        try:
            for line in Path(dmon_log_path).read_text().splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                toks = line.split()
                try:
                    util.append(int(toks[2]))
                except (IndexError, ValueError):
                    continue
        except FileNotFoundError:
            pass
    mean_util = float(np.mean(util)) if util else float("nan")

    return {
        "device": device,
        "total_timesteps": total_timesteps,
        "wall_s": elapsed,
        "steps_per_s": total_timesteps / elapsed,
        "gpu_sm_util_mean_pct": mean_util,
        "policy": "ParcelScoringPolicy",
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python profile_gpu_ppo.py <dataset_dir>")
        sys.exit(2)
    res = measure_ppo(sys.argv[1])
    out = Path(__file__).resolve().parent / "gpu_ppo_profile.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))