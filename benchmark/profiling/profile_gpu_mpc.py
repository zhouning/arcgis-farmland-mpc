"""Colab T4 profile: time + GPU util for MPC inner loop (K=50 candidates).

How to run on Colab:
1. Upload zip bundle (repo subset + D:/test sources), extract into /content/src
2. Generate one plain_small_cons dataset inside Colab (~30s for 800 blocks)
3. !python /content/src/benchmark/profiling/profile_gpu_mpc.py /content/data

Outputs <benchmark>/profiling/gpu_mpc_profile.json + side-channel
nvidia-smi dmon log /content/gpu_dmon.log.

Source-location assumptions (match baselines/run_mpc.py, verified 2026-05-13):
- mpc_planner.py sits at D:/test top level, surfaced on sys.path
- contrastive_trainer.py sits at D:/test/paper9_contrastive, surfaced on sys.path
- data_agent.transition_model lives at D:/adk/data_agent
When bundled for Colab, extraction should set:
    TEST_SRC_ROOT pointing at the copied D:/test tree
    ADK_SRC_ROOT  pointing at the copied D:/adk tree
so the sys.path inserts below resolve correctly.
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


def measure_mpc(dataset_dir: str, n_steps_warm: int = 10, n_steps_measure: int = 50):
    """Run MPC inner loop on real env; report timings + GPU util."""
    BENCH_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(BENCH_ROOT))

    test_src = Path(os.environ.get("TEST_SRC_ROOT", "/content/src/test"))
    adk_src = Path(os.environ.get("ADK_SRC_ROOT", "/content/src/adk"))
    sys.path.insert(0, str(test_src))
    sys.path.insert(0, str(test_src / "paper9_contrastive"))
    sys.path.insert(0, str(adk_src))

    from synthetic_env_loader import make_synthetic_env
    from mpc_planner import mpc_select_action
    from data_agent.transition_model import TransitionModel, EnsembleTransitionModel

    # NOTE: mpc_planner does not move inputs to model device -- production
    # baselines/run_mpc.py runs everything on CPU for this reason. We keep
    # the ensemble on CPU regardless of cuda availability so this profile
    # faithfully matches what Task 15 MPC sweep will actually run.
    cuda_available = torch.cuda.is_available()
    device = "cpu"
    env = make_synthetic_env(dataset_dir, total_budget=100, swaps_per_step=5)
    n_blocks = env.n_blocks

    ensemble = EnsembleTransitionModel(n_blocks, n_models=3)
    for i in range(3):
        torch.manual_seed(i)
        m = TransitionModel(n_blocks)
        ensemble.models[i] = m

    env.reset(seed=0)
    rng = np.random.default_rng(0)

    for _ in range(n_steps_warm):
        bf = env._get_block_features()
        gf = env._get_global_features()
        mask = env.action_masks()
        action, _ = mpc_select_action(ensemble, bf, gf, mask,
                                      horizon=5, top_k=50, gamma=0.99,
                                      n_rollouts=1, continuation="greedy",
                                      greedy_sample=50, scoring="reward", rng=rng)
        env.step(action)

    dmon_log_path = os.environ.get("GPU_DMON_LOG", "/content/gpu_dmon.log")
    dmon_proc = None
    dmon_log_file = None
    try:
        dmon_log_file = open(dmon_log_path, "w")
        dmon_proc = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "u", "-c", str(n_steps_measure + 5), "-o", "T"],
            stdout=dmon_log_file,
        )
    except (FileNotFoundError, OSError):
        if dmon_log_file is not None:
            dmon_log_file.close()
        dmon_log_file = None
        dmon_proc = None

    t0 = time.time()
    for _ in range(n_steps_measure):
        bf = env._get_block_features()
        gf = env._get_global_features()
        mask = env.action_masks()
        action, _ = mpc_select_action(ensemble, bf, gf, mask,
                                      horizon=5, top_k=50, gamma=0.99,
                                      n_rollouts=1, continuation="greedy",
                                      greedy_sample=50, scoring="reward", rng=rng)
        env.step(action)
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
        "cuda_available": cuda_available,
        "note": "mpc_planner lacks device propagation; ensemble forced to CPU to match production run_mpc.py",
        "n_steps_measured": n_steps_measure,
        "wall_s": elapsed,
        "s_per_step": elapsed / n_steps_measure,
        "gpu_sm_util_mean_pct": mean_util,
        "ensemble_members": 3,
        "horizon": 5,
        "top_k": 50,
        "n_blocks": n_blocks,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python profile_gpu_mpc.py <dataset_dir>")
        sys.exit(2)
    res = measure_mpc(sys.argv[1])
    out = Path(__file__).resolve().parent / "gpu_mpc_profile.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
