"""Colab T4 profile: time + GPU util for MPC inner loop (K=50 candidates).

How to run on Colab:
1. Mount Drive, clone arcgis-farmland-mpc into /content/repo
2. Generate one plain_small_cons dataset locally and upload to /content/data
   (or generate inside Colab; ~30s for 800 blocks)
3. !python /content/repo/benchmark/profiling/profile_gpu_mpc.py /content/data

Outputs /content/repo/benchmark/profiling/gpu_mpc_profile.json + side-channel
nvidia-smi dmon log /content/gpu_dmon.log.
"""
from __future__ import annotations
import json
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
    sys.path.insert(0, "/content/repo")
    sys.path.insert(0, str(Path(dataset_dir).resolve().parent / "test_src"))

    from synthetic_env_loader import make_synthetic_env
    from paper9_contrastive.mpc_planner import mpc_select_action
    from data_agent.transition_model import TransitionModel, EnsembleTransitionModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_synthetic_env(dataset_dir, total_budget=100, swaps_per_step=5)
    bf_dim = env._get_block_features().shape[-1]
    gf_dim = env._get_global_features().shape[-1]
    n_blocks = env.n_blocks

    models = []
    for i in range(3):  # 3-member ensemble matching paper 9
        torch.manual_seed(i)
        m = TransitionModel(n_blocks=n_blocks, block_feat_dim=bf_dim,
                            global_feat_dim=gf_dim).to(device)
        models.append(m)
    ensemble = EnsembleTransitionModel(models)

    env.reset(seed=0)
    rng = np.random.default_rng(0)

    # Warm-up so CUDA kernels JIT
    for _ in range(n_steps_warm):
        bf = env._get_block_features()
        gf = env._get_global_features()
        mask = env.action_masks()
        action, _ = mpc_select_action(ensemble, bf, gf, mask,
                                      horizon=5, top_k=50, gamma=0.99,
                                      n_rollouts=1, continuation="greedy",
                                      greedy_sample=50, scoring="reward", rng=rng)
        env.step(action)

    # Start nvidia-smi dmon in background
    dmon_log = Path("/content/gpu_dmon.log")
    dmon_proc = subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "u", "-c", str(n_steps_measure + 5), "-o", "T"],
        stdout=open(dmon_log, "w"))

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
    dmon_proc.wait(timeout=30)

    # Parse nvidia-smi dmon output (cols: gpu sm mem enc dec)
    util = []
    for line in dmon_log.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        toks = line.split()
        try:
            util.append(int(toks[2]))  # sm column
        except (IndexError, ValueError):
            continue
    mean_util = float(np.mean(util)) if util else float("nan")

    return {
        "device": device,
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
