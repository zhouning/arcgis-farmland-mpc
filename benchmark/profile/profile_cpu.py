"""CPU profile: time generator + Random + Greedy + GA on plain_small_cons.

Writes profile/cpu_profile.json with per-stage wall seconds.
Run from benchmark/ root: python -m profile.profile_cpu
"""
from __future__ import annotations
import json
import sys
import time
import os
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))
D_TEST = Path("D:/test")
if str(D_TEST) not in sys.path:
    sys.path.insert(0, str(D_TEST))


def main():
    from generator.schema import load_preset
    from generator.generate import generate_dataset
    from synthetic_env_loader import make_synthetic_env
    import baselines_county
    import baseline_ga

    results: dict = {}
    out_root = BENCH_ROOT / "profile" / "_tmp_cpu_profile"
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. Generator wall time on plain_small_cons (~800 blocks)
    cfg = load_preset(BENCH_ROOT / "presets" / "plain_small_cons.yaml")
    out = out_root / "plain_small_cons_seed0"
    t0 = time.time()
    manifest = generate_dataset(cfg, seed=0, out_dir=out)
    results["generator_plain_small_s"] = time.time() - t0
    results["generator_n_blocks"] = manifest["n_blocks"]

    # 2. Random baseline
    env = make_synthetic_env(out, total_budget=200, swaps_per_step=5)
    env.reset(seed=0)
    t0 = time.time()
    baselines_county.run_random_block(env, seed=0)
    results["random_plain_small_s"] = time.time() - t0

    # 3. Greedy-Sequential
    env = make_synthetic_env(out, total_budget=200, swaps_per_step=5)
    env.reset(seed=0)
    t0 = time.time()
    baselines_county.run_greedy_sequential(env)
    results["greedy_plain_small_s"] = time.time() - t0

    # 4. GA (50 generations, pop 50, short budget for profile)
    env = make_synthetic_env(out, total_budget=200, swaps_per_step=5)
    env.reset(seed=0)
    t0 = time.time()
    baseline_ga.run_ga(env, max_pairs=100, pop_size=50, generations=50, seed=0, verbose=False)
    results["ga_plain_small_50gen_s"] = time.time() - t0
    results["ga_plain_small_500gen_estimate_s"] = results["ga_plain_small_50gen_s"] * 10

    out_path = BENCH_ROOT / "profile" / "cpu_profile.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
