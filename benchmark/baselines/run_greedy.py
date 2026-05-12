"""Run the Greedy-Sequential baseline on a synthetic dataset."""
from __future__ import annotations
import time
from pathlib import Path

from .common import build_env, init_snapshot, write_result_json


def run_greedy(
    dataset_dir,
    preset_id: str,
    seed: int,
    out_path,
    total_budget: int = 100,
    swaps_per_step: int = 5,
) -> dict:
    import sys
    sys.path.insert(0, "D:/test")
    import baselines_county
    from eval.metrics import extract_run_result

    env = build_env(dataset_dir, total_budget=total_budget,
                    swaps_per_step=swaps_per_step)
    env.reset(seed=seed)
    init = init_snapshot(env)
    t0 = time.time()
    res_dict = baselines_county.run_greedy_sequential(env)
    wall = time.time() - t0

    result = extract_run_result(
        preset_id=preset_id, seed=seed, method="Greedy-Sequential",
        env=env, init_snap=init,
        total_reward=float(res_dict.get("total_reward", 0.0)),
        wall_seconds=wall,
        exchange_pairs_used=int(res_dict.get("exchange_pairs_used",
                                             total_budget)),
        extra={"baseline_returned": {k: float(v) for k, v in res_dict.items()
                                     if isinstance(v, (int, float))}},
    )
    write_result_json(result, out_path)
    return result
