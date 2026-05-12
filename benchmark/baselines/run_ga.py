"""Run the Genetic Algorithm baseline on a synthetic dataset.

Production defaults match Paper 2 (`pop_size=100, generations=500`).

Notes on the underlying `baseline_ga.run_ga` contract (verified 2026-05-13):
- Returns `(best_results, ga_info)`.
- `best_results` is the dict built by `_simulate_best_solution`; it holds
  step-by-step trajectory info (`step_slopes`, `completed_pairs`, ...) plus
  `total_reward=0.0` and `method='Genetic Algorithm'`. It does not contain
  the encoded individual.
- `_simulate_best_solution` already applies the winning individual to env
  via `env._swap_to_forest` / `env._swap_to_farmland`, so env is in the
  terminal state when `run_ga` returns. Baimu attributes are refreshed
  inside `extract_run_result`.
- Real fitness value lives at `ga_info["best_fitness"]`.
"""
from __future__ import annotations
import time
from pathlib import Path

from .common import build_env, init_snapshot, write_result_json


def run_ga_baseline(
    dataset_dir,
    preset_id: str,
    seed: int,
    out_path,
    total_budget: int = 100,
    swaps_per_step: int = 5,
    pop_size: int = 100,
    generations: int = 500,
) -> dict:
    import sys
    sys.path.insert(0, "D:/test")
    import baseline_ga
    from eval.metrics import extract_run_result

    env = build_env(dataset_dir, total_budget=total_budget,
                    swaps_per_step=swaps_per_step)
    env.reset(seed=seed)
    init = init_snapshot(env)
    t0 = time.time()
    best_results, ga_info = baseline_ga.run_ga(
        env, max_pairs=total_budget,
        pop_size=pop_size, generations=generations,
        seed=seed, verbose=False,
    )
    wall = time.time() - t0

    result = extract_run_result(
        preset_id=preset_id, seed=seed, method="GA",
        env=env, init_snap=init,
        total_reward=float(ga_info.get("best_fitness", 0.0)),
        wall_seconds=wall,
        exchange_pairs_used=int(best_results.get("completed_pairs", total_budget)),
        extra={
            "pop_size": pop_size,
            "generations": generations,
            "best_fitness": float(ga_info.get("best_fitness", 0.0)),
            "ga_elapsed_seconds": float(ga_info.get("elapsed_seconds", wall)),
            "ga_final_avg_slope": float(best_results.get("final_avg_slope", 0.0)),
            "ga_final_contiguity": float(best_results.get("final_contiguity", 0.0)),
        },
    )
    write_result_json(result, out_path)
    return result
