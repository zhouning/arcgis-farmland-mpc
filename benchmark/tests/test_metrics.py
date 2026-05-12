import json
import sys
from pathlib import Path
import pytest

BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_ROOT))
sys.path.insert(0, "D:/test")

from eval.metrics import extract_run_result, RESULT_SCHEMA_KEYS


def test_result_schema_keys_are_complete():
    expected = {
        "preset_id", "seed", "method", "n_blocks",
        "init_slope_deg", "final_slope_deg", "slope_pct",
        "init_contiguity", "final_contiguity", "cont_delta", "cont_pct",
        "init_baimu_count", "final_baimu_count", "baimu_count_delta",
        "init_baimu_area_ha", "final_baimu_area_ha", "baimu_area_delta_ha",
        "total_reward", "wall_seconds", "exchange_pairs_used",
        "extra",
    }
    assert RESULT_SCHEMA_KEYS == expected


def test_extract_run_result_returns_all_keys(tmp_path):
    # Build a tiny synthetic dataset, run env one step, extract
    from generator.schema import load_preset
    from generator.generate import generate_dataset
    cfg = load_preset(BENCH_ROOT / "presets" / "plain_small_cons.yaml")
    cfg.n_blocks_target = 60
    cfg.parcels.parcels_per_block_mean = 6
    cfg.parcels.parcels_per_block_std = 2
    generate_dataset(cfg, seed=0, out_dir=tmp_path)
    from synthetic_env_loader import make_synthetic_env
    env = make_synthetic_env(tmp_path, total_budget=20, swaps_per_step=2)
    env.reset(seed=0)
    init_snap = {
        "init_slope_deg": float(env.avg_farmland_slope),
        "init_contiguity": float(env.contiguity),
        "init_baimu_count": int(env.baimu_count),
        "init_baimu_area": float(env.baimu_total_area),
    }
    # one random step
    action = env.action_space.sample()
    env.step(action)
    result = extract_run_result(
        preset_id="plain_small_cons", seed=0, method="Random",
        env=env, init_snap=init_snap,
        total_reward=0.5, wall_seconds=0.1, exchange_pairs_used=1,
        extra={"note": "smoke"},
    )
    assert set(result.keys()) == RESULT_SCHEMA_KEYS
    # JSON round-trip
    s = json.dumps(result)
    back = json.loads(s)
    assert back["preset_id"] == "plain_small_cons"
    assert back["extra"]["note"] == "smoke"


def test_extract_run_result_refreshes_stale_baimu(tmp_path):
    """Regression: runners that bypass env.step() (Random/Greedy/GA) leave
    env.baimu_count stale because parcel-level _swap_to_forest/_swap_to_farmland
    do not update the baimu attributes. extract_run_result must refresh them
    from _count_baimu_fang() before reading final_baimu_*."""
    from generator.schema import load_preset
    from generator.generate import generate_dataset
    cfg = load_preset(BENCH_ROOT / "presets" / "plain_small_cons.yaml")
    cfg.n_blocks_target = 60
    cfg.parcels.parcels_per_block_mean = 6
    cfg.parcels.parcels_per_block_std = 2
    generate_dataset(cfg, seed=0, out_dir=tmp_path)
    from synthetic_env_loader import make_synthetic_env
    import numpy as np

    env = make_synthetic_env(tmp_path, total_budget=20, swaps_per_step=2)
    env.reset(seed=0)
    init_snap = {
        "init_slope_deg": float(env.avg_farmland_slope),
        "init_contiguity": float(env.contiguity),
        "init_baimu_count": int(env.baimu_count),
        "init_baimu_area": float(env.baimu_total_area),
    }
    # Mutate env via parcel-level helpers (mirrors Random/Greedy/GA path).
    FARMLAND, FOREST = 1, 2
    farm_idx = np.where(env.land_use == FARMLAND)[0]
    forest_idx = np.where(env.land_use == FOREST)[0]
    n_swap = min(10, len(farm_idx), len(forest_idx))
    for i in range(n_swap):
        env._swap_to_forest(int(farm_idx[i]))
        env._swap_to_farmland(int(forest_idx[i]))
    # Confirm the stale-attribute condition holds pre-extract:
    true_final_count, true_final_area = env._count_baimu_fang()
    assert env.baimu_count == init_snap["init_baimu_count"], \
        "prerequisite: parcel-level swaps must leave env.baimu_count stale"

    result = extract_run_result(
        preset_id="plain_small_cons", seed=0, method="Random-Block",
        env=env, init_snap=init_snap,
        total_reward=0.0, wall_seconds=0.01, exchange_pairs_used=n_swap,
    )
    assert result["final_baimu_count"] == true_final_count
    assert abs(result["final_baimu_area_ha"] - true_final_area / 10_000.0) < 1e-6
