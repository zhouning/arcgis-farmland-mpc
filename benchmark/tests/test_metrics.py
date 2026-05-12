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
