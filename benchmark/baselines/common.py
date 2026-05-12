"""Shared helpers: env build, init snapshot, result write."""
from __future__ import annotations
import json
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))
D_TEST = Path("D:/test")
if str(D_TEST) not in sys.path:
    sys.path.insert(0, str(D_TEST))


def build_env(dataset_dir, total_budget=100, swaps_per_step=5):
    from synthetic_env_loader import make_synthetic_env
    return make_synthetic_env(dataset_dir, total_budget=total_budget,
                              swaps_per_step=swaps_per_step)


def init_snapshot(env) -> dict:
    """Call immediately after env.reset(...) and before any step()."""
    return {
        "init_slope_deg": float(env.avg_farmland_slope),
        "init_contiguity": float(env.contiguity),
        "init_baimu_count": int(env.baimu_count),
        "init_baimu_area": float(env.baimu_total_area),
    }


def write_result_json(result: dict, out_path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write
    tmp = out_path.with_suffix(out_path.suffix + ".partial.json")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(out_path)
