"""Uniform run-result schema for all benchmark baselines."""
from __future__ import annotations
from typing import Any


RESULT_SCHEMA_KEYS = {
    "preset_id", "seed", "method", "n_blocks",
    "init_slope_deg", "final_slope_deg", "slope_pct",
    "init_contiguity", "final_contiguity", "cont_delta", "cont_pct",
    "init_baimu_count", "final_baimu_count", "baimu_count_delta",
    "init_baimu_area_ha", "final_baimu_area_ha", "baimu_area_delta_ha",
    "total_reward", "wall_seconds", "exchange_pairs_used",
    "extra",
}


def extract_run_result(
    preset_id: str,
    seed: int,
    method: str,
    env,
    init_snap: dict[str, float],
    total_reward: float,
    wall_seconds: float,
    exchange_pairs_used: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the 21-key result dict for a single (preset, seed, method) run.

    `env` must be post-rollout (terminal state). `init_snap` is captured
    immediately after env.reset(), before any action.
    """
    init_slope = float(init_snap["init_slope_deg"])
    init_cont = float(init_snap["init_contiguity"])
    init_baimu_count = int(init_snap["init_baimu_count"])
    init_baimu_area = float(init_snap["init_baimu_area"])

    final_slope = float(env.avg_farmland_slope)
    final_cont = float(env.contiguity)
    # Parcel-level runners (Random/Greedy/GA) bypass env.step(), so env.baimu_*
    # attributes remain at their initial values. Recompute the terminal state.
    final_baimu_count, final_baimu_area_m2 = env._count_baimu_fang()
    env.baimu_count = final_baimu_count
    env.baimu_total_area = final_baimu_area_m2
    final_baimu_count = int(final_baimu_count)
    final_baimu_area = float(final_baimu_area_m2)

    slope_pct = (final_slope - init_slope) / init_slope * 100.0 if init_slope > 0 else 0.0
    cont_delta = final_cont - init_cont
    cont_pct = cont_delta / (abs(init_cont) + 1e-8) * 100.0
    baimu_count_delta = final_baimu_count - init_baimu_count
    baimu_area_delta_ha = (final_baimu_area - init_baimu_area) / 10_000.0

    return {
        "preset_id": preset_id,
        "seed": int(seed),
        "method": method,
        "n_blocks": int(env.n_blocks),
        "init_slope_deg": init_slope,
        "final_slope_deg": final_slope,
        "slope_pct": slope_pct,
        "init_contiguity": init_cont,
        "final_contiguity": final_cont,
        "cont_delta": cont_delta,
        "cont_pct": cont_pct,
        "init_baimu_count": init_baimu_count,
        "final_baimu_count": final_baimu_count,
        "baimu_count_delta": baimu_count_delta,
        "init_baimu_area_ha": init_baimu_area / 10_000.0,
        "final_baimu_area_ha": final_baimu_area / 10_000.0,
        "baimu_area_delta_ha": baimu_area_delta_ha,
        "total_reward": float(total_reward),
        "wall_seconds": float(wall_seconds),
        "exchange_pairs_used": int(exchange_pairs_used),
        "extra": extra or {},
    }
