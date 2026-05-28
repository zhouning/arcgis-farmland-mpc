#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backfill 105-cell Random/Greedy/GA sweep on local machine.

Why this exists
---------------
Paper 9 v7 §sec:bench Table 5 cites Random/Greedy/GA numbers for 7 presets ×
5 seeds = 105 cells, but the raw seed JSONs were lost between Colab runs and
the original sweep_state.json points at non-existent paths. PPO+MPC's 70
cells survived in G:/我的云端硬盘/arcgis-farmland-mpc/sweep_run/results/.

This runner regenerates the 105 missing cells by calling
baselines_county.{run_random_block, run_greedy_sequential} and
baseline_ga.run_ga directly on each synthetic preset's CountyLevelEnv.

Estimated wall time
-------------------
- Random: ~30 s/cell × 35 = ~18 min
- Greedy: ~60 s/cell × 35 = ~35 min
- GA (pop=100, gen=500): ~150 s avg/cell × 35 = ~90 min
Total: roughly 2 h 20 min on a typical desktop CPU. Outputs are written
incrementally; if interrupted, cells already written are skipped on rerun.

Usage
-----
    python verification/run_baselines_local.py \
        --datasets D:/test/_publish/arcgis-farmland-mpc/benchmark/data_dev \
        --out      G:/我的云端硬盘/arcgis-farmland-mpc/sweep_run/results

Outputs
-------
    <out>/<preset>/<method>/seed<n>.json    (one per cell, schema matches PPO/MPC)

Re-run with --methods Random Greedy if you want to skip GA temporarily.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap: make the repo's farmland_mpc package importable, plus the
# legacy D:/test/ directory where baselines_county.py + baseline_ga.py live.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
LEGACY_TEST_ROOT = Path(os.environ.get("LEGACY_TEST_ROOT", "D:/test"))

for p in (REPO_ROOT, REPO_ROOT / "benchmark", LEGACY_TEST_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ---------------------------------------------------------------------------
# Patched env factory so synthetic_env_loader works against farmland_mpc's
# CountyLevelEnv (the exact same class CountyLevelEnv uses in deployment).
# ---------------------------------------------------------------------------
def make_synthetic_env(dataset_dir: Path, total_budget: int = 100,
                       swaps_per_step: int = 5):
    dataset_dir = Path(dataset_dir).resolve()
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    n_townships = int(manifest["n_townships"])
    county_code = "999999"
    township_codes = {
        f"{county_code}{i + 1:03d}": f"S{i + 1:02d}"
        for i in range(n_townships)
    }

    # Patch farmland_mpc.county_env globals so CountyLevelEnv reads from this
    # synthetic dataset rather than the production DLTB path.
    from farmland_mpc import county_env
    county_env.DLTB_PATH = str(dataset_dir / "DLTB_with_slope.gpkg")
    county_env.BLOCK_DIR = str(dataset_dir)
    county_env.ALL_TOWNSHIPS = dict(township_codes)
    county_env.TOWNSHIP_CODES = sorted(township_codes.keys())
    county_env.PROJ_CRS = "EPSG:4523"

    return county_env.CountyLevelEnv(total_budget=total_budget,
                                     swaps_per_step=swaps_per_step)


# ---------------------------------------------------------------------------
# Result schema (matches benchmark/eval/metrics.py)
# ---------------------------------------------------------------------------
def init_snapshot(env) -> dict:
    return {
        "init_slope_deg": float(env.avg_farmland_slope),
        "init_contiguity": float(env.contiguity),
        "init_baimu_count": int(env.baimu_count),
        "init_baimu_area": float(env.baimu_total_area),
    }


def build_result(preset_id: str, seed: int, method: str, env, init_snap: dict,
                 total_reward: float, wall_seconds: float,
                 exchange_pairs_used: int, extra: dict | None = None) -> dict:
    init_slope = init_snap["init_slope_deg"]
    init_cont = init_snap["init_contiguity"]
    init_baimu_count = init_snap["init_baimu_count"]
    init_baimu_area = init_snap["init_baimu_area"]

    final_slope = float(env.avg_farmland_slope)
    final_cont = float(env.contiguity)
    final_baimu_count, final_baimu_area_m2 = env._count_baimu_fang()
    env.baimu_count = int(final_baimu_count)
    env.baimu_total_area = float(final_baimu_area_m2)

    slope_pct = (final_slope - init_slope) / init_slope * 100.0 if init_slope > 0 else 0.0
    cont_delta = final_cont - init_cont
    cont_pct = cont_delta / (abs(init_cont) + 1e-8) * 100.0
    baimu_count_delta = int(final_baimu_count) - init_baimu_count
    baimu_area_delta_ha = (float(final_baimu_area_m2) - init_baimu_area) / 10_000.0

    return {
        "preset_id": preset_id,
        "seed": int(seed),
        "method": method,
        "n_blocks": int(env.n_blocks),
        "init_slope_deg": float(init_slope),
        "final_slope_deg": final_slope,
        "slope_pct": slope_pct,
        "init_contiguity": float(init_cont),
        "final_contiguity": final_cont,
        "cont_delta": cont_delta,
        "cont_pct": cont_pct,
        "init_baimu_count": init_baimu_count,
        "final_baimu_count": int(final_baimu_count),
        "baimu_count_delta": baimu_count_delta,
        "init_baimu_area_ha": init_baimu_area / 10_000.0,
        "final_baimu_area_ha": float(final_baimu_area_m2) / 10_000.0,
        "baimu_area_delta_ha": baimu_area_delta_ha,
        "total_reward": float(total_reward),
        "wall_seconds": float(wall_seconds),
        "exchange_pairs_used": int(exchange_pairs_used),
        "extra": extra or {},
    }


def write_json_atomic(result: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".partial.json")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    tmp.replace(out_path)


# ---------------------------------------------------------------------------
# Method dispatchers — each returns (result_dict_extra, total_reward, pairs_used)
# ---------------------------------------------------------------------------
def run_random(env, seed: int):
    import baselines_county
    t0 = time.time()
    res = baselines_county.run_random_block(env, seed=seed)
    return ({"baseline_returned": {k: float(v) for k, v in res.items()
                                   if isinstance(v, (int, float))}},
            float(res.get("total_reward", 0.0)),
            int(res.get("exchange_pairs_used", env.total_budget)),
            time.time() - t0)


def run_greedy(env, seed: int):
    import baselines_county
    t0 = time.time()
    res = baselines_county.run_greedy_sequential(env)
    return ({"baseline_returned": {k: float(v) for k, v in res.items()
                                   if isinstance(v, (int, float))}},
            float(res.get("total_reward", 0.0)),
            int(res.get("exchange_pairs_used", env.total_budget)),
            time.time() - t0)


def run_ga(env, seed: int, pop_size: int = 100, generations: int = 500):
    import baseline_ga
    t0 = time.time()
    best_results, ga_info = baseline_ga.run_ga(
        env, max_pairs=env.total_budget,
        pop_size=pop_size, generations=generations,
        seed=seed, verbose=False,
    )
    elapsed = time.time() - t0
    extra = {
        "pop_size": pop_size,
        "generations": generations,
        "best_fitness": float(ga_info.get("best_fitness", 0.0)),
        "ga_elapsed_seconds": float(ga_info.get("elapsed_seconds", elapsed)),
        "ga_final_avg_slope": float(best_results.get("final_avg_slope", 0.0)),
        "ga_final_contiguity": float(best_results.get("final_contiguity", 0.0)),
    }
    return (extra,
            float(ga_info.get("best_fitness", 0.0)),
            int(best_results.get("completed_pairs", env.total_budget)),
            elapsed)


METHOD_DISPATCH = {
    "Random": (run_random, "Random-Block"),
    "Greedy": (run_greedy, "Greedy-Sequential"),
    "GA":     (run_ga, "GA"),
}


# ---------------------------------------------------------------------------
# Sweep loop
# ---------------------------------------------------------------------------
def run_sweep(datasets_root: Path, out_root: Path,
              methods: list[str], presets: list[str],
              seeds: list[int], skip_existing: bool = True,
              budget: int = 100, swaps_per_step: int = 5):
    cells = []
    for preset in presets:
        for method in methods:
            for seed in seeds:
                cells.append((preset, method, seed))

    print(f"Sweep: {len(cells)} cells", flush=True)
    print(f"Datasets root : {datasets_root}", flush=True)
    print(f"Output root   : {out_root}", flush=True)
    print(f"Methods       : {methods}", flush=True)
    print(f"Skip existing : {skip_existing}", flush=True)
    print("=" * 70, flush=True)

    sweep_t0 = time.time()
    n_done = 0
    n_skipped = 0
    n_failed = 0

    for i, (preset, method, seed) in enumerate(cells, 1):
        out_path = out_root / preset / method / f"seed{seed}.json"
        if skip_existing and out_path.exists():
            n_skipped += 1
            print(f"[{i}/{len(cells)}] {preset} / {method} / seed{seed}: SKIP (exists)",
                  flush=True)
            continue

        dataset_dir = datasets_root / f"{preset}_seed{seed}"
        if not dataset_dir.exists():
            print(f"[{i}/{len(cells)}] {preset} / {method} / seed{seed}: SKIP "
                  f"(dataset missing: {dataset_dir})", flush=True)
            continue

        print(f"[{i}/{len(cells)}] {preset} / {method} / seed{seed}: running ...",
              flush=True)
        try:
            env = make_synthetic_env(dataset_dir, total_budget=budget,
                                     swaps_per_step=swaps_per_step)
            env.reset(seed=seed)
            init = init_snapshot(env)

            fn, method_label = METHOD_DISPATCH[method]
            extra, total_reward, pairs_used, wall = fn(env, seed)

            result = build_result(preset, seed, method_label, env, init,
                                  total_reward=total_reward,
                                  wall_seconds=wall,
                                  exchange_pairs_used=pairs_used,
                                  extra=extra)
            write_json_atomic(result, out_path)
            n_done += 1

            print(f"  -> slope={result['slope_pct']:+.4f}% "
                  f"cont={result['cont_delta']:+.5f} "
                  f"baimu_n={result['baimu_count_delta']:+d} "
                  f"wall={wall:.1f}s "
                  f"path={out_path.name}", flush=True)
        except Exception as e:
            n_failed += 1
            err_path = out_path.with_suffix(".error.txt")
            err_path.parent.mkdir(parents=True, exist_ok=True)
            err_path.write_text(f"{type(e).__name__}: {e}", encoding="utf-8")
            print(f"  !! FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print("=" * 70, flush=True)
    print(f"Done in {time.time() - sweep_t0:.1f}s. "
          f"Completed: {n_done}, skipped: {n_skipped}, failed: {n_failed}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True,
                    help="Path to data_dev/ (the 35 preset_seed dirs)")
    ap.add_argument("--out", required=True,
                    help="Output root for seed JSONs (e.g. .../sweep_run/results)")
    ap.add_argument("--methods", nargs="+",
                    default=["Random", "Greedy", "GA"],
                    choices=["Random", "Greedy", "GA"],
                    help="Methods to run (default: all 3)")
    ap.add_argument("--presets", nargs="+", default=[
        "bishan_clone", "neijiang_clone",
        "plain_small_cons", "plain_large_cons",
        "plain_medium_frag", "mixed_medium_frag",
        "hilly_small_cons",
    ])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--budget", type=int, default=100,
                    help="Total swap pairs per episode (matches paper §sec:bench).")
    ap.add_argument("--swaps-per-step", type=int, default=5)
    ap.add_argument("--no-skip", action="store_true",
                    help="Re-run cells even if output already exists.")
    args = ap.parse_args()

    datasets_root = Path(args.datasets).resolve()
    out_root = Path(args.out).resolve()
    if not datasets_root.exists():
        raise FileNotFoundError(f"--datasets {datasets_root} does not exist")
    out_root.mkdir(parents=True, exist_ok=True)

    run_sweep(
        datasets_root=datasets_root,
        out_root=out_root,
        methods=args.methods,
        presets=args.presets,
        seeds=args.seeds,
        skip_existing=not args.no_skip,
        budget=args.budget,
        swaps_per_step=args.swaps_per_step,
    )


if __name__ == "__main__":
    main()
