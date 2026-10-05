#!/usr/bin/env python3
"""Run matched rule baselines on a prepared real county dataset.

The baselines use the same CountyLevelEnv, action masks, budget and hard
cultivated-area floor as the no-net-loss MPC deployment.  Existing outputs are
never overwritten unless --force is supplied.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LEGACY = Path(os.environ.get("LEGACY_TEST_ROOT", "D:/test"))
for p in (REPO, LEGACY):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def run_one(prepared: Path, method: str, seed: int, floor: float):
    from farmland_mpc.blocks_env import make_env
    import baselines_county

    env = make_env(
        prepared_dir=str(prepared),
        cultivated_area_floor_delta_ha=float(floor),
    )
    env.reset(seed=seed)
    init_slope = float(env.avg_farmland_slope)
    init_cont = float(env.contiguity)
    init_baimu_area = float(env.baimu_total_area)
    init_baimu_count = int(env.baimu_count)
    t0 = time.time()
    if method == "random_block":
        row = baselines_county.run_random_block(env, seed=seed)
    elif method == "greedy_sequential":
        row = baselines_county.run_greedy_sequential(env)
    else:
        raise ValueError(method)
    row = dict(row)
    row["initial_slope"] = init_slope
    row["initial_cont"] = init_cont
    row["initial_baimu_area_ha"] = init_baimu_area / 10000.0
    row["initial_baimu_count"] = init_baimu_count
    row["cultivated_area_ha"] = float(env.total_farm_area / 10000.0)
    row["cultivated_area_change_ha"] = float(
        (env.total_farm_area - env.initial_farm_area) / 10000.0
    )
    row.update({
        "method_id": method,
        "seed": int(seed),
        "prepared_dir": str(prepared),
        "cultivated_area_floor_delta_ha": float(floor),
        "runner_elapsed_s": float(time.time() - t0),
        "constraint_pass": bool(row["cultivated_area_change_ha"] >= floor - 1e-8),
    })
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", type=Path, required=True)
    ap.add_argument("--region", required=True, choices=["bishan", "neijiang"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--floor", type=float, default=0.0)
    ap.add_argument("--methods", nargs="+", default=["random_block", "greedy_sequential"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.out.exists() and not args.force:
        print(f"exists, refusing to overwrite: {args.out}")
        return
    rows = []
    for method in args.methods:
        for seed in args.seeds:
            print(f"[{args.region}] {method} seed={seed}", flush=True)
            rows.append(run_one(args.prepared_dir, method, seed, args.floor))
    payload = {
        "region": args.region,
        "prepared_dir": str(args.prepared_dir),
        "budget": {"total_budget": 500, "swaps_per_step": 5, "max_steps": 100},
        "cultivated_area_floor_delta_ha": args.floor,
        "methods": args.methods,
        "seeds": args.seeds,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
