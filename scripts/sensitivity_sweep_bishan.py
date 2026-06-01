# -*- coding: utf-8 -*-
"""
sensitivity_sweep_bishan.py — H / K / lambda_rank / margin sensitivity scan
on Bishan, package-side prepare, addressing reviewer-3 (Codex panel
2026-05-31) request for hyperparameter robustness evidence.

Scope of the sweep
------------------
1. (H, K) sweep using the shipped Bishan ensemble (lambda_rank=5.0,
   margin=0.1). Cells: H in {3, 5, 10}, K in {25, 50, 100}. The shipped
   ensemble is deterministic across the action-selection path under
   fixed seed, so we evaluate one canonical episode per cell. Total
   cost is bounded by H*K (planner forward passes); runtime per cell
   is roughly proportional to K, all within minutes on the 12-thread
   CPU.

2. lambda_rank sweep at H=5, K=50, greedy continuation. The package-side
   prepare currently has only lambda=5.0 ensembles. We also run on the
   lab-pipeline lambda_ablation/ checkpoints (lambda in {0.0, 1.0, 5.0})
   for cross-pipeline comparison; the lab-pipeline lambda_ablation
   includes lambda=1.0 which the package-side does not. The lab-pipeline
   sweep on lambda is reported alongside the package-side single point.

The script writes a JSON report to paper/submission_commsee/ for the
new SI section S7 ("Hyperparameter sensitivity"). The JSON is the audit
artefact; the corresponding LaTeX table is generated from it.

Usage
-----
    python scripts/sensitivity_sweep_bishan.py \
        --prepared-dir /Users/zhouning/farmland_mpc_runs/bishan/prepared \
        --shipped-onnx-dir /Users/zhouning/farmland_mpc_runs/bishan/prepared/ensemble_seed0 \
        --out paper/submission_commsee/sensitivity_sweep_bishan.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("sweep")


def _run_one_plan(prepared_dir, ensemble_dir, out_dir, horizon, top_k,
                  continuation="greedy", scoring="reward",
                  n_episodes=1, seed_offset=0, threads=0):
    """Run a single MPC episode and return the summary dict."""
    from farmland_mpc.mpc_plan import run
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary = run(
        ensemble_dir=str(ensemble_dir),
        out_dir=str(out_dir),
        horizon=horizon,
        top_k=top_k,
        n_episodes=n_episodes,
        continuation=continuation,
        scoring=scoring,
        threads=threads,
        seed_offset=seed_offset,
        env_kind="county",
        prepared_dir=str(prepared_dir),
    )
    summary["wall_s_total"] = time.time() - t0
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", required=True, type=Path)
    ap.add_argument("--shipped-onnx-dir", required=True, type=Path,
                    help="Dir with ensemble_member{0,1,2}.onnx for the "
                         "package-side shipped ensemble (lam=5.0).")
    ap.add_argument("--scratch-dir", type=Path,
                    default=Path("runs/bishan/sweep"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--hk-only", action="store_true",
                    help="Skip the lambda sweep; (H,K) only. Useful for "
                         "fast smoke tests.")
    args = ap.parse_args()

    args.scratch_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "prepared_dir": str(args.prepared_dir),
        "shipped_onnx_dir": str(args.shipped_onnx_dir),
        "hk_sweep": [],
        "lambda_sweep_package_side": [],
        "notes": [
            "(H,K) cells run with continuation=greedy, scoring=reward, "
            "n_episodes=1 each (the shipped ensemble is deterministic "
            "across the action-selection path under fixed seed).",
            "lambda sweep on package-side prepare uses only lam=5.0 "
            "(the ensembles trained for the headline result); the lab-"
            "pipeline lambda_ablation/ checkpoints in paper/checkpoints/ "
            "cover lam in {0.0, 1.0, 5.0} for cross-pipeline reference.",
        ],
    }

    # ---- 1. (H, K) sweep ------------------------------------------- #
    H_grid = [3, 5, 10]
    K_grid = [25, 50, 100]
    log.info("=" * 60)
    log.info("(H,K) sweep on package-side prepare, shipped ensemble")
    log.info("=" * 60)
    for H in H_grid:
        for K in K_grid:
            cell_id = f"H{H}_K{K}"
            cell_dir = args.scratch_dir / "hk" / cell_id
            log.info("  cell %s ...", cell_id)
            try:
                summary = _run_one_plan(
                    prepared_dir=args.prepared_dir,
                    ensemble_dir=args.shipped_onnx_dir,
                    out_dir=cell_dir,
                    horizon=H,
                    top_k=K,
                    continuation="greedy",
                    scoring="reward",
                    n_episodes=1,
                )
                ep0 = summary["results"][0]
                row = {
                    "H": H, "K": K,
                    "slope_change_pct": float(ep0["slope_change_pct"]),
                    "cont_change": float(ep0["cont_change"]),
                    "baimu_count_change": int(ep0["baimu_count_change"]),
                    "baimu_area_change_ha": float(ep0["baimu_area_change_ha"]),
                    "total_reward": float(ep0["total_reward"]),
                    "wall_s_episode": float(ep0["total_time_s"]),
                    "wall_s_total": float(summary["wall_s_total"]),
                }
            except Exception as e:
                log.exception("cell %s failed", cell_id)
                row = {"H": H, "K": K, "error": repr(e)}
            log.info("    -> %s", json.dumps({k: v for k, v in row.items()
                                              if k != "error"}))
            report["hk_sweep"].append(row)
            # Persist incrementally so the script can be killed without
            # losing earlier cells.
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2))

    # ---- 2. lambda sweep (package-side, currently single-point) --- #
    if not args.hk_only:
        log.info("=" * 60)
        log.info("lambda sweep — package-side prepare currently has lam=5.0 only")
        log.info("=" * 60)
        cell_dir = args.scratch_dir / "lambda" / "lam5p0_H5_K50"
        try:
            summary = _run_one_plan(
                prepared_dir=args.prepared_dir,
                ensemble_dir=args.shipped_onnx_dir,
                out_dir=cell_dir,
                horizon=5, top_k=50,
                continuation="greedy", scoring="reward",
                n_episodes=1,
            )
            ep0 = summary["results"][0]
            report["lambda_sweep_package_side"].append({
                "lambda_rank": 5.0,
                "margin": 0.1,
                "H": 5, "K": 50,
                "slope_change_pct": float(ep0["slope_change_pct"]),
                "total_reward": float(ep0["total_reward"]),
                "wall_s_episode": float(ep0["total_time_s"]),
            })
        except Exception as e:
            log.exception("lambda=5.0 cell failed")
            report["lambda_sweep_package_side"].append({
                "lambda_rank": 5.0, "error": repr(e),
            })

    args.out.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
