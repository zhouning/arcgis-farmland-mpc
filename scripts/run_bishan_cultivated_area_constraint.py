#!/usr/bin/env python3
"""Run a no-net-loss cultivated-area constrained MPC experiment.

The constraint is enforced inside CountyLevelEnv at each committed parcel pair:

    cumulative cultivated-area delta >= --floor-delta-ha

The default floor is 0 ha, i.e. exact no-net-loss. The script writes the Tool 4
summary and optimized shapefile, then optionally runs the independent GIS audit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from farmland_mpc.mpc_plan import run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="Bishan")
    ap.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path("/Users/zhouning/farmland_mpc_runs/bishan/prepared"),
    )
    ap.add_argument(
        "--ensemble-dir",
        type=Path,
        default=Path("/Users/zhouning/farmland_mpc_runs/bishan/prepared/ensemble_seed0"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/bishan/cultivated_area_constraint/no_net_loss"),
    )
    ap.add_argument(
        "--report-out",
        type=Path,
        default=Path("paper/submission_scirep_corrected/bishan_cultivated_area_constraint.json"),
    )
    ap.add_argument("--floor-delta-ha", type=float, default=0.0)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--skip-gis-audit", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    optimized = args.out_dir / "optimized.shp"
    slope_shp = (
        args.prepared_dir / "dem_slope_analysis" / "output" / "DLTB_with_slope.shp"
    )

    summary = run(
        ensemble_dir=str(args.ensemble_dir),
        out_dir=str(args.out_dir),
        horizon=args.horizon,
        top_k=args.top_k,
        n_episodes=1,
        continuation="greedy",
        scoring="reward",
        threads=args.threads,
        seed_offset=0,
        env_kind="county",
        prepared_dir=str(args.prepared_dir),
        output_fc=str(optimized),
        input_dltb_fc=str(slope_shp),
        cultivated_area_floor_delta_ha=args.floor_delta_ha,
    )

    record = {
        "experiment": "cultivated_area_constraint",
        "region": args.region,
        "constraint": {
            "kind": "hard_cumulative_cultivated_area_floor",
            "floor_delta_ha": args.floor_delta_ha,
            "enforcement": (
                "Every committed farm->forest / forest->farm parcel pair must "
                "keep cumulative cultivated area at or above the configured floor."
            ),
        },
        "tool4_summary": summary,
    }

    if not args.skip_gis_audit:
        validate_out = args.out_dir / "validate_report.json"
        cmd = [
            sys.executable,
            "verification/validate_optimized_shp.py",
            "--optimized",
            str(optimized),
            "--slope-shp",
            str(slope_shp),
            "--summary",
            str(args.out_dir / "mpc_summary.json"),
            "--out",
            str(validate_out),
        ]
        completed = subprocess.run(cmd, check=False)
        record["gis_audit_exit_code"] = completed.returncode
        if validate_out.exists():
            record["gis_audit"] = json.loads(validate_out.read_text())

    args.report_out.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"Wrote {args.report_out}")
    return int(record.get("gis_audit_exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
