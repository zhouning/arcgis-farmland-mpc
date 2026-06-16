#!/usr/bin/env python3
"""Bishan execution-constraint frontier for policy audit.

This sweep is intentionally an execution-constraint frontier, not a full
reward-weight retraining sweep. Tool 4 can override reward weights at runtime,
but the ONNX reward head remains trained under its original weights; hard
constraints and local pair-selection heuristic parameters are the settings that
actually change the committed on-disk cadastre without re-running Tools 2-3.

Outputs:
  * per-cell Tool 4 summaries and optimized shapefiles under runs/bishan/pareto/
  * per-cell GIS-only policy audits
  * paper/submission_scirep_corrected/bishan_constraint_frontier.json
  * paper/submission_scirep_corrected/bishan_constraint_frontier.md
  * paper/submission_scirep_corrected/04_figures/bishan_constraint_pareto.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


CONFIGS = [
    {
        "id": "unconstrained",
        "label": "Unconstrained",
        "description": "Default shipped Tool 4 execution; audit scenario only.",
        "constraints": {},
        "pair_selection": {},
    },
    {
        "id": "loss_tolerance_500ha",
        "label": "Cultivated floor -500 ha",
        "description": "Slope-heavy setting with explicit maximum cultivated-area loss near the audited unconstrained loss.",
        "constraints": {"cultivated_area_floor_delta_ha": -500.0},
        "pair_selection": {},
    },
    {
        "id": "loss_tolerance_250ha",
        "label": "Cultivated floor -250 ha",
        "description": "Intermediate area-preservation setting.",
        "constraints": {"cultivated_area_floor_delta_ha": -250.0},
        "pair_selection": {},
    },
    {
        "id": "loss_tolerance_100ha",
        "label": "Cultivated floor -100 ha",
        "description": "Strict area-preservation setting with small explicit tolerance.",
        "constraints": {"cultivated_area_floor_delta_ha": -100.0},
        "pair_selection": {},
    },
    {
        "id": "no_net_loss",
        "label": "No net cultivated loss",
        "description": "Hard cultivated-area no-net-loss execution floor.",
        "constraints": {"cultivated_area_floor_delta_ha": 0.0},
        "pair_selection": {},
        "reuse_optimized": Path("runs/bishan/cultivated_area_constraint/no_net_loss/optimized.shp"),
        "reuse_summary": Path("runs/bishan/cultivated_area_constraint/no_net_loss/mpc_summary.json"),
        "reuse_policy": Path("paper/submission_scirep_corrected/policy_translation_bishan_constrained.json"),
    },
    {
        "id": "conservative_baimu_no_loss",
        "label": "No net cultivated + baimu area",
        "description": "Conservative profile with cultivated-area and qualifying baimu-fang-area no-net-loss floors.",
        "constraints": {
            "cultivated_area_floor_delta_ha": 0.0,
            "baimu_area_floor_delta_ha": 0.0,
        },
        "pair_selection": {},
    },
    {
        "id": "connectivity_conservative",
        "label": "Connectivity conservative",
        "description": "No-net-loss floor with stronger local connectivity protection during pair selection.",
        "constraints": {"cultivated_area_floor_delta_ha": 0.0},
        "pair_selection": {"gamma_conn": 2.0, "delta_conn": 1.0},
    },
]


def _json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_cmd(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")


def _run_plan(args, cfg: dict, cell_dir: Path) -> tuple[dict, float]:
    from farmland_mpc.mpc_plan import run

    optimized = cell_dir / "optimized.shp"
    slope_shp = args.prepared_dir / "dem_slope_analysis" / "output" / "DLTB_with_slope.shp"
    kwargs = {
        "ensemble_dir": str(args.ensemble_dir),
        "out_dir": str(cell_dir),
        "horizon": args.horizon,
        "top_k": args.top_k,
        "n_episodes": 1,
        "continuation": "greedy",
        "scoring": "reward",
        "threads": args.threads,
        "seed_offset": 0,
        "env_kind": "county",
        "prepared_dir": str(args.prepared_dir),
        "output_fc": str(optimized),
        "input_dltb_fc": str(slope_shp),
    }
    kwargs.update(cfg.get("constraints", {}))
    kwargs.update(cfg.get("pair_selection", {}))
    t0 = time.time()
    summary = run(**kwargs)
    return summary, time.time() - t0


def _run_policy_audit(args, cfg: dict, cell_dir: Path) -> dict:
    optimized = cell_dir / "optimized.shp"
    summary = cell_dir / "mpc_summary.json"
    out = cell_dir / "policy_translation.json"
    cmd = [
        sys.executable,
        "scripts/policy_translation_optimized.py",
        "--optimized", str(optimized),
        "--townships", str(args.prepared_dir / "townships.json"),
        "--region", f"Bishan:{cfg['id']}",
        "--summary", str(summary),
        "--out", str(out),
    ]
    _run_cmd(cmd)
    return _json_load(out)


def _run_validation(args, cell_dir: Path) -> dict:
    optimized = cell_dir / "optimized.shp"
    summary = cell_dir / "mpc_summary.json"
    slope_shp = args.prepared_dir / "dem_slope_analysis" / "output" / "DLTB_with_slope.shp"
    out = cell_dir / "validate_report.json"
    cmd = [
        sys.executable,
        "verification/validate_optimized_shp.py",
        "--optimized", str(optimized),
        "--slope-shp", str(slope_shp),
        "--summary", str(summary),
        "--out", str(out),
    ]
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0 and not out.exists():
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")
    return _json_load(out)


def _copy_reuse_outputs(cfg: dict, cell_dir: Path) -> None:
    import shutil

    optimized = cfg.get("reuse_optimized")
    summary = cfg.get("reuse_summary")
    if optimized is None or summary is None:
        return
    optimized = Path(optimized)
    summary = Path(summary)
    if not optimized.exists() or not summary.exists():
        return
    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        src = optimized.with_suffix(ext)
        if src.exists():
            shutil.copy2(src, cell_dir / f"optimized{ext}")
    shutil.copy2(summary, cell_dir / "mpc_summary.json")


def _extract_row(cfg: dict, summary: dict, policy: dict, validation: dict, wall_s: float) -> dict:
    ep0 = summary.get("results", [{}])[0]
    bands = policy["slope_bands"]["delta_ha"]
    steep_tail = float(bands["15_25"] + bands["gt25"])
    cultivated = policy["cultivated_area"]
    baimu = policy["baimu_fang"]
    swaps = policy["swap_area_totals"]
    recomputed = validation.get("delta_recomputed", {})
    return {
        "id": cfg["id"],
        "label": cfg["label"],
        "description": cfg["description"],
        "constraints": cfg.get("constraints", {}),
        "pair_selection": cfg.get("pair_selection", {}),
        "slope_change_pct": float(recomputed.get("slope_pct", ep0.get("slope_change_pct", 0.0))),
        "cont_change": float(recomputed.get("cont", ep0.get("cont_change", 0.0))),
        "baimu_count_change": int(baimu["delta_count"]),
        "baimu_area_change_ha": float(recomputed.get("baimu_ha", baimu["delta_area_ha"])),
        "cultivated_area_change_ha": float(recomputed.get("cultivated_area_ha", cultivated["delta_ha"])),
        "cultivated_area_change_pct": float(recomputed.get("cultivated_area_pct", cultivated["delta_pct"])),
        "steep_tail_change_ha": steep_tail,
        "total_reward": float(ep0.get("total_reward", 0.0)),
        "swaps": int(swaps["farm_to_forest_count"]),
        "steps_run": int(ep0.get("steps_run", 0)),
        "runtime_s": float(wall_s if wall_s else ep0.get("total_time_s", 0.0)),
        "source_optimized_shp": policy["source_optimized_shp"],
        "validation_overall_pass": bool(validation.get("overall_pass", False)),
    }


def _write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Bishan execution-constraint frontier",
        "",
        "Source date: 2026-06-02",
        "",
        "This is an execution-constraint frontier. Runtime reward-weight overrides are not treated as a full policy Pareto sweep unless Tool 2 sampling and Tool 3 training are re-run under those weights.",
        "",
        "| Mode | Slope delta (%) | Contiguity delta | Steep-tail delta (ha) | Baimu count delta | Baimu area delta (ha) | Cultivated-area delta (ha) | Reward | Swaps | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['slope_change_pct']:+.3f} | "
            f"{r['cont_change']:+.4f} | {r['steep_tail_change_ha']:+.1f} | "
            f"{r['baimu_count_change']:+d} | {r['baimu_area_change_ha']:+.1f} | "
            f"{r['cultivated_area_change_ha']:+.1f} | {r['total_reward']:+.2f} | "
            f"{r['swaps']} | {r['runtime_s']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(path_pdf: Path, path_png: Path, rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    labels = {
        "unconstrained": "Unconstrained",
        "loss_tolerance_500ha": "Floor -500 ha",
        "loss_tolerance_250ha": "Floor -250 ha",
        "loss_tolerance_100ha": "Floor -100 ha",
        "no_net_loss": "No-net loss",
        "conservative_baimu_no_loss": "No-net + baimu",
        "connectivity_conservative": "Connectivity",
    }
    label_offsets = {
        "unconstrained": (8, -13),
        "loss_tolerance_500ha": (8, 12),
        "loss_tolerance_250ha": (8, 8),
        "loss_tolerance_100ha": (8, 8),
        "no_net_loss": (9, -12),
        "conservative_baimu_no_loss": (9, 5),
        "connectivity_conservative": (9, 16),
    }
    x = [r["cultivated_area_change_ha"] for r in rows]
    y = [r["slope_change_pct"] for r in rows]
    sizes = [max(40, min(220, abs(r["baimu_area_change_ha"]) / 3.0)) for r in rows]
    colors = ["#b13a32" if r["cultivated_area_change_ha"] < 0 else "#1f7a5a" for r in rows]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.axvline(0, color="#222222", lw=1.0, ls="--", alpha=0.7)
    ax.axhline(0, color="#666666", lw=0.8, alpha=0.4)
    ax.scatter(x, y, s=sizes, c=colors, edgecolor="#222222", linewidth=0.8, zorder=3)
    for r, xi, yi in zip(rows, x, y):
        ax.annotate(
            labels.get(r["id"], r["label"]),
            (xi, yi),
            xytext=label_offsets.get(r["id"], (6, 6)),
            textcoords="offset points",
            fontsize=7.5,
            arrowprops={"arrowstyle": "-", "lw": 0.45, "color": "#555555", "shrinkA": 0, "shrinkB": 4},
        )
    ax.set_xlabel("Cultivated-area change (ha)")
    ax.set_ylabel("Farmland-slope change (%)")
    ax.set_title("Bishan constraint frontier")
    ax.grid(True, color="#d8d8d8", lw=0.6, alpha=0.8)
    ax.set_xlim(min(x) - 60, max(x) + 95)
    ax.set_ylim(min(y) - 0.15, max(y) + 0.15)
    fig.tight_layout()
    path_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_pdf)
    fig.savefig(path_png, dpi=240)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", type=Path, default=Path("/Users/zhouning/farmland_mpc_runs/bishan/prepared"))
    ap.add_argument("--ensemble-dir", type=Path, default=Path("/Users/zhouning/farmland_mpc_runs/bishan/prepared/ensemble_seed0"))
    ap.add_argument("--scratch-dir", type=Path, default=Path("runs/bishan/pareto/constraints"))
    ap.add_argument("--out-json", type=Path, default=Path("paper/submission_scirep_corrected/bishan_constraint_frontier.json"))
    ap.add_argument("--out-md", type=Path, default=Path("paper/submission_scirep_corrected/bishan_constraint_frontier.md"))
    ap.add_argument("--out-fig-pdf", type=Path, default=Path("paper/submission_scirep_corrected/04_figures/bishan_constraint_pareto.pdf"))
    ap.add_argument("--out-fig-png", type=Path, default=Path("paper/submission_scirep_corrected/04_figures/bishan_constraint_pareto.png"))
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    report = {
        "kind": "bishan_execution_constraint_frontier",
        "date": "2026-06-02",
        "prepared_dir": str(args.prepared_dir),
        "ensemble_dir": str(args.ensemble_dir),
        "note": (
            "Hard constraints and local pair-selection settings alter the real "
            "execution path. Runtime reward-weight overrides are not interpreted "
            "as a full reward-policy Pareto sweep because the ONNX reward head is "
            "not retrained under new weights."
        ),
        "configs": [],
    }

    for cfg in CONFIGS:
        print(f"\n=== {cfg['id']} ===", flush=True)
        cell_dir = args.scratch_dir / cfg["id"]
        cell_dir.mkdir(parents=True, exist_ok=True)

        reuse_summary = cfg.get("reuse_summary")
        reuse_policy = cfg.get("reuse_policy")
        if not args.force and (cell_dir / "mpc_summary.json").exists() and (cell_dir / "policy_translation.json").exists():
            summary = _json_load(cell_dir / "mpc_summary.json")
            policy = _json_load(cell_dir / "policy_translation.json")
            if (cell_dir / "validate_report.json").exists():
                validation = _json_load(cell_dir / "validate_report.json")
            else:
                validation = _run_validation(args, cell_dir)
            wall_s = float(summary.get("results", [{}])[0].get("total_time_s", 0.0))
            print(f"  reused {cell_dir}", flush=True)
        elif (
            not args.force
            and reuse_summary is not None
            and reuse_policy is not None
            and Path(reuse_summary).exists()
            and Path(reuse_policy).exists()
        ):
            summary = _json_load(Path(reuse_summary))
            policy = _json_load(Path(reuse_policy))
            validation = _run_validation(args, cell_dir)
            wall_s = float(summary.get("results", [{}])[0].get("total_time_s", 0.0))
            print(f"  reused {reuse_summary} and {reuse_policy}", flush=True)
        elif (
            not args.force
            and cfg.get("reuse_optimized") is not None
            and cfg.get("reuse_summary") is not None
            and Path(cfg["reuse_optimized"]).exists()
            and Path(cfg["reuse_summary"]).exists()
        ):
            _copy_reuse_outputs(cfg, cell_dir)
            summary = _json_load(cell_dir / "mpc_summary.json")
            validation = _run_validation(args, cell_dir)
            policy = _run_policy_audit(args, cfg, cell_dir)
            wall_s = float(summary.get("results", [{}])[0].get("total_time_s", 0.0))
            print(f"  reused optimized shapefile and regenerated generic policy audit", flush=True)
        else:
            summary, wall_s = _run_plan(args, cfg, cell_dir)
            validation = _run_validation(args, cell_dir)
            policy = _run_policy_audit(args, cfg, cell_dir)

        row = _extract_row(cfg, summary, policy, validation, wall_s)
        rows.append(row)
        report["configs"].append(row)
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"  slope {row['slope_change_pct']:+.3f}% | cultivated "
            f"{row['cultivated_area_change_ha']:+.1f} ha | baimu "
            f"{row['baimu_area_change_ha']:+.1f} ha | swaps {row['swaps']}",
            flush=True,
        )

    _write_markdown(args.out_md, rows)
    _write_plot(args.out_fig_pdf, args.out_fig_png, rows)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_fig_pdf}")
    print(f"Wrote {args.out_fig_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
