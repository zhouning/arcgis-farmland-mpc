#!/usr/bin/env python3
"""Neijiang execution-constraint frontier for Paper 9 policy audit.

This mirrors the Bishan execution-frontier sweep, but targets the packaged
Neijiang prepared directory and ONNX ensemble created for the Scientific
Reports revision. It is an execution-constraint frontier, not a retrained
reward-weight Pareto sweep.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
        "description": "Explicit maximum cultivated-area loss of 500 ha.",
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
    },
    {
        "id": "conservative_baimu_no_loss",
        "label": "No net cultivated + baimu area",
        "description": "Cultivated-area and qualifying baimu-fang-area no-net-loss floors.",
        "constraints": {
            "cultivated_area_floor_delta_ha": 0.0,
            "baimu_area_floor_delta_ha": 0.0,
        },
        "pair_selection": {},
    },
    {
        "id": "connectivity_conservative",
        "label": "Connectivity conservative",
        "description": "No-net-loss floor with stronger local connectivity protection.",
        "constraints": {"cultivated_area_floor_delta_ha": 0.0},
        "pair_selection": {"gamma_conn": 2.0, "delta_conn": 1.0},
    },
]


def ensure_repo_root_on_path() -> None:
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path("runs/scirep_extra/prepared_neijiang"),
    )
    ap.add_argument(
        "--ensemble-dir",
        type=Path,
        default=Path("runs/scirep_extra/onnx/neijiang/ensemble_seed0"),
    )
    ap.add_argument(
        "--scratch-dir",
        type=Path,
        default=Path("runs/neijiang/pareto/constraints"),
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=Path("paper/submission_scirep_corrected/neijiang_constraint_frontier.json"),
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=Path("paper/submission_scirep_corrected/neijiang_constraint_frontier.md"),
    )
    ap.add_argument(
        "--out-fig-pdf",
        type=Path,
        default=Path("paper/submission_scirep_corrected/04_figures/neijiang_constraint_frontier.pdf"),
    )
    ap.add_argument(
        "--out-fig-png",
        type=Path,
        default=Path("paper/submission_scirep_corrected/04_figures/neijiang_constraint_frontier.png"),
    )
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--mpc-batch-size", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--only", nargs="+", choices=[c["id"] for c in CONFIGS])
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--skip-policy-audit", action="store_true")
    ap.add_argument("--force", action="store_true")
    return ap


def slope_source(args: argparse.Namespace) -> Path:
    out = args.prepared_dir / "dem_slope_analysis" / "output"
    shp = out / "DLTB_with_slope.shp"
    if shp.exists():
        return shp
    return out / "DLTB_with_slope.gpkg"


def json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def has_plan_outputs(cell_dir: Path) -> bool:
    return (cell_dir / "mpc_summary.json").exists() and (cell_dir / "optimized.shp").exists()


def run_cmd(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")


def run_plan(args: argparse.Namespace, cfg: dict, cell_dir: Path) -> tuple[dict, float]:
    ensure_repo_root_on_path()
    from farmland_mpc.mpc_plan import run

    optimized = cell_dir / "optimized.shp"
    kwargs = {
        "ensemble_dir": str(args.ensemble_dir),
        "out_dir": str(cell_dir),
        "horizon": args.horizon,
        "top_k": args.top_k,
        "n_episodes": 1,
        "continuation": "greedy",
        "scoring": "reward",
        "threads": args.threads,
        "mpc_batch_size": args.mpc_batch_size,
        "max_steps": args.max_steps,
        "seed_offset": 0,
        "env_kind": "county",
        "prepared_dir": str(args.prepared_dir),
        "output_fc": str(optimized),
        "input_dltb_fc": str(slope_source(args)),
    }
    kwargs.update(cfg.get("constraints", {}))
    kwargs.update(cfg.get("pair_selection", {}))
    t0 = time.time()
    summary = run(**kwargs)
    return summary, time.time() - t0


def run_policy_audit(args: argparse.Namespace, cfg: dict, cell_dir: Path) -> dict:
    optimized = cell_dir / "optimized.shp"
    summary = cell_dir / "mpc_summary.json"
    out = cell_dir / "policy_translation.json"
    cmd = [
        sys.executable,
        "scripts/policy_translation_optimized.py",
        "--optimized",
        str(optimized),
        "--townships",
        str(args.prepared_dir / "townships.json"),
        "--region",
        f"Neijiang Dongxing:{cfg['id']}",
        "--summary",
        str(summary),
        "--out",
        str(out),
    ]
    run_cmd(cmd)
    return json_load(out)


def run_validation(args: argparse.Namespace, cell_dir: Path) -> dict:
    optimized = cell_dir / "optimized.shp"
    summary = cell_dir / "mpc_summary.json"
    out = cell_dir / "validate_report.json"
    cmd = [
        sys.executable,
        "verification/validate_optimized_shp.py",
        "--optimized",
        str(optimized),
        "--slope-shp",
        str(slope_source(args)),
        "--summary",
        str(summary),
        "--out",
        str(out),
    ]
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0 and not out.exists():
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")
    return json_load(out)


def copy_shapefile(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        src_part = src.with_suffix(ext)
        if src_part.exists():
            shutil.copy2(src_part, dst.with_suffix(ext))


def extract_row(cfg: dict, summary: dict, policy: dict, validation: dict, wall_s: float) -> dict:
    ep0 = summary.get("results", [{}])[0]
    if policy:
        bands = policy["slope_bands"]["delta_ha"]
        steep_tail = float(bands["15_25"] + bands["gt25"])
        cultivated = policy["cultivated_area"]
        baimu = policy["baimu_fang"]
        swaps = policy["swap_area_totals"]
    else:
        steep_tail = float("nan")
        cultivated = {"delta_ha": ep0.get("cultivated_area_change_ha", float("nan")), "delta_pct": ep0.get("cultivated_area_change_pct", float("nan"))}
        baimu = {"delta_count": ep0.get("baimu_count_change", 0), "delta_area_ha": ep0.get("baimu_area_change_ha", float("nan"))}
        swaps = {"farm_to_forest_count": ep0.get("swaps_completed", 0)}
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
        "cultivated_area_change_ha": float(
            recomputed.get("cultivated_area_ha", cultivated["delta_ha"])
        ),
        "cultivated_area_change_pct": float(
            recomputed.get("cultivated_area_pct", cultivated["delta_pct"])
        ),
        "steep_tail_change_ha": steep_tail,
        "total_reward": float(ep0.get("total_reward", 0.0)),
        "swaps": int(swaps["farm_to_forest_count"]),
        "steps_run": int(ep0.get("steps_run", 0)),
        "runtime_s": float(wall_s if wall_s else ep0.get("total_time_s", 0.0)),
        "source_optimized_shp": policy.get("source_optimized_shp") if policy else None,
        "validation_overall_pass": bool(validation.get("overall_pass", False)),
    }


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Neijiang Dongxing execution-constraint frontier",
        "",
        "This is an execution-constraint frontier. It changes Tool 4 hard constraints and local pair-selection settings, not Tool 2 sampling or Tool 3 training.",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plot(path_pdf: Path, path_png: Path, rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    label_offsets = {
        "unconstrained": (8, -13),
        "loss_tolerance_500ha": (8, 20),
        "loss_tolerance_250ha": (8, 5),
        "loss_tolerance_100ha": (8, -10),
        "no_net_loss": (9, -12),
        "conservative_baimu_no_loss": (9, 5),
        "connectivity_conservative": (9, 16),
    }
    plot_labels = {
        "loss_tolerance_500ha": "Floor -500 ha",
        "loss_tolerance_250ha": "Floor -250 ha",
        "loss_tolerance_100ha": "Floor -100 ha",
        "conservative_baimu_no_loss": "No net + baimu",
    }
    x = [r["cultivated_area_change_ha"] for r in rows]
    y = [r["slope_change_pct"] for r in rows]
    sizes = [max(40, min(260, abs(r["baimu_area_change_ha"]) / 2.0)) for r in rows]
    colors = ["#b13a32" if r["cultivated_area_change_ha"] < 0 else "#1f7a5a" for r in rows]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.axvline(0, color="#222222", lw=1.0, ls="--", alpha=0.7)
    ax.axhline(0, color="#666666", lw=0.8, alpha=0.4)
    ax.scatter(x, y, s=sizes, c=colors, edgecolor="#222222", linewidth=0.8, zorder=3)
    for r, xi, yi in zip(rows, x, y):
        ax.annotate(
            plot_labels.get(r["id"], r["label"]),
            (xi, yi),
            xytext=label_offsets.get(r["id"], (6, 6)),
            textcoords="offset points",
            fontsize=7.2,
            arrowprops={"arrowstyle": "-", "lw": 0.45, "color": "#555555", "shrinkA": 0, "shrinkB": 4},
        )
    ax.set_xlabel("Cultivated-area change (ha)")
    ax.set_ylabel("Farmland-slope change (%)")
    ax.set_title("Neijiang Dongxing constraint frontier")
    ax.grid(True, color="#d8d8d8", lw=0.6, alpha=0.8)
    if x and y:
        ax.set_xlim(min(x) - 80, max(x) + 120)
        ax.set_ylim(min(y) - 0.08, max(y) + 0.08)
    fig.tight_layout()
    path_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_pdf)
    fig.savefig(path_png, dpi=240)
    plt.close(fig)


def selected_configs(args: argparse.Namespace) -> list[dict]:
    if not args.only:
        return CONFIGS
    keep = set(args.only)
    return [cfg for cfg in CONFIGS if cfg["id"] in keep]


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(sys.argv[1:] if argv is None else argv)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    report = {
        "kind": "neijiang_execution_constraint_frontier",
        "date": "2026-06-14",
        "prepared_dir": str(args.prepared_dir),
        "ensemble_dir": str(args.ensemble_dir),
        "slope_source": str(slope_source(args)),
        "note": (
            "Hard constraints and local pair-selection settings alter the real "
            "execution path. Runtime reward-weight overrides are not interpreted "
            "as a full reward-policy Pareto sweep because the ONNX reward head is "
            "not retrained under new weights."
        ),
        "configs": [],
    }

    for cfg in selected_configs(args):
        print(f"\n=== {cfg['id']} ===", flush=True)
        cell_dir = args.scratch_dir / cfg["id"]
        cell_dir.mkdir(parents=True, exist_ok=True)
        if not args.force and has_plan_outputs(cell_dir):
            summary = json_load(cell_dir / "mpc_summary.json")
            wall_s = float(summary.get("results", [{}])[0].get("total_time_s", 0.0))
            print(f"  reused Tool 4 plan outputs in {cell_dir}", flush=True)
        else:
            summary, wall_s = run_plan(args, cfg, cell_dir)

        validate_report = cell_dir / "validate_report.json"
        if not args.force and validate_report.exists():
            validation = json_load(validate_report)
            print(f"  reused validation report", flush=True)
        elif args.skip_validation:
            validation = {}
        else:
            validation = run_validation(args, cell_dir)

        policy_report = cell_dir / "policy_translation.json"
        if not args.force and policy_report.exists():
            policy = json_load(policy_report)
            print(f"  reused policy audit", flush=True)
        elif args.skip_policy_audit:
            policy = {}
        else:
            policy = run_policy_audit(args, cfg, cell_dir)

        row = extract_row(cfg, summary, policy, validation, wall_s)
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

    write_markdown(args.out_md, rows)
    write_plot(args.out_fig_pdf, args.out_fig_png, rows)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_fig_pdf}")
    print(f"Wrote {args.out_fig_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
