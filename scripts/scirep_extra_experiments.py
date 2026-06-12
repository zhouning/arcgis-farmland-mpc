#!/usr/bin/env python3
"""Scientific Reports extra experiments for Paper 9.

The runner does three reproducible jobs:

1. Build clean package-style prepared directories for Bishan and Neijiang.
2. Convert research-side five-seed .pt ensembles into package ONNX folders.
3. Run no-net-loss constrained MPC for each independent ensemble and aggregate
   cross-seed metrics.

The script intentionally does not treat runtime reward overrides as retrained
reward-weight sensitivity. A retrained sweep must re-sample/re-label Tool 2 data
and re-train Tool 3 under each weight profile.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NEIJIANG_TOWNSHIPS = {
    "511011001": "N01",
    "511011002": "N02",
    "511011003": "N03",
    "511011100": "N04",
    "511011101": "N05",
    "511011102": "N06",
    "511011103": "N07",
    "511011104": "N08",
    "511011105": "N09",
    "511011106": "N10",
    "511011107": "N11",
    "511011108": "N12",
    "511011109": "N13",
    "511011110": "N14",
    "511011111": "N15",
    "511011200": "N16",
    "511011201": "N17",
    "511011202": "N18",
    "511011203": "N19",
    "511011204": "N20",
    "511011205": "N21",
    "511011206": "N22",
    "511011207": "N23",
    "511011208": "N24",
    "511011209": "N25",
    "511011210": "N26",
    "511011211": "N27",
    "511011212": "N28",
    "511011213": "N29",
}


@dataclass(frozen=True)
class RegionConfig:
    region: str
    label: str
    n_blocks: int
    pt_glob: Path
    source_blocks: Path
    source_dltb: Path | None
    townships_json: Path | None
    townships: dict[str, str] | None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _workspace_root(repo: Path) -> Path:
    # Repo is normally D:/test/_publish/arcgis-farmland-mpc.
    return repo.parents[1]


def _copy_file_once(src: Path, dst: Path, force: bool = False) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if force or not dst.exists() or src.stat().st_size != dst.stat().st_size:
        shutil.copy2(src, dst)


def _copy_tree_once(src: Path, dst: Path, force: bool = False) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if force and dst.exists():
        shutil.rmtree(dst)
    if not dst.exists():
        shutil.copytree(src, dst)


def _find_neijiang_gpkg(adk_root: Path) -> Path:
    matches = [p for p in adk_root.rglob("DLTB_with_slope.gpkg") if p.is_file()]
    if not matches:
        raise FileNotFoundError(
            f"No DLTB_with_slope.gpkg found under {adk_root}. "
            "Pass --neijiang-dltb explicitly."
        )
    # The Neijiang restricted dataset is the largest DLTB_with_slope.gpkg under
    # D:/adk on the working machine. Keep this deterministic by sorting.
    matches.sort(key=lambda p: (p.stat().st_size, str(p)), reverse=True)
    return matches[0]


def build_region_config(
    region: str,
    repo: Path,
    source_root: Path,
    neijiang_dltb: Path | None,
) -> RegionConfig:
    if region == "bishan":
        return RegionConfig(
            region="bishan",
            label="Bishan District",
            n_blocks=2600,
            pt_glob=repo / "paper/checkpoints/bishan/contrastive_5seed/ensemble_seed*_lam5.0_member*.pt",
            source_blocks=source_root / "results_real/blocks",
            source_dltb=source_root / "dem_slope_analysis/output/DLTB_with_slope.gpkg",
            townships_json=source_root / "townships.json",
            townships=None,
        )
    if region == "neijiang":
        pt_glob = source_root / "neijiang_cross_region/ensembles/baseline/ensemble_seed*_lam5.0_member*.pt"
        if not glob.glob(str(pt_glob)):
            pt_glob = repo / "paper/checkpoints/neijiang/baseline/ensemble_seed*_lam5.0_member*.pt"
        return RegionConfig(
            region="neijiang",
            label="Neijiang Dongxing",
            n_blocks=3711,
            pt_glob=pt_glob,
            source_blocks=source_root / "neijiang_cross_region/blocks",
            source_dltb=neijiang_dltb,
            townships_json=None,
            townships=NEIJIANG_TOWNSHIPS,
        )
    raise ValueError(f"Unknown region: {region}")


def ensure_prepared(cfg: RegionConfig, out_root: Path, force: bool = False) -> Path:
    prepared = out_root / f"prepared_{cfg.region}"
    dltb_dst = prepared / "dem_slope_analysis/output/DLTB_with_slope.gpkg"
    blocks_dst = prepared / "results_real/blocks"
    townships_dst = prepared / "townships.json"

    if cfg.source_dltb is None:
        raise FileNotFoundError(f"No source DLTB configured for {cfg.region}")

    print(f"[prepared:{cfg.region}] {prepared}", flush=True)
    _copy_file_once(cfg.source_dltb, dltb_dst, force=force)
    _copy_tree_once(cfg.source_blocks, blocks_dst, force=force)
    if cfg.townships_json is not None:
        _copy_file_once(cfg.townships_json, townships_dst, force=force)
    else:
        townships_dst.write_text(json.dumps(cfg.townships, indent=2), encoding="utf-8")
    return prepared


def convert_pt_to_onnx(cfg: RegionConfig, onnx_root: Path, force: bool = False) -> Path:
    import torch

    from farmland_mpc.train_ensemble import _export_onnx
    from farmland_mpc.transition_model import TransitionModel

    pat = re.compile(r"ensemble_seed(\d+)_lam[\d.]+_member(\d+)\.pt$")
    files = sorted(glob.glob(str(cfg.pt_glob)))
    if not files:
        raise FileNotFoundError(f"No .pt files matched {cfg.pt_glob}")

    grouped: dict[int, dict[int, Path]] = {}
    for name in files:
        m = pat.search(name)
        if not m:
            continue
        grouped.setdefault(int(m.group(1)), {})[int(m.group(2))] = Path(name)

    region_onnx = onnx_root / cfg.region
    region_onnx.mkdir(parents=True, exist_ok=True)
    for seed in sorted(grouped):
        members = grouped[seed]
        if set(members) != {0, 1, 2}:
            raise RuntimeError(f"{cfg.region} seed {seed} has incomplete members: {sorted(members)}")
        out_dir = region_onnx / f"ensemble_seed{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[convert:{cfg.region}] seed {seed} -> {out_dir}", flush=True)
        for member in range(3):
            out_path = out_dir / f"ensemble_member{member}.onnx"
            if out_path.exists() and not force:
                continue
            state = torch.load(members[member], map_location="cpu", weights_only=True)
            model = TransitionModel(n_blocks=cfg.n_blocks, k_global=12)
            model.load_state_dict(state)
            model.eval()
            _export_onnx(
                model,
                cfg.n_blocks,
                12,
                out_path,
                say=lambda msg: print(f"  {msg}", flush=True),
            )
        provenance = {
            "region": cfg.region,
            "n_blocks": cfg.n_blocks,
            "pt_glob": str(cfg.pt_glob),
            "members": {
                str(member): str(members[member])
                for member in sorted(members)
            },
        }
        (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return region_onnx


def _stats(values: Iterable[float]) -> dict[str, float | None]:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": None, "std_population": None, "std_sample": None, "min": None, "max": None}
    return {
        "mean": float(arr.mean()),
        "std_population": float(arr.std(ddof=0)),
        "std_sample": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def run_no_net_loss(
    cfg: RegionConfig,
    prepared: Path,
    onnx_region_root: Path,
    out_root: Path,
    seeds: list[int],
    max_steps: int | None,
    force: bool,
    threads: int,
) -> Path:
    from farmland_mpc.mpc_plan import run as plan_run

    run_root = out_root / "no_net_loss" / cfg.region
    run_root.mkdir(parents=True, exist_ok=True)
    per_seed = []

    for seed in seeds:
        ens_dir = onnx_region_root / f"ensemble_seed{seed}"
        if not any(ens_dir.glob("ensemble_member*.onnx")):
            raise FileNotFoundError(f"Missing ONNX ensemble: {ens_dir}")
        cell_dir = run_root / f"seed{seed}"
        summary_path = cell_dir / "mpc_summary.json"
        if summary_path.exists() and not force:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            print(f"[run:{cfg.region}] seed {seed} reused {summary_path}", flush=True)
        else:
            print(f"[run:{cfg.region}] seed {seed} no-net-loss MPC", flush=True)
            t0 = time.time()
            summary = plan_run(
                ensemble_dir=str(ens_dir),
                out_dir=str(cell_dir),
                horizon=5,
                top_k=50,
                gamma=0.99,
                n_episodes=1,
                continuation="greedy",
                scoring="reward",
                threads=threads,
                seed_offset=seed,
                prepared_dir=str(prepared),
                env_kind="county",
                cultivated_area_floor_delta_ha=0.0,
                max_steps=max_steps,
            )
            summary["wall_s_total_runner"] = time.time() - t0

        ep = summary["results"][0]
        row = {
            "seed": seed,
            "ensemble_dir": str(ens_dir),
            "run_dir": str(cell_dir),
            "slope_change_pct": float(ep.get("slope_change_pct", math.nan)),
            "cont_change": float(ep.get("cont_change", math.nan)),
            "baimu_count_change": int(ep.get("baimu_count_change", 0)),
            "baimu_area_change_ha": float(ep.get("baimu_area_change_ha", math.nan)),
            "cultivated_area_change_ha": float(ep.get("cultivated_area_change_ha", math.nan)),
            "cultivated_area_change_pct": float(ep.get("cultivated_area_change_pct", math.nan)),
            "total_reward": float(ep.get("total_reward", math.nan)),
            "swaps_completed": int(ep.get("swaps_completed", 0)),
            "steps_run": int(ep.get("steps_run", 0)),
            "total_time_s": float(ep.get("total_time_s", math.nan)),
        }
        print(
            f"  seed {seed}: slope={row['slope_change_pct']:+.4f}% "
            f"cultivated={row['cultivated_area_change_ha']:+.2f}ha "
            f"baimu={row['baimu_area_change_ha']:+.2f}ha "
            f"cont={row['cont_change']:+.5f}",
            flush=True,
        )
        per_seed.append(row)

        report = aggregate_report(cfg, prepared, per_seed, max_steps)
        (run_root / "multiensemble_no_net_loss.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        write_markdown(run_root / "multiensemble_no_net_loss.md", report)

    return run_root / "multiensemble_no_net_loss.json"


def aggregate_report(
    cfg: RegionConfig,
    prepared: Path,
    per_seed: list[dict],
    max_steps: int | None,
) -> dict:
    metrics = [
        "slope_change_pct",
        "cont_change",
        "baimu_count_change",
        "baimu_area_change_ha",
        "cultivated_area_change_ha",
        "cultivated_area_change_pct",
        "total_reward",
        "swaps_completed",
        "total_time_s",
    ]
    return {
        "experiment": "multi_ensemble_no_net_loss_constrained_mpc",
        "region": cfg.region,
        "label": cfg.label,
        "prepared_dir": str(prepared),
        "constraint": {
            "cultivated_area_floor_delta_ha": 0.0,
            "interpretation": "Each committed parcel pair must keep cumulative cultivated area at or above the initial area.",
        },
        "mpc_config": {
            "horizon": 5,
            "top_k": 50,
            "gamma": 0.99,
            "continuation": "greedy",
            "scoring": "reward",
            "episodes_per_ensemble": 1,
            "max_steps": max_steps,
        },
        "n_ensembles_completed": len(per_seed),
        "per_seed": per_seed,
        "cross_seed": {
            metric: _stats(row[metric] for row in per_seed)
            for metric in metrics
        },
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        f"# {report['label']} multi-ensemble no-net-loss constrained MPC",
        "",
        f"Completed ensembles: {report['n_ensembles_completed']}",
        "",
        "| Seed | Slope delta (%) | Contiguity delta | Baimu count delta | Baimu area delta (ha) | Cultivated area delta (ha) | Reward | Swaps | Time (s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['slope_change_pct']:+.4f} | "
            f"{row['cont_change']:+.5f} | {row['baimu_count_change']:+d} | "
            f"{row['baimu_area_change_ha']:+.2f} | "
            f"{row['cultivated_area_change_ha']:+.2f} | "
            f"{row['total_reward']:+.2f} | {row['swaps_completed']} | "
            f"{row['total_time_s']:.1f} |"
        )
    lines.extend(["", "## Cross-seed mean +/- sample SD", ""])
    for metric, vals in report["cross_seed"].items():
        lines.append(
            f"- {metric}: {vals['mean']:+.6g} +/- {vals['std_sample']:.6g} "
            f"(range {vals['min']:+.6g} to {vals['max']:+.6g})"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    repo = _repo_root()
    source = _workspace_root(repo)
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", choices=["bishan", "neijiang"], default=["bishan", "neijiang"])
    ap.add_argument("--source-root", type=Path, default=source)
    ap.add_argument("--out-root", type=Path, default=repo / "runs/scirep_extra")
    ap.add_argument("--neijiang-dltb", type=Path, default=None)
    ap.add_argument("--adk-root", type=Path, default=Path("D:/adk"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--force-prepared", action="store_true")
    ap.add_argument("--force-convert", action="store_true")
    ap.add_argument("--force-run", action="store_true")
    ap.add_argument("--skip-convert", action="store_true")
    ap.add_argument("--skip-run", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    repo = _repo_root()
    source_root = args.source_root.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    neijiang_dltb = args.neijiang_dltb
    if neijiang_dltb is None and "neijiang" in args.regions:
        neijiang_dltb = _find_neijiang_gpkg(args.adk_root)
        print(f"[neijiang] using DLTB {neijiang_dltb}", flush=True)

    for region in args.regions:
        cfg = build_region_config(region, repo, source_root, neijiang_dltb)
        prepared = ensure_prepared(cfg, out_root, force=args.force_prepared)
        onnx_root = out_root / "onnx"
        if not args.skip_convert:
            convert_pt_to_onnx(cfg, onnx_root, force=args.force_convert)
        if not args.skip_run:
            run_no_net_loss(
                cfg=cfg,
                prepared=prepared,
                onnx_region_root=onnx_root / cfg.region,
                out_root=out_root,
                seeds=args.seeds,
                max_steps=args.max_steps,
                force=args.force_run,
                threads=args.threads,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
