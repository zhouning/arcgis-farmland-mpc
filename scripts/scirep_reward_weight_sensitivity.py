#!/usr/bin/env python3
"""Retrained reward-weight sensitivity for the Scientific Reports revision.

This is a real Tool 2 -> Tool 3 -> Tool 4 sensitivity runner. It does not use
Tool 4 runtime reward overrides as a substitute for retraining. For each reward
profile, the script creates an isolated prepared directory, re-samples Tool 2
under that profile's reward weights, re-trains a contrastive ensemble, then runs
no-net-loss constrained MPC.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class RewardProfile:
    profile_id: str
    label: str
    description: str
    weights: dict[str, float]


DEFAULT_WEIGHTS = {
    "slope_weight": 4000.0,
    "cont_weight": 500.0,
    "baimu_weight": 1500.0,
    "baimu_bonus": 5.0,
    "baimu_area_penalty": 2000.0,
}


PROFILES = {
    "default": RewardProfile(
        profile_id="default",
        label="Default reward",
        description="Paper default reward weights.",
        weights=DEFAULT_WEIGHTS,
    ),
    "baimu_low": RewardProfile(
        profile_id="baimu_low",
        label="Baimu low",
        description="Half-weighted baimu-fang area and loss penalty.",
        weights={
            **DEFAULT_WEIGHTS,
            "baimu_weight": 750.0,
            "baimu_bonus": 2.5,
            "baimu_area_penalty": 1000.0,
        },
    ),
    "baimu_high": RewardProfile(
        profile_id="baimu_high",
        label="Baimu high",
        description="Double-weighted baimu-fang area and loss penalty.",
        weights={
            **DEFAULT_WEIGHTS,
            "baimu_weight": 3000.0,
            "baimu_bonus": 10.0,
            "baimu_area_penalty": 4000.0,
        },
    ),
}


def _copy_prepared_template(src: Path, dst: Path, force: bool) -> None:
    if force and dst.exists():
        shutil.rmtree(dst)
    if dst.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("dem_slope_analysis", "results_real"):
        source = src / name
        target = dst / name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copytree(source, target)
    townships = src / "townships.json"
    if not townships.exists():
        raise FileNotFoundError(townships)
    shutil.copy2(townships, dst / "townships.json")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _has_onnx_ensemble(path: Path, n_members: int) -> bool:
    return all((path / f"ensemble_member{i}.onnx").exists() for i in range(n_members))


def _stats(values: Iterable[float]) -> dict[str, float | None]:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": None, "std_sample": None, "min": None, "max": None}
    return {
        "mean": float(arr.mean()),
        "std_sample": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _run_plan_in_subprocess(
    plan_config: dict,
    summary_path: Path,
    run_cmd=subprocess.run,
) -> dict:
    """Run Tool 4 in a fresh Python process and load its written summary."""
    code = "\n".join(
        [
            "import json",
            "from farmland_mpc.mpc_plan import run",
            f"config = json.loads({json.dumps(plan_config)!r})",
            "run(**config)",
        ]
    )
    completed = run_cmd(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        check=False,
    )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, completed.args)
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Tool 4 subprocess finished but did not write {summary_path}"
        )
    return _load_json(summary_path)


def _run_profile(args, profile: RewardProfile) -> dict:
    from farmland_mpc.sample import run as sample_run
    from farmland_mpc.train_ensemble import run as train_run

    profile_root = args.out_root / profile.profile_id
    prepared = profile_root / "prepared"
    plan_dir = profile_root / "plan_no_net_loss"
    profile_root.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {profile.profile_id}: {profile.label} ===", flush=True)
    _copy_prepared_template(args.prepared_template, prepared, force=args.force_prepared)

    tool2_summary_path = prepared / "tool2" / "sample_transitions_summary.json"
    if tool2_summary_path.exists() and not args.force_sample:
        print(f"[Tool 2] reuse {tool2_summary_path}", flush=True)
        sample_summary = _load_json(tool2_summary_path)
    else:
        t0 = time.time()
        sample_summary = sample_run(
            prepared_dir=prepared,
            n_transition_episodes=args.n_transition_episodes,
            n_pairwise_states=args.n_pairwise_states,
            n_pairwise_actions=args.n_pairwise_actions,
            seed=args.seed,
            proj_crs=args.proj_crs,
            env_kind="county",
            **profile.weights,
        )
        sample_summary["runner_elapsed_s"] = time.time() - t0

    tool3_dir = prepared / "tool3"
    train_summary_path = tool3_dir / "train_summary.json"
    if (
        train_summary_path.exists()
        and _has_onnx_ensemble(tool3_dir, args.n_members)
        and not args.force_train
    ):
        print(f"[Tool 3] reuse {train_summary_path}", flush=True)
        train_summary = _load_json(train_summary_path)
    else:
        t0 = time.time()
        train_summary = train_run(
            prepared_dir=str(prepared),
            n_members=args.n_members,
            epochs=args.epochs,
            patience=args.patience,
            lambda_rank=args.lambda_rank,
            margin=args.margin,
            batch_size=args.batch_size,
            seed_base=args.train_seed_base,
            torch_threads=args.torch_threads,
            out_subdir="tool3",
        )
        train_summary["runner_elapsed_s"] = time.time() - t0

    plan_summary_path = plan_dir / "mpc_summary.json"
    if plan_summary_path.exists() and not args.force_plan:
        print(f"[Tool 4] reuse {plan_summary_path}", flush=True)
        plan_summary = _load_json(plan_summary_path)
    else:
        t0 = time.time()
        plan_summary = _run_plan_in_subprocess(
            {
                "ensemble_dir": str(tool3_dir),
                "out_dir": str(plan_dir),
                "horizon": args.horizon,
                "top_k": args.top_k,
                "gamma": args.gamma,
                "mpc_batch_size": args.mpc_batch_size,
                "n_episodes": 1,
                "continuation": "greedy",
                "scoring": "reward",
                "threads": args.plan_threads,
                "seed_offset": args.seed,
                "prepared_dir": str(prepared),
                "env_kind": "county",
                "cultivated_area_floor_delta_ha": 0.0,
                "max_steps": args.max_steps,
            },
            plan_summary_path,
        )
        plan_summary["runner_elapsed_s"] = time.time() - t0

    ep = plan_summary["results"][0]
    row = {
        "profile": profile.profile_id,
        "label": profile.label,
        "description": profile.description,
        "weights": profile.weights,
        "prepared_dir": str(prepared),
        "tool2_summary": str(tool2_summary_path),
        "tool3_summary": str(train_summary_path),
        "plan_summary": str(plan_summary_path),
        "sample_reward_std_median": sample_summary.get("pairwise", {}).get("reward_std_median"),
        "sample_reward_mean": sample_summary.get("pairwise", {}).get("reward_mean"),
        "final_ranking_acc_mean": _stats(
            member.get("final_ranking_acc", np.nan)
            for member in train_summary.get("members", [])
        )["mean"],
        "slope_change_pct": float(ep.get("slope_change_pct", np.nan)),
        "cont_change": float(ep.get("cont_change", np.nan)),
        "baimu_count_change": int(ep.get("baimu_count_change", 0)),
        "baimu_area_change_ha": float(ep.get("baimu_area_change_ha", np.nan)),
        "cultivated_area_change_ha": float(ep.get("cultivated_area_change_ha", np.nan)),
        "cultivated_area_change_pct": float(ep.get("cultivated_area_change_pct", np.nan)),
        "total_reward": float(ep.get("total_reward", np.nan)),
        "swaps_completed": int(ep.get("swaps_completed", 0)),
        "steps_run": int(ep.get("steps_run", 0)),
        "total_time_s": float(ep.get("total_time_s", np.nan)),
    }
    print(
        f"[result] slope={row['slope_change_pct']:+.4f}% "
        f"baimu_area={row['baimu_area_change_ha']:+.2f}ha "
        f"cultivated={row['cultivated_area_change_ha']:+.2f}ha "
        f"rank_acc={row['final_ranking_acc_mean']}",
        flush=True,
    )
    return row


def _write_outputs(args, rows: list[dict]) -> None:
    region_label = getattr(args, "region_label", "Bishan").strip() or "Bishan"
    region_id = region_label.lower().replace(" ", "_")
    report = {
        "experiment": "retrained_reward_weight_sensitivity",
        "region": region_id,
        "prepared_template": str(args.prepared_template),
        "scope": (
            "Single retrained 3-member ensemble per reward profile unless "
            "--n-members is changed. All profile runs use the no-net-loss "
            "cultivated-area execution floor."
        ),
        "mpc_config": {
            "horizon": args.horizon,
            "top_k": args.top_k,
            "gamma": args.gamma,
            "mpc_batch_size": args.mpc_batch_size,
            "continuation": "greedy",
            "scoring": "reward",
            "cultivated_area_floor_delta_ha": 0.0,
            "max_steps": args.max_steps,
        },
        "tool2_config": {
            "n_transition_episodes": args.n_transition_episodes,
            "n_pairwise_states": args.n_pairwise_states,
            "n_pairwise_actions": args.n_pairwise_actions,
            "seed": args.seed,
        },
        "tool3_config": {
            "n_members": args.n_members,
            "epochs": args.epochs,
            "patience": args.patience,
            "lambda_rank": args.lambda_rank,
            "margin": args.margin,
            "batch_size": args.batch_size,
            "train_seed_base": args.train_seed_base,
        },
        "profiles": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        f"# Retrained {region_label} reward-weight sensitivity",
        "",
        "Each profile re-runs Tool 2 sampling and Tool 3 ensemble training before no-net-loss Tool 4 planning.",
        "",
        "| Profile | Slope delta (%) | Contiguity delta | Baimu count delta | Baimu area delta (ha) | Cultivated area delta (ha) | Rank acc mean | Reward std median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        rank_acc = r["final_ranking_acc_mean"]
        rank_txt = "" if rank_acc is None else f"{rank_acc:.4f}"
        std_val = r["sample_reward_std_median"]
        std_txt = "" if std_val is None else f"{float(std_val):.4f}"
        lines.append(
            f"| {r['label']} | {r['slope_change_pct']:+.4f} | "
            f"{r['cont_change']:+.5f} | {r['baimu_count_change']:+d} | "
            f"{r['baimu_area_change_ha']:+.2f} | "
            f"{r['cultivated_area_change_ha']:+.2f} | {rank_txt} | {std_txt} |"
        )
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region-label", default="Bishan")
    ap.add_argument(
        "--prepared-template",
        type=Path,
        default=REPO_ROOT / "runs/scirep_extra/prepared_bishan",
    )
    ap.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "runs/scirep_reward_sensitivity/bishan",
    )
    ap.add_argument("--profiles", nargs="+", choices=sorted(PROFILES), default=["default", "baimu_low", "baimu_high"])
    ap.add_argument("--n-transition-episodes", type=int, default=60)
    ap.add_argument("--n-pairwise-states", type=int, default=1000)
    ap.add_argument("--n-pairwise-actions", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--proj-crs", default=None)
    ap.add_argument("--n-members", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lambda-rank", type=float, default=5.0)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--train-seed-base", type=int, default=0)
    ap.add_argument("--torch-threads", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--mpc-batch-size", type=int, default=1024)
    ap.add_argument("--plan-threads", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--force-prepared", action="store_true")
    ap.add_argument("--force-sample", action="store_true")
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--force-plan", action="store_true")
    ap.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "runs/scirep_reward_sensitivity/bishan/reward_weight_sensitivity.json",
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=REPO_ROOT / "runs/scirep_reward_sensitivity/bishan/reward_weight_sensitivity.md",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rows = []
    for profile_id in args.profiles:
        rows.append(_run_profile(args, PROFILES[profile_id]))
        _write_outputs(args, rows)
    print(f"\nWrote {args.out_json}", flush=True)
    print(f"Wrote {args.out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
