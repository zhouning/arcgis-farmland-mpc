# -*- coding: utf-8 -*-
"""
mae_aggregate.py — offline aggregation of per-step MAE records from
ensemble_1step_mae.py.

The main script writes mae_per_step.json incrementally; if the aggregate
stage crashes (e.g. numpy corrcoef segfault on some platforms), run this
script separately on the per-step JSON to produce mae_summary.json.

Usage:
    python mae_aggregate.py --per-step mae_run/mae_per_step.json

Outputs mae_run/mae_summary.json in the same directory as --per-step.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Hand-rolled Pearson to avoid numpy.corrcoef segfault on some platforms."""
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-step", required=True,
                    help="Path to mae_per_step.json produced by ensemble_1step_mae.py")
    args = ap.parse_args()

    per_step_path = Path(args.per_step)
    out_path = per_step_path.parent / "mae_summary.json"

    with open(per_step_path, "r", encoding="utf-8") as f:
        r = json.load(f)

    rp = np.array(r["r_pred"], dtype=np.float64)
    rs = np.array(r["r_std"], dtype=np.float64)
    rt = np.array(r["r_true"], dtype=np.float64)
    gp = np.array(r["gf_pred"], dtype=np.float64)
    gt = np.array(r["gf_true"], dtype=np.float64)

    err = rp - rt
    abs_err = np.abs(err)
    rel_err = abs_err / (np.abs(rt) + 1e-8)

    ra = np.argsort(np.argsort(rp)).astype(float)
    rb = np.argsort(np.argsort(rt)).astype(float)
    spearman = pearson(ra, rb)
    corr = pearson(rs, abs_err)

    channels = {
        "gf1_global_slope_norm":  1,
        "gf4_slope_improvement":  4,
        "gf5_cont_improvement":   5,
        "gf6_baimu_count_norm":   6,
        "gf7_baimu_area_frac":    7,
    }
    gf_metrics: dict = {}
    for name, idx in channels.items():
        e = gp[:, idx] - gt[:, idx]
        gf_metrics[name] = {
            "mae":         float(np.mean(np.abs(e))),
            "rmse":        float(np.sqrt(np.mean(e ** 2))),
            "median_abs":  float(np.median(np.abs(e))),
            "p90_abs":     float(np.percentile(np.abs(e), 90)),
            "p99_abs":     float(np.percentile(np.abs(e), 99)),
            "mean_signed": float(np.mean(e)),
            "true_range":  [float(gt[:, idx].min()), float(gt[:, idx].max())],
            "true_std":    float(gt[:, idx].std()),
        }

    summary = {
        "n_steps_run": int(len(rp)),
        "reward_1step": {
            "r_true_mean":             float(rt.mean()),
            "r_true_std":              float(rt.std()),
            "r_true_min":              float(rt.min()),
            "r_true_max":              float(rt.max()),
            "r_pred_mean":             float(rp.mean()),
            "r_pred_std":              float(rp.std()),
            "r_pred_min":              float(rp.min()),
            "r_pred_max":              float(rp.max()),
            "mae":                     float(abs_err.mean()),
            "rmse":                    float(np.sqrt(np.mean(err ** 2))),
            "median_abs":              float(np.median(abs_err)),
            "p90_abs":                 float(np.percentile(abs_err, 90)),
            "p99_abs":                 float(np.percentile(abs_err, 99)),
            "mean_signed":             float(err.mean()),
            "median_rel":              float(np.median(rel_err)),
            "p90_rel":                 float(np.percentile(rel_err, 90)),
            "ensemble_std_mean":       float(rs.mean()),
            "ensemble_std_median":     float(np.median(rs)),
            "ensemble_std_max":        float(rs.max()),
            "corr_ensemble_std_abs_err": corr,
            "spearman_pred_true":      spearman,
            "naive_constant_mean_mae": float(np.abs(rt - rp.mean()).mean()),
            "naive_zero_mae":          float(np.abs(rt).mean()),
        },
        "global_features_1step": gf_metrics,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"n steps        : {len(rp)}")
    print(f"r_true         : mean={rt.mean():+.3f} std={rt.std():.3f} "
          f"range [{rt.min():+.3f}, {rt.max():+.3f}]")
    print(f"r_pred         : mean={rp.mean():+.3f} std={rp.std():.3f} "
          f"range [{rp.min():+.3f}, {rp.max():+.3f}]")
    print(f"reward MAE     : {abs_err.mean():.4f}")
    print(f"reward RMSE    : {np.sqrt(np.mean(err**2)):.4f}")
    print(f"median |err|   : {np.median(abs_err):.4f}")
    print(f"p90 |err|      : {np.percentile(abs_err, 90):.4f}")
    print(f"ensemble std   : mean {rs.mean():.4f}  median {np.median(rs):.4f}  max {rs.max():.4f}")
    print(f"corr(std,|err|): {corr:+.3f}")
    print(f"Spearman(rp,rt): {spearman:+.3f}")
    print(f"")
    print(f"{'gf channel':<30} {'MAE':>8} {'RMSE':>8} {'p90':>8} {'true_std':>10}")
    for name, m in gf_metrics.items():
        print(f"  {name:<28} {m['mae']:8.5f} {m['rmse']:8.5f} "
              f"{m['p90_abs']:8.5f} {m['true_std']:10.5f}")
    print(f"\nSummary written to {out_path}")


if __name__ == "__main__":
    main()
