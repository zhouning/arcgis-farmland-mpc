#!/usr/bin/env python3
"""5-seed cross-ensemble MPC eval, mirroring paper §5 / research-side
neijiang_cross_region/eval_mpc_neijiang.py protocol.

Iterates 5 independently trained ensembles (one per ensemble_seedX/ subdir
under prepared_dir), runs ONE deterministic MPC episode each (continuation
greedy, H=5, K=50, scoring=reward), aggregates per-seed and cross-seed stats
into a JSON matching the research-side schema, so we can diff against the
windows-side artefacts in neijiang_cross_region/.

Usage:
    python -m farmland_mpc.tests.eval_5seed_paper \\
        --prepared-dir runs/dongxing/prepared \\
        --out-json runs/dongxing/5seed_multiobj_results_paper.json \\
        --region "Neijiang Dongxing"
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from farmland_mpc.mpc_plan import run as plan_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_5seed_paper")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--region", type=str, default="Neijiang Dongxing")
    ap.add_argument("--ensemble-prefix", type=str, default="ensemble_seed",
                    help="Subdir prefix; expect ensemble_seed0..4 under prepared_dir.")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-episodes-per-seed", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--continuation", type=str, default="greedy")
    ap.add_argument("--scoring", type=str, default="reward")
    ap.add_argument("--lambda-rank", type=float, default=5.0,
                    help="Recorded in output for documentation only.")
    args = ap.parse_args()

    prepared_dir = args.prepared_dir.resolve()
    out_json = args.out_json.resolve()
    out_root = out_json.parent / out_json.stem
    out_root.mkdir(parents=True, exist_ok=True)

    log.info("5-seed eval: prepared=%s seeds=%d eps/seed=%d", prepared_dir, args.n_seeds, args.n_episodes_per_seed)

    all_results = {}
    for seed in range(args.n_seeds):
        ens_subdir = f"{args.ensemble_prefix}{seed}"
        ens_dir = prepared_dir / ens_subdir
        if not ens_dir.exists() or not any(ens_dir.glob("ensemble_member*.onnx")):
            raise FileNotFoundError(
                f"Expected ensemble at {ens_dir} (containing ensemble_member*.onnx). "
                f"Did you train --out-subdir {ens_subdir}?"
            )

        run_out = out_root / f"seed{seed}"
        log.info("=" * 60)
        log.info("EVAL SEED %d  ens=%s  out=%s", seed, ens_dir, run_out)
        log.info("=" * 60)
        t0 = time.time()
        summary = plan_run(
            ensemble_dir=str(ens_dir),
            out_dir=str(run_out),
            horizon=args.horizon,
            top_k=args.top_k,
            gamma=0.99,
            n_episodes=args.n_episodes_per_seed,
            continuation=args.continuation,
            scoring=args.scoring,
            prepared_dir=str(prepared_dir),
            seed_offset=seed,  # different episode RNG per ensemble seed
        )
        elapsed = time.time() - t0
        log.info("seed=%d done in %.1fs", seed, elapsed)

        # Adapt mpc_summary -> research-side per-seed schema
        per_seed = {
            'slopes':              [r['slope_change_pct']          for r in summary['results']],
            'rewards':             [r['total_reward']              for r in summary['results']],
            'times':               [r['total_time_s']              for r in summary['results']],
            'cont_deltas':         [r['cont_change']               for r in summary['results']],
            # research output stores cont_pcts as % of |C_0|; recompute approximately
            # by reading initial_cont from mpc_summary if present (not always emitted) — fallback to absolute delta percent.
            'cont_pcts':           [r['cont_change'] * 100.0       for r in summary['results']],
            'baimu_count_deltas':  [r['baimu_count_change']        for r in summary['results']],
            'baimu_area_deltas_ha':[r['baimu_area_change_ha']      for r in summary['results']],
        }
        all_results[seed] = per_seed

    # Cross-seed aggregation
    def _stats(getter):
        vals = [float(np.mean(getter(all_results[s]))) for s in range(args.n_seeds)]
        a = np.array(vals, dtype=float)
        return float(a.mean()), float(a.std())

    slope_mean, slope_std       = _stats(lambda d: d['slopes'])
    cont_pct_mean, cont_pct_std = _stats(lambda d: d['cont_pcts'])
    cont_d_mean, cont_d_std     = _stats(lambda d: d['cont_deltas'])
    bcnt_mean, bcnt_std         = _stats(lambda d: d['baimu_count_deltas'])
    barea_mean, barea_std       = _stats(lambda d: d['baimu_area_deltas_ha'])
    rew_mean, rew_std           = _stats(lambda d: d['rewards'])

    out = {
        'region': args.region,
        'mode': 'baseline',
        'lambda_rank': args.lambda_rank,
        'n_seeds': args.n_seeds,
        'eval_episodes_per_seed': args.n_episodes_per_seed,
        'per_seed': {str(s): all_results[s] for s in range(args.n_seeds)},
        'cross_seed': {
            'slope_pct_mean': slope_mean,
            'slope_pct_std':  slope_std,
            'cont_pct_mean':  cont_pct_mean,
            'cont_pct_std':   cont_pct_std,
            'cont_raw_delta_mean': cont_d_mean,
            'cont_raw_delta_std':  cont_d_std,
            'baimu_count_delta_mean': bcnt_mean,
            'baimu_count_delta_std':  bcnt_std,
            'baimu_area_delta_ha_mean': barea_mean,
            'baimu_area_delta_ha_std':  barea_std,
            'reward_mean': rew_mean,
            'reward_std':  rew_std,
        },
        'mpc_config': {
            'horizon': args.horizon, 'top_k': args.top_k,
            'continuation': args.continuation, 'scoring': args.scoring,
        },
        'package': 'farmland_mpc (open-source)',
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_json)
    print()
    print(f"Cross-seed aggregate ({args.region}, n={args.n_seeds}):")
    print(f"  slope %     : {slope_mean:+.4f} ± {slope_std:.4f}")
    print(f"  Δcont       : {cont_d_mean:+.5f} ± {cont_d_std:.5f}")
    print(f"  Δbaimu #    : {bcnt_mean:+.2f} ± {bcnt_std:.2f}")
    print(f"  Δbaimu (ha) : {barea_mean:+.2f} ± {barea_std:.2f}")
    print(f"  reward      : {rew_mean:+.2f} ± {rew_std:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
