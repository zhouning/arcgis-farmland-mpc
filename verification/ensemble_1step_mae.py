# -*- coding: utf-8 -*-
"""
ensemble_1step_mae.py — quantify ensemble 1-step prediction error against
the TRUE env.step on the same trajectory MPC would actually visit.

For each of N steps:
  1. Build (bf, gf, mask) from the env
  2. Pick action via MPC (so that the action distribution matches the one
     used to generate the headline result, NOT a uniform-random OOD distribution)
  3. ensemble.batch_predict([bf],[gf],[action]) -> (nbf_pred, ngf_pred, r_pred, r_std)
  4. env.step(action) -> (next_obs_true, r_true, info)
  5. Decode (nbf_true, ngf_true) from next_obs_true
  6. Log per-step diff for: r, ngf[1] global slope, ngf[4] slope improvement,
     ngf[5] cont improvement, ngf[6] baimu count, ngf[7] baimu area frac

Outputs:
  - mae_per_step.json        full per-step record
  - mae_summary.json         aggregate MAE/RMSE/percentiles
  - mae_run.log              tee'd stdout

Usage:
  python ensemble_1step_mae.py \\
    --prepared <prepared_dir> \\
    --ensemble <prepared_dir>/tool3 \\
    --proj-crs EPSG:32648 \\
    --n-steps 100 \\
    --out-dir mae_run

Wall time: ~25 minutes for 100 steps on a 53k-parcel county on a desktop CPU.
After it finishes, run mae_aggregate.py on mae_per_step.json to produce
mae_summary.json with reward MAE/RMSE/percentiles and Spearman rank
correlation between predicted and true reward (the metric that actually
matters at planning time)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--ensemble", required=True,
                    help="dir containing ensemble_member*.onnx")
    ap.add_argument("--proj-crs", default="EPSG:32648")
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "mae_run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("mae")

    log.info(f"prepared = {args.prepared}")
    log.info(f"ensemble = {args.ensemble}")
    log.info(f"steps={args.n_steps} horizon={args.horizon} top_k={args.top_k} "
             f"gamma={args.gamma} seed={args.seed}")

    # Imports happen after logging is set up so failures show up in the log
    from farmland_mpc.blocks_env import make_env
    from farmland_mpc.ensemble_runner import EnsembleOrtRunner
    from farmland_mpc.mpc_plan import mpc_select_action

    log.info("Loading ensemble ...")
    ensemble = EnsembleOrtRunner(args.ensemble, n_threads=args.threads)
    log.info(f"  {ensemble.n_members} members loaded; n_blocks={ensemble.n_blocks}")

    log.info("Building env (this is the expensive step) ...")
    t0 = time.time()
    env = make_env(prepared_dir=args.prepared, proj_crs=args.proj_crs)
    log.info(f"  env built in {time.time()-t0:.1f}s; n_blocks={env.n_blocks}, "
             f"n_parcels={env.n_parcels}, max_steps={env.max_steps}")
    ensemble.assert_compatible(env.n_blocks)

    # Cap steps (smoke runs)
    n_steps = min(args.n_steps, env.max_steps)
    env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    # Per-step records
    rec = {
        "step": [],
        "action": [],
        "r_pred": [],          # ensemble mean reward
        "r_std": [],           # ensemble std (epistemic uncertainty)
        "r_true": [],          # actual env reward
        "gf_pred": [],         # ensemble predicted global features (12,)
        "gf_true": [],         # actual next global features (12,)
        "step_time_s": [],
    }

    # Constants from county_env
    K_BLOCK = 17

    log.info(f"Rolling {n_steps} steps with MPC action selection ...")
    for step in range(n_steps):
        t_step = time.time()
        bf = env._get_block_features()
        gf = env._get_global_features()
        mask = env.action_masks()

        # MPC pick (canonical config: H=5, K=50, gamma=0.99, greedy continuation, scoring=reward)
        action, _ = mpc_select_action(
            ensemble, bf, gf, mask,
            horizon=args.horizon, top_k=args.top_k, gamma=args.gamma,
            n_rollouts=1, continuation="greedy",
            scoring="reward", rng=rng,
        )

        # Single 1-step prediction for the chosen action (cheap; ~5ms)
        bf_b = bf[np.newaxis].astype(np.float32)
        gf_b = gf[np.newaxis].astype(np.float32)
        a_b = np.array([action], dtype=np.int64)
        nbf_pred, ngf_pred, r_pred, r_std = ensemble.batch_predict(bf_b, gf_b, a_b)
        # squeeze batch dim
        ngf_pred_v = ngf_pred[0]
        r_pred_v = float(r_pred[0])
        r_std_v = float(r_std[0])

        # True env step
        obs_true, r_true, terminated, truncated, info = env.step(int(action))
        # Decode next gf from obs (layout: [n_blocks*K_BLOCK | K_GLOBAL])
        ngf_true_v = obs_true[env.n_blocks * K_BLOCK:].astype(np.float32)

        rec["step"].append(step)
        rec["action"].append(int(action))
        rec["r_pred"].append(r_pred_v)
        rec["r_std"].append(r_std_v)
        rec["r_true"].append(float(r_true))
        rec["gf_pred"].append(ngf_pred_v.tolist())
        rec["gf_true"].append(ngf_true_v.tolist())
        rec["step_time_s"].append(time.time() - t_step)

        if (step + 1) % 10 == 0 or step == 0 or step + 1 == n_steps:
            log.info(f"  step {step+1:3d}/{n_steps} "
                     f"r_pred={r_pred_v:+.3f} r_true={r_true:+.3f} "
                     f"r_std={r_std_v:.3f} "
                     f"slope_pct={info['slope_change_pct']:+.4f}% "
                     f"step={rec['step_time_s'][-1]:.1f}s")
            # Incremental checkpoint so a crash in aggregate doesn't lose the trajectory.
            try:
                with open(out_dir / "mae_per_step.json", "w", encoding="utf-8") as f:
                    json.dump(rec, f, indent=2, ensure_ascii=False)
            except Exception as ex:
                log.warning(f"  checkpoint write failed: {ex}")

        if terminated or truncated:
            log.info(f"  episode ended early at step {step+1}")
            break

    # Aggregate (defensive — a numpy RuntimeWarning shouldn't kill us before
    # we get to write the summary).
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    np.seterr(all="ignore")

    log.info("Aggregating per-step records ...")
    try:
        r_pred = np.array(rec["r_pred"], dtype=np.float64)
        r_std = np.array(rec["r_std"], dtype=np.float64)
        r_true = np.array(rec["r_true"], dtype=np.float64)
        gf_pred = np.array(rec["gf_pred"], dtype=np.float64)   # (T, 12)
        gf_true = np.array(rec["gf_true"], dtype=np.float64)
    except Exception as ex:
        log.error(f"  aggregate failed at array construction: {ex}")
        log.error(f"  shapes: r_pred={len(rec['r_pred'])}, gf_pred row0 len="
                  f"{len(rec['gf_pred'][0]) if rec['gf_pred'] else 'N/A'}")
        # dump raw record so we can post-mortem
        with open(out_dir / "mae_per_step.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        raise

    err = r_pred - r_true
    abs_err = np.abs(err)
    rel_err = abs_err / (np.abs(r_true) + 1e-8)

    # Key gf channels (see county_env._get_global_features)
    #   [1] cur_slope norm
    #   [4] slope_improvement (negative = good)
    #   [5] cont_improvement
    #   [6] baimu_count_norm
    #   [7] baimu_area_frac
    channels = {
        "gf1_global_slope_norm": 1,
        "gf4_slope_improvement": 4,
        "gf5_cont_improvement": 5,
        "gf6_baimu_count_norm": 6,
        "gf7_baimu_area_frac": 7,
    }
    gf_metrics = {}
    for name, idx in channels.items():
        e = gf_pred[:, idx] - gf_true[:, idx]
        gf_metrics[name] = {
            "mae": float(np.mean(np.abs(e))),
            "rmse": float(np.sqrt(np.mean(e**2))),
            "median_abs": float(np.median(np.abs(e))),
            "p90_abs": float(np.percentile(np.abs(e), 90)),
            "p99_abs": float(np.percentile(np.abs(e), 99)),
            "mean_signed": float(np.mean(e)),
        }

    # Member-disagreement vs error correlation
    if abs_err.std() > 0 and r_std.std() > 0:
        corr_std_err = float(np.corrcoef(r_std, abs_err)[0, 1])
    else:
        corr_std_err = 0.0

    summary = {
        "config": {
            "prepared": args.prepared,
            "ensemble": args.ensemble,
            "horizon": args.horizon, "top_k": args.top_k, "gamma": args.gamma,
            "n_steps_run": int(len(rec["step"])),
            "n_steps_requested": args.n_steps,
            "max_steps": int(env.max_steps),
            "n_blocks": int(env.n_blocks), "n_parcels": int(env.n_parcels),
            "seed": args.seed,
        },
        "reward_1step": {
            "n": int(len(r_pred)),
            "r_true_mean": float(r_true.mean()),
            "r_true_std": float(r_true.std()),
            "r_pred_mean": float(r_pred.mean()),
            "mae": float(abs_err.mean()),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "median_abs_err": float(np.median(abs_err)),
            "p90_abs_err": float(np.percentile(abs_err, 90)),
            "p99_abs_err": float(np.percentile(abs_err, 99)),
            "median_rel_err": float(np.median(rel_err)),
            "p90_rel_err": float(np.percentile(rel_err, 90)),
            "mean_signed_err": float(err.mean()),
            "ensemble_std_mean": float(r_std.mean()),
            "ensemble_std_median": float(np.median(r_std)),
            "corr_ensemble_std_abs_err": corr_std_err,
        },
        "global_features_1step": gf_metrics,
        "trajectory_check": {
            "final_slope_change_pct_env": float(info.get("slope_change_pct", 0.0)),
            "final_cont_change_env": float(info.get("cont_change", 0.0)),
            "final_baimu_area_change_ha_env": float(info.get("baimu_area_change_ha", 0.0)),
        },
    }

    with open(out_dir / "mae_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / "mae_per_step.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)

    log.info("")
    log.info("=" * 70)
    log.info("ENSEMBLE 1-STEP MAE SUMMARY")
    log.info("=" * 70)
    log.info(f"  steps                        : {summary['config']['n_steps_run']}")
    log.info(f"  reward MAE                   : {summary['reward_1step']['mae']:.4f}")
    log.info(f"  reward RMSE                  : {summary['reward_1step']['rmse']:.4f}")
    log.info(f"  reward median |err|          : {summary['reward_1step']['median_abs_err']:.4f}")
    log.info(f"  reward p90 |err|             : {summary['reward_1step']['p90_abs_err']:.4f}")
    log.info(f"  reward median rel-err        : {summary['reward_1step']['median_rel_err']:.3%}")
    log.info(f"  ensemble std (mean)          : {summary['reward_1step']['ensemble_std_mean']:.4f}")
    log.info(f"  corr(ensemble_std, |err|)    : {summary['reward_1step']['corr_ensemble_std_abs_err']:+.3f}")
    log.info(f"  r_true range [{r_true.min():+.3f}, {r_true.max():+.3f}], mean {r_true.mean():+.3f}")
    log.info("")
    log.info(" gf channel                       MAE      RMSE    p90|err|    p99|err|")
    for name, m in gf_metrics.items():
        log.info(f"  {name:<28} {m['mae']:.4f}  {m['rmse']:.4f}   {m['p90_abs']:.4f}    {m['p99_abs']:.4f}")
    log.info(f"  trajectory final slope_pct    : "
             f"{summary['trajectory_check']['final_slope_change_pct_env']:+.4f}% "
             f"(should match the corresponding episode in mpc_summary.json)")
    log.info("=" * 70)
    log.info(f"  outputs: {out_dir}")


if __name__ == "__main__":
    main()
