# -*- coding: utf-8 -*-
"""
mpc_true_env.py — replace the learned ensemble with the actual env.step
inside the MPC scoring loop. Goal: bound how much the ensemble's
prediction error costs us in final slope. If the slope this script
achieves is close to the ensemble pipeline's, the ensemble is at the
planning frontier; if it's substantially better, the ensemble is
leaving slope on the table due to dynamics error.

Stage 1 (1-step ranking): for every valid action (subsampled to
--stage1-sample), snapshot env, step, record true 1-step reward,
restore. Pick top-K.
Stage 2 (H-1 random continuation, n_rollouts=1): for each of K
candidates, restore initial state, commit candidate, then
random-continue for H-1 steps using actual env.step, accumulate
gamma-discounted reward. Commit the best candidate. Repeat for
max_steps outer steps.

Notes:
- Random continuation is used (not greedy) to keep wall time manageable.
  Fully matched comparison with the ensemble pipeline (which uses
  greedy continuation) requires implementing greedy continuation
  against the true env, which would multiply per-step cost ~10x.
- _count_baimu_fang is monkey-patched to a no-op during scoring trials
  so that the trial step's modulo-5 baimu trigger doesn't dominate
  cost. The outer commit step uses the original method, so the
  reported baimu metrics are correct.

Outputs (under --out-dir):
    true_env_summary.json    final aggregate
    true_env_per_step.json   per-step record
    true_env_run.log

Usage:
    python mpc_true_env.py \\
        --prepared <prepared_dir> \\
        --proj-crs EPSG:32648 \\
        --n-steps 100 --horizon 5 --top-k 50 --gamma 0.99 \\
        --stage1-sample 200 --seed 0 \\
        --out-dir true_env_run

Wall time: ~6 minutes for 100 steps on a 53k-parcel county on a
desktop CPU (about 4x faster than the ensemble pipeline; the speed
comes from sub-sampling stage 1, not from being more efficient
per-action).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def make_snapshotter(env):
    """Snapshot/restore the mutable state needed by env.step incremental updates."""
    arrs = ("land_use", "swapped", "farmland_nbr_count", "swaps_in_block")
    bavail = ("_block_farm_avail", "_block_forest_avail")
    scalars = ("total_weighted_slope", "total_farm_area", "total_farmland_adj",
               "n_farmland", "n_forest", "budget_used", "step_count",
               "baimu_count", "baimu_total_area",
               "prev_slope", "prev_cont", "prev_baimu_count", "prev_baimu_area")

    def snapshot():
        return {
            "arrs": {k: getattr(env, k).copy() for k in arrs},
            "bavail": {k: getattr(env, k).copy() for k in bavail},
            "scalars": {k: getattr(env, k) for k in scalars},
        }

    def restore(snap):
        for k in arrs:
            np.copyto(getattr(env, k), snap["arrs"][k])
        for k in bavail:
            np.copyto(getattr(env, k), snap["bavail"][k])
        for k in scalars:
            setattr(env, k, snap["scalars"][k])

    return snapshot, restore


@contextmanager
def baimu_frozen(env):
    """Skip baimu re-counting inside trial steps. Outer commit step gets the
    real baimu after the context manager exits and the next env.step runs.
    """
    cached = (env.baimu_count, env.baimu_total_area)
    orig = env._count_baimu_fang
    env._count_baimu_fang = lambda: cached
    try:
        yield
    finally:
        env._count_baimu_fang = orig


def mpc_true_env_select(env, snapshot, restore, valid_actions, horizon, top_k,
                        gamma, n_rollouts, rng, stage1_sample):
    """Pick action using TRUE env.step instead of an ensemble.
    Returns (chosen_action, info_dict).

    stage1_sample : if len(valid_actions) > stage1_sample, randomly subsample
        that many actions for stage-1 ranking. This bounds wall time and is
        analogous to the ensemble's batch_predict scoring of every action --
        for the true env that scoring is not free, so we approximate with a
        random subset and rely on top-K selection for the second-stage filter.
    """
    n_valid = len(valid_actions)
    if n_valid == 0:
        return 0, {"n_valid": 0}

    if n_valid > stage1_sample:
        sampled = rng.choice(valid_actions, size=stage1_sample, replace=False)
    else:
        sampled = valid_actions
    n_s = len(sampled)

    snap0 = snapshot()

    # Stage 1: score each sampled action by 1-step true reward
    rewards1 = np.zeros(n_s, dtype=np.float64)
    for i in range(n_s):
        a = int(sampled[i])
        _, r, _, _, _ = env.step(a)
        rewards1[i] = r
        restore(snap0)

    k = min(top_k, n_s)
    top_idx = np.argpartition(rewards1, -k)[-k:]
    candidates = sampled[top_idx]
    cand_cumrew = rewards1[top_idx].astype(np.float64).copy()

    # Stage 2: H-1 random continuation rollouts per candidate
    rollout_rewards = np.zeros(k, dtype=np.float64)
    for _ in range(n_rollouts):
        for c_idx in range(k):
            restore(snap0)
            _, r0, term0, trunc0, _ = env.step(int(candidates[c_idx]))
            # cand_cumrew already includes r0 from stage 1; do not double-count
            discount = gamma
            for _h in range(1, horizon):
                if term0 or trunc0:
                    break
                mask_in = env.action_masks()
                valid_in = np.where(mask_in)[0]
                if len(valid_in) == 0:
                    break
                a_rand = int(rng.choice(valid_in))
                _, r, term, trunc, _ = env.step(a_rand)
                rollout_rewards[c_idx] += discount * r
                discount *= gamma
                if term or trunc:
                    break
    cand_cumrew += rollout_rewards / max(n_rollouts, 1)

    best = int(np.argmax(cand_cumrew))
    chosen = int(candidates[best])

    restore(snap0)
    return chosen, {
        "n_valid": int(n_valid),
        "n_sampled": int(n_s),
        "n_candidates": int(k),
        "best_cumrew": float(cand_cumrew[best]),
        "stage1_max_reward": float(rewards1.max()),
        "stage1_mean_reward": float(rewards1.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--proj-crs", default="EPSG:32648")
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--n-rollouts", type=int, default=1)
    ap.add_argument("--stage1-sample", type=int, default=200,
                    help="Random subset size for stage-1 ranking. ensemble MPC "
                         "scores all valid actions for free; true env can't, so "
                         "we sample. 200 covers ~10%% of n_valid (~2000).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "true_env_run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("trueenv")

    log.info(f"prepared={args.prepared} steps={args.n_steps} H={args.horizon} "
             f"K={args.top_k} gamma={args.gamma} continuation=random "
             f"n_rollouts={args.n_rollouts} seed={args.seed}")

    from farmland_mpc.blocks_env import make_env

    log.info("Building env ...")
    t0 = time.time()
    env = make_env(prepared_dir=args.prepared, proj_crs=args.proj_crs)
    log.info(f"env built in {time.time()-t0:.1f}s; "
             f"n_blocks={env.n_blocks}, n_parcels={env.n_parcels}, "
             f"max_steps={env.max_steps}")

    n_steps = min(args.n_steps, env.max_steps)
    env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    snapshot, restore = make_snapshotter(env)

    rec = {
        "step": [], "action": [], "n_valid": [],
        "stage1_max_reward": [], "stage1_mean_reward": [],
        "best_cumrew": [],
        "true_reward": [],
        "slope_change_pct": [], "cont_change": [],
        "baimu_count_change": [], "baimu_area_change_ha": [],
        "step_time_s": [],
    }

    log.info(f"Rolling {n_steps} outer steps with TRUE-ENV MPC ...")
    for step in range(n_steps):
        t_step = time.time()
        mask = env.action_masks()
        valid = np.where(mask)[0]
        if len(valid) == 0:
            log.info(f"  step {step+1}: no valid actions; terminating")
            break

        with baimu_frozen(env):
            action, mpc_info = mpc_true_env_select(
                env, snapshot, restore, valid,
                args.horizon, args.top_k, args.gamma, args.n_rollouts, rng,
                args.stage1_sample,
            )

        # Commit (real env.step, baimu unfrozen, may update baimu)
        _, r_true, term, trunc, info = env.step(int(action))

        elapsed = time.time() - t_step
        rec["step"].append(step + 1)
        rec["action"].append(int(action))
        rec["n_valid"].append(int(mpc_info.get("n_valid", 0)))
        rec["stage1_max_reward"].append(float(mpc_info.get("stage1_max_reward", 0.0)))
        rec["stage1_mean_reward"].append(float(mpc_info.get("stage1_mean_reward", 0.0)))
        rec["best_cumrew"].append(float(mpc_info.get("best_cumrew", 0.0)))
        rec["true_reward"].append(float(r_true))
        rec["slope_change_pct"].append(float(info["slope_change_pct"]))
        rec["cont_change"].append(float(info["cont_change"]))
        rec["baimu_count_change"].append(int(info["baimu_count_change"]))
        rec["baimu_area_change_ha"].append(float(info["baimu_area_change_ha"]))
        rec["step_time_s"].append(elapsed)

        if (step + 1) % 5 == 0 or step == 0 or step + 1 == n_steps:
            log.info(f"  step {step+1:3d}/{n_steps} "
                     f"a={int(action):4d} "
                     f"r_true={r_true:+.3f} "
                     f"slope={info['slope_change_pct']:+.4f}% "
                     f"cont={info['cont_change']:+.4f} "
                     f"baimu_ha={info['baimu_area_change_ha']:+.1f} "
                     f"t={elapsed:.1f}s")
            # Incremental checkpoint
            try:
                with open(out_dir / "true_env_per_step.json", "w", encoding="utf-8") as f:
                    json.dump(rec, f, indent=2, ensure_ascii=False)
            except Exception as ex:
                log.warning(f"  checkpoint write failed: {ex}")

        if term or trunc:
            log.info(f"  episode ended at step {step+1}")
            break

    final_info = info
    log.info("")
    log.info("=" * 70)
    log.info("TRUE-ENV MPC RESULT")
    log.info("=" * 70)
    log.info(f"  steps run                   : {len(rec['step'])}")
    log.info(f"  final slope_change_pct      : {final_info['slope_change_pct']:+.4f}%")
    log.info(f"  final cont_change           : {final_info['cont_change']:+.4f}")
    log.info(f"  final baimu_count_change    : {final_info['baimu_count_change']:+d}")
    log.info(f"  final baimu_area_change_ha  : {final_info['baimu_area_change_ha']:+.2f}")
    log.info(f"  total wall time             : {sum(rec['step_time_s']):.1f}s")
    log.info(f"  mean per-step time          : {np.mean(rec['step_time_s']):.2f}s")
    log.info("")
    log.info(f"  Compare these numbers against your ensemble pipeline's "
             f"mpc_summary.json to bound the dynamics-error contribution.")

    summary = {
        "config": {
            "horizon": args.horizon, "top_k": args.top_k, "gamma": args.gamma,
            "n_rollouts": args.n_rollouts, "continuation": "random",
            "stage1_sample": args.stage1_sample,
            "n_steps_run": len(rec["step"]),
            "max_steps": int(env.max_steps),
            "n_blocks": int(env.n_blocks),
            "n_parcels": int(env.n_parcels),
            "seed": args.seed,
        },
        "final": {
            "slope_change_pct": float(final_info["slope_change_pct"]),
            "cont_change": float(final_info["cont_change"]),
            "baimu_count_change": int(final_info["baimu_count_change"]),
            "baimu_area_change_ha": float(final_info["baimu_area_change_ha"]),
        },
        "wall_time_s": float(sum(rec["step_time_s"])),
    }
    with open(out_dir / "true_env_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out_dir / "true_env_per_step.json", "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)

    log.info("")
    log.info(f"  outputs: {out_dir}")


if __name__ == "__main__":
    main()
