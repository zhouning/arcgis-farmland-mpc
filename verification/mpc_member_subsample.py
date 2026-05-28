# -*- coding: utf-8 -*-
"""
mpc_member_subsample.py — replace seed-jitter (which is a no-op under
deterministic top-K + greedy continuation) with ensemble-member subset
ablation as the cross-replicate variance estimator.

For a 3-member ensemble there are C(3,2)=3 distinct 2-of-3 subsets; this
script enumerates all three and runs the full MPC pipeline once per
subset. The resulting variance is interpretable (it measures the spread
among ensemble members' planning outputs) rather than nominal-zero.

Otherwise matches the standard pipeline exactly:
    H=5, K=50, gamma=0.99, greedy continuation, scoring=reward,
    all-action stage-1.

Outputs (under --out-dir):
    summary.json                        aggregate stats
    episode_<i>_members_<a>-<b>.json    per-replicate result
    summary_partial.json                rolling checkpoint
    run.log

Usage:
    python mpc_member_subsample.py \\
        --prepared <prepared_dir> \\
        --ensemble <prepared_dir>/tool3 \\
        --proj-crs EPSG:32648 \\
        --n-episodes 3 --n-keep 2 \\
        --horizon 5 --top-k 50 --gamma 0.99 \\
        --seed-offset 100 \\
        --out-dir mss_run

Wall time: roughly the same as a full ensemble episode multiplied by the
number of distinct subsets (3 for 2-of-3 ensemble); on 53k-parcel data,
about 60 minutes total on a desktop CPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np


class SubsetEnsembleHandle:
    """Replaces base_runner._sessions with a random subset for the duration
    of an episode. Restores on close. This keeps the original batch_predict
    code path (same speed as the full ensemble pipeline) and lets cross-episode
    variance come from which 2-of-3 members are active.
    """
    def __init__(self, base_runner, n_keep: int, rng: np.random.Generator):
        self.base = base_runner
        self.n_keep = n_keep
        self._all = list(base_runner._sessions)
        idx = rng.choice(len(self._all), size=n_keep, replace=False)
        self.idx = sorted(int(i) for i in idx)
        self._original = None

    def __enter__(self):
        self._original = self.base._sessions
        self.base._sessions = [self._all[i] for i in self.idx]
        return self.base

    def __exit__(self, *_):
        self.base._sessions = self._original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", required=True)
    ap.add_argument("--ensemble", required=True)
    ap.add_argument("--proj-crs", default="EPSG:32648")
    ap.add_argument("--n-episodes", type=int, default=5)
    ap.add_argument("--n-keep", type=int, default=2,
                    help="Members per batch_predict call (default 2 of 3)")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--seed-offset", type=int, default=100,
                    help="Use seeds offset+0..offset+n-1 to avoid collision "
                         "with the canonical seed range (0..n-1) used in your "
                         "main 5-seed sweep")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(out_dir / "run.log", encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("mss")

    log.info(f"prepared={args.prepared}")
    log.info(f"ensemble={args.ensemble}")
    log.info(f"n_episodes={args.n_episodes} n_keep={args.n_keep}/3 H={args.horizon} K={args.top_k}")
    log.info(f"seeds = {args.seed_offset}..{args.seed_offset + args.n_episodes - 1}")

    from farmland_mpc.blocks_env import make_env
    from farmland_mpc.ensemble_runner import EnsembleOrtRunner
    from farmland_mpc.mpc_plan import mpc_select_action

    log.info("Loading ensemble ...")
    base_runner = EnsembleOrtRunner(args.ensemble, n_threads=0)
    log.info(f"  {base_runner.n_members} ONNX members; n_blocks={base_runner.n_blocks}")

    log.info("Building env (one-time) ...")
    t0 = time.time()
    env = make_env(prepared_dir=args.prepared, proj_crs=args.proj_crs)
    log.info(f"  env built in {time.time()-t0:.1f}s; n_blocks={env.n_blocks}, "
             f"n_parcels={env.n_parcels}, max_steps={env.max_steps}")
    base_runner.assert_compatible(env.n_blocks)

    # When sub-sampling 2-of-3, only C(3,2)=3 distinct subsets exist; we
    # enumerate them deterministically rather than randomly so each episode
    # gets a different one. Cross-episode variance therefore measures
    # ensemble-member ablation rather than seed jitter (the latter is moot
    # under deterministic top-K + greedy continuation, which was the issue
    # with the canonical 5-seed sweep's std=0 in the first place).
    if base_runner.n_members == 3 and args.n_keep == 2:
        all_subsets = [[0, 1], [0, 2], [1, 2]]
    else:
        # Generic case: enumerate all combinations
        from itertools import combinations
        all_subsets = [list(c) for c in combinations(range(base_runner.n_members),
                                                      args.n_keep)]
    n_eps_actual = min(args.n_episodes, len(all_subsets))
    if n_eps_actual < args.n_episodes:
        log.info(f"  Note: only {len(all_subsets)} distinct {args.n_keep}-of-"
                 f"{base_runner.n_members} subsets exist; running {n_eps_actual} "
                 f"episodes instead of {args.n_episodes}.")
    log.info(f"  enumerating subsets: {all_subsets[:n_eps_actual]}")

    results = []
    for ep in range(n_eps_actual):
        seed = args.seed_offset + ep
        chosen_idx = all_subsets[ep]
        log.info(f"\n=== Episode {ep+1}/{n_eps_actual} (seed={seed}, members={chosen_idx}) ===")
        env.reset(seed=seed)
        rng = np.random.default_rng(seed)

        # Manually swap _sessions for this episode
        original_sessions = base_runner._sessions
        base_runner._sessions = [original_sessions[i] for i in chosen_idx]
        try:
            ens = base_runner

            ep_t0 = time.time()
            last_info = {}
            total_reward = 0.0
            for step in range(env.max_steps):
                t_step = time.time()
                bf = env._get_block_features()
                gf = env._get_global_features()
                mask = env.action_masks()
                action, _ = mpc_select_action(
                    ens, bf, gf, mask,
                    horizon=args.horizon, top_k=args.top_k, gamma=args.gamma,
                    n_rollouts=1, continuation="greedy",
                    scoring="reward", rng=rng,
                )
                _, r, term, trunc, info = env.step(int(action))
                total_reward += r
                last_info = info
                if (step + 1) % 10 == 0 or step == 0 or step + 1 == env.max_steps:
                    log.info(f"  step {step+1:3d}/{env.max_steps} a={int(action):4d} "
                             f"slope={info['slope_change_pct']:+.4f}% "
                             f"cont={info['cont_change']:+.4f} "
                             f"baimu_ha={info['baimu_area_change_ha']:+.1f} "
                             f"t={time.time()-t_step:.1f}s")
                if term or trunc:
                    break

            ep_t = time.time() - ep_t0
            ep_result = {
                "episode": ep,
                "seed": seed,
                "members_used": chosen_idx,
                "slope_change_pct": float(last_info["slope_change_pct"]),
                "cont_change": float(last_info["cont_change"]),
                "baimu_count_change": int(last_info["baimu_count_change"]),
                "baimu_area_change_ha": float(last_info["baimu_area_change_ha"]),
                "total_reward": float(total_reward),
                "wall_time_s": float(ep_t),
            }
            results.append(ep_result)
            log.info(f"  ep {ep}: members={chosen_idx} "
                     f"slope={ep_result['slope_change_pct']:+.4f}% "
                     f"cont={ep_result['cont_change']:+.4f} "
                     f"baimu_ha={ep_result['baimu_area_change_ha']:+.2f} "
                     f"time={ep_t:.1f}s")

            with open(out_dir / f"episode_{ep}_members_{'-'.join(map(str, chosen_idx))}.json",
                      "w", encoding="utf-8") as f:
                json.dump(ep_result, f, indent=2, ensure_ascii=False)
            with open(out_dir / "summary_partial.json", "w", encoding="utf-8") as f:
                json.dump({"results": results}, f, indent=2, ensure_ascii=False)
        finally:
            base_runner._sessions = original_sessions

    # Aggregate
    slopes = np.array([r["slope_change_pct"] for r in results])
    conts = np.array([r["cont_change"] for r in results])
    bcounts = np.array([r["baimu_count_change"] for r in results])
    bareas = np.array([r["baimu_area_change_ha"] for r in results])

    summary = {
        "config": {
            "n_episodes": args.n_episodes,
            "n_keep": args.n_keep,
            "n_members_total": int(base_runner.n_members),
            "subsample_strategy": "per-episode random",
            "horizon": args.horizon, "top_k": args.top_k, "gamma": args.gamma,
            "continuation": "greedy",
            "seed_offset": args.seed_offset,
        },
        "results": results,
        "aggregate": {
            "slope_pct_mean": float(slopes.mean()),
            "slope_pct_std": float(slopes.std(ddof=1)),
            "slope_pct_min": float(slopes.min()),
            "slope_pct_max": float(slopes.max()),
            "cont_mean": float(conts.mean()),
            "cont_std": float(conts.std(ddof=1)),
            "baimu_count_mean": float(bcounts.mean()),
            "baimu_count_std": float(bcounts.std(ddof=1)),
            "baimu_ha_mean": float(bareas.mean()),
            "baimu_ha_std": float(bareas.std(ddof=1)),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log.info("")
    log.info("=" * 70)
    log.info("MEMBER-SUBSAMPLE 5-SEED SUMMARY")
    log.info("=" * 70)
    log.info(f"  slope_pct: {summary['aggregate']['slope_pct_mean']:+.4f}% "
             f"+- {summary['aggregate']['slope_pct_std']:.4f} "
             f"[{summary['aggregate']['slope_pct_min']:+.4f}, "
             f"{summary['aggregate']['slope_pct_max']:+.4f}]")
    log.info(f"  cont     : {summary['aggregate']['cont_mean']:+.4f} "
             f"+- {summary['aggregate']['cont_std']:.4f}")
    log.info(f"  baimu_ha : {summary['aggregate']['baimu_ha_mean']:+.2f} "
             f"+- {summary['aggregate']['baimu_ha_std']:.2f}")
    log.info(f"  baimu_n  : {summary['aggregate']['baimu_count_mean']:+.1f} "
             f"+- {summary['aggregate']['baimu_count_std']:.1f}")
    log.info(f"  vs reference run (full ensemble): see your mpc_summary.json")
    log.info(f"  outputs: {out_dir}")


if __name__ == "__main__":
    main()
