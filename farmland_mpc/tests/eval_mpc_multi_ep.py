#!/usr/bin/env python3
"""Multi-episode MPC eval for fair comparison against SA's iteration budget.

SA in or_baselines.py runs 300 iterations, each one a full 50-step episode
rollout — that's 15,000 env.step() calls. The single-episode MPC eval
makes only 50 env.step() calls plus 2,500 ensemble evaluations. To make
the comparison apples-to-apples on planning *budget*, this script reruns
MPC over many episodes from different starting RNG states and reports
the best (and the budget at which best was reached).

Usage:
    python -m farmland_mpc.tests.eval_mpc_multi_ep \\
        --prepared-dir runs/restoration/buchanan_va/prepared_delayed \\
        --ensemble-dir runs/restoration/buchanan_va/prepared_delayed/ensemble_seed0 \\
        --n-episodes 50 --horizon 5 --top-k 50 \\
        --out-json runs/restoration/buchanan_va/profiles/delayed/mpc_multi_ep_h5.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from farmland_mpc.restoration_env import make_restoration_env
from farmland_mpc.ensemble_runner import EnsembleOrtRunner
from farmland_mpc.mpc_plan import mpc_select_action

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("eval_mpc_multi_ep")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir",  type=Path, required=True)
    ap.add_argument("--ensemble-dir",  type=Path, required=True)
    ap.add_argument("--out-json",      type=Path, required=True)
    ap.add_argument("--n-episodes",    type=int, default=50)
    ap.add_argument("--horizon",       type=int, default=5)
    ap.add_argument("--top-k",         type=int, default=50)
    ap.add_argument("--continuation",  default="greedy")
    args = ap.parse_args()

    env = make_restoration_env(args.prepared_dir)
    ensemble = EnsembleOrtRunner(str(args.ensemble_dir))
    ensemble.assert_compatible(env.n_blocks)

    results = []
    t_total = time.time()
    best_so_far_curve = []
    best_R = -1e18

    for ep in range(args.n_episodes):
        t0 = time.time()
        env.reset(seed=ep * 17 + 1)  # different RNG path per episode
        rng = np.random.default_rng(ep * 17 + 1)
        cum_R = 0.0
        n_committed = 0
        for _ in range(env.max_steps):
            bf = env._get_block_features()
            gf = env._get_global_features()
            mask = env.action_masks()
            if not mask.any(): break
            action, _ = mpc_select_action(
                ensemble, bf, gf, mask,
                horizon=args.horizon, top_k=args.top_k, gamma=0.99,
                n_rollouts=1, continuation=args.continuation,
                scoring="reward", rng=rng,
            )
            _, r, term, trunc, _ = env.step(int(action))
            cum_R += r
            n_committed += 1
            if term or trunc: break
        elapsed = time.time() - t0
        if cum_R > best_R:
            best_R = cum_R
        best_so_far_curve.append(best_R)
        results.append({
            "episode":     ep,
            "total_reward": float(cum_R),
            "n_committed": int(n_committed),
            "budget_used": float(env.budget_used),
            "wall_s":      float(elapsed),
        })
        log.info("  ep %3d/%d: R=%+.2f  (best so far %+.2f)",
                 ep + 1, args.n_episodes, cum_R, best_R)

    rs = np.array([r["total_reward"] for r in results])
    out = {
        "n_episodes": len(results),
        "horizon":    args.horizon,
        "top_k":      args.top_k,
        "wall_total_s": float(time.time() - t_total),
        "per_episode": results,
        "best_so_far_curve": best_so_far_curve,
        "stats": {
            "best":     float(rs.max()),
            "mean":     float(rs.mean()),
            "std":      float(rs.std()),
            "median":   float(np.median(rs)),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    log.info("wrote %s", args.out_json)
    print(f"\n{args.n_episodes} eps  best={rs.max():+.2f}  mean={rs.mean():+.2f}±{rs.std():.2f}  total_wall={time.time()-t_total:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
