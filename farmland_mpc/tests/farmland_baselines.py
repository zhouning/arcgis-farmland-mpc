#!/usr/bin/env python3
"""SA / random / greedy baselines on farmland CountyLevelEnv (Bishan), with
explicit wall-clock budgets matching the restoration cells.

The farmland env has expensive step (60–120 ms baimu BFS + adjacency rebuild),
so the same SA loop that takes 1 s on restoration takes ~10 min on farmland.
This is the empirical confirmation of the simulator-cost crossover argument
of §6.5: even with the same algorithm, farmland's step cost makes SA's
search budget collapse.

Usage:
    python -m farmland_mpc.tests.farmland_baselines \\
        --prepared-dir /Users/zhouning/farmland_mpc_runs/bishan/prepared \\
        --wall-budget-s 180 \\
        --out-json runs/restoration/farmland_baselines_180s.json
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("farmland_baselines")


def _replay_episode(env, action_seq):
    """Roll out a full episode under the given action sequence; return total reward."""
    env.reset(seed=0)
    total = 0.0
    n_committed = 0
    for a in action_seq:
        if env.step_count >= env.max_steps:
            break
        m = env.action_masks() if hasattr(env, "action_masks") else None
        if m is not None and not m.any():
            break
        if m is not None and not m[a]:
            continue  # skip masked, try next
        _, r, term, trunc, _ = env.step(int(a))
        total += r
        n_committed += 1
        if term or trunc:
            break
    return total, n_committed


def run_random(env, wall_budget_s, seed=0):
    """Random-action baseline: keep generating random episodes, return best."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    best = -1e18
    n_eps = 0
    n_blocks = env.n_blocks
    while time.time() - t0 < wall_budget_s:
        order = rng.permutation(n_blocks).tolist()
        r, _ = _replay_episode(env, order)
        if r > best: best = r
        n_eps += 1
    return {"best": float(best), "n_eps": n_eps, "wall_s": time.time() - t0}


def run_sa(env, wall_budget_s, seed=0):
    """SA over a permutation of block ids."""
    rng = np.random.default_rng(seed)
    n_blocks = env.n_blocks
    cur = rng.permutation(n_blocks).tolist()
    cur_R, _ = _replay_episode(env, cur)
    best, best_R = list(cur), cur_R
    T0 = max(1.0, abs(cur_R) * 0.05)
    t0 = time.time()
    n_iter = 0
    while time.time() - t0 < wall_budget_s:
        T = T0 * (0.995 ** n_iter)
        i, j = rng.integers(0, min(80, n_blocks), size=2)
        if i != j:
            prop = list(cur); prop[i], prop[j] = prop[j], prop[i]
            prop_R, _ = _replay_episode(env, prop)
            d = prop_R - cur_R
            if d > 0 or rng.random() < np.exp(d / max(T, 1e-9)):
                cur, cur_R = prop, prop_R
                if prop_R > best_R: best, best_R = list(prop), prop_R
        n_iter += 1
    return {"best": float(best_R), "n_iter": n_iter, "wall_s": time.time() - t0}


def run_greedy_random_action(env, n_runs, seed=0):
    """Trivial baseline: random valid action each step (no replanning).
    Used to anchor what 'no method' looks like.
    """
    rng = np.random.default_rng(seed)
    rs = []
    for ep in range(n_runs):
        env.reset(seed=ep)
        r_ep = 0.0
        for _ in range(env.max_steps):
            m = env.action_masks()
            if not m.any(): break
            a = int(rng.choice(np.where(m)[0]))
            _, r, t, tr, _ = env.step(a)
            r_ep += r
            if t or tr: break
        rs.append(r_ep)
    return {"mean": float(np.mean(rs)), "std": float(np.std(rs)), "n_runs": n_runs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", type=Path, required=True)
    ap.add_argument("--wall-budget-s", type=float, default=180)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    from farmland_mpc.blocks_env import make_env
    log.info("Building farmland CountyLevelEnv from %s ...", args.prepared_dir)
    t0 = time.time()
    env = make_env(prepared_dir=str(args.prepared_dir))
    log.info("env built in %.1fs; n_blocks=%d, max_steps=%d", time.time() - t0,
             env.n_blocks, env.max_steps)

    # Estimate step cost from a 5-step probe
    log.info("Probing step cost ...")
    env.reset(seed=0)
    t0 = time.time()
    rng = np.random.default_rng(0)
    n_probe = 0
    for _ in range(5):
        m = env.action_masks()
        if not m.any(): break
        a = int(rng.choice(np.where(m)[0]))
        env.step(a); n_probe += 1
    step_cost_ms = (time.time() - t0) / max(n_probe, 1) * 1000
    log.info("  measured step cost: %.1f ms/step (over %d steps)", step_cost_ms, n_probe)

    # Random-action 5-episode baseline (analogue of restoration's random_baseline)
    log.info("Running random-action 5-episode baseline ...")
    rnd = run_greedy_random_action(env, n_runs=5, seed=0)
    log.info("  random: %.4f ± %.4f", rnd["mean"], rnd["std"])

    # SA at the requested wall budget
    log.info("Running SA at %.0fs wall budget ...", args.wall_budget_s)
    sa = run_sa(env, wall_budget_s=args.wall_budget_s, seed=0)
    log.info("  SA  : best=%.4f, n_iter=%d, wall=%.1fs", sa["best"], sa["n_iter"], sa["wall_s"])

    # Random restart at the same budget for comparison
    log.info("Running random-restart at %.0fs wall budget ...", args.wall_budget_s)
    rr = run_random(env, wall_budget_s=args.wall_budget_s, seed=42)
    log.info("  rand: best=%.4f, n_eps=%d, wall=%.1fs", rr["best"], rr["n_eps"], rr["wall_s"])

    out = {
        "prepared_dir": str(args.prepared_dir),
        "wall_budget_s": args.wall_budget_s,
        "step_cost_ms": step_cost_ms,
        "random_action_5ep": rnd,
        "sa": sa,
        "random_restart": rr,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    log.info("wrote %s", args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
