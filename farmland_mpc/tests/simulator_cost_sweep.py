#!/usr/bin/env python3
"""Simulator-cost sensitivity: how does the MPC vs OR-baseline gap shift as
env.step becomes more expensive?

Farmland's CountyLevelEnv has env.step ≈ 60–120 ms (baimu BFS + adjacency
recompute). Restoration's RestorationEnv has env.step ≈ 5 µs (dict lookup).
That's a 4-order-of-magnitude difference, and it is the most plausible
explanation for why classical OR baselines (which rely on cheap repeated env
calls) compete with our learned-surrogate MPC on restoration but not on
farmland.

To make this falsifiable, we wrap env.step with an artificial sleep and re-run
SA and MPC at matched WALL-CLOCK budgets across a sweep of step delays. The
crossover point — the env.step cost at which MPC's wall-clock beats SA's —
is the empirical answer to "when does the learned surrogate pay for its
training cost?"

Usage:
    python -m farmland_mpc.tests.simulator_cost_sweep \\
        --prepared-dir runs/restoration/buchanan_va/prepared_delayed \\
        --ensemble-dir runs/restoration/buchanan_va/prepared_delayed/ensemble_seed0 \\
        --units-attributes runs/restoration/buchanan_va/planning_units_2km_attributes.csv \\
        --out-json runs/restoration/buchanan_va/profiles/delayed/sim_cost_sweep.json \\
        --case buchanan
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from farmland_mpc.restoration_env import RestorationEnv, make_restoration_env
from farmland_mpc.ensemble_runner import EnsembleOrtRunner
from farmland_mpc.mpc_plan import mpc_select_action
from farmland_mpc.tests.or_baselines import _replay_with_action_seq

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("simulator_cost_sweep")


# Monkey-patch hook: add a fixed sleep to every env.step call.
# We modify _compute_step_reward (the dispatcher) since that's the heavy-cost
# computation path; in farmland it's the baimu fang BFS / adjacency rebuild.
_ORIGINAL_STEP_REWARD = RestorationEnv._compute_step_reward


def _make_delayed_step_reward(delay_seconds: float):
    def _delayed(self, a, cost):
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        return _ORIGINAL_STEP_REWARD(self, a, cost)
    return _delayed


def _set_step_delay(delay_seconds: float):
    RestorationEnv._compute_step_reward = _make_delayed_step_reward(delay_seconds)


def _reset_step_delay():
    RestorationEnv._compute_step_reward = _ORIGINAL_STEP_REWARD


def _run_mpc_to_wall_budget(prepared_dir, ensemble_dir, wall_budget_s,
                             horizon=5, top_k=50, continuation="random"):
    """Run MPC repeatedly with different RNG seeds until wall budget is exhausted.
    Return best total_reward seen and the actual wall time spent.
    """
    env = make_restoration_env(prepared_dir)
    ensemble = EnsembleOrtRunner(str(ensemble_dir))
    ensemble.assert_compatible(env.n_blocks)

    t0 = time.time()
    best_R = -1e18
    n_eps = 0
    while time.time() - t0 < wall_budget_s:
        env.reset(seed=n_eps * 17 + 1)
        rng = np.random.default_rng(n_eps * 17 + 1)
        cum_R = 0.0
        for _ in range(env.max_steps):
            bf = env._get_block_features()
            gf = env._get_global_features()
            mask = env.action_masks()
            if not mask.any(): break
            action, _ = mpc_select_action(
                ensemble, bf, gf, mask,
                horizon=horizon, top_k=top_k, gamma=0.99,
                n_rollouts=1, continuation=continuation,
                scoring="reward", rng=rng,
            )
            _, r, term, trunc, _ = env.step(int(action))
            cum_R += r
            if term or trunc: break
        if cum_R > best_R: best_R = cum_R
        n_eps += 1
    return {"best": float(best_R), "n_eps": int(n_eps), "wall_s": float(time.time() - t0)}


def _run_sa_to_wall_budget(prepared_dir, attrs, wall_budget_s, seed=0,
                            score_col="priority_score"):
    """Run SA in batches until wall budget is hit. Return best total_reward."""
    rng = np.random.default_rng(seed)
    cand = attrs[attrs.get("candidate", 1) == 1] if "candidate" in attrs.columns else attrs
    units = cand["unit_id"].astype(int).to_numpy()
    n = len(units)
    score_actual = score_col if score_col in attrs.columns else (
        "risk_index" if "risk_index" in attrs.columns else "risk_reduction"
    )
    init_order = list(cand.sort_values(score_actual, ascending=False)["unit_id"].astype(int))
    cur = list(init_order)
    cur_eval = _replay_with_action_seq(Path(prepared_dir), cur)
    best, best_eval = list(cur), cur_eval
    T0 = max(1.0, abs(cur_eval["total_reward"]) * 0.05)

    t0 = time.time()
    n_iter = 0
    while time.time() - t0 < wall_budget_s:
        T = T0 * (0.995 ** n_iter)
        i, j = rng.integers(0, min(80, n), size=2)
        if i != j:
            prop = list(cur)
            prop[i], prop[j] = prop[j], prop[i]
            prop_eval = _replay_with_action_seq(Path(prepared_dir), prop)
            d = prop_eval["total_reward"] - cur_eval["total_reward"]
            if d > 0 or rng.random() < np.exp(d / max(T, 1e-9)):
                cur, cur_eval = prop, prop_eval
                if prop_eval["total_reward"] > best_eval["total_reward"]:
                    best, best_eval = list(prop), prop_eval
        n_iter += 1
    return {"best": float(best_eval["total_reward"]), "n_iter": int(n_iter), "wall_s": float(time.time() - t0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir",     type=Path, required=True)
    ap.add_argument("--ensemble-dir",     type=Path, required=True)
    ap.add_argument("--units-attributes", type=Path, required=True)
    ap.add_argument("--out-json",         type=Path, required=True)
    ap.add_argument("--case",             choices=["buchanan", "synthetic"])
    args = ap.parse_args()

    attrs = pd.read_csv(args.units_attributes)
    if "unit_id" not in attrs.columns:
        attrs = attrs.copy(); attrs.insert(0, "unit_id", range(len(attrs)))

    # Sweep over step delays (seconds per env.step).
    # 0 = restoration native (~5 µs); 0.001 = 1ms; 0.01 = 10ms; 0.1 = 100ms (farmland);
    # 0.05 = midway mark.
    DELAYS = [0.0, 0.001, 0.01, 0.05, 0.1]
    # Wall budget per (delay, method). 30s is enough to expose the crossover
    # for restoration-scale problems; 60s gives more confidence at high delay.
    WALL_BUDGET_S = 30.0

    results = {"case": args.case, "wall_budget_s": WALL_BUDGET_S, "sweep": {}}

    for d in DELAYS:
        log.info("=== step_delay = %.3f s ===", d)
        _set_step_delay(d)
        try:
            sa = _run_sa_to_wall_budget(args.prepared_dir, attrs, WALL_BUDGET_S, seed=0)
            log.info("  SA  : best=%.2f  n_iter=%d  wall=%.1fs",
                     sa["best"], sa["n_iter"], sa["wall_s"])
            mpc = _run_mpc_to_wall_budget(args.prepared_dir, args.ensemble_dir,
                                           WALL_BUDGET_S, horizon=5, top_k=50,
                                           continuation="random")
            log.info("  MPC : best=%.2f  n_eps=%d   wall=%.1fs",
                     mpc["best"], mpc["n_eps"], mpc["wall_s"])
            results["sweep"][f"{d:.4f}"] = {"sa": sa, "mpc": mpc, "delay_s": d}
        finally:
            _reset_step_delay()

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2))
    log.info("wrote %s", args.out_json)

    # Summary table
    print()
    print(f"{'delay (s)':>10s}  {'SA best':>10s}  {'SA iters':>10s}  {'MPC best':>10s}  {'MPC eps':>10s}  {'who wins':>10s}")
    for k, v in results["sweep"].items():
        winner = "MPC" if v["mpc"]["best"] > v["sa"]["best"] else "SA"
        margin = abs(v["mpc"]["best"] - v["sa"]["best"])
        print(f"{v['delay_s']:>10.4f}  {v['sa']['best']:>+10.2f}  {v['sa']['n_iter']:>10d}  "
              f"{v['mpc']['best']:>+10.2f}  {v['mpc']['n_eps']:>10d}  {winner:>10s} ({margin:+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
