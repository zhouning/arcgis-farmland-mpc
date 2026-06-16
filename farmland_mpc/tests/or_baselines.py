#!/usr/bin/env python3
"""Operations-research baselines for the restoration cases.

Implements four classical baselines that are standard for spatial
prioritisation problems and that provide non-learned comparators against
the learned-surrogate + MPC pipeline:

  greedy        : sort candidates by priority_score, take top-K under budget
  sa            : simulated annealing on the full episode reward
  nsga2         : multi-objective genetic via pymoo (each reward term as a
                  separate objective; Pareto front; weighted-sum tiebreaker)
  milp          : weighted-sum 0/1 knapsack-with-connectivity via PuLP CBC
                  (no commercial solver, uses bundled CBC)

Each baseline emits the same JSON schema as random_baseline.json so the
existing reporting/figure code can consume it uniformly.

Usage:
    python -m farmland_mpc.tests.or_baselines \\
        --prepared-dir runs/restoration/buchanan_va/prepared \\
        --units-attributes runs/restoration/buchanan_va/planning_units_2km_attributes.csv \\
        --out-dir runs/restoration/buchanan_va/or_baselines \\
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("or_baselines")


def _replay_with_action_seq(prepared_dir: Path, action_seq: list[int]) -> dict:
    """Push a (possibly out-of-order) action sequence through the restoration env
    to compute the actually-realised cumulative reward and reward-component
    breakdown. The env enforces masking, budget, and connectivity bonuses.
    """
    from farmland_mpc.restoration_env import make_restoration_env
    env = make_restoration_env(prepared_dir)
    env.reset(seed=0)
    cum_reward = 0.0
    n_committed = 0
    for a in action_seq:
        m = env.action_masks()
        if not m.any():
            break
        if not m[a]:
            # baseline picked an invalid action under the env's incremental
            # masking; skip and try next. Greedy by priority_score sometimes
            # picks units that exceed the cumulative budget.
            continue
        _, reward, term, trunc, _ = env.step(int(a))
        cum_reward += reward
        n_committed += 1
        if term or trunc:
            break
    return {
        "total_reward":   float(cum_reward),
        "n_selected":     int(env.selected.sum()),
        "n_committed":    int(n_committed),
        "budget_used":    float(env.budget_used),
        "cum_components": {k: float(env._cum_reward_components[j])
                           for j, k in enumerate(list(env.reward_terms.keys())[:8])},
    }


# -----------------------------------------------------------------------
# 1. Greedy by priority_score
# -----------------------------------------------------------------------
def baseline_greedy(prepared_dir: Path, attrs: pd.DataFrame, score_col: str = "priority_score"):
    """Sort all candidate units by ``score_col`` desc; submit that order to env."""
    if score_col not in attrs.columns:
        # fall back to risk_index for buchanan / risk_reduction for synthetic
        for c in ["risk_index", "risk_reduction", "benefit_proxy"]:
            if c in attrs.columns:
                score_col = c; break
    cand = attrs[attrs.get("candidate", 1) == 1].copy() if "candidate" in attrs.columns else attrs.copy()
    order = cand.sort_values(score_col, ascending=False)["unit_id"].astype(int).tolist()
    return _replay_with_action_seq(prepared_dir, order)


# -----------------------------------------------------------------------
# 2. Simulated Annealing
# -----------------------------------------------------------------------
def baseline_sa(prepared_dir: Path, attrs: pd.DataFrame, max_iters: int = 500,
                seed: int = 0):
    """Permutation SA: state is a candidate ordering; cost is -cum_reward.
    Neighbours = swap two positions in the prefix. Cools geometrically.
    """
    rng = np.random.default_rng(seed)
    cand = attrs[attrs.get("candidate", 1) == 1] if "candidate" in attrs.columns else attrs
    units = cand["unit_id"].astype(int).to_numpy()
    n = len(units)
    # Init with greedy-by-priority order
    score_col = "priority_score" if "priority_score" in attrs.columns else (
        "risk_index" if "risk_index" in attrs.columns else "risk_reduction"
    )
    init_order = list(cand.sort_values(score_col, ascending=False)["unit_id"].astype(int).to_numpy())
    cur = list(init_order)
    cur_eval = _replay_with_action_seq(prepared_dir, cur)
    best = list(cur); best_eval = cur_eval

    T0 = max(1.0, abs(cur_eval["total_reward"]) * 0.05)
    for it in range(max_iters):
        T = T0 * (0.995 ** it)
        # Swap two positions in the first 80 (only matters within episode horizon)
        i, j = rng.integers(0, min(80, n), size=2)
        if i == j: continue
        prop = list(cur)
        prop[i], prop[j] = prop[j], prop[i]
        prop_eval = _replay_with_action_seq(prepared_dir, prop)
        d = prop_eval["total_reward"] - cur_eval["total_reward"]
        if d > 0 or rng.random() < np.exp(d / max(T, 1e-9)):
            cur, cur_eval = prop, prop_eval
            if prop_eval["total_reward"] > best_eval["total_reward"]:
                best, best_eval = list(prop), prop_eval
    log.info("  SA: best total_reward=%.2f after %d iters", best_eval["total_reward"], max_iters)
    return best_eval


# -----------------------------------------------------------------------
# 3. NSGA-II (pymoo)
# -----------------------------------------------------------------------
def baseline_nsga2(prepared_dir: Path, attrs: pd.DataFrame,
                   max_steps: int, n_gen: int = 50, pop_size: int = 60,
                   seed: int = 0):
    """Multi-objective GA: each reward component is a separate objective.
    Decision variable: a permutation of unit_ids of length max_steps. The env
    is rolled out for each candidate, and the negative cum-component values
    are minimized (pymoo minimizes by default).
    """
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.optimize import minimize
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.operators.sampling.rnd import PermutationRandomSampling
        from pymoo.operators.crossover.ox import OrderCrossover
        from pymoo.operators.mutation.inversion import InversionMutation
        from pymoo.termination import get_termination
    except ImportError:
        log.warning("pymoo not installed; skipping NSGA-II")
        return None

    cand = attrs[attrs.get("candidate", 1) == 1] if "candidate" in attrs.columns else attrs
    units = cand["unit_id"].astype(int).to_numpy()
    n = len(units)

    class _Prob(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=n, n_obj=4, xl=0, xu=n - 1, type_var=int)

        def _evaluate(self, x, out, *args, **kwargs):
            seq = [int(units[i]) for i in x[:max_steps]]
            r = _replay_with_action_seq(prepared_dir, seq)
            comps = r["cum_components"]
            keys = list(comps.keys())
            # Minimise negative reward terms; cost penalty stays as-is
            f = []
            for k in keys[:4]:
                v = comps[k]
                f.append(-v if "cost" not in k else +v)
            # pad to consistent length
            while len(f) < 4:
                f.append(0.0)
            out["F"] = np.array(f[:4])
            out["_total"] = r["total_reward"]
            out["_seq"] = seq

    log.info("  NSGA-II: n_var=%d, pop=%d, gen=%d", n, pop_size, n_gen)
    prob = _Prob()
    alg = NSGA2(pop_size=pop_size,
                sampling=PermutationRandomSampling(),
                crossover=OrderCrossover(),
                mutation=InversionMutation(),
                eliminate_duplicates=True)
    res = minimize(prob, alg, get_termination("n_gen", n_gen),
                   seed=seed, verbose=False)
    # Pick the Pareto-front member with the highest total weighted reward
    best_total = -np.inf; best_seq = None
    for x in res.X:
        seq = [int(units[i]) for i in x[:max_steps]]
        r = _replay_with_action_seq(prepared_dir, seq)
        if r["total_reward"] > best_total:
            best_total = r["total_reward"]; best_seq = seq
    out = _replay_with_action_seq(prepared_dir, best_seq)
    log.info("  NSGA-II: best total_reward=%.2f over Pareto front of %d", out["total_reward"], len(res.X))
    return out


# -----------------------------------------------------------------------
# 4. MILP via PuLP CBC
# -----------------------------------------------------------------------
def baseline_milp(prepared_dir: Path, attrs: pd.DataFrame,
                  max_steps: int, budget: float, reward_weights: dict,
                  cost_col: str = "restoration_cost_proxy"):
    """Weighted-sum 0/1 knapsack-style MILP:
        max sum_i x_i * unit_score(i)
        s.t. sum_i x_i * cost(i) <= budget
             sum_i x_i <= max_steps

    Connectivity bonus is non-linear in the env (depends on which neighbours are
    selected), so we approximate by computing each unit's max-possible
    connectivity (1.0 * w_conn if all neighbours selected) and adding it to the
    static score.
    """
    try:
        import pulp
    except ImportError:
        log.warning("PuLP not installed; skipping MILP")
        return None

    cand = attrs[attrs.get("candidate", 1) == 1] if "candidate" in attrs.columns else attrs
    units = cand["unit_id"].astype(int).to_numpy()
    n = len(units)

    # Compute static unit score
    unit_score = np.zeros(n, dtype=float)
    for term, w in reward_weights.items():
        if term == "connectivity":
            unit_score += w  # max 1.0 * w (upper bound)
        elif term == "cost_penalty":
            continue
        else:
            v = cand[term].astype(float).to_numpy() if term in cand.columns else np.zeros(n)
            unit_score += w * v
    cost = (cand[cost_col].astype(float).to_numpy() if cost_col in cand.columns
            else np.zeros(n))
    cost_w = float(reward_weights.get("cost_penalty", -0.1))
    # Cost penalty in env is per-unit cost / budget * |w_cost|; subtract
    unit_score += cost_w * cost / max(budget, 1.0)

    prob = pulp.LpProblem("restoration_milp", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in range(n)}
    prob += pulp.lpSum(unit_score[i] * x[i] for i in range(n))
    prob += pulp.lpSum(cost[i] * x[i] for i in range(n)) <= budget
    prob += pulp.lpSum(x[i] for i in range(n)) <= max_steps

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
    prob.solve(solver)
    selected_idx = [i for i in range(n) if pulp.value(x[i]) > 0.5]
    seq = [int(units[i]) for i in selected_idx]
    # Order seq by static score desc to pick high-score units first under env masking
    seq_sorted = sorted(seq, key=lambda u: -unit_score[np.where(units == u)[0][0]])
    out = _replay_with_action_seq(prepared_dir, seq_sorted)
    log.info("  MILP: selected %d units, env-realised total_reward=%.2f",
             len(seq), out["total_reward"])
    return out


# -----------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", type=Path, required=True)
    ap.add_argument("--units-attributes", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--case", choices=["buchanan", "synthetic"], required=True)
    ap.add_argument("--methods", nargs="+",
                    default=["greedy", "sa", "nsga2", "milp"],
                    choices=["greedy", "sa", "nsga2", "milp"])
    args = ap.parse_args()

    cfg = json.loads((args.prepared_dir / "scenario_config.json").read_text())
    attrs = pd.read_csv(args.units_attributes)
    if "unit_id" not in attrs.columns:
        attrs = attrs.copy()
        attrs.insert(0, "unit_id", range(len(attrs)))

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("=== OR baselines: case=%s ===", args.case)
    log.info("  prepared_dir=%s", args.prepared_dir)
    log.info("  out_dir=%s", out_dir)

    results = {}
    timings = {}

    if "greedy" in args.methods:
        log.info("--- Method: GREEDY (sort by priority_score) ---")
        t = time.time()
        try:
            results["greedy"] = baseline_greedy(args.prepared_dir, attrs)
            timings["greedy"] = time.time() - t
            log.info("  greedy: total_reward=%.2f, n_committed=%d, time=%.1fs",
                     results["greedy"]["total_reward"],
                     results["greedy"]["n_committed"], timings["greedy"])
        except Exception as e:
            log.exception("  greedy FAILED: %s", e); results["greedy"] = None

    if "sa" in args.methods:
        log.info("--- Method: SIMULATED ANNEALING ---")
        t = time.time()
        try:
            results["sa"] = baseline_sa(args.prepared_dir, attrs,
                                        max_iters=300, seed=0)
            timings["sa"] = time.time() - t
            log.info("  sa: total_reward=%.2f, time=%.1fs",
                     results["sa"]["total_reward"], timings["sa"])
        except Exception as e:
            log.exception("  sa FAILED: %s", e); results["sa"] = None

    if "nsga2" in args.methods:
        log.info("--- Method: NSGA-II ---")
        t = time.time()
        try:
            results["nsga2"] = baseline_nsga2(args.prepared_dir, attrs,
                                              max_steps=cfg["max_steps"],
                                              n_gen=30, pop_size=40, seed=0)
            timings["nsga2"] = time.time() - t
            if results["nsga2"]:
                log.info("  nsga2: total_reward=%.2f, time=%.1fs",
                         results["nsga2"]["total_reward"], timings["nsga2"])
        except Exception as e:
            log.exception("  nsga2 FAILED: %s", e); results["nsga2"] = None

    if "milp" in args.methods:
        log.info("--- Method: MILP (PuLP CBC) ---")
        t = time.time()
        try:
            results["milp"] = baseline_milp(args.prepared_dir, attrs,
                                            max_steps=cfg["max_steps"],
                                            budget=cfg["budget"],
                                            reward_weights=cfg["reward_terms"])
            timings["milp"] = time.time() - t
            if results["milp"]:
                log.info("  milp: total_reward=%.2f, time=%.1fs",
                         results["milp"]["total_reward"], timings["milp"])
        except Exception as e:
            log.exception("  milp FAILED: %s", e); results["milp"] = None

    out = {
        "case": args.case,
        "results": results,
        "timings_s": timings,
        "scenario_config": cfg,
    }
    out_json = out_dir / "or_baselines.json"
    out_json.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_json)

    print()
    print(f"{'Method':<10s} {'total_reward':>14s} {'n_committed':>12s} {'time (s)':>10s}")
    for k, r in results.items():
        if r is None: continue
        print(f"{k:<10s} {r['total_reward']:>+14.2f} {r['n_committed']:>12d} {timings[k]:>10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
