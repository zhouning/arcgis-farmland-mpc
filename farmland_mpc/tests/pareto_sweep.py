#!/usr/bin/env python3
"""Pareto-front sweep for the restoration cases (reviewer M10).

Re-runs the env+greedy plan under multiple reward-weight configurations to
expose the Pareto front of (risk_reduction, water_protection, connectivity,
cost) trade-offs. Each weight setting is a different policy preference, and
the cross-policy trade-off surface lets practitioners choose a point that
matches their stated priority rather than committing to a single scalar.

This is the analogue of the farmland paper's slope-vs-baimu trade-off
discussion, generalised to the restoration setting where four objective
terms compete.

Usage:
    python -m farmland_mpc.tests.pareto_sweep \\
        --prepared-dir runs/restoration/buchanan_va/prepared \\
        --units-attributes runs/restoration/buchanan_va/planning_units_2km_attributes.csv \\
        --case buchanan \\
        --out-json runs/restoration/buchanan_va/pareto_sweep.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from farmland_mpc.tests.or_baselines import _replay_with_action_seq, baseline_greedy

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("pareto_sweep")


# Each row is a different policy weighting. Sums need not be 1 (the env
# treats them as raw multipliers), but for interpretability we keep them so.
_WEIGHT_GRID = {
    "risk_dominant":   {"risk":  0.70, "water": 0.10, "conn": 0.10, "cost": -0.10},
    "water_dominant":  {"risk":  0.10, "water": 0.70, "conn": 0.10, "cost": -0.10},
    "conn_dominant":   {"risk":  0.10, "water": 0.10, "conn": 0.70, "cost": -0.10},
    "cost_dominant":   {"risk":  0.20, "water": 0.20, "conn": 0.10, "cost": -0.50},
    "balanced":        {"risk":  0.25, "water": 0.25, "conn": 0.25, "cost": -0.25},
    "default":         {"risk":  0.45, "water": 0.25, "conn": 0.20, "cost": -0.10},
    "no_cost":         {"risk":  0.45, "water": 0.25, "conn": 0.20, "cost":  0.00},
}


def _patch_scenario_config(prepared_dir: Path, weights_4: dict[str, float],
                           reward_term_keys: tuple[str, str, str]) -> Path:
    """Write a temporary prepared/ that matches the requested weight vector.
    Returns path to the temp dir.
    """
    import shutil
    import tempfile
    src_cfg = json.loads((prepared_dir / "scenario_config.json").read_text())
    cfg = dict(src_cfg)
    rt = dict(cfg["reward_terms"])
    risk_k, water_k, conn_k = reward_term_keys
    rt[risk_k]   = float(weights_4["risk"])
    rt[water_k]  = float(weights_4["water"])
    rt["connectivity"] = float(weights_4["conn"])
    rt["cost_penalty"] = float(weights_4["cost"])
    cfg["reward_terms"] = rt
    tmp = Path(tempfile.mkdtemp(prefix="pareto_"))
    for f in ["attributes.csv", "adjacency.csv"]:
        shutil.copy(prepared_dir / f, tmp / f)
    (tmp / "scenario_config.json").write_text(json.dumps(cfg, indent=2))
    return tmp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared-dir", type=Path, required=True)
    ap.add_argument("--units-attributes", type=Path, required=True)
    ap.add_argument("--case", choices=["buchanan", "synthetic"], required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    attrs = pd.read_csv(args.units_attributes)
    if "unit_id" not in attrs.columns:
        attrs = attrs.copy()
        attrs.insert(0, "unit_id", range(len(attrs)))

    if args.case == "buchanan":
        terms = ("risk_index", "water_priority", "connectivity")
    else:
        # synthetic uses risk_reduction / habitat_gain (proxy for water+habitat) / connectivity
        terms = ("risk_reduction", "water_gain", "connectivity")

    log.info("=== pareto_sweep: case=%s ===", args.case)
    log.info("  reward terms: %s", terms)

    out = {"case": args.case, "weights": {}, "results": {}}

    for label, w in _WEIGHT_GRID.items():
        log.info("--- config=%s  weights=%s ---", label, w)
        tmp_dir = _patch_scenario_config(args.prepared_dir, w, terms)
        try:
            r = baseline_greedy(tmp_dir, attrs)
            out["weights"][label] = w
            out["results"][label] = {
                "total_reward":   r["total_reward"],
                "n_committed":    r["n_committed"],
                "n_selected":     r["n_selected"],
                "budget_used":    r["budget_used"],
                "cum_components": r["cum_components"],
            }
            log.info("  -> reward=%.2f, components=%s", r["total_reward"],
                     {k: f"{v:.2f}" for k, v in r["cum_components"].items()})
        except Exception as e:
            log.exception("  config %s FAILED: %s", label, e)
            out["results"][label] = None

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    log.info("wrote %s", args.out_json)

    print()
    print(f"=== Pareto sweep ({args.case}) ===")
    print(f"{'config':<18s} {'risk':>10s} {'water':>10s} {'conn':>10s} {'cost':>10s}")
    for label, r in out["results"].items():
        if r is None: continue
        cc = r["cum_components"]
        risk_v  = sum(v for k, v in cc.items() if "risk"  in k or "reduction" in k)
        water_v = sum(v for k, v in cc.items() if "water" in k)
        conn_v  = sum(v for k, v in cc.items() if "conn"  in k)
        cost_v  = sum(v for k, v in cc.items() if "cost"  in k)
        print(f"{label:<18s} {risk_v:>+10.2f} {water_v:>+10.2f} {conn_v:>+10.4f} {cost_v:>+10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
