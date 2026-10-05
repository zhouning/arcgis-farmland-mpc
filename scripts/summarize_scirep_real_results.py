#!/usr/bin/env python3
"""Build one auditable table from the two real-county evidence tracks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows_for_no_net_loss(root: Path, region: str):
    out = []
    base = root / region
    multi = base / "multiensemble_no_net_loss.json"
    if multi.exists():
        payload = load(multi)
        for i, row in enumerate(payload.get("results", [])):
            out.append({"region": region, "track": "no_net_loss", "seed": i, **row})
    for p in sorted(base.glob("seed*/mpc_summary.json")):
        payload = load(p)
        if payload.get("results"):
            row = payload["results"][0]
            out.append({"region": region, "track": "no_net_loss", "seed": p.parent.name, **row})
    return out


def rows_for_sensitivity(root: Path, region: str):
    out = []
    p = root / region / "reward_weight_sensitivity.json"
    if not p.exists():
        return out
    payload = load(p)
    for row in payload.get("results", payload.get("profiles", [])):
        out.append({"region": region, "track": "reward_sensitivity", **row})
    return out


def rows_for_existing_baselines(prepared_root: Path, region: str):
    prepared_name = "prepared_bishan" if region == "bishan" else "prepared_neijiang"
    path = prepared_root / prepared_name / "results_real" / "blocks" / "county" / "baselines_county.json"
    if not path.exists():
        return []
    payload = load(path)
    out = []
    for method, row in payload.items():
        if not isinstance(row, dict):
            continue
        if method == "random_block":
            for seed_row in row.get("per_seed", []):
                out.append({"region": region, "track": "rule_baseline", "method": method, **seed_row})
        else:
            out.append({"region": region, "track": "rule_baseline", "method": method, **row})
    return out


def rows_for_matched_baselines(experiment_root: Path, region: str):
    path = experiment_root / f"{region}_rule_baselines_matched.json"
    if not path.exists():
        return []
    payload = load(path)
    return [{"region": region, "track": "matched_rule_baseline", **row}
            for row in payload.get("rows", [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    extra = args.repo / "runs" / "scirep_extra"
    sens = args.repo / "runs" / "scirep_reward_sensitivity"
    rows = []
    for region in ("bishan", "neijiang"):
        rows += rows_for_no_net_loss(extra / "no_net_loss", region)
        rows += rows_for_sensitivity(sens, region)
        rows += rows_for_existing_baselines(args.repo / "runs" / "scirep_extra", region)
        rows += rows_for_matched_baselines(
            args.repo.parents[1] / "ScientificReports_submission_paper9_corrected" / "08_revision_experiments",
            region,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"regions": ["bishan", "neijiang"], "rows": rows}, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
