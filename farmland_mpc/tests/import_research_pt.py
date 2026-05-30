#!/usr/bin/env python3
"""Convert research-side .pt ensembles to ONNX, organized for our package.

Scans paper/checkpoints/<county>/<set>/ensemble_seed{0..4}_lam5.0_member{0..2}.pt
and emits package-style:

    <prepared_dir>/<out_subdir>_seed<S>/ensemble_member{0..2}.onnx (+ .onnx.data)

so eval_5seed_paper.py can pick them up via --ensemble-prefix.

Usage:
    python -m farmland_mpc.tests.import_research_pt \\
        --pt-glob "paper/checkpoints/neijiang/baseline/ensemble_seed*_lam5.0_member*.pt" \\
        --prepared-dir runs/dongxing/prepared \\
        --out-prefix research_ens \\
        --n-blocks 3711
"""
from __future__ import annotations

import argparse
import glob
import logging
import re
from pathlib import Path

import torch

from farmland_mpc.transition_model import TransitionModel
from farmland_mpc.train_ensemble import _export_onnx

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("import_research_pt")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-glob", required=True,
                    help="Glob matching ensemble_seed*_lam5.0_member*.pt files.")
    ap.add_argument("--prepared-dir", type=Path, required=True)
    ap.add_argument("--out-prefix", default="research_ens",
                    help="Output subdir prefix; per-seed dirs become <prefix><S>/")
    ap.add_argument("--n-blocks", type=int, required=True)
    ap.add_argument("--k-global", type=int, default=12)
    args = ap.parse_args()

    pat = re.compile(r"ensemble_seed(\d+)_lam[\d.]+_member(\d+)\.pt$")
    files = sorted(glob.glob(args.pt_glob))
    if not files:
        raise FileNotFoundError(f"No .pt files matched: {args.pt_glob}")
    log.info("Found %d .pt files", len(files))

    grouped = {}
    for f in files:
        m = pat.search(f)
        if not m:
            log.warning("Skipping unrecognised filename: %s", f); continue
        seed = int(m.group(1)); member = int(m.group(2))
        grouped.setdefault(seed, {})[member] = f

    for seed in sorted(grouped):
        members = grouped[seed]
        if set(members) != {0, 1, 2}:
            log.warning("seed %d incomplete: members=%s", seed, sorted(members)); continue
        out_dir = args.prepared_dir / f"{args.out_prefix}{seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info("=== seed %d -> %s ===", seed, out_dir)
        for i in range(3):
            pt_path = members[i]
            sd = torch.load(pt_path, map_location="cpu", weights_only=True)
            model = TransitionModel(n_blocks=args.n_blocks, k_global=args.k_global)
            model.load_state_dict(sd)
            model.eval()
            onnx_path = out_dir / f"ensemble_member{i}.onnx"
            _export_onnx(model, args.n_blocks, args.k_global, onnx_path,
                         say=lambda s: log.info("  %s", s))
            log.info("  [member %d]  %s -> %s (%.1f KB)",
                     i, Path(pt_path).name, onnx_path.name,
                     onnx_path.stat().st_size / 1024)
            # also drop a sidecar so we know the source
            (out_dir / "_provenance.txt").write_text(
                f"converted from research-side .pt files:\n"
                + "\n".join(f"  member{j}: {grouped[seed][j]}" for j in range(3))
                + f"\nn_blocks={args.n_blocks}, k_global={args.k_global}\n"
            )
    log.info("done; %d seeds processed", len(grouped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
