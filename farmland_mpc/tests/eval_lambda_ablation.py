#!/usr/bin/env python3
"""Reproduce paper §3 Table 1 ranking-failure diagnostic.

Loads research-side λ-ablation .pt files (paper/checkpoints/bishan/lambda_ablation/)
and evaluates pairwise ranking accuracy on a pairwise dataset. Matches the
research-side `discriminative_results.json` schema so we can diff field-by-field.

Two evaluation modes:

  --pairwise-source <path/to/pairwise.npz>
      Use a pre-existing pairwise dataset (e.g. our bishan prepared/tool2/pairwise.npz).
      Drops pairs where either action >= n_blocks of the .pt (research = 2600).

  --research-pairwise-source <s,a,r tuple file>
      Bit-exact research-side pairwise; not currently shipped with the repo
      (raw 三调 derivative). If the windows side later pushes a researche-side
      pairwise.npz, point this here for a paper-Table-1 bit-for-bit reproduction.

Usage:
    python -m farmland_mpc.tests.eval_lambda_ablation \\
        --ablation-dir paper/checkpoints/bishan/lambda_ablation \\
        --pairwise /Users/zhouning/farmland_mpc_runs/bishan/prepared/tool2/pairwise.npz \\
        --n-blocks 2600 \\
        --out-json paper/repro_artifacts/macos_2026-05-29/lambda_ablation_eval.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from farmland_mpc.transition_model import TransitionModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("eval_lambda_ablation")


def _ranking_acc(model, bf, gf, actions, rewards, *, max_pairs_per_state=None):
    """Compute pairwise ranking accuracy of model.reward on (state, action_i, action_j).

    bf: (S, n_blocks, K_BLOCK)
    gf: (S, K_GLOBAL)
    actions: (S, A)  per-state action ids
    rewards: (S, A)  per-state ground-truth rewards
    """
    model.eval()
    n_states, n_actions = actions.shape
    K_BLOCK = bf.shape[2]
    correct = 0; total = 0
    pred_stds = []

    with torch.no_grad():
        for s in range(n_states):
            bf_s = torch.from_numpy(bf[s]).float().unsqueeze(0)  # (1, B, 17)
            gf_s = torch.from_numpy(gf[s]).float().unsqueeze(0)  # (1, K_G)
            acts = torch.from_numpy(actions[s].astype(np.int64))  # (A,)
            # Repeat state across all actions
            bf_rep = bf_s.expand(n_actions, -1, -1)
            gf_rep = gf_s.expand(n_actions, -1)
            _, _, pred_r = model(bf_rep, gf_rep, acts)
            pred = pred_r.squeeze(-1).numpy()
            true = rewards[s]
            pred_stds.append(float(pred.std()))

            # Sample pair indices
            if max_pairs_per_state is None:
                # All ordered pairs: (i,j) with i<j
                ii, jj = np.triu_indices(n_actions, k=1)
            else:
                rng = np.random.default_rng(s)
                ii = rng.integers(0, n_actions, max_pairs_per_state)
                jj = rng.integers(0, n_actions, max_pairs_per_state)
                mask = ii != jj
                ii, jj = ii[mask], jj[mask]

            true_diff = true[ii] - true[jj]
            pred_diff = pred[ii] - pred[jj]
            valid = np.abs(true_diff) > 1e-9  # exclude ties
            correct += int(((np.sign(true_diff) == np.sign(pred_diff)) & valid).sum())
            total += int(valid.sum())

    return {
        "ranking_acc": correct / max(total, 1),
        "n_pairs": total,
        "pred_reward_std_mean": float(np.mean(pred_stds)),
        "pred_reward_std_median": float(np.median(pred_stds)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-dir", type=Path, required=True,
                    help="Directory containing ensemble_lam{0.0,1.0,5.0}_member{0..2}.pt")
    ap.add_argument("--pairwise", type=Path, required=True,
                    help="pairwise.npz with keys block_features, global_features, actions, rewards")
    ap.add_argument("--n-blocks", type=int, required=True,
                    help="n_blocks of the .pt model (research bishan = 2600)")
    ap.add_argument("--k-global", type=int, default=12)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    log.info("Loading pairwise from %s", args.pairwise)
    npz = np.load(args.pairwise)
    log.info("  npz keys: %s", list(npz.files))
    # Support both schemas: research-side (block_features/global_features) and
    # package-side (states_bf/states_gf).
    bf_key = "states_bf"   if "states_bf"   in npz.files else "block_features"
    gf_key = "states_gf"   if "states_gf"   in npz.files else "global_features"
    bf = npz[bf_key]; gf = npz[gf_key]
    actions = npz["actions"]; rewards = npz["rewards"]
    log.info("  %s %s, %s %s, actions %s, rewards %s",
             bf_key, bf.shape, gf_key, gf.shape, actions.shape, rewards.shape)

    if bf.shape[1] != args.n_blocks:
        log.warning("pairwise n_blocks=%d != model n_blocks=%d; truncating block_features",
                    bf.shape[1], args.n_blocks)
        bf = bf[:, :args.n_blocks]
        # also drop pairs where action >= n_blocks
        valid_action_mask = actions < args.n_blocks
        # for now require ALL actions per state to be valid (else discard state)
        keep_states = valid_action_mask.all(axis=1)
        bf = bf[keep_states]; gf = gf[keep_states]
        actions = actions[keep_states]; rewards = rewards[keep_states]
        log.warning("  kept %d/%d states after action-id filter", len(bf), len(npz["actions"]))

    true_reward_std = float(np.std(rewards))
    log.info("  true_reward_std (over all states & actions) = %.4f", true_reward_std)

    results = {}
    for lam_str in ["0.0", "1.0", "5.0"]:
        log.info("=== λ_rank = %s ===", lam_str)
        per_member = []
        for i in range(3):
            pt = args.ablation_dir / f"ensemble_lam{lam_str}_member{i}.pt"
            log.info("  loading %s", pt.name)
            sd = torch.load(pt, map_location="cpu", weights_only=True)
            model = TransitionModel(n_blocks=args.n_blocks, k_global=args.k_global)
            model.load_state_dict(sd)
            stats = _ranking_acc(model, bf, gf, actions, rewards)
            log.info("    rank_acc=%.4f  pred_std_mean=%.4f  pred_std_median=%.4f  n_pairs=%d",
                     stats["ranking_acc"], stats["pred_reward_std_mean"],
                     stats["pred_reward_std_median"], stats["n_pairs"])
            per_member.append({
                "pred_reward_std_mean":   stats["pred_reward_std_mean"],
                "pred_reward_std_median": stats["pred_reward_std_median"],
                "true_reward_std_mean":   true_reward_std,
                "ranking_acc":            stats["ranking_acc"],
                "n_pairs":                stats["n_pairs"],
            })
        accs = [m["ranking_acc"] for m in per_member]
        stds = [m["pred_reward_std_mean"] for m in per_member]
        results[lam_str] = {
            "lambda_rank":          float(lam_str),
            "mean_pred_reward_std": float(np.mean(stds)),
            "true_reward_std":      true_reward_std,
            "mean_ranking_acc":     float(np.mean(accs)),
            "per_member":           per_member,
        }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", args.out_json)
    print()
    print("=" * 60)
    print(f"{'λ_rank':>6}  {'mean rank_acc':>15}  {'pred_std':>10}  {'paper §3 Table 1':>25}")
    paper_ref = {"0.0": 0.516, "1.0": 0.732, "5.0": 0.855}
    for lam in ["0.0", "1.0", "5.0"]:
        r = results[lam]
        print(f"{lam:>6}  {r['mean_ranking_acc']:>15.4f}  {r['mean_pred_reward_std']:>10.4f}"
              f"  {paper_ref[lam]:>25.4f}")
    print(f"  true_reward_std = {true_reward_std:.4f} (paper: 0.811)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
