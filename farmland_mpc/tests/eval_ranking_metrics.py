#!/usr/bin/env python3
"""Planner-relevant ranking metrics: NDCG@K, top-K regret, Spearman by state.

Reviewer M7: pairwise ranking accuracy is a clean diagnostic but not what MPC's
top-K selection actually consumes. This adds the standard learning-to-rank
metrics (NDCG, regret, Kendall/Spearman) computed on the same pairwise.npz
the contrastive trainer evaluates against, for each trained ensemble.

Key metrics computed per state and aggregated:

  NDCG@K            — discounted cumulative gain at top-K, normalised by
                      ideal DCG. K = 1, 5, 10, 50.
  Precision@K       — fraction of model's top-K that are in the true top-K.
  Top-K regret      — (sum of true rewards for true top-K) minus
                      (sum of true rewards for model's top-K), normalised
                      by the true top-K sum.
  Spearman by state — rank correlation between model and true rewards across
                      the action menu, averaged over states.
  Pairwise accuracy — same as the existing eval_lambda_ablation, kept for
                      reference.

Usage:
    python -m farmland_mpc.tests.eval_ranking_metrics \\
        --ensemble-dir runs/restoration/buchanan_va/prepared/ensemble_seed0 \\
        --pairwise runs/restoration/buchanan_va/prepared/tool2/pairwise.npz \\
        --n-blocks 562 \\
        --label "buchanan-contrastive-seed0" \\
        --out-json runs/restoration/buchanan_va/ranking_metrics_seed0.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau

from farmland_mpc.transition_model import TransitionModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("eval_ranking_metrics")


def _ndcg_at_k(true_r: np.ndarray, pred_r: np.ndarray, k: int) -> float:
    """Standard normalised discounted cumulative gain at rank k.

    true_r: ground-truth reward per action (length n_actions)
    pred_r: predicted reward per action (length n_actions)
    """
    n = len(true_r)
    k = min(k, n)
    # Predicted top-k order
    pred_order = np.argsort(-pred_r)[:k]
    # Use shifted true reward so DCG is non-negative even with negative true rewards
    shift = max(0.0, -true_r.min() + 1e-9)
    true_shifted = true_r + shift
    dcg = sum(true_shifted[idx] / np.log2(i + 2) for i, idx in enumerate(pred_order))
    # Ideal: top-k of true_r
    ideal_order = np.argsort(-true_r)[:k]
    idcg = sum(true_shifted[idx] / np.log2(i + 2) for i, idx in enumerate(ideal_order))
    return float(dcg / idcg) if idcg > 0 else 0.0


def _precision_at_k(true_r: np.ndarray, pred_r: np.ndarray, k: int) -> float:
    """Fraction of model's top-k that are in true top-k."""
    n = len(true_r)
    k = min(k, n)
    pred_top = set(np.argsort(-pred_r)[:k])
    true_top = set(np.argsort(-true_r)[:k])
    return float(len(pred_top & true_top) / k)


def _topk_regret(true_r: np.ndarray, pred_r: np.ndarray, k: int) -> float:
    """Relative top-k regret: 1 - sum(true_r[pred top-k]) / sum(true_r[true top-k]).
    0 = optimal selection; 1 = worst possible selection.
    """
    n = len(true_r)
    k = min(k, n)
    pred_top = np.argsort(-pred_r)[:k]
    true_top = np.argsort(-true_r)[:k]
    pred_score = float(true_r[pred_top].sum())
    true_score = float(true_r[true_top].sum())
    if true_score - true_r.sum() / n * k <= 0:  # all-tied case
        return 0.0
    # normalise: 0 when pred_top == true_top, larger when worse
    return float(1.0 - (pred_score - true_r.mean() * k) / max(true_score - true_r.mean() * k, 1e-9))


def _per_state_metrics(model, bf, gf, actions, rewards, k_list=(1, 5, 10, 50)):
    """Run the model on every state, accumulate per-K metrics."""
    model.eval()
    n_states, n_actions = actions.shape
    metrics = {f"ndcg@{k}": [] for k in k_list}
    metrics.update({f"precision@{k}": [] for k in k_list})
    metrics.update({f"regret@{k}": [] for k in k_list})
    metrics["spearman"] = []
    metrics["kendall"] = []
    metrics["pairwise_acc"] = []
    metrics["pred_reward_std"] = []

    correct, total = 0, 0
    with torch.no_grad():
        for s in range(n_states):
            bf_s = torch.from_numpy(bf[s]).float().unsqueeze(0).expand(n_actions, -1, -1)
            gf_s = torch.from_numpy(gf[s]).float().unsqueeze(0).expand(n_actions, -1)
            acts = torch.from_numpy(actions[s].astype(np.int64))
            _, _, pred_r = model(bf_s, gf_s, acts)
            pred = pred_r.squeeze(-1).numpy()
            true = rewards[s]

            # NDCG / Precision / Regret per K
            for k in k_list:
                metrics[f"ndcg@{k}"].append(_ndcg_at_k(true, pred, k))
                metrics[f"precision@{k}"].append(_precision_at_k(true, pred, k))
                metrics[f"regret@{k}"].append(_topk_regret(true, pred, k))

            # Rank correlation
            if true.std() > 1e-9 and pred.std() > 1e-9:
                rho, _ = spearmanr(true, pred)
                tau, _ = kendalltau(true, pred)
                metrics["spearman"].append(rho)
                metrics["kendall"].append(tau)

            metrics["pred_reward_std"].append(float(pred.std()))

            # Pairwise accuracy (existing diagnostic)
            ii, jj = np.triu_indices(n_actions, k=1)
            true_diff = true[ii] - true[jj]
            pred_diff = pred[ii] - pred[jj]
            valid = np.abs(true_diff) > 1e-9
            correct += int(((np.sign(true_diff) == np.sign(pred_diff)) & valid).sum())
            total += int(valid.sum())

    metrics["pairwise_acc_total"] = float(correct / max(total, 1))
    metrics["pairwise_acc_n_pairs"] = int(total)
    # aggregate
    out = {}
    for k, v in metrics.items():
        if isinstance(v, list) and len(v) > 0:
            arr = np.array(v, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                out[k + "_mean"]   = float(arr.mean())
                out[k + "_median"] = float(np.median(arr))
                out[k + "_std"]    = float(arr.std())
        else:
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble-dir", type=Path, required=True,
                    help="Dir containing ensemble_member{0..2}.pt")
    ap.add_argument("--pairwise",    type=Path, required=True)
    ap.add_argument("--n-blocks",    type=int, required=True)
    ap.add_argument("--k-global",    type=int, default=12)
    ap.add_argument("--label",       type=str, required=True)
    ap.add_argument("--out-json",    type=Path, required=True)
    args = ap.parse_args()

    # Load pairwise
    npz = np.load(args.pairwise)
    bf_key = "states_bf" if "states_bf" in npz.files else "block_features"
    gf_key = "states_gf" if "states_gf" in npz.files else "global_features"
    bf = npz[bf_key]; gf = npz[gf_key]
    actions = npz["actions"]; rewards = npz["rewards"]
    log.info("pairwise: %s %s, actions %s, rewards %s",
             bf_key, bf.shape, actions.shape, rewards.shape)

    # Load each ensemble member
    out = {"label": args.label, "n_blocks": args.n_blocks, "members": []}
    for i in range(3):
        pt = args.ensemble_dir / f"ensemble_member{i}.pt"
        log.info("  member %d: %s", i, pt.name)
        sd = torch.load(pt, map_location="cpu", weights_only=True)
        model = TransitionModel(n_blocks=args.n_blocks, k_global=args.k_global)
        model.load_state_dict(sd)
        m = _per_state_metrics(model, bf, gf, actions, rewards)
        out["members"].append(m)

    # Aggregate over members
    keys = [k for k in out["members"][0] if k.endswith("_mean") or k.endswith("_total")]
    out["ensemble_mean"] = {}
    for k in keys:
        vals = [m.get(k) for m in out["members"] if k in m and m[k] is not None]
        out["ensemble_mean"][k] = float(np.mean(vals)) if vals else None

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    log.info("wrote %s", args.out_json)

    # Print summary
    em = out["ensemble_mean"]
    print()
    print(f"=== {args.label} (ensemble of 3) ===")
    print(f"  pairwise_acc       = {em.get('pairwise_acc_total', 0):.4f}")
    print(f"  spearman_mean      = {em.get('spearman_mean', 0):.4f}")
    print(f"  kendall_mean       = {em.get('kendall_mean', 0):.4f}")
    for k in [1, 5, 10, 50]:
        print(f"  NDCG@{k:>2} mean      = {em.get(f'ndcg@{k}_mean', 0):.4f}    "
              f"Precision@{k:>2} = {em.get(f'precision@{k}_mean', 0):.4f}    "
              f"regret@{k:>2} = {em.get(f'regret@{k}_mean', 0):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
