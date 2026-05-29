"""5-seed contrastive ensemble training for Neijiang cross-region experiment.

Two modes:
  --mode baseline : train 5 seeds × 3 members from scratch on Neijiang data.
  --mode partial  : transfer block_enc/global_enc/heads from Bishan ensemble,
                    reinit action_emb (different n_blocks: 2600 vs 3711),
                    finetune 5 epochs.

Bishan ckpts: D:/test/paper9_contrastive/multi_seed/ensemble_seed{0..4}_lam5.0_member{0..2}.pt
Neijiang out: D:/test/neijiang_cross_region/ensembles/{baseline,partial}/
"""
import os, sys
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.environ.get("P9_ADK_DIR", "/path/to/adk"))
sys.path.insert(0, os.environ.get("P9_RESEARCH_DIR", "/path/to/research/checkout"))
sys.path.insert(0, os.path.join(os.environ.get("P9_RESEARCH_DIR", "/path/to/research/checkout"), "paper9_contrastive"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_agent.transition_model import TransitionModel, EnsembleTransitionModel  # noqa: E402
from contrastive_trainer import ContrastiveTransitionTrainer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRAIN_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trajectories_6k_neijiang.npz")
PAIRWISE_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairwise_data_neijiang.npz")
BISHAN_CKPT_DIR = Path(os.environ.get("P9_RESEARCH_DIR", "/path/to/research/checkout")) / "paper9_contrastive" / "multi_seed"

OUT_ROOT = Path(os.path.dirname(os.path.abspath(__file__))) / "ensembles"

LAMBDA_RANK = 5.0
N_MEMBERS = 3


def transfer_bishan_weights(target_model: TransitionModel, bishan_ckpt_path: Path):
    """Copy block_enc / global_enc / heads from Bishan ckpt; keep action_emb fresh.

    Bishan action_emb has shape (2600, 32); Neijiang has (3711, 32) → skip copy.
    All other modules are independent of n_blocks so copy verbatim.
    """
    bishan_sd = torch.load(bishan_ckpt_path, map_location='cpu')
    target_sd = target_model.state_dict()

    copied, skipped = [], []
    for k, v in bishan_sd.items():
        if k.startswith('action_emb'):
            skipped.append(k)
            continue
        if k in target_sd and target_sd[k].shape == v.shape:
            target_sd[k] = v
            copied.append(k)
        else:
            skipped.append(f"{k} (shape mismatch: {v.shape} vs {target_sd.get(k, 'missing').shape if k in target_sd else 'missing'})")

    target_model.load_state_dict(target_sd)
    logger.info("Transfer: copied %d tensors, skipped %d (%s)",
                len(copied), len(skipped), skipped[:3])
    return target_model


def train_one_seed(seed, mode, train_data, pairwise_data, pw_train_idx, out_dir):
    """Train one ensemble (N_MEMBERS models) with given seed + mode."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_blocks = train_data['block_features'].shape[1]

    train_pw_data = {
        'states_bf': pairwise_data['states_bf'][pw_train_idx],
        'states_gf': pairwise_data['states_gf'][pw_train_idx],
        'actions': pairwise_data['actions'][pw_train_idx],
        'rewards': pairwise_data['rewards'][pw_train_idx],
    }

    ensemble = EnsembleTransitionModel(n_blocks, n_models=N_MEMBERS)
    epochs = 15 if mode == 'baseline' else 5

    for i in range(N_MEMBERS):
        member_seed = seed * 1000 + i
        torch.manual_seed(member_seed)
        np.random.seed(member_seed)
        model = TransitionModel(n_blocks)

        if mode == 'partial':
            bishan_path = BISHAN_CKPT_DIR / f"ensemble_seed{seed}_lam5.0_member{i}.pt"
            if not bishan_path.exists():
                raise FileNotFoundError(f"Bishan ckpt not found: {bishan_path}")
            transfer_bishan_weights(model, bishan_path)

        trainer = ContrastiveTransitionTrainer(
            model, lr=1e-3, epochs=epochs, patience=20,
            lambda_rank=LAMBDA_RANK, margin=0.1, n_pairs_per_state=10,
            batch_size=256, pw_subsample=50,
        )
        t0 = time.time()
        history = trainer.train(train_data, train_pw_data)
        elapsed = time.time() - t0
        be = history['best_epoch']
        cs = history['cosine_sim'][be-1] if be and be > 0 else -1
        ra = history['ranking_acc'][be-1] if be and be > 0 else -1
        logger.info("  seed=%d member=%d [%s]: best_epoch=%s cos_sim=%.4f rank_acc=%.3f (%.1fs)",
                    seed, i, mode, be, cs, ra, elapsed)
        ensemble.models[i] = model

    save_path = out_dir / f"ensemble_seed{seed}_lam{LAMBDA_RANK}"
    for i in range(N_MEMBERS):
        torch.save(ensemble.models[i].state_dict(), f"{save_path}_member{i}.pt")
    return ensemble


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['baseline', 'partial'], required=True)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated list of seeds (overrides n_seeds)")
    args = parser.parse_args()

    out_dir = OUT_ROOT / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading training data from %s...", TRAIN_DATA)
    data = np.load(TRAIN_DATA)
    train_data = {k: data[k] for k in data.files}
    logger.info("  transitions: %d, n_blocks=%d",
                len(train_data['actions']), train_data['block_features'].shape[1])

    logger.info("Loading pairwise data from %s...", PAIRWISE_DATA)
    pw_raw = np.load(PAIRWISE_DATA)
    pairwise_data = {k: pw_raw[k] for k in pw_raw.files}
    n_pw = len(pairwise_data['states_bf'])
    logger.info("  %d states × %d actions", n_pw, pairwise_data['actions'].shape[1])

    rng = np.random.default_rng(0)
    perm = rng.permutation(n_pw)
    split = int(0.8 * n_pw)
    pw_train_idx = perm[:split]

    if args.seeds is not None:
        seeds = [int(s) for s in args.seeds.split(',')]
    else:
        seeds = list(range(args.n_seeds))

    for seed in seeds:
        logger.info("=" * 60)
        logger.info("SEED %d  MODE=%s", seed, args.mode)
        t0 = time.time()
        train_one_seed(seed, args.mode, train_data, pairwise_data, pw_train_idx, out_dir)
        logger.info("  seed total: %.1fs", time.time() - t0)

    logger.info("All done. Ensembles saved to %s", out_dir)


if __name__ == "__main__":
    main()
