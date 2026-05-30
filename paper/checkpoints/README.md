# Trained checkpoints — Paper 9 v7 ensembles

The contrastive 3-member transition-model ensembles that produce every
headline number in §5 and §7 of the paper. Provided so external readers (and
the macOS workstation running follow-on retrains) can:

1. **Reproduce inference** — load these and re-run MPC; results should match
   the per-seed JSON in `neijiang_cross_region/` and the toolbox numbers in
   the paper to floating-point reduction noise.
2. **Sanity-check a re-train** — train your own ensemble from scratch on the
   same pairwise data, then compare its eval distribution against ours.
3. **Run partial-transfer experiments** — Neijiang `train_5seed_neijiang.py`
   in partial mode requires the Bishan backbone in
   `bishan/contrastive_5seed/`.

## Layout

```
checkpoints/
├── MANIFEST.sha256              # bit-level integrity check
├── bishan/
│   ├── contrastive_5seed/       # 5 seeds × 3 members = 15 .pt
│   │   └── ensemble_seed{0..4}_lam5.0_member{0..2}.pt
│   ├── shipped_onnx/            # 1 ensemble × 3 members = 3 .onnx
│   │   └── ensemble_lam5.0_member{0..2}.onnx
│   └── lambda_ablation/         # λ-sweep diagnostic (§3 ranking failure)
│       ├── ensemble_lam{0.0,1.0,5.0}_member{0..2}.pt   # 9 .pt
│       └── discriminative_results.json
└── neijiang/
    ├── baseline/                # from-scratch, 5 × 3 = 15 .pt
    │   └── ensemble_seed{0..4}_lam5.0_member{0..2}.pt
    └── partial_transfer/        # Bishan backbone + local fine-tune, 5 × 3 = 15 .pt
        └── ensemble_seed{0..4}_lam5.0_member{0..2}.pt
```

## What each set produces

| Directory | Paper section | Headline result |
|---|---|---|
| `bishan/contrastive_5seed/` | §5 Table 4 | slope **−1.289 ± 0.079 %**, baimu area **−312 ± 34 ha** |
| `bishan/shipped_onnx/` | §7 toolchain | slope **−1.544 ± 0.041 %** (1 ensemble × 5 episodes) |
| `bishan/lambda_ablation/` | §3 Table 1 (ranking-failure diagnostic) | pairwise ranking accuracy: λ=0.0 → **51.6 %** (essentially random); λ=1.0 → 73.2 %; λ=5.0 → **85.5 %**. The same 0.0 → 5.0 sweep is what motivates the contrastive intervention reported in later sections. Numerical breakdown in `discriminative_results.json`. |
| `neijiang/baseline/` | §5 cross-region | slope **−0.501 ± 0.024 %**, baimu area **+267 ± 38 ha** |
| `neijiang/partial_transfer/` | §5 (transfer caveat) | slope **−0.493 ± 0.017 %**, baimu area **+62 ± 80 ha** |

## Architecture notes

All `.pt` files are PyTorch state dicts (23 keys) for a `TransitionModel`
with shape determined by `n_blocks`:

| | n_blocks | action_emb.weight | total params |
|---|---|---|---|
| Bishan | 2,600 | `[2600, 32]` | 236,958 |
| Neijiang | 3,711 | `[3711, 32]` | 272,510 |

The `.onnx` files are the same model exported with `n_blocks=2600` baked
into the input shape (`block_features: [batch, 2600, 17]`,
`global_features: [batch, 12]`, `action: [batch]` → `next_block`,
`next_global`, `reward`). Opset is the ArcGIS-Pro-compatible default.
Cross-county reuse therefore requires re-export, which is why `Tool 4` in
the toolbox carries an explicit pre-flight check (paper §7).

## Loading

```python
import torch
from data_agent.transition_model import TransitionModel, EnsembleTransitionModel

# Bishan single member
sd = torch.load("paper/checkpoints/bishan/contrastive_5seed/ensemble_seed0_lam5.0_member0.pt",
                map_location="cpu", weights_only=True)
model = TransitionModel(n_blocks=2600)
model.load_state_dict(sd)
model.eval()

# Bishan full 3-member ensemble for seed 0
ens = EnsembleTransitionModel(n_blocks=2600, n_models=3)
for i in range(3):
    sd = torch.load(f"paper/checkpoints/bishan/contrastive_5seed/ensemble_seed0_lam5.0_member{i}.pt",
                    map_location="cpu", weights_only=True)
    ens.models[i].load_state_dict(sd)
ens.eval()
```

`TransitionModel` and `EnsembleTransitionModel` live in the research
checkout's `data_agent/` (path: `$P9_ADK_DIR/data_agent/transition_model.py`).
The published `farmland_mpc` package consumes the `.onnx` exports rather
than these `.pt` files.

## Bit-level integrity

```bash
cd paper/checkpoints
sha256sum -c MANIFEST.sha256
# all 48 files should report OK
```

If `sha256sum -c` reports any failure, re-clone with `git lfs install`
disabled and ensure `core.autocrlf` is `false` (these are binary tensors;
LF/CRLF rewriting would corrupt them — they are tracked as binary by git
already, but worth checking on Windows runners).

## Training commit traceability

| Set | Trained on | Commit / mtime |
|---|---|---|
| Bishan contrastive 5-seed | research-side `paper9_contrastive/contrastive_trainer.py` | 2026-05-06 06:02–07:24 (5 sequential seeds, pre-`f8ad31c` package release) |
| Bishan shipped ONNX | exported from `bishan/contrastive_5seed/` seed 0 | 2026-05-10 |
| Bishan λ-ablation (lam0/1/5) | same trainer, single run per λ value (NOT one of the 5 seeds above) | 2026-05-06 00:32–01:24 |
| Neijiang baseline | research-side `train_5seed_neijiang.py` from scratch | 2026-05-08 |
| Neijiang partial-transfer | same script, mode=partial, backbone=Bishan seed-matched | 2026-05-08 |

Note: `bishan/lambda_ablation/ensemble_lam5.0_member*.pt` and
`bishan/contrastive_5seed/ensemble_seed0_lam5.0_member*.pt` are NOT the
same checkpoints (different mtimes, different SHA-256). The ablation set
is the dedicated single-run reference for the §3 Table 1 pairwise-accuracy
numbers; the 5-seed set is the multi-seed evaluation that produces §5
Table 4. Use whichever matches the paper section you're reproducing.

All trained with `baimu_area_penalty=2000` (research env default; see Eq.1
in the v7 draft after `dca73a8`, and the §sec:methods-trueenv discussion
of why the planning behaviour is determined by the ONNX reward head).

## Relation to data redistribution

These checkpoints are model weights — float tensors fitted to the pairwise
ranking loss on per-block aggregated features. They do not encode raw
parcel geometry, ownership, or per-parcel attributes from the Third
National Land Survey records. Releasing them is consistent with the paper's
Data Availability statement, which permits release of "(b) aggregated
block-level features and the anonymised pairwise dataset"; the trained
ensembles are a deterministic function of (b) plus the random seeds, also
fixed.
