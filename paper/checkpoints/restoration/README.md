# Restoration ensembles

These contrastive three-member ensembles support the cross-domain boundary-check experiments in the Scientific Reports manuscript. Two non-farmland geospatial planning cases are included:

```text
restoration/
|-- buchanan_va/                 real public-data case
|   |-- ensemble_seed{0..4}/      lambda=5.0 contrastive ensembles
|   |-- ensemble_lam0/            lambda=0.0 MSE-only baseline ensemble
|-- synthetic/                    deterministic synthetic case
    |-- ensemble_seed{0..4}/      lambda=5.0 contrastive ensembles
    |-- ensemble_lam0/            lambda=0.0 MSE-only baseline ensemble
```

## What each set produces

| Directory | Result reproduced |
|---|---|
| `buchanan_va/ensemble_seed{0..4}/` | Buchanan VA five-seed contrastive MPC: cumulative reward +229.4 +/- 0.0, +158% over random baseline. |
| `buchanan_va/ensemble_lam0/` | MSE-only ensemble on Buchanan: cumulative reward +229.3, within noise of contrastive, showing that the farmland MSE-vs-contrastive gap is not present in this restoration regime. |
| `synthetic/ensemble_seed{0..4}/` | Synthetic mine five-seed contrastive MPC: cumulative reward +8,128 +/- 63, +32% over random. |
| `synthetic/ensemble_lam0/` | MSE-only baseline on synthetic: +8,245, also within noise of contrastive. |

## Architecture

The restoration cases use the same `farmland_mpc.transition_model.TransitionModel` as the farmland ensembles in `paper/checkpoints/{bishan,neijiang}/`, instantiated with the case-specific number of planning units.

| Case | `n_units` | `action_emb.weight` | total params |
|---|---:|---|---:|
| Buchanan | 562 | `[562, 32]` | 165,486 |
| Synthetic | 420 | `[420, 32]` | 158,366 |

State features are 17-dimensional and global state is 12-dimensional, matching the farmland network so the same MPC code path applies.

## Loading

```python
import torch
from farmland_mpc.transition_model import TransitionModel

sd = torch.load(
    "paper/checkpoints/restoration/buchanan_va/ensemble_seed0/ensemble_member0.pt",
    map_location="cpu",
    weights_only=True,
)
model = TransitionModel(n_blocks=562, k_global=12)
model.load_state_dict(sd)
model.eval()
```

The `.onnx` exports are baked in to the relevant `n_blocks` and are consumed by `farmland_mpc.ensemble_runner.EnsembleOrtRunner` and by `farmland_mpc.mpc_plan`.

## Provenance

| Set | Trained on | Date |
|---|---|---|
| Buchanan VA contrastive five-seed | `farmland_mpc.train_ensemble`, 14-core CPU, three ensembles in parallel | 2026-05-30 |
| Buchanan VA lambda=0 baseline | same script with `--lambda-rank 0.0`; one ensemble, three members | 2026-05-30 |
| Synthetic contrastive five-seed | same as above on the synthetic case | 2026-05-30 |
| Synthetic lambda=0 baseline | same | 2026-05-30 |

All contrastive sets use `lambda_rank=5.0`, margin 0.1, batch size 256, 30 epochs, Adam optimizer with learning rate 1e-3 and weight decay 1e-5, and an 80/20 train/validation split on the case pairwise dataset.

## Reproducing

The full restoration verification workflow is described in `docs/REPRODUCE.md` under "Cross-domain verification (restoration)". On a 14-core Apple Silicon Mac, wall time is approximately 30 minutes per case, dominated by the ensemble training jobs.

## Data Redistribution

These checkpoints are model weights fitted to:

- Buchanan: aggregated 2-km planning-unit features derived from public US datasets, including e-AMLIS abandoned-mine inventory, USGS NHD flowlines, USGS 3DEP elevation, and Census TIGER county boundaries.
- Synthetic: deterministic seed-controlled synthetic data released under CC-BY 4.0.

Neither set encodes restricted Bishan or Neijiang cadastral data; both are safe to redistribute.
