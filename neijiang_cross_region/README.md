# Neijiang Dongxing — cross-region replication (Paper 9 §5)

Artifacts for the second-county generalisation test reported in Paper 9 v7 §5
(Neijiang Dongxing District, Sichuan: 29 townships, 3,711 blocks, 76,376 parcels).

## What's here (public-safe subset)

| File | Contents |
|---|---|
| `5seed_multiobj_results_baseline.json` | 5-seed from-scratch contrastive-MPC results. `cross_seed.baimu_area_delta_ha_mean = 267.09 ± 37.96` → paper's "+267 ± 38 ha". |
| `5seed_multiobj_results_partial.json` | 5-seed partial-transfer (Bishan backbone + 5-ep local fine-tune). |
| `blocks/neijiang_summary.json` | Per-township block counts, swappable-parcel counts, baimu counts, total area. |
| `blocks/township_*/block_features.json` | Aggregated 17-dim block feature vectors (the "(b) aggregated block-level features" released under the paper's Data Availability statement). |
| `*.py` | The exact pipeline scripts (block build → transition collection → pairwise gen → 5-seed train → MPC eval). |

## What's NOT here (and why)

Per the paper's **Data Availability** statement, the raw Third National Land
Survey (三调) cadastral records cannot be redistributed. The following were
deliberately excluded from this public repo:

- **`blocks/township_*/block_compositions.json`** — raw per-parcel index lists
  per block (parcel-level cadastral structure).
- **`blocks/township_*/parcel_block_mapping.csv`** — raw parcel→block mapping.
- **`*.npz`** (`trajectories_6k_neijiang.npz` 2.1 GB, `pairwise_data_neijiang.npz`
  176 MB) — also exceed GitHub's 100 MB file limit.
- **`ensembles/**/*.pt`** — 30 trained ensemble members (32 MB). Model weights,
  not raw data; left out by default. Available on request for bit-level diffing.
- **`*.log`** — de-identified training/eval logs, kept locally.

## Running these scripts

They are provided as a **configuration reference**, not a turnkey artifact. They
depend on (a) the research-side modules (`county_env.py`, `block_definition*.py`,
`mpc_planner.py`, `data_agent.transition_model`) that live in the private research
checkout, not in this package, and (b) the restricted raw GeoPackage. Paths are
read from environment variables — set before use:

```bash
export P9_RESEARCH_DIR=/path/to/research/checkout   # has county_env.py, paper9_contrastive/, ...
export P9_ADK_DIR=/path/to/adk                       # has data_agent/
export NEIJIANG_GPKG=/path/to/neijiang_DLTB_with_slope.gpkg
# NEIJIANG_BLOCK_DIR defaults to ./blocks
```

## Key result (reproducible from the JSONs above)

| Metric | from-scratch (5 seeds) | partial-transfer (5 seeds) |
|---|---|---|
| slope | −0.50 ± 0.02 % | −0.49 ± 0.02 % |
| Δcontiguity | +0.0337 ± 0.0011 | +0.0238 ± 0.0032 |
| Δbaimu area | **+267 ± 38 ha** | +62 ± 80 ha |
| Δbaimu count | −11.1 ± 3.4 | — |

The baimu **area** rises while **count** falls: consolidation operates on small
fragments rather than retiring the largest existing patches (paper §5).
