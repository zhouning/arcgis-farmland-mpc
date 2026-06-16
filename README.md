# ArcGIS Farmland MPC

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zhouning/arcgis-farmland-mpc/blob/main/notebooks/farmland_mpc_colab_demo.ipynb)
[![Paper draft](https://img.shields.io/badge/paper-Scientific%20Reports%20submission-brightgreen)](paper/submission_scirep_corrected/01_main_document/01_main_manuscript_scirep.pdf)

End-to-end pipeline for **county-scale farmland-consolidation planning**, runnable on a desktop CPU. The repository contains the trained ensembles, the deterministic synthetic benchmark, the public-data cross-domain test, the ArcGIS Pro toolbox, and the non-commercial Python CLI used to generate the headline results in the accompanying paper.

> **Headline result**: on Bishan District, Chongqing (52,515 parcels) the contrastive learned-surrogate + MPC pipeline reduces area-weighted farmland slope by **−1.289 ± 0.079 %** (5 seeds), **1.6× the magnitude** of in-house Centralised-PPO and multi-agent baselines while reducing wall time by roughly two orders of magnitude (8–12 GPU-hours per seed → ~7 minutes per scenario on a 12-thread desktop CPU). The result reproduces on Neijiang Dongxing (76,376 parcels). Under matched 180 s wall budgets, classical operations-research methods (simulated annealing, random restart) reach less than **11%** of the pipeline's reward on the same simulator — the empirical reason the pipeline dominates this domain.

## What's in this repository

| Path | Contents |
|------|----------|
| `farmland_mpc/` | Python algorithm core: env, ensemble runner, MPC planner, contrastive trainer, sampler |
| `farmland_mpc/tests/` | OR baselines (greedy, SA, NSGA-II, CBC-MILP), simulator-cost sweep, ranking metrics, Pareto sweep |
| `paper/` | Manuscript drafts (NCS variant, **CommsEE submission variant**), supplementary, cover letter, all figures |
| `paper/submission_scirep_corrected/` | Corrected submission package for *Scientific Reports* |
| `paper/submission_commsee/` | Previous *Communications Earth & Environment* submission reference |
| `paper/checkpoints/` | Trained contrastive ensembles for Bishan, Neijiang, restoration cases (~250 MB) |
| `runs/` | Reproduction artefacts: pairwise datasets, plan results, OR baselines, sim-cost sweeps, ranking metrics |
| `benchmark/` | Open synthetic farmland benchmark (7 deterministic landscape presets, CC-BY 4.0) |
| `verification/` | 5-layer independent verification stack (GIS recompute, ensemble subset, true-env ablation, MAE) |
| `LandUseOptimization_P9.pyt` | ArcGIS Pro Python Toolbox (5 tools) |
| `notebooks/farmland_mpc_colab_demo.ipynb` | End-to-end Colab demo, no install required |

## Two ways to deploy

### Option A — Pure Python CLI (recommended for reviewers, headless servers, QGIS users)

```bash
git clone https://github.com/zhouning/arcgis-farmland-mpc.git
cd arcgis-farmland-mpc
conda env create -f environment.yml
conda activate farmland-mpc
farmland-mpc --help
```

No ArcGIS license required. Works on Windows, macOS (Intel + Apple Silicon), and Linux.
**macOS users**: see [docs/MACOS.md](docs/MACOS.md) for Apple Silicon notes.
**Try the [Colab demo](notebooks/farmland_mpc_colab_demo.ipynb)** — runs end-to-end in your browser.

### Option B — ArcGIS Pro toolbox (GUI for planners)

```
1. Copy this repository to the target machine.
2. In the ArcGIS Python Command Prompt:
     pip install torch --index-url https://download.pytorch.org/whl/cpu
     pip install onnx onnxruntime gymnasium
3. In ArcGIS Pro: Add Toolbox -> LandUseOptimization_P9.pyt
4. Double-click "5. Check Dependencies" until every line is [OK].
5. Run Tool 1 -> 2 -> 3 -> 4 in order.
```

Requires ArcGIS Pro 3.7 + Spatial Analyst + Image Analyst extensions.

## Pipeline (4 + 1 tools)

| # | Tool | Input | Output | Wall time |
|---|------|-------|--------|-----------|
| 1 | Prepare Data & Blocks | DLTB.shp + DEM (+ optional XZQ.shp) | `<prepared_dir>/` | 10–15 min |
| 2 | Sample Transitions | Tool 1 prepared_dir | `<prepared_dir>/tool2/*.npz` | 15–25 min |
| 3 | Train Contrastive Ensemble | Tool 2 npz | `<prepared_dir>/tool3/*.onnx` | 30–60 min |
| 4 | **MPC Planning** | Tool 1 prepared + Tool 3 onnx | `optimized_dltb.shp` | **~7 min / scenario** |
| 5 | Check Dependencies | (none) | diagnostic log | seconds |

Tools 1–3 run once per region (one-time setup); Tool 4 is the iterated planning loop a planner re-runs under different reward weightings.

## Reproducibility

Everything required to reproduce the paper's qualitative findings is **public now** (not gated on publication):

- **Open synthetic farmland benchmark**: 7 deterministic landscape presets under CC-BY 4.0 (`benchmark/`). End-to-end sufficient to reproduce the contrastive-MPC qualitative findings without any access to restricted cadastral data.
- **Public-data Buchanan VA mine-restoration cross-domain case**: planning units, OSMRE e-AMLIS extracts, USGS NHD flowlines, USGS 3DEP slope, Census TIGER boundary — all US-government open data (`runs/restoration/buchanan_va/`).
- **Trained contrastive ensembles**: 5 × 3 ensemble members per case for both Chinese counties + lambda-ablation set + restoration ensembles (`paper/checkpoints/`, ~250 MB).
- **Aggregated block-level features**: 17-dimensional vectors for all 2,600 Bishan and 3,711 Neijiang blocks plus the anonymised pairwise dataset (`runs/pairwise/`).
- **All hyperparameters, random seeds, and de-identified training logs** for every reported experiment.
- **Smoke test**: `bash scripts/smoke_test.sh` runs Tool 1–4 on a 30-block subset in ~70 s as a continuous-integration check.

The raw cadastral records for Bishan and Neijiang are derived from the Third National Land Survey of China and cannot be publicly redistributed under existing data-governance restrictions. The synthetic benchmark and the Buchanan VA case jointly form a fully open reproduction track that does not require restricted-data access.

## Key findings

- **§5 (Bishan/Neijiang real-county results)**: −1.289 ± 0.079 % slope reduction on Bishan, reproducing on Neijiang Dongxing. Five-layer verification stack (independent GIS recomputation, ensemble subset ablation, MAE measurement, true-env ablation, cross-county replication).
- **§6 (synthetic benchmark)**: Contrastive MPC strongest on 5/7 fragmented landscape presets; GA wins only on the small consolidated preset where the action space is small enough for population search to cover.
- **§6.5 (cross-domain Buchanan VA)**: Contrastive MPC and MSE-only MPC are statistically indistinguishable on restoration cases — the contrastive intervention is conditional on σ_a/σ_s < 1, the regime to which farmland's geometric reward function belongs but restoration's attribute-lookup reward does not.
- **§sec:simcost (simulator-cost crossover)**: We sweep ten (case × reward-profile) cells × five injected step delays. The MPC-vs-SA gap is monotonic in step cost, with crossover at ~30 ms/step. Farmland's actual 17.9 ms/step combined with combinatorial cost interactions in the reward function is the empirical reason the pipeline dominates this domain.

## Repository structure

```
arcgis-farmland-mpc/
├── farmland_mpc/                    # Algorithm core (Python 3.13)
│   ├── blocks_env.py                # Region-agnostic env factory
│   ├── county_env.py                # Cadastral simulator (gym.Env)
│   ├── restoration_env.py           # Restoration-case env (5 reward profiles)
│   ├── contrastive_trainer.py       # MSE + margin ranking trainer
│   ├── ensemble_runner.py           # ONNX runtime wrapper
│   ├── mpc_plan.py                  # MPC scoring + commit (Tool 4)
│   ├── prepare_data.py              # Tool 1
│   ├── sample_transitions.py        # Tool 2
│   ├── train_ensemble.py            # Tool 3
│   ├── transition_model.py          # 237 k-parameter transition head
│   └── tests/
│       ├── or_baselines.py          # greedy / SA / NSGA-II / CBC-MILP
│       ├── eval_ranking_metrics.py  # NDCG, top-K regret, Spearman, Kendall
│       ├── simulator_cost_sweep.py  # step-delay × method sweep
│       ├── eval_mpc_multi_ep.py     # multi-episode budget-matched eval
│       ├── farmland_baselines.py    # SA / random-restart on Bishan
│       └── pareto_sweep.py          # 7-config Pareto front
├── benchmark/                       # Synthetic landscape generator + 7 presets (CC-BY 4.0)
├── verification/                    # Independent GIS recomputation + 4 other layers
├── paper/                           # Manuscripts and figures
│   ├── paper9_v7_draft.tex          # Original v7 draft
│   ├── paper9_v7_draft_ncs.tex      # NCS variant (36 pages, methodological framing)
│   ├── paper9_v7_draft_commsee.tex  # CommsEE variant (27 pages, application framing)
│   ├── paper9_v7_supplementary.tex  # Supplementary information
│   ├── checkpoints/                 # Trained ensembles (~250 MB)
│   ├── figures_v2/                  # All paper figures (PDF + PNG)
│   └── submission_commsee/          # Ready-to-submit CommsEE package
├── runs/                            # Reproduction artefacts (pairwise data, results, sweeps)
├── docs/                            # Deployment / quickstart / user guide / macOS / Docker
├── notebooks/farmland_mpc_colab_demo.ipynb
├── scripts/                         # Pipeline drivers (training grids, eval grids, etc.)
├── LandUseOptimization_P9.pyt       # ArcGIS Pro Python Toolbox
├── environment.yml                  # Conda environment definition
└── LICENSE                          # MIT
```

## Documentation

| Document | Audience |
|----------|----------|
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | First-time deployers — install guide for all paths |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | Post-deployment 5-minute end-to-end verification |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | Operators — full pipeline + parameters + troubleshooting |
| [docs/MACOS.md](docs/MACOS.md) | macOS install, Apple Silicon notes |
| [docs/DOCKER.md](docs/DOCKER.md) | CLI / JupyterLab / FastAPI form-UI containers |
| [verification/README.md](verification/README.md) | Reviewers — 5-layer headline-result verification stack |
| [benchmark/README.md](benchmark/README.md) | Benchmarkers — synthetic landscape generator |
| [paper/submission_commsee/README.md](paper/submission_commsee/README.md) | CommsEE submission package reference |

## Operational pitfalls

1. **Region swaps require retraining the ensemble.** `n_blocks` is baked statically into each ONNX member. Tool 4's `assert_compatible` check surfaces this early.
2. **Default projected CRS (EPSG:32648, UTM Zone 48N) only covers central and western China.** Override `proj_crs` on Tool 1 for other regions; otherwise slope and area metrics are computed in a distorted frame.
3. **Tool 1 requires the Spatial Analyst extension.** The Check Dependencies tool blocks the pipeline until this is resolved.
4. **The reward-weight overrides on Tool 4 are cosmetic without retraining.** UI fields change only what `env.step()` reports; MPC's candidate ranking uses the reward head learned by Tool 3 under a fixed weight set. To actually steer planning, retrain Tools 2 + 3 with the new weights.

## Citation

A peer-reviewed publication is currently under submission to *Scientific Reports*. The current submission package is at `paper/submission_scirep_corrected/`. Zenodo release preparation files are in `paper/submission_scirep_corrected/07_zenodo_release/`; citation details will be updated after the release DOI and article DOI are available.

## License

MIT (code). CC-BY 4.0 (synthetic benchmark). See [LICENSE](LICENSE).
