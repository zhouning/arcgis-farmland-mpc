# ArcGIS Farmland MPC

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zhouning/arcgis-farmland-mpc/blob/main/notebooks/farmland_mpc_colab_demo.ipynb)
[![Paper draft](https://img.shields.io/badge/paper-Scientific%20Reports%20submission-brightgreen)](paper/submission_scirep_corrected/01_main_document/01_main_manuscript_scirep.pdf)

Reproducible model-based planning for county-scale farmland consolidation in fragmented mountain landscapes. The repository contains the Python algorithm core, ArcGIS Pro toolbox, QGIS wrapper, trained ensembles, open synthetic benchmark, public-data boundary-check case, verification scripts, and the active Scientific Reports submission package.

> Headline result: on Bishan District, Chongqing (52,515 parcels), contrastive learned-surrogate MPC reduces area-weighted farmland slope by -1.289 +/- 0.079% across five independently trained ensembles, while running in desktop planning time. The same workflow is audited on Neijiang Dongxing (76,376 parcels) and stress-tested on open synthetic and public-data restoration cases.

## Repository Contents

| Path | Contents |
|---|---|
| `farmland_mpc/` | Python algorithm core: environment, transition model, contrastive trainer, ensemble runner, MPC planner, sampler, and CLI entry points. |
| `LandUseOptimization_P9.pyt` | ArcGIS Pro Python toolbox with the five-stage planning workflow. |
| `qgis_plugin/` | QGIS Processing wrapper around the same command-line workflow. |
| `benchmark/` | Open synthetic farmland benchmark with seven deterministic landscape presets, released under CC-BY 4.0. |
| `runs/` | Reproduction artefacts, pairwise datasets, public restoration results, OR baselines, simulator-cost sweeps, and ranking metrics. |
| `verification/` | Independent GIS recomputation and audit checks for headline results. |
| `paper/checkpoints/` | Trained contrastive ensembles and ablation checkpoints for Bishan, Neijiang, and restoration cases. |
| `paper/submission_scirep_corrected/` | Active Scientific Reports submission package, including main manuscript, supplementary information, cover letter, figures, source files, declarations, checklist, and Zenodo release preparation files. |
| `docs/` | Reproduction, deployment, quickstart, user-guide, macOS, and Docker notes. |
| `notebooks/` | Colab demonstration notebook. |
| `scripts/` | Training, evaluation, policy-audit, sensitivity, figure, and frontier-generation drivers. |

## Deployment

### Option A: Python CLI

```bash
git clone https://github.com/zhouning/arcgis-farmland-mpc.git
cd arcgis-farmland-mpc
conda env create -f environment.yml
conda activate farmland-mpc
farmland-mpc --help
```

No ArcGIS licence is required for the Python CLI. It supports Windows, macOS, and Linux. macOS users should also read [docs/MACOS.md](docs/MACOS.md). The Colab path is available through [notebooks/farmland_mpc_colab_demo.ipynb](notebooks/farmland_mpc_colab_demo.ipynb).

### Option B: ArcGIS Pro Toolbox

```text
1. Copy this repository to the target machine.
2. In the ArcGIS Python Command Prompt:
     pip install torch --index-url https://download.pytorch.org/whl/cpu
     pip install onnx onnxruntime gymnasium
3. In ArcGIS Pro: Add Toolbox -> LandUseOptimization_P9.pyt
4. Double-click "5. Check Dependencies" until every line is [OK].
5. Run Tool 1 -> 2 -> 3 -> 4 in order.
```

The ArcGIS path requires ArcGIS Pro 3.7 with Spatial Analyst and Image Analyst extensions.

## Pipeline

| # | Tool | Input | Output | Typical wall time |
|---|---|---|---|---|
| 1 | Prepare Data & Blocks | `DLTB.shp` + DEM (+ optional `XZQ.shp`) | `<prepared_dir>/` | 10-15 min |
| 2 | Sample Transitions | Tool 1 prepared directory | `<prepared_dir>/tool2/*.npz` | 15-25 min |
| 3 | Train Contrastive Ensemble | Tool 2 transition samples | `<prepared_dir>/tool3/*.onnx` | 30-60 min |
| 4 | MPC Planning | Tool 1 prepared data + Tool 3 ONNX ensemble | `optimized_dltb.shp` | about 7 min per scenario |
| 5 | Check Dependencies | none | diagnostic log | seconds |

Tools 1-3 are normally run once per region and training configuration. Tool 4 is the planning loop that users re-run under operational scenarios.

## Reproducibility Boundary

Everything needed to reproduce the open-track findings is public:

- Synthetic farmland benchmark: seven deterministic presets under CC-BY 4.0.
- Public Buchanan County, Virginia restoration boundary-check case using OSMRE e-AMLIS, USGS NHD, USGS 3DEP, and Census TIGER data.
- Trained contrastive ensembles and ablation checkpoints under `paper/checkpoints/`.
- Aggregated block-level features for Bishan and Neijiang plus anonymised pairwise data under `runs/pairwise/`.
- Random seeds, hyperparameters, de-identified logs, verification scripts, and a small smoke test.

Raw Bishan and Neijiang cadastral records derive from China's Third National Land Survey and are not redistributed. The public synthetic and Buchanan cases form a fully open reproduction track; the derived Bishan and Neijiang artefacts support verification and re-analysis without redistributing raw parcel geometries.

## Scientific Reports Package

The active manuscript package is:

```text
paper/submission_scirep_corrected/
```

Key subfolders:

| Path | Contents |
|---|---|
| `01_main_document/` | Main manuscript PDF for upload. |
| `02_cover_letter/` | Scientific Reports cover letter PDF. |
| `03_supplementary_information/` | Supplementary Information PDF. |
| `04_figures/` | Figure files for upload. |
| `05_source_editable/` | Editable LaTeX sources, bibliography, generated `.bbl`, and local figure copies. |
| `06_declarations_and_checks/` | Declarations and submission checklist. |
| `07_zenodo_release/` | GitHub-Zenodo release notes, metadata drafts, and DOI backfill instructions. |

## Documentation

| Document | Audience |
|---|---|
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | Reproduction workflow. |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | First-time deployment. |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | Five-minute post-deployment check. |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | Operator workflow and parameters. |
| [docs/MACOS.md](docs/MACOS.md) | macOS and Apple Silicon notes. |
| [docs/DOCKER.md](docs/DOCKER.md) | Containerized CLI, JupyterLab, and FastAPI paths. |
| [verification/README.md](verification/README.md) | Independent verification stack. |
| [benchmark/README.md](benchmark/README.md) | Synthetic benchmark details. |
| [paper/submission_scirep_corrected/README_ScientificReports_submission_package.md](paper/submission_scirep_corrected/README_ScientificReports_submission_package.md) | Scientific Reports submission package map. |

## Operational Notes

1. Region changes require retraining the ensemble because `n_blocks` is baked into each ONNX member.
2. The default projected CRS is EPSG:32648 (UTM Zone 48N), which is appropriate for the Chinese study regions here but should be overridden for other regions.
3. Tool 1 in the ArcGIS workflow requires the Spatial Analyst extension.
4. Reward-weight overrides at Tool 4 do not retrain the learned reward head. To steer planning under new reward weights, rerun Tools 2 and 3 with the new configuration.

## Citation

The associated manuscript is under submission to Scientific Reports. The cleaned submission release is archived on Zenodo as version `v1.0.1-scirep`: https://doi.org/10.5281/zenodo.20713695. Cite this version DOI for reproducible manuscript review; the article DOI can be added after acceptance.

## License

MIT for code. CC-BY 4.0 for the synthetic benchmark. See [LICENSE](LICENSE).
