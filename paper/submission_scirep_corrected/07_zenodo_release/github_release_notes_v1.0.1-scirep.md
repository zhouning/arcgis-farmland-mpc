# Paper 9 Scientific Reports cleaned submission release

This release archives the code, trained ensembles, benchmark generators, verification scripts and Scientific Reports submission package for:

**Reproducible model-based planning for county-scale farmland consolidation in fragmented mountain landscapes**

## Included

- Python package for contrastive transition-model training, model-predictive planning and evaluation.
- ArcGIS Pro Python toolbox used for the Tool 1-5 deployment workflow.
- Deterministic synthetic farmland benchmark under CC-BY 4.0.
- Public-data Buchanan County restoration boundary-check workflow.
- Trained contrastive ensembles and ablation checkpoints tracked under `paper/checkpoints/`.
- Verification scripts for independent GIS recomputation and related audit checks.
- Scientific Reports submission package under `paper/submission_scirep_corrected/`, including main manuscript, supplementary information, cover letter, figures, source files and declarations.
- Neijiang execution-constraint frontier summary outputs.

## Restricted data boundary

The raw Bishan District and Neijiang Dongxing District cadastral records derive from the Third National Land Survey of China and are not redistributed in this release. Derived, anonymised and aggregate artefacts needed for verification and re-analysis are included where public redistribution is permitted.

## Reproducibility notes

Open reproduction tracks are available through the synthetic benchmark, the public Buchanan restoration case, trained ensembles, de-identified logs and the smoke-test workflow. See `docs/REPRODUCE.md`, `benchmark/README.md`, `verification/README.md` and the Scientific Reports Data Availability statement for the exact access boundary.

## Suggested citation

Use this cleaned release DOI for reproducible Scientific Reports review:

https://doi.org/10.5281/zenodo.20713695
