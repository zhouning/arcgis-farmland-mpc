# Scientific Reports Submission Package, Paper 9

This is the active Scientific Reports submission package for Paper 9:

**Reproducible model-based planning for county-scale farmland consolidation in fragmented mountain landscapes**

Use this package for the journal upload and for DOI backfilling after the Zenodo release has been minted.

## Upload Folders

- `01_main_document/01_main_manuscript_scirep.pdf`: main manuscript PDF.
- `02_cover_letter/00_cover_letter_scirep.pdf`: cover letter PDF.
- `03_supplementary_information/02_supplementary_information_scirep.pdf`: Supplementary Information PDF.
- `04_figures/`: upload-ready figure files.
- `05_source_editable/`: editable LaTeX sources, bibliography, generated `.bbl`, and local figure copies.
- `06_declarations_and_checks/`: declarations and submission checklist.
- `07_zenodo_release/`: GitHub-Zenodo release notes, metadata drafts, and DOI backfill instructions.

## 2026-06-14 Robustness Evidence

- Added a Neijiang Dongxing seven-profile Tool 4 execution-constraint frontier to mirror the Bishan execution-frontier evidence.
- Summary outputs are saved as `neijiang_constraint_frontier.json` and `neijiang_constraint_frontier.md`.
- The rendered frontier figure is saved under `04_figures/` and copied into `05_source_editable/figures/` for LaTeX compilation.
- The full per-profile GIS outputs under `runs/neijiang/pareto/` are local, reproducible run artefacts and are not part of the upload set.

## Source Caveats

- The Scientific Reports package keeps the claims bounded to technical validity, reproducibility, and auditability.
- The raw Bishan and Neijiang cadastral records remain restricted. The open reproduction tracks cover the synthetic benchmark, the Buchanan boundary check, and training/planning diagnostics; derived cadastral products support real-county verification without redistributing raw parcel geometries.
- For LaTeX upload, include both `references_v6_codex.bib` and the generated `.bbl` files if the submission system requests source files.
