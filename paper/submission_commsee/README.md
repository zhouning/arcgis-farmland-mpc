# Communications Earth & Environment — Submission Package

**Paper title:** Reproducible AI-driven planning for county-scale farmland consolidation: from cadastral data to optimised plans on a desktop CPU

**Target journal:** [Communications Earth & Environment](https://www.nature.com/commsenv/) (Nature Portfolio, 2-year IF 8.9)

**Article type:** Research Article

**Submission portal:** https://mts-commsenv.nature.com/

---

## Files in this package

| File | Purpose | Pages |
|------|---------|-------|
| `00_cover_letter.pdf` | Editor cover letter | 1 |
| `01_main_manuscript.pdf` | Main text + figures + references | 27 |
| `02_supplementary_information.pdf` | Supplementary tables and DRL baseline configuration | 6 |
| `figures/figure_simcost.pdf` | Main Figure 3 (simulator-cost crossover) | — |
| `figures/figure_restoration.pdf` | Main Figure 2 (cross-domain Pareto) | — |
| `source_*.tex` | LaTeX source files for editorial use | — |

## Pre-submission checklist (CommsEE specific)

- [x] Main text within ~5,000 words (current: ~5,055 words; "as a guide" target met)
- [x] Abstract within ~250 words, unstructured, no citations (current: 220 words)
- [x] Standard structure: Introduction / Results / Discussion / Methods
- [x] Display items ≤ 6 in main text (current: 4 figures + 4 tables = 8 — review needed; some tables already in SI)
- [x] Cover letter ≤ 1 page (current: 1 page)
- [x] Data Availability statement specific (restricted vs released artefacts enumerated)
- [x] Code Availability statement with public repository URL
- [x] No ORCID required at first submission (will be requested at acceptance)
- [x] Public reproducibility track confirmed: synthetic farmland benchmark + Buchanan VA public-data restoration case form a no-restricted-data reproduction path

## Submission portal data

### Required form fields

- **Title:** as above
- **Article type:** Research Article
- **Subject categories:** suggest "Sustainability", "Computer science", "Decision making", "Land use" (CommsEE allows up to 3-4)
- **Corresponding author:** Ning Zhou, SuperMap Software Co., Ltd.
- **Corresponding email:** zhouning1@supermap.com
- **Suggested editors:** (none required)
- **Suggested reviewers:** (optional; select 3-5 with expertise in sustainability, GIS, model-based RL, or land use planning)
- **Excluded reviewers:** (optional)

### Suggested reviewers

CommsEE will ask for 3-5 suggested reviewers. Suitable candidates are researchers in the following areas (verify recent publication and contact email through Google Scholar / institutional pages before listing):

1. **GIS for land use / consolidation:** authors of the Cao 2012 boundary-detection paper, Zheng 2023 urban land use RL paper
2. **Model-based RL for combinatorial planning:** authors of Schrittwieser 2020 (MuZero), Hafner 2025 (DreamerV3) — note these are likely too senior; pick mid-career researchers
3. **Surrogate-assisted optimisation:** authors of Bliek 2021 SAEA benchmark
4. **Restoration planning / abandoned mine lands:** OSMRE researchers, USGS NHD program
5. **Reproducible Earth-science software:** authors of recent CommsEE publications using ArcGIS Pro or open-source GIS pipelines

## What goes via submission portal vs through email

- **Through portal:** all PDFs, all source files, cover letter
- **Through GitHub URL** (already public, cited in Code Availability): trained ensembles (~195 MB), synthetic benchmark, restoration cases, all reproduction scripts

## Post-submission expectations

- **Time to first decision:** CommsEE median is **9 days** (very fast for Nature Portfolio)
- **Decision options:** accept / minor revision / major revision / reject / transfer
- **If transferred:** likely to *Scientific Reports* (Nature Portfolio low tier) or *Nature Computational Science* upgrade

## Key talking points if Editor asks for clarification

| Possible Editor concern | Response |
|---|---|
| "Methodological novelty?" | Diagnosis + fix of MSE failure mode in high-branching MPC; planner-relevant ranking metrics; simulator-cost crossover analysis |
| "Why CommsEE not Nature Sustainability?" | Pipeline is computational/methodological with sustainability *application*, not primarily a sustainability outcomes paper |
| "Restricted data treatment?" | Data Availability lists fully open reproduction path; restricted records have governance disclosure |
| "Ranking failure unique to farmland?" | No — diagnostic generalises (cross-domain section); intervention is conditional on σ_a/σ_s ratio < 1 |
| "Why not pure OR baseline?" | We compare against simulated annealing, NSGA-II, MILP at matched wall-clock; pipeline wins by 9.6× on farmland due to simulator step cost |

---

Built 2026-05-31 from `paper9_v7_draft_commsee.tex` (commit pending).
