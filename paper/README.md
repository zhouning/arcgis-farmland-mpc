# Paper 9 — v7 manuscript bundle

Latest submission-ready bundle for the Nature Sustainability candidate paper:
**"Model-based AI planning enables county-scale farmland consolidation in fragmented mountain landscapes."**

## Files

| File | What it is |
|---|---|
| `paper9_v7_draft.tex` / `.pdf` | Main manuscript (v7, 24 pages) |
| `paper9_v7_supplementary.tex` / `.pdf` | Supplementary information (6 pages) |
| `paper9_v7_cover_letter.tex` / `.pdf` | Editor cover letter (2 pages) |
| `references_v6.bib` | Shared BibTeX database |
| `si_tables_v7.tex` | SI tables included by the supplementary |

PDFs in this folder are the most recent local builds (Windows, TeX Live).
On macOS the build command is the same:

```bash
cd paper
pdflatex paper9_v7_draft && bibtex paper9_v7_draft && pdflatex paper9_v7_draft && pdflatex paper9_v7_draft
pdflatex paper9_v7_supplementary && bibtex paper9_v7_supplementary && pdflatex paper9_v7_supplementary && pdflatex paper9_v7_supplementary
pdflatex paper9_v7_cover_letter
```

## Validation status (as of this commit)

- 5-layer verification stack (under `../verification/`) all green on Windows
- macOS Apple Silicon end-to-end run validated for Bishan 53k:
  slope **−2.0392 %** (Windows: −2.0006 %, within contrastive σ)
- Member-subset variance (3 subsets of 2-of-3 ensemble): slope −1.998 ± 0.026 %, baimu −474.5 ± 31.1 ha
- True-env MPC ablation: matched H/K/γ slope −1.819 % (ensemble wins by structure, not tuning)

See the relevant `§sec:methods-*` sections in the draft for the source of each number.
