# DOI Update Record

Zenodo minted the cleaned release DOI `10.5281/zenodo.23163184` for `v1.0.2-scirep-revision`. These locations were updated before final submission.

## 1. `CITATION.cff`

The software DOI is recorded near the release metadata:

```yaml
doi: "10.5281/zenodo.23163184"
```

If a separate dataset DOI is created, do not put the dataset DOI as the software DOI. Mention it in the Data Availability statement instead.

## 2. Main manuscript: Code availability

The manuscript Code Availability sentence now cites:

```text
All code for the contrastive training, MPC planner, synthetic generators, ArcGIS Pro toolchain, the non-commercial CLI workflow, the operations-research baselines, the planner-relevant ranking metrics, the restoration Pareto-sweep driver, and the Bishan and Neijiang execution-constraint frontier drivers is archived on Zenodo at https://doi.org/10.5281/zenodo.23163184 and maintained at https://github.com/zhouning/arcgis-farmland-mpc, under an MIT licence, with step-by-step reproduction instructions in docs/REPRODUCE.md and a smoke test that completes a full Tool 1-4 cycle in approximately 70 seconds on a small subset.
```

Keep the LaTeX-specific math and section references when applying this sentence inside `manuscript_scirep.tex`.

## 3. Main manuscript: Data availability

The Data Availability paragraph cites the archived release because the deposited record includes the shareable derived data products:

```text
... all derived data products required to reproduce the reported experiments are publicly available now (not gated on publication) in the archived project release https://doi.org/10.5281/zenodo.23163184 and maintained at the project repository (Code Availability) under the stated licences:
```

If a separate dataset DOI exists, use:

```text
... all derived data products required to reproduce the reported experiments are publicly available now (not gated on publication) in the separate Zenodo dataset record DOI and the archived project release https://doi.org/10.5281/zenodo.23163184 under the stated licences:
```

## 4. Declarations file

Mirror the DOI wording in:

`paper/submission_scirep_corrected/06_declarations_and_checks/declarations_scirep.md`

## 5. Compile and verify

Run:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error manuscript_scirep.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript_scirep.tex
```

Then copy `05_source_editable/manuscript_scirep.pdf` to `01_main_document/01_main_manuscript_scirep.pdf`, check the LaTeX log for undefined references/citations and compare SHA256 hashes.

## 6. Commit the DOI update

The DOI was minted by Zenodo before it was added here. After recompilation and verification, commit and push this DOI backfill as a separate post-Zenodo commit.


