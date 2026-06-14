# Paper 9 Scientific Reports Handoff

Saved at: 2026-06-12 20:44:50 +08:00

## 2026-06-15 Neijiang frontier completion and submission sync

Current package path:

- `D:\test\_publish\arcgis-farmland-mpc\paper\submission_scirep_corrected`

Completed stronger experiment:

- Added and ran a Neijiang Dongxing seven-profile execution-constraint frontier using the current Scientific Reports prepared package and seed-0 ONNX ensemble.
- Summary outputs:
  - `paper/submission_scirep_corrected/neijiang_constraint_frontier.json`
  - `paper/submission_scirep_corrected/neijiang_constraint_frontier.md`
  - `paper/submission_scirep_corrected/04_figures/neijiang_constraint_frontier.pdf`
  - `paper/submission_scirep_corrected/04_figures/neijiang_constraint_frontier.png`
  - `paper/submission_scirep_corrected/05_source_editable/figures/neijiang_constraint_frontier.pdf`
- Full local per-profile GIS outputs are under `runs/neijiang/pareto/`; they are about 4 GB and are ignored by Git as reproducible run artefacts.

Manuscript/package integration:

- SI now includes the Neijiang frontier paragraph, Table S18, and Figure S2.
- Main manuscript adds one bounded sentence in the Neijiang deployment section and updates methods/code-availability references to both Bishan and Neijiang frontier drivers.
- Cover letter notes that the revised SI includes seven-profile execution frontiers for both counties.
- `\clearpage` guards keep the Bishan and Neijiang frontier floats in their own sequence before SI S12.
- The Neijiang frontier plot labels were shortened/staggered for the three overlapping cultivated-floor points.

Compilation and validation completed:

- `pdflatex -interaction=nonstopmode -halt-on-error supplementary_information_scirep.tex` run twice; final SI is 18 pages.
- `pdflatex -interaction=nonstopmode -halt-on-error manuscript_scirep.tex` run twice; final main manuscript is 27 pages.
- `pdflatex -interaction=nonstopmode -halt-on-error cover_letter_scirep.tex` run once; final cover letter is 1 page.
- Log screen found no undefined references/citations, missing files, overfull boxes, float-too-large warnings, or fatal LaTeX errors.
- Rendered SI pages 13--16 were visually checked; the Neijiang table and figure are present and readable.
- Synced upload PDFs:
  - `01_main_document/01_main_manuscript_scirep.pdf`
  - `02_cover_letter/00_cover_letter_scirep.pdf`
  - `03_supplementary_information/02_supplementary_information_scirep.pdf`
- Tests:
  `python -m pytest farmland_mpc\tests\test_neijiang_constraint_frontier_runner.py -q -p no:cacheprovider --basetemp .\tmp_pytest_neijiang_frontier_final2`
  Result: `7 passed`.
- `git diff --check` passed.
- No `python.exe` or `py.exe` process was running after checks.

Do not auto-commit unless requested. Before committing, review whether to include `docs/superpowers/` planning files; the core submission evidence does not depend on them.

## 2026-06-14 reviewer-risk revision after strict Scientific Reports-style review

Performed in response to a strict reviewer-style critique focused on technical validity, audit symmetry, policy realism, and readability.

Files edited:

- `D:\test\ScientificReports_submission_paper9_corrected\05_source_editable\manuscript_scirep.tex`
- `D:\test\ScientificReports_submission_paper9_corrected\05_source_editable\supplementary_information_scirep.tex`
- `D:\test\ScientificReports_submission_paper9_corrected\05_source_editable\cover_letter_scirep.tex`

Substantive manuscript changes:

- Added the already-existing Neijiang canonical no-net-loss GIS audit evidence into the main manuscript.
  - Source evidence: `D:\test\_publish\arcgis-farmland-mpc\paper\submission_commsee\neijiang_cultivated_area_constraint.json`
  - GIS audit pass: `overall_pass=true`
  - Neijiang constrained recomputation:
    - slope `-0.4918266%` vs planner `-0.4918312%`
    - contiguity `+0.0373018` vs planner `+0.0373027`
    - baimu area `+252.56 ha`
    - cultivated area `+62.24 ha`
    - qualifying patch count change `0`
  - Policy-band audit: `330.4 ha` less farmland in `>=15°` slope bands.
- Clarified evidence roles:
  - package-side GIS anchors verify on-disk canonical geometry;
  - lab pipeline supports matched DRL comparison;
  - CLI/toolbox replications test released workflow;
  - five-ensemble and reward-weight sweeps remain CountyLevelEnv-side estimates and do not replace canonical GIS-only recomputation.
- Removed an over-broad Introduction phrase implying tenure constraints are modelled; replaced it with administrative/cultivated-area controls.
- Strengthened Discussion limitations:
  - current state does not encode parcel ownership, soil quality, irrigation access, ecological red lines, construction/transaction costs, or stakeholder consent;
  - reward weights are transparent objective coefficients, not calibrated welfare, cost-benefit, or statutory preference weights;
  - slope percentage gains are interpreted with area equivalents for Bishan and Neijiang constrained audits.
- Updated SI verification wording so it no longer frames Neijiang only as cross-county replication.
- Updated cover letter to mention canonical Bishan and Neijiang GIS recomputation and Neijiang steep-band area reduction.

Compilation and package sync after this revision:

- Main manuscript compiled twice:
  `pdflatex -interaction=nonstopmode -halt-on-error manuscript_scirep.tex`
  - Result: `manuscript_scirep.pdf`, 27 pages, `935631 bytes`.
- Supplementary information compiled twice:
  `pdflatex -interaction=nonstopmode -halt-on-error supplementary_information_scirep.tex`
  - Result: `supplementary_information_scirep.pdf`, 15 pages, `299117 bytes`.
- Cover letter compiled once:
  `pdflatex -interaction=nonstopmode -halt-on-error cover_letter_scirep.tex`
  - Result: `cover_letter_scirep.pdf`, 1 page, `63531 bytes`.
- Log check:
  `rg -n "LaTeX Warning:|Package natbib Warning|undefined|Rerun to get|Fatal|LaTeX Error|Emergency stop|Overfull" manuscript_scirep.log supplementary_information_scirep.log cover_letter_scirep.log`
  - Result: no matches.
- Ordinary `Underfull \hbox` diagnostics remain; no `Overfull`, undefined references, rerun warnings, natbib warnings, or fatal/errors.

Synced upload PDFs:

- Main:
  `D:\test\ScientificReports_submission_paper9_corrected\01_main_document\01_main_manuscript_scirep.pdf`
  - SHA256: `0A32AD7F2CBA59790B874E8C77ABFABEBE37C2317F14757296578EFAE7643B45`
- Supplementary:
  `D:\test\ScientificReports_submission_paper9_corrected\03_supplementary_information\02_supplementary_information_scirep.pdf`
  - SHA256: `BD9F70D2918F11E4DAE4B2EB5888EC196DED40E1E39F5EE512BB086780DF8603`
- Cover letter:
  `D:\test\ScientificReports_submission_paper9_corrected\02_cover_letter\00_cover_letter_scirep.pdf`
  - SHA256: `B2810745AE2598E18589FF6BA46F29EB2B17F100E38EA50238E02610E1429E38`

No new training or planning experiment was run in this revision. The Neijiang GIS audit was not fabricated; it was recovered from the existing validated JSON in the earlier submission workspace and incorporated with explicit boundary language.

## Critical paths

- Correct original CEE manuscript:
  `D:\test\_publish\arcgis-farmland-mpc\paper\submission_commsee\source_main_codex.tex`
- Correct Scientific Reports package:
  `D:\test\ScientificReports_submission_paper9_corrected`
- Current manuscript source:
  `D:\test\ScientificReports_submission_paper9_corrected\05_source_editable\manuscript_scirep.tex`
- Current compiled main PDF:
  `D:\test\ScientificReports_submission_paper9_corrected\01_main_document\01_main_manuscript_scirep.pdf`
- Do not continue from the old mistaken package:
  `D:\test\ScientificReports_submission_paper9`

## Current scientific status

The manuscript has been revised using real additional experiments, not fabricated results.

Completed added experiments:

- Bishan retrained reward-weight sensitivity.
- Output directory:
  `D:\test\_publish\arcgis-farmland-mpc\runs\scirep_reward_sensitivity\bishan`
- Summary files:
  `reward_weight_sensitivity.json`
  `reward_weight_sensitivity.md`
- Neijiang retrained reward-weight sensitivity.
- Output directory:
  `D:\test\_publish\arcgis-farmland-mpc\runs\scirep_reward_sensitivity\neijiang`
- Summary files:
  `reward_weight_sensitivity.json`
  `reward_weight_sensitivity.md`

The sensitivity experiment reran the full chain for each reward profile:

1. Tool 2 sampling.
2. Tool 3 three-member contrastive ensemble training.
3. Tool 4 no-net-loss MPC planning.

Final sensitivity settings:

- `horizon=5`
- `top_k=50`
- `gamma=0.99`
- `mpc_batch_size=256`
- `continuation=greedy`
- `scoring=reward`
- `cultivated_area_floor_delta_ha=0.0`

Final Bishan sensitivity results written into the manuscript:

| Profile | Slope delta | Contiguity delta | Baimu count delta | Baimu area delta | Cultivated area delta | Rank acc mean | Reward std median |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default reward | `-0.7247%` | `+0.0270` | `+1` | `+27.49 ha` | `+0.44 ha` | `0.9033` | `0.6855` |
| Baimu low | `-0.6522%` | `+0.0209` | `0` | `+17.34 ha` | `+0.18 ha` | `0.9273` | `0.5928` |
| Baimu high | `-0.8379%` | `+0.0343` | `+1` | `+56.21 ha` | `+0.64 ha` | `0.8733` | `0.6843` |

Final Neijiang sensitivity results written into the manuscript:

| Profile | Slope delta | Contiguity delta | Baimu count delta | Baimu area delta | Cultivated area delta | Rank acc mean | Reward std median |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default reward | `-0.4705%` | `+0.0278` | `+9` | `+188.60 ha` | `+0.64 ha` | `0.8656` | `0.2193` |
| Baimu low | `-0.4496%` | `+0.0290` | `+2` | `+185.80 ha` | `+0.18 ha` | `0.8778` | `0.2181` |
| Baimu high | `-0.4468%` | `+0.0309` | `+5` | `+205.63 ha` | `+30.15 ha` | `0.8536` | `0.2196` |

Interpretation boundary now in manuscript:

- The Bishan and Neijiang sensitivities are evidence against a trivial default-weight artefact.
- It is not a full cross-county Pareto analysis.
- It is two counties, three profiles per county, and one retrained three-member ensemble per county-profile cell.
- The next stronger experiment would be a broader constrained-frontier study across additional counties and policy floors, ideally with multiple retrained ensembles per profile.

## Manuscript edits already made

File:

- `D:\test\ScientificReports_submission_paper9_corrected\05_source_editable\manuscript_scirep.tex`

Important inserted/updated locations:

- Results, line around 164:
  retrained Bishan and Neijiang reward-weight sensitivity paragraph.
- Results, audit boundaries paragraph:
  updated from four checks to five checks.
- Discussion, `Bounds on the claim`:
  added bounded interpretation of the two-county sensitivity experiment.
- Methods, `Retrained reward-weight sensitivity`:
  added profile weights and reproducible parameters.
- Methods training paragraph:
  updated to up to 30 epochs, patience 8, and 25-60 min CPU training time.

The Results sensitivity paragraph was also polished to avoid internal/debugging language. It now says the reward head and planner used the same profile weights.

## Code changes in arcgis-farmland-mpc

Repository:

- `D:\test\_publish\arcgis-farmland-mpc`

Modified tracked files:

- `farmland_mpc/cli.py`
- `farmland_mpc/ensemble_runner.py`
- `farmland_mpc/mpc_plan.py`
- `farmland_mpc/sample.py`

Untracked but important files:

- `farmland_mpc/tests/test_ensemble_runner.py`
- `farmland_mpc/tests/test_sample_reward_overrides.py`
- `farmland_mpc/tests/test_scirep_reward_sensitivity_runner.py`
- `scripts/scirep_reward_weight_sensitivity.py`
- `scripts/scirep_extra_experiments.py`
- `runs/scirep_reward_sensitivity/`

Purpose of code changes:

- Add reward override support to Tool 2 sampling.
- Thread reward overrides through CLI.
- Add `mpc_batch_size` to Tool 4 planning for memory control.
- Add reward-only prediction path for greedy continuation.
- Reduce ONNX/NumPy memory pressure with streamed/batched ensemble prediction.
- Run Tool 4 planning from the sensitivity runner in a fresh subprocess.

## Verification already performed

LaTeX:

- Compiled from:
  `D:\test\ScientificReports_submission_paper9_corrected\05_source_editable`
- Command:
  `pdflatex -interaction=nonstopmode -halt-on-error manuscript_scirep.tex`
- Run twice after final text edits.
- Log check:
  `rg -n "LaTeX Warning:|Package natbib Warning|undefined|Rerun to get|Fatal|LaTeX Error|Emergency stop|Overfull" manuscript_scirep.log`
- Final check returned no matches.
- PDF copied to:
  `D:\test\ScientificReports_submission_paper9_corrected\01_main_document\01_main_manuscript_scirep.pdf`

Python tests:

- Command:
  `python -m pytest farmland_mpc\tests\test_sample_reward_overrides.py farmland_mpc\tests\test_ensemble_runner.py farmland_mpc\tests\test_scirep_reward_sensitivity_runner.py farmland_mpc\tests\test_cultivated_area_floor.py -q -p no:cacheprovider --basetemp .\tmp_pytest_scirep_final_combined_after_tex`
- Result:
  `11 passed in 0.52s`
- Working directory:
  `D:\test\_publish\arcgis-farmland-mpc`
- Python used:
  `C:\Python314\python.exe`

Note:

- `D:\test\envs\farmland-mpc-pure\python.exe` was useful for full experiment execution, but it does not have `pytest` installed.

## 2026-06-13 final package check

Performed at: 2026-06-13 15:27:24 +08:00

Actions completed:

- Recompiled main manuscript from:
  `D:\test\ScientificReports_submission_paper9_corrected\05_source_editable`
- Command:
  `pdflatex -interaction=nonstopmode -halt-on-error manuscript_scirep.tex`
- Result:
  `Output written on manuscript_scirep.pdf (26 pages, 933475 bytes).`
- Copied latest main PDF to:
  `D:\test\ScientificReports_submission_paper9_corrected\01_main_document\01_main_manuscript_scirep.pdf`

Fresh log checks:

- Command:
  `rg -n "LaTeX Warning:|Package natbib Warning|undefined|Rerun to get|Fatal|LaTeX Error|Emergency stop|Overfull" manuscript_scirep.log supplementary_information_scirep.log cover_letter_scirep.log`
- Result:
  no matches.
- Separate `Underfull \hbox` diagnostics remain in the main manuscript and supplementary logs. They are ordinary line-breaking diagnostics around long paragraphs, code identifiers, and dense SI table text; no `Overfull`, undefined citation/reference, rerun, fatal, or LaTeX error diagnostics were found.

PDF hash consistency after the final copy:

- Main source/upload:
  `C8A3185202E6A4D31487723A65D15B2F34CC47497F6C71B973E5CC3A7AEB2807`
- Supplementary source/upload:
  `C368D52D018CBA6503D8CFDB2857A3D684775AE94ACDB84CEBA9DAAB96D478BC`
- Cover letter source/upload:
  `91259E7779F1A9D2EAF806040411741E3BEE3D9D34B35BC74F24A9BC76DD43C2`

PDF text checks:

- Main PDF contains the corrected DOI strings:
  `10.4060/ca5561en`,
  `10.1016/j.jclepro.2023.138962`,
  `10.3390/land12071410`.
- Main, cover-letter, and SI PDF text scans found no placeholder or old wrong-package residues for the checked patterns.
- SI text scan matched only normal prose uses of `wrong` (`wrong direction`, model members being wrong), not package-residue text.

Current submission-readiness interpretation:

- The upload PDFs are synchronized with source PDFs.
- The main bibliography fixes are rendered in the current PDF.
- No Neijiang retrained reward-weight sensitivity run is recommended before submission unless the user wants stronger cross-county sensitivity evidence beyond the bounded Bishan sensitivity probe.

## 2026-06-13 Neijiang sensitivity run paused

Paused at: 2026-06-13 17:26:11 +08:00

User requested training pause. The Neijiang reward-weight sensitivity runner was stopped intentionally.

Stopped process:

- PID:
  `20792`
- Command:
  `D:\test\envs\farmland-mpc-pure\python.exe scripts\scirep_reward_weight_sensitivity.py --region-label Neijiang --prepared-template D:\test\_publish\arcgis-farmland-mpc\runs\scirep_extra\prepared_neijiang --out-root D:\test\_publish\arcgis-farmland-mpc\runs\scirep_reward_sensitivity\neijiang --out-json D:\test\_publish\arcgis-farmland-mpc\runs\scirep_reward_sensitivity\neijiang\reward_weight_sensitivity.json --out-md D:\test\_publish\arcgis-farmland-mpc\runs\scirep_reward_sensitivity\neijiang\reward_weight_sensitivity.md --mpc-batch-size 256`
- No remaining `python.exe` process with `scirep_reward_weight_sensitivity.py` was found after stopping.

Code/test change made before starting the run:

- `scripts/scirep_reward_weight_sensitivity.py` now accepts `--region-label` so Neijiang reports do not say Bishan.
- `farmland_mpc/tests/test_scirep_reward_sensitivity_runner.py` includes a regression test for the region label.
- Tests run:
  `python -m pytest farmland_mpc\tests\test_scirep_reward_sensitivity_runner.py -q -p no:cacheprovider --basetemp .\tmp_pytest_region_label_green`
  result: `2 passed`.
- Related tests run:
  `python -m pytest farmland_mpc\tests\test_sample_reward_overrides.py farmland_mpc\tests\test_ensemble_runner.py farmland_mpc\tests\test_cultivated_area_floor.py -q -p no:cacheprovider --basetemp .\tmp_pytest_related_green`
  result: `10 passed`.

Partial Neijiang outputs:

- Output root:
  `D:\test\_publish\arcgis-farmland-mpc\runs\scirep_reward_sensitivity\neijiang`
- Logs:
  `D:\test\_publish\arcgis-farmland-mpc\runs\scirep_reward_sensitivity\neijiang_stdout.log`
  `D:\test\_publish\arcgis-farmland-mpc\runs\scirep_reward_sensitivity\neijiang_stderr.log`
- Default profile Tool 2 completed:
  `D:\test\_publish\arcgis-farmland-mpc\runs\scirep_reward_sensitivity\neijiang\default\prepared\tool2\sample_transitions_summary.json`
  `transitions.npz` size: about 2.21 GB.
  `pairwise.npz` size: about 184 MB.
- Default profile Tool 2 metrics observed in stdout:
  `6000` transition rows.
  `1000` pairwise states x `50` actions.
  Pairwise reward std median: `0.2193`.
- Default profile Tool 3 partially completed:
  member 0 trained in `1756.6 s`, `best_epoch=24`, `best_val_loss=0.40262`, final cosine similarity `0.9998`, ranking accuracy `0.870`.
  ONNX parity max diff: `9.54e-07`.
  `ensemble_member0.onnx`, `ensemble_member0.onnx.data`, and `ensemble_member0.pt` exist.
- Training was stopped during default profile Tool 3 member 1.
  Last train log lines showed member 1 epoch 10/30:
  `mse=0.14796`, `rank_val=0.04855`, `val=0.42748`, `cos=0.9993`, `rank_acc=0.809`.
- `train_summary.json` does not exist yet, so the default profile ensemble is incomplete.
- No Neijiang `reward_weight_sensitivity.json` or `.md` final report exists yet.

Resume behavior:

- Running the same command without force flags should reuse the completed default-profile Tool 2 summary and data.
- Current runner logic requires a complete `train_summary.json` plus all three ONNX members to reuse Tool 3. Because the stop occurred during member 1, default Tool 3 will restart from scratch unless the runner is enhanced to resume per-member training.
- Do not treat Neijiang sensitivity as manuscript evidence until all three profiles finish Tool 2 -> Tool 3 -> Tool 4 and the final JSON/Markdown report has been verified.

## 2026-06-13 Neijiang sensitivity resumed and running

Status checked at: 2026-06-13 20:58:01 +08:00

User requested continuing training. The Neijiang reward-weight sensitivity runner is running and should not be stopped unless the user asks.

Running processes:

- Main runner PID:
  `5472`
- Tool 4 MPC subprocess PID:
  `23472`
- Main command:
  `D:\test\envs\farmland-mpc-pure\python.exe scripts\scirep_reward_weight_sensitivity.py --region-label Neijiang --prepared-template D:\test\_publish\arcgis-farmland-mpc\runs\scirep_extra\prepared_neijiang --out-root D:\test\_publish\arcgis-farmland-mpc\runs\scirep_reward_sensitivity\neijiang --out-json D:\test\_publish\arcgis-farmland-mpc\runs\scirep_reward_sensitivity\neijiang\reward_weight_sensitivity.json --out-md D:\test\_publish\arcgis-farmland-mpc\runs\scirep_reward_sensitivity\neijiang\reward_weight_sensitivity.md --mpc-batch-size 256`

Current logs:

- `D:\test\_publish\arcgis-farmland-mpc\runs\scirep_reward_sensitivity\neijiang_resume_20260613_191143_stdout.log`
- `D:\test\_publish\arcgis-farmland-mpc\runs\scirep_reward_sensitivity\neijiang_resume_20260613_191143_stderr.log`
- `D:\test\_publish\arcgis-farmland-mpc\runs\scirep_reward_sensitivity\neijiang\default\prepared\tool3\train.log`

Default profile state:

- Tool 2 was reused from the prior partial run.
- Tool 3 completed all three ensemble members.
- `D:\test\_publish\arcgis-farmland-mpc\runs\scirep_reward_sensitivity\neijiang\default\prepared\tool3\train_summary.json` exists.
- ONNX files exist for members 0, 1, and 2.
- Member ranking accuracies:
  `0.8701`, `0.8652`, `0.8614`; mean about `0.8656`.
- ONNX parity max diffs:
  member 0 `9.54e-07`, member 1 `2.38e-07`, member 2 `1.19e-07`.
- Tool 4 no-net-loss MPC planning has started for the default profile.
- Tool 4 loaded all three ONNX members and built the Neijiang county environment:
  `3711` blocks, `76376` swappable parcels, `100` max steps, initial slope `10.5476`, initial contiguity `2.6314`, initial baimu fang `384` patches and `74341.9 ha`.
- At the last check, the MPC subprocess was alive, responsive, and using about `3.7 GB` working set.
- Tool 4 had reached step `10/100` with stdout progress:
  slope `-0.0427%`, contiguity `+0.0017`, baimu area `+12.8 ha`, `mpc_step=56.78s`.
- `mpc_summary.json` had not been written yet.

Next monitoring steps:

- Check whether PIDs `5472` and `23472` are still alive.
- Tail:
  `D:\test\_publish\arcgis-farmland-mpc\runs\scirep_reward_sensitivity\neijiang_resume_20260613_191143_stdout.log`
- Check for:
  `D:\test\_publish\arcgis-farmland-mpc\runs\scirep_reward_sensitivity\neijiang\default\plan_no_net_loss\mpc_summary.json`
- After default Tool 4 finishes, the runner should continue to `baimu_low`, then `baimu_high`, each with Tool 2 -> Tool 3 -> Tool 4.
- Do not treat Neijiang sensitivity as manuscript evidence until all three profiles finish and the final JSON/Markdown are verified.

## 2026-06-14 Neijiang sensitivity completed and manuscript updated

Performed at: 2026-06-14

Neijiang retrained reward-weight sensitivity completed successfully:

- Final JSON:
  `D:\test\_publish\arcgis-farmland-mpc\runs\scirep_reward_sensitivity\neijiang\reward_weight_sensitivity.json`
- Final Markdown:
  `D:\test\_publish\arcgis-farmland-mpc\runs\scirep_reward_sensitivity\neijiang\reward_weight_sensitivity.md`
- The final JSON contains all three profiles:
  `default`, `baimu_low`, `baimu_high`.

Manuscript source updated:

- File:
  `D:\test\ScientificReports_submission_paper9_corrected\05_source_editable\manuscript_scirep.tex`
- Results now reports retrained reward-weight sensitivity in both Bishan and Neijiang.
- Audit boundaries now state that reward-weight sensitivity re-runs sampling and training in both counties.
- Discussion `Bounds on the claim` now treats the sensitivity as two-county evidence against a trivial default-weight artefact, while retaining the boundary that it is not a full Pareto analysis.
- Methods `Retrained reward-weight sensitivity` now states that each Bishan and Neijiang profile copied its county-specific prepared template and re-ran Tool 2, Tool 3, and no-net-loss Tool 4.

Compilation and PDF sync:

- Compiled twice from:
  `D:\test\ScientificReports_submission_paper9_corrected\05_source_editable`
- Command:
  `pdflatex -interaction=nonstopmode -halt-on-error manuscript_scirep.tex`
- Result:
  `Output written on manuscript_scirep.pdf (26 pages, 933491 bytes).`
- Log check:
  `rg -n "LaTeX Warning:|Package natbib Warning|undefined|Rerun to get|Fatal|LaTeX Error|Emergency stop|Overfull" manuscript_scirep.log`
- Result:
  no matches.
- Copied latest main PDF to:
  `D:\test\ScientificReports_submission_paper9_corrected\01_main_document\01_main_manuscript_scirep.pdf`
- Source and upload PDF SHA256 after the final compile/copy:
  `9945FD59FB459B2E78D53ECE08A9C05E652CAD87059438B37A9D336D322ACFDB`

## Recommended next steps

1. Optional: manually read the rendered main PDF, SI PDF, and cover letter once before web upload.
2. Optional: resume Neijiang retrained reward-weight sensitivity for stronger cross-county sensitivity evidence.
3. If resuming, either accept restarting default Tool 3 or first enhance the runner to resume complete per-member ONNX/PT outputs.
4. Consider whether to keep, move, or ignore large untracked `runs/` outputs before any future git commit.
5. Do not commit automatically. User has not requested a commit.

## Resume prompt for next session

Continue from:

`D:\test\ScientificReports_submission_paper9_corrected\NEXT_SESSION_HANDOFF.md`

Goal:

Continue Paper 9 Scientific Reports transfer. Use the corrected package only. Do not use `ScientificReports_submission_paper9`. First read this handoff, then inspect `manuscript_scirep.tex`, the compiled PDF, and the Bishan sensitivity result files before deciding whether to run Neijiang sensitivity or finalize the submission package.
