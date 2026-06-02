# Codex pre-submission review for Communications Earth & Environment

Date: 2026-06-02

Target manuscript: `paper/submission_commsee/source_main_codex.tex`

Journal target: *Communications Earth & Environment*

Purpose: record the remaining submission-readiness issues after the 2026-06-02 Codex review, so the next round of experiments can be run from the macOS workstation after syncing the GitHub repository.

## Bottom line

Do not submit the current `source_main_codex.tex` as a "meets Communications Earth & Environment requirements" manuscript yet.

The manuscript is much more defensible than the previous version because it now frames the Bishan result as a constraint-audited candidate scenario rather than a deployment-ready or red-line-compliant plan. However, two scientific issues remain likely reviewer blockers:

1. The headline Bishan candidate loses 505 ha of cultivated area.
2. Neijiang is used as a second real-county replication, but it does not yet receive the same independent policy audit as Bishan.

The manuscript can probably be positioned as an auditable scenario-generation tool, but it needs a constrained-planning result before it can credibly claim to solve county-scale farmland-consolidation planning under policy constraints.

## Official policy / format basis checked

I checked the relevant Nature/Communications pages during the review:

- Communications Earth & Environment Guide to Authors: <https://www.nature.com/commsenv/submit/guide-to-authors>
- Communications Earth & Environment content types: <https://www.nature.com/commsenv/submit/content-types>
- Nature Portfolio reporting standards / availability expectations: <https://www.nature.com/nature-portfolio/editorial-policies/reporting-standards>

Key practical implications for this manuscript:

- Article-format length is acceptable in the current draft: main text is approximately 4,153 words excluding Methods and figure legends.
- Main display items are acceptable: 4 figures + 2 tables = 6 display items.
- The abstract is too long for the usual Communications-journal Article target: approximately 239 words. It should be compressed to about 150 words if following the final accepted format strictly.
- Data Availability and Code Availability must be specific and durable. A GitHub URL alone is weaker than a frozen release with a DOI, commit hash, license, and data package manifest.

## Blocking issues before submission

### 1. Bishan result violates cultivated-area preservation

Current manuscript locations:

- `source_main_codex.tex`, line 158: the independent audit reports a net cultivated-area decrease of 505 ha, equal to 1.05% of the initial farmland stock.
- `source_main_codex.tex`, line 320: the Discussion correctly states that this should be treated as a rejected or constraint-repair scenario, not a ready implementation plan.
- `source_main_codex.tex`, line 326: the Bounds paragraph correctly identifies this as the strongest limitation.
- `source_main_codex.tex`, line 344 and line 462: Methods/toolchain caveats state that cumulative cultivated-area preservation is not enforced.

Reviewer risk:

CEE reviewers are likely to read the headline application result as follows: the method can quickly find a high-slope-retirement candidate, but the candidate fails a central policy constraint. This does not invalidate the algorithmic contribution, but it weakens the Earth/environmental planning claim.

Required fix:

Run at least one cultivated-area-constrained Bishan experiment and report it next to the unconstrained candidate. The minimum acceptable experiment is:

- Constraint: net cultivated-area change >= 0 ha, or a clearly justified tolerance such as >= -0.1% if exact equality is infeasible.
- Report: slope change, contiguity change, baimu count change, baimu area change, total cultivated-area change, number of swaps, runtime.
- Figure/table: one compact constrained-vs-unconstrained table is enough for the first revision.

If a hard action-mask constraint is not implemented yet, use a conservative post-planning rejection or area-matched swap version as an interim result, but state exactly how the constraint is enforced.

### 2. Neijiang lacks a matched independent policy audit

Current manuscript location:

- `source_main_codex.tex`, line 164: Neijiang is reported as a five-seed real-county replication with slope decrease, contiguity increase, and positive baimu-area change.
- `source_main_codex.tex`, line 168: the manuscript states that red-line compliance must be audited separately for every deployment case.

Reviewer risk:

The manuscript claims two real Chinese counties as the empirical anchor, but the full GIS-only/policy-audit stack is concentrated on Bishan. If Neijiang is kept as a major real-county replication, reviewers may ask whether it also preserves total cultivated area and whether its township-level impacts are spatially fair.

Required fix:

Run the Neijiang analogue of the Bishan policy-translation audit:

- slope-band shifts: <6 deg, 6-15 deg, 15-25 deg, >25 deg;
- net cultivated-area change;
- baimu count and baimu area deltas;
- township-level slope-change distribution and fairness summary;
- farm->forest and forest->farm area totals.

If the raw Neijiang optimized shapefile is available, clone the logic of `scripts/policy_translation_bishan.py` into a Neijiang-specific script or parameterise the existing script.

## Strongly recommended before submission

### 3. Bishan reward-weight / constraint Pareto sweep

Current manuscript location:

- `source_main_codex.tex`, line 326 already admits that a reward-weight Pareto sweep would help practitioners decide how to trade slope against area preservation.
- `scripts/sensitivity_sweep_bishan.py` currently covers H/K sensitivity and only a limited lambda point on the package-side prepare. It does not yet cover reward weights or cultivated-area-constrained variants.

Recommended experiment:

Run 5-7 Bishan configurations spanning slope-heavy to area-preserving behaviour. At minimum:

- default unconstrained;
- cultivated-area hard constrained;
- baimu-area-preserving;
- lower slope weight / higher area weight;
- higher contiguity weight;
- one conservative policy profile with both cultivated-area and baimu-area safeguards.

For each cell, report:

- slope change;
- contiguity change;
- baimu count change;
- baimu area change;
- cultivated-area change;
- reward under the profile;
- runtime.

The ideal output is a small Pareto table plus a slope-vs-cultivated-area plot in Supplementary Information.

### 4. Freeze data/code release with persistent identifiers

Current manuscript location:

- `source_main_codex.tex`, lines 490-502.

Current statement is specific, but it relies on GitHub at submission. For Nature-family review this is weaker than a citable archived release.

Recommended fix:

- create a GitHub release;
- archive the release to Zenodo/Figshare/OSF and obtain a DOI;
- add the release commit hash;
- include a data manifest listing which files reproduce each main figure/table;
- clarify licences separately for code, synthetic benchmark, public restoration data, and restricted cadastral derivatives;
- avoid implying that raw Third National Land Survey cadastral records are available from the authors.

### 5. Compress abstract and make the first sentence more journal-facing

Current manuscript location:

- `source_main_codex.tex`, lines 71-74.

Approximate current abstract length: 239 words.

Recommended target: about 150 words.

The compressed abstract should keep:

- problem scale;
- method;
- Bishan unconstrained audit result and 505 ha constraint failure;
- matched-cadastre method comparison;
- public/open reproduction track.

It should remove secondary details such as every runtime and every baseline ratio unless essential.

### 6. Align final formatting with Nature/Communications style

Current issues:

- `\bibliographystyle{plainnat}` is used rather than Nature-style numbered references.
- The main body uses multiple result-like section titles but no explicit `Results` section heading.
- `Funding` and `ORCID` appear as standalone manuscript sections. They may be better handled through the submission system or journal-standard end-matter format.

These are not as serious as the scientific issues above, but they should be cleaned before final submission or accepted-format revision.

## What is currently strong

The manuscript now has several strengths that should be preserved:

- It no longer overclaims Bishan as a red-line-compliant plan.
- It clearly separates the package-side prepare from the lab pipeline.
- It provides an independent GIS-only recomputation for Bishan.
- It discloses model miscalibration and explains why ranking still supports MPC action selection.
- It includes cross-domain restoration evidence showing where the learned-surrogate pipeline is not superior to classical OR baselines.
- It has a concrete Data Availability / Code Availability structure, even though the release still needs a persistent DOI.

## Minimal macOS work plan

After syncing this repository on the macOS machine, run the following work in order.

1. Reproduce the current Bishan package-side result from the repository instructions and confirm the existing `-2.001%` GIS-audit anchor.

2. Implement or run a cultivated-area-constrained Bishan planning variant:

   - preferred: constrain candidate swaps inside Tool 4 / `farmland_mpc/mpc_plan.py` or the action mask;
   - acceptable first pass: post-planning rejection/repair, if clearly labelled;
   - output JSON should include cultivated-area delta in hectares and percent.

3. Run a Bishan constraint / reward-weight Pareto sweep:

   - extend `scripts/sensitivity_sweep_bishan.py`, or create a new `scripts/pareto_sweep_bishan_constraints.py`;
   - save results to `paper/submission_commsee/pareto_sweep_bishan_constraints.json`;
   - generate a supplementary table and, if useful, a small Pareto figure.

4. Run Neijiang policy audit:

   - parameterise `scripts/policy_translation_bishan.py` or create `scripts/policy_translation_neijiang.py`;
   - save results to `paper/submission_commsee/policy_translation_neijiang.json`;
   - add a Supplementary table analogous to the Bishan audit tables.

5. Revise `source_main_codex.tex`:

   - add constrained Bishan result beside the unconstrained candidate;
   - add one sentence/table for Neijiang cultivated-area compliance;
   - update Discussion so the limitation becomes "unconstrained mode can violate policy; constrained mode retains X of the benefit" rather than "future work must add constraints";
   - compress the abstract to about 150 words;
   - update Data/Code Availability with DOI/commit hash when the release is frozen.

## Recommended decision

The manuscript should remain in revision mode until the constrained Bishan experiment is complete. If compute time is limited, prioritise the cultivated-area-constrained Bishan run over all other experiments. That one result will determine whether the paper reads as a planning solution with an audit loop or only as an audit of an infeasible optimisation candidate.
