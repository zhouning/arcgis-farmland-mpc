# Paper 9 Neijiang Constraint Frontier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a Neijiang execution-constraint frontier for Paper 9 Scientific Reports robustness evidence.

**Architecture:** Add a small Neijiang frontier runner that mirrors the existing Bishan execution-frontier script while using Neijiang prepared data and ONNX ensembles. Keep manuscript edits separate until the experiment is complete and validated.

**Tech Stack:** Python, pytest, GeoPandas, matplotlib, existing `farmland_mpc.mpc_plan`, existing GIS audit scripts.

---

### Task 1: Add Script-Level Unit Tests

**Files:**
- Create: `farmland_mpc/tests/test_neijiang_constraint_frontier_runner.py`
- Create: `scripts/pareto_sweep_neijiang_constraints.py`

- [ ] Write tests that import the Neijiang runner and assert the defaults point to `runs/scirep_extra/prepared_neijiang`, `runs/scirep_extra/onnx/neijiang/ensemble_seed0`, and `DLTB_with_slope.gpkg`.
- [ ] Write tests that call the markdown writer with a representative row and assert the title says `Neijiang Dongxing execution-constraint frontier`.
- [ ] Run the test and confirm it fails because the script does not exist.
- [ ] Add the minimal script API and markdown writer to pass the tests.
- [ ] Run the test again and confirm it passes.

### Task 2: Implement Neijiang Frontier Runner

**Files:**
- Modify: `scripts/pareto_sweep_neijiang_constraints.py`

- [ ] Add the same seven execution profiles used by the Bishan frontier.
- [ ] Implement Tool 4 execution with `farmland_mpc.mpc_plan.run`.
- [ ] Implement generic policy audit via `scripts/policy_translation_optimized.py`.
- [ ] Implement GIS validation via `verification/validate_optimized_shp.py`.
- [ ] Implement JSON, markdown, and figure writers.
- [ ] Add reuse behavior for existing per-profile outputs.

### Task 3: Run Smoke Profile

**Command:**
`python scripts/pareto_sweep_neijiang_constraints.py --only no_net_loss --max-steps 5 --scratch-dir runs/neijiang/pareto/smoke_constraints --out-json runs/neijiang/pareto/smoke_constraints.json --out-md runs/neijiang/pareto/smoke_constraints.md --out-fig-pdf runs/neijiang/pareto/smoke_constraints.pdf --out-fig-png runs/neijiang/pareto/smoke_constraints.png`

- [ ] Confirm the runner builds the Neijiang environment.
- [ ] Confirm it writes Tool 4, policy, validation, JSON, markdown, and figure outputs.

### Task 4: Run Full Frontier

**Command:**
`D:\test\envs\farmland-mpc-pure\python.exe scripts\pareto_sweep_neijiang_constraints.py --mpc-batch-size 256`

- [ ] Run all seven profiles unless runtime or validation shows an issue.
- [ ] Monitor progress logs and process status.
- [ ] Verify final JSON contains all completed profiles.

### Task 5: Decide Manuscript Integration

**Files:**
- Candidate modify: `paper/submission_scirep_corrected/05_source_editable/supplementary_information_scirep.tex`
- Candidate copy: `paper/submission_scirep_corrected/04_figures/neijiang_constraint_frontier.{pdf,png}`
- Candidate copy: `paper/submission_scirep_corrected/05_source_editable/figures/neijiang_constraint_frontier.pdf`

- [ ] Inspect the final frontier table for coherent monotone policy trade-offs.
- [ ] If useful, add a concise SI paragraph and table/figure reference.
- [ ] Recompile SI and check LaTeX logs.
- [ ] Do not alter the main manuscript unless the result materially changes the claims.
