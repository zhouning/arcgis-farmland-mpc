# Paper 9 Neijiang Constraint Frontier Design

## Goal

Add a Neijiang execution-constraint frontier to strengthen the Scientific Reports submission against reviewer concerns that the policy-floor trade-off was only demonstrated on Bishan.

## Scope

The experiment reuses the existing Neijiang prepared package and exported five-seed ONNX ensembles under `runs/scirep_extra`. It sweeps Tool 4 execution constraints only: cultivated-area floors, a baimu-area no-loss floor, and a connectivity-conservative profile. It does not retrain Tool 2 or Tool 3.

## Outputs

- `runs/neijiang/pareto/constraints/<profile>/mpc_summary.json`
- `runs/neijiang/pareto/constraints/<profile>/optimized.shp`
- `runs/neijiang/pareto/constraints/<profile>/policy_translation.json`
- `runs/neijiang/pareto/constraints/<profile>/validate_report.json`
- `paper/submission_scirep_corrected/neijiang_constraint_frontier.json`
- `paper/submission_scirep_corrected/neijiang_constraint_frontier.md`
- `paper/submission_scirep_corrected/04_figures/neijiang_constraint_frontier.{pdf,png}`

## Method

Create a Neijiang-specific frontier runner that mirrors `scripts/pareto_sweep_bishan_constraints.py` but parameterizes the region label, prepared directory, ensemble directory, slope source path, and output locations. The runner uses `DLTB_with_slope.gpkg` as the Tool 4 input because the current Neijiang prepared directory does not contain a shapefile copy.

The sweep will first run with seed-0 ONNX ensemble from `runs/scirep_extra/onnx/neijiang/ensemble_seed0`, matching the canonical constrained-audit role. If this produces a coherent frontier, the result can be written into the SI as a cross-county execution-frontier robustness check.

## Success Criteria

- At least the unconstrained, no-net-loss, and one stricter profile complete.
- Each completed profile has Tool 4 summary, optimized shapefile, policy translation, and GIS validation output.
- The aggregate markdown table reports slope, contiguity, steep-tail area, baimu count/area, cultivated-area change, reward, swaps, and runtime.
- The figure clearly marks profiles that pass or violate cultivated-area no-net-loss.
- No manuscript text is changed until experiment outputs are verified.
