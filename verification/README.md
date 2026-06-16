# Verification scripts

These scripts back the verification stack disclosed in the Scientific
Reports manuscript (`paper/submission_scirep_corrected/05_source_editable/
manuscript_scirep.tex`, Methods sections `sec:methods-validate`,
`sec:methods-mae`, and `sec:methods-trueenv`).

Each script takes a `prepared_dir` produced by Tool 1 (and, where
applicable, an ensemble produced by Tool 3) and runs an independent
audit of the headline MPC result. They share no source with the
optimisation pipeline beyond importing `farmland_mpc.blocks_env.make_env`
to reload the same environment the planner used.

| Script | Verification layer | Wall time | What it answers |
|---|---|---|---|
| [`validate_optimized_shp.py`](validate_optimized_shp.py) | (i) Physical | <2 min | Does the on-disk `optimized.shp` actually match the slope/cont/baimu deltas in `mpc_summary.json`? (Recomputes from raw geometry; no learned model.) |
| [`mpc_member_subsample.py`](mpc_member_subsample.py) | (ii) Variance | ~1 hour | What is a real cross-replicate variance estimator, given that the canonical 5-seed sweep is deterministic-by-construction? (Enumerates all 2-of-3 ensemble subsets.) |
| [`ensemble_1step_mae.py`](ensemble_1step_mae.py) + [`mae_aggregate.py`](mae_aggregate.py) | (iii) Dynamics | ~25 min | How accurate is the ensemble's 1-step prediction on the same trajectory MPC visits? (Compares ensemble output against true `env.step` along the deployed trajectory.) |
| [`mpc_true_env.py`](mpc_true_env.py) | (iv) Counter-factual | ~6 min | What happens if you replace the ensemble entirely with the perfect dynamics model? (Snapshot/restore on env mutable state, true-env scoring under matched H/K/γ.) |

A fifth verification layer (cross-county replication on Neijiang) is
exercised by re-running the standard pipeline (Tool 1→4) on a second
county; no separate script is needed. See `docs/USER_GUIDE.md`.

## Prerequisites

All scripts run inside the same conda environment as the rest of the
pipeline. From the repo root:

```bash
conda activate farmland-mpc
```

You will need:

- A `prepared_dir` produced by `farmland-mpc prepare ...` (Tool 1 output).
- For (ii)/(iii): an ONNX ensemble produced by `farmland-mpc train ...`
  (Tool 3 output, typically `<prepared_dir>/tool3/`).
- For (i): the `optimized.shp` and `mpc_summary.json` produced by
  `farmland-mpc plan ...` (Tool 4 output).

The verification scripts have no extra dependencies beyond what
`environment.yml` already installs.

## Reproducing the paper's verification stack on your data

After running Tool 1→4 on your own county:

```bash
PREPARED=$HOME/farmland_mpc_runs/$REGION/prepared
OUT=$HOME/farmland_mpc_runs/$REGION/mpc_output

# (i) Physical — does the shapefile match the summary?
python verification/validate_optimized_shp.py \
    --optimized $OUT/optimized.shp \
    --slope-shp $PREPARED/dem_slope_analysis/output/DLTB_with_slope.shp \
    --summary   $OUT/mpc_summary.json \
    --proj-crs  EPSG:32648 \
    --out       $OUT/validate_report.json

# (ii) Variance — n=3 ensemble subset replicates
python verification/mpc_member_subsample.py \
    --prepared $PREPARED \
    --ensemble $PREPARED/tool3 \
    --proj-crs EPSG:32648 \
    --n-episodes 3 --n-keep 2 \
    --horizon 5 --top-k 50 --gamma 0.99 \
    --seed-offset 100 \
    --out-dir $OUT/member_subsample_run

# (iii) Dynamics — 1-step prediction MAE along MPC trajectory
python verification/ensemble_1step_mae.py \
    --prepared $PREPARED \
    --ensemble $PREPARED/tool3 \
    --proj-crs EPSG:32648 \
    --n-steps 100 --seed 0 \
    --out-dir $OUT/mae_run

python verification/mae_aggregate.py \
    --per-step $OUT/mae_run/mae_per_step.json

# (iv) Counter-factual — replace ensemble with true env.step
python verification/mpc_true_env.py \
    --prepared $PREPARED \
    --proj-crs EPSG:32648 \
    --n-steps 100 --horizon 5 --top-k 50 --gamma 0.99 \
    --stage1-sample 200 --seed 0 \
    --out-dir $OUT/true_env_run
```

## Expected behaviour

For a working pipeline, each layer should produce these signals:

- **(i)** `validate_report.json` shows |Δ slope|, |Δ cont|, |Δ baimu_ha| at
  floating-point reduction noise (≪ 0.01 pp / ≪ 1e-3 / ≪ 0.01 ha). Exit
  code 0 = pass.
- **(ii)** `summary.json["aggregate"]["slope_pct_std"]` should be a small
  positive number (typically 0.01–0.05 pp on a well-trained ensemble).
  The full-ensemble result from `mpc_summary.json` should sit inside the
  subset mean ± std.
- **(iii)** `mae_summary.json` reports `reward_1step.spearman_pred_true`;
  positive Spearman is enough at H=5 to reproduce the trajectory. The
  reward MAE itself can be large — what matters for planning is
  ranking, not absolute regression accuracy.
- **(iv)** `true_env_summary.json["final"]["slope_change_pct"]` will
  typically be slightly worse than the ensemble pipeline (because true-env
  stage-1 must subsample, while the ensemble can score all valid actions
  in one batch). This is the expected sign — the ensemble's contribution
  is structural depth/breadth at near-constant cost, not per-action
  statistical accuracy.

## Cross-platform note

These scripts are pure Python (NumPy + GeoPandas + libpysal +
onnxruntime) and have been validated on Windows 11 and macOS Apple
Silicon (see [`docs/MACOS.md`](../docs/MACOS.md) for the macOS run).
Floating-point reduction order can produce slope deltas at 1e-3 pp
between platforms; the verification layers are designed with
tolerances that absorb this noise.

## Troubleshooting

- `numpy.corrcoef` segfaults on some macOS / Windows numpy 2.4.x
  builds. The aggregator uses a hand-rolled Pearson; if you see
  segfaults inside `ensemble_1step_mae.py` itself, the aggregate
  step has been moved to `mae_aggregate.py` for this reason.
- `OMP: Error #15` on macOS: `export KMP_DUPLICATE_LIB_OK=TRUE` (also
  flagged in `docs/MACOS.md`).
- Long output paths with non-ASCII characters: the scripts pass UTF-8
  through everywhere, but make sure `PYTHONIOENCODING=utf-8` is set
  if your shell is not UTF-8 by default.
