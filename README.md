# ArcGIS Farmland MPC

An ArcGIS Pro Python Toolbox for farmland-consolidation planning via a
contrastive world-model + Model Predictive Control (MPC). Wraps the
research pipeline of a contrastive learning paper into a four-tool
workflow usable from the ArcGIS Pro UI.

## TL;DR

```
1. Copy this repository to the target machine.
2. In the ArcGIS Python Command Prompt:
     conda install -n arcgispro-py3-clone-<yours> \
         -c conda-forge gymnasium libpysal
3. In ArcGIS Pro: Add Toolbox -> LandUseOptimization_P9.pyt
4. Double-click "5. Check Dependencies" until every line is [OK].
5. Run Tool 1 -> 2 -> 3 -> 4 in order.
```

Tools 1-3 are a one-time setup per region; Tool 4 is the planning loop
you re-run for different parameters.

## Pipeline

| # | Tool | Input | Output | Runtime (county scale) |
|---|---|---|---|---|
| 1 | Prepare Data & Blocks | DLTB.shp + DEM (+ optional XZQ.shp) | `<prepared_dir>/` | 10 - 15 min |
| 2 | Sample Transitions | Tool 1 prepared_dir | `<prepared_dir>/tool2/*.npz` | 15 - 25 min |
| 3 | Train Contrastive Ensemble | Tool 2 npz | `<prepared_dir>/tool3/*.onnx` | 30 - 60 min |
| 4 | **MPC Planning** | Tool 1 prepared + Tool 3 onnx | `optimized_dltb.shp` | ~7 min / episode |
| 5 | Check Dependencies | (none) | diagnostic log | seconds |

## Pitfalls

1. **Swapping regions requires retraining the ensemble.** `n_blocks` is
   baked statically into each ONNX member. An ensemble trained on one
   region will not load against a different region. Tool 4's
   `assert_compatible` check surfaces this early.
2. **The default projected CRS (EPSG:32648, UTM Zone 48N) only covers
   central and western China.** Regions in the east, north-east, or
   north-west must override `proj_crs` on Tool 1; otherwise slope and
   area metrics will be wrong. See the user guide for the zone table.
3. **Tool 1 requires the Spatial Analyst extension.** The Check
   Dependencies tool blocks the pipeline until this is resolved.
4. **The reward-weight overrides on Tool 4 are cosmetic unless you
   retrain the ensemble.** The UI fields change only what `env.step()`
   reports; MPC's candidate ranking uses the reward head learned by
   Tool 3, which was trained under a fixed weight set. To actually
   steer planning, retrain Tools 2 + 3 with the new weights.

## Repository layout

```
arcgis-farmland-mpc/
|-- LandUseOptimization_P9.pyt     # ArcGIS Pro toolbox (load this in Pro)
|-- core/                          # Algorithm modules (Python 3.13)
|   |-- blocks_env.py              # Region-agnostic env factory
|   |-- block_definition.py        # Block segmentation (Paper 3 hybrid)
|   |-- contrastive_trainer.py     # MSE + margin ranking trainer
|   |-- county_env.py              # gym.Env, county-scale MDP
|   |-- ensemble_runner.py         # ONNX runtime wrapper
|   |-- mpc_plan.py                # Tool 4 entry
|   |-- prepare_data.py            # Tool 1 entry
|   |-- sample_transitions.py      # Tool 2 entry
|   |-- shapefile_io.py            # DLTB read/write helpers
|   |-- train_ensemble.py          # Tool 3 entry
|   |-- transition_model.py        # 237K-param transition head
|-- LICENSE
|-- README.md
```

## System requirements

- **ArcGIS Pro 3.6+** with Spatial Analyst extension
- **arcgispro-py3** Python environment (Python 3.13) with torch, onnx,
  onnxruntime, geopandas (all present by default)
- Extra dependencies not in vanilla arcgispro-py3: `gymnasium`,
  `libpysal` (install via `conda install -c conda-forge`, not pip, to
  avoid clobbering numpy/pandas versions).

The Check Dependencies tool (#5) validates all of the above before you
run any pipeline stage.

## Input data schema

Tool 1 expects a standard [Third National Land Survey][tnls] DLTB
shapefile with these fields:

- `BSM` (text, unique patch ID)
- `DLBM` (text, 3-digit land use code, e.g. `011` for paddy field)
- `DLMC` (text, land use name)
- `QSDWDM` (text, ownership unit code; first 9 digits encode township)
- Standard `Shape_Length`, `Shape_Area`

Plus a DEM raster for slope derivation. An optional XZQ (administrative
boundary) shapefile provides the township Chinese labels; if absent,
Tool 1 falls back to `QSDWDM` prefix matching.

[tnls]: https://en.wikipedia.org/wiki/Third_National_Land_Survey_of_China

## Method overview

The pipeline trains an ensemble of three contrastive world-model
members on (state, action, reward, next_state) tuples sampled by random
policy plus pairwise margin-ranking pairs. At planning time, an MPC
loop rolls out the top-K candidate blocks for H steps under the
ensemble and commits the one whose rollout accumulates the highest
predicted discounted return. The objective combines slope reduction
(more parcels on flat, cultivable land), contiguity (reducing patch
fragmentation), and baimu-fang area (aggregate areas exceeding the
100-mu / 6.67-ha threshold used in Chinese land-use planning).

## Citation

If you use this toolbox in academic work, please cite the underlying
research paper (citation TBD; author: Ning Zhou, Peking University).

## License

MIT. See `LICENSE`.
