# Farmland MPC — QGIS Processing plugin

QGIS GUI for the Communications Earth & Environment manuscript "Reproducible model-based AI planning for county-scale farmland consolidation in fragmented mountain landscapes" (Zhou & Jing 2026).

Wraps the four `farmland-mpc` CLI subcommands as native QGIS Processing algorithms so you can run the full pipeline without touching a terminal. Equivalent in scope to `LandUseOptimization_P9.pyt` (the ArcGIS Pro Python Toolbox shipped with the same project).

The plugin **calls the `farmland-mpc` executable as a subprocess**; it does not bundle the heavy dependencies (PyTorch, ONNX Runtime, libpysal, geopandas, etc.) inside QGIS. You install `farmland-mpc` once into a separate conda environment, point the plugin at that environment, and the plugin then orchestrates the four stages from inside QGIS.

## Algorithms

Once installed, the plugin appears in `Processing Toolbox → Farmland MPC → Pipeline (run in order)` with four algorithms:

| # | Name | CLI equivalent | Wall time |
|---|---|---|---|
| 1 | Prepare (DLTB + DEM → slope + blocks) | `farmland-mpc prepare` | ~3–5 min |
| 2 | Sample (transitions + pairwise dataset) | `farmland-mpc sample` | ~16 min |
| 3 | Train (contrastive ensemble + ONNX export) | `farmland-mpc train` | ~40–60 min |
| 4 | Plan (MPC → optimised cadastre) | `farmland-mpc plan` | ~3–5 min per episode |

## Installation

### 1. Install `farmland-mpc` in a conda environment (one-time)

```bash
git clone https://github.com/zhouning/arcgis-farmland-mpc.git
cd arcgis-farmland-mpc
conda env create -f environment.yml
conda activate farmland-mpc
farmland-mpc version    # should print farmland-mpc 0.x.y
```

The above is the canonical installation path documented in `docs/REPRODUCE.md`. Note the **absolute path of the executable**, e.g. `~/miniconda3/envs/farmland-mpc/bin/farmland-mpc`.

### 2. Install the QGIS plugin

Copy (or symlink) the `farmland_mpc/` subdirectory into your QGIS profile's `python/plugins/` directory. Note that **QGIS 4 reads from the `QGIS3/` profile path** (the directory name has not been bumped), so the same path works for both QGIS 3.34 LTR and QGIS 4.0:

**macOS:**
```bash
mkdir -p ~/Library/Application\ Support/QGIS/QGIS3/profiles/default/python/plugins/
cp -r qgis_plugin/farmland_mpc \
      ~/Library/Application\ Support/QGIS/QGIS3/profiles/default/python/plugins/
```

**Linux:**
```bash
cp -r qgis_plugin/farmland_mpc \
      ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/
```

**Windows:**
```
copy qgis_plugin\farmland_mpc to %APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\
```

Then in QGIS:

1. Plugins → Manage and Install Plugins → check **Farmland MPC**.
2. Settings → Options → Advanced → search for `farmland_mpc/executable_path` and set it to the absolute path printed by `which farmland-mpc` inside the conda environment (e.g. `/Users/you/miniconda3/envs/farmland-mpc/bin/farmland-mpc`).
   *Alternative:* export `FARMLAND_MPC_EXECUTABLE` in the shell that launches QGIS.

Or from a terminal (no GUI required):
```bash
qgis_process plugins enable farmland_mpc
qgis_process list | grep -A4 "Farmland MPC"
```

### 3. Verify

Open `Processing Toolbox` and look for the **Farmland MPC** provider with four algorithms. Run **1 — Prepare** with the `tests/` smoke-test data to confirm the executable is wired up:

```
DLTB:  <repo>/farmland_mpc/tests/data/dongxing_2townships.shp
DEM:   <repo>/farmland_mpc/tests/data/dongxing_dem_4326.tif
Out:   /tmp/farmland_mpc_smoke
CRS:   EPSG:32648
```

The 70-second smoke test should complete and write a `prepared/` directory readable by stage 2.

A second verification path (no GUI required) exercises the full pipeline against the public `paper/checkpoints/bishan/` artefacts: run `4 — Plan` for one episode against the shipped Bishan ensemble:

```bash
qgis_process run farmland_mpc:plan \
    --ENSEMBLE_DIR=<repo>/paper/checkpoints/bishan/contrastive_5seed_seed0_ensemble \
    --PREPARED_DIR=<your_run>/prepared \
    --OUT_DIR=/tmp/qgis_plan_smoke \
    --HORIZON=5 --TOP_K=50 --N_EPISODES=1 \
    --CONTINUATION=0 --SCORING=0
```

A correct end-to-end install reaches step 100 with `slope ≈ −1.75%` (matching `paper/repro_artifacts/macos_2026-05-29/bishan_5seed.json` seed 0 to floating-point identity), writes `optimized.shp` and `mpc_summary.json`, and finishes in ~5 minutes on a 12-thread CPU.

## Pipeline workflow

The four stages are designed to run in order; each writes outputs the next reads.

```
Tool 1 ─► prepared/                ┐
Tool 2 ─► prepared/tool2/...       │  one prepared/ per county
Tool 3 ─► prepared/tool3/*.onnx    ┘
Tool 4 ─► out_dir/optimized.shp + mpc_summary.json
```

Drag `optimized.shp` back into the QGIS map canvas to inspect the swap pattern: parcels with `CHG_FLAG=1` are farm→forest swaps (slope-shed), `CHG_FLAG=2` are forest→farm swaps (slope-take), and `CHG_FLAG=0` are unchanged.

## Limitations vs the ArcGIS Pro toolbox

- **No "Tool 5 — Check Dependencies" equivalent**: the plugin assumes the conda environment is healthy. If `farmland-mpc version` works in your terminal, the plugin will work.
- **Progress bar granularity**: QGIS shows `INFO`/`WARNING`/`ERROR` log lines streamed from the subprocess but no fine-grained percentage progress (the same limitation applies to QGIS's GRASS / GDAL algorithms).
- **N_blocks bake-in**: stage 4 will fail with a clear error if the trained ensemble's `N_blocks` does not match the prepared dataset (same pre-flight check as the ArcGIS Pro toolbox).

## Citation

If you use this plugin or the underlying pipeline in published work, please cite:

> Zhou N. & Jing X. (2026). *Reproducible model-based AI planning for county-scale farmland consolidation in fragmented mountain landscapes.* Communications Earth & Environment (under review). https://github.com/zhouning/arcgis-farmland-mpc

## Licence

MIT, identical to the parent repository.
