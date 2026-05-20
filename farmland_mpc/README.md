# farmland-mpc

County-scale farmland consolidation via a contrastive world-model + MPC, packaged as a pure-Python library and CLI. No proprietary GIS dependencies.

This is the open-source backend used by both the standalone CLI (`farmland-mpc plan ...`) and the optional ArcGIS Pro toolbox (`LandUseOptimization_P9.pyt`). The same pipeline runs unchanged on Linux / macOS / Windows / Colab.

## Install

```bash
pip install -e .
# Optional benchmark generator deps:
pip install -e .[benchmark]
```

Python ≥ 3.11.

## Quick start

```bash
# Phase A: DEM + DLTB → per-parcel slope_mean shapefile
farmland-mpc prepare \
    --dltb path/to/DLTB.shp \
    --dem  path/to/DEM.tif  \
    --out  prepared_dir/    \
    --crs  EPSG:32648
```

Subsequent phases (`sample`, `train`, `plan`) are being lifted from the existing toolbox source over the next iteration; for now run them through `core/sample_transitions.py`, `core/train_ensemble.py`, `core/mpc_plan.py` directly.

## Why this exists

The same pipeline previously required ArcGIS Pro + a Spatial Analyst license (for `arcpy.sa.Slope` + `ZonalStatisticsAsTable`). This package replaces those steps with `rasterio` + `geopandas` + a hand-rolled Horn 3×3 slope, producing identical output schemas (`DLTB_with_slope.shp` with a `slope_mean` field) so downstream code works unchanged.

License: MIT.
