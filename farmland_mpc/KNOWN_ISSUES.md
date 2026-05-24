# farmland_mpc — known issues

## Tool 3 NaN at epoch 1 / ONNX parity AssertionError (FIXED in 0.2.1)

**Symptom** (Colab, Tool 3 contrastive trainer):

```
Epoch  1/30  mse=nan  rank_val=0.00000  val=nan  cos=nan  rank_acc=0.500
Early stopping at epoch 8 (best=-1)
    onnx parity max diff = nan
AssertionError: ONNX parity check failed: nan
```

**Root cause**: `_zonal_mean` returns NaN for parcels with no valid DEM cells (DEM tile coverage gaps, polygons that don't intersect any cell centre, ocean tiles silently skipped at 404). The NaN propagates:

1. `prepare.py` writes NaN into `slope_mean` column of `DLTB_with_slope.shp`.
2. `county_env.py` sets `slope_min/max/range = nan`, so `inv_sr = 1/nan = nan` poisons every block-feature column 0-6 and global-feature columns 1, 4.
3. Tool 2 saves NaN-laden npz; Tool 3's first forward pass hits NaN.
4. The ONNX parity assertion correctly catches it but points at the wrong file.

**Fix** (defense in depth across three layers):

- **Layer 1 (`prepare.py`)**: NaN slopes filled with `np.nanmedian` before the shapefile is written, with a warning that lists how many parcels were affected.
- **Layer 2 (`county_env.py`)**: NaN survivors filled at env-load time, so already-prepared `run_*/prepared/` directories from older Tool 1 runs work without re-running phase A.
- **Layer 3 (`sample.py`)**: `_assert_finite` raises with a clear "re-run Tool 1" message before any NaN reaches Tool 3.

If you hit this on `farmland_mpc < 0.2.1`, upgrade with:

```bash
pip install --upgrade git+https://github.com/zhouning/arcgis-farmland-mpc.git@main
```

Then re-run Tool 1 (cell 16 in the Colab notebook) so the median fill takes effect, OR delete `prepared/dem_slope_analysis/output/DLTB_with_slope.shp` to force regeneration.

---

## Smoke test on ArcGIS Pro-managed envs: fiona init crashes on Chinese Windows

**Symptom** (encountered 2026-05-20 on `arcgispro-py3-clone-new2`, Python 3.13 + esri-channel rasterio 1.4.3 + fiona):

```
ImportError: DLL load failed while importing _warp: 找不到指定的程序
```

This first one is fixed — see the `_isolate_conda_geostack()` shim in `farmland_mpc/__init__.py`. After fixing rasterio.warp, fiona then crashes with:

```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xd5 in position 88: invalid continuation byte
```

cascading into `SystemError: ... returned a result with an exception set` from the Python logging RLock.

**Root cause chain**:

1. `arcpy_init.set_product_paths()` (auto-loaded via `ArcGISPro.pth` at site init) calls `os.add_dll_directory(ArcGISPro/bin)`. This puts ArcGIS Pro's `bin/` directory ahead of the conda env's `Library/bin/` for native DLL resolution.
2. When `rasterio` or `fiona` initialises GDAL, GDAL's driver manager scans ArcGIS Pro's `bin/gdalplugins/` directory for plugins (probably because the directory is in the DLL search path, or via compiled-in fallback — unclear).
3. The ArcGIS-shipped plugins (e.g. `gdal_E57.dll`, `gdal_SDTS.dll`) are built for ArcGIS's GDAL runtime, not the esri-channel one in the conda env, so they fail to load.
4. The GDAL error messages are emitted in the system codepage (CP936/GBK on Chinese Windows): `127: 找不到指定的程序`.
5. `fiona._env.logging_error_handler` tries to decode these messages as UTF-8 and fails with `UnicodeDecodeError`.
6. The decode failure inside a Cython callback corrupts Python's logging RLock state, leading to `SystemError`s on subsequent log calls. Fiona's init never completes.

`rasterio` is more tolerant (it survives the same plugin scan), but `fiona` is not.

**Workarounds attempted** (all unsuccessful on `arcgispro-py3-clone-new2`):

- Setting `GDAL_DRIVER_PATH` to an empty directory via env var (in `__init__.py` or via `sitecustomize.py` that runs before `.pth` files) — GDAL still finds ArcGIS's plugins via the DLL search path added by `arcpy_init`.
- Disabling `ArcGISPro.pth` — breaks `osgeo._gdal` import because rasterio/osgeo depend on ArcGIS-side DLLs (`gdal_e.dll`) that `arcpy_init` puts on the DLL search path.
- Installing `pyogrio` to bypass fiona — not available in esri channel.

**Working solution**: use a separate conda-forge-only environment that does NOT inherit ArcGIS Pro's site-packages or DLL search paths.

```
conda create -p D:/test/envs/farmland-mpc-pure --override-channels -c conda-forge -y \
    python=3.11 geopandas rasterio pyogrio fiona shapely numpy scipy networkx \
    libpysal scikit-learn matplotlib tqdm typer pytorch cpuonly
```

`farmland_mpc` is pure Python; the standalone CLI and module-level usage work
in any env that has the dependencies. The ArcGIS Pro toolbox shell
(`LandUseOptimization_P9.pyt`) is the only entry point that requires the
ArcGIS-managed env, and Esri may need to fix the fiona/GBK issue upstream
before it becomes reliable on Chinese Windows.

## DLL chain fix in `farmland_mpc/__init__.py`

The package's `__init__.py` registers `Library/bin`, `Library/mingw-w64/bin`,
and `Library/usr/bin` with `os.add_dll_directory` on Windows + Python 3.8+,
because PATH is no longer consulted for native DLL resolution. It also sets
`PROJ_LIB`, `PROJ_DATA`, `GDAL_DATA`, and clears `GDAL_DRIVER_PATH` to point
at the active conda env's own files. This shim is harmless on non-conda envs
(it just no-ops when `Library/` doesn't exist).

## CRS handling note

`prepare.py` accepts `proj_crs` as `EPSG:nnnn`, raw WKT, or PROJ-string. EPSG-string lookups depend on a current `proj.db`; the ArcGIS-bundled proj.db is at version 5 while rasterio 1.4.3 expects ≥6 and emits warnings. Pass WKT directly to bypass the database when running inside ArcGIS-managed envs (the smoke test does this already).
