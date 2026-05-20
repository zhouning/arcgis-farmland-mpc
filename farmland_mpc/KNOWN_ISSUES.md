# farmland_mpc — known issues

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
