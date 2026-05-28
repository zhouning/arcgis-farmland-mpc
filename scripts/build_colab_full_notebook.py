"""Build farmland_mpc_colab_full.ipynb — production-grade Colab notebook.

Differences vs farmland_mpc_colab_demo.ipynb:
- User uploads/points-at real DLTB shapefile on Google Drive (not synthesised)
- DEM is auto-fetched from Copernicus GLO-30 public AWS bucket
  (no auth, ~50-300 MB depending on AOI)
- UTM zone auto-detected from shapefile centroid
- Production hyperparameters: 1,000 pairwise states, 3-member ensemble,
  30 epochs, MPC H=5 K=50, 5 evaluation episodes
- All outputs (prepared_dir, ONNX ensemble, optimised shapefile) land on Drive

Run from the repo root:
    python scripts/build_colab_full_notebook.py
The output notebook is written to notebooks/farmland_mpc_colab_full.ipynb
relative to the repo root (i.e. relative to this script's parent directory).
"""
from __future__ import annotations
import json
from pathlib import Path
from textwrap import dedent

# Repo root = parent of this script's directory (scripts/)
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "notebooks" / "farmland_mpc_colab_full.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.rstrip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.rstrip("\n").splitlines(keepends=True),
    }


cells = []

# --- 1. Title ---
cells.append(md(dedent("""\
    # `farmland-mpc` — Production Colab Pipeline (A → B → C → D)

    **Model-based AI planning for county-scale farmland consolidation.**

    This notebook runs the **full production pipeline** on your own cadastral data:

    1. **A — Prepare**: project a user-supplied DLTB shapefile, fetch a Copernicus GLO-30 DEM for its bounding box from the AWS public mirror, compute Horn-3×3 slope per parcel, define spatial blocks.
    2. **B — Sample**: collect 6,000 transitions + 50,000 ground-truth $(s,a,r)$ pairwise evaluations from random-policy rollouts.
    3. **C — Train**: train a 3-member contrastive world-model ensemble ($\\lambda_{\\text{rank}}=5$) and export each member to ONNX.
    4. **D — Plan**: run 5 MPC episodes (horizon 5, top-K 50, greedy continuation), write the optimised shapefile back to Drive.

    **What's different from the demo notebook:**

    | | demo | this notebook (production) |
    |---|---|---|
    | Input | 36-parcel synthetic toy | your real DLTB shapefile on Drive |
    | DEM | inline synthetic raster | Copernicus GLO-30 auto-downloaded |
    | Pairwise states | 20 | **1,000** (paper-default) |
    | Ensemble | 2 members × 3 epochs | **3 members × 30 epochs** |
    | MPC eval | 1 episode, H=2, K=3 | **5 episodes, H=5, K=50** |
    | Wall time | <1 min | 1–3 hours (county-scale) on T4 / desktop |

    **Output**: an optimised shapefile (with `CHG_FLAG` per parcel) and full evaluation logs in your Drive folder, ready for ArcGIS Pro / QGIS inspection.

    > ⚠ This notebook expects ~5 GB free on Drive and a recent Colab Pro / T4 runtime. For tiny test runs (e.g. 100-parcel AOI) you can keep the free tier.
    """)))

# --- 2. Mount Drive + paths ---
cells.append(md(dedent("""\
    ## 1 · Mount Google Drive

    Mount your Drive read-write. We'll read the shapefile from `INPUT_DIR` and write all outputs under `WORK_DIR`.

    **Before running**: upload your DLTB shapefile (the full `.shp` + `.shx` + `.dbf` + `.prj` + `.cpg` set) into a single folder on your Drive. Note its absolute path.
    """)))

cells.append(code(dedent("""\
    from google.colab import drive
    drive.mount('/content/drive')
    """)))

cells.append(code(dedent("""\
    # === EDIT THESE TWO PATHS ===
    INPUT_DLTB = '/content/drive/MyDrive/farmland_mpc/input/DLTB.shp'   # ← your shapefile
    WORK_DIR   = '/content/drive/MyDrive/farmland_mpc/run_1'            # ← outputs land here

    from pathlib import Path
    INPUT_DLTB = Path(INPUT_DLTB)
    WORK_DIR   = Path(WORK_DIR)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    assert INPUT_DLTB.exists(), f'Not found: {INPUT_DLTB}'
    print(f'INPUT_DLTB: {INPUT_DLTB}  ({INPUT_DLTB.stat().st_size/1e6:.1f} MB)')
    for ext in ('.shx', '.dbf', '.prj'):
        sib = INPUT_DLTB.with_suffix(ext)
        assert sib.exists(), f'Missing companion file: {sib}'
    print('Shapefile companion files OK (.shx, .dbf, .prj)')
    print(f'WORK_DIR:   {WORK_DIR}')
    """)))

# --- 3. Install + import ---
cells.append(md(dedent("""\
    ## 2 · Install dependencies and `farmland-mpc`
    """)))

cells.append(code(dedent("""\
    # Geospatial + ML stack. Colab pre-installs most of torch/sklearn/networkx.
    !pip install --quiet geopandas rasterio pyogrio shapely fiona libpysal \\
                          typer tqdm onnx onnxruntime onnxscript gymnasium pyproj
    """)))

cells.append(code(dedent("""\
    # Pull the latest farmland-mpc release directly from GitHub.
    !pip install --quiet git+https://github.com/zhouning/arcgis-farmland-mpc.git@main
    """)))

cells.append(code(dedent("""\
    import farmland_mpc, torch, rasterio, geopandas as gpd, pyproj
    print(f'farmland_mpc: {farmland_mpc.__version__}')
    print(f'torch:        {torch.__version__}  (CUDA: {torch.cuda.is_available()})')
    print(f'rasterio:     {rasterio.__version__}')
    print(f'geopandas:    {gpd.__version__}')
    print(f'pyproj:       {pyproj.__version__}')
    """)))

# --- 4. Inspect shapefile + pick UTM ---
cells.append(md(dedent("""\
    ## 3 · Inspect the shapefile and pick a projected CRS

    The DLTB shapefile must contain at minimum:

    - `DLBM` — land-class code (Third National Land Survey codes; farmland classes 011/012/013, forest 031)
    - `QSDWDM` — owner/township code (first 9 characters identify the township; used by Phase B)
    - `BSM` — parcel identifier (any unique string per parcel)

    We pick a UTM zone from the shapefile's centroid so that slope / area calculations are in metres.
    """)))

cells.append(code(dedent("""\
    import geopandas as gpd

    dltb = gpd.read_file(INPUT_DLTB)
    print(f'parcels:  {len(dltb):,}')
    print(f'CRS:      {dltb.crs}')
    print(f'columns:  {list(dltb.columns)[:12]}{"..." if len(dltb.columns)>12 else ""}')

    required = ['DLBM', 'QSDWDM', 'BSM']
    missing = [c for c in required if c not in dltb.columns]
    if missing:
        print(f'\\n⚠ Missing required columns: {missing}')
        print('   You can rename your columns to match, or override the field names in the prepare.run() call below.')
    else:
        print('\\nRequired columns present.')
        print(f"farmland parcels (DLBM ∈ 011/012/013): {dltb['DLBM'].astype(str).isin(['011','012','013']).sum():,}")
        print(f"forest parcels   (DLBM = 031):          {(dltb['DLBM'].astype(str) == '031').sum():,}")
    """)))

cells.append(code(dedent("""\
    # Auto-pick a UTM zone from the shapefile centroid (WGS84).
    # GLO-30 tiles are indexed in WGS84 1°×1° squares; UTM is for the metric pipeline.
    import math

    centroid_wgs = dltb.to_crs('EPSG:4326').geometry.union_all().centroid
    lon, lat = centroid_wgs.x, centroid_wgs.y
    utm_zone = int((lon + 180) // 6) + 1
    north = lat >= 0
    epsg = (32600 if north else 32700) + utm_zone
    PROJ_CRS = f'EPSG:{epsg}'
    print(f'Centroid: lon={lon:.4f}, lat={lat:.4f}')
    print(f'Auto-picked UTM zone: {utm_zone}{"N" if north else "S"} → {PROJ_CRS}')

    # WGS84 bbox for DEM tile selection
    bbox_wgs = dltb.to_crs('EPSG:4326').total_bounds  # (minx, miny, maxx, maxy)
    print(f'WGS84 bbox: lon [{bbox_wgs[0]:.4f}, {bbox_wgs[2]:.4f}], '
          f'lat [{bbox_wgs[1]:.4f}, {bbox_wgs[3]:.4f}]')
    """)))

# --- 5. Fetch Copernicus DEM ---
cells.append(md(dedent("""\
    ## 4 · Fetch Copernicus GLO-30 DEM tiles

    [Copernicus GLO-30](https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model) is a 30 m global DEM, public-domain (CC-BY 4.0). The AWS public mirror at `s3://copernicus-dem-30m/` exposes tiles as **1°×1° Cloud-Optimized GeoTIFFs**, no auth required.

    Tile naming pattern (positive lat / lon shown):
    `Copernicus_DSM_COG_10_N31_00_E106_00_DEM.tif`

    For each 1° square covering the AOI, we download the tile, then mosaic + clip to the shapefile bbox (with a small buffer), reproject to UTM, and save as the input DEM.
    """)))

cells.append(code(dedent("""\
    import math
    import urllib.request
    import urllib.error
    from pathlib import Path

    DEM_DIR = WORK_DIR / 'dem_tiles'
    DEM_DIR.mkdir(parents=True, exist_ok=True)

    minx, miny, maxx, maxy = bbox_wgs
    # 1° tiles touched by the bbox (inclusive at edges)
    lon_tiles = range(math.floor(minx), math.floor(maxx) + 1)
    lat_tiles = range(math.floor(miny), math.floor(maxy) + 1)

    def tile_name(lat: int, lon: int) -> str:
        ns = 'N' if lat >= 0 else 'S'
        ew = 'E' if lon >= 0 else 'W'
        return f'Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM'

    BASE = 'https://copernicus-dem-30m.s3.amazonaws.com'

    tiles_to_fetch = []
    for lat in lat_tiles:
        for lon in lon_tiles:
            tname = tile_name(lat, lon)
            url   = f'{BASE}/{tname}/{tname}.tif'
            local = DEM_DIR / f'{tname}.tif'
            tiles_to_fetch.append((tname, url, local))

    print(f'AOI touches {len(tiles_to_fetch)} GLO-30 tile(s):')
    for tname, url, local in tiles_to_fetch:
        print(f'  {tname}')
    """)))

cells.append(code(dedent("""\
    # Download tiles (skips already-downloaded). Each tile is ~50-150 MB.
    import shutil, sys

    fetched = []
    for tname, url, local in tiles_to_fetch:
        if local.exists() and local.stat().st_size > 1_000_000:
            print(f'[skip] {tname}  ({local.stat().st_size/1e6:.1f} MB cached)')
            fetched.append(local); continue
        print(f'[get]  {tname}  ← {url}', flush=True)
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, open(local, 'wb') as f:
                shutil.copyfileobj(resp, f)
            print(f'       {local.stat().st_size/1e6:.1f} MB ✓')
            fetched.append(local)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Ocean / no-data tile — skip, mosaic step tolerates gaps
                print(f'       404 (no land data here) — skipped')
            else:
                raise

    assert fetched, 'No DEM tiles fetched — check bbox / network.'
    print(f'\\n{len(fetched)} tile(s) on disk')
    """)))

cells.append(code(dedent("""\
    # Mosaic + clip + reproject to UTM.
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import box

    # Merge in WGS84
    srcs = [rasterio.open(p) for p in fetched]
    mosaic, m_transform = merge(srcs)
    m_meta = srcs[0].meta.copy()
    m_meta.update({'height': mosaic.shape[1], 'width': mosaic.shape[2], 'transform': m_transform})
    mosaic_wgs = WORK_DIR / 'dem_mosaic_wgs84.tif'
    with rasterio.open(mosaic_wgs, 'w', **m_meta) as dst:
        dst.write(mosaic)
    for s in srcs: s.close()
    print(f'mosaic (WGS84):  {mosaic_wgs}  {mosaic.shape}')

    # Buffer bbox by 0.01° (~1 km) so slope edges aren't NaN at the shapefile boundary
    buf = 0.01
    aoi_wgs = box(minx - buf, miny - buf, maxx + buf, maxy + buf)

    # Clip in WGS84, then reproject to UTM
    with rasterio.open(mosaic_wgs) as src:
        clipped, clip_transform = rio_mask(src, [aoi_wgs], crop=True, nodata=src.nodata)
        clip_meta = src.meta.copy()
        clip_meta.update({'height': clipped.shape[1], 'width': clipped.shape[2],
                          'transform': clip_transform})

    clipped_wgs = WORK_DIR / 'dem_clipped_wgs84.tif'
    with rasterio.open(clipped_wgs, 'w', **clip_meta) as dst:
        dst.write(clipped)

    # Now reproject to chosen UTM at 30 m
    DEM_FINAL = WORK_DIR / 'dem.tif'
    with rasterio.open(clipped_wgs) as src:
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src.crs, PROJ_CRS, src.width, src.height, *src.bounds, resolution=30.0)
        dst_meta = src.meta.copy()
        dst_meta.update({'crs': PROJ_CRS, 'transform': dst_transform,
                         'width': dst_w, 'height': dst_h})
        with rasterio.open(DEM_FINAL, 'w', **dst_meta) as dst:
            for b in range(1, src.count + 1):
                reproject(source=rasterio.band(src, b),
                          destination=rasterio.band(dst, b),
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=dst_transform, dst_crs=PROJ_CRS,
                          resampling=Resampling.bilinear)

    # Clean up intermediates
    mosaic_wgs.unlink(missing_ok=True)
    clipped_wgs.unlink(missing_ok=True)
    print(f'DEM (UTM 30m):   {DEM_FINAL}  ({DEM_FINAL.stat().st_size/1e6:.1f} MB)')
    """)))

# --- 6. Phase A+B+C: prepare ---
cells.append(md(dedent("""\
    ## 5 · Phase A: project DLTB, compute slope, define blocks

    `farmland_mpc.prepare.run` reprojects the DLTB to UTM, runs Horn-3×3 slope from the DEM into a `slope_mean` per parcel, then groups parcels into spatial blocks per township (Paper 3 hybrid algorithm). Block-level features are what the world model consumes downstream.

    **Default block parameters** match the Paper 9 v6 toolbox (`min_parcels=3`, `max_parcels=30`, `min_area_ha=0.5`). For very small AOIs, you may need to lower `min_parcels_per_township` (otherwise small townships are dropped).
    """)))

cells.append(code(dedent("""\
    from farmland_mpc.prepare import run as prepare_run

    PREPARED = WORK_DIR / 'prepared'
    PREPARED.mkdir(exist_ok=True)

    prepare_summary = prepare_run(
        dltb_path=str(INPUT_DLTB),
        dem_path=str(DEM_FINAL),
        prepared_dir=str(PREPARED),
        proj_crs=PROJ_CRS,
        # Field names — override if your shapefile uses different column names
        dlbm_field='DLBM',
        qsdwdm_field='QSDWDM',
        bsm_field='BSM',
        # Block parameters (Paper 3 defaults)
        run_phase_bc=True,
        min_parcels=3,
        min_area_ha=0.5,
        max_parcels=30,
        min_parcels_per_township=50,
    )
    print('\\nPhase A done. prepared_dir contents:')
    !ls -la "$PREPARED"
    """)))

# --- 7. Phase B sample ---
cells.append(md(dedent("""\
    ## 6 · Phase B: sample transitions + pairwise data

    Two sweeps over the block-level environment under random policy:

    - **Transitions**: 60 episodes × 100 steps = ~6,000 $(s, a, s', r)$ tuples for the dynamics objective.
    - **Pairwise**: 1,000 states, each with 50 candidate actions snapshotted from the env, evaluated against ground-truth reward. This is the contrastive-MPC signal that turns the MSE-only ensemble into a useful ranker.

    On county-scale data (~2,600 blocks) this step is the slowest non-training phase: budget ~30–60 minutes on Colab T4.
    """)))

cells.append(code(dedent("""\
    from farmland_mpc.sample import run as sample_run

    sample_summary = sample_run(
        prepared_dir=str(PREPARED),
        n_transition_episodes=60,
        n_pairwise_states=1000,
        n_pairwise_actions=50,
        seed=0,
        proj_crs=PROJ_CRS,
    )
    print('\\nPhase B done.')
    for k, v in sample_summary.items():
        print(f'  {k}: {v}')
    """)))

# --- 8. Phase C train ---
cells.append(md(dedent("""\
    ## 7 · Phase C: train the contrastive ensemble

    Three independently initialised world-model members are trained for 30 epochs each with the contrastive objective ($\\lambda_{\\text{rank}}=5$, margin $m=0.1$). Each member is exported to ONNX with the block count baked in.

    Wall-time: ~30–60 minutes total on T4 (the bulk is the pairwise pass through `tool2/pairwise.npz`).
    """)))

cells.append(code(dedent("""\
    from farmland_mpc.train_ensemble import run as train_run

    train_run(
        prepared_dir=str(PREPARED),
        n_members=3,
        epochs=30,
        patience=8,
        lambda_rank=5.0,
        margin=0.1,
        batch_size=256,
        n_pairs_per_state=10,
        pw_subsample=100,
        lr=1e-3,
        weight_decay=1e-5,
        seed_base=0,
        torch_threads=0,
    )

    onnx_files = sorted((PREPARED / 'tool3').glob('*.onnx'))
    print(f'\\nPhase C done. Exported {len(onnx_files)} ONNX members:')
    for f in onnx_files:
        print(f'  {f.name}  ({f.stat().st_size/1e6:.1f} MB)')
    """)))

# --- 9. Phase D plan ---
cells.append(md(dedent("""\
    ## 8 · Phase D: MPC planning (5 episodes, H=5, K=50)

    The Paper 9 v6 production configuration:

    - **Horizon** H = 5 (5-step lookahead under the learned model)
    - **Candidates** K = 50 per step
    - **Continuation** = greedy (argmax of predicted reward)
    - **Episodes** = 5 with different seeds → reports mean ± std

    Wall-time: ~10–15 minutes per episode on T4 / desktop CPU, so the 5-episode run is ~1 hour.
    """)))

cells.append(code(dedent("""\
    from farmland_mpc.mpc_plan import run as plan_run

    MPC_OUT = WORK_DIR / 'mpc_output'
    MPC_OUT.mkdir(exist_ok=True)

    plan_summary = plan_run(
        ensemble_dir=str(PREPARED / 'tool3'),
        out_dir=str(MPC_OUT),
        prepared_dir=str(PREPARED),
        proj_crs=PROJ_CRS,
        horizon=5,
        top_k=50,
        n_episodes=5,
        continuation='greedy',
        scoring='reward',
        seed_offset=0,
        input_dltb_fc=str(INPUT_DLTB),
    )
    print('\\nPhase D done.')
    for k, v in plan_summary.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.4f}')
        else:
            print(f'  {k}: {v}')
    """)))

# --- 10. Visualise + finalise ---
cells.append(md(dedent("""\
    ## 9 · Visualise before vs after

    Compare the original and optimised cadastres. `CHG_FLAG=1` (farm → forest) is red; `CHG_FLAG=2` (forest → farm) is green.
    """)))

cells.append(code(dedent("""\
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import geopandas as gpd

    optimised = sorted(MPC_OUT.glob('**/optimized.shp'))
    assert optimised, f'No optimised.shp under {MPC_OUT}'
    out_shp_opt = optimised[0]
    print(f'Reading optimised cadastre: {out_shp_opt}')

    orig = gpd.read_file(PREPARED / 'dem_slope_analysis' / 'output' / 'DLTB_with_slope.shp')
    opt  = gpd.read_file(out_shp_opt)
    print(f'orig: {len(orig):,} parcels  |  opt: {len(opt):,} parcels')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    farm_codes = {'011', '012', '013'}
    forest_codes = {'031'}
    def colour_landuse(gdf):
        c = []
        for v in gdf['DLBM'].astype(str):
            if v in farm_codes:   c.append('#7CB342')
            elif v in forest_codes: c.append('#1B5E20')
            else:                  c.append('#CFD8DC')
        return c

    orig.plot(ax=axes[0], color=colour_landuse(orig), edgecolor='none')
    axes[0].set_title(f'Original ({len(orig):,} parcels)')
    axes[0].set_axis_off()

    base_colour = colour_landuse(opt)
    if 'CHG_FLAG' in opt.columns:
        for i, flag in enumerate(opt['CHG_FLAG'].fillna(0).astype(int)):
            if flag == 1: base_colour[i] = '#D32F2F'   # farm -> forest
            elif flag == 2: base_colour[i] = '#43A047' # forest -> farm
    opt.plot(ax=axes[1], color=base_colour, edgecolor='none')
    n_changed = int((opt['CHG_FLAG'].fillna(0) > 0).sum()) if 'CHG_FLAG' in opt.columns else 0
    axes[1].set_title(f'Optimised  ({n_changed} parcels swapped)')
    axes[1].set_axis_off()

    legend = [
        mpatches.Patch(color='#7CB342', label='farmland (kept)'),
        mpatches.Patch(color='#1B5E20', label='forest (kept)'),
        mpatches.Patch(color='#D32F2F', label='farm → forest'),
        mpatches.Patch(color='#43A047', label='forest → farm'),
    ]
    fig.legend(handles=legend, loc='lower center', ncols=4, frameon=False)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()
    """)))

# --- 11. Wrap up ---
cells.append(md(dedent("""\
    ## 10 · Summary

    All artefacts are now on Drive under `WORK_DIR`:

    ```
    WORK_DIR/
    ├── dem.tif                                       UTM-projected 30m DEM
    ├── dem_tiles/                                    raw GLO-30 1° tiles (kept for reuse)
    ├── prepared/
    │   ├── dem_slope_analysis/output/DLTB_with_slope.shp   parcels + slope_mean
    │   ├── results_real/blocks/township_*/                 block compositions + features
    │   ├── tool2/transitions.npz + pairwise.npz            sampled data (Phase B)
    │   ├── tool3/*.onnx                                    trained ensemble (3 members)
    │   └── prepare_data_summary.json                       provenance
    └── mpc_output/
        ├── optimized.shp                                   ← the planning result
        └── mpc_run.log + per-episode JSON                  full evaluation trace
    ```

    Next steps:

    - Open `optimized.shp` in ArcGIS Pro / QGIS to inspect the consolidation plan.
    - Re-run `Phase D` (cell 8) with different reward weights or `top_k` to explore alternative scenarios. **Phases A–C only need to be run once** per AOI, so iterating on scenarios is cheap.
    - For deeper customisation (block sizing, township grouping, reward decomposition), see the [User Guide](https://github.com/zhouning/arcgis-farmland-mpc/blob/main/docs/USER_GUIDE.md) and the [QUICKSTART](https://github.com/zhouning/arcgis-farmland-mpc/blob/main/docs/QUICKSTART.md).

    For methodology details, see [Paper 9](https://github.com/zhouning/arcgis-farmland-mpc).
    """)))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {OUT}\n  cells: {len(cells)}  size: {OUT.stat().st_size/1024:.1f} KB")
