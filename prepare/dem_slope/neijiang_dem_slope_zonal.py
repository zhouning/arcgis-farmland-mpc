"""Copernicus DEM GLO-30 Download + Slope Zonal Statistics Pipeline.
(Neijiang Dongxing adaptation of the Bishan pipeline — shapefile input.)

Downloads Copernicus DEM tiles covering the study area, computes slope,
runs zonal statistics for each parcel, and writes results to GPKG.

CRS Note (Neijiang vs Bishan):
  - Bishan:  GDB labeled EPSG:4610 but contains CGCS2000 lat/lon; pipeline
             overrides CRS to EPSG:4326 (safe, coords unchanged).
  - Neijiang: DLTB.shp is EPSG:2359 (CGCS2000 / 3-degree Gauss-Kruger CM 105E),
              i.e. *projected* meters, not lat/lon. We therefore REPROJECT
              (to_crs), not override, before DEM overlay.
  - DBF encoding: Neijiang DBF is GBK; Bishan was UTF-8. Use encoding='gbk'
              when reading the shapefile.

PUBLIC RELEASE NOTE
-------------------
The DLTB cadastral input is restricted Third National Land Survey (三调)
data and cannot be redistributed (see paper Data Availability). Set the
paths below to your own copy before running. The Copernicus DEM GLO-30
tiles ARE public — the script downloads them on demand from AWS S3.

Required env vars (or edit the constants directly):
    NEIJIANG_DLTB_SHP   -> DLTB.shp file
    NEIJIANG_XZQ_SHP    -> 行政区 (district boundary) shapefile
    NEIJIANG_OUTPUT_DIR -> writable working dir for intermediate + output
"""

import os
import sys
import time
import math
import struct
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from PIL import Image
import requests
from scipy.ndimage import map_coordinates
from shapely import points as make_points
from shapely import STRtree
from pyproj import Geod

# ==============================================================================
# Configuration (Neijiang Dongxing)
# ==============================================================================

SHP_PATH = Path(os.environ.get("NEIJIANG_DLTB_SHP", "/path/to/neijiang/DLTB.shp"))
DBF_ENCODING = "gbk"
XIANGZHEN_PATH = Path(os.environ.get("NEIJIANG_XZQ_SHP", "/path/to/neijiang/XZQ.shp"))  # 行政区

OUTPUT_DIR = Path(os.environ.get(
    "NEIJIANG_OUTPUT_DIR",
    str(Path(__file__).resolve().parent / "neijiang_run"),
))
INTERMEDIATE_DIR = OUTPUT_DIR / "intermediate"
OUTPUT_RESULTS_DIR = OUTPUT_DIR / "output"

# DEM tile configuration (Copernicus GLO-30 on AWS S3)
DEM_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"
DEM_PIXEL_SIZE_DEG = 1.0 / 3600  # 1 arc-second in degrees
DEM_TILE_SIZE = 3600  # pixels per tile (1 degree)

# Land use classification codes (Third National Land Survey standard)
FARMLAND_CODES = ['011', '012', '013']  # paddy, irrigated, dry land
FOREST_CODES = ['031', '032', '033']    # forest, shrub, other forest
ORCHARD_CODES = ['021', '022', '023']   # orchard, tea, rubber


def classify_land_use(dlbm):
    """Classify DLBM code into major categories."""
    if dlbm in FARMLAND_CODES:
        return 'Farmland'
    elif dlbm in FOREST_CODES:
        return 'Forest'
    elif dlbm in ORCHARD_CODES:
        return 'Orchard'
    else:
        return 'Other'


# ==============================================================================
# Step 1: Read DLTB layer
# ==============================================================================

def step1_read_parcels():
    """Read DLTB layer from shapefile (Neijiang) and classify land use types.

    Difference vs Bishan: source is .shp (not .gdb), DBF encoding is GBK
    (not UTF-8), and the CRS is EPSG:2359 (projected meters, CM 105E zone 35)
    so we REPROJECT to EPSG:4326 rather than overriding in place.
    """
    print("=" * 70)
    print("STEP 1: Reading DLTB shapefile (Neijiang Dongxing)")
    print("=" * 70)
    t0 = time.time()

    gdf = gpd.read_file(SHP_PATH, encoding=DBF_ENCODING)
    print(f"  Loaded {len(gdf):,} parcels in {time.time()-t0:.1f}s")
    print(f"  Source CRS: {gdf.crs} (EPSG:{gdf.crs.to_epsg()})")

    # Reproject to WGS84 for DEM overlay (DEM tiles are in lat/lon).
    t1 = time.time()
    gdf = gdf.to_crs("EPSG:4326")
    print(f"  Reprojected to EPSG:4326 in {time.time()-t1:.1f}s")

    bounds = gdf.total_bounds
    print(f"  Bounds: Lon [{bounds[0]:.6f}, {bounds[2]:.6f}], "
          f"Lat [{bounds[1]:.6f}, {bounds[3]:.6f}]")

    # Classify land use
    gdf['category'] = gdf['DLBM'].apply(classify_land_use)
    cat_counts = gdf['category'].value_counts()
    print(f"\n  Land use classification:")
    for cat, cnt in cat_counts.items():
        print(f"    {cat:<12} {cnt:>8,} parcels ({cnt/len(gdf)*100:.1f}%)")

    # Save attributes (without geometry) as intermediate CSV
    attrs_path = INTERMEDIATE_DIR / "parcels_attributes.csv"
    gdf.drop(columns='geometry').to_csv(attrs_path, index=False, encoding='utf-8-sig')
    print(f"\n  Saved attributes to {attrs_path}")

    return gdf


# ==============================================================================
# Step 2: Download DEM tiles
# ==============================================================================

def get_required_tiles(bounds):
    """Determine which 1x1 degree DEM tiles are needed for the given bounds."""
    lon_min, lat_min, lon_max, lat_max = bounds
    tiles = []
    for lat in range(int(math.floor(lat_min)), int(math.floor(lat_max)) + 1):
        for lon in range(int(math.floor(lon_min)), int(math.floor(lon_max)) + 1):
            lat_str = f"N{abs(lat):02d}" if lat >= 0 else f"S{abs(lat):02d}"
            lon_str = f"E{abs(lon):03d}" if lon >= 0 else f"W{abs(lon):03d}"
            tile_name = f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM"
            tile_url = f"{DEM_BASE_URL}/{tile_name}/{tile_name}.tif"
            tiles.append({
                'name': tile_name,
                'url': tile_url,
                'lat_origin': lat + 1,  # North edge (DEM origin is top-left)
                'lon_origin': lon,       # West edge
            })
    return tiles


def step2_download_dem(bounds):
    """Download required Copernicus DEM GLO-30 tiles."""
    print("\n" + "=" * 70)
    print("STEP 2: Downloading Copernicus DEM GLO-30 tiles")
    print("=" * 70)

    tiles = get_required_tiles(bounds)
    print(f"  Required tiles: {len(tiles)}")
    for t in tiles:
        print(f"    {t['name']}")

    downloaded_paths = []
    for tile in tiles:
        dem_path = INTERMEDIATE_DIR / f"{tile['name']}.tif"

        if dem_path.exists() and dem_path.stat().st_size > 1_000_000:
            size_mb = dem_path.stat().st_size / (1024 * 1024)
            print(f"\n  Already downloaded: {dem_path.name} ({size_mb:.1f} MB)")
            tile['local_path'] = dem_path
            downloaded_paths.append(tile)
            continue

        print(f"\n  Downloading {tile['name']}...")
        t0 = time.time()
        try:
            r = requests.get(tile['url'], stream=True, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  ERROR: Failed to download {tile['url']}: {e}")
            continue

        total = int(r.headers.get('Content-Length', 0))
        downloaded = 0
        with open(dem_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    dl_mb = downloaded / (1024 * 1024)
                    tot_mb = total / (1024 * 1024)
                    print(f"\r    {dl_mb:.1f}/{tot_mb:.1f} MB ({pct:.0f}%)",
                          end='', flush=True)

        elapsed = time.time() - t0
        size_mb = dem_path.stat().st_size / (1024 * 1024)
        print(f"\n    Downloaded {size_mb:.1f} MB in {elapsed:.1f}s "
              f"({size_mb/elapsed:.1f} MB/s)")

        tile['local_path'] = dem_path
        downloaded_paths.append(tile)

    return downloaded_paths


# ==============================================================================
# Step 3: Read DEM into numpy array
# ==============================================================================

def read_geotiff_with_pillow(tif_path):
    """Read a GeoTIFF file using Pillow, return array and geotransform info."""
    img = Image.open(str(tif_path))
    dem = np.array(img, dtype=np.float32)

    # Parse GeoTIFF tags for georeferencing
    tags = img.tag_v2 if hasattr(img, 'tag_v2') else {}
    pixel_scale = tags.get(33550, None)   # ModelPixelScale
    tiepoint = tags.get(33922, None)      # ModelTiepoint

    info = {
        'pixel_scale': pixel_scale,
        'tiepoint': tiepoint,
    }

    if tiepoint and pixel_scale:
        # Standard GeoTIFF: tiepoint gives (col, row, z, lon, lat, elev)
        info['origin_lon'] = tiepoint[3]
        info['origin_lat'] = tiepoint[4]
        info['pixel_size_x'] = pixel_scale[0]
        info['pixel_size_y'] = pixel_scale[1]
    return dem, info


def step3_read_dem(tiles):
    """Read DEM tiles into a numpy array, mosaic if multiple tiles."""
    print("\n" + "=" * 70)
    print("STEP 3: Reading DEM GeoTIFF into numpy array")
    print("=" * 70)
    t0 = time.time()

    if len(tiles) == 1:
        # Single tile - simple case
        tile = tiles[0]
        dem, geo_info = read_geotiff_with_pillow(tile['local_path'])
        origin_lon = tile['lon_origin']
        origin_lat = tile['lat_origin']
        print(f"  Shape: {dem.shape}, dtype: {dem.dtype}")
        print(f"  Origin: ({origin_lon}E, {origin_lat}N)")
        if geo_info.get('pixel_scale'):
            print(f"  Pixel scale (from tag): {geo_info['pixel_scale']}")
        if geo_info.get('tiepoint'):
            print(f"  Tiepoint (from tag): lon={geo_info.get('origin_lon')}, "
                  f"lat={geo_info.get('origin_lat')}")
    else:
        # Multiple tiles - need to mosaic
        print(f"  Mosaicking {len(tiles)} tiles...")
        # Determine overall extent
        all_lons = [t['lon_origin'] for t in tiles]
        all_lats = [t['lat_origin'] for t in tiles]
        origin_lon = min(all_lons)
        origin_lat = max(all_lats)
        n_cols_tiles = max(all_lons) - min(all_lons) + 1
        n_rows_tiles = max(all_lats) - min(all_lats) + 1
        total_rows = n_rows_tiles * DEM_TILE_SIZE
        total_cols = n_cols_tiles * DEM_TILE_SIZE

        dem = np.full((total_rows, total_cols), np.nan, dtype=np.float32)
        for tile in tiles:
            tile_dem, _ = read_geotiff_with_pillow(tile['local_path'])
            row_offset = (origin_lat - tile['lat_origin']) * DEM_TILE_SIZE
            col_offset = (tile['lon_origin'] - origin_lon) * DEM_TILE_SIZE
            r0, c0 = int(row_offset), int(col_offset)
            dem[r0:r0+DEM_TILE_SIZE, c0:c0+DEM_TILE_SIZE] = tile_dem

        print(f"  Mosaic shape: {dem.shape}")

    # Handle NoData
    nodata_mask = dem < -1000
    n_nodata = nodata_mask.sum()
    if n_nodata > 0:
        dem[nodata_mask] = np.nan
        print(f"  NoData pixels: {n_nodata:,} (set to NaN)")

    elev_valid = dem[~np.isnan(dem)]
    print(f"  Elevation range: {elev_valid.min():.1f} to {elev_valid.max():.1f} m")
    print(f"  Mean elevation: {elev_valid.mean():.1f} m")

    # Build geotransform dict
    geotransform = {
        'origin_lon': float(origin_lon),
        'origin_lat': float(origin_lat),
        'pixel_size': DEM_PIXEL_SIZE_DEG,
        'n_rows': dem.shape[0],
        'n_cols': dem.shape[1],
    }

    # Save intermediate
    npy_path = INTERMEDIATE_DIR / "dem_full_tile.npy"
    np.save(npy_path, dem)
    print(f"\n  Saved to {npy_path} ({npy_path.stat().st_size/1024/1024:.1f} MB)")
    print(f"  Read time: {time.time()-t0:.1f}s")

    return dem, geotransform


# ==============================================================================
# Step 4: Compute slope raster
# ==============================================================================

def step4_compute_slope(dem, geotransform):
    """Compute slope in degrees from DEM using numpy gradient."""
    print("\n" + "=" * 70)
    print("STEP 4: Computing slope raster")
    print("=" * 70)
    t0 = time.time()

    # Compute pixel sizes in meters at study area center latitude
    lat_center = geotransform['origin_lat'] - \
        geotransform['n_rows'] / 2 * geotransform['pixel_size']
    geod = Geod(ellps='WGS84')
    ps = geotransform['pixel_size']

    _, _, dy_m = geod.inv(
        geotransform['origin_lon'], lat_center - ps / 2,
        geotransform['origin_lon'], lat_center + ps / 2
    )
    _, _, dx_m = geod.inv(
        geotransform['origin_lon'] - ps / 2, lat_center,
        geotransform['origin_lon'] + ps / 2, lat_center
    )

    print(f"  Center latitude: {lat_center:.3f}")
    print(f"  Pixel size: {dx_m:.2f} m (E-W) x {dy_m:.2f} m (N-S)")

    # Compute gradients: axis 0 = rows (N-S), axis 1 = cols (E-W)
    dz_dy, dz_dx = np.gradient(dem, dy_m, dx_m)

    # Slope in degrees
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    # Stats (ignoring NaN)
    valid = slope_deg[~np.isnan(slope_deg)]
    print(f"  Slope range: {valid.min():.2f} to {valid.max():.2f} degrees")
    print(f"  Mean slope: {valid.mean():.2f} degrees")
    print(f"  Computed in {time.time()-t0:.2f}s")

    # Save intermediate
    npy_path = INTERMEDIATE_DIR / "slope_degrees.npy"
    np.save(npy_path, slope_deg)
    print(f"  Saved to {npy_path} ({npy_path.stat().st_size/1024/1024:.1f} MB)")

    return slope_deg


# ==============================================================================
# Step 5: Zonal Statistics
# ==============================================================================

def step5_zonal_statistics(slope_deg, geotransform, parcels_gdf):
    """Compute per-parcel slope statistics using STRtree + centroid fallback."""
    print("\n" + "=" * 70)
    print("STEP 5: Computing zonal statistics (per-parcel slope)")
    print("=" * 70)
    t_total = time.time()

    n_parcels = len(parcels_gdf)
    origin_lon = geotransform['origin_lon']
    origin_lat = geotransform['origin_lat']
    ps = geotransform['pixel_size']

    # ---- Phase A: Generate pixel center grid for study area ----
    print("\n  Phase A: Building pixel grid for study area...")
    t0 = time.time()

    lon_min, lat_min, lon_max, lat_max = parcels_gdf.total_bounds

    # Convert bounds to pixel indices (with 1-pixel buffer)
    col_min = max(0, int((lon_min - origin_lon) / ps) - 1)
    col_max = min(geotransform['n_cols'] - 1, int((lon_max - origin_lon) / ps) + 1)
    row_min = max(0, int((origin_lat - lat_max) / ps) - 1)
    row_max = min(geotransform['n_rows'] - 1, int((origin_lat - lat_min) / ps) + 1)

    cols = np.arange(col_min, col_max + 1)
    rows = np.arange(row_min, row_max + 1)
    print(f"    Pixel range: rows [{row_min}, {row_max}], cols [{col_min}, {col_max}]")
    print(f"    Grid size: {len(rows)} x {len(cols)} = {len(rows)*len(cols):,} pixels")

    # Create pixel center coordinates
    col_grid, row_grid = np.meshgrid(cols, rows)
    col_flat = col_grid.ravel()
    row_flat = row_grid.ravel()

    lon_centers = origin_lon + (col_flat + 0.5) * ps
    lat_centers = origin_lat - (row_flat + 0.5) * ps
    n_pixels = len(lon_centers)

    # Create shapely points (vectorized)
    coords = np.column_stack([lon_centers, lat_centers])
    pixel_points = make_points(coords)
    print(f"    Created {n_pixels:,} pixel center points in {time.time()-t0:.1f}s")

    # ---- Phase A: STRtree query ----
    print("\n  Phase A: Running STRtree spatial query...")
    t0 = time.time()

    tree = STRtree(parcels_gdf.geometry.values)
    pixel_idx, parcel_idx = tree.query(pixel_points, predicate='intersects')

    print(f"    STRtree query: {len(pixel_idx):,} pixel-parcel matches "
          f"in {time.time()-t0:.1f}s")

    # Get slope values for matched pixels
    matched_rows = row_flat[pixel_idx]
    matched_cols = col_flat[pixel_idx]
    slope_values = slope_deg[matched_rows, matched_cols]

    # Aggregate per parcel
    t0 = time.time()
    df = pd.DataFrame({
        'parcel_idx': parcel_idx,
        'slope': slope_values
    })

    # Remove NaN slopes before aggregation
    df = df.dropna(subset=['slope'])

    stats = df.groupby('parcel_idx')['slope'].agg(
        slope_mean='mean',
        slope_max='max',
        slope_pixel_count='count'
    )
    n_matched_parcels = len(stats)
    print(f"    Aggregated stats for {n_matched_parcels:,} parcels "
          f"in {time.time()-t0:.2f}s")

    # ---- Phase B: Centroid fallback for unmatched parcels ----
    all_indices = set(range(n_parcels))
    matched_indices = set(stats.index)
    unmatched = sorted(all_indices - matched_indices)
    n_unmatched = len(unmatched)

    print(f"\n  Phase B: Centroid sampling for {n_unmatched:,} unmatched parcels "
          f"({n_unmatched/n_parcels*100:.1f}%)")

    if n_unmatched > 0:
        t0 = time.time()
        unmatched_gdf = parcels_gdf.iloc[unmatched]
        centroids = unmatched_gdf.geometry.representative_point()

        cx = centroids.x.values
        cy = centroids.y.values

        # Convert to continuous pixel coordinates
        sample_col = (cx - origin_lon) / ps - 0.5
        sample_row = (origin_lat - cy) / ps - 0.5

        # Bilinear interpolation
        sampled = map_coordinates(
            slope_deg,
            [sample_row, sample_col],
            order=1, mode='nearest'
        )

        # Add to stats
        centroid_df = pd.DataFrame({
            'slope_mean': sampled,
            'slope_max': sampled,
            'slope_pixel_count': np.zeros(n_unmatched, dtype=int),
        }, index=unmatched)

        stats = pd.concat([stats, centroid_df])
        print(f"    Centroid sampling completed in {time.time()-t0:.2f}s")

    # ---- Join results back to parcels ----
    stats = stats.sort_index()
    parcels_gdf = parcels_gdf.copy()
    parcels_gdf['slope_mean'] = stats['slope_mean'].values
    parcels_gdf['slope_max'] = stats['slope_max'].values
    parcels_gdf['slope_pixel_count'] = stats['slope_pixel_count'].values.astype(int)

    # Check for any remaining NaN
    n_nan = parcels_gdf['slope_mean'].isna().sum()
    if n_nan > 0:
        print(f"  WARNING: {n_nan} parcels still have NaN slope (DEM NoData area)")

    # Save pixel-parcel match table
    match_path = INTERMEDIATE_DIR / "pixel_parcel_matches.csv"
    # Save a summary rather than all 2.6M rows to keep file manageable
    match_summary = df.groupby('parcel_idx').agg(
        n_pixels=('slope', 'count'),
        mean_slope=('slope', 'mean'),
        max_slope=('slope', 'max'),
    )
    match_summary.to_csv(match_path)
    print(f"\n  Saved match summary to {match_path} "
          f"({match_path.stat().st_size/1024/1024:.1f} MB)")

    print(f"\n  Total zonal statistics time: {time.time()-t_total:.1f}s")
    print(f"  Parcels with pixel coverage: {n_matched_parcels:,} ({n_matched_parcels/n_parcels*100:.1f}%)")
    print(f"  Parcels centroid-sampled: {n_unmatched:,} ({n_unmatched/n_parcels*100:.1f}%)")

    return parcels_gdf


# ==============================================================================
# Step 6: CRS Alignment Verification
# ==============================================================================

def step6_verify_alignment(parcels_gdf):
    """Verify CRS alignment using xiangzhen boundaries and slope-landuse correlation."""
    print("\n" + "=" * 70)
    print("STEP 6: CRS Alignment Verification")
    print("=" * 70)

    all_pass = True

    # ---- Test 1: xiangzhen spatial overlay ----
    if XIANGZHEN_PATH.exists():
        print("\n  Test 1: Spatial overlay with XZQ.shp (administrative boundaries)...")
        bounds = parcels_gdf.total_bounds
        # Neijiang XZQ is EPSG:2359 (GBK encoding); read full then reproject.
        xz_raw = gpd.read_file(XIANGZHEN_PATH, encoding=DBF_ENCODING)
        if xz_raw.crs is not None and xz_raw.crs.to_epsg() != 4326:
            xz_raw = xz_raw.to_crs('EPSG:4326')
        xz = xz_raw.cx[bounds[0]-0.1:bounds[2]+0.1, bounds[1]-0.1:bounds[3]+0.1]
        print(f"    Loaded {len(xz)} administrative features in study area")

        # Sample centroids
        np.random.seed(42)
        n_sample = min(3000, len(parcels_gdf))
        sample = parcels_gdf.sample(n_sample).copy()
        sample['geometry'] = sample.geometry.representative_point()

        joined = gpd.sjoin(sample, xz, how='left', predicate='within')
        match_rate = joined.index_right.notna().sum() / len(sample) * 100

        if match_rate > 95:
            print(f"    PASS: {match_rate:.1f}% of sampled centroids fall within WGS84 townships")
        else:
            print(f"    WARNING: Only {match_rate:.1f}% match -- possible CRS offset")
            all_pass = False
    else:
        print("\n  Test 1: SKIPPED (xiangzhen.shp not found)")

    # ---- Test 2: Slope-landuse correlation ----
    print("\n  Test 2: Slope-landuse correlation check...")
    for cat in ['Farmland', 'Forest', 'Orchard', 'Other']:
        mask = parcels_gdf['category'] == cat
        if mask.any():
            slopes = parcels_gdf.loc[mask, 'slope_mean'].dropna()
            if len(slopes) > 0:
                print(f"    {cat:<12}: n={len(slopes):>8,}, "
                      f"mean_slope={slopes.mean():>6.2f}, "
                      f"median={slopes.median():>6.2f}, "
                      f"std={slopes.std():>6.2f}")

    farm_slope = parcels_gdf.loc[
        parcels_gdf['category'] == 'Farmland', 'slope_mean'
    ].dropna().mean()
    forest_slope = parcels_gdf.loc[
        parcels_gdf['category'] == 'Forest', 'slope_mean'
    ].dropna().mean()

    diff = forest_slope - farm_slope
    if diff > 2.0:
        print(f"\n    PASS: Forest slope ({forest_slope:.2f}) > "
              f"Farmland slope ({farm_slope:.2f}), "
              f"difference = {diff:.2f} degrees")
        print(f"    Geographic pattern is correct -- data is properly aligned with DEM")
    elif diff > 0:
        print(f"\n    MARGINAL: Forest slope ({forest_slope:.2f}) > "
              f"Farmland slope ({farm_slope:.2f}), "
              f"but difference only {diff:.2f} degrees")
        all_pass = False
    else:
        print(f"\n    FAIL: Forest slope ({forest_slope:.2f}) <= "
              f"Farmland slope ({farm_slope:.2f})")
        print(f"    WARNING: This suggests possible CRS misalignment!")
        all_pass = False

    if all_pass:
        print(f"\n  OVERALL: ALL CHECKS PASSED -- CRS alignment verified")
    else:
        print(f"\n  OVERALL: SOME CHECKS FAILED -- review alignment carefully")

    return all_pass


# ==============================================================================
# Step 7: Write results
# ==============================================================================

def step7_write_results(parcels_gdf):
    """Write results to GDB and GPKG."""
    print("\n" + "=" * 70)
    print("STEP 7: Writing results")
    print("=" * 70)

    # Prepare output columns (ensure clean types)
    parcels_gdf['slope_mean'] = parcels_gdf['slope_mean'].astype(np.float64)
    parcels_gdf['slope_max'] = parcels_gdf['slope_max'].astype(np.float64)
    parcels_gdf['slope_pixel_count'] = parcels_gdf['slope_pixel_count'].astype(int)

    # Drop the temporary 'category' column for output
    output_gdf = parcels_gdf.copy()

    # ---- Write GDB ----
    gdb_path = OUTPUT_RESULTS_DIR / "DLTB_with_slope.gdb"
    t0 = time.time()
    try:
        output_gdf.to_file(str(gdb_path), layer='DLTB', driver='OpenFileGDB')
        size_mb = sum(
            f.stat().st_size for f in gdb_path.rglob('*') if f.is_file()
        ) / (1024 * 1024)
        print(f"  GDB: {gdb_path} ({size_mb:.1f} MB, {time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  GDB write failed: {e}")
        gdb_path = None

    # ---- Write GPKG (backup) ----
    gpkg_path = OUTPUT_RESULTS_DIR / "DLTB_with_slope.gpkg"
    t0 = time.time()
    try:
        output_gdf.to_file(str(gpkg_path), layer='DLTB', driver='GPKG')
        size_mb = gpkg_path.stat().st_size / (1024 * 1024)
        print(f"  GPKG: {gpkg_path} ({size_mb:.1f} MB, {time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  GPKG write failed: {e}")
        gpkg_path = None

    # ---- Write CSV summary ----
    csv_path = OUTPUT_RESULTS_DIR / "slope_statistics_summary.csv"
    summary = parcels_gdf[['BSM', 'DLBM', 'DLMC', 'QSDWDM', 'TBMJ',
                           'category', 'slope_mean', 'slope_max',
                           'slope_pixel_count']].copy()
    summary.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  CSV: {csv_path} ({csv_path.stat().st_size/1024/1024:.1f} MB)")

    return gdb_path, gpkg_path


# ==============================================================================
# Step 8: Final summary
# ==============================================================================

def step8_summary(parcels_gdf, total_time):
    """Print final summary and comparison with PoC data."""
    print("\n" + "=" * 70)
    print("STEP 8: Final Summary")
    print("=" * 70)

    slopes = parcels_gdf['slope_mean'].dropna()
    print(f"\n  Total parcels: {len(parcels_gdf):,}")
    print(f"  Parcels with slope: {len(slopes):,}")
    print(f"  Missing slope: {parcels_gdf['slope_mean'].isna().sum():,}")

    print(f"\n  Slope distribution (all parcels):")
    print(f"    Min:    {slopes.min():>8.2f} deg")
    print(f"    P5:     {slopes.quantile(0.05):>8.2f} deg")
    print(f"    P25:    {slopes.quantile(0.25):>8.2f} deg")
    print(f"    Median: {slopes.quantile(0.50):>8.2f} deg")
    print(f"    P75:    {slopes.quantile(0.75):>8.2f} deg")
    print(f"    P95:    {slopes.quantile(0.95):>8.2f} deg")
    print(f"    Max:    {slopes.max():>8.2f} deg")
    print(f"    Mean:   {slopes.mean():>8.2f} deg")
    print(f"    Std:    {slopes.std():>8.2f} deg")

    print(f"\n  Slope by land use category:")
    print(f"    {'Category':<12} {'Count':>8} {'Mean':>8} {'Median':>8} {'Max':>8}")
    print(f"    {'-'*48}")
    for cat in ['Farmland', 'Forest', 'Orchard', 'Other']:
        mask = parcels_gdf['category'] == cat
        if mask.any():
            s = parcels_gdf.loc[mask, 'slope_mean'].dropna()
            print(f"    {cat:<12} {len(s):>8,} {s.mean():>8.2f} "
                  f"{s.median():>8.2f} {s.max():>8.2f}")

    # Comparison with PoC data (Banzhucun)
    print(f"\n  Comparison with PoC (Banzhucun village):")
    print(f"    {'Metric':<25} {'PoC':>12} {'Real Data':>12}")
    print(f"    {'-'*50}")
    print(f"    {'Total parcels':<25} {'10,653':>12} {len(parcels_gdf):>12,}")
    print(f"    {'Mean slope (deg)':<25} {'12.43':>12} {slopes.mean():>12.2f}")
    print(f"    {'Slope std (deg)':<25} {'7.44':>12} {slopes.std():>12.2f}")
    print(f"    {'Slope range':<25} {'0-44.5':>12} "
          f"{f'{slopes.min():.1f}-{slopes.max():.1f}':>12}")

    # Steep slope statistics (relevant for PhD: >25 deg mandatory retirement)
    steep_mask = parcels_gdf['slope_mean'] > 25
    steep_farm = (parcels_gdf['category'] == 'Farmland') & steep_mask
    print(f"\n  Steep slope analysis (>25 deg policy threshold):")
    print(f"    Total parcels > 25 deg: {steep_mask.sum():,}")
    print(f"    Farmland parcels > 25 deg: {steep_farm.sum():,} "
          f"(mandatory retirement candidates)")

    print(f"\n  Total pipeline time: {total_time:.1f}s")

    print("\n" + "=" * 70)
    print("  OUTPUT FILES:")
    print("=" * 70)
    for f in sorted(INTERMEDIATE_DIR.glob('*')):
        print(f"  [intermediate] {f.name} ({f.stat().st_size/1024/1024:.1f} MB)")
    for f in sorted(OUTPUT_RESULTS_DIR.iterdir()):
        if f.is_file():
            print(f"  [output]       {f.name} ({f.stat().st_size/1024/1024:.1f} MB)")
        elif f.is_dir():
            size = sum(x.stat().st_size for x in f.rglob('*') if x.is_file())
            print(f"  [output]       {f.name}/ ({size/1024/1024:.1f} MB)")


# ==============================================================================
# Main
# ==============================================================================

def main():
    print()
    print("*" * 70)
    print("*  Copernicus DEM GLO-30 Slope Zonal Statistics Pipeline")
    print("*  Input: DLTB layer from GDB (101,657 parcels)")
    print("*  Output: Per-parcel slope_mean and slope_max")
    print("*" * 70)
    print()

    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)

    t_start = time.time()

    # Step 1: Read parcels
    parcels_gdf = step1_read_parcels()

    # Step 2: Download DEM
    bounds = parcels_gdf.total_bounds
    tiles = step2_download_dem(bounds)
    if not tiles:
        print("ERROR: No DEM tiles downloaded. Aborting.")
        sys.exit(1)

    # Step 3: Read DEM
    dem, geotransform = step3_read_dem(tiles)

    # Step 4: Compute slope
    slope_deg = step4_compute_slope(dem, geotransform)

    # Step 5: Zonal statistics
    parcels_gdf = step5_zonal_statistics(slope_deg, geotransform, parcels_gdf)

    # Step 6: Verify CRS alignment
    step6_verify_alignment(parcels_gdf)

    # Step 7: Write results
    step7_write_results(parcels_gdf)

    # Step 8: Summary
    total_time = time.time() - t_start
    step8_summary(parcels_gdf, total_time)


if __name__ == '__main__':
    main()
