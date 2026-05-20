"""Pure-Python data-preparation pipeline (no arcpy dependency).

This module replaces the arcpy-based ``_phase_a_arcgis`` and ancillary
helpers from the ArcGIS Pro toolbox with rasterio + geopandas equivalents.

The output schema matches the toolbox version exactly, so downstream
``sample_transitions`` / ``train_ensemble`` / ``mpc_plan`` modules consume
the same files regardless of whether preparation ran through arcpy or
through this open-source path.

Output layout under ``prepared_dir`` (identical to toolbox v1.2):

    dem_slope_analysis/output/DLTB_with_slope.shp     ('slope_mean' field)
    townships.json                                    (code -> label)
    prepare_data_summary.json                         (provenance)

CRS handling: ``proj_crs`` accepts ``"EPSG:nnnn"``, a raw WKT string, or
a PROJ-string. The internal pipeline goes through ``pyproj.CRS.from_user_input``
which handles all three. EPSG lookups depend on the local proj.db being
current; if your PROJ database is stale (e.g. ArcGIS-bundled version 5
where rasterio expects 6+), pass the WKT directly to bypass the database.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import rasterize
from rasterio import windows
from pyproj import CRS

logger = logging.getLogger(__name__)


# =============================================================================
# Public entry point
# =============================================================================
def run(
    dltb_path: str | Path,
    dem_path: str | Path,
    prepared_dir: str | Path,
    *,
    proj_crs: str = "EPSG:32648",
    dlbm_field: str = "DLBM",
    qsdwdm_field: str = "QSDWDM",
    bsm_field: str = "BSM",
    dem_resampling: str = "bilinear",
) -> Path:
    """End-to-end Phase A: DEM -> per-parcel slope_mean -> DLTB_with_slope.shp.

    Parameters
    ----------
    dltb_path : str | Path
        Polygon vector file (any format readable by ``geopandas.read_file``:
        shapefile, GeoPackage, FlatGeobuf, GeoJSON).
    dem_path : str | Path
        DEM raster (any format readable by ``rasterio.open``).
    prepared_dir : str | Path
        Output root directory.
    proj_crs : str
        Target projected CRS in ``EPSG:nnnn`` form. Defaults to
        ``EPSG:32648`` (UTM Zone 48N) covering central/western China. For
        deployments outside this zone, override with the correct UTM zone.
    dlbm_field, qsdwdm_field, bsm_field : str
        Column names in the DLTB file. Defaults match the Third National
        Land Survey schema.
    dem_resampling : str
        Resampling method when the DEM has to be reprojected to ``proj_crs``.
        ``"bilinear"`` is recommended for elevation rasters; ``"nearest"``
        for categorical rasters (not applicable here).

    Returns
    -------
    Path to the written ``DLTB_with_slope.shp``.
    """
    t0 = time.time()
    prepared_dir = Path(prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    out_dir = prepared_dir / "dem_slope_analysis" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    shp_out = out_dir / "DLTB_with_slope.shp"

    logger.info("Phase A: pure-Python DEM->slope pipeline")
    logger.info("  DLTB:        %s", dltb_path)
    logger.info("  DEM:         %s", dem_path)
    logger.info("  prepared:    %s", prepared_dir)
    logger.info("  target CRS:  %s",
                str(proj_crs)[:80] + "..." if len(str(proj_crs)) > 80 else proj_crs)

    target_crs = CRS.from_user_input(proj_crs)
    target_epsg = target_crs.to_epsg()  # may be None for WKT-only CRS
    target_crs_label = (
        f"EPSG:{target_epsg}" if target_epsg is not None else "(custom WKT)"
    )

    # Load DLTB
    dltb = gpd.read_file(dltb_path)
    _validate_dltb_columns(dltb, [bsm_field, dlbm_field, qsdwdm_field])
    if dltb.crs is None:
        raise ValueError(f"DLTB has no CRS defined: {dltb_path}")
    if not _crs_equals(dltb.crs, target_crs):
        logger.info("  Projecting DLTB %s -> %s ...", dltb.crs, target_crs_label)
        dltb = dltb.to_crs(target_crs)
    logger.info("  DLTB rows: %d", len(dltb))

    # Reproject DEM to target CRS in a temp .tif
    dem_proj_path = out_dir / "_dem_reproj.tif"
    _reproject_dem(dem_path, dem_proj_path, target_crs, resampling=dem_resampling)

    # Compute slope (degrees) via Horn 3x3
    slope_path = out_dir / "_slope_deg.tif"
    _compute_slope_horn(dem_proj_path, slope_path)

    # Per-polygon zonal mean
    logger.info("  Running zonal mean on %d parcels ...", len(dltb))
    slope_means = _zonal_mean(dltb, slope_path)
    n_unmatched = int(np.isnan(slope_means).sum())
    if n_unmatched:
        logger.warning(
            "  %d parcels have no slope_mean (outside DEM coverage or tiny polygons)",
            n_unmatched,
        )
    dltb["slope_mean"] = slope_means

    # Drop intermediate rasters before exporting (we keep them under
    # _dem_reproj.tif / _slope_deg.tif for debugging; remove if you want
    # a slim prepared_dir)
    # _dem_reproj.tif and _slope_deg.tif are intentionally kept; the
    # toolbox writes equivalents to scratchGDB. They can be deleted by
    # passing keep_intermediate=False at a future API extension.

    # Write DLTB_with_slope.shp; geopandas truncates to <=10 char DBF names
    # so we explicitly check that the schema is shapefile-safe.
    dltb_export = _trim_to_shapefile_schema(dltb)
    if shp_out.exists():
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            try:
                (shp_out.with_suffix(ext)).unlink()
            except FileNotFoundError:
                pass
    dltb_export.to_file(shp_out, driver="ESRI Shapefile", encoding="utf-8")
    logger.info("  Wrote %s (%d rows)", shp_out, len(dltb_export))

    # Provenance summary
    elapsed = time.time() - t0
    summary = {
        "phase_a_backend": "open_source_rasterio_geopandas",
        "dltb_input": str(dltb_path),
        "dem_input": str(dem_path),
        "proj_crs": target_crs_label,
        "n_parcels": int(len(dltb_export)),
        "n_parcels_with_slope": int((~np.isnan(slope_means)).sum()),
        "n_parcels_unmatched": n_unmatched,
        "elapsed_seconds": round(elapsed, 2),
    }
    summary_path = prepared_dir / "prepare_data_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    logger.info("  Phase A done in %.1fs", elapsed)
    return shp_out


# =============================================================================
# Helpers
# =============================================================================
def _parse_epsg(proj_crs: str) -> int:
    """Parse 'EPSG:nnnn' into the integer code (legacy helper, kept for tests)."""
    s = str(proj_crs).strip()
    if s.upper().startswith("EPSG:"):
        s = s.split(":", 1)[1]
    try:
        return int(s)
    except ValueError as e:
        raise ValueError(f"Cannot parse proj_crs={proj_crs!r}; expected 'EPSG:nnnn'") from e


def _crs_equals(a: CRS | str, b: CRS | str) -> bool:
    """Robust CRS comparison that tolerates EPSG / WKT / proj-string forms."""
    ca = a if isinstance(a, CRS) else CRS.from_user_input(a)
    cb = b if isinstance(b, CRS) else CRS.from_user_input(b)
    try:
        if ca.to_epsg() is not None and cb.to_epsg() is not None:
            return ca.to_epsg() == cb.to_epsg()
    except Exception:
        pass
    return ca.equals(cb)


def _validate_dltb_columns(dltb: gpd.GeoDataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in dltb.columns]
    if missing:
        raise ValueError(
            f"DLTB missing required columns: {missing}. "
            f"Available: {list(dltb.columns)}"
        )


def _reproject_dem(
    dem_path: str | Path,
    out_path: str | Path,
    target_crs: CRS | str | int,
    *,
    resampling: str = "bilinear",
) -> None:
    """Project ``dem_path`` to ``target_crs`` and write to ``out_path``.

    If the source DEM is already in the target CRS, this still re-writes
    it (cheap copy) so downstream code can rely on a single output path.
    """
    resampling_map = {
        "bilinear": Resampling.bilinear,
        "nearest": Resampling.nearest,
        "cubic": Resampling.cubic,
    }
    if resampling not in resampling_map:
        raise ValueError(f"Unsupported resampling={resampling!r}")
    rs = resampling_map[resampling]

    target_crs = target_crs if isinstance(target_crs, CRS) else CRS.from_user_input(target_crs)

    with rasterio.open(dem_path) as src:
        if src.crs is None:
            raise ValueError(f"DEM has no CRS defined: {dem_path}")
        src_crs = src.crs
        if _crs_equals(src_crs, target_crs):
            logger.info("  DEM already in target CRS; copying without resampling")
            data = src.read()
            profile = src.profile.copy()
            profile["crs"] = target_crs.to_wkt()
        else:
            logger.info(
                "  Reprojecting DEM %s -> %s (%s) ...",
                src_crs, target_crs, resampling,
            )
            transform, width, height = calculate_default_transform(
                src_crs, target_crs.to_wkt(),
                src.width, src.height, *src.bounds,
            )
            profile = src.profile.copy()
            profile.update({
                "crs": target_crs.to_wkt(),
                "transform": transform,
                "width": width,
                "height": height,
            })
            data = np.empty((src.count, height, width), dtype=src.dtypes[0])
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=data[i - 1],
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=target_crs.to_wkt(),
                    resampling=rs,
                )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)


def _compute_slope_horn(dem_path: str | Path, slope_out_path: str | Path) -> None:
    """Compute slope (degrees) using Horn's 3x3 algorithm.

    Equivalent to ``arcpy.sa.Slope(dem, "DEGREE")``.

    References: Horn, B. K. P. (1981). Hill shading and the reflectance map.
    Proceedings of the IEEE, 69(1), 14-47.
    """
    with rasterio.open(dem_path) as src:
        if src.count != 1:
            raise ValueError(f"DEM must be single-band; got {src.count} bands")
        z = src.read(1).astype(np.float64)
        cellsize_x = abs(src.transform.a)
        cellsize_y = abs(src.transform.e)
        nodata = src.nodata

        # Mark NoData as NaN so Horn ignores them via padding logic
        if nodata is not None:
            z = np.where(z == nodata, np.nan, z)

        # Horn's slope algorithm: each cell's slope is computed from a 3x3
        # window with weights [[1,2,1],[0,0,0],[-1,-2,-1]] for dz/dy and
        # the transpose for dz/dx, then slope = atan(sqrt((dz/dx)^2+(dz/dy)^2)).
        # We use np.gradient as a numerically equivalent shortcut for the
        # interior; near edges we rely on np.gradient's edge-mode (forward
        # differences), which differs from arcpy at the very outer ring of
        # cells but agrees everywhere else. For typical county-scale DEMs
        # this is below the precision of zonal averaging.
        # For exact arcpy parity we implement the Horn weights explicitly:
        slope_deg = _horn_slope_deg(z, cellsize_x, cellsize_y)

        profile = src.profile.copy()
        profile.update({"dtype": "float32", "nodata": -9999.0, "count": 1})
        slope_arr = np.where(np.isnan(slope_deg), -9999.0, slope_deg).astype("float32")
        with rasterio.open(slope_out_path, "w", **profile) as dst:
            dst.write(slope_arr, 1)
    logger.info("  Slope raster (degrees) -> %s", slope_out_path)


def _horn_slope_deg(z: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Horn 3x3 slope (degrees), matching arcpy.sa.Slope semantics."""
    # Pad with edge values so the convolution doesn't shrink the output
    pad = np.pad(z, 1, mode="edge")
    # 3x3 windows
    a = pad[0:-2, 0:-2]; b = pad[0:-2, 1:-1]; c = pad[0:-2, 2:]
    d = pad[1:-1, 0:-2]; e = pad[1:-1, 1:-1]; f = pad[1:-1, 2:]
    g = pad[2:,   0:-2]; h = pad[2:,   1:-1]; i = pad[2:,   2:]

    # Horn's algorithm: dz/dx, dz/dy with [1,2,1] weights
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * dx)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * dy)

    rise_run = np.sqrt(dzdx * dzdx + dzdy * dzdy)
    slope_rad = np.arctan(rise_run)
    slope_deg = np.degrees(slope_rad)
    # Where any of the 9 inputs was NaN, the slope is undefined
    nan_mask = (
        np.isnan(a) | np.isnan(b) | np.isnan(c)
        | np.isnan(d) | np.isnan(e) | np.isnan(f)
        | np.isnan(g) | np.isnan(h) | np.isnan(i)
    )
    return np.where(nan_mask, np.nan, slope_deg)


def _zonal_mean(
    polygons: gpd.GeoDataFrame,
    raster_path: str | Path,
) -> np.ndarray:
    """Per-polygon mean of a raster, equivalent to arcpy.sa.ZonalStatisticsAsTable.

    Returns a length-N float array with NaN where a polygon has no
    valid raster cells (outside coverage, all-nodata, or tiny polygons
    that don't intersect any cell centre).
    """
    n = len(polygons)
    means = np.full(n, np.nan, dtype=np.float64)

    # Precompute polygon FID -> integer zone id (1..n) for rasterize()
    zone_ids = np.arange(1, n + 1, dtype=np.int32)
    geom_iter = ((geom, zid) for geom, zid in zip(polygons.geometry.values, zone_ids))

    with rasterio.open(raster_path) as src:
        slope = src.read(1)
        nodata = src.nodata
        valid = slope != nodata if nodata is not None else np.ones_like(slope, dtype=bool)
        valid &= ~np.isnan(slope)

        # Rasterize all polygons in one pass, using dtype large enough for n
        if n < 2**16 - 1:
            zone_dtype = "uint16"
        else:
            zone_dtype = "int32"

        zone_raster = rasterize(
            shapes=geom_iter,
            out_shape=slope.shape,
            transform=src.transform,
            fill=0,
            dtype=zone_dtype,
            all_touched=False,
        )

        # Aggregate: for each zone id, mean of slope[valid & zone==id]
        # We use np.bincount for O(N) aggregation across all cells
        flat_zones = zone_raster.ravel()
        flat_slope = slope.ravel()
        flat_valid = valid.ravel() & (flat_zones > 0)

        if not np.any(flat_valid):
            logger.warning("  Zonal mean: no valid (zone, slope) pairs found")
            return means

        zone_sel = flat_zones[flat_valid].astype(np.int64)
        slope_sel = flat_slope[flat_valid].astype(np.float64)
        sums = np.bincount(zone_sel, weights=slope_sel, minlength=n + 1)
        counts = np.bincount(zone_sel, minlength=n + 1)
        with np.errstate(invalid="ignore"):
            zone_means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        # Map zone_id (1..n) back to polygon index (0..n-1)
        means = zone_means[1 : n + 1]
    return means


def _trim_to_shapefile_schema(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Truncate field names to <=10 chars (DBF limit) and warn on collisions.

    Mirrors the toolbox helper ``_build_shp_safe_field_mappings``.
    """
    rename_map: dict[str, str] = {}
    used_targets: set[str] = set()
    for col in gdf.columns:
        if col == "geometry":
            continue
        target = col[:10]
        if target in used_targets:
            # Resolve collision by appending a digit
            for k in range(1, 10):
                cand = f"{col[: 10 - len(str(k))]}{k}"
                if cand not in used_targets:
                    target = cand
                    break
        if target != col:
            logger.warning("  DBF schema: '%s' truncated to '%s'", col, target)
        used_targets.add(target)
        rename_map[col] = target
    if not rename_map:
        return gdf
    return gdf.rename(columns=rename_map)
