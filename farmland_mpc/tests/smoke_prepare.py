"""Synthetic toy fixture + smoke test for farmland_mpc.prepare.

Generates a small DEM raster and a 4-polygon DLTB shapefile, runs
prepare.run, then asserts that:
  - DLTB_with_slope.shp is created
  - the slope_mean column is populated for every polygon
  - slope values are physically reasonable (degrees in [0, 90])
  - the prepare_data_summary.json provenance file is written
  - re-running prepare overwrites the previous shapefile cleanly

Run from the repo root (D:/test/_publish/arcgis-farmland-mpc/) with any
Python that has geopandas + rasterio + numpy:

    python -m farmland_mpc.tests.smoke_prepare

CRS is encoded as a PROJ-string (UTM Zone 48N on WGS84), bypassing the
EPSG-database lookup in case the local PROJ database is out of date or
unavailable. The EPSG numeric code 32648 is resolved later via the same
PROJ-string for downstream prepare.run() calls.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("smoke_prepare")

# UTM Zone 48N on WGS84 (equivalent to EPSG:32648). We write the PROJ-string
# directly so that rasterio / pyproj don't have to consult a possibly-outdated
# proj.db.
CRS_WKT = (
    'PROJCRS["WGS 84 / UTM zone 48N",'
    'BASEGEOGCRS["WGS 84",'
    'DATUM["World Geodetic System 1984",'
    'ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],'
    'PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]]],'
    'CONVERSION["UTM zone 48N",'
    'METHOD["Transverse Mercator",ID["EPSG",9807]],'
    'PARAMETER["Latitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433]],'
    'PARAMETER["Longitude of natural origin",105,ANGLEUNIT["degree",0.0174532925199433]],'
    'PARAMETER["Scale factor at natural origin",0.9996,SCALEUNIT["unity",1]],'
    'PARAMETER["False easting",500000,LENGTHUNIT["metre",1]],'
    'PARAMETER["False northing",0,LENGTHUNIT["metre",1]]],'
    'CS[Cartesian,2],'
    'AXIS["easting",east,ORDER[1],LENGTHUNIT["metre",1]],'
    'AXIS["northing",north,ORDER[2],LENGTHUNIT["metre",1]]]'
)
CRS_FOR_PREPARE = CRS_WKT  # passed through to prepare.run as proj_crs


def _make_synthetic_dem(path: Path, *, width: int = 100, height: int = 100,
                       cellsize_m: float = 30.0) -> None:
    """Synthesise a tilted-plane DEM with a small bump near the top-right corner."""
    xs = np.arange(width, dtype=np.float64) * cellsize_m
    ys = np.arange(height, dtype=np.float64) * cellsize_m
    xx, yy = np.meshgrid(xs, ys)
    z = 100.0 + 0.05 * xx + 0.02 * yy
    bump = 50.0 * np.exp(-((xx - 0.75 * width * cellsize_m) ** 2
                           + (yy - 0.25 * height * cellsize_m) ** 2)
                         / (2 * (10 * cellsize_m) ** 2))
    z = z + bump

    origin_x, origin_y = 500000.0, 4400000.0
    transform = from_origin(origin_x, origin_y, cellsize_m, cellsize_m)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": -9999.0,
        "width": width, "height": height, "count": 1,
        "crs": CRS_WKT,
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(z.astype("float32"), 1)


def _make_synthetic_dltb(path: Path, *,
                        origin_x: float = 500000.0, origin_y: float = 4400000.0,
                        cellsize_m: float = 30.0, width: int = 100, height: int = 100) -> None:
    """Build a 4-polygon DLTB shapefile, each polygon covering a quadrant of the DEM."""
    half_x = origin_x + (width // 2) * cellsize_m
    half_y_top = origin_y - (height // 2) * cellsize_m
    bottom_y = origin_y - height * cellsize_m
    right_x = origin_x + width * cellsize_m

    polys = [
        box(origin_x, half_y_top, half_x, origin_y),    # top-left
        box(half_x, half_y_top, right_x, origin_y),     # top-right (where the bump is)
        box(origin_x, bottom_y, half_x, half_y_top),    # bottom-left
        box(half_x, bottom_y, right_x, half_y_top),     # bottom-right
    ]
    rows = [
        {"BSM": "P001", "DLBM": "011", "QSDWDM": "500227001", "geometry": polys[0]},
        {"BSM": "P002", "DLBM": "011", "QSDWDM": "500227001", "geometry": polys[1]},
        {"BSM": "P003", "DLBM": "012", "QSDWDM": "500227002", "geometry": polys[2]},
        {"BSM": "P004", "DLBM": "012", "QSDWDM": "500227002", "geometry": polys[3]},
    ]
    gdf = gpd.GeoDataFrame(rows, crs=CRS_WKT)
    gdf.to_file(path, driver="ESRI Shapefile", encoding="utf-8")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="farmland_mpc_smoke_"))
    log.info("smoke fixture root: %s", tmp)

    dem_path = tmp / "dem.tif"
    dltb_path = tmp / "dltb.shp"
    prepared_dir = tmp / "prepared"

    _make_synthetic_dem(dem_path)
    _make_synthetic_dltb(dltb_path)
    log.info("synthetic DEM + DLTB written")

    from farmland_mpc.prepare import run

    out_shp = run(
        dltb_path=dltb_path,
        dem_path=dem_path,
        prepared_dir=prepared_dir,
        proj_crs=CRS_FOR_PREPARE,
    )

    # Assertion 1: output shapefile exists
    assert out_shp.exists(), f"expected output not produced: {out_shp}"

    # Assertion 2: slope_mean is populated for every polygon
    out = gpd.read_file(out_shp)
    assert "slope_mean" in out.columns, f"slope_mean column missing; got {list(out.columns)}"
    n_missing = out["slope_mean"].isna().sum()
    assert n_missing == 0, f"{n_missing}/{len(out)} polygons missing slope_mean"

    # Assertion 3: slope values are in [0, 90]
    smin, smax = out["slope_mean"].min(), out["slope_mean"].max()
    assert 0.0 <= smin <= smax <= 90.0, f"slope_mean out of [0,90]: [{smin}, {smax}]"
    log.info("slope_mean range: [%.3f, %.3f] degrees", smin, smax)

    # Assertion 4: top-right quadrant (where the bump sits) has the highest mean slope
    by_bsm = dict(zip(out["BSM"], out["slope_mean"]))
    log.info("per-quadrant slope_mean: %s", {k: round(v, 3) for k, v in by_bsm.items()})
    assert by_bsm["P002"] > by_bsm["P001"], (
        "top-right (bump) quadrant should have higher mean slope than top-left; "
        f"got P002={by_bsm['P002']:.3f} vs P001={by_bsm['P001']:.3f}"
    )

    # Assertion 5: provenance summary written
    summary_path = prepared_dir / "prepare_data_summary.json"
    assert summary_path.exists(), f"missing provenance summary: {summary_path}"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_parcels"] == 4
    assert summary["n_parcels_with_slope"] == 4
    assert summary["n_parcels_unmatched"] == 0
    log.info("provenance summary: %s", summary)

    # Assertion 6: re-run is idempotent (overwrites cleanly)
    out_shp_2 = run(
        dltb_path=dltb_path, dem_path=dem_path, prepared_dir=prepared_dir,
        proj_crs=CRS_FOR_PREPARE,
    )
    assert out_shp_2 == out_shp
    log.info("re-run produced identical path: %s", out_shp_2)

    log.info("smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
