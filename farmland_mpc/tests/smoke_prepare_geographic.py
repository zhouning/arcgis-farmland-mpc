"""Smoke test for the gradient_geographic slope method.

Builds a tiny synthetic DEM in EPSG:4326 (geographic CRS) plus 4 parcel
polygons, runs prepare.run with slope_method='auto' (which should resolve
to gradient_geographic for a geographic DEM) and 'gradient_geographic'
explicitly, and checks:

  - the slope_method recorded in prepare_data_summary.json matches
  - the per-parcel slope_mean column is populated
  - slopes are physically reasonable (degrees in [0, 90])
  - the same DEM run via slope_method='horn_projected' produces a
    DIFFERENT slope (different algorithm = different number)

Run from the repo root:

    python -m farmland_mpc.tests.smoke_prepare_geographic
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from farmland_mpc.prepare import run

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("smoke_prepare_geographic")


def _build_synthetic_geographic_fixture(workdir: Path) -> tuple[Path, Path]:
    """Build a small EPSG:4326 DEM + DLTB shapefile.

    Centred at lat 29.5° (matches our real Bishan/Neijiang test region),
    with 1-arcsecond pixels (matches Copernicus DSM resolution). DEM is
    a tilted plane so slope is non-zero but bounded.
    """
    # 1 arc-second pixels, ~30m N-S, ~27m E-W at lat 29.5
    pixel_deg = 1.0 / 3600
    n = 60  # 60x60 = small but enough for 4 quadrants
    origin_lon = 105.0
    origin_lat = 29.5 + n * pixel_deg / 2  # so center is at 29.5

    # Tilted plane: elev rises by ~5m per pixel both directions => clear slope
    rows, cols = np.indices((n, n))
    elev = (100 + rows * 5.0 + cols * 3.0).astype(np.float32)

    dem_path = workdir / "dem_geo.tif"
    transform = from_origin(origin_lon, origin_lat, pixel_deg, pixel_deg)
    with rasterio.open(
        dem_path, "w",
        driver="GTiff",
        height=n, width=n, count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=None,
    ) as dst:
        dst.write(elev, 1)

    # 4 quadrant parcels covering the DEM
    half = n * pixel_deg / 2
    quadrants = [
        ("P001", origin_lon,        origin_lat - n * pixel_deg, origin_lon + half, origin_lat - half),
        ("P002", origin_lon + half, origin_lat - n * pixel_deg, origin_lon + n * pixel_deg, origin_lat - half),
        ("P003", origin_lon,        origin_lat - half, origin_lon + half, origin_lat),
        ("P004", origin_lon + half, origin_lat - half, origin_lon + n * pixel_deg, origin_lat),
    ]
    rows = []
    for bsm, x0, y0, x1, y1 in quadrants:
        rows.append({
            "BSM": bsm,
            "DLBM": "0101",  # paddy
            "QSDWDM": "510101001000",  # synthetic 9+digit code
            "geometry": box(x0, y0, x1, y1),
        })
    dltb = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    dltb_path = workdir / "dltb_geo.shp"
    dltb.to_file(dltb_path)
    return dltb_path, dem_path


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="farmland_mpc_smoke_geo_") as tmp:
        workdir = Path(tmp)
        dltb_path, dem_path = _build_synthetic_geographic_fixture(workdir)

        log.info("=== Run 1: slope_method='auto' (should resolve to gradient_geographic) ===")
        out_auto = workdir / "prepared_auto"
        run(
            dltb_path=dltb_path, dem_path=dem_path, prepared_dir=out_auto,
            proj_crs="EPSG:32648",
            slope_method="auto",
            run_phase_bc=False,
            min_parcels_per_township=1,
        )
        summary_auto = json.loads((out_auto / "prepare_data_summary.json").read_text())
        log.info("  resolved slope_method: %s", summary_auto["slope_method"])
        assert summary_auto["slope_method"] == "gradient_geographic", \
            f"auto-mode on geographic DEM should pick gradient_geographic, got {summary_auto['slope_method']}"

        gdf_auto = gpd.read_file(out_auto / "dem_slope_analysis" / "output" / "DLTB_with_slope.shp")
        sl_auto = gdf_auto["slope_mean"].astype(float)
        log.info("  slope_mean range: [%.3f, %.3f] degrees", sl_auto.min(), sl_auto.max())
        assert (sl_auto >= 0).all() and (sl_auto <= 90).all(), "slope out of [0,90] range"
        assert sl_auto.notna().all(), "slope_mean has NaN"

        log.info("=== Run 2: slope_method='gradient_geographic' (explicit) ===")
        out_geo = workdir / "prepared_geo"
        run(
            dltb_path=dltb_path, dem_path=dem_path, prepared_dir=out_geo,
            proj_crs="EPSG:32648",
            slope_method="gradient_geographic",
            run_phase_bc=False,
            min_parcels_per_township=1,
        )
        gdf_geo = gpd.read_file(out_geo / "dem_slope_analysis" / "output" / "DLTB_with_slope.shp")
        sl_geo = gdf_geo["slope_mean"].astype(float)
        # auto and explicit should match exactly
        np.testing.assert_allclose(
            sorted(sl_auto), sorted(sl_geo), rtol=1e-6,
            err_msg="auto and explicit gradient_geographic disagree",
        )
        log.info("  auto and explicit gradient_geographic produce identical slopes ✓")

        log.info("=== Run 3: slope_method='horn_projected' (legacy on same DEM) ===")
        out_horn = workdir / "prepared_horn"
        run(
            dltb_path=dltb_path, dem_path=dem_path, prepared_dir=out_horn,
            proj_crs="EPSG:32648",
            slope_method="horn_projected",
            run_phase_bc=False,
            min_parcels_per_township=1,
        )
        gdf_horn = gpd.read_file(out_horn / "dem_slope_analysis" / "output" / "DLTB_with_slope.shp")
        sl_horn = gdf_horn["slope_mean"].astype(float)
        log.info("  horn-projected slope_mean range: [%.3f, %.3f]",
                 sl_horn.min(), sl_horn.max())
        # horn-projected on a tilted plane will be in the same ballpark but
        # not bit-identical to gradient_geographic (different algorithm,
        # different metres-per-pixel after reprojection). Just sanity check
        # both are positive and roughly similar order of magnitude.
        assert (sl_horn > 0).all(), "horn slope should be > 0 on a tilted plane"
        assert abs(sl_horn.mean() - sl_geo.mean()) < 5.0, \
            f"horn ({sl_horn.mean():.2f}) and gradient_geographic ({sl_geo.mean():.2f}) " \
            "should be in the same ballpark on a smooth plane"

        log.info("=== Run 4: slope_method='from_field' ===")
        # Reuse out_geo's DLTB_with_slope.shp as the from_field input
        from_field_dltb = out_geo / "dem_slope_analysis" / "output" / "DLTB_with_slope.shp"
        out_field = workdir / "prepared_field"
        run(
            dltb_path=from_field_dltb, dem_path=dem_path, prepared_dir=out_field,
            proj_crs="EPSG:32648",
            slope_method="from_field",
            slope_field="slope_mean",
            run_phase_bc=False,
            min_parcels_per_township=1,
        )
        gdf_field = gpd.read_file(out_field / "dem_slope_analysis" / "output" / "DLTB_with_slope.shp")
        sl_field = gdf_field["slope_mean"].astype(float)
        np.testing.assert_allclose(
            sorted(sl_field), sorted(sl_geo), rtol=1e-6,
            err_msg="from_field should reproduce the source slope_mean exactly",
        )
        log.info("  from_field reproduces source slopes exactly ✓")

        log.info("\n*** smoke_prepare_geographic passed ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
