"""Generate a synthetic DLTB shapefile + matching DEM for end-to-end testing.

Useful when you want to demo / benchmark the pipeline without sharing real
DLTB data. Output is shaped like a real county: multiple townships, a tilted
DEM with a few hills, mixed farmland / forest / barriers (roads + water).

Why this exists: tests/smoke_end_to_end.py builds a 36-parcel fixture, but
that's too small for the contrastive ensemble to find meaningful signal
(reward std collapses to ~0). This generator hits a 5k-50k parcel sweet
spot where Tool 2 reports non-zero reward variance and Tool 3 / 4 produce
non-trivial slope improvements.

Output:
  <out>/dltb.shp + .dbf + .shx + .prj + .cpg
  <out>/dem.tif

Schema matches Third-Survey DLTB: BSM / DLBM / DLMC / QSDWDM / TBMJ /
SHAPE_Leng / SHAPE_Area, plus geometry. CRS defaults to EPSG:32648 (UTM 48N
covering central / western China).

Usage:
  python scripts/make_synthetic_dltb.py \\
      --grid 80 --townships 4 --out /tmp/synth_5k

  # Then:
  farmland-mpc prepare --dltb /tmp/synth_5k/dltb.shp \\
                       --dem  /tmp/synth_5k/dem.tif \\
                       --out  /tmp/synth_5k/prepared --crs EPSG:32648
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box


CELL_M = 100.0  # 100 m × 100 m parcels = 1 ha each
DEM_RES = 30.0  # 30 m DEM (matches Copernicus GLO-30)


def _build_dem(grid_w_parcels: int, grid_h_parcels: int, ox: float, oy: float,
               crs: str, seed: int) -> tuple[np.ndarray, "rasterio.Affine"]:
    """Tilted plane plus a few Gaussian hills, in UTM metres.

    Hills make slopes range ~0-25 degrees, which is the regime where the
    contrastive model can actually learn farm-vs-forest swap value.
    """
    rng = np.random.default_rng(seed)
    px_w = int(grid_w_parcels * CELL_M / DEM_RES)
    px_h = int(grid_h_parcels * CELL_M / DEM_RES)
    xs = np.arange(px_w) * DEM_RES
    ys = np.arange(px_h) * DEM_RES
    xx, yy = np.meshgrid(xs, ys)
    z = 100 + 0.012 * xx + 0.008 * yy

    n_hills = max(3, grid_w_parcels // 20)
    for _ in range(n_hills):
        cx = rng.uniform(0, px_w * DEM_RES)
        cy = rng.uniform(0, px_h * DEM_RES)
        amp = rng.uniform(60, 180)
        sigma = rng.uniform(40, 120) * DEM_RES
        z = z + amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

    transform = from_origin(ox, oy, DEM_RES, DEM_RES)
    return z.astype("float32"), transform


def _assign_dlbm(rng: np.random.Generator, slope_at_centroid: float,
                 col: int, row: int) -> tuple[str, str]:
    """Land-use code that's correlated with slope so MPC can find real swaps.

    Real DLTB has farms on flat land and forest on slopes; we add some noise
    (the "wrong" land-use placements MPC is supposed to fix) so the dataset
    isn't already optimal.
    """
    # Roads on every 20th column, water on every 31st row — sparse barriers.
    if col % 20 == 0:
        return "1011", "公路用地"
    if row % 31 == 0 and col % 7 == 0:
        return "1107", "水域"

    # Slope-driven prior with 25% noise to leave room for MPC to act.
    if rng.random() < 0.25:
        # noise → place inappropriately
        return ("011", "水田") if slope_at_centroid > 12 else ("031", "有林地")
    if slope_at_centroid < 5:
        return ("011", "水田") if rng.random() < 0.7 else ("012", "水浇地")
    if slope_at_centroid < 12:
        return ("013", "旱地") if rng.random() < 0.6 else ("011", "水田")
    return ("031", "有林地")


def _slope_from_z(z: np.ndarray, dem_res: float) -> np.ndarray:
    """Quick slope-degrees raster for placement priors. Not the same routine
    farmland_mpc.prepare uses; this is just to seed land-use plausibly.
    """
    dz_dy, dz_dx = np.gradient(z, dem_res)
    return np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="Output directory")
    ap.add_argument("--grid", type=int, default=70,
                    help="Parcels per side per township (default 70 → ~5k parcels for 1 township).")
    ap.add_argument("--townships", type=int, default=4,
                    help="Number of townships, laid out left-to-right (default 4).")
    ap.add_argument("--crs", default="EPSG:32648",
                    help="Output CRS for both shapefile and DEM (default EPSG:32648 UTM 48N).")
    ap.add_argument("--origin", default="500000,4400000",
                    help="UTM origin x,y for the raster top-left (default 500000,4400000).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qsdwdm-base", default="500227",
                    help="6-digit county code; townships get appended as XXX (default 500227 = Bishan).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("make_synth")
    args.out.mkdir(parents=True, exist_ok=True)

    ox_str, oy_str = args.origin.split(",")
    ox, oy = float(ox_str), float(oy_str)

    grid_w = args.grid * args.townships
    grid_h = args.grid
    n_parcels_total = grid_w * grid_h
    log.info("Generating %d parcels (%d townships × %dx%d) in %s",
             n_parcels_total, args.townships, args.grid, args.grid, args.crs)

    z, transform = _build_dem(grid_w, grid_h, ox, oy, args.crs, args.seed)
    dem_path = args.out / "dem.tif"
    with rasterio.open(
        dem_path, "w", driver="GTiff", dtype="float32", nodata=-9999.0,
        width=z.shape[1], height=z.shape[0], count=1, crs=args.crs,
        transform=transform,
    ) as dst:
        dst.write(z, 1)
    log.info("Wrote DEM %s  shape=%s  range=[%.1f, %.1f] m",
             dem_path, z.shape, float(z.min()), float(z.max()))

    slope_grid = _slope_from_z(z, DEM_RES)
    log.info("Synthetic slope range: %.2f – %.2f deg (median %.2f)",
             float(slope_grid.min()), float(slope_grid.max()), float(np.median(slope_grid)))

    rng = np.random.default_rng(args.seed)
    rows = []
    parcel_id = 0
    for col in range(grid_w):
        township_idx = min(col // args.grid, args.townships - 1)
        qsdwdm = f"{args.qsdwdm_base}{township_idx + 1:03d}"
        for row in range(grid_h):
            x0 = ox + col * CELL_M
            x1 = x0 + CELL_M
            y1 = oy - row * CELL_M
            y0 = y1 - CELL_M
            geom = box(x0, y0, x1, y1)

            # Slope at parcel centroid (CELL_M / DEM_RES px wide)
            cx_px = int((col + 0.5) * CELL_M / DEM_RES)
            cy_px = int((row + 0.5) * CELL_M / DEM_RES)
            cx_px = min(cx_px, slope_grid.shape[1] - 1)
            cy_px = min(cy_px, slope_grid.shape[0] - 1)
            slope_here = float(slope_grid[cy_px, cx_px])
            dlbm, dlmc = _assign_dlbm(rng, slope_here, col, row)

            parcel_id += 1
            rows.append({
                "BSM": f"S{parcel_id:07d}",
                "YSDM": "2002A0400",
                "DLBM": dlbm,
                "DLMC": dlmc,
                "QSDWDM": qsdwdm,
                "QSDWMC": f"合成乡镇{township_idx + 1}",
                "TBMJ": CELL_M * CELL_M / 10000.0,  # ha
                "SHAPE_Leng": 4 * CELL_M,
                "SHAPE_Area": CELL_M * CELL_M,
                "geometry": geom,
            })

    gdf = gpd.GeoDataFrame(rows, crs=args.crs)
    dltb_path = args.out / "dltb.shp"
    gdf.to_file(dltb_path, driver="ESRI Shapefile", encoding="utf-8")

    farm_codes = {"011", "012", "013"}
    forest = (gdf["DLBM"] == "031").sum()
    farm = gdf["DLBM"].isin(farm_codes).sum()
    other = len(gdf) - farm - forest
    log.info("Wrote DLTB %s  parcels=%d  farm=%d  forest=%d  barriers/other=%d",
             dltb_path, len(gdf), int(farm), int(forest), int(other))
    log.info("Townships: %s", sorted(gdf["QSDWDM"].unique()))
    log.info("Done. Use:")
    log.info("  farmland-mpc prepare --dltb %s --dem %s --out %s --crs %s",
             dltb_path, dem_path, args.out / "prepared", args.crs)


if __name__ == "__main__":
    main()
