"""End-to-end smoke test: prepare (A+B+C) -> sample -> train -> plan.

Builds a 6x6 = 36 parcel synthetic fixture in 2 townships with mixed
farmland (DLBM 011/012) and forest (031) parcels, then runs the full
pure-Python pipeline. Verifies:
  - prepare() writes townships.json + results_real/blocks/township_*/
  - sample() writes tool2/transitions.npz + pairwise.npz
  - train() writes tool3/ensemble_member*.onnx
  - plan() writes mpc_summary.json + optimized.shp

Runtime: ~30-60 seconds depending on CPU. Train epochs reduced to 3.

Run from the repo root:

    python -m farmland_mpc.tests.smoke_end_to_end
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

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("smoke_e2e")

# UTM Zone 48N WGS84 PROJ-string-equivalent WKT, bypassing the local proj.db
CRS_WKT = (
    'PROJCRS["WGS 84 / UTM zone 48N",'
    'BASEGEOGCRS["WGS 84",DATUM["World Geodetic System 1984",'
    'ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],'
    'PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]]],'
    'CONVERSION["UTM zone 48N",METHOD["Transverse Mercator",ID["EPSG",9807]],'
    'PARAMETER["Latitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433]],'
    'PARAMETER["Longitude of natural origin",105,ANGLEUNIT["degree",0.0174532925199433]],'
    'PARAMETER["Scale factor at natural origin",0.9996,SCALEUNIT["unity",1]],'
    'PARAMETER["False easting",500000,LENGTHUNIT["metre",1]],'
    'PARAMETER["False northing",0,LENGTHUNIT["metre",1]]],'
    'CS[Cartesian,2],'
    'AXIS["easting",east,ORDER[1],LENGTHUNIT["metre",1]],'
    'AXIS["northing",north,ORDER[2],LENGTHUNIT["metre",1]]]'
)

GRID = 6  # 6x6 parcels per township
CELL_M = 100.0  # 100 m parcels = 1 ha
DEM_RES = 30.0  # 30 m DEM resolution


def _make_fixture(tmp: Path) -> tuple[Path, Path]:
    """Build a 36-parcel DLTB (mixed farm/forest, 2 townships) + a DEM."""
    dltb_path = tmp / "dltb.shp"
    dem_path = tmp / "dem.tif"

    # DEM: tilted plane with a bump in the top-right. Covers both townships.
    ox, oy = 500000.0, 4400000.0  # raster top-left
    grid_w = 2 * GRID  # 12 parcels wide (T01 left, T02 right)
    grid_h = GRID
    px_w = int(grid_w * CELL_M / DEM_RES)
    px_h = int(grid_h * CELL_M / DEM_RES)
    xs, ys = np.arange(px_w) * DEM_RES, np.arange(px_h) * DEM_RES
    xx, yy = np.meshgrid(xs, ys)
    z = 100 + 0.04 * xx + 0.015 * yy
    bump = 80.0 * np.exp(
        -((xx - 0.75 * px_w * DEM_RES) ** 2 + (yy - 0.25 * px_h * DEM_RES) ** 2)
        / (2 * (50 * DEM_RES) ** 2)
    )
    z = z + bump
    tf = from_origin(ox, oy, DEM_RES, DEM_RES)
    with rasterio.open(
        dem_path, "w", driver="GTiff", dtype="float32", nodata=-9999.0,
        width=px_w, height=px_h, count=1, crs=CRS_WKT, transform=tf,
    ) as dst:
        dst.write(z.astype("float32"), 1)
    log.info("wrote DEM %s (%dx%d at %s m)", dem_path.name, px_w, px_h, DEM_RES)

    # DLTB: 12x6 grid. Left half (cols 0-5) -> township 500227001; right -> 500227002.
    rows = []
    rng = np.random.default_rng(42)
    parcel_id = 0
    for col in range(grid_w):
        for row in range(grid_h):
            x0 = ox + col * CELL_M
            x1 = x0 + CELL_M
            y1 = oy - row * CELL_M
            y0 = y1 - CELL_M
            geom = box(x0, y0, x1, y1)
            twn = "500227001" if col < GRID else "500227002"
            # Mix farm (60%) and forest (40%) with a deterministic-ish layout.
            r = rng.random()
            if r < 0.5:
                dlbm = "011"
                dlmc = "水田"
            elif r < 0.6:
                dlbm = "012"
                dlmc = "水浇地"
            else:
                dlbm = "031"
                dlmc = "有林地"
            parcel_id += 1
            rows.append({
                "BSM": f"P{parcel_id:04d}",
                "DLBM": dlbm,
                "DLMC": dlmc,
                "QSDWDM": twn,
                "geometry": geom,
            })
    gdf = gpd.GeoDataFrame(rows, crs=CRS_WKT)
    gdf.to_file(dltb_path, driver="ESRI Shapefile", encoding="utf-8")
    log.info("wrote DLTB %s (%d parcels, %d townships)",
             dltb_path.name, len(gdf), gdf["QSDWDM"].nunique())
    return dltb_path, dem_path


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="farmland_mpc_e2e_",
                                     ignore_cleanup_errors=True) as td:
        tmp = Path(td)
        dltb_path, dem_path = _make_fixture(tmp)
        prepared = tmp / "prepared"

        # ---- Phase A + B + C ----
        from farmland_mpc.prepare import run as prepare_run
        prepare_run(
            dltb_path=dltb_path,
            dem_path=dem_path,
            prepared_dir=prepared,
            proj_crs=CRS_WKT,
            run_phase_bc=True,
            min_parcels=2,
            min_area_ha=0.0,
            max_parcels=20,
            min_parcels_per_township=10,  # well below toolbox default 50
        )
        assert (prepared / "townships.json").exists()
        twn = json.loads((prepared / "townships.json").read_text(encoding="utf-8"))
        assert len(twn) >= 1, f"expected >=1 township, got {twn}"
        log.info("Phase A+B+C OK: %d townships, sample: %s", len(twn), list(twn.items())[:2])

        # ---- Phase B (sampling) ----
        from farmland_mpc.sample import run as sample_run
        sample_run(
            prepared_dir=prepared,
            n_transition_episodes=5,
            n_pairwise_states=20,
            n_pairwise_actions=4,
            seed=0,
            proj_crs=CRS_WKT,
        )
        assert (prepared / "tool2" / "transitions.npz").exists()
        assert (prepared / "tool2" / "pairwise.npz").exists()
        log.info("Phase B (sample) OK")

        # ---- Phase C (training) ----
        from farmland_mpc.train_ensemble import run as train_run
        train_run(
            prepared_dir=str(prepared),
            n_members=2,
            epochs=2,
            patience=0,
            lambda_rank=5.0,
            margin=0.1,
            batch_size=32,
            seed_base=0,
            torch_threads=0,
        )
        onnx_members = list((prepared / "tool3").glob("ensemble_member*.onnx"))
        assert len(onnx_members) == 2, f"expected 2 onnx members, got {onnx_members}"
        log.info("Phase C (train) OK: %d members", len(onnx_members))

        # ---- Phase D (MPC plan) ----
        from farmland_mpc.mpc_plan import run as plan_run
        out_dir = tmp / "mpc_out"
        out_shp = out_dir / "optimized.shp"
        plan_run(
            ensemble_dir=str(prepared / "tool3"),
            out_dir=str(out_dir),
            horizon=2,
            top_k=3,
            n_episodes=1,
            continuation="random",
            scoring="reward",
            threads=0,
            seed_offset=0,
            prepared_dir=str(prepared),
            proj_crs=CRS_WKT,
            output_fc=str(out_shp),
            input_dltb_fc=str(prepared / "dem_slope_analysis" / "output" / "DLTB_with_slope.shp"),
            farm_dlbm="011",
            forest_dlbm="031",
        )
        assert (out_dir / "mpc_summary.json").exists()
        assert out_shp.exists()
        summary = json.loads((out_dir / "mpc_summary.json").read_text(encoding="utf-8"))
        log.info("Phase D (plan) OK: summary keys = %s", list(summary.keys())[:8])

        log.info("END-TO-END SMOKE PASSED ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
