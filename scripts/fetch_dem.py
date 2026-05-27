"""Fetch + mosaic + clip + reproject Copernicus GLO-30 DEM for a DLTB shapefile.

Lifted from notebooks/farmland_mpc_colab_full.ipynb section 4.
Output: <work_dir>/dem.tif in target UTM CRS at 30 m.
"""
from __future__ import annotations

import argparse
import math
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box


BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def tile_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def auto_utm(lon: float, lat: float) -> str:
    zone = int((lon + 180) // 6) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return f"EPSG:{epsg}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dltb", required=True, help="DLTB shapefile")
    ap.add_argument("--work-dir", required=True, help="Output directory")
    ap.add_argument("--proj-crs", default=None,
                    help="Target UTM CRS (e.g. EPSG:32648). Auto-picked if omitted.")
    ap.add_argument("--buffer-deg", type=float, default=0.01,
                    help="WGS84 buffer around bbox before clipping")
    args = ap.parse_args()

    dltb_path = Path(args.dltb)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[fetch_dem] reading {dltb_path}")
    gdf = gpd.read_file(dltb_path)
    gw = gdf.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = gw.total_bounds
    cen = gw.geometry.union_all().centroid

    if args.proj_crs:
        proj_crs = args.proj_crs
    else:
        proj_crs = auto_utm(cen.x, cen.y)
    print(f"[fetch_dem] WGS84 bbox: lon [{minx:.4f},{maxx:.4f}] lat [{miny:.4f},{maxy:.4f}]")
    print(f"[fetch_dem] target CRS: {proj_crs}")

    dem_dir = work_dir / "dem_tiles"
    dem_dir.mkdir(exist_ok=True)
    lon_tiles = range(math.floor(minx), math.floor(maxx) + 1)
    lat_tiles = range(math.floor(miny), math.floor(maxy) + 1)
    todo = []
    for lat in lat_tiles:
        for lon in lon_tiles:
            t = tile_name(lat, lon)
            todo.append((t, f"{BASE}/{t}/{t}.tif", dem_dir / f"{t}.tif"))
    print(f"[fetch_dem] {len(todo)} GLO-30 tile(s) touch the AOI")

    fetched = []
    for t, url, local in todo:
        if local.exists() and local.stat().st_size > 1_000_000:
            print(f"  [skip] {t}  ({local.stat().st_size/1e6:.1f} MB cached)")
            fetched.append(local)
            continue
        print(f"  [get]  {t}", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=300) as resp, open(local, "wb") as f:
                shutil.copyfileobj(resp, f)
            print(f"         {local.stat().st_size/1e6:.1f} MB ok")
            fetched.append(local)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print("         404 (no land here) — skipped")
            else:
                raise
    if not fetched:
        raise SystemExit("[fetch_dem] no tiles fetched; check bbox/network")

    srcs = [rasterio.open(p) for p in fetched]
    mosaic, m_transform = merge(srcs)
    m_meta = srcs[0].meta.copy()
    m_meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2],
                   "transform": m_transform})
    mosaic_wgs = work_dir / "dem_mosaic_wgs84.tif"
    with rasterio.open(mosaic_wgs, "w", **m_meta) as dst:
        dst.write(mosaic)
    for s in srcs:
        s.close()
    print(f"[fetch_dem] mosaic -> {mosaic_wgs}  shape={mosaic.shape}")

    aoi = box(minx - args.buffer_deg, miny - args.buffer_deg,
              maxx + args.buffer_deg, maxy + args.buffer_deg)
    with rasterio.open(mosaic_wgs) as src:
        clipped, clip_transform = rio_mask(src, [aoi], crop=True, nodata=src.nodata)
        clip_meta = src.meta.copy()
        clip_meta.update({"height": clipped.shape[1], "width": clipped.shape[2],
                          "transform": clip_transform})
    clipped_wgs = work_dir / "dem_clipped_wgs84.tif"
    with rasterio.open(clipped_wgs, "w", **clip_meta) as dst:
        dst.write(clipped)

    dem_final = work_dir / "dem.tif"
    with rasterio.open(clipped_wgs) as src:
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src.crs, proj_crs, src.width, src.height, *src.bounds, resolution=30.0)
        dst_meta = src.meta.copy()
        dst_meta.update({"crs": proj_crs, "transform": dst_transform,
                         "width": dst_w, "height": dst_h})
        with rasterio.open(dem_final, "w", **dst_meta) as dst:
            for b in range(1, src.count + 1):
                reproject(source=rasterio.band(src, b),
                          destination=rasterio.band(dst, b),
                          src_transform=src.transform, src_crs=src.crs,
                          dst_transform=dst_transform, dst_crs=proj_crs,
                          resampling=Resampling.bilinear)

    mosaic_wgs.unlink(missing_ok=True)
    clipped_wgs.unlink(missing_ok=True)
    print(f"[fetch_dem] DEM (UTM 30m): {dem_final}  ({dem_final.stat().st_size/1e6:.1f} MB)")
    print(f"[fetch_dem] proj_crs={proj_crs}")


if __name__ == "__main__":
    main()
