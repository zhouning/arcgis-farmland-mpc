# Prepare — DEM-derived slope per parcel

Stage 0 of the data pipeline: take a Third National Land Survey (DLTB)
cadastral layer plus Copernicus DEM GLO-30 elevation tiles, compute
per-pixel slope, run zonal statistics per parcel, and write a unified
`DLTB_with_slope.gpkg` that the rest of the pipeline (`prepare/blocks/` →
`farmland_mpc/sample` → `farmland_mpc/train` → `farmland_mpc/plan`)
consumes verbatim.

## Two variants — same algorithm, different I/O

| Script | Input | CRS handling | Encoding | DEM tile coverage |
|---|---|---|---|---|
| `bishan_dem_slope_zonal.py` | file GDB (`GDB.gdb`, layer `DLTB`) | EPSG:4610 → 4326 override (safe; the GDB metadata is mislabelled, coords are CGCS2000 lat/lon) | UTF-8 | 1 tile: `N29-E106` |
| `neijiang_dem_slope_zonal.py` | shapefile (`DLTB.shp`) | EPSG:2359 → 4326 reproject (Neijiang stores Gauss-Kruger meters, must transform) | GBK (DBF) | 2 tiles: `N29-E104` + `N29-E105`, mosaic'd |

Same Copernicus DEM GLO-30 source, same slope/zonal-stats logic, same
output schema. The two variants exist because the two counties' raw
cadastral exports differ in vector format, projection, and DBF encoding.

## Inputs you need to supply

The DLTB cadastral data are **restricted Third National Land Survey records**
and are NOT included in this repo (see paper Data Availability).
Get a copy from your data provider, then point the script at it via env vars:

```bash
# Bishan
export BISHAN_GDB_PATH=/path/to/bishan/GDB.gdb
export BISHAN_XIANGZHEN_PATH=/path/to/bishan/xiangzhen.shp
export BISHAN_OUTPUT_DIR=/path/to/bishan_run
python bishan_dem_slope_zonal.py

# Neijiang
export NEIJIANG_DLTB_SHP=/path/to/neijiang/DLTB.shp
export NEIJIANG_XZQ_SHP=/path/to/neijiang/XZQ.shp
export NEIJIANG_OUTPUT_DIR=/path/to/neijiang_run
python neijiang_dem_slope_zonal.py
```

The Copernicus DEM tiles are **public** — both scripts download them on
demand from `https://copernicus-dem-30m.s3.amazonaws.com/` into
`<OUTPUT_DIR>/intermediate/` and re-use the cache on subsequent runs. No
manual DEM provisioning needed.

## Outputs

For each county:

```
<OUTPUT_DIR>/
├── intermediate/
│   ├── Copernicus_DSM_COG_10_*_DEM.tif    # cached, ~45 MB per 1°×1° tile
│   ├── dem_full_tile.npy                   # mosaic'd numpy array
│   ├── slope_degrees.npy                   # per-pixel slope (numpy gradient)
│   ├── parcels_attributes.csv              # parcel metadata
│   └── pixel_parcel_matches.csv            # which DEM pixel falls in which parcel
└── output/
    ├── DLTB_with_slope.gpkg                # ← the deliverable consumed downstream
    ├── DLTB_with_slope.gdb                 # OpenFileGDB mirror for ArcGIS users
    └── slope_statistics_summary.csv
```

The `DLTB_with_slope.gpkg` adds three columns to the original DLTB schema:
`slope_mean`, `slope_max`, `slope_pixel_count` (plus a `category` column
remapping `DLBM` to farmland / forest / orchard / other via the standard
三调 codes).

Order of operations (each variant runs all steps end-to-end):

1. Load DLTB → keep relevant fields → classify land use
2. Determine DEM tile coverage from parcel bounds → download if missing
3. Read DEM tiles → mosaic to a single float32 array
4. Compute slope raster (numpy gradient → degrees)
5. Match each parcel's pixels via STRtree spatial index
6. Aggregate slope_mean / slope_max / slope_pixel_count per parcel
7. Write GPKG (and optional file GDB)

## Wall-clock budget (commodity laptop)

| Stage | Bishan (~52k parcels, 1 tile) | Neijiang (~76k parcels, 2 tiles) |
|---|---|---|
| DEM download (first run only) | ~30 s | ~60 s |
| Slope + zonal stats | ~5 min | ~8 min |
| GPKG write | ~30 s | ~45 s |

Fully cached re-run: ~5 min Bishan / ~8 min Neijiang.

## After this stage

Feed `output/DLTB_with_slope.gpkg` into block construction
(`neijiang_cross_region/build_blocks.py` for Neijiang, the equivalent
`block_definition_all.py` in the research checkout for Bishan), which
emits `block_features.json` + `block_compositions.json` per township.
From there the published `farmland_mpc` package handles
sample → train → plan.
