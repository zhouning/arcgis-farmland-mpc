"""Write synthetic dataset in the schema county_env.py expects."""
from __future__ import annotations
import json
from pathlib import Path
from typing import TypedDict, Sequence
import geopandas as gpd
from shapely.geometry import Polygon


SWAPPABLE_PREFIXES = ("011", "012", "013", "031", "032", "033")


class SyntheticDataset(TypedDict):
    blocks: Sequence[Polygon]
    parcels_per_block: Sequence[Sequence[Polygon]]
    parcel_dlbm: Sequence[Sequence[str]]
    parcel_slope: Sequence[Sequence[float]]
    block_township_id: Sequence[int]
    block_compactness: Sequence[float]
    township_codes: Sequence[str]


def write_synthetic_dataset(
    data: SyntheticDataset,
    out_dir: str | Path,
    crs: str = "EPSG:4523",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks = data["blocks"]
    parcels_per_block = data["parcels_per_block"]
    parcel_dlbm = data["parcel_dlbm"]
    parcel_slope = data["parcel_slope"]
    block_township_id = data["block_township_id"]
    block_compactness = data["block_compactness"]
    township_codes = data["township_codes"]
    n_townships = len(township_codes)

    geoms: list[Polygon] = []
    dlbm_col: list[str] = []
    qsdwdm_col: list[str] = []
    slope_col: list[float] = []
    block_to_swap_local: dict[int, list[int]] = {bi: [] for bi in range(len(blocks))}
    parcel_counter_per_township: list[int] = [0] * n_townships
    swap_counter_per_township: list[int] = [0] * n_townships

    for bi, parcels in enumerate(parcels_per_block):
        ti = block_township_id[bi]
        tcode = township_codes[ti]
        for k, parcel in enumerate(parcels):
            dlbm = parcel_dlbm[bi][k]
            slope = parcel_slope[bi][k]
            parcel_idx_in_township = parcel_counter_per_township[ti]
            qsdwdm = f"{tcode}{parcel_idx_in_township:03d}"
            parcel_counter_per_township[ti] += 1
            geoms.append(parcel)
            dlbm_col.append(dlbm)
            qsdwdm_col.append(qsdwdm)
            slope_col.append(float(slope))
            if dlbm.startswith(SWAPPABLE_PREFIXES):
                local_swap = swap_counter_per_township[ti]
                swap_counter_per_township[ti] += 1
                block_to_swap_local[bi].append(local_swap)

    gdf = gpd.GeoDataFrame(
        {"DLBM": dlbm_col, "QSDWDM": qsdwdm_col, "slope_mean": slope_col},
        geometry=geoms,
        crs=crs,
    )
    gdf.to_file(out_dir / "DLTB_with_slope.gpkg", driver="GPKG")

    for ti, tcode in enumerate(township_codes):
        tdir = out_dir / f"township_{tcode}"
        tdir.mkdir(parents=True, exist_ok=True)
        local_blocks = [bi for bi in range(len(blocks)) if block_township_id[bi] == ti]
        compositions = {
            str(local_idx): block_to_swap_local[bi]
            for local_idx, bi in enumerate(local_blocks)
        }
        with open(tdir / "block_compositions.json", "w") as f:
            json.dump(compositions, f)
        features = [
            {"compactness": float(block_compactness[bi])} for bi in local_blocks
        ]
        with open(tdir / "block_features.json", "w") as f:
            json.dump(features, f)
