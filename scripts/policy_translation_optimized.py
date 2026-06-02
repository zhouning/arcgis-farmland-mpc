#!/usr/bin/env python3
"""Generic policy-translation audit for an optimized DLTB shapefile.

The script reads a Tool 4 optimized shapefile with ORIG_DLBM / OPT_DLBM /
CHG_FLAG / slope_mean and computes the policy-facing quantities requested by
the Codex CEE pre-submission review:

* slope-band shifts (<6, 6-15, 15-25, >25 deg)
* net cultivated-area change
* baimu-fang count and area deltas
* township-level slope-change distribution and fairness summary
* farm->forest and forest->farm parcel and area totals
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np

FARMLAND = 1
FOREST = 2
FARMLAND_PREFIXES = ("011", "012", "013")
FOREST_PREFIXES = ("031", "032", "033")
BAIMU_THRESHOLD_M2 = 66700.0
SLOPE_BANDS = [
    (0.0, 6.0, "lt6"),
    (6.0, 15.0, "6_15"),
    (15.0, 25.0, "15_25"),
    (25.0, 90.0, "gt25"),
]


def classify(dlbm) -> int:
    s = str(dlbm).strip()
    if s.startswith(FARMLAND_PREFIXES):
        return FARMLAND
    if s.startswith(FOREST_PREFIXES):
        return FOREST
    return 0


def slope_band(deg: float) -> str:
    for lo, hi, name in SLOPE_BANDS:
        if lo <= deg < hi:
            return name
    return SLOPE_BANDS[-1][2]


def gini(values) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    x = np.sort(np.maximum(x, 0.0))
    if x.sum() == 0:
        return 0.0
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def build_queen_adjacency(gdf: gpd.GeoDataFrame) -> list[np.ndarray]:
    try:
        from libpysal.weights import Queen

        w = Queen.from_dataframe(gdf, use_index=False)
        return [np.array(w.neighbors[i], dtype=np.intp) for i in range(len(gdf))]
    except Exception:
        from shapely.strtree import STRtree

        geoms = gdf.geometry.values
        tree = STRtree(geoms)
        out = []
        for i, geom in enumerate(geoms):
            cand = tree.query(geom, predicate="intersects")
            out.append(np.array([int(j) for j in cand if int(j) != i], dtype=np.intp))
        return out


def count_baimu(types: np.ndarray, areas: np.ndarray, adj: list[np.ndarray]) -> tuple[int, float]:
    n = len(types)
    is_farm = types == FARMLAND
    parent = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int32)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for i in range(n):
        if not is_farm[i]:
            continue
        for j in adj[i]:
            if j > i and is_farm[j]:
                ri, rj = find(i), find(int(j))
                if ri != rj:
                    if rank[ri] < rank[rj]:
                        ri, rj = rj, ri
                    parent[rj] = ri
                    if rank[ri] == rank[rj]:
                        rank[ri] += 1

    farm_idx = np.where(is_farm)[0]
    if len(farm_idx) == 0:
        return 0, 0.0
    roots = np.array([find(int(i)) for i in farm_idx], dtype=np.int32)
    uniq, inv = np.unique(roots, return_inverse=True)
    comp_area = np.zeros(len(uniq), dtype=np.float64)
    np.add.at(comp_area, inv, areas[farm_idx])
    keep = comp_area >= BAIMU_THRESHOLD_M2
    return int(keep.sum()), float(comp_area[keep].sum())


def weighted_mean_slope(gdf: gpd.GeoDataFrame, mask: np.ndarray) -> float:
    area = gdf.loc[mask, "area_m2"].to_numpy(dtype=float)
    if area.sum() <= 0:
        return float("nan")
    slope = gdf.loc[mask, "slope_mean"].to_numpy(dtype=float)
    return float((slope * area).sum() / area.sum())


def band_report(gdf: gpd.GeoDataFrame, mask_col: str) -> dict:
    farm = gdf[gdf[mask_col]].copy()
    farm["band"] = farm["slope_mean"].apply(slope_band)
    agg = farm.groupby("band")["area_m2"].sum() / 10000.0
    total = float(farm["area_m2"].sum() / 10000.0)
    by_band = {name: float(agg.get(name, 0.0)) for _, _, name in SLOPE_BANDS}
    pct = {k: 100.0 * v / total if total > 0 else 0.0 for k, v in by_band.items()}
    return {"total_farmland_ha": total, "by_band_ha": by_band, "by_band_pct": pct}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimized", required=True, type=Path)
    ap.add_argument("--townships", type=Path)
    ap.add_argument("--region", required=True)
    ap.add_argument("--proj-crs", default="EPSG:32648")
    ap.add_argument("--town-code-len", type=int, default=9)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    g = gpd.read_file(args.optimized)
    required = {"ORIG_DLBM", "OPT_DLBM", "CHG_FLAG", "QSDWDM", "slope_mean"}
    missing = sorted(required - set(g.columns))
    if missing:
        raise ValueError(f"{args.optimized} missing required columns: {missing}")
    if g.crs is None or g.crs.to_string() != args.proj_crs:
        g = g.to_crs(args.proj_crs)

    g["area_m2"] = g.geometry.area
    g["orig_type"] = g["ORIG_DLBM"].apply(classify)
    g["opt_type"] = g["OPT_DLBM"].apply(classify)
    swappable = g[g["orig_type"].isin([FARMLAND, FOREST])].copy().reset_index(drop=True)
    swappable["orig_is_farm"] = swappable["orig_type"] == FARMLAND
    swappable["opt_is_farm"] = swappable["opt_type"] == FARMLAND
    swappable["orig_is_forest"] = swappable["orig_type"] == FOREST
    swappable["opt_is_forest"] = swappable["opt_type"] == FOREST

    orig_bands = band_report(swappable, "orig_is_farm")
    opt_bands = band_report(swappable, "opt_is_farm")
    band_delta = {
        name: opt_bands["by_band_ha"][name] - orig_bands["by_band_ha"][name]
        for _, _, name in SLOPE_BANDS
    }

    chg = swappable["CHG_FLAG"].astype(int)
    farm_to_forest = swappable[(chg == 1) & swappable["orig_is_farm"]]
    forest_to_farm = swappable[(chg == 2) & swappable["orig_is_forest"]]
    swap_areas = {
        "farm_to_forest_count": int(len(farm_to_forest)),
        "forest_to_farm_count": int(len(forest_to_farm)),
        "farm_to_forest_area_ha": float(farm_to_forest["area_m2"].sum() / 10000.0),
        "forest_to_farm_area_ha": float(forest_to_farm["area_m2"].sum() / 10000.0),
        "net_cultivated_area_change_ha_from_swaps": float(
            (forest_to_farm["area_m2"].sum() - farm_to_forest["area_m2"].sum()) / 10000.0
        ),
    }

    orig_types = swappable["orig_type"].to_numpy(dtype=np.int8)
    opt_types = swappable["opt_type"].to_numpy(dtype=np.int8)
    adj = build_queen_adjacency(swappable)
    orig_baimu_count, orig_baimu_area = count_baimu(orig_types, swappable["area_m2"].to_numpy(), adj)
    opt_baimu_count, opt_baimu_area = count_baimu(opt_types, swappable["area_m2"].to_numpy(), adj)

    if args.townships and args.townships.exists():
        townships = json.loads(args.townships.read_text())
        town_codes = sorted(townships.keys())
    else:
        town_codes = sorted(
            swappable["QSDWDM"].astype(str).str[: args.town_code_len].unique()
        )

    swappable["town_code"] = swappable["QSDWDM"].astype(str).str[: args.town_code_len]
    town_rows = []
    for code in town_codes:
        sub = swappable[swappable["town_code"] == code]
        if sub.empty:
            continue
        orig_mask = sub["orig_is_farm"].to_numpy()
        opt_mask = sub["opt_is_farm"].to_numpy()
        orig_area_ha = float(sub.loc[orig_mask, "area_m2"].sum() / 10000.0)
        opt_area_ha = float(sub.loc[opt_mask, "area_m2"].sum() / 10000.0)
        orig_slope = weighted_mean_slope(sub, orig_mask)
        opt_slope = weighted_mean_slope(sub, opt_mask)
        row = {
            "town_code": code,
            "n_parcels_swappable": int(len(sub)),
            "farmland_area_orig_ha": orig_area_ha,
            "farmland_area_opt_ha": opt_area_ha,
            "farmland_area_change_ha": opt_area_ha - orig_area_ha,
            "slope_mean_orig_deg": orig_slope,
            "slope_mean_opt_deg": opt_slope,
            "slope_change_pct": (
                100.0 * (opt_slope - orig_slope) / orig_slope
                if orig_slope > 0 and not math.isnan(opt_slope)
                else float("nan")
            ),
            "farm_to_forest_count": int(((sub["CHG_FLAG"] == 1) & sub["orig_is_farm"]).sum()),
            "forest_to_farm_count": int(((sub["CHG_FLAG"] == 2) & sub["orig_is_forest"]).sum()),
            "farm_to_forest_area_ha": float(
                sub.loc[(sub["CHG_FLAG"] == 1) & sub["orig_is_farm"], "area_m2"].sum()
                / 10000.0
            ),
            "forest_to_farm_area_ha": float(
                sub.loc[(sub["CHG_FLAG"] == 2) & sub["orig_is_forest"], "area_m2"].sum()
                / 10000.0
            ),
        }
        town_rows.append(row)

    slope_abs = np.array(
        [abs(r["slope_change_pct"]) for r in town_rows if np.isfinite(r["slope_change_pct"])],
        dtype=float,
    )
    area_change = np.array([r["farmland_area_change_ha"] for r in town_rows], dtype=float)
    fairness = {
        "n_townships": len(town_rows),
        "slope_pct_min_abs": float(np.min(slope_abs)) if slope_abs.size else None,
        "slope_pct_mean_abs": float(np.mean(slope_abs)) if slope_abs.size else None,
        "slope_pct_max_abs": float(np.max(slope_abs)) if slope_abs.size else None,
        "slope_pct_std_abs": float(np.std(slope_abs, ddof=1)) if slope_abs.size > 1 else None,
        "slope_pct_gini_abs": gini(slope_abs),
        "farmland_area_change_ha_min": float(np.min(area_change)) if area_change.size else None,
        "farmland_area_change_ha_mean": float(np.mean(area_change)) if area_change.size else None,
        "farmland_area_change_ha_max": float(np.max(area_change)) if area_change.size else None,
    }

    summary = None
    if args.summary and args.summary.exists():
        summary = json.loads(args.summary.read_text())

    out = {
        "region": args.region,
        "source_optimized_shp": str(args.optimized),
        "n_swappable_parcels": int(len(swappable)),
        "slope_bands": {
            "definition_deg": [
                {"name": name, "lo": lo, "hi": hi} for lo, hi, name in SLOPE_BANDS
            ],
            "orig": orig_bands,
            "opt": opt_bands,
            "delta_ha": band_delta,
        },
        "cultivated_area": {
            "orig_ha": orig_bands["total_farmland_ha"],
            "opt_ha": opt_bands["total_farmland_ha"],
            "delta_ha": opt_bands["total_farmland_ha"] - orig_bands["total_farmland_ha"],
            "delta_pct": 100.0
            * (opt_bands["total_farmland_ha"] - orig_bands["total_farmland_ha"])
            / max(abs(orig_bands["total_farmland_ha"]), 1e-8),
        },
        "baimu_fang": {
            "orig_count": int(orig_baimu_count),
            "opt_count": int(opt_baimu_count),
            "delta_count": int(opt_baimu_count - orig_baimu_count),
            "orig_area_ha": orig_baimu_area / 10000.0,
            "opt_area_ha": opt_baimu_area / 10000.0,
            "delta_area_ha": (opt_baimu_area - orig_baimu_area) / 10000.0,
        },
        "swap_area_totals": swap_areas,
        "township_fairness": {
            "townships": town_rows,
            "summary": fairness,
        },
        "mpc_summary": summary,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {args.out}")
    print(
        "cultivated delta "
        f"{out['cultivated_area']['delta_ha']:+.2f} ha; "
        f"slope steep-tail delta "
        f"{band_delta['15_25'] + band_delta['gt25']:+.2f} ha; "
        f"baimu delta {out['baimu_fang']['delta_area_ha']:+.2f} ha"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
