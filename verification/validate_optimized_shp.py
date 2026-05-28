# -*- coding: utf-8 -*-
"""
validate_optimized_shp.py — independent GIS-grounded recomputation of the
three reported MPC metrics (slope / contiguity / baimu-fang area) directly
from the optimized.shp written by Tool 4.

Why this script exists
----------------------
Reviewers cannot tell whether the slope reduction reported in
mpc_summary.json is a real physical change in the optimized DLTB or an
artefact of the learned ensemble dynamics used inside MPC. This script
bypasses the env + ensemble entirely:

    1. Read optimized.shp (Tool 4 output) — has ORIG_DLBM, OPT_DLBM, CHG_FLAG
    2. Join slope_mean from prepared_dir/dem_slope_analysis/output/DLTB_with_slope.shp
    3. Recompute, on the SAME parcels env saw, twice:
         baseline   = use ORIG_DLBM
         optimized  = use OPT_DLBM
    4. Print Δslope, Δcont, Δbaimu_count, Δbaimu_area_ha
    5. Compare against mpc_summary.json["aggregate"]

Algorithm matches farmland_mpc/county_env.py exactly:
    slope     : area-weighted mean of farmland (DLBM ∈ {011,012,013}) parcels
    contiguity: total_farmland_adj / n_farmland (Queen contiguity)
    baimu     : connected-component union-find on farmland subgraph,
                count / total area where component_area >= 66700 m² (100 mu)

Usage
-----
    python validate_optimized_shp.py \
        --optimized <out_dir>/optimized.shp \
        --slope-shp <prepared_dir>/dem_slope_analysis/output/DLTB_with_slope.shp \
        --summary   <out_dir>/mpc_summary.json \
        --proj-crs  EPSG:32648 \
        --out       validate_report.json

Exit code is 0 when all three deltas match within tolerance, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import geopandas as gpd

FARMLAND = 1
FOREST = 2
FARMLAND_PREFIXES = ("011", "012", "013")
FOREST_PREFIXES = ("031", "032", "033")
BAIMU_THRESHOLD_M2 = 66700.0  # 100 mu


def classify(dlbm: str) -> int:
    s = str(dlbm).strip()
    if s.startswith(FARMLAND_PREFIXES):
        return FARMLAND
    if s.startswith(FOREST_PREFIXES):
        return FOREST
    return 0


def build_queen_adjacency(gdf: gpd.GeoDataFrame) -> list[np.ndarray]:
    """Queen contiguity. libpysal first, STRtree intersects fallback."""
    n = len(gdf)
    try:
        from libpysal.weights import Queen
        w = Queen.from_dataframe(gdf, use_index=False)
        return [np.array(w.neighbors[i], dtype=np.intp) for i in range(n)]
    except Exception as e:
        print(f"  libpysal failed ({e}); falling back to STRtree intersects", flush=True)
        from shapely.strtree import STRtree
        geoms = gdf.geometry.values
        tree = STRtree(geoms)
        adj = []
        for i in range(n):
            cand = tree.query(geoms[i], predicate="intersects")
            adj.append(np.array([int(j) for j in cand if int(j) != i], dtype=np.intp))
        return adj


def compute_slope_cont(types: np.ndarray, slopes: np.ndarray, areas: np.ndarray,
                      adjacency: list[np.ndarray]):
    fm = types == FARMLAND
    n_farmland = int(fm.sum())
    total_farm_area = float(areas[fm].sum())
    total_weighted_slope = float((slopes[fm] * areas[fm]).sum())
    avg_slope = total_weighted_slope / max(total_farm_area, 1e-8)

    # contiguity: mean farmland-neighbor count per farmland parcel
    n = len(types)
    farmland_nbr = np.zeros(n, dtype=np.int32)
    for i in range(n):
        nbrs = adjacency[i]
        if len(nbrs):
            farmland_nbr[i] = int((types[nbrs] == FARMLAND).sum())
    total_adj = int(farmland_nbr[fm].sum())
    contiguity = total_adj / max(n_farmland, 1)

    return avg_slope, contiguity, n_farmland, total_farm_area


def count_baimu(types: np.ndarray, areas: np.ndarray,
                adjacency: list[np.ndarray]) -> tuple[int, float]:
    """Union-find on farmland subgraph; return (count, total_area_m2) of
    components whose area >= BAIMU_THRESHOLD_M2."""
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
        for j in adjacency[i]:
            if j > i and is_farm[j]:
                ri, rj = find(i), find(j)
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
    mask = comp_area >= BAIMU_THRESHOLD_M2
    return int(mask.sum()), float(comp_area[mask].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimized", required=True,
                    help="Path to Tool 4 optimized.shp (must have ORIG_DLBM, OPT_DLBM, CHG_FLAG)")
    ap.add_argument("--slope-shp", required=True,
                    help="Path to DLTB_with_slope.shp (must have BSM, slope_mean)")
    ap.add_argument("--summary", required=True,
                    help="Path to mpc_summary.json (for cross-check)")
    ap.add_argument("--proj-crs", default="EPSG:32648",
                    help="Projected CRS for area computation; must match the env (default EPSG:32648)")
    ap.add_argument("--out", default="validate_report.json")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[1/6] Reading optimized.shp ...", flush=True)
    opt = gpd.read_file(args.optimized)
    required_opt = ["BSM", "ORIG_DLBM", "OPT_DLBM", "CHG_FLAG"]
    miss = [c for c in required_opt if c not in opt.columns]
    if miss:
        print(f"ERROR: optimized.shp missing columns: {miss}", file=sys.stderr)
        sys.exit(2)
    print(f"      {len(opt)} rows", flush=True)

    print(f"[2/6] Reading slope shp ...", flush=True)
    slope_gdf = gpd.read_file(args.slope_shp, columns=["BSM", "slope_mean"])
    if "slope_mean" not in slope_gdf.columns or "BSM" not in slope_gdf.columns:
        print(f"ERROR: slope shp missing BSM or slope_mean", file=sys.stderr)
        sys.exit(2)
    print(f"      {len(slope_gdf)} rows", flush=True)

    # Normalize BSM to string for join (mirrors shapefile_io._norm_bsm)
    def _norm(v):
        try:
            f = float(v)
            if f.is_integer():
                return str(int(f))
        except (TypeError, ValueError):
            pass
        return str(v).strip()

    opt["_BSM_KEY"] = opt["BSM"].apply(_norm)
    slope_df = slope_gdf[["BSM", "slope_mean"]].copy()
    slope_df["_BSM_KEY"] = slope_df["BSM"].apply(_norm)
    bsm_to_slope = dict(zip(slope_df["_BSM_KEY"], slope_df["slope_mean"]))

    print(f"[3/6] Filtering to env-resident parcels (CHG_FLAG-bearing rows that match slope shp BSM) ...",
          flush=True)
    # env resident set = parcels classified as farmland or forest by ORIG_DLBM AND that have a slope value
    orig_type = opt["ORIG_DLBM"].apply(classify).values
    opt_type = opt["OPT_DLBM"].apply(classify).values
    has_slope = opt["_BSM_KEY"].isin(bsm_to_slope).values
    swap_eligible = np.isin(orig_type, [FARMLAND, FOREST])
    keep = swap_eligible & has_slope

    sub = opt.loc[keep].reset_index(drop=True).copy()
    sub_orig_type = orig_type[keep]
    sub_opt_type = opt_type[keep]
    print(f"      kept {len(sub)} rows (env-resident swappable parcels with slope)", flush=True)

    # Fall back: if OPT_DLBM is non-farm/forest (shouldn't happen for swapped rows;
    # CHG_FLAG=1 -> forest, 2 -> farm), trust CHG_FLAG.
    chg = sub["CHG_FLAG"].astype(int).values
    final_type = sub_orig_type.copy()
    final_type[chg == 1] = FOREST    # farm -> forest
    final_type[chg == 2] = FARMLAND  # forest -> farm
    # When CHG_FLAG = 0 the OPT_DLBM equals ORIG_DLBM by construction, so
    # final_type == sub_orig_type already holds.

    print(f"      swap counts from CHG_FLAG: farm->forest={(chg==1).sum()}, "
          f"forest->farm={(chg==2).sum()}, unchanged={(chg==0).sum()}", flush=True)

    # slopes (parcel value, fill any NaN with median to mirror env behaviour)
    sub_slope = np.array([bsm_to_slope[k] for k in sub["_BSM_KEY"].values], dtype=np.float64)
    nan_mask = np.isnan(sub_slope)
    n_nan = int(nan_mask.sum())
    if n_nan:
        finite = sub_slope[~nan_mask]
        fill = float(np.median(finite)) if finite.size else 0.0
        sub_slope = np.where(nan_mask, fill, sub_slope)
        print(f"      WARN: {n_nan} parcels had NaN slope_mean; filled with median={fill:.3f}",
              flush=True)

    # areas in the projected CRS (mirrors env)
    print(f"[4/6] Reprojecting to {args.proj_crs} for area computation ...", flush=True)
    sub_proj = sub.to_crs(args.proj_crs)
    sub_area = sub_proj.geometry.area.values.astype(np.float64)

    print(f"[5/6] Building Queen adjacency for {len(sub)} parcels ...", flush=True)
    t_adj = time.time()
    adj = build_queen_adjacency(sub)
    print(f"      adjacency built in {time.time()-t_adj:.1f}s "
          f"(median deg={np.median([len(a) for a in adj]):.0f})", flush=True)

    print(f"[6/6] Computing baseline + optimized metrics ...", flush=True)
    base_slope, base_cont, base_n_farm, base_farm_area = compute_slope_cont(
        sub_orig_type, sub_slope, sub_area, adj
    )
    base_baimu_n, base_baimu_area = count_baimu(sub_orig_type, sub_area, adj)

    opt_slope, opt_cont, opt_n_farm, opt_farm_area = compute_slope_cont(
        final_type, sub_slope, sub_area, adj
    )
    opt_baimu_n, opt_baimu_area = count_baimu(final_type, sub_area, adj)

    slope_change_pct = 100.0 * (opt_slope - base_slope) / (abs(base_slope) + 1e-8)
    cont_change = opt_cont - base_cont
    baimu_count_change = opt_baimu_n - base_baimu_n
    baimu_area_change_ha = (opt_baimu_area - base_baimu_area) / 10000.0

    # Cross-check against mpc_summary.json
    with open(args.summary, "r", encoding="utf-8") as f:
        summary = json.load(f)
    agg = summary["aggregate"]
    reported = {
        "slope_pct": agg["slope_pct_mean"],
        "cont": agg["cont_mean"],
        "baimu_ha": agg["baimu_ha_mean"],
    }
    recomputed = {
        "slope_pct": slope_change_pct,
        "cont": cont_change,
        "baimu_ha": baimu_area_change_ha,
    }
    diff = {
        "slope_pct": recomputed["slope_pct"] - reported["slope_pct"],
        "cont": recomputed["cont"] - reported["cont"],
        "baimu_ha": recomputed["baimu_ha"] - reported["baimu_ha"],
    }

    print()
    print("=" * 70)
    print("VALIDATION REPORT — GIS-recomputed vs MPC-reported")
    print("=" * 70)
    print(f"Parcels in scope          : {len(sub)} (env claimed n_in_env="
          f"{summary['shapefile_output']['n_in_env']})")
    print(f"Swaps farm->forest        : {int((chg==1).sum())} (env reported "
          f"{summary['shapefile_output']['n_farm_to_forest']})")
    print(f"Swaps forest->farm        : {int((chg==2).sum())} (env reported "
          f"{summary['shapefile_output']['n_forest_to_farm']})")
    print()
    print(f"Baseline slope (deg)      : {base_slope:.6f}")
    print(f"Optimized slope (deg)     : {opt_slope:.6f}")
    print(f"  slope change pct        : recomputed {recomputed['slope_pct']:+.4f}%   "
          f"reported {reported['slope_pct']:+.4f}%   diff {diff['slope_pct']:+.4f}")
    print()
    print(f"Baseline contiguity       : {base_cont:.6f}")
    print(f"Optimized contiguity      : {opt_cont:.6f}")
    print(f"  cont change             : recomputed {recomputed['cont']:+.6f}   "
          f"reported {reported['cont']:+.6f}   diff {diff['cont']:+.6f}")
    print()
    print(f"Baseline baimu count      : {base_baimu_n}")
    print(f"Optimized baimu count     : {opt_baimu_n}")
    print(f"  baimu count change      : recomputed {baimu_count_change:+d}")
    print()
    print(f"Baseline baimu area (ha)  : {base_baimu_area/10000:.2f}")
    print(f"Optimized baimu area (ha) : {opt_baimu_area/10000:.2f}")
    print(f"  baimu area change (ha)  : recomputed {recomputed['baimu_ha']:+.4f}   "
          f"reported {reported['baimu_ha']:+.4f}   diff {diff['baimu_ha']:+.4f}")
    print()
    print(f"Total wall time           : {time.time()-t0:.1f}s")
    print("=" * 70)

    # Pass/fail thresholds (loose; tighten as you trust the pipeline more)
    tol_slope_pct = 0.05   # absolute percentage points
    tol_cont = 0.001
    tol_baimu_ha = 1.0     # hectare

    pass_flags = {
        "slope_pct": abs(diff["slope_pct"]) <= tol_slope_pct,
        "cont": abs(diff["cont"]) <= tol_cont,
        "baimu_ha": abs(diff["baimu_ha"]) <= tol_baimu_ha,
    }
    overall_pass = all(pass_flags.values())
    print(f"Pass within tolerances    : slope {pass_flags['slope_pct']}, "
          f"cont {pass_flags['cont']}, baimu {pass_flags['baimu_ha']}  "
          f"-> overall {'PASS' if overall_pass else 'FAIL'}")

    report = {
        "n_parcels_in_scope": int(len(sub)),
        "swap_counts": {
            "farm_to_forest": int((chg == 1).sum()),
            "forest_to_farm": int((chg == 2).sum()),
            "unchanged": int((chg == 0).sum()),
        },
        "baseline": {
            "slope_deg": base_slope,
            "contiguity": base_cont,
            "n_farmland": int(base_n_farm),
            "farm_area_ha": base_farm_area / 10000,
            "baimu_count": int(base_baimu_n),
            "baimu_area_ha": base_baimu_area / 10000,
        },
        "optimized": {
            "slope_deg": opt_slope,
            "contiguity": opt_cont,
            "n_farmland": int(opt_n_farm),
            "farm_area_ha": opt_farm_area / 10000,
            "baimu_count": int(opt_baimu_n),
            "baimu_area_ha": opt_baimu_area / 10000,
        },
        "delta_recomputed": recomputed,
        "delta_reported": reported,
        "delta_diff": diff,
        "tolerance": {
            "slope_pct_abs": tol_slope_pct,
            "cont_abs": tol_cont,
            "baimu_ha_abs": tol_baimu_ha,
        },
        "pass_flags": pass_flags,
        "overall_pass": overall_pass,
        "wall_time_s": time.time() - t0,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report written to {args.out}")

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
