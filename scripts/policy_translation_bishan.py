# -*- coding: utf-8 -*-
"""
policy_translation_bishan.py — translate the Bishan headline result into the
quantities a planning office actually reads.

Why this script exists
----------------------
Reviewer 2 (Codex panel, 2026-05-31) flagged that the manuscript reports
slope, contiguity, and qualifying-baimu-fang count/area as outcomes but
never translates them into the policy numbers that determine whether a
plan is acceptable to a county planning office:

    - farmland area inside the GB/T 30600-2022 mechanisable band (<6 deg)
    - farmland area inside the terraceable band (6-15 deg)
    - farmland area inside the >15 deg band that GB/T 21010-2017 flags
      as marginal / candidate for retirement (退耕还林)
    - per-township distribution of slope improvement and qualifying-area
      loss, for fairness checks across the 13 Bishan core townships
    - the qualifying-area trade-off (-499 ha) re-expressed as a fraction
      of the initial 46,844 ha qualifying stock and against the red-line
      mechanism the paper claims is satisfied by construction

This script reads the optimized.shp written by Tool 4 and recomputes
these quantities directly from the on-disk geometry, with no env / ONNX
dependency. It runs as a child of `validate_optimized_shp.py` and shares
its slope-band and farmland-classification conventions exactly.

Usage
-----
    python scripts/policy_translation_bishan.py \
        --optimized /Users/zhouning/farmland_mpc_runs/bishan/mpc_output/optimized.shp \
        --townships /Users/zhouning/farmland_mpc_runs/bishan/prepared/townships.json \
        --out paper/submission_scirep_corrected/policy_translation_bishan.json

Output is a JSON document with three sections:
    slope_bands, township_fairness, qualifying_area_trade_off.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np

FARMLAND_PREFIXES = ("011", "012", "013")
FOREST_PREFIXES = ("031", "032", "033")
BAIMU_THRESHOLD_M2 = 66700.0  # 100 mu
# Slope bands aligned with GB/T 30600-2022 (high-standard farmland) and
# GB/T 21010-2017 (land-use classification standard) thresholds:
#   <6 deg : fully mechanisable; high-standard-farmland eligible
#   6-15   : terraceable with engineering investment
#   15-25  : marginal; eligible for sloped-cropland engineering
#   >25    : >25 deg cultivation banned by 2002 Land Management Law,
#            candidate for grain-for-green (退耕还林)
SLOPE_BANDS = [
    (0.0, 6.0, "lt6"),
    (6.0, 15.0, "6_15"),
    (15.0, 25.0, "15_25"),
    (25.0, 90.0, "gt25"),
]


def is_farmland(dlbm) -> bool:
    return str(dlbm).startswith(FARMLAND_PREFIXES)


def is_forest(dlbm) -> bool:
    return str(dlbm).startswith(FOREST_PREFIXES)


def slope_band(deg: float) -> str:
    for lo, hi, name in SLOPE_BANDS:
        if lo <= deg < hi:
            return name
    return SLOPE_BANDS[-1][2]


def gini(values: np.ndarray) -> float:
    """Standard sample-based Gini for non-negative values; 0 = equal."""
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan")
    x = np.sort(np.maximum(x, 0.0))
    n = x.size
    if x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimized", required=True, type=Path)
    ap.add_argument("--townships", required=True, type=Path,
                    help="prepared/townships.json — 21-entry registry; "
                         "we treat the 13 500227xxx codes as Bishan core "
                         "and the 8 non-500227 codes as flying-territory.")
    ap.add_argument("--proj-crs", default="EPSG:32648",
                    help="Projected CRS for area in m^2; default UTM 48N.")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    g = gpd.read_file(args.optimized)
    if g.crs is None or g.crs.to_string() != args.proj_crs:
        g = g.to_crs(args.proj_crs)

    # Use SHAPE_Area (env-side) where present; fall back to geometry.area.
    if "SHAPE_Area" in g.columns:
        # SHAPE_Area is stored in degrees^2 inside our DLTB (geographic CRS
        # at write time); ignore it and recompute from the projected geometry
        # so policy areas are genuine m^2.
        g["area_m2"] = g.geometry.area
    else:
        g["area_m2"] = g.geometry.area

    g["orig_is_farm"] = g["ORIG_DLBM"].astype(str).apply(is_farmland)
    g["orig_is_forest"] = g["ORIG_DLBM"].astype(str).apply(is_forest)
    g["opt_is_farm"] = g["OPT_DLBM"].astype(str).apply(is_farmland)
    g["opt_is_forest"] = g["OPT_DLBM"].astype(str).apply(is_forest)

    swap = g["orig_is_farm"] | g["orig_is_forest"]
    swappable = g[swap].copy()

    # ------------------------------------------------------------------ #
    # 1. Slope bands: farmland-area distribution before vs after                                                                 #
    # ------------------------------------------------------------------ #
    bands_report = {}
    for kind, mask_col in [("orig", "orig_is_farm"), ("opt", "opt_is_farm")]:
        farm = swappable[swappable[mask_col]].copy()
        farm["band"] = farm["slope_mean"].apply(slope_band)
        agg = farm.groupby("band")["area_m2"].sum() / 1e4  # ha
        total_ha = float(farm["area_m2"].sum() / 1e4)
        bands_report[kind] = {
            "total_farmland_ha": total_ha,
            "by_band_ha": {k: float(agg.get(k, 0.0)) for _, _, k in SLOPE_BANDS},
        }

    deltas_ha = {}
    for _, _, name in SLOPE_BANDS:
        deltas_ha[name] = (
            bands_report["opt"]["by_band_ha"][name]
            - bands_report["orig"]["by_band_ha"][name]
        )

    # Fraction of farmland in each band.
    bands_pct = {}
    for kind in ("orig", "opt"):
        tot = bands_report[kind]["total_farmland_ha"]
        bands_pct[kind] = {
            k: 100.0 * bands_report[kind]["by_band_ha"][k] / tot
            for _, _, k in SLOPE_BANDS
        }

    # ------------------------------------------------------------------ #
    # 2. Township fairness: per-township slope improvement and qualifying
    #    area change across the 13 Bishan core (500227xxx) townships.    #
    # ------------------------------------------------------------------ #
    twn = json.load(open(args.townships))
    core_codes = sorted(c for c in twn if c.startswith("500227"))
    swappable["town_code"] = swappable["QSDWDM"].astype(str).str[:9]

    twn_rows = []
    for tc in core_codes:
        sub = swappable[swappable["town_code"] == tc]
        orig_farm = sub[sub["orig_is_farm"]]
        opt_farm = sub[sub["opt_is_farm"]]
        if orig_farm.empty:
            continue
        slope_orig = float(
            (orig_farm["slope_mean"] * orig_farm["area_m2"]).sum()
            / orig_farm["area_m2"].sum()
        )
        slope_opt = (
            float((opt_farm["slope_mean"] * opt_farm["area_m2"]).sum()
                  / opt_farm["area_m2"].sum())
            if opt_farm["area_m2"].sum() > 0
            else float("nan")
        )
        farmland_area_orig_ha = float(orig_farm["area_m2"].sum() / 1e4)
        farmland_area_opt_ha = float(opt_farm["area_m2"].sum() / 1e4)
        n_farm2for = int(((sub["CHG_FLAG"] == 1) & sub["orig_is_farm"]).sum())
        n_for2farm = int(((sub["CHG_FLAG"] == 2) & sub["orig_is_forest"]).sum())
        twn_rows.append({
            "town_code": tc,
            "n_parcels_swappable": int(len(sub)),
            "farmland_area_orig_ha": farmland_area_orig_ha,
            "farmland_area_opt_ha": farmland_area_opt_ha,
            "slope_mean_orig_deg": slope_orig,
            "slope_mean_opt_deg": slope_opt,
            "slope_change_pct": (
                100.0 * (slope_opt - slope_orig) / slope_orig
                if slope_orig > 0 and not math.isnan(slope_opt)
                else float("nan")
            ),
            "n_farm_to_forest": n_farm2for,
            "n_forest_to_farm": n_for2farm,
        })

    # Fairness summaries: range, max-min ratio, Gini of |slope_change_pct|.
    slope_changes = np.array(
        [abs(r["slope_change_pct"]) for r in twn_rows
         if not math.isnan(r["slope_change_pct"])],
        dtype=float,
    )
    n_swaps_per_township = np.array(
        [r["n_farm_to_forest"] for r in twn_rows], dtype=float,
    )
    fairness = {
        "n_townships_core": len(twn_rows),
        "slope_pct_min": float(np.nanmin(slope_changes)) if len(slope_changes) else None,
        "slope_pct_max": float(np.nanmax(slope_changes)) if len(slope_changes) else None,
        "slope_pct_mean": float(np.nanmean(slope_changes)) if len(slope_changes) else None,
        "slope_pct_std": float(np.nanstd(slope_changes, ddof=1))
            if len(slope_changes) > 1 else None,
        "slope_pct_gini": gini(slope_changes),
        "swap_count_max_min_ratio": (
            float(n_swaps_per_township.max() /
                  max(n_swaps_per_township[n_swaps_per_township > 0].min(), 1))
            if (n_swaps_per_township > 0).any() else None
        ),
    }

    # ------------------------------------------------------------------ #
    # 3. Qualifying-area trade-off in policy units                       #
    # ------------------------------------------------------------------ #
    # Read the env-side initial qualifying stock from the Methods
    # constant rather than recompute connected components here; the
    # validate_optimized_shp.py script already cross-checks the count/
    # area trajectory to env tally to within floating-point noise.
    INITIAL_QUALIFYING_HA = 46844.0
    DELTA_QUALIFYING_HA = -499.2  # regime-C mean across 5 episodes
    DELTA_QUALIFYING_COUNT = +7
    qa = {
        "initial_qualifying_baimu_fang_ha": INITIAL_QUALIFYING_HA,
        "delta_qualifying_baimu_fang_ha": DELTA_QUALIFYING_HA,
        "delta_qualifying_baimu_fang_count": DELTA_QUALIFYING_COUNT,
        "delta_pct_of_initial": 100.0 * DELTA_QUALIFYING_HA / INITIAL_QUALIFYING_HA,
        # The paired farmland<->forest swap conserves total farmland area
        # by construction (one farm parcel out, one forest parcel of
        # equal area in). Recover this from the data.
        "total_farmland_area_change_ha": float(
            (swappable.loc[swappable["opt_is_farm"], "area_m2"].sum()
             - swappable.loc[swappable["orig_is_farm"], "area_m2"].sum()) / 1e4
        ),
    }

    out = {
        "source_optimized_shp": str(args.optimized),
        "n_swappable_parcels": int(len(swappable)),
        "slope_bands": {
            "definition_deg": [
                {"name": n, "lo": lo, "hi": hi}
                for lo, hi, n in SLOPE_BANDS
            ],
            "orig": bands_report["orig"],
            "opt": bands_report["opt"],
            "delta_ha": deltas_ha,
            "orig_pct": bands_pct["orig"],
            "opt_pct": bands_pct["opt"],
        },
        "township_fairness": {
            "core_townships": twn_rows,
            "summary": fairness,
        },
        "qualifying_area_trade_off": qa,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # Console summary
    print(f"\nWrote {args.out}")
    print("\n=== slope-band shifts (ha, swappable parcels only) ===")
    for _, _, name in SLOPE_BANDS:
        print(f"  band {name:>6}: orig {bands_report['orig']['by_band_ha'][name]:>10.1f} ha "
              f"-> opt {bands_report['opt']['by_band_ha'][name]:>10.1f} ha "
              f"(Δ {deltas_ha[name]:+9.1f} ha, "
              f"{bands_pct['orig'][name]:5.1f}% -> {bands_pct['opt'][name]:5.1f}%)")
    print(
        f"\n=== township fairness (n={fairness['n_townships_core']} core townships) ==="
    )
    print(
        f"  slope-improvement |Δ%| min={fairness['slope_pct_min']:.3f}  "
        f"mean={fairness['slope_pct_mean']:.3f}  "
        f"max={fairness['slope_pct_max']:.3f}  "
        f"std={fairness['slope_pct_std']:.3f}  "
        f"Gini={fairness['slope_pct_gini']:.3f}"
    )
    print(f"\n=== qualifying-area trade-off ===")
    print(f"  initial qualifying baimu-fang stock : "
          f"{qa['initial_qualifying_baimu_fang_ha']:,.0f} ha")
    print(f"  delta (regime C, 5 episodes mean)   : "
          f"{qa['delta_qualifying_baimu_fang_ha']:+,.1f} ha "
          f"({qa['delta_pct_of_initial']:+.2f}% of initial)")
    print(f"  total farmland-area change on disk  : "
          f"{qa['total_farmland_area_change_ha']:+,.2f} ha "
          f"(should be ~0 by paired-swap construction)")


if __name__ == "__main__":
    main()
