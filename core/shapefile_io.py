"""DLTB shapefile / feature class output helpers for Tool 4 v0.2.

Writes an optimized DLTB feature class with these added fields:

    OPT_DLBM   Text(3)   - optimized 3-digit land-use code
    OPT_DLMC   Text(40)  - optimized land-use Chinese name (looked up via DLMC dict)
    CHG_FLAG   Short     - 0=unchanged, 1=farm->forest, 2=forest->farm
    ORIG_DLBM  Text(3)   - original DLBM (preserved for audit)

Convention follows the Paper 1 toolbox (OPT_DLMC, CHG_FLAG, ORIG_DLMC), with
the difference that Paper 9 operates on DLBM codes -- DLMC is derived from
the input feature's existing DLMC field (Chinese name, looked up per row).
"""
import os

import numpy as np


# Default representative DLBM codes for the "after" state when a swap happens.
# Users can override via Tool 4 UI parameters.
DEFAULT_FARM_DLBM = "011"   # 水田 (paddy)
DEFAULT_FOREST_DLBM = "031" # 有林地 (forest)

# CountyLevelEnv land_use enum (mirrors county_env.FARMLAND/FOREST)
ENV_FARMLAND = 1
ENV_FOREST = 2

# Shapefile DBF text field cap
SHP_TEXT_MAX = 254


def write_optimized_dltb(input_fc, output_fc, env,
                         farm_dlbm=DEFAULT_FARM_DLBM,
                         forest_dlbm=DEFAULT_FOREST_DLBM,
                         messages=None):
    """Copy input_fc to output_fc, then write OPT_* / CHG_FLAG fields.

    Parameters
    ----------
    input_fc : str
        Path or feature class reference for the source DLTB. Required fields:
        BSM (Text or Long), DLBM (Text), DLMC (Text). All other fields are
        copied through unchanged.
    output_fc : str
        Output feature class path. Will be overwritten if it exists.
    env : CountyLevelEnv with attached _parcel_bsm
        Built by core.blocks_env.make_env(). Provides:
            env._parcel_bsm     (n_parcels,) BSM values aligned with env indices
            env.initial_types   (n_parcels,) int8: 1=farm 2=forest
            env.land_use        (n_parcels,) int8: post-MPC types
    farm_dlbm, forest_dlbm : str
        DLBM codes to write for forest->farm and farm->forest swaps respectively.
    messages : arcpy messages or None

    Returns
    -------
    dict with counts: n_input, n_in_env, n_farm_to_forest, n_forest_to_farm
    """
    import arcpy

    def _say(msg, level="info"):
        if messages is not None:
            getattr(messages, "addMessage" if level == "info"
                    else "addWarningMessage")(msg)
        print(msg, flush=True)

    # Build BSM -> env_idx lookup. BSM may be float (from gpkg) or int/str
    # (from shapefile). Normalize to string for cross-format matching.
    bsm_arr = env._parcel_bsm
    bsm_to_env_idx = {}
    for i, bsm in enumerate(bsm_arr):
        bsm_to_env_idx[_norm_bsm(bsm)] = i

    initial_types = env.initial_types
    final_types = env.land_use

    # Lookup table: any forest DLBM in the source to a "forest" representative,
    # any farm DLBM to a "farm" representative. We do not change DLMC text by
    # ourselves -- we copy the row's DLMC if unchanged, or look it up from a
    # per-row DLMC->DLBM cache built from the input.
    farm_dlbm_set = {"011", "012", "013"}
    forest_dlbm_set = {"031", "032", "033"}

    _say(f"[shp_out] Copying {input_fc} -> {output_fc} ...")
    if arcpy.Exists(output_fc):
        arcpy.management.Delete(output_fc)

    # arcpy <-> gpkg layer naming compatibility: gpkg layers written by
    # geopandas with a dotted name (e.g. "main.DLTB") get re-listed by
    # arcpy as "main.main.DLTB" (sqlite schema "main" + the dotted layer
    # treated as a name). Auto-resolve.
    input_fc = _resolve_gpkg_path(input_fc)

    # CopyFeatures to .shp fails on BigInteger columns and on Text columns
    # wider than 254 chars (DBF limit). Build a FieldMappings that drops
    # unsupported / oversized fields before copying.
    fms = _build_shp_safe_field_mappings(input_fc)
    out_path = str(output_fc)
    out_dir = os.path.dirname(out_path)
    out_name = os.path.basename(out_path)
    if not out_dir:
        raise ValueError(f"output_fc must be an absolute path: {output_fc}")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    arcpy.conversion.FeatureClassToFeatureClass(
        in_features=input_fc, out_path=out_dir, out_name=out_name,
        field_mapping=fms,
    )

    # Add fields. CopyFeatures preserved the original DLBM and DLMC.
    arcpy.management.AddField(output_fc, "OPT_DLBM", "TEXT", field_length=3)
    arcpy.management.AddField(output_fc, "OPT_DLMC", "TEXT", field_length=40)
    arcpy.management.AddField(output_fc, "CHG_FLAG", "SHORT")
    arcpy.management.AddField(output_fc, "ORIG_DLBM", "TEXT", field_length=3)

    # Build a DLBM -> DLMC lookup from the input itself, so OPT_DLMC matches
    # the user's source data (Chinese name conventions vary slightly).
    dlbm_to_dlmc = {}
    with arcpy.da.SearchCursor(input_fc, ["DLBM", "DLMC"]) as cur:
        for dlbm, dlmc in cur:
            if dlbm and dlmc and dlbm not in dlbm_to_dlmc:
                dlbm_to_dlmc[str(dlbm).strip()] = str(dlmc).strip()
    # Sane fallbacks if the input data didn't include the chosen reps.
    dlbm_to_dlmc.setdefault(farm_dlbm, "耕地")    # 耕地
    dlbm_to_dlmc.setdefault(forest_dlbm, "林地")  # 林地

    n_input = 0
    n_in_env = 0
    n_farm_to_forest = 0
    n_forest_to_farm = 0
    n_unchanged = 0

    fields = ["BSM", "DLBM", "DLMC", "OPT_DLBM", "OPT_DLMC", "CHG_FLAG", "ORIG_DLBM"]
    with arcpy.da.UpdateCursor(output_fc, fields) as cur:
        for row in cur:
            n_input += 1
            bsm, dlbm, dlmc = row[0], row[1], row[2]
            orig_dlbm = str(dlbm).strip() if dlbm else ""
            row[6] = orig_dlbm  # ORIG_DLBM

            env_idx = bsm_to_env_idx.get(_norm_bsm(bsm))
            if env_idx is None:
                # Parcel not in env (e.g. township code didn't match, or
                # not a swappable type). Pass through.
                row[3] = orig_dlbm
                row[4] = dlmc if dlmc else ""
                row[5] = 0
                cur.updateRow(row)
                continue

            n_in_env += 1
            init = int(initial_types[env_idx])
            fin = int(final_types[env_idx])

            if init == ENV_FARMLAND and fin == ENV_FOREST:
                row[3] = forest_dlbm
                row[4] = dlbm_to_dlmc.get(forest_dlbm, "")
                row[5] = 1
                n_farm_to_forest += 1
            elif init == ENV_FOREST and fin == ENV_FARMLAND:
                row[3] = farm_dlbm
                row[4] = dlbm_to_dlmc.get(farm_dlbm, "")
                row[5] = 2
                n_forest_to_farm += 1
            else:
                row[3] = orig_dlbm
                row[4] = dlmc if dlmc else ""
                row[5] = 0
                n_unchanged += 1

            cur.updateRow(row)

    _say(f"[shp_out] {n_input} input rows, {n_in_env} matched to env "
         f"({n_input - n_in_env} pass-through)")
    _say(f"[shp_out] swaps: farm->forest={n_farm_to_forest}, "
         f"forest->farm={n_forest_to_farm}, unchanged={n_unchanged}")
    if n_farm_to_forest != n_forest_to_farm:
        _say(f"[shp_out] WARNING: farmland count delta = "
             f"{n_forest_to_farm - n_farm_to_forest} (Paper 9 does NOT "
             "guarantee FC=0; this is expected)", level="warn")
    return {
        "n_input": n_input, "n_in_env": n_in_env,
        "n_farm_to_forest": n_farm_to_forest,
        "n_forest_to_farm": n_forest_to_farm,
        "n_unchanged": n_unchanged,
    }


def _norm_bsm(v):
    """Normalize BSM value to a string key, handling float/int/str variants.

    Numeric BSMs sometimes come through as 821839.0 (float), sometimes as
    821839 (int), sometimes as '821839' (text). We strip the .0 suffix on
    float values to make all three collide on the same key.
    """
    if v is None:
        return ""
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return f"{v:.6f}"
    return str(v).strip()


def _resolve_gpkg_path(input_fc):
    """If input_fc looks like a gpkg layer path that arcpy can't find,
    try alternative arcpy-style paths.

    geopandas/pyogrio writes gpkg layers with names like 'main.DLTB'.
    arcpy lists them as 'main.main.DLTB' (schema prefix). We try three
    variants in order:
        1. the original path
        2. inserting a 'main' schema prefix before a dotted layer name
           (foo.gpkg/main.DLTB -> foo.gpkg/main.main.DLTB)
        3. stripping a leading 'main.' from the layer
           (foo.gpkg/main.DLTB -> foo.gpkg/DLTB)

    Returns the first path arcpy.Exists for.
    """
    import arcpy
    s = str(input_fc)
    if ".gpkg" not in s.lower():
        return input_fc
    if arcpy.Exists(s):
        return s

    lower = s.lower()
    idx = lower.find(".gpkg")
    after = s[idx + len(".gpkg"):]
    if not after or not after.startswith(("/", "\\")):
        return s
    prefix = s[:idx + len(".gpkg")]
    sep = after[0]
    layer = after[1:]

    # Variant A: schema prefix inserted
    candidate_a = f"{prefix}{sep}main.{layer}"
    if arcpy.Exists(candidate_a):
        return candidate_a
    # Variant B: leading 'main.' stripped
    if layer.startswith("main."):
        candidate_b = f"{prefix}{sep}{layer[len('main.'):]}"
        if arcpy.Exists(candidate_b):
            return candidate_b
    return s


def _build_shp_safe_field_mappings(input_fc):
    """Return a FieldMappings that skips BigInteger fields and truncates
    oversized Text fields so the result can be copied to a shapefile.

    - BigInteger (Int64): dropped entirely
    - Text with length > 254: clamp output length to 254
    - Everything else: pass-through
    """
    import arcpy

    fms = arcpy.FieldMappings()
    for f in arcpy.ListFields(input_fc):
        # Skip geometry / OID / implicit fields (FeatureClassToFeatureClass
        # handles these automatically).
        if f.type in ("OID", "Geometry"):
            continue
        if f.name.lower() in ("shape", "shape_length", "shape_area", "geom", "fid"):
            continue
        # Drop BigInteger (shp has no Int64 counterpart)
        if f.type == "BigInteger":
            continue
        fm = arcpy.FieldMap()
        try:
            fm.addInputField(input_fc, f.name)
        except Exception:
            continue
        out_f = fm.outputField
        # Clamp oversized text to DBF limit
        if f.type == "String" and (f.length or 0) > SHP_TEXT_MAX:
            out_f.length = SHP_TEXT_MAX
        fm.outputField = out_f
        fms.addFieldMap(fm)
    return fms
