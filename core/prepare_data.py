"""Tool 1: Prepare Data & Blocks.

Input:
    - DLTB feature class (Third National Land Survey polygons)
    - XZQ feature class (administrative boundaries; used for township codes)
    - DEM: user-supplied raster OR auto-download Copernicus GLO-30

Output (under prepared_dir, matching Tool 4 v0.2's consumption layout):
    dem_slope_analysis/output/DLTB_with_slope.shp     (with 'slope_mean' field)
    results_real/blocks/township_<code>/block_compositions.json
    results_real/blocks/township_<code>/block_features.json
    results_real/blocks/township_<code>/parcel_block_mapping.csv
    townships.json                                    (code -> label)
    prepare_data_summary.json                         (provenance)

Phases:
    A. DEM -> slope (Spatial Analyst if user-supplied raster;
                     vendored dem_slope_zonal.py for auto-download)
    B. Blocks (monkey-patch block_definition.py; Paper 3 hybrid barrier +
               AgglomerativeClustering)
    C. townships.json + consistency sanity

Runtime estimate (county scale, ~50k parcels):
    A: 5-10 min (raster) / 15-30 min (auto DL)
    B: 10-30 min depending on township count
    C: <10s
    Total: 20-60 min
"""

import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Entry point
# =============================================================================
def run(dltb_fc, xzq_fc, prepared_dir,
        dem_mode="user", dem_raster=None,
        dlbm_field="DLBM", qsdwdm_field="QSDWDM",
        xzq_code_field="XZQDM", xzq_name_field="XZQMC",
        reference_layer=None, reference_name_field="乡",
        proj_crs="EPSG:32648",
        min_parcels=3, min_area_ha=0.5, max_parcels=30,
        messages=None):
    """Build all Tool 4 consumables under prepared_dir.

    Parameters
    ----------
    dltb_fc : str
        Path to DLTB feature class (Polygon). Required fields:
        BSM (text or int), DLBM (text, 3-digit), QSDWDM (text, 9+ digits).
    xzq_fc : str or None
        Path to XZQ feature class. Optional -- if None, townships are
        extracted from DLTB.QSDWDM instead. When provided, used only to
        obtain Chinese township labels via xzq_name_field.
    prepared_dir : str
        Output directory root.
    dem_mode : {"user", "auto"}
        "user": use dem_raster; "auto": download Copernicus GLO-30 tiles
        (NOT IMPLEMENTED in v1).
    dem_raster : str or None
        Required if dem_mode == "user".
    dlbm_field, qsdwdm_field : str
        Column names in dltb_fc (default Third-Survey standard).
    xzq_code_field, xzq_name_field : str
        Column names in xzq_fc (default XZQDM / XZQMC). Only used if
        xzq_fc is provided.
    reference_layer : str or None
        Optional national/regional township polygon layer. If provided,
        township Chinese labels are derived by spatial-joining DLTB
        centroids into this layer's reference_name_field. Used as a
        fallback when XZQ is unavailable.
        Should NOT be combined with xzq_fc -- if both are set, XZQ wins.
    reference_name_field : str
        Column in reference_layer holding the township Chinese name
        (default "乡").
    proj_crs : str
        Projected CRS for area / slope calculations. Default UTM 48N
        (valid for central-west China). Users in other zones must override.
    min_parcels, min_area_ha, max_parcels : int, float, int
        Block filtering parameters (Paper 3 defaults).
    messages : arcpy messages or None.

    Returns
    -------
    summary dict (also written to <prepared_dir>/prepare_data_summary.json).
    """
    def _say(msg, level="info"):
        if messages is not None:
            getattr(messages,
                    "addMessage" if level == "info" else "addWarningMessage")(msg)
        logger.info(msg)
        print(msg, flush=True)

    prepared_dir = Path(prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    log_path = prepared_dir / "prepare_data.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)

    summary = {"config": dict(
        dltb_fc=str(dltb_fc), xzq_fc=str(xzq_fc) if xzq_fc else None,
        prepared_dir=str(prepared_dir),
        dem_mode=dem_mode, dem_raster=str(dem_raster) if dem_raster else None,
        dlbm_field=dlbm_field, qsdwdm_field=qsdwdm_field,
        xzq_code_field=xzq_code_field, xzq_name_field=xzq_name_field,
        reference_layer=str(reference_layer) if reference_layer else None,
        reference_name_field=reference_name_field,
        proj_crs=proj_crs,
        min_parcels=min_parcels, min_area_ha=min_area_ha,
        max_parcels=max_parcels,
    )}
    t_total = time.time()

    try:
        _say(f"[Tool 1] Preparing data under {prepared_dir}")
        _say(f"  DLTB: {dltb_fc}")
        _say(f"  XZQ:  {xzq_fc}")
        _say(f"  DEM mode: {dem_mode}")
        _say(f"  proj_crs: {proj_crs}")

        # ---- Phase A: DEM -> slope ----
        t_a = time.time()
        _say("\n[Phase A] Computing per-parcel slope_mean ...")
        shp_out = _phase_a_slope(
            dltb_fc=dltb_fc, prepared_dir=prepared_dir,
            dem_mode=dem_mode, dem_raster=dem_raster,
            dlbm_field=dlbm_field, qsdwdm_field=qsdwdm_field,
            proj_crs=proj_crs, messages=messages,
        )
        summary["phase_a"] = {
            "elapsed_s": round(time.time() - t_a, 1),
            "output_shapefile": str(shp_out),
        }
        _say(f"[Phase A] done in {summary['phase_a']['elapsed_s']}s -> {shp_out}")

        # ---- Phase C1: townships.json (needed before Phase B) ----
        _say("\n[Phase C1] Building townships.json ...")
        townships = _extract_townships(
            dltb_fc=dltb_fc, xzq_fc=xzq_fc,
            qsdwdm_field=qsdwdm_field,
            xzq_code_field=xzq_code_field, xzq_name_field=xzq_name_field,
            messages=messages,
        )

        # Optional: override labels via spatial join with a reference layer
        # (e.g. national 1:5M xiangzhen.shp). Only triggers when XZQ did NOT
        # already inject useful labels (all labels still == codes).
        label_source = "code"
        if xzq_fc:
            labels_still_codes = all(v == k for k, v in townships.items())
            if not labels_still_codes:
                label_source = "xzq"
        if reference_layer and label_source == "code":
            _say(f"  [Phase C1] Joining labels from reference layer {reference_layer} ...")
            label_map = _labels_from_reference_layer(
                dltb_fc=dltb_fc, qsdwdm_field=qsdwdm_field,
                reference_layer=reference_layer,
                reference_name_field=reference_name_field,
                proj_crs=proj_crs, messages=messages,
                dltb_gpkg_layer=str(shp_out),
            )
            resolved = 0
            for code in townships.keys():
                if code in label_map and label_map[code]:
                    townships[code] = label_map[code]
                    resolved += 1
            _say(f"  [Phase C1] reference layer resolved {resolved}/{len(townships)} "
                 "Chinese labels")
            label_source = "reference"

        townships_path = prepared_dir / "townships.json"
        with open(townships_path, "w", encoding="utf-8") as f:
            json.dump(townships, f, ensure_ascii=False, indent=2)
        _say(f"[Phase C1] {len(townships)} townships -> {townships_path} "
             f"(labels from: {label_source})")
        summary["townships"] = dict(
            n_townships=len(townships),
            codes=list(townships.keys()),
            label_source=label_source,
            file=str(townships_path),
        )

        # ---- Phase B: blocks ----
        t_b = time.time()
        _say("\n[Phase B] Defining blocks (Paper 3 hybrid) ...")
        blocks_info = _phase_b_blocks(
            prepared_dir=prepared_dir, townships=townships,
            proj_crs=proj_crs,
            min_parcels=min_parcels, min_area_ha=min_area_ha,
            max_parcels=max_parcels, messages=messages,
        )
        summary["phase_b"] = {
            "elapsed_s": round(time.time() - t_b, 1),
            **blocks_info,
        }
        _say(f"[Phase B] done in {summary['phase_b']['elapsed_s']}s, "
             f"{blocks_info['total_blocks']} blocks total across "
             f"{blocks_info['n_townships_processed']} townships")

        # ---- Phase C2: sanity via make_env ----
        _say("\n[Phase C2] Sanity: try make_env(prepared_dir) ...")
        sanity = _phase_c_sanity(prepared_dir, proj_crs, messages=messages)
        summary["phase_c_sanity"] = sanity

        summary["total_elapsed_s"] = round(time.time() - t_total, 1)
        summary["status"] = "ok"

        summary_path = prepared_dir / "prepare_data_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        _say(f"\n[Tool 1] Total {summary['total_elapsed_s']}s. "
             f"Summary -> {summary_path}")
        return summary
    finally:
        logging.getLogger().removeHandler(fh)
        fh.close()


# =============================================================================
# Phase A: DEM -> slope_mean per parcel
# =============================================================================
def _phase_a_slope(dltb_fc, prepared_dir, dem_mode, dem_raster,
                   dlbm_field, qsdwdm_field, proj_crs, messages):
    """Compute per-parcel slope_mean and write
    <prepared>/dem_slope_analysis/output/DLTB_with_slope.shp.

    Note: historical code paths (Paper 9 v6 adk pipeline + this toolbox
    v1.1) wrote a .gpkg here. We switched to .shp in v1.2 because both
    arcpy and geopandas handle shapefiles reliably, while gpkg produced
    by one tool is often unreadable by the other. Field names used here
    (BSM / DLBM / DLMC / QSDWDM / slope_mean) are all <= 10 chars so they
    survive the DBF cap cleanly.
    """
    import arcpy

    out_dir = prepared_dir / "dem_slope_analysis" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    shp_out = out_dir / "DLTB_with_slope.shp"

    if dem_mode == "auto":
        raise NotImplementedError(
            "Auto-download DEM branch not implemented in Tool 1 v1. "
            "Use 'user' mode with a local DEM raster for now."
        )
    if not dem_raster:
        raise ValueError("dem_mode='user' requires dem_raster")

    if arcpy.CheckExtension("Spatial") != "Available":
        raise RuntimeError(
            "Spatial Analyst extension is required for DEM slope calculation "
            "but is not available in this license."
        )
    arcpy.CheckOutExtension("Spatial")
    try:
        _phase_a_arcgis(
            dltb_fc=dltb_fc, dem_raster=dem_raster, shp_out=shp_out,
            proj_crs=proj_crs, dlbm_field=dlbm_field,
            qsdwdm_field=qsdwdm_field, messages=messages,
        )
    finally:
        arcpy.CheckInExtension("Spatial")
    return shp_out


def _phase_a_arcgis(dltb_fc, dem_raster, shp_out, proj_crs,
                    dlbm_field, qsdwdm_field, messages):
    """arcpy Spatial Analyst version:
        1. project DEM to proj_crs (meters) if it's geographic (degrees)
        2. arcpy.sa.Slope (degrees)
        3. arcpy.sa.ZonalStatisticsAsTable on DLTB (FID zone)
        4. Join slope_mean back to DLTB and write to gpkg_out
    """
    import arcpy
    from arcpy.sa import Slope
    from core.shapefile_io import _build_shp_safe_field_mappings

    def _say(m, level="info"):
        if messages is not None:
            getattr(messages,
                    "addMessage" if level == "info" else "addWarningMessage")(m)
        print(m, flush=True)

    scratch = arcpy.env.scratchGDB
    _say(f"  scratchGDB: {scratch}")

    # Validate DLTB fields
    dltb_fields = {f.name for f in arcpy.ListFields(dltb_fc)}
    for req in ("BSM", dlbm_field, qsdwdm_field):
        if req not in dltb_fields:
            raise RuntimeError(f"DLTB missing required field: {req}")

    # --- Project DEM to meters if needed ---
    dem_desc = arcpy.Describe(dem_raster)
    dem_sr = dem_desc.spatialReference
    target_sr = arcpy.SpatialReference()
    target_sr.loadFromString(proj_crs.replace("EPSG:", ""))
    # Try interpreting proj_crs as factoryCode
    try:
        target_sr = arcpy.SpatialReference(int(proj_crs.split(":")[-1]))
    except Exception:
        raise ValueError(f"Cannot parse proj_crs={proj_crs}")

    if dem_sr.type == "Geographic" or dem_sr.factoryCode != target_sr.factoryCode:
        _say(f"  Projecting DEM {dem_sr.name} -> {target_sr.name} ...")
        dem_proj = os.path.join(scratch, "dem_proj")
        if arcpy.Exists(dem_proj):
            arcpy.management.Delete(dem_proj)
        arcpy.management.ProjectRaster(
            in_raster=dem_raster, out_raster=dem_proj,
            out_coor_system=target_sr, resampling_type="BILINEAR",
        )
    else:
        dem_proj = dem_raster

    # --- Compute slope (degrees) ---
    _say("  Computing slope raster (degrees) ...")
    slope_raster = Slope(dem_proj, "DEGREE")
    slope_path = os.path.join(scratch, "slope_deg")
    if arcpy.Exists(slope_path):
        arcpy.management.Delete(slope_path)
    slope_raster.save(slope_path)

    # --- Project DLTB to same CRS, preserving attrs, and add OID_ZONE ---
    _say(f"  Projecting DLTB to {target_sr.name} ...")
    dltb_proj = os.path.join(scratch, "dltb_proj")
    if arcpy.Exists(dltb_proj):
        arcpy.management.Delete(dltb_proj)
    arcpy.management.Project(dltb_fc, dltb_proj, target_sr)

    # --- Zonal statistics: mean slope per FID zone ---
    _say("  Running ZonalStatisticsAsTable (this is the slow step) ...")
    zones_oid_field = arcpy.Describe(dltb_proj).OIDFieldName
    zstat_table = os.path.join(scratch, "zstat_slope")
    if arcpy.Exists(zstat_table):
        arcpy.management.Delete(zstat_table)
    arcpy.sa.ZonalStatisticsAsTable(
        in_zone_data=dltb_proj, zone_field=zones_oid_field,
        in_value_raster=slope_path, out_table=zstat_table,
        ignore_nodata="DATA", statistics_type="MEAN",
    )

    # --- Join MEAN back to dltb_proj as slope_mean ---
    _say("  Joining slope MEAN -> dltb_proj.slope_mean ...")
    arcpy.management.AddField(dltb_proj, "slope_mean", "DOUBLE")

    # ZonalStatisticsAsTable output typically has columns:
    #   OBJECTID       -- the zstat row's own PK (1..n_zones), NOT the zone OID
    #   OBJECTID_1     -- the zone OID from the input feature class (what we want)
    #   COUNT, AREA, MEAN, ...
    # If the input's OID column wasn't called OBJECTID, the zstat column is
    # named after it directly; otherwise it's suffixed with _1.
    z_fields = [f.name for f in arcpy.ListFields(zstat_table)]
    if zones_oid_field == "OBJECTID":
        # Zstat renames the input's OID column to OBJECTID_1 to avoid colliding
        # with its own PK
        zstat_zone_col = "OBJECTID_1" if "OBJECTID_1" in z_fields else "OBJECTID"
    else:
        # Non-OBJECTID zone fields are passed through unchanged
        zstat_zone_col = zones_oid_field
    if zstat_zone_col not in z_fields:
        raise RuntimeError(
            f"Cannot find zone OID column in zstat table. Expected "
            f"{zstat_zone_col}, got {z_fields}"
        )
    _say(f"  zstat zone-OID column: {zstat_zone_col}")
    oid_to_mean = {}
    with arcpy.da.SearchCursor(zstat_table, [zstat_zone_col, "MEAN"]) as cur:
        for oid, m in cur:
            if m is not None:
                oid_to_mean[oid] = float(m)
    n_unmatched = 0
    with arcpy.da.UpdateCursor(dltb_proj, [zones_oid_field, "slope_mean"]) as cur:
        for row in cur:
            m = oid_to_mean.get(row[0])
            if m is None:
                n_unmatched += 1
                row[1] = None
            else:
                row[1] = m
            cur.updateRow(row)
    _say(f"  {len(oid_to_mean)} parcels got slope_mean; {n_unmatched} unmatched "
         "(likely outside DEM or tiny polygons)")

    # --- Export to shapefile ---
    _say(f"  Writing {shp_out} ...")
    if arcpy.Exists(str(shp_out)):
        arcpy.management.Delete(str(shp_out))
    arcpy.conversion.FeatureClassToFeatureClass(
        in_features=dltb_proj,
        out_path=str(shp_out.parent),
        out_name=shp_out.name,
        field_mapping=_build_shp_safe_field_mappings(dltb_proj),
    )
    n_rows = int(arcpy.management.GetCount(str(shp_out))[0])
    _say(f"  [Phase A] wrote shapefile: {shp_out} ({n_rows} rows)")


# =============================================================================
# Phase B: blocks (monkey-patch block_definition.py)
# =============================================================================
def _phase_b_blocks(prepared_dir, townships, proj_crs,
                    min_parcels, min_area_ha, max_parcels, messages):
    """Run block_definition.define_blocks + save_results per township."""
    def _say(m, level="info"):
        if messages is not None:
            getattr(messages,
                    "addMessage" if level == "info" else "addWarningMessage")(m)
        print(m, flush=True)

    # Import block_definition (sibling module under core/)
    core_dir = str(Path(__file__).resolve().parent)
    if core_dir not in sys.path:
        sys.path.insert(0, core_dir)
    import block_definition as bd  # noqa: E402

    shp_path = prepared_dir / "dem_slope_analysis" / "output" / "DLTB_with_slope.shp"
    blocks_root = prepared_dir / "results_real" / "blocks"
    blocks_root.mkdir(parents=True, exist_ok=True)

    # block_definition reads DLTB_PATH as a string and passes it to
    # gpd.read_file(), which handles .shp natively.
    bd.DLTB_PATH = str(shp_path)
    bd.OUTPUT_DIR = str(blocks_root)
    bd.PROJ_CRS = proj_crs
    bd.TOWNSHIPS = dict(townships)

    info = {
        "n_townships_processed": 0,
        "n_townships_skipped": 0,
        "total_blocks": 0,
        "per_township": {},
    }

    for code, label in sorted(townships.items()):
        _say(f"  [Phase B] Township {code} ({label}) ...")
        try:
            gdf_sw, block_features, valid_blocks = bd.define_blocks(
                code, min_parcels=min_parcels,
                min_area_ha=min_area_ha, max_parcels=max_parcels,
            )
        except Exception as e:
            _say(f"  [Phase B] Township {code} FAILED: {e}", level="warn")
            info["n_townships_skipped"] += 1
            info["per_township"][code] = {"status": "failed", "error": str(e)}
            continue

        if len(valid_blocks) == 0:
            _say(f"  [Phase B] Township {code} -> 0 blocks, skipping save",
                 level="warn")
            info["n_townships_skipped"] += 1
            info["per_township"][code] = {"status": "empty", "n_blocks": 0}
            continue

        bd.save_results(code, gdf_sw, block_features, valid_blocks)
        info["n_townships_processed"] += 1
        info["total_blocks"] += len(valid_blocks)
        info["per_township"][code] = {
            "status": "ok", "n_blocks": len(valid_blocks),
            "n_parcels_assigned": int((gdf_sw["block_id"] >= 0).sum()),
        }

    return info


# =============================================================================
# Phase C: townships extraction + sanity
# =============================================================================
def _extract_townships(dltb_fc, xzq_fc, qsdwdm_field,
                       xzq_code_field, xzq_name_field, messages):
    """Build {9-digit-code: label} dict.

    Primary source: unique 9-digit prefixes of dltb_fc.<qsdwdm_field>.
    Optional: xzq_fc supplies Chinese labels via xzq_name_field looked up
    by xzq_code_field prefix. If xzq_fc is None or lookup fails, labels
    fall back to the code string itself.

    Filters out prefixes with < MIN_PARCELS_PER_TOWNSHIP parcels in DLTB,
    since those are usually border artifacts (e.g. 1-parcel entries).
    """
    import arcpy

    MIN_PARCELS_PER_TOWNSHIP = 50

    def _say(m, level="info"):
        if messages is not None:
            getattr(messages,
                    "addMessage" if level == "info" else "addWarningMessage")(m)
        print(m, flush=True)

    # Validate DLTB field
    if qsdwdm_field not in {f.name for f in arcpy.ListFields(dltb_fc)}:
        raise RuntimeError(f"DLTB missing required field: {qsdwdm_field}")

    # Count parcels per 9-digit prefix
    from collections import Counter
    counts = Counter()
    with arcpy.da.SearchCursor(dltb_fc, [qsdwdm_field]) as cur:
        for (raw,) in cur:
            if raw is None:
                continue
            code = str(raw).strip()
            if len(code) >= 9:
                counts[code[:9]] += 1

    # Build code -> label (initially label = code)
    codes = sorted(p for p, n in counts.items() if n >= MIN_PARCELS_PER_TOWNSHIP)
    dropped = [(p, n) for p, n in counts.items() if n < MIN_PARCELS_PER_TOWNSHIP]
    if dropped:
        _say(f"  Dropped {len(dropped)} townships with <{MIN_PARCELS_PER_TOWNSHIP} "
             f"parcels: {dropped[:5]}...", level="warn")
    if not codes:
        raise RuntimeError(
            f"No townships with >= {MIN_PARCELS_PER_TOWNSHIP} parcels found in "
            f"DLTB.{qsdwdm_field}."
        )

    prefix_to_label = {c: c for c in codes}

    # Try XZQ for friendly labels
    if xzq_fc:
        xzq_fields = {f.name for f in arcpy.ListFields(xzq_fc)}
        if xzq_code_field not in xzq_fields:
            _say(f"  XZQ missing code field '{xzq_code_field}' (have "
                 f"{sorted(xzq_fields)}); using raw codes as labels.",
                 level="warn")
        else:
            name_col = xzq_name_field if xzq_name_field in xzq_fields else None
            read_fields = [xzq_code_field] + ([name_col] if name_col else [])
            hits = 0
            with arcpy.da.SearchCursor(xzq_fc, read_fields) as cur:
                for row in cur:
                    raw = row[0]
                    if raw is None:
                        continue
                    c = str(raw).strip()
                    if len(c) < 9:
                        continue
                    prefix = c[:9]
                    if prefix not in prefix_to_label:
                        continue
                    label = row[1] if len(row) > 1 and row[1] else None
                    if label:
                        label = str(label).strip()
                        # Prefer shortest (== rougher / lower-admin level) label
                        if (prefix_to_label[prefix] == prefix or
                                len(label) < len(prefix_to_label[prefix])):
                            prefix_to_label[prefix] = label
                            hits += 1
            _say(f"  XZQ label lookup: resolved {hits} label assignments")

    return dict(sorted(prefix_to_label.items()))


def _labels_from_reference_layer(dltb_fc, qsdwdm_field, reference_layer,
                                 reference_name_field, proj_crs, messages,
                                 dltb_gpkg_layer=None):
    """Spatial-join DLTB centroids into a reference township polygon layer
    (e.g. national 1:5M xiangzhen.shp) to map each 9-digit QSDWDM prefix to
    the dominant Chinese name found in reference_name_field.

    Returns {prefix9: chinese_name_or_None}. Prefixes with no hits map to None.

    Parameters
    ----------
    dltb_fc : str
        Original DLTB feature class. May be a shapefile, gpkg layer, or
        file-gdb feature class. We try geopandas first; if that fails
        (e.g. file gdb is unreadable by pyogrio), we fall back to arcpy.
    dltb_gpkg_layer : str or None
        Optional path to the gpkg already produced by Phase A
        (DLTB_with_slope.gpkg). When provided, we read from it instead of
        the original dltb_fc -- this avoids pyogrio's file-gdb limitation.

    Notes
    -----
    - DLTB is reprojected to proj_crs before taking centroids (geographic
      CRS centroids are incorrect).
    - At most MAX_SAMPLE_PER_PREFIX parcels are sampled per prefix.
      Boundary precision of reference layers (esp. 1:5M) is lower than
      DLTB, so a handful of samples with majority vote is enough.
    """
    import geopandas as gpd

    def _say(m, level="info"):
        if messages is not None:
            getattr(messages, "addMessage" if level == "info"
                    else "addWarningMessage")(m)
        print(m, flush=True)

    MAX_SAMPLE_PER_PREFIX = 5

    ref = gpd.read_file(str(reference_layer))
    if reference_name_field not in ref.columns:
        raise RuntimeError(
            f"Reference layer missing column '{reference_name_field}'. "
            f"Available: {list(ref.columns)}"
        )
    ref = ref[[reference_name_field, "geometry"]].rename(
        columns={reference_name_field: "_ref_name"}
    )
    _say(f"    reference layer: {len(ref)} polygons, crs={ref.crs}")

    # Prefer reading from the Phase A gpkg (standard format, pyogrio-friendly).
    src = dltb_gpkg_layer if dltb_gpkg_layer else dltb_fc
    try:
        dltb = gpd.read_file(str(src), columns=[qsdwdm_field])
    except Exception as e:
        _say(f"    geopandas read failed ({e}); falling back to arcpy", level="warn")
        dltb = _arcpy_read_qsdwdm(dltb_fc, qsdwdm_field)
    _say(f"    dltb: {len(dltb)} parcels, crs={dltb.crs}")

    dltb["_prefix9"] = dltb[qsdwdm_field].astype(str).str[:9]
    dltb = dltb[dltb["_prefix9"].str.len() >= 9]

    sampled = (dltb.groupby("_prefix9", group_keys=False)
                   .apply(lambda g: g.sample(min(len(g), MAX_SAMPLE_PER_PREFIX),
                                             random_state=0)))
    _say(f"    sampled {len(sampled)} parcels across {sampled['_prefix9'].nunique()} prefixes")

    sampled_proj = sampled.to_crs(proj_crs)
    centroids = sampled_proj.copy()
    centroids.geometry = sampled_proj.geometry.centroid
    centroids = centroids.to_crs(ref.crs)

    joined = gpd.sjoin(
        centroids[["_prefix9", "geometry"]],
        ref[["_ref_name", "geometry"]],
        how="left", predicate="within",
    )
    n_unmatched = joined["_ref_name"].isna().sum()
    if n_unmatched:
        _say(f"    {n_unmatched}/{len(joined)} parcel centroids fell outside "
             "reference polygons", level="warn")

    def _dominant(series):
        s = series.dropna()
        if len(s) == 0:
            return None
        return s.mode().iloc[0]

    label_map = (joined.groupby("_prefix9")["_ref_name"]
                       .apply(_dominant)
                       .to_dict())
    resolved = sum(1 for v in label_map.values() if v)
    _say(f"    resolved {resolved}/{len(label_map)} prefixes to Chinese names")
    return label_map


def _arcpy_read_qsdwdm(dltb_fc, qsdwdm_field):
    """Fallback: read QSDWDM + geometry via arcpy and build a GeoDataFrame.

    Used when the DLTB lives in a file gdb that pyogrio cannot open. Slow
    on large datasets, so we only use it when the gpkg path isn't available.
    """
    import arcpy
    import geopandas as gpd
    from shapely import wkt as shwkt

    sr = arcpy.Describe(dltb_fc).spatialReference
    rows = []
    geoms = []
    with arcpy.da.SearchCursor(dltb_fc, [qsdwdm_field, "SHAPE@WKT"]) as cur:
        for q, w in cur:
            rows.append({qsdwdm_field: q})
            geoms.append(shwkt.loads(w) if w else None)
    crs = sr.factoryCode
    gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs=f"EPSG:{crs}")
    return gdf


def _phase_c_sanity(prepared_dir, proj_crs, messages):
    """Verify core.blocks_env.make_env(prepared_dir) succeeds.

    This is a lightweight sanity check -- we don't actually run MPC,
    just confirm the data contract is met.
    """
    def _say(m, level="info"):
        if messages is not None:
            getattr(messages,
                    "addMessage" if level == "info" else "addWarningMessage")(m)
        print(m, flush=True)

    try:
        from core.blocks_env import make_env
    except ImportError:
        # Try adding toolbox dir
        toolbox_dir = str(Path(__file__).resolve().parent.parent)
        if toolbox_dir not in sys.path:
            sys.path.insert(0, toolbox_dir)
        from core.blocks_env import make_env

    try:
        env = make_env(prepared_dir=str(prepared_dir), proj_crs=proj_crs)
    except Exception as e:
        _say(f"  [Phase C2] make_env FAILED: {e}", level="warn")
        return {"status": "failed", "error": str(e)}

    result = {
        "status": "ok",
        "n_blocks": int(env.n_blocks),
        "n_parcels": int(env.n_parcels),
        "initial_slope": float(env.avg_farmland_slope),
        "initial_contiguity": float(env.contiguity),
        "baimu_count": int(env.baimu_count),
    }
    _say(f"  [Phase C2] OK. n_blocks={result['n_blocks']} "
         f"n_parcels={result['n_parcels']} "
         f"slope={result['initial_slope']:.4f} "
         f"baimu={result['baimu_count']}")
    return result
