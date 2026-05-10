# -*- coding: utf-8 -*-
"""ArcGIS Pro Python Toolbox: Contrastive World-Model + MPC
Farmland Consolidation Planner.

Four-tool pipeline for arbitrary user regions:

    1. Prepare Data & Blocks
         DEM raster + DLTB.shp (+ optional XZQ.shp)
         -> DLTB_with_slope.shp  +  block_compositions/features
    2. Sample Transitions & Pairwise
         prepared_dir -> transitions.npz + pairwise.npz
    3. Train Contrastive Ensemble
         npz bundles -> ensemble_memberN.onnx (three members)
    4. MPC Planning
         prepared_dir + ensemble.onnx -> optimized_dltb.shp

Plus a CheckDependencies utility.

Stages 1-3 are one-time setup per region. Stage 4 is the planning loop
users will re-run.

Architecture:
    - This .pyt is a thin wrapper: parameter UI + arcpy progress only.
    - Algorithm bodies live in core/*.py.
    - Model weights are ONNX under <prepared_dir>/tool3/, never .pt.
"""

import os
import sys
import traceback


TOOLBOX_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(TOOLBOX_DIR, "core")


def _ensure_sys_path():
    for p in (CORE_DIR, TOOLBOX_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)


class Toolbox(object):
    def __init__(self):
        self.label = "Farmland Consolidation Planner"
        self.alias = "FarmlandConsolidation"
        self.description = (
            "Contrastive world-model + MPC farmland consolidation planner. "
            "Four-tool pipeline: prepare data -> sample transitions -> "
            "train ensemble -> MPC plan. Stages 1-3 are one-time setup "
            "per region; stage 4 is the planning loop."
        )
        self.tools = [
            PrepareDataTool,
            SampleTransitionsTool,
            TrainEnsembleTool,
            MPCPlanTool,
            CheckDependenciesTool,
        ]


# ======================================================================
# Tool 1: Prepare Data & Blocks
# ======================================================================
class PrepareDataTool(object):
    """Tool 1: DEM + DLTB (+optional XZQ) -> DLTB_with_slope.gpkg + blocks/.

    Produces the prepared_dir layout that Tool 4 consumes:
        <out_dir>/dem_slope_analysis/output/DLTB_with_slope.gpkg
        <out_dir>/results_real/blocks/township_<code>/...
        <out_dir>/townships.json
        <out_dir>/prepare_data_summary.json

    Block definition uses Paper 3 hybrid (DLTB barriers + AgglomerativeClustering
    within each township).

    DEM modes:
      - User-supplied raster: arcpy.sa.Slope + ZonalStatisticsAsTable (fast,
        requires Spatial Analyst extension)
      - Auto-download Copernicus GLO-30: NOT IMPLEMENTED in v1; use the
        vendored dem_slope_zonal.py for that path.

    Runtime estimate (county scale, 50k parcels): 30-60 minutes.
    """

    def __init__(self):
        self.label = "1. Prepare Data & Blocks"
        self.description = self.__doc__
        self.canRunInBackground = True
        self.category = "Pipeline"

    def getParameterInfo(self):
        import arcpy
        p_dltb = arcpy.Parameter(
            displayName="DLTB Feature Class (Third Survey land-use polygons)",
            name="dltb", datatype="DEFeatureClass",
            parameterType="Required", direction="Input")
        p_dltb.filter.list = ["Polygon"]

        p_xzq = arcpy.Parameter(
            displayName="XZQ Feature Class (optional, only used for Chinese township labels)",
            name="xzq", datatype="DEFeatureClass",
            parameterType="Optional", direction="Input")
        p_xzq.filter.list = ["Polygon"]

        p_dem_mode = arcpy.Parameter(
            displayName="DEM Source",
            name="dem_mode", datatype="GPString",
            parameterType="Required", direction="Input")
        p_dem_mode.filter.type = "ValueList"
        p_dem_mode.filter.list = ["User-supplied raster",
                                   "Auto-download Copernicus GLO-30 (NOT IMPLEMENTED)"]
        p_dem_mode.value = "User-supplied raster"

        p_dem = arcpy.Parameter(
            displayName="DEM Raster (required when DEM Source = user-supplied)",
            name="dem_raster", datatype="DERasterDataset",
            parameterType="Optional", direction="Input")

        p_dlbm = arcpy.Parameter(
            displayName="DLBM field in DLTB (3-digit land-use code, default 'DLBM')",
            name="dlbm_field", datatype="Field",
            parameterType="Optional", direction="Input")
        p_dlbm.parameterDependencies = [p_dltb.name]
        p_dlbm.filter.list = ["Text"]
        p_dlbm.value = "DLBM"

        p_qs = arcpy.Parameter(
            displayName="QSDWDM field in DLTB (9+ digit admin code, default 'QSDWDM')",
            name="qsdwdm_field", datatype="Field",
            parameterType="Optional", direction="Input")
        p_qs.parameterDependencies = [p_dltb.name]
        p_qs.filter.list = ["Text"]
        p_qs.value = "QSDWDM"

        p_xzq_code = arcpy.Parameter(
            displayName="XZQ code field (default 'XZQDM', ignored if XZQ blank)",
            name="xzq_code_field", datatype="Field",
            parameterType="Optional", direction="Input")
        p_xzq_code.parameterDependencies = [p_xzq.name]
        p_xzq_code.filter.list = ["Text"]
        p_xzq_code.value = "XZQDM"

        p_xzq_name = arcpy.Parameter(
            displayName="XZQ name field (default 'XZQMC', ignored if XZQ blank)",
            name="xzq_name_field", datatype="Field",
            parameterType="Optional", direction="Input")
        p_xzq_name.parameterDependencies = [p_xzq.name]
        p_xzq_name.filter.list = ["Text"]
        p_xzq_name.value = "XZQMC"

        p_ref_layer = arcpy.Parameter(
            displayName="Reference Township Layer (optional, e.g. national xiangzhen.shp; "
                        "used only when XZQ unavailable to inject Chinese labels)",
            name="reference_layer", datatype="DEFeatureClass",
            parameterType="Optional", direction="Input")
        p_ref_layer.filter.list = ["Polygon"]

        p_ref_name = arcpy.Parameter(
            displayName="Reference layer Chinese-name field (default '乡')",
            name="reference_name_field", datatype="GPString",
            parameterType="Optional", direction="Input")
        p_ref_name.value = "乡"

        p_proj_crs = arcpy.Parameter(
            displayName="Projected CRS (blank = EPSG:32648 UTM 48N, valid for "
                        "central-west China; override for other zones)",
            name="proj_crs", datatype="GPString",
            parameterType="Optional", direction="Input")
        p_proj_crs.value = "EPSG:32648"

        p_min_parcels = arcpy.Parameter(
            displayName="Block min parcels (Paper 3 default 3)",
            name="min_parcels", datatype="GPLong",
            parameterType="Optional", direction="Input")
        p_min_parcels.value = 3
        p_min_parcels.filter.type = "Range"
        p_min_parcels.filter.list = [1, 50]

        p_min_area = arcpy.Parameter(
            displayName="Block min area in hectares (Paper 3 default 0.5)",
            name="min_area_ha", datatype="GPDouble",
            parameterType="Optional", direction="Input")
        p_min_area.value = 0.5

        p_max_parcels = arcpy.Parameter(
            displayName="Block max parcels (subdivide threshold, Paper 3 default 30)",
            name="max_parcels", datatype="GPLong",
            parameterType="Optional", direction="Input")
        p_max_parcels.value = 30
        p_max_parcels.filter.type = "Range"
        p_max_parcels.filter.list = [10, 200]

        p_outdir = arcpy.Parameter(
            displayName="Output Directory (the prepared_dir for Tool 4)",
            name="out_dir", datatype="DEFolder",
            parameterType="Required", direction="Output")

        return [p_dltb, p_xzq, p_dem_mode, p_dem,
                p_dlbm, p_qs, p_xzq_code, p_xzq_name,
                p_ref_layer, p_ref_name,
                p_proj_crs, p_min_parcels, p_min_area, p_max_parcels,
                p_outdir]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        # Enable/disable DEM raster based on mode
        mode_v = parameters[2].valueAsText or ""
        is_user_dem = mode_v.startswith("User-supplied")
        parameters[3].enabled = is_user_dem
        return

    def updateMessages(self, parameters):
        mode_v = parameters[2].valueAsText or ""
        if mode_v.startswith("User-supplied") and not parameters[3].value:
            parameters[3].setErrorMessage(
                "DEM Raster is required when DEM Source = user-supplied"
            )
        if "NOT IMPLEMENTED" in mode_v:
            parameters[2].setWarningMessage(
                "Auto-download branch is not yet implemented in v1. Use the "
                "user-supplied raster mode."
            )
        return

    def execute(self, parameters, messages):
        _ensure_sys_path()
        import arcpy
        try:
            from core.prepare_data import run as prep_run
        except ImportError as e:
            arcpy.AddError(f"Failed to import core.prepare_data: {e}")
            return

        dltb_fc       = parameters[0].valueAsText
        xzq_fc        = parameters[1].valueAsText  # may be None
        dem_mode_raw  = parameters[2].valueAsText or "User-supplied raster"
        dem_raster    = parameters[3].valueAsText
        dlbm_field    = parameters[4].valueAsText or "DLBM"
        qsdwdm_field  = parameters[5].valueAsText or "QSDWDM"
        xzq_code_field = parameters[6].valueAsText or "XZQDM"
        xzq_name_field = parameters[7].valueAsText or "XZQMC"
        reference_layer = parameters[8].valueAsText  # may be None
        reference_name_field = parameters[9].valueAsText or "乡"
        proj_crs      = parameters[10].valueAsText or "EPSG:32648"
        min_parcels   = int(parameters[11].value or 3)
        min_area_ha   = float(parameters[12].value or 0.5)
        max_parcels   = int(parameters[13].value or 30)
        out_dir       = parameters[14].valueAsText

        dem_mode = "user" if dem_mode_raw.startswith("User-supplied") else "auto"

        try:
            prep_run(
                dltb_fc=dltb_fc, xzq_fc=xzq_fc, prepared_dir=out_dir,
                dem_mode=dem_mode, dem_raster=dem_raster,
                dlbm_field=dlbm_field, qsdwdm_field=qsdwdm_field,
                xzq_code_field=xzq_code_field, xzq_name_field=xzq_name_field,
                reference_layer=reference_layer,
                reference_name_field=reference_name_field,
                proj_crs=proj_crs,
                min_parcels=min_parcels, min_area_ha=min_area_ha,
                max_parcels=max_parcels,
                messages=messages,
            )
        except Exception as e:
            arcpy.AddError(f"Tool 1 failed: {e}")
            arcpy.AddError(traceback.format_exc())
        return


# ======================================================================
# Tool 2: Sample Transitions & Pairwise
# ======================================================================
class SampleTransitionsTool(object):
    """Sample random-policy transitions + pairwise ranking data from an env
    built on Tool 1's prepared_dir. Output consumed by Tool 3 training.

    Produces <prepared_dir>/tool2/:
        transitions.npz  (for contrastive trainer's MSE loss; 6 keys)
        pairwise.npz     (for ranking loss; state -> N_actions rewards)
        sample_transitions_summary.json
        sample_transitions.log

    Runtime (county scale, 2600 blocks):
        ~10-20 min for default 60 episodes + 1000 states x 50 actions.
    Small region (30 blocks): ~1 min.
    """

    def __init__(self):
        self.label = "2. Sample Transitions & Pairwise"
        self.description = self.__doc__
        self.canRunInBackground = True
        self.category = "Pipeline"

    def getParameterInfo(self):
        import arcpy

        p_prepared = arcpy.Parameter(
            displayName="Prepared Data Directory (output from Tool 1)",
            name="prepared_dir", datatype="DEFolder",
            parameterType="Required", direction="Input")

        p_n_eps = arcpy.Parameter(
            displayName="Number of transition episodes (Paper 9 default 60)",
            name="n_transition_episodes", datatype="GPLong",
            parameterType="Optional", direction="Input")
        p_n_eps.value = 60
        p_n_eps.filter.type = "Range"
        p_n_eps.filter.list = [1, 500]

        p_n_pw = arcpy.Parameter(
            displayName="Number of pairwise states (Paper 9 default 1000)",
            name="n_pairwise_states", datatype="GPLong",
            parameterType="Optional", direction="Input")
        p_n_pw.value = 1000
        p_n_pw.filter.type = "Range"
        p_n_pw.filter.list = [10, 10000]

        p_n_act = arcpy.Parameter(
            displayName="Actions per pairwise state (Paper 9 default 50)",
            name="n_pairwise_actions", datatype="GPLong",
            parameterType="Optional", direction="Input")
        p_n_act.value = 50
        p_n_act.filter.type = "Range"
        p_n_act.filter.list = [2, 200]

        p_seed = arcpy.Parameter(
            displayName="Random seed",
            name="seed", datatype="GPLong",
            parameterType="Optional", direction="Input")
        p_seed.value = 0

        p_proj_crs = arcpy.Parameter(
            displayName="Projected CRS (blank = auto, should match Tool 1)",
            name="proj_crs", datatype="GPString",
            parameterType="Optional", direction="Input")

        return [p_prepared, p_n_eps, p_n_pw, p_n_act, p_seed, p_proj_crs]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, messages):
        _ensure_sys_path()
        import arcpy
        try:
            from core.sample_transitions import run as sample_run
        except ImportError as e:
            arcpy.AddError(f"Failed to import core.sample_transitions: {e}")
            return

        prepared_dir  = parameters[0].valueAsText
        n_eps         = int(parameters[1].value or 60)
        n_pw_states   = int(parameters[2].value or 1000)
        n_pw_actions  = int(parameters[3].value or 50)
        seed          = int(parameters[4].value or 0)
        proj_crs      = parameters[5].valueAsText or None

        try:
            sample_run(
                prepared_dir=prepared_dir,
                n_transition_episodes=n_eps,
                n_pairwise_states=n_pw_states,
                n_pairwise_actions=n_pw_actions,
                seed=seed, proj_crs=proj_crs,
                messages=messages,
            )
        except Exception as e:
            arcpy.AddError(f"Tool 2 failed: {e}")
            arcpy.AddError(traceback.format_exc())
        return


# ======================================================================
# Tool 3: Train Contrastive Ensemble
# ======================================================================
class TrainEnsembleTool(object):
    """Train N TransitionModels with contrastive pairwise ranking loss
    (Paper 9 v6: lambda=5.0, margin=0.1, epochs=30, patience=8). Each
    best-val ckpt is torch.onnx.export'd; n_blocks is statically baked
    into the ONNX graph.

    Reads: <prepared_dir>/tool2/{transitions,pairwise}.npz
    Writes: <prepared_dir>/tool3/ensemble_memberN.onnx + train_summary.json
    """

    def __init__(self):
        self.label = "3. Train Contrastive Ensemble"
        self.description = self.__doc__
        self.canRunInBackground = True
        self.category = "Pipeline"

    def getParameterInfo(self):
        import arcpy
        p_prepared = arcpy.Parameter(
            displayName="Prepared Data Directory (output of Tool 1, must also "
                        "contain tool2/ from Tool 2)",
            name="prepared_dir", datatype="DEFolder",
            parameterType="Required", direction="Input")

        p_n_members = arcpy.Parameter(
            displayName="Ensemble size (Paper 9 v6 default 3)",
            name="n_members", datatype="GPLong",
            parameterType="Optional", direction="Input")
        p_n_members.value = 3
        p_n_members.filter.type = "Range"
        p_n_members.filter.list = [1, 10]

        p_epochs = arcpy.Parameter(
            displayName="Epochs per member (Paper 9 v6 default 30)",
            name="epochs", datatype="GPLong",
            parameterType="Optional", direction="Input")
        p_epochs.value = 30
        p_epochs.filter.type = "Range"
        p_epochs.filter.list = [5, 500]

        p_patience = arcpy.Parameter(
            displayName="Early-stop patience (Paper 9 v6 default 8, 0 = off)",
            name="patience", datatype="GPLong",
            parameterType="Optional", direction="Input")
        p_patience.value = 8

        p_lambda = arcpy.Parameter(
            displayName="Contrastive lambda_rank (Paper 9 v6 default 5.0)",
            name="lambda_rank", datatype="GPDouble",
            parameterType="Optional", direction="Input")
        p_lambda.value = 5.0

        p_margin = arcpy.Parameter(
            displayName="Ranking margin (Paper 9 v6 default 0.1)",
            name="margin", datatype="GPDouble",
            parameterType="Optional", direction="Input")
        p_margin.value = 0.1

        p_batch = arcpy.Parameter(
            displayName="Batch size (Paper 9 v6 default 256)",
            name="batch_size", datatype="GPLong",
            parameterType="Optional", direction="Input")
        p_batch.value = 256

        p_seed = arcpy.Parameter(
            displayName="Seed base (member i uses seed_base + i*1000)",
            name="seed_base", datatype="GPLong",
            parameterType="Optional", direction="Input")
        p_seed.value = 0

        p_threads = arcpy.Parameter(
            displayName="torch_threads (0 = default; 12 observed best alongside ArcGIS)",
            name="torch_threads", datatype="GPLong",
            parameterType="Optional", direction="Input")
        p_threads.value = 0

        return [p_prepared, p_n_members, p_epochs, p_patience,
                p_lambda, p_margin, p_batch, p_seed, p_threads]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        return

    def execute(self, parameters, messages):
        _ensure_sys_path()
        import arcpy
        try:
            from core.train_ensemble import run as train_run
        except ImportError as e:
            arcpy.AddError(f"Failed to import core.train_ensemble: {e}")
            return

        prepared_dir = parameters[0].valueAsText
        n_members    = int(parameters[1].value or 3)
        epochs       = int(parameters[2].value or 30)
        patience     = int(parameters[3].value or 8)
        lambda_rank  = float(parameters[4].value or 5.0)
        margin       = float(parameters[5].value or 0.1)
        batch_size   = int(parameters[6].value or 256)
        seed_base    = int(parameters[7].value or 0)
        torch_threads = int(parameters[8].value or 0)

        try:
            train_run(
                prepared_dir=prepared_dir,
                n_members=n_members, epochs=epochs, patience=patience,
                lambda_rank=lambda_rank, margin=margin,
                batch_size=batch_size, seed_base=seed_base,
                torch_threads=torch_threads,
                messages=messages,
            )
        except Exception as e:
            arcpy.AddError(f"Tool 3 failed: {e}")
            arcpy.AddError(traceback.format_exc())
        return


# ======================================================================
# Tool 4: MPC Planning
# ======================================================================
class MPCPlanTool(object):
    """Model-Predictive planning: at each env step, roll out top-K
    candidate blocks under the ensemble for H steps, pick the block
    whose rollout accumulates the highest reward, execute one real step.

    Reference config: H=5, K=50, gamma=0.99, scoring=reward.

    Features:
        - Accepts a Prepared Data Directory (output of Tool 1) via
          "Prepared Data Directory" parameter.
        - Optionally writes optimized DLTB feature class with
          OPT_DLBM / OPT_DLMC / CHG_FLAG / ORIG_DLBM fields (via BSM).
        - Runtime (county scale, ~50k parcels): ~30-70s env build +
          ~7 min / episode.
    """

    def __init__(self):
        self.label = "4. MPC Planning"
        self.description = self.__doc__
        self.canRunInBackground = True
        self.category = "Planning"

    def getParameterInfo(self):
        import arcpy

        p_prepared = arcpy.Parameter(
            displayName="Prepared Data Directory (from Tool 1)",
            name="prepared_dir", datatype="DEFolder",
            parameterType="Required", direction="Input")

        p_ensemble_dir = arcpy.Parameter(
            displayName="Ensemble Directory (from Tool 3)",
            name="ensemble_dir", datatype="DEFolder",
            parameterType="Required", direction="Input")

        p_out = arcpy.Parameter(
            displayName="Output Directory (for land_use.npy + summary.json + log)",
            name="out_dir", datatype="DEFolder",
            parameterType="Required", direction="Output")

        p_horizon = arcpy.Parameter(
            displayName="MPC horizon H",
            name="horizon", datatype="GPLong",
            parameterType="Optional", direction="Input")
        p_horizon.value = 5
        p_horizon.filter.type = "Range"
        p_horizon.filter.list = [1, 25]

        p_topk = arcpy.Parameter(
            displayName="Top-K candidates per step",
            name="top_k", datatype="GPLong",
            parameterType="Optional", direction="Input")
        p_topk.value = 50
        p_topk.filter.type = "Range"
        p_topk.filter.list = [5, 200]

        p_gamma = arcpy.Parameter(
            displayName="Discount gamma",
            name="gamma", datatype="GPDouble",
            parameterType="Optional", direction="Input")
        p_gamma.value = 0.99

        p_continuation = arcpy.Parameter(
            displayName="Rollout continuation policy",
            name="continuation", datatype="GPString",
            parameterType="Optional", direction="Input")
        p_continuation.filter.type = "ValueList"
        p_continuation.filter.list = ["random", "greedy"]
        p_continuation.value = "random"

        p_scoring = arcpy.Parameter(
            displayName="1-step scoring signal",
            name="scoring", datatype="GPString",
            parameterType="Optional", direction="Input")
        p_scoring.filter.type = "ValueList"
        p_scoring.filter.list = ["reward", "slope"]
        p_scoring.value = "reward"

        p_episodes = arcpy.Parameter(
            displayName="Number of MPC episodes",
            name="n_episodes", datatype="GPLong",
            parameterType="Optional", direction="Input")
        p_episodes.value = 1
        p_episodes.filter.type = "Range"
        p_episodes.filter.list = [1, 20]

        p_max_steps = arcpy.Parameter(
            displayName="Max steps per episode (0 = env default = 100)",
            name="max_steps", datatype="GPLong",
            parameterType="Optional", direction="Input")
        p_max_steps.value = 0

        p_threads = arcpy.Parameter(
            displayName="ONNX Runtime threads (0 = auto)",
            name="threads", datatype="GPLong",
            parameterType="Optional", direction="Input")
        p_threads.value = 0

        p_proj_crs = arcpy.Parameter(
            displayName="Projected CRS for area calc (blank = EPSG:32648 UTM 48N)",
            name="proj_crs", datatype="GPString",
            parameterType="Optional", direction="Input")

        p_input_dltb = arcpy.Parameter(
            displayName="Input DLTB Feature Class (required if writing output)",
            name="input_dltb", datatype="DEFeatureClass",
            parameterType="Optional", direction="Input")
        p_input_dltb.filter.list = ["Polygon"]

        p_output_fc = arcpy.Parameter(
            displayName="Output Optimized DLTB (optional; blank = skip)",
            name="output_fc", datatype="DEFeatureClass",
            parameterType="Optional", direction="Output")

        p_farm_dlbm = arcpy.Parameter(
            displayName="Representative Farmland DLBM (for forest->farm swaps)",
            name="farm_dlbm", datatype="GPString",
            parameterType="Optional", direction="Input")
        p_farm_dlbm.value = "011"

        p_forest_dlbm = arcpy.Parameter(
            displayName="Representative Forest DLBM (for farm->forest swaps)",
            name="forest_dlbm", datatype="GPString",
            parameterType="Optional", direction="Input")
        p_forest_dlbm.value = "031"

        # --- v0.3: reward-weight overrides ---
        # These modify CountyLevelEnv's per-step reward formula:
        #   r = slope_weight*slope_delta + cont_weight*cont_delta
        #       + baimu_weight*baimu_area_delta + baimu_bonus*baimu_new_count
        # Tool 3 ensemble was trained with the Paper 9 v6 defaults; changing
        # weights here is a quick lever (no retrain) but strictly speaking
        # mis-calibrates Tool 3's reward head. updateMessages warns.
        p_slope_weight = arcpy.Parameter(
            displayName="slope_weight (blank = Paper 9 default 4000.0)",
            name="slope_weight", datatype="GPDouble",
            parameterType="Optional", direction="Input")
        p_cont_weight = arcpy.Parameter(
            displayName="cont_weight (blank = Paper 9 default 500.0)",
            name="cont_weight", datatype="GPDouble",
            parameterType="Optional", direction="Input")
        p_baimu_weight = arcpy.Parameter(
            displayName="baimu_weight (blank = Paper 9 default 1500.0)",
            name="baimu_weight", datatype="GPDouble",
            parameterType="Optional", direction="Input")
        p_baimu_bonus = arcpy.Parameter(
            displayName="baimu_bonus (blank = Paper 9 default 5.0)",
            name="baimu_bonus", datatype="GPDouble",
            parameterType="Optional", direction="Input")

        return [p_prepared, p_ensemble_dir, p_out, p_horizon, p_topk,
                p_gamma, p_continuation, p_scoring, p_episodes, p_max_steps,
                p_threads, p_proj_crs, p_input_dltb, p_output_fc,
                p_farm_dlbm, p_forest_dlbm,
                p_slope_weight, p_cont_weight, p_baimu_weight, p_baimu_bonus]

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        return

    def updateMessages(self, parameters):
        # If output_fc is set, input_dltb becomes required
        p_input_dltb = parameters[12]
        p_output_fc  = parameters[13]
        if p_output_fc.value and not p_input_dltb.value:
            p_input_dltb.setErrorMessage(
                "Input DLTB is required when Output Optimized DLTB is set."
            )
        # Warn if any reward weight is overridden
        reward_params = [parameters[16], parameters[17],
                         parameters[18], parameters[19]]
        if any(p.value is not None for p in reward_params):
            reward_params[0].setWarningMessage(
                "Overriding reward weights. IMPORTANT: Tool 3's ensemble "
                "predicts reward under its TRAINING weights; the override "
                "only changes env.step()'s reported reward, not what MPC "
                "uses to rank candidates (when scoring='reward'). To make "
                "the override actually steer planning, retrain Tool 2+3 "
                "with these weights first."
            )
        return

    def execute(self, parameters, messages):
        _ensure_sys_path()
        import arcpy
        try:
            from core.mpc_plan import run as mpc_run
        except ImportError as e:
            arcpy.AddError(f"Failed to import core.mpc_plan: {e}")
            return

        prepared_dir = parameters[0].valueAsText
        ensemble_dir = parameters[1].valueAsText
        out_dir      = parameters[2].valueAsText
        horizon      = int(parameters[3].value or 5)
        top_k        = int(parameters[4].value or 50)
        gamma        = float(parameters[5].value or 0.99)
        continuation = parameters[6].valueAsText or "random"
        scoring      = parameters[7].valueAsText or "reward"
        n_episodes   = int(parameters[8].value or 1)
        max_steps_v  = int(parameters[9].value or 0)
        max_steps    = None if max_steps_v == 0 else max_steps_v
        threads      = int(parameters[10].value or 0)
        proj_crs     = parameters[11].valueAsText or None
        input_dltb   = parameters[12].valueAsText or None
        output_fc    = parameters[13].valueAsText or None
        farm_dlbm    = parameters[14].valueAsText or "011"
        forest_dlbm  = parameters[15].valueAsText or "031"
        # None if blank = use Paper 9 v6 default in CountyLevelEnv.__init__
        slope_weight = parameters[16].value
        cont_weight  = parameters[17].value
        baimu_weight = parameters[18].value
        baimu_bonus  = parameters[19].value

        try:
            mpc_run(
                ensemble_dir=ensemble_dir, out_dir=out_dir,
                horizon=horizon, top_k=top_k, gamma=gamma,
                threads=threads, n_episodes=n_episodes,
                continuation=continuation, scoring=scoring,
                max_steps=max_steps, seed_offset=0,
                prepared_dir=prepared_dir, proj_crs=proj_crs,
                output_fc=output_fc, input_dltb_fc=input_dltb,
                farm_dlbm=farm_dlbm, forest_dlbm=forest_dlbm,
                slope_weight=slope_weight, cont_weight=cont_weight,
                baimu_weight=baimu_weight, baimu_bonus=baimu_bonus,
                messages=messages,
            )
        except Exception as e:
            arcpy.AddError(f"MPC run failed: {e}")
            arcpy.AddError(traceback.format_exc())


# ======================================================================
# Tool 5: Check Dependencies
# ======================================================================
class CheckDependenciesTool(object):
    """Verify arcgispro-py3 has everything needed for the four-tool pipeline.

    Checks (in order):
      - Python + arcpy version
      - Core python deps: numpy, torch, onnx, onnxruntime, geopandas
      - Extra deps: gymnasium, libpysal (not in vanilla arcgispro-py3)
      - ArcGIS licenses: core product + Spatial Analyst (required by Tool 1)

    Errors stop the pipeline; warnings are informational.
    """

    def __init__(self):
        self.label = "5. Check Dependencies"
        self.description = self.__doc__
        self.canRunInBackground = False
        self.category = "Utilities"

    def getParameterInfo(self):
        return []

    def isLicensed(self):
        return True

    def execute(self, parameters, messages):
        import arcpy

        messages.addMessage("=== Paper 9 Toolbox Dependency Check ===")
        messages.addMessage(f"Python: {sys.version.splitlines()[0]}")
        try:
            messages.addMessage(f"arcpy:  {arcpy.GetInstallInfo()['Version']} "
                                f"({arcpy.ProductInfo()})")
        except Exception as e:
            messages.addErrorMessage(f"arcpy: failed to introspect ({e})")

        errors = []
        warnings = []
        pip_install = []

        def _check(name, import_expr, required=True, install_hint=None):
            try:
                mod = __import__(import_expr)
                ver = getattr(mod, "__version__", "?")
                messages.addMessage(f"  [OK] {name}: {ver}")
                return True
            except ImportError:
                tag = "ERROR" if required else "WARN"
                msg = f"  [{tag}] {name}: not installed"
                if required:
                    messages.addErrorMessage(msg)
                    errors.append(name)
                else:
                    messages.addWarningMessage(msg)
                    warnings.append(name)
                if install_hint:
                    pip_install.append(install_hint)
                return False

        # --- Core deps (arcgispro-py3 default) ---
        messages.addMessage("\n-- Core Python dependencies --")
        _check("numpy",        "numpy")
        _check("torch",        "torch",
               install_hint="pip install torch --index-url "
                            "https://download.pytorch.org/whl/cpu")
        _check("onnx",         "onnx",
               install_hint="pip install onnx")
        _check("onnxruntime",  "onnxruntime",
               install_hint="pip install onnxruntime")
        _check("geopandas",    "geopandas",
               install_hint="pip install geopandas")

        # --- Paper 9 extras (not in vanilla arcgispro-py3) ---
        messages.addMessage("\n-- Paper 9 additional dependencies --")
        _check("gymnasium",    "gymnasium",
               install_hint="pip install gymnasium")
        _check("libpysal",     "libpysal",
               install_hint="pip install libpysal")

        # --- ArcGIS licenses ---
        messages.addMessage("\n-- ArcGIS licenses --")
        try:
            sa = arcpy.CheckExtension("Spatial")
            if sa == "Available":
                messages.addMessage("  [OK] Spatial Analyst: Available "
                                    "(required by Tool 1 for slope)")
            else:
                messages.addErrorMessage(
                    f"  [ERROR] Spatial Analyst: {sa} "
                    "(Tool 1 will fail -- needed for arcpy.sa.Slope and "
                    "ZonalStatisticsAsTable)"
                )
                errors.append("Spatial Analyst license")
        except Exception as e:
            messages.addErrorMessage(f"  [ERROR] CheckExtension('Spatial') failed: {e}")
            errors.append("Spatial Analyst license")

        # --- Summary + install hints ---
        messages.addMessage("\n" + "=" * 40)
        if errors:
            messages.addErrorMessage(
                f"FAIL: {len(errors)} blocking issue(s): {', '.join(errors)}"
            )
            if pip_install:
                messages.addErrorMessage(
                    "To install missing Python packages, run from the "
                    "'Python Command Prompt' launched by ArcGIS Pro:"
                )
                for cmd in pip_install:
                    messages.addErrorMessage(f"    {cmd}")
        elif warnings:
            messages.addWarningMessage(
                f"All required dependencies satisfied. "
                f"{len(warnings)} optional item(s) missing: {', '.join(warnings)}"
            )
        else:
            messages.addMessage("All dependencies satisfied. Ready to run the pipeline.")
