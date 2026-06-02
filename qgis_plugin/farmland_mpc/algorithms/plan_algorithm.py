"""Tool 4: MPC planning — emits the optimised cadastral shapefile."""
from __future__ import annotations

from pathlib import Path

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFile,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterEnum,
    QgsProcessingParameterCrs,
    QgsProcessingParameterVectorDestination,
    QgsProcessingParameterString,
)

from ._common import run_cli


class PlanAlgorithm(QgsProcessingAlgorithm):
    """Wraps ``farmland-mpc plan``."""

    ENSEMBLE_DIR = "ENSEMBLE_DIR"
    PREPARED_DIR = "PREPARED_DIR"
    OUT_DIR = "OUT_DIR"
    OUTPUT_SHP = "OUTPUT_SHP"
    HORIZON = "HORIZON"
    TOP_K = "TOP_K"
    N_EPISODES = "N_EPISODES"
    CONTINUATION = "CONTINUATION"
    SCORING = "SCORING"
    THREADS = "THREADS"
    SEED_OFFSET = "SEED_OFFSET"
    CRS = "CRS"
    ENV_KIND = "ENV_KIND"
    FARM_DLBM = "FARM_DLBM"
    FOREST_DLBM = "FOREST_DLBM"
    CULTIVATED_AREA_FLOOR_DELTA_HA = "CULTIVATED_AREA_FLOOR_DELTA_HA"
    BAIMU_AREA_FLOOR_DELTA_HA = "BAIMU_AREA_FLOOR_DELTA_HA"
    GAMMA_CONN = "GAMMA_CONN"
    DELTA_CONN = "DELTA_CONN"

    CONTINUATIONS = ("greedy", "random")
    SCORINGS = ("reward", "slope_only")
    ENV_KINDS = ("county", "restoration")

    def name(self) -> str:
        return "plan"

    def displayName(self) -> str:  # noqa: N802
        return "4 — Plan (MPC planning → optimised cadastre)"

    def group(self) -> str:
        return "Pipeline (run in order)"

    def groupId(self) -> str:  # noqa: N802
        return "pipeline"

    def shortHelpString(self) -> str:  # noqa: N802
        return (
            "Phase D of the farmland-mpc pipeline. Runs the model-"
            "predictive-control planner on the trained ensemble, writes "
            "per-step traces (mpc_summary.json + mpc_run.log + "
            "mpc_land_use.npy) and an optimised DLTB shapefile carrying "
            "OPT_DLBM / CHG_FLAG / ORIG_DLBM audit fields. ~3-5 min per "
            "100-step episode on a 12-thread CPU.\n\n"
            "Equivalent CLI: farmland-mpc plan --ensemble-dir ... "
            "--prepared-dir ... --out-dir ... --horizon 5 --top-k 50 "
            "--continuation greedy"
        )

    def createInstance(self):  # noqa: N802
        return PlanAlgorithm()

    def initAlgorithm(self, config=None):  # noqa: N802
        self.addParameter(QgsProcessingParameterFile(
            self.ENSEMBLE_DIR,
            "Trained ensemble directory (contains *.onnx members)",
            behavior=QgsProcessingParameterFile.Folder,
        ))
        self.addParameter(QgsProcessingParameterFile(
            self.PREPARED_DIR,
            "prepared/ directory (output of stage 1)",
            behavior=QgsProcessingParameterFile.Folder,
        ))
        self.addParameter(QgsProcessingParameterFolderDestination(
            self.OUT_DIR,
            "Output directory (per-step traces + summary)",
        ))
        self.addParameter(QgsProcessingParameterVectorDestination(
            self.OUTPUT_SHP,
            "Optimised DLTB shapefile (auto-named under output dir if blank)",
            optional=True,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.HORIZON, "Planning horizon H",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=5, minValue=1,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.TOP_K, "Candidates K",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=50, minValue=1,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.N_EPISODES, "Evaluation episodes",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=1, minValue=1,
        ))
        self.addParameter(QgsProcessingParameterEnum(
            self.CONTINUATION, "Stage-2 continuation",
            options=list(self.CONTINUATIONS), defaultValue=0,  # greedy default
        ))
        self.addParameter(QgsProcessingParameterEnum(
            self.SCORING, "Scoring channel",
            options=list(self.SCORINGS), defaultValue=0,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.THREADS,
            "ONNX runtime threads (0 = auto)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=0, minValue=0,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.SEED_OFFSET, "Seed offset",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=0, minValue=0,
        ))
        self.addParameter(QgsProcessingParameterCrs(
            self.CRS,
            "Projected CRS for area metrics (blank = inherit from prepared/)",
            optional=True,
        ))
        self.addParameter(QgsProcessingParameterEnum(
            self.ENV_KIND, "Environment",
            options=list(self.ENV_KINDS), defaultValue=0,
        ))
        self.addParameter(QgsProcessingParameterString(
            self.FARM_DLBM, "Farm land-use code", defaultValue="011",
        ))
        self.addParameter(QgsProcessingParameterString(
            self.FOREST_DLBM, "Forest land-use code", defaultValue="031",
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.CULTIVATED_AREA_FLOOR_DELTA_HA,
            "Cultivated-area floor delta in ha (blank = disabled; 0 = no net loss)",
            type=QgsProcessingParameterNumber.Double,
            optional=True,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.BAIMU_AREA_FLOOR_DELTA_HA,
            "Baimu-fang area floor delta in ha (blank = disabled; 0 = no net loss)",
            type=QgsProcessingParameterNumber.Double,
            optional=True,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.GAMMA_CONN,
            "Forest-entry connectivity weight gamma",
            type=QgsProcessingParameterNumber.Double,
            optional=True,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.DELTA_CONN,
            "Farmland-retirement connectivity protection delta",
            type=QgsProcessingParameterNumber.Double,
            optional=True,
        ))

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        ensemble_dir = self.parameterAsFile(parameters, self.ENSEMBLE_DIR, context)
        prepared_dir = self.parameterAsFile(parameters, self.PREPARED_DIR, context)
        out_dir = self.parameterAsString(parameters, self.OUT_DIR, context)

        # Optional explicit output shapefile path
        output_shp_param = self.parameterAsOutputLayer(parameters, self.OUTPUT_SHP, context)
        if not output_shp_param:
            output_shp_param = str(Path(out_dir) / "optimized.shp")

        crs = self.parameterAsCrs(parameters, self.CRS, context)

        args = [
            "--ensemble-dir", ensemble_dir,
            "--prepared-dir", prepared_dir,
            "--out-dir", out_dir,
            "--horizon", str(self.parameterAsInt(parameters, self.HORIZON, context)),
            "--top-k", str(self.parameterAsInt(parameters, self.TOP_K, context)),
            "--n-episodes", str(self.parameterAsInt(parameters, self.N_EPISODES, context)),
            "--continuation",
            self.CONTINUATIONS[self.parameterAsEnum(parameters, self.CONTINUATION, context)],
            "--scoring",
            self.SCORINGS[self.parameterAsEnum(parameters, self.SCORING, context)],
            "--threads", str(self.parameterAsInt(parameters, self.THREADS, context)),
            "--seed-offset", str(self.parameterAsInt(parameters, self.SEED_OFFSET, context)),
            "--env",
            self.ENV_KINDS[self.parameterAsEnum(parameters, self.ENV_KIND, context)],
            "--farm-dlbm", self.parameterAsString(parameters, self.FARM_DLBM, context) or "011",
            "--forest-dlbm", self.parameterAsString(parameters, self.FOREST_DLBM, context) or "031",
            "--output-shp", str(output_shp_param),
            "--verbose",
        ]
        if crs.isValid() and crs.authid():
            args.extend(["--crs", crs.authid()])
        area_floor = parameters.get(self.CULTIVATED_AREA_FLOOR_DELTA_HA)
        if area_floor is not None:
            args.extend(["--cultivated-area-floor-delta-ha", str(area_floor)])
        baimu_floor = parameters.get(self.BAIMU_AREA_FLOOR_DELTA_HA)
        if baimu_floor is not None:
            args.extend(["--baimu-area-floor-delta-ha", str(baimu_floor)])
        gamma_conn = parameters.get(self.GAMMA_CONN)
        if gamma_conn is not None:
            args.extend(["--gamma-conn", str(gamma_conn)])
        delta_conn = parameters.get(self.DELTA_CONN)
        if delta_conn is not None:
            args.extend(["--delta-conn", str(delta_conn)])

        run_cli("plan", args, feedback)
        return {
            self.OUT_DIR: out_dir,
            self.OUTPUT_SHP: str(output_shp_param),
        }
