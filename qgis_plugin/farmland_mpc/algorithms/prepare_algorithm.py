"""Tool 1: Prepare DLTB + DEM into per-parcel slope + spatial blocks."""
from __future__ import annotations

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterCrs,
    QgsProcessingParameterEnum,
    QgsProcessingParameterString,
    QgsProcessingParameterNumber,
    QgsProcessingParameterBoolean,
)

from ._common import run_cli


class PrepareAlgorithm(QgsProcessingAlgorithm):
    """Wraps ``farmland-mpc prepare``: per-parcel slope + block construction."""

    DLTB = "DLTB"
    DEM = "DEM"
    OUT = "OUT"
    CRS = "CRS"
    SLOPE_METHOD = "SLOPE_METHOD"
    DLBM_FIELD = "DLBM_FIELD"
    QSDWDM_FIELD = "QSDWDM_FIELD"
    BSM_FIELD = "BSM_FIELD"
    MIN_PARCELS = "MIN_PARCELS"
    MIN_AREA_HA = "MIN_AREA_HA"
    MAX_PARCELS = "MAX_PARCELS"
    MIN_PARCELS_PER_TOWNSHIP = "MIN_PARCELS_PER_TOWNSHIP"
    SKIP_BLOCKS = "SKIP_BLOCKS"

    SLOPE_METHODS = ("auto", "gradient_geographic", "horn_projected", "from_field")

    def name(self) -> str:
        return "prepare"

    def displayName(self) -> str:  # noqa: N802
        return "1 — Prepare (DLTB + DEM → slope + blocks)"

    def group(self) -> str:
        return "Pipeline (run in order)"

    def groupId(self) -> str:  # noqa: N802
        return "pipeline"

    def shortHelpString(self) -> str:  # noqa: N802
        return (
            "Phase A+B+C of the farmland-mpc pipeline. Reads a Third "
            "National Land Survey DLTB polygon layer plus a DEM raster, "
            "computes per-parcel slope (auto-selecting the geographic-CRS "
            "or projected algorithm), aggregates parcels into spatial "
            "blocks (the planning units the model sees), and writes a "
            "prepared/ directory consumable by the next three stages.\n\n"
            "Equivalent CLI: farmland-mpc prepare --dltb ... --dem ... "
            "--out <prepared_dir> --crs EPSG:32648"
        )

    def createInstance(self):  # noqa: N802
        return PrepareAlgorithm()

    def initAlgorithm(self, config=None):  # noqa: N802
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.DLTB,
            "DLTB cadastral polygons",
            types=[0],  # vector polygon
        ))
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.DEM,
            "DEM raster (geographic CRS preferred; e.g. Copernicus 30 m)",
        ))
        self.addParameter(QgsProcessingParameterFolderDestination(
            self.OUT,
            "Output prepared/ directory",
        ))
        self.addParameter(QgsProcessingParameterCrs(
            self.CRS,
            "Target projected CRS for slope/area metrics",
            defaultValue="EPSG:32648",
        ))
        self.addParameter(QgsProcessingParameterEnum(
            self.SLOPE_METHOD,
            "Slope algorithm",
            options=list(self.SLOPE_METHODS),
            defaultValue=0,
        ))
        self.addParameter(QgsProcessingParameterString(
            self.DLBM_FIELD,
            "Land-use code field",
            defaultValue="DLBM",
            optional=True,
        ))
        self.addParameter(QgsProcessingParameterString(
            self.QSDWDM_FIELD,
            "Township-ownership field",
            defaultValue="QSDWDM",
            optional=True,
        ))
        self.addParameter(QgsProcessingParameterString(
            self.BSM_FIELD,
            "Parcel-ID field",
            defaultValue="BSM",
            optional=True,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_PARCELS,
            "Block min parcels",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=3,
            minValue=1,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_AREA_HA,
            "Block min area (ha)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.5,
            minValue=0.0,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.MAX_PARCELS,
            "Block max parcels before subdivision",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=30,
            minValue=1,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.MIN_PARCELS_PER_TOWNSHIP,
            "Drop townships with fewer than N parcels",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=50,
            minValue=0,
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.SKIP_BLOCKS,
            "Skip block construction (run only Phase A slope)",
            defaultValue=False,
        ))

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        dltb_layer = self.parameterAsVectorLayer(parameters, self.DLTB, context)
        if dltb_layer is None:
            # parameterAsSource handles non-layer feature sources (e.g. memory)
            src = self.parameterAsSource(parameters, self.DLTB, context)
            dltb_path = src.sourceName() if src else parameters[self.DLTB]
        else:
            dltb_path = dltb_layer.dataProvider().dataSourceUri().split("|")[0]

        dem_layer = self.parameterAsRasterLayer(parameters, self.DEM, context)
        dem_path = dem_layer.dataProvider().dataSourceUri()

        out_dir = self.parameterAsString(parameters, self.OUT, context)
        crs = self.parameterAsCrs(parameters, self.CRS, context)
        crs_str = crs.authid() or "EPSG:32648"
        slope_method = self.SLOPE_METHODS[
            self.parameterAsEnum(parameters, self.SLOPE_METHOD, context)
        ]
        dlbm_field = self.parameterAsString(parameters, self.DLBM_FIELD, context) or "DLBM"
        qsdwdm_field = self.parameterAsString(parameters, self.QSDWDM_FIELD, context) or "QSDWDM"
        bsm_field = self.parameterAsString(parameters, self.BSM_FIELD, context) or "BSM"
        min_parcels = self.parameterAsInt(parameters, self.MIN_PARCELS, context)
        min_area_ha = self.parameterAsDouble(parameters, self.MIN_AREA_HA, context)
        max_parcels = self.parameterAsInt(parameters, self.MAX_PARCELS, context)
        min_per_township = self.parameterAsInt(
            parameters, self.MIN_PARCELS_PER_TOWNSHIP, context
        )
        skip_blocks = self.parameterAsBool(parameters, self.SKIP_BLOCKS, context)

        args = [
            "--dltb", str(dltb_path),
            "--dem", str(dem_path),
            "--out", str(out_dir),
            "--crs", crs_str,
            "--slope-method", slope_method,
            "--dlbm-field", dlbm_field,
            "--qsdwdm-field", qsdwdm_field,
            "--bsm-field", bsm_field,
            "--min-parcels", str(min_parcels),
            "--min-area-ha", str(min_area_ha),
            "--max-parcels", str(max_parcels),
            "--min-parcels-per-township", str(min_per_township),
            "--verbose",
        ]
        if skip_blocks:
            args.append("--skip-blocks")

        run_cli("prepare", args, feedback)

        return {self.OUT: out_dir}
