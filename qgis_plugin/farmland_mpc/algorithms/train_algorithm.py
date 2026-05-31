"""Tool 3: Train the contrastive transition-model ensemble."""
from __future__ import annotations

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
)

from ._common import run_cli


class TrainAlgorithm(QgsProcessingAlgorithm):
    """Wraps ``farmland-mpc train``."""

    PREPARED_DIR = "PREPARED_DIR"
    N_MEMBERS = "N_MEMBERS"
    EPOCHS = "EPOCHS"
    PATIENCE = "PATIENCE"
    LAMBDA_RANK = "LAMBDA_RANK"
    MARGIN = "MARGIN"
    BATCH_SIZE = "BATCH_SIZE"
    SEED_BASE = "SEED_BASE"
    TORCH_THREADS = "TORCH_THREADS"
    OUT_SUBDIR = "OUT_SUBDIR"

    def name(self) -> str:
        return "train"

    def displayName(self) -> str:  # noqa: N802
        return "3 — Train (contrastive ensemble + ONNX export)"

    def group(self) -> str:
        return "Pipeline (run in order)"

    def groupId(self) -> str:  # noqa: N802
        return "pipeline"

    def shortHelpString(self) -> str:  # noqa: N802
        return (
            "Phase C of the farmland-mpc pipeline. Trains a 3-member "
            "transition-model ensemble with the auxiliary pairwise "
            "large-margin loss (λ_rank=5.0 by default) and exports each "
            "member to ONNX with N_blocks baked in. ~40-60 min per ensemble "
            "on a 12-thread CPU.\n\n"
            "Equivalent CLI: farmland-mpc train --prepared-dir ... "
            "--epochs 30 --lambda-rank 5.0 --margin 0.1"
        )

    def createInstance(self):  # noqa: N802
        return TrainAlgorithm()

    def initAlgorithm(self, config=None):  # noqa: N802
        self.addParameter(QgsProcessingParameterFile(
            self.PREPARED_DIR,
            "prepared/ directory (output of stage 1+2)",
            behavior=QgsProcessingParameterFile.Folder,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.N_MEMBERS, "Ensemble members",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=3, minValue=1,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.EPOCHS, "Epochs",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=30, minValue=1,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.PATIENCE, "Early-stop patience",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=8, minValue=1,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.LAMBDA_RANK,
            "λ_rank (auxiliary pairwise margin loss weight)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=5.0, minValue=0.0,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.MARGIN, "Pairwise margin",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.1, minValue=0.0,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.BATCH_SIZE, "Batch size",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=256, minValue=1,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.SEED_BASE, "Seed base",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=0, minValue=0,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.TORCH_THREADS,
            "PyTorch threads (0 = auto)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=0, minValue=0,
        ))
        self.addParameter(QgsProcessingParameterString(
            self.OUT_SUBDIR,
            "Output subdirectory under prepared/ (use distinct values for "
            "parallel ensemble training)",
            defaultValue="tool3",
        ))

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        prepared_dir = self.parameterAsFile(parameters, self.PREPARED_DIR, context)

        args = [
            "--prepared-dir", prepared_dir,
            "--n-members", str(self.parameterAsInt(parameters, self.N_MEMBERS, context)),
            "--epochs", str(self.parameterAsInt(parameters, self.EPOCHS, context)),
            "--patience", str(self.parameterAsInt(parameters, self.PATIENCE, context)),
            "--lambda-rank", str(self.parameterAsDouble(parameters, self.LAMBDA_RANK, context)),
            "--margin", str(self.parameterAsDouble(parameters, self.MARGIN, context)),
            "--batch-size", str(self.parameterAsInt(parameters, self.BATCH_SIZE, context)),
            "--seed-base", str(self.parameterAsInt(parameters, self.SEED_BASE, context)),
            "--torch-threads", str(self.parameterAsInt(parameters, self.TORCH_THREADS, context)),
            "--out-subdir", self.parameterAsString(parameters, self.OUT_SUBDIR, context),
            "--verbose",
        ]
        run_cli("train", args, feedback)
        return {self.PREPARED_DIR: prepared_dir}
