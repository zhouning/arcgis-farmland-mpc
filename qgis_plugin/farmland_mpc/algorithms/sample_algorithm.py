"""Tool 2: Sample transitions + pairwise data."""
from __future__ import annotations

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterFile,
    QgsProcessingParameterNumber,
    QgsProcessingParameterEnum,
)

from ._common import run_cli


class SampleAlgorithm(QgsProcessingAlgorithm):
    """Wraps ``farmland-mpc sample``."""

    PREPARED_DIR = "PREPARED_DIR"
    N_EPISODES = "N_EPISODES"
    N_STATES = "N_STATES"
    N_ACTIONS = "N_ACTIONS"
    SEED = "SEED"
    ENV_KIND = "ENV_KIND"

    ENV_KINDS = ("county", "restoration")

    def name(self) -> str:
        return "sample"

    def displayName(self) -> str:  # noqa: N802
        return "2 — Sample (transitions + pairwise dataset)"

    def group(self) -> str:
        return "Pipeline (run in order)"

    def groupId(self) -> str:  # noqa: N802
        return "pipeline"

    def shortHelpString(self) -> str:  # noqa: N802
        return (
            "Phase B of the farmland-mpc pipeline. Generates two datasets "
            "from the prepared environment: an MSE-side transition dataset "
            "(random-policy rollouts) and a pairwise-ranking dataset "
            "(snapshot/restore/execute one action at a time). Both are "
            "consumed by the next stage's training. ~16 min on a 12-thread "
            "CPU per county.\n\n"
            "Equivalent CLI: farmland-mpc sample --prepared-dir ... "
            "--n-episodes 60 --n-states 1000 --n-actions 50"
        )

    def createInstance(self):  # noqa: N802
        return SampleAlgorithm()

    def initAlgorithm(self, config=None):  # noqa: N802
        self.addParameter(QgsProcessingParameterFile(
            self.PREPARED_DIR,
            "prepared/ directory (output of stage 1)",
            behavior=QgsProcessingParameterFile.Folder,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.N_EPISODES, "Random-policy episodes for transitions",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=60, minValue=1,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.N_STATES, "Pairwise: states to snapshot",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=1000, minValue=1,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.N_ACTIONS, "Pairwise: actions per state",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=50, minValue=2,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.SEED, "Random seed",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=0, minValue=0,
        ))
        self.addParameter(QgsProcessingParameterEnum(
            self.ENV_KIND, "Environment",
            options=list(self.ENV_KINDS), defaultValue=0,
        ))

    def processAlgorithm(self, parameters, context, feedback):  # noqa: N802
        prepared_dir = self.parameterAsFile(parameters, self.PREPARED_DIR, context)
        n_episodes = self.parameterAsInt(parameters, self.N_EPISODES, context)
        n_states = self.parameterAsInt(parameters, self.N_STATES, context)
        n_actions = self.parameterAsInt(parameters, self.N_ACTIONS, context)
        seed = self.parameterAsInt(parameters, self.SEED, context)
        env_kind = self.ENV_KINDS[
            self.parameterAsEnum(parameters, self.ENV_KIND, context)
        ]

        args = [
            "--prepared-dir", prepared_dir,
            "--n-episodes", str(n_episodes),
            "--n-states", str(n_states),
            "--n-actions", str(n_actions),
            "--seed", str(seed),
            "--env", env_kind,
            "--verbose",
        ]
        run_cli("sample", args, feedback)
        return {self.PREPARED_DIR: prepared_dir}
