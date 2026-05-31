"""QGIS Processing Provider for the four farmland-mpc CLI stages."""
from __future__ import annotations

from qgis.core import QgsProcessingProvider

from .algorithms.prepare_algorithm import PrepareAlgorithm
from .algorithms.sample_algorithm import SampleAlgorithm
from .algorithms.train_algorithm import TrainAlgorithm
from .algorithms.plan_algorithm import PlanAlgorithm


class FarmlandMpcProvider(QgsProcessingProvider):
    """Exposes four algorithms wrapping ``farmland-mpc`` subcommands."""

    def id(self) -> str:
        return "farmland_mpc"

    def name(self) -> str:
        return "Farmland MPC"

    def longName(self) -> str:
        return "Farmland MPC (county-scale consolidation pipeline)"

    def icon(self):  # noqa: D401
        from qgis.PyQt.QtGui import QIcon
        from pathlib import Path
        ico = Path(__file__).parent / "icon.png"
        if ico.exists():
            return QIcon(str(ico))
        return QgsProcessingProvider.icon(self)

    def loadAlgorithms(self) -> None:  # noqa: N802 — QGIS API name
        for cls in (
            PrepareAlgorithm,
            SampleAlgorithm,
            TrainAlgorithm,
            PlanAlgorithm,
        ):
            self.addAlgorithm(cls())
