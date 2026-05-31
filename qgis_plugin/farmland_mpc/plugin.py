"""QGIS plugin lifecycle hooks for Farmland MPC.

The plugin only registers a Processing Provider; it has no menu items
or toolbar buttons of its own. Algorithms appear under
``Processing Toolbox → Farmland MPC``.
"""
from __future__ import annotations

from qgis.core import QgsApplication

from .farmland_mpc_provider import FarmlandMpcProvider


class FarmlandMpcPlugin:
    """Registered with QGIS via classFactory in __init__.py."""

    def __init__(self, iface):
        self.iface = iface
        self._provider: FarmlandMpcProvider | None = None

    def initProcessing(self) -> None:  # noqa: N802 — QGIS API name
        self._provider = FarmlandMpcProvider()
        QgsApplication.processingRegistry().addProvider(self._provider)

    def initGui(self) -> None:  # noqa: N802
        self.initProcessing()

    def unload(self) -> None:
        if self._provider is not None:
            QgsApplication.processingRegistry().removeProvider(self._provider)
            self._provider = None
