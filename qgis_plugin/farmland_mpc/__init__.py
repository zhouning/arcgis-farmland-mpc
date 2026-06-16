"""Farmland MPC QGIS plugin entry point.

Companion to the Scientific Reports submission
"Reproducible model-based planning for county-scale farmland
consolidation in fragmented mountain landscapes" (Zhou & Jing 2026).
Wraps the four farmland-mpc CLI subcommands (prepare / sample / train /
plan) as native QGIS Processing algorithms.

Calls the farmland-mpc executable as a subprocess. You must install the
farmland-mpc package in a separate conda environment first (see the
project README).
"""
from __future__ import annotations


def classFactory(iface):  # noqa: N802 — required by QGIS plugin loader
    """Entry point QGIS calls when activating the plugin."""
    from .plugin import FarmlandMpcPlugin
    return FarmlandMpcPlugin(iface)
