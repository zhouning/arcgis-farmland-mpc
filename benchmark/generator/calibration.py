"""Anchor-preset calibration: compare synthetic init stats to real targets."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class AnchorTargets:
    name: str
    init_slope_deg: float
    init_contiguity: float
    init_baimu_count: Optional[float]
    init_baimu_area_ha: Optional[float]


def compute_init_stats(env) -> dict[str, float]:
    """After env.reset() has been called, snapshot key initial state stats."""
    return {
        "init_slope_deg": float(env.avg_farmland_slope),
        "init_contiguity": float(env.contiguity),
        "init_baimu_count": float(env.baimu_count),
        "init_baimu_area_ha": float(env.baimu_total_area) / 10_000.0,
    }


def verify_anchor_within_tolerance(
    target: AnchorTargets,
    actual: dict[str, float],
    tolerance: float = 0.5,
) -> dict:
    """Return a calibration report.

    Pass means every non-None target is within ±tolerance fraction of actual.
    When a target field is None, the field is reported as passing (skipped).
    """
    deltas: dict[str, dict] = {}
    passed = True
    for key in (
        "init_slope_deg",
        "init_contiguity",
        "init_baimu_count",
        "init_baimu_area_ha",
    ):
        tgt = getattr(target, key)
        if tgt is None:
            deltas[key] = {"target": None, "actual": actual.get(key), "within_tolerance": True}
            continue
        act = actual.get(key)
        if act is None:
            deltas[key] = {"target": tgt, "actual": None, "within_tolerance": False}
            passed = False
            continue
        denom = max(abs(tgt), 1e-9)
        frac = abs(act - tgt) / denom
        ok = frac <= tolerance
        deltas[key] = {
            "target": tgt,
            "actual": act,
            "frac_off": frac,
            "within_tolerance": ok,
        }
        if not ok:
            passed = False
    return {
        "anchor": target.name,
        "tolerance": tolerance,
        "passed": passed,
        "deltas": deltas,
    }
