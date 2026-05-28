"""Block-level feature derivation matching county_env.py expectations."""
from __future__ import annotations
import math
from shapely.geometry import Polygon


def polsby_popper_compactness(poly: Polygon) -> float:
    """Polsby-Popper compactness: 4*pi*area / perimeter^2, in [0, 1]."""
    perim = poly.length
    if perim <= 0:
        return 0.0
    return max(0.0, min(1.0, 4.0 * math.pi * poly.area / (perim * perim)))
