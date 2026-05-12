import math
import numpy as np
import pytest
from shapely.geometry import box, Polygon
from generator.features import polsby_popper_compactness


def test_circle_compactness_close_to_one():
    r = 100.0
    angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    coords = [(r * np.cos(a), r * np.sin(a)) for a in angles]
    circle = Polygon(coords)
    c = polsby_popper_compactness(circle)
    assert 0.98 <= c <= 1.0


def test_square_compactness():
    sq = box(0, 0, 100, 100)
    assert polsby_popper_compactness(sq) == pytest.approx(math.pi / 4.0, abs=0.01)


def test_thin_rectangle_compactness_low():
    rect = box(0, 0, 1000, 10)
    assert polsby_popper_compactness(rect) < 0.1
