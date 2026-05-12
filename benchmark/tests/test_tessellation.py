import numpy as np
import pytest
from shapely.geometry import Polygon, box
from generator.tessellation import tessellate_domain


def test_block_count_close_to_target(rng_seed):
    n_target = 800
    blocks, domain = tessellate_domain(n_blocks_target=n_target, seed=rng_seed)
    # Clipping at domain boundary may drop a few cells; accept ±5%
    assert abs(len(blocks) - n_target) / n_target < 0.05


def test_blocks_are_non_degenerate_polygons(rng_seed):
    blocks, _ = tessellate_domain(n_blocks_target=200, seed=rng_seed)
    for poly in blocks:
        assert isinstance(poly, Polygon)
        assert poly.is_valid
        assert poly.area > 0


def test_blocks_tile_domain(rng_seed):
    blocks, domain = tessellate_domain(n_blocks_target=200, seed=rng_seed)
    total_area = sum(b.area for b in blocks)
    # Allow 1% gap from floating-point clipping
    assert total_area > 0.99 * domain.area


def test_deterministic_same_seed():
    b1, _ = tessellate_domain(n_blocks_target=100, seed=7)
    b2, _ = tessellate_domain(n_blocks_target=100, seed=7)
    assert len(b1) == len(b2)
    assert all(abs(a.area - b.area) < 1e-9 for a, b in zip(b1, b2))
