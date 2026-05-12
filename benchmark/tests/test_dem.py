import numpy as np
import pytest
from generator.dem import synthesize_dem, derive_slope_degrees, sample_block_slopes
from generator.tessellation import tessellate_domain


def test_dem_shape_and_range(rng_seed):
    dem, transform = synthesize_dem(
        bounds=(0.0, 0.0, 1000.0, 1000.0),
        resolution_m=50.0,
        amplitude_m=100.0,
        lengthscale_m=300.0,
        seed=rng_seed,
    )
    assert dem.shape == (20, 20)
    assert np.all(dem >= 0.0)
    assert dem.max() <= 100.0 * 1.1


def test_plain_dem_is_flat(rng_seed):
    dem, _ = synthesize_dem(
        bounds=(0.0, 0.0, 1000.0, 1000.0),
        resolution_m=50.0,
        amplitude_m=5.0,
        lengthscale_m=800.0,
        seed=rng_seed,
    )
    slope = derive_slope_degrees(dem, pixel_size_m=50.0)
    assert slope.mean() < 5.0


def test_hilly_dem_has_steep_slopes(rng_seed):
    dem, _ = synthesize_dem(
        bounds=(0.0, 0.0, 2000.0, 2000.0),
        resolution_m=50.0,
        amplitude_m=120.0,
        lengthscale_m=400.0,
        seed=rng_seed,
    )
    slope = derive_slope_degrees(dem, pixel_size_m=50.0)
    assert slope.mean() > 5.0
    assert slope.max() > 15.0


def test_sample_block_slopes_returns_per_block_mean(rng_seed):
    blocks, domain = tessellate_domain(n_blocks_target=50, seed=rng_seed)
    minx, miny, maxx, maxy = domain.bounds
    dem, transform = synthesize_dem(
        bounds=(minx, miny, maxx, maxy),
        resolution_m=50.0,
        amplitude_m=80.0,
        lengthscale_m=500.0,
        seed=rng_seed,
    )
    slope = derive_slope_degrees(dem, pixel_size_m=50.0)
    block_slopes = sample_block_slopes(blocks, slope, transform)
    assert len(block_slopes) == len(blocks)
    assert np.all(np.isfinite(block_slopes))
    assert np.all(block_slopes >= 0.0)
