"""Gaussian random field landuse assignment for blocks."""
from __future__ import annotations
import numpy as np
from opensimplex import OpenSimplex


FARMLAND_CODE = "0110"
FOREST_CODE = "0310"
OTHER_CODE = "0510"


def _sample_grf_at_points(xs, ys, lengthscale_m: float, seed: int) -> np.ndarray:
    """Sample a single-octave OpenSimplex field at point coordinates."""
    noise = OpenSimplex(seed=seed)
    freq = 1.0 / max(lengthscale_m, 1.0)
    return np.array([noise.noise2(x * freq, y * freq) for x, y in zip(xs, ys)])


def assign_landuse_per_block(
    blocks,
    farmland_frac: float,
    forest_frac: float,
    other_frac: float,
    grf_lengthscale_m: float,
    patch_threshold: float,
    seed: int,
) -> list[str]:
    """Assign each block a landuse code via thresholded GRF.

    Sample two GRFs at block centroids, mix as
    `(1 - patch_threshold) * field1 + patch_threshold * field2`, then assign
    codes by quantiles of the mixed field so the resulting fractions match
    the targets exactly. The patch_threshold parameter controls the
    raggedness of patch boundaries (0 = strict spatial, 1 = mostly noise).
    """
    centroids = np.array([(b.centroid.x, b.centroid.y) for b in blocks])
    xs, ys = centroids[:, 0], centroids[:, 1]
    field1 = _sample_grf_at_points(xs, ys, grf_lengthscale_m, seed)
    field2 = _sample_grf_at_points(xs, ys, grf_lengthscale_m * 0.6, seed + 1)
    mixed = (1.0 - patch_threshold) * field1 + patch_threshold * field2

    n = len(blocks)
    n_farm = int(round(n * farmland_frac))
    n_forest = int(round(n * forest_frac))
    order = np.argsort(-mixed)  # descending
    codes = [OTHER_CODE] * n
    for k, idx in enumerate(order):
        if k < n_farm:
            codes[idx] = FARMLAND_CODE
        elif k < n_farm + n_forest:
            codes[idx] = FOREST_CODE
        else:
            codes[idx] = OTHER_CODE
    return codes
