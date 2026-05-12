"""Synthetic DEM generation (Perlin/simplex noise) and Horn-algorithm slope."""
from __future__ import annotations
import numpy as np
from opensimplex import OpenSimplex
from rasterio.features import rasterize
from rasterio.transform import from_origin


def synthesize_dem(
    bounds: tuple[float, float, float, float],
    resolution_m: float,
    amplitude_m: float,
    lengthscale_m: float,
    seed: int,
    n_octaves: int = 4,
    persistence: float = 0.5,
):
    """Return (dem_2d, rasterio_transform). DEM values in metres, >= 0."""
    minx, miny, maxx, maxy = bounds
    width = int(np.ceil((maxx - minx) / resolution_m))
    height = int(np.ceil((maxy - miny) / resolution_m))
    noise = OpenSimplex(seed=seed)
    dem = np.zeros((height, width), dtype=np.float64)
    norm = 0.0
    amp = 1.0
    freq = 1.0 / lengthscale_m
    for _ in range(n_octaves):
        for j in range(height):
            y = miny + (j + 0.5) * resolution_m
            for i in range(width):
                x = minx + (i + 0.5) * resolution_m
                dem[j, i] += amp * noise.noise2(x * freq, y * freq)
        norm += amp
        amp *= persistence
        freq *= 2.0
    dem /= norm
    dem = (dem - dem.min())
    if dem.max() > 0:
        dem = dem / dem.max() * amplitude_m
    transform = from_origin(minx, maxy, resolution_m, resolution_m)
    return dem, transform


def derive_slope_degrees(dem: np.ndarray, pixel_size_m: float) -> np.ndarray:
    """Horn's algorithm slope in degrees on a regular grid."""
    dz_dx = np.zeros_like(dem)
    dz_dy = np.zeros_like(dem)
    dz_dx[:, 1:-1] = (dem[:, 2:] - dem[:, :-2]) / (2.0 * pixel_size_m)
    dz_dy[1:-1, :] = (dem[2:, :] - dem[:-2, :]) / (2.0 * pixel_size_m)
    dz_dx[:, 0] = dz_dx[:, 1]
    dz_dx[:, -1] = dz_dx[:, -2]
    dz_dy[0, :] = dz_dy[1, :]
    dz_dy[-1, :] = dz_dy[-2, :]
    magnitude = np.sqrt(dz_dx**2 + dz_dy**2)
    return np.degrees(np.arctan(magnitude))


def sample_block_slopes(blocks, slope_raster: np.ndarray, transform) -> np.ndarray:
    """Area-weighted mean slope per block via rasterio.features.rasterize."""
    height, width = slope_raster.shape
    out = np.zeros(len(blocks), dtype=np.float64)
    for idx, poly in enumerate(blocks):
        mask = rasterize(
            [(poly, 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)
        if mask.sum() == 0:
            cx, cy = poly.centroid.x, poly.centroid.y
            col = int((cx - transform.c) / transform.a)
            row = int((transform.f - cy) / (-transform.e))
            col = np.clip(col, 0, width - 1)
            row = np.clip(row, 0, height - 1)
            out[idx] = slope_raster[row, col]
        else:
            out[idx] = slope_raster[mask].mean()
    return out
