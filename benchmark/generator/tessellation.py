"""Voronoi tessellation for blocks and intra-block parcel subdivision."""
from __future__ import annotations
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box, MultiPolygon


def _finite_voronoi_polygons(points: np.ndarray, bbox: tuple[float, float, float, float]):
    """Build finite Voronoi polygons clipped to bbox via the reflection trick.

    Mirror the input points across each of the four bbox edges so every
    original point's Voronoi cell is bounded by mirrored neighbours.
    """
    x0, y0, x1, y1 = bbox
    pts = np.asarray(points, dtype=float)
    mirrored = np.vstack([
        pts,
        np.column_stack([2 * x0 - pts[:, 0], pts[:, 1]]),
        np.column_stack([2 * x1 - pts[:, 0], pts[:, 1]]),
        np.column_stack([pts[:, 0], 2 * y0 - pts[:, 1]]),
        np.column_stack([pts[:, 0], 2 * y1 - pts[:, 1]]),
    ])
    vor = Voronoi(mirrored)
    n = len(pts)
    domain = box(x0, y0, x1, y1)
    polys: list[Polygon] = []
    for i in range(n):
        region_idx = vor.point_region[i]
        vertex_ids = vor.regions[region_idx]
        if -1 in vertex_ids or not vertex_ids:
            continue
        poly = Polygon([vor.vertices[v] for v in vertex_ids])
        clipped = poly.intersection(domain)
        if clipped.is_empty:
            continue
        if isinstance(clipped, MultiPolygon):
            clipped = max(clipped.geoms, key=lambda g: g.area)
        if isinstance(clipped, Polygon) and clipped.area > 0:
            polys.append(clipped)
    return polys


def tessellate_domain(
    n_blocks_target: int,
    seed: int,
    domain_size_m: float | None = None,
) -> tuple[list[Polygon], Polygon]:
    """Return (blocks, domain_polygon).

    Domain is sized so the mean block area is ~500_000 m^2 (50 ha) by default,
    matching the Bishan mean block scale. If domain_size_m is provided, use it
    as the side of a square domain instead.
    """
    rng = np.random.default_rng(seed)
    if domain_size_m is None:
        mean_block_area = 500_000.0
        total_area = n_blocks_target * mean_block_area
        side = float(np.sqrt(total_area))
    else:
        side = float(domain_size_m)
    bbox = (0.0, 0.0, side, side)
    points = rng.uniform(low=[0.0, 0.0], high=[side, side], size=(n_blocks_target, 2))
    blocks = _finite_voronoi_polygons(points, bbox)
    domain = box(*bbox)
    return blocks, domain


def subdivide_block_into_parcels(
    block: Polygon,
    n_parcels: int,
    seed: int,
) -> list[Polygon]:
    """Voronoi subdivision of a single block into n_parcels.

    Samples n_parcels seed points inside the block (rejection sampling on
    its bounding box), runs Voronoi with the same reflection trick used at
    the block level, and clips to the block boundary.
    """
    from shapely.geometry import Point
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = block.bounds
    seeds: list[tuple[float, float]] = []
    max_tries = n_parcels * 50
    tries = 0
    while len(seeds) < n_parcels and tries < max_tries:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        if block.contains(Point(x, y)):
            seeds.append((x, y))
        tries += 1
    if len(seeds) < 3:
        return [block]
    pts = np.array(seeds)
    raw_polys = _finite_voronoi_polygons(pts, (minx, miny, maxx, maxy))
    out: list[Polygon] = []
    for poly in raw_polys:
        clipped = poly.intersection(block)
        if clipped.is_empty:
            continue
        if isinstance(clipped, MultiPolygon):
            clipped = max(clipped.geoms, key=lambda g: g.area)
        if isinstance(clipped, Polygon) and clipped.area > 0:
            out.append(clipped)
    return out
