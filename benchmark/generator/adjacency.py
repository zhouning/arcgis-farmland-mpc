"""Planar block adjacency via shared edges (positive-length intersection)."""
from __future__ import annotations
import numpy as np
from shapely.geometry import Polygon, LineString, MultiLineString
from shapely.strtree import STRtree


def build_block_adjacency(blocks: list[Polygon]) -> list[np.ndarray]:
    """Return adjacency list: blocks[i] neighbours have shared boundary segment.

    Uses STRtree to find candidate neighbours, then filters by intersection
    being a 1-D shape with positive length (true edge-sharing, not point touches).
    """
    n = len(blocks)
    tree = STRtree(blocks)
    out: list[list[int]] = [[] for _ in range(n)]
    for i, poly in enumerate(blocks):
        cand_ids = tree.query(poly)
        for j in cand_ids:
            if j == i or j < i:
                continue
            inter = poly.boundary.intersection(blocks[j].boundary)
            if isinstance(inter, (LineString, MultiLineString)) and inter.length > 1e-6:
                out[i].append(int(j))
                out[j].append(int(i))
    return [np.array(sorted(lst), dtype=np.intp) for lst in out]
