"""_fix_island_assignments joins a disconnected group across a one-tile gap by shifting each region at most one tile."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_array

from carto_flow.symbol_cartogram.layouts.grid._placement import _fix_island_assignments


def _grid(width: int, height: int):
    """Centers and edge adjacency of a square tiling, tile index = y * width + x."""
    centers = np.array([(x, y) for y in range(height) for x in range(width)], dtype=float)
    m = len(centers)
    edge = np.zeros((m, m), dtype=bool)
    for a in range(m):
        for b in range(m):
            edge[a, b] = abs(centers[a] - centers[b]).sum() == 1
    return centers, csr_array(edge)


def test_one_tile_gap_is_closed_by_small_moves():
    centers, tile_adj = _grid(9, 3)
    # Two 3x3 blocks of regions on columns 0-2 and 4-6, with the free column 3 between them.
    assignments = np.array([y * 9 + x for y in range(3) for x in (0, 1, 2, 4, 5, 6)])
    centroids = centers[assignments]
    n = len(assignments)
    region_adj = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            region_adj[a, b] = float(a != b and abs(centroids[a] - centroids[b]).sum() <= 2)
    fixed = _fix_island_assignments(assignments, tile_adj, centroids, centers, region_adj)

    # The gap column is filled by one region; the old behavior shifted all nine regions of the right block.
    assert int((fixed != assignments).sum()) <= 3
    assert np.abs(centers[fixed] - centers[assignments]).sum(axis=1).max() <= 1.0
