"""
Shared adjacency computation utilities for polygon geometries.

Works directly with shapely geometry objects (no geopandas dependency),
making it reusable across submodules.

Functions
---------
find_adjacent_pairs
    Find adjacent geometry pairs using buffered intersection.

Examples
--------
>>> from shapely.geometry import box
>>> from carto_flow.geo_utils.adjacency import find_adjacent_pairs
>>>
>>> geometries = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(5, 5, 6, 6)]
>>> pairs = find_adjacent_pairs(geometries)
>>> # Returns [(0, 1, 1.0)] — only the first two boxes touch
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely import STRtree

__all__ = ["find_adjacent_pairs"]


def find_adjacent_pairs(
    geometries: list,
    distance_tolerance: float | None = None,
) -> list[tuple[int, int, float]]:
    """Find adjacent geometry pairs using buffered intersection.

    Uses a buffer-based approach to handle small gaps in real-world data
    (e.g., floating-point imprecision between polygon boundaries).

    Parameters
    ----------
    geometries : list of shapely.Geometry
        Polygon geometries to test for adjacency.
    distance_tolerance : float, optional
        Buffer distance for detecting adjacency. Pairs whose shared border
        is shorter than this threshold are discarded. If None, auto-computed
        as 0.1% of the average geometry diameter (same heuristic as
        ``symbol_cartogram.adjacency``).

    Returns
    -------
    list[tuple[int, int, float]]
        List of ``(i, j, shared_length)`` tuples for each adjacent pair
        where ``i < j``.
    """
    n = len(geometries)
    if n < 2:
        return []

    if distance_tolerance is None:
        avg_diameter = float(np.mean([np.sqrt(max(g.area, 0.0)) for g in geometries]))
        distance_tolerance = avg_diameter * 0.001

    buffered = [shapely.buffer(g, distance_tolerance / 2) for g in geometries]
    boundaries = [shapely.boundary(g) for g in geometries]

    # Build spatial index on original geometries; query with buffered geometry
    # of i to find candidates where g_j intersects buffered[i].
    tree = STRtree(geometries)

    pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        candidates = tree.query(buffered[i], predicate="intersects")
        for j in candidates:
            if i >= int(j):
                continue
            j = int(j)
            detection = buffered[i].intersection(buffered[j])
            if detection.is_empty or detection.length <= distance_tolerance:
                continue
            # True shared border: portion of i's boundary inside j's buffer
            shared_length = boundaries[i].intersection(buffered[j]).length
            if shared_length > 0:
                pairs.append((i, j, float(shared_length)))

    return pairs
