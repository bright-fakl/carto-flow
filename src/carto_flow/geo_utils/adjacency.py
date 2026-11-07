"""
Shared adjacency computation utilities for polygon geometries.

Works directly with shapely geometry objects (no geopandas dependency),
making it reusable across submodules.

Functions
---------
find_adjacent_pairs
    Find adjacent geometry pairs using buffered intersection.
find_bordering_pairs
    Find geometry pairs that share a border of a minimum length.

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

__all__ = ["find_adjacent_pairs", "find_bordering_pairs"]


def find_adjacent_pairs(
    geometries: list,
    distance_tolerance: float | None = None,
    min_shared_length: float | None = None,
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
    min_shared_length : float or None
        Minimum shared border length (in the same units as the coordinate
        system) for a pair to be considered adjacent. Pairs whose shared
        border is shorter than this value are excluded even when they pass
        the ``distance_tolerance`` check. ``None`` = no extra filter.

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
            # True shared border: portion of i's boundary inside j's buffer.
            # shapely.boundary is None for GeometryCollections and an EMPTY
            # GeometryCollection for degenerate (Point / zero-area) geometries;
            # fall back to the buffered intersection length in both cases.
            if boundaries[i] is None or boundaries[i].is_empty:
                shared_length = detection.length
            else:
                shared_length = boundaries[i].intersection(buffered[j]).length
            if shared_length > 0:
                if min_shared_length is not None and shared_length < min_shared_length:
                    continue
                pairs.append((i, j, float(shared_length)))

    return pairs


def find_bordering_pairs(
    geometries: list,
    tile_size: float | None = None,
    buffer_fraction: float = 0.01,
    min_length_fraction: float = 0.05,
) -> list[tuple[int, int, float]]:
    """Find geometry pairs that share a border of a minimum length.

    Two geometries are neighbors when their boundaries run alongside each
    other for more than a minimum length. The shared length of a pair is
    measured twice: as the length of the boundary of ``a`` that lies within a
    thin buffer of the boundary of ``b``, and the reverse. The mean of both
    is the shared border length. Unlike a distance tolerance, this separates
    regions that border each other from regions that only come close (across
    a strait) or that touch at a corner.

    Parameters
    ----------
    geometries : list of shapely.Geometry
        Polygon geometries to test.
    tile_size : float, optional
        Reference length for the buffer and the minimum length. Default: the
        diameter of a circle with the mean area of the geometries.
    buffer_fraction : float
        Buffer distance around each boundary, as a fraction of *tile_size*.
        Default: 0.01.
    min_length_fraction : float
        A pair is returned when its shared border length exceeds this
        fraction of *tile_size*. Default: 0.05.

    Returns
    -------
    list[tuple[int, int, float]]
        ``(i, j, shared_length)`` for each neighboring pair with ``i < j``,
        in increasing order of ``i`` then ``j``.
    """
    n = len(geometries)
    if n < 2:
        return []
    geoms = shapely.make_valid(np.asarray(geometries, dtype=object))
    if tile_size is None:
        mean_area = float(np.mean(shapely.area(geoms)))
        tile_size = 2.0 * float(np.sqrt(max(mean_area, 0.0) / np.pi))
    if tile_size <= 0:
        return []
    width = buffer_fraction * tile_size
    min_length = min_length_fraction * tile_size

    boundaries = shapely.boundary(geoms)
    zones = shapely.buffer(boundaries, width)
    tree = STRtree(zones)
    left, right = tree.query(zones, predicate="intersects")
    keep = left < right
    left, right = left[keep], right[keep]
    if len(left) == 0:
        return []
    forward = shapely.length(shapely.intersection(boundaries[left], zones[right]))
    backward = shapely.length(shapely.intersection(boundaries[right], zones[left]))
    shared = 0.5 * (forward + backward)
    order = np.lexsort((right, left))
    return [(int(left[k]), int(right[k]), float(shared[k])) for k in order if shared[k] > min_length]
