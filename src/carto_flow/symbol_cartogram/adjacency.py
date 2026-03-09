"""Adjacency computation for symbol cartograms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .options import AdjacencyMode

if TYPE_CHECKING:
    import geopandas as gpd


def _find_adjacent_pairs(
    gdf: gpd.GeoDataFrame,
    distance_tolerance: float,
) -> list[tuple[int, int, float]]:
    """Find adjacent geometry pairs using buffered intersection.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with polygon geometries.
    distance_tolerance : float
        Buffer distance for detecting adjacency.

    Returns
    -------
    list[tuple[int, int, float]]
        List of ``(i, j, shared_length)`` tuples for each adjacent pair
        where ``i < j``.

    """
    from carto_flow.geo_utils.adjacency import find_adjacent_pairs

    return find_adjacent_pairs(gdf.geometry.tolist(), distance_tolerance)


def compute_adjacency(
    gdf: gpd.GeoDataFrame,
    mode: AdjacencyMode = AdjacencyMode.BINARY,
    distance_tolerance: float | None = None,
) -> NDArray[np.floating]:
    """Compute adjacency matrix from polygon geometries.

    Uses buffering to handle small gaps in real-world data.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with polygon geometries.
    mode : AdjacencyMode
        - ``BINARY``: 1 if adjacent, 0 otherwise.
        - ``WEIGHTED``: ``w[i,j] = shared_length / perimeter[i]`` (asymmetric).
        - ``AREA_WEIGHTED``: ``w[i,j] = area[j] / sum(neighbor_areas_of_i)``.
          Rows sum to 1. Larger neighbors pull more strongly.
    distance_tolerance : float, optional
        Buffer distance for detecting adjacency. If None, auto-computed
        as 0.1% of average region diameter.

    Returns
    -------
    np.ndarray
        Adjacency matrix of shape (n, n).
        For WEIGHTED and AREA_WEIGHTED modes, the matrix is asymmetric.

    """
    n = len(gdf)

    # Auto-compute tolerance if not provided
    if distance_tolerance is None:
        avg_diameter = np.mean([np.sqrt(g.area) for g in gdf.geometry])
        distance_tolerance = avg_diameter * 0.001

    pairs = _find_adjacent_pairs(gdf, distance_tolerance)
    adjacency: NDArray[np.floating] = np.zeros((n, n), dtype=float)

    if mode == AdjacencyMode.BINARY:
        for i, j, _length in pairs:
            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0

    elif mode == AdjacencyMode.WEIGHTED:
        perimeters = np.array([g.length for g in gdf.geometry])
        for i, j, shared_length in pairs:
            adjacency[i, j] = shared_length / perimeters[i]
            adjacency[j, i] = shared_length / perimeters[j]

    elif mode == AdjacencyMode.AREA_WEIGHTED:
        # First build binary adjacency, then weight by neighbor area fraction
        for i, j, _length in pairs:
            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0

        areas = np.array([g.area for g in gdf.geometry])
        weighted = np.zeros((n, n), dtype=float)
        for i in range(n):
            neighbor_mask = adjacency[i] > 0
            if not neighbor_mask.any():
                continue
            neighbor_areas = areas[neighbor_mask]
            total = neighbor_areas.sum()
            if total > 0:
                weighted[i, neighbor_mask] = areas[neighbor_mask] / total
        adjacency = weighted

    else:
        raise ValueError(f"Unknown adjacency mode: {mode}")

    return adjacency
