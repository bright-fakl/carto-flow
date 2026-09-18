"""Connected-component detection and uniform pre-scaling of geometry groups.

Functions
---------
compute_connected_components
    Detect connected components among geometries (Union-Find over adjacency).
prescale_connected_components
    Uniformly scale each component to its target total area.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .adjacency import find_adjacent_pairs

__all__ = [
    "components_from_adjacency",
    "compute_connected_components",
    "prescale_connected_components",
]


# ---------------------------------------------------------------------------
# Internal Union-Find
# ---------------------------------------------------------------------------


def _union_find(n: int, pairs: list[tuple[int, int, float]]) -> list[list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in pairs:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


# ---------------------------------------------------------------------------
# Internal: component detection from adjacency matrix
# ---------------------------------------------------------------------------


def components_from_adjacency(adj: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """Derive connected components from a dense adjacency matrix.

    Uses the existing Union-Find; no geometry operations required.

    Parameters
    ----------
    adj : np.ndarray, shape (n, n)
        Dense adjacency matrix; a nonzero entry above the diagonal marks an
        edge between the corresponding pair of nodes.

    Returns
    -------
    component_labels : np.ndarray, shape (n,)
        Zero-based component index for each node.
    components : list of list of int
        Each inner list contains the node indices of one component.
    """
    rows, cols = np.where(np.triu(adj, k=1) > 0)
    pairs = [(int(r), int(c), 1.0) for r, c in zip(rows, cols, strict=False)]
    groups = _union_find(adj.shape[0], pairs)
    labels = np.empty(adj.shape[0], dtype=np.int32)
    for comp_idx, indices in enumerate(groups):
        for g_idx in indices:
            labels[g_idx] = comp_idx
    return labels, groups


# Backwards-compatible alias for the pre-public name.
_components_from_adjacency = components_from_adjacency


# ---------------------------------------------------------------------------
# Public: component detection
# ---------------------------------------------------------------------------


def compute_connected_components(
    geometries: list,
    distance_tolerance: float | None = None,
) -> tuple[np.ndarray, list[list[int]]]:
    """Detect connected components among geometries.

    Two geometries belong to the same component when they are geographically
    adjacent (share a boundary or overlap within *distance_tolerance*).

    Parameters
    ----------
    geometries : list of shapely.Geometry
        Input polygon geometries.
    distance_tolerance : float or None
        Passed to :func:`~carto_flow.geo_utils.adjacency.find_adjacent_pairs`.
        ``None`` → auto-computed as 0.1 % of the average geometry diameter.

    Returns
    -------
    component_labels : np.ndarray, shape (n,)
        Zero-based component index for each geometry.
    components : list of list of int
        Each inner list contains the geometry indices of one component.
    """
    n = len(geometries)
    pairs = find_adjacent_pairs(geometries, distance_tolerance)
    groups = _union_find(n, pairs)

    component_labels = np.empty(n, dtype=np.int32)
    for comp_idx, indices in enumerate(groups):
        for g_idx in indices:
            component_labels[g_idx] = comp_idx

    return component_labels, groups


# ---------------------------------------------------------------------------
# Public: pre-scaling
# ---------------------------------------------------------------------------


def prescale_connected_components(
    geometries: list,
    values: np.ndarray,
    target_density: float,
    *,
    components: list[list[int]] | None = None,
    distance_tolerance: float | None = None,
) -> list:
    """Pre-scale each connected component to its target total area.

    Uniformly scales each group of geometrically adjacent polygons so that
    the component's total area equals the area implied by the global target
    density.  This reduces over-deformation of outer polygons in subsequent
    morphing steps.

    Parameters
    ----------
    geometries : list of shapely.Geometry
        Input polygon geometries.
    values : array-like
        Data values (e.g. population or tile counts) for each geometry.
    target_density : float
        Target equilibrium density (values per area unit). Each component's
        target area is computed as ``sum(component_values) / target_density``.
    components : list of list of int, optional
        Pre-computed connected components (second return value of
        :func:`compute_connected_components`).  When provided, adjacency
        detection is skipped.  When ``None``, components are detected
        internally.
    distance_tolerance : float or None
        Passed to adjacency detection when *components* is ``None``.

    Returns
    -------
    list of shapely.Geometry
        Scaled geometries in the same order as the input.

    Notes
    -----
    Each component is scaled uniformly around its area-weighted centroid so
    shape is preserved and the centroid stays in place.  Components with zero
    current or target area are left unchanged.
    """
    from shapely import affinity

    values_array = np.asarray(values, dtype=float)

    if components is None:
        _, components = compute_connected_components(geometries, distance_tolerance)

    result = list(geometries)

    for component_indices in components:
        component_geoms = [geometries[i] for i in component_indices]
        component_values = values_array[component_indices]

        current_area = sum(g.area for g in component_geoms)
        target_area = float(np.sum(component_values)) / target_density

        if current_area <= 0 or target_area <= 0:
            continue

        scale_factor = math.sqrt(target_area / current_area)
        if abs(scale_factor - 1.0) < 1e-8:
            continue

        cx = sum(g.centroid.x * g.area for g in component_geoms) / current_area
        cy = sum(g.centroid.y * g.area for g in component_geoms) / current_area

        for idx, geom in zip(component_indices, component_geoms, strict=False):
            result[idx] = affinity.scale(geom, scale_factor, scale_factor, origin=(cx, cy))

    return result
