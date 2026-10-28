"""The vectorized cost matrix of assign_to_grid_hungarian against a loop over regions, pairs and tiles."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_array
from scipy.spatial.distance import cdist

from carto_flow.symbol_cartogram.layouts.grid import _placement
from carto_flow.symbol_cartogram.layouts.grid._placement import _bfs_distance_matrix, assign_to_grid_hungarian


def _square_grid(side: int):
    """Grid cell centers and the edge and vertex-only adjacency of a square tiling."""
    centers = np.array([(x, y) for y in range(side) for x in range(side)], dtype=float)
    m = len(centers)
    edge = np.zeros((m, m), dtype=bool)
    vertex = np.zeros((m, m), dtype=bool)
    for a in range(m):
        for b in range(m):
            dx, dy = abs(centers[a, 0] - centers[b, 0]), abs(centers[a, 1] - centers[b, 1])
            edge[a, b] = dx + dy == 1
            vertex[a, b] = dx == 1 and dy == 1
    return centers, csr_array(edge), csr_array(vertex)


def _reference_combined(
    assignment, centroids, grid_centers, adjacency, tile_adj, vertex_adj, source_indices, weights
) -> np.ndarray:
    """Cost matrix of the first refinement pass, one scalar operation per region pair and tile."""
    origin_w, neighbor_w, topology_w = weights
    n, m = len(centroids), len(grid_centers)
    origin_cost = cdist(centroids, grid_centers, metric="sqeuclidean")
    origin_cost = origin_cost / np.max(origin_cost)
    bfs = _bfs_distance_matrix(tile_adj)
    tile_adj_d, vertex_adj_d = tile_adj.toarray(), vertex_adj.toarray()

    orig_angles = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i != j and adjacency[i, j] > 0 and source_indices[i] != source_indices[j]:
                diff = centroids[j] - centroids[i]
                orig_angles[i, j] = np.arctan2(diff[1], diff[0])

    neighbor_cost = np.zeros((n, m))
    orientation_cost = np.zeros((n, m))
    for i in range(n):
        for j in range(n):
            if i == j or adjacency[i, j] <= 0:
                continue
            j_cell, w = assignment[j], adjacency[i, j]
            raw = np.maximum(bfs[:, j_cell] - 1, 0).astype(float)
            penalty = raw * raw
            penalty[np.flatnonzero(vertex_adj_d[:, j_cell])] = 0.2
            penalty[np.flatnonzero(tile_adj_d[:, j_cell])] = 0.0
            penalty[j_cell] = 0.0
            neighbor_cost[i, :] += penalty * w
            orig_angle = orig_angles[i, j]
            if np.isnan(orig_angle):
                continue
            for k in range(m):
                diff = grid_centers[j_cell] - grid_centers[k]
                if np.linalg.norm(diff) < 1e-12:
                    orientation_cost[i, k] += w
                else:
                    angle_err = abs(orig_angle - np.arctan2(diff[1], diff[0]))
                    angle_err = min(angle_err, 2 * np.pi - angle_err)
                    orientation_cost[i, k] += (angle_err / np.pi) * w

    tile_to_region = np.full(m, -1, dtype=int)
    for r in range(n):
        tile_to_region[assignment[r]] = r
    reverse_cost = np.zeros((n, m))
    for k in range(m):
        for t_nb in np.flatnonzero(tile_adj_d[k]):
            j_region = tile_to_region[t_nb]
            if j_region < 0:
                continue
            non_adj = adjacency[:, j_region] <= 0
            non_adj[j_region] = False
            reverse_cost[non_adj, k] += 1.0

    max_oc = np.max(orientation_cost)
    if max_oc > 0:
        orientation_cost /= max_oc
    return origin_w * origin_cost + neighbor_w * (neighbor_cost + reverse_cost) + topology_w * orientation_cost


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vectorized_cost_equals_loop(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    grid_centers, tile_adj, vertex_adj = _square_grid(7)
    n = 30
    centroids = rng.uniform(0, 6, size=(n, 2))
    adjacency = np.triu(rng.uniform(size=(n, n)) < 0.15, 1).astype(float)
    adjacency = adjacency * rng.uniform(0.5, 2.0, size=(n, n))
    adjacency = np.maximum(adjacency, adjacency.T)
    source_indices = rng.integers(0, 22, size=n)
    weights = (1.0, 0.3, 0.5)

    captured = []
    real = _placement.linear_sum_assignment

    def spy(cost):
        captured.append(np.array(cost))
        return real(cost)

    monkeypatch.setattr(_placement, "linear_sum_assignment", spy)
    assign_to_grid_hungarian(
        centroids,
        grid_centers,
        adjacency,
        tile_adj,
        vertex_adj,
        origin_weight=weights[0],
        neighbor_weight=weights[1],
        topology_weight=weights[2],
        source_indices=source_indices,
    )
    initial = real(captured[0])[1]
    expected = _reference_combined(
        initial, centroids, grid_centers, adjacency, tile_adj, vertex_adj, source_indices, weights
    )
    np.testing.assert_array_equal(captured[1], expected)
