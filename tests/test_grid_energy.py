"""GridEnergy: defect count, monotone descent, margin insensitivity, determinism."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.sparse import csr_array

from carto_flow.symbol_cartogram.layouts.grid._energy import GridEnergy


def _grid(width: int, height: int, x0: int = 0, y0: int = 0):
    """Centers and edge adjacency of a square tiling with corners (x0, y0) and (x0 + width - 1, y0 + height - 1)."""
    centers = np.array([(x, y) for y in range(y0, y0 + height) for x in range(x0, x0 + width)], dtype=float)
    diff = np.abs(centers[:, None, :] - centers[None, :, :]).sum(axis=2)
    return centers, csr_array(diff == 1)


def _block(width: int, height: int, jitter: float = 0.0, seed: int = 0):
    """Centroids of a width x height block of square regions and their rook adjacency."""
    rng = np.random.default_rng(seed)
    cells = np.array([(x, y) for y in range(height) for x in range(width)], dtype=float)
    centroids = cells + rng.uniform(-jitter, jitter, cells.shape)
    adj = (np.abs(cells[:, None, :] - cells[None, :, :]).sum(axis=2) == 1).astype(float)
    return centroids, adj


def _index(centers, points):
    return np.array([int(np.argmin(np.abs(centers - p).sum(axis=1))) for p in points])


def test_defects_count_pieces_and_holes():
    centers, edge = _grid(7, 7)
    centroids, adj = _block(3, 3)
    model = GridEnergy(centroids + 2, centers, adj, edge)
    block = _index(centers, centroids + 2)
    assert model.defects(block) == 0
    # Center tile free: one enclosed hole
    ring = block.copy()
    ring[4] = _index(centers, [(0, 0)])[0]
    assert model.defects(ring) == 2  # the hole, and the region at (0, 0) apart from the rest
    # Corner region moved away: two pieces
    apart = block.copy()
    apart[0] = _index(centers, [(6, 6)])[0]
    assert model.defects(apart) == 1


def test_separate_groups_must_stay_apart():
    centers, edge = _grid(7, 3)
    centroids = np.array([(1.0, 1.0), (2.0, 1.0), (5.0, 1.0)])
    adj = np.zeros((3, 3))
    adj[0, 1] = adj[1, 0] = 1.0
    model = GridEnergy(centroids, centers, adj, edge)
    assert model.defects(_index(centers, centroids)) == 0
    assert model.defects(_index(centers, [(1, 1), (2, 1), (3, 1)])) == 1  # island merged into the mainland


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_descent_lowers_energy_monotonically_and_ends_connected(seed):
    centers, edge = _grid(10, 10)
    centroids, adj = _block(5, 5, jitter=0.45, seed=seed)
    centroids = centroids * 1.3 + 2.0  # spread out, so a nearest-tile start leaves gaps
    model = GridEnergy(centroids, centers, adj, edge)
    rng = np.random.default_rng(seed)
    start = rng.permutation(len(centers))[: len(centroids)]
    trace: list[float] = []
    out = model.descend(start, trace=trace)
    assert len(trace) > 1
    assert all(b < a for a, b in itertools.pairwise(trace))
    assert trace[-1] == pytest.approx(model.energy(out))
    assert model.defects(out) == 0
    assert len(set(out.tolist())) == len(out)


def test_margin_tiles_do_not_change_the_result():
    centroids, adj = _block(4, 4, jitter=0.3, seed=3)
    centroids = centroids * 1.2 + 3.0
    small_c, small_e = _grid(10, 10)
    big_c, big_e = _grid(16, 16, x0=-3, y0=-3)
    results = []
    for centers, edge in ((small_c, small_e), (big_c, big_e)):
        model = GridEnergy(centroids, centers, adj, edge)
        start = _index(centers, np.round(centroids))
        results.append(centers[model.descend(start)])
    np.testing.assert_array_equal(results[0], results[1])


def test_descent_is_deterministic():
    centers, edge = _grid(9, 9)
    centroids, adj = _block(4, 4, jitter=0.4, seed=5)
    centroids = centroids * 1.5 + 1.0
    model = GridEnergy(centroids, centers, adj, edge)
    start = np.arange(len(centroids)) * 5
    a = model.descend(start)
    b = GridEnergy(centroids, centers, adj, edge).descend(start)
    np.testing.assert_array_equal(a, b)


def test_single_region():
    centers, edge = _grid(3, 3)
    model = GridEnergy(np.array([[2.2, 1.9]]), centers, np.zeros((1, 1)), edge)
    out = model.descend(np.array([0]))
    np.testing.assert_array_equal(centers[out], [[2.0, 2.0]])


@pytest.mark.parametrize("tiling", ["hexagon", "square"])
def test_layout_states_connected_and_deterministic(tiling):
    from scipy.sparse.csgraph import connected_components

    from carto_flow.data import load_us_census
    from carto_flow.symbol_cartogram import create_layout
    from carto_flow.symbol_cartogram.layouts.grid import GridBasedLayout

    gdf = load_us_census(population=True)
    runs = [create_layout(gdf, layout=GridBasedLayout(tiling=tiling), show_progress=False) for _ in range(2)]
    a = np.asarray(runs[0].assignments)
    np.testing.assert_array_equal(a, np.asarray(runs[1].assignments))
    adj = runs[0].tiling_result.adjacency.tocsr()
    assert connected_components(adj[a][:, a], directed=False)[0] == 1


def test_refine_assignment_still_available():
    from carto_flow.data import load_us_census
    from carto_flow.symbol_cartogram import create_layout
    from carto_flow.symbol_cartogram.layouts.grid import GridBasedLayout

    gdf = load_us_census(population=True)
    res = create_layout(gdf, layout=GridBasedLayout(assignment="refine"), show_progress=False)
    assert len(set(np.asarray(res.assignments).tolist())) == len(gdf)
    with pytest.raises(ValueError, match="assignment"):
        GridBasedLayout(assignment="other")
