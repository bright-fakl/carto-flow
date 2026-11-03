"""GridEnergy: defect count, monotone descent, margin insensitivity, determinism."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import shapely
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


def _pockets(centers_adj, chosen):
    adj = (centers_adj.toarray() != 0).astype(float)
    kk = int(adj.sum(axis=1).max()) // 2 + 1
    return int((~chosen & (adj @ chosen.astype(float) >= kk)).sum())


def test_packed_candidates_have_no_pockets_and_grow_with_lower_compactness():
    from carto_flow.symbol_cartogram.layouts.grid._compact import candidate_tiles

    centers, edge = _grid(12, 9)
    centroids, _ = _block(5, 4, jitter=0.4, seed=2)
    centroids = centroids * 2.0 + 1.0
    sizes = []
    for c in (1.0, 0.5, 0.0):
        tiles = candidate_tiles(centroids, centers, edge, c)
        sizes.append(len(tiles))
        if c == 1.0:
            chosen = np.zeros(len(centers), dtype=bool)
            chosen[tiles] = True
            assert _pockets(edge, chosen) == 0
    assert sizes == sorted(sizes)
    assert sizes[0] == len(centroids)
    assert sizes[-1] == len(centers)


@pytest.mark.parametrize("tiling", ["hexagon", "square", "triangle", "tree"])
def test_layout_full_compactness_has_no_pockets(tiling):
    from carto_flow.data import load_us_census
    from carto_flow.symbol_cartogram import create_layout
    from carto_flow.symbol_cartogram.layouts.grid import GridBasedLayout
    from carto_flow.symbol_cartogram.tiling import IsohedralTiling

    if tiling == "tree":
        tiling = IsohedralTiling.from_preset("tree")

    gdf = load_us_census(population=True)
    res = create_layout(gdf, layout=GridBasedLayout(tiling=tiling, compactness=1.0), show_progress=False)
    tr = res.tiling_result
    chosen = np.zeros(tr.n_tiles, dtype=bool)
    chosen[np.asarray(res.assignments)] = True
    assert _pockets(tr.adjacency, chosen) == 0


def test_compactness_above_one_rejected_for_energy():
    from carto_flow.symbol_cartogram.layouts.grid import GridBasedLayout

    with pytest.raises(ValueError, match="compactness"):
        GridBasedLayout(compactness=1.5)
    GridBasedLayout(compactness=1.5, assignment="refine")


def _states_with_islands():
    """The 51 states in Albers coordinates; Alaska and Hawaii lie apart from the contiguous states."""
    from carto_flow.data import load_us_states

    gdf = load_us_states().to_crs("ESRI:102003")
    return gdf[[g.area > 0 for g in gdf.geometry]].reset_index(drop=True)


def _groups(res, gdf):
    """Connected tile group of each region and the group sizes."""
    from scipy.sparse.csgraph import connected_components

    a = np.asarray(res.assignments)
    adj = res.tiling_result.adjacency.tocsr()
    lab = connected_components(adj[a][:, a], directed=False)[1]
    return dict(zip(gdf["postal"], lab, strict=True)), np.bincount(lab)


@pytest.mark.parametrize("tiling", ["hexagon", "square"])
def test_components_modes_place_islands(tiling):
    from scipy.sparse.csgraph import dijkstra

    from carto_flow.symbol_cartogram import create_layout
    from carto_flow.symbol_cartogram.layouts.grid import GridBasedLayout

    gdf = _states_with_islands()
    names = list(gdf["postal"])
    out = {}
    for comp in ("separate", "connected", "spaced"):
        res = create_layout(
            gdf, layout=GridBasedLayout(tiling=tiling, compactness=1.0, components=comp), show_progress=False
        )
        a = np.asarray(res.assignments)
        assert len(set(a.tolist())) == len(gdf)  # n tiles at compactness 1
        adj = res.tiling_result.adjacency.tocsr() != 0
        main = a[[i for i, nm in enumerate(names) if nm not in ("AK", "HI")]]
        steps = dijkstra(adj, unweighted=True, indices=main, min_only=True)
        out[comp] = (_groups(res, gdf), {nm: int(steps[a[names.index(nm)]]) for nm in ("AK", "HI")})
    (lab, sizes), steps = out["separate"]
    assert len(sizes) == 3
    assert sizes[lab["AK"]] == 1 and sizes[lab["HI"]] == 1 and lab["AK"] != lab["HI"]
    assert min(steps.values()) > 2
    assert len(out["connected"][0][1]) == 1
    assert out["connected"][1] == {"AK": 1, "HI": 1}
    assert len(out["spaced"][0][1]) == 3
    assert out["spaced"][1] == {"AK": 2, "HI": 2}


def test_single_tile_part_sits_next_to_its_islands():
    from carto_flow.symbol_cartogram import create_layout
    from carto_flow.symbol_cartogram.layouts.grid import GridBasedLayout

    gdf = _states_with_islands()
    res = create_layout(gdf, layout=GridBasedLayout(tiling="hexagon", compactness=1.0), show_progress=False)
    tr = res.tiling_result
    hi = int(np.asarray(res.assignments)[list(gdf["postal"]).index("HI")])
    pt = shapely.Point(tr.centers[hi])
    assert pt.distance(gdf.geometry.iloc[list(gdf["postal"]).index("HI")]) < tr.tile_size


def test_set_size_is_n_at_full_compactness_and_grows():
    from carto_flow.data import load_us_census
    from carto_flow.symbol_cartogram import create_layout
    from carto_flow.symbol_cartogram.layouts.grid import GridBasedLayout
    from carto_flow.symbol_cartogram.layouts.grid._compact import candidate_tiles

    gdf = load_us_census(population=True)
    res = create_layout(gdf, layout=GridBasedLayout(compactness=1.0), show_progress=False)
    assert len(set(np.asarray(res.assignments).tolist())) == len(gdf)
    tr = res.tiling_result
    cent = np.array([[g.centroid.x, g.centroid.y] for g in gdf.geometry])
    sizes = [len(candidate_tiles(cent, tr.centers, tr.adjacency, c, list(gdf.geometry))) for c in (1.0, 0.5, 0.0)]
    assert sizes[0] == len(gdf) and sizes == sorted(sizes) and sizes[-1] == tr.n_tiles


def test_components_validated():
    from carto_flow.symbol_cartogram.layouts.grid import GridBasedLayout

    with pytest.raises(ValueError, match="components"):
        GridBasedLayout(components="joined")
    GridBasedLayout(components="connected")


@pytest.mark.parametrize("compactness", [0.0, 0.5, 1.0])
def test_allocated_tiles_contain_assigned_tiles(compactness):
    import matplotlib

    matplotlib.use("Agg")
    from carto_flow.data import load_us_census
    from carto_flow.symbol_cartogram import create_layout
    from carto_flow.symbol_cartogram.layouts.grid import GridBasedLayout, plot_tile_sets

    gdf = load_us_census(population=True)
    res = create_layout(gdf, layout=GridBasedLayout(compactness=compactness), show_progress=False)
    allowed = set(res.metrics.algorithm.allowed_tiles.tolist())
    assert set(np.asarray(res.assignments).tolist()) <= allowed
    assert len(allowed) == len(gdf) + round((1 - compactness) * (res.tiling_result.n_tiles - len(gdf)))
    plot_tile_sets(res, labels=gdf["State Abbreviation"].tolist())


@pytest.mark.parametrize("preset", ["regular_hexagon", "square"])
def test_allocated_set_at_full_compactness_is_one_piece(preset):
    from scipy.sparse.csgraph import connected_components

    from carto_flow.data import load_us_census
    from carto_flow.symbol_cartogram import create_layout
    from carto_flow.symbol_cartogram.layouts.grid import GridBasedLayout
    from carto_flow.symbol_cartogram.tiling import IsohedralTiling

    gdf = load_us_census(population=True)
    tiling = IsohedralTiling.from_preset(preset).with_offset((0.0, 0.0))
    res = create_layout(gdf, layout=GridBasedLayout(tiling=tiling, compactness=1.0), show_progress=False)
    allowed = res.metrics.algorithm.allowed_tiles
    adj = res.tiling_result.adjacency.tocsr()
    assert connected_components(adj[allowed][:, allowed], directed=False)[0] == 1
