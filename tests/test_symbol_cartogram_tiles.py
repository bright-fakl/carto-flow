"""Tests for TileLayout (symbol_cartogram.layouts.tiles)."""

from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Point, box

from carto_flow.data import load_us_census
from carto_flow.symbol_cartogram import SquareTiling, TileLayout, TileLayoutOptions, create_symbol_cartogram
from carto_flow.symbol_cartogram.layouts import TileLayoutResult, get_layout, group_by_layouts
from carto_flow.symbol_cartogram.layouts.tiles._allocation import allocate_tiles, allocation_size, land_parts
from carto_flow.symbol_cartogram.layouts.tiles._assignment import neighbor_lists, split_groups
from carto_flow.symbol_cartogram.layouts.tiles._descent import contact_descent


def _grid(nx: int, ny: int, x0: float = 0.0, y0: float = 0.0) -> list:
    return [box(x0 + i, y0 + j, x0 + i + 1, y0 + j + 1) for j in range(ny) for i in range(nx)]


def _gdf(geoms, **cols) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(cols, geometry=geoms)


def _tiles_per_region(result, n: int) -> np.ndarray:
    return np.bincount(result.layout_result.source_indices, minlength=n)


def _components(tiles, nbrs) -> int:
    from carto_flow.symbol_cartogram.layouts.tiles._assignment import components_of

    return len(components_of(list(tiles), nbrs))


def test_registered_and_supports_group_by():
    assert isinstance(get_layout("tiles"), TileLayout)
    assert "tiles" in group_by_layouts()


def test_option_validation():
    with pytest.raises(ValueError, match="compactness"):
        TileLayout(compactness=1.5)
    with pytest.raises(ValueError, match="components"):
        TileLayout(components="joined")
    with pytest.raises(TypeError):
        TileLayout(not_an_option=1)
    assert TileLayoutOptions().pre_morph is False


def test_allocation_size_geometric_scale():
    assert allocation_size(10, 100, 1.0) == 10
    assert allocation_size(10, 100, 0.0) == 100
    assert allocation_size(10, 100, 0.5) == round(10 * 10**0.5)
    assert allocation_size(10, 5, 0.3) == 5


def test_land_parts_merge_multipart_region_and_split_islands():
    a, b, c = box(0, 0, 1, 1), box(1, 0, 2, 1), box(10, 0, 11, 1)
    parts, labels = land_parts([a, b, c])
    assert len(parts) == 2
    assert labels[0] == labels[1] != labels[2]
    parts, labels = land_parts([a, MultiPolygon([b, c.buffer(0)])])
    assert len(parts) == 1


def _lattice(nx=14, ny=5):
    tr = SquareTiling().generate(bounds=(0, 0, nx, ny), tile_size=1.0)
    return tr, np.asarray(tr.centers)


@pytest.mark.parametrize("components", ["separate", "spaced"])
def test_allocation_parts_and_gap(components):
    tr, centers = _lattice()
    left = box(0, 0, 3, 3)
    right = box(9, 0, 11, 2)
    alloc = allocate_tiles([left, right], np.array([9, 4]), centers, tr.adjacency, 13, components)
    assert len(alloc.tiles) == 13
    part = alloc.part_of_tile
    assert (part[alloc.tiles] >= 0).all()
    assert (part >= 0).sum() == 13
    assert (part == 0).sum() == 9 and (part == 1).sum() == 4
    nbrs = neighbor_lists(tr.adjacency)
    for p in (0, 1):
        assert _components(np.flatnonzero(part == p), nbrs) == 1
    if components == "spaced":
        first = set(np.flatnonzero(part == 0).tolist())
        assert not any(u in first for t in np.flatnonzero(part == 1) for u in nbrs[t])


def test_exact_tile_counts_and_part_membership():
    # two land masses; regions of the right one must stay right
    geoms = _grid(4, 3) + _grid(2, 2, x0=8.0)
    gdf = _gdf(geoms, n=[2] * 12 + [3] * 4)
    for c in (0.5, 1.0):
        r = create_symbol_cartogram(gdf, tile_count="n", layout=TileLayout(compactness=c), show_progress=False)
        lr = r.layout_result
        assert isinstance(lr, TileLayoutResult)
        assert lr.layout_type == "tiles"
        assert (_tiles_per_region(r, 16) == gdf["n"].to_numpy()).all()
        pos = np.array([t.position for t in lr.transforms])
        right = np.isin(lr.source_indices, np.arange(12, 16))
        assert pos[right, 0].min() > pos[~right, 0].max()
        assert lr.metrics.algorithm.regions_correct == 16
        if c < 1:
            assert lr.metrics.algorithm.n_split_groups == 0


def test_deterministic():
    gdf = _gdf(_grid(5, 4), v=np.arange(20))
    runs = [create_symbol_cartogram(gdf, layout=TileLayout(compactness=0.6), show_progress=False) for _ in range(2)]
    a, b = (r.layout_result for r in runs)
    assert np.array_equal(a.source_indices, b.source_indices)
    assert np.array_equal([t.position for t in a.transforms], [t.position for t in b.transforms])


def test_plain_one_tile_per_region_is_near_geometry():
    gdf = _gdf(_grid(5, 4))
    r = create_symbol_cartogram(gdf, layout=TileLayout(compactness=0.8), show_progress=False)
    lr = r.layout_result
    assert sorted(lr.source_indices.tolist()) == list(range(20))
    pos = np.array([t.position for t in lr.transforms])
    cent = np.array([g.centroid.coords[0] for g in gdf.geometry])[lr.source_indices]
    tile = lr.metrics.algorithm.tile_size
    assert np.hypot(*(pos - cent).T).mean() < 1.5 * tile


def test_group_by_groups_stay_contiguous():
    geoms = _grid(6, 6)
    groups = [f"g{(i // 2) % 3 + 3 * (j // 3)}" for j in range(6) for i in range(6)]
    gdf = _gdf(geoms, g=groups)
    r = create_symbol_cartogram(gdf, group_by="g", layout=TileLayout(compactness=0.6), show_progress=False)
    lr = r.layout_result
    assert lr.metrics.algorithm.n_split_groups == 0
    assert len(lr.transforms) == 36
    assert len(lr.regions_gdf) == 6
    assert (lr.regions_gdf["tile_count"] == lr.regions_gdf["target_count"]).all()


def test_multipart_region_splits_capacity_by_part_area():
    big = box(0, 0, 4, 3)
    remote = box(12, 0, 14, 2)
    geoms = [MultiPolygon([big, remote]), box(4, 0, 6, 3)]
    gdf = _gdf(geoms, n=[10, 4])
    r = create_symbol_cartogram(gdf, tile_count="n", layout=TileLayout(compactness=1.0), show_progress=False)
    lr = r.layout_result
    assert (_tiles_per_region(r, 2) == [10, 4]).all()
    pos = np.array([t.position for t in lr.transforms])
    mine = lr.source_indices == 0
    assert (pos[mine, 0] > pos[~mine, 0].max() + 2).any()  # some tiles of region 0 sit on the remote part


def test_one_tile_multipart_region_keeps_main_part():
    geoms = [*_grid(3, 3), MultiPolygon([box(3, 0, 4, 3), Point(30, 1).buffer(0.4)])]
    r = create_symbol_cartogram(_gdf(geoms), layout=TileLayout(compactness=0.7), show_progress=False)
    lr = r.layout_result
    pos = np.array([t.position for t in lr.transforms])
    assert pos[lr.source_indices == 9][:, 0].max() < 8


def test_contact_descent_protect_never_loses_contacts():
    tr = SquareTiling().generate(bounds=(0, 0, 4, 1), tile_size=1.0)
    centers = np.asarray(tr.centers)
    nbrs = neighbor_lists(tr.adjacency)
    row = np.flatnonzero(np.isclose(centers[:, 1], centers[0, 1]))
    order = row[np.argsort(centers[row, 0])][:4]
    allowed = order  # a row of four tiles
    tile_of = order[[0, 2, 1, 3]]  # regions 0..3 placed in scrambled order
    targets = centers[order[[0, 1, 2, 3]]]
    pairs = [(0, 1), (1, 2), (2, 3)]
    for protect in (False, True):
        out = contact_descent(
            tile_of, allowed, centers, nbrs, targets, pairs, 1.0, protect_contacts=protect, max_move=3.0
        )
        assert sorted(out.tolist()) == sorted(tile_of.tolist())
        touching = sum(out[b] in nbrs[out[a]] for a, b in pairs)
        before = sum(tile_of[b] in nbrs[tile_of[a]] for a, b in pairs)
        assert touching >= before
    free = contact_descent(tile_of, allowed, centers, nbrs, targets, pairs, 1.0, max_move=3.0)
    assert all(free[b] in nbrs[free[a]] for a, b in pairs)


def test_pre_morph_runs_on_small_fixture():
    geoms = [box(0, 0, 2, 2), box(2, 0, 3, 1), box(2, 1, 3, 2), box(0, 2, 1, 3), box(1, 2, 3, 3)]
    gdf = _gdf(geoms, n=[4, 1, 1, 1, 2])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = create_symbol_cartogram(
            gdf, tile_count="n", layout=TileLayout(compactness=0.7, pre_morph=True), show_progress=False
        )
    lr = r.layout_result
    assert (_tiles_per_region(r, 5) == gdf["n"].to_numpy()).all()
    assert lr.metrics.algorithm.pre_morphed in (True, False)


def test_split_groups_counts_pieces():
    tr = SquareTiling().generate(bounds=(0, 0, 4, 1), tile_size=1.0)
    nbrs = neighbor_lists(tr.adjacency)
    centers = np.asarray(tr.centers)
    row = np.flatnonzero(np.isclose(centers[:, 1], centers[0, 1]))
    order = row[np.argsort(centers[row, 0])][:4]
    label = np.full(len(centers), -1)
    label[order[[0, 1, 3]]] = 0
    assert split_groups(label, 1, nbrs) == (1, 1)


def test_states_smoke():
    states = load_us_census(population=True, contiguous_only=True).reset_index(drop=True)
    r = create_symbol_cartogram(states, layout=TileLayout(compactness=0.5), show_progress=False)
    lr = r.layout_result
    assert len(lr.transforms) == len(states)
    assert sorted(lr.source_indices.tolist()) == list(range(len(states)))
    assert lr.metrics.algorithm.n_allowed < lr.metrics.algorithm.n_tiles
