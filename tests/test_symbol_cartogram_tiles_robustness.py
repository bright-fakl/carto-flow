"""Row-order invariance, contact protection, size guard and metrics of TileLayout."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from carto_flow.data import load_us_census, load_world
from carto_flow.geo_utils.adjacency import find_bordering_pairs
from carto_flow.symbol_cartogram import SquareTiling, TileLayout, create_symbol_cartogram


def _grid(nx: int, ny: int) -> list:
    return [box(i, j, i + 1, j + 1) for j in range(ny) for i in range(nx)]


def _by_row(result, frame_rows: np.ndarray) -> dict[int, tuple]:
    """Tile centers of each input row, with row ``i`` of the run labeled ``frame_rows[i]``."""
    lr = result.layout_result
    pos = np.round(np.array([t.position for t in lr.transforms]), 3)
    return {
        int(frame_rows[g]): tuple(sorted(map(tuple, pos[lr.source_indices == g].tolist())))
        for g in sorted(set(lr.source_indices.tolist()))
    }


@pytest.fixture(scope="module")
def europe_subset():
    w = load_world()
    g = w[w.continent == "Europe"].to_crs("EPSG:3035")
    keep = ["France", "Germany", "Spain", "Italy", "Poland", "Norway", "Sweden", "Finland", "United Kingdom"]
    keep += ["Ireland", "Iceland", "Austria", "Switzerland", "Belgium", "Netherlands", "Denmark", "Portugal"]
    return g[g["name"].isin(keep)].reset_index(drop=True)


@pytest.mark.parametrize(
    "options",
    [{}, {"compactness": 1.0}, {"compactness": 0.3, "components": "spaced"}, {"protect_contacts": True}],
)
def test_result_does_not_depend_on_row_order(europe_subset, options):
    perm = np.random.RandomState(3).permutation(len(europe_subset))
    shuffled = europe_subset.iloc[perm].reset_index(drop=True)
    a = create_symbol_cartogram(europe_subset, layout=TileLayout(**options), show_progress=False)
    b = create_symbol_cartogram(shuffled, layout=TileLayout(**options), show_progress=False)
    assert _by_row(a, np.arange(len(europe_subset))) == _by_row(b, perm)


def test_states_do_not_depend_on_row_order():
    states = load_us_census(population=True, contiguous_only=True).reset_index(drop=True)
    perm = np.random.RandomState(5).permutation(len(states))
    shuffled = states.iloc[perm].reset_index(drop=True)
    a = create_symbol_cartogram(states, layout=TileLayout(compactness=0.5), show_progress=False)
    b = create_symbol_cartogram(shuffled, layout=TileLayout(compactness=0.5), show_progress=False)
    assert _by_row(a, np.arange(len(states))) == _by_row(b, perm)


def test_group_by_does_not_depend_on_row_order():
    groups = [f"g{(i // 2) % 3 + 3 * (j // 3)}" for j in range(6) for i in range(6)]
    gdf = gpd.GeoDataFrame({"g": groups}, geometry=_grid(6, 6))
    perm = np.random.RandomState(1).permutation(36)
    shuffled = gdf.iloc[perm].reset_index(drop=True)
    kwargs = {"group_by": "g", "layout": TileLayout(compactness=0.6), "show_progress": False}
    a = create_symbol_cartogram(gdf, **kwargs)
    b = create_symbol_cartogram(shuffled, **kwargs)
    assert _by_row(a, np.arange(36)) == _by_row(b, perm)


def _true_contacts(result, gdf) -> int:
    lr = result.layout_result
    adj = lr.tiling_result.adjacency.tocsr()
    centers = np.asarray(lr.tiling_result.centers)
    pos = np.array([t.position for t in lr.transforms])
    tile = np.argmin(((pos[:, None] - centers[None]) ** 2).sum(-1), axis=1)
    owner = dict(zip(tile.tolist(), lr.source_indices.tolist(), strict=True))
    touching = set()
    for t, g in owner.items():
        for u in adj.indices[adj.indptr[t] : adj.indptr[t + 1]]:
            h = owner.get(int(u))
            if h is not None and h != g:
                touching.add((min(g, h), max(g, h)))
    return len(touching & {(i, j) for i, j, _ in find_bordering_pairs(list(gdf.geometry))})


@pytest.mark.parametrize("compactness", [0.25, 0.75, 1.0])
def test_protect_contacts_never_has_fewer_true_contacts(europe_subset, compactness):
    counts = [
        _true_contacts(
            create_symbol_cartogram(
                europe_subset,
                layout=TileLayout(compactness=compactness, protect_contacts=protect),
                show_progress=False,
            ),
            europe_subset,
        )
        for protect in (False, True)
    ]
    assert counts[1] >= counts[0]


def test_memory_guard_warns(monkeypatch):
    import carto_flow.symbol_cartogram.layouts.tiles as tiles

    monkeypatch.setattr(tiles, "_MEMORY_WARNING_BYTES", 1.0)
    gdf = gpd.GeoDataFrame({"v": range(9)}, geometry=_grid(3, 3))
    with pytest.warns(UserWarning, match="dense matrices"):
        create_symbol_cartogram(gdf, layout=TileLayout(compactness=0.5), show_progress=False)


def test_metrics_report_the_tiling_name():
    gdf = gpd.GeoDataFrame({"v": range(9)}, geometry=_grid(3, 3))
    for tiling, name in ((SquareTiling(), "SquareTiling"), ("square", "square")):
        r = create_symbol_cartogram(gdf, layout=TileLayout(tiling=tiling), show_progress=False)
        assert r.layout_result.metrics.algorithm.tiling == name
