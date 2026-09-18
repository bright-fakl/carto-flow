"""Tests for MosaicLayout (symbol_cartogram.layouts.mosaic).

Most tests use small synthetic grids of unit squares with ``morph=False`` so
the module runs in a few seconds; one test uses the bundled US states.

What the layout guarantees, and what it only attempts:

* exact tile counts per region — structural (one Hungarian slot per tile), so
  asserted everywhere;
* intra-region contiguity and inter-region adjacency — best effort. The
  iterative repair loop keeps the best-scoring assignment it finds, which is
  not always violation-free. Contiguity is therefore asserted on fixtures
  where the current implementation achieves it, and the known failure on the
  3x3 fixture is captured as a strict xfail.
"""

from __future__ import annotations

from collections import deque

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from carto_flow.geo_utils import find_adjacent_pairs
from carto_flow.symbol_cartogram import (
    HungarianOptions,
    MosaicLayout,
    MosaicLayoutOptions,
    create_layout,
    create_symbol_cartogram,
)
from carto_flow.symbol_cartogram.layouts import (
    LayoutResult,
    MosaicMetrics,
    prepare_layout_data,
)
from carto_flow.symbol_cartogram.layouts.layout_result import MosaicLayoutResult

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

COUNTS_3X3 = [4, 5, 3, 6, 4, 5, 3, 4, 6]  # 40 tiles
COUNTS_4X3 = [4, 3, 5, 4, 6, 3, 4, 5, 3, 4, 3, 5]  # 49 tiles


def grid_gdf(cols: int, rows: int, counts: list[int], x0: float = 0.0) -> gpd.GeoDataFrame:
    """Grid of unit squares with a ``tiles`` column."""
    geoms = [box(x0 + c, r, x0 + c + 1, r + 1) for r in range(rows) for c in range(cols)]
    assert len(geoms) == len(counts)
    return gpd.GeoDataFrame({"tiles": counts}, geometry=geoms)


def compute(gdf: gpd.GeoDataFrame, *, pre_scale: bool = False, **kwargs) -> MosaicLayoutResult:
    """Run MosaicLayout on *gdf* using its ``tiles`` column."""
    kwargs.setdefault("morph", False)
    data = prepare_layout_data(gdf, tile_count="tiles", pre_scale=pre_scale)
    result = MosaicLayout(**kwargs).compute(data, show_progress=False)
    assert isinstance(result, MosaicLayoutResult)
    return result


def tile_adjacency(result: MosaicLayoutResult) -> dict[int, set[int]]:
    """Adjacency between the assigned tiles, keyed by position in ``assignments``."""
    polys = [result.tiling_result.polygons[int(t)] for t in result.assignments]
    adjacency: dict[int, set[int]] = {i: set() for i in range(len(polys))}
    for i, j, _ in find_adjacent_pairs(polys):
        adjacency[i].add(j)
        adjacency[j].add(i)
    return adjacency


def connected_size(nodes: list[int], adjacency: dict[int, set[int]]) -> int:
    """Size of the connected component of *nodes* reachable from ``nodes[0]``."""
    remaining = set(nodes)
    seen = {nodes[0]}
    queue = deque([nodes[0]])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor in remaining and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return len(seen)


def tiles_by_key(result: MosaicLayoutResult, keys: list) -> dict:
    """Group tile positions by ``keys[geometry index]``."""
    grouped: dict = {}
    for slot, geom in enumerate(np.asarray(result.source_indices).tolist()):
        grouped.setdefault(keys[geom], []).append(slot)
    return grouped


def non_contiguous_regions(result: MosaicLayoutResult) -> list[int]:
    """Geometry indices whose tiles do not form a single connected component."""
    adjacency = tile_adjacency(result)
    grouped = tiles_by_key(result, list(range(len(result.counts))))
    return [g for g, tiles in grouped.items() if connected_size(tiles, adjacency) != len(tiles)]


def adjacency_preserved_fraction(result: MosaicLayoutResult, gdf: gpd.GeoDataFrame) -> float:
    """Fraction of input neighbour pairs whose tile sets touch in the output."""
    adjacency = tile_adjacency(result)
    grouped = tiles_by_key(result, list(range(len(gdf))))
    pairs = find_adjacent_pairs(list(gdf.geometry))
    if not pairs:
        return 1.0
    kept = 0
    for i, j, _ in pairs:
        tiles_i, tiles_j = set(grouped.get(i, [])), set(grouped.get(j, []))
        if tiles_i and tiles_j and any(nb in tiles_j for t in tiles_i for nb in adjacency[t]):
            kept += 1
    return kept / len(pairs)


def tile_counts_per_geometry(result: MosaicLayoutResult) -> np.ndarray:
    """Number of tiles assigned to each geometry."""
    return np.bincount(np.asarray(result.source_indices), minlength=len(result.counts))


# ---------------------------------------------------------------------------
# 1. Exact tile counts
# ---------------------------------------------------------------------------


class TestTileCounts:
    """Every region receives exactly its requested number of tiles."""

    @pytest.mark.parametrize("tiling", ["hexagon", "square"])
    @pytest.mark.parametrize(("cols", "rows", "counts"), [(3, 3, COUNTS_3X3), (4, 3, COUNTS_4X3)])
    def test_exact_counts(self, tiling, cols, rows, counts):
        gdf = grid_gdf(cols, rows, counts)
        result = compute(gdf, tiling=tiling)

        metrics = result.metrics.algorithm
        assert isinstance(metrics, MosaicMetrics)
        assert metrics.regions_correct == metrics.regions_total == len(gdf)
        assert result.metrics.converged
        np.testing.assert_array_equal(tile_counts_per_geometry(result), counts)
        assert len(result.transforms) == sum(counts)
        assert metrics.tiling == tiling
        assert metrics.tile_size > 0

    def test_counts_with_morph(self):
        """The flow pre-morph does not change the exact-count guarantee."""
        gdf = grid_gdf(3, 3, COUNTS_3X3)
        result = compute(gdf, morph=True)

        np.testing.assert_array_equal(tile_counts_per_geometry(result), COUNTS_3X3)

    def test_regions_gdf_matches_targets(self):
        """``regions_gdf`` reports the requested count per geometry."""
        gdf = grid_gdf(4, 3, COUNTS_4X3)
        result = compute(gdf)

        assert list(result.regions_gdf["target_count"]) == COUNTS_4X3
        assert len(result.tiles_gdf) == len(result.transforms)


# ---------------------------------------------------------------------------
# 2. Contiguity
# ---------------------------------------------------------------------------


class TestContiguity:
    """Each region's tiles should form a single connected component."""

    @pytest.mark.parametrize("tiling", ["hexagon", "square"])
    def test_all_regions_contiguous_4x3(self, tiling):
        """On the 4x3 fixture the current implementation reaches full contiguity."""
        gdf = grid_gdf(4, 3, COUNTS_4X3)
        result = compute(gdf, tiling=tiling)

        assert non_contiguous_regions(result) == []

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Contiguity is best effort: on the 3x3 fixture (hexagon, morph=False) "
            "2 of 9 regions come out split. See issue #20 follow-ups."
        ),
    )
    def test_all_regions_contiguous_3x3(self):
        gdf = grid_gdf(3, 3, COUNTS_3X3)
        result = compute(gdf)

        assert non_contiguous_regions(result) == []


# ---------------------------------------------------------------------------
# 3. Adjacency preservation
# ---------------------------------------------------------------------------


class TestAdjacencyPreservation:
    """Input neighbours should stay neighbours in the tiling."""

    @pytest.mark.parametrize("tiling", ["hexagon", "square"])
    def test_most_neighbour_pairs_preserved(self, tiling):
        """Measured 0.79 (hexagon) / 0.86 (square) on this fixture; assert 0.70."""
        gdf = grid_gdf(4, 3, COUNTS_4X3)
        result = compute(gdf, tiling=tiling)

        assert adjacency_preserved_fraction(result, gdf) >= 0.70


# ---------------------------------------------------------------------------
# 4. group_by
# ---------------------------------------------------------------------------


def group_fixture() -> tuple[gpd.GeoDataFrame, list[str], list[int]]:
    """4x2 mainland plus a 2x1 island; group ``B`` spans both components."""
    geoms = [box(c, r, c + 1, r + 1) for r in range(2) for c in range(4)]
    geoms += [box(6, 0, 7, 1), box(7, 0, 8, 1)]
    groups = ["A", "A", "B", "B"] * 2 + ["B", "B"]
    components = [0] * 8 + [1] * 2
    return gpd.GeoDataFrame({"grp": groups}, geometry=geoms), groups, components


class TestGroupBy:
    """Groups are split at component boundaries and stay contiguous per part."""

    @pytest.mark.parametrize("tiling", ["hexagon", "square"])
    def test_group_parts_get_the_right_tile_counts(self, tiling):
        gdf, groups, components = group_fixture()
        data = prepare_layout_data(gdf, group_by="grp")
        result = MosaicLayout(morph=False, tiling=tiling).compute(data, show_progress=False)

        keys = list(zip(groups, components, strict=True))
        grouped = tiles_by_key(result, keys)
        assert {k: len(v) for k, v in grouped.items()} == {
            ("A", 0): 4,
            ("B", 0): 4,
            ("B", 1): 2,
        }
        assert result.metrics.algorithm.n_components == 2

    def test_group_parts_are_contiguous(self):
        """Each (group, component) part forms one connected tile block."""
        gdf, groups, components = group_fixture()
        data = prepare_layout_data(gdf, group_by="grp")
        result = MosaicLayout(morph=False).compute(data, show_progress=False)

        adjacency = tile_adjacency(result)
        grouped = tiles_by_key(result, list(zip(groups, components, strict=True)))
        for key, tiles in grouped.items():
            assert connected_size(tiles, adjacency) == len(tiles), key

    def test_regions_gdf_has_one_row_per_group(self):
        gdf, _, _ = group_fixture()
        data = prepare_layout_data(gdf, group_by="grp")
        result = MosaicLayout(morph=False).compute(data, show_progress=False)

        assert len(result.regions_gdf) == 2
        assert list(result.regions_gdf["tile_count"]) == [4, 6]
        assert list(result.regions_gdf["target_count"]) == [4, 6]


# ---------------------------------------------------------------------------
# 5. Islands
# ---------------------------------------------------------------------------


class TestIslands:
    """A detached geometry is solved as its own component."""

    def test_detached_square_gets_its_own_tiles(self):
        gdf = grid_gdf(3, 3, COUNTS_3X3)
        island = gpd.GeoDataFrame({"tiles": [5]}, geometry=[box(6, 1, 7, 2)])
        gdf = gpd.GeoDataFrame(gpd.pd.concat([gdf, island], ignore_index=True), geometry="geometry")

        result = compute(gdf)

        assert result.metrics.algorithm.n_components == 2
        counts = tile_counts_per_geometry(result)
        np.testing.assert_array_equal(counts, [*COUNTS_3X3, 5])

        # The island's tiles are contiguous and disjoint from the mainland's.
        adjacency = tile_adjacency(result)
        grouped = tiles_by_key(result, list(range(len(gdf))))
        island_tiles = grouped[9]
        assert connected_size(island_tiles, adjacency) == 5
        mainland = {t for g, tiles in grouped.items() if g != 9 for t in tiles}
        assert not any(nb in mainland for t in island_tiles for nb in adjacency[t])


# ---------------------------------------------------------------------------
# 6. pre_scale
# ---------------------------------------------------------------------------


class TestPreScale:
    """``pre_scale`` rescales components before assignment, counts unchanged."""

    def test_pre_scale_preserves_counts(self):
        gdf = grid_gdf(3, 3, COUNTS_3X3)
        island = gpd.GeoDataFrame({"tiles": [5]}, geometry=[box(6, 1, 7, 2)])
        gdf = gpd.GeoDataFrame(gpd.pd.concat([gdf, island], ignore_index=True), geometry="geometry")

        result = compute(gdf, pre_scale=True)

        np.testing.assert_array_equal(tile_counts_per_geometry(result), [*COUNTS_3X3, 5])
        assert result.metrics.algorithm.n_components == 2

    def test_pre_scale_grows_the_under_sized_component(self):
        """The island asks for 5 of 45 tiles, so pre-scaling enlarges it."""
        gdf = grid_gdf(3, 3, COUNTS_3X3)
        island = gpd.GeoDataFrame({"tiles": [5]}, geometry=[box(6, 1, 7, 2)])
        gdf = gpd.GeoDataFrame(gpd.pd.concat([gdf, island], ignore_index=True), geometry="geometry")

        plain = prepare_layout_data(gdf, tile_count="tiles")
        scaled = prepare_layout_data(gdf, tile_count="tiles", pre_scale=True)
        island_area_plain = plain.source_gdf.geometry.iloc[9].area
        island_area_scaled = scaled.source_gdf.geometry.iloc[9].area

        assert island_area_scaled > island_area_plain


# ---------------------------------------------------------------------------
# 7. Options and API plumbing
# ---------------------------------------------------------------------------


class TestOptions:
    """Option validation and the public entry points."""

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"tile_size": 0.0}, "tile_size must be positive"),
            ({"tile_size": -1.0}, "tile_size must be positive"),
            ({"spacing": -0.1}, "spacing must be between 0 and 1"),
            ({"spacing": 1.5}, "spacing must be between 0 and 1"),
            ({"extra_tile_rings": -1}, "extra_tile_rings must be >= 0"),
            ({"min_overlap_frac": 0.0}, "min_overlap_frac must be in"),
            ({"min_overlap_frac": 1.5}, "min_overlap_frac must be in"),
        ],
    )
    def test_invalid_options_raise(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            MosaicLayoutOptions(**kwargs).validate()
        with pytest.raises(ValueError, match=message):
            MosaicLayout(**kwargs)

    def test_hungarian_options_validate(self):
        with pytest.raises(ValueError, match="max_connectivity_iters must be >= 0"):
            HungarianOptions(max_connectivity_iters=-1)

    def test_hungarian_cost_weight_names(self):
        """The cost weights carry their descriptive names."""
        options = HungarianOptions(distance_weight=2.0, outside_penalty=0.5, interior_bonus=0.1)

        assert (options.distance_weight, options.outside_penalty, options.interior_bonus) == (2.0, 0.5, 0.1)

    def test_kwargs_are_applied_to_options(self):
        layout = MosaicLayout(tiling="square", morph=False, tile_size=0.5, spacing=0.1)

        assert layout._options.tiling == "square"
        assert layout._options.tile_size == 0.5
        assert layout._options.spacing == 0.1

    def test_unknown_kwarg_raises(self):
        with pytest.raises(TypeError, match="Unknown option"):
            MosaicLayout(not_an_option=1)

    def test_explicit_tile_size_is_used(self):
        gdf = grid_gdf(3, 3, COUNTS_3X3)
        result = compute(gdf, tile_size=0.3)

        assert result.metrics.algorithm.tile_size == pytest.approx(0.3)
        np.testing.assert_array_equal(tile_counts_per_geometry(result), COUNTS_3X3)

    def test_registry_key_and_create_layout(self):
        gdf = grid_gdf(3, 3, COUNTS_3X3)
        result = create_layout(gdf, layout="mosaic", tile_count="tiles", show_progress=False)

        assert result.layout_type == "mosaic"
        assert len(result.transforms) == sum(COUNTS_3X3)

    def test_create_symbol_cartogram_with_mosaic(self):
        gdf = grid_gdf(3, 3, COUNTS_3X3)
        cartogram = create_symbol_cartogram(
            gdf,
            layout=MosaicLayout(morph=False),
            tile_count="tiles",
            show_progress=False,
        )

        assert len(cartogram.symbols) == sum(COUNTS_3X3)
        assert cartogram.layout_result is not None
        assert cartogram.layout_result.layout_type == "mosaic"


# ---------------------------------------------------------------------------
# 8. Visualization
# ---------------------------------------------------------------------------


class TestPlotting:
    """``plot_tiling`` renders the mosaic result, with and without the pool."""

    @pytest.mark.parametrize("show_pool", [False, True])
    def test_plot_tiling(self, show_pool):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gdf = grid_gdf(3, 3, COUNTS_3X3)
        result = compute(gdf)

        plot = result.plot_tiling(show_pool=show_pool, show_symbols=False)
        assert plot.ax is not None
        if show_pool:
            # Assigned tiles are split across the core/ring collections.
            assert plot.assigned_tiles is None
            assert plot.core_tiles is not None
        else:
            assert plot.assigned_tiles is not None
            assert plot.core_tiles is None
        plt.close("all")


# ---------------------------------------------------------------------------
# 9. Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    """A mosaic result survives a serialize / from_serialized round trip."""

    def test_round_trip(self):
        gdf = grid_gdf(3, 3, COUNTS_3X3)
        result = compute(gdf)

        restored = LayoutResult.from_serialized(result.serialize())

        assert isinstance(restored, MosaicLayoutResult)
        assert restored.layout_type == "mosaic"
        assert len(restored.transforms) == len(result.transforms)
        np.testing.assert_allclose(restored.positions, result.positions)


# ---------------------------------------------------------------------------
# Bundled data
# ---------------------------------------------------------------------------


class TestUsStates:
    """One end-to-end run on the bundled US states (about 150 tiles, ~2 s)."""

    def test_states_population_tiles(self):
        from carto_flow.data import load_us_census

        gdf = load_us_census(level="state", population=True, contiguous_only=True)
        population = gdf["Population"].to_numpy(dtype=float)
        tiles = np.maximum(1, np.round(population / population.sum() * 150)).astype(int)
        gdf = gdf.assign(tiles=tiles)

        data = prepare_layout_data(gdf, tile_count="tiles")
        result = MosaicLayout().compute(data, show_progress=False)

        metrics = result.metrics.algorithm
        assert metrics.regions_correct == metrics.regions_total == len(gdf)
        np.testing.assert_array_equal(tile_counts_per_geometry(result), tiles)
        assert metrics.n_components == 1
