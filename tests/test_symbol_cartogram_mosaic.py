"""Tests for MosaicLayout (symbol_cartogram.layouts.mosaic).

Most tests use small synthetic grids of unit squares with ``morph=False`` so
the module runs in a few seconds; one test uses the bundled US states.

What the layout guarantees, and what it only attempts:

* exact tile counts per region — structural (one Hungarian slot per tile), so
  asserted everywhere;
* intra-region contiguity and inter-region adjacency — best effort. The
  iterative repair loop keeps the best-scoring assignment it finds, which is
  not always violation-free, but a post-ring chain-swap repair then closes
  remaining splits when it can do so without breaking another region.
  Contiguity is therefore asserted on the fixtures the implementation
  achieves it on.
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
        # ``converged`` is not asserted here: it now needs exact counts *and*
        # region-level contiguity, which some of these fixtures do not reach.
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

    @pytest.mark.parametrize(("cols", "rows", "counts"), [(3, 3, COUNTS_3X3), (4, 3, COUNTS_4X3)])
    def test_regions_gdf_tile_count_is_aggregated_per_geometry(self, cols, rows, counts):
        """One ``regions_gdf`` row per geometry, holding exactly its own tiles.

        Regression test: ``tile_count`` used to be aggregated with the N-level
        ``group_ids``, which mixed tiles of unrelated geometries (a 4-tile
        region reported 18).
        """
        gdf = grid_gdf(cols, rows, counts)
        result = compute(gdf)

        assert len(result.regions_gdf) == len(gdf)
        assert list(result.regions_gdf["tile_count"]) == counts
        assert list(result.regions_gdf["target_count"]) == counts
        assert int(result.regions_gdf["tile_count"].sum()) == len(result.transforms)


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

    def test_all_regions_contiguous_3x3(self):
        """The 3x3 fixture reaches full contiguity via the chain-swap repair.

        Was a strict xfail: 2 of 9 regions came out split because the repair
        loop's best pass was the raw Hungarian solve and it had no move left.
        The post-ring chain-swap repair closes both.
        """
        gdf = grid_gdf(3, 3, COUNTS_3X3)
        result = compute(gdf)

        assert non_contiguous_regions(result) == []

    def test_repair_never_worse_than_raw_solve_3x3(self):
        """The repair loop may not split more regions than the raw solve.

        The loop used to score disconnected *tiles*, which is not monotone in
        the number of split regions; it now ranks passes by split regions.
        """
        gdf = grid_gdf(3, 3, COUNTS_3X3)
        raw = compute(gdf, hungarian_options=HungarianOptions(max_connectivity_iters=0))
        repaired = compute(gdf)

        assert len(non_contiguous_regions(repaired)) <= len(non_contiguous_regions(raw))


# ---------------------------------------------------------------------------
# 2a. Chain-swap repair
# ---------------------------------------------------------------------------


def _core_holes(result: MosaicLayoutResult) -> int:
    """Unassigned core tiles every one of whose neighbours is assigned."""
    adjacency = result.tiling_result.adjacency
    assigned = {int(t) for t in result.assignments}
    holes = 0
    for t in (int(x) for x in result.core_tile_indices):
        if t in assigned:
            continue
        neighbours = np.flatnonzero(adjacency[t])
        if len(neighbours) and all(int(nb) in assigned for nb in neighbours):
            holes += 1
    return holes


def _unassigned_core(result: MosaicLayoutResult) -> int:
    """Core tiles that received no symbol."""
    assigned = {int(t) for t in result.assignments}
    return sum(1 for t in (int(x) for x in result.core_tile_indices) if t not in assigned)


def _ring_tiles_used(result: MosaicLayoutResult) -> int:
    """Assigned tiles that lie outside the core pool."""
    core = {int(x) for x in result.core_tile_indices}
    return len({int(t) for t in result.assignments} - core)


class TestRingSwapBack:
    """Stranded extra-ring tiles are pulled back into unassigned core tiles.

    A ring tile sits outside the core and leaves a core tile empty one-for-one,
    so the protruding tail and the hole are the same defect.  The swap-back
    walks a BFS path of occupied tiles to the nearest empty core tile and shifts
    ownership along it.
    """

    @pytest.mark.parametrize(("cols", "rows", "counts"), [(3, 3, COUNTS_3X3), (4, 3, COUNTS_4X3)])
    def test_exact_tile_counts_preserved(self, cols, rows, counts):
        """Shifting ownership along a path makes every region lose and gain one tile."""
        gdf = grid_gdf(cols, rows, counts)

        off = compute(gdf, hungarian_options=HungarianOptions(ring_swapback_max_hops=0))
        on = compute(gdf, hungarian_options=HungarianOptions(ring_swapback_max_hops=8))

        np.testing.assert_array_equal(tile_counts_per_geometry(off), counts)
        np.testing.assert_array_equal(tile_counts_per_geometry(on), counts)

    @pytest.mark.parametrize(("cols", "rows", "counts"), [(3, 3, COUNTS_3X3), (4, 3, COUNTS_4X3)])
    def test_unassigned_core_tiles_never_increase(self, cols, rows, counts):
        gdf = grid_gdf(cols, rows, counts)

        off = compute(gdf, hungarian_options=HungarianOptions(ring_swapback_max_hops=0))
        on = compute(gdf, hungarian_options=HungarianOptions(ring_swapback_max_hops=8))

        assert _unassigned_core(on) <= _unassigned_core(off)
        assert _ring_tiles_used(on) <= _ring_tiles_used(off)

    @pytest.mark.parametrize(("cols", "rows", "counts"), [(3, 3, COUNTS_3X3), (4, 3, COUNTS_4X3)])
    def test_split_regions_never_increase(self, cols, rows, counts):
        """The accept-guard makes 'never regress' structural, not tuned."""
        gdf = grid_gdf(cols, rows, counts)

        off = compute(gdf, hungarian_options=HungarianOptions(ring_swapback_max_hops=0))
        on = compute(gdf, hungarian_options=HungarianOptions(ring_swapback_max_hops=8))

        assert len(non_contiguous_regions(on)) <= len(non_contiguous_regions(off))

    def test_split_groups_never_increase(self):
        """Same guard on the grouped path, where the unit is the group."""
        gdf, _, _ = group_fixture()
        data = prepare_layout_data(gdf, group_by="grp")

        off = MosaicLayout(morph=False, hungarian_options=HungarianOptions(ring_swapback_max_hops=0)).compute(
            data, show_progress=False
        )
        on = MosaicLayout(morph=False, hungarian_options=HungarianOptions(ring_swapback_max_hops=8)).compute(
            data, show_progress=False
        )

        assert on.metrics.algorithm.n_split_groups <= off.metrics.algorithm.n_split_groups
        assert _unassigned_core(on) <= _unassigned_core(off)

    def test_zero_hops_disables_and_one_hop_is_adjacent_only(self):
        """The single knob spans "off" and the old adjacent-only behaviour."""
        gdf = grid_gdf(4, 3, COUNTS_4X3)

        off = compute(gdf, hungarian_options=HungarianOptions(ring_swapback_max_hops=0))
        adjacent_only = compute(gdf, hungarian_options=HungarianOptions(ring_swapback_max_hops=1))
        far = compute(gdf, hungarian_options=HungarianOptions(ring_swapback_max_hops=8))

        assert _unassigned_core(far) <= _unassigned_core(adjacent_only) <= _unassigned_core(off)

    def test_negative_hops_rejected(self):
        with pytest.raises(ValueError, match="ring_swapback_max_hops must be >= 0"):
            HungarianOptions(ring_swapback_max_hops=-1)


class TestChainSwapRepair:
    """The post-ring chain-swap repair closes splits without side effects."""

    @pytest.mark.parametrize(("cols", "rows", "counts"), [(3, 3, COUNTS_3X3), (4, 3, COUNTS_4X3)])
    def test_exact_tile_counts_preserved_under_repair(self, cols, rows, counts):
        """The repair only permutes ownership of occupied tiles."""
        gdf = grid_gdf(cols, rows, counts)

        off = compute(gdf, hungarian_options=HungarianOptions(swap_repair_passes=0))
        on = compute(gdf, hungarian_options=HungarianOptions(swap_repair_passes=10))

        np.testing.assert_array_equal(tile_counts_per_geometry(on), counts)
        np.testing.assert_array_equal(tile_counts_per_geometry(off), counts)

    @pytest.mark.parametrize(("cols", "rows", "counts"), [(3, 3, COUNTS_3X3), (4, 3, COUNTS_4X3)])
    def test_repair_never_increases_split_regions(self, cols, rows, counts):
        """Turning the repair on may not split more regions than leaving it off."""
        gdf = grid_gdf(cols, rows, counts)

        off = compute(gdf, hungarian_options=HungarianOptions(swap_repair_passes=0))
        on = compute(gdf, hungarian_options=HungarianOptions(swap_repair_passes=10))

        assert len(non_contiguous_regions(on)) <= len(non_contiguous_regions(off))

    @pytest.mark.parametrize(("cols", "rows", "counts"), [(3, 3, COUNTS_3X3), (4, 3, COUNTS_4X3)])
    def test_repair_occupies_the_same_tiles(self, cols, rows, counts):
        """The occupied tile set is untouched, so the ring swap-back's work stands."""
        gdf = grid_gdf(cols, rows, counts)

        off = compute(gdf, hungarian_options=HungarianOptions(swap_repair_passes=0))
        on = compute(gdf, hungarian_options=HungarianOptions(swap_repair_passes=10))

        assert {int(t) for t in on.assignments} == {int(t) for t in off.assignments}

    @pytest.mark.parametrize(("cols", "rows", "counts"), [(3, 3, COUNTS_3X3), (4, 3, COUNTS_4X3)])
    def test_repair_introduces_no_core_holes(self, cols, rows, counts):
        """The repair adds no unassigned tile enclosed by assigned ones."""
        gdf = grid_gdf(cols, rows, counts)

        off = compute(gdf, hungarian_options=HungarianOptions(swap_repair_passes=0))
        on = compute(gdf, hungarian_options=HungarianOptions(swap_repair_passes=10))

        assert _core_holes(on) <= _core_holes(off)


# ---------------------------------------------------------------------------
# 2b. Topology metrics
# ---------------------------------------------------------------------------


class TestTopologyMetrics:
    """``MosaicMetrics`` reports region-level topology, and ``converged`` uses it."""

    def test_metrics_report_split_regions(self):
        """With the chain-swap repair off, the 3x3 fixture splits regions.

        ``converged`` must then be False even though every region has exactly
        its requested tile count — exact counts alone do not imply convergence.
        """
        gdf = grid_gdf(3, 3, COUNTS_3X3)
        result = compute(gdf, hungarian_options=HungarianOptions(swap_repair_passes=0))

        metrics = result.metrics.algorithm
        assert metrics.n_noncontiguous_regions == len(non_contiguous_regions(result))
        assert metrics.n_noncontiguous_regions > 0
        assert metrics.n_split_groups == 0
        assert result.metrics.converged is False
        assert metrics.regions_correct == metrics.regions_total

    def test_metrics_report_unassigned_core_tiles(self):
        """``n_unassigned_core_tiles`` counts every empty core tile, enclosed or not.

        The enclosed-hole check misses gaps in concave boundary pockets — on US
        states it reported 0 while five were plainly visible — so the metric is
        deliberately the wider count.  It is *not* folded into ``converged``:
        that would be a public behaviour change, and on some inputs no legal
        relocation exists.
        """
        gdf = grid_gdf(4, 3, COUNTS_4X3)
        result = compute(gdf, hungarian_options=HungarianOptions(ring_swapback_max_hops=0))

        metrics = result.metrics.algorithm
        assert metrics.n_unassigned_core_tiles == _unassigned_core(result)
        assert metrics.n_unassigned_core_tiles >= _core_holes(result)

    def test_metrics_report_convergence_after_repair(self):
        """With the repair on (the default) the same fixture converges."""
        gdf = grid_gdf(3, 3, COUNTS_3X3)
        result = compute(gdf)

        metrics = result.metrics.algorithm
        assert metrics.n_noncontiguous_regions == 0
        assert result.metrics.converged is True

    def test_repair_passes_is_the_real_pass_count(self):
        """``iterations`` reports passes run, not ``max_connectivity_iters``."""
        gdf = grid_gdf(4, 3, COUNTS_4X3)
        result = compute(gdf, hungarian_options=HungarianOptions(max_connectivity_iters=7))

        metrics = result.metrics.algorithm
        assert 1 <= metrics.repair_passes <= 8
        assert result.metrics.iterations == metrics.repair_passes

    def test_converged_true_when_contiguous(self):
        """The 4x3 fixture is fully contiguous with exact counts."""
        gdf = grid_gdf(4, 3, COUNTS_4X3)
        result = compute(gdf)

        assert non_contiguous_regions(result) == []
        assert result.metrics.algorithm.n_noncontiguous_regions == 0
        assert result.metrics.converged is True


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


class TestGroupIdLevels:
    """``group_ids`` is N-level (per symbol); ``group_ids_G`` is the G-level grouping."""

    def test_tile_count_has_no_user_grouping(self):
        gdf = grid_gdf(3, 3, COUNTS_3X3)
        data = prepare_layout_data(gdf, tile_count="tiles")

        assert data.group_ids is not None
        assert len(data.group_ids) == sum(COUNTS_3X3)  # N-level
        assert data.group_ids_G is None  # no group_by, so no user grouping

        result = MosaicLayout(morph=False).compute(data, show_progress=False)
        assert result.group_ids is None

    def test_group_by_sets_g_level_ids(self):
        gdf, groups, _ = group_fixture()
        data = prepare_layout_data(gdf, group_by="grp")

        expected = np.unique(groups, return_inverse=True)[1]
        assert data.group_ids_G is not None
        assert len(data.group_ids_G) == len(gdf)  # G-level
        np.testing.assert_array_equal(data.group_ids_G, expected)
        np.testing.assert_array_equal(data.group_ids, expected)  # N == G here

        result = MosaicLayout(morph=False).compute(data, show_progress=False)
        assert result.group_ids is not None
        np.testing.assert_array_equal(result.group_ids, expected[np.asarray(result.source_indices)])

    def test_tile_count_and_group_by_cannot_be_combined(self):
        gdf = grid_gdf(3, 3, COUNTS_3X3).assign(grp="a")
        with pytest.raises(ValueError, match="Cannot set both tile_count and group_by"):
            prepare_layout_data(gdf, tile_count="tiles", group_by="grp")


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


@pytest.fixture(scope="module")
def states_gdf() -> gpd.GeoDataFrame:
    """Bundled US states with a ``tiles`` column totalling about 150 tiles."""
    from carto_flow.data import load_us_census

    gdf = load_us_census(level="state", population=True, contiguous_only=True)
    population = gdf["Population"].to_numpy(dtype=float)
    tiles = np.maximum(1, np.round(population / population.sum() * 150)).astype(int)
    return gdf.assign(tiles=tiles)


class TestUsStates:
    """One end-to-end run on the bundled US states (about 150 tiles, ~2 s)."""

    def test_states_population_tiles(self, states_gdf):
        gdf = states_gdf
        tiles = gdf["tiles"].to_numpy()

        data = prepare_layout_data(gdf, tile_count="tiles")
        result = MosaicLayout().compute(data, show_progress=False)

        metrics = result.metrics.algorithm
        assert metrics.regions_correct == metrics.regions_total == len(gdf)
        np.testing.assert_array_equal(tile_counts_per_geometry(result), tiles)
        assert metrics.n_components == 1
        # One regions_gdf row per state, each holding exactly its requested tiles.
        assert len(result.regions_gdf) == len(gdf)
        np.testing.assert_array_equal(result.regions_gdf["tile_count"].to_numpy(), tiles)
        np.testing.assert_array_equal(result.regions_gdf["target_count"].to_numpy(), tiles)

    def test_ring_swapback_recovers_stranded_tiles(self, states_gdf):
        """Regression: 5 ring tiles and 5 empty core tiles, 4 of them by the Great Lakes.

        Michigan's empty core tile is two rows from its ring tile, so the old
        adjacent-only swap-back could never make the first hop.  Three of the
        five are legally recoverable; Florida's and Washington's only reachable
        empty tiles are ten-plus hops away and moving them would split
        intervening states, so the guard correctly declines.
        """
        data = prepare_layout_data(states_gdf, tile_count="tiles")
        tiles = states_gdf["tiles"].to_numpy()

        off = MosaicLayout(hungarian_options=HungarianOptions(ring_swapback_max_hops=0)).compute(
            data, show_progress=False
        )
        on = MosaicLayout().compute(data, show_progress=False)

        assert _unassigned_core(on) < _unassigned_core(off)
        assert _ring_tiles_used(on) < _ring_tiles_used(off)
        # Never regress the topology metrics while doing it.
        assert on.metrics.algorithm.n_noncontiguous_regions <= off.metrics.algorithm.n_noncontiguous_regions
        assert on.metrics.algorithm.n_split_groups <= off.metrics.algorithm.n_split_groups
        # Exact tile counts stay structural.
        np.testing.assert_array_equal(tile_counts_per_geometry(on), tiles)

    def test_repair_never_worse_than_raw_solve(self, states_gdf):
        """Regression: repair used to turn 2 split states into 9.

        The loop scored disconnected tiles; the pass it preferred was worse in
        the only number that shows, the count of states in more than one block.
        """
        data = prepare_layout_data(states_gdf, tile_count="tiles")
        raw = MosaicLayout(hungarian_options=HungarianOptions(max_connectivity_iters=0)).compute(
            data, show_progress=False
        )
        repaired = MosaicLayout().compute(data, show_progress=False)

        n_raw = len(non_contiguous_regions(raw))
        n_repaired = len(non_contiguous_regions(repaired))
        assert n_repaired <= n_raw
        # `n_noncontiguous_regions` counts *sub-regions*, while
        # `non_contiguous_regions` counts geometries, and Michigan is two
        # sub-regions: its two peninsulas are two blocks on purpose, so the
        # geometry-level helper sees one split that the metric does not.
        split_geometries = repaired.metrics.algorithm.n_split_geometries
        metric = repaired.metrics.algorithm.n_noncontiguous_regions
        assert n_repaired - split_geometries <= metric <= n_repaired


# ---------------------------------------------------------------------------
# 8. Enclosed unassigned cells, whatever their core status
# ---------------------------------------------------------------------------


def _enclosed_cells(result: MosaicLayoutResult) -> set[int]:
    """Unassigned lattice cells whose every lattice neighbour is assigned.

    Deliberately ignores core status: that is the whole point of the widening.
    ``_core_holes`` above only looks at core tiles and cannot see a hole that
    fell below ``min_overlap_frac``.
    """
    adjacency = result.tiling_result.adjacency
    assigned = {int(t) for t in result.assignments}
    out = set()
    for t in range(len(result.tiling_result.polygons)):
        if t in assigned:
            continue
        neighbours = np.flatnonzero(adjacency[t])
        if len(neighbours) and all(int(nb) in assigned for nb in neighbours):
            out.add(t)
    return out


def donut_gdf() -> gpd.GeoDataFrame:
    """4x4 block of unit boxes with the 2x2 centre missing: a genuine inner sea."""
    geoms, counts = [], []
    for r in range(4):
        for c in range(4):
            if r in (1, 2) and c in (1, 2):
                continue
            geoms.append(box(c, r, c + 1, r + 1))
            counts.append(4)
    return gpd.GeoDataFrame({"tiles": counts}, geometry=geoms)


class TestInnerSeaCharacterisation:
    """RECORD OF CURRENT BEHAVIOUR -- NOT AN ENDORSEMENT.

    Mosaic pays no attention to genuine holes in the coverage: it will pave over an
    inner sea.  This is deliberate for now.  The source is upstream of anything the
    relocation does -- calibration counts cells that sit in the hole as *core* tiles,
    because ``min_overlap_frac`` defaults to 0.1 to let marginal tiles into the pool
    and give the assignment room to manoeuvre.  It is a pool-admission knob, not a
    land/water classifier, and the tile budget therefore already expects the sea to be
    covered.

    These assertions exist so the queued calibration-level issue has a reference point
    and so a future change that alters this behaviour is noticed.  ``load_world()`` has
    a real instance: exactly one interior ring, the Caspian Sea.
    """

    @staticmethod
    def _water(gdf):
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        union = unary_union(list(gdf.geometry))
        polys = [union] if union.geom_type == "Polygon" else list(union.geoms)
        rings = [Polygon(r) for p in polys if p.geom_type == "Polygon" for r in p.interiors]
        return unary_union(rings).difference(union)

    def test_the_fixture_really_has_an_inner_sea(self):
        water = self._water(donut_gdf())

        assert not water.is_empty
        assert water.area > 0

    def test_cells_in_the_inner_sea_are_counted_as_core(self):
        """Characterisation: calibration admits lake cells to the core tile budget."""
        gdf = donut_gdf()
        data = prepare_layout_data(gdf, tile_count="tiles")
        result = MosaicLayout(morph=False).compute(data, show_progress=False)
        water = self._water(gdf)

        core = {int(x) for x in result.core_tile_indices}
        mostly_water = {
            t for t, poly in enumerate(result.tiling_result.polygons) if poly.intersection(water).area > 0.5 * poly.area
        }

        assert mostly_water, "fixture should put some cells in the water"
        # Today most of them are core tiles, so the budget expects them covered.
        assert mostly_water & core

    def test_the_inner_sea_gets_paved_over(self):
        """Characterisation: some lake cells end up occupied.  Known, not endorsed."""
        gdf = donut_gdf()
        data = prepare_layout_data(gdf, tile_count="tiles")
        result = MosaicLayout(morph=False).compute(data, show_progress=False)
        water = self._water(gdf)

        assigned = {int(t) for t in result.assignments}
        mostly_water = {
            t for t, poly in enumerate(result.tiling_result.polygons) if poly.intersection(water).area > 0.5 * poly.area
        }

        assert mostly_water & assigned
        # Whatever it does with the sea, the counts stay exact.
        np.testing.assert_array_equal(tile_counts_per_geometry(result), gdf["tiles"].to_numpy())

    def test_relocation_does_not_make_it_worse(self):
        """The relocation neither creates nor removes lake coverage on this fixture."""
        gdf = donut_gdf()
        data = prepare_layout_data(gdf, tile_count="tiles")
        water = self._water(gdf)

        off = MosaicLayout(morph=False, hungarian_options=HungarianOptions(ring_swapback_max_hops=0)).compute(
            data, show_progress=False
        )
        on = MosaicLayout(morph=False, hungarian_options=HungarianOptions(ring_swapback_max_hops=8)).compute(
            data, show_progress=False
        )

        def wet_occupied(result):
            return len({
                int(t)
                for t in result.assignments
                if result.tiling_result.polygons[int(t)].intersection(water).area
                > 0.5 * result.tiling_result.polygons[int(t)].area
            })

        assert wet_occupied(on) <= wet_occupied(off)


class TestEnclosedUnassignedMetric:
    """``n_enclosed_unassigned_tiles`` is the widened, core-status-blind count."""

    @pytest.mark.parametrize(("cols", "rows", "counts"), [(3, 3, COUNTS_3X3), (4, 3, COUNTS_4X3)])
    def test_metric_matches_recomputation(self, cols, rows, counts):
        result = compute(grid_gdf(cols, rows, counts))

        assert result.metrics.algorithm.n_enclosed_unassigned_tiles == len(_enclosed_cells(result))

    def test_metric_is_not_bounded_by_the_core_only_count(self):
        """The widened count sees at least what the core-only hole count sees.

        ``_core_holes`` is the metric PR #30/#31 used; every hole it finds is a
        lattice cell with all neighbours assigned, so the widened count must
        include it.  The reverse does not hold, which is the defect.
        """
        result = compute(grid_gdf(4, 3, COUNTS_4X3))

        assert result.metrics.algorithm.n_enclosed_unassigned_tiles >= _core_holes(result)


class TestEnclosedHoleRelocation:
    """Enclosed cells are relocation targets whatever their core status."""

    @pytest.mark.parametrize(("cols", "rows", "counts"), [(3, 3, COUNTS_3X3), (4, 3, COUNTS_4X3)])
    def test_exact_tile_counts_preserved(self, cols, rows, counts):
        """Ownership shifts along a path, so every region loses and gains one tile."""
        gdf = grid_gdf(cols, rows, counts)

        off = compute(gdf, hungarian_options=HungarianOptions(ring_swapback_max_hops=0))
        on = compute(gdf, hungarian_options=HungarianOptions(ring_swapback_max_hops=8))

        np.testing.assert_array_equal(tile_counts_per_geometry(off), counts)
        np.testing.assert_array_equal(tile_counts_per_geometry(on), counts)

    @pytest.mark.parametrize(("cols", "rows", "counts"), [(3, 3, COUNTS_3X3), (4, 3, COUNTS_4X3)])
    def test_enclosed_cells_never_increase(self, cols, rows, counts):
        """The guard makes 'never regress' structural rather than tuned."""
        gdf = grid_gdf(cols, rows, counts)

        off = compute(gdf, hungarian_options=HungarianOptions(ring_swapback_max_hops=0))
        on = compute(gdf, hungarian_options=HungarianOptions(ring_swapback_max_hops=8))

        assert len(_enclosed_cells(on)) <= len(_enclosed_cells(off))
        assert _unassigned_core(on) <= _unassigned_core(off)
        assert len(non_contiguous_regions(on)) <= len(non_contiguous_regions(off))

    def test_enclosed_cells_never_increase_without_morph(self):
        """``morph=False`` is the harder case for the assignment and was untested."""
        gdf = grid_gdf(4, 3, COUNTS_4X3)
        data = prepare_layout_data(gdf, tile_count="tiles")

        off = MosaicLayout(morph=False, hungarian_options=HungarianOptions(ring_swapback_max_hops=0)).compute(
            data, show_progress=False
        )
        on = MosaicLayout(morph=False, hungarian_options=HungarianOptions(ring_swapback_max_hops=8)).compute(
            data, show_progress=False
        )

        assert len(_enclosed_cells(on)) <= len(_enclosed_cells(off))
        assert _unassigned_core(on) <= _unassigned_core(off)
        assert on.metrics.algorithm.n_split_groups <= off.metrics.algorithm.n_split_groups
        np.testing.assert_array_equal(tile_counts_per_geometry(on), COUNTS_4X3)

    def test_groups_never_split_more_without_morph(self):
        """Grouped path, no pre-morph: the split guard still holds."""
        gdf, _, _ = group_fixture()
        data = prepare_layout_data(gdf, group_by="grp")

        off = MosaicLayout(morph=False, hungarian_options=HungarianOptions(ring_swapback_max_hops=0)).compute(
            data, show_progress=False
        )
        on = MosaicLayout(morph=False, hungarian_options=HungarianOptions(ring_swapback_max_hops=8)).compute(
            data, show_progress=False
        )

        assert on.metrics.algorithm.n_split_groups <= off.metrics.algorithm.n_split_groups
        assert len(_enclosed_cells(on)) <= len(_enclosed_cells(off))


# ---------------------------------------------------------------------------
# 9. Multi-part regions become sub-regions
# ---------------------------------------------------------------------------


def multipart_gdf(count_multi: int = 16) -> gpd.GeoDataFrame:
    """Two regions, the right one a MultiPolygon with a genuine 6-unit gap.

    ``g1``'s parts are 4x4 boxes with several tile widths of empty space between
    them, so no chain of tiles can join them.  Required to be one block -- which
    is what happened before sub-regions existed -- the solver reaches across the
    gap and shreds its *neighbour* ``g0`` to do it.
    """
    from shapely.geometry import MultiPolygon

    g0 = box(0.0, 0.0, 4.0, 4.0)
    g1 = MultiPolygon([box(4.0, 0.0, 8.0, 4.0), box(14.0, 0.0, 18.0, 4.0)])
    return gpd.GeoDataFrame({"tiles": [16, count_multi]}, geometry=[g0, g1])


def _blocks_of_geometry(result: MosaicLayoutResult, geom: int) -> int:
    """How many connected tile blocks geometry *geom* occupies."""
    adjacency = tile_adjacency(result)
    tiles = tiles_by_key(result, list(range(len(result.counts)))).get(geom, [])
    if not tiles:
        return 0
    seen, blocks = set(), 0
    for start in tiles:
        if start in seen:
            continue
        blocks += 1
        queue, seen = deque([start]), seen | {start}
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor in tiles and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
    return blocks


class TestApportionment:
    """Largest-remainder apportionment of a region's tiles across its parts."""

    def test_sums_exactly(self):
        from carto_flow.symbol_cartogram.layouts.mosaic._subregions import apportion_largest_remainder

        for total in (1, 2, 3, 5, 7, 11, 154):
            for weights in ([0.709, 0.283, 0.008], [1.0, 1.0], [0.5, 0.3, 0.2], [9.0, 1.0]):
                allot = apportion_largest_remainder(total, np.array(weights))
                assert allot.sum() == total
                assert (allot >= 0).all()

    def test_follows_area_shares(self):
        from carto_flow.symbol_cartogram.layouts.mosaic._subregions import apportion_largest_remainder

        # Michigan's 70.9 / 28.3 / residual split over 5 tiles.
        allot = apportion_largest_remainder(5, np.array([0.709, 0.283, 0.008]))
        assert list(allot) == [4, 1, 0]

    def test_ties_are_deterministic(self):
        from carto_flow.symbol_cartogram.layouts.mosaic._subregions import apportion_largest_remainder

        allot = apportion_largest_remainder(3, np.array([1.0, 1.0]))
        assert allot.sum() == 3
        assert list(allot) == list(apportion_largest_remainder(3, np.array([1.0, 1.0])))


class TestMultipartSubRegions:
    """A MultiPolygon whose parts are big enough is laid out as separate blocks."""

    def test_parts_become_separate_blocks(self):
        result = compute(multipart_gdf())
        metrics = result.metrics.algorithm

        assert metrics.n_split_geometries == 1
        assert metrics.n_subregions == 3  # g0 + g1's two parts
        assert _blocks_of_geometry(result, 1) == 2
        # The neighbour is left alone, which forcing the gap closed did not do.
        assert _blocks_of_geometry(result, 0) == 1
        # Each sub-region is itself one block, so nothing is reported as split.
        assert metrics.n_noncontiguous_regions == 0
        assert result.metrics.converged

    def test_tile_counts_are_preserved_exactly(self):
        gdf = multipart_gdf()
        result = compute(gdf)
        np.testing.assert_array_equal(tile_counts_per_geometry(result), gdf["tiles"].to_numpy())
        assert int(np.sum(tile_counts_per_geometry(result))) == int(gdf["tiles"].sum())

    def test_apportionment_splits_equal_parts_evenly(self):
        """g1's parts have equal area, so its 16 tiles must go 8 and 8."""
        result = compute(multipart_gdf(count_multi=16))
        adjacency = tile_adjacency(result)
        tiles = tiles_by_key(result, list(range(len(result.counts))))[1]
        sizes = []
        seen = set()
        for start in tiles:
            if start in seen:
                continue
            queue, block = deque([start]), {start}
            seen.add(start)
            while queue:
                node = queue.popleft()
                for neighbor in adjacency[node]:
                    if neighbor in tiles and neighbor not in seen:
                        seen.add(neighbor)
                        block.add(neighbor)
                        queue.append(neighbor)
            sizes.append(len(block))
        assert sorted(sizes) == [8, 8]

    def test_threshold_off_restores_previous_behaviour(self):
        """A very large threshold keeps every geometry whole, as before this feature.

        This is the defect, measured: required to bridge a gap it cannot bridge,
        the solver breaks the neighbouring region into three blocks and the
        layout cannot converge.
        """
        whole = compute(multipart_gdf(), multipart_min_tiles=1e9)
        assert whole.metrics.algorithm.n_split_geometries == 0
        assert whole.metrics.algorithm.n_subregions == len(whole.counts)
        assert whole.metrics.algorithm.n_noncontiguous_regions >= 1
        assert not whole.metrics.converged
        # Counts are still exact -- the damage is topological, not arithmetic.
        np.testing.assert_array_equal(tile_counts_per_geometry(whole), [16, 16])

    def test_threshold_rejects_parts_below_one_tile(self):
        """A part far smaller than a tile is not a sub-region -- an island stays an island."""
        from shapely.geometry import MultiPolygon

        g0 = box(0.0, 0.0, 4.0, 4.0)
        g1 = MultiPolygon([box(4.0, 0.0, 8.0, 4.0), box(14.0, 1.9, 14.1, 2.0)])
        gdf = gpd.GeoDataFrame({"tiles": [16, 16]}, geometry=[g0, g1])
        result = compute(gdf)
        assert result.metrics.algorithm.n_split_geometries == 0
        np.testing.assert_array_equal(tile_counts_per_geometry(result), [16, 16])

    def test_one_tile_region_is_never_split(self):
        """A sub-region with no tiles is not a sub-region: a 1-tile region stays whole."""
        result = compute(multipart_gdf(count_multi=1))
        assert result.metrics.algorithm.n_split_geometries == 0
        assert int(tile_counts_per_geometry(result)[1]) == 1

    def test_morph_path_also_splits(self):
        """The pre-morph is the easy case; ``morph=False`` above is the hard one."""
        result = compute(multipart_gdf(), morph=True)
        assert result.metrics.algorithm.n_split_geometries == 1
        np.testing.assert_array_equal(tile_counts_per_geometry(result), [16, 16])

    def test_group_by_inputs_never_split(self):
        """``group_by`` gives every geometry one symbol, so nothing can split.

        A one-tile region is never a candidate, which is why the grouped path --
        districts, and districts by state -- is untouched by this feature.
        """
        from shapely.geometry import MultiPolygon

        geoms = [
            box(0.0, 0.0, 4.0, 4.0),
            MultiPolygon([box(4.0, 0.0, 8.0, 4.0), box(14.0, 0.0, 18.0, 4.0)]),
            box(0.0, 4.0, 4.0, 8.0),
        ]
        gdf = gpd.GeoDataFrame({"grp": ["a", "a", "b"]}, geometry=geoms)
        data = prepare_layout_data(gdf, group_by="grp")
        result = MosaicLayout(morph=False).compute(data, show_progress=False)
        assert result.metrics.algorithm.n_split_geometries == 0
        assert result.metrics.algorithm.n_subregions == len(geoms)


class TestMultipartUsStates:
    """Michigan and Rhode Island, the two real multi-part cases on the bundled data."""

    @pytest.mark.parametrize("morph", [True, False])
    def test_michigan_is_two_blocks(self, states_gdf, morph):
        gdf = states_gdf
        michigan = list(gdf["State Name"]).index("Michigan")
        data = prepare_layout_data(gdf, tile_count="tiles")
        result = MosaicLayout(morph=morph).compute(data, show_progress=False)
        metrics = result.metrics.algorithm

        assert metrics.n_split_geometries == 1
        assert metrics.n_subregions == len(gdf) + 1
        # Every sub-region is one block, so the layout legitimately converges --
        # rather than converging because the repair forced the peninsulas together.
        assert metrics.n_noncontiguous_regions == 0
        assert result.metrics.converged
        # The two sub-regions may still end up side by side in the lattice, so
        # the *geometry* can read as one block; what changed is that it is no
        # longer required to.
        assert _blocks_of_geometry(result, michigan) >= 1
        # Exact counts, unchanged.
        np.testing.assert_array_equal(tile_counts_per_geometry(result), gdf["tiles"].to_numpy())

    def test_rhode_island_stays_whole(self, states_gdf):
        """RI gets one tile in total, so it cannot be two sub-regions.

        Its second-largest part is ~8% of its area -- well under a tile at this
        budget -- and the zero-allotment rule would fold it back in regardless.
        """
        gdf = states_gdf
        rhode_island = list(gdf["State Name"]).index("Rhode Island")
        data = prepare_layout_data(gdf, tile_count="tiles")
        result = MosaicLayout().compute(data, show_progress=False)
        assert int(gdf["tiles"].to_numpy()[rhode_island]) == 1
        assert _blocks_of_geometry(result, rhode_island) == 1

    def test_totals_are_preserved(self, states_gdf):
        data = prepare_layout_data(states_gdf, tile_count="tiles")
        result = MosaicLayout().compute(data, show_progress=False)
        assert int(np.sum(tile_counts_per_geometry(result))) == int(states_gdf["tiles"].sum())
        np.testing.assert_array_equal(result.regions_gdf["tile_count"].to_numpy(), states_gdf["tiles"].to_numpy())


class TestRingSwapbackReach:
    """The relocation reach is 16 hops, and US states needs more than 8 of them."""

    def test_default_is_sixteen(self):
        """Pinned deliberately.

        8 was the original default, justified by "on US states the fixable count
        saturates at 8 hops".  Sub-regions invalidated that measurement: a freed
        core cell can sit much further from the nearest stranded ring tile.  If
        this assertion is ever failing because someone lowered the number, read
        ``ring_swapback_max_hops`` in ``HungarianOptions`` before changing it.
        """
        assert HungarianOptions().ring_swapback_max_hops == 16
        assert MosaicLayoutOptions().hungarian_options is None  # defaults come from HungarianOptions

    @pytest.mark.parametrize("morph", [True, False])
    def test_us_states_needs_more_than_eight_hops(self, states_gdf, morph):
        """A reach of 8 leaves defects on US states that 16 closes.

        With Michigan laid out as two sub-regions, the core cell the split frees
        is ringed by Illinois, Indiana, Michigan and Wisconsin, while the nearest
        stranded ring tile is in Vermont -- far outside 8 hops.
        """
        data = prepare_layout_data(states_gdf, tile_count="tiles")
        near = MosaicLayout(morph=morph, hungarian_options=HungarianOptions(ring_swapback_max_hops=8)).compute(
            data, show_progress=False
        )
        far = MosaicLayout(morph=morph).compute(data, show_progress=False)

        assert _unassigned_core(far) < _unassigned_core(near)
        assert _ring_tiles_used(far) < _ring_tiles_used(near)
        assert len(_enclosed_cells(far)) <= len(_enclosed_cells(near))
        # The wider search never regresses the topology metrics or the counts.
        assert far.metrics.algorithm.n_noncontiguous_regions <= near.metrics.algorithm.n_noncontiguous_regions
        assert far.metrics.algorithm.n_split_groups <= near.metrics.algorithm.n_split_groups
        np.testing.assert_array_equal(tile_counts_per_geometry(far), states_gdf["tiles"].to_numpy())
