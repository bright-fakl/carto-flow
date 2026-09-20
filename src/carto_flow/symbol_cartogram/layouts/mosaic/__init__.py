"""Mosaic (tilegram) layout: MosaicLayout and MosaicLayoutOptions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..base import Layout, _apply_kwargs_to_options
from ..data_prep import LayoutData
from ..layout_result import LayoutResult, Transform

__all__ = ["HungarianOptions", "MosaicLayout", "MosaicLayoutOptions", "MosaicMetrics"]


@dataclass
class MosaicMetrics:
    """Algorithm-specific metrics for MosaicLayout.

    Attributes
    ----------
    tiling, tile_size, n_components
        Tiling used, its calibrated tile size, and the number of disconnected
        geographic components the study area was split into.
    regions_correct, regions_total
        How many geometries received exactly their requested tile count.
    n_noncontiguous_regions : int
        Number of geometries whose assigned tiles do not form a single
        connected block in the tile adjacency graph.
    n_split_groups : int
        With ``group_by``, the number of groups whose tiles form more than one
        block.  0 without a grouping.  Groups are counted after splitting at
        geographic component boundaries, since a group spread over separate
        land masses cannot be one block.
    repair_passes : int
        Linear-assignment solves actually run by the connectivity-repair loop
        (the maximum over components, so 1 means "no repair pass was needed").
    n_unassigned_core_tiles : int
        Core tiles (inside the study union) that received no symbol.  Because
        calibration matches the core tile count to the tile budget, this also
        counts the tiles the solver had to take from the extra rings outside
        the core, which show up as protruding tails.  Unlike an enclosed-hole
        count it also catches gaps in concave boundary pockets.
    n_enclosed_unassigned_tiles : int
        Lattice cells that received no symbol yet have every lattice neighbour
        assigned.  Deliberately independent of core status: a cell ringed by
        assigned tiles reads as a hole whatever its overlap fraction, and a
        core-only count misses the ones that fall below ``min_overlap_frac``.
    """

    tiling: str = ""
    tile_size: float = 0.0
    n_components: int = 0
    regions_correct: int = 0
    regions_total: int = 0
    n_noncontiguous_regions: int = 0
    n_split_groups: int = 0
    repair_passes: int = 0
    n_unassigned_core_tiles: int = 0
    n_enclosed_unassigned_tiles: int = 0


def _enclosed_unassigned(occupied: set[int], adj_list: list[list[int]]) -> set[int]:
    """Unassigned lattice cells whose every lattice neighbour is assigned.

    Core status is ignored on purpose -- see ``MosaicMetrics``.  A cell with no
    neighbours at all (an isolated lattice corner) is not a hole.
    """
    holes = set()
    for t in range(len(adj_list)):
        if t in occupied:
            continue
        nbs = adj_list[t]
        if nbs and all(nb in occupied for nb in nbs):
            holes.add(t)
    return holes


def _count_split_units(
    assignment: np.ndarray,
    tile_indices: list[int],
    adj_list: list[list[int]],
    unit_of_geom: np.ndarray,
) -> int:
    """Number of units whose assigned tiles are not one connected block.

    ``unit_of_geom`` maps geometry index to unit id (the geometry itself, or
    its group).  One BFS per unit over the assigned tiles.
    """
    from collections import deque

    unit_tiles: dict[int, list[int]] = {}
    for t in tile_indices:
        g = int(assignment[t])
        if g >= 0:
            unit_tiles.setdefault(int(unit_of_geom[g]), []).append(t)

    split = 0
    for tiles in unit_tiles.values():
        if len(tiles) <= 1:
            continue
        tile_set = set(tiles)
        visited = {tiles[0]}
        queue: deque[int] = deque([tiles[0]])
        while queue:
            t = queue.popleft()
            for nb in adj_list[t]:
                if nb in tile_set and nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        if len(visited) != len(tile_set):
            split += 1
    return split


def _relocate_ring_tiles(
    assignment: np.ndarray,
    adj_list: list[list[int]],
    core_set: set[int],
    G: int,
    group_labels: np.ndarray | None,
    max_hops: int,
    show_progress: bool = False,
) -> np.ndarray:
    """Pull assigned extra-ring tiles back into unassigned core tiles.

    A tile the solver took from an extra ring sits outside the core, so — because
    calibration matches the core tile count to the tile budget — it leaves a core
    tile empty one-for-one.  The ring tile reads as a protruding tail and the core
    tile as a hole; they are the same defect.

    BFS runs from each assigned ring tile over *occupied* tiles until it reaches an
    unassigned core tile, then ownership is shifted one step along the path
    ``t -> n1 -> ... -> nk -> empty``: the empty tile takes ``n_k``'s geometry, each
    ``n_i`` takes ``n_{i-1}``'s, ``n_1`` takes the ring tile's, and the ring tile is
    freed.  Every geometry on the path loses one tile and gains one, so per-region
    tile counts are preserved exactly.

    Targets are unassigned core tiles *and* enclosed unassigned cells whatever their
    core status.  A cell ringed by assigned tiles is a visible hole even when its
    overlap with the study union falls below ``min_overlap_frac``; filling one from a
    protruding ring tile removes a hole and a tail at once, and leaves the empty-core
    count untouched.

    A relocation is applied only when the combined defect count (unassigned core tiles
    plus enclosed unassigned non-core cells) strictly decreases, the unassigned-core
    count on its own does not increase, and neither the per-geometry nor the per-group
    split count increases -- so the step can never regress any of these relative to
    leaving it off.  The ``max_hops`` bound keeps the search local; ``max_hops=0``
    disables the step and ``max_hops=1`` reproduces the plain adjacent-tile swap-back
    this generalises.
    """
    from collections import deque

    # Endpoints scored per ring tile.  The first (shortest) path is usually accepted;
    # the cap only bounds the pathological case where many are rejected in a row.
    max_candidate_paths = 20

    geom_units = np.arange(G, dtype=np.int32)

    def defect_holes(occupied: set[int]) -> set[int]:
        """Enclosed unassigned cells outside the core."""
        return _enclosed_unassigned(occupied, adj_list) - core_set

    def score(a: np.ndarray) -> tuple[int, int, int, int]:
        """(split groups, split geometries, empty core tiles, enclosed non-core holes)."""
        occupied_l = [t for t in range(len(a)) if a[t] >= 0]
        occupied = set(occupied_l)
        by_group = _count_split_units(a, occupied_l, adj_list, group_labels) if group_labels is not None else 0
        by_geom = _count_split_units(a, occupied_l, adj_list, geom_units)
        n_empty_core = len(core_set - occupied)
        n_enclosed = len(defect_holes(occupied))
        return by_group, by_geom, n_empty_core, n_enclosed

    def accept(after: tuple[int, int, int, int], before: tuple[int, int, int, int]) -> bool:
        """No split regression, and holes strictly fall -- core tiles first.

        The empty-core count can never rise here (the freed tile is always outside the
        core), so comparing ``(empty_core, empty_core + enclosed)`` lexicographically
        accepts every relocation onto an empty core tile exactly as before this
        widening, and accepts one onto an enclosed non-core cell -- which leaves the
        core count alone -- only when it removes a hole on balance.
        """
        return (
            after[0] <= before[0]
            and after[1] <= before[1]
            and (after[2], after[2] + after[3]) < (before[2], before[2] + before[3])
        )

    def candidate_paths(a: np.ndarray, start: int, empties: set[int]) -> list[list[int]]:
        """Shortest paths from ``start`` over occupied tiles to unassigned core tiles."""
        found: list[list[int]] = []
        prev: dict[int, int | None] = {start: None}
        queue: deque[tuple[int, int]] = deque([(start, 0)])
        while queue and len(found) < max_candidate_paths:
            u, d = queue.popleft()
            if d >= max_hops:
                continue
            for nb in adj_list[u]:
                if nb in prev:
                    continue
                prev[nb] = u
                if nb in empties:
                    path, p = [nb], u
                    while p != start:
                        path.append(p)
                        p = prev[p]  # type: ignore[assignment]
                    found.append(path[::-1])
                    if len(found) >= max_candidate_paths:
                        break
                elif a[nb] >= 0:
                    queue.append((nb, d + 1))
        return found

    moved = 0
    before = score(assignment)
    # Two passes.  The first targets empty core tiles only and so reproduces the
    # original swap-back move for move; the second additionally offers enclosed
    # non-core holes, and is a no-op on inputs that have none.  Splitting them keeps
    # the widening from perturbing the greedy order on inputs it should not touch.
    for include_enclosed in (False, True):
        progress = True
        while progress:
            progress = False
            occupied = {t for t in range(len(assignment)) if assignment[t] >= 0}
            empties = core_set - occupied
            if include_enclosed:
                empties = empties | defect_holes(occupied)
            if not empties:
                break
            for t in sorted(occupied - core_set):
                for path in candidate_paths(assignment, t, empties):
                    trial = assignment.copy()
                    chain = [t, *path]
                    for i in range(len(chain) - 1, 0, -1):
                        trial[chain[i]] = trial[chain[i - 1]]
                    trial[t] = -1
                    # Everything is guarded explicitly: an enclosed non-core target leaves
                    # the empty-core count unchanged, and freeing the ring tile can in
                    # principle open a fresh hole, so neither direction is safe by
                    # construction.  (Tested in test_mosaic_*.)
                    after = score(trial)
                    if accept(after, before):
                        assignment = trial
                        before = after
                        moved += 1
                        progress = True
                        break
                if progress:
                    break

    if show_progress:
        occupied = {t for t in range(len(assignment)) if assignment[t] >= 0}
        print(
            f"[mosaic]   ring swap-back: {moved} tile(s) relocated; "
            f"{len(occupied - core_set)} ring tile(s) and {len(core_set - occupied)} "
            f"unassigned core tile(s) remain"
        )
    return assignment


def _chain_swap_repair(
    assignment: np.ndarray,
    adj_list: list[list[int]],
    G: int,
    group_labels: np.ndarray | None,
    max_passes: int,
    show_progress: bool = False,
) -> np.ndarray:
    """Close remaining split units by swapping geometry ownership along tile chains.

    Runs :func:`~carto_flow.geo_utils.contiguity.repair_contiguity` over the
    *final* set of occupied tiles — after the extra-ring swap-back — so the
    repair sees the topology the metrics are computed on.  Because it only
    permutes which geometry owns each already-occupied tile, the occupied tile
    set (and therefore every region's tile count, and the absence of holes the
    ring swap-back achieved) is preserved exactly.

    The permutation is applied only when it strictly reduces the number of
    split units and increases neither the per-geometry nor the per-group count,
    so the repair can never make either metric worse than leaving it off.
    """
    from ....geo_utils.contiguity import repair_contiguity

    tiles = sorted(t for t in range(len(assignment)) if assignment[t] >= 0)
    if len(tiles) < 2:
        return assignment
    local = {t: i for i, t in enumerate(tiles)}
    adjacency = [{local[nb] for nb in adj_list[t] if nb in local} for t in tiles]
    geoms = [int(assignment[t]) for t in tiles]

    geom_units = np.arange(G, dtype=np.int32)
    units = [int(group_labels[g]) for g in geoms] if group_labels is not None else geoms

    def score(a: np.ndarray) -> tuple[int, int]:
        by_group = _count_split_units(a, tiles, adj_list, group_labels) if group_labels is not None else 0
        return by_group, _count_split_units(a, tiles, adj_list, geom_units)

    before = score(assignment)
    if before == (0, 0):
        return assignment

    # max_candidate_paths above the library default: the first chains found are
    # often rejected because rerouting them would split another unit, and the
    # accept-guard below makes a wider search free of downside.
    slot_of, _ = repair_contiguity(None, units, max_passes=max_passes, adjacency=adjacency, max_candidate_paths=60)

    repaired = assignment.copy()
    repaired[tiles] = -1
    for d, slot in enumerate(slot_of):
        repaired[tiles[int(slot)]] = geoms[d]

    after = score(repaired)
    accepted = after[0] <= before[0] and after[1] <= before[1] and after < before
    if show_progress:
        print(
            f"[mosaic]   chain-swap repair: (group, geom) splits {before} -> {after}"
            f"  {'accepted' if accepted else 'rejected'}"
        )
    return repaired if accepted else assignment


@dataclass
class HungarianOptions:
    """Cost-function parameters for mosaic Hungarian assignment.

    All cost terms are normalized to [0, 1] so the parameters are
    comparable across datasets.

    Parameters
    ----------
    distance_weight : float
        Weight on the normalized centroid-distance term.  Fix at 1.0 and
        tune *outside_penalty* relative to it.
    outside_penalty : float
        Weight on the outside-fraction penalty.  Tiles whose centroid lies
        outside the study union incur cost
        ``outside_penalty * outside_frac``.
    interior_bonus : float
        Connectivity bonus — subtracted from cost for tiles that are
        well-surrounded by other valid tiles (interior tiles), pushing
        unselected surplus toward the periphery.
    max_connectivity_iters : int
        Maximum number of iterative re-solve passes for connectivity
        repair (intra-region disconnections + inter-region gap bridging).
    disconnected_penalty_mult : float
        Per-iteration cost raise for intra-region disconnected tiles:
        ``cost_iter.max() x disconnected_penalty_mult``.
    gap_bridge_mult : float
        Per-iteration cost reduction for inter-region gap bridge candidates:
        ``cost_iter.max() x gap_bridge_mult``.
    disconnected_score_weight : int
        How many gap-pairs a single disconnected tile counts as in the
        *tie-breaking* score.  The repair loop first compares the number of
        split regions (or groups); this score only separates assignments that
        split the same number of them.  Only the ratio to the gap weight
        (fixed at 1) matters; higher values prioritise intra-region
        connectivity over inter-region adjacency.
    neighbor_weight : float
        Weight on the neighbor cost term. After each solve, penalizes placing
        geometry g's tiles far from the pool centroids of g's geographic
        neighbors. 0 disables. Default 0.3.
    neighbor_bfs : bool
        If True, measure neighbor proximity using BFS hop distance on the tile
        adjacency graph instead of Euclidean distance to the neighbor's tile
        pool centroid. More accurate for non-convex or tightly-packed regions
        (e.g. New England). Default False.
    swap_repair_passes : int
        Maximum passes of the chain-swap contiguity repair that runs on the
        final assignment, after the extra-ring swap-back.  It permutes which
        geometry owns each occupied tile along short chains, so tile counts
        and the occupied tile set are preserved exactly, and the result is
        kept only when it strictly reduces the number of split regions or
        groups without increasing either.  0 disables it.  Default 10.
    ring_swapback_max_hops : int
        How far the extra-ring swap-back may search for an unassigned core
        tile to pull a stranded ring tile into.  Ownership is shifted along
        the path, so tile counts are preserved and the move is kept only when
        it increases neither the split-region nor the split-group count.
        0 disables the swap-back; 1 restricts it to directly adjacent tiles.
        A larger reach closes more holes but can cost compactness, so the
        default is a deliberate balance.  An earlier default of 8 was set
        before the reach was measured across a range of inputs; do not lower
        it again without re-running that comparison.
    """

    distance_weight: float = 1.0
    outside_penalty: float = 1.0
    interior_bonus: float = 0.5
    max_connectivity_iters: int = 15
    disconnected_penalty_mult: float = 10.0
    gap_bridge_mult: float = 5.0
    disconnected_score_weight: int = 100
    neighbor_weight: float = 0.3
    neighbor_bfs: bool = False
    swap_repair_passes: int = 10
    ring_swapback_max_hops: int = 16

    def __post_init__(self) -> None:
        if self.max_connectivity_iters < 0:
            raise ValueError(f"max_connectivity_iters must be >= 0, got {self.max_connectivity_iters}")
        if self.ring_swapback_max_hops < 0:
            raise ValueError(f"ring_swapback_max_hops must be >= 0, got {self.ring_swapback_max_hops}")


@dataclass
class MosaicLayoutOptions:
    """Options for mosaic (tilegram) layout.

    Parameters
    ----------
    tiling : str or Tiling
        Tile shape. Default ``"hexagon"``.
    morph : bool
        Run flow cartogram to pre-morph geometries proportionally to tile
        counts before assignment. Default True.
    morph_options : MorphOptions or None
        Options for flow morphing. None → MorphOptions(n_iter=100).
    hungarian_options : HungarianOptions or None
        Assignment cost-function parameters. None → HungarianOptions() defaults.
    tile_size : float or None
        Explicit tile size; skips calibration if provided.
    spacing : float
        Gap between symbols as a fraction of tile size (0-1). 0 means symbols
        touch flat-edge to flat-edge; 0.05 matches GridBasedLayout's default.
        Default 0.0.
    extra_tile_rings : int
        Number of rings of adjacent tiles to add to each component's pool beyond
        those produced by calibration. Extra tiles are outside the study union
        (high ``outside_frac`` cost) so they act as reserve: selected only when
        interior tiles are exhausted by other cost pressures (e.g. high
        ``neighbor_weight``). Default 1.
    min_overlap_frac : float
        Minimum fraction of a tile's area that must intersect the study union
        for the tile to be counted as a core tile during calibration. Values
        below 0.5 capture tiles whose centroid falls outside the geometry
        (e.g. narrow peninsulas like Florida). Default 0.1.

    """

    tiling: object = "hexagon"  # Tiling | str, but avoid circular import
    morph: bool = True
    morph_options: object = None  # MorphOptions | None — lazy import
    hungarian_options: object = None  # HungarianOptions | None — lazy import
    tile_size: float | None = None
    spacing: float = 0.0
    extra_tile_rings: int = 1
    min_overlap_frac: float = 0.1

    def validate(self) -> None:
        """Validate options."""
        if self.tile_size is not None and self.tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {self.tile_size}")
        if not 0 <= self.spacing <= 1:
            raise ValueError(f"spacing must be between 0 and 1, got {self.spacing}")
        if self.extra_tile_rings < 0:
            raise ValueError(f"extra_tile_rings must be >= 0, got {self.extra_tile_rings}")
        if not 0.0 < self.min_overlap_frac <= 1.0:
            raise ValueError(f"min_overlap_frac must be in (0, 1], got {self.min_overlap_frac}")


class MosaicLayout(Layout):
    """Layout using mosaic (tilegram) assignment.

    Assigns each geometry an exact integer number of tiles from a regular
    tiling that covers the study area. Optionally pre-morphs geometries
    using a flow cartogram so their areas are proportional to tile counts.

    Parameters
    ----------
    options : MosaicLayoutOptions, optional
        Full options object. Defaults to MosaicLayoutOptions().
    **kwargs
        Individual option overrides.

    Examples
    --------
    >>> layout = MosaicLayout()  # 1 tile per geometry
    >>> layout = MosaicLayout(count_column="n_tiles")  # N tiles per geometry
    >>> layout = MosaicLayout(tiling="square", morph=False)

    """

    def __init__(self, options: MosaicLayoutOptions | None = None, /, **kwargs) -> None:
        if options is None:
            options = MosaicLayoutOptions()
        self._options = _apply_kwargs_to_options(options, kwargs)
        self._options.validate()

    def compute(self, data: LayoutData, show_progress: bool = True, save_history: bool = False) -> LayoutResult:
        """Run mosaic assignment and return result.

        Parameters
        ----------
        data : LayoutData
            Preprocessed layout data.
        show_progress : bool
            Display progress feedback during assignment.
        save_history : bool
            Not used for mosaic layout (kept for API compatibility).

        Returns
        -------
        LayoutResult
            Immutable layout result with appropriate canonical symbol.

        """
        import shapely
        from shapely.ops import unary_union
        from shapely.strtree import STRtree

        from ...tiling import resolve_tiling
        from ._assignment import hungarian_morphed_assignment
        from ._calibration import calibrate_tiling

        opts = self._options
        source_gdf = data.source_gdf
        geometries = list(source_gdf.geometry)
        G = len(geometries)

        counts = data.counts_G if data.counts_G is not None else np.ones(G, dtype=np.int32)
        sizes_G = data.sizes_G if data.sizes_G is not None else data.sizes
        components = data.components
        component_labels = data.component_labels
        if components is None or component_labels is None:  # pragma: no cover - always set
            raise ValueError("MosaicLayout needs LayoutData.components; build data with prepare_layout_data()")
        n_components = len(components)

        target_count = len(data.positions)  # = N = Σcounts

        # Step 1: Optional flow morphing
        if opts.morph and target_count > 0:
            from ....flow_cartogram.algorithm import morph_geometries
            from ....flow_cartogram.options import MorphOptions

            morph_opts = opts.morph_options or MorphOptions(n_iter=100, show_progress=show_progress)
            flow_result = morph_geometries(
                geometries,
                values=counts.astype(np.float64),
                options=morph_opts,
            )
            latest = flow_result.latest
            morphed = latest.geometry if latest is not None else None
            if morphed is None:  # pragma: no cover - morph always stores a snapshot
                raise RuntimeError("flow morph returned no morphed geometry")
            working_geometries = list(morphed)
        else:
            working_geometries = geometries

        working_geometries = [shapely.make_valid(g) for g in working_geometries]
        working_union = unary_union(working_geometries)

        comp_union_list = [unary_union([working_geometries[i] for i in geom_idx_list]) for geom_idx_list in components]
        comp_tree = STRtree(comp_union_list)

        # Step 3: Calibrate tiling
        tiling_obj = resolve_tiling(opts.tiling)
        setup = calibrate_tiling(
            tiling_obj,
            working_union.bounds,
            working_union,
            target_count,
            tile_size=opts.tile_size,
            buffer_rings=opts.extra_tile_rings,
            min_overlap_frac=opts.min_overlap_frac,
        )
        tiling_result = setup.tiling_result
        adj_list = setup.adj_list
        valid_tile_indices = setup.valid_tile_indices
        core_set = setup.core_set
        tile_size = setup.tile_size
        T = len(tiling_result.polygons)

        # Step 3b: Partition valid tiles into per-component pools
        tile_to_comp = np.full(T, -1, dtype=np.int32)
        for t in valid_tile_indices:
            tile_poly = tiling_result.polygons[t]
            candidate_comps = comp_tree.query(tile_poly, predicate="intersects")
            if len(candidate_comps) == 0:
                tile_centroid = tile_poly.centroid
                tile_to_comp[t] = min(
                    range(n_components),
                    key=lambda c: tile_centroid.distance(comp_union_list[c].centroid),
                )
            elif len(candidate_comps) == 1:
                tile_to_comp[t] = int(candidate_comps[0])
            else:
                best_c, best_area = -1, 0.0
                for c in candidate_comps:
                    area = tile_poly.intersection(comp_union_list[c]).area
                    if area > best_area:
                        best_area, best_c = area, c
                tile_to_comp[t] = best_c
        comp_tile_pools: list[list[int]] = [[] for _ in range(n_components)]
        for t in valid_tile_indices:
            comp_tile_pools[int(tile_to_comp[t])].append(t)

        # Expand each component's pool by extra_tile_rings of adjacent tiles.
        # These ring tiles lie outside the component union (high outside_frac cost)
        # and serve as reserve: used only under strong cost pressure (e.g. high
        # neighbor_weight), preventing zero-surplus situations after partitioning.
        if opts.extra_tile_rings > 0:
            for c in range(n_components):
                pool_set = set(comp_tile_pools[c])
                for _ in range(opts.extra_tile_rings):
                    ring = {nb for t in pool_set for nb in adj_list[t] if nb not in pool_set}
                    pool_set |= ring
                comp_tile_pools[c] = sorted(pool_set)

        # Step 3c: Build group labels for group_by mode.
        # Uses the G-level user grouping (group_ids_G), which is None unless
        # group_by was used; data.group_ids is N-level (one entry per symbol)
        # and must not be zipped against the G geometries. None does not mean
        # "no grouping": the assignment then treats each geometry as its own
        # group, so a geometry and its tiles still have to form one block.
        group_labels = None
        if data.group_ids_G is not None:
            raw_group_labels = data.group_ids_G.astype(np.int32)
            # Split groups at geographic component boundaries
            eff: dict[tuple[int, int], int] = {}
            effective_group_labels = np.empty(G, dtype=np.int32)
            next_id = int(raw_group_labels.max()) + 1 if len(raw_group_labels) > 0 else 0
            for i, (gl, cl) in enumerate(zip(raw_group_labels.tolist(), component_labels.tolist(), strict=False)):
                key = (int(gl), int(cl))
                if key not in eff:
                    eff[key] = int(gl) if cl == 0 else next_id
                    if cl != 0:
                        next_id += 1
                effective_group_labels[i] = eff[key]
            group_labels = effective_group_labels

        # Step 4: Per-component Hungarian assignment
        hopts = opts.hungarian_options or HungarianOptions()
        assignment = np.full(T, -1, dtype=np.int32)
        repair_passes = 0
        for c, geom_indices_c in enumerate(components):
            geom_indices_arr = np.array(geom_indices_c, dtype=np.intp)
            geom_c = [working_geometries[i] for i in geom_indices_c]
            counts_c = counts[geom_indices_arr]
            if group_labels is not None:
                gl_c = group_labels[geom_indices_arr]
                _, inv = np.unique(gl_c, return_inverse=True)
                group_labels_c = inv.astype(np.int32)
            else:
                group_labels_c = None
            pool_c = comp_tile_pools[c]
            if not pool_c:
                continue
            stats_c: dict = {}
            assignment_c = hungarian_morphed_assignment(
                tiling_result,
                geom_c,
                counts_c,
                pool_c,
                adj_list,
                core_set=core_set,
                working_union=comp_union_list[c],
                options=hopts,
                group_labels=group_labels_c,
                show_progress=show_progress,
                stats=stats_c,
            )
            repair_passes = max(repair_passes, int(stats_c.get("passes", 0)))
            for t in pool_c:
                local_g = int(assignment_c[t])
                if local_g >= 0:
                    assignment[t] = int(geom_indices_arr[local_g])

        # Post-process: swap extra-ring tiles back to unassigned core tiles where possible.
        # A ring tile the solver used sits outside the core and — since calibration matches
        # the core tile count to the tile budget — leaves a core tile empty one-for-one, so
        # the protruding tail and the hole are the same defect.  The relocation below walks
        # a BFS path of occupied tiles from the ring tile to the nearest empty core tile and
        # shifts ownership along it, which preserves every region's tile count exactly.
        if opts.extra_tile_rings > 0:
            assignment = _relocate_ring_tiles(
                assignment,
                adj_list,
                set(valid_tile_indices),
                G,
                group_labels,
                max_hops=hopts.ring_swapback_max_hops,
                show_progress=show_progress,
            )

        # Post-process: close any remaining split regions/groups by swapping
        # geometry ownership along tile chains.  Runs after the extra-ring
        # swap-back, which is itself a strong repair (it pulls ring tiles back
        # into the core and reconnects most satellites); repairing before it
        # would spend the chain swaps on splits the ring step removes anyway.
        if hopts.swap_repair_passes > 0:
            assignment = _chain_swap_repair(
                assignment, adj_list, G, group_labels, hopts.swap_repair_passes, show_progress
            )

        # Store tile index sets for visualization.
        # core_tile_indices: tiles whose centroid is inside the study union (= valid_tile_indices).
        # pool_tile_indices: full solver pool, including extra rings beyond the core.
        core_tile_indices = np.array(valid_tile_indices, dtype=np.intp)
        pool_tile_indices = np.array(sorted({t for pool in comp_tile_pools for t in pool}), dtype=np.intp)

        # Derive the set of tiles to use for all output steps from the actual assignment.
        # This correctly includes any extra ring tiles that the solver chose to use.
        output_tile_indices = sorted(t for t in range(T) if assignment[t] >= 0)

        # Step 5: Build output GeoDataFrames
        tiles_gdf, regions_gdf = _build_geodataframes(
            assignment,
            output_tile_indices,
            tiling_result,
            tile_size,
            counts,
            source_gdf,
            data.group_ids_G,
        )

        # Step 6: Build transforms (one per assigned tile)
        # assignment[t] is always a geometry index (0..G-1); no item-level indirection needed
        transforms, src_idx = _build_transforms(
            assignment,
            output_tile_indices,
            tiling_result,
            sizes_G,
            spacing=opts.spacing,
        )

        # Extract CRS
        crs = None
        if source_gdf.crs is not None:
            crs = source_gdf.crs.to_wkt()

        # Get canonical symbol
        canonical = tiling_obj.canonical_symbol()

        # geometry_positions: G-level original centroids
        geom_positions = data.geometry_positions if data.geometry_positions is not None else data.positions

        # Compute how many regions received their exact tile count
        tile_count_per_region = np.array(
            [len([t for t in output_tile_indices if assignment[t] == g]) for g in range(G)],
            dtype=np.intp,
        )
        regions_correct = int(np.sum(tile_count_per_region == counts))

        # Region-level topology of the final assignment: a region (and, with
        # group_by, a group) should occupy exactly one connected block.
        n_noncontiguous_regions = _count_split_units(
            assignment, output_tile_indices, adj_list, np.arange(G, dtype=np.int32)
        )
        n_split_groups = (
            _count_split_units(assignment, output_tile_indices, adj_list, group_labels)
            if group_labels is not None
            else 0
        )

        # Core tiles left empty.  Deliberately *not* folded into `converged`: that
        # would be a public behaviour change, and on inputs where a region genuinely
        # extends past the core there may be no legal relocation to close them.
        n_unassigned_core_tiles = int(sum(1 for t in valid_tile_indices if assignment[t] < 0))

        # Holes the core-only count cannot see: a lattice cell ringed by assigned tiles
        # but below `min_overlap_frac`, so never a core tile.  Counted regardless of core
        # status, which is the whole point -- see MosaicMetrics.
        n_enclosed_unassigned_tiles = len(_enclosed_unassigned({t for t in range(T) if assignment[t] >= 0}, adj_list))

        from ..layout_result import AlgorithmMetrics, MosaicLayoutResult

        metrics = AlgorithmMetrics(
            converged=regions_correct == G and n_noncontiguous_regions == 0 and n_split_groups == 0,
            iterations=repair_passes,
            final_overlaps=0,
            algorithm=MosaicMetrics(
                tiling=str(opts.tiling),
                tile_size=float(tile_size),
                n_components=n_components,
                regions_correct=regions_correct,
                regions_total=G,
                n_noncontiguous_regions=n_noncontiguous_regions,
                n_split_groups=n_split_groups,
                repair_passes=repair_passes,
                n_unassigned_core_tiles=n_unassigned_core_tiles,
                n_enclosed_unassigned_tiles=n_enclosed_unassigned_tiles,
            ),
        )

        assigned_tile_indices = np.array(output_tile_indices, dtype=np.intp)

        return MosaicLayoutResult(
            canonical_symbol=canonical,
            transforms=transforms,
            base_size=float(tiling_result.tile_size),
            positions=geom_positions,
            sizes=data.sizes,
            adjacency=data.adjacency,
            bounds=data.bounds,
            crs=crs,
            metrics=metrics,
            tiling_result=tiling_result,
            assignments=assigned_tile_indices,
            tiles_gdf=tiles_gdf,
            regions_gdf=regions_gdf,
            counts=counts.astype(np.intp),
            core_tile_indices=core_tile_indices,
            pool_tile_indices=pool_tile_indices,
            valid_mask=data.valid_mask,
            source_indices=src_idx,
            group_ids=data.group_ids_G[src_idx] if data.group_ids_G is not None else None,
        )


def _build_geodataframes(
    assignment: np.ndarray,
    valid_tile_indices: list[int],
    tiling_result,
    tile_size: float,
    counts: np.ndarray,
    source_gdf,
    group_ids_G: np.ndarray | None,
):
    """Build tiles_gdf and regions_gdf from the assignment.

    ``group_ids_G`` is the G-level user grouping (one entry per geometry) or
    None. With None, ``regions_gdf`` has one row per geometry; otherwise one
    row per group.
    """
    import geopandas as gpd
    import shapely
    from shapely.ops import unary_union as _uu

    G = len(source_gdf)

    # Tile-level GDF: geometry_id is item index from assignment
    tile_df = gpd.GeoDataFrame(
        {"geometry_id": assignment.tolist()},
        geometry=list(tiling_result.polygons),
        crs=source_gdf.crs,
    )
    tiles_gdf = tile_df.iloc[valid_tile_indices].reset_index(drop=True)

    # Region-level GDF: snap tile coordinates to avoid MultiPolygon artifacts
    _grid_size = tile_size * 1e-4
    _snapped_polygons = [shapely.set_precision(p, grid_size=_grid_size) for p in tiling_result.polygons]

    def _merge(tile_indices_for_region):
        if not tile_indices_for_region:
            return None
        polys = [_snapped_polygons[t] for t in tile_indices_for_region]
        return _uu(polys)

    if group_ids_G is not None:
        # group_by mode: one region per unique group
        unique_grp_ids = np.unique(group_ids_G)
        region_geoms = []
        tile_count_list = []
        target_count_list = []
        for gid in unique_grp_ids:
            gmask_set = set(np.where(group_ids_G == gid)[0].tolist())
            g_tile_indices = [t for t in valid_tile_indices if int(assignment[t]) in gmask_set]
            region_geoms.append(_merge(g_tile_indices))
            tile_count_list.append(len(g_tile_indices))
            target_count_list.append(int(np.sum(group_ids_G == gid)))
        regions_gdf = gpd.GeoDataFrame(
            {"tile_count": tile_count_list, "target_count": target_count_list, "group_id": unique_grp_ids.tolist()},
            geometry=region_geoms,
            crs=source_gdf.crs,
        )
    else:
        # Per-geometry mode: one region per geometry (assignment[t] is always a geometry index)
        region_geoms = []
        tile_count_list = []
        for g in range(G):
            g_tile_indices = [t for t in valid_tile_indices if assignment[t] == g]
            region_geoms.append(_merge(g_tile_indices))
            tile_count_list.append(len(g_tile_indices))
        regions_gdf = gpd.GeoDataFrame(
            {"tile_count": tile_count_list, "target_count": counts.tolist()},
            geometry=region_geoms,
            index=source_gdf.index,
            crs=source_gdf.crs,
        )

    return tiles_gdf, regions_gdf


def _build_transforms(
    assignment: np.ndarray,
    valid_tile_indices: list[int],
    tiling_result,
    sizes_G: np.ndarray,
    spacing: float = 0.0,
) -> tuple[list[Transform], np.ndarray]:
    """Build one Transform per assigned tile.

    assignment[t] is always a geometry index (0..G-1) from the Hungarian assignment.
    Returns transforms (one per tile) and geom_ids (tile → source geometry index).
    Scale encodes the geometry's relative size so within-tile symbols remain proportional.
    """
    max_size = float(np.max(sizes_G)) if len(sizes_G) > 0 else 1.0
    transforms = []
    geom_ids = []
    for t in valid_tile_indices:
        g = int(assignment[t])
        if g < 0:
            continue
        tf = tiling_result.transforms[t]
        scale = (float(sizes_G[g] / max_size) if max_size > 0 else 1.0) / (1 + spacing)
        transforms.append(
            Transform(
                position=tf.center,
                rotation=np.radians(tf.rotation),
                scale=scale,
                reflection=tf.flipped,
            )
        )
        geom_ids.append(g)
    return transforms, np.array(geom_ids, dtype=np.intp)
