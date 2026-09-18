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
    """Algorithm-specific metrics for MosaicLayout."""

    tiling: str = ""
    tile_size: float = 0.0
    n_components: int = 0
    regions_correct: int = 0
    regions_total: int = 0


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
        convergence score.  Only the ratio to the gap weight (fixed at 1)
        matters; higher values prioritise intra-region connectivity over
        inter-region adjacency.
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
        Maximum passes for the swap-based repair stage (Stage 1: contiguity,
        Stage 2: adjacency) run after the iterative Hungarian loop.
        Default 0 (disabled).
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
    swap_repair_passes: int = 0

    def __post_init__(self) -> None:
        if self.max_connectivity_iters < 0:
            raise ValueError(f"max_connectivity_iters must be >= 0, got {self.max_connectivity_iters}")


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
        # and must not be zipped against the G geometries.
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
            )
            for t in pool_c:
                local_g = int(assignment_c[t])
                if local_g >= 0:
                    assignment[t] = int(geom_indices_arr[local_g])

        # Post-process: swap extra-ring tiles back to unassigned core tiles where possible.
        # Extra-ring tiles that ended up assigned create visual holes (gaps inside the core
        # pool). Greedily replace each extra-ring tile with an adjacent unassigned core tile
        # of the same geometry to eliminate holes without breaking tile counts.
        if opts.extra_tile_rings > 0:
            valid_set = set(valid_tile_indices)
            unassigned_core = {t for t in valid_tile_indices if assignment[t] < 0}
            changed = True
            while changed:
                changed = False
                for t in [t for t in range(T) if assignment[t] >= 0 and t not in valid_set]:
                    g = int(assignment[t])
                    for nb in adj_list[t]:
                        if nb in unassigned_core:
                            assignment[nb] = g
                            assignment[t] = -1
                            unassigned_core.discard(nb)
                            unassigned_core.discard(t)
                            changed = True
                            break

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

        from ..layout_result import AlgorithmMetrics, MosaicLayoutResult

        metrics = AlgorithmMetrics(
            converged=regions_correct == G,
            iterations=hopts.max_connectivity_iters,
            final_overlaps=0,
            algorithm=MosaicMetrics(
                tiling=str(opts.tiling),
                tile_size=float(tile_size),
                n_components=n_components,
                regions_correct=regions_correct,
                regions_total=G,
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
