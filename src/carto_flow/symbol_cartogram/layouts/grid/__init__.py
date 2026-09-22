"""Grid-based layout: GridBasedLayout and GridBasedLayoutOptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from ...options import SymbolOrientation, SymbolShape
from ..base import Layout, _apply_kwargs_to_options
from ..data_prep import LayoutData
from ..layout_result import LayoutResult, Transform

if TYPE_CHECKING:
    pass

__all__ = ["GridBasedLayout", "GridBasedLayoutOptions", "GridMetrics"]


@dataclass
class GridMetrics:
    """Algorithm-specific metrics for GridBasedLayout."""

    tiling: str = ""
    tile_size: float = 0.0
    n_tiles: int = 0


@dataclass
class GridBasedLayoutOptions:
    """Options for grid-based placement.

    The assignment cost function combines four terms, harmonized with the
    ``TopologyPreservingSimulator`` force terminology:

    - **origin_weight**: Preference for assigning regions near their
      original centroid positions (analogous to origin attraction force).
    - **neighbor_weight**: Keep adjacent regions close on the grid
      (analogous to neighbor tangency force).
    - **topology_weight**: Preserve the relative direction between
      neighbors (analogous to angular topology force).
    - **compactness**: Prefer central/compact placement on the grid
      (analogous to global centroid attraction).

    Parameters
    ----------
    tiling : Tiling or str
        Tiling to use. Can be a ``Tiling`` instance for full control, or a
        string shorthand: ``"square"``, ``"hexagon"``, ``"triangle"``,
        ``"quadrilateral"``. Default is ``"hexagon"``.
    grid_size : int, tuple, or "auto"
        Number of grid cells. "auto" creates ~2x regions cells.
    origin_weight : float
        Weight for assigning regions near their original centroids.
        Default: 1.0.
    neighbor_weight : float
        Weight for keeping adjacent regions close on the grid.
        Default: 0.3.
    topology_weight : float
        Weight for preserving relative neighbor orientations.
        Default: 0.0.
    compactness : float
        Weight for compact/central placement. Default: 0.0.
    spacing : float
        Gap between symbols as fraction of cell size (0-1). Used when
        computing the maximum symbol size that fits within grid cells.
    fill_holes : bool
        If True, post-process the assignment to fill internal holes —
        unoccupied tiles surrounded by occupied tiles that don't correspond
        to geographic gaps in the original geometries. Islands and genuine
        geographic gaps (e.g. internal lakes) are preserved. Default: False.
    min_hole_fraction : float
        Minimum area of a geographic interior ring, as a fraction of one
        tile's area, for it to count as a genuine geographic gap. Rings
        smaller than this are treated as boundary artifacts and ignored.
        Default: 0.5.
    fix_islands : bool
        If True, post-process the assignment to reassign regions that are
        disconnected from the main assignment cluster to tiles adjacent to
        the cluster. True geographic islands (regions not adjacent to any
        other region) are preserved. Default: False.
    verbose : bool
        If True, print diagnostic information during fill_holes and
        fix_islands processing. Default: False.
    symbol_shape : SymbolShape or None
        Target symbol shape. When set, the grid algorithm computes
        grid-appropriate sizes to prevent overlap. None skips adjustment.
    symbol_orientation : SymbolOrientation
        How symbols are oriented relative to their tile. ``UPRIGHT`` keeps
        symbols axis-aligned; ``WITH_TILE`` rotates/flips with the tile.
    rotation : float
        Rotation angle for the entire tiling grid in degrees (counter-clockwise).
        Default: 0.0 (no rotation).

    """

    tiling: object = "hexagon"  # Tiling | str, but avoid circular import
    grid_size: int | tuple[int, int] | Literal["auto"] = "auto"
    origin_weight: float = 0.5
    neighbor_weight: float = 0.5
    topology_weight: float = 0.5
    compactness: float = 0.1
    spacing: float = 0.05
    fill_holes: bool = True
    min_hole_fraction: float = 0.5
    fix_islands: bool = True
    verbose: bool = False
    symbol_shape: SymbolShape | None = None
    symbol_orientation: SymbolOrientation = SymbolOrientation.UPRIGHT
    rotation: float = 0.0

    def validate(self) -> None:
        """Validate options."""
        if self.origin_weight < 0:
            raise ValueError("origin_weight must be >= 0")
        if self.neighbor_weight < 0:
            raise ValueError("neighbor_weight must be >= 0")
        if self.topology_weight < 0:
            raise ValueError("topology_weight must be >= 0")
        if self.compactness < 0:
            raise ValueError("compactness must be >= 0")
        if not 0 <= self.spacing <= 1:
            raise ValueError("spacing must be between 0 and 1")
        if self.min_hole_fraction < 0:
            raise ValueError("min_hole_fraction must be >= 0")


class GridBasedLayout(Layout):
    """Layout from grid-based assignment.

    Returns LayoutResult with appropriate Symbol based on tiling type.

    Parameters
    ----------
    options : GridBasedLayoutOptions, optional
        Full options object (positional only). Defaults to GridBasedLayoutOptions().
    **kwargs
        Individual option overrides applied on top of *options*.
        Raises TypeError for unrecognized names.

    Examples
    --------
    >>> layout = GridBasedLayout(tiling="hexagon")
    >>> layout = GridBasedLayout(my_opts, neighbor_weight=0.5)

    """

    def __init__(self, options: GridBasedLayoutOptions | None = None, /, **kwargs) -> None:
        if options is None:
            options = GridBasedLayoutOptions()
        self._options = _apply_kwargs_to_options(options, kwargs)
        self._options.validate()

    def _compute(self, data: LayoutData, show_progress: bool = True, save_history: bool = False) -> LayoutResult:
        """Run grid assignment and return result.

        Parameters
        ----------
        data : LayoutData
            Preprocessed layout data.
        show_progress : bool
            Display progress feedback during placement.
        save_history : bool
            Record position snapshots per iteration.

        Returns
        -------
        LayoutResult
            Immutable layout result with appropriate canonical symbol.

        """
        from ...tiling import resolve_tiling
        from ._placement import (
            _fix_island_assignments,
            assign_to_grid_hungarian,
            bfs_expand_assignments,
            fill_internal_holes,
        )

        # Resolve tiling
        tiling = resolve_tiling(self._options.tiling)

        # Get canonical symbol for this tiling
        canonical = tiling.canonical_symbol()

        # Convert area-equivalent sizes to native half-extents
        # data.sizes are area-equivalent (circle radii)
        # native_sizes are the half-extents for rendering
        native_sizes = data.sizes * canonical.area_factor

        # Compute tile_size from largest native size BEFORE spacing scale-down
        # This keeps tile size purely geographic (based on unit cell area)
        max_native = float(np.max(native_sizes)) if len(native_sizes) > 0 else 1.0
        tile_size = tiling.tile_size_for_symbol_size(max_native, spacing=0)

        # Apply spacing by scaling down symbols (not by increasing tile size)
        # This makes spacing relative to final symbol size: spacing=1 means gap equals symbol size
        spacing = self._options.spacing
        effective_native = native_sizes / (1 + spacing)

        # Expand bounds if needed to ensure surplus of tiles for optimal assignment
        # Each tile has area ~ tile_size², so we need bounds_area >= n * tile_size²
        # Use 2x surplus (like old code: n_tiles=max(n, int(n * 2)))
        n = len(data.positions)
        minx, miny, maxx, maxy = data.bounds
        bounds_width = maxx - minx
        bounds_height = maxy - miny
        bounds_area = bounds_width * bounds_height
        min_area_needed = n * tile_size * tile_size * 2  # 2x surplus for flexibility

        if bounds_area < min_area_needed:
            # Scale bounds to fit at least n tiles while preserving aspect ratio
            scale_factor = np.sqrt(min_area_needed / bounds_area)
            cx = (minx + maxx) / 2
            cy = (miny + maxy) / 2
            half_w = bounds_width * scale_factor / 2
            half_h = bounds_height * scale_factor / 2
            bounds = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
        else:
            bounds = data.bounds

        # Generate tiling with explicit tile_size
        tiling_result = tiling.generate(
            bounds=bounds,
            tile_size=tile_size,
        )

        # Apply rotation if specified
        if self._options.rotation != 0.0:
            tiling_result = tiling_result.rotate(self._options.rotation)

        # Run Hungarian assignment
        if data.source_indices is not None:
            # Two-phase: G-level anchor assignment + BFS tile expansion.
            # Running Hungarian at N level (N = G x K) would be O(N²xM) in pure
            # Python and degenerate (duplicate centroids for within-group items).
            geometry_positions = data.geometry_positions
            if geometry_positions is None:  # pragma: no cover - set with source_indices
                raise ValueError("geometry_positions is required when source_indices is set")
            G_full = len(geometry_positions)
            counts: NDArray[np.int32] = (
                data.counts_G
                if data.counts_G is not None
                else np.bincount(data.source_indices, minlength=G_full).astype(np.int32)
            )
            first_items = np.concatenate([[0], np.cumsum(counts[:-1])]).astype(np.intp)
            # Recover GxG adjacency: cross-block entries in the expanded matrix
            # are identical to the original GxG values.
            adjacency_G = data.adjacency[np.ix_(first_items, first_items)]

            # Phase 1: G-level Hungarian anchor placement
            anchor_assignments = assign_to_grid_hungarian(
                centroids=geometry_positions,
                grid_centers=tiling_result.centers,
                adjacency=adjacency_G,
                tile_adjacency=tiling_result.adjacency,
                vertex_adjacency=tiling_result.vertex_adjacency,
                origin_weight=self._options.origin_weight,
                neighbor_weight=self._options.neighbor_weight,
                topology_weight=self._options.topology_weight,
                compactness=self._options.compactness,
            )

            # Phase 2: BFS expansion — each geometry grows from its anchor
            geom_tiles = bfs_expand_assignments(
                anchor_assignments,
                counts,
                tiling_result.adjacency,
                tiling_result.centers,
            )

            # Flatten to N-level (item k → tile_idx) in source_indices order
            geom_counters = np.zeros(G_full, dtype=np.intp)
            assignments: np.ndarray = np.empty(len(data.source_indices), dtype=np.intp)
            for k, g in enumerate(data.source_indices.tolist()):
                assignments[k] = geom_tiles[g][geom_counters[g]]
                geom_counters[g] += 1
        else:
            assignments = assign_to_grid_hungarian(
                centroids=data.positions,
                grid_centers=tiling_result.centers,
                adjacency=data.adjacency,
                tile_adjacency=tiling_result.adjacency,
                vertex_adjacency=tiling_result.vertex_adjacency,
                origin_weight=self._options.origin_weight,
                neighbor_weight=self._options.neighbor_weight,
                topology_weight=self._options.topology_weight,
                compactness=self._options.compactness,
            )

        # Post-process: fill holes if requested
        if self._options.fill_holes:
            assignments = fill_internal_holes(
                assignments,
                tiling_result.adjacency,
                data.positions,
                tiling_result.centers,
                list(data.source_gdf.geometry),
                min_hole_fraction=self._options.min_hole_fraction,
                verbose=self._options.verbose,
            )

        # Post-process: fix islands if requested
        if self._options.fix_islands:
            assignments = _fix_island_assignments(
                assignments,
                tiling_result.adjacency,
                data.positions,
                tiling_result.centers,
                data.adjacency,
                verbose=self._options.verbose,
            )

        # Compute base_native as average effective native size for proportional scaling
        base_native = float(np.mean(effective_native)) if len(effective_native) > 0 else 1.0

        # Create transforms with tile rotations/reflections and proportional scales
        transforms = []
        for i, tile_idx in enumerate(assignments):
            tile_transform = tiling_result.transforms[tile_idx]
            # Scale is proportional to effective native size relative to base_native
            scale = float(effective_native[i] / base_native) if base_native > 0 else 1.0
            transforms.append(
                Transform(
                    position=tile_transform.center,
                    rotation=np.radians(tile_transform.rotation),
                    reflection=tile_transform.flipped,
                    scale=scale,
                ),
            )

        # Extract CRS from source_gdf (use WKT to preserve projection info)
        crs = None
        if data.source_gdf.crs is not None:
            crs = data.source_gdf.crs.to_wkt()

        from ..layout_result import AlgorithmMetrics, GridLayoutResult

        metrics = AlgorithmMetrics(
            converged=True,
            iterations=1,
            final_overlaps=0,
            algorithm=GridMetrics(
                tiling=str(self._options.tiling),
                tile_size=float(tiling_result.tile_size),
                n_tiles=len(tiling_result.polygons),
            ),
        )

        return GridLayoutResult(
            canonical_symbol=canonical,
            transforms=transforms,
            base_size=base_native,
            positions=data.geometry_positions if data.geometry_positions is not None else data.positions,
            sizes=data.sizes,
            adjacency=data.adjacency,
            bounds=data.bounds,
            crs=crs,
            metrics=metrics,
            tiling_result=tiling_result,
            assignments=assignments,
            valid_mask=data.valid_mask,
            source_indices=data.source_indices,
            group_ids=data.group_ids,
        )
