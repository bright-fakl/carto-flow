"""Grid-based layout: GridBasedLayout and GridBasedLayoutOptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from ...options import SymbolOrientation, SymbolShape
from ..base import Layout, _apply_kwargs_to_options, _orient_tiling, _validate_lattice_options
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

    With ``assignment="energy"`` (the default) the assignment of regions to
    tiles minimizes one energy, in tile units (lengths in steps between
    neighboring tile centers)::

        E = origin_weight * (sum of the distances from centroid to tile)
          + neighbor_weight * (contact penalty)
          + connectivity_weight * (connectivity defects)
          + 10 * compactness * neighbor_weight * (pockets)

    - The distance is linear: every tile step costs ``origin_weight``.
    - The contact penalty is, for two neighboring regions, 0 on tiles sharing
      an edge, 0.2 on tiles sharing only a vertex and ``(h - 1)**2`` on tiles
      ``h`` steps apart; and 1 for every two regions that are not neighbors
      but sit on tiles sharing an edge.
    - A connectivity defect is an extra piece of a group of connected regions,
      two groups joined into one piece, or a free area enclosed by occupied
      tiles beyond the holes of the input geometries.
    - A pocket is a free tile with more than half of the largest number of
      edge neighbors of a tile occupied: a notch in the tile set.

    ``compactness`` (0 to 1) also restricts the tiles a region may use. The
    tiles are ordered by a packed growth from the middle of the centroids that
    always adds the free tile with the most grown neighbors, so every prefix
    of the order is one connected set without notches. The assignment may use
    the first ``n + (1 - compactness) * (m - n)`` tiles of ``m``: the whole
    lattice at 0, where the layout follows the geography as closely as the
    other terms allow, and exactly ``n`` packed tiles at 1, where the layout
    is a solid block without pockets.

    With the defaults, one tile step of distance costs as much as one missed
    contact, and one defect as much as 100 missed contacts, so the result is
    one connected set of tiles without holes whenever a single move, a shift
    chain or the move of a detached piece can reach it. The minimization
    starts from a refined Hungarian assignment on the allowed tiles, in which
    ``topology_weight`` also takes part, and then moves single regions to free
    tiles, swaps pairs of regions, fills holes by shift chains and moves
    detached pieces, accepting a move only when it lowers ``E``. Tiles of the
    lattice that no region needs therefore change the result only through
    the distances.

    With ``assignment="refine"`` the assignment is the refined Hungarian
    assignment itself, followed by ``fill_holes`` and ``fix_islands``. Its
    origin cost is the squared distance divided by the largest squared
    distance between a centroid and a tile, so it depends on the extent of
    the lattice.

    Parameters
    ----------
    tiling : Tiling or str
        Tiling to use. Can be a ``Tiling`` instance for full control, or a
        string shorthand: ``"square"``, ``"hexagon"``, ``"triangle"``,
        ``"quadrilateral"``. Default is ``"hexagon"``.
    grid_size : int, tuple, or "auto"
        Number of grid cells. "auto" creates ~2x regions cells.
    origin_weight : float
        Weight for assigning regions near their original centroids: the
        energy of one tile step of distance. Default: 0.5.
    neighbor_weight : float
        Weight for keeping adjacent regions close on the grid: the energy of
        one missed contact. Default: 0.5.
    topology_weight : float
        Weight for preserving relative neighbor orientations. With
        ``assignment="energy"`` it shapes the starting assignment only.
        Default: 0.5.
    compactness : float
        With ``assignment="energy"``: from 0 to 1, how packed the tile set
        is, through the allowed tiles and the pocket term above. With
        ``assignment="refine"``: weight of the distance from the grid center
        in the assignment cost. Default: 0.25.
    spacing : float
        Gap between symbols as fraction of cell size (0-1). Used when
        computing the maximum symbol size that fits within grid cells.
    fill_holes : bool
        If True, post-process the assignment to fill internal holes —
        unoccupied tiles surrounded by occupied tiles that don't correspond
        to geographic gaps in the original geometries. Islands and genuine
        geographic gaps (e.g. internal lakes) are preserved. Used with
        ``assignment="refine"`` only. Default: True.
    min_hole_fraction : float
        Minimum area of a geographic interior ring, as a fraction of one
        tile's area, for it to count as a genuine geographic gap. Rings
        smaller than this are treated as boundary artifacts and ignored.
        The geographic gaps are the holes that are not connectivity defects.
        Default: 0.5.
    fix_islands : bool
        If True, post-process the assignment to reassign regions that are
        disconnected from the main assignment cluster to tiles adjacent to
        the cluster. True geographic islands (regions not adjacent to any
        other region) are preserved. Used with ``assignment="refine"`` only.
        Default: True.
    assignment : {"energy", "refine"}
        How regions are assigned to tiles: ``"energy"`` minimizes the energy
        above; ``"refine"`` uses the refined Hungarian assignment followed by
        ``fill_holes`` and ``fix_islands``. Default: ``"energy"``.
    connectivity_weight : float
        Energy of one connectivity defect, with ``assignment="energy"``.
        Default: 50.0.
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
        Rotation of the tile lattice in degrees, counter-clockwise, about the
        center of the region that the tiles cover. The tiles cover the region
        whatever the angle. Default: 0.0.
    offset : tuple of float
        Shift of the tile lattice as ``(u, v)``, in multiples of the two
        translation vectors of the lattice (see
        :meth:`~carto_flow.symbol_cartogram.Tiling.with_offset`): for the
        square tiling the tile sides along the x and y axes, for the hexagon
        tiling two vectors between neighboring tile centers, 60 degrees
        apart. The lattice turns with ``rotation``. The offset is periodic: a
        component that changes by 1 gives the same tiles. Default:
        ``(0.0, 0.0)``, the lattice anchored at the center of the region.

    """

    tiling: object = "hexagon"  # Tiling | str, but avoid circular import
    grid_size: int | tuple[int, int] | Literal["auto"] = "auto"
    origin_weight: float = 0.5
    neighbor_weight: float = 0.5
    topology_weight: float = 0.5
    compactness: float = 0.25
    spacing: float = 0.05
    fill_holes: bool = True
    min_hole_fraction: float = 0.5
    fix_islands: bool = True
    assignment: Literal["energy", "refine"] = "energy"
    connectivity_weight: float = 50.0
    verbose: bool = False
    symbol_shape: SymbolShape | None = None
    symbol_orientation: SymbolOrientation = SymbolOrientation.UPRIGHT
    rotation: float = 0.0
    offset: tuple[float, float] = (0.0, 0.0)

    def validate(self) -> None:
        """Validate options."""
        _validate_lattice_options(self.rotation, self.offset)
        if self.origin_weight < 0:
            raise ValueError("origin_weight must be >= 0")
        if self.neighbor_weight < 0:
            raise ValueError("neighbor_weight must be >= 0")
        if self.topology_weight < 0:
            raise ValueError("topology_weight must be >= 0")
        if self.compactness < 0:
            raise ValueError("compactness must be >= 0")
        if self.assignment == "energy" and self.compactness > 1:
            raise ValueError("compactness must be between 0 and 1 with assignment='energy'")
        if not 0 <= self.spacing <= 1:
            raise ValueError("spacing must be between 0 and 1")
        if self.min_hole_fraction < 0:
            raise ValueError("min_hole_fraction must be >= 0")
        if self.assignment not in ("energy", "refine"):
            raise ValueError("assignment must be 'energy' or 'refine'")
        if self.connectivity_weight < 0:
            raise ValueError("connectivity_weight must be >= 0")


# Energy of a pocket at compactness 1, in missed contacts
_POCKET_SCALE = 10.0


def _centroid_extent(centroids: NDArray) -> float | None:
    """Squared diagonal of the bounding box of the centroids, or None for a single point."""
    span = np.ptp(np.asarray(centroids, dtype=float), axis=0) if len(centroids) else np.zeros(2)
    ext = float((span**2).sum())
    return ext if ext > 0 else None


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

    Notes
    -----
    The tile lattice is calibrated from the largest symbol, so
    ``size_normalization`` rescales the whole grid — lattice and symbols
    together — without changing how symbols are arranged. The layout
    therefore defaults to ``"max"``, which keeps the grid at the scale of
    the input geometries.

    """

    default_size_normalization: ClassVar[Literal["max", "total"]] = "max"

    def __init__(self, options: GridBasedLayoutOptions | None = None, /, **kwargs) -> None:
        if options is None:
            options = GridBasedLayoutOptions()
        self._options = _apply_kwargs_to_options(options, kwargs)
        self._options.validate()

    def _assign(self, centroids: NDArray, adjacency: NDArray, tiling_result, data: LayoutData) -> NDArray[np.intp]:
        """One tile per centroid, by the method set in ``assignment``."""
        from ._energy import GridEnergy
        from ._placement import _geographic_holes, assign_to_grid_hungarian

        opts = self._options
        energy = opts.assignment == "energy"
        centers, tile_adj, vert_adj = tiling_result.centers, tiling_result.adjacency, tiling_result.vertex_adjacency
        allowed = None
        if energy and opts.compactness > 0:
            from ._compact import candidate_tiles

            allowed = candidate_tiles(centroids, centers, tile_adj, opts.compactness)
            sub = np.ix_(allowed, allowed)
            centers, tile_adj = centers[allowed], tile_adj.tocsr()[sub]
            vert_adj = vert_adj.tocsr()[sub] if vert_adj is not None else None
        start = assign_to_grid_hungarian(
            centroids=centroids,
            grid_centers=centers,
            adjacency=adjacency,
            tile_adjacency=tile_adj,
            vertex_adjacency=vert_adj,
            origin_weight=opts.origin_weight,
            neighbor_weight=opts.neighbor_weight,
            topology_weight=opts.topology_weight,
            compactness=0.0 if energy else opts.compactness,
            damping=0.5 if energy else 1.0,
            origin_scale=_centroid_extent(centroids) if energy else None,
        )
        if not energy:
            return start
        allowed_holes = _geographic_holes(
            list(data.source_gdf.geometry), tiling_result.adjacency, tiling_result.centers, opts.min_hole_fraction
        )[0]
        model = GridEnergy(
            centroids,
            centers,
            adjacency,
            tile_adj,
            vert_adj,
            distance_weight=opts.origin_weight,
            contact_weight=opts.neighbor_weight,
            connectivity_weight=opts.connectivity_weight,
            allowed_holes=allowed_holes,
            pocket_weight=_POCKET_SCALE * opts.compactness * opts.neighbor_weight,
        )
        out = model.descend(start)
        return out if allowed is None else allowed[out]

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
            bfs_expand_assignments,
            fill_internal_holes,
        )

        # Resolve tiling
        tiling = _orient_tiling(resolve_tiling(self._options.tiling), self._options.rotation, self._options.offset)

        # Get canonical symbol for this tiling
        canonical = tiling.canonical_symbol()

        # Convert area-equivalent sizes to native half-extents
        # data.sizes are area-equivalent (circle radii)
        # native_sizes are the half-extents for rendering
        native_sizes = data.sizes * canonical.area_factor

        # Compute tile_size from largest native size BEFORE spacing scale-down
        # This keeps tile size purely geographic (based on unit cell area)
        max_native = float(np.max(native_sizes)) if len(native_sizes) > 0 else 1.0
        if max_native <= 0:
            # Every symbol is zero-sized (all sizing values are zero). The
            # lattice still needs a cell size, so fall back to the unit cell.
            max_native = float(np.sqrt(data.mean_area / np.pi)) * canonical.area_factor
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
            anchor_assignments = self._assign(geometry_positions, adjacency_G, tiling_result, data)

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
            assignments = self._assign(data.positions, data.adjacency, tiling_result, data)

        # Post-process: fill holes if requested
        refine = self._options.assignment == "refine"
        if refine and self._options.fill_holes:
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
        if refine and self._options.fix_islands:
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
