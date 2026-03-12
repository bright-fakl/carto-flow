"""Layout algorithms for symbol cartograms.

This module defines the Layout ABC and concrete layout implementations
that compute positions and transforms, returning immutable LayoutResult.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .data_prep import LayoutData
from .layout_result import LayoutResult, Transform
from .options import (
    CentroidLayoutOptions,
    CirclePackingLayoutOptions,
    CirclePhysicsLayoutOptions,
    GridBasedLayoutOptions,
)
from .symbols import CircleSymbol


def _apply_kwargs_to_options(options: Any, kwargs: dict) -> Any:
    """Apply kwargs to a dataclass options instance. Raises on unknown keys.

    Parameters
    ----------
    options : dataclass instance
        Options to update.
    kwargs : dict
        Keyword arguments to apply.

    Returns
    -------
    dataclass instance
        Updated options (new instance, original unchanged).

    Raises
    ------
    TypeError
        If kwargs contains unknown keys.

    """
    if not kwargs:
        return options
    unknown = {k for k in kwargs if not hasattr(options, k)}
    if unknown:
        valid_fields = [f.name for f in dataclasses.fields(options)]
        raise TypeError(f"Unknown option(s): {', '.join(sorted(unknown))}. Valid options: {', '.join(valid_fields)}")
    return dataclasses.replace(options, **kwargs)


# ---------------------------------------------------------------------------
# Layout Registry
# ---------------------------------------------------------------------------

_LAYOUT_REGISTRY: dict[str, type[Layout]] = {}


def register_layout(name: str, cls: type[Layout]) -> None:
    """Register a layout class by name.

    Parameters
    ----------
    name : str
        Name to register under.
    cls : type[Layout]
        Layout class to register.

    """
    _LAYOUT_REGISTRY[name] = cls


def get_layout(name: str) -> Layout:
    """Instantiate a registered layout by name.

    Parameters
    ----------
    name : str
        Registered layout name.

    Returns
    -------
    Layout
        New layout instance.

    Raises
    ------
    ValueError
        If name is not registered.

    """
    if name not in _LAYOUT_REGISTRY:
        valid = ", ".join(sorted(_LAYOUT_REGISTRY.keys()))
        raise ValueError(f"Unknown layout {name!r}. Valid layouts: {valid}")
    return _LAYOUT_REGISTRY[name]()


# ---------------------------------------------------------------------------
# Layout ABC
# ---------------------------------------------------------------------------


class Layout(ABC):
    """Abstract base for layout algorithms.

    Layouts compute positions and transforms, returning immutable LayoutResult.
    """

    @abstractmethod
    def compute(self, data: LayoutData, show_progress: bool = True, save_history: bool = False) -> LayoutResult:
        """Run layout algorithm and return immutable result.

        Parameters
        ----------
        data : LayoutData
            Preprocessed data from prepare_layout_data().
        show_progress : bool
            Display progress feedback during placement.
        save_history : bool
            Record position snapshots per iteration.

        Returns
        -------
        LayoutResult
            Immutable result with canonical symbol and transforms.

        """
        ...


# ---------------------------------------------------------------------------
# PhysicsBasedLayout
# ---------------------------------------------------------------------------


def _build_physics_layout_result(
    positions: NDArray[np.floating],
    info: dict[str, Any],
    history: list[NDArray[np.floating]] | None,
    data: LayoutData,
) -> LayoutResult:
    """Build LayoutResult from physics simulation output."""
    # Compute base_size as average size
    base_size = float(np.mean(data.sizes))

    # Create transforms (position + scale, no rotation/reflection for physics)
    transforms = [
        Transform(
            position=(float(positions[i, 0]), float(positions[i, 1])),
            scale=float(data.sizes[i] / base_size) if base_size > 0 else 1.0,
        )
        for i in range(len(positions))
    ]

    # Extract CRS from source_gdf (use WKT to preserve projection info)
    crs = None
    if data.source_gdf.crs is not None:
        # Use to_wkt() to preserve full projection information
        crs = data.source_gdf.crs.to_wkt()

    # Create simulation history if available
    from .result import SimulationHistory

    sim_history = None
    if history is not None:
        sim_history = SimulationHistory(positions=history)

    return LayoutResult(
        canonical_symbol=CircleSymbol(),
        transforms=transforms,
        base_size=base_size,
        positions=data.positions,
        sizes=data.sizes,
        adjacency=data.adjacency,
        bounds=data.bounds,
        crs=crs,
        algorithm_info={"type": "physics", "info": info},
        simulation_history=sim_history,
        valid_mask=data.valid_mask,
    )


class CirclePhysicsLayout(Layout):
    """Layout using velocity-based physics simulation.

    Two-phase approach:
    1. Separation phase: Strong repulsion until overlaps resolved
    2. Settling phase: Gentle attraction while maintaining separation

    Parameters
    ----------
    options : CirclePhysicsLayoutOptions, optional
        Full options object. Defaults to CirclePhysicsLayoutOptions().
    **kwargs
        Individual option overrides.

    """

    def __init__(self, options: CirclePhysicsLayoutOptions | None = None, /, **kwargs) -> None:
        if options is None:
            options = CirclePhysicsLayoutOptions()
        self._options = _apply_kwargs_to_options(options, kwargs)
        self._options.validate()

    def compute(self, data: LayoutData, show_progress: bool = True, save_history: bool = False) -> LayoutResult:
        """Run physics simulation and return result.

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
            Immutable layout result with CircleSymbol as canonical.

        """
        from .placement import CirclePhysicsSimulator

        sim = CirclePhysicsSimulator(
            positions=data.positions,
            radii=data.sizes,
            adjacency=data.adjacency,
            spacing=self._options.spacing,
            compactness=self._options.compactness,
            topology_weight=self._options.topology_weight,
            damping=self._options.damping,
            dt=self._options.dt,
            max_velocity=self._options.max_velocity,
            k_repel=self._options.k_repel,
            k_attract=self._options.k_attract,
        )
        positions, info, history = sim.run(
            max_iterations=self._options.max_iterations,
            tolerance=self._options.convergence_tolerance,
            show_progress=show_progress,
            save_history=save_history,
        )
        return _build_physics_layout_result(positions, info, history, data)


class CirclePackingLayout(Layout):
    """Layout using two-stage circle packing simulation.

    Two-stage approach:
    1. Overlap Resolution: Global expansion + overlap projection
    2. Circle Packing: Force-based with contact reaction constraint

    This algorithm preserves topology better than CirclePhysicsLayout
    by using distance-gated angular forces and contact reaction.

    Parameters
    ----------
    options : CirclePackingLayoutOptions, optional
        Full options object. Defaults to CirclePackingLayoutOptions().
    **kwargs
        Individual option overrides.

    """

    def __init__(self, options: CirclePackingLayoutOptions | None = None, /, **kwargs) -> None:
        if options is None:
            options = CirclePackingLayoutOptions()
        self._options = _apply_kwargs_to_options(options, kwargs)
        self._options.validate()

    def compute(self, data: LayoutData, show_progress: bool = True, save_history: bool = False) -> LayoutResult:
        """Run circle packing simulation and return result.

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
            Immutable layout result with CircleSymbol as canonical.

        """
        from .placement import TopologyPreservingSimulator

        sim = TopologyPreservingSimulator(
            positions=data.positions,
            radii=data.sizes,
            adjacency=data.adjacency,
            spacing=self._options.spacing,
            compactness=self._options.compactness,
            topology_weight=self._options.topology_weight,
            overlap_tolerance=self._options.overlap_tolerance,
            expansion_max_iterations=self._options.expansion_max_iterations,
            max_expansion_factor=self._options.max_expansion_factor,
            topology_gate_distance=self._options.topology_gate_distance,
            neighbor_weight=self._options.neighbor_weight,
            origin_weight=self._options.origin_weight,
            force_mode=self._options.force_mode.value,
            contact_tolerance=self._options.contact_tolerance,
            contact_iterations=self._options.contact_iterations,
            max_step=self._options.max_step,
            contact_transfer_ratio=self._options.contact_transfer_ratio,
            contact_elasticity=self._options.contact_elasticity,
            size_sensitivity=self._options.size_sensitivity,
            global_step_fraction=self._options.global_step_fraction,
            local_step_fraction=self._options.local_step_fraction,
            overlap_projection_iters=self._options.overlap_projection_iters,
            step_smoothing_window=self._options.step_smoothing_window,
            convergence_window=self._options.convergence_window,
            adaptive_ema=self._options.adaptive_ema,
        )
        positions, info, history = sim.run(
            max_iterations=self._options.max_iterations,
            tolerance=self._options.convergence_tolerance,
            show_progress=show_progress,
            save_history=save_history,
        )
        return _build_physics_layout_result(positions, info, history, data)


# Register physics layouts
register_layout("physics", CirclePhysicsLayout)
register_layout("packing", CirclePackingLayout)
register_layout("topology", CirclePackingLayout)  # Backward compat


# ---------------------------------------------------------------------------
# GridBasedLayout
# ---------------------------------------------------------------------------


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

    def compute(self, data: LayoutData, show_progress: bool = True, save_history: bool = False) -> LayoutResult:
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
        # Import here to avoid circular import
        from .placement import (
            _fix_island_assignments,
            assign_to_grid_hungarian,
            fill_internal_holes,
        )
        from .tiling import resolve_tiling

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
            # Use to_wkt() to preserve full projection information
            crs = data.source_gdf.crs.to_wkt()

        return LayoutResult(
            canonical_symbol=canonical,
            transforms=transforms,
            base_size=base_native,  # Average native half-extent for proportional scaling
            positions=data.positions,
            sizes=data.sizes,
            adjacency=data.adjacency,
            bounds=data.bounds,
            crs=crs,
            algorithm_info={
                "type": "grid",
                "tiling": str(self._options.tiling),
                "n_tiles": len(tiling_result.polygons),
                "tile_size": tiling_result.tile_size,
                "tiling_result": tiling_result,  # Store for visualization
                "assignments": assignments,  # Store for visualization
            },
            valid_mask=data.valid_mask,
        )


# Register grid layout
register_layout("grid", GridBasedLayout)


# ---------------------------------------------------------------------------
# CentroidLayout
# ---------------------------------------------------------------------------


class CentroidLayout(Layout):
    """Layout that places symbols at geometry centroids.

    Places symbols at the centroids of the original geometries, with optional
    local overlap removal.

    Returns LayoutResult with CircleSymbol as canonical symbol.

    Parameters
    ----------
    options : CentroidLayoutOptions, optional
        Full options object. Defaults to CentroidLayoutOptions().
    **kwargs
        Individual option overrides applied on top of *options*.

    Examples
    --------
    >>> layout = CentroidLayout()  # Default: remove overlap
    >>> layout = CentroidLayout(remove_overlap=False)  # Just centroids
    >>> layout = CentroidLayout(spacing=0.1)  # Larger gaps

    """

    def __init__(self, options: CentroidLayoutOptions | None = None, /, **kwargs) -> None:
        if options is None:
            options = CentroidLayoutOptions()
        self._options = _apply_kwargs_to_options(options, kwargs)
        self._options.validate()

    def compute(self, data: LayoutData, show_progress: bool = True, save_history: bool = False) -> LayoutResult:
        """Place symbols at centroids with optional overlap removal.

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
            Immutable layout result with CircleSymbol as canonical.

        """
        from .placement import resolve_circle_overlaps

        positions = data.positions.copy()

        if self._options.remove_overlap:
            positions, info = resolve_circle_overlaps(
                positions=positions,
                radii=data.sizes,
                spacing=self._options.spacing,
                max_iterations=self._options.max_iterations,
                overlap_tolerance=self._options.overlap_tolerance,
                global_step_fraction=0.0,  # Disable global expansion - only local overlap resolution
            )
        else:
            info = {"iterations": 0, "final_max_overlap": 0.0}

        # Compute base_size as average size
        base_size = float(np.mean(data.sizes))

        # Create transforms
        transforms = [
            Transform(
                position=(float(positions[i, 0]), float(positions[i, 1])),
                scale=float(data.sizes[i] / base_size) if base_size > 0 else 1.0,
            )
            for i in range(len(positions))
        ]

        # Extract CRS
        crs = None
        if data.source_gdf.crs is not None:
            crs = data.source_gdf.crs.to_wkt()

        return LayoutResult(
            canonical_symbol=CircleSymbol(),
            transforms=transforms,
            base_size=base_size,
            positions=data.positions,
            sizes=data.sizes,
            adjacency=data.adjacency,
            bounds=data.bounds,
            crs=crs,
            algorithm_info={"type": "centroid", "info": info},
            valid_mask=data.valid_mask,
        )


# Register centroid layout
register_layout("centroid", CentroidLayout)
