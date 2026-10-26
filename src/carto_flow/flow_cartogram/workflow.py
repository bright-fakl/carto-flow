"""
Workflow class for iterative cartogram refinement.

This module provides the CartogramWorkflow class for managing iterative
cartogram refinement workflows. It stores a sequence of Cartogram objects
and provides methods for morphing, multi-resolution morphing, and state
management.

Classes
-------
CartogramWorkflow
    Container for managing cartogram refinement workflows.

Examples
--------
>>> from carto_flow import CartogramWorkflow, MorphOptions
>>>
>>> workflow = CartogramWorkflow(gdf, 'population')
>>> workflow.morph()                      # Initial morph
>>> workflow.morph(mean_tol=0.01)         # Refine with tighter tolerance
>>>
>>> # Access results
>>> workflow.original                     # Initial unmorphed state
>>> workflow.last                         # Most recent result
>>> workflow[1]                           # First morph result
>>>
>>> # Export
>>> gdf_result = workflow.to_geodataframe()
"""

import copy
import warnings
from collections.abc import Iterator
from typing import Any, Literal

import numpy as np

from ..geo_utils.validity import check_input
from .algorithm import morph_geometries
from .cartogram import Cartogram
from .errors import compute_error_metrics
from .grid import Grid, LadderSpec, build_multilevel_grids
from .history import CartogramSnapshot, History
from .options import MorphOptions, MorphStatus
from .resolution import auto_levels, auto_start, check_grid, check_resolution, check_segments

__all__ = ["CartogramWorkflow"]

# Unit conversion factors: value is how many m² equal one unit
# e.g., 1 km² = 1e6 m²
_AREA_UNIT_SCALES = {
    "m2": 1.0,
    "km2": 1e6,
    "ha": 1e4,  # hectare = 10,000 m²
    "acre": 4046.8564224,  # 1 acre ≈ 4046.86 m²
    "sqft": 0.09290304,  # 1 sq ft = 0.0929 m²
    "sqmi": 2.589988e6,  # 1 sq mile ≈ 2.59 km²
}


def _get_crs_area_unit(gdf: Any) -> str:
    """Detect the area unit from a GeoDataFrame's CRS.

    Raises
    ------
    ValueError
        If CRS is None or has an unrecognized unit.
    """
    if gdf.crs is None:
        raise ValueError(
            "Cannot determine area units: GeoDataFrame has no CRS. "
            "Either set a CRS or use area_scale in MorphOptions instead of density_per."
        )

    try:
        unit_name = gdf.crs.axis_info[0].unit_name.lower()
    except (AttributeError, IndexError) as e:
        raise ValueError(
            f"Cannot determine area units from CRS: {gdf.crs}. Use area_scale in MorphOptions instead of density_per."
        ) from e

    if "metre" in unit_name or "meter" in unit_name:
        return "m2"
    elif "foot" in unit_name or "feet" in unit_name:
        return "sqft"
    elif "kilometre" in unit_name or "kilometer" in unit_name:
        return "km2"
    else:
        raise ValueError(
            f"Unrecognized CRS unit: '{unit_name}'. "
            f"Supported units: metre, foot, kilometre. "
            "Use area_scale in MorphOptions instead of density_per."
        )


def _compute_area_scale(source_unit: str, target_unit: str) -> float:
    """Compute scale factor to convert areas from source to target units."""
    if source_unit not in _AREA_UNIT_SCALES:
        raise ValueError(f"Unknown source unit: {source_unit}")
    if target_unit not in _AREA_UNIT_SCALES:
        raise ValueError(f"Unknown target unit: {target_unit}. Supported: {list(_AREA_UNIT_SCALES.keys())}")
    return _AREA_UNIT_SCALES[source_unit] / _AREA_UNIT_SCALES[target_unit]


class CartogramWorkflow:
    """
    Workflow for iterative cartogram refinement.

    Manages a sequence of Cartogram objects where index 0 is the original
    (unmorphed) state. Provides methods for morphing, multi-resolution
    morphing, and state management.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame containing polygon geometries and data values.
    column : str
        Name of the column containing data values for cartogram generation.
    landmarks : GeoDataFrame, optional
        Optional landmarks GeoDataFrame for tracking reference points.
    coords : array-like, optional
        Coordinates for displacement field computation.
    options : MorphOptions, optional
        Base options for morphing.
    density_per : {'m2', 'km2', 'ha', 'acre', 'sqft', 'sqmi'}, optional
        Unit for density calculations. If specified, automatically computes
        the appropriate area_scale based on the GeoDataFrame's CRS units.
    validate : bool, default=True
        Whether to check the input first (see :func:`~carto_flow.geo_utils.validate_geometries`): invalid, empty or
        null geometries and values that are not finite or are negative raise a ValueError that names the rows;
        overlapping polygons and geographic coordinates give a warning.

    Examples
    --------
    >>> workflow = CartogramWorkflow(gdf, 'population')
    >>> workflow.morph()                      # Initial morph
    >>> workflow.morph(mean_tol=0.01)         # Refine
    >>>
    >>> # With density in people per km²
    >>> workflow = CartogramWorkflow(gdf, 'population', density_per='km2')
    >>>
    >>> # Access results
    >>> workflow.original                     # Initial state
    >>> workflow.last                         # Most recent
    >>> workflow[1]                           # First morph result
    >>>
    >>> # Export
    >>> gdf_result = workflow.to_geodataframe()
    """

    def __init__(
        self,
        gdf: Any,
        column: str,
        landmarks: Any = None,
        coords: Any = None,
        options: MorphOptions | None = None,
        density_per: Literal["m2", "km2", "ha", "acre", "sqft", "sqmi"] | None = None,
        validate: bool = True,
    ):
        if validate:
            check_input(gdf, column, coverage=True, zero="ok", stacklevel=3)

        # Handle density_per by computing area_scale
        if density_per is not None:
            source_unit = _get_crs_area_unit(gdf)
            area_scale = _compute_area_scale(source_unit, density_per)

            if options is None:
                options = MorphOptions(area_scale=area_scale)
            else:
                options = options.copy_with(area_scale=area_scale)

        self._original_gdf = gdf.copy()
        self._column = column
        self._original_landmarks_gdf = landmarks
        self._coords0 = coords
        # Probe points, uniform in every region, carried with the landmarks to measure the crowding near the borders
        self._probe_xy0 = None
        self._probe_region = None
        if options is not None and options.landmark_probes > 0 and (landmarks is not None or coords is not None):
            from .landmarks import probe_points

            self._probe_xy0, self._probe_region = probe_points(
                self._original_gdf.geometry.values, options.landmark_probes
            )
        self._results: list[Cartogram] = [self._create_initial_cartogram(coords, options or MorphOptions())]

    def _create_initial_cartogram(
        self,
        coords: Any,
        options: MorphOptions,
    ) -> Cartogram:
        """Create Cartogram representing original (unmorphed) state."""
        geometries = self._original_gdf.geometry.values
        values = np.array(self._original_gdf[self._column])
        original_areas = np.array(self._original_gdf.area)
        target_areas = np.sum(original_areas) * values / np.sum(values)

        errors = compute_error_metrics(
            original_areas, target_areas, eps=options.error_floor * float(np.mean(target_areas))
        )
        # Apply area_scale for density calculation (e.g., m² to km²)
        density = values / (original_areas * options.area_scale)
        landmarks_geoms = (
            self._original_landmarks_gdf.geometry.values if self._original_landmarks_gdf is not None else None
        )

        snapshot = CartogramSnapshot(
            iteration=0,
            geometry=geometries,
            errors=errors,
            density=density,
            landmarks=landmarks_geoms,
            coords=coords,
            probe_positions=self._probe_xy0,
            probe_area_scaling=np.ones(len(self._probe_xy0)) if self._probe_xy0 is not None else None,
        )

        return Cartogram(
            snapshots=History([snapshot]),
            status=MorphStatus.ORIGINAL,
            niterations=0,
            duration=0.0,
            options=options,
            grid=options.grid if isinstance(options.grid, Grid) else None,
            _source_gdf=self._original_gdf,
            _source_landmarks_gdf=self._original_landmarks_gdf,
            _value_column=self._column,
            _source_coords=self._coords0,
            _probe_region=self._probe_region,
        )

    # ========================================================================
    # Core Morphing Methods
    # ========================================================================

    def morph(
        self,
        options: MorphOptions | None = None,
        **overrides,
    ) -> Cartogram:
        """
        Morph geometries (initial or refinement).

        Parameters
        ----------
        options : MorphOptions, optional
            Complete options replacement. If None, uses previous options.
        **overrides
            Individual option overrides applied to previous options.
            Example: mean_tol=0.01, dt=0.3

        Returns
        -------
        Cartogram
            The morphing result (also appended to workflow).

        Examples
        --------
        >>> workflow.morph()  # Use default/previous options
        >>> workflow.morph(options=MorphOptions.preset_fast())
        >>> workflow.morph(mean_tol=0.01, n_iter=200)
        """
        effective_options = self._effective_options(options, overrides)
        spec = effective_options.grid_spec
        if isinstance(spec, LadderSpec):
            return self._warn_if_stalled(self._run_ladder(spec, effective_options))
        return self._warn_if_stalled(self._morph(effective_options, check=True))

    @staticmethod
    def _warn_if_stalled(result: Cartogram) -> Cartogram:
        """Warn that a run ended as stalled and how to continue it; returns ``result``."""
        errors = result.get_errors()
        if result.status == MorphStatus.STALLED and errors is not None:
            warnings.warn(
                f"The morph stalled after {result.niterations} iterations (mean error {errors.mean_error_pct:.2g} %, "
                f"max {errors.max_error_pct:.3g} %). The regions that are still off may keep improving: "
                "workflow.morph(stall_patience=None, n_iter=N) continues from the best iterate with the stall rule "
                "off and runs N more iterations; other options (a finer grid=, a looser max_tol=) can be passed in the same call.",
                UserWarning,
                stacklevel=3,
            )
        return result

    def _effective_options(self, options: MorphOptions | None, overrides: dict) -> MorphOptions:
        """Options of a pass: the given ones, or the previous ones with the overrides applied."""
        if options is not None:
            effective_options = copy.deepcopy(options)
        else:
            effective_options = copy.deepcopy(self._results[-1].options)  # type: ignore[assignment]
            for key, value in overrides.items():
                if hasattr(effective_options, key):
                    setattr(effective_options, key, value)
                else:
                    raise AttributeError(f"MorphOptions has no attribute '{key}'")
        return effective_options

    def _morph(self, effective_options: MorphOptions, check: bool) -> Cartogram:
        """One morphing pass with resolved options; ``check`` warns about the grid resolution."""
        latest = self._results[-1].best
        values = np.array(self._original_gdf[self._column])

        # Compute target density from original data (invariant across refinements)
        # Apply area_scale for consistent density units (e.g., m² to km²)
        original_areas = np.array(self._original_gdf.area)
        scaled_areas = original_areas * effective_options.area_scale
        target_density = float(np.sum(values) / np.sum(scaled_areas))

        if check:
            bounds = effective_options._calculate_bounds_from_geometries(latest.geometry)  # type: ignore[union-attr]
            grid = effective_options.get_grid(bounds)
            spec = effective_options.grid_spec
            check_grid(
                grid,
                bounds,
                np.sum(original_areas) * values / np.sum(values),
                getattr(spec, "margin", 0.5),
                getattr(spec, "square", False),
            )
            check_segments(
                latest.geometry,  # type: ignore[union-attr]
                np.sum(original_areas) * values / np.sum(values),
                max(grid.dx, grid.dy),
                5,
            )

        result = morph_geometries(
            geometries=latest.geometry,  # type: ignore[union-attr]
            values=values,
            target_density=target_density,
            landmarks=latest.landmarks,  # type: ignore[union-attr]
            probes=latest.probe_positions,  # type: ignore[union-attr]
            coords=latest.coords,  # type: ignore[union-attr]
            options=effective_options,
        )

        # The area scaling of the landmarks, the probes and the coordinates accumulates over the passes
        previous = self._results[-1]
        for name in ("landmark_area_scaling", "probe_area_scaling", "coords_area_scaling"):
            before = getattr(previous.best, name, None) if previous.best is not None else None
            for snapshot in result.snapshots.snapshots:
                current = getattr(snapshot, name)
                if current is None:
                    continue
                if before is not None:
                    setattr(snapshot, name, before * current)
                elif previous.status != MorphStatus.ORIGINAL:
                    setattr(snapshot, name, None)  # an earlier pass had none

        # Set source references for to_geodataframe()
        result._source_gdf = self._original_gdf
        result._source_coords = self._coords0
        result._probe_region = self._probe_region
        result._source_landmarks_gdf = self._original_landmarks_gdf
        result._value_column = self._column

        self._results.append(result)
        return result

    def morph_multiresolution(
        self,
        min_resolution: int | Literal["auto"] = 128,
        levels: int | Literal["auto"] = 3,
        margin: float = 0.5,
        square: bool = True,
        options: MorphOptions | list[MorphOptions] | None = None,
    ) -> Cartogram:
        """
        Multi-resolution morphing with progressive refinement.

        Each level is stored as a separate Cartogram in the workflow.

        Parameters
        ----------
        min_resolution : int or "auto", default=128
            Resolution of the coarsest (first) grid level. Each subsequent
            level doubles this, up to ``min_resolution * 2^(levels-1)``.
            ``"auto"`` takes the smallest power of two from 64 to 512 at which
            the smallest target region covers at least 2.5 cells (512 if none does).
        levels : int or "auto", default=3
            Number of resolution levels. ``"auto"`` doubles up to a finest grid
            of 2048; the morph ends at the first level that converges.
        margin : float, default=0.5
            Margin around data bounds for grid creation.
        square : bool, default=True
            Whether to create square grids.
        options : MorphOptions, list[MorphOptions], or None
            Options for each level. When a single MorphOptions is provided (or
            None), the same options are used at every level. When a list is
            provided, each entry maps directly to a level. With a single
            MorphOptions, its ``n_iter`` is the total number of iterations of
            the ladder, and each level may use what is left; with a list, every
            level has its own ``n_iter``.

        Returns
        -------
        Cartogram
            Final level result (also accessible via self.last).

        Examples
        --------
        >>> workflow.morph_multiresolution(min_resolution=128, levels=3)
        >>> for cartogram in workflow:
        ...     print(cartogram.status, cartogram.get_errors().mean_error_pct)
        """
        spec = LadderSpec(min_resolution, levels, margin, square)
        return self._warn_if_stalled(self._run_ladder(spec, options))

    def _run_ladder(self, spec: LadderSpec, options: MorphOptions | list[MorphOptions] | None) -> Cartogram:
        """Run the levels of ``spec`` from coarse to fine, one pass each, ending at the first converged level.

        With one ``MorphOptions`` (or None), its ``n_iter`` is the total number of iterations of the ladder: each
        level may use what is left. With a list, every level has its own ``n_iter``.
        """
        margin, square = spec.margin, spec.square
        min_resolution, levels = spec.min_resolution, spec.levels
        bounds = self._original_gdf.total_bounds
        values = np.asarray(self._original_gdf[self._column], dtype=float)
        areas = np.asarray(self._original_gdf.area)
        targets = areas.sum() * values / values.sum()
        if min_resolution == "auto":
            min_resolution = auto_start(bounds, targets, margin, square)
        if levels == "auto":
            levels = auto_levels(min_resolution)
        check_resolution(bounds, targets, min_resolution, levels, margin, square)
        grids = build_multilevel_grids(
            bounds,
            min_resolution,
            levels,
            margin=margin,
            square=square,
        )
        check_segments(self._original_gdf.geometry, targets, max(grids[-1].dx, grids[-1].dy), 4)

        # Normalize options
        if options is None:
            options_list = [copy.deepcopy(self._results[-1].options) for _ in range(levels)]
        elif isinstance(options, MorphOptions):
            options_list = [copy.deepcopy(options) for _ in range(levels)]
        else:
            if len(options) != levels:
                raise ValueError(f"Options length ({len(options)}) must match levels ({levels})")
            options_list = [copy.deepcopy(opt) for opt in options]

        remaining: int | None = None if isinstance(options, list) else options_list[0].n_iter  # type: ignore[union-attr]
        for level, (grid, level_options) in enumerate(zip(grids, options_list, strict=False)):
            if remaining is not None:
                if remaining < 1:
                    break
                level_options.n_iter = remaining  # type: ignore[union-attr]
            level_options.grid = grid  # type: ignore[union-attr]
            # the same area at every level: area_tol_cells counts cells of the finest grid
            level_options.area_tol_cells *= (grids[-1].dx * grids[-1].dy) / (grid.dx * grid.dy)  # type: ignore[union-attr]
            level_options.progress_message = (  # type: ignore[union-attr]
                f"{'Refining' if self.is_morphed else 'Morphing'} with {grid.sx}x{grid.sy} grid"
            )

            if level > 0:
                # Pre-scaling is a one-time correction on the initial geometry;
                # suppress it on refinement levels where geometry is already scaled.
                level_options.prescale_components = False  # type: ignore[union-attr]

            result = self._morph(level_options, check=False)  # type: ignore[arg-type]
            if remaining is not None:
                remaining -= result.niterations

            if result.status == MorphStatus.CONVERGED:
                break

        return self.last

    # ========================================================================
    # Container Access
    # ========================================================================

    def __len__(self) -> int:
        """Number of results (including original)."""
        return len(self._results)

    def __getitem__(self, idx: int) -> Cartogram:
        """Get result by index (0 = original)."""
        return self._results[idx]

    def __iter__(self) -> Iterator[Cartogram]:
        """Iterate over all results."""
        return iter(self._results)

    @property
    def original(self) -> Cartogram:
        """The original (unmorphed) state."""
        return self._results[0]

    @property
    def last(self) -> Cartogram:
        """The most recent result."""
        return self._results[-1]

    @property
    def latest(self) -> Cartogram:
        """The most recent result.

        .. deprecated:: 2.1
            Use :attr:`last`.
        """
        warnings.warn("CartogramWorkflow.latest is deprecated; use last", DeprecationWarning, stacklevel=2)
        return self.last

    @property
    def results(self) -> list[Cartogram]:
        """All results (copy of list)."""
        return self._results.copy()

    @property
    def is_morphed(self) -> bool:
        """Whether any morphing has been performed."""
        return len(self._results) > 1

    # ========================================================================
    # GeoDataFrame Export
    # ========================================================================

    def to_geodataframe(
        self,
        run_id: int | None = None,
        iteration: int | None = None,
        include_errors: bool = True,
        include_density: bool = True,
        repair: bool = False,
    ) -> Any:
        """
        Export a cartogram result as GeoDataFrame.

        Parameters
        ----------
        run_id : int, optional
            Which result to export (default: last).
        iteration : int, optional
            Which iteration snapshot (default: latest in result).
        include_errors : bool, default=True
            Add '_morph_error_pct' column.
        include_density : bool, default=True
            Add '_morph_density' column.
        repair : bool, default=False
            Make invalid (self-intersecting) regions valid; this adds vertices (see
            :meth:`Cartogram.to_geodataframe`).

        Returns
        -------
        GeoDataFrame
            Copy of original with morphed geometry and optional metrics.
        """
        cartogram = self._results[run_id] if run_id is not None else self.last
        return cartogram.to_geodataframe(
            iteration=iteration,
            include_errors=include_errors,
            include_density=include_density,
            repair=repair,
        )

    # ========================================================================
    # State Management
    # ========================================================================

    def pop(self, n: int = 1) -> list[Cartogram]:
        """
        Remove and return the last n results.

        Cannot remove the original (index 0).

        Parameters
        ----------
        n : int, default=1
            Number of results to remove.

        Returns
        -------
        list[Cartogram]
            The removed cartograms.

        Raises
        ------
        ValueError
            If attempting to remove the original state.
        """
        if n >= len(self._results):
            raise ValueError("Cannot remove original state")
        removed = self._results[-n:]
        self._results = self._results[:-n]
        return removed

    def reset(self) -> None:
        """Remove all morph results, keeping only original."""
        self._results = [self._results[0]]

    # ========================================================================
    # Representation
    # ========================================================================

    def __repr__(self) -> str:
        """Concise string representation."""
        n_geoms = len(self._original_gdf)
        n_results = len(self._results)
        status = self.last.status.value

        if self.is_morphed:
            errors = self.last.get_errors()
            error_str = f", mean_error={errors.mean_error_pct:.1f}%" if errors else ""
        else:
            error_str = ""

        return f"CartogramWorkflow(geoms={n_geoms}, runs={n_results}, status={status}{error_str})"
