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
>>> workflow.latest                       # Most recent result
>>> workflow[1]                           # First morph result
>>>
>>> # Export
>>> gdf_result = workflow.to_geodataframe()
"""

import copy
from collections.abc import Iterator
from typing import Any, Literal, Optional, Union

import numpy as np

from .algorithm import morph_geometries
from .cartogram import Cartogram
from .errors import compute_error_metrics
from .grid import build_multilevel_grids
from .history import CartogramSnapshot, History
from .options import MorphOptions, MorphStatus

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
    >>> workflow.latest                       # Most recent
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
        options: Optional[MorphOptions] = None,
        density_per: Optional[Literal["m2", "km2", "ha", "acre", "sqft", "sqmi"]] = None,
    ):
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

        errors = compute_error_metrics(original_areas, target_areas)
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
        )

        return Cartogram(
            snapshots=History([snapshot]),
            status=MorphStatus.ORIGINAL,
            niterations=0,
            duration=0.0,
            options=options,
            grid=options.grid,
            _source_gdf=self._original_gdf,
            _source_landmarks_gdf=self._original_landmarks_gdf,
        )

    # ========================================================================
    # Core Morphing Methods
    # ========================================================================

    def morph(
        self,
        options: Optional[MorphOptions] = None,
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
        # Determine effective options
        if options is not None:
            effective_options = copy.deepcopy(options)
        else:
            effective_options = copy.deepcopy(self._results[-1].options)
            for key, value in overrides.items():
                if hasattr(effective_options, key):
                    setattr(effective_options, key, value)
                else:
                    raise AttributeError(f"MorphOptions has no attribute '{key}'")

        latest = self._results[-1].latest
        values = np.array(self._original_gdf[self._column])

        # Compute target density from original data (invariant across refinements)
        # Apply area_scale for consistent density units (e.g., m² to km²)
        original_areas = np.array(self._original_gdf.area)
        scaled_areas = original_areas * effective_options.area_scale
        target_density = float(np.sum(values) / np.sum(scaled_areas))

        result = morph_geometries(
            geometries=latest.geometry,
            values=values,
            target_density=target_density,
            landmarks=latest.landmarks,
            coords=latest.coords,
            options=effective_options,
        )

        # Set source references for to_geodataframe()
        result._source_gdf = self._original_gdf
        result._source_landmarks_gdf = self._original_landmarks_gdf

        self._results.append(result)
        return result

    def morph_multiresolution(
        self,
        resolution: int = 512,
        levels: int = 3,
        margin: float = 0.5,
        square: bool = True,
        options: Union[MorphOptions, list[MorphOptions], None] = None,
    ) -> Cartogram:
        """
        Multi-resolution morphing with progressive refinement.

        Each level is stored as a separate Cartogram in the workflow.

        Parameters
        ----------
        resolution : int, default=512
            Base resolution for the highest level grid.
        levels : int, default=3
            Number of resolution levels.
        margin : float, default=0.5
            Margin around data bounds for grid creation.
        square : bool, default=True
            Whether to create square grids.
        options : MorphOptions, list[MorphOptions], or None
            Options for each level. Single options applied to all levels.

        Returns
        -------
        Cartogram
            Final level result (also accessible via self.latest).

        Examples
        --------
        >>> workflow.morph_multiresolution(resolution=512, levels=3)
        >>> for cartogram in workflow:
        ...     print(cartogram.status, cartogram.get_errors().mean_error_pct)
        """
        grids = build_multilevel_grids(
            self._original_gdf.total_bounds,
            resolution,
            levels,
            margin=margin,
            square=square,
        )

        # Normalize options
        if options is None:
            options_list = [copy.deepcopy(self._results[-1].options) for _ in range(levels)]
        elif isinstance(options, MorphOptions):
            options_list = [copy.deepcopy(options) for _ in range(levels)]
        else:
            if len(options) != levels:
                raise ValueError(f"Options length ({len(options)}) must match levels ({levels})")
            options_list = [copy.deepcopy(opt) for opt in options]

        for _level, (grid, level_options) in enumerate(zip(grids, options_list)):
            level_options.grid = grid
            level_options.progress_message = (
                f"{'Refining' if self.is_morphed else 'Morphing'} with {grid.sx}x{grid.sy} grid"
            )

            result = self.morph(options=level_options)

            if result.status == MorphStatus.CONVERGED:
                break

        return self.latest

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
    def latest(self) -> Cartogram:
        """The most recent result."""
        return self._results[-1]

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
        run_id: Optional[int] = None,
        iteration: Optional[int] = None,
        include_errors: bool = True,
        include_density: bool = True,
    ) -> Any:
        """
        Export a cartogram result as GeoDataFrame.

        Parameters
        ----------
        run_id : int, optional
            Which result to export (default: latest).
        iteration : int, optional
            Which iteration snapshot (default: latest in result).
        include_errors : bool, default=True
            Add '_morph_error_pct' column.
        include_density : bool, default=True
            Add '_morph_density' column.

        Returns
        -------
        GeoDataFrame
            Copy of original with morphed geometry and optional metrics.
        """
        cartogram = self._results[run_id] if run_id is not None else self.latest
        return cartogram.to_geodataframe(
            iteration=iteration,
            include_errors=include_errors,
            include_density=include_density,
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
        status = self.latest.status.value

        if self.is_morphed:
            errors = self.latest.get_errors()
            error_str = f", mean_error={errors.mean_error_pct:.1f}%" if errors else ""
        else:
            error_str = ""

        return f"CartogramWorkflow(geoms={n_geoms}, runs={n_results}, status={status}{error_str})"
