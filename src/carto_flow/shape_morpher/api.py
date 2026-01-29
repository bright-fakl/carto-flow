"""
===========================================================================
High-level API for cartogram generation
===========================================================================

User-facing functions for creating cartograms from GeoDataFrames.

Main Components
---------------
- morph_gdf: GeoDataFrame-based interface for cartogram generation
- multiresolution_morph: Multi-resolution morphing with progressive refinement

Example
-------
>>> from carto_flow.shape_morpher import morph_gdf, MorphOptions
>>>
>>> result = morph_gdf(gdf, 'population', options=MorphOptions.preset_fast())
>>> cartogram = result.geometries
>>> print(f"Mean error: {result.final_mean_error:.1%}")
"""

import copy
import uuid
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from .algorithm import morph_geometries
from .grid import build_multilevel_grids
from .options import MorphOptions
from .result import MorphResult

if TYPE_CHECKING:
    from .computer import MorphComputer

__all__ = [
    "morph_gdf",
    "multiresolution_morph",
]


def multiresolution_morph(
    gdf: Any,
    column: str,
    landmarks: Any = None,
    resolution: int = 512,
    levels: int = 3,
    margin: float = 0.5,
    square: bool = True,
    options: Union[MorphOptions, list[MorphOptions], None] = None,
    return_computer: bool = False,
    return_all_levels: bool = False,
    displacement_coords: Any = None,
) -> Union[MorphResult, list[MorphResult], "MorphComputer"]:
    """Multi-resolution morphing using MorphComputer infrastructure.

    This function creates cartograms at multiple resolution levels, using the
    MorphComputer class for better state management and refinement capabilities.
    Each resolution level refines the results from the previous level.

    Parameters
    ----------
    gdf : Any
        GeoDataFrame-like object containing polygon geometries and data values
    column : str
        Name of the column containing data values for cartogram generation
    landmarks : Any, optional
        Optional landmark geometries for tracking reference points
    resolution : int, default=512
        Base resolution for the highest level grid
    levels : int, default=3
        Number of resolution levels to use
    margin : float, default=0.5
        Margin around the data bounds for grid creation
    square : bool, default=True
        Whether to create a square grid or rectangular
    options : MorphOptions, List[MorphOptions], or None, optional
        Configuration options for the morphing process:
        - None: Use default options for all levels
        - MorphOptions: Apply same options to all levels
        - List[MorphOptions]: Different options for each level
    return_computer : bool, default=False
        If True, return the MorphComputer instance for advanced usage and
        further refinement. Takes precedence over return_all_levels.
    return_all_levels : bool, default=False
        If True, return a list of MorphResult for each resolution level.
        If False (default), return only the final level MorphResult.
        Ignored if return_computer=True.
    displacement_coords : array-like, optional
        Coordinates for displacement field computation in various formats:
        - (N, 2) array for point coordinates
        - (X, Y) tuple for meshgrid coordinates
        - (M, N, 2) array for mesh format
        Format is automatically detected from the coordinate structure.

    Returns
    -------
    MorphResult
        Final morphing result (default mode). Contains geometries, history,
        grid, status, and optional displacement field data.
    list[MorphResult]
        List of MorphResult for each level (if return_all_levels=True).
        Each result contains the grid used for that level.
    MorphComputer
        MorphComputer instance for further refinement (if return_computer=True).
        Access all results via computer.get_refinement_history().

    Examples
    --------
    >>> # Basic usage - returns final MorphResult
    >>> result = multiresolution_morph(gdf, 'population')
    >>> cartogram = result.geometries
    >>> grid = result.grid
    >>> history = result.history

    >>> # Get results for all levels
    >>> results = multiresolution_morph(gdf, 'population', return_all_levels=True)
    >>> for i, result in enumerate(results):
    ...     print(f"Level {i}: grid={result.grid.sx}x{result.grid.sy}")

    >>> # Get MorphComputer for further refinement
    >>> computer = multiresolution_morph(gdf, 'population', return_computer=True)
    >>> computer.set_computation(mean_tol=0.01)
    >>> refined_result = computer.morph()

    >>> # With custom options per level
    >>> level_opts = [
    ...     MorphOptions(n_iter=50, dt=0.3, mean_tol=0.1),
    ...     MorphOptions(n_iter=100, dt=0.2, mean_tol=0.05),
    ...     MorphOptions(n_iter=150, dt=0.1, mean_tol=0.02)
    ... ]
    >>> result = multiresolution_morph(gdf, 'population', options=level_opts)

    >>> # With displacement field computation
    >>> x = np.linspace(0, 100, 50)
    >>> y = np.linspace(0, 80, 40)
    >>> X, Y = np.meshgrid(x, y)
    >>> displacement_coords = np.column_stack([X.ravel(), Y.ravel()])
    >>> result = multiresolution_morph(
    ...     gdf, 'population',
    ...     displacement_coords=displacement_coords
    ... )
    >>> displacement_field = result.displacement_field
    """
    # Import here to avoid circular imports
    from .computer import MorphComputer

    # 1. Build multi-resolution grids with desired maximum resolution and number of levels
    grids = build_multilevel_grids(gdf.total_bounds, resolution, levels, margin=margin, square=square)

    # 2. Normalize options input
    if options is None:
        options = [MorphOptions()]
    elif not isinstance(options, (tuple, list)):
        options = [options]

    # For single option or None, cycle it for all levels
    # For multiple options, validate length matches levels
    if len(options) == 1:
        # Simple cycle implementation for single option
        options_cycle = (options[0] for _ in range(levels))
    else:
        if len(options) != levels:
            raise ValueError(f"Length of options ({len(options)}) must match levels ({levels})")
        options_cycle = options

    # 3. Initialize MorphComputer (options will be set in the loop)
    computer = MorphComputer(
        gdf=gdf,
        column=column,
        landmarks=landmarks,
        displacement_coords=displacement_coords,
        options=MorphOptions(),  # Basic options, will be overridden in loop
    )

    # 4. Progressive refinement through resolution levels
    level_results: list[MorphResult] = []

    for level, (grid, opts) in enumerate(zip(grids, options_cycle)):
        # Create options with grid and custom message only for levels we actually compute
        level_opts = copy.deepcopy(opts)
        level_opts.grid = grid
        level_opts.progress_message = f"{'Refining' if level > 0 else 'Morphing'} with {grid.sx}x{grid.sy} grid"

        # Update options for current level
        computer.set_options(level_opts)

        # Run morphing for this level
        morph_result = computer.morph()
        level_results.append(morph_result)

        # Check for convergence using the status from the last refinement run
        last_run = computer.get_run_info()
        if last_run and last_run.status == "converged":
            break

    # 5. Return results based on mode
    if return_computer:
        # Return MorphComputer for further refinement
        # All results accessible via computer.get_refinement_history()
        return computer
    elif return_all_levels:
        # Return list of MorphResult for each level
        return level_results
    else:
        # Return only the final level MorphResult (default)
        return level_results[-1]


def morph_gdf(
    gdf: Any,
    column: str,
    landmarks: Any = None,
    # New options parameter for simplified API
    options: Optional[MorphOptions] = None,
    # Displacement field computation
    displacement_coords=None,
    previous_displaced_coords=None,
) -> MorphResult:
    """
    Generate flow-based cartogram using dataframe interface with morph_polygons core algorithm.

    This function provides a GeoDataFrame-based interface that handles dataframe validation,
    metadata management, and target area computation, while delegating the core morphing
    algorithm to the morph_polygons function.

    Parameters
    ----------
    gdf : Any
        GeoDataFrame-like object containing polygon geometries and data values.
        Must have 'geometry' column with Polygon/MultiPolygon objects and
        the specified column with numeric data values.
    column : str
        Name of the column in gdf containing the data values that drive the
        cartogram deformation (e.g., 'population', 'income', 'density').
    landmarks : Any, optional
        Optional landmarks GeoDataFrame for tracking reference points
    options : MorphOptions, optional
        Algorithm options (dt, n_iter, density_smooth, etc.)
    displacement_coords : array-like, optional
        Coordinates for displacement field computation in various formats:
        - (N, 2) array for point coordinates
        - (X, Y) tuple for meshgrid coordinates
        - (M, N, 2) array for mesh format
        Format is automatically detected from the coordinate structure.
    previous_displaced_coords : np.ndarray, optional
        Previously displaced coordinates for refinement mode

    Returns
    -------
    result : MorphResult
        Complete morphing results containing:
        - geometries: GeoDataFrame with deformed geometries
        - history: History object with computation snapshots
        - status: Computation status ("completed", "converged", etc.)
        - Optional: landmarks, history_internals, performance metrics
        - displacement_field: Displacement field in same format as input coordinates (if displacement_coords provided)
        - displaced_coords: Final displaced coordinates for refinement (if displacement_coords provided)

        For backward compatibility, also returns tuple: (gdf_result, history)

    Examples
    --------
    >>> import geopandas as gpd
    >>> from carto_flow.shape_morpher import morph_gdf, MorphResult
    >>> from carto_flow.grid import Grid
    >>>
    >>> # Load geographic data
    >>> gdf = gpd.read_file('regions.geojson')
    >>> grid = Grid.from_bounds(gdf.total_bounds, size=100)
    >>>
    >>> # Create population-based cartogram (returns MorphResult)
    >>> result = morph_gdf(gdf, 'population', grid, n_iter=150)
    >>> print(f"Status: {result.status}")
    >>> print(f"Iterations: {result.iterations_completed}")
    >>>
    >>> # Access results
    >>> cartogram = result.gdf_result
    >>> history = result.history
    >>>
    >>> # Backward compatibility - still works as tuple
    >>> cartogram, history = morph_gdf(gdf, 'population', grid, n_iter=150)
    >>> print(f"Converged in {len(history.snapshots)} iterations")

    >>> # Compute displacement field with GeoDataFrame interface
    >>> # Create regular grid of coordinates for displacement field
    >>> x = np.linspace(0, 100, 50)
    >>> y = np.linspace(0, 80, 40)
    >>> X, Y = np.meshgrid(x, y)
    >>> displacement_coords = np.column_stack([X.ravel(), Y.ravel()])
    >>>
    >>> # Run morphing with displacement field computation
    >>> result = morph_gdf(
    ...     gdf, 'population',
    ...     displacement_coords=displacement_coords,
    ...     options=MorphOptions(n_iter=100, dt=0.2)
    ... )
    >>>
    >>> # Access displacement field results
    >>> displacement_field = result.displacement_field  # Format matches displacement_coords
    >>> displaced_coords = result.displaced_coords      # For refinement
    >>>
    >>> # Refine with displacement field
    >>> refined_result = morph_gdf(
    ...     result.geometries, 'population',
    ...     displacement_coords=displacement_coords,
    ...     previous_displaced_coords=result.displaced_coords,
    ...     options=MorphOptions(n_iter=50, dt=0.1)
    ... )
    """

    # --- Handle options ---
    if options is None:
        # Create default options
        options = MorphOptions()

    # --- Preserve CRS ---
    original_crs = getattr(gdf, "crs", None)

    # --- Input validation and dataframe setup ---
    is_morphed, _refinement_id = _validate_morphed_dataframe(gdf, column)

    if is_morphed:
        # Refinement mode - use existing _original_areas
        gdf_work = gdf.copy()
        original_areas = gdf["_original_areas"].values
    else:
        # Initial mode - establish baseline
        gdf_work = gdf.copy()
        original_areas = gdf_work.area.values
        gdf_work["_original_areas"] = original_areas

        # Store metadata for future refinements
        gdf_work.attrs["cartoflow"] = {"morphed": True, "column_used": column, "refinement_id": str(uuid.uuid4())[:8]}

    # --- Prepare geometries and landmarks for morph_polygons ---
    geometries = gdf_work.geometry.values
    column_values = np.array(gdf_work[column])

    # Handle landmarks
    landmarks_geoms = None
    if landmarks is not None:
        landmarks_geoms = landmarks.geometry.values

    # Call the core morphing algorithm
    geometry_result = morph_geometries(
        geometries,
        column_values,
        original_areas,
        landmarks_geoms,
        options=options,
        displacement_coords=displacement_coords,
        previous_displaced_coords=previous_displaced_coords,
    )

    # --- Process results and snapshots ---
    if options.save_history and geometry_result.history:
        for snapshot in geometry_result.history.snapshots:
            snapshot_geoms = snapshot.geometry
            snapshot.geometry = gdf_work.copy()
            snapshot.geometry.geometry = snapshot_geoms

    # --- Create final GeoDataFrame ---
    gdf_work.geometry = geometry_result.geometries
    if original_crs is not None:
        gdf_work.crs = original_crs
    geometry_result.geometries = gdf_work

    # Handle landmarks if provided
    if geometry_result.landmarks is not None:
        landmarks_result = landmarks.copy()
        landmarks_result.geometry = geometry_result.landmarks
        if original_crs is not None:
            landmarks_result.crs = original_crs
        geometry_result.landmarks = landmarks_result

    return geometry_result


def _validate_morphed_dataframe(gdf: Any, column: str) -> tuple[bool, str]:
    """
    Validate if dataframe is properly morphed and return refinement_id.

    This helper function checks if a GeoDataFrame has been previously morphed
    by looking for cartoflow metadata and _original_areas column. Used to
    determine if subsequent operations should be refinement or initial morphing.

    Parameters
    ----------
    gdf : Any
        GeoDataFrame-like object to validate
    column : str
        Name of the column being used for morphing

    Returns
    -------
    tuple[bool, str]
        (is_morphed, refinement_id) where:
        - is_morphed: True if dataframe was previously morphed
        - refinement_id: UUID string of the original morphing session, or None

    Raises
    ------
    ValueError
        If dataframe was morphed with different column than currently specified
    """
    # Check for metadata
    if not hasattr(gdf, "attrs") or "cartoflow" not in gdf.attrs:
        return False, None

    meta = gdf.attrs["cartoflow"]
    if not meta.get("morphed", False):
        return False, None

    # Check for _original_areas column
    if "_original_areas" not in gdf.columns:
        return False, None

    # Validate column consistency
    stored_column = meta.get("column_used")
    if stored_column and stored_column != column:
        raise ValueError(
            f"Dataframe was created using column '{stored_column}' "
            f"but refinement is using column '{column}'. "
            f"Use column '{stored_column}' for refinement."
        )

    return True, meta.get("refinement_id")
