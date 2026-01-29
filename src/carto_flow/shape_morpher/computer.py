"""
===========================================================================
Object-oriented interface for cartogram generation
===========================================================================

Stateful cartogram generation with refinement support.

Main Components
---------------
- MorphComputer: Stateful cartogram generation class
- RefinementRun: Record of a single refinement run

Example
-------
>>> from carto_flow.shape_morpher import MorphComputer, MorphOptions
>>>
>>> computer = MorphComputer(gdf, 'population', options=MorphOptions.preset_fast())
>>> result = computer.morph()
>>> cartogram = result.geometries
>>>
>>> # Refine with stricter tolerance
>>> computer.set_computation(mean_tol=0.02)
>>> refined_result = computer.morph()
"""

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .algorithm import _detect_coordinate_format, _normalize_coordinates
from .api import morph_gdf
from .grid import Grid
from .history import History
from .options import MorphOptions
from .result import MorphResult

__all__ = [
    "MorphComputer",
    "RefinementRun",
]


@dataclass
class RefinementRun:
    """Record of a single refinement run."""

    run_id: int
    options: MorphOptions
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None

    # Complete morphing result object
    morph_result: Optional[MorphResult] = None

    def __repr__(self) -> str:
        """Concise string representation for terminal display."""
        # Duration info
        if self.duration is not None:
            duration_str = f", duration={self.duration:.2f}s"
        elif self.end_time is not None:
            duration_str = f", duration={self.end_time - self.start_time:.2f}s"
        else:
            duration_str = ""

        # Status info
        if self.morph_result is not None:
            status_str = f", status={self.morph_result.status}"
            if self.morph_result.iterations_completed is not None:
                status_str += f", iter={self.morph_result.iterations_completed}"
        else:
            status_str = ", status=running"

        return f"RefinementRun(id={self.run_id}{duration_str}{status_str})"

    # Convenience properties for backward compatibility
    @property
    def gdf_result(self) -> Optional[Any]:
        """Get the morphed geometries (GeoDataFrame from morph_gdf, List from morph_geometries)."""
        return self.morph_result.geometries if self.morph_result else None

    @property
    def geometries(self) -> Optional[Any]:
        """Get the morphed geometries."""
        return self.morph_result.geometries if self.morph_result else None

    @property
    def history(self) -> Optional[History]:
        """Get the computation history."""
        return self.morph_result.history if self.morph_result else None

    @property
    def landmarks_result(self) -> Optional[Any]:
        """Get the morphed landmark geometries."""
        return self.morph_result.landmarks if self.morph_result else None

    @property
    def landmarks(self) -> Optional[Any]:
        """Get the morphed landmark geometries."""
        return self.morph_result.landmarks if self.morph_result else None

    @property
    def status(self) -> Optional[str]:
        """Get the computation status."""
        return self.morph_result.status if self.morph_result else None

    @property
    def history_internals(self) -> Optional[History]:
        """Get internal computation data."""
        return self.morph_result.history_internals if self.morph_result else None

    @property
    def iterations_completed(self) -> Optional[int]:
        """Get the number of iterations completed."""
        return self.morph_result.iterations_completed if self.morph_result else None

    @property
    def final_mean_error(self) -> Optional[float]:
        """Get the final mean error."""
        return self.morph_result.final_mean_error if self.morph_result else None

    @property
    def final_max_error(self) -> Optional[float]:
        """Get the final maximum error."""
        return self.morph_result.final_max_error if self.morph_result else None


class MorphComputer:
    """
    Object-oriented interface for flow-based cartogram generation with refinement support.

    This class provides a stateful alternative to the flow_morph function, allowing for
    iterative refinement of cartograms with different options while maintaining history
    and the ability to rollback to previous states. Supports displacement field computation
    and refinement workflows.

    Parameters
    ----------
    gdf : Any
        GeoDataFrame-like object containing polygon geometries and data values
    column : str
        Name of the column containing data values for cartogram generation
    landmarks : List[Geometry], optional
        Optional landmark geometries for tracking reference points
    options : MorphOptions, optional
        Optional configuration options. If None, default options are used.
    displacement_coords : array-like, optional
        Coordinates for displacement field computation. Stored immutably for refinement.

    Example
    -------
    >>> # Basic usage
    >>> computer = MorphComputer(gdf, 'population', options=MorphOptions(grid=grid))
    >>> result = computer.morph()
    >>> cartogram, history = result.geometries, result.history
    >>>
    >>> # With displacement field support
    >>> computer = MorphComputer(
    ...     gdf, 'population',
    ...     displacement_coords=displacement_coords,  # Auto-detects format
    ...     options=MorphOptions(grid=grid)
    ... )
    >>> result = computer.morph()
    >>> displacement_field = computer.get_displacement_field_result()
    >>>
    >>> # Refine with displacement field - coordinates automatically propagated
    >>> computer.set_computation(mean_tol=0.02)
    >>> refined_result = computer.morph()
    >>>
    >>> # More refinements are easy
    >>> computer.morph()  # Automatically refines from current state
    >>>
    >>> # Rollback if needed
    >>> computer.rollback()
    >>>
    >>> # Start over when ready
    >>> computer.reset()
    >>> computer.morph()  # Fresh start
    """

    def __init__(
        self,
        gdf: Any,
        column: str,
        landmarks: Any = None,
        options: Optional[MorphOptions] = None,
        displacement_coords: Any = None,
    ):
        """
        Initialize the MorphComputer.

        Parameters
        ----------
        gdf : Any
            GeoDataFrame-like object containing polygon geometries and data values
        column : str
            Name of the column containing data values for cartogram generation
        landmarks : List[Geometry], optional
            Optional landmark geometries for tracking reference points
        options : MorphOptions, optional
            Optional configuration options. If None, default options are used.
        displacement_coords : array-like, optional
            Coordinates for displacement field computation. Stored immutably for refinement.
        """
        # Immutable state
        self._gdf_original = gdf.copy()
        self._column = column
        self._landmarks_original = landmarks.copy() if landmarks is not None else None

        # Store original displacement coordinates for format conversion
        self._displacement_coords_format = (
            _detect_coordinate_format(displacement_coords) if displacement_coords is not None else None
        )
        self._displacement_coords_original = copy.deepcopy(displacement_coords)

        # Current working state (can be modified during refinement)
        self._gdf_current = gdf.copy()
        self._landmarks_current = landmarks.copy() if landmarks is not None else None
        self._displacement_coords_current = _normalize_coordinates(displacement_coords)

        # Refinement history
        self._refinement_runs: list[RefinementRun] = []

        # Initialize options with grid
        if options is None:
            self._current_options = MorphOptions()
        else:
            self._current_options = options

    def get_grid(self) -> Grid:
        """Get the current grid."""
        return self._get_grid()

    def _compute_target_areas(self) -> np.ndarray:
        """
        Compute target areas from original geometries and data.

        This method calculates the target areas that each geometry should achieve
        based on the data values in the specified column, maintaining the overall
        mass conservation principle of cartogram generation.

        Returns
        -------
        np.ndarray
            Array of target areas for each geometry
        """
        values = np.array(self._gdf_original[self._column])
        initial_areas = np.array(self._gdf_original.area)
        return np.sum(initial_areas) * values / np.sum(values)

    def _get_grid(self) -> Grid:
        """
        Get the current grid from options.

        This helper method retrieves the Grid object from the current options,
        ensuring that a valid grid is available for computation.

        Returns
        -------
        Grid
            Current grid object for spatial discretization

        Raises
        ------
        ValueError
            If no grid is set in the current options
        """
        if self._current_options.grid is None:
            raise ValueError("Grid is not set in options")
        return self._current_options.grid

    # ============================================================================
    # Fluent Options Interface
    # ============================================================================

    def set_computation(
        self,
        density_smooth: Optional[float] = None,
        dt: Optional[float] = None,
        n_iter: Optional[int] = None,
        recompute_every: Optional[int] = None,
        snapshot_every: Optional[int] = None,
        mean_tol: Optional[float] = None,
        max_tol: Optional[float] = None,
    ) -> "MorphComputer":
        """Set computation-related options."""
        if density_smooth is not None:
            self._current_options.density_smooth = density_smooth
        if dt is not None:
            self._current_options.dt = dt
        if n_iter is not None:
            self._current_options.n_iter = n_iter
        if recompute_every is not None:
            self._current_options.recompute_every = recompute_every
        if snapshot_every is not None:
            self._current_options.snapshot_every = snapshot_every
        if mean_tol is not None:
            self._current_options.mean_tol = mean_tol
        if max_tol is not None:
            self._current_options.max_tol = max_tol
        return self

    def set_anisotropy(
        self,
        Dx: Optional[float] = None,
        Dy: Optional[float] = None,
        anisotropy: Optional[Callable[[Grid], tuple[np.ndarray, np.ndarray]]] = None,
    ) -> "MorphComputer":
        """Set anisotropy-related options."""
        if Dx is not None:
            self._current_options.Dx = Dx
        if Dy is not None:
            self._current_options.Dy = Dy
        if anisotropy is not None:
            self._current_options.anisotropy = anisotropy
        return self

    def set_smoothing(self, vsmooth: Optional[float] = None) -> "MorphComputer":
        """Set velocity field smoothing options."""
        if vsmooth is not None:
            self._current_options.vsmooth = vsmooth
        return self

    def set_grid(self, grid: Grid) -> "MorphComputer":
        """Set the grid for spatial discretization.

        Parameters
        ----------
        grid : Grid
            Grid defining spatial discretization for computation
        """
        self._current_options.grid = grid
        return self

    def set_output(
        self,
        save_history: Optional[bool] = None,
        save_internals: Optional[bool] = None,
        show_progress: Optional[bool] = None,
    ) -> "MorphComputer":
        """Set output and progress options."""
        if save_history is not None:
            self._current_options.save_history = save_history
        if save_internals is not None:
            self._current_options.save_internals = save_internals
        if show_progress is not None:
            self._current_options.show_progress = show_progress
        return self

    def set_options(self, options: MorphOptions) -> "MorphComputer":
        """Set complete options configuration.

        Parameters
        ----------
        options : MorphOptions
            Complete options object to use for morphing

        Returns
        -------
        MorphComputer
            Self for method chaining
        """
        self._current_options = options
        return self

    # ============================================================================
    # Core Computation Methods
    # ============================================================================

    def morph(self) -> MorphResult:
        """
        Morph geometries into cartogram.

        First call: Performs initial morphing from original data
        Subsequent calls: Refines the current morphed cartogram

        Returns
        -------
        MorphResult
            Complete morphing result containing geometries, history, status, etc.
        """
        # Simple logic: if no morphed state, we're starting fresh from original
        # (no need to copy since _gdf_current is already a copy of original)

        return self._run_morph()

    def _run_morph(self) -> MorphResult:
        """
        Internal method to run morphing computation.

        This method handles the core morphing computation, including:
        - Setting up run records and timing
        - Creating appropriate options for the current refinement level
        - Executing the morphing algorithm
        - Recording results and updating internal state

        Returns
        -------
        MorphResult
            Complete morphing result
        """
        start_time = time.time()

        # Create run record
        run_id = len(self._refinement_runs)

        # Set dynamic progress message based on refinement stage
        num_geometries = len(self._gdf_current)
        if run_id == 0:
            # First run - morphing initial geometries
            default_progress_msg = f"Morphing {num_geometries} geometries"
        else:
            # Subsequent runs - refinement
            default_progress_msg = f"Morph refinement {run_id}"

        # Create options with progress message (respect custom message if provided)
        morph_options = copy.deepcopy(self._current_options)
        if morph_options.progress_message is None:
            morph_options.progress_message = default_progress_msg

        run_record = RefinementRun(
            run_id=run_id,
            options=morph_options,
            start_time=start_time,
        )

        try:
            # Run the morphing computation - always use current state
            # morph_gdf will automatically detect if this is initial or refinement
            result = morph_gdf(
                gdf=self._gdf_current,
                column=self._column,
                landmarks=self._landmarks_current,
                options=morph_options,
                displacement_coords=self._displacement_coords_original,
                previous_displaced_coords=self._displacement_coords_current,
            )

            # Store complete MorphResult object
            run_record.morph_result = result

            # Update current state
            self._gdf_current = result.geometries
            if self._landmarks_current is not None:
                self._landmarks_current = result.landmarks
            if self._displacement_coords_current is not None:
                self._displacement_coords_current = result.displaced_coords

            # Complete run record
            end_time = time.time()
            run_record.end_time = end_time
            run_record.duration = end_time - start_time

            # Add to refinement history
            self._refinement_runs.append(run_record)

            return result

        except Exception as e:
            # Mark run as failed
            end_time = time.time()
            run_record.end_time = end_time
            run_record.duration = end_time - start_time

            # Create a failed MorphResult for the run record
            run_record.morph_result = MorphResult(geometries=None, history=None, status=f"failed: {e!s}")

            self._refinement_runs.append(run_record)
            raise

    # ============================================================================
    # State Management and Utilities
    # ============================================================================

    def rollback(self, steps: int = 1) -> bool:
        """
        Rollback to a previous refinement state.

        Parameters
        ----------
        steps : int, default=1
            Number of refinement runs to rollback

        Returns
        -------
        success : bool
            True if rollback was successful, False if not enough history
        """
        if steps <= 0 or len(self._refinement_runs) < steps:
            return False

        # Remove the last 'steps' refinement runs
        self._refinement_runs = self._refinement_runs[:-steps]

        # Reset to state before the rolled-back runs
        if len(self._refinement_runs) == 0:
            # No refinement runs left, reset to original
            self._gdf_current = self._gdf_original.copy()
            self._landmarks_current = self._landmarks_original.copy() if self._landmarks_original is not None else None
            self._displacement_coords_current = copy.deepcopy(self._displacement_coords_original)
        else:
            # Reset to the last completed refinement run
            last_run = self._refinement_runs[-1]
            self._gdf_current = last_run.geometries.copy()
            self._landmarks_current = last_run.landmarks.copy() if last_run.landmarks is not None else None
            # For displacement coordinates, we need to get them from the run's result
            if last_run.morph_result is not None:
                self._displacement_coords_current = last_run.morph_result.displaced_coords

        return True

    def reset(self) -> None:
        """Reset to initial state, clearing all refinement history."""
        self._gdf_current = self._gdf_original.copy()
        self._landmarks_current = self._landmarks_original.copy() if self._landmarks_original is not None else None
        self._displacement_coords_current = copy.deepcopy(self._displacement_coords_original)
        self._refinement_runs.clear()
        self._current_options = MorphOptions(grid=self._current_options.grid)

    def get_result(self) -> Any:
        """Get the current morphed geometries."""
        return self._gdf_current

    def get_refinement_history(self) -> list[RefinementRun]:
        """Get the complete refinement history."""
        return self._refinement_runs.copy()

    def get_current_options(self) -> MorphOptions:
        """Get the current options configuration."""
        return copy.deepcopy(self._current_options)

    def get_run_info(self, run_id: Optional[int] = None) -> RefinementRun:
        """
        Get information about a specific refinement run.

        Parameters
        ----------
        run_id : int, optional
            ID of the run to retrieve. If None, returns the last run.

        Returns
        -------
        run_info : RefinementRun
            Information about the requested run
        """
        if run_id is None:
            return self._refinement_runs[-1] if self._refinement_runs else None
        return self._refinement_runs[run_id] if 0 <= run_id < len(self._refinement_runs) else None

    def get_landmarks_result(self) -> Optional[Any]:
        """Get the current morphed landmark geometries.

        Returns
        -------
        Optional[Any]
            Current landmark geometries if landmarks were provided, None otherwise
        """
        return self._landmarks_current

    def get_displacement_coords_result(self) -> Optional[Any]:
        """Get the current displacement field coordinates.

        Returns
        -------
        Optional[Any]
            Current displacement coordinates if displacement_coords were provided, None otherwise
        """
        return self._displacement_coords_current

    def get_displacement_field_result(self) -> Optional[Any]:
        """Get the latest displacement field result.

        Returns
        -------
        Optional[Any]
            Latest displacement field in the same format as originally provided, None otherwise
        """
        if self._refinement_runs and self._refinement_runs[-1].morph_result:
            displacement_field = self._refinement_runs[-1].morph_result.displacement_field
            return displacement_field
        return None

    def __repr__(self) -> str:
        """Concise string representation for terminal display."""
        # Basic info
        geom_count = len(self._gdf_current) if self._gdf_current is not None else 0
        run_count = len(self._refinement_runs)

        # Landmark and displacement info
        landmark_str = ", landmarks" if self._landmarks_current is not None else ""
        displacement_str = ", displacement" if self._displacement_coords_current is not None else ""

        # Current status
        if run_count > 0:
            last_run = self._refinement_runs[-1]
            if last_run.morph_result is not None:
                status_str = f", status={last_run.morph_result.status}"
            else:
                status_str = ", status=running"
        else:
            status_str = ", status=initialized"

        return f"MorphComputer(geoms={geom_count}, runs={run_count}{status_str}{landmark_str}{displacement_str})"
