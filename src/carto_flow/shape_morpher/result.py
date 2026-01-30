"""
===========================================================================
Result container for morphing operations
===========================================================================

Data class for storing complete results from cartogram generation.

Main Components
---------------
- MorphResult: Complete results container with metadata

Example
-------
>>> from carto_flow.shape_morpher import morph_gdf
>>>
>>> result = morph_gdf(gdf, 'population')
>>> print(f"Status: {result.status}")
>>> print(f"Mean error: {result.final_mean_error:.1%}")
>>> cartogram = result.geometries
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from .grid import Grid
    from .history import History

__all__ = [
    "MorphResult",
]


@dataclass
class MorphResult:
    """Complete results from a morphing operation.

    This class contains all results from a morphing operation including
    geometries, computation history, convergence status, and optional
    displacement field data.

    Attributes
    ----------
    geometries : Any
        The morphed geometries (GeoDataFrame or list of geometries)
    history : History
        Computation history with snapshots of the algorithm state
    landmarks : Any, optional
        Morphed landmark geometries if landmarks were provided
    status : str, default="completed"
        Computation status ("completed", "converged", "stalled", etc.)
    history_internals : History, optional
        Internal computation data if save_internals=True
    iterations_completed : int, optional
        Number of iterations completed before termination
    final_mean_error : float, optional
        Final mean of absolute log2 area errors at convergence.
        Log2 representation is symmetric: a value of 1.0 corresponds to
        regions being 2x too large or too small.
    final_max_error : float, optional
        Final maximum of absolute log2 area errors at convergence.
    final_area_errors : np.ndarray, optional
        Per-geometry log2 area errors at convergence (same order as geometries).
        Computed as log2(current_area / target_area). Positive = oversized,
        negative = undersized.
    displacement_field : Union[Tuple[np.ndarray, np.ndarray], np.ndarray], optional
        Displacement field in same format as input coordinates
    displaced_coords : np.ndarray, optional
        Final displaced coordinates for refinement workflows
    grid : Grid, optional
        Grid used for computation
    """

    # Core results
    geometries: Any
    history: "History"
    landmarks: Optional[Any] = None
    status: str = "completed"

    # Optional computation metadata
    history_internals: Optional["History"] = None
    iterations_completed: Optional[int] = None
    final_mean_error: Optional[float] = None
    final_max_error: Optional[float] = None
    final_area_errors: Optional[np.ndarray] = None

    # Displacement field results
    displacement_field: Optional[Union[tuple[np.ndarray, np.ndarray], np.ndarray]] = None
    displaced_coords: Optional[np.ndarray] = None

    # Grid used for computation
    grid: Optional["Grid"] = None

    def __repr__(self) -> str:
        """Return concise string representation for terminal display.

        Returns
        -------
        str
            String representation showing key result information
        """
        # Basic info
        geom_count = self._get_geometry_count()
        snapshot_count = len(self.history.snapshots) if self.history else 0

        # Status and metrics
        status_str = f"status={self.status}"
        if self.iterations_completed is not None:
            status_str += f", iter={self.iterations_completed}"
        if self.final_mean_error is not None:
            status_str += f", mean_error={self.final_mean_error:.4f}"
        if self.final_max_error is not None:
            status_str += f", max_error={self.final_max_error:.4f}"

        # Landmark info
        landmark_str = ", landmarks" if self.landmarks is not None else ""

        return f"MorphResult(geoms={geom_count}, snapshots={snapshot_count}, {status_str}{landmark_str})"

    def _get_geometry_count(self) -> int:
        """
        Get the number of geometries in the result.

        This helper method provides a robust way to count geometries that works
        with different input types (GeoDataFrame, list of geometries, single geometry).

        Returns
        -------
        int
            Number of geometries in the result
        """
        try:
            if hasattr(self.geometries, "__len__"):
                return len(self.geometries)
            elif hasattr(self.geometries, "shape"):
                return self.geometries.shape[0]
            else:
                return 1  # Single geometry
        except:
            return 1  # Fallback

    def get_result_tuple(self) -> tuple[Any, "History"]:
        """Get backward-compatible tuple format: (geometries, history)"""
        return self.geometries, self.history

    def get_full_tuple(self) -> tuple[Any, "History", Optional[Any]]:
        """Get full tuple format: (geometries, history, landmarks)"""
        if self.landmarks is not None:
            return self.geometries, self.history, self.landmarks
        else:
            return self.geometries, self.history
