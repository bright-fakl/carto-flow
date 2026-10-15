"""
History management system for iterative algorithms.

This module provides a flexible and extensible system for tracking algorithm
state across iterations. It supports any iterative algorithm through the
BaseSnapshot interface, making it reusable across different domains. The
system provides both access to all data for a single snapshot (by index or
iteration), and access to all snapshots for a single variable (by variable name).

Classes
-------
BaseSnapshot
    Abstract base class for algorithm state snapshots.
CartogramSnapshot
    Snapshot class for cartogram algorithm statistics.
CartogramInternalsSnapshot
    Snapshot class for cartogram internal state.
History
    Container class for managing collections of snapshots.
ErrorRecord
    Lightweight scalar error metrics for a single iteration.
ConvergenceHistory
    Lightweight history of scalar convergence metrics for all iterations.

Notes
-----
To create a custom snapshot class, inherit from BaseSnapshot and add your
algorithm-specific attributes:

>>> from carto_flow.flow_cartogram.history import BaseSnapshot
>>> from dataclasses import dataclass
>>> import numpy as np
>>>
>>> @dataclass
... class CustomSnapshot(BaseSnapshot):
...     iteration: int
...     loss: float = None
...     accuracy: float = None
...     weights: np.ndarray = None

Examples
--------
>>> from carto_flow.flow_cartogram.history import History, CartogramSnapshot
>>> from carto_flow.flow_cartogram.errors import MorphErrors
>>> import numpy as np
>>> history = History()
>>> errors = MorphErrors(np.array([0.1]), 0.1, 0.1, np.array([7.2]), 7.2, 7.2)
>>> snapshot = CartogramSnapshot(iteration=0, errors=errors)
>>> history.add_snapshot(snapshot)
>>> error_history = history.get_variable_history('errors')
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from carto_flow._history import BaseSnapshot, History

from .errors import MorphErrors

# Module-level exports - Public API
__all__ = [
    "BaseSnapshot",
    "CartogramInternalsSnapshot",
    "CartogramSnapshot",
    "ConvergenceHistory",
    "ErrorRecord",
    "History",
    "MorphErrors",
]


@dataclass
class CartogramInternalsSnapshot(BaseSnapshot):
    """Snapshot of cartogram internal state at one iteration.

    This class holds all the internal data for one iteration of the cartogram
    algorithm, making it easy to access all variables for a specific time step.

    Attributes
    ----------
    iteration : int
        The iteration number this snapshot represents.
    rho : Optional[np.ndarray]
        Density field array.
    vx : Optional[np.ndarray]
        X-component of the effective velocity field used for displacement
        (after any anisotropy modulation and smoothing).
    vy : Optional[np.ndarray]
        Y-component of the effective velocity field used for displacement
        (after any anisotropy modulation and smoothing).
    geometry_mask : Optional[np.ndarray]
        Geometry index mask where:
        - -1 = outside all geometries
        - k = inside geometry k (0 <= k < number of geometries)
    """

    iteration: int
    rho: np.ndarray | None = None
    vx: np.ndarray | None = None
    vy: np.ndarray | None = None
    geometry_mask: np.ndarray | None = None

    def __repr__(self) -> str:
        """Concise string representation for terminal display."""
        fields = []
        if self.rho is not None:
            fields.append(f"rho={self.rho.shape}")
        if self.vx is not None:
            fields.append(f"vx={self.vx.shape}")
        if self.vy is not None:
            fields.append(f"vy={self.vy.shape}")

        fields_str = ", ".join(fields) if fields else "no_fields"
        return f"CartogramInternalsSnapshot(iter={self.iteration}, {fields_str})"


@dataclass
class CartogramSnapshot(BaseSnapshot):
    """Snapshot of cartogram statistics at one iteration.

    This class holds statistical data for one iteration of the cartogram
    algorithm, making it easy to access all variables for a specific time step.

    Attributes
    ----------
    iteration : int
        The iteration number this snapshot represents.
    geometry : Optional[Any]
        List of shapely geometries or GeoDataFrame containing the current polygon geometries.
    landmarks : Optional[Any]
        List of morphed landmark geometries if landmarks were provided.
    coords : Optional[Any]
        Displaced coordinates for displacement field computation.
        Format matches the input coordinates format (points, grid, or mesh).
    errors : Optional[MorphErrors]
        Structured error metrics object containing all error fields.
        Provides consistent access to log_errors, percentage errors, etc.
    density : Optional[np.ndarray]
        Current density values for each geometry (values / current_areas).
    landmark_area_scaling : Optional[np.ndarray]
        With landmarks, the area scaling of the flow along the path of every landmark vertex up to this
        iteration (the exponential of the integral of the divergence of the velocity field along the path).
    probe_positions, probe_area_scaling : Optional[np.ndarray]
        With ``MorphOptions.landmark_probes``, the positions of the probe points and the area scaling along their paths.
    coords_area_scaling : Optional[np.ndarray]
        With displacement coordinates, the area scaling of the flow along the path of every coordinate point.
    """

    iteration: int
    geometry: Any | None = None
    landmarks: Any | None = None
    coords: Any | None = None
    errors: MorphErrors | None = None
    density: np.ndarray | None = None
    landmark_area_scaling: np.ndarray | None = None
    probe_positions: np.ndarray | None = None
    probe_area_scaling: np.ndarray | None = None
    coords_area_scaling: np.ndarray | None = None

    def __repr__(self) -> str:
        """Concise string representation for terminal display."""
        # Geometry info
        geom_info = "geom"
        if self.geometry is not None:
            try:
                if hasattr(self.geometry, "__len__"):
                    geom_info = f"geom={len(self.geometry)}"
                elif hasattr(self.geometry, "shape"):
                    geom_info = f"geom={self.geometry.shape[0]}"
            except Exception:
                geom_info = "geom"

        # Error info from MorphErrors
        error_info = ""
        if self.errors is not None:
            error_info = f", mean={self.errors.mean_log_error:.4f}, max={self.errors.max_log_error:.4f}"

        # Optional fields
        extras = []
        if self.landmarks is not None:
            extras.append("landmarks")
        if self.coords is not None:
            extras.append("coords")
        extras_str = f", {', '.join(extras)}" if extras else ""

        return f"CartogramSnapshot(iter={self.iteration}, {geom_info}{error_info}{extras_str})"


@dataclass
class ErrorRecord:
    """Scalar error metrics for a single iteration.

    This lightweight class stores only the 4 scalar error metrics,
    unlike MorphErrors which also stores per-geometry error arrays.
    Memory footprint: ~40 bytes per instance.

    Attributes
    ----------
    iteration : int
        The iteration number.
    mean_log_error : float
        Mean of absolute log2 errors across all geometries.
    max_log_error : float
        Maximum of absolute log2 errors across all geometries.
    mean_error_pct : float
        Mean approximate percentage error.
    max_error_pct : float
        Maximum approximate percentage error.
    """

    iteration: int
    mean_log_error: float
    max_log_error: float
    mean_error_pct: float
    max_error_pct: float

    def __repr__(self) -> str:
        """Concise string representation."""
        return (
            f"ErrorRecord(iter={self.iteration}, "
            f"mean_log={self.mean_log_error:.4f}, max_log={self.max_log_error:.4f}, "
            f"mean_pct={self.mean_error_pct:.1f}%, max_pct={self.max_error_pct:.1f}%)"
        )


class ConvergenceHistory:
    """Lightweight history of scalar convergence metrics.

    Stores scalar error metrics for every iteration of the morphing algorithm,
    enabling detailed convergence analysis without the memory overhead of
    full snapshots. Uses pre-allocated numpy arrays for efficiency.

    Parameters
    ----------
    capacity : int, optional
        Maximum number of iterations to store. If not provided, uses dynamic
        resizing (less efficient but more flexible).

    Attributes
    ----------
    iterations : np.ndarray
        Array of iteration numbers (int64).
    mean_log_errors : np.ndarray
        Array of mean log2 errors (float64).
    max_log_errors : np.ndarray
        Array of max log2 errors (float64).
    mean_errors_pct : np.ndarray
        Array of mean percentage errors (float64).
    max_errors_pct : np.ndarray
        Array of max percentage errors (float64).

    Examples
    --------
    >>> from carto_flow.flow_cartogram.history import ConvergenceHistory
    >>> from carto_flow.flow_cartogram.errors import MorphErrors
    >>> import numpy as np
    >>> convergence = ConvergenceHistory(capacity=100)
    >>> errors = MorphErrors(np.array([0.1]), 0.1, 0.2, np.array([7.2]), 7.2, 14.9)
    >>> convergence.add(1, errors)
    >>> print(len(convergence))
    1
    >>> record = convergence[0]
    >>> print(record.iteration)
    1
    """

    __slots__ = (
        "_iterations",
        "_max_errors_pct",
        "_max_log_errors",
        "_mean_errors_pct",
        "_mean_log_errors",
        "_size",
    )

    def __init__(self, capacity: int | None = None):
        """Initialize with optional pre-allocated capacity."""
        cap = capacity if capacity is not None else 0
        self._iterations: np.ndarray = np.empty(cap, dtype=np.int64)
        self._mean_log_errors: np.ndarray = np.empty(cap, dtype=np.float64)
        self._max_log_errors: np.ndarray = np.empty(cap, dtype=np.float64)
        self._mean_errors_pct: np.ndarray = np.empty(cap, dtype=np.float64)
        self._max_errors_pct: np.ndarray = np.empty(cap, dtype=np.float64)
        self._size = 0

    def add(self, iteration: int, errors: MorphErrors) -> None:
        """Add error metrics for an iteration.

        Parameters
        ----------
        iteration : int
            The iteration number.
        errors : MorphErrors
            Error metrics object (only scalar values are extracted).
        """
        # Grow arrays if needed (only happens if capacity wasn't specified)
        if self._size >= len(self._iterations):
            new_cap = max(16, len(self._iterations) * 2)
            self._iterations = np.resize(self._iterations, new_cap)
            self._mean_log_errors = np.resize(self._mean_log_errors, new_cap)
            self._max_log_errors = np.resize(self._max_log_errors, new_cap)
            self._mean_errors_pct = np.resize(self._mean_errors_pct, new_cap)
            self._max_errors_pct = np.resize(self._max_errors_pct, new_cap)

        self._iterations[self._size] = iteration
        self._mean_log_errors[self._size] = errors.mean_log_error
        self._max_log_errors[self._size] = errors.max_log_error
        self._mean_errors_pct[self._size] = errors.mean_error_pct
        self._max_errors_pct[self._size] = errors.max_error_pct
        self._size += 1

    def finalize(self) -> None:
        """Trim arrays to actual size, freeing unused memory."""
        if self._size < len(self._iterations):
            self._iterations = self._iterations[: self._size].copy()
            self._mean_log_errors = self._mean_log_errors[: self._size].copy()
            self._max_log_errors = self._max_log_errors[: self._size].copy()
            self._mean_errors_pct = self._mean_errors_pct[: self._size].copy()
            self._max_errors_pct = self._max_errors_pct[: self._size].copy()

    @property
    def iterations(self) -> np.ndarray:
        """Array of iteration numbers."""
        return self._iterations[: self._size]

    @property
    def mean_log_errors(self) -> np.ndarray:
        """Array of mean log2 errors."""
        return self._mean_log_errors[: self._size]

    @property
    def max_log_errors(self) -> np.ndarray:
        """Array of max log2 errors."""
        return self._max_log_errors[: self._size]

    @property
    def mean_errors_pct(self) -> np.ndarray:
        """Array of mean percentage errors."""
        return self._mean_errors_pct[: self._size]

    @property
    def max_errors_pct(self) -> np.ndarray:
        """Array of max percentage errors."""
        return self._max_errors_pct[: self._size]

    def __len__(self) -> int:
        """Return number of recorded iterations."""
        return self._size

    def __getitem__(self, index: int) -> ErrorRecord:
        """Get error record by index.

        Parameters
        ----------
        index : int
            Index into the history (supports negative indexing).

        Returns
        -------
        ErrorRecord
            Error metrics for the requested index.
        """
        if index < 0:
            index = self._size + index
        if index < 0 or index >= self._size:
            raise IndexError(f"index {index} out of range for size {self._size}")
        return ErrorRecord(
            iteration=int(self._iterations[index]),
            mean_log_error=float(self._mean_log_errors[index]),
            max_log_error=float(self._max_log_errors[index]),
            mean_error_pct=float(self._mean_errors_pct[index]),
            max_error_pct=float(self._max_errors_pct[index]),
        )

    def __iter__(self) -> Iterator[ErrorRecord]:
        """Iterate over error records."""
        for i in range(self._size):
            yield self[i]

    def get_by_iteration(self, iteration: int) -> ErrorRecord | None:
        """Get error record for a specific iteration.

        Parameters
        ----------
        iteration : int
            The iteration number to retrieve.

        Returns
        -------
        ErrorRecord or None
            The error record, or None if iteration not found.
        """
        indices = np.where(self._iterations[: self._size] == iteration)[0]
        if len(indices) == 0:
            return None
        return self[indices[0]]

    def to_dict(self) -> dict[str, np.ndarray]:
        """Convert to dictionary of arrays for easy export.

        Returns
        -------
        dict
            Dictionary with keys 'iteration', 'mean_log_error', 'max_log_error',
            'mean_error_pct', 'max_error_pct'.
        """
        return {
            "iteration": self.iterations,
            "mean_log_error": self.mean_log_errors,
            "max_log_error": self.max_log_errors,
            "mean_error_pct": self.mean_errors_pct,
            "max_error_pct": self.max_errors_pct,
        }

    def __repr__(self) -> str:
        """Concise string representation."""
        if self._size == 0:
            return "ConvergenceHistory(empty)"
        return f"ConvergenceHistory(n={self._size}, iters={self._iterations[0]}..{self._iterations[self._size - 1]})"
