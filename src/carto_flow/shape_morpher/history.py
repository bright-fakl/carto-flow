"""
History management system for iterative algorithms
==================================================

This module provides a flexible and extensible system for tracking algorithm
state across iterations. It supports any iterative algorithm through the
BaseSnapshot interface, making it reusable across different domains. The
system provides both access to all data for a single snapshot (by iteration),
and access to all snapshots for a single variable (by variable name).

Key components
--------------
- `BaseSnapshot`: Abstract base class for algorithm state snapshots
- `CartogramSnapshot`: Snapshot class for cartogram algorithm statistics
- `CartogramInternalsSnapshot`: Snapshot class for cartogram internal state
- `History`: Container class for managing collections of snapshots

Creating Custom Snapshots
-------------------------
To create a custom snapshot class, inherit from BaseSnapshot and add your
algorithm-specific attributes. The `iteration` attribute can be omitted,
as it is inherited from `BaseSnapshot`.

    >>> from carto_flow.history import BaseSnapshot
    >>> from dataclasses import dataclass
    >>> import numpy as np
    >>>
    >>> @dataclass
    ... class CustomSnapshot(BaseSnapshot):
    ...     iteration: int
    ...     loss: Optional[float] = None
    ...     accuracy: Optional[float] = None
    ...     weights: Optional[np.ndarray] = None

Example
-------
    >>> from carto_flow.history import History, CartogramSnapshot
    >>> history = History()
    >>> snapshot = CartogramSnapshot(iteration=0, mean_error=0.1)
    >>> history.add_snapshot(snapshot)
    >>> errors = history.get_variable_history('mean_error')
"""

from abc import ABC
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

# Module-level exports - Public API
__all__ = [
    "BaseSnapshot",
    "CartogramInternalsSnapshot",
    "CartogramSnapshot",
    "History",
]


@dataclass
class BaseSnapshot(ABC):
    """Abstract base class for algorithm snapshots.

    This class defines the minimum interface required for any snapshot
    to work with the History class. Users can extend this class to
    add their own algorithm-specific variables.

    The History class only requires:
    1. An 'iteration' attribute (int)
    2. A 'has_variable(name: str)' method
    3. A 'get_variable(name: str)' method

    All other attributes and methods are optional and algorithm-specific.

    Attributes
    ----------
    iteration : int
        The iteration number this snapshot represents.
    """

    iteration: int
    """The iteration number this snapshot represents."""

    def has_variable(self, name: str) -> bool:
        """Check if this snapshot contains a specific variable.

        Parameters
        ----------
        name : str
            Name of the variable to check.

        Returns
        -------
        bool
            True if the variable exists and is not None, False otherwise.
        """
        return hasattr(self, name) and getattr(self, name) is not None

    def get_variable(self, name: str) -> Any:
        """Get the value of a specific variable.

        Parameters
        ----------
        name : str
            Name of the variable to retrieve.

        Returns
        -------
        Any
            The value of the variable.

        Raises
        ------
        AttributeError
            If the variable doesn't exist in this snapshot.
        """
        if hasattr(self, name):
            return getattr(self, name)
        raise AttributeError(f"Variable '{name}' not found in snapshot")

    def get_all_variables(self) -> dict[str, Any]:
        """Get all non-None variables in this snapshot.

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping variable names to their values.
        """
        variables = {}
        for attr_name in dir(self):
            if not attr_name.startswith("_") and attr_name != "iteration":
                try:
                    value = getattr(self, attr_name)
                    if value is not None:
                        variables[attr_name] = value
                except:
                    pass  # Skip properties that might fail
        return variables


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
        X-component of velocity field.
    vy : Optional[np.ndarray]
        Y-component of velocity field.
    vx_mod : Optional[np.ndarray]
        Modified X-component of velocity field (after anisotropy).
    vy_mod : Optional[np.ndarray]
        Modified Y-component of velocity field (after anisotropy).
    mean_density : Optional[float]
        Target equilibrium density (total_values / total_area).
        Used for normalizing density field in visualizations.
    outside_mask : Optional[np.ndarray]
        Boolean mask where True indicates grid cells outside all geometries.
        Used to exclude background cells in percentile calculations.
    """

    iteration: int
    rho: Optional[np.ndarray] = None
    vx: Optional[np.ndarray] = None
    vy: Optional[np.ndarray] = None
    vx_mod: Optional[np.ndarray] = None
    vy_mod: Optional[np.ndarray] = None
    mean_density: Optional[float] = None
    outside_mask: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        """Concise string representation for terminal display."""
        fields = []
        if self.rho is not None:
            fields.append(f"rho={self.rho.shape}")
        if self.vx is not None:
            fields.append(f"vx={self.vx.shape}")
        if self.vy is not None:
            fields.append(f"vy={self.vy.shape}")
        if self.vx_mod is not None:
            fields.append(f"vx_mod={self.vx_mod.shape}")
        if self.vy_mod is not None:
            fields.append(f"vy_mod={self.vy_mod.shape}")

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
        GeoDataFrame containing the current polygon geometries.
    area_errors : Optional[np.ndarray]
        Array of log2 area errors for each polygon.
        Computed as log2(current_area / target_area).
        Positive values indicate oversized regions, negative values indicate
        undersized regions. This representation is symmetric: a value of +1
        means 2x too large, -1 means 2x too small.
    mean_error : Optional[float]
        Mean of absolute log2 area errors across all polygons.
    max_error : Optional[float]
        Maximum of absolute log2 area errors across all polygons.
    """

    iteration: int
    geometry: Optional[Any] = None  # GeoDataFrame
    area_errors: Optional[np.ndarray] = None
    mean_error: Optional[float] = None
    max_error: Optional[float] = None

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
            except:
                geom_info = "geom"

        # Error info
        error_info = ""
        if self.mean_error is not None or self.max_error is not None:
            error_parts = []
            if self.mean_error is not None:
                error_parts.append(f"mean={self.mean_error:.4f}")
            if self.max_error is not None:
                error_parts.append(f"max={self.max_error:.4f}")
            error_info = f", {', '.join(error_parts)}"

        return f"CartogramSnapshot(iter={self.iteration}, {geom_info}{error_info})"


@dataclass
class History:
    """Manages a collection of snapshots with convenient access patterns.

    This class provides both:
    - Access to all data for a single snapshot (by iteration)
    - Access to all snapshots for a single variable (by variable name)

    It works with any snapshot class that inherits from BaseSnapshot,
    making it reusable for any iterative algorithm.

    Attributes
    ----------
    snapshots : List[BaseSnapshot]
        List of snapshot objects in chronological order.

    Examples
    --------
    >>> from carto_flow.history import History, CartogramSnapshot
    >>> history = History()
    >>> snapshot = CartogramSnapshot(iteration=0, mean_error=0.1)
    >>> history.add_snapshot(snapshot)
    >>> latest = history.latest()
    >>> errors = history.get_variable_history('mean_error')
    """

    snapshots: list[BaseSnapshot] = field(default_factory=list)

    def add_snapshot(self, snapshot: BaseSnapshot) -> None:
        """Add a new snapshot to the history.

        Parameters
        ----------
        snapshot : BaseSnapshot
            The snapshot object to add to the collection.
        """
        self.snapshots.append(snapshot)

    def get_snapshot(self, iteration: int) -> Optional[BaseSnapshot]:
        """Get snapshot for a specific iteration.

        Parameters
        ----------
        iteration : int
            The iteration number to retrieve.

        Returns
        -------
        Optional[BaseSnapshot]
            The snapshot for the given iteration, or None if not found.
        """
        for snapshot in self.snapshots:
            if snapshot.iteration == iteration:
                return snapshot
        return None

    def get_variable_history(self, variable_name: str) -> list[Any]:
        """Get the history of a specific variable across all iterations.

        Parameters
        ----------
        variable_name : str
            Name of the variable to retrieve history for.

        Returns
        -------
        List[Any]
            List of values for the variable, in chronological order.
            Empty list if variable not found in any snapshot.
        """
        history = []
        for snapshot in self.snapshots:
            if snapshot.has_variable(variable_name):
                history.append(snapshot.get_variable(variable_name))
            else:
                history.append(None)
        return history

    def get_iterations(self) -> list[int]:
        """Get list of all iteration numbers.

        Returns
        -------
        List[int]
            List of all iteration numbers in chronological order.
        """
        return [snapshot.iteration for snapshot in self.snapshots]

    def get_variable_at_iteration(self, variable_name: str, iteration: int) -> Any:
        """Get a specific variable value at a specific iteration.

        Parameters
        ----------
        variable_name : str
            Name of the variable to retrieve.
        iteration : int
            The iteration number to retrieve the variable from.

        Returns
        -------
        Any
            The value of the variable at the specified iteration.

        Raises
        ------
        ValueError
            If no snapshot exists for the given iteration.
        """
        snapshot = self.get_snapshot(iteration)
        if snapshot is None:
            raise ValueError(f"No snapshot found for iteration {iteration}")
        return snapshot.get_variable(variable_name)

    def __len__(self) -> int:
        """Return number of snapshots.

        Returns
        -------
        int
            The number of snapshots in the history.
        """
        return len(self.snapshots)

    def __getitem__(self, iteration: int) -> BaseSnapshot:
        """Get snapshot by iteration number.

        Parameters
        ----------
        iteration : int
            The iteration number to retrieve.

        Returns
        -------
        BaseSnapshot
            The snapshot for the given iteration.

        Raises
        ------
        KeyError
            If no snapshot exists for the given iteration.
        """
        snapshot = self.get_snapshot(iteration)
        if snapshot is None:
            raise KeyError(f"No snapshot found for iteration {iteration}")
        return snapshot

    def __iter__(self) -> Iterator[CartogramSnapshot]:
        """Iterate over snapshots in chronological order.

        Returns
        -------
        Iterator[CartogramSnapshot]
            Iterator over snapshots in chronological order.
        """
        return iter(self.snapshots)

    def latest(self) -> Optional[CartogramSnapshot]:
        """Get the most recent snapshot.

        Returns
        -------
        Optional[CartogramSnapshot]
            The most recent snapshot, or None if no snapshots exist.
        """
        return self.snapshots[-1] if self.snapshots else None

    def variable_summary(self, variable_name: str) -> dict[str, Any]:
        """Get summary statistics for a variable across all iterations.

        Parameters
        ----------
        variable_name : str
            Name of the variable to summarize.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing summary statistics. For arrays, includes
            shape information and data type. For scalars, includes min/max
            values and type information.
        """
        values = self.get_variable_history(variable_name)
        valid_values = [v for v in values if v is not None]

        if not valid_values:
            return {"count": 0, "available": False}

        # Handle both scalar and array values
        if isinstance(valid_values[0], np.ndarray):
            # For arrays, return shape and value range info
            shapes = [v.shape for v in valid_values]
            all_same_shape = all(s == shapes[0] for s in shapes)
            return {
                "count": len(valid_values),
                "available": True,
                "shape_consistent": all_same_shape,
                "shapes": shapes if not all_same_shape else shapes[0],
                "dtype": str(valid_values[0].dtype),
            }
        else:
            # For scalars, return basic statistics
            return {
                "count": len(valid_values),
                "available": True,
                "min": float(min(valid_values)),
                "max": float(max(valid_values)),
                "type": type(valid_values[0]).__name__,
            }

    def __repr__(self) -> str:
        """Concise string representation for terminal display."""
        if not self.snapshots:
            return "History(empty)"

        min_iter = min(s.iteration for s in self.snapshots)
        max_iter = max(s.iteration for s in self.snapshots)
        snapshot_count = len(self.snapshots)

        # Show iteration range and count
        iter_range = f"iter={min_iter}" if min_iter == max_iter else f"iters={min_iter}..{max_iter}"

        return f"History(snapshots={snapshot_count}, {iter_range})"
