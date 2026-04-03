"""Shared history framework for iterative algorithms.

Provides domain-agnostic base classes that any iterative algorithm can use
to record and retrieve per-iteration state snapshots.

Classes
-------
BaseSnapshot
    Abstract base class for algorithm state snapshots.
History
    Generic container for managing collections of BaseSnapshot instances.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

__all__ = ["BaseSnapshot", "History"]

S = TypeVar("S", bound="BaseSnapshot")


@dataclass
class BaseSnapshot(ABC):
    """Abstract base class for algorithm snapshots.

    Defines the minimum interface required for any snapshot to work with
    :class:`History`.  Subclass this and add algorithm-specific attributes.

    Attributes
    ----------
    iteration : int
        The iteration number this snapshot represents.
    """

    iteration: int

    def has_variable(self, name: str) -> bool:
        """Return True if the attribute exists and is not None."""
        return hasattr(self, name) and getattr(self, name) is not None

    def get_variable(self, name: str) -> Any:
        """Return the value of attribute *name*.

        Raises ``AttributeError`` if the attribute does not exist.
        """
        if hasattr(self, name):
            return getattr(self, name)
        raise AttributeError(f"Variable '{name}' not found in snapshot")

    def get_all_variables(self) -> dict[str, Any]:
        """Return a dict of all non-None, non-private attributes."""
        variables = {}
        for attr_name in dir(self):
            if not attr_name.startswith("_") and attr_name != "iteration":
                try:
                    value = getattr(self, attr_name)
                    if value is not None and not callable(value):
                        variables[attr_name] = value
                except Exception:  # noqa: S110
                    pass
        return variables


@dataclass
class History(Generic[S]):
    """Manages a collection of :class:`BaseSnapshot` objects.

    Provides list-like index access, iteration-based lookup, and
    per-variable history across all snapshots.  Works with any snapshot
    class that inherits from :class:`BaseSnapshot`.

    Attributes
    ----------
    snapshots : list[S]
        Snapshots in chronological order.
    """

    snapshots: list[S] = field(default_factory=list)

    def add_snapshot(self, snapshot: S) -> None:
        """Append *snapshot* to the history."""
        self.snapshots.append(snapshot)

    def get_snapshot(self, iteration: int) -> S | None:
        """Return the snapshot for *iteration*, or None if not found."""
        for snapshot in self.snapshots:
            if snapshot.iteration == iteration:
                return snapshot
        return None

    def get_variable_history(self, variable_name: str) -> list[Any]:
        """Return a list of *variable_name* values across all snapshots.

        Missing values (attribute absent or None) are represented as None.
        """
        return [
            snapshot.get_variable(variable_name) if snapshot.has_variable(variable_name) else None
            for snapshot in self.snapshots
        ]

    def get_iterations(self) -> list[int]:
        """Return the list of recorded iteration numbers."""
        return [s.iteration for s in self.snapshots]

    def get_variable_at_iteration(self, variable_name: str, iteration: int) -> Any:
        """Return *variable_name* at *iteration*.

        Raises ``ValueError`` if no snapshot exists for that iteration.
        """
        snapshot = self.get_snapshot(iteration)
        if snapshot is None:
            raise ValueError(f"No snapshot found for iteration {iteration}")
        return snapshot.get_variable(variable_name)

    def latest(self) -> S | None:
        """Return the most recent snapshot, or None if empty."""
        return self.snapshots[-1] if self.snapshots else None

    def variable_summary(self, variable_name: str) -> dict[str, Any]:
        """Return summary statistics for *variable_name* across all snapshots."""
        import numpy as np

        values = self.get_variable_history(variable_name)
        valid = [v for v in values if v is not None]
        if not valid:
            return {"count": 0, "available": False}
        if isinstance(valid[0], np.ndarray):
            shapes = [v.shape for v in valid]
            all_same = all(s == shapes[0] for s in shapes)
            return {
                "count": len(valid),
                "available": True,
                "shape_consistent": all_same,
                "shapes": shapes[0] if all_same else shapes,
                "dtype": str(valid[0].dtype),
            }
        return {
            "count": len(valid),
            "available": True,
            "min": float(min(valid)),
            "max": float(max(valid)),
            "type": type(valid[0]).__name__,
        }

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, index: int | slice) -> S | list[S]:
        return self.snapshots[index]

    def __iter__(self) -> Iterator[S]:
        return iter(self.snapshots)

    def __repr__(self) -> str:
        if not self.snapshots:
            return "History(empty)"
        iters = [s.iteration for s in self.snapshots]
        return f"History(snapshots={len(self.snapshots)}, iters={iters[0]}..{iters[-1]})"
