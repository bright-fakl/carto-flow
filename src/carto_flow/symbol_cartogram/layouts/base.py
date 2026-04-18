"""Layout ABC, registry, and shared helpers."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..symbols import CircleSymbol
from .data_prep import LayoutData
from .layout_result import LayoutResult, Transform


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

_LAYOUT_REGISTRY: dict[str, type] = {}


def register_layout(name: str, cls: type) -> None:
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
# Shared result builder for physics-based layouts
# ---------------------------------------------------------------------------


def _build_physics_layout_result(
    positions: NDArray[np.floating],
    info: dict[str, Any],
    history: list[NDArray[np.floating]] | None,
    data: LayoutData,
    metrics: Any | None = None,
    sim_history: Any | None = None,
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
        crs = data.source_gdf.crs.to_wkt()

    # Fall back to legacy SimulationHistory(positions=...) when no typed history given
    if sim_history is None:
        from .layout_result import SimulationHistory

        if history is not None:
            sim_history = SimulationHistory(positions=history)

    return LayoutResult(
        canonical_symbol=CircleSymbol(),
        transforms=transforms,
        base_size=base_size,
        positions=data.geometry_positions if data.geometry_positions is not None else data.positions,
        sizes=data.sizes,
        adjacency=data.adjacency,
        bounds=data.bounds,
        crs=crs,
        layout_type="physics",
        history=sim_history,
        metrics=metrics,
        valid_mask=data.valid_mask,
        source_indices=data.source_indices,
        group_ids=data.group_ids,
    )
