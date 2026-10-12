"""Grid resolution of a multiresolution morph: the size of the smallest regions in cells.

A region whose target area is a few cells or fewer cannot be represented by the density
field, so the coarsest grid is chosen from the smallest target region and a warning names
the regions the finest grid still cannot resolve.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from .grid import build_multilevel_grids

# Smallest target region, in cells, that the coarsest grid should resolve.
MIN_CELLS = 2.5
# Bounds of the automatically chosen coarsest grid, and the finest grid of the automatic ladder.
MIN_START = 64
MAX_START = 512
MAX_RESOLUTION = 2048
# Share of regions with a target below one cell above which the finest grid triggers a warning.
SUBCELL_SHARE = 0.01


def _cell_area(bounds: Any, resolution: int, margin: float, square: bool) -> float:
    grid = build_multilevel_grids(bounds, resolution, 1, margin=margin, square=square)[0]
    return float(grid.dx * grid.dy)


def _positive(target_areas: Any) -> np.ndarray:
    targets = np.asarray(target_areas, dtype=float)
    return targets[targets > 0]


def smallest_resolution(bounds: Any, target_areas: Any, margin: float = 0.5, square: bool = True) -> int | None:
    """Smallest power-of-two resolution, from ``MIN_START``, at which every target region covers
    at least ``MIN_CELLS`` cells; None if the inputs are empty or no resolution up to 2**16 does."""
    targets = _positive(target_areas)
    if targets.size == 0:
        return None
    smallest = float(targets.min())
    n = MIN_START
    while n <= 2**16:
        if smallest / _cell_area(bounds, n, margin, square) >= MIN_CELLS:
            return n
        n *= 2
    return None


def auto_start(bounds: Any, target_areas: Any, margin: float = 0.5, square: bool = True) -> int:
    """Coarsest resolution of the automatic ladder: ``smallest_resolution`` limited to ``MAX_START``."""
    n = smallest_resolution(bounds, target_areas, margin, square)
    return MAX_START if n is None or n > MAX_START else n


def auto_levels(min_resolution: int) -> int:
    """Number of levels that double ``min_resolution`` up to ``MAX_RESOLUTION`` (at least 1)."""
    return max(int(math.log2(MAX_RESOLUTION / min_resolution)) + 1, 1)


def _messages(
    targets: np.ndarray, cell: float, finest_cell: float, coarse_label: str, fine_label: str, suggestion: str
) -> list[str]:
    messages = []
    smallest = float(targets.min())
    if smallest / cell < MIN_CELLS:
        messages.append(
            f"the smallest target region covers {smallest / cell:.2g} cells at {coarse_label}; "
            f"at least {MIN_CELLS:g} cells are needed to resolve it ({suggestion})"
        )
    below = int(np.sum(targets / finest_cell < 1.0))
    if below > SUBCELL_SHARE * targets.size:
        messages.append(
            f"{below} of {targets.size} target regions ({100 * below / targets.size:.0f} %) are smaller than one "
            f"cell at {fine_label}; they cannot be resolved and the morph is unlikely to converge"
        )
    return messages


def _suggestion(name: str, bounds: Any, targets: np.ndarray, margin: float, square: bool) -> str:
    needed = smallest_resolution(bounds, targets, margin, square)
    if needed is not None and needed <= MAX_RESOLUTION:
        return f"{name}={needed}"
    return f"no grid up to {MAX_RESOLUTION} does"


def check_resolution(
    bounds: Any, target_areas: Any, min_resolution: int, levels: int, margin: float = 0.5, square: bool = True
) -> None:
    """Warn when the coarsest grid of a ladder is too coarse for the smallest target region or the
    finest grid leaves more than ``SUBCELL_SHARE`` of the regions below one cell."""
    targets = _positive(target_areas)
    if targets.size == 0:
        return
    cell = _cell_area(bounds, min_resolution, margin, square)
    messages = _messages(
        targets,
        cell,
        cell / 4 ** (levels - 1),
        f"the coarsest grid ({min_resolution})",
        f"the finest grid ({min_resolution * 2 ** (levels - 1)})",
        _suggestion("min_resolution", bounds, targets, margin, square),
    )
    if messages:
        warnings.warn("Grid resolution: " + "; ".join(messages) + ".", UserWarning, stacklevel=4)


def check_grid(grid: Any, bounds: Any, target_areas: Any, margin: float = 0.5, square: bool = True) -> None:
    """Warn when ``grid`` is too coarse for the smallest target region or leaves more than
    ``SUBCELL_SHARE`` of the regions below one cell."""
    targets = _positive(target_areas)
    if targets.size == 0:
        return
    label = f"the grid ({max(grid.sx, grid.sy)})"
    cell = float(grid.dx * grid.dy)
    messages = _messages(targets, cell, cell, label, label, _suggestion("grid_size", bounds, targets, margin, square))
    if messages:
        warnings.warn("Grid resolution: " + "; ".join(messages) + ".", UserWarning, stacklevel=5)
