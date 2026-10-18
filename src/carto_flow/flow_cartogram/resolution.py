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
# Length, in cells of the finest grid, that a segment may reach once the morph has enlarged its region, and the
# share of the segments that may exceed it before the input is reported.
SEGMENT_CELLS = 4
SEGMENT_SHARE = 0.005


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
    messages = _messages(targets, cell, cell, label, label, _suggestion("grid", bounds, targets, margin, square))
    if messages:
        warnings.warn("Grid resolution: " + "; ".join(messages) + ".", UserWarning, stacklevel=5)


def check_segments(geometries: Any, target_areas: Any, cell: float, stacklevel: int = 4) -> None:
    """Warn when more than ``SEGMENT_SHARE`` of the segments are expected to grow beyond ``SEGMENT_CELLS`` cells.

    The vertices move with the flow, so a segment grows with the linear enlargement of its region, estimated by the
    square root of the ratio of the target to the current area (not below 1). Segments that long cannot follow the
    flow, and the straight edges between sparse vertices cross where the polygon is narrow. ``cell`` is the side of a
    cell of the finest grid.
    """
    import shapely

    geoms = np.asarray(geometries, dtype=object)
    targets = np.asarray(target_areas, dtype=float)
    areas = shapely.area(geoms)
    valid = (areas > 0) & (targets > 0)
    if not valid.any():
        return
    stretch = np.ones(len(geoms))
    stretch[valid] = np.sqrt(np.maximum(targets[valid] / areas[valid], 1.0))
    limit = SEGMENT_CELLS * cell
    total = 0
    long_segments = 0
    longest_now = 0.0
    longest_expected = 0.0
    for geom, factor in zip(geoms, stretch, strict=True):
        if geom is None or geom.is_empty:
            continue
        for poly in shapely.get_parts(geom):
            for ring in (poly.exterior, *poly.interiors):
                coords = np.asarray(ring.coords)
                if len(coords) < 2:
                    continue
                lengths = np.hypot(*np.diff(coords, axis=0).T)
                total += lengths.size
                long_segments += int(np.sum(lengths * factor > limit))
                longest_now = max(longest_now, float(lengths.max()))
                longest_expected = max(longest_expected, float(lengths.max()) * factor)
    if total == 0 or long_segments <= SEGMENT_SHARE * total:
        return
    suggestion = limit / float(stretch.max())
    warnings.warn(
        f"Vertex density: {long_segments} of {total} segments ({100 * long_segments / total:.1f} %) are expected to "
        f"grow beyond {SEGMENT_CELLS} cells of the finest grid ({cell:.4g} each) when the morph enlarges their regions "
        f"(the longest is {longest_now:.4g} now and up to {longest_expected:.4g} after). Segments that long cannot "
        f"follow the flow and the polygons tangle. Densify the input before morphing with "
        f"carto_flow.geo_utils.densify_coverage(gdf, max_segment_length={suggestion:.3g}).",
        UserWarning,
        stacklevel=stacklevel,
    )
