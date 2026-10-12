"""Detect and repair invalid polygons (self-intersecting rings)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon

__all__ = ["invalid_indices", "repair_polygons"]

MIN_PART_FRACTION = 1e-6
"""Parts of a repaired polygon with less than this fraction of its area are dropped (zero-area lobes)."""


def invalid_indices(geometries: Sequence[Any]) -> np.ndarray:
    """Indices of the geometries that are not valid (empty geometries count as valid)."""
    arr = np.empty(len(geometries), dtype=object)
    arr[:] = list(geometries)
    return np.flatnonzero(~shapely.is_valid(arr))


def _polygon_parts(geometry: Any) -> list[Polygon]:
    """Polygons of a geometry or geometry collection."""
    if isinstance(geometry, Polygon):
        return [] if geometry.is_empty else [geometry]
    if isinstance(geometry, MultiPolygon) or hasattr(geometry, "geoms"):
        return [part for member in geometry.geoms for part in _polygon_parts(member)]
    return []


def repair_polygons(geometries: Sequence[Any], indices: Sequence[int] | np.ndarray | None = None) -> list[Any]:
    """Valid replacements for the invalid geometries of a sequence.

    Each invalid geometry goes through :func:`shapely.make_valid`; the polygonal parts are kept, and
    parts with less than a millionth of the original area (the zero-area lobes of a pinched ring) are
    dropped. The result is a Polygon if one part remains and a MultiPolygon otherwise. A geometry
    that has no polygonal part left is returned unchanged.

    Parameters
    ----------
    geometries : sequence of shapely geometries
    indices : sequence or array of int, optional
        Geometries to repair (default: every invalid geometry).

    Returns
    -------
    list
        The geometries, with the repaired ones replaced.
    """
    out = list(geometries)
    for i in invalid_indices(out) if indices is None else indices:
        geometry = out[i]
        area = geometry.area
        parts = [p for p in _polygon_parts(shapely.make_valid(geometry)) if p.area > MIN_PART_FRACTION * area]
        if parts:
            out[i] = parts[0] if len(parts) == 1 else MultiPolygon(parts)
    return out
