"""Voronoi field backend implementations.

Class hierarchy
---------------
BaseField
    Shared init (points, boundary, weights, topology springs, adhesion).
    _constrain_points, _apply_boundary_adhesion, _topology_spring, get_points.

ExactField(BaseField)
    Exact scipy Voronoi + shapely clipping.

RasterField(BaseField)
    Raster nearest-neighbor Lloyd, optionally with elastic boundary.
"""

from ._base import BaseField, _extract_exact_cells
from ._exact import ExactField
from ._raster import RasterField

__all__ = [
    "BaseField",
    "ExactField",
    "RasterField",
    "_extract_exact_cells",
]
