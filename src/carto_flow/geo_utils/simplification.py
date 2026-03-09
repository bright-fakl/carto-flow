"""
Topology-preserving simplification for polygon GeoDataFrames.

Uses ``shapely.coverage_simplify`` (GEOS 3.12 / Shapely ≥ 2.1) which builds an
internal arc-topology graph so that shared edges between adjacent polygons are
simplified once and applied identically to both sides — no gaps or overlaps are
introduced.

Functions
---------
simplify_coverage
    Simplify polygon geometries while preserving shared boundaries.

Notes
-----
**Known limitation**

Compact small features (e.g. DC in a US states dataset) may collapse to a
near-triangle at high tolerances because Visvalingam-Whyatt removes vertices
proportional to the triangle area they form. Choose ``tolerance`` carefully or
pre-exclude very small features.

Examples
--------
>>> import geopandas as gpd
>>> from shapely.geometry import box
>>> from carto_flow.geo_utils.simplification import simplify_coverage
>>>
>>> gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)])
>>> simplified = simplify_coverage(gdf, tolerance=0.05)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon

if TYPE_CHECKING:
    import geopandas as gpd

__all__ = ["simplify_coverage"]

# Minimum Shapely version required for coverage_simplify
_MIN_SHAPELY_VERSION = (2, 1, 0)


def _check_shapely_version() -> None:
    """Raise ImportError if Shapely < 2.1.0."""
    version = tuple(int(x) for x in shapely.__version__.split(".")[:3])
    if version < _MIN_SHAPELY_VERSION:
        raise ImportError(
            f"simplify_coverage requires Shapely >= 2.1.0 "
            f"(coverage_simplify was added in that release). "
            f"Installed version: {shapely.__version__}"
        )


def _remove_small_parts(
    geom: Polygon | MultiPolygon,
    min_island_area: float | None,
    min_hole_area: float | None,
) -> Polygon | MultiPolygon:
    """Remove small sub-polygons and interior rings from a geometry.

    Parameters
    ----------
    geom :
        Input Polygon or MultiPolygon.
    min_island_area :
        Area threshold below which sub-polygons are dropped (already squared
        from the user-facing ``min_island_size`` parameter).
    min_hole_area :
        Area threshold below which interior rings are dropped (already squared
        from the user-facing ``min_hole_size`` parameter).

    Returns
    -------
    Polygon or MultiPolygon
        Cleaned geometry. Returns the original if no parts were removed.
    """

    def _clean_polygon(poly: Polygon) -> Polygon:
        if min_hole_area is None:
            return poly
        kept_holes = [ring for ring in poly.interiors if Polygon(ring).area >= min_hole_area]
        if len(kept_holes) == len(list(poly.interiors)):
            return poly
        return Polygon(poly.exterior, kept_holes)

    if isinstance(geom, Polygon):
        return _clean_polygon(geom)

    if isinstance(geom, MultiPolygon):
        parts = list(geom.geoms)

        # Drop small sub-polygons
        if min_island_area is not None:
            parts = [p for p in parts if p.area >= min_island_area]
            if not parts:
                # Keep the largest part to avoid empty geometry
                parts = [max(geom.geoms, key=lambda p: p.area)]

        # Clean holes from remaining parts
        parts = [_clean_polygon(p) for p in parts]

        if len(parts) == 1:
            return parts[0]
        return MultiPolygon(parts)

    return geom


def simplify_coverage(
    gdf: gpd.GeoDataFrame,
    tolerance: float,
    min_island_size: float | None = None,
    min_hole_size: float | None = None,
    simplify_outer: bool = True,
) -> gpd.GeoDataFrame:
    """Simplify polygon geometries while preserving shared boundaries.

    Uses ``shapely.coverage_simplify`` (GEOS 3.12 / Shapely ≥ 2.1) which
    internally builds an arc-topology graph: shared edges are simplified once
    and applied to all touching polygons, guaranteeing no gaps or overlaps.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame with Polygon or MultiPolygon geometries.
    tolerance : float
        Simplification tolerance in the CRS coordinate units. Follows the same
        convention as ``shapely.coverage_simplify``: vertices are removed if
        the triangle they form with their neighbours has an area below
        ``tolerance``. Use larger values for more aggressive simplification.
    min_island_size : float, optional
        Before simplification, sub-polygons in MultiPolygon geometries with
        area less than ``min_island_size ** 2`` are discarded. Specified in
        CRS linear units so that it scales consistently with ``tolerance``.
    min_hole_size : float, optional
        Before simplification, interior rings (holes) with area less than
        ``min_hole_size ** 2`` are removed from each polygon. Specified in
        CRS linear units.
    simplify_outer : bool, default True
        If ``True`` (default), both shared interior edges and outer boundary
        edges are simplified. If ``False``, only shared interior edges are
        simplified; the outer boundary of the coverage is left unchanged.
        Passed as ``simplify_boundary`` to ``shapely.coverage_simplify``.

    Returns
    -------
    GeoDataFrame
        Copy of ``gdf`` with simplified geometries. CRS, index, and all
        non-geometry columns are preserved unchanged.

    Raises
    ------
    ImportError
        If Shapely < 2.1.0 is installed (``coverage_simplify`` is not available).

    Warns
    -----
    UserWarning
        If the input contains geometries that are not Polygon or MultiPolygon.

    Notes
    -----
    ``tolerance`` is an area-based threshold passed directly to GEOS's
    Visvalingam-Whyatt coverage simplifier. It is **not** a maximum-deviation
    distance like ``shapely.simplify``'s tolerance. For comparable visual
    results, a rough guideline is ``tolerance ≈ (desired_deviation) ** 2 / 4``.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import box
    >>> from carto_flow.geo_utils.simplification import simplify_coverage
    >>>
    >>> gdf = gpd.GeoDataFrame(
    ...     geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2), box(1, 1, 2, 2)]
    ... )
    >>> simplified = simplify_coverage(gdf, tolerance=0.05)
    >>> # No gaps: union is unchanged
    >>> simplified.union_all().equals(gdf.union_all())
    True
    """
    _check_shapely_version()

    # Pre-compute area thresholds from linear-unit sizes
    min_island_area = min_island_size**2 if min_island_size is not None else None
    min_hole_area = min_hole_size**2 if min_hole_size is not None else None

    # Pre-process geometries
    geoms = []
    has_unsupported = False
    for geom in gdf.geometry:
        if isinstance(geom, (Polygon, MultiPolygon)):
            if min_island_area is not None or min_hole_area is not None:
                geom = _remove_small_parts(geom, min_island_area, min_hole_area)
        elif geom is not None and not geom.is_empty:
            has_unsupported = True
        geoms.append(geom)

    if has_unsupported:
        warnings.warn(
            "simplify_coverage: some geometries are not Polygon or MultiPolygon and will be passed through unchanged.",
            UserWarning,
            stacklevel=2,
        )

    geom_array = np.array(geoms, dtype=object)
    simplified = shapely.coverage_simplify(geom_array, tolerance, simplify_boundary=simplify_outer)

    result = gdf.copy()
    result.geometry = simplified
    return result
