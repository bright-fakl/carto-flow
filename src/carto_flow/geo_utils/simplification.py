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
from shapely.geometry import (
    LinearRing,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
)

if TYPE_CHECKING:
    import geopandas as gpd

__all__ = ["densify_coverage", "simplify_coverage"]

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


def _densify_coords(coords: np.ndarray, max_segment_length: float) -> np.ndarray:
    """Insert vertices along a coordinate sequence, independent of direction.

    Each segment is split into ``ceil(length / max_segment_length)`` equal
    parts, like ``shapely.segmentize``. Unlike ``shapely.segmentize``, the
    inserted points are computed from a *canonical* segment orientation (the
    lexicographically smaller endpoint first), so a segment and its reverse
    yield bit-for-bit identical points. That is what keeps an edge shared by
    two polygons identical on both sides after densification.

    Parameters
    ----------
    coords :
        ``(n, 2)`` array of coordinates.
    max_segment_length :
        Maximum segment length, in the coordinates' units.

    Returns
    -------
    numpy.ndarray
        ``(m, 2)`` array with ``m >= n``; the input vertices are preserved
        unchanged and in order.
    """
    if len(coords) < 2:
        return coords

    start = coords[:-1]
    end = coords[1:]

    # Canonical orientation: lexicographically smaller endpoint first.
    swap = (start[:, 0] > end[:, 0]) | ((start[:, 0] == end[:, 0]) & (start[:, 1] > end[:, 1]))
    a = np.where(swap[:, None], end, start)
    b = np.where(swap[:, None], start, end)

    lengths = np.hypot(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1])
    n_parts = np.maximum(np.ceil(lengths / max_segment_length).astype(np.int64), 1)

    if not np.any(n_parts > 1):
        return coords

    # One output point per part (the part's start), plus the final vertex.
    seg_index = np.repeat(np.arange(len(n_parts)), n_parts)
    offsets = np.concatenate([[0], np.cumsum(n_parts)[:-1]])
    k = np.arange(len(seg_index)) - offsets[seg_index]

    # Parameter measured along the canonical direction, as an exact integer
    # ratio on both sides: traversing the segment the other way turns index
    # ``k`` into ``n - k``, and ``(n - k) / n`` is computed identically there.
    k_canonical = np.where(swap[seg_index], n_parts[seg_index] - k, k)
    t_canonical = k_canonical / n_parts[seg_index]

    points = a[seg_index] + (b[seg_index] - a[seg_index]) * t_canonical[:, None]
    # Keep original vertices exactly: t == 0 rounds to the segment start, but
    # the canonical form may evaluate it as ``a + (b - a) * 1.0``.
    points[k == 0] = start[seg_index[k == 0]]

    return np.vstack([points, coords[-1]])


def _segmentize_exact(geom, max_segment_length: float):
    """Direction-independent replacement for ``shapely.segmentize``.

    Supports (Multi)Polygon, (Multi)LineString and LinearRing; any other
    geometry is returned unchanged.
    """
    if geom is None or geom.is_empty:
        return geom

    gtype = geom.geom_type

    if gtype == "Polygon":
        shell = _densify_coords(np.asarray(geom.exterior.coords), max_segment_length)
        holes = [_densify_coords(np.asarray(r.coords), max_segment_length) for r in geom.interiors]
        return Polygon(shell, holes)
    if gtype == "LinearRing":
        return LinearRing(_densify_coords(np.asarray(geom.coords), max_segment_length))
    if gtype == "LineString":
        return LineString(_densify_coords(np.asarray(geom.coords), max_segment_length))
    if gtype == "MultiPolygon":
        return MultiPolygon([_segmentize_exact(p, max_segment_length) for p in geom.geoms])
    if gtype == "MultiLineString":
        return MultiLineString([_segmentize_exact(p, max_segment_length) for p in geom.geoms])
    if gtype == "GeometryCollection":
        return shapely.geometry.GeometryCollection([_segmentize_exact(p, max_segment_length) for p in geom.geoms])
    return geom


def _segmentize_exact_array(geoms, max_segment_length: float) -> np.ndarray:
    """Apply :func:`_segmentize_exact` to an array of geometries."""
    return np.array([_segmentize_exact(g, max_segment_length) for g in geoms], dtype=object)


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
    max_segment_length: float | None = None,
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
        the triangle they form with their neighbors has an area below
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
    max_segment_length : float, optional
        If given, re-densify the simplified geometries with
        ``shapely.segmentize`` so that no straight segment exceeds this
        length (CRS linear units). Coverage simplification can leave very
        long straight edges (e.g. a state border reduced to two or three
        vertices); algorithms that move vertices rather than edges — such as
        the flow cartogram morph — cannot bend a segment that has no
        interior vertices, so long edges stall convergence. ``segmentize``
        is applied independently per input segment and is deterministic, so
        edges shared between adjacent polygons stay identical after
        densification (no new gaps or overlaps are introduced).

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

    if max_segment_length is not None:
        simplified = _segmentize_exact_array(simplified, max_segment_length)

    result = gdf.copy()
    result.geometry = simplified
    return result


def densify_coverage(gdf: gpd.GeoDataFrame, max_segment_length: float) -> gpd.GeoDataFrame:
    """Insert vertices so no straight segment exceeds ``max_segment_length``.

    Thin wrapper around ``shapely.segmentize``. Useful after simplification
    (e.g. ``simplify_coverage``) for algorithms that move vertices rather
    than edges — such as the flow cartogram morph — which cannot bend a
    segment that has no interior vertices. Long straight segments (e.g.
    Wyoming's borders after a 1000 m ``simplify_coverage`` have ~69 km
    segments) can stall convergence.

    Note that ``shapely.segmentize`` itself is *not* safe on a coverage: it
    interpolates from each polygon's own traversal direction, so the two sides
    of a shared edge end up ~1e-9 units apart and the union of the coverage
    grows invisible near-collinear "spike" rings (1e-11 .. 1e-6 m2 on the US
    census data) that make GEOS overlays collapse. This function instead
    interpolates from a canonical segment orientation and an exact integer
    parameter, so a segment and its reverse produce bit-for-bit identical
    points — no gaps or overlaps are introduced.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame with any geometry type supported by
        ``shapely.segmentize`` (Polygon, MultiPolygon, LineString, etc.).
    max_segment_length : float
        Maximum segment length, in the CRS's linear units.

    Returns
    -------
    GeoDataFrame
        Copy of ``gdf`` with densified geometries. CRS, index, and all
        non-geometry columns are preserved unchanged.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import box
    >>> from carto_flow.geo_utils.simplification import densify_coverage
    >>>
    >>> gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)])
    >>> densified = densify_coverage(gdf, max_segment_length=2)
    >>> len(densified.geometry.iloc[0].exterior.coords) > 5
    True
    """
    result = gdf.copy()
    result.geometry = _segmentize_exact_array(gdf.geometry.to_numpy(), max_segment_length)
    return result
