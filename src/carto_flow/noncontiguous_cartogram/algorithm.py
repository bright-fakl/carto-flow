"""
Non-contiguous (Olson) area cartogram.

Every region keeps its shape and is scaled about its own center to the area that its value asks for, so the
regions do not touch and the map shows the value of a region by its size, as in Olson (1976).
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable
from typing import TYPE_CHECKING, Literal

import numpy as np
from shapely import affinity

from ..geo_utils.validity import check_input
from .result import NoncontiguousCartogram

if TYPE_CHECKING:
    import geopandas as gpd
    from shapely.geometry.base import BaseGeometry

__all__ = ["create_noncontiguous_cartogram"]

SCALE_COLUMN = "_ncont_scale"


def create_noncontiguous_cartogram(
    gdf: gpd.GeoDataFrame,
    column: str,
    *,
    reference: Literal["densest", "largest"] | float = "densest",
    reference_region: Hashable | None = None,
    max_scale: float | None = 1.0,
    multipart: Literal["group", "parts"] = "group",
    validate: bool = True,
) -> NoncontiguousCartogram:
    """Scale every region about its own center so that its area is proportional to its value.

    The area of a region becomes ``value / density`` for one reference density (value per unit of area) that is the
    same for all regions. Each region keeps its shape and is scaled by the square root of the ratio of its new area to
    its area, about the centroid of the region (``multipart="group"``) or about the centroid of each of its parts
    (``multipart="parts"``).

    Parameters
    ----------
    gdf : GeoDataFrame
        Regions (polygons or multipolygons) in a projected coordinate system, so that areas are meaningful.
    column : str
        Column with the value of each region, non-negative. A region with the value 0 becomes an empty geometry.
    reference : {"densest", "largest"} or float, default "densest"
        The reference density.

        * ``"densest"`` — the density of the region with the highest value per area. Its region keeps its size and no
          region grows, so every area is proportional to its value.
        * ``"largest"`` — the density of the region with the largest value, as in the Olson cartogram of the R
          package ``cartogram``. Regions that are denser than that region would have to grow; see *max_scale*.
        * a float — the density to use, in value per unit of area of the coordinate system. A smaller density makes all
          regions larger.
    reference_region : label, optional
        The index label of a region that keeps its size: the reference density is the density of that region.
        Regions that are denser than it would have to grow; see *max_scale*. Use it to leave out an outlier that is
        much denser than the rest (DC among the US states): with New Jersey as the reference only DC is denser. It
        cannot be combined with a *reference* other than the default.
    max_scale : float or None, default 1.0
        The largest linear scale factor of a region. With the default 1.0 no region grows, and a region whose value
        asks for more is kept at its size (a ``UserWarning`` counts them, and ``NoncontiguousCartogram.capped`` shows
        which); with a larger value regions may grow that much (and overlap their neighbors); ``None`` sets no limit.
    multipart : {"group", "parts"}, default "group"
        How a multipolygon is scaled. ``"group"`` scales the region as one shape about its centroid, so that it stays
        a smaller copy of itself and its parts keep their distances in proportion; ``"parts"`` scales each part about
        its own centroid, so that every part stays on its own land and the gaps between the parts grow in proportion
        to the parts.
    validate : bool, default True
        Check the input first (:func:`carto_flow.geo_utils.validate_geometries`): invalid, empty or null geometries raise a ValueError that names the rows; geographic coordinates give a warning.

    Returns

    -------
    NoncontiguousCartogram
        The scaled regions, the scale factor of each region and the reference density.

    Raises
    ------
    ValueError
        If the values are not numbers or not all non-negative and finite, if no region has a positive value and
        area, or if an argument has an unsupported value.

    Notes
    -----
    A reference that keeps the largest value at full size does not work for most data: with the density of the
    region with the largest value, every denser region would have to grow, in the US states by population New Jersey
    to 4.6 times its area and DC to 41.6 times. ``"densest"`` is the reference under which nothing grows, and it makes
    the regions that are much less dense than the densest one small.

    Examples
    --------
    >>> from carto_flow.noncontiguous_cartogram import create_noncontiguous_cartogram
    >>> cartogram = create_noncontiguous_cartogram(states, "Population")
    >>> cartogram.plot()
    """
    if validate:
        check_input(gdf)
    if multipart not in ("group", "parts"):
        raise ValueError(f"multipart must be 'group' or 'parts', got {multipart!r}")
    if max_scale is not None and not max_scale > 0:
        raise ValueError(f"max_scale must be positive or None, got {max_scale!r}")
    if column not in gdf.columns:
        raise ValueError(f"column {column!r} is not in the GeoDataFrame")

    try:
        values = np.asarray(gdf[column], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"column {column!r} must hold numbers") from error
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"column {column!r} must hold finite, non-negative values")

    crs = gdf.crs
    if crs is not None and crs.is_geographic:
        warnings.warn(
            "the coordinate system is geographic, so areas are in square degrees; project the GeoDataFrame first",
            UserWarning,
            stacklevel=2,
        )

    areas = np.asarray(gdf.geometry.area, dtype=float)
    usable = (values > 0) & (areas > 0)
    if not usable.any():
        raise ValueError(f"no region has both a positive value in {column!r} and a positive area")
    density = values[usable] / areas[usable]

    if reference_region is not None:
        if not (isinstance(reference, str) and reference == "densest"):
            raise ValueError("pass either reference or reference_region, not both")
        if reference_region not in gdf.index:
            raise ValueError(f"reference_region {reference_region!r} is not in the index of the GeoDataFrame")
        position = gdf.index.get_loc(reference_region)
        if not isinstance(position, (int, np.integer)):
            raise ValueError(f"reference_region {reference_region!r} is not a unique label of the index")
        if not usable[position]:
            raise ValueError(f"the reference region {reference_region!r} has no positive value and area")
        reference_density = float(values[position] / areas[position])
    elif isinstance(reference, str):
        if reference == "densest":
            reference_density = float(density.max())
        elif reference == "largest":
            largest = values[usable].argmax()
            reference_density = float(density[largest])
        else:
            raise ValueError(f"reference must be 'densest', 'largest' or a number, got {reference!r}")
    else:
        reference_density = float(reference)
        if not reference_density > 0:
            raise ValueError(f"a reference density must be positive, got {reference!r}")

    scale: np.ndarray = np.zeros(len(gdf))
    scale[usable] = np.sqrt(density / reference_density)
    capped: np.ndarray = np.zeros(len(gdf), dtype=bool)
    if max_scale is not None:
        capped = usable & (scale > max_scale * (1 + 1e-9))
        if capped.any():
            warnings.warn(
                f"{int(capped.sum())} region(s) would have to grow beyond max_scale={max_scale:g} to be proportional "
                "to their value and are kept at that scale; use reference='densest', a smaller reference density or a "
                "larger max_scale",
                UserWarning,
                stacklevel=2,
            )
        scale = np.minimum(scale, max_scale)

    geometries = gdf.copy()
    geometries[SCALE_COLUMN] = scale
    geometries[geometries.geometry.name] = [
        _scaled(geometry, factor, multipart) for geometry, factor in zip(gdf.geometry, scale, strict=True)
    ]
    return NoncontiguousCartogram(
        geometries=geometries,
        source=gdf,
        column=column,
        reference_density=reference_density,
        scale=scale,
        capped=capped,
        multipart=multipart,
    )


def _scaled(geometry: BaseGeometry | None, factor: float, multipart: str) -> BaseGeometry | None:
    """``geometry`` scaled by ``factor`` about its centroid, or about the centroid of each of its parts."""
    if geometry is None or geometry.is_empty:
        return geometry
    if factor == 0:
        return type(geometry)()
    if multipart == "parts" and geometry.geom_type.startswith("Multi"):
        return type(geometry)([affinity.scale(part, factor, factor, origin="centroid") for part in geometry.geoms])
    return affinity.scale(geometry, factor, factor, origin="centroid")
