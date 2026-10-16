"""Result of a non-contiguous cartogram."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import geopandas as gpd


@dataclass
class NoncontiguousPlotResult:
    """Artists produced by :meth:`NoncontiguousCartogram.plot`."""

    ax: Any
    collections: list
    original_collections: list


@dataclass
class NoncontiguousCartogram:
    """Result of :func:`create_noncontiguous_cartogram`.

    Attributes
    ----------
    geometries : GeoDataFrame
        A copy of the input with the scaled geometries and the column ``"_ncont_scale"``.
    source : GeoDataFrame
        The input regions.
    column : str
        The column that sized the regions.
    reference_density : float
        The value per unit of area at which a region keeps its size (a scale factor of 1).
    scale : np.ndarray, shape (n,)
        The linear scale factor of each region.
    capped : np.ndarray of bool, shape (n,)
        Regions whose scale factor was limited by ``max_scale``, so that their area is smaller than proportional.
    multipart : str
        How multipolygons were scaled, ``"group"`` or ``"parts"``.
    """

    geometries: gpd.GeoDataFrame
    source: gpd.GeoDataFrame
    column: str
    reference_density: float
    scale: np.ndarray
    capped: np.ndarray
    multipart: str

    @property
    def area_error_pct(self) -> np.ndarray:
        """Percent difference of each area from proportional to the value at the reference density.

        0 for every region unless ``max_scale`` limited the scale; NaN for a region with no area or the value 0.
        """
        target = np.asarray(self.source[self.column], dtype=float) / self.reference_density
        actual = np.asarray(self.geometries.geometry.area, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            error = (actual / target - 1.0) * 100.0
        return np.where(target > 0, error, np.nan)

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """The scaled regions with the columns of the input and ``"_ncont_scale"``."""
        return self.geometries.copy()

    def plot(self, *args: Any, **kwargs: Any) -> NoncontiguousPlotResult:
        """Plot the scaled regions; see :func:`carto_flow.noncontiguous_cartogram.visualization.plot_cartogram`."""
        from .visualization import plot_cartogram

        return plot_cartogram(self, *args, **kwargs)
