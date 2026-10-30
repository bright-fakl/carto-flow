"""
High-level API for cartogram generation.

User-facing function for creating a flow cartogram from a GeoDataFrame. It returns a
Cartogram object with rich methods for visualization, export, and analysis.

Functions
---------
create_flow_cartogram
    Generate a flow cartogram from a GeoDataFrame, at one resolution or on a ladder of resolutions.

Examples
--------
>>> from carto_flow.flow_cartogram import LadderSpec, MorphOptions, create_flow_cartogram
>>>
>>> # Simple usage
>>> cartogram = create_flow_cartogram(gdf, 'population')
>>> cartogram.plot()
>>> gdf_result = cartogram.to_geodataframe()
>>>
>>> # A ladder of resolutions, chosen from the data
>>> cartogram = create_flow_cartogram(gdf, 'population', options=MorphOptions(grid="auto"))
>>> print(f"Status: {cartogram.status}")
"""

from typing import Any, Literal

from .cartogram import Cartogram
from .options import MorphOptions

__all__ = [
    "create_flow_cartogram",
]


def create_flow_cartogram(
    gdf: Any,
    column: str,
    *,
    landmarks: Any = None,
    options: MorphOptions | None = None,
    displacement_coords: Any = None,
    density_per: Literal["m2", "km2", "ha", "acre", "sqft", "sqmi"] | None = None,
    validate: bool = True,
) -> Cartogram:
    """
    Generate a flow cartogram from a GeoDataFrame.

    Returns a Cartogram with rich methods for visualization, export, and
    analysis. ``options.grid`` selects the resolution: an integer or
    ``GridSpec`` for one grid, a ``LadderSpec`` or ``"auto"`` for a ladder of
    resolutions that is run from coarse to fine and ends at the first level
    that converges. For refinement and history, use CartogramWorkflow directly.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame with geometries and data column.
    column : str
        Column name containing values for cartogram generation.
    landmarks : GeoDataFrame, optional
        Optional landmarks to transform alongside main geometries.
    options : MorphOptions, optional
        Algorithm configuration options.
    displacement_coords : array-like, optional
        Coordinates for displacement field computation.
    density_per : {'m2', 'km2', 'ha', 'acre', 'sqft', 'sqmi'}, optional
        Unit for density calculations. If specified, automatically computes
        the appropriate area_scale based on the GeoDataFrame's CRS units.
        For example, if CRS uses meters and density_per='km2', density
        values will be expressed as values per km².
    validate : bool, default=True
        Whether to check the input first (see :func:`~carto_flow.geo_utils.validate_geometries`): invalid, empty or
        null geometries and values that are not finite or are negative raise a ValueError that names the rows;
        overlapping polygons and geographic coordinates give a warning.

    Returns
    -------
    Cartogram
        Cartogram object with methods for plotting, export, etc. For a ladder,
        the result of the last level that ran.

    Examples
    --------
    >>> # Basic usage
    >>> cartogram = create_flow_cartogram(gdf, 'population')
    >>> cartogram.plot()
    >>>
    >>> # A ladder of resolutions, chosen from the data
    >>> cartogram = create_flow_cartogram(gdf, 'population', options=MorphOptions(grid="auto"))
    >>>
    >>> # An explicit ladder: 128, 256 and 512 points along the long axis
    >>> options = MorphOptions(grid=LadderSpec(min_resolution=128, levels=3))
    >>> cartogram = create_flow_cartogram(gdf, 'population', options=options)
    >>>
    >>> # Get density in people per km² (assuming CRS in meters)
    >>> cartogram = create_flow_cartogram(gdf, 'population', density_per='km2')
    >>>
    >>> # Export to GeoDataFrame
    >>> gdf_result = cartogram.to_geodataframe()
    >>>
    >>> # Access metadata
    >>> print(f"Status: {cartogram.status}")
    >>> print(f"Error: {cartogram.get_errors().mean_error_pct:.1f}%")
    >>>
    >>> # Save to file
    >>> cartogram.save('output/cartogram.gpkg')
    """
    from .workflow import CartogramWorkflow

    workflow = CartogramWorkflow(
        gdf, column, landmarks, displacement_coords, options, density_per=density_per, validate=validate
    )
    return workflow.morph()
