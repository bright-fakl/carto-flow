"""
High-level API for cartogram generation.

User-facing functions for creating cartograms from GeoDataFrames.
These functions provide simple interfaces returning Cartogram objects
with rich methods for visualization, export, and analysis.

Functions
---------
morph_gdf
    Generate cartogram from GeoDataFrame.
multiresolution_morph
    Multi-resolution cartogram generation.

Examples
--------
>>> from carto_flow import morph_gdf, multiresolution_morph
>>>
>>> # Simple usage
>>> cartogram = morph_gdf(gdf, 'population')
>>> cartogram.plot()
>>> gdf_result = cartogram.to_geodataframe()
>>>
>>> # Multi-resolution for better convergence
>>> cartogram = multiresolution_morph(gdf, 'population', levels=3)
>>> print(f"Status: {cartogram.status}")
"""

from typing import Any, Literal, Optional, Union

from .cartogram import Cartogram
from .options import MorphOptions

__all__ = [
    "morph_gdf",
    "multiresolution_morph",
]


def morph_gdf(
    gdf: Any,
    column: str,
    landmarks: Any = None,
    options: Optional[MorphOptions] = None,
    displacement_coords: Any = None,
    density_per: Optional[Literal["m2", "km2", "ha", "acre", "sqft", "sqmi"]] = None,
) -> Cartogram:
    """
    Generate cartogram from GeoDataFrame.

    Simple interface returning a Cartogram with rich methods for
    visualization, export, and analysis. For advanced workflows
    (refinement, history), use CartogramWorkflow directly.

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

    Returns
    -------
    Cartogram
        Cartogram object with methods for plotting, export, etc.

    Examples
    --------
    >>> # Basic usage
    >>> cartogram = morph_gdf(gdf, 'population')
    >>> cartogram.plot()
    >>>
    >>> # Get density in people per km² (assuming CRS in meters)
    >>> cartogram = morph_gdf(gdf, 'population', density_per='km2')
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

    workflow = CartogramWorkflow(gdf, column, landmarks, displacement_coords, options, density_per=density_per)
    return workflow.morph()


def multiresolution_morph(
    gdf: Any,
    column: str,
    landmarks: Any = None,
    resolution: int = 512,
    levels: int = 3,
    margin: float = 0.5,
    square: bool = True,
    options: Union[MorphOptions, list[MorphOptions], None] = None,
    displacement_coords: Any = None,
    density_per: Optional[Literal["m2", "km2", "ha", "acre", "sqft", "sqmi"]] = None,
) -> Cartogram:
    """
    Multi-resolution cartogram generation.

    Progressively refines the cartogram at increasing grid resolutions
    for better convergence. Returns the final Cartogram. For access to
    intermediate levels, use CartogramWorkflow directly.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input GeoDataFrame.
    column : str
        Data column name.
    landmarks : GeoDataFrame, optional
        Optional landmarks.
    resolution : int, default=512
        Base resolution for highest level grid.
    levels : int, default=3
        Number of resolution levels.
    margin : float, default=0.5
        Grid margin around data bounds.
    square : bool, default=True
        Whether to create square grids.
    options : MorphOptions or list[MorphOptions], optional
        Options per level.
    displacement_coords : array-like, optional
        Displacement field coordinates.
    density_per : {'m2', 'km2', 'ha', 'acre', 'sqft', 'sqmi'}, optional
        Unit for density calculations. If specified, automatically computes
        the appropriate area_scale based on the GeoDataFrame's CRS units.

    Returns
    -------
    Cartogram
        Final cartogram after all resolution levels.

    Examples
    --------
    >>> cartogram = multiresolution_morph(gdf, 'population', levels=3)
    >>> cartogram.plot()
    >>> print(f"Converged: {cartogram.status}")
    >>>
    >>> # With density in people per km²
    >>> cartogram = multiresolution_morph(gdf, 'population', density_per='km2')
    """
    from .workflow import CartogramWorkflow

    workflow = CartogramWorkflow(gdf, column, landmarks, displacement_coords, options, density_per=density_per)
    return workflow.morph_multiresolution(
        resolution=resolution,
        levels=levels,
        margin=margin,
        square=square,
        options=options,
    )
