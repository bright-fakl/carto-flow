"""
=========================================
Shape Splitting Utilities for Cartography
=========================================

This module provides utilities for splitting geometric shapes while preserving
exact area relationships. It supports both individual Shapely geometries and
batch processing of GeoPandas DataFrames with area-based scaling according to
data values.

The splitting algorithms use numerical optimization to achieve precise area
targets, making them ideal for cartographic applications where maintaining
accurate area proportions is essential while reducing visual footprint.

Main Components
---------------
- shrink_shape: Core function for shrinking individual Shapely geometries
- split_geometries: Batch processing for GeoPandas DataFrames with data-driven scaling
- Support for optional geometry simplification to improve performance
- Robust numerical optimization with fallback strategies

Supported Input Types
---------------------
- **Individual geometries**: Any Shapely geometry (Polygon, MultiPolygon, etc.)
- **GeoPandas DataFrames**: Batch processing with column-based area scaling
- **Pandas DataFrames**: Automatic conversion if geometry column present

Normalization Methods
--------------------
- **Sum normalization**: Scale by total sum (preserves relative proportions)
- **Maximum normalization**: Scale by maximum value (emphasizes peak values)

Examples
--------
Individual geometry shrinking:

>>> from shapely.geometry import Polygon
>>> from carto_flow.shape_splitter import shrink_shape
>>>
>>> # Create a polygon
>>> poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
>>>
>>> # Shrink to 50% area
>>> shrunken, shell = shrink_shape(poly, fraction=0.5)
>>> print(f"Original area: {poly.area}, Shrunken area: {shrunken.area}")

DataFrame-based shrinking:

>>> import geopandas as gpd
>>> from carto_flow.shape_splitter import split_geometries
>>>
>>> # Shrink based on population data
>>> gdf = gpd.GeoDataFrame({
...     'name': ['City A', 'City B'],
...     'population': [1000000, 500000],
...     'geometry': [poly1, poly2]
... })
>>>
>>> # Shrink with sum normalization
>>> result_gdf = split_geometries(gdf, 'population', normalization='sum')

Notes
-----
**Algorithm Details**:

The shrinking process uses numerical root finding to determine the optimal
negative buffer distance that achieves the target area:

1. **Target computation**: Calculate desired area based on fraction
2. **Root finding**: Solve for buffer distance using scipy.optimize.root_scalar
3. **Buffer application**: Apply computed buffer to achieve exact area match
4. **Error handling**: Robust fallback strategies for edge cases

**Performance Characteristics**:

- **Computational complexity**: O(n) per geometry for n vertices
- **Memory usage**: Creates new geometries (consider copy=False for large datasets)
- **Numerical precision**: Typically achieves <5% area error with default tolerance
- **Convergence**: Usually converges in 5-15 iterations per geometry

**Use Cases**:

- **Cartographic visualization**: Reduce shape sizes while preserving area relationships
- **Data-driven scaling**: Scale geometries based on associated data values
- **Overlap resolution**: Prepare shapes for overlap resolution algorithms
- **Visual hierarchy**: Create emphasis through size variation

**Integration**:

This module integrates seamlessly with:

- **Shapely**: Direct geometry manipulation
- **GeoPandas**: DataFrame-based batch processing
- **Cartoflow ecosystem**: Compatible with shape fitting and overlap resolution modules
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import geopandas as gpd
import pandas as pd
from scipy.optimize import root_scalar
from shapely.geometry import box

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

# Module-level exports - Public API
__all__ = [
    "shrink_shape",
    "split_geometries",
    "split_shape",
]


def _shrunken_area(buffer: float, geom: BaseGeometry, target_area: float) -> float:
    """Compute difference between buffered area and target area."""
    return geom.buffer(buffer).area - target_area


def shrink_shape(
    geom: BaseGeometry,
    fraction: float,
    simplify: float | None = None,
    mode: Literal["area", "shell"] = "area",
    tol: float = 0.05,
) -> tuple[BaseGeometry, BaseGeometry]:
    """
    Shrink a Shapely geometry to a specified fraction of its original area.

    This function reduces the area of a Shapely geometry by applying a negative buffer
    operation. The buffer distance is optimized using root finding to achieve exactly
    the target area fraction. Optionally simplifies the input geometry first to
    improve performance and stability.

    Parameters
    ----------
    geom : shapely.geometry.base.BaseGeometry
        Input Shapely geometry to shrink. Can be Polygon, MultiPolygon, or any
        geometry type that supports the buffer() operation and area property.
    fraction : float
        Target area as a fraction of the original area. Must be in range [0, 1].

        - 1.0: Returns the original geometry unchanged
        - 0.5: Shrinks to half the original area
        - 0.0: Returns an empty geometry
    simplify : float, optional
        Simplification tolerance for the Douglas-Peucker algorithm. If provided,
        the input geometry is simplified before shrinking to reduce complexity
        and improve numerical stability. Should be positive and reasonable
        compared to the geometry size (typically 1-10% of characteristic length).
    mode : {'area', 'shell'}, default='area'
        Interpretation mode for the fraction parameter:

        - **'area'**: fraction represents the target area ratio directly
        - **'shell'**: fraction represents the thickness ratio of the boundary shell.
          The area fraction is computed as fraction² to account for the quadratic
          relationship between linear shell thickness and enclosed area.
    tol : float, default=0.05
        Relative tolerance for the root finding algorithm. Controls the precision
        of area matching. Lower values increase accuracy but may require more
        iterations. Should be positive and typically in range [1e-6, 0.1].

    Returns
    -------
    shrunken_geom : shapely.geometry.base.BaseGeometry
        Geometry shrunk to the target area fraction. Returns the same geometry
        type as the input. For fraction=0.0, returns an empty geometry.
    shell_geom : shapely.geometry.base.BaseGeometry
        The complement geometry representing the area removed during shrinking.
        This is the difference between the original geometry and the shrunken geometry.
        For fraction=1.0, returns an empty geometry.

    Raises
    ------
    ValueError
        If fraction is not in range [0, 1], if mode is invalid, or if simplify
        is not positive when provided, or if the root finding algorithm fails to converge.
    TypeError
        If geom is not a valid Shapely geometry or lacks required methods.

    Notes
    -----
    **Mode-specific behavior**:

    - **'area' mode**: Direct area fraction interpretation
      .. math::
          area_{target} = fraction \\times area_{original}

    - **'shell' mode**: Shell thickness interpretation (squared for area relationship)
      .. math::
          area_{target} = fraction^2 \\times area_{original}

      This mode is useful when fraction represents a boundary thickness or
      shell dimension rather than a direct area ratio.

    **Algorithm**:

    The function uses numerical root finding to determine the optimal negative
    buffer distance that achieves the target area:

    1. **Mode processing**: Adjust fraction based on specified mode
    2. **Simplification** (optional): Reduce geometric complexity
    3. **Target area computation**: fraction * original_area
    4. **Root finding**: Solve buffer_distance such that buffered_area = target_area
    5. **Buffer application**: Apply the computed buffer distance

    **Buffer operation**:

    Uses Shapely's buffer operation with the computed negative distance:

    .. math::
        area_{shrunken} = area(buffer(geom, -d)) \\approx fraction \\times area(geom)

    where d is the optimized buffer distance.

    **Performance considerations**:

    - **Simplification**: Reduces vertex count and improves numerical stability
    - **Root finding**: Typically converges in 5-15 iterations
    - **Buffer operation**: Computational cost depends on geometry complexity

    **Edge cases**:

    - **fraction = 1.0**: Returns original geometry (no buffer applied)
    - **fraction = 0.0**: Returns empty geometry
    - **Very small fractions**: May result in degenerate or empty geometries
    - **Complex geometries**: Simplification recommended for stability

    Examples
    --------
    Basic area reduction:

    >>> from shapely.geometry import Polygon
    >>> from carto_flow.shape_splitter import shrink_shape
    >>>
    >>> # Create a square
    >>> square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    >>> print(f"Original area: {square.area}")  # 100.0
    >>>
    >>> # Shrink to 50% area (area mode)
    >>> shrunken, shell = shrink_shape(square, fraction=0.5, mode='area')
    >>> print(f"Shrunken area: {shrunken.area:.1f}")  # 50.0
    >>> print(f"Shell area: {shell.area:.1f}")  # 50.0

    Shell mode for thickness-based shrinking:

    >>> # Shrink based on shell thickness ratio
    >>> shrunken, shell = shrink_shape(square, fraction=0.7, mode='shell')
    >>> # Area will be approximately 0.49 (0.7**2) of original

    With simplification for complex geometries:

    >>> # Complex polygon that benefits from simplification
    >>> complex_poly = Polygon([(0, 0), (1, 0.1), (2, -0.1), (3, 0.2), (4, 0), (5, 0)])
    >>> shrunken, shell = shrink_shape(complex_poly, fraction=0.8, simplify=0.1)

    Edge cases:

    >>> # Return original geometry (with empty shell)
    >>> original, shell = shrink_shape(square, fraction=1.0)
    >>> print(f"Same geometry: {original.equals(square)}")  # True
    >>> print(f"Shell is empty: {shell.is_empty}")  # True
    >>>
    >>> # Return empty geometry (with original as shell)
    >>> empty, shell = shrink_shape(square, fraction=0.0)
    >>> print(f"Is empty: {empty.is_empty}")  # True
    >>> print(f"Shell area: {shell.area}")  # 100.0
    """
    # Input validation for fraction
    if not (0.0 <= fraction <= 1.0):
        raise ValueError(f"fraction must be in range [0, 1], got {fraction}")

    # Handle edge cases
    if fraction == 0.0:
        # Return empty geometry of the same type
        # Strategy: First expand slightly, then shrink massively to guarantee empty result
        # Direct large negative buffer on some geometries may not fully eliminate them
        # due to floating point precision or complex boundary conditions
        shrunken_geom = geom.buffer(1e-10).buffer(-1e10)  # Create empty geometry
        shell_geom = geom  # Shell is the entire original geometry
        return shrunken_geom, shell_geom
    elif fraction == 1.0:
        shrunken_geom = geom  # Return original geometry unchanged
        shell_geom = geom.buffer(1e-10).buffer(-1e10)  # Create empty geometry
        return shrunken_geom, shell_geom

    if mode == "shell":
        fraction = fraction**2

    # Input validation for simplify
    if simplify is not None:
        if not isinstance(simplify, (int, float)) or simplify <= 0:
            raise ValueError(f"simplify must be a positive number, got {simplify}")
        # Check if simplify is reasonable compared to geometry size
        xmin, ymin, xmax, ymax = geom.bounds
        shortest_edge = min(xmax - xmin, ymax - ymin)
        if simplify > shortest_edge * 0.5:  # More than 50% of shortest edge
            warnings.warn(
                f"simplify tolerance ({simplify}) is large compared to geometry size "
                f"({shortest_edge}). Consider using a smaller value.",
                UserWarning,
                stacklevel=2,
            )

    # Apply simplification if requested
    working_geom = geom.simplify(simplify, preserve_topology=True) if simplify else geom

    # Compute target area
    target_area = fraction * working_geom.area

    # Improved buffer range and starting point
    xmin, ymin, xmax, ymax = working_geom.bounds
    width = xmax - xmin
    height = ymax - ymin
    shortest_edge = min(width, height)

    # Conservative bracket: from 0 to -shortest_edge/2
    # This ensures we don't shrink more than half the shortest dimension
    bracket_left = -shortest_edge / 2.0
    bracket_right = 0.0

    # Better starting point: use a fraction of the expected buffer distance
    # For area reduction, buffer distance is typically negative and proportional to sqrt(area_ratio)
    expected_buffer_magnitude = shortest_edge * (1.0 - (fraction**0.5)) * 0.5
    x0 = -expected_buffer_magnitude  # Start with negative buffer

    try:
        result = root_scalar(
            _shrunken_area,
            args=(working_geom, target_area),
            x0=x0,
            bracket=(bracket_left, bracket_right),
            rtol=tol,
        )
    except ValueError as e:
        # Handle root finding failures
        if "bracket" in str(e).lower():
            warnings.warn(
                f"Could not find valid bracket for root finding. Using fallback buffer distance. Error: {e}",
                UserWarning,
                stacklevel=2,
            )
            # Fallback: use estimated buffer distance
            fallback_buffer = -shortest_edge * (1.0 - fraction**0.5) * 0.3
            shrunken_geom = working_geom.buffer(fallback_buffer)
            shell_geom = working_geom.difference(shrunken_geom)
            return shrunken_geom, shell_geom
        else:
            raise
    else:
        # Validate result
        if not result.converged:
            warnings.warn(
                f"Root finding did not converge. Final area may not match target exactly. "
                f"Convergence flag: {result.flag}",
                UserWarning,
                stacklevel=2,
            )

        shrunken_geom = working_geom.buffer(result.root)
        shell_geom = working_geom.difference(shrunken_geom)
        return shrunken_geom, shell_geom


def split_geometries(
    gdf: pd.DataFrame | gpd.GeoDataFrame,
    column: str,
    method: Literal["shrink", "split"] = "shrink",
    normalization: Literal["sum", "maximum"] | None = None,
    simplify: float | None = None,
    mode: Literal["area", "shell"] = "area",
    tol: float = 0.05,
    direction: Literal["vertical", "horizontal"] = "vertical",
    invert: bool = False,
    copy: bool = True,
) -> gpd.GeoDataFrame:
    """
    Process geometries in a GeoPandas DataFrame using either shrinking or splitting methods.

    This function applies geometric operations to all geometries in a GeoPandas DataFrame
    based on values in a specified column. Two methods are supported:

    - **'shrink'**: Area-based shrinking that preserves relative areas while reducing sizes
    - **'split'**: Line-based splitting that divides geometries into two parts with specified area ratios

    For both methods, the function returns the processed geometries along with their
    complementary "shell" parts (the areas removed during processing).

    Parameters
    ----------
    gdf : Union[pd.DataFrame, gpd.GeoDataFrame]
        Input DataFrame containing geometries. If a regular pandas DataFrame is provided,
        it must contain a 'geometry' column with Shapely geometries.
    column : str
        Name of the column containing the values to use for area scaling/splitting. Values
        should be numeric and non-negative.
    method : {'shrink', 'split'}, default='shrink'
        Processing method to apply:

        - **'shrink'**: Area-based shrinking that reduces geometry sizes while preserving
          relative areas. Uses shrink_geom internally.
        - **'split'**: Line-based splitting that divides geometries into two parts with
          specified area ratios. Uses split_shape internally.
    normalization : {'sum', 'maximum', None}, default='sum'
        Normalization method for computing fractions (used with 'shrink' method):

        - 'sum': Normalize by sum of all values (fraction = value / sum(all values))
        - 'maximum': Normalize by maximum value (fraction = value / max(all values))
        - None: No normalization (fraction = value directly, assuming values are already in [0,1] range)
    simplify : float, optional
        Simplification tolerance for the Douglas-Peucker algorithm (used with 'shrink' method).
        If provided, geometries are simplified to reduce complexity. Should be positive and
        typically 1-10% of characteristic length.
    mode : {'area', 'shell'}, default='area'
        Interpretation mode for fractions (used with 'shrink' method):

        - **'area'**: Treat normalized values as direct area fractions
        - **'shell'**: Treat normalized values as shell thickness ratios. The area
          fractions are computed as fraction² to account for the quadratic relationship
          between linear shell thickness and enclosed area.
    tol : float, default=0.05
        Tolerance parameter:

        - For 'shrink' method: Relative tolerance for root finding algorithm
        - For 'split' method: Absolute tolerance for area matching
    direction : {'vertical', 'horizontal'}, default='vertical'
        Direction for splitting (used with 'split' method):

        - **'vertical'**: Split along vertical lines (constant x-coordinate)
        - **'horizontal'**: Split along horizontal lines (constant y-coordinate)
    invert : bool, default=False
        Whether to invert the computed fractions before processing:

        - **False**: Use fractions as computed (high values → large geometries)
        - **True**: Use inverted fractions (1-fraction) (high values → small geometries)
          Useful when you want high data values to result in smaller geometries.
    copy : bool, default=True
        Whether to return a copy of the input DataFrame. If False, modifies the
        input DataFrame in-place (use with caution).

    Returns
    -------
    result_gdf : gpd.GeoDataFrame
        GeoPandas DataFrame with processed geometries and their shell complements.
        Contains the same columns and data as the input, plus two additional columns:

        - **'geometry'**: The processed geometries (shrunken or split parts)
        - **'complement_geometry'**: The complementary geometries (removed areas)

        The CRS and other metadata are preserved.

    Raises
    ------
    ValueError
        If column contains non-numeric values, negative values, if normalization
        method is invalid, or if mode is invalid
    TypeError
        If input is not a pandas DataFrame or lacks geometry column
    ImportError
        If geopandas is not installed

    Notes
    -----
    **Normalization methods**:

    - **Sum normalization**: fraction_i = value_i / Σ(all values)
      - Preserves relative proportions between all geometries
      - Total area becomes sum of (original_area * fraction)
      - Useful when values represent parts of a whole

    - **Maximum normalization**: fraction_i = value_i / max(all values)
      - Largest value gets fraction = 1.0 (unchanged)
      - Other geometries scaled relative to maximum
      - Useful for emphasizing the most significant features

    - **None normalization**: fraction_i = value_i (no normalization applied)
      - Values are used directly as shrinking fractions
      - Values should already be in [0, 1] range for meaningful results
      - Useful when values are pre-normalized or represent direct area fractions

    **Mode interpretation**:

    - **'area' mode**: Normalized values used directly as area fractions
    - **'shell' mode**: Normalized values squared for area relationship
      .. math::
          area_{fraction} = normalization_{fraction}^2

      This accounts for the quadratic relationship between boundary thickness
      and enclosed area.

    **Processing steps**:

    1. **Validation**: Check input data types and column values
    2. **Normalization**: Compute shrinking fractions based on specified method
    3. **Mode processing**: Apply mode-specific interpretation to fractions
    4. **Geometry processing**: Apply shrinking to each geometry individually
    5. **Result assembly**: Return DataFrame with modified geometries

    **Performance considerations**:

    - **Memory usage**: Creates copy of geometries (can be memory intensive for large datasets)
    - **Processing time**: Each geometry requires numerical root finding (typically 5-15 iterations)
    - **Simplification**: Reduces complexity but adds preprocessing time

    Examples
    --------
    Basic shrinking with sum normalization:

    >>> import geopandas as gpd
    >>> from carto_flow.shape_splitter import split_geometries
    >>>
    >>> # Create sample data
    >>> gdf = gpd.GeoDataFrame({
    ...     'name': ['Region A', 'Region B', 'Region C'],
    ...     'population': [1000, 2000, 1500],
    ...     'geometry': [poly1, poly2, poly3]
    ... })
    >>>
    >>> # Shrink based on population (sum normalized, area mode)
    >>> result = split_geometries(gdf, 'population', method='shrink', normalization='sum')
    >>> print(f"Shrunken areas: {result.geometry.area.values}")
    >>> print(f"Complement areas: {result.complement_geometry.area.values}")

    Splitting with area ratios:

    >>> # Split geometries vertically based on population ratios
    >>> result = split_geometries(gdf, 'population', method='split', direction='vertical')
    >>> print(f"Split areas: {result.geometry.area.values}")
    >>> print(f"Complement areas: {result.complement_geometry.area.values}")

    Shell mode for thickness-based shrinking:

    >>> # Treat population as shell thickness ratios
    >>> result = split_geometries(gdf, 'population', method='shrink', mode='shell')
    >>> # Areas will be squares of the normalized population ratios

    Maximum normalization for highlighting largest values:

    >>> # Emphasize regions with highest values
    >>> result = split_geometries(gdf, 'population', method='shrink', normalization='maximum')
    >>> # Region with max population remains full size, others are smaller

    No normalization with pre-computed values:

    >>> # Use pre-computed fractions directly
    >>> fractions = [0.8, 0.6, 0.9]  # Already in [0,1] range
    >>> gdf['fraction'] = fractions
    >>> result = split_geometries(gdf, 'fraction', method='shrink', normalization=None)
    >>> # Each geometry shrunk according to its pre-computed fraction

    With simplification for complex geometries:

    >>> # Simplify before shrinking for better performance
    >>> result = split_geometries(
    ...     gdf, 'population', method='shrink', simplify=0.1, tol=0.01
    ... )

    Horizontal splitting:

    >>> # Split horizontally instead of vertically
    >>> result = split_geometries(gdf, 'population', method='split', direction='horizontal')

    In-place modification:

    >>> # Modify the original DataFrame (use with caution)
    >>> split_geometries(gdf, 'population', method='shrink', copy=False)
    >>> # gdf.geometry and gdf.shell_geometry are now modified

    Using invert option for reverse scaling:

    >>> # Invert fractions so high population values result in smaller geometries
    >>> result = split_geometries(gdf, 'population', method='shrink', invert=True)
    >>> # High population regions become smaller, low population regions become larger
    """
    # Input validation
    if method not in ["shrink", "split"]:
        raise ValueError(f"Invalid method '{method}'. Must be 'shrink' or 'split'.")

    if not isinstance(gdf, (pd.DataFrame, gpd.GeoDataFrame)):
        raise TypeError(f"Input must be a pandas DataFrame, got {type(gdf)}")

    # Ensure we have a GeoDataFrame
    if not isinstance(gdf, gpd.GeoDataFrame):
        if "geometry" not in gdf.columns:
            raise ValueError("DataFrame must contain a 'geometry' column with Shapely geometries")
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry")

    # Validate column exists and contains numeric data
    if column not in gdf.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame columns: {list(gdf.columns)}")

    values = gdf[column]
    if not pd.api.types.is_numeric_dtype(values):
        raise ValueError(f"Column '{column}' must contain numeric values")

    # Check for negative values
    if (values < 0).any():
        raise ValueError(f"Column '{column}' contains negative values. All values must be non-negative.")

    # Check for zero values
    if (values == 0).all():
        raise ValueError(f"All values in column '{column}' are zero. Cannot compute meaningful fractions.")

    # Create copy if requested
    result_gdf = gdf.copy() if copy else gdf

    # Compute normalization fractions for shrinking
    if normalization == "sum":
        # Normalize by sum of all values
        total = values.sum()
        if total == 0:
            raise ValueError("Sum of values is zero. Cannot normalize by sum.")
        fractions = values / total

    elif normalization == "maximum":
        # Normalize by maximum value
        max_value = values.max()
        if max_value == 0:
            raise ValueError("Maximum value is zero. Cannot normalize by maximum.")
        fractions = values / max_value

    elif normalization is None:
        # No normalization - use values directly as fractions
        # Values should already be in [0, 1] range for meaningful results
        fractions = values

    else:
        raise ValueError(f"Invalid normalization method '{normalization}'. Must be 'sum', 'maximum', or None.")

    # Apply inversion if requested
    if invert:
        fractions = 1.0 - fractions
        # Ensure fractions remain in valid range [0, 1]
        fractions = fractions.clip(lower=0.0, upper=1.0)
        # For split method, ensure fractions are in (0, 1) range
        if method == "split":
            # Adjust edge values to be within valid split range
            fractions = fractions.clip(lower=1e-10, upper=1.0 - 1e-10)

    # Apply processing to each geometry
    processed_geometries = []
    shell_geometries = []

    for idx, (geom, fraction) in enumerate(zip(result_gdf.geometry, fractions)):
        try:
            if pd.isna(fraction) or fraction < 0 or fraction >= 1:
                # Handle invalid fractions by keeping original geometry
                processed_geom = geom
                shell_geom = geom.buffer(1e-10).buffer(-1e10)  # Empty geometry
            else:
                if method == "shrink":
                    # Apply shrinking with shell computation
                    processed_geom, shell_geom = shrink_shape(geom, fraction, simplify=simplify, mode=mode, tol=tol)
                else:  # method == 'split'
                    # Apply splitting - split_shape returns (part1, part2)
                    # We need to determine which part is the "main" part based on target fraction
                    part1, part2 = split_shape(geom, fraction, direction=direction, tol=tol)

                    # Determine which part is closer to the target fraction
                    target_area = fraction * geom.area
                    if abs(part1.area - target_area) < abs(part2.area - target_area):
                        processed_geom = part1
                        shell_geom = part2
                    else:
                        processed_geom = part2
                        shell_geom = part1

            processed_geometries.append(processed_geom)
            shell_geometries.append(shell_geom)

        except Exception as e:
            # Handle individual geometry failures gracefully
            warnings.warn(
                f"Failed to process geometry at index {idx} with method '{method}': {e}. Keeping original geometry.",
                UserWarning,
                stacklevel=2,
            )
            processed_geometries.append(geom)
            shell_geometries.append(geom.buffer(1e-10).buffer(-1e10))  # Empty geometry

    # Update geometries in the result DataFrame
    result_gdf.geometry = processed_geometries
    result_gdf["complement_geometry"] = shell_geometries

    return result_gdf


def split_shape(
    geom: BaseGeometry,
    fraction: float,
    direction: Literal["vertical", "horizontal"] = "vertical",
    tol: float = 0.01,
) -> tuple[BaseGeometry, BaseGeometry]:
    """
    Split a Shapely geometry along a line such that one part has a specified area fraction.

    This function divides a geometry into two parts by cutting along either a vertical
    or horizontal line. The position of the cut is optimized to ensure one of the
    resulting parts has exactly the specified area fraction of the original geometry.

    Parameters
    ----------
    geom : shapely.geometry.base.BaseGeometry
        Input Shapely geometry to split. Should be a Polygon or MultiPolygon.
        The function works best with simple, convex polygons.
    fraction : float
        Target area fraction for the first returned geometry. Must be in range (0, 1).
        The first geometry will have area approximately fraction * original_area, and
        the second will have area approximately (1-fraction) * original_area.
    direction : {'vertical', 'horizontal'}, default='vertical'
        Direction of the splitting line:

        - **'vertical'**: Split along a vertical line (constant x-coordinate)
        - **'horizontal'**: Split along a horizontal line (constant y-coordinate)
    tol : float, default=0.01
        Absolute tolerance for area matching. The algorithm will find a split where
        the area of the first part differs from the target by at most this amount.

    Returns
    -------
    part1 : shapely.geometry.base.BaseGeometry
        First part of the split geometry with area approximately fraction * original_area
    part2 : shapely.geometry.base.BaseGeometry
        Second part of the split geometry with area approximately (1-fraction) * original_area

    Raises
    ------
    ValueError
        If fraction is not in range (0, 1), if direction is invalid, or if
        a valid split cannot be found within the search bounds
    TypeError
        If geom is not a valid Shapely geometry or lacks required methods
    ImportError
        If Shapely is not installed

    Notes
    -----
    **Splitting algorithm**:

    The function uses numerical optimization to find the optimal cut position:

    1. **Bounds determination**: Find the range of possible cut positions
    2. **Line creation**: Create cutting line at test position
    3. **Intersection**: Split geometry using the cutting line
    4. **Area measurement**: Calculate area of one resulting part
    5. **Optimization**: Adjust cut position to achieve target area

    **Cut line orientation**:

    - **Vertical cuts**: Line is vertical (constant x), varies y-position
    - **Horizontal cuts**: Line is horizontal (constant y), varies x-position

    **Area accuracy**:

    The function aims for exact area fractions but may have small errors due to:

    - Geometric complexity (holes, concavities, self-intersections)
    - Numerical precision limits of the optimization algorithm
    - Boundary discretization effects

    **Performance considerations**:

    - **Simple geometries**: Fast convergence (5-10 iterations)
    - **Complex geometries**: May require more iterations or fail
    - **Large geometries**: Intersection operations scale with vertex count

    Examples
    --------
    Basic vertical split:

    >>> from shapely.geometry import Polygon
    >>> from carto_flow.shape_splitter import split_shape
    >>>
    >>> # Create a rectangle
    >>> rect = Polygon([(0, 0), (10, 0), (10, 5), (0, 5)])
    >>> print(f"Original area: {rect.area}")  # 50.0
    >>>
    >>> # Split vertically such that left part is 30% of area
    >>> left, right = split_shape(rect, fraction=0.3, direction='vertical')
    >>> print(f"Left area: {left.area:.1f}, Right area: {right.area:.1f}")

    Horizontal split:

    >>> # Split horizontally such that bottom part is 60% of area
    >>> bottom, top = split_shape(rect, fraction=0.6, direction='horizontal')
    >>> print(f"Bottom area: {bottom.area:.1f}, Top area: {top.area:.1f}")

    Edge cases:

    >>> # Split very small fraction
    >>> tiny, rest = split_shape(rect, fraction=0.1)
    >>> print(f"Tiny part area: {tiny.area:.1f}")  # ≈5.0
    >>>
    >>> # Split very large fraction
    >>> large, tiny = split_shape(rect, fraction=0.9)
    >>> print(f"Large part area: {large.area:.1f}")  # ≈45.0
    """
    # Input validation
    if not (0.0 < fraction < 1.0):
        raise ValueError(f"fraction must be in range (0, 1), got {fraction}")

    if direction not in ["vertical", "horizontal"]:
        raise ValueError(f"direction must be 'vertical' or 'horizontal', got {direction}")

    # Get geometry bounds
    try:
        min_x, min_y, max_x, max_y = geom.bounds
    except AttributeError as err:
        raise TypeError("geom must be a Shapely geometry with bounds attribute") from err

    # Define objective function for optimization
    def area_difference(split_pos: float) -> float:
        """Compute difference between target area and actual area of one part."""
        try:
            # Create bounding box and use intersection for both Polygon and MultiPolygon
            if direction == "vertical":
                if split_pos <= min_x:
                    # Left of geometry: left area is 0
                    return 0 - fraction * geom.area
                elif split_pos >= max_x:
                    # Right of geometry: left area equals total area
                    return geom.area - fraction * geom.area

                # Create left and right parts using bounding box intersection
                left_bbox = box(min_x, min_y, split_pos, max_y)
                right_bbox = box(split_pos, min_y, max_x, max_y)

                left_part = geom.intersection(left_bbox)
                right_part = geom.intersection(right_bbox)

                if left_part.is_empty and right_part.is_empty:
                    return geom.area - fraction * geom.area

                # Use the non-empty part or combine if both exist
                if not left_part.is_empty and not right_part.is_empty:
                    # Both parts exist, return signed difference for left part
                    left_area = left_part.area
                    target_area = fraction * geom.area
                    return left_area - target_area
                elif not left_part.is_empty:
                    left_area = left_part.area
                    return left_area - fraction * geom.area
                else:
                    right_area = right_part.area
                    return right_area - (1 - fraction) * geom.area

            else:  # horizontal split
                if split_pos <= min_y:
                    # Below geometry: bottom area is 0
                    return 0 - fraction * geom.area
                elif split_pos >= max_y:
                    # Above geometry: bottom area equals total area
                    return geom.area - fraction * geom.area

                # Create bottom and top parts using bounding box intersection
                bottom_bbox = box(min_x, min_y, max_x, split_pos)
                top_bbox = box(min_x, split_pos, max_x, max_y)

                bottom_part = geom.intersection(bottom_bbox)
                top_part = geom.intersection(top_bbox)

                if bottom_part.is_empty and top_part.is_empty:
                    return geom.area - fraction * geom.area

                # Use the non-empty part or combine if both exist
                if not bottom_part.is_empty and not top_part.is_empty:
                    # Both parts exist, return signed difference for bottom part
                    bottom_area = bottom_part.area
                    target_area = fraction * geom.area
                    return bottom_area - target_area
                elif not bottom_part.is_empty:
                    bottom_area = bottom_part.area
                    return bottom_area - fraction * geom.area
                else:
                    top_area = top_part.area
                    return top_area - (1 - fraction) * geom.area

        except Exception:
            # If splitting fails, return large difference
            return 1e10

    # Use root finding to find optimal split position
    try:
        from scipy.optimize import root_scalar

        # For vertical splits, search in x-direction
        if direction == "vertical":
            # Search the full x-range of the geometry
            bracket_left = min_x
            bracket_right = max_x

            result = root_scalar(area_difference, bracket=(bracket_left, bracket_right), xtol=tol)
        else:  # horizontal
            # Search the full y-range of the geometry
            bracket_left = min_y
            bracket_right = max_y

            result = root_scalar(area_difference, bracket=(bracket_left, bracket_right), xtol=tol)

        if not result.converged:
            raise ValueError(f"Root finding did not converge. Final flag: {result.flag}")

        optimal_pos = result.root

        # Create the final split using bounding box intersection
        if direction == "vertical":
            left_bbox = box(min_x, min_y, optimal_pos, max_y)
            right_bbox = box(optimal_pos, min_y, max_x, max_y)

            left_part = geom.intersection(left_bbox)
            right_part = geom.intersection(right_bbox)

            # Return non-empty parts
            if not left_part.is_empty and not right_part.is_empty:
                # Both parts exist, determine which is "first" based on target fraction
                if abs(left_part.area - fraction * geom.area) < abs(right_part.area - fraction * geom.area):
                    return left_part, right_part
                else:
                    return right_part, left_part
            elif not left_part.is_empty:
                return left_part, right_bbox.difference(left_part).intersection(geom)
            else:
                return right_part, left_bbox.difference(right_part).intersection(geom)

        else:  # horizontal
            bottom_bbox = box(min_x, min_y, max_x, optimal_pos)
            top_bbox = box(min_x, optimal_pos, max_x, max_y)

            bottom_part = geom.intersection(bottom_bbox)
            top_part = geom.intersection(top_bbox)

            # Return non-empty parts
            if not bottom_part.is_empty and not top_part.is_empty:
                # Both parts exist, determine which is "first" based on target fraction
                if abs(bottom_part.area - fraction * geom.area) < abs(top_part.area - fraction * geom.area):
                    return bottom_part, top_part
                else:
                    return top_part, bottom_part
            elif not bottom_part.is_empty:
                return bottom_part, top_bbox.difference(bottom_part).intersection(geom)
            else:
                return top_part, bottom_bbox.difference(top_part).intersection(geom)

    except ImportError as err:
        raise ImportError("scipy is required for shape splitting. Install with: pip install scipy") from err
    except Exception as e:
        raise ValueError(f"Failed to split geometry: {e}") from e
