"""
===========================================================================
Comparison utilities for cartogram results
===========================================================================

Utilities for comparing original and morphed geometries.

Main Components
---------------
- ComparisonResult: Dataclass containing comparison metrics
- compare_results: Compare two morphing results statistically
- compute_displacement_vectors: Compute centroid displacement vectors

Example
-------
>>> from carto_flow.shape_morpher import morph_gdf
>>> from carto_flow.shape_morpher.comparison import compare_results
>>>
>>> # Compare original to morphed
>>> result = morph_gdf(gdf, 'population')
>>> comparison = compare_results(gdf, result)
>>> print(f"Mean displacement: {comparison.mean_displacement:.2f}")
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .result import MorphResult

__all__ = [
    "ComparisonResult",
    "compare_results",
    "compute_centroid_shifts",
    "compute_displacement_vectors",
]


@dataclass
class ComparisonResult:
    """Result of comparing original geometries to morphed geometries.

    Attributes
    ----------
    centroid_shifts : np.ndarray
        Array of (dx, dy) shifts for each geometry's centroid.
        Positive dx means the centroid moved right (east).
        Positive dy means the centroid moved up (north).
    mean_displacement : float
        Mean displacement distance across all geometries.
    max_displacement : float
        Maximum displacement distance.
    area_changes : np.ndarray
        Relative area change for each geometry (morphed_area / original_area).
        Values > 1 indicate the geometry grew; values < 1 indicate shrinkage.
    mean_area_change : float
        Mean relative area change across all geometries.

    Examples
    --------
    >>> comparison = compare_results(gdf, result)
    >>> print(f"Mean displacement: {comparison.mean_displacement:.2f}")
    >>> df = comparison.to_dataframe()
    >>> df.sort_values('displacement', ascending=False).head(10)
    """

    centroid_shifts: np.ndarray
    mean_displacement: float
    max_displacement: float
    area_changes: np.ndarray
    mean_area_change: float

    def to_dataframe(self) -> pd.DataFrame:
        """Convert per-geometry metrics to a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - dx: X-component of centroid shift
            - dy: Y-component of centroid shift
            - displacement: Euclidean distance moved
            - area_change: Relative area change (morphed/original)

        Examples
        --------
        >>> df = comparison.to_dataframe()
        >>> # Find geometries that moved the most
        >>> df.nlargest(5, 'displacement')
        >>> # Find geometries that grew the most
        >>> df.nlargest(5, 'area_change')

        Join with original GeoDataFrame to see which regions changed:

        >>> comparison = compare_results(gdf, result)
        >>> df = comparison.to_dataframe()
        >>> # Join metrics with original data (same row order)
        >>> gdf_with_metrics = gdf.join(df)
        >>> # Now filter/sort by any column
        >>> gdf_with_metrics.nlargest(5, 'displacement')[['name', 'population', 'displacement']]
        """
        displacements = np.sqrt(self.centroid_shifts[:, 0] ** 2 + self.centroid_shifts[:, 1] ** 2)
        return pd.DataFrame({
            "dx": self.centroid_shifts[:, 0],
            "dy": self.centroid_shifts[:, 1],
            "displacement": displacements,
            "area_change": self.area_changes,
        })


def compare_results(
    original: Any,
    morphed: "MorphResult",
) -> ComparisonResult:
    """Compare original geometries to a morphing result.

    Parameters
    ----------
    original : GeoDataFrame or MorphResult
        Original geometries (GeoDataFrame) or a MorphResult to compare against
    morphed : MorphResult
        Morphing result containing transformed geometries

    Returns
    -------
    ComparisonResult
        Statistical comparison of the two geometry sets

    Examples
    --------
    Compare original GeoDataFrame to morphed result:

    >>> result = morph_gdf(gdf, 'population')
    >>> comparison = compare_results(gdf, result)
    >>> print(f"Mean displacement: {comparison.mean_displacement:.2f}")

    Compare two different morphing results:

    >>> result1 = morph_gdf(gdf, 'population', options=MorphOptions.preset_fast())
    >>> result2 = morph_gdf(gdf, 'population', options=MorphOptions.preset_high_quality())
    >>> comparison = compare_results(result1, result2)
    """
    # Extract geometries from either GeoDataFrame or MorphResult
    gdf1 = original.geometries if hasattr(original, "geometries") else original
    gdf2 = morphed.geometries

    # Compute centroid shifts
    shifts = compute_centroid_shifts(gdf1, gdf2)
    distances = np.sqrt(shifts[:, 0] ** 2 + shifts[:, 1] ** 2)

    # Compute area changes
    areas1 = np.array([g.area for g in gdf1.geometry])
    areas2 = np.array([g.area for g in gdf2.geometry])
    area_changes = np.where(areas1 > 0, areas2 / areas1, 1.0)

    return ComparisonResult(
        centroid_shifts=shifts,
        mean_displacement=float(np.mean(distances)),
        max_displacement=float(np.max(distances)),
        area_changes=area_changes,
        mean_area_change=float(np.mean(area_changes)),
    )


def compute_displacement_vectors(
    original: Any,
    morphed: "MorphResult",
) -> np.ndarray:
    """Compute displacement vectors between original and morphed centroids.

    Parameters
    ----------
    original : GeoDataFrame
        Original GeoDataFrame before morphing
    morphed : MorphResult
        Morphing result containing transformed geometries

    Returns
    -------
    np.ndarray
        Array of shape (n_geometries, 2) containing (dx, dy) displacement vectors

    Examples
    --------
    >>> vectors = compute_displacement_vectors(gdf, result)
    >>> for i, (dx, dy) in enumerate(vectors):
    ...     print(f"Geometry {i}: moved ({dx:.2f}, {dy:.2f})")
    """
    return compute_centroid_shifts(original, morphed.geometries)


def compute_centroid_shifts(
    geom1: Any,
    geom2: Any,
) -> np.ndarray:
    """Compute centroid shifts between two geometry collections.

    Parameters
    ----------
    geom1 : GeoDataFrame or geometry array
        First set of geometries
    geom2 : GeoDataFrame or geometry array
        Second set of geometries

    Returns
    -------
    np.ndarray
        Array of shape (n_geometries, 2) containing (dx, dy) shifts
    """
    # Extract geometry lists
    geoms1 = geom1.geometry if hasattr(geom1, "geometry") else geom1
    geoms2 = geom2.geometry if hasattr(geom2, "geometry") else geom2

    # Compute centroids
    centroids1 = np.array([[g.centroid.x, g.centroid.y] for g in geoms1])
    centroids2 = np.array([[g.centroid.x, g.centroid.y] for g in geoms2])

    # Compute shifts
    return centroids2 - centroids1
