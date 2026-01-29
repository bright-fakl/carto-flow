"""
===========================================================================
Quality metrics and validation for cartogram results
===========================================================================

Validation utilities to assess cartogram quality and detect issues.

Main Components
---------------
- QualityReport: Dataclass containing validation results
- validate_result: Comprehensive validation of morphing result
- check_topology: Detect self-intersections and invalid geometries

Example
-------
>>> from carto_flow.shape_morpher import morph_gdf, MorphOptions
>>> from carto_flow.shape_morpher.metrics import validate_result
>>>
>>> result = morph_gdf(gdf, 'population', options=MorphOptions.preset_fast())
>>> report = validate_result(result)
>>> print(f"Valid: {report.is_valid}, Mean error: {report.mean_error:.1%}")
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .result import MorphResult

__all__ = [
    "QualityReport",
    "check_topology",
    "validate_result",
]


@dataclass
class QualityReport:
    """Quality assessment report for cartogram results.

    Attributes
    ----------
    is_valid : bool
        True if all geometries are valid (no self-intersections)
    has_self_intersections : bool
        True if any geometry has self-intersections
    topology_preserved : bool
        True if no geometries became invalid during morphing
    invalid_geometry_indices : list[int]
        Indices of geometries that are invalid
    mean_error : float
        Mean area error across all geometries (from result)
    max_error : float
        Maximum area error (from result)
    warnings : list[str]
        List of warning messages about potential issues
    """

    is_valid: bool
    has_self_intersections: bool
    topology_preserved: bool
    invalid_geometry_indices: list[int] = field(default_factory=list)
    mean_error: float = 0.0
    max_error: float = 0.0
    warnings: list[str] = field(default_factory=list)


def validate_result(result: "MorphResult") -> QualityReport:
    """Comprehensive validation of morphing result.

    Checks topology (self-intersections) and reports area errors from
    the result. Area errors are computed during morphing and stored in
    result.final_mean_error, result.final_max_error, and result.final_area_errors.

    Parameters
    ----------
    result : MorphResult
        Morphing result to validate

    Returns
    -------
    QualityReport
        Detailed quality assessment report

    Examples
    --------
    >>> report = validate_result(result)
    >>> if not report.is_valid:
    ...     print(f"Invalid geometries: {report.invalid_geometry_indices}")
    >>> print(f"Mean error: {report.mean_error:.1%}")
    >>> print(f"Max error: {report.max_error:.1%}")
    """
    warnings = []

    # Check topology
    topology_ok, invalid_indices = check_topology(result.geometries)

    has_self_intersections = len(invalid_indices) > 0
    is_valid = topology_ok

    if has_self_intersections:
        warnings.append(f"Found {len(invalid_indices)} invalid geometries")

    # Get error metrics from result (computed during morphing)
    mean_error = result.final_mean_error if result.final_mean_error is not None else 0.0
    max_error = result.final_max_error if result.final_max_error is not None else 0.0

    if max_error > 0.1:
        warnings.append(f"Max area error {max_error:.1%} exceeds 10% threshold")

    # Check convergence status
    if hasattr(result, "status"):
        if str(result.status) == "stalled":
            warnings.append("Algorithm stalled before convergence")

    return QualityReport(
        is_valid=is_valid,
        has_self_intersections=has_self_intersections,
        topology_preserved=not has_self_intersections,
        invalid_geometry_indices=invalid_indices,
        mean_error=mean_error,
        max_error=max_error,
        warnings=warnings,
    )


def check_topology(geometries: Any) -> tuple[bool, list[int]]:
    """Check for self-intersections and invalid geometries.

    Parameters
    ----------
    geometries : GeoDataFrame or array-like
        Geometries to check

    Returns
    -------
    tuple[bool, list[int]]
        (all_valid, invalid_indices) where:
        - all_valid: True if all geometries are valid
        - invalid_indices: List of indices with invalid geometries

    Examples
    --------
    >>> is_valid, invalid = check_topology(result.geometries)
    >>> if not is_valid:
    ...     print(f"Found {len(invalid)} invalid geometries")
    """
    invalid_indices = []

    # Handle GeoDataFrame
    if hasattr(geometries, "geometry"):
        geom_list = geometries.geometry
    else:
        geom_list = geometries

    for i, geom in enumerate(geom_list):
        if (hasattr(geom, "is_valid") and not geom.is_valid) or (hasattr(geom, "is_simple") and not geom.is_simple):
            invalid_indices.append(i)

    all_valid = len(invalid_indices) == 0
    return all_valid, invalid_indices
