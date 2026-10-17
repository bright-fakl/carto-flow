"""Check the input of a cartogram, and detect and repair invalid polygons (self-intersecting rings)."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon

__all__ = ["GeometryIssue", "GeometryReport", "invalid_indices", "repair_polygons", "validate_geometries"]

MIN_PART_FRACTION = 1e-6
"""Parts of a repaired polygon with less than this fraction of its area are dropped (zero-area lobes)."""


def invalid_indices(geometries: Sequence[Any]) -> np.ndarray:
    """Indices of the geometries that are not valid (empty geometries count as valid)."""
    arr = np.empty(len(geometries), dtype=object)
    arr[:] = list(geometries)
    return np.flatnonzero(~shapely.is_valid(arr))


def _polygon_parts(geometry: Any) -> list[Polygon]:
    """Polygons of a geometry or geometry collection."""
    if isinstance(geometry, Polygon):
        return [] if geometry.is_empty else [geometry]
    if isinstance(geometry, MultiPolygon) or hasattr(geometry, "geoms"):
        return [part for member in geometry.geoms for part in _polygon_parts(member)]
    return []


def repair_polygons(geometries: Sequence[Any], indices: Sequence[int] | np.ndarray | None = None) -> list[Any]:
    """Valid replacements for the invalid geometries of a sequence.

    Each invalid geometry goes through :func:`shapely.make_valid`; the polygonal parts are kept, and
    parts with less than a millionth of the original area (the zero-area lobes of a pinched ring) are
    dropped. The result is a Polygon if one part remains and a MultiPolygon otherwise. A geometry
    that has no polygonal part left is returned unchanged.

    Parameters
    ----------
    geometries : sequence of shapely geometries
    indices : sequence or array of int, optional
        Geometries to repair (default: every invalid geometry).

    Returns
    -------
    list
        The geometries, with the repaired ones replaced.
    """
    out = list(geometries)
    for i in invalid_indices(out) if indices is None else indices:
        geometry = out[i]
        area = geometry.area
        parts = [p for p in _polygon_parts(shapely.make_valid(geometry)) if p.area > MIN_PART_FRACTION * area]
        if parts:
            out[i] = parts[0] if len(parts) == 1 else MultiPolygon(parts)
    return out


MAX_LISTED_ROWS = 5
"""Rows named in the message of an issue; the rest is counted."""


@dataclass(frozen=True)
class GeometryIssue:
    """One kind of problem in the input, with the rows it affects.

    Attributes
    ----------
    kind : str
        ``"invalid"``, ``"empty"``, ``"values"``, ``"repeated_vertices"`` or ``"coverage"``.
    severity : {"error", "warning", "info"}
        An error stops the cartogram; a warning is issued and the run goes on; information is only in the report.
    rows : tuple
        Index labels of the affected rows.
    message : str
        Description with the first rows, the reason and a hint for the repair.
    """

    kind: str
    severity: Literal["error", "warning", "info"]
    rows: tuple
    message: str


@dataclass(frozen=True)
class GeometryReport:
    """Result of :func:`validate_geometries`: the problems found in the input, none if it is fine."""

    issues: tuple[GeometryIssue, ...] = ()

    @property
    def errors(self) -> tuple[GeometryIssue, ...]:
        """Issues that stop a cartogram."""
        return tuple(i for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> tuple[GeometryIssue, ...]:
        """Issues that do not stop a cartogram."""
        return tuple(i for i in self.issues if i.severity == "warning")

    @property
    def info(self) -> tuple[GeometryIssue, ...]:
        """Findings that are neither errors nor warnings."""
        return tuple(i for i in self.issues if i.severity == "info")

    @property
    def ok(self) -> bool:
        """True if there is no error (warnings are allowed)."""
        return not self.errors

    def raise_for_errors(self) -> None:
        """Raise a ValueError that describes every error."""
        if self.errors:
            raise ValueError("invalid input: " + " ".join(i.message for i in self.errors))

    def warn(self, stacklevel: int = 2) -> None:
        """Issue a UserWarning for every warning."""
        for issue in self.warnings:
            warnings.warn(issue.message, UserWarning, stacklevel=stacklevel + 1)


def _rows_text(labels: list) -> str:
    shown = ", ".join(repr(x) for x in labels[:MAX_LISTED_ROWS])
    more = len(labels) - MAX_LISTED_ROWS
    return f"{shown}, and {more} more" if more > 0 else shown


def _issue(
    kind: str, severity: Literal["error", "warning", "info"], labels: list, text: str, hint: str = ""
) -> GeometryIssue:
    message = f"{text} in {len(labels)} row{'' if len(labels) == 1 else 's'}: {_rows_text(labels)}."
    return GeometryIssue(kind, severity, tuple(labels), f"{message} {hint}".strip())


def validate_geometries(
    gdf: Any,
    values: str | Sequence[float] | np.ndarray | None = None,
    *,
    coverage: bool = False,
    zero: Literal["error", "ok"] = "error",
) -> GeometryReport:
    """Check the input of a cartogram.

    Errors: null or empty geometries, invalid geometries (with the reason from
    :func:`shapely.explain_validity`), and values that are not finite, negative or (with ``zero="error"``) zero.
    Warnings: with ``coverage=True``, polygons that overlap or whose shared edges do not match
    (:func:`shapely.coverage_invalid_edges`). Information: repeated consecutive vertices, which do not affect a cartogram.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input regions; the report names rows by index label.
    values : str or array-like, optional
        Column name or one value per row, for example the data column of a cartogram.
    coverage : bool, default=False
        Whether to check that the polygons form a valid coverage; for methods that need neighbors to touch.
    zero : {"error", "ok"}, default="error"
        Whether a value of zero is an error.

    Returns
    -------
    GeometryReport
    """
    labels = np.asarray(gdf.index, dtype=object)
    geoms = np.asarray(gdf.geometry.values, dtype=object)
    issues: list[GeometryIssue] = []

    missing = shapely.is_missing(geoms) | shapely.is_empty(geoms)
    if missing.any():
        issues.append(
            _issue(
                "empty",
                "error",
                labels[missing].tolist(),
                "Null or empty geometry",
                "Drop these rows, e.g. gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].",
            )
        )
    present = ~missing

    invalid = present & ~shapely.is_valid(geoms)
    if invalid.any():
        reasons = {labels[i]: shapely.is_valid_reason(geoms[i]) for i in np.flatnonzero(invalid)}
        first = next(iter(reasons))
        issues.append(
            _issue(
                "invalid",
                "error",
                list(reasons),
                f"Invalid geometry (first: {reasons[first]})",
                "Repair with shapely.make_valid or carto_flow.geo_utils.repair_polygons.",
            )
        )

    if isinstance(values, str) and values not in gdf.columns:
        issues.append(GeometryIssue("values", "error", (), f"column {values!r} not found in the GeoDataFrame."))
        values = None
    if values is not None:
        v = gdf[values] if isinstance(values, str) else values
        try:
            arr = np.asarray(v, dtype=float)
        except (TypeError, ValueError):
            issues.append(_issue("values", "error", labels.tolist(), "Value is not a number"))
        else:
            bad = ~np.isfinite(arr) | (arr < 0) | ((arr == 0) if zero == "error" else False)
            if bad.any():
                what = "missing, infinite, negative or zero" if zero == "error" else "missing, infinite or negative"
                issues.append(
                    _issue("values", "error", labels[bad].tolist(), f"Value is {what}", "Fix or drop these rows.")
                )

    ok = present & ~invalid
    if ok.any():
        sub = geoms[ok]
        counts = shapely.get_num_coordinates(sub)
        repeated = counts != shapely.get_num_coordinates(shapely.remove_repeated_points(sub, 0.0))
        if repeated.any():
            issues.append(
                _issue(
                    "repeated_vertices",
                    "info",
                    labels[ok][repeated].tolist(),
                    "Repeated consecutive vertices",
                    "Remove them with shapely.remove_repeated_points.",
                )
            )
        if coverage and np.isin(shapely.get_type_id(sub), (3, 6)).all():
            bad_edges = ~shapely.is_empty(shapely.coverage_invalid_edges(sub))
            if bad_edges.any():
                issues.append(
                    _issue(
                        "coverage",
                        "warning",
                        labels[ok][bad_edges].tolist(),
                        "Polygon overlaps a neighbor or shares an edge that does not match",
                        "Neighbors may be missed and the result may overlap; see shapely.coverage_invalid_edges.",
                    )
                )
    return GeometryReport(tuple(issues))


def check_input(
    gdf: Any,
    values: str | Sequence[float] | np.ndarray | None = None,
    *,
    coverage: bool = False,
    zero: Literal["error", "ok"] = "error",
    stacklevel: int = 2,
) -> None:
    """Run :func:`validate_geometries`, raise for errors and warn for the rest."""
    report = validate_geometries(gdf, values, coverage=coverage, zero=zero)
    report.raise_for_errors()
    report.warn(stacklevel + 1)
