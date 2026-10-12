"""
Cartogram result class with rich interface.

A cartogram is the result of morphing geometries to make their areas
proportional to data values. This module provides the Cartogram class
which contains morphed geometries, convergence metrics, and methods
for visualization, export, and serialization.

Classes
-------
Cartogram
    A cartogram with morphed geometries, metadata, and convenience methods.

Examples
--------
>>> from carto_flow import create_flow_cartogram
>>>
>>> cartogram = create_flow_cartogram(gdf, 'population')
>>> print(f"Status: {cartogram.status}")
>>> print(f"Mean error: {cartogram.get_errors().mean_error_pct:.1f}%")
>>> cartogram.plot()
>>> gdf_result = cartogram.to_geodataframe()
"""

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..geo_utils.validity import invalid_indices, repair_polygons

if TYPE_CHECKING:
    from pathlib import Path

    import geopandas as gpd

    from .errors import MorphErrors
    from .grid import Grid
    from .history import CartogramInternalsSnapshot, CartogramSnapshot, ConvergenceHistory, ErrorRecord, History
    from .options import MorphOptions, MorphStatus
    from .plot_results import CartogramPlotResult
    from .timings import Benchmark

__all__ = ["Cartogram"]


def _repaired(geometry: Any) -> Any:
    """The geometries with the invalid ones made valid."""
    if geometry is None:
        return None
    invalid = invalid_indices(geometry)
    return repair_polygons(geometry, invalid) if len(invalid) else geometry


@dataclass
class Cartogram:
    """
    A cartogram with morphed geometries and metadata.

    This class represents the result of a cartogram morphing operation,
    containing the morphed geometries, convergence status, error metrics,
    and methods for visualization, export, and serialization.

    Attributes
    ----------
    snapshots : History
        Collection of CartogramSnapshot objects capturing algorithm state
        at saved iterations. ``best`` is the snapshot of the returned state and ``last`` the last stored one.
    status : MorphStatus
        Computation status (ORIGINAL, CONVERGED, STALLED, COMPLETED, FAILED).
    niterations : int
        Number of iterations run (the final iterate).
    duration : float
        Computation time in seconds.
    options : MorphOptions, optional
        Options used for this computation.
    grid : Grid, optional
        Grid used for computation.
    target_density : float, optional
        Target equilibrium density.
    internals : History, optional
        Internal computation data (if save_internals=True).
    invalid_regions : tuple of int
        Indices of the regions whose returned geometry is not a valid polygon (a self-intersecting
        boundary). ``get_geometry`` and ``to_geodataframe`` return the computed geometries, which keep
        the vertices of the input, unless ``repair=True``.

    Examples
    --------
    >>> cartogram = create_flow_cartogram(gdf, 'population')
    >>> print(f"Status: {cartogram.status}")
    >>> print(f"Mean error: {cartogram.get_errors().mean_error_pct:.1f}%")
    >>> cartogram.plot()
    >>> gdf_result = cartogram.to_geodataframe()
    """

    # Core results
    snapshots: "History[CartogramSnapshot]"
    convergence: Optional["ConvergenceHistory"] = None
    status: "MorphStatus" = None  # type: ignore[assignment]
    # Computation metadata
    niterations: int = 0
    duration: float = 0.0
    options: Optional["MorphOptions"] = None
    grid: Optional["Grid"] = None
    target_density: float | None = None
    internals: Optional["History[CartogramInternalsSnapshot]"] = None
    benchmark: Optional["Benchmark"] = None
    invalid_regions: tuple[int, ...] = ()

    # Source references for GeoDataFrame reconstruction (not shown in repr)
    _source_gdf: Any | None = field(default=None, repr=False)
    _source_landmarks_gdf: Any | None = field(default=None, repr=False)
    _value_column: str | None = field(default=None, repr=False)

    # ========================================================================
    # Convenience Access
    # ========================================================================

    @property
    def best(self) -> Optional["CartogramSnapshot"]:
        """Snapshot of the returned state, or None if no snapshots exist.

        This is the iterate with the lowest convergence score
        ``max(mean_error / mean_tol, max_error / max_tol)``, at ``best_iteration``.
        It is the same snapshot as :attr:`last` when the run converged.
        """
        return self.snapshots.best() if self.snapshots else None

    @property
    def last(self) -> Optional["CartogramSnapshot"]:
        """Last stored snapshot (the final iterate), or None if no snapshots exist."""
        return self.snapshots.last() if self.snapshots else None

    @property
    def best_iteration(self) -> int | None:
        """Iteration of :attr:`best`, or None if no snapshots exist."""
        best = self.best
        return best.iteration if best is not None else None

    @property
    def latest(self) -> Optional["CartogramSnapshot"]:
        """Snapshot of the returned state.

        .. deprecated:: 2.1
            Use :attr:`best`, or :attr:`last` for the last stored snapshot.
        """
        warnings.warn(
            "Cartogram.latest is deprecated; use best (the returned state) or last (the last stored snapshot)",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.best

    def get_errors(self, iteration: int | None = None) -> Optional["MorphErrors"]:
        """Get error metrics for a specific iteration.

        Parameters
        ----------
        iteration : int, optional
            Iteration number. If None, returns latest.

        Returns
        -------
        MorphErrors or None
            Error metrics, or None if not available.
        """
        snapshot = self.best if iteration is None else self.snapshots.get_snapshot(iteration)
        return snapshot.errors if snapshot else None

    def get_convergence_errors(self, iteration: int | None = None) -> Optional["ErrorRecord"]:
        """Get scalar error metrics from convergence history.

        Unlike get_errors() which returns full MorphErrors from snapshots
        (only available at snapshot intervals), this method accesses the
        lightweight convergence history that records scalar metrics for
        every iteration.

        Parameters
        ----------
        iteration : int, optional
            Iteration number. If None, returns the last recorded iteration.

        Returns
        -------
        ErrorRecord or None
            Scalar error metrics, or None if convergence history not available.
        """
        if self.convergence is None or len(self.convergence) == 0:
            return None
        if iteration is None:
            return self.convergence[-1]
        return self.convergence.get_by_iteration(iteration)

    def get_geometry(self, iteration: int | None = None, repair: bool = False) -> Any:
        """Get geometry sequence for a specific iteration.

        Parameters
        ----------
        iteration : int, optional
            Iteration number. If None, returns latest.
        repair : bool, default=False
            Make invalid (self-intersecting) regions valid (:func:`shapely.make_valid`, zero-area
            parts dropped). The repair adds vertices, so the repaired regions no longer correspond
            vertex by vertex to the input and to other cartograms of the same input, and a region
            can become a MultiPolygon. Do not repair geometries that are animated between cartograms.

        Returns
        -------
        array-like
            Sequence of geometries.
        """
        snapshot = self.best if iteration is None else self.snapshots.get_snapshot(iteration)
        if snapshot is None:
            return None
        return _repaired(snapshot.geometry) if repair else snapshot.geometry

    def get_density(self, iteration: int | None = None) -> np.ndarray | None:
        """Get density values for a specific iteration.

        Parameters
        ----------
        iteration : int, optional
            Iteration number. If None, returns latest.

        Returns
        -------
        np.ndarray or None
            Density values per geometry.
        """
        snapshot = self.best if iteration is None else self.snapshots.get_snapshot(iteration)
        return snapshot.density if snapshot else None

    def get_landmarks(self, iteration: int | None = None) -> Any:
        """Get landmark geometries for a specific iteration.

        Parameters
        ----------
        iteration : int, optional
            Iteration number. If None, returns latest.

        Returns
        -------
        array-like or None
            Landmark geometries, or None if not available.
        """
        snapshot = self.best if iteration is None else self.snapshots.get_snapshot(iteration)
        return snapshot.landmarks if snapshot else None

    def get_coords(self, iteration: int | None = None) -> Any:
        """Get displaced coordinates for a specific iteration.

        Parameters
        ----------
        iteration : int, optional
            Iteration number. If None, returns latest.

        Returns
        -------
        array-like or None
            Displaced coordinates, or None if not available.
        """
        snapshot = self.best if iteration is None else self.snapshots.get_snapshot(iteration)
        return snapshot.coords if snapshot else None

    # ========================================================================
    # GeoDataFrame Export
    # ========================================================================

    def to_geodataframe(
        self,
        iteration: int | None = None,
        include_errors: bool = True,
        include_density: bool = True,
        repair: bool = False,
    ) -> "gpd.GeoDataFrame":
        """Export cartogram as GeoDataFrame.

        Creates a GeoDataFrame with the morphed geometries and optionally
        includes error and density columns.

        Parameters
        ----------
        iteration : int, optional
            Which iteration snapshot to export (default: latest).
        include_errors : bool, default=True
            Add '_morph_error_pct' column with percentage errors.
        include_density : bool, default=True
            Add '_morph_density' column with density values.
        repair : bool, default=False
            Make invalid (self-intersecting) regions valid (:func:`shapely.make_valid`, zero-area
            parts dropped). The repair adds vertices, so the repaired regions no longer correspond
            vertex by vertex to the input and to other cartograms of the same input, and a region
            can become a MultiPolygon. Do not repair geometries that are animated between cartograms.

        Returns
        -------
        GeoDataFrame
            Copy of source GeoDataFrame with morphed geometry and metrics.

        Raises
        ------
        ValueError
            If no source GeoDataFrame is available or iteration not found.

        Examples
        --------
        >>> cartogram = create_flow_cartogram(gdf, 'population')
        >>> result_gdf = cartogram.to_geodataframe()
        >>> result_gdf.plot(column='_morph_error_pct', cmap='RdYlGn_r')
        """
        if self._source_gdf is None:
            raise ValueError(
                "No source GeoDataFrame available. Use CartogramWorkflow.to_geodataframe() or set _source_gdf."
            )

        snapshot = self.best if iteration is None else self.snapshots.get_snapshot(iteration)

        if snapshot is None:
            raise ValueError(f"No snapshot found for iteration {iteration}")

        output = self._source_gdf.copy()
        output.geometry = _repaired(snapshot.geometry) if repair else snapshot.geometry

        if include_errors and snapshot.errors is not None:
            output["_morph_error_pct"] = snapshot.errors.errors_pct
        if include_density and snapshot.density is not None:
            output["_morph_density"] = snapshot.density

        return output

    # ========================================================================
    # Visualization
    # ========================================================================

    def plot(
        self,
        column: str | None = None,
        iteration: int | None = None,
        cmap: str = "RdYlGn_r",
        legend: bool = True,
        ax: Any | None = None,
        **kwargs,
    ) -> "CartogramPlotResult":
        """Plot the cartogram.

        Parameters
        ----------
        column : str, optional
            Column to use for coloring. Special values:
            - '_morph_error_pct': Percentage errors (default)
            - '_morph_density': Density values
            - Any column from source GeoDataFrame
            If None, plots without color mapping.
        iteration : int, optional
            Which iteration to plot (default: latest).
        cmap : str, default='RdYlGn_r'
            Matplotlib colormap name.
        legend : bool, default=True
            Show colorbar legend.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates new figure.
        **kwargs
            Additional arguments passed to GeoDataFrame.plot().

        Returns
        -------
        CartogramPlotResult
            Result with the axes and named artist references.

        Examples
        --------
        >>> cartogram.plot()
        >>> cartogram.plot(column='population', cmap='Blues')
        """
        from .visualization import plot_cartogram

        return plot_cartogram(
            self,
            ax=ax,
            column=column,
            iteration=iteration,
            cmap=cmap,
            legend=legend,
            **kwargs,
        )

    # ========================================================================
    # Serialization
    # ========================================================================

    def save(self, path: "str | Path") -> None:
        """Save cartogram to file.

        Parameters
        ----------
        path : str
            Output file path. Supported formats:
            - '.json': JSON format (metadata + GeoJSON geometries)
            - '.gpkg', '.shp', etc.: GeoDataFrame export via to_geodataframe()

        Examples
        --------
        >>> cartogram.save('output/cartogram.json')
        >>> cartogram.save('output/cartogram.gpkg')
        """
        from pathlib import Path

        path = Path(path)

        if path.suffix == ".json":
            from .serialization import save_cartogram

            save_cartogram(self, path)
        else:
            # Export as GeoDataFrame to spatial format
            gdf = self.to_geodataframe()
            gdf.to_file(path)

    @classmethod
    def load(cls, path: "str | Path") -> "Cartogram":
        """Load cartogram from file.

        Parameters
        ----------
        path : str
            Input file path (JSON format).

        Returns
        -------
        Cartogram
            Loaded cartogram.

        Examples
        --------
        >>> cartogram = Cartogram.load('output/cartogram.json')
        """
        from .serialization import load_cartogram

        return load_cartogram(path)

    # ========================================================================
    # Representation
    # ========================================================================

    def __repr__(self) -> str:
        """Concise string representation."""
        snapshot_count = len(self.snapshots.snapshots) if self.snapshots else 0
        convergence_count = len(self.convergence) if self.convergence else 0

        # Geometry count
        geom_count = 0
        if self.best and self.best.geometry is not None:
            try:
                geom_count = len(self.best.geometry)
            except TypeError:
                geom_count = 1

        # Status and metrics
        status_str = self.status.value if hasattr(self.status, "value") else str(self.status)
        parts = [f"status={status_str}"]

        if self.niterations > 0:
            best = self.best_iteration
            suffix = f" (best {best})" if best is not None and best != self.niterations else ""
            parts.append(f"iter={self.niterations}{suffix}")

        if self.duration > 0:
            parts.append(f"duration={self.duration:.2f}s")

        errors = self.get_errors()
        if errors:
            parts.append(f"mean_error={errors.mean_error_pct:.1f}%")

        # Include convergence count if different from snapshots
        if convergence_count > snapshot_count:
            return f"Cartogram(geoms={geom_count}, snapshots={snapshot_count}, convergence={convergence_count}, {', '.join(parts)})"
        return f"Cartogram(geoms={geom_count}, snapshots={snapshot_count}, {', '.join(parts)})"
