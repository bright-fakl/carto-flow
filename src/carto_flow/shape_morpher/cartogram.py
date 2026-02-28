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
>>> from carto_flow import morph_gdf
>>>
>>> cartogram = morph_gdf(gdf, 'population')
>>> print(f"Status: {cartogram.status}")
>>> print(f"Mean error: {cartogram.get_errors().mean_error_pct:.1f}%")
>>> cartogram.plot()
>>> gdf_result = cartogram.to_geodataframe()
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    import geopandas as gpd

    from .errors import MorphErrors
    from .grid import Grid
    from .history import CartogramSnapshot, ConvergenceHistory, ErrorRecord, History
    from .options import MorphOptions, MorphStatus

__all__ = ["Cartogram"]


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
        at saved iterations. Access final results via `latest` property.
    status : MorphStatus
        Computation status (ORIGINAL, CONVERGED, STALLED, COMPLETED, FAILED).
    niterations : int
        Number of iterations completed.
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

    Examples
    --------
    >>> cartogram = morph_gdf(gdf, 'population')
    >>> print(f"Status: {cartogram.status}")
    >>> print(f"Mean error: {cartogram.get_errors().mean_error_pct:.1f}%")
    >>> cartogram.plot()
    >>> gdf_result = cartogram.to_geodataframe()
    """

    # Core results
    snapshots: "History"
    convergence: Optional["ConvergenceHistory"] = None
    status: "MorphStatus" = None  # type: ignore[assignment]

    # Computation metadata
    niterations: int = 0
    duration: float = 0.0
    options: Optional["MorphOptions"] = None
    grid: Optional["Grid"] = None
    target_density: Optional[float] = None
    internals: Optional["History"] = None

    # Source references for GeoDataFrame reconstruction (not shown in repr)
    _source_gdf: Optional[Any] = field(default=None, repr=False)
    _source_landmarks_gdf: Optional[Any] = field(default=None, repr=False)

    # ========================================================================
    # Convenience Access
    # ========================================================================

    @property
    def latest(self) -> Optional["CartogramSnapshot"]:
        """Get the most recent snapshot.

        Returns
        -------
        CartogramSnapshot or None
            The latest snapshot, or None if no snapshots exist.
        """
        return self.snapshots.latest() if self.snapshots else None

    def get_errors(self, iteration: Optional[int] = None) -> Optional["MorphErrors"]:
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
        if iteration is None:
            snapshot = self.snapshots.latest() if self.snapshots else None
        else:
            snapshot = self.snapshots.get_snapshot(iteration)
        return snapshot.errors if snapshot else None

    def get_convergence_errors(self, iteration: Optional[int] = None) -> Optional["ErrorRecord"]:
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

    def get_geometry(self, iteration: Optional[int] = None) -> Any:
        """Get geometry sequence for a specific iteration.

        Parameters
        ----------
        iteration : int, optional
            Iteration number. If None, returns latest.

        Returns
        -------
        array-like
            Sequence of geometries.
        """
        if iteration is None:
            snapshot = self.snapshots.latest() if self.snapshots else None
        else:
            snapshot = self.snapshots.get_snapshot(iteration)
        return snapshot.geometry if snapshot else None

    def get_density(self, iteration: Optional[int] = None) -> Optional[np.ndarray]:
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
        if iteration is None:
            snapshot = self.snapshots.latest() if self.snapshots else None
        else:
            snapshot = self.snapshots.get_snapshot(iteration)
        return snapshot.density if snapshot else None

    def get_landmarks(self, iteration: Optional[int] = None) -> Any:
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
        if iteration is None:
            snapshot = self.snapshots.latest() if self.snapshots else None
        else:
            snapshot = self.snapshots.get_snapshot(iteration)
        return snapshot.landmarks if snapshot else None

    def get_coords(self, iteration: Optional[int] = None) -> Any:
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
        if iteration is None:
            snapshot = self.snapshots.latest() if self.snapshots else None
        else:
            snapshot = self.snapshots.get_snapshot(iteration)
        return snapshot.coords if snapshot else None

    # ========================================================================
    # GeoDataFrame Export
    # ========================================================================

    def to_geodataframe(
        self,
        iteration: Optional[int] = None,
        include_errors: bool = True,
        include_density: bool = True,
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
        >>> cartogram = morph_gdf(gdf, 'population')
        >>> result_gdf = cartogram.to_geodataframe()
        >>> result_gdf.plot(column='_morph_error_pct', cmap='RdYlGn_r')
        """
        if self._source_gdf is None:
            raise ValueError(
                "No source GeoDataFrame available. Use CartogramWorkflow.to_geodataframe() or set _source_gdf."
            )

        snapshot = self.snapshots.latest() if iteration is None else self.snapshots.get_snapshot(iteration)

        if snapshot is None:
            raise ValueError(f"No snapshot found for iteration {iteration}")

        output = self._source_gdf.copy()
        output.geometry = snapshot.geometry

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
        column: Optional[str] = None,
        iteration: Optional[int] = None,
        cmap: str = "RdYlGn_r",
        legend: bool = True,
        ax: Optional[Any] = None,
        **kwargs,
    ) -> Any:
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
        matplotlib.axes.Axes
            The plot axes.

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

    def save(self, path: str) -> None:
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
    def load(cls, path: str) -> "Cartogram":
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
        if self.latest and self.latest.geometry is not None:
            try:
                geom_count = len(self.latest.geometry)
            except TypeError:
                geom_count = 1

        # Status and metrics
        status_str = self.status.value if hasattr(self.status, "value") else str(self.status)
        parts = [f"status={status_str}"]

        if self.niterations > 0:
            parts.append(f"iter={self.niterations}")

        if self.duration > 0:
            parts.append(f"duration={self.duration:.2f}s")

        errors = self.get_errors()
        if errors:
            parts.append(f"mean_error={errors.mean_error_pct:.1f}%")

        # Include convergence count if different from snapshots
        if convergence_count > snapshot_count:
            return f"Cartogram(geoms={geom_count}, snapshots={snapshot_count}, convergence={convergence_count}, {', '.join(parts)})"
        return f"Cartogram(geoms={geom_count}, snapshots={snapshot_count}, {', '.join(parts)})"
