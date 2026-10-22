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

from .._plot_params import deprecated_params
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
    from .validity import VertexLineage

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
        the vertices of the input (with ``validity="windows"`` or ``"quadtree"``: the vertices of the input and the
        inserted ones), unless ``repair=True``.
    vertex_lineage : VertexLineage or None
        With ``validity="windows"`` or ``"quadtree"``, the vertices that the morph inserted into the segments of the
        geometries, and with ``"quadtree"`` the inserted vertices it removed again (``None`` otherwise).
        :meth:`lift_coordinates` lifts the coordinates of an earlier state to the vertices of the returned geometries.
        Landmarks, probes and ``coords`` are separate point sets and are not affected.
    landmark_area_scaling : np.ndarray or None
        Property. With landmarks, the area scaling of the flow along the path of every landmark vertex (the
        exponential of the integral of the divergence of the velocity field along the path) for the returned
        state: a value above 1 means the neighborhood of the landmark was expanded, below 1 compressed. For point
        landmarks there is one value per landmark. Every snapshot has its own (``CartogramSnapshot.landmark_area_scaling``);
        ``get_landmarks(uniform=True)`` uses them.

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
    vertex_lineage: Optional["VertexLineage"] = None

    # Source references for GeoDataFrame reconstruction (not shown in repr)
    _source_gdf: Any | None = field(default=None, repr=False)
    _source_landmarks_gdf: Any | None = field(default=None, repr=False)
    _value_column: str | None = field(default=None, repr=False)
    # Cached for get_landmarks(uniform=True): the region of every landmark, and the distance distributions of the regions
    _source_coords: Any | None = field(default=None, repr=False)
    _probe_region: Any | None = field(default=None, repr=False)
    _landmark_owner: Any | None = field(default=None, repr=False)
    _coords_owner: Any | None = field(default=None, repr=False)
    _distance_distributions: Any | None = field(default=None, repr=False)

    # ========================================================================
    # Convenience Access
    # ========================================================================

    @property
    def landmark_area_scaling(self) -> np.ndarray | None:
        """Area scaling of the flow along the path of every landmark vertex, for the returned state (None without landmarks)."""
        snapshot = self.best
        return snapshot.landmark_area_scaling if snapshot is not None else None

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

    def lift_coordinates(self, coords: Any, state: int | None = None) -> np.ndarray:
        """Lift the vertex coordinates of an earlier state to the vertices of the returned geometries.

        With ``validity="windows"`` or ``"quadtree"`` the morph inserts vertices into segments, so the returned
        geometries have more vertices than the input, and a snapshot recorded in the middle of a run
        (``snapshot_every``) keeps the vertices it had when it was recorded. This method gives the coordinates of such
        a state at the vertices of the returned geometries: every vertex that was inserted later lies between the two
        older vertices of the same ring that it split, so the lifted shape is the same polyline as the original. With
        ``"quadtree"`` the vertices that were removed later are dropped from the lifted state. The returned best
        iterate is the end of the lineage: a snapshot recorded after it is not a state of the lineage and cannot be
        lifted.

        Parameters
        ----------
        coords : array-like of shape (n, 2)
            Coordinates in the order of ``shapely.get_coordinates`` of the geometries of the input of this morph or
            of one of its snapshots. The number of rows must be that of such a state.
        state : int, optional
            Number of insertion and removal rounds that the state has seen (0 for the input). Only needed when the
            number of rows is shared by two states of the run, which removals can cause.

        Returns
        -------
        np.ndarray
            Coordinates of shape (n_final, 2), n_final being the number of vertices of the returned geometries
            (``len(shapely.get_coordinates(self.best.geometry))`` for geometries without a repair).

        Raises
        ------
        ValueError
            If the number of rows is not that of a state of the run, or is that of several states and ``state`` is not
            given.

        Notes
        -----
        Without ``validity="windows"`` or ``"quadtree"`` no vertex is inserted and the coordinates are returned
        unchanged. A cartogram
        of a ladder level starts from the output of the level before: lift through the levels in turn.
        """
        coords = np.asarray(coords, dtype=float)
        return coords if self.vertex_lineage is None else self.vertex_lineage.lift(coords, state)

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

    def get_landmarks(self, iteration: int | None = None, uniform: bool = False) -> Any:
        """Get landmark geometries for a specific iteration.

        Parameters
        ----------
        iteration : int, optional
            Iteration number. If None, returns the returned state.
        uniform : bool, default=False
            Correct the landmarks that were uniform in their region before the morph for the border effect of the
            flow, so that they are uniform in the morphed region again (see
            :mod:`carto_flow.flow_cartogram.landmarks`). Works for every saved iteration, with the area
            scaling stored in its snapshot, and changes smoothly from one iteration to the next, so it can be
            used for the frames of an animation. For a cartogram created with ``create_flow_cartogram`` or
            ``CartogramWorkflow``. Point landmarks are corrected with probe points
            (``MorphOptions.landmark_probes``) or, without them, with themselves; lines and polygons
            (every vertex is moved) need probes. The moves are small, a few percent of the width of the region.

        Returns
        -------
        array-like or None
            Landmark geometries, or None if not available.
        """
        snapshot = self.best if iteration is None else self.snapshots.get_snapshot(iteration)
        if not uniform or snapshot is None or snapshot.landmarks is None:
            return snapshot.landmarks if snapshot else None
        import shapely

        final = np.asarray(snapshot.landmarks, dtype=object)
        if self._source_landmarks_gdf is None:
            raise ValueError(
                "uniform=True needs the source data of a cartogram created with create_flow_cartogram or CartogramWorkflow"
            )
        original = np.asarray(self._source_landmarks_gdf.geometry.values, dtype=object)
        points = bool(np.all(shapely.get_type_id(final) == 0))
        xy = self._uniform_xy(
            snapshot,
            shapely.get_coordinates(final),
            shapely.get_coordinates(original),
            snapshot.landmark_area_scaling,
            "_landmark_owner",
            points,
        )
        return shapely.points(xy) if points else shapely.set_coordinates(final.copy(), xy)

    def _uniform_xy(
        self,
        snapshot: Any,
        xy_final: np.ndarray,
        xy_original: np.ndarray,
        scaling: Any,
        owner_attr: str,
        self_profile: bool,
    ) -> np.ndarray:
        """Positions of the points ``xy_final`` of a snapshot after the correction of the border effect of the flow."""
        import shapely

        from .landmarks import distance_distributions, region_of_points, uniform_landmarks

        if scaling is None or self._source_gdf is None:
            raise ValueError(
                "uniform=True needs the area scaling of the points (not defined for displacement coordinates with "
                "prescale_components unless prescale_coords is set) and the source data of a cartogram created with "
                "create_flow_cartogram or CartogramWorkflow"
            )
        source_regions = np.asarray(self._source_gdf.geometry.values, dtype=object)
        if getattr(self, owner_attr) is None:
            setattr(self, owner_attr, region_of_points(source_regions, xy_original))
        owner = getattr(self, owner_attr)
        if self._distance_distributions is None:
            self._distance_distributions = distance_distributions(self.best.geometry)  # type: ignore[union-attr]
        if (
            snapshot.probe_positions is not None
            and snapshot.probe_area_scaling is not None
            and self._probe_region is not None
        ):
            profile = (snapshot.probe_positions, np.asarray(snapshot.probe_area_scaling), self._probe_region)
        elif self_profile:
            profile = (xy_final, np.asarray(scaling), owner)
        else:
            raise ValueError(
                "uniform=True needs probe points for this kind of points: set MorphOptions(landmark_probes=50)"
            )
        regions = np.asarray(snapshot.geometry, dtype=object)
        return uniform_landmarks(
            regions,
            xy_final,
            owner,
            shapely.area(regions) / shapely.area(source_regions),
            self._distance_distributions,
            *profile,
        )

    def get_coords(self, iteration: int | None = None, uniform: bool = False) -> Any:
        """Get displaced coordinates for a specific iteration.

        Parameters
        ----------
        iteration : int, optional
            Iteration number. If None, returns latest.
        uniform : bool, default=False
            Correct the coordinates for the border effect of the flow (see :meth:`get_landmarks`). Needs probe
            points (``MorphOptions(landmark_probes=50)``).

        Returns
        -------
        array-like or None
            Displaced coordinates, or None if not available.
        """
        snapshot = self.best if iteration is None else self.snapshots.get_snapshot(iteration)
        if not uniform or snapshot is None or snapshot.coords is None:
            return snapshot.coords if snapshot else None
        from .algorithm import _convert_coords_to_input_format, _detect_coordinate_format, _normalize_coordinates

        if self._source_coords is None:
            raise ValueError(
                "uniform=True needs the source data of a cartogram created with create_flow_cartogram or CartogramWorkflow"
            )
        fmt, sz = _detect_coordinate_format(snapshot.coords)
        xy = self._uniform_xy(
            snapshot,
            _normalize_coordinates(snapshot.coords, fmt),
            _normalize_coordinates(self._source_coords, fmt),
            snapshot.coords_area_scaling,
            "_coords_owner",
            False,
        )
        return _convert_coords_to_input_format(xy, fmt, sz)

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

    @deprecated_params(facecolor="column", legend_kwargs="legend_kwds")
    def plot(
        self,
        facecolor: Any = None,
        iteration: int | None = None,
        cmap: Any = None,
        legend: bool = True,
        ax: Any | None = None,
        legend_kwargs: dict | None = None,
        legend_label: str | None = None,
        norm: Any = None,
        center: float | None = None,
        categorical: bool | None = None,
        style: Any = None,
        **kwargs,
    ) -> "CartogramPlotResult":
        """Plot the cartogram.

        Parameters
        ----------
        facecolor : str or array-like, optional
            Fill of the geometries: a column of the source GeoDataFrame, the built-in quantity
            ``"errors_pct"`` (percentage errors, the default if available) or ``"density"``,
            a numeric array with one value per geometry, or a color. See
            :func:`carto_flow.flow_cartogram.visualization.plot_cartogram`.
        iteration : int, optional
            Which iteration to plot (default: latest).
        cmap : str or Colormap, optional
            Colormap of continuous data. Defaults to the palette of the style for the data role, else
            ``'RdYlGn_r'``.
        legend : bool, default=True
            Show colorbar legend.
        legend_kwargs : dict, optional
            Keyword arguments of the legend of the fill.
        legend_label : str, optional
            Label of the colorbar. Defaults to the name of the data.
        norm : Normalize, {"linear", "sqrt", "log", "auto"}, optional
            Normalization of continuous data: an instance, or the scale of the style's normalization. A ``vmin`` or
            ``vmax`` fixes a linear normalization unless a scale is named.
        center : float, optional
            Treat the data as diverging around this value.
        categorical : bool, optional
            Force (True) or prevent (False) the categorical treatment of the data.
        style : PlotStyle or str, optional
            Style of the plot (default: the style of the block or session, else ``"default"``).
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
        >>> cartogram.plot(facecolor='population', cmap='Blues')
        """
        from .visualization import plot_cartogram

        return plot_cartogram(
            self,
            ax=ax,
            facecolor=facecolor,
            iteration=iteration,
            cmap=cmap,
            legend=legend,
            legend_kwargs=legend_kwargs,
            legend_label=legend_label,
            norm=norm,
            center=center,
            categorical=categorical,
            style=style,
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
