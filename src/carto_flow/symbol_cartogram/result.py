"""Result container for symbol cartogram operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .status import SymbolCartogramStatus

if TYPE_CHECKING:
    import geopandas as gpd
    import matplotlib.pyplot as plt

    from .layout_result import LayoutResult
    from .plot_results import SymbolsPlotResult
    from .styling import Styling


@dataclass
class SimulationHistory:
    """Per-iteration diagnostics and optional position snapshots from simulation.

    All array fields share an iteration axis: index *i* corresponds to
    iteration *i*.  Fields are ``None`` when the simulator does not
    produce them.

    Attributes
    ----------
    positions : list[np.ndarray] | None
        Position snapshots, each of shape ``(n, 2)``.  Only populated
        when ``save_history=True``.
    drift : np.ndarray | None
        Per-iteration drift (mean relative smoothed displacement).
        Populated by ``TopologyPreservingSimulator``.  Shape: ``(n_iters,)``.
    jitter : np.ndarray | None
        Per-iteration jitter (mean relative displacement std).
        Populated by ``TopologyPreservingSimulator``.  Shape: ``(n_iters,)``.
    drift_rate : np.ndarray | None
        EMA-smoothed derivative of drift (drift[k] - drift[k-1]).
        Approaches zero when drift plateaus (system converged to stable
        oscillation).
        Populated by ``TopologyPreservingSimulator``.  Shape: ``(n_iters,)``.
    drift_rate_neg_frac : np.ndarray | None
        Running sign test: EMA of ``1{drift_rate < 0}``.  Approaches 0.5
        at plateau (equal positive/negative signs), near 1.0 during active
        convergence (drift predominantly decreasing).
        Populated by ``TopologyPreservingSimulator``.  Shape: ``(n_iters,)``.
    velocity : np.ndarray | None
        Per-iteration max velocity.
        Populated by ``CirclePhysicsSimulator``.  Shape: ``(n_iters,)``.
    overlaps : np.ndarray | None
        Per-iteration overlap count.
        Populated by both simulators.  Shape: ``(n_iters,)``.

    """

    positions: list[NDArray[np.floating]] | None = None
    drift: NDArray[np.floating] | None = None
    jitter: NDArray[np.floating] | None = None
    drift_rate: NDArray[np.floating] | None = None
    drift_rate_neg_frac: NDArray[np.floating] | None = None
    velocity: NDArray[np.floating] | None = None
    overlaps: NDArray[np.intp] | None = None

    def __len__(self) -> int:
        """Number of recorded iterations."""
        for arr in (self.drift, self.jitter, self.drift_rate, self.velocity, self.overlaps):
            if arr is not None:
                return len(arr)
        if self.positions is not None:
            return len(self.positions)
        return 0


@dataclass
class SymbolCartogram:
    """Rendered symbol cartogram ready for visualization and export.

    Supports the layout-styling separation pattern with layout_result and
    styling fields for the new API.

    Attributes
    ----------
    symbols : gpd.GeoDataFrame
        GeoDataFrame containing symbol geometries with columns:
        - geometry: Symbol polygon (circle/square/hexagon)
        - _symbol_x, _symbol_y: Symbol center position
        - _symbol_size: Symbol size (radius for circles, half-side for squares)
        - _displacement: Distance from original centroid to symbol center
        - original_index: Index in original GeoDataFrame

    status : SymbolCartogramStatus
        Computation status (CONVERGED, COMPLETED, ORIGINAL)

    metrics : dict
        Quality metrics:
        - displacement_mean: Mean distance from original centroid
        - displacement_max: Maximum displacement
        - displacement_std: Standard deviation of displacement
        - topology_preservation: Fraction of adjacencies preserved (if computed)
        - iterations: Number of iterations used (free placement)
        - n_skipped: Number of geometries skipped due to null values

    simulation_history : SimulationHistory | None
        Per-iteration diagnostics and optional position snapshots.
        ``None`` when no simulation was run (e.g. grid placement).

    layout_result : LayoutResult | None
        Immutable layout result (new API). Contains canonical symbol, transforms,
        and preprocessing data (positions, sizes, adjacency, bounds, crs).

    styling : Styling | None
        Styling configuration used (new API).

    Private Attributes
    ------------------
    _source_gdf : gpd.GeoDataFrame | None
        Reference to original input GeoDataFrame (for attribute merging).
        Only set when created via create_symbol_cartogram() with the original gdf.
    _valid_mask : np.ndarray | None
        Boolean mask indicating which rows had valid (non-null) values.
    _tiling_result : Any | None
        Tiling result for grid layouts (algorithm-specific).
    _assignments : np.ndarray | None
        Grid assignments for grid layouts.

    """

    # Core results
    symbols: gpd.GeoDataFrame
    status: SymbolCartogramStatus = field(default_factory=lambda: SymbolCartogramStatus.COMPLETED)
    metrics: dict[str, Any] = field(default_factory=dict)

    # Optional simulation history
    simulation_history: SimulationHistory | None = None

    # New API fields (layout-styling separation)
    layout_result: LayoutResult | None = field(default=None, repr=False)
    styling: Styling | None = field(default=None, repr=False)

    # Source reference (not shown in repr) - for attribute merging
    _source_gdf: Any | None = field(default=None, repr=False)
    _valid_mask: NDArray[np.bool_] | None = field(default=None, repr=False)
    _tiling_result: Any | None = field(default=None, repr=False)
    _assignments: NDArray[np.intp] | None = field(default=None, repr=False)

    def restyle(
        self,
        styling: Styling | None = None,
        **kwargs,
    ) -> SymbolCartogram:
        """Create a NEW cartogram with different styling.

        Requires that the cartogram was created using the new API
        (with layout_result stored).

        Parameters
        ----------
        styling : Styling or None
            Pre-configured Styling object. If None, creates from kwargs.
        **kwargs
            Convenience kwargs for simple cases (symbol, scale, etc.)
            Creates a temporary Styling object internally.

        Returns
        -------
        SymbolCartogram
            New cartogram with updated styling.

        Raises
        ------
        ValueError
            If layout_result is not available (legacy cartogram).

        Examples
        --------
        >>> # Restyle with different symbol
        >>> new_cartogram = cartogram.restyle(symbol="hexagon")

        >>> # Restyle with Styling object
        >>> styling = Styling().set_symbol("hexagon").transform(scale=0.9)
        >>> new_cartogram = cartogram.restyle(styling)

        """
        if self.layout_result is None:
            raise ValueError(
                "Cannot restyle: layout_result not available. "
                "This cartogram was created using the legacy API. "
                "Use create_layout() + style() for the new API.",
            )

        from .styling import Styling

        if styling is None:
            styling = Styling(**kwargs)
        new_cartogram = styling.apply(self.layout_result)
        # Preserve source_gdf reference for attribute merging
        new_cartogram._source_gdf = self._source_gdf
        new_cartogram._valid_mask = self._valid_mask
        return new_cartogram

    def plot(
        self,
        column: str | None = None,
        cmap: str = "viridis",
        norm: Any | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        legend: bool = True,
        legend_kwds: dict[str, Any] | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple[float, float] = (10, 8),
        source_gdf: Any | None = None,
        facecolor: Any = None,
        color_map: dict[str, Any] | None = None,
        alpha: Any = 0.9,
        alpha_range: tuple[float, float] = (0.2, 1.0),
        edgecolor: Any = "none",
        edge_cmap: str | None = None,
        linewidth: Any = 0.5,
        linewidth_range: tuple[float, float] = (0.5, 3.0),
        hatch: Any = None,
        hatch_map: dict[str, str] | None = None,
        # Legend
        edge_legend: bool = True,
        edge_legend_kwds: dict[str, Any] | None = None,
        hatch_legend: bool = True,
        hatch_legend_kwds: dict[str, Any] | None = None,
        # Labels
        label: Any = None,
        label_color: Any = "black",
        label_fontsize: Any = 8,
        label_fontsize_range: tuple[float, float] = (6.0, 14.0),
        label_kwargs: dict[str, Any] | None = None,
        title: str | None = None,
        zorder: int = 1,
    ) -> SymbolsPlotResult:
        """Plot the symbol cartogram with per-symbol visual styling.

        Every visual property can be set **globally** (scalar / colour string),
        **data-driven** (column name → automatic mapping), or
        **per-symbol** (list / NumPy array, one value per symbol).

        Parameters
        ----------
        column : str, optional
            Convenience shorthand for ``facecolor=column``.  Kept for
            backward compatibility.  When both *column* and *facecolor* are
            provided, *facecolor* takes precedence.
        cmap : str
            Colormap name for numeric data columns.  Default ``"viridis"``.
        norm : matplotlib Normalize, optional
            Custom normalisation for the colour mapping.
        vmin, vmax : float, optional
            Explicit data limits for the colourmap normalisation.
        legend : bool
            Display a colorbar (numeric) or patch legend (categorical) when
            *facecolor* is data-driven.  Default ``True``.
        legend_kwds : dict, optional
            Extra keyword arguments forwarded to ``Figure.colorbar()``
            (numeric columns) or ``Axes.legend()`` (categorical columns).
            Use ``"title"`` to label the legend.
        ax : plt.Axes, optional
            Axes to draw on.  A new figure is created when ``None``.
        figsize : tuple
            Figure size when a new figure is created.  Default ``(10, 8)``.
        source_gdf : gpd.GeoDataFrame, optional
            External GeoDataFrame for column lookups (highest priority).
            Useful when the column you want to map is not stored on the
            cartogram's own ``symbols`` table.
        facecolor : color, column name, or array-like, optional
            Symbol fill colour.  Accepts:

            * A matplotlib colour string (``"steelblue"``).
            * A **column name** → numeric: mapped via *cmap* / *norm*;
              categorical: auto-assigned from the ``"tab10"`` palette
              (override with *color_map*).
            * A 1-D numeric array → mapped via *cmap* / *norm*.
            * An ``(n, 3)`` or ``(n, 4)`` float array (RGB / RGBA).
            * A list of colour strings or RGBA tuples.

            Defaults to ``"steelblue"`` when ``None``.
        color_map : dict, optional
            Per-category colour overrides when *facecolor* maps a categorical
            column.  Unspecified categories receive auto-assigned colours.
            Example: ``{"Europe": "#2ca02c", "Africa": "#d62728"}``.
        alpha : float, column name, or array-like, optional
            Symbol opacity (0 = transparent, 1 = opaque).  A column name is
            linearly interpolated into *alpha_range*.  Default ``0.9``.
        alpha_range : (float, float)
            ``(min, max)`` alpha range for column-driven transparency.
            Default ``(0.2, 1.0)``.
        edgecolor : color, column name, or array-like, optional
            Symbol edge colour.  Same forms as *facecolor*.
            Default ``"none"`` (no visible border).
        edge_cmap : str, optional
            Colormap for edge colour column mapping.  Falls back to *cmap*.
        linewidth : float, column name, or array-like, optional
            Edge line width.  A column name is linearly interpolated into
            *linewidth_range*.  Default ``0.5``.
        linewidth_range : (float, float)
            ``(min, max)`` linewidth range for column-driven widths.
            Default ``(0.5, 3.0)``.
        hatch : str, column name, or sequence, optional
            Fill hatching.  A matplotlib hatch pattern string (``"///"``) is
            applied globally; a column name maps each category to a pattern
            from *hatch_map* or the default cycle; a list / array applies
            per-symbol patterns.

            .. note::
                Hatching is only visible when *edgecolor* is not ``"none"``.
        hatch_map : dict, optional
            Per-category hatch overrides when *hatch* is a column name.
            Example: ``{"urban": "///", "rural": "..."}``.
        edge_legend : bool
            Show a separate legend for *edgecolor* when it is data-driven from
            a different column than *facecolor*.  Default ``True``.
        edge_legend_kwds : dict, optional
            Same as *legend_kwds* but for the edge-colour legend.
        hatch_legend : bool
            Show a patch legend for the hatch ↔ category mapping when *hatch*
            is a column name.  No effect for global patterns or lists.
            Default ``True``.
        hatch_legend_kwds : dict, optional
            ``Axes.legend()`` kwargs for the hatch legend, plus an optional
            nested ``"patch_kw"`` dict controlling legend-patch appearance.
        label : str, column name, or sequence, optional
            Per-symbol text labels.  A column name uses the string
            representation of each value; a list / array provides explicit
            strings.
        label_color : color, column name, or array-like, optional
            Text colour for labels.  Accepts the same forms as *facecolor*.
            Default ``"black"``.
        label_fontsize : float, column name, or array-like, optional
            Label font size.  A column name is linearly interpolated into
            *label_fontsize_range*.  Default ``8``.
        label_fontsize_range : (float, float)
            ``(min, max)`` font-size range for column-driven sizing.
            Default ``(6.0, 14.0)``.
        label_kwargs : dict, optional
            Extra keyword arguments forwarded to ``Axes.text()`` for every
            label (e.g. ``{"fontweight": "bold"}``).
        title : str, optional
            Axes title.
        zorder : int
            Matplotlib drawing order.  Default ``1``.

        Returns
        -------
        SymbolsPlotResult
            Result with the axes and captured artists (collections, labels,
            colorbars, legends).  Access the axes via ``result.ax``.

        Examples
        --------
        >>> # Backward-compatible usage
        >>> result.plot(column="pop_est", cmap="YlOrRd")

        >>> # Global style
        >>> result.plot(facecolor="steelblue", edgecolor="white", linewidth=0.8)

        >>> # Data-driven fill → automatic colorbar
        >>> result.plot(facecolor="gdp_per_capita", cmap="plasma")

        >>> # Categorical fill with qualitative palette
        >>> result.plot(facecolor="continent")

        >>> # Per-symbol alpha driven by a data column
        >>> result.plot(facecolor="#4C72B0", alpha="pop_est", alpha_range=(0.3, 1.0))

        >>> # Hatching by category (needs visible edge colour)
        >>> result.plot(facecolor="none", edgecolor="black", linewidth=0.5,
        ...             hatch="region",
        ...             hatch_map={"A": "///", "B": "...", "C": "xxx"})

        >>> # Explicit per-symbol colour array
        >>> import numpy as np
        >>> colors = np.random.rand(len(result.symbols), 4)
        >>> result.plot(facecolor=colors, legend=False)

        >>> # Data-driven edge colour → second legend auto-added
        >>> result.plot(facecolor="steelblue", edgecolor="region")

        >>> # Labels from a column
        >>> result.plot(facecolor="pop_est", label="name")

        >>> # Labels with column-driven colour and size
        >>> result.plot(facecolor="steelblue",
        ...             label="iso_a3",
        ...             label_color="region",
        ...             label_fontsize="pop_est",
        ...             label_fontsize_range=(6, 12))

        """
        from .visualization import plot_symbols

        # Backward-compatible column kwarg: maps to facecolor when not already set
        effective_facecolor = facecolor if facecolor is not None else column

        return plot_symbols(
            self,
            ax=ax,
            figsize=figsize,
            source_gdf=source_gdf,
            facecolor=effective_facecolor,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            color_map=color_map,
            alpha=alpha,
            alpha_range=alpha_range,
            edgecolor=edgecolor,
            edge_cmap=edge_cmap,
            linewidth=linewidth,
            linewidth_range=linewidth_range,
            hatch=hatch,
            hatch_map=hatch_map,
            legend=legend,
            legend_kwds=legend_kwds,
            edge_legend=edge_legend,
            edge_legend_kwds=edge_legend_kwds,
            hatch_legend=hatch_legend,
            hatch_legend_kwds=hatch_legend_kwds,
            label=label,
            label_color=label_color,
            label_fontsize=label_fontsize,
            label_fontsize_range=label_fontsize_range,
            label_kwargs=label_kwargs,
            zorder=zorder,
            title=title,
        )

    def to_geodataframe(
        self,
        source_gdf: gpd.GeoDataFrame | None = None,
    ) -> gpd.GeoDataFrame:
        """Export symbols as GeoDataFrame with original attributes.

        Parameters
        ----------
        source_gdf : gpd.GeoDataFrame, optional
            Original GeoDataFrame to merge attributes from. If None, uses
            the stored reference from creation time (if available).

        Returns
        -------
        gpd.GeoDataFrame
            Symbol geometries with original data columns.

        Notes
        -----
        If no source_gdf is available (neither passed nor stored), returns
        just the symbols GeoDataFrame without original attributes.

        """
        # Use provided source_gdf or fall back to stored reference
        gdf = source_gdf if source_gdf is not None else self._source_gdf

        if gdf is None:
            # No source reference, return symbols as-is
            return self.symbols.copy()

        # Get valid rows from source
        source_subset = gdf.loc[self._valid_mask].copy() if self._valid_mask is not None else gdf.copy()

        # Replace geometry with symbols
        source_subset = source_subset.set_geometry(self.symbols.geometry.values)

        # Add symbol-specific columns
        for col in ["_symbol_x", "_symbol_y", "_symbol_size", "_displacement"]:
            if col in self.symbols.columns:
                source_subset[col] = self.symbols[col].values

        return source_subset

    def save(self, path: str) -> None:
        """Save the symbol cartogram to a JSON file.

        Saves the symbol geometries, layout result, source GeoDataFrame, and
        metrics so the cartogram can be fully restored with SymbolCartogram.load().

        Parameters
        ----------
        path : str
            Output file path (typically .json extension).

        Examples
        --------
        >>> result.save('cartogram.json')
        >>> loaded = SymbolCartogram.load('cartogram.json')
        >>> loaded.plot(facecolor='population')

        """
        import json
        from pathlib import Path

        path = Path(path)

        data: dict = {
            "status": self.status.value,
            "metrics": {k: (v.item() if hasattr(v, "item") else v) for k, v in self.metrics.items()},
        }

        # Symbols GeoDataFrame — stored as GeoJSON FeatureCollection
        symbols_fc = json.loads(self.symbols.to_json())
        data["symbols"] = symbols_fc
        if self.symbols.crs is not None:
            data["symbols_crs"] = self.symbols.crs.to_string()

        # Layout result
        if self.layout_result is not None:
            data["layout_result"] = self.layout_result.serialize()

        # Source GeoDataFrame
        if self._source_gdf is not None:
            src = self._source_gdf
            if src.crs is not None:
                data["source_crs"] = src.crs.to_string()
            data["source_index"] = list(src.index)
            non_geom_cols = [c for c in src.columns if c != src.geometry.name]
            data["source_columns"] = non_geom_cols
            records = src[non_geom_cols].to_dict(orient="records")
            data["source_records"] = [
                {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()} for row in records
            ]
            data["source_geometries"] = [geom.__geo_interface__ for geom in src.geometry]

        # Valid mask
        if self._valid_mask is not None:
            data["valid_mask"] = self._valid_mask.tolist()

        # Grid layout data: tiling polygons/transforms and symbol assignments.
        # adjacency is NOT saved — it's only used during assignment (already done)
        # and is not needed by plot_tiling().
        if self._tiling_result is not None:
            tr = self._tiling_result
            data["tiling_result"] = {
                "polygons": [p.__geo_interface__ for p in tr.polygons],
                "transforms": [
                    {"center": list(t.center), "rotation": t.rotation, "flipped": t.flipped} for t in tr.transforms
                ],
                "tile_size": tr.tile_size,
                "inscribed_radius": tr.inscribed_radius,
                "canonical_tile": tr.canonical_tile.__geo_interface__,
                "n_base_vertices": tr.n_base_vertices,
            }
        if self._assignments is not None:
            data["assignments"] = self._assignments.tolist()

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> SymbolCartogram:
        """Load a symbol cartogram from a JSON file.

        Restores a SymbolCartogram saved by save(), including symbol geometries,
        layout result, source GeoDataFrame, and metrics.

        Parameters
        ----------
        path : str
            Path to saved JSON file.

        Returns
        -------
        SymbolCartogram
            Restored cartogram. Supports plot(), to_geodataframe(), and restyle()
            (if layout_result was present at save time).

        Examples
        --------
        >>> loaded = SymbolCartogram.load('cartogram.json')
        >>> loaded.plot(facecolor='population')
        >>> restyled = loaded.restyle(symbol='hexagon')

        """
        import json
        from pathlib import Path

        import geopandas as gpd
        import numpy as np
        from shapely.geometry import shape

        from .layout_result import LayoutResult
        from .status import SymbolCartogramStatus

        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        # Reconstruct symbols GeoDataFrame
        symbols = gpd.GeoDataFrame.from_features(data["symbols"]["features"])
        if "symbols_crs" in data:
            symbols = symbols.set_crs(data["symbols_crs"])

        # Reconstruct layout_result
        layout_result = None
        if "layout_result" in data:
            layout_result = LayoutResult.from_serialized(data["layout_result"])

        # Reconstruct source GeoDataFrame
        source_gdf = None
        if "source_geometries" in data:
            src_geoms = [shape(g) for g in data["source_geometries"]]
            records = data.get("source_records", [{} for _ in src_geoms])
            index = data.get("source_index", list(range(len(src_geoms))))
            source_gdf = gpd.GeoDataFrame(records, geometry=src_geoms, index=index)
            if "source_crs" in data:
                source_gdf = source_gdf.set_crs(data["source_crs"])

        # Reconstruct valid_mask
        valid_mask = None
        if "valid_mask" in data:
            valid_mask = np.array(data["valid_mask"])

        # Reconstruct tiling data (grid layouts only)
        tiling_result = None
        if "tiling_result" in data:
            from shapely.geometry import shape as shapely_shape

            from .tiling import TileTransform, TilingResult

            tr_data = data["tiling_result"]
            n = len(tr_data["polygons"])
            tiling_result = TilingResult(
                polygons=[shapely_shape(p) for p in tr_data["polygons"]],
                transforms=[
                    TileTransform(
                        center=tuple(t["center"]),
                        rotation=t["rotation"],
                        flipped=t["flipped"],
                    )
                    for t in tr_data["transforms"]
                ],
                adjacency=np.zeros((n, n), dtype=bool),  # not needed post-assignment
                vertex_adjacency=None,
                tile_size=tr_data["tile_size"],
                inscribed_radius=tr_data["inscribed_radius"],
                canonical_tile=shapely_shape(tr_data["canonical_tile"]),
                n_base_vertices=tr_data.get("n_base_vertices"),
            )

        assignments = None
        if "assignments" in data:
            assignments = np.array(data["assignments"], dtype=np.intp)

        try:
            status = SymbolCartogramStatus(data.get("status", "completed"))
        except ValueError:
            status = SymbolCartogramStatus.COMPLETED

        return cls(
            symbols=symbols,
            status=status,
            metrics=data.get("metrics", {}),
            layout_result=layout_result,
            _source_gdf=source_gdf,
            _valid_mask=valid_mask,
            _tiling_result=tiling_result,
            _assignments=assignments,
        )

    def get_displacement_vectors(self) -> NDArray[np.floating]:
        """Get displacement vectors from original centroid to symbol center.

        Returns
        -------
        np.ndarray of shape (n, 2)
            Displacement vectors [dx, dy] for each symbol.

        Raises
        ------
        ValueError
            If no original positions are available (via layout_result or _source_gdf).

        """
        # Prefer layout_result.positions (new API)
        if self.layout_result is not None:
            original = self.layout_result.positions
        elif self._source_gdf is not None:
            # Fall back to computing from source geometries
            if self._valid_mask is not None:
                source_geoms = self._source_gdf.loc[self._valid_mask].geometry
            else:
                source_geoms = self._source_gdf.geometry
            original = np.array([[g.centroid.x, g.centroid.y] for g in source_geoms])
        else:
            raise ValueError(
                "No original positions available. "
                "This cartogram was created without storing layout_result or source_gdf.",
            )

        final = np.column_stack([self.symbols["_symbol_x"], self.symbols["_symbol_y"]])
        return final - original
