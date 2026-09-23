"""Result container for Voronoi cartogram operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import geopandas as gpd

    from .options import VoronoiOptions

from carto_flow._history import History

__all__ = [
    "TopologyAnalysis",
    "TopologyRepairReport",
    "VoronoiCartogram",
    "VoronoiPlotResult",
]


@dataclass
class TopologyAnalysis:
    """Topology state of a Voronoi cartogram across all three repair stages.

    All identifier fields use the GDF index labels of the source GeoDataFrame
    (not integer slot positions), making them easy to cross-reference with the
    original data.

    Attributes
    ----------
    discontiguous_groups : list of (group_id, [gdf_label, ...])
        One entry per satellite component: the group identifier and the GDF
        index labels of the cells in that satellite.  Empty when
        ``group_by`` was not provided or all groups are fully contiguous.
    violated_adjacency : list of (label_i, label_j)
        Input-adjacent district pairs that are **not** Voronoi-adjacent in
        the current tessellation.  Empty when adjacency information was not
        available.
    n_adjacency_pairs : int
        Total number of input-adjacent pairs checked (denominator for
        :attr:`adjacency_fraction`).  ``0`` when adjacency was not checked.
    misaligned_orientation : list of (label_i, label_j, cosine)
        Voronoi-adjacent pairs whose relative spatial direction is inverted
        (cosine similarity ≤ 0) compared to the input geometry directions.
        Empty when adjacency information was not available.
    mean_orientation_cosine : float or None
        Mean cosine similarity across **all** Voronoi-adjacent pairs that
        appear in the input adjacency list.  ``None`` when adjacency was not
        available or no Voronoi-adjacent pairs exist.
    degenerate_cells : list of gdf_label
        GDF index labels of cells that collapsed to a Point or a zero-area
        geometry (see :attr:`VoronoiCartogram.degenerate_cells`).
    """

    discontiguous_groups: list[tuple[Any, list[Any]]]
    violated_adjacency: list[tuple[Any, Any]]
    n_adjacency_pairs: int
    misaligned_orientation: list[tuple[Any, Any, float]]
    mean_orientation_cosine: float | None
    degenerate_cells: list[Any] = field(default_factory=list)

    @property
    def n_discontiguous_groups(self) -> int:
        """Number of satellite components across all groups."""
        return len(self.discontiguous_groups)

    @property
    def n_violated_adjacency(self) -> int:
        """Number of input-adjacent pairs that are not Voronoi-adjacent."""
        return len(self.violated_adjacency)

    @property
    def adjacency_fraction(self) -> float | None:
        """Fraction of input-adjacent pairs that ARE Voronoi-adjacent.

        ``None`` when no adjacency pairs were checked.
        """
        if self.n_adjacency_pairs == 0:
            return None
        return (self.n_adjacency_pairs - len(self.violated_adjacency)) / self.n_adjacency_pairs

    @property
    def n_misaligned_orientation(self) -> int:
        """Number of Voronoi-adjacent pairs with cosine similarity ≤ 0."""
        return len(self.misaligned_orientation)

    @property
    def any_issues(self) -> bool:
        """``True`` if any topology issue exists across all three stages."""
        return bool(self.discontiguous_groups or self.violated_adjacency or self.misaligned_orientation)

    def __repr__(self) -> str:
        lines = ["TopologyAnalysis:"]
        if self.discontiguous_groups:
            lines.append(f"  group contiguity : {self.n_discontiguous_groups} satellite component(s)")
        else:
            lines.append("  group contiguity : OK")
        if self.n_adjacency_pairs > 0:
            sat = self.n_adjacency_pairs - self.n_violated_adjacency
            pct = 100.0 * sat / self.n_adjacency_pairs
            lines.append(
                f"  adjacency        : {self.n_violated_adjacency} / {self.n_adjacency_pairs}"
                f" pairs violated  ({pct:.1f}% satisfied)"
            )
        else:
            lines.append("  adjacency        : (not checked)")
        if self.mean_orientation_cosine is not None:
            lines.append(
                f"  orientation      : {self.n_misaligned_orientation} misaligned pair(s)"
                f"  mean cosine={self.mean_orientation_cosine:.3f}"
            )
        else:
            lines.append("  orientation      : (not checked)")
        if self.degenerate_cells:
            lines.append(
                f"  degenerate cells : {len(self.degenerate_cells)} cell(s) collapsed to a"
                f" point/zero area  (e.g. {self.degenerate_cells[0]!r})"
            )
        return "\n".join(lines)

    def plot(
        self,
        cartogram: VoronoiCartogram,
        *,
        show_base: bool | None = None,
        show_contiguity: bool = True,
        show_adjacency: bool = True,
        show_orientation: bool = True,
        ax: Any | None = None,
        figsize: tuple[float, float] = (10, 8),
        title: str = "Topology Analysis",
    ) -> VoronoiPlotResult:
        """Plot topology issues overlaid on the Voronoi cells.

        Parameters
        ----------
        cartogram : VoronoiCartogram
            The cartogram whose cells provide the spatial context.
        show_base : bool or None
            Draw all cells as a gray background layer.  ``None`` (default)
            draws the base only when *ax* is ``None`` (new figure); when an
            existing *ax* is supplied the base is skipped so prior content is
            not hidden.  Pass ``True`` or ``False`` to override.
        show_contiguity : bool
            Highlight satellite cells (stage 1 issues).  Default ``True``.
        show_adjacency : bool
            Draw lines between centroids of violated adjacency pairs (stage 2
            issues).  Default ``True``.
        show_orientation : bool
            Draw arrows for misaligned Voronoi-adjacent pairs (stage 3 issues).
            Default ``True``.
        ax : matplotlib Axes or None
            Axes to draw on.  A new figure is created if ``None``.
        figsize : tuple of float
            Figure size when ``ax`` is ``None``.
        title : str
            Axis title.

        Returns
        -------
        VoronoiPlotResult
            Dataclass with ``ax`` and ``collections``.
        """
        from .visualization import plot_topology

        return plot_topology(
            self,
            cartogram,
            show_base=show_base,
            show_contiguity=show_contiguity,
            show_adjacency=show_adjacency,
            show_orientation=show_orientation,
            ax=ax,
            figsize=figsize,
            title=title,
        )


def _plot_topology_issues(
    analysis: TopologyAnalysis,
    cartogram: VoronoiCartogram,
    ax: Any,
    *,
    show_base: bool = True,
    show_contiguity: bool = True,
    show_adjacency: bool = True,
    show_orientation: bool = True,
) -> None:
    """Draw topology-issue layers onto *ax* (no axis formatting)."""
    from .visualization import _plot_topology_issues as _viz_plot

    _viz_plot(
        analysis,
        cartogram,
        ax,
        show_base=show_base,
        show_contiguity=show_contiguity,
        show_adjacency=show_adjacency,
        show_orientation=show_orientation,
    )


@dataclass
class TopologyRepairReport:
    """Result of a post-hoc topology repair on a :class:`VoronoiCartogram`.

    Attributes
    ----------
    cartogram : VoronoiCartogram
        New cartogram with cells (and positions) permuted to address topology
        issues.  The original cartogram is unchanged.
    before : TopologyAnalysis
        Topology state *before* repair.
    after : TopologyAnalysis
        Topology state *after* repair.
    stages_run : list of str
        Repair stages that were executed (subset of
        ``["group_contiguity", "adjacency", "orientation"]``).
    reassigned : list
        GDF index labels of districts whose cell assignment changed.
    """

    cartogram: VoronoiCartogram
    before: TopologyAnalysis
    after: TopologyAnalysis
    stages_run: list[str]
    reassigned: list[Any]
    _original_cartogram: VoronoiCartogram | None = field(default=None, repr=False)

    @property
    def fully_repaired(self) -> bool:
        """``True`` when no topology issues remain after repair."""
        return not self.after.any_issues

    @property
    def improved(self) -> bool:
        """``True`` when at least one issue count decreased."""
        return (
            self.after.n_discontiguous_groups < self.before.n_discontiguous_groups
            or self.after.n_violated_adjacency < self.before.n_violated_adjacency
            or self.after.n_misaligned_orientation < self.before.n_misaligned_orientation
        )

    def __repr__(self) -> str:
        stages = ", ".join(self.stages_run) if self.stages_run else "none"
        lines = [f"TopologyRepairReport (stages: {stages}):"]

        def _delta(before_val: int, after_val: int, label: str, unit: str) -> str:
            if before_val == 0 and after_val == 0:
                return f"  {label:<17}: OK"
            arrow = "→"
            return f"  {label:<17}: {before_val} {arrow} {after_val} {unit}"

        lines.append(
            _delta(
                self.before.n_discontiguous_groups,
                self.after.n_discontiguous_groups,
                "group contiguity",
                "satellite(s)",
            )
        )
        if self.before.n_adjacency_pairs > 0 or self.after.n_adjacency_pairs > 0:
            lines.append(
                _delta(
                    self.before.n_violated_adjacency,
                    self.after.n_violated_adjacency,
                    "adjacency",
                    "violated pair(s)",
                )
            )
        else:
            lines.append("  adjacency        : (not checked)")
        if self.before.mean_orientation_cosine is not None or self.after.mean_orientation_cosine is not None:
            lines.append(
                _delta(
                    self.before.n_misaligned_orientation,
                    self.after.n_misaligned_orientation,
                    "orientation",
                    "misaligned pair(s)",
                )
            )
        else:
            lines.append("  orientation      : (not checked)")
        lines.append(f"  reassigned       : {len(self.reassigned)} district(s)")
        return "\n".join(lines)

    def plot(
        self,
        *,
        show_base: bool | None = None,
        show_issues: bool = True,
        show_reassigned: bool = True,
        axes: Any | None = None,
        figsize: tuple[float, float] = (16, 8),
        title: str | None = None,
    ) -> VoronoiPlotResult:
        """Plot a before/after comparison of the topology repair.

        Parameters
        ----------
        show_base : bool or None
            Draw all cells as a gray background layer in each panel.  ``None``
            (default) draws the base only when *axes* is ``None`` (new figure);
            when existing axes are supplied the base is skipped so prior content
            is not hidden.  Pass ``True`` or ``False`` to override.
        show_issues : bool
            Overlay topology-issue highlights (satellite cells, violated
            adjacency lines, misaligned orientation arrows) on both panels.
            Default ``True``.
        show_reassigned : bool
            Highlight reassigned cells in the *after* panel with a thick green
            border.  Default ``True``.
        axes : sequence of two Axes, or None
            Existing axes to draw on.  A new ``(1, 2)`` figure is created when
            ``None``.
        figsize : tuple of float
            Figure size when ``axes`` is ``None``.  Default ``(16, 8)``.
        title : str or None
            Figure suptitle.  Defaults to
            ``"Topology Repair — {stages_run}"``.

        Returns
        -------
        VoronoiPlotResult
            ``ax`` is a tuple ``(ax_before, ax_after)``.

        Notes
        -----
        When :attr:`_original_cartogram` is ``None`` (e.g., the report was
        reconstructed from disk), only the *after* panel is rendered and a
        ``UserWarning`` is emitted.
        """
        from .visualization import plot_topology_repair

        return plot_topology_repair(
            self,
            show_base=show_base,
            show_issues=show_issues,
            show_reassigned=show_reassigned,
            axes=axes,
            figsize=figsize,
            title=title,
        )


@dataclass
class VoronoiPlotResult:
    """Artists produced by :meth:`VoronoiCartogram.plot`."""

    ax: Any
    collections: list


@dataclass
class VoronoiCartogram:
    """Result of a Voronoi cartogram (Lloyd relaxation) computation.

    Attributes
    ----------
    positions : np.ndarray, shape (G, 2)
        Final centroid positions after relaxation.
    cells : np.ndarray of shapely.Geometry, shape (G,)
        Voronoi cell polygon for each centroid, clipped to the outer boundary.
    metrics : dict
        Scalar summary of the run:

        - ``"n_iterations"``: number of iterations completed
        - ``"converged"``: ``True`` if the displacement tolerance was reached
          before ``n_iter`` was exhausted
        - ``"initial_area_cv"``: coefficient of variation of the **input**
          geometry areas before any relaxation (baseline for comparison)
        - ``"final_area_cv"``: coefficient of variation of cell areas at the
          last iteration (0 = perfect equal-area distribution)
        - ``"mean_area_error_pct"``: mean absolute area error (%) across all
          cells, where error = (actual_area / target_area - 1) x 100.  A
          degenerate cell (see :attr:`degenerate_cells`) has zero area and so
          contributes an error of -100%.
        - ``"max_area_error_pct"``: maximum absolute area error (%) across all
          cells.  A degenerate cell counts as -100%.

    options : VoronoiOptions
        Options used for this run.
    convergence_history : list of float
        Area CV recorded after each completed iteration, in order.
        Length equals ``metrics["n_iterations"]``.
    history : History or None
        Per-iteration snapshots (:class:`~carto_flow.voronoi_cartogram.history.VoronoiSnapshot`)
        captured when ``options.record_history=True``.  ``None`` otherwise.
    """

    positions: np.ndarray
    cells: np.ndarray
    metrics: dict[str, Any]
    options: VoronoiOptions
    _source_gdf: gpd.GeoDataFrame | None = field(default=None, repr=False)
    convergence_history: list[float] = field(default_factory=list, repr=False)
    history: History | None = field(default=None, repr=False)
    _field: Any = field(default=None, repr=False)
    area_errors: np.ndarray | None = field(default=None, repr=False)
    _weighted: bool = field(default=False, repr=False)

    @property
    def degenerate_cells(self) -> list[Any]:
        """Index labels of cells that collapsed to a point or (near-)zero area.

        A cell degenerates for one of two reasons:

        * **starved generator** -- the generator lost all of its raster pixels,
          typically because the grid resolution is too coarse for the smallest
          target areas.  Raising the backend resolution is the usual remedy.
        * **extraction failure** -- the generator owns pixels, but converting
          them to a polygon failed (e.g. the clip against a boundary carrying
          degenerate rings collapsed the cell).  This emits its own
          ``RuntimeWarning`` naming the seed; raising the resolution does not
          help.

        Either way the cell has no area, so it contributes an area error of
        -100%, is omitted from plots, and is exported by
        :meth:`to_geodataframe` as a ``Point``.

        A polygon counts as degenerate when its area is below ``1e-9`` of the
        mean cell area, which catches slivers that are not exactly zero.

        Returns
        -------
        list
            GDF index labels of the degenerate cells (integer positions when no
            source GeoDataFrame is available).  Empty when all cells are proper
            polygons.
        """
        areas = np.array([c.area for c in self.cells], dtype=float)
        mean_area = float(areas.mean()) if len(areas) else 0.0
        eps = 1e-9 * mean_area
        bad = [i for i, c in enumerate(self.cells) if c.geom_type not in ("Polygon", "MultiPolygon") or c.area <= eps]
        if self._source_gdf is None:
            return bad
        return [self._source_gdf.index[i] for i in bad]

    @property
    def boundary_debug(self) -> dict[str, np.ndarray] | None:
        """Diagnostic arrays from the last boundary-deformation step.

        Returns ``None`` when ``boundary_elasticity == 0`` or when no
        deformation step has run yet.

        Keys
        ----
        area_pressure : (G,) float
            Per-centroid fractional density excess at the last iteration:
            ``density_i / target_density - 1``.  Positive = cell is smaller
            than target (high pressure, boundary expands); negative = cell is
            larger than target (low pressure, boundary contracts).
        rho : (ny, nx) float
            Density field fed to the FFT velocity solver (``area_pressure + 1``
            for active pixels, ``1.0`` for inactive pixels).
        vx : (ny, nx) float
            X-component of the normalized velocity field from the FFT solver.
        vy : (ny, nx) float
            Y-component of the normalized velocity field from the FFT solver.
        vertex_xy : (V, 2) float
            Current (displaced) boundary vertex positions.
        vertex_disp : (V,) float
            Scalar displacement of each vertex from its original position.
        vertex_disp_vec : (V, 2) float
            2-D displacement vector for each vertex
            (``current_position - original_position``).  Pass the x and y
            components directly to ``ax.quiver``.
        """
        if self._field is None:
            return None
        f = self._field
        if not hasattr(f, "_debug_area_pressure"):
            return None
        return {
            "area_pressure": f._debug_area_pressure,
            "rho": f._debug_rho,
            "vx": f._debug_vx,
            "vy": f._debug_vy,
            "vertex_xy": f._debug_vertex_xy,
            "vertex_disp": f._debug_vertex_disp,
            "vertex_disp_vec": f._debug_vertex_disp_vec,
        }

    @property
    def geodesic_debug(self) -> dict | None:
        """Diagnostic dict from the last geodesic-labeling call.

        Returns ``None`` when ``debug_geodesic=False`` (the default) or when
        the field has not been labeled yet.

        Requires ``RasterBackend(labeling="geodesic", debug_geodesic=True)``.

        Keys
        ----
        seed_rows, seed_cols : (n_seeds,) int32
            Grid row/column of every placed BFS seed.
        seed_components : (n_seeds,) int32
            scipy component id (1-based) of each seed.
        centroid_components : (G,) int32
            Component id of the nearest active pixel to each original centroid.
        misplaced_mask : (G,) bool
            True for centroids whose seed is in a different component than
            their nearest active pixel.
        n_misplaced : int
            Number of misplaced seeds.
        comp_2d : (ny, nx) int32
            Full component label grid (0 = inactive, 1..n = components).
        unseeded_comp_ids : (k,) int32
            Component IDs that had no seed before the fix; non-empty means
            the fix fired this call.
        n_unseeded : int
            Number of components that required an extra seed.
        unseeded_pixel_mask : (ny, nx) bool
            Active pixels that belonged to unseeded components.
        seed_centroid_distances : (G,) float or None
            Euclidean distance from each centroid to its seed's world
            position.  ``None`` if grid coordinate arrays were not available.
        """
        if self._field is None:
            return None
        return getattr(self._field, "_debug_geodesic_last", None)

    def analyze_topology(
        self,
        group_by: str | None = None,
        *,
        adj_pairs: list[tuple[int, int]] | None = None,
        group_contiguity: bool | None = None,
        adjacency: bool | None = None,
        orientation: bool | None = None,
        adj_min_shared_length: float | None = None,
    ) -> TopologyAnalysis:
        """Analyze the topology of the current Voronoi tessellation.

        Checks all three topology stages without modifying the cartogram.

        Parameters
        ----------
        group_by : str or None
            Column in the source GeoDataFrame identifying groups (e.g. state
            for congressional districts).  Required for stage 1 (group
            contiguity).  ``None`` → stage 1 is skipped.
        adj_pairs : list of (i, j) or None
            Input-adjacent district index pairs for stages 2 (adjacency) and
            3 (orientation).  ``None`` → auto-computed from the source
            GeoDataFrame geometries.  If no source GDF is available, stages
            2 & 3 are skipped.
        group_contiguity : bool or None
            Run stage 1 (group contiguity check).  ``None`` (default) →
            ``True`` when *group_by* is provided, ``False`` otherwise.
        adjacency : bool or None
            Run stage 2 (adjacency violation check).  ``None`` (default) →
            ``True`` when *group_by* is ``None``, ``False`` otherwise.
        orientation : bool or None
            Run stage 3 (orientation alignment check).  ``None`` (default) →
            ``True`` when *group_by* is ``None``, ``False`` otherwise.
        adj_min_shared_length : float or None
            Minimum shared border length for two cells to be considered
            adjacent.  ``None`` = no extra filter.

        Returns
        -------
        TopologyAnalysis
            Snapshot of the current topology state with GDF index labels.
        """
        from carto_flow.geo_utils.adjacency import find_adjacent_pairs

        from .contiguity import _repair_contiguity

        # Auto-resolve stage flags
        if group_contiguity is None:
            group_contiguity = group_by is not None
        if adjacency is None:
            adjacency = group_by is None
        if orientation is None:
            orientation = group_by is None

        cells = list(self.cells)
        n = len(cells)
        labels: list = list(self._source_gdf.index) if self._source_gdf is not None else list(range(n))

        # --- Stage 1: group contiguity ---
        discontiguous_groups: list[tuple[Any, list[Any]]] = []
        if group_contiguity and group_by is not None:
            if self._source_gdf is None or group_by not in self._source_gdf.columns:
                raise RuntimeError(f"group_by column {group_by!r} not found in source GeoDataFrame")
            _, satellites = _repair_contiguity(cells, list(self._source_gdf[group_by]), max_passes=0)
            # Convert slot indices → GDF index labels
            discontiguous_groups = [(grp_id, [labels[s] for s in slot_indices]) for grp_id, slot_indices in satellites]

        # --- Stages 2 & 3: need Voronoi adjacency + input adjacency ---
        violated_adjacency: list[tuple[Any, Any]] = []
        n_adjacency_pairs = 0
        misaligned_orientation: list[tuple[Any, Any, float]] = []
        mean_orientation_cosine: float | None = None

        if adjacency or orientation:
            # Resolve adj_pairs
            _adj: list[tuple[int, int]] | None = adj_pairs
            if _adj is None and self._source_gdf is not None:
                _adj = [
                    (i, j)
                    for i, j, _ in find_adjacent_pairs(
                        list(self._source_gdf.geometry),
                        min_shared_length=adj_min_shared_length,
                    )
                ]

            if _adj:
                # Build Voronoi adjacency set (slot-indexed)
                voronoi_adj: set[tuple[int, int]] = set()
                for s1, s2, _ in find_adjacent_pairs(cells, min_shared_length=adj_min_shared_length):
                    voronoi_adj.add((min(s1, s2), max(s1, s2)))

                def _v_adj(s1: int, s2: int) -> bool:
                    return (min(s1, s2), max(s1, s2)) in voronoi_adj

                # Stage 2: adjacency violations
                if adjacency:
                    n_adjacency_pairs = len(_adj)
                    for di, dj in _adj:
                        if not _v_adj(di, dj):
                            violated_adjacency.append((labels[di], labels[dj]))

                # Stage 3: orientation
                if orientation:
                    geom_pos: np.ndarray | None = None
                    if self._source_gdf is not None:
                        geom_pos = np.array(
                            [[g.centroid.x, g.centroid.y] for g in self._source_gdf.geometry],
                            dtype=np.float64,
                        )
                    if geom_pos is not None:
                        cell_xy_arr = np.array([[c.centroid.x, c.centroid.y] for c in cells], dtype=np.float64)
                        cosines: list[float] = []
                        for di, dj in _adj:
                            if not _v_adj(di, dj):
                                continue
                            d_in = geom_pos[dj] - geom_pos[di]
                            d_out = cell_xy_arr[dj] - cell_xy_arr[di]
                            n_in = float(np.linalg.norm(d_in))
                            n_out = float(np.linalg.norm(d_out))
                            cos = 1.0 if n_in < 1e-10 or n_out < 1e-10 else float(np.dot(d_in, d_out) / (n_in * n_out))
                            cosines.append(cos)
                            if cos <= 0:
                                misaligned_orientation.append((labels[di], labels[dj], round(cos, 6)))
                        mean_orientation_cosine = float(np.mean(cosines)) if cosines else None

        return TopologyAnalysis(
            discontiguous_groups=discontiguous_groups,
            violated_adjacency=violated_adjacency,
            n_adjacency_pairs=n_adjacency_pairs,
            misaligned_orientation=misaligned_orientation,
            mean_orientation_cosine=mean_orientation_cosine,
            degenerate_cells=self.degenerate_cells,
        )

    def repair_topology(
        self,
        group_by: str | None = None,
        *,
        adj_pairs: list[tuple[int, int]] | None = None,
        max_passes: int = 20,
        group_contiguity: bool | None = None,
        compactness: bool | None = None,
        adjacency: bool | None = None,
        orientation: bool | None = None,
        adj_min_shared_length: float | None = None,
    ) -> TopologyRepairReport:
        """Return a new cartogram with topology issues repaired.

        Applies the same three-stage permutation pipeline used during
        Lloyd relaxation — group contiguity (stage 1), adjacency (stage 2),
        and orientation (stage 3) — as a post-hoc operation.  No generator
        positions are moved; only the cell-to-district mapping is permuted.

        Parameters
        ----------
        group_by : str or None
            Column in the source GeoDataFrame identifying groups.  Required
            for stage 1; ``None`` or ``group_contiguity=False`` disables it.
        adj_pairs : list of (i, j) or None
            Input-adjacent district index pairs.  ``None`` → auto-computed
            from the source GeoDataFrame geometries.
        max_passes : int
            Maximum repair passes per active stage (default 20).
        group_contiguity : bool or None
            Run stage 1 (group contiguity repair).  ``None`` (default) →
            ``True`` when *group_by* is provided, ``False`` otherwise.
        compactness : bool or None
            Run stage 1.5 (compactness enhancement).  ``None`` (default) →
            ``True`` when *group_by* is provided, ``False`` otherwise.
        adjacency : bool or None
            Run stage 2 (adjacency permutation).  ``None`` (default) →
            ``True`` when *group_by* is ``None``, ``False`` otherwise.
        orientation : bool or None
            Run stage 3 (orientation alignment).  ``None`` (default) →
            ``True`` when *group_by* is ``None``, ``False`` otherwise.
        adj_min_shared_length : float or None
            Minimum shared border length for two cells to be considered
            adjacent.  ``None`` = no extra filter.

        Returns
        -------
        TopologyRepairReport
            Contains the repaired :attr:`~TopologyRepairReport.cartogram`,
            :attr:`~TopologyRepairReport.before` and
            :attr:`~TopologyRepairReport.after` analyzes, the list of
            :attr:`~TopologyRepairReport.stages_run`, and the
            :attr:`~TopologyRepairReport.reassigned` district labels.
        """
        import dataclasses
        import warnings

        from carto_flow.geo_utils.adjacency import find_adjacent_pairs

        from .contiguity import _compose_topology_permutation

        if self._weighted:
            warnings.warn(
                "repair_topology() permutes existing Voronoi cell geometries without "
                "recomputing them. When weights are non-uniform the repaired cartogram "
                "will not preserve the intended proportional cell areas. "
                "Use in-loop topology repair (VoronoiOptions._topology_repair) instead, "
                "or call repair_topology() only for visual/topological diagnostics.",
                UserWarning,
                stacklevel=2,
            )

        # Auto-resolve stage flags (same logic as analyze_topology)
        if group_contiguity is None:
            group_contiguity = group_by is not None
        if compactness is None:
            compactness = group_by is not None
        if adjacency is None:
            adjacency = group_by is None
        if orientation is None:
            orientation = group_by is None

        before = self.analyze_topology(
            group_by,
            adj_pairs=adj_pairs,
            group_contiguity=group_contiguity,
            adjacency=adjacency,
            orientation=orientation,
            adj_min_shared_length=adj_min_shared_length,
        )

        # Resolve adj_pairs for repair
        _adj: list[tuple[int, int]] | None = adj_pairs
        if _adj is None and self._source_gdf is not None:
            _adj = [
                (i, j)
                for i, j, _ in find_adjacent_pairs(
                    list(self._source_gdf.geometry),
                    min_shared_length=adj_min_shared_length,
                )
            ]

        groups = (
            list(self._source_gdf[group_by])
            if group_by is not None and self._source_gdf is not None and group_by in self._source_gdf.columns
            else None
        )
        geom_positions: np.ndarray | None = None
        if self._source_gdf is not None:
            geom_positions = np.array(
                [[g.centroid.x, g.centroid.y] for g in self._source_gdf.geometry],
                dtype=np.float64,
            )

        cells = list(self.cells)
        slot_of, stages_run = _compose_topology_permutation(
            cells,
            groups,
            _adj,
            group_contiguity=group_contiguity,
            compactness=compactness,
            adjacency=adjacency,
            orientation=orientation,
            max_passes=max_passes,
            geom_positions=geom_positions,
            min_shared_length=adj_min_shared_length,
        )

        # Apply permutation to cells and positions
        new_cells = np.array([cells[s] for s in slot_of])
        new_positions = self.positions[slot_of]

        labels: list = list(self._source_gdf.index) if self._source_gdf is not None else list(range(len(cells)))
        reassigned = [labels[d] for d in range(len(slot_of)) if slot_of[d] != d]

        # Keep _source_gdf unchanged: group labels and original polygon geometries
        # are properties of each district (generator), not of the cells.
        # repair_topology() permutes cells (slot_of), but district d still belongs
        # to the same group and still originated from the same polygon.
        # Permuting _source_gdf would make analyze_topology() compare the wrong
        # group labels against the wrong cells (same mapping as pre-repair → no
        # improvement visible), and plot_displacement() would start arrows from the
        # wrong original locations.
        new_source_gdf = self._source_gdf
        new_cartogram = dataclasses.replace(
            self,
            cells=new_cells,
            positions=new_positions,
            _source_gdf=new_source_gdf,
            _field=None,
        )

        after = new_cartogram.analyze_topology(
            group_by,
            adj_pairs=_adj,
            group_contiguity=group_contiguity,
            adjacency=adjacency,
            orientation=orientation,
            adj_min_shared_length=adj_min_shared_length,
        )

        return TopologyRepairReport(
            cartogram=new_cartogram,
            before=before,
            after=after,
            stages_run=stages_run,
            _original_cartogram=self,
            reassigned=reassigned,
        )

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Return a GeoDataFrame with original columns and Voronoi cell geometries.

        Replaces the geometry column of the source GeoDataFrame with the
        corresponding Voronoi cell polygon.  Index and all non-geometry columns
        are preserved.

        Returns
        -------
        GeoDataFrame

        Raises
        ------
        RuntimeError
            If the source GeoDataFrame was not stored (should not happen in
            normal usage).
        """
        import geopandas as gpd

        if self._source_gdf is None:
            return gpd.GeoDataFrame(geometry=list(self.cells))

        result = self._source_gdf.copy()
        result = result.set_geometry(list(self.cells))
        if self.area_errors is not None:
            result["area_error_pct"] = self.area_errors
        return result

    def plot(
        self,
        column: str | None = None,
        cmap: str | None = None,
        show_edges: bool = True,
        legend: bool = False,
        labels: bool | str | list[str] | None = None,
        label_fontsize: int = 8,
        label_color: str = "black",
        ax: Any | None = None,
        **kwargs: Any,
    ) -> VoronoiPlotResult:
        """Plot Voronoi cells.

        Parameters
        ----------
        column : str or None
            Column from the source GeoDataFrame to use for choropleth coloring.
            ``None`` (default) colors each cell with a distinct categorical
            color derived from its index.  Pass ``"area_error_pct"`` to plot
            the signed per-cell area error as a diverging choropleth.
        cmap : str or None
            Colormap name.  Defaults to ``"tab20"`` for categorical coloring
            (``column=None``), ``"RdBu_r"`` for ``"area_error_pct"``, and
            ``"viridis"`` for other numeric columns.
        show_edges : bool
            Draw cell borders.  Default ``True``.
        legend : bool
            Show a legend or colorbar.  Default ``False``.
        labels : bool, str, list of str, or None
            Annotate each cell with a text label.  ``True`` = index values;
            ``str`` = column name; ``list[str]`` = explicit strings; ``None``
            = no labels (default).
        label_fontsize : int
            Font size for cell labels.  Default ``8``.
        label_color : str
            Text color for cell labels.  Default ``"black"``.
        ax : matplotlib Axes or None
            Axes to draw on.  A new figure is created if ``None``.
        **kwargs
            Passed to ``GeoDataFrame.plot()``.

        Returns
        -------
        VoronoiPlotResult
            Dataclass with ``ax`` and ``collections``.

        Examples
        --------
        >>> pr = result.plot()
        >>> pr.ax.set_title("Equal-area Voronoi")

        >>> pr = result.plot(column="population", cmap="Blues", legend=True)

        >>> pr = result.plot(column="area_error_pct", legend=True)

        >>> pr = result.plot(labels=True)
        """
        from .visualization import plot_cartogram

        return plot_cartogram(
            self,
            column=column,
            cmap=cmap,
            show_edges=show_edges,
            legend=legend,
            labels=labels,
            label_fontsize=label_fontsize,
            label_color=label_color,
            ax=ax,
            **kwargs,
        )

    def plot_displacement(
        self,
        *,
        state: str = "final",
        show_geometries: bool = True,
        show_centroids: bool = True,
        show_adjacency: bool = True,
        show_positions: bool = False,
        show_displacement: bool = True,
        legend: bool = True,
        geometry_style: dict | None = None,
        adjacency_style: dict | None = None,
        centroid_style: dict | None = None,
        position_style: dict | None = None,
        displacement_style: dict | None = None,
        ax: Any | None = None,
        figsize: tuple[float, float] = (10, 8),
        title: str = "Voronoi cartogram — centroid redistribution",
    ) -> VoronoiPlotResult:
        """Plot centroids and displacement arrows on a single axis.

        Parameters
        ----------
        state : {"original", "final"}
            Which state to visualize.  ``"original"`` draws the source
            geometries and their centroids; ``"final"`` (default) draws the
            Voronoi cells and their centroids.  Displacement arrows always run
            original → final regardless of this setting.
        show_geometries : bool
            Draw geometry polygons behind the centroid scatter.  Default ``True``.
        show_centroids : bool
            Draw centroid scatter markers.  Default ``True``.
        show_adjacency : bool
            Draw adjacency graph edges.  Default ``True``.
        show_positions : bool
            Also scatter ``self.positions`` (the Lloyd generator points) as a
            secondary marker (only used when ``state="final"``).  Default ``False``.
        show_displacement : bool
            Draw colored arrows from original to final cell-centroid positions.
            Default ``True``.
        legend : bool
            Show the displacement-magnitude colorbar (and any other legends).
            Default ``True``.
        geometry_style : dict or None
            Styling for geometry polygons, merged over defaults
            ``{"facecolor": "#e8eef4", "edgecolor": "#9aafc4", "linewidth": 0.4, "alpha": 0.6}``.
            All keys forwarded to ``GeoDataFrame.plot()``.

            The ``"color"`` key supports multi-type resolution:

            - ``str`` matching a column → choropleth; use ``"cmap"`` (default
              ``"Blues"``) to set the colormap.
            - any other ``str`` → solid fill color.
            - 1-D numeric array → per-geometry values mapped via ``"cmap"``.
            - list/array of color strings → per-geometry colors.
        adjacency_style : dict or None
            Styling for adjacency graph edges, merged over defaults
            ``{"color": "#6b8fa8", "linewidth": 0.6, "alpha": 0.5, "zorder": 2}``.
            All keys forwarded to ``ax.plot()``.
        centroid_style : dict or None
            Styling for centroid scatter markers, merged over defaults
            ``{"s": 18, "c": "#1a5276", "zorder": 3, "linewidths": 0}``.
            All keys forwarded to ``ax.scatter()``.

            The ``"c"`` (or ``"color"``) key supports the same multi-type
            resolution as ``geometry_style["color"]``; use ``"cmap"`` (default
            ``"viridis"``) for column/numeric cases.
        position_style : dict or None
            Styling for the ``self.positions`` generator-point overlay (only
            used when ``show_positions=True``), merged over defaults
            ``{"s": 14, "marker": "x", "c": "#e74c3c", "zorder": 5, "linewidths": 0.8}``.
            All keys forwarded to ``ax.scatter()``.
        displacement_style : dict or None
            Styling for displacement arrows, merged over defaults
            ``{"cmap": "plasma", "lw": 0.9, "mutation_scale": 6, "arrowstyle": "-|>", "min_frac": 0.01}``.

            - ``"cmap"`` — arrow/colorbar colormap.
            - ``"min_frac"`` — skip arrows whose magnitude is below this
              fraction of the maximum.
            - remaining keys forwarded into ``arrowprops``.
        ax : matplotlib Axes or None
            Axes to draw on.  A new figure is created if ``None``.
        figsize : tuple of float
            Figure size used when ``ax`` is ``None``.
        title : str
            Axis title.

        Returns
        -------
        VoronoiPlotResult
            Dataclass with ``ax`` and ``collections``.

        Raises
        ------
        RuntimeError
            If no source GeoDataFrame is stored on the result.
        ValueError
            If ``state`` is not ``"original"`` or ``"final"``.
        """
        from .visualization import plot_displacement

        return plot_displacement(
            self,
            state=state,
            show_geometries=show_geometries,
            show_centroids=show_centroids,
            show_adjacency=show_adjacency,
            show_positions=show_positions,
            show_displacement=show_displacement,
            legend=legend,
            geometry_style=geometry_style,
            adjacency_style=adjacency_style,
            centroid_style=centroid_style,
            position_style=position_style,
            displacement_style=displacement_style,
            ax=ax,
            figsize=figsize,
            title=title,
        )
