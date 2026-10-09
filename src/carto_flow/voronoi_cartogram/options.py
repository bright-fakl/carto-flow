"""Run-control options for Voronoi cartogram generation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

__all__ = ["TopologyRepair", "VoronoiOptions"]


@dataclass
class TopologyRepair:
    """Settings for the periodic topology permutation pipeline.

    The pipeline permutes which geometry each generator represents (without
    moving generators); the next Lloyd step absorbs each swap.  Three stages
    can be enabled independently:

    1. **group_contiguity** — ensure each group's generators occupy adjacent
       Voronoi cells (requires ``group_by`` in
       :func:`~carto_flow.voronoi_cartogram.api.create_voronoi_cartogram`).
    2. **adjacency** — permute slots so that input-adjacent geometry pairs have
       adjacent Voronoi cells.
    3. **orientation** — among already-adjacent pairs, align the relative
       direction between cell centroids with the direction between input
       centroids.

    Parameters
    ----------
    every : int
        Run the pipeline every *N* iterations.
    max_passes : int
        Maximum repair passes per active stage per call.  Kept small (default
        3) to avoid slowing down each iteration; the loop over all iterations
        compensates.
    group_contiguity : bool
        Enable stage 1.  Default ``True``.
    adjacency : bool
        Enable stage 2.  Default ``True``.
    orientation : bool
        Enable stage 3.  Default ``True``.
    """

    every: int = 5
    max_passes: int = 3
    group_contiguity: bool = True
    adjacency: bool = True
    orientation: bool = True

    def __post_init__(self) -> None:
        if self.every < 1:
            raise ValueError(f"TopologyRepair.every must be >= 1, got {self.every}")
        if self.max_passes < 1:
            raise ValueError(f"TopologyRepair.max_passes must be >= 1, got {self.max_passes}")


@dataclass
class VoronoiOptions:
    """Run-control options for Voronoi cartogram generation.

    Algorithm-specific parameters (relaxation schedule, resolution, boundary
    treatment, distance mode, etc.) live on the backend objects
    (:class:`~carto_flow.voronoi_cartogram.backends.ExactBackend`,
    :class:`~carto_flow.voronoi_cartogram.backends.RasterBackend`).
    These options control only iteration count, stopping criteria, output
    recording, and topology repair.

    Parameters
    ----------
    n_iter : int
        Maximum number of Lloyd iterations. ``0`` returns the initial
        clipped Voronoi diagram without any relaxation.
    tol : float or None
        Stop early when max centroid displacement falls below this value.
        ``None`` = run all ``n_iter`` iterations.
    area_cv_tol : float or None
        Stop early when the area coefficient of variation drops below this
        value.  ``None`` = no area-CV stopping criterion.  Can be combined
        with ``tol``; the first condition that fires wins.
    simplify_tol : float or None
        Simplification tolerance for the outer boundary polygon before
        clipping.  Reduces per-iteration cost for complex coastlines.
        ``None`` = no simplification.
    record_history : bool or int
        ``False`` — no history (default).  ``True`` — record a snapshot at
        every iteration.  ``int N`` — record a snapshot every *N* iterations.
        Snapshots are accessible via ``VoronoiCartogram.history``.
    record_cells : bool
        Include clipped Voronoi cell polygons in each snapshot.  Polygon
        construction is expensive; disable when only centroid positions are
        needed.  Ignored when ``record_history=False``.
    show_progress : bool
        Print per-iteration diagnostics (displacement, area CV, relaxation
        factor).
    debug : bool
        Enable verbose diagnostic output from backend internals: topology
        repair cycle details and geodesic cross-water labeling warnings.
    prescale_components : bool
        If ``True``, detect connected-component groups (sets of geometries that
        touch or share a boundary) and uniformly scale each group to its target
        total area before running relaxation.  Important when disconnected
        components (e.g. Alaska, Hawaii) start at very different sizes relative
        to their weights.
    fix_topology : int, TopologyRepair, or None
        Periodic topology permutation pipeline.

        * ``None`` *(default)* — disabled.
        * **int** *N* — enable all three stages, run every *N* iterations with
          ``max_passes=3``.
        * :class:`TopologyRepair` — full control over cadence, passes, and
          which stages are active.
    adj_min_shared_length : float or None
        Minimum shared border length for two cells to be considered adjacent
        in topology repair.  Pairs whose shared border is shorter than this
        value are not counted as adjacent.  ``None`` (default) = no extra
        filter beyond the auto-computed ``distance_tolerance`` in
        :func:`~carto_flow.geo_utils.adjacency.find_adjacent_pairs`.
    area_error_tol : float
        Area accuracy for weighted runs, as a relative error (``0.01`` =
        1 %).  With ``weights``, :class:`~carto_flow.voronoi_cartogram.backends.RasterBackend`
        (euclidean distance) returns exact power-diagram cells and solves
        their offsets until every cell area is within this relative error of
        its weight-proportional target.  For every weighted run, the result
        reports ``metrics["converged"] = False`` and a ``RuntimeWarning``
        states the mean and maximum errors when the mean absolute relative
        area error of the output cells exceeds this value.  Not used without
        ``weights``.  Default ``0.01``.

    Examples
    --------
    >>> from carto_flow import create_voronoi_cartogram, RasterBackend, VoronoiOptions
    >>> result = create_voronoi_cartogram(
    ...     gdf,
    ...     backend=RasterBackend(resolution=300),
    ...     options=VoronoiOptions(n_iter=50, show_progress=True),
    ... )
    >>> # Record history every 5 iterations
    >>> result = create_voronoi_cartogram(
    ...     gdf,
    ...     options=VoronoiOptions(record_history=5),
    ... )
    >>> # Topology repair every 10 iterations, adjacency stage only
    >>> from carto_flow import TopologyRepair
    >>> result = create_voronoi_cartogram(
    ...     gdf,
    ...     options=VoronoiOptions(
    ...         fix_topology=TopologyRepair(every=10, group_contiguity=False, orientation=False)
    ...     ),
    ... )
    """

    n_iter: int = 30
    tol: float | None = None
    area_cv_tol: float | None = None
    simplify_tol: float | None = None
    record_history: bool | int = False
    record_cells: bool = False
    show_progress: bool = False
    debug: bool = False
    prescale_components: bool = False
    fix_topology: int | TopologyRepair | None = None
    adj_min_shared_length: float | None = None
    area_error_tol: float = 0.01

    def __post_init__(self) -> None:
        if self.n_iter < 0:
            raise ValueError(f"n_iter must be >= 0, got {self.n_iter}")
        if not self.area_error_tol > 0:
            raise ValueError(f"area_error_tol must be > 0, got {self.area_error_tol}")
        if self.area_cv_tol is not None and self.area_cv_tol < 0:
            raise ValueError(f"area_cv_tol must be >= 0 or None, got {self.area_cv_tol}")
        if (
            isinstance(self.record_history, int)
            and not isinstance(self.record_history, bool)
            and self.record_history < 1
        ):
            raise ValueError(f"record_history as int must be >= 1, got {self.record_history}")
        if isinstance(self.fix_topology, int) and not isinstance(self.fix_topology, bool) and self.fix_topology < 1:
            raise ValueError(f"fix_topology as int must be >= 1, got {self.fix_topology}")

    def _topology_repair(self) -> TopologyRepair | None:
        """Return a resolved TopologyRepair, or None if disabled."""
        if self.fix_topology is None or self.fix_topology is False:
            return None
        if isinstance(self.fix_topology, TopologyRepair):
            return self.fix_topology
        return TopologyRepair(every=int(self.fix_topology))

    def copy_with(self, **kwargs: Any) -> VoronoiOptions:
        """Return a new VoronoiOptions with specified fields overridden."""
        valid = set(self.__dataclass_fields__)
        invalid = set(kwargs) - valid
        if invalid:
            raise ValueError(f"Invalid field(s): {invalid}. Valid: {valid}")
        return replace(self, **kwargs)
