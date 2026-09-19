"""Backend objects for create_voronoi_cartogram.

Each backend encapsulates algorithm-specific parameters and factory methods.
``api.py`` calls ``build_field()`` and ``relax_step()`` without knowing which
concrete field class is in use.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from .fields import ExactField, RasterField

__all__ = [
    "AdhesiveBoundary",
    "ElasticBoundary",
    "ExactBackend",
    "RasterBackend",
    "RelaxationSchedule",
]

# ---------------------------------------------------------------------------
# Relaxation schedule
# ---------------------------------------------------------------------------


@dataclass
class RelaxationSchedule:
    """Decaying successive over-relaxation (SOR) schedule.

    At each iteration *i*, the relaxation factor is:

    .. code-block:: python

        max(minimum, start * decay**i)

    Parameters
    ----------
    start : float
        Initial relaxation factor.  ``1.0`` = standard Lloyd (move straight
        to cell centroid).  Values above 1 overshoot, cutting the number of
        iterations needed by 3-5x.  Values above 2 may diverge.
    decay : float
        Multiplicative factor applied each iteration, in ``[0, 1]``.
        ``1.0`` = constant; values below 1 gradually anneal toward
        ``minimum``.
    minimum : float
        Floor value after decay.  ``1.0`` (default) transitions to standard
        Lloyd in the final iterations.
    """

    start: float = 1.9
    decay: float = 0.98
    minimum: float = 1.0

    def __call__(self, iteration: int) -> float:
        return max(self.minimum, self.start * (self.decay**iteration))

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"RelaxationSchedule.start must be >= 0, got {self.start}")
        if self.minimum < 0:
            raise ValueError(f"RelaxationSchedule.minimum must be >= 0, got {self.minimum}")
        if self.minimum > self.start:
            raise ValueError(f"RelaxationSchedule.minimum ({self.minimum}) must be <= start ({self.start})")
        if not (0.0 <= self.decay <= 1.0):
            raise ValueError(f"RelaxationSchedule.decay must be in [0, 1], got {self.decay}")


_RELAXATION_PRESETS: dict[str, Callable[[int], float]] = {
    "lloyd": lambda _: 1.0,
    "overrelax": RelaxationSchedule(start=1.9, decay=0.98, minimum=1.0),
}


def _resolve_relaxation(
    relaxation: float | str | RelaxationSchedule | Callable[[int], float],
) -> Callable[[int], float]:
    """Normalise any accepted relaxation spec to a ``(iteration) -> float`` callable."""
    if callable(relaxation):
        return relaxation
    if isinstance(relaxation, str):
        if relaxation not in _RELAXATION_PRESETS:
            raise ValueError(f"Unknown relaxation preset {relaxation!r}. Valid presets: {list(_RELAXATION_PRESETS)}")
        return _RELAXATION_PRESETS[relaxation]
    if isinstance(relaxation, (int, float)):
        v = float(relaxation)
        return lambda _: v
    raise TypeError(
        f"relaxation must be a float, str preset, RelaxationSchedule, or callable, got {type(relaxation).__name__!r}"
    )


# ---------------------------------------------------------------------------
# Boundary types
# ---------------------------------------------------------------------------


@dataclass
class AdhesiveBoundary:
    """Snap boundary centroids toward the outer boundary after each Lloyd step.

    Parameters
    ----------
    strength : float
        Adhesion strength in ``[0, 1]``.  ``1.0`` = full snap to boundary;
        ``0.3-0.5`` = partial pull.  Only affects centroids whose original
        geometry intersects the outer boundary.
    """

    strength: float = 0.3

    def __post_init__(self) -> None:
        if not (0.0 <= self.strength <= 1.0):
            raise ValueError(f"AdhesiveBoundary.strength must be in [0, 1], got {self.strength}")


@dataclass
class ElasticBoundary:
    """FFT-driven elastic boundary deformation (RasterBackend only).

    Each iteration the outer boundary is deformed by a velocity field driven
    by cell-area pressure, allowing the outer shape to adapt to the
    centroid distribution.  Set ``relaxation=0.0`` in :class:`RasterBackend`
    for pure FFT-flow mode (no Lloyd centroid updates).

    Parameters
    ----------
    strength : float
        Deformation strength in ``[0, 1]``.  ``0`` = rigid hard-clip.
    step_scale : float
        Boundary advection speed as a fraction of the centroid advection
        speed, in ``(0, 1]``.  ``1.0`` = boundary and centroids co-advect at
        the same rate.  Values < 1 make the boundary move more slowly.
    density_smooth : float or None
        Gaussian smoothing sigma applied to the density field before solving
        the FFT velocity field.  ``None`` = no smoothing.
    min_boundary_points : int or None
        Minimum number of vertices to distribute along the boundary for elastic
        deformation tracking.  Simple shapes such as ``"bbox"`` or
        ``"convex_hull"`` have only a handful of corners after internal
        simplification — set this to roughly ``backend.resolution`` for smooth
        deformation.  ``None`` (default) uses whatever vertices remain after
        simplification, which preserves the original behaviour for complex union
        boundaries.
    adhesion_strength : float
        Snap strength for boundary-touching centroids in ``[0, 1]``.
        ``0`` (default) = no snapping.  When ``> 0``, centroids whose original
        geometry touches the outer boundary are pulled toward the *current*
        deformed boundary after each elastic step — combining the elastic shape
        change with centroid adhesion in one pass.  Equivalent to using
        :class:`AdhesiveBoundary` together with elastic deformation, but with
        the snap target tracking the evolving boundary shape.
    """

    strength: float = 0.3
    step_scale: float = 1.0
    density_smooth: float | None = None
    min_boundary_points: int | None = None
    adhesion_strength: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.strength <= 1.0):
            raise ValueError(f"ElasticBoundary.strength must be in [0, 1], got {self.strength}")
        if not (0.0 < self.step_scale <= 1.0):
            raise ValueError(f"ElasticBoundary.step_scale must be in (0, 1], got {self.step_scale}")
        if self.min_boundary_points is not None and self.min_boundary_points < 4:
            raise ValueError(f"ElasticBoundary.min_boundary_points must be >= 4, got {self.min_boundary_points}")
        if not (0.0 <= self.adhesion_strength <= 1.0):
            raise ValueError(f"ElasticBoundary.adhesion_strength must be in [0, 1], got {self.adhesion_strength}")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


@dataclass
class ExactBackend:
    """Exact scipy Voronoi + shapely clipping backend (Lloyd relaxation).

    Parameters
    ----------
    relaxation : float, str, RelaxationSchedule, or callable
        Relaxation factor schedule.  Accepted forms:

        * **float** — constant factor at every iteration.  ``1.0`` = standard
          Lloyd; values above 1 overshoot (faster convergence).
        * ``"lloyd"`` — constant 1.0 (standard Lloyd).
        * ``"overrelax"`` *(default)* — :class:`RelaxationSchedule` decaying
          from 1.9 → 1.0 over the run.
        * :class:`RelaxationSchedule` — explicit ``(start, decay, minimum)``.
        * **callable** ``(iteration: int) -> float`` — fully custom schedule.
    adjacency_spring : float
        Strength of the spring force pulling input-adjacent centroids together,
        blended into each Lloyd step.  ``0`` = pure Lloyd.  Values of 0.1-0.3
        gently prevent neighbouring centroids from drifting apart.
    boundary : AdhesiveBoundary or None
        Boundary snap behaviour.  ``None`` = off.  Pass
        :class:`AdhesiveBoundary` to snap boundary centroids toward the outer
        boundary after each step.

    Examples
    --------
    >>> from carto_flow import create_voronoi_cartogram, ExactBackend, VoronoiOptions
    >>> result = create_voronoi_cartogram(
    ...     gdf,
    ...     backend=ExactBackend(relaxation=RelaxationSchedule(start=1.9, decay=0.95)),
    ...     options=VoronoiOptions(n_iter=100),
    ... )
    """

    relaxation: float | str | RelaxationSchedule | Callable[[int], float] = "overrelax"
    adjacency_spring: float = 0.0
    intra_group_spring: float | None = None
    boundary: AdhesiveBoundary | None = None

    def __post_init__(self) -> None:
        _resolve_relaxation(self.relaxation)  # validate early
        if self.adjacency_spring < 0:
            raise ValueError(f"adjacency_spring must be >= 0, got {self.adjacency_spring}")
        if self.intra_group_spring is not None and self.intra_group_spring < 0:
            raise ValueError(f"intra_group_spring must be >= 0, got {self.intra_group_spring}")
        if self.boundary is not None and not isinstance(self.boundary, AdhesiveBoundary):
            raise TypeError(
                "ExactBackend.boundary must be AdhesiveBoundary or None; "
                "ElasticBoundary is only supported by RasterBackend"
            )

    def build_field(
        self,
        positions: np.ndarray,
        outer,
        *,
        weights: np.ndarray | None,
        adj_pairs: list[tuple[int, int]] | None,
        intra_adj_pairs: list[tuple[int, int]] | None = None,
        boundary_mask: np.ndarray | None,
        adhesion_boundary=None,
    ) -> ExactField:
        from .fields import ExactField

        adhesion_strength = self.boundary.strength if self.boundary is not None else 0.0
        return ExactField(
            positions,
            outer,
            adj_pairs=adj_pairs,
            intra_adj_pairs=intra_adj_pairs,
            boundary_mask=boundary_mask,
            adhesion_boundary=adhesion_boundary,
            adhesion_strength=adhesion_strength,
            weights=weights,
        )

    def relax_step(self, field: ExactField, factor: float, iteration: int = 0) -> None:
        field.relax(
            relaxation_factor=factor,
            topology_weight=self.adjacency_spring,
            intra_topology_weight=self.intra_group_spring or 0.0,
        )


@dataclass
class RasterBackend:
    """Raster nearest-neighbour Lloyd relaxation backend.

    Uses a precomputed grid of land pixels for Voronoi labeling — typically
    10-50x faster than the exact scipy backend.

    Parameters
    ----------
    resolution : int
        Grid resolution applied to the longest bounding-box dimension; the
        shorter dimension is scaled proportionally.  200-400 is a good
        starting point.
    relaxation : float, str, RelaxationSchedule, or callable
        Relaxation factor schedule (see :class:`ExactBackend`).
    adjacency_spring : float
        Spring force strength for input-adjacent centroids (see
        :class:`ExactBackend`).
    intra_group_spring : float or None
        Spring force strength applied to centroids that belong to the **same
        group** (as specified via ``group_by`` in
        :func:`~carto_flow.voronoi_cartogram.create_voronoi_cartogram`).
        When set, ``adjacency_spring`` applies only to *cross-group* adjacent
        pairs and ``intra_group_spring`` applies only to *same-group* adjacent
        pairs.  Use a higher value than ``adjacency_spring`` (e.g.
        ``adjacency_spring=0.1, intra_group_spring=0.5``) to keep districts
        within the same state/region spatially cohesive.  ``None`` (default)
        disables the intra-group spring; ``adjacency_spring`` then applies to
        all adjacent pairs regardless of group membership.
    boundary : AdhesiveBoundary, ElasticBoundary, or None
        Boundary behaviour.  ``None`` = rigid hard-clip (default).

        * :class:`AdhesiveBoundary` — snap boundary centroids toward the outer
          boundary after each step.
        * :class:`ElasticBoundary` — deform the outer boundary each iteration
          via an FFT velocity field.  Set ``relaxation=0.0`` for pure
          FFT-flow mode (no Lloyd centroid updates).
    distance_mode : {"euclidean", "geodesic"}
        Pixel-to-centroid distance algorithm.

        ``"euclidean"``
            cKDTree nearest-neighbour (default, fastest).  Compatible with
            ``area_equalizer_rate > 0``.
        ``"geodesic"``
            Multi-source BFS; wavefront cannot cross inactive (water) pixels,
            eliminating cross-bay assignments.  ``area_equalizer_rate`` is
            ignored in this mode.  Weights are not supported in
            geodesic mode (ignored with a warning).
    area_equalizer_rate : float
        Learning rate for the power-diagram area equaliser
        (``distance_mode="euclidean"`` only, **unweighted** case).  Each
        iteration, per-centroid offsets ``λᵢ`` are updated so that
        ``dist² - λᵢ`` drives cells toward equal area.  Ignored when
        ``weights`` are provided (use ``weight_ramp_iters`` instead).
        Default ``0.1``.  Ignored when ``distance_mode="geodesic"``.
    weight_ramp_iters : int
        Number of iterations over which weights are linearly ramped from
        1 (uniform) to their target values.  Prevents early engulfment when
        weight ratios are large (e.g. US state populations).  ``0`` = no
        ramp.  Default ``10``.
    output_resolution : int or None
        Upsampled grid resolution used only for final cell extraction when
        ``distance_mode="geodesic"`` or ``area_equalizer_rate > 0``.
        Ignored for plain euclidean with ``area_equalizer_rate=0``.  ``None`` = same as
        ``resolution``.
    cell_smoothing_px : float
        Tolerance for smoothing pixel staircases on extracted cells, in
        final-grid pixels (the extraction grid is ``output_resolution`` or
        4x ``resolution``). Resolution-independent by construction. For
        cartographic generalisation in map units, apply
        :func:`carto_flow.geo_utils.simplify_coverage` to the result
        instead. ``0`` disables smoothing. Default ``3.0``.

    Examples
    --------
    >>> from carto_flow import create_voronoi_cartogram, RasterBackend
    >>> result = create_voronoi_cartogram(gdf, backend=RasterBackend(resolution=300))
    >>> result = create_voronoi_cartogram(
    ...     gdf, backend=RasterBackend(resolution=300, distance_mode="geodesic")
    ... )
    >>> result = create_voronoi_cartogram(
    ...     gdf, backend=RasterBackend(resolution=300, boundary=ElasticBoundary(0.3))
    ... )

    Pure FFT-flow mode (no Lloyd relaxation):

    >>> result = create_voronoi_cartogram(
    ...     gdf,
    ...     backend=RasterBackend(
    ...         resolution=300,
    ...         relaxation=0.0,
    ...         boundary=ElasticBoundary(strength=0.3, step_scale=0.8),
    ...         distance_mode="geodesic",
    ...     ),
    ... )
    """

    resolution: int = 300
    relaxation: float | str | RelaxationSchedule | Callable[[int], float] = "overrelax"
    adjacency_spring: float = 0.0
    intra_group_spring: float | None = None
    boundary: AdhesiveBoundary | ElasticBoundary | None = None
    distance_mode: Literal["euclidean", "geodesic"] = "euclidean"
    area_equalizer_rate: float = 0.1
    weight_ramp_iters: int = 10
    output_resolution: int | None = None
    cell_smoothing_px: float = 3.0

    def __post_init__(self) -> None:
        if self.resolution < 10:
            raise ValueError(f"resolution must be >= 10, got {self.resolution}")
        if self.output_resolution is not None and self.output_resolution < 10:
            raise ValueError(f"output_resolution must be >= 10, got {self.output_resolution}")
        if self.cell_smoothing_px < 0:
            raise ValueError(f"cell_smoothing_px must be >= 0, got {self.cell_smoothing_px}")
        _resolve_relaxation(self.relaxation)  # validate early
        if self.adjacency_spring < 0:
            raise ValueError(f"adjacency_spring must be >= 0, got {self.adjacency_spring}")
        if self.intra_group_spring is not None and self.intra_group_spring < 0:
            raise ValueError(f"intra_group_spring must be >= 0, got {self.intra_group_spring}")
        if self.boundary is not None and not isinstance(self.boundary, (AdhesiveBoundary, ElasticBoundary)):
            raise TypeError("RasterBackend.boundary must be AdhesiveBoundary, ElasticBoundary, or None")
        if self.distance_mode not in ("euclidean", "geodesic"):
            raise ValueError(f"distance_mode must be 'euclidean' or 'geodesic', got {self.distance_mode!r}")

        if self.area_equalizer_rate < 0:
            raise ValueError(f"area_equalizer_rate must be >= 0, got {self.area_equalizer_rate}")

    def build_field(
        self,
        positions: np.ndarray,
        outer,
        *,
        weights: np.ndarray | None,
        adj_pairs: list[tuple[int, int]] | None,
        intra_adj_pairs: list[tuple[int, int]] | None = None,
        boundary_mask: np.ndarray | None,
        adhesion_boundary=None,
        debug: bool = False,
    ) -> RasterField:
        from .fields import RasterField

        adhesion_strength = (
            self.boundary.strength
            if isinstance(self.boundary, AdhesiveBoundary)
            else (self.boundary.adhesion_strength if isinstance(self.boundary, ElasticBoundary) else 0.0)
        )
        elasticity = self.boundary.strength if isinstance(self.boundary, ElasticBoundary) else 0.0
        step_scale = self.boundary.step_scale if isinstance(self.boundary, ElasticBoundary) else 1.0
        density_smooth = self.boundary.density_smooth if isinstance(self.boundary, ElasticBoundary) else None
        min_boundary_points = self.boundary.min_boundary_points if isinstance(self.boundary, ElasticBoundary) else None

        return RasterField(
            positions,
            outer,
            resolution=self.resolution,
            final_resolution=self.output_resolution,
            boundary_elasticity=elasticity,
            boundary_step_scale=step_scale,
            density_smooth=density_smooth,
            min_boundary_points=min_boundary_points,
            labeling=self.distance_mode,
            area_eq_weight=self.area_equalizer_rate,
            weight_ramp_iters=self.weight_ramp_iters,
            debug_geodesic=debug,
            cell_smoothing_px=self.cell_smoothing_px,
            adj_pairs=adj_pairs,
            intra_adj_pairs=intra_adj_pairs,
            boundary_mask=boundary_mask,
            adhesion_boundary=adhesion_boundary,
            adhesion_strength=adhesion_strength,
            weights=weights,
        )

    def relax_step(self, field: RasterField, factor: float, iteration: int = 0) -> None:
        field.relax(
            relaxation_factor=factor,
            topology_weight=self.adjacency_spring,
            intra_topology_weight=self.intra_group_spring or 0.0,
            area_eq_weight=self.area_equalizer_rate,
            iteration=iteration,
        )
