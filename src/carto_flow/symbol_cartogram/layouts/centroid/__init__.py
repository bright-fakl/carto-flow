"""Centroid-based layout: CentroidLayout and CentroidLayoutOptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ...symbols import CircleSymbol, SquareSymbol
from ..base import Layout, _apply_kwargs_to_options
from ..data_prep import LayoutData
from ..layout_result import LayoutResult, Transform

__all__ = ["CentroidLayout", "CentroidLayoutOptions", "CentroidMetrics"]

SQUARE_FINAL_ITERATIONS = 500
"""Upper bound of the passes that settle square overlaps after the last attraction cycle."""


@dataclass
class CentroidMetrics:
    """Algorithm-specific metrics for CentroidLayout."""

    final_max_overlap: float = 0.0
    cleanup_iterations: int = 0


@dataclass
class CentroidLayoutOptions:
    """Options for centroid-based placement.

    Places symbols at geometry centroids with optional overlap removal.

    Parameters
    ----------
    shape : {"circle", "square"}
        Shape the symbols are separated as. ``"circle"`` (default) separates along the line between the centers.
        ``"square"`` separates axis-aligned squares with the same area as the circles, so that their edges can touch
        (with ``spacing=0`` they sit flush); the layout's canonical symbol is then a square, and drawing squares on it
        leaves no gaps and no overlaps.
    spacing : float
        Gap between symbols as fraction of average symbol size. Default: 0.05
    remove_overlap : bool
        Whether to run overlap removal after placing at centroids. Default: True
    max_iterations : int
        Maximum iterations per overlap resolution pass. Default: 20
    overlap_tolerance : float
        Tolerance for overlap resolution convergence. Default: 1e-4
    origin_attraction : float
        After overlap resolution, pull each symbol this fraction toward its
        pre-resolution position, then re-resolve overlaps. Repeated
        ``attraction_cycles`` times. Keeps symbols from drifting far from
        their starting positions. 0 = off. Default: 0.2
    group_attraction : float
        After overlap resolution, pull each symbol this fraction toward its
        group centroid, then re-resolve overlaps. Repeated
        ``attraction_cycles`` times. Active when ``group_by`` or
        ``tile_count`` is set. 0 = off. Default: 0.15
    attraction_cycles : int
        Number of attract → re-resolve iterations to run after the primary
        overlap resolution. Each cycle applies one attraction step then one
        separation step, so attraction and separation have equal weight.
        More cycles pull symbols closer to their origins or group centroids.
        Default: 100

    """

    shape: Literal["circle", "square"] = "circle"
    spacing: float = 0.05
    remove_overlap: bool = True
    max_iterations: int = 20
    overlap_tolerance: float = 1e-4
    origin_attraction: float = 0.2
    group_attraction: float = 0.15
    attraction_cycles: int = 100

    def validate(self) -> None:
        """Validate options."""
        if self.shape not in ("circle", "square"):
            raise ValueError(f"shape must be 'circle' or 'square', got {self.shape!r}")
        if not 0 <= self.spacing <= 1:
            raise ValueError("spacing must be between 0 and 1")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.overlap_tolerance <= 0:
            raise ValueError("overlap_tolerance must be positive")
        if not 0 <= self.origin_attraction <= 1:
            raise ValueError("origin_attraction must be between 0 and 1")
        if not 0 <= self.group_attraction <= 1:
            raise ValueError("group_attraction must be between 0 and 1")
        if self.attraction_cycles < 0:
            raise ValueError("attraction_cycles must be >= 0")


class CentroidLayout(Layout):
    """Layout that places symbols at geometry centroids.

    Places symbols at the centroids of the original geometries, with optional
    local overlap removal.

    Returns LayoutResult with CircleSymbol as canonical symbol, or SquareSymbol with ``shape="square"``.

    Parameters
    ----------
    options : CentroidLayoutOptions, optional
        Full options object. Defaults to CentroidLayoutOptions().
    **kwargs
        Individual option overrides applied on top of *options*.

    Examples
    --------
    >>> layout = CentroidLayout()  # Default: remove overlap
    >>> layout = CentroidLayout(remove_overlap=False)  # Just centroids
    >>> layout = CentroidLayout(spacing=0.1)  # Larger gaps
    >>> layout = CentroidLayout(shape="square", spacing=0.0)  # Squares that sit flush

    """

    #: ``group_by`` drives the intra-group attraction step (``group_attraction``).
    supports_group_by = True

    def __init__(self, options: CentroidLayoutOptions | None = None, /, **kwargs) -> None:
        if options is None:
            options = CentroidLayoutOptions()
        self._options = _apply_kwargs_to_options(options, kwargs)
        self._options.validate()

    def _compute(self, data: LayoutData, show_progress: bool = True, save_history: bool = False) -> LayoutResult:
        """Place symbols at centroids with optional overlap removal.

        Parameters
        ----------
        data : LayoutData
            Preprocessed layout data.
        show_progress : bool
            Display progress feedback during placement.
        save_history : bool
            Record position snapshots per iteration.

        Returns
        -------
        LayoutResult
            Immutable layout result with CircleSymbol as canonical.

        """
        from ._placement import resolve_overlaps

        # Tiles from tile_count expansion start coincident at the geometry centroid.
        # Coincident starts settle 3-5r from centroid after primary separation,
        # much closer than a pre-spread arrangement would produce.
        positions = data.positions.copy()

        # Squares of the same area as the circles: the half-side is sqrt(pi)/2 times the radius
        square = self._options.shape == "square"
        sizes = data.sizes * (np.sqrt(np.pi) / 2) if square else data.sizes

        # group_ids is already N-level from data_prep
        n_group_ids = data.group_ids

        # Record pre-resolution positions for origin attraction
        start_positions = positions.copy()

        if self._options.remove_overlap:
            # Global expansion (global_step_fraction=0.5) guarantees monotonic convergence:
            # it computes the exact factor needed and applies a fraction of it each iteration.
            # Local-only separation (global_step_fraction=0.0) is Jacobi iteration and can
            # oscillate non-convergently for dense clusters. The attraction cycles counteract
            # any drift from global expansion.
            positions, info = resolve_overlaps(
                positions=positions,
                radii=sizes,
                spacing=self._options.spacing,
                max_iterations=self._options.max_iterations,
                overlap_tolerance=self._options.overlap_tolerance,
                global_step_fraction=0.5,
                shape=self._options.shape,
            )

            use_origin = self._options.origin_attraction > 0
            use_group = n_group_ids is not None and self._options.group_attraction > 0

            if (use_origin or use_group) and self._options.attraction_cycles > 0:
                total_cleanup_iters = 0
                cleanup_info: dict = {}
                for _ in range(self._options.attraction_cycles):
                    if use_origin:
                        positions += self._options.origin_attraction * (start_positions - positions)

                    if use_group and n_group_ids is not None:
                        for g in np.unique(n_group_ids):
                            mask = n_group_ids == g
                            group_center = positions[mask].mean(axis=0)
                            positions[mask] += self._options.group_attraction * (group_center - positions[mask])

                    # Local-only separation for cleanup: attraction introduces only small,
                    # bounded overlaps so global expansion is not needed and would compound
                    # across cycles, causing unbounded drift.
                    positions, cleanup_info = resolve_overlaps(
                        positions=positions,
                        radii=sizes,
                        spacing=self._options.spacing,
                        max_iterations=20,
                        overlap_tolerance=self._options.overlap_tolerance,
                        global_step_fraction=0.0,
                        shape=self._options.shape,
                    )
                    total_cleanup_iters += cleanup_info.get("iterations", 0)

                if square:
                    # Squares separate along one axis per pair, which needs more passes than circles to settle
                    # after the last attraction step; run until no overlap is left.
                    positions, cleanup_info = resolve_overlaps(
                        positions=positions,
                        radii=sizes,
                        spacing=self._options.spacing,
                        max_iterations=SQUARE_FINAL_ITERATIONS,
                        overlap_tolerance=self._options.overlap_tolerance,
                        global_step_fraction=0.0,
                        local_step_fraction=1.0,
                        shape="square",
                    )
                    total_cleanup_iters += cleanup_info.get("iterations", 0)

                info["cleanup_iterations"] = total_cleanup_iters
                info["cleanup_max_overlap"] = cleanup_info.get("final_max_overlap", 0.0)
        else:
            info = {"iterations": 0, "final_max_overlap": 0.0}

        # Compute base_size as average size
        base_size = float(np.mean(sizes))

        # Create transforms
        transforms = [
            Transform(
                position=(float(positions[i, 0]), float(positions[i, 1])),
                scale=float(sizes[i] / base_size) if base_size > 0 else 1.0,
            )
            for i in range(len(positions))
        ]

        # Extract CRS
        crs = None
        if data.source_gdf.crs is not None:
            crs = data.source_gdf.crs.to_wkt()

        from ..layout_result import AlgorithmMetrics

        metrics = AlgorithmMetrics(
            converged=True,
            iterations=info.get("iterations", 0),
            final_overlaps=0,
            algorithm=CentroidMetrics(
                final_max_overlap=float(info.get("final_max_overlap", 0.0)),
                cleanup_iterations=int(info.get("cleanup_iterations", 0)),
            ),
        )

        return LayoutResult(
            canonical_symbol=SquareSymbol() if square else CircleSymbol(),
            transforms=transforms,
            base_size=base_size,
            positions=data.geometry_positions if data.geometry_positions is not None else data.positions,
            sizes=sizes,
            adjacency=data.adjacency,
            bounds=data.bounds,
            crs=crs,
            layout_type="centroid",
            metrics=metrics,
            valid_mask=data.valid_mask,
            source_indices=data.source_indices,
            group_ids=data.group_ids,
        )
