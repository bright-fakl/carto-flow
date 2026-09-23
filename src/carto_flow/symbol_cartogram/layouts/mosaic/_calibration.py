"""Tile size calibration for mosaic layout."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import shapely

if TYPE_CHECKING:
    from ...tiling import Tiling, TilingResult

__all__ = ["TilingSetup", "calibrate_tiling"]

# Maximum correction iterations after the analytical estimate.
_MAX_ADJUST = 20

# Pre-flight guard: if estimated total tiles in bounding box exceeds this,
# skip tiling.generate() and return early.
_MAX_TILES = 50_000


@dataclass
class TilingSetup:
    """All tiling-related data needed by the assignment step.

    Attributes
    ----------
    tile_size : float
        Calibrated tile size.
    tiling_result : TilingResult
        Tiling generated at the calibrated tile size.
    adj_list : list of list of int
        Global tile adjacency lists (``adj_list[t]`` = neighbor tile indices).
    core_set : set of int
        Tile indices whose intersection with the study union covers at least
        ``min_overlap_frac`` of the tile area.  ``len(core_set) == target_count``
        after successful calibration.
    valid_tile_indices : list of int
        Same as ``core_set`` as a sorted list. Reserve tiles are added
        separately via ``extra_tile_rings`` in ``MosaicLayoutOptions``.
    """

    tile_size: float
    tiling_result: TilingResult
    adj_list: list[list[int]]
    core_set: set[int]
    valid_tile_indices: list[int]


def calibrate_tiling(
    tiling: Tiling,
    bounds: tuple[float, float, float, float],
    study_union,
    target_count: int,
    *,
    tile_size: float | None = None,
    buffer_rings: int = 1,
    min_overlap_frac: float = 0.1,
) -> TilingSetup:
    """Calibrate tile size and build the full tiling setup.

    Finds the tile size such that exactly *target_count* tiles have at least
    *min_overlap_frac* of their area inside *study_union*.

    Parameters
    ----------
    tiling : Tiling
        Any Tiling subclass instance.
    bounds : (minx, miny, maxx, maxy)
        Bounding box of the study area (typically ``study_union.bounds``).
    study_union : shapely.Geometry
        Union of all (morphed) input geometries.
    target_count : int
        Total target tile count (= sum of per-geometry counts).
    tile_size : float or None
        If provided, skip calibration and use this exact tile size.
    buffer_rings : int
        Number of extra tile-width rings to add to the tiling bounds on the
        final build. Ensures that ring-expansion in the caller (``extra_tile_rings``)
        always finds a complete set of neighbor tiles. Default 1.
    min_overlap_frac : float
        Minimum fraction of a tile's area that must intersect *study_union*
        for the tile to count as a core tile.  Values below 0.5 capture tiles
        whose centroid falls outside the geometry (e.g. narrow peninsulas).
        Default 0.1.

    Returns
    -------
    TilingSetup
    """
    if target_count <= 0:
        raise ValueError(f"target_count must be positive, got {target_count}")

    study_area = study_union.area
    if study_area <= 0:
        raise ValueError("study_union has zero area; cannot calibrate tile size.")

    # Reference tiling at unit scale for area/size conversion. Only
    # tile_size and canonical_tile.area are used, so skip adjacency.
    ref = tiling.generate(n_tiles=1, compute_adjacency=False)
    ref_size = ref.tile_size
    unit_area = ref.canonical_tile.area
    if unit_area <= 0:
        raise ValueError("Tiling produced a degenerate canonical tile with zero area.")

    bbox_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])

    def _tile_area_at(size: float) -> float:
        return unit_area * (size / ref_size) ** 2

    def _estimated_bbox_tiles(size: float) -> int:
        return int(bbox_area / _tile_area_at(size))

    def _build(size: float, tile_bounds=None, study_union_override=None, compute_adjacency: bool = True):
        union = study_union_override if study_union_override is not None else study_union
        result = tiling.generate(tile_bounds or bounds, tile_size=size, compute_adjacency=compute_adjacency)
        polys = np.asarray(result.polygons)
        # Restrict to tiles whose bounding box intersects the study union.
        tree = shapely.STRtree(polys)
        candidates = tree.query(union, predicate="intersects")
        if len(candidates) == 0:
            return result, len(polys), []
        cand_polys = polys[candidates]
        inter_areas = shapely.area(shapely.intersection(cand_polys, union))
        tile_areas = shapely.area(cand_polys)
        # Avoid division by zero for degenerate tiles.
        fracs = np.where(tile_areas > 0, inter_areas / tile_areas, 0.0)
        core = sorted(int(candidates[i]) for i in np.where(fracs >= min_overlap_frac)[0])
        return result, len(polys), core

    # Tracks whether `result` already carries real lattice adjacency (vs. the
    # all-False placeholder from a compute_adjacency=False trial build).
    adjacency_fresh = True

    if tile_size is not None:
        # Explicit tile size: skip calibration
        result, T, core = _build(tile_size)
    else:
        # Analytical estimate: tile area ≈ study_area / target_count
        tile_size = ref_size * math.sqrt(study_area / (unit_area * target_count))

        if _estimated_bbox_tiles(tile_size) > _MAX_TILES:
            result, T, core = _build(tile_size)
        else:
            # Pre-simplify study_union for calibration iterations.  Intersection
            # time scales with vertex count; simplify(tile_size/4) keeps the shape
            # accurate enough for tile counting while drastically reducing cost.
            # The final build always uses the original study_union.
            calibration_union = study_union.simplify(tile_size / 4, preserve_topology=True)

            def _build_fast(size: float):
                # Lattice adjacency is only needed for the accepted tile
                # size (built again below); trial builds only need polygons
                # to count core tiles, so skip the adjacency computation.
                return _build(size, study_union_override=calibration_union, compute_adjacency=False)

            result, T, core = _build_fast(tile_size)
            adjacency_fresh = False

            # Sqrt gradient descent: n_core ∝ 1/tile_size², so
            # tile_size_new = tile_size x sqrt(n_core/target).
            # Stop when within 1 tile of target (integer counts oscillate
            # at this scale; extra_tile_rings covers any small deficit).
            for _ in range(_MAX_ADJUST):
                n_core = len(core)
                if abs(n_core - target_count) <= 1:
                    break
                ratio = n_core / target_count if n_core > 0 else 0.5
                new_size = tile_size * math.sqrt(ratio)
                if _estimated_bbox_tiles(new_size) > _MAX_TILES:
                    break
                tile_size = new_size
                result, T, core = _build_fast(tile_size)
                adjacency_fresh = False

    # Final build with expanded bounds so ring-expansion in the caller always
    # finds a complete set of neighbors around every core tile. Also covers
    # the case where the accepted tile size only ever went through a
    # compute_adjacency=False trial build above (buffer_rings == 0): real
    # adjacency must be computed for the returned TilingSetup exactly once.
    if buffer_rings > 0:
        buf = tile_size * (buffer_rings + 0.5)
        expanded = (bounds[0] - buf, bounds[1] - buf, bounds[2] + buf, bounds[3] + buf)
        result, T, core = _build(tile_size, expanded)
    elif not adjacency_fresh:
        result, T, core = _build(tile_size)

    adj_list: list[list[int]] = [list(np.where(result.adjacency[t])[0]) for t in range(T)]
    core_set = set(core)
    valid_tile_indices = core

    return TilingSetup(
        tile_size=tile_size,
        tiling_result=result,
        adj_list=adj_list,
        core_set=core_set,
        valid_tile_indices=valid_tile_indices,
    )
