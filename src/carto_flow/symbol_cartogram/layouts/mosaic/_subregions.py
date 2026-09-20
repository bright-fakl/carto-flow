"""Split multi-part geometries into tileable sub-regions.

The mosaic solver requires every unit it lays out to occupy a single connected
block of tiles.  Its unit was the *geometry*, so a MultiPolygon whose parts are
genuinely separated on the ground -- Michigan's two peninsulas -- was forced
together by the contiguity repair, and ``converged`` could never legitimately be
True on such an input.

This module splits a geometry into sub-regions, one per part large enough to
hold a tile, and apportions the geometry's tile count across them by area.
Everything downstream of :func:`split_multipart_regions` works at sub-region
granularity; the geometry index is recovered through ``part_to_geom`` when the
output tables are built, so ``G`` -- and therefore counts, sizes, positions,
``source_indices`` and ``group_ids`` -- is untouched.

Area is the only per-part quantity the pipeline has.  There is no sub-regional
density anywhere: ``flow_cartogram.density`` sets ``pop_density = value / area``
and writes that single number into every cell of the polygon, so integrating the
density field over a part is *exactly* area-proportional.  Users who do hold
per-part values should split the input upstream and pass a per-part
``tile_count`` -- the only exact route.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SubRegions", "apportion_largest_remainder", "split_multipart_regions"]


@dataclass
class SubRegions:
    """Sub-region decomposition of the working geometries.

    Attributes
    ----------
    geometries : list of shapely.Geometry
        One working-space geometry per sub-region, length ``P >= G``.
    counts : np.ndarray of int32, shape (P,)
        Tile count apportioned to each sub-region.  Sums to ``counts_G.sum()``
        exactly, and per geometry to that geometry's count exactly.
    part_to_geom : np.ndarray of int32, shape (P,)
        Sub-region index -> source geometry index.
    n_split_geometries : int
        How many geometries produced more than one sub-region.
    """

    geometries: list
    counts: np.ndarray
    part_to_geom: np.ndarray
    n_split_geometries: int

    @property
    def is_identity(self) -> bool:
        """True when no geometry was split, so P == G and nothing changes."""
        return self.n_split_geometries == 0


def apportion_largest_remainder(total: int, weights: np.ndarray) -> np.ndarray:
    """Split ``total`` across ``weights`` so the parts sum to ``total`` exactly.

    Standard largest-remainder (Hamilton) apportionment: floor the exact quotas,
    then hand the leftover units to the largest fractional remainders, ties broken
    by larger weight and then by lower index so the result is deterministic.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if total <= 0 or weights.sum() <= 0:
        return np.zeros(len(weights), dtype=np.int64)
    quota = total * weights / weights.sum()
    base = np.floor(quota).astype(np.int64)
    leftover = total - int(base.sum())
    if leftover > 0:
        frac = quota - base
        order = sorted(range(len(weights)), key=lambda i: (-frac[i], -weights[i], i))
        for i in order[:leftover]:
            base[i] += 1
    return base


def _group_parts(
    part_polys: list,
    qualifying: list[int],
    areas_for_weight: np.ndarray,
) -> tuple[list[list[int]], np.ndarray]:
    """Attach every non-qualifying part to the nearest qualifying one.

    Returns the part-index groups (one per qualifying part, in order) and their
    summed weights.  Nothing is discarded, so the weights add up to the
    geometry's whole area and the apportionment below cannot leak tiles to a
    sub-region that does not exist.
    """
    groups: list[list[int]] = [[j] for j in qualifying]
    slot_of = {j: s for s, j in enumerate(qualifying)}
    centroids = [part_polys[j].centroid for j in qualifying]
    for j in range(len(part_polys)):
        if j in slot_of:
            continue
        c = part_polys[j].centroid
        s = min(range(len(qualifying)), key=lambda k: c.distance(centroids[k]))
        groups[s].append(j)
    weights = np.array([float(areas_for_weight[g].sum()) for g in groups], dtype=np.float64)
    return groups, weights


def split_multipart_regions(
    original_geometries: list,
    working_geometries: list,
    counts: np.ndarray,
    tile_area: float,
    min_part_tiles: float,
) -> SubRegions:
    """Decompose multi-part geometries into tileable sub-regions.

    Parameters
    ----------
    original_geometries : list of shapely.Geometry
        Input geometries, before any morph.  Only their part *areas* are read,
        and only to weight the apportionment: how a region's data divides over
        its parts must not depend on how far the morph pushed them.
    working_geometries : list of shapely.Geometry
        The geometries the tiling actually runs on (morphed, or the originals
        when ``morph=False``).  Qualification, attachment and the emitted
        sub-region geometries all live in this space, because that is the space
        the tiles occupy.
    counts : np.ndarray, shape (G,)
        Tile count per geometry.
    tile_area : float
        Area of one tile, from calibration.
    min_part_tiles : float
        A part qualifies as a sub-region when its working area is at least
        ``min_part_tiles * tile_area``.  A tile-size threshold is self-scaling
        and a part smaller than a tile could not hold one anyway.

    Notes
    -----
    Parts are matched between the two geometry lists **by index**, and a geometry
    whose part count changes under the morph is left whole -- shapely preserves
    part order through the vertex displacement, but ``make_valid`` may merge or
    drop degenerate slivers, and a silent mismatch would attribute one part's
    area to another.

    Only ``tile_count`` inputs can split: with ``group_by`` every geometry
    carries exactly one symbol, and a one-tile region is never split (below).

    A sub-region that would receive **zero** tiles is demoted back to an ordinary
    part of its nearest neighbour and the apportionment is redone: a sub-region
    with no tiles is not a sub-region, and flooring it at 1 instead would have to
    take that tile from another part -- impossible when a region has fewer tiles
    than parts (a one-tile state cannot be two blocks).
    """
    import shapely
    from shapely.ops import unary_union

    G = len(working_geometries)
    geoms_out: list = []
    counts_out: list[int] = []
    part_to_geom: list[int] = []
    n_split = 0

    for g in range(G):
        n_g = int(counts[g])
        work = working_geometries[g]
        parts_w = list(shapely.get_parts(work))
        parts_o = list(shapely.get_parts(original_geometries[g]))

        split_groups: list[list[int]] | None = None
        alloc: np.ndarray | None = None
        if n_g >= 2 and len(parts_w) >= 2 and len(parts_w) == len(parts_o):
            areas_o = np.array([p.area for p in parts_o], dtype=np.float64)
            limit = min_part_tiles * tile_area
            qualifying = [j for j, p in enumerate(parts_w) if p.area >= limit]
            while len(qualifying) >= 2:
                groups, weights = _group_parts(parts_w, qualifying, areas_o)
                allot = apportion_largest_remainder(n_g, weights)
                zeros = [s for s in range(len(groups)) if allot[s] == 0]
                if not zeros:
                    split_groups, alloc = groups, allot
                    break
                # Demote the emptiest zero-tile sub-region and re-apportion.
                drop = min(zeros, key=lambda s: (weights[s], qualifying[s]))
                qualifying.pop(drop)

        if split_groups is None or alloc is None:
            geoms_out.append(work)
            counts_out.append(n_g)
            part_to_geom.append(g)
            continue

        n_split += 1
        for s, idx in enumerate(split_groups):
            polys = [parts_w[j] for j in idx]
            geoms_out.append(polys[0] if len(polys) == 1 else unary_union(polys))
            counts_out.append(int(alloc[s]))
            part_to_geom.append(g)

    return SubRegions(
        geometries=geoms_out,
        counts=np.array(counts_out, dtype=np.int32),
        part_to_geom=np.array(part_to_geom, dtype=np.int32),
        n_split_geometries=n_split,
    )
