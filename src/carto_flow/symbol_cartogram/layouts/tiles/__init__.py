"""Tile layout: TileLayout and TileLayoutOptions."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from ..base import Layout, _apply_kwargs_to_options, _orient_tiling, _validate_lattice_options
from ..data_prep import LayoutData
from ..layout_result import AlgorithmMetrics, LayoutResult, TileLayoutResult

__all__ = ["TileLayout", "TileLayoutOptions", "TileMetrics"]

_COMPONENTS = ("separate", "connected", "spaced")
_REPAIR_ROUNDS = 5  # swap and chain searches repeated while they reduce the number of split groups
_CHAIN_DEPTH = 4  # groups in a chain move
_MEMORY_WARNING_BYTES = 2.0e9  # estimated size of the dense matrices above which the layout warns


@dataclass
class TileMetrics:
    """Algorithm-specific metrics for TileLayout.

    Attributes
    ----------
    tiling : str
        Tiling name.
    tile_size : float
        Size of the lattice tiles.
    n_tiles : int
        Lattice tiles in the window.
    n_allowed : int
        Tiles the assignment could use.
    n_parts : int
        Land parts of the input.
    regions_correct : int
        Geometries with exactly their tile count.
    regions_total : int
        Geometries placed.
    n_split_groups : int
        Groups (the geometries themselves without ``group_by``) whose tiles
        form more than one connected piece.
    n_stray_tiles : int
        Tiles outside the largest piece of their group.
    n_pockets : int
        Unused tiles with more than half of their neighbors used.
    pre_morphed : bool
        Whether the assignment used pre-morphed geometries.
    """

    tiling: str = ""
    tile_size: float = 0.0
    n_tiles: int = 0
    n_allowed: int = 0
    n_parts: int = 0
    regions_correct: int = 0
    regions_total: int = 0
    n_split_groups: int = 0
    n_stray_tiles: int = 0
    n_pockets: int = 0
    pre_morphed: bool = False


@dataclass
class TileLayoutOptions:
    """Options for the tile layout.

    Parameters
    ----------
    tiling : Tiling or str
        Tile shape: a ``Tiling`` instance or ``"square"``, ``"hexagon"``,
        ``"triangle"``, ``"quadrilateral"``. Default ``"hexagon"``.
    compactness : float
        Share of the lattice window that is left unused, on a geometric scale
        in [0, 1]: the layout may use ``round(n * (m / n) ** (1 - compactness))``
        of the ``m`` lattice tiles, where ``n`` tiles are needed. At 1 the
        layout uses exactly ``n`` tiles and follows the outline of the map
        exactly, at the cost of larger displacements; below 1 surplus tiles
        let regions stay near their position. Contiguity of groups is
        repaired for compactness below 1 and not guaranteed at 1 or close to it;
        a warning reports groups left in more than one piece. Default 0.75.
    components : {"separate", "connected", "spaced"}
        Where separate land masses (islands, exclaves) are placed: at their
        own position (``"separate"``), against the block of the largest land
        mass on the side of their position (``"connected"``), or with one
        unused tile between them and the block (``"spaced"``). Regions only
        use tiles of their own land mass in all modes.
    pre_morph : bool
        Morph the geometries into a flow cartogram with equal area per tile
        before the assignment, and assign by the morphed geometries. The land
        masses and the usable tiles still follow the input geometries.
        Recommended when region sizes differ widely. Default False.
    morph_options : MorphOptions or None
        Options for the pre-morph. None gives ``MorphOptions(n_iter=100)``.
    protect_contacts : bool
        With one tile per region, reject a move or swap that lowers the number
        of touching pairs of neighboring regions among the regions it moves,
        and keep the result of the protected descent only if it leaves at least
        as many neighbor pairs touching as the unprotected descent. This keeps
        more neighbors in contact and can allow more false contacts and larger
        displacement. Default False.
    spacing : float
        Gap between drawn symbols as a fraction of tile size (0-1). It does
        not change the assignment. Default 0.05.
    rotation : float
        Rotation of the tile lattice in degrees, counter-clockwise, about the
        center of the region that the tiles cover. Default 0.0.
    offset : tuple of float
        Shift of the tile lattice as ``(u, v)`` in multiples of the two
        translation vectors of the lattice (see
        :meth:`~carto_flow.symbol_cartogram.Tiling.with_offset`). Default
        ``(0.0, 0.0)``.
    """

    tiling: object = "hexagon"  # Tiling | str, but avoid circular import
    compactness: float = 0.75
    components: Literal["separate", "connected", "spaced"] = "separate"
    pre_morph: bool = False
    morph_options: object = None  # MorphOptions | None, imported lazily
    protect_contacts: bool = False
    spacing: float = 0.05
    rotation: float = 0.0
    offset: tuple[float, float] = (0.0, 0.0)

    def validate(self) -> None:
        """Validate options."""
        _validate_lattice_options(self.rotation, self.offset)
        if not 0.0 <= self.compactness <= 1.0:
            raise ValueError(f"compactness must be between 0 and 1, got {self.compactness}")
        if self.components not in _COMPONENTS:
            raise ValueError(f"components must be one of {_COMPONENTS}, got {self.components!r}")
        if not 0 <= self.spacing <= 1:
            raise ValueError(f"spacing must be between 0 and 1, got {self.spacing}")


def _part_key(part: shapely.Geometry) -> tuple[float, float, float]:
    """Sort key of a part: largest first, ties by position."""
    c = part.centroid
    return (-round(part.area, 6), round(c.x, 6), round(c.y, 6))


def _canonical_order(geometries: list, bounds: tuple[float, float, float, float]) -> NDArray[np.intp]:
    """Order of geometries by centroid x, then y, then area, independent of their input order.

    Coordinates are rounded relative to the extent of *bounds* so that
    floating-point noise does not change the order; ties keep the input order.
    """
    extent = max(bounds[2] - bounds[0], bounds[3] - bounds[1], 1e-12)
    cents = np.array([g.centroid.coords[0] for g in geometries], dtype=float)
    key = np.round((cents - np.array(bounds[:2])) / extent * 1e6)
    area = np.round(np.array([g.area for g in geometries]) / max(sum(g.area for g in geometries), 1e-300) * 1e6)
    return np.lexsort((area, key[:, 1], key[:, 0])).astype(np.intp)


def _split_capacity(areas: NDArray[np.floating], k: int) -> NDArray[np.intp]:
    """Split *k* tiles over parts in proportion to their areas; the first part keeps at least one."""
    caps = np.zeros(len(areas), dtype=np.intp)
    if k <= 1 or len(areas) == 1:
        caps[0] = k
        return caps
    share = k * areas / areas.sum()
    caps = np.floor(share).astype(np.intp)
    caps[0] = max(caps[0], 1)
    while caps.sum() < k:
        caps[int(np.argmax(share - caps))] += 1
    while caps.sum() > k and (caps > 1).any():
        caps[int(np.argmax(np.where(caps > 1, caps - share, -np.inf)))] -= 1
    return caps


class TileLayout(Layout):
    """Layout that gives every region tiles of a lattice, by allocation and assignment.

    The layout first allocates the tiles it may use: the silhouette of each
    land mass, with a number of tiles set by ``compactness``. It then assigns
    tiles to regions, or to groups with ``group_by``, so that each gets
    exactly its tile count and stays near its geometry. Regions use only tiles
    of their own land mass. A region with parts far from its main part
    receives a share of its tile count for each remote part when the share
    rounds to a whole tile.

    The tiles are assigned by exact optimal transport with a reserve that takes
    the unused tiles. With one tile per region and no grouping this is the
    transport problem with a capacity of 1 per region, followed by swaps and
    moves that keep neighboring regions in contact. With ``tile_count`` or
    ``group_by``, the capacity is the tile count, and a swap search and chain
    moves, repeated while they help, keep each region or group in one piece. Within a group, tiles go to the
    group's geometries by matching their centroids, after fitting the
    centroids to the tiles by translation and uniform scaling.

    Parameters
    ----------
    options : TileLayoutOptions, optional
        Full options object (positional only). Defaults to TileLayoutOptions().
    **kwargs
        Individual option overrides applied on top of *options*.

    Examples
    --------
    >>> layout = TileLayout()
    >>> layout = TileLayout(compactness=0.5, pre_morph=True)

    Notes
    -----
    The tiles are solved as a dense assignment whose size is the number of
    usable tiles, so time and memory grow with its square; the layout warns
    when the dense matrices are estimated to exceed 2 GB. The result does not
    depend on the order of the input rows. The lattice window
    and tile size follow the geometries: a tile has the mean area of a tile
    needed. ``size_normalization`` does not change the assignment.
    """

    supports_group_by = True

    def __init__(self, options: TileLayoutOptions | None = None, /, **kwargs) -> None:
        if options is None:
            options = TileLayoutOptions()
        self._options = _apply_kwargs_to_options(options, kwargs)
        self._options.validate()

    def _compute(self, data: LayoutData, show_progress: bool = True, save_history: bool = False) -> LayoutResult:
        """Allocate and assign tiles and return the result.

        Parameters
        ----------
        data : LayoutData
            Preprocessed layout data.
        show_progress : bool
            Not used.
        save_history : bool
            Not used.

        Returns
        -------
        LayoutResult
            A :class:`TileLayoutResult`.
        """
        from ...tiling import resolve_tiling
        from ..mosaic import _build_geodataframes, _build_transforms
        from ._allocation import allocate_tiles, allocation_size, land_parts
        from ._assignment import (
            FORBIDDEN,
            chain_search,
            group_cost,
            neighbor_lists,
            split_groups,
            swap_search,
            tile_step,
            transport_labels,
        )

        opts = self._options
        source_gdf = data.source_gdf
        geometries = list(shapely.make_valid(np.asarray(list(source_gdf.geometry), dtype=object)))
        n_geoms = len(geometries)
        counts = (data.counts_G if data.counts_G is not None else np.ones(n_geoms)).astype(np.intp)
        n = int(counts.sum())
        if n == 0:
            raise ValueError("TileLayout needs at least one tile to place")

        # Units: the geometries themselves, or the groups of group_by.
        if data.group_ids_G is not None:
            _, unit_of_geom = np.unique(data.group_ids_G, return_inverse=True)
        else:
            unit_of_geom = np.arange(n_geoms)
        unit_of_geom = unit_of_geom.astype(np.intp)
        n_units = int(unit_of_geom.max()) + 1
        unit_cap = np.bincount(unit_of_geom, weights=counts, minlength=n_units).astype(np.intp)
        unit_geom = [
            shapely.union_all([geometries[i] for i in np.flatnonzero(unit_of_geom == u)]) for u in range(n_units)
        ]

        # Lattice: tiles with the area of one needed tile, over the window of the geometries.
        tiling_obj = _orient_tiling(resolve_tiling(opts.tiling), opts.rotation, opts.offset)
        tile_area = math.fsum(g.area for g in unit_geom) / n
        probe = tiling_obj.generate(bounds=(-1.0, -1.0, 1.0, 1.0), tile_size=1.0)
        unit_area = probe.canonical_tile.area / probe.tile_size**2
        tile_size = float(np.sqrt(tile_area / unit_area))
        minx, miny, maxx, maxy = data.bounds
        width, height = maxx - minx, maxy - miny
        needed = 2.0 * n * tile_area
        if width * height < needed:
            grow = float(np.sqrt(needed / (width * height)))
            cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
            minx, maxx = cx - width * grow / 2, cx + width * grow / 2
            miny, maxy = cy - height * grow / 2, cy + height * grow / 2
        tiling_result = tiling_obj.generate(bounds=(minx, miny, maxx, maxy), tile_size=tile_size)
        centers = np.asarray(tiling_result.centers, dtype=float)
        n_lattice = len(centers)
        if n_lattice < n:
            raise ValueError(f"The lattice window has {n_lattice} tiles for {n} needed")
        nbrs = neighbor_lists(tiling_result.adjacency)
        step = tile_step(centers)

        # Canonical order of the units (by position, then area) so that the result does not depend on row order.
        unit_order = _canonical_order(unit_geom, data.bounds)
        geom_rank = np.empty(n_geoms, dtype=np.intp)
        geom_rank[_canonical_order(geometries, data.bounds)] = np.arange(n_geoms)

        # Pieces: the parts of a unit that receive tiles. A part within one tile of the main part joins it;
        # remote parts receive a share of the unit's tiles that rounds to at least one.
        piece_geom: list = []
        piece_cap: list[int] = []
        piece_unit: list[int] = []
        for u in unit_order.tolist():
            if unit_cap[u] == 0:
                continue
            parts = sorted(getattr(unit_geom[u], "geoms", [unit_geom[u]]), key=_part_key)
            parts = [p for p in parts if not p.is_empty]
            main = parts[0]
            core = [p for p in parts if p.distance(main) < step]
            remote = [p for p in parts if all(p is not q for q in core)]
            shapes = [shapely.union_all(core), *remote]
            caps = _split_capacity(np.array([s.area for s in shapes]), int(unit_cap[u]))
            for shape, cap in zip(shapes, caps.tolist(), strict=True):
                if cap > 0:
                    piece_geom.append(shape)
                    piece_cap.append(cap)
                    piece_unit.append(u)
        n_pieces = len(piece_geom)
        capacity = np.array(piece_cap, dtype=np.intp)
        piece_unit_arr = np.array(piece_unit, dtype=np.intp)

        # Land parts of the input geometries; outlines for the silhouette come from the working geometries.
        working, morphed = self._working_geometries(piece_geom, capacity)
        parts, piece_part = land_parts(piece_geom)
        need = np.bincount(piece_part, weights=capacity, minlength=len(parts)).astype(np.intp)
        order = np.argsort(-need, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        piece_part = rank[piece_part]
        need = need[order]
        outlines = []
        for j in range(len(parts)):
            shapes_j = [working[p] for p in np.flatnonzero(piece_part == j)]
            outline = shapely.union_all(shapes_j).buffer(0)
            outlines.append(outline if not outline.is_empty else parts[order[j]])

        size = allocation_size(n, n_lattice, opts.compactness)
        allocation = allocate_tiles(outlines, need, centers, tiling_result.adjacency, size, opts.components)
        allowed = allocation.tiles

        dense_bytes = 8.0 * (n_pieces * n_lattice + 3.0 * len(allowed) ** 2)
        if dense_bytes > _MEMORY_WARNING_BYTES:
            warnings.warn(
                f"TileLayout builds dense matrices of about {dense_bytes / 1e9:.1f} GB "
                f"({n_pieces} pieces, {len(allowed)} usable of {n_lattice} lattice tiles); "
                "lower compactness toward 1 or use fewer regions to reduce it",
                UserWarning,
                stacklevel=3,
            )

        # Transport of the allowed tiles to the pieces, restricted to the land part of each piece.
        cost = np.full((n_pieces, n_lattice), FORBIDDEN)
        foreign = allocation.part_of_tile[allowed][None, :] != piece_part[:, None]
        cost[:, allowed] = group_cost(working, centers[allowed], step) + foreign * FORBIDDEN
        plain = data.group_ids_G is None and bool(np.all(capacity == 1)) and n_pieces == n_geoms
        label, reserve_cost = transport_labels(
            cost, capacity, allowed, centers, nbrs, step, neighbor_iterations=0 if plain else 3
        )
        if plain:
            label = self._contact_descent(
                label,
                allowed,
                centers,
                nbrs,
                working,
                geometries,
                piece_unit_arr,
                allocation.part_of_tile,
                piece_part,
                step,
            )
        else:
            for _ in range(_REPAIR_ROUNDS):
                before = split_groups(label, n_pieces, nbrs)
                label = swap_search(label, cost, reserve_cost, nbrs)
                label = chain_search(label, cost, reserve_cost, nbrs, depth=_CHAIN_DEPTH)
                if split_groups(label, n_pieces, nbrs) >= before:
                    break
        used = np.flatnonzero(label >= 0)
        n_split_pieces = split_groups(label, n_pieces, nbrs)[0]
        if n_split_pieces:
            warnings.warn(
                f"{n_split_pieces} group(s) are split into more than one piece; contiguity is not guaranteed "
                "for compactness close to 1",
                UserWarning,
                stacklevel=3,
            )
        if np.any(cost[label[used], used] >= FORBIDDEN / 2):
            warnings.warn("Some tiles lie outside the land mass of their region", UserWarning, stacklevel=3)

        # Tiles of each unit go to its geometries.
        assignment = np.full(n_lattice, -1, dtype=np.intp)
        for u in range(n_units):
            tiles = used[piece_unit_arr[label[used]] == u]
            members = np.flatnonzero(unit_of_geom == u).astype(np.intp)
            members = members[np.argsort(geom_rank[members], kind="stable")]
            if len(tiles) == 0:
                continue
            if len(members) == 1:
                assignment[tiles] = members[0]
            else:
                assignment[tiles] = self._match_members(members, counts, geometries, centers[tiles], tiles)

        output_tiles = sorted(np.flatnonzero(assignment >= 0).tolist())
        sizes_G = data.sizes_G if data.sizes_G is not None else data.sizes
        tiles_gdf, regions_gdf = _build_geodataframes(
            assignment,
            output_tiles,
            tiling_result,
            float(tiling_result.tile_size),
            counts,
            source_gdf,
            data.group_ids_G,
        )
        transforms, src_idx = _build_transforms(assignment, output_tiles, tiling_result, sizes_G, spacing=opts.spacing)
        crs = source_gdf.crs.to_wkt() if source_gdf.crs is not None else None
        per_region = np.bincount(assignment[assignment >= 0], minlength=n_geoms)
        regions_correct = int(np.sum(per_region == counts))
        n_split, n_stray = split_groups(label, n_pieces, nbrs)
        unused = np.flatnonzero(label < 0)
        k_pocket = max(len(x) for x in nbrs) // 2 + 1
        n_pockets = int(sum(1 for t in unused if sum(1 for u in nbrs[t] if label[u] >= 0) >= k_pocket))
        metrics = AlgorithmMetrics(
            converged=regions_correct == n_geoms and n_split == 0,
            iterations=1,
            final_overlaps=0,
            algorithm=TileMetrics(
                tiling=opts.tiling if isinstance(opts.tiling, str) else type(tiling_obj).__name__,
                tile_size=float(tiling_result.tile_size),
                n_tiles=n_lattice,
                n_allowed=len(allowed),
                n_parts=len(parts),
                regions_correct=regions_correct,
                regions_total=n_geoms,
                n_split_groups=n_split,
                n_stray_tiles=n_stray,
                n_pockets=n_pockets,
                pre_morphed=morphed,
            ),
        )
        return TileLayoutResult(
            canonical_symbol=tiling_obj.canonical_symbol(),
            transforms=transforms,
            base_size=float(tiling_obj.symbol_size_for_tile_size(tiling_result.tile_size)),
            positions=data.geometry_positions if data.geometry_positions is not None else data.positions,
            sizes=data.sizes,
            adjacency=data.adjacency,
            bounds=data.bounds,
            crs=crs,
            metrics=metrics,
            tiling_result=tiling_result,
            assignments=np.array(output_tiles, dtype=np.intp),
            tiles_gdf=tiles_gdf,
            regions_gdf=regions_gdf,
            counts=counts,
            core_tile_indices=allowed,
            pool_tile_indices=allowed,
            valid_mask=data.valid_mask,
            source_indices=src_idx,
            group_ids=data.group_ids_G[src_idx] if data.group_ids_G is not None else None,
        )

    def _working_geometries(self, pieces: list, capacity: NDArray[np.intp]) -> tuple[list, bool]:
        """Geometries the assignment draws tiles to (the pieces, or their flow cartogram) and whether they are morphed."""
        if not self._options.pre_morph or len(pieces) < 2:
            return list(pieces), False
        import geopandas as gpd

        from ....flow_cartogram import create_flow_cartogram
        from ....flow_cartogram.options import MorphOptions

        morph_opts = self._options.morph_options or MorphOptions(n_iter=100, show_progress=False)
        source = gpd.GeoDataFrame({"count": capacity.astype(np.float64)}, geometry=list(pieces))
        try:
            best = create_flow_cartogram(source, "count", options=morph_opts, validate=False).best
            morphed = None if best is None else list(shapely.make_valid(np.asarray(best.geometry, dtype=object)))
        except Exception as error:  # the layout falls back to the input geometries
            warnings.warn(f"pre_morph failed ({error}); using the input geometries", UserWarning, stacklevel=4)
            return list(pieces), False
        if morphed is None or any(g is None or g.is_empty for g in morphed):
            warnings.warn(
                "pre_morph returned no usable geometry; using the input geometries", UserWarning, stacklevel=4
            )
            return list(pieces), False
        return morphed, True

    def _contact_descent(
        self,
        label: NDArray[np.intp],
        allowed: NDArray[np.intp],
        centers: NDArray[np.floating],
        nbrs: list[list[int]],
        working: list,
        geometries: list,
        piece_unit: NDArray[np.intp],
        part_of_tile: NDArray[np.intp],
        piece_part: NDArray[np.intp],
        step: float,
    ) -> NDArray[np.intp]:
        """Improve the contacts of regions with one tile each; return the updated labels.

        With ``protect_contacts`` the descent that rejects moves losing contacts
        is run next to the plain descent, and the one that leaves more neighbor
        pairs touching is kept (the protected one on a tie).
        """
        from ....geo_utils.adjacency import find_bordering_pairs
        from ._descent import contact_descent

        n = len(working)
        tile_of = np.empty(n, dtype=np.intp)
        used = np.flatnonzero(label >= 0)
        tile_of[label[used]] = used
        targets = np.array([g.centroid.coords[0] for g in working], dtype=float)
        piece_of_unit = np.empty(len(piece_unit), dtype=np.intp)
        piece_of_unit[piece_unit] = np.arange(n)
        pairs = sorted(
            (min(piece_of_unit[i], piece_of_unit[j]), max(piece_of_unit[i], piece_of_unit[j]))
            for i, j, _ in find_bordering_pairs(geometries)
        )

        def descend(protect: bool) -> NDArray[np.intp]:
            return contact_descent(
                tile_of,
                allowed,
                centers,
                nbrs,
                targets,
                pairs,
                step,
                tile_ok=lambda t, g: bool(part_of_tile[t] == piece_part[g]),
                protect_contacts=protect,
            )

        result = descend(self._options.protect_contacts)
        if self._options.protect_contacts:
            plain = descend(False)
            nbr_sets = [set(x) for x in nbrs]

            def touching(tiles: NDArray[np.intp]) -> int:
                return sum(int(tiles[b] in nbr_sets[tiles[a]]) for a, b in pairs)

            if touching(plain) > touching(result):
                result = plain
        tile_of = result
        out = np.where(label == -2, -2, -1).astype(np.intp)
        out[tile_of] = np.arange(n)
        return out

    @staticmethod
    def _match_members(
        members: NDArray[np.intp],
        counts: NDArray[np.intp],
        geometries: list,
        tile_centers: NDArray[np.floating],
        tiles: NDArray[np.intp],
    ) -> NDArray[np.intp]:
        """Geometry of each tile of a group: centroids of the members fitted to the tiles, then matched."""
        cents = np.array([geometries[i].centroid.coords[0] for i in members], dtype=float)
        rep = np.repeat(np.arange(len(members)), counts[members])
        weights = counts[members].astype(float)
        mu_c = (cents * weights[:, None]).sum(0) / weights.sum()
        mu_t = tile_centers.mean(0)
        spread_c = float(np.sqrt((weights * ((cents - mu_c) ** 2).sum(1)).sum() / weights.sum()))
        spread_t = float(np.sqrt(((tile_centers - mu_t) ** 2).sum(1).mean()))
        scale = spread_t / spread_c if spread_c > 0 else 1.0
        fitted = (cents[rep] - mu_c) * scale + mu_t
        d = ((fitted[:, None, :] - tile_centers[None, :, :]) ** 2).sum(-1)
        rows, cols = linear_sum_assignment(d)
        out = np.empty(len(tiles), dtype=np.intp)
        out[cols] = members[rep[rows]]
        return out
