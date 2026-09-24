"""Layout result classes for symbol cartograms.

This module defines the immutable LayoutResult that captures the output
of layout algorithms, and the Transform dataclass for per-geometry transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..symbols import Symbol
    from ..tiling import TilingResult


@dataclass
class AlgorithmMetrics:
    """Final scalar summaries common to all layouts.

    Attributes
    ----------
    converged : bool or None
        Whether the algorithm met its convergence criterion.
    iterations : int or None
        Number of iterations executed.
    final_overlaps : int or None
        Number of overlapping nearest-neighbor pairs at termination.
    algorithm : Any
        Algorithm-specific final scalars (PackingMetrics,
        FlowDensityMetrics, CentroidMetrics, GridMetrics, etc.).
    """

    converged: bool | None = None
    iterations: int | None = None
    final_overlaps: int | None = None
    algorithm: Any = None


@dataclass
class SimulationHistory:
    """Per-iteration arrays common to all iterative layouts.

    Attributes
    ----------
    positions : list[np.ndarray] | None
        Position snapshots, each of shape ``(n, 2)``.  Only populated
        when ``save_history=True``.
    overlaps : np.ndarray | None
        Per-iteration overlap count.  Shape: ``(n_iters,)``.
    algorithm : Any
        Algorithm-specific per-iteration arrays (PackingHistory or
        FlowDensityHistory depending on the layout used).
    """

    positions: list[NDArray[np.floating]] | None = None
    overlaps: NDArray[np.intp] | None = None
    algorithm: Any = None

    def __len__(self) -> int:
        """Number of recorded iterations."""
        if self.overlaps is not None:
            return len(self.overlaps)
        if self.algorithm is not None:
            for arr in vars(self.algorithm).values():
                if hasattr(arr, "__len__"):
                    return len(arr)
        if self.positions is not None:
            return len(self.positions)
        return 0


@dataclass(frozen=True)
class Transform:
    """Transformation applied to a symbol.

    Note: This is an internal class that stores rotation in radians.
    User-facing APIs (Styling, TransformedSymbol) use degrees for convenience.

    Attributes
    ----------
    position : tuple[float, float]
        Center position (x, y) for the symbol.
    rotation : float
        Rotation angle in radians (counter-clockwise). Default: 0.0
        User-facing APIs accept degrees and convert internally.
    scale : float
        Scale multiplier. Default: 1.0
    reflection : bool
        Whether to reflect the symbol about the vertical axis. Default: False

    """

    position: tuple[float, float] = (0.0, 0.0)
    rotation: float = 0.0
    scale: float = 1.0
    reflection: bool = False

    def compose(self, other: Transform) -> Transform:
        """Compose this transform with another.

        The resulting transform applies this transform first, then the other.

        Parameters
        ----------
        other : Transform
            Transform to apply after this one.

        Returns
        -------
        Transform
            Composed transform.

        """
        # Compose positions (add translation)
        new_x = self.position[0] + other.position[0]
        new_y = self.position[1] + other.position[1]

        # Compose rotations (add angles)
        new_rotation = self.rotation + other.rotation

        # Compose scales (multiply)
        new_scale = self.scale * other.scale

        # Compose reflections (XOR)
        new_reflection = self.reflection != other.reflection

        return Transform(
            position=(new_x, new_y),
            rotation=new_rotation,
            scale=new_scale,
            reflection=new_reflection,
        )


@dataclass
class LayoutResult:
    """Immutable output from layout algorithm.

    Contains canonical symbol and per-geometry transforms, plus the
    serializable preprocessing data (positions, sizes, adjacency, bounds, crs).
    The original GeoDataFrame is NOT stored here (too large for serialization).

    Attributes
    ----------
    canonical_symbol : Symbol
        The base symbol shape for this layout.
    transforms : list[Transform]
        One transform per input geometry.
    base_size : float
        Reference size for scaling.
    positions : NDArray[np.floating]
        Original centroid positions, shape (n, 2).
    sizes : NDArray[np.floating]
        Symbol sizes, shape (n,).
    adjacency : NDArray[np.floating]
        Adjacency matrix from original geometries, shape (n, n).
    bounds : tuple[float, float, float, float]
        Geographic bounds (xmin, ymin, xmax, ymax).
    crs : str | None
        CRS information as WKT string from source GeoDataFrame.
    layout_type : str
        Registry key of the layout that produced the result (``"packing"``,
        ``"flow_density"``, ``"centroid"``, ``"grid"``, ``"mosaic"``). It
        records which layout ran, and selects the result class to rebuild
        when a serialized result is read back with :meth:`from_dict`.
    metrics : AlgorithmMetrics | None
        Final scalar summaries (converged, iterations, overlaps, algorithm-specific).
        Populated by all layout types.
    history : SimulationHistory | None
        Per-iteration diagnostics and optional position snapshots.
        Only populated by iterative layouts (packing, flow density, centroid).

    """

    canonical_symbol: Symbol
    transforms: list[Transform]
    base_size: float
    positions: NDArray[np.floating]
    sizes: NDArray[np.floating]
    adjacency: NDArray[np.floating]
    bounds: tuple[float, float, float, float]
    crs: str | None = None
    layout_type: str = ""
    metrics: Any = None
    history: Any = None
    valid_mask: NDArray[np.bool_] | None = None
    source_indices: NDArray[np.intp] | None = None
    group_ids: NDArray[np.intp] | None = None

    @property
    def converged(self) -> bool | None:
        """Whether the layout algorithm converged."""
        return self.metrics.converged if self.metrics is not None else None

    @property
    def iterations(self) -> int | None:
        """Number of iterations executed."""
        return self.metrics.iterations if self.metrics is not None else None

    @property
    def overlaps(self) -> Any:
        """Per-iteration overlap counts from simulation history."""
        if self.history is not None:
            return self.history.overlaps
        return None

    def style(
        self,
        styling: Styling | None = None,
        **kwargs,
    ) -> SymbolCartogram:
        """Apply styling to create symbol cartogram.

        Parameters
        ----------
        styling : Styling or None
            Pre-configured Styling object.
        **kwargs
            Convenience kwargs for simple cases (symbol, scale, etc.)
            Creates a temporary Styling object internally.

        Returns
        -------
        SymbolCartogram
            Rendered cartogram with styled symbols.

        """
        # Import here to avoid circular import
        from ..styling import Styling

        if styling is None:
            styling = Styling(**kwargs)
        return styling.apply(self)

    def serialize(self) -> dict[str, Any]:
        """Serialize for persistence.

        Returns
        -------
        dict
            Serialized representation of the layout result.

        Notes
        -----
        All data is serialized including positions, sizes, adjacency, bounds, crs.
        The original GeoDataFrame is NOT stored - it must be provided separately
        when attribute merging is needed via SymbolCartogram.to_geodataframe().

        Examples
        --------
        >>> import json
        >>> result = create_layout(gdf, "population")
        >>> serialized = result.serialize()
        >>> json.dump(serialized, open("layout.json", "w"))

        """
        # Serialize transforms
        transforms_data = [
            {
                "position": list(t.position),
                "rotation": t.rotation,
                "scale": t.scale,
                "reflection": t.reflection,
            }
            for t in self.transforms
        ]

        # Serialize canonical symbol (by class name and params)
        symbol_class = type(self.canonical_symbol).__name__
        symbol_params: dict[str, Any] = {}

        # Handle different symbol types
        if hasattr(self.canonical_symbol, "pointy_top"):
            symbol_params["pointy_top"] = self.canonical_symbol.pointy_top
        if hasattr(self.canonical_symbol, "tiling_type"):
            # IsohedralTileSymbol
            symbol_params["tiling_type"] = self.canonical_symbol.tiling_type
            if (
                hasattr(self.canonical_symbol, "prototile_params")
                and self.canonical_symbol.prototile_params is not None
            ):
                symbol_params["prototile_params"] = self.canonical_symbol.prototile_params
            if hasattr(self.canonical_symbol, "edge_curves") and self.canonical_symbol.edge_curves:
                symbol_params["edge_curves"] = self.canonical_symbol.edge_curves

        return {
            "canonical_symbol": {
                "class": symbol_class,
                "params": symbol_params,
            },
            "transforms": transforms_data,
            "base_size": self.base_size,
            "positions": self.positions.tolist(),
            "sizes": self.sizes.tolist(),
            "adjacency": self.adjacency.tolist(),
            "bounds": list(self.bounds),
            "crs": self.crs,
            "layout_type": self.layout_type,
            "source_indices": self.source_indices.tolist() if self.source_indices is not None else None,
            "group_ids": self.group_ids.tolist() if self.group_ids is not None else None,
        }

    @classmethod
    def from_serialized(cls, data: dict[str, Any]) -> LayoutResult:
        """Reconstruct from serialized data.

        Parameters
        ----------
        data : dict
            Serialized layout result from serialize().

        Returns
        -------
        LayoutResult
            Reconstructed layout result.

        Examples
        --------
        >>> import json
        >>> serialized = json.load(open("layout.json"))
        >>> result = LayoutResult.from_serialized(serialized)
        >>> cartogram = result.style(symbol="circle")

        """
        from ..symbols import (
            CircleSymbol,
            HexagonSymbol,
            IsohedralTileSymbol,
            SquareSymbol,
        )

        # Reconstruct canonical symbol
        symbol_data = data["canonical_symbol"]
        symbol_class_name = symbol_data["class"]
        symbol_params = symbol_data.get("params", {})

        symbol_classes = {
            "CircleSymbol": CircleSymbol,
            "SquareSymbol": SquareSymbol,
            "HexagonSymbol": HexagonSymbol,
            "IsohedralTileSymbol": IsohedralTileSymbol,
        }

        symbol_cls = symbol_classes.get(symbol_class_name)
        if symbol_cls is None:
            raise ValueError(f"Unknown symbol class: {symbol_class_name}")

        # Handle IsohedralTileSymbol specially
        if symbol_class_name == "IsohedralTileSymbol":
            tiling_type = symbol_params.get("tiling_type", 1)
            prototile_params = symbol_params.get("prototile_params")
            edge_curves = symbol_params.get("edge_curves")
            canonical_symbol = IsohedralTileSymbol(
                tiling_type=tiling_type,
                prototile_params=prototile_params,
                edge_curves=edge_curves,
            )
        else:
            canonical_symbol = symbol_cls(**symbol_params)

        # Reconstruct transforms
        transforms = [
            Transform(
                position=tuple(t["position"]),
                rotation=t["rotation"],
                scale=t["scale"],
                reflection=t["reflection"],
            )
            for t in data["transforms"]
        ]

        # Reconstruct arrays
        positions = np.array(data["positions"], dtype=float)
        sizes = np.array(data["sizes"], dtype=float)
        adjacency = np.array(data["adjacency"], dtype=float)
        bounds = tuple(data["bounds"])
        crs = data.get("crs")

        source_indices_raw = data.get("source_indices")
        source_indices = np.array(source_indices_raw, dtype=np.intp) if source_indices_raw is not None else None
        group_ids_raw = data.get("group_ids")
        group_ids = np.array(group_ids_raw, dtype=np.intp) if group_ids_raw is not None else None

        layout_type = data.get("layout_type", "")
        result_cls: type[LayoutResult]
        if layout_type == "grid":
            result_cls = GridLayoutResult
        elif layout_type == "mosaic":
            result_cls = MosaicLayoutResult
        else:
            result_cls = LayoutResult

        return result_cls(
            canonical_symbol=canonical_symbol,
            transforms=transforms,
            base_size=data["base_size"],
            positions=positions,
            sizes=sizes,
            adjacency=adjacency,
            bounds=bounds,
            crs=crs,
            layout_type=layout_type,
            source_indices=source_indices,
            group_ids=group_ids,
        )


@dataclass
class TiledLayoutResult(LayoutResult):
    """Base for grid and mosaic layouts that carry tiling-specific data.

    Attributes
    ----------
    tiling_result : TilingResult | None
        Tiling polygons, transforms, and adjacency. Set at creation;
        ``None`` immediately after ``from_serialized()`` until restored by
        ``SymbolCartogram.load()``.
    assignments : NDArray[np.intp] | None
        Per-tile geometry index. Shape ``(n_valid_tiles,)``.
    """

    tiling_result: TilingResult | None = None
    assignments: NDArray[np.intp] | None = None

    def plot_tiling(
        self,
        cartogram: SymbolCartogram | None = None,
        ax: plt.Axes | None = None,
        show_symbols: bool = True,
        show_assigned: bool = True,
        show_unassigned: bool = True,
        show_pool: bool = False,
        assigned_color: str = "#d4e6f1",
        unassigned_color: str = "#f5f5f5",
        core_edgecolor: str = "#e6972a",
        ring_edgecolor: str = "#d9534f",
        tile_edgecolor: str = "#999999",
        tile_linewidth: float = 0.5,
        pool_linewidth: float = 1.5,
        tile_alpha: float = 0.5,
        **kwargs: Any,
    ) -> TilingPlotResult:
        """Visualize the tiling grid underlying this layout result.

        Parameters
        ----------
        cartogram : SymbolCartogram, optional
            If provided and ``show_symbols=True``, overlays symbol geometries.
        ax : plt.Axes, optional
            Axes to plot on. Created if not provided.
        show_symbols : bool
            Overlay symbol geometries. Requires *cartogram*. Default True.
        show_assigned : bool
            Show occupied tiles. Default True.
        show_unassigned : bool
            Show empty tiles. Default True.
        show_pool : bool
            Use edge color to mark solver-pool membership on every tile
            (``MosaicLayoutResult`` only). Core tiles get *core_edgecolor*
            and extra-ring tiles get *ring_edgecolor* regardless of whether
            they are assigned; tiles outside the pool keep *tile_edgecolor*.
            Default False.
        core_edgecolor : str
            Edge color for core tiles when *show_pool* = True. Default amber.
        ring_edgecolor : str
            Edge color for extra-ring tiles when *show_pool* = True.
            Default salmon.
        pool_linewidth : float
            Line width for pool-tile borders. Default 1.5.
        **kwargs
            Forwarded to ``cartogram.plot()`` when ``show_symbols=True``.
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon as MplPolygon

        from ..plot_results import TilingPlotResult

        if self.tiling_result is None:
            raise ValueError("Tiling data not available (result was deserialized without tiling).")

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(10, 8))

        assigned_set = set(self.assignments.tolist()) if self.assignments is not None else set()

        # Pool membership sets (MosaicLayoutResult only)
        core_set_vis: set[int] = set()
        ring_set_vis: set[int] = set()
        if show_pool:
            pool_indices = getattr(self, "pool_tile_indices", None)
            if pool_indices is not None:
                pool_set = set(pool_indices.tolist())
                core_indices = getattr(self, "core_tile_indices", None)
                if core_indices is not None:
                    core_set_vis = set(core_indices.tolist())
                    ring_set_vis = pool_set - core_set_vis
                else:
                    core_set_vis = pool_set

        # Classify each tile into one of five buckets:
        #   outside_unassigned, core_unassigned, ring_unassigned,
        #   core_assigned, ring_assigned
        buckets: dict[str, list[MplPolygon]] = {
            "outside_unassigned": [],
            "core_unassigned": [],
            "ring_unassigned": [],
            "core_assigned": [],
            "ring_assigned": [],
        }
        for i, poly in enumerate(self.tiling_result.polygons):
            coords = np.array(poly.exterior.coords)
            patch = MplPolygon(coords, closed=True)
            is_assigned = i in assigned_set
            if i in core_set_vis:
                buckets["core_assigned" if is_assigned else "core_unassigned"].append(patch)
            elif i in ring_set_vis:
                buckets["ring_assigned" if is_assigned else "ring_unassigned"].append(patch)
            elif is_assigned:
                # Assigned but pool membership unknown (non-mosaic result).
                buckets["core_assigned"].append(patch)
            else:
                buckets["outside_unassigned"].append(patch)

        def _pc(patches: list[MplPolygon], facecolor: str, edgecolor: str, linewidth: float) -> PatchCollection | None:
            if not patches:
                return None
            pc = PatchCollection(
                patches,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=tile_alpha,
            )
            ax.add_collection(pc)
            return pc

        # Draw order: outside -> ring -> core (pool membership visible through z-order)
        pc_unassigned = None
        if show_unassigned:
            pc_unassigned = _pc(buckets["outside_unassigned"], unassigned_color, tile_edgecolor, tile_linewidth)

        pc_assigned = pc_core = pc_ring = None
        if show_pool:
            pc_ring = _pc(buckets["ring_unassigned"], unassigned_color, ring_edgecolor, pool_linewidth)
            if show_assigned:
                _pc(buckets["ring_assigned"], assigned_color, ring_edgecolor, pool_linewidth)
            pc_core = _pc(buckets["core_unassigned"], unassigned_color, core_edgecolor, pool_linewidth)
            if show_assigned:
                # pc_assigned stays None: assigned tiles are split across core and ring.
                _pc(buckets["core_assigned"], assigned_color, core_edgecolor, pool_linewidth)
        elif show_assigned:
            pc_assigned = _pc(
                buckets["core_assigned"] + buckets["ring_assigned"],
                assigned_color,
                tile_edgecolor,
                tile_linewidth,
            )

        symbols_result = None
        if show_symbols and cartogram is not None:
            symbols_result = cartogram.plot(ax=ax, **kwargs)

        all_coords = np.vstack([np.array(p.exterior.coords) for p in self.tiling_result.polygons])
        ax.set_xlim(all_coords[:, 0].min(), all_coords[:, 0].max())
        ax.set_ylim(all_coords[:, 1].min(), all_coords[:, 1].max())
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title("Tiling Grid")

        return TilingPlotResult(
            ax=ax,
            assigned_tiles=pc_assigned,
            unassigned_tiles=pc_unassigned,
            core_tiles=pc_core,
            ring_tiles=pc_ring,
            symbols=symbols_result,
        )


@dataclass
class GridLayoutResult(TiledLayoutResult):
    """Layout result from GridBasedLayout."""

    layout_type: str = "grid"


@dataclass
class MosaicLayoutResult(TiledLayoutResult):
    """Layout result from MosaicLayout.

    Attributes
    ----------
    tiles_gdf : gpd.GeoDataFrame | None
        Tile-level GeoDataFrame with ``geometry_id`` column.
    regions_gdf : gpd.GeoDataFrame | None
        Region-level GeoDataFrame with ``tile_count`` and ``target_count``.
    counts : NDArray[np.intp] | None
        Target tile count per geometry, shape ``(n_geometries,)``.
    core_tile_indices : NDArray[np.intp] | None
        Indices of tiles whose overlap with the study union reached
        ``min_overlap_frac`` (the calibrated core pool).
    pool_tile_indices : NDArray[np.intp] | None
        Indices of all tiles in the solver pool (core + extra rings).
        Useful for visualizing which tiles were available to the solver.
    """

    layout_type: str = "mosaic"
    tiles_gdf: Any = None
    regions_gdf: Any = None
    counts: NDArray[np.intp] | None = None
    core_tile_indices: NDArray[np.intp] | None = None
    pool_tile_indices: NDArray[np.intp] | None = None


# Import at end to avoid circular import
if TYPE_CHECKING:
    import matplotlib.pyplot as plt

    from ..plot_results import TilingPlotResult
    from ..result import SymbolCartogram
    from ..styling import Styling
