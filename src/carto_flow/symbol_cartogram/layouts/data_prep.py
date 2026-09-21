"""Data preparation for symbol cartogram layouts.

This module provides utilities to prepare GeoDataFrame data for layout
algorithms, including computing symbol sizes based on data values and
extracting geographic information.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from ...geo_utils.prescale import components_from_adjacency
from ..adjacency import compute_adjacency
from ..options import AdjacencyMode

if TYPE_CHECKING:
    import geopandas as gpd


@dataclass
class LayoutData:
    """Preprocessed data for layout algorithms.

    Created by prepare_layout_data(). Advanced users can inspect/modify
    before passing to Layout.compute().

    Attributes
    ----------
    positions : NDArray[np.floating]
        Initial positions (centroids), shape (N, 2).
        When tile_count is set, N = Σcounts (expanded); otherwise N = G.
    sizes : NDArray[np.floating]
        Area-equivalent symbol sizes (circle radii), shape (N,).
        Symbol area = pi x size^2. Use area_factor to convert to native
        half-extent for rendering.
    adjacency : NDArray[np.floating]
        Adjacency matrix, shape (N, N).
    bounds : tuple[float, float, float, float]
        Geographic bounds (xmin, ymin, xmax, ymax).
    mean_area : float
        Mean area of input geometries (unit cell area).
    source_gdf : gpd.GeoDataFrame
        Original input geometries.
    valid_mask : NDArray[np.bool_] | None
        Boolean mask indicating which rows had valid (non-null) values.
        None if all values were valid.
    geometry_positions : NDArray[np.floating]
        Original centroid positions at geometry level, shape (G, 2).
        Always G-level (before tile_count expansion).
        When tile_count is absent, equals positions.
    source_indices : NDArray[np.intp] | None
        Mapping from item k (0..N-1) to geometry row in valid source_gdf.
        None when tile_count is absent (implicit identity mapping).
    group_ids : NDArray[np.intp] | None
        Integer group label per item at N-level, shape (N,).
        None when neither group_by nor tile_count is set.
        Already expanded via source_indices when tile_count is used, where
        each geometry is its own group (group k = geometry k), so this is a
        per-symbol source label rather than a user grouping.
    group_ids_G : NDArray[np.intp] | None
        User grouping at G-level (one entry per geometry), shape (G,).
        Set only when ``group_by`` was used (``tile_count`` and ``group_by``
        cannot be combined). None means each geometry is its own group: with
        ``tile_count``, a geometry and its tiles form one block and
        MosaicLayout still enforces per-geometry contiguity. Layouts that
        group geometries rather than symbols (e.g. mosaic) must use this,
        not ``group_ids``.

    """

    positions: NDArray[np.floating]
    sizes: NDArray[np.floating]
    adjacency: NDArray[np.floating]
    bounds: tuple[float, float, float, float]
    mean_area: float
    source_gdf: gpd.GeoDataFrame
    valid_mask: NDArray[np.bool_] | None = None
    geometry_positions: NDArray[np.floating] | None = None
    source_indices: NDArray[np.intp] | None = None
    group_ids: NDArray[np.intp] | None = None
    group_ids_G: NDArray[np.intp] | None = None
    counts_G: NDArray[np.int32] | None = None
    sizes_G: NDArray[np.floating] | None = None
    components: list[list[int]] | None = None
    component_labels: NDArray[np.intp] | None = None


def compute_symbol_sizes(
    values: ArrayLike,
    scale: Literal["sqrt", "linear", "log"] = "sqrt",
    target_max_size: float | None = None,
    size_max_value: float | None = None,
    size_clip: bool = True,
) -> NDArray[np.floating]:
    """Compute symbol sizes from data values.

    Uses abs(values) for size computation. Sign can be used for styling.

    The returned sizes are **area-equivalent radii**, meaning the symbol
    area equals pi x size^2. This ensures symbols of different shapes with
    the same size have the same visual area.

    Parameters
    ----------
    values : array-like
        Data values to scale.
    scale : str
        "sqrt": Area ∝ value (perceptually accurate, default)
        "linear": Size ∝ value
        "log": Logarithmic scaling via log(1 + x)
    target_max_size : float, optional
        Maximum symbol size (area-equivalent radius). If None, normalized
        to 1.0.
    size_max_value : float, optional
        Reference max value for consistent scaling across cartograms.
        When set, sizes are scaled relative to this value instead of
        the data maximum.
    size_clip : bool
        Whether to clip values exceeding size_max_value. Default True.

    Returns
    -------
    np.ndarray
        Symbol sizes (area-equivalent radii), always non-negative.

    Examples
    --------
    >>> sizes = compute_symbol_sizes([1, 4, 9], scale="sqrt", target_max_size=1.0)
    >>> # sqrt scaling: sqrt([1,4,9]) = [1,2,3], normalized to [0.33, 0.67, 1.0]

    """
    values_arr = np.abs(np.asarray(values, dtype=float))

    # Apply scaling
    if scale == "sqrt":
        scaled = np.sqrt(values_arr)
        ref_max = np.sqrt(size_max_value) if size_max_value is not None else None
    elif scale == "linear":
        scaled = values_arr.copy()
        ref_max = size_max_value
    elif scale == "log":
        scaled = np.log1p(values_arr)
        ref_max = np.log1p(size_max_value) if size_max_value is not None else None
    else:
        raise ValueError(f"Unknown scale: {scale}")

    # Normalize to [0, 1] range
    if ref_max is not None:
        if ref_max > 0:
            normalized = scaled / ref_max
            if size_clip:
                normalized = np.clip(normalized, 0, 1)
        else:
            normalized = np.zeros_like(scaled)
    else:
        max_val = scaled.max()
        normalized = scaled / max_val if max_val > 0 else np.zeros_like(scaled)

    # Scale to target max size
    if target_max_size is not None:
        return normalized * target_max_size
    return normalized


def prepare_layout_data(
    gdf: gpd.GeoDataFrame,
    size: str | None = None,
    *,
    tile_count: str | None = None,
    group_by: str | None = None,
    size_scale: Literal["sqrt", "linear", "log"] = "sqrt",
    size_max_value: float | None = None,
    size_clip: bool = True,
    adjacency_mode: AdjacencyMode | Literal["binary", "weighted", "area_weighted"] = "binary",
    distance_tolerance: float | None = None,
    size_normalization: Literal["max", "total"] = "max",
    tile_size_expansion: Literal["shared", "copied"] = "copied",
    collapse_group: float = 0.0,
    pre_scale: bool = False,
) -> LayoutData:
    """Prepare data for layout algorithms.

    This function:

    1. Computes unit cell area as mean geometry area
    2. Extracts centroids as initial positions
    3. Computes symbol sizes (proportional or uniform)
    4. Computes adjacency matrix
    5. Optionally expands rows for tile_count (N = Σcounts)
    6. Optionally encodes group_by as integer labels

    The unit cell is a circle with area equal to the mean geometry area.
    Symbol sizes are area-equivalent radii relative to this unit cell.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with polygon geometries.
    size : str, optional
        Column for proportional sizing. If None, uses uniform sizing
        (all symbols same size as unit cell).
    tile_count : str, optional
        Column with integer tile counts per geometry (N = Σcounts).
        Cannot be combined with group_by.
    group_by : str, optional
        Column whose values define connectivity groups.
        Cannot be combined with tile_count.
    size_scale : str
        Scaling method: "sqrt" (area ∝ value, default), "linear", or "log".
    size_max_value : float, optional
        Reference max value for consistent scaling across cartograms.
    size_clip : bool
        Whether to clip values exceeding size_max_value.
    adjacency_mode : str
        "binary" (adjacent or not), "weighted" (shared border fraction),
        or "area_weighted" (neighbor area fraction).
    distance_tolerance : float, optional
        Buffer for adjacency detection.
    size_normalization : str
        How to normalise symbol sizes after tile expansion, relative to
        original geometry areas:

        - ``"max"`` *(default)*: the largest N-level symbol has area equal to
          the mean geometry area (``π x unit_cell_radius²``).
        - ``"total"``: all N-level sizes are scaled by a single global factor
          so that ``Σ(π x size²) = Σ(geometry_area)``.  With ``"copied"``
          expansion, this accounts for tile counts; with ``"shared"`` the
          result equals what G-level normalisation would give.
    tile_size_expansion : str
        How to expand symbol sizes when ``tile_count`` is set:

        - ``"copied"`` *(default)*: each tile has the same radius as the
          original geometry symbol (``r_g``).  Combined with
          ``size_normalization="total"``, tiles are all equal when
          ``size=None``.
        - ``"shared"``: each tile's radius is scaled by ``1/√K_g`` so that
          the K_g tiles of geometry g together cover the same area as one
          un-expanded symbol.
    collapse_group : float
        Pre-collapse starting positions toward each group's area-weighted
        centroid (0-1). At 1.0 all items in a group start coincident at the
        group centroid, matching ``tile_count`` behavior. Area-weighting
        avoids bias toward regions with many small polygons (e.g. NY
        congressional districts clustered around NYC — a simple mean would
        place the NY centroid too far south). At 0.0 (default) geographic
        centroids are used unchanged. ``geometry_positions`` is always left
        unchanged so that ``origin_weight`` in physics layouts can attract
        symbols back toward geography. Only has effect when ``group_by`` or
        ``tile_count`` is set. Default: 0.0.
    pre_scale : bool
        Uniformly scale each geographically connected component so its area
        matches its share of the data (``tile_count`` if given, else ``size``)
        before any layout step. No effect for single-component inputs.
        This rescales geometry positions and areas, which affects layouts
        that read component geometry (e.g. mosaic). It does not affect
        symbol sizing under ``size_normalization="max"`` (the default) or
        the ``mean_area`` reported on the returned ``LayoutData``, both of
        which are computed from the original, unscaled geometries.
        Default: False.

    Returns
    -------
    LayoutData
        Preprocessed data ready for Layout.compute().

    Examples
    --------
    >>> # Prepare data for layout
    >>> data = prepare_layout_data(gdf, "population")
    >>> # Inspect or modify data
    >>> data.sizes *= 0.9  # Scale down all sizes
    >>> # Pass to layout
    >>> result = layout.compute(data)

    """
    if tile_count is not None and group_by is not None:
        raise ValueError("Cannot set both tile_count and group_by.")

    if not 0 <= collapse_group <= 1:
        raise ValueError("collapse_group must be between 0 and 1")

    if len(gdf) == 0:
        raise ValueError("GeoDataFrame is empty")

    if size is not None and size not in gdf.columns:
        raise ValueError(f"Column '{size}' not found in GeoDataFrame")

    if tile_count is not None and tile_count not in gdf.columns:
        raise ValueError(f"Column '{tile_count}' not found in GeoDataFrame")

    if group_by is not None and group_by not in gdf.columns:
        raise ValueError(f"Column '{group_by}' not found in GeoDataFrame")

    n = len(gdf)

    # 1. Compute unit cell area from mean geometry area
    geometry_areas = np.array([g.area for g in gdf.geometry])
    mean_area = float(np.mean(geometry_areas))
    unit_cell_radius = np.sqrt(mean_area / np.pi)

    # 2. Extract centroids (G-level)
    geometry_positions = np.array([[g.centroid.x, g.centroid.y] for g in gdf.geometry])

    # 3. Compute symbol sizes (G-level)
    valid_mask = np.ones(len(gdf), dtype=bool)
    if size is not None:
        # Proportional sizing
        values = gdf[size].values

        # Handle null values
        null_mask = pd.isnull(values)
        if np.any(null_mask):
            n_null = np.sum(null_mask)
            warnings.warn(f"Skipping {n_null} rows with null values", UserWarning, stacklevel=2)
            valid_mask = ~null_mask
            # Keep only non-null values
            values = values[valid_mask]
            geometry_positions = geometry_positions[valid_mask]
            geometry_areas = geometry_areas[valid_mask]
            gdf = gdf[valid_mask]

        # Handle infinite values: replace with max finite value (or 0 if all infinite)
        values = np.asarray(values, dtype=float)
        inf_mask = np.isinf(values)
        if np.any(inf_mask):
            n_inf = int(np.sum(inf_mask))
            finite_vals = values[~inf_mask]
            replacement = float(np.max(finite_vals)) if len(finite_vals) > 0 else 0.0
            warnings.warn(
                f"Found {n_inf} infinite value(s) in '{size}'; replacing with max finite value ({replacement})",
                UserWarning,
                stacklevel=2,
            )
            values = values.copy()
            values[inf_mask] = replacement

        sizes_G = compute_symbol_sizes(
            values,
            scale=size_scale,
            target_max_size=1.0,
            size_max_value=size_max_value,
            size_clip=size_clip,
        )
    else:
        # Uniform relative sizes: all ones (absolute scale applied after expansion)
        sizes_G = np.ones(n)

    # 4. Compute adjacency (G-level)
    adjacency_G = compute_adjacency(
        gdf,
        mode=AdjacencyMode(adjacency_mode),
        distance_tolerance=distance_tolerance,
    )

    G = len(gdf)

    # 4b. Optional prescaling of connected components
    if pre_scale:
        import shapely

        from ...geo_utils.prescale import prescale_connected_components

        _, components = components_from_adjacency(adjacency_G)
        if len(components) > 1:
            if tile_count is not None:
                ps_values = np.asarray(gdf[tile_count].to_numpy(), dtype=float)
            elif size is not None:
                ps_values = np.asarray(gdf[size].to_numpy(), dtype=float)
            else:
                ps_values = np.ones(G, dtype=float)
            total_area = float(geometry_areas.sum())
            target_density = float(ps_values.sum()) / total_area if total_area > 0 else 1.0
            prescaled = prescale_connected_components(
                list(gdf.geometry),
                ps_values,
                target_density,
                components=components,
            )
            gdf = gdf.copy()
            gdf.geometry = [shapely.make_valid(g) for g in prescaled]
            geometry_areas = np.array([g.area for g in gdf.geometry])
            geometry_positions = np.array([[g.centroid.x, g.centroid.y] for g in gdf.geometry])

    # 5. tile_count expansion
    source_indices = None
    if tile_count is not None:
        counts_raw = np.asarray(gdf[tile_count].values, dtype=float)
        inf_mask = np.isinf(counts_raw)
        if inf_mask.any():
            raise ValueError(
                f"Column '{tile_count}' contains {int(inf_mask.sum())} infinite value(s); "
                "tile counts must be finite integers."
            )
        counts = counts_raw.astype(np.int32)
        if (counts < 0).any():
            raise ValueError(f"Column '{tile_count}' contains negative values.")
        source_indices = np.repeat(np.arange(G, dtype=np.intp), counts)  # (N,)

        positions = geometry_positions[source_indices]  # (N, 2)

        if tile_size_expansion == "shared":
            # Area-preserving: radius_copy = r_i / sqrt(K_i)
            sizes = sizes_G[source_indices] / np.sqrt(counts[source_indices].astype(float))
        else:  # "copied"
            sizes = sizes_G[source_indices]

        # Block-expand adjacency
        expanded = adjacency_G[np.ix_(source_indices, source_indices)].copy()
        same = source_indices[:, None] == source_indices[None, :]
        np.fill_diagonal(same, False)
        expanded[same] = 1.0  # within-block: copies of same geometry are adjacent
        np.fill_diagonal(expanded, 0)
        adjacency = expanded
    else:
        positions = geometry_positions
        sizes = sizes_G
        adjacency = adjacency_G

    # 5b. Apply normalization to N-level sizes
    # When tile_count is set, anchor to tile-level area rather than G-level mean geometry area,
    # so state-level+tile_count and district-level (no tile_count) give comparable symbol sizes.
    if tile_count is not None:
        N_total = int(counts.sum())
        effective_ucr = np.sqrt(float(np.sum(geometry_areas)) / (N_total * np.pi))
    else:
        effective_ucr = unit_cell_radius

    norm_factor = 1.0
    if size_normalization == "total":
        current_total = float(np.pi * np.sum(sizes**2))
        target_total = float(np.sum(geometry_areas))
        if current_total > 0:
            norm_factor = float(np.sqrt(target_total / current_total))
    else:  # "max"
        max_size = float(sizes.max())
        if max_size > 0:
            norm_factor = float(effective_ucr / max_size)
    sizes = sizes * norm_factor

    # 5c. G-level counts/sizes and connected components, for layouts that
    # operate on geometries rather than expanded tiles (e.g. mosaic, grid).
    counts_G_out = None
    sizes_G_out = None
    if tile_count is not None:
        counts_G_out = counts
        sizes_G_out = sizes_G * norm_factor
    component_labels_out, components_out = components_from_adjacency(adjacency_G)

    # 6. group_by / tile_count group encoding — always expanded to N-level
    # When tile_count is set, each geometry is its own group: group k = geometry k.
    # When group_by is set, encode the column as zero-based integers.
    # `group_ids` is always N-level (one entry per symbol); `group_ids_G` is
    # G-level (one entry per geometry) and is only set when the user asked for
    # a grouping via group_by. With tile_count alone it stays None, which means
    # each geometry is its own group (its tiles must stay together), not that
    # grouping is switched off. See the LayoutData docstring.
    group_ids = None
    group_ids_G = None
    if tile_count is not None:
        geometry_group_ids = np.arange(G, dtype=np.intp)  # geometry k is group k
        group_ids = geometry_group_ids[source_indices]  # expand to N-level
    elif group_by is not None:
        labels = gdf[group_by].to_numpy()
        _, group_ids_arr = np.unique(labels, return_inverse=True)
        group_ids = group_ids_arr.astype(np.intp)  # already G-level = N-level (no tile_count)
        group_ids_G = group_ids

    # 7. Collapse positions toward group centroid when requested.
    # For group_by (no tile expansion), use area-weighted centroid so that
    # groups with many small polygons in one corner (e.g. NY congressional
    # districts clustered in NYC) collapse toward the geographic center of
    # the group's territory, matching the state polygon centroid that
    # tile_count would use. For tile_count, all tiles share the same (state)
    # area, so area-weighting reduces to the simple mean.
    if collapse_group > 0 and group_ids is not None:
        positions = positions.copy()  # don't mutate geometry_positions reference
        # N-level weights: for tile_count use per-tile state area; for group_by use district area.
        weights = geometry_areas[source_indices] if source_indices is not None else geometry_areas
        unique_groups = np.unique(group_ids)
        for g in unique_groups:
            mask = group_ids == g
            w = weights[mask]
            w_sum = float(w.sum())
            group_center = (
                (positions[mask] * w[:, None]).sum(axis=0) / w_sum if w_sum > 0 else positions[mask].mean(axis=0)
            )
            positions[mask] = positions[mask] + collapse_group * (group_center - positions[mask])

    return LayoutData(
        positions=positions,
        sizes=sizes,
        adjacency=adjacency,
        bounds=tuple(gdf.total_bounds),
        mean_area=mean_area,
        source_gdf=gdf,
        valid_mask=valid_mask if np.any(~valid_mask) else None,
        geometry_positions=geometry_positions,
        source_indices=source_indices,
        group_ids=group_ids,
        group_ids_G=group_ids_G,
        counts_G=counts_G_out,
        sizes_G=sizes_G_out,
        components=components_out,
        component_labels=component_labels_out,
    )
