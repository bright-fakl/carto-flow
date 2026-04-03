"""
High-performance geometry processing utilities.

This module provides Numba-optimized functions for polygon area computation,
geometry unpacking/reconstruction, and coordinate manipulation. It enables
efficient batch processing of geometries by separating coordinate storage
from geometry objects.

Classes
-------
GeometryCoordinateInfo
    Container for flattened coordinates with reconstruction metadata.

Functions
---------
unpack_geometries
    Convert list of geometries to flattened coordinate array.
unpack_geometry
    Convert single geometry to coordinates and metadata.
reconstruct_geometries
    Rebuild geometries from GeometryCoordinateInfo.
reconstruct_geometry
    Rebuild single geometry from coordinates and metadata.
compute_polygon_area_numba
    Fast shoelace formula for single polygon ring.
compute_complex_polygon_areas_numba
    Parallel area computation for polygons with holes.

Examples
--------
>>> from carto_flow.geo_utils import unpack_geometries, reconstruct_geometries
>>> from shapely.geometry import Polygon
>>>
>>> # Process multiple polygons efficiently
>>> polygons = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
>>> coord_info = unpack_geometries(polygons, precompute_ring_info=True)
>>>
>>> # Transform coordinates in-place
>>> coord_info.coords += displacement_vector
>>> coord_info.invalidate_cache()
>>>
>>> # Compute areas efficiently without reconstruction
>>> areas = coord_info.compute_areas(use_parallel=True)
>>>
>>> # Reconstruct geometries only when needed
>>> final_polygons = reconstruct_geometries(coord_info)
"""

from typing import Any

import numpy as np
import shapely
from numba import jit, prange
from shapely.geometry import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry

# Module-level exports - Public API
__all__ = [
    # Classes
    "GeometryCoordinateInfo",
    "compute_complex_polygon_areas_numba",
    # Area computation functions
    "compute_polygon_area_numba",
    # Reconstruction functions
    "reconstruct_geometries",
    "reconstruct_geometry",
    # Unpacking functions
    "unpack_geometries",
    "unpack_geometry",
]


# ============================================================================
# AREA COMPUTATION - SHOELACE FORMULA
# ============================================================================


@jit(nopython=True, fastmath=True, cache=True)
def compute_polygon_area_numba(coords: np.ndarray, start: int = 0, size: int = -1) -> float:
    """
    Compute area of a polygon ring using the shoelace formula.

    This function works for both:
    - A standalone coordinate array: compute_polygon_area_numba(coords)
    - A slice within a larger array: compute_polygon_area_numba(all_coords, start, size)

    Parameters
    ----------
    coords : np.ndarray
        Coordinate array, shape (n_points, 2)
    start : int
        Starting index (default 0 for standalone arrays)
    size : int
        Number of vertices to use (default -1 means use all from start)

    Returns
    -------
    area : float
        Absolute area of the polygon ring

    Notes
    -----
    Uses the shoelace formula: A = 0.5 * |Σ(x_i * y_{i+1} - x_{i+1} * y_i)|
    """
    if size == -1:
        size = len(coords) - start

    area = 0.0
    for i in range(size - 1):
        idx = start + i
        area += coords[idx, 0] * coords[idx + 1, 1]
        area -= coords[idx + 1, 0] * coords[idx, 1]

    # Close the polygon
    last_idx = start + size - 1
    first_idx = start
    area += coords[last_idx, 0] * coords[first_idx, 1]
    area -= coords[first_idx, 0] * coords[last_idx, 1]

    return abs(area) * 0.5


@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def compute_complex_polygon_areas_numba(coords: np.ndarray, polygon_ring_info: np.ndarray) -> np.ndarray:
    """
    Compute areas for polygons with holes and MultiPolygons in parallel.

    Parameters
    ----------
    coords : np.ndarray
        All coordinates stacked, shape (total_points, 2)
    polygon_ring_info : np.ndarray
        Array describing ring structure, shape (n_rings, 4)
        Each row: [polygon_id, ring_type, start_idx, size]
        - polygon_id: which polygon this ring belongs to (0, 1, 2, ...)
        - ring_type: 1 for exterior, -1 for hole
        - start_idx: starting index in coords array
        - size: number of vertices in this ring

    Returns
    -------
    areas : np.ndarray
        Area of each polygon (indexed by polygon_id)
    """
    n_rings = polygon_ring_info.shape[0]

    # Find maximum polygon_id to allocate result array
    max_poly_id = 0
    for i in range(n_rings):
        if polygon_ring_info[i, 0] > max_poly_id:
            max_poly_id = int(polygon_ring_info[i, 0])

    areas = np.zeros(max_poly_id + 1)

    # Compute each ring area in parallel
    ring_areas = np.zeros(n_rings)
    for i in prange(n_rings):
        start = int(polygon_ring_info[i, 2])
        size = int(polygon_ring_info[i, 3])
        ring_areas[i] = compute_polygon_area_numba(coords, start, size)

    # Accumulate areas by polygon (must be sequential due to shared writes)
    for i in range(n_rings):
        poly_id = int(polygon_ring_info[i, 0])
        ring_type = polygon_ring_info[i, 1]
        areas[poly_id] += ring_type * ring_areas[i]  # +1 for exterior, -1 for holes

    return areas


# ============================================================================
# GEOMETRY UNPACKING
# ============================================================================


def unpack_geometry(geom: BaseGeometry) -> tuple[np.ndarray, dict]:
    """
    Unpack a single geometry into coordinates and reconstruction metadata.

    Parameters
    ----------
    geom : BaseGeometry
        Any shapely geometry object

    Returns
    -------
    coords : np.ndarray
        Array of coordinates, shape (n_points, 2)
    metadata : dict
        Information needed to reconstruct the geometry
    """
    geom_type = geom.geom_type

    if geom_type == "Point":
        coords = np.array([[geom.x, geom.y]])
        metadata = {"type": "Point", "size": 1}

    elif geom_type == "MultiPoint":
        coords = np.array([[p.x, p.y] for p in geom.geoms])
        metadata = {"type": "MultiPoint", "size": len(coords)}

    elif geom_type == "LineString":
        coords = np.array(geom.coords)
        metadata = {"type": "LineString", "size": len(coords)}

    elif geom_type == "MultiLineString":
        all_coords = []
        sizes = []
        for line in geom.geoms:
            line_coords = np.array(line.coords)
            all_coords.append(line_coords)
            sizes.append(len(line_coords))
        coords = np.vstack(all_coords) if all_coords else np.empty((0, 2))
        metadata = {"type": "MultiLineString", "sizes": sizes}

    elif geom_type == "Polygon":
        # Handle exterior and holes
        exterior_coords = np.array(geom.exterior.coords)
        hole_coords = []
        hole_sizes = []

        for hole in geom.interiors:
            hole_c = np.array(hole.coords)
            hole_coords.append(hole_c)
            hole_sizes.append(len(hole_c))

        coords = np.vstack([exterior_coords, *hole_coords]) if hole_coords else exterior_coords

        metadata = {"type": "Polygon", "exterior_size": len(exterior_coords), "hole_sizes": hole_sizes}

    elif geom_type == "MultiPolygon":
        all_coords = []
        polygon_metadata = []

        for poly in geom.geoms:
            poly_coords, poly_meta = unpack_geometry(poly)
            all_coords.append(poly_coords)
            polygon_metadata.append(poly_meta)

        coords = np.vstack(all_coords) if all_coords else np.empty((0, 2))
        metadata = {"type": "MultiPolygon", "polygon_metadata": polygon_metadata}

    elif geom_type == "GeometryCollection":
        all_coords = []
        geom_metadata = []

        for g in geom.geoms:
            g_coords, g_meta = unpack_geometry(g)
            all_coords.append(g_coords)
            geom_metadata.append(g_meta)

        coords = np.vstack(all_coords) if all_coords else np.empty((0, 2))
        metadata = {"type": "GeometryCollection", "geom_metadata": geom_metadata}

    elif geom_type == "LinearRing":
        coords = np.array(geom.coords)
        metadata = {"type": "LinearRing", "size": len(coords)}

    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")

    return coords, metadata


# ============================================================================
# GEOMETRY RECONSTRUCTION
# ============================================================================


def reconstruct_geometry(coords: np.ndarray, metadata: dict, start_idx: int = 0) -> tuple[BaseGeometry, int]:
    """
    Reconstruct a single geometry from coordinates and metadata.

    Parameters
    ----------
    coords : np.ndarray
        Flattened coordinate array
    metadata : dict
        Reconstruction information
    start_idx : int
        Starting index in coords array

    Returns
    -------
    geom : BaseGeometry
        Reconstructed geometry
    end_idx : int
        Next index after this geometry's coordinates
    """
    geom_type = metadata["type"]

    if geom_type == "Point":
        geom = Point(coords[start_idx])
        return geom, start_idx + 1

    elif geom_type == "MultiPoint":
        size = metadata["size"]
        points = [Point(c) for c in coords[start_idx : start_idx + size]]
        return MultiPoint(points), start_idx + size

    elif geom_type == "LineString":
        size = metadata["size"]
        geom = LineString(coords[start_idx : start_idx + size])
        return geom, start_idx + size

    elif geom_type == "MultiLineString":
        sizes = metadata["sizes"]
        lines = []
        idx = start_idx
        for size in sizes:
            lines.append(LineString(coords[idx : idx + size]))
            idx += size
        return MultiLineString(lines), idx

    elif geom_type == "Polygon":
        exterior_size = metadata["exterior_size"]
        hole_sizes = metadata["hole_sizes"]

        # Reconstruct exterior
        exterior = coords[start_idx : start_idx + exterior_size]
        idx = start_idx + exterior_size

        # Reconstruct holes
        holes = []
        for hole_size in hole_sizes:
            hole = coords[idx : idx + hole_size]
            holes.append(hole)
            idx += hole_size

        geom = Polygon(exterior, holes=holes if holes else None)
        return geom, idx

    elif geom_type == "MultiPolygon":
        polygon_metadata = metadata["polygon_metadata"]
        polygons = []
        idx = start_idx

        for poly_meta in polygon_metadata:
            poly, idx = reconstruct_geometry(coords, poly_meta, idx)
            polygons.append(poly)

        return MultiPolygon(polygons), idx

    elif geom_type == "GeometryCollection":
        geom_metadata = metadata["geom_metadata"]
        geoms = []
        idx = start_idx

        for g_meta in geom_metadata:
            g, idx = reconstruct_geometry(coords, g_meta, idx)
            geoms.append(g)

        return GeometryCollection(geoms), idx

    elif geom_type == "LinearRing":
        size = metadata["size"]
        geom = LinearRing(coords[start_idx : start_idx + size])
        return geom, start_idx + size

    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")


def reconstruct_geometries(coord_info: "GeometryCoordinateInfo") -> list[BaseGeometry]:
    """
    Reconstruct geometries from coordinate info.

    Parameters
    ----------
    coord_info : GeometryCoordinateInfo
        Object containing coordinates and metadata

    Returns
    -------
    geometries : List[BaseGeometry]
        List of reconstructed geometry objects
    """
    geometries = []
    idx = 0

    for metadata in coord_info.metadata:
        geom, idx = reconstruct_geometry(coord_info.coords, metadata, idx)
        geometries.append(geom)

    return geometries


# ============================================================================
# COORDINATE INFO CLASS
# ============================================================================


class GeometryCoordinateInfo:
    """
    Container for flattened coordinates with reconstruction metadata.

    This class enables efficient separation of coordinate transformation from
    geometry manipulation, allowing batch processing of coordinates and
    fast area computation without reconstruction overhead.

    Parameters
    ----------
    coords : np.ndarray
        Flattened array of all coordinates, shape (n_points, 2).
    metadata : list[dict]
        Reconstruction information for each geometry.
    ring_info : np.ndarray, optional
        Precomputed ring information for parallel area computation.
        Shape (n_rings, 4) with [polygon_id, ring_type, start_idx, size].
    polygon_indices : np.ndarray, optional
        Array mapping polygon_id (in ring_info) to geometry index.

    Attributes
    ----------
    coords : np.ndarray
        Flattened array of all coordinates, shape (n_points, 2).
    metadata : list[dict]
        Reconstruction information for each geometry.
    ring_info : np.ndarray
        Ring information array for area computation (lazily computed).
    polygon_indices : np.ndarray
        Mapping from polygon_id to geometry index (lazily computed).

    Notes
    -----
    **Key Features**

    - Lazy computation of ring_info for parallel area calculation
    - Direct array-based area computation (no list building)
    - Optional caching of computed areas
    - Statistics and analysis methods

    **Typical Workflow**

    1. Create via ``unpack_geometries()``
    2. Transform ``coords`` array in-place
    3. Call ``invalidate_cache()`` after transformations
    4. Call ``compute_areas()`` for efficient area calculation
    5. Call ``reconstruct_geometries()`` only when needed

    Examples
    --------
    >>> from carto_flow.geo_utils import unpack_geometries, reconstruct_geometries
    >>> from shapely.geometry import Polygon
    >>>
    >>> polygons = [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]
    >>> coord_info = unpack_geometries(polygons, precompute_ring_info=True)
    >>>
    >>> # Transform coordinates
    >>> coord_info.coords = transform(coord_info.coords)
    >>> coord_info.invalidate_cache()
    >>>
    >>> # Compute areas efficiently without reconstruction
    >>> areas = coord_info.compute_areas(use_parallel=True)
    >>>
    >>> # Reconstruct only when needed
    >>> final_polygons = reconstruct_geometries(coord_info)
    """

    def __init__(
        self,
        coords: np.ndarray,
        metadata: list[dict],
        ring_info: np.ndarray | None = None,
        polygon_indices: np.ndarray | None = None,
        ragged_geometry_type: int | None = None,
        ragged_offsets: tuple | None = None,
    ):
        self.coords = coords
        self.metadata = metadata
        self._ring_info = ring_info
        self._polygon_indices = polygon_indices
        self._cached_areas: np.ndarray | None = None
        self._n_geometries = len(metadata)
        self.ragged_geometry_type = ragged_geometry_type
        self.ragged_offsets = ragged_offsets

    @property
    def ring_info(self) -> np.ndarray:
        """
        Get ring info array, computing it lazily if needed.

        Returns
        -------
        ring_info : np.ndarray
            Array of shape (n_rings, 4) with [polygon_id, ring_type, start_idx, size]
        """
        if self._ring_info is None:
            self._ring_info, self._polygon_indices = self._build_ring_info()
        return self._ring_info

    @property
    def polygon_indices(self) -> np.ndarray:
        """
        Get mapping from polygon_id to geometry index.

        Returns
        -------
        polygon_indices : np.ndarray
            Array where polygon_indices[polygon_id] = geometry_index
        """
        if self._polygon_indices is None:
            self._ring_info, self._polygon_indices = self._build_ring_info()
        return self._polygon_indices

    def _build_ring_info(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Build the ring info array and polygon index mapping from metadata.

        Returns
        -------
        ring_info : np.ndarray
            Array of shape (n_rings, 4) with [polygon_id, ring_type, start_idx, size]
        polygon_indices : np.ndarray
            Mapping from polygon_id to geometry index
        """
        ring_info_list = []
        polygon_index_list = []
        polygon_id = 0
        coord_idx = 0

        for geom_idx, metadata in enumerate(self.metadata):
            geom_type = metadata["type"]

            if geom_type == "Polygon":
                # Record which geometry index this polygon_id corresponds to
                polygon_index_list.append(geom_idx)

                # Exterior ring
                exterior_size = metadata["exterior_size"]
                ring_info_list.append([polygon_id, 1, coord_idx, exterior_size])
                coord_idx += exterior_size

                # Hole rings
                hole_sizes = metadata["hole_sizes"]
                for hole_size in hole_sizes:
                    ring_info_list.append([polygon_id, -1, coord_idx, hole_size])
                    coord_idx += hole_size

                polygon_id += 1

            elif geom_type == "MultiPolygon":
                # Record which geometry index this polygon_id corresponds to
                polygon_index_list.append(geom_idx)

                # Each component polygon gets the same polygon_id
                polygon_metadata = metadata["polygon_metadata"]

                for poly_meta in polygon_metadata:
                    # Exterior
                    exterior_size = poly_meta["exterior_size"]
                    ring_info_list.append([polygon_id, 1, coord_idx, exterior_size])
                    coord_idx += exterior_size

                    # Holes
                    hole_sizes = poly_meta["hole_sizes"]
                    for hole_size in hole_sizes:
                        ring_info_list.append([polygon_id, -1, coord_idx, hole_size])
                        coord_idx += hole_size

                polygon_id += 1

            else:
                # Skip non-polygon types, but advance coord_idx
                if geom_type == "Point":
                    coord_idx += 1
                elif geom_type == "MultiPoint" or geom_type == "LineString":
                    coord_idx += metadata["size"]
                elif geom_type == "MultiLineString":
                    for size in metadata["sizes"]:
                        coord_idx += size
                elif geom_type == "LinearRing":
                    coord_idx += metadata["size"]

        if ring_info_list:
            ring_info = np.array(ring_info_list, dtype=np.int32)
            polygon_indices = np.array(polygon_index_list, dtype=np.int32)
        else:
            ring_info = np.empty((0, 4), dtype=np.int32)
            polygon_indices = np.empty(0, dtype=np.int32)

        return ring_info, polygon_indices

    def compute_areas(self, use_parallel: bool = True, use_cache: bool = True) -> np.ndarray:
        """
        Compute areas for all geometries.

        Parameters
        ----------
        use_parallel : bool, default=True
            If True, use parallel Numba computation (faster for many polygons).
        use_cache : bool, default=True
            If True and areas are cached, return cached values.
            Cache is invalidated by ``invalidate_cache()``.

        Returns
        -------
        np.ndarray
            Area for each geometry, shape (n_geometries,).
            Non-polygon types get area = 0.0.

        Examples
        --------
        >>> areas = coord_info.compute_areas(use_parallel=True)
        >>> print(f"Total area: {areas.sum():.2f}")
        """
        if use_cache and self._cached_areas is not None:
            return self._cached_areas

        if use_parallel and len(self.ring_info) > 0:
            areas = self._compute_areas_parallel()
        else:
            areas = self._compute_areas_sequential()

        if use_cache:
            self._cached_areas = areas

        return areas

    def _compute_areas_parallel(self) -> np.ndarray:
        """
        Compute areas using parallel Numba with direct array assignment.

        This is the most efficient approach: compute polygon areas in parallel,
        then assign directly to the output array using polygon_indices.
        """
        # Initialize output array (all zeros for non-polygons)
        areas = np.zeros(self._n_geometries, dtype=np.float64)

        # Compute polygon areas in parallel
        polygon_areas = compute_complex_polygon_areas_numba(self.coords, self.ring_info)

        # Direct array assignment using precomputed indices
        areas[self.polygon_indices] = polygon_areas

        return areas

    def _compute_areas_sequential(self) -> np.ndarray:
        """
        Compute areas sequentially (fallback for small datasets or no polygons).
        """
        # Initialize output array
        areas = np.zeros(self._n_geometries, dtype=np.float64)

        coord_idx = 0

        for geom_idx, metadata in enumerate(self.metadata):
            geom_type = metadata["type"]

            if geom_type == "Polygon":
                # Exterior area
                exterior_size = metadata["exterior_size"]
                area = compute_polygon_area_numba(self.coords, coord_idx, exterior_size)
                coord_idx += exterior_size

                # Subtract hole areas
                hole_sizes = metadata["hole_sizes"]
                for hole_size in hole_sizes:
                    area -= compute_polygon_area_numba(self.coords, coord_idx, hole_size)
                    coord_idx += hole_size

                areas[geom_idx] = area

            elif geom_type == "MultiPolygon":
                total_area = 0.0
                polygon_metadata = metadata["polygon_metadata"]

                for poly_meta in polygon_metadata:
                    # Exterior
                    exterior_size = poly_meta["exterior_size"]
                    poly_area = compute_polygon_area_numba(self.coords, coord_idx, exterior_size)
                    coord_idx += exterior_size

                    # Holes
                    hole_sizes = poly_meta["hole_sizes"]
                    for hole_size in hole_sizes:
                        poly_area -= compute_polygon_area_numba(self.coords, coord_idx, hole_size)
                        coord_idx += hole_size

                    total_area += poly_area

                areas[geom_idx] = total_area

            else:
                # Skip non-polygon coordinates (area already 0.0)
                if geom_type == "Point":
                    coord_idx += 1
                elif geom_type == "MultiPoint" or geom_type == "LineString":
                    coord_idx += metadata["size"]
                elif geom_type == "MultiLineString":
                    for size in metadata["sizes"]:
                        coord_idx += size
                elif geom_type == "LinearRing":
                    coord_idx += metadata["size"]

        return areas

    def invalidate_cache(self) -> None:
        """
        Invalidate cached computed values.

        Call this after transforming coordinates to ensure fresh area
        calculations on the next ``compute_areas()`` call.

        Examples
        --------
        >>> coord_info.coords += displacement
        >>> coord_info.invalidate_cache()
        >>> new_areas = coord_info.compute_areas()
        """
        self._cached_areas = None

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the geometry collection.

        Returns
        -------
        dict
            Dictionary containing:

            - ``n_geometries``: Total number of geometries
            - ``n_coordinates``: Total number of coordinate points
            - ``n_polygons``: Number of polygon/multipolygon geometries
            - ``n_rings``: Total number of rings (for area computation)
            - ``geometry_types``: Count of each geometry type

        Examples
        --------
        >>> stats = coord_info.get_statistics()
        >>> print(f"Processing {stats['n_polygons']} polygons")
        """
        geom_types: dict[str, int] = {}
        n_polygons = 0

        for metadata in self.metadata:
            geom_type = metadata["type"]
            geom_types[geom_type] = geom_types.get(geom_type, 0) + 1
            if geom_type in ["Polygon", "MultiPolygon"]:
                n_polygons += 1

        return {
            "n_geometries": len(self.metadata),
            "n_coordinates": len(self.coords),
            "n_polygons": n_polygons,
            "n_rings": len(self.ring_info) if self._ring_info is not None else 0,
            "geometry_types": geom_types,
        }


# ============================================================================
# UNPACKING FUNCTION
# ============================================================================


def unpack_geometries(geometries: list[BaseGeometry], precompute_ring_info: bool = True) -> GeometryCoordinateInfo:
    """
    Unpack multiple geometries into a single coordinate array with metadata.

    This function extracts all coordinates from a list of geometries into
    a single flat array, enabling vectorized operations. Metadata is stored
    to allow perfect reconstruction of the original geometry structure.

    Parameters
    ----------
    geometries : List[BaseGeometry]
        List of shapely geometry objects (any type)
    precompute_ring_info : bool
        If True, precompute ring_info array for fast parallel area computation.
        Recommended if you'll be computing areas frequently.

    Returns
    -------
    GeometryCoordinateInfo
        Object containing flattened coordinates, reconstruction metadata,
        and optionally precomputed ring info for area calculations.
    """
    geom_array = np.asarray(geometries, dtype=object)
    geometry_type, flat_coords, offsets = shapely.to_ragged_array(geom_array)

    all_metadata = []
    for geom in geometries:
        _, metadata = unpack_geometry(geom)
        all_metadata.append(metadata)

    coord_info = GeometryCoordinateInfo(
        flat_coords,
        all_metadata,
        ragged_geometry_type=int(geometry_type),
        ragged_offsets=offsets,
    )

    if precompute_ring_info:
        _ = coord_info.ring_info

    return coord_info
