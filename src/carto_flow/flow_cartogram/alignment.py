"""
Common vertex sets for the outputs of flow cartograms of the same input.

A flow morph with ``validity="windows"`` inserts vertices into the segments of the geometries, so morphs of the same
input with different values (for example different years) return geometries with different vertices and cannot be
interpolated linearly. :func:`align_vertices` gives the outputs of several morphs one vertex set.

Functions
---------
align_vertices
    Re-sample the outputs of several morphs of the same input on the union of their vertices.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import shapely

from .cartogram import Cartogram
from .workflow import CartogramWorkflow

__all__ = ["align_vertices"]

_POLYGON = 3  # shapely.get_type_id of a Polygon


def _chain_and_output(source: Any) -> tuple[list[Cartogram], int, np.ndarray]:
    """Cartograms to lift through, vertex count of the input, and output geometries of a source."""
    if isinstance(source, CartogramWorkflow):
        output = np.asarray(source[-1].get_geometry(), dtype=object)
        n_input = len(shapely.get_coordinates(np.asarray(source.original.get_geometry(), dtype=object)))
        return list(source)[1:], n_input, output
    if isinstance(source, Cartogram):
        output = np.asarray(source.get_geometry(), dtype=object)
        n_input = (
            source.vertex_lineage.n_input if source.vertex_lineage is not None else len(shapely.get_coordinates(output))
        )
        return [source], n_input, output
    raise TypeError(f"Sources must be Cartogram or CartogramWorkflow objects, got {type(source).__name__}")


def _input_positions(chain: list[Cartogram], n_input: int, n_output: int, index: int) -> np.ndarray:
    """Position along the input of every vertex of the output: the index of the input vertex, fractional between."""
    position = np.arange(n_input, dtype=float)
    for cartogram in chain:
        try:
            position = cartogram.lift_coordinates(position)
        except ValueError as error:
            raise ValueError(
                f"Source {index} is not a morph of the input shared by the other sources, or of its whole chain "
                f"(pass the workflow of a ladder, not one of its levels): {error}"
            ) from error
    if len(position) != n_output:
        raise ValueError(f"Source {index}: the lineage gives {len(position)} vertices, the output has {n_output}")
    if len(position) > 1 and not np.all(np.diff(position) > 0):
        raise ValueError(f"Source {index}: the positions of the vertices along the input rings are not increasing")
    return position


def align_vertices(sources: Any) -> list[np.ndarray]:
    """Give the outputs of several morphs of the same input one vertex set.

    Flow cartograms with ``validity="windows"`` insert vertices, different ones for different values, so their
    outputs cannot be interpolated linearly. Every vertex of an output has a position along the rings of the input:
    the input vertices are at the integers of their index in the coordinate array and an inserted vertex is
    at the position that its vertex lineage gives it between its parents. The result has, in every ring, the vertices
    at the union of the positions of all sources. Each output is re-sampled at these positions by linear
    interpolation along its own segments, so every output keeps its exact shape (only vertices on its own segments
    are added) and its validity.

    Parameters
    ----------
    sources : sequence of Cartogram or CartogramWorkflow
        Two or more morphs of the same input geometries, for example the same map with the values of different
        years. A :class:`CartogramWorkflow` is a chain of morphs (a ladder, or refinements) that start from the
        output of the one before, and its returned state is the last result; pass the workflow, not one of its
        levels. A :class:`Cartogram` is taken to be a morph of the input itself. Sources without a vertex lineage
        (``validity="none"``) can be mixed with others.

    Returns
    -------
    list of np.ndarray
        One object array of geometries per source, in the order of the sources. They have equal numbers of
        geometries, parts, rings and vertices; vertex ``k`` of all of them lies at the same position along the input.
        Use them as keyframes of :func:`~carto_flow.flow_cartogram.animation.animate_geometry_keyframes`.

    Raises
    ------
    ValueError
        If there are fewer than two sources, or the sources do not come from the same input (different numbers of
        geometries, parts, rings or input vertices per ring).
    TypeError
        If a source is neither a Cartogram nor a CartogramWorkflow.

    Examples
    --------
    >>> options = MorphOptions(validity="windows")
    >>> a = create_flow_cartogram(gdf, "pop_2010", options=options)
    >>> b = create_flow_cartogram(gdf, "pop_2020", options=options)
    >>> frames = [gdf.assign(geometry=geometry) for geometry in align_vertices([a, b])]
    >>> anim = animate_geometry_keyframes(frames, interpolation="linear")
    """
    sources = list(sources)
    if len(sources) < 2:
        raise ValueError("align_vertices needs at least two sources")

    positions, outputs, ragged, signatures = [], [], [], []
    for index, source in enumerate(sources):
        chain, n_input, output = _chain_and_output(source)
        position = _input_positions(chain, n_input, len(shapely.get_coordinates(output)), index)
        geometry_type, coords, offsets = shapely.to_ragged_array(output)
        rings = np.asarray(offsets[0])
        start, end = position[rings[:-1]], position[rings[1:] - 1]
        if not (np.all(start == np.round(start)) and np.all(end == np.round(end))):
            raise ValueError(f"Source {index}: the rings do not start and end at vertices of the input")
        positions.append(position)
        outputs.append(output)
        ragged.append((geometry_type, coords, offsets))
        signatures.append((
            start,
            end,
            shapely.get_type_id(output),
            geometry_type,
            [np.asarray(o) for o in offsets[1:]],
        ))

    first = signatures[0]
    for index, signature in enumerate(signatures[1:], start=1):
        same = (
            all(np.array_equal(a, b) for a, b in zip(signature[:4], first[:4], strict=True))
            and len(signature[4]) == len(first[4])
            and all(np.array_equal(a, b) for a, b in zip(signature[4], first[4], strict=True))
        )
        if not same:
            raise ValueError(
                f"Source {index} does not come from the same input as source 0 "
                "(different numbers of geometries, parts, rings or input vertices per ring)"
            )

    union = np.unique(np.concatenate(positions))
    ring_of = np.searchsorted(first[0], union, side="right") - 1
    ring_counts = np.bincount(ring_of, minlength=len(first[0]))

    aligned = []
    for output, (geometry_type, coords, offsets), position in zip(outputs, ragged, positions, strict=True):
        resampled = np.column_stack([
            np.interp(union, position, coords[:, 0]),
            np.interp(union, position, coords[:, 1]),
        ])
        ring_offsets = np.concatenate([[0], np.cumsum(ring_counts)]).astype(np.asarray(offsets[0]).dtype)
        geometries = np.asarray(
            shapely.from_ragged_array(geometry_type, resampled, (ring_offsets, *offsets[1:])), dtype=object
        )
        # a mix of polygons and multipolygons is built as multipolygons: give a polygon back to the single-part ones
        for i in np.flatnonzero(
            (shapely.get_type_id(output) == _POLYGON) & (shapely.get_type_id(geometries) != _POLYGON)
        ):
            geometries[i] = geometries[i].geoms[0]
        aligned.append(geometries)
    return aligned
