"""Golden comparison of the tilings with the output of the tactile-based tilings of carto-flow 2.0.

``tests/data/tilings.npz`` holds, for 19 named tilings, the 15 isohedral presets and the 81 isohedral types, the tiles of
the former implementation that overlap the requested region: center, rotation and flip of each tile, the canonical tile,
and the edge and vertex-only adjacency pairs among these tiles (the polygons of a tile are the canonical tile placed by
its transform). The region is ``(-1, -1, 1, 1)`` with margin 0.1; the named tilings have a fixed ``tile_size``, the
presets and types ``n_tiles=100``. ``tests/data/symbols.npz`` holds the unit polygons of the symbols of the presets and
``tests/data/saved_cartogram.json`` a ``SymbolCartogram`` written by ``save``. The scripts that wrote them and the commit
they ran against are on the report page of the pull request that introduced the files.

The comparison is made tile by tile, matched by center. The former implementations anchored the lattice of the square,
hexagon, triangle and quadrilateral tilings at other places than the region center, so for these the isotile tiling of
the wrapper is moved to the former anchor before the tiles are compared. The order of the tiles is not compared.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import shapely
from scipy.sparse import csr_array
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

from carto_flow.symbol_cartogram import SymbolCartogram, plot_tiling
from carto_flow.symbol_cartogram.tiling import (
    HexagonTiling,
    IsohedralTiling,
    QuadrilateralTiling,
    SquareTiling,
    TriangleTiling,
    _orientation,
)

DATA = Path(__file__).parent / "data"
BOUNDS = (-1.0, -1.0, 1.0, 1.0)
MARGIN = 0.1
REGION = (-1.2, -1.2, 1.2, 1.2)  # BOUNDS with the margin

_ARRAYS = np.load(DATA / "tilings.npz")
META: dict[str, dict] = json.loads(str(_ARRAYS["meta"]))

NAMED = {
    "SquareTiling()": SquareTiling,
    "HexagonTiling(pointy_top=True)": lambda: HexagonTiling(pointy_top=True),
    "HexagonTiling(pointy_top=False)": lambda: HexagonTiling(pointy_top=False),
    "TriangleTiling.equilateral()": TriangleTiling.equilateral,
    "TriangleTiling.right_isosceles()": TriangleTiling.right_isosceles,
    "TriangleTiling.right(2.0)": lambda: TriangleTiling.right(2.0),
    "TriangleTiling.right(0.3)": lambda: TriangleTiling.right(0.3),
    "TriangleTiling.isosceles(30)": lambda: TriangleTiling.isosceles(30.0),
    "TriangleTiling.isosceles(120)": lambda: TriangleTiling.isosceles(120.0),
    "TriangleTiling.from_polygon(clockwise scalene)": lambda: TriangleTiling.from_polygon(
        Polygon([(0, 0), (0.3, 1.0), (1.7, 0.2)])
    ),
    "QuadrilateralTiling.parallelogram(60, 1)": QuadrilateralTiling.parallelogram,
    "QuadrilateralTiling.parallelogram(75, 1.7)": lambda: QuadrilateralTiling.parallelogram(75.0, 1.7),
    "QuadrilateralTiling.rectangle(1)": QuadrilateralTiling.rectangle,
    "QuadrilateralTiling.rectangle(2.5)": lambda: QuadrilateralTiling.rectangle(2.5),
    "QuadrilateralTiling.rhombus(45)": lambda: QuadrilateralTiling.rhombus(45.0),
    "QuadrilateralTiling.trapezoid(0.5, 1)": QuadrilateralTiling.trapezoid,
    "QuadrilateralTiling.trapezoid(0.2, 0.6)": lambda: QuadrilateralTiling.trapezoid(0.2, 0.6),
    "QuadrilateralTiling.from_polygon(convex)": lambda: QuadrilateralTiling.from_polygon(
        Polygon([(0, 0), (1.2, 0.1), (1.0, 0.9), (0.1, 0.7)])
    ),
    "QuadrilateralTiling.from_polygon(non-convex)": lambda: QuadrilateralTiling.from_polygon(
        Polygon([(0, 0), (1.5, 0.0), (0.6, 0.5), (0.0, 1.2)])
    ),
}

# The former geometric adjacency counted vertex-only contacts of curved tiles as edge contacts.
EDGE_PAIRS_THAT_ARE_VERTEX_ONLY = {"preset/wavy_square", "preset/scalloped_quad"}
# The former implementation sampled a spline with 11 control points at 100 points, isotile at 110.
SPLINE_SAMPLING = {"preset/puzzle_piece"}
# Presets with curved edges. The former implementation returned tiles whose curves reach into the region while the
# straight outline does not, isotile selects tiles by the straight outline.
CURVED_PRESETS = {
    f"preset/{name}"
    for name in (
        "scalloped_hexagon",
        "wavy_square",
        "puzzle_piece",
        "arrow_quad",
        "wavy_triangle",
        "interlocking_hexagon",
        "flowers",
        "scalloped_quad",
        "tree",
    )
}

PRESETS = [key for key in META if key.startswith("preset/")]
TYPES = [key for key in META if key.startswith("type/")]


def test_fixture_holds_all_cases():
    assert sorted(k for k in META if k.startswith("named/")) == sorted(f"named/{n}" for n in NAMED)
    assert len(PRESETS) == 15
    assert len(TYPES) == 81


def _old_tiles(key: str) -> dict:
    return {
        name: _ARRAYS[f"{key}/{name}"]
        for name in ("centers", "rotation", "flipped", "canonical", "edge_pairs", "vertex_pairs")
    }


def _place(canonical: np.ndarray, centers: np.ndarray, rotation: np.ndarray, flipped: np.ndarray) -> np.ndarray:
    """(n, k, 2) polygons: the canonical tile flipped about the vertical axis, rotated (degrees) and moved to the centers."""
    angle = np.radians(rotation)
    cos, sin = np.cos(angle)[:, None], np.sin(angle)[:, None]
    x = canonical[None, :, 0] * np.where(flipped, -1.0, 1.0)[:, None]
    y = np.broadcast_to(canonical[None, :, 1], x.shape)
    return np.stack([cos * x - sin * y + centers[:, :1], sin * x + cos * y + centers[:, 1:]], axis=2)


def _vertex_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Largest distance from a vertex of one polygon to the vertex set of the other."""
    d = np.linalg.norm(a[:, None] - b[None], axis=2)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def _dense(pairs: np.ndarray, n: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    m[pairs[:, 0], pairs[:, 1]] = True
    return m | m.T


def _anchored_tile_set(key: str, old: dict):
    """Tiles of the wrapper's isotile tiling, moved to the lattice position of the former implementation.

    The prototile (at the orientation and scale of the wrapper) is placed where the former implementation had an
    unrotated tile, and the lattice is covered over the padded region.
    """
    name = key.removeprefix("named/")
    tiling = NAMED[name]()
    result = tiling.generate(BOUNDS, tile_size=META[key]["tile_size_argument"], margin=MARGIN)
    iso = result.tile_set.tiling
    if name == "HexagonTiling(pointy_top=False)":
        iso = iso.rotated(30.0)  # the former implementation drew pointy-top tiles for both values of the flag
    unrotated = np.flatnonzero((np.abs(old["rotation"]) < 1e-6) & ~old["flipped"])[0]
    frame = iso.frame
    frame[:, 2] += old["centers"][unrotated] - iso.centroid
    return iso.with_frame(frame).cover(REGION)


def _cell_keys(tile_set) -> list[tuple[int, int, int]]:
    return [
        (int(u), int(v), int(a)) for (u, v), a in zip(tile_set.cell.tolist(), tile_set.aspect.tolist(), strict=True)
    ]


def _compare(key: str, tile_set, tol: float, match_tol: float) -> None:
    old = _old_tiles(key)
    size = META[key]["tile_size"]
    n_old = len(old["centers"])
    assert n_old == META[key]["n_old_kept"]

    # The tiles of tile_set and the ring around them. The former implementation also returned tiles with curved edges
    # whose curves reach into the region while their straight outline does not; isotile selects by the straight outline.
    wide = tile_set.tiling.cover(tuple(np.array(REGION) + 2.0 * size * np.array([-1.0, -1.0, 1.0, 1.0])))
    members = set(_cell_keys(tile_set))
    is_member = np.array([k in members for k in _cell_keys(wide)])

    polygons = wide.polygons()
    new_vertices = [np.array(p.exterior.coords[:-1]) for p in polygons]
    new_centers = shapely.get_coordinates(shapely.centroid(np.asarray(polygons, dtype=object)))
    distance, matched = cKDTree(new_centers).query(old["centers"])
    assert (distance < match_tol * size).all(), "tiles of the former implementation without a tile at their center"
    assert len(set(matched.tolist())) == n_old, "two former tiles matched the same tile"
    outside = ~is_member[matched]
    assert not outside.any() or key in CURVED_PRESETS, (
        f"{outside.sum()} tiles of the former implementation not returned"
    )

    old_vertices = _place(old["canonical"], old["centers"], old["rotation"], old["flipped"])
    gap = max(_vertex_gap(old_vertices[i], new_vertices[j]) for i, j in enumerate(matched)) / size
    assert gap < tol, f"largest vertex gap {gap:.2e} of the tile size"

    rotation, flipped = _orientation(wide.transforms[matched])
    assert (flipped == old["flipped"]).all()
    assert np.abs((rotation - old["rotation"] + 180.0) % 360.0 - 180.0).max() < 1e-9

    edge_old = _dense(old["edge_pairs"], n_old)
    vertex_old = _dense(old["vertex_pairs"], n_old)
    edge_new = wide.adjacency("edge")[matched][:, matched].toarray()
    vertex_new = wide.adjacency("vertex-only")[matched][:, matched].toarray()
    if key in EDGE_PAIRS_THAT_ARE_VERTEX_ONLY:
        # contacts that the former geometric adjacency reported as edges are vertex-only contacts of the lattice table
        extra = edge_old & ~edge_new
        assert not (edge_new & ~edge_old).any()
        assert extra.any()
        assert (extra <= vertex_new).all()
        assert (vertex_old <= vertex_new).all()
    else:
        assert (edge_new == edge_old).all()
        assert (vertex_new == vertex_old).all()


@pytest.mark.parametrize("name", NAMED)
def test_named_tilings(name):
    key = f"named/{name}"
    _compare(key, _anchored_tile_set(key, _old_tiles(key)), tol=1e-9, match_tol=1e-6)


def _wrapper_tile_set(key: str):
    name = key.split("/")[1]
    tiling = IsohedralTiling.from_preset(name) if key.startswith("preset/") else IsohedralTiling(int(name))
    return tiling.generate(BOUNDS, n_tiles=META[key]["n_tiles"], margin=MARGIN).tile_set


@pytest.mark.parametrize("key", PRESETS)
def test_presets(key):
    if key in SPLINE_SAMPLING:
        _compare(key, _wrapper_tile_set(key), tol=1.5e-2, match_tol=0.05)
    else:
        _compare(key, _wrapper_tile_set(key), tol=1e-9, match_tol=1e-6 if key != "preset/scalloped_quad" else 0.05)


@pytest.mark.parametrize("key", TYPES)
def test_types(key):
    _compare(key, _wrapper_tile_set(key), tol=1e-9, match_tol=1e-6)


def test_former_implementation_returned_tiles_outside_the_region():
    """The tiles of the former implementation that do not overlap the region are not returned any more."""
    total = sum(m["n_old_total"] for k, m in META.items() if k.startswith("named/"))
    kept = sum(m["n_old_kept"] for k, m in META.items() if k.startswith("named/"))
    assert (total, kept) == (3673, 3264)
    for name in ("SquareTiling()", "TriangleTiling.equilateral()"):
        result = NAMED[name]().generate(BOUNDS, tile_size=0.2, margin=MARGIN)
        assert all(p.intersection(Polygon.from_bounds(*REGION)).area > 0 for p in result.polygons)


# ---------------------------------------------------------------------------
# Symbols and saved results
# ---------------------------------------------------------------------------


_SYMBOLS = np.load(DATA / "symbols.npz")


@pytest.mark.parametrize("name", sorted({k.split("/")[0] for k in _SYMBOLS.files}))
def test_preset_symbols(name):
    bounding_radius, inscribed_radius = _SYMBOLS[f"{name}/radii"]
    symbol = IsohedralTiling.from_preset(name).canonical_symbol()
    tol = 1.5e-2 if f"preset/{name}" in SPLINE_SAMPLING else 1e-9
    assert _vertex_gap(np.array(symbol.unit_polygon().exterior.coords[:-1]), _SYMBOLS[f"{name}/unit_polygon"]) < tol
    assert symbol.bounding_radius == pytest.approx(bounding_radius, abs=tol)
    assert symbol.inscribed_radius == pytest.approx(inscribed_radius, abs=tol)


def test_saved_cartogram_loads_and_plots():
    """A SymbolCartogram that the tactile-based code saved loads, and plots its tiling."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((DATA / "saved_cartogram.json").read_text())
    loaded = SymbolCartogram.load(DATA / "saved_cartogram.json")
    tiling_result = loaded.layout_result.tiling_result
    assert tiling_result.n_tiles == len(data["tiling_result"]["polygons"])
    assert tiling_result.tile_set is None
    assert tiling_result.adjacency.shape == (tiling_result.n_tiles, tiling_result.n_tiles)
    assert loaded.layout_result.assignments is not None
    assert plot_tiling(loaded) is not None
    assert loaded.plot() is not None
    plt.close("all")


def test_saved_cartogram_round_trips_with_the_new_code(tmp_path):
    loaded = SymbolCartogram.load(DATA / "saved_cartogram.json")
    loaded.save(tmp_path / "again.json")
    again = SymbolCartogram.load(tmp_path / "again.json")
    first, second = loaded.layout_result.tiling_result, again.layout_result.tiling_result
    assert second.n_tiles == first.n_tiles
    assert second.tile_size == pytest.approx(first.tile_size)
    assert isinstance(second.adjacency, csr_array)
