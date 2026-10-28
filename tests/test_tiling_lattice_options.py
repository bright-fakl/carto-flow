"""Rotation and offset of the tile lattice: ``Tiling.rotated``, ``Tiling.with_offset`` and the layout options.

``rotation`` turns the lattice about the center of the region it covers, ``offset`` shifts it by a fraction of its two
translation vectors. Both are shared by ``GridBasedLayout`` and ``MosaicLayout``.
"""

from __future__ import annotations

import dataclasses

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import box

from carto_flow.symbol_cartogram import (
    GridBasedLayout,
    GridBasedLayoutOptions,
    HexagonTiling,
    IsohedralTiling,
    MosaicLayout,
    MosaicLayoutOptions,
    QuadrilateralTiling,
    SquareTiling,
    TriangleTiling,
)
from carto_flow.symbol_cartogram.layouts import prepare_layout_data
from carto_flow.symbol_cartogram.layouts.mosaic._calibration import calibrate_tiling

BOUNDS = (0.0, 0.0, 10.0, 6.0)
TILINGS = {
    "square": SquareTiling,
    "hexagon": HexagonTiling,
    "triangle": TriangleTiling.equilateral,
    "quadrilateral": lambda: QuadrilateralTiling.parallelogram(70.0, 1.4),
    "scalloped_hexagon": lambda: IsohedralTiling.from_preset("scalloped_hexagon"),
}


def _centers(result) -> np.ndarray:
    """Tile centers sorted lexicographically, so that two results compare independent of tile order."""
    c = np.array([t.center for t in result.transforms])
    return c[np.lexsort((np.round(c[:, 1], 6), np.round(c[:, 0], 6)))]


def _grid_gdf(cols: int, rows: int, x0: float = 0.0) -> gpd.GeoDataFrame:
    geoms = [box(x0 + c, r, x0 + c + 1, r + 1) for r in range(rows) for c in range(cols)]
    return gpd.GeoDataFrame({"tiles": [3 + (i % 3) for i in range(len(geoms))]}, geometry=geoms)


@pytest.fixture(params=sorted(TILINGS))
def tiling(request):
    return TILINGS[request.param]()


class TestRotated:
    def test_default_is_unchanged(self, tiling):
        plain = tiling.generate(BOUNDS, tile_size=1.0)
        same = tiling.rotated(0.0).generate(BOUNDS, tile_size=1.0)
        assert np.array_equal(_centers(plain), _centers(same))

    def test_returns_a_copy(self, tiling):
        before = _centers(tiling.generate(BOUNDS, tile_size=1.0))
        rotated = tiling.rotated(30.0)
        assert rotated is not tiling
        assert np.array_equal(before, _centers(tiling.generate(BOUNDS, tile_size=1.0)))

    def test_rotations_add_up(self, tiling):
        a = tiling.rotated(10.0).rotated(20.0).generate(BOUNDS, tile_size=1.0)
        b = tiling.rotated(30.0).generate(BOUNDS, tile_size=1.0)
        assert a.n_tiles == b.n_tiles
        assert np.allclose(_centers(a), _centers(b), atol=1e-9)

    @pytest.mark.parametrize("angle", [30.0, 45.0, -17.0, 90.0])
    def test_rotated_lattice_covers_the_region(self, tiling, angle):
        result = tiling.rotated(angle).generate(BOUNDS, tile_size=1.0, margin=0.0)
        region = box(*BOUNDS)
        union = shapely.union_all(result.polygons)
        assert region.difference(union).area < 1e-9 * region.area

    def test_rotation_is_counter_clockwise_about_the_region_center(self):
        result = SquareTiling().rotated(30.0).generate(BOUNDS, tile_size=1.0)
        # The tiles are squares turned by 30 degrees: an edge direction is 30 degrees from the x axis.
        coords = np.array(result.polygons[0].exterior.coords)
        edges = np.diff(coords, axis=0)
        angles = np.degrees(np.arctan2(edges[:, 1], edges[:, 0])) % 90.0
        assert np.allclose(angles, 30.0, atol=1e-6)
        # The lattice is anchored at the region center: it is a corner of four squares, whatever the angle.
        corners = np.concatenate([np.array(p.exterior.coords) for p in result.polygons])
        assert np.min(np.linalg.norm(corners - np.array([5.0, 3.0]), axis=1)) < 1e-9

    def test_tile_transforms_include_the_rotation(self):
        result = SquareTiling().rotated(30.0).generate(BOUNDS, tile_size=1.0)
        assert np.allclose([t.rotation % 90.0 for t in result.transforms], 30.0, atol=1e-6)

    @pytest.mark.parametrize("angle", [0.0, 30.0, -75.0])
    def test_transforms_place_the_canonical_tile_on_the_polygons(self, tiling, angle):
        """canonical tile, flipped, rotated by the transform and moved to its center, is the tile."""
        from shapely.affinity import rotate, scale, translate

        result = tiling.rotated(angle).with_offset((0.2, 0.3)).generate(BOUNDS, tile_size=1.0)
        for poly, tr in zip(result.polygons, result.transforms, strict=True):
            placed = result.canonical_tile
            if tr.flipped:
                placed = scale(placed, xfact=-1, yfact=1, origin=(0, 0))
            placed = translate(rotate(placed, tr.rotation, origin=(0, 0)), *tr.center)
            assert poly.hausdorff_distance(placed) < 1e-6 * result.tile_size

    def test_tile_size_is_unchanged(self, tiling):
        plain = tiling.generate(BOUNDS, tile_size=1.0)
        rotated = tiling.rotated(33.0).generate(BOUNDS, tile_size=1.0)
        assert rotated.tile_size == pytest.approx(plain.tile_size, rel=1e-12)

    @pytest.mark.parametrize("angle", [np.nan, np.inf, -np.inf, "x", None])
    def test_rejects_a_non_finite_angle(self, tiling, angle):
        with pytest.raises(ValueError, match="angle"):
            tiling.rotated(angle)

    def test_about_the_region_center_not_the_origin(self, tiling):
        """Data far from the origin: the tiles relative to the region center are those of a centered region."""
        far = 8.0e6
        near = tiling.rotated(30.0).generate(BOUNDS, tile_size=1.0)
        moved = (BOUNDS[0] + far, BOUNDS[1], BOUNDS[2] + far, BOUNDS[3])
        shifted = tiling.rotated(30.0).generate(moved, tile_size=1.0)
        assert shifted.n_tiles == near.n_tiles
        delta = _centers(shifted) - np.array([far, 0.0])
        assert np.allclose(delta, _centers(near), atol=1e-6)

    def test_tiling_result_rotate_is_about_the_origin(self):
        """The utility keeps its definition: it turns the tiles about (0, 0)."""
        result = SquareTiling().generate((100.0, 100.0, 110.0, 106.0), tile_size=1.0)
        turned = result.rotate(90.0)
        c, t = result.transforms[0].center, turned.transforms[0].center
        assert t == pytest.approx((-c[1], c[0]), abs=1e-9)


class TestWithOffset:
    def test_default_is_identical(self, tiling):
        plain = tiling.generate(BOUNDS, tile_size=1.0)
        same = tiling.with_offset((0.0, 0.0)).generate(BOUNDS, tile_size=1.0)
        assert plain.n_tiles == same.n_tiles
        assert all(a.equals_exact(b, 0) for a, b in zip(plain.polygons, same.polygons, strict=True))

    def test_returns_a_copy(self, tiling):
        shifted = tiling.with_offset((0.3, 0.6))
        assert shifted is not tiling
        assert tiling.generate(BOUNDS, tile_size=1.0).n_tiles == tiling.generate(BOUNDS, tile_size=1.0).n_tiles

    @pytest.mark.parametrize("step", [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (-1.0, 2.0)])
    def test_one_period_gives_the_same_tile_set(self, tiling, step):
        base = (0.3, 0.7)
        a = tiling.with_offset(base).generate(BOUNDS, tile_size=1.0)
        b = tiling.with_offset((base[0] + step[0], base[1] + step[1])).generate(BOUNDS, tile_size=1.0)
        assert a.n_tiles == b.n_tiles
        assert np.allclose(_centers(a), _centers(b), atol=1e-9)

    @pytest.mark.parametrize("step", [(1.0, 0.0), (0.0, 1.0)])
    def test_one_period_from_the_default(self, tiling, step):
        a = tiling.generate(BOUNDS, tile_size=1.0)
        b = tiling.with_offset(step).generate(BOUNDS, tile_size=1.0)
        assert a.n_tiles == b.n_tiles
        assert np.allclose(_centers(a), _centers(b), atol=1e-9)

    def test_square_offset_moves_the_lattice_by_fractions_of_the_tile_side(self):
        a = SquareTiling().generate(BOUNDS, tile_size=1.0, margin=0.0)
        b = SquareTiling().with_offset((0.25, 0.5)).generate(BOUNDS, tile_size=1.0, margin=0.0)
        ca, cb = _centers(a), _centers(b)
        # Lattice positions modulo one tile side differ by the offset.
        phase = (cb[0] - ca[0] - np.array([0.25, 0.5]) + 0.5) % 1.0 - 0.5
        assert np.allclose(phase, 0.0, atol=1e-9)
        assert np.allclose((cb - ca[0] - np.array([0.25, 0.5]) + 0.5) % 1.0 - 0.5, 0.0, atol=1e-9)

    def test_offset_follows_the_translation_vectors_of_the_placed_tiling(self, tiling):
        offset = (0.35, 0.2)
        a = tiling.rotated(20.0).generate(BOUNDS, tile_size=1.0)
        b = tiling.rotated(20.0).with_offset(offset).generate(BOUNDS, tile_size=1.0)
        placed = a.tile_set.tiling
        shift = offset[0] * np.asarray(placed.translation_u) + offset[1] * np.asarray(placed.translation_v)
        # Every tile of b that also lies well inside the region is a tile of a moved by the shift, up to whole periods.
        moved = b.tile_set.tiling.centroid - a.tile_set.tiling.centroid
        assert np.allclose(moved, shift, atol=1e-9)

    @pytest.mark.parametrize("kind", ["n_tiles", "tile_size"])
    def test_generate_is_fill_region_with_the_offset(self, tiling, kind):
        """The offset is applied by one fill_region call: same tiles, in the same order, as isotile returns."""
        value = {"n_tiles": 60, "tile_size": 1.3}[kind]
        offset = (0.3, 0.8)
        result = tiling.with_offset(offset).generate(BOUNDS, margin=0.1, **{kind: value})
        argument = value * tiling._side_per_size if kind == "tile_size" else value
        direct = tiling._iso.fill_region(BOUNDS, margin=0.1, offset=offset, **{kind: argument})
        assert np.array_equal(result.tile_set.cell, direct.cell)
        assert np.array_equal(result.tile_set.aspect, direct.aspect)
        assert np.array_equal(result.tile_set.transforms, direct.transforms)

    def test_the_offset_changes_the_tiles(self):
        a = SquareTiling().generate(BOUNDS, tile_size=1.0)
        b = SquareTiling().with_offset((0.5, 0.5)).generate(BOUNDS, tile_size=1.0)
        assert not np.allclose(_centers(a)[:3], _centers(b)[:3])

    def test_the_shifted_lattice_covers_the_region(self, tiling):
        result = tiling.with_offset((0.4, 0.8)).generate(BOUNDS, tile_size=1.0, margin=0.0)
        region = box(*BOUNDS)
        assert region.difference(shapely.union_all(result.polygons)).area < 1e-9 * region.area

    def test_offset_is_about_the_region_center_far_from_the_origin(self, tiling):
        far = 8.0e6
        near = tiling.with_offset((0.3, 0.6)).generate(BOUNDS, tile_size=1.0)
        moved = (BOUNDS[0] + far, BOUNDS[1], BOUNDS[2] + far, BOUNDS[3])
        shifted = tiling.with_offset((0.3, 0.6)).generate(moved, tile_size=1.0)
        assert shifted.n_tiles == near.n_tiles
        assert np.allclose(_centers(shifted) - np.array([far, 0.0]), _centers(near), atol=1e-6)

    def test_offset_and_rotation_commute(self, tiling):
        a = tiling.rotated(25.0).with_offset((0.2, 0.9)).generate(BOUNDS, tile_size=1.0)
        b = tiling.with_offset((0.2, 0.9)).rotated(25.0).generate(BOUNDS, tile_size=1.0)
        assert a.n_tiles == b.n_tiles
        assert np.allclose(_centers(a), _centers(b), atol=1e-9)

    @pytest.mark.parametrize(
        "offset",
        [(np.nan, 0.0), (0.0, np.inf), (-np.inf, 0.0), (0.5,), (0.1, 0.2, 0.3), 0.5, "ab", None, ("a", 0.0)],
    )
    def test_rejects_a_bad_offset(self, tiling, offset):
        with pytest.raises(ValueError, match="offset"):
            tiling.with_offset(offset)

    def test_accepts_a_list_and_an_array(self, tiling):
        a = tiling.with_offset((0.2, 0.4)).generate(BOUNDS, tile_size=1.0)
        for offset in ([0.2, 0.4], np.array([0.2, 0.4])):
            b = tiling.with_offset(offset).generate(BOUNDS, tile_size=1.0)
            assert np.array_equal(_centers(a), _centers(b))


@pytest.mark.parametrize("options_cls", [GridBasedLayoutOptions, MosaicLayoutOptions])
class TestOptions:
    def test_defaults(self, options_cls):
        options = options_cls()
        assert options.rotation == 0.0
        assert options.offset == (0.0, 0.0)

    def test_round_trip(self, options_cls):
        options = options_cls(rotation=30.0, offset=(0.25, 0.75))
        copy = dataclasses.replace(options)
        assert copy == options
        assert dataclasses.asdict(copy)["rotation"] == 30.0
        assert dataclasses.asdict(copy)["offset"] == (0.25, 0.75)
        changed = dataclasses.replace(options, rotation=10.0)
        assert changed.offset == (0.25, 0.75)

    @pytest.mark.parametrize("rotation", [np.nan, np.inf, -np.inf, "x", None])
    def test_rejects_a_bad_rotation(self, options_cls, rotation):
        with pytest.raises(ValueError, match="rotation"):
            options_cls(rotation=rotation).validate()

    @pytest.mark.parametrize("offset", [(np.nan, 0.0), (0.0, np.inf), (0.5,), (0.1, 0.2, 0.3), 0.5, "ab", None])
    def test_rejects_a_bad_offset(self, options_cls, offset):
        with pytest.raises(ValueError, match="offset"):
            options_cls(offset=offset).validate()

    def test_accepts_other_pairs(self, options_cls):
        options_cls(rotation=-45, offset=[0.1, 0.2]).validate()
        options_cls(offset=np.array([1.0, 2.0])).validate()


class TestLayouts:
    def test_layouts_validate_at_construction(self):
        for cls in (GridBasedLayout, MosaicLayout):
            with pytest.raises(ValueError, match="rotation"):
                cls(rotation=float("nan"))
            with pytest.raises(ValueError, match="offset"):
                cls(offset=(0.0, 1.0, 2.0))

    def test_unknown_names_still_fail(self):
        with pytest.raises(TypeError, match="Unknown option"):
            GridBasedLayout(rotate=30.0)

    def test_grid_default_options_equal_explicit_defaults(self):
        data = prepare_layout_data(_grid_gdf(4, 3))
        a = GridBasedLayout(tiling="hexagon").compute(data, show_progress=False)
        b = GridBasedLayout(tiling="hexagon", rotation=0.0, offset=(0.0, 0.0)).compute(data, show_progress=False)
        assert np.array_equal(a.assignments, b.assignments)
        assert a.tiling_result.n_tiles == b.tiling_result.n_tiles

    def test_grid_rotation_covers_the_region_and_turns_the_tiles(self):
        data = prepare_layout_data(_grid_gdf(4, 3))
        result = GridBasedLayout(tiling="square", rotation=30.0).compute(data, show_progress=False)
        tiles = result.tiling_result
        assert np.allclose([t.rotation % 90.0 for t in tiles.transforms], 30.0, atol=1e-6)
        region = box(*data.bounds)
        assert region.difference(shapely.union_all(tiles.polygons)).area < 1e-9 * region.area

    @pytest.mark.parametrize("tiling_name", ["square", "hexagon"])
    @pytest.mark.parametrize(
        "options", [{"rotation": 30.0}, {"offset": (0.5, 0.5)}, {"rotation": 20.0, "offset": (0.3, 0.1)}]
    )
    def test_grid_layout_does_not_depend_on_where_the_data_are(self, tiling_name, options):
        """Rotation and offset are relative to the data: moving them 8e6 m east gives the same displacement."""
        far = 8.0e6
        near = _grid_gdf(5, 4)
        shifted = _grid_gdf(5, 4, x0=far)
        layout = GridBasedLayout(tiling=tiling_name, **options)
        a = layout.compute(prepare_layout_data(near), show_progress=False)
        b = layout.compute(prepare_layout_data(shifted), show_progress=False)
        shift = np.array([far, 0.0])
        assert np.allclose(a.positions + shift, b.positions, atol=1e-6)

    @pytest.mark.parametrize("options", [{"rotation": 30.0}, {"offset": (0.5, 0.5)}])
    def test_mosaic_uses_the_options_in_calibration_and_final_build(self, options):
        gdf = _grid_gdf(4, 3)
        data = prepare_layout_data(gdf, tile_count="tiles")
        result = MosaicLayout(tiling="square", morph=False, **options).compute(data, show_progress=False)
        counts = np.bincount(result.source_indices, minlength=len(gdf))
        assert np.array_equal(counts, gdf["tiles"].to_numpy())
        tiles = result.tiling_result
        # The lattice in the result is that of the oriented tiling: the same tile centers as a direct build of the
        # oriented tiling with the same size, on the expanded bounds of the final build.
        oriented = SquareTiling()
        if "rotation" in options:
            oriented = oriented.rotated(options["rotation"])
            assert np.allclose([t.rotation % 90.0 for t in tiles.transforms], options["rotation"], atol=1e-6)
        if "offset" in options:
            oriented = oriented.with_offset(options["offset"])
        buf = tiles.tile_size * 1.5
        b = data.bounds
        expanded = (b[0] - buf, b[1] - buf, b[2] + buf, b[3] + buf)
        direct = oriented.generate(expanded, tile_size=tiles.tile_size)
        assert np.allclose(_centers(direct), _centers(tiles), atol=1e-9)

    def test_mosaic_calibration_trial_and_final_builds_share_the_lattice(self):
        gdf = _grid_gdf(4, 3)
        union = shapely.union_all(list(gdf.geometry))
        oriented = SquareTiling().rotated(30.0).with_offset((0.4, 0.1))
        setup = calibrate_tiling(oriented, union.bounds, union, target_count=int(gdf["tiles"].sum()))
        assert abs(len(setup.core_set) - int(gdf["tiles"].sum())) <= 1
        assert np.allclose([t.rotation % 90.0 for t in setup.tiling_result.transforms], 30.0, atol=1e-6)

    def test_mosaic_rotation_far_from_the_origin(self):
        far = 8.0e6
        a = MosaicLayout(tiling="hexagon", morph=False, rotation=30.0).compute(
            prepare_layout_data(_grid_gdf(4, 3), tile_count="tiles"), show_progress=False
        )
        b = MosaicLayout(tiling="hexagon", morph=False, rotation=30.0).compute(
            prepare_layout_data(_grid_gdf(4, 3, x0=far), tile_count="tiles"), show_progress=False
        )
        assert np.array_equal(np.sort(np.bincount(a.source_indices)), np.sort(np.bincount(b.source_indices)))
        assert a.tiling_result.n_tiles == b.tiling_result.n_tiles
