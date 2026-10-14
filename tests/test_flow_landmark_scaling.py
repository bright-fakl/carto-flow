"""Area scaling of the landmarks and uniform landmarks after the morph."""

import warnings

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import box

import carto_flow.flow_cartogram as flow
import carto_flow.proportional_cartogram as prop
from carto_flow import data


def _quad(center, half=0.02):
    cx, cy = center
    return [(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half), (cx - half, cy + half)]


def _scaled_quads(cartogram_landmarks):
    pts = shapely.get_coordinates(np.asarray(cartogram_landmarks, dtype=object)).reshape(2, 4, 2)
    return np.array([shapely.Polygon(q).area for q in pts]) / (0.04**2)


def _two_squares(grid):
    gdf = gpd.GeoDataFrame({"v": [1.0, 4.0]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)])
    quads = gpd.GeoSeries(shapely.points(_quad((0.5, 0.5)) + _quad((1.5, 0.5))))
    options = flow.MorphOptions(show_progress=False, grid=grid, mean_tol=0.005, max_tol=0.01)
    return flow.create_flow_cartogram(gdf, "v", landmarks=quads, options=options)


def test_no_landmarks_no_area_scaling():
    gdf = gpd.GeoDataFrame({"v": [1.0, 4.0]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)])
    cartogram = flow.create_flow_cartogram(gdf, "v", options=flow.MorphOptions(show_progress=False, grid=64))
    assert cartogram.landmark_area_scaling is None


@pytest.mark.parametrize("grid", [128, flow.LadderSpec(min_resolution=32, levels=2)])
def test_area_scaling_is_the_area_ratio_of_a_small_square_of_landmarks(grid):
    cartogram = _two_squares(grid)
    scaling = np.asarray(cartogram.landmark_area_scaling).reshape(2, 4).mean(axis=1)
    np.testing.assert_allclose(scaling, _scaled_quads(cartogram.get_landmarks()), rtol=0.15)
    assert scaling[0] < 1 < scaling[1]


@pytest.fixture(scope="module")
def two_squares():
    return _two_squares(64), None


def test_uniform_needs_the_source_data():
    gdf = gpd.GeoDataFrame({"v": [1.0, 4.0]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)])
    centers = shapely.points([[0.5, 0.5], [1.5, 0.5]])
    from carto_flow.flow_cartogram import morph_geometries

    cartogram = morph_geometries(
        list(gdf.geometry),
        gdf["v"].to_numpy(),
        landmarks=centers,
        options=flow.MorphOptions(show_progress=False, grid=64),
    )
    with pytest.raises(ValueError, match="source data"):
        cartogram.get_landmarks(uniform=True)


def test_uniform_is_for_point_landmarks_only():
    gdf = gpd.GeoDataFrame({"v": [1.0, 4.0]}, geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)])
    lines = gpd.GeoSeries([shapely.LineString([(0.2, 0.2), (0.4, 0.4)]), shapely.LineString([(1.2, 0.2), (1.4, 0.4)])])
    cartogram = flow.create_flow_cartogram(
        gdf, "v", landmarks=lines, options=flow.MorphOptions(show_progress=False, grid=64)
    )
    with pytest.raises(ValueError, match="point landmarks"):
        cartogram.get_landmarks(uniform=True)


@pytest.fixture(scope="module")
def states_with_dots():
    return _states_with_dots(None)


@pytest.fixture(scope="module")
def states_with_frames():
    return _states_with_dots(4)


def _states_with_dots(snapshot_every):
    g = data.load_us_census(population=True, poverty=True)
    col = "Below Poverty Level"
    dots = prop.generate_dot_density(
        g, columns=[col], n_dots=round(g[col].max() / 25000), normalization="maximum", seed=0
    )
    owner = np.array([list(g.index).index(o) for o in dots["original_index"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cartogram = flow.create_flow_cartogram(
            g,
            "Population",
            landmarks=gpd.GeoSeries(list(dots.geometry), crs=g.crs),
            options=flow.MorphOptions(area_scale=1e-6, show_progress=False, snapshot_every=snapshot_every),
        )
    return cartogram, owner


def _band_z(geoms, points, owner, frac=0.06):
    obs = exp = var = 0.0
    for r in range(len(geoms)):
        sel = owner == r
        if sel.sum() < 8:
            continue
        poly = geoms[r].buffer(0)
        inner = poly.buffer(-frac * np.sqrt(poly.area))
        p = 1 - inner.area / poly.area
        obs += (~shapely.contains(inner, points[sel])).sum()
        exp += sel.sum() * p
        var += sel.sum() * p * (1 - p)
    return (obs - exp) / np.sqrt(var)


def test_uniform_landmarks_remove_the_border_excess(states_with_dots):
    cartogram, owner = states_with_dots
    geoms = list(cartogram.get_geometry())
    carried = np.asarray(cartogram.get_landmarks(), dtype=object)
    uniform = np.asarray(cartogram.get_landmarks(uniform=True), dtype=object)
    assert _band_z(geoms, carried, owner) > 2.5
    assert _band_z(geoms, uniform, owner) < 1.5


def test_uniform_landmarks_move_a_little_and_stay_in_their_region(states_with_dots):
    cartogram, owner = states_with_dots
    geoms = list(cartogram.get_geometry())
    carried = shapely.get_coordinates(np.asarray(cartogram.get_landmarks(), dtype=object))
    uniform_geoms = np.asarray(cartogram.get_landmarks(uniform=True), dtype=object)
    uniform = shapely.get_coordinates(uniform_geoms)
    size = np.array([np.sqrt(geoms[o].area) for o in owner])
    moved = np.linalg.norm(uniform - carried, axis=1) / size
    assert np.median(moved) < 0.02
    assert np.percentile(moved, 95) < 0.1
    inside = np.array([shapely.contains(geoms[o].buffer(0), p) for o, p in zip(owner, uniform_geoms, strict=False)])
    assert inside.mean() > 0.99


def test_uniform_landmarks_are_reproducible(states_with_dots):
    cartogram, _ = states_with_dots
    a = shapely.get_coordinates(np.asarray(cartogram.get_landmarks(uniform=True), dtype=object))
    b = shapely.get_coordinates(np.asarray(cartogram.get_landmarks(uniform=True), dtype=object))
    np.testing.assert_array_equal(a, b)


def test_every_snapshot_has_its_area_scaling(states_with_frames):
    cartogram, _ = states_with_frames
    snapshots = cartogram.snapshots.snapshots
    assert len(snapshots) > 10
    assert all(
        s.landmark_area_scaling is not None and len(s.landmark_area_scaling) == len(s.landmarks) for s in snapshots
    )
    assert np.all(snapshots[-1].landmark_area_scaling == cartogram.landmark_area_scaling)
    # at the start of the morph the flow has hardly scaled any area
    assert 0.9 < np.mean(snapshots[0].landmark_area_scaling) < 1.1


def test_uniform_landmarks_of_an_intermediate_iteration(states_with_frames):
    cartogram, owner = states_with_frames
    middle = cartogram.snapshots.snapshots[len(cartogram.snapshots.snapshots) // 2].iteration
    carried = np.asarray(cartogram.get_landmarks(iteration=middle), dtype=object)
    uniform = np.asarray(cartogram.get_landmarks(iteration=middle, uniform=True), dtype=object)
    geoms = list(cartogram.get_geometry(middle))
    assert _band_z(geoms, uniform, owner) < _band_z(geoms, carried, owner)


def test_uniform_landmarks_change_smoothly_over_the_frames(states_with_frames):
    """The correction vector of a dot grows from frame to frame without jumps (a sampled distribution made it jitter)."""
    cartogram, owner = states_with_frames
    snapshots = cartogram.snapshots.snapshots
    deltas = []
    for snapshot in snapshots:
        carried = shapely.get_coordinates(np.asarray(snapshot.landmarks, dtype=object))
        uniform = shapely.get_coordinates(
            np.asarray(cartogram.get_landmarks(iteration=snapshot.iteration, uniform=True), dtype=object)
        )
        deltas.append(uniform - carried)
    deltas = np.array(deltas)
    width = np.array([np.sqrt(shapely.area(snapshots[-1].geometry[o])) for o in owner])
    second = np.linalg.norm(deltas[2:] - 2 * deltas[1:-1] + deltas[:-2], axis=2) / width[None, :]
    assert np.percentile(second, 95) < 0.003
    # the first frame is hardly corrected
    assert np.median(np.linalg.norm(deltas[0], axis=1) / width) < 0.002


def test_distribution_of_the_distance_to_the_border_of_a_square():
    from carto_flow.flow_cartogram.landmarks import _SGRID, distance_distributions

    (F,) = distance_distributions([box(0, 0, 10, 10)])
    exact = 1 - (1 - 2 * _SGRID) ** 2  # the part of a square within s x width of the border
    assert np.max(np.abs(F - exact)) < 0.015


def test_no_area_scaling_with_prescale_components():
    """The pre-scaling moves the geometries but not the landmarks, so the area scaling of the landmarks is not defined."""
    gdf = gpd.GeoDataFrame({"v": [1.0, 4.0]}, geometry=[box(0, 0, 1, 1), box(3, 0, 4, 1)])
    centers = gpd.GeoSeries(shapely.points([[0.5, 0.5], [3.5, 0.5]]))
    options = flow.MorphOptions(show_progress=False, grid=64, prescale_components=True)
    cartogram = flow.create_flow_cartogram(gdf, "v", landmarks=centers, options=options)
    assert cartogram.landmark_area_scaling is None
    with pytest.raises(ValueError, match="prescale_components"):
        cartogram.get_landmarks(uniform=True)
