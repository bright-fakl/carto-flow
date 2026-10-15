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
    with pytest.raises(ValueError, match="probe"):
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


def _two_far_squares():
    gdf = gpd.GeoDataFrame({"v": [1.0, 4.0]}, geometry=[box(0, 0, 1, 1), box(3, 0, 4, 1)])
    rng = np.random.default_rng(0)
    points = np.vstack([
        np.column_stack([rng.uniform(0, 1, 300), rng.uniform(0, 1, 300)]),
        np.column_stack([rng.uniform(3, 4, 300), rng.uniform(0, 1, 300)]),
    ])
    return gdf, points, np.repeat([0, 1], 300)


def test_prescale_components_moves_the_landmarks_with_their_component():
    gdf, points, owner = _two_far_squares()
    options = flow.MorphOptions(
        show_progress=False, grid=128, prescale_components=True, mean_tol=0.005, max_tol=0.01, landmark_probes=50
    )
    cartogram = flow.create_flow_cartogram(gdf, "v", landmarks=gpd.GeoSeries(shapely.points(points)), options=options)
    geoms = list(cartogram.get_geometry())
    carried = shapely.get_coordinates(np.asarray(cartogram.get_landmarks(), dtype=object))
    for r in (0, 1):
        assert shapely.contains(geoms[r], shapely.points(carried[owner == r])).mean() > 0.99
    # the area scaling includes the pre-scaling: it is the area ratio of the region
    ratios = np.array([g.area for g in geoms]) / gdf.area.to_numpy()
    scaling = np.asarray(cartogram.landmark_area_scaling)
    np.testing.assert_allclose([scaling[owner == r].mean() for r in (0, 1)], ratios, rtol=0.05)
    uniform = shapely.get_coordinates(np.asarray(cartogram.get_landmarks(uniform=True), dtype=object))
    assert np.linalg.norm(uniform - carried, axis=1).max() < 0.2


def test_prescale_coords_moves_the_displacement_coordinates_with_their_component():
    gdf, points, owner = _two_far_squares()
    options = flow.MorphOptions(
        show_progress=False, grid=128, prescale_components=True, prescale_coords=True, mean_tol=0.005, max_tol=0.01
    )
    cartogram = flow.create_flow_cartogram(gdf, "v", displacement_coords=points, options=options)
    geoms = list(cartogram.get_geometry())
    carried = np.asarray(cartogram.get_coords())
    for r in (0, 1):
        assert shapely.contains(geoms[r], shapely.points(carried[owner == r])).mean() > 0.99
    assert cartogram.best.coords_area_scaling is not None


def test_displacement_coordinates_are_not_prescaled_by_default_and_have_no_area_scaling():
    gdf, points, _ = _two_far_squares()
    options = flow.MorphOptions(show_progress=False, grid=128, prescale_components=True, mean_tol=0.005, max_tol=0.01)
    cartogram = flow.create_flow_cartogram(gdf, "v", displacement_coords=points, options=options)
    assert cartogram.best.coords_area_scaling is None
    with pytest.raises(ValueError, match="prescale_coords"):
        cartogram.get_coords(uniform=True)


def test_scale_points_scales_like_the_geometries():
    from carto_flow.geo_utils.prescale import prescale_connected_components, prescale_scalings, scale_points

    geoms = [box(0, 0, 2, 2), box(10, 0, 12, 2)]
    values = np.array([1.0, 3.0])
    scalings = prescale_scalings(geoms, values, target_density=0.5)  # target areas 2 and 6
    scaled = prescale_connected_components(geoms, values, 0.5)
    points = np.array([[0.0, 0.0], [2.0, 2.0], [10.0, 0.0], [5.0, 5.0]])
    moved, log_scaling = scale_points(points, geoms, scalings)
    # the corners of the first square follow the corners of the scaled geometry
    np.testing.assert_allclose(moved[:2], [scaled[0].bounds[:2], scaled[0].bounds[2:]])
    np.testing.assert_allclose(moved[2], scaled[1].bounds[:2])
    np.testing.assert_allclose(moved[3], points[3])  # outside every geometry
    np.testing.assert_allclose(log_scaling, [np.log(scaled[0].area / 4)] * 2 + [np.log(scaled[1].area / 4), 0.0])


PROBES = 50


def _states_with_probes(probes, **kwargs):
    g = data.load_us_census(population=True, poverty=True)
    options = flow.MorphOptions(area_scale=1e-6, show_progress=False, snapshot_every=8, landmark_probes=probes)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return flow.create_flow_cartogram(g, "Population", options=options, **kwargs), g


@pytest.fixture(scope="module")
def dots_with_probes():
    g = data.load_us_census(population=True, poverty=True)
    col = "Below Poverty Level"
    dots = prop.generate_dot_density(
        g, columns=[col], n_dots=round(g[col].max() / 25000), normalization="maximum", seed=0
    )
    owner = np.array([list(g.index).index(o) for o in dots["original_index"]])
    landmarks = gpd.GeoSeries(list(dots.geometry), crs=g.crs)
    with_probes, _ = _states_with_probes(PROBES, landmarks=landmarks)
    without, _ = _states_with_probes(0, landmarks=landmarks)
    return with_probes, without, owner, g


def test_probes_are_stored_per_snapshot_and_do_not_change_the_landmarks(dots_with_probes):
    with_probes, without, _, g = dots_with_probes
    assert without.best.probe_positions is None
    snapshots = with_probes.snapshots.snapshots
    assert len(snapshots) > 5
    for snapshot in snapshots:
        assert snapshot.probe_positions.shape == (PROBES * len(g), 2)
        assert snapshot.probe_area_scaling.shape == (PROBES * len(g),)
    a = shapely.get_coordinates(np.asarray(with_probes.get_landmarks(), dtype=object))
    b = shapely.get_coordinates(np.asarray(without.get_landmarks(), dtype=object))
    np.testing.assert_allclose(a, b, rtol=1e-9)


def test_dots_corrected_with_probes_have_no_border_excess(dots_with_probes):
    cartogram, _, owner, _ = dots_with_probes
    geoms = list(cartogram.get_geometry())
    carried = np.asarray(cartogram.get_landmarks(), dtype=object)
    uniform = np.asarray(cartogram.get_landmarks(uniform=True), dtype=object)
    assert _band_z(geoms, carried, owner) > 2.5
    assert _band_z(geoms, uniform, owner) < 1.5


def test_uniform_line_landmarks_keep_their_vertices():
    g = data.load_us_census(population=True)
    points = g.geometry.representative_point()
    lines = gpd.GeoSeries(
        [shapely.LineString([(p.x - 20000, p.y), (p.x, p.y), (p.x + 20000, p.y + 10000)]) for p in points], crs=g.crs
    )
    cartogram, _ = _states_with_probes(PROBES, landmarks=lines)
    carried = cartogram.get_landmarks()
    uniform = cartogram.get_landmarks(uniform=True)
    assert all(u.geom_type == "LineString" and len(u.coords) == 3 for u in uniform)
    moved = np.array([
        np.linalg.norm(np.array(u.coords) - np.array(c.coords), axis=1).max()
        for u, c in zip(uniform, carried, strict=False)
    ])
    width = np.sqrt(shapely.area(cartogram.get_geometry()))
    assert np.max(moved / width) < 0.2
    assert (moved > 0).any()


def test_uniform_displacement_coordinates():
    g = data.load_us_census(population=True)
    rng = np.random.default_rng(0)
    b = g.total_bounds
    coords = np.column_stack([rng.uniform(b[0], b[2], 500), rng.uniform(b[1], b[3], 500)])
    cartogram, _ = _states_with_probes(PROBES, displacement_coords=coords)
    carried = np.asarray(cartogram.get_coords())
    uniform = np.asarray(cartogram.get_coords(uniform=True))
    assert uniform.shape == carried.shape == (500, 2)
    width = np.sqrt(np.mean(shapely.area(cartogram.get_geometry())))
    moved = np.linalg.norm(uniform - carried, axis=1) / width
    assert np.percentile(moved, 99) < 0.2
    assert (moved > 0).any()
    without, _ = _states_with_probes(0, displacement_coords=coords)
    with pytest.raises(ValueError, match="landmark_probes"):
        without.get_coords(uniform=True)


def test_probes_through_a_ladder():
    g = data.load_us_census(population=True, poverty=True)
    dots = prop.generate_dot_density(g, columns=["Below Poverty Level"], n_dots=3, normalization="maximum", seed=1)
    options = flow.MorphOptions(
        area_scale=1e-6,
        show_progress=False,
        landmark_probes=20,
        grid=flow.LadderSpec(min_resolution=64, levels=2),
        mean_tol=0.01,
        max_tol=0.03,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cartogram = flow.create_flow_cartogram(
            g, "Population", landmarks=gpd.GeoSeries(list(dots.geometry), crs=g.crs), options=options
        )
    scaling = cartogram.best.probe_area_scaling
    assert scaling.shape == (20 * len(g),)
    ratios = shapely.area(cartogram.get_geometry()) / g.area.to_numpy()
    region = np.repeat(np.arange(len(g)), 20)
    # the probes of a region scale, on average, with the region
    mean_log = np.array([np.log(scaling[region == r]).mean() for r in range(len(g))])
    assert np.corrcoef(mean_log, np.log(ratios))[0, 1] > 0.9


def test_landmark_probes_option_is_validated():
    with pytest.raises(flow.options.MorphOptionsValidationError):
        flow.MorphOptions(landmark_probes=-1)
