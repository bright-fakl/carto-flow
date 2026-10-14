"""The pre-scaling moves the landmarks and the displacement coordinates with their component."""

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import box

import carto_flow.flow_cartogram as flow
from carto_flow.geo_utils.prescale import prescale_connected_components, prescale_scalings, scale_points


def _two_far_squares():
    gdf = gpd.GeoDataFrame({"v": [1.0, 4.0]}, geometry=[box(0, 0, 1, 1), box(3, 0, 4, 1)])
    rng = np.random.default_rng(0)
    points = np.vstack([
        np.column_stack([rng.uniform(0, 1, 300), rng.uniform(0, 1, 300)]),
        np.column_stack([rng.uniform(3, 4, 300), rng.uniform(0, 1, 300)]),
    ])
    return gdf, points, np.repeat([0, 1], 300)


def _options():
    return flow.MorphOptions(show_progress=False, grid=128, prescale_components=True, mean_tol=0.005, max_tol=0.01)


def test_landmarks_stay_in_their_region_with_prescale_components():
    gdf, points, owner = _two_far_squares()
    cartogram = flow.create_flow_cartogram(
        gdf, "v", landmarks=gpd.GeoSeries(shapely.points(points)), options=_options()
    )
    geoms = list(cartogram.get_geometry())
    carried = shapely.get_coordinates(np.asarray(cartogram.get_landmarks(), dtype=object))
    for r in (0, 1):
        assert shapely.contains(geoms[r], shapely.points(carried[owner == r])).mean() > 0.99


def test_displacement_coordinates_stay_in_their_region_with_prescale_components():
    gdf, points, owner = _two_far_squares()
    cartogram = flow.create_flow_cartogram(gdf, "v", displacement_coords=points, options=_options())
    geoms = list(cartogram.get_geometry())
    carried = np.asarray(cartogram.get_coords())
    for r in (0, 1):
        assert shapely.contains(geoms[r], shapely.points(carried[owner == r])).mean() > 0.99


def test_scale_points_scales_like_the_geometries():
    geoms = [box(0, 0, 2, 2), box(10, 0, 12, 2)]
    values = np.array([1.0, 3.0])
    scalings = prescale_scalings(geoms, values, target_density=0.5)  # target areas 2 and 6
    scaled = prescale_connected_components(geoms, values, 0.5)
    points = np.array([[0.0, 0.0], [2.0, 2.0], [10.0, 0.0], [5.0, 5.0]])
    moved, log_scaling = scale_points(points, geoms, scalings)
    # the corners of a square follow the corners of the scaled geometry
    np.testing.assert_allclose(moved[:2], [scaled[0].bounds[:2], scaled[0].bounds[2:]])
    np.testing.assert_allclose(moved[2], scaled[1].bounds[:2])
    np.testing.assert_allclose(moved[3], points[3])  # outside every geometry
    np.testing.assert_allclose(log_scaling, [np.log(scaled[0].area / 4)] * 2 + [np.log(scaled[1].area / 4), 0.0])
