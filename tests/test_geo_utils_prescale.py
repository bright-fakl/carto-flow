"""Tests for carto_flow.geo_utils.prescale."""

import numpy as np
from shapely.geometry import box

from carto_flow.geo_utils.prescale import (
    components_from_adjacency,
    compute_connected_components,
    prescale_connected_components,
)


def _two_touching_plus_island():
    # Two adjacent unit squares (one component) and one isolated square far away.
    geometries = [box(0, 0, 1, 1), box(1, 0, 2, 1), box(10, 10, 11, 11)]
    values = np.array([100.0, 200.0, 50.0])
    return geometries, values


def test_compute_connected_components_groups_touching_geometries():
    geometries, _ = _two_touching_plus_island()
    labels, components = compute_connected_components(geometries)

    assert len(components) == 2
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_components_from_adjacency_matches_union_find():
    adj = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ])
    labels, components = components_from_adjacency(adj)

    assert len(components) == 2
    assert labels[0] == labels[1]
    assert labels[2] != labels[0]


def test_prescale_scales_each_component_to_target_area():
    geometries, values = _two_touching_plus_island()
    target_density = 100.0

    scaled = prescale_connected_components(geometries, values, target_density)

    _, components = compute_connected_components(geometries)
    for indices in components:
        component_values = values[indices]
        target_area = component_values.sum() / target_density
        actual_area = sum(scaled[i].area for i in indices)
        assert actual_area == target_area or abs(actual_area - target_area) / target_area < 1e-6


def test_prescale_preserves_aspect_ratio_within_component():
    geometries, values = _two_touching_plus_island()
    target_density = 100.0

    scaled = prescale_connected_components(geometries, values, target_density)

    for orig, new in zip(geometries[:2], scaled[:2], strict=False):
        ox0, oy0, ox1, oy1 = orig.bounds
        nx0, ny0, nx1, ny1 = new.bounds
        orig_ratio = (ox1 - ox0) / (oy1 - oy0)
        new_ratio = (nx1 - nx0) / (ny1 - ny0)
        assert new_ratio == orig_ratio or abs(new_ratio - orig_ratio) / orig_ratio < 1e-9
