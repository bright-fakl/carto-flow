"""Tests for carto_flow.geo_utils.prescale."""

import numpy as np
import pytest
from shapely.geometry import Polygon, box

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


# ---------------------------------------------------------------------------
# Zero-value and zero-area cases (five-case analysis)
# ---------------------------------------------------------------------------


def test_target_density_zero_returns_geometries_unchanged():
    """Case 5: an all-zero dataset (target_density == 0) is a documented no-op."""
    geometries, _ = _two_touching_plus_island()
    values = np.zeros(3)

    result = prescale_connected_components(geometries, values, target_density=0.0)

    assert result == geometries
    for orig, new in zip(geometries, result, strict=False):
        assert new.equals(orig)


def test_component_with_zero_target_collapses_to_zero_area():
    """Case 2: current_area > 0, target_area == 0 -> exact zero-area collapse."""
    geometries, _ = _two_touching_plus_island()
    values = np.array([0.0, 0.0, 50.0])  # first component (indices 0,1) sums to zero

    scaled = prescale_connected_components(geometries, values, target_density=100.0)

    assert scaled[0].area == 0.0
    assert scaled[1].area == 0.0
    assert scaled[0].is_valid is False
    # The island keeps its normal scaling behaviour.
    assert scaled[2].area > 0


def test_total_area_preserved_including_when_a_component_collapses():
    """The total-area invariant holds exactly (by construction) even with a collapse."""
    geometries, _ = _two_touching_plus_island()
    values = np.array([0.0, 0.0, 50.0])
    target_density = 100.0

    scaled = prescale_connected_components(geometries, values, target_density)

    total_before = sum(g.area for g in geometries)
    total_target = float(values.sum()) / target_density  # only island has nonzero target
    total_after = sum(g.area for g in scaled)

    assert total_after == pytest.approx(total_target, rel=1e-9)
    assert total_after != total_before  # scaling did change the total, as expected


def test_zero_current_area_with_positive_target_warns_and_is_left_unchanged():
    """Case 3: a degenerate zero-area input carrying real data is surfaced, not hidden."""
    collapsed = Polygon([(0, 0), (0, 0), (0, 0)])  # zero-area degenerate polygon
    geometries = [collapsed, box(5, 5, 6, 6)]
    values = np.array([10.0, 20.0])

    with pytest.warns(UserWarning, match="zero current area"):
        scaled = prescale_connected_components(geometries, values, target_density=1.0)

    assert scaled[0] is collapsed


def test_mixed_sign_values_in_one_component_are_not_treated_as_zero():
    """[5, -5] in one component must not collapse it: abs() is applied before summing."""
    geometries = [box(0, 0, 1, 1), box(1, 0, 2, 1)]
    values = np.array([5.0, -5.0])
    values_for_prescale = np.abs(values)

    scaled = prescale_connected_components(geometries, values_for_prescale, target_density=5.0)

    total_area = sum(g.area for g in scaled)
    assert total_area == pytest.approx(2.0, rel=1e-9)  # (5+5)/5 = 2, not 0


def test_net_negative_dataset_behaves_like_its_absolute_value():
    """A component that nets to a negative raw sum still scales by its abs total."""
    geometries = [box(0, 0, 1, 1), box(1, 0, 2, 1)]
    raw_values = np.array([-10.0, 4.0])  # nets to -6, abs sums to 14
    values_for_prescale = np.abs(raw_values)

    scaled = prescale_connected_components(geometries, values_for_prescale, target_density=7.0)

    total_area = sum(g.area for g in scaled)
    assert total_area == pytest.approx(2.0, rel=1e-9)  # 14/7 = 2
