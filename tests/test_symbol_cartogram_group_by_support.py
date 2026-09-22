"""Tests for the ``group_by`` support check on symbol-cartogram layouts.

Layouts that do not let ``group_by`` affect placement must refuse it instead
of returning a cartogram in which the grouping was silently dropped.
"""

from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from carto_flow.symbol_cartogram import create_layout
from carto_flow.symbol_cartogram.layouts import (
    CentroidLayout,
    CirclePackingLayout,
    CirclePhysicsLayout,
    FlowDensityLayout,
    GridBasedLayout,
    MosaicLayout,
    check_group_by_support,
    get_layout,
    group_by_layouts,
    prepare_layout_data,
)
from carto_flow.symbol_cartogram.layouts.base import _LAYOUT_REGISTRY
from carto_flow.symbol_cartogram.presets import (
    dorling_grouped_cartogram,
    geographic_grouped_cartogram,
)

SUPPORTING = ["centroid", "flow_density", "mosaic", "packing", "topology"]
NOT_SUPPORTING = ["grid", "physics"]


def grouped_gdf() -> gpd.GeoDataFrame:
    """2x3 grid of unit squares, left column one group and right column another."""
    geoms = [box(c, r, c + 1, r + 1) for r in range(3) for c in range(2)]
    groups = ["left" if (i % 2 == 0) else "right" for i in range(len(geoms))]
    return gpd.GeoDataFrame(
        {"value": np.arange(1.0, len(geoms) + 1.0), "grp": groups, "tiles": [1] * len(geoms)},
        geometry=geoms,
    )


class TestSupportFlags:
    """The class-level flag matches what each layout actually reads."""

    @pytest.mark.parametrize(
        "cls",
        [CentroidLayout, CirclePackingLayout, FlowDensityLayout, MosaicLayout],
    )
    def test_supporting_layouts_declare_support(self, cls):
        assert cls.supports_group_by is True

    @pytest.mark.parametrize("cls", [GridBasedLayout, CirclePhysicsLayout])
    def test_non_supporting_layouts_do_not(self, cls):
        assert cls.supports_group_by is False

    def test_every_registered_layout_is_classified(self):
        for name in _LAYOUT_REGISTRY:
            assert name in SUPPORTING or name in NOT_SUPPORTING

    def test_group_by_layouts_lists_mosaic_without_aliases(self):
        names = group_by_layouts()
        assert "mosaic" in names
        assert "grid" not in names
        assert "physics" not in names
        # "topology" is an alias of "packing" and must not be listed twice
        assert len(names) == len(set(names))
        assert "topology" not in names


class TestCheckRaises:
    """``group_by`` on a layout that ignores it is an error, not a warning."""

    @pytest.mark.parametrize("name", NOT_SUPPORTING)
    def test_create_layout_raises(self, name):
        gdf = grouped_gdf()
        with pytest.raises(ValueError, match="does not support group_by"):
            create_layout(gdf, "value", group_by="grp", layout=name, show_progress=False)

    @pytest.mark.parametrize("name", NOT_SUPPORTING)
    def test_error_message_names_mosaic(self, name):
        gdf = grouped_gdf()
        with pytest.raises(ValueError, match="mosaic"):
            create_layout(gdf, "value", group_by="grp", layout=name, show_progress=False)

    @pytest.mark.parametrize("name", NOT_SUPPORTING)
    def test_compute_raises_directly(self, name):
        """The check also fires when ``compute`` is called without ``create_layout``."""
        data = prepare_layout_data(grouped_gdf(), "value", group_by="grp")
        with pytest.raises(ValueError, match="does not support group_by"):
            get_layout(name).compute(data, show_progress=False)

    @pytest.mark.parametrize("name", SUPPORTING)
    def test_supporting_layouts_never_raise(self, name):
        check_group_by_support(get_layout(name), True)


class TestCheckDoesNotFire:
    """Nothing changes for callers that do not pass ``group_by``."""

    @pytest.mark.parametrize("name", NOT_SUPPORTING)
    def test_no_group_by_runs(self, name):
        result = create_layout(grouped_gdf(), "value", layout=name, show_progress=False)
        assert len(result.transforms) == 6

    @pytest.mark.parametrize("name", NOT_SUPPORTING)
    def test_tile_count_runs(self, name):
        """``tile_count`` sets ``group_ids`` but not the user grouping, so it is allowed."""
        result = create_layout(grouped_gdf(), tile_count="tiles", layout=name, show_progress=False)
        assert len(result.transforms) == 6

    def test_supporting_layout_runs_with_group_by(self):
        result = create_layout(
            grouped_gdf(),
            "value",
            group_by="grp",
            layout=CirclePackingLayout(max_iterations=10, group_weight=0.5),
            show_progress=False,
        )
        assert result.group_ids is not None
        assert len(result.transforms) == 6


class TestInertGroupingWarning:
    """Layouts that support ``group_by`` warn when their group force is off."""

    def test_packing_warns_at_default_group_weight(self):
        with pytest.warns(UserWarning, match="group_weight"):
            create_layout(
                grouped_gdf(),
                "value",
                group_by="grp",
                layout=CirclePackingLayout(max_iterations=5),
                show_progress=False,
            )

    def test_flow_warns_at_default_cross_group_pull_scale(self):
        with pytest.warns(UserWarning, match="cross_group_pull_scale"):
            create_layout(
                grouped_gdf(),
                "value",
                group_by="grp",
                layout=FlowDensityLayout(max_iterations=5, grid_size=64),
                show_progress=False,
            )

    def test_packing_silent_once_group_weight_is_set(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            create_layout(
                grouped_gdf(),
                "value",
                group_by="grp",
                layout=CirclePackingLayout(max_iterations=5, group_weight=0.5),
                show_progress=False,
            )

    def test_flow_silent_once_cross_group_pull_scale_is_set(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            create_layout(
                grouped_gdf(),
                "value",
                group_by="grp",
                layout=FlowDensityLayout(max_iterations=5, grid_size=64, cross_group_pull_scale=0.0),
                show_progress=False,
            )

    @pytest.mark.parametrize(
        "layout",
        [CirclePackingLayout(max_iterations=5), FlowDensityLayout(max_iterations=5, grid_size=64)],
    )
    def test_no_warning_without_group_by(self, layout):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            create_layout(grouped_gdf(), "value", layout=layout, show_progress=False)

    @pytest.mark.parametrize("preset", [dorling_grouped_cartogram, geographic_grouped_cartogram])
    def test_grouped_presets_do_not_warn(self, preset):
        """The shipped grouped presets enable the group force, so they stay silent."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            preset(grouped_gdf(), "value", group_by="grp", show_progress=False)

    def test_layouts_without_an_inert_option_say_nothing(self):
        for name in ("centroid", "mosaic"):
            assert get_layout(name)._inert_group_by_warning() is None
