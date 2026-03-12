"""
Test suite for grid construction and management utilities.

This module tests the functionality of the grid module including:
- Grid class for structured grid information with lazy computation
- build_multilevel_grids function for multi-resolution grids
"""

import numpy as np
import pytest

from carto_flow.flow_cartogram.grid import (
    Grid,
    build_multilevel_grids,
)


class TestGrid:
    """Test cases for Grid class."""

    def test_creation_with_valid_data(self):
        """Test creating Grid with bounds and size."""
        bounds = (0, 0, 10, 5)
        grid = Grid(bounds, size=20)

        assert grid.sx == 20
        assert grid.sy == 10
        assert grid.bounds == bounds
        # Check that coordinate arrays have correct shapes
        assert grid.x_coords.shape == (20,)
        assert grid.y_coords.shape == (10,)
        assert grid.X.shape == (10, 20)
        assert grid.Y.shape == (10, 20)

    def test_shape_property(self):
        """Test shape property returns correct (rows, columns) tuple."""
        bounds = (0, 0, 10, 5)
        grid = Grid(bounds, size=(40, 20))

        assert grid.shape == (20, 40)

    def test_bounds_property(self):
        """Test bounds property returns correct (xmin, ymin, xmax, ymax) tuple."""
        bounds = (10, 20, 30, 25)
        grid = Grid(bounds, size=20)

        assert grid.bounds == bounds

    def test_size_and_spacing_properties(self):
        """Test size and spacing properties."""
        bounds = (0, 0, 10, 5)
        grid = Grid(bounds, size=20)

        assert grid.size == (20, 10)  # sx, sy
        assert abs(grid.spacing[0] - 0.5) < 1e-10  # dx
        assert abs(grid.spacing[1] - 0.5) < 1e-10  # dy

    def test_lazy_properties_and_caching(self):
        """Test that lazy properties are computed and cached correctly."""
        bounds = (0, 0, 10, 5)
        grid = Grid(bounds, size=20)

        # Test caching
        x_coords_1 = grid.x_coords
        x_coords_2 = grid.x_coords
        assert x_coords_1 is x_coords_2  # Same object

        X_1 = grid.X
        X_2 = grid.X
        assert X_1 is X_2  # Same object

        # Test correct shapes
        assert x_coords_1.shape == (20,)
        assert X_1.shape == (10, 20)
        assert grid.Y.shape == (10, 20)
        assert grid.x_edges.shape == (21,)
        assert grid.y_edges.shape == (11,)


class TestBuildMultilevelGrids:
    """Test cases for build_multilevel_grids function."""

    def test_three_level_grids(self):
        """Test creating three resolution levels."""
        bounds = (0, 0, 100, 80)
        grids = build_multilevel_grids(bounds, N=64, n_levels=3)

        assert len(grids) == 3

        # Level 0 (lowest resolution) - N=64 is the coarsest
        assert grids[0].sx == 64
        assert grids[0].sy == 51

        # Level 1 (middle resolution) - doubled from level 0
        assert grids[1].sx == 128
        assert grids[1].sy == 102

        # Level 2 (highest resolution) - doubled from level 1
        assert grids[2].sx == 256
        assert grids[2].sy == 204

    def test_two_level_grids(self):
        """Test creating two resolution levels."""
        bounds = (0, 0, 100, 100)  # Square for simplicity
        grids = build_multilevel_grids(bounds, N=64, n_levels=2)

        assert len(grids) == 2

        # Level 0 (lowest) - N=64 is the coarsest
        assert grids[0].sx == 64
        assert grids[0].sy == 64

        # Level 1 (highest) - doubled from level 0
        assert grids[1].sx == 128
        assert grids[1].sy == 128

    def test_single_level_grid(self):
        """Test creating single resolution level."""
        bounds = (0, 0, 100, 50)
        grids = build_multilevel_grids(bounds, N=32, n_levels=1)

        assert len(grids) == 1
        assert grids[0].sx == 32
        assert grids[0].sy == 16  # Adjusted for aspect ratio

    def test_bounds_preservation(self):
        """Test that all levels preserve the same bounds."""
        bounds = (10, 20, 110, 70)
        grids = build_multilevel_grids(bounds, N=128, n_levels=3)

        # All levels should have the same expanded bounds
        expected_bounds = (-40.0, -5.0, 160.0, 95.0)

        for grid in grids:
            assert grid.bounds == pytest.approx(expected_bounds, abs=1e-10)

    def test_fft_friendly_dimensions(self):
        """Test that higher grid levels are exact multiples of the base level."""
        bounds = (0, 0, 100, 80)
        grids = build_multilevel_grids(bounds, N=64, n_levels=3)

        # Each level i should have dimensions exactly 2^i times the base level
        for level, grid in enumerate(grids):
            scale = 2**level
            assert grid.sx == grids[0].sx * scale
            assert grid.sy == grids[0].sy * scale

    def test_with_margin(self):
        """Test multi-level grids with custom margin."""
        bounds = (0, 0, 100, 50)
        grids = build_multilevel_grids(bounds, N=128, n_levels=2, margin=0.5)

        # With 50% margin, bounds should be expanded
        expected_bounds = (-50, -25, 150, 75)

        for grid in grids:
            assert grid.bounds == pytest.approx(expected_bounds, abs=1e-10)


class TestIntegration:
    """Integration tests for grid functions."""

    def test_multilevel_grid_hierarchy(self):
        """Test that multi-level grids form a proper hierarchy."""
        bounds = (0, 0, 100, 80)
        grids = build_multilevel_grids(bounds, N=256, n_levels=3)

        # Each level should have exactly double the dimensions of the previous
        for i in range(1, len(grids)):
            prev_grid = grids[i - 1]
            curr_grid = grids[i]

            # Dimensions should be exactly double (allowing for rounding)
            assert abs(curr_grid.sx - 2 * prev_grid.sx) <= 1
            assert abs(curr_grid.sy - 2 * prev_grid.sy) <= 1

    def test_grid_coordinate_continuity(self):
        """Test that grid coordinates are continuous and well-formed."""
        bounds = (0, 0, 10, 5)
        grid = Grid.from_bounds(bounds, size=20)

        # Check that x_coords are monotonically increasing
        assert np.all(np.diff(grid.x_coords) > 0)

        # Check that y_coords are monotonically increasing
        assert np.all(np.diff(grid.y_coords) > 0)

        # Check that coordinate spacing is approximately constant
        x_diffs = np.diff(grid.x_coords)
        y_diffs = np.diff(grid.y_coords)

        assert np.allclose(x_diffs, grid.dx, rtol=1e-10)
        assert np.allclose(y_diffs, grid.dy, rtol=1e-10)

    def test_large_grid_creation(self):
        """Test creation of large grids."""
        bounds = (0, 0, 1000, 500)
        grid = Grid.from_bounds(bounds, size=1000)

        # Should handle large grids without issues
        assert grid.sx == 1000
        assert grid.sy == 500
        assert grid.shape == (500, 1000)

        # Check that arrays have correct shape
        assert grid.X.shape == (500, 1000)
        assert grid.Y.shape == (500, 1000)
        assert len(grid.x_coords) == 1000
        assert len(grid.y_coords) == 500


class TestErrorConditions:
    """Test error conditions and edge cases."""

    def test_build_multilevel_grids_invalid_parameters(self):
        """Test build_multilevel_grids with invalid parameters."""
        bounds = (0, 0, 10, 5)

        # Test with zero levels - this creates an empty list
        grids = build_multilevel_grids(bounds, N=64, n_levels=0)
        assert len(grids) == 0  # Creates empty list for n_levels=0

        # Test with negative levels - this also creates an empty list
        grids = build_multilevel_grids(bounds, N=64, n_levels=-1)
        assert len(grids) == 0  # Creates empty list for negative n_levels

    def test_grid_validation_edge_cases(self):
        """Test Grid validation with edge cases."""
        # Test with small bounds
        bounds = (0, 0, 1, 1)
        grid = Grid(bounds, size=1)

        assert grid.shape == (1, 1)
        assert grid.bounds == bounds


class TestFromSizeAndSpacing:
    """Test cases for Grid.from_size_and_spacing method."""

    def test_default_center_origin(self):
        """Test from_size_and_spacing with default center at origin."""
        grid = Grid.from_size_and_spacing(size=10, spacing=1.0)

        # Should be centered at origin (0,0)
        assert grid.sx == 10
        assert grid.sy == 10
        # Width = 10 * 1.0 = 10, so bounds should be -5 to 5
        expected_bounds = (-5.0, -5.0, 5.0, 5.0)
        assert grid.bounds == pytest.approx(expected_bounds, abs=1e-10)

    def test_custom_center(self):
        """Test from_size_and_spacing with custom center coordinates."""
        center = (10.0, 20.0)
        grid = Grid.from_size_and_spacing(size=10, spacing=1.0, center=center)

        # Should be centered at (10, 20)
        assert grid.sx == 10
        assert grid.sy == 10
        # Width = 10 * 1.0 = 10, so bounds should be 5 to 15, 15 to 25
        expected_bounds = (5.0, 15.0, 15.0, 25.0)
        assert grid.bounds == pytest.approx(expected_bounds, abs=1e-10)

    def test_rectangular_size_with_center(self):
        """Test from_size_and_spacing with rectangular size and custom center."""
        grid = Grid.from_size_and_spacing(size=(20, 10), spacing=(1.0, 2.0), center=(5.0, 10.0))

        # Should have correct dimensions
        assert grid.sx == 20
        assert grid.sy == 10
        # Width = 20 * 1.0 = 20, Height = 10 * 2.0 = 20
        # So bounds should be 5±10, 10±10 = -5 to 15, 0 to 20
        expected_bounds = (-5.0, 0.0, 15.0, 20.0)
        assert grid.bounds == pytest.approx(expected_bounds, abs=1e-10)

    def test_center_affects_bounds_correctly(self):
        """Test that different centers produce different bounds."""
        size, spacing = 10, 1.0

        # Grid centered at origin
        grid1 = Grid.from_size_and_spacing(size, spacing, center=(0.0, 0.0))

        # Grid centered at (10, 10)
        grid2 = Grid.from_size_and_spacing(size, spacing, center=(10.0, 10.0))

        # Bounds should be shifted by (10, 10)
        expected_bounds1 = (-5.0, -5.0, 5.0, 5.0)
        expected_bounds2 = (5.0, 5.0, 15.0, 15.0)

        assert grid1.bounds == pytest.approx(expected_bounds1, abs=1e-10)
        assert grid2.bounds == pytest.approx(expected_bounds2, abs=1e-10)

    def test_coordinate_arrays_with_center(self):
        """Test that coordinate arrays are computed correctly with custom center."""
        grid = Grid.from_size_and_spacing(size=4, spacing=1.0, center=(2.0, 3.0))

        # Check that center of coordinate arrays matches the specified center
        # For a 4x4 grid with spacing 1.0, coordinates should be at:
        # x: 2.0-1.5, 2.0-0.5, 2.0+0.5, 2.0+1.5 = 0.5, 1.5, 2.5, 3.5
        # y: 3.0-1.5, 3.0-0.5, 3.0+0.5, 3.0+1.5 = 1.5, 2.5, 3.5, 4.5
        expected_x = np.array([0.5, 1.5, 2.5, 3.5])
        expected_y = np.array([1.5, 2.5, 3.5, 4.5])

        assert np.allclose(grid.x_coords, expected_x, atol=1e-10)
        assert np.allclose(grid.y_coords, expected_y, atol=1e-10)
