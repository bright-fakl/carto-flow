import numpy as np
import pytest

from carto_flow.symbol_cartogram.styling import Styling


class TestStyling:
    def test_set_symbol_basic(self):
        """Test basic set_symbol functionality"""
        styling = Styling()
        styling.set_symbol("circle")
        assert styling._global_symbol == "circle"

        styling.set_symbol("hexagon", indices=[0, 1])
        assert styling._per_geometry[0]["symbol"] == "hexagon"
        assert styling._per_geometry[1]["symbol"] == "hexagon"

    def test_set_symbol_array(self):
        """Test set_symbol with array of values"""
        styling = Styling()
        symbols = ["circle", "hexagon", "square"]
        styling.set_symbol(symbols)

        assert 0 in styling._per_geometry
        assert 1 in styling._per_geometry
        assert 2 in styling._per_geometry
        assert styling._per_geometry[0]["symbol"] == "circle"
        assert styling._per_geometry[1]["symbol"] == "hexagon"
        assert styling._per_geometry[2]["symbol"] == "square"

    def test_set_symbol_numpy_array(self):
        """Test set_symbol with numpy array"""
        styling = Styling()
        symbols = np.array(["circle", "hexagon", "square"])
        styling.set_symbol(symbols)

        assert 0 in styling._per_geometry
        assert 1 in styling._per_geometry
        assert 2 in styling._per_geometry
        assert styling._per_geometry[0]["symbol"] == "circle"
        assert styling._per_geometry[1]["symbol"] == "hexagon"
        assert styling._per_geometry[2]["symbol"] == "square"

    def test_set_symbol_mask(self):
        """Test set_symbol with boolean mask"""
        styling = Styling()
        mask = [True, False, True, False]
        styling.set_symbol("star", mask=mask)

        assert 0 in styling._per_geometry
        assert 2 in styling._per_geometry
        assert 1 not in styling._per_geometry
        assert 3 not in styling._per_geometry
        assert styling._per_geometry[0]["symbol"] == "star"
        assert styling._per_geometry[2]["symbol"] == "star"

    def test_set_symbol_numpy_mask(self):
        """Test set_symbol with numpy boolean array"""
        styling = Styling()
        mask = np.array([True, False, True, False])
        styling.set_symbol("star", mask=mask)

        assert 0 in styling._per_geometry
        assert 2 in styling._per_geometry
        assert styling._per_geometry[0]["symbol"] == "star"
        assert styling._per_geometry[2]["symbol"] == "star"

    def test_set_params_basic(self):
        """Test basic set_params functionality"""
        styling = Styling()
        params = {"pointy_top": False}
        styling.set_params(params)
        assert styling._global_params["pointy_top"] is False

        params = {"color": "red"}
        styling.set_params(params, indices=[0])
        assert styling._per_geometry[0]["params"]["color"] == "red"

    def test_set_params_array(self):
        """Test set_params with array of dicts"""
        styling = Styling()
        params_list = [{"a": 1}, {"b": 2}, {"c": 3}]
        styling.set_params(params_list)

        assert styling._per_geometry[0]["params"]["a"] == 1
        assert styling._per_geometry[1]["params"]["b"] == 2
        assert styling._per_geometry[2]["params"]["c"] == 3

    def test_set_params_mask(self):
        """Test set_params with boolean mask"""
        styling = Styling()
        mask = [True, False, True]
        params = {"pointy_top": True}
        styling.set_params(params, mask=mask)

        assert styling._per_geometry[0]["params"]["pointy_top"] is True
        assert styling._per_geometry[2]["params"]["pointy_top"] is True
        assert 1 not in styling._per_geometry

    def test_transform_basic(self):
        """Test basic transform functionality"""
        styling = Styling()
        styling.transform(scale=0.9, rotation=45, reflection=True)

        assert styling._global_transform.scale == 0.9
        assert styling._global_transform.reflection is True

    def test_transform_indices(self):
        """Test transform with indices"""
        styling = Styling()
        styling.transform(scale=0.8, indices=[1, 2])

        assert 1 in styling._per_geometry
        assert 2 in styling._per_geometry
        assert styling._per_geometry[1]["transform"].scale == 0.8
        assert styling._per_geometry[2]["transform"].scale == 0.8

    def test_transform_array(self):
        """Test transform with array values"""
        styling = Styling()
        styling.transform(scale=[0.9, 1.0, 1.1])

        assert 0 in styling._per_geometry
        assert 1 in styling._per_geometry
        assert 2 in styling._per_geometry
        assert styling._per_geometry[0]["transform"].scale == 0.9
        assert styling._per_geometry[1]["transform"].scale == 1.0
        assert styling._per_geometry[2]["transform"].scale == 1.1

    def test_transform_mask(self):
        """Test transform with boolean mask"""
        styling = Styling()
        mask = [True, False, True]
        styling.transform(scale=0.8, mask=mask)

        assert styling._per_geometry[0]["transform"].scale == 0.8
        assert styling._per_geometry[2]["transform"].scale == 0.8

    def test_transform_multiple_array_params(self):
        """Test transform with multiple array parameters"""
        styling = Styling()
        styling.transform(
            scale=[0.9, 1.0, 1.1],
            rotation=[45, 90, 135],
            reflection=[False, True, False],
        )

        assert styling._per_geometry[0]["transform"].scale == 0.9
        assert styling._per_geometry[1]["transform"].scale == 1.0
        assert styling._per_geometry[2]["transform"].scale == 1.1

    def test_combined_methods(self):
        """Test combined methods with different selectors"""
        styling = Styling()
        styling.set_symbol("hexagon")
        # Note: Using array with indices requires zip logic, which is not implemented
        # For now, set them separately
        styling.set_symbol("circle", indices=[0])
        styling.set_symbol("star", indices=[1])
        styling.set_params({"a": 1}, mask=[True, False, True])
        styling.transform(scale=0.9, indices=[0, 2])

        assert styling._global_symbol == "hexagon"
        assert styling._per_geometry[0]["symbol"] == "circle"
        assert styling._per_geometry[0]["transform"].scale == 0.9
        assert styling._per_geometry[0]["params"]["a"] == 1
        assert styling._per_geometry[1]["symbol"] == "star"
        assert styling._per_geometry[2]["params"]["a"] == 1
        assert styling._per_geometry[2]["transform"].scale == 0.9


if __name__ == "__main__":
    pytest.main([__file__])
