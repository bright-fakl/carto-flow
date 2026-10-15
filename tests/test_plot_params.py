"""Tests for the helpers of the plot parameter vocabulary."""

import warnings

import numpy as np
import pytest

from carto_flow._plot_params import classify_string, deprecated_init_params, deprecated_params


@deprecated_params(facecolor=("column", "color_by"), legend="colorbar")
def _plot(facecolor=None, legend=False):
    return facecolor, legend


def test_old_name_is_an_alias_with_a_warning():
    with pytest.warns(DeprecationWarning, match="'column' is deprecated; use 'facecolor'"):
        assert _plot(column="pop") == ("pop", False)
    with pytest.warns(DeprecationWarning, match="color_by"):
        assert _plot(color_by="pop") == ("pop", False)
    with pytest.warns(DeprecationWarning, match="colorbar"):
        assert _plot(colorbar=True) == (None, True)


def test_new_name_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _plot(facecolor="pop", legend=True) == ("pop", True)


def test_old_and_new_name_together_raise():
    with pytest.raises(TypeError, match="both"):
        _plot(column="a", facecolor="b")


def test_deprecated_init_params():
    from dataclasses import dataclass

    @deprecated_init_params(legend="colorbar")
    @dataclass
    class Options:
        legend: bool = False

    with pytest.warns(DeprecationWarning):
        assert Options(colorbar=True).legend is True
    assert Options(legend=True).legend is True


def test_classify_string_order_and_warning():
    assert classify_string("pop", ["pop"]) == "column"
    assert classify_string("density", ["pop"], ["density"]) == "builtin"
    assert classify_string("red", ["pop"]) == "color"
    with pytest.warns(UserWarning, match="column and a color"):
        assert classify_string("wheat", ["wheat"]) == "column"
    with pytest.warns(UserWarning, match="column and a builtin"):
        assert classify_string("density", ["density"], ["density"]) == "column"
    with pytest.raises(ValueError, match="neither a column"):
        classify_string("nope", ["pop"], ["density"])


def test_resolve_fill_kinds():
    from carto_flow._plot_params import resolve_fill

    assert resolve_fill(None, ["pop"]) == ("none", None)
    assert resolve_fill("pop", ["pop"]) == ("column", "pop")
    assert resolve_fill("density", ["pop"], ["density"]) == ("builtin", "density")
    assert resolve_fill("#d4e6f1", ["pop"]) == ("color", "#d4e6f1")
    kind, values = resolve_fill(np.arange(3.0), ["pop"], n=3)
    assert kind == "values" and np.array_equal(values, np.arange(3.0))
    # a numeric array of another length, or an array of colors, is passed on as colors
    assert resolve_fill(np.arange(4.0), ["pop"], n=3)[0] == "color"
    assert resolve_fill(["red", "blue", "green"], ["pop"], n=3)[0] == "color"
    with pytest.warns(UserWarning, match="column and a color"):
        assert resolve_fill("wheat", ["wheat"]) == ("column", "wheat")
    with pytest.raises(ValueError, match="neither a column"):
        resolve_fill("nope", ["pop"])
