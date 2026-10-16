"""
Non-contiguous (Olson) area cartograms.

A non-contiguous cartogram keeps the shape of every region and scales it about its own center to an area that is
proportional to the value of the region. The regions do not touch, so there is no distortion of the shapes and no
need for a solver; the map shows the value by the size of the region (Olson, 1976).

Functions
---------
create_noncontiguous_cartogram
    Scale each region of a GeoDataFrame to an area proportional to its value.

Classes
-------
NoncontiguousCartogram
    Result with the scaled regions, the scale factors and the reference density.
NoncontiguousPlotResult
    Artists of a plot.

Examples
--------
>>> from carto_flow.noncontiguous_cartogram import create_noncontiguous_cartogram
>>> cartogram = create_noncontiguous_cartogram(states, "Population")
>>> cartogram.plot(facecolor="Region")

See Also
--------
[carto_flow.flow_cartogram][] : Contiguous cartograms.
"""

from .algorithm import create_noncontiguous_cartogram
from .result import NoncontiguousCartogram, NoncontiguousPlotResult
from .visualization import plot_cartogram

__all__ = [
    "NoncontiguousCartogram",
    "NoncontiguousPlotResult",
    "create_noncontiguous_cartogram",
    "plot_cartogram",
]
