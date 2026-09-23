"""Voronoi cartogram via Lloyd relaxation / CVT.

Distributes geometry centroids so that each point claims an equal-area
(or proportionally weighted) Voronoi cell within the outer union boundary.

Main entry points
-----------------
create_voronoi_cartogram
    Run Lloyd relaxation (or FFT flow) on a GeoDataFrame and return a
    :class:`VoronoiCartogram`.

Backend classes
---------------
RasterBackend
    Raster nearest-neighbor Lloyd (default; 10-50x faster than exact).
ExactBackend
    Exact scipy Voronoi + shapely clipping.

Backend helper types
--------------------
RelaxationSchedule
    Decaying SOR schedule; also accepted as the ``relaxation`` argument to
    either backend.
AdhesiveBoundary
    Snap boundary centroids toward the outer boundary.
ElasticBoundary
    FFT-driven elastic boundary deformation (RasterBackend only).

Other classes
-------------
VoronoiCartogram
    Result object with final positions, clipped cells, convergence metrics,
    and ``to_geodataframe()`` / ``plot()`` methods.
VoronoiOptions
    Run-control options (iterations, stopping criteria, history recording).
TopologyRepair
    Fine-grained control over the periodic topology permutation pipeline.
"""

from . import animation, visualization
from .api import create_voronoi_cartogram
from .backends import AdhesiveBoundary, ElasticBoundary, ExactBackend, RasterBackend, RelaxationSchedule
from .contiguity import make_groups_contiguous
from .fields import BaseField, ExactField, RasterField
from .history import VoronoiSnapshot
from .options import TopologyRepair, VoronoiOptions
from .result import TopologyAnalysis, TopologyRepairReport, VoronoiCartogram
from .visualization import (
    plot_cartogram,
    plot_comparison,
    plot_convergence,
    plot_displacement,
    plot_topology,
    plot_topology_repair,
)

__all__ = [
    "AdhesiveBoundary",
    "BaseField",
    "ElasticBoundary",
    "ExactBackend",
    "ExactField",
    "RasterBackend",
    "RasterField",
    "RelaxationSchedule",
    "TopologyAnalysis",
    "TopologyRepair",
    "TopologyRepairReport",
    "VoronoiCartogram",
    "VoronoiOptions",
    "VoronoiSnapshot",
    "animation",
    "create_voronoi_cartogram",
    "make_groups_contiguous",
    "plot_cartogram",
    "plot_comparison",
    "plot_convergence",
    "plot_displacement",
    "plot_topology",
    "plot_topology_repair",
    "visualization",
]
