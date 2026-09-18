"""Layout subpackage: all layout classes and registry."""

from .base import Layout, _apply_kwargs_to_options, get_layout, register_layout
from .centroid import CentroidLayout, CentroidLayoutOptions, CentroidMetrics
from .data_prep import LayoutData, compute_symbol_sizes, prepare_layout_data
from .flow import FlowDensityHistory, FlowDensityLayout, FlowDensityLayoutOptions, FlowDensityMetrics
from .grid import GridBasedLayout, GridBasedLayoutOptions, GridMetrics
from .layout_result import AlgorithmMetrics, LayoutResult, SimulationHistory, Transform
from .mosaic import HungarianOptions, MosaicLayout, MosaicLayoutOptions, MosaicMetrics
from .packing import (
    CirclePackingAdvancedOptions,
    CirclePackingLayout,
    CirclePackingLayoutOptions,
    PackingHistory,
    PackingMetrics,
)
from .physics import CirclePhysicsLayout, CirclePhysicsLayoutOptions, PhysicsHistory, PhysicsMetrics

__all__ = [
    "AlgorithmMetrics",
    "CentroidLayout",
    "CentroidLayoutOptions",
    "CentroidMetrics",
    "CirclePackingAdvancedOptions",
    "CirclePackingLayout",
    "CirclePackingLayoutOptions",
    "CirclePhysicsLayout",
    "CirclePhysicsLayoutOptions",
    "FlowDensityHistory",
    "FlowDensityLayout",
    "FlowDensityLayoutOptions",
    "FlowDensityMetrics",
    "GridBasedLayout",
    "GridBasedLayoutOptions",
    "GridMetrics",
    "HungarianOptions",
    "Layout",
    "LayoutData",
    "LayoutResult",
    "MosaicLayout",
    "MosaicLayoutOptions",
    "MosaicMetrics",
    "PackingHistory",
    "PackingMetrics",
    "PhysicsHistory",
    "PhysicsMetrics",
    "SimulationHistory",
    "Transform",
    "_apply_kwargs_to_options",
    "compute_symbol_sizes",
    "get_layout",
    "prepare_layout_data",
    "register_layout",
]

register_layout("centroid", CentroidLayout)
register_layout("flow_density", FlowDensityLayout)
register_layout("grid", GridBasedLayout)
register_layout("mosaic", MosaicLayout)
register_layout("packing", CirclePackingLayout)
register_layout("physics", CirclePhysicsLayout)
register_layout("topology", CirclePackingLayout)  # backward-compat alias
