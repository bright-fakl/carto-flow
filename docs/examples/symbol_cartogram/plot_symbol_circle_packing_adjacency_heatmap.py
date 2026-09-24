"""
Adjacency Heatmap
=================

This example demonstrates how to visualize a heatmap of adjacency relationships used to create the symbol cartogram.
"""

# %%
# Load data and create the symbol cartogram.

import matplotlib.pyplot as plt

import carto_flow.data as examples
import carto_flow.symbol_cartogram as smb

us_states = examples.load_us_census(population=True)

symbol_carto = smb.create_symbol_cartogram(
    us_states,
    # provide a column for proportional sizing of the symbols
    "Population",
    # use a force-based circle packing layout
    layout=smb.CirclePackingLayout(),
    # size the circles such that their total area
    # matches the total area of the original geometries
    size_normalization="total",
    # compute adjacency as the fraction of a geometry's perimeter that
    # is shared with another geometry
    adjacency_mode="weighted",
    show_progress=False,
)

# %%
# Plot the symbol cartogram

result = smb.plot_adjacency_heatmap(
    symbol_carto,
    labels=us_states["State Abbreviation"],
    colorbar=True,
    sort_by="label",
    figsize=(10, 8),
    tick_fontsize=8,
    colorbar_kwds={"shrink": 0.5},
)

plt.tight_layout()
