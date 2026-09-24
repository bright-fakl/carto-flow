"""
Visualize Adjacency
===================

This example demonstrates how to visualize adjacency used to create the symbol cartogram.
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

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6),
    sharex=True,
    sharey=True,
    gridspec_kw={"wspace": 0.02, "left": 0.02, "right": 0.98, "top": 0.98, "bottom": 0.02},
)

_ = smb.plot_adjacency(
    symbol_carto,
    ax=axes[0],
    edge_cmap="cool",
    edge_alpha=1,
    facecolor="lightblue",
    colorbar=False,
    use_original_positions=True,
    show_original=True,
    original_gdf=us_states,
    show_symbols=False,
)

_ = smb.plot_adjacency(
    symbol_carto,
    ax=axes[1],
    edge_cmap="cool",
    edge_alpha=1,
    facecolor="lightblue",
    colorbar_kwds={"shrink": 0.4, "ax": axes, "pad": 0.01},
)
