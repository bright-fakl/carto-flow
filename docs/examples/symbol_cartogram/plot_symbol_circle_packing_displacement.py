"""
Visualize Displacement
======================

This example demonstrates how to visualize displacement of the symbols relative to the original geometries.
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
    # use a physics-based circle packing layout
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

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

_ = us_states.plot(ax=ax, facecolor="0.9", edgecolor="white")

_ = smb.plot_displacement(
    symbol_carto,
    ax=ax,
    arrow_scale=2,
    arrow_color="black",
    alpha=0.4,
    facecolor="lightblue",
)

plt.tight_layout()
