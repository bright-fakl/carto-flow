"""
Circle Packing Layout
=====================

Create a symbol cartogram with symbols placed using a physics-based circle packing layout.
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
    size_normalization="total",
    show_progress=False,
)

# %%
# Plot the symbol cartogram

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

_ = symbol_carto.plot(
    ax=ax,
    # color by population size
    column="Population (Millions)",
    # shrink the colorbar
    legend_kwds={"shrink": 0.75},
    # label all states
    # use the State abbreviation look-up table in the censusdis package
    label=us_states["State Abbreviation"],
    # set label color to white and use small bold font
    label_color="w",
    label_kwargs={"fontweight": "bold"},
    label_fontsize=7,
)

plt.tight_layout()
