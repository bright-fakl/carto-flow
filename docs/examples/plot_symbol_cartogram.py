"""
Symbol Cartogram
================

A symbol cartogram in which each US State is represented by a
(hexagon-shaped) symbol in a hexagon tiling grid.
Uses `carto_flow.symbol_cartogram.create_symbol_cartogram`
"""

# %%
# Load dataset and create cartogram.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import carto_flow.data as examples
import carto_flow.symbol_cartogram as smb

us_states = examples.load_us_census(population=True)

# compute symbol cartogram
symbol_carto = smb.create_symbol_cartogram(
    us_states,
    layout=smb.GridBasedLayout(tiling=smb.HexagonTiling()),
)

# %%
# Plot the cartogram.
p = symbol_carto.plot(
    # color by population size
    column="Population (Millions)",
    # shrink the colorbar
    legend_kwds={"shrink": 0.75},
    # label all states
    # the abbreviation column comes from the bundled census snapshot
    label=us_states["State Abbreviation"],
    # set label color to white and use bold font
    label_color="w",
    label_kwargs={"fontweight": "bold"},
)

p.ax.set(title="Symbol Cartogram")
plt.tight_layout()
