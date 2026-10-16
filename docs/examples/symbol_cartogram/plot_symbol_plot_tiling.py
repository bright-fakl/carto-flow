"""
Visualize Tiling Grid
=====================

Visualize the tiling grid used to create the symbol cartogram.
"""

# %%
# Load data and create the symbol cartogram.

import matplotlib.pyplot as plt

import carto_flow.data as examples
import carto_flow.symbol_cartogram as smb
from carto_flow import get_style

us_states = examples.load_us_census(population=True)

# use a grid-based layout with hexagonal tiling
# and a square symbol
symbol_carto = smb.create_symbol_cartogram(
    us_states,
    layout=smb.GridBasedLayout(tiling=smb.HexagonTiling()),
    styling=smb.Styling(symbol="square"),
    show_progress=False,
)


# %%
# Plot the symbol cartogram

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
get_style().apply(fig)

result = smb.plot_tiling(
    symbol_carto,
    ax=ax,
    show_symbols=True,
    show_assigned=True,
    show_unassigned=True,
)
