"""
Symbol Styling
==============

Custom styling of symbol cartograms is possible via the `Styling` class.
"""

# %%
# Load data and create the symbol cartogram.

import matplotlib.pyplot as plt

import carto_flow.data as examples
import carto_flow.symbol_cartogram as smb

us_states = examples.load_us_census(population=True)

symbol_carto = smb.create_symbol_cartogram(
    us_states,
    layout=smb.GridBasedLayout(tiling=smb.HexagonTiling()),
    show_progress=False,
)

# %%
# Custom styling of the symbol cartogram

# We can use the `Styling` class to customize the symbol styling.
# Here, we set the symbol to be a square and transform the rotation
# of the symbols based on their x-coordinate.
xmin = symbol_carto.symbols._symbol_x.min()
xmax = symbol_carto.symbols._symbol_x.max()

symbol_rotation = 90 * (symbol_carto.symbols._symbol_x.values - xmin) / (xmax - xmin)

symbol_carto_restyled = symbol_carto.restyle(
    smb.Styling().set_symbol(smb.SquareSymbol()).transform(rotation=symbol_rotation)
)

# %%
# Plot the symbol cartogram

fig, ax = plt.subplots(1, 1, figsize=(8, 6))

result = symbol_carto_restyled.plot(
    ax=ax,
    column="Population (Millions)",
    legend_kwds={"shrink": 0.75},
    label=us_states["State Abbreviation"],
    label_color="w",
    label_kwargs={"fontweight": "bold"},
)

plt.tight_layout()
