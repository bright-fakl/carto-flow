"""
Centroid Layout
===============

Create a symbol cartogram with symbols placed at the centroids of the original geometries.
"""

# %%
# Load data and create the symbol cartogram.

import matplotlib.pyplot as plt

import carto_flow.data as examples
import carto_flow.symbol_cartogram as smb

us_states = examples.load_us_census(population=True)

# place symbols at the state centroids and remove overlap
symbol_carto = smb.centroid_cartogram(
    us_states,
    # provide a column for proportional sizing of the symbols
    "Population",
)

# %%
# Plot the symbol cartogram

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

_ = symbol_carto.plot(
    ax=ax,
    # color by population size
    facecolor="Population (Millions)",
    # shrink the colorbar
    legend_kwargs={"shrink": 0.5},
    # label the largest states
    # the abbreviation column comes from the bundled census snapshot
    labels=[
        row["State Abbreviation"] if row["State Abbreviation"] in ["CA", "TX", "FL", "NY"] else ""
        for _, row in us_states.iterrows()
    ],
)

plt.tight_layout()
