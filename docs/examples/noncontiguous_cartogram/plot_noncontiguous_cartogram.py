"""
Non-contiguous Cartogram
========================

US states scaled about their centers to an area proportional to their population.
"""

# %%
# Every state keeps its shape and is scaled to ``population / density``, with the density of the densest state, so that
# no state grows and every area is proportional to its population.

import warnings

import carto_flow.data as examples
from carto_flow.noncontiguous_cartogram import create_noncontiguous_cartogram

states = examples.load_us_census(population=True, simplify=1000)

cartogram = create_noncontiguous_cartogram(states, "Population")
plot = cartogram.plot(facecolor="Population")
_ = plot.ax.set_title("US states scaled by population")

# %%
# With the density of the state with the largest population (California), 11 states would have to grow, DC to 41.6
# times its area. Growth is capped at the original size by default, so these states are smaller than proportional.

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    capped = create_noncontiguous_cartogram(states, "Population", reference="largest")
plot = capped.plot(facecolor="_ncont_scale", legend_label="scale factor")
_ = plot.ax.set_title(f"Reference: the largest state ({capped.capped.sum()} regions capped)")
