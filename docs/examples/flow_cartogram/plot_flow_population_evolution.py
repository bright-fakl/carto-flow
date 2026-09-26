"""
US State Population (1920-2024)
==========================================

Animates multiple US State population cartograms over a century.

Each keyframe is a cartogram sized to that year's population.  Colors
show each state's annualised growth rate *relative to the national
average*: green states are growing faster than the US as a whole,
magenta states are growing slower (or shrinking).
"""

# %%
# Load data
# ---------
# Join census boundaries with historical population estimates and
# restrict to the 48 contiguous states.

import numpy as np

import carto_flow
import carto_flow.data as examples
import carto_flow.flow_cartogram as flow
from carto_flow.geo_utils import densify_coverage

us_states = examples.load_us_census()
population = examples.load_us_state_population()

us_states = us_states.join(
    population.pivot(index="state_abbr", columns="year", values="population"),
    on="State Abbreviation",
)
us_states = us_states[~us_states["State Abbreviation"].isin(("AK", "HI", "PR"))]
# Simplifying drops vertices along gentle boundaries, which leaves a few very
# long straight segments - here up to 519 km. A segment with no interior
# vertices cannot bend during the morph, so the solver warns about it. Densify
# after simplifying to put the vertices back where they are needed, evenly
# rather than where the original outline happened to be detailed. The pair is
# also faster than simplifying alone, because boundaries that can bend
# converge in fewer iterations.
us_states = carto_flow.simplify_coverage(us_states, tolerance=5000)
us_states = densify_coverage(us_states, max_segment_length=2000)

# %%
# Compute per-state population change relative to national average
# ---------------------------------------------------------------

years = [
    1920,
    1925,
    1930,
    1935,
    1940,
    1945,
    1950,
    1955,
    1960,
    1965,
    1970,
    1975,
    1980,
    1985,
    1990,
    1995,
    2000,
    2005,
    2010,
    2015,
    2020,
    2024,
]

all_years = population["year"].unique()
total_population = us_states[all_years].sum()
total_change = 100 * np.gradient(total_population) / total_population
per_state_change = us_states[all_years].apply(lambda x: 100 * np.gradient(x) / x, axis=1)
per_state_change_relative = per_state_change - total_change

# Set to False to color by raw per-state change instead of US-average-adjusted change.
# The adjusted version removes the national trend so colors highlight which states
# are growing faster or slower *relative to the country as a whole*.
us_average_adjusted = True

if us_average_adjusted:
    color_values = np.split(per_state_change_relative[years].values, len(years), axis=1)
    colorbar_label = "State Population Change (%)\nrelative to US Average"
else:
    color_values = np.split(per_state_change[years].values, len(years), axis=1)
    colorbar_label = "State Population Change (%)"

# %%
# Build one cartogram per keyframe year
# -------------------------------------
# Set scale_to_max_population=True to also scale the overall map size with the
# national population — the total map area then grows over time, making absolute
# population growth visible alongside the redistribution between states.
scale_to_max_population = False

scale_factors = total_population / total_population.max()
origin = us_states.geometry.union_all().centroid

cartograms = []
for year in years:
    df = us_states
    if scale_to_max_population:
        df = us_states.copy()
        sf = scale_factors[year]
        df.geometry = df.geometry.scale(xfact=sf, yfact=sf, origin=origin)

    cartogram = flow.multiresolution_morph(
        df,
        year,
        min_resolution=128,
        levels=4,
        options=flow.MorphOptions(show_progress=False),
    )
    cartograms.append(cartogram)

# %%
# Animate
# -------


anim = flow.animation.animate_geometry_keyframes(
    keyframes=cartograms,
    duration=10,
    color_values=color_values,
    colorbar=True,
    colorbar_label=colorbar_label,
    colorbar_kwargs={"shrink": 0.7},
    vmin=-4,
    vmax=4,
    cmap="PiYG",
    edgecolor="black",
    linewidth=0.5,
    title=lambda key, n, t: f"US Population Cartogram {years[key - 1]}",
    show_axes=False,
    figsize=(6, 5),
)
