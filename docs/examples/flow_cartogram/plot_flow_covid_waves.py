"""
US COVID-19 Waves of 2020
=========================

Animates the first three waves of US COVID-19 infection as a flow cartogram.

Keyframes are every second week of 2020.  Each one is a cartogram sized to
that week's confirmed cases per 100,000 residents, so a state's area is its
infection intensity, not its population.  Color shows the same week's
*absolute* case count on a log scale.
The two encodings disagree on purpose: the largest state in a frame is the one
with the most intense outbreak per person, while the brightest is the one
recording the most cases.  A state can be one without being the other.

Case data: Johns Hopkins University CSSE COVID-19 Data Repository (CC BY 4.0).
"""

# %%
# Load data
# ---------
# Join the census boundaries with the weekly case series.  The 49 contiguous
# geographies (48 states plus DC) match on ``State Name`` directly.

import numpy as np

import carto_flow.data as examples
import carto_flow.flow_cartogram as flow
from carto_flow.geo_utils import densify_coverage

us_states = examples.load_us_census()
covid = examples.load_us_covid_weekly()

weekly_cases = covid.pivot(index="week_ending", columns="state_name", values="new_cases")
weekly_cases = weekly_cases[list(us_states["State Name"])]

# The flow cartogram moves vertices, so every boundary needs interior vertices
# to bend.  The bundled boundaries already carry a vertex every 5 km; going to
# 1 km gives the coastlines enough slack to absorb the extreme area changes
# below.  A few "straight segment" warnings still appear during the run: they
# come from the intermediate levels of the multiresolution morph, where a
# heavily stretched state can end up with a long straight edge that the next,
# finer level then reports.  They are advisory and the morph still converges.
us_states = densify_coverage(us_states, max_segment_length=1000)

# %%
# Pick keyframes and build the two encodings
# ------------------------------------------
# The series starts on 2020-03-22, the first week in which every contiguous
# state had recorded cases, and runs fortnightly to the end of 2020.

keyframe_weeks = weekly_cases.loc["2020-03-22":"2020-12-27"].index[::2]

cases = weekly_cases.loc[keyframe_weeks]
population = us_states.set_index("State Name")["Population"]
cases_per_100k = cases.div(population[cases.columns], axis=1) * 1e5

# Area: cases per 100k, written into the frame as the sizing column.
# Color: log10 of the raw weekly count.
color_values = [np.log10(cases.loc[week].values) for week in keyframe_weeks]

# %%
# Build one cartogram per keyframe
# --------------------------------

cartograms = []
for week in keyframe_weeks:
    df = us_states.copy()
    df["cases_per_100k"] = cases_per_100k.loc[week].values
    cartogram = flow.multiresolution_morph(
        df,
        "cases_per_100k",
        min_resolution=128,
        levels=4,
        options=flow.MorphOptions(show_progress=False),
    )
    cartograms.append(cartogram)

# %%
# Animate
# -------
# Three waves cross the map.  In late March and April the Northeast swells
# alone: New York reaches 374 cases per 100k in the week ending 2020-04-05,
# its highest of the year, with New Jersey, Massachusetts, Rhode Island and
# Connecticut behind it, while the interior shrinks to slivers.  Through July
# the Sun Belt takes over as the Northeast collapses — Arizona, Florida,
# Louisiana and Mississippi all pass 250 per 100k.  From September the Upper
# Midwest dominates: North Dakota is the largest shape on the map in every
# keyframe from 2020-10-04 to 2020-11-29, with South Dakota just behind it,
# and both pass 1,100 per 100k in mid-November despite holding 1.6 million
# people between them.  Tennessee leads in mid-December and California closes
# the year.
#
# Color separates that from raw caseload.  California and Texas stay near the
# bright end of the scale all year without growing much, because a large
# population turns a moderate rate into a large count.  In the week ending
# 2020-11-15 the Dakotas are the two biggest shapes on the map at 1,273 and
# 1,135 per 100k, yet they recorded 9,682 and 9,977 cases against California's
# 60,704 — roughly three times their combined total — so they sit well below
# California in color.

anim = flow.animation.animate_geometry_keyframes(
    keyframes=cartograms,
    duration=7,
    fps=10,
    color_values=color_values,
    colorbar=True,
    colorbar_label="Weekly confirmed cases\n(log10)",
    colorbar_kwargs={"shrink": 0.7},
    vmin=2,
    vmax=5,
    cmap="viridis",
    edgecolor="black",
    linewidth=0.4,
    title=lambda key, n, t: f"Weekly COVID-19 cases per 100k — week ending {keyframe_weeks[key - 1].date()}",
    show_axes=False,
    figsize=(6, 4),
)
