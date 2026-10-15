"""The parameter names of the plot, animate and preview functions stay within the vocabulary.

Every public ``plot*``, ``animate*`` and ``preview*`` function and the ``*PlotOptions`` dataclasses of a cartogram
module may only use parameter names of the allowlist of the module. A new name has to be added to the allowlist on
purpose, so the vocabulary does not drift (see the style-system proposal, "Parameter vocabulary").
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "carto_flow"

# Parameters that are functional, not about appearance, are listed with the vocabulary names.
ALLOWED = {
    "flow_cartogram": {
        # data and objects
        "ax",
        "bounds",
        "cartogram",
        "gdf",
        "iteration",
        "keyframes",
        "left",
        "left_iteration",
        "metric",
        "modulator",
        "right",
        "through",
        "values",
        "velocity",
        "density",
        "workflow",
        # fill and legend
        "facecolor",
        "cmap",
        "vmin",
        "vmax",
        "legend",
        "legend_kwargs",
        "legend_label",
        "alpha",
        "alpha_range",
        "clip_percentile",
        "max_scale",
        "normalize",
        "imshow_kwargs",
        # element dicts and previews
        "arrows_kwargs",
        "diff_arrows_kwargs",
        "diff_cmap",
        "diff_legend_kwargs",
        "heatmap",
        "heatmap_alpha_from_magnitude",
        "heatmap_cmap",
        "heatmap_kwargs",
        "heatmap_legend_kwargs",
        "heatmap_type",
        "image_kwargs",
        "input_angle",
        "input_arrows_kwargs",
        "quiver_kwargs",
        # velocity arrows
        "alpha_by_magnitude",
        "arrow_scale",
        "color",
        "color_by",
        "ref_magnitude",
        "skip",
        "velocity_scale",
        # series and timing
        "color_by_run",
        "duration",
        "fps",
        "hold_at_keyframes",
        "hold_frames",
        "interpolation",
        "keyframe_times",
        "position_mapper",
        "precompute",
        # layout, toggles and modulators
        "Dx",
        "Dy",
        "area_scale",
        "column",
        "figsize",
        "grid_size",
        "margin",
        "show",
        "show_axes",
        "show_both",
        "show_density",
        "show_geometry",
        "show_recompute",
        "show_run_boundaries",
        "show_run_info",
        "show_tolerance",
        "show_vectors",
        "show_velocity",
        "title",
        "use_pct",
    },
}

# Names that were replaced; they must not come back as parameters.
REMOVED = {
    "colorbar",
    "colorbar_kwargs",
    "colorbar_label",
    "color_values",
    "show_colorbar",
    "diff_colorbar_kwargs",
    "heatmap_colorbar_kwargs",
}


def _parameters(module: str) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in sorted((SRC / module).glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and not node.name.startswith("_")
                and node.name.startswith(("plot", "animate", "preview"))
            ):
                for arg in node.args.args + node.args.kwonlyargs:
                    if arg.arg not in ("self", "cls"):
                        found.setdefault(arg.arg, set()).add(f"{path.stem}.{node.name}")
            if isinstance(node, ast.ClassDef) and node.name.endswith("PlotOptions"):
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        found.setdefault(item.target.id, set()).add(f"{path.stem}.{node.name}")
    return found


@pytest.mark.parametrize("module", sorted(ALLOWED))
def test_parameters_are_in_the_vocabulary(module):
    found = _parameters(module)
    unknown = {name: sorted(where) for name, where in found.items() if name not in ALLOWED[module]}
    assert not unknown, f"parameters outside the vocabulary of {module}: {unknown}"
    assert not (set(found) & REMOVED), f"replaced names are back: {sorted(set(found) & REMOVED)}"
