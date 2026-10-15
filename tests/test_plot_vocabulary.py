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
    "symbol_cartogram": {
        # objects and data
        "ax",
        "cartogram",
        "result",
        "original_gdf",
        "source",
        "source_gdf",
        "adjacency",
        "sort_by",
        # fill, edge and the other channels (the vocabulary names)
        "facecolor",
        "cmap",
        "norm",
        "vmin",
        "vmax",
        "alpha",
        "alpha_range",
        "edgecolor",
        "edge_cmap",
        "linewidth",
        "linewidth_range",
        "hatch",
        "hatch_map",
        "labels",
        "label_color",
        "label_cmap",
        "label_fontsize",
        "label_fontsize_range",
        "label_kwargs",
        # guides
        "legend",
        "legend_kwargs",
        "edge_legend",
        "edge_legend_kwargs",
        "hatch_legend",
        "hatch_legend_kwargs",
        "linewidth_legend",
        "linewidth_legend_kwargs",
        "alpha_legend",
        "alpha_legend_kwargs",
        "label_legend",
        "label_legend_kwargs",
        # layers: tiles, adjacency links and nodes, arrows
        "assigned_facecolor",
        "unassigned_facecolor",
        "tile_edgecolor",
        "tile_linewidth",
        "tile_alpha",
        "core_edgecolor",
        "ring_edgecolor",
        "pool_linewidth",
        "link_color",
        "link_cmap",
        "link_alpha",
        "link_linewidth",
        "link_legend",
        "link_legend_kwargs",
        "node_markersize",
        "arrows_kwargs",
        "arrow_scale",
        # modes, toggles and layout
        "color_by",
        "figsize",
        "show_assigned",
        "show_labels",
        "show_original",
        "show_pool",
        "show_symbols",
        "show_unassigned",
        "show_values",
        "show_vertex_labels",
        "show_vertices",
        "tick_labelsize",
        "title",
        "use_original_positions",
        "value_fmt",
        "zorder",
    },
}

# Names that were replaced; they must not come back as parameters.
REMOVED = {
    "flow_cartogram": {
        "colorbar",
        "colorbar_kwargs",
        "colorbar_label",
        "color_values",
        "show_colorbar",
        "diff_colorbar_kwargs",
        "heatmap_colorbar_kwargs",
    },
    "symbol_cartogram": {
        "column",
        "colorbar",
        "colorbar_kwds",
        "legend_kwds",
        "edge_legend_kwds",
        "hatch_legend_kwds",
        "linewidth_legend_kwds",
        "alpha_legend_kwds",
        "label_legend_kwds",
        "label",
        "face_color",
        "edge_color",
        "edge_alpha",
        "edge_width",
        "node_size",
        "colormap",
        "arrow_color",
        "arrow_alpha",
        "assigned_color",
        "unassigned_color",
        "tick_fontsize",
    },
}


def _parameters(module: str) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for path in sorted((SRC / module).rglob("*.py")):
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
    back = set(found) & REMOVED[module]
    assert not back, f"replaced names are back: {sorted(back)}"
