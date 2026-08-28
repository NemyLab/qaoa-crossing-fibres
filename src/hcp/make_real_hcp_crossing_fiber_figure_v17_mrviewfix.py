#!/usr/bin/env python3
"""Create the real-HCP crossing-fiber paper figure as a direct v17 adaptation.

The panel structure, palette, typography, line styling, 2×4 full-figure layout,
individual-panel export, and PNG/PDF/SVG output behavior are retained from
make_crossing_fiber_figure_v17.py.

Only the data-loading layer and panel A are changed for the real-data analysis:

A. Real diffusion-MRI anatomical context
B. Unlabelled diffusion-peak problem setting
C. Sparse Ising graph after RCM relabelling
D. Exact graph assignment from a classical MILP solver
E. Exact fiber assignment mapped back to voxel space
F. QAOA/QPU energy sampling distribution
G. Best measured QPU graph assignment
H. Best measured QPU fiber assignment

Default QPU result: hardware-calibrated p=1 angles, fractional gates,
XY4 dynamical decoupling, and 10,000 IBM Aachen shots.

Required project modules in the current directory or PYTHONPATH:
    step1_crossing_graph.py
    step5_curved_signal_graph.py
    step9_common.py
"""
from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from step1_crossing_graph import CrossingGraph
from step5_curved_signal_graph import exact_milp
from step9_common import qiskit_bitstring_to_spins


# =============================================================================
# EDITABLE STYLE SETTINGS
# =============================================================================
PALETTE = {
    "bundle_pos": "#0072B2",       # blue
    "bundle_neg": "#D55E00",       # vermillion
    "unlabelled_peak": "#4D4D4D",
    "continuation": "#4C78A8",
    "conflict": "#D62728",
    "violated": "#B8B8B8",
    "hardware": "#009E73",         # green
    "uniform": "#8C8C8C",
    "exact": "#111111",
    "mismatch": "#222222",         # dark marker for assignment differences
    "anchor": "#111111",
    "voxel": "#ECECEC",
    "node_face": "#F2F2F2",
}

PANEL_TITLES = {
    "A": "Real diffusion-MRI patch",
    "B": "Unlabelled diffusion peaks",
    "C": "Sparse Ising graph (RCM relabelled)",
    "D": "Exact graph assignment (MILP)",
    "E": "Exact fiber assignment",
    "F": "QAOA sampling on the QPU",
    "G": "Best measured QPU graph",
    "H": "Best measured QPU fiber assignment",
}

FIGURE_DPI = 300
PEAK_HALF_LENGTH = 0.34
PEAK_LINEWIDTH = 2.25
GRAPH_NODE_OFFSET_X = 0.20  # horizontal display offset within a voxel
GRAPH_NODE_OFFSET_Y = 0.10  # vertical display offset within a voxel
GRAPH_NODE_SIZE = 54
GRAPH_GLYPH_HALF_LENGTH = 0.075
EDGE_WIDTH_SCALE = 2.6
MISMATCH_MARKER_SIZE = 64
MISMATCH_MARKER_LINEWIDTH = 1.4
MISMATCH_STAR_OFFSET_X = -0.07  # reflected with the MRView left--right display
MISMATCH_STAR_OFFSET_Y = 0.05
ANCHOR_MARKER_SIZE = 30
ODF_SPHERE_RADIUS = 0.34
ODF_SPHERE_SPACING = 1.0
ODF_SURFACE_U = 64
ODF_SURFACE_V = 64
SPHERE_PATTERN_ROTATION_DEG = 20.0  # rotate only the colored diffusion pattern


@dataclass(frozen=True)
class NodeGeometry:
    x: np.ndarray
    y: np.ndarray
    dx: np.ndarray
    dy: np.ndarray
    peak_index: np.ndarray
    voxel_id: np.ndarray


# =============================================================================
# DATA AND GRAPH HELPERS
# =============================================================================
def _required_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Could not locate {label}: {path}")
    return path


def _required_column(frame: pd.DataFrame, aliases: Iterable[str]) -> str:
    lookup = {str(column).lower(): str(column) for column in frame.columns}
    for alias in aliases:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    raise ValueError(
        f"Could not identify any of columns {list(aliases)} in "
        f"{list(frame.columns)}"
    )


def _normalise_vectors(dx: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    norm = np.hypot(dx, dy)
    norm[norm == 0] = 1.0
    return dx / norm, dy / norm


def load_real_graph(graph_dir: Path) -> tuple[CrossingGraph, NodeGeometry, dict[str, Any]]:
    """Load the already constructed real-data graph in its RCM node order."""
    nodes_path = _required_file(
        graph_dir / "real_graph_nodes_rcm.csv",
        "RCM node table",
    )
    edges_path = _required_file(
        graph_dir / "real_graph_edges_rcm.csv",
        "RCM edge table",
    )
    arrays_path = _required_file(
        graph_dir / "real_graph_solution_arrays.npz",
        "real graph solution arrays",
    )
    validation_path = _required_file(
        graph_dir / "real_graph_validation.json",
        "real graph validation JSON",
    )

    nodes = pd.read_csv(nodes_path).sort_values("node_id").reset_index(drop=True)
    edges = pd.read_csv(edges_path)
    validation = json.loads(validation_path.read_text(encoding="utf-8"))

    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}

    required_node_columns = {"node_id", "old_node_id", "true_spin"}
    missing = required_node_columns.difference(nodes.columns)
    if missing:
        raise ValueError(
            f"RCM node table is missing columns: {sorted(missing)}"
        )

    old_ids = nodes["old_node_id"].to_numpy(dtype=int)
    fields_pre_rcm = np.asarray(arrays["fields_pre_rcm"], dtype=float)
    fields_rcm = fields_pre_rcm[old_ids]
    true_spins_rcm = nodes["true_spin"].to_numpy(dtype=np.int8)

    graph = CrossingGraph(
        nodes=nodes,
        edges=edges,
        fields=fields_rcm,
        true_spins=true_spins_rcm,
        grid_size=5,
        seed=100307,
    )
    if graph.n_nodes != 50:
        raise RuntimeError(f"Expected 50 nodes, found {graph.n_nodes}")

    row_col = _required_column(nodes, ["row", "grid_row", "voxel_row"])
    col_col = _required_column(nodes, ["col", "column", "grid_col", "voxel_col"])
    dx_col = _required_column(nodes, ["dir_x", "dx", "peak_x"])
    dz_col = _required_column(nodes, ["dir_z", "dz", "peak_z"])
    peak_col = (
        _required_column(nodes, ["peak", "peak_index", "candidate_index"])
        if any(
            alias in {str(column).lower() for column in nodes.columns}
            for alias in ("peak", "peak_index", "candidate_index")
        )
        else None
    )
    voxel_col = (
        _required_column(nodes, ["voxel", "voxel_id", "site"])
        if any(
            alias in {str(column).lower() for column in nodes.columns}
            for alias in ("voxel", "voxel_id", "site")
        )
        else None
    )

    rows = nodes[row_col].to_numpy(dtype=float)
    cols = nodes[col_col].to_numpy(dtype=float)

    # Match the validated graph plot vertically (row 0 at the top) and apply
    # only the left-right reflection needed for MRView radiological display.
    # The corresponding direction transform is (-dir_x, +dir_z).
    x = float(np.max(cols)) - cols
    y = float(np.max(rows)) - rows

    dx = -nodes[dx_col].to_numpy(dtype=float)
    dy = nodes[dz_col].to_numpy(dtype=float)
    dx, dy = _normalise_vectors(dx, dy)

    peak_index = (
        nodes[peak_col].to_numpy(dtype=int)
        if peak_col is not None
        else nodes["old_node_id"].to_numpy(dtype=int) % 2
    )
    voxel_id = (
        nodes[voxel_col].to_numpy(dtype=int)
        if voxel_col is not None
        else nodes["old_node_id"].to_numpy(dtype=int) // 2
    )

    geometry = NodeGeometry(
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        peak_index=peak_index,
        voxel_id=voxel_id,
    )
    return graph, geometry, validation


def load_fractional_results(
    hardware_dir: Path,
    strategy: str,
) -> tuple[dict[str, Any], dict[str, int], np.ndarray]:
    results_path = _required_file(
        hardware_dir / "real_fractional_hardware_results.json",
        "fractional-gate hardware result JSON",
    )
    counts_path = _required_file(
        hardware_dir / "real_fractional_hardware_counts.json.gz",
        "fractional-gate hardware counts",
    )
    assignments_path = _required_file(
        hardware_dir / "real_fractional_best_assignments.npz",
        "fractional-gate best assignments",
    )

    results = json.loads(results_path.read_text(encoding="utf-8"))
    matches = [
        row for row in results["strategies"]
        if str(row["angle_strategy"]) == str(strategy)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one strategy row for {strategy!r}; found {len(matches)}"
        )
    row = matches[0]

    with gzip.open(counts_path, "rt", encoding="utf-8") as handle:
        count_payload = json.load(handle)
    if strategy not in count_payload:
        raise KeyError(
            f"Strategy {strategy!r} was not found in {counts_path}"
        )
    counts = {
        str(bitstring): int(count)
        for bitstring, count in count_payload[strategy].items()
    }

    raw_key = f"{strategy}_best_spins_rcm"
    with np.load(assignments_path, allow_pickle=False) as archive:
        if raw_key not in archive.files:
            raise KeyError(
                f"{raw_key!r} was not found in {assignments_path}; "
                f"available keys: {archive.files}"
            )
        best_raw = np.asarray(archive[raw_key], dtype=np.int8)

    if best_raw.shape != (50,):
        raise ValueError(
            f"Expected best RCM assignment shape (50,), found {best_raw.shape}"
        )
    return row, counts, best_raw


def energy_from_graph(graph, spins: np.ndarray) -> float:
    return float(graph.energy(np.asarray(spins, dtype=np.int8)))


def expand_hardware_energies(graph, counts: dict[str, int]) -> np.ndarray:
    energies: list[float] = []
    for bitstring, count in counts.items():
        spins = qiskit_bitstring_to_spins(bitstring, graph.n_nodes)
        energies.extend([energy_from_graph(graph, spins)] * int(count))
    return np.asarray(energies, dtype=float)


def uniform_energies(graph, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n, graph.n_nodes), dtype=np.int8)
    spins = 1 - 2 * bits
    return np.asarray(
        [energy_from_graph(graph, state) for state in spins],
        dtype=float,
    )


def align_for_display(
    spins: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Globally align color names without changing the recorded energy."""
    spins = np.asarray(spins, dtype=np.int8)
    reference = np.asarray(reference, dtype=np.int8)
    direct = np.mean(spins == reference)
    flipped = np.mean(-spins == reference)
    return (-spins, True) if flipped > direct else (spins.copy(), False)


def graph_node_coordinates(geometry: NodeGeometry) -> np.ndarray:
    """Separate each candidate pair using the MRView-mirrored v17 offset.

    The voxel centres and peak directions have already been reflected
    left--right in ``load_real_graph``.  The horizontal component of the
    within-voxel node offset must therefore be reflected as well; otherwise
    the two candidates are swapped spatially inside every voxel and diagonal
    continuation edges appear to attach to the wrong side of the pair.
    """
    sign = np.where(geometry.peak_index % 2 == 0, -1.0, 1.0)
    return np.column_stack([
        geometry.x - sign * GRAPH_NODE_OFFSET_X,
        geometry.y - sign * GRAPH_NODE_OFFSET_Y,
    ])


def edge_records(graph):
    required = {"i", "j", "weight", "edge_type"}
    if not required.issubset(set(graph.edges.columns)):
        raise RuntimeError(
            f"graph.edges must contain {sorted(required)}; "
            f"found {list(graph.edges.columns)}"
        )
    return list(graph.edges.itertuples(index=False))


def edge_is_satisfied(edge, spins: np.ndarray) -> bool:
    same = spins[int(edge.i)] == spins[int(edge.j)]
    return bool(
        same if edge.edge_type == "continuation" else not same
    )


def anchor_indices(graph) -> np.ndarray:
    fields = np.asarray(
        getattr(graph, "fields", np.zeros(graph.n_nodes)),
        dtype=float,
    )
    return np.flatnonzero(np.abs(fields) > 1e-12)


# =============================================================================
# DRAWING HELPERS
# =============================================================================
def save_figure(fig: plt.Figure, base: Path, formats: Iterable[str]) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        fig.savefig(base.with_suffix(f".{extension}"), dpi=FIGURE_DPI, bbox_inches="tight")


def panel_label(ax, label: str) -> None:
    if getattr(ax, "name", "") == "3d":
        ax.text2D(
            -0.08, 1.06, label, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left"
        )
    else:
        ax.text(
            -0.08, 1.06, label, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="left"
        )

def style_spatial_axis(ax: plt.Axes, geometry: NodeGeometry) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(geometry.x.min() - 0.78, geometry.x.max() + 0.78)
    ax.set_ylim(geometry.y.min() - 0.78, geometry.y.max() + 0.78)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_voxel_lattice(ax: plt.Axes, geometry: NodeGeometry) -> None:
    unique_xy = np.unique(np.column_stack([geometry.x, geometry.y]), axis=0)
    ax.scatter(
        unique_xy[:, 0],
        unique_xy[:, 1],
        marker="s",
        s=405,
        facecolor=PALETTE["voxel"],
        edgecolor="white",
        linewidth=1.0,
        zorder=0,
    )


def draw_peak_segments(
    ax: plt.Axes,
    geometry: NodeGeometry,
    spins: np.ndarray | None = None,
    linewidth: float = PEAK_LINEWIDTH,
) -> None:
    for i in range(len(geometry.x)):
        color = (
            PALETTE["unlabelled_peak"]
            if spins is None
            else PALETTE["bundle_pos"] if spins[i] > 0 else PALETTE["bundle_neg"]
        )
        x0 = geometry.x[i] - PEAK_HALF_LENGTH * geometry.dx[i]
        x1 = geometry.x[i] + PEAK_HALF_LENGTH * geometry.dx[i]
        y0 = geometry.y[i] - PEAK_HALF_LENGTH * geometry.dy[i]
        y1 = geometry.y[i] + PEAK_HALF_LENGTH * geometry.dy[i]
        ax.plot(
            [x0, x1],
            [y0, y1],
            color=color,
            lw=linewidth,
            solid_capstyle="round",
            zorder=2,
        )


def draw_orientation_glyphs(ax: plt.Axes, geometry: NodeGeometry, coords: np.ndarray) -> None:
    for i, (cx, cy) in enumerate(coords):
        ax.plot(
            [cx - GRAPH_GLYPH_HALF_LENGTH * geometry.dx[i], cx + GRAPH_GLYPH_HALF_LENGTH * geometry.dx[i]],
            [cy - GRAPH_GLYPH_HALF_LENGTH * geometry.dy[i], cy + GRAPH_GLYPH_HALF_LENGTH * geometry.dy[i]],
            color="#222222",
            lw=0.9,
            zorder=5,
        )


def draw_mismatch_circles(ax: plt.Axes, coords: np.ndarray) -> None:
    ax.scatter(
        coords[:, 0], coords[:, 1],
        s=MISMATCH_MARKER_SIZE,
        facecolor="none",
        edgecolor=PALETTE["mismatch"],
        linewidth=MISMATCH_MARKER_LINEWIDTH,
        zorder=7,
    )



def draw_mri_context_panel(
    ax: plt.Axes,
    screenshot_path: Path,
    subject_label: str,
    patch_label: str,
) -> None:
    """Draw the real anatomical context without changing the v17 styling."""
    image = plt.imread(screenshot_path)
    ax.imshow(image)
    ax.set_title(PANEL_TITLES["A"], loc="left", pad=9)
    ax.set_axis_off()
    ax.text(
        0.02,
        0.02,
        f"{subject_label}; {patch_label}",
        transform=ax.transAxes,
        fontsize=7.3,
        va="bottom",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "alpha": 0.88,
            "edgecolor": "none",
        },
    )


def draw_odf_panel(ax, geometry: NodeGeometry) -> None:
    """Draw a front-facing 5×5 grid of equal-size colorful spheres.

    Each sphere is a true 3D object with color mapped on its surface according to
    a schematic local angular diffusion profile. The sphere centres lie in a flat
    5×5 grid so none of the spheres recede behind the others.
    """
    ax.set_title(PANEL_TITLES["A"], loc="left", pad=10)
    ax.set_axis_off()

    u = np.linspace(0.0, 2.0 * np.pi, ODF_SURFACE_U)
    v = np.linspace(0.0, np.pi, ODF_SURFACE_V)
    uu, vv = np.meshgrid(u, v)
    sx = np.cos(uu) * np.sin(vv)
    sy = np.sin(uu) * np.sin(vv)
    sz = np.cos(vv)

    x_values = np.sort(np.unique(geometry.x))
    y_values = np.sort(np.unique(geometry.y))
    x_mid = float(np.mean(x_values))
    y_mid = float(np.mean(y_values))
    cmap = cm.get_cmap("turbo")
    colour_norm = colors.Normalize(vmin=0.0, vmax=1.0)

    # Sphere centres are placed on an x-z grid. The y-coordinate is kept at zero,
    # so the array is front-facing and no row hides behind another one.
    for gx in x_values:
        for gy in y_values:
            indices = np.flatnonzero(
                (np.abs(geometry.x - gx) < 1e-9)
                & (np.abs(geometry.y - gy) < 1e-9)
            )
            if len(indices) != 2:
                continue

            profile = np.full_like(sx, 0.10)
            for idx in indices:
                px = float(geometry.dx[idx])
                py = float(geometry.dy[idx])

                # Rotate only the colored diffusion-profile pattern. The sphere
                # geometry and the front-facing 5x5 layout remain unchanged.
                theta = np.deg2rad(SPHERE_PATTERN_ROTATION_DEG)
                px_rot = np.cos(theta) * px - np.sin(theta) * py
                py_rot = np.sin(theta) * px + np.cos(theta) * py

                # Map the rotated peak directions into the visible front-facing
                # disc of the sphere so both maxima can be displayed clearly.
                dot = np.abs(sy * px_rot + sz * py_rot)
                profile += np.exp(-0.5 * ((1.0 - dot) / 0.10) ** 2)
            profile = (profile - profile.min()) / max(
                float(profile.max() - profile.min()), 1e-12
            )

            cx = (float(gx) - x_mid) * ODF_SPHERE_SPACING
            cz = (y_mid - float(gy)) * ODF_SPHERE_SPACING
            X = cx + ODF_SPHERE_RADIUS * sx
            Y = 0.0 + ODF_SPHERE_RADIUS * sy
            Z = cz + ODF_SPHERE_RADIUS * sz

            ax.plot_surface(
                X, Y, Z,
                facecolors=cmap(colour_norm(profile)),
                linewidth=0.0,
                antialiased=True,
                shade=False,
                alpha=0.98,
            )

    half = 0.5 * (len(x_values) - 1) * ODF_SPHERE_SPACING + 0.55
    ax.set_xlim(-half, half)
    ax.set_ylim(-0.52, 0.52)
    ax.set_zlim(-half, half)
    ax.set_box_aspect((1.0, 0.22, 1.0))
    ax.set_proj_type('ortho')
    ax.view_init(elev=0, azim=-90)

    scalar_map = cm.ScalarMappable(norm=colour_norm, cmap=cmap)
    scalar_map.set_array([])
    colorbar = ax.figure.colorbar(
        scalar_map, ax=ax, orientation='horizontal',
        fraction=0.055, pad=0.02, shrink=0.78
    )
    colorbar.set_label('Relative angular diffusion-profile value', fontsize=7.2)
    colorbar.ax.tick_params(labelsize=6.6, length=2)

    ax.text2D(
        0.03, 0.02,
        'A front-facing 5×5 grid of equal-size spheres; surface color shows the local diffusion profile',
        transform=ax.transAxes, fontsize=7.1, va='bottom', ha='left',
        bbox={'boxstyle': 'round,pad=0.22', 'facecolor': 'white', 'alpha': 0.90, 'edgecolor': 'none'},
    )


def draw_problem_setting(ax: plt.Axes, geometry: NodeGeometry) -> None:
    draw_voxel_lattice(ax, geometry)
    draw_peak_segments(ax, geometry, spins=None)
    ax.set_title(PANEL_TITLES["B"], loc="left", pad=9)
    ax.text(
        0.02,
        0.02,
        "Two unlabelled peak axes per voxel",
        transform=ax.transAxes,
        fontsize=7.6,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
    )
    style_spatial_axis(ax, geometry)


def draw_generated_graph(ax: plt.Axes, graph, geometry: NodeGeometry) -> None:
    coords = graph_node_coordinates(geometry)
    continuation_segments, continuation_widths = [], []
    conflict_segments, conflict_widths = [], []
    for edge in edge_records(graph):
        segment = [coords[int(edge.i)], coords[int(edge.j)]]
        width = 0.45 + EDGE_WIDTH_SCALE * float(edge.weight)
        if edge.edge_type == "continuation":
            continuation_segments.append(segment)
            continuation_widths.append(width)
        else:
            conflict_segments.append(segment)
            conflict_widths.append(width)

    if continuation_segments:
        ax.add_collection(LineCollection(
            continuation_segments,
            colors=PALETTE["continuation"],
            linewidths=continuation_widths,
            alpha=0.47,
            zorder=1,
        ))
    if conflict_segments:
        # Solid centre-to-centre connectors avoid the asymmetric dash-phase
        # artifact that made one end appear detached from its node.
        ax.add_collection(LineCollection(
            conflict_segments,
            colors=PALETTE["conflict"],
            linewidths=conflict_widths,
            alpha=0.72,
            zorder=1.5,
        ))

    ax.scatter(
        coords[:, 0], coords[:, 1],
        s=GRAPH_NODE_SIZE,
        facecolor=PALETTE["node_face"],
        edgecolor="#222222",
        linewidth=0.8,
        zorder=3,
    )
    draw_orientation_glyphs(ax, geometry, coords)
    anchors = anchor_indices(graph)
    if anchors.size:
        ax.scatter(
            coords[anchors, 0], coords[anchors, 1],
            marker="*", s=ANCHOR_MARKER_SIZE,
            facecolor=PALETTE["anchor"], edgecolor="none", zorder=6,
        )

    n_cont = int(np.sum(graph.edges["edge_type"] == "continuation"))
    n_conf = int(np.sum(graph.edges["edge_type"] == "conflict"))
    ax.text(
        0.02, 0.02,
        f"{graph.n_nodes} nodes; {n_cont} continuation; {n_conf} conflict edges",
        transform=ax.transAxes,
        fontsize=7.2,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
    )
    ax.set_title(PANEL_TITLES["C"], loc="left", pad=9)
    style_spatial_axis(ax, geometry)
    ax.legend(
        handles=[
            Line2D([0], [0], color=PALETTE["continuation"], lw=2, label="Continuation"),
            Line2D([0], [0], color=PALETTE["conflict"], lw=2, label="Within-voxel conflict"),
            Line2D([0], [0], marker="*", color="none", markerfacecolor=PALETTE["anchor"], label="Anchor field", markersize=7),
        ],
        frameon=False,
        fontsize=6.5,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.14),
        borderaxespad=0.0,
        ncol=1,
        handlelength=1.8,
        labelspacing=0.3,
    )


def draw_assignment_graph(
    ax: plt.Axes,
    graph,
    geometry: NodeGeometry,
    spins_for_display: np.ndarray,
    title: str,
    comparison: np.ndarray | None = None,
    annotation: str | None = None,
) -> None:
    coords = graph_node_coordinates(geometry)
    sat_cont, sat_cont_w = [], []
    sat_conf, sat_conf_w = [], []
    violated, violated_w = [], []

    for edge in edge_records(graph):
        segment = [coords[int(edge.i)], coords[int(edge.j)]]
        width = 0.40 + EDGE_WIDTH_SCALE * float(edge.weight)
        if edge_is_satisfied(edge, spins_for_display):
            if edge.edge_type == "continuation":
                sat_cont.append(segment)
                sat_cont_w.append(width)
            else:
                sat_conf.append(segment)
                sat_conf_w.append(width)
        else:
            violated.append(segment)
            violated_w.append(width)

    if sat_cont:
        ax.add_collection(LineCollection(
            sat_cont, colors=PALETTE["continuation"], linewidths=sat_cont_w,
            alpha=0.48, zorder=1,
        ))
    if sat_conf:
        ax.add_collection(LineCollection(
            sat_conf, colors=PALETTE["conflict"], linewidths=sat_conf_w,
            alpha=0.72, zorder=1.5,
        ))
    if violated:
        ax.add_collection(LineCollection(
            violated, colors=PALETTE["violated"], linewidths=violated_w,
            linestyles="dotted", alpha=0.95, zorder=1,
        ))

    node_colors = np.where(
        spins_for_display > 0, PALETTE["bundle_pos"], PALETTE["bundle_neg"]
    )
    ax.scatter(
        coords[:, 0], coords[:, 1],
        s=GRAPH_NODE_SIZE + 5,
        c=node_colors,
        edgecolor="#222222",
        linewidth=0.8,
        zorder=3,
    )
    draw_orientation_glyphs(ax, geometry, coords)

    anchors = anchor_indices(graph)
    if anchors.size:
        ax.scatter(
            coords[anchors, 0], coords[anchors, 1],
            marker="*", s=ANCHOR_MARKER_SIZE,
            facecolor=PALETTE["anchor"], edgecolor="none", zorder=6,
        )

    if comparison is not None:
        mismatch = np.asarray(spins_for_display) != np.asarray(comparison)
        if np.any(mismatch):
            for x, y in coords[mismatch]:
                ax.text(
                    x + MISMATCH_STAR_OFFSET_X,
                    y + MISMATCH_STAR_OFFSET_Y,
                    "*",
                    color=PALETTE["mismatch"],
                    fontsize=13,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=7,
                )

    if annotation:
        ax.text(
            0.02, 0.02, annotation,
            transform=ax.transAxes,
            fontsize=7.3,
            va="bottom",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
        )
    ax.set_title(title, loc="left", pad=9)
    style_spatial_axis(ax, geometry)


def draw_fiber_assignment(
    ax: plt.Axes,
    geometry: NodeGeometry,
    spins: np.ndarray,
    title: str,
    comparison: np.ndarray | None = None,
    annotation: str | None = None,
) -> None:
    draw_voxel_lattice(ax, geometry)
    draw_peak_segments(ax, geometry, spins=spins, linewidth=2.45)

    if comparison is not None:
        mismatch = np.asarray(spins) != np.asarray(comparison)
        coords = graph_node_coordinates(geometry)
        if np.any(mismatch):
            for x, y in coords[mismatch]:
                ax.text(
                    x + MISMATCH_STAR_OFFSET_X,
                    y + MISMATCH_STAR_OFFSET_Y,
                    "*",
                    color=PALETTE["mismatch"],
                    fontsize=13,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=6,
                )

    if annotation:
        ax.text(
            0.02, 0.02, annotation,
            transform=ax.transAxes,
            fontsize=7.3,
            va="bottom",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.88, "edgecolor": "none"},
        )
    ax.set_title(title, loc="left", pad=9)
    style_spatial_axis(ax, geometry)


def draw_energy_distribution(
    ax: plt.Axes,
    hardware_energies: np.ndarray,
    uniform: np.ndarray,
    exact_energy: float,
    best_hardware_energy: float,
    mean_hardware_energy: float,
    n_unique: int,
) -> None:
    """Plot the complete sampled energy range on a linear density scale."""
    hardware_energies = np.asarray(hardware_energies, dtype=float)
    uniform = np.asarray(uniform, dtype=float)
    combined = np.concatenate([hardware_energies, uniform])

    data_min = min(float(np.min(combined)), float(best_hardware_energy))
    data_max = float(np.max(combined))
    span = max(data_max - data_min, 1e-9)
    nominal_width = span / 41.0

    # Half-bin padding ensures that the observed minimum and maximum lie inside
    # the first and last bins instead of exactly on the histogram boundaries.
    bins = np.linspace(
        data_min - 0.5 * nominal_width,
        data_max + 0.5 * nominal_width,
        43,
    )

    ax.hist(
        uniform,
        bins=bins,
        density=True,
        alpha=0.40,
        color=PALETTE["uniform"],
        label="Uniform complete assignments",
    )
    ax.hist(
        hardware_energies,
        bins=bins,
        density=True,
        alpha=0.68,
        color=PALETTE["hardware"],
        label="Measured QAOA bitstrings",
    )

    ax.axvline(
        exact_energy,
        color=PALETTE["exact"],
        lw=2.0,
        label="Exact MILP optimum",
        zorder=6,
    )
    ax.axvline(
        best_hardware_energy,
        color=PALETTE["bundle_neg"],
        lw=2.0,
        ls="--",
        label="Best measured QPU bitstring",
        zorder=7,
    )
    ax.axvline(
        mean_hardware_energy,
        color=PALETTE["hardware"],
        lw=1.4,
        ls=":",
        label="QPU mean",
        zorder=6,
    )

    # Retain the exact optimum even when it lies left of all sampled states.
    x_left = min(float(bins[0]), float(exact_energy))
    x_right = float(bins[-1])
    x_pad = 0.025 * max(x_right - x_left, 1.0)
    ax.set_xlim(x_left - x_pad, x_right + x_pad)

    # Inline labels for the three reference energies so each vertical line can
    # be identified directly in the panel.
    ymax = float(ax.get_ylim()[1])
    xspan = max(float(ax.get_xlim()[1] - ax.get_xlim()[0]), 1e-9)
    label_specs = [
        (exact_energy, 0.58, f"E={exact_energy:.2f}", PALETTE["exact"]),
        (best_hardware_energy, 0.74, f"E={best_hardware_energy:.2f}", PALETTE["bundle_neg"]),
    ]
    for xval, yfrac, label, color in label_specs:
        xoff = 0.016 * xspan
        if xval > (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2.0:
            xoff = -xoff
            halign = "right"
        else:
            halign = "left"
        ax.text(
            xval + xoff,
            yfrac * ymax,
            label,
            rotation=90,
            rotation_mode="anchor",
            color=color,
            fontsize=7.0,
            ha=halign,
            va="center",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            zorder=8,
        )

    ax.set_xlabel("Ising energy (lower is better)")
    ax.set_ylabel("Density")
    ax.set_title(f"QAOA sampling on the QPU ({hardware_energies.size:,} shots)", loc="left", pad=9)
    ax.legend(
        frameon=False,
        fontsize=6.5,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.16),
        borderaxespad=0.0,
        ncol=1,
        handlelength=1.8,
        labelspacing=0.3,
    )
    ax.spines[["top", "right"]].set_visible(False)


def configure_matplotlib(use_latex: bool) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10.2,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.usetex": bool(use_latex),
    })


def _candidate_data_paths(path: Path) -> list[Path]:
    """Return plausible locations for a packaged or project-local data file."""
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    candidates = [
        path,
        cwd / path,
        script_dir / path,
        cwd / "data" / path.name,
        script_dir / "data" / path.name,
        script_dir / path.name,
        script_dir.parent / "data" / path.name,
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            unique.append(resolved)
    return unique


def resolve_required_data_file(path: Path, label: str) -> Path:
    for candidate in _candidate_data_paths(path):
        if candidate.is_file():
            return candidate
    searched = "\n  - ".join(str(p) for p in _candidate_data_paths(path))
    raise FileNotFoundError(
        f"Could not locate {label}. Searched:\n  - {searched}\n"
        "Use the complete figure package or pass the file explicitly."
    )


def resolve_optional_data_file(path: Path) -> Path | None:
    for candidate in _candidate_data_paths(path):
        if candidate.is_file():
            return candidate
    return None


EMBEDDED_HARDWARE_ROWS: dict[tuple[int, str], dict[str, object]] = {
    (2010, "hardware_calibrated"): {
        "pub_index": 12,
        "seed": 2010,
        "angle_strategy": "hardware_calibrated",
        "exact_energy": -35.02675522569289,
        "exact_solution_label_accuracy": 1.0,
        "shots": 5000,
        "n_unique_bitstrings": 5000,
        "mean_energy": -18.22019077124083,
        "best_bitstring": "10100111000110000110011010111011000011100110110010",
        "best_energy": -29.85527625721178,
        "best_normalized_gap": 0.14764367795871983,
        "best_label_accuracy": 0.32,
    }
}


def load_hardware_row_with_fallback(
    results_csv: Path | None, seed: int, strategy: str
) -> pd.Series:
    if results_csv is not None and results_csv.is_file():
        return load_hardware_row(results_csv, seed, strategy)
    embedded = EMBEDDED_HARDWARE_ROWS.get((int(seed), str(strategy)))
    if embedded is not None:
        print(
            f"Note: replication CSV not found; using the embedded recorded "
            f"hardware summary for seed {seed}."
        )
        return pd.Series(embedded)
    raise FileNotFoundError(
        f"No hardware-results file or embedded fallback is available for "
        f"seed {seed}, strategy {strategy!r}."
    )


def select_hardware_files(
    seed: int,
    heldout_results: Path,
    heldout_counts: Path,
    replication_results: Path,
    replication_counts: Path,
) -> tuple[Path, Path]:
    """Select the matching recorded QPU dataset for a seed."""
    if 2001 <= int(seed) <= 2003:
        return heldout_results, heldout_counts
    if 2004 <= int(seed) <= 2015:
        return replication_results, replication_counts
    raise ValueError(
        f"No packaged hardware data are available for seed {seed}. "
        "Use one of seeds 2001-2015."
    )


def prepare_seed_payload(
    seed: int,
    strategy: str,
    heldout_results: Path,
    heldout_counts: Path,
    replication_results: Path,
    replication_counts: Path,
    *,
    require_counts: bool,
):
    graph, geometry, diagnostics, rcm_order = build_graph(seed)
    results_path, counts_path = select_hardware_files(
        seed, heldout_results, heldout_counts,
        replication_results, replication_counts
    )

    if 2001 <= int(seed) <= 2003:
        resolved_results = resolve_required_data_file(
            results_path, "held-out hardware results"
        )
    else:
        resolved_results = resolve_optional_data_file(results_path)

    row = load_hardware_row_with_fallback(
        resolved_results, seed, strategy
    )

    if require_counts:
        resolved_counts = resolve_required_data_file(
            counts_path, "hardware bitstring counts"
        )
        counts = load_counts(resolved_counts, int(row["pub_index"]))
    else:
        counts = None

    exact_spins, exact_energy, solver_info = exact_milp(graph)
    exact_spins = np.asarray(exact_spins, dtype=np.int8)
    best_bitstring = str(row["best_bitstring"]).zfill(graph.n_nodes)
    best_raw = qiskit_bitstring_to_spins(best_bitstring, graph.n_nodes)
    best_display, display_flipped = align_for_display(best_raw, exact_spins)
    return {
        "graph": graph,
        "geometry": geometry,
        "diagnostics": diagnostics,
        "rcm_order": rcm_order,
        "row": row,
        "counts": counts,
        "exact_spins": exact_spins,
        "exact_energy": float(exact_energy),
        "solver_info": solver_info,
        "best_bitstring": best_bitstring,
        "best_display": best_display,
        "display_flipped": display_flipped,
    }


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=Path(
            "real_data/HCP_100307_R_89_89_80/graph_validation"
        ),
    )
    parser.add_argument(
        "--hardware-dir",
        type=Path,
        default=Path(
            "real_data/HCP_100307_R_89_89_80/"
            "hardware_results_fractional"
        ),
    )
    parser.add_argument(
        "--mrview-screenshot",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--strategy",
        default="hardware_calibrated",
        choices=["hardware_calibrated", "transferred_center"],
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("paper_figures_real_hcp_v17"),
    )
    parser.add_argument("--uniform-samples", type=int, default=10000)
    parser.add_argument("--uniform-seed", type=int, default=800307)
    parser.add_argument("--subject-label", default="HCP subject 100307")
    parser.add_argument(
        "--patch-label",
        default="right CST 5×5 crossing-fibre patch",
    )
    parser.add_argument("--latex", action="store_true")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf", "svg"],
        choices=["png", "pdf", "svg", "pgf"],
    )
    args = parser.parse_args()

    configure_matplotlib(args.latex)
    args.out.mkdir(parents=True, exist_ok=True)

    _required_file(args.mrview_screenshot, "MRView screenshot")
    graph, geometry, graph_validation = load_real_graph(args.graph_dir)
    row, counts, best_qpu_raw = load_fractional_results(
        args.hardware_dir,
        args.strategy,
    )

    exact_raw, exact_energy, solver_info = exact_milp(graph)
    exact_raw = np.asarray(exact_raw, dtype=np.int8)

    # Color naming is aligned first to the TractSeg-derived CST orientation
    # reference and then the QPU assignment is aligned to the exact assignment.
    # This changes only displayed color names, never the measured energies.
    exact_display, exact_display_flipped = align_for_display(
        exact_raw,
        np.asarray(graph.true_spins, dtype=np.int8),
    )
    best_qpu_display, qpu_display_flipped = align_for_display(
        best_qpu_raw,
        exact_display,
    )

    hardware_energies = expand_hardware_energies(graph, counts)
    random_energies = uniform_energies(
        graph,
        args.uniform_samples,
        seed=args.uniform_seed,
    )

    exact_tom_accuracy = float(
        max(
            np.mean(exact_raw == np.asarray(graph.true_spins)),
            np.mean(-exact_raw == np.asarray(graph.true_spins)),
        )
    )
    qpu_display_accuracy = float(
        np.mean(best_qpu_display == exact_display)
    )

    exact_annotation = (
        f"Exact energy = {float(exact_energy):.2f}; "
        f"agreement with CST reference = {exact_tom_accuracy:.0%}"
    )
    qpu_annotation = (
        f"Best measured gap = {float(row['best_normalized_gap']):.3f}; "
        f"agreement with exact = {qpu_display_accuracy:.0%}"
    )

    drawers = {
        "A": lambda ax: draw_mri_context_panel(
            ax,
            args.mrview_screenshot,
            args.subject_label,
            args.patch_label,
        ),
        "B": lambda ax: draw_problem_setting(ax, geometry),
        "C": lambda ax: draw_generated_graph(ax, graph, geometry),
        "D": lambda ax: draw_assignment_graph(
            ax,
            graph,
            geometry,
            exact_display,
            PANEL_TITLES["D"],
            annotation=exact_annotation,
        ),
        "E": lambda ax: draw_fiber_assignment(
            ax,
            geometry,
            exact_display,
            PANEL_TITLES["E"],
            annotation="MILP labels mapped back to the two measured CSD peak axes",
        ),
        "F": lambda ax: draw_energy_distribution(
            ax,
            hardware_energies,
            random_energies,
            float(exact_energy),
            float(row["best_energy"]),
            float(row["mean_energy"]),
            len(counts),
        ),
        "G": lambda ax: draw_assignment_graph(
            ax,
            graph,
            geometry,
            best_qpu_display,
            PANEL_TITLES["G"],
            comparison=exact_display,
            annotation=qpu_annotation,
        ),
        "H": lambda ax: draw_fiber_assignment(
            ax,
            geometry,
            best_qpu_display,
            PANEL_TITLES["H"],
            comparison=exact_display,
            annotation="Asterisks mark peaks differing from the exact assignment",
        ),
    }

    panel_filenames = {
        "A": "panel_A_real_mri_patch",
        "B": "panel_B_problem_setting",
        "C": "panel_C_generated_graph",
        "D": "panel_D_exact_graph",
        "E": "panel_E_exact_fiber_assignment",
        "F": "panel_F_qaoa_sampling",
        "G": "panel_G_best_qpu_graph",
        "H": "panel_H_best_qpu_fiber_assignment",
    }

    # Retain v17 individual-panel dimensions and export behavior.
    for label, drawer in drawers.items():
        figsize = (5.0, 3.6) if label == "F" else (4.0, 3.55)
        if label == "A":
            figsize = (4.6, 4.0)
        fig, ax = plt.subplots(figsize=figsize)
        drawer(ax)
        panel_label(ax, label)
        save_figure(
            fig,
            args.out / panel_filenames[label],
            args.formats,
        )
        plt.close(fig)

    # Retain the exact v17 balanced 2×4 full-figure layout.
    fig = plt.figure(figsize=(16.0, 9.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 4)
    axes = {
        "A": fig.add_subplot(grid[0, 0]),
        "B": fig.add_subplot(grid[0, 1]),
        "C": fig.add_subplot(grid[0, 2]),
        "D": fig.add_subplot(grid[0, 3]),
        "E": fig.add_subplot(grid[1, 0]),
        "F": fig.add_subplot(grid[1, 1]),
        "G": fig.add_subplot(grid[1, 2]),
        "H": fig.add_subplot(grid[1, 3]),
    }
    for label, ax in axes.items():
        drawers[label](ax)
        panel_label(ax, label)

    fig.suptitle(
        "Crossing-fiber Ising optimization and QAOA sampling "
        "on 50 qubits (real HCP diffusion MRI)",
        fontsize=13,
        fontweight="bold",
    )
    save_figure(
        fig,
        args.out / "figure_real_hcp_crossing_fiber_workflow_v17",
        args.formats,
    )
    plt.close(fig)

    summary = {
        "subject": args.subject_label,
        "patch": args.patch_label,
        "strategy": args.strategy,
        "shots": int(hardware_energies.size),
        "n_unique_bitstrings": int(len(counts)),
        "graph_nodes": int(graph.n_nodes),
        "graph_edges": int(len(graph.edges)),
        "exact_energy": float(exact_energy),
        "exact_assignment_agreement_up_to_global_sign_with_CST_reference": (
            exact_tom_accuracy
        ),
        "best_qpu_bitstring": str(row["best_bitstring"]),
        "best_qpu_energy": float(row["best_energy"]),
        "best_qpu_normalized_gap": float(row["best_normalized_gap"]),
        "best_qpu_display_agreement_with_exact": qpu_display_accuracy,
        "best_qpu_voxel_agreement_with_CST_reference": float(
            row["best_assignment"][
                "voxel_cst_peak_agreement_with_tom"
            ]
        ),
        "exact_colors_globally_flipped_for_display": bool(
            exact_display_flipped
        ),
        "qpu_colors_globally_flipped_for_display": bool(
            qpu_display_flipped
        ),
        "uniform_samples": int(args.uniform_samples),
        "uniform_seed": int(args.uniform_seed),
        "graph_validation": graph_validation,
        "solver_info": solver_info,
        "source_script": "make_crossing_fiber_figure_v17.py",
        "adaptation": (
            "Direct v17 style/export adaptation; only real-data loading "
            "and anatomical panel A were changed."
        ),
    }
    (args.out / "figure_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, default=str))
    print(f"\nFigures written to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
