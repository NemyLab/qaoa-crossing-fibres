#!/usr/bin/env python3
"""Create a supplementary logical-versus-transpiled QAOA circuit figure.

Panel a is a compact register-level representation of the logical p=1 QAOA
circuit used for the representative 50-node HCP graph. Panel b is generated
directly from the selected bound ISA circuit stored in QPY and shows the exact
operation schedule across active physical qubits. A raster image exported from
IBM Quantum can optionally replace either generated panel.

Example
-------
python make_supplementary_logical_transpiled_circuit.py \
  --qpy real_data_8n_replication/103414_R_88_86_76/heron_preflight/real_qpu_hardware_calibrated_best.qpy \
  --metadata real_data_8n_replication/103414_R_88_86_76/heron_preflight/real_qpu_preflight.json \
  --outdir paper_figures_supplementary_circuit

The first circuit in the QPY is used by default. According to the submission
metadata, this is the hardware-calibrated angle strategy.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from qiskit import qpy
from qiskit.converters import circuit_to_dag


SCRIPT_VERSION = "2026-08-06-paired-endpoints-v4"


COLORS = {
    "navy": "#214761",
    "blue": "#4C78A8",
    "light_blue": "#9ECAE1",
    "orange": "#E58A2B",
    "grey": "#707070",
    "light_grey": "#D9D9D9",
    "grid": "#E8E8E8",
    "text": "#202020",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the representative logical and Heron-transpiled HCP QAOA circuit."
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {SCRIPT_VERSION}",
    )
    parser.add_argument("--qpy", required=True, type=Path, help="Selected bound ISA QPY file.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional preflight-validation JSON used to validate circuit metrics.",
    )
    parser.add_argument(
        "--circuit-index",
        type=int,
        default=0,
        help="Circuit index within the QPY (default: 0, hardware-calibrated angles).",
    )
    parser.add_argument("--outdir", type=Path, default=Path("paper_figures_supplementary_circuit"))
    parser.add_argument(
        "--logical-image",
        type=Path,
        default=None,
        help="Optional PNG/JPG image that replaces the generated logical schematic.",
    )
    parser.add_argument(
        "--compiled-image",
        type=Path,
        default=None,
        help="Optional PNG/JPG exported by IBM Quantum that replaces the QPY schedule panel.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def set_paper_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 5.5,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_selected_circuit(qpy_path: Path, circuit_index: int):
    with qpy_path.open("rb") as handle:
        circuits = qpy.load(handle)
    if not circuits:
        raise ValueError(f"No circuits were found in {qpy_path}")
    if not 0 <= circuit_index < len(circuits):
        raise IndexError(
            f"Circuit index {circuit_index} is invalid; {qpy_path} contains {len(circuits)} circuits."
        )
    return circuits[circuit_index], len(circuits)


def operation_qubit_indices(circuit, instruction) -> list[int]:
    return [circuit.find_bit(qubit).index for qubit in instruction.qubits]


def two_qubit_depth(circuit) -> int:
    """Return depth after retaining only instructions acting on two qubits."""
    return int(circuit.depth(filter_function=lambda inst: len(inst.qubits) == 2))


def circuit_metrics(circuit) -> dict[str, object]:
    counts = Counter(circuit.count_ops())
    active = sorted(
        {
            qubit
            for instruction in circuit.data
            for qubit in operation_qubit_indices(circuit, instruction)
        }
    )
    twoq_count = sum(
        1 for instruction in circuit.data if len(instruction.qubits) == 2
    )
    return {
        "depth": int(circuit.depth()),
        "twoq_depth": two_qubit_depth(circuit),
        "twoq_count": int(twoq_count),
        "rzz_count": int(counts.get("rzz", 0)),
        "cz_count": int(counts.get("cz", 0)),
        "active_qubits": active,
        "n_active_qubits": len(active),
    }


def validate_metrics(metrics: dict[str, object], metadata_path: Path | None) -> dict:
    if metadata_path is None:
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = metadata.get("best_layout", {})
    for key in ("depth", "twoq_depth", "twoq_count", "rzz_count", "cz_count", "n_active_qubits"):
        if key in expected and int(metrics[key]) != int(expected[key]):
            raise ValueError(
                f"QPY/metadata mismatch for {key}: QPY={metrics[key]}, metadata={expected[key]}"
            )
    return metadata


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.045,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="top",
        color=COLORS["text"],
    )


def rounded_box(ax, x: float, y: float, width: float, height: float, text: str,
                facecolor: str, edgecolor: str, fontsize: float = 8.0) -> None:
    patch = FancyBboxPatch(
        (x, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y, text, ha="center", va="center", fontsize=fontsize, color=COLORS["text"])


def draw_logical_schematic(ax) -> None:
    """Draw an exact block-level description without pretending to show 50 readable wires."""
    ax.set_xlim(0, 12.8)
    ax.set_ylim(-1.0, 1.35)
    ax.axis("off")

    y = 0.18
    ax.plot([1.30, 12.25], [y, y], color=COLORS["text"], linewidth=1.2, zorder=0)
    ax.text(0.05, y, r"$q_0,\ldots,q_{49}$", ha="left", va="center", fontsize=9)

    rounded_box(ax, 1.50, y, 1.15, 0.72, r"$H^{\otimes 50}$", "#EAF2F8", COLORS["blue"], 9)
    rounded_box(
        ax,
        3.25,
        y,
        4.20,
        0.92,
        r"$U_C(\gamma)=e^{-i\gamma H_C}$" + "\n54 weighted $R_{ZZ}$ interactions",
        "#FCE8D2",
        COLORS["orange"],
        8.5,
    )
    rounded_box(
        ax,
        8.05,
        y,
        2.05,
        0.72,
        r"$U_B(\beta)$" + "\n50 $R_X$ rotations",
        "#EAF2F8",
        COLORS["blue"],
        8.5,
    )
    rounded_box(ax, 10.68, y, 1.22, 0.72, "Measure\n50 bits", "#EFEFEF", COLORS["grey"], 8.0)

    ax.plot([3.67, 7.03], [-0.43, -0.43], color=COLORS["grey"], linewidth=0.8)
    ax.plot([3.67, 3.67], [-0.43, -0.31], color=COLORS["grey"], linewidth=0.8)
    ax.plot([7.03, 7.03], [-0.43, -0.31], color=COLORS["grey"], linewidth=0.8)
    ax.text(
        5.35,
        -0.61,
        "five sequential two-qubit layers",
        ha="center",
        va="top",
        fontsize=7.5,
        color=COLORS["grey"],
    )
    ax.set_title(
        "Logical $p=1$ QAOA circuit for the representative HCP graph\n"
        "50 logical qubits; 54 problem-graph edges; logical two-qubit depth = 5",
        loc="left",
        pad=3,
        fontweight="bold",
    )


def crop_white_margin(image):
    if image.ndim == 2:
        rgb = image[..., None]
    else:
        rgb = image[..., :3]
    nonwhite = (rgb < 0.985).any(axis=-1)
    rows, cols = nonwhite.nonzero()
    if len(rows) == 0:
        return image
    pad = 8
    r0, r1 = max(0, rows.min() - pad), min(image.shape[0], rows.max() + pad + 1)
    c0, c1 = max(0, cols.min() - pad), min(image.shape[1], cols.max() + pad + 1)
    return image[r0:r1, c0:c1]


def draw_raster_panel(ax, image_path: Path, title: str) -> None:
    image = crop_white_margin(mpimg.imread(image_path))
    ax.imshow(image, interpolation="none")
    ax.axis("off")
    ax.set_title(title, loc="left", pad=3, fontweight="bold")


def layer_operations(circuit) -> Iterable[list]:
    dag = circuit_to_dag(circuit)
    for layer in dag.layers():
        yield list(layer["graph"].op_nodes())


def routed_connector_offsets(number_of_connectors: int) -> list[float]:
    """Return distinct side lanes within one operation-layer column.

    The gate markers remain centred on the integer layer coordinate. Each
    two-qubit connector is routed through a narrow lane between that centre
    and the boundary of the layer column. Alternating the lanes between the
    left and right sides prevents connectors from being superimposed.
    """
    if number_of_connectors <= 0:
        return []

    number_left = (number_of_connectors + 1) // 2
    number_right = number_of_connectors // 2

    def side_lanes(number: int, sign: float) -> list[float]:
        if number == 0:
            return []
        if number == 1:
            magnitudes = [0.40]
        else:
            # Stay inside the +/-0.5 layer boundary while placing every
            # connector outside the +/-0.32 gate-marker footprint.
            magnitudes = [0.35 + 0.12 * index / (number - 1) for index in range(number)]
        return [sign * magnitude for magnitude in magnitudes]

    left = side_lanes(number_left, -1.0)
    right = side_lanes(number_right, 1.0)
    offsets: list[float] = []
    for index in range(max(number_left, number_right)):
        if index < number_left:
            offsets.append(left[index])
        if index < number_right:
            offsets.append(right[index])
    return offsets


def add_routed_two_qubit_connector(ax, x: float, y0: float, y1: float,
                                   offset: float, color: str) -> None:
    """Draw a bracket-shaped connector with a white separation halo."""
    route_x = x + offset
    vertices = [(x, y0), (route_x, y0), (route_x, y1), (x, y1)]
    path = MplPath(vertices, [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO])

    # The white under-stroke keeps crossings visually separable. Gate markers
    # are drawn afterwards, so no connector can pass over a gate box.
    ax.add_patch(
        PathPatch(
            path,
            fill=False,
            edgecolor="white",
            linewidth=2.10,
            capstyle="round",
            joinstyle="round",
            zorder=1,
        )
    )
    ax.add_patch(
        PathPatch(
            path,
            fill=False,
            edgecolor=color,
            linewidth=0.82,
            alpha=0.95,
            capstyle="round",
            joinstyle="round",
            zorder=2,
        )
    )


def draw_transpiled_schedule(ax, circuit, metrics: dict[str, object]) -> None:
    """Draw every instruction from the QPY as a compact exact schedule raster."""
    active = list(metrics["active_qubits"])
    row_for = {qubit: row for row, qubit in enumerate(active)}
    layers = list(layer_operations(circuit))
    n_layers = len(layers)

    # Put lower-numbered physical qubits at the top.
    def plot_y(qubit: int) -> float:
        return len(active) - 1 - row_for[qubit]

    for y in range(len(active)):
        ax.axhline(y, color=COLORS["grid"], linewidth=0.32, zorder=0)
    for x in range(4, n_layers, 5):
        ax.axvline(x + 0.5, color=COLORS["grid"], linewidth=0.55, zorder=0)

    # Each two-qubit operation receives a short identifier within its layer.
    # The same identifier is printed at both endpoint markers, so the paired
    # physical qubits remain unambiguous even in a visually dense layer.
    two_qubit_markers: list[tuple[float, float, str, str]] = []
    single_qubit_markers: list[tuple[float, float, str]] = []

    for layer_index, nodes in enumerate(layers):
        x = layer_index + 1
        two_qubit_nodes = [node for node in nodes if len(node.qargs) == 2]

        # Longer connectors receive the inner lanes first. This deterministic
        # ordering reduces long horizontal excursions and keeps paired gate
        # markers easy to trace in dense layers.
        two_qubit_nodes.sort(
            key=lambda node: -abs(
                plot_y(circuit.find_bit(node.qargs[0]).index)
                - plot_y(circuit.find_bit(node.qargs[1]).index)
            )
        )
        connector_offsets = routed_connector_offsets(len(two_qubit_nodes))

        for pair_number, (node, offset) in enumerate(
            zip(two_qubit_nodes, connector_offsets), start=1
        ):
            qubits = [circuit.find_bit(qubit).index for qubit in node.qargs]
            name = node.op.name.lower()
            color = COLORS["orange"] if name == "rzz" else COLORS["navy"]
            ys = [plot_y(qubit) for qubit in qubits]
            add_routed_two_qubit_connector(ax, x, ys[0], ys[1], offset, color)
            pair_label = str(pair_number)
            two_qubit_markers.extend((x, y, color, pair_label) for y in ys)

        for node in nodes:
            if len(node.qargs) == 1:
                qubits = [circuit.find_bit(qubit).index for qubit in node.qargs]
                name = node.op.name.lower()
                y = plot_y(qubits[0])
                color = COLORS["grey"] if name == "measure" else COLORS["light_blue"]
                single_qubit_markers.append((x, y, color))

    # Draw every marker after every connector. This two-pass order guarantees
    # that neither RZZ nor CZ connector lines can cover any gate marker.
    for x, y, color, pair_label in two_qubit_markers:
        ax.add_patch(
            Rectangle(
                (x - 0.32, y - 0.31),
                0.64,
                0.62,
                facecolor=color,
                edgecolor="white",
                linewidth=0.28,
                zorder=4,
            )
        )
        ax.text(
            x,
            y,
            pair_label,
            ha="center",
            va="center",
            fontsize=3.4,
            fontweight="bold",
            color="white",
            clip_on=True,
            zorder=5,
        )
    for x, y, color in single_qubit_markers:
        ax.add_patch(
            Rectangle(
                (x - 0.26, y - 0.25),
                0.52,
                0.50,
                facecolor=color,
                edgecolor="white",
                linewidth=0.24,
                zorder=4,
            )
        )

    labels = [str(qubit) for qubit in reversed(active)]
    ax.set_xlim(0.3, n_layers + 0.7)
    ax.set_ylim(-0.8, len(active) - 0.2)
    ax.set_xticks([1] + list(range(5, n_layers + 1, 5)))
    ax.set_yticks(range(len(active)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Transpiled operation layer")
    ax.set_ylabel("Active physical qubit")
    ax.tick_params(axis="both", length=2.0, width=0.5, pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(
        "Hardware-transpiled circuit selected for IBM Heron r3 (ibm_aachen)\n"
        f"54 active physical qubits; total depth = {metrics['depth']}; "
        f"two-qubit depth = {metrics['twoq_depth']}; "
        f"two-qubit operations = {metrics['twoq_count']}",
        loc="left",
        pad=6,
        fontweight="bold",
    )

    legend = [
        Line2D([0], [0], marker="s", linestyle="none", markersize=6, markerfacecolor=COLORS["orange"],
               markeredgecolor="none", label=rf"$R_{{ZZ}}$ ({metrics['rzz_count']})"),
        Line2D([0], [0], marker="s", linestyle="none", markersize=6, markerfacecolor=COLORS["navy"],
               markeredgecolor="none", label=f"CZ ({metrics['cz_count']})"),
        Line2D([0], [0], marker="s", linestyle="none", markersize=6, markerfacecolor=COLORS["light_blue"],
               markeredgecolor="none", label="Single-qubit operation"),
        Line2D([0], [0], marker="s", linestyle="none", markersize=6, markerfacecolor=COLORS["grey"],
               markeredgecolor="none", label="Measurement"),
    ]
    ax.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.090),
        ncol=4,
        frameon=False,
        handletextpad=0.4,
        columnspacing=1.2,
        title="Matching numerals identify the two endpoints of each two-qubit operation within a layer.",
        title_fontsize=6.3,
    )


def save_figure(fig, stem: Path, dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def make_single_panels(args, circuit, metrics) -> None:
    fig_a, ax_a = plt.subplots(figsize=(7.2, 2.0))
    if args.logical_image:
        draw_raster_panel(ax_a, args.logical_image, "Logical $p=1$ QAOA circuit")
    else:
        draw_logical_schematic(ax_a)
    add_panel_label(ax_a, "a")
    save_figure(fig_a, args.outdir / "supplementary_circuit_panel_a_logical", args.dpi)
    plt.close(fig_a)

    fig_b, ax_b = plt.subplots(figsize=(7.2, 7.8))
    if args.compiled_image:
        draw_raster_panel(ax_b, args.compiled_image, "Hardware-transpiled circuit selected for IBM Heron r3")
    else:
        draw_transpiled_schedule(ax_b, circuit, metrics)
    add_panel_label(ax_b, "b")
    fig_b.subplots_adjust(left=0.09, right=0.995, top=0.94, bottom=0.15)
    save_figure(fig_b, args.outdir / "supplementary_circuit_panel_b_transpiled", args.dpi)
    plt.close(fig_b)


def make_combined_figure(args, circuit, metrics) -> None:
    fig = plt.figure(figsize=(7.2, 10.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.65, 7.8], hspace=0.21)
    ax_a = fig.add_subplot(grid[0])
    ax_b = fig.add_subplot(grid[1])

    if args.logical_image:
        draw_raster_panel(ax_a, args.logical_image, "Logical $p=1$ QAOA circuit")
    else:
        draw_logical_schematic(ax_a)
    if args.compiled_image:
        draw_raster_panel(ax_b, args.compiled_image, "Hardware-transpiled circuit selected for IBM Heron r3")
    else:
        draw_transpiled_schedule(ax_b, circuit, metrics)

    add_panel_label(ax_a, "a")
    add_panel_label(ax_b, "b")
    fig.subplots_adjust(left=0.09, right=0.995, top=0.975, bottom=0.10)
    save_figure(fig, args.outdir / "supplementary_logical_vs_transpiled_circuit", args.dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_paper_style()
    circuit, number_in_qpy = load_selected_circuit(args.qpy, args.circuit_index)
    metrics = circuit_metrics(circuit)
    validate_metrics(metrics, args.metadata)
    args.outdir.mkdir(parents=True, exist_ok=True)

    make_single_panels(args, circuit, metrics)
    make_combined_figure(args, circuit, metrics)

    summary = {
        "script_version": SCRIPT_VERSION,
        "qpy": str(args.qpy),
        "circuit_index": args.circuit_index,
        "number_of_circuits_in_qpy": number_in_qpy,
        **metrics,
    }
    with (args.outdir / "supplementary_circuit_figure_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote figure files to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
