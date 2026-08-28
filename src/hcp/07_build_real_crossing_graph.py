#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

THRESHOLD = 25.0
SIDES = ("top", "bottom", "left", "right")


def load_base():
    path = Path(__file__).with_name("07_build_real_crossing_graph_base.py")
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    spec = importlib.util.spec_from_file_location("real_graph_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_wrapper_args() -> Path:
    global THRESHOLD
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--max-tom-anchor-angle", type=float, default=25.0)
    known, remaining = parser.parse_known_args()
    if not 0 <= known.max_tom_anchor_angle <= 90:
        parser.error("--max-tom-anchor-angle must be between 0 and 90")
    THRESHOLD = float(known.max_tom_anchor_angle)

    out = None
    for i, value in enumerate(remaining):
        if value == "--out" and i + 1 < len(remaining):
            out = Path(remaining[i + 1])
            break
        if value.startswith("--out="):
            out = Path(value.split("=", 1)[1])
            break
    if out is None:
        parser.error("The base command must contain --out")

    sys.argv = [sys.argv[0], *remaining]
    return out


def confident_boundary_anchor_voxels(tom_angles: np.ndarray) -> list[dict[str, Any]]:
    angles = np.asarray(tom_angles, dtype=float)
    if angles.shape != (25,):
        raise ValueError(f"Expected 25 TOM angles, found {angles.shape}")

    side_voxels = {
        "top": [col for col in range(5)],
        "bottom": [20 + col for col in range(5)],
        "left": [5 * row for row in range(5)],
        "right": [5 * row + 4 for row in range(5)],
    }

    selected = []
    used: set[int] = set()
    for side, voxels in side_voxels.items():
        eligible = [
            v for v in voxels
            if v not in used and np.isfinite(angles[v]) and angles[v] <= THRESHOLD
        ]
        if not eligible:
            continue
        chosen = min(eligible, key=lambda v: (float(angles[v]), v))
        used.add(chosen)
        selected.append({
            "side": side,
            "voxel_index": int(chosen),
            "row": int(chosen // 5),
            "col": int(chosen % 5),
            "tom_match_angle_deg": float(angles[chosen]),
            "tom_confident": True,
            "maximum_allowed_tom_angle_deg": THRESHOLD,
        })
    return selected


def augment_summary(out: Path) -> None:
    path = out / "real_graph_validation.json"
    data = json.loads(path.read_text())
    anchors = data.get("graph", {}).get("anchors", [])
    anchored = [str(x.get("side")) for x in anchors]
    angles = [float(x["tom_match_angle_deg"]) for x in anchors]

    data.setdefault("graph", {}).update({
        "tom_anchor_confidence_threshold_deg": THRESHOLD,
        "anchor_selection": "At most one unique TOM-confident voxel per boundary side",
        "anchored_boundary_sides": anchored,
        "missing_boundary_sides": [s for s in SIDES if s not in anchored],
    })
    data.setdefault("safety_checks", {}).update({
        "all_tom_anchors_within_confidence_threshold": all(a <= THRESHOLD for a in angles),
        "at_least_two_confident_boundary_anchors": len(anchors) >= 2,
    })
    data.setdefault("interpretation", {})["tom_anchor_role"] = (
        f"Only TOM matches <= {THRESHOLD:g} degrees can define boundary anchors; "
        "all continuation edges remain CSD/displacement based."
    )
    data["confidence_gated_wrapper"] = {
        "threshold_deg": THRESHOLD,
        "base_file": "07_build_real_crossing_graph_base.py",
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    out = parse_wrapper_args()
    base = load_base()
    base.boundary_anchor_voxels = confident_boundary_anchor_voxels
    base.main()
    augment_summary(out)


if __name__ == "__main__":
    main()
