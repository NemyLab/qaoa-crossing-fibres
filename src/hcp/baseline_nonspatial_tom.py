#!/usr/bin/env python3
"""Non-spatial baselines for the 30 HCP patches (requires locally built graphs).

For every ``real_graph_exact_assignment.csv`` found under ``--root`` (written by
``07_build_real_crossing_graph.py``), compare the TOM pseudo-reference labels with:

* ``exact_milp``      - the exact Ising solution (as reported in the manuscript);
* ``axial_2means``    - two-means clustering of all 50 candidate axes (sign-invariant,
                        |dot| similarity), candidates of a voxel forced to opposite groups;
* ``most_vertical``   - in each voxel, the peak with the larger |z| (superior-inferior)
                        component is labelled as the CST population;
* ``amplitude``       - in each voxel, the larger-amplitude peak is labelled as population A;
* ``random_feasible`` - expected agreement of random within-voxel-feasible labellings.

None of the baselines uses between-voxel information or the TOM. Agreement is the
sign-invariant fraction used in the manuscript, over all 25 voxels and over
TOM-confident voxels (TOM-to-nearest-peak angle <= 25 deg).

Usage:
    python baseline_nonspatial_tom.py --root real_data_8n_replication --out hcp_baselines.csv
"""
from __future__ import annotations

import argparse
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd


def sign_invariant(a: np.ndarray, b: np.ndarray) -> float:
    m = float(np.mean(a == b))
    return max(m, 1.0 - m)


def per_voxel(nodes: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    """Within each voxel, give +1 to the node with the larger score."""
    out = np.empty(len(nodes), dtype=int)
    for _, idx in nodes.groupby(["row", "col"]).groups.items():
        i, j = list(idx)
        out[i], out[j] = (1, -1) if score[i] >= score[j] else (-1, 1)
    return out


def axial_two_means(vectors: np.ndarray, nodes: pd.DataFrame, iterations: int = 100) -> np.ndarray:
    v = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    # initialise with the two most orthogonal axes
    sim = np.abs(v @ v.T)
    i0, j0 = np.unravel_index(np.argmin(sim), sim.shape)
    centres = np.stack([v[i0], v[j0]])
    for _ in range(iterations):
        lab = np.abs(v @ centres[0]) >= np.abs(v @ centres[1])
        new = []
        for group in (lab, ~lab):
            if not group.any():
                new.append(centres[len(new)])
                continue
            scatter = v[group].T @ v[group]
            w, vec = np.linalg.eigh(scatter)
            new.append(vec[:, -1])
        new = np.stack(new)
        if np.allclose(np.abs(np.sum(new * centres, axis=1)), 1.0, atol=1e-10):
            break
        centres = new
    score = np.abs(v @ centres[0]) - np.abs(v @ centres[1])
    return per_voxel(nodes, score)


def expected_random_feasible(n_voxels: int) -> float:
    return sum(comb(n_voxels, k) * max(k, n_voxels - k) for k in range(n_voxels + 1)) / 2**n_voxels / n_voxels


def analyse(path: Path) -> dict:
    t = pd.read_csv(path).sort_values("node_id").reset_index(drop=True)
    truth = t["true_spin"].to_numpy(int)
    vec = t[["dir_x", "dir_y", "dir_z"]].to_numpy(float)
    confident = (t["tom_match_angle_deg"].to_numpy(float) <= 25.0)
    methods = {
        "exact_milp": t["exact_spin_aligned_to_tom"].to_numpy(int),
        "axial_2means": axial_two_means(vec, t),
        "most_vertical": per_voxel(t, np.abs(vec[:, 2])),
        "amplitude": per_voxel(t, t["amplitude"].to_numpy(float)),
    }
    row = {"patch_dir": str(path.parent), "n_confident_voxels": int(confident.sum() // 2)}
    for name, spins in methods.items():
        row[f"{name}_all"] = sign_invariant(spins, truth)
        row[f"{name}_confident"] = sign_invariant(spins[confident], truth[confident]) if confident.any() else np.nan
    row["random_feasible_expected_all"] = expected_random_feasible(25)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("hcp_nonspatial_baselines.csv"))
    args = parser.parse_args()
    files = sorted(args.root.rglob("real_graph_exact_assignment.csv"))
    if not files:
        raise SystemExit(f"No real_graph_exact_assignment.csv under {args.root}")
    table = pd.DataFrame([analyse(p) for p in files])
    table.to_csv(args.out, index=False)
    cols = [c for c in table.columns if c.endswith("_all") or c.endswith("_confident")]
    print(f"{len(table)} patches")
    print(table[cols].agg(["mean", "std", "min"]).T.round(3).to_string())


if __name__ == "__main__":
    main()
