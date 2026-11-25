"""Shared non-spatial orientation baseline (order-independent multi-start two-means)."""
import numpy as np
import pandas as pd


def two_means_labels(nodes: pd.DataFrame, n_starts: int = 720) -> np.ndarray:
    """Non-spatial axial two-means rule: one label per peak, opposite within a voxel.

    Peak orientations are mapped to the doubled-angle unit circle and clustered
    into two groups. Lloyd iterations are started from ``n_starts`` splits by a
    line through the origin and the lowest within-cluster cost is kept, so the
    result does not depend on node order or on a single initialisation.
    """
    angle = np.arctan2(nodes['dir_y'].to_numpy(), nodes['dir_x'].to_numpy()) % np.pi
    z = np.exp(2j * angle)
    best = None
    for theta in np.linspace(0.0, 2.0 * np.pi, n_starts, endpoint=False):
        first = np.real(z * np.exp(-1j * theta)) > 0
        if first.all() or (~first).all():
            continue
        for _ in range(50):
            c0, c1 = z[first].mean(), z[~first].mean()
            updated = np.abs(z - c0) < np.abs(z - c1)
            if updated.all() or (~updated).all() or (updated == first).all():
                break
            first = updated
        c0, c1 = z[first].mean(), z[~first].mean()
        cost = float((np.abs(z[first] - c0) ** 2).sum() + (np.abs(z[~first] - c1) ** 2).sum())
        if best is None or cost < best[0] - 1e-12:
            best = (cost, c0, c1)
    _, c0, c1 = best
    labels = np.empty(len(nodes), dtype=int)
    for _, index in nodes.groupby(['row', 'col']).groups.items():
        i, j = list(index)
        di = abs(z[i] - c0) - abs(z[i] - c1)
        dj = abs(z[j] - c0) - abs(z[j] - c1)
        labels[i], labels[j] = (1, -1) if di < dj else (-1, 1)
    return labels
