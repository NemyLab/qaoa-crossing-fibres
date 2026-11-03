# Non-spatial baselines on the 30 HCP patches

Generated with `src/hcp/baseline_nonspatial_tom.py` on the per-patch
`graph_validation/real_graph_exact_assignment.csv` files of the 8-neighbour HCP
replication (one patch per participant):

```bash
python src/hcp/baseline_nonspatial_tom.py --root <real_data_8n_replication> --out hcp_baselines.csv
```

Agreement is the sign-invariant label agreement with the TractSeg TOM
pseudo-reference across all 25 voxels (`*_all`) and across TOM-confident voxels
(`*_confident`, TOM-to-nearest-peak angle <= 25 deg). Baselines use no
between-voxel information: axial two-means of the candidate axes, the more
superior-inferior candidate, and the larger-amplitude candidate. The summary
gives exact paired sign-flip P values against the exact MILP assignment.

| Method | Mean | SD | Patches at 100% | P vs MILP |
|---|---|---|---|---|
| Exact MILP | 0.989 | 0.028 | 25/30 | – |
| Axial two-means | 0.987 | 0.032 | 24/30 | 0.63 |
| Most superior-inferior | 0.980 | 0.044 | 23/30 | 0.27 |
| Larger amplitude | 0.967 | 0.087 | 21/30 | 0.25 |
| Random feasible (expected) | 0.581 | – | – | – |

Because patches were selected for one candidate within 25 deg of the TOM and an
inter-peak angle of 50-100 deg, the two populations are largely separable by
orientation alone. The HCP agreement validates the encoding on measured data;
the benefit of spatial coupling is tested on the higher-curvature phantoms
(`analysis/classical_baselines`). The files contain only per-patch agreement
values, no HCP image data.
