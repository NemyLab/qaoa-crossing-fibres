# Step 19: Heron run of the higher-curvature phantoms

Preregistered hardware run of the two higher-curvature geometries from Supplementary Note 2
(`c90_k50`: 90° centre, curvature 50°; `c60_k60`: 60° centre, curvature 60°), 15 seeds each, plus
five standard `curved_90` seeds as a drift control. Script: `src/hardware/heron/step19_harder_phantoms.py`.

- Backend: ibm_aachen, fractional RZZ gates, frozen Heron angles (γ = 0.7104, β = 1.2263)
- 35 circuits × 10,000 shots, XY4 dynamical decoupling, no twirling
- Best of 20 optimization-level-3 transpilations per circuit (lowest error proxy)
- Job `dbl464hul9ic73fc49ug`, 3 November 2026; 95 s QPU time (1.57 RU)

| File | Content |
|---|---|
| `prepared.json` | Run metadata and the preregistered endpoints (E1–E3) |
| `candidates.csv` | Per-circuit graph summary, exact energy, MILP and two-means agreement, transpilation |
| `transpile_trials.csv` | All transpilation trials |
| `submission.json` | Job record |
| `hardware_counts.json.gz` | Raw counts per circuit and job metrics |
| `hardware_seed_level.csv` | Per-seed hardware and random-sampling metrics |
| `hardware_summary.csv` | Endpoint summary per geometry |
| `graphs.json.gz` | The 35 graphs as built and executed (macOS arm64, NumPy 2.4.3, SciPy 1.17.1), with exact MILP solutions |

## Results

| | c90_k50 | c60_k60 |
|---|---|---|
| MILP agreement | 0.888 | 0.773 |
| Two-means agreement | 0.653 | 0.627 |
| Noiseless p = 1 gap | 0.248 | 0.262 |
| Hardware mean gap (SD) | 0.353 (0.018) | 0.360 (0.027) |
| E1 exact sign-flip P (gap < 1) | 6 × 10⁻⁵ | 6 × 10⁻⁵ |
| E2 execution penalty | 0.105 | 0.098 |
| CVaR-10% gap | 0.173 | 0.183 |
| Lowest-energy-sample agreement | 0.691 | 0.667 |
| E3: minus two-means (P) | +0.037 (0.52) | +0.040 (0.22) |
| Best random sample agreement | 0.580 | 0.588 |
| Minus best random (Holm P) | +0.111 (0.032) | +0.079 (0.005) |

Drift control (`curved_90`, 5 seeds): mean gap 0.463, execution penalty 0.104.

E1 and E2 were met. E3 was not: the lowest-energy hardware sample beat the best of an equal number of
random samples but not the non-spatial two-means baseline.

## Reproducing the analysis

The graphs are generated from simulated diffusion signals. In a few voxels of these geometries two
candidate axes fit almost equally well, so floating-point differences between platforms can change a
peak, an edge weight or a reference label. Regenerating the graphs on another machine therefore gives
values that differ by about one voxel on some seeds. Two of the 30 graphs also differ in exact energy.
The conclusions above do not change. The graphs from the machine that executed the job are stored in `graphs.json.gz`.
`reanalyse` on Linux with these graphs reproduces every value in `hardware_seed_level.csv` to
floating-point precision (maximum absolute difference below 1e-12). To reproduce:

```bash
# on the machine that ran `prepare`
python src/hardware/heron/step19_harder_phantoms.py export-graphs --out runs/step19
# anywhere, from saved counts (no IBM access needed)
python src/hardware/heron/step19_harder_phantoms.py reanalyse --out runs/step19
```

`reanalyse` uses `graphs.json.gz` when it is present and regenerates the graphs otherwise.
