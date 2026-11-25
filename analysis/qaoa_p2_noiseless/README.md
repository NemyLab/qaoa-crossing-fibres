# Noiseless p = 2 QAOA on the step-19 graphs

Exact noiseless expectation values for p = 1 (frozen Heron angles) and p = 2 on the two
higher-curvature geometries (15 seeds each). Each term of the cost Hamiltonian is evaluated on
its reverse light cone (all nodes within graph distance p), at most 14 qubits here, so the
result is exact for 50-qubit graphs. The p = 2 angles were optimized (Nelder–Mead, four starts)
on seeds 2001–2003 of both geometries and then applied to all seeds.

| | p = 1 gap | p = 2 gap | p = 2 gap, held-out seeds |
|---|---|---|---|
| c90_k50 | 0.249 | 0.160 | 0.159 |
| c60_k60 | 0.261 | 0.167 | 0.167 |

Under a global depolarizing approximation, the step-19 hardware run retained about 86% of the
signal at p = 1 (1 − 0.353 = F × (1 − 0.249)). Doubling the depth gives roughly F² ≈ 0.74 and an
expected hardware gap of about 0.375, slightly worse than the measured p = 1 value. p = 2 was
therefore not run on hardware.

Run from the repository root: `python analysis/qaoa_p2_noiseless/p2_lightcone.py analysis/qaoa_p2_noiseless`
(requires networkx).
