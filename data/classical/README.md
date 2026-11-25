# Classical-control data

- `figure5_source_data.csv`: source data for manuscript Figure 5 (Classical Performance Context). Hardware-calibrated QAOA, uniform sampling, first-improvement greedy, best-improvement greedy and the seed-2000-tuned simulated annealing (`selected_sa_configuration.json`); one run per seed and method, 15 held-out 90° seeds. These are the values quoted in the manuscript text.
- `control_results_by_seed_budget.csv`, `control_aggregate.csv`, `qpu_vs_classical_paired.csv`, `classical_control_aggregate.json`: an earlier, larger run of the controls (100 repeats per seed, budget and method) that used the untuned simulated-annealing schedule (4 restarts) and did not include first-improvement greedy. They also include the `qpu_transferred_center` strategy (QAOA at the noiseless-optimized centre angles). They are kept for transparency, but **they are not the source of Figure 5**.

Budget convention: one QPU shot, or one full-energy or single-flip energy-difference evaluation by a classical heuristic, counts as one query. For the heuristics, only visited assignments enter the best-observed set.
