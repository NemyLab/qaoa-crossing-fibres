# Main-text figures (regenerated)

`python src/figures/make_main_figures.py --out figures/main` rebuilds Figures 2, 4 and 5
from `data/phantom/phantom_study_level_results.csv`, `data/hcp/hcp_subject_level_results.csv`,
`analysis/hcp_baselines/hcp_nonspatial_baselines_summary.csv` and
`data/classical/figure5_source_data.csv`. `figure2_paired_tests.csv` lists the paired t-tests
and Holm-adjusted P values printed in Figure 2.

`PYTHONPATH=src/figures python src/figures/make_figure1.py --out figures/main` builds the
complete Figure 1: panels a-b from the renderings in `data/hcp/representative_103414/`, panel c
from the released graph (drawn as in Figure 3c) and panels d-f from the vector schematic in
`make_figure1_schematic.py`. `PYTHONPATH=src/figures python src/figures/make_figure3_representative_hcp.py --out figures/main`
rebuilds Figure 3 from `data/hcp/representative_103414/` (graph, MILP solution, Heron counts and the
MRtrix mrview FOD rendering); it recomputes all energies and checks them against the stored values.

Style: Arial-metric sans-serif, 5.5-7 pt text, bold lowercase panel letters, Okabe-Ito colours,
183 mm (double column) or 89 mm (single column) width; PDF/SVG keep text editable.
