# Main-text figures (regenerated)

`python src/figures/make_main_figures.py --out figures/main` rebuilds Figures 2, 4 and 5
from `data/phantom/phantom_study_level_results.csv`, `data/hcp/hcp_subject_level_results.csv`,
`analysis/hcp_baselines/hcp_nonspatial_baselines_summary.csv` and
`data/classical/figure5_source_data.csv`. `figure2_paired_tests.csv` lists the paired t-tests
and Holm-adjusted P values printed in Figure 2.

`PYTHONPATH=src/figures python src/figures/make_figure1_schematic.py --out figures/main`
draws the vector schematic used for Figure 1d-f. Figure 1a-c and Figure 3 combine
MRtrix renderings with the graph panels from `src/hcp/make_real_hcp_crossing_fiber_figure_v17_mrviewfix.py`.

Style: Arial-metric sans-serif, 5.5-7 pt text, bold lowercase panel letters, Okabe-Ito colours,
183 mm (double column) or 89 mm (single column) width; PDF/SVG keep text editable.
