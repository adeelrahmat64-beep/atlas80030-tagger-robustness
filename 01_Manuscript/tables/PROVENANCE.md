# Provenance of the figures and tables in this directory

Revision 2, 9 September 2026. Everything below is derived from the frozen analysis
artifacts by a script in this directory. No value is retyped, rounded or recomputed
by hand, and each generator asserts that fact before it writes.

## Main-text tables

Byte-identical copies of the frozen publication artifacts. The `layout/` variants
change only the LaTeX float type and add `\resizebox`; `make_layout_tables.py`
regenerates them and verifies that every numeric token is unchanged.

| File | SHA-256 | Frozen source | Match |
|---|---|---|---|
| `tables/table1_nominal_performance.tex` | `47a2d34a5857c861a5929f385339422d...` | `publication_tables_round1/table1_nominal_performance.tex` | OK |
| `tables/table2_inclusive_systematics_wp50.tex` | `e13d26cc3e4d880ae374ffc79e06bfd9...` | `publication_tables_round1/table2_inclusive_systematics_wp50.tex` | OK |
| `tables/table3_shap_delta_100k_summary.tex` | `97f8e8ed09ba3272ee0228b1c2310682...` | `publication_tables_round1/table3_shap_delta_100k_summary.tex` | OK |

| Layout variant | SHA-256 | Transformation |
|---|---|---|
| `tables/layout/table1_nominal_performance.tex` | `ca1eae4d42d9042a6e762bc3404c7f4f...` | `table` -> `table*`, `\resizebox{\textwidth}` |
| `tables/layout/table2_inclusive_systematics_wp50.tex` | `64aa1fb42a10cdd55386c77a5b1af8f5...` | `table` -> `table*` |
| `tables/layout/table3_shap_delta_100k_summary.tex` | `6c6f9d1b5097fafbc868d471ad0a9f63...` | `\resizebox{\textwidth}` |

The pairing table (Table 2 of the main text) is written in the manuscript itself
from the frozen pairing logs `ATLAS80030_pair_all_round1.txt` and
`ATLAS80030_pair_bias_round1.txt` and the frozen evaluation log
`ATLAS80030_tagger_xgb_final_production_round1.txt`; `verify_manuscript.py`
re-checks every entry against those logs.

## Supplementary tables

The external review asked for the two twenty-column supplementary tables to be
split into grouped, readable pieces, and for the paired SHAP-Delta table to be
separated into its per-feature part and its systematic-level score and rank part.
`make_supplementary_tables.py` performs that split as a **mechanical re-grouping**:
it parses the frozen sources, re-emits the same cells in a different arrangement,
and then asserts that the multiset of numeric tokens in its outputs is exactly the
multiset in the frozen sources. The SHAP outputs are additionally cross-checked
value by value, keyed on (subset, feature, systematic, sample size), against the
frozen twenty-column `supplementary_shap_delta_round3_all_features.tex` of the
previous revision, which is retained in this directory as the second source.

| Generated file | SHA-256 | Frozen source | Check |
|---|---|---|---|
| `tables/supp_s2_systematics_detector.tex` | `9a046d233512f028...` | `supplementary_systematics_14bin.tex` (cols 1–11) | token multiset preserved |
| `tables/supp_s3_systematics_modelling.tex` | `f03a62b15b2f3d1b...` | `supplementary_systematics_14bin.tex` (cols 1–2, 12–20) | token multiset preserved |
| `tables/supp_s4_conformal_coverage.tex` | `2c2e620aca2267ee...` | `supplementary_conformal_round1.tex` (coverage cols) | token multiset preserved |
| `tables/supp_s5_conformal_sets.tex` | `08702303832a3edb...` | `supplementary_conformal_round1.tex` (set-structure cols) | token multiset preserved |
| `tables/supp_s6_shap_per_feature.tex` | `95b3e3da0602cb06...` | `shap_delta_final_production_round3/shap_delta_per_feature.csv` | 540 rows, value-by-value vs the frozen 20-column table |
| `tables/supp_s7_shap_scores.tex` | `105d31f949edbf1b...` | same CSV, one row per systematic/size/subset | 36 rows, value-by-value vs the frozen 20-column table |

`tables/supp_s1_stale_hlf_audit.tex` (SHA-256
`28a9bfc4a0989da94ab5fc01afc3d10ce515ab967e99dd00a70ae0d553dbf20d`)
is the completed, layout-adapted Table S1 from
`Code files/analysis_results/stale_hlf_audit/`. The source directory preserves
the full-sample audit log, sample CSV, per-branch CSV, JSON and generated table.
The four calorimeter-cluster samples are exactly pairable and have 100% float32
identity across all fifteen eligible stored branches. The track samples and
`bias` are diagnostic/not pairable, so their HLF identity entries are N/A.
No frozen physics result is recalculated for this table.

## Figures

The four figure PDFs are re-rendered by `figures_src/render_figures.py` from the
frozen per-figure data CSVs written by the project's `make_publication_figures.py`.
Every plotted value is read from those CSVs; nothing is recomputed. The changes
relative to the previous revision are cosmetic, and answer the review's figure
comments:

* Fig. 1 (ROC) — a lower panel gives the ratio of each energy-scale curve to the
  nominal one on a linear axis. On the logarithmic axis above, the three curves
  are too close to separate by eye; the ratio introduces no quantity that is not
  already in the upper panel.
* Fig. 2 (performance and systematics) — unchanged from the previous revision.
  The caption now states that no uncertainty band is drawn on any curve, and the
  main text no longer offers a finite-statistics reading of the bin-to-bin
  fluctuation.
* Fig. 3 (SHAP-Delta) — every heat-map cell now carries its numerical value, so
  the small entries can be read without the colour scale, and the colour bar is
  labelled.
* Fig. 4 (conformal coverage) — the connecting lines are removed entirely. The
  horizontal axis is categorical, so the markers are drawn alone against light
  vertical guides.

| Rendered figure | SHA-256 | Frozen data source |
|---|---|---|
| `figures/figure1_roc.pdf` | `2a522fdbf4448d70...` | `publication_figures_round1/roc_curve_data.csv` |
| `figures/figure2_performance_systematics.pdf` | `8747334be473abe0...` | `publication_figures_round1/figure1_performance_systematics_data.csv` |
| `figures/figure3_shap_delta.pdf` | `b0862224ce8a5a04...` | `publication_figures_round1/figure2_shap_delta_data.csv` |
| `figures/figure4_conformal_coverage.pdf` | `ab988f074bc73753...` | `publication_figures_round1/figure3_conformal_coverage_data.csv` |

The three AUC values in the Fig. 1 legend are asserted equal, bit for bit, to the
AUC values in the frozen metrics artifact, and its working-point markers carry the
frozen background-rejection values rather than values read off the curve.

All frozen table copies match their sources: **YES**.

## Revision 3 (10 October 2026): robustness extension

Nothing above is changed. The new tables are written by
`tables/make_extension_tables.py` from three new result folders, each produced by a
script in `03_Code_files/` that first reproduces the frozen result exactly:

| Result folder | Script | Self-check before any new number |
|---|---|---|
| `04_Results/stat_bootstrap_round1/` | `stat_bootstrap.py` (uses `uncertainty_chain.py`) | unit weights reproduce every frozen metric exactly |
| `04_Results/tagger_variants_round1/` | `tagger_variants.py` | seed-42 refit has the frozen SHA-256 and bit-identical predictions |
| `04_Results/feature_response_round1/` | `feature_response.py` | ECF1 median equals the stale-HLF audit (+7.406764%) |

The generator refuses to write unless: the frozen rows of Tables 3 and 4 are carried
over verbatim and equal the bootstrap central values and the seed-42 run; the
bootstrap standard deviations recompute from `replicas.jsonl`; and every six-seed
summary recomputes from the individual runs.

| Generated file | SHA-256 | Used as |
|---|---|---|
| `tables/ext/table1_nominal_performance_precision.tex` | `9819dbebdf7933a0...` | main Table 3 (frozen rows + stat. and seed rows) |
| `tables/ext/table2_inclusive_systematics_precision.tex` | `8edf040b299c58a2...` | main Table 4 (frozen column + stat. and seed columns) |
| `tables/ext/table6_feature_response_degree.tex` | `1130ee189ff83f20...` | main Table 6 |
| `tables/ext/table7_scale_free_variants.tex` | `5814f66f20b17660...` | main Table 7 |
| `tables/ext/supp_s8_bootstrap_binned.tex` | `88b898c37483505b...` | Supplementary Table S8 |
| `tables/ext/supp_s9_seed_runs.tex` | `e79d14224c8324d2...` | Supplementary Table S9 |
| `tables/ext/supp_s10_feature_response.tex` | `604532356e525ed8...` | Supplementary Table S10 |

`figures/figure2_performance_systematics.pdf` (SHA-256 `4d7daa720077861b...`) now
carries the bootstrap error bars and bands. `render_figures.py` reads
`stat_bootstrap_round1/stat_bootstrap_summary.json` and asserts that its central
values equal the frozen figure CSV (to 1e-9) before drawing them; the plotted
central values are unchanged.

`sources/atlas_readme_5bc3c8d7_quotes.md` records the README statements the text
quotes, with the README's git blob id and SHA-256 at the frozen commit.
`verify_manuscript.py` checks every quoted fragment against it.
