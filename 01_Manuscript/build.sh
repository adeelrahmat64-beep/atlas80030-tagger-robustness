#!/usr/bin/env bash
# Regenerate every derived artifact, compile both documents, then verify.
#
# Nothing here reruns the physics. The figure renderer reads the frozen
# per-figure data CSVs, the table generators read the frozen publication tables
# and the frozen SHAP CSV, and verify_manuscript.py re-reads the frozen
# artifacts and re-checks every numerical claim in the text.
set -e
cd "$(dirname "$0")"

python3 tables/make_layout_tables.py
python3 tables/make_supplementary_tables.py
python3 figures_src/render_figures.py

for i in 1 2 3; do pdflatex -interaction=nonstopmode atlas80030_main.tex >/dev/null; done
for i in 1 2 3; do pdflatex -interaction=nonstopmode atlas80030_supplementary.tex >/dev/null; done

python3 verify_manuscript.py
