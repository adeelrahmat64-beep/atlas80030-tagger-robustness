# Systematic robustness of a boosted top-quark tagger on ATLAS open data

Code, frozen results and manuscript sources for

> A. Rahmat, M. I. Jamil, A. Awan, *Systematic robustness of a boosted top-quark
> tagger on ATLAS open data: a constituent-level feature reconstruction, paired
> SHAP-Delta attribution drift, and conformal coverage*, Zenodo preprint (2026),
> [doi:10.5281/zenodo.23079101](https://doi.org/10.5281/zenodo.23079101)

The analysis uses the ATLAS top-tagging open data set with systematic
uncertainties, CERN Open Data Portal Record 80030
([doi:10.7483/OPENDATA.ATLAS.SOAY.LABE](https://doi.org/10.7483/OPENDATA.ATLAS.SOAY.LABE)).

## What the work does

In the detector-systematic samples of Record 80030 the fifteen stored
high-level substructure branches are copies of the nominal values, while the
released constituents carry the variation. This repository:

1. rebuilds all fifteen high-level quantities from the released constituents
   with one implementation applied to every sample, and audits the
   reconstruction against the stored nominal branches;
2. trains an XGBoost tagger on a five-million-jet nominal subset and evaluates
   it, with the record's prescribed systematic-uncertainty procedure, on all
   seventeen samples;
3. measures jet-by-jet TreeSHAP attribution drift on the four calorimeter
   variations that pair exactly with the nominal sample (SHAP-Delta);
4. measures class-conditional split-conformal coverage on the nominal test
   sample and all nine detector-systematic samples.

## Repository layout

| Folder | Contents |
|---|---|
| `03_Code_files/` | The complete analysis pipeline (see order below). `atlas_config.py` is the single source of truth for paths, dataset names and the frozen reconstruction fingerprint. |
| `04_Results/` | Frozen outputs of every campaign the paper quotes: closure audits, cache audit, pairing logs, metrics, SHAP-Delta round 3, conformal coverage, the stored-branch audit, publication figure data and tables. Each run log sits beside a snapshot of the exact code that produced it. |
| `ATLAS_80030_Data/models/` | The final tagger (`atlas_tagger_xgb_final_production_round1.joblib`), its train/validation split and metadata. |
| `ATLAS_80030_Data/predictions/` | Final metrics (JSON/CSV/TXT), the uncertainty summary and the prediction manifest. Per-jet prediction files are not included (size). |
| `01_Manuscript/` | LaTeX sources of the paper and supplement, the four figures, all tables, the figure and table generators, and `verify_manuscript.py`. |

## Quick check: verify every number in the paper (no data download needed)

```bash
pip install numpy
cd 01_Manuscript
pdflatex atlas80030_main.tex            # run three times
pdflatex atlas80030_supplementary.tex   # run three times
python verify_manuscript.py
```

The verifier re-reads the frozen artifacts in `04_Results/` and
`ATLAS_80030_Data/` and checks every numerical claim, cross-reference and
citation in the manuscript (437 checks). Expected output:

```
PASSED 437   FAILED 0
All manuscript numerical and cross-reference checks passed.
```

## Full reproduction

**Environment.** Python 3.11; the exact environment used is pinned in
`requirements.txt` (`pip install -r requirements.txt`).

**Data.** About 200 GB. `download_data.py` fetches Record 80030 with the
`cernopendata-client` command-line tool, which is a separate prerequisite and
not part of the recorded environment: `pip install cernopendata-client`. Files
go into `ATLAS_80030_Data/data/` (override the location with the
`ATLAS_80030_DATA_ROOT` environment variable). Run the script with no argument
first: it only reports status and never downloads without being told to.

**Pipeline order.** Run from the repository root. Each script documents its
own options in its header.

| Step | Script | Produces |
|---|---|---|
| 1 | `download_data.py`, `organise_data.py` | Raw HDF5 files per dataset |
| 2 | `validate_data5.py`, `validate_bias.py` | Premise checks on the stored branches |
| 3 | `recompute_hl.py` | Fifteen high-level quantities from constituents (frozen fingerprint `c28a2625…`) |
| 4 | `strict_closure_audit.py`, `strict_closure_outlier_diagnostics.py`, `final_targeted_closure_audit.py` | Closure against stored nominal branches; classification of the 44 affected jets |
| 5 | `cache_integrity_audit.py` | 810-file, 80,481,421-entry cache audit |
| 6 | `stale_hlf_audit.py` | Stored-branch audit (Supplementary Table S1) |
| 7 | `atlas_tagger.py --final-production` | Tagger training and inference on all samples |
| 8 | `atlas_metrics.py` | AUC, background rejection, 14 pT bins, systematic envelopes |
| 9 | `pair_features.py`, `bias_pairing_diagnostic.py` | Exact-key jet pairing |
| 10 | `shap_delta.py --run-campaign` | Paired TreeSHAP attribution drift |
| 11 | `conformal_coverage.py` | Class-conditional conformal coverage |
| 12 | `publication/make_publication_figures.py`, `publication/make_publication_tables.py` | Frozen figure data and tables |
| 13 | `01_Manuscript/build.sh` | Tables, figures, both PDFs, verification |

Steps 3–11 need the downloaded data. Step 13 needs only this repository plus
`numpy`, `matplotlib`, a LaTeX installation with `pdflatex`, and `bash` (on
Windows, Git Bash): it regenerates the layout and supplementary tables and the
four figures from the frozen inputs in `04_Results/`, compiles both PDFs and
runs the verifier. The regenerated tables are byte-identical to the committed
ones; the figure PDFs are redrawn from the same data but are not byte-identical
(PDF metadata and matplotlib version), and the verifier asserts that the ROC
AUCs drawn equal the frozen metrics.

## Notes

- **Two copies of the code, on purpose.** Beside every run log in `04_Results/`
  sits a `*_code.py` snapshot of the exact source that produced it; the model
  metadata records the SHA-256 of the tagger snapshot
  (`ATLAS80030_tagger_xgb_final_production_round1_code.py`, `7ce74841…`).
  `03_Code_files/` holds the working scripts, which are those snapshots plus
  changes made after the frozen runs: every script resolves results through
  `atlas_config.RESULTS_ROOT` (the reorganised folder layout);
  `atlas_tagger.py` additionally excludes the `−999` sentinel in jet
  eligibility; and `pair_features.py` restricts future pairing to the four
  calorimeter variations against `test_nominal` only (the frozen pairing pool
  covered both nominal partitions, and every match came from `test_nominal`,
  as the paper states). **To reproduce a frozen number exactly, run the
  matching snapshot.** All files are published unedited, so usage notes inside
  some scripts still mention the earlier folder name `Code files\` and Windows
  paths, and some frozen logs record local drive paths; these are cosmetic.
- **Official ATLAS reference scripts.** The analysis follows the ATLAS
  reference implementation at commit
  [`5bc3c8d7`](https://gitlab.cern.ch/atlas/ATLAS-top-tagging-open-data/-/tree/5bc3c8d70cf8736d24b969972a3dd365017a9e24).
  Those scripts are not redistributed here; `03_Code_files/reference/atlas_5bc3c8d7/README.md`
  gives their git blob and SHA-256 hashes so a downloaded copy can be checked.
- **One empty log is kept on purpose.**
  `04_Results/modelling_closure/ATLAS80030_closure_modelling_ttbar_pythia_round1.txt`
  is from an interrupted run that wrote no output. Round 2 reran the identical
  code (its `_code.py` snapshot is byte-identical) and is the result used.
  Run directories and round numbers are never reused or renumbered, so the
  empty file stays to keep the record complete.
- **Not included (size):** raw HDF5 data, the 810-file feature cache, paired
  frames, per-jet prediction files, numba caches, per-jet SHAP exports of
  round 2 and superseded audit rounds. All except the raw data are regenerated
  by the pipeline.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`) and the
ATLAS open data record.

## License

Code: MIT (see `LICENSE`). Manuscript text and figures: CC BY 4.0, as in the
Zenodo preprint. `01_Manuscript/sn-jnl.cls` is Springer Nature's journal class,
included unmodified for building the manuscript; its own terms apply.
