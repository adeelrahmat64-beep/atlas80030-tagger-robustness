# ATLAS 80030 — Training Readiness Audit

Date: 8 September 2026

Scope: source inspection, cache metadata checks, tiny synthetic invariants,
32-row real-cache smoke test, selected modelling-shard medians, and comparison
with the official ATLAS reference repository. No model was fitted and no SHAP,
conformal, or full-dataset analysis was run.

## Verdict

`TRAINING READINESS = PASS`

This verdict applies to `atlas_tagger.py` SHA-256
`91b74721bbfb2a9ba65a56283c065bc143a427bee75661a149f31649400935d4`.

## Reconstruction provenance

- Previous frozen `recompute_hl.py` SHA-256:
  `3bb6e45a7e3e7bdd5b4c44c9cb5fbdd86f9cf67cf3a7c0b23553bc5cdd54ff2d`
- Current `recompute_hl.py` SHA-256:
  `3a57c02734e6ab0f82e54d9639eb3a8a484e6e3ba5f7ee70bf34c20702b89136`
- Scientific `CODE_FP` in both versions:
  `c28a262540824e9db90f34c421f10c70`

The previous file was reconstructed in memory from the saved frozen scientific
snapshot and its former operational block. Its SHA-256 matched the recorded
`3bb6e45a...` value exactly. The complete diff contains only these changes:

1. Current lines 242–245 import the shared operational configuration and
   replace `STORE = Path(r"E:\ATLAS_80030_Data")` with `STORE = DATA_ROOT`.
2. Current lines 252–259 replace a `train_nominal`-only campaign list with all
   17 datasets from `REQUIRED_DATASETS`; adjacent comments describe that
   operational change.
3. Current lines 1698–1719 make `summarise()` skip distributional medians for
   valid zero-row modelling shards while still counting and fingerprinting
   those files.

No changed line intersects the fingerprinted reconstruction regions. Kernels
and `load()` occupy current lines 390–1164; `produce()` occupies lines
1537–1671. These regions are byte-identical between the previous and current
sources. Their combined SHA-256 is
`2574528126bb7e04d4547848f3e22749a91da82d17f7f72dd5a76cd6afe86e1e`.
No scientific kernel, observable formula, constituent treatment,
normalisation, cache value, or tolerance changed.

## Trainer audit and corrections

The pre-audit script had three blocking gaps: jet-level validation splitting,
an unweighted optional MLP branch, and no persisted model/split metadata. The
following training-pipeline-only corrections were made:

- `HL` remains the fixed 15-feature order and is identical to
  `recompute_hl.py`.
- `require_frozen_cache()` checks every required dataset before training.
  `load_features()` now rechecks each file's scalar fingerprint before reading
  any feature and rejects missing required cache fields.
- All 15 model inputs are stacked exclusively from
  `hl_recomputed/<dataset>/*.npz`.
- Raw HDF5 reads are allow-listed to `training_weights` and
  `EventInfo_mcEventWeights`; any `fjet_*` request raises an error.
- Eligibility is exactly `isfinite(all 15 features) AND n_constit >= 3`.
  There is no imputation or `nan_to_num`. Therefore all 284 known L3
  infinities are excluded deterministically.
- Validation assignment is a stable SplitMix64 hash of
  `EventInfo_mcEventNumber` and seed 42. Every occurrence of an event number
  receives one assignment, preventing event leakage and avoiding dependence
  on row order or sklearn split internals.
- Training weights are checked for shape, finiteness, and non-negativity, then
  supplied as `sample_weight` to both XGBoost and sklearn 1.9's MLP.
- Evaluation calls the same `load_features()` function, same `HL` order,
  same eligibility mask, and the same fitted estimator/pipeline as training.
- Each completed training run saves a non-overwriting joblib model, exact
  eligibility/validation masks, file stems and row counts, and JSON metadata.
  The JSON records feature order, preprocessing, seeds, split rule, source and
  artifact hashes, cache fingerprint, class counts, and library versions.

Lightweight checks passed: syntax, frozen-cache preflight, feature-order
identity, NaN/Inf eligibility rejection, event-group invariance, raw-HLF read
rejection, XGBoost/MLP serialization, and a 32-row real cache/weight alignment
smoke test.

Active pipeline files contain no `E:\ATLAS_80030_Data` reference. That string
remains only in immutable historical source snapshots and archive files; none
is imported or invoked by the active pipeline.

## Cluster and string file-spread check

The alarming summary spans are artifacts of an unweighted min–max across file
medians:

- `cluster_050` contains only 4 rows.
- `string_050` contains only 1 row.

Their individual values account for the extreme reported spans. As a bounded
cross-check, three representative 100,000-row shards from each sample were
read. Their feature-median spans were 0.08–0.86% for cluster and 0.03–2.26%
for string, rather than the reported 72% and 110% extremes.

No special sampling change is needed because `atlas_tagger.py` evaluates every
row from every modelling file. If modelling evaluation is later downsampled,
it must sample rows across all full shards (preferably stratified in pT and
file), never take a file prefix or weight the 1-row/4-row tail shards equally
with 100,000-row shards.

## Official methodology cross-check

Reference repository:
`https://gitlab.cern.ch/atlas/ATLAS-top-tagging-open-data`, master commit
`5bc3c8d70cf8736d24b969972a3dd365017a9e24`.

The planned metric implementation matches the reference:

- nominal metrics are unweighted AUC, accuracy at score 0.5, and inverse
  background efficiency at fixed signal efficiency;
- 50% is the primary systematic working point; 80% is additionally reported
  for the paper's nominal benchmark comparison;
- pT uses 15 edges from 350,000 to 3,150,000 MeV and the reference's strict
  open-bin inequalities;
- detector variations use unit evaluation weights and are compared through
  `abs(BR_systematic / BR_nominal - 1)` in each pT bin and inclusively;
- energy scale is the envelope of `esup`/`esdown`; cluster total combines that
  envelope with `cer` and `cpos` in quadrature;
- track efficiency is the `teg`/`tej` envelope, fake rate is the `tfl`/`tfj`
  envelope, and track total combines those with `bias` in quadrature;
- modelling single-class samples receive the missing nominal class exactly as
  in `calc.py`; signal modelling is measured below 1.5 TeV and extrapolated by
  the maximum measured uncertainty.

The official methodology is approximate and intended for this tagger study,
not direct use in a physics analysis. No metric computation was run during
this audit.

## Authorized local command

From any PowerShell location:

```powershell
& 'D:\SSD\HEP\Research Papers\ATLAS 80030\.venv\Scripts\python.exe' 'D:\SSD\HEP\Research Papers\ATLAS 80030\Code files\atlas_tagger.py'
```

