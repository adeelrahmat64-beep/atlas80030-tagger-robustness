# Targeted audit remediation log — round 1

Date: 2026-09-09. Scope is limited to C-1, M-1--M-4, and the requested minor
correctness/reproducibility guards. No reconstruction, cache, model training,
prediction, or new physics-analysis campaign was run.

## C-1 — STALE ARTIFACT / WORDING-CLAIM ISSUE

Moved `EPJC_Clean_Submission/atlas_sn_FINAL.tex` and `.pdf` into
`EPJC_Clean_Submission/_archive_dead_v1/`. Added hash-bound do-not-submit and
active-directory status notices. No new manuscript was written.

## M-1 — CONFIRMED BUG

Fixed `shap_delta.py` so score drift is masked by the same `all`, `signal`, or
`background` selection as each SHAP summary. Generated round 3 directly from
the saved round-2 per-jet relation CSVs. TreeSHAP, model, pairs, identities,
sample sizes, nested samples, seed, and attribution fields were not changed.

Round-3 cross-check: 540/540 non-score per-feature records equal round 2; all
36 systematic/sample-size/subset score groups match their per-jet rows; PASS.
See `../shap_delta_final_production_round3/score_drift_crosscheck.csv`.

## M-2 — WORDING/CLAIM ISSUE; NOT A PROJECT NUMERICAL BUG

Retrieved the exact official `calc.py` and `plot_everything.py` at commit
`5bc3c8d70cf8736d24b969972a3dd365017a9e24`, preserved them locally, and
verified their hashes. Official `calc.py` produces five measured values plus a
plot-closing duplicate. Official `plot_everything.py` assigns all six entries
through index 5, so the 1.35--1.55 TeV plotting interval receives the duplicate
10.797788710250034%, not a measurement and not the 20.28012040699554% maximum.
The maximum begins at 1.55 TeV. The current project output matches this exact
convention: bin total modelling 12.983124731703303%, bin total uncertainty
26.987625616368145%, inclusive total 25.82855586414511%. `atlas_metrics.py`
and frozen metrics outputs were not changed; zero pT bins changed.

## M-3 — STALE ARTIFACT

The old fingerprint `0d53c1d8...` result (7/3,509 failures, maximum relative
deviation 0.0482926191) is superseded. Frozen fingerprint `c28a2625...` gives
0/3,509 failures, then 0/14,103 failures in the expanded all-shards test at
relative tolerance 1e-3; maximum deviation is 0.000462851531310629. The later
`final_targeted_closure_audit` explicitly excludes ThrustMaj and is not a later
ThrustMaj measurement. Added a superseded notice to the old result directory;
the reconstruction kernel was not changed.

## M-4 — STALE ARTIFACT plus premise qualification

One `bias` shard (100,000 rows) was compared against identity columns from all
104 `test_nominal` shards (10,306,813 rows). Exact key match: 83,223 (83.223%).
Of 16,777 unmatched rows, 12,095 have event numbers absent from test_nominal;
4,682 share an event number but have no exact event/label/eta/phi key. Of the
latter, 3,043 have a same-label nominal candidate but neither eta nor phi
float32 bits match (median nearest delta-R 1.3836, so this is not rounding);
1,639 have no same-label candidate. Bias therefore has different event/jet
content and non-identical axes and remains unpaired. The obsolete 83.341%-match
`bias_paired.npz` was moved to an archive with its hash and is not used by SHAP.

## Minor correctness fixes applied

- Added an explicit finite reconstruction-sentinel exclusion to tagger
  eligibility and a corresponding cache-audit guard. A one-file CER check saw
  7 sentinel rows and 0 that would survive the former finite/multiplicity mask.
- Enforced paired SHAP nominal identities start with `test_nominal_`; read-only
  preflight passed for esup, esdown, cpos, and cer. Pair creation now indexes
  only `test_nominal`, and its default list is the four validated calorimeter
  variations.
- Updated `download_data.py` from 30 to 50 `train_nominal` files to match the
  frozen 5M model metadata.
- Preserved exact official ATLAS reference scripts with Git blob and SHA-256
  hashes.

## Frozen identities preserved

- reconstruction fingerprint: `c28a262540824e9db90f34c421f10c70`
- final model SHA-256: `63a23308efbf2ce90418c8e657730f571db4798fc9ca2fb3692cb986de405f4c`
- `atlas_metrics.py` SHA-256 (unchanged):
  `061fa989094a303507b50080d5886b3a6767b9efa11105ec9e4eb835657a807e`

No local commands remain to be run.
