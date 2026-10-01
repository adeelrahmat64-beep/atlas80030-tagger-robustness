# Frozen official ATLAS reference

These files are exact blobs from the official repository
`https://gitlab.cern.ch/atlas/ATLAS-top-tagging-open-data` at commit
`5bc3c8d70cf8736d24b969972a3dd365017a9e24`.

| File | Git blob SHA-1 | SHA-256 |
|---|---|---|
| `calc.py` | `0c82011916c38bc7518855c6d1e4c3ca5740db5f` | `7d932109151ddec1868c39b8d7fb57b4aaa0153abddbb7f5471b8e9524208d14` |
| `plot_everything.py` | `a2bcd16264e95c64d0bd7f48675c3105179e4ba4` | `8cafdc01ab3ef2560c1064c402c4263c306889b4763f2c82b34ec919d35f1999` |

M-2 evidence: `calc.py` truncates the signal-modelling edges with
`pt_bins = full_bins[:high_bin_edge]`, producing five measured bins plus a
sixth plot-closing value. `plot_everything.py` then executes
`extrapolated_sig_model[:max_bin] = sig_model`, so that sixth padding value is
displayed in the 1.35--1.55 TeV interval. Only indices 6 onward receive
`max_sig_model`. This is the exact official plotting convention at the frozen
commit, not a project-specific numerical defect.
