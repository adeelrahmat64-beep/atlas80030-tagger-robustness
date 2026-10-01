# Final targeted strict-closure audit

ThrustMaj was excluded from all computations in this audit.
Production NPZ files were neither read for reconstruction nor written.

## Endpoint

- Expanded sample: 14,103 jets.
- Remaining strict failures: 153 rows on 44 physical jets.
- Classification counts: {'1_proven_implementation_mismatch': 0, '2_proven_numerical_precision_or_degeneracy': 1, '3_consistent_with_information_lost_in_export': 43, '4_unresolved': 0}.
- Further production-code correction scientifically justified: NO.
- RECONSTRUCTION FREEZE = PASS.

## Exact remaining counts

| Observable | Strict failures |
|---|---:|
| Tau1 | 3 |
| Tau2 | 8 |
| Tau3 | 17 |
| Tau4 | 24 |
| Split12 | 9 |
| Split23 | 11 |
| ECF1 | 1 |
| ECF2 | 10 |
| ECF3 | 40 |
| C2 | 9 |
| D2 | 4 |
| L2 | 6 |
| L3 | 4 |
| Qw | 7 |

## Control results

The recoverable C++-order float64 ECF replay changed the current results by at most 6.56e-14 relative and fixed 0 failures. Float32 accumulation, float32 rapidity with float64 accumulation, and float64 geometry with float32 accumulation also fixed 0 failures. Raw-energy rapidity fixed 0. The massless/eta control moved a few rows inside tolerance but contradicts the sourced rapidity definition and did not repair the ECF1 sum invariant. Thirteen constituent orderings fixed 0 failures. The independent full-rescan exclusive-kt replay differed by at most 2.63e-13 relative, fixed 0, and found no merge gap below 1e-6. Physically valid deterministic float32-cell controls accounted for the lone Qw-only jet; the other 43 jets retain a disjoint ECF1 input/output interval. All 9 failed C2 and all 4 failed D2 rows are self-consistent with their stored ECF branches, so they are propagated ECF discrepancies. For generalized ECFs, the source-like replay agrees at 4.32e-16 relative for L2 and exactly for L3; all 6 L2 failures and all 4 L3 failures occur on class-3 jets.

## Observable families

| Family | Failed rows | Affected jets |
|---|---:|---:|
| WTA n-subjettiness | 52 | 24 |
| exclusive-kt E-scheme | 27 | 19 |
| generalized ECF | 10 | 6 |
| standard ECF | 64 | 40 |

## Decisive invariant

For 43 of 44 affected jets, the complete float32 interval of stored ECF1 lies above the maximum scalar-pT sum allowed by every released constituent rounding cell. Because ECF1 is exactly sum(pt), arithmetic order, rapidity, clustering, and ECF formula choices cannot create that missing scalar pT.

## Signed bias

The strict-failure rows are one-sided: reconstructed-stored is negative for 147/153 rows and positive for 6/153. ECF3 is negative for 40/40 independent affected jets.  The bulk medians over all 14,103 jets remain near zero, so this is a rare missing-information tail, not a global scale bias.

## Reconstruction-freeze criterion

Freeze reconstruction when (i) authoritative formulas and units are unchanged; (ii) source-like double accumulation and an independent full-rescan clustering replay reveal zero class-1 mismatches; (iii) every strict residual is either reproduced by a deterministic numerical/degeneracy control or linked to a violated input invariant outside the combined float32 cells; and (iv) no jet remains class 4.  Do not require zero strict rows when the released constituents demonstrably lack information used by the stored high-level branch.

## Per-jet classification

| Physical jet | Failed observables | Class | Missing scalar pT (MeV) |
|---|---|---:|---:|
| angular/angular_001.h5:81361 | Tau2, Tau3, Tau4, Split23, ECF2, ECF3, Qw | 3 | 96.4644 |
| angular/angular_019.h5:55617 | ECF3 | 3 | 50.9928 |
| angular/angular_042.h5:9858 | ECF3 | 3 | 153.042 |
| angular/angular_042.h5:34242 | Tau1, Tau2, Tau3, Tau4 | 3 | 97.2425 |
| cluster/cluster_019.h5:7054 | Tau3, Tau4, ECF3 | 3 | 413.527 |
| cluster/cluster_019.h5:75828 | Tau2, Tau3, Tau4, ECF3 | 3 | 56.5173 |
| cluster/cluster_034.h5:57977 | Tau1, Tau2, Tau3, Tau4, Split12, ECF2, ECF3, C2, D2, L2, L3, Qw | 3 | 179.764 |
| dipole/dipole_003.h5:64012 | ECF3 | 3 | 139.236 |
| dipole/dipole_008.h5:90676 | Qw | 2 | 0.0561142 |
| dipole/dipole_014.h5:60990 | Tau3, Tau4, ECF3 | 3 | 201.894 |
| string/string_000.h5:37014 | Split23, ECF3 | 3 | 126.76 |
| string/string_001.h5:89362 | Tau2, Tau3, Tau4, Split23, ECF2, ECF3, C2, L2 | 3 | 654.547 |
| string/string_020.h5:58420 | Split12, ECF2, ECF3 | 3 | 933.255 |
| string/string_022.h5:46204 | Tau3, Tau4 | 3 | 167.738 |
| string/string_022.h5:66159 | Tau3, Tau4, ECF3, C2, D2, L2, L3 | 3 | 264.188 |
| string/string_039.h5:58667 | Tau4, ECF3, C2, D2, L2 | 3 | 96.6037 |
| string/string_041.h5:71903 | Tau1, Tau2, Tau3, Tau4, Split12, Split23, ECF2, ECF3, C2, D2, L2, L3, Qw | 3 | 144.569 |
| string/string_045.h5:16529 | ECF3 | 3 | 86.6116 |
| string/string_048.h5:38616 | Tau4, Split23, ECF3 | 3 | 62.5473 |
| test_nominal/test_nominal_026.h5:33937 | ECF3 | 3 | 221.242 |
| test_nominal/test_nominal_035.h5:27156 | Tau2, Tau3, Tau4, Split12, Split23, ECF1, ECF2, ECF3, L2, L3, Qw | 3 | 6891.46 |
| test_nominal/test_nominal_042.h5:73812 | Tau4, Split12, ECF3, C2 | 3 | 79.5275 |
| test_nominal/test_nominal_057.h5:24070 | ECF3 | 3 | 76.2242 |
| test_nominal/test_nominal_070.h5:77795 | ECF3 | 3 | 131.555 |
| test_nominal/test_nominal_072.h5:41739 | Tau2, Tau3, Tau4, Split12, ECF2, ECF3, C2 | 3 | 148.798 |
| test_nominal/test_nominal_077.h5:96689 | Tau4, ECF3 | 3 | 288.977 |
| test_nominal/test_nominal_080.h5:76851 | Split23, ECF3, Qw | 3 | 73.5313 |
| ttbar_herwig/ttbar_herwig_000.h5:10529 | Tau4, ECF3, Qw | 3 | 99.5034 |
| ttbar_herwig/ttbar_herwig_000.h5:14155 | Tau4, ECF3 | 3 | 84.5492 |
| ttbar_herwig/ttbar_herwig_000.h5:18606 | Tau3, Tau4, ECF2, ECF3, C2 | 3 | 95.4796 |
| ttbar_herwig/ttbar_herwig_000.h5:39148 | Tau3, Tau4, ECF3, C2 | 3 | 80.3639 |
| ttbar_herwig/ttbar_herwig_000.h5:86907 | ECF3 | 3 | 76.8163 |
| ttbar_herwig/ttbar_herwig_001.h5:21376 | ECF3 | 3 | 69.5533 |
| ttbar_herwig/ttbar_herwig_001.h5:27171 | Tau3, Tau4, Split23, ECF3 | 3 | 89.1497 |
| ttbar_herwig/ttbar_herwig_001.h5:36420 | ECF3 | 3 | 79.5059 |
| ttbar_herwig/ttbar_herwig_001.h5:44955 | Split12, ECF2, ECF3 | 3 | 165.298 |
| ttbar_herwig/ttbar_herwig_001.h5:68041 | Split12, Split23 | 3 | 42.712 |
| ttbar_herwig/ttbar_herwig_001.h5:73003 | Tau4, Split23, ECF3 | 3 | 106.981 |
| ttbar_pythia/ttbar_pythia_000.h5:15795 | Tau3, Tau4, ECF2, ECF3 | 3 | 165.988 |
| ttbar_pythia/ttbar_pythia_000.h5:37626 | ECF3 | 3 | 96.889 |
| ttbar_pythia/ttbar_pythia_000.h5:60592 | Split12, ECF3 | 3 | 92.7188 |
| ttbar_pythia/ttbar_pythia_000.h5:98064 | ECF3 | 3 | 123.747 |
| ttbar_pythia/ttbar_pythia_001.h5:80941 | Tau3, Tau4, Split23, ECF3 | 3 | 84.6775 |
| ttbar_pythia/ttbar_pythia_001.h5:84680 | ECF3 | 3 | 86.8241 |

**RECONSTRUCTION FREEZE = PASS**
