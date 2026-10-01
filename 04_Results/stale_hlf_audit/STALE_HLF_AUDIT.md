# Stored high-level branch audit

Rendered (UTC): 2026-09-14T09:00:13.162881+00:00
Current script SHA-256: 1b19e21706e0a953fadb1428e8aef85a1336671353036282799cd0c3b01ac4dc
Full-scan script SHA-256: 1b19e21706e0a953fadb1428e8aef85a1336671353036282799cd0c3b01ac4dc
Nominal pool: 10,306,813 jets in 104 files

| sample | pairability | exact-key hits (diagnostic) | exact float32 HLF identity | constituent evidence |
|---|---|---:|---|---|
| esup | exact key | 5,000,000/5,000,000 (100.000000%) | 75,000,000/75,000,000 (100.000000%) | relative sum-pT median [%]: +7.406764% |
| esdown | exact key | 5,000,000/5,000,000 (100.000000%) | 75,000,000/75,000,000 (100.000000%) | relative sum-pT median [%]: -7.406822% |
| cer | exact key | 4,995,059/5,000,000 (99.901180%) | 74,925,885/74,925,885 (100.000000%) | relative sum-E RMS [%]: +3.770123% |
| cpos | exact key | 5,000,000/5,000,000 (100.000000%) | 75,000,000/75,000,000 (100.000000%) | constituent-centroid DeltaR median: 4.9387e-05 |
| teg | N/A / not pairable | 2,250,423/5,000,000 (45.008460%) | N/A / not pairable | unpaired multiplicity median: 49 |
| tej | N/A / not pairable | 1,906,619/5,000,000 (38.132380%) | N/A / not pairable | unpaired multiplicity median: 49 |
| tfl | N/A / not pairable | 2,749,571/5,000,000 (54.991420%) | N/A / not pairable | unpaired multiplicity median: 48 |
| tfj | N/A / not pairable | 2,006,933/5,000,000 (40.138660%) | N/A / not pairable | unpaired multiplicity median: 49 |
| bias | N/A / not pairable | 4,167,067/5,000,000 (83.341340%) | N/A / not pairable | N/A: N/A |

## Pairability decisions

- **esup** — unique exact keys; coverage 100.000000% meets 99.95% floor
- **esdown** — unique exact keys; coverage 100.000000% meets 99.95% floor
- **cer** — unique exact keys; coverage 99.901180% meets 99.80% floor
- **cpos** — unique exact keys; coverage 100.000000% meets 99.95% floor
- **teg** — 183,802 duplicate systematic keys (183,802 extra rows); exact join is ambiguous
- **tej** — exact-key coverage 38.132380% is below the validated 99.95% floor
- **tfl** — exact-key coverage 54.991420% is below the validated 99.95% floor
- **tfj** — exact-key coverage 40.138660% is below the validated 99.95% floor
- **bias** — exact-key coverage 83.341340% is below the validated 99.95% floor

## Esup manuscript discrepancy

The full exact-pair arithmetic mean is +6.780753%, which rounds to the manuscript's +6.8%; the full exact-pair median is +7.406764%. Thus mean versus median numerically resolves the discrepancy on the same 5,000,000 pairs. The original source of the manuscript figure is not documented, so this does not establish its historical derivation. No manuscript file was changed.

## Evidence definitions and full statistics

### esup

Kind: relative constituent-pT response
Metric: (sum pT_sys / sum pT_nom) - 1

    {"kind": "relative constituent-pT response", "metric": "(sum pT_sys / sum pT_nom) - 1", "statistics": {"mean": 0.06780752831641004, "median": 0.07406763969314734, "median_absolute": 0.07406763969314734, "n": 5000000, "p05": 0.020192095200065575, "p95": 0.09640886076619884, "rms": 0.07217811263009068}}

### esdown

Kind: relative constituent-pT response
Metric: (sum pT_sys / sum pT_nom) - 1

    {"kind": "relative constituent-pT response", "metric": "(sum pT_sys / sum pT_nom) - 1", "statistics": {"mean": -0.06780748663564062, "median": -0.0740682185926475, "median_absolute": 0.0740682185926475, "n": 5000000, "p05": -0.09640886819531982, "p95": -0.020191653048153298, "rms": 0.07217813615979277}}

### cer

Kind: constituent energy-response width
Metric: (sum E_sys / sum E_nom) - 1; RMS and median absolute response are primary

    {"kind": "constituent energy-response width", "metric": "(sum E_sys / sum E_nom) - 1; RMS and median absolute response are primary", "statistics": {"mean": -8.520580291677228e-06, "median": -3.085762653953772e-06, "median_absolute": 0.021428503076922234, "n": 4995059, "p05": -0.060257793361747324, "p95": 0.06027128334870321, "rms": 0.037701231360764705}}

### cpos

Kind: constituent angular displacement
Metric: DeltaR of the pT-weighted constituent eta/phi centroid

    {"kind": "constituent angular displacement", "metric": "DeltaR of the pT-weighted constituent eta/phi centroid", "nonzero_fraction": 0.794122, "statistics": {"mean": 0.00018941071815245912, "median": 4.9386986435694614e-05, "median_absolute": 4.9386986435694614e-05, "n": 5000000, "p05": 0.0, "p95": 0.0007119460347735369, "rms": 0.0005939859298195125}}

### teg

Kind: unpaired constituent multiplicity/content
Metric: descriptive distributions only; no pairwise identity fraction is valid

    {"kind": "unpaired constituent multiplicity/content", "metric": "descriptive distributions only; no pairwise identity fraction is valid", "nominal_distribution": {"mean": 55.81870011612707, "median": 50.0, "n": 10306813, "p05": 22.0, "p95": 109.0}, "systematic_distribution": {"mean": 54.0322384, "median": 49.0, "n": 5000000, "p05": 20.0, "p95": 107.0}}

### tej

Kind: unpaired constituent multiplicity/content
Metric: descriptive distributions only; no pairwise identity fraction is valid

    {"kind": "unpaired constituent multiplicity/content", "metric": "descriptive distributions only; no pairwise identity fraction is valid", "nominal_distribution": {"mean": 55.81870011612707, "median": 50.0, "n": 10306813, "p05": 22.0, "p95": 109.0}, "systematic_distribution": {"mean": 54.4682718, "median": 49.0, "n": 5000000, "p05": 21.0, "p95": 107.0}}

### tfl

Kind: unpaired constituent multiplicity/content
Metric: descriptive distributions only; no pairwise identity fraction is valid

    {"kind": "unpaired constituent multiplicity/content", "metric": "descriptive distributions only; no pairwise identity fraction is valid", "nominal_distribution": {"mean": 55.81870011612707, "median": 50.0, "n": 10306813, "p05": 22.0, "p95": 109.0}, "systematic_distribution": {"mean": 53.3101882, "median": 48.0, "n": 5000000, "p05": 19.0, "p95": 105.0}}

### tfj

Kind: unpaired constituent multiplicity/content
Metric: descriptive distributions only; no pairwise identity fraction is valid

    {"kind": "unpaired constituent multiplicity/content", "metric": "descriptive distributions only; no pairwise identity fraction is valid", "nominal_distribution": {"mean": 55.81870011612707, "median": 50.0, "n": 10306813, "p05": 22.0, "p95": 109.0}, "systematic_distribution": {"mean": 54.1785884, "median": 49.0, "n": 5000000, "p05": 20.0, "p95": 107.0}}

### bias

Kind: N/A
Metric: N/A because exact one-to-one pairing is invalid

    {"kind": "N/A", "metric": "N/A because exact one-to-one pairing is invalid"}


## Exact per-branch float32 identity counts

### esup

| branch | identical | compared | fraction |
|---|---:|---:|---:|
| fjet_Tau1_wta | 5000000 | 5000000 | 1.000000000 |
| fjet_Tau2_wta | 5000000 | 5000000 | 1.000000000 |
| fjet_Tau3_wta | 5000000 | 5000000 | 1.000000000 |
| fjet_Tau4_wta | 5000000 | 5000000 | 1.000000000 |
| fjet_Split12 | 5000000 | 5000000 | 1.000000000 |
| fjet_Split23 | 5000000 | 5000000 | 1.000000000 |
| fjet_ECF1 | 5000000 | 5000000 | 1.000000000 |
| fjet_ECF2 | 5000000 | 5000000 | 1.000000000 |
| fjet_ECF3 | 5000000 | 5000000 | 1.000000000 |
| fjet_C2 | 5000000 | 5000000 | 1.000000000 |
| fjet_D2 | 5000000 | 5000000 | 1.000000000 |
| fjet_L2 | 5000000 | 5000000 | 1.000000000 |
| fjet_L3 | 5000000 | 5000000 | 1.000000000 |
| fjet_Qw | 5000000 | 5000000 | 1.000000000 |
| fjet_ThrustMaj | 5000000 | 5000000 | 1.000000000 |

### esdown

| branch | identical | compared | fraction |
|---|---:|---:|---:|
| fjet_Tau1_wta | 5000000 | 5000000 | 1.000000000 |
| fjet_Tau2_wta | 5000000 | 5000000 | 1.000000000 |
| fjet_Tau3_wta | 5000000 | 5000000 | 1.000000000 |
| fjet_Tau4_wta | 5000000 | 5000000 | 1.000000000 |
| fjet_Split12 | 5000000 | 5000000 | 1.000000000 |
| fjet_Split23 | 5000000 | 5000000 | 1.000000000 |
| fjet_ECF1 | 5000000 | 5000000 | 1.000000000 |
| fjet_ECF2 | 5000000 | 5000000 | 1.000000000 |
| fjet_ECF3 | 5000000 | 5000000 | 1.000000000 |
| fjet_C2 | 5000000 | 5000000 | 1.000000000 |
| fjet_D2 | 5000000 | 5000000 | 1.000000000 |
| fjet_L2 | 5000000 | 5000000 | 1.000000000 |
| fjet_L3 | 5000000 | 5000000 | 1.000000000 |
| fjet_Qw | 5000000 | 5000000 | 1.000000000 |
| fjet_ThrustMaj | 5000000 | 5000000 | 1.000000000 |

### cer

| branch | identical | compared | fraction |
|---|---:|---:|---:|
| fjet_Tau1_wta | 4995059 | 4995059 | 1.000000000 |
| fjet_Tau2_wta | 4995059 | 4995059 | 1.000000000 |
| fjet_Tau3_wta | 4995059 | 4995059 | 1.000000000 |
| fjet_Tau4_wta | 4995059 | 4995059 | 1.000000000 |
| fjet_Split12 | 4995059 | 4995059 | 1.000000000 |
| fjet_Split23 | 4995059 | 4995059 | 1.000000000 |
| fjet_ECF1 | 4995059 | 4995059 | 1.000000000 |
| fjet_ECF2 | 4995059 | 4995059 | 1.000000000 |
| fjet_ECF3 | 4995059 | 4995059 | 1.000000000 |
| fjet_C2 | 4995059 | 4995059 | 1.000000000 |
| fjet_D2 | 4995059 | 4995059 | 1.000000000 |
| fjet_L2 | 4995059 | 4995059 | 1.000000000 |
| fjet_L3 | 4995059 | 4995059 | 1.000000000 |
| fjet_Qw | 4995059 | 4995059 | 1.000000000 |
| fjet_ThrustMaj | 4995059 | 4995059 | 1.000000000 |

### cpos

| branch | identical | compared | fraction |
|---|---:|---:|---:|
| fjet_Tau1_wta | 5000000 | 5000000 | 1.000000000 |
| fjet_Tau2_wta | 5000000 | 5000000 | 1.000000000 |
| fjet_Tau3_wta | 5000000 | 5000000 | 1.000000000 |
| fjet_Tau4_wta | 5000000 | 5000000 | 1.000000000 |
| fjet_Split12 | 5000000 | 5000000 | 1.000000000 |
| fjet_Split23 | 5000000 | 5000000 | 1.000000000 |
| fjet_ECF1 | 5000000 | 5000000 | 1.000000000 |
| fjet_ECF2 | 5000000 | 5000000 | 1.000000000 |
| fjet_ECF3 | 5000000 | 5000000 | 1.000000000 |
| fjet_C2 | 5000000 | 5000000 | 1.000000000 |
| fjet_D2 | 5000000 | 5000000 | 1.000000000 |
| fjet_L2 | 5000000 | 5000000 | 1.000000000 |
| fjet_L3 | 5000000 | 5000000 | 1.000000000 |
| fjet_Qw | 5000000 | 5000000 | 1.000000000 |
| fjet_ThrustMaj | 5000000 | 5000000 | 1.000000000 |

### teg

| branch | identical | compared | fraction |
|---|---:|---:|---:|
| fjet_Tau1_wta | N/A | N/A | N/A |
| fjet_Tau2_wta | N/A | N/A | N/A |
| fjet_Tau3_wta | N/A | N/A | N/A |
| fjet_Tau4_wta | N/A | N/A | N/A |
| fjet_Split12 | N/A | N/A | N/A |
| fjet_Split23 | N/A | N/A | N/A |
| fjet_ECF1 | N/A | N/A | N/A |
| fjet_ECF2 | N/A | N/A | N/A |
| fjet_ECF3 | N/A | N/A | N/A |
| fjet_C2 | N/A | N/A | N/A |
| fjet_D2 | N/A | N/A | N/A |
| fjet_L2 | N/A | N/A | N/A |
| fjet_L3 | N/A | N/A | N/A |
| fjet_Qw | N/A | N/A | N/A |
| fjet_ThrustMaj | N/A | N/A | N/A |

### tej

| branch | identical | compared | fraction |
|---|---:|---:|---:|
| fjet_Tau1_wta | N/A | N/A | N/A |
| fjet_Tau2_wta | N/A | N/A | N/A |
| fjet_Tau3_wta | N/A | N/A | N/A |
| fjet_Tau4_wta | N/A | N/A | N/A |
| fjet_Split12 | N/A | N/A | N/A |
| fjet_Split23 | N/A | N/A | N/A |
| fjet_ECF1 | N/A | N/A | N/A |
| fjet_ECF2 | N/A | N/A | N/A |
| fjet_ECF3 | N/A | N/A | N/A |
| fjet_C2 | N/A | N/A | N/A |
| fjet_D2 | N/A | N/A | N/A |
| fjet_L2 | N/A | N/A | N/A |
| fjet_L3 | N/A | N/A | N/A |
| fjet_Qw | N/A | N/A | N/A |
| fjet_ThrustMaj | N/A | N/A | N/A |

### tfl

| branch | identical | compared | fraction |
|---|---:|---:|---:|
| fjet_Tau1_wta | N/A | N/A | N/A |
| fjet_Tau2_wta | N/A | N/A | N/A |
| fjet_Tau3_wta | N/A | N/A | N/A |
| fjet_Tau4_wta | N/A | N/A | N/A |
| fjet_Split12 | N/A | N/A | N/A |
| fjet_Split23 | N/A | N/A | N/A |
| fjet_ECF1 | N/A | N/A | N/A |
| fjet_ECF2 | N/A | N/A | N/A |
| fjet_ECF3 | N/A | N/A | N/A |
| fjet_C2 | N/A | N/A | N/A |
| fjet_D2 | N/A | N/A | N/A |
| fjet_L2 | N/A | N/A | N/A |
| fjet_L3 | N/A | N/A | N/A |
| fjet_Qw | N/A | N/A | N/A |
| fjet_ThrustMaj | N/A | N/A | N/A |

### tfj

| branch | identical | compared | fraction |
|---|---:|---:|---:|
| fjet_Tau1_wta | N/A | N/A | N/A |
| fjet_Tau2_wta | N/A | N/A | N/A |
| fjet_Tau3_wta | N/A | N/A | N/A |
| fjet_Tau4_wta | N/A | N/A | N/A |
| fjet_Split12 | N/A | N/A | N/A |
| fjet_Split23 | N/A | N/A | N/A |
| fjet_ECF1 | N/A | N/A | N/A |
| fjet_ECF2 | N/A | N/A | N/A |
| fjet_ECF3 | N/A | N/A | N/A |
| fjet_C2 | N/A | N/A | N/A |
| fjet_D2 | N/A | N/A | N/A |
| fjet_L2 | N/A | N/A | N/A |
| fjet_L3 | N/A | N/A | N/A |
| fjet_Qw | N/A | N/A | N/A |
| fjet_ThrustMaj | N/A | N/A | N/A |

### bias

| branch | identical | compared | fraction |
|---|---:|---:|---:|
| fjet_Tau1_wta | N/A | N/A | N/A |
| fjet_Tau2_wta | N/A | N/A | N/A |
| fjet_Tau3_wta | N/A | N/A | N/A |
| fjet_Tau4_wta | N/A | N/A | N/A |
| fjet_Split12 | N/A | N/A | N/A |
| fjet_Split23 | N/A | N/A | N/A |
| fjet_ECF1 | N/A | N/A | N/A |
| fjet_ECF2 | N/A | N/A | N/A |
| fjet_ECF3 | N/A | N/A | N/A |
| fjet_C2 | N/A | N/A | N/A |
| fjet_D2 | N/A | N/A | N/A |
| fjet_L2 | N/A | N/A | N/A |
| fjet_L3 | N/A | N/A | N/A |
| fjet_Qw | N/A | N/A | N/A |
| fjet_ThrustMaj | N/A | N/A | N/A |

