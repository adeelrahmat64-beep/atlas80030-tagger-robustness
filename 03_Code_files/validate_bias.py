from pathlib import Path
import numpy as np

from atlas_config import DATA_ROOT

# ============================================================
# ATLAS 80030 — Bias NPZ validation
# This script ONLY reads files. It does not modify anything.
# ============================================================

BIAS_DIR = DATA_ROOT / "hl_recomputed" / "bias"

EXPECTED_FILES = 50
EXPECTED_JETS_PER_FILE = 100_000
EXPECTED_TOTAL_JETS = EXPECTED_FILES * EXPECTED_JETS_PER_FILE

HL_FEATURES = [
    "fjet_Tau1_wta",
    "fjet_Tau2_wta",
    "fjet_Tau3_wta",
    "fjet_Tau4_wta",
    "fjet_Split12",
    "fjet_Split23",
    "fjet_ECF1",
    "fjet_ECF2",
    "fjet_ECF3",
    "fjet_C2",
    "fjet_D2",
    "fjet_L2",
    "fjet_L3",
    "fjet_Qw",
    "fjet_ThrustMaj",
]

errors = []
warnings = []

print("=" * 78)
print("ATLAS 80030 — BIAS NPZ VALIDATION")
print("=" * 78)
print(f"Directory: {BIAS_DIR}")
print()

# ------------------------------------------------------------
# 1. Directory exists
# ------------------------------------------------------------

if not BIAS_DIR.is_dir():
    raise SystemExit(f"ERROR: directory does not exist:\n{BIAS_DIR}")

# ------------------------------------------------------------
# 2. Find NPZ files
# ------------------------------------------------------------

files = sorted(BIAS_DIR.glob("bias_*.npz"))

print(f"[1] NPZ files found: {len(files)}")

if len(files) != EXPECTED_FILES:
    errors.append(
        f"Expected {EXPECTED_FILES} NPZ files, found {len(files)}"
    )

# Check expected filenames
expected_names = {f"bias_{i:03d}.npz" for i in range(EXPECTED_FILES)}
actual_names = {f.name for f in files}

missing = sorted(expected_names - actual_names)
extra = sorted(actual_names - expected_names)

if missing:
    errors.append(f"Missing files: {missing}")

if extra:
    errors.append(f"Unexpected bias NPZ files: {extra}")

print(f"    Expected sequence: bias_000.npz ... bias_049.npz")

# ------------------------------------------------------------
# 3. Inspect every file
# ------------------------------------------------------------

code_fps = set()
total_jets = 0

print()
print("[2] Checking files...")
print()

for i, path in enumerate(files, start=1):

    try:
        with np.load(path, allow_pickle=False) as data:

            keys = set(data.files)

            # ------------------------------------------------
            # Required feature keys
            # ------------------------------------------------

            missing_features = [
                k for k in HL_FEATURES if k not in keys
            ]

            if missing_features:
                errors.append(
                    f"{path.name}: missing features {missing_features}"
                )

            # ------------------------------------------------
            # Check jet count
            # ------------------------------------------------

            file_lengths = {}

            for feature in HL_FEATURES:
                if feature in data:
                    arr = data[feature]

                    if arr.ndim != 1:
                        errors.append(
                            f"{path.name}: {feature} has shape {arr.shape}, "
                            f"expected 1D"
                        )
                    else:
                        file_lengths[feature] = len(arr)

                        if len(arr) != EXPECTED_JETS_PER_FILE:
                            errors.append(
                                f"{path.name}: {feature} has "
                                f"{len(arr)} entries, expected "
                                f"{EXPECTED_JETS_PER_FILE}"
                            )

                        # ------------------------------------
                        # Numerical validity
                        # ------------------------------------

                        if not np.isfinite(arr).all():
                            bad = np.count_nonzero(~np.isfinite(arr))
                            errors.append(
                                f"{path.name}: {feature} contains "
                                f"{bad} NaN/Inf values"
                            )

            # ------------------------------------------------
            # n_constit
            # ------------------------------------------------

            if "n_constit" not in keys:
                errors.append(f"{path.name}: missing n_constit")
            else:
                nc = data["n_constit"]

                if nc.ndim != 1:
                    errors.append(
                        f"{path.name}: n_constit has shape {nc.shape}"
                    )
                elif len(nc) != EXPECTED_JETS_PER_FILE:
                    errors.append(
                        f"{path.name}: n_constit has {len(nc)} entries, "
                        f"expected {EXPECTED_JETS_PER_FILE}"
                    )
                else:
                    if not np.isfinite(nc).all():
                        errors.append(
                            f"{path.name}: n_constit contains NaN/Inf"
                        )

                    if np.any(nc < 0):
                        errors.append(
                            f"{path.name}: n_constit contains negative values"
                        )

                    if np.any(nc > 200):
                        errors.append(
                            f"{path.name}: n_constit contains values > 200"
                        )

            # ------------------------------------------------
            # code fingerprint
            # ------------------------------------------------

            if "code_fp" not in keys:
                errors.append(f"{path.name}: missing code_fp")
            else:
                fp = data["code_fp"]

                try:
                    if np.ndim(fp) == 0:
                        fp_value = str(fp.item())
                    else:
                        fp_value = str(fp.tolist())

                    code_fps.add(fp_value)

                except Exception as exc:
                    errors.append(
                        f"{path.name}: could not read code_fp: {exc}"
                    )

            # ------------------------------------------------
            # Determine jet count from first available feature
            # ------------------------------------------------

            if "fjet_Tau1_wta" in data:
                total_jets += len(data["fjet_Tau1_wta"])

    except Exception as exc:
        errors.append(
            f"{path.name}: FAILED TO READ FILE: {type(exc).__name__}: {exc}"
        )

    # Progress
    if i == 1 or i % 10 == 0 or i == len(files):
        print(f"    checked {i:2d}/{len(files)}")

# ------------------------------------------------------------
# 4. Cross-file fingerprint check
# ------------------------------------------------------------

print()
print("[3] Code fingerprint check")

if len(code_fps) == 1:
    print(f"    Unique code fingerprints: 1")
    print(f"    Fingerprint: {next(iter(code_fps))}")
elif len(code_fps) == 0:
    print("    ERROR: no readable code fingerprint found")
else:
    print(f"    ERROR: {len(code_fps)} different fingerprints found")
    errors.append(
        f"Multiple code fingerprints found: {sorted(code_fps)}"
    )

# ------------------------------------------------------------
# 5. Total jet count
# ------------------------------------------------------------

print()
print("[4] Total jet count")
print(f"    Expected: {EXPECTED_TOTAL_JETS:,}")
print(f"    Found:    {total_jets:,}")

if total_jets != EXPECTED_TOTAL_JETS:
    errors.append(
        f"Total jets = {total_jets}, expected {EXPECTED_TOTAL_JETS}"
    )

# ------------------------------------------------------------
# 6. Final result
# ------------------------------------------------------------

print()
print("=" * 78)

if errors:
    print("VALIDATION FAILED")
    print("=" * 78)
    print()
    print(f"Number of errors: {len(errors)}")
    print()

    for n, err in enumerate(errors, start=1):
        print(f"{n}. {err}")

    raise SystemExit(1)

else:
    print("VALIDATION PASSED")
    print("=" * 78)
    print()
    print("50/50 NPZ files are present and readable.")
    print("Each file contains 100,000 jets.")
    print("All 15 required high-level features are present.")
    print("All feature arrays have the expected length.")
    print("No NaN/Inf values were found.")
    print("n_constit is valid.")
    print("All files use the same code fingerprint.")
    print("Total jets: 5,000,000")
    print()
    print("BIAS FEATURE CACHE IS READY FOR THE NEXT STAGE.")
