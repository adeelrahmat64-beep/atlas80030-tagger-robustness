"""Strict, audit-only closure check for ATLAS Open Data record 80030.

This driver imports the scientific kernels from ``recompute_hl.py`` without
editing that file.  HDF5 inputs and production NPZ caches are opened read-only.
Only new audit CSV/JSON files are written to the requested output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


FEATURE_ORDER = [
    "Tau1", "Tau2", "Tau3", "Tau4", "Split12", "Split23",
    "ECF1", "ECF2", "ECF3", "C2", "D2", "L2", "L3", "Qw",
    "ThrustMaj",
]

STORED_BRANCH = {
    "Tau1": "fjet_Tau1_wta",
    "Tau2": "fjet_Tau2_wta",
    "Tau3": "fjet_Tau3_wta",
    "Tau4": "fjet_Tau4_wta",
    "Split12": "fjet_Split12",
    "Split23": "fjet_Split23",
    "ECF1": "fjet_ECF1",
    "ECF2": "fjet_ECF2",
    "ECF3": "fjet_ECF3",
    "C2": "fjet_C2",
    "D2": "fjet_D2",
    "L2": "fjet_L2",
    "L3": "fjet_L3",
    "Qw": "fjet_Qw",
    "ThrustMaj": "fjet_ThrustMaj",
}

# Convert the reconstruction output to the units of the stored ATLAS branch.
UNIT_SCALE = {
    "Tau1": 1.0, "Tau2": 1.0, "Tau3": 1.0, "Tau4": 1.0,
    "Split12": 1.0e-6, "Split23": 1.0e-6,
    "ECF1": 1.0e-6, "ECF2": 1.0e-12, "ECF3": 1.0e-18,
    "C2": 1.0, "D2": 1.0, "L2": 1.0, "L3": 1.0,
    "Qw": 1.0e-6, "ThrustMaj": 1.0,
}

# These rules are fixed before looking at the audit results.  A row exceeds
# tolerance only if it exceeds both the relative and the float32-aware absolute
# tolerance.  The protected relative denominator is 1e-6 of the feature's
# robust median magnitude, preventing tiny physical values from creating
# meaningless relative-error explosions.
REL_TOL = 1.0e-3                 # 0.1%
NEAR_ZERO_SCALE_FRACTION = 1.0e-6
FLOAT32_ULP_ALLOWANCE = 8.0
ABS_TOL_MIN = 1.0e-12
SENTINEL_CUT = -900.0

DATASETS = {
    "test_nominal": "test_nominal",
    "angular": "modelling/angular",
    "cluster": "modelling/cluster",
    "dipole": "modelling/dipole",
    "string": "modelling/string",
    "ttbar_herwig": "modelling/ttbar_herwig",
    "ttbar_pythia": "modelling/ttbar_pythia",
}


def as_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump_json(path, obj):
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=False, default=as_builtin,
                   allow_nan=False),
        encoding="utf-8",
    )


def import_reconstruction(path):
    module_dir = str(path.resolve().parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location("atlas_recompute_hl", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_row_count(path):
    with h5py.File(path, "r") as handle:
        return int(handle["fjet_clus_pt"].shape[0])


def spaced_positions(n, count):
    if n <= count:
        return list(range(n))
    return sorted(set(int(round(x)) for x in np.linspace(0, n - 1, count)))


def choose_files(directory, dataset, files_per_dataset=None):
    inventory = [(path, file_row_count(path))
                 for path in sorted(directory.glob("*.h5"))]
    nonempty = [(path, rows) for path, rows in inventory if rows > 0]
    main = [(path, rows) for path, rows in nonempty if rows >= 1000]
    tiny = [(path, rows) for path, rows in nonempty if rows < 1000]
    if files_per_dataset == 0:
        desired = len(main)
    elif files_per_dataset is None:
        desired = 4 if dataset == "test_nominal" else 3
    else:
        desired = min(files_per_dataset, len(main))
    selected = [main[i] for i in spaced_positions(len(main), desired)]
    # Tiny official terminal shards are included in full as constituent-edge
    # checks but do not consume the main deterministic sample budget.
    selected.extend(tiny)
    return inventory, selected, tiny


def deterministic_rng(seed, *parts):
    token = "|".join(str(part) for part in (seed,) + parts)
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def source_diagnostics(pt, eta, phi, energy, n_constit, max_constits):
    n = int(n_constit)
    active = slice(0, n)
    pta = np.asarray(pt[active], dtype=np.float64)
    etaa = np.asarray(eta[active], dtype=np.float64)
    phia = np.asarray(phi[active], dtype=np.float64)
    ena = np.asarray(energy[active], dtype=np.float64)
    finite = (np.isfinite(pta) & np.isfinite(etaa) & np.isfinite(phia)
              & np.isfinite(ena))
    momentum = pta * np.cosh(etaa)
    active_nonpositive = int(np.sum(pta <= 0.0))
    tail_positive = int(np.sum(np.asarray(pt[n:], dtype=np.float64) > 0.0))
    return {
        "n_constit": n,
        "at_200_constituent_cap": bool(n == max_constits),
        "active_constituents_all_finite": bool(finite.all()),
        "active_nonpositive_pt_count": active_nonpositive,
        "positive_pt_after_active_prefix": tail_positive,
        "padding_is_contiguous": bool(active_nonpositive == 0 and tail_positive == 0),
        "energy_clamp_count": int(np.sum(ena < momentum)) if n else 0,
        "sum_pt": float(np.sum(pta)) if n else 0.0,
        "min_positive_pt": float(np.min(pta[pta > 0.0])) if np.any(pta > 0.0) else None,
        "max_pt": float(np.max(pta)) if n else None,
        "max_abs_eta": float(np.max(np.abs(etaa))) if n else None,
    }


def read_samples(data_root, sample_per_dataset, seed, max_constits,
                 files_per_dataset=None):
    clus_names = ["fjet_clus_pt", "fjet_clus_eta", "fjet_clus_phi",
                  "fjet_clus_E"]
    parts = {name: [] for name in clus_names}
    stored_parts = {name: [] for name in FEATURE_ORDER}
    metadata = []
    sampling = {}

    for dataset, relative in DATASETS.items():
        directory = data_root / relative
        inventory, chosen, tiny = choose_files(
            directory, dataset, files_per_dataset
        )
        main_chosen = [(path, rows) for path, rows in chosen if rows >= 1000]
        per_file = int(math.ceil(sample_per_dataset / max(len(main_chosen), 1)))
        dataset_rows = 0
        file_records = []
        for path, rows in chosen:
            take = rows if rows < 1000 else min(per_file, rows)
            rng = deterministic_rng(seed, dataset, path.name)
            indices = np.sort(rng.choice(rows, size=take, replace=False)).astype(np.int64)
            with h5py.File(path, "r") as handle:
                for name in clus_names:
                    parts[name].append(
                        handle[name][indices, :max_constits].astype(np.float64)
                    )
                for feature in FEATURE_ORDER:
                    stored_parts[feature].append(
                        handle[STORED_BRANCH[feature]][indices].astype(np.float64)
                    )
                events = (handle["EventInfo_mcEventNumber"][indices]
                          if "EventInfo_mcEventNumber" in handle
                          else np.full(take, -1, dtype=np.int64))
                labels = handle["labels"][indices]
            for row, event, label in zip(indices, events, labels):
                metadata.append({
                    "dataset": dataset,
                    "relative_dataset_path": relative,
                    "file": path.name,
                    "jet_index": int(row),
                    "event_identifier": None if int(event) == -1 else int(event),
                    "label": int(label),
                })
            dataset_rows += take
            file_records.append({
                "file": path.name,
                "file_rows": rows,
                "sampled_rows": take,
                "first_sampled_index": int(indices[0]) if take else None,
                "last_sampled_index": int(indices[-1]) if take else None,
            })
        sampling[dataset] = {
            "relative_path": relative,
            "hdf5_file_count": len(inventory),
            "hdf5_nonempty_file_count": sum(rows > 0 for _, rows in inventory),
            "hdf5_total_rows": sum(rows for _, rows in inventory),
            "sampled_rows": dataset_rows,
            "files": file_records,
            "tiny_terminal_files_included": [
                {"file": path.name, "rows": rows} for path, rows in tiny
            ],
        }

    clus = {name: np.concatenate(values, axis=0)
            for name, values in parts.items()}
    stored = {name: np.concatenate(values, axis=0)
              for name, values in stored_parts.items()}
    return clus, stored, metadata, sampling


def calculate_features(reco, clus):
    pt = clus["fjet_clus_pt"]
    eta = clus["fjet_clus_eta"]
    phi = clus["fjet_clus_phi"]
    energy = clus["fjet_clus_E"]
    n_constit = (pt > 0.0).sum(axis=1).astype(np.int64)
    use_e = 1 if reco.USE_CLUS_E else 0

    yrap = reco.constituent_rapidity(pt, eta, energy, n_constit, use_e)
    split, qw, axeta, axphi, naxes = reco.kt_observables(
        pt, eta, phi, energy, n_constit, 4, use_e
    )
    tau_raw = reco.nsubjettiness(
        pt, eta, phi, energy, n_constit, axeta, axphi, naxes, 4, 1.0,
        use_e,
    )
    ecf = reco.ecf123(pt, yrap, phi, n_constit, beta=1.0, block=256)
    thrust = reco.thrust_3d(
        pt, eta, phi, energy, n_constit, 360, use_e
    )
    gen3 = reco.gen_ecf3(pt, yrap, phi, n_constit, 1.0)
    gen2_b2 = reco.gen_ecf2(pt, yrap, phi, n_constit, 2.0)

    sumpt = ecf[:, 0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        tau = tau_raw / sumpt[:, None]
        c2 = ecf[:, 2] * ecf[:, 0] / ecf[:, 1] ** 2
        d2 = ecf[:, 2] * ecf[:, 0] ** 3 / ecf[:, 1] ** 3
        l2 = gen3[:, 2] / gen2_b2 ** 1.5
        l3 = gen3[:, 0] / gen3[:, 2] ** (1.0 / 3.0)

    raw = {
        "Tau1": tau[:, 0], "Tau2": tau[:, 1],
        "Tau3": tau[:, 2], "Tau4": tau[:, 3],
        "Split12": split[:, 0], "Split23": split[:, 1],
        "ECF1": ecf[:, 0], "ECF2": ecf[:, 1], "ECF3": ecf[:, 2],
        "C2": c2, "D2": d2, "L2": l2, "L3": l3,
        "Qw": qw, "ThrustMaj": thrust[:, 0],
    }
    scaled = {name: raw[name] * UNIT_SCALE[name] for name in FEATURE_ORDER}
    extras = {
        "n_constit": n_constit,
        "yrap": yrap,
        "thrust_minor": thrust[:, 1],
        "thrust_perpendicular_control": thrust[:, 2],
        "gen3": gen3,
        "gen2_b2": gen2_b2,
    }
    return scaled, extras


def percentile_summary(values):
    if values.size == 0:
        return {"median": None, "p95": None, "p99": None,
                "p99_9": None, "max": None}
    q = np.percentile(values, [50.0, 95.0, 99.0, 99.9, 100.0])
    return {
        "median": float(q[0]), "p95": float(q[1]),
        "p99": float(q[2]), "p99_9": float(q[3]), "max": float(q[4]),
    }


def analyze_one_feature(feature, stored, reconstructed, metadata, clus,
                        n_constit, row_mask=None, max_outliers=12):
    if row_mask is None:
        row_mask = np.ones(stored.shape[0], dtype=bool)
    finite_stored = np.isfinite(stored)
    finite_reco = np.isfinite(reconstructed)
    sentinel = stored <= SENTINEL_CUT
    valid = row_mask & finite_stored & finite_reco & ~sentinel
    invalid_stored = row_mask & (~finite_stored | sentinel)
    invalid_reco = row_mask & finite_stored & ~sentinel & ~finite_reco

    s = stored[valid]
    r = reconstructed[valid]
    mag = np.maximum(np.abs(s), np.abs(r))
    nonzero_mag = mag[mag > 0.0]
    robust_scale = (float(np.median(nonzero_mag)) if nonzero_mag.size
                    else 1.0)
    protection = max(robust_scale * NEAR_ZERO_SCALE_FRACTION, ABS_TOL_MIN)
    absolute_tolerance = max(
        robust_scale * FLOAT32_ULP_ALLOWANCE * np.finfo(np.float32).eps,
        ABS_TOL_MIN,
    )
    abs_error = np.abs(r - s)
    rel_error = abs_error / np.maximum(mag, protection)
    near_zero = mag <= protection
    above = (abs_error > absolute_tolerance) & (rel_error > REL_TOL)
    above_nonzero = above & ~near_zero
    f32_equal = r.astype(np.float32) == s.astype(np.float32)

    valid_indices = np.flatnonzero(valid)
    selected = set()
    if valid_indices.size:
        for local in np.argsort(rel_error)[-max_outliers // 2:]:
            selected.add(int(valid_indices[local]))
        for local in np.argsort(abs_error)[-max_outliers // 2:]:
            selected.add(int(valid_indices[local]))

    outliers = []
    for global_index in sorted(
        selected,
        key=lambda index: (
            abs(reconstructed[index] - stored[index])
            / max(abs(reconstructed[index]), abs(stored[index]), protection)
        ),
        reverse=True,
    ):
        abs_e = float(abs(reconstructed[global_index] - stored[global_index]))
        magnitude = float(max(abs(reconstructed[global_index]),
                              abs(stored[global_index])))
        rel_e = abs_e / max(magnitude, protection)
        is_near_zero = magnitude <= protection
        diag = source_diagnostics(
            clus["fjet_clus_pt"][global_index],
            clus["fjet_clus_eta"][global_index],
            clus["fjet_clus_phi"][global_index],
            clus["fjet_clus_E"][global_index],
            n_constit[global_index],
            clus["fjet_clus_pt"].shape[1],
        )
        if is_near_zero:
            preliminary = "near-zero relative-error artifact candidate"
        elif np.float32(reconstructed[global_index]) == np.float32(stored[global_index]):
            preliminary = "compatible with float32 storage rounding"
        elif abs_e <= absolute_tolerance:
            preliminary = "within float32-aware absolute tolerance"
        elif feature.startswith("Tau") and diag["n_constit"] < int(feature[-1]):
            preliminary = "constituent-count edge case candidate"
        elif diag["at_200_constituent_cap"] or not diag["padding_is_contiguous"]:
            preliminary = "constituent truncation/padding edge case candidate"
        else:
            preliminary = "unresolved by first-pass diagnostics"
        outliers.append({
            **metadata[global_index],
            "stored_value": float(stored[global_index]),
            "reconstructed_value": float(reconstructed[global_index]),
            "absolute_error": abs_e,
            "protected_relative_error": float(rel_e),
            "near_zero": bool(is_near_zero),
            "float32_equal_after_cast": bool(
                np.float32(reconstructed[global_index])
                == np.float32(stored[global_index])
            ),
            "above_hybrid_tolerance": bool(
                abs_e > absolute_tolerance and rel_e > REL_TOL
            ),
            "preliminary_classification": preliminary,
            "source_diagnostics": diag,
        })

    return {
        "feature": feature,
        "unit_scale_applied_to_reconstruction": UNIT_SCALE[feature],
        "n_rows_in_scope": int(np.sum(row_mask)),
        "n_valid": int(np.sum(valid)),
        "n_invalid_or_sentinel_stored": int(np.sum(invalid_stored)),
        "n_nonfinite_reconstructed_with_valid_stored": int(np.sum(invalid_reco)),
        "robust_median_magnitude": robust_scale,
        "protected_relative_denominator_floor": protection,
        "near_zero_count": int(np.sum(near_zero)),
        "relative_tolerance": REL_TOL,
        "absolute_tolerance": absolute_tolerance,
        "absolute_error": percentile_summary(abs_error),
        "protected_relative_error": percentile_summary(rel_error),
        "fraction_above_relative_tolerance": (
            float(np.mean(rel_error > REL_TOL)) if rel_error.size else None
        ),
        "fraction_above_hybrid_tolerance": (
            float(np.mean(above)) if above.size else None
        ),
        "fraction_non_near_zero_above_hybrid_tolerance": (
            float(np.mean(above_nonzero)) if above_nonzero.size else None
        ),
        "fraction_float32_equal_after_cast": (
            float(np.mean(f32_equal)) if f32_equal.size else None
        ),
        "outliers": outliers,
    }


def write_stats_csv(path, rows):
    fields = [
        "scope", "feature", "n_rows_in_scope", "n_valid",
        "n_invalid_or_sentinel_stored",
        "n_nonfinite_reconstructed_with_valid_stored",
        "unit_scale_applied_to_reconstruction", "robust_median_magnitude",
        "protected_relative_denominator_floor", "relative_tolerance",
        "absolute_tolerance", "abs_median", "abs_p95", "abs_p99",
        "abs_p99_9", "abs_max", "rel_median", "rel_p95", "rel_p99",
        "rel_p99_9", "rel_max", "near_zero_count",
        "fraction_above_relative_tolerance",
        "fraction_above_hybrid_tolerance",
        "fraction_non_near_zero_above_hybrid_tolerance",
        "fraction_float32_equal_after_cast",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scope, stat in rows:
            writer.writerow({
                "scope": scope,
                "feature": stat["feature"],
                "n_rows_in_scope": stat["n_rows_in_scope"],
                "n_valid": stat["n_valid"],
                "n_invalid_or_sentinel_stored": stat["n_invalid_or_sentinel_stored"],
                "n_nonfinite_reconstructed_with_valid_stored": stat["n_nonfinite_reconstructed_with_valid_stored"],
                "unit_scale_applied_to_reconstruction": stat["unit_scale_applied_to_reconstruction"],
                "robust_median_magnitude": stat["robust_median_magnitude"],
                "protected_relative_denominator_floor": stat["protected_relative_denominator_floor"],
                "relative_tolerance": stat["relative_tolerance"],
                "absolute_tolerance": stat["absolute_tolerance"],
                "abs_median": stat["absolute_error"]["median"],
                "abs_p95": stat["absolute_error"]["p95"],
                "abs_p99": stat["absolute_error"]["p99"],
                "abs_p99_9": stat["absolute_error"]["p99_9"],
                "abs_max": stat["absolute_error"]["max"],
                "rel_median": stat["protected_relative_error"]["median"],
                "rel_p95": stat["protected_relative_error"]["p95"],
                "rel_p99": stat["protected_relative_error"]["p99"],
                "rel_p99_9": stat["protected_relative_error"]["p99_9"],
                "rel_max": stat["protected_relative_error"]["max"],
                "near_zero_count": stat["near_zero_count"],
                "fraction_above_relative_tolerance": stat["fraction_above_relative_tolerance"],
                "fraction_above_hybrid_tolerance": stat["fraction_above_hybrid_tolerance"],
                "fraction_non_near_zero_above_hybrid_tolerance": stat["fraction_non_near_zero_above_hybrid_tolerance"],
                "fraction_float32_equal_after_cast": stat["fraction_float32_equal_after_cast"],
            })


def write_eventwise_csv(path, metadata, clus, n_constit, stored,
                        reconstructed, full_stats):
    fields = [
        "dataset", "relative_dataset_path", "file", "jet_index",
        "event_identifier", "label", "n_constit", "energy_clamp_count",
        "at_200_constituent_cap", "padding_is_contiguous", "feature",
        "stored_value", "reconstructed_value", "absolute_error",
        "protected_relative_error", "near_zero", "valid_for_closure",
        "float32_equal_after_cast", "above_hybrid_tolerance",
    ]
    diagnostics = [
        source_diagnostics(
            clus["fjet_clus_pt"][i], clus["fjet_clus_eta"][i],
            clus["fjet_clus_phi"][i], clus["fjet_clus_E"][i],
            n_constit[i], clus["fjet_clus_pt"].shape[1],
        )
        for i in range(len(metadata))
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, meta in enumerate(metadata):
            diag = diagnostics[i]
            for feature in FEATURE_ORDER:
                s = float(stored[feature][i])
                r = float(reconstructed[feature][i])
                stat = full_stats[feature]
                valid = (math.isfinite(s) and math.isfinite(r)
                         and s > SENTINEL_CUT)
                if valid:
                    abs_error = abs(r - s)
                    magnitude = max(abs(r), abs(s))
                    relative_error = abs_error / max(
                        magnitude,
                        stat["protected_relative_denominator_floor"],
                    )
                    near_zero = (
                        magnitude
                        <= stat["protected_relative_denominator_floor"]
                    )
                    f32_equal = bool(np.float32(r) == np.float32(s))
                    above = bool(
                        abs_error > stat["absolute_tolerance"]
                        and relative_error > REL_TOL
                    )
                else:
                    abs_error = None
                    relative_error = None
                    near_zero = None
                    f32_equal = None
                    above = None
                writer.writerow({
                    **meta,
                    "n_constit": diag["n_constit"],
                    "energy_clamp_count": diag["energy_clamp_count"],
                    "at_200_constituent_cap": diag["at_200_constituent_cap"],
                    "padding_is_contiguous": diag["padding_is_contiguous"],
                    "feature": feature,
                    "stored_value": s if math.isfinite(s) else "",
                    "reconstructed_value": r if math.isfinite(r) else "",
                    "absolute_error": abs_error,
                    "protected_relative_error": relative_error,
                    "near_zero": near_zero,
                    "valid_for_closure": valid,
                    "float32_equal_after_cast": f32_equal,
                    "above_hybrid_tolerance": above,
                })


def scan_cer_nonfinite(data_root, reco, max_constits):
    cache_dir = data_root / "hl_recomputed" / "cer"
    raw_dir = data_root / "data" / "cer"
    occurrences = []
    sentinel_total = 0
    sentinel_rows = set()

    for cache_path in sorted(cache_dir.glob("*.npz")):
        raw_path = raw_dir / f"{cache_path.stem}.h5"
        with np.load(cache_path, allow_pickle=False) as cache:
            for key in cache.files:
                value = cache[key]
                if value.dtype.kind not in "fc" or value.ndim == 0:
                    continue
                bad = np.flatnonzero(~np.isfinite(value))
                for row in bad:
                    occurrences.append({
                        "cache_file": cache_path.name,
                        "source_file": raw_path.name,
                        "jet_index": int(row),
                        "cache_feature": key,
                        "cache_value": None,
                    })
        with h5py.File(raw_path, "r") as handle:
            l3 = handle["fjet_L3"][:]
        rows = np.flatnonzero(l3 <= SENTINEL_CUT)
        sentinel_total += int(rows.size)
        sentinel_rows.update((raw_path.name, int(row)) for row in rows)

    unique_rows = sorted(set((item["source_file"], item["jet_index"])
                             for item in occurrences))
    row_payload = {}
    clus_parts = {name: [] for name in (
        "fjet_clus_pt", "fjet_clus_eta", "fjet_clus_phi", "fjet_clus_E"
    )}
    stored_parts = {name: [] for name in FEATURE_ORDER}

    for source_file, row in unique_rows:
        raw_path = raw_dir / source_file
        with h5py.File(raw_path, "r") as handle:
            for name in clus_parts:
                clus_parts[name].append(
                    handle[name][row:row + 1, :max_constits].astype(np.float64)
                )
            for feature in FEATURE_ORDER:
                stored_parts[feature].append(
                    handle[STORED_BRANCH[feature]][row:row + 1].astype(np.float64)
                )
            event = (int(handle["EventInfo_mcEventNumber"][row])
                     if "EventInfo_mcEventNumber" in handle else None)
            label = int(handle["labels"][row])
        row_payload[(source_file, row)] = {
            "source_file": source_file,
            "jet_index": row,
            "event_identifier": event,
            "label": label,
        }

    if unique_rows:
        clus = {name: np.concatenate(values, axis=0)
                for name, values in clus_parts.items()}
        stored = {name: np.concatenate(values, axis=0)
                  for name, values in stored_parts.items()}
        reconstructed, extras = calculate_features(reco, clus)
        for i, key in enumerate(unique_rows):
            n = int(extras["n_constit"][i])
            diag = source_diagnostics(
                clus["fjet_clus_pt"][i], clus["fjet_clus_eta"][i],
                clus["fjet_clus_phi"][i], clus["fjet_clus_E"][i], n,
                max_constits,
            )
            row_payload[key].update({
                "stored_features": {name: float(stored[name][i])
                                    for name in FEATURE_ORDER},
                "reconstructed_features": {
                    name: (float(reconstructed[name][i])
                           if np.isfinite(reconstructed[name][i]) else None)
                    for name in FEATURE_ORDER
                },
                "source_diagnostics": diag,
                "is_known_l3_sentinel": bool(key in sentinel_rows),
                "active_constituents": {
                    "pt": clus["fjet_clus_pt"][i, :n].tolist(),
                    "eta": clus["fjet_clus_eta"][i, :n].tolist(),
                    "phi": clus["fjet_clus_phi"][i, :n].tolist(),
                    "E": clus["fjet_clus_E"][i, :n].tolist(),
                },
                "generalised_correlators": {
                    "1e3_b1": float(extras["gen3"][i, 0]),
                    "2e3_b1": float(extras["gen3"][i, 1]),
                    "3e3_b1": float(extras["gen3"][i, 2]),
                    "1e2_b2": float(extras["gen2_b2"][i]),
                },
            })

    for item in occurrences:
        key = (item["source_file"], item["jet_index"])
        item.update(row_payload[key])

    unique_nonfinite_rows = len(unique_rows)
    all_known_sentinel = bool(unique_rows) and all(
        key in sentinel_rows for key in unique_rows
    )
    return {
        "cache_directory": str(cache_dir),
        "cache_nonfinite_occurrence_count": len(occurrences),
        "unique_affected_jet_count": unique_nonfinite_rows,
        "full_cer_stored_l3_sentinel_count": sentinel_total,
        "all_nonfinite_rows_are_known_l3_sentinels": all_known_sentinel,
        "occurrences": occurrences,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Path containing data/, hl_recomputed/, paired/")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-per-dataset", type=int, default=500)
    parser.add_argument(
        "--files-per-dataset", type=int, default=None,
        help="Main shards to sample per dataset; use 0 for every shard",
    )
    parser.add_argument("--candidate-source-modified", action="store_true")
    parser.add_argument("--seed", type=int, default=80030)
    args = parser.parse_args()

    started = time.time()
    script_dir = Path(__file__).resolve().parent
    recompute_path = script_dir / "recompute_hl.py"
    data_root = args.data_root.resolve()
    hdf5_root = data_root / "data"
    if not hdf5_root.is_dir():
        raise SystemExit(f"missing data directory: {hdf5_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reco = import_reconstruction(recompute_path)
    # Runtime path routing only.  The scientific source file remains unchanged.
    reco.STORE = data_root
    reco.DATA = hdf5_root

    clus, stored, metadata, sampling = read_samples(
        hdf5_root, args.sample_per_dataset, args.seed, reco.MAX_CONSTITS,
        args.files_per_dataset,
    )
    reconstructed, extras = calculate_features(reco, clus)

    all_stats = []
    full = {}
    for feature in FEATURE_ORDER:
        stat = analyze_one_feature(
            feature, stored[feature], reconstructed[feature], metadata, clus,
            extras["n_constit"],
        )
        full[feature] = stat
        all_stats.append(("ALL", stat))

    by_dataset = {}
    datasets = np.asarray([item["dataset"] for item in metadata], dtype=object)
    for dataset in DATASETS:
        mask = datasets == dataset
        by_dataset[dataset] = {}
        for feature in FEATURE_ORDER:
            stat = analyze_one_feature(
                feature, stored[feature], reconstructed[feature], metadata,
                clus, extras["n_constit"], row_mask=mask,
            )
            by_dataset[dataset][feature] = stat
            all_stats.append((dataset, stat))

    cer = scan_cer_nonfinite(data_root, reco, reco.MAX_CONSTITS)
    strict_sample_pass = all(
        stat["n_nonfinite_reconstructed_with_valid_stored"] == 0
        and (stat["fraction_non_near_zero_above_hybrid_tolerance"] or 0.0) == 0.0
        for stat in full.values()
    )

    source_hash = hashlib.sha256(recompute_path.read_bytes()).hexdigest()
    report = {
        "audit_kind": "strict read-only eventwise closure audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "recompute_source": str(recompute_path),
        "recompute_sha256": source_hash,
        "recompute_code_fingerprint": reco.CODE_FP,
        "scientific_source_modified": args.candidate_source_modified,
        "production_npz_modified": False,
        "sample_seed": args.seed,
        "sample_per_dataset_target": args.sample_per_dataset,
        "files_per_dataset_target": args.files_per_dataset,
        "total_sampled_jets": len(metadata),
        "tolerance_policy": {
            "protected_relative_error": (
                "abs(reconstructed-stored) / max(abs(reconstructed), "
                "abs(stored), 1e-6 * robust median feature magnitude)"
            ),
            "relative_tolerance": REL_TOL,
            "absolute_tolerance": (
                "max(8 * float32_epsilon * robust median feature magnitude, "
                "1e-12)"
            ),
            "above_hybrid_tolerance": (
                "absolute error > absolute tolerance AND protected relative "
                "error > relative tolerance"
            ),
            "near_zero": (
                "max(abs(reconstructed), abs(stored)) <= protected relative "
                "denominator floor"
            ),
        },
        "sampling": sampling,
        "strict_sample_pass": strict_sample_pass,
        "closure_all_samples": full,
        "closure_by_dataset": by_dataset,
        "cer_nonfinite": cer,
        "elapsed_seconds": time.time() - started,
    }

    dump_json(args.output_dir / "strict_closure_audit.json", report)
    dump_json(args.output_dir / "cer_nonfinite_details.json", cer)
    write_stats_csv(args.output_dir / "strict_closure_stats.csv", all_stats)
    write_eventwise_csv(
        args.output_dir / "strict_closure_eventwise.csv", metadata, clus,
        extras["n_constit"], stored, reconstructed, full,
    )

    print(f"sampled jets: {len(metadata)}")
    for dataset, info in sampling.items():
        print(f"  {dataset:15s} {info['sampled_rows']:5d} jets  "
              f"files {[x['file'] for x in info['files']]}")
    print(f"strict sample pass: {strict_sample_pass}")
    print(f"CER non-finite occurrences: {cer['cache_nonfinite_occurrence_count']} "
          f"across {cer['unique_affected_jet_count']} jets")
    print(f"CER stored L3 sentinels: {cer['full_cer_stored_l3_sentinel_count']}")
    print(f"output: {args.output_dir}")


if __name__ == "__main__":
    main()
