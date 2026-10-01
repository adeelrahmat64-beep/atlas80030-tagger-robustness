"""Class-conditional split-conformal coverage for the frozen ATLAS80030 tagger.

The calibration sample is exactly the validation subset stored with the frozen
5M final-production model.  Class-specific thresholds are calibrated once on
that nominal sample, then applied without pairing or refitting to the nominal
test sample and the nine detector variations.

Use ``--preflight`` for read-only provenance, schema, split, and count checks.
The normal mode performs the campaign and creates a new, non-overwriting
``analysis_results/conformal_coverage_round1`` output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parent
DATA_ROOT = Path(
    os.environ.get(
        "ATLAS_80030_DATA_ROOT", str(WORKSPACE_ROOT / "ATLAS_80030_Data")
    )
).expanduser().resolve()

MODEL_STEM = "atlas_tagger_xgb_final_production_round1"
MODEL_PATH = DATA_ROOT / "models" / f"{MODEL_STEM}.joblib"
METADATA_PATH = DATA_ROOT / "models" / f"{MODEL_STEM}_metadata.json"
SPLIT_PATH = DATA_ROOT / "models" / f"{MODEL_STEM}_split.npz"
PREDICTION_MANIFEST_PATH = (
    DATA_ROOT / "predictions" / f"{MODEL_STEM}_prediction_manifest.json"
)
CACHE_ROOT = DATA_ROOT / "hl_recomputed"
from atlas_config import RESULTS_ROOT  # noqa: E402
OUTPUT_DIR = RESULTS_ROOT / "conformal_coverage_round1"

MODEL_SHA256 = "63a23308efbf2ce90418c8e657730f571db4798fc9ca2fb3692cb986de405f4c"
SPLIT_SHA256 = "4f500536d3d80cc4b12e52624f822253a9f1092db36c440b9097c819a9834174"
SOURCE_SHA256 = "7ce74841a213b37e8e5d1298b2fc16739be4c4895aa85d0de715acff112b8046"
FROZEN_RECONSTRUCTION_FP = "c28a262540824e9db90f34c421f10c70"

ALPHA = 0.10
TARGET_COVERAGE = 1.0 - ALPHA
SEED = 42
EXPECTED_LOADED_ROWS = 5_000_000
EXPECTED_ELIGIBLE_ROWS = 4_999_997
EXPECTED_TRAINING_ROWS = 4_499_612
EXPECTED_CALIBRATION_ROWS = 500_385
# The frozen metadata stores class counts in np.bincount order: [0, 1].
EXPECTED_CALIBRATION_CLASS_COUNTS = {0: 248_170, 1: 252_215}
EXPECTED_TRAINING_CLASS_COUNTS = {0: 2_227_557, 1: 2_272_055}

FEATURE_ORDER = [
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

DATASETS = (
    "test_nominal",
    "esup",
    "esdown",
    "cer",
    "cpos",
    "teg",
    "tej",
    "tfl",
    "tfj",
    "bias",
)


def fail(message: str) -> None:
    raise SystemExit(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"cannot read JSON provenance {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"JSON provenance is not an object: {path}")
    return value


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        fail(f"missing {description}: {path}")


def require_equal(actual: Any, expected: Any, description: str) -> None:
    if actual != expected:
        fail(f"{description} mismatch: got {actual!r}, expected {expected!r}")


def scalar_cache_fingerprint(data: np.lib.npyio.NpzFile, path: Path) -> str:
    if "code_fp" not in data.files:
        fail(f"cache has no reconstruction fingerprint: {path}")
    stamp = np.asarray(data["code_fp"])
    if stamp.size != 1:
        fail(f"cache fingerprint is not scalar: {path} shape={stamp.shape}")
    return str(stamp.reshape(()).item())


def verify_training_cache_and_counts(
    file_stems: list[str],
    file_rows: list[int],
    eligibility: np.ndarray,
    validation: np.ndarray,
) -> tuple[dict[int, int], dict[int, int]]:
    """Verify cache identities and recover exact frozen split class counts.

    This intentionally reads only labels and scalar cache fingerprints.  It
    does not read the 15 feature arrays or make predictions, so it remains a
    lightweight preflight.
    """
    cache_dir = CACHE_ROOT / "train_nominal"
    if not cache_dir.is_dir():
        fail(f"missing train_nominal cache directory: {cache_dir}")

    expected_names = [f"{stem}.npz" for stem in file_stems]
    actual_names = sorted(path.name for path in cache_dir.glob("*.npz"))
    require_equal(actual_names, expected_names, "train_nominal cache file identity")

    calibration_counts = np.zeros(2, dtype=np.int64)
    training_counts = np.zeros(2, dtype=np.int64)
    raw_offset = 0
    eligible_offset = 0
    required_fields = set(FEATURE_ORDER) | {"labels", "n_constit", "code_fp"}

    for stem, expected_rows in zip(file_stems, file_rows, strict=True):
        path = cache_dir / f"{stem}.npz"
        with np.load(path, allow_pickle=False) as data:
            missing = sorted(required_fields - set(data.files))
            if missing:
                fail(f"cache schema is incomplete in {path}: missing {missing}")
            require_equal(
                scalar_cache_fingerprint(data, path),
                FROZEN_RECONSTRUCTION_FP,
                f"reconstruction fingerprint for {path.name}",
            )
            labels = np.asarray(data["labels"])

        if labels.shape != (expected_rows,):
            fail(
                f"label shape mismatch in {path}: got {labels.shape}, "
                f"expected {(expected_rows,)}"
            )
        if not np.isin(labels, (0, 1)).all():
            fail(f"non-binary labels in frozen training cache: {path}")

        raw_eligible = eligibility[raw_offset : raw_offset + expected_rows]
        n_eligible = int(raw_eligible.sum())
        file_validation = validation[
            eligible_offset : eligible_offset + n_eligible
        ]
        eligible_labels = labels[raw_eligible].astype(np.int8, copy=False)
        calibration_labels = eligible_labels[file_validation]
        training_labels = eligible_labels[~file_validation]
        calibration_counts += np.bincount(calibration_labels, minlength=2)
        training_counts += np.bincount(training_labels, minlength=2)

        raw_offset += expected_rows
        eligible_offset += n_eligible

    require_equal(raw_offset, eligibility.size, "raw split extent")
    require_equal(eligible_offset, validation.size, "eligible split extent")
    return (
        {0: int(calibration_counts[0]), 1: int(calibration_counts[1])},
        {0: int(training_counts[0]), 1: int(training_counts[1])},
    )


def verify_prediction_manifest(
    manifest: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, int]:
    expected = {
        "run_role": "final_production",
        "final_model_path": str(MODEL_PATH),
        "model_sha256": MODEL_SHA256,
        "tagger_source_sha256": SOURCE_SHA256,
        "reconstruction_fingerprint": FROZEN_RECONSTRUCTION_FP,
        "training_rows_requested": EXPECTED_LOADED_ROWS,
        "seed": SEED,
    }
    for key, value in expected.items():
        require_equal(manifest.get(key), value, f"prediction manifest {key}")
    require_equal(
        manifest.get("model_sha256"),
        metadata.get("model_sha256"),
        "model SHA agreement between metadata and prediction manifest",
    )

    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict):
        fail("prediction manifest has no datasets object")
    rows: dict[str, int] = {}
    for dataset in DATASETS:
        entry = datasets.get(dataset)
        if not isinstance(entry, dict):
            fail(f"prediction manifest has no entry for {dataset}")
        n_rows = entry.get("prediction_rows")
        if not isinstance(n_rows, int) or n_rows <= 0:
            fail(f"invalid prediction row count for {dataset}: {n_rows!r}")
        path = DATA_ROOT / "predictions" / f"public_{dataset}.npz"
        require_file(path, f"frozen public predictions for {dataset}")
        with np.load(path, allow_pickle=False) as data:
            missing = sorted({"predictions", "labels", "pt"} - set(data.files))
        if missing:
            fail(f"prediction schema is incomplete in {path}: missing {missing}")
        rows[dataset] = n_rows
    return rows


def preflight() -> dict[str, Any]:
    """Perform read-only checks of every frozen input contract."""
    required = (
        (MODEL_PATH, "frozen final XGBoost model"),
        (METADATA_PATH, "frozen final model metadata"),
        (SPLIT_PATH, "frozen final split"),
        (PREDICTION_MANIFEST_PATH, "frozen prediction manifest"),
    )
    for path, description in required:
        require_file(path, description)

    if OUTPUT_DIR.exists():
        fail(
            "output directory already exists; refusing any possible overwrite: "
            f"{OUTPUT_DIR}"
        )

    metadata = load_json(METADATA_PATH)
    expected_metadata = {
        "model_type": "xgb",
        "run_role": "final_production",
        "training_rows_requested": EXPECTED_LOADED_ROWS,
        "reconstruction_fingerprint": FROZEN_RECONSTRUCTION_FP,
        "training_dataset": "train_nominal",
        "feature_preprocessing": "none",
        "event_group_field": "EventInfo_mcEventNumber",
        "split_strategy": "SplitMix64(event number, seed) < validation cutoff",
        "seed": SEED,
        "validation_fraction": 0.1,
        "loaded_rows": EXPECTED_LOADED_ROWS,
        "eligible_rows": EXPECTED_ELIGIBLE_ROWS,
        "training_rows": EXPECTED_TRAINING_ROWS,
        "validation_rows": EXPECTED_CALIBRATION_ROWS,
        "model_sha256": MODEL_SHA256,
        "split_sha256": SPLIT_SHA256,
        "tagger_source_sha256": SOURCE_SHA256,
        "source_snapshot_sha256": SOURCE_SHA256,
    }
    for key, value in expected_metadata.items():
        require_equal(metadata.get(key), value, f"model metadata {key}")
    require_equal(metadata.get("feature_order"), FEATURE_ORDER, "frozen feature order")
    require_equal(
        metadata.get("validation_class_counts"),
        [EXPECTED_CALIBRATION_CLASS_COUNTS[0], EXPECTED_CALIBRATION_CLASS_COUNTS[1]],
        "metadata calibration class counts",
    )
    require_equal(
        metadata.get("training_class_counts"),
        [EXPECTED_TRAINING_CLASS_COUNTS[0], EXPECTED_TRAINING_CLASS_COUNTS[1]],
        "metadata training class counts",
    )
    require_equal(
        EXPECTED_TRAINING_ROWS + EXPECTED_CALIBRATION_ROWS,
        EXPECTED_ELIGIBLE_ROWS,
        "frozen split accounting",
    )

    require_equal(sha256(MODEL_PATH), MODEL_SHA256, "frozen model SHA-256")
    require_equal(sha256(SPLIT_PATH), SPLIT_SHA256, "frozen split SHA-256")

    source_snapshot_raw = metadata.get("source_snapshot_path")
    if not isinstance(source_snapshot_raw, str):
        fail("model metadata has no source_snapshot_path")
    source_snapshot = Path(source_snapshot_raw).resolve()
    require_file(source_snapshot, "frozen tagger source snapshot")
    require_equal(
        sha256(source_snapshot), SOURCE_SHA256, "frozen tagger source snapshot SHA-256"
    )

    with np.load(SPLIT_PATH, allow_pickle=False) as split:
        required_split = {
            "eligibility_mask",
            "validation_mask_within_eligible",
            "file_stems",
            "file_rows",
        }
        missing = sorted(required_split - set(split.files))
        if missing:
            fail(f"frozen split is missing fields: {missing}")
        eligibility_raw = np.asarray(split["eligibility_mask"])
        validation_raw = np.asarray(split["validation_mask_within_eligible"])
        file_stems = [str(value) for value in np.asarray(split["file_stems"]).tolist()]
        file_rows = [int(value) for value in np.asarray(split["file_rows"]).tolist()]

    if eligibility_raw.dtype != np.bool_ or validation_raw.dtype != np.bool_:
        fail("frozen split masks are not boolean")
    eligibility = eligibility_raw
    validation = validation_raw
    if eligibility.shape != (EXPECTED_LOADED_ROWS,):
        fail(f"unexpected eligibility-mask shape: {eligibility.shape}")
    if int(eligibility.sum()) != EXPECTED_ELIGIBLE_ROWS:
        fail(f"unexpected eligible count: {int(eligibility.sum())}")
    if validation.shape != (EXPECTED_ELIGIBLE_ROWS,):
        fail(f"unexpected validation-mask shape: {validation.shape}")
    if int(validation.sum()) != EXPECTED_CALIBRATION_ROWS:
        fail(f"unexpected calibration count: {int(validation.sum())}")
    if len(file_stems) != len(file_rows) or sum(file_rows) != EXPECTED_LOADED_ROWS:
        fail("frozen file provenance has invalid lengths or row accounting")
    require_equal(file_stems, metadata.get("training_file_stems"), "split file stems")
    require_equal(file_rows, metadata.get("training_file_rows"), "split file rows")

    calibration_counts, training_counts = verify_training_cache_and_counts(
        file_stems, file_rows, eligibility, validation
    )
    require_equal(
        calibration_counts,
        EXPECTED_CALIBRATION_CLASS_COUNTS,
        "independently recovered calibration class counts",
    )
    require_equal(
        training_counts,
        EXPECTED_TRAINING_CLASS_COUNTS,
        "independently recovered training class counts",
    )

    model = joblib.load(MODEL_PATH)
    require_equal(
        int(getattr(model, "n_features_in_", -1)),
        len(FEATURE_ORDER),
        "loaded model feature count",
    )
    classes = np.asarray(getattr(model, "classes_", []))
    if classes.shape != (2,) or not np.array_equal(classes, np.array([0, 1])):
        fail(f"loaded model has unexpected class order: {classes!r}")

    manifest = load_json(PREDICTION_MANIFEST_PATH)
    prediction_rows = verify_prediction_manifest(manifest, metadata)

    return {
        "metadata": metadata,
        "manifest": manifest,
        "model": model,
        "eligibility": eligibility,
        "validation": validation,
        "file_stems": file_stems,
        "file_rows": file_rows,
        "calibration_counts": calibration_counts,
        "training_counts": training_counts,
        "prediction_rows": prediction_rows,
        "source_snapshot": source_snapshot,
    }


def load_calibration_features(contract: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Load only the rows selected by the immutable validation split."""
    eligibility = contract["eligibility"]
    validation = contract["validation"]
    file_stems = contract["file_stems"]
    file_rows = contract["file_rows"]
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    raw_offset = 0
    eligible_offset = 0

    for stem, expected_rows in zip(file_stems, file_rows, strict=True):
        path = CACHE_ROOT / "train_nominal" / f"{stem}.npz"
        raw_eligible = eligibility[raw_offset : raw_offset + expected_rows]
        eligible_indices = np.flatnonzero(raw_eligible)
        n_eligible = int(eligible_indices.size)
        file_validation = validation[
            eligible_offset : eligible_offset + n_eligible
        ]
        calibration_indices = eligible_indices[file_validation]

        if calibration_indices.size:
            with np.load(path, allow_pickle=False) as data:
                # float64 matches the frozen trainer's feature-loading convention.
                features = np.column_stack(
                    [np.asarray(data[name])[calibration_indices] for name in FEATURE_ORDER]
                ).astype(np.float64, copy=False)
                labels = np.asarray(data["labels"])[calibration_indices].astype(
                    np.int8, copy=False
                )
                n_constit = np.asarray(data["n_constit"])[calibration_indices]
            if not np.isfinite(features).all():
                fail(f"nonfinite calibration features selected from {path}")
            if (features <= -9.0e5).any():
                fail(f"sentinel calibration feature selected from {path}")
            if (n_constit < 3).any():
                fail(f"ineligible calibration jet selected from {path}")
            if not np.isin(labels, (0, 1)).all():
                fail(f"non-binary calibration label selected from {path}")
            feature_chunks.append(features)
            label_chunks.append(labels)

        raw_offset += expected_rows
        eligible_offset += n_eligible

    if raw_offset != EXPECTED_LOADED_ROWS or eligible_offset != EXPECTED_ELIGIBLE_ROWS:
        fail("calibration feature loading did not consume the frozen split exactly")
    features = np.concatenate(feature_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    if features.shape != (EXPECTED_CALIBRATION_ROWS, len(FEATURE_ORDER)):
        fail(f"unexpected calibration feature shape: {features.shape}")
    if labels.shape != (EXPECTED_CALIBRATION_ROWS,):
        fail(f"unexpected calibration label shape: {labels.shape}")
    recovered = np.bincount(labels, minlength=2)
    expected = np.array(
        [EXPECTED_CALIBRATION_CLASS_COUNTS[0], EXPECTED_CALIBRATION_CLASS_COUNTS[1]]
    )
    if not np.array_equal(recovered, expected):
        fail(f"calibration feature/label class counts changed: {recovered.tolist()}")
    return features, labels


def checked_probabilities(model: Any, features: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features))
    if probabilities.shape != (features.shape[0], 2):
        fail(f"model returned unexpected probability shape: {probabilities.shape}")
    if not np.isfinite(probabilities).all():
        fail("model returned nonfinite calibration probabilities")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        fail("model returned calibration probabilities outside [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=2e-6):
        fail("model calibration probability rows do not sum to one")
    return probabilities[:, 1].astype(np.float64, copy=False)


def finite_sample_thresholds(
    labels: np.ndarray, probability_signal: np.ndarray
) -> dict[int, dict[str, Any]]:
    """Return class-conditional split-conformal thresholds.

    For class c with n calibration examples, the order-statistic rank is
    ceil((n + 1) * (1 - alpha)), capped at n.  Scores are 1 - p_true.
    """
    thresholds: dict[int, dict[str, Any]] = {}
    for class_id, class_name in ((0, "background"), (1, "signal")):
        selected = labels == class_id
        n_class = int(selected.sum())
        if n_class <= 0:
            fail(f"no {class_name} calibration examples")
        p_true = (
            probability_signal[selected]
            if class_id == 1
            else 1.0 - probability_signal[selected]
        )
        scores = 1.0 - p_true
        if not np.isfinite(scores).all():
            fail(f"nonfinite {class_name} calibration nonconformity scores")
        rank = min(n_class, int(math.ceil((n_class + 1) * TARGET_COVERAGE)))
        threshold = float(np.partition(scores, rank - 1)[rank - 1])
        calibration_coverage = float(np.mean(scores <= threshold))
        thresholds[class_id] = {
            "class_id": class_id,
            "class_name": class_name,
            "count": n_class,
            "finite_sample_rank_one_based": rank,
            "order_statistic_fraction": rank / n_class,
            "nonconformity_threshold": threshold,
            "calibration_coverage_at_threshold": calibration_coverage,
        }
    return thresholds


def load_public_predictions(
    dataset: str, expected_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    path = DATA_ROOT / "predictions" / f"public_{dataset}.npz"
    with np.load(path, allow_pickle=False) as data:
        probability_signal = np.asarray(data["predictions"])
        labels = np.asarray(data["labels"])
        pt = np.asarray(data["pt"])
    shapes = {probability_signal.shape, labels.shape, pt.shape}
    if len(shapes) != 1 or probability_signal.ndim != 1:
        fail(f"unaligned prediction vectors in {path}: {sorted(map(str, shapes))}")
    if probability_signal.size != expected_rows:
        fail(
            f"prediction row count mismatch for {dataset}: "
            f"{probability_signal.size} != {expected_rows}"
        )
    if not np.isfinite(probability_signal).all():
        fail(f"nonfinite probabilities in {path}")
    if ((probability_signal < 0.0) | (probability_signal > 1.0)).any():
        fail(f"probabilities outside [0, 1] in {path}")
    if not np.isfinite(pt).all():
        fail(f"nonfinite pT values in {path}")
    if not np.isin(labels, (0, 1)).all():
        fail(f"non-binary labels in {path}")
    return probability_signal, labels.astype(np.int8, copy=False)


def dataset_metrics(
    dataset: str,
    probability_signal: np.ndarray,
    labels: np.ndarray,
    thresholds: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    q_background = thresholds[0]["nonconformity_threshold"]
    q_signal = thresholds[1]["nonconformity_threshold"]

    # Candidate label c enters the set when 1 - p_c <= q_c.
    include_background = probability_signal <= q_background
    include_signal = (1.0 - probability_signal) <= q_signal
    set_size = include_background.astype(np.int8) + include_signal.astype(np.int8)
    signal_mask = labels == 1
    background_mask = labels == 0
    covered = np.where(signal_mask, include_signal, include_background)

    n_total = int(labels.size)
    n_signal = int(signal_mask.sum())
    n_background = int(background_mask.sum())
    n_empty = int(np.sum(set_size == 0))
    n_singleton = int(np.sum(set_size == 1))
    n_ambiguous = int(np.sum(set_size == 2))
    if n_total <= 0 or n_signal <= 0 or n_background <= 0:
        fail(f"{dataset}: zero total or class count")
    if n_signal + n_background != n_total:
        fail(f"{dataset}: class accounting failure")
    if n_empty + n_singleton + n_ambiguous != n_total:
        fail(f"{dataset}: prediction-set accounting failure")
    if ((set_size < 0) | (set_size > 2)).any():
        fail(f"{dataset}: invalid prediction-set size")

    result: dict[str, Any] = {
        "dataset": dataset,
        "total_count": n_total,
        "signal_count": n_signal,
        "background_count": n_background,
        "overall_coverage": float(np.mean(covered)),
        "signal_coverage": float(np.mean(covered[signal_mask])),
        "background_coverage": float(np.mean(covered[background_mask])),
        "overall_deviation_from_target": float(np.mean(covered) - TARGET_COVERAGE),
        "signal_deviation_from_target": float(
            np.mean(covered[signal_mask]) - TARGET_COVERAGE
        ),
        "background_deviation_from_target": float(
            np.mean(covered[background_mask]) - TARGET_COVERAGE
        ),
        "mean_prediction_set_size": float(np.mean(set_size)),
        "singleton_rate": n_singleton / n_total,
        "ambiguous_0_1_rate": n_ambiguous / n_total,
        "empty_set_rate": n_empty / n_total,
        "singleton_count": n_singleton,
        "ambiguous_0_1_count": n_ambiguous,
        "empty_set_count": n_empty,
    }
    rates_sum = (
        result["singleton_rate"]
        + result["ambiguous_0_1_rate"]
        + result["empty_set_rate"]
    )
    if not math.isclose(rates_sum, 1.0, rel_tol=0.0, abs_tol=2e-15):
        fail(f"{dataset}: prediction-set rates do not sum to one")
    expected_mean_size = (
        result["singleton_rate"] + 2.0 * result["ambiguous_0_1_rate"]
    )
    if not math.isclose(
        result["mean_prediction_set_size"],
        expected_mean_size,
        rel_tol=0.0,
        abs_tol=2e-15,
    ):
        fail(f"{dataset}: mean prediction-set size fails accounting check")
    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            fail(f"{dataset}: nonfinite result in {key}")
    return result


def add_nominal_deviations(results: list[dict[str, Any]]) -> None:
    nominal = results[0]
    if nominal["dataset"] != "test_nominal":
        fail("internal error: nominal dataset is not first")
    for result in results:
        result["overall_deviation_from_nominal"] = (
            result["overall_coverage"] - nominal["overall_coverage"]
        )
        result["signal_deviation_from_nominal"] = (
            result["signal_coverage"] - nominal["signal_coverage"]
        )
        result["background_deviation_from_nominal"] = (
            result["background_coverage"] - nominal["background_coverage"]
        )


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        fail(f"refusing to overwrite output: {path}")
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_text_new(path: Path, value: str) -> None:
    if path.exists():
        fail(f"refusing to overwrite output: {path}")
    path.write_text(value, encoding="utf-8")


def write_outputs(
    contract: dict[str, Any],
    thresholds: dict[int, dict[str, Any]],
    results: list[dict[str, Any]],
) -> None:
    if OUTPUT_DIR.exists():
        fail(f"refusing to overwrite existing output directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    created = datetime.now(timezone.utc).isoformat()
    threshold_document = {
        "method": "class-conditional split conformal prediction",
        "alpha": ALPHA,
        "target_coverage": TARGET_COVERAGE,
        "nonconformity_score": "1 - p_true",
        "finite_sample_rule": "ceil((n_class + 1) * (1 - alpha))-th order statistic, capped at n_class",
        "calibration_dataset": "train_nominal",
        "calibration_subset": "frozen held-out validation split",
        "calibrated_once_on_nominal": True,
        "total_calibration_count": EXPECTED_CALIBRATION_ROWS,
        "background": thresholds[0],
        "signal": thresholds[1],
    }
    provenance = {
        "created_utc": created,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "data_root": str(DATA_ROOT),
        "model_path": str(MODEL_PATH),
        "model_sha256": sha256(MODEL_PATH),
        "metadata_path": str(METADATA_PATH),
        "metadata_sha256": sha256(METADATA_PATH),
        "split_path": str(SPLIT_PATH),
        "split_sha256": sha256(SPLIT_PATH),
        "tagger_source_snapshot_path": str(contract["source_snapshot"]),
        "tagger_source_snapshot_sha256": sha256(contract["source_snapshot"]),
        "prediction_manifest_path": str(PREDICTION_MANIFEST_PATH),
        "prediction_manifest_sha256": sha256(PREDICTION_MANIFEST_PATH),
        "reconstruction_fingerprint": FROZEN_RECONSTRUCTION_FP,
        "feature_order": FEATURE_ORDER,
        "feature_preprocessing": "none",
        "training_or_refitting_performed": False,
        "imputation_performed": False,
        "pairing_used": False,
        "coverage_weighting": "unweighted jets",
        "calibration_split": {
            "strategy": contract["metadata"]["split_strategy"],
            "event_group_field": contract["metadata"]["event_group_field"],
            "seed": SEED,
            "total_count": EXPECTED_CALIBRATION_ROWS,
            "background_count": EXPECTED_CALIBRATION_CLASS_COUNTS[0],
            "signal_count": EXPECTED_CALIBRATION_CLASS_COUNTS[1],
        },
        "evaluated_datasets": list(DATASETS),
        "prediction_rows_from_frozen_manifest": contract["prediction_rows"],
    }
    combined = {
        "provenance": provenance,
        "calibration": threshold_document,
        "datasets": results,
    }

    write_json_new(OUTPUT_DIR / "conformal_coverage_results.json", combined)
    write_json_new(OUTPUT_DIR / "calibration_thresholds.json", threshold_document)
    write_json_new(OUTPUT_DIR / "provenance.json", provenance)

    csv_path = OUTPUT_DIR / "conformal_coverage_results.csv"
    if csv_path.exists():
        fail(f"refusing to overwrite output: {csv_path}")
    fieldnames = list(results[0].keys())
    with csv_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    lines = [
        "ATLAS80030 class-conditional split-conformal coverage",
        f"Created (UTC): {created}",
        f"Target coverage: {TARGET_COVERAGE:.6f} (alpha={ALPHA:.6f})",
        "Nonconformity score: 1 - p_true",
        "Thresholds calibrated once on the frozen nominal validation split.",
        (
            "Calibration counts: total="
            f"{EXPECTED_CALIBRATION_ROWS}, signal={EXPECTED_CALIBRATION_CLASS_COUNTS[1]}, "
            f"background={EXPECTED_CALIBRATION_CLASS_COUNTS[0]}"
        ),
        (
            "Background threshold: "
            f"{thresholds[0]['nonconformity_threshold']:.17g} "
            f"(rank {thresholds[0]['finite_sample_rank_one_based']})"
        ),
        (
            "Signal threshold: "
            f"{thresholds[1]['nonconformity_threshold']:.17g} "
            f"(rank {thresholds[1]['finite_sample_rank_one_based']})"
        ),
        "",
        (
            "dataset total signal background overall_cov signal_cov background_cov "
            "d_target d_nominal mean_size singleton ambiguous empty"
        ),
    ]
    for result in results:
        lines.append(
            f"{result['dataset']} {result['total_count']} {result['signal_count']} "
            f"{result['background_count']} {result['overall_coverage']:.10f} "
            f"{result['signal_coverage']:.10f} {result['background_coverage']:.10f} "
            f"{result['overall_deviation_from_target']:+.10f} "
            f"{result['overall_deviation_from_nominal']:+.10f} "
            f"{result['mean_prediction_set_size']:.10f} "
            f"{result['singleton_rate']:.10f} "
            f"{result['ambiguous_0_1_rate']:.10f} "
            f"{result['empty_set_rate']:.10f}"
        )
    write_text_new(OUTPUT_DIR / "conformal_coverage_summary.txt", "\n".join(lines) + "\n")


def run_campaign(contract: dict[str, Any]) -> None:
    calibration_features, calibration_labels = load_calibration_features(contract)
    calibration_probability_signal = checked_probabilities(
        contract["model"], calibration_features
    )
    thresholds = finite_sample_thresholds(
        calibration_labels, calibration_probability_signal
    )

    # Thresholds are now frozen in memory and are never recalibrated below.
    results: list[dict[str, Any]] = []
    for dataset in DATASETS:
        probability_signal, labels = load_public_predictions(
            dataset, contract["prediction_rows"][dataset]
        )
        results.append(
            dataset_metrics(dataset, probability_signal, labels, thresholds)
        )
    add_nominal_deviations(results)
    write_outputs(contract, thresholds, results)
    print(f"Wrote conformal coverage campaign to: {OUTPUT_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="verify frozen provenance, schemas, split, and exact counts; write nothing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = preflight()
    if args.preflight:
        print("PREFLIGHT PASSED")
        print(f"model_sha256={MODEL_SHA256}")
        print(f"split_sha256={SPLIT_SHA256}")
        print(f"calibration_total={EXPECTED_CALIBRATION_ROWS}")
        print(f"calibration_signal={contract['calibration_counts'][1]}")
        print(f"calibration_background={contract['calibration_counts'][0]}")
        print("No predictions were computed and no output files were written.")
        return
    run_campaign(contract)


if __name__ == "__main__":
    main()
