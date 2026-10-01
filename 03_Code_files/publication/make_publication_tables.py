"""Build publication tables from frozen final ATLAS80030 artifacts only.

The script performs no reconstruction, training, inference, metric evaluation,
SHAP calculation, pairing, or conformal calibration.  It validates the saved
final result artifacts, assembles publication-ready CSV and LaTeX tables, and
refuses to overwrite its output directory.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
PROJECT_ROOT = CODE_ROOT.parent


def _resolve_results_root(code_root, project_root):
    """Campaign outputs: the top-level results folder, or the historical
    location beside the code, whichever is actually present."""
    import os as _os
    env = _os.environ.get("ATLAS_80030_RESULTS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    names = ("04_Results", "03_Results", "Results", "results")
    cands = [project_root / n for n in names] + [code_root / "analysis_results"]
    for c in cands:
        if c.is_dir() and any(c.iterdir()):
            return c
    for c in cands:
        if c.is_dir():
            return c
    return code_root / "analysis_results"
DATA_ROOT = PROJECT_ROOT / "ATLAS_80030_Data"
RESULTS_ROOT = _resolve_results_root(CODE_ROOT, PROJECT_ROOT)

METRICS_JSON = (
    DATA_ROOT
    / "predictions"
    / "final_metrics_atlas_tagger_xgb_final_production_round1_round1.json"
)
TAGGER_SOURCE_SNAPSHOT = (
    RESULTS_ROOT / "ATLAS80030_tagger_xgb_final_production_round1_code.py"
)
SHAP_DIR = RESULTS_ROOT / "shap_delta_final_production_round3"
SHAP_STATUS_JSON = SHAP_DIR / "campaign_status.json"
SHAP_RESULTS_JSON = SHAP_DIR / "shap_delta_results.json"
SHAP_PER_FEATURE_CSV = SHAP_DIR / "shap_delta_per_feature.csv"
CONFORMAL_DIR = RESULTS_ROOT / "conformal_coverage_round1"
CONFORMAL_PROVENANCE_JSON = CONFORMAL_DIR / "provenance.json"
CONFORMAL_CALIBRATION_JSON = CONFORMAL_DIR / "calibration_thresholds.json"
CONFORMAL_RESULTS_JSON = CONFORMAL_DIR / "conformal_coverage_results.json"
CONFORMAL_RESULTS_CSV = CONFORMAL_DIR / "conformal_coverage_results.csv"
OUTPUT_DIR = RESULTS_ROOT / "publication_tables_round2"

MODEL_SHA256 = "63a23308efbf2ce90418c8e657730f571db4798fc9ca2fb3692cb986de405f4c"
TAGGER_SOURCE_SHA256 = "7ce74841a213b37e8e5d1298b2fc16739be4c4895aa85d0de715acff112b8046"
OFFICIAL_REFERENCE_COMMIT = "5bc3c8d70cf8736d24b969972a3dd365017a9e24"
PUBLISHED_HLDNN_ACCURACY_UNCERTAINTY = 0.0002
EXPECTED_PT_EDGES_MEV = tuple(float(value) for value in range(350_000, 3_150_001, 200_000))

FEATURES = (
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
)
SHAP_SYSTEMATICS = ("esup", "esdown", "cpos", "cer")
SHAP_SAMPLE_SIZES = (25_000, 50_000, 100_000)
SUBSETS = ("all", "signal", "background")

BASE_METRIC_DATASETS = (
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
    "ttbar_pythia",
    "ttbar_herwig",
    "cluster",
    "string",
    "angular",
    "dipole",
)
SCALE_METRIC_DATASETS = (
    "sig_ISRx2",
    "sig_ISRxp5",
    "sig_FSRx2",
    "sig_FSRxp5",
    "bkg_ISRx2",
    "bkg_ISRxp5",
    "bkg_FSRx2",
    "bkg_FSRxp5",
)
CONFORMAL_DATASETS = (
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

SHAP_FIELDS = (
    "subset",
    "feature",
    "n_pairs",
    "mean_signed_delta_shap",
    "median_signed_delta_shap",
    "mean_abs_delta_shap",
    "median_abs_delta_shap",
    "nominal_mean_abs_shap",
    "systematic_mean_abs_shap",
    "relative_attribution_drift",
    "relative_attribution_drift_defined",
    "nominal_importance_rank",
    "systematic_importance_rank",
    "global_importance_spearman_rho",
    "global_importance_spearman_pvalue",
    "systematic",
    "sample_size",
    "mean_signed_delta_score",
    "median_signed_delta_score",
    "mean_abs_delta_score",
)

CONFORMAL_FIELDS = (
    "dataset",
    "total_count",
    "signal_count",
    "background_count",
    "overall_coverage",
    "signal_coverage",
    "background_coverage",
    "overall_deviation_from_target",
    "signal_deviation_from_target",
    "background_deviation_from_target",
    "mean_prediction_set_size",
    "singleton_rate",
    "ambiguous_0_1_rate",
    "empty_set_rate",
    "singleton_count",
    "ambiguous_0_1_count",
    "empty_set_count",
    "overall_deviation_from_nominal",
    "signal_deviation_from_nominal",
    "background_deviation_from_nominal",
)

TABLE_OUTPUTS = {
    "table1_nominal_performance": (
        "table1_nominal_performance.csv",
        "table1_nominal_performance.tex",
    ),
    "table2_inclusive_systematics_wp50": (
        "table2_inclusive_systematics_wp50.csv",
        "table2_inclusive_systematics_wp50.tex",
    ),
    "table3_shap_delta_100k_summary": (
        "table3_shap_delta_100k_summary.csv",
        "table3_shap_delta_100k_summary.tex",
    ),
    "supplementary_systematics_14bin": (
        "supplementary_systematics_14bin.csv",
        "supplementary_systematics_14bin.tex",
    ),
    "supplementary_shap_delta_round3_all_features": (
        "supplementary_shap_delta_round3_all_features.csv",
        "supplementary_shap_delta_round3_all_features.tex",
    ),
    "supplementary_conformal_round1": (
        "supplementary_conformal_round1.csv",
        "supplementary_conformal_round1.tex",
    ),
}

SOURCE_PATHS = {
    "frozen_final_metrics": METRICS_JSON,
    "frozen_final_tagger_source_snapshot_for_published_hldnn_reference": TAGGER_SOURCE_SNAPSHOT,
    "frozen_shap_round3_status": SHAP_STATUS_JSON,
    "frozen_shap_round3_results": SHAP_RESULTS_JSON,
    "frozen_shap_round3_per_feature": SHAP_PER_FEATURE_CSV,
    "frozen_conformal_round1_provenance": CONFORMAL_PROVENANCE_JSON,
    "frozen_conformal_round1_calibration": CONFORMAL_CALIBRATION_JSON,
    "frozen_conformal_round1_results": CONFORMAL_RESULTS_JSON,
    "frozen_conformal_round1_results_csv": CONFORMAL_RESULTS_CSV,
}


def fail(message: str) -> None:
    raise SystemExit(f"VALIDATION FAILED: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing frozen source file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"cannot read {path}: {exc}")
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    require(path.is_file(), f"missing frozen source file: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
    except Exception as exc:
        fail(f"cannot read {path}: {exc}")
    require(fieldnames, f"CSV has no header: {path}")
    require(len(fieldnames) == len(set(fieldnames)), f"CSV has duplicate fields: {path}")
    require(rows, f"CSV has no data rows: {path}")
    return fieldnames, rows


def finite_float(value: Any, description: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        fail(f"{description} is not numeric: {value!r}")
    require(math.isfinite(number), f"{description} is nonfinite")
    return number


def exact_int(value: Any, description: str) -> int:
    require(not isinstance(value, bool), f"{description} is Boolean, not an integer")
    try:
        number = int(value)
    except (TypeError, ValueError):
        fail(f"{description} is not an integer: {value!r}")
    if isinstance(value, str):
        require(str(number) == value, f"{description} is not an exact integer: {value!r}")
    else:
        require(number == value, f"{description} is not an exact integer: {value!r}")
    return number


def bool_value(value: Any, description: str) -> bool:
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    fail(f"{description} is not Boolean: {value!r}")


def require_close(actual: float, expected: float, description: str, tolerance: float = 1e-14) -> None:
    require(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance),
        f"{description} mismatch: {actual!r} != {expected!r}",
    )


def require_vector_close(
    actual: Sequence[float], expected: Sequence[float], description: str, tolerance: float = 1e-14
) -> None:
    require(len(actual) == len(expected), f"{description} length mismatch")
    for index, (left, right) in enumerate(zip(actual, expected)):
        require_close(left, right, f"{description}[{index}]", tolerance)


def numeric_vector(value: Any, description: str, expected_size: int | None = None) -> list[float]:
    require(isinstance(value, list), f"{description} is not a list")
    result = [finite_float(item, f"{description}[{index}]") for index, item in enumerate(value)]
    if expected_size is not None:
        require(len(result) == expected_size, f"{description} length is {len(result)}, expected {expected_size}")
    return result


def rel_scalar(value: float, reference: float) -> float:
    require(reference != 0.0, "zero reference in relative uncertainty")
    return abs(value / reference - 1.0)


def rel_vector(values: Sequence[float], reference: Sequence[float]) -> list[float]:
    require(len(values) == len(reference), "relative-uncertainty vectors have different lengths")
    return [rel_scalar(value, nominal) for value, nominal in zip(values, reference)]


def envelope_scalar(left: float, right: float) -> float:
    return max(left, right)


def envelope_vector(left: Sequence[float], right: Sequence[float]) -> list[float]:
    require(len(left) == len(right), "envelope vectors have different lengths")
    return [max(a, b) for a, b in zip(left, right)]


def quadrature_scalar(*parts: float) -> float:
    require(parts, "empty quadrature")
    return math.sqrt(sum(part * part for part in parts))


def quadrature_vector(*parts: Sequence[float]) -> list[float]:
    require(parts, "empty vector quadrature")
    size = len(parts[0])
    require(all(len(part) == size for part in parts), "quadrature vectors have different lengths")
    return [math.sqrt(sum(part[index] ** 2 for part in parts)) for index in range(size)]


def validate_metrics() -> tuple[dict[tuple[str, float], dict[str, Any]], dict[str, Any]]:
    payload = load_json(METRICS_JSON)
    require(set(payload) == {"provenance", "metrics", "uncertainty_summary"}, "metrics JSON top-level schema mismatch")

    provenance = payload["provenance"]
    require(isinstance(provenance, dict), "metrics provenance is not an object")
    require(provenance.get("model_sha256") == MODEL_SHA256, "metrics model SHA is not the frozen final model")
    require(provenance.get("tagger_source_sha256") == TAGGER_SOURCE_SHA256, "metrics tagger source SHA mismatch")
    require(
        provenance.get("official_reference_commit") == OFFICIAL_REFERENCE_COMMIT,
        "metrics official-reference commit mismatch",
    )
    require(provenance.get("training_rows_requested") == 5_000_000, "metrics are not from the frozen 5M campaign")
    require(provenance.get("seed") == 42, "metrics seed mismatch")

    rows = payload["metrics"]
    require(isinstance(rows, list), "metrics record collection is not a list")
    expected_keys = {
        (dataset, working_point)
        for dataset in BASE_METRIC_DATASETS
        for working_point in (0.5, 0.8)
    } | {(dataset, 0.5) for dataset in SCALE_METRIC_DATASETS}
    require(len(rows) == len(expected_keys), f"metrics record count is {len(rows)}, expected {len(expected_keys)}")

    metric_map: dict[tuple[str, float], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        require(isinstance(row, dict), f"metrics row {index} is not an object")
        dataset = row.get("dataset")
        require(isinstance(dataset, str), f"metrics row {index} has invalid dataset")
        working_point = finite_float(row.get("working_point"), f"metrics working point for {dataset}")
        key = (dataset, working_point)
        require(key in expected_keys, f"unexpected metrics record {key}")
        require(key not in metric_map, f"duplicate metrics record {key}")

        auc = finite_float(row.get("auc"), f"AUC for {key}")
        accuracy = finite_float(row.get("acc"), f"accuracy for {key}")
        background_rejection = finite_float(row.get("br"), f"background rejection for {key}")
        require(0.0 <= auc <= 1.0, f"AUC outside [0,1] for {key}")
        require(0.0 <= accuracy <= 1.0, f"accuracy outside [0,1] for {key}")
        require(background_rejection > 1.0, f"background rejection is not greater than one for {key}")
        require(exact_int(row.get("n"), f"row count for {key}") > 0, f"empty metric sample for {key}")
        require(exact_int(row.get("n_bad_bins"), f"bad-bin count for {key}") == 0, f"invalid pT bins for {key}")

        edges = numeric_vector(row.get("pt_bins"), f"pT bins for {key}")
        binned = numeric_vector(row.get("binned_br"), f"binned BR for {key}")
        require(len(edges) == len(binned), f"pT-edge/BR length mismatch for {key}")
        require(len(edges) >= 2, f"insufficient pT bins for {key}")
        require(all(binned_value > 1.0 for binned_value in binned), f"invalid binned BR for {key}")
        require(binned[-1] == binned[-2], f"missing closing duplicate in binned BR for {key}")
        metric_map[key] = row

    require(set(metric_map) == expected_keys, "metrics dataset/working-point grid is incomplete")
    nominal_50 = metric_map[("test_nominal", 0.5)]
    nominal_80 = metric_map[("test_nominal", 0.8)]
    require_close(float(nominal_50["auc"]), float(nominal_80["auc"]), "nominal AUC across working points", 0.0)
    require_close(float(nominal_50["acc"]), float(nominal_80["acc"]), "nominal accuracy across working points", 0.0)
    require(tuple(float(value) for value in nominal_50["pt_bins"]) == EXPECTED_PT_EDGES_MEV, "nominal pT grid mismatch")

    prediction_rows = provenance.get("prediction_rows")
    require(isinstance(prediction_rows, dict), "metrics provenance lacks prediction row counts")
    for dataset in ("test_nominal", "esup", "esdown", "cer", "cpos", "teg", "tej", "tfl", "tfj", "bias"):
        source_entry = prediction_rows.get(dataset)
        require(isinstance(source_entry, dict), f"missing provenance row count for {dataset}")
        require(
            exact_int(source_entry.get("prediction_rows"), f"provenance rows for {dataset}")
            == exact_int(metric_map[(dataset, 0.5)]["n"], f"metrics rows for {dataset}"),
            f"prediction row count mismatch for {dataset}",
        )

    summary = payload["uncertainty_summary"]
    require(isinstance(summary, dict), "metrics uncertainty summary is not an object")
    require(finite_float(summary.get("wp"), "systematic working point") == 0.5, "systematic working point is not 0.5")
    require(tuple(numeric_vector(summary.get("pt_bins"), "summary pT bins", 15)) == EXPECTED_PT_EDGES_MEV, "summary pT grid mismatch")
    for key in ("nominal_binned_br", "cluster", "track", "scale", "model", "total"):
        vector = numeric_vector(summary.get(key), f"uncertainty summary {key}", 15)
        require(all(value >= 0.0 for value in vector), f"uncertainty summary {key} has negative values")
        require(vector[-1] == vector[-2], f"uncertainty summary {key} lacks closing duplicate")
    for key in ("nominal_br", "inc_cluster", "inc_track", "inc_scale", "inc_model", "inc_total"):
        require(finite_float(summary.get(key), f"uncertainty summary {key}") >= 0.0, f"negative summary value {key}")
    require_close(float(summary["nominal_br"]), float(nominal_50["br"]), "inclusive nominal BR")
    require_vector_close(
        [float(value) for value in summary["nominal_binned_br"]],
        [float(value) for value in nominal_50["binned_br"]],
        "nominal binned BR",
    )
    return metric_map, summary


def calculate_systematics(
    metrics: Mapping[tuple[str, float], Mapping[str, Any]], summary: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    wp = 0.5
    nominal = metrics[("test_nominal", wp)]
    nominal_binned = [float(value) for value in nominal["binned_br"]]
    nominal_inclusive = float(nominal["br"])

    def binned(dataset: str) -> list[float]:
        return [float(value) for value in metrics[(dataset, wp)]["binned_br"]]

    def inclusive(dataset: str) -> float:
        return float(metrics[(dataset, wp)]["br"])

    def detector_component(dataset: str) -> tuple[list[float], float]:
        return (
            rel_vector(binned(dataset), nominal_binned),
            rel_scalar(inclusive(dataset), nominal_inclusive),
        )

    esup_binned, esup_inclusive = detector_component("esup")
    esdown_binned, esdown_inclusive = detector_component("esdown")
    cer_binned, cer_inclusive = detector_component("cer")
    cpos_binned, cpos_inclusive = detector_component("cpos")
    cluster_energy_scale_binned = envelope_vector(esup_binned, esdown_binned)
    cluster_energy_scale_inclusive = envelope_scalar(esup_inclusive, esdown_inclusive)
    cluster_total_binned = quadrature_vector(cluster_energy_scale_binned, cer_binned, cpos_binned)
    cluster_total_inclusive = quadrature_scalar(cluster_energy_scale_inclusive, cer_inclusive, cpos_inclusive)

    teg_binned, teg_inclusive = detector_component("teg")
    tej_binned, tej_inclusive = detector_component("tej")
    tfl_binned, tfl_inclusive = detector_component("tfl")
    tfj_binned, tfj_inclusive = detector_component("tfj")
    bias_binned, bias_inclusive = detector_component("bias")
    track_efficiency_binned = envelope_vector(teg_binned, tej_binned)
    track_efficiency_inclusive = envelope_scalar(teg_inclusive, tej_inclusive)
    fake_rate_binned = envelope_vector(tfl_binned, tfj_binned)
    fake_rate_inclusive = envelope_scalar(tfl_inclusive, tfj_inclusive)
    track_total_binned = quadrature_vector(track_efficiency_binned, fake_rate_binned, bias_binned)
    track_total_inclusive = quadrature_scalar(track_efficiency_inclusive, fake_rate_inclusive, bias_inclusive)

    raw_signal_modelling_binned = rel_vector(binned("ttbar_herwig"), binned("ttbar_pythia"))
    extrapolation_start = next(index for index, edge in enumerate(EXPECTED_PT_EDGES_MEV) if edge > 1_500_000.0)
    require(
        len(raw_signal_modelling_binned) == extrapolation_start,
        "signal-modelling measured-bin count does not match the frozen extrapolation convention",
    )
    signal_modelling_binned = [0.0] * len(EXPECTED_PT_EDGES_MEV)
    signal_modelling_binned[:extrapolation_start] = raw_signal_modelling_binned
    signal_modelling_binned[extrapolation_start:] = [max(raw_signal_modelling_binned)] * (
        len(signal_modelling_binned) - extrapolation_start
    )
    signal_modelling_inclusive = rel_scalar(inclusive("ttbar_herwig"), inclusive("ttbar_pythia"))

    background_parton_shower_binned = rel_vector(binned("dipole"), binned("angular"))
    background_hadronisation_binned = rel_vector(binned("cluster"), binned("string"))
    background_modelling_binned = quadrature_vector(background_parton_shower_binned, background_hadronisation_binned)
    background_parton_shower_inclusive = rel_scalar(inclusive("dipole"), inclusive("angular"))
    background_hadronisation_inclusive = rel_scalar(inclusive("cluster"), inclusive("string"))
    background_modelling_inclusive = quadrature_scalar(
        background_parton_shower_inclusive, background_hadronisation_inclusive
    )
    modelling_total_binned = quadrature_vector(signal_modelling_binned, background_modelling_binned)
    modelling_total_inclusive = quadrature_scalar(signal_modelling_inclusive, background_modelling_inclusive)

    def scale_component(up_dataset: str, down_dataset: str) -> tuple[list[float], float]:
        return (
            envelope_vector(
                rel_vector(binned(up_dataset), nominal_binned),
                rel_vector(binned(down_dataset), nominal_binned),
            ),
            envelope_scalar(
                rel_scalar(inclusive(up_dataset), nominal_inclusive),
                rel_scalar(inclusive(down_dataset), nominal_inclusive),
            ),
        )

    signal_isr_binned, signal_isr_inclusive = scale_component("sig_ISRx2", "sig_ISRxp5")
    signal_fsr_binned, signal_fsr_inclusive = scale_component("sig_FSRx2", "sig_FSRxp5")
    background_isr_binned, background_isr_inclusive = scale_component("bkg_ISRx2", "bkg_ISRxp5")
    background_fsr_binned, background_fsr_inclusive = scale_component("bkg_FSRx2", "bkg_FSRxp5")
    isr_fsr_total_binned = quadrature_vector(
        signal_isr_binned, signal_fsr_binned, background_isr_binned, background_fsr_binned
    )
    isr_fsr_total_inclusive = quadrature_scalar(
        signal_isr_inclusive, signal_fsr_inclusive, background_isr_inclusive, background_fsr_inclusive
    )
    total_binned = quadrature_vector(
        cluster_total_binned, track_total_binned, modelling_total_binned, isr_fsr_total_binned
    )
    total_inclusive = quadrature_scalar(
        cluster_total_inclusive, track_total_inclusive, modelling_total_inclusive, isr_fsr_total_inclusive
    )

    components: dict[str, dict[str, Any]] = {
        "cluster_energy_scale": {
            "category": "Cluster",
            "label": "Cluster energy scale",
            "combination": "Envelope of esup/esdown",
            "inclusive": cluster_energy_scale_inclusive,
            "binned": cluster_energy_scale_binned,
        },
        "cer": {
            "category": "Cluster",
            "label": "CER",
            "combination": "Cluster energy resolution",
            "inclusive": cer_inclusive,
            "binned": cer_binned,
        },
        "cpos": {
            "category": "Cluster",
            "label": "CPOS",
            "combination": "Cluster position resolution",
            "inclusive": cpos_inclusive,
            "binned": cpos_binned,
        },
        "cluster_total": {
            "category": "Cluster",
            "label": "Cluster total",
            "combination": "Quadrature",
            "inclusive": cluster_total_inclusive,
            "binned": cluster_total_binned,
        },
        "track_efficiency": {
            "category": "Track",
            "label": "Track efficiency",
            "combination": "Envelope of teg/tej",
            "inclusive": track_efficiency_inclusive,
            "binned": track_efficiency_binned,
        },
        "fake_rate": {
            "category": "Track",
            "label": "Fake rate",
            "combination": "Envelope of tfl/tfj",
            "inclusive": fake_rate_inclusive,
            "binned": fake_rate_binned,
        },
        "bias": {
            "category": "Track",
            "label": "Bias",
            "combination": "Single variation",
            "inclusive": bias_inclusive,
            "binned": bias_binned,
        },
        "track_total": {
            "category": "Track",
            "label": "Track total",
            "combination": "Quadrature",
            "inclusive": track_total_inclusive,
            "binned": track_total_binned,
        },
        "signal_modelling": {
            "category": "Modelling",
            "label": "Signal modelling",
            "combination": "Herwig/Pythia; frozen extrapolation above 1.5 TeV",
            "inclusive": signal_modelling_inclusive,
            "binned": signal_modelling_binned,
        },
        "background_modelling": {
            "category": "Modelling",
            "label": "Background modelling",
            "combination": "Quadrature of parton shower and hadronisation",
            "inclusive": background_modelling_inclusive,
            "binned": background_modelling_binned,
        },
        "modelling_total": {
            "category": "Modelling",
            "label": "Modelling total",
            "combination": "Quadrature",
            "inclusive": modelling_total_inclusive,
            "binned": modelling_total_binned,
        },
        "signal_isr": {
            "category": "ISR/FSR",
            "label": "Signal ISR",
            "combination": "Envelope of x2/x0.5",
            "inclusive": signal_isr_inclusive,
            "binned": signal_isr_binned,
        },
        "signal_fsr": {
            "category": "ISR/FSR",
            "label": "Signal FSR",
            "combination": "Envelope of x2/x0.5",
            "inclusive": signal_fsr_inclusive,
            "binned": signal_fsr_binned,
        },
        "background_isr": {
            "category": "ISR/FSR",
            "label": "Background ISR",
            "combination": "Envelope of x2/x0.5",
            "inclusive": background_isr_inclusive,
            "binned": background_isr_binned,
        },
        "background_fsr": {
            "category": "ISR/FSR",
            "label": "Background FSR",
            "combination": "Envelope of x2/x0.5",
            "inclusive": background_fsr_inclusive,
            "binned": background_fsr_binned,
        },
        "isr_fsr_total": {
            "category": "ISR/FSR",
            "label": "ISR/FSR total",
            "combination": "Quadrature",
            "inclusive": isr_fsr_total_inclusive,
            "binned": isr_fsr_total_binned,
        },
        "total_uncertainty": {
            "category": "Total",
            "label": "Total uncertainty",
            "combination": "Quadrature of category totals",
            "inclusive": total_inclusive,
            "binned": total_binned,
        },
    }

    summary_checks = {
        "cluster": "cluster_total",
        "track": "track_total",
        "model": "modelling_total",
        "scale": "isr_fsr_total",
        "total": "total_uncertainty",
    }
    inclusive_summary_checks = {
        "inc_cluster": "cluster_total",
        "inc_track": "track_total",
        "inc_model": "modelling_total",
        "inc_scale": "isr_fsr_total",
        "inc_total": "total_uncertainty",
    }
    for summary_key, component_key in summary_checks.items():
        require_vector_close(
            components[component_key]["binned"],
            [float(value) for value in summary[summary_key]],
            f"calculated {component_key} versus frozen summary {summary_key}",
        )
    for summary_key, component_key in inclusive_summary_checks.items():
        require_close(
            float(components[component_key]["inclusive"]),
            float(summary[summary_key]),
            f"calculated {component_key} versus frozen summary {summary_key}",
        )
    return components


def validate_published_hldnn_reference() -> dict[str, float | None]:
    require(TAGGER_SOURCE_SNAPSHOT.is_file(), f"missing frozen tagger source snapshot: {TAGGER_SOURCE_SNAPSHOT}")
    text = TAGGER_SOURCE_SNAPSHOT.read_text(encoding="utf-8")
    require("high-level-quantity baseline" in text, "hlDNN reference is not labelled as a high-level feature baseline")
    require("same fifteen quantities" in text, "hlDNN feature-space statement is missing")
    pattern = re.compile(
        r'AUC\s+(?P<auc>[0-9.]+)\s*\+-\s*(?P<auc_unc>[0-9.]+),\s*ACC\s+(?P<acc>[0-9.]+),\s*'
        r'background rejection\s+(?P<br50>[0-9.]+)\s*\+-\s*(?P<br50_unc>[0-9.]+)\s*'
        r'at 50% signal efficiency and\s+(?P<br80>[0-9.]+)\s*\+-\s*(?P<br80_unc>[0-9.]+)\s*at 80%',
        re.DOTALL,
    )
    match = pattern.search(text)
    require(match is not None, "cannot parse the published hlDNN reference from the frozen tagger snapshot")
    parsed = {key: finite_float(value, f"published hlDNN {key}") for key, value in match.groupdict().items()}
    expected = {
        "auc": 0.9374,
        "auc_unc": 0.0001,
        "acc": 0.8628,
        "br50": 47.2,
        "br50_unc": 0.4,
        "br80": 10.36,
        "br80_unc": 0.03,
    }
    require(parsed == expected, f"published hlDNN values differ from the frozen expected reference: {parsed}")
    return {
        "auc": parsed["auc"],
        "auc_uncertainty": parsed["auc_unc"],
        "accuracy": parsed["acc"],
        "accuracy_uncertainty": PUBLISHED_HLDNN_ACCURACY_UNCERTAINTY,
        "background_rejection_at_50pct_signal_efficiency": parsed["br50"],
        "background_rejection_at_50pct_uncertainty": parsed["br50_unc"],
        "background_rejection_at_80pct_signal_efficiency": parsed["br80"],
        "background_rejection_at_80pct_uncertainty": parsed["br80_unc"],
    }


def typed_shap_row(row: Mapping[str, Any], description: str) -> dict[str, Any]:
    require(set(row) == set(SHAP_FIELDS), f"{description} fields do not match the frozen SHAP schema")
    result: dict[str, Any] = {}
    for field in SHAP_FIELDS:
        value = row[field]
        if field in {"subset", "feature", "systematic"}:
            require(isinstance(value, str) and value, f"{description} {field} is invalid")
            result[field] = value
        elif field in {"n_pairs", "nominal_importance_rank", "systematic_importance_rank", "sample_size"}:
            result[field] = exact_int(value, f"{description} {field}")
        elif field == "relative_attribution_drift_defined":
            result[field] = bool_value(value, f"{description} {field}")
        else:
            result[field] = finite_float(value, f"{description} {field}")
    return result


def validate_shap() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    status = load_json(SHAP_STATUS_JSON)
    require(status.get("status") == "complete", "SHAP round3 campaign is not complete")
    require(status.get("round") == 3, "SHAP campaign status is not round3")
    require(status.get("remediation_only") is True, "SHAP round3 is not marked remediation-only")
    require(status.get("tree_shap_recomputed") is False, "SHAP round3 unexpectedly recomputed TreeSHAP")
    require(status.get("crosscheck") == "PASS", "SHAP round3 cross-check did not pass")
    require(status.get("model_sha256") == MODEL_SHA256, "SHAP round3 model SHA mismatch")

    results = load_json(SHAP_RESULTS_JSON)
    require(set(results) == {"provenance", "systematics", "stability_policy", "per_feature", "remediation"}, "SHAP results schema mismatch")
    provenance = results["provenance"]
    require(isinstance(provenance, dict), "SHAP provenance is not an object")
    require(provenance.get("model_sha256") == MODEL_SHA256, "SHAP results model SHA mismatch")
    metadata = provenance.get("metadata")
    require(isinstance(metadata, dict), "SHAP provenance metadata is missing")
    require(tuple(metadata.get("feature_order", [])) == FEATURES, "SHAP feature order mismatch")
    require(provenance.get("tagger_source_sha256") == TAGGER_SOURCE_SHA256, "SHAP tagger source SHA mismatch")

    remediation = results["remediation"]
    require(isinstance(remediation, dict), "SHAP remediation record is missing")
    require(remediation.get("round") == 3, "SHAP remediation record is not round3")
    require(remediation.get("classification") == "CONFIRMED BUG", "SHAP round3 remediation classification mismatch")
    require(remediation.get("crosscheck") == "PASS", "SHAP round3 remediation cross-check did not pass")
    require(remediation.get("tree_shap_recomputed") is False, "SHAP round3 remediation reran TreeSHAP")
    require(remediation.get("pairing_recomputed") is False, "SHAP round3 remediation reran pairing")
    require(remediation.get("samples_or_identities_changed") is False, "SHAP round3 samples changed")
    require(remediation.get("model_sha256") == MODEL_SHA256, "SHAP remediation model SHA mismatch")
    require(
        remediation.get("non_score_attribution_payload_sha256_before")
        == remediation.get("non_score_attribution_payload_sha256_after"),
        "SHAP non-score attribution payload changed during round3 remediation",
    )

    systematic_records = results["systematics"]
    require(isinstance(systematic_records, dict), "SHAP systematic records are missing")
    require(set(systematic_records) == set(SHAP_SYSTEMATICS), "SHAP systematic identities mismatch")
    for systematic in SHAP_SYSTEMATICS:
        record = systematic_records[systematic]
        require(isinstance(record, dict), f"SHAP systematic record is invalid for {systematic}")
        require(tuple(record.get("sample_sizes", [])) == SHAP_SAMPLE_SIZES, f"SHAP sample-size grid mismatch for {systematic}")
        require(record.get("source_reconstruction_fingerprint") == provenance.get("metadata", {}).get("reconstruction_fingerprint"), f"SHAP reconstruction fingerprint mismatch for {systematic}")

    csv_fields, csv_rows = load_csv(SHAP_PER_FEATURE_CSV)
    require(tuple(csv_fields) == SHAP_FIELDS, "SHAP CSV field order/schema mismatch")
    json_rows = results["per_feature"]
    require(isinstance(json_rows, list), "SHAP per-feature JSON collection is not a list")
    require(len(csv_rows) == 540 and len(json_rows) == 540, "SHAP round3 must contain exactly 540 complete rows")

    typed_csv_rows = [typed_shap_row(row, f"SHAP CSV row {index}") for index, row in enumerate(csv_rows)]
    typed_json_rows = [typed_shap_row(row, f"SHAP JSON row {index}") for index, row in enumerate(json_rows)]
    key_fields = ("systematic", "sample_size", "subset", "feature")

    def keyed(rows: Iterable[dict[str, Any]], description: str) -> dict[tuple[Any, ...], dict[str, Any]]:
        result: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in rows:
            key = tuple(row[field] for field in key_fields)
            require(key not in result, f"duplicate {description} SHAP row {key}")
            result[key] = row
        return result

    csv_map = keyed(typed_csv_rows, "CSV")
    json_map = keyed(typed_json_rows, "JSON")
    require(set(csv_map) == set(json_map), "SHAP CSV/JSON row grids differ")
    for key in csv_map:
        require(csv_map[key] == json_map[key], f"SHAP CSV/JSON value mismatch for {key}")

    expected_keys = {
        (systematic, sample_size, subset, feature)
        for systematic in SHAP_SYSTEMATICS
        for sample_size in SHAP_SAMPLE_SIZES
        for subset in SUBSETS
        for feature in FEATURES
    }
    require(set(csv_map) == expected_keys, "SHAP systematic/sample/subset/feature grid is incomplete")

    for systematic in SHAP_SYSTEMATICS:
        for sample_size in SHAP_SAMPLE_SIZES:
            subset_counts: dict[str, int] = {}
            for subset in SUBSETS:
                group = [
                    csv_map[(systematic, sample_size, subset, feature)]
                    for feature in FEATURES
                ]
                require({row["feature"] for row in group} == set(FEATURES), f"SHAP feature set mismatch for {systematic}/{sample_size}/{subset}")
                n_pairs = {row["n_pairs"] for row in group}
                require(len(n_pairs) == 1, f"SHAP n_pairs varies across features for {systematic}/{sample_size}/{subset}")
                subset_counts[subset] = next(iter(n_pairs))
                require(subset_counts[subset] > 0, f"SHAP group is empty for {systematic}/{sample_size}/{subset}")
                for score_field in ("mean_signed_delta_score", "median_signed_delta_score", "mean_abs_delta_score"):
                    require(len({row[score_field] for row in group}) == 1, f"SHAP {score_field} varies across features for {systematic}/{sample_size}/{subset}")
                require({row["nominal_importance_rank"] for row in group} == set(range(1, 16)), f"SHAP nominal ranks are invalid for {systematic}/{sample_size}/{subset}")
                require({row["systematic_importance_rank"] for row in group} == set(range(1, 16)), f"SHAP systematic ranks are invalid for {systematic}/{sample_size}/{subset}")
                require(all(row["mean_abs_delta_shap"] >= 0.0 for row in group), f"negative mean |Delta SHAP| for {systematic}/{sample_size}/{subset}")
                require(all(row["mean_abs_delta_score"] >= 0.0 for row in group), f"negative mean |Delta score| for {systematic}/{sample_size}/{subset}")
                require(all(-1.0 <= row["global_importance_spearman_rho"] <= 1.0 for row in group), f"invalid SHAP Spearman rho for {systematic}/{sample_size}/{subset}")
                require(all(0.0 <= row["global_importance_spearman_pvalue"] <= 1.0 for row in group), f"invalid SHAP Spearman p-value for {systematic}/{sample_size}/{subset}")
            require(subset_counts["all"] == sample_size, f"SHAP all-pair count mismatch for {systematic}/{sample_size}")
            require(subset_counts["signal"] + subset_counts["background"] == subset_counts["all"], f"SHAP class-pair counts do not sum for {systematic}/{sample_size}")

    table3_rows: list[dict[str, Any]] = []
    for systematic in SHAP_SYSTEMATICS:
        groups = {
            subset: [csv_map[(systematic, 100_000, subset, feature)] for feature in FEATURES]
            for subset in SUBSETS
        }
        all_group = groups["all"]
        max_value = max(row["mean_abs_delta_shap"] for row in all_group)
        leaders = [row for row in all_group if row["mean_abs_delta_shap"] == max_value]
        require(len(leaders) == 1, f"SHAP leading feature is tied for {systematic}")
        table3_rows.append(
            {
                "systematic": systematic,
                "sample_size": 100_000,
                "mean_signed_delta_score_all": groups["all"][0]["mean_signed_delta_score"],
                "mean_signed_delta_score_signal": groups["signal"][0]["mean_signed_delta_score"],
                "mean_signed_delta_score_background": groups["background"][0]["mean_signed_delta_score"],
                "mean_abs_delta_score_overall": groups["all"][0]["mean_abs_delta_score"],
                "leading_feature_by_mean_abs_delta_shap": leaders[0]["feature"],
                "leading_mean_abs_delta_shap": max_value,
            }
        )

    ordered_rows = [
        csv_map[(systematic, sample_size, subset, feature)]
        for systematic in SHAP_SYSTEMATICS
        for sample_size in SHAP_SAMPLE_SIZES
        for subset in SUBSETS
        for feature in FEATURES
    ]
    return table3_rows, ordered_rows


def typed_conformal_row(row: Mapping[str, Any], description: str) -> dict[str, Any]:
    require(set(row) == set(CONFORMAL_FIELDS), f"{description} fields do not match the frozen conformal schema")
    result: dict[str, Any] = {}
    count_fields = {
        "total_count",
        "signal_count",
        "background_count",
        "singleton_count",
        "ambiguous_0_1_count",
        "empty_set_count",
    }
    for field in CONFORMAL_FIELDS:
        value = row[field]
        if field == "dataset":
            require(isinstance(value, str) and value, f"{description} dataset is invalid")
            result[field] = value
        elif field in count_fields:
            result[field] = exact_int(value, f"{description} {field}")
        else:
            result[field] = finite_float(value, f"{description} {field}")
    return result


def validate_conformal() -> list[dict[str, Any]]:
    provenance_file = load_json(CONFORMAL_PROVENANCE_JSON)
    calibration_file = load_json(CONFORMAL_CALIBRATION_JSON)
    results = load_json(CONFORMAL_RESULTS_JSON)
    require(set(results) == {"provenance", "calibration", "datasets"}, "conformal results schema mismatch")
    require(results["provenance"] == provenance_file, "conformal provenance sidecar disagrees with results JSON")
    require(results["calibration"] == calibration_file, "conformal calibration sidecar disagrees with results JSON")

    provenance = results["provenance"]
    require(provenance.get("model_sha256") == MODEL_SHA256, "conformal model SHA mismatch")
    require(provenance.get("tagger_source_snapshot_sha256") == TAGGER_SOURCE_SHA256, "conformal tagger source SHA mismatch")
    require(tuple(provenance.get("feature_order", [])) == FEATURES, "conformal feature order mismatch")
    require(provenance.get("training_or_refitting_performed") is False, "conformal round1 refit the model")
    require(provenance.get("imputation_performed") is False, "conformal round1 performed imputation")
    require(provenance.get("pairing_used") is False, "conformal round1 unexpectedly used pairing")
    require(provenance.get("coverage_weighting") == "unweighted jets", "conformal coverage weighting mismatch")
    require(tuple(provenance.get("evaluated_datasets", [])) == CONFORMAL_DATASETS, "conformal dataset order/grid mismatch")

    calibration = results["calibration"]
    require(isinstance(calibration, dict), "conformal calibration record is invalid")
    require_close(finite_float(calibration.get("alpha"), "conformal alpha"), 0.1, "conformal alpha", 0.0)
    require_close(finite_float(calibration.get("target_coverage"), "conformal target"), 0.9, "conformal target", 0.0)
    require(calibration.get("calibrated_once_on_nominal") is True, "conformal calibration was not frozen nominal-only")
    require(calibration.get("calibration_dataset") == "train_nominal", "conformal calibration dataset mismatch")

    csv_fields, csv_rows = load_csv(CONFORMAL_RESULTS_CSV)
    require(tuple(csv_fields) == CONFORMAL_FIELDS, "conformal CSV field order/schema mismatch")
    json_rows = results["datasets"]
    require(isinstance(json_rows, list), "conformal dataset results are not a list")
    require(len(csv_rows) == len(CONFORMAL_DATASETS) and len(json_rows) == len(CONFORMAL_DATASETS), "conformal result row count mismatch")

    typed_csv_rows = [typed_conformal_row(row, f"conformal CSV row {index}") for index, row in enumerate(csv_rows)]
    typed_json_rows = [typed_conformal_row(row, f"conformal JSON row {index}") for index, row in enumerate(json_rows)]
    csv_map = {row["dataset"]: row for row in typed_csv_rows}
    json_map = {row["dataset"]: row for row in typed_json_rows}
    require(len(csv_map) == len(csv_rows), "duplicate dataset in conformal CSV")
    require(len(json_map) == len(json_rows), "duplicate dataset in conformal JSON")
    require(set(csv_map) == set(CONFORMAL_DATASETS), "conformal CSV dataset grid mismatch")
    require(set(json_map) == set(CONFORMAL_DATASETS), "conformal JSON dataset grid mismatch")
    for dataset in CONFORMAL_DATASETS:
        require(csv_map[dataset] == json_map[dataset], f"conformal CSV/JSON value mismatch for {dataset}")
        row = csv_map[dataset]
        require(row["signal_count"] + row["background_count"] == row["total_count"], f"conformal class counts do not sum for {dataset}")
        require(row["singleton_count"] + row["ambiguous_0_1_count"] + row["empty_set_count"] == row["total_count"], f"conformal set-type counts do not sum for {dataset}")
        for field in ("overall_coverage", "signal_coverage", "background_coverage", "singleton_rate", "ambiguous_0_1_rate", "empty_set_rate"):
            require(0.0 <= row[field] <= 1.0, f"conformal {field} outside [0,1] for {dataset}")
        require_close(row["overall_deviation_from_target"], row["overall_coverage"] - 0.9, f"overall target deviation for {dataset}")
        require_close(row["signal_deviation_from_target"], row["signal_coverage"] - 0.9, f"signal target deviation for {dataset}")
        require_close(row["background_deviation_from_target"], row["background_coverage"] - 0.9, f"background target deviation for {dataset}")
        require_close(row["singleton_rate"] + row["ambiguous_0_1_rate"] + row["empty_set_rate"], 1.0, f"prediction-set rates for {dataset}")

    nominal = csv_map["test_nominal"]
    for dataset in CONFORMAL_DATASETS:
        row = csv_map[dataset]
        require_close(row["overall_deviation_from_nominal"], row["overall_coverage"] - nominal["overall_coverage"], f"overall nominal deviation for {dataset}")
        require_close(row["signal_deviation_from_nominal"], row["signal_coverage"] - nominal["signal_coverage"], f"signal nominal deviation for {dataset}")
        require_close(row["background_deviation_from_nominal"], row["background_coverage"] - nominal["background_coverage"], f"background nominal deviation for {dataset}")
    return [csv_map[dataset] for dataset in CONFORMAL_DATASETS]


def csv_text(fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def tex_number(value: Any, significant_digits: int = 7) -> str:
    if value is None or value == "":
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    number = finite_float(value, "LaTeX table number")
    return format(number, f".{significant_digits}g")


def tex_row(values: Sequence[Any]) -> str:
    return " & ".join(str(value) for value in values) + r" \\"


def table1_tex(rows: Sequence[Mapping[str, Any]]) -> str:
    body: list[str] = []
    for row in rows:
        def measurement(value_key: str, uncertainty_key: str) -> str:
            value = tex_number(row[value_key], 6)
            uncertainty = row[uncertainty_key]
            return value if uncertainty in (None, "") else rf"${value} \pm {tex_number(uncertainty, 3)}$"

        body.append(
            tex_row(
                (
                    latex_escape(row["model"]),
                    latex_escape(row["comparison_role"]),
                    measurement("auc", "auc_uncertainty"),
                    measurement("accuracy", "accuracy_uncertainty"),
                    measurement("background_rejection_at_50pct_signal_efficiency", "background_rejection_at_50pct_uncertainty"),
                    measurement("background_rejection_at_80pct_signal_efficiency", "background_rejection_at_80pct_uncertainty"),
                )
            )
        )
    return "\n".join(
        (
            r"% Requires \usepackage{booktabs}.",
            r"\begin{table}[tbp]",
            r"\centering",
            r"\caption{Nominal XGBoost performance. The ATLAS hlDNN row is a published external reference in the same 15-variable high-level feature space; it is not a reproduction.}",
            r"\label{tab:nominal-performance}",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            tex_row(("Model", "Role", "AUC", "Accuracy", r"BR@50\%", r"BR@80\%")),
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        )
    )


def table2_tex(rows: Sequence[Mapping[str, Any]]) -> str:
    body = [
        tex_row(
            (
                latex_escape(row["category"]),
                latex_escape(row["component"]),
                f'{float(row["relative_uncertainty_percent"]):.2f}',
            )
        )
        for row in rows
    ]
    return "\n".join(
        (
            r"% Requires \usepackage{booktabs}.",
            r"\begin{table}[tbp]",
            r"\centering",
            r"\caption{Inclusive relative systematic uncertainties at 50\% signal efficiency. Envelopes are taken before quadrature combination.}",
            r"\label{tab:inclusive-systematics-wp50}",
            r"\begin{tabular}{llr}",
            r"\toprule",
            tex_row(("Category", "Component", r"Relative uncertainty [\%]")),
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        )
    )


def table3_tex(rows: Sequence[Mapping[str, Any]]) -> str:
    body = [
        tex_row(
            (
                latex_escape(row["systematic"]),
                tex_number(row["mean_signed_delta_score_all"], 7),
                tex_number(row["mean_signed_delta_score_signal"], 7),
                tex_number(row["mean_signed_delta_score_background"], 7),
                tex_number(row["mean_abs_delta_score_overall"], 7),
                latex_escape(row["leading_feature_by_mean_abs_delta_shap"]),
                tex_number(row["leading_mean_abs_delta_shap"], 7),
            )
        )
        for row in rows
    ]
    return "\n".join(
        (
            r"% Requires \usepackage{booktabs}.",
            r"\begin{table*}[tbp]",
            r"\centering",
            r"\caption{Final 100k paired SHAP-Delta summary from round 3. Score changes are in probability space; SHAP changes are in XGBoost margin space.}",
            r"\label{tab:shap-delta-100k}",
            r"\begin{tabular}{lrrrrlr}",
            r"\toprule",
            tex_row(("Systematic", r"Mean $\Delta$s, all", r"Mean $\Delta$s, signal", r"Mean $\Delta$s, background", r"Mean $|\Delta s|$, all", r"Leading feature", r"Leading mean $|\Delta\mathrm{SHAP}|$")),
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        )
    )


def supplementary_systematics_tex(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = list(rows[0])
    headers = (
        r"$p_{\mathrm{T}}$ low [GeV]",
        r"$p_{\mathrm{T}}$ high [GeV]",
        r"Nominal BR",
        "CES",
        "CER",
        "CPOS",
        "Cluster total",
        "Track eff.",
        "Fake rate",
        "Bias",
        "Track total",
        "Signal model",
        "Background model",
        "Model total",
        "Signal ISR",
        "Signal FSR",
        "Background ISR",
        "Background FSR",
        "ISR/FSR total",
        "Total",
    )
    require(len(fields) == len(headers), "supplementary systematics header length mismatch")
    body = [tex_row(tuple(tex_number(row[field], 6) for field in fields)) for row in rows]
    return "\n".join(
        (
            r"% Requires \usepackage{booktabs,graphicx}.",
            r"\begin{table*}[p]",
            r"\centering",
            r"\caption{Full 14-bin relative systematic uncertainties at 50\% signal efficiency. All uncertainty columns are percentages.}",
            r"\label{tab:supp-systematics-14bin}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{" + "r" * len(fields) + "}",
            r"\toprule",
            tex_row(headers),
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        )
    )


def supplementary_shap_tex(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = (
        "Subset", "Feature", "$n$", r"Mean $\Delta$SHAP", r"Median $\Delta$SHAP",
        r"Mean $|\Delta$SHAP$|$", r"Median $|\Delta$SHAP$|$", "Nom. mean |SHAP|",
        "Sys. mean |SHAP|", "Rel. attribution drift", "Defined", "Nom. rank", "Sys. rank",
        r"$\rho$", "$p$", "Systematic", "Sample size", r"Mean $\Delta$score",
        r"Median $\Delta$score", r"Mean $|\Delta$score$|$",
    )
    body: list[str] = []
    for row in rows:
        values = []
        for field in SHAP_FIELDS:
            value = row[field]
            if field in {"subset", "feature", "systematic"}:
                values.append(latex_escape(value))
            else:
                values.append(tex_number(value, 6))
        body.append(tex_row(tuple(values)))
    column_spec = "ll" + "r" * 13 + "l" + "r" * 4
    return "\n".join(
        (
            r"% Requires \usepackage{booktabs,longtable,pdflscape}.",
            r"\begin{landscape}",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2pt}",
            r"\begin{longtable}{" + column_spec + "}",
            r"\caption{Complete 15-feature paired SHAP-Delta round-3 results. SHAP values are in XGBoost margin space; score values are in probability space.}",
            r"\label{tab:supp-shap-round3}\\",
            r"\toprule",
            tex_row(headers),
            r"\midrule",
            r"\endfirsthead",
            r"\toprule",
            tex_row(headers),
            r"\midrule",
            r"\endhead",
            *body,
            r"\bottomrule",
            r"\end{longtable}",
            r"\end{landscape}",
            "",
        )
    )


def supplementary_conformal_tex(rows: Sequence[Mapping[str, Any]]) -> str:
    headers = (
        "Dataset", "$n$", "$n_s$", "$n_b$", "Coverage", "Signal cov.", "Background cov.",
        r"$\Delta$ target", r"$\Delta_s$ target", r"$\Delta_b$ target", "Mean set size",
        "Singleton rate", "Ambiguous rate", "Empty rate", "Singleton $n$", "Ambiguous $n$",
        "Empty $n$", r"$\Delta$ nominal", r"$\Delta_s$ nominal", r"$\Delta_b$ nominal",
    )
    body: list[str] = []
    for row in rows:
        values = [latex_escape(row["dataset"])] + [tex_number(row[field], 7) for field in CONFORMAL_FIELDS[1:]]
        body.append(tex_row(tuple(values)))
    return "\n".join(
        (
            r"% Requires \usepackage{booktabs,graphicx}.",
            r"\begin{table*}[p]",
            r"\centering",
            r"\caption{Complete class-conditional split-conformal round-1 results at target coverage 0.9.}",
            r"\label{tab:supp-conformal-round1}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{l" + "r" * (len(CONFORMAL_FIELDS) - 1) + "}",
            r"\toprule",
            tex_row(headers),
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        )
    )


def validate_rendered_csv(text: str, expected_fields: Sequence[str], expected_rows: int, description: str) -> None:
    reader = csv.DictReader(io.StringIO(text))
    require(tuple(reader.fieldnames or []) == tuple(expected_fields), f"{description} rendered CSV fields mismatch")
    rows = list(reader)
    require(len(rows) == expected_rows, f"{description} rendered CSV row count mismatch")
    require(all(set(row) == set(expected_fields) for row in rows), f"{description} rendered CSV row schema mismatch")


def validate_rendered_tex(text: str, expected_rows: int, description: str) -> None:
    require(text.endswith("\n"), f"{description} LaTeX lacks terminal newline")
    require("\\caption{" in text and "\\label{" in text, f"{description} LaTeX lacks caption or label")
    require(text.count(r" \\") >= expected_rows, f"{description} LaTeX lacks expected data rows")
    require(re.search(r"(?i)(?<![A-Za-z])(nan|inf)(?![A-Za-z])", text) is None, f"{description} LaTeX contains nonfinite text")


def build_tables() -> tuple[dict[str, str], dict[str, int]]:
    metrics, summary = validate_metrics()
    components = calculate_systematics(metrics, summary)
    published_hldnn = validate_published_hldnn_reference()
    shap_table3_rows, shap_complete_rows = validate_shap()
    conformal_rows = validate_conformal()

    nominal_50 = metrics[("test_nominal", 0.5)]
    nominal_80 = metrics[("test_nominal", 0.8)]
    table1_fields = (
        "model",
        "comparison_role",
        "feature_space",
        "auc",
        "auc_uncertainty",
        "accuracy",
        "accuracy_uncertainty",
        "background_rejection_at_50pct_signal_efficiency",
        "background_rejection_at_50pct_uncertainty",
        "background_rejection_at_80pct_signal_efficiency",
        "background_rejection_at_80pct_uncertainty",
    )
    table1_rows = [
        {
            "model": "XGBoost (this work)",
            "comparison_role": "Frozen final result",
            "feature_space": "15 high-level jet-substructure features",
            "auc": float(nominal_50["auc"]),
            "auc_uncertainty": "",
            "accuracy": float(nominal_50["acc"]),
            "accuracy_uncertainty": "",
            "background_rejection_at_50pct_signal_efficiency": float(nominal_50["br"]),
            "background_rejection_at_50pct_uncertainty": "",
            "background_rejection_at_80pct_signal_efficiency": float(nominal_80["br"]),
            "background_rejection_at_80pct_uncertainty": "",
        },
        {
            "model": "ATLAS hlDNN",
            "comparison_role": "Published external reference; not reproduced",
            "feature_space": "Same 15 high-level jet-substructure features",
            **{key: "" if value is None else value for key, value in published_hldnn.items()},
        },
    ]

    table2_fields = (
        "category",
        "component",
        "combination",
        "relative_uncertainty_fraction",
        "relative_uncertainty_percent",
    )
    table2_rows = [
        {
            "category": component["category"],
            "component": component["label"],
            "combination": component["combination"],
            "relative_uncertainty_fraction": float(component["inclusive"]),
            "relative_uncertainty_percent": 100.0 * float(component["inclusive"]),
        }
        for component in components.values()
    ]

    table3_fields = tuple(shap_table3_rows[0])
    supplementary_systematics_fields = (
        "pt_low_GeV",
        "pt_high_GeV",
        "nominal_background_rejection_wp50",
        "cluster_energy_scale_percent",
        "cer_percent",
        "cpos_percent",
        "cluster_total_percent",
        "track_efficiency_percent",
        "fake_rate_percent",
        "bias_percent",
        "track_total_percent",
        "signal_modelling_percent",
        "background_modelling_percent",
        "modelling_total_percent",
        "signal_isr_percent",
        "signal_fsr_percent",
        "background_isr_percent",
        "background_fsr_percent",
        "isr_fsr_total_percent",
        "total_uncertainty_percent",
    )
    component_keys = tuple(components)
    supplementary_systematics_rows: list[dict[str, Any]] = []
    nominal_binned = [float(value) for value in summary["nominal_binned_br"]]
    for index in range(14):
        row: dict[str, Any] = {
            "pt_low_GeV": EXPECTED_PT_EDGES_MEV[index] / 1000.0,
            "pt_high_GeV": EXPECTED_PT_EDGES_MEV[index + 1] / 1000.0,
            "nominal_background_rejection_wp50": nominal_binned[index],
        }
        for field, component_key in zip(supplementary_systematics_fields[3:], component_keys):
            row[field] = 100.0 * float(components[component_key]["binned"][index])
        require(tuple(row) == supplementary_systematics_fields, f"14-bin table field order mismatch at bin {index}")
        supplementary_systematics_rows.append(row)

    outputs = {
        TABLE_OUTPUTS["table1_nominal_performance"][0]: csv_text(table1_fields, table1_rows),
        TABLE_OUTPUTS["table1_nominal_performance"][1]: table1_tex(table1_rows),
        TABLE_OUTPUTS["table2_inclusive_systematics_wp50"][0]: csv_text(table2_fields, table2_rows),
        TABLE_OUTPUTS["table2_inclusive_systematics_wp50"][1]: table2_tex(table2_rows),
        TABLE_OUTPUTS["table3_shap_delta_100k_summary"][0]: csv_text(table3_fields, shap_table3_rows),
        TABLE_OUTPUTS["table3_shap_delta_100k_summary"][1]: table3_tex(shap_table3_rows),
        TABLE_OUTPUTS["supplementary_systematics_14bin"][0]: csv_text(supplementary_systematics_fields, supplementary_systematics_rows),
        TABLE_OUTPUTS["supplementary_systematics_14bin"][1]: supplementary_systematics_tex(supplementary_systematics_rows),
        TABLE_OUTPUTS["supplementary_shap_delta_round3_all_features"][0]: csv_text(SHAP_FIELDS, shap_complete_rows),
        TABLE_OUTPUTS["supplementary_shap_delta_round3_all_features"][1]: supplementary_shap_tex(shap_complete_rows),
        TABLE_OUTPUTS["supplementary_conformal_round1"][0]: csv_text(CONFORMAL_FIELDS, conformal_rows),
        TABLE_OUTPUTS["supplementary_conformal_round1"][1]: supplementary_conformal_tex(conformal_rows),
    }
    row_counts = {
        "table1_nominal_performance": len(table1_rows),
        "table2_inclusive_systematics_wp50": len(table2_rows),
        "table3_shap_delta_100k_summary": len(shap_table3_rows),
        "supplementary_systematics_14bin": len(supplementary_systematics_rows),
        "supplementary_shap_delta_round3_all_features": len(shap_complete_rows),
        "supplementary_conformal_round1": len(conformal_rows),
    }

    csv_fields_by_table = {
        "table1_nominal_performance": table1_fields,
        "table2_inclusive_systematics_wp50": table2_fields,
        "table3_shap_delta_100k_summary": table3_fields,
        "supplementary_systematics_14bin": supplementary_systematics_fields,
        "supplementary_shap_delta_round3_all_features": SHAP_FIELDS,
        "supplementary_conformal_round1": CONFORMAL_FIELDS,
    }
    for table_name, (csv_filename, tex_filename) in TABLE_OUTPUTS.items():
        validate_rendered_csv(outputs[csv_filename], csv_fields_by_table[table_name], row_counts[table_name], table_name)
        validate_rendered_tex(outputs[tex_filename], row_counts[table_name], table_name)
    require(len(outputs) == 12, "expected exactly 12 table output files")
    return outputs, row_counts


def write_outputs(outputs: Mapping[str, str], row_counts: Mapping[str, int], source_hashes: Mapping[str, str]) -> Path:
    require(not OUTPUT_DIR.exists(), f"output directory already exists; refusing overwrite: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    for filename, content in outputs.items():
        path = OUTPUT_DIR / filename
        require(not path.exists(), f"refusing to overwrite output: {path}")
        with path.open("x", encoding="utf-8", newline="") as handle:
            handle.write(content)

    output_records: dict[str, Any] = {}
    for table_name, (csv_filename, tex_filename) in TABLE_OUTPUTS.items():
        csv_path = OUTPUT_DIR / csv_filename
        tex_path = OUTPUT_DIR / tex_filename
        require(csv_path.is_file() and csv_path.stat().st_size > 0, f"missing or empty output: {csv_path}")
        require(tex_path.is_file() and tex_path.stat().st_size > 0, f"missing or empty output: {tex_path}")
        output_records[table_name] = {
            "row_count": row_counts[table_name],
            "csv_filename": csv_filename,
            "csv_sha256": sha256(csv_path),
            "tex_filename": tex_filename,
            "tex_sha256": sha256(tex_path),
        }

    manifest = {
        "manifest_schema_version": 1,
        "script": {
            "path": relative_path(Path(__file__)),
            "sha256": sha256(Path(__file__)),
        },
        "frozen_model_sha256": MODEL_SHA256,
        "frozen_only": True,
        "analysis_rerun": False,
        "reconstruction_run": False,
        "training_or_inference_run": False,
        "metrics_run": False,
        "shap_run": False,
        "conformal_run": False,
        "shap_round": 3,
        "conformal_round": 1,
        "sources": {
            role: {"path": relative_path(path), "sha256": source_hashes[role]}
            for role, path in SOURCE_PATHS.items()
        },
        "outputs": output_records,
        "output_filenames": list(outputs),
        "manifest_filename": "table_manifest.json",
        "regeneration_command": 'python "Code files/publication/make_publication_tables.py"',
    }
    manifest_path = OUTPUT_DIR / "table_manifest.json"
    require(not manifest_path.exists(), f"refusing to overwrite output: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    require(manifest_path.stat().st_size > 0, "empty table manifest")
    return manifest_path


def main() -> None:
    require(not OUTPUT_DIR.exists(), f"output directory already exists; refusing overwrite: {OUTPUT_DIR}")
    for role, path in SOURCE_PATHS.items():
        require(path.is_file(), f"missing {role}: {path}")
    require(sha256(TAGGER_SOURCE_SNAPSHOT) == TAGGER_SOURCE_SHA256, "frozen tagger source snapshot SHA mismatch")
    source_hashes = {role: sha256(path) for role, path in SOURCE_PATHS.items()}

    outputs, row_counts = build_tables()

    for role, path in SOURCE_PATHS.items():
        require(sha256(path) == source_hashes[role], f"source changed during validation: {path}")
    manifest_path = write_outputs(outputs, row_counts, source_hashes)
    for role, path in SOURCE_PATHS.items():
        require(sha256(path) == source_hashes[role], f"source changed while writing tables: {path}")

    print("1. validation status: PASSED")
    print("2. files generated:")
    for filename in outputs:
        print(f"   {relative_path(OUTPUT_DIR / filename)}")
    print(f"   {relative_path(manifest_path)}")
    print("3. frozen sources used:")
    for path in SOURCE_PATHS.values():
        print(f"   {relative_path(path)}")
    print('4. regeneration command: python "Code files/publication/make_publication_tables.py"')


if __name__ == "__main__":
    main()
