"""Render the three publication figures from frozen ATLAS80030 results only.

This script performs no reconstruction, training, inference, metric evaluation,
SHAP computation, pairing, or conformal calibration.  It validates and plots
already-saved final result tables, and refuses to overwrite its output round.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
SHAP_DIR = RESULTS_ROOT / "shap_delta_final_production_round3"
SHAP_CSV = SHAP_DIR / "shap_delta_per_feature.csv"
SHAP_STATUS_JSON = SHAP_DIR / "campaign_status.json"
CONFORMAL_JSON = (
    RESULTS_ROOT
    / "conformal_coverage_round1"
    / "conformal_coverage_results.json"
)
OUTPUT_DIR = RESULTS_ROOT / "publication_figures_round2"

MODEL_SHA256 = "63a23308efbf2ce90418c8e657730f571db4798fc9ca2fb3692cb986de405f4c"
OFFICIAL_REFERENCE_COMMIT = "5bc3c8d70cf8736d24b969972a3dd365017a9e24"

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
FEATURE_LABELS = (
    r"$\tau_1^{\mathrm{WTA}}$",
    r"$\tau_2^{\mathrm{WTA}}$",
    r"$\tau_3^{\mathrm{WTA}}$",
    r"$\tau_4^{\mathrm{WTA}}$",
    r"$\sqrt{d_{12}}$",
    r"$\sqrt{d_{23}}$",
    r"$\mathrm{ECF}_1$",
    r"$\mathrm{ECF}_2$",
    r"$\mathrm{ECF}_3$",
    r"$C_2$",
    r"$D_2$",
    r"$L_2$",
    r"$L_3$",
    r"$Q_w$",
    r"$\mathrm{ThrustMaj}$",
)
SHAP_SYSTEMATICS = ("esup", "esdown", "cpos", "cer")
SHAP_SYSTEMATIC_LABELS = (
    "ES up",
    "ES down",
    "CPOS\n(cluster position resolution)",
    "CER\n(cluster energy resolution)",
)
SUBSETS = ("all", "signal", "background")
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

FIGURE_OUTPUTS = {
    "figure1": {
        "pdf": "figure1_performance_systematics.pdf",
        "png": "figure1_performance_systematics.png",
        "data_csv": "figure1_performance_systematics_data.csv",
    },
    "figure2": {
        "pdf": "figure2_shap_delta.pdf",
        "png": "figure2_shap_delta.png",
        "data_csv": "figure2_shap_delta_data.csv",
    },
    "figure3": {
        "pdf": "figure3_conformal_coverage.pdf",
        "png": "figure3_conformal_coverage.png",
        "data_csv": "figure3_conformal_coverage_data.csv",
    },
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


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing frozen source file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(f"cannot read {path}: {exc}")
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def finite_float(value: Any, description: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        fail(f"{description} is not numeric: {value!r}")
    require(math.isfinite(number), f"{description} is nonfinite")
    return number


def exact_int(value: Any, description: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        fail(f"{description} is not an integer: {value!r}")
    require(str(number) == str(value), f"{description} is not an exact integer: {value!r}")
    return number


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "legend.frameon": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "lines.linewidth": 1.5,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.02,
        0.97,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )


def write_csv_new(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    require(not path.exists(), f"refusing to overwrite output: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def save_figure_new(figure: plt.Figure, pdf_path: Path, png_path: Path, title: str) -> None:
    require(not pdf_path.exists(), f"refusing to overwrite output: {pdf_path}")
    require(not png_path.exists(), f"refusing to overwrite output: {png_path}")
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={"Title": title, "Creator": Path(__file__).name},
    )
    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        metadata={"Title": title, "Software": Path(__file__).name},
    )
    plt.close(figure)


def vector(summary: dict[str, Any], key: str, expected_size: int = 15) -> np.ndarray:
    require(key in summary, f"metrics uncertainty summary lacks {key}")
    values = np.asarray(summary[key], dtype=np.float64)
    require(values.shape == (expected_size,), f"{key} shape {values.shape}, expected {(expected_size,)}")
    require(np.isfinite(values).all(), f"{key} contains nonfinite values")
    return values


def validate_metrics() -> dict[str, np.ndarray]:
    payload = load_json(METRICS_JSON)
    provenance = payload.get("provenance")
    require(isinstance(provenance, dict), "metrics JSON has no provenance object")
    require(provenance.get("model_sha256") == MODEL_SHA256, "metrics model SHA is not frozen final model")
    require(
        provenance.get("official_reference_commit") == OFFICIAL_REFERENCE_COMMIT,
        "metrics official-reference commit mismatch",
    )
    require(provenance.get("training_rows_requested") == 5_000_000, "metrics are not from frozen 5M campaign")

    summary = payload.get("uncertainty_summary")
    require(isinstance(summary, dict), "metrics JSON has no uncertainty_summary object")
    require(finite_float(summary.get("wp"), "systematic working point") == 0.5, "systematic working point is not 0.5")

    edges_mev = vector(summary, "pt_bins")
    expected_edges = np.linspace(350_000.0, 3_150_000.0, 15)
    require(np.array_equal(edges_mev, expected_edges), "pT edges are not the frozen 350--3150 GeV grid")

    arrays = {
        "edges_mev": edges_mev,
        "nominal_br": vector(summary, "nominal_binned_br"),
        "cluster": vector(summary, "cluster"),
        "track": vector(summary, "track"),
        "modelling": vector(summary, "model"),
        "scale": vector(summary, "scale"),
        "total": vector(summary, "total"),
    }
    for key, values in arrays.items():
        if key == "edges_mev":
            continue
        require((values >= 0.0).all(), f"{key} contains negative values")
        require(values[-1] == values[-2], f"{key} lacks the expected plot-closing duplicate")

    metrics = payload.get("metrics")
    require(isinstance(metrics, list), "metrics JSON has no metrics record list")
    nominal_records = [
        row
        for row in metrics
        if isinstance(row, dict)
        and row.get("dataset") == "test_nominal"
        and finite_float(row.get("working_point"), "nominal working point") == 0.5
    ]
    require(len(nominal_records) == 1, "expected exactly one nominal 50% working-point record")
    nominal_inclusive = finite_float(
        nominal_records[0].get("br"), "nominal inclusive background rejection"
    )
    require(
        math.isclose(nominal_inclusive, finite_float(summary.get("nominal_br"), "summary nominal BR"), rel_tol=0.0, abs_tol=1e-14),
        "nominal inclusive background rejection disagrees across metrics fields",
    )
    return arrays


def make_figure1(arrays: dict[str, np.ndarray]) -> None:
    outputs = FIGURE_OUTPUTS["figure1"]
    edges_tev = arrays["edges_mev"] / 1_000_000.0
    # The stored result arrays have 14 physical values plus one closing duplicate.
    physical = slice(0, 14)
    centers_tev = 0.5 * (edges_tev[:-1] + edges_tev[1:])
    require(centers_tev.shape == (14,), "Figure 1 does not have exactly 14 physical bins")

    rows: list[dict[str, Any]] = []
    for index in range(14):
        rows.append(
            {
                "physical_bin_index": index,
                "pt_low_GeV": format(arrays["edges_mev"][index] / 1000.0, ".17g"),
                "pt_high_GeV": format(arrays["edges_mev"][index + 1] / 1000.0, ".17g"),
                "pt_center_TeV": format(centers_tev[index], ".17g"),
                "nominal_background_rejection_wp50": format(arrays["nominal_br"][index], ".17g"),
                "cluster_uncertainty_percent": format(100.0 * arrays["cluster"][index], ".17g"),
                "track_uncertainty_percent": format(100.0 * arrays["track"][index], ".17g"),
                "modelling_uncertainty_percent": format(100.0 * arrays["modelling"][index], ".17g"),
                "isr_fsr_uncertainty_percent": format(100.0 * arrays["scale"][index], ".17g"),
                "total_uncertainty_percent": format(100.0 * arrays["total"][index], ".17g"),
            }
        )
    write_csv_new(
        OUTPUT_DIR / outputs["data_csv"],
        list(rows[0].keys()),
        rows,
    )

    figure, axes = plt.subplots(1, 2, figsize=(7.35, 3.15), sharex=True)
    axis = axes[0]
    axis.stairs(arrays["nominal_br"][physical], edges_tev, color="#202020", linewidth=1.8)
    axis.plot(centers_tev, arrays["nominal_br"][physical], linestyle="none", marker="o", markersize=3.0, color="#202020")
    axis.set_xlabel(r"Jet $p_{\mathrm{T}}$ [TeV]")
    axis.set_ylabel(r"Background rejection $1/\epsilon_{\mathrm{bkg}}$")
    axis.set_xlim(edges_tev[0], edges_tev[-1])
    axis.set_ylim(bottom=0.0)
    axis.minorticks_on()
    panel_label(axis, "(a)")

    axis = axes[1]
    styles = (
        ("cluster", "Cluster", "#0072B2", "-"),
        ("track", "Track", "#E69F00", "--"),
        ("modelling", "Modelling", "#009E73", "-."),
        ("scale", "ISR/FSR", "#CC79A7", ":"),
        ("total", "Total", "#202020", "-"),
    )
    for key, label, color, linestyle in styles:
        linewidth = 2.1 if key == "total" else 1.35
        axis.stairs(
            100.0 * arrays[key][physical],
            edges_tev,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
        )
    axis.set_xlabel(r"Jet $p_{\mathrm{T}}$ [TeV]")
    axis.set_ylabel("Relative uncertainty [%]")
    axis.set_xlim(edges_tev[0], edges_tev[-1])
    axis.set_ylim(bottom=0.0)
    axis.minorticks_on()
    axis.legend(loc="upper right", ncol=1, handlelength=2.7)
    panel_label(axis, "(b)")
    figure.suptitle("50% signal efficiency", y=1.01)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96), w_pad=2.0)
    save_figure_new(
        figure,
        OUTPUT_DIR / outputs["pdf"],
        OUTPUT_DIR / outputs["png"],
        "ATLAS80030 nominal performance and systematic uncertainties at 50% signal efficiency",
    )


def load_shap_rows() -> tuple[np.ndarray, dict[tuple[str, str], tuple[int, float]]]:
    status = load_json(SHAP_STATUS_JSON)
    require(status.get("status") == "complete", "SHAP round3 campaign is not complete")
    require(status.get("round") == 3, "SHAP status is not round3")
    require(status.get("remediation_only") is True, "SHAP round3 is not marked remediation-only")
    require(status.get("tree_shap_recomputed") is False, "SHAP round3 unexpectedly recomputed TreeSHAP")
    require(status.get("crosscheck") == "PASS", "SHAP round3 cross-check did not pass")
    require(status.get("model_sha256") == MODEL_SHA256, "SHAP round3 model SHA mismatch")
    require(SHAP_CSV.is_file(), f"missing frozen SHAP table: {SHAP_CSV}")

    with SHAP_CSV.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or [])
    required_fields = {
        "systematic",
        "sample_size",
        "subset",
        "feature",
        "n_pairs",
        "mean_abs_delta_shap",
        "mean_signed_delta_score",
    }
    require(required_fields <= fields, f"SHAP table lacks fields: {sorted(required_fields - fields)}")
    require(rows, "SHAP table is empty")
    require({row["systematic"] for row in rows} == set(SHAP_SYSTEMATICS), "SHAP systematic identities mismatch")
    require({row["feature"] for row in rows} == set(FEATURES), "SHAP feature identities mismatch")

    selected = [row for row in rows if row["sample_size"] == "100000"]
    require(len(selected) == 4 * 3 * 15, "SHAP table does not contain exactly 180 round3 100k rows")
    expected_keys = {
        (systematic, subset, feature)
        for systematic in SHAP_SYSTEMATICS
        for subset in SUBSETS
        for feature in FEATURES
    }
    found_keys = {(row["systematic"], row["subset"], row["feature"]) for row in selected}
    require(found_keys == expected_keys, "SHAP 100k systematic/subset/feature grid is incomplete")
    require(len(found_keys) == len(selected), "SHAP 100k table contains duplicate grid rows")

    matrix = np.empty((len(FEATURES), len(SHAP_SYSTEMATICS)), dtype=np.float64)
    score_summary: dict[tuple[str, str], tuple[int, float]] = {}
    for systematic_index, systematic in enumerate(SHAP_SYSTEMATICS):
        for feature_index, feature in enumerate(FEATURES):
            match = [
                row
                for row in selected
                if row["systematic"] == systematic
                and row["subset"] == "all"
                and row["feature"] == feature
            ]
            require(len(match) == 1, f"SHAP panel-A row count mismatch for {systematic}/{feature}")
            value = finite_float(match[0]["mean_abs_delta_shap"], f"mean |Delta SHAP| {systematic}/{feature}")
            require(value >= 0.0, f"negative mean |Delta SHAP| for {systematic}/{feature}")
            matrix[feature_index, systematic_index] = value

        for subset in SUBSETS:
            group = [
                row
                for row in selected
                if row["systematic"] == systematic and row["subset"] == subset
            ]
            require(len(group) == 15, f"SHAP score group has {len(group)} rows for {systematic}/{subset}")
            score_strings = {row["mean_signed_delta_score"] for row in group}
            pair_strings = {row["n_pairs"] for row in group}
            require(len(score_strings) == 1, f"SHAP score is inconsistent across features for {systematic}/{subset}")
            require(len(pair_strings) == 1, f"SHAP pair count is inconsistent across features for {systematic}/{subset}")
            n_pairs = exact_int(next(iter(pair_strings)), f"SHAP n_pairs {systematic}/{subset}")
            score = finite_float(next(iter(score_strings)), f"mean signed Delta score {systematic}/{subset}")
            require(n_pairs > 0, f"zero SHAP pairs for {systematic}/{subset}")
            score_summary[(systematic, subset)] = (n_pairs, score)

    require(np.isfinite(matrix).all(), "SHAP panel-A matrix is nonfinite")
    return matrix, score_summary


def make_figure2(matrix: np.ndarray, score_summary: dict[tuple[str, str], tuple[int, float]]) -> None:
    outputs = FIGURE_OUTPUTS["figure2"]
    rows: list[dict[str, Any]] = []
    for systematic_index, systematic in enumerate(SHAP_SYSTEMATICS):
        for feature_index, feature in enumerate(FEATURES):
            rows.append(
                {
                    "panel": "a",
                    "quantity": "mean_abs_delta_shap_margin_space",
                    "systematic": systematic,
                    "subset": "all",
                    "feature": feature,
                    "sample_size": 100000,
                    "n_pairs": score_summary[(systematic, "all")][0],
                    "value": format(matrix[feature_index, systematic_index], ".17g"),
                }
            )
    for systematic in SHAP_SYSTEMATICS:
        for subset in SUBSETS:
            n_pairs, score = score_summary[(systematic, subset)]
            rows.append(
                {
                    "panel": "b",
                    "quantity": "mean_signed_delta_score_probability_space",
                    "systematic": systematic,
                    "subset": subset,
                    "feature": "",
                    "sample_size": 100000,
                    "n_pairs": n_pairs,
                    "value": format(score, ".17g"),
                }
            )
    write_csv_new(OUTPUT_DIR / outputs["data_csv"], list(rows[0].keys()), rows)

    figure = plt.figure(figsize=(7.8, 5.25))
    grid = figure.add_gridspec(1, 2, width_ratios=(1.15, 1.0), wspace=0.48)
    axis_a = figure.add_subplot(grid[0, 0])
    image = axis_a.imshow(matrix, aspect="auto", origin="upper", cmap="viridis", vmin=0.0)
    axis_a.set_xticks(np.arange(len(SHAP_SYSTEMATICS)), labels=SHAP_SYSTEMATIC_LABELS, rotation=25, ha="right")
    axis_a.set_yticks(np.arange(len(FEATURES)), labels=FEATURE_LABELS)
    axis_a.set_title(r"Mean $|\Delta\mathrm{SHAP}|$ [margin], all jets", pad=7.0)
    axis_a.tick_params(which="both", top=False, right=False)
    colorbar = figure.colorbar(image, ax=axis_a, pad=0.025, fraction=0.055)
    colorbar.ax.tick_params(direction="in")
    panel_label(axis_a, "(a)")

    axis_b = figure.add_subplot(grid[0, 1])
    x = np.arange(len(SHAP_SYSTEMATICS), dtype=np.float64)
    width = 0.24
    subset_styles = (
        ("all", "All", "#4D4D4D"),
        ("signal", "Signal", "#0072B2"),
        ("background", "Background", "#D55E00"),
    )
    for offset_index, (subset, label, color) in enumerate(subset_styles):
        values = np.array(
            [score_summary[(systematic, subset)][1] for systematic in SHAP_SYSTEMATICS],
            dtype=np.float64,
        )
        offset = (offset_index - 1) * width
        axis_b.bar(x + offset, values, width=width, label=label, color=color, linewidth=0.0)
    axis_b.axhline(0.0, color="#202020", linewidth=0.8)
    axis_b.set_xticks(x, labels=SHAP_SYSTEMATIC_LABELS, rotation=25, ha="right")
    axis_b.set_ylabel("Mean signed $\\Delta$score\n(probability space)")
    axis_b.set_xlim(-0.55, len(SHAP_SYSTEMATICS) - 0.45)
    axis_b.legend(loc="lower right")
    axis_b.minorticks_on()
    panel_label(axis_b, "(b)")
    save_figure_new(
        figure,
        OUTPUT_DIR / outputs["pdf"],
        OUTPUT_DIR / outputs["png"],
        "ATLAS80030 paired SHAP-Delta results",
    )


def validate_conformal() -> tuple[list[dict[str, Any]], float]:
    payload = load_json(CONFORMAL_JSON)
    provenance = payload.get("provenance")
    calibration = payload.get("calibration")
    rows = payload.get("datasets")
    require(isinstance(provenance, dict), "conformal JSON has no provenance object")
    require(isinstance(calibration, dict), "conformal JSON has no calibration object")
    require(isinstance(rows, list), "conformal JSON has no dataset record list")
    require(provenance.get("model_sha256") == MODEL_SHA256, "conformal model SHA mismatch")
    require(provenance.get("training_or_refitting_performed") is False, "conformal provenance reports training/refitting")
    require(provenance.get("pairing_used") is False, "conformal provenance reports pairing")
    require(provenance.get("evaluated_datasets") == list(CONFORMAL_DATASETS), "conformal provenance dataset order mismatch")
    require(calibration.get("calibrated_once_on_nominal") is True, "conformal thresholds were not marked nominal-only")
    target = finite_float(calibration.get("target_coverage"), "conformal target coverage")
    require(target == 0.9, "conformal target coverage is not 90%")

    require(len(rows) == len(CONFORMAL_DATASETS), "conformal dataset record count mismatch")
    by_name: dict[str, dict[str, Any]] = {}
    coverage_fields = ("overall_coverage", "signal_coverage", "background_coverage")
    for row in rows:
        require(isinstance(row, dict), "conformal dataset row is not an object")
        dataset = row.get("dataset")
        require(isinstance(dataset, str), "conformal dataset identity is invalid")
        require(dataset not in by_name, f"duplicate conformal dataset: {dataset}")
        total = exact_int(row.get("total_count"), f"{dataset} total_count")
        signal = exact_int(row.get("signal_count"), f"{dataset} signal_count")
        background = exact_int(row.get("background_count"), f"{dataset} background_count")
        require(total > 0 and signal > 0 and background > 0, f"{dataset} has a zero class count")
        require(signal + background == total, f"{dataset} class counts do not sum to total")
        for field in coverage_fields:
            value = finite_float(row.get(field), f"{dataset} {field}")
            require(0.0 <= value <= 1.0, f"{dataset} {field} lies outside [0,1]")
        by_name[dataset] = row
    require(set(by_name) == set(CONFORMAL_DATASETS), "conformal dataset identities mismatch")
    ordered = [by_name[name] for name in CONFORMAL_DATASETS]

    nominal = ordered[0]
    for row in ordered:
        require(
            math.isclose(
                finite_float(row.get("overall_deviation_from_target"), f"{row['dataset']} target deviation"),
                finite_float(row["overall_coverage"], f"{row['dataset']} overall coverage") - target,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            f"{row['dataset']} stored target deviation fails accounting",
        )
        require(
            math.isclose(
                finite_float(row.get("overall_deviation_from_nominal"), f"{row['dataset']} nominal deviation"),
                finite_float(row["overall_coverage"], f"{row['dataset']} overall coverage")
                - finite_float(nominal["overall_coverage"], "nominal overall coverage"),
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            f"{row['dataset']} stored nominal deviation fails accounting",
        )
    return ordered, target


def make_figure3(rows_in: list[dict[str, Any]], target: float) -> None:
    outputs = FIGURE_OUTPUTS["figure3"]
    nominal_reference = finite_float(rows_in[0]["overall_coverage"], "nominal overall coverage")
    rows: list[dict[str, Any]] = []
    for source in rows_in:
        rows.append(
            {
                "dataset": source["dataset"],
                "total_count": source["total_count"],
                "signal_count": source["signal_count"],
                "background_count": source["background_count"],
                "overall_coverage_percent": format(100.0 * float(source["overall_coverage"]), ".17g"),
                "signal_coverage_percent": format(100.0 * float(source["signal_coverage"]), ".17g"),
                "background_coverage_percent": format(100.0 * float(source["background_coverage"]), ".17g"),
                "target_coverage_percent": format(100.0 * target, ".17g"),
                "nominal_overall_reference_percent": format(100.0 * nominal_reference, ".17g"),
            }
        )
    write_csv_new(OUTPUT_DIR / outputs["data_csv"], list(rows[0].keys()), rows)

    x = np.arange(len(CONFORMAL_DATASETS), dtype=np.float64)
    coverages = {
        "Overall": np.array([float(row["overall_coverage"]) for row in rows_in]) * 100.0,
        "Signal": np.array([float(row["signal_coverage"]) for row in rows_in]) * 100.0,
        "Background": np.array([float(row["background_coverage"]) for row in rows_in]) * 100.0,
    }
    require(all(values.shape == (10,) for values in coverages.values()), "Figure 3 coverage vectors are not length 10")
    require(all(np.isfinite(values).all() for values in coverages.values()), "Figure 3 coverage is nonfinite")

    figure, axis = plt.subplots(figsize=(7.45, 3.65))
    styles = (
        ("Overall", "#202020", "o", -0.18),
        ("Signal", "#0072B2", "s", 0.0),
        ("Background", "#D55E00", "^", 0.18),
    )
    for label, color, marker, offset in styles:
        axis.scatter(
            x + offset,
            coverages[label],
            label=label,
            color=color,
            marker=marker,
            s=22.0,
            linewidths=0.0,
        )
    axis.axhline(100.0 * target, color="#555555", linestyle="--", linewidth=1.0, label="90% target")
    axis.axhline(
        100.0 * nominal_reference,
        color="#777777",
        linestyle=":",
        linewidth=1.2,
        label="Nominal overall",
    )
    display_labels = (
        "test\nnominal",
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
    axis.set_xticks(x, labels=display_labels)
    axis.set_xlim(-0.35, len(CONFORMAL_DATASETS) - 0.65)
    all_values = np.concatenate(list(coverages.values()) + [np.array([100.0 * target, 100.0 * nominal_reference])])
    axis.set_ylim(float(all_values.min() - 0.15), float(all_values.max() + 0.15))
    axis.set_ylabel("Empirical coverage [%]")
    axis.set_xlabel("Evaluation dataset")
    axis.minorticks_on()
    axis.legend(loc="lower right", ncol=2, handlelength=2.5, columnspacing=1.2)
    figure.tight_layout()
    save_figure_new(
        figure,
        OUTPUT_DIR / outputs["pdf"],
        OUTPUT_DIR / outputs["png"],
        "ATLAS80030 empirical conformal coverage under detector variations",
    )


def main() -> None:
    require(not OUTPUT_DIR.exists(), f"output directory already exists; refusing overwrite: {OUTPUT_DIR}")
    source_paths = {
        "frozen_final_metrics": METRICS_JSON,
        "frozen_shap_round3_per_feature": SHAP_CSV,
        "frozen_shap_round3_status": SHAP_STATUS_JSON,
        "frozen_conformal_round1_results": CONFORMAL_JSON,
    }
    for role, path in source_paths.items():
        require(path.is_file(), f"missing {role}: {path}")
    source_hashes = {role: sha256(path) for role, path in source_paths.items()}

    metrics = validate_metrics()
    shap_matrix, score_summary = load_shap_rows()
    conformal_rows, target = validate_conformal()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    configure_matplotlib()
    make_figure1(metrics)
    make_figure2(shap_matrix, score_summary)
    make_figure3(conformal_rows, target)

    for role, path in source_paths.items():
        require(sha256(path) == source_hashes[role], f"source changed while rendering: {path}")

    output_records: dict[str, Any] = {}
    for figure_name, names in FIGURE_OUTPUTS.items():
        paths = {kind: OUTPUT_DIR / filename for kind, filename in names.items()}
        for kind, path in paths.items():
            require(path.is_file() and path.stat().st_size > 0, f"missing/empty {figure_name} {kind}: {path}")
        output_records[figure_name] = {
            "filenames": names,
            "sha256": {kind: sha256(path) for kind, path in paths.items()},
        }

    manifest = {
        "creation_time_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "frozen_model_sha256": MODEL_SHA256,
        "render_only": True,
        "reconstruction_run": False,
        "training_or_inference_run": False,
        "metrics_run": False,
        "shap_run": False,
        "conformal_run": False,
        "sources": {
            role: {"path": str(path.resolve()), "sha256": source_hashes[role]}
            for role, path in source_paths.items()
        },
        "outputs": output_records,
        "manifest_filename": "figure_manifest.json",
    }
    manifest_path = OUTPUT_DIR / "figure_manifest.json"
    require(not manifest_path.exists(), f"refusing to overwrite output: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    require(manifest_path.stat().st_size > 0, "empty figure manifest")

    print("VALIDATION PASSED")
    for figure_name, names in FIGURE_OUTPUTS.items():
        for filename in names.values():
            print(str(OUTPUT_DIR / filename))
    print(str(manifest_path))


if __name__ == "__main__":
    main()
