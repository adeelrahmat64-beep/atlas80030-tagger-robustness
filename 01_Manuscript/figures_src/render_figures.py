"""Re-render Figures 1-4 for the manuscript from the FROZEN figure data CSVs.

Rendering only.  The inputs are the frozen per-figure data files written by
`make_publication_figures.py`; every plotted number is read from them and is
re-verified against them at the end of this script.  No analysis is run and no
value is recomputed.

Cosmetic revisions applied in this version, in response to the external review:

  Fig 1 (ROC)        a lower panel gives the ratio of each systematic curve to
                     the nominal one, so the three curves are distinguishable
                     where they overlap on the logarithmic axis; colours and
                     dash patterns are colour-blind safe and mutually distinct.
  Fig 2 (perf/syst)  panel labels sit above the frame and the 50% working point
                     is stated on both axis labels.  October 2026: statistical
                     bands (+-1 s.d. of 100 bootstrap replicas) are drawn on
                     every curve of panel (b) and as error bars in panel (a),
                     read from stat_bootstrap_round1/stat_bootstrap_summary.json
                     after checking that its central values equal the frozen CSV.
  Fig 3 (SHAP)       every heat-map cell now carries its numerical value, and
                     the colour bar is labelled; column labels use the dataset
                     names CPOS and CER.
  Fig 4 (conformal)  the horizontal axis is categorical, so the connecting
                     lines are removed entirely; markers are drawn alone
                     against light vertical guides.

Set ATLAS80030_FIG_SRC / ATLAS80030_METRICS to relocate the frozen inputs.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
PROJECT = HERE.parent.parent            # the ATLAS 80030 project root
LOCAL = HERE.parent / "data"            # self-contained copy beside the manuscript


def _first(*candidates: Path) -> Path:
    """The first candidate that exists; the last one otherwise."""
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return Path(candidates[-1])


SRC = Path(os.environ["ATLAS80030_FIG_SRC"]) if "ATLAS80030_FIG_SRC" in os.environ \
    else _first(PROJECT / "04_Results" / "publication_figures_round1",
                PROJECT / "Code files" / "analysis_results"
                / "publication_figures_round1",
                LOCAL / "figs")
METRICS = Path(os.environ["ATLAS80030_METRICS"]) if "ATLAS80030_METRICS" in os.environ \
    else _first(PROJECT / "ATLAS_80030_Data" / "predictions"
                / "final_metrics_atlas_tagger_xgb_final_production_round1_round1.json",
                LOCAL
                / "final_metrics_atlas_tagger_xgb_final_production_round1_round1.json")

FEATURE_LABELS = (
    r"$\tau_1^{\mathrm{WTA}}$", r"$\tau_2^{\mathrm{WTA}}$",
    r"$\tau_3^{\mathrm{WTA}}$", r"$\tau_4^{\mathrm{WTA}}$",
    r"$\sqrt{d_{12}}$", r"$\sqrt{d_{23}}$",
    r"ECF$_1$", r"ECF$_2$", r"ECF$_3$", r"$C_2$", r"$D_2$", r"$L_2$", r"$L_3$",
    r"$Q_w$", "ThrustMaj",
)
FEATURES = (
    "fjet_Tau1_wta", "fjet_Tau2_wta", "fjet_Tau3_wta", "fjet_Tau4_wta",
    "fjet_Split12", "fjet_Split23", "fjet_ECF1", "fjet_ECF2", "fjet_ECF3",
    "fjet_C2", "fjet_D2", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_ThrustMaj",
)
SYSTEMATICS = ("esup", "esdown", "cpos", "cer")
SYSTEMATIC_LABELS = ("ES up", "ES down", "CPOS", "CER")
BOOTSTRAP = Path(os.environ["ATLAS80030_BOOTSTRAP"]) if "ATLAS80030_BOOTSTRAP" in os.environ \
    else _first(PROJECT / "04_Results" / "stat_bootstrap_round1" / "stat_bootstrap_summary.json",
                LOCAL / "stat_bootstrap_summary.json")
# Figure 2(b) curve -> key of the same quantity in the bootstrap summary
BOOT_KEY = {"cluster": "cluster", "track": "track", "modelling": "model",
            "isr_fsr": "scale", "total": "total"}
CONFORMAL_ORDER = ("test_nominal", "esup", "esdown", "cer", "cpos",
                   "teg", "tej", "tfl", "tfj", "bias")


def rc() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.0,
        "axes.labelsize": 9.5, "axes.titlesize": 9.5, "axes.linewidth": 0.8,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.2, "legend.frameon": False,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "lines.linewidth": 1.5,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def load(name: str) -> list[dict[str, str]]:
    with (SRC / name).open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def outside_label(axis, text: str, x: float = 0.0) -> None:
    """Panel label above the frame, never over the data."""
    axis.text(x, 1.015, text, transform=axis.transAxes,
              ha="left", va="bottom", fontweight="bold")


# --------------------------------------------------------------- figure 1 ---
def figure_roc(rows: list[dict[str, str]]) -> None:
    """ROC of the frozen predictions; working points from the metrics artifact.

    The two energy-scale curves sit within a few per cent of the nominal one on
    a logarithmic axis, where they are hard to tell apart.  The lower panel
    plots their ratio to nominal on a linear axis, which separates them
    without introducing any quantity that is not already in the frozen file.
    """
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    M = {(m["dataset"], m["working_point"]): m for m in metrics["metrics"]}

    series = (("test_nominal", "Nominal", "#202020", "-", 1.7),
              ("esup", "ES up", "#0072B2", (0, (5, 1.6)), 1.4),
              ("esdown", "ES down", "#D55E00", (0, (1.4, 1.4)), 1.5))

    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(3.55, 4.25), sharex=True,
        gridspec_kw={"height_ratios": (2.55, 1.0), "hspace": 0.07})

    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, label, colour, ls, lw in series:
        sub = [r for r in rows if r["dataset"] == name]
        eff = np.array([float(r["signal_efficiency"]) for r in sub])
        bkg = np.array([float(r["background_efficiency"]) for r in sub])
        auc = float(sub[0]["auc"])
        # the AUC in the file must equal the frozen metrics AUC
        assert auc == M[(name, 0.5)]["auc"], name
        good = (bkg > 0.0) & (eff > 0.0)
        curves[name] = (eff[good], 1.0 / bkg[good])
        ax.plot(eff[good], 1.0 / bkg[good], label=f"{label} (AUC {auc:.4f})",
                color=colour, linestyle=ls, linewidth=lw)

    # working-point markers: frozen values, not read off the curve
    for wp, marker in ((0.5, "o"), (0.8, "s")):
        br = M[("test_nominal", wp)]["br"]
        ax.plot([wp], [br], marker=marker, markersize=5.5, color="#202020",
                markerfacecolor="white", markeredgewidth=1.3, linestyle="none",
                zorder=5)
        ax.annotate(f"{wp:.0%}: {br:.2f}",
                    xy=(wp, br), xytext=(-8, -11), textcoords="offset points",
                    ha="right", va="top", fontsize=7.8)
    ax.set_yscale("log")
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(2.0, 400.0)
    ax.set_ylabel(r"Background rejection $1/\epsilon_{\mathrm{bkg}}$")
    ax.legend(loc="upper right")
    ax.minorticks_on()

    # lower panel: ratio to nominal on a common efficiency grid
    grid = np.linspace(0.2, 1.0, 401)
    nom_e, nom_r = curves["test_nominal"]
    nom_i = np.interp(grid, nom_e, nom_r)
    for name, label, colour, ls, lw in series[1:]:
        e, r = curves[name]
        rax.plot(grid, np.interp(grid, e, r) / nom_i, color=colour,
                 linestyle=ls, linewidth=lw, label=label)
    rax.axhline(1.0, color="#202020", linewidth=0.8)
    rax.set_ylim(0.80, 1.06)
    rax.set_xlabel(r"Signal efficiency $\epsilon_{\mathrm{sig}}$")
    rax.set_ylabel("Ratio to\nnominal", fontsize=8.6)
    rax.minorticks_on()

    fig.savefig(OUT / "figure1_roc.pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 2 ---
def figure_performance(rows: list[dict[str, str]]) -> None:
    lo = np.array([float(r["pt_low_GeV"]) for r in rows]) / 1000.0
    hi = np.array([float(r["pt_high_GeV"]) for r in rows]) / 1000.0
    edges = np.append(lo, hi[-1])
    ctr = np.array([float(r["pt_center_TeV"]) for r in rows])
    br = np.array([float(r["nominal_background_rejection_wp50"]) for r in rows])
    comp = {k: np.array([float(r[f"{k}_uncertainty_percent"]) for r in rows])
            for k in ("cluster", "track", "modelling", "isr_fsr")}
    comp["total"] = np.array([float(r["total_uncertainty_percent"]) for r in rows])

    # Statistical bands (October 2026 extension): +-1 standard deviation of the
    # 100 Poisson bootstrap replicas, drawn around the frozen measured values.
    # The bootstrap re-runs the frozen chain; its central values must reproduce
    # the frozen figure CSV before any band is drawn.
    boot = json.loads(BOOTSTRAP.read_text(encoding="utf-8"))
    assert boot["replicas"] == 100
    band = {}
    for key, bkey in BOOT_KEY.items():
        central = 100.0 * np.array(boot["central"][bkey])
        assert np.allclose(central, comp[key], rtol=0.0, atol=1e-9), key
        band[key] = 100.0 * np.array(boot["bootstrap"][bkey]["std"])
    assert np.allclose(boot["central"]["nominal_binned_br"], br, rtol=0.0, atol=1e-9)
    br_sd = np.array(boot["bootstrap"]["nominal_binned_br"]["std"])

    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.30), sharex=True)

    ax = axes[0]
    ax.stairs(br, edges, color="#202020", linewidth=1.8)
    ax.errorbar(ctr, br, yerr=br_sd, linestyle="none", marker="o", markersize=3.0,
                color="#202020", elinewidth=1.0, capsize=1.8)
    ax.set_xlabel(r"Jet $p_{\mathrm{T}}$ [TeV]")
    ax.set_ylabel(r"Background rejection $1/\epsilon_{\mathrm{bkg}}$"
                  "\n" r"at $\epsilon_{\mathrm{sig}}=50\%$")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0.0, float(br.max()) * 1.14)
    ax.minorticks_on()
    outside_label(ax, "(a)")

    ax = axes[1]
    styles = (("cluster", "Cluster", "#0072B2", "-"),
              ("track", "Track", "#E69F00", "--"),
              ("modelling", "Modelling", "#009E73", "-."),
              ("isr_fsr", "ISR/FSR", "#CC79A7", ":"),
              ("total", "Total", "#202020", "-"))
    for key, label, colour, ls in styles:
        ax.stairs(comp[key] + band[key], edges, baseline=comp[key] - band[key],
                  fill=True, color=colour, alpha=0.20 if key == "total" else 0.14,
                  linewidth=0.0)
        ax.stairs(comp[key], edges, label=label, color=colour, linestyle=ls,
                  linewidth=2.1 if key == "total" else 1.35)
    ax.set_xlabel(r"Jet $p_{\mathrm{T}}$ [TeV]")
    ax.set_ylabel("Relative uncertainty [%]"
                  + "\n" r"at $\epsilon_{\mathrm{sig}}=50\%$")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0.0, float(comp["total"].max()) * 1.30)
    ax.minorticks_on()
    ax.legend(loc="upper right", ncol=2, handlelength=2.4, columnspacing=1.1)
    outside_label(ax, "(b)")

    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "figure2_performance_systematics.pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 3 ---
def figure_shap(rows: list[dict[str, str]]) -> None:
    heat = {(r["systematic"], r["feature"]): float(r["value"]) for r in rows
            if r["quantity"] == "mean_abs_delta_shap_margin_space"}
    matrix = np.array([[heat[(s, f)] for s in SYSTEMATICS] for f in FEATURES])
    score = {(r["systematic"], r["subset"]): float(r["value"]) for r in rows
             if r["quantity"] == "mean_signed_delta_score_probability_space"}

    fig = plt.figure(figsize=(7.4, 5.05))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 0.92), wspace=0.60)

    ax = fig.add_subplot(grid[0, 0])
    im = ax.imshow(matrix, aspect="auto", origin="upper", cmap="viridis", vmin=0.0)
    ax.set_xticks(np.arange(len(SYSTEMATICS)), labels=SYSTEMATIC_LABELS,
                  rotation=25, ha="right")
    ax.set_yticks(np.arange(len(FEATURES)), labels=FEATURE_LABELS)
    ax.set_title(r"Mean $|\Delta\mathrm{SHAP}|$ [margin], all jets", pad=6.0)
    # numerical annotation: every cell carries its own value
    span = float(matrix.max())
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.4,
                    color="white" if v < 0.55 * span else "#101010")
    bar = fig.colorbar(im, ax=ax, pad=0.030, fraction=0.055)
    bar.ax.tick_params(labelsize=8.0)
    bar.set_label(r"Mean $|\Delta\mathrm{SHAP}|$ [margin]", fontsize=8.2)
    ax.text(-0.34, 1.045, "(a)", transform=ax.transAxes, ha="left", va="bottom",
            fontweight="bold")

    ax = fig.add_subplot(grid[0, 1])
    x = np.arange(len(SYSTEMATICS))
    width = 0.26
    for offset, (subset, label, colour) in zip(
            (-width, 0.0, width),
            (("all", "All", "#4D4D4D"), ("signal", "Signal", "#0072B2"),
             ("background", "Background", "#D55E00"))):
        ax.bar(x + offset, [score[(s, subset)] for s in SYSTEMATICS],
               width=width, label=label, color=colour, linewidth=0.0)
    ax.axhline(0.0, color="#202020", linewidth=0.8)
    ax.set_xticks(x, labels=SYSTEMATIC_LABELS, rotation=25, ha="right")
    ax.set_ylabel(r"Mean signed $\Delta$score [probability]", labelpad=2.0)
    ax.minorticks_on()
    ax.legend(loc="lower right")
    ax.text(-0.22, 1.045, "(b)", transform=ax.transAxes, ha="left", va="bottom",
            fontweight="bold")

    fig.savefig(OUT / "figure3_shap_delta.pdf", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure 4 ---
def figure_conformal(rows: list[dict[str, str]]) -> None:
    order = [r["dataset"] for r in rows]
    assert order == list(CONFORMAL_ORDER), order
    cov = {k: np.array([float(r[f"{k}_coverage_percent"]) for r in rows])
           for k in ("overall", "signal", "background")}
    target = float(rows[0]["target_coverage_percent"])
    nominal = float(rows[0]["nominal_overall_reference_percent"])
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(7.45, 3.65))
    # The horizontal axis is categorical: markers only, no connecting lines.
    for xi in x:
        ax.axvline(xi, color="#DDDDDD", linewidth=0.7, zorder=0)
    styles = (("signal", "Signal", "#0072B2", "s", 5.4),
              ("background", "Background", "#D55E00", "^", 5.4),
              ("overall", "Overall", "#202020", "o", 4.4))
    for key, label, colour, marker, size in styles:
        ax.plot(x, cov[key], label=label, color=colour, marker=marker,
                markersize=size, linestyle="none",
                markerfacecolor="none" if key == "overall" else colour,
                markeredgewidth=1.2, zorder=3)
    ax.axhline(target, color="#555555", linestyle="--", linewidth=1.0,
               label="90% target", zorder=1)
    ax.axhline(nominal, color="#777777", linestyle=":", linewidth=1.2,
               label="Nominal overall", zorder=1)
    ax.set_xticks(x, labels=("test\nnominal", "esup", "esdown", "cer", "cpos",
                             "teg", "tej", "tfl", "tfj", "bias"))
    ax.set_xlim(-0.5, len(order) - 0.5)
    allv = np.concatenate(list(cov.values()) + [np.array([target, nominal])])
    ax.set_ylim(float(allv.min() - 0.15), float(allv.max() + 0.40))
    ax.set_ylabel("Empirical coverage [%]")
    ax.set_xlabel("Evaluation dataset")
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.legend(loc="lower right", ncol=2, handlelength=2.5, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(OUT / "figure4_conformal_coverage.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    rc()
    OUT.mkdir(exist_ok=True)
    roc = load("roc_curve_data.csv")
    f1 = load("figure1_performance_systematics_data.csv")
    f2 = load("figure2_shap_delta_data.csv")
    f3 = load("figure3_conformal_coverage_data.csv")
    assert len(roc) == 6000 and len(f1) == 14 and len(f2) == 72 and len(f3) == 10
    figure_roc(roc)
    figure_performance(f1)
    figure_shap(f2)
    figure_conformal(f3)
    print("rendered 4 figures from the frozen data CSVs "
          f"({len(roc)}, {len(f1)}, {len(f2)}, {len(f3)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
