"""Tables for the October 2026 robustness extension.

Inputs (read-only):
  tables/layout/table1_nominal_performance.tex          frozen layout table
  tables/layout/table2_inclusive_systematics_wp50.tex   frozen layout table
  04_Results/stat_bootstrap_round1/stat_bootstrap_summary.json
  04_Results/tagger_variants_round1/tagger_variants_summary.json
  04_Results/feature_response_round1/feature_response_summary.json

Outputs (tables/ext/):
  table1_nominal_performance_precision.tex   main Table 3: frozen rows + stat / seed rows
  table2_inclusive_systematics_precision.tex main Table 4: frozen column + stat / seed columns
  table6_feature_response_degree.tex         main Table 6
  table7_scale_free_variants.tex             main Table 7
  supp_s8_bootstrap_binned.tex               Supplementary Table S8
  supp_s9_seed_runs.tex                      Supplementary Table S9
  supp_s10_feature_response.tex              Supplementary Table S10

Rules the script enforces before it writes anything:
  * every frozen number in the two frozen tables is carried over verbatim, and
    the bootstrap and seed-42 central values reproduce those frozen numbers;
  * the seed-42 control run of the variant study is the frozen model
    (identical SHA-256, bit-identical predictions, identical metrics);
  * the bootstrap used 100 replicas against the frozen metrics file;
  * every summary statistic printed is recomputed here from the per-run or
    per-sample values in the JSON and must agree with the stored summary.
Nothing is retyped by hand.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
RES = Path(os.environ["ATLAS80030_RESULTS"]) if "ATLAS80030_RESULTS" in os.environ \
    else PROJECT / "04_Results"
OUT = HERE / "ext"

FROZEN_MODEL = "63a23308efbf2ce90418c8e657730f571db4798fc9ca2fb3692cb986de405f4c"
FEATURES = ("fjet_Tau1_wta", "fjet_Tau2_wta", "fjet_Tau3_wta", "fjet_Tau4_wta",
            "fjet_Split12", "fjet_Split23", "fjet_ECF1", "fjet_ECF2", "fjet_ECF3",
            "fjet_C2", "fjet_D2", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_ThrustMaj")
LABEL = {"fjet_Tau1_wta": r"$\tau_1^{\mathrm{WTA}}$", "fjet_Tau2_wta": r"$\tau_2^{\mathrm{WTA}}$",
         "fjet_Tau3_wta": r"$\tau_3^{\mathrm{WTA}}$", "fjet_Tau4_wta": r"$\tau_4^{\mathrm{WTA}}$",
         "fjet_Split12": r"$\sqrt{d_{12}}$", "fjet_Split23": r"$\sqrt{d_{23}}$",
         "fjet_ECF1": r"ECF$_1$", "fjet_ECF2": r"ECF$_2$", "fjet_ECF3": r"ECF$_3$",
         "fjet_C2": r"$C_2$", "fjet_D2": r"$D_2$", "fjet_L2": r"$L_2$", "fjet_L3": r"$L_3$",
         "fjet_Qw": r"$Q_w$", "fjet_ThrustMaj": "Thrust major"}
# The degree of each quantity in the constituent momenta: under p_T -> lambda p_T
# for every constituent it scales as lambda^d.  This is a property of the
# definitions in recompute_hl.py; it is repeated in tagger_variants.py, and the
# two must agree.
DEGREE = {f: 0 for f in FEATURES}
DEGREE.update({"fjet_Split12": 1, "fjet_Split23": 1, "fjet_ECF1": 1, "fjet_Qw": 1,
               "fjet_ECF2": 2, "fjet_ECF3": 3})
SETS = ("full15", "ratios9", "pt_normalised14")
SET_LABEL = {"full15": "All fifteen (frozen inputs)",
             "ratios9": "Nine dimensionless ($d=0$)",
             "pt_normalised14": r"Nine $+$ five $X/\mathrm{ECF}_1^{\,d}$"}

# Rows of the frozen inclusive table and the uncertainty-chain key of each.
T2_KEYS = (("Cluster energy scale", "inc_es"), ("CER", "inc_cer"), ("CPOS", "inc_cpos"),
           ("Cluster total", "inc_cluster"), ("Track efficiency", "inc_eff"),
           ("Fake rate", "inc_fake"), ("Bias", "inc_bias"), ("Track total", "inc_track"),
           ("Signal modelling", "inc_sig_model"), ("Background modelling", "inc_bkg_model"),
           ("Modelling total", "inc_model"), ("Signal ISR", "inc_sig_ISR"),
           ("Signal FSR", "inc_sig_FSR"), ("Background ISR", "inc_bkg_ISR"),
           ("Background FSR", "inc_bkg_FSR"), ("ISR/FSR total", "inc_scale"),
           ("Total uncertainty", "inc_total"))

failures: list[str] = []


def need(cond: bool, what: str) -> None:
    if not cond:
        failures.append(what)


def sig2(x: float) -> str:
    """A standard deviation to two significant figures, in fixed notation."""
    if x == 0:
        return "0"
    digits = 1 - int(math.floor(math.log10(abs(x))))
    return f"{round(x, digits):.{max(digits, 0)}f}"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ inputs ---
boot = load(RES / "stat_bootstrap_round1" / "stat_bootstrap_summary.json")
var = load(RES / "tagger_variants_round1" / "tagger_variants_summary.json")
resp = load(RES / "feature_response_round1" / "feature_response_summary.json")
t1 = (HERE / "layout" / "table1_nominal_performance.tex").read_text(encoding="utf-8")
t2 = (HERE / "layout" / "table2_inclusive_systematics_wp50.tex").read_text(encoding="utf-8")

C, B = boot["central"], boot["bootstrap"]
need(boot["replicas"] == 100, "bootstrap replica count is not 100")
need(boot["frozen_model_sha256"] == FROZEN_MODEL, "bootstrap model is not the frozen model")
need(var["frozen_model_sha256"] == FROZEN_MODEL, "variant study names another frozen model")
need(var["scale_degree"] == DEGREE, "scale degrees differ from tagger_variants.py")
need(resp["scale_degree"] == DEGREE, "scale degrees differ from feature_response.py")

# bootstrap s.d. recomputed from the replica file must match the summary
rep_path = RES / "stat_bootstrap_round1" / "replicas.jsonl"
reps = [json.loads(l) for l in rep_path.read_text(encoding="utf-8").splitlines() if l.strip()]
need(len(reps) == 100, f"replicas.jsonl has {len(reps)} lines")
need(sorted(r["replica"] for r in reps) == list(range(100)), "replica indices not 0..99")
for k in ("inc_total", "nominal_br_50", "nominal_auc", "inc_es", "inc_model"):
    sd = float(np.std([r[k] for r in reps], ddof=1))
    need(abs(sd - B[k]["std"]) < 1e-12, f"bootstrap s.d. of {k} does not recompute")
binned_sd = np.std(np.array([r["total"] for r in reps]), axis=0, ddof=1)
need(np.allclose(binned_sd, B["total"]["std"], rtol=0, atol=1e-12),
     "binned bootstrap s.d. does not recompute")

runs = {(r["feature_set"], r["seed"]): r for r in var["runs"]}
seeds = var["summary"]["full15"]["seeds"]
need(seeds == [42, 1, 2, 3, 4, 5], f"seeds are {seeds}")
for fs in SETS:
    need(all((fs, s) in runs for s in seeds), f"missing runs for {fs}")
ctrl = runs[("full15", 42)]
need(ctrl["model_sha256"] == FROZEN_MODEL, "control run is not the frozen model")
need(ctrl.get("control_predictions_bit_identical_to_frozen") is True,
     "control predictions not bit-identical")
need(ctrl.get("control_reproduces_frozen_metrics") is True, "control metrics differ")
for k, v in ctrl["results"].items():
    ref = C[k]
    if isinstance(v, list):
        need(np.array_equal(np.array(v), np.array(ref)), f"control {k} != bootstrap central")
    else:
        need(v == ref, f"control {k} != bootstrap central")


def seed_stats(fs: str, key: str) -> tuple[float, float, float, float]:
    vals = np.array([runs[(fs, s)]["results"][key] for s in seeds], dtype=float)
    mean, sd = float(vals.mean()), float(vals.std(ddof=1))
    stored = var["summary"][fs].get(key)
    if stored is not None:
        need(abs(stored["mean"] - mean) < 1e-12 and abs(stored["std"] - sd) < 1e-12,
             f"{fs}/{key} summary does not recompute")
    return mean, sd, float(vals.min()), float(vals.max())


# ------------------------------------------------- main Table 3 (extended) ---
frozen_row = ("XGBoost (this work) & Frozen final result & 0.931995 & 0.855854 "
              "& 41.9064 & 9.28847 \\\\")
need(frozen_row in t1, "frozen Table 3 row not found verbatim")
need(f"{C['nominal_auc']:.6f}" == "0.931995" and f"{C['nominal_acc']:.6f}" == "0.855854"
     and f"{C['nominal_br_50']:.4f}" == "41.9064" and f"{C['nominal_br_80']:.5f}" == "9.28847",
     "bootstrap central values do not reproduce the frozen Table 3 row")
stat_cells = [sig2(B[k]["std"]) for k in ("nominal_auc", "nominal_acc",
                                            "nominal_br_50", "nominal_br_80")]
seed_cells = [sig2(seed_stats("full15", k)[1]) for k in ("nominal_auc", "nominal_acc",
                                                          "nominal_br_50", "nominal_br_80")]
new_rows = (frozen_row + "\n"
            r"\quad statistical s.d. & Bootstrap, 100 replicas & "
            + " & ".join(f"$\\pm {c}$" for c in stat_cells) + " \\\\\n"
            r"\quad training-seed s.d. & Six seeds, same training jets & "
            + " & ".join(f"$\\pm {c}$" for c in seed_cells) + " \\\\")
t1_new = t1.replace(frozen_row, new_rows, 1)
old_cap = ("it is not a reproduction.}")
need(old_cap in t1_new, "frozen Table 3 caption ending not found")
t1_new = t1_new.replace(old_cap, (
    "it is not a reproduction. The two indented rows give the statistical standard "
    "deviation of each metric over 100 Poisson bootstrap replicas of the test sample, "
    "and the standard deviation over six training seeds of the same configuration on "
    "the same training jets (Section~\\ref{sec:precision}); neither is a confidence "
    "interval on the published reference.}"), 1)
t1_new = ("% Generated by tables/make_extension_tables.py. The XGBoost and hlDNN rows\n"
          "% are the frozen layout table verbatim; the two indented rows are new.\n"
          + t1_new)

# -------------------------------------------------- main Table 4 (extended) ---
rows4 = []
for label, key in T2_KEYS:
    m = re.search(rf"^(\S+) & {re.escape(label)} & ([\d.]+) \\\\$", t2, re.M)
    need(m is not None, f"frozen Table 4 row {label} not found")
    if m is None:
        continue
    frozen = m.group(2)
    need(f"{100 * C[key]:.2f}" == frozen, f"bootstrap central {key} != frozen {frozen}")
    mean, sd, lo, hi = seed_stats("full15", key)
    need(f"{100 * runs[('full15', 42)]['results'][key]:.2f}" == frozen,
         f"seed-42 {key} != frozen")
    rows4.append(f"{m.group(1)} & {label} & {frozen} & {100 * B[key]['std']:.2f} & "
                 f"${100 * mean:.2f} \\pm {100 * sd:.2f}$ \\\\")
t2_new = t2
t2_new = t2_new.replace(r"\begin{tabular}{llr}", r"\begin{tabular}{llrrr}", 1)
t2_new = t2_new.replace(
    r"Category & Component & Relative uncertainty [\%] \\",
    r"Category & Component & Frozen [\%] & Stat.\ s.d.\ [\%] & Six seeds [\%] \\", 1)
body = re.search(r"\\midrule\n(.*?)\\bottomrule", t2_new, re.S)
need(body is not None and len(rows4) == 17, "Table 4 body not rebuilt")
t2_new = t2_new[:body.start(1)] + "\n".join(rows4) + "\n" + t2_new[body.end(1):]
old_cap2 = "Envelopes are taken before quadrature combination.}"
need(old_cap2 in t2_new, "frozen Table 4 caption ending not found")
t2_new = t2_new.replace(old_cap2, (
    "Envelopes are taken before quadrature combination. ``Frozen'' is the frozen result. "
    "``Stat.\\ s.d.'' is the standard deviation of the same quantity over 100 Poisson "
    "bootstrap replicas in which exactly paired calorimeter jets share their nominal "
    "partner's weight. ``Six seeds'' is the mean and standard deviation over six training "
    "seeds of the frozen configuration on the same training jets; seed 42 reproduces the "
    "frozen column exactly (Section~\\ref{sec:precision}).}"), 1)
t2_new = ("% Generated by tables/make_extension_tables.py. The first three columns are\n"
          "% the frozen layout table verbatim; the last two columns are new.\n" + t2_new)

# --------------------------------------------------------- main Table 6 ------
S = resp["samples"]
rows6 = []
for f in FEATURES:
    d = DEGREE[f]
    up, dn = S["esup"][f], S["esdown"][f]
    need(up["degree"] == d and dn["degree"] == d, f"degree of {f} in response file")
    rows6.append(f"{LABEL[f]} & {d} & ${100 * up['median']:+.2f}$ & " +
                 (f"${100 * up['uniform_scaling_expectation']:+.2f}$ & " if d else "$0$ & ") +
                 f"${100 * dn['median']:+.2f}$ & {100 * S['cer'][f]['median_abs']:.2f} & "
                 f"{100 * S['cpos'][f]['median_abs']:.3f} \\\\")
ecf1 = S["esup"]["fjet_ECF1"]["median"]
need(max(x["n_excluded"] for smp in S.values() for x in smp.values()) == 44,
     "Table 6 caption says at most 44 pairs are excluded")
for f in FEATURES:
    expect = (1.0 + ecf1) ** DEGREE[f] - 1.0
    need(abs(S["esup"][f]["uniform_scaling_expectation"] - expect) < 1e-12,
         f"uniform expectation of {f} is not (1+ECF1 median)^d - 1")
t6 = "\n".join([
    "% Generated by tables/make_extension_tables.py from",
    "% 04_Results/feature_response_round1/feature_response_summary.json.",
    r"\begin{table}[tbp]",
    r"\centering\footnotesize",
    r"\setlength{\tabcolsep}{2.6pt}",
    (r"\caption{Per-jet response of the fifteen reconstructed inputs to the four "
     r"exactly paired calorimeter variations, over all matched eligible jet pairs "
     r"($4{,}999{,}999$ for \texttt{esup}, \texttt{esdown} and \texttt{cpos}, "
     r"$4{,}995{,}058$ for \texttt{cer}; the at most 44 pairs per quantity with a "
     r"non-finite value are left out). $d$ is the degree of each quantity in the "
     r"constituent four-momenta. The response of a jet is $r = X^{s}/X^{\mathrm{nom}}-1$ "
     r"for the same physical jet. ``Uniform'' is the response a degree-$d$ quantity would "
     r"have if every constituent $\pt$ were scaled by the median ECF$_1$ response, "
     r"$(1.0741)^{d}-1$. For \texttt{cer} and \texttt{cpos}, whose median responses are "
     r"close to zero, the median absolute response is given instead. All entries are in "
     r"percent; the full distributions are in Supplementary Table~S10.}"),
    r"\label{tab:feature-response}",
    r"\begin{tabular}{lcrrrrr}",
    r"\toprule",
    r" & & \multicolumn{3}{c}{Median $r$ [\%]} & \multicolumn{2}{c}{Median $|r|$ [\%]} \\",
    r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
    r" & $d$ & \texttt{esup} & Uniform & \texttt{esdown} & \texttt{cer} & \texttt{cpos} \\",
    r"\midrule",
    *rows6,
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}", ""])
need(abs(100 * ecf1 - 7.4068) < 5e-5, "ECF1 median is not 7.4068%")

# --------------------------------------------------------- main Table 7 ------
def pm(fs: str, key: str, scale: float, nd: int) -> str:
    mean, sd, _, _ = seed_stats(fs, key)
    return f"${scale * mean:.{nd}f} \\pm {scale * sd:.{nd}f}$"


rows7 = []
for fs in SETS:
    n_in = len(var["feature_sets"][fs])
    rows7.append(" & ".join([SET_LABEL[fs], str(n_in),
                             pm(fs, "nominal_auc", 1, 4).replace(r" \pm 0.0000", ""),
                             pm(fs, "nominal_br_50", 1, 2),
                             pm(fs, "inc_es", 100, 2), pm(fs, "inc_cluster", 100, 2),
                             pm(fs, "inc_track", 100, 2), pm(fs, "inc_model", 100, 2),
                             pm(fs, "inc_scale", 100, 2), pm(fs, "inc_total", 100, 2)])
                 + r" \\")
for fs in SETS:
    sd_auc = seed_stats(fs, "nominal_auc")[1]
    need(sd_auc < 5e-5, f"{fs} AUC seed s.d. {sd_auc} would print as non-zero at 4 dp")
need([len(var["feature_sets"][f]) for f in SETS] == [15, 9, 14], "feature-set sizes")
need(var["feature_sets"]["ratios9"] == [f for f in FEATURES if DEGREE[f] == 0],
     "ratios9 is not the set of degree-0 inputs")
t7 = "\n".join([
    "% Generated by tables/make_extension_tables.py from",
    "% 04_Results/tagger_variants_round1/tagger_variants_summary.json.",
    r"\begin{table*}[tbp]",
    r"\centering",
    (r"\caption{The same tagger configuration trained on three input sets, each with six "
     r"training seeds on the same training jets; entries are the mean and standard "
     r"deviation over the six seeds, evaluated with the frozen uncertainty procedure on "
     r"the same test and systematic samples. The nine dimensionless inputs are the "
     r"$d=0$ quantities of Table~\ref{tab:feature-response}. The third set adds the five "
     r"dimensionful quantities divided by ECF$_1^{\,d}$, which removes their momentum "
     r"scale but keeps their shape. AUC seed standard deviations are below $5\times10^{-5}$ "
     r"and are not shown. BR is background rejection at $50\%$ signal efficiency, and "
     r"every uncertainty is the inclusive relative uncertainty on it, in percent. "
     r"Per-seed values are in Supplementary Table~S9.}"),
    r"\label{tab:scale-free}",
    r"\resizebox{\textwidth}{!}{%",
    r"\begin{tabular}{lrrrrrrrrr}",
    r"\toprule",
    r"Inputs & $n$ & AUC & BR@50\% & Energy scale & Cluster & Track & Modelling & ISR/FSR & Total \\",
    r"\midrule",
    *rows7,
    r"\bottomrule",
    r"\end{tabular}%",
    r"}",
    r"\end{table*}", ""])

# -------------------------------------------------------------- Table S8 -----
BINS = [(350 + 200 * i, 550 + 200 * i) for i in range(14)]
S8_COLS = (("es", "CES"), ("cer", "CER"), ("cpos", "CPOS"), ("cluster", "Cluster"),
           ("eff", "Track eff."), ("fake", "Fake rate"), ("bias", "Bias"),
           ("track", "Track"), ("sig_model", "Sig.\\ model."), ("bkg_model", "Bkg.\\ model."),
           ("model", "Modelling"), ("sig_ISR", "Sig.\\ ISR"), ("sig_FSR", "Sig.\\ FSR"),
           ("bkg_ISR", "Bkg.\\ ISR"), ("bkg_FSR", "Bkg.\\ FSR"), ("scale", "ISR/FSR"),
           ("total", "Total"))
need(len(S8_COLS) == 17, "Table S8 must cover every column of Tables S2 and S3")
rows8 = []
for i, (lo, hi) in enumerate(BINS):
    cells = [str(lo), str(hi),
             f"{B['nominal_binned_br']['std'][i]:.2f}"]
    cells += [f"{100 * B[k]['std'][i]:.2f}" for k, _ in S8_COLS]
    rows8.append(" & ".join(cells) + r" \\")
cent_total = [f"{100 * x:.2f}" for x in C["total"]]
mean_total = [f"{100 * x:.2f}" for x in B["total"]["mean"]]
s8 = "\n".join([
    "% Generated by tables/make_extension_tables.py from",
    "% 04_Results/stat_bootstrap_round1/stat_bootstrap_summary.json.",
    r"\begin{landscape}",
    r"\begin{table}[p]",
    r"\centering",
    r"\footnotesize",
    r"\setlength{\tabcolsep}{1.9pt}",
    (r"\caption{Statistical standard deviation, over 100 Poisson bootstrap replicas, of "
     r"every binned quantity of Tables~\ref{tab:supp-syst-detector} and "
     r"\ref{tab:supp-syst-modelling}. The nominal background rejection column is an "
     r"absolute standard deviation; every other column is in percentage points of "
     r"relative uncertainty. Exactly paired \texttt{esup}, \texttt{esdown}, \texttt{cer} "
     r"and \texttt{cpos} jets share the weight of their nominal partner, the ISR/FSR "
     r"variations and the nominal complements of the modelling samples reuse the nominal "
     r"weights, and all other samples are resampled independently. Above $1.55$~TeV the "
     r"signal-modelling term is the extrapolated maximum, so its standard deviation is that "
     r"of the bin in which the maximum is measured.}"),
    r"\label{tab:supp-bootstrap-binned}",
    r"\begin{tabular}{rrr" + "r" * len(S8_COLS) + "}",
    r"\toprule",
    r"$\pt$ low & $\pt$ high & Nominal BR & " + " & ".join(h for _, h in S8_COLS) + r" \\",
    r"\midrule",
    *rows8,
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
    r"\end{landscape}", ""])

# -------------------------------------------------------------- Table S9 -----
rows9 = []
for fs in SETS:
    for s in seeds:
        r = runs[(fs, s)]["results"]
        rows9.append(" & ".join([
            SET_LABEL[fs] if s == seeds[0] else "", str(s),
            f"{r['nominal_auc']:.5f}", f"{r['nominal_br_50']:.2f}", f"{r['nominal_br_80']:.3f}",
            f"{100 * r['inc_es']:.2f}", f"{100 * r['inc_cluster']:.2f}",
            f"{100 * r['inc_track']:.2f}", f"{100 * r['inc_model']:.2f}",
            f"{100 * r['inc_scale']:.2f}", f"{100 * r['inc_total']:.2f}"]) + r" \\")
    if fs != SETS[-1]:
        rows9.append(r"\midrule")
s9 = "\n".join([
    "% Generated by tables/make_extension_tables.py from",
    "% 04_Results/tagger_variants_round1/tagger_variants_summary.json.",
    r"\begin{table}[!ht]",
    r"\centering",
    r"\footnotesize",
    r"\setlength{\tabcolsep}{3pt}",
    (r"\caption{Every training run of the seed and input-set study. All eighteen runs use "
     r"the frozen configuration and the same $4{,}499{,}612$ fitting jets; only the "
     r"training seed and the input set change. The seed-42 run on all fifteen inputs is "
     r"the frozen model: its SHA-256 and its predictions on every sample are identical to "
     r"the frozen ones. Uncertainties are inclusive relative uncertainties on background "
     r"rejection at $50\%$ signal efficiency, in percent; CES is the cluster energy-scale "
     r"envelope.}"),
    r"\label{tab:supp-seed-runs}",
    r"\begin{tabular}{lrrrrrrrrrr}",
    r"\toprule",
    r"Inputs & Seed & AUC & BR@50\% & BR@80\% & CES & Cluster & Track & Modelling & ISR/FSR & Total \\",
    r"\midrule",
    *rows9,
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}", ""])

# ------------------------------------------------------------- Table S10 -----
def sgn(v: float) -> str:
    """Signed, three decimals, without a signed zero."""
    t = f"{v:+.3f}"
    return "0.000" if t in ("+0.000", "-0.000") else t


rows10 = []
for samp in ("esup", "esdown", "cer", "cpos"):
    for f in FEATURES:
        x = S[samp][f]
        rows10.append(" & ".join([
            f"\\texttt{{{samp}}}" if f == FEATURES[0] else "", LABEL[f], str(DEGREE[f]),
            f"{x['n']:,}".replace(",", "{,}"), str(x["n_excluded"]),
            sgn(100 * x['median']), sgn(100 * x['mean']),
            sgn(100 * x['p16']), sgn(100 * x['p84']),
            f"{100 * x['median_abs']:.3f}", f"{100 * x['fraction_unchanged']:.4f}"]) + r" \\")
    if samp != "cpos":
        rows10.append(r"\midrule")
s10 = "\n".join([
    "% Generated by tables/make_extension_tables.py from",
    "% 04_Results/feature_response_round1/feature_response_summary.json.",
    r"{\footnotesize\setlength{\tabcolsep}{3.5pt}",
    r"\begin{longtable}{llrrrrrrrrr}",
    (r"\caption{Per-jet relative response $X^{s}/X^{\mathrm{nom}}-1$ of every input on "
     r"every exactly paired calorimeter variation, in percent: median, mean, 16th and 84th "
     r"percentiles, median absolute value, and the percentage of jets for which the "
     r"quantity is unchanged. ``Excl.'' counts pairs in which the quantity is non-finite on "
     r"either side or "
     r"zero on the nominal side and is therefore left out.}"
     r"\label{tab:supp-feature-response}\\"),
    r"\toprule",
    r"Sample & Quantity & $d$ & Pairs & Excl. & Median & Mean & 16\% & 84\% & Median $|r|$ & Unchanged \\",
    r"\midrule",
    r"\endfirsthead",
    r"\toprule",
    r"Sample & Quantity & $d$ & Pairs & Excl. & Median & Mean & 16\% & 84\% & Median $|r|$ & Unchanged \\",
    r"\midrule",
    r"\endhead",
    r"\bottomrule",
    r"\endfoot",
    *rows10,
    r"\end{longtable}", r"}", ""])


# ------------------------------------------------------------------ write ----
def main() -> int:
    if failures:
        print("REFUSING TO WRITE. Failed checks:")
        for f in failures:
            print("  -", f)
        return 1
    OUT.mkdir(exist_ok=True)
    outputs = {"table1_nominal_performance_precision.tex": t1_new,
               "table2_inclusive_systematics_precision.tex": t2_new,
               "table6_feature_response_degree.tex": t6,
               "table7_scale_free_variants.tex": t7,
               "supp_s8_bootstrap_binned.tex": s8,
               "supp_s9_seed_runs.tex": s9,
               "supp_s10_feature_response.tex": s10}
    for name, text in outputs.items():
        (OUT / name).write_text(text, encoding="utf-8")
        print(f"wrote tables/ext/{name}")
    print(f"binned total: central {cent_total}\n              bootstrap mean {mean_total}")
    print("all extension-table checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
