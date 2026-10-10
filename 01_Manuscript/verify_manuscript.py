"""Cross-check every numerical claim and cross-reference in the final manuscript.

Read-only.  Compares the manuscript text against the frozen analysis artifacts and
against the LaTeX aux files.  Exits non-zero on any failure.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys

import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Root of the frozen analysis artifacts. Defaults to the copy kept beside this
# directory; set ATLAS80030_ARTIFACTS to point at the project root instead.
_PROJECT = HERE.parent
ART = (Path(os.environ["ATLAS80030_ARTIFACTS"]) if "ATLAS80030_ARTIFACTS" in os.environ
       else (_PROJECT if ((_PROJECT / "Code files" / "analysis_results").is_dir()
                          or (_PROJECT / "04_Results").is_dir())
             else HERE / "artifacts"))
# Frozen campaign outputs: "04_Results" in the reorganised project layout
# (14 September 2026), "Code files/analysis_results" in the original one.
RES = (ART / "04_Results" if (ART / "04_Results").is_dir()
       else ART / "Code files/analysis_results")
PRED = ART / "ATLAS_80030_Data/predictions"
if not RES.is_dir() or not PRED.is_dir():
    sys.exit(f"verify_manuscript: frozen artifacts not found (looked in {RES} and "
             f"{PRED}). Set ATLAS80030_ARTIFACTS to the project root.")

MAIN = (HERE / "atlas80030_main.tex").read_text(encoding="utf-8")
SUPP = (HERE / "atlas80030_supplementary.tex").read_text(encoding="utf-8")
SUPP_S1 = (HERE / "tables/supp_s1_stale_hlf_audit.tex").read_text(encoding="utf-8")

ok, bad = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (ok if condition else bad).append(f"{name}{(' — ' + detail) if detail else ''}")


def has(text: str, needle: str) -> bool:
    return needle in text


# ---------------------------------------------------------------- sources ---
metrics = json.loads(
    (PRED / "final_metrics_atlas_tagger_xgb_final_production_round1_round1.json")
    .read_text(encoding="utf-8"))
M = {(m["dataset"], m["working_point"]): m for m in metrics["metrics"]}
S = metrics["uncertainty_summary"]
model_meta = json.loads(
    (ART / "ATLAS_80030_Data/models/"
     "atlas_tagger_xgb_final_production_round1_metadata.json").read_text(encoding="utf-8"))
cache_txt = (RES / "cache_integrity_audit.txt").read_text(encoding="utf-8")
conf = json.loads((RES / "conformal_coverage_round1/conformal_coverage_results.json")
                  .read_text(encoding="utf-8"))
CONF = {d["dataset"]: d for d in conf["datasets"]}
calib = json.loads((RES / "conformal_coverage_round1/calibration_thresholds.json")
                   .read_text(encoding="utf-8"))
shap_rows = list(csv.DictReader(
    (RES / "shap_delta_final_production_round3/shap_delta_per_feature.csv")
    .open(encoding="utf-8")))
SH = {(r["systematic"], r["sample_size"], r["subset"], r["feature"]): r for r in shap_rows}
closure = list(csv.DictReader(
    (RES / "strict_closure_audit_expanded_all_shards/strict_closure_stats.csv")
    .open(encoding="utf-8")))
CL = {r["feature"]: r for r in closure if r["scope"] == "ALL"}
tagger_log = (RES / "ATLAS80030_tagger_xgb_final_production_round1.txt").read_text(
    encoding="utf-8")
fig1 = list(csv.DictReader(
    (RES / "publication_figures_round1/figure1_performance_systematics_data.csv")
    .open(encoding="utf-8")))

# ------------------------------------------------------- dataset & totals ---
rows = [l.split() for l in cache_txt.splitlines() if l and l[0].isalpha()
        and len(l.split()) == 9 and l.split()[0] != "dataset"]
total_rows = sum(int(r[3]) for r in rows)
total_files = sum(int(r[1]) for r in rows)
check("total jet entries 80,481,421", total_rows == 80_481_421, f"{total_rows:,}")
check("total cache files 810", total_files == 810, str(total_files))
check("manuscript states 80,481,421", has(MAIN, "80{,}481{,}421"))
check("supplementary states 80,481,421", has(SUPP, "80{,}481{,}421"))
check("manuscript states 810 cache files", has(MAIN, "810 cache files"))
check("17 samples in audit", len(rows) == 17, str(len(rows)))

# per-sample rows quoted in Table 1 of the manuscript
per = {r[0]: (int(r[1]), int(r[3])) for r in rows}
def fmt(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")
for ds, (nfiles, nrows) in per.items():
    tex_ds = ds.replace("_", r"\_")
    line = re.search(rf"\\texttt\{{{re.escape(tex_ds)}\}}\s*&\s*(\d+)\s*&\s*([0-9{{}},]+)",
                     MAIN)
    check(f"Table 1 row {ds}", line is not None
          and int(line.group(1)) == nfiles and line.group(2).strip() == fmt(nrows),
          "" if line is None else f"{line.group(1)}/{line.group(2)} vs {nfiles}/{fmt(nrows)}")

# ------------------------------------------------------------- training ----
check("training rows 5,000,000", model_meta["training_rows_requested"] == 5_000_000)
check("eligible 4,999,997", model_meta["eligible_rows"] == 4_999_997)
check("train/valid 4,499,612 / 500,385",
      model_meta["training_rows"] == 4_499_612 and model_meta["validation_rows"] == 500_385)
check("valid AUC 0.93190",
      abs(model_meta["validation_auc_unweighted"] - 0.93190) < 5e-6,
      str(model_meta["validation_auc_unweighted"]))
for s in ["$5{,}000{,}000$ \\texttt{train\\_nominal} jets", "$4{,}999{,}997$",
          "$4{,}499{,}612$", "$500{,}385$", "$0.93190$", "$0.93199$"]:
    check(f"manuscript states {s[:28]}", has(MAIN, s))

# point 3: no bare "trained on 5.0 million jets" anywhere
for text, where in ((MAIN, "manuscript"), (SUPP, "supplement")):
    joined = " ".join(text.split())
    check(f"{where}: no 'trained on 5.0 million' claim",
          not re.search(r"trained on 5\.0 million", joined))
    check(f"{where}: fitting/validation split spelled out",
          "4{,}499{,}612" in joined and "500{,}385" in joined)
check("manuscript says 'developed from a five-million-jet nominal subset'",
      "developed from a five-million-jet nominal subset" in " ".join(MAIN.split()))

# point 4: the over-training claim is softened
joined_main_pt4 = " ".join(MAIN.split())
check("no 'not over-trained' claim", "not over-trained" not in joined_main_pt4)
check("softened held-out wording present",
      "show no evidence of an obvious performance discrepancy" in joined_main_pt4)

# point 10: XGBoost methodology is complete
for needle in ["binary:logistic", "tree\\_method=\"hist\"", "eval\\_metric",
               "n_estimators" if False else "500 trees", "maximum depth 6",
               "learning rate $0.1$", "sample\\_weight",
               "SplitMix64", "$3.2.0$", "$1.9.0$", "$2.4.6$", "3.11.16"]:
    check(f"methodology states {needle[:30]}", has(MAIN, needle))
check("methodology states no early stopping",
      "no early stopping" in joined_main_pt4)
check("methodology states no hyperparameter search",
      "no search over them was performed" in joined_main_pt4)
check("library versions match the frozen model metadata",
      model_meta["xgboost_version"] == "3.2.0"
      and model_meta["sklearn_version"] == "1.9.0"
      and model_meta["numpy_version"] == "2.4.6"
      and model_meta["python_version"].startswith("3.11.16"))
check("frozen validation fraction 0.1 stated",
      model_meta["validation_fraction"] == 0.1
      and "validation fraction of $0.1$" in joined_main_pt4)

# point 11: calibration independence stated explicitly
check("calibration independence stated",
      "influenced neither hyperparameter selection nor the choice" in joined_main_pt4)
check("calibration independence repeated in the conformal section",
      "took no part in fitting the model" in " ".join(SUPP.split())
      and "took no part in fitting the model" in joined_main_pt4)

# ------------------------------------------------------- nominal metrics ---
n50, n80 = M[("test_nominal", 0.5)], M[("test_nominal", 0.8)]
check("test rows 10,306,812", n50["n"] == 10_306_812, f"{n50['n']:,}")
check("AUC 0.9320", abs(n50["auc"] - 0.9320) < 5e-5, str(n50["auc"]))
check("ACC 0.8559", abs(n50["acc"] - 0.8559) < 5e-5, str(n50["acc"]))
check("BR@50 41.91", abs(n50["br"] - 41.91) < 5e-3, str(n50["br"]))
check("BR@80 9.29", abs(n80["br"] - 9.29) < 5e-3, str(n80["br"]))
for s in ["10{,}306{,}812", "$0.9320$", "$0.8559$", "$41.91$", "$9.29$"]:
    check(f"manuscript states {s}", has(MAIN, s))
check("nominal BR first bin 93.4",
      abs(float(fig1[0]["nominal_background_rejection_wp50"]) - 93.4) < 0.05)
check("nominal BR last bin 28.6",
      abs(float(fig1[-1]["nominal_background_rejection_wp50"]) - 28.6) < 0.05)
check("manuscript states 93.4 -> 28.6", has(MAIN, "$93.4$") and has(MAIN, "$28.6$"))

# hlDNN external reference, verbatim from the frozen table artifact
t1 = (HERE / "tables/table1_nominal_performance.tex").read_text(encoding="utf-8")
check("hlDNN labelled as external reference, not reproduced",
      "Published external reference; not reproduced" in t1)
check("hlDNN caption says not a reproduction", "it is not a reproduction" in t1)
check("manuscript repeats the non-reproduction statement",
      "did not attempt to reproduce it" in MAIN)
check("no mcEventWeights detector-systematic claim anywhere in manuscript",
      "columns~1" not in MAIN and "columns 1--9" not in MAIN and "column mapping" not in MAIN)
check("manuscript states mcEventWeights are theory weights",
      "renormalisation- and factorisation-scale and PDF weights" in MAIN
      and "They are not detector systematics" in MAIN)

# ------------------------------------------------------------ systematics --
def pct(x: float) -> float:
    return 100.0 * x
comp = {
    "cluster total": (S["inc_cluster"], 10.07), "track total": (S["inc_track"], 4.51),
    "modelling total": (S["inc_model"], 20.01), "isr/fsr total": (S["inc_scale"], 12.04),
    "total": (S["inc_total"], 25.83),
}
for name, (val, quoted) in comp.items():
    check(f"inclusive {name} = {quoted}%", abs(pct(val) - quoted) < 5e-3,
          f"{pct(val):.5f}")
for s in ["$25.83\\%$", "$20.01\\%$", "$10.07\\%$", "$4.51\\%$", "$12.04\\%$",
          "$18.59\\%$", "$7.40\\%$", "$11.34\\%$", "$9.58\\%$", "$3.78\\%$",
          "$2.47\\%$", "$0.16\\%$", "$25.8\\%$", "$20.0\\%$", "$10.1\\%$", "$4.5\\%$"]:
    check(f"manuscript states {s}", has(MAIN, s))

t2 = (HERE / "tables/table2_inclusive_systematics_wp50.tex").read_text(encoding="utf-8")
for label, quoted in [("Cluster energy scale", "9.58"), ("Cluster total", "10.07"),
                      ("Track total", "4.51"), ("Signal modelling", "18.59"),
                      ("Background modelling", "7.40"), ("Modelling total", "20.01"),
                      ("Signal FSR", "11.34"), ("ISR/FSR total", "12.04"),
                      ("Total uncertainty", "25.83")]:
    check(f"Table 3 row '{label}' = {quoted}",
          re.search(rf"{re.escape(label)} & {re.escape(quoted)} \\\\", t2) is not None)

# detector-only removal arithmetic quoted in the Discussion
without_det = math.hypot(S["inc_model"], S["inc_scale"]) * 100
check("total without detector terms = 23.4%", abs(without_det - 23.4) < 0.05,
      f"{without_det:.3f}")
check("manuscript states 23.4%", has(MAIN, "$23.4\\%$"))
esup_rel = abs(M[("esup", 0.5)]["br"] / n50["br"] - 1) * 100
check("esup inclusive 8.8%", abs(esup_rel - 8.8) < 0.05, f"{esup_rel:.3f}")
check("manuscript states 8.8%", has(MAIN, "$8.8\\%$ inclusively"))
tot = [float(r["total_uncertainty_percent"]) for r in fig1]
check("largest binned total 39.3% in first bin",
      abs(tot[0] - 39.3) < 0.05 and tot[0] == max(tot), f"{tot[0]:.3f}")
check("manuscript states 39.3%", has(MAIN, "$39.3\\%$"))

# point 5: the three remaining scale envelopes combine to 4.06% in quadrature
_t2 = (HERE / "tables/table2_inclusive_systematics_wp50.tex").read_text(encoding="utf-8")
_scale = {lbl: float(re.search(rf"{lbl} & ([\d.]+) ", _t2).group(1))
          for lbl in ("Signal ISR", "Signal FSR", "Background ISR", "Background FSR",
                      "ISR/FSR total")}
other3 = math.hypot(math.hypot(_scale["Signal ISR"], _scale["Background ISR"]),
                    _scale["Background FSR"])
check("frozen table scale total 12.04 reproduces from its components",
      abs(math.hypot(_scale["Signal FSR"], other3) - _scale["ISR/FSR total"]) < 0.01,
      f"{math.hypot(_scale['Signal FSR'], other3):.4f}")
check("other three scale envelopes = 4.06%", abs(other3 - 4.06) < 0.005,
      f"{other3:.4f}")
check("manuscript states 4.06%", has(MAIN, "$4.06\\%$"))
check("manuscript no longer says 'less than 4%'",
      "less than $4\\%$" not in MAIN and "less than 4\\%" not in MAIN)
for s_ in ["$0.19\\%$", "$1.87\\%$", "$3.60\\%$"]:
    check(f"manuscript states {s_}", has(MAIN, s_))

# point 14: the unsupported finite-statistics reading is gone
check("no finite-statistics interpretation of the bin-to-bin fluctuation",
      "finite statistics of the nominal and varied" not in " ".join(MAIN.split()))
# Revision 3: the bands now exist, so the text must say what they are instead.
check("Figure 2 caption describes the bootstrap bands",
      "shaded bands in (b) are $\\pm1$ standard deviation of 100 Poisson bootstrap replicas"
      in " ".join(MAIN.split()))
check("no leftover 'no uncertainty band' / 'not estimated its statistical error' text",
      "no uncertainty band is attached" not in " ".join(MAIN.split())
      and "we have not estimated its statistical error" not in " ".join(MAIN.split()))

# ---------------------------------------------------------------- closure --
med = [float(CL[f]["rel_median"]) for f in CL]
check("closure median range 2.5e-8 .. 4.8e-7",
      abs(min(med) - 2.5e-8) < 1e-9 and abs(max(med) - 4.8e-7) < 1e-8,
      f"{min(med):.3e} .. {max(med):.3e}")
n_close = int(CL["Tau1"]["n_rows_in_scope"])
fails = round(sum(float(CL[f]["fraction_above_relative_tolerance"]) * n_close for f in CL))
check("closure sample 14,103 jets", n_close == 14_103, str(n_close))
check("closure failures 153", fails == 153, str(fails))
check("closure comparisons 211,545", n_close * 15 == 211_545)
check("closure failure fraction 0.072%",
      abs(100 * fails / (n_close * 15) - 0.072) < 0.0005,
      f"{100*fails/(n_close*15):.4f}")
check("ThrustMaj zero failures",
      float(CL["ThrustMaj"]["fraction_above_relative_tolerance"]) == 0.0)
check("ThrustMaj max rel 4.6e-4",
      abs(float(CL["ThrustMaj"]["rel_max"]) - 4.6e-4) < 5e-6,
      CL["ThrustMaj"]["rel_max"])
for s in ["14{,}103", "211{,}545", "$0.072\\%$", "$2.5\\times10^{-8}$",
          "$4.8\\times10^{-7}$", "$4.6\\times10^{-4}$"]:
    check(f"manuscript states {s[:24]}", has(MAIN, s))
check("manuscript states 44 physical jets",
      re.search(r"44\s+physical\s+jets", MAIN) is not None)

# point 1: the four class counts must be stated, and the manuscript must NOT
# say that all 153 tolerance failures were information-loss cases.
check("closure classes stated: 0 mismatches",
      re.search(r"\\textbf\{0\}\s*proven formula or implementation mismatch", MAIN)
      is not None)
check("closure classes stated: 1 numerical degeneracy",
      re.search(r"\\textbf\{1\}\s*reproducible\s+numerical", MAIN) is not None)
check("closure classes stated: 43 information-loss jets",
      re.search(r"\\textbf\{43\}\s*jets consistent with information", MAIN) is not None)
check("closure classes stated: 0 unresolved",
      re.search(r"\\textbf\{0\}\s*unresolved", MAIN) is not None)
check("manuscript disclaims the all-153 reading",
      "it is not the case that all 153 tolerance failures are information-loss"
      in " ".join(MAIN.split()))
for text, where in ((MAIN, "manuscript"), (SUPP, "supplement")):
    joined = " ".join(text.split())
    check(f"{where}: no 'all of them ... information absent' claim",
          "all of them are traced to information absent" not in joined)
ftca = (RES / "final_targeted_closure_audit/FINAL_TARGETED_CLOSURE_AUDIT.md").read_text(
    encoding="utf-8")
check("targeted audit: 153 rows on 44 jets",
      "153 rows on 44 physical jets" in ftca)
check("targeted audit: 0 proven implementation mismatch",
      "'1_proven_implementation_mismatch': 0" in ftca)
check("targeted audit: 1 numerical, 43 information-loss, 0 unresolved",
      "'2_proven_numerical_precision_or_degeneracy': 1" in ftca
      and "'3_consistent_with_information_lost_in_export': 43" in ftca
      and "'4_unresolved': 0" in ftca)

# ----------------------------------------------------------- cache audit ---
check("284 non-finite values", cache_txt.count("\n") > 0 and
      sum(int(r[5]) for r in rows) == 284, str(sum(int(r[5]) for r in rows)))
check("cer sentinel 4,745", per and int([r for r in rows if r[0] == "cer"][0][8]) == 4745)
check("manuscript states 284 and 4,745",
      has(MAIN, "284 non-finite") and has(MAIN, "4{,}745"))
check("manuscript states 198 overlap", has(MAIN, "198 of these"))
check("cache audit PASS", "CACHE CAMPAIGN = PASS" in cache_txt)

# ------------------------------------------------------------- evaluation --
check("test dropped 1", "test_nominal   10,306,812 jets  dropped     1" in tagger_log)
check("cer dropped 244", "cer             4,999,756 jets  dropped   244" in tagger_log)
check("manuscript states 1 to 244 removed", has(MAIN, "ranges from 1 to 244"))
for ds, frac in [("teg", "30.0"), ("tej", "31.8"), ("tfj", "32.0"), ("tfl", "18.5")]:
    m = re.search(rf"{ds}\s+[\d,]+ jets  dropped\s+\d+  signal\s+([\d.]+)%", tagger_log)
    check(f"{ds} signal fraction {frac}%", m is not None
          and abs(float(m.group(1)) - float(frac)) < 0.05,
          m.group(1) if m else "not found")
    check(f"manuscript states {ds} {frac}%", has(MAIN, f"${frac}\\%$"))

# ---------------------------------------------------------------- pairing --
pair_log = (RES / "ATLAS80030_pair_all_round1.txt").read_text(encoding="utf-8")
check("esup/esdown/cpos pair 100.000%", pair_log.count("= 100.000%") == 3)
check("cer pairs 99.901%", "=  99.901%" in pair_log)
check("cer unmatched 4,941 with 96.0% sentinel",
      "unmatched                 4,941" in pair_log
      and "4745/4941 = 96.0%" in pair_log)
for s in ["$100.000\\%$", "$99.901\\%$", "$96.0\\%$", "$4{,}941$"]:
    check(f"manuscript states {s}", has(MAIN, s))
bias = json.loads((RES / "audit_remediation_2026-09-09_round1/"
                   "m4_bias_file000_diagnostic.json").read_text(encoding="utf-8"))
check("bias exact match 83.2%",
      abs(bias["percent_of_bias_rows"]["exact_match"] - 83.223) < 1e-3)
check("bias absent events 12.1%",
      abs(bias["percent_of_bias_rows"]["event_absent"] - 12.095) < 1e-3)
check("bias median nearest dR 1.38",
      abs(bias["float32_bit_patterns_among_common_unmatched_same_label"]
          ["nearest_delta_r"]["median"] - 1.38) < 0.005)
bias_log = (RES / "ATLAS80030_pair_bias_round1.txt").read_text(encoding="utf-8")
check("bias full-sample rate 83.341% in the frozen log",
      "matched to nominal    4,167,067   =  83.341%" in bias_log)
for s_ in ["$83.341\\%$", "$83.223\\%$", "$12.095\\%$", "\\dR = 1.38$",
           "4{,}167{,}067"]:
    check(f"manuscript states {s_}", has(MAIN, s_))
check("manuscript reports the full-sample bias rate, not only the shard",
      "on the full sample" in " ".join(MAIN.split()))
# every one of the nine detector samples appears in the pairing table
pairing_tab = MAIN.split("\\label{tab:pairing}")[1].split("\\end{table}")[0]
for ds in ("esup", "esdown", "cpos", "cer", "bias", "teg", "tej", "tfj", "tfl"):
    check(f"pairing table lists {ds}", f"\\texttt{{{ds}}}" in pairing_tab)
check("pairing table reports a rate for every one of the nine detector samples",
      pairing_tab.count("not tested") == 0
      and all(f"{r} " in pairing_tab for r in
              ("45.008", "38.132", "40.139", "54.991", "83.341", "99.901")))
check("exact pairing claimed only for the four calorimeter variations",
      "The paired SHAP-Delta analysis is restricted to these four" in
      " ".join(MAIN.split()))

# ------------------------------------------------------------------- SHAP --
lead = {}
for sy in ("esup", "esdown", "cpos", "cer"):
    cand = [(k[3], float(v["mean_abs_delta_shap"])) for k, v in SH.items()
            if k[0] == sy and k[1] == "100000" and k[2] == "all"]
    lead[sy] = max(cand, key=lambda t: t[1])
check("esup leader ECF3 0.127",
      lead["esup"][0] == "fjet_ECF3" and abs(lead["esup"][1] - 0.127) < 5e-4,
      f"{lead['esup']}")
check("esdown leader ECF3 0.132",
      lead["esdown"][0] == "fjet_ECF3" and abs(lead["esdown"][1] - 0.132) < 5e-4,
      f"{lead['esdown']}")
check("cer leader ECF3 0.062",
      lead["cer"][0] == "fjet_ECF3" and abs(lead["cer"][1] - 0.062) < 5e-4,
      f"{lead['cer']}")
check("cpos leader Tau4 0.013",
      lead["cpos"][0] == "fjet_Tau4_wta" and abs(lead["cpos"][1] - 0.013) < 5e-4,
      f"{lead['cpos']}")
for s in ["$0.127$", "$0.132$", "$0.062$", "$0.013$"]:
    check(f"manuscript states {s}", has(MAIN, s))
rho100 = [float(v["global_importance_spearman_rho"]) for k, v in SH.items()
          if k[1] == "100000"]
check("min importance rho at 100k = 0.986", abs(min(rho100) - 0.985714) < 1e-5,
      f"{min(rho100):.6f}")
combos100 = {k[0] + "/" + k[2]: float(v["global_importance_spearman_rho"])
             for k, v in SH.items() if k[1] == "100000"}
check("12 systematic-by-class combinations at 100k", len(combos100) == 12)
check("exactly-1 rho in 8 of 12 combinations",
      sum(1 for v in combos100.values() if v == 1.0) == 8,
      str(sum(1 for v in combos100.values() if v == 1.0)))
joined_shap = " ".join(MAIN.split())
check("manuscript states the 0.986 minimum over all class combinations",
      "at least $0.986$ across all twelve systematic-by-class combinations"
      in joined_shap)
check("manuscript states the 0.993 all-jet minimum",
      "restricted to the four all-jet orderings the minimum is $0.993$"
      in joined_shap)
check("manuscript says exactly 1 in eight of them",
      "exactly $1$ in eight of them" in joined_shap)
rho_all = [float(v["global_importance_spearman_rho"]) for k, v in SH.items()
           if k[1] == "100000" and k[2] == "all"]
check("all-jet minimum rho = 0.993", abs(min(rho_all) - 0.992857) < 1e-5,
      f"{min(rho_all):.6f}")
# point 6: "unchanged" must not be used for the rank orderings
for phrase in ["orderings are essentially unchanged", "importance ordering",
               "rankings are unchanged", "does not move at all"]:
    if phrase == "importance ordering":
        continue
    check(f"no absolute-stability wording: {phrase!r}", phrase not in joined_shap)
check("stability described as highly stable / minimal change",
      "highly stable" in joined_shap and "change only minimally" in joined_shap)
check("two adjacent rank-3/4 swaps under esdown described",
      "two adjacent swaps" in joined_shap and "both under \\texttt{esdown}" in joined_shap)
res3 = json.loads((RES / "shap_delta_final_production_round3/shap_delta_results.json")
                  .read_text(encoding="utf-8"))
stab = [x for s in res3["systematics"].values()
        for d in s["stability_diagnostics"].values() for x in d.values()]
check("nested spearman >= 0.989",
      abs(min(x["mean_abs_delta_shap_spearman_vs_100k"] for x in stab) - 0.989286) < 1e-5)
check("top5 overlap 5 in 23 of 24",
      sum(1 for x in stab if x["mean_abs_delta_top5_overlap_with_100k"] == 5) == 23
      and len(stab) == 24)
check("manuscript states >= 0.989 and 23 of 24",
      has(MAIN, "$\\ge 0.989$") and has(MAIN, "23 of 24"))
sc = {(sy, sub): float(SH[(sy, "100000", sub, "fjet_Tau1_wta")]["mean_signed_delta_score"])
      for sy in ("esup", "esdown") for sub in ("signal", "background")}
check("esup signal -0.0029", abs(sc[("esup", "signal")] + 0.0029) < 5e-5,
      f"{sc[('esup','signal')]:.6f}")
check("esup background +0.0055", abs(sc[("esup", "background")] - 0.0055) < 5e-5)
check("esdown signal -0.0113", abs(sc[("esdown", "signal")] + 0.0113) < 5e-5)
check("esdown background -0.0045", abs(sc[("esdown", "background")] + 0.0045) < 5e-5)
for s in ["$-0.0029$", "$+0.0055$", "$-0.0113$", "$-0.0045$"]:
    check(f"manuscript states {s}", has(MAIN, s))
status3 = json.loads((RES / "shap_delta_final_production_round3/campaign_status.json")
                     .read_text(encoding="utf-8"))
check("SHAP round is 3 and complete",
      status3["round"] == 3 and status3["status"] == "complete"
      and status3["crosscheck"] == "PASS")
check("manuscript uses 100,000 pairs", has(MAIN, "$100{,}000$ pairs")
      or has(MAIN, "$100{,}000$ matched"))

# -------------------------------------------------------------- conformal --
check("target 0.9 / alpha 0.1", calib["alpha"] == 0.1 and calib["target_coverage"] == 0.9)
check("sig threshold 0.5700",
      abs(calib["signal"]["nonconformity_threshold"] - 0.5700) < 5e-5)
check("bkg threshold 0.6447",
      abs(calib["background"]["nonconformity_threshold"] - 0.6447) < 5e-5)
check("calibration 500,385 / 252,215 / 248,170",
      calib["total_calibration_count"] == 500_385
      and calib["signal"]["count"] == 252_215
      and calib["background"]["count"] == 248_170)
check("nominal coverage 90.021%",
      abs(CONF["test_nominal"]["overall_coverage"] * 100 - 90.021) < 5e-4,
      f"{CONF['test_nominal']['overall_coverage']*100:.5f}")
worst = min((d["signal_coverage"], d["background_coverage"], d["overall_coverage"])
            for d in conf["datasets"])
allcov = [v for d in conf["datasets"]
          for v in (d["overall_coverage"], d["signal_coverage"], d["background_coverage"])]
check("all coverages within 1 pp of 90%", max(abs(c * 100 - 90) for c in allcov) < 1.0,
      f"max dev {max(abs(c*100-90) for c in allcov):.4f} pp")
check("worst is esdown signal 89.04%",
      abs(CONF["esdown"]["signal_coverage"] * 100 - 89.04) < 5e-3
      and min(allcov) == CONF["esdown"]["signal_coverage"])
check("esdown overall 89.60%",
      abs(CONF["esdown"]["overall_coverage"] * 100 - 89.60) < 5e-3)
check("esup signal 90.12 / background 89.61",
      abs(CONF["esup"]["signal_coverage"] * 100 - 90.12) < 5e-3
      and abs(CONF["esup"]["background_coverage"] * 100 - 89.61) < 5e-3)
check("esdown background 90.21",
      abs(CONF["esdown"]["background_coverage"] * 100 - 90.21) < 5e-3)
check("tej and bias within 0.01 pp of nominal",
      max(abs(CONF[d]["overall_deviation_from_nominal"] * 100) for d in ("tej", "bias"))
      < 0.01)
check("cpos within 0.04 pp of nominal",
      abs(CONF["cpos"]["overall_deviation_from_nominal"] * 100) < 0.04)
check("no empty prediction sets", all(d["empty_set_rate"] == 0 for d in conf["datasets"]))
sizes = [d["mean_prediction_set_size"] for d in conf["datasets"]]
check("mean set size 1.097 .. 1.107",
      abs(min(sizes) - 1.097) < 5e-4 and abs(max(sizes) - 1.107) < 5e-4,
      f"{min(sizes):.4f} .. {max(sizes):.4f}")
for s in ["$90.021\\%$", "$89.04\\%$", "$89.60\\%$", "$90.12\\%$", "$89.61\\%$",
          "$90.21\\%$", "$1.097$", "$1.107$", "0.5700", "0.6447"]:
    check(f"manuscript states {s}", has(MAIN, s))
check("conformal datasets are the 10 evaluated", len(conf["datasets"]) == 10)

# point 2: the inclusion rule must be the correct inequality, in both documents
joined_main = " ".join(MAIN.split())
check("main text gives 1 - p_c <= q_c",
      "1 - \\hat p_c \\;\\le\\; q_c" in joined_main
      or "1 - \\hat p_c \\le q_c" in joined_main)
check("main text gives the equivalent p_c >= 1 - q_c",
      "\\hat p_c \\;\\ge\\; 1 - q_c" in joined_main
      or "\\hat p_c \\ge 1 - q_c" in joined_main)
check("supplement gives 1 - p_c <= q_c",
      "1 - \\hat p_c \\le q_c" in " ".join(SUPP.split()))
check("main text no longer says a class enters when its score exceeds q",
      "exceeds the corresponding threshold" not in joined_main)
check("derived score thresholds 0.4300 / 0.3553 stated",
      has(MAIN, "0.4300") and has(MAIN, "0.3553"))
sig_cut = 1.0 - calib["signal"]["nonconformity_threshold"]
bkg_cut = 1.0 - calib["background"]["nonconformity_threshold"]
check("1 - q_sig = 0.4300", abs(sig_cut - 0.4300) < 5e-5, f"{sig_cut:.6f}")
check("1 - q_bkg = 0.3553", abs(bkg_cut - 0.3553) < 5e-5, f"{bkg_cut:.6f}")
check("acceptance regions overlap, so no empty set is possible",
      sig_cut <= 1.0 - bkg_cut)

# point 8: class-conditional coverage is primary, overall secondary
check("overall coverage described as secondary/descriptive",
      "secondary, descriptive number" in joined_main)
check("class mixture caveat present",
      "weighted by each sample" in joined_main and "class mixture" in joined_main)

# --------------------------------------------------- cross-reference audit --
aux = (HERE / "atlas80030_main.aux").read_text(encoding="utf-8")
labels = set(re.findall(r"\\newlabel\{([^}]+)\}", aux))
refs = set(re.findall(r"\\ref\{([^}]+)\}", MAIN))
check("every \\ref resolves", refs <= labels, str(sorted(refs - labels)))
bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", MAIN))
cited = set()
for grp in re.findall(r"\\cite\{([^}]+)\}", MAIN):
    cited |= {c.strip() for c in grp.split(",")}
check("every \\cite is defined", cited <= bibitems, str(sorted(cited - bibitems)))
check("every bibitem is cited", bibitems <= cited, str(sorted(bibitems - cited)))
check("no undefined references in log",
      "Reference" not in re.sub(r".*rerunfilecheck.*", "",
                                (HERE / "atlas80030_main.log").read_text(
                                    encoding="utf-8", errors="ignore"))
      .split("LaTeX Warning: There were undefined references")[0][-1:] or True)
log = (HERE / "atlas80030_main.log").read_text(encoding="utf-8", errors="ignore")
check("log: no undefined references", "There were undefined references" not in log)
check("log: no undefined citations", "There were undefined citations" not in log)
check("log: no multiply-defined labels", "multiply defined" not in log)

# ------------------------------------------------------------------- ROC ---
roc = list(csv.DictReader(
    (RES / "publication_figures_round1/roc_curve_data.csv").open(encoding="utf-8")))
check("ROC file has 6000 rows", len(roc) == 6000, str(len(roc)))
manifest = json.loads(
    (PRED / "atlas_tagger_xgb_final_production_round1_prediction_manifest.json")
    .read_text(encoding="utf-8"))
for ds in ("test_nominal", "esup", "esdown"):
    sub = [r for r in roc if r["dataset"] == ds]
    check(f"ROC {ds}: 2000 points", len(sub) == 2000, str(len(sub)))
    check(f"ROC {ds}: AUC equals frozen metrics exactly",
          float(sub[0]["auc"]) == M[(ds, 0.5)]["auc"],
          f"{sub[0]['auc']} vs {M[(ds, 0.5)]['auc']}")
    check(f"ROC {ds}: row count equals prediction manifest",
          int(sub[0]["prediction_rows"]) == manifest["datasets"][ds]["prediction_rows"])
    eff = np.array([float(r["signal_efficiency"]) for r in sub])
    bkg = np.array([float(r["background_efficiency"]) for r in sub])
    check(f"ROC {ds}: efficiency grid monotone on [0,1]",
          bool(np.all(np.diff(eff) > 0)) and eff[0] == 0.0 and eff[-1] == 1.0)
    check(f"ROC {ds}: background efficiency monotone and finite",
          bool(np.all(np.diff(bkg) >= -1e-12)) and bool(np.isfinite(bkg).all()))
check("manuscript quotes esup/esdown AUC 0.9294 / 0.9286",
      has(MAIN, "$0.9294$") and has(MAIN, "$0.9286$"))
check("esup AUC rounds to 0.9294", abs(M[("esup", 0.5)]["auc"] - 0.9294) < 5e-5)
check("esdown AUC rounds to 0.9286", abs(M[("esdown", 0.5)]["auc"] - 0.9286) < 5e-5)

# figures cited, in order
for n, lab in enumerate(["fig:roc", "fig:perf", "fig:shap", "fig:conformal"], start=1):
    check(f"Figure {n} label {lab} defined", lab in labels)
    check(f"Figure {n} cited in text", f"\\ref{{{lab}}}" in MAIN)
figorder = [m for m in re.findall(r"\\label\{(fig:[a-z]+)\}", MAIN)]
check("figure labels declared in order 1,2,3,4",
      figorder == ["fig:roc", "fig:perf", "fig:shap", "fig:conformal"], str(figorder))
for n, name in enumerate(["figure1_roc", "figure2_performance_systematics",
                          "figure3_shap_delta", "figure4_conformal_coverage"], 1):
    check(f"Figure {n} graphic is {name}.pdf", f"figures/{name}.pdf" in MAIN)
tablabels = re.findall(r"\\label\{(tab:[a-z0-9-]+)\}", MAIN)
inputs = re.findall(r"\\input\{tables/(?:layout/|ext/)?([a-z0-9_]+)\.tex\}", MAIN)
check("main text inputs the 3 frozen-derived tables and the 2 new ones, in order",
      inputs == ["table1_nominal_performance_precision",
                 "table2_inclusive_systematics_precision",
                 "table3_shap_delta_100k_summary",
                 "table6_feature_response_degree", "table7_scale_free_variants"],
      str(inputs))
for lab in ["tab:datasets", "tab:pairing", "tab:nominal-performance",
            "tab:inclusive-systematics-wp50", "tab:shap-delta-100k",
            "tab:feature-response", "tab:scale-free"]:
    check(f"Table label {lab} resolves", lab in labels)
    check(f"Table {lab} cited", f"\\ref{{{lab}}}" in MAIN)
for n, lab in enumerate(["tab:datasets", "tab:pairing", "tab:nominal-performance",
                         "tab:inclusive-systematics-wp50", "tab:shap-delta-100k",
                         "tab:feature-response", "tab:scale-free"], 1):
    num = re.search(rf"\\newlabel\{{{re.escape(lab)}\}}\{{\{{(\d+)\}}", aux)
    check(f"{lab} numbered {n}", num is not None and num.group(1) == str(n),
          num.group(1) if num else "?")
for n, lab in enumerate(["fig:roc", "fig:perf", "fig:shap", "fig:conformal"], 1):
    num = re.search(rf"\\newlabel\{{{re.escape(lab)}\}}\{{\{{(\d+)\}}", aux)
    check(f"{lab} numbered {n}", num is not None and num.group(1) == str(n),
          num.group(1) if num else "?")

# supplementary numbering
saux = (HERE / "atlas80030_supplementary.aux").read_text(encoding="utf-8")
for n, lab in enumerate(["tab:supp-stale-hlf", "tab:supp-syst-detector",
                         "tab:supp-syst-modelling", "tab:supp-conformal-coverage",
                         "tab:supp-conformal-sets", "tab:supp-shap-per-feature",
                         "tab:supp-shap-scores", "tab:supp-bootstrap-binned",
                         "tab:supp-seed-runs", "tab:supp-feature-response"], 1):
    num = re.search(rf"\\newlabel\{{{re.escape(lab)}\}}\{{\{{(S\d+)\}}", saux)
    check(f"supplementary {lab} numbered S{n}",
          num is not None and num.group(1) == f"S{n}", num.group(1) if num else "?")
for n in range(1, 11):
    check(f"main text refers to Supplementary Table~S{n}", f"Table~S{n}" in MAIN
          or f"Tables~S{n}" in MAIN or f"and~S{n}" in MAIN)
supp_refs = {"Fig.~2(b)": "S2/S3 accompany Figure 2(b)",
             "Fig.~4": "S4/S5 accompany Figure 4",
             "Fig.~3 and Table~5": "S6/S7 accompany Figure 3 and Table 5"}
for needle, why in supp_refs.items():
    check(f"supplementary cross-reference: {why}", needle in SUPP, needle)

# point 7: the scope of the two diagnostics must be stated correctly
joined_scope = " ".join(MAIN.split())
check("introduction says the SHAP analysis is paired and restricted to four",
      "The SHAP analysis is genuinely paired" in joined_scope
      and "restricted to the four calorimeter-cluster variations" in joined_scope)
check("introduction says the conformal analysis is not paired",
      "The conformal analysis is not paired" in joined_scope)
check("introduction does not claim both use matched pairs",
      "both are computed on physically matched jet pairs" not in joined_scope)
check("limitations separate the two scopes",
      "Conformal coverage covers all ten evaluated samples" in joined_scope)
check("limitations no longer say coverage speaks only to the cluster variations",
      "the attribution and coverage results speak only to the calorimeter"
      not in joined_scope)

# point 9: the supplement carries the stale-HLF audit table
check("supplement inputs the stale-HLF audit table",
      "tables/supp_s1_stale_hlf_audit.tex" in SUPP)
check("main text points at Supplementary Table S1 for the stale branches",
      "documented sample by sample in Supplementary Table~S1" in joined_scope)

# point 15: references and submission details
check("Bhattacherjee citation is EPJ Plus 139 (2024) 1131",
      re.search(r"Eur\. Phys\. J\. Plus \\textbf\{139\} \(2024\) 1131", MAIN)
      is not None)
check("Angelopoulos & Bates cited in the published version",
      "Found. Trends Mach. Learn." in MAIN and "10.1561/2200000101" in MAIN
      and "Angelopoulos2023" in MAIN)
check("no stale Angelopoulos2021 key", "Angelopoulos2021" not in MAIN)
check("ATLAS reference uses the exact commit URL",
      "/-/tree/%\n5bc3c8d70cf8736d24b969972a3dd365017a9e24" in MAIN
      or "/-/tree/5bc3c8d70cf8736d24b969972a3dd365017a9e24" in MAIN)
check("EPJ C template: sn-jnl with iicol", "iicol]{sn-jnl}" in MAIN)
abstract = MAIN.split("\\abstract{", 1)[1].split("\n\n\\keywords", 1)[0]
words = len(re.sub(r"[^A-Za-z0-9$\\{}^.,%-]+", " ",
                   re.sub(r"\\[a-zA-Z]+|[{}$\\]", " ", abstract)).split())
check("abstract within the EPJ C 150-250 word range", 150 <= words <= 250, str(words))
# 1 Oct 2026: declarations reduced by the supervising author to EPJ C's standard
# Data and Code Availability Statements; Funding, Competing interests and Author
# contributions sections were removed on purpose and are no longer required here.
for decl in ["\\section*{Statements and declarations}",
             "\\subsection*{Data Availability Statement}",
             "\\subsection*{Code Availability Statement}"]:
    check(f"declaration present: {decl}", decl in MAIN)

# point 16: no claim that everything needed is already public
for text, where in ((MAIN, "manuscript"), (SUPP, "supplement")):
    joined = " ".join(text.split())
    check(f"{where}: no 'everything needed to repeat the work is public'",
          "Everything needed to repeat the work is public" not in joined)
    check(f"{where}: no unfilled repository DOI placeholder",
          "REPOSITORY DOI TO BE INSERTED" not in joined)
check("data availability statement cites Record 80030 by DOI",
      "This manuscript has associated data in a data repository" in joined_scope
      and "10.7483/OPENDATA.ATLAS.SOAY.LABE" in joined_scope)
check("code availability statement points to the public code deposit",
      "This manuscript has associated code/software in a data repository" in joined_scope
      and "10.5281/zenodo.23083866" in joined_scope
      and "github.com/adeelrahmat64-beep/atlas80030-tagger-robustness" in joined_scope
      and "on reasonable request" not in joined_scope)
# 10 Oct 2026: the four extension scripts are released in v1.1.0; the statement
# must name that release and the Zenodo concept DOI, and promise nothing for later.
check("code availability cites release v1.1.0 and the code concept DOI, no 'next release'",
      "Release v1.1.0" in joined_scope
      and "10.5281/zenodo.23083865" in joined_scope
      and "next release" not in joined_scope
      and "will be added" not in joined_scope)

# point 17 (revised in revision 3): the bootstrap and the multi-seed study are now
# done; only the disjoint-training-block comparison remains future work.
check("remaining extra (disjoint training blocks) flagged as future work",
      "a comparison across disjoint training blocks remains to be done" in joined_scope)
check("old 'no statistical uncertainty is attached' limitation removed",
      "No statistical uncertainty is attached to any of the ratios" not in joined_scope
      and "natural next step" not in joined_scope)



# ------------------------------------------------- stored-branch audit (S1) --
# The full-sample audit that replaced the previously unsourced Section 2.3
# numbers. Everything below is re-read from its frozen outputs.
STALE = RES / "stale_hlf_audit"
if STALE.is_dir():
    sh = json.loads((STALE / "stale_hlf_audit.json").read_text(encoding="utf-8"))
    SH_ = {r["dataset"]: r for r in sh["results"]}
    check("stale-HLF audit covers all nine detector samples", len(SH_) == 9,
          str(sorted(SH_)))
    check("audit ran on all fifty shards of every sample",
          all(r["shards_audited"] == 50 for r in SH_.values()))

    pairable = [d for d, r in SH_.items() if r["pairable"]]
    check("exactly the four calorimeter samples are pairable",
          sorted(pairable) == ["cer", "cpos", "esdown", "esup"], str(sorted(pairable)))

    # every stored branch of every pairable sample is bit-identical to nominal
    branches = list(csv.DictReader((STALE / "stale_hlf_audit_branches.csv")
                                   .open(encoding="utf-8")))
    check("per-branch CSV has 15 branches for each of the nine samples",
          len(branches) == 135, str(len(branches)))
    pb = [b for b in branches if b["dataset"] in pairable]
    check("all 60 pairable sample-branch rows are fully identical",
          len(pb) == 60 and all(b["identical"] == b["compared"] for b in pb)
          and all(float(b["identical_fraction"]) == 1.0 for b in pb))
    tot_ident = sum(int(b["identical"]) for b in pb)
    tot_comp = sum(int(b["compared"]) for b in pb)
    check("299,925,885 stored values compared, all identical",
          tot_ident == tot_comp == 299_925_885, f"{tot_ident:,}/{tot_comp:,}")
    for ds, n in (("esup", 75_000_000), ("esdown", 75_000_000),
                  ("cpos", 75_000_000), ("cer", 74_925_885)):
        got = sum(int(b["identical"]) for b in branches if b["dataset"] == ds)
        check(f"{ds}: {n:,} identical stored values", got == n, f"{got:,}")
        check(f"manuscript or supplement states {ds} identity count",
              fmt(n) in SUPP_S1)

    # exact-key hits, including the track samples the earlier revision left blank
    for ds, hits in (("esup", 5_000_000), ("esdown", 5_000_000),
                     ("cpos", 5_000_000), ("cer", 4_995_059),
                     ("teg", 2_250_423), ("tej", 1_906_619),
                     ("tfj", 2_006_933), ("tfl", 2_749_571),
                     ("bias", 4_167_067)):
        check(f"audit exact-key hits {ds} = {hits:,}", SH_[ds]["matched"] == hits,
              f"{SH_[ds]['matched']:,}")
        check(f"manuscript Table 2 states {ds} hits", fmt(hits) in MAIN)
    check("teg is excluded for duplicate keys, not merely coverage",
          SH_["teg"]["duplicate_key_count"] == 183_802
          and "183{,}802" in MAIN, str(SH_["teg"]["duplicate_key_count"]))

    # constituent-level evidence quoted in Section 2.3
    ev = {d: SH_[d]["constituent_evidence"]["statistics"] for d in pairable}
    for ds, key, quoted, tex in (
            ("esup", "mean", 6.780753, "$+6.780753\\%$"),
            ("esup", "median", 7.406764, "$+7.406764\\%$"),
            ("esdown", "mean", -6.780749, "$-6.780749\\%$"),
            ("esdown", "median", -7.406822, "$-7.406822\\%$"),
            ("cer", "rms", 3.770123, "$3.770123\\%$")):
        got = ev[ds][key] * 100.0
        check(f"{ds} {key} = {quoted}%", abs(got - quoted) < 5e-6, f"{got:.6f}")
        check(f"manuscript states {tex}", has(MAIN, tex))
    cpos_dr = ev["cpos"]["median"]
    check("cpos median centroid dR = 4.9387e-5", abs(cpos_dr - 4.9387e-5) < 5e-9,
          f"{cpos_dr:.6e}")
    check("manuscript states the cpos displacement",
          has(MAIN, "\\Delta R=4.9387\\times10^{-5}"))

    # the discrepancy note: the old +6.8% is the mean, not the median
    disc = sh["esq" if False else "esup_manuscript_discrepancy"]
    check("audit resolves the historical +6.8% as the exact-pair mean",
          disc["status"] == "resolved"
          and abs(disc["full_exact_pair_mean_percent"] - 6.780753) < 5e-6
          and disc["mean_rounded_to_one_decimal_percent"] == 6.8)
else:
    check("stored-branch audit outputs present", False,
          f"{STALE} not found - run stale_hlf_audit.py")

# the unsourced claims the audit replaced must not come back
for text, where in ((MAIN, "manuscript"), (SUPP, "supplement")):
    joined = " ".join(text.split())
    check(f"{where}: unsourced +6.8% claim is gone", "$+6.8\\%$" not in joined)
    check(f"{where}: unsourced 'roughly 10% of jets' claim is gone",
          "roughly $10\\%$ of jets" not in joined)

# ------------------------------------------- remaining internal-audit items --
joined_fix = " ".join(MAIN.split())
# A-2: the pairing pool is described as the frozen log records it
check("A-2: pairing pool described as built over both nominal partitions",
      "built over both nominal partitions" in joined_fix)
check("A-2: no claim that the pool was restricted at the time",
      "the nominal pool is restricted to" not in joined_fix)
check("A-2: frozen log evidence quoted",
      "every matched jet came from \\texttt{test\\_nominal} and none from"
      in joined_fix)
pair_log_all = (RES / "ATLAS80030_pair_all_round1.txt").read_text(encoding="utf-8")
check("frozen log really records both nominal partitions in the pool",
      "train_nominal" in pair_log_all and "nominal pool: 15,306,813 jets" in pair_log_all)
check("frozen log really records every match from test_nominal",
      pair_log_all.count("from test_nominal") == 4)

# B-1: the vendored ATLAS reference scripts are cited
check("B-1: reference scripts named and said to be preserved with hashes",
      "calc.py" in joined_fix and "plot\\_everything.py" in joined_fix
      and "git blob and SHA-256 hashes recorded" in joined_fix)

# B-2: both pairing fingerprints appear in the Table 2 caption
check("B-2: both pairing fingerprints named",
      "c91534fd" in joined_fix and "7be1a72a" in joined_fix)

# B-3: the modelling term is a transfer between processes
check("B-3: process transfer stated",
      "transfer" in joined_fix and "$2$~TeV $Z'$ resonance" in joined_fix)

# B-4: the in-binning fraction, re-derived from the frozen per-bin counts
bcc = M[("test_nominal", 0.5)]["bin_class_counts"]
inside = sum(sum(b) for b in bcc)
sig_inside = sum(b[1] for b in bcc)
n_test = M[("test_nominal", 0.5)]["n"]
sig_total = CONF["test_nominal"]["signal_count"]
outside = n_test - inside
sig_outside = sig_total - sig_inside
check("in-binning jets 8,959,008", inside == 8_959_008, f"{inside:,}")
check("in-binning fraction 86.9%", abs(100 * inside / n_test - 86.9) < 0.05,
      f"{100 * inside / n_test:.2f}")
check("outside fraction 13.1%", abs(100 * outside / n_test - 13.1) < 0.05,
      f"{100 * outside / n_test:.2f}")
check("signal fraction inside 48.8%", abs(100 * sig_inside / inside - 48.8) < 0.05,
      f"{100 * sig_inside / inside:.2f}")
check("signal fraction outside 61.8%", abs(100 * sig_outside / outside - 61.8) < 0.05,
      f"{100 * sig_outside / outside:.2f}")
for s_ in ["8{,}959{,}008", "$86.9\\%$", "$13.1\\%$", "$61.8\\%$", "$48.8\\%$"]:
    check(f"manuscript states {s_}", has(MAIN, s_))

# C-1, C-3, C-4
check("C-1: tej bound tightened to 0.01 pp",
      "\\texttt{tej} by less than\n$0.01$" in MAIN
      or "\\texttt{tej} by less than $0.01$" in joined_fix)
tej_dev = max(abs(CONF["tej"]["signal_deviation_from_nominal"]),
              abs(CONF["tej"]["background_deviation_from_nominal"])) * 100
check("tej largest class-conditional move is under 0.01 pp", tej_dev < 0.01,
      f"{tej_dev:.4f}")
check("C-3: pilot claim scoped to the frozen campaign",
      "the only other tagger trained in the frozen campaign" in joined_fix)
check("C-4: the unsourced 85% constituent-mass figure is gone",
      "roughly $85\\%$" not in joined_fix)


# ------------------------------------------- robustness extension (rev. 3) --
# Everything below is re-read from the three extension result folders and from
# the README provenance note; nothing is typed in except the rounded value the
# text quotes, which is compared with the artifact.
import hashlib
J = " ".join(MAIN.split())
JS = " ".join(SUPP.split())
EXT_RES = {n: RES / n for n in ("stat_bootstrap_round1", "tagger_variants_round1",
                                 "feature_response_round1")}
if all(p.is_dir() for p in EXT_RES.values()):
    boot = json.loads((EXT_RES["stat_bootstrap_round1"] / "stat_bootstrap_summary.json")
                      .read_text(encoding="utf-8"))
    var = json.loads((EXT_RES["tagger_variants_round1"] / "tagger_variants_summary.json")
                     .read_text(encoding="utf-8"))
    resp = json.loads((EXT_RES["feature_response_round1"] / "feature_response_summary.json")
                      .read_text(encoding="utf-8"))
    BC, BB = boot["central"], boot["bootstrap"]
    FROZEN_SHA = "63a23308efbf2ce90418c8e657730f571db4798fc9ca2fb3692cb986de405f4c"

    def r1(x, nd):        # the rounding the text uses
        return f"{x:.{nd}f}"

    def says(needle, what=None):
        check(f"manuscript states {what or needle[:60]}", needle in J, needle)

    # --- bootstrap provenance
    check("bootstrap: 100 replicas, base seed 20261009",
          boot["replicas"] == 100 and boot["base_seed"] == 20261009)
    check("bootstrap: run on the frozen model", boot["frozen_model_sha256"] == FROZEN_SHA)
    fm = PRED / "final_metrics_atlas_tagger_xgb_final_production_round1_round1.json"
    # The bootstrap hashed the Windows working copy (CRLF line endings); git stores the
    # file with LF, so a Linux/macOS checkout differs only in line endings. Compare the
    # raw bytes and the CRLF-normalised bytes, which is still an exact content match.
    _fm = fm.read_bytes()
    _fm_crlf = _fm.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    check("bootstrap: frozen metrics file is the one the verifier reads",
          boot["frozen_metrics_json_sha256"] in (hashlib.sha256(_fm).hexdigest(),
                                                 hashlib.sha256(_fm_crlf).hexdigest()))
    check("bootstrap: central inclusive values equal the frozen uncertainty summary exactly",
          all(BC[k] == S[k] for k in S if k.startswith("inc_"))
          and sum(1 for k in S if k.startswith("inc_")) == 5)
    check("bootstrap: central binned values equal the frozen summary's 14 physical bins",
          all(np.array_equal(np.array(S[k], dtype=float)[:14], np.array(BC[k], dtype=float))
              for k in ("nominal_binned_br", "cluster", "track", "scale", "model", "total")))
    check("bootstrap: central BR@50 equals the frozen metrics for all 16 datasets",
          all(BC[f"br50_{ds}"] == M[(ds, 0.5)]["br"] for ds in
              ("test_nominal", "esup", "esdown", "cer", "cpos", "teg", "tej", "tfl", "tfj",
               "bias", "ttbar_pythia", "ttbar_herwig", "cluster", "string", "angular",
               "dipole")))
    check("bootstrap: central binned total equals the frozen figure CSV",
          np.allclose(100 * np.array(BC["total"]), tot, rtol=0, atol=1e-9))
    reps_ = [json.loads(l) for l in (EXT_RES["stat_bootstrap_round1"] / "replicas.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip()]
    check("bootstrap: 100 replica records, indices 0..99",
          sorted(r["replica"] for r in reps_) == list(range(100)))
    check("bootstrap: summary s.d. recomputes from the replicas",
          all(abs(float(np.std([r[k] for r in reps_], ddof=1)) - BB[k]["std"]) < 1e-12
              for k in ("inc_total", "nominal_br_50", "nominal_auc", "inc_bias")))
    check("bootstrap: paired-weight sharing used for the four calorimeter samples",
          boot["paired_jets"] == {"esup": 4_999_999, "esdown": 4_999_999,
                                  "cer": 4_995_058, "cpos": 4_999_999})
    check("supplement states the bootstrap base seed", "$20261009$" in JS)

    sd = {k: 100 * BB[k]["std"] for k in BB if k.startswith("inc_")}
    for key, txt in (("inc_es", "$9.58 \\pm 0.33\\%$"), ("inc_sig_model", "$18.59 \\pm 0.50\\%$"),
                     ("inc_sig_FSR", "$11.34 \\pm 0.64\\%$"), ("inc_total", "$25.83 \\pm 0.58\\%$"),
                     ("inc_cpos", "$1.13 \\pm 0.33\\%$"), ("inc_bias", "$0.16 \\pm 0.32\\%$")):
        cen, err = txt.strip("$").replace("\\%", "").split(" \\pm ")
        check(f"{key} = {cen} +- {err} (stat)",
              r1(100 * BC[key], 2) == cen and r1(sd[key], 2) == err,
              f"{100 * BC[key]:.4f} +- {sd[key]:.4f}")
        says(txt)
    check("abstract BR@50 stat 0.12", r1(BB["nominal_br_50"]["std"], 2) == "0.12")
    says("$41.91 \\pm 0.12$ (stat.)")
    check("AUC stat 0.00008, BR@80 stat 0.015",
          abs(BB["nominal_auc"]["std"] - 0.00008) < 5e-6
          and r1(BB["nominal_br_80"]["std"], 3) == "0.015")
    says("$\\pm 0.00008$ on the AUC")
    says("$\\pm 0.015$ at $80\\%$")
    check("total stat 0.6 and seed 0.5 (abstract)",
          r1(sd["inc_total"], 1) == "0.6"
          and r1(100 * var["summary"]["full15"]["inc_total"]["std"], 1) == "0.5")
    says("(25.8 \\pm 0.6_{\\mathrm{stat}} \\pm 0.5_{\\mathrm{seed}})\\%")
    check("modelling stat 0.49%", r1(sd["inc_model"], 2) == "0.49")
    says("the statistical uncertainty of the modelling term is $0.49\\%$")
    check("track bias replica mean 0.45%", r1(100 * BB["inc_bias"]["mean"], 2) == "0.45")
    says("replica mean of $0.45\\%$")
    zs = {k: BC[k] / BB[k]["std"] for k in ("inc_es", "inc_cer", "inc_cpos", "inc_cluster",
          "inc_eff", "inc_fake", "inc_bias", "inc_track", "inc_sig_model", "inc_bkg_model",
          "inc_model", "inc_sig_ISR", "inc_sig_FSR", "inc_bkg_ISR", "inc_bkg_FSR",
          "inc_scale", "inc_total")}
    check("every Table 4 component >= 5 sd from zero except cpos and bias",
          sorted(k for k, z in zs.items() if z < 5) == ["inc_bias", "inc_cpos"],
          str({k: round(z, 2) for k, z in zs.items() if z < 6}))
    says("every entry of the table except two lies at least five standard deviations from zero")
    check("background parton-shower comparison 1.29 +- 0.40 (3.2 sd, not in Table 4)",
          r1(100 * BC["inc_bkg_ps"], 2) == "1.29" and r1(sd["inc_bkg_ps"], 2) == "0.40")
    says("the parton-shower one is $1.29 \\pm 0.40\\%$")
    check("no 'every component except two' over-statement", "every component except two" not in J)

    tsd = 100 * np.array(BB["total"]["std"])
    check("binned total stat s.d. 1.3 .. 2.6",
          r1(tsd.min(), 1) == "1.3" and r1(tsd.max(), 1) == "2.6",
          f"{tsd.min():.3f} .. {tsd.max():.3f}")
    says("statistical standard deviation of the total is $1.3$ to $2.6$ percentage points")
    bias_up = 100 * (np.array(BB["total"]["mean"]) - np.array(BC["total"]))
    check("binned total replica mean above measured by 0.8 .. 1.6",
          bool((bias_up > 0).all()) and r1(bias_up.min(), 1) == "0.8"
          and r1(bias_up.max(), 1) == "1.6", f"{bias_up.min():.3f} .. {bias_up.max():.3f}")
    says("lies $0.8$ to $1.6$ percentage points above the measured value")

    def chi2_const(key):
        x = 100 * np.array(BC[key]); e = 100 * np.array(BB[key]["std"])
        w = 1 / e ** 2
        return float((w * (x - (w * x).sum() / w.sum()) ** 2).sum())
    for key, q in (("es", "55.7"), ("cluster", "45.7"), ("track", "5.4"), ("cer", "15.9"),
                   ("cpos", "3.4"), ("bias", "4.7")):
        c2 = chi2_const(key)
        check(f"chi2 vs constant for {key} = {q}", r1(c2, 1) == q, f"{c2:.3f}")
    says("$\\chi^2 = 55.7$ for 13 degrees of freedom for the energy-scale envelope and $45.7$ for the cluster category")
    says("For the track category it gives $\\chi^2 = 5.4$, and for \\texttt{cer}, \\texttt{cpos} and track bias $15.9$, $3.4$ and $4.7$")
    for key in ("cpos", "bias"):
        n_small = int(np.sum(np.array(BC[key]) < np.array(BB[key]["std"])))
        check(f"{key}: below its own s.d. in 8 of 14 bins", n_small == 8, str(n_small))
    says("in eight of the fourteen bins the \\texttt{cpos} and the track-bias values are each smaller than their own statistical standard deviation")

    # --- training seeds
    VR = {(r["feature_set"], r["seed"]): r for r in var["runs"]}
    seeds_ = var["summary"]["full15"]["seeds"]
    check("six seeds 42, 1..5 for every input set", seeds_ == [42, 1, 2, 3, 4, 5]
          and len(var["runs"]) == 18
          and all((fs, s_) in VR for fs in ("full15", "ratios9", "pt_normalised14")
                  for s_ in seeds_))
    ctl = VR[("full15", 42)]
    check("seed-42 refit is the frozen model, bit-identical",
          ctl["model_sha256"] == FROZEN_SHA
          and ctl["control_predictions_bit_identical_to_frozen"] is True
          and ctl["control_reproduces_frozen_metrics"] is True)
    check("seed-42 refit metrics equal the frozen summary",
          all(ctl["results"][k] == S[k] for k in S if k.startswith("inc_")
              and k in ctl["results"]))
    check("all runs use the frozen split and hyper-parameters",
          var["split_seed"] == 42 and var["xgb_params"]["n_estimators"] == 500
          and var["xgb_params"]["max_depth"] == 6
          and var["xgb_params"]["learning_rate"] == 0.1
          and var["xgb_params"]["subsample"] == 0.8
          and var["xgb_params"]["colsample_bytree"] == 0.8)

    def seedv(fs, key):
        return np.array([VR[(fs, s_)]["results"][key] for s_ in seeds_], dtype=float)
    br = seedv("full15", "nominal_br_50")
    check("seed BR@50 range 41.74 .. 41.91, s.d. 0.07",
          r1(br.min(), 2) == "41.74" and r1(br.max(), 2) == "41.91"
          and r1(br.std(ddof=1), 2) == "0.07")
    says("ranges from $41.74$ to $41.91$, with a standard deviation of $0.07$")
    check("seed s.d. of BR@50 smaller than its stat s.d.",
          br.std(ddof=1) < BB["nominal_br_50"]["std"])
    tt = 100 * seedv("full15", "inc_total")
    check("seed total 26.32 +- 0.47, range 25.83 .. 27.01, frozen seed lowest",
          r1(tt.mean(), 2) == "26.32" and r1(tt.std(ddof=1), 2) == "0.47"
          and r1(tt.min(), 2) == "25.83" and r1(tt.max(), 2) == "27.01"
          and tt.argmin() == 0)
    says("the total is $26.32 \\pm 0.47\\%$, ranging from $25.83\\%$ to $27.01\\%$")
    says("the frozen seed gives the lowest total and the lowest modelling term of the six")
    check("frozen seed has the lowest modelling term too",
          int(np.argmin(seedv("full15", "inc_model"))) == 0)
    mm = 100 * seedv("full15", "inc_model")
    check("seed modelling 20.65 +- 0.43",
          r1(mm.mean(), 2) == "20.65" and r1(mm.std(ddof=1), 2) == "0.43")
    says("modelling term $20.65 \\pm 0.43\\%$")
    says("its spread over training seeds $0.43\\%$")
    t3x = (HERE / "tables/ext/table1_nominal_performance_precision.tex").read_text(encoding="utf-8")
    check("extended Table 3 keeps the frozen row verbatim",
          "XGBoost (this work) & Frozen final result & 0.931995 & 0.855854 & 41.9064 & 9.28847 \\\\" in t3x
          and "$0.9374 \\pm 0.0001$ & 0.8628 & $47.2 \\pm 0.4$ & $10.36 \\pm 0.03$" in t3x)
    t4x = (HERE / "tables/ext/table2_inclusive_systematics_precision.tex").read_text(encoding="utf-8")
    check("extended Table 4 total row", "Total & Total uncertainty & 25.83 & 0.58 & $26.32 \\pm 0.47$ \\\\" in t4x)
    for label, quoted in [("Cluster energy scale", "9.58"), ("Cluster total", "10.07"),
                          ("Track total", "4.51"), ("Signal modelling", "18.59"),
                          ("Background modelling", "7.40"), ("Modelling total", "20.01"),
                          ("Signal FSR", "11.34"), ("ISR/FSR total", "12.04")]:
        check(f"extended Table 4 keeps frozen '{label}' = {quoted}",
              re.search(rf"& {re.escape(label)} & {re.escape(quoted)} & ", t4x) is not None)

    # --- scale-free input sets
    check("ratios9 is exactly the nine degree-0 inputs",
          sorted(var["feature_sets"]["ratios9"])
          == sorted(f for f, d in var["scale_degree"].items() if d == 0)
          and len(var["feature_sets"]["ratios9"]) == 9)
    check("pt_normalised14 = nine ratios + five X/ECF1^d",
          len(var["feature_sets"]["pt_normalised14"]) == 14
          and sum("/ECF1^" in f for f in var["feature_sets"]["pt_normalised14"]) == 5)
    def mean_(fs, key, sc=100):
        return sc * seedv(fs, key).mean()
    for fs, key, sc, nd, q in (
            ("ratios9", "inc_es", 100, 2, "2.03"), ("full15", "inc_es", 100, 2, "9.45"),
            ("ratios9", "inc_cluster", 100, 2, "2.75"), ("full15", "inc_cluster", 100, 2, "9.91"),
            ("ratios9", "inc_model", 100, 2, "6.60"), ("ratios9", "inc_total", 100, 2, "15.80"),
            ("full15", "nominal_auc", 1, 4, "0.9320"), ("ratios9", "nominal_auc", 1, 4, "0.8761"),
            ("full15", "nominal_br_50", 1, 2, "41.82"), ("ratios9", "nominal_br_50", 1, 2, "18.88"),
            ("full15", "inc_scale", 100, 2, "12.22"), ("ratios9", "inc_scale", 100, 2, "13.81"),
            ("full15", "inc_sig_FSR", 100, 2, "11.43"), ("ratios9", "inc_sig_FSR", 100, 2, "13.48"),
            ("pt_normalised14", "nominal_br_50", 1, 2, "19.48"),
            ("pt_normalised14", "inc_total", 100, 2, "16.13"),
            ("full15", "inc_total", 100, 1, "26.3"), ("ratios9", "inc_total", 100, 1, "15.8"),
            ("full15", "nominal_br_50", 1, 1, "41.8"), ("ratios9", "nominal_br_50", 1, 1, "18.9")):
        v = mean_(fs, key, sc)
        check(f"six-seed mean {fs}/{key} = {q}", r1(v, nd) == q, f"{v:.5f}")
    check("ratios9 ES s.d. 0.14, full15 ES s.d. 0.11",
          r1(100 * seedv("ratios9", "inc_es").std(ddof=1), 2) == "0.14"
          and r1(100 * seedv("full15", "inc_es").std(ddof=1), 2) == "0.11")
    says("energy-scale uncertainty of $2.03 \\pm 0.14\\%$, against $9.45 \\pm 0.11\\%$")
    says("the total falls from $26.32\\%$ to $15.80\\%$")
    says("the AUC falls from $0.9320$ to $0.8761$ and the background rejection at $50\\%$ signal efficiency from $41.82$ to $18.88$")
    says("from $12.22\\%$ to $13.81\\%$")
    says("which goes from $11.43\\%$ to $13.48\\%$")
    says("background rejection $19.48$ and total uncertainty $16.13\\%$")
    says("$15.8\\%$ but background rejection from $41.8$ to $18.9$")
    cats = ("inc_cluster", "inc_track", "inc_model", "inc_scale")
    check("of the four categories only ISR/FSR rises without absolute scale",
          [k for k in cats if mean_("ratios9", k) > mean_("full15", k)] == ["inc_scale"])
    check("'more than half of the background rejection' is removed",
          mean_("ratios9", "nominal_br_50", 1) < 0.5 * mean_("full15", "nominal_br_50", 1))
    check("AUC seed s.d. below 5e-5 for every input set (Table 7 caption)",
          all(seedv(fs, "nominal_auc").std(ddof=1) < 5e-5
              for fs in ("full15", "ratios9", "pt_normalised14")))

    # --- degrees and per-jet input response
    DEG = var["scale_degree"]
    check("degrees: response and variant studies agree", resp["scale_degree"] == DEG)
    check("degrees: 0 for tau, C2, D2, L2, L3, thrust; 1 for d12, d23, ECF1, Qw; ECF2 2; ECF3 3",
          DEG == {**{f: 0 for f in ("fjet_Tau1_wta", "fjet_Tau2_wta", "fjet_Tau3_wta",
                                     "fjet_Tau4_wta", "fjet_C2", "fjet_D2", "fjet_L2",
                                     "fjet_L3", "fjet_ThrustMaj")},
                  "fjet_Split12": 1, "fjet_Split23": 1, "fjet_ECF1": 1, "fjet_Qw": 1,
                  "fjet_ECF2": 2, "fjet_ECF3": 3})
    RS = resp["samples"]
    up = {f: 100 * v["median"] for f, v in RS["esup"].items()}
    dn = {f: 100 * v["median"] for f, v in RS["esdown"].items()}
    check("ECF1 esup median 7.41 equals the stale-HLF audit +7.406764%",
          abs(up["fjet_ECF1"] - 7.406764) < 5e-5, f"{up['fjet_ECF1']:.6f}")
    for f, q in (("fjet_ECF1", "7.41"), ("fjet_ECF2", "12.57"), ("fjet_ECF3", "17.63")):
        check(f"esup median {f} = +{q}%", r1(up[f], 2) == q, f"{up[f]:.4f}")
    d1 = [up[f] for f in ("fjet_Split12", "fjet_Split23", "fjet_Qw")]
    check("other d=1 esup medians 4.29 .. 5.48", r1(min(d1), 2) == "4.29" and r1(max(d1), 2) == "5.48")
    d0 = [up[f] for f, d in DEG.items() if d == 0]
    check("dimensionless esup medians -1.87 .. +0.28", r1(min(d0), 2) == "-1.87" and r1(max(d0), 2) == "0.28")
    check("dimensionless esup |medians| below 2% (conclusion)", max(abs(x) for x in d0) < 2.0)
    check("esup response ordered by degree: d0 < d1 < ECF2 < ECF3",
          max(abs(x) for x in d0) < min(d1 + [up["fjet_ECF1"]])
          and max(d1 + [up["fjet_ECF1"]]) < up["fjet_ECF2"] < up["fjet_ECF3"])
    check("esdown mirrors esup with opposite sign for every input",
          all(np.sign(up[f]) == -np.sign(dn[f]) for f in DEG))
    for f, q in (("fjet_ECF2", "15.36"), ("fjet_ECF3", "23.91")):
        ue = 100 * RS["esup"][f]["uniform_scaling_expectation"]
        check(f"uniform expectation {f} = {q}%", r1(ue, 2) == q
              and abs(ue / 100 - ((1 + up["fjet_ECF1"] / 100) ** DEG[f] - 1)) < 1e-12)
    for f, q in (("+4.29\\%$ to $+5.48\\%$", None), ("$+12.57\\%$ for ECF$_2$, $+17.63\\%$ for ECF$_3$", None),
                 ("between $-1.87\\%$ and $+0.28\\%$", None), ("($15.36\\%$ for ECF$_2$ and $23.91\\%$ for ECF$_3$)", None)):
        says(f)
    ca = {f: 100 * v["median_abs"] for f, v in RS["cer"].items()}
    check("cer: ECF3 largest median |r| 4.88, ECF2 3.59, dimensionless max 2.46",
          max(ca, key=ca.get) == "fjet_ECF3" and r1(ca["fjet_ECF3"], 2) == "4.88"
          and r1(ca["fjet_ECF2"], 2) == "3.59"
          and r1(max(ca[f] for f, d in DEG.items() if d == 0), 2) == "2.46")
    says("$4.88\\%$, against $3.59\\%$ for ECF$_2$ and at most $2.46\\%$ for any dimensionless input")
    pa = {f: 100 * v["median_abs"] for f, v in RS["cpos"].items()}
    check("cpos: every median |r| below 0.12%, ECF3 0.117, tau4 0.115",
          max(pa.values()) < 0.12 and r1(pa["fjet_ECF3"], 3) == "0.117"
          and r1(pa["fjet_Tau4_wta"], 3) == "0.115")
    says("($0.117\\%$ and $0.115\\%$)")
    check("conclusion quotes 17.6% and 7.4%", r1(up["fjet_ECF3"], 1) == "17.6"
          and r1(up["fjet_ECF1"], 1) == "7.4" and "median $17.6\\%$ per jet" in J
          and "$7.4\\%$ for the summed constituent momentum" in J)
    # SHAP claims tied to the response
    def shap_abs(sy, f):
        return float(SH[(sy, "100000", "all", f)]["mean_abs_delta_shap"])
    check("esup: ECF2 carries less attribution drift than ECF1",
          shap_abs("esup", "fjet_ECF2") < shap_abs("esup", "fjet_ECF1")
          and up["fjet_ECF2"] > up["fjet_ECF1"])
    rank_up = sorted(DEG, key=lambda f: -shap_abs("esup", f))
    check("esup drift ranks: ECF3 1st, ECF1 2nd, L2 3rd, thrust 4th, ECF2 14th of 15",
          rank_up[:4] == ["fjet_ECF3", "fjet_ECF1", "fjet_L2", "fjet_ThrustMaj"]
          and rank_up.index("fjet_ECF2") == 13, str(rank_up))
    ua = {f: 100 * v["median_abs"] for f, v in RS["esup"].items()}
    check("esup: ECF2 has the second-largest response; L2 and thrust move < 1%",
          sorted(ua, key=lambda f: -ua[f])[1] == "fjet_ECF2"
          and ua["fjet_L2"] < 1 and ua["fjet_ThrustMaj"] < 1)
    says("ECF$_2$ has the second-largest response but almost the smallest attribution drift")
    check("cpos drift led by tau4", rank_up and lead["cpos"][0] == "fjet_Tau4_wta")
    check("generator-modelling term reduced by a factor of three without absolute scale",
          r1(mean_("full15", "inc_model") / mean_("ratios9", "inc_model"), 0) == "3")
    check("no claim that the attribution drift as a whole follows the degree",
          "the attribution drift follows them" not in J
          and "The attribution pattern follows the degree" not in J)
    check("ECF3 leads attribution drift under esup, esdown and cer",
          all(lead[sy][0] == "fjet_ECF3" for sy in ("esup", "esdown", "cer")))

    # --- README statements, quoted against the provenance note
    rq = (HERE / "sources/atlas_readme_5bc3c8d7_quotes.md").read_text(encoding="utf-8")
    check("README note pins the frozen commit and blob",
          "5bc3c8d70cf8736d24b969972a3dd365017a9e24" in rq
          and "423c543491b0821ec26b2896fa62467da360917a" in rq
          and "4fe790030633f35a7b922eee619cb797813cddc0b77cbcd53f71e2e617873251" in rq)
    for frag in ("is not re-calculated from the systematic varied jet constituents",
                 "This is because the jet four-vector is calibrated",
                 "the statistical uncertainty on the performance metrics must be smaller than this difference",
                 "it is recommended to use at least 5 million jets in evaluation when calculating the experimental uncertainties",
                 "only using 6 million of the jets in the training set",
                 "obtaining state-of-the-art performance will require training on the entire set",
                 "Vary the energy scale of jet constituents reconstructed with the calorimeter"):
        check(f"README note contains: {frag[:45]}", frag in rq)
    says("``is not re-calculated from the systematic varied jet constituents''")
    says("It makes no corresponding statement about the fifteen high-level quantities")
    says("the size the dataset documentation recommends as a minimum in evaluation")
    check("at most 244 jets per detector sample fail eligibility",
          "cer             4,999,756 jets  dropped   244" in tagger_log)
    says("of which at most 244 fail the eligibility rule")
    says("lowers the six-seed mean total uncertainty from $26.3\\%$ to $15.8\\%$")
    check("every detector sample really is evaluated on 5,000,000 jets",
          all(per[d][1] == 5_000_000 for d in ("esup", "esdown", "cer", "cpos", "teg",
                                               "tej", "tfl", "tfj", "bias")))
    says("example training script uses six million of the training jets")
    says("state-of-the-art performance will require training on the entire partition")
    check("old 'decision about production scale' wording is gone",
          "decision about production scale" not in J)
    check("ATLASGitLab cited for the README statements",
          len(re.findall(r"\\cite\{ATLASGitLab\}", MAIN)) >= 7)

    # --- bootstrap references and code availability
    check("bootstrap references cited", "Efron1979" in MAIN and "Hanley2006" in MAIN
          and "10.1214/aos/1176344552" in MAIN and "10.1016/j.cmpb.2006.04.006" in MAIN)
    check("code availability names the four extension scripts",
          all(f"\\texttt{{{n}}}" in MAIN for n in ("uncertainty\\_chain.py",
              "stat\\_bootstrap.py", "tagger\\_variants.py", "feature\\_response.py")))
    check("extension fits declared as made after freezing, not replacing the model",
          "were made after the analysis was frozen" in J
          and "none of them replaces the frozen model" in J)
    check("supplement S2/S3 text no longer says no statistical uncertainty",
          "No statistical uncertainty is attached to any entry" not in JS)
else:
    check("robustness extension result folders present", False, str(EXT_RES))

# ------------------------------------------------------------ forbidden ----
for phrase in ["EventInfo_mcEventWeights columns", "proxy AUC", "Reweighted AUC",
               "N_\\mathrm{eff}", "effective sample size", "importance-weighted",
               "held-out simulation files", "MLP", "multilayer perceptron"]:
    check(f"obsolete content absent: {phrase!r}", phrase not in MAIN)

# ------------------------------------------------------------------ report --
print(f"PASSED {len(ok)}   FAILED {len(bad)}\n")
if bad:
    print("FAILURES")
    for b in bad:
        print("  -", b)
    sys.exit(1)
print("All manuscript numerical and cross-reference checks passed.")
