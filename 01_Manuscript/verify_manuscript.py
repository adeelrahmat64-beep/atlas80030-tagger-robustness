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
check("manuscript says no uncertainty band is attached",
      "no uncertainty band is attached" in " ".join(MAIN.split()))

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
inputs = re.findall(r"\\input\{tables/(?:layout/)?([a-z0-9_]+)\.tex\}", MAIN)
check("main text inputs exactly 3 frozen tables", len(inputs) == 3, str(inputs))
for lab in ["tab:datasets", "tab:pairing", "tab:nominal-performance",
            "tab:inclusive-systematics-wp50", "tab:shap-delta-100k"]:
    check(f"Table label {lab} resolves", lab in labels)
    check(f"Table {lab} cited", f"\\ref{{{lab}}}" in MAIN)
for n, lab in enumerate(["tab:datasets", "tab:pairing", "tab:nominal-performance",
                         "tab:inclusive-systematics-wp50", "tab:shap-delta-100k"], 1):
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
                         "tab:supp-shap-scores"], 1):
    num = re.search(rf"\\newlabel\{{{re.escape(lab)}\}}\{{\{{(S\d+)\}}", saux)
    check(f"supplementary {lab} numbered S{n}",
          num is not None and num.group(1) == f"S{n}", num.group(1) if num else "?")
for n in range(1, 8):
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
check("code availability statement uses the EPJ C wording",
      "Code/software will be made available on reasonable request" in joined_scope)

# point 17: statistical extras are named as future work, not claimed
check("bootstrap / multi-seed / training-block extras flagged as next steps",
      "Bootstrap confidence intervals" in joined_scope
      and "natural next step" in joined_scope)



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
