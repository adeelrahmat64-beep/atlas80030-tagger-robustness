"""ATLAS 80030 - performance metrics and systematic uncertainties, ATLAS's way.

This is step 3, and it is deliberately a faithful port of ATLAS's own
calc.py and plot_everything.py from

    https://gitlab.cern.ch/atlas/ATLAS-top-tagging-open-data

rather than a method of our own. The record's README is explicit: "This script
implements the recommended procedure for setting systematic uncertainties using
the raw performance metrics generated in step 3. Other methods of setting
systematic uncertainties with this dataset are not supported." Every formula
below is transcribed from that source and the transcription is noted inline, so
a referee can check it line by line against the published code.

NO PAIRING IS INVOLVED, BY DESIGN
    ATLAS compares whole datasets, not matched jets: the uncertainty is
    |systematic_br / nominal_br - 1|, evaluated per pT bin. Jet-by-jet pairing
    is needed only for the SHAP-delta and conformal parts of our own study.
    That is why a dataset which will not pair - bias - is still perfectly
    usable here.

WHAT IT READS
    <data-root>\\predictions\\public_<dataset>.npz, written by
    atlas_tagger.py with keys 'predictions', 'labels', 'pt', and
    'shower_weights' on the nominal test set.

WHAT IT WRITES
    <data-root>\\predictions\\<dataset>_metrics.npz  (per dataset)
    <data-root>\\predictions\\uncertainty_summary.npz
    Plus a round-numbered log in Code files\\analysis_results\\.

MISSING DATASETS ARE REPORTED, NOT GUESSED
    The six modelling datasets - ttbar_pythia, ttbar_herwig, cluster, string,
    angular, dipole - are needed for the theoretical uncertainties, which ATLAS
    finds DOMINANT. If they are absent the script computes everything it can,
    prints exactly which components are missing, and refuses to present a
    "total" that silently omits them. A total uncertainty that quietly drops
    the dominant term is the single most likely thing for a referee to catch.

RUNNING IT
    python "H:\\HEP\\Research Papers\\ATLAS 80030\\Code files\\atlas_metrics.py"
    Needs only numpy and scikit-learn. Runs in a couple of minutes.
"""

import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from atlas_config import DATA_ROOT, EVALUATION_DATASETS, FROZEN_RECONSTRUCTION_FP

# ----------------------------------------------------------------- config ---
STORE = DATA_ROOT
HERE = Path(__file__).resolve().parent
PRED = STORE / "predictions"
MODELS = STORE / "models"
RESULTS = HERE / "analysis_results"

# Metrics are authorised only for the frozen 5M final-production prediction
# campaign.  The 4M pilot remains available to atlas_tagger.py's pilot-resume
# mode but can never be selected by this metrics entrypoint.
FINAL_MODEL_ROUND = 1
FINAL_MODEL_STEM = f"atlas_tagger_xgb_final_production_round{FINAL_MODEL_ROUND}"
FINAL_MODEL_PATH = MODELS / f"{FINAL_MODEL_STEM}.joblib"
FINAL_METADATA_PATH = MODELS / f"{FINAL_MODEL_STEM}_metadata.json"
FINAL_SPLIT_PATH = MODELS / f"{FINAL_MODEL_STEM}_split.npz"
FINAL_MANIFEST_PATH = PRED / f"{FINAL_MODEL_STEM}_prediction_manifest.json"
OFFICIAL_REFERENCE_COMMIT = "5bc3c8d70cf8736d24b969972a3dd365017a9e24"
OFFICIAL_REPOSITORY = "https://gitlab.cern.ch/atlas/ATLAS-top-tagging-open-data"
OPEN_DATA_RECORD = "https://opendata.cern.ch/record/80030"
ATLAS_PAPER = "arXiv:2407.20127; JINST 19 (2024) P08018"
FROZEN_HL = [
    "fjet_Tau1_wta", "fjet_Tau2_wta", "fjet_Tau3_wta", "fjet_Tau4_wta",
    "fjet_Split12", "fjet_Split23", "fjet_ECF1", "fjet_ECF2", "fjet_ECF3",
    "fjet_C2", "fjet_D2", "fjet_L2", "fjet_L3", "fjet_Qw",
    "fjet_ThrustMaj",
]

# calc.py: full_bins = np.linspace(350000, 3150000, 15)   [MeV]
FULL_BINS = np.linspace(350000, 3150000, 15)
# calc.py: wp = 0.5. ATLAS also quotes 0.8 in the paper's Table 4, and
# plot_everything.py only draws its benchmark comparison when wp == 0.5.
WORKING_POINTS = [0.5, 0.8]
PRIMARY_WP = 0.5

# calc.py's shower-weight column map. Column 0 is the nominal MC weight and a
# variation is applied as shower_weights[:, col] / shower_weights[:, 0].
# Only these four ISR/FSR scale variations are combined by the frozen ATLAS
# calc.py/plot_everything.py prescription, separately for signal/background.
# PDF columns and other shower-weight columns are not part of that prescribed
# tagger-uncertainty total. No shower-weight column is a detector systematic.
SCALE_COLS = {"ISRx2": 4, "ISRxp5": 9, "FSRx2": 6, "FSRxp5": 7}

# Transcribed from calc.py's metrics_dict. The modelling datasets contain only
# one class - ttbar_pythia/herwig are all signal, cluster/string/angular/dipole
# are all background - so calc.py appends the NOMINAL jets of the missing class
# before computing a ROC. Without that there is only one class present and
# every metric is undefined. 'extrapolate' limits the binned measurement to
# below 1.5 TeV, because the SM ttbar samples run out of statistics above it.
SPEC = {
    "test_nominal": {},
    "esup": {}, "esdown": {}, "cer": {}, "cpos": {},
    "teg": {}, "tej": {}, "tfl": {}, "tfj": {}, "bias": {},
    "ttbar_pythia": {"add_nominal": 0, "extrapolate": True},
    "ttbar_herwig": {"add_nominal": 0, "extrapolate": True},
    "cluster": {"add_nominal": 1}, "string": {"add_nominal": 1},
    "angular": {"add_nominal": 1}, "dipole": {"add_nominal": 1},
}

line = "=" * 78
sub = "-" * 78


def file_sha256(path):
    """SHA-256 used to bind metrics to the frozen final production model."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_final_prediction_manifest():
    """Refuse pilot, mixed, or unprovenanced prediction inputs."""
    for path in (FINAL_MODEL_PATH, FINAL_METADATA_PATH, FINAL_SPLIT_PATH,
                 FINAL_MANIFEST_PATH):
        if not path.is_file():
            raise SystemExit(f"final metrics provenance artifact is missing: {path}")
    manifest = json.loads(FINAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    metadata = json.loads(FINAL_METADATA_PATH.read_text(encoding="utf-8"))
    required_metadata = {
        "model_type": "xgb",
        "run_role": "final_production",
        "training_rows_requested": 5_000_000,
        "reconstruction_fingerprint": FROZEN_RECONSTRUCTION_FP,
        "seed": 42,
    }
    bad_metadata = [
        key for key, value in required_metadata.items() if metadata.get(key) != value
    ]
    if metadata.get("feature_order") != FROZEN_HL:
        bad_metadata.append("feature_order")
    if metadata.get("model_sha256") != file_sha256(FINAL_MODEL_PATH):
        bad_metadata.append("model_sha256")
    if metadata.get("split_sha256") != file_sha256(FINAL_SPLIT_PATH):
        bad_metadata.append("split_sha256")
    if bad_metadata:
        raise SystemExit(f"final model metadata verification failed: {bad_metadata}")
    expected_manifest = {
        "run_role": "final_production",
        "final_model_path": str(FINAL_MODEL_PATH),
        "model_sha256": metadata["model_sha256"],
        "tagger_source_sha256": metadata["tagger_source_sha256"],
        "reconstruction_fingerprint": FROZEN_RECONSTRUCTION_FP,
        "training_rows_requested": 5_000_000,
        "seed": 42,
    }
    bad_manifest = [
        key for key, value in expected_manifest.items() if manifest.get(key) != value
    ]
    if set(manifest.get("datasets", {})) != set(EVALUATION_DATASETS):
        bad_manifest.append("datasets")
    if bad_manifest:
        raise SystemExit(f"final prediction manifest verification failed: {bad_manifest}")
    return manifest, metadata


class Tee:
    """Round-numbered log, never overwritten. Same convention as the others."""

    def __init__(self, tag):
        RESULTS.mkdir(parents=True, exist_ok=True)
        used = [0]
        for p in RESULTS.glob(f"ATLAS80030_{tag}_round*"):
            m = re.search(r"round(\d+)", p.name)
            if m:
                used.append(int(m.group(1)))
        self.round = max(used)
        while True:
            self.round += 1
            stem = RESULTS / f"ATLAS80030_{tag}_round{self.round}"
            self.path = stem.with_suffix(".txt")
            self.code = Path(f"{stem}_code.py")
            if not self.path.exists() and not self.code.exists():
                break
        self.code.write_bytes(Path(__file__).read_bytes())
        self.fh = self.path.open("w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.fh.write(s)

    def flush(self):
        self.out.flush()
        self.fh.flush()


def background_rejection(labels, preds, weights, wp):
    """Background rejection at a fixed signal efficiency.

    Transcribed from calc.py:
        fpr, tpr, thresholds = metrics.roc_curve(labels, preds, sample_weight=weights)
        fprinv = 1 / fpr
        index = np.argmax(tpr > wp)
        br = fprinv[index]

    np.argmax returns 0 when nothing satisfies the condition, which would
    silently hand back the rejection at the wrong working point, so that case
    is reported rather than returned.
    """
    from sklearn.metrics import roc_curve
    if labels.size == 0 or len(np.unique(labels)) < 2:
        raise ValueError("background rejection requires both signal and background")
    fpr, tpr, _ = roc_curve(labels, preds, sample_weight=weights)
    if not (tpr > wp).any():
        raise ValueError(f"ROC never exceeds the {wp} signal-efficiency working point")
    with np.errstate(divide="ignore"):
        fprinv = 1.0 / fpr
    index = int(np.argmax(tpr > wp))
    br = float(fprinv[index])
    if not np.isfinite(br):
        raise ValueError("zero/non-finite background efficiency at working point")
    return br, float(tpr[index])


def metrics_for(labels, preds, weights, pt, wp, bins):
    """Inclusive and pT-binned metrics for one dataset, exactly as calc.py."""
    from sklearn.metrics import roc_auc_score, accuracy_score
    out = {}
    out["auc"] = float(roc_auc_score(labels, preds, sample_weight=weights))
    out["acc"] = float(accuracy_score(labels, (preds > 0.5).astype(int),
                                      sample_weight=weights))
    out["br"], _ = background_rejection(labels, preds, weights, wp)

    binned = np.full(len(bins), np.nan)
    bin_counts = []
    for i in range(len(bins) - 1):
        sel = (pt > bins[i]) & (pt < bins[i + 1])
        counts = np.bincount(labels[sel].astype(int), minlength=2)
        bin_counts.append(counts.tolist())
        if not sel.any() or (counts == 0).any():
            raise ValueError(
                f"pT bin ({bins[i]}, {bins[i + 1]}) has insufficient class counts "
                f"{counts.tolist()}"
            )
        binned[i], _ = background_rejection(labels[sel], preds[sel],
                                            weights[sel], wp)
    # calc.py repeats the last value so the step plot closes.
    binned[-1] = binned[-2]
    out["binned_br"] = binned
    out["n"] = int(labels.size)
    # A bin with no surviving background jet gives 1/0 = inf, and a bin that
    # never reaches the working point gives NaN. ATLAS's own figures note that
    # "fluctuations in the uncertainty result from the finite statistics of the
    # nominal and systematic varied datasets"; this counts them explicitly so a
    # thin bin is visible rather than silently poisoning a quadrature sum.
    if not (np.isfinite(out["auc"]) and np.isfinite(out["acc"]) and
            np.isfinite(out["br"]) and np.isfinite(binned).all()):
        raise ValueError("non-finite metric result")
    out["n_bad_bins"] = 0
    out["bin_class_counts"] = bin_counts
    return out


def load(ds, manifest):
    p = PRED / f"public_{ds}.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    required = {"predictions", "labels", "pt"}
    if ds == "test_nominal":
        required.add("shower_weights")
    missing = sorted(required - set(d))
    if missing:
        raise SystemExit(f"{p}: missing ATLAS prediction fields {missing}")
    shapes = {np.asarray(d[key]).shape for key in ("predictions", "labels", "pt")}
    if len(shapes) != 1 or len(next(iter(shapes))) != 1:
        raise SystemExit(f"{p}: predictions, labels, and pt must be aligned vectors")
    rows = int(np.asarray(d["predictions"]).size)
    if manifest["datasets"][ds].get("prediction_rows") != rows:
        raise SystemExit(f"{p}: row count does not match final prediction manifest")
    labels = np.asarray(d["labels"])
    if not np.isin(labels, (0, 1)).all():
        raise SystemExit(f"{p}: labels are not binary")
    if not (np.isfinite(d["predictions"]).all() and np.isfinite(d["pt"]).all()):
        raise SystemExit(f"{p}: non-finite predictions or pT values")
    return d


def rel(a, b):
    """|a/b - 1|, refusing invalid denominators rather than masking them."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("non-finite background rejection in systematic ratio")
    if (b == 0).any():
        raise ValueError("zero nominal background rejection in systematic ratio")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.abs(a / b - 1.0)
    if not np.isfinite(r).all():
        raise ValueError("non-finite relative systematic uncertainty")
    return r


def env(a, b):
    """Envelope of two variations - plot_everything.py uses np.maximum."""
    return np.maximum(a, b)


def quad(*parts):
    """Quadrature sum."""
    values = [np.asarray(p, dtype=np.float64) for p in parts]
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError("non-finite uncertainty component in quadrature sum")
    return np.sqrt(sum(value ** 2 for value in values))


def _json_value(value):
    """Convert NumPy scalars/arrays for the metrics audit sidecar."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def write_publication_artifacts(tee, manifest, metadata, metrics, summary):
    """Write human- and machine-readable metrics sidecars with final provenance."""
    stem = PRED / f"final_metrics_{FINAL_MODEL_STEM}_round{tee.round}"
    rows = []
    for (dataset, wp), result in sorted(metrics.items()):
        rows.append({
            "dataset": dataset,
            "working_point": wp,
            "prediction_rows": result["n"],
            "auc": result["auc"],
            "accuracy_score_threshold_0p5": result["acc"],
            "background_rejection": result["br"],
            "invalid_pt_bins": result["n_bad_bins"],
        })
    provenance = {
        "final_prediction_manifest": str(FINAL_MANIFEST_PATH),
        "final_model_path": metadata["model_path"],
        "model_sha256": metadata["model_sha256"],
        "split_sha256": metadata["split_sha256"],
        "tagger_source_sha256": metadata["tagger_source_sha256"],
        "reconstruction_fingerprint": metadata["reconstruction_fingerprint"],
        "training_rows_requested": metadata["training_rows_requested"],
        "seed": metadata["seed"],
        "prediction_rows": manifest["datasets"],
        "official_repository": OFFICIAL_REPOSITORY,
        "official_reference_commit": OFFICIAL_REFERENCE_COMMIT,
        "open_data_record": OPEN_DATA_RECORD,
        "ATLAS_paper": ATLAS_PAPER,
        "metrics_generation_timestamp": datetime.now().astimezone().isoformat(),
    }
    payload = {
        "provenance": provenance,
        "metrics": [
            {"dataset": dataset, "working_point": wp, **_json_value(result)}
            for (dataset, wp), result in sorted(metrics.items())
        ],
        "uncertainty_summary": _json_value(summary),
    }
    json_path = Path(f"{stem}.json")
    csv_path = Path(f"{stem}.csv")
    txt_path = Path(f"{stem}.txt")
    if any(path.exists() for path in (json_path, csv_path, txt_path)):
        raise FileExistsError(f"refusing to overwrite final metrics audit artifact {stem}")
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tee.fh.flush()
    txt_path.write_text(
        "FINAL METRICS PROVENANCE\n"
        + json.dumps(provenance, indent=2)
        + "\n\nMETRICS LOG\n"
        + tee.path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return txt_path, json_path, csv_path


def main():
    if not PRED.is_dir():
        raise SystemExit(f"predictions folder not found:\n  {PRED}\n"
                         "run atlas_tagger.py first")
    manifest, metadata = require_final_prediction_manifest()
    tee = Tee("metrics")
    sys.stdout = tee
    try:
        print(f"# ATLAS 80030 - atlas_metrics round {tee.round} - "
              f"{datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"# final model {metadata['model_path']}")
        print(f"# final model SHA-256 {metadata['model_sha256']}")
        print(f"# tagger source SHA-256 {metadata['tagger_source_sha256']}")
        print(f"# pT bins {FULL_BINS[0]/1000:.0f}-{FULL_BINS[-1]/1000:.0f} GeV "
              f"in {len(FULL_BINS)-1} bins; working points {WORKING_POINTS}")

        nom = load("test_nominal", manifest)
        if nom is None:
            raise SystemExit("public_test_nominal.npz missing - run atlas_tagger.py")

        detector = ["esup", "esdown", "cer", "cpos", "teg", "tej", "tfl", "tfj", "bias"]
        modelling = ["ttbar_pythia", "ttbar_herwig", "cluster", "string",
                     "angular", "dipole"]

        # ------------------------------------------------ per-dataset metrics ---
        M = {}
        for wp in WORKING_POINTS:
            print(f"\n{line}\nperformance at signal efficiency {wp}\n{line}")
            print(f"  {'dataset':<16}{'jets':>12}{'AUC':>10}{'ACC':>10}{'1/eps_bkg':>12}")
            for ds in ["test_nominal"] + detector + modelling:
                d = nom if ds == "test_nominal" else load(ds, manifest)
                if d is None:
                    continue
                spec = SPEC.get(ds, {})
                lab, prd, pt = d["labels"], d["predictions"], d["pt"]

                # calc.py: append the nominal jets of the class this dataset
                # does not contain, so a ROC can be formed at all.
                if "add_nominal" in spec:
                    t = spec["add_nominal"]
                    sel = nom["labels"] == t
                    prd = np.concatenate((prd, nom["predictions"][sel]))
                    pt = np.concatenate((pt, nom["pt"][sel]))
                    lab = np.concatenate((lab, np.full(int(sel.sum()), t,
                                                       dtype=lab.dtype)))

                # calc.py: the SM ttbar samples are measured only below 1.5 TeV
                bins = FULL_BINS
                if spec.get("extrapolate"):
                    bins = FULL_BINS[:int(np.argwhere(FULL_BINS > 1500000)[0][0])]

                w = np.ones(lab.shape)              # calc.py: unweighted
                m = metrics_for(lab, prd, w, pt, wp, bins)
                m["pt_bins"] = bins
                M[(ds, wp)] = m
                flag = "" if m["n_bad_bins"] == 0 else f"   {m['n_bad_bins']} empty bins"
                print(f"  {ds:<16}{m['n']:>12,}{m['auc']:>10.4f}"
                      f"{m['acc']:>10.4f}{m['br']:>12.2f}{flag}")

        # ------------------------------------------------ theory variations ---
        # calc.py applies shower-weight ratios to the NOMINAL predictions,
        # separately for signal and for background.
        have_sw = "shower_weights" in nom
        if have_sw:
            print(f"\n{line}\nscale variations from the shower weights "
                  f"(nominal predictions, reweighted)\n{line}")
            sw = np.asarray(nom["shower_weights"], dtype=np.float64)
            nw = sw[:, 0]
            lab = nom["labels"]
            for target, tname in ((1, "sig"), (0, "bkg")):
                for vname, col in SCALE_COLS.items():
                    w = np.ones(lab.shape)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = sw[:, col] / nw
                    if not np.isfinite(ratio).all():
                        raise ValueError(
                            f"non-finite shower-weight ratio in column {col}; "
                            "refusing to substitute a value"
                        )
                    w[lab == target] = ratio[lab == target]
                    key = f"{tname}_{vname}"
                    M[(key, PRIMARY_WP)] = metrics_for(
                        lab, nom["predictions"], w, nom["pt"],
                        PRIMARY_WP, FULL_BINS)
                    M[(key, PRIMARY_WP)]["pt_bins"] = FULL_BINS
                    print(f"  {key:<16} 1/eps_bkg = "
                          f"{M[(key, PRIMARY_WP)]['br']:.2f}")
        else:
            print("\n*** shower_weights absent from public_test_nominal.npz - "
                  "the four scale uncertainties cannot be computed ***")

        # -------------------------------------------- uncertainty combination ---
        # Every line below is transcribed from plot_everything.py.
        print(f"\n{line}\nsystematic uncertainties at signal efficiency "
              f"{PRIMARY_WP}\n{line}")
        wp = PRIMARY_WP
        nb = M[("test_nominal", wp)]["binned_br"]
        ni = M[("test_nominal", wp)]["br"]

        def U(ds):
            m = M.get((ds, wp))
            if m is None:
                return None, None
            return rel(m["binned_br"], nb), rel(m["br"], ni)

        missing = []
        comp, inc = {}, {}
        for ds in detector:
            b, i = U(ds)
            if b is None:
                missing.append(ds)
            else:
                comp[ds], inc[ds] = b, i

        cluster_ok = all(k in comp for k in ("esup", "esdown", "cer", "cpos"))
        track_ok = all(k in comp for k in ("teg", "tej", "tfl", "tfj", "bias"))

        if cluster_ok:
            es = env(comp["esup"], comp["esdown"])          # envelope
            inc_es = env(inc["esup"], inc["esdown"])
            cluster = quad(es, comp["cer"], comp["cpos"])
            inc_cluster = quad(inc_es, inc["cer"], inc["cpos"])
            print(f"  cluster energy scale (envelope esup/esdown)  "
                  f"inclusive {100*inc_es:6.2f}%")
            print(f"  cluster energy resolution (cer)              "
                  f"inclusive {100*inc['cer']:6.2f}%")
            print(f"  cluster position resolution (cpos)           "
                  f"inclusive {100*inc['cpos']:6.2f}%")
            print(f"  CLUSTER TOTAL                                "
                  f"inclusive {100*inc_cluster:6.2f}%")
        if track_ok:
            eff = env(comp["teg"], comp["tej"])
            inc_eff = env(inc["teg"], inc["tej"])
            fake = env(comp["tfl"], comp["tfj"])
            inc_fake = env(inc["tfl"], inc["tfj"])
            track = quad(eff, fake, comp["bias"])
            inc_track = quad(inc_eff, inc_fake, inc["bias"])
            print(f"  track efficiency (envelope teg/tej)          "
                  f"inclusive {100*inc_eff:6.2f}%")
            print(f"  track fake rate (envelope tfl/tfj)           "
                  f"inclusive {100*inc_fake:6.2f}%")
            print(f"  track bias                                   "
                  f"inclusive {100*inc['bias']:6.2f}%")
            print(f"  TRACK TOTAL                                  "
                  f"inclusive {100*inc_track:6.2f}%")

        # Modelling: signal from herwig/pythia, background from the two pairs.
        model_ok = all((d, wp) in M for d in modelling)
        if model_ok:
            sig_model = rel(M[("ttbar_herwig", wp)]["binned_br"],
                            M[("ttbar_pythia", wp)]["binned_br"])
            inc_sig_model = rel(M[("ttbar_herwig", wp)]["br"],
                                M[("ttbar_pythia", wp)]["br"])
            bkg_ps = rel(M[("dipole", wp)]["binned_br"], M[("angular", wp)]["binned_br"])
            inc_bkg_ps = rel(M[("dipole", wp)]["br"], M[("angular", wp)]["br"])
            bkg_had = rel(M[("cluster", wp)]["binned_br"], M[("string", wp)]["binned_br"])
            inc_bkg_had = rel(M[("cluster", wp)]["br"], M[("string", wp)]["br"])
            # plot_everything.py extrapolates signal modelling above 1.5 TeV by
            # assigning the maximum measured value.
            ext = np.zeros_like(bkg_ps)
            mb = int(np.argwhere(FULL_BINS > 1500000)[0][0])
            ext[:mb] = sig_model[:mb]
            ext[mb:] = np.nanmax(sig_model[:mb])
            print(f"  (signal modelling measured in {mb} bins below 1.5 TeV, "
                  f"then held at its maximum {100*np.nanmax(sig_model[:mb]):.2f}% above)")
            bkg_model = quad(bkg_ps, bkg_had)
            total_model = quad(ext, bkg_model)
            inc_total_model = quad(inc_sig_model, quad(inc_bkg_ps, inc_bkg_had))
            print(f"  signal modelling (herwig/pythia, extrapolated >1.5 TeV) "
                  f"inclusive {100*inc_sig_model:6.2f}%")
            print(f"  background modelling                         "
                  f"inclusive {100*quad(inc_bkg_ps, inc_bkg_had):6.2f}%")
        else:
            absent = [d for d in modelling if (d, wp) not in M]
            print(f"\n  *** MODELLING UNCERTAINTIES NOT AVAILABLE ***")
            print(f"      missing datasets: {', '.join(absent)}")
            print(f"      ATLAS finds these DOMINANT for every tagger, so no")
            print(f"      total uncertainty is reported below. Download them")
            print(f"      from record 80030 before quoting a total.")

        if have_sw:
            scale_parts, inc_scale_parts = [], []
            for tname in ("sig", "bkg"):
                for a, b in (("ISRx2", "ISRxp5"), ("FSRx2", "FSRxp5")):
                    ka, kb = f"{tname}_{a}", f"{tname}_{b}"
                    ua = rel(M[(ka, wp)]["binned_br"], nb)
                    ub = rel(M[(kb, wp)]["binned_br"], nb)
                    ia = rel(M[(ka, wp)]["br"], ni)
                    ib = rel(M[(kb, wp)]["br"], ni)
                    scale_parts.append(env(ua, ub))
                    inc_scale_parts.append(env(ia, ib))
                    print(f"  scale {tname} {a[:3]:<4} (envelope up/down)          "
                          f"inclusive {100*env(ia, ib):6.2f}%")
            scale = quad(*scale_parts)
            inc_scale = quad(*inc_scale_parts)
            print(f"  SCALE TOTAL                                  "
                  f"inclusive {100*inc_scale:6.2f}%")

        save = {"pt_bins": FULL_BINS, "wp": wp,
                "nominal_binned_br": nb, "nominal_br": ni}
        if cluster_ok:
            save.update(cluster=cluster, inc_cluster=inc_cluster)
        if track_ok:
            save.update(track=track, inc_track=inc_track)
        if have_sw:
            save.update(scale=scale, inc_scale=inc_scale)
        if model_ok:
            save.update(model=total_model, inc_model=inc_total_model)

        if cluster_ok and track_ok and have_sw and model_ok:
            total = quad(cluster, track, total_model, scale)
            inc_total = quad(inc_cluster, inc_track, inc_total_model, inc_scale)
            save.update(total=total, inc_total=inc_total)
            print(f"\n  TOTAL UNCERTAINTY                            "
                  f"inclusive {100*inc_total:6.2f}%")
        else:
            print(f"\n  TOTAL not reported: components missing "
                  f"(cluster {cluster_ok}, track {track_ok}, "
                  f"scale {have_sw}, modelling {model_ok})")

        for (ds, w_), m in M.items():
            if w_ == wp:
                np.savez(PRED / f"{ds}_metrics.npz",
                         pt_bins=m.get("pt_bins", FULL_BINS), wp=wp,
                         br=m["br"], binned_br=m["binned_br"],
                         auc=m["auc"], acc=m["acc"],
                         n_bad_bins=m["n_bad_bins"])
        np.savez(PRED / "uncertainty_summary.npz", **save)
        txt_path, json_path, csv_path = write_publication_artifacts(
            tee, manifest, metadata, M, save
        )
        print(f"  audit summary   {txt_path}")
        print(f"  audit JSON      {json_path}")
        print(f"  audit CSV       {csv_path}")
        print(f"\n{line}\ndone. metrics in {PRED}\n{line}")
    finally:
        sys.stdout = tee.out
        tee.fh.close()
    print(f"log: {tee.path}")


if __name__ == "__main__":
    main()
