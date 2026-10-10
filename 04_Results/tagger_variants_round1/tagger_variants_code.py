"""ATLAS 80030 - training-seed spread and a scale-invariant feature set.

WHAT IT ANSWERS
    1. Training-seed dependence.  The frozen paper uses one XGBoost model
       (random_state 42).  How much do AUC, background rejection (BR) and every
       systematic-uncertainty component move when only the training seed
       changes?  (ATLAS reports the same kind of spread, "train_dev", for its
       hlDNN baseline.)
    2. Scale invariance.  Six of the fifteen inputs carry powers of momentum
       (SCALE_DEGREE below); the other nine are dimensionless ratios.  A
       cluster energy-scale shift moves the dimensionful inputs and, to first
       order, not the ratios.  A tagger trained on the nine ratios alone should
       therefore show a smaller energy-scale uncertainty, at some cost in
       performance.  This script measures both, for two scale-free input
       sets (FEATURE_SETS below).

WHAT IS HELD FIXED (identical to the frozen final model)
    The same 5,000,000 train_nominal jets, the same eligibility rule, the same
    event-level split (SplitMix64 with split seed 42, imported unchanged from
    atlas_tagger.py and checked against the frozen saved split), the same
    training weights, the same XGBoost hyper-parameters, and the same frozen
    feature cache.  Only the XGBoost random_state, and for the second study the
    input columns, change.

    Evaluation uses exactly the frozen eligible rows: labels, pT and shower
    weights are taken from the frozen prediction files after checking, row by
    row, that they equal the cache.  Metrics and the uncertainty combination
    are computed by uncertainty_chain.py, which calls atlas_metrics.py's own
    functions and is checked against the frozen metrics before use.

CONTROL
    The full 15-feature model with random_state 42 is retrained as a control.
    Its predictions are compared with the frozen predictions; the log states
    whether they are bit-identical.

SAFETY
    Nothing under ATLAS_80030_Data is written.  Models and results go to a new
    folder 04_Results/tagger_variants_round<N>/ that is never reused.

RUN
    python tagger_variants.py                  # seeds 42 (control) + 1..5
    python tagger_variants.py --seeds 1 2 3    # fewer extra seeds
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import atlas_metrics as am
import atlas_tagger as at
import uncertainty_chain as uc
from atlas_config import RESULTS_ROOT

# Power of the constituent momentum scale carried by each input, read off the
# definitions in recompute_hl.py (produce mode): under p -> lambda p for every
# constituent, a feature of degree k scales as lambda^k.
#   Tau_n  = tau_n / sum(pt)                         degree 0
#   Split12, Split23 (kT splitting scales)           degree 1
#   ECF1 = sum pt, ECF2, ECF3                        degree 1, 2, 3
#   C2 = ECF3 ECF1 / ECF2^2, D2 = ECF3 ECF1^3/ECF2^3 degree 0
#   L2, L3 (normalised generalised correlators)      degree 0
#   Qw (smallest subjet-pair mass)                   degree 1
#   ThrustMaj = max sum|p.n| / sum|p|                degree 0
SCALE_DEGREE = {
    "fjet_Tau1_wta": 0, "fjet_Tau2_wta": 0, "fjet_Tau3_wta": 0, "fjet_Tau4_wta": 0,
    "fjet_Split12": 1, "fjet_Split23": 1,
    "fjet_ECF1": 1, "fjet_ECF2": 2, "fjet_ECF3": 3,
    "fjet_C2": 0, "fjet_D2": 0, "fjet_L2": 0, "fjet_L3": 0,
    "fjet_Qw": 1, "fjet_ThrustMaj": 0,
}
_RATIOS = [f for f in at.HL if SCALE_DEGREE[f] == 0]
_DIMENSIONFUL = [f for f in at.HL if SCALE_DEGREE[f] > 0 and f != "fjet_ECF1"]
# Three input sets, all built from the same frozen fifteen columns:
#   full15           the frozen model's inputs, unchanged;
#   ratios9          only the nine inputs that are dimensionless by construction;
#   pt_normalised14  the nine ratios plus every dimensionful input divided by
#                    ECF1 (= sum of constituent pT) to its own power,
#                    X / ECF1^k.  This keeps all the shape information of the
#                    fifteen inputs and removes only the overall momentum
#                    scale, which is what a cluster energy-scale shift moves.
FEATURE_SETS = {
    "full15": list(at.HL),
    "ratios9": list(_RATIOS),
    "pt_normalised14": list(_RATIOS) + [f"{f}/ECF1^{SCALE_DEGREE[f]}" for f in _DIMENSIONFUL],
}


def build_features(name, X):
    """Input matrix for one feature set from the frozen 15-column matrix."""
    if X.ndim != 2 or X.shape[1] != len(at.HL):
        raise ValueError(f"expected the frozen {len(at.HL)}-column matrix, got {X.shape}")
    col = {f: X[:, i] for i, f in enumerate(at.HL)}
    if name == "full15":
        return X
    if name == "ratios9":
        return np.stack([col[f] for f in _RATIOS], axis=1)
    if name == "pt_normalised14":
        e1 = col["fjet_ECF1"]
        if not (e1 > 0).all():
            raise ValueError("ECF1 must be positive for every eligible jet")
        extra = [col[f] / e1 ** SCALE_DEGREE[f] for f in _DIMENSIONFUL]
        M = np.stack([col[f] for f in _RATIOS] + extra, axis=1)
        if not np.isfinite(M).all():
            raise ValueError("non-finite normalised feature")
        return M
    raise ValueError(f"unknown feature set {name}")
CONTROL_SEED = 42
DEFAULT_EXTRA_SEEDS = [1, 2, 3, 4, 5]
XGB_PARAMS = dict(n_estimators=500, max_depth=6, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8,
                  eval_metric="logloss", tree_method="hist", n_jobs=-1)
FROZEN_JSON = am.PRED / f"final_metrics_{am.FINAL_MODEL_STEM}_round1.json"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def new_output_dir(tag):
    used = [0]
    for p in RESULTS_ROOT.glob(f"{tag}_round*"):
        m = re.fullmatch(rf"{tag}_round(\d+)", p.name)
        if m:
            used.append(int(m.group(1)))
    out = RESULTS_ROOT / f"{tag}_round{max(used) + 1}"
    out.mkdir(parents=True, exist_ok=False)
    return out


class Tee:
    def __init__(self, path):
        self.fh = open(path, "a", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.fh.write(s)
        self.fh.flush()

    def flush(self):
        self.out.flush()
        self.fh.flush()


def check_frozen_xgb_params():
    """The hyper-parameters here must equal the frozen build_model()."""
    frozen = at.build_model().get_params()
    for k, v in XGB_PARAMS.items():
        if frozen.get(k) != v:
            raise SystemExit(f"hyper-parameter {k}: {v!r} here vs {frozen.get(k)!r} frozen")
    if frozen.get("random_state") != CONTROL_SEED:
        raise SystemExit("frozen random_state is not 42")


def load_training_set():
    """Load the frozen training rows and split, and prove they are the frozen ones."""
    X, y, _pt, ev, sizes, keep, stems = at.load_features(
        "train_nominal", at.N_TRAIN_JETS, require_event_number=True)
    w = at.load_h5_column("train_nominal", stems, sizes, "training_weights")
    if w.shape != y.shape or not np.isfinite(w).all() or (w < 0).any():
        raise SystemExit("training weights are misaligned, non-finite, or negative")
    with np.load(am.FINAL_SPLIT_PATH, allow_pickle=False) as split:
        if list(split["file_stems"]) != list(stems):
            raise SystemExit("training files differ from the frozen split")
        if list(split["file_rows"]) != list(sizes):
            raise SystemExit("training rows per file differ from the frozen split")
        if not np.array_equal(split["eligibility_mask"], keep):
            raise SystemExit("eligibility mask differs from the frozen split")
        frozen_valid = np.asarray(split["validation_mask_within_eligible"], dtype=bool)
    X, y, w, ev = X[keep], y[keep], w[keep], ev[keep]
    train_mask, valid_mask = at.deterministic_event_split(ev)
    if not np.array_equal(valid_mask, frozen_valid):
        raise SystemExit("event-level validation split differs from the frozen split")
    return X[train_mask], y[train_mask], w[train_mask], X[valid_mask], y[valid_mask]


def load_reference(manifest):
    """Frozen labels, pT and shower weights for every evaluation sample."""
    ref = {}
    for ds in uc.ORDER:
        d = am.load(ds, manifest)
        if d is None:
            raise SystemExit(f"frozen predictions missing for {ds}")
        ref[ds] = d
    return ref


def predict_dataset(model, feature_set, ds, ref_d):
    """Score the frozen eligible rows of one dataset and check row alignment."""
    preds, labels, pts = [], [], []
    for _stem, X, y, pt, keep in at.iter_evaluation_cache_chunks(ds):
        Xk = build_features(feature_set, X[keep])
        preds.append(model.predict_proba(Xk)[:, 1] if Xk.shape[0] else np.empty(0))
        labels.append(y[keep])
        pts.append(pt[keep])
    p = np.concatenate(preds).astype(np.float32)
    lab = np.concatenate(labels).astype(np.int32)
    pt = np.concatenate(pts).astype(np.float32)
    if not np.array_equal(lab, ref_d["labels"]):
        raise SystemExit(f"{ds}: labels do not align with the frozen prediction rows")
    if not np.array_equal(pt, ref_d["pt"]):
        raise SystemExit(f"{ds}: pT does not align with the frozen prediction rows")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_EXTRA_SEEDS,
                    help="extra XGBoost seeds besides the control seed 42")
    args = ap.parse_args()
    seeds = [CONTROL_SEED] + [s for s in args.seeds if s != CONTROL_SEED]
    if len(set(seeds)) != len(seeds):
        raise SystemExit("duplicate seeds")

    check_frozen_xgb_params()
    if len(_RATIOS) != 9 or len(_DIMENSIONFUL) != 5:
        raise SystemExit("unexpected split of the fifteen inputs by scale degree")
    if sorted(SCALE_DEGREE) != sorted(at.HL):
        raise SystemExit("SCALE_DEGREE must cover exactly the frozen fifteen inputs")
    at.require_frozen_cache()
    manifest, metadata = am.require_final_prediction_manifest()

    out = new_output_dir("tagger_variants")
    (out / "models").mkdir()
    for src in (Path(__file__), Path(uc.__file__)):
        (out / f"{src.stem}_code.py").write_bytes(src.read_bytes())
    tee = Tee(out / "tagger_variants_log.txt")
    sys.stdout = tee
    try:
        from sklearn.metrics import roc_auc_score
        from xgboost import XGBClassifier
        import joblib
        import xgboost

        print(f"# ATLAS 80030 tagger_variants  {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"# output {out}")
        print(f"# xgboost {xgboost.__version__}; seeds {seeds}")
        for name, feats in FEATURE_SETS.items():
            print(f"# {name}: {len(feats)} features: {', '.join(feats)}")

        # ------------------------------------------------------ training ---
        t0 = time.time()
        Xtr, ytr, wtr, Xva, yva = load_training_set()
        print(f"  training set verified against frozen split: "
              f"{ytr.size:,} train / {yva.size:,} validation  ({time.time() - t0:.0f}s)")
        frozen_model = joblib.load(am.FINAL_MODEL_PATH)
        frozen_va = frozen_model.predict_proba(Xva)[:, 1]

        runs = []
        for name, feats in FEATURE_SETS.items():
            Ftr, Fva = build_features(name, Xtr), build_features(name, Xva)
            if Ftr.shape[1] != len(feats):
                raise SystemExit(f"{name}: built {Ftr.shape[1]} columns, expected {len(feats)}")
            for seed in seeds:
                t1 = time.time()
                model = XGBClassifier(random_state=seed, **XGB_PARAMS)
                model.fit(Ftr, ytr, sample_weight=wtr)
                va = model.predict_proba(Fva)[:, 1]
                vauc = float(roc_auc_score(yva, va))
                path = out / "models" / f"{name}_seed{seed}.joblib"
                joblib.dump(model, path)
                run = {"feature_set": name, "features": feats, "seed": seed,
                       "model_path": str(path), "model_sha256": sha256(path),
                       "validation_auc": vauc}
                if name == "full15" and seed == CONTROL_SEED:
                    diff = float(np.max(np.abs(va - frozen_va)))
                    run["control_validation_max_abs_diff_vs_frozen"] = diff
                    run["control_validation_auc_frozen"] = metadata.get(
                        "validation_auc_unweighted")
                    print(f"  CONTROL full15 seed 42: validation AUC {vauc:.6f} "
                          f"(frozen {metadata.get('validation_auc_unweighted'):.6f}); "
                          f"max |score difference| vs frozen model {diff:.3g}")
                runs.append(run)
                print(f"  trained {name:<17} seed {seed:>3}  validation AUC {vauc:.5f}  "
                      f"{time.time() - t1:.0f}s")
            del Ftr, Fva
        del Xtr, ytr, wtr, Xva, yva

        # ---------------------------------------------------- evaluation ---
        ref = load_reference(manifest)
        print(f"  frozen reference rows loaded for {len(ref)} samples")
        for run in runs:
            t1 = time.time()
            model = joblib.load(run["model_path"])
            data = {}
            identical = True
            for ds in uc.ORDER:
                p = predict_dataset(model, run["feature_set"], ds, ref[ds])
                if run["feature_set"] == "full15" and run["seed"] == CONTROL_SEED:
                    identical &= bool(np.array_equal(p, ref[ds]["predictions"]))
                d = {"predictions": p, "labels": ref[ds]["labels"], "pt": ref[ds]["pt"]}
                if ds == "test_nominal":
                    d["shower_weights"] = ref[ds]["shower_weights"]
                data[ds] = d
            M, S = uc.compute_chain(data, working_points=(0.5,), light=True,
                                    full_metric_sets=("test_nominal",))
            nom = data["test_nominal"]
            m80 = am.metrics_for(nom["labels"], nom["predictions"],
                                 np.ones(nom["labels"].shape), nom["pt"], 0.8, am.FULL_BINS)
            flat = uc.flatten_summary(S, M)
            flat["nominal_br_80"] = float(m80["br"])
            for ds in uc.ORDER:
                flat[f"br50_{ds}"] = float(M[(ds, 0.5)]["br"])
            run["results"] = flat
            if run["feature_set"] == "full15" and run["seed"] == CONTROL_SEED:
                run["control_predictions_bit_identical_to_frozen"] = identical
                if identical:
                    Mf, Sf = uc.compute_chain(data, working_points=(0.5, 0.8))
                    problems = uc.check_against_frozen(Mf, Sf, FROZEN_JSON)
                    run["control_reproduces_frozen_metrics"] = not problems
                    if problems:
                        print("  CONTROL: identical predictions but metrics differ:")
                        for p in problems[:10]:
                            print("    " + p)
                        raise SystemExit(2)
                print(f"  CONTROL predictions bit-identical to frozen: {identical}")
            print(f"  evaluated {run['feature_set']:<17} seed {run['seed']:>3}  "
                  f"AUC {flat['nominal_auc']:.5f}  BR50 {flat['nominal_br_50']:.2f}  "
                  f"ES {100 * flat['inc_es']:.2f}%  cluster {100 * flat['inc_cluster']:.2f}%  "
                  f"total {100 * flat['inc_total']:.2f}%  {time.time() - t1:.0f}s")
            (out / "runs.json").write_text(json.dumps(runs, indent=1) + "\n", encoding="utf-8")

        # ---------------------------------------------------- aggregation ---
        summary = {}
        keys = [k for k in runs[0]["results"]]
        for name in FEATURE_SETS:
            rs = [r for r in runs if r["feature_set"] == name]
            agg = {"seeds": [r["seed"] for r in rs],
                   "validation_auc": {"mean": float(np.mean([r["validation_auc"] for r in rs])),
                                      "std": float(np.std([r["validation_auc"] for r in rs], ddof=1))}}
            for k in keys:
                arr = np.asarray([r["results"][k] for r in rs], dtype=np.float64)
                agg[k] = {"mean": np.mean(arr, axis=0).tolist(),
                          "std": np.std(arr, axis=0, ddof=1).tolist(),
                          "min": np.min(arr, axis=0).tolist(),
                          "max": np.max(arr, axis=0).tolist()}
            summary[name] = agg
        result = {
            "description": "training-seed spread and scale-invariant feature-set study",
            "scale_degree": SCALE_DEGREE,
            "feature_sets": FEATURE_SETS,
            "xgb_params": XGB_PARAMS,
            "split_seed": at.SEED,
            "frozen_model_sha256": metadata["model_sha256"],
            "runs": runs,
            "summary": summary,
            "timestamp": datetime.now().astimezone().isoformat(),
        }
        (out / "tagger_variants_summary.json").write_text(json.dumps(result, indent=1) + "\n",
                                                         encoding="utf-8")
        show = [("nominal_auc", 1, "AUC"), ("nominal_br_50", 1, "BR@50"),
                ("nominal_br_80", 1, "BR@80"), ("inc_es", 100, "ES %"),
                ("inc_cluster", 100, "cluster %"), ("inc_track", 100, "track %"),
                ("inc_model", 100, "modelling %"), ("inc_scale", 100, "ISR/FSR %"),
                ("inc_total", 100, "total %")]
        print("\n  mean +- std over seeds")
        print(f"  {'quantity':<12}" + "".join(f"{n:>26}" for n in FEATURE_SETS))
        for k, f, label in show:
            row = f"  {label:<12}"
            for name in FEATURE_SETS:
                a = summary[name][k]
                row += f"{f * a['mean']:>16.4f} +- {f * a['std']:<7.4f}"
            print(row)
        print(f"\n  wrote {out / 'tagger_variants_summary.json'}")
        print("done.")
    finally:
        sys.stdout = tee.out
        tee.fh.close()


if __name__ == "__main__":
    main()
