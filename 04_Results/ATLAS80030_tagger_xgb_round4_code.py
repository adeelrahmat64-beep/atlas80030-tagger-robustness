"""ATLAS 80030 - train the high-level-quantity tagger and write ATLAS-format predictions.

This is step 2 of the rebuilt analysis. It trains a tagger on the fifteen
recomputed high-level quantities and writes one prediction file per dataset in
exactly the format ATLAS's own calc.py expects, so that every number quoted in
the paper can be produced by ATLAS's prescribed procedure rather than by a
method of our own invention.

WHY THIS TAGGER IS THE RIGHT ONE
    ATLAS's Table 4 publishes a high-level-quantity baseline, "hlDNN", trained
    on these same fifteen quantities: AUC 0.9374 +- 0.0001, ACC 0.8628,
    background rejection 47.2 +- 0.4 at 50% signal efficiency and 10.36 +- 0.03
    at 80%. Our tagger reads the same fifteen inputs, so hlDNN is a direct
    published anchor. If we land near it, the whole recomputation chain is
    validated against ATLAS in one number. That anchor is worth more to a
    referee than any internal cross-check.

WHAT IT READS
    <data-root>\\hl_recomputed\\<dataset>\\*.npz   the recomputed features
    <data-root>\\data\\train_nominal\\*.h5          training_weights
    <data-root>\\data\\test_nominal\\*.h5           EventInfo_mcEventWeights
    The stale fjet_* high-level branches in the .h5 files are never read. That
    is the entire reason the recomputation exists.

WHAT IT WRITES
    <data-root>\\predictions\\public_<dataset>.npz
    with keys 'predictions', 'labels', 'pt' - and 'shower_weights' for the
    nominal test set - which is precisely what ATLAS's calc.py loads.
    <data-root>\\models\\atlas_tagger_<model>_round<N>.joblib
    with a split-mask NPZ and JSON provenance sidecar. Plus a round-numbered
    log and source snapshot in Code files\\analysis_results\\.

THE TWO ATLAS RULES, BOTH HONOURED
    1. "The contribution to a loss function from any jet should always be
       weighted by the training weight." training_weights go in as
       sample_weight. They live only in train_nominal: signal jets carry
       exactly 1.0 and background jets carry a pT-dependent weight (0.12 at low
       pT rising past 1.4) that flattens the background spectrum onto the
       signal one.
    2. Performance claims need systematic uncertainties. That is atlas_metrics.py,
       which consumes what this script writes.
    Note the weights are for TRAINING ONLY. ATLAS's calc.py evaluates AUC, ACC
    and background rejection with a vector of ones for the nominal and detector
    systematic datasets, and applies shower-weight ratios only for the theory
    variations. We do the same. Weighting the evaluation metrics was one of the
    errors in the first version of this paper.

SELECTION
    ATLAS's own jet selection (paper Table 1) already requires at least three
    constituents. A handful of cer jets fall below that after the variation is
    applied, so they no longer satisfy ATLAS's selection and are dropped here,
    with the count reported. Any jet that is non-finite in any feature is
    dropped the same way. No NaN/Inf is replaced by a numeric sentinel. Both
    counts are expected to be tiny and are printed so they can be quoted.

SPLIT
    Validation assignment is a deterministic hash of
    EventInfo_mcEventNumber and SEED. All jets carrying one event number enter
    the same subset, so there is no event leakage. The exact eligibility and
    validation masks are saved with the model.

RUNNING IT
    Set MODEL, then:
        python "Code files\\atlas_tagger.py"
    Needs scikit-learn and xgboost installed on the Windows Python.

COST
    Training on 4,000,000 jets x 15 features is a few minutes for XGBoost.
    Inference over the sixteen evaluation datasets is a few minutes more.
    Peak RAM about 3 GB.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from atlas_config import (
    DATA_ROOT,
    EVALUATION_DATASETS,
    FROZEN_RECONSTRUCTION_FP,
    REQUIRED_DATASETS,
)

# ----------------------------------------------------------------- config ---
STORE = DATA_ROOT
HERE = Path(__file__).resolve().parent

MODEL = "xgb"              # "xgb" or "mlp"
SEED = 42

# ATLAS's own train.py uses n_train_jets = 4,000,000 out of the 92M available,
# so 4M is the reference scale, not a compromise. We hold 5M train_nominal
# jets, which is more than ATLAS itself used.
N_TRAIN_JETS = 4_000_000
VALID_FRACTION = 0.1       # same fraction as the official example trainer

# The nominal test set, the nine detector systematics, and the six modelling
# datasets. Anything not downloaded is skipped with a printed note - the
# modelling six are needed for the theoretical uncertainties, which ATLAS finds
# dominant, so their absence is reported loudly rather than passed over.
EVAL_SETS = list(EVALUATION_DATASETS)

CACHE = STORE / "hl_recomputed"
DATA = STORE / "data"
PRED = STORE / "predictions"
MODELS = STORE / "models"
RESULTS = HERE / "analysis_results"

# Feature order is fixed here and must never be reordered, because the model
# is trained on positions. It matches the order in recompute_hl.py.
HL = ["fjet_Tau1_wta", "fjet_Tau2_wta", "fjet_Tau3_wta", "fjet_Tau4_wta",
      "fjet_Split12", "fjet_Split23",
      "fjet_ECF1", "fjet_ECF2", "fjet_ECF3",
      "fjet_C2", "fjet_D2", "fjet_L2", "fjet_L3",
      "fjet_Qw", "fjet_ThrustMaj"]

MIN_CONSTITS = 3           # ATLAS paper Table 1
RAW_AUX_BRANCHES = {"training_weights", "EventInfo_mcEventWeights"}
CACHE_INFERENCE_FIELDS = set(HL) | {"labels", "fjet_pt", "n_constit"}
TRAINING_EVENT_FIELD = "EventInfo_mcEventNumber"
line = "=" * 78
sub = "-" * 78


def cache_fingerprint(data):
    """Return a cache stamp only when it is present and scalar."""
    if "code_fp" not in data.files:
        return "(unstamped)"
    stamp = np.asarray(data["code_fp"])
    if stamp.size != 1:
        return f"(non-scalar shape {stamp.shape})"
    return str(stamp.reshape(()).item())


def require_frozen_cache():
    """Refuse to train or evaluate against missing, stale, or mixed caches."""
    problems = []
    for ds in REQUIRED_DATASETS:
        files = sorted((CACHE / ds).glob("*.npz"))
        if not files:
            problems.append(f"{ds}: no cache files")
            continue
        fingerprints = set()
        for path in files:
            try:
                with np.load(path, allow_pickle=False) as data:
                    fingerprints.add(cache_fingerprint(data))
            except Exception as exc:
                problems.append(
                    f"{ds}/{path.name}: unreadable ({exc.__class__.__name__})"
                )
        if fingerprints != {FROZEN_RECONSTRUCTION_FP}:
            problems.append(f"{ds}: fingerprints {sorted(fingerprints)}")
    if problems:
        raise SystemExit(
            "frozen-cache preflight failed; do not train:\n  "
            + "\n  ".join(problems)
        )


def eligibility_mask(features, n_constit):
    """Frozen deterministic jet eligibility; no value is ever imputed."""
    if features.ndim != 2 or features.shape[1] != len(HL):
        raise ValueError(f"unexpected feature matrix shape {features.shape}")
    if n_constit.shape != (features.shape[0],):
        raise ValueError(
            f"n_constit shape {n_constit.shape} does not match features"
        )
    return np.isfinite(features).all(axis=1) & (n_constit >= MIN_CONSTITS)


def deterministic_event_split(event_number):
    """Assign complete events to validation with a stable SplitMix64 hash.

    The assignment is a pure function of EventInfo_mcEventNumber and SEED, so
    two jets carrying the same event number cannot enter opposite subsets.
    Unlike a saved PRNG state, it is independent of row ordering and sklearn
    implementation details.
    """
    event_number = np.asarray(event_number)
    if event_number.ndim != 1 or not np.issubdtype(
        event_number.dtype, np.integer
    ):
        raise ValueError(
            "EventInfo_mcEventNumber must be a one-dimensional integer array"
        )
    with np.errstate(over="ignore"):
        hashed = event_number.astype(np.uint64, copy=True)
        hashed += np.uint64(SEED) + np.uint64(0x9E3779B97F4A7C15)
        hashed = (hashed ^ (hashed >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        hashed = (hashed ^ (hashed >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
        hashed ^= hashed >> np.uint64(31)
    cutoff = np.uint64(int(VALID_FRACTION * (1 << 64)))
    valid = hashed < cutoff
    if not valid.any() or valid.all():
        raise ValueError("event hash produced an empty training or validation set")
    return ~valid, valid


def log_abs_features(values):
    """Pickle-safe transform used only inside the optional MLP pipeline."""
    return np.log(np.abs(values) + 1e-30)


class Tee:
    """Mirror the console to a round-numbered log; never overwrite.

    Same convention as recompute_hl.py and pair_features.py: the round number
    is one past the highest already on disk for this tag, and a verbatim copy
    of the script is written beside the log.
    """

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
        self.fh.flush()

    def flush(self):
        self.out.flush()
        self.fh.flush()


def load_features(ds, limit=None, *, require_event_number=False):
    """Read the recomputed features for one dataset, in HL order.

    Returns the design matrix, labels, jet pT, the per-file row counts, and the
    boolean mask of jets that pass ATLAS's selection and are finite. Row order
    within a file is the order recompute_hl.py wrote it, which is the order of
    the source .h5, so anything read from the .h5 lines up by position.

    ``EventInfo_mcEventNumber`` is requested only for the training load: it is
    needed for the deterministic event-level split, is not a classifier input,
    and is absent from the released modelling schemas. Evaluation never creates
    a synthetic event number. Its stable row identity is cache file stem plus
    original zero-based row index, and filtering preserves that order exactly.
    """
    fs = sorted((CACHE / ds).glob("*.npz"))
    if not fs:
        raise SystemExit(f"no recomputed features for {ds} in {CACHE / ds}")
    X, y, pt, nc, event_number, sizes = [], [], [], [], [], []
    got = 0
    for f in fs:
        with np.load(f, allow_pickle=False) as z:
            stamp = cache_fingerprint(z)
            if stamp != FROZEN_RECONSTRUCTION_FP:
                raise SystemExit(
                    f"refusing to load {f}: fingerprint {stamp!r}, expected "
                    f"{FROZEN_RECONSTRUCTION_FP!r}"
                )
            required = set(CACHE_INFERENCE_FIELDS)
            if require_event_number:
                required.add(TRAINING_EVENT_FIELD)
            missing = sorted(required - set(z.files))
            if missing:
                raise SystemExit(f"{f}: missing required cache fields {missing}")
            n = z["labels"].size
            bad_shapes = [
                name for name in required if np.asarray(z[name]).shape != (n,)
            ]
            if bad_shapes:
                raise SystemExit(
                    f"{f}: fields with row shape incompatible with labels "
                    f"({sorted(bad_shapes)})"
                )
            take = n if limit is None else min(n, limit - got)
            if take <= 0:
                break
            X.append(np.stack([z[c][:take] for c in HL], axis=1))
            y.append(z["labels"][:take])
            pt.append(z["fjet_pt"][:take])
            nc.append(z["n_constit"][:take])
            if require_event_number:
                event_number.append(z[TRAINING_EVENT_FIELD][:take])
            sizes.append(take)
            got += take
        if limit is not None and got >= limit:
            break
    X = np.concatenate(X).astype(np.float64)
    y = np.concatenate(y).astype(np.int32)
    pt = np.concatenate(pt).astype(np.float64)
    nc = np.concatenate(nc)
    event_number = np.concatenate(event_number) if require_event_number else None
    keep = eligibility_mask(X, nc)
    return (
        X, y, pt, event_number, sizes, keep,
        [f.stem for f in fs[:len(sizes)]],
    )


def load_h5_column(ds, stems, sizes, branch):
    """Read one branch from the source .h5 files, positionally aligned."""
    if branch not in RAW_AUX_BRANCHES:
        raise ValueError(
            f"raw HDF5 branch {branch!r} is not an approved auxiliary input"
        )
    import h5py
    out = []
    for stem, n in zip(stems, sizes):
        with h5py.File(DATA / ds / f"{stem}.h5", "r") as fh:
            out.append(fh[branch][:n])
    return np.concatenate(out)


def smoke_test_modelling_cache_schemas():
    """Exercise inference loading on all six real modelling cache schemas.

    This intentionally does not request an event-number column.  It is a
    cache-schema smoke test, not a reconstruction or modelling calculation.
    """
    from atlas_config import MODELLING_DATASETS

    for ds in MODELLING_DATASETS:
        X, y, pt, event_number, _sizes, keep, _stems = load_features(
            ds, limit=1, require_event_number=False
        )
        if event_number is not None:
            raise AssertionError(f"{ds}: inference unexpectedly returned event IDs")
        if X.shape != (1, len(HL)) or y.shape != (1,) or pt.shape != (1,):
            raise AssertionError(f"{ds}: invalid inference smoke-test shapes")
        if keep.shape != (1,):
            raise AssertionError(f"{ds}: invalid eligibility smoke-test shape")
        print(f"  smoke {ds:<14} PASS (inference schema; no event number required)")


def build_model():
    """The tagger. XGBoost by default.

    XGBoost splits on rank, so it is invariant to the monotone rescaling that
    these features badly need - ECF3 runs over many orders of magnitude - and
    needs no standardisation to be reproducible. The MLP branch standardises
    on log-transformed inputs because a neural network does care.
    """
    if MODEL == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", tree_method="hist",
            random_state=SEED, n_jobs=-1)
    if MODEL == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, FunctionTransformer
        return Pipeline([
            ("log", FunctionTransformer(log_abs_features)),
            ("scale", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(256, 256, 256),
                                  alpha=1e-4, batch_size=1024,
                                  learning_rate_init=1e-3, max_iter=60,
                                  early_stopping=True,
                                  n_iter_no_change=5, random_state=SEED))])
    raise SystemExit(f"unknown MODEL {MODEL!r}")


def file_sha256(path):
    """SHA-256 of a saved training artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_training_artifacts(
    model, tee, stems, sizes, eligibility, validation, y_train, y_valid,
    validation_auc,
):
    """Save the estimator and sufficient metadata to reproduce its split."""
    import joblib
    import sklearn

    MODELS.mkdir(parents=True, exist_ok=True)
    stem = MODELS / f"atlas_tagger_{MODEL}_round{tee.round}"
    model_path = Path(f"{stem}.joblib")
    split_path = Path(f"{stem}_split.npz")
    metadata_path = Path(f"{stem}_metadata.json")
    for path in (model_path, split_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite training artifact {path}")

    # Boolean masks are small and exactly preserve the eligibility and split
    # decisions without duplicating the four-million-row feature matrix.
    np.savez_compressed(
        split_path,
        eligibility_mask=np.asarray(eligibility, dtype=bool),
        validation_mask_within_eligible=np.asarray(validation, dtype=bool),
        file_stems=np.asarray(stems),
        file_rows=np.asarray(sizes, dtype=np.int64),
    )
    joblib.dump(model, model_path)

    xgboost_version = None
    if MODEL == "xgb":
        import xgboost
        xgboost_version = xgboost.__version__
    metadata = {
        "model_type": MODEL,
        "model_path": str(model_path),
        "model_sha256": file_sha256(model_path),
        "split_path": str(split_path),
        "split_sha256": file_sha256(split_path),
        "tagger_source_sha256": file_sha256(Path(__file__)),
        "reconstruction_fingerprint": FROZEN_RECONSTRUCTION_FP,
        "data_root": str(STORE),
        "training_dataset": "train_nominal",
        "feature_order": HL,
        "feature_preprocessing": (
            "none" if MODEL == "xgb"
            else "abs-log then StandardScaler fitted on the training subset"
        ),
        "eligibility": (
            f"all 15 features finite and n_constit >= {MIN_CONSTITS}; no imputation"
        ),
        "event_group_field": "EventInfo_mcEventNumber",
        "split_strategy": "SplitMix64(event number, seed) < validation cutoff",
        "seed": SEED,
        "validation_fraction": VALID_FRACTION,
        "training_file_stems": list(stems),
        "training_file_rows": [int(n) for n in sizes],
        "loaded_rows": int(np.asarray(eligibility).size),
        "eligible_rows": int(np.asarray(eligibility).sum()),
        "training_rows": int(y_train.size),
        "validation_rows": int(y_valid.size),
        "training_class_counts": np.bincount(y_train, minlength=2).tolist(),
        "validation_class_counts": np.bincount(y_valid, minlength=2).tolist(),
        "validation_auc_unweighted": float(validation_auc),
        "training_weight_branch": "training_weights",
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost_version,
        "python_version": sys.version,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return model_path, split_path, metadata_path


def load_saved_model(round_number):
    """Load and verify a previously saved model without entering training."""
    import joblib

    stem = MODELS / f"atlas_tagger_{MODEL}_round{round_number}"
    model_path = Path(f"{stem}.joblib")
    split_path = Path(f"{stem}_split.npz")
    metadata_path = Path(f"{stem}_metadata.json")
    for path in (model_path, split_path, metadata_path):
        if not path.is_file():
            raise SystemExit(f"saved training artifact is missing: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "model_type": MODEL,
        "reconstruction_fingerprint": FROZEN_RECONSTRUCTION_FP,
        "feature_order": HL,
        "event_group_field": TRAINING_EVENT_FIELD,
    }
    mismatches = [
        key for key, value in expected.items() if metadata.get(key) != value
    ]
    if mismatches:
        raise SystemExit(
            f"saved round {round_number} metadata mismatch: {mismatches}"
        )
    if metadata.get("model_sha256") != file_sha256(model_path):
        raise SystemExit(f"saved round {round_number} model checksum mismatch")
    if metadata.get("split_sha256") != file_sha256(split_path):
        raise SystemExit(f"saved round {round_number} split checksum mismatch")
    with np.load(split_path, allow_pickle=False) as split:
        required = {"eligibility_mask", "validation_mask_within_eligible",
                    "file_stems", "file_rows"}
        missing = sorted(required - set(split.files))
        if missing:
            raise SystemExit(f"saved split is missing fields {missing}")
        eligibility = np.asarray(split["eligibility_mask"], dtype=bool)
        validation = np.asarray(split["validation_mask_within_eligible"], dtype=bool)
        if validation.shape != (int(eligibility.sum()),):
            raise SystemExit("saved split eligibility/validation masks are misaligned")
        if eligibility.size != int(metadata.get("loaded_rows", -1)):
            raise SystemExit("saved split row count does not match metadata")
    model = joblib.load(model_path)
    if getattr(model, "n_features_in_", len(HL)) != len(HL):
        raise SystemExit("saved model feature count is incompatible with frozen HLFs")
    print(f"  loaded saved model round {round_number}: {model_path}")
    return model


def iter_evaluation_cache_chunks(ds):
    """Yield one feature-cache file at a time for low-memory evaluation/audit."""
    files = sorted((CACHE / ds).glob("*.npz"))
    if not files:
        raise SystemExit(f"no recomputed features for {ds} in {CACHE / ds}")
    for path in files:
        with np.load(path, allow_pickle=False) as cache:
            stamp = cache_fingerprint(cache)
            if stamp != FROZEN_RECONSTRUCTION_FP:
                raise SystemExit(
                    f"refusing to audit {path}: fingerprint {stamp!r}, expected "
                    f"{FROZEN_RECONSTRUCTION_FP!r}"
                )
            missing = sorted(CACHE_INFERENCE_FIELDS - set(cache.files))
            if missing:
                raise SystemExit(f"{path}: missing required cache fields {missing}")
            labels = cache["labels"].astype(np.int32)
            pt = cache["fjet_pt"].astype(np.float64)
            n_constit = cache["n_constit"]
            n = labels.size
            bad_shapes = [
                name for name in CACHE_INFERENCE_FIELDS
                if np.asarray(cache[name]).shape != (n,)
            ]
            if bad_shapes:
                raise SystemExit(
                    f"{path}: fields with row shape incompatible with labels "
                    f"({bad_shapes})"
                )
            features = np.stack([cache[name] for name in HL], axis=1).astype(
                np.float64
            )
        yield path.stem, features, labels, pt, eligibility_mask(features, n_constit)


def prediction_output_complete(ds):
    """Check that an existing public prediction is complete and source-aligned."""
    path = PRED / f"public_{ds}.npz"
    if not path.is_file():
        return False, "missing output"
    try:
        with np.load(path, allow_pickle=False) as output:
            required = {"predictions", "labels", "pt"}
            if ds == "test_nominal":
                required.add("shower_weights")
            missing = sorted(required - set(output.files))
            if missing:
                return False, f"missing output fields {missing}"
            predictions = output["predictions"]
            labels = output["labels"]
            pt = output["pt"]
        if predictions.ndim != labels.ndim or predictions.ndim != pt.ndim:
            return False, "prediction, label, and pt ranks differ"
        if not (predictions.shape == labels.shape == pt.shape):
            return False, "prediction, label, and pt shapes differ"

        # The source-stem / original-row ordering is preserved by the cache
        # iteration.  Check it chunk-wise to avoid materialising test_nominal
        # just to audit an output that is already complete.
        offset = 0
        for _stem, _X, expected_y, expected_pt, keep in iter_evaluation_cache_chunks(ds):
            count = int(keep.sum())
            end = offset + count
            if end > labels.size:
                return False, "output ends before eligible source rows"
            if not np.array_equal(labels[offset:end], expected_y[keep]):
                return False, "labels are not aligned to eligible source rows"
            if not np.array_equal(
                pt[offset:end], expected_pt[keep].astype(np.float32)
            ):
                return False, "pt is not aligned to eligible source rows"
            offset = end
        if labels.shape != (offset,):
            return False, "output length does not equal eligible source rows"
    except Exception as exc:
        return False, f"unreadable or invalid ({exc.__class__.__name__}: {exc})"
    return True, f"{labels.size:,} aligned eligible rows"


def evaluate_datasets(model, *, resume):
    """Write only unfinished public predictions when resuming an evaluation."""
    PRED.mkdir(parents=True, exist_ok=True)
    print(f"\n{line}\nevaluating -> {PRED}\n{line}")
    for ds in EVAL_SETS:
        if not (CACHE / ds).is_dir():
            raise SystemExit(f"required evaluation cache is missing: {CACHE / ds}")
        if resume:
            complete, detail = prediction_output_complete(ds)
            if complete:
                print(f"  {ds:<14} already complete; preserved ({detail})")
                continue
            print(f"  {ds:<14} incomplete ({detail}); regenerating")
        t1 = time.time()
        prediction_chunks, label_chunks, pt_chunks, shower_chunks = [], [], [], []
        drop = 0
        for stem, Xe, ye, pte, keep in iter_evaluation_cache_chunks(ds):
            drop += int((~keep).sum())
            Xe, ye, pte = Xe[keep], ye[keep], pte[keep]
            prediction_chunks.append(
                model.predict_proba(Xe)[:, 1] if Xe.shape[0] else np.empty(0)
            )
            label_chunks.append(ye)
            pt_chunks.append(pte)
            if ds == "test_nominal":
                sw = load_h5_column(
                    ds, [stem], [keep.size], "EventInfo_mcEventWeights"
                )
                shower_chunks.append(sw[keep])
        preds = np.concatenate(prediction_chunks)
        ye = np.concatenate(label_chunks)
        pte = np.concatenate(pt_chunks)
        out = {"predictions": preds.astype(np.float32),
               "labels": ye.astype(np.int32),
               "pt": pte.astype(np.float32)}
        if ds == "test_nominal":
            out["shower_weights"] = np.concatenate(shower_chunks).astype(np.float32)
            print(f"  {ds:<14} shower_weights {out['shower_weights'].shape}")
        np.savez_compressed(PRED / f"public_{ds}.npz", **out)
        print(f"  {ds:<14} {ye.size:>10,} jets  dropped {drop:>5}  "
              f"signal {100.0 * ye.mean():5.2f}%  "
              f"mean score {preds.mean():.4f}  {time.time() - t1:.0f}s")

    incomplete = []
    print(f"\n{sub}\nevaluation output alignment audit\n{sub}")
    for ds in EVAL_SETS:
        complete, detail = prediction_output_complete(ds)
        print(f"  {ds:<14} {'PASS' if complete else 'FAIL'}  {detail}")
        if not complete:
            incomplete.append(ds)
    if incomplete:
        raise SystemExit(f"incomplete or misaligned predictions: {incomplete}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluate-only", action="store_true",
        help="load a saved model and evaluate without retraining",
    )
    parser.add_argument(
        "--resume-evaluation", action="store_true",
        help="evaluation-only mode; preserve valid completed outputs",
    )
    parser.add_argument(
        "--model-round", type=int, default=1,
        help="saved model round for --evaluate-only (default: 1)",
    )
    parser.add_argument(
        "--smoke-test-modelling-cache", action="store_true",
        help="exercise all six real modelling cache schemas without inference",
    )
    args = parser.parse_args()
    if args.resume_evaluation:
        args.evaluate_only = True
    return args


def main():
    args = parse_args()
    if not CACHE.is_dir():
        raise SystemExit(f"feature cache not found:\n  {CACHE}")
    require_frozen_cache()
    tee = Tee(f"tagger_{MODEL}")
    sys.stdout = tee
    try:
        print(f"# ATLAS 80030 - atlas_tagger round {tee.round} - "
              f"{datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"# model {MODEL}  seed {SEED}  n_train_jets {N_TRAIN_JETS:,}")
        if args.smoke_test_modelling_cache:
            print(f"\n{line}\nreal-cache modelling schema smoke test\n{line}")
            smoke_test_modelling_cache_schemas()
            return
        if args.evaluate_only:
            print(f"\n{line}\nevaluation-only resume (no training)\n{line}")
            model = load_saved_model(args.model_round)
            evaluate_datasets(model, resume=True)
            print(f"\n{line}\ndone. predictions in {PRED}\n{line}")
            return

        # ---------------------------------------------------------- train ---
        print(f"\n{line}\ntraining on train_nominal\n{line}")
        t0 = time.time()
        X, y, pt, event_number, sizes, keep, stems = load_features(
            "train_nominal", N_TRAIN_JETS, require_event_number=True
        )
        w = load_h5_column("train_nominal", stems, sizes, "training_weights")
        print(f"  loaded          {X.shape[0]:>10,} jets, {X.shape[1]} features")
        print(f"  dropped by selection (n_constit < {MIN_CONSTITS} or non-finite): "
              f"{int((~keep).sum()):,}")
        if w.shape != y.shape or not np.isfinite(w).all() or (w < 0).any():
            raise SystemExit("training weights are misaligned, non-finite, or negative")
        X, y, w, event_number = (
            X[keep], y[keep], w[keep], event_number[keep]
        )
        del pt
        print(f"  training on     {X.shape[0]:>10,} jets  "
              f"signal {100.0 * y.mean():.2f}%")
        print(f"  training weight  signal mean {w[y == 1].mean():.4f}  "
              f"background mean {w[y == 0].mean():.4f}")

        train_mask, valid_mask = deterministic_event_split(event_number)
        Xtr, Xva = X[train_mask], X[valid_mask]
        ytr, yva = y[train_mask], y[valid_mask]
        wtr, wva = w[train_mask], w[valid_mask]
        print(f"  event-level split {Xtr.shape[0]:,} train / "
              f"{Xva.shape[0]:,} validation; seed {SEED}")
        for label in (0, 1):
            class_rows = y == label
            fraction = valid_mask[class_rows].mean()
            print(f"    class {label} validation fraction {fraction:.4%}")
        del X, y, w, event_number
        model = build_model()
        print(f"  fitting {MODEL} on {Xtr.shape[0]:,} / validating on {Xva.shape[0]:,}")
        model.fit(Xtr, ytr, sample_weight=wtr)
        print(f"  trained in {time.time() - t0:.0f}s")

        from sklearn.metrics import roc_auc_score
        va = model.predict_proba(Xva)[:, 1]
        validation_auc = roc_auc_score(yva, va)
        print(f"  validation AUC (unweighted) {validation_auc:.5f}")
        model_path, split_path, metadata_path = save_training_artifacts(
            model, tee, stems, sizes, keep, valid_mask, ytr, yva,
            validation_auc,
        )
        print(f"  saved model     {model_path}")
        print(f"  saved split     {split_path}")
        print(f"  saved metadata  {metadata_path}")
        del Xtr, Xva, ytr, yva, wtr, wva

        # ------------------------------------------------------- evaluate ---
        evaluate_datasets(model, resume=False)

        print(f"\n{line}\ndone. predictions in {PRED}\n"
              f"next: atlas_metrics.py\n{line}")
    finally:
        sys.stdout = tee.out
        tee.fh.close()
    print(f"log: {tee.path}")


if __name__ == "__main__":
    main()
