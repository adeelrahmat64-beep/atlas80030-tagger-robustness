"""Paired SHAP-Delta attribution-drift analysis for the frozen final XGBoost.

This program deliberately has two phases.  ``--preflight`` is read-only and
validates the final-production model and the exact-pairing inputs.  The costly
TreeSHAP campaign runs only with the explicit ``--run-campaign`` switch.
It never reads, rewrites, or relies on the public prediction files.

For a physical nominal/systematic jet pair, the primary quantity is
    Delta_SHAP_i = SHAP_i(systematic) - SHAP_i(nominal)
with native XGBoost TreeSHAP values in margin space.  Classifier-score drift
is reported separately in probability space.  Absolute SHAP-Delta values are
never normalised away.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from scipy.stats import rankdata, spearmanr

from atlas_config import DATA_ROOT, FROZEN_RECONSTRUCTION_FP

HERE = Path(__file__).resolve().parent
MODEL = DATA_ROOT / "models" / "atlas_tagger_xgb_final_production_round1.joblib"
METADATA = DATA_ROOT / "models" / "atlas_tagger_xgb_final_production_round1_metadata.json"
PAIRED = DATA_ROOT / "paired"
# Each campaign/remediation must use a fresh directory. Round 2 contains the
# valid TreeSHAP evaluation with the score-summary transcription bug; round 3
# is reserved for the corrected summaries derived from its per-jet exports.
from atlas_config import RESULTS_ROOT  # noqa: E402
DEFAULT_OUT = RESULTS_ROOT / "shap_delta_final_production_round3"
ROUND2_OUT = RESULTS_ROOT / "shap_delta_final_production_round2"

FINAL_MODEL_SHA256 = "63a23308efbf2ce90418c8e657730f571db4798fc9ca2fb3692cb986de405f4c"
SEED = 42
SYSTEMATICS = ("esup", "esdown", "cpos", "cer")
SAMPLE_SIZES = (25_000, 50_000, 100_000)
PT_EDGES_MEV = np.linspace(350_000.0, 3_150_000.0, 15)
PAIR_FLOORS_PERCENT = {"cer": 99.80}
DEFAULT_PAIR_FLOOR_PERCENT = 99.95
HL = ("fjet_Tau1_wta", "fjet_Tau2_wta", "fjet_Tau3_wta", "fjet_Tau4_wta",
      "fjet_Split12", "fjet_Split23", "fjet_ECF1", "fjet_ECF2", "fjet_ECF3",
      "fjet_C2", "fjet_D2", "fjet_L2", "fjet_L3", "fjet_Qw",
      "fjet_ThrustMaj")
KEY_FIELDS = ("EventInfo_mcEventNumber", "fjet_eta", "fjet_phi", "labels")
IDENTITY_FIELDS = ("nom_source_file_stem", "nom_source_row_index",
                   "sys_source_file_stem", "sys_source_row_index")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(z: np.lib.npyio.NpzFile, name: str):
    return z[name].item()


def progress(message: str) -> None:
    """Emit flush-on-write campaign milestones for long-running local jobs."""
    print(f"[shap-delta] {message}", flush=True)


def write_json_atomic(path: Path, value: dict) -> None:
    """Avoid presenting a partly written JSON artifact as a completed result."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def python_scalar(value):
    """Convert NumPy scalars for CSV rows without touching an NpzFile."""
    return value.item() if hasattr(value, "item") else value


def require_final_model() -> dict:
    """Refuse every non-final artifact before any SHAP calculation."""
    if not MODEL.is_file() or not METADATA.is_file():
        raise SystemExit("frozen final-production model or metadata is missing")
    model_hash = sha256(MODEL)
    if model_hash != FINAL_MODEL_SHA256:
        raise SystemExit(f"final model SHA-256 mismatch: {model_hash}")
    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    required = {
        "run_role": "final_production",
        "model_sha256": FINAL_MODEL_SHA256,
        "training_rows_requested": 5_000_000,
        "reconstruction_fingerprint": FROZEN_RECONSTRUCTION_FP,
        "seed": SEED,
    }
    for key, expected in required.items():
        if meta.get(key) != expected:
            raise SystemExit(f"final metadata {key!r} is {meta.get(key)!r}; expected {expected!r}")
    if tuple(meta.get("feature_order", ())) != HL:
        raise SystemExit("final metadata feature order is not the frozen 15-HLF order")
    return {"model_path": str(MODEL), "model_sha256": model_hash,
            "metadata_path": str(METADATA), "metadata": meta}


def pair_path(systematic: str) -> Path:
    return PAIRED / f"{systematic}_paired.npz"


def require_pair_schema(systematic: str) -> dict:
    """Read only pair metadata and enforce the one-to-one pairing contract."""
    path = pair_path(systematic)
    if not path.is_file():
        raise SystemExit(f"missing paired frame: {path}; build it with pair_features.py first")
    with np.load(path, allow_pickle=False) as z:
        needed = set(KEY_FIELDS + IDENTITY_FIELDS + ("fjet_pt", "nom_n_constit",
                     "sys_n_constit", "pair_fp", "src_code_fp", "n_input_jets",
                     "n_unmatched", "n_dropped_nonfinite", "n_nominal_key_collisions",
                     "n_systematic_key_collisions", "n_duplicate_nominal_matches"))
        needed.update(f"nom_{f}" for f in HL)
        needed.update(f"sys_{f}" for f in HL)
        missing = sorted(needed.difference(z.files))
        if missing:
            raise SystemExit(f"{path.name}: missing pairing-audit fields {missing}")
        if str(scalar(z, "src_code_fp")) != FROZEN_RECONSTRUCTION_FP:
            raise SystemExit(f"{path.name}: reconstruction fingerprint is not frozen")
        for field in ("n_nominal_key_collisions", "n_systematic_key_collisions",
                      "n_duplicate_nominal_matches"):
            if int(scalar(z, field)) != 0:
                raise SystemExit(f"{path.name}: pairing collision check failed: {field}")
        n_input = int(scalar(z, "n_input_jets"))
        n_unmatched = int(scalar(z, "n_unmatched"))
        if n_input <= 0 or not 0 <= n_unmatched <= n_input:
            raise SystemExit(f"{path.name}: invalid pairing counts")
        rate = 100.0 * (n_input - n_unmatched) / n_input
        floor = PAIR_FLOORS_PERCENT.get(systematic, DEFAULT_PAIR_FLOOR_PERCENT)
        if rate < floor:
            raise SystemExit(f"{path.name}: pairing rate {rate:.5f}% below {floor:.2f}% floor")
        if not str(scalar(z, "pair_fp")):
            raise SystemExit(f"{path.name}: no pairing fingerprint")
        nominal_stems = np.asarray(z["nom_source_file_stem"]).astype(str)
        if nominal_stems.shape != z["labels"].shape or not np.all(
                np.char.startswith(nominal_stems, "test_nominal_")):
            raise SystemExit(
                f"{path.name}: paired SHAP is restricted to test_nominal jets")
        return {"path": str(path), "pair_fingerprint": str(scalar(z, "pair_fp")),
                "source_reconstruction_fingerprint": str(scalar(z, "src_code_fp")),
                "input_rows": n_input, "unmatched_rows": n_unmatched,
                "dropped_nonfinite_rows": int(scalar(z, "n_dropped_nonfinite")),
                "matched_rate_percent": rate, "pair_rows_after_finite_drop": int(z["labels"].shape[0]),
                "collision_checks": {field: int(scalar(z, field)) for field in
                                     ("n_nominal_key_collisions", "n_systematic_key_collisions",
                                      "n_duplicate_nominal_matches")}}


def stable_seed(systematic: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{SEED}:{systematic}".encode()).digest()[:8], "little")


def allocate(available: np.ndarray, wanted: int) -> np.ndarray:
    """Largest-remainder proportional allocation, capped by availability."""
    if wanted > int(available.sum()):
        raise SystemExit(f"requested {wanted:,} pairs but only {available.sum():,} are jointly eligible")
    raw = wanted * available / available.sum()
    out = np.floor(raw).astype(np.int64)
    remaining = wanted - int(out.sum())
    order = np.argsort(-(raw - out), kind="stable")
    for i in order:
        if remaining == 0:
            break
        if out[i] < available[i]:
            out[i] += 1
            remaining -= 1
    if remaining:
        for i in np.where(out < available)[0]:
            take = min(remaining, int(available[i] - out[i]))
            out[i] += take
            remaining -= take
            if not remaining:
                break
    if remaining:
        raise RuntimeError("could not allocate deterministic sample")
    return out


def nested_stratified_indices(labels: np.ndarray, pt: np.ndarray, systematic: str) -> tuple[dict, dict]:
    """Nested 25k/50k/100k samples stratified on class and pT, not files."""
    pt_bin = np.searchsorted(PT_EDGES_MEV, pt, side="right")
    strata = labels.astype(np.int16) * (len(PT_EDGES_MEV) + 1) + pt_bin.astype(np.int16)
    ids, inverse = np.unique(strata, return_inverse=True)
    available = np.bincount(inverse, minlength=ids.size).astype(np.int64)
    quotas = {n: allocate(available, n) for n in SAMPLE_SIZES}
    rng = np.random.default_rng(stable_seed(systematic))
    chosen = {n: [] for n in SAMPLE_SIZES}
    for st in range(ids.size):
        rows = np.flatnonzero(inverse == st)
        rows = rows[rng.permutation(rows.size)]
        previous = 0
        for n in SAMPLE_SIZES:
            now = int(quotas[n][st])
            chosen[n].append(rows[:now])
            previous = now
    # Quotas are independently proportional.  Reconcile them into truly nested
    # sets by retaining the smaller quota first and filling only new rows.
    nested = {}
    prior = {st: np.empty(0, dtype=np.int64) for st in range(ids.size)}
    for n in SAMPLE_SIZES:
        groups = []
        for st in range(ids.size):
            rows = chosen[n][st]
            old = prior[st]
            if old.size > rows.size:
                raise RuntimeError("nested allocation decreased in a stratum")
            # rows were generated from one deterministic stratum permutation.
            merged = rows
            prior[st] = merged
            groups.append(merged)
        nested[n] = np.concatenate(groups)
        if nested[n].size != n:
            raise RuntimeError("nested sample has wrong size")
    # Independent proportional quotas can in theory decrease in a very small
    # stratum; with 25k+ samples this is practically avoided, but never hidden.
    for small, large in zip(SAMPLE_SIZES, SAMPLE_SIZES[1:]):
        if not np.isin(nested[small], nested[large]).all():
            raise SystemExit("nested stratified sampling contract failed")
    return nested, {"stratum_ids": ids.tolist(), "available": available.tolist(),
                    "allocations": {str(n): quotas[n].tolist() for n in SAMPLE_SIZES}}


def feature_summary(nom_shap: np.ndarray, sys_shap: np.ndarray, labels: np.ndarray,
                    nom_score: np.ndarray, sys_score: np.ndarray) -> list[dict]:
    """Summarise SHAP and probability-score drift under identical class masks."""
    expected = (labels.size,)
    if nom_score.shape != expected or sys_score.shape != expected:
        raise ValueError("score arrays do not align with the SHAP/label rows")
    rows = []
    for subset, mask in (("all", np.ones(labels.size, bool)), ("signal", labels == 1),
                         ("background", labels == 0)):
        if not mask.any():
            raise SystemExit(f"sample contains no {subset} pairs")
        nom = nom_shap[mask]
        syst = sys_shap[mask]
        delta = syst - nom
        nom_abs = np.abs(nom).mean(axis=0)
        sys_abs = np.abs(syst).mean(axis=0)
        delta_abs = np.abs(delta)
        delta_score = sys_score[mask] - nom_score[mask]
        nom_rank = rankdata(-nom_abs, method="min").astype(int)
        sys_rank = rankdata(-sys_abs, method="min").astype(int)
        rho, p_value = spearmanr(nom_abs, sys_abs)
        for i, feature in enumerate(HL):
            denom = float(nom_abs[i])
            rows.append({"subset": subset, "feature": feature, "n_pairs": int(mask.sum()),
                         "mean_signed_delta_shap": float(delta[:, i].mean()),
                         "median_signed_delta_shap": float(np.median(delta[:, i])),
                         "mean_abs_delta_shap": float(delta_abs[:, i].mean()),
                         "median_abs_delta_shap": float(np.median(delta_abs[:, i])),
                         "nominal_mean_abs_shap": denom,
                         "systematic_mean_abs_shap": float(sys_abs[i]),
                         "relative_attribution_drift": (float((sys_abs[i] - nom_abs[i]) / denom)
                                                       if denom > 0.0 else None),
                         "relative_attribution_drift_defined": bool(denom > 0.0),
                         "nominal_importance_rank": int(nom_rank[i]),
                         "systematic_importance_rank": int(sys_rank[i]),
                         "global_importance_spearman_rho": float(rho) if np.isfinite(rho) else None,
                         "global_importance_spearman_pvalue": float(p_value) if np.isfinite(p_value) else None,
                         "mean_signed_delta_score": float(delta_score.mean()),
                         "median_signed_delta_score": float(np.median(delta_score)),
                         "mean_abs_delta_score": float(np.abs(delta_score).mean())})
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path.name}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


SCORE_FIELDS = ("mean_signed_delta_score", "median_signed_delta_score",
                "mean_abs_delta_score")


def _score_stats(values: np.ndarray) -> dict:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError("invalid per-jet Delta_score values in round-2 export")
    return {
        "mean_signed_delta_score": float(values.mean()),
        "median_signed_delta_score": float(np.median(values)),
        "mean_abs_delta_score": float(np.abs(values).mean()),
    }


def _attribution_payload_sha256(rows: list[dict]) -> str:
    """Hash every per-feature field except the three repaired score summaries."""
    payload = [{k: v for k, v in row.items() if k not in SCORE_FIELDS} for row in rows]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def repair_score_summaries(source_dir: Path, output_dir: Path,
                           final_model_provenance: dict) -> None:
    """Create round-3 summaries from round-2 per-jet score exports.

    TreeSHAP values, pairing, samples, identities, and seeds are not recomputed.
    The per-jet relation CSV is the authoritative source for Delta_score.
    """
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if source_dir == output_dir:
        raise SystemExit("round-3 output must differ from the frozen source directory")
    source_results = source_dir / "shap_delta_results.json"
    if not source_results.is_file():
        raise SystemExit(f"missing source result: {source_results}")
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise SystemExit(f"refusing to write into existing remediation directory: {output_dir}")

    result = json.loads(source_results.read_text(encoding="utf-8"))
    rows = result.get("per_feature")
    if not isinstance(rows, list) or not rows:
        raise SystemExit("round-2 JSON has no per_feature rows")
    source_model_sha = result.get("provenance", {}).get("model_sha256")
    if source_model_sha != final_model_provenance["model_sha256"]:
        raise SystemExit("round-2 SHAP results are not tied to the frozen final model")

    before_attribution_hash = _attribution_payload_sha256(rows)
    stats = {}
    relation_hashes = {}
    crosscheck_rows = []
    for systematic in SYSTEMATICS:
        relation = source_dir / f"shap_delta_{systematic}_score_attribution_relation.csv"
        if not relation.is_file():
            raise SystemExit(f"missing round-2 per-jet export: {relation}")
        relation_hashes[relation.name] = sha256(relation)
        grouped = {(n, label): [] for n in SAMPLE_SIZES for label in (0, 1)}
        with relation.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            required = {"systematic", "sample_size", "label", "delta_score"}
            if not required.issubset(reader.fieldnames or ()):
                raise SystemExit(f"{relation.name}: missing required columns")
            for record in reader:
                if record["systematic"] != systematic:
                    raise SystemExit(f"{relation.name}: mixed systematic identities")
                key = (int(record["sample_size"]), int(record["label"]))
                if key not in grouped:
                    raise SystemExit(f"{relation.name}: unexpected sample/label {key}")
                grouped[key].append(float(record["delta_score"]))

        for n in SAMPLE_SIZES:
            arrays = {label: np.asarray(grouped[(n, label)], dtype=np.float64)
                      for label in (0, 1)}
            for subset, values in (("all", np.concatenate((arrays[0], arrays[1]))),
                                   ("signal", arrays[1]),
                                   ("background", arrays[0])):
                key = (systematic, n, subset)
                stats[key] = _score_stats(values)
                crosscheck_rows.append({"systematic": systematic,
                                        "sample_size": n, "subset": subset,
                                        "n_relation_rows": int(values.size),
                                        **stats[key], "summary_feature_rows_checked": len(HL),
                                        "status": "PASS"})

    seen = {}
    for row in rows:
        key = (row.get("systematic"), int(row.get("sample_size")), row.get("subset"))
        if key not in stats:
            raise SystemExit(f"unexpected per-feature summary group: {key}")
        if int(row.get("n_pairs")) != int(next(r["n_relation_rows"] for r in crosscheck_rows
                                                if (r["systematic"], r["sample_size"], r["subset"]) == key)):
            raise SystemExit(f"{key}: per-feature n_pairs disagrees with per-jet export")
        row.update(stats[key])
        seen[key] = seen.get(key, 0) + 1
    bad_groups = {str(k): v for k, v in seen.items() if v != len(HL)}
    expected_groups = len(SYSTEMATICS) * len(SAMPLE_SIZES) * 3
    if len(seen) != expected_groups or bad_groups:
        raise SystemExit(f"incomplete per-feature coverage: groups={len(seen)}, bad={bad_groups}")
    after_attribution_hash = _attribution_payload_sha256(rows)
    if after_attribution_hash != before_attribution_hash:
        raise SystemExit("non-score SHAP payload changed during summary repair")

    created = datetime.now(timezone.utc).isoformat()
    result["per_feature"] = rows
    result["remediation"] = {
        "round": 3,
        "created_timestamp_utc": created,
        "classification": "CONFIRMED BUG",
        "issue": "M-1 SHAP per-class score drift",
        "source_round": str(source_dir),
        "source_results_sha256": sha256(source_results),
        "source_relation_sha256": relation_hashes,
        "method": "score summaries recomputed from saved round-2 per-jet relation CSVs; TreeSHAP not rerun",
        "tree_shap_recomputed": False,
        "pairing_recomputed": False,
        "samples_or_identities_changed": False,
        "seed_changed": False,
        "model_sha256": final_model_provenance["model_sha256"],
        "non_score_attribution_payload_sha256_before": before_attribution_hash,
        "non_score_attribution_payload_sha256_after": after_attribution_hash,
        "crosscheck": "PASS",
    }
    write_csv(output_dir / "shap_delta_per_feature.csv", rows)
    write_csv(output_dir / "shap_delta_figure_inputs.csv", rows)
    write_csv(output_dir / "score_drift_crosscheck.csv", crosscheck_rows)
    write_json_atomic(output_dir / "shap_delta_results.json", result)
    write_json_atomic(output_dir / "campaign_status.json", {
        "status": "complete",
        "output_directory": str(output_dir),
        "created_timestamp_utc": created,
        "round": 3,
        "source_round": str(source_dir),
        "remediation_only": True,
        "tree_shap_recomputed": False,
        "crosscheck": "PASS",
        "model_sha256": final_model_provenance["model_sha256"],
        "shap_delta_source_sha256": sha256(Path(__file__)),
    })
    (output_dir / "shap_delta_summary.txt").write_text(
        "PAIRED SHAP-DELTA ROUND 3: M-1 SCORE-SUMMARY REMEDIATION\n"
        f"model SHA-256: {final_model_provenance['model_sha256']}\n"
        f"source round: {source_dir}\n"
        "TreeSHAP/pairing/samples/identities/seed changed: no\n"
        "Per-class score summaries recomputed from saved per-jet Delta_score rows.\n"
        "Cross-check: PASS\n", encoding="utf-8")
    progress(f"round-3 score-summary remediation complete: {output_dir}")


def stability_diagnostics(by_size: dict[int, list[dict]]) -> dict:
    """Report, but do not silently decide, nested-sample stability."""
    out = {}
    reference = by_size[100_000]
    for subset in ("all", "signal", "background"):
        ref = sorted((r for r in reference if r["subset"] == subset), key=lambda r: r["feature"])
        item = {}
        for size in (25_000, 50_000):
            other = sorted((r for r in by_size[size] if r["subset"] == subset), key=lambda r: r["feature"])
            nominal_rho = spearmanr([r["nominal_mean_abs_shap"] for r in other],
                                    [r["nominal_mean_abs_shap"] for r in ref]).statistic
            systematic_rho = spearmanr([r["systematic_mean_abs_shap"] for r in other],
                                       [r["systematic_mean_abs_shap"] for r in ref]).statistic
            drift_rho = spearmanr([r["mean_abs_delta_shap"] for r in other],
                                  [r["mean_abs_delta_shap"] for r in ref]).statistic
            top5 = [r["feature"] for r in sorted(other, key=lambda r: -r["mean_abs_delta_shap"])[:5]]
            top5_ref = [r["feature"] for r in sorted(ref, key=lambda r: -r["mean_abs_delta_shap"])[:5]]
            item[str(size)] = {"nominal_importance_spearman_vs_100k": float(nominal_rho),
                               "systematic_importance_spearman_vs_100k": float(systematic_rho),
                               "mean_abs_delta_shap_spearman_vs_100k": float(drift_rho),
                               "mean_abs_delta_top5_overlap_with_100k": len(set(top5) & set(top5_ref)),
                               "mean_abs_delta_top5": top5,
                               "mean_abs_delta_top5_100k": top5_ref}
        out[subset] = item
    return out


def campaign(systematics: tuple[str, ...], provenance: dict, output_dir: Path) -> None:
    """Run one isolated campaign; an interrupted directory is never resumed."""
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise SystemExit(
            f"refusing to write into existing campaign directory: {output_dir}\n"
            "This may be a completed or interrupted campaign. Choose a new --output-dir; "
            "this program never resumes into or mixes with an existing directory.")
    status = {"status": "running", "output_directory": str(output_dir.resolve()),
              "systematics_requested": list(systematics), "provenance": provenance}
    write_json_atomic(output_dir / "campaign_status.json", status)
    completed = False

    def preserve_interrupted_status() -> None:
        """Label ordinary Ctrl+C/errors without permitting a later resume."""
        if completed:
            return
        status.update({"status": "interrupted_or_failed",
                       "ended_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                       "note": "Campaign ended before completion; this directory is isolated and must not be resumed."})
        try:
            write_json_atomic(output_dir / "campaign_status.json", status)
        except Exception:
            # A forced process kill cannot always write status. The existing
            # directory guard remains the non-mixing safety boundary.
            pass

    atexit.register(preserve_interrupted_status)
    progress(f"campaign directory created: {output_dir}")
    model = joblib.load(MODEL)
    if getattr(model, "n_features_in_", len(HL)) != len(HL):
        raise SystemExit("final XGBoost model does not have 15 input features")
    booster = model.get_booster()
    all_rows, run = [], {"provenance": provenance, "systematics": {},
                          "stability_policy": "Diagnostics only: 100k is not declared final automatically; review nested rank and mean-|Delta_SHAP| comparisons."}
    for systematic in systematics:
        progress(f"{systematic}: loading paired frame")
        info = require_pair_schema(systematic)
        with np.load(pair_path(systematic), allow_pickle=False) as z:
            labels_all = z["labels"].astype(np.int8)
            pt_all = z["fjet_pt"].astype(np.float64)
            eligible = ((z["nom_n_constit"] >= 3) & (z["sys_n_constit"] >= 3) &
                        np.isin(labels_all, [0, 1]) & np.isfinite(pt_all))
            raw_indices = np.flatnonzero(eligible)
            nested_local, strata = nested_stratified_indices(labels_all[eligible], pt_all[eligible], systematic)
            nested = {n: raw_indices[idx] for n, idx in nested_local.items()}
            max_idx = nested[SAMPLE_SIZES[-1]]
            progress(f"{systematic}: sampling complete (nested 25k/50k/100k)")
            nom_x = np.column_stack([z[f"nom_{f}"][max_idx] for f in HL]).astype(np.float32)
            sys_x = np.column_stack([z[f"sys_{f}"][max_idx] for f in HL]).astype(np.float32)
            if not (np.isfinite(nom_x).all() and np.isfinite(sys_x).all() and
                    (nom_x > -9.0e5).all() and (sys_x > -9.0e5).all()):
                raise SystemExit(
                    f"{systematic}: non-finite/sentinel features survived paired eligibility")
            nom_shap = booster.predict(xgb.DMatrix(nom_x), pred_contribs=True)[:, :-1]
            progress(f"{systematic}: TreeSHAP nominal complete")
            sys_shap = booster.predict(xgb.DMatrix(sys_x), pred_contribs=True)[:, :-1]
            if nom_shap.shape != nom_x.shape or sys_shap.shape != sys_x.shape:
                raise SystemExit(f"{systematic}: unexpected XGBoost TreeSHAP shape")
            progress(f"{systematic}: TreeSHAP systematic complete")
            nom_score = model.predict_proba(nom_x)[:, 1]
            sys_score = model.predict_proba(sys_x)[:, 1]
            # NpzFile lazily decompresses each member access. Read every
            # identity/key member once for the 100k nested superset; the row
            # export below indexes ordinary in-memory arrays only.
            identity_arrays = {field: z[field][max_idx] for field in KEY_FIELDS + IDENTITY_FIELDS}
            loc = {int(row): i for i, row in enumerate(max_idx)}
            identity_rows, relation_rows, by_size = [], [], {}
            progress(f"{systematic}: writing identities")
            for n, indexes in nested.items():
                take = np.asarray([loc[int(x)] for x in indexes])
                for row, local in zip(indexes, take):
                    identity_rows.append({"systematic": systematic, "sample_size": n,
                        "paired_row_index": int(row), "label": int(labels_all[row]), "fjet_pt": float(pt_all[row]),
                        **{k: python_scalar(identity_arrays[k][local]) for k in KEY_FIELDS + IDENTITY_FIELDS}})
                summary = feature_summary(nom_shap[take], sys_shap[take], labels_all[indexes],
                                          nom_score[take], sys_score[take])
                for item in summary:
                    item.update({"systematic": systematic, "sample_size": n})
                all_rows.extend(summary)
                by_size[n] = summary
                paired_abs_drift = np.abs(sys_shap[take] - nom_shap[take]).mean(axis=1)
                score_delta = sys_score[take] - nom_score[take]
                for row, local, attribution_drift, score_drift in zip(indexes, take, paired_abs_drift, score_delta):
                    relation_rows.append({"systematic": systematic, "sample_size": n,
                        "paired_row_index": int(row), "label": int(labels_all[row]), "fjet_pt": float(pt_all[row]),
                        "delta_score": float(score_drift),
                        "mean_abs_delta_shap_across_features": float(attribution_drift)})
            write_csv(output_dir / f"shap_delta_{systematic}_sampled_row_identities.csv", identity_rows)
            write_csv(output_dir / f"shap_delta_{systematic}_score_attribution_relation.csv", relation_rows)
            run["systematics"][systematic] = {**info, "jointly_eligible_pairs": int(eligible.sum()),
                "sample_strata": strata, "sample_sizes": list(SAMPLE_SIZES),
                "stability_diagnostics": stability_diagnostics(by_size),
                "candidate_100k_status": "requires review of recorded nested-sample diagnostics; not auto-declared stable"}
            progress(f"{systematic}: systematic complete")
    write_csv(output_dir / "shap_delta_per_feature.csv", all_rows)
    write_csv(output_dir / "shap_delta_figure_inputs.csv", all_rows)
    write_json_atomic(output_dir / "shap_delta_results.json", run | {"per_feature": all_rows})
    lines = ["PAIRED SHAP-DELTA: frozen final 5M XGBoost", f"model SHA-256: {provenance['model_sha256']}",
             f"reconstruction fingerprint: {FROZEN_RECONSTRUCTION_FP}", "", "Outputs retain absolute Delta_SHAP as the primary quantity."]
    for systematic, item in run["systematics"].items():
        lines.append(f"{systematic}: paired={item['pair_rows_after_finite_drop']:,}, jointly eligible={item['jointly_eligible_pairs']:,}, match={item['matched_rate_percent']:.5f}%")
    (output_dir / "shap_delta_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    status.update({"status": "complete", "ended_timestamp_utc": datetime.now(timezone.utc).isoformat()})
    write_json_atomic(output_dir / "campaign_status.json", status)
    completed = True
    atexit.unregister(preserve_interrupted_status)
    progress("campaign complete")


def self_test() -> None:
    labels = np.array([0, 1] * 100_000, dtype=np.int8)
    pt = np.linspace(200_000, 3_300_000, labels.size)
    samples, _ = nested_stratified_indices(labels, pt, "esup")
    assert [samples[n].size for n in SAMPLE_SIZES] == list(SAMPLE_SIZES)
    assert np.isin(samples[25_000], samples[50_000]).all()
    assert np.isin(samples[50_000], samples[100_000]).all()
    z = np.vstack([np.arange(15, dtype=float), np.arange(15, dtype=float) + 0.2])
    summary = feature_summary(z, z + 0.1, np.array([0, 1]),
                              np.array([0.2, 0.8]), np.array([0.5, 0.7]))
    assert len(summary) == 45
    by_subset = {subset: next(r for r in summary if r["subset"] == subset)
                 for subset in ("all", "signal", "background")}
    assert np.isclose(by_subset["signal"]["mean_signed_delta_score"], -0.1)
    assert np.isclose(by_subset["background"]["mean_signed_delta_score"], 0.3)
    assert np.isclose(by_subset["all"]["mean_signed_delta_score"], 0.1)
    print("self-test PASS: nested sampler and class-masked SHAP/score summaries")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Read-only model/pair schema audit.")
    parser.add_argument("--self-test", action="store_true", help="Small synthetic sampler/summary test.")
    parser.add_argument("--run-campaign", action="store_true", help="Run the costly 25k/50k/100k TreeSHAP campaign.")
    parser.add_argument("--repair-score-summaries-from", type=Path,
                        help="Create corrected summaries from an existing round's per-jet relation CSVs; does not rerun TreeSHAP.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT,
                        help="New campaign/remediation directory; existing directories are always refused (default: round3).")
    parser.add_argument("--systematics", nargs="+", choices=SYSTEMATICS, default=list(SYSTEMATICS))
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.repair_score_summaries_from and (args.run_campaign or args.preflight):
        parser.error("--repair-score-summaries-from is a separate remediation-only mode")
    provenance = require_final_model()
    if args.repair_score_summaries_from:
        repair_score_summaries(args.repair_score_summaries_from, args.output_dir, provenance)
        return
    systematics = tuple(args.systematics)
    if args.preflight or not args.run_campaign:
        for systematic in systematics:
            print(json.dumps({systematic: require_pair_schema(systematic)}, indent=2))
        print("preflight PASS: final model and paired-frame contracts verified")
        return
    provenance.update({"tagger_source_sha256": sha256(HERE / "atlas_tagger.py"),
                       "shap_delta_source_sha256": sha256(Path(__file__)), "seed": SEED,
                       "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                       "python": sys.version, "platform": platform.platform(),
                       "tree_shap": "XGBoost Booster.predict(pred_contribs=True), margin space"})
    campaign(systematics, provenance, args.output_dir)


if __name__ == "__main__":
    main()
