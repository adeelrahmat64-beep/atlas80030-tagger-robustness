"""Lightweight M-4 diagnostic for one bias shard against test_nominal.

This does not create pairs and never reads feature matrices. It compares only
event number, label, and the exact float32 bit patterns of fjet_eta/fjet_phi.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from atlas_config import DATA_ROOT, FROZEN_RECONSTRUCTION_FP


HERE = Path(__file__).resolve().parent
from atlas_config import RESULTS_ROOT  # noqa: E402
DEFAULT_OUTPUT = (RESULTS_ROOT /
                  "audit_remediation_2026-09-09_round1" /
                  "m4_bias_file000_diagnostic.json")
KEY_DTYPE = np.dtype([("event", "<i8"), ("eta_bits", "<u4"),
                      ("phi_bits", "<u4"), ("label", "<i2")])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_identity(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"EventInfo_mcEventNumber", "fjet_eta", "fjet_phi", "labels", "code_fp"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise SystemExit(f"{path}: missing identity fields {missing}")
        fingerprint = str(data["code_fp"].item())
        if fingerprint != FROZEN_RECONSTRUCTION_FP:
            raise SystemExit(f"{path}: unexpected reconstruction fingerprint {fingerprint}")
        return {
            "event": np.asarray(data["EventInfo_mcEventNumber"], dtype=np.int64),
            "eta": np.asarray(data["fjet_eta"], dtype=np.float32),
            "phi": np.asarray(data["fjet_phi"], dtype=np.float32),
            "label": np.asarray(data["labels"], dtype=np.int16),
        }


def build_key(identity: dict[str, np.ndarray]) -> np.ndarray:
    eta = identity["eta"].copy()
    phi = identity["phi"].copy()
    eta[eta == 0.0] = 0.0
    phi[phi == 0.0] = 0.0
    key = np.empty(identity["event"].size, dtype=KEY_DTYPE)
    key["event"] = identity["event"]
    key["eta_bits"] = eta.view(np.uint32)
    key["phi_bits"] = phi.view(np.uint32)
    key["label"] = identity["label"]
    return key


def membership(values: np.ndarray, sorted_unique: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_unique, values)
    valid = positions < sorted_unique.size
    out = np.zeros(values.size, dtype=bool)
    out[valid] = sorted_unique[positions[valid]] == values[valid]
    return out


def summary(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"min": None, "median": None, "p95": None, "max": None}
    return {"min": float(np.min(values)),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(np.max(values))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bias-file", default="bias_000.npz")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    bias_path = DATA_ROOT / "hl_recomputed" / "bias" / args.bias_file
    nominal_paths = sorted((DATA_ROOT / "hl_recomputed" / "test_nominal").glob("*.npz"))
    if not bias_path.is_file() or not nominal_paths:
        raise SystemExit("bias shard or test_nominal cache is missing")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing diagnostic: {args.output}")

    bias = load_identity(bias_path)
    bias_events = np.unique(bias["event"])
    nominal_common = {name: [] for name in ("event", "eta", "phi", "label")}
    nominal_files = []
    nominal_rows = []
    total_nominal_rows = 0
    for file_index, path in enumerate(nominal_paths):
        identity = load_identity(path)
        total_nominal_rows += identity["event"].size
        keep = membership(identity["event"], bias_events)
        for name in nominal_common:
            nominal_common[name].append(identity[name][keep])
        nominal_files.append(np.full(int(keep.sum()), file_index, dtype=np.int16))
        nominal_rows.append(np.flatnonzero(keep).astype(np.int32))
    nominal = {name: np.concatenate(parts) for name, parts in nominal_common.items()}
    nominal_file_index = np.concatenate(nominal_files)
    nominal_row_index = np.concatenate(nominal_rows)

    bias_key = build_key(bias)
    nominal_key = build_key(nominal)
    void_dtype = np.dtype((np.void, KEY_DTYPE.itemsize))
    bias_void = np.ascontiguousarray(bias_key).view(void_dtype).reshape(-1)
    nominal_void = np.ascontiguousarray(nominal_key).view(void_dtype).reshape(-1)
    exact = np.isin(bias_void, nominal_void)
    common_event = membership(bias["event"], np.unique(nominal["event"]))
    absent_event = ~common_event
    common_not_exact = common_event & ~exact

    order = np.argsort(nominal["event"], kind="stable")
    sorted_events = nominal["event"][order]
    unique_events, starts, counts = np.unique(sorted_events, return_index=True,
                                               return_counts=True)
    event_bounds = {int(event): (int(start), int(count))
                    for event, start, count in zip(unique_events, starts, counts)}
    eta_bit_match = 0
    phi_bit_match = 0
    same_label_candidate = 0
    label_only_mismatch = 0
    nearest_dr = []
    nearest_abs_deta = []
    nearest_abs_dphi = []
    examples = []
    for bias_index in np.flatnonzero(common_not_exact):
        start, count = event_bounds[int(bias["event"][bias_index])]
        candidate_indices = order[start:start + count]
        candidate_indices = candidate_indices[
            nominal["label"][candidate_indices] == bias["label"][bias_index]]
        if candidate_indices.size == 0:
            label_only_mismatch += 1
            continue
        same_label_candidate += 1
        bias_eta_bits = bias["eta"][bias_index:bias_index + 1].view(np.uint32)[0]
        bias_phi_bits = bias["phi"][bias_index:bias_index + 1].view(np.uint32)[0]
        eta_matches = nominal["eta"][candidate_indices].view(np.uint32) == bias_eta_bits
        phi_matches = nominal["phi"][candidate_indices].view(np.uint32) == bias_phi_bits
        eta_bit_match += int(eta_matches.any())
        phi_bit_match += int(phi_matches.any())
        deta = nominal["eta"][candidate_indices].astype(np.float64) - float(bias["eta"][bias_index])
        dphi = nominal["phi"][candidate_indices].astype(np.float64) - float(bias["phi"][bias_index])
        dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi
        dr = np.hypot(deta, dphi)
        nearest_local = int(np.argmin(dr))
        nearest = int(candidate_indices[nearest_local])
        nearest_dr.append(float(dr[nearest_local]))
        nearest_abs_deta.append(float(abs(deta[nearest_local])))
        nearest_abs_dphi.append(float(abs(dphi[nearest_local])))
        if len(examples) < 20:
            examples.append({
                "bias_row": int(bias_index),
                "event": int(bias["event"][bias_index]),
                "label": int(bias["label"][bias_index]),
                "bias_eta_bits_hex": f"0x{int(bias_eta_bits):08x}",
                "bias_phi_bits_hex": f"0x{int(bias_phi_bits):08x}",
                "nearest_nominal_file": nominal_paths[int(nominal_file_index[nearest])].name,
                "nearest_nominal_row": int(nominal_row_index[nearest]),
                "nominal_eta_bits_hex": f"0x{int(nominal['eta'][nearest:nearest + 1].view(np.uint32)[0]):08x}",
                "nominal_phi_bits_hex": f"0x{int(nominal['phi'][nearest:nearest + 1].view(np.uint32)[0]):08x}",
                "delta_r": float(dr[nearest_local]),
            })

    nearest_dr_array = np.asarray(nearest_dr, dtype=np.float64)
    result = {
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "classification": "STALE ARTIFACT plus diagnostic premise qualification",
        "issue": "M-4 bias pairing",
        "scope": "one bias shard; identity columns only; all test_nominal shards",
        "bias_file": str(bias_path.resolve()),
        "bias_file_sha256": sha256(bias_path),
        "diagnostic_source_sha256": sha256(Path(__file__)),
        "reconstruction_fingerprint": FROZEN_RECONSTRUCTION_FP,
        "nominal_files_scanned": len(nominal_paths),
        "nominal_rows_scanned": total_nominal_rows,
        "bias_rows": int(bias["event"].size),
        "bias_unique_events": int(bias_events.size),
        "nominal_rows_in_common_events": int(nominal["event"].size),
        "exact_key_duplicates": {
            "bias": int(bias_key.size - np.unique(bias_void).size),
            "nominal_candidates": int(nominal_key.size - np.unique(nominal_void).size),
        },
        "classification_counts": {
            "exact_event_label_eta_phi_bit_match": int(exact.sum()),
            "event_number_absent_from_test_nominal": int(absent_event.sum()),
            "common_event_but_no_exact_key": int(common_not_exact.sum()),
            "common_event_same_label_but_shifted_axis": same_label_candidate,
            "common_event_without_same_label_candidate": label_only_mismatch,
        },
        "percent_of_bias_rows": {
            "exact_match": 100.0 * float(exact.mean()),
            "event_absent": 100.0 * float(absent_event.mean()),
            "common_event_no_exact_key": 100.0 * float(common_not_exact.mean()),
        },
        "float32_bit_patterns_among_common_unmatched_same_label": {
            "rows": same_label_candidate,
            "any_candidate_eta_bits_equal": eta_bit_match,
            "any_candidate_phi_bits_equal": phi_bit_match,
            "nearest_delta_r": summary(nearest_dr_array),
            "nearest_absolute_delta_eta": summary(np.asarray(nearest_abs_deta)),
            "nearest_absolute_delta_phi": summary(np.asarray(nearest_abs_dphi)),
        },
        "examples": examples,
        "conclusion": (
            "Unmatched bias jets are separated into absent event numbers and common-event "
            "rows whose float32 jet axes do not match. Bias remains excluded from paired SHAP."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result["classification_counts"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
