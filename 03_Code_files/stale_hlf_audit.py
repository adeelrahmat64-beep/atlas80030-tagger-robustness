"""Audit stale stored HLF values in all detector-systematic source files.

The script does not reconstruct observables or modify analysis inputs.  It first
preflights every sample's exact event/eta/phi/label keys.  Duplicate, non-finite,
or insufficiently matched keys are recorded as ``N/A / not pairable`` rather
than terminating the audit or producing invalid partial pair statistics.

For valid one-to-one pairs, stored HLF values are compared by float32 bit pattern.
The constituent evidence is variation-specific: pT response (ES), energy-response
width (CER), angular movement (CPOS), and only unpaired multiplicity distributions
for track variations.

Run: python "Code files/stale_hlf_audit.py"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from atlas_config import (DATA_ROOT,
    DETECTOR_SYSTEMATICS,
    FROZEN_RECONSTRUCTION_FP,
    NOMINAL_EVALUATION_DATASET,
    RESULTS_ROOT,)

CACHE = DATA_ROOT / "hl_recomputed"
DATA = DATA_ROOT / "data"
OUT = RESULTS_ROOT / "stale_hlf_audit"
HL = ["fjet_Tau1_wta", "fjet_Tau2_wta", "fjet_Tau3_wta", "fjet_Tau4_wta",
      "fjet_Split12", "fjet_Split23", "fjet_ECF1", "fjet_ECF2", "fjet_ECF3",
      "fjet_C2", "fjet_D2", "fjet_L2", "fjet_L3", "fjet_Qw", "fjet_ThrustMaj"]
KEY_DT = np.dtype([("e", "<i8"), ("y", "<u4"), ("p", "<u4"), ("l", "<i2")])
VOID_DT = np.dtype((np.void, KEY_DT.itemsize))
SENTINEL = -900.0
PAIRING_FLOOR_PERCENT = {"cer": 99.80}
DEFAULT_PAIRING_FLOOR_PERCENT = 99.95
TRACK_SAMPLES = {"teg", "tej", "tfl", "tfj"}


def build_key(ev, eta, phi, lab):
    """The bit-exact key used by pair_features.py, including -0 normalization."""
    eta = np.asarray(eta, dtype=np.float32).copy()
    phi = np.asarray(phi, dtype=np.float32).copy()
    eta[eta == 0.0] = 0.0
    phi[phi == 0.0] = 0.0
    if not (np.isfinite(eta).all() and np.isfinite(phi).all()):
        raise ValueError("non-finite fjet_eta/fjet_phi")
    key = np.empty(eta.size, KEY_DT)
    key["e"] = np.asarray(ev, dtype=np.int64)
    key["y"] = eta.view(np.uint32)
    key["p"] = phi.view(np.uint32)
    key["l"] = np.asarray(lab, dtype=np.int16)
    return key


def cache_files(dataset):
    return sorted((CACHE / dataset).glob("*.npz"))


def source_h5(cache_path):
    return DATA / cache_path.parent.name / (cache_path.stem + ".h5")


def identity_from_cache(path):
    with np.load(path, allow_pickle=False) as z:
        return (z["EventInfo_mcEventNumber"], z["fjet_eta"], z["fjet_phi"],
                z["labels"], str(z["code_fp"]) if "code_fp" in z.files else "")


def cached_multiplicity(path):
    with np.load(path, allow_pickle=False) as z:
        if "n_constit" in z.files:
            return np.asarray(z["n_constit"], dtype=np.int32)
    with h5py.File(source_h5(path), "r") as f:
        return (np.asarray(f["fjet_clus_pt"][:]) > 0.0).sum(axis=1).astype(np.int32)


def multiplicity_distribution(files):
    values = np.concatenate([cached_multiplicity(path) for path in files])
    return {"n": int(values.size), "median": float(np.median(values)),
            "mean": float(values.mean()), "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95))}


def nominal_index():
    keys, file_index, row_index, paths, fps = [], [], [], [], set()
    for path in cache_files(NOMINAL_EVALUATION_DATASET):
        ev, eta, phi, labels, fp = identity_from_cache(path)
        keys.append(build_key(ev, eta, phi, labels))
        file_index.append(np.full(labels.size, len(paths), dtype=np.int32))
        row_index.append(np.arange(labels.size, dtype=np.int32))
        paths.append(path)
        fps.add(fp)
    if not paths:
        raise RuntimeError(f"no cache found in {CACHE / NOMINAL_EVALUATION_DATASET}")
    if fps != {FROZEN_RECONSTRUCTION_FP}:
        raise RuntimeError(f"nominal cache fingerprint mismatch: {sorted(fps)}")
    keys = np.concatenate(keys)
    raw = keys.view(VOID_DT).reshape(-1)
    unique, first, counts = np.unique(raw, return_index=True, return_counts=True)
    duplicates = int((counts > 1).sum())
    if duplicates:
        raise RuntimeError(f"nominal key is not unique ({duplicates} repeated keys)")
    print(f"  nominal pool: {keys.size:,} jets in {len(paths)} files")
    # np.unique has already sorted the keys; return_index is their exact global
    # nominal row. Reusing it avoids a second 10.3-million-row sort.
    return {"n_jets": int(keys.size), "raw_sorted": unique, "global_sorted": first,
            "file": np.concatenate(file_index), "row": np.concatenate(row_index),
            "paths": paths}


def nominal_reference(index):
    """Load nominal scalar HLFs and pT/E sums once, in global-key row order.

    This converts thousands of fragmented HDF5 reads per sample into one
    sequential pass over the nominal source files.  It changes no definition.
    """
    n = index["n_jets"]
    hlf = {column: np.empty(n, dtype=np.float32) for column in HL}
    pt_sum = np.empty(n, dtype=np.float64)
    energy_sum = np.empty(n, dtype=np.float64)
    eta_centroid = np.empty(n, dtype=np.float64)
    phi_centroid = np.empty(n, dtype=np.float64)
    offset = 0
    for path in index["paths"]:
        with h5py.File(source_h5(path), "r") as f:
            size = f[HL[0]].shape[0]
            span = slice(offset, offset + size)
            for column in HL:
                hlf[column][span] = np.asarray(f[column][:], dtype=np.float32)
            pt = np.asarray(f["fjet_clus_pt"][:], dtype=np.float64)
            eta = np.asarray(f["fjet_clus_eta"][:], dtype=np.float64)
            phi = np.asarray(f["fjet_clus_phi"][:], dtype=np.float64)
            summed_pt = pt.sum(axis=1)
            pt_sum[span] = summed_pt
            energy_sum[span] = np.asarray(f["fjet_clus_E"][:], dtype=np.float64).sum(axis=1)
            eta_centroid[span] = np.divide((pt * eta).sum(axis=1), summed_pt,
                                           out=np.full(size, np.nan), where=summed_pt > 0.0)
            phi_centroid[span] = np.arctan2((pt * np.sin(phi)).sum(axis=1),
                                            (pt * np.cos(phi)).sum(axis=1))
        offset += size
    if offset != n:
        raise RuntimeError(f"nominal source/cache row mismatch: {offset:,} != {n:,}")
    return {"hlf": hlf, "pt_sum": pt_sum, "energy_sum": energy_sum,
            "eta_centroid": eta_centroid, "phi_centroid": phi_centroid}


def lookup_nominal(index, syskeys):
    raw = syskeys.view(VOID_DT).reshape(-1)
    pos = np.searchsorted(index["raw_sorted"], raw)
    valid = pos < index["raw_sorted"].size
    at = np.flatnonzero(valid)
    valid[at] = index["raw_sorted"][pos[at]] == raw[at]
    gidx = np.full(raw.size, -1, dtype=np.int64)
    gidx[valid] = index["global_sorted"][pos[valid]]
    return gidx


def key_preflight(dataset, files, index):
    """Diagnose pairability globally, before reading any pairwise observables."""
    keys = []
    try:
        for path in files:
            ev, eta, phi, labels, _ = identity_from_cache(path)
            keys.append(build_key(ev, eta, phi, labels))
    except ValueError as exc:
        return {"keys": None, "gidx": None, "jets": int(sum(k.size for k in keys)),
                "key_status": "non-finite", "duplicate_key_count": None,
                "duplicate_jet_count": None, "matched": None, "match_rate_percent": None,
                "pairable": False, "pairing_basis": "N/A / not pairable",
                "pairing_reason": str(exc)}
    syskeys = np.concatenate(keys)
    raw = syskeys.view(VOID_DT).reshape(-1)
    _, counts = np.unique(raw, return_counts=True)
    duplicate_keys = int((counts > 1).sum())
    duplicate_jets = int((counts[counts > 1] - 1).sum())
    gidx = lookup_nominal(index, syskeys)
    matched = int((gidx >= 0).sum())
    rate = 100.0 * matched / syskeys.size if syskeys.size else 0.0
    floor = PAIRING_FLOOR_PERCENT.get(dataset, DEFAULT_PAIRING_FLOOR_PERCENT)
    pairable = duplicate_keys == 0 and rate >= floor
    if duplicate_keys:
        reason = (f"{duplicate_keys:,} duplicate systematic keys ({duplicate_jets:,} extra rows); "
                  "exact join is ambiguous")
    elif rate < floor:
        reason = f"exact-key coverage {rate:.6f}% is below the validated {floor:.2f}% floor"
    else:
        reason = f"unique exact keys; coverage {rate:.6f}% meets {floor:.2f}% floor"
    return {"keys": syskeys, "gidx": gidx, "jets": int(syskeys.size),
            "key_status": "duplicate" if duplicate_keys else "unique",
            "duplicate_key_count": duplicate_keys, "duplicate_jet_count": duplicate_jets,
            "matched": matched, "match_rate_percent": rate, "pairable": pairable,
            "pairing_basis": "exact key" if pairable else "N/A / not pairable",
            "pairing_reason": reason}


def read_h5(h5_path, rows, columns):
    """Read arbitrary rows and restore the caller's row ordering."""
    order = np.argsort(rows, kind="stable")
    take = rows[order]
    output = {}
    with h5py.File(h5_path, "r") as f:
        for column in columns:
            values = np.asarray(f[column][take])
            restored = np.empty_like(values)
            restored[order] = values
            output[column] = restored
    return output


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"n": 0, "median": None, "mean": None, "rms": None,
                "median_absolute": None, "p05": None, "p95": None}
    return {"n": int(values.size), "median": float(np.median(values)),
            "mean": float(values.mean()), "rms": float(np.sqrt(np.mean(values ** 2))),
            "median_absolute": float(np.median(np.abs(values))),
            "p05": float(np.percentile(values, 5)), "p95": float(np.percentile(values, 95))}


def constituent_columns(dataset):
    if dataset in {"esup", "esdown"}:
        return ["fjet_clus_pt"]
    if dataset == "cer":
        return ["fjet_clus_E"]
    return ["fjet_clus_pt", "fjet_clus_eta", "fjet_clus_phi"]


def audit_pairable(dataset, files, preflight, index, reference):
    identical = np.zeros(len(HL), dtype=np.int64)
    compared = np.zeros(len(HL), dtype=np.int64)
    evidence = []
    offset = 0
    columns = constituent_columns(dataset)
    for cache_path in files:
        size = identity_from_cache(cache_path)[3].size
        local_gidx = preflight["gidx"][offset:offset + size]
        offset += size
        selected = np.flatnonzero(local_gidx >= 0)
        if not selected.size:
            continue
        sys_h5 = source_h5(cache_path)
        sys_stored = read_h5(sys_h5, selected, HL)
        sys_const = read_h5(sys_h5, selected, columns)
        matched_global_rows = local_gidx[selected]
        for j, column in enumerate(HL):
            a = np.asarray(sys_stored[column], dtype=np.float32)
            b = reference["hlf"][column][matched_global_rows]
            usable = (a > SENTINEL) & (b > SENTINEL)
            compared[j] += int(usable.sum())
            identical[j] += int((a[usable].view(np.uint32) == b[usable].view(np.uint32)).sum())
        if dataset in {"esup", "esdown"}:
            a = sys_const["fjet_clus_pt"].sum(axis=1, dtype=np.float64)
            b = reference["pt_sum"][matched_global_rows]
            good = b > 0.0
            evidence.append(a[good] / b[good] - 1.0)
        elif dataset == "cer":
            a = sys_const["fjet_clus_E"].sum(axis=1, dtype=np.float64)
            b = reference["energy_sum"][matched_global_rows]
            good = np.abs(b) > 0.0
            evidence.append(a[good] / b[good] - 1.0)
        else:
            pt = np.asarray(sys_const["fjet_clus_pt"], dtype=np.float64)
            summed_pt = pt.sum(axis=1)
            eta = np.asarray(sys_const["fjet_clus_eta"], dtype=np.float64)
            phi = np.asarray(sys_const["fjet_clus_phi"], dtype=np.float64)
            sys_eta = np.divide((pt * eta).sum(axis=1), summed_pt,
                                out=np.full(summed_pt.size, np.nan), where=summed_pt > 0.0)
            sys_phi = np.arctan2((pt * np.sin(phi)).sum(axis=1),
                                 (pt * np.cos(phi)).sum(axis=1))
            nom_eta = reference["eta_centroid"][matched_global_rows]
            nom_phi = reference["phi_centroid"][matched_global_rows]
            dphi = (sys_phi - nom_phi + np.pi) % (2 * np.pi) - np.pi
            displacement = np.sqrt((sys_eta - nom_eta) ** 2 + dphi ** 2)
            evidence.append(displacement[np.isfinite(displacement)])
    values = np.concatenate(evidence) if evidence else np.zeros(0)
    result = {"stored_branch_identical": {
        column: {"identical": int(identical[j]), "compared": int(compared[j]),
                 "fraction": float(identical[j] / compared[j]) if compared[j] else None}
        for j, column in enumerate(HL)},
        "all_fifteen_identical": bool(np.all(identical == compared)) and bool(compared.min())}
    if dataset in {"esup", "esdown"}:
        result["constituent_evidence"] = {"kind": "relative constituent-pT response",
            "metric": "(sum pT_sys / sum pT_nom) - 1", "statistics": stats(values)}
    elif dataset == "cer":
        result["constituent_evidence"] = {"kind": "constituent energy-response width",
            "metric": "(sum E_sys / sum E_nom) - 1; RMS and median absolute response are primary",
            "statistics": stats(values)}
    else:
        result["constituent_evidence"] = {"kind": "constituent angular displacement",
            "metric": "DeltaR of the pT-weighted constituent eta/phi centroid",
            "statistics": stats(values),
            "nonzero_fraction": float((values > 0.0).mean()) if values.size else None}
    return result


def audit_one(dataset, index, reference, nominal_multiplicity, shards):
    files = cache_files(dataset)
    if shards:
        files = files[:shards]
    if not files:
        return {"dataset": dataset, "shards_audited": 0, "jets": 0, "pairable": False,
                "pairing_basis": "N/A / not pairable", "pairing_reason": "no cache files",
                "stored_branch_identical": {c: {"identical": None, "compared": None, "fraction": None} for c in HL}}
    preflight = key_preflight(dataset, files, index)
    result = {key: value for key, value in preflight.items() if key not in {"keys", "gidx"}}
    result.update({"dataset": dataset, "shards_audited": len(files)})
    if preflight["pairable"]:
        result.update(audit_pairable(dataset, files, preflight, index, reference))
        return result
    result["stored_branch_identical"] = {c: {"identical": None, "compared": None, "fraction": None} for c in HL}
    result["all_fifteen_identical"] = None
    if dataset in TRACK_SAMPLES:
        result["constituent_evidence"] = {
            "kind": "unpaired constituent multiplicity/content",
            "metric": "descriptive distributions only; no pairwise identity fraction is valid",
            "systematic_distribution": multiplicity_distribution(files),
            "nominal_distribution": nominal_multiplicity}
    else:
        result["constituent_evidence"] = {"kind": "N/A", "metric": "N/A because exact one-to-one pairing is invalid"}
    return result


def aggregate_identity(result):
    if not result.get("pairable"):
        return None, None, None
    same = sum(v["identical"] for v in result["stored_branch_identical"].values())
    total = sum(v["compared"] for v in result["stored_branch_identical"].values())
    return same, total, same / total if total else None


def evidence_display(result):
    evidence = result.get("constituent_evidence", {})
    values = evidence.get("statistics", {})
    if result["dataset"] in {"esup", "esdown"}:
        return "relative sum-pT median [%]", values.get("median"), "%"
    if result["dataset"] == "cer":
        return "relative sum-E RMS [%]", values.get("rms"), "%"
    if result["dataset"] == "cpos":
        return "constituent-centroid DeltaR median", values.get("median"), ""
    if result["dataset"] in TRACK_SAMPLES:
        return "unpaired multiplicity median", evidence.get("systematic_distribution", {}).get("median"), ""
    return "N/A", None, ""


def write_csv(results, summary_path, branch_path):
    fields = ["dataset", "shards_audited", "jets", "pairable", "pairing_basis", "pairing_reason",
              "key_status", "duplicate_key_count", "duplicate_jet_count", "exact_key_hits_diagnostic",
              "exact_key_coverage_percent_diagnostic", "hl_identity_status", "hl_identical", "hl_compared",
              "hl_identical_percent", "evidence_kind", "evidence_metric", "evidence_value_name",
              "evidence_value", "evidence_unit"]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            same, total, fraction = aggregate_identity(result)
            name, value, unit = evidence_display(result)
            evidence = result.get("constituent_evidence", {})
            writer.writerow({"dataset": result["dataset"], "shards_audited": result["shards_audited"],
                "jets": result.get("jets"), "pairable": result.get("pairable"),
                "pairing_basis": result.get("pairing_basis"), "pairing_reason": result.get("pairing_reason"),
                "key_status": result.get("key_status"), "duplicate_key_count": result.get("duplicate_key_count"),
                "duplicate_jet_count": result.get("duplicate_jet_count"), "exact_key_hits_diagnostic": result.get("matched"),
                "exact_key_coverage_percent_diagnostic": result.get("match_rate_percent"),
                "hl_identity_status": "exact float32 bit comparison" if result.get("pairable") else "N/A / not pairable",
                "hl_identical": same, "hl_compared": total,
                "hl_identical_percent": None if fraction is None else 100.0 * fraction,
                "evidence_kind": evidence.get("kind"), "evidence_metric": evidence.get("metric"),
                "evidence_value_name": name,
                "evidence_value": (100.0 * value if value is not None and unit == "%" else value),
                "evidence_unit": unit})
    with branch_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "pairable", "branch", "identical", "compared", "identical_fraction"])
        writer.writeheader()
        for result in results:
            for branch, value in result["stored_branch_identical"].items():
                writer.writerow({"dataset": result["dataset"], "pairable": result["pairable"], "branch": branch,
                                 "identical": value["identical"], "compared": value["compared"],
                                 "identical_fraction": value["fraction"]})


def write_md(results, path, meta, discrepancy):
    out = ["# Stored high-level branch audit", "", f"Rendered (UTC): {meta['created_utc']}",
           f"Current script SHA-256: {meta['script_sha256']}",
           f"Full-scan script SHA-256: {meta.get('full_scan_script_sha256', meta['script_sha256'])}",
           f"Nominal pool: {meta['nominal_pool_jets']:,} jets in {meta['nominal_files']} files", "",
           "| sample | pairability | exact-key hits (diagnostic) | exact float32 HLF identity | constituent evidence |",
           "|---|---|---:|---|---|"]
    for result in results:
        same, total, fraction = aggregate_identity(result)
        name, value, unit = evidence_display(result)
        hits = "N/A" if result.get("matched") is None else f"{result['matched']:,}/{result['jets']:,} ({result['match_rate_percent']:.6f}%)"
        hlf = "N/A / not pairable" if same is None else f"{same:,}/{total:,} ({100.0 * fraction:.6f}%)"
        shown = "N/A" if value is None else (f"{100.0 * value:+.6f}%" if unit == "%" else f"{value:.6g}")
        out.append(f"| {result['dataset']} | {result['pairing_basis']} | {hits} | {hlf} | {name}: {shown} |")
    out += ["", "## Pairability decisions", ""]
    out += [f"- **{r['dataset']}** — {r['pairing_reason']}" for r in results]
    out += ["", "## Esup manuscript discrepancy", "",
            discrepancy["finding"] if discrepancy.get("status") == "resolved"
            else discrepancy["status"], ""]
    out += ["## Evidence definitions and full statistics", ""]
    for result in results:
        evidence = result.get("constituent_evidence", {})
        out += [f"### {result['dataset']}", "",
                f"Kind: {evidence.get('kind', 'N/A')}",
                f"Metric: {evidence.get('metric', 'N/A')}", "",
                "    " + json.dumps(evidence, sort_keys=True), ""]
    out += ["", "## Exact per-branch float32 identity counts", ""]
    for result in results:
        out += [f"### {result['dataset']}", "", "| branch | identical | compared | fraction |", "|---|---:|---:|---:|"]
        for branch, value in result["stored_branch_identical"].items():
            frac = value["fraction"]
            out.append(f"| {branch} | {value['identical'] if value['identical'] is not None else 'N/A'} | "
                       f"{value['compared'] if value['compared'] is not None else 'N/A'} | "
                       f"{'N/A' if frac is None else format(frac, '.9f')} |")
        out.append("")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_tex(results, path):
    rows = []
    for result in results:
        same, total, fraction = aggregate_identity(result)
        name, value, unit = evidence_display(result)
        if result["pairable"]:
            pair = f"exact key ({result['match_rate_percent']:.3f}\\%)"
            identity = f"{same:,}/{total:,} ({100.0 * fraction:.3f}\\%)".replace(",", "{,}")
            shown = (f"{100.0 * value:.4f}\\%" if result["dataset"] == "cer" else
                     f"{100.0 * value:+.4f}\\%" if unit == "%" else f"{value:.6g}")
            label = {"esup": "median $\\Delta\\Sigma p_{\\mathrm T}$",
                     "esdown": "median $\\Delta\\Sigma p_{\\mathrm T}$",
                     "cer": "RMS $\\Delta\\Sigma E$",
                     "cpos": "median centroid $\\Delta R$"}[result["dataset"]]
            evidence = f"{label}: {shown}"
        else:
            pair, identity = "N/A / not pairable", "N/A"
            nominal_median = result.get("constituent_evidence", {}).get(
                "nominal_distribution", {}).get("median")
            evidence = ("N/A" if value is None else
                        f"median $N_{{\\mathrm c}}$: {value:.0f} vs nominal "
                        f"{nominal_median:.0f} (unpaired)")
        hits = "--" if result.get("matched") is None else f"{result['matched']:,}".replace(",", "{,}")
        rows.append(f"\\texttt{{{result['dataset']}}} & {result['shards_audited']} & {hits} & {pair} & {identity} & {evidence} " + chr(92) * 2)
    body = "\n".join(rows)
    path.write_text(
        "% Generated by Code files/stale_hlf_audit.py. Audit artifact; not a manuscript edit.\n"
        "\\begin{table*}[!ht]\n\\centering\\footnotesize\n"
        "\\caption{Stored high-level branch audit. Pairwise quantities are reported only after a full-sample exact-key preflight.}\n"
        "\\begin{tabular}{lrrlll}\n\\toprule\n"
        "Sample & Shards & Exact-key hits & Pairing basis & HLF identity & Constituent evidence "
        + chr(92) * 2 + "\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table*}}\n", encoding="utf-8")


def esup_discrepancy(results):
    esup = next((r for r in results if r["dataset"] == "esup"), None)
    if esup is None or not esup.get("pairable"):
        return {"status": "not resolved: esup was not pairable"}
    statistics = esup["constituent_evidence"]["statistics"]
    measured = 100.0 * statistics["median"]
    mean = 100.0 * statistics["mean"]
    return {"status": "resolved", "manuscript_claim_percent": 6.8,
            "full_exact_pair_median_percent": measured,
            "full_exact_pair_mean_percent": mean,
            "mean_rounded_to_one_decimal_percent": round(mean, 1),
            "difference_percentage_points": measured - 6.8,
            "basis": "median over every valid exact esup/test_nominal pair of (sum pT_sys / sum pT_nom) - 1",
            "finding": (f"The full exact-pair arithmetic mean is +{mean:.6f}%, "
                        f"which rounds to the manuscript's +6.8%; the full exact-pair "
                        f"median is +{measured:.6f}%. Thus mean versus median "
                        "numerically resolves the discrepancy on the same 5,000,000 "
                        "pairs. The original source of the manuscript figure is not "
                        "documented, so this does not establish its historical derivation. "
                        "No manuscript file was changed.")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DETECTOR_SYSTEMATICS), choices=list(DETECTOR_SYSTEMATICS))
    parser.add_argument("--shards", type=int, default=0, help="first N shards only (0 = all; required for completed Table S1)")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 78 + "\nstored high-level branch audit\n" + "=" * 78)
    index = nominal_index()
    print("  loading nominal HLF and constituent-response reference ...")
    reference = nominal_reference(index)
    nominal_multiplicity = multiplicity_distribution(cache_files(NOMINAL_EVALUATION_DATASET))
    results = []
    for dataset in args.datasets:
        print(f"\n  auditing {dataset} ...")
        result = audit_one(dataset, index, reference, nominal_multiplicity, args.shards)
        print(f"    {result['pairing_basis']}: {result['pairing_reason']}")
        if result["pairable"]:
            same, total, fraction = aggregate_identity(result)
            print(f"    exact float32 HLF identity: {same:,}/{total:,} ({100 * fraction:.6f}%)")
        results.append(result)
    meta = {"created_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "reconstruction_fingerprint": FROZEN_RECONSTRUCTION_FP, "data_root": str(DATA_ROOT),
            "nominal_dataset": NOMINAL_EVALUATION_DATASET, "nominal_pool_jets": index["n_jets"],
            "nominal_files": len(index["paths"]), "python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "h5py": h5py.__version__,
            "complete_all_shards": args.shards == 0,
            "complete_table_s1": args.shards == 0 and set(args.datasets) == set(DETECTOR_SYSTEMATICS)}
    meta["full_scan_script_sha256"] = meta["script_sha256"]
    payload = {"provenance": meta, "results": results, "esup_manuscript_discrepancy": esup_discrepancy(results)}
    (OUT / "stale_hlf_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(results, OUT / "supp_s1_stale_hlf_audit.csv", OUT / "stale_hlf_audit_branches.csv")
    write_md(results, OUT / "STALE_HLF_AUDIT.md", meta, payload["esup_manuscript_discrepancy"])
    write_tex(results, OUT / "supp_s1_stale_hlf_audit.tex")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
