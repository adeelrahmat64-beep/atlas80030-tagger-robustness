"""Audit-only diagnostics for strict_closure_audit.py outliers.

No production source or data are changed.  The controls implemented here are
deliberately separate from recompute_hl.py:

* replay within the rounding cell of the stored float32 constituents;
* a raw-energy/rapidity control that omits the production spacelike clamp;
* a direct transcription of the seed order and early-stop control flow in the
  cited ATLAS JetSubStructureUtils Thrust.cxx implementation;
* exclusive-kt merge-gap diagnostics for near-degenerate clustering.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np


FOCUS = {"Tau3", "Tau4", "L2", "L3", "ThrustMaj"}


def import_module(path, name):
    module_dir = str(path.resolve().parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_failed_rows(eventwise_path):
    rows = []
    with eventwise_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["above_hybrid_tolerance"] == "True":
                rows.append(row)
    return rows


def load_unique_jets(data_root, failed, audit):
    keys = sorted(set(
        (row["dataset"], row["relative_dataset_path"], row["file"],
         int(row["jet_index"]))
        for row in failed
    ))
    clus_names = ("fjet_clus_pt", "fjet_clus_eta", "fjet_clus_phi",
                  "fjet_clus_E")
    clus = {name: [] for name in clus_names}
    stored = {name: [] for name in audit.FEATURE_ORDER}
    metadata = []
    for dataset, relative, filename, jet_index in keys:
        path = data_root / "data" / relative / filename
        with h5py.File(path, "r") as handle:
            for name in clus_names:
                clus[name].append(
                    handle[name][jet_index:jet_index + 1, :200].astype(np.float64)
                )
            for feature in audit.FEATURE_ORDER:
                stored[feature].append(
                    handle[audit.STORED_BRANCH[feature]][jet_index:jet_index + 1]
                    .astype(np.float64)
                )
            event = (int(handle["EventInfo_mcEventNumber"][jet_index])
                     if "EventInfo_mcEventNumber" in handle else None)
            label = int(handle["labels"][jet_index])
        metadata.append({
            "dataset": dataset, "relative_dataset_path": relative,
            "file": filename, "jet_index": jet_index,
            "event_identifier": event, "label": label,
        })
    return (
        {name: np.concatenate(values) for name, values in clus.items()},
        {name: np.concatenate(values) for name, values in stored.items()},
        metadata,
        keys,
    )


def raw_constituent_rapidity(pt, eta, energy, n_constit):
    out = np.zeros_like(pt, dtype=np.float64)
    for k, n in enumerate(n_constit):
        n = int(n)
        pz = pt[k, :n] * np.sinh(eta[k, :n])
        a = energy[k, :n] + pz
        b = energy[k, :n] - pz
        good = (a > 0.0) & (b > 0.0)
        out[k, :n][good] = 0.5 * np.log(a[good] / b[good])
        out[k, :n][~good] = eta[k, :n][~good]
    return out


def raw_rapidity_controls(reco, audit, clus, n_constit):
    pt = clus["fjet_clus_pt"]
    eta = clus["fjet_clus_eta"]
    phi = clus["fjet_clus_phi"]
    energy = clus["fjet_clus_E"]
    yraw = raw_constituent_rapidity(pt, eta, energy, n_constit)

    # WTA clustering depends on pt, rapidity and phi.  A massless replay with
    # eta=yraw supplies precisely those coordinates without changing the
    # production kernel.
    massless_energy = pt * np.cosh(yraw)
    _, _, axeta, axphi, naxes = reco.kt_observables(
        pt, yraw, phi, massless_energy, n_constit, 4, 0
    )
    tau_raw = reco.nsubjettiness(
        pt, yraw, phi, massless_energy, n_constit, axeta, axphi, naxes,
        4, 1.0, 0,
    )
    sumpt = np.sum(pt, axis=1)

    ecf = reco.ecf123(pt, yraw, phi, n_constit, beta=1.0, block=64)
    gen3 = reco.gen_ecf3(pt, yraw, phi, n_constit, 1.0)
    gen2 = reco.gen_ecf2(pt, yraw, phi, n_constit, 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = tau_raw / sumpt[:, None]
        c2 = ecf[:, 2] * ecf[:, 0] / ecf[:, 1] ** 2
        d2 = ecf[:, 2] * ecf[:, 0] ** 3 / ecf[:, 1] ** 3
        l2 = gen3[:, 2] / gen2 ** 1.5
        l3 = gen3[:, 0] / gen3[:, 2] ** (1.0 / 3.0)
    values = {
        "Tau1": tau[:, 0], "Tau2": tau[:, 1],
        "Tau3": tau[:, 2], "Tau4": tau[:, 3],
        "ECF1": ecf[:, 0] * audit.UNIT_SCALE["ECF1"],
        "ECF2": ecf[:, 1] * audit.UNIT_SCALE["ECF2"],
        "ECF3": ecf[:, 2] * audit.UNIT_SCALE["ECF3"],
        "C2": c2, "D2": d2, "L2": l2, "L3": l3,
    }
    return values, yraw


def raw_energy_kt_controls(reco, audit, clus, n_constit):
    pt = clus["fjet_clus_pt"]
    eta = clus["fjet_clus_eta"]
    phi = clus["fjet_clus_phi"]
    energy = clus["fjet_clus_E"]
    split12 = np.zeros(len(n_constit))
    split23 = np.zeros(len(n_constit))
    qw = np.zeros(len(n_constit))
    for k, n0 in enumerate(n_constit):
        n = int(n0)
        px = pt[k, :n] * np.cos(phi[k, :n])
        py = pt[k, :n] * np.sin(phi[k, :n])
        pz = pt[k, :n] * np.sinh(eta[k, :n])
        en = energy[k, :n].copy()  # diagnostic: no E >= |p| clamp
        d = np.zeros(4, dtype=np.float64)
        sx = np.zeros((4, 4), dtype=np.float64)
        sy = np.zeros((4, 4), dtype=np.float64)
        sz = np.zeros((4, 4), dtype=np.float64)
        se = np.zeros((4, 4), dtype=np.float64)
        count = np.zeros(4, dtype=np.int64)
        reco._kt_cluster(px, py, pz, en, n, 0, d, sx, sy, sz, se, count)
        split12[k] = math.sqrt(max(d[0], 0.0)) * audit.UNIT_SCALE["Split12"]
        split23[k] = math.sqrt(max(d[1], 0.0)) * audit.UNIT_SCALE["Split23"]
        if count[2] == 3:
            masses = []
            for i in range(3):
                for j in range(i + 1, 3):
                    ee = se[2, i] + se[2, j]
                    xx = sx[2, i] + sx[2, j]
                    yy = sy[2, i] + sy[2, j]
                    zz = sz[2, i] + sz[2, j]
                    masses.append(math.sqrt(max(ee * ee - xx * xx - yy * yy
                                                - zz * zz, 0.0)))
            qw[k] = min(masses) * audit.UNIT_SCALE["Qw"]
    return {"Split12": split12, "Split23": split23, "Qw": qw}


ADD = np.asarray([
    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
    [1, 1, 1, 1], [1, 1, 1, -1], [1, 1, -1, 1], [1, 1, -1, -1],
    [1, -1, 1, 1], [1, -1, 1, -1], [1, -1, -1, 1], [1, -1, -1, -1],
    [-1, 1, 1, 1], [-1, 1, 1, -1], [-1, 1, -1, 1], [-1, 1, -1, -1],
    [-1, -1, 1, 1], [-1, -1, 1, -1], [-1, -1, -1, 1],
    [-1, -1, -1, -1],
], dtype=np.float64)


def unit(vector):
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else vector.copy()


def atlas_thrust_one(pt, eta, phi, energy, n, clamp_energy):
    if n < 4:
        return None
    px = pt[:n] * np.cos(phi[:n])
    py = pt[:n] * np.sin(phi[:n])
    pz = pt[:n] * np.sinh(eta[:n])
    momentum = np.sqrt(px * px + py * py + pz * pz)
    ee = np.maximum(energy[:n], momentum) if clamp_energy else energy[:n].copy()
    jet_p = np.asarray([px.sum(), py.sum(), pz.sum()])
    jet_e = float(ee.sum())
    m2 = jet_e * jet_e - float(np.dot(jet_p, jet_p))
    if m2 <= 0.0:
        return None
    gamma = jet_e / math.sqrt(m2)
    beta = jet_p / jet_e
    b2 = float(np.dot(beta, beta))
    p = np.column_stack((px, py, pz))
    bp = p @ beta
    factor = ((gamma - 1.0) * bp / b2 - gamma * ee) if b2 > 0 else np.zeros(n)
    boosted = p + factor[:, None] * beta[None, :]
    boosted_e = gamma * (ee - bp)

    leading = boosted[np.argsort(-boosted_e)[:4]]
    max_tests = min(20, n)
    axes = []
    agree = 0
    disagree = 0
    test = 0
    while True:
        axis = unit(ADD[test] @ leading)
        for loop in range(11):
            dots = boosted @ axis
            next_axis = unit(np.sum(
                np.where((dots > 0.0)[:, None], boosted, -boosted), axis=0
            ))
            same = np.array_equal(axis, next_axis)
            opposite = np.array_equal(-axis, next_axis)
            axis = next_axis
            if same or opposite or loop == 10:
                break
        axes.append(axis)
        if test > 0:
            if np.array_equal(axes[0], axis) or np.array_equal(-axes[0], axis):
                agree += 1
            else:
                disagree += 1
        condition = disagree > 0 or agree < 4
        test += 1
        if not condition or test >= max_tests:
            break

    denominator = float(np.linalg.norm(boosted, axis=1).sum())
    if denominator < 1.0e-20:
        return None
    used_axes = axes if disagree > 0 else axes[:1]
    return max(float(np.abs(boosted @ axis).sum() / denominator)
               for axis in used_axes)


def clustering_gap(pt, eta, phi, energy, n, wta):
    px = pt[:n] * np.cos(phi[:n])
    py = pt[:n] * np.sin(phi[:n])
    pz = pt[:n] * np.sinh(eta[:n])
    momentum = np.sqrt(px * px + py * py + pz * pz)
    ee = np.maximum(energy[:n], momentum)
    vectors = [np.asarray([px[i], py[i], pz[i], ee[i]], dtype=np.float64)
               for i in range(n)]
    min_gap = math.inf
    min_gap_multiplicity = None
    final_four_gap = math.inf
    while len(vectors) > 1:
        candidates = []
        for i in range(len(vectors)):
            vi = vectors[i]
            pti2 = vi[0] ** 2 + vi[1] ** 2
            yi = 0.5 * math.log(max(vi[3] + vi[2], 1e-300)
                                / max(vi[3] - vi[2], 1e-300))
            phii = math.atan2(vi[1], vi[0])
            for j in range(i + 1, len(vectors)):
                vj = vectors[j]
                ptj2 = vj[0] ** 2 + vj[1] ** 2
                yj = 0.5 * math.log(max(vj[3] + vj[2], 1e-300)
                                    / max(vj[3] - vj[2], 1e-300))
                phij = math.atan2(vj[1], vj[0])
                dphi = phii - phij
                while dphi > math.pi:
                    dphi -= 2.0 * math.pi
                while dphi < -math.pi:
                    dphi += 2.0 * math.pi
                distance = min(pti2, ptj2) * ((yi - yj) ** 2 + dphi ** 2)
                candidates.append((distance, i, j))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        best, i, j = candidates[0]
        second = candidates[1][0] if len(candidates) > 1 else math.inf
        gap = ((second - best) / max(abs(best), 1e-300)
               if math.isfinite(second) else math.inf)
        if gap < min_gap:
            min_gap = gap
            min_gap_multiplicity = len(vectors)
        if len(vectors) <= 5:
            final_four_gap = min(final_four_gap, gap)
        vi = vectors[i]
        vj = vectors[j]
        if wta:
            pti = math.hypot(vi[0], vi[1])
            ptj = math.hypot(vj[0], vj[1])
            keep = vi if pti >= ptj else vj
            ky = 0.5 * math.log(max(keep[3] + keep[2], 1e-300)
                                / max(keep[3] - keep[2], 1e-300))
            kp = math.atan2(keep[1], keep[0])
            newpt = pti + ptj
            merged = np.asarray([
                newpt * math.cos(kp), newpt * math.sin(kp),
                newpt * math.sinh(ky), newpt * math.cosh(ky),
            ])
        else:
            merged = vi + vj
        vectors[i] = merged
        del vectors[j]
    return {
        "minimum_relative_pair_distance_gap": min_gap,
        "multiplicity_at_minimum_gap": min_gap_multiplicity,
        "minimum_relative_gap_at_multiplicity_5_or_less": final_four_gap,
        "near_degenerate_below_1e-6": bool(min_gap < 1.0e-6),
    }


def rounding_cell_jitter(clus, selected_indices, replicas, seed):
    rng = np.random.default_rng(seed)
    output = {name: [] for name in clus}
    owners = []
    for index in selected_indices:
        pt = clus["fjet_clus_pt"][index]
        n = int(np.sum(pt > 0.0))
        for _ in range(replicas):
            for name, array in clus.items():
                original = array[index].astype(np.float32)
                previous = np.nextafter(original, np.float32(-np.inf)).astype(np.float64)
                following = np.nextafter(original, np.float32(np.inf)).astype(np.float64)
                center = original.astype(np.float64)
                low = 0.5 * (previous + center)
                high = 0.5 * (following + center)
                jittered = center.copy()
                jittered[:n] = rng.uniform(low[:n], high[:n])
                output[name].append(jittered[None, :])
            owners.append(index)
    return {name: np.concatenate(values) for name, values in output.items()}, owners


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--replicas", type=int, default=64)
    parser.add_argument("--seed", type=int, default=91403)
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    audit = import_module(here / "strict_closure_audit.py", "strict_audit_module")
    reco = import_module(here / "recompute_hl.py", "atlas_recompute_diag")
    reco.STORE = args.data_root.resolve()
    reco.DATA = reco.STORE / "data"

    failed = load_failed_rows(args.audit_dir / "strict_closure_eventwise.csv")
    clus, stored, metadata, keys = load_unique_jets(args.data_root, failed, audit)
    current, extras = audit.calculate_features(reco, clus)
    n_constit = extras["n_constit"]
    raw_rap, _ = raw_rapidity_controls(reco, audit, clus, n_constit)
    raw_kt = raw_energy_kt_controls(reco, audit, clus, n_constit)

    thrust_atlas_clamped = []
    thrust_atlas_raw = []
    merge_gaps = []
    for i, n0 in enumerate(n_constit):
        n = int(n0)
        args_one = (
            clus["fjet_clus_pt"][i], clus["fjet_clus_eta"][i],
            clus["fjet_clus_phi"][i], clus["fjet_clus_E"][i], n,
        )
        thrust_atlas_clamped.append(atlas_thrust_one(*args_one, True))
        thrust_atlas_raw.append(atlas_thrust_one(*args_one, False))
        merge_gaps.append({
            "wta": clustering_gap(*args_one, True),
            "e_scheme": clustering_gap(*args_one, False),
        })

    focus_indices = sorted(set(
        keys.index((row["dataset"], row["relative_dataset_path"], row["file"],
                    int(row["jet_index"])))
        for row in failed if row["feature"] in FOCUS
    ))
    jitter_clus, owners = rounding_cell_jitter(
        clus, focus_indices, args.replicas, args.seed
    )
    jitter_values, _ = audit.calculate_features(reco, jitter_clus)

    with (args.audit_dir / "strict_closure_audit.json").open(encoding="utf-8") as handle:
        main_report = json.load(handle)

    records = []
    for row in failed:
        key = (row["dataset"], row["relative_dataset_path"], row["file"],
               int(row["jet_index"]))
        index = keys.index(key)
        feature = row["feature"]
        stat = main_report["closure_all_samples"][feature]
        s = float(stored[feature][index])
        cur = float(current[feature][index])
        abs_tol = float(stat["absolute_tolerance"])
        protection = float(stat["protected_relative_denominator_floor"])

        if feature in raw_rap:
            raw_value = float(raw_rap[feature][index])
        elif feature in raw_kt:
            raw_value = float(raw_kt[feature][index])
        else:
            raw_value = None

        focus = feature in FOCUS
        if focus:
            positions = [j for j, owner in enumerate(owners) if owner == index]
            variants = jitter_values[feature][positions]
            jitter = {
                "replicas": len(positions),
                "min": float(np.min(variants)),
                "median": float(np.median(variants)),
                "max": float(np.max(variants)),
                "standard_deviation": float(np.std(variants)),
                "stored_inside_envelope_with_absolute_tolerance": bool(
                    np.min(variants) - abs_tol <= s <= np.max(variants) + abs_tol
                ),
                "relative_output_range": float(
                    (np.max(variants) - np.min(variants))
                    / max(abs(s), abs(cur), protection)
                ),
            }
        else:
            jitter = None

        atlas_clamped = (thrust_atlas_clamped[index]
                         if feature == "ThrustMaj" else None)
        atlas_raw = thrust_atlas_raw[index] if feature == "ThrustMaj" else None
        current_error = abs(cur - s)
        raw_error = abs(raw_value - s) if raw_value is not None else None
        atlas_error = (abs(atlas_clamped - s)
                       if atlas_clamped is not None else None)

        if feature == "ThrustMaj" and atlas_error is not None and atlas_error <= abs_tol:
            diagnosis = "ATLAS seed ordering/early-stop mismatch established"
        elif jitter and jitter["stored_inside_envelope_with_absolute_tolerance"]:
            if (feature.startswith("Tau")
                    and jitter["relative_output_range"] > audit.REL_TOL):
                diagnosis = "float32 rounding triggers WTA clustering/axis instability"
            else:
                diagnosis = "compatible with float32 constituent quantization"
        elif raw_error is not None and raw_error < 0.2 * current_error:
            diagnosis = "strong raw-energy/rapidity sensitivity; upstream precision cause not proven"
        else:
            diagnosis = "unresolved; current sample does not prove a cause"

        records.append({
            **metadata[index],
            "feature": feature,
            "stored_value": s,
            "current_reconstructed_value": cur,
            "current_absolute_error": current_error,
            "current_protected_relative_error": current_error / max(
                abs(s), abs(cur), protection
            ),
            "raw_energy_or_rapidity_control_value": raw_value,
            "raw_control_absolute_error": raw_error,
            "atlas_exact_seed_control_value": atlas_clamped,
            "atlas_exact_seed_raw_energy_value": atlas_raw,
            "atlas_exact_seed_absolute_error": atlas_error,
            "rounding_cell_replay": jitter,
            "clustering_gap_diagnostics": merge_gaps[index],
            "source_diagnostics": audit.source_diagnostics(
                clus["fjet_clus_pt"][index], clus["fjet_clus_eta"][index],
                clus["fjet_clus_phi"][index], clus["fjet_clus_E"][index],
                n_constit[index], reco.MAX_CONSTITS,
            ),
            "diagnosis": diagnosis,
        })

    summary = {}
    for feature in audit.FEATURE_ORDER:
        selected = [record for record in records if record["feature"] == feature]
        if selected:
            summary[feature] = {
                "outlier_count": len(selected),
                "diagnosis_counts": {
                    diagnosis: sum(record["diagnosis"] == diagnosis
                                   for record in selected)
                    for diagnosis in sorted(set(record["diagnosis"]
                                                for record in selected))
                },
            }

    result = {
        "kind": "strict closure outlier diagnostics",
        "production_source_modified": False,
        "failed_feature_row_count": len(failed),
        "unique_failed_jet_count": len(keys),
        "rounding_cell_replicas_per_focus_jet": args.replicas,
        "summary": summary,
        "records": records,
    }
    audit.dump_json(args.audit_dir / "outlier_diagnostics.json", result)
    print(f"failed feature rows: {len(failed)}")
    print(f"unique failed jets: {len(keys)}")
    for feature, info in summary.items():
        print(feature, info)
    print(args.audit_dir / "outlier_diagnostics.json")


if __name__ == "__main__":
    main()
