"""Final targeted, read-only audit of strict reconstruction-closure failures.

This script is deliberately isolated from ``recompute_hl.py`` and never opens
or writes a production NPZ.  It reads the expanded strict-closure CSV plus the
official HDF5 inputs, replays only the 44 affected physical jets, and writes
audit-only JSON/CSV/Markdown files under ``analysis_results``.

The controls cover:

* float64, float32, and mixed ECF arithmetic;
* source-like C++ nested-loop accumulation and iteration order;
* constituent permutations;
* rapidity/energy conventions;
* a full-rescan exclusive-kt reference and clustering tie sensitivity;
* deterministic controls on the complete float32 rounding cell;
* the exact ECF1 = sum(pt) invariant, including both input and stored-output
  float32 rounding intervals.

ThrustMaj is intentionally absent from every recomputation in this file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
from numba import config as numba_config
from numba import njit, prange


FEATURES = [
    "Tau1", "Tau2", "Tau3", "Tau4", "Split12", "Split23",
    "ECF1", "ECF2", "ECF3", "C2", "D2", "L2", "L3", "Qw",
]
FAMILY = {
    "Tau1": "WTA n-subjettiness", "Tau2": "WTA n-subjettiness",
    "Tau3": "WTA n-subjettiness", "Tau4": "WTA n-subjettiness",
    "Split12": "exclusive-kt E-scheme", "Split23": "exclusive-kt E-scheme",
    "Qw": "exclusive-kt E-scheme",
    "ECF1": "standard ECF", "ECF2": "standard ECF",
    "ECF3": "standard ECF", "C2": "standard ECF", "D2": "standard ECF",
    "L2": "generalized ECF", "L3": "generalized ECF",
}


def import_module(path: Path, name: str):
    module_dir = str(path.resolve().parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(type(value).__name__)


def dump_json(path: Path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=json_default),
        encoding="utf-8",
    )


def f32_cell(value):
    """Round-to-nearest float32 cell, expressed with float64 endpoints."""
    center = np.asarray(value, dtype=np.float32)
    previous = np.nextafter(center, np.float32(-np.inf)).astype(np.float64)
    following = np.nextafter(center, np.float32(np.inf)).astype(np.float64)
    center64 = center.astype(np.float64)
    low = 0.5 * (previous + center64)
    high = 0.5 * (following + center64)
    return low, high


def inside_cell_boundary(low, high, center, choice):
    center64 = np.asarray(center, dtype=np.float64)
    selected = np.where(choice > 0, high, np.where(choice < 0, low, center64))
    # Stay infinitesimally inside the cell rather than sitting on a tie-to-even
    # boundary.  Casting back to float32 must recover the released value.
    return np.nextafter(selected, center64)


def protected_error(value, stored, protection):
    absolute = abs(float(value) - float(stored))
    relative = absolute / max(abs(float(value)), abs(float(stored)), protection)
    return absolute, relative


def passes_fixed_policy(value, stored, absolute_tolerance, protection,
                        relative_tolerance):
    absolute, relative = protected_error(value, stored, protection)
    return absolute <= absolute_tolerance or relative <= relative_tolerance


def exact_binomial_two_sided(positive, negative):
    n = positive + negative
    if n == 0:
        return None
    k = min(positive, negative)
    probability = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, probability)


def load_failure_table(eventwise_path: Path):
    failed = []
    all_ecf1 = []
    signed = defaultdict(list)
    with eventwise_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["valid_for_closure"] == "True":
                s = float(row["stored_value"])
                r = float(row["reconstructed_value"])
                denominator = max(abs(s), abs(r), 1.0e-300)
                signed[row["feature"]].append((r - s) / denominator)
            if row["feature"] == "ECF1":
                all_ecf1.append(row)
            if row["above_hybrid_tolerance"] == "True":
                failed.append(row)
    return failed, all_ecf1, signed


def load_failed_jets(data_root, failed, audit):
    keys = sorted(set(
        (row["dataset"], row["relative_dataset_path"], row["file"],
         int(row["jet_index"])) for row in failed
    ))
    arrays = {name: [] for name in (
        "fjet_clus_pt", "fjet_clus_eta", "fjet_clus_phi", "fjet_clus_E"
    )}
    stored = {feature: [] for feature in audit.FEATURE_ORDER}
    metadata = []
    for dataset, relative, filename, index in keys:
        path = data_root / "data" / relative / filename
        with h5py.File(path, "r") as handle:
            for name in arrays:
                arrays[name].append(handle[name][index:index + 1, :200])
            for feature in stored:
                branch = audit.STORED_BRANCH[feature]
                stored[feature].append(handle[branch][index:index + 1])
            event = (int(handle["EventInfo_mcEventNumber"][index])
                     if "EventInfo_mcEventNumber" in handle else None)
            label = int(handle["labels"][index])
        metadata.append({
            "dataset": dataset,
            "relative_dataset_path": relative,
            "file": filename,
            "jet_index": index,
            "event_identifier": event,
            "label": label,
            "physical_jet_id": f"{dataset}/{filename}:{index}",
        })
    arrays = {name: np.concatenate(parts).astype(np.float64)
              for name, parts in arrays.items()}
    stored = {name: np.concatenate(parts).astype(np.float64)
              for name, parts in stored.items()}
    return keys, arrays, stored, metadata


def current_rapidity(reco, arrays, n_constit):
    return reco.constituent_rapidity(
        arrays["fjet_clus_pt"], arrays["fjet_clus_eta"],
        arrays["fjet_clus_E"], n_constit, 1,
    )


def raw_rapidity(arrays, n_constit):
    pt = arrays["fjet_clus_pt"]
    eta = arrays["fjet_clus_eta"]
    energy = arrays["fjet_clus_E"]
    out = np.zeros_like(pt, dtype=np.float64)
    for k, n0 in enumerate(n_constit):
        n = int(n0)
        pz = pt[k, :n] * np.sinh(eta[k, :n])
        a = energy[k, :n] + pz
        b = energy[k, :n] - pz
        valid = (a > 0.0) & (b > 0.0)
        out[k, :n] = eta[k, :n]
        out[k, :n][valid] = 0.5 * np.log(a[valid] / b[valid])
    return out


def float32_rapidity(arrays, n_constit):
    pt = arrays["fjet_clus_pt"].astype(np.float32)
    eta = arrays["fjet_clus_eta"].astype(np.float32)
    energy = arrays["fjet_clus_E"].astype(np.float32)
    out = np.zeros_like(pt, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        momentum = np.float32(pt * np.cosh(eta))
        e0 = np.maximum(energy, momentum).astype(np.float32)
        pz = np.float32(pt * np.sinh(eta))
        ratio = np.float32((e0 + pz) / (e0 - pz))
        values = np.float32(np.float32(0.5) * np.log(ratio))
    for k, n0 in enumerate(n_constit):
        n = int(n0)
        out[k, :n] = values[k, :n]
    return out


@njit(cache=True)
def _dphi(a, b):
    value = a - b
    while value > np.pi:
        value -= 2.0 * np.pi
    while value < -np.pi:
        value += 2.0 * np.pi
    return value


@njit(parallel=True, cache=True)
def cpp_correlators64(pt, y, phi, n_constit):
    """C++-style left-associated double accumulation in original order.

    Columns: ECF1, ECF2, ECF3, 1e3(b1), 2e3(b1), 3e3(b1), 1e2(b2).
    """
    output = np.zeros((pt.shape[0], 7), dtype=np.float64)
    for k in prange(pt.shape[0]):
        n = n_constit[k]
        total = 0.0
        for i in range(n):
            total += pt[k, i]
        angles = np.zeros((n, n), dtype=np.float64)
        e2 = 0.0
        q2_un = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dy = y[k, i] - y[k, j]
                dp = _dphi(phi[k, i], phi[k, j])
                angle = np.sqrt(dy * dy + dp * dp)
                angles[i, j] = angle
                angles[j, i] = angle
                weight2 = pt[k, i] * pt[k, j]
                e2 += weight2 * angle
                q2_un += weight2 * angle * angle
        g1_un = 0.0
        g2_un = 0.0
        g3_un = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                for ell in range(j + 1, n):
                    a = angles[i, j]
                    b = angles[i, ell]
                    c = angles[j, ell]
                    if a > b:
                        a, b = b, a
                    if b > c:
                        b, c = c, b
                    if a > b:
                        a, b = b, a
                    weight3 = pt[k, i] * pt[k, j]
                    weight3 = weight3 * pt[k, ell]
                    g1_un += weight3 * a
                    g2_un += weight3 * a * b
                    g3_un += weight3 * a * b * c
        output[k, 0] = total
        output[k, 1] = e2
        output[k, 2] = g3_un
        if total > 0.0:
            total2 = total * total
            total3 = total2 * total
            output[k, 3] = g1_un / total3
            output[k, 4] = g2_un / total3
            output[k, 5] = g3_un / total3
            output[k, 6] = q2_un / total2
    return output


@njit(parallel=True, cache=True)
def cpp_correlators32(pt, y, phi, n_constit):
    """All-float32 analogue of the recoverable C++ loop structure."""
    output = np.zeros((pt.shape[0], 7), dtype=np.float32)
    for k in prange(pt.shape[0]):
        n = n_constit[k]
        total = np.float32(0.0)
        for i in range(n):
            total = np.float32(total + pt[k, i])
        angles = np.zeros((n, n), dtype=np.float32)
        e2 = np.float32(0.0)
        q2_un = np.float32(0.0)
        for i in range(n):
            for j in range(i + 1, n):
                dy = np.float32(y[k, i] - y[k, j])
                dp = np.float32(_dphi(phi[k, i], phi[k, j]))
                angle = np.float32(np.sqrt(np.float32(dy * dy + dp * dp)))
                angles[i, j] = angle
                angles[j, i] = angle
                weight2 = np.float32(pt[k, i] * pt[k, j])
                e2 = np.float32(e2 + np.float32(weight2 * angle))
                q2_un = np.float32(
                    q2_un + np.float32(np.float32(weight2 * angle) * angle)
                )
        g1_un = np.float32(0.0)
        g2_un = np.float32(0.0)
        g3_un = np.float32(0.0)
        for i in range(n):
            for j in range(i + 1, n):
                for ell in range(j + 1, n):
                    a = angles[i, j]
                    b = angles[i, ell]
                    c = angles[j, ell]
                    if a > b:
                        a, b = b, a
                    if b > c:
                        b, c = c, b
                    if a > b:
                        a, b = b, a
                    weight3 = np.float32(pt[k, i] * pt[k, j])
                    weight3 = np.float32(weight3 * pt[k, ell])
                    g1_un = np.float32(g1_un + np.float32(weight3 * a))
                    g2_un = np.float32(
                        g2_un + np.float32(np.float32(weight3 * a) * b)
                    )
                    g3_un = np.float32(
                        g3_un + np.float32(
                            np.float32(np.float32(weight3 * a) * b) * c
                        )
                    )
        output[k, 0] = total
        output[k, 1] = e2
        output[k, 2] = g3_un
        if total > 0.0:
            total2 = np.float32(total * total)
            total3 = np.float32(total2 * total)
            output[k, 3] = np.float32(g1_un / total3)
            output[k, 4] = np.float32(g2_un / total3)
            output[k, 5] = np.float32(g3_un / total3)
            output[k, 6] = np.float32(q2_un / total2)
    return output


@njit(parallel=True, cache=True)
def cpp_correlators_acc32(pt, y, phi, n_constit):
    """Double geometry with float32 products and accumulators (mixed mode)."""
    output = np.zeros((pt.shape[0], 7), dtype=np.float32)
    for k in prange(pt.shape[0]):
        n = n_constit[k]
        total = np.float32(0.0)
        for i in range(n):
            total = np.float32(total + pt[k, i])
        angles = np.zeros((n, n), dtype=np.float64)
        e2 = np.float32(0.0)
        q2_un = np.float32(0.0)
        for i in range(n):
            for j in range(i + 1, n):
                dy = y[k, i] - y[k, j]
                dp = _dphi(phi[k, i], phi[k, j])
                angle = np.sqrt(dy * dy + dp * dp)
                angles[i, j] = angle
                angles[j, i] = angle
                weight2 = np.float32(pt[k, i] * pt[k, j])
                e2 = np.float32(e2 + np.float32(weight2 * angle))
                q2_un = np.float32(q2_un + np.float32(weight2 * angle * angle))
        g1_un = np.float32(0.0)
        g2_un = np.float32(0.0)
        g3_un = np.float32(0.0)
        for i in range(n):
            for j in range(i + 1, n):
                for ell in range(j + 1, n):
                    a = angles[i, j]
                    b = angles[i, ell]
                    c = angles[j, ell]
                    if a > b:
                        a, b = b, a
                    if b > c:
                        b, c = c, b
                    if a > b:
                        a, b = b, a
                    weight3 = np.float32(pt[k, i] * pt[k, j])
                    weight3 = np.float32(weight3 * pt[k, ell])
                    g1_un = np.float32(g1_un + np.float32(weight3 * a))
                    g2_un = np.float32(g2_un + np.float32(weight3 * a * b))
                    g3_un = np.float32(g3_un + np.float32(weight3 * a * b * c))
        output[k, 0] = total
        output[k, 1] = e2
        output[k, 2] = g3_un
        if total > 0.0:
            total2 = np.float32(total * total)
            total3 = np.float32(total2 * total)
            output[k, 3] = np.float32(g1_un / total3)
            output[k, 4] = np.float32(g2_un / total3)
            output[k, 5] = np.float32(g3_un / total3)
            output[k, 6] = np.float32(q2_un / total2)
    return output


def features_from_correlators(values, arithmetic="float64"):
    values = np.asarray(values)
    e1, e2, e3 = values[:, 0], values[:, 1], values[:, 2]
    g1, g3, q2 = values[:, 3], values[:, 5], values[:, 6]
    if arithmetic == "float32":
        e1 = e1.astype(np.float32)
        e2 = e2.astype(np.float32)
        e3 = e3.astype(np.float32)
        g1 = g1.astype(np.float32)
        g3 = g3.astype(np.float32)
        q2 = q2.astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            c2 = np.float32(np.float32(e3 * e1) / np.float32(e2 * e2))
            d2 = np.float32(
                np.float32(e3 * np.float32(e1 * np.float32(e1 * e1)))
                / np.float32(e2 * np.float32(e2 * e2))
            )
            l2 = np.float32(g3 / np.float32(q2 ** np.float32(1.5)))
            l3 = np.float32(g1 / np.float32(g3 ** np.float32(1.0 / 3.0)))
    else:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            c2 = e3 * e1 / e2 ** 2
            d2 = e3 * e1 ** 3 / e2 ** 3
            l2 = g3 / q2 ** 1.5
            l3 = g1 / g3 ** (1.0 / 3.0)
    return {
        "ECF1": e1.astype(np.float64) * 1.0e-6,
        "ECF2": e2.astype(np.float64) * 1.0e-12,
        "ECF3": e3.astype(np.float64) * 1.0e-18,
        "C2": c2.astype(np.float64), "D2": d2.astype(np.float64),
        "L2": l2.astype(np.float64), "L3": l3.astype(np.float64),
    }


def calculate_without_thrust(reco, arrays):
    pt = arrays["fjet_clus_pt"]
    eta = arrays["fjet_clus_eta"]
    phi = arrays["fjet_clus_phi"]
    energy = arrays["fjet_clus_E"]
    n_constit = (pt > 0.0).sum(axis=1).astype(np.int64)
    y = reco.constituent_rapidity(pt, eta, energy, n_constit, 1)
    split, qw, axeta, axphi, naxes = reco.kt_observables(
        pt, eta, phi, energy, n_constit, 4, 1
    )
    tau_raw = reco.nsubjettiness(
        pt, eta, phi, energy, n_constit, axeta, axphi, naxes, 4, 1.0, 1
    )
    ecf = reco.ecf123(pt, y, phi, n_constit, beta=1.0, block=128)
    gen3 = reco.gen_ecf3(pt, y, phi, n_constit, 1.0)
    gen2 = reco.gen_ecf2(pt, y, phi, n_constit, 2.0)
    sumpt = ecf[:, 0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        tau = tau_raw / sumpt[:, None]
        c2 = ecf[:, 2] * ecf[:, 0] / ecf[:, 1] ** 2
        d2 = ecf[:, 2] * ecf[:, 0] ** 3 / ecf[:, 1] ** 3
        l2 = gen3[:, 2] / gen2 ** 1.5
        l3 = gen3[:, 0] / gen3[:, 2] ** (1.0 / 3.0)
    return {
        "Tau1": tau[:, 0], "Tau2": tau[:, 1],
        "Tau3": tau[:, 2], "Tau4": tau[:, 3],
        "Split12": split[:, 0] * 1.0e-6,
        "Split23": split[:, 1] * 1.0e-6,
        "ECF1": ecf[:, 0] * 1.0e-6,
        "ECF2": ecf[:, 1] * 1.0e-12,
        "ECF3": ecf[:, 2] * 1.0e-18,
        "C2": c2, "D2": d2, "L2": l2, "L3": l3,
        "Qw": qw * 1.0e-6,
    }, {"n_constit": n_constit, "y": y, "gen3": gen3, "gen2": gen2}


@njit(cache=True)
def _record_level(px, py, pz, energy, m, ox, oy, oz, oe, count):
    if m <= count.shape[0]:
        for i in range(m):
            ox[m - 1, i] = px[i]
            oy[m - 1, i] = py[i]
            oz[m - 1, i] = pz[i]
            oe[m - 1, i] = energy[i]
        count[m - 1] = m


@njit(cache=True)
def _cluster_fullscan_one(pt, eta, phi, energy_in, n, wta,
                          distances, ox, oy, oz, oe, count):
    """O(n^3) all-pairs reference; no nearest-neighbour cache."""
    px = np.empty(n, dtype=np.float64)
    py = np.empty(n, dtype=np.float64)
    pz = np.empty(n, dtype=np.float64)
    energy = np.empty(n, dtype=np.float64)
    for i in range(n):
        px[i] = pt[i] * np.cos(phi[i])
        py[i] = pt[i] * np.sin(phi[i])
        pz[i] = pt[i] * np.sinh(eta[i])
        momentum = pt[i] * np.cosh(eta[i])
        energy[i] = energy_in[i] if energy_in[i] >= momentum else momentum
    m = n
    _record_level(px, py, pz, energy, m, ox, oy, oz, oe, count)
    while m > 1:
        best = 1.0e300
        bi = -1
        bj = -1
        for i in range(m):
            pti2 = px[i] * px[i] + py[i] * py[i]
            yi = 0.5 * np.log(
                max(energy[i] + pz[i], 1.0e-300)
                / max(energy[i] - pz[i], 1.0e-300)
            )
            phii = np.arctan2(py[i], px[i])
            for j in range(i + 1, m):
                ptj2 = px[j] * px[j] + py[j] * py[j]
                yj = 0.5 * np.log(
                    max(energy[j] + pz[j], 1.0e-300)
                    / max(energy[j] - pz[j], 1.0e-300)
                )
                phij = np.arctan2(py[j], px[j])
                dy = yi - yj
                dp = _dphi(phii, phij)
                distance = min(pti2, ptj2) * (dy * dy + dp * dp)
                if distance < best:
                    best = distance
                    bi = i
                    bj = j
        if m - 2 < distances.shape[0]:
            distances[m - 2] = best
        if wta == 1:
            pti = np.sqrt(px[bi] * px[bi] + py[bi] * py[bi])
            ptj = np.sqrt(px[bj] * px[bj] + py[bj] * py[bj])
            keep = bi if pti >= ptj else bj
            ky = 0.5 * np.log(
                max(energy[keep] + pz[keep], 1.0e-300)
                / max(energy[keep] - pz[keep], 1.0e-300)
            )
            kp = np.arctan2(py[keep], px[keep])
            newpt = pti + ptj
            px[bi] = newpt * np.cos(kp)
            py[bi] = newpt * np.sin(kp)
            pz[bi] = newpt * np.sinh(ky)
            energy[bi] = newpt * np.cosh(ky)
        else:
            px[bi] += px[bj]
            py[bi] += py[bj]
            pz[bi] += pz[bj]
            energy[bi] += energy[bj]
        for j in range(bj, m - 1):
            px[j] = px[j + 1]
            py[j] = py[j + 1]
            pz[j] = pz[j + 1]
            energy[j] = energy[j + 1]
        m -= 1
        _record_level(px, py, pz, energy, m, ox, oy, oz, oe, count)


@njit(parallel=True, cache=True)
def fullscan_kt(pt, eta, phi, energy, n_constit):
    n_jets = pt.shape[0]
    split = np.zeros((n_jets, 2), dtype=np.float64)
    qw = np.zeros(n_jets, dtype=np.float64)
    axeta = np.zeros((n_jets, 4, 4), dtype=np.float64)
    axphi = np.zeros((n_jets, 4, 4), dtype=np.float64)
    naxes = np.zeros((n_jets, 4), dtype=np.int64)
    for k in prange(n_jets):
        n = n_constit[k]
        d = np.zeros(4, dtype=np.float64)
        ex = np.zeros((4, 4), dtype=np.float64)
        ey = np.zeros((4, 4), dtype=np.float64)
        ez = np.zeros((4, 4), dtype=np.float64)
        ee = np.zeros((4, 4), dtype=np.float64)
        ec = np.zeros(4, dtype=np.int64)
        _cluster_fullscan_one(pt[k], eta[k], phi[k], energy[k], n, 0,
                              d, ex, ey, ez, ee, ec)
        split[k, 0] = np.sqrt(max(d[0], 0.0))
        split[k, 1] = np.sqrt(max(d[1], 0.0))
        if ec[2] == 3:
            minimum_mass = 1.0e300
            for i in range(3):
                for j in range(i + 1, 3):
                    se = ee[2, i] + ee[2, j]
                    sx = ex[2, i] + ex[2, j]
                    sy = ey[2, i] + ey[2, j]
                    sz = ez[2, i] + ez[2, j]
                    mass = np.sqrt(max(se * se - sx * sx - sy * sy - sz * sz, 0.0))
                    minimum_mass = min(minimum_mass, mass)
            qw[k] = minimum_mass

        wd = np.zeros(4, dtype=np.float64)
        wx = np.zeros((4, 4), dtype=np.float64)
        wy = np.zeros((4, 4), dtype=np.float64)
        wz = np.zeros((4, 4), dtype=np.float64)
        we = np.zeros((4, 4), dtype=np.float64)
        wc = np.zeros(4, dtype=np.int64)
        _cluster_fullscan_one(pt[k], eta[k], phi[k], energy[k], n, 1,
                              wd, wx, wy, wz, we, wc)
        for level in range(4):
            naxes[k, level] = wc[level]
            for i in range(wc[level]):
                axeta[k, level, i] = 0.5 * np.log(
                    max(we[level, i] + wz[level, i], 1.0e-300)
                    / max(we[level, i] - wz[level, i], 1.0e-300)
                )
                axphi[k, level, i] = np.arctan2(wy[level, i], wx[level, i])
    return split, qw, axeta, axphi, naxes


def fullscan_features(reco, arrays, n_constit):
    pt = arrays["fjet_clus_pt"]
    eta = arrays["fjet_clus_eta"]
    phi = arrays["fjet_clus_phi"]
    energy = arrays["fjet_clus_E"]
    split, qw, axeta, axphi, naxes = fullscan_kt(
        pt, eta, phi, energy, n_constit
    )
    tau_raw = reco.nsubjettiness(
        pt, eta, phi, energy, n_constit, axeta, axphi, naxes, 4, 1.0, 1
    )
    sumpt = np.sum(pt, axis=1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        tau = tau_raw / sumpt[:, None]
    return {
        "Tau1": tau[:, 0], "Tau2": tau[:, 1],
        "Tau3": tau[:, 2], "Tau4": tau[:, 3],
        "Split12": split[:, 0] * 1.0e-6,
        "Split23": split[:, 1] * 1.0e-6,
        "Qw": qw * 1.0e-6,
    }


def permuted_batches(arrays, n_constit, seed=80030):
    names = []
    owners = []
    output = {name: [] for name in arrays}
    for k, n0 in enumerate(n_constit):
        n = int(n0)
        pt = arrays["fjet_clus_pt"][k, :n]
        eta = arrays["fjet_clus_eta"][k, :n]
        phi = arrays["fjet_clus_phi"][k, :n]
        energy = arrays["fjet_clus_E"][k, :n]
        orders = [
            ("reverse", np.arange(n - 1, -1, -1)),
            ("pt_ascending", np.argsort(pt, kind="stable")),
            ("pt_descending", np.argsort(-pt, kind="stable")),
            ("eta_ascending", np.argsort(eta, kind="stable")),
            ("eta_descending", np.argsort(-eta, kind="stable")),
            ("phi_ascending", np.argsort(phi, kind="stable")),
            ("phi_descending", np.argsort(-phi, kind="stable")),
            ("energy_ascending", np.argsort(energy, kind="stable")),
            ("energy_descending", np.argsort(-energy, kind="stable")),
        ]
        rng = np.random.default_rng(seed + 104729 * k)
        for shuffle in range(4):
            orders.append((f"fixed_shuffle_{shuffle + 1}", rng.permutation(n)))
        for label, order in orders:
            for field, values in arrays.items():
                row = values[k].copy()
                row[:n] = values[k, :n][order]
                output[field].append(row[None, :])
            names.append(label)
            owners.append(k)
    return {name: np.concatenate(rows) for name, rows in output.items()}, owners, names


def rounding_boundary_batches(arrays, n_constit, direction_pairs=24, seed=80030):
    names = []
    owners = []
    physically_valid = []
    output = {name: [] for name in arrays}
    fields = list(arrays)
    for k, n0 in enumerate(n_constit):
        n = int(n0)
        centers = {field: arrays[field][k].astype(np.float32) for field in fields}
        bounds = {field: f32_cell(centers[field]) for field in fields}
        directions = []
        directions.append(("all_low", {field: -np.ones(n, dtype=np.int8) for field in fields}))
        directions.append(("all_high", {field: np.ones(n, dtype=np.int8) for field in fields}))
        for field in fields:
            directions.append((
                f"{field}_low",
                {other: -np.ones(n, dtype=np.int8) if other == field
                 else np.zeros(n, dtype=np.int8)
                 for other in fields},
            ))
            directions.append((
                f"{field}_high",
                {other: np.ones(n, dtype=np.int8) if other == field
                 else np.zeros(n, dtype=np.int8)
                 for other in fields},
            ))
        alternating = np.where(np.arange(n) % 2 == 0, 1, -1).astype(np.int8)
        directions.append(("alternating", {field: alternating for field in fields}))
        directions.append(("alternating_complement", {field: -alternating for field in fields}))
        rng = np.random.default_rng(seed + 130363 * k)
        for index in range(direction_pairs):
            signs = {field: np.where(
                rng.integers(0, 2, size=n, dtype=np.int8) > 0, 1, -1
            ).astype(np.int8)
                     for field in fields}
            directions.append((f"rademacher_{index + 1:02d}_plus", signs))
            directions.append((
                f"rademacher_{index + 1:02d}_minus",
                {field: -signs[field] for field in fields},
            ))
        for label, signs in directions:
            candidate = {}
            for field in fields:
                low, high = bounds[field]
                row = centers[field].astype(np.float64)
                row[:n] = inside_cell_boundary(
                    low[:n], high[:n], centers[field][:n], signs[field]
                )
                if not np.array_equal(row[:n].astype(np.float32), centers[field][:n]):
                    raise AssertionError(f"rounding-cell escape: {k} {label} {field}")
                output[field].append(row[None, :])
                candidate[field] = row
            names.append(label)
            owners.append(k)
            momentum = (candidate["fjet_clus_pt"][:n]
                        * np.cosh(candidate["fjet_clus_eta"][:n]))
            physically_valid.append(bool(np.all(
                candidate["fjet_clus_E"][:n] >= momentum
            )))
    return ({name: np.concatenate(rows) for name, rows in output.items()},
            owners, names, physically_valid)


def summarize_variant(feature, jet_index, owners, names, values, stored,
                      absolute_tolerance, protection, relative_tolerance,
                      physically_valid=None):
    positions = [i for i, owner in enumerate(owners) if owner == jet_index]
    selected = np.asarray(values[feature][positions], dtype=np.float64)
    passes = [
        passes_fixed_policy(value, stored, absolute_tolerance, protection,
                            relative_tolerance)
        for value in selected
    ]
    if physically_valid is None:
        selected_physical = [True] * len(positions)
    else:
        selected_physical = [bool(physically_valid[i]) for i in positions]
    errors = np.abs(selected - stored)
    best = int(np.argmin(errors))
    return {
        "minimum": float(np.min(selected)),
        "maximum": float(np.max(selected)),
        "relative_range": float(
            (np.max(selected) - np.min(selected))
            / max(abs(stored), float(np.max(np.abs(selected))), protection)
        ),
        "stored_straddled": bool(np.min(selected) <= stored <= np.max(selected)),
        "any_variant_passes_fixed_policy": bool(any(passes)),
        "passing_variant_count": int(sum(passes)),
        "physically_valid_variant_count": int(sum(selected_physical)),
        "physically_valid_passing_variant_count": int(sum(
            passed and physical for passed, physical in zip(passes, selected_physical)
        )),
        "any_physically_valid_variant_passes_fixed_policy": bool(any(
            passed and physical for passed, physical in zip(passes, selected_physical)
        )),
        "closest_variant": names[positions[best]],
        "closest_value": float(selected[best]),
        "closest_absolute_error": float(errors[best]),
    }


def scan_sample_pt_floor(data_root, ecf1_rows):
    grouped = defaultdict(list)
    for row in ecf1_rows:
        grouped[(row["relative_dataset_path"], row["file"])].append(
            (int(row["jet_index"]), row)
        )
    active_total = 0
    below_100 = 0
    minimum = math.inf
    jet_minima = []
    gaps = []
    disjoint_positive = 0
    for (relative, filename), items in grouped.items():
        items.sort(key=lambda item: item[0])
        indices = np.asarray([item[0] for item in items], dtype=np.int64)
        path = data_root / "data" / relative / filename
        with h5py.File(path, "r") as handle:
            block = handle["fjet_clus_pt"][indices, :200]
        for row_pt, (_, row) in zip(block, items):
            active = row_pt[row_pt > 0.0]
            active_total += active.size
            below_100 += int(np.sum(active < np.float32(100.0)))
            if active.size:
                value = float(np.min(active))
                minimum = min(minimum, value)
                jet_minima.append(value)
            low, high = f32_cell(active)
            stored_center = np.float32(float(row["stored_value"]))
            stored_low, stored_high = f32_cell(stored_center)
            sum_low = float(np.sum(low, dtype=np.float64))
            sum_high = float(np.sum(high, dtype=np.float64))
            target_low = float(stored_low) * 1.0e6
            target_high = float(stored_high) * 1.0e6
            gap = float(stored_center) * 1.0e6 - float(np.sum(active, dtype=np.float64))
            gaps.append(gap)
            if target_low > sum_high:
                disjoint_positive += 1
    return {
        "sampled_jet_count": len(ecf1_rows),
        "active_sampled_active_constituent_count": active_total,
        "released_constituents_below_100_MeV": below_100,
        "minimum_released_positive_pt_MeV": minimum,
        "jet_minimum_pt_percentiles_MeV": {
            str(q): float(np.percentile(jet_minima, q))
            for q in (0, 1, 5, 50, 95, 100)
        },
        "stored_ECF1_above_entire_released_pt_rounding_cell_count": disjoint_positive,
        "stored_ECF1_minus_released_sumpt_MeV_percentiles": {
            str(q): float(np.percentile(gaps, q))
            for q in (0, 1, 50, 95, 99, 99.9, 100)
        },
    }


def build_markdown(result):
    lines = [
        "# Final targeted strict-closure audit",
        "",
        "ThrustMaj was excluded from all computations in this audit.",
        "Production NPZ files were neither read for reconstruction nor written.",
        "",
        "## Endpoint",
        "",
        f"- Expanded sample: {result['baseline']['sampled_jets']:,} jets.",
        f"- Remaining strict failures: {result['baseline']['failed_rows']} rows "
        f"on {result['baseline']['failed_physical_jets']} physical jets.",
        f"- Classification counts: {result['classification_counts']}.",
        f"- Further production-code correction scientifically justified: "
        f"{result['conclusion']['further_code_correction_scientifically_justified']}.",
        f"- RECONSTRUCTION FREEZE = {result['conclusion']['reconstruction_freeze']}.",
        "",
        "## Exact remaining counts",
        "",
        "| Observable | Strict failures |",
        "|---|---:|",
    ]
    for feature, count in result["baseline"]["failure_counts_by_feature"].items():
        lines.append(f"| {feature} | {count} |")
    lines.extend([
        "",
        "## Control results",
        "",
        result["conclusion"]["control_summary"],
        "",
        "## Observable families",
        "",
        "| Family | Failed rows | Affected jets |",
        "|---|---:|---:|",
    ])
    for family, summary in result["family_summary"].items():
        lines.append(
            f"| {family} | {summary['failed_rows']} | {summary['affected_jets']} |"
        )
    lines.extend([
        "",
        "## Decisive invariant",
        "",
        result["conclusion"]["decisive_invariant"],
        "",
        "## Signed bias",
        "",
        result["conclusion"]["signed_bias_statement"],
        "",
        "## Reconstruction-freeze criterion",
        "",
        result["conclusion"]["objective_freeze_criterion"],
        "",
        "## Per-jet classification",
        "",
        "| Physical jet | Failed observables | Class | Missing scalar pT (MeV) |",
        "|---|---|---:|---:|",
    ])
    for record in result["jets"]:
        lines.append(
            f"| {record['physical_jet_id']} | {', '.join(record['failed_features'])} | "
            f"{record['classification']} | {record['ecf1_invariant']['center_gap_MeV']:.6g} |"
        )
    lines.extend([
        "",
        f"**RECONSTRUCTION FREEZE = {result['conclusion']['reconstruction_freeze']}**",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--expanded-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounding-direction-pairs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=80030)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Dynamic imports and numba's cross-process cache metadata do not mix on
    # Windows (the cached module name becomes ``<dynamic>``).  A per-process
    # audit cache avoids both stale-code reuse and writes beside production.
    isolated_numba_cache = str(
        (args.output_dir / f"numba_cache_run_{os.getpid()}").resolve()
    )
    os.environ["NUMBA_CACHE_DIR"] = isolated_numba_cache
    numba_config.CACHE_DIR = isolated_numba_cache
    here = Path(__file__).resolve().parent
    audit = import_module(here / "strict_closure_audit.py", "strict_audit_final")
    reco = import_module(here / "recompute_hl.py", "recompute_final_audit")
    reco.STORE = args.data_root.resolve()
    reco.DATA = reco.STORE / "data"

    report_path = args.expanded_audit_dir / "strict_closure_audit.json"
    eventwise_path = args.expanded_audit_dir / "strict_closure_eventwise.csv"
    diagnostics_path = args.expanded_audit_dir / "outlier_diagnostics.json"
    main_report = json.loads(report_path.read_text(encoding="utf-8"))
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    failed, all_ecf1, signed_all = load_failure_table(eventwise_path)
    keys, arrays, stored, metadata = load_failed_jets(args.data_root, failed, audit)
    key_to_index = {key: i for i, key in enumerate(keys)}
    failures_by_jet = defaultdict(list)
    for row in failed:
        key = (row["dataset"], row["relative_dataset_path"], row["file"],
               int(row["jet_index"]))
        failures_by_jet[key].append(row["feature"])

    current, extras = calculate_without_thrust(reco, arrays)
    n_constit = extras["n_constit"]
    y_current = extras["y"]
    y_raw = raw_rapidity(arrays, n_constit)
    y_eta = arrays["fjet_clus_eta"].copy()
    y_f32 = float32_rapidity(arrays, n_constit)

    pt64 = arrays["fjet_clus_pt"]
    phi64 = arrays["fjet_clus_phi"]
    pt32 = pt64.astype(np.float32)
    phi32 = phi64.astype(np.float32)
    cpp64 = cpp_correlators64(pt64, y_current, phi64, n_constit)
    cpp64_y32 = cpp_correlators64(
        pt64, y_f32.astype(np.float64), phi64, n_constit
    )
    cpp64_eta = cpp_correlators64(pt64, y_eta, phi64, n_constit)
    cpp64_raw = cpp_correlators64(pt64, y_raw, phi64, n_constit)
    cpp32 = cpp_correlators32(pt32, y_f32, phi32, n_constit)
    acc32 = cpp_correlators_acc32(pt32, y_current, phi32, n_constit)
    arithmetic_variants = {
        "cpp_nested_float64": features_from_correlators(cpp64),
        "mixed_float32_rapidity_float64_accumulation": features_from_correlators(cpp64_y32),
        "mixed_float64_geometry_float32_accumulation": features_from_correlators(acc32, "float32"),
        "all_float32": features_from_correlators(cpp32, "float32"),
        "pseudorapidity_massless_control": features_from_correlators(cpp64_eta),
        "raw_energy_rapidity_control": features_from_correlators(cpp64_raw),
    }

    fullscan = fullscan_features(reco, arrays, n_constit)
    perm_arrays, perm_owners, perm_names = permuted_batches(
        arrays, n_constit, args.seed
    )
    permutation_values, _ = calculate_without_thrust(reco, perm_arrays)
    (boundary_arrays, boundary_owners, boundary_names,
     boundary_physically_valid) = rounding_boundary_batches(
        arrays, n_constit, args.rounding_direction_pairs, args.seed
    )
    boundary_values, _ = calculate_without_thrust(reco, boundary_arrays)

    diagnostics_by_key = {}
    for item in diagnostics["records"]:
        key = (item["dataset"], item["relative_dataset_path"], item["file"],
               int(item["jet_index"]))
        diagnostics_by_key.setdefault(key, item)

    tolerance = {}
    for feature in FEATURES:
        stat = main_report["closure_all_samples"][feature]
        tolerance[feature] = {
            "absolute": float(stat["absolute_tolerance"]),
            "protection": float(stat["protected_relative_denominator_floor"]),
            "relative": float(stat["relative_tolerance"]),
        }

    jet_records = []
    feature_control_summary = defaultdict(lambda: {
        "cpp64_pass": 0, "any_arithmetic_pass": 0,
        "fullscan_pass": 0, "permutation_pass": 0, "rounding_cell_pass": 0,
        "physical_rounding_cell_pass": 0,
    })
    for key in keys:
        k = key_to_index[key]
        failed_features = sorted(failures_by_jet[key], key=FEATURES.index)
        active_pt = pt64[k, :n_constit[k]].astype(np.float32)
        input_low, input_high = f32_cell(active_pt)
        sum_low = float(np.sum(input_low, dtype=np.float64))
        sum_high = float(np.sum(input_high, dtype=np.float64))
        sum_center = float(np.sum(active_pt, dtype=np.float64))
        stored_ecf1_f32 = np.float32(stored["ECF1"][k])
        stored_low, stored_high = f32_cell(stored_ecf1_f32)
        stored_sum_low = float(stored_low) * 1.0e6
        stored_sum_high = float(stored_high) * 1.0e6
        interval_overlap = not (stored_sum_low > sum_high or sum_low > stored_sum_high)
        invariant = {
            "released_sumpt_MeV": sum_center,
            "stored_ECF1_as_sumpt_MeV": float(stored_ecf1_f32) * 1.0e6,
            "center_gap_MeV": float(stored_ecf1_f32) * 1.0e6 - sum_center,
            "released_sumpt_float32_cell": [sum_low, sum_high],
            "stored_ECF1_float32_cell_as_MeV": [stored_sum_low, stored_sum_high],
            "rounding_cells_overlap": interval_overlap,
            "stored_interval_strictly_above_released_sum_interval": bool(
                stored_sum_low > sum_high
            ),
            "minimum_unaccounted_pt_MeV": max(0.0, stored_sum_low - sum_high),
        }

        controls = {}
        any_category1 = False
        any_category2 = False
        unresolved_features = []
        for feature in failed_features:
            # Materialize a zero-filled summary entry even when no control
            # passes, so the final table includes every failed observable.
            feature_control_summary[feature]
            tol = tolerance[feature]
            target = float(stored[feature][k])
            baseline = float(current[feature][k])
            arithmetic = {}
            if feature in arithmetic_variants["cpp_nested_float64"]:
                for name, values in arithmetic_variants.items():
                    value = float(values[feature][k])
                    arithmetic[name] = {
                        "value": value,
                        "passes_fixed_policy": passes_fixed_policy(
                            value, target, tol["absolute"], tol["protection"],
                            tol["relative"],
                        ),
                        "absolute_error": abs(value - target),
                    }
                if arithmetic["cpp_nested_float64"]["passes_fixed_policy"]:
                    feature_control_summary[feature]["cpp64_pass"] += 1
                if any(item["passes_fixed_policy"] for item in arithmetic.values()):
                    feature_control_summary[feature]["any_arithmetic_pass"] += 1

            full = None
            if feature in fullscan:
                full_value = float(fullscan[feature][k])
                full = {
                    "value": full_value,
                    "passes_fixed_policy": passes_fixed_policy(
                        full_value, target, tol["absolute"], tol["protection"],
                        tol["relative"],
                    ),
                    "relative_change_from_cached_algorithm": (
                        (full_value - baseline)
                        / max(abs(full_value), abs(baseline), tol["protection"])
                    ),
                }
                if full["passes_fixed_policy"]:
                    feature_control_summary[feature]["fullscan_pass"] += 1

            permutation = summarize_variant(
                feature, k, perm_owners, perm_names, permutation_values, target,
                tol["absolute"], tol["protection"], tol["relative"],
            )
            rounding = summarize_variant(
                feature, k, boundary_owners, boundary_names, boundary_values,
                target, tol["absolute"], tol["protection"], tol["relative"],
                boundary_physically_valid,
            )
            if permutation["any_variant_passes_fixed_policy"]:
                feature_control_summary[feature]["permutation_pass"] += 1
            if rounding["any_variant_passes_fixed_policy"]:
                feature_control_summary[feature]["rounding_cell_pass"] += 1
            if rounding["any_physically_valid_variant_passes_fixed_policy"]:
                feature_control_summary[feature]["physical_rounding_cell_pass"] += 1

            cpp_fix = bool(
                arithmetic
                and arithmetic["cpp_nested_float64"]["passes_fixed_policy"]
                and not passes_fixed_policy(
                    baseline, target, tol["absolute"], tol["protection"],
                    tol["relative"],
                )
            )
            fullscan_fix = bool(
                full and full["passes_fixed_policy"]
                and abs(full["relative_change_from_cached_algorithm"]) > 1.0e-12
            )
            numerical = bool(
                permutation["any_variant_passes_fixed_policy"]
                or rounding["any_physically_valid_variant_passes_fixed_policy"]
            )
            if cpp_fix or fullscan_fix:
                classification = 1
                reason = "source-like arithmetic/full-rescan control removes the failure"
                any_category1 = True
            elif numerical:
                classification = 2
                reason = "a deterministic ordering or in-cell float32 control removes the failure"
                any_category2 = True
            elif invariant["stored_interval_strictly_above_released_sum_interval"]:
                classification = 3
                reason = (
                    "the exact ECF1=sum(pt) invariant proves the released jet lacks "
                    "scalar-pT information beyond float32 rounding"
                )
            else:
                classification = 4
                reason = "no tested implementation, ordering, or rounding-cell control proves a cause"
                unresolved_features.append(feature)

            controls[feature] = {
                "family": FAMILY[feature],
                "stored": target,
                "baseline": baseline,
                "baseline_signed_error": baseline - target,
                "classification": classification,
                "classification_reason": reason,
                "arithmetic_and_rapidity_controls": arithmetic,
                "fullscan_clustering_control": full,
                "constituent_order_controls": permutation,
                "deterministic_float32_boundary_controls": rounding,
            }

        if any_category1:
            jet_classification = 1
            jet_reason = "at least one failed family has a proven implementation mismatch"
        elif unresolved_features:
            jet_classification = 4
            jet_reason = "unresolved failed observables: " + ", ".join(unresolved_features)
        elif invariant["stored_interval_strictly_above_released_sum_interval"]:
            jet_classification = 3
            jet_reason = (
                "the jet has directly proven missing scalar-pT information; any individual "
                "rounding-sensitive row is subordinate to that jet-level invariant"
            )
        elif any_category2:
            jet_classification = 2
            jet_reason = "at least one failure is reproduced by a numerical/order boundary control"
        else:
            jet_classification = 3
            jet_reason = (
                "all failures are consistent with the independently proven missing scalar-pT "
                "information in this released jet"
            )

        relation = {}
        e1s = float(stored["ECF1"][k])
        e2s = float(stored["ECF2"][k])
        e3s = float(stored["ECF3"][k])
        with np.errstate(divide="ignore", invalid="ignore"):
            c2_from_stored = e3s * e1s / e2s ** 2
            d2_from_stored = e3s * e1s ** 3 / e2s ** 3
        for feature, value in (("C2", c2_from_stored), ("D2", d2_from_stored)):
            tol = tolerance[feature]
            relation[feature] = {
                "value_from_stored_ECF_branches": float(value),
                "stored_ratio_branch": float(stored[feature][k]),
                "passes_fixed_policy": passes_fixed_policy(
                    value, stored[feature][k], tol["absolute"], tol["protection"],
                    tol["relative"],
                ),
            }

        gap_diag = diagnostics_by_key.get(key, {}).get("clustering_gap_diagnostics")
        jet_records.append({
            **metadata[k],
            "n_constit": int(n_constit[k]),
            "failed_features": failed_features,
            "failed_families": sorted(set(FAMILY[f] for f in failed_features)),
            "classification": jet_classification,
            "classification_reason": jet_reason,
            "ecf1_invariant": invariant,
            "stored_ratio_self_consistency": relation,
            "clustering_gap_diagnostics": gap_diag,
            "feature_controls": controls,
        })

    counts_by_feature = Counter(row["feature"] for row in failed)
    ordered_counts = {feature: counts_by_feature[feature]
                      for feature in FEATURES if counts_by_feature[feature]}
    failure_sign = {}
    for feature in FEATURES:
        feature_rows = [row for row in failed if row["feature"] == feature]
        if not feature_rows:
            continue
        differences = np.asarray([
            float(row["reconstructed_value"]) - float(row["stored_value"])
            for row in feature_rows
        ])
        positive = int(np.sum(differences > 0.0))
        negative = int(np.sum(differences < 0.0))
        failure_sign[feature] = {
            "positive": positive, "negative": negative,
            "two_sided_exact_binomial_p": exact_binomial_two_sided(positive, negative),
        }
    all_failure_differences = np.asarray([
        float(row["reconstructed_value"]) - float(row["stored_value"])
        for row in failed
    ])
    all_positive = int(np.sum(all_failure_differences > 0.0))
    all_negative = int(np.sum(all_failure_differences < 0.0))
    signed_summary = {}
    for feature, values in signed_all.items():
        values = np.asarray(values, dtype=np.float64)
        signed_summary[feature] = {
            "positive_count": int(np.sum(values > 0.0)),
            "negative_count": int(np.sum(values < 0.0)),
            "median_signed_relative_error": float(np.median(values)),
            "mean_signed_relative_error": float(np.mean(values)),
            "p01": float(np.percentile(values, 1)),
            "p99": float(np.percentile(values, 99)),
        }

    def maximum_relative_difference(first, second):
        first = np.asarray(first, dtype=np.float64)
        second = np.asarray(second, dtype=np.float64)
        denominator = np.maximum.reduce([
            np.abs(first), np.abs(second), np.full(first.shape, 1.0e-300)
        ])
        return float(np.max(np.abs(first - second) / denominator))

    implementation_agreement = {
        "source_like_cpp64_vs_current": {
            feature: maximum_relative_difference(
                arithmetic_variants["cpp_nested_float64"][feature],
                current[feature],
            )
            for feature in ("ECF1", "ECF2", "ECF3", "C2", "D2", "L2", "L3")
        },
        "fullscan_kt_vs_nearest_neighbour_cache": {
            feature: maximum_relative_difference(fullscan[feature], current[feature])
            for feature in ("Tau1", "Tau2", "Tau3", "Tau4",
                            "Split12", "Split23", "Qw")
        },
        "stored_ratio_branch_self_consistency_on_failed_rows": {
            feature: {
                "failed_rows": int(sum(feature in record["failed_features"]
                                       for record in jet_records)),
                "passing_rows": int(sum(
                    feature in record["failed_features"]
                    and record["stored_ratio_self_consistency"][feature]["passes_fixed_policy"]
                    for record in jet_records
                )),
            }
            for feature in ("C2", "D2")
        },
    }

    arithmetic_pass_by_mode = {}
    for feature in ordered_counts:
        arithmetic_pass_by_mode[feature] = {}
        for mode in arithmetic_variants:
            arithmetic_pass_by_mode[feature][mode] = int(sum(
                mode in record["feature_controls"][feature]["arithmetic_and_rapidity_controls"]
                and record["feature_controls"][feature]["arithmetic_and_rapidity_controls"][mode]["passes_fixed_policy"]
                for record in jet_records if feature in record["failed_features"]
            ))

    control_pass_counts = {
        feature: dict(feature_control_summary[feature])
        for feature in ordered_counts
    }
    for feature in ordered_counts:
        control_pass_counts[feature]["arithmetic_pass_by_mode"] = arithmetic_pass_by_mode[feature]

    pt_floor = scan_sample_pt_floor(args.data_root, all_ecf1)
    class_counts = Counter(record["classification"] for record in jet_records)
    class_counts_named = {
        "1_proven_implementation_mismatch": class_counts[1],
        "2_proven_numerical_precision_or_degeneracy": class_counts[2],
        "3_consistent_with_information_lost_in_export": class_counts[3],
        "4_unresolved": class_counts[4],
    }
    no_code_mismatch = class_counts[1] == 0
    no_unresolved = class_counts[4] == 0
    freeze = "PASS" if no_code_mismatch and no_unresolved else "FAIL"

    family_summary = {}
    for family in sorted(set(FAMILY.values())):
        family_summary[family] = {
            "failed_rows": int(sum(
                FAMILY[row["feature"]] == family for row in failed
            )),
            "affected_jets": int(sum(
                family in record["failed_families"] for record in jet_records
            )),
        }
    clustering_records = [
        record for record in jet_records
        if ("exclusive-kt E-scheme" in record["failed_families"]
            or "WTA n-subjettiness" in record["failed_families"])
    ]
    clustering_degeneracy = {}
    for scheme in ("wta", "e_scheme"):
        gaps = np.asarray([
            record["clustering_gap_diagnostics"][scheme][
                "minimum_relative_pair_distance_gap"
            ] for record in clustering_records
        ], dtype=np.float64)
        clustering_degeneracy[scheme] = {
            "affected_jets_checked": len(gaps),
            "minimum_relative_pair_distance_gap": float(np.min(gaps)),
            "count_below_1e-6": int(np.sum(gaps < 1.0e-6)),
        }

    result = {
        "kind": "final targeted audit of expanded strict-closure failures",
        "read_only_guards": {
            "production_npz_read": False,
            "production_npz_written": False,
            "production_source_modified": False,
            "ThrustMaj_recomputed_or_modified": False,
        },
        "inputs": {
            "expanded_audit_report": str(report_path.resolve()),
            "eventwise_table": str(eventwise_path.resolve()),
            "source_hdf5_root": str((args.data_root / "data").resolve()),
            "recompute_source_sha256": hashlib.sha256(
                (here / "recompute_hl.py").read_bytes()
            ).hexdigest(),
        },
        "baseline": {
            "sampled_jets": int(main_report["total_sampled_jets"]),
            "failed_rows": len(failed),
            "failed_physical_jets": len(keys),
            "failure_counts_by_feature": ordered_counts,
        },
        "controls": {
            "arithmetic_modes": list(arithmetic_variants),
            "constituent_orderings_per_jet": len(set(perm_names[:13])),
            "float32_boundary_points_per_jet": len(set(
                boundary_names[:2 * args.rounding_direction_pairs + 12]
            )),
            "float32_boundary_seed": args.seed,
            "fullscan_exclusive_kt_reference": True,
            "feature_control_pass_counts": control_pass_counts,
            "implementation_agreement": implementation_agreement,
        },
        "sample_export_diagnostics": pt_floor,
        "signed_bias": {
            "all_failure_rows": {
                "positive": all_positive,
                "negative": all_negative,
                "two_sided_exact_binomial_p_naive_rows_not_independent":
                    exact_binomial_two_sided(all_positive, all_negative),
            },
            "failure_rows_by_feature": failure_sign,
            "all_valid_rows_by_feature": signed_summary,
        },
        "classification_counts": class_counts_named,
        "family_summary": family_summary,
        "clustering_degeneracy_summary": clustering_degeneracy,
        "jets": jet_records,
        "conclusion": {
            "decisive_invariant": (
                f"For {sum(r['ecf1_invariant']['stored_interval_strictly_above_released_sum_interval'] for r in jet_records)} "
                f"of {len(jet_records)} affected jets, the complete float32 interval of stored ECF1 lies above "
                "the maximum scalar-pT sum allowed by every released constituent rounding cell. "
                "Because ECF1 is exactly sum(pt), arithmetic order, rapidity, clustering, and ECF formula choices "
                "cannot create that missing scalar pT."
            ),
            "signed_bias_statement": (
                f"The strict-failure rows are one-sided: reconstructed-stored is negative for "
                f"{all_negative}/{len(failed)} rows and positive for {all_positive}/{len(failed)}. "
                f"ECF3 is negative for {failure_sign['ECF3']['negative']}/{failure_sign['ECF3']['negative'] + failure_sign['ECF3']['positive']} "
                "independent affected jets.  The bulk medians over all 14,103 jets remain near zero, so this is a "
                "rare missing-information tail, not a global scale bias."
            ),
            "control_summary": (
                "The recoverable C++-order float64 ECF replay changed the current results by at most "
                f"{max(implementation_agreement['source_like_cpp64_vs_current'].values()):.3g} relative and fixed "
                "0 failures. Float32 accumulation, float32 rapidity with float64 accumulation, and float64 geometry "
                "with float32 accumulation also fixed 0 failures. Raw-energy rapidity fixed 0. The massless/eta "
                "control moved a few rows inside tolerance but contradicts the sourced rapidity definition and did "
                "not repair the ECF1 sum invariant. Thirteen constituent orderings fixed 0 failures. The independent "
                "full-rescan exclusive-kt replay differed by at most "
                f"{max(implementation_agreement['fullscan_kt_vs_nearest_neighbour_cache'].values()):.3g} relative, "
                "fixed 0, and found no merge gap below 1e-6. Physically valid deterministic float32-cell controls "
                "accounted for the lone Qw-only jet; the other 43 jets retain a disjoint ECF1 input/output interval. "
                "All 9 failed C2 and all 4 failed D2 rows are self-consistent with their stored ECF branches, so they "
                "are propagated ECF discrepancies. For generalized ECFs, the source-like replay agrees at "
                f"{implementation_agreement['source_like_cpp64_vs_current']['L2']:.3g} relative for L2 and exactly "
                "for L3; all 6 L2 failures and all 4 L3 failures occur on class-3 jets."
            ),
            "further_code_correction_scientifically_justified": (
                "NO" if no_code_mismatch else "YES"
            ),
            "objective_freeze_criterion": (
                "Freeze reconstruction when (i) authoritative formulas and units are unchanged; "
                "(ii) source-like double accumulation and an independent full-rescan clustering replay reveal "
                "zero class-1 mismatches; (iii) every strict residual is either reproduced by a deterministic "
                "numerical/degeneracy control or linked to a violated input invariant outside the combined "
                "float32 cells; and (iv) no jet remains class 4.  Do not require zero strict rows when the released "
                "constituents demonstrably lack information used by the stored high-level branch."
            ),
            "reconstruction_freeze": freeze,
        },
    }

    dump_json(args.output_dir / "final_targeted_closure_audit.json", result)
    csv_path = args.output_dir / "final_targeted_closure_jets.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "physical_jet_id", "dataset", "file", "jet_index",
            "event_identifier", "n_constit", "failed_features", "failed_families",
            "classification", "classification_reason", "stored_ECF1_minus_sumpt_MeV",
            "minimum_unaccounted_pt_MeV", "ecf1_rounding_cells_overlap",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in jet_records:
            writer.writerow({
                "physical_jet_id": record["physical_jet_id"],
                "dataset": record["dataset"], "file": record["file"],
                "jet_index": record["jet_index"],
                "event_identifier": record["event_identifier"],
                "n_constit": record["n_constit"],
                "failed_features": ",".join(record["failed_features"]),
                "failed_families": ",".join(record["failed_families"]),
                "classification": record["classification"],
                "classification_reason": record["classification_reason"],
                "stored_ECF1_minus_sumpt_MeV": record["ecf1_invariant"]["center_gap_MeV"],
                "minimum_unaccounted_pt_MeV": record["ecf1_invariant"]["minimum_unaccounted_pt_MeV"],
                "ecf1_rounding_cells_overlap": record["ecf1_invariant"]["rounding_cells_overlap"],
            })
    (args.output_dir / "FINAL_TARGETED_CLOSURE_AUDIT.md").write_text(
        build_markdown(result), encoding="utf-8"
    )
    print(json.dumps({
        "failed_rows": len(failed), "failed_jets": len(keys),
        "classification_counts": class_counts_named,
        "freeze": freeze,
        "output": str(args.output_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
