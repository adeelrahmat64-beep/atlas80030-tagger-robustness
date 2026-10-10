"""ATLAS 80030 - the frozen uncertainty chain of atlas_metrics.py as a pure function.

WHY THIS FILE EXISTS
    atlas_metrics.main() computes every metric and every systematic uncertainty
    of the paper, but it also writes the frozen per-dataset metric files and the
    frozen uncertainty summary.  The robustness extensions (bootstrap,
    multi-seed and feature-set studies) need the same numbers for many inputs
    without ever touching those frozen files.  This module therefore reproduces
    the body of atlas_metrics.main() as a function that takes prediction arrays
    in memory and returns the results in memory.  It writes nothing.

HOW IT STAYS IDENTICAL TO THE FROZEN CHAIN
    Every metric is computed by the frozen functions imported from
    atlas_metrics.py itself (metrics_for, background_rejection, rel, env, quad,
    SPEC, FULL_BINS, SCALE_COLS).  Only the control flow of main() is restated
    here, line for line and in the same order.  check_against_frozen() then
    compares the output for the frozen predictions with the frozen metrics JSON
    written by atlas_metrics.py; every script that uses this module calls it
    before doing anything else and stops on any mismatch.

    light=True skips AUC and accuracy for datasets whose AUC/accuracy are not
    used by the uncertainty combination.  Background rejections are then
    computed by the same frozen background_rejection(), on the same pT
    selection, with the same plot-closing duplicate in the last binned entry,
    so they are identical to metrics_for()'s.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from atlas_metrics import (
    FULL_BINS,
    PRIMARY_WP,
    SCALE_COLS,
    SPEC,
    background_rejection,
    env,
    metrics_for,
    quad,
    rel,
)

DETECTOR = ["esup", "esdown", "cer", "cpos", "teg", "tej", "tfl", "tfj", "bias"]
MODELLING = ["ttbar_pythia", "ttbar_herwig", "cluster", "string", "angular", "dipole"]
ORDER = ["test_nominal"] + DETECTOR + MODELLING


def _br_only(labels, preds, weights, pt, wp, bins):
    """Inclusive and binned background rejection exactly as metrics_for()."""
    out = {}
    out["br"], _ = background_rejection(labels, preds, weights, wp)
    binned = np.full(len(bins), np.nan)
    for i in range(len(bins) - 1):
        sel = (pt > bins[i]) & (pt < bins[i + 1])
        counts = np.bincount(labels[sel].astype(int), minlength=2)
        if not sel.any() or (counts == 0).any():
            raise ValueError(
                f"pT bin ({bins[i]}, {bins[i + 1]}) has insufficient class counts "
                f"{counts.tolist()}"
            )
        binned[i], _ = background_rejection(labels[sel], preds[sel], weights[sel], wp)
    binned[-1] = binned[-2]
    out["binned_br"] = binned
    out["n"] = int(labels.size)
    if not (np.isfinite(out["br"]) and np.isfinite(binned).all()):
        raise ValueError("non-finite metric result")
    return out


def _dataset_inputs(ds, d, nom, w_ds=None, w_nom=None):
    """Labels, predictions, pT, weights and bins for one dataset.

    Exactly as atlas_metrics.main() when no weights are given: the weights are
    then np.ones over the (possibly nominal-extended) sample.  With bootstrap
    weights, the nominal jets appended to a one-class modelling sample carry
    their nominal bootstrap weights.
    """
    spec = SPEC.get(ds, {})
    lab, prd, pt = d["labels"], d["predictions"], d["pt"]
    w = None if w_ds is None else np.asarray(w_ds, dtype=np.float64)
    if "add_nominal" in spec:
        t = spec["add_nominal"]
        sel = nom["labels"] == t
        prd = np.concatenate((prd, nom["predictions"][sel]))
        pt = np.concatenate((pt, nom["pt"][sel]))
        lab = np.concatenate((lab, np.full(int(sel.sum()), t, dtype=lab.dtype)))
        if w is not None:
            w = np.concatenate((w, np.asarray(w_nom, dtype=np.float64)[sel]))
    if w is None:
        w = np.ones(lab.shape)                           # calc.py: unweighted
    if w.shape != lab.shape:
        raise ValueError(f"{ds}: weights do not align with the sample")
    bins = FULL_BINS
    if spec.get("extrapolate"):
        bins = FULL_BINS[:int(np.argwhere(FULL_BINS > 1500000)[0][0])]
    return lab, prd, pt, w, bins


def compute_chain(data, working_points=(0.5, 0.8), light=False, full_metric_sets=(),
                  weights=None):
    """Run the atlas_metrics.main() chain on in-memory predictions.

    data: {dataset: {"predictions", "labels", "pt"[, "shower_weights"]}} with
          all sixteen evaluation datasets present.
    working_points: working points for the per-dataset metrics.  PRIMARY_WP
          (0.5) must be included; the uncertainties are combined at 0.5 only,
          exactly as in atlas_metrics.main().
    light: compute only background rejections, except for the datasets named
          in full_metric_sets, which also get AUC and accuracy.
    weights: None for the frozen (unweighted) chain, or {dataset: per-jet
          bootstrap weights} for every dataset.  The shower-weight variations
          then use nominal weight x shower ratio for the varied class.
    Returns (M, summary).  M[(dataset, wp)] holds the metrics; summary holds
    every inclusive and binned component of the uncertainty combination.
    """
    if PRIMARY_WP not in working_points:
        raise ValueError("the primary working point 0.5 must be computed")
    missing = [ds for ds in ORDER if ds not in data]
    if missing:
        raise ValueError(f"missing datasets for the uncertainty chain: {missing}")
    nom = data["test_nominal"]
    if "shower_weights" not in nom:
        raise ValueError("test_nominal shower_weights are required")

    if weights is not None and [ds for ds in ORDER if ds not in weights]:
        raise ValueError("bootstrap weights must be given for every dataset")
    M = {}
    for wp in working_points:
        for ds in ORDER:
            lab, prd, pt, w, bins = _dataset_inputs(
                ds, data[ds], nom,
                None if weights is None else weights[ds],
                None if weights is None else weights["test_nominal"])
            if light and ds not in full_metric_sets:
                m = _br_only(lab, prd, w, pt, wp, bins)
            else:
                m = metrics_for(lab, prd, w, pt, wp, bins)
            m["pt_bins"] = bins
            M[(ds, wp)] = m

    # theory variations: shower-weight ratios applied to nominal predictions
    sw = np.asarray(nom["shower_weights"], dtype=np.float64)
    nw = sw[:, 0]
    lab = nom["labels"]
    base = (np.ones(lab.shape) if weights is None
            else np.asarray(weights["test_nominal"], dtype=np.float64))
    for target, tname in ((1, "sig"), (0, "bkg")):
        for vname, col in SCALE_COLS.items():
            w = base.copy()
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = sw[:, col] / nw
            if not np.isfinite(ratio).all():
                raise ValueError(f"non-finite shower-weight ratio in column {col}")
            # frozen chain: w = 1, then w[target] = ratio; 1.0 * ratio == ratio
            w[lab == target] = base[lab == target] * ratio[lab == target]
            key = f"{tname}_{vname}"
            if light:
                m = _br_only(lab, nom["predictions"], w, nom["pt"], PRIMARY_WP, FULL_BINS)
            else:
                m = metrics_for(lab, nom["predictions"], w, nom["pt"], PRIMARY_WP, FULL_BINS)
            m["pt_bins"] = FULL_BINS
            M[(key, PRIMARY_WP)] = m

    # uncertainty combination, transcribed from atlas_metrics.main()
    wp = PRIMARY_WP
    nb = M[("test_nominal", wp)]["binned_br"]
    ni = M[("test_nominal", wp)]["br"]
    comp, inc = {}, {}
    for ds in DETECTOR:
        comp[ds] = rel(M[(ds, wp)]["binned_br"], nb)
        inc[ds] = rel(M[(ds, wp)]["br"], ni)

    S = {"nominal_br": ni, "nominal_binned_br": nb}
    S["es"], S["inc_es"] = env(comp["esup"], comp["esdown"]), env(inc["esup"], inc["esdown"])
    S["cer"], S["inc_cer"] = comp["cer"], inc["cer"]
    S["cpos"], S["inc_cpos"] = comp["cpos"], inc["cpos"]
    S["cluster"] = quad(S["es"], comp["cer"], comp["cpos"])
    S["inc_cluster"] = quad(S["inc_es"], inc["cer"], inc["cpos"])
    S["eff"], S["inc_eff"] = env(comp["teg"], comp["tej"]), env(inc["teg"], inc["tej"])
    S["fake"], S["inc_fake"] = env(comp["tfl"], comp["tfj"]), env(inc["tfl"], inc["tfj"])
    S["bias"], S["inc_bias"] = comp["bias"], inc["bias"]
    S["track"] = quad(S["eff"], S["fake"], comp["bias"])
    S["inc_track"] = quad(S["inc_eff"], S["inc_fake"], inc["bias"])

    sig_model = rel(M[("ttbar_herwig", wp)]["binned_br"], M[("ttbar_pythia", wp)]["binned_br"])
    S["inc_sig_model"] = rel(M[("ttbar_herwig", wp)]["br"], M[("ttbar_pythia", wp)]["br"])
    bkg_ps = rel(M[("dipole", wp)]["binned_br"], M[("angular", wp)]["binned_br"])
    S["inc_bkg_ps"] = rel(M[("dipole", wp)]["br"], M[("angular", wp)]["br"])
    bkg_had = rel(M[("cluster", wp)]["binned_br"], M[("string", wp)]["binned_br"])
    S["inc_bkg_had"] = rel(M[("cluster", wp)]["br"], M[("string", wp)]["br"])
    ext = np.zeros_like(bkg_ps)
    mb = int(np.argwhere(FULL_BINS > 1500000)[0][0])
    ext[:mb] = sig_model[:mb]
    ext[mb:] = np.nanmax(sig_model[:mb])
    S["sig_model"] = ext
    S["bkg_ps"], S["bkg_had"] = bkg_ps, bkg_had
    S["bkg_model"] = quad(bkg_ps, bkg_had)
    S["inc_bkg_model"] = quad(S["inc_bkg_ps"], S["inc_bkg_had"])
    S["model"] = quad(ext, S["bkg_model"])
    S["inc_model"] = quad(S["inc_sig_model"], quad(S["inc_bkg_ps"], S["inc_bkg_had"]))

    scale_parts, inc_scale_parts = [], []
    for tname in ("sig", "bkg"):
        for a, b in (("ISRx2", "ISRxp5"), ("FSRx2", "FSRxp5")):
            ka, kb = f"{tname}_{a}", f"{tname}_{b}"
            ua = rel(M[(ka, wp)]["binned_br"], nb)
            ub = rel(M[(kb, wp)]["binned_br"], nb)
            ia = rel(M[(ka, wp)]["br"], ni)
            ib = rel(M[(kb, wp)]["br"], ni)
            part = f"{tname}_{a[:3]}"
            S[part], S[f"inc_{part}"] = env(ua, ub), env(ia, ib)
            scale_parts.append(S[part])
            inc_scale_parts.append(S[f"inc_{part}"])
    S["scale"] = quad(*scale_parts)
    S["inc_scale"] = quad(*inc_scale_parts)
    S["total"] = quad(S["cluster"], S["track"], S["model"], S["scale"])
    S["inc_total"] = quad(S["inc_cluster"], S["inc_track"], S["inc_model"], S["inc_scale"])
    return M, S


def check_against_frozen(M, S, frozen_json_path, *, tol=0.0):
    """Compare a chain result for the frozen predictions with the frozen JSON.

    Returns a list of human-readable mismatch descriptions; empty means every
    compared number agrees to within tol (relative).  tol=0.0 demands exact
    floating-point equality.
    """
    payload = json.loads(Path(frozen_json_path).read_text(encoding="utf-8"))
    problems = []

    def same(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.shape != b.shape:
            return False
        if tol == 0.0:
            return bool(np.array_equal(a, b))
        return bool(np.all(np.abs(a - b) <= tol * np.maximum(np.abs(b), 1e-300)))

    compared = 0
    for entry in payload["metrics"]:
        key = (entry["dataset"], entry["working_point"])
        if key not in M:
            continue
        m = M[key]
        for field in ("auc", "acc", "br", "binned_br"):
            if field not in m:
                continue
            compared += 1
            if not same(m[field], entry[field]):
                problems.append(f"{key} {field}: {m[field]!r} vs frozen {entry[field]!r}")
    frozen = payload["uncertainty_summary"]
    for field in ("nominal_br", "nominal_binned_br", "cluster", "inc_cluster",
                  "track", "inc_track", "scale", "inc_scale", "model",
                  "inc_model", "total", "inc_total"):
        compared += 1
        if not same(S[field], frozen[field]):
            problems.append(f"summary {field}: {S[field]!r} vs frozen {frozen[field]!r}")
    if compared < 40:
        problems.append(f"only {compared} quantities were compared; expected at least 40")
    return problems


SCALAR_KEYS = [
    "nominal_br", "inc_es", "inc_cer", "inc_cpos", "inc_cluster", "inc_eff",
    "inc_fake", "inc_bias", "inc_track", "inc_sig_model", "inc_bkg_ps",
    "inc_bkg_had", "inc_bkg_model", "inc_model", "inc_sig_ISR", "inc_sig_FSR",
    "inc_bkg_ISR", "inc_bkg_FSR", "inc_scale", "inc_total",
]
BINNED_KEYS = [
    "nominal_binned_br", "es", "cer", "cpos", "cluster", "eff", "fake", "bias",
    "track", "sig_model", "bkg_model", "model", "sig_ISR", "sig_FSR", "bkg_ISR",
    "bkg_FSR", "scale", "total",
]


def flatten_summary(S, M=None):
    """JSON-ready dict of the scalar and binned results (binned: 14 real bins)."""
    out = {k: float(S[k]) for k in SCALAR_KEYS}
    for k in BINNED_KEYS:
        out[k] = [float(v) for v in np.asarray(S[k])[:14]]
    if M is not None:
        for wp in (0.5, 0.8):
            m = M.get(("test_nominal", wp))
            if m is None:
                continue
            tag = "50" if wp == 0.5 else "80"
            out[f"nominal_br_{tag}"] = float(m["br"])
            if "auc" in m:
                out["nominal_auc"] = float(m["auc"])
                out["nominal_acc"] = float(m["acc"])
    return out
