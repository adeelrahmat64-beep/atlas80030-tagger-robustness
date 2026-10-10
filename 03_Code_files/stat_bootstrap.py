"""ATLAS 80030 - bootstrap statistical uncertainty of every frozen metric.

WHAT IT ANSWERS
    The frozen paper quotes AUC, background rejection (BR) and every
    systematic-uncertainty component without a statistical error.  The ATLAS
    README for this dataset says the statistical uncertainty on the metrics
    "must be smaller than" the systematic differences for those differences to
    be meaningful.  This script measures that statistical uncertainty.

METHOD
    Poisson bootstrap of the FROZEN predictions (nothing is retrained and no
    model is evaluated).  In each replica every jet receives a weight drawn
    from a Poisson distribution of mean 1, the large-sample equivalent of
    resampling with replacement, and the full frozen uncertainty chain
    (uncertainty_chain.compute_chain, which calls atlas_metrics.py's own
    functions) is recomputed with those weights.  The correlations that the
    frozen chain contains are kept:
      * esup, esdown, cer and cpos are re-simulations of the same jets as the
        nominal test sample.  Each of their jets that pairs exactly with a
        nominal jet (pair_features.py's key: event number, float32 bits of
        jet eta and phi, label) carries the weight of its nominal partner;
        the few unpaired cer jets get their own weights;
      * the ISR/FSR shower-weight variations reuse the nominal jets, so they
        carry the nominal weights times the shower ratio;
      * the nominal jets appended to the one-class modelling samples carry
        their nominal weights;
      * the track samples, bias and the modelling samples contain different
        jets and get independent weights.
    The standard deviation of each quantity over the replicas is its
    statistical uncertainty.

SAFETY
    Before any bootstrap replica, the chain is run once on the unweighted frozen
    predictions and compared with the frozen metrics JSON written by
    atlas_metrics.py.  Any difference, even in the last bit, stops the script.
    Nothing under ATLAS_80030_Data is written.  Output goes to a new folder
    04_Results/stat_bootstrap_round<N>/ that is never reused.

RUN
    python stat_bootstrap.py                    # 100 replicas (default)
    python stat_bootstrap.py --replicas 20      # fewer, quicker
    python stat_bootstrap.py --resume <folder>  # continue an interrupted run
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
import pair_features as pf
import uncertainty_chain as uc
from atlas_config import FROZEN_RECONSTRUCTION_FP
from atlas_config import RESULTS_ROOT

BASE_SEED = 20261009          # fixed; replica r uses default_rng([BASE_SEED, r])
DEFAULT_REPLICAS = 100
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
    n = max(used) + 1
    out = RESULTS_ROOT / f"{tag}_round{n}"
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


CALORIMETER = ["esup", "esdown", "cer", "cpos"]


def eligible_keys(ds):
    """Pairing keys and labels of the frozen eligible rows, in prediction order."""
    keys, labels = [], []
    files = sorted((at.CACHE / ds).glob("*.npz"))
    if not files:
        raise SystemExit(f"no feature cache for {ds}")
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            if at.cache_fingerprint(z) != FROZEN_RECONSTRUCTION_FP:
                raise SystemExit(f"{f}: not the frozen reconstruction")
            feats = np.stack([z[c] for c in at.HL], axis=1).astype(np.float64)
            keep = at.eligibility_mask(feats, z["n_constit"])
            keys.append(pf.build_key(z["EventInfo_mcEventNumber"][keep],
                                     z["fjet_eta"][keep], z["fjet_phi"][keep],
                                     z["labels"][keep]))
            labels.append(np.asarray(z["labels"][keep]).astype(np.int32))
    return np.concatenate(keys), np.concatenate(labels)


def build_partners(data):
    """For each calorimeter sample: index of the paired nominal prediction row, or -1."""
    nomk, nlab = eligible_keys("test_nominal")
    if not np.array_equal(nlab, data["test_nominal"]["labels"]):
        raise SystemExit("test_nominal cache rows do not align with the frozen predictions")
    partners = {}
    for ds in CALORIMETER:
        sysk, slab = eligible_keys(ds)
        if not np.array_equal(slab, data[ds]["labels"]):
            raise SystemExit(f"{ds} cache rows do not align with the frozen predictions")
        idx, _, _ = pf.match(nomk, sysk)
        hit = idx >= 0
        if hit.any() and not np.array_equal(data[ds]["labels"][hit],
                                            data["test_nominal"]["labels"][idx[hit]]):
            raise SystemExit(f"{ds}: paired labels disagree")
        partners[ds] = idx
        print(f"  paired {ds:<7} {int(hit.sum()):>10,} of {idx.size:>10,} eligible jets "
              f"({100 * hit.mean():.3f}%) share their nominal partner's weight")
    return partners


def replica_weights(data, partners, r):
    rng = np.random.default_rng([BASE_SEED, r])
    w = {"test_nominal": rng.poisson(1.0, data["test_nominal"]["labels"].size).astype(np.float64)}
    for ds in uc.ORDER[1:]:                    # fixed order -> reproducible draws
        w[ds] = rng.poisson(1.0, data[ds]["labels"].size).astype(np.float64)
        if ds in partners:
            hit = partners[ds] >= 0
            w[ds][hit] = w["test_nominal"][partners[ds][hit]]
    return w


def one_replica(data, partners, r):
    w = replica_weights(data, partners, r)
    # Uncertainty chain at 0.5 (BR only, as the combination needs; AUC and
    # accuracy for the nominal sample), plus the nominal BR at 0.8 for the
    # performance table, computed by the frozen metrics_for().
    M, S = uc.compute_chain(data, working_points=(0.5,), light=True,
                            full_metric_sets=("test_nominal",), weights=w)
    nom = data["test_nominal"]
    m80 = am.metrics_for(nom["labels"], nom["predictions"], w["test_nominal"],
                         nom["pt"], 0.8, am.FULL_BINS)
    flat = uc.flatten_summary(S, M)
    flat["nominal_br_80"] = float(m80["br"])
    for ds in uc.ORDER:
        flat[f"br50_{ds}"] = float(M[(ds, 0.5)]["br"])
    return flat


def summarise(rows):
    keys = list(rows[0].keys())
    out = {}
    for k in keys:
        arr = np.asarray([r[k] for r in rows], dtype=np.float64)
        out[k] = {
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0, ddof=1).tolist(),
            "p16": np.percentile(arr, 15.865, axis=0).tolist(),
            "p84": np.percentile(arr, 84.135, axis=0).tolist(),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replicas", type=int, default=DEFAULT_REPLICAS)
    ap.add_argument("--resume", type=str, default=None,
                    help="existing stat_bootstrap_round<N> folder to continue")
    args = ap.parse_args()
    if args.replicas < 2:
        raise SystemExit("--replicas must be at least 2")

    manifest, metadata = am.require_final_prediction_manifest()
    if not FROZEN_JSON.is_file():
        raise SystemExit(f"frozen metrics JSON not found: {FROZEN_JSON}")

    if args.resume:
        out = Path(args.resume)
        if not (out / "replicas.jsonl").is_file():
            raise SystemExit(f"nothing to resume in {out}")
    else:
        out = new_output_dir("stat_bootstrap")
        (out / "stat_bootstrap_code.py").write_bytes(Path(__file__).read_bytes())
        (out / "uncertainty_chain_code.py").write_bytes(Path(uc.__file__).read_bytes())
    tee = Tee(out / "stat_bootstrap_log.txt")
    sys.stdout = tee
    try:
        print(f"# ATLAS 80030 stat_bootstrap  {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"# output {out}")
        print(f"# frozen model SHA-256 {metadata['model_sha256']}")
        print(f"# base seed {BASE_SEED}; replicas requested {args.replicas}")

        data = {}
        for ds in uc.ORDER:
            d = am.load(ds, manifest)
            if d is None:
                raise SystemExit(f"frozen predictions missing for {ds}")
            data[ds] = d
            print(f"  loaded {ds:<14} {d['labels'].size:>11,} jets")

        # ------------------------------------------------ frozen self-check ---
        t0 = time.time()
        M, S = uc.compute_chain(data, working_points=(0.5, 0.8))
        problems = uc.check_against_frozen(M, S, FROZEN_JSON)
        if problems:
            print("SELF-CHECK FAILED - the chain does not reproduce the frozen metrics:")
            for p in problems[:20]:
                print("  " + p)
            raise SystemExit(2)
        print(f"  self-check PASS: frozen metrics reproduced exactly "
              f"(total {100 * S['inc_total']:.4f}%) in {time.time() - t0:.0f}s")
        central = uc.flatten_summary(S, M)
        for ds in uc.ORDER:
            central[f"br50_{ds}"] = float(M[(ds, 0.5)]["br"])

        partners = build_partners(data)

        # ---------------------------------------------------- replicas ---
        jl = out / "replicas.jsonl"
        rows = []
        if jl.is_file():
            rows = [json.loads(x) for x in jl.read_text(encoding="utf-8").splitlines() if x.strip()]
            print(f"  resuming after {len(rows)} completed replicas")
        start = len(rows)
        with jl.open("a", encoding="utf-8") as fh:
            for r in range(start, args.replicas):
                t1 = time.time()
                flat = one_replica(data, partners, r)
                flat["replica"] = r
                fh.write(json.dumps(flat) + "\n")
                fh.flush()
                rows.append(flat)
                dt = time.time() - t1
                left = (args.replicas - r - 1) * dt
                print(f"  replica {r + 1:>4}/{args.replicas}  {dt:6.1f}s  "
                      f"total {100 * flat['inc_total']:.3f}%  ~{left / 60:.0f} min left")

        rows = [r for r in rows if r["replica"] < args.replicas]
        if sorted(r["replica"] for r in rows) != list(range(args.replicas)):
            raise SystemExit("replica set is incomplete or duplicated")
        stats = summarise([{k: v for k, v in r.items() if k != "replica"} for r in rows])
        result = {
            "description": "bootstrap statistical uncertainty of the frozen metrics",
            "method": "Poisson(1) bootstrap weights per jet; exactly paired "
                      "esup/esdown/cer/cpos jets share their nominal partner's "
                      "weight; shower variations and modelling complements reuse "
                      "the nominal weights; track, bias and modelling samples "
                      "independent; frozen atlas_metrics chain recomputed",
            "paired_jets": {ds: int((partners[ds] >= 0).sum()) for ds in CALORIMETER},
            "replicas": args.replicas,
            "base_seed": BASE_SEED,
            "frozen_model_sha256": metadata["model_sha256"],
            "frozen_metrics_json": str(FROZEN_JSON),
            "frozen_metrics_json_sha256": sha256(FROZEN_JSON),
            "central": central,
            "bootstrap": stats,
            "timestamp": datetime.now().astimezone().isoformat(),
        }
        (out / "stat_bootstrap_summary.json").write_text(json.dumps(result, indent=1) + "\n",
                                                        encoding="utf-8")

        print("\n  inclusive results (central value, bootstrap std), in percent:")
        for k in uc.SCALAR_KEYS:
            if k == "nominal_br":
                continue
            print(f"    {k:<16} {100 * central[k]:8.3f}  +- {100 * stats[k]['std']:.3f}")
        print(f"    nominal BR@50   {central['nominal_br_50']:8.3f}  +- "
              f"{stats['nominal_br_50']['std']:.3f}")
        print(f"    nominal BR@80   {central['nominal_br_80']:8.3f}  +- "
              f"{stats['nominal_br_80']['std']:.3f}")
        print(f"    nominal AUC     {central['nominal_auc']:8.5f}  +- "
              f"{stats['nominal_auc']['std']:.5f}")
        print(f"\n  wrote {out / 'stat_bootstrap_summary.json'}")
        print("done.")
    finally:
        sys.stdout = tee.out
        tee.fh.close()


if __name__ == "__main__":
    main()
