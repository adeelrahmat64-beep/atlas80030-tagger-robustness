"""ATLAS 80030 - how each input feature responds to the calorimeter variations.

WHAT IT ANSWERS
    The frozen SHAP-Delta campaign finds the largest attribution drift on
    ECF3 under the cluster energy-scale variations.  The physical reason is the
    momentum power each feature carries (SCALE_DEGREE in tagger_variants.py):
    a coherent constituent momentum shift (1 + s) moves a degree-k feature by
    about (1 + s)^k - 1, so ECF3 (degree 3) moves about three times as much as
    ECF1 (degree 1), and the dimensionless ratios hardly move.  This script
    measures the per-jet relative response sys/nom - 1 of every feature on the
    exactly paired jets and prints it next to that expectation.

INPUT
    The frozen paired frames <data-root>/paired/<sample>_paired.npz written by
    pair_features.py (nom_<feature> and sys_<feature> for each matched jet).
    Only read.

OUTPUT
    A new folder 04_Results/feature_response_round<N>/ with a JSON summary, a
    CSV table and a log.  Nothing else is written.

    The expectation column is illustrative only.  The ATLAS energy-scale
    variation shifts calorimeter-based constituents and not track-based ones,
    so the shift is not exactly uniform and the ratios are not exactly
    invariant; the measured columns are the result.

RUN
    python feature_response.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from atlas_config import DATA_ROOT, RESULTS_ROOT
from tagger_variants import SCALE_DEGREE

SAMPLES = ["esup", "esdown", "cer", "cpos"]
HL = list(SCALE_DEGREE)
PAIRED = DATA_ROOT / "paired"


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


def response(nom, sys_):
    nom = np.asarray(nom, dtype=np.float64)
    sys_ = np.asarray(sys_, dtype=np.float64)
    good = (np.isfinite(nom) & np.isfinite(sys_) & (nom > -9.0e5) & (sys_ > -9.0e5)
            & (nom != 0))
    r = sys_[good] / nom[good] - 1.0
    a = np.abs(r)
    return {
        "n": int(r.size),
        "n_excluded": int((~good).sum()),
        "median": float(np.median(r)),
        "mean": float(np.mean(r)),
        "p16": float(np.percentile(r, 15.865)),
        "p84": float(np.percentile(r, 84.135)),
        "median_abs": float(np.median(a)),
        "fraction_unchanged": float(np.mean(r == 0.0)),
    }


def main():
    out = new_output_dir("feature_response")
    (out / "feature_response_code.py").write_bytes(Path(__file__).read_bytes())
    tee = Tee(out / "feature_response_log.txt")
    sys.stdout = tee
    try:
        print(f"# ATLAS 80030 feature_response  {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"# paired frames from {PAIRED}")
        result = {"scale_degree": SCALE_DEGREE, "samples": {}}
        rows = []
        for ds in SAMPLES:
            path = PAIRED / f"{ds}_paired.npz"
            if not path.is_file():
                raise SystemExit(f"paired frame missing: {path}")
            with np.load(path, allow_pickle=False) as z:
                missing = [f for f in HL if f"nom_{f}" not in z.files or f"sys_{f}" not in z.files]
                if missing:
                    raise SystemExit(f"{path}: missing paired columns {missing}")
                stats = {f: response(z[f"nom_{f}"], z[f"sys_{f}"]) for f in HL}
            s1 = stats["fjet_ECF1"]["median"]          # ECF1 = sum pt exactly
            for f in HL:
                k = SCALE_DEGREE[f]
                stats[f]["degree"] = k
                stats[f]["uniform_scaling_expectation"] = float((1.0 + s1) ** k - 1.0)
            result["samples"][ds] = stats
            n_pairs = stats["fjet_ECF1"]["n"] + stats["fjet_ECF1"]["n_excluded"]
            print(f"\n  {ds}: {n_pairs:,} paired jets; median sum-pT shift (ECF1) "
                  f"{100 * s1:+.4f}%")
            print(f"    {'feature':<16}{'deg':>4}{'median %':>11}{'p16 %':>10}{'p84 %':>10}"
                  f"{'median|r| %':>13}{'(1+s)^k-1 %':>13}")
            for f in HL:
                st = stats[f]
                print(f"    {f:<16}{st['degree']:>4}{100 * st['median']:>11.4f}"
                      f"{100 * st['p16']:>10.4f}{100 * st['p84']:>10.4f}"
                      f"{100 * st['median_abs']:>13.4f}"
                      f"{100 * st['uniform_scaling_expectation']:>13.4f}")
                rows.append({"sample": ds, "feature": f, **st})
        result["timestamp"] = datetime.now().astimezone().isoformat()
        (out / "feature_response_summary.json").write_text(json.dumps(result, indent=1) + "\n",
                                                          encoding="utf-8")
        with (out / "feature_response_table.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  wrote {out / 'feature_response_summary.json'}")
        print("done.")
    finally:
        sys.stdout = tee.out
        tee.fh.close()


if __name__ == "__main__":
    main()
