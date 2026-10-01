"""Write a small ROC-curve data file from the FROZEN prediction files.

Run this once on the machine that holds ATLAS_80030_Data.  It reads only the
already-written prediction files, computes the ROC exactly as
`atlas_metrics.py` does (sklearn, unweighted, vector of ones), thins it to a
few thousand points, and writes one small CSV.  It trains nothing, changes
nothing, and writes nothing except that CSV.

    python "Code files\\publication\\make_roc_data.py"

Output:
    Code files\\analysis_results\\publication_figures_round1\\roc_curve_data.csv
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
PROJECT_ROOT = CODE_ROOT.parent


def _resolve_results_root(code_root, project_root):
    """Campaign outputs: the top-level results folder, or the historical
    location beside the code, whichever is actually present."""
    import os as _os
    env = _os.environ.get("ATLAS_80030_RESULTS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    names = ("04_Results", "03_Results", "Results", "results")
    cands = [project_root / n for n in names] + [code_root / "analysis_results"]
    for c in cands:
        if c.is_dir() and any(c.iterdir()):
            return c
    for c in cands:
        if c.is_dir():
            return c
    return code_root / "analysis_results"
PRED = PROJECT_ROOT / "ATLAS_80030_Data" / "predictions"
RESULTS_ROOT = _resolve_results_root(CODE_ROOT, PROJECT_ROOT)
OUT = RESULTS_ROOT / "publication_figures_round1" / "roc_curve_data.csv"

MODEL_SHA256 = "63a23308efbf2ce90418c8e657730f571db4798fc9ca2fb3692cb986de405f4c"
MANIFEST = PRED / "atlas_tagger_xgb_final_production_round1_prediction_manifest.json"
DATASETS = ("test_nominal", "esup", "esdown")
N_POINTS = 2000          # thinned curve points per dataset


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["model_sha256"] != MODEL_SHA256:
        raise SystemExit("prediction manifest is not the frozen final model")

    rows: list[dict[str, object]] = []
    for name in DATASETS:
        path = PRED / f"public_{name}.npz"
        if not path.is_file():
            raise SystemExit(f"missing frozen prediction file: {path}")
        with np.load(path, allow_pickle=False) as data:
            labels = np.asarray(data["labels"]).astype(np.int32)
            scores = np.asarray(data["predictions"]).astype(np.float64)
        expected = manifest["datasets"][name]["prediction_rows"]
        if labels.size != expected:
            raise SystemExit(f"{name}: {labels.size} rows, manifest says {expected}")

        weights = np.ones(labels.shape)          # calc.py convention: unweighted
        fpr, tpr, _ = roc_curve(labels, scores, sample_weight=weights)
        auc = float(roc_auc_score(labels, scores, sample_weight=weights))

        # thin on a fixed signal-efficiency grid so the file stays small
        grid = np.linspace(0.0, 1.0, N_POINTS)
        fpr_grid = np.interp(grid, tpr, fpr)
        for eff, bkg in zip(grid, fpr_grid):
            rows.append({
                "dataset": name,
                "auc": format(auc, ".17g"),
                "prediction_rows": int(labels.size),
                "signal_efficiency": format(float(eff), ".17g"),
                "background_efficiency": format(float(bkg), ".17g"),
            })
        print(f"{name:<14} rows={labels.size:>12,}  AUC={auc:.6f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"\nwrote {OUT}\n{len(rows):,} rows  SHA-256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
