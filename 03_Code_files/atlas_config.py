"""Shared operational configuration for the ATLAS 80030 pipeline.

The reconstruction algorithms do not live here.  This module only defines the
data root, campaign membership, and the released reconstruction fingerprint.
Set ATLAS_80030_DATA_ROOT to override the workspace-local default.
"""

from __future__ import annotations

import os
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "ATLAS_80030_Data"
DATA_ROOT = Path(
    os.environ.get("ATLAS_80030_DATA_ROOT", str(DEFAULT_DATA_ROOT))
).expanduser().resolve()

FROZEN_RECONSTRUCTION_FP = "c28a262540824e9db90f34c421f10c70"
def _resolve_results_root() -> Path:
    """Directory that holds every campaign output.

    Historically this sat beside the code as ``Code files/analysis_results``.
    The project root was later reorganised into numbered folders and the
    results were lifted to the top level, so this resolves whichever layout is
    actually on disk and the pipeline runs unchanged in both.

    Precedence: an explicit ATLAS_80030_RESULTS_ROOT override; then the first
    candidate directory that exists and is non-empty; then the first that
    merely exists; then the historical location.
    """
    override = os.environ.get("ATLAS_80030_RESULTS_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    beside_code = Path(__file__).resolve().parent / "analysis_results"
    candidates = [WORKSPACE_ROOT / name
                  for name in ("04_Results", "03_Results", "Results", "results")]
    candidates.append(beside_code)
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return beside_code


RESULTS_ROOT = _resolve_results_root()


TRAINING_DATASET = "train_nominal"
NOMINAL_EVALUATION_DATASET = "test_nominal"
DETECTOR_SYSTEMATICS = (
    "esup", "esdown", "cer", "cpos", "teg", "tej", "tfl", "tfj", "bias",
)
MODELLING_DATASETS = (
    "ttbar_pythia", "ttbar_herwig", "cluster", "string", "angular", "dipole",
)
EVALUATION_DATASETS = (
    NOMINAL_EVALUATION_DATASET, *DETECTOR_SYSTEMATICS, *MODELLING_DATASETS,
)
REQUIRED_DATASETS = (TRAINING_DATASET, *EVALUATION_DATASETS)


def source_dataset_dir(dataset: str) -> Path:
    """Return the source-HDF5 directory for a required dataset."""
    direct = DATA_ROOT / "data" / dataset
    nested = DATA_ROOT / "data" / "modelling" / dataset
    if direct.is_dir():
        return direct
    if dataset in MODELLING_DATASETS and nested.is_dir():
        return nested
    return direct
