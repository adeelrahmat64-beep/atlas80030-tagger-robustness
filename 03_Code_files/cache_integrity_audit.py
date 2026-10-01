"""Read-only integrity audit for the frozen ATLAS 80030 HLF cache campaign."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from atlas_config import (
    DATA_ROOT,
    FROZEN_RECONSTRUCTION_FP,
    REQUIRED_DATASETS,
    source_dataset_dir,
)


HL = (
    "fjet_Tau1_wta", "fjet_Tau2_wta", "fjet_Tau3_wta", "fjet_Tau4_wta",
    "fjet_Split12", "fjet_Split23", "fjet_ECF1", "fjet_ECF2", "fjet_ECF3",
    "fjet_C2", "fjet_D2", "fjet_L2", "fjet_L3", "fjet_Qw",
    "fjet_ThrustMaj",
)
OLD_FP = "0d53c1d830a1f88142af02405c4f3d72"
CER_SENTINEL_CUT = -900.0
RECONSTRUCTION_SENTINEL = -999000.0
from atlas_config import RESULTS_ROOT  # noqa: E402
RESULTS = RESULTS_ROOT


def scalar_text(value) -> str:
    arr = np.asarray(value)
    if arr.size != 1:
        return f"(non-scalar shape {arr.shape})"
    return str(arr.reshape(()).item())


def audit_file(dataset: str, source: Path, cache: Path) -> dict:
    issues = []
    with h5py.File(source, "r") as handle:
        source_rows = int(handle["labels"].shape[0])
        sentinel = None
        if dataset == "cer":
            sentinel = np.asarray(handle["fjet_L3"][:]) <= CER_SENTINEL_CUT

    fingerprint = "(unreadable)"
    feature_nonfinite = {}
    auxiliary_nonfinite = {}
    nonfinite_rows = np.zeros(source_rows, dtype=bool)
    feature_nonfinite_rows = np.zeros(source_rows, dtype=bool)
    reconstruction_sentinel_rows = np.zeros(source_rows, dtype=bool)
    other_feature_nonfinite = np.zeros(source_rows, dtype=bool)
    l3_nonfinite = np.zeros(source_rows, dtype=bool)
    l3_negative_infinity = np.zeros(source_rows, dtype=bool)
    l3_sentinel_zero_denominator = np.zeros(source_rows, dtype=bool)
    low_multiplicity = np.zeros(source_rows, dtype=bool)
    try:
        with np.load(cache, allow_pickle=False) as data:
            keys = set(data.files)
            missing_features = sorted(set(HL) - keys)
            if missing_features:
                issues.append(f"missing features: {missing_features}")

            if "labels" not in keys:
                cache_rows = None
                issues.append("missing labels")
            else:
                labels = data["labels"]
                cache_rows = int(labels.shape[0]) if labels.ndim == 1 else None
                if labels.ndim != 1:
                    issues.append(f"labels is not 1D: {labels.shape}")

            if "code_fp" not in keys:
                fingerprint = "(unstamped)"
                issues.append("missing code_fp")
            else:
                fingerprint = scalar_text(data["code_fp"])
                if fingerprint != FROZEN_RECONSTRUCTION_FP:
                    issues.append(f"wrong fingerprint: {fingerprint}")

            if cache_rows != source_rows:
                issues.append(
                    f"row mismatch: source={source_rows}, cache={cache_rows}"
                )

            if "n_constit" not in keys:
                issues.append("missing n_constit")
            else:
                n_constit = data["n_constit"]
                if n_constit.ndim != 1 or n_constit.shape[0] != source_rows:
                    issues.append(
                        f"n_constit shape mismatch: {n_constit.shape}"
                    )
                else:
                    low_multiplicity = n_constit < 3

            for feature in HL:
                if feature not in keys:
                    continue
                values = data[feature]
                if values.ndim != 1:
                    issues.append(f"{feature} is not 1D: {values.shape}")
                    continue
                if values.shape[0] != source_rows:
                    issues.append(
                        f"{feature} row mismatch: {values.shape[0]} != {source_rows}"
                    )
                    continue
                bad = ~np.isfinite(values)
                reconstruction_sentinel_rows |= values <= -9.0e5
                count = int(bad.sum())
                if count:
                    feature_nonfinite[feature] = count
                    nonfinite_rows |= bad
                    feature_nonfinite_rows |= bad
                    if feature == "fjet_L3":
                        l3_nonfinite |= bad
                        l3_negative_infinity |= np.isneginf(values)
                    else:
                        other_feature_nonfinite |= bad

            for field in sorted(keys - set(HL) - {"code_fp"}):
                values = data[field]
                if (
                    not np.issubdtype(values.dtype, np.number)
                    or values.shape != (source_rows,)
                ):
                    continue
                bad = ~np.isfinite(values)
                count = int(bad.sum())
                if count:
                    auxiliary_nonfinite[field] = count
                    nonfinite_rows |= bad
                    other_feature_nonfinite |= bad

            # Frozen L3 is 1e3/(3e3)^(1/3).  For fewer than four constituents,
            # frozen gen_ecf3 returns the reconstruction sentinel for 1e3 and
            # exactly zero for 3e3, deterministically producing -Inf.  The
            # training selection explicitly removes non-finite feature rows.
            # Record this boundary policy separately from corrupt or otherwise
            # unexpected non-finite cache content.
            if (
                "gen_1e3_b1" in keys
                and "gen_3e3_b1" in keys
                and data["gen_1e3_b1"].shape == (source_rows,)
                and data["gen_3e3_b1"].shape == (source_rows,)
            ):
                gen1 = data["gen_1e3_b1"]
                gen3 = data["gen_3e3_b1"]
                l3_sentinel_zero_denominator = (
                    l3_negative_infinity
                    & (gen1 == RECONSTRUCTION_SENTINEL)
                    & (gen3 == 0.0)
                )
    except Exception as exc:
        cache_rows = None
        issues.append(f"NPZ read failed: {exc.__class__.__name__}: {exc}")

    bad_rows = int(nonfinite_rows.sum())
    low_multiplicity_rows = int(low_multiplicity.sum())
    low_multiplicity_bad_rows = int((low_multiplicity & nonfinite_rows).sum())
    sentinel_rows = int(sentinel.sum()) if sentinel is not None else 0
    sentinel_bad_rows = int((sentinel & nonfinite_rows).sum()) if sentinel is not None else 0
    zero_denominator_rows = int(l3_sentinel_zero_denominator.sum())
    zero_denominator_low_multiplicity_rows = int(
        (l3_sentinel_zero_denominator & low_multiplicity).sum()
    )
    zero_denominator_sentinel_rows = int(
        (l3_sentinel_zero_denominator & sentinel).sum()
    ) if sentinel is not None else 0
    allowed_nonfinite = low_multiplicity.copy()
    if sentinel is not None:
        allowed_nonfinite |= sentinel
    allowed_nonfinite |= l3_sentinel_zero_denominator
    unexpected_mask = (
        other_feature_nonfinite
        | (l3_nonfinite & ~allowed_nonfinite)
    )
    unexpected_bad_rows = int(unexpected_mask.sum())
    if unexpected_bad_rows:
        issues.append(f"unexpected non-finite rows: {unexpected_bad_rows}")
    sentinel_feature_rows = int(reconstruction_sentinel_rows.sum())
    eligible_without_sentinel_guard = (~feature_nonfinite_rows) & (~low_multiplicity)
    sentinel_eligible_rows = int(
        (reconstruction_sentinel_rows & eligible_without_sentinel_guard).sum()
    )
    if sentinel_eligible_rows:
        issues.append(
            "reconstruction sentinel would survive finite/multiplicity eligibility: "
            f"{sentinel_eligible_rows} rows"
        )

    return {
        "file": cache.name,
        "source_rows": source_rows,
        "cache_rows": cache_rows,
        "fingerprint": fingerprint,
        "feature_nonfinite": feature_nonfinite,
        "auxiliary_nonfinite": auxiliary_nonfinite,
        "nonfinite_rows": bad_rows,
        "low_multiplicity_rows": low_multiplicity_rows,
        "low_multiplicity_nonfinite_rows": low_multiplicity_bad_rows,
        "cer_sentinel_rows": sentinel_rows,
        "cer_sentinel_nonfinite_rows": sentinel_bad_rows,
        "l3_sentinel_zero_denominator_rows": zero_denominator_rows,
        "l3_sentinel_zero_denominator_low_multiplicity_rows": (
            zero_denominator_low_multiplicity_rows
        ),
        "l3_sentinel_zero_denominator_cer_sentinel_rows": (
            zero_denominator_sentinel_rows
        ),
        "unexpected_nonfinite_rows": unexpected_bad_rows,
        "reconstruction_sentinel_feature_rows": sentinel_feature_rows,
        "reconstruction_sentinel_rows_eligible_without_guard": sentinel_eligible_rows,
        "issues": issues,
    }


def main() -> int:
    cache_root = DATA_ROOT / "hl_recomputed"
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(DATA_ROOT),
        "expected_fingerprint": FROZEN_RECONSTRUCTION_FP,
        "datasets": {},
        "global_issues": [],
    }
    all_fingerprints = Counter()

    for dataset in REQUIRED_DATASETS:
        source_dir = source_dataset_dir(dataset)
        cache_dir = cache_root / dataset
        sources = {p.stem: p for p in sorted(source_dir.glob("*.h5"))}
        caches = {p.stem: p for p in sorted(cache_dir.glob("*.npz"))}
        missing = sorted(set(sources) - set(caches))
        extra = sorted(set(caches) - set(sources))
        dataset_issues = []
        if not source_dir.is_dir():
            dataset_issues.append(f"source directory missing: {source_dir}")
        if len(caches) != len(sources):
            dataset_issues.append(
                f"file-count mismatch: source={len(sources)}, cache={len(caches)}"
            )
        if missing:
            dataset_issues.append(f"missing cache files: {missing}")
        if extra:
            dataset_issues.append(f"extra cache files: {extra}")

        source_total_rows = 0
        for source in sources.values():
            with h5py.File(source, "r") as handle:
                source_total_rows += int(handle["labels"].shape[0])

        files = []
        for stem in sorted(set(sources) & set(caches)):
            result = audit_file(dataset, sources[stem], caches[stem])
            files.append(result)
            all_fingerprints[result["fingerprint"]] += 1
            dataset_issues.extend(
                f"{result['file']}: {issue}" for issue in result["issues"]
            )

        report["datasets"][dataset] = {
            "source_directory": str(source_dir),
            "source_files": len(sources),
            "cache_files": len(caches),
            "source_rows": source_total_rows,
            "cache_rows": sum(
                item["cache_rows"] or 0 for item in files
            ),
            "nonfinite_rows": sum(item["nonfinite_rows"] for item in files),
            "low_multiplicity_rows": sum(
                item["low_multiplicity_rows"] for item in files
            ),
            "low_multiplicity_nonfinite_rows": sum(
                item["low_multiplicity_nonfinite_rows"] for item in files
            ),
            "cer_sentinel_rows": sum(item["cer_sentinel_rows"] for item in files),
            "cer_sentinel_nonfinite_rows": sum(
                item["cer_sentinel_nonfinite_rows"] for item in files
            ),
            "l3_sentinel_zero_denominator_rows": sum(
                item["l3_sentinel_zero_denominator_rows"] for item in files
            ),
            "l3_sentinel_zero_denominator_low_multiplicity_rows": sum(
                item["l3_sentinel_zero_denominator_low_multiplicity_rows"]
                for item in files
            ),
            "l3_sentinel_zero_denominator_cer_sentinel_rows": sum(
                item["l3_sentinel_zero_denominator_cer_sentinel_rows"]
                for item in files
            ),
            "issues": dataset_issues,
            "files": files,
        }

    every_cache = sorted(cache_root.rglob("*.npz")) if cache_root.is_dir() else []
    required_cache_paths = {
        str(cache_root / ds / f["file"])
        for ds, detail in report["datasets"].items()
        for f in detail["files"]
    }
    unclaimed = [str(p) for p in every_cache if str(p) not in required_cache_paths]
    if unclaimed:
        report["global_issues"].append(f"unclaimed cache NPZ files: {unclaimed}")
        for path in every_cache:
            if str(path) in required_cache_paths:
                continue
            try:
                with np.load(path, allow_pickle=False) as data:
                    fp = scalar_text(data["code_fp"]) if "code_fp" in data.files else "(unstamped)"
            except Exception:
                fp = "(unreadable)"
            all_fingerprints[fp] += 1

    report["fingerprints"] = dict(sorted(all_fingerprints.items()))
    if OLD_FP in all_fingerprints:
        report["global_issues"].append(
            f"old fingerprint present in {all_fingerprints[OLD_FP]} cache files"
        )
    unexpected_fps = sorted(
        fp for fp in all_fingerprints if fp != FROZEN_RECONSTRUCTION_FP
    )
    if unexpected_fps:
        report["global_issues"].append(
            f"unexpected cache fingerprints: {unexpected_fps}"
        )

    mismatches = sum(
        len(detail["issues"]) for detail in report["datasets"].values()
    ) + len(report["global_issues"])
    report["mismatch_count"] = mismatches
    report["status"] = "PASS" if mismatches == 0 else "FAIL"

    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "cache_integrity_audit.json"
    text_path = RESULTS / "cache_integrity_audit.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "ATLAS 80030 frozen-cache integrity audit",
        f"data root: {DATA_ROOT}",
        f"expected fingerprint: {FROZEN_RECONSTRUCTION_FP}",
        "",
        "dataset          source cache  source rows  cache rows  nonfinite  L3-sent  low-N  sentinel",
    ]
    for dataset, detail in report["datasets"].items():
        lines.append(
            f"{dataset:<16}{detail['source_files']:>6}{detail['cache_files']:>6}"
            f"{detail['source_rows']:>13}{detail['cache_rows']:>12}"
            f"{detail['nonfinite_rows']:>11}{detail['l3_sentinel_zero_denominator_rows']:>9}"
            f"{detail['low_multiplicity_rows']:>7}"
            f"{detail['cer_sentinel_rows']:>10}"
        )
        lines.extend(f"  MISMATCH: {issue}" for issue in detail["issues"])
    lines.extend(["", f"fingerprints: {report['fingerprints']}"])
    lines.extend(f"MISMATCH: {issue}" for issue in report["global_issues"])
    lines.extend(["", f"CACHE CAMPAIGN = {report['status']}"])
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nJSON report: {json_path}")
    print(f"Text report: {text_path}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
