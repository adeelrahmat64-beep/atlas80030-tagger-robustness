"""Produce layout-only variants of the frozen publication tables.

The frozen artifacts in this directory are byte-identical copies of
    Code files/analysis_results/publication_tables_round1/*.tex

Two of them do not fit the journal's column geometry as written.  This script
writes layout/<name>.tex, changing ONLY:

  * the float type, single-column `table` -> two-column `table*`;
  * the addition of \resizebox{\textwidth}{!}{ ... } around the tabular block.

No caption, label, column header, or numeric value is touched.  The script
verifies that afterwards by extracting every numeric token from both files and
requiring the two sequences to be identical; it exits non-zero otherwise.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "layout"

# name -> (promote single-column float to table*, resizebox target: False|"text"|"column")
JOBS = {
    "table1_nominal_performance": (True, "text"),
    "table2_inclusive_systematics_wp50": (True, False),
    "table3_shap_delta_100k_summary": (False, "text"),
}

NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def numeric_tokens(text: str) -> list[str]:
    """Every numeric token, in order, ignoring LaTeX length units."""
    stripped = re.sub(r"\\[a-zA-Z]+", " ", text)
    return NUMBER.findall(stripped)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(text: str, promote: bool, resize) -> str:
    if promote:
        text = text.replace(r"\begin{table}", r"\begin{table*}")
        text = text.replace(r"\end{table}", r"\end{table*}")
    if resize:
        assert r"\resizebox" not in text, "already resized"
        width = r"\textwidth" if resize == "text" else r"\columnwidth"
        text = text.replace(
            r"\begin{tabular}", "\\resizebox{" + width + "}{!}{%\n\\begin{tabular}", 1
        )
        text = text.replace(r"\end{tabular}", "\\end{tabular}%\n}", 1)
    return text


def main() -> int:
    OUT.mkdir(exist_ok=True)
    failures = 0
    print(f"{'table':<40}{'frozen sha256'[:16]:<18}{'layout sha256'[:16]:<18}numbers")
    for name, (promote, resize) in JOBS.items():
        src = HERE / f"{name}.tex"
        dst = OUT / f"{name}.tex"
        text = src.read_text(encoding="utf-8")
        new = transform(text, promote, resize)
        dst.write_text(new, encoding="utf-8")
        same = numeric_tokens(text) == numeric_tokens(new)
        if not same:
            failures += 1
        print(f"{name:<40}{sha256(src)[:16]:<18}{sha256(dst)[:16]:<18}"
              f"{'IDENTICAL' if same else 'CHANGED — ERROR'}")
    if failures:
        print(f"\n{failures} table(s) changed a numeric token; refusing.")
        return 1
    print("\nAll layout variants preserve every numeric token exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
