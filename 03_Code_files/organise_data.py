# organise_data.py
#
# Takes whatever download_data.py left in _downloads and:
#   1. decompresses it if it is a .gz
#   2. files it as  data\<dataset>\<dataset>_NNN.h5
#   3. deletes the .gz afterwards, to keep disk use down
#
# No external gzip binary is needed; Python handles it.
#
# Safe to run at any time, including while a download is still going, so run
# it periodically during long downloads to stop the staging folder growing.
#
# Run:  python "Code files\organise_data.py"

import gzip
import re
import shutil
from pathlib import Path

from atlas_config import DATA_ROOT

STORE = DATA_ROOT
STAGING = STORE / "_downloads"
DATA = STORE / "data"

NAME_RE = re.compile(r"^(?P<ds>.+)_(?P<num>\d+)\.h5(?P<gz>\.gz)?$")
line = "=" * 70


def human(n):
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def main():
    if not STAGING.exists():
        raise SystemExit(f"staging folder not found:\n  {STAGING}")

    found = [p for p in STAGING.rglob("*") if p.is_file()
             and (p.name.endswith(".h5") or p.name.endswith(".h5.gz"))]
    if not found:
        raise SystemExit(f"no .h5 files in _downloads.\n  {STAGING}")

    print(line)
    print(f"{len(found)} files found")
    print(line)

    moved, skipped, unknown, bad = 0, 0, [], []
    for src in sorted(found):
        m = NAME_RE.match(src.name)
        if not m:
            unknown.append(src.name)
            continue

        ds = m.group("ds")
        target_dir = DATA / ds
        target_dir.mkdir(parents=True, exist_ok=True)
        final = target_dir / f"{ds}_{m.group('num')}.h5"

        if final.exists():
            print(f"  already filed, skipping : {ds}/{final.name}")
            skipped += 1
            continue

        if m.group("gz"):
            print(f"  decompressing : {src.name}  ->  {ds}/{final.name}")
            tmp = final.with_suffix(".h5.part")
            try:
                with gzip.open(src, "rb") as fin, open(tmp, "wb") as fout:
                    shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)
            except Exception as err:
                # truncated archive: remove both, download_data.py will refetch
                tmp.unlink(missing_ok=True)
                src.unlink()
                print(f"     FILE WAS TRUNCATED ({err.__class__.__name__}) - removed.")
                bad.append(src.name)
                continue
            tmp.rename(final)
            src.unlink()          # the .gz is no longer needed
        else:
            print(f"  moving : {src.name}  ->  {ds}/{final.name}")
            shutil.move(str(src), str(final))
        moved += 1

    if unknown:
        print("\n  unrecognised names (file these by hand):")
        for n in unknown:
            print("    " + n)

    print("\n" + line)
    print(f"data folder now ({DATA}):")
    print(line)
    grand = 0
    for d in sorted(DATA.iterdir()):
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.h5"))
        size = sum(f.stat().st_size for f in files)
        grand += size
        state = human(size) if files else "empty"
        print(f"  {d.name:<16s} {len(files):>3d} files   {state}")
    print(f"\n  total: {human(grand)}")
    print(f"  ({moved} new, {skipped} already present)")

    if bad:
        print(f"\n  {len(bad)} files were truncated and have been removed:")
        for n in bad:
            print("    " + n)
        print("\n  Next:  python \"Code files\\download_data.py\"  (run again)")
    else:
        print("\nNext:  python \"Code files\\validate_data5.py\"")


if __name__ == "__main__":
    main()
