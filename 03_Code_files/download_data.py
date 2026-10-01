# download_data.py   (version 4 - parts)
#
# Downloads ATLAS Open Data record 80030 into the configured data root.
#
# The full set is about 584 files / 200 GB, which is too long for one run.
# The work is therefore split into named PARTS. Run one part, check it, move
# on. Every part is independently resumable.
#
#   python "Code files\download_data.py"           show status, do nothing
#   python "Code files\download_data.py" 1         download part 1
#   python "Code files\download_data.py" all       every part, one after another
#
# Running with no argument never downloads anything. That is deliberate: it
# lets you confirm the storage path is correct before committing to hours of
# transfer.
#
# Why this wrapper exists: cernopendata-client prints "Success!" even when the
# transfer was cut short, so a truncated .gz looks like a finished download.
# Every file is decompressed end-to-end before it is accepted.
#
# Each verified file is extracted into data\<dataset>\ immediately and its .gz
# is deleted, so the staging folder never grows and disk use stays flat.
#
# Run organise_data.py afterwards only if you want a summary; extraction is
# already done here.

import gzip
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from atlas_config import DATA_ROOT

STORE = DATA_ROOT
STAGING = STORE / "_downloads"
DATA = STORE / "data"

RECID = "80030"
RETRIES = 5            # attempts per file
PAUSE = 60             # seconds to wait after a failed attempt
ROUNDS = 3             # extra passes over the files that failed

# How many files to take from each dataset.
#   systematics   50 files = 5M jets, meets the ATLAS README recommendation
#   test_nominal  104 = every file; supplies the nominal partner for pairing
#   train_nominal 50 = 5M jets, the exact files used by the frozen final model
N_FILES = {
    "train_nominal": 50,
    "test_nominal": 104,
    "esup": 50, "esdown": 50, "cer": 50, "cpos": 50,
    "teg": 50, "tej": 50, "tfl": 50, "tfj": 50, "bias": 50,
}

# The download is split into these parts. Each is roughly 40 GB except where
# noted. Order is deliberate: the calorimeter systematics come first because
# they are the ones with a 100% pairing rate.
PARTS = {
    1: ["esup", "esdown"],          # energy scale up / down
    2: ["cer", "cpos"],             # cluster energy resolution / position
    3: ["teg", "tej"],              # track efficiency / fake, ghost + jet
    4: ["tfl", "tfj"],              # track fake, loose + jet
    5: ["bias"],                    # about 21 GB
    6: ["train_nominal"],           # about 11 GB, needed for training
    7: ["test_nominal"],            # already complete; here for repair only
}

NAME_RE = re.compile(r"^(?P<ds>.+)_(?P<num>\d+)\.h5(?P<gz>\.gz)?$")
line = "=" * 72


# ---------------------------------------------------------------- helpers
def human(n):
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def gz_is_complete(path):
    """Decompress the whole file. A truncated archive raises here."""
    try:
        with gzip.open(path, "rb") as f:
            while f.read(8 * 1024 * 1024):
                pass
        return True
    except Exception:
        return False


def find_in_staging(name):
    hits = list(STAGING.rglob(name))
    return hits[0] if hits else None


def final_path(name):
    m = NAME_RE.match(name)
    ds, num = m.group("ds"), m.group("num")
    return DATA / ds / f"{ds}_{num}.h5"


def already_extracted(name):
    # is_file(), not exists(): an old "train_nominal_000.h5" was actually a
    # DIRECTORY, and exists() reported it as present.
    return final_path(name).is_file()


def extract_and_file(name):
    """Decompress a staged .gz straight into data\\<dataset>\\ and drop the .gz.

    Writes to a .part file first, so an interrupted extraction never leaves a
    half-written .h5 that later looks complete."""
    src = find_in_staging(name)
    if src is None:
        return False
    final = final_path(name)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_suffix(".h5.part")
    try:
        if src.name.endswith(".gz"):
            with gzip.open(src, "rb") as fin, open(tmp, "wb") as fout:
                shutil.copyfileobj(fin, fout, length=16 * 1024 * 1024)
            tmp.rename(final)
            src.unlink()
        else:
            shutil.move(str(src), str(final))
    except Exception as err:
        tmp.unlink(missing_ok=True)
        print(f"      extraction failed ({err.__class__.__name__})")
        return False
    return True


def list_record_files():
    res = subprocess.run(
        ["cernopendata-client", "get-file-locations", "--recid", RECID],
        capture_output=True, text=True)
    if res.returncode != 0:
        print("\ncernopendata-client failed:\n")
        print(res.stdout or "", res.stderr or "")
        print("\nIf the command was not recognised:  pip install cernopendata-client")
        sys.exit(1)
    names = [t.rsplit("/", 1)[-1] for t in res.stdout.split()
             if t.endswith(".h5") or t.endswith(".h5.gz")]
    if not names:
        print("\nNo .h5 names found. Raw output:\n")
        print(res.stdout[:3000])
        sys.exit(1)
    groups = defaultdict(list)
    for n in sorted(set(names)):
        m = NAME_RE.match(n)
        if m:
            groups[m.group("ds")].append(n)
    return groups


def wanted_for(ds, groups):
    """The file names this dataset should end up with."""
    return sorted(groups.get(ds, []))[:N_FILES.get(ds, 0)]


# ---------------------------------------------------------------- status
def show_status(groups):
    print(line)
    print("STATUS")
    print(line)
    print(f"  storage root : {STORE}")
    print(f"  data folder  : {DATA}")
    print()
    print(f"  {'part':>4s}  {'dataset':<15s} {'have':>5s} {'want':>5s} "
          f"{'missing':>8s} {'on disk':>9s}")
    print("  " + "-" * 60)

    grand_missing = 0
    for part, datasets in PARTS.items():
        for ds in datasets:
            want = wanted_for(ds, groups)
            have = sum(1 for n in want if already_extracted(n))
            missing = len(want) - have
            grand_missing += missing
            d = DATA / ds
            size = sum(f.stat().st_size for f in d.glob("*.h5")) if d.is_dir() else 0
            flag = "" if missing == 0 else "   <-- to do"
            print(f"  {part:>4d}  {ds:<15s} {have:>5d} {len(want):>5d} "
                  f"{missing:>8d} {human(size):>9s}{flag}")

    total_size = sum(f.stat().st_size
                     for d in DATA.iterdir() if d.is_dir()
                     for f in d.glob("*.h5")) if DATA.is_dir() else 0
    print("  " + "-" * 60)
    print(f"  total on disk : {human(total_size)}")
    print(f"  still missing : {grand_missing} files "
          f"(about {human(grand_missing * 419e6)})")
    print()
    print("  To download one part:   python \"Code files\\download_data.py\" 1")
    print("  To download everything: python \"Code files\\download_data.py\" all")


# ---------------------------------------------------------------- download
def fetch_one(name):
    """Download one file, verify it decompresses, then file it away."""
    for attempt in range(1, RETRIES + 1):
        res = subprocess.run(
            ["cernopendata-client", "download-files",
             "--recid", RECID, "--filter-name", name],
            cwd=str(STAGING), capture_output=True, text=True)

        path = find_in_staging(name)
        if res.returncode == 0 and path and gz_is_complete(path):
            kib = path.stat().st_size // 1024
            if extract_and_file(name):
                print(f"      OK  ({kib:,} KiB downloaded, extracted)")
                return True
            return False

        why = "truncated file" if path else "nothing downloaded"
        if "getaddrinfo" in (res.stderr or ""):
            why = "network or DNS unavailable"
        print(f"      attempt {attempt}/{RETRIES} failed - {why}")

        if path:
            path.unlink()               # a partial file is of no use
        if attempt < RETRIES:
            print(f"      waiting {PAUSE}s before retrying ...")
            time.sleep(PAUSE)
    return False


def run_part(part, groups):
    datasets = PARTS[part]
    todo = []
    for ds in datasets:
        for name in wanted_for(ds, groups):
            if already_extracted(name):
                continue
            path = find_in_staging(name)
            if path and not gz_is_complete(path):
                print(f"  TRUNCATED, deleting : {name}")
                path.unlink()
            elif path:
                if extract_and_file(name):
                    print(f"  staged already, extracted : {name}")
                    continue
            todo.append(name)

    print("\n" + line)
    print(f"PART {part}: {', '.join(datasets)}")
    print(line)
    if not todo:
        print("  Nothing to do, this part is complete.")
        return True

    print(f"  {len(todo)} files to fetch (about {human(len(todo) * 419e6)})")

    done, pending = [], list(todo)
    for rnd in range(1, ROUNDS + 1):
        if not pending:
            break
        if rnd > 1:
            print(f"\n  round {rnd}/{ROUNDS} - retrying {len(pending)} files")
        failed = []
        for i, name in enumerate(pending, 1):
            print(f"\n  [{i}/{len(pending)}] {name}")
            (done if fetch_one(name) else failed).append(name)
        pending = failed

    print(f"\n  part {part}: {len(done)} succeeded, {len(pending)} failed")
    for n in pending:
        print(f"      {n}")
    return not pending


# ---------------------------------------------------------------- main
def main():
    if not STORE.is_dir():
        raise SystemExit(
            f"storage root not found:\n  {STORE}\n\n"
            "Check the drive letter and the folder name, then edit STORE at "
            "the top of this script.")
    STAGING.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    groups = list_record_files()
    missing_ds = [d for d in N_FILES if d not in groups]
    if missing_ds:
        print(f"  NOTE: not in the record: {', '.join(missing_ds)}\n")

    arg = sys.argv[1].lower() if len(sys.argv) > 1 else None

    if arg is None:
        show_status(groups)
        return

    if arg == "all":
        parts = list(PARTS)
    elif arg.isdigit() and int(arg) in PARTS:
        parts = [int(arg)]
    else:
        print(f"  Unknown argument: {arg}")
        print(f"  Use a part number ({min(PARTS)}-{max(PARTS)}) or 'all'.")
        return

    for p in parts:
        ok = run_part(p, groups)
        if not ok and arg == "all":
            print("\n  Stopping. Fix the connection and run this part again.")
            break

    print("\n")
    show_status(groups)


if __name__ == "__main__":
    main()
