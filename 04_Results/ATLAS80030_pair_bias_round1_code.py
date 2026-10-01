"""ATLAS 80030 - build the paired nominal <-> systematic analysis frame.

This is the first step of the rebuilt analysis. Everything downstream -
SHAP-delta, split conformal coverage, and the pT-binned background rejection
metric - compares a systematic jet against THE SAME jet in the nominal sample.
That comparison is only meaningful if the two sides are genuinely the same
jet, so the join is the load-bearing step of the whole paper and this script
does nothing except perform it and prove that it worked.

WHAT IT READS
    E:\\ATLAS_80030_Data\\hl_recomputed\\<dataset>\\*.npz
    The recomputed feature cache. It already carries everything needed: the
    15 high-level features, the four pairing-key columns, fjet_pt and fjet_m
    for the pT-binned metric, n_constit, and the code fingerprint. The source
    .h5 files are NOT needed for the join itself; they are opened only for the
    optional sentinel cross-check described below.

WHAT IT WRITES
    E:\\ATLAS_80030_Data\\paired\\<dataset>_paired.npz
    One file per systematic, holding the matched jets with the nominal side
    under nom_* and the systematic side under sys_*.
    Plus a round-numbered log in Code files\\analysis_results\\, following the
    same never-overwrite convention as recompute_hl.py.

THE KEY
    (EventInfo_mcEventNumber, fjet_eta, fjet_phi, labels)
    fjet_eta and fjet_phi are compared as their exact float32 BIT PATTERNS,
    never as floats. A float equality test on a 32-bit coordinate is a
    tolerance question with no correct answer; a bit test is exact. The
    systematic variation lives in fjet_clus_*, so the jet-level axis is
    bit-identical on both sides and the bit test is the right one. Negative
    zero is normalised to positive zero first, because -0.0 == 0.0 as a
    number but has a different bit pattern.

WHAT "SUCCESS" LOOKS LIKE
    Every systematic should pair at essentially 100%. The one known exception
    is cer, measured at 99.904% on cer_000: about 95 jets per 100k exist in
    cer with no nominal counterpart. ATLAS marks exactly those jets by writing
    the sentinel -999 into the stored fjet_L3 branch of the source .h5, and
    they are all low-multiplicity (n_constit 3-20, median 8). CHECK_SENTINELS
    verifies that the jets this script fails to match are that same set. If a
    dataset pairs materially below 99.9%, or if the unmatched jets are NOT the
    sentinel jets, stop and investigate before using the output.

RUNNING IT
    Edit PAIR_SETS to the dataset(s) for this sitting and run:
        python "H:\\HEP\\Research Papers\\ATLAS 80030\\Code files\\pair_features.py"
    Interrupting is safe. An existing output is skipped only when its
    pair fingerprint matches; a mismatched or unstamped one is rebuilt with
    the reason printed. Editing PAIR_SETS does not invalidate finished work.

COST
    Peak RAM is about 2 GB: the nominal key index for 15.3M jets is roughly
    550 MB and one systematic's two feature blocks are roughly 600 MB.
    Output is roughly 0.5 GB per systematic, so about 4.5 GB for all nine.
    Runtime is a few minutes per systematic, dominated by reading the nominal
    cache once per dataset.
"""

import hashlib
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------- config ---
STORE = Path(r"E:\ATLAS_80030_Data")
HERE = Path(__file__).resolve().parent

# Which systematics to pair this sitting, in order. Empty list = all nine.
# Full order: bias, cer, cpos, esdown, esup, teg, tej, tfj, tfl
PAIR_SETS = ["bias"]

# The nominal pool the systematics are matched against. Both are searched;
# the log reports how each systematic's matches split between them.
NOMINAL_SETS = ["test_nominal", "train_nominal"]

# Open the source .h5 to confirm that the jets which fail to match are the
# ones ATLAS itself marked with a -999 sentinel. Costs one extra branch read
# per file. Leave it on; it is the check that distinguishes "expected data
# feature" from "broken join".
CHECK_SENTINELS = True

# Pairing floor below which a dataset is treated as a problem, in percent.
# cer is the one dataset known to contain jets with no nominal counterpart -
# about 95 per 100k, every one of them flagged by ATLAS with a -999 sentinel,
# measured at 99.898% on a three-file sample. It gets its own floor so the
# alarm stays meaningful. Every other systematic should pair essentially
# completely, and anything below its floor is a real finding, not a nuisance.
FLOOR = {"cer": 99.80}
DEFAULT_FLOOR = 99.95

CACHE = STORE / "hl_recomputed"
DATA = STORE / "data"
OUT = STORE / "paired"
RESULTS = HERE / "analysis_results"

HL = ["fjet_Tau1_wta", "fjet_Tau2_wta", "fjet_Tau3_wta", "fjet_Tau4_wta",
      "fjet_Split12", "fjet_Split23",
      "fjet_ECF1", "fjet_ECF2", "fjet_ECF3",
      "fjet_C2", "fjet_D2", "fjet_L2", "fjet_L3",
      "fjet_Qw", "fjet_ThrustMaj"]

# Columns carried from the nominal side only: the key, plus what the
# pT-binned background-rejection metric needs. fjet_pt/fjet_m are jet-level
# and bit-identical on both sides, so one copy is enough.
KEEP = ["EventInfo_mcEventNumber", "fjet_eta", "fjet_phi", "fjet_pt",
        "fjet_m", "labels"]

KEY_DT = np.dtype([("e", "<i8"), ("y", "<u4"), ("p", "<u4"), ("l", "<i2")])
VOID_DT = np.dtype((np.void, KEY_DT.itemsize))   # 18 raw bytes, compared whole
SENTINEL = -900.0          # stored HL values at or below this are ATLAS flags

line = "=" * 78
sub = "-" * 78


# --- fingerprint: pair start ---
def _pair_fingerprint():
    """Hash the part of this file that can change a paired output.

    Same idea as recompute_hl.py's code fingerprint, and deliberately blind to
    the same kinds of edit. PAIR_SETS, CHECK_SENTINELS, comments and blank
    lines cannot change a stored number, so editing them between sittings must
    not invalidate finished work. The key definition and the join can, so they
    are inside the hash.
    """
    src = Path(__file__).read_text(encoding="utf-8")
    lo = src.index("# --- fingerprint: pair start ---")
    body = src[lo:]
    keep = [ln for ln in body.splitlines()
            if ln.strip() and not ln.strip().startswith("#")]
    keep.append(f"HL={HL}")
    keep.append(f"KEEP={KEEP}")
    keep.append(f"KEY_DT={KEY_DT}")
    return hashlib.md5("\n".join(keep).encode()).hexdigest()


def build_key(ev, eta, phi, lab):
    """The pairing key, as exact bits.

    eta and phi go in as their float32 bit patterns. Negative zero is folded
    onto positive zero first: -0.0 == 0.0 as a number, so two jets that are
    the same jet must not be split apart by a sign bit that carries no
    information.
    """
    eta = np.asarray(eta, dtype=np.float32).copy()
    phi = np.asarray(phi, dtype=np.float32).copy()
    eta[eta == 0.0] = 0.0
    phi[phi == 0.0] = 0.0
    if not (np.isfinite(eta).all() and np.isfinite(phi).all()):
        raise SystemExit("non-finite fjet_eta/fjet_phi: the key is unusable")
    k = np.empty(eta.size, KEY_DT)
    k["e"] = np.asarray(ev, dtype=np.int64)
    k["y"] = eta.view(np.uint32)
    k["p"] = phi.view(np.uint32)
    k["l"] = np.asarray(lab, dtype=np.int16)
    return k


def files_of(ds):
    return sorted((CACHE / ds).glob("*.npz"))


def load_cols(path, cols):
    with np.load(path, allow_pickle=False) as z:
        return {c: z[c] for c in cols}, str(z["code_fp"]) if "code_fp" in z.files else ""


def build_nominal_index():
    """Read the key columns of every nominal .npz.

    Only the key is held in memory. The features are fetched later, and only
    for the rows that actually matched, so nothing is read that is not used.
    Returns the keys in global row order plus the (file, row) address of every
    global row; the ordering work happens once per systematic inside match().
    """
    keys, gfile, grow, paths, owner = [], [], [], [], []
    for ds in NOMINAL_SETS:
        fs = files_of(ds)
        if not fs:
            raise SystemExit(f"no .npz found for nominal set {ds} in {CACHE / ds}")
        for f in fs:
            d, _ = load_cols(f, ["EventInfo_mcEventNumber", "fjet_eta",
                                 "fjet_phi", "labels"])
            n = d["labels"].size
            keys.append(build_key(d["EventInfo_mcEventNumber"], d["fjet_eta"],
                                  d["fjet_phi"], d["labels"]))
            gfile.append(np.full(n, len(paths), dtype=np.int32))
            grow.append(np.arange(n, dtype=np.int32))
            paths.append(f)
            owner.append(ds)
        print(f"  {ds:<14} {len(fs):>3} files")
    k = np.concatenate(keys)
    print(f"  nominal pool: {k.size:,} jets")
    return k, np.concatenate(gfile), np.concatenate(grow), paths, owner


def match(nomk, sysk):
    """Exact equality join. Returns, for each systematic jet, the global
    nominal row it pairs with, or -1 if it has no partner.

    The two key sets are grouped in a single pass through a void view, so the
    comparison is a memcmp over all eighteen bytes: two keys are the same only
    if every byte agrees, which is exactly the intended test and about five
    times faster than sorting the structured dtype field by field. Nothing
    here depends on the *order* numpy happens to put the bytes in - only on
    equal keys landing in the same group.

    The nominal key must be unique or the join is ambiguous. recompute_hl.py's
    brief records it as 100% unique; that is a claim about the data, so it is
    re-checked on every run rather than assumed.
    """
    n = nomk.size
    allk = np.concatenate([nomk, sysk]).view(VOID_DT)
    u, inv = np.unique(allk, return_inverse=True)
    inv = np.asarray(inv).ravel().astype(np.int64)
    del allk
    dup = int((np.bincount(inv[:n], minlength=u.size) > 1).sum())
    if dup:
        raise SystemExit(f"nominal key is not unique ({dup} repeated keys) - "
                         "the join would be ambiguous; stop here")
    lut = np.full(u.size, -1, dtype=np.int64)
    lut[inv[:n]] = np.arange(n, dtype=np.int64)
    return lut[inv[n:]]


def gather_nominal(paths, gidx, gfile, grow, cols):
    """Fetch `cols` for a set of global nominal row ids.

    Grouped by file so every nominal .npz is opened at most once, and the
    result is returned in the order gidx was given.
    """
    out = {c: None for c in cols}
    fidx = gfile[gidx]
    ridx = grow[gidx]
    for fi in np.unique(fidx):
        sel = np.where(fidx == fi)[0]
        with np.load(paths[fi], allow_pickle=False) as z:
            for c in cols:
                v = z[c]
                if out[c] is None:
                    out[c] = np.empty(gidx.size, dtype=v.dtype)
                out[c][sel] = v[ridx[sel]]
    return out


def sentinel_rows(ds):
    """Row indices, per systematic file, where ATLAS stored a -999 in fjet_L3.

    These are the jets ATLAS could not fill, and they are the jets expected to
    have no nominal counterpart. Read from the source .h5 because the sentinel
    lives in the stored branch, which the recomputation deliberately discards.
    """
    import h5py
    out = {}
    for f in sorted((DATA / ds).glob("*.h5")):
        with h5py.File(f, "r") as fh:
            if "fjet_L3" not in fh:
                return {}
            v = fh["fjet_L3"][:].astype(np.float64)
        out[f.stem] = v <= SENTINEL
    return out


class Tee:
    """Mirror the console to a round-numbered log; never overwrite.

    Same convention as recompute_hl.py: the round number is one past the
    highest already on disk for this tag, and a verbatim copy of the script is
    written beside the log so a result stays readable years later.
    """

    def __init__(self, tag):
        RESULTS.mkdir(parents=True, exist_ok=True)
        used = [0]
        for p in RESULTS.glob(f"ATLAS80030_{tag}_round*"):
            m = re.search(r"round(\d+)", p.name)
            if m:
                used.append(int(m.group(1)))
        self.round = max(used)
        while True:
            self.round += 1
            stem = RESULTS / f"ATLAS80030_{tag}_round{self.round}"
            self.path = stem.with_suffix(".txt")
            self.code = Path(f"{stem}_code.py")
            if not self.path.exists() and not self.code.exists():
                break
        self.code.write_bytes(Path(__file__).read_bytes())
        self.fh = self.path.open("w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.fh.write(s)

    def flush(self):
        self.out.flush()
        self.fh.flush()


def pair_one(ds, nomk, gfile, grow, paths, owner, pair_fp):
    """Join one systematic dataset to the nominal pool and write the result."""
    dst = OUT / f"{ds}_paired.npz"
    if dst.exists():
        try:
            with np.load(dst, allow_pickle=False) as z:
                prev = str(z["pair_fp"]) if "pair_fp" in z.files else ""
        except Exception:
            prev = ""
        if prev == pair_fp:
            print(f"\n{ds}: up to date, skipping")
            return
        print(f"\n{ds}: built by pair code {prev[:8] or '(unstamped)'}, "
              f"this is {pair_fp[:8]} - rebuilding")

    fs = files_of(ds)
    if not fs:
        print(f"\n{ds}: no .npz found in {CACHE / ds} - skipping")
        return
    print(f"\n{line}\n{ds}: {len(fs)} files -> {dst}\n{line}")
    t0 = time.time()

    sent = sentinel_rows(ds) if CHECK_SENTINELS else {}
    src_fps = set()

    # Read the systematic side once: keys, the fifteen features, and
    # n_constit. Held in memory (about 400 MB for 5M jets) so the join can be
    # done in one pass instead of once per file.
    ks_, feats, sys_nc, stems, sizes = [], {c: [] for c in HL}, [], [], []
    for f in fs:
        with np.load(f, allow_pickle=False) as z:
            ks_.append(build_key(z["EventInfo_mcEventNumber"], z["fjet_eta"],
                                 z["fjet_phi"], z["labels"]))
            for c in HL:
                feats[c].append(z[c])
            sys_nc.append(z["n_constit"])
            src_fps.add(str(z["code_fp"]))
            stems.append(f.stem)
            sizes.append(ks_[-1].size)

    if len(src_fps) != 1:
        raise SystemExit(f"{ds}: inputs carry {len(src_fps)} code fingerprints "
                         f"{sorted(src_fps)} - the cache is not homogeneous")
    sysk = np.concatenate(ks_)
    n_seen = sysk.size
    g = match(nomk, sysk)
    hit = g >= 0
    gidx = g[hit]
    matched = gidx.size

    # Which unmatched jets carry ATLAS's own -999 sentinel in the stored
    # branch? That is the check that separates "expected data feature" from
    # "broken join".
    un_nc = np.concatenate(sys_nc)[~hit]
    un_sent = 0
    if sent:
        off = 0
        for stem, sz in zip(stems, sizes):
            m = ~hit[off:off + sz]
            un_sent += int(sent.get(stem, np.zeros(sz, bool))[m].sum())
            off += sz

    # Distinct systematic keys must map to distinct nominal rows. If this ever
    # trips, one nominal jet is being claimed by two systematic jets and every
    # paired statistic downstream would be wrong.
    if np.unique(gidx).size != matched:
        raise SystemExit(f"{ds}: {matched - np.unique(gidx).size} nominal jets "
                         "matched more than once - the join is not one-to-one")

    nom = gather_nominal(paths, gidx, gfile, grow, HL + KEEP + ["n_constit"])
    res = {f"nom_{c}": nom[c] for c in HL}
    res.update({f"sys_{c}": np.concatenate(feats[c])[hit] for c in HL})
    res.update({c: nom[c] for c in KEEP})
    res["nom_n_constit"] = nom["n_constit"]
    res["sys_n_constit"] = np.concatenate(sys_nc)[hit]

    # Drop any jet that is non-finite on either side. Expected to be zero:
    # cer's eleven NaN L3 jets are all in the unmatched set already.
    good = np.ones(matched, bool)
    for c in HL:
        good &= np.isfinite(res[f"nom_{c}"]) & np.isfinite(res[f"sys_{c}"])
    dropped = int((~good).sum())
    if dropped:
        for k in list(res):
            res[k] = res[k][good]

    split = {ds_: int((np.isin(gfile[gidx], [i for i, o in enumerate(owner)
                                             if o == ds_])).sum())
             for ds_ in NOMINAL_SETS}

    print(f"  jets in {ds:<14} {n_seen:>10,}")
    print(f"  matched to nominal   {matched:>10,}   = {100.0 * matched / n_seen:7.3f}%")
    print(f"  unmatched            {n_seen - matched:>10,}   "
          f"= {100.0 * (n_seen - matched) / n_seen:7.3f}%")
    if un_nc.size:
        u = un_nc
        print(f"    their n_constit    min={u.min()} median={int(np.median(u))} max={u.max()}")
        if CHECK_SENTINELS:
            pct = 100.0 * un_sent / u.size if u.size else 0.0
            print(f"    carrying ATLAS -999 sentinel: {un_sent}/{u.size} = {pct:.1f}%")
            if pct < 90.0:
                print("    *** LOOK AT THIS: the unmatched jets are NOT the "
                      "sentinel jets - investigate before using this output ***")
    for k_, v_ in split.items():
        if v_:
            print(f"  from {k_:<14} {v_:>10,}")
    print(f"  dropped non-finite   {dropped:>10,}")
    print(f"  written              {res['labels'].size:>10,} paired jets")
    print(f"  signal fraction      {100.0 * float(res['labels'].sum()) / res['labels'].size:.2f}%")

    floor = FLOOR.get(ds, DEFAULT_FLOOR)
    rate = 100.0 * matched / n_seen
    print(f"  expected floor       {floor:>10.2f}%   -> "
          f"{'OK' if rate >= floor else 'BELOW FLOOR'}")
    if rate < floor:
        print(f"  *** LOOK AT THIS: match rate {rate:.3f}% is below this "
              f"dataset's floor of {floor:.2f}% - stop and investigate "
              "before using this output ***")

    OUT.mkdir(parents=True, exist_ok=True)
    res["pair_fp"] = np.array(pair_fp)
    res["src_code_fp"] = np.array(sorted(src_fps)[0])
    res["built"] = np.array(datetime.now().isoformat(timespec="seconds"))
    res["n_input_jets"] = np.array(n_seen)
    res["n_unmatched"] = np.array(n_seen - matched)
    res["n_dropped_nonfinite"] = np.array(dropped)
    np.savez_compressed(dst, **res)
    print(f"  {time.time() - t0:.0f}s -> {dst}")


def main():
    if not CACHE.is_dir():
        raise SystemExit(f"feature cache not found:\n  {CACHE}")
    todo = PAIR_SETS if PAIR_SETS else [
        p.name for p in sorted(CACHE.iterdir())
        if p.is_dir() and p.name not in NOMINAL_SETS]
    tag = "pair_" + ("_".join(todo) if len(todo) <= 3 else "all")
    tee = Tee(tag)
    sys.stdout = tee
    try:
        fp = _pair_fingerprint()
        print(f"# ATLAS 80030 - pair_features round {tee.round} - "
              f"{datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"# pair fingerprint {fp}")
        print(f"\n{line}\nbuilding nominal key index\n{line}")
        nomk, gfile, grow, paths, owner = build_nominal_index()
        for ds in todo:
            pair_one(ds, nomk, gfile, grow, paths, owner, fp)
        print(f"\n{line}\ndone. paired frames in {OUT}\n{line}")
    finally:
        sys.stdout = tee.out
        tee.fh.close()
    print(f"log: {tee.path}")


if __name__ == "__main__":
    main()
