# validate_data5.py
#
# Answers one question: WHERE DOES THE DETECTOR VARIATION LIVE?
#
# validate_data4 found that jets matched on (event, eta, phi, label) were
# bit-identical between nominal and systematic files - all 15 high-level
# features and the four-vector. That cannot be right for esup, which is an
# energy-scale-up variation and must move pt and the energy correlators.
#
# So one of these is true:
#   (1) the high-level branches in the systematic files are stale copies of
#       nominal, and the variation only exists in the constituents
#       (fjet_clus_*); or
#   (2) the matched jets really are unvaried, and the global differences seen
#       earlier come from a different jet selection, not from any variation.
#
# Sections:
#   P  every file opens, and how many jets each holds
#   K  the real values of the HDF5 attributes
#   L  compare constituents on matched pairs
#   M  rebuild simple high-level quantities from constituents and compare
#   N  row-hash test: is a systematic file a resample of nominal?
#
# Memory: the full 104-file nominal set needs about 1.2 GB. Set
# NOM_MAX_FILES = 40 if that is too much; the answer still comes out.
#
# Run:  python "Code files\validate_data5.py"

from pathlib import Path

import h5py
import numpy as np

from atlas_config import DATA_ROOT

DATA = DATA_ROOT / "data"

NOM_MAX_FILES = None             # None = use every file
PROBE = ["esup", "teg"]          # one calorimeter systematic, one track systematic
MAX_PAIRS = 2000                 # reading constituents is expensive
SEED = 42

FEATURES = ["fjet_Tau1_wta", "fjet_Tau2_wta", "fjet_Tau3_wta", "fjet_Tau4_wta",
            "fjet_Split12", "fjet_Split23",
            "fjet_ECF1", "fjet_ECF2", "fjet_ECF3",
            "fjet_C2", "fjet_D2", "fjet_L2", "fjet_L3",
            "fjet_Qw", "fjet_ThrustMaj"]
KINEM = ["fjet_pt", "fjet_eta", "fjet_phi", "fjet_m"]
CLUS = ["fjet_clus_pt", "fjet_clus_E", "fjet_clus_eta", "fjet_clus_phi"]
EVENT = "EventInfo_mcEventNumber"

line = "=" * 78
sub = "-" * 78


def files_of(ds, cap=None):
    d = DATA / ds
    if not d.is_dir():
        return []
    fs = sorted(d.glob("*.h5"))
    return fs[:cap] if cap else fs


# ---------------------------------------------------------------- P
def part_p():
    """Does every file open, how many jets, and is any of them damaged?"""
    print(line)
    print("P  -  file health")
    print(line)

    ok = True
    for ds in ["test_nominal"] + PROBE:
        fs = files_of(ds)
        counts, broken = [], []
        for p in fs:
            try:
                with h5py.File(p, "r") as f:
                    counts.append(int(f["labels"].shape[0]))
            except Exception as err:
                broken.append((p.name, err.__class__.__name__))
        if not counts:
            print(f"  {ds:<14s} no files")
            continue
        c = np.array(counts)
        print(f"  {ds:<14s} {len(fs):>3d} files   total jets {c.sum():>12,}   "
              f"min {c.min():>7,}  max {c.max():>7,}")
        odd = [(fs[i].name, counts[i]) for i in range(len(counts))
               if counts[i] != c.max()]
        for name, n in odd:
            print(f"       short file : {name}  {n:,} jets "
                  f"(others {c.max():,}) - final chunk of the dataset, normal")
        if broken:
            ok = False
            print(f"       WOULD NOT OPEN:")
            for name, err in broken:
                print(f"         {name}  ({err})")
    if not ok:
        raise SystemExit("\n  Some files are damaged. Delete them and run "
                         "download_data.py again.")
    print()


# ---------------------------------------------------------------- loading
def load_hl(ds, cap=None):
    """High-level features, kinematics, and file/row bookkeeping.

    Counts the rows first and fills one preallocated array, which avoids the
    memory spike np.vstack would cause across 104 files."""
    paths = files_of(ds, cap)
    counts = []
    for p in paths:
        with h5py.File(p, "r") as f:
            counts.append(int(f["labels"].shape[0]))
    N = sum(counts)

    X = np.empty((N, len(FEATURES)), np.float32)
    K = np.empty((N, len(KINEM)), np.float32)
    e = np.empty(N, np.int64)
    y = np.empty(N, np.int8)
    fid = np.empty(N, np.int16)
    row = np.empty(N, np.int32)

    off = 0
    for i, p in enumerate(paths):
        n = counts[i]
        with h5py.File(p, "r") as f:
            for j, c in enumerate(FEATURES):
                X[off:off + n, j] = f[c][:]
            for j, c in enumerate(KINEM):
                K[off:off + n, j] = f[c][:]
            e[off:off + n] = f[EVENT][:]
            y[off:off + n] = f["labels"][:]
        fid[off:off + n] = i
        row[off:off + n] = np.arange(n, dtype=np.int32)
        off += n

    return dict(X=X, K=K, e=e, y=y, fid=fid, row=row, paths=paths)


def read_clus(d, idx):
    """Read constituents for the selected jets, one file at a time.

    h5py fancy indexing needs strictly increasing row numbers, so the rows are
    sorted and the original position of each jet is returned alongside."""
    out = {c: [] for c in CLUS}
    order = []
    for i in np.unique(d["fid"][idx]):
        sel = idx[d["fid"][idx] == i]
        srt = np.argsort(d["row"][sel])
        rows = d["row"][sel][srt]
        with h5py.File(d["paths"][i], "r") as f:
            for c in CLUS:
                out[c].append(np.asarray(f[c][rows]))
        order.append(sel[srt])
    if not order:
        return None, None
    return {c: np.concatenate(v) for c, v in out.items()}, np.concatenate(order)


def realign(order, targets):
    """Put the rows read by read_clus back into pair order."""
    lut = {int(v): i for i, v in enumerate(order)}
    return np.array([lut[int(t)] for t in targets], dtype=np.int64)


def key4(d):
    """Jet key: event number, label, eta, phi.

    The label has to be part of the key because mcEventNumber is only unique
    within one MC sample - the Z' signal and the dijet background are separate
    samples - and no DSID or channel branch exists in these files."""
    out = np.empty(len(d["e"]), dtype=[("e", "<i8"), ("y", "i1"),
                                       ("eta", "<f4"), ("phi", "<f4")])
    out["e"] = d["e"]
    out["y"] = d["y"]
    out["eta"] = d["K"][:, 1]
    out["phi"] = d["K"][:, 2]
    return out


def unique_only(key):
    """Keep only jets whose key occurs once in their own dataset. The rest are
    ambiguous and could be matched to the wrong partner."""
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return cnt[inv] == 1


# ---------------------------------------------------------------- K
def part_k():
    print(line)
    print("K  -  actual attribute values")
    print(line)
    for ds in ["test_nominal"] + PROBE:
        fs = files_of(ds)
        if not fs:
            continue
        print(f"\n  {ds}  ({fs[0].name})")
        with h5py.File(fs[0], "r") as f:
            for k in sorted(f.attrs):
                v = f.attrs[k]
                if isinstance(v, np.ndarray):
                    v = [x.decode() if isinstance(x, bytes) else x
                         for x in v.tolist()]
                elif isinstance(v, bytes):
                    v = v.decode()
                s = str(v)
                print(f"      {k:<20s} {s if len(s) < 300 else s[:300] + ' ...'}")
            print(f"      {'clus shape':<20s} {f[CLUS[0]].shape}")


# ---------------------------------------------------------------- L + M
def part_lm(nom, kn, un):
    print("\n" + line)
    print("L  -  compare CONSTITUENTS on matched pairs")
    print(line)

    rng = np.random.default_rng(SEED)

    for ds in PROBE:
        s = load_hl(ds)
        ks = key4(s)
        us = unique_only(ks)

        common, i_n, i_s = np.intersect1d(kn[un], ks[us], return_indices=True)
        gi_n = np.flatnonzero(un)[i_n]
        gi_s = np.flatnonzero(us)[i_s]

        print(f"\n  {ds}: {len(s['paths'])} files, {len(s['e']):,} jets")
        if len(common) == 0:
            print("      no pairs found.")
            continue
        print(f"      pairs found : {len(common):,}  "
              f"({100.0 * len(common) / len(s['e']):.1f}% of systematic jets "
              f"have a nominal partner)")

        # Random sample. The output of intersect1d is sorted by key, so taking
        # the first N would only cover the smallest event numbers.
        take = min(MAX_PAIRS, len(common))
        pick = np.sort(rng.choice(len(common), take, replace=False))
        gi_n, gi_s = gi_n[pick], gi_s[pick]

        hl_same = int((nom["X"][gi_n] == s["X"][gi_s]).all(axis=1).sum())
        pt_same = int((nom["K"][gi_n, 0] == s["K"][gi_s, 0]).sum())
        m_same = int((nom["K"][gi_n, 3] == s["K"][gi_s, 3]).sum())
        print(f"      tested on {take:,} random pairs:")
        print(f"        15 high-level features identical : {hl_same:,} / {take:,}")
        print(f"        fjet_pt identical                : {pt_same:,} / {take:,}")
        print(f"        fjet_m  identical                : {m_same:,} / {take:,}")

        cn, on = read_clus(nom, gi_n)
        cs, os_ = read_clus(s, gi_s)
        if cn is None or cs is None:
            print("        constituents could not be read.")
            continue
        an = realign(on, gi_n)
        a_s = realign(os_, gi_s)

        print(f"        {'branch':<18s} {'identical':>12s} {'max|diff|':>14s}"
              f" {'mean|diff|':>14s}")
        for c in CLUS:
            A = cn[c][an].astype(np.float64)
            B = cs[c][a_s].astype(np.float64)
            same = int((A == B).all(axis=1).sum())
            d = np.abs(A - B)
            print(f"        {c:<18s} {same:>6,}/{take:<5,} {d.max():>14.6g}"
                  f" {d.mean():>14.6g}")

        print("\n" + sub)
        print(f"M  -  {ds}: rebuild high-level quantities from constituents")
        print(sub)
        A_pt = cn["fjet_clus_pt"][an].astype(np.float64)
        B_pt = cs["fjet_clus_pt"][a_s].astype(np.float64)
        A_E = cn["fjet_clus_E"][an].astype(np.float64)
        B_E = cs["fjet_clus_E"][a_s].astype(np.float64)
        for name, A, B in (("sum clus_pt", A_pt, B_pt),
                           ("sum clus_E", A_E, B_E),
                           ("n constits>0", (A_pt > 0).astype(np.float64),
                            (B_pt > 0).astype(np.float64))):
            a = A.sum(axis=1)
            b = B.sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.where(a != 0, (b - a) / np.abs(a) * 100.0, np.nan)
            print(f"      {name:<14s} mean shift {np.nanmean(rel):>10.4f} %   "
                  f"changed jets {int((a != b).sum()):>6,}/{take:,}")


# ---------------------------------------------------------------- N
def part_n(nom):
    print("\n" + line)
    print("N  -  is a systematic file a resample of nominal?")
    print(line)
    print("  Ignore the event number. Hash the 15 high-level values of each")
    print("  row and look for exact matches. If most systematic jets appear")
    print("  verbatim in nominal, the high-level branches carry no variation.\n")

    try:
        hn = row_hash(nom["X"])
        for ds in PROBE:
            s = load_hl(ds)
            hs = row_hash(s["X"])
            hit = int(np.isin(hs, hn).sum())
            print(f"  {ds:<8s} {len(hs):,} jets - found verbatim in nominal: "
                  f"{hit:,}  ({100.0 * hit / len(hs):.2f}%)")
    except MemoryError:
        print("  Ran out of memory. Skip section N; section L already answers")
        print("  the question. To retry, set NOM_MAX_FILES = 40.")
        return


def row_hash(X):
    A = np.ascontiguousarray(X, dtype=np.float32)
    dt = [(f"f{i}", "<f4") for i in range(A.shape[1])]
    return A.view(dt).ravel()


def main():
    if not DATA.is_dir():
        raise SystemExit(f"data folder not found: {DATA}")

    part_p()
    part_k()

    n_files = len(files_of("test_nominal", NOM_MAX_FILES))
    print(f"\n  reading test_nominal ({n_files} files) ...")
    nom = load_hl("test_nominal", NOM_MAX_FILES)
    print(f"  {len(nom['e']):,} jets, {len(np.unique(nom['e'])):,} unique events")

    kn = key4(nom)
    un = unique_only(kn)
    print(f"  unambiguous keys: {int(un.sum()):,} ({100.0 * un.mean():.1f}%)")

    part_lm(nom, kn, un)
    part_n(nom)

    print("\n" + line)
    print("Send the full output.")
    print(line)


if __name__ == "__main__":
    main()
