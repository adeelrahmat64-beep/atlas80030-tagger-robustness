"""Closure of the six modelling datasets against recompute_hl.py.

READ-ONLY on all existing data. Imports recompute_hl as a module and calls its
own closure() and verdict() -- no formula is re-implemented here. Logs go to a
NEW subfolder analysis_results/modelling_closure/, round-numbered by the
project's own Tee, so nothing existing is touched.
"""
import sys, io, json, time, warnings
from pathlib import Path
import numpy as np

CODE = Path.home() / "mnt" / "ATLAS 80030" / "Code files"
sys.path.insert(0, str(CODE))
import recompute_hl as rh

# The module's STORE/DATA are Windows paths; remap to the Linux mounts.
rh.DATA    = Path.home() / "mnt" / "ATLAS_80030_Data" / "data"
rh.RESULTS = CODE / "analysis_results"
rh.N_FILES = 1
rh.N_JETS  = 20000          # jets per dataset, as requested
# N_TRIPLET, MAX_CONSTITS, USE_CLUS_E, SEED left at production values.

DATASETS = sys.argv[1:] or ["ttbar_pythia"]
OUTDIR   = Path.home() / "verify" / "logs"   # staged; copied to analysis_results at the end

# The 15 production definitions, exactly as produce() stores them.
WANT = {
    "Tau1_wta /sumpt": "Tau1", "Tau2_wta /sumpt": "Tau2",
    "Tau3_wta /sumpt": "Tau3", "Tau4_wta /sumpt": "Tau4",
    "Split12": "Split12", "Split23": "Split23",
    "ECF1": "ECF1", "ECF2": "ECF2", "ECF3": "ECF3",
    "C2": "C2", "D2": "D2", "Qw": "Qw",
    "ATLAS thrust_major": "ThrustMaj",
    "L2 = 3e3/(1e2_b2)^1.5": "L2",
    "L3 = 1e3/(3e3)^(1/3)": "L3",
}

# --- memory only: this VM has 3 GB, ecf123's default block=2000 allocates
# ~2 GB of (block x 200 x 200) temporaries and gets OOM-killed. `block` is pure
# chunking -- jets are sorted by nc, each block trimmed to its widest jet, and
# padded entries are exact zeros -- so the per-jet result is unchanged.
_orig_ecf123 = rh.ecf123
def _ecf123_lowmem(pt, y, phi, nc, beta=1.0, block=2000):
    return _orig_ecf123(pt, y, phi, nc, beta=beta, block=250)
rh.ecf123 = _ecf123_lowmem

_orig_premise = rh.premise
def _premise_spy(hl, nc):
    current["nc"] = nc
    return _orig_premise(hl, nc)
rh.premise = _premise_spy

_orig_verdict = rh.verdict
current = {"ds": None}
records = []

def spy(name, stored, calc):
    s = _orig_verdict(name, stored, calc)
    key = name.strip()
    if key in WANT:
        ok = np.isfinite(stored) & np.isfinite(calc) & (np.abs(calc) > 0)
        ok &= (stored > -900) & (calc > -900)
        n = int(ok.sum())
        rec = {"dataset": current["ds"], "feature": WANT[key], "n": n,
               "tag": s.split()[-1]}
        if n >= 20:
            a, b = stored[ok], calc[ok]
            with np.errstate(all="ignore"):
                rec["r"]   = float(np.corrcoef(a, b)[0, 1])
                ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
                rec["rho"] = float(np.corrcoef(ra, rb)[0, 1])
                q = a / b
                med = float(np.median(q))
                lo, hi = np.percentile(q, [5, 95])
                rec["ratio_med"]    = med
                rec["spread_pct"]   = float((hi - lo) / abs(med) * 100) if med else float("inf")
                rec["max_abs_dev"]  = float(np.max(np.abs(q / med - 1))) if med else float("inf")
                d = np.abs(q / med - 1.0)
                rec["p99"]   = float(np.percentile(d, 99))
                rec["p999"]  = float(np.percentile(d, 99.9))
                for thr, key in ((1e-3,"f_gt_1e-3"), (1e-2,"f_gt_1e-2"), (1e-1,"f_gt_1e-1")):
                    rec[key] = float((d > thr).mean())
                nc = current.get("nc")
                if nc is not None and len(nc) == len(stored):
                    ncok = nc[ok]
                    out = d > 1e-2
                    rec["nc_med_all"] = float(np.median(ncok))
                    rec["nc_med_out"] = float(np.median(ncok[out])) if out.any() else None
                    rec["n_out_1e-2"] = int(out.sum())
        records.append(rec)
    return s

rh.verdict = spy

warnings.filterwarnings("ignore")
t_all = time.time()
for ds in DATASETS:
    current["ds"] = ds
    rh.DATASET = f"modelling/{ds}"
    tee = rh.Tee(f"closure_modelling_{ds}", OUTDIR)
    real = sys.stdout

    class Dual:
        def write(self, s):
            tee.write(s)
        def flush(self):
            tee.flush()

    sys.stdout = Dual()
    t0 = time.time()
    try:
        print(f"# ATLAS 80030 - closure_modelling_{ds} round {tee.round}"
              f" - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"# code fingerprint {rh.CODE_FP}")
        print(f"# NOTE: modelling dataset, single class. Closure only asks how")
        print(f"#       closely the stored fjet_* branches agree with our")
        print(f"#       reconstruction. It does NOT validate them for use.")
        rh.closure()
    except Exception as e:
        print(f"!! FAILED: {type(e).__name__}: {e}")
        records.append({"dataset": ds, "feature": "*", "tag": "ERROR",
                        "error": f"{type(e).__name__}: {e}"})
    finally:
        sys.stdout = real
        tee.close()
    print(f"[{ds}] done in {time.time()-t0:6.1f}s -> {tee.path.name}", flush=True)
    jf = Path.home()/"verify"/"closure_modelling.json"
    old = json.loads(jf.read_text()) if jf.exists() else []
    old = [r for r in old if r["dataset"] != ds]
    jf.write_text(json.dumps(old + [r for r in records if r["dataset"] == ds], indent=1))

jf = Path.home()/"verify"/"closure_modelling.json"
old = json.loads(jf.read_text()) if jf.exists() else []
old = [r for r in old if r["dataset"] not in DATASETS]
jf.write_text(json.dumps(old + records, indent=1))
print(f"ALL DONE in {time.time()-t_all:.1f}s  ({len(records)} rows)")
