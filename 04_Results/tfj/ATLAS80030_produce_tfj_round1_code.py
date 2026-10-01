# recompute_hl.py
#
# WHY THIS EXISTS
# ---------------
# validate_data5.py established that in the systematic datasets the 15
# high-level fjet_* branches are stale copies of the nominal values. The
# detector variation lives only in fjet_clus_pt / _E / _eta / _phi. So every
# high-level feature the paper uses has to be recomputed from the constituents,
# with exactly the same code applied to the nominal side and to the systematic
# side.
#
# Before that recomputation can be trusted it has to pass a closure test:
# recompute the 15 quantities from NOMINAL constituents and check that they
# reproduce the 15 stored NOMINAL branches. If closure fails, our
# reimplementation is wrong - not the data.
#
# WHAT THIS SCRIPT DOES
# ---------------------
# MODE = "closure"  ->  runs on nominal jets, recomputes everything, and prints
#                       a per-feature verdict table (correlation, ratio,
#                       spread). The unit convention is still measured rather
#                       than assumed, because ATLAS never documented which unit
#                       the stored high-level branches are in - round 1 found
#                       1e-6 per power of pt, i.e. TeV.
# MODE = "produce"  ->  once closure is understood, writes the recomputed
#                       features for a dataset into a .npz cache.
#
# Every run of either mode writes its console output to
#   Code files\analysis_results\ATLAS80030_<mode>_round<N>.txt
# with N one past the highest already there. Nothing is ever overwritten, so
# the folder is the run history and any number quoted from an earlier round
# stays readable exactly as it was quoted.
#
# NO fastjet DEPENDENCY. The fastjet python bindings do not build on Windows,
# so everything here is numpy plus numba:
#   - one exclusive kt clustering routine supplies the winner-take-all axes for
#     Tau1-4, the exclusive splitting scales for Split12 / Split23, and the
#     three subjets for Qw;
#   - the energy correlation functions use the identity ECF3 = tr(W^3)/6 with
#     W_ij = sqrt(pt_i pt_j) * dR_ij^beta, which turns an O(n^3) triple sum into
#     two matrix products.
#
# VERIFICATION ALREADY DONE (26 Aug 2026, in a Linux container where the
# fastjet wheel installs, so fastjet 3.5.1 could be used as the referee):
#   - ECF1/2/3 from the trace identity vs an explicit brute-force triple sum:
#     max relative deviation 2.3e-14.
#   - Split12, Split23 and the three Qw subjets vs fastjet exclusive_dmerge /
#     exclusive_jets: agree to 3e-15 on 38 of 40 random jets. The two that
#     differ are jets where fastjet's beam distance breaks the constituents
#     into two inclusive jets; see the note in _kt_cluster.
#   - Winner-take-all axes vs fastjet WTA_pt_scheme at levels 2, 3 and 4:
#     7e-16. Level 1 differs only on those same two jets.
#   - Whole script run end to end on a synthetic HDF5 fixture whose high-level
#     branches were filled from those independent references, with three traps
#     planted in it: Tau stored normalised, Split12 stored in GeV, ThrustMaj
#     stored as noise. The report called all three correctly (EXACT on the
#     /sumpt row, SCALED at ratio 1e-3, FAIL) and reproduced the other twelve
#     features exactly, and the L2 scan recovered the fixture's formula.
# So the kernels are sound. What is NOT yet known is whether they match what
# ATLAS actually stored - that is the whole point of running this on real data.
#
# ROUND 3 CHANGES - the last three features now have SOURCED definitions
# ---------------------------------------------------------------------
# Rounds 1 and 2 searched for ThrustMaj, L2 and L3 by fitting. That is over:
# all three definitions now come from ATLAS's own documents, and this round
# only checks them.
#
#   ThrustMaj.  From the ATLAS implementation, JetSubStructureUtils Thrust.cxx:
#     the constituents are boosted to the jet centre-of-mass, and
#       thrust_major = max_n sum_i |p_i . n| / sum_i |p_i|
#       thrust_minor =       sum_i |p_i x n| / sum_i |p_i|
#     with useThreeD = true. So ATLAS's "major" IS the ordinary thrust of the
#     jet in its rest frame - it is NOT the transverse axis perpendicular to
#     the thrust axis, which is what round 1 computed and why round 1 came out
#     ANTI-correlated at rho = -0.33. Round 2 already measured the rest-frame
#     thrust at ratio 1.0000; the source now explains why, so this is a
#     definition, not a fitted match.
#
#   L2 and L3.  From ATL-PHYS-PUB-2021-028 section 5:
#       L2 = 3e3^(beta=1) / ( 1e2^(beta=2) )^(3/2)
#       L3 = 1e3^(beta=1) / ( 3e3^(beta=1) )^(1/3)
#     where ve_N^(beta) are the generalised energy correlation functions of
#     Moult, Necib and Thaler (arXiv:1609.07483), built from the energy
#     FRACTIONS z_i = pt_i / sum pt - so they are normalised by (sum pt)^N and
#     both L2 and L3 are dimensionless. Note beta = 2 in L2's denominator; it
#     is the only place in the whole feature set where beta is not 1.
#     One caveat, carried openly: the left subscript on L2's numerator is
#     fragile under PDF text extraction. 3e3 is used as the primary because it
#     is the reading that makes L2 angular-scale invariant, and 1e3 and 2e3 are
#     run alongside it as controls. Closure decides between them on the data,
#     which is a verification, not a fit - only one of the three can come back
#     at ratio 1.0000 with rho 1.0000.
#
# Round 3 kernels were tested in a container before being put here, against
# independent references written a different way: gen_ecf3 and gen_ecf2 against
# an itertools.combinations brute force (6e-15 and 2e-15), and the thrust axis
# against a 30,000-direction sphere scan. Two planted fixtures were used, one
# filled from the sourced formulas and one filled from the wrong variants; the
# report called both correctly, and a third fixture with L3 stored at 1e-6
# came back SCALED rather than EXACT as it should.
#
# ROUND 4 CHANGES - the Tau3 / Tau4 residual, diagnosed
# -----------------------------------------------------
# Round 3 closed 15 of 15 on real data. ThrustMaj, L2 and L3 all came back
# EXACT at ratio 1.0000, which confirms the sourced definitions. Tau1 and Tau2
# are EXACT; Tau3 and Tau4 sit at CLOSE, with stored/calc = 0.9967 and 0.9948
# and a spread growing 0.9% -> 5.6% across N. Our value is systematically the
# LARGER one, and the gap grows with N.
#
# Four candidate causes were tested in a container with fastjet 3.5.1 as the
# referee, configured the way ATLAS's own NSubjettinessTool.cxx configures
# fjcontrib - WTA_KT_Axes plus NormalizedCutoffMeasure(alpha, size, 1e6):
#
#   1. Our WTA axes are wrong.            FALSIFIED. They agree with fastjet's
#      WTA_KT_Axes to 5e-15 at every level 1-4, massive constituents included.
#   2. The distance measure is wrong.     CONFIRMED but small. See the note in
#      nsubjettiness: constituent pseudorapidity was being compared against an
#      axis rapidity. Worth 1e-5 on massive constituents and FLAT in N, so it
#      is a correctness fix, not the residual. Fixed in this round; the patched
#      kernel now reproduces fastjet exactly.
#   3. float32 storage of the constituents. FALSIFIED, and this was the brief's
#      leading explanation. Computing tau in double and again from float32
#      constituents gives ratio 1.00000 with 0.00% spread at every N.
#   4. ATLAS used minimised (OnePass) axes. FALSIFIED, twice over. ATLAS's
#      NSubjettinessTool.cxx uses plain WTA_KT_Axes - no one-pass. And running
#      the minimisation to convergence gives 0.961 0.985 0.992 0.993: it
#      IMPROVES with N and is ten times too large, while the data WORSENS with
#      N. Round 1 reached the same conclusion and was right.
#
# So the recomputation is correct against the authority. What remains is a
# difference between the constituents ATLAS fed its tool and the constituents
# this dataset stores. Section A3 of the closure report tests the one form of
# that which is checkable here - truncation of the constituent list - by
# breaking the residual down against the constituent count.
#
# This does not block anything. rho is 0.9999, so the feature ordering that
# SHAP reads is intact, and the paper compares nominal against systematic with
# the same code on both sides, where a reconstruction-path bias cancels. Tau3
# and Tau4 are usable; they need one sentence in the text.
#
# ROUND 5 CHANGES - the same coordinate correction, finished
# ----------------------------------------------------------
# Round 4 fixed the dR coordinate in nsubjettiness only, and all four Tau went
# to ratio 1.0000 with 0.0% spread. It left the same mismatch in the three
# correlator kernels, and round 4's own table shows the consequence: every
# feature with no (eta, phi) dR inside it sits at 0.0% spread - ECF1, Split12,
# Split23, Qw, ThrustMaj, all four Tau - while every feature that carries one
# does not: ECF2 0.4%, ECF3 1.2%, C2 0.5%, D2 0.5%, L2 0.5%, L3 0.5%. No
# exception either way, and the size tracks the number of angle factors.
#
# ecf123, gen_ecf3 and gen_ecf2 now take the rapidity from
# constituent_rapidity instead of the stored pseudorapidity. Nothing else in
# them changed. kt_observables, nsubjettiness and thrust_3d are untouched:
# their uses of eta build momentum components (pz = pt sinh eta), which are
# correct as they stand.
#
# The definition is taken from source, not inferred. fjcontrib's
# EnergyCorrelator.cc defines the pt_R measure as
#     energy(jet)           -> jet.perp()
#     angleSquared(j1, j2)  -> j1.squared_distance(j2)
# and multiply_angles picks the smallest n of the available angles, which is
# the convention gen_ecf3 already used. fastjet's squared_distance is
# (dy)^2 + (dphi)^2 in rapidity - measured directly in the container, a
# PseudoJet with rap 0.25541 and eta 0.29567 returns the rap value from
# delta_R. ATLAS's EnergyCorrelatorTool calls fjcontrib with the default
# pt_R, so this is what produced the stored branches.
#
# Verified before delivery, in a container, on soft massive constituents where
# rapidity and pseudorapidity genuinely differ (largest |y - eta| 0.078):
#   - constituent_rapidity vs fastjet PseudoJet::rap()          2.2e-15
#   - gen_ecf3 and gen_ecf2 vs an itertools brute force         5.4e-15, 2.3e-15
#   - ecf123's trace identity vs the same brute force           4.2e-15
#   - ECF2 and ECF3 vs fastjet's OWN delta_R, so the reference
#     never touches our rapidity formula                        1.1e-15, 2.9e-15
#   - CONTROL: with use_e = 0 the constituents are massless, rapidity IS
#     pseudorapidity, and all three kernels reproduce their pre-patch output
#     BIT-IDENTICALLY. That is what establishes the change is the coordinate
#     and nothing else.
#   - Two full end-to-end runs on synthetic HDF5 fixtures with massive
#     constituents: one whose stored branches were built in rapidity, one built
#     in pseudorapidity. The first returns 0.0% spread on ECF1/2/3, C2, D2, L2,
#     L3; the second returns 0.0% on ECF1 and 0.7 / 2.7 / 1.4 / 1.0 / 0.9 /
#     0.9% on the rest - the same fingerprint, in the same order.
#
# READ THE SPREAD COLUMN, NOT THE VERDICT. The EXACT threshold is 3%, so the
# pseudorapidity fixture above is still labelled EXACT at 2.7%. The label
# separates "right observable" from "wrong observable"; only the spread
# separates "reproduced" from "nearly reproduced".
#
# ROUND 6 CHANGES - production path only, no physics touched
# ----------------------------------------------------------
# Round 5 closed all 15 features at 0.0% spread and the definitions are frozen.
# Two things changed here, both inside produce(), neither affecting any
# observable definition, normalisation, constituent handling or closure logic.
#
#   1. ecf123 is no longer called in production. ECF1/2/3 are assembled from
#      the generalised correlators production already computes for L2 and L3:
#          ECF1 = sum pt
#          ECF2 = 1e2^(b=1) * (sum pt)^2
#          ECF3 = 3e3^(b=1) * (sum pt)^3
#      The only new work is one O(n^2) pass at beta = 1. Round 5's own
#      timings put ecf123 at 794 jets/s - 21.1 of the 30 hours the full pass
#      would have taken - against 13,221 jets/s for the O(n^2) kernel.
#      closure() still calls ecf123 on purpose: keeping the trace-identity
#      route alive there re-checks this identity against real data on every
#      closure run, for free.
#
#   2. Every .npz now carries code_fp, and an existing output is skipped only
#      if its stamp matches the running code. A mismatched or unstamped file
#      is RECOMPUTED with a printed reason, never silently kept. See
#      _code_fingerprint for what is and is not hashed.
#
# Tested before delivery, container, targeted - no closure or production run:
#   - identity ecf123 vs the production route, on soft massive constituents:
#     max relative deviation 4.4e-16 (ECF1), 1.1e-14 (ECF2), 5.9e-14 (ECF3),
#     6.6e-14 (C2), 7.1e-14 (D2). Not bit-identical, because summing sum pt
#     over a different array width reorders numpy's pairwise summation.
#   - REGRESSION: produce() run on the same fixture with the round-5 code and
#     with this code. All 27 arrays in the .npz are BIT-IDENTICAL. The 1e-14
#     difference is far below float32, which is what the cache stores, so the
#     delivered product does not change at all. Two keys are added, gen_1e2_b1
#     and code_fp; none are lost.
#   - stamp behaviour: same code skips; a tampered stamp and an unstamped
#     legacy file both recompute with the reason printed.
#   - fingerprint: unchanged by editing PRODUCE_SETS, MODE, STORE, N_JETS or a
#     top-of-file comment; changed by MAX_CONSTITS, USE_CLUS_E or a kernel
#     constant. All 8 cases as intended.
#   - closure(): report identical, row for row, before and after this edit.
#
# Run:  python "Code files\recompute_hl.py"

import hashlib
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from numba import njit, prange

# ----------------------------------------------------------------- config ---
STORE = Path(r"E:\ATLAS_80030_Data")
HERE = Path(__file__).resolve().parent

MODE = "produce"          # "closure" or "produce"
DATASET = "test_nominal"  # closure must run on a nominal dataset

# produce mode: which datasets to process, in order. Empty list = everything
# found under data\. Run it in sittings - the cache is resumable, finished
# files are skipped, and the log is named after whatever is in this list.
# Editing this line does NOT invalidate finished work; see _code_fingerprint.
#
# Full order, one or a few at a time:
#   bias, cer, cpos, esdown, esup, teg, tej, tfj, tfl,
#   train_nominal, test_nominal
PRODUCE_SETS = ["tfj"]
N_FILES = 1               # files to read
N_JETS = 20000            # jets to use for the closure test (None = all)
N_TRIPLET = 4000          # jets used for the expensive O(n^3) generalised ECFs
MAX_CONSTITS = 200
USE_CLUS_E = True         # True = build four-vectors from the stored
                          # fjet_clus_E; False = force massless constituents.
                          # If closure comes out MONOTONE instead of EXACT,
                          # flip this before touching anything else.
SEED = 42

DATA = STORE / "data"
RESULTS = HERE / "analysis_results"

HL = ["fjet_Tau1_wta", "fjet_Tau2_wta", "fjet_Tau3_wta", "fjet_Tau4_wta",
      "fjet_Split12", "fjet_Split23",
      "fjet_ECF1", "fjet_ECF2", "fjet_ECF3",
      "fjet_C2", "fjet_D2", "fjet_L2", "fjet_L3",
      "fjet_Qw", "fjet_ThrustMaj"]
CLUS = ["fjet_clus_pt", "fjet_clus_eta", "fjet_clus_phi", "fjet_clus_E"]

def _code_fingerprint():
    """md5 of this file with comment lines and blank lines removed.

    Stamped into every .npz produce() writes, and checked before an existing
    output is skipped. The point is narrow: this file has been revised five
    times, and a production pass takes hours and is meant to be run in
    sittings. Without a stamp, a run resumed after an edit would keep the old
    outputs for the files already done and write new ones for the rest - and
    because nominal and systematic datasets are processed in the same pass,
    the two sides of the comparison could end up computed by different code
    with nothing on screen to say so. Mismatched files are RECOMPUTED, never
    silently kept.

    What is hashed is exactly the code that decides the numbers in a .npz, and
    nothing else. Two regions of this file are marked off:
        # --- fingerprint: kernels begin/end ---   the kernels and load()
        # --- fingerprint: produce begin/end ---   produce() itself
    plus the three config values that reach a produced file by value.

    Everything outside those markers is excluded on purpose, because a stamp
    that fires on changes which cannot alter a single stored number is a stamp
    that will be ignored, or worse, worked around. Excluded: the config block,
    the Tee logging class, summarise(), main(), and the closure-side reporting
    - verdict(), premise(), fit_monomial(), closure(). None of them writes a
    .npz. PRODUCE_SETS in particular is meant to be edited between sittings,
    one dataset at a time; if that invalidated the cache the stamp would
    destroy the very workflow it exists to protect.

    MAX_CONSTITS, USE_CLUS_E and CLUS DO reach a produced file, so they are
    appended by value rather than by their line of source.

    Comment and blank lines are stripped, so the notes at the top of this file
    and the markers themselves cost nothing. Docstrings are not stripped -
    they are part of the parsed source and telling them apart from code
    reliably is not worth the machinery, so an edit to a kernel's docstring
    costs a recompute. Edit before starting a sitting, not during one.
    """
    src = Path(__file__).read_bytes().decode("utf-8", "replace")
    lines = []
    for name in ("kernels", "produce"):
        a = src.find(f"# --- fingerprint: {name} begin ---")
        b = src.find(f"# --- fingerprint: {name} end ---")
        if a < 0 or b <= a:                # markers lost: fall back to the
            return hashlib.md5(src.encode("utf-8")).hexdigest()   # whole file
        lines += src[a:b].splitlines()
    body = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    body.append(f"MAX_CONSTITS={MAX_CONSTITS} USE_CLUS_E={USE_CLUS_E} "
                f"CLUS={CLUS}")
    return hashlib.md5("\n".join(body).encode("utf-8")).hexdigest()


CODE_FP = _code_fingerprint()

line = "=" * 78
sub = "-" * 78


# ------------------------------------------------------------ result log ---
class Tee:
    """Write the console output of a run to analysis_results as well.

    Every run gets its own numbered file and nothing is ever overwritten. The
    round number is one past the highest already on disk for that tag, so the
    folder reads as a history: round 1, round 2, round 3 ... and a result that
    has already been quoted somewhere stays exactly as it was quoted.

    In produce mode the tag carries the dataset, so a run over bias writes
    ATLAS80030_produce_bias_round1.txt and the next one round2, kept separate
    from a run over esup. Closure keeps the plain "closure" tag it has used
    since round 2.

    Alongside each log a verbatim copy of this file is written as
    ..._code.py. The .npz cache carries a fingerprint of the code that made
    it, which tells you whether two files agree; the snapshot tells you what
    that code actually was. Together a result in this folder can be read years
    later without needing to reconstruct which revision produced it.
    """

    def __init__(self, tag, outdir=None):
        d = outdir or RESULTS
        d.mkdir(parents=True, exist_ok=True)
        used = [0]
        for p in d.glob(f"ATLAS80030_{tag}_round*"):
            m = re.search(r"round(\d+)", p.name)
            if m:
                used.append(int(m.group(1)))
        self.round = max(used)
        while True:                        # belt and braces: never overwrite
            self.round += 1
            stem = d / f"ATLAS80030_{tag}_round{self.round}"
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

    def close(self):
        self.fh.close()


# --- fingerprint: kernels begin ---
# ------------------------------------------------------------- kt kernels ---
@njit(cache=True)
def _dphi(a, b):
    d = a - b
    while d > np.pi:
        d -= 2.0 * np.pi
    while d < -np.pi:
        d += 2.0 * np.pi
    return d


@njit(cache=True)
def _rap(e, pz):
    """Rapidity. fastjet clusters in rapidity, not pseudorapidity. For the
    massless input constituents the two coincide, but every merged pseudojet
    carries mass, so the distinction matters from the first recombination on."""
    a = e + pz
    b = e - pz
    if a <= 0.0:
        a = 1e-300
    if b <= 0.0:
        b = 1e-300
    return 0.5 * np.log(a / b)


@njit(parallel=True, cache=True)
def constituent_rapidity(pt, eta, en, nc, use_e):
    """The angular coordinate every dR in this file has to be measured in.

    fjcontrib's EnergyCorrelator, which is what ATLAS's EnergyCorrelatorTool
    calls, defines its pt_R measure as (EnergyCorrelator.cc):
        energy(jet)              -> jet.perp()
        angleSquared(j1, j2)     -> j1.squared_distance(j2)
    and fastjet's squared_distance is (dy)^2 + (dphi)^2 with y the RAPIDITY,
    not the pseudorapidity. That is what makes the measure boost invariant
    along the beam, which is what the header advertises; a pseudorapidity
    version would not be. Verified in a container: for a PseudoJet with
    rap 0.25541 and eta 0.29567, delta_R returns the rap value.

    The stored dataset gives pseudorapidity, so the two differ for any
    constituent with mass. ATLAS's UFO inputs are tracks combined with
    clusters, and a track constituent at a few hundred MeV carrying the pion
    mass has m/pt of order one - which is exactly where y and eta part
    company. Round 4 measured what this is worth: switching nsubjettiness
    alone took all four Tau from ratio 0.9995 / 0.9984 / 0.9967 / 0.9948 with
    0.9-5.6% spread to ratio 1.0000 with 0.0% spread, on real data, with
    nothing else changed.

    use_e = 0 forces massless constituents, where rapidity IS pseudorapidity
    and this returns eta unchanged - the control that must reproduce the old
    numbers exactly.

    nsubjettiness computes the same quantity inline rather than taking it from
    here. That duplication is deliberate: that kernel is already validated
    against fastjet and against real data, and it is not being touched.
    """
    n_jets = pt.shape[0]
    out = np.zeros(pt.shape, np.float64)
    for k in prange(n_jets):
        n = nc[k]
        for i in range(n):
            if use_e == 0:
                out[k, i] = eta[k, i]
                continue
            p0 = pt[k, i] * np.cosh(eta[k, i])
            e0 = en[k, i] if en[k, i] >= p0 else p0
            out[k, i] = _rap(e0, pt[k, i] * np.sinh(eta[k, i]))
    return out


@njit(cache=True)
def _kt_cluster(px, py, pz, en, n, wta, d_out, ox, oy, oz, oe, cnt):
    """Exclusive kt clustering of one jet's constituents, in four-vectors.

    Runs the pp kt algorithm with R = 1 and NO beam distance, so the sequence
    always ends with a single pseudojet. That choice is deliberate. Verified
    against fastjet 3.5.1 on 40 random jets: 38 agree to machine precision
    (1e-15) on Split12, Split23 and the three Qw subjets; the only two that
    differ are jets where fastjet's beam distance splits the constituents into
    two inclusive jets, at which point exclusive_dmerge is measuring something
    else. Clustering everything into one jet is well defined, deterministic,
    and - the point that matters here - identical on the nominal and the
    systematic side, so any residual offset against ATLAS is a constant that
    cancels in the nominal-vs-systematic comparison. The closure test will show
    it as SCALED rather than EXACT.

    Two recombination schemes are needed:

      wta = 0  E-scheme: the merged pseudojet is the four-vector sum. This is
               what fastjet's exclusive_dmerge and exclusive_jets return, and
               it is the scheme behind Split12 / Split23 and the Qw subjets.
               Working in (pt, eta, phi) with a pt-weighted average is NOT the
               same thing and gives splitting scales ~30% off.
      wta = 1  winner-take-all: the merged pseudojet is massless with pt equal
               to the sum and the direction of the harder parent. This is what
               makes Tau_N a "wta" variable.

    Fills:
      d_out[k]   = the kt distance of the merge that takes k+2 pseudojets to
                   k+1, so d_out[0] is d12 and d_out[1] is d23.
      o*[m-1, :m] = the m surviving pseudojets, recorded for EVERY multiplicity
                   m from n_lvl down to 1, and cnt[m-1] = m for those that were
                   reached. tau_N needs the axes from the N-axis clustering, not
                   the first N axes of a 4-axis clustering - those are different
                   objects and using the wrong one leaves tau1..tau3 failing
                   closure while tau4 passes.
    """
    a_px = px[:n].copy()
    a_py = py[:n].copy()
    a_pz = pz[:n].copy()
    a_e = en[:n].copy()
    a_pt2 = np.empty(n, np.float64)
    a_y = np.empty(n, np.float64)
    a_phi = np.empty(n, np.float64)
    alive = np.ones(n, np.uint8)
    for i in range(n):
        a_pt2[i] = a_px[i] * a_px[i] + a_py[i] * a_py[i]
        a_y[i] = _rap(a_e[i], a_pz[i])
        a_phi[i] = np.arctan2(a_py[i], a_px[i])

    # Nearest-neighbour cache, the same idea as fastjet's N2Plain strategy.
    # Rescanning every pair at every merge would be O(n^3) per jet, which for
    # the 200-constituent jets in this dataset is about 4 million distance
    # evaluations each - unusable at the scale the paper needs. Caching each
    # pseudojet's nearest neighbour and repairing only what a merge disturbs
    # brings it back to roughly O(n^2).
    nn = np.full(n, -1, np.int64)
    dnn = np.full(n, 1e300, np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dy = a_y[i] - a_y[j]
            dp = _dphi(a_phi[i], a_phi[j])
            dr2 = dy * dy + dp * dp
            u = a_pt2[i]
            v = a_pt2[j]
            d = (u if u < v else v) * dr2
            if d < dnn[i]:
                dnn[i] = d
                nn[i] = j

    m = n
    n_lvl = cnt.shape[0]

    if m <= n_lvl:
        for i in range(n):
            ox[m - 1, i] = a_px[i]
            oy[m - 1, i] = a_py[i]
            oz[m - 1, i] = a_pz[i]
            oe[m - 1, i] = a_e[i]
        cnt[m - 1] = m

    while m > 1:
        best = 1e300
        bi = -1
        for i in range(n):
            if alive[i] == 1 and nn[i] >= 0 and dnn[i] < best:
                best = dnn[i]
                bi = i
        if bi < 0:
            break
        bj = nn[bi]

        if m - 2 < d_out.shape[0]:
            d_out[m - 2] = best

        if wta == 1:
            # keep the harder parent's direction, add the transverse momenta,
            # and rebuild a massless four-vector from (pt, y, phi)
            if a_pt2[bi] >= a_pt2[bj]:
                ky = a_y[bi]
                kp = a_phi[bi]
            else:
                ky = a_y[bj]
                kp = a_phi[bj]
            npt = np.sqrt(a_pt2[bi]) + np.sqrt(a_pt2[bj])
            a_px[bi] = npt * np.cos(kp)
            a_py[bi] = npt * np.sin(kp)
            a_pz[bi] = npt * np.sinh(ky)
            a_e[bi] = npt * np.cosh(ky)
        else:
            a_px[bi] = a_px[bi] + a_px[bj]
            a_py[bi] = a_py[bi] + a_py[bj]
            a_pz[bi] = a_pz[bi] + a_pz[bj]
            a_e[bi] = a_e[bi] + a_e[bj]

        a_pt2[bi] = a_px[bi] * a_px[bi] + a_py[bi] * a_py[bi]
        a_y[bi] = _rap(a_e[bi], a_pz[bi])
        a_phi[bi] = np.arctan2(a_py[bi], a_px[bi])
        alive[bj] = 0
        nn[bj] = -1
        dnn[bj] = 1e300
        m -= 1

        # repair the cache: the merged pseudojet needs a full rescan, and so
        # does anything that used to point at either parent. Everything else
        # only has to be compared against the one new pseudojet.
        dnn[bi] = 1e300
        nn[bi] = -1
        for k2 in range(n):
            if alive[k2] == 0 or k2 == bi:
                continue
            dy = a_y[bi] - a_y[k2]
            dp = _dphi(a_phi[bi], a_phi[k2])
            dr2 = dy * dy + dp * dp
            u = a_pt2[bi]
            v = a_pt2[k2]
            d = (u if u < v else v) * dr2
            if d < dnn[bi]:
                dnn[bi] = d
                nn[bi] = k2
            if nn[k2] == bi or nn[k2] == bj:
                # its neighbour is gone or has changed: full rescan
                dnn[k2] = 1e300
                nn[k2] = -1
                for j2 in range(n):
                    if alive[j2] == 0 or j2 == k2:
                        continue
                    dy2 = a_y[k2] - a_y[j2]
                    dp2 = _dphi(a_phi[k2], a_phi[j2])
                    r2 = dy2 * dy2 + dp2 * dp2
                    uu = a_pt2[k2]
                    vv = a_pt2[j2]
                    dd = (uu if uu < vv else vv) * r2
                    if dd < dnn[k2]:
                        dnn[k2] = dd
                        nn[k2] = j2
            elif d < dnn[k2]:
                dnn[k2] = d
                nn[k2] = bi

        if m <= n_lvl:
            k = 0
            for i in range(n):
                if alive[i] == 1:
                    ox[m - 1, k] = a_px[i]
                    oy[m - 1, k] = a_py[i]
                    oz[m - 1, k] = a_pz[i]
                    oe[m - 1, k] = a_e[i]
                    k += 1
            cnt[m - 1] = k


@njit(parallel=True, cache=True)
def kt_observables(pt, eta, phi, en, nc, n_ax, use_e):
    """Split12, Split23, Qw and the winner-take-all axes for every jet.

    Returns
      split : (n_jets, 2)                    sqrt(d12), sqrt(d23) in MeV
      qw    : (n_jets,)                      min pairwise subjet mass, 3 subjets
      axeta/axphi : (n_jets, n_ax, n_ax)     [N-1] holds the N winner-take-all
                                             axes from the N-axis clustering
      naxes : (n_jets, n_ax)                 axes actually found at each level
    """
    n_jets = pt.shape[0]
    split = np.zeros((n_jets, 2), np.float64)
    qw = np.zeros(n_jets, np.float64)
    axeta = np.zeros((n_jets, n_ax, n_ax), np.float64)
    axphi = np.zeros((n_jets, n_ax, n_ax), np.float64)
    naxes = np.zeros((n_jets, n_ax), np.int64)

    for k in prange(n_jets):
        n = nc[k]
        if n < 2:
            continue

        # Constituents -> Cartesian four-vectors. eta in this dataset is a
        # pseudorapidity, so the momentum always follows from (pt, eta, phi).
        # The energy is the open question: use_e = 1 takes the stored
        # fjet_clus_E, which lets the constituent carry a mass; use_e = 0
        # forces E = pt cosh(eta), i.e. massless. ATLAS never states which one
        # went into the stored high-level values, so it is a knob the closure
        # test settles rather than an assumption baked into the code.
        cpx = np.empty(n, np.float64)
        cpy = np.empty(n, np.float64)
        cpz = np.empty(n, np.float64)
        cen = np.empty(n, np.float64)
        for i in range(n):
            cpx[i] = pt[k, i] * np.cos(phi[k, i])
            cpy[i] = pt[k, i] * np.sin(phi[k, i])
            cpz[i] = pt[k, i] * np.sinh(eta[k, i])
            if use_e == 1:
                e0 = en[k, i]
                p0 = np.sqrt(cpx[i] * cpx[i] + cpy[i] * cpy[i] + cpz[i] * cpz[i])
                cen[i] = e0 if e0 >= p0 else p0   # never allow a spacelike vector
            else:
                cen[i] = pt[k, i] * np.cosh(eta[k, i])

        # pass 1: E-scheme -> Split12, Split23 and the three Qw subjets
        d = np.zeros(4, np.float64)
        sx = np.zeros((n_ax, n_ax), np.float64)
        sy = np.zeros((n_ax, n_ax), np.float64)
        sz = np.zeros((n_ax, n_ax), np.float64)
        se = np.zeros((n_ax, n_ax), np.float64)
        c1 = np.zeros(n_ax, np.int64)
        _kt_cluster(cpx, cpy, cpz, cen, n, 0, d, sx, sy, sz, se, c1)

        split[k, 0] = np.sqrt(d[0])
        split[k, 1] = np.sqrt(d[1])

        if c1[2] == 3:
            best = 1e300
            for i in range(3):
                for j in range(i + 1, 3):
                    ee = se[2, i] + se[2, j]
                    xx = sx[2, i] + sx[2, j]
                    yy = sy[2, i] + sy[2, j]
                    zz = sz[2, i] + sz[2, j]
                    m2 = ee * ee - xx * xx - yy * yy - zz * zz
                    if m2 < 0.0:
                        m2 = 0.0
                    mm = np.sqrt(m2)
                    if mm < best:
                        best = mm
            qw[k] = best

        # pass 2: winner-take-all, snapshot the axes at every level 1..n_ax
        d2 = np.zeros(4, np.float64)
        wx = np.zeros((n_ax, n_ax), np.float64)
        wy = np.zeros((n_ax, n_ax), np.float64)
        wz = np.zeros((n_ax, n_ax), np.float64)
        we = np.zeros((n_ax, n_ax), np.float64)
        c2 = np.zeros(n_ax, np.int64)
        _kt_cluster(cpx, cpy, cpz, cen, n, 1, d2, wx, wy, wz, we, c2)
        for lv in range(n_ax):
            naxes[k, lv] = c2[lv]
            for i in range(c2[lv]):
                axeta[k, lv, i] = _rap(we[lv, i], wz[lv, i])
                axphi[k, lv, i] = np.arctan2(wy[lv, i], wx[lv, i])

    return split, qw, axeta, axphi, naxes


@njit(parallel=True, cache=True)
def nsubjettiness(pt, eta, phi, en, nc, axeta, axphi, naxes, n_max, beta,
                  use_e):
    """tau_N for N = 1..n_max, UNNORMALISED.

    tau_N = sum_k pt_k * min_j dR(k, axis_j)^beta

    The normalised form divides by d0 = R0^beta * sum_k pt_k. Both are in use
    in the literature and ATLAS does not state which one the open dataset
    stores, so the raw sum is returned here and the closure scan decides.

    dR is measured in RAPIDITY, not pseudorapidity, on both sides. ATLAS's
    NSubjettinessTool.cxx builds the measure as
        NormalizedCutoffMeasure(alpha, jet.getSizeParameter(), 1000000)
    and fjcontrib's measure calls PseudoJet::squared_distance, which is
    (dy)^2 + (dphi)^2. Rounds 1-3 compared the constituent's PSEUDOrapidity
    against the axis's rapidity - the axes were already stored as rapidity by
    kt_observables. For a massless constituent the two coincide, but ATLAS's
    UFO inputs combine tracks with clusters and can carry a mass, so the
    mismatch was real. Measured against fastjet on massive synthetic
    constituents it is worth about 1e-5 and carries NO dependence on N, so it
    is a correctness fix and NOT an explanation of the tau3 / tau4 residual.
    """
    n_jets = pt.shape[0]
    out = np.zeros((n_jets, n_max), np.float64)
    for k in prange(n_jets):
        n = nc[k]
        if n < 1:
            continue
        y = np.empty(n, np.float64)
        for c in range(n):
            pz = pt[k, c] * np.sinh(eta[k, c])
            if use_e == 1:
                p0 = pt[k, c] * np.cosh(eta[k, c])
                e0 = en[k, c] if en[k, c] >= p0 else p0
                y[c] = _rap(e0, pz)
            else:
                y[c] = eta[k, c]      # massless: rapidity IS pseudorapidity
        for N in range(1, n_max + 1):
            use = naxes[k, N - 1]
            if use < 1:
                continue
            s = 0.0
            for c in range(n):
                best = 1e300
                for a in range(use):
                    de = y[c] - axeta[k, N - 1, a]
                    dp = _dphi(phi[k, c], axphi[k, N - 1, a])
                    dr = np.sqrt(de * de + dp * dp)
                    if dr < best:
                        best = dr
                s += pt[k, c] * best ** beta
            out[k, N - 1] = s
    return out


@njit(parallel=True, cache=True)
def gen_ecf3(pt, y, phi, nc, beta):
    """The three generalised 3-point correlators, normalised by (sum pt)^3.

    For each triplet there are three pairwise angles. ATLAS's L-series is built
    from correlators that keep only the smallest angle, the two smallest, or
    all three:
        col 0 = 1e3  (smallest angle only)
        col 1 = 2e3  (product of the two smallest)
        col 2 = 3e3  (product of all three)

    The pairwise angles are cached in an n x n matrix before the triple loop.
    Round 2 recomputed two square roots inside the innermost loop; with the
    matrix cached the inner body is three array lookups and two comparisons.
    Measured on a 3,000-jet fixture with median 50 constituents: 16.2x faster,
    bit-identical output (max relative deviation exactly 0.0). That is the
    difference between 1e3 being a 100-hour diagnostic and 1e3 being something
    production can afford, which matters because L3 needs it.
    """
    n_jets = pt.shape[0]
    out = np.zeros((n_jets, 3), np.float64)
    for k in prange(n_jets):
        n = nc[k]
        if n < 3:
            continue
        tot = 0.0
        for i in range(n):
            tot += pt[k, i]
        if tot <= 0.0:
            continue
        R = np.zeros((n, n), np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                de = y[k, i] - y[k, j]
                dp = _dphi(phi[k, i], phi[k, j])
                r = np.sqrt(de * de + dp * dp) ** beta
                R[i, j] = r
                R[j, i] = r
        e1 = 0.0
        e2 = 0.0
        e3 = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                rij = R[i, j]
                wij = pt[k, i] * pt[k, j]
                for l in range(j + 1, n):
                    # order the three angles
                    a = rij
                    b = R[i, l]
                    c = R[j, l]
                    if a > b:
                        a, b = b, a
                    if b > c:
                        b, c = c, b
                    if a > b:
                        a, b = b, a
                    w = wij * pt[k, l]
                    e1 += w * a
                    e2 += w * a * b
                    e3 += w * a * b * c
        t3 = tot ** 3
        out[k, 0] = e1 / t3
        out[k, 1] = e2 / t3
        out[k, 2] = e3 / t3
    return out


@njit(parallel=True, cache=True)
def gen_ecf2(pt, y, phi, nc, beta):
    """1e2^(beta) = sum_{i<j} z_i z_j dR_ij^beta, normalised by (sum pt)^2.

    A pair has only one angle, so 1e2 is the entire 2-point correlator and no
    ordering is needed. L2's denominator uses it at beta = 2, which is the only
    place in the whole feature set where beta is not 1 - hence the argument.
    """
    n_jets = pt.shape[0]
    out = np.zeros(n_jets, np.float64)
    for k in prange(n_jets):
        n = nc[k]
        if n < 2:
            continue
        tot = 0.0
        for i in range(n):
            tot += pt[k, i]
        if tot <= 0.0:
            continue
        s = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                de = y[k, i] - y[k, j]
                dp = _dphi(phi[k, i], phi[k, j])
                s += pt[k, i] * pt[k, j] * np.sqrt(de * de + dp * dp) ** beta
        out[k] = s / (tot * tot)
    return out


@njit(parallel=True, cache=True)
def thrust_3d(pt, eta, phi, en, nc, n_scan, use_e):
    """ATLAS thrust major and minor from the three-momenta.

    Definition taken from the ATLAS implementation itself
    (JetSubStructureUtils/Root/Thrust.cxx), not inferred:

        clusters = boostToCenterOfMass(jet, jet.constituents())
        numerator_t += fabs(c.Dot(n));            thrust_major = num_t / den
        numerator_m += (c.Cross(n)).Mag();        thrust_minor = num_m / den
        denominator += c.Mag()

    with useThreeD = true and n the axis that maximises the sum. So what ATLAS
    stores as ThrustMaj is the ordinary thrust of the constituents in the jet
    rest frame - NOT the transverse "major" axis perpendicular to the thrust
    axis. Round 1's flat (eta, phi) major came out anti-correlated (rho -0.33)
    and round 2 found the rest-frame thrust matched at ratio 1.0000; the source
    explains why.

    Returns three columns:
      0  ATLAS thrust_major  = max_n sum|p.n| / sum|p|   <- fjet_ThrustMaj
      1  ATLAS thrust_minor  = sum|p x n| / sum|p|
      2  perpendicular-axis major, kept only as a control: it is what round 1
         computed, and it must NOT match.
    """
    n_jets = pt.shape[0]
    out = np.zeros((n_jets, 3), np.float64)
    for k in prange(n_jets):
        n = nc[k]
        if n < 3:
            continue
        px = np.empty(n)
        py = np.empty(n)
        pz = np.empty(n)
        ee = np.empty(n)
        for i in range(n):
            px[i] = pt[k, i] * np.cos(phi[k, i])
            py[i] = pt[k, i] * np.sin(phi[k, i])
            pz[i] = pt[k, i] * np.sinh(eta[k, i])
            if use_e == 1:
                p0 = np.sqrt(px[i] ** 2 + py[i] ** 2 + pz[i] ** 2)
                ee[i] = en[k, i] if en[k, i] >= p0 else p0
            else:
                ee[i] = pt[k, i] * np.cosh(eta[k, i])

        jx = 0.0; jy = 0.0; jz = 0.0; je = 0.0
        for i in range(n):
            jx += px[i]; jy += py[i]; jz += pz[i]; je += ee[i]
        m2 = je * je - jx * jx - jy * jy - jz * jz
        if m2 <= 0.0:
            continue
        mj = np.sqrt(m2)
        bx = jx / je; by = jy / je; bz = jz / je
        b2 = bx * bx + by * by + bz * bz
        if b2 >= 1.0:
            continue
        gam = je / mj

        rx = np.empty(n); ry = np.empty(n); rz = np.empty(n)
        tot = 0.0
        for i in range(n):
            bp = bx * px[i] + by * py[i] + bz * pz[i]
            f = 0.0
            if b2 > 0.0:
                f = ((gam - 1.0) * bp / b2) - gam * ee[i]
            rx[i] = px[i] + f * bx
            ry[i] = py[i] + f * by
            rz[i] = pz[i] + f * bz
            tot += np.sqrt(rx[i] ** 2 + ry[i] ** 2 + rz[i] ** 2)
        if tot <= 0.0:
            continue

        # Thrust axis. Two seed families, both refined by sign alignment until
        # the axis stops moving:
        #   - every single constituent direction;
        #   - all sign combinations of the four hardest constituents, which is
        #     exactly what ATLAS's Thrust.cxx uses as its 20 starting points.
        # Seeding from a strict superset of ATLAS's seeds can never return a
        # SMALLER maximum. Tested against a 30,000-direction sphere scan: with
        # constituent seeds alone, 2 jets in 120 landed in a local maximum
        # 0.4% low, so the sign-combination seeds are not redundant.
        bestT = -1.0
        ax = 0.0; ay = 0.0; az = 1.0
        n_lead = 4 if n >= 4 else n
        n_seed = n + (1 << n_lead)
        for s in range(n_seed):
            if s < n:
                ux = rx[s]; uy = ry[s]; uz = rz[s]
            else:
                bits = s - n
                ux = 0.0; uy = 0.0; uz = 0.0
                for i in range(n_lead):
                    sg = 1.0 if (bits >> i) & 1 else -1.0
                    ux += sg * rx[i]; uy += sg * ry[i]; uz += sg * rz[i]
            nrm = np.sqrt(ux * ux + uy * uy + uz * uz)
            if nrm <= 0.0:
                continue
            ux /= nrm; uy /= nrm; uz /= nrm
            for _ in range(20):
                sx = 0.0; sy = 0.0; sz = 0.0
                for i in range(n):
                    d = rx[i] * ux + ry[i] * uy + rz[i] * uz
                    sg = 1.0 if d >= 0.0 else -1.0
                    sx += sg * rx[i]; sy += sg * ry[i]; sz += sg * rz[i]
                q = np.sqrt(sx * sx + sy * sy + sz * sz)
                if q <= 0.0:
                    break
                sx /= q; sy /= q; sz /= q
                if abs(sx - ux) + abs(sy - uy) + abs(sz - uz) < 1e-12:
                    ux = sx; uy = sy; uz = sz
                    break
                ux = sx; uy = sy; uz = sz
            acc = 0.0
            for i in range(n):
                acc += abs(rx[i] * ux + ry[i] * uy + rz[i] * uz)
            if acc > bestT:
                bestT = acc
                ax = ux; ay = uy; az = uz

        # minor: sum of |p x n| along the same axis
        mi = 0.0
        for i in range(n):
            cx = ry[i] * az - rz[i] * ay
            cy = rz[i] * ax - rx[i] * az
            cz = rx[i] * ay - ry[i] * ax
            mi += np.sqrt(cx * cx + cy * cy + cz * cz)

        # control: best axis in the plane perpendicular to the thrust axis
        if abs(ax) < 0.9:
            hx = 1.0; hy = 0.0; hz = 0.0
        else:
            hx = 0.0; hy = 1.0; hz = 0.0
        e1x = hy * az - hz * ay
        e1y = hz * ax - hx * az
        e1z = hx * ay - hy * ax
        q = np.sqrt(e1x ** 2 + e1y ** 2 + e1z ** 2)
        e1x /= q; e1y /= q; e1z /= q
        e2x = ay * e1z - az * e1y
        e2y = az * e1x - ax * e1z
        e2z = ax * e1y - ay * e1x
        bestM = -1.0
        for s in range(n_scan):
            th = np.pi * s / n_scan
            ux = np.cos(th) * e1x + np.sin(th) * e2x
            uy = np.cos(th) * e1y + np.sin(th) * e2y
            uz = np.cos(th) * e1z + np.sin(th) * e2z
            acc = 0.0
            for i in range(n):
                acc += abs(rx[i] * ux + ry[i] * uy + rz[i] * uz)
            if acc > bestM:
                bestM = acc

        out[k, 0] = bestT / tot
        out[k, 1] = mi / tot
        out[k, 2] = bestM / tot
    return out


# --------------------------------------------------------- ECF via traces ---
def ecf123(pt, y, phi, nc, beta=1.0, block=2000):
    """ECF1, ECF2, ECF3 for a batch of jets, in MeV^N.

    ECF3 = sum_{i<j<k} pt_i pt_j pt_k dR_ij dR_ik dR_jk is a sum over triangles.
    Writing W_ij = sqrt(pt_i pt_j) dR_ij^beta with a zero diagonal gives
        W_ij W_jk W_ki = pt_i pt_j pt_k dR_ij dR_jk dR_ki,
    and because the diagonal vanishes every term of tr(W^3) with a repeated
    index is zero. Each distinct triangle appears 6 times, so ECF3 = tr(W^3)/6.
    That replaces the triple sum with two matrix products.

    Jets are processed in blocks sorted by constituent count, and each block is
    trimmed to the widest jet it contains. Because the cost goes as n^3, a
    block of 60-constituent jets costs (60/200)^3 - under 3% - of what it would
    cost padded out to the full 200 columns. On this dataset, where the median
    jet has around 50 constituents, that is the difference between a run that
    finishes overnight and one that does not.
    """
    n_jets = pt.shape[0]
    out = np.zeros((n_jets, 3), np.float64)
    order = np.argsort(nc, kind="stable")
    for s in range(0, n_jets, block):
        e = min(s + block, n_jets)
        sel = order[s:e]
        w = int(nc[sel].max())
        if w < 1:
            continue
        p = pt[sel, :w].astype(np.float64)
        et = y[sel, :w].astype(np.float64)
        ph = phi[sel, :w].astype(np.float64)
        mask = (np.arange(w)[None, :] < nc[sel, None])
        p = np.where(mask, p, 0.0)

        de = et[:, :, None] - et[:, None, :]
        dp = ph[:, :, None] - ph[:, None, :]
        dp = (dp + np.pi) % (2.0 * np.pi) - np.pi
        dr = np.sqrt(de * de + dp * dp) ** beta
        m2 = mask[:, :, None] & mask[:, None, :]
        dr = np.where(m2, dr, 0.0)
        idx = np.arange(dr.shape[1])
        dr[:, idx, idx] = 0.0

        out[sel, 0] = p.sum(axis=1)

        pp = p[:, :, None] * p[:, None, :]
        out[sel, 1] = 0.5 * (pp * dr).sum(axis=(1, 2))

        r = np.sqrt(p)
        wm = r[:, :, None] * r[:, None, :] * dr
        w2 = np.matmul(wm, wm)
        out[sel, 2] = np.einsum("bij,bji->b", w2, wm) / 6.0
        del de, dp, dr, pp, wm, w2
    return out


# ------------------------------------------------------------------- io -----
def load(ds, n_files, n_jets):
    d = DATA / ds
    fs = sorted(d.glob("*.h5"))[:n_files]
    if not fs:
        raise SystemExit(f"no .h5 files in {d}")
    cl = {c: [] for c in CLUS}
    hl = {h: [] for h in HL}
    got = 0
    for f in fs:
        with h5py.File(f, "r") as fh:
            k = fh[CLUS[0]].shape[0]
            take = k if n_jets is None else min(k, n_jets - got)
            for c in CLUS:
                cl[c].append(fh[c][:take, :MAX_CONSTITS])
            for h in HL:
                hl[h].append(fh[h][:take])
            got += take
        if n_jets is not None and got >= n_jets:
            break
    cl = {k: np.concatenate(v).astype(np.float64) for k, v in cl.items()}
    hl = {k: np.concatenate(v).astype(np.float64) for k, v in hl.items()}
    return cl, hl, [f.name for f in fs]


# --- fingerprint: kernels end ---
# -------------------------------------------------------------- reporting ---
def verdict(name, stored, calc):
    """One line of the closure table.

    r      Pearson correlation - is it the same physics?
    rho    Spearman - is the ordering the same? (this is what SHAP cares about)
    ratio  median stored/calc and the half-width of its middle 90%. A tight
           ratio near 1 is exact closure; a tight ratio away from 1 means we
           found the right observable with the wrong normalisation, which is
           fixable. A loose ratio means the observable itself is wrong.
    """
    ok = np.isfinite(stored) & np.isfinite(calc) & (np.abs(calc) > 0)
    ok &= (stored > -900) & (calc > -900)
    n = int(ok.sum())
    if n < 20:
        return f"{name:<18} {'n/a':>8}  too few usable jets ({n})"
    a = stored[ok]
    b = calc[ok]
    r = np.corrcoef(a, b)[0, 1]
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    rho = np.corrcoef(ra, rb)[0, 1]
    q = a / b
    med = np.median(q)
    lo, hi = np.percentile(q, [5, 95])
    spread = (hi - lo) / abs(med) if med != 0 else np.inf

    # The verdict is driven by rho and by the spread of the ratio, NOT by
    # Pearson r. Round 1 mislabelled Split23, ECF2 and ECF3 as MONOTONE purely
    # because r fell below 0.999, while rho was exactly 1.0000 and the ratio
    # was constant to 0.0%. These features are extremely heavy tailed - ECF3
    # runs over many orders of magnitude - and on such distributions a handful
    # of extreme jets drag r down without anything being wrong. Rank
    # correlation and the constancy of the ratio are the honest tests.
    if abs(rho) > 0.9995 and spread < 0.03:
        tag = "EXACT" if abs(med - 1) < 0.01 else "SCALED"
    elif abs(rho) > 0.999 and spread < 0.12:
        tag = "CLOSE"
    elif abs(rho) > 0.98:
        tag = "MONOTONE"
    elif abs(rho) > 0.85:
        tag = "WEAK"
    else:
        tag = "FAIL"
    return (f"{name:<26} r={r:+.4f}  rho={rho:+.4f}  "
            f"ratio={med:11.4g} +-{spread * 100:6.1f}%  n={n:<6}  {tag}")


def premise(hl, nc):
    """What are the stored branches ACTUALLY like?

    Round 1 tried to reproduce ThrustMaj, L2 and L3 without ever looking at
    their stored distributions. That is the wrong order: check what the target
    IS before checking whether some formula reproduces it. The ranges printed
    here settle the unit question outright - every dimensionful feature in
    round 1 came back with stored/calc = 1e-6 per power of pt, which means the
    stored values are NOT in the MeV the constituents are given in - and they
    tell us immediately whether ThrustMaj lives in [0, 1] like a normalised
    thrust or somewhere else entirely.
    """
    print(sub)
    print("PREMISE CHECK - what the stored branches actually contain")
    print(sub)
    print(f"{'branch':<18}{'min':>12}{'5%':>12}{'median':>12}"
          f"{'95%':>12}{'max':>12}{'<=0':>7}")
    for h in HL:
        v = hl[h]
        g = v[np.isfinite(v) & (v > -900)]
        if g.size == 0:
            print(f"{h:<18}  all sentinel / non-finite")
            continue
        p = np.percentile(g, [5, 50, 95])
        print(f"{h.replace('fjet_',''):<18}{g.min():12.4g}{p[0]:12.4g}"
              f"{p[1]:12.4g}{p[2]:12.4g}{g.max():12.4g}"
              f"{int((v <= 0).sum()):7d}")
    print(f"\nconstituent count: min {nc.min()}  median {int(np.median(nc))}"
          f"  max {nc.max()}")


def fit_monomial(target, basis, names, lo=-3, hi=3, top=5):
    """Find integer exponents a_k with target proportional to prod(basis_k^a_k).

    Round 1 scanned a hand-written list of candidate L2 / L3 formulas and the
    best it could do was rho = 0.86 and 0.95 - close enough to show the right
    ingredients are in the basis, not close enough to be the formula. Rather
    than keep guessing individual ratios, this searches the whole monomial
    grid and ranks by how CONSTANT the stored/candidate ratio is, which is the
    property that actually identifies a formula. If nothing scores well, the
    honest conclusion is that L2 and L3 are not monomials in this basis and
    the definition has to come from ATLAS, not from fitting.
    """
    nb = len(basis)
    ok = np.isfinite(target) & (target > -900) & (target != 0)
    for b in basis:
        ok &= np.isfinite(b) & (b > 0)
    if ok.sum() < 50:
        return []
    # 7^6 monomials is a large grid, so identify the formula on a subsample -
    # a few hundred jets is far more than enough to tell a constant ratio from
    # a drifting one, and the winner is re-scored on the full sample after.
    idx = np.flatnonzero(ok)
    if idx.size > 800:
        idx = np.random.default_rng(SEED).choice(idx, 800, replace=False)
    t = np.log(np.abs(target[idx]))
    L = np.array([np.log(b[idx]) for b in basis])

    rows = []
    exps = [0] * nb

    def walk(k):
        if k == nb:
            if all(e == 0 for e in exps):
                return
            pred = np.zeros_like(t)
            for i in range(nb):
                if exps[i]:
                    pred = pred + exps[i] * L[i]
            d = t - pred
            # constant ratio  <=>  small spread of the log difference
            q5, q50, q95 = np.percentile(d, [5, 50, 95])
            rows.append((q95 - q5, tuple(exps), float(np.exp(q50))))
            return
        for e in range(lo, hi + 1):
            exps[k] = e
            walk(k + 1)
        exps[k] = 0

    walk(0)
    # Break near-ties towards the simplest monomial. With six basis elements a
    # six-term product can always chase a bit more of the noise, and that is
    # overfitting, not a discovery. A real definition is short.
    rows.sort(key=lambda x: (round(x[0], 3),
                             sum(1 for e in x[1] if e),
                             sum(abs(e) for e in x[1])))
    out = []
    for spread, ex, scale in rows[:top]:
        s = " ".join(f"{names[i]}^{ex[i]}" for i in range(nb) if ex[i])
        out.append((s, spread, scale, ex))
    return out


def closure():
    print(line)
    print("CLOSURE TEST - recomputed high-level features vs stored nominal")
    print(line)
    cl, hl, names = load(DATASET, N_FILES, N_JETS)
    pt = cl["fjet_clus_pt"]
    eta = cl["fjet_clus_eta"]
    phi = cl["fjet_clus_phi"]
    nc = (pt > 0).sum(axis=1).astype(np.int64)
    n = pt.shape[0]
    print(f"dataset  : {DATASET}  files {names}")
    print(f"jets     : {n}")
    premise(hl, nc)
    print(sub)

    # Compile every kernel on a handful of jets first. Round 2's 12.6 s for
    # thrust and 140 jets/s for the correlators both had numba's JIT inside
    # them, which is a one-off cost paid per process, not a per-jet cost - and
    # the production estimate in section C is only worth anything if the rates
    # it is built from exclude it.
    # The angular coordinate for every dR in this file, computed once here and
    # handed to all three correlator kernels. See constituent_rapidity.
    yrap = constituent_rapidity(pt, eta, cl["fjet_clus_E"], nc,
                                1 if USE_CLUS_E else 0)

    w = slice(0, min(8, n))
    t = time.time()
    _, _, wae, wap, wan = kt_observables(
        pt[w], eta[w], phi[w], cl["fjet_clus_E"][w], nc[w], 4,
        1 if USE_CLUS_E else 0)
    nsubjettiness(pt[w], eta[w], phi[w], cl["fjet_clus_E"][w], nc[w],
                  wae, wap, wan, 4, 1.0, 1 if USE_CLUS_E else 0)
    ecf123(pt[w], yrap[w], phi[w], nc[w], beta=1.0)
    thrust_3d(pt[w], eta[w], phi[w], cl["fjet_clus_E"][w], nc[w], 360,
              1 if USE_CLUS_E else 0)
    gen_ecf3(pt[w], yrap[w], phi[w], nc[w], 1.0)
    gen_ecf2(pt[w], yrap[w], phi[w], nc[w], 2.0)
    print(f"numba compile            {time.time() - t:7.1f} s  (excluded below)")

    t = time.time()
    split, qw, axeta, axphi, naxes = kt_observables(
        pt, eta, phi, cl["fjet_clus_E"], nc, 4, 1 if USE_CLUS_E else 0)
    print(f"kt clustering            {time.time() - t:7.1f} s"
          f"   (constituent E: {'stored' if USE_CLUS_E else 'massless'})")

    t = time.time()
    tau_raw = nsubjettiness(pt, eta, phi, cl["fjet_clus_E"], nc,
                            axeta, axphi, naxes, 4, 1.0,
                            1 if USE_CLUS_E else 0)
    print(f"n-subjettiness           {time.time() - t:7.1f} s")

    t = time.time()
    ecf = ecf123(pt, yrap, phi, nc, beta=1.0)
    print(f"energy correlators       {time.time() - t:7.1f} s")

    t = time.time()
    thr3 = thrust_3d(pt, eta, phi, cl["fjet_clus_E"], nc, 360,
                     1 if USE_CLUS_E else 0)
    dt_thr = max(time.time() - t, 1e-9)
    print(f"thrust (rest frame)      {dt_thr:7.1f} s   {n / dt_thr:7.0f} jets/s")

    sumpt = ecf[:, 0]
    e1, e2, e3 = ecf[:, 0], ecf[:, 1], ecf[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        c2 = e3 * e1 / e2 ** 2
        d2 = e3 * e1 ** 3 / e2 ** 3
        tau_norm = tau_raw / sumpt[:, None]

    print(sub)
    print("A. the twelve features that follow from kt clustering and the ECFs")
    print(sub)
    for i in range(4):
        print(verdict(f"Tau{i+1}_wta raw", hl[f"fjet_Tau{i+1}_wta"], tau_raw[:, i]))
        print(verdict(f"Tau{i+1}_wta /sumpt", hl[f"fjet_Tau{i+1}_wta"], tau_norm[:, i]))
    print(verdict("Split12", hl["fjet_Split12"], split[:, 0]))
    print(verdict("Split23", hl["fjet_Split23"], split[:, 1]))
    print(verdict("ECF1", hl["fjet_ECF1"], e1))
    print(verdict("ECF2", hl["fjet_ECF2"], e2))
    print(verdict("ECF3", hl["fjet_ECF3"], e3))
    print(verdict("C2", hl["fjet_C2"], c2))
    print(verdict("D2", hl["fjet_D2"], d2))
    print(verdict("Qw", hl["fjet_Qw"], qw))

    print(sub)
    print("A3. Tau3 / Tau4 residual - where does the last 0.5% live?")
    print("    Everything else has been eliminated in a container against")
    print("    fastjet 3.5.1 as the referee:")
    print("      - our WTA axes vs fastjet WTA_KT_Axes      : 5e-15")
    print("      - our measure vs fastjet delta_R           : 1e-5, flat in N")
    print("      - float32 storage of the constituents      : 0.00000, no effect")
    print("      - minimised (OnePass) axes                 : 0.961 0.985 0.992")
    print("        0.993 - improves with N, the data worsens with N")
    print("    What is left is a difference between the constituents ATLAS fed")
    print("    its tool and the constituents this file stores. The two rows")
    print("    below test that: a residual that grows with the constituent")
    print("    count, or a pile-up at exactly 200, would mean truncation.")
    print(sub)
    edges = [0, 30, 50, 80, 120, 199, 201]
    print(f"    {'n constit':<14}{'jets':>7}"
          + "".join(f"{'tau' + str(i + 1):>10}" for i in range(4)))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (nc >= lo) & (nc < hi)
        if m.sum() < 20:
            continue
        cells = ""
        for i in range(4):
            a = hl[f"fjet_Tau{i+1}_wta"][m]
            b = tau_norm[m, i]
            ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
            cells += f"{np.median(a[ok] / b[ok]):10.5f}" if ok.sum() else \
                f"{'-':>10}"
        print(f"    {lo:>4}-{hi - 1:<9}{int(m.sum()):>7}{cells}")
    print(f"    jets at exactly 200 constituents: {int((nc == 200).sum())}"
          f" of {n}  ({100.0 * (nc == 200).sum() / n:.2f}%)")

    print(sub)
    print("A2. ThrustMaj - definition taken from ATLAS's own Thrust.cxx:")
    print("    thrust_major = max_n sum|p.n| / sum|p| in the jet rest frame,")
    print("    i.e. ordinary thrust. The perpendicular-axis 'major' that round")
    print("    1 computed is kept below only as a control and must NOT match.")
    print(sub)
    tm = hl["fjet_ThrustMaj"]
    print(verdict("ATLAS thrust_major", tm, thr3[:, 0]))
    print(verdict("  control: thrust_minor", tm, thr3[:, 1]))
    print(verdict("  control: perp-axis maj", tm, thr3[:, 2]))
    print(verdict("  control: 1 - thrust_major", tm, 1.0 - thr3[:, 0]))

    print(sub)
    print("B. L2 and L3 - definitions from ATL-PHYS-PUB-2021-028 section 5:")
    print("     L2 = 3e3^(b=1) / (1e2^(b=2))^(3/2)")
    print("     L3 = 1e3^(b=1) / (3e3^(b=1))^(1/3)")
    print("   with ve_N built from z_i = pt_i / sum pt, so both are")
    print("   dimensionless. The left subscript on L2's numerator does not")
    print("   survive PDF text extraction cleanly, so 1e3 and 2e3 run beside")
    print("   3e3 as controls - only one of the three can land at 1.0000.")
    print(sub)
    m = min(N_TRIPLET, n)
    rng = np.random.default_rng(SEED)
    sel = rng.choice(n, size=m, replace=False)
    t = time.time()
    g1 = gen_ecf3(pt[sel], yrap[sel], phi[sel], nc[sel], 1.0)
    dt_g = max(time.time() - t, 1e-9)
    print(f"generalised 3-pt (beta=1){dt_g:7.1f} s  on {m} jets"
          f"   {m / dt_g:7.0f} jets/s")
    t = time.time()
    q2 = gen_ecf2(pt, yrap, phi, nc, 2.0)
    dt_q = max(time.time() - t, 1e-9)
    print(f"generalised 2-pt (beta=2){dt_q:7.1f} s  on {n} jets"
          f"   {n / dt_q:7.0f} jets/s")

    with np.errstate(divide="ignore", invalid="ignore"):
        e1e3, e2e3, e3e3 = g1[:, 0], g1[:, 1], g1[:, 2]
        e1e2b2 = q2[sel]
        den2 = e1e2b2 ** 1.5
        l2 = e3e3 / den2
        l3 = e1e3 / e3e3 ** (1.0 / 3.0)
    print(verdict("L2 = 3e3/(1e2_b2)^1.5", hl["fjet_L2"][sel], l2))
    print(verdict("  control: 1e3 numerator", hl["fjet_L2"][sel], e1e3 / den2))
    print(verdict("  control: 2e3 numerator", hl["fjet_L2"][sel], e2e3 / den2))
    print(verdict("L3 = 1e3/(3e3)^(1/3)", hl["fjet_L3"][sel], l3))
    print(verdict("  control: 2e3 denominator", hl["fjet_L3"][sel],
                  e1e3 / e2e3 ** (1.0 / 3.0)))

    # The monomial search is the fallback, not the method. It only runs if the
    # sourced formulas failed, because fitting a formula to the data it is
    # supposed to predict is not evidence and must not be reported as if it
    # were. Round 2's best fits reached rho 0.891 and 0.974 and were rightly
    # refused as definitions.
    def landed(stored, calc):
        ok = np.isfinite(stored) & np.isfinite(calc) & (np.abs(calc) > 0)
        ok &= stored > -900
        if ok.sum() < 20:
            return False
        q = stored[ok] / calc[ok]
        lo, hi = np.percentile(q, [5, 95])
        med = np.median(q)
        return med != 0 and (hi - lo) / abs(med) < 0.03

    if landed(hl["fjet_L2"][sel], l2) and landed(hl["fjet_L3"][sel], l3):
        print("\n  Both sourced formulas reproduce a constant ratio. The"
              " monomial")
        print("  search is not run - there is nothing left to search for.")
    else:
        print("\n  A sourced formula did not close. Falling back to the"
              " monomial")
        print("  search over the correlator basis. Anything it finds is a"
              " LEAD to")
        print("  take back to ATLAS documentation, NOT a definition.")
        with np.errstate(divide="ignore", invalid="ignore"):
            z2b1 = e2[sel] / sumpt[sel] ** 2       # 1e2 at beta = 1
            basis = [z2b1, e1e2b2, e1e3, e2e3, e3e3]
            bnames = ["1e2b1", "1e2b2", "1e3b1", "2e3b1", "3e3b1"]
        for target in ("fjet_L2", "fjet_L3"):
            print(f"  -- {target}: best monomials over the correlator basis --")
            got = fit_monomial(hl[target][sel], basis, bnames)
            if not got:
                print("     nothing usable")
                continue
            for s, spr, scale, ex in got:
                v = np.ones(m)
                for i, e_ in enumerate(ex):
                    if e_:
                        v = v * basis[i] ** e_
                print("   " + verdict(s, hl[target][sel], v))

    print(sub)
    print("C. what production would cost at the rates measured above")
    print(sub)
    tot_jets = 60.3e6
    for nm, rate in [("thrust (rest frame)", n / dt_thr),
                     ("generalised 3-pt", m / dt_g),
                     ("generalised 2-pt", n / dt_q)]:
        print(f"  {nm:<24}{rate:8.0f} jets/s"
              f"   ->{tot_jets / rate / 3600:7.1f} h for 60.3M jets")
    print("  The 3-pt correlator is the binding cost and it is now needed in")
    print("  production, because L3 depends on 1e3. L2 does not: 3e3 comes")
    print("  free from the ECF trace trick and 1e2 is O(n^2).")

    print(line)
    print("HOW TO READ THIS")
    print("  EXACT     reproduced outright - use it.")
    print("  SCALED    right observable, constant normalisation offset. Fine:")
    print("            the paper compares nominal against systematic with the")
    print("            same code on both sides, so a constant factor cancels.")
    print("  MONOTONE  same ordering, non-constant offset. Usable for SHAP and")
    print("            for a tagger, but say so in the text.")
    print("  WEAK/FAIL do not use this feature until the definition is fixed.")
    print(line)


# --- fingerprint: produce begin ---
def produce():
    """Recompute the features for whole datasets and cache them to disk.

    One .npz per input .h5, written to STORE\\hl_recomputed\\<dataset>\\. The
    run is resumable: an output that already exists is skipped, so it can be
    killed and restarted, and it can be run one dataset at a time overnight.
    At roughly 360 jets/s the full 60.3M jets is about 47 hours, and there is
    no budget to do it twice - so everything needed downstream is written now,
    including the pairing key.

    What goes in each file:
      - all 15 features under their ATLAS names
      - the raw generalised correlators as well. They cost nothing extra once
        computed and they mean a normalisation change never requires a second
        pass over 250 GB
      - EventInfo_mcEventNumber, fjet_eta, fjet_phi, labels: the pairing key
        is (mcEventNumber, fjet_eta, fjet_phi, label) and pairing is useless
        without all four
      - fjet_pt and fjet_m, needed for the pT-binned background-rejection
        metric ATLAS prescribes

    gen_ecf3 IS called here, which is new in round 3. It was excluded before
    at 140 jets/s (109 hours on its own); with the angle matrix cached it runs
    16x faster, and L3's definition needs 1e3, so it has to be in. Check the
    rate the closure run reports for your machine before committing a sitting.
    """
    out_root = STORE / "hl_recomputed"
    todo = PRODUCE_SETS if PRODUCE_SETS else sorted(
        p.name for p in DATA.iterdir() if p.is_dir())

    for ds in todo:
        src = DATA / ds
        if not src.is_dir():
            print(f"skip {ds}: not found")
            continue
        dst = out_root / ds
        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(src.glob("*.h5"))
        print(f"\n{line}\n{ds}: {len(files)} files -> {dst}\n{line}")

        for fi, f in enumerate(files):
            o = dst / (f.stem + ".npz")
            if o.exists():
                try:
                    with np.load(o, allow_pickle=False) as z:
                        prev = str(z["code_fp"]) if "code_fp" in z else ""
                except Exception:
                    prev = ""
                if prev == CODE_FP:
                    continue
                print(f"  [{fi + 1:3d}/{len(files)}] {f.name}  written by code "
                      f"{prev[:8] or '(unstamped)'}, this is {CODE_FP[:8]}"
                      f" - recomputing")
            t0 = time.time()
            with h5py.File(f, "r") as fh:
                cpt = fh["fjet_clus_pt"][:, :MAX_CONSTITS].astype(np.float64)
                cet = fh["fjet_clus_eta"][:, :MAX_CONSTITS].astype(np.float64)
                cph = fh["fjet_clus_phi"][:, :MAX_CONSTITS].astype(np.float64)
                cen = fh["fjet_clus_E"][:, :MAX_CONSTITS].astype(np.float64)
                keep = {k: fh[k][:] for k in
                        ("EventInfo_mcEventNumber", "fjet_eta", "fjet_phi",
                         "fjet_pt", "fjet_m", "labels") if k in fh}
            nc = (cpt > 0).sum(axis=1).astype(np.int64)
            cy = constituent_rapidity(cpt, cet, cen, nc,
                                      1 if USE_CLUS_E else 0)

            split, qw, axeta, axphi, naxes = kt_observables(
                cpt, cet, cph, cen, nc, 4, 1 if USE_CLUS_E else 0)
            tau = nsubjettiness(cpt, cet, cph, cen, nc, axeta, axphi, naxes,
                                4, 1.0, 1 if USE_CLUS_E else 0)
            th3 = thrust_3d(cpt, cet, cph, cen, nc, 360,
                            1 if USE_CLUS_E else 0)
            gen = gen_ecf3(cpt, cy, cph, nc, 1.0)
            q2 = gen_ecf2(cpt, cy, cph, nc, 2.0)
            q1 = gen_ecf2(cpt, cy, cph, nc, 1.0)

            # ECF1/2/3 from the correlators that production already has to
            # compute, instead of a separate ecf123 pass. The generalised
            # correlators are normalised by (sum pt)^N and are otherwise the
            # same sums:
            #     ECF1 = sum pt
            #     ECF2 = 1e2^(b=1) * (sum pt)^2
            #     ECF3 = 3e3^(b=1) * (sum pt)^3     (all three angles, so the
            #                                        ordering inside gen_ecf3
            #                                        does not enter)
            # gen_ecf3 is needed for L3 and gen_ecf2 for L2 regardless, so the
            # only new work is one O(n^2) pass at beta = 1. What it replaces
            # is ecf123, which round 5 measured at 794 jets/s - 21 of the 30
            # hours the full production would have taken.
            # NOTE: closure() deliberately still calls ecf123. Keeping the
            # trace-identity route alive there means every closure run
            # re-checks this identity against real data for free.
            e1 = cpt.sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                e2 = q1 * e1 ** 2
                e3 = gen[:, 2] * e1 ** 3
            with np.errstate(divide="ignore", invalid="ignore"):
                taun = tau / e1[:, None]
                c2 = e3 * e1 / e2 ** 2
                d2 = e3 * e1 ** 3 / e2 ** 3
                l2 = gen[:, 2] / q2 ** 1.5
                l3 = gen[:, 0] / gen[:, 2] ** (1.0 / 3.0)

            res = {
                "fjet_Tau1_wta": taun[:, 0], "fjet_Tau2_wta": taun[:, 1],
                "fjet_Tau3_wta": taun[:, 2], "fjet_Tau4_wta": taun[:, 3],
                "fjet_Split12": split[:, 0], "fjet_Split23": split[:, 1],
                "fjet_ECF1": e1, "fjet_ECF2": e2, "fjet_ECF3": e3,
                "fjet_C2": c2, "fjet_D2": d2, "fjet_Qw": qw,
                "fjet_L2": l2, "fjet_L3": l3,
                "fjet_ThrustMaj": th3[:, 0],
                # the raw correlators cost nothing extra to store and save a
                # second 250 GB pass if a normalisation ever has to be redone
                "gen_1e3_b1": gen[:, 0], "gen_2e3_b1": gen[:, 1],
                "gen_3e3_b1": gen[:, 2], "gen_1e2_b2": q2,
                "gen_1e2_b1": q1,
                "thrust_minor": th3[:, 1],
                "n_constit": nc,
                "code_fp": np.array(CODE_FP),
            }
            res.update(keep)
            np.savez_compressed(
                o, **{k: v.astype(np.float32) if v.dtype.kind == "f" else v
                      for k, v in res.items()})
            n = cpt.shape[0]
            dt = time.time() - t0
            print(f"  [{fi + 1:3d}/{len(files)}] {f.name}  {n} jets  "
                  f"{dt:6.1f}s  {n / dt:6.0f} jets/s")

        summarise(ds, dst)

    print(f"\n{line}\ndone. cache in {out_root}\n{line}")


# --- fingerprint: produce end ---
def summarise(ds, dst):
    """Read back what is on disk for one dataset and print what it contains.

    Deliberately reads the .npz files rather than reusing values held in
    memory: the point is to check the delivered product, and it works the same
    way on a resumed run where most files were skipped.

    This is a description, not a verdict. There is no stored branch to compare
    against - the systematic files carry stale high-level values, which is why
    the recomputation exists - so what it gives you is the distribution of
    every feature, how far it moves from file to file, and whether the whole
    dataset was written by one version of the code. A feature whose median
    jumps between files, or a second fingerprint appearing, is the thing to
    look for.
    """
    fs = sorted(dst.glob("*.npz"))
    if not fs:
        return
    feats = ["fjet_Tau1_wta", "fjet_Tau2_wta", "fjet_Tau3_wta", "fjet_Tau4_wta",
             "fjet_Split12", "fjet_Split23", "fjet_ECF1", "fjet_ECF2",
             "fjet_ECF3", "fjet_C2", "fjet_D2", "fjet_L2", "fjet_L3",
             "fjet_Qw", "fjet_ThrustMaj"]
    med = {k: [] for k in feats}
    fps, jets, sig, bad, ncs = set(), 0, 0.0, 0, []
    for f in fs:
        with np.load(f, allow_pickle=False) as z:
            for k in feats:
                v = z[k].astype(np.float64)
                bad += int((~np.isfinite(v)).sum())
                med[k].append(np.median(v))
            n = z["n_constit"]
            jets += n.size
            ncs.append(np.median(n))
            if "labels" in z:
                sig += float(z["labels"].sum())
            fps.add(str(z["code_fp"]) if "code_fp" in z else "(unstamped)")

    print(f"\n{sub}\nSUMMARY - {ds}: {len(fs)} file"
          f"{'' if len(fs) == 1 else 's'}, {jets:,} jets\n{sub}")
    print(f"{'feature':<16}{'median':>14}{'file-to-file spread':>22}")
    for k in feats:
        m = float(np.median(med[k]))
        span = (max(med[k]) - min(med[k])) / abs(m) * 100 if m else float("nan")
        print(f"{k.replace('fjet_', ''):<16}{m:>14.5g}{span:>21.2f}%")
    print(f"\nmedian constituents  {int(np.median(ncs))}")
    print(f"signal fraction      {100.0 * sig / jets:.2f}%")
    print(f"non-finite values    {bad}")
    print(f"code fingerprints    {len(fps)}  {sorted(fps)}")
    if len(fps) != 1 or bad:
        print("*** LOOK AT THIS: more than one code version, or non-finite"
              " values present ***")


def main():
    if not DATA.is_dir():
        raise SystemExit(f"data folder not found:\n  {DATA}")
    # In produce mode the log is named after the datasets it covers, so a run
    # over bias never lands in the same file as a run over esup. Beyond three
    # datasets the name would stop being a name, so it becomes "all".
    tag = MODE
    outdir = None
    if MODE != "closure":
        sets = PRODUCE_SETS if PRODUCE_SETS else ["all"]
        name = "_".join(sets) if len(sets) <= 3 else "all"
        tag = "produce_" + name
        # one subfolder per dataset, so bias, test_nominal and the rest never
        # share a directory. Windows folder names are case insensitive, so a
        # folder already created by hand as "Bias" is the same folder.
        outdir = RESULTS / name

    tee = Tee(tag, outdir)
    sys.stdout = tee
    try:
        print(f"# ATLAS 80030 - {tag} round {tee.round}"
              f" - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"# code fingerprint {CODE_FP}")
        if MODE == "closure":
            closure()
        else:
            produce()
    finally:
        sys.stdout = tee.out
        tee.close()
    print(f"\nwritten to {tee.path}")
    print(f"code snapshot {tee.code}")


if __name__ == "__main__":
    main()
