#!/usr/bin/env python3
"""Does the z-ROC coherence result replicate on a SECOND task sequence?

WHY (2026-09-13). The paper's primary claim -- a cell's checkpoints slide along one
z-ROC under sequential tuning, and an anchor regularizer breaks that -- rests
entirely on UCIT. The obvious question is whether it is a property of continual
tuning or a property of that benchmark. The pilot suite answers it for free: a
DIFFERENT task sequence (ScienceQA -> TextVQA -> Flickr30k -> VizWiz), the same
backbone, the same instrument, run months earlier for a different purpose.

This is a second BENCHMARK, not a second backbone. It does not answer the
generality-across-models question, and must not be presented as if it did.

RECOVERING H AND FA. The pilot's pope.csv stores recall and yes-rate rather than
hit/false-alarm. POPE is balanced 50/50, so
    H  = recall
    FA = 2*yes_rate - H
and the identity is checked against the reported precision, H/(H+FA), on every row
before the row is used -- a silent mismatch would mean the balance assumption is
wrong for that cell and the cell is dropped rather than fitted.

The pilot suite has a known defect that does NOT apply here: its answer-token
casing drifts across stages, which corrupts any LOGIT-based readout. pope.csv is
TEXT-scored (string match after lowercasing), so it absorbs the casing flip
entirely. See the 2026-09-12 prereg entry.

READING. Asymmetric, as in zroc_coherence.py: a LOW R^2 rejects fixed
distributions for that arm and is the secure direction; a HIGH R^2 is only
consistent with them. Four to five points per cell here against UCIT's six, so the
fits are weaker and the contrast between arms carries the result, not any absolute.
"""
import csv, json, math, os, sys
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MPLBACKEND", "Agg")
from zroc_coherence import z, fit          # the certified fit logic

# Where the pilot per-stage result directories live. Override with --root;
# the default assumes they sit beside this repository.
DEFAULT_ROOT = os.environ.get(
    "PILOT_RESULTS_ROOT",
    os.path.join(os.path.dirname(HERE), "pilot_results"))

# pilot arms, as the cells were named on the cluster
CELLS = {
    "seq":    [["S1", "S2", "S3", "S4"],
               ["seq_rev_k1", "seq_rev_k2", "seq_rev_k3", "seq_rev_k4"],
               ["seq_s2_k1", "seq_s2_k2", "seq_s2_k3", "seq_s2_k4"],
               ["seq_s3_k1", "seq_s3_k2", "seq_s3_k3", "seq_s3_k4"]],
    "anchor": [["G1", "G2", "G3", "G4"],
               ["g_rev_k1", "g_rev_k2", "g_rev_k3", "g_rev_k4"],
               ["g_s2_k1", "g_s2_k2", "g_s2_k3", "g_s2_k4"],
               ["g_s3_k1", "g_s3_k2", "g_s3_k3", "g_s3_k4"]],
    "joint":  [["J1", "J2", "J3", "J4"]],
    "er":     [["E1", "E2", "E3", "E4"]],
    "anchor_v1": [["F1", "F2", "F3", "F4"]],
}


def read_point(root, stage):
    """(z(FA), z(H)) for one checkpoint, or None if the balance check fails."""
    p = os.path.join(root, stage, "pope.csv")
    if not os.path.isfile(p):
        return None
    for row in csv.DictReader(open(p)):
        if row.get("category") != "all":
            continue
        H = float(row["recall"])
        FA = 2.0 * float(row["yes_rate"]) - H
        if not (0.0 < H < 1.0 and 0.0 < FA < 1.0):
            return None
        # the balance identity, verified rather than assumed
        pred = H / (H + FA)
        rep = float(row["precision"])
        if abs(pred - rep) > 0.002:
            print("  [drop] %s: precision %.4f != %.4f implied by a 50/50 probe"
                  % (stage, rep, pred))
            return None
        return (z(FA), z(H))
    return None


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT
    if not os.path.isdir(root):
        raise SystemExit("pilot results not found at %s -- pass the path as argv[1]" % root)
    print("Pilot suite (ScienceQA -> TextVQA -> Flickr30k -> VizWiz), LLaVA-1.5-7B")
    print("Second BENCHMARK for the coherence claim; not a second backbone.\n")
    print("%-11s %4s %9s %9s %10s %10s" % ("arm", "n", "mean R2", "min R2", "mean d_a", "sd d_a"))
    print("-" * 58)
    rep = {}
    for arm, cells in CELLS.items():
        r2s, das, rows = [], [], []
        for cell in cells:
            pts = [q for q in (read_point(root, s) for s in cell) if q]
            if len(pts) < 4:
                continue
            f = fit(pts)
            if not f:
                continue
            a, b, r2, d_a = f
            r2s.append(r2); das.append(d_a)
            rows.append({"cell": cell[0], "n_pts": len(pts), "r2": round(r2, 4),
                         "slope": round(b, 4), "d_a": round(d_a, 4)})
        if not r2s:
            print("%-11s   -- no usable cells" % arm)
            continue
        sd = (st.pstdev(das) * math.sqrt(len(das) / max(1, len(das) - 1))
              if len(das) > 1 else 0.0)
        print("%-11s %4d %9.4f %9.4f %10.4f %10.4f"
              % (arm, len(r2s), st.mean(r2s), min(r2s), st.mean(das), sd))
        rep[arm] = {"n_cells": len(r2s), "mean_r2": round(st.mean(r2s), 4),
                    "min_r2": round(min(r2s), 4), "mean_d_a": round(st.mean(das), 4),
                    "sd_d_a": round(sd, 4), "per_cell": rows}

    s, a = rep.get("seq"), rep.get("anchor")
    if s and a:
        print("\nReading:")
        print("  sequential R2 %.4f vs anchor R2 %.4f  (UCIT: 0.9569 vs 0.5857)"
              % (s["mean_r2"], a["mean_r2"]))
        print("  d_a spread   %.4f vs %.4f            (UCIT: 0.0402 vs 0.3618)"
              % (s["sd_d_a"], a["sd_d_a"]))
        same_dir = s["mean_r2"] > a["mean_r2"] and s["sd_d_a"] < a["sd_d_a"]
        print("\n  %s" % ("REPLICATES in direction: sequential fits one ROC better and holds"
                          " d_a tighter,\n  on a different task sequence."
                          if same_dir else
                          "DOES NOT replicate. The UCIT result may be benchmark-specific and"
                          "\n  the paper must say so."))
        print("\n  Caveats that bound this: 4 points per cell against UCIT's 6, so fits are")
        print("  weaker; 4 cells per arm against 9; same backbone, so this is a second")
        print("  benchmark and NOT evidence of generality across models.")

    out = os.path.join(HERE, "readout", "zroc_pilot_replication.json")
    json.dump(rep, open(out, "w"), indent=2)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
