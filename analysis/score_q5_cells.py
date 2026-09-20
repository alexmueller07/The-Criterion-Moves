"""Score the five-task Qwen (q5) UCIT cells from their raw generations.

Written 2026-09-20. The project aggregator (fs_aggregate.py) cannot read these
cells, for two independent reasons, both verified rather than assumed:

  1. fs_common.FAMILY_RE requires the ordering token to match ``o\\d+``, so the
     runtag ``fsQ_seq_q5_s17`` returns None from parse_runtag -- the cells are
     silently "not one of ours" and are dropped without a warning.
  2. fs_common.suite_info sets n_stages from the FIRST ordering in the manifest
     (``len(next(iter(orders.values())))``), i.e. one number for the whole
     suite. The UCIT suite's o1/o2/o3 are six tasks, so a five-task ordering is
     judged incomplete against T=6 even if the regex matched.

Fixing (2) means touching the completeness logic that decides which LLaVA cells
enter every headline number, which is not a change to make days before a
deadline. So the q5 cells are scored here instead, by the same arithmetic.

CALIBRATION IS NOT OPTIONAL. This script refuses to report anything until it has
reproduced known LLaVA cells from the released readout to 1e-4 on H, FA, c and
d'. A scorer that has not been shown to agree with the one that produced the
paper's numbers is not evidence about anything.

Usage (on the cluster, where the generations live):
    python3 score_q5_cells.py --results ~/cl-halluc/results_fs \\
                              --readout  ~/cl-halluc/readout/fs_aggregate.json
"""
import argparse
import glob
import json
import math
import os
import re
import sys

TOL = 1e-4
# A yes/no scorer measures a criterion only while the model still answers yes/no.
# Qwen collapses to ImageNet-R class labels after that stage ("Backpack",
# "Ballplayer", "Snowboard"), and at seed 23 only 1066 of 9000 answers remained
# parseable. Scoring the survivors is a selected subsample, not a criterion, and
# it produced c = +2.32 out of nothing. Stages below this floor are reported as
# FORMAT COLLAPSE and excluded from every derived quantity.
PARSE_FLOOR = 0.95
# Cells used as the positive control, with the readout key they must match.
CONTROLS = [("fsL_seq_o1_s17_k1", "seq|o1|s17", "1"),
            ("fsL_seq_o1_s17_k2", "seq|o1|s17", "2"),
            ("fsL_anchor_o1_s17_k3", "anchor|o1|s17", "3")]


def z(p, n=4500):
    """Inverse normal CDF by bisection, with the scorer's 1/(2n) edge correction."""
    p = min(max(p, 1.0 / (2 * n)), 1 - 1.0 / (2 * n))
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def parse_yn(out):
    """Format-robust yes/no extraction: strip leading punctuation, match the
    first token. Returns None when the answer is neither, which is counted as
    an exclusion and never coerced to a label."""
    t = re.sub(r"^[^a-z]+", "", (out or "").strip().lower())
    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0
    return None


def score_cell(path):
    f = os.path.join(path, "pope_gen.jsonl")
    if not (os.path.exists(os.path.join(path, "EVAL_DONE")) and os.path.exists(f)):
        return None
    hy = hn = fy = fn = bad = trunc = tot = 0
    for line in open(f):
        r = json.loads(line)
        tot += 1
        trunc += 1 if r.get("truncated") else 0
        a = parse_yn(r.get("output", ""))
        if a is None:
            bad += 1
            continue
        if r["gt"] == "yes":
            hy += a == 1
            hn += a == 0
        else:
            fy += a == 1
            fn += a == 0
    if not (hy + hn) or not (fy + fn):
        return None
    H, FA = hy / (hy + hn), fy / (fy + fn)
    rate = (tot - bad) / tot if tot else 0.0
    return {"n_total": tot, "n_parsed": tot - bad, "truncated": trunc,
            "parse_rate": rate, "usable": rate >= PARSE_FLOOR,
            "H": H, "FA": FA, "c": -0.5 * (z(H) + z(FA)), "dprime": z(H) - z(FA),
            "yes_rate": (hy + fy) / max(tot - bad, 1)}


def calibrate(results, readout):
    """Refuse to proceed unless this scorer reproduces the released readout."""
    arms = json.load(open(readout))["backbones"]["llava15"]["arms"]
    checked = 0
    for cell, arm, stage in CONTROLS:
        got = score_cell(os.path.join(results, cell))
        want = arms.get(arm, {}).get(stage, {}).get("pope")
        if got is None or want is None:
            print("  control %-24s SKIPPED (cell or readout entry absent)" % cell)
            continue
        for k in ("H", "FA", "c", "dprime"):
            if abs(got[k] - want[k]) > TOL:
                sys.exit("CALIBRATION FAILED on %s.%s: got %.6f want %.6f"
                         % (cell, k, got[k], want[k]))
        print("  control %-24s OK (H/FA/c/d' match readout within %g)" % (cell, TOL))
        checked += 1
    if checked == 0:
        sys.exit("CALIBRATION IMPOSSIBLE: no control cell was available. "
                 "Refusing to report q5 numbers from an uncalibrated scorer.")
    return checked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.expanduser("~/cl-halluc/results_fs"))
    ap.add_argument("--readout", default=os.path.expanduser("~/cl-halluc/readout/fs_aggregate.json"))
    ap.add_argument("--out", default=None, help="write the q5 trajectories here as JSON")
    a = ap.parse_args()

    print("calibration against the released readout:")
    n = calibrate(a.results, a.readout)
    print("  %d/%d controls reproduced\n" % (n, len(CONTROLS)))

    cells = {}
    for d in sorted(glob.glob(os.path.join(a.results, "fsQ_seq_q5_s*_k*"))):
        m = re.match(r".*fsQ_seq_q5_s(\d+)_k(\d+)$", d)
        if not m:
            continue
        s = score_cell(d)
        if s:
            cells.setdefault("s" + m.group(1), {})[m.group(2)] = s

    print("q5 (five-task) Qwen2.5-VL cells:")
    for seed in sorted(cells):
        ks = sorted(cells[seed], key=int)
        print("  seed %s  stages scored: %s" % (seed, ",".join(ks)))
        for k in ks:
            v = cells[seed][k]
            if not v["usable"]:
                print("     S%s  FORMAT COLLAPSE: only %d/%d answers parseable (%.1f%%). "
                      "No criterion is defined here; c/d' withheld."
                      % (k, v["n_parsed"], v["n_total"], 100 * v["parse_rate"]))
                continue
            print("     S%s  c=%+.4f  d'=%.4f  H=%.4f FA=%.4f  parsed %d/%d  trunc %d"
                  % (k, v["c"], v["dprime"], v["H"], v["FA"],
                     v["n_parsed"], v["n_total"], v["truncated"]))
        usable = [k for k in ks if cells[seed][k]["usable"]]
        if len(usable) != len(ks):
            print("     %d of %d stages unusable; drift and ranges below cover the "
                  "usable prefix only and are NOT an endpoint."
                  % (len(ks) - len(usable), len(ks)))
        ks = usable
        cs = [cells[seed][k]["c"] for k in ks]
        ds = [cells[seed][k]["dprime"] for k in ks]
        if len(ks) >= 2:
            # post-settling: every consecutive pair here is already post-settling,
            # because the excluded k=1 step is base -> S1 and the base is not in
            # this list. (Indexing from 2 drops a real transition.)
            drift = sum(abs(cs[i] - cs[i - 1]) for i in range(1, len(cs)))
            print("     post-settling sum|dc| over scored stages: %.4f" % drift)
            print("     c range %.4f   d' range %.4f" % (max(cs) - min(cs), max(ds) - min(ds)))
        if len(ks) < 5:
            print("     INCOMPLETE: %d of 5 stages scored; no endpoint claim." % len(ks))

    if a.out:
        json.dump(cells, open(a.out, "w"), indent=1)
        print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
