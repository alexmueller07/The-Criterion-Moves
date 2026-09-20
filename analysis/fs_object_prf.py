#!/usr/bin/env python3
"""Object-mention precision/recall/F1 across the FULL-STUDY cell matrix.

WHY (2026-09-09). analysis/object_prf.py established, on a single bench cell,
that CHAIR is precision-only and that the anchor's apparent lack of a generative
benefit is a metric artifact: on Object HalBench the anchor recovered +0.085 F1
over sequential while CHAIR called it worse. That was ONE cell (o1/s17), and its
bootstrap interval was over images within one run -- image-sampling noise, not
run-to-run variance. It could not support a claim.

This script removes that limitation without spending any GPU time. The in-domain
CHAIR generations (`chair_gen.jsonl`) already exist for every full-study cell, so
the same precision/recall/F1 decomposition can be run across the whole 3-ordering
x 3-seed matrix and paired cell-by-cell. The unit of replication becomes the CELL,
which is the unit the rest of the pre-registration uses ("8/9 cells" style rules).

Reports, per arm pair, the per-cell F1 delta and the count of cells favouring the
method -- the same form as the E2 freeze rule -- plus a two-sided exact sign test.
It does NOT bootstrap over images here: with 9 paired cells the honest statement
is the cell-level one, and mixing an image-level interval into a cell-level claim
is exactly the noise-floor conflation this project has recorded before.

Usage:
  python3 fs_object_prf.py --results <results_fs> --coco_gt <coco_gt.json>
      --arms seq,anchor --prefix fsL --stage k6 [--out out.json]
"""
import argparse
import itertools
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fs_common  # noqa: E402
from object_prf import aggregate, per_image_counts  # noqa: E402


def discover(results, prefix, arm, stage):
    """{(ordering, seed): cell_dir} for prefix_arm_o<O>_s<S>_<stage>."""
    pat = re.compile(r"^%s_%s_o(?P<o>\d+)_s(?P<s>\d+)_%s$"
                     % (re.escape(prefix), re.escape(arm), re.escape(stage)))
    found = {}
    for d in sorted(os.listdir(results)):
        m = pat.match(d)
        if m and os.path.isfile(os.path.join(results, d, "chair_gen.jsonl")):
            found[(int(m.group("o")), int(m.group("s")))] = os.path.join(results, d)
    return found


def sign_test_p(k, n):
    """Two-sided exact binomial p under p=0.5 (no scipy on this cluster)."""
    if n == 0:
        return None
    c = [math.comb(n, i) for i in range(n + 1)]
    tot = float(sum(c))
    kk = min(k, n - k)
    tail = sum(c[i] for i in range(0, kk + 1)) / tot
    return round(min(1.0, 2.0 * tail), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--coco_gt", required=True)
    ap.add_argument("--prefix", default="fsL")
    ap.add_argument("--arms", default="seq,anchor")
    ap.add_argument("--stage", default="k6",
                    help="endpoint checkpoint suffix (UCIT has 6 tasks -> k6)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    coco_gt = fs_common.load_coco_gt(args.coco_gt)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    cells = {}
    for arm in arms:
        cells[arm] = discover(args.results, args.prefix, arm, args.stage)
        print("arm %-8s: %d cells %s" % (arm, len(cells[arm]),
                                         sorted(cells[arm].keys())))
    if not any(cells.values()):
        print("no cells found under %s" % args.results)
        return 1

    report = {"prefix": args.prefix, "stage": args.stage, "per_cell": {}, "pairs": {}}

    print("\nPer-cell object-mention precision / recall / F1 (in-domain CHAIR):")
    hdr = "%-10s %-8s %9s %9s %9s %9s %9s" % (
        "cell", "arm", "precision", "recall", "F1", "ment/cap", "CHAIR_i")
    print("\n" + hdr)
    print("-" * len(hdr))
    per = {}
    for arm in arms:
        for key in sorted(cells[arm]):
            try:
                cnt = per_image_counts(os.path.join(cells[arm][key], "chair_gen.jsonl"),
                                       coco_gt)
            except Exception as e:  # noqa: BLE001
                print("SKIP %s %s: %s: %s" % (arm, key, type(e).__name__, e))
                continue
            a = aggregate(cnt)
            per[(arm, key)] = a
            report["per_cell"]["%s|o%d_s%d" % (arm, key[0], key[1])] = a
            print("%-10s %-8s %9.4f %9.4f %9.4f %9.3f %9.4f" %
                  ("o%d_s%d" % key, arm, a["precision"], a["recall"], a["f1"],
                   a["mentions_per_caption"], a["chair_i_check"]))

    for a1, a2 in itertools.combinations(arms, 2):
        shared = sorted(set(cells[a1]) & set(cells[a2]))
        shared = [k for k in shared if (a1, k) in per and (a2, k) in per]
        if not shared:
            continue
        print("\n=== %s vs %s over %d paired cells ===" % (a1, a2, len(shared)))
        print("%-10s %10s %10s %10s %10s" %
              ("cell", "dF1", "dprecision", "drecall", "dCHAIR_i"))
        d_f1 = []
        rows = []
        for k in shared:
            p1, p2 = per[(a1, k)], per[(a2, k)]
            d = {"cell": "o%d_s%d" % k,
                 "d_f1": round(p1["f1"] - p2["f1"], 4),
                 "d_precision": round(p1["precision"] - p2["precision"], 4),
                 "d_recall": round(p1["recall"] - p2["recall"], 4),
                 "d_chair_i": round(p1["chair_i_check"] - p2["chair_i_check"], 4)}
            rows.append(d)
            d_f1.append(d["d_f1"])
            print("%-10s %10.4f %10.4f %10.4f %10.4f" %
                  (d["cell"], d["d_f1"], d["d_precision"], d["d_recall"], d["d_chair_i"]))
        n = len(d_f1)
        k_pos = sum(1 for x in d_f1 if x > 0)
        mean = sum(d_f1) / n
        sd = (sum((x - mean) ** 2 for x in d_f1) / (n - 1)) ** 0.5 if n > 1 else 0.0
        p = sign_test_p(k_pos, n)
        # CHAIR's own verdict on the same cells, to show whether the two metrics
        # disagree in DIRECTION -- the whole point of the decomposition.
        k_chair = sum(1 for r in rows if r["d_chair_i"] < 0)   # lower CHAIR = better
        print("\n  F1 favours %-8s in %d/%d cells   mean dF1 %+.4f (sd %.4f)  sign-test p=%s"
              % (a1, k_pos, n, mean, sd, p))
        print("  CHAIR favours %-8s in %d/%d cells  -- if these two counts disagree, "
              "CHAIR and F1\n  are ranking the arms in OPPOSITE directions on the same "
              "generations." % (a1, k_chair, n))
        report["pairs"]["%s|%s" % (a1, a2)] = {
            "n_cells": n, "f1_favours_%s" % a1: k_pos, "chair_favours_%s" % a1: k_chair,
            "mean_d_f1": round(mean, 4), "sd_d_f1": round(sd, 4), "sign_test_p": p,
            "per_cell": rows}

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
