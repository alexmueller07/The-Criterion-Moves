#!/usr/bin/env python3
"""Does the criterion account hold in all three POPE regimes, or only one?

WHY (2026-09-11). Every number in the paper's measurement comes from POPE. The
obvious reviewer question is whether the criterion story is an artifact of one
benchmark, or of one way of choosing negatives. We cannot answer the first for
free, but we can answer the second, because POPE is not one benchmark: it is
three, differing precisely in how the ABSENT objects are chosen.

  random       an absent object sampled uniformly
  popular      an absent object sampled from the most frequent classes
  adversarial  an absent object sampled from those that most often CO-OCCUR with
               what is actually in the image

Those three make an increasingly hard demand on grounding while leaving the task,
the images and the prompt identical. If criterion drift were an artifact of easy
negatives it would weaken across the strata; if it is a property of the decision
rule it should persist, and d' should fall with difficulty while c does not track
that fall.

WARNING (2026-09-12): COMPARING d' AND c AS *LEVELS* ACROSS THE STRATA IS
ARITHMETIC, NOT A RESULT. The three strata share their POSITIVE items, so the hit
rate is constant across them (0.8480/0.8480/0.8467). With z(H) fixed, both indices
are affine in z(FA) alone, which forces

    delta d' = 2 * delta c    exactly, for any model.

Measured on our own 9-cell seq readout, random -> adversarial: delta d' = -0.6816
vs 2*delta c = -0.6820, residual +0.0004. So a "double dissociation" read off the
level columns is guaranteed by POPE's construction and is evidence for nothing.
Only the PATH column (mean_path_c) is a second difference and is not forced; that
is the one the paper may use. Note also that popular and adversarial share their
negative object set almost completely, so these are NOT three independent
conditions and must not be treated as n = 3.

This is a stratified re-analysis of generations we already have: no GPU, no new
runs. It reads pope_gen.jsonl, which carries a per-row `category`.

Usage:
  python3 pope_strata.py [--results <results_fs>] [--arm seq] [--out out.json]
"""
import argparse
import collections
import glob
import json
import math
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fs_common

STRATA = ("random", "popular", "adversarial")


def strat_sdt(path):
    """{category: sdt dict} from one pope_gen.jsonl.

    Uses fs_common.sdt_from_counts so the loglinear edge correction and the
    clipping flag are identical to every other number the paper reports; only
    the row subset differs.
    """
    hits = collections.Counter(); nyes = collections.Counter()
    fas = collections.Counter(); nno = collections.Counter()
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        cat = r.get("category")
        gt = r.get("gt")
        if cat not in STRATA or gt not in ("yes", "no"):
            continue
        out = (r.get("output") or "").strip().lower()
        said_yes = out.startswith("yes")
        if not (said_yes or out.startswith("no")):
            continue                      # unparseable: excluded, as elsewhere
        if gt == "yes":
            nyes[cat] += 1; hits[cat] += said_yes
        else:
            nno[cat] += 1; fas[cat] += said_yes
    out = {}
    for cat in STRATA:
        if nyes[cat] and nno[cat]:
            d, c, clipped = fs_common.sdt_from_counts(hits[cat], nyes[cat],
                                                      fas[cat], nno[cat])
            out[cat] = {"dprime": d, "c": c, "clipped": clipped,
                        "H": hits[cat]/nyes[cat], "FA": fas[cat]/nno[cat],
                        "n": nyes[cat]+nno[cat]}
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/home/alexmueller/cl-halluc/results_fs")
    ap.add_argument("--arm", default="seq")
    ap.add_argument("--prefix", default="fsL")
    ap.add_argument("--base", default=None, help="base-model cell dir name")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pat = re.compile(r"^%s_%s_o(\d+)_s(\d+)_k(\d+)$"
                     % (re.escape(args.prefix), re.escape(args.arm)))
    cells = collections.defaultdict(dict)
    for d in sorted(glob.glob(os.path.join(args.results, "*"))):
        m = pat.match(os.path.basename(d))
        if not m:
            continue
        p = os.path.join(d, "pope_gen.jsonl")
        if os.path.isfile(p):
            cells[(int(m.group(1)), int(m.group(2)))][int(m.group(3))] = strat_sdt(p)
    if not cells:
        print("no %s_%s cells under %s" % (args.prefix, args.arm, args.results))
        return 1

    # base reference, if present
    base = None
    bdir = os.path.join(args.results, args.base or "llava15_base")
    if os.path.isfile(os.path.join(bdir, "pope_gen.jsonl")):
        base = strat_sdt(os.path.join(bdir, "pope_gen.jsonl"))
        print("untuned base by stratum:")
        for cat in STRATA:
            if cat in base:
                print("  %-12s c = %+.4f   d' = %.4f   (n=%d)"
                      % (cat, base[cat]["c"], base[cat]["dprime"], base[cat]["n"]))
        print()

    print("Criterion drift and discriminability, per POPE stratum, arm=%s, %d cells\n"
          % (args.arm, len(cells)))
    print("%-13s %10s %10s %10s %10s %9s" %
          ("stratum", "mean |c|", "sum|dc|", "mean d'", "end c", "d' range"))
    print("-" * 68)
    report = {"arm": args.arm, "n_cells": len(cells), "strata": {}}
    # Per-cell values, kept so downstream figures can show dispersion instead of
    # three bare point estimates. The mean_* keys above them are byte-identical
    # to the version first committed -- the accumulation order is unchanged and
    # nothing below feeds back into them.
    percell = {}
    for cat in STRATA:
        paths, absc, dps, ends, dranges = [], [], [], [], []
        cell_ids = []
        for cell, stages in cells.items():
            ks = sorted(stages)
            series = [stages[k][cat] for k in ks if cat in stages[k]]
            if len(series) < 4:
                continue
            cs = [x["c"] for x in series]; ds = [x["dprime"] for x in series]
            start = base[cat]["c"] if base and cat in base else cs[0]
            paths.append(sum(abs(b - a) for a, b in zip([start] + cs, cs)))
            absc.append(st.mean(abs(x) for x in cs))
            dps.append(st.mean(ds)); ends.append(cs[-1])
            dranges.append(max(ds) - min(ds))
            cell_ids.append("o%d/s%d" % cell)
        if not paths:
            continue
        print("%-13s %10.4f %10.4f %10.4f %10.4f %9.4f" %
              (cat, st.mean(absc), st.mean(paths), st.mean(dps),
               st.mean(ends), st.mean(dranges)))
        report["strata"][cat] = {
            "n": len(paths), "mean_abs_c": round(st.mean(absc), 4),
            "mean_path_c": round(st.mean(paths), 4),
            "mean_dprime": round(st.mean(dps), 4),
            "mean_endpoint_c": round(st.mean(ends), 4),
            "mean_dprime_range": round(st.mean(dranges), 4)}
        percell[cat] = {"cells": cell_ids, "abs_c": absc, "path_c": paths,
                        "dprime": dps, "endpoint_c": ends,
                        "dprime_range": dranges}
        # SE of the mean over cells. Cells are the unit; there is no bootstrap
        # here on purpose (a resample-with-multiplicity bug has bitten this
        # project before, and at n=9 the SE and the sign counts below say
        # everything a bootstrap would).
        report["strata"][cat]["se"] = {
            k: round(st.stdev(v) / math.sqrt(len(v)), 4) if len(v) > 1 else None
            for k, v in (("abs_c", absc), ("path_c", paths), ("dprime", dps),
                         ("endpoint_c", ends), ("dprime_range", dranges))}
        report["strata"][cat]["per_cell"] = {
            k: [round(x, 4) for x in v]
            for k, v in percell[cat].items() if k != "cells"}
        report["strata"][cat]["per_cell"]["cells"] = cell_ids

    # PAIRED across strata. The three strata are scored on the SAME cells and
    # the same rows differ only by which negatives are drawn, so every
    # cross-stratum comparison is paired and the paired contrast is far more
    # informative than three independent means. `consistent` counts cells whose
    # difference runs the way the stratum ordering predicts (easier negatives
    # give the larger value) -- the house convention for n<=9.
    pairs = (("random", "popular"), ("popular", "adversarial"),
             ("random", "adversarial"))
    report["paired"] = {}
    for metric in ("dprime", "path_c", "abs_c", "endpoint_c"):
        if not all(c in percell for c in STRATA):
            break
        report["paired"][metric] = {}
        for a, b in pairs:
            va, vb = percell[a][metric], percell[b][metric]
            if len(va) != len(vb):
                continue
            diffs = [x - y for x, y in zip(va, vb)]
            report["paired"][metric]["%s-%s" % (a, b)] = {
                "n": len(diffs), "mean_diff": round(st.mean(diffs), 4),
                "se_diff": (round(st.stdev(diffs) / math.sqrt(len(diffs)), 4)
                            if len(diffs) > 1 else None),
                "n_positive": sum(1 for d in diffs if d > 0),
                "per_cell": [round(d, 4) for d in diffs]}

    s = report["strata"]
    if len(s) == 3:
        print("\nReading:")
        dps = [s[c]["mean_dprime"] for c in STRATA]
        print("  d' falls with negative difficulty: %.3f -> %.3f -> %.3f  (%s)"
              % (*dps, "as designed" if dps[0] >= dps[2] else "NOT as designed"))
        paths = [s[c]["mean_path_c"] for c in STRATA]
        spread = max(paths) - min(paths)
        print("  criterion path length across strata: %.3f / %.3f / %.3f  (spread %.3f)"
              % (*paths, spread))
        print("\n  If the criterion moves in ALL THREE strata while d' separates them,")
        print("  the drift is a property of the decision rule and not of how the")
        print("  negatives were drawn. If it appears only for easy negatives, it is")
        print("  an artifact of the sampling scheme and the paper must say so.")

    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
