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
    for cat in STRATA:
        paths, absc, dps, ends, dranges = [], [], [], [], []
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
