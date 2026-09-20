#!/usr/bin/env python3
"""Mention-index-matched hallucination rate: the opportunity control without a budget.

WHY (2026-09-09). analysis/bench_lc_chair.py showed that comparing CHAIR across
arms with different caption lengths is confounded by opportunity, badly enough to
invert the sign of the tuned-vs-base effect. Its fix -- truncate to a common word
budget k chosen so mentions-per-caption match -- works, but it has two defects:

  1. k is a researcher degree of freedom. We pick it to make mentions match, which
     is principled, but it is still a choice made after seeing the data.
  2. At the matched k (12 words) the base's captions are cut to ~15% of their
     length, so the comparison sees only the earliest mentions.

This script removes both. For each caption we walk its object mentions IN ORDER
and record whether mention #1, #2, #3 ... is hallucinated. Then we compare arms
AT THE SAME MENTION INDEX. Index i is by construction the same amount of
opportunity in every arm: every caption that produced an i-th mention gets
exactly one observation. No budget, no truncation, no post-hoc k.

This is the cleanest statement of the confound too: if raw CHAIR differences are
pure opportunity, then the per-index curves will coincide while the arms' MASS
over indices differs (a verbose arm simply reaches higher indices). If an arm
genuinely hallucinates more, its curve sits above the others at matched index.

Reads the same *_gen.jsonl files the bench already produced; regenerates nothing.

Usage:
  python3 mention_index_chair.py --results <results_fs_bench> --gt_root <data_ucit>
      [--eval noncoco|objhal] [--max_index 8] [--out out.json]
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fs_common  # noqa: E402

EVAL_FILES = {"noncoco": ("noncoco_gen.jsonl", "noncoco/gt.json"),
              "objhal": ("objhal_gen.jsonl", "objhal/coco_gt.json")}


def per_caption_flags(gen_path, coco_gt):
    """{coco_id: [0/1 per mention, in caption order]} — 1 = hallucinated.

    Uses the SAME extractor and the SAME ground-truth construction (instance
    objects UNION objects found in the GT captions) as fs_common.score_chair, so
    these numbers are directly comparable to the CHAIR the paper reports.
    """
    sc = fs_common.import_pilot_scorers()
    ext_m, ext_o = sc["extract_object_mentions"], sc["extract_objects"]
    out = {}
    gt_cache = {}
    for r in fs_common.read_rows_dedup(str(gen_path)):
        if "coco_id" not in r or "output" not in r:
            raise fs_common.MalformedResult(
                "%s: row missing 'coco_id'/'output' (id=%r)" % (gen_path, r.get("id")))
        cid = str(r["coco_id"])
        if cid not in coco_gt:
            raise fs_common.MalformedResult(
                "%s: coco_id %s absent from GT table" % (gen_path, cid))
        gto = gt_cache.get(cid)
        if gto is None:
            gto = set(coco_gt[cid]["objects"])
            for cap in coco_gt[cid].get("captions", []):
                gto |= ext_o(cap)
            gt_cache[cid] = gto
        # extract_object_mentions returns mentions in caption order; that order
        # is the whole point of this analysis, so do not sort or dedup it.
        out[cid] = [int(x not in gto) for x in ext_m(r["output"])]
    return out


def index_curve(flags, max_index, min_mentions=0):
    """[(index, n_captions, hallucination_rate), ...] (1-based).

    With min_mentions=0 the caption set SHRINKS as the index grows: only long
    captions reach index 5. That makes a rising curve ambiguous -- later indices
    could look worse merely because they are computed on longer captions, which
    tend to depict busier scenes. Passing min_mentions=M restricts every index to
    the FIXED set of captions with at least M mentions, so the curve is a
    within-caption hazard and the selection effect is gone.
    """
    if min_mentions:
        sub = {c: f for c, f in flags.items() if len(f) >= min_mentions}
    else:
        sub = flags
    rows = []
    for i in range(max_index):
        vals = [f[i] for f in sub.values() if len(f) > i]
        if not vals:
            break
        rows.append((i + 1, len(vals), sum(vals) / len(vals)))
    return rows


def boot_index_diff(fa, fb, index, reps=4000, seed=17):
    """CI for rate_A(index) - rate_B(index), resampling the SHARED images.

    Only images where BOTH arms produced a mention at this index are used, so
    the pairing is exact. The resample is a LIST with multiplicity; a set() here
    would silently make it a ~63% subsample and shrink the interval.
    """
    ids = sorted([c for c in (set(fa) & set(fb))
                  if len(fa[c]) > index - 1 and len(fb[c]) > index - 1])
    if len(ids) < 30:
        return None
    j = index - 1
    rnd = random.Random(seed)
    obs = None
    diffs = []
    for r in range(reps + 1):
        pick = ids if r == 0 else [ids[rnd.randrange(len(ids))] for _ in ids]
        sa = sum(fa[c][j] for c in pick)
        sb = sum(fb[c][j] for c in pick)
        d = (sa - sb) / len(pick)
        if r == 0:
            obs = d
        else:
            diffs.append(d)
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs)) - 1]
    return {"index": index, "n_paired": len(ids), "delta": round(obs, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--eval", default="noncoco", choices=sorted(EVAL_FILES))
    ap.add_argument("--max_index", type=int, default=8)
    ap.add_argument("--min_mentions", type=int, default=0,
                    help="also report a within-caption hazard on the fixed set of\ncaptions having at least this many mentions")
    ap.add_argument("--pairs", default=None, help="comma-separated cellA:cellB")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    genfile, gtrel = EVAL_FILES[args.eval]
    gt_path = os.path.join(args.gt_root, gtrel)
    if not os.path.isfile(gt_path):
        print("GT absent: %s" % gt_path)
        return 1
    coco_gt = fs_common.load_coco_gt(gt_path)

    cells, flags = [], {}
    for d in sorted(os.listdir(args.results)):
        gp = os.path.join(args.results, d, genfile)
        if os.path.isfile(gp):
            try:
                flags[d] = per_caption_flags(gp, coco_gt)
                cells.append(d)
            except Exception as e:  # noqa: BLE001
                print("SKIP %s: %s: %s" % (d, type(e).__name__, e))
    if not cells:
        print("no %s generations under %s" % (genfile, args.results))
        return 1

    report = {"eval": args.eval, "cells": {}, "pairs": {}}
    print("Hallucination rate by MENTION INDEX (%s). n = captions reaching that index.\n"
          "Index i is equal opportunity across arms by construction." % args.eval)
    hdr = "%-32s %s" % ("cell", " ".join("%12s" % ("#%d" % (i + 1))
                                         for i in range(args.max_index)))
    print("\n" + hdr)
    print("-" * len(hdr))
    for c in cells:
        curve = index_curve(flags[c], args.max_index)
        report["cells"][c] = [{"index": i, "n": n, "rate": round(r, 4)} for i, n, r in curve]
        cellstr = " ".join("%12s" % ("%.3f(%d)" % (r, n)) for _, n, r in curve)
        print("%-32s %s" % (c[:32], cellstr))

    if args.min_mentions:
        print("\nWITHIN-CAPTION hazard: fixed set of captions with >= %d mentions,\n"
              "so the caption set does NOT change across indices (removes the\n"
              "selection effect where long captions depict busier scenes)."
              % args.min_mentions)
        print("\n" + hdr)
        print("-" * len(hdr))
        report["within_caption"] = {"min_mentions": args.min_mentions, "cells": {}}
        for c in cells:
            curve = index_curve(flags[c], args.max_index, min_mentions=args.min_mentions)
            report["within_caption"]["cells"][c] = [
                {"index": i, "n": n, "rate": round(r, 4)} for i, n, r in curve]
            cellstr = " ".join("%12s" % ("%.3f(%d)" % (r, n)) for _, n, r in curve)
            print("%-32s %s" % (c[:32], cellstr))

    if args.pairs:
        print("\nPaired bootstrap at matched mention index (%d reps):" % 4000)
        print("%-24s %-24s %5s %7s %8s %18s %s" %
              ("A", "B", "idx", "n_pair", "delta", "95% CI", "verdict"))
        for spec in args.pairs.split(","):
            if ":" not in spec:
                continue
            a, b = spec.split(":", 1)
            if a not in flags or b not in flags:
                print("  missing cell in pair %r" % spec)
                continue
            for i in range(1, args.max_index + 1):
                bs = boot_index_diff(flags[a], flags[b], i, seed=args.seed)
                if bs is None:
                    break
                report["pairs"].setdefault("%s|%s" % (a, b), []).append(bs)
                print("%-24s %-24s %5d %7d %8.4f  [%7.4f,%7.4f] %s" %
                      (a[:24], b[:24], i, bs["n_paired"], bs["delta"],
                       bs["ci95"][0], bs["ci95"][1],
                       "excludes 0" if bs["excludes_zero"] else "null"))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
