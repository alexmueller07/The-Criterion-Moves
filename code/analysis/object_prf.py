#!/usr/bin/env python3
"""Object-mention PRECISION, RECALL and F1 for generative hallucination evals.

WHY (2026-09-09). CHAIR_i is (hallucinated mentions / total mentions), i.e.
exactly 1 - PRECISION over object mentions. It has no recall term. Nothing in it
rewards a model for mentioning the objects that are actually present, so the
metric is trivially gamed by saying less: a caption that names one obvious object
and stops scores a near-perfect CHAIR.

Our arms do exactly this. On the non-COCO eval the LLaVA base writes 81 words and
6.28 object mentions per caption; after sequential continual tuning the same model
writes 21 words and 1.81 mentions. Raw CHAIR "improves" from 0.340 to 0.261 --
and analysis/bench_lc_chair.py showed that once opportunity is matched the sign of
that comparison REVERSES. The degenerate direction is being scored as progress.

So we report the full picture on the same generations:
  precision = 1 - CHAIR_i                    (what CHAIR already measures)
  recall    = |mentioned objects ∩ GT| / |GT| (what CHAIR omits)
  F1        = harmonic mean

Recall is computed over the SET of distinct objects per image (an object named
three times is one object), while precision stays mention-weighted to match the
CHAIR convention exactly -- so the precision column here reproduces 1 - chair_i
from fs_common.score_chair rather than silently redefining it.

Reads the *_gen.jsonl the bench already produced. Regenerates nothing.

Usage:
  python3 object_prf.py --results <results_fs_bench> --gt_root <data_ucit>
      [--eval noncoco] [--pairs A:B,...] [--out out.json]
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


def per_image_counts(gen_path, coco_gt):
    """{coco_id: (n_mentions, n_hal_mentions, n_gt, n_gt_hit)}.

    GT per image is built exactly as fs_common.score_chair builds it: instance
    objects UNION objects the same extractor finds in the reference captions.
    Using a different GT here would make precision disagree with the CHAIR the
    rest of the paper reports.
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
        ments = ext_m(r["output"])                      # ordered, with repeats
        hal = sum(1 for x in ments if x not in gto)
        hit = len(set(ments) & gto)                     # distinct objects found
        out[cid] = (len(ments), hal, len(gto), hit)
    return out


def aggregate(counts):
    """Corpus-level precision/recall/F1 (pooled, not macro-averaged).

    Pooled so that an image with 8 GT objects weighs more than one with 1, and so
    precision matches CHAIR's own pooled definition.
    """
    m = h = g = k = 0
    for nm, nh, ng, nk in counts.values():
        m += nm
        h += nh
        g += ng
        k += nk
    prec = 1.0 - (h / m) if m else 0.0
    rec = (k / g) if g else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "n_images": len(counts), "n_mentions": m, "n_hal": h,
            "n_gt_objects": g, "n_gt_hit": k,
            "mentions_per_caption": round(m / max(1, len(counts)), 3),
            "chair_i_check": round(h / m, 4) if m else None}


def boot_f1_diff(ca, cb, reps=4000, seed=17):
    """CI for F1(A) - F1(B) over shared images. LIST resample, with multiplicity."""
    ids = sorted(set(ca) & set(cb))
    if len(ids) < 30:
        return None
    rnd = random.Random(seed)
    obs = None
    diffs = []

    def f1_of(pick, c):
        m = h = g = k = 0
        for cid in pick:
            nm, nh, ng, nk = c[cid]
            m += nm
            h += nh
            g += ng
            k += nk
        p = 1.0 - (h / m) if m else 0.0
        r = (k / g) if g else 0.0
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    for rep in range(reps + 1):
        pick = ids if rep == 0 else [ids[rnd.randrange(len(ids))] for _ in ids]
        d = f1_of(pick, ca) - f1_of(pick, cb)
        if rep == 0:
            obs = d
        else:
            diffs.append(d)
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs)) - 1]
    return {"delta_f1": round(obs, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "n_images": len(ids), "excludes_zero": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--gt_root", required=True)
    ap.add_argument("--eval", default="noncoco", choices=sorted(EVAL_FILES))
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    genfile, gtrel = EVAL_FILES[args.eval]
    gt_path = os.path.join(args.gt_root, gtrel)
    if not os.path.isfile(gt_path):
        print("GT absent: %s" % gt_path)
        return 1
    coco_gt = fs_common.load_coco_gt(gt_path)

    cells, counts = [], {}
    for d in sorted(os.listdir(args.results)):
        gp = os.path.join(args.results, d, genfile)
        if os.path.isfile(gp):
            try:
                counts[d] = per_image_counts(gp, coco_gt)
                cells.append(d)
            except Exception as e:  # noqa: BLE001
                print("SKIP %s: %s: %s" % (d, type(e).__name__, e))
    if not cells:
        print("no %s generations under %s" % (genfile, args.results))
        return 1

    report = {"eval": args.eval, "cells": {}, "pairs": {}}
    hdr = ("%-32s %9s %9s %9s %9s %9s" %
           ("cell", "precision", "recall", "F1", "ment/cap", "CHAIR_i"))
    print("Object-mention precision / recall / F1 (%s).\n"
          "CHAIR reports precision ONLY; recall is the axis it omits." % args.eval)
    print("\n" + hdr)
    print("-" * len(hdr))
    for c in cells:
        a = aggregate(counts[c])
        report["cells"][c] = a
        print("%-32s %9.4f %9.4f %9.4f %9.3f %9.4f" %
              (c[:32], a["precision"], a["recall"], a["f1"],
               a["mentions_per_caption"], a["chair_i_check"]))

    if args.pairs:
        print("\nPaired bootstrap on F1 (4000 reps, shared images):")
        print("%-26s %-26s %9s %20s %s" % ("A", "B", "dF1", "95% CI", "verdict"))
        for spec in args.pairs.split(","):
            if ":" not in spec:
                continue
            a, b = spec.split(":", 1)
            if a not in counts or b not in counts:
                print("  missing cell in %r" % spec)
                continue
            bs = boot_f1_diff(counts[a], counts[b], seed=args.seed)
            if bs is None:
                continue
            report["pairs"]["%s|%s" % (a, b)] = bs
            print("%-26s %-26s %9.4f  [%7.4f,%7.4f] %s" %
                  (a[:26], b[:26], bs["delta_f1"], bs["ci95"][0], bs["ci95"][1],
                   "excludes 0" if bs["excludes_zero"] else "null"))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
