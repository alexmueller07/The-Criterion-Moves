#!/usr/bin/env python3
"""Length-controlled rescoring of the external-benchmark generative evals.

WHY THIS EXISTS (2026-09-09). The bench pipeline scores Object HalBench and the
non-COCO Open Images eval with pilot/metrics_chair.py, which emits RAW CHAIR_i
and CHAIR_s only. Raw CHAIR is not comparable across arms that emit captions of
different lengths, and our arms do differ sharply: on the non-COCO eval the
LLaVA base writes 104.2 tokens/caption, the anchor 52.1, sequential 26.5. Across
those cells CHAIR_s tracks mentions-per-caption almost monotonically, i.e. the
raw between-arm difference is substantially a length effect. Read naively the
anchor looks WORSE than sequential (0.3276 vs 0.2605) while producing twice the
text.

This script does NOT regenerate anything. It re-reads the existing *_gen.jsonl
files and rescores them with analysis/fs_common.py::score_chair -- the SAME
audited scorer, with the SAME k-word truncation, that produces the in-domain
CHAIR@60 numbers. So bench numbers become directly comparable to in-domain ones
and to each other.

It also reports the length distribution per cell so a reader can see how much
truncation the control is actually doing: a cell whose captions are already
under k words is UNAFFECTED by the control, and a comparison between one such
cell and a much longer one is still length-limited (the control can shorten a
long caption but cannot lengthen a short one). That asymmetry is printed
explicitly as `frac_over_budget`; when it is near zero for one arm and large for
another, say so rather than pretending the control equalized them.

Usage:
  python3 bench_lc_chair.py --results <results_fs> --gt_root <data/bench>
      [--k 60] [--out out.json]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fs_common  # noqa: E402

# (subdir of the bench cell, gen filename, GT filename relative to gt_root)
EVALS = [
    ("noncoco", "noncoco_gen.jsonl", "noncoco/gt.json"),
    ("objhal", "objhal_gen.jsonl", "objhal/coco_gt.json"),
]


def rescore_cell(cell_dir, gt_root, k, per_image=None):
    """Return {eval_name: metrics} for every bench eval present in cell_dir.

    When `per_image` is a dict it receives {eval_name: {coco_id: (ment@k, hal@k)}}
    for the paired bootstrap; the map is stripped from the returned metrics so
    the JSON report stays small.
    """
    out = {}
    for name, genfile, gtrel in EVALS:
        gen = os.path.join(cell_dir, genfile)
        gt = os.path.join(gt_root, gtrel)
        if not os.path.isfile(gen):
            continue
        if not os.path.isfile(gt):
            out[name] = {"error": "GT absent: %s" % gt}
            continue
        try:
            coco_gt = fs_common.load_coco_gt(gt)
            m = fs_common.score_chair(gen, coco_gt, k_words=k)
        except Exception as e:  # noqa: BLE001 - report, never crash the sweep
            out[name] = {"error": "%s: %s" % (type(e).__name__, e)}
            continue
        pk = m.pop("_per_image_k", None)
        if per_image is not None and pk:
            per_image[name] = pk
        out[name] = m
    return out


def paired_bootstrap(per_k_a, per_k_b, reps=4000, seed=17):
    """CI for CHAIR_i@k(A) - CHAIR_i@k(B) over the SHARED image set.

    Both cells answer the same prompts on the same images, so the correct
    resampling unit is the image, resampled WITH REPLACEMENT and kept as a LIST:
    a set() here would silently turn the resample into a ~63.2% subsample and
    shrink every interval (that bug cost this project nine wrong intervals
    once already -- see env_bootstrap_set_drops_duplicates).

    CHAIR_i is a RATIO of summed mentions, so each replicate re-sums numerator
    and denominator over the resampled ids rather than averaging per-image
    rates, which would weight a 1-mention caption like a 6-mention one.
    """
    import random
    ids = sorted(set(per_k_a) & set(per_k_b))
    if len(ids) < 30:
        return None
    rnd = random.Random(seed)
    obs = None
    diffs = []
    for r in range(reps + 1):
        pick = ids if r == 0 else [ids[rnd.randrange(len(ids))] for _ in ids]
        ma = ha = mb = hb = 0
        for cid in pick:                      # list, with multiplicity
            m, h = per_k_a[cid]
            ma += m
            ha += h
            m, h = per_k_b[cid]
            mb += m
            hb += h
        d = (ha / ma if ma else 0.0) - (hb / mb if mb else 0.0)
        if r == 0:
            obs = d
        else:
            diffs.append(d)
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs)) - 1]
    return {"delta": round(obs, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "n_images": len(ids), "reps": reps,
            "excludes_zero": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="results_fs root holding the *_bench cell directories")
    ap.add_argument("--gt_root", required=True,
                    help="bench data root (contains noncoco/gt.json, objhal/coco_gt.json)")
    ap.add_argument("--k", type=int, default=fs_common.CHAIR_K_WORDS,
                    help="word budget for the length control (default: the in-domain k)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pairs", default=None,
                    help="comma-separated cellA:cellB comparisons to bootstrap")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    cells = sorted(d for d in os.listdir(args.results)
                   if os.path.isdir(os.path.join(args.results, d))
                   and any(os.path.isfile(os.path.join(args.results, d, g))
                           for _, g, _ in EVALS))
    if not cells:
        print("NO BENCH GENERATION FILES under %s" % args.results)
        return 1

    report = {"k_words": args.k, "cells": {}}
    per_image = {}
    for c in cells:
        pi = {}
        report["cells"][c] = rescore_cell(os.path.join(args.results, c),
                                          args.gt_root, args.k, per_image=pi)
        per_image[c] = pi

    hdr = ("%-32s %-8s %8s %8s %8s %8s %8s %8s %8s" %
           ("cell", "eval", "raw_i", "lc_i@%d" % args.k, "raw_s",
            "ment/cap", "mnt@k/cap", "mean_wd", "over_bud"))
    print(hdr)
    print("-" * len(hdr))
    for c in cells:
        for name in ("noncoco", "objhal"):
            m = report["cells"][c].get(name)
            if not m:
                continue
            if "error" in m:
                print("%-32s %-8s  ERROR: %s" % (c[:32], name, m["error"][:70]))
                continue
            # mentions/caption AFTER the budget: the quantity that actually has
            # to match before two cells' CHAIR can be compared. Stored so the
            # matching check is machine-readable, not eyeballed off mean_wd.
            m["mentions_at_k_per_caption"] = round(
                m["n_mentions60"] / max(1, m["n_captions"]), 3)
            print("%-32s %-8s %8.4f %8.4f %8.4f %8.3f %8.3f %8.1f %8.4f" %
                  (c[:32], name, m["chair_i"], m["chair_i60"], m["chair_s"],
                   m["mentions_per_caption"], m["mentions_at_k_per_caption"],
                   m["mean_words"], m["frac_over_budget"]))

    print("\nREADING RULE: lc_i@%d is comparable across cells ONLY to the extent both "
          "cells actually exceed the budget.\nWhen one cell's over_bud is ~0 and "
          "another's is large, the control truncated only one of them and the\n"
          "comparison remains length-limited -- report mean_wd alongside every "
          "such comparison." % args.k)

    # Pairwise paired bootstrap. Each pair is only interpretable when the two
    # cells' mentions@k actually match; the mismatch ratio is printed so a
    # reader can reject a comparison rather than trusting the interval alone.
    if args.pairs:
        report["pairs"] = {}
        print("\n%-26s %-26s %-8s %8s %18s %9s %s" %
              ("A", "B", "eval", "dCHAIR", "95% CI", "mnt-ratio", "verdict"))
        for spec in args.pairs.split(","):
            if ":" not in spec:
                print("  bad --pairs entry %r (want A:B)" % spec)
                continue
            a, b = spec.split(":", 1)
            for name in ("noncoco", "objhal"):
                pa = per_image.get(a, {}).get(name)
                pb = per_image.get(b, {}).get(name)
                if not pa or not pb:
                    continue
                bs = paired_bootstrap(pa, pb, seed=args.seed)
                if bs is None:
                    continue
                ma = report["cells"][a][name].get("mentions_at_k_per_caption")
                mb = report["cells"][b][name].get("mentions_at_k_per_caption")
                ratio = (ma / mb) if (ma and mb) else float("nan")
                # A comparison whose opportunity counts differ by >10% is
                # length-limited, not a hallucination comparison.
                if abs(ratio - 1.0) > 0.10:
                    verdict = "LENGTH-LIMITED (do not report as an effect)"
                elif bs["excludes_zero"]:
                    verdict = "matched; CI excludes 0"
                else:
                    verdict = "matched; CI includes 0 (null)"
                bs["mentions_ratio"] = round(ratio, 3)
                bs["verdict"] = verdict
                report["pairs"]["%s|%s|%s" % (a, b, name)] = bs
                print("%-26s %-26s %-8s %8.4f  [%7.4f,%7.4f] %9.3f %s" %
                      (a[:26], b[:26], name, bs["delta"], bs["ci95"][0],
                       bs["ci95"][1], ratio, verdict))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
