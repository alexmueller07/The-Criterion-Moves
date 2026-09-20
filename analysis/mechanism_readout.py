#!/usr/bin/env python3
"""Readout for the two mechanism experiments, so results are immediate when arms land.

Both experiments ask WHERE the criterion drift comes from, and both are scored
against the same reference: the untuned base criterion and the standard sequential
arm. Reading results_fs directly rather than waiting on an aggregate refresh.

  --mode dose
      The answer-format dose-response. Single-stage arms trained on Flickr30k
      truncated to k words (k = 1, 2, 4, 8, untruncated). Same images, same
      prompts, same row count -- only answer length varies, so length is decoupled
      from task identity, which it is not across the six UCIT tasks.

      PRE-REGISTERED PREDICTION (recorded before the runs, in
      data_ucit/tasks/Flickr30k_lengthdose_report.json): the criterion shift is
      MONOTONE in k, negative at k=1 and positive untruncated. A flat profile
      refutes the format account, and that outcome is reported as a refutation
      rather than reinterpreted.

  --mode loc
      The drift-localization ablation. Sequential arms that differ only in which
      parameters may adapt: locproj freezes the multimodal projector, locearly
      adapts decoder layers 0-15, loclate adapts 16-31. Early and late adapt the
      SAME number of modules, so that contrast isolates location, not capacity.
      The LM head is frozen in every arm including the baseline, so the head's
      weights are already excluded as an explanation.

      Read as: how much of the baseline's drift does each freeze remove?

Usage:
  python3 mechanism_readout.py --mode dose [--results <results_fs>]
  python3 mechanism_readout.py --mode loc  [--results <results_fs>]
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The reference for criterion PLACEMENT. JOINT's empirical endpoint criterion,
# adopted 2026-09-11 when "c = 0 is derivable" was retracted (it holds only under
# equal variance; our fitted z-ROC slopes are 0.55-0.80). Ranking placement by
# closeness to zero instead of to C_STAR reversed the localization verdict on
# 2026-09-12 and had to be withdrawn -- see the prereg entries for both dates.
C_STAR = 0.0878
N_STAGES_FULL = 6
# Full POPE row count for one cell. Checked against the aggregate before being
# hard-coded: all 246 scored POPE blocks in fs_aggregate.json carry n_total = 9000.
N_POPE_TOTAL = 9000

import fs_common

BASE_C = None  # filled from the aggregate; the untuned reference


def load_base(agg_path):
    d = json.load(open(agg_path))
    p = d["backbones"]["llava15"]["base"]["pope"]
    return p["c"], p["dprime"], p["f1"]


def pope_of(cell_dir):
    """Score POPE for one checkpoint dir, REFUSING any cell that is not fully
    evaluated. Returns (scored, None) for a usable cell, (None, reason) otherwise.

    Cells store `pope_gen.jsonl` only -- the scored numbers live in the aggregate,
    which is refreshed by a separate job. Scoring here with the SAME audited
    scorer (fs_common.score_pope) keeps these readouts numerically identical to
    everything else the paper reports, and means a finished arm can be read the
    moment its eval lands rather than waiting on an aggregate pass.

    2026-09-13: completeness is now checked on the cell itself rather than on a
    proxy for it, and BOTH conditions are required because neither implies the
    other:

      * `EVAL_DONE` exists -- the eval job reached its end. This is the same
        marker `fs_aggregate.py` gates on, so the two agree on what "done" means.
      * the scored `n_total` == N_POPE_TOTAL -- the file the job left behind
        actually holds the whole POPE set. A marker written beside a truncated or
        still-growing dump passes the first check and fails this one.

    A cell failing either condition is NOT a smaller measurement of the same
    quantity; it is a different quantity, and it may not be pathed or averaged
    with the rest. This is the 2026-09-12 failure: a partially evaluated
    `locproj` arm was pathed over fewer transitions, which is mechanically
    smaller, and so impersonated a 71% drift reduction.
    """
    if not os.path.isdir(cell_dir):
        return None, "not a directory"
    if not os.path.isfile(os.path.join(cell_dir, "EVAL_DONE")):
        return None, "no EVAL_DONE marker -- eval unfinished or still running"
    p = os.path.join(cell_dir, "pope_gen.jsonl")
    if not os.path.isfile(p):
        return None, "EVAL_DONE present but pope_gen.jsonl is absent"
    try:
        sc = fs_common.score_pope(p)
    except Exception as e:                      # report, never crash the sweep
        return None, "unscorable -- %s: %s" % (type(e).__name__, e)
    n = sc.get("n_total")
    if n != N_POPE_TOTAL:
        return None, ("n_total = %r, need %d -- pope_gen.jsonl is truncated or "
                      "still being written (EVAL_DONE is present, so the marker "
                      "alone would have passed this cell)" % (n, N_POPE_TOTAL))
    return sc, None


def cells_for(results, prefix):
    """-> ({k: scored}, [(cell, reason), ...]) for one arm.

    Refusals are RETURNED, never swallowed. Every caller prints them: a cell
    dropped silently is exactly how a 5/6 arm reads as a finished one, since the
    only trace it leaves is a stage count nobody was shown.
    """
    out = {}
    skipped = []
    for d in sorted(glob.glob(os.path.join(results, prefix + "*"))):
        m = re.search(r"_k(\d+)$", d)
        if not m or not os.path.isdir(d):
            continue
        p, why = pope_of(d)
        if p is None:
            skipped.append((os.path.basename(d), why))
            continue
        out[int(m.group(1))] = p
    return out, skipped


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("dose", "loc"))
    ap.add_argument("--results", default="/home/alexmueller/cl-halluc/results_fs")
    ap.add_argument("--agg", default=None,
                    help="fs_aggregate.json for the untuned base reference; "
                         "defaults to the repo copy, then the cluster copy")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    agg = args.agg
    if agg is None:
        for cand in (os.path.join(here, "readout", "fs_aggregate.json"),
                     "/home/alexmueller/cl-halluc/readout/fs_aggregate.json"):
            if os.path.isfile(cand):
                agg = cand; break
    if not agg or not os.path.isfile(agg):
        raise SystemExit("cannot find fs_aggregate.json for the base reference; pass --agg")
    base_c, base_dp, base_f1 = load_base(agg)
    print("untuned base:  c = %+.4f   d' = %.4f   F1 = %.4f\n" % (base_c, base_dp, base_f1))
    report = {"base_c": base_c, "mode": args.mode,
              "n_pope_total_required": N_POPE_TOTAL,
              "n_stages_full": N_STAGES_FULL,
              "rows": [], "refused_cells": []}
    refused = report["refused_cells"]   # (arm, cell, reason), printed again at the end

    if args.mode == "dose":
        # k -> mean answer words, from the builder's own report where available
        words = {1: 1.00, 2: 2.00, 4: 4.00, 8: 7.74, 0: 12.29}   # 0 = untruncated
        specs = [("fsL_dosew1_o1_s17", 1), ("fsL_dosew2_o1_s17", 2),
                 ("fsL_dosew4_o1_s17", 4), ("fsL_dosew8_o1_s17", 8),
                 ("fsL_dosefull_o1_s17", 0)]
        print("Answer-length dose-response. One stage on Flickr30k, truncated to k words.")
        print("Prediction (pre-registered): dc monotone in length, negative at k=1.\n")
        print("%-22s %7s %11s %10s %10s %10s" %
              ("arm", "words", "dc vs base", "c", "d'", "F1"))
        print("-" * 76)
        got = []
        for tag, k in specs:
            cs, skipped = cells_for(args.results, tag)
            refused += [{"arm": tag, "cell": c, "reason": w} for c, w in skipped]
            if not cs:
                print("%-22s %7.2f   (not finished)" % (tag, words[k]))
                for c, w in skipped:
                    print("    !! REFUSED %s: %s" % (c, w))
                continue
            p = cs[max(cs)]
            dc = p["c"] - base_c
            print("%-22s %7.2f %+11.4f %10.4f %10.4f %10.4f" %
                  (tag, words[k], dc, p["c"], p["dprime"], p.get("f1", float("nan"))))
            for c, w in skipped:
                print("    !! REFUSED %s: %s" % (c, w))
            got.append((words[k], dc))
            report["rows"].append({"arm": tag, "k": k, "words": words[k],
                                   "dc": round(dc, 4), "c": p["c"],
                                   "dprime": p["dprime"], "f1": p.get("f1"),
                                   "n_refused_cells": len(skipped)})
        if len(got) >= 3:
            got.sort()
            xs = [g[0] for g in got]; ys = [g[1] for g in got]
            n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
            sxy = sum((a-mx)*(b-my) for a, b in zip(xs, ys))
            sxx = sum((a-mx)**2 for a in xs); syy = sum((b-my)**2 for b in ys)
            r = sxy/((sxx*syy)**0.5) if sxx > 0 and syy > 0 else float("nan")
            mono = all(ys[i] <= ys[i+1] for i in range(len(ys)-1))
            print("\n  Pearson r(words, dc) = %.4f over %d points" % (r, n))
            print("  monotone increasing in length: %s" % ("YES" if mono else "NO"))
            print("\n  VERDICT: %s" % (
                "consistent with the format account (monotone, positive slope)"
                if mono and r > 0 else
                "NOT monotone -- the pre-registered prediction fails and the format "
                "account is refuted, not reinterpreted"))
            report["pearson_r"] = round(r, 4); report["monotone"] = bool(mono)

    else:
        specs = [("fsL_seq_o1_s17", "baseline (all adapt)"),
                 ("fsL_locproj_o1_s17", "projector FROZEN"),
                 ("fsL_locearly_o1_s17", "layers 0-15 only"),
                 ("fsL_loclate_o1_s17", "layers 16-31 only")]
        print("Drift localization. Same sequence, different parameters allowed to adapt.")
        print("The LM head is frozen in EVERY arm, so its weights are already excluded.\n")
        # 2026-09-13: STAGE COUNT IS NOW MANDATORY IN THE OUTPUT. The first
        # localization entry was written from an arm that was still in eval:
        # this table printed a number with no indication that fewer than six
        # stages existed, and sum|dc| over fewer transitions is mechanically
        # smaller, so a truncated run impersonated a large drift reduction.
        # Placement is scored as |c - C_STAR|, never as closeness to zero --
        # c = 0 is optimal only under equal variance and this project's fitted
        # z-ROC slopes are 0.55-0.80. Ranking by |c| reversed this arm's verdict
        # once already.
        print("%-22s %-22s %6s %10s %11s %10s %10s" %
              ("arm", "what adapts", "n_stg", "sum|dc|", "endpoint c",
               "|c - c*|", "end d'"))
        print("-" * 96)
        ref = None
        ref_partial = False
        for i, (tag, what) in enumerate(specs):
            cs, skipped = cells_for(args.results, tag)
            refused += [{"arm": tag, "cell": c, "reason": w} for c, w in skipped]
            if not cs:
                print("%-22s %-22s   (not finished)" % (tag, what))
                for c, w in skipped:
                    print("    !! REFUSED %s: %s" % (c, w))
                if i == 0:
                    ref_partial = True
                continue
            ks = sorted(cs)
            prev = base_c; path = 0.0
            for k in ks:
                path += abs(cs[k]["c"] - prev); prev = cs[k]["c"]
            end = cs[ks[-1]]
            # The share reference is the BASELINE arm and only when it is 6/6.
            # A path over fewer transitions is mechanically smaller, so a partial
            # baseline would inflate every "% vs baseline" beneath it -- the same
            # arithmetic that produced the 2026-09-12 entry, one level up. With a
            # complete baseline this is exactly the previous behaviour.
            if i == 0:
                if len(ks) >= N_STAGES_FULL:
                    ref = path
                else:
                    ref_partial = True
            share = "" if ref in (None, 0) else "  (%+.0f%% vs baseline)" % (100*(path-ref)/ref)
            partial = "  << PARTIAL, NOT QUOTABLE" if len(ks) < N_STAGES_FULL else ""
            place = abs(end["c"] - C_STAR)
            print("%-22s %-22s %6s %10.4f %11.4f %10.4f %10.4f%s%s" %
                  (tag, what, "%d/%d" % (len(ks), N_STAGES_FULL), path, end["c"],
                   place, end["dprime"], share, partial))
            for c, w in skipped:
                print("    !! REFUSED %s: %s" % (c, w))
            report["rows"].append({"arm": tag, "what": what, "sum_abs_dc": round(path, 4),
                                   "endpoint_c": end["c"],
                                   "placement_err_vs_cstar": round(place, 4),
                                   "endpoint_dprime": end["dprime"],
                                   "n_stages": len(ks),
                                   "complete": len(ks) >= N_STAGES_FULL,
                                   # The per-stage vectors, which this loop used to
                                   # compute and throw away. per_stage_k labels them:
                                   # without it a short vector is ambiguous about
                                   # WHICH stage is missing, and the pre-registered
                                   # post-settling path is defined by dropping the
                                   # FIRST step, so the stage index is load-bearing.
                                   "per_stage_k": ks,
                                   "per_stage_c": [round(cs[k]["c"], 4) for k in ks],
                                   "per_stage_dprime": [round(cs[k]["dprime"], 4) for k in ks],
                                   "refused_cells": [{"cell": c, "reason": w}
                                                     for c, w in skipped]})
        if ref_partial:
            print("\n  !! the BASELINE arm is not %d/%d, so '%% vs baseline' is"
                  " suppressed: a" % (N_STAGES_FULL, N_STAGES_FULL))
            print("  !! path over fewer transitions is mechanically smaller and would")
            print("  !! inflate every share computed against it.")
        inc = [r["arm"] for r in report["rows"] if not r.get("complete")]
        if inc:
            print("\n  !! %d arm(s) are INCOMPLETE and their numbers may not be quoted: %s"
                  % (len(inc), ", ".join(inc)))
            print("  !! sum|dc| over fewer transitions is mechanically smaller, so a")
            print("  !! truncated arm impersonates a drift reduction. Wait for %d/%d."
                  % (N_STAGES_FULL, N_STAGES_FULL))
        print("\n  Placement is |c - c*| with c* = %.4f (JOINT's empirical endpoint)." % C_STAR)
        print("\n  A freeze that removes most of the drift localises it. If every freeze")
        print("  leaves the drift intact, it is distributed and no single component")
        print("  carries it -- which is itself a reportable negative.")

    # Every refused cell is restated here, in BOTH modes, so the run cannot end
    # with a clean-looking table above a silently shortened arm.
    if refused:
        print("\n" + "!" * 78)
        print("%d CELL(S) REFUSED as not fully evaluated. They are NOT in any number above."
              % len(refused))
        print("A refused cell is not a smaller measurement of the same quantity, so an")
        print("arm missing one is short by a stage, not slightly noisier.")
        print("!" * 78)
        for r in refused:
            print("  %-22s %-28s %s" % (r["arm"], r["cell"], r["reason"]))

    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
