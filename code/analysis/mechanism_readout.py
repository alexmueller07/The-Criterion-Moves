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
import fs_common

BASE_C = None  # filled from the aggregate; the untuned reference


def load_base(agg_path):
    d = json.load(open(agg_path))
    p = d["backbones"]["llava15"]["base"]["pope"]
    return p["c"], p["dprime"], p["f1"]


def pope_of(cell_dir):
    """Score POPE for one checkpoint dir from its raw generations.

    Cells store `pope_gen.jsonl` only -- the scored numbers live in the aggregate,
    which is refreshed by a separate job. Scoring here with the SAME audited
    scorer (fs_common.score_pope) keeps these readouts numerically identical to
    everything else the paper reports, and means a finished arm can be read the
    moment its eval lands rather than waiting on an aggregate pass.
    """
    p = os.path.join(cell_dir, "pope_gen.jsonl")
    if not os.path.isfile(p):
        return None
    try:
        return fs_common.score_pope(p)
    except Exception as e:                      # report, never crash the sweep
        print("  [warn] %s: %s: %s" % (cell_dir, type(e).__name__, e))
        return None


def cells_for(results, prefix):
    out = {}
    for d in sorted(glob.glob(os.path.join(results, prefix + "*"))):
        m = re.search(r"_k(\d+)$", d)
        if not m or not os.path.isdir(d):
            continue
        p = pope_of(d)
        if p:
            out[int(m.group(1))] = p
    return out


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
    report = {"base_c": base_c, "mode": args.mode, "rows": []}

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
            cs = cells_for(args.results, tag)
            if not cs:
                print("%-22s %7.2f   (not finished)" % (tag, words[k]))
                continue
            p = cs[max(cs)]
            dc = p["c"] - base_c
            print("%-22s %7.2f %+11.4f %10.4f %10.4f %10.4f" %
                  (tag, words[k], dc, p["c"], p["dprime"], p.get("f1", float("nan"))))
            got.append((words[k], dc))
            report["rows"].append({"arm": tag, "k": k, "words": words[k],
                                   "dc": round(dc, 4), "c": p["c"],
                                   "dprime": p["dprime"], "f1": p.get("f1")})
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
        print("%-22s %-22s %10s %11s %10s" %
              ("arm", "what adapts", "sum|dc|", "endpoint c", "end d'"))
        print("-" * 80)
        ref = None
        for tag, what in specs:
            cs = cells_for(args.results, tag)
            if not cs:
                print("%-22s %-22s   (not finished)" % (tag, what))
                continue
            ks = sorted(cs)
            prev = base_c; path = 0.0
            for k in ks:
                path += abs(cs[k]["c"] - prev); prev = cs[k]["c"]
            end = cs[ks[-1]]
            if ref is None:
                ref = path
            share = "" if ref in (None, 0) else "  (%+.0f%% vs baseline)" % (100*(path-ref)/ref)
            print("%-22s %-22s %10.4f %11.4f %10.4f%s" %
                  (tag, what, path, end["c"], end["dprime"], share))
            report["rows"].append({"arm": tag, "what": what, "sum_abs_dc": round(path, 4),
                                   "endpoint_c": end["c"], "endpoint_dprime": end["dprime"],
                                   "n_stages": len(ks)})
        print("\n  A freeze that removes most of the drift localises it. If every freeze")
        print("  leaves the drift intact, it is distributed and no single component")
        print("  carries it -- which is itself a reportable negative.")

    if args.out:
        json.dump(report, open(args.out, "w"), indent=2)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
