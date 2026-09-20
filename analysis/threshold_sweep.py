#!/usr/bin/env python3
"""Empirical threshold sweep on per-item POPE logits: the assumption-free ceiling.

WHY (2026-09-09). The anchor's endpoint POPE F1 (0.7963) sits 6.6 points below
plain sequential (0.8625) in 9/9 cells with no overlap, and single-point d' is
flat across arms, which invites the conclusion that the deficit is pure threshold
placement and one scalar would recover it. Computing that recoverable amount
analytically gave OPPOSITE answers depending on an untestable assumption:

  equal-variance SDT, ceiling = Phi(d'/2):     anchor beats SEQ in 9/9 cells
  unequal-variance, z-ROC slope b = 0.55-0.72: anchor beats SEQ in 0/9 cells

so the analytic route is worthless here -- the assumption decides the result. This
script removes the model entirely. Given per-item (z_yes, z_no, gt) it sweeps the
decision threshold directly and reads off the best achievable balanced accuracy.
No Gaussians, no variance ratio, no d'.

  g_i    = z_yes_i - z_no_i          (the decision statistic actually used)
  pred_i = yes  iff  g_i > t
  BA(t)  = 0.5 * (TPR(t) + TNR(t))
  ceiling = max_t BA(t),  t* = argmax

t = 0 is the model's own operating point, so BA(0) is what the checkpoint
actually achieves and ceiling - BA(0) is exactly what a perfect scalar criterion
correction would buy. t* is the size of the correction needed.

POSITIVE CONTROL, not optional: the script recomputes hit and false-alarm rates
at t = 0 and prints them so they can be checked against the aggregate's recorded
H/FA for the same cell. If they disagree, this file is not the statistic the
paper's c and d' were computed from and the sweep is meaningless. A mismatch is
reported loudly rather than silently tolerated.

Usage:
  python3 threshold_sweep.py --dumps <cell>=<pope_logits.jsonl> [...] [--out out.json]
"""
import argparse
import json
import os
import sys


def load(path):
    """[(g, is_yes)] plus counts. Fails loud rather than skipping bad rows."""
    rows = []
    n_bad = 0
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            gt = r.get("gt")
            if gt not in ("yes", "no"):
                n_bad += 1
                continue
            if "z_yes" in r and "z_no" in r:
                g = float(r["z_yes"]) - float(r["z_no"])
            elif "gap" in r:
                g = float(r["gap"])
            else:
                raise SystemExit("%s:%d: no z_yes/z_no and no gap" % (path, ln))
            rows.append((g, gt == "yes"))
    if n_bad:
        raise SystemExit("%s: %d rows with unusable gt" % (path, n_bad))
    if not rows:
        raise SystemExit("%s: empty" % path)
    return rows


def sweep(rows):
    """Exact max over thresholds. O(n log n): sort once, walk the cut points.

    Thresholds are placed BETWEEN adjacent distinct g values, so every reachable
    partition is evaluated and the maximum is exact, not grid-approximated.
    """
    n_pos = sum(1 for _, y in rows if y)
    n_neg = len(rows) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise SystemExit("degenerate: one class absent (pos=%d neg=%d)"
                         % (n_pos, n_neg))
    srt = sorted(rows, key=lambda t: t[0])

    # Start with t below every g: everything predicted yes.
    tp, fp = n_pos, n_neg
    best = (0.5 * (tp / n_pos + 1 - fp / n_neg), float("-inf"))
    i = 0
    while i < len(srt):
        g = srt[i][0]
        # move every item with this exact g from "predicted yes" to "predicted no"
        while i < len(srt) and srt[i][0] == g:
            if srt[i][1]:
                tp -= 1
            else:
                fp -= 1
            i += 1
        t = g if i >= len(srt) else 0.5 * (g + srt[i][0])
        ba = 0.5 * (tp / n_pos + 1 - fp / n_neg)
        if ba > best[0]:
            best = (ba, t)
    # the model's own operating point
    tp0 = sum(1 for g, y in rows if y and g > 0)
    fp0 = sum(1 for g, y in rows if (not y) and g > 0)
    h0, fa0 = tp0 / n_pos, fp0 / n_neg
    return {"n": len(rows), "n_pos": n_pos, "n_neg": n_neg,
            "H_at_0": round(h0, 4), "FA_at_0": round(fa0, 4),
            "bal_at_0": round(0.5 * (h0 + 1 - fa0), 4),
            "bal_ceiling": round(best[0], 4),
            "t_star": round(best[1], 4),
            "recoverable": round(best[0] - 0.5 * (h0 + 1 - fa0), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--expect", nargs="*", default=[],
                    help="name=H,FA recorded elsewhere, for the positive control")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    expect = {}
    for spec in args.expect:
        name, hf = spec.split("=", 1)
        h, fa = hf.split(",")
        expect[name] = (float(h), float(fa))

    out = {}
    hdr = ("%-26s %7s %8s %8s %9s %11s %9s %9s" %
           ("cell", "n", "H@0", "FA@0", "bal@0", "bal ceiling", "gain", "t*"))
    print("Empirical threshold sweep on per-item POPE logits (no SDT assumption).")
    print("bal@0 is what the checkpoint achieves; ceiling is the best any scalar")
    print("criterion correction could reach; t* is the size of that correction.\n")
    print(hdr)
    print("-" * len(hdr))
    for spec in args.dumps:
        if "=" not in spec:
            raise SystemExit("bad --dumps entry %r" % spec)
        name, path = spec.split("=", 1)
        if not os.path.isfile(path):
            print("%-26s MISSING %s" % (name[:26], path))
            continue
        r = sweep(load(path))
        out[name] = r
        print("%-26s %7d %8.4f %8.4f %9.4f %11.4f %9.4f %9.3f" %
              (name[:26], r["n"], r["H_at_0"], r["FA_at_0"], r["bal_at_0"],
               r["bal_ceiling"], r["recoverable"], r["t_star"]))
        if name in expect:
            eh, efa = expect[name]
            dh, dfa = abs(r["H_at_0"] - eh), abs(r["FA_at_0"] - efa)
            ok = dh < 0.01 and dfa < 0.01
            print("      positive control vs recorded H=%.4f FA=%.4f: %s "
                  "(dH=%.4f dFA=%.4f)"
                  % (eh, efa, "MATCH" if ok else "*** MISMATCH -- this dump is "
                     "not the statistic behind the paper's c/d'; sweep is void ***",
                     dh, dfa))
            out[name]["control_match"] = bool(ok)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
