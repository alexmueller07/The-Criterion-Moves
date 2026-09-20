#!/usr/bin/env python3
"""Polarity-antisymmetry diagnostic: is there a LABEL-FREE estimate of the criterion?

WHY (2026-09-09). The full study established that our anchor reduces criterion
DRIFT but lands at c = +0.697 while the Bayes-optimal point on balanced POPE is
c = 0, costing 6.6 POPE F1 points relative to running no method at all (9/9 cells,
no overlap). The diagnosis is that the anchor and critp both regularize toward the
FROZEN BASE, and the base itself sits at c = +0.431. They inherit a mis-placed
target. Fixing that needs a target that is correct by construction and estimable
WITHOUT labels, since the continual stream's new tasks carry no POPE-style
annotation.

THE IDEA. Write the model's decision statistic for probe item i as
    g_i = z(" Yes") - z(" No")
Ask the same item in native polarity and in flipped polarity. If the model is
logically consistent, the truth signal t_i reverses while any polarity-independent
answer bias b does not:
    g_i^native  =  t_i + b
    g_i^flipped = -t_i + b
so their mean is the bias alone:
    r_i = (g_i^native + g_i^flipped) / 2  =  b        <- NO LABELS USED
This makes b estimable on unlabeled images, and b is what a criterion correction
needs. Target b = 0 is correct by logic, not assumed from a possibly-miscalibrated
reference.

THE WAY IT FAILS, PRE-REGISTERED BEFORE RUNNING. If the model does not process the
polarity flip, then g^flipped ~ g^native, r_i ~ g_i, and driving r to zero would
drive the whole decision statistic to zero -- destroying d' rather than centring
the criterion. That failure is INVISIBLE unless tested, so this script tests it
directly and the method is abandoned if the test fails.

DECISION RULE (pre-registered, evaluated on the BASE model):
  PASS  if corr(g_native, g_flipped) <= -0.30 for at least one flip template
        AND the positive-control template reaches corr >= +0.60.
  FAIL  if the best flip correlation is >= -0.10 (the model ignores polarity).
  INCONCLUSIVE otherwise -> report, do not build on it.
The positive control is REQUIRED: a weak flip correlation means nothing if the
pipeline cannot even reproduce a paraphrase, so a null without it is NO-DATA
rather than evidence (this project has produced exactly that error before).

Consumes the *_dump.jsonl files written by pilot/eval_gen.py --dump_logits.

Usage:
  python3 antisymmetry_diag.py --dumps native=<f0> pos_control=<f3> neg=<f1> absent=<f2>
      [--measured_c 0.4314] [--out out.json]
"""
import argparse
import json
import math
import os
import sys


def read_dump(path):
    """{id: gap} from an eval_gen --dump_logits file. Fails loud on a bad row."""
    out = {}
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit("%s:%d: invalid JSON (%s)" % (path, ln, e))
            if "id" not in r:
                raise SystemExit("%s:%d: row has no 'id'" % (path, ln))
            # eval_gen writes the yes/no logits and their gap; recompute from the
            # logits when both are present so a renamed 'gap' field cannot silently
            # change the sign convention underneath us.
            # g_pooled is the casing-robust statistic (method/answer_tokens.py).
            # z_yes/z_no is the OLD single-id pair, kept only so historical dumps
            # remain readable -- it is wrong at stages where the emitted casing
            # drifted, so prefer g_pooled whenever present.
            if "g_pooled" in r:
                g = float(r["g_pooled"])
            elif "z_yes" in r and "z_no" in r:
                g = float(r["z_yes"]) - float(r["z_no"])
            elif "gap" in r:
                g = float(r["gap"])
            else:
                raise SystemExit("%s:%d: no g_pooled, no z_yes/z_no, no gap"
                                 % (path, ln))
            out[r["id"]] = g
    if not out:
        raise SystemExit("%s: empty dump" % path)
    return out


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def mean_sd(v):
    n = len(v)
    m = sum(v) / n
    sd = (sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return m, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True,
                    help="name=path pairs; 'native' and 'pos_control' are required")
    ap.add_argument("--measured_c", type=float, default=None,
                    help="POPE criterion c measured for this checkpoint, for reference")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dumps = {}
    for spec in args.dumps:
        if "=" not in spec:
            raise SystemExit("bad --dumps entry %r (want name=path)" % spec)
        name, path = spec.split("=", 1)
        if not os.path.isfile(path):
            print("MISSING %s -> %s" % (name, path))
            continue
        dumps[name] = read_dump(path)
    if "native" not in dumps:
        raise SystemExit("need a 'native' dump")
    if "pos_control" not in dumps:
        print("WARNING: no positive control supplied; a weak flip correlation "
              "will be NO-DATA, not evidence.")

    nat = dumps["native"]
    report = {"n_native": len(nat), "measured_c": args.measured_c, "templates": {}}

    print("Polarity-antisymmetry diagnostic (base decision statistic g = z_yes - z_no)")
    m, s = mean_sd(list(nat.values()))
    print("\nnative: n=%d  mean g=%+.4f  sd=%.4f" % (len(nat), m, s))
    report["native_mean_g"] = round(m, 4)
    report["native_sd_g"] = round(s, 4)

    hdr = "%-14s %6s %9s %9s %10s %12s" % (
        "template", "n", "corr", "mean g", "mean r", "role")
    print("\n" + hdr)
    print("-" * len(hdr))
    best_flip = None
    poscorr = None
    for name, d in dumps.items():
        if name == "native":
            continue
        ids = sorted(set(nat) & set(d))
        if len(ids) < 30:
            print("%-14s %6d  too few shared ids" % (name, len(ids)))
            continue
        xs = [nat[i] for i in ids]
        ys = [d[i] for i in ids]
        c = pearson(xs, ys)
        gm, _ = mean_sd(ys)
        # r_i = (g_native + g_flipped)/2 is the bias estimate ONLY for a genuine
        # polarity flip; for the positive control it is just an average of two
        # same-signed statistics and is reported but meaningless.
        rm, rsd = mean_sd([(a + b) / 2.0 for a, b in zip(xs, ys)])
        role = "pos-control" if name == "pos_control" else "flip"
        print("%-14s %6d %9.4f %9.4f %10.4f %12s" % (name, len(ids), c, gm, rm, role))
        report["templates"][name] = {
            "n": len(ids), "corr_with_native": round(c, 4),
            "mean_g": round(gm, 4), "mean_r": round(rm, 4), "sd_r": round(rsd, 4),
            "role": role}
        if role == "flip" and (best_flip is None or c < best_flip[1]):
            best_flip = (name, c, rm)
        if name == "pos_control":
            poscorr = c

    print()
    if best_flip is None:
        verdict = "NO-DATA (no flip template produced enough shared ids)"
    elif poscorr is None:
        verdict = "NO-DATA (no positive control; cannot distinguish a real null "
        verdict += "from a broken pipeline)"
    elif poscorr < 0.60:
        verdict = ("NO-DATA: positive control corr %.4f < 0.60 -- the pipeline or the "
                   "probe cannot even reproduce a paraphrase, so the flip result "
                   "carries no information." % poscorr)
    elif best_flip[1] <= -0.30:
        verdict = ("PASS: flip %r reaches corr %.4f <= -0.30 with positive control "
                   "%.4f. The model processes polarity, so r = (g+ + g-)/2 is a "
                   "usable LABEL-FREE bias estimate (mean r = %+.4f)."
                   % (best_flip[0], best_flip[1], poscorr, best_flip[2]))
    elif best_flip[1] >= -0.10:
        verdict = ("FAIL: best flip corr %.4f >= -0.10 with a healthy positive control "
                   "%.4f. The model does NOT invert on polarity, so driving r to zero "
                   "would collapse the decision statistic and destroy d'. Do not build "
                   "the antisymmetry method on this backbone."
                   % (best_flip[1], poscorr))
    else:
        verdict = ("INCONCLUSIVE: best flip corr %.4f sits between -0.30 and -0.10 "
                   "(positive control %.4f). Report; do not build on it."
                   % (best_flip[1], poscorr))
    report["verdict"] = verdict
    print("VERDICT: %s" % verdict)

    if args.measured_c is not None and best_flip is not None:
        print("\nMeasured POPE criterion c = %+.4f for this checkpoint. A usable bias "
              "estimate should have\nmean r of the OPPOSITE sign (positive c = "
              "conservative = says 'No' more = negative g bias)." % args.measured_c)
        agree = (args.measured_c > 0) == (best_flip[2] < 0)
        print("Sign agreement: %s" % ("YES" if agree else "NO -- investigate before use"))
        report["sign_agreement"] = bool(agree)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
