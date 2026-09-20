#!/usr/bin/env python3
"""Is the anchor's low z-ROC R^2 just a compressed operating range?

WHY THIS EXISTS (2026-09-12). R^2 = 1 - SS_res/SS_tot is range-dependent: shrink
the spread of z(FA) and SS_tot shrinks with it, so the SAME residual scatter
reports a lower R^2. An anchor regularizer is exactly the kind of intervention
that would compress the operating range, so "the anchor's R^2 is low" is open to
the objection that we measured the compression, not incoherence. A reviewer from
the ROC literature raises this immediately, and it would be fatal if unanswered.

THE ANSWER IS THE JOINT ARM, and it is an internal control we already have.
JOINT's z(FA) range is SMALLER than the anchor's, yet its R^2 is high, because
its residuals are several times smaller. So a compressed range does not by itself
produce a low R^2 in this data, and the anchor's does not come from its range.

The range-free statistic is the RESIDUAL RMSE in z-units, which this script
reports alongside R^2 so the paper can quote a number that the objection does not
touch. Read the two together: R^2 says how much of the spread the line explains,
RMSE says how far the points sit off it in the units the fit is performed in.

Reuses the certified z()/fit()/cells_for logic rather than reimplementing it.
"""
import json, math, os, sys
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MPLBACKEND", "Agg")
from zroc_coherence import z, fit  # noqa: E402  certified fit logic


def cells(arms, arm):
    out = []
    for key, cell in sorted(arms.items()):
        if key.split("|")[0] != arm:
            continue
        pts = []
        for s in sorted((x for x in cell if x.isdigit()), key=int):
            p = cell[s].get("pope")
            if p and 0 < p["H"] < 1 and 0 < p["FA"] < 1:
                pts.append((z(p["FA"]), z(p["H"])))
        if len(pts) < 4:
            continue
        f = fit(pts)
        if f:
            out.append((pts, f))
    return out



def binormal_diagnostics(arms, cells_fn):
    """The source discipline's OWN diagnostics, applied to our fits.

    Our z(H) = a + b*z(FA) IS the binormal ROC model, canonical in diagnostic
    radiology since Dorfman & Alf (1969) and textbook there. Estimating b rather
    than assuming b = 1 is that field's DEFAULT, not a refinement of it, and
    b < 1 is the expected empirical regularity (ten real reader-modality fits in
    Hillis & Berbaum 2011 Table 5 give b in [0.20, 0.83], mean 0.51). So our
    0.55-0.72 is ordinary, not anomalous, and the paper should say so.

    Two indices, both verified at MathML level from two independent sources:
        d_a = sqrt(2)*a / sqrt(1 + b^2)        (equivalently sqrt(2/(1+b^2))*a)
        A_z = Phi(a / sqrt(1 + b^2)) = Phi(d_a / sqrt(2))   [CONVENTIONAL binormal]
    A_z above is the conventional-binormal area. The PROPER/binormal-LR curve has
    an extra bivariate-normal term; do not mix them.

    The degeneracy question -- "is your fit in a regime where the conventional
    binormal model misbehaves?" -- has the field's own answer, the mean-to-sigma
    ratio r = a/(1-b), classified by Hillis & Berbaum (2011) as improperness
    INDISCERNIBLE at |r| >= 3, slight at 2 < |r| < 3, noticeable at |r| <= 2. At
    |r| >= 3 the conventional and proper binormal fits are indistinguishable and
    the question is closed.

    This is a THIRD independent indicator that the anchor distorts the geometry,
    alongside R^2 (range-dependent) and residual RMSE (range-free): the anchor
    leaves only 5 of 9 cells in the indiscernible regime where sequential has 9/9
    and JOINT 2/2.
    """
    import math
    def Phi(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    rep = {}
    for arm in ("seq", "anchor", "joint"):
        rows = cells_fn(arms, arm)
        if not rows:
            continue
        bs, das, azs, rs = [], [], [], []
        for _, f in rows:
            a, b, _, d_a = f
            bs.append(b); das.append(d_a); azs.append(Phi(d_a / math.sqrt(2)))
            rs.append(a / (1 - b) if abs(1 - b) > 1e-9 else float("inf"))
        rep[arm] = {
            "n": len(rows),
            "mean_b": round(st.mean(bs), 4),
            "mean_d_a": round(st.mean(das), 4),
            "mean_A_z": round(st.mean(azs), 4),
            "mean_r": round(st.mean(rs), 3),
            "cells_improperness_indiscernible": sum(1 for x in rs if abs(x) >= 3),
        }
    return rep


def main():
    agg = json.load(open(os.path.join(HERE, "readout", "fs_aggregate.json")))
    arms = agg["backbones"]["llava15"]["arms"]
    rep = {}
    print("%-8s %4s %9s %12s %12s" % ("arm", "n", "mean R2", "z(FA) range", "resid RMSE"))
    print("-" * 50)
    for arm in ("seq", "anchor", "joint"):
        rows = cells(arms, arm)
        if not rows:
            continue
        r2s, rngs, rmses = [], [], []
        for pts, f in rows:
            a, b = f[0], f[1]
            xs = [p[0] for p in pts]
            r2s.append(f[2])
            rngs.append(max(xs) - min(xs))
            res = [y - (a + b * x) for x, y in pts]
            rmses.append(math.sqrt(sum(e * e for e in res) / len(res)))
        rep[arm] = {"n": len(rows), "mean_r2": round(st.mean(r2s), 4),
                    "mean_zfa_range": round(st.mean(rngs), 4),
                    "mean_resid_rmse": round(st.mean(rmses), 4)}
        print("%-8s %4d %9.4f %12.4f %12.4f"
              % (arm, len(rows), st.mean(r2s), st.mean(rngs), st.mean(rmses)))

    a, j = rep.get("anchor"), rep.get("joint")
    if a and j:
        print("\nThe control that answers the range objection:")
        print("  JOINT's operating range (%.4f) is %.0f%% of the anchor's (%.4f),"
              % (j["mean_zfa_range"], 100 * j["mean_zfa_range"] / a["mean_zfa_range"],
                 a["mean_zfa_range"]))
        print("  yet JOINT's R2 is %.4f against the anchor's %.4f, because JOINT's"
              % (j["mean_r2"], a["mean_r2"]))
        print("  residuals are %.1fx smaller (%.4f vs %.4f)."
              % (a["mean_resid_rmse"] / j["mean_resid_rmse"],
                 j["mean_resid_rmse"], a["mean_resid_rmse"]))
        print("  A narrow range therefore does NOT produce a low R2 here.")
        assert j["mean_zfa_range"] < a["mean_zfa_range"], (
            "the control no longer holds: JOINT's range is not smaller than the "
            "anchor's, so this script's argument must be re-derived before use")
    diag = binormal_diagnostics(arms, cells)
    print("\nBinormal diagnostics (the source discipline's own):")
    print("  %-8s %8s %8s %9s %10s %s" % ("arm", "b", "d_a", "A_z", "r=a/(1-b)", "|r|>=3"))
    for arm, d in diag.items():
        print("  %-8s %8.3f %8.3f %9.4f %10.2f %d/%d"
              % (arm, d["mean_b"], d["mean_d_a"], d["mean_A_z"], d["mean_r"],
                 d["cells_improperness_indiscernible"], d["n"]))
    print("  |r| >= 3 = improperness indiscernible (Hillis & Berbaum 2011): the")
    print("  conventional and proper binormal fits are then indistinguishable.")
    rep["binormal_diagnostics"] = diag

    out = os.path.join(HERE, "readout", "zroc_range_check.json")
    json.dump(rep, open(out, "w"), indent=2)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    sys.exit(main())
