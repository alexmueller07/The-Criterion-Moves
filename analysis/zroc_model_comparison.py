#!/usr/bin/env python3
"""Do a cell's six checkpoints lie on ONE z-ROC? Stated as a model comparison.

WHY THIS EXISTS (2026-09-13). `zroc_coherence.py` evidences the primary claim
with the R^2 of a per-cell OLS fit (sequential 0.957, anchor 0.586). R^2 is a
goodness-of-fit PROXY: it is range-dependent, it has no null distribution, and on
six points with two parameters it is a weak statement. Worse, it invites the
circularity objection -- a manipulation ROC traces one isosensitivity contour
only if the manipulation is criterion-only, which is the very thing we are trying
to establish. Quoting a high R^2 does not answer that, because no alternative was
ever written down.

Naming the two models answers it, because the null and the alternative become
explicit and the data are allowed to choose between them.

  M0, ONE ROC. A single shared (a, b) for the whole cell; each checkpoint gets
     its own criterion and therefore its own operating point, which must lie on
     that one curve. Parameters: a, b, and K criteria = K + 2 = 8 for K = 6.

  M1, DRIFTING SENSITIVITY. Each checkpoint gets its own sensitivity, so the
     points need not lie on any single curve. Parameters: 2K = 12 for K = 6.

ON THE PARAMETER COUNT OF M1, because this is the part that is easy to get wrong.
Each checkpoint contributes exactly two free binomial cells -- one hit rate, one
false-alarm rate -- so the whole cell carries 2K = 12 free probabilities. Letting
sensitivity vary per checkpoint therefore SATURATES the design: with one
operating point per checkpoint there is no identified intermediate between "one
shared curve" and "fit every point exactly". Writing per-checkpoint (a_k, b_k, x_k)
would be 3K = 18 parameters for 12 observations, and the extra 6 are not
estimable. So M1 is the saturated model, at 2K parameters, and its MLE is just
the observed rates. M0 is nested inside it -- M0's fitted probabilities are a
point in the same 12-dimensional space that M1 ranges over freely -- so the
likelihood-ratio comparison is legitimate, with df = 2K - (K + 2) = K - 2 = 4.

A DIRECTED THIRD MODEL, because the saturated alternative spends its degrees of
freedom on arbitrary scatter while the substantive worry is specifically that
sensitivity DECAYS along the task sequence:

  M1_trend. a_k = a0 + g*(k - kbar), shared b, K criteria = K + 3 = 9 parameters.
     M0 is the g = 0 restriction, so df = 1 against M0, and M1_trend is itself
     nested in M1. One degree of freedom aimed at the alternative we actually
     care about is a far more powerful test than four spent on generic scatter.

WHAT THIS SCRIPT CONCLUDES. By AIC, M0 is favoured in 9/9 sequential cells, 2/2
JOINT cells, and 6/9 anchor cells. By the likelihood-ratio test the one-ROC model
is not rejected at the 5% level in any sequential or JOINT cell, and is rejected
in none-to-few anchor cells. So the model comparison SUPPORTS the paper's claim
for sequential while declining to convict the anchor: the anchor's low R^2 is not
by itself evidence that its points leave a single curve. The arms still separate,
but on the size of the departure rather than on a verdict.

WHAT WOULD FALSIFY THIS. If M1 beat M0 on AIC in a majority of sequential cells,
or if the sequential likelihood-ratio tests rejected M0, the primary claim would
be dead in its current form -- the points would not lie on one curve and the
criterion-only reading would fail on its own terms. If the anchor's G^2 were
indistinguishable from sequential's, the arm contrast would carry no information
and the R^2 gap would have to be attributed to range compression after all.

THE LIMITATION THAT MATTERS, stated because it is not visible in the numbers. The
binomial likelihood treats a checkpoint's 9000 POPE answers as 9000 independent
Bernoulli trials. They are not, in TWO separate ways, and the two push the
statistic in OPPOSITE directions:

  - WITHIN a checkpoint, POPE's three splits SHARE their positive items (verified
    byte-identical in this project's own stratum audit -- that is what retracted
    the stratum "double dissociation"). The 4500 ground-truth-positive trials are
    therefore roughly 1500 distinct items asked three times, so the true sampling
    variance of H is larger than Binomial(4500, .) implies. That INFLATES G^2 and
    makes M0 look worse than it is.
  - ACROSS checkpoints, all six are scored on the SAME 9000 items, so their
    item-level idiosyncrasies are common-mode rather than independent and the six
    operating points scatter around a common curve LESS than independent
    binomials would. That DEFLATES G^2 and makes M0 look better than it is.

The net cannot be signed without per-item responses, which this project does not
have on disk for these cells. It is NO-DATA, not clearance. So the absolute
p-values are indicative rather than exact, and the ARM CONTRAST -- internal, and
carrying both flaws equally -- is what should be quoted. An earlier draft of this
docstring claimed the bias had a single known direction; that was wrong, and the
error is recorded here rather than quietly deleted.

Reuses the certified OLS and the ML machinery rather than reimplementing either.
"""
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import binom, chi2
from scipy.stats import norm as _norm

_cdf = _norm.cdf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from zroc_binormal_ml import (ARMS, BACKBONE, BOOT_REPS, SEED, arm_cells,  # noqa: E402
                              boot_ci_mean, cell_arrays, fit_ml,
                              loglik_saturated, ols_on_published_rates,
                              profile_ab, profile_x)

ALPHA = 0.05
CAL_REPS = 300


def calibrate_g2(h, n_yes, f, n_no, a, b, x, g2_obs, rng, reps=CAL_REPS):
    """Is the chi^2_4 reference actually right here? Parametric bootstrap.

    The asymptotics behind the likelihood-ratio test are in the number of trials
    per operating point (~9000 here), not in the number of operating points, so
    chi^2 SHOULD be well behaved. But "should" is an assertion, and this project
    has been bitten before by reference distributions that were never checked.
    We simulate from the fitted M0 -- the null is true by construction -- refit
    M0, and read the empirical distribution of G^2 off the replicates.

    Returns the calibrated p-value, the simulated mean (which should sit near
    df = 4) and the simulated 95th percentile (near 9.49) if chi^2_4 holds.
    """
    K = len(h)
    p_fa = np.asarray([float(v) for v in _cdf(x)])
    p_h = np.asarray([float(v) for v in _cdf(a + b * x)])
    fs = rng.binomial(n_no.astype(int), p_fa, size=(reps, K))
    hs = rng.binomial(n_yes.astype(int), p_h, size=(reps, K))
    keep = ((fs > 0) & (fs < n_no.astype(int)) & (hs > 0)
            & (hs < n_yes.astype(int))).all(axis=1)
    fs, hs = fs[keep], hs[keep]
    assert len(fs) >= 0.95 * reps, (
        "more than 5% of null replicates hit a rate boundary; the calibration "
        "would be conditioned on a truncated sample")
    sims = []
    for i in range(len(fs)):
        hh, ff = hs[i].astype(float), fs[i].astype(float)
        ll_sat = loglik_saturated(hh, n_yes, ff, n_no)
        r = minimize(lambda q: -profile_ab(q[0], q[1], hh, n_yes, ff, n_no),
                     np.array([a, b], float), method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-6,
                              "maxiter": 2000, "maxfev": 2000})
        sims.append(max(2.0 * (ll_sat + float(r.fun)), 0.0))
    sims = np.asarray(sims)
    return (float((sims >= g2_obs).mean()), float(sims.mean()),
            float(np.percentile(sims, 95)), int(len(sims)))


def fit_trend(h, n_yes, f, n_no, a0, b0):
    """M1_trend: intercept drifts linearly with checkpoint index, slope shared.

    Seeded at the M0 solution with g = 0 so the fitted likelihood can never come
    out below M0's by optimiser bad luck; the nesting is asserted afterwards
    regardless.
    """
    K = len(h)
    idx = np.arange(1, K + 1, dtype=float)
    idx = idx - idx.mean()

    def nll(p):
        a0_, g_, b_ = p
        if b_ <= 0:
            return 1e18
        return -profile_x(a0_ + g_ * idx, b_, h, n_yes, f, n_no)[1]

    best = None
    for s in ([a0, 0.0, b0], [a0, 0.05, b0], [a0, -0.05, b0]):
        r = minimize(nll, np.array(s, float), method="Nelder-Mead",
                     options={"xatol": 1e-10, "fatol": 1e-10,
                              "maxiter": 8000, "maxfev": 8000})
        if np.isfinite(r.fun) and (best is None or -r.fun > best[1]):
            best = (r.x, -float(r.fun))
    assert best is not None, "every M1_trend start diverged"
    return best[0], best[1]


def aic(loglik, k):
    return -2.0 * loglik + 2.0 * k


def main():
    agg = json.load(open(os.path.join(HERE, "readout", "fs_aggregate.json")))
    arms = agg["backbones"][BACKBONE]["arms"]
    rng = np.random.default_rng(SEED)

    report = {"backbone": BACKBONE, "source": "readout/fs_aggregate.json",
              "seed": SEED, "alpha": ALPHA,
              "models": {
                  "M0_one_roc": "shared (a,b) + K criteria; K+2 params",
                  "M1_saturated": "per-checkpoint sensitivity; 2K params "
                                  "(the design saturates: 2 free binomial "
                                  "cells per checkpoint)",
                  "M1_trend": "a_k = a0 + g*(k-kbar), shared b, + K criteria; "
                              "K+3 params",
                  "df_M1_vs_M0": "2K-(K+2) = K-2 = 4",
                  "df_M1trend_vs_M0": 1},
              "arms": {}}

    print("%-7s %-8s %2s %3s %3s %3s  %8s %5s %8s  %9s %8s %7s"
          % ("arm", "cell", "K", "p0", "p1", "pT", "G2_M1", "p", "dAIC",
             "G2_trend", "p_trend", "R2_OLS"))
    print("-" * 96)

    for arm in ARMS:
        cells = arm_cells(arms, arm)
        if not cells:
            continue
        per_cell = []
        for label, rows, raw in cells:
            h, n_yes, f, n_no = cell_arrays(rows)
            K = len(rows)
            a_o, b_o, r2_o, _ = ols_on_published_rates(raw)

            k_m0, k_m1, k_tr = K + 2, 2 * K, K + 3
            a_m, b_m, ll_m0 = fit_ml(
                h, n_yes, f, n_no,
                [[a_o, b_o], [1.5, 0.5], [2.0, 1.0], [3.0, 1.5], [1.0, 0.3]])
            x_m0, _ll_chk = profile_x(a_m, b_m, h, n_yes, f, n_no)
            assert abs(_ll_chk - ll_m0) < 1e-6, (
                "%s/%s: re-profiling the criteria at the ML (a,b) did not "
                "reproduce the M0 loglik" % (arm, label))
            ll_m1 = loglik_saturated(h, n_yes, f, n_no)
            tr_p, ll_tr = fit_trend(h, n_yes, f, n_no, a_m, b_m)

            # Nesting is a mathematical fact; if the numbers break it the
            # optimiser failed and no statistic below may be emitted.
            assert ll_m0 <= ll_m1 + 1e-6, (
                "%s/%s: M0 loglik (%.6f) exceeds the saturated model's (%.6f); "
                "M0 is nested in M1 so this is impossible"
                % (arm, label, ll_m0, ll_m1))
            assert ll_tr <= ll_m1 + 1e-6, (
                "%s/%s: M1_trend beat the saturated model" % (arm, label))
            assert ll_tr >= ll_m0 - 1e-6, (
                "%s/%s: M1_trend (%.6f) fell below M0 (%.6f) despite containing "
                "it at g=0; optimiser failure" % (arm, label, ll_tr, ll_m0))

            g2 = 2.0 * (ll_m1 - ll_m0)
            g2_tr = 2.0 * (ll_tr - ll_m0)
            assert g2 >= -1e-6 and g2_tr >= -1e-6, "negative LR statistic"
            df1, df_tr = k_m1 - k_m0, k_tr - k_m0
            assert df1 == K - 2 and df_tr == 1, "parameter counting is wrong"
            p1 = float(chi2.sf(max(g2, 0.0), df1))
            p_tr = float(chi2.sf(max(g2_tr, 0.0), df_tr))

            p_cal, g2_null_mean, g2_null_p95, n_cal = calibrate_g2(
                h, n_yes, f, n_no, a_m, b_m, x_m0, g2, rng)

            a_m0, a_m1, a_tr = (aic(ll_m0, k_m0), aic(ll_m1, k_m1),
                                aic(ll_tr, k_tr))
            d_aic = a_m1 - a_m0          # > 0 favours M0
            best = min([("M0", a_m0), ("M1", a_m1), ("M1_trend", a_tr)],
                       key=lambda t: t[1])[0]

            print("%-7s %-8s %2d %3d %3d %3d  %8.3f %5.3f %+8.2f  %9.3f %8.3f %7.4f"
                  % (arm, label, K, k_m0, k_m1, k_tr, g2, p1, d_aic,
                     g2_tr, p_tr, r2_o))

            per_cell.append({
                "cell": label, "K": K,
                "params_M0": k_m0, "params_M1": k_m1, "params_M1_trend": k_tr,
                "loglik_M0": round(ll_m0, 4),
                "loglik_M1_saturated": round(ll_m1, 4),
                "loglik_M1_trend": round(ll_tr, 4),
                "G2_M1_vs_M0": round(g2, 4), "df_M1_vs_M0": df1,
                "p_M1_vs_M0": round(p1, 5),
                "p_M1_vs_M0_calibrated": round(p_cal, 5),
                "null_G2_mean_simulated": round(g2_null_mean, 3),
                "null_G2_p95_simulated": round(g2_null_p95, 3),
                "calibration_reps": n_cal,
                "reject_M0_at_alpha": bool(p1 < ALPHA),
                "reject_M0_at_alpha_calibrated": bool(p_cal < ALPHA),
                "G2_trend_vs_M0": round(g2_tr, 4), "df_trend_vs_M0": df_tr,
                "p_trend_vs_M0": round(p_tr, 5),
                "trend_g_per_stage": round(float(tr_p[1]), 5),
                "AIC_M0": round(a_m0, 3), "AIC_M1": round(a_m1, 3),
                "AIC_M1_trend": round(a_tr, 3),
                "delta_AIC_M1_minus_M0": round(d_aic, 3),
                "aic_favours_M0_over_M1": bool(d_aic > 0),
                "aic_best_model": best,
                "b_ml": round(b_m, 4), "b_ols": round(b_o, 4),
                "r2_ols": round(r2_o, 4),
            })

        n = len(per_cell)
        d = {"n_cells": n,
             "cells_aic_favours_M0": sum(c["aic_favours_M0_over_M1"]
                                         for c in per_cell),
             "cells_aic_best_is_M0": sum(1 for c in per_cell
                                         if c["aic_best_model"] == "M0"),
             "cells_M0_rejected_at_alpha": sum(c["reject_M0_at_alpha"]
                                               for c in per_cell),
             "cells_M0_rejected_calibrated": sum(
                 c["reject_M0_at_alpha_calibrated"] for c in per_cell),
             "mean_null_G2_simulated": round(float(np.mean(
                 [c["null_G2_mean_simulated"] for c in per_cell])), 3),
             "mean_null_G2_p95_simulated": round(float(np.mean(
                 [c["null_G2_p95_simulated"] for c in per_cell])), 3),
             "cells_trend_significant": sum(1 for c in per_cell
                                            if c["p_trend_vs_M0"] < ALPHA),
             "mean_G2": round(float(np.mean([c["G2_M1_vs_M0"] for c in per_cell])), 4),
             "max_G2": round(float(max(c["G2_M1_vs_M0"] for c in per_cell)), 4),
             "mean_delta_AIC": round(float(np.mean(
                 [c["delta_AIC_M1_minus_M0"] for c in per_cell])), 4),
             "mean_r2_ols": round(float(np.mean([c["r2_ols"] for c in per_cell])), 4),
             "per_cell": per_cell}
        # An arm-level read on the per-cell tests. Under the null each cell
        # rejects with probability ALPHA, so the COUNT of rejecting cells is
        # Binomial(n, ALPHA) and its upper tail is the arm's own p-value. This
        # is what keeps "3 of 9 anchor cells drift" from being read as noise in
        # one direction or dismissed as multiplicity in the other.
        n_sig_trend = sum(1 for c in per_cell if c["p_trend_vs_M0"] < ALPHA)
        n_sig_m0 = sum(1 for c in per_cell if c["p_M1_vs_M0"] < ALPHA)
        d["trend_arm_exceedance_p"] = round(
            float(binom.sf(n_sig_trend - 1, n, ALPHA)) if n_sig_trend > 0 else 1.0, 5)
        d["M0_reject_arm_exceedance_p"] = round(
            float(binom.sf(n_sig_m0 - 1, n, ALPHA)) if n_sig_m0 > 0 else 1.0, 5)
        if n >= 3:
            d["boot_ci95_mean_G2"] = boot_ci_mean(
                [c["G2_M1_vs_M0"] for c in per_cell], rng)
            d["boot_ci95_mean_delta_AIC"] = boot_ci_mean(
                [c["delta_AIC_M1_minus_M0"] for c in per_cell], rng)
        else:
            d["interval_note"] = (
                "n=%d cells: per-cell values only. An interval over %d cells "
                "would imply replication this arm does not have." % (n, n))
        report["arms"][arm] = d
        print()

    print("Verdict per arm (M0 = one ROC, M1 = per-checkpoint sensitivity):")
    for arm in ARMS:
        d = report["arms"].get(arm)
        if not d:
            continue
        ci = d.get("boot_ci95_mean_G2")
        print("  %-7s n=%d | AIC favours M0 in %d/%d | M0 rejected (p<%.2f) in "
              "%d/%d | mean G2 %.3f %s"
              % (arm, d["n_cells"], d["cells_aic_favours_M0"], d["n_cells"],
                 ALPHA, d["cells_M0_rejected_at_alpha"], d["n_cells"],
                 d["mean_G2"],
                 ("95%% CI [%.3f, %.3f]" % (ci[0], ci[1])) if ci
                 else "(no interval: n=%d)" % d["n_cells"]))

    print("\nDirected alternative (M1_trend: sensitivity drifts with stage, df=1):")
    for arm in ARMS:
        d = report["arms"].get(arm)
        if d:
            print("  %-7s significant in %d/%d cells; arm-level exceedance "
                  "p = %.4f (Binomial(%d, %.2f) upper tail)"
                  % (arm, d["cells_trend_significant"], d["n_cells"],
                     d["trend_arm_exceedance_p"], d["n_cells"], ALPHA))

    print("\nIs chi^2_4 the right reference? (parametric bootstrap under M0)")
    for arm in ARMS:
        d = report["arms"].get(arm)
        if d:
            print("  %-7s simulated null mean G2 %.3f (df=4 expected), 95th pct "
                  "%.3f (chi2_4 = 9.488)"
                  % (arm, d["mean_null_G2_simulated"],
                     d["mean_null_G2_p95_simulated"]))

    s, a = report["arms"].get("seq"), report["arms"].get("anchor")
    if s and a:
        print("\nThe arm contrast, which is what the paper should quote:")
        print("  mean G2  sequential %.3f  vs  anchor %.3f  (ratio %.2fx)"
              % (s["mean_G2"], a["mean_G2"], a["mean_G2"] / s["mean_G2"]))
        print("  Same direction as the R^2 gap (%.4f vs %.4f), but now against a"
              % (s["mean_r2_ols"], a["mean_r2_ols"]))
        print("  named alternative with a null distribution attached.")
        # Cells are independent across arms, so the difference can be bootstrapped
        # directly rather than read off two overlapping-or-not intervals.
        gs = np.array([c["G2_M1_vs_M0"] for c in s["per_cell"]])
        ga = np.array([c["G2_M1_vs_M0"] for c in a["per_cell"]])
        r2 = np.random.default_rng(SEED + 1)
        i_s = r2.integers(0, len(gs), size=(BOOT_REPS, len(gs)))
        i_a = r2.integers(0, len(ga), size=(BOOT_REPS, len(ga)))
        diff = ga[i_a].mean(axis=1) - gs[i_s].mean(axis=1)
        lo, hi = np.percentile(diff, [2.5, 97.5])
        report["anchor_minus_seq_mean_G2"] = {
            "point": round(float(ga.mean() - gs.mean()), 4),
            "boot_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "n_seq_cells": len(gs), "n_anchor_cells": len(ga)}
        print("  anchor - sequential mean G2 = %+.3f, 95%% CI [%+.3f, %+.3f]"
              % (ga.mean() - gs.mean(), lo, hi))

    out = os.path.join(HERE, "readout", "zroc_model_comparison.json")
    json.dump(report, open(out, "w"), indent=2)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
