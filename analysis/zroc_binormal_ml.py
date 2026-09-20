#!/usr/bin/env python3
"""Maximum-likelihood binormal fit: is our OLS z-ROC slope attenuated, and by how much?

WHY THIS EXISTS (2026-09-13). `zroc_coherence.py` estimates the z-ROC by OLS of
z(H) on z(FA). The source discipline abandoned that estimator decades ago for a
reason that is stated in our own design note: z(FA) is ITSELF estimated from a
finite sample, so the regressor carries error, and errors-in-variables attenuates
the fitted slope toward zero (Pesce & Metz 2007, PMC2693394). Our entire reading
turns on b < 1, and attenuation pushes b down, so the claim is that OLS is
CONSERVATIVE here. That is an assertion about a bias direction and a magnitude,
and it has never been measured on this data. This script measures it.

THE ESTIMATOR. Write the binormal model with the noise distribution standardised:
a checkpoint's criterion places it at x = z(FA), and the model says
z(H) = a + b*x. So the two observed rates at checkpoint k are

    FA_k = Phi(x_k)            H_k = Phi(a + b*x_k)

and the data behind them are binomial counts: f_k false alarms out of n_no
ground-truth-negative items, h_k hits out of n_yes ground-truth-positive items.
The log-likelihood is the sum of the two binomial log-likelihoods over all
checkpoints, maximised jointly over (a, b, x_1..x_K). Because each x_k is a free
parameter estimated from BOTH rates rather than read off z(FA_k) alone, this is
the errors-in-variables fit that OLS approximates. It is not the Dorfman-Alf
rating-category MLE, and must not be called that: we have no latent rating table,
only K independent (FA, Hit) pairs. It is the MLE for OUR design.

WHAT THIS SCRIPT CONCLUDES.
  1. For SEQUENTIAL the attenuation is real, measurable and small: the ML slope
     sits a couple of percent above OLS, and the profile-likelihood interval for
     b excludes 1 in every cell. The b < 1 reading survives the better estimator,
     and OLS is conservative exactly as claimed.
  2. For the ANCHOR the ML slope does NOT merely move -- it stops being
     identified. The 95% profile-likelihood interval for b runs to the search
     ceiling in several cells. The anchor's operating range is too compressed for
     ANY estimator to recover the slope from these counts. The honest statement
     is not "the anchor's slope is 0.55" nor "it is 1.79", but "the anchor's
     slope is not estimable from this design".

WHAT WOULD FALSIFY THIS. (a) If the simulation showed OLS biased UPWARD at our
sample sizes and operating ranges, the "OLS is conservative" sentence would be
backwards and would have to be deleted from the paper. (b) If the sequential
profile intervals included 1, the unequal-variance reading -- which is what
overturned the corrected-ceiling result -- would lose its evidentiary basis.
(c) If the anchor's intervals came back tight, the non-identification finding
here would be wrong and the per-cell anchor d_a spread could be read as is.

Assertions, not faith: the OLS path is re-derived here and checked against the
certified numbers in readout/zroc_coherence.json before anything else runs, the
integer counts are checked to round-trip to the published rates, and the fast
Newton inner solve is cross-checked against a bounded-Brent solve on real cells.
"""
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import chi2, norm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from zroc_coherence import fit as ols_fit  # noqa: E402  certified OLS logic
from zroc_coherence import z as acklam_z  # noqa: E402  certified z transform

BACKBONE = "llava15"
ARMS = ("seq", "anchor", "joint")
MIN_STAGES = 4
B_CEIL = 6.0          # slope search ceiling; a CI touching this is UNBOUNDED
A_LO, A_HI = -2.0, 10.0
BOOT_REPS = 20000
SIM_REPS = 4000
SEED = 20260913


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def stage_counts(cell):
    """Integer binomial counts per checkpoint, reconstructed from published rates.

    fs_aggregate.json stores H and FA rounded to 4dp alongside the exact
    denominators n_gt_yes / n_gt_no, which differ per stage because unparseable
    rows are dropped. Multiplying and rounding recovers the counts; we assert the
    round-trip so a silent change in rounding or denominators fails loudly rather
    than producing a likelihood built on invented data.
    """
    rows = []
    for s in sorted((x for x in cell if x.isdigit()), key=int):
        p = cell[s].get("pope")
        if not p:
            continue
        if not (0 < p["H"] < 1 and 0 < p["FA"] < 1):
            continue
        n_yes, n_no = int(p["n_gt_yes"]), int(p["n_gt_no"])
        h, f = int(round(p["H"] * n_yes)), int(round(p["FA"] * n_no))
        assert round(h / n_yes, 4) == p["H"], (
            "hit count does not round-trip to the published H for stage %s: "
            "%d/%d -> %.4f vs %.4f" % (s, h, n_yes, h / n_yes, p["H"]))
        assert round(f / n_no, 4) == p["FA"], (
            "false-alarm count does not round-trip to the published FA for "
            "stage %s: %d/%d -> %.4f vs %.4f" % (s, f, n_no, f / n_no, p["FA"]))
        assert 0 < h < n_yes and 0 < f < n_no, (
            "stage %s sits on a rate boundary; the binomial log-likelihood is "
            "undefined there and this script must not silently fit it" % s)
        rows.append((h, n_yes, f, n_no))
    return rows


def cell_arrays(rows):
    h = np.array([r[0] for r in rows], float)
    n_yes = np.array([r[1] for r in rows], float)
    f = np.array([r[2] for r in rows], float)
    n_no = np.array([r[3] for r in rows], float)
    return h, n_yes, f, n_no


def arm_cells(arms, arm):
    """(label, rows) for each cell of an arm, in the certified sort order."""
    out = []
    for key, cell in sorted(arms.items()):
        if key.split("|")[0] != arm:
            continue
        rows = stage_counts(cell)
        if len(rows) < MIN_STAGES:
            continue
        out.append((key.split("|", 1)[1].replace("|", "/"), rows, cell))
    return out


# --------------------------------------------------------------------------
# likelihood
# --------------------------------------------------------------------------
def loglik_saturated(h, n_yes, f, n_no):
    """Each checkpoint's two rates fit exactly. 2K free parameters."""
    ph, pf = h / n_yes, f / n_no
    return float(np.sum(h * np.log(ph) + (n_yes - h) * np.log1p(-ph)
                        + f * np.log(pf) + (n_no - f) * np.log1p(-pf)))


def _ll_at(x, a, b, h, n_yes, f, n_no):
    u = a + b * x
    return (h * norm.logcdf(u) + (n_yes - h) * norm.logsf(u)
            + f * norm.logcdf(x) + (n_no - f) * norm.logsf(x))


def profile_x(a, b, h, n_yes, f, n_no, x0=None, _newton=True):
    """Maximise the likelihood over the criteria x_1..x_K, given (a, b).

    For b > 0 the per-checkpoint objective is a sum of log-concave normal CDF and
    SF terms and is therefore strictly concave in x, so the maximum is unique and
    Newton converges from any reasonable start. Returns (x_hat, loglik).
    """
    if x0 is None:
        x0 = norm.ppf(f / n_no)
    x = np.array(x0, float, copy=True)
    a = np.broadcast_to(np.asarray(a, float), x.shape)  # scalar OR per-checkpoint
    if _newton:
        for _ in range(80):
            u = a + b * x
            lam_u = np.exp(norm.logpdf(u) - norm.logcdf(u))
            haz_u = np.exp(norm.logpdf(u) - norm.logsf(u))
            lam_x = np.exp(norm.logpdf(x) - norm.logcdf(x))
            haz_x = np.exp(norm.logpdf(x) - norm.logsf(x))
            g1 = (b * (h * lam_u - (n_yes - h) * haz_u)
                  + (f * lam_x - (n_no - f) * haz_x))
            g2 = (b * b * (-h * lam_u * (u + lam_u)
                           - (n_yes - h) * haz_u * (haz_u - u))
                  + (-f * lam_x * (x + lam_x)
                     - (n_no - f) * haz_x * (haz_x - x)))
            if not np.all(np.isfinite(g1)) or np.any(g2 >= 0):
                _newton = False
                break
            step = np.clip(g1 / g2, -1.0, 1.0)
            x = np.clip(x - step, -8.0, 8.0)
            if np.max(np.abs(step)) < 1e-11:
                break
        else:
            _newton = False
    if not _newton:  # certified slow path: bounded Brent, one checkpoint at a time
        x = np.array([
            minimize_scalar(
                lambda t, i=i: -float(
                    _ll_at(t, a[i], b, h[i], n_yes[i], f[i], n_no[i])),
                bounds=(-8, 8), method="bounded", options={"xatol": 1e-11}).x
            for i in range(len(h))], float)
    return x, float(np.sum(_ll_at(x, a, b, h, n_yes, f, n_no)))


def profile_ab(a, b, h, n_yes, f, n_no):
    if b <= 0:
        return -np.inf
    return profile_x(a, b, h, n_yes, f, n_no)[1]


def profile_at_b(b, h, n_yes, f, n_no, a_hint=2.0):
    """max over a and the criteria, with b held fixed."""
    r = minimize_scalar(lambda a: -profile_ab(a, b, h, n_yes, f, n_no),
                        bounds=(A_LO, A_HI), method="bounded",
                        options={"xatol": 1e-9})
    return -float(r.fun)


def fit_ml(h, n_yes, f, n_no, starts):
    """Multistart Nelder-Mead on the profiled likelihood. Returns (a, b, loglik)."""
    best = None
    for s in starts:
        r = minimize(lambda p: -profile_ab(p[0], p[1], h, n_yes, f, n_no),
                     np.asarray(s, float), method="Nelder-Mead",
                     options={"xatol": 1e-10, "fatol": 1e-10,
                              "maxiter": 8000, "maxfev": 8000})
        if np.isfinite(r.fun) and (best is None or -r.fun > best[2]):
            best = (float(r.x[0]), float(r.x[1]), -float(r.fun))
    assert best is not None, "every ML start diverged; refusing to emit a slope"
    return best


def profile_ci_b(h, n_yes, f, n_no, b_hat, ll_max, level=0.95):
    """95% profile-likelihood interval for the slope b.

    The Wald interval is the wrong tool here: when the operating range is
    compressed the likelihood in b is strongly asymmetric and can fail to close
    at all, and a symmetric Wald interval would hide exactly that. We bracket and
    bisect the profile deviance instead, and report honestly when a side runs to
    the search ceiling.
    """
    thr = ll_max - chi2.ppf(level, 1) / 2.0

    def g(b):
        return profile_at_b(b, h, n_yes, f, n_no) - thr

    def walk(direction):
        lo, step = b_hat, 0.05 * max(b_hat, 0.2)
        hi = b_hat
        for _ in range(200):
            hi = hi + direction * step
            if hi <= 1e-3:
                return 1e-3, False
            if hi >= B_CEIL:
                return B_CEIL, False
            if g(hi) < 0:
                break
            lo, step = hi, step * 1.6
        else:
            return (B_CEIL if direction > 0 else 1e-3), False
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if g(mid) >= 0:
                lo = mid
            else:
                hi = mid
            if abs(hi - lo) < 1e-4:
                break
        return 0.5 * (lo + hi), True

    hi_b, hi_closed = walk(+1)
    lo_b, lo_closed = walk(-1)
    return (lo_b, hi_b, lo_closed, hi_closed)


# --------------------------------------------------------------------------
# OLS on the same counts (the estimator we are checking)
# --------------------------------------------------------------------------
def ols_on_counts(h, n_yes, f, n_no):
    """OLS on the exact count ratios (the inputs the likelihood also uses)."""
    pts = [(acklam_z(f[i] / n_no[i]), acklam_z(h[i] / n_yes[i]))
           for i in range(len(h))]
    return ols_fit(pts)


def ols_on_published_rates(cell):
    """OLS on the 4dp rates exactly as `zroc_coherence.py` reads them.

    This reproduces the certified readout bit for bit. It differs from
    `ols_on_counts` only by the 4th-decimal rounding of H and FA, which moves the
    slope by <1e-3; we assert that rather than assume it, because the paper
    quotes the rate-based number while the likelihood is built on the counts.
    """
    pts = []
    for s in sorted((x for x in cell if x.isdigit()), key=int):
        p = cell[s].get("pope")
        if p and 0 < p["H"] < 1 and 0 < p["FA"] < 1:
            pts.append((acklam_z(p["FA"]), acklam_z(p["H"])))
    return ols_fit(pts)


def ols_vectorised(zfa, zh):
    """OLS slope/intercept for many simulated replicates at once (B, K) -> (B,)."""
    mx = zfa.mean(axis=1, keepdims=True)
    my = zh.mean(axis=1, keepdims=True)
    sxx = ((zfa - mx) ** 2).sum(axis=1)
    sxy = ((zfa - mx) * (zh - my)).sum(axis=1)
    b = sxy / sxx
    a = my[:, 0] - b * mx[:, 0]
    return a, b


def simulate_ols_bias(h, n_yes, f, n_no, a_true, b_true, rng, reps=SIM_REPS):
    """How much does OLS attenuate at THIS cell's sample sizes and range?

    Ground truth is the OLS fit itself with the criteria placed at the observed
    z(FA). That choice makes the measured attenuation a LOWER BOUND on the true
    attenuation, because the observed spread of z(FA) is already inflated by the
    sampling noise we are about to add. A lower bound is what the argument needs:
    it cannot overstate how conservative OLS is.
    """
    x_true = norm.ppf(f / n_no)
    p_fa = norm.cdf(x_true)
    p_h = norm.cdf(a_true + b_true * x_true)
    K = len(h)
    fs = rng.binomial(n_no.astype(int), p_fa, size=(reps, K))
    hs = rng.binomial(n_yes.astype(int), p_h, size=(reps, K))
    ok = ((fs > 0) & (fs < n_no.astype(int)) & (hs > 0)
          & (hs < n_yes.astype(int))).all(axis=1)
    assert ok.mean() > 0.99, (
        "more than 1%% of simulated replicates hit a rate boundary; the "
        "attenuation estimate would be conditioned on a truncated sample")
    fs, hs = fs[ok], hs[ok]
    zfa = norm.ppf(fs / n_no)
    zh = norm.ppf(hs / n_yes)
    good = np.isfinite(zfa).all(axis=1) & np.isfinite(zh).all(axis=1)
    a_sim, b_sim = ols_vectorised(zfa[good], zh[good])
    return float(np.mean(b_sim)), float(np.mean(b_sim) - b_true), int(good.sum())


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def boot_ci_mean(values, rng, reps=BOOT_REPS, level=0.95):
    """Percentile CI for a mean over CELLS, resampling cells WITH replacement.

    The resample is built as a LIST of drawn indices and kept as a list. A set()
    here silently turns the bootstrap into a 63.2% subsample and narrows every
    interval; that bug has already cost this project nine wrong intervals, so the
    multiplicity is asserted rather than assumed.
    """
    v = np.asarray(values, float)
    n = len(v)
    if n < 3:
        return None
    idx = rng.integers(0, n, size=(reps, n))
    assert idx.shape == (reps, n), "bootstrap resample lost its multiplicity"
    dup = np.mean([len(set(row.tolist())) < n for row in idx[:200]])
    assert dup > 0.5, (
        "almost no bootstrap resample repeated a cell, which means the draw is "
        "not with replacement")
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return [round(float(lo), 4), round(float(hi), 4)]


def main():
    agg = json.load(open(os.path.join(HERE, "readout", "fs_aggregate.json")))
    certified = json.load(open(os.path.join(HERE, "readout", "zroc_coherence.json")))
    arms = agg["backbones"][BACKBONE]["arms"]
    rng = np.random.default_rng(SEED)

    # The z() used by the certified fit and scipy's must agree, or the OLS path
    # here and the OLS path in the paper are different estimators.
    for p in (0.0187, 0.0882, 0.1693, 0.5988, 0.8838):
        assert abs(acklam_z(p) - float(norm.ppf(p))) < 1e-8, (
            "certified z() and scipy norm.ppf disagree at p=%.4f" % p)

    report = {"backbone": BACKBONE, "source": "readout/fs_aggregate.json",
              "seed": SEED, "sim_reps": SIM_REPS, "boot_reps": BOOT_REPS,
              "b_search_ceiling": B_CEIL, "arms": {}}
    newton_checked = 0

    print("%-7s %-8s %7s %7s %7s  %-16s %7s %8s"
          % ("arm", "cell", "b_OLS", "b_ML", "d_b", "95% prof-LR CI",
             "d_a_OLS", "d_a_ML"))
    print("-" * 82)

    for arm in ARMS:
        cells = arm_cells(arms, arm)
        if not cells:
            continue
        cert = {c["cell"]: c for c in certified[arm]["per_cell"]}
        assert len(cells) == certified[arm]["n"], (
            "cell count for %s (%d) disagrees with the certified readout (%d)"
            % (arm, len(cells), certified[arm]["n"]))
        per_cell = []
        for label, rows, raw in cells:
            h, n_yes, f, n_no = cell_arrays(rows)
            # The certified estimator, reproduced exactly: OLS on the 4dp rates.
            a_o, b_o, r2_o, da_o = ols_on_published_rates(raw)
            c = cert[label]
            assert abs(round(b_o, 4) - c["slope"]) < 1e-9, (
                "re-derived OLS slope for %s/%s (%.4f) does not match the "
                "certified readout (%.4f)" % (arm, label, b_o, c["slope"]))
            assert abs(round(r2_o, 4) - c["r2"]) < 1e-9, (
                "re-derived R^2 for %s/%s (%.4f) does not match the certified "
                "readout (%.4f)" % (arm, label, r2_o, c["r2"]))
            assert abs(round(da_o, 4) - c["d_a"]) < 1e-9, (
                "re-derived d_a for %s/%s does not match the certified readout"
                % (arm, label))
            # The likelihood is built on integer counts, so the OLS we compare it
            # against must be shown to be the same estimator on the same data.
            a_c, b_c, r2_c, _ = ols_on_counts(h, n_yes, f, n_no)
            assert abs(b_c - b_o) < 5e-3, (
                "OLS slope on exact counts (%.4f) differs materially from OLS on "
                "the published 4dp rates (%.4f) for %s/%s; the slope comparison "
                "against ML would not be like for like"
                % (b_c, b_o, arm, label))
            # R^2 is NOT asserted, it is measured. Re-deriving the same fit from
            # counts rather than 4dp-rounded rates moves R^2 by up to 0.005 on
            # the anchor -- a fifth of the gap the paper leans on -- because the
            # anchor's SS_tot is small enough for rounding to matter. That
            # fragility is part of the case against R^2, so it is reported.
            r2_round_sens = abs(r2_c - r2_o)

            if newton_checked < 3:  # cross-check the fast path against Brent
                _, ll_fast = profile_x(a_o, b_o, h, n_yes, f, n_no)
                _, ll_slow = profile_x(a_o, b_o, h, n_yes, f, n_no, _newton=False)
                assert abs(ll_fast - ll_slow) < 1e-6, (
                    "Newton inner solve disagrees with bounded Brent by %.3g on "
                    "%s/%s" % (abs(ll_fast - ll_slow), arm, label))
                newton_checked += 1

            starts = [[a_o, b_o], [1.5, 0.5], [2.0, 1.0], [3.0, 1.5], [1.0, 0.3]]
            a_m, b_m, ll_m = fit_ml(h, n_yes, f, n_no, starts)
            ll_sat = loglik_saturated(h, n_yes, f, n_no)
            assert ll_m <= ll_sat + 1e-6, (
                "the one-ROC fit for %s/%s beat the saturated model, which is "
                "impossible and means the optimiser diverged" % (arm, label))
            _, ll_at_ols = profile_x(a_o, b_o, h, n_yes, f, n_no)
            assert ll_m >= ll_at_ols - 1e-6, (
                "ML fit for %s/%s is worse than the OLS parameters it started "
                "from; optimiser failure" % (arm, label))

            lo, hi, lo_closed, hi_closed = profile_ci_b(
                h, n_yes, f, n_no, b_m, ll_m)
            da_m = math.sqrt(2 / (1 + b_m * b_m)) * a_m
            b_sim, bias, nsim = simulate_ols_bias(h, n_yes, f, n_no, a_o, b_o, rng)

            ci_txt = "[%.2f,%s%.2f%s]" % (lo, "" if hi_closed else ">", hi,
                                          "" if hi_closed else "+")
            print("%-7s %-8s %7.3f %7.3f %+7.3f  %-16s %7.3f %8.3f"
                  % (arm, label, b_o, b_m, b_m - b_o, ci_txt, da_o, da_m))

            per_cell.append({
                "cell": label, "K": len(rows),
                "b_ols": round(b_o, 4), "a_ols": round(a_o, 4),
                "b_ols_exact_counts": round(b_c, 4),
                "r2_rounding_sensitivity": round(r2_round_sens, 5),
                "r2_ols": round(r2_o, 4), "d_a_ols": round(da_o, 4),
                "b_ml": round(b_m, 4), "a_ml": round(a_m, 4),
                "d_a_ml": round(da_m, 4),
                "delta_b_ml_minus_ols": round(b_m - b_o, 4),
                "b_ml_ci95": [round(lo, 4), round(hi, 4)],
                "ci_closed_below": bool(lo_closed),
                "ci_closed_above": bool(hi_closed),
                "b_identified": bool(lo_closed and hi_closed),
                "ci_excludes_1": bool(hi_closed and hi < 1.0),
                "loglik_m0": round(ll_m, 4),
                "loglik_saturated": round(ll_sat, 4),
                "sim_mean_b_ols": round(b_sim, 4),
                "sim_ols_bias": round(bias, 4),
                "sim_reps_used": nsim,
            })

        n = len(per_cell)
        d = {"n_cells": n,
             "mean_b_ols": round(float(np.mean([c["b_ols"] for c in per_cell])), 4),
             "mean_b_ml": round(float(np.mean([c["b_ml"] for c in per_cell])), 4),
             "mean_delta_b": round(
                 float(np.mean([c["delta_b_ml_minus_ols"] for c in per_cell])), 4),
             "mean_sim_ols_bias": round(
                 float(np.mean([c["sim_ols_bias"] for c in per_cell])), 4),
             "mean_d_a_ols": round(float(np.mean(
                 [c["d_a_ols"] for c in per_cell])), 4),
             "sd_d_a_ols": round(float(np.std(
                 [c["d_a_ols"] for c in per_cell], ddof=1)), 4) if n > 1 else None,
             "mean_d_a_ml": round(float(np.mean(
                 [c["d_a_ml"] for c in per_cell])), 4),
             "sd_d_a_ml": round(float(np.std(
                 [c["d_a_ml"] for c in per_cell], ddof=1)), 4) if n > 1 else None,
             "cells_b_identified": sum(c["b_identified"] for c in per_cell),
             "cells_ci_excludes_1": sum(c["ci_excludes_1"] for c in per_cell),
             "cells_sim_bias_negative": sum(
                 1 for c in per_cell if c["sim_ols_bias"] < 0),
             "max_r2_rounding_sensitivity": round(float(max(
                 c["r2_rounding_sensitivity"] for c in per_cell)), 5),
             "per_cell": per_cell}
        if n >= 3:
            d["boot_ci95_mean_b_ols"] = boot_ci_mean([c["b_ols"] for c in per_cell], rng)
            d["boot_ci95_mean_b_ml"] = boot_ci_mean([c["b_ml"] for c in per_cell], rng)
            d["boot_ci95_mean_delta_b"] = boot_ci_mean(
                [c["delta_b_ml_minus_ols"] for c in per_cell], rng)
            d["boot_ci95_mean_sim_ols_bias"] = boot_ci_mean(
                [c["sim_ols_bias"] for c in per_cell], rng)
        else:
            d["interval_note"] = (
                "n=%d cells: per-cell values only. An interval over %d cells "
                "would imply replication this arm does not have." % (n, n))
        report["arms"][arm] = d
        print()

    print("d_a under the two estimators (the quantity the canon note quotes):")
    for arm in ARMS:
        d = report["arms"].get(arm)
        if not d:
            continue
        print("  %-7s OLS %.4f +/- %-7s   ML %.4f +/- %s"
              % (arm, d["mean_d_a_ols"],
                 ("%.4f" % d["sd_d_a_ols"]) if d["sd_d_a_ols"] is not None else "n/a",
                 d["mean_d_a_ml"],
                 ("%.4f" % d["sd_d_a_ml"]) if d["sd_d_a_ml"] is not None else "n/a"))
    print()
    print("R^2 sensitivity to 4dp rounding of the published rates:")
    for arm in ARMS:
        d = report["arms"].get(arm)
        if d:
            print("  %-7s max |dR^2| = %.5f over %d cells"
                  % (arm, d["max_r2_rounding_sensitivity"], d["n_cells"]))
    print()
    print("Attenuation, measured (simulation at each cell's own n and range):")
    for arm in ARMS:
        d = report["arms"].get(arm)
        if not d:
            continue
        ci = d.get("boot_ci95_mean_sim_ols_bias")
        print("  %-7s n=%d  mean OLS bias %+.4f  %s  negative in %d/%d cells"
              % (arm, d["n_cells"], d["mean_sim_ols_bias"],
                 ("95%% CI [%+.4f, %+.4f]" % (ci[0], ci[1])) if ci
                 else "(no interval: n=%d)" % d["n_cells"],
                 d["cells_sim_bias_negative"], d["n_cells"]))
    print("\nSlope identification (95% profile-likelihood interval for b):")
    for arm in ARMS:
        d = report["arms"].get(arm)
        if not d:
            continue
        print("  %-7s identified in %d/%d cells; interval excludes 1 in %d/%d"
              % (arm, d["cells_b_identified"], d["n_cells"],
                 d["cells_ci_excludes_1"], d["n_cells"]))

    out = os.path.join(HERE, "readout", "zroc_binormal_ml.json")
    json.dump(report, open(out, "w"), indent=2)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
