#!/usr/bin/env python3
"""
theory_sanitycheck.py -- numerical support for design_notes/method_calibration_theory.md

Three self-contained checks (Python stdlib only; no numpy, so no Accelerate
matmul warnings and it runs anywhere):

  (A) Additive-bias decomposition of the observed criterion drift.
      Claim (Task 1): a CL stage update acts, to first order, as a class-
      INDEPENDENT additive shift b on the decision variable Delta = z_yes - z_no.
      Such a shift moves c by -b and leaves d' unchanged. We fit the best
      additive b for each observed S0->Sk transition and report the residual
      d' change -- the part of the drift that is NOT pure bias. If |Delta d'|
      << |b| the drift is ~pure criterion motion, quantifying "d' is flat".

  (B) Bias-equilibrium of the anchor objective, symmetric (v2) vs one-sided (v1).
      Claim (Task 3): the symmetric decision-axis margin has a unique interior
      stable bias equilibrium (restoring / criterion-freezing); the v1 objective
      (absent-side term only) has a strictly one-signed force with NO interior
      zero, so it drives the criterion unboundedly conservative. We minimize the
      population potential V(b) over a grid for a Gaussian decision-variable
      model matched to the base d', and report the equilibrium criterion.

  (C) Inverse-strength law (Task 5, prediction 1). With a constant task-loss
      "drift force" f on the bias, the anchored equilibrium solves
      lambda_F * V'(b) + f = 0; we show residual |c - c*| ~ 1/lambda_F.

All inputs in (A) are the pilot's pooled POPE numbers from
analysis/diag_signal_detection.md (the 'all' rows). Nothing here is fit to data
beyond those published rates; (B)/(C) are structural demonstrations.
"""

import math
from statistics import NormalDist

N = NormalDist()          # standard normal
z = N.inv_cdf             # probit / quantile
Phi = N.cdf               # standard normal CDF
sp = lambda u: math.log1p(math.exp(u)) if u < 30 else u   # softplus, overflow-safe
sig = lambda u: 1.0 / (1.0 + math.exp(-u)) if u > -30 else 0.0  # logistic


def dprime_c(H, FA):
    zH, zFA = z(H), z(FA)
    return zH - zFA, -0.5 * (zH + zFA)


# ----------------------------------------------------------------------
# (A) Additive-bias decomposition of observed criterion drift
# ----------------------------------------------------------------------
# Pooled ('all') POPE hit / false-alarm rates, S0..S4 (diag_signal_detection.md).
SEQ = {
    "S0": (0.7711, 0.0542),
    "S1": (0.7989, 0.0649),
    "S2": (0.8713, 0.1436),
    "S3": (0.8364, 0.0911),
    "S4": (0.6740, 0.0262),
}

def check_A():
    print("=" * 74)
    print("(A) Additive-bias decomposition of the observed SEQ criterion drift")
    print("=" * 74)
    print("Model: equal-variance Gaussian decision variable, sigma=1, so")
    print("       mu1 = z(H), mu0 = z(FA); a class-independent additive shift b")
    print("       is the maximum-likelihood/least-squares fit to (z(H),z(FA)) moves.")
    print()
    zH0, zFA0 = z(SEQ["S0"][0]), z(SEQ["S0"][1])
    d0, c0 = dprime_c(*SEQ["S0"])
    print(f"  base S0: H={SEQ['S0'][0]:.4f} FA={SEQ['S0'][1]:.4f} "
          f"d'={d0:.4f} c={c0:.4f}")
    print()
    hdr = ("  trans    Dc(obs)    b_fit(=-Dc)   Dd'(resid)   "
           "frac_nonadditive   H,FA co-move?")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for k in ["S1", "S2", "S3", "S4"]:
        H, FA = SEQ[k]
        zH, zFA = z(H), z(FA)
        dk, ck = dprime_c(H, FA)
        # best additive shift: b = mean of the two z-shifts
        b = 0.5 * ((zH - zH0) + (zFA - zFA0))
        dc = ck - c0
        dd = dk - d0
        # fraction of the total (bias + discriminability) motion that is NOT bias
        frac = abs(dd) / (abs(b) + abs(dd)) if (abs(b) + abs(dd)) > 0 else 0.0
        comove = "yes" if (zH - zH0) * (zFA - zFA0) > 0 else "NO (d' move)"
        # sanity: additive model predicts dc == -b exactly
        assert abs(dc + b) < 1e-9, (dc, b)
        print(f"  S0->{k}  {dc:+.4f}    {b:+.4f}      {dd:+.4f}      "
              f"{frac:6.1%}            {comove}")
    print()
    print("  Reading: b (the pure-bias part) is large; Dd' (the residual,")
    print("  = the only non-additive part) is small. The drift is a criterion")
    print("  shift with a minor discriminability wobble -- exactly the")
    print("  'd' flat, c moves' signature. c == c0 - b holds to machine")
    print("  precision by construction (asserted), which is the Task-1 claim:")
    print("  additive bias moves c one-for-one and leaves d' fixed.")
    print()


# ----------------------------------------------------------------------
# (B) Anchor bias-equilibrium: symmetric (v2) vs one-sided (v1)
# ----------------------------------------------------------------------
def anchor_potentials(dprime=2.35, sigma=1.0, m=2.0):
    """Return callables V2, V1, and their forces, for a Gaussian model of the
    decision variable Delta with class means +/- dprime*sigma/2 (criterion-
    neutral at bias b=0) and common sd sigma. We integrate the softplus margin
    terms over the class-conditional Gaussians by Gauss-Hermite-free simple
    quadrature (dense grid)."""
    mu_pos = +0.5 * dprime * sigma
    mu_neg = -0.5 * dprime * sigma
    # dense grid for E over each conditional
    lo, hi, npts = -12.0, 12.0, 2401
    xs = [lo + (hi - lo) * i / (npts - 1) for i in range(npts)]
    dx = xs[1] - xs[0]
    w_pos = [math.exp(-0.5 * ((x - mu_pos) / sigma) ** 2) for x in xs]
    w_neg = [math.exp(-0.5 * ((x - mu_neg) / sigma) ** 2) for x in xs]
    Zp, Zn = sum(w_pos) * dx, sum(w_neg) * dx
    w_pos = [w / Zp for w in w_pos]
    w_neg = [w / Zn for w in w_neg]

    def E(f, w):
        return sum(f(x) * wi for x, wi in zip(xs, w)) * dx

    # v2: present term sp(m - (Delta+b)) on positives + absent term sp(m+(Delta+b)) on negatives
    def V2(b):
        return (E(lambda x: sp(m - (x + b)), w_pos)
                + E(lambda x: sp(m + (x + b)), w_neg))

    # v1: only the absent-side decision term (present decision term dropped)
    def V1(b):
        return E(lambda x: sp(m + (x + b)), w_neg)

    # forces = -dV/db (analytic derivative of softplus is sigmoid)
    def F2(b):
        return -(-E(lambda x: sig(m - (x + b)), w_pos)
                 + E(lambda x: sig(m + (x + b)), w_neg))

    def F1(b):
        return -(E(lambda x: sig(m + (x + b)), w_neg))

    return V2, V1, F2, F1, mu_pos, mu_neg


def bias_to_c(b, dprime=2.35, sigma=1.0):
    """Criterion c under bias b: means shift to mu_pos+b, mu_neg+b; with sigma=1,
    z(H)=mu_pos+b, z(FA)=mu_neg+b, c = -(z(H)+z(FA))/2 = -(0 + b) = -b for the
    criterion-neutral base. (d' = mu_pos-mu_neg unchanged.)"""
    zH = (0.5 * dprime * sigma + b) / sigma
    zFA = (-0.5 * dprime * sigma + b) / sigma
    return -0.5 * (zH + zFA)


def check_B():
    print("=" * 74)
    print("(B) Anchor bias-equilibrium: symmetric v2 vs one-sided v1")
    print("=" * 74)
    print("Gaussian decision variable, base d'=2.35 (pilot pooled), margin m=2.0.")
    print("b<0 means the criterion is pushed conservative (toward 'No').")
    print()
    V2, V1, F2, F1, mu_pos, mu_neg = anchor_potentials()
    grid = [(-8.0 + 0.01 * i) for i in range(1601)]  # b in [-8, 8]
    # v2 minimizer
    b2 = min(grid, key=V2)
    # curvature (finite diff)
    h = 1e-3
    curv2 = (V2(b2 + h) - 2 * V2(b2) + V2(b2 - h)) / h**2
    print(f"  v2 (symmetric): interior minimizer b* = {b2:+.3f}, "
          f"V''(b*) = {curv2:.3f} > 0 (stable)")
    print(f"                  equilibrium criterion c* = {bias_to_c(b2):+.3f} "
          f"(near-neutral, small offset)")
    print(f"                  force at b=0: F2(0) = {F2(0):+.4f} "
          f"(|.|~0 => no injected bias at neutral)")
    print()
    # v1 force is one-signed: report it never crosses zero on the grid
    f1vals = [F1(b) for b in grid]
    signs = set(1 if v > 1e-9 else (-1 if v < -1e-9 else 0) for v in f1vals)
    b1 = min(grid, key=V1)  # will sit at the conservative boundary of the grid
    print(f"  v1 (absent-side only): force sign set over b in [-8,8] = {signs} "
          f"(one-signed)")
    print(f"                  => NO interior equilibrium; potential minimized at "
          f"grid edge b={b1:+.2f}")
    print(f"                  driving c -> {bias_to_c(b1):+.2f} and beyond "
          f"(unbounded conservative)")
    print(f"                  force at b=0: F1(0) = {F1(0):+.4f} "
          f"(pushes 'No' even when already balanced)")
    print()
    print("  Retrodiction: v1 runs the criterion hyper-conservative (pilot: "
          "c=+2.02,")
    print("  yes-rate 7% at stage 1); v2 parks near-neutral with a small offset")
    print("  (pilot: yes-rate 33-36%, 'slightly conservative of base').")
    print()
    return V2, F2


def check_C(V2, F2):
    print("=" * 74)
    print("(C) Inverse-strength law: residual criterion drift ~ 1/lambda_F")
    print("=" * 74)
    print("Task-loss exerts a constant drift force f (toward 'yes', f>0) on the")
    print("bias. Anchored equilibrium solves lambda_F * F2(b) + f = 0.")
    print()
    # The symmetric anchor force is BOUNDED: F2(b) in (-1, 1) (a sum of sigmoids).
    # So an interior equilibrium exists only if f/lambda_F < sup|F2| ~ 1, i.e.
    # lambda_F > f. Below that the task drift overwhelms the anchor (no freeze).
    # This is a real, honest feature -- a bounded-strength anchor is defeasible.
    # Linearize F2 near b*=0: F2(b) ~ -kappa*b with kappa = V''(b*), giving the
    # small-drift law b_eq ~ f/(lambda_F*kappa), i.e. residual |c| ~ 1/lambda_F.
    f = 0.3
    h = 1e-3
    kappa = (V2(h) - 2 * V2(0.0) + V2(-h)) / h**2   # = V''(0) = stiffness
    Fmax = 0.999                                     # sup|F2| ~ 1 (bounded force)
    lam_crit = f / Fmax
    grid = [(-8.0 + 0.001 * i) for i in range(16001)]
    c_star = bias_to_c(0.0)
    print(f"  stiffness kappa = V''(0) = {kappa:.4f};  bounded force sup|F2| ~ 1")
    print(f"  => interior equilibrium requires lambda_F > f/sup|F2| ~ {lam_crit:.2f}")
    print(f"     (below that: anchor overwhelmed, criterion NOT frozen).")
    print(f"  linear-regime prediction: |c-c*| ~ f/(kappa*lambda_F), so")
    print(f"     (|c-c*| * lambda_F) -> f/kappa = {f / kappa:.3f} (constant).")
    print()
    print("   lambda_F   status         b_eq      |c-c*|   (|c-c*|*lam)  f/kappa")
    print("   " + "-" * 62)
    for lam in [0.2, 0.3, 0.5, 0.8, 1.6, 3.2, 6.4]:
        target = -f / lam
        if abs(target) >= Fmax:
            print(f"   {lam:6.2f}   OVERWHELMED    (no interior equilibrium; "
                  f"|f/lam|={abs(target):.2f} > sup|F2|)")
            continue
        b_eq = min(grid, key=lambda b: abs(F2(b) - target))
        c_eq = bias_to_c(b_eq)
        resid = abs(c_eq - c_star)
        print(f"   {lam:6.2f}   frozen         {b_eq:+.3f}   {resid:.4f}   "
              f"{resid * lam:7.4f}      {f / kappa:.3f}")
    print()
    print("  Reading: once lambda_F clears the overwhelm threshold, residual")
    print("  drift falls monotonically like 1/lambda_F and (|c-c*| * lambda_F)")
    print("  approaches the linear-regime constant f/kappa -- the anchor-")
    print("  strength law of Task 5, prediction 1. The bounded-force overwhelm")
    print("  regime at small lambda_F is a genuine caveat, not hidden.")
    print()


if __name__ == "__main__":
    check_A()
    V2, F2 = check_B()
    check_C(V2, F2)
    print("done.")
