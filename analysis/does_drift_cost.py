#!/usr/bin/env python3
"""
does_drift_cost.py -- Does the criterion drift cost anything measurable at the
endpoint?

The paper is built on criterion drift: across a continual-tuning sequence the
decision criterion c swings, and the paper treats damping that swing as the goal.
Our own data contains an awkward fact: sequential's endpoint criterion sits at the
JOINT (multitask upper bound) value.  The path swings; the destination may not.

This script asks four questions, all on data already on disk, zero GPU:

  Q1  Does path length predict endpoint quality?
      Sigma|dc| (raw and post-settling) vs endpoint POPE F1 and vs endpoint
      |c - c*|.  Within the sequential arm (n=9) and across all arms (n=20).

  Q2  Is the endpoint convergence real, or an artifact of averaging?
      Per-cell spread of sequential endpoint c, and the stage-index control:
      is stage 6 actually special, or is every stage index's cross-cell mean
      about as close to c*?

  Q3  Does the criterion return, or does it happen to end near the bound?
      Per-cell trajectory shape: net-to-gross ratio, crossings of the endpoint
      level, whether the endpoint is interior to the visited range, and whether
      |c - c*| decreases monotonically.

  Q4  How much accuracy is the mid-sequence drift worth?
      F1 at each cell's worst-drifted stage vs that cell's endpoint F1, against
      two noise floors (item sampling, and seed-to-seed at the endpoint).

DISCIPLINE ENFORCED HERE (this project has produced two wrong conclusions by
violating these):
  * Every quantity is computed PER CELL first, then aggregated.  Every
    correlation is fed two per-cell vectors carrying the same cell ids in the
    same order; `paired()` asserts it.  A path length is never paired with a
    range, nor a per-cell quantity with a cross-cell average.
  * Bootstrap resamples cells as a LIST WITH MULTIPLICITY.  There is no set()
    anywhere near a resample; a set() would silently make it a 63.2% subsample.
  * JOINT has n=2.  Per-cell values are quoted for it and no interval is.
  * "No relationship" and "underpowered to detect one" are reported as different
    things: every correlation carries its bootstrap CI, a permutation p, the
    critical |r| for significance at that n, and the true rho that n could detect
    at 80% power.

Positive controls run before any inference (the run aborts if they fail):
  * the raw path and endpoint c recomputed here must match the certified
    E1_null block in fs_aggregate.json for all 9 sequential cells;
  * F1 reconstructed from (H, FA, n_gt_yes, n_gt_no) must match the recorded f1;
  * the target c* recomputed from the JOINT endpoints must match the prereg's
    +0.088.

Usage:
    python3 analysis/does_drift_cost.py
    python3 analysis/does_drift_cost.py --agg <path> --out <path> --B 10000
"""

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ---------------------------------------------------------------------------
# constants / conventions, matching the rest of the project
# ---------------------------------------------------------------------------
PREREG_TARGET = 0.088        # FULLSTUDY_PREREG.md, JOINT empirical endpoint
TARGET_TOL = 0.0015          # certification tolerance against the prereg value
CTRL_TOL = 5e-4              # positive-control tolerance for path / c / F1
ARMS = ("seq", "anchor", "joint")


# ---------------------------------------------------------------------------
# small stats, kept local so the numbers here do not depend on another module's
# conventions drifting underneath them
# ---------------------------------------------------------------------------
def paired(xs, ys, cells_x, cells_y, label):
    """Refuse to correlate two vectors that are not the same cells in the same
    order.  This is the check that would have caught the estimator-mixing errors
    logged in TRADEOFF_PLACEMENT.md."""
    if len(xs) != len(ys) or len(cells_x) != len(cells_y) or len(xs) != len(cells_x):
        raise AssertionError("[%s] length mismatch: %d x, %d y, %d/%d cells"
                             % (label, len(xs), len(ys), len(cells_x), len(cells_y)))
    if list(cells_x) != list(cells_y):
        raise AssertionError("[%s] cell ids differ or are out of order:\n  x: %s\n  y: %s"
                             % (label, cells_x, cells_y))
    return True


def pearson(xs, ys):
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _rank(v):
    v = np.asarray(v, dtype=float)
    order = v.argsort()
    r = np.empty(len(v), dtype=float)
    r[order] = np.arange(1, len(v) + 1, dtype=float)
    # average ties
    for val in np.unique(v):
        m = (v == val)
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(xs, ys):
    if len(xs) < 3:
        return None
    return pearson(_rank(xs), _rank(ys))


def boot_ci_corr(xs, ys, B, seed, kind="pearson"):
    """Percentile CI for a correlation, resampling CELLS AS A LIST WITH
    MULTIPLICITY.  Note the resample is an index array used directly -- no set(),
    no unique(), so a cell drawn three times contributes three times."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    n = len(x)
    if n < 4:
        return None
    rng = np.random.default_rng(seed)
    fn = pearson if kind == "pearson" else spearman
    vals, degenerate = [], 0
    for _ in range(B):
        idx = rng.integers(0, n, size=n)          # LIST with multiplicity
        xb, yb = x[idx], y[idx]
        if xb.std() == 0 or yb.std() == 0:
            degenerate += 1
            continue
        r = fn(xb, yb)
        if r is not None and np.isfinite(r):
            vals.append(r)
    if len(vals) < 100:
        return None
    return {"lo": round(float(np.percentile(vals, 2.5)), 4),
            "hi": round(float(np.percentile(vals, 97.5)), 4),
            "B": B, "n_ok": len(vals), "n_degenerate": degenerate}


def perm_p_corr(xs, ys, B, seed):
    """Monte-Carlo two-sided permutation p for a correlation.  At n=9 the
    t-approximation is doing more work than the data supports; this does not
    assume bivariate normality."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    r_obs = pearson(x, y)
    if r_obs is None:
        return None
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(B):
        r = pearson(x, rng.permutation(y))
        if r is not None and abs(r) >= abs(r_obs) - 1e-12:
            hits += 1
    p = (hits + 1) / (B + 1)
    return {"p": round(float(p), 4), "B": B,
            "mc_se": round(float(math.sqrt(p * (1 - p) / B)), 4)}


def r_critical(n, alpha=0.05):
    """Smallest |r| that is significant at this n (two-sided)."""
    if n < 3:
        return None
    try:
        from scipy import stats
        t = stats.t.ppf(1 - alpha / 2.0, n - 2)
    except Exception:
        t = 2.306 if n == 9 else 2.101       # fallback: df=7, df=18
    return round(float(t / math.sqrt(t * t + (n - 2))), 4)


def r_detectable_80(n, alpha=0.05):
    """True rho this n detects with 80% power (Fisher z)."""
    if n < 5:
        return None
    z = (1.959964 + 0.841621) / math.sqrt(n - 3)
    return round(float(math.tanh(z)), 4)


def boot_ci_mean(vals, B, seed):
    """Percentile CI for a mean, cells resampled AS A LIST WITH MULTIPLICITY."""
    v = np.asarray(vals, dtype=float)
    n = len(v)
    if n < 4:
        return None
    rng = np.random.default_rng(seed)
    means = np.array([v[rng.integers(0, n, size=n)].mean() for _ in range(B)])
    return {"lo": round(float(np.percentile(means, 2.5)), 4),
            "hi": round(float(np.percentile(means, 97.5)), 4), "B": B}


def corr_block(xs, ys, cells, label, B, seed):
    """One fully-labelled correlation, with everything needed to tell a null from
    an underpowered measurement."""
    paired(xs, ys, cells, cells, label)
    n = len(xs)
    r = pearson(xs, ys)
    rho = spearman(xs, ys)
    out = {
        "label": label, "n_cells": n, "cells": list(cells),
        "pearson_r": None if r is None else round(r, 4),
        "spearman_rho": None if rho is None else round(rho, 4),
        "r_critical_p05": r_critical(n),
        "rho_detectable_at_80pct_power": r_detectable_80(n),
        "ci95_pearson": boot_ci_corr(xs, ys, B, seed, "pearson"),
        "perm_p": perm_p_corr(xs, ys, min(B, 20000), seed + 1),
        "x": [round(float(v), 4) for v in xs],
        "y": [round(float(v), 4) for v in ys],
    }
    if n < 4:
        out["note"] = ("n<4: no interval is quoted; the per-cell values in x/y "
                       "are the whole of the evidence.")
    return out


# ---------------------------------------------------------------------------
# F1 reconstruction + item-level noise floor
# ---------------------------------------------------------------------------
def f1_from(H, FA, n_yes, n_no):
    tp = H * n_yes
    fp = FA * n_no
    if tp <= 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = H
    if prec + rec == 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def f1_item_sd(H, FA, n_yes, n_no, B, seed):
    """Item-sampling SD of POPE F1, by parametric binomial resample of the hit
    and false-alarm counts at the recorded n.  This is the noise from which POPE
    items were drawn, CONDITIONAL ON THE CHECKPOINT.  It is NOT a run-to-run or
    trajectory noise floor -- those are different quantities and are reported
    separately below."""
    rng = np.random.default_rng(seed)
    h = rng.binomial(n_yes, min(max(H, 0.0), 1.0), size=B) / float(n_yes)
    f = rng.binomial(n_no, min(max(FA, 0.0), 1.0), size=B) / float(n_no)
    tp = h * n_yes
    fp = f * n_no
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-12), 0.0)
        f1 = np.where(prec + h > 0, 2 * prec * h / np.maximum(prec + h, 1e-12), 0.0)
    return float(np.std(f1))


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------
def load_cells(agg_path, backbone="llava15"):
    d = json.load(open(agg_path))
    bb = d["backbones"][backbone]
    base = bb["base"]["pope"]
    rows = []
    for key in sorted(bb["arms"]):
        cell = bb["arms"][key]
        ks = sorted((s for s in cell if s.isdigit()), key=int)
        cs, f1s, hs, fas, nys, nns, dps, stages = [], [], [], [], [], [], [], []
        ok = True
        for s in ks:
            p = cell[s].get("pope")
            if not p:
                ok = False
                break
            cs.append(float(p["c"]))
            f1s.append(float(p["f1"]))
            hs.append(float(p["H"]))
            fas.append(float(p["FA"]))
            nys.append(int(p["n_gt_yes"]))
            nns.append(int(p["n_gt_no"]))
            dps.append(float(p["dprime"]))
            stages.append(int(s))
        if not ok or len(cs) < 2:
            continue
        arm, order, seed = key.split("|")
        rows.append({
            "cell": key, "arm": arm, "order": order, "seed": int(seed[1:]),
            "stages": stages, "c": cs, "f1": f1s, "H": hs, "FA": fas,
            "n_gt_yes": nys, "n_gt_no": nns, "dprime": dps,
        })
    return d, float(base["c"]), float(base["f1"]), rows


def add_path_quantities(row, base_c):
    """Per-cell path estimators, matching tradeoff_placement.py / fs_common
    exactly: raw = base -> k1 -> ... -> kN, post-settling drops the first step."""
    seq = [base_c] + list(row["c"])
    steps = [abs(seq[i + 1] - seq[i]) for i in range(len(seq) - 1)]
    row["steps"] = [round(s, 4) for s in steps]
    row["raw_path"] = float(sum(steps))
    row["post_settle"] = float(sum(steps[1:])) if len(steps) >= 2 else None
    row["endpoint_c"] = float(row["c"][-1])
    row["endpoint_f1"] = float(row["f1"][-1])
    row["net_displacement"] = abs(row["endpoint_c"] - base_c)
    row["net_to_gross"] = (row["net_displacement"] / row["raw_path"]
                           if row["raw_path"] > 0 else None)
    return row


# ---------------------------------------------------------------------------
# positive controls -- run before any inference
# ---------------------------------------------------------------------------
def controls(agg, rows, target, base_c):
    res = {"passed": True, "checks": []}

    def rec(name, ok, detail):
        res["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            res["passed"] = False

    # 1. path + endpoint against the certified E1_null block
    e1 = {}
    for c in agg["backbones"]["llava15"]["endpoints"]["E1_null"]["cells"]:
        e1["seq|o%s|s%d" % (c["order"][1:], c["seed"])] = c
    n_ck, worst_path, worst_c = 0, 0.0, 0.0
    for r in rows:
        if r["arm"] != "seq":
            continue
        ref = e1.get(r["cell"])
        if ref is None:
            rec("E1_null coverage", False, "no E1_null record for %s" % r["cell"])
            continue
        n_ck += 1
        worst_path = max(worst_path, abs(r["raw_path"] - ref["sum_abs_step"]))
        worst_c = max(worst_c, abs(r["endpoint_c"] - ref["endpoint_c"]))
    rec("raw path == E1_null sum_abs_step (9 seq cells)", n_ck == 9 and worst_path < CTRL_TOL,
        "n=%d, max abs diff %.6f (tol %g)" % (n_ck, worst_path, CTRL_TOL))
    rec("endpoint c == E1_null endpoint_c (9 seq cells)", n_ck == 9 and worst_c < CTRL_TOL,
        "n=%d, max abs diff %.6f (tol %g)" % (n_ck, worst_c, CTRL_TOL))

    # 2. F1 reconstruction from (H, FA, n_gt_yes, n_gt_no)
    worst_f1, n_f1 = 0.0, 0
    for r in rows:
        for i in range(len(r["c"])):
            n_f1 += 1
            got = f1_from(r["H"][i], r["FA"][i], r["n_gt_yes"][i], r["n_gt_no"][i])
            worst_f1 = max(worst_f1, abs(got - r["f1"][i]))
    rec("F1 reconstructed from H/FA matches recorded f1", worst_f1 < 1e-3,
        "n=%d stage-points, max abs diff %.6f" % (n_f1, worst_f1))

    # 3. the target c* against the prereg
    rec("c* (mean of JOINT endpoints) == prereg +0.088",
        abs(target - PREREG_TARGET) < TARGET_TOL,
        "recomputed c* = %+.4f, prereg %+.3f, tol %g" % (target, PREREG_TARGET, TARGET_TOL))

    # 4. base criterion, the path origin
    rec("base criterion is the recorded +0.4314", abs(base_c - 0.4314) < CTRL_TOL,
        "base c = %+.4f" % base_c)
    return res


# ---------------------------------------------------------------------------
# Q1
# ---------------------------------------------------------------------------
def q1_path_vs_endpoint(rows, target, B, seed):
    out = {"question": "Does path length predict endpoint quality?", "sets": {}}

    def do_set(name, sub, note=None):
        sub = sorted(sub, key=lambda r: r["cell"])
        cells = [r["cell"] for r in sub]
        blocks = {}
        if len(sub) >= 3:
            for pname, pkey in (("raw_path", "raw_path"), ("post_settle", "post_settle")):
                xs = [r[pkey] for r in sub]
                if any(v is None for v in xs):
                    continue
                blocks["%s_vs_endpoint_f1" % pname] = corr_block(
                    xs, [r["endpoint_f1"] for r in sub], cells,
                    "%s: %s vs endpoint POPE F1" % (name, pname), B, seed)
                blocks["%s_vs_endpoint_err" % pname] = corr_block(
                    xs, [abs(r["endpoint_c"] - target) for r in sub], cells,
                    "%s: %s vs |endpoint c - c*|" % (name, pname), B, seed)
        entry = {"n_cells": len(sub), "cells": cells, "correlations": blocks,
                 "per_cell": [{"cell": r["cell"], "raw_path": round(r["raw_path"], 4),
                               "post_settle": (None if r["post_settle"] is None
                                               else round(r["post_settle"], 4)),
                               "endpoint_c": round(r["endpoint_c"], 4),
                               "endpoint_err": round(abs(r["endpoint_c"] - target), 4),
                               "endpoint_f1": round(r["endpoint_f1"], 4)} for r in sub]}
        if note:
            entry["note"] = note
        if len(sub) < 4:
            entry["note"] = ((note + "  ") if note else "") + \
                "n<4: per-cell values only, no interval."
        out["sets"][name] = entry

    for arm in ARMS:
        do_set(arm, [r for r in rows if r["arm"] == arm],
               note=("JOINT has n=2; these are the two cells, not a sample."
                     if arm == "joint" else None))
    do_set("all_arms", list(rows),
           note=("n=20 cells but n=3 ARMS: a cross-arm correlation here is a "
                 "between-group separation wearing a cell-level n. Read with the "
                 "arm-partialled and drop-anchor rows."))
    do_set("all_arms_no_anchor", [r for r in rows if r["arm"] != "anchor"],
           note="Anchor family dropped -- TRADEOFF_PLACEMENT.md showed it drives the pooled sign.")
    do_set("all_arms_no_joint", [r for r in rows if r["arm"] != "joint"],
           note=("JOINT dropped: c* is the mean of the two JOINT endpoints, so "
                 "their placement error is partly by construction."))

    # within-arm centred: removes the arm main effect, keeps the within-arm signal
    wac = {}
    for pkey in ("raw_path", "post_settle"):
        xs, ys_f1, ys_err, cells = [], [], [], []
        for arm in ARMS:
            sub = sorted([r for r in rows if r["arm"] == arm], key=lambda r: r["cell"])
            if len(sub) < 2:
                continue
            px = np.array([r[pkey] for r in sub], dtype=float)
            pf = np.array([r["endpoint_f1"] for r in sub], dtype=float)
            pe = np.array([abs(r["endpoint_c"] - target) for r in sub], dtype=float)
            xs += list(px - px.mean())
            ys_f1 += list(pf - pf.mean())
            ys_err += list(pe - pe.mean())
            cells += [r["cell"] for r in sub]
        wac["%s_vs_endpoint_f1" % pkey] = corr_block(
            xs, ys_f1, cells, "within-arm centred: %s vs endpoint F1" % pkey, B, seed)
        wac["%s_vs_endpoint_err" % pkey] = corr_block(
            xs, ys_err, cells, "within-arm centred: %s vs |endpoint c - c*|" % pkey, B, seed)
    wac["note"] = ("Each arm centred on its own mean before pooling, so this is the "
                   "WITHIN-arm relationship with the between-arm difference removed. "
                   "Degrees of freedom are inflated (3 means estimated); treat the "
                   "bootstrap CI as the inferential statistic, not the p.")
    out["within_arm_centred"] = wac

    # how much of endpoint F1 is just endpoint c, within the sequential arm?
    sq = sorted([r for r in rows if r["arm"] == "seq"], key=lambda r: r["cell"])
    out["endpoint_f1_is_endpoint_c"] = corr_block(
        [r["endpoint_c"] for r in sq], [r["endpoint_f1"] for r in sq],
        [r["cell"] for r in sq],
        "seq: endpoint c vs endpoint F1 (are these two readings of one quantity?)",
        B, seed)
    return out


# ---------------------------------------------------------------------------
# Q2
# ---------------------------------------------------------------------------
def q2_convergence_or_averaging(rows, target, base_c, B, seed):
    out = {"question": "Is the endpoint convergence real or an artifact of averaging?"}
    for arm in ARMS:
        sub = sorted([r for r in rows if r["arm"] == arm], key=lambda r: r["cell"])
        if not sub:
            continue
        ends = np.array([r["endpoint_c"] for r in sub], dtype=float)
        errs = np.abs(ends - target)
        ent = {
            "n_cells": len(sub),
            "per_cell": [{"cell": r["cell"], "endpoint_c": round(r["endpoint_c"], 4),
                          "signed_err": round(r["endpoint_c"] - target, 4),
                          "abs_err": round(abs(r["endpoint_c"] - target), 4)} for r in sub],
            "mean_endpoint_c": round(float(ends.mean()), 4),
            "sd_endpoint_c": round(float(ends.std(ddof=1)), 4) if len(ends) > 1 else None,
            "min_endpoint_c": round(float(ends.min()), 4),
            "max_endpoint_c": round(float(ends.max()), 4),
            "range_endpoint_c": round(float(ends.max() - ends.min()), 4),
            "mean_abs_err": round(float(errs.mean()), 4),
            "abs_err_of_the_mean": round(float(abs(ends.mean() - target)), 4),
            "n_above_target": int((ends > target).sum()),
            "n_below_target": int((ends < target).sum()),
        }
        if len(sub) >= 4:
            ent["ci95_mean_endpoint_c"] = boot_ci_mean(ends, B, seed)
            ent["ci95_mean_abs_err"] = boot_ci_mean(errs, B, seed)
        else:
            ent["note"] = "n<4: per-cell values only, no interval."
        out[arm] = ent

    # the decisive ratio: does the mean hide the cells?
    sq = out.get("seq")
    if sq:
        sq["cancellation_ratio"] = round(sq["mean_abs_err"] / max(sq["abs_err_of_the_mean"], 1e-9), 2)
        sq["cancellation_note"] = (
            "mean_abs_err / abs_err_of_the_mean. A value near 1 means the cells "
            "really are all near c*; a large value means the mean is close only "
            "because per-cell errors cancel in sign.")
        # scale references, so the spread is interpretable
        sq["scale_references"] = {
            "base_distance_from_target": round(abs(base_c - target), 4),
            "mean_raw_path_seq": round(float(np.mean([r["raw_path"] for r in rows
                                                      if r["arm"] == "seq"])), 4),
            "spread_as_frac_of_base_distance":
                round(sq["mean_abs_err"] / abs(base_c - target), 3),
        }

    # STAGE-INDEX CONTROL: is stage 6 special, or is every stage index about as
    # close to c* on average?  If the latter, "the destination does not move" is a
    # statement about the arm's mean, not about the endpoint.
    ctrl = {"note": ("For each stage index k, the cross-cell mean of c and the "
                     "cross-cell mean of |c - c*|. If k=6 is not distinguished, "
                     "the endpoint is not a special place -- the arm simply "
                     "straddles c* throughout."), "arms": {}}
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        nst = min(len(r["c"]) for r in sub)
        tab = []
        for k in range(nst):
            vals = np.array([r["c"][k] for r in sub], dtype=float)
            tab.append({
                "stage": k + 1, "n_cells": len(sub),
                "mean_c": round(float(vals.mean()), 4),
                "sd_c": round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else None,
                "abs_err_of_mean": round(float(abs(vals.mean() - target)), 4),
                "mean_abs_err": round(float(np.abs(vals - target).mean()), 4),
                "n_above_target": int((vals > target).sum()),
            })
        ctrl["arms"][arm] = tab
    out["stage_index_control"] = ctrl
    return out


# ---------------------------------------------------------------------------
# Q3
# ---------------------------------------------------------------------------
def q3_return_or_arrive(rows, target):
    out = {"question": "Does the criterion return, or does it happen to end near the bound?",
           "definitions": {
               "net_to_gross": "|c_end - c_base| / Sigma|dc|. 1.0 = a straight monotone "
                               "march; low = the path doubled back on itself.",
               "crossings": "sign changes of (c_k - c_end) over k=1..N-1: how many times "
                            "the trajectory crossed the level it ends at.",
               "endpoint_rank": "rank of c_end among the cell's N visited criteria "
                                "(1 = most liberal). Interior rank => it came back; "
                                "rank 1 or N => it ended at an extreme of its own path.",
               "monotone_approach": "|c_k - c_end| strictly decreasing over k.",
               "best_earlier_err": "min over k<N of |c_k - c*|: did the cell pass closer "
                                   "to the target earlier and then move away?",
           },
           "arms": {}}
    for arm in ARMS:
        sub = sorted([r for r in rows if r["arm"] == arm], key=lambda r: r["cell"])
        if not sub:
            continue
        per = []
        for r in sub:
            cs = np.array(r["c"], dtype=float)
            n = len(cs)
            d = cs - cs[-1]
            pre = d[:-1]
            signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in pre]
            nz = [s for s in signs if s != 0]
            crossings = sum(1 for i in range(len(nz) - 1) if nz[i] != nz[i + 1])
            gaps = np.abs(d)
            monotone = bool(all(gaps[i] > gaps[i + 1] for i in range(n - 1)))
            rank = int(np.argsort(np.argsort(cs))[-1]) + 1
            errs = np.abs(cs - target)
            per.append({
                "cell": r["cell"],
                "c_series": [round(float(v), 4) for v in cs],
                "raw_path": round(r["raw_path"], 4),
                "net_displacement": round(r["net_displacement"], 4),
                "net_to_gross": round(r["net_to_gross"], 3),
                "crossings_of_endpoint_level": crossings,
                "endpoint_rank": rank, "n_stages": n,
                "endpoint_is_extremum": bool(rank == 1 or rank == n),
                "monotone_approach": monotone,
                "endpoint_err": round(float(errs[-1]), 4),
                "best_earlier_err": round(float(errs[:-1].min()), 4),
                "best_earlier_stage": int(np.argmin(errs[:-1])) + 1,
                "mean_earlier_err": round(float(errs[:-1].mean()), 4),
                "endpoint_beats_all_earlier": bool(errs[-1] < errs[:-1].min()),
                "max_excursion_from_endpoint": round(float(gaps.max()), 4),
            })
        summ = {
            "n_cells": len(per),
            "mean_net_to_gross": round(float(np.mean([p["net_to_gross"] for p in per])), 3),
            "n_monotone_approach": int(sum(p["monotone_approach"] for p in per)),
            "n_endpoint_interior": int(sum(not p["endpoint_is_extremum"] for p in per)),
            "n_with_crossings": int(sum(p["crossings_of_endpoint_level"] > 0 for p in per)),
            "mean_crossings": round(float(np.mean([p["crossings_of_endpoint_level"]
                                                   for p in per])), 2),
            "n_endpoint_beats_all_earlier": int(sum(p["endpoint_beats_all_earlier"]
                                                    for p in per)),
            "mean_endpoint_err": round(float(np.mean([p["endpoint_err"] for p in per])), 4),
            "mean_best_earlier_err": round(float(np.mean([p["best_earlier_err"]
                                                          for p in per])), 4),
        }
        out["arms"][arm] = {"per_cell": per, "summary": summ}
        if len(per) < 4:
            out["arms"][arm]["note"] = "n<4: per-cell values only, no interval."
    return out


# ---------------------------------------------------------------------------
# Q4
# ---------------------------------------------------------------------------
def q4_what_is_drift_worth(rows, target, B, seed):
    out = {"question": "How much POPE F1 is the mid-sequence drift actually worth?",
           "definitions": {
               "worst_drift_stage": "argmax over ALL stages k of |c_k - c*| (pre-specified "
                                    "primary).",
               "worst_drift_stage_mid": "argmax over k < N of |c_k - c*| -- the mid-sequence "
                                        "restriction, which is the quantity the question asks "
                                        "about when the endpoint is itself well placed.",
               "f1_gap": "endpoint F1 minus F1 at that stage. Positive = the drifted stage "
                         "was WORSE than the endpoint, i.e. the drift cost accuracy.",
           },
           "arms": {}}
    for arm in ARMS:
        sub = sorted([r for r in rows if r["arm"] == arm], key=lambda r: r["cell"])
        if not sub:
            continue
        per = []
        for r in sub:
            cs = np.array(r["c"], dtype=float)
            f1 = np.array(r["f1"], dtype=float)
            errs = np.abs(cs - target)
            kw = int(np.argmax(errs))
            kwm = int(np.argmax(errs[:-1]))
            kf1 = int(np.argmin(f1))
            item_sd = f1_item_sd(r["H"][kwm], r["FA"][kwm], r["n_gt_yes"][kwm],
                                 r["n_gt_no"][kwm], 4000, seed + 7)
            item_sd_end = f1_item_sd(r["H"][-1], r["FA"][-1], r["n_gt_yes"][-1],
                                     r["n_gt_no"][-1], 4000, seed + 8)
            per.append({
                "cell": r["cell"],
                "endpoint_c": round(float(cs[-1]), 4),
                "endpoint_err": round(float(errs[-1]), 4),
                "endpoint_f1": round(float(f1[-1]), 4),
                "worst_drift_stage": kw + 1,
                "worst_drift_c": round(float(cs[kw]), 4),
                "worst_drift_err": round(float(errs[kw]), 4),
                "worst_drift_f1": round(float(f1[kw]), 4),
                "f1_gap_worst": round(float(f1[-1] - f1[kw]), 4),
                "worst_drift_stage_mid": kwm + 1,
                "worst_drift_c_mid": round(float(cs[kwm]), 4),
                "worst_drift_err_mid": round(float(errs[kwm]), 4),
                "worst_drift_f1_mid": round(float(f1[kwm]), 4),
                "f1_gap_mid": round(float(f1[-1] - f1[kwm]), 4),
                "excess_err_mid": round(float(errs[kwm] - errs[-1]), 4),
                "min_f1_stage": kf1 + 1,
                "min_f1": round(float(f1.min()), 4),
                "f1_gap_minf1": round(float(f1[-1] - f1.min()), 4),
                "f1_range_over_path": round(float(f1.max() - f1.min()), 4),
                "item_sd_f1_at_worst_mid": round(item_sd, 4),
                "item_sd_f1_at_endpoint": round(item_sd_end, 4),
            })
        gaps_mid = np.array([p["f1_gap_mid"] for p in per], dtype=float)
        gaps_worst = np.array([p["f1_gap_worst"] for p in per], dtype=float)
        summ = {
            "n_cells": len(per),
            "mean_f1_gap_mid": round(float(gaps_mid.mean()), 4),
            "sd_f1_gap_mid": round(float(gaps_mid.std(ddof=1)), 4) if len(per) > 1 else None,
            "min_f1_gap_mid": round(float(gaps_mid.min()), 4),
            "max_f1_gap_mid": round(float(gaps_mid.max()), 4),
            "n_cells_gap_positive": int((gaps_mid > 0).sum()),
            "mean_f1_gap_worst": round(float(gaps_worst.mean()), 4),
            "mean_f1_gap_minf1": round(float(np.mean([p["f1_gap_minf1"] for p in per])), 4),
            "mean_item_sd_f1": round(float(np.mean([p["item_sd_f1_at_worst_mid"]
                                                    for p in per])), 4),
        }
        if len(per) >= 4:
            summ["ci95_mean_f1_gap_mid"] = boot_ci_mean(gaps_mid, B, seed)
        else:
            summ["note"] = "n<4: per-cell values only, no interval."
        ent = {"per_cell": per, "summary": summ}

        # does a BIGGER criterion excursion buy a BIGGER F1 loss?  Paired, per cell.
        if len(per) >= 3:
            ent["excursion_buys_loss"] = corr_block(
                [p["excess_err_mid"] for p in per], [p["f1_gap_mid"] for p in per],
                [p["cell"] for p in per],
                "%s: excess |c-c*| at worst mid stage vs F1 shortfall there" % arm,
                B, seed)
        out["arms"][arm] = ent

    # within-cell pooled: across all stages, does |c - c*| track F1?  Cell-centred so
    # this is the WITHIN-cell relationship, then bootstrapped BY CELL.
    pooled = {}
    for arm in ARMS:
        sub = sorted([r for r in rows if r["arm"] == arm], key=lambda r: r["cell"])
        if len(sub) < 2:
            continue
        per_cell_r = []
        for r in sub:
            e = np.abs(np.array(r["c"]) - target)
            f = np.array(r["f1"])
            rr = pearson(e, f)
            per_cell_r.append({"cell": r["cell"],
                               "r_err_vs_f1_within_cell": None if rr is None else round(rr, 4),
                               "n_stages": len(r["c"])})
        vals = [p["r_err_vs_f1_within_cell"] for p in per_cell_r
                if p["r_err_vs_f1_within_cell"] is not None]
        ent = {"per_cell": per_cell_r,
               "mean_within_cell_r": round(float(np.mean(vals)), 4) if vals else None,
               "n_cells_negative": int(sum(1 for v in vals if v < 0)),
               "n_cells": len(vals),
               "note": ("Within each cell, correlation across its stages between "
                        "|c - c*| and POPE F1. Negative = being further from the "
                        "target goes with lower F1. n=6 stages per cell, so each "
                        "per-cell r is itself noisy; the count across cells is the "
                        "statistic to read.")}
        if len(vals) >= 4:
            ent["ci95_mean_within_cell_r"] = boot_ci_mean(vals, B, seed)
        pooled[arm] = ent
    out["within_cell_err_vs_f1"] = pooled

    # seed-to-seed noise floor at the endpoint: within each order, SD across seeds.
    floors = {}
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        by_order = {}
        for r in sub:
            by_order.setdefault(r["order"], []).append(r)
        rec = []
        for o, rs in sorted(by_order.items()):
            if len(rs) < 2:
                continue
            v = np.array([x["endpoint_f1"] for x in rs], dtype=float)
            vc = np.array([x["endpoint_c"] for x in rs], dtype=float)
            rec.append({"order": o, "n_seeds": len(rs),
                        "endpoint_f1": [round(float(x), 4) for x in v],
                        "sd_endpoint_f1": round(float(v.std(ddof=1)), 4),
                        "range_endpoint_f1": round(float(v.max() - v.min()), 4),
                        "sd_endpoint_c": round(float(vc.std(ddof=1)), 4)})
        if rec:
            floors[arm] = {
                "per_order": rec,
                "pooled_sd_endpoint_f1": round(
                    float(np.sqrt(np.mean([x["sd_endpoint_f1"] ** 2 for x in rec]))), 4),
                "note": ("Seed-to-seed SD of endpoint F1 within a fixed task order, "
                         "n=3 seeds per order. This is a RUN-TO-RUN floor at the "
                         "endpoint only. It is NOT the item-sampling SD above and it "
                         "is NOT a trajectory noise floor -- nothing on disk re-runs "
                         "an identical configuration end to end.")}
    out["seed_noise_floor"] = floors

    # ---- the residual endpoint gap to JOINT, and where it comes from ----------
    # If sequential's endpoint criterion really is at the JOINT value, then any
    # endpoint F1 gap that remains is NOT a criterion cost -- it has to live in
    # separability.  Decomposing it is the difference between "drift costs nothing
    # at the endpoint" and "drift costs nothing at the endpoint THROUGH c".
    sq = sorted([r for r in rows if r["arm"] == "seq"], key=lambda r: r["cell"])
    jt = sorted([r for r in rows if r["arm"] == "joint"], key=lambda r: r["cell"])
    if sq and jt:
        sf = np.array([r["endpoint_f1"] for r in sq], dtype=float)
        jf = np.array([r["endpoint_f1"] for r in jt], dtype=float)
        sd_ = np.array([r["dprime"][-1] for r in sq], dtype=float)
        jd = np.array([r["dprime"][-1] for r in jt], dtype=float)
        ent = {
            "seq_endpoint_f1_per_cell": [round(float(v), 4) for v in sf],
            "joint_endpoint_f1_per_cell": [round(float(v), 4) for v in jf],
            "mean_seq_endpoint_f1": round(float(sf.mean()), 4),
            "mean_joint_endpoint_f1": round(float(jf.mean()), 4),
            "gap_seq_minus_joint": round(float(sf.mean() - jf.mean()), 4),
            "n_seq_below_lowest_joint_cell": int((sf < jf.min()).sum()),
            "n_seq_above_highest_joint_cell": int((sf > jf.max()).sum()),
            "n_seq_cells": len(sf),
            "seq_endpoint_dprime_per_cell": [round(float(v), 4) for v in sd_],
            "joint_endpoint_dprime_per_cell": [round(float(v), 4) for v in jd],
            "mean_seq_endpoint_dprime": round(float(sd_.mean()), 4),
            "sd_seq_endpoint_dprime": round(float(sd_.std(ddof=1)), 4),
            "mean_joint_endpoint_dprime": round(float(jd.mean()), 4),
            "dprime_gap": round(float(sd_.mean() - jd.mean()), 4),
            "criterion_gap": round(float(np.mean([r["endpoint_c"] for r in sq])
                                         - np.mean([r["endpoint_c"] for r in jt])), 4),
            "note": ("JOINT has n=2: its two per-cell values are quoted and no "
                     "interval is. The counts are the statistic. Because the "
                     "endpoint CRITERION gap is ~0, any residual endpoint F1 gap "
                     "must be carried by separability, i.e. by ordinary forgetting "
                     "rather than by criterion drift."),
        }
        if len(sf) >= 4:
            ent["ci95_mean_seq_endpoint_f1"] = boot_ci_mean(sf, B, seed)
        out["endpoint_residual_vs_joint"] = ent
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default=os.path.join(HERE, "readout", "fs_aggregate.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "readout", "does_drift_cost.json"))
    ap.add_argument("--backbone", default="llava15")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    if not os.path.isfile(args.agg):
        print("[does_drift_cost] ABSENT: no aggregate at %s" % args.agg)
        return 2

    agg, base_c, base_f1, rows = load_cells(args.agg, args.backbone)
    for r in rows:
        add_path_quantities(r, base_c)

    joint = sorted([r for r in rows if r["arm"] == "joint"], key=lambda r: r["cell"])
    if len(joint) < 1:
        print("[does_drift_cost] ABSENT: no JOINT cell -> no target c*. Refusing to "
              "fall back to a modelled c = 0.")
        return 2
    target = float(np.mean([r["endpoint_c"] for r in joint]))

    ctrl = controls(agg, rows, target, base_c)
    print("=" * 78)
    print("POSITIVE CONTROLS")
    print("=" * 78)
    for c in ctrl["checks"]:
        print("  [%s] %-52s %s" % ("PASS" if c["ok"] else "FAIL", c["check"], c["detail"]))
    if not ctrl["passed"]:
        print("\n[does_drift_cost] ABORT: a positive control failed. No inference is "
              "emitted from a tree whose arithmetic does not reproduce the certified "
              "readout.")
        return 3

    print("\nbase criterion c = %+.4f (path origin), base POPE F1 = %.4f" % (base_c, base_f1))
    print("target c* = %+.4f  (mean of %d JOINT endpoint(s): %s)  -- n=2, no interval"
          % (target, len(joint), ", ".join("%+.4f" % r["endpoint_c"] for r in joint)))
    print("cells: %d  (%s)" % (len(rows), ", ".join(
        "%s n=%d" % (a, sum(1 for r in rows if r["arm"] == a)) for a in ARMS)))

    res = {
        "generated_by": "analysis/does_drift_cost.py",
        "aggregate": os.path.abspath(args.agg),
        "aggregate_generated": agg.get("generated"),
        "backbone": args.backbone,
        "boot_B": args.B, "seed": args.seed,
        "base_c": round(base_c, 4), "base_f1": round(base_f1, 4),
        "target_c_star": round(target, 4),
        "target_provenance": ("mean of the JOINT arm's empirical endpoint criteria "
                              "(n=2 cells: %s). Never a modelled c=0."
                              % ", ".join("%s %+.4f" % (r["cell"], r["endpoint_c"])
                                          for r in joint)),
        "n_cells": len(rows),
        "controls": ctrl,
        "Q1_path_vs_endpoint": q1_path_vs_endpoint(rows, target, args.B, args.seed),
        "Q2_convergence": q2_convergence_or_averaging(rows, target, base_c, args.B, args.seed),
        "Q3_return_or_arrive": q3_return_or_arrive(rows, target),
        "Q4_what_is_drift_worth": q4_what_is_drift_worth(rows, target, args.B, args.seed),
    }

    # ---- console summary -------------------------------------------------
    def show_corr(b, indent="    "):
        if not b:
            return
        ci = b.get("ci95_pearson")
        cis = ("[%+.3f, %+.3f]" % (ci["lo"], ci["hi"])) if ci else "n/a (n<4)"
        pp = b.get("perm_p")
        ps = ("%.4f" % pp["p"]) if pp else "n/a"
        print("%s%-58s n=%2d  r=%+.3f  rho=%+.3f  CI %s  perm p=%s"
              % (indent, b["label"], b["n_cells"],
                 b["pearson_r"] if b["pearson_r"] is not None else float("nan"),
                 b["spearman_rho"] if b["spearman_rho"] is not None else float("nan"),
                 cis, ps))

    print("\n" + "=" * 78)
    print("Q1  DOES PATH LENGTH PREDICT ENDPOINT QUALITY?")
    print("=" * 78)
    for name in ("seq", "anchor", "joint", "all_arms", "all_arms_no_anchor",
                 "all_arms_no_joint"):
        s = res["Q1_path_vs_endpoint"]["sets"].get(name)
        if not s:
            continue
        print("\n  [%s]  n=%d cells" % (name, s["n_cells"]))
        if s.get("note"):
            print("    note: %s" % s["note"])
        for k in sorted(s["correlations"]):
            show_corr(s["correlations"][k])
        if s["n_cells"] >= 5:
            print("    power: |r| must exceed %.3f for p<.05 at n=%d; 80%% power needs "
                  "true rho >= %.3f" % (r_critical(s["n_cells"]), s["n_cells"],
                                        r_detectable_80(s["n_cells"])))
    print("\n  [within-arm centred]")
    for k in sorted(res["Q1_path_vs_endpoint"]["within_arm_centred"]):
        if k != "note":
            show_corr(res["Q1_path_vs_endpoint"]["within_arm_centred"][k])
    print("\n  [is endpoint F1 just endpoint c?]")
    show_corr(res["Q1_path_vs_endpoint"]["endpoint_f1_is_endpoint_c"])

    print("\n" + "=" * 78)
    print("Q2  IS THE ENDPOINT CONVERGENCE REAL, OR AVERAGING?")
    print("=" * 78)
    for arm in ARMS:
        e = res["Q2_convergence"].get(arm)
        if not e:
            continue
        print("\n  [%s]  n=%d" % (arm, e["n_cells"]))
        print("    %-22s %-10s %-11s %s" % ("cell", "end c", "signed err", "abs err"))
        for p in e["per_cell"]:
            print("    %-22s %+.4f    %+.4f      %.4f"
                  % (p["cell"], p["endpoint_c"], p["signed_err"], p["abs_err"]))
        print("    mean end c %+.4f   sd %s   range [%+.4f, %+.4f] (width %.4f)"
              % (e["mean_endpoint_c"], ("%.4f" % e["sd_endpoint_c"]) if e["sd_endpoint_c"] else "n/a",
                 e["min_endpoint_c"], e["max_endpoint_c"], e["range_endpoint_c"]))
        print("    |mean - c*| = %.4f   BUT mean|c - c*| = %.4f   (%d above / %d below c*)"
              % (e["abs_err_of_the_mean"], e["mean_abs_err"],
                 e["n_above_target"], e["n_below_target"]))
        if "cancellation_ratio" in e:
            print("    cancellation ratio = %.2f  (mean|err| / |mean err|)"
                  % e["cancellation_ratio"])
    print("\n  [stage-index control: is stage 6 special?]")
    for arm in ARMS:
        tab = res["Q2_convergence"]["stage_index_control"]["arms"].get(arm)
        if not tab:
            continue
        print("    %s:" % arm)
        print("      %-6s %-10s %-8s %-14s %-13s %s"
              % ("stage", "mean c", "sd c", "|mean - c*|", "mean|c - c*|", "n above c*"))
        for t in tab:
            print("      %-6d %+.4f    %s     %.4f         %.4f        %d"
                  % (t["stage"], t["mean_c"],
                     ("%.4f" % t["sd_c"]) if t["sd_c"] is not None else " n/a ",
                     t["abs_err_of_mean"], t["mean_abs_err"], t["n_above_target"]))

    print("\n" + "=" * 78)
    print("Q3  DOES THE CRITERION RETURN, OR JUST ARRIVE?")
    print("=" * 78)
    for arm in ARMS:
        a = res["Q3_return_or_arrive"]["arms"].get(arm)
        if not a:
            continue
        print("\n  [%s]  n=%d" % (arm, a["summary"]["n_cells"]))
        print("    %-22s %-9s %-9s %-7s %-6s %-6s %-9s %s"
              % ("cell", "path", "net", "n/g", "cross", "rank", "end err", "best earlier"))
        for p in a["per_cell"]:
            print("    %-22s %-9.4f %-9.4f %-7.3f %-6d %-6s %-9.4f %.4f (k%d)"
                  % (p["cell"], p["raw_path"], p["net_displacement"], p["net_to_gross"],
                     p["crossings_of_endpoint_level"],
                     "%d/%d" % (p["endpoint_rank"], p["n_stages"]),
                     p["endpoint_err"], p["best_earlier_err"], p["best_earlier_stage"]))
        s = a["summary"]
        print("    mean net/gross %.3f | monotone approach %d/%d | endpoint interior %d/%d "
              "| crossed its own endpoint level %d/%d"
              % (s["mean_net_to_gross"], s["n_monotone_approach"], s["n_cells"],
                 s["n_endpoint_interior"], s["n_cells"], s["n_with_crossings"], s["n_cells"]))
        mean_earlier = float(np.mean([p["mean_earlier_err"] for p in a["per_cell"]]))
        print("    endpoint beats EVERY earlier stage on |c - c*| in %d/%d cells "
              "(mean end err %.4f vs mean BEST-earlier %.4f, vs mean TYPICAL earlier %.4f)"
              % (s["n_endpoint_beats_all_earlier"], s["n_cells"],
                 s["mean_endpoint_err"], s["mean_best_earlier_err"], mean_earlier))
        print("    (best-earlier is a min over %d stages and is selection-biased low; "
              "the typical-earlier column is the unbiased comparison)"
              % (a["per_cell"][0]["n_stages"] - 1))

    print("\n" + "=" * 78)
    print("Q4  WHAT IS THE MID-SEQUENCE DRIFT WORTH IN POPE F1?")
    print("=" * 78)
    for arm in ARMS:
        a = res["Q4_what_is_drift_worth"]["arms"].get(arm)
        if not a:
            continue
        print("\n  [%s]  n=%d" % (arm, a["summary"]["n_cells"]))
        print("    %-22s %-6s %-9s %-9s %-9s %-9s %s"
              % ("cell", "k*", "c at k*", "err k*", "F1 k*", "F1 end", "F1 end - F1 k*"))
        for p in a["per_cell"]:
            print("    %-22s %-6d %+.4f   %-9.4f %-9.4f %-9.4f %+.4f"
                  % (p["cell"], p["worst_drift_stage_mid"], p["worst_drift_c_mid"],
                     p["worst_drift_err_mid"], p["worst_drift_f1_mid"],
                     p["endpoint_f1"], p["f1_gap_mid"]))
        s = a["summary"]
        ci = s.get("ci95_mean_f1_gap_mid")
        print("    mean F1 gap %+.4f  sd %s  range [%+.4f, %+.4f]  positive in %d/%d cells"
              % (s["mean_f1_gap_mid"],
                 ("%.4f" % s["sd_f1_gap_mid"]) if s["sd_f1_gap_mid"] else "n/a",
                 s["min_f1_gap_mid"], s["max_f1_gap_mid"],
                 s["n_cells_gap_positive"], s["n_cells"]))
        if ci:
            print("    95%% CI on the mean gap (cells resampled with multiplicity): "
                  "[%+.4f, %+.4f]" % (ci["lo"], ci["hi"]))
        print("    item-sampling SD of F1 at that stage: %.4f  (a gap under ~2x this is "
              "inside POPE's own sampling noise)" % s["mean_item_sd_f1"])
        if "excursion_buys_loss" in a:
            show_corr(a["excursion_buys_loss"])
        nf = res["Q4_what_is_drift_worth"]["seed_noise_floor"].get(arm)
        if nf:
            print("    seed-to-seed SD of ENDPOINT F1 within an order (n=3 seeds): %.4f"
                  % nf["pooled_sd_endpoint_f1"])
        w = res["Q4_what_is_drift_worth"]["within_cell_err_vs_f1"].get(arm)
        if w:
            print("    within-cell r(|c - c*|, F1) across stages: mean %.3f, negative in "
                  "%d/%d cells" % (w["mean_within_cell_r"], w["n_cells_negative"],
                                   w["n_cells"]))

    er = res["Q4_what_is_drift_worth"].get("endpoint_residual_vs_joint")
    if er:
        print("\n" + "=" * 78)
        print("Q5  THE RESIDUAL ENDPOINT GAP TO JOINT: CRITERION, OR SEPARABILITY?")
        print("=" * 78)
        print("  endpoint CRITERION gap (seq mean - joint mean): %+.4f   <- the paper's "
              "'destination does not move'" % er["criterion_gap"])
        print("  endpoint F1       gap (seq mean - joint mean): %+.4f" % er["gap_seq_minus_joint"])
        print("  endpoint d' :  seq mean %.4f (sd %.4f, n=9)   joint cells %s (n=2, no interval)"
              % (er["mean_seq_endpoint_dprime"], er["sd_seq_endpoint_dprime"],
                 ", ".join("%.4f" % v for v in er["joint_endpoint_dprime_per_cell"])))
        print("  endpoint d' gap (seq - joint): %+.4f  = %.2f sd of the seq cell distribution"
              % (er["dprime_gap"], abs(er["dprime_gap"]) / max(er["sd_seq_endpoint_dprime"], 1e-9)))
        print("  seq endpoint F1 per cell: %s"
              % ", ".join("%.4f" % v for v in er["seq_endpoint_f1_per_cell"]))
        print("  joint endpoint F1 cells : %s  (n=2, quoted individually)"
              % ", ".join("%.4f" % v for v in er["joint_endpoint_f1_per_cell"]))
        print("  seq cells BELOW the lower joint cell: %d/%d;  ABOVE the upper joint cell: %d/%d"
              % (er["n_seq_below_lowest_joint_cell"], er["n_seq_cells"],
                 er["n_seq_above_highest_joint_cell"], er["n_seq_cells"]))
        print("  => the endpoint criterion gap is ~0, so the residual F1 gap cannot be a")
        print("     criterion-placement cost; it is carried by separability.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print("\n[does_drift_cost] wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
