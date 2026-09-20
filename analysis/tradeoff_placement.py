#!/usr/bin/env python3
"""Is "less criterion drift costs endpoint placement" a real trade-off, or three
arms that happen to line up?

WHY (2026-09-12). Three interventions appeared to show the same shape -- the anchor,
critp, and the projector freeze `fsL_locproj_o1_s17`, all shortening the criterion
path and landing the endpoint FURTHER from where the joint arm lands. The projector
freeze was the one that tempted a reframing, because it has no reference model: it
cannot be explained by "it inherited the frozen base's mis-placed c = +0.431". If the
pattern were structural, the paper's method section would stop being "our candidate
failed" and become "a family of interventions is defeated by a trade-off we can
characterise".

That is a large claim to rest on three points, so this script tries to break it.

THE THIRD POINT DOES NOT EXIST (corrected 2026-09-13). `locproj`'s motivating numbers
-- Sigma|dc| 0.2237 with endpoint c = +0.2739 -- were read off an arm that was STILL
IN EVAL, and a path over fewer transitions is mechanically smaller. They are withdrawn
in full: see FULLSTUDY_PREREG.md, the 2026-09-12 CORRECTION entry and the 2026-09-13
SECOND CORRECTION entry. At 6/6 the arm has MORE path than the sequential baseline
(0.9895 vs 0.7643), and against this project's adopted reference c* = +0.0878 its
placement is WORSE (0.0837 vs 0.0199) -- it is not on the trade-off frontier, it is
behind the control on both axes. What remains of the motivating pattern is the anchor
family, which is ONE mechanism; that is the conclusion this script reaches below by
independent means. The corrected values are carried in TEXT_RECORDED so the machine
output CONTRADICTS the premise rather than merely omitting it.

WHAT WOULD FALSIFY THE TRADE-OFF (stated before the numbers, and each one is
actually computed below):

  F1. SIGN. A trade-off means corr(drift, placement error) < 0: less path, worse
      placement. A correlation that is zero or POSITIVE refutes it.
  F2. THE ANCHOR FAMILY. anchor / critp / anchorlo / anchorOOD share one mechanism
      (regularize toward the frozen base). If the relationship exists only while
      they are in the sample, it is one family's signature, not a trade-off.
      Recomputed with them dropped; if it dies, that IS the result.
  F3. ARM IDENTITY. Pooling cells from arms that differ in both quantities makes a
      two-group separation look like a correlation with n = 20. The relationship is
      recomputed with arm means removed (arm identity partialled out). If it only
      lives between arms and not within them, the honest n is the number of ARMS.
  F4. A COUNTEREXAMPLE. One arm with BOTH low drift and good placement kills
      "structural". The joint arm is a candidate and is checked explicitly.
  F5. PAIRWISE DIRECTION. With n this small a correlation is a summary of a handful
      of pairwise comparisons. All arm pairs are enumerated and each is labelled
      trade-off-consistent or not. Chance is half.

ESTIMATOR RULES ENFORCED HERE (this project has broken both before):

  * Never pair a path length with a range, nor a per-cell quantity with a cross-cell
    average. Every quantity is computed PER CELL and only then averaged.
  * The RAW path (base -> k1 -> ... -> k6) and the POST-SETTLING path (first step
    dropped; the pre-registered endpoint) are different estimators and are never
    mixed in one comparison. An arm whose per-stage criteria we do not hold cannot
    appear on the post-settling axis at all, and is dropped from it rather than
    approximated. The motivating table that prompted this script mixes them --
    seq 0.7643 is a RAW single cell, anchor 0.364 is a POST-SETTLING 9-cell mean --
    and `--check-premise` prints that mismatch with both estimators side by side.
  * Bootstrap resamples cells as a LIST with multiplicity. A set() around a
    resample silently makes it a 63.2% subsample and made nine intervals in this
    project ~24% too narrow.
  * Single-cell arms are labelled SINGLE-CELL everywhere and never receive an
    interval that implies replication.

REFERENCE POINT. Placement error is |c_endpoint - c_joint|, where c_joint is the
JOINT arm's EMPIRICAL endpoint criterion (+0.088, the mean of its cells), not a
modelled 0. Balanced accuracy peaks at c = 0 only under equal variance and our
z-ROC slopes are 0.55-0.72, so 0 is not the optimum we can defend; the joint arm's
endpoint is an optimum measured from data. Note the circularity this creates -- the
joint arm is both the yardstick and a data point, so its own error is ~0 by
construction -- and every test is therefore also run with joint dropped.

Usage:
  python3 tradeoff_placement.py [--agg <fs_aggregate.json>] [--results <results_fs>]
                                [--out readout/tradeoff_placement.json]
                                [--check-premise] [--B 10000] [--seed 17]
"""
import argparse
import glob
import json
import math
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fs_common

# Arms that regularize the decision statistic toward a frozen reference model.
# They share a mechanism, so they are one point of evidence, not four.
ANCHOR_FAMILY = {"anchor", "critp", "anchorcrit", "anchorlo", "anchorOOD",
                 "anchorood", "anchorgca"}

# Arms whose numbers exist only as text in the repo, because the cells were scored
# on the cluster and the generations were never pulled into the local mirror.
# Kept ONLY as a fallback: if the cell is reachable under --results it is rescored
# with fs_common.score_pope and this entry is ignored. Each carries its provenance
# and the estimator it was computed with, because a text-recorded number whose
# estimator is unknown is worse than no number.
TEXT_RECORDED = {
    # CORRECTED 2026-09-13. The values here were 0.2237 / +0.2739 / 2.2955 until
    # today. Those came from a read taken while the arm was STILL IN EVAL, were
    # withdrawn in full by the 2026-09-12 CORRECTION entry, and their `n_stages: 6`
    # label was never true of that read -- the whole point is that the path ran over
    # fewer transitions than six, which is why it looked like a 71% drift reduction.
    # Replaced (not deleted) with the 6/6 numbers, so this file states the correction
    # instead of going quiet about it.
    "fsL_locproj_o1_s17": {
        "arm": "locproj",
        "order": "o1", "seed": 17, "n_stages": 6,
        "n_stages_verified": False,  # 6/6 is asserted in prose, by a human
        "raw_path": 0.9895,          # Sigma|dc| from base through k6, at 6/6
        "post_settle": None,         # STILL not recoverable: no per-stage criteria
        "endpoint_c": 0.0041,
        "endpoint_dprime": 2.2298,
        "retracted_values": {"raw_path": 0.2237, "endpoint_c": 0.2739,
                             "endpoint_dprime": 2.2955,
                             "why": "read while the arm was still in eval; path over "
                                    "fewer than six transitions, which is "
                                    "mechanically smaller",
                             "withdrawn": "2026-09-12 CORRECTION entry"},
        "source": "fullstudy/FULLSTUDY_PREREG.md, 2026-09-12 CORRECTION entry "
                  "(6/6 table), re-scored against c* = +0.0878 in the 2026-09-13 "
                  "SECOND CORRECTION entry and analysis/LOCALIZATION_DISSOCIATION.md "
                  "section 2; produced by analysis/mechanism_readout.py --mode loc "
                  "on the cluster",
        "note": "SINGLE CELL, and NOT a finding -- the localization dissociation was "
                "WITHDRAWN on 2026-09-13 (this arm is worse than the baseline on "
                "BOTH axes and sits inside the baseline's own nine-cell envelope). "
                "Per-stage criteria were never recorded, so it can still only be "
                "placed on the RAW axis. The 6/6 stage count is asserted in prose "
                "and is NOT recorded by any artifact in this repo: the arm has no "
                "cell under results_fs and fs_aggregate.json (2026-09-09) predates "
                "it. To verify the stage count AND reach the pre-registered "
                "post-settling axis, pull fsL_locproj_o1_s17_k*/pope_gen.jsonl into "
                "results_fs and rerun -- mechanism_readout.py now refuses any cell "
                "without EVAL_DONE or with pope n_total != 9000, and dumps the "
                "per-stage criterion vector it used to discard.",
    },
}


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def _ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            r[order[t]] = avg
        i = j + 1
    return r


def spearman(xs, ys):
    return pearson(_ranks(xs), _ranks(ys))


def boot_pearson(xs, ys, B=10000, seed=17):
    """Percentile CI for Pearson r by resampling the UNITS with replacement.

    The resample is built as a LIST of indices and indexed positionally, so a unit
    drawn twice contributes twice. Anything that deduplicates here (a set, a dict
    keyed by cell name) turns B resamples of n units into resamples of ~0.632n
    DISTINCT units and narrows every interval; that bug has already cost this
    project nine intervals.

    Degenerate resamples (all-identical x or y, so r is undefined) are counted and
    skipped rather than coerced to 0, and the count is reported: with n this small
    they are not rare and a reader should see how many.

    Divergence regime, named: this interval treats CELLS as the sampling unit, so
    it covers cell-to-cell noise WITHIN the arms we ran. It does NOT cover
    arm-to-arm uncertainty -- the question "would a fifth intervention land on this
    line" -- which is the dominant uncertainty here and is not estimable from 3-4
    arms. Read it as a lower bound on the true width.
    """
    n = len(xs)
    if n < 3:
        return None
    rng = random.Random(seed)
    rs = []
    degenerate = 0
    idx = range(n)
    for _ in range(B):
        s = rng.choices(idx, k=n)              # LIST, with multiplicity
        r = pearson([xs[i] for i in s], [ys[i] for i in s])
        if r is None:
            degenerate += 1
            continue
        rs.append(r)
    if len(rs) < 100:
        return {"ci": None, "B": B, "n": n, "degenerate": degenerate,
                "note": "too few non-degenerate resamples for an interval"}
    rs.sort()
    m = len(rs)
    lo = rs[int(0.025 * m)]
    hi = rs[min(m - 1, int(0.975 * m))]
    return {"ci": [round(lo, 4), round(hi, 4)], "B": B, "n": n,
            "n_effective": m, "degenerate": degenerate, "seed": seed,
            "excludes_zero": bool(lo > 0 or hi < 0)}


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------
def cell_record(arm, cell_id, cs, base_c, n_stages_expected=None, source="aggregate",
                single_cell=False):
    """One cell -> its two path estimators and its endpoint. Per cell, always."""
    sw = fs_common.swings([base_c] + list(cs))
    return {
        "arm": arm,
        "cell": cell_id,
        "n_stages": len(cs),
        "raw_path": round(sum(sw), 4),
        "post_settle": (None if fs_common.post_settle_sum(sw) is None
                        else round(fs_common.post_settle_sum(sw), 4)),
        "endpoint_c": round(cs[-1], 4),
        "source": source,
        "single_cell": single_cell,
    }


def cells_from_aggregate(agg_path, backbone="llava15"):
    d = json.load(open(agg_path))
    bb = d["backbones"][backbone]
    base_c = bb["base"]["pope"]["c"]
    rows = []
    for key in sorted(bb["arms"]):
        cell = bb["arms"][key]
        ks = sorted((s for s in cell if s.isdigit()), key=int)
        cs = []
        for s in ks:
            p = cell[s].get("pope")
            if not p:
                break
            cs.append(p["c"])
        if len(cs) < 2:
            continue
        rows.append(cell_record(key.split("|")[0], key, cs, base_c))
    return base_c, rows


def cells_from_results(results, base_c, known_runtags, backbone="llava15",
                       suite="ucit"):
    """Rescore any results_fs cells the aggregate does not already cover.

    Scored with fs_common.score_pope -- the same audited scorer behind every other
    number in the paper -- so a locally mirrored arm is numerically identical to an
    aggregate one. Restricted to the SAME backbone and the SAME suite: a pilot
    (mth_*/psL_*) cell runs 4 stages over different tasks, so its path length is a
    different quantity and pooling it would be exactly the estimator error this
    script exists to avoid.
    """
    if not results or not os.path.isdir(results):
        return [], {"scanned": 0, "note": "results dir absent"}
    stages = {}
    scanned = 0
    excluded = {}
    for d in sorted(glob.glob(os.path.join(results, "*"))):
        if not os.path.isdir(d):
            continue
        parsed = fs_common.parse_cell_name(os.path.basename(d))
        if not parsed or parsed[1] != "k":
            continue
        runtag, _, k = parsed
        meta = fs_common.parse_runtag(runtag)
        # Excluded runtags are RECORDED, never dropped silently. A silent
        # exclusion is how this project has lost data before; a reader needs to
        # see that e.g. a pilot-suite mth_* arm was found and deliberately left
        # out because its 4-stage path is a different quantity from a 6-stage one.
        if not meta:
            excluded.setdefault("unparseable runtag", set()).add(runtag)
            continue
        if meta["backbone"] != backbone:
            excluded.setdefault("backbone != %s" % backbone, set()).add(runtag)
            continue
        if meta["suite"] != suite:
            excluded.setdefault("suite != %s (path length not comparable)" % suite,
                                set()).add(runtag)
            continue
        if runtag in known_runtags:
            continue
        p = os.path.join(d, "pope_gen.jsonl")
        if not os.path.isfile(p):
            continue
        try:
            s = fs_common.score_pope(p)
        except Exception as e:                     # report, never crash the sweep
            print("  [warn] %s: %s: %s" % (d, type(e).__name__, e))
            continue
        scanned += 1
        stages.setdefault((runtag, meta["arm"]), {})[k] = s["c"]
    rows = []
    for (runtag, arm), by_k in sorted(stages.items()):
        ks = fs_common.contiguous_prefix(by_k)     # a gap makes later swings span 2 stages
        if len(ks) < 2:
            print("  [skip] %s: %d contiguous stage(s), need >= 2" % (runtag, len(ks)))
            continue
        rows.append(cell_record(arm, runtag, [by_k[k] for k in ks], base_c,
                                source="results_fs rescored", single_cell=True))
    for why, tags in sorted(excluded.items()):
        print("  [excluded] %s: %s" % (why, ", ".join(sorted(tags))))
    return rows, {"scanned": scanned, "runtags": sorted(set(r for r, _ in stages)),
                  "excluded": {k: sorted(v) for k, v in excluded.items()}}


def apply_text_recorded(rows, ref):
    """Add arms recorded only as text, unless the same cell was already rescored."""
    have = {r["cell"] for r in rows}
    added = []
    for runtag, t in sorted(TEXT_RECORDED.items()):
        if runtag in have:
            print("  [info] %s reachable under --results; the text-recorded value "
                  "is ignored in favour of the rescored one" % runtag)
            continue
        r = {"arm": t["arm"], "cell": runtag, "n_stages": t["n_stages"],
             "n_stages_verified": t.get("n_stages_verified", True),
             "raw_path": t["raw_path"], "post_settle": t["post_settle"],
             "endpoint_c": t["endpoint_c"], "source": "TEXT-RECORDED",
             "single_cell": True, "provenance": t["source"], "note": t["note"]}
        if t.get("retracted_values"):
            r["supersedes_retracted"] = t["retracted_values"]
        rows.append(r)
        added.append(runtag)
    return added


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def arm_means(rows, axis):
    """Arm-level point = MEAN OF PER-CELL quantities (never a path computed from
    averaged criteria). Arms with no cell on this axis are absent, not zero."""
    by = {}
    for r in rows:
        if r.get(axis) is None:
            continue
        by.setdefault(r["arm"], []).append(r)
    out = []
    for arm in sorted(by):
        sub = by[arm]
        out.append({"arm": arm, "n_cells": len(sub),
                    "single_cell": len(sub) == 1,
                    "drift": round(st.mean(r[axis] for r in sub), 4),
                    "err": round(st.mean(r["err"] for r in sub), 4),
                    "endpoint_c": round(st.mean(r["endpoint_c"] for r in sub), 4)})
    return out


def corr_block(rows, axis, label, B, seed, bootstrap=True):
    sub = [r for r in rows if r.get(axis) is not None]
    xs = [r[axis] for r in sub]
    ys = [r["err"] for r in sub]
    p = pearson(xs, ys)
    s = spearman(xs, ys)
    out = {"label": label, "axis": axis, "n": len(sub),
           "cells": [r["cell"] for r in sub],
           "pearson": None if p is None else round(p, 4),
           "spearman": None if s is None else round(s, 4)}
    if bootstrap and len(sub) >= 4:
        out["boot"] = boot_pearson(xs, ys, B=B, seed=seed)
    return out


def fmt_corr(b):
    r = b["pearson"]
    if r is None:
        return "%-38s n=%2d   r = n/a (too few points)" % (b["label"], b["n"])
    ci = ""
    boot = b.get("boot")
    if boot and boot.get("ci"):
        ci = "  95%% CI [%+.3f, %+.3f]" % tuple(boot["ci"])
    elif b["n"] < 4:
        ci = "  (no interval: n < 4)"
    return "%-38s n=%2d   r = %+.4f   rho = %+.4f%s" % (
        b["label"], b["n"], r, b["spearman"], ci)


def pairwise_directions(pts):
    """Every arm pair labelled by whether it goes the trade-off way.

    Trade-off-consistent = the arm with the SHORTER drift path has the WORSE
    placement. With a handful of arms this is what a correlation is actually
    summarising, so it is worth seeing directly. Under a null of random signs the
    expected share is one half.
    """
    out = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            a, b = pts[i], pts[j]
            lo, hi = (a, b) if a["drift"] <= b["drift"] else (b, a)
            consistent = lo["err"] > hi["err"]
            out.append({"less_drift": lo["arm"], "more_drift": hi["arm"],
                        "d_drift": round(hi["drift"] - lo["drift"], 4),
                        "d_err": round(lo["err"] - hi["err"], 4),
                        "tradeoff_consistent": bool(consistent)})
    return out


def within_arm_centred(rows, axis, B, seed):
    """Arm identity partialled out: subtract each arm's mean from both variables.

    This is the F3 test. If the relationship survives centring, cells within an arm
    that drift less really do land worse, and the trade-off is continuous. If it
    vanishes or flips, the pooled correlation is a between-arm group separation
    wearing an n = 20 costume. Arms with one cell contribute nothing (their centred
    values are exactly 0, 0) and are excluded rather than silently pinned to the
    origin, which would drag r toward the pooled value.
    """
    by = {}
    for r in rows:
        if r.get(axis) is None:
            continue
        by.setdefault(r["arm"], []).append(r)
    xs, ys, used = [], [], []
    for arm, sub in sorted(by.items()):
        if len(sub) < 2:
            continue
        mx = st.mean(r[axis] for r in sub)
        my = st.mean(r["err"] for r in sub)
        for r in sub:
            xs.append(r[axis] - mx)
            ys.append(r["err"] - my)
        used.append("%s(%d)" % (arm, len(sub)))
    p = pearson(xs, ys)
    out = {"label": "within-arm centred", "axis": axis, "n": len(xs),
           "arms_used": used,
           "pearson": None if p is None else round(p, 4),
           "spearman": None if not xs else round(spearman(xs, ys), 4)}
    if len(xs) >= 4:
        out["boot"] = boot_pearson(xs, ys, B=B, seed=seed)
    return out


def check_premise(rows):
    """The motivating table, with both estimators shown, because it mixes them."""
    print("\n" + "=" * 78)
    print("PREMISE CHECK -- the motivating table pairs two different estimators")
    print("=" * 78)
    print("""
The table that prompted this analysis read

    sequential  0.7643     anchor  ~0.364     locproj  0.2237

and it has TWO independent defects.

ESTIMATOR MISMATCH. 0.7643 is the RAW path of ONE sequential cell (o1/s17) and 0.364
is the anchor's POST-SETTLING mean over 9 cells. The post-settling estimator drops the
base -> k1 step, which is the largest single step in most cells, so the two columns
are not the same quantity and the apparent ordering is partly an artifact of which
one each arm was quoted with.

A NUMBER THAT WAS NOT ABOUT A FINISHED ARM. locproj's 0.2237 was read while the arm
was still in eval. Its path ran over fewer than six transitions, and a shorter path is
mechanically smaller, so the truncated run impersonated a 71% drift reduction. The 6/6
value is 0.9895: locproj drifts MORE than the sequential cell it was quoted against.
0.2237 is RETRACTED (2026-09-12) and appears above only as the withdrawn number it is;
the row below carries the corrected value.
""")
    print("%-10s %7s %22s %22s" % ("arm", "cells", "raw Sigma|dc|", "post-settling Sigma|dc|"))
    print("-" * 64)
    for arm in ("seq", "anchor", "joint", "locproj"):
        sub = [r for r in rows if r["arm"] == arm]
        if not sub:
            continue
        raw = [r["raw_path"] for r in sub if r["raw_path"] is not None]
        ps = [r["post_settle"] for r in sub if r["post_settle"] is not None]
        f = lambda v: ("%.4f [%.4f-%.4f]" % (st.mean(v), min(v), max(v))
                       if len(v) > 1 else ("%.4f (single cell)" % v[0] if v else "not recoverable"))
        print("%-10s %7d %22s %22s" % (arm, len(sub), f(raw), f(ps)))
    seq_raw = [r["raw_path"] for r in rows if r["arm"] == "seq"]
    anc_raw = [r["raw_path"] for r in rows if r["arm"] == "anchor"]
    if seq_raw and anc_raw:
        print("\n  On the RAW axis the anchor's advantage over sequential is small and the")
        print("  per-cell ranges overlap heavily (seq %.4f-%.4f vs anchor %.4f-%.4f)."
              % (min(seq_raw), max(seq_raw), min(anc_raw), max(anc_raw)))
        print("  The anchor's drift reduction is largely a POST-SETTLING phenomenon, and")
        print("  locproj has no post-settling number, so the two cannot be compared on")
        print("  the pre-registered axis with what is currently on disk.")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default=os.path.join(here, "readout", "fs_aggregate.json"))
    ap.add_argument("--results", default=None,
                    help="results_fs dir to rescore uncovered cells from "
                         "[default: auto-detected layout]")
    ap.add_argument("--backbone", default="llava15")
    ap.add_argument("--ref", type=float, default=None,
                    help="endpoint reference criterion [default: the JOINT arm's "
                         "empirical mean endpoint c]")
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--check-premise", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.path.isfile(args.agg):
        raise SystemExit("no aggregate at %s" % args.agg)
    if args.results is None:
        lay = fs_common.find_layout()
        args.results = str(lay["results"]) if lay.get("results") else None

    base_c, rows = cells_from_aggregate(args.agg, args.backbone)
    print("untuned base criterion c = %+.4f   (path origin for every cell)\n" % base_c)
    known = set()
    for r in rows:
        arm, o, s = r["cell"].split("|")
        known.add("fsL_%s_%s_%s" % (arm, o, s))
    extra, scan = cells_from_results(args.results, base_c, known, args.backbone)
    print("results_fs scan: %s" % args.results)
    print("  cells rescored with fs_common.score_pope beyond the aggregate: %d"
          % len(extra))
    if extra:
        for r in extra:
            print("    %s (%d stages)" % (r["cell"], r["n_stages"]))
    rows += extra
    text_added = apply_text_recorded(rows, None)

    # ---- reference point -------------------------------------------------
    joint = [r["endpoint_c"] for r in rows if r["arm"] == "joint"]
    ref = args.ref if args.ref is not None else (st.mean(joint) if joint else 0.0)
    print("\nendpoint reference c_ref = %+.5f  (%s)"
          % (ref, "--ref override" if args.ref is not None else
             "JOINT arm empirical endpoint, mean of %d cells" % len(joint)))
    print("  NOT a modelled 0: balanced accuracy peaks at c=0 only under equal")
    print("  variance, and our z-ROC slopes are 0.55-0.72.")
    if joint:
        print("  CIRCULARITY: the joint arm is both the yardstick and a data point, so")
        print("  its placement error is ~0 by construction. Every test below is also")
        print("  run with joint dropped.")
    for r in rows:
        r["err"] = round(abs(r["endpoint_c"] - ref), 4)

    if text_added:
        print("\nTEXT-RECORDED arms folded in (not rescorable from local data):")
        for t in text_added:
            e = TEXT_RECORDED[t]
            print("  %s  -- %s" % (t, e["source"]))
            print("     %s" % e["note"])

    if args.check_premise:
        check_premise(rows)

    # ---- per-cell table --------------------------------------------------
    print("\n" + "=" * 78)
    print("PER-CELL POINTS  (every quantity computed per cell, then aggregated)")
    print("=" * 78)
    print("%-10s %-22s %6s %10s %13s %11s %9s" %
          ("arm", "cell", "stages", "raw path", "post-settle", "endpoint c", "|err|"))
    print("-" * 86)
    for r in sorted(rows, key=lambda r: (r["arm"], r["cell"])):
        tag = "  [SINGLE-CELL]" if r.get("single_cell") else ""
        print("%-10s %-22s %6d %10.4f %13s %+11.4f %9.4f%s" %
              (r["arm"], r["cell"], r["n_stages"], r["raw_path"],
               ("%.4f" % r["post_settle"]) if r["post_settle"] is not None else "n/a",
               r["endpoint_c"], r["err"], tag))

    report = {"base_c": base_c, "ref": round(ref, 5),
              "ref_source": ("--ref override" if args.ref is not None
                             else "joint arm empirical endpoint (n=%d cells)" % len(joint)),
              "agg": args.agg, "results": args.results,
              "n_cells": len(rows), "cells": rows,
              "text_recorded": text_added, "results_scan": scan,
              "anchor_family": sorted(ANCHOR_FAMILY), "axes": {}}

    # ---- the tests, per axis --------------------------------------------
    for axis, axis_name in (("raw_path", "RAW Sigma|dc| (base -> k1 -> ... -> kN)"),
                            ("post_settle", "POST-SETTLING Sigma|dc| (pre-registered endpoint)")):
        avail = [r for r in rows if r.get(axis) is not None]
        dropped = sorted(r["cell"] for r in rows if r.get(axis) is None)
        print("\n" + "=" * 78)
        print("AXIS: %s" % axis_name)
        print("=" * 78)
        if dropped:
            print("  cells with no value on this axis (DROPPED, not approximated): %s"
                  % ", ".join(dropped))
        if len(avail) < 3:
            print("  too few cells on this axis to test.")
            continue

        ax = {"axis": axis, "name": axis_name, "dropped_cells": dropped}

        print("\n-- arm level (mean of per-cell values) --")
        pts = arm_means(avail, axis)
        print("%-10s %7s %12s %12s %12s" % ("arm", "cells", "drift", "endpoint c", "|err|"))
        print("-" * 56)
        for p in pts:
            print("%-10s %7d %12.4f %+12.4f %12.4f%s" %
                  (p["arm"], p["n_cells"], p["drift"], p["endpoint_c"], p["err"],
                   "   [SINGLE-CELL: no interval]" if p["single_cell"] else ""))
        ax["arm_points"] = pts

        xs = [p["drift"] for p in pts]
        ys = [p["err"] for p in pts]
        r_arm = pearson(xs, ys)
        ax["arm_level"] = {"n_arms": len(pts),
                           "pearson": None if r_arm is None else round(r_arm, 4),
                           "note": "no interval: n = %d arms, and arm-to-arm "
                                   "variability is not estimable from them" % len(pts)}
        print("\n  arm-level Pearson over n = %d ARMS: %s" %
              (len(pts), "%+.4f" % r_arm if r_arm is not None else "n/a (n<3)"))
        print("  (no interval. The honest n for 'is this a trade-off' is the number")
        print("   of arms, and 3-4 arms cannot support one.)")

        print("\n-- pairwise arm directions (chance = half) --")
        pw = pairwise_directions(pts)
        for d in pw:
            print("   %-9s vs %-9s : less drift by %.4f, placement %s by %.4f   %s"
                  % (d["less_drift"], d["more_drift"], d["d_drift"],
                     "WORSE" if d["d_err"] > 0 else "BETTER", abs(d["d_err"]),
                     "trade-off" if d["tradeoff_consistent"] else "ANTI-trade-off"))
        k = sum(1 for d in pw if d["tradeoff_consistent"])
        print("   -> %d of %d arm pairs go the trade-off way." % (k, len(pw)))
        ax["pairwise"] = {"pairs": pw, "consistent": k, "total": len(pw)}

        print("\n-- cell level (n is cells, NOT independent arms) --")
        blocks = [
            corr_block(avail, axis, "all cells", args.B, args.seed),
            corr_block([r for r in avail if r["arm"] not in ANCHOR_FAMILY], axis,
                       "F2: anchor family DROPPED", args.B, args.seed),
            corr_block([r for r in avail if r["arm"] != "joint"], axis,
                       "F4/circularity: joint DROPPED", args.B, args.seed),
            corr_block([r for r in avail if r["arm"] not in ANCHOR_FAMILY
                        and r["arm"] != "joint"], axis,
                       "anchor family AND joint dropped", args.B, args.seed),
        ]
        for b in blocks:
            print("   " + fmt_corr(b))
        wc = within_arm_centred(avail, axis, args.B, args.seed)
        ci = ""
        if wc.get("boot") and wc["boot"].get("ci"):
            ci = "  95%% CI [%+.3f, %+.3f]" % tuple(wc["boot"]["ci"])
        print("   %-38s n=%2d   r = %s%s"
              % ("F3: " + wc["label"], wc["n"],
                 "%+.4f" % wc["pearson"] if wc["pearson"] is not None else "n/a", ci))
        print("        arms contributing (>=2 cells): %s" % ", ".join(wc["arms_used"]))
        ax["cell_level"] = blocks
        ax["within_arm_centred"] = wc

        print("\n   Sign convention: a trade-off predicts r < 0 (less drift, worse")
        print("   placement). r > 0 means MORE drift went with worse placement, which")
        print("   is the opposite of the hypothesis and is what a random walk does.")

        # ---- F4: counterexample by PARETO DOMINANCE ----------------------
        # A trade-off says the two objectives cannot both improve. So the
        # counterexample is not "below average on both" (which depends on the
        # arbitrary mix of arms in the sample and can flag an arm that is in fact
        # worse than every non-anchor arm). It is DOMINANCE: arm A dominates arm B
        # when A has both the shorter path and the better placement. One such pair
        # is an existence proof that the two objectives are jointly improvable.
        best_drift = min(pts, key=lambda p: p["drift"])
        best_err = min(pts, key=lambda p: p["err"])
        dom = [d for d in pw if not d["tradeoff_consistent"]]
        print("\n-- F4: is there an arm that DOMINATES another (less drift AND better")
        print("       placement)? That is what refutes 'you cannot have both'. --")
        print("   lowest drift:     %-9s (drift %.4f, err %.4f)"
              % (best_drift["arm"], best_drift["drift"], best_drift["err"]))
        print("   best placement:   %-9s (drift %.4f, err %.4f)"
              % (best_err["arm"], best_err["drift"], best_err["err"]))
        if dom:
            for d in dom:
                print("   DOMINATES: %-9s over %-9s  (drift -%.4f, err -%.4f)"
                      % (d["less_drift"], d["more_drift"], d["d_drift"], -d["d_err"]))
            print("   An arm that buys drift reduction without paying placement refutes")
            print("   'structural'. What remains is a claim about particular mechanisms.")
        else:
            print("   none: every arm pair trades one objective off against the other.")
            print("   Consistent with the trade-off, though with %d arms that is weak"
                  % len(pts))
            print("   evidence -- %d pairs is not a frontier." % len(pw))
        ax["dominating_pairs"] = dom
        ax["counterexample"] = sorted({d["less_drift"] for d in dom}) or None

        # ---- sensitivity: leave-one-arm-out on the pooled correlation ----
        # F2 generalized. The anchor family is the confound we suspected, but the
        # honest display is what the pooled r does when ANY single arm is removed:
        # if one arm's removal moves it across zero, the pooled number is that
        # arm's signature, whichever arm it turns out to be.
        print("\n-- sensitivity: pooled cell-level r with ONE ARM removed --")
        loo = []
        for p in pts:
            sub = [r for r in avail if r["arm"] != p["arm"]]
            rr = pearson([r[axis] for r in sub], [r["err"] for r in sub])
            loo.append({"dropped": p["arm"], "n_cells": len(sub),
                        "pearson": None if rr is None else round(rr, 4)})
            print("   drop %-9s -> n=%2d cells, r = %s"
                  % (p["arm"], len(sub), "%+.4f" % rr if rr is not None else "n/a"))
        ax["leave_one_arm_out"] = loo
        report["axes"][axis] = ax

    # ---- verdict ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT (mechanical, from the numbers above)")
    print("=" * 78)
    verdict = []
    for axis in ("raw_path", "post_settle"):
        ax = report["axes"].get(axis)
        if not ax:
            continue
        allb = next(b for b in ax["cell_level"] if b["label"] == "all cells")
        noanc = next(b for b in ax["cell_level"] if b["label"].startswith("F2"))
        wc = ax["within_arm_centred"]
        lines = []
        lines.append("%s:" % ax["name"])
        lines.append("  pooled over %d cells         r = %s"
                     % (allb["n"], "%+.4f" % allb["pearson"] if allb["pearson"] is not None else "n/a"))
        lines.append("  anchor family dropped (%d)   r = %s"
                     % (noanc["n"], "%+.4f" % noanc["pearson"] if noanc["pearson"] is not None else "n/a"))
        lines.append("  arm identity partialled out  r = %s"
                     % ("%+.4f" % wc["pearson"] if wc["pearson"] is not None else "n/a"))
        lines.append("  arm level (n = %d arms)       r = %s"
                     % (ax["arm_level"]["n_arms"],
                        "%+.4f" % ax["arm_level"]["pearson"]
                        if ax["arm_level"]["pearson"] is not None else "n/a"))
        lines.append("  arm pairs going the trade-off way: %d of %d"
                     % (ax["pairwise"]["consistent"], ax["pairwise"]["total"]))
        # NEGLIGIBLE: below this, r is reported as "no relationship" rather than
        # given a direction. With n this small a |r| of 0.1 is indistinguishable
        # from zero and calling its sign a surviving "direction" would be reading
        # noise as a finding.
        NEGLIGIBLE = 0.10
        if allb["pearson"] is not None and noanc["pearson"] is not None:
            if allb["pearson"] >= -NEGLIGIBLE:
                lines.append("  => no negative relationship even pooled (|r| negligible "
                             "or wrong sign).")
            elif abs(noanc["pearson"]) < NEGLIGIBLE:
                lines.append("  => dropping the anchor family leaves NO relationship at all")
                lines.append("     (r ~ 0). The pooled number is the anchor family's.")
            elif noanc["pearson"] > 0:
                lines.append("  => the relationship REVERSES SIGN when the anchor family is")
                lines.append("     dropped. On this axis the trade-off is the anchor family.")
            else:
                lines.append("  => direction survives dropping the anchor family.")
        if wc["pearson"] is not None and wc["pearson"] >= 0:
            lines.append("  => with arm identity removed there is NO within-arm trade-off;")
            lines.append("     the pooled correlation is a between-arm group separation.")
        for d in ax.get("dominating_pairs") or []:
            lines.append("  => counterexample: %s has BOTH less drift and better placement"
                         % d["less_drift"])
            lines.append("     than %s (drift -%.4f, err -%.4f), so the two objectives"
                         % (d["more_drift"], d["d_drift"], -d["d_err"]))
            lines.append("     are jointly improvable here.")
        verdict += lines + [""]
    for line in verdict:
        print(line)
    report["verdict_lines"] = verdict

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(report, open(args.out, "w"), indent=2)
        print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
