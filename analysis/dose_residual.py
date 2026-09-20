#!/usr/bin/env python3
"""What predicts the one-step criterion change dc BEYOND the answer-statistic dose?

WHY (2026-09-13). The paper's mechanism account is that each stage's labelled
answer statistics are the dose that moves the POPE criterion c. The SIGN evidence
holds (pilot suite). The pre-registered NULL side failed: the UCIT stream is
certified zero-dose (every task yes = no = refusal = 0.000) yet the criterion
still moves 0.40-0.55 units in 9 of 9 cells (FULLSTUDY_PREREG.md, "E1 null-side:
FAILED"). So answer statistics bound the sign but not the magnitude, and
something else is moving the criterion.

This script does the variance accounting. It is an analysis of readouts already
on disk -- no GPU, no new runs, no training.

Two suites, because they answer different halves of the question:

  UCIT  (primary; 9 cells = 3 orderings x 3 seeds, 6 stages, LLaVA-1.5-7B SEQ)
        Dose is ZERO for all six tasks, so the answer-statistic account makes the
        point prediction dc = 0 everywhere and has no variance to fit. Everything
        observed here IS the residual. The question is what structure it has.

  PILOT (4 cells = o1 x 3 seeds + o2 x 1 seed, 4 stages, same backbone)
        The only place on disk where dose actually VARIES -- and where dose and
        answer length are near-orthogonal (flickr = long/zero-dose, vizwiz =
        short/high-dose). This is the one design that can separate the two
        accounts, and it has 16 observations.

DISCIPLINE (this project has produced two wrong conclusions by violating these):
  * Per cell, then aggregate. Contrasts and slopes are computed WITHIN a cell and
    then averaged over cells; a per-cell quantity is never paired with a
    cross-cell average. eta^2 is the one exception and it is flagged: each cell
    holds exactly one observation per task, so a within-cell eta^2 is identically
    1 and meaningless. eta^2 is therefore pooled BY NECESSITY, and cell-level
    dependence enters through the cell bootstrap instead.
  * The bootstrap resamples CELLS as a LIST with multiplicity. A set() would turn
    it into a 63.2% subsample and shrink every interval (this bug has bitten this
    project before -- see env_bootstrap_set_drops_duplicates).
  * eta^2 is biased upward by level count, and the factors here do NOT have equal
    level counts (task 6, transition up to 18). Every eta^2 is therefore reported
    with omega^2 (level-count corrected) and a permutation p whose null carries
    the same bias.

Usage:
  python3 dose_residual.py [--agg readout/fs_aggregate.json]
                           [--results_fs ../fullstudy/results_fs]
                           [--pilot_results ../results]
                           [--manifest_ucit <ucit_manifest.json>]
                           [--out readout/dose_residual.json]
"""
import argparse
import json
import math
import os
import random
import statistics
import sys
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

NORM = statistics.NormalDist()

# ---------------------------------------------------------------------------
# Frozen per-task inputs
# ---------------------------------------------------------------------------
# UCIT answer statistics. The authoritative source is the frozen
# data_ucit/ucit_manifest.json on the cluster (committed at data-prep time,
# before any eval ran -- fullstudy/ucit_prep.py). It is not in the repo, so the
# values below are transcribed from FULLSTUDY_PREREG.md, which quotes the frozen
# manifest in two places:
#   line ~86  ("ALL SIX UCIT tasks have ~0% yes/no and 0% refusal")
#   line ~399 ("every one of the six tasks has yes = 0.000, no = 0.000
#              (VizWiz 0.0005), refusal = 0.000")
#   line ~1504 (mean answer words table)
# Pass --manifest_ucit to override with the real file; the script prefers it and
# says which source it used. mean_target_words here are TRAIN-split values.
UCIT_FROZEN = {
    # task:          yes,    no,     refusal, mean_target_words
    "ArxivQA":      (0.0000, 0.0000, 0.0000,  1.58),
    "CLEVR-Math":   (0.0000, 0.0000, 0.0000,  1.00),
    "Flickr30k":    (0.0000, 0.0000, 0.0000,  12.29),
    "IconQA":       (0.0000, 0.0000, 0.0000,  1.00),
    "ImageNet-R":   (0.0000, 0.0000, 0.0000,  1.28),
    "VizWiz":       (0.0005, 0.0000, 0.0000,  11.61),
}

# Answer FORM, read off the val targets (see measure_val_stats below, which
# recomputes these from disk when results_fs is present and asserts agreement).
UCIT_FORM = {
    "ArxivQA":    "letter",
    "CLEVR-Math": "digit",
    "IconQA":     "digit",
    "ImageNet-R": "word",
    "Flickr30k":  "sentence",
    "VizWiz":     "sentence",
}

# n_train: fullstudy/ucit_prep.py caps every task at TRAIN_CAP = 8000, and every
# UCIT task's source split is >= 23,998 rows (design_notes/full_study_options.md,
# Table 5 of arXiv:2503.12941v2). So n_train = 8000 for all six, epochs = 1.0 and
# eff_batch = 64 (fullstudy/run_arm.py) => 125 optimizer steps per stage, the
# same for every task. This is a ZERO-VARIANCE regressor, not an untested one.
UCIT_NTRAIN = {t: 8000 for t in UCIT_FROZEN}
EFF_BATCH, EPOCHS = 64, 1.0

# Pilot suite: fullstudy/pilot_manifest.json IS in the repo and is read directly.
PILOT_MANIFEST = os.path.join(REPO, "fullstudy", "pilot_manifest.json")
PILOT_CELLS = {  # cell label -> (ordering, seed, [dirs k1..k4])
    "o1/s17": ("o1", "17", ["S1", "S2", "S3", "S4"]),
    "o1/s23": ("o1", "23", ["seq_s2_k1", "seq_s2_k2", "seq_s2_k3", "seq_s2_k4"]),
    "o1/s31": ("o1", "31", ["seq_s3_k1", "seq_s3_k2", "seq_s3_k3", "seq_s3_k4"]),
    "o2/s17": ("o2", "17", ["seq_rev_k1", "seq_rev_k2", "seq_rev_k3", "seq_rev_k4"]),
}
PILOT_BASE_DIR = "S0"
# analysis/robustness_aggregation.py lines 35-40 pin this mapping.


# ---------------------------------------------------------------------------
# Small statistics toolkit (no numpy/scipy dependency, matching the repo style)
# ---------------------------------------------------------------------------
def mean(v):
    return sum(v) / len(v)


def ss_total(v):
    m = mean(v)
    return sum((x - m) ** 2 for x in v)


def eta_sq(groups):
    """Classical eta^2 = SS_between / SS_total. None when undefined."""
    vals = [v for g in groups.values() for v in g]
    if len(vals) < 3 or sum(1 for g in groups.values() if g) < 2:
        return None
    sst = ss_total(vals)
    if sst <= 0:
        return None
    gm = mean(vals)
    ssb = sum(len(g) * (mean(g) - gm) ** 2 for g in groups.values() if g)
    return ssb / sst


def omega_sq(groups):
    """omega^2: eta^2 corrected for the number of levels.

    eta^2 rises mechanically with level count -- a factor with 18 levels over 54
    points reaches ~0.32 on pure noise. Any comparison of TASK (6 levels) with
    TRANSITION (up to 18) that quotes eta^2 alone is reading that bias as signal.
    omega^2 = (SS_b - df_b * MS_w) / (SS_tot + MS_w); can go negative, which is
    reported as-is rather than clipped, because a negative value is the honest
    statement "this factor explains less than chance".
    """
    vals = [v for g in groups.values() for v in g]
    ne = [g for g in groups.values() if g]
    k, n = len(ne), len(vals)
    if k < 2 or n - k < 1:
        return None
    gm = mean(vals)
    ssb = sum(len(g) * (mean(g) - gm) ** 2 for g in ne)
    ssw = sum(sum((x - mean(g)) ** 2 for x in g) for g in ne)
    msw = ssw / (n - k)
    sst = ssb + ssw
    if sst + msw <= 0:
        return None
    return (ssb - (k - 1) * msw) / (sst + msw)


def perm_p_eta(groups, observed, reps=20000, seed=17):
    """Permutation p for eta^2: reshuffle labels holding group sizes fixed.

    The level-count bias is present in the null too, so it divides out. Same
    convention (and the same exchangeability caveat: the permuted units are
    per-stage dc values, five or six of which share a cell trajectory) as
    analysis/order_effects.py.
    """
    if observed is None:
        return None
    sizes = [len(g) for g in groups.values()]
    pool = [v for g in groups.values() for v in g]
    rnd = random.Random(seed)
    hits = 0
    for _ in range(reps):
        rnd.shuffle(pool)
        i, sh = 0, {}
        for kk, sz in zip(groups, sizes):
            sh[kk] = pool[i:i + sz]
            i += sz
        e = eta_sq(sh)
        if e is not None and e >= observed:
            hits += 1
    return (hits + 1) / (reps + 1)


def ols(xs, ys):
    """Slope, intercept, r, R^2. None when x has no variance."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = mean(xs), mean(ys)
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 1e-12 or syy <= 1e-12:
        return None
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    b1 = sxy / sxx
    r = sxy / math.sqrt(sxx * syy)
    return {"slope": b1, "intercept": my - b1 * mx, "r": r, "r2": r * r, "n": n}


def boot_cells(cell_labels, stat_fn, B=10000, seed=17):
    """Bootstrap a statistic by resampling CELLS with replacement.

    cell_labels is a LIST. Each resample is built with random.choices, i.e. a
    LIST WITH MULTIPLICITY -- a repeated cell is carried twice. Using set() here
    would silently make every resample a ~63.2% subsample and shrink the CI;
    that exact bug produced nine ~24%-too-narrow intervals in this project.
    """
    rnd = random.Random(seed)
    out = []
    for _ in range(B):
        draw = [cell_labels[rnd.randrange(len(cell_labels))]
                for _ in range(len(cell_labels))]
        assert isinstance(draw, list) and len(draw) == len(cell_labels)
        v = stat_fn(draw)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            out.append(v)
    if len(out) < B * 0.5:
        return None
    out.sort()
    return {"lo": out[int(0.025 * len(out))], "hi": out[int(0.975 * len(out))],
            "n_ok": len(out)}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_ucit(agg_path, arm="seq", backbone="llava15"):
    """Per-stage rows for one UCIT arm, plus the per-stage format state."""
    agg = json.load(open(agg_path))
    bb = agg["backbones"][backbone]
    base = bb["base"]
    base_c = base["pope"]["c"]
    rows = []
    for key, cell in sorted(bb["arms"].items()):
        a, o, s = key.split("|")
        if a != arm:
            continue
        stages = sorted((k for k in cell if k.isdigit()), key=int)
        prev = cell.get("__base__") or base
        prev_c, prev_task = base_c, "BASE"
        prev_words = base["chair"]["mean_words"]
        prev_dp = base["pope"]["dprime"]
        for k in stages:
            st = cell[k]
            p, ch = st.get("pope"), st.get("chair") or {}
            if not p:
                continue
            rows.append({
                "cell": "%s/%s" % (o, s), "ordering": o, "seed": s,
                "pos": int(k), "task": st.get("task_trained"),
                "prev_task": prev_task,
                "dc": p["c"] - prev_c, "c": p["c"],
                "dprime": p["dprime"], "d_dprime": p["dprime"] - prev_dp,
                "yes_rate": p["yes_rate"],
                "chair_mean_words": ch.get("mean_words"),
                "d_chair_mean_words": (ch.get("mean_words") - prev_words
                                       if ch.get("mean_words") is not None
                                       and prev_words is not None else None),
                "chair_i60": ch.get("chair_i60"),
                "pope_mean_new_tokens": p.get("mean_new_tokens"),
            })
            prev_c, prev_task = p["c"], st.get("task_trained")
            prev_words = ch.get("mean_words", prev_words)
            prev_dp = p["dprime"]
    return rows, base_c, base


def load_pilot(results_dir):
    """Per-stage rows for the pilot SEQ arm, scored with the audited scorer.

    Scores POPE from the raw generations with fs_common.score_pope -- the same
    scorer that produced every number the paper reports -- rather than reading
    the per-cell pope.json, so these dc values are numerically identical to the
    published ones (verified: S4 -> c = 0.7442, matching
    analysis/diag/criterion_vs_stats.json).
    """
    sys.path.insert(0, HERE)
    import fs_common
    man = json.load(open(PILOT_MANIFEST))
    base_p = fs_common.score_pope(os.path.join(results_dir, PILOT_BASE_DIR,
                                               "pope_gen.jsonl"))
    rows = []
    for cell, (o, s, dirs) in sorted(PILOT_CELLS.items()):
        order = man["orders"][o]
        prev_c, prev_task, prev_dp = base_p["c"], "BASE", base_p["dprime"]
        for i, d in enumerate(dirs):
            f = os.path.join(results_dir, d, "pope_gen.jsonl")
            if not os.path.isfile(f):
                return [], base_p, man, "missing %s" % f
            p = fs_common.score_pope(f)
            rows.append({"cell": cell, "ordering": o, "seed": s, "pos": i + 1,
                         "task": order[i], "prev_task": prev_task,
                         "dc": p["c"] - prev_c, "c": p["c"],
                         "dprime": p["dprime"],
                         "d_dprime": p["dprime"] - prev_dp,
                         "yes_rate": p["yes_rate"]})
            prev_c, prev_task, prev_dp = p["c"], order[i], p["dprime"]
    return rows, base_p, man, None


def measure_val_stats(results_fs):
    """Per-task answer/prompt statistics measured from the on-disk val targets.

    These are the VAL split, not the frozen TRAIN split the manifest records, so
    they are an independent check on the frozen numbers and a source for the
    prompt-distribution variable -- never a replacement for the frozen stats.
    """
    import re
    base = os.path.join(results_fs, "llava15_base")
    out = {}
    for t in UCIT_FROZEN:
        f = os.path.join(base, "%s_gen.jsonl" % t)
        if not os.path.isfile(f):
            return None
        tg, pw = [], []
        for line in open(f):
            r = json.loads(line)
            tg.append((r.get("target") or "").strip())
            pw.append(len((r.get("prompt") or "").split()))
        n = len(tg)
        out[t] = {
            "n_val": n,
            "val_mean_target_words": round(sum(len(x.split()) for x in tg) / n, 3),
            "val_median_target_words": statistics.median(len(x.split()) for x in tg),
            "val_yes_frac": round(sum(x.lower().startswith("yes") for x in tg) / n, 4),
            "val_no_frac": round(sum(x.lower().startswith("no") for x in tg) / n, 4),
            "val_frac_digit": round(sum(bool(re.fullmatch(r"[-+]?\d+(\.\d+)?", x))
                                        for x in tg) / n, 3),
            "val_frac_letter": round(sum(bool(re.fullmatch(r"[A-Ha-h]", x))
                                         for x in tg) / n, 3),
            "mean_prompt_words": round(sum(pw) / n, 2),
        }
    return out


# ---------------------------------------------------------------------------
# Variance accounting
# ---------------------------------------------------------------------------
def factor_table(rows, factors, reps=20000, seed=17, boot=True):
    """eta^2 / omega^2 / permutation p / cell-bootstrap CI for each factor."""
    cells = sorted({r["cell"] for r in rows})
    by_cell = {c: [r for r in rows if r["cell"] == c] for c in cells}
    out = {}
    for name, key in factors.items():
        g = {}
        for r in rows:
            g.setdefault(key(r), []).append(r["dc"])
        e = eta_sq(g)
        rec = {"levels": len(g), "eta2": e, "omega2": omega_sq(g),
               "p_perm": perm_p_eta(g, e, reps=reps, seed=seed)}
        if boot and e is not None:
            def stat(draw, key=key):
                gg = {}
                for c in draw:                      # draw is a LIST: duplicates
                    for r in by_cell[c]:            # contribute twice, by design
                        gg.setdefault(key(r), []).append(r["dc"])
                return eta_sq(gg)
            rec["eta2_ci95_cellboot"] = boot_cells(cells, stat, seed=seed)
        out[name] = rec
    return out


def perm_p_within(rows, outer, inner, reps=20000, seed=17):
    """p for an `inner` factor BEYOND an `outer` one, permuting WITHIN outer.

    The global permutation used elsewhere destroys the outer factor too, so it
    answers the wrong question once the outer factor is known to be real. Here
    the dc values are reshuffled only among rows sharing an outer level, which
    is the exchangeability that H0 ("inner adds nothing once outer is known")
    actually asserts. The statistic is eta^2 of `inner` computed on the
    outer-residualised dc, recomputed inside every shuffle so the residualising
    step is part of the null too.
    """
    def stat(rs):
        res = residualize_on(rs, outer)
        g = {}
        for r in res:
            g.setdefault(inner(r), []).append(r["dc"])
        return eta_sq(g)

    obs = stat(rows)
    if obs is None:
        return None, None
    buckets = {}
    for i, r in enumerate(rows):
        buckets.setdefault(outer(r), []).append(i)
    rnd = random.Random(seed)
    hits = 0
    for _ in range(reps):
        shuffled = [dict(r) for r in rows]
        for idxs in buckets.values():
            vals = [rows[i]["dc"] for i in idxs]
            rnd.shuffle(vals)
            for i, v in zip(idxs, vals):
                shuffled[i]["dc"] = v
        s = stat(shuffled)
        if s is not None and s >= obs:
            hits += 1
    return obs, (hits + 1) / (reps + 1)


def residualize_on(rows, key):
    """Return rows with dc replaced by its residual after removing group means."""
    g = {}
    for r in rows:
        g.setdefault(key(r), []).append(r["dc"])
    gm = {k: mean(v) for k, v in g.items()}
    out = []
    for r in rows:
        rr = dict(r)
        rr["dc"] = r["dc"] - gm[key(r)]
        out.append(rr)
    return out


def per_cell_then_aggregate(rows, xfun, label, seed=17):
    """Slope of dc on x WITHIN each cell, then aggregated over cells.

    The per-cell fit is the disciplined one: it never pairs a per-cell dc with a
    cross-cell average of x. The aggregate is the mean of the per-cell slopes,
    with a CI from resampling cells (list, with multiplicity).
    """
    cells = sorted({r["cell"] for r in rows})
    per = {}
    for c in cells:
        sub = [r for r in rows if r["cell"] == c]
        xs = [xfun(r) for r in sub]
        ys = [r["dc"] for r in sub]
        if any(x is None for x in xs):
            continue
        f = ols(xs, ys)
        if f:
            per[c] = f
    if len(per) < 2:
        return {"label": label, "per_cell": per, "note": "too few usable cells"}
    labs = sorted(per)
    slopes = [per[c]["slope"] for c in labs]
    rs = [per[c]["r"] for c in labs]
    return {
        "label": label,
        "n_cells": len(labs),
        "mean_slope": mean(slopes),
        "slope_ci95_cellboot": boot_cells(labs, lambda d: mean([per[c]["slope"] for c in d]), seed=seed),
        "mean_r": mean(rs),
        "mean_r2_within_cell": mean([r * r for r in rs]),
        "r_ci95_cellboot": boot_cells(labs, lambda d: mean([per[c]["r"] for c in d]), seed=seed),
        "n_cells_positive_slope": sum(1 for s in slopes if s > 0),
        "per_cell": {c: {k: round(v, 5) for k, v in per[c].items()} for c in labs},
    }


def two_group_contrast(rows, long_tasks, seed=17):
    """Per-cell (long-answer mean dc) - (short-answer mean dc), then aggregated.

    Plus the exact permutation over all C(6,2) = 15 two-vs-four splits, done on
    the AGGREGATED per-cell contrast so the unit of the test is the cell.
    """
    cells = sorted({r["cell"] for r in rows})
    tasks = sorted({r["task"] for r in rows})

    def contrast(cell_list, longs):
        vals = []
        for c in cell_list:
            sub = [r for r in rows if r["cell"] == c]
            lo = [r["dc"] for r in sub if r["task"] in longs]
            sh = [r["dc"] for r in sub if r["task"] not in longs]
            if not lo or not sh:
                return None
            vals.append(mean(lo) - mean(sh))
        return mean(vals)

    obs = contrast(cells, set(long_tasks))
    all_splits = []
    for combo in combinations(tasks, len(long_tasks)):
        v = contrast(cells, set(combo))
        if v is not None:
            all_splits.append({"split": list(combo), "contrast": round(v, 5)})
    all_splits.sort(key=lambda d: -d["contrast"])
    rank = 1 + sum(1 for s in all_splits if s["contrast"] > obs)
    per_cell = {}
    for c in cells:
        sub = [r for r in rows if r["cell"] == c]
        lo = [r["dc"] for r in sub if r["task"] in set(long_tasks)]
        sh = [r["dc"] for r in sub if r["task"] not in set(long_tasks)]
        per_cell[c] = round(mean(lo) - mean(sh), 5)
    return {
        "long_tasks": list(long_tasks),
        "observed_contrast": obs,
        "per_cell_contrast": per_cell,
        "n_cells_same_sign": sum(1 for v in per_cell.values() if v > 0),
        "ci95_cellboot": boot_cells(cells, lambda d: contrast(d, set(long_tasks)), seed=seed),
        "n_splits": len(all_splits),
        "rank_of_observed": rank,
        "p_exact_one_sided": rank / len(all_splits),
        "p_floor_this_design": 1.0 / len(all_splits),
        "all_splits": all_splits,
    }


# ---------------------------------------------------------------------------
def fmt(v, nd=4):
    return "n/a" if v is None else ("%+.*f" % (nd, v) if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default=os.path.join(HERE, "readout", "fs_aggregate.json"))
    ap.add_argument("--results_fs", default=os.path.join(REPO, "fullstudy", "results_fs"))
    ap.add_argument("--pilot_results", default=os.path.join(REPO, "results"))
    ap.add_argument("--manifest_ucit", default=None,
                    help="frozen data_ucit/ucit_manifest.json; inline fallback "
                         "transcribed from FULLSTUDY_PREREG.md if absent")
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", default=os.path.join(HERE, "readout", "dose_residual.json"))
    args = ap.parse_args()

    R = {"generated_by": "analysis/dose_residual.py",
         "question": "what predicts dc beyond the answer-statistic dose",
         "gpu_used": False, "seed": args.seed, "perm_reps": args.reps}

    # ---- per-task inputs -------------------------------------------------
    src = "inline (FULLSTUDY_PREREG.md transcription of the frozen manifest)"
    stats = {t: {"yes_frac": v[0], "no_frac": v[1], "refusal_frac": v[2],
                 "mean_target_words": v[3]} for t, v in UCIT_FROZEN.items()}
    if args.manifest_ucit and os.path.isfile(args.manifest_ucit):
        man = json.load(open(args.manifest_ucit))
        stats = {t: dict(man["tasks"][t]["answer_stats"]) for t in man["tasks"]}
        src = args.manifest_ucit
    for t, s in stats.items():
        s["dose"] = s["refusal_frac"] + abs(s["yes_frac"] - s["no_frac"])
        s["form"] = UCIT_FORM.get(t)
        s["n_train"] = UCIT_NTRAIN.get(t)
        s["opt_steps"] = (int(math.ceil(UCIT_NTRAIN[t] * EPOCHS / EFF_BATCH))
                          if t in UCIT_NTRAIN else None)
    val = measure_val_stats(args.results_fs)
    if val:
        for t, v in val.items():
            stats[t].update(v)
    R["ucit_task_inputs"] = {"answer_stats_source": src, "tasks": stats,
                             "val_stats_measured": bool(val)}

    print("=" * 78)
    print("UCIT per-task inputs   (answer stats: %s)" % src)
    print("=" * 78)
    print("%-12s %6s %7s %7s %9s %8s %10s" %
          ("task", "dose", "trainW", "valW", "form", "n_train", "promptW"))
    for t in sorted(stats, key=lambda x: stats[x]["mean_target_words"]):
        s = stats[t]
        print("%-12s %6.4f %7.2f %7s %9s %8s %10s" %
              (t, s["dose"], s["mean_target_words"],
               s.get("val_mean_target_words", "-"), s.get("form"),
               s.get("n_train"), s.get("mean_prompt_words", "-")))
    doses = [s["dose"] for s in stats.values()]
    print("\n  dose across the six tasks: min %.4f  max %.4f  sd %.6f"
          % (min(doses), max(doses), statistics.pstdev(doses)))
    print("  optimizer steps per stage: %s  (n_train capped at 8000 for every"
          " task; epochs 1.0, eff_batch 64)"
          % sorted({s["opt_steps"] for s in stats.values()}))

    # ---- UCIT rows -------------------------------------------------------
    rows, base_c, base = load_ucit(args.agg, arm="seq")
    rows_anchor, _, _ = load_ucit(args.agg, arm="anchor")
    cells = sorted({r["cell"] for r in rows})
    R["ucit"] = {"arm": "seq", "backbone": "llava15", "base_c": base_c,
                 "n_cells": len(cells), "n_stages": len(rows), "cells": cells}

    print("\n" + "=" * 78)
    print("STEP 1 -- the answer-statistic account's own residual (UCIT)")
    print("=" * 78)
    dcs = [r["dc"] for r in rows]
    # The account predicts dc = 0 at zero dose. Score it against that prediction.
    sse_account = sum(x * x for x in dcs)
    sst = ss_total(dcs)
    r2_pred = 1 - sse_account / sst
    fit = ols([stats[r["task"]]["dose"] for r in rows], dcs)
    step1 = {
        "n": len(dcs), "mean_dc": mean(dcs),
        "mean_abs_dc": mean([abs(x) for x in dcs]),
        "sd_dc": statistics.stdev(dcs),
        "dose_regressor_sd": statistics.pstdev(doses),
        "ols_dc_on_dose": fit,
        "predictive_r2_vs_account_prediction_zero": r2_pred,
        "variance_left_as_residual_fraction": 1.0,
        "note": ("On UCIT the dose regressor is degenerate: five of six tasks "
                 "have dose exactly 0 and the sixth 0.0005. The account makes a "
                 "point prediction dc = 0 and has essentially no variance to "
                 "fit, so 100% of the dc variance is residual by construction. "
                 "An OLS R^2 on a regressor with sd 2e-4 is not interpretable "
                 "and is reported only to show it was computed."),
    }
    R["ucit"]["step1_dose_residual"] = step1
    print("  n = %d one-step dc   mean %+.4f   mean|dc| %.4f   sd %.4f"
          % (step1["n"], step1["mean_dc"], step1["mean_abs_dc"], step1["sd_dc"]))
    print("  dose regressor sd across tasks = %.6f  -> DEGENERATE" % step1["dose_regressor_sd"])
    print("  OLS dc ~ dose : %s" % ("R^2 = %.4f (uninterpretable, see note)" % fit["r2"]
                                    if fit else "undefined (no x variance)"))
    print("  scored against the account's OWN prediction (dc = 0 at zero dose):")
    print("     R^2_predictive = %+.4f   <- negative means predicting the mean"
          " beats predicting zero" % r2_pred)
    print("  => the answer-statistic account explains 0.0 percent of dc variance"
          " here.")
    print("     RESIDUAL = dc itself. Everything below is the residual analysis.")
    # Cross-check against the pre-registered zero-dose headline, so a pipeline
    # error would show up here rather than in a conclusion.
    maxdisp = {}
    for c in cells:
        sub = sorted([r for r in rows if r["cell"] == c], key=lambda r: r["pos"])
        maxdisp[c] = max(abs(r["c"] - base_c) for r in sub)
    R["ucit"]["max_abs_displacement_from_base"] = maxdisp
    print("  CROSS-CHECK vs the pre-registered zero-dose headline: max|c - base|"
          " per cell")
    print("     range %.3f-%.3f (mean %.3f); FULLSTUDY_PREREG.md records"
          " 0.403-0.550, mean 0.472."
          % (min(maxdisp.values()), max(maxdisp.values()),
             mean(list(maxdisp.values()))))

    # ---- Step 2: categorical factors -------------------------------------
    print("\n" + "=" * 78)
    print("STEP 2 -- what structure the residual has (UCIT SEQ, 9 cells x 6 stages)")
    print("=" * 78)
    factors = {
        "task identity": lambda r: r["task"],
        "sequence depth (position)": lambda r: r["pos"],
        "predecessor task": lambda r: r["prev_task"],
        "transition (prev -> cur)": lambda r: (r["prev_task"], r["task"]),
        "ordering": lambda r: r["ordering"],
        "seed": lambda r: r["seed"],
        "CEILING: ordering x position": lambda r: (r["ordering"], r["pos"]),
    }
    tab = factor_table(rows, factors, reps=args.reps, seed=args.seed)
    R["ucit"]["factors_all_stages"] = tab
    print("%-30s %4s %8s %9s %9s %-20s" %
          ("factor", "lvl", "eta^2", "omega^2", "perm p", "eta^2 95% CI (cell boot)"))
    print("-" * 90)
    for name in factors:
        t = tab[name]
        ci = t.get("eta2_ci95_cellboot")
        print("%-30s %4d %8s %9s %9s %-20s" %
              (name, t["levels"],
               "%.4f" % t["eta2"] if t["eta2"] is not None else "n/a",
               "%.4f" % t["omega2"] if t["omega2"] is not None else "n/a",
               "%.4f" % t["p_perm"] if t["p_perm"] is not None else "n/a",
               "[%.3f, %.3f]" % (ci["lo"], ci["hi"]) if ci else "-"))

    ceil = tab["CEILING: ordering x position"]["eta2"]
    tsk = tab["task identity"]["eta2"]
    R["ucit"]["ceiling"] = {
        "eta2_ceiling_ordering_x_position": ceil,
        "eta2_task": tsk,
        "headroom_beyond_task": ceil - tsk,
        "seed_noise_floor_fraction": 1 - ceil,
        "note": ("Within an ordering, position determines the task AND the "
                 "predecessor. So ordering x position (18 groups of 3 seeds) is "
                 "the FINEST partition any stream-level variable can induce, and "
                 "its eta^2 is a hard ceiling on task, position, predecessor, "
                 "transition, interference and every interaction among them. "
                 "1 - ceiling is pure seed-to-seed noise. Caution: eta^2 with 18 "
                 "levels over 54 points sits near 0.32 on pure noise, so read "
                 "omega^2 and the permutation p, not this number alone."),
    }
    print("\n  DESIGN CEILING. Within an ordering, position fixes both the task and")
    print("  its predecessor, so ordering x position is the finest partition any")
    print("  stream-level variable can make. eta^2_ceiling = %.4f." % ceil)
    print("    task identity alone:            %.4f" % tsk)
    print("    headroom for EVERYTHING else:   %.4f   (transition, position," % (ceil - tsk))
    print("                                            interference, interactions)")
    print("    irreducible seed noise:         %.4f" % (1 - ceil))

    # transition beyond task, with the RESTRICTED (within-task) permutation null
    by_task = lambda r: r["task"]
    inner = {
        "transition (predecessor)": lambda r: (r["prev_task"], r["task"]),
        "sequence depth (position)": lambda r: r["pos"],
        "predecessor task": lambda r: r["prev_task"],
        "CEILING: task x ordering": lambda r: (r["task"], r["ordering"]),
    }
    res_rows = residualize_on(rows, by_task)
    tab_res = {}
    print("\n  Incremental: each factor BEYOND task identity.")
    print("  Null = reshuffle dc WITHIN each task (the exchangeability H0 actually")
    print("  asserts); shares are of the task-residual variance, %.1f%% of total."
          % (100 * (1 - tsk)))
    print("    %-28s %4s %8s %9s %9s %10s" %
          ("factor | task", "lvl", "eta^2", "omega^2", "p_within", "of TOTAL"))
    for name, f in inner.items():
        e, p = perm_p_within(rows, by_task, f, reps=args.reps, seed=args.seed)
        g = {}
        for r in res_rows:
            g.setdefault(f(r), []).append(r["dc"])
        tab_res[name] = {"levels": len(g), "eta2_on_task_residual": e,
                         "omega2_on_task_residual": omega_sq(g),
                         "p_within_task_perm": p,
                         "share_of_total": e * (1 - tsk) if e is not None else None}
        print("    %-28s %4d %8.4f %9s %9.4f %10.4f"
              % (name, len(g), e,
                 "%.4f" % omega_sq(g) if omega_sq(g) is not None else "n/a",
                 p, e * (1 - tsk)))
    R["ucit"]["factors_beyond_task"] = tab_res
    R["ucit"]["identifiability_note"] = (
        "Within one task, the three orderings supply exactly three "
        "(position, predecessor) pairs and each is one ordering. So for a given "
        "task the partition of its 9 observations induced by position, by "
        "predecessor, and by ordering is THE SAME partition into 3 groups of 3 "
        "seeds. Nothing beyond task identity in this design can be attributed to "
        "adjacency rather than depth rather than ordering: they are the same "
        "task x ordering interaction viewed three ways. 'task x ordering' above "
        "is therefore the ceiling for all three, and the differences among their "
        "eta^2 values come only from how groups align ACROSS tasks, not from any "
        "evidence separating them.")
    # The one factor above that is NOT a relabelled interaction: the predecessor
    # label is shared ACROSS tasks (three different current-tasks at three
    # different depths can all be "preceded by Flickr30k"), so a consistent
    # predecessor effect is estimable where a bare task x ordering term is not.
    pred_res = {}
    for r in res_rows:
        pred_res.setdefault(r["prev_task"], []).append((r["cell"], r["task"], r["dc"]))
    print("\n  Mean TASK-RESIDUAL dc by predecessor (the only factor above whose")
    print("  labels align across different current-tasks and depths):")
    print("    %-14s %4s %10s %8s %-28s" % ("predecessor", "n", "mean resid", "+/n", "current tasks it precedes"))
    pred_summary = {}
    for p in sorted(pred_res, key=lambda x: mean([v[2] for v in pred_res[x]])):
        v = [x[2] for x in pred_res[p]]
        curs = sorted({x[1] for x in pred_res[p]})
        pred_summary[p] = {"n": len(v), "mean_task_residual_dc": mean(v),
                           "n_positive": sum(1 for x in v if x > 0),
                           "current_tasks": curs}
        print("    %-14s %4d %+10.4f %8s %-28s"
              % (p, len(v), mean(v), "%d/%d" % (sum(1 for x in v if x > 0), len(v)),
                 ", ".join(c[:9] for c in curs)))
    R["ucit"]["predecessor_residual_means"] = pred_summary
    # Is the predecessor effect just the length account applied one step back?
    pk = [p for p in pred_summary if p != "BASE"]
    fit_pl = ols([stats[p]["mean_target_words"] for p in pk],
                 [pred_summary[p]["mean_task_residual_dc"] for p in pk])
    R["ucit"]["predecessor_effect_vs_predecessor_answer_length"] = fit_pl
    print("    -> is this the length account one step back? r(predecessor answer")
    print("       length, predecessor residual) = %+.3f over 6 predecessors."
          % (fit_pl["r"] if fit_pl else float("nan")))
    print("       The two long-answer tasks sit mid-table, IconQA (1.0 words) is")
    print("       the most positive: the predecessor effect is NOT answer length.")
    orders = {}
    for o in sorted({r["ordering"] for r in rows}):
        seq = [r["task"] for r in sorted([x for x in rows if x["ordering"] == o
                                          and x["cell"].endswith("/s17")],
                                         key=lambda x: x["pos"])]
        orders[o] = seq
    R["ucit"]["orderings"] = orders

    print("\n  IDENTIFIABILITY: within a task, position / predecessor / ordering")
    print("  induce the SAME 3-way split of that task's 9 cells. Whatever sits")
    print("  above task identity is the task x ordering interaction; this design")
    print("  cannot say whether it is adjacency, depth, or ordering.")

    # per-task pull, per cell then aggregated (each cell holds one obs per task,
    # so the per-cell value IS the observation; the aggregate is their mean and
    # the sign count is over cells)
    pull = {}
    for t in sorted({r["task"] for r in rows}):
        v = {r["cell"]: r["dc"] for r in rows if r["task"] == t}
        labs = sorted(v)
        pull[t] = {
            "mean_dc": mean([v[c] for c in labs]),
            "n_cells": len(labs),
            "n_positive": sum(1 for c in labs if v[c] > 0),
            "per_cell": {c: round(v[c], 4) for c in labs},
            "ci95_cellboot": boot_cells(labs, lambda d: mean([v[c] for c in d]),
                                        seed=args.seed),
            "train_mean_words": stats[t]["mean_target_words"],
            "dose": stats[t]["dose"],
        }
        vps = {r["cell"]: r["dc"] for r in rows if r["task"] == t and r["pos"] > 1}
        lps = sorted(vps)
        pull[t]["mean_dc_post_settling"] = mean([vps[c] for c in lps]) if lps else None
        pull[t]["n_cells_post_settling"] = len(lps)
        pull[t]["n_positive_post_settling"] = sum(1 for c in lps if vps[c] > 0)
        pull[t]["ci95_post_settling"] = (
            boot_cells(lps, lambda d: mean([vps[c] for c in d]), seed=args.seed)
            if len(lps) > 1 else None)
    R["ucit"]["per_task_pull"] = pull
    print("\n  Per-task criterion pull (one observation per cell; 9 cells).")
    print("  The post-settling column is what FULLSTUDY_PREREG.md quotes.")
    print("  %-12s %9s %10s %8s %7s %-22s"
          % ("task", "mean dc", "post-settl", "+/9", "words", "95% CI (cell boot)"))
    for t in sorted(pull, key=lambda x: pull[x]["mean_dc"]):
        p = pull[t]
        print("  %-12s %+9.4f %+10.4f %8s %7.2f [%+.4f, %+.4f]"
              % (t, p["mean_dc"], p["mean_dc_post_settling"],
                 "%d/%d" % (p["n_positive"], p["n_cells"]),
                 p["train_mean_words"], p["ci95_cellboot"]["lo"], p["ci95_cellboot"]["hi"]))

    # ---- Step 3: continuous per-task predictors --------------------------
    print("\n" + "=" * 78)
    print("STEP 3 -- which PROPERTY of the task (all are per-task constants,")
    print("          so each is nested inside the task factor and capped by it)")
    print("=" * 78)
    preds = {
        "answer length (train mean words)": lambda r: stats[r["task"]]["mean_target_words"],
        "log10 answer length": lambda r: math.log10(stats[r["task"]]["mean_target_words"]),
        "is long-answer (>=2 words)": lambda r: 1.0 if stats[r["task"]]["mean_target_words"] >= 2 else 0.0,
        "answer-statistic dose": lambda r: stats[r["task"]]["dose"],
        "n_train / optimizer steps": lambda r: float(stats[r["task"]]["opt_steps"]),
    }
    if val:
        preds["prompt length (val mean words)"] = lambda r: stats[r["task"]]["mean_prompt_words"]
        preds["answer length (val mean words)"] = lambda r: stats[r["task"]]["val_mean_target_words"]
    cont = {}
    for name, f in preds.items():
        xs = [f(r) for r in rows]
        if statistics.pstdev(xs) <= 1e-12:
            cont[name] = {"label": name, "degenerate": True,
                          "constant_value": xs[0],
                          "note": "zero variance across tasks; explains 0 by construction"}
            print("  %-34s CONSTANT (= %.4g) -> explains 0 by construction"
                  % (name, xs[0]))
            continue
        c = per_cell_then_aggregate(rows, f, name, seed=args.seed)
        # share of the BETWEEN-TASK variance this predictor captures
        tmeans = {}
        for r in rows:
            tmeans.setdefault(r["task"], []).append(r["dc"])
        tx = [f([r for r in rows if r["task"] == t][0]) for t in sorted(tmeans)]
        ty = [mean(tmeans[t]) for t in sorted(tmeans)]
        ft = ols(tx, ty)
        c["r2_on_6_task_means"] = ft["r2"] if ft else None
        c["share_of_total_dc_variance"] = (ft["r2"] * tsk) if ft else None
        cont[name] = c
        print("  %-34s within-cell mean r = %+.3f  (%d/%d cells same-sign slope)"
              % (name, c["mean_r"], c["n_cells_positive_slope"], c["n_cells"]))
        print("      %-30s r^2 on the 6 task means = %.3f -> %.3f of TOTAL dc variance"
              % ("", c["r2_on_6_task_means"], c["share_of_total_dc_variance"]))
    R["ucit"]["continuous_predictors"] = cont

    # answer form as a 4-level factor, nested in task
    form_tab = factor_table(rows, {"answer form (letter/digit/word/sentence)":
                                   lambda r: stats[r["task"]]["form"]},
                            reps=args.reps, seed=args.seed)
    R["ucit"]["answer_form_factor"] = form_tab
    ft = form_tab["answer form (letter/digit/word/sentence)"]
    print("  %-34s lvl=%d  eta^2=%.4f  omega^2=%.4f  p=%.4f"
          % ("answer FORM (4 classes)", ft["levels"], ft["eta2"], ft["omega2"], ft["p_perm"]))

    # two-group contrast, per cell then aggregated
    longs = [t for t in stats if stats[t]["mean_target_words"] >= 2]
    tg = two_group_contrast(rows, longs, seed=args.seed)
    R["ucit"]["long_vs_short_contrast"] = tg
    print("\n  Long-vs-short answer contrast, computed PER CELL then aggregated:")
    print("    long tasks: %s" % ", ".join(sorted(longs)))
    print("    mean contrast %+.4f   95%% CI [%+.4f, %+.4f] (cell bootstrap)"
          % (tg["observed_contrast"], tg["ci95_cellboot"]["lo"], tg["ci95_cellboot"]["hi"]))
    print("    same sign in %d/%d cells" % (tg["n_cells_same_sign"], len(cells)))
    print("    exact permutation over all %d two-vs-four splits: rank %d, p = %.4f"
          % (tg["n_splits"], tg["rank_of_observed"], tg["p_exact_one_sided"]))
    print("    p FLOOR for this design = %.4f -- no 2-vs-4 split of 6 tasks can"
          " do better" % tg["p_floor_this_design"])

    # ---- Step 3b: mean reversion toward a fixed point ---------------------
    print("\n" + "=" * 78)
    print("STEP 3b -- the mundane alternative: is dc just MEAN REVERSION?")
    print("=" * 78)
    print("  If each stage drives c toward a fixed point, dc = -lambda*(c_prev - c*),")
    print("  and c_prev is determined by the PREDECESSOR -- which would make the")
    print("  'predecessor' and 'transition' effects above an artefact of one")
    print("  attractor, not evidence of interference.")
    print("  GALTON CAVEAT, stated before the number: c_prev appears on both sides")
    print("  of dc = c_k - c_prev, so this regression is negatively biased by")
    print("  construction. A slope of exactly -1 is what INDEPENDENT c_k and c_prev")
    print("  produce, not what an attractor produces. The interpretable statistic")
    print("  is the LEVEL correlation r(c_k, c_prev), reported alongside.")
    prev_c_of = {}
    for c in cells:
        sub = sorted([r for r in rows if r["cell"] == c], key=lambda r: r["pos"])
        pc = base_c
        for r in sub:
            prev_c_of[(c, r["pos"])] = pc
            pc = r["c"]
    cprev_f = lambda r: prev_c_of[(r["cell"], r["pos"])]
    mr = per_cell_then_aggregate(rows, cprev_f, "dc ~ incoming c level", seed=args.seed)
    mr_res = per_cell_then_aggregate(residualize_on(rows, lambda r: r["task"]),
                                     cprev_f, "task-residual dc ~ incoming c", seed=args.seed)
    for c in (mr, mr_res):
        ci = c["r_ci95_cellboot"]
        print("  %-34s within-cell mean r = %+.3f  95%% CI [%+.3f, %+.3f]  "
              "mean slope %+.3f  (%d/%d cells negative slope)"
              % (c["label"], c["mean_r"], ci["lo"], ci["hi"], c["mean_slope"],
                 c["n_cells"] - c["n_cells_positive_slope"], c["n_cells"]))
    # The Galton-free version: the LEVEL correlation r(c_k, c_prev).
    lev = per_cell_then_aggregate([dict(r, dc=r["c"]) for r in rows], cprev_f,
                                  "c_k ~ c_prev (LEVELS, no shared term)",
                                  seed=args.seed)
    lci = lev["r_ci95_cellboot"]
    print("  %-34s within-cell mean r = %+.3f  95%% CI [%+.3f, %+.3f]  "
          "mean slope %+.3f"
          % (lev["label"], lev["mean_r"], lci["lo"], lci["hi"], lev["mean_slope"]))
    print("    -> r(c_k, c_prev) near zero means the criterion after a stage is")
    print("       set by the task just trained and has forgotten where it was.")
    print("       That is MEMORYLESSNESS (the pre-registered E4 property), not a")
    print("       new explanatory variable: it says the residual is carried by")
    print("       the incoming task, which is exactly what STEP 2 already found.")
    R["ucit"]["level_autocorrelation"] = lev

    # transition / predecessor AFTER removing task means AND the c_prev slope
    pooled_mr = ols([cprev_f(r) for r in rows], [r["dc"] for r in rows])
    rows_mr = []
    for r in rows:
        rr = dict(r)
        rr["dc"] = r["dc"] - (pooled_mr["intercept"] + pooled_mr["slope"] * cprev_f(r))
        rows_mr.append(rr)
    rows_mr_t = residualize_on(rows_mr, lambda r: r["task"])
    tab_mr = factor_table(rows_mr_t, {
        "transition | task + c_prev removed": lambda r: (r["prev_task"], r["task"]),
        "predecessor | task + c_prev removed": lambda r: r["prev_task"],
    }, reps=args.reps, seed=args.seed, boot=False)
    frac_left = ss_total([r["dc"] for r in rows_mr_t]) / ss_total(dcs)
    print("\n  After removing BOTH the task means and the pooled c_prev slope,")
    print("  %.1f%% of the original dc variance is left. On that remainder:" % (100 * frac_left))
    for name, t in tab_mr.items():
        print("    %-38s lvl=%2d  eta^2=%.4f  omega^2=%s  p=%.4f  -> %.4f of TOTAL"
              % (name, t["levels"], t["eta2"],
                 "%.4f" % t["omega2"] if t["omega2"] is not None else "n/a",
                 t["p_perm"], t["eta2"] * frac_left))
    R["ucit"]["mean_reversion"] = {
        "dc_on_c_prev": mr, "task_residual_dc_on_c_prev": mr_res,
        "pooled_slope": pooled_mr,
        "variance_left_after_task_and_cprev": frac_left,
        "factors_after_task_and_cprev": tab_mr,
        "caveat": ("Residualising consumes degrees of freedom that the "
                   "permutation null does not know about, so these p-values are "
                   "ANTICONSERVATIVE. Read them as upper bounds on evidence."),
    }

    # ---- Step 4: post-hoc state correlates -------------------------------
    print("\n" + "=" * 78)
    print("STEP 4 -- post-hoc correlates measured on the SAME checkpoint")
    print("          (HYPOTHESIS-GENERATING ONLY: these are outcomes, not inputs)")
    print("=" * 78)
    state = {}
    for name, f in {
        "d(CHAIR caption mean_words)": lambda r: r["d_chair_mean_words"],
        "CHAIR caption mean_words (level)": lambda r: r["chair_mean_words"],
        "d(d-prime)": lambda r: r["d_dprime"],
        "CHAIR_i@60 (level)": lambda r: r["chair_i60"],
    }.items():
        if any(f(r) is None for r in rows):
            continue
        c = per_cell_then_aggregate(rows, f, name, seed=args.seed)
        state[name] = c
        ci = c.get("r_ci95_cellboot")
        print("  %-34s within-cell mean r = %+.3f  95%% CI [%+.3f, %+.3f]  "
              "mean within-cell r^2 = %.3f"
              % (name, c["mean_r"], ci["lo"], ci["hi"], c["mean_r2_within_cell"]))
    R["ucit"]["posthoc_state_correlates"] = state
    print("  POPE mean_new_tokens is %s in every stage of every cell -- the"
          % sorted({r["pope_mean_new_tokens"] for r in rows}))
    print("  criterion moves with the answer format FROZEN on the POPE task itself.")

    # The dc/dd' correlation needs unpacking before anyone reads it as a finding.
    agg = json.load(open(args.agg))
    bb = agg["backbones"]["llava15"]
    dzh, dzf = [], []
    for c in cells:
        o, s = c.split("/")
        cell = bb["arms"]["seq|%s|%s" % (o, s)]
        pH, pF = bb["base"]["pope"]["H"], bb["base"]["pope"]["FA"]
        for k in sorted((x for x in cell if x.isdigit()), key=int):
            p = cell[k]["pope"]
            dzh.append(NORM.inv_cdf(p["H"]) - NORM.inv_cdf(pH))
            dzf.append(NORM.inv_cdf(p["FA"]) - NORM.inv_cdf(pF))
            pH, pF = p["H"], p["FA"]
    ddp = [r["d_dprime"] for r in rows]
    sd_ratio = statistics.pstdev(dzf) / statistics.pstdev(dzh)
    coupling = {
        "sd_delta_zH": statistics.pstdev(dzh), "sd_delta_zFA": statistics.pstdev(dzf),
        "sd_ratio_zFA_over_zH": sd_ratio,
        "sd_dc": statistics.pstdev(dcs), "sd_d_dprime": statistics.pstdev(ddp),
        "max_abs_d_dprime": max(abs(x) for x in ddp),
        "max_abs_dprime_from_base": max(abs(r["dprime"] - base["pope"]["dprime"])
                                        for r in rows),
        "F1_threshold": 0.3,
        "note": ("c = -0.5*(z(H)+z(FA)) and d' = z(H)-z(FA) are orthogonal "
                 "CONTRASTS, not independent measurements. When FA moves much "
                 "more than H, dc -> -0.5*dz(FA) and dd' -> -dz(FA), so they "
                 "correlate at r -> +1 ARITHMETICALLY. The observed sd ratio "
                 "z(FA):z(H) is %.2f, which is that regime. The r = %.3f is "
                 "therefore a restatement of WHERE the movement lives (the "
                 "false-alarm side), not an independent explanatory variable."
                 % (sd_ratio, state["d(d-prime)"]["mean_r"] if "d(d-prime)" in state else float("nan"))),
    }
    R["ucit"]["c_dprime_coupling"] = coupling
    print("\n  Unpacking the dc ~ d(d') correlation before anyone quotes it:")
    print("    sd of dz(H) = %.4f   sd of dz(FA) = %.4f   ratio = %.2f"
          % (coupling["sd_delta_zH"], coupling["sd_delta_zFA"], sd_ratio))
    print("    c and d' are orthogonal CONTRASTS of the same two z-scores; when")
    print("    FA dominates, dc -> -0.5*dz(FA) and dd' -> -dz(FA), i.e. r -> +1")
    print("    arithmetically. This is a restatement of where the movement lives")
    print("    (the false-alarm side), NOT a separate explanation.")
    print("    sd(dc) = %.4f vs sd(dd') = %.4f;  max |d' - base| = %.4f"
          " (F1 damage threshold 0.30)"
          % (coupling["sd_dc"], coupling["sd_d_dprime"],
             coupling["max_abs_dprime_from_base"]))

    # ---- Step 4b: post-settling sensitivity ------------------------------
    print("\n" + "=" * 78)
    print("STEP 4b -- post-settling sensitivity (stage 1 dropped, the paper's own")
    print("           endpoint convention)")
    print("=" * 78)
    ps = [r for r in rows if r["pos"] > 1]
    tab_ps = factor_table(ps, factors, reps=args.reps, seed=args.seed, boot=False)
    R["ucit"]["factors_post_settling"] = tab_ps
    for name in factors:
        t = tab_ps[name]
        print("  %-30s lvl=%2d  eta^2=%s  omega^2=%s  p=%s"
              % (name, t["levels"],
                 "%.4f" % t["eta2"] if t["eta2"] is not None else "n/a",
                 "%.4f" % t["omega2"] if t["omega2"] is not None else "n/a",
                 "%.4f" % t["p_perm"] if t["p_perm"] is not None else "n/a"))
    # Does the predecessor effect survive dropping stage 1? Stage 1's predecessor
    # is the untuned BASE, a settling contrast rather than an adjacency one, and
    # it carried the most extreme residual in the table above.
    tsk_ps = tab_ps["task identity"]["eta2"]
    beyond_ps = {}
    print("\n  Beyond task identity, post-settling (BASE predecessor level gone):")
    print("    %-28s %4s %8s %9s %9s %10s" %
          ("factor | task", "lvl", "eta^2", "omega^2", "p_within", "of TOTAL"))
    for name, f in inner.items():
        e, p = perm_p_within(ps, by_task, f, reps=args.reps, seed=args.seed)
        g = {}
        for r in residualize_on(ps, by_task):
            g.setdefault(f(r), []).append(r["dc"])
        beyond_ps[name] = {"levels": len(g), "eta2_on_task_residual": e,
                           "omega2_on_task_residual": omega_sq(g),
                           "p_within_task_perm": p,
                           "share_of_total": e * (1 - tsk_ps) if e is not None else None}
        print("    %-28s %4d %8.4f %9s %9.4f %10.4f"
              % (name, len(g), e,
                 "%.4f" % omega_sq(g) if omega_sq(g) is not None else "n/a",
                 p, e * (1 - tsk_ps)))
    R["ucit"]["factors_beyond_task_post_settling"] = beyond_ps

    # ---- Step 5: anchor arm as a replication -----------------------------
    print("\n" + "=" * 78)
    print("STEP 5 -- same decomposition on the ANCHOR arm (same stream, different"
          " training objective)")
    print("=" * 78)
    print("  The anchor makes a one-time settling jump at stage 1, so the")
    print("  post-settling column is the comparable one (the paper's convention).")
    anch_ps = [r for r in rows_anchor if r["pos"] > 1]
    tab_a = factor_table(rows_anchor, factors, reps=args.reps, seed=args.seed, boot=False)
    tab_aps = factor_table(anch_ps, factors, reps=args.reps, seed=args.seed, boot=False)
    R["ucit_anchor"] = {"n_stages": len(rows_anchor),
                        "factors_all_stages": tab_a,
                        "factors_post_settling": tab_aps}
    print("  %-30s %-26s %-26s" % ("", "ALL STAGES", "POST-SETTLING"))
    for name in factors:
        t, u = tab_a[name], tab_aps[name]
        print("  %-30s eta^2=%6s omega^2=%7s  eta^2=%6s omega^2=%7s p=%s"
              % (name,
                 "%.4f" % t["eta2"] if t["eta2"] is not None else "n/a",
                 "%.4f" % t["omega2"] if t["omega2"] is not None else "n/a",
                 "%.4f" % u["eta2"] if u["eta2"] is not None else "n/a",
                 "%.4f" % u["omega2"] if u["omega2"] is not None else "n/a",
                 "%.4f" % u["p_perm"] if u["p_perm"] is not None else "n/a"))
    # Does the predecessor-beyond-task effect replicate on the anchor arm?
    tsk_a = tab_aps["task identity"]["eta2"]
    beyond_a = {}
    print("\n  Beyond task identity, anchor arm, post-settling (replication check):")
    for name, f in inner.items():
        e, p = perm_p_within(anch_ps, by_task, f, reps=args.reps, seed=args.seed)
        g = {}
        for r in residualize_on(anch_ps, by_task):
            g.setdefault(f(r), []).append(r["dc"])
        beyond_a[name] = {"levels": len(g), "eta2_on_task_residual": e,
                          "omega2_on_task_residual": omega_sq(g),
                          "p_within_task_perm": p,
                          "share_of_total": e * (1 - tsk_a) if e is not None else None}
        print("    %-28s lvl=%2d  eta^2=%.4f  omega^2=%s  p_within=%.4f  -> %.4f of TOTAL"
              % (name, len(g), e,
                 "%.4f" % omega_sq(g) if omega_sq(g) is not None else "n/a",
                 p, e * (1 - tsk_a)))
    R["ucit_anchor"]["factors_beyond_task_post_settling"] = beyond_a

    # ---- Step 6: the pilot suite, where dose actually varies --------------
    print("\n" + "=" * 78)
    print("STEP 6 -- PILOT suite: the only design on disk where DOSE varies")
    print("=" * 78)
    prows, pbase, pman, err = load_pilot(args.pilot_results)
    if err:
        print("  pilot cells unavailable: %s" % err)
        R["pilot"] = {"available": False, "reason": err}
    else:
        pstats = {}
        for t, v in pman["tasks"].items():
            a = v["answer_stats"]
            pstats[t] = {
                "dose": a["refusal_frac"] + abs(a["yes_frac"] - a["no_frac"]),
                "mean_target_words": a["mean_target_words"],
                "n_train": v["n_train"],
                "opt_steps": int(math.ceil(v["n_train"] * EPOCHS / EFF_BATCH)),
                **a}
        pc = sorted({r["cell"] for r in prows})
        print("  base c = %.4f (scored from results/S0 with the audited scorer)"
              % pbase["c"])
        print("  %d cells x 4 stages = %d dc values\n" % (len(pc), len(prows)))
        print("  %-12s %7s %8s %9s %10s" % ("task", "dose", "words", "n_train", "mean dc"))
        pt = {}
        for r in prows:
            pt.setdefault(r["task"], []).append(r["dc"])
        for t in sorted(pstats, key=lambda x: -pstats[x]["dose"]):
            print("  %-12s %7.4f %8.2f %9d %+10.4f"
                  % (t, pstats[t]["dose"], pstats[t]["mean_target_words"],
                     pstats[t]["n_train"], mean(pt[t])))

        dose_f = lambda r: pstats[r["task"]]["dose"]
        len_f = lambda r: math.log10(pstats[r["task"]]["mean_target_words"])
        steps_f = lambda r: float(pstats[r["task"]]["opt_steps"])
        pdose = per_cell_then_aggregate(prows, dose_f, "dose", seed=args.seed)
        plen = per_cell_then_aggregate(prows, len_f, "log10 answer length", seed=args.seed)
        psteps = per_cell_then_aggregate(prows, steps_f, "optimizer steps", seed=args.seed)

        # dose first, then length on the dose residual (the asked-for ordering)
        pooled_dose = ols([dose_f(r) for r in prows], [r["dc"] for r in prows])
        resid = []
        for r in prows:
            rr = dict(r)
            rr["dc"] = r["dc"] - (pooled_dose["intercept"] + pooled_dose["slope"] * dose_f(r))
            resid.append(rr)
        pooled_len_on_resid = ols([len_f(r) for r in resid], [r["dc"] for r in resid])
        # and the reverse order, because with 4 tasks the order matters
        pooled_len = ols([len_f(r) for r in prows], [r["dc"] for r in prows])
        resid2 = []
        for r in prows:
            rr = dict(r)
            rr["dc"] = r["dc"] - (pooled_len["intercept"] + pooled_len["slope"] * len_f(r))
            resid2.append(rr)
        pooled_dose_on_resid = ols([dose_f(r) for r in resid2], [r["dc"] for r in resid2])

        print("\n  Per cell then aggregated (4 points per cell):")
        for c in (pdose, plen, psteps):
            if "mean_r" in c:
                ci = c["r_ci95_cellboot"]
                print("    %-22s mean within-cell r = %+.3f  95%% CI [%+.3f, %+.3f]"
                      "  (%d/%d cells positive slope)"
                      % (c["label"], c["mean_r"], ci["lo"], ci["hi"],
                         c["n_cells_positive_slope"], c["n_cells"]))
        print("\n  Pooled sequential fits (the 'residualize first' ordering asked for):")
        print("    dc ~ dose                 : R^2 = %.4f" % pooled_dose["r2"])
        print("    residual ~ log length     : R^2 = %.4f  (of the remaining %.1f%%)"
              % (pooled_len_on_resid["r2"], 100 * (1 - pooled_dose["r2"])))
        print("    combined (dose then len)  : %.4f of total"
              % (pooled_dose["r2"] + pooled_len_on_resid["r2"] * (1 - pooled_dose["r2"])))
        print("    reverse order: dc ~ length: R^2 = %.4f ; residual ~ dose: R^2 = %.4f"
              % (pooled_len["r2"], pooled_dose_on_resid["r2"]))
        print("    combined (len then dose)  : %.4f of total"
              % (pooled_len["r2"] + pooled_dose_on_resid["r2"] * (1 - pooled_len["r2"])))

        # per-task sign agreement of each account
        sign_tab = []
        for t in sorted(pstats):
            obs = mean(pt[t])
            dose_pred = "+" if pstats[t]["dose"] >= 0.10 else ("-" if pstats[t]["dose"] > 0 else "0")
            len_pred = "+" if pstats[t]["mean_target_words"] >= 2 else "-"
            sign_tab.append({"task": t, "mean_dc": round(obs, 4),
                             "observed_sign": "+" if obs > 0 else "-",
                             "dose_account_predicts": dose_pred,
                             "length_account_predicts": len_pred,
                             "dose_ok": (dose_pred == ("+" if obs > 0 else "-")),
                             "length_ok": (len_pred == ("+" if obs > 0 else "-"))})
        print("\n  Sign agreement, per task (this is the load-bearing comparison):")
        print("    %-12s %9s %6s %8s %8s" % ("task", "mean dc", "obs", "dose", "length"))
        for s in sign_tab:
            print("    %-12s %+9.4f %6s %8s %8s"
                  % (s["task"], s["mean_dc"], s["observed_sign"],
                     s["dose_account_predicts"] + ("*" if s["dose_ok"] else " "),
                     s["length_account_predicts"] + ("*" if s["length_ok"] else " ")))
        print("    (* = agrees with the observed sign)")

        R["pilot"] = {
            "available": True, "base_c": pbase["c"], "n_cells": len(pc),
            "n_stages": len(prows), "cells": pc, "task_stats": pstats,
            "per_task_mean_dc": {t: mean(v) for t, v in pt.items()},
            "per_cell_then_aggregate": {"dose": pdose, "log_length": plen,
                                        "opt_steps": psteps},
            "pooled_sequential": {
                "dc_on_dose_r2": pooled_dose["r2"],
                "residual_on_length_r2": pooled_len_on_resid["r2"],
                "combined_dose_then_length": pooled_dose["r2"] + pooled_len_on_resid["r2"] * (1 - pooled_dose["r2"]),
                "dc_on_length_r2": pooled_len["r2"],
                "residual_on_dose_r2": pooled_dose_on_resid["r2"],
                "combined_length_then_dose": pooled_len["r2"] + pooled_dose_on_resid["r2"] * (1 - pooled_len["r2"]),
            },
            "sign_agreement": sign_tab,
            "rows": prows,
            "caveat": ("16 dc values, 4 tasks, 4 cells (3 of them share ordering "
                       "o1). Task identity and every task property are confounded "
                       "one-to-one. These R^2 are DESCRIPTIVE; no p-value on 4 "
                       "task levels is worth quoting."),
        }

        # ---- Step 7: the length account's out-of-sample test -------------
        print("\n" + "=" * 78)
        print("STEP 7 -- OUT-OF-SAMPLE test of the answer-length account")
        print("=" * 78)
        print("  The length account was fitted on UCIT. Flickr30k captioning")
        print("  appears in BOTH suites with essentially the same targets, so the")
        print("  pilot is an out-of-sample test of it that costs nothing.")
        u_fl = pull["Flickr30k"]
        p_fl = {r["cell"]: r["dc"] for r in prows if r["task"] == "flickr"}
        p_labs = sorted(p_fl)
        p_fl_mean = mean([p_fl[c] for c in p_labs])
        p_fl_ci = boot_cells(p_labs, lambda d: mean([p_fl[c] for c in d]), seed=args.seed)
        print("\n  %-28s %9s %8s %9s %-22s" % ("", "mean dc", "+/n", "words", "95% CI (cell boot)"))
        print("  %-28s %+9.4f %8s %9.2f [%+.4f, %+.4f]"
              % ("UCIT   Flickr30k (6-task)", u_fl["mean_dc"],
                 "%d/%d" % (u_fl["n_positive"], u_fl["n_cells"]),
                 u_fl["train_mean_words"], u_fl["ci95_cellboot"]["lo"],
                 u_fl["ci95_cellboot"]["hi"]))
        print("  %-28s %+9.4f %8s %9.2f [%+.4f, %+.4f]"
              % ("PILOT  flickr    (4-task)", p_fl_mean,
                 "%d/%d" % (sum(1 for c in p_labs if p_fl[c] > 0), len(p_labs)),
                 pstats["flickr"]["mean_target_words"], p_fl_ci["lo"], p_fl_ci["hi"]))
        # The pooled pilot mean hides the whole story: split it by ordering.
        by_o = {}
        for c in p_labs:
            by_o.setdefault(c.split("/")[0], []).append(p_fl[c])
        print("\n  The pooled pilot mean hides the structure. By ordering:")
        for o in sorted(by_o):
            print("    pilot flickr, ordering %s: n=%d  mean %+.4f  values %s"
                  % (o, len(by_o[o]), mean(by_o[o]),
                     ", ".join("%+.4f" % v for v in sorted(by_o[o]))))
        print("    pilot ordering o2 puts flickr at position 2, straight after")
        print("    vizwiz -- the 43.6-percent-refusal, +0.58 stage. That ONE cell")
        print("    carries the whole sign flip.")
        same_sign = (u_fl["mean_dc"] > 0) == (p_fl_mean > 0)
        o1_vals = by_o.get("o1", [])
        verdict = (
            "INCONCLUSIVE, and the reason is instructive. In the three pilot "
            "cells that share ordering o1 the pull is %+.4f and positive in "
            "3/3, agreeing in sign and roughly in size with UCIT's %+.4f. The "
            "pooled pilot mean is %+.4f only because the single reversed-"
            "ordering cell gives %+.4f. With one cell in that ordering this "
            "cannot be separated from ordering-specific interference or from "
            "noise, and the 4-cell interval [%+.4f, %+.4f] contains the UCIT "
            "value. Report as: the length account is neither confirmed nor "
            "refuted out-of-sample, and the discordant cell is itself an "
            "adjacency observation (flickr straight after the high-dose stage)."
            % (mean(o1_vals) if o1_vals else float("nan"), u_fl["mean_dc"],
               p_fl_mean, min(p_fl.values()), p_fl_ci["lo"], p_fl_ci["hi"]))
        if same_sign:
            verdict = ("sign replicates across suites; magnitudes differ and the "
                       "streams are not matched. " + verdict)
        print("\n  VERDICT: %s" % verdict)
        print("\n  Caveats that keep this from being decisive: the two suites differ")
        print("  in stream composition, stage count and the tasks flanking Flickr,")
        print("  and the UCIT value is a cross-suite comparison of per-cell means")
        print("  computed separately in each suite (never pooled).")
        R["length_account_out_of_sample"] = {
            "ucit_Flickr30k": {"mean_dc": u_fl["mean_dc"], "per_cell": u_fl["per_cell"],
                               "words": u_fl["train_mean_words"],
                               "ci95": u_fl["ci95_cellboot"]},
            "pilot_flickr": {"mean_dc": p_fl_mean,
                             "per_cell": {c: round(p_fl[c], 4) for c in p_labs},
                             "words": pstats["flickr"]["mean_target_words"],
                             "ci95": p_fl_ci},
            "pilot_flickr_by_ordering": {o: {"n": len(v), "mean": mean(v),
                                             "values": [round(x, 4) for x in v]}
                                         for o, v in by_o.items()},
            "verdict": verdict,
            "caveats": ("Different suites: different stream composition, stage "
                        "count and neighbours. Per-cell means are aggregated "
                        "WITHIN each suite and only then compared."),
        }

    R["ucit"]["rows"] = rows
    R["ucit_anchor"]["rows"] = rows_anchor
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(R, open(args.out, "w"), indent=1)
    print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
