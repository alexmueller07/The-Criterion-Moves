#!/usr/bin/env python3
"""logit_bias_analysis.py -- additive-bias test + post-hoc criterion correction
from POPE answer-position logit dumps (pilot/eval_gen.py --dump_logits).

Theory under test (design_notes/method_calibration_theory.md, Claim 1): a CL
stage shifts the decision variable g = z_yes - z_no by a CLASS-INDEPENDENT
constant b_k. Greedy decoding thresholds g at 0, so such a shift moves the
criterion c one-for-one and leaves d' invariant; H and FA move together.

(a) ADDITIVE-BIAS TEST, per stage k vs base (and vs stage k-1):
      shift_yes = mean(g_k - g_0 | GT yes), shift_no = mean(g_k - g_0 | GT no)
      diff      = shift_yes - shift_no   (= change in class-mean separation)
                  with a within-class bootstrap CI;
      b_common  = pooled mean(g_k - g_0)  (the single offset the theory allows)
      explained: R2_item  = 1 - SS(delta - b)/SS(delta)      (item level)
                 nonadd_classmean = |diff| / (|b_common| + |diff|)
                 nonadd_z = |dd'| / (|b_z| + |dd'|)  (the note's z-space
                            decomposition, sanity check (A), now from gaps)
                 dc_undone = 1 - |c(g_k - b_common) - c_0| / |c_k - c_0|
      plus c / d' recomputed from the gaps at threshold 0 (clipped z, same
      primitives as fs_aggregate.pope_sdt) and, when the text-based
      pope_gen.jsonl cells are given, the text c / d' and the per-item
      agreement between (g > 0) and the parsed generated answer.
      THEORY HOLDS  -> diff ~ 0 (CI covers 0 or |diff| << |b_common|),
                       nonadd_classmean and nonadd_z small, dd' ~ 0, H and
                       FA co-move, class sds unchanged.
      THEORY FAILS  -> shift_yes and shift_no differ materially (separation
                       changes: a d' move, not a criterion move); sign of
                       diff says whether the stage sharpened (>0) or blurred
                       (<0) the classes. If nonadd_classmean is large but
                       nonadd_z is small, the class SPREAD changed (scale,
                       not location: see the sd columns) -- the equal-
                       variance location model [A1/A2] is what failed, not
                       d'-invariance.
      NOTE dc_undone is NOT a theory test: on balanced data c is the class
      midpoint, so the pooled offset restores it even when the separation
      changed (verified on synthetic data). It is the feasibility check for
      (b): it says the criterion is restorable by one scalar.
(b) POST-HOC CORRECTION (zero-training inference-time baseline): on a
    stratified 10% calibration split find the scalar b* such that
    c(g_k + b*) matches the base criterion on that split (bisection on the
    monotone step function), decide g_k + b* > 0 on the other 90%, and report
    c / d' / F1 corrected vs uncorrected vs base on that same 90%, with a
    paired bootstrap CI on dF1 and mean +- sd over several split seeds. A
    label-free variant (b_mean = mean-gap matching to base, no GT needed) is
    reported alongside. Every training-time method must beat this.
(c) Figure: gap histograms (base vs final stage, GT-yes / GT-no, class-mean
    shift arrows) + F1 bars (uncorrected vs corrected per stage, base line).

Outputs <out_prefix>.json, <out_prefix>.md, <out_prefix>_fig.{png,pdf}.

Inputs (either form):
  --logits_dir D --backbone B --runtag RT [--results_fs R]
      base   D/<B>_base/pope_logits.jsonl
      stages D/<RT>_k<K>/pope_logits.jsonl (K=1..; joint: ckpt_step<N>, final)
      text   R/<same cell>/pope_gen.jsonl  (optional check)
  --base PATH --stage LABEL=PATH ... [--text_base PATH --text_stage LABEL=PATH ...]

numpy + fs_common (same directory; ships to the cluster as code/analysis/);
matplotlib optional (figure skipped with a message). Python 3.9.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
try:
    from fs_common import (BASE_DIRS, _z, clipped_rate, import_pilot_scorers,  # noqa: E402
                           read_rows_dedup)
except ImportError as e:
    raise SystemExit(f"[logit_bias] fs_common.py must sit next to this script: {e}")

REQ = ("id", "gt", "gap", "z_yes", "z_no", "argmax_is")


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load_logits(path):
    audit = {}
    rows = read_rows_dedup(path, audit)
    out = {}
    for r in rows:
        for k in REQ:
            if k not in r:
                raise SystemExit(f"[logit_bias] {path}: row {r.get('id')!r} missing {k!r}")
        if r["gt"] not in ("yes", "no"):
            raise SystemExit(f"[logit_bias] {path}: row {r['id']!r} gt={r['gt']!r}")
        if not np.isfinite(r["gap"]):
            raise SystemExit(f"[logit_bias] {path}: row {r['id']!r} non-finite gap")
        out[r["id"]] = r
    audit["n_rows"] = len(rows)
    return out, audit


def discover(logits_dir, backbone, runtag):
    base = os.path.join(logits_dir, BASE_DIRS[backbone], "pope_logits.jsonl")
    stages = []
    k = 1
    while True:
        p = os.path.join(logits_dir, f"{runtag}_k{k}", "pope_logits.jsonl")
        if not os.path.isfile(p):
            break
        stages.append((f"{runtag}_k{k}", p))
        k += 1
    if not stages:  # joint arm: stage-boundary checkpoints, then final
        found = []
        for p in glob.glob(os.path.join(logits_dir, f"{runtag}_ckpt_step*", "pope_logits.jsonl")):
            m = re.search(r"_ckpt_step(\d+)$", os.path.basename(os.path.dirname(p)))
            if m:
                found.append((int(m.group(1)), os.path.basename(os.path.dirname(p)), p))
        stages = [(lab, p) for _, lab, p in sorted(found)]
        p = os.path.join(logits_dir, f"{runtag}_final", "pope_logits.jsonl")
        if os.path.isfile(p):
            stages.append((f"{runtag}_final", p))
    return base, stages


def parse_labeled(items, flag):
    out = []
    for s in items or []:
        if "=" not in s:
            raise SystemExit(f"[logit_bias] {flag} expects LABEL=PATH, got {s!r}")
        lab, p = s.split("=", 1)
        out.append((lab, p))
    return out


# ---------------------------------------------------------------------------
# SDT on gaps (threshold 0; same clipping/z primitives as fs_aggregate)
# ---------------------------------------------------------------------------
def sdt(gaps, yes, thr=0.0):
    pred = gaps > thr
    ny = int(yes.sum())
    nn = int((~yes).sum())
    if ny == 0 or nn == 0:
        raise SystemExit(f"[logit_bias] degenerate split (n_yes={ny}, n_no={nn})")
    hits = int((pred & yes).sum())
    fas = int((pred & ~yes).sum())
    Hc, hcl = clipped_rate(hits, ny)
    FAc, fcl = clipped_rate(fas, nn)
    zH, zFA = _z(Hc), _z(FAc)
    tp, fp, fn, tn = hits, fas, ny - hits, nn - fas
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {
        "H": hits / ny, "FA": fas / nn, "dprime": zH - zFA, "c": -0.5 * (zH + zFA),
        "yes_rate": float(pred.mean()), "accuracy": (tp + tn) / (ny + nn),
        "precision": prec, "recall": rec, "f1": 2 * prec * rec / max(1e-9, prec + rec),
        "n": ny + nn, "n_yes": ny, "n_no": nn,
        "clipped": ("H" if hcl else "") + ("FA" if fcl else "") or "none",
    }


def f1_of(pred, yes):
    tp = float((pred & yes).sum())
    fp = float((pred & ~yes).sum())
    fn = float((~pred & yes).sum())
    p = tp / max(1.0, tp + fp)
    r = tp / max(1.0, tp + fn)
    return 2 * p * r / max(1e-9, p + r)


def match_criterion(gaps, yes, c_target, iters=80):
    """Scalar b with c(gaps + b) closest to c_target. c(b) is a non-increasing
    step function of b (raising every gap raises H and FA), so bisection
    brackets the jump; return the bracket end with the smaller residual."""
    R = float(np.abs(gaps).max()) + 1.0
    lo, hi = -R, R

    def f(b):
        return sdt(gaps + b, yes)["c"] - c_target

    if f(lo) <= 0:
        return lo
    if f(hi) >= 0:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return lo if abs(f(lo)) <= abs(f(hi)) else hi


# ---------------------------------------------------------------------------
# (a) additive-bias test
# ---------------------------------------------------------------------------
def additive_test(g_ref, g_k, yes, n_boot, rng):
    delta = g_k - g_ref
    dy, dn = delta[yes], delta[~yes]
    shift_yes, shift_no = float(dy.mean()), float(dn.mean())
    diff = shift_yes - shift_no
    iy = rng.integers(0, len(dy), size=(n_boot, len(dy)))
    inn = rng.integers(0, len(dn), size=(n_boot, len(dn)))
    boots = dy[iy].mean(axis=1) - dn[inn].mean(axis=1)
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    b = float(delta.mean())
    ss_tot = float((delta ** 2).sum())
    r2_item = 1.0 - float(((delta - b) ** 2).sum()) / ss_tot if ss_tot > 0 else 1.0
    # item-level dispersion of delta that is class structure (0 under the theory)
    ss_within = float(((dy - shift_yes) ** 2).sum() + ((dn - shift_no) ** 2).sum())
    ss_around_mean = float(((delta - b) ** 2).sum())
    r2_class = 1.0 - ss_within / ss_around_mean if ss_around_mean > 0 else 0.0
    nonadd = abs(diff) / (abs(b) + abs(diff)) if (abs(b) + abs(diff)) > 0 else 0.0
    s_ref, s_k = sdt(g_ref, yes), sdt(g_k, yes)
    s_undone = sdt(g_k - b, yes)
    dc = s_k["c"] - s_ref["c"]
    dd = s_k["dprime"] - s_ref["dprime"]
    dc_undone = 1.0 - abs(s_undone["c"] - s_ref["c"]) / abs(dc) if abs(dc) > 1e-9 else None
    # the note's z-space decomposition, now computed from gaps
    zH0, zFA0 = _z(clipped_rate(int((g_ref[yes] > 0).sum()), int(yes.sum()))[0]), \
        _z(clipped_rate(int((g_ref[~yes] > 0).sum()), int((~yes).sum()))[0])
    zHk, zFAk = _z(clipped_rate(int((g_k[yes] > 0).sum()), int(yes.sum()))[0]), \
        _z(clipped_rate(int((g_k[~yes] > 0).sum()), int((~yes).sum()))[0])
    b_z = 0.5 * ((zHk - zH0) + (zFAk - zFA0))
    nonadd_z = abs(dd) / (abs(b_z) + abs(dd)) if (abs(b_z) + abs(dd)) > 0 else 0.0
    return {
        "shift_yes": shift_yes, "shift_no": shift_no, "diff": diff, "diff_ci95": ci,
        "diff_ci_covers_0": bool(ci[0] <= 0.0 <= ci[1]),
        "b_common": b, "r2_item": r2_item, "r2_class_of_delta": r2_class,
        "nonadd_classmean": nonadd,
        "sep_ref": float(g_ref[yes].mean() - g_ref[~yes].mean()),
        "sep_k": float(g_k[yes].mean() - g_k[~yes].mean()),
        "std_yes_ref": float(g_ref[yes].std()), "std_no_ref": float(g_ref[~yes].std()),
        "std_yes_k": float(g_k[yes].std()), "std_no_k": float(g_k[~yes].std()),
        "H_FA_comove": bool((zHk - zH0) * (zFAk - zFA0) > 0),
        "dc": dc, "dd": dd, "b_z": b_z, "nonadd_z": nonadd_z, "dc_undone": dc_undone,
        "c_after_common_offset": s_undone["c"], "dprime_after_common_offset": s_undone["dprime"],
    }


# ---------------------------------------------------------------------------
# (b) post-hoc correction
# ---------------------------------------------------------------------------
def correction(g0, gk, yes, calib_frac, n_splits, seed, n_boot):
    per_split = []
    for s in range(n_splits):
        rng = np.random.default_rng(seed + 1000 + s)
        calib = np.zeros(len(yes), dtype=bool)
        for cls in (True, False):
            idx = np.flatnonzero(yes == cls)
            rng.shuffle(idx)
            calib[idx[:max(1, int(round(calib_frac * len(idx))))]] = True
        test = ~calib
        c0_cal = sdt(g0[calib], yes[calib])["c"]
        b_star = match_criterion(gk[calib], yes[calib], c0_cal)
        b_mean = float(g0[calib].mean() - gk[calib].mean())     # label-free
        base_t = sdt(g0[test], yes[test])
        unc_t = sdt(gk[test], yes[test])
        cor_t = sdt(gk[test] + b_star, yes[test])
        mean_t = sdt(gk[test] + b_mean, yes[test])
        # paired bootstrap on dF1 (corrected - uncorrected), test split
        yt = yes[test]
        pu = gk[test] > 0
        pc = gk[test] + b_star > 0
        ii = rng.integers(0, len(yt), size=(n_boot, len(yt)))
        d = np.array([f1_of(pc[i], yt[i]) - f1_of(pu[i], yt[i]) for i in ii])
        per_split.append({
            "split_seed": seed + 1000 + s, "n_calib": int(calib.sum()), "n_test": int(test.sum()),
            "c0_calib": c0_cal, "b_star": b_star,
            "c_calib_after": sdt(gk[calib] + b_star, yes[calib])["c"],
            "b_mean_labelfree": b_mean,
            "base": base_t, "uncorrected": unc_t, "corrected": cor_t,
            "corrected_labelfree": mean_t,
            "df1_corr_minus_unc": cor_t["f1"] - unc_t["f1"],
            "df1_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        })

    def ms(key_fn):
        v = np.array([key_fn(p) for p in per_split], dtype=float)
        return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0}

    summary = {
        "b_star": ms(lambda p: p["b_star"]),
        "b_mean_labelfree": ms(lambda p: p["b_mean_labelfree"]),
        "f1_base": ms(lambda p: p["base"]["f1"]),
        "f1_uncorrected": ms(lambda p: p["uncorrected"]["f1"]),
        "f1_corrected": ms(lambda p: p["corrected"]["f1"]),
        "f1_corrected_labelfree": ms(lambda p: p["corrected_labelfree"]["f1"]),
        "c_uncorrected": ms(lambda p: p["uncorrected"]["c"]),
        "c_corrected": ms(lambda p: p["corrected"]["c"]),
        "c_base": ms(lambda p: p["base"]["c"]),
        "dprime_uncorrected": ms(lambda p: p["uncorrected"]["dprime"]),
        "dprime_corrected": ms(lambda p: p["corrected"]["dprime"]),
        "dprime_base": ms(lambda p: p["base"]["dprime"]),
        "acc_uncorrected": ms(lambda p: p["uncorrected"]["accuracy"]),
        "acc_corrected": ms(lambda p: p["corrected"]["accuracy"]),
        "acc_base": ms(lambda p: p["base"]["accuracy"]),
    }
    return {"splits": per_split, "summary": summary, "main_split": per_split[0]}


# ---------------------------------------------------------------------------
# text-based check
# ---------------------------------------------------------------------------
def text_check(path, ids, gaps, yes):
    parse_yn = import_pilot_scorers()["parse_yn"]
    audit = {}
    rows = read_rows_dedup(path, audit)
    pred = {}
    nfail = ntot = 0
    for r in rows:
        if "output" not in r or "gt" not in r:
            raise SystemExit(f"[logit_bias] {path}: POPE row missing output/gt")
        ntot += 1
        p = parse_yn(r["output"])
        if p is None:
            nfail += 1
            continue
        pred[r["id"]] = (p == "yes", r["gt"] == "yes")
    hits = fas = ny = nn = 0
    for _id, (py, gy) in pred.items():
        if gy:
            ny += 1
            hits += py
        else:
            nn += 1
            fas += py
    if ny == 0 or nn == 0:
        raise SystemExit(f"[logit_bias] {path}: degenerate text POPE")
    zH, zFA = _z(clipped_rate(hits, ny)[0]), _z(clipped_rate(fas, nn)[0])
    # per-item agreement on the joined + parsed items
    agree = tot = 0
    for i, _id in enumerate(ids):
        if _id in pred:
            tot += 1
            agree += (pred[_id][0] == bool(gaps[i] > 0))
    return {
        "path": path, "n_total": ntot, "parse_fail": nfail, "parse_fail_rate": nfail / max(1, ntot),
        "H": hits / ny, "FA": fas / nn, "dprime": zH - zFA, "c": -0.5 * (zH + zFA),
        "sign_agreement": agree / max(1, tot), "n_compared": tot,
    }


# ---------------------------------------------------------------------------
# (c) figure
# ---------------------------------------------------------------------------
def make_figure(out_prefix, ids, g0, gk_final, yes, labels, corr, base_f1, final_label):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[logit_bias] matplotlib not available; figure skipped", flush=True)
        return None
    try:
        import fig_style
        fig_style.apply_style()
        W = fig_style.FULL_W
        save = lambda fig, name: fig_style.save_fig(fig, os.path.dirname(name) or ".",
                                                    os.path.basename(name), [], "logit_bias")
    except ImportError:
        W = 6.08
        save = None
    C_YES, C_NO, C_BASE = "#009E73", "#D55E00", "#999999"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, 2.3),
                                   gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.32})
    lo = float(min(g0.min(), gk_final.min()))
    hi = float(max(g0.max(), gk_final.max()))
    bins = np.linspace(lo, hi, 61)
    short_final = re.sub(r"^.*_(k\d+|ckpt_step\d+|final)$", r"\1", final_label)
    for cls, col, name in ((True, C_YES, "GT yes"), (False, C_NO, "GT no")):
        m = yes == cls
        ax1.hist(g0[m], bins=bins, color=col, alpha=0.28, density=True,
                 label=f"base, {name}")
        ax1.hist(gk_final[m], bins=bins, histtype="step", color=col, linewidth=1.1,
                 density=True, label=f"{short_final}, {name}")
    ymax = ax1.get_ylim()[1]
    # class-mean shift arrows in a dedicated band above the histograms
    ax1.set_ylim(0, ymax * 1.42)
    for j, (cls, col) in enumerate(((True, C_YES), (False, C_NO))):
        m = yes == cls
        y = ymax * (1.30 - 0.13 * j)
        x0, x1 = float(g0[m].mean()), float(gk_final[m].mean())
        ax1.annotate("", xy=(x1, y), xytext=(x0, y),
                     arrowprops=dict(arrowstyle="->", color=col, lw=1.2, shrinkA=0, shrinkB=0))
        ax1.text(max(x0, x1), y, f" {x1 - x0:+.2f}", color=col, va="center",
                 ha="left", fontsize=6.5)
    ax1.axvline(0, color="black", linewidth=0.7, linestyle=":")
    ax1.set_xlabel(r"$g = z_{yes} - z_{no}$ (answer position)")
    ax1.set_ylabel("density")
    ax1.set_title("gap distributions: base vs final stage", fontsize=7.5)
    ax1.legend(fontsize=5.6, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2)

    x = np.arange(len(labels))
    f_unc = [corr[l]["summary"]["f1_uncorrected"]["mean"] for l in labels]
    f_cor = [corr[l]["summary"]["f1_corrected"]["mean"] for l in labels]
    e_cor = [corr[l]["summary"]["f1_corrected"]["sd"] for l in labels]
    e_unc = [corr[l]["summary"]["f1_uncorrected"]["sd"] for l in labels]
    ax2.bar(x - 0.19, f_unc, 0.36, yerr=e_unc, color="#0072B2", edgecolor="black",
            linewidth=0.5, label="uncorrected", error_kw=dict(lw=0.6, capsize=1.5))
    ax2.bar(x + 0.19, f_cor, 0.36, yerr=e_cor, color="#E69F00", edgecolor="black",
            linewidth=0.5, label="+ scalar $b^*$ (10% calib)", error_kw=dict(lw=0.6, capsize=1.5))
    ax2.axhline(base_f1, color=C_BASE, linewidth=1.0, linestyle="--", label="base")
    short = [re.sub(r"^.*_(k\d+|ckpt_step\d+|final)$", r"\1", l) for l in labels]
    ax2.set_xticks(x)
    ax2.set_xticklabels(short, fontsize=6.5)
    ax2.set_ylabel("POPE F1 (90% test split)")
    allv = f_unc + f_cor + [base_f1]
    ax2.set_ylim(max(0.0, min(allv) - 0.05), min(1.0, max(allv) + 0.03))
    ax2.set_title("post-hoc criterion correction", fontsize=7.5)
    ax2.legend(fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
    name = out_prefix + "_fig"
    if save is not None:
        save(fig, name)
    else:
        for ext in ("png", "pdf"):
            fig.savefig(name + "." + ext, bbox_inches="tight", pad_inches=0.02, dpi=300)
        plt.close(fig)
    return name


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------
def f3(v, spec="+.3f"):
    return "n/a" if v is None else format(v, spec)


def write_md(path, res):
    L = []
    L.append(f"# Logit-level additive-bias test: {res['runtag']}")
    L.append("")
    L.append(f"Base: `{res['base_path']}`; stages: {len(res['stages'])}; joined ids: "
             f"{res['join']['n_joined']} (base rows {res['join']['n_base']}; per-file drops: "
             f"{res['join']['dropped_per_file']}). Bootstrap n={res['n_boot']}, seed={res['seed']}.")
    L.append("")
    L.append("Theory (Claim 1): a stage shifts g by one class-independent constant -> "
             "`diff` ~ 0 (CI covers 0 or |diff| << |b_common|), `nonadd(class)` and "
             "`nonadd(z)` small, `dd'` ~ 0, H and FA co-move, class sds unchanged. "
             "Failure: `shift_yes` != `shift_no` (separation changes = a d' move). "
             "`nonadd(class)` large with `nonadd(z)` small means the class SPREAD "
             "changed (scale, not location; see the sd table): the equal-variance "
             "location model failed, not d'-invariance. `dc undone` is the "
             "feasibility check for (b), not a theory test: on balanced data the pooled "
             "offset restores the midpoint c even when the separation changed.")
    L.append("")
    L.append("## (a) Additive-bias test vs base (gap units = logits)")
    L.append("")
    L.append("| stage | shift_yes | shift_no | diff [95% CI] | b_common | R2_item | nonadd(class) | dc undone | c_gap | dc | d'_gap | dd' | nonadd(z) | H,FA co-move |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    b = res["base_sdt"]
    L.append(f"| base | - | - | - | - | - | - | - | {b['c']:+.3f} | - | {b['dprime']:.3f} | - | - | - |")
    for st in res["stages"]:
        a = st["additive_vs_base"]
        L.append(f"| {st['label']} | {a['shift_yes']:+.3f} | {a['shift_no']:+.3f} | "
                 f"{a['diff']:+.3f} [{a['diff_ci95'][0]:+.3f}, {a['diff_ci95'][1]:+.3f}] | "
                 f"{a['b_common']:+.3f} | {a['r2_item']:.2f} | {a['nonadd_classmean']:.1%} | "
                 f"{f3(a['dc_undone'], '.0%')} | {st['sdt']['c']:+.3f} | {a['dc']:+.3f} | "
                 f"{st['sdt']['dprime']:.3f} | {a['dd']:+.3f} | {a['nonadd_z']:.1%} | "
                 f"{'yes' if a['H_FA_comove'] else 'NO'} |")
    L.append("")
    L.append("Stage-to-stage (k-1 -> k):")
    L.append("")
    L.append("| transition | shift_yes | shift_no | diff [95% CI] | b_common | nonadd(class) | dc undone | dc | dd' |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    prev = "base"
    for st in res["stages"]:
        a = st["additive_vs_prev"]
        L.append(f"| {prev} -> {st['label']} | {a['shift_yes']:+.3f} | {a['shift_no']:+.3f} | "
                 f"{a['diff']:+.3f} [{a['diff_ci95'][0]:+.3f}, {a['diff_ci95'][1]:+.3f}] | "
                 f"{a['b_common']:+.3f} | {a['nonadd_classmean']:.1%} | {f3(a['dc_undone'], '.0%')} | "
                 f"{a['dc']:+.3f} | {a['dd']:+.3f} |")
        prev = st["label"]
    L.append("")
    L.append("Class-conditional spread (equal-variance check [A2]) and the logit-level "
             "'parse failure' (argmax is neither ' Yes' nor ' No'):")
    L.append("")
    L.append("| stage | mean g|yes | mean g|no | sd g|yes | sd g|no | yes-rate(g>0) | argmax other | (g>0)==argmax |")
    L.append("|---|---|---|---|---|---|---|---|")
    for st in [{"label": "base", "sdt": res["base_sdt"], "spread": res["base_spread"]}] + res["stages"]:
        s = st["spread"]
        L.append(f"| {st['label']} | {s['mean_yes']:+.3f} | {s['mean_no']:+.3f} | {s['sd_yes']:.3f} | "
                 f"{s['sd_no']:.3f} | {st['sdt']['yes_rate']:.3f} | {s['argmax_other_rate']:.2%} | "
                 f"{s['gap_argmax_agreement']:.4f} |")
    if any(st.get("text") for st in res["stages"]) or res.get("base_text"):
        L.append("")
        L.append("Text-based check (pope_gen.jsonl, parse_yn; generation used left padding, "
                 "the dump right padding -- agreement < ~0.99 means the instrument and the "
                 "generated answer disagree beyond bf16 noise):")
        L.append("")
        L.append("| stage | c_gap | c_text | d'_gap | d'_text | text parse-fail | sign agreement (n) |")
        L.append("|---|---|---|---|---|---|---|")
        for st in [{"label": "base", "sdt": res["base_sdt"], "text": res.get("base_text")}] + res["stages"]:
            t = st.get("text")
            if not t:
                L.append(f"| {st['label']} | {st['sdt']['c']:+.3f} | ABSENT | {st['sdt']['dprime']:.3f} | ABSENT | - | - |")
                continue
            L.append(f"| {st['label']} | {st['sdt']['c']:+.3f} | {t['c']:+.3f} | {st['sdt']['dprime']:.3f} | "
                     f"{t['dprime']:.3f} | {t['parse_fail_rate']:.2%} | {t['sign_agreement']:.4f} ({t['n_compared']}) |")
    L.append("")
    L.append(f"## (b) Post-hoc scalar correction ({res['calib_frac']:.0%} stratified calibration split, "
             f"{res['n_splits']} split seeds; metrics on the other {1 - res['calib_frac']:.0%}; mean +- sd over splits)")
    L.append("")
    L.append("| stage | b* | F1 base | F1 uncorr | F1 corr | dF1 [95% CI, split 0] | F1 corr (label-free b_mean) | c uncorr | c corr | c base | d' uncorr | d' corr | acc uncorr | acc corr |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for st in res["stages"]:
        c = st["correction"]
        S = c["summary"]
        m = c["main_split"]
        L.append(f"| {st['label']} | {S['b_star']['mean']:+.3f}+-{S['b_star']['sd']:.3f} | "
                 f"{S['f1_base']['mean']:.4f} | {S['f1_uncorrected']['mean']:.4f}+-{S['f1_uncorrected']['sd']:.4f} | "
                 f"{S['f1_corrected']['mean']:.4f}+-{S['f1_corrected']['sd']:.4f} | "
                 f"{m['df1_corr_minus_unc']:+.4f} [{m['df1_ci95'][0]:+.4f}, {m['df1_ci95'][1]:+.4f}] | "
                 f"{S['f1_corrected_labelfree']['mean']:.4f} | {S['c_uncorrected']['mean']:+.3f} | "
                 f"{S['c_corrected']['mean']:+.3f} | {S['c_base']['mean']:+.3f} | "
                 f"{S['dprime_uncorrected']['mean']:.3f} | {S['dprime_corrected']['mean']:.3f} | "
                 f"{S['acc_uncorrected']['mean']:.4f} | {S['acc_corrected']['mean']:.4f} |")
    L.append("")
    L.append("Reading: a training-time method that fixes criterion drift must beat "
             "`F1 corr` (one scalar, 10% labeled calibration, zero training). If `F1 corr` "
             "~ `F1 base` the drift is fully bias-removable at inference time; the gap "
             "`F1 base - F1 corr` is the non-additive (d') damage no scalar can undo.")
    if res.get("figure"):
        L.append("")
        L.append(f"Figure: `{res['figure']}.png`")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logits_dir", default=None)
    ap.add_argument("--backbone", default=None, choices=sorted(BASE_DIRS))
    ap.add_argument("--runtag", default=None)
    ap.add_argument("--results_fs", default=None, help="text cells dir for the check")
    ap.add_argument("--base", default=None, help="base pope_logits.jsonl")
    ap.add_argument("--stage", action="append", default=None, help="LABEL=PATH (repeat, in order)")
    ap.add_argument("--text_base", default=None)
    ap.add_argument("--text_stage", action="append", default=None, help="LABEL=PATH")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib_frac", type=float, default=0.10)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--no_fig", action="store_true")
    args = ap.parse_args()

    text_stage = dict(parse_labeled(args.text_stage, "--text_stage"))
    text_base = args.text_base
    if args.base:
        base_path = args.base
        stages = parse_labeled(args.stage, "--stage")
        runtag = args.runtag or "custom"
    else:
        if not (args.logits_dir and args.backbone and args.runtag):
            raise SystemExit("[logit_bias] give --base/--stage or --logits_dir/--backbone/--runtag")
        base_path, stages = discover(args.logits_dir, args.backbone, args.runtag)
        runtag = args.runtag
        if args.results_fs:
            p = os.path.join(args.results_fs, BASE_DIRS[args.backbone], "pope_gen.jsonl")
            text_base = text_base or (p if os.path.isfile(p) else None)
            for lab, _ in stages:
                p = os.path.join(args.results_fs, lab, "pope_gen.jsonl")
                if lab not in text_stage and os.path.isfile(p):
                    text_stage[lab] = p
    if not os.path.isfile(base_path):
        raise SystemExit(f"[logit_bias] base logits missing: {base_path}")
    if not stages:
        raise SystemExit("[logit_bias] no stage logits found")
    for lab, p in stages:
        if not os.path.isfile(p):
            raise SystemExit(f"[logit_bias] stage {lab} logits missing: {p}")

    base_rows, base_audit = load_logits(base_path)
    stage_rows = {}
    audits = {"base": base_audit}
    for lab, p in stages:
        stage_rows[lab], audits[lab] = load_logits(p)
    ids = sorted(base_rows)
    common = set(ids)
    for lab, _ in stages:
        common &= set(stage_rows[lab])
    dropped = {"base": len(base_rows) - len(common)}
    for lab, _ in stages:
        dropped[lab] = len(stage_rows[lab]) - len(common)
    ids = [i for i in ids if i in common]
    if len(ids) < 0.5 * len(base_rows):
        raise SystemExit(f"[logit_bias] only {len(ids)}/{len(base_rows)} ids joined across files")
    for lab, _ in stages:
        for i in ids:
            if stage_rows[lab][i]["gt"] != base_rows[i]["gt"]:
                raise SystemExit(f"[logit_bias] gt mismatch for {i} in {lab}")
    yes = np.array([base_rows[i]["gt"] == "yes" for i in ids])
    cats = np.array([base_rows[i].get("category", "unknown") for i in ids])
    g0 = np.array([float(base_rows[i]["gap"]) for i in ids])
    G = {lab: np.array([float(stage_rows[lab][i]["gap"]) for i in ids]) for lab, _ in stages}
    A = {"base": [base_rows[i]["argmax_is"] for i in ids]}
    for lab, _ in stages:
        A[lab] = [stage_rows[lab][i]["argmax_is"] for i in ids]

    def spread(g, arg):
        arg = np.array(arg)
        known = arg != "other"
        return {"mean_yes": float(g[yes].mean()), "mean_no": float(g[~yes].mean()),
                "sd_yes": float(g[yes].std()), "sd_no": float(g[~yes].std()),
                "argmax_other_rate": float((~known).mean()),
                "gap_argmax_agreement": float(((g > 0) == (arg == "yes"))[known].mean()) if known.any() else None}

    def per_cat(g):
        out = {}
        for c in sorted(set(cats)):
            m = cats == c
            out[c] = sdt(g[m], yes[m])
        return out

    rng = np.random.default_rng(args.seed)
    res = {
        "runtag": runtag, "base_path": base_path, "n_boot": args.n_boot, "seed": args.seed,
        "calib_frac": args.calib_frac, "n_splits": args.n_splits,
        "join": {"n_base": len(base_rows), "n_joined": len(ids), "dropped_per_file": dropped,
                 "file_audits": audits},
        "base_sdt": sdt(g0, yes), "base_spread": spread(g0, A["base"]),
        "base_per_category": per_cat(g0),
        "stages": [],
    }
    if text_base:
        res["base_text"] = text_check(text_base, ids, g0, yes)
    prev_g = g0
    for lab, p in stages:
        gk = G[lab]
        st = {"label": lab, "path": p, "sdt": sdt(gk, yes), "spread": spread(gk, A[lab]),
              "per_category": per_cat(gk),
              "additive_vs_base": additive_test(g0, gk, yes, args.n_boot, rng),
              "additive_vs_prev": additive_test(prev_g, gk, yes, args.n_boot, rng),
              "correction": correction(g0, gk, yes, args.calib_frac, args.n_splits,
                                       args.seed, min(1000, args.n_boot))}
        if lab in text_stage:
            st["text"] = text_check(text_stage[lab], ids, gk, yes)
        res["stages"].append(st)
        prev_g = gk
        a = st["additive_vs_base"]
        print(f"[logit_bias] {lab}: shift_yes={a['shift_yes']:+.3f} shift_no={a['shift_no']:+.3f} "
              f"diff={a['diff']:+.3f} CI=[{a['diff_ci95'][0]:+.3f},{a['diff_ci95'][1]:+.3f}] "
              f"nonadd={a['nonadd_classmean']:.1%} dc={a['dc']:+.3f} dd'={a['dd']:+.3f} "
              f"dc_undone={f3(a['dc_undone'], '.0%')} | F1 unc={st['correction']['summary']['f1_uncorrected']['mean']:.4f} "
              f"corr={st['correction']['summary']['f1_corrected']['mean']:.4f} "
              f"base={st['correction']['summary']['f1_base']['mean']:.4f}", flush=True)

    # verdict flags (soft; the numbers are the result)
    res["flags"] = {
        st["label"]: {
            "additive_dominant": st["additive_vs_base"]["nonadd_classmean"] < 0.25,
            "diff_ci_covers_0": st["additive_vs_base"]["diff_ci_covers_0"],
            "correction_recovers_base_f1": (st["correction"]["summary"]["f1_base"]["mean"]
                                            - st["correction"]["summary"]["f1_corrected"]["mean"]) < 0.01,
        } for st in res["stages"]}

    out_dir = os.path.dirname(os.path.abspath(args.out_prefix))
    os.makedirs(out_dir, exist_ok=True)
    if not args.no_fig:
        labels = [lab for lab, _ in stages]
        corr = {st["label"]: st["correction"] for st in res["stages"]}
        base_f1 = float(np.mean([st["correction"]["summary"]["f1_base"]["mean"] for st in res["stages"]]))
        res["figure"] = make_figure(args.out_prefix, ids, g0, G[labels[-1]], yes, labels, corr,
                                    base_f1, labels[-1])
    with open(args.out_prefix + ".json", "w") as f:
        json.dump(res, f, indent=1)
    write_md(args.out_prefix + ".md", res)
    print(f"[logit_bias] wrote {args.out_prefix}.json / .md"
          + (f" / {res['figure']}.png" if res.get("figure") else ""), flush=True)


if __name__ == "__main__":
    main()
