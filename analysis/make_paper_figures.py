"""Reframed-paper figures for the CL x hallucination pilot.

USAGE:  python analysis/make_paper_figures.py --diag analysis/diag \
                                              --results analysis/out \
                                              --out analysis/out

Builds the two figures the REFRAMED paper leads with (paper/main.tex:
"Same Checkpoints, Opposite Verdicts ..."):

  fig1_paradox             The reviewer's-5-second figure. TOP row (the
                           paradox): on the SAME sequential checkpoints
                           (x = training stage) POPE-adversarial F1 and raw
                           CHAIR_i both end BELOW base after the sequence --
                           F1 down reads as MORE hallucination, CHAIR down
                           reads as LESS hallucination: two field-standard
                           instruments, opposite verdicts on identical model
                           states. BOTTOM row (the resolution): the
                           signal-detection split -- the answer criterion c
                           moves about 4x as much as discrimination d' does,
                           with BOTH summarised by the SAME two statistics
                           (within-arm range and endpoint displacement) and
                           drawn on a MATCHED y-span, so the comparison is
                           like-for-like and not a zoom. A shared S4 (VizWiz)
                           highlight ties all four panels together: the
                           endpoint drop in both metrics sits exactly where
                           the criterion spikes conservative -- one cause,
                           both "results".

                           THE CLAIM IS RELATIVE, NEVER ABSOLUTE (2026-09-11).
                           "d' is flat" / "d' does not move" is RETRACTED: on
                           the full study the sequential endpoint d' falls
                           0.121 with SE 0.0108 over nine cells, ~11 SE from
                           zero. What survives, and is stronger, is (i) the
                           relative statement above and (ii) a TOST
                           equivalence result -- the change in d' is
                           statistically equivalent to zero within the
                           pre-registered +/-0.30 margin, which the old
                           falsifier could only ever fail to trip. An earlier
                           version of this figure printed "criterion swings
                           0.78" on panel (c) against "d' flat (endpoint
                           +0.04)" on panel (d): a RANGE against an ENDPOINT
                           DELTA, reading as 19x where like-for-like it is 4x.
                           Both panels now carry both statistics.

  fig_method_vs_baselines  What the anchor does AND what it costs -- both
                           halves, in one view. (a) per-stage criterion drift
                           |Delta c| for SEQ, ER (replay -- the key baseline)
                           and the anchor. ER tracks (indeed slightly exceeds)
                           SEQ at every stage -- the ER null: the standard
                           forgetting remedy does NOT touch the criterion --
                           while the anchor damps it after a one-time settling
                           step. (b) THE COST, which may never be omitted:
                           mean |c|, i.e. how far the criterion sits from the
                           optimum. POPE is exactly balanced (n_gt_yes =
                           n_gt_no), so c = 0 is derivable, not a preference,
                           and the anchor holds the criterion ~2.5x further
                           from it than SEQ does. It buys stability by parking
                           a mis-placed threshold. (c) endpoint
                           length-controlled CHAIR_i@60 (SEQ/ER/anchor/JOINT),
                           shown ONLY as the pilot observation that the full
                           study then tested and did not reproduce.

                           THE ANCHOR IS NOT A METHOD (2026-09-09..11). It is
                           a causal probe that criterion drift is
                           controllable, and simultaneously a demonstration
                           that controlling it toward the frozen base is
                           harmful. Its endpoint POPE F1 is 0.7963 against
                           SEQ's 0.8625 on the full study (9/9 cells, no
                           overlap -- worse than running no method at all),
                           and the z-ROC coherence fit shows it does not
                           merely mis-place the threshold but distorts the
                           evidence geometry (R^2 0.586 vs SEQ's 0.957; d_a
                           1.788 vs 2.159). The project's standing rule is
                           that the Sigma|Delta c| win may not be reported
                           without this cost in the same view -- hence panel
                           (b). The pilot CHAIR_i@60 reduction is likewise
                           WITHDRAWN as a claim: the full study posts 0.0983
                           for the anchor against 0.0977 for SEQ.

WHY fig1_paradox SUPERSEDES fig_motivation for the reframe
----------------------------------------------------------
make_fig_motivation.py (fig_motivation) leads with a hand-drawn signal-detection
SCHEMATIC and shows the criterion/d' data only for SEQ vs the anchor. That was
the motivation figure for the METHOD-first framing. The reframed paper leads
with the *measurement paradox* -- two standard metrics disagreeing on the same
checkpoints -- so the opening figure must show the two metrics FIRST and the SDT
decomposition as the resolution, from real data end to end (no schematic).
fig1_paradox is that figure. fig_motivation is NOT deleted (make_fig_motivation.py
is untouched); if the reframe is adopted, Figure 1 should point at fig1_paradox
and fig_motivation retires to the method section or the appendix.

DATA PROVENANCE (all REAL committed pilot data; nothing hardcoded/fabricated)
----------------------------------------------------------------------------
  analysis/diag/method_comparison.json   c, d', chair_i, yes_rate per checkpoint
                                         (the task's named source of truth for
                                         the SEQ/ER/anchor trajectories). Every
                                         number taken from it is re-read and
                                         asserted against the value the paper
                                         quotes; a mismatch ABORTS (fail-loud)
                                         rather than letting a figure drift.
  analysis/out/results_full.csv          pope_adversarial_f1 per checkpoint.
                                         (method_comparison.json carries pooled
                                         H/FA/d'/c but NOT per-split F1, so the
                                         adversarial-F1 trajectory -- the paper's
                                         pre-registered gate quantity -- is read
                                         from the committed results CSV and its
                                         S0..S4 values asserted.)
  analysis/diag/length_controlled_chair.json  CHAIR_i@60 endpoint values.
  analysis/diag/g4_bootstrap_cis.json    the two anchor-endpoint paired-bootstrap
                                         CIs quoted in the paper; each point gap
                                         is recomputed from the @60 data and
                                         asserted to match.

matplotlib only; colorblind-safe Okabe-Ito palette (analysis/fig_style.py);
Python 3.9 compatible.
"""
import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch  # noqa: F401 (kept for readers)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_style as FS

FS.apply_style()

# ---- checkpoint sequences (S0 = shared untuned base) -----------------------
SEQ_CKPTS = ["S0", "S1", "S2", "S3", "S4"]
E_CKPTS = ["S0", "E1", "E2", "E3", "E4"]
G_CKPTS = ["S0", "G1", "G2", "G3", "G4"]
J_CKPTS = ["S0", "J1", "J2", "J3", "J4"]
# Task trained at each SEQ stage (ScienceQA -> TextVQA -> Flickr30k -> VizWiz).
STAGE_TASKS = ["base", "SciQA", "TextVQA", "Flickr", "VizWiz"]

# ---- semantic verdict colors (NOT arm identities: this figure has no green/
#      red LINES, so soft red/green annotation boxes cannot be confused with an
#      arm; they carry the "reads as worse / reads as better" punch) ----------
WORSE_BG = "#F6D6D1"
WORSE_FG = "#A03123"
BETTER_BG = "#CFE8DC"
BETTER_FG = "#1B6E4F"
HILITE = "#FBE9A7"   # shared S4 (VizWiz) highlight band


# ------------------------------------------------------------- strict loaders
def die(msg):
    raise SystemExit("[make_paper_figures] ERROR: %s" % msg)


def load_json(path):
    if not os.path.isfile(path):
        die("required data file not found: %s" % path)
    with open(path) as f:
        try:
            return json.load(f)
        except ValueError as e:
            die("could not parse %s as JSON: %s" % (path, e))


def jnum(obj, keys, fname):
    """Strict nested numeric lookup; missing/non-numeric key aborts loudly."""
    cur = obj
    for i, k in enumerate(keys):
        if not isinstance(cur, dict) or k not in cur:
            die("missing key '%s' in %s (no silent defaults; refusing to "
                "fabricate a value)"
                % (" -> ".join(str(x) for x in keys[:i + 1]), fname))
        cur = cur[k]
    try:
        return float(cur)
    except (TypeError, ValueError):
        die("key '%s' in %s is not a number (got %r)"
            % (" -> ".join(str(x) for x in keys), fname, cur))


def mc_series(mcj, ckpts, col):
    return [jnum(mcj, ["rows", c, col], "method_comparison.json") for c in ckpts]


def load_adv_f1(results_csv):
    """pope_adversarial_f1 per checkpoint from the committed results CSV.
    Returns {ckpt: value}. Strict: aborts if the column/metric is absent."""
    if not os.path.isfile(results_csv):
        die("required results CSV not found: %s" % results_csv)
    out = {}
    with open(results_csv, newline="") as f:
        rdr = csv.DictReader(f)
        need = {"ckpt", "source", "metric", "value"}
        if not need.issubset(set(rdr.fieldnames or [])):
            die("results CSV %s missing columns %s (has %s)"
                % (results_csv, sorted(need), rdr.fieldnames))
        for row in rdr:
            if row["source"] == "pope" and row["metric"] == "pope_adversarial_f1":
                try:
                    out[row["ckpt"]] = float(row["value"])
                except (TypeError, ValueError):
                    die("non-numeric pope_adversarial_f1 for %s in %s (got %r)"
                        % (row["ckpt"], results_csv, row["value"]))
    return out


def at60(lcc, ckpt):
    return jnum(lcc, ["checkpoints", ckpt, "at_k", "60", "chair_i"],
                "length_controlled_chair.json")


# ---------------------------------------------------------- assertion helpers
def approx(a, b, tol):
    return abs(a - b) <= tol


# =========================================================================
# FIGURE 1 -- the paradox + its resolution
# =========================================================================
def build_fig1(mcj, adv_f1, out_dir, rendered):
    fname_mc = "method_comparison.json"

    # ---- data (all from committed files) -----------------------------------
    seq_c = mc_series(mcj, SEQ_CKPTS, "c")
    seq_d = mc_series(mcj, SEQ_CKPTS, "dprime")
    seq_chair = mc_series(mcj, SEQ_CKPTS, "chair_i")
    seq_f1 = [adv_f1.get(c) for c in SEQ_CKPTS]
    if any(v is None for v in seq_f1):
        die("pope_adversarial_f1 missing for one of %s in results_full.csv"
            % SEQ_CKPTS)

    # ---- magnitudes computed from the data (never hardcoded) ---------------
    # ESTIMATOR DISCIPLINE.  Panels (c) and (d) report the SAME TWO statistics
    # as each other -- a within-arm RANGE and an ENDPOINT DISPLACEMENT -- so no
    # comparison a reader can make across the two panels pairs a range on one
    # against an endpoint delta on the other.  (An earlier version printed
    # "criterion swings 0.78" on (c) against "endpoint +0.04" on (d), which
    # reads as 19x; like-for-like it is 0.78 / 0.19 = 4x.)
    c_swing = max(seq_c) - min(seq_c)
    c_move = seq_c[-1] - seq_c[0]
    d_range = max(seq_d) - min(seq_d)
    d_move = seq_d[-1] - seq_d[0]
    range_ratio = c_swing / d_range
    f1_drop = seq_f1[0] - seq_f1[-1]          # net over the sequence (S0 -> S4)
    chair_drop = seq_chair[0] - seq_chair[-1]

    # Endpoint SEQ-vs-JOINT gaps that the paper's paradox quotes (+0.048/-0.029).
    j4_f1 = adv_f1.get("J4")
    j4_chair = jnum(mcj, ["rows", "J4", "chair_i"], fname_mc)
    if j4_f1 is None:
        die("pope_adversarial_f1 missing for J4 in results_full.csv")
    gap_f1 = j4_f1 - seq_f1[-1]               # J4 - S4  (paper: +0.048)
    gap_chair = seq_chair[-1] - j4_chair      # S4 - J4  (paper: -0.029)

    # ---- fail-loud: the committed data must still match the paper -----------
    if not approx(c_swing, 0.778, 5e-3):
        die("SEQ criterion swing recomputed as %.4f; paper quotes 0.78 "
            "(0.744 at S4 minus -0.034 at S2). Data/caption drift -- refusing "
            "to render." % c_swing)
    if not approx(d_range, 0.193, 5e-3):
        die("SEQ d' range recomputed as %.4f; paper quotes 0.19. Drift -- "
            "refusing to render." % d_range)
    if not approx(d_move, 0.042, 5e-3):
        die("SEQ d' endpoint displacement recomputed as %.4f; paper quotes "
            "+0.04. Drift -- refusing to render." % d_move)
    if not approx(c_move, 0.313, 5e-3):
        die("SEQ criterion endpoint displacement recomputed as %.4f; paper "
            "quotes +0.31. Drift -- refusing to render." % c_move)
    # F1 trajectory: the pre-registered gate quantity; assert every SEQ value.
    exp_f1 = {"S0": 0.8272, "S1": 0.8347, "S2": 0.8312, "S3": 0.8392,
              "S4": 0.7819}
    for c in SEQ_CKPTS:
        if not approx(adv_f1[c], exp_f1[c], 1e-3):
            die("pope_adversarial_f1[%s]=%.4f but paper/data expect %.4f -- "
                "results_full.csv drifted; refusing to render."
                % (c, adv_f1[c], exp_f1[c]))
    # The two verdicts must actually be OPPOSITE, or the figure lies.
    if not (f1_drop > 0 and chair_drop > 0):
        die("Both metrics must END BELOW base for the opposite-verdicts story "
            "(F1 down => 'worse', CHAIR down => 'better'); got f1_drop=%.4f "
            "chair_drop=%.4f." % (f1_drop, chair_drop))
    if not (gap_f1 > 0 and gap_chair < 0):
        die("SEQ-vs-JOINT endpoint gaps must be opposite-signed (F1 gap>0 -> "
            "SEQ looks worse; CHAIR gap<0 -> SEQ looks better); got "
            "gap_f1=%+.4f gap_chair=%+.4f." % (gap_f1, gap_chair))
    if not approx(gap_f1, 0.048, 6e-3):
        die("J4-S4 adversarial-F1 gap recomputed %+.4f; paper quotes +0.048."
            % gap_f1)
    if not approx(gap_chair, -0.029, 6e-3):
        die("S4-J4 raw-CHAIR gap recomputed %+.4f; paper quotes -0.029."
            % gap_chair)

    xs = list(range(5))

    # ---- layout: two rows, each 1x2 ------------------------------------------
    # No titles, banners or annotation prose on the canvas: every number and
    # every reading of the figure is stated in the caption (printed below).
    fig = plt.figure(figsize=(FS.FULL_W, FS.h_full(4.0)))
    gs_top = fig.add_gridspec(1, 2, left=0.085, right=0.975, top=0.935,
                              bottom=0.635, wspace=0.28)
    gs_bot = fig.add_gridspec(1, 2, left=0.085, right=0.975, top=0.475,
                              bottom=0.130, wspace=0.28)
    ax_f1 = fig.add_subplot(gs_top[0, 0])
    ax_ch = fig.add_subplot(gs_top[0, 1])
    ax_c = fig.add_subplot(gs_bot[0, 0])
    ax_d = fig.add_subplot(gs_bot[0, 1])

    def stage_axis(ax, top=False):
        ax.set_xlim(-0.28, 4.30)
        ax.set_xticks(xs)
        if top:
            ax.set_xticklabels([str(i) for i in xs])
        else:
            ax.set_xticklabels(["%d\n%s" % (i, STAGE_TASKS[i]) for i in xs],
                               fontsize=6.0)
        # Shared S4 (VizWiz) highlight, drawn behind everything.
        ax.axvspan(3.55, 4.30, color=HILITE, alpha=0.75, zorder=0, lw=0)

    seq_st = FS.arm_line("SEQ")

    # ---------------- (a) POPE-adversarial F1 -------------------------------
    stage_axis(ax_f1, top=True)
    ax_f1.axhline(seq_f1[0], color="#888888", ls=":", lw=0.7, zorder=1)
    ax_f1.plot(xs, seq_f1, **seq_st)
    ax_f1.annotate("", xy=(4.02, seq_f1[4]), xytext=(4.02, seq_f1[0]),
                   arrowprops=dict(arrowstyle="-|>", color=WORSE_FG, lw=1.3))
    ax_f1.set_ylabel("F1")
    fpad = 0.10 * (max(seq_f1) - min(seq_f1))
    ax_f1.set_ylim(min(seq_f1) - fpad - 0.006, max(seq_f1) + fpad)
    ax_f1.set_title("(a) POPE-adversarial F1", fontsize=7.4)

    # ---------------- (b) raw CHAIR_i ---------------------------------------
    stage_axis(ax_ch, top=True)
    ax_ch.axhline(seq_chair[0], color="#888888", ls=":", lw=0.7, zorder=1)
    ax_ch.plot(xs, seq_chair, **seq_st)
    ax_ch.annotate("", xy=(4.02, seq_chair[4]), xytext=(4.02, seq_chair[0]),
                   arrowprops=dict(arrowstyle="-|>", color=BETTER_FG, lw=1.3))
    ax_ch.set_ylabel("CHAIR$_i$")
    cpad = 0.12 * (max(seq_chair) - min(seq_chair))
    ax_ch.set_ylim(min(seq_chair) - cpad, max(seq_chair) + cpad)
    ax_ch.set_title("(b) raw CHAIR$_i$", fontsize=7.4)

    # ---------------- (c) criterion c ---------------------------------------
    stage_axis(ax_c, top=False)
    ax_c.axhline(seq_c[0], color="#888888", ls=":", lw=0.7, zorder=1)
    c_lo_i, c_hi_i = min(seq_c), max(seq_c)
    for yv in (c_lo_i, c_hi_i):
        ax_c.axhline(yv, color=FS.SEQ_COLOR, ls=(0, (1, 2)), lw=0.6,
                     alpha=0.45, zorder=1)
    ax_c.plot(xs, seq_c, **seq_st)
    xb = 0.30
    ax_c.annotate("", xy=(xb, c_hi_i), xytext=(xb, c_lo_i),
                  arrowprops=dict(arrowstyle="<->", color=FS.SEQ_COLOR, lw=1.0))
    ax_c.set_ylabel("criterion $c$")
    cpad2 = 0.16 * c_swing
    ax_c.set_ylim(c_lo_i - cpad2 - 0.02, c_hi_i + cpad2)
    ax_c.set_xlabel("training stage (task)")
    ax_c.set_title("(c) decision criterion $c$", fontsize=7.4)
    c_axspan = ax_c.get_ylim()[1] - ax_c.get_ylim()[0]

    # ---------------- (d) d' on a MATCHED y-span (honest flatness) -----------
    stage_axis(ax_d, top=False)
    ax_d.axhline(seq_d[0], color="#888888", ls=":", lw=0.7, zorder=1)
    ax_d.plot(xs, seq_d, **seq_st)
    d_mid = 0.5 * (max(seq_d) + min(seq_d))
    ax_d.set_ylim(d_mid - c_axspan / 2.0, d_mid + c_axspan / 2.0)
    ax_d.set_ylabel("$d'$")
    ax_d.set_xlabel("training stage (task)")
    ax_d.set_title("(d) discriminability $d'$, same span as (c)", fontsize=7.4)

    print("[fig1_paradox] F1 S0->S4 drop %.4f | CHAIR S0->S4 drop %.4f | "
          "c range %.3f endpoint %+.3f | d' range %.3f endpoint %+.3f | ratio %.2f"
          % (f1_drop, chair_drop, c_swing, c_move, d_range, d_move, range_ratio))

    FS.save_fig(fig, out_dir, "fig1_paradox", rendered, "make_paper_figures")

    facts = dict(seq_f1=seq_f1, seq_chair=seq_chair, seq_c=seq_c, seq_d=seq_d,
                 c_swing=c_swing, c_move=c_move, d_range=d_range,
                 d_move=d_move, range_ratio=range_ratio,
                 f1_drop=f1_drop, chair_drop=chair_drop,
                 gap_f1=gap_f1, gap_chair=gap_chair)
    return facts


# =========================================================================
# FIGURE 2 -- method vs baselines
# =========================================================================
def build_method_fig(mcj, lcc, g4cis, out_dir, rendered):
    fname_mc = "method_comparison.json"

    # ---- left: per-stage |Delta c| for SEQ / ER / anchor-v2 ----------------
    def deltas(ckpts):
        cs = mc_series(mcj, ckpts, "c")
        return [abs(cs[i + 1] - cs[i]) for i in range(4)], cs

    seq_d, seq_c = deltas(SEQ_CKPTS)
    er_d, er_c = deltas(E_CKPTS)
    g_d, g_c = deltas(G_CKPTS)
    _, j_c = deltas(J_CKPTS)
    sum_seq, sum_er, sum_g = sum(seq_d), sum(er_d), sum(g_d)
    # Post-settling (stages 2-4): drops the anchor's one-time settling step.
    post_seq, post_er, post_g = sum(seq_d[1:]), sum(er_d[1:]), sum(g_d[1:])

    # ---- THE COST that must accompany any Sigma|Delta c| number -------------
    # Mean |c| over the four trained checkpoints = how far the criterion sits
    # from the optimum.  POPE is exactly balanced (n_gt_yes = n_gt_no), so c=0
    # is derivable rather than a preference, and |c| is therefore a placement
    # error, not a values judgement.  Same window (stages 1-4) and same
    # estimator for every arm, so the four numbers are like-for-like.
    def mean_abs_c(cs):
        return sum(abs(v) for v in cs[1:]) / 4.0

    placement = {"SEQ": mean_abs_c(seq_c), "ER": mean_abs_c(er_c),
                 "V2": mean_abs_c(g_c), "JOINT": mean_abs_c(j_c)}

    # ---- fail-loud: Sigma|Delta c| and the ER-null must still hold ----------
    for name, got, want in (("SEQ", sum_seq, 1.243), ("ER", sum_er, 1.358),
                            ("anchor-v2", sum_g, 0.791)):
        if not approx(got, want, 5e-3):
            die("Sum|Delta c| for %s recomputed %.4f; METHOD_COMPARISON.md "
                "records %.3f. Data drift -- refusing to render."
                % (name, got, want))
    if not (sum_er >= sum_seq):
        die("ER null broken: ER Sum|Delta c| (%.3f) is below SEQ's (%.3f); the "
            "replay-does-not-help claim would be unsupported."
            % (sum_er, sum_seq))
    if not (post_g < 0.5 * post_seq):
        die("Anchor post-settling Sum|Delta c| (%.3f) is not below half SEQ's "
            "(%.3f); the committed pilot values have drifted -- refusing to "
            "render." % (post_g, post_seq))
    # The cost must be present and must point the way the record says it does:
    # the anchor is MORE mis-placed than SEQ.  If that ever flips, the figure's
    # panel (b) framing is wrong and the figure must not render silently.
    if not (placement["V2"] > placement["SEQ"]):
        die("Anchor mean |c| (%.3f) is not above SEQ's (%.3f); the recorded "
            "finding is that the anchor buys stability at a MIS-PLACED "
            "operating point. Data drift -- refusing to render."
            % (placement["V2"], placement["SEQ"]))
    stage4_ratio_seq = seq_d[3] / g_d[3]
    stage4_ratio_er = er_d[3] / g_d[3]

    # ---- right: endpoint CHAIR_i@60 bars + anchor CIs ----------------------
    endpoints = [("S4", "SEQ", "SEQ"), ("E4", "ER", "ER"),
                 ("G4", "V2", "anchor\nv2"), ("J4", "JOINT", "JOINT")]
    at60_vals = {c: at60(lcc, c) for c, _, _ in endpoints}
    # Recompute & assert the two paper-quoted anchor CIs from the @60 data.
    ci_map = {}
    for ref, key in (("S4", "G4_minus_S4_at60"), ("J4", "G4_minus_J4_at60")):
        node = g4cis.get(key)
        if not isinstance(node, dict) or "point" not in node or "ci95" not in node:
            die("g4_bootstrap_cis.json missing '%s' point/ci95" % key)
        doc_gap = float(node["point"])
        lo, hi = float(node["ci95"][0]), float(node["ci95"][1])
        recomputed = at60_vals["G4"] - at60_vals[ref]
        if not approx(recomputed, doc_gap, 5e-4):
            die("Recomputed G4-%s@60 gap %.6f from length_controlled_chair.json "
                "does not match g4_bootstrap_cis.json point %.4f -- refusing to "
                "render a caption whose CIs no longer match the data."
                % (ref, recomputed, doc_gap))
        if not (lo < 0 and hi < 0):
            die("Anchor CHAIR@60 CI vs %s = [%.4f, %.4f] does not exclude zero "
                "on the reducing side; the 'significant reduction' annotation "
                "would be false." % (ref, lo, hi))
        ci_map[ref] = (doc_gap, lo, hi)

    # ------------------------------- draw -----------------------------------
    # Three panels: the drift win, the placement cost that must travel with it,
    # and the withdrawn generative observation.  Two rows, because at FULL_W a
    # 1x3 layout leaves each lower panel ~1.1in and its arm tick labels collide.
    fig = plt.figure(figsize=(FS.FULL_W, FS.h_full(4.50)))
    gs_t = fig.add_gridspec(1, 1, left=0.085, right=0.988, top=0.935,
                            bottom=0.585)
    gs_b = fig.add_gridspec(1, 2, left=0.085, right=0.988, top=0.455,
                            bottom=0.085, wspace=0.34)
    ax_l = fig.add_subplot(gs_t[0, 0])
    ax_m = fig.add_subplot(gs_b[0, 0])
    ax_r = fig.add_subplot(gs_b[0, 1])

    # ---- LEFT: grouped |Delta c| bars --------------------------------------
    arms = [("SEQ", seq_d, sum_seq), ("ER", er_d, sum_er), ("V2", g_d, sum_g)]
    w = 0.26
    for j, (key, dl, tot) in enumerate(arms):
        xs = [i + (j - 1) * w for i in range(4)]
        ax_l.bar(xs, dl, width=w,
                 label="%s  ($\\Sigma$=%.2f)" % (FS.ARM_LABEL[key], tot),
                 **FS.arm_bar(key))
    # Mark the anchor's one-time settling step (stage 1, j=2).
    ax_l.annotate("one-time\nsettling", xy=(0 + 1 * w, g_d[0]),
                  xytext=(0.55, g_d[0] + 0.02), ha="left", va="bottom",
                  fontsize=5.7, color=FS.V2_COLOR,
                  arrowprops=dict(arrowstyle="-|>", color=FS.V2_COLOR, lw=0.7))
    # Highlight the VizWiz-stage suppression (stage 4). Curve the connector up
    # and over the tall SEQ/ER bars so it does not cut across them.
    top4 = max(seq_d[3], er_d[3], g_d[3])
    ax_l.annotate("anchor damps $c$ here:\n$\\sim$%.0f$\\times$ vs SEQ, "
                  "$\\sim$%.0f$\\times$ vs ER" % (stage4_ratio_seq,
                                                  stage4_ratio_er),
                  xy=(3 + 1 * w + 0.02, g_d[3] + 0.01),
                  xytext=(2.02, top4 * 0.985),
                  ha="left", va="top", fontsize=6.2, color=FS.V2_COLOR,
                  fontweight="bold",
                  arrowprops=dict(arrowstyle="-|>", color=FS.V2_COLOR, lw=0.9,
                                  connectionstyle="arc3,rad=-0.32"))
    ax_l.set_xticks(range(4))
    ax_l.set_xticklabels(["stage %d\n%s" % (k, STAGE_TASKS[k])
                          for k in (1, 2, 3, 4)], fontsize=6.0)
    ax_l.set_xlabel("training stage")
    ax_l.set_ylabel("per-stage $|\\Delta c|$  (pooled POPE)")
    ax_l.set_ylim(0, top4 * 1.16)
    ax_l.legend(fontsize=6.3, loc="upper left", handlelength=1.1,
                handletextpad=0.5, labelspacing=0.3)
    ax_l.set_title("(a) THE WIN: replay does not damp criterion drift, the "
                   "anchor does", fontsize=7.0)

    # ---- MIDDLE: the placement cost (never show (a) without this) ----------
    pl_order = [("SEQ", "SEQ"), ("ER", "ER"), ("V2", "anchor\nv2"),
                ("JOINT", "JOINT")]
    pl_vals = [placement[k] for k, _ in pl_order]
    for i, ((key, _), v) in enumerate(zip(pl_order, pl_vals)):
        bar_kw = FS.arm_bar(key)
        if key == "V2":
            bar_kw = dict(bar_kw, linewidth=1.4)
        ax_m.bar([i], [v], width=0.64, **bar_kw)
        ax_m.text(i, v + 0.012, "%.2f" % v, ha="center", va="bottom",
                  fontsize=6.3,
                  fontweight="bold" if key == "V2" else "normal")
    ax_m.axhline(0.0, color="#333333", lw=0.8, zorder=4)
    ax_m.text(0.015, 0.955, "0 = the optimum ($c=0$; POPE is balanced)",
              transform=ax_m.transAxes, ha="left", va="top", fontsize=5.9,
              color="#555555")
    ax_m.set_xticks(range(4))
    ax_m.set_xticklabels([lbl for _, lbl in pl_order], fontsize=6.2)
    ax_m.set_xlabel("arm (mean over stages 1$-$4)")
    ax_m.set_ylabel("mean $|c|$")
    ax_m.set_xlim(-0.60, 3.60)
    ax_m.set_ylim(0, max(pl_vals) * 1.22)
    ax_m.set_title("(b) THE COST: a worse operating point", fontsize=7.0)

    # ---- RIGHT: endpoint CHAIR_i@60 bars -----------------------------------
    order = [("S4", "SEQ"), ("E4", "ER"), ("G4", "V2"), ("J4", "JOINT")]
    vals = [at60_vals[c] for c, _ in order]
    xpos = list(range(4))
    for i, ((ckpt, key), v) in enumerate(zip(order, vals)):
        bar_kw = FS.arm_bar(key)
        if key == "V2":
            bar_kw = dict(bar_kw, linewidth=1.4)   # bold the anchor
        ax_r.bar([i], [v], width=0.62, **bar_kw)
        lbl = "%.3f" % v
        ax_r.text(i, v + 0.0016, lbl, ha="center", va="bottom",
                  fontsize=6.3,
                  fontweight="bold" if key == "V2" else "normal")

    gi = 2  # anchor bar index
    top = max(vals)
    # Significance brackets: anchor vs SEQ (i=0) and anchor vs JOINT (i=3).
    def bracket(i_ref, ylev, label):
        x0, x1 = min(gi, i_ref), max(gi, i_ref)
        ax_r.plot([x0, x0, x1, x1],
                  [vals[x0] + 0.006, ylev, ylev, vals[x1] + 0.006],
                  color="#333333", lw=0.7, zorder=5)
        ax_r.text((x0 + x1) / 2.0, ylev + 0.0012, label, ha="center",
                  va="bottom", fontsize=6.0, color="#222222")

    bracket(0, top + 0.011, "$-$%.03f$^{*}$" % abs(ci_map["S4"][0]))
    bracket(3, top + 0.030, "$-$%.03f$^{*}$" % abs(ci_map["J4"][0]))
    ax_r.set_xticks(xpos)
    ax_r.set_xticklabels([lbl for _, lbl in
                          [("S4", "SEQ"), ("E4", "ER"), ("G4", "anchor\nv2"),
                           ("J4", "JOINT")]], fontsize=6.2)
    ax_r.set_xlabel("endpoint (after stage 4)")
    ax_r.set_ylabel("CHAIR$_i$@60")
    ax_r.set_ylim(0, top + 0.072)
    ax_r.set_title("(c) Endpoint CHAIR: WITHDRAWN\n"
                   "($^{*}$pilot CI excludes 0)", fontsize=7.0,
                   color="#8a2a2a")
    # The panel stays only as the pilot observation the full study then tested.
    # Stamping it inside the axes means the withdrawal cannot be lost if the
    # panel is ever lifted out of its caption.
    ax_r.text(0.5, 0.015, "did not replicate at full-study scale",
              transform=ax_r.transAxes, ha="center", va="bottom",
              fontsize=5.9, color="#8a2a2a", fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.26", fc="#FBEDEA",
                        ec="#8a2a2a", lw=0.6))

    FS.save_fig(fig, out_dir, "fig_method_vs_baselines", rendered,
                "make_paper_figures")

    return dict(sum_seq=sum_seq, sum_er=sum_er, sum_g=sum_g,
                post_seq=post_seq, post_er=post_er, post_g=post_g,
                seq_d=seq_d, er_d=er_d, g_d=g_d, placement=placement,
                at60=at60_vals, ci=ci_map,
                r_seq=stage4_ratio_seq, r_er=stage4_ratio_er)


# =========================================================================
def write_caption(out_dir, f1):
    """Standalone Fig. 1 caption -> analysis/out/fig1_paradox_caption.txt.
    Numbers are the ones the figure just asserted, not retyped constants."""
    txt = (
        "Figure 1. Same checkpoints, opposite verdicts -- and their "
        "resolution. All panels track the sequential arm (SEQ: continual "
        "instruction tuning of LLaVA-1.5-7B on ScienceQA -> TextVQA -> "
        "Flickr30k -> VizWiz), evaluated at every stage boundary S0..S4; "
        "S0 is the untuned base. Top (the paradox): the two field-standard "
        "hallucination instruments move in opposite interpretive directions "
        "over the same sequence. (a) POPE-adversarial F1 ends %.3f below "
        "base (0.827 -> 0.782), the textbook signature of MORE hallucination; "
        "(b) raw CHAIR_i ends %.3f below base (0.156 -> 0.127), the textbook "
        "signature of LESS hallucination. The disagreement is not an artifact "
        "of comparing to base: against the data- and step-matched joint twin "
        "at the endpoint, F1 says SEQ is worse (J4-S4 = +%.3f) while raw "
        "CHAIR says SEQ is better (S4-J4 = %.3f), both with bootstrap CIs "
        "excluding zero. Two standard scores, identical model states, "
        "contradictory readings. Bottom (the resolution): decomposing every "
        "POPE evaluation into signal-detection coordinates shows one cause "
        "for both. (c) The answer criterion c has a within-arm range of %.2f "
        "across the sequence and an endpoint displacement of %+.2f, spiking "
        "conservative at the abstention-heavy VizWiz stage (shaded S4) -- "
        "which simultaneously lowers F1 (fewer 'yes', so lower recall) and "
        "lowers raw CHAIR (the model asserts less and stops earlier). "
        "(d) Discrimination d' moves in the same directions but about %.0fx "
        "less, on a y-axis of the SAME span as c and summarised with the SAME "
        "two statistics (range %.2f, endpoint %+.2f). The claim is RELATIVE "
        "and is stated that way: d' is not immobile -- on the full study the "
        "sequential endpoint d' falls 0.121 (SE 0.0108 over nine cells), a "
        "small but real decline that is nonetheless statistically equivalent "
        "to zero within the pre-registered +/-0.30 margin -- it moves far "
        "less than the criterion does when both are summarised the same way. "
        "What the standard metrics register as changing hallucination is "
        "therefore a moving answer criterion at near-constant grounding; the "
        "field's endpoint audit cannot see the difference, and reads it wrong "
        "in opposite directions depending on the instrument. Pilot scale: one "
        "backbone, four tasks, LoRA, one seed/ordering for the primary run "
        "(criterion dynamics replicated 3/3 seeds and under reverse "
        "ordering)."
        % (f1["f1_drop"], f1["chair_drop"], f1["gap_f1"], f1["gap_chair"],
           f1["c_swing"], f1["c_move"], f1["range_ratio"], f1["d_range"],
           f1["d_move"])
    )
    path = os.path.join(out_dir, "fig1_paradox_caption.txt")
    with open(path, "w") as fh:
        fh.write(txt + "\n")
    print("[make_paper_figures] wrote %s" % path)
    return txt


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag", default="analysis/diag",
                    help="diagnostics dir (method_comparison.json, "
                         "length_controlled_chair.json, g4_bootstrap_cis.json)")
    ap.add_argument("--results", default="analysis/out",
                    help="dir holding results_full.csv (per-split POPE F1)")
    ap.add_argument("--out", default="analysis/out",
                    help="output dir (paper graphicspath: ../analysis/out/)")
    args = ap.parse_args()

    mcj = load_json(os.path.join(args.diag, "method_comparison.json"))
    lcc = load_json(os.path.join(args.diag, "length_controlled_chair.json"))
    g4cis = load_json(os.path.join(args.diag, "g4_bootstrap_cis.json"))
    adv_f1 = load_adv_f1(os.path.join(args.results, "results_full.csv"))
    os.makedirs(args.out, exist_ok=True)

    rendered = []
    f1 = build_fig1(mcj, adv_f1, args.out, rendered)
    m = build_method_fig(mcj, lcc, g4cis, args.out, rendered)
    cap = write_caption(args.out, f1)

    print("[make_paper_figures] fig1  : F1 drop -%.3f | CHAIR drop -%.3f | "
          "c swing %.2f | d' range %.2f, move +%.2f | gaps F1 %+.3f CHAIR %+.3f"
          % (f1["f1_drop"], f1["chair_drop"], f1["c_swing"], f1["d_range"],
             f1["d_move"], f1["gap_f1"], f1["gap_chair"]))
    print("[make_paper_figures] method: Sigma|dc| SEQ %.2f / ER %.2f / anchor "
          "%.2f (post-settling %.2f / %.2f / %.2f) | CHAIR@60 SEQ %.3f ER %.3f "
          "anchor %.3f JOINT %.3f"
          % (m["sum_seq"], m["sum_er"], m["sum_g"], m["post_seq"],
             m["post_er"], m["post_g"], m["at60"]["S4"], m["at60"]["E4"],
             m["at60"]["G4"], m["at60"]["J4"]))
    print("[make_paper_figures] DONE: %s" % ", ".join(rendered))
    print("[make_paper_figures] fig1 caption (%d chars) -> "
          "fig1_paradox_caption.txt" % len(cap))


if __name__ == "__main__":
    main()
