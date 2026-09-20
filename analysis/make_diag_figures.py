"""Diagnostic figure pipeline for the CL x hallucination pilot (PILOT_FINDINGS.md).

USAGE:  python analysis/make_diag_figures.py --diag analysis/diag --out analysis/out
                                             [--results results]

Reads the six diagnostic outputs:
  signal_detection.csv          (A1: d'/c per checkpoint x POPE category)
  length_controlled_chair.json  (A3/A10: CHAIR at fixed word budgets + bootstrap CIs;
                                 includes the E/F/G method-arm checkpoints)
  criterion_vs_stats.json       (A11: dose-response of yes-rate vs stage answer stats)
  format_drift.json             (A9: per-checkpoint/task answer-format aggregates)
  method_comparison.json        (five-arm method comparison: pooled dprime/c/yes_rate
                                 etc. per checkpoint, incl. ER (E), anchor-v1 (F),
                                 anchor-v2 (G))
  robustness_aggregation.json   (T1-T5 robustness targets: per-seed swing sums,
                                 reverse-ordering gaps, S1->S3 per-seed deltas,
                                 single-stage control criteria)

fig_robustness additionally recomputes the per-checkpoint criterion c directly
from results/<ckpt>/pope_gen.jsonl (the aggregation JSON stores only the summary
targets), using the identical SDT recipe as analysis/robustness_aggregation.py
(parse_yn from pilot/metrics_pope; clip to [1/(2N), 1-1/(2N)]; c = -(zH+zF)/2),
and asserts its recomputed swing sums and control criteria against the JSON --
any mismatch or missing checkpoint aborts.

Produces (each as .png at 300 dpi AND vector .pdf, built at final printed size --
see fig_style.py for the size/font contract shared with make_figures.py; the
interpretive text lives in the paper captions, not in the figures):
  fig_robustness         (left) 3-seed forward criterion trajectories, SEQ thin
                         vs anchor-v2 thick, shared S0 reference; (right)
                         reverse-ordering trajectories (SEQ-REV, G-REV) with the
                         two single-stage control endpoints as isolated markers
                         at the reverse stage training the same task.
  fig_criterion_dprime   criterion c (left) vs d' (right) across checkpoints, all
                         five arms, pooled POPE. SEQ/JOINT/ER are muted context
                         lines, anchor v1 (falsified) is thin+dashed, anchor v2 is
                         emphasized. Right panel y-span matched to the left panel.
  fig_method_endpoint    (left) per-stage |Delta c| bars for the S/E/F/G arms;
                         (right) endpoint CHAIR_i@60 bars for S4/E4/F4/G4/J4. The
                         two G4 paired-bootstrap CIs quoted in the paper caption
                         are re-derived and asserted here (see G4_ENDPOINT_CIS).
  fig_dose_response      per-stage Delta yes-rate bars, colored/labeled by the stage's
                         training answer statistics.
  fig_length_controlled  CHAIR_i full (top) vs @60-words (bottom), SEQ vs JOINT,
                         with the S1->S3 and S4-vs-J4 bootstrap gaps annotated.
  fig_leakage            exact-"Unanswerable" rate on TextVQA prompts, SEQ vs JOINT bars.

Every number shown is read from the diag files (or arithmetic on them), with ONE
documented exception: the two G4 endpoint bootstrap CIs (quoted in the paper's
fig_method_endpoint caption) are transcribed from analysis/METHOD_COMPARISON.md
(paired bootstrap computed by analysis/diag/g4_bootstrap_cis.py), and each
transcribed point gap is recomputed here from length_controlled_chair.json and
asserted to match -- a mismatch aborts the figure. A missing file/key/column
aborts with a clear error naming the file and key path -- no silent defaults,
nothing is ever fabricated.

matplotlib only; colorblind-safe Okabe-Ito palette; Python 3.9 compatible.
"""
import argparse
import csv
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_style as FS

FS.apply_style()

CKPTS = ["S0", "S1", "S2", "S3", "S4", "J1", "J2", "J3", "J4"]
SEQ_CKPTS = ["S0", "S1", "S2", "S3", "S4"]      # SEQ trajectory (S0 = shared base)
JOINT_CKPTS = ["S0", "J1", "J2", "J3", "J4"]    # JOINT trajectory (S0 = shared base)
E_CKPTS = ["S0", "E1", "E2", "E3", "E4"]        # ER (experience replay) trajectory
F_CKPTS = ["S0", "F1", "F2", "F3", "F4"]        # anchor v1 trajectory (falsified)
G_CKPTS = ["S0", "G1", "G2", "G3", "G4"]        # anchor v2 trajectory
STAGE_LABELS = ["S0", "S1/J1", "S2/J2", "S3/J3", "S4/J4"]

TASK_PRETTY = {"scienceqa": "ScienceQA", "textvqa": "TextVQA",
               "flickr": "Flickr30k", "vizwiz": "VizWiz"}
TASK_SHORT = {"ScienceQA": "SciQA", "TextVQA": "TextVQA",
              "Flickr30k": "Flickr", "VizWiz": "VizWiz"}


# ------------------------------------------------------------- strict data access

def die(msg):
    raise SystemExit("[make_diag_figures] ERROR: %s" % msg)


def load_json(diag_dir, fname):
    path = os.path.join(diag_dir, fname)
    if not os.path.isfile(path):
        die("required diag file not found: %s" % path)
    with open(path) as f:
        try:
            return json.load(f)
        except ValueError as e:
            die("could not parse %s as JSON: %s" % (path, e))


def jget(obj, keys, fname):
    """Strict nested lookup; a missing key aborts naming file + full key path."""
    cur = obj
    for i, k in enumerate(keys):
        if not isinstance(cur, dict) or k not in cur:
            die("missing key '%s' in %s (no silent defaults; refusing to "
                "fabricate a value)" % (" -> ".join(str(x) for x in keys[:i + 1]),
                                        fname))
        cur = cur[k]
    return cur


def jnum(obj, keys, fname):
    v = jget(obj, keys, fname)
    try:
        return float(v)
    except (TypeError, ValueError):
        die("key '%s' in %s is not a number (got %r)"
            % (" -> ".join(str(x) for x in keys), fname, v))


def load_signal_detection(diag_dir):
    """signal_detection.csv -> {(ckpt, category): {col: str}}; strict columns."""
    fname = "signal_detection.csv"
    path = os.path.join(diag_dir, fname)
    if not os.path.isfile(path):
        die("required diag file not found: %s" % path)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        need_cols = {"ckpt", "category", "dprime", "c"}
        missing = need_cols - set(reader.fieldnames or [])
        if missing:
            die("%s is missing column(s): %s" % (fname, ", ".join(sorted(missing))))
        table = {}
        for row in reader:
            table[(row["ckpt"], row["category"])] = row
    return table


def sd_num(sd, ckpt, category, col):
    fname = "signal_detection.csv"
    key = (ckpt, category)
    if key not in sd:
        die("no row for ckpt=%s category=%s in %s" % (ckpt, category, fname))
    if col not in sd[key]:
        die("no column '%s' for ckpt=%s category=%s in %s"
            % (col, ckpt, category, fname))
    try:
        return float(sd[key][col])
    except (TypeError, ValueError):
        die("column '%s' for ckpt=%s category=%s in %s is not a number (got %r)"
            % (col, ckpt, category, fname, sd[key][col]))


def stage_task_names(cvs):
    """{stage_index 1..4: pretty task name} from criterion_vs_stats.json."""
    fname = "criterion_vs_stats.json"
    names = {}
    for k in (1, 2, 3, 4):
        raw = jget(cvs, ["train_stats_verified_at_source", "S%d" % k, "task"], fname)
        names[k] = TASK_PRETTY.get(str(raw), str(raw))
    return names


def ci_verdict(lo, hi):
    return "CI excludes 0" if (lo > 0 or hi < 0) else "CI straddles 0"


def ci_verdict_short(lo, hi):
    return "excl. 0" if (lo > 0 or hi < 0) else "straddles 0"


def save_both(fig, out_dir, name, rendered):
    FS.save_fig(fig, out_dir, name, rendered, "make_diag_figures")


# -------------------------------------------------------- fig_criterion_dprime

def fig_criterion_dprime(sd, mcj, cvs, out_dir, rendered):
    """Five arms, pooled POPE: c swings (SEQ/JOINT/ER), collapses (v1) or
    freezes after a settling step (v2); d' stays in a band for every arm
    except v1. SEQ/JOINT come from signal_detection.csv (category 'all');
    E/F/G come from method_comparison.json (rows -> ckpt -> dprime/c).
    Interpretation lives in the paper caption; the figure keeps only the
    data, the S0 reference lines, and a compact 5-arm legend."""
    fname_mc = "method_comparison.json"

    def mc_vals(ckpts, col):
        return [jnum(mcj, ["rows", c, col], fname_mc) for c in ckpts]

    # (arm key, c series, d' series, muted?)
    arms = [
        ("SEQ", [sd_num(sd, c, "all", "c") for c in SEQ_CKPTS],
         [sd_num(sd, c, "all", "dprime") for c in SEQ_CKPTS], True),
        ("JOINT", [sd_num(sd, c, "all", "c") for c in JOINT_CKPTS],
         [sd_num(sd, c, "all", "dprime") for c in JOINT_CKPTS], True),
        ("ER", mc_vals(E_CKPTS, "c"), mc_vals(E_CKPTS, "dprime"), True),
        ("V1", mc_vals(F_CKPTS, "c"), mc_vals(F_CKPTS, "dprime"), False),
        ("V2", mc_vals(G_CKPTS, "c"), mc_vals(G_CKPTS, "dprime"), False),
    ]

    fig, (ax_c, ax_d) = plt.subplots(1, 2, figsize=(FS.COL_W, FS.h_col(2.15)))

    # S0 reference (dotted): both panels, from the SEQ series' shared base.
    s0_c, s0_d = arms[0][1][0], arms[0][2][0]
    ax_c.axhline(s0_c, color="#777777", linestyle=":", linewidth=0.8, zorder=1)
    ax_d.axhline(s0_d, color="#777777", linestyle=":", linewidth=0.8, zorder=1)

    handles, labels = [], []
    for key, cs, ds, mute in arms:
        style = FS.arm_line(key, mute=mute)
        if key == "V1":
            style["alpha"] = 0.8
        (ln,) = ax_c.plot(range(5), cs, **style)
        ax_d.plot(range(5), ds, **style)
        handles.append(ln)
        labels.append(FS.ARM_LABEL[key])

    for ax in (ax_c, ax_d):
        ax.set_xticks(range(5))
        ax.set_xticklabels(["0", "1", "2", "3", "4"])
        ax.set_xlabel("stage")

    ax_c.set_ylabel("criterion $c$")
    ax_d.set_ylabel("$d'$")

    c_all = [v for _, cs, _, _ in arms for v in cs]
    c_span = max(c_all) - min(c_all)
    ax_c.set_ylim(min(c_all) - 0.06 * c_span, max(c_all) + 0.06 * c_span)

    # Honest flatness: give the d' panel the same y-axis SPAN as the c panel
    # (the paper caption states this property -- keep it).
    c_lo, c_hi = ax_c.get_ylim()
    span = c_hi - c_lo
    d_all = [v for _, _, ds, _ in arms for v in ds]
    mid = 0.5 * (max(d_all) + min(d_all))
    ax_d.set_ylim(mid - span / 2.0, mid + span / 2.0)

    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=6.5,
               frameon=False, bbox_to_anchor=(0.5, 0.0),
               columnspacing=0.7, handlelength=1.3, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.10, 1, 1), w_pad=1.2)
    save_both(fig, out_dir, "fig_criterion_dprime", rendered)


# ------------------------------------------------------- fig_method_endpoint

# Transcribed from analysis/METHOD_COMPARISON.md (paired bootstrap over images,
# computed by analysis/diag/g4_bootstrap_cis.py, output g4_bootstrap_cis.json).
# The paper's fig_method_endpoint caption quotes these CIs verbatim, so each
# point gap is recomputed below from length_controlled_chair.json and asserted
# to match the documented value -- a mismatch aborts the figure rather than
# letting the caption drift from the data.  {ref ckpt: (documented gap, lo, hi)}.
G4_ENDPOINT_CIS = {
    "S4": (-0.0128, -0.0228, -0.0028),
    "J4": (-0.0138, -0.0257, -0.0019),
}


def fig_method_endpoint(mcj, lcc, cvs, out_dir, rendered):
    """(left) per-stage |Delta c| bars for the S/E/F/G arms; (right) endpoint
    CHAIR_i@60 bars for S4/E4/F4/G4/J4. The G4 CIs quoted in the paper
    caption are consistency-checked against the diag data (see above)."""
    fname_mc = "method_comparison.json"
    fname_lc = "length_controlled_chair.json"
    tasks = stage_task_names(cvs)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(FS.FULL_W, FS.h_full(3.0)))

    # ---------------- left: per-stage |Delta c| bars (S / E / F / G) --------
    def c_series(ckpts):
        return [jnum(mcj, ["rows", c, "c"], fname_mc) for c in ckpts]

    arms = [("SEQ", SEQ_CKPTS), ("ER", E_CKPTS), ("V1", F_CKPTS),
            ("V2", G_CKPTS)]
    # Placement travels with drift: an arm can win on summed drift while
    # holding a mis-placed threshold. Both numbers go in the caption (printed
    # below for transcription). mean |c| is the distance from c = 0, which is
    # NOT the optimum: c = 0 is the balanced-accuracy optimum only under equal
    # variance, and the fitted z-ROC slopes reject equal variance. The paper
    # references placement to the matched joint arm's endpoint instead.
    w = 0.19
    for j, (key, ckpts) in enumerate(arms):
        cs = c_series(ckpts)
        deltas = [abs(cs[i + 1] - cs[i]) for i in range(4)]
        mean_abs_c = sum(abs(v) for v in cs[1:]) / 4.0
        xs = [i + (j - 1.5) * w for i in range(4)]
        ax_l.bar(xs, deltas, width=w, **FS.arm_bar(key))
        print("[fig_method_endpoint] %s: sum|dc| %.2f  mean|c| %.2f"
              % (FS.ARM_LABEL[key], sum(deltas), mean_abs_c))

    short = {"ScienceQA": "SciQA", "Flickr30k": "Flickr"}
    ax_l.set_xticks(range(4))
    ax_l.set_xticklabels(["%d\n%s" % (k, short.get(tasks[k], tasks[k]))
                          for k in (1, 2, 3, 4)])
    ax_l.set_xlabel("training stage")
    ax_l.set_ylabel("per-stage $|\\Delta c|$")
    l_top = max(abs(c_series(ck)[i + 1] - c_series(ck)[i])
                for _, ck in arms for i in range(4))
    ax_l.set_ylim(0, l_top * 1.08)
    ax_l.set_title("(a) criterion displacement per stage", fontsize=7.4)

    # ------------- right: endpoint CHAIR_i@60 bars ---------------------------
    def at60(ckpt):
        return jnum(lcc, ["checkpoints", ckpt, "at_k", "60", "chair_i"],
                    fname_lc)

    # Consistency check: recompute each transcribed gap from the diag JSON.
    for ref, (doc_gap, lo, hi) in G4_ENDPOINT_CIS.items():
        gap = at60("G4") - at60(ref)
        if abs(gap - doc_gap) > 5e-5:
            die("recomputed G4-%s@60 gap %.6f from %s does not match the "
                "value %.4f documented in METHOD_COMPARISON.md (and quoted "
                "in the paper caption) -- refusing to render a figure whose "
                "caption CIs no longer match the data" % (ref, gap, fname_lc,
                                                          doc_gap))

    endpoints = [("S4", "SEQ"), ("E4", "ER"), ("F4", "V1"), ("G4", "V2"),
                 ("J4", "JOINT")]
    vals = [at60(c) for c, _ in endpoints]
    for i, ((ckpt, key), v) in enumerate(zip(endpoints, vals)):
        ax_r.bar([i], [v], width=0.6, **FS.arm_bar(key))
        ax_r.text(i, v + 0.0015, "%.4f" % v, ha="center", va="bottom",
                  fontsize=6.3)

    ax_r.set_xticks(range(5))
    ax_r.set_xticklabels([FS.ARM_LABEL[k].replace("anchor ", "anchor\n")
                          for _, k in endpoints])
    ax_r.set_xlabel("endpoint checkpoint")
    ax_r.set_ylabel("CHAIR$_i$@60")
    ax_r.set_ylim(0, max(vals) * 1.16)
    # Withdrawn as a claim (the full study did not reproduce it). The panel
    # title keeps that attached to the panel without prose on the canvas; the
    # caption carries the full statement.
    ax_r.set_title("(b) endpoint CHAIR$_i$@60, pilot only", fontsize=7.4)

    # One legend row for both panels, below the axes, so it covers no bars.
    keys = ["SEQ", "ER", "V1", "V2", "JOINT"]
    handles = [mpatches.Patch(label=FS.ARM_LABEL[k], **FS.arm_bar(k)) for k in keys]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.3,
               frameon=False, handlelength=1.2, columnspacing=1.1,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=1.6)
    save_both(fig, out_dir, "fig_method_endpoint", rendered)


# ----------------------------------------------------------- fig_dose_response

def fig_dose_response(cvs, out_dir, rendered):
    """Per-stage Delta yes-rate follows the just-trained stage's answer prior."""
    fname = "criterion_vs_stats.json"
    dose = jget(cvs, ["dose_response"], fname)
    if not isinstance(dose, list) or len(dose) != 4:
        die("'dose_response' in %s must be a list of 4 stage entries (got %r)"
            % (fname, type(dose)))
    by_stage = {}
    for entry in dose:
        by_stage[jget(entry, ["stage"], fname + ":dose_response")] = entry
    for s in ("S1", "S2", "S3", "S4"):
        if s not in by_stage:
            die("stage '%s' missing from 'dose_response' in %s" % (s, fname))

    C_YES, C_REFUSE, C_NEUTRAL = "#D55E00", "#56B4E9", "#999999"
    deltas, colors, ticklabels = [], [], []
    for k in (1, 2, 3, 4):
        e = by_stage["S%d" % k]
        ctx = fname + ":dose_response:S%d" % k
        pretty = TASK_PRETTY.get(str(jget(e, ["task"], ctx)),
                                 str(jget(e, ["task"], ctx)))
        task = TASK_SHORT.get(pretty, pretty)
        d = jnum(e, ["d_yes_rate"], ctx)
        yes = jnum(e, ["yes_pct"], ctx)
        no = jnum(e, ["no_pct"], ctx)
        una = jnum(e, ["una_pct"], ctx)
        net_yes = jnum(e, ["net_yes"], ctx)
        deltas.append(d)
        if una >= 0.10:
            colors.append(C_REFUSE)
        elif net_yes >= 0.01:
            colors.append(C_YES)
        else:
            colors.append(C_NEUTRAL)
        if yes == 0.0 and no == 0.0:
            ratio = "no y/n ans."
        elif no > 0.0:
            ratio = "yes:no %.1f:1" % (yes / no)
        else:
            ratio = "yes:no %.1f:0" % (yes * 100.0)
        ticklabels.append("S%d %s\n%s\nunans. %.1f%%"
                          % (k, task, ratio, una * 100.0))

    # SE of a difference of two independent per-checkpoint yes-rates.
    se_yes = jnum(cvs, ["se_approx", "yes_rate"], fname)
    se_diff = math.sqrt(2.0) * se_yes

    fig, ax = plt.subplots(figsize=(FS.COL_W, FS.h_col(2.55)))
    x = range(4)
    ax.bar(x, deltas, width=0.62, color=colors, edgecolor="black",
           linewidth=0.5, yerr=[se_diff] * 4, capsize=2,
           error_kw=dict(ecolor="#444444", lw=0.8), zorder=3)
    ax.axhline(0.0, color="black", linewidth=0.7, zorder=2)
    for xi, d in zip(x, deltas):
        off = 0.006 if d >= 0 else -0.006
        ax.text(xi, d + off + (se_diff if d >= 0 else -se_diff), "%+.3f" % d,
                ha="center", va="bottom" if d >= 0 else "top", fontsize=7,
                fontweight="bold")
    # Headroom so value labels never collide with the axes edges.
    hi = max(d + se_diff for d in deltas)
    lo = min(d - se_diff for d in deltas)
    ax.set_ylim(lo - 0.033, hi + 0.028)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ticklabels, fontsize=6.0)
    ax.set_ylabel("$\\Delta$ POPE yes-rate (pooled)")
    # error-bar definition is stated in the caption, not on the canvas
    ax.legend(handles=[
        mpatches.Patch(fc=C_YES, ec="black", lw=0.5,
                       label="prior pushes toward yes"),
        mpatches.Patch(fc=C_REFUSE, ec="black", lw=0.5,
                       label="prior pushes toward refusal/no"),
        mpatches.Patch(fc=C_NEUTRAL, ec="black", lw=0.5,
                       label="no yes/no supervision"),
    ], fontsize=6.5, loc="lower left", handlelength=1.2, handleheight=0.9)
    fig.tight_layout()
    save_both(fig, out_dir, "fig_dose_response", rendered)


# ------------------------------------------------------- fig_length_controlled

def fig_length_controlled(lcc, out_dir, rendered):
    """CHAIR_i full (top) vs @60 words (bottom): the S1->S3 rise survives
    length control; the S4-vs-J4 endpoint gap does not."""
    fname = "length_controlled_chair.json"
    ks = jget(lcc, ["ks"], fname)
    if 60 not in [int(k) for k in ks]:
        die("word budget 60 not present in 'ks' of %s (got %r)" % (fname, ks))

    def full_ci(ckpt):
        return jnum(lcc, ["checkpoints", ckpt, "full", "chair_i"], fname)

    def at60_ci(ckpt):
        return jnum(lcc, ["checkpoints", ckpt, "at_k", "60", "chair_i"], fname)

    seq_full = [full_ci(c) for c in SEQ_CKPTS]
    joint_full = [full_ci(c) for c in JOINT_CKPTS]
    seq_60 = [at60_ci(c) for c in SEQ_CKPTS]
    joint_60 = [at60_ci(c) for c in JOINT_CKPTS]

    boots = {}
    for key in ("S3_minus_S1@full", "S3_minus_S1@60",
                "S4_minus_J4@full", "S4_minus_J4@60"):
        gap = jnum(lcc, ["bootstrap", key, "gap"], fname)
        ci = jget(lcc, ["bootstrap", key, "ci95"], fname)
        if not (isinstance(ci, list) and len(ci) == 2):
            die("bootstrap '%s' ci95 in %s is not a [lo, hi] pair (got %r)"
                % (key, fname, ci))
        boots[key] = (gap, float(ci[0]), float(ci[1]))

    fig, (ax_f, ax_k) = plt.subplots(2, 1, figsize=(FS.COL_W, FS.h_col(3.5)), sharex=True)
    panels = ((ax_f, seq_full, joint_full, "full", "full captions"),
              (ax_k, seq_60, joint_60, "60", "fixed 60-word budget"))
    for ax, seq, joint, tag, title in panels:
        # BOTH callout boxes live in the clear band ABOVE every marker, one
        # left one right.  They used to be anchored at y = 0.05 of the axes,
        # where the opaque box sat directly on top of the SEQ line's
        # S1->S2 segment in the @60 panel and erased it.  There is no room
        # inside the data for a 22-character box at this aspect ratio, so the
        # headroom is widened and the boxes are pinned to it.
        vals = seq + joint
        vspan = max(vals) - min(vals)
        ax.set_ylim(min(vals) - 0.14 * vspan, max(vals) + 0.20 * vspan)
        ax.set_xlim(-0.30, 4.30)
        gap_w, lo_w, hi_w = boots["S3_minus_S1@%s" % tag]
        gap_e, lo_e, hi_e = boots["S4_minus_J4@%s" % tag]
        ax.axvspan(1, 3, color="#009E73", alpha=0.10, zorder=1)
        # The two statistics are stated in the caption rather than boxed on the
        # canvas; printed here so the caption is transcribed from the data.
        print("[fig_length_controlled] %s: S1->S3 %+.4f CI [%+.4f, %+.4f]; "
              "S4-J4 %+.4f CI [%+.4f, %+.4f]"
              % (tag, gap_w, lo_w, hi_w, gap_e, lo_e, hi_e))
        ax.plot(range(5), seq, label="SEQ", **FS.arm_line("SEQ"))
        ax.plot(range(5), joint, label="JOINT", **FS.arm_line("JOINT"))
        # The endpoint gap itself stays in the data: a bare double arrow, no
        # text, so nothing can occlude a line.
        ax.annotate("", xy=(4.0, seq[4]), xytext=(4.0, joint[4]),
                    arrowprops=dict(arrowstyle="<->", color="#444444", lw=0.8,
                                    shrinkA=3, shrinkB=3))
        ax.set_ylabel("CHAIR$_i$")
        ax.set_title(title, fontsize=8)
    ax_k.set_xticks(range(5))
    ax_k.set_xticklabels(STAGE_LABELS)
    ax_k.set_xlabel("checkpoint (stage-matched)")

    handles, labels = ax_f.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=6.5,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.055, 1, 1), h_pad=1.2)
    save_both(fig, out_dir, "fig_length_controlled", rendered)


# ---------------------------------------------------------------- fig_leakage

def fig_leakage(fd, out_dir, rendered):
    """Exact-'Unanswerable' rate on TextVQA prompts: SEQ vs JOINT bars."""
    fname = "format_drift.json"

    def unans(ckpt):
        rate = jnum(fd, ["aggregates", "%s/textvqa" % ckpt, "unans", "eq"], fname)
        n = jnum(fd, ["aggregates", "%s/textvqa" % ckpt, "unans", "eq_n"], fname)
        return rate * 100.0, int(n)

    seq = {c: unans(c) for c in SEQ_CKPTS}
    joint = {c: unans(c) for c in JOINT_CKPTS[1:]}   # S0 shared with SEQ

    fig, ax = plt.subplots(figsize=(FS.COL_W, FS.h_col(2.3)))
    w = 0.35
    # Shared base checkpoint as one neutral bar.
    ax.bar([0], [seq["S0"][0]], width=w, color=FS.BASE_COLOR, edgecolor="black",
           linewidth=0.5, label="S0 (shared base)", zorder=3)
    seq_x = [k - w / 2 for k in (1, 2, 3, 4)]
    joint_x = [k + w / 2 for k in (1, 2, 3, 4)]
    seq_v = [seq["S%d" % k][0] for k in (1, 2, 3, 4)]
    joint_v = [joint["J%d" % k][0] for k in (1, 2, 3, 4)]
    ax.bar(seq_x, seq_v, width=w, label="SEQ",
           **FS.arm_bar("SEQ"))
    ax.bar(joint_x, joint_v, width=w, label="JOINT",
           **FS.arm_bar("JOINT"))

    for xs, vs in (([0], [seq["S0"][0]]), (seq_x, seq_v), (joint_x, joint_v)):
        for xi, v in zip(xs, vs):
            ax.text(xi, v + 0.25, "%.1f" % v, ha="center", va="bottom",
                    fontsize=6.5)

    ax.set_xticks(range(5))
    ax.set_xticklabels(STAGE_LABELS)
    ax.set_xlabel("checkpoint (stage-matched)")
    ax.set_ylabel('exact "Unanswerable" (%)')
    ax.set_ylim(0, max(seq_v + joint_v) * 1.18 + 1.0)
    ax.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    save_both(fig, out_dir, "fig_leakage", rendered)


# -------------------------------------------------------------- fig_robustness

ROB_FNAME = "robustness_aggregation.json"

# Checkpoint-directory naming per arm/seed (matches analysis/robustness_aggregation.py).
ROB_SEED_ARMS = {
    "SEQ": {"seed17": ["S1", "S2", "S3", "S4"],
            "seed23": ["seq_s2_k%d" % k for k in range(1, 5)],
            "seed31": ["seq_s3_k%d" % k for k in range(1, 5)]},
    "V2":  {"seed17": ["G1", "G2", "G3", "G4"],
            "seed23": ["g_s2_k%d" % k for k in range(1, 5)],
            "seed31": ["g_s3_k%d" % k for k in range(1, 5)]},
}
ROB_REV_ARMS = {"SEQ": ["seq_rev_k%d" % k for k in range(1, 5)],
                "V2": ["g_rev_k%d" % k for k in range(1, 5)]}
# Single-stage controls: (results dir, legend label, reverse-panel stage index
# = the reverse stage that trains the same task, JSON key in targets.T5).
ROB_CONTROLS = [
    ("vw_only_k1", "VizWiz-only (control)", 1, "VizWiz-only vs S4"),
    ("tv_only_k1", "TextVQA-only (control)", 3, "TextVQA-only vs S2"),
]
FWD_TASK_LABELS = ["S0", "SciQA", "TextVQA", "Flickr", "VizWiz"]
REV_TASK_LABELS = ["S0", "VizWiz", "Flickr", "TextVQA", "SciQA"]


def rob_criterion(results_dir, ck):
    """Criterion c for one checkpoint, recomputed from pope_gen.jsonl with the
    identical SDT recipe as analysis/robustness_aggregation.py. Fail-loud."""
    from statistics import NormalDist
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "pilot"))
    from metrics_pope import parse_yn
    path = os.path.join(results_dir, ck, "pope_gen.jsonl")
    if not os.path.isfile(path):
        die("required robustness checkpoint file not found: %s (no silent "
            "defaults; refusing to fabricate a trajectory)" % path)
    hits = fas = nyes = nno = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            pred = parse_yn(r["output"])
            if pred is None:
                continue
            if r["gt"] == "yes":
                nyes += 1
                hits += pred == "yes"
            else:
                nno += 1
                fas += pred == "yes"
    if nyes == 0 or nno == 0:
        die("no parsed yes/no ground-truth trials in %s" % path)

    def clip(p, n):
        return min(max(p, 1.0 / (2 * n)), 1.0 - 1.0 / (2 * n))

    nd = NormalDist()
    zh = nd.inv_cdf(clip(hits / nyes, nyes))
    zf = nd.inv_cdf(clip(fas / nno, nno))
    return -0.5 * (zh + zf)


def fig_robustness(rob, results_dir, out_dir, rendered):
    """(left) forward-ordering criterion trajectories for 3 seeds, SEQ thin vs
    anchor-v2 thick, S0 reference; (right) reverse-ordering trajectories with
    the two single-stage control endpoints as isolated markers. Per-checkpoint
    c is recomputed from results/ (the aggregation JSON stores only summaries)
    and asserted against the JSON's swing sums and control criteria."""
    if not os.path.isdir(results_dir):
        die("results dir not found: %s (needed to recompute per-checkpoint "
            "criteria for fig_robustness)" % results_dir)

    s0_c = rob_criterion(results_dir, "S0")

    # ---- recompute all trajectories, asserting against the aggregation JSON.
    fwd = {}   # (arm, seed) -> [c at S0, k1..k4]
    for arm, seeds in ROB_SEED_ARMS.items():
        for seed, cks in seeds.items():
            cs = [s0_c] + [rob_criterion(results_dir, c) for c in cks]
            fwd[(arm, seed)] = cs
            swings = [abs(cs[i + 1] - cs[i]) for i in range(4)]
            key = "%s/%s" % ("SEQ" if arm == "SEQ" else "G", seed)
            doc = jget(rob, ["targets", "T3_swing_sums", key], ROB_FNAME)
            if not (isinstance(doc, list) and len(doc) == 2):
                die("targets.T3_swing_sums.%s in %s is not a [total, post] "
                    "pair (got %r)" % (key, ROB_FNAME, doc))
            for got, want, tag in ((sum(swings), float(doc[0]), "total"),
                                   (sum(swings[1:]), float(doc[1]), "post")):
                if abs(got - want) > 1e-6:
                    die("recomputed %s swing sum (%s) %.6f does not match "
                        "%.6f in %s -- refusing to render a figure that "
                        "disagrees with the committed aggregation"
                        % (key, tag, got, want, ROB_FNAME))

    rev = {arm: [s0_c] + [rob_criterion(results_dir, c) for c in cks]
           for arm, cks in ROB_REV_ARMS.items()}

    ctrls = []
    for ck, label, stage, t5key in ROB_CONTROLS:
        c = rob_criterion(results_dir, ck)
        doc = jnum(rob, ["targets", "T5", t5key, "ctrl_c"], ROB_FNAME)
        if abs(c - doc) > 1e-6:
            die("recomputed %s criterion %.6f does not match targets.T5 "
                "value %.6f in %s" % (ck, c, doc, ROB_FNAME))
        ctrls.append((label, stage, c))

    # ---- render.
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(FS.FULL_W, FS.h_full(2.5)),
                                     sharey=True)
    for ax in (ax_l, ax_r):
        ax.axhline(s0_c, color="#777777", linestyle=":", linewidth=0.8,
                   zorder=1)

    seed_alpha = {"seed17": 1.0, "seed23": 0.75, "seed31": 0.55}
    handles, labels = [], []
    for arm, style_over in (("SEQ", dict(linewidth=0.9, markersize=2.6)),
                            ("V2", dict())):
        for i, seed in enumerate(sorted(ROB_SEED_ARMS[arm])):
            st = FS.arm_line(arm, alpha=seed_alpha[seed], **style_over)
            (ln,) = ax_l.plot(range(5), fwd[(arm, seed)], **st)
            if i == 0:
                handles.append(ln)
                labels.append(FS.ARM_LABEL[arm])
        ax_r.plot(range(5), rev[arm], **FS.arm_line(arm, **style_over))
    for (label, stage, c), marker in zip(ctrls, ("*", "P")):
        (ln,) = ax_r.plot([stage], [c], marker=marker, color="black",
                          linestyle="none", markersize=6.5,
                          markerfacecolor="white", markeredgewidth=0.9,
                          zorder=5)
        handles.append(ln)
        labels.append(label)

    for ax, xlabels, title in (
            (ax_l, FWD_TASK_LABELS, "forward ordering (3 seeds)"),
            (ax_r, REV_TASK_LABELS, "reverse ordering (1 run)")):
        ax.set_xticks(range(5))
        ax.set_xticklabels(xlabels, fontsize=6.0)
        ax.set_xlabel("stage (task trained)")
        ax.set_title(title, fontsize=8)
    ax_l.set_ylabel("criterion $c$ (pooled POPE)")

    vals = ([v for cs in fwd.values() for v in cs]
            + [v for cs in rev.values() for v in cs]
            + [c for _, _, c in ctrls])
    span = max(vals) - min(vals)
    ax_l.set_ylim(min(vals) - 0.08 * span, max(vals) + 0.08 * span)

    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6.5,
               frameon=False, bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.0, handlelength=1.4, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=1.4)
    save_both(fig, out_dir, "fig_robustness", rendered)


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag", required=True,
                    help="diagnostics dir (signal_detection.csv + 5 JSONs)")
    ap.add_argument("--out", required=True, help="output dir for figures")
    ap.add_argument("--results", default="results",
                    help="results dir with per-checkpoint pope_gen.jsonl "
                         "(fig_robustness; default: results)")
    args = ap.parse_args()

    sd = load_signal_detection(args.diag)
    lcc = load_json(args.diag, "length_controlled_chair.json")
    cvs = load_json(args.diag, "criterion_vs_stats.json")
    fd = load_json(args.diag, "format_drift.json")
    mcj = load_json(args.diag, "method_comparison.json")
    rob = load_json(args.diag, ROB_FNAME)
    os.makedirs(args.out, exist_ok=True)

    rendered = []
    fig_criterion_dprime(sd, mcj, cvs, args.out, rendered)
    fig_dose_response(cvs, args.out, rendered)
    fig_length_controlled(lcc, args.out, rendered)
    fig_leakage(fd, args.out, rendered)
    fig_method_endpoint(mcj, lcc, cvs, args.out, rendered)
    fig_robustness(rob, args.results, args.out, rendered)

    print("[make_diag_figures] DONE: %d figures (%s)"
          % (len(rendered), ", ".join(rendered)))


if __name__ == "__main__":
    main()
