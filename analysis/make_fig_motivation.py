"""Figure 1 (motivation) for the CL x hallucination PILOT.

USAGE:  python analysis/make_fig_motivation.py --diag analysis/diag --out analysis/out

The one-sentence message the figure must convey WITHOUT its caption:

  Across a continual instruction-tuning sequence the model's yes/no answer
  CRITERION c moves far more than its discrimination d' does -- on this pilot,
  4.0x as much when both are summarised with the SAME estimator -- so a large
  part of what standard metrics read as "more hallucination" is criterion
  placement, not lost grounding.

WHAT THIS FIGURE MAY NOT SAY (retracted claims; do not reintroduce):

  * "d' is flat / fixed / unchanged", "discrimination never moves".  FALSE.
    On the full study sequential's ENDPOINT d' FALLS (the defensible statements
    are the RELATIVE one above and a TOST equivalence result, both of which
    live in the paper text, not here).  This script is PILOT-only and states
    the relative fact with a like-for-like estimator; it never asserts
    flatness, and it never mixes a range on one panel with an endpoint delta
    on the other.
  * The anchor as a WORKING METHOD.  It is a CAUSAL PROBE ONLY: it removes the
    task-specific component of criterion movement and adds a large constant
    offset, holding c at a MIS-PLACED operating point.  On this pilot that is
    visible directly -- the anchor's endpoint criterion shift (+0.34) is
    LARGER than sequential's (+0.31), and both arms finish far from c = 0.

TWO panels, built at FULL_W (figure* span, 6.08in) per fig_style's size
contract (figsize == included width, so every point size is the printed size):

  (a) LEFT -- the INTUITION, a signal-detection SCHEMATIC drawn with
      matplotlib (NOT data): an "object absent" and an "object present"
      distribution on an internal-evidence axis, and a decision criterion
      shown in three ghosted positions (liberal <- neutral -> conservative).
      The d' bracket is LABELLED ONLY, never called fixed.  The geometry the
      panel draws -- one evidence axis along which the criterion slides -- is
      what the z-ROC coherence result supports (a cell's six stages do lie on
      one ROC); what the panel must not do is claim d' is constant.
      Grayscale + one muted accent so it reads as a schematic, never as data.

  (b) RIGHT -- the PILOT DATA, two stacked mini-axes sharing the stage axis:
      pooled-POPE criterion c (top) and d' (bottom) across S0..S4 for SEQ
      (plain sequential) vs the anchor (v2, the G arm).  BOTH panels report
      BOTH summaries for BOTH arms -- full-window range AND endpoint
      displacement -- so no comparison in the figure pairs incompatible
      estimators.  The two mini-axes are given the SAME data-units span (and
      the same annotation headroom in those units) so "c moves more than d'"
      is an honest visual, not a zoom artifact.

  The c panel draws c = 0 as a reference.  That is derivable, not a values
  choice: this JSON's own rows satisfy yes_rate == (H + FA) / 2 to 5e-5, i.e.
  the POPE split is exactly balanced, so an unbiased responder sits at c = 0.

EVERY number in panel (b) and every annotated magnitude is read from
analysis/diag/method_comparison.json (rows -> S0..S4/G1..G4 -> c/dprime) or is
arithmetic on it.  A DATA-DRIFT guard re-derives the six summaries the figure
prints and aborts if the committed JSON no longer yields them, so the ink
cannot drift from the data.  The guard is deliberately NOT a claim guard: it
asserts nothing about which arm wins, and it compares like windows with like
(every range below is taken over the arm's full S0..endpoint window).

matplotlib only; colorblind-safe Okabe-Ito palette; Python 3.9 compatible.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_style as FS

FS.apply_style()

# SEQ trajectory (S0 = shared base) and anchor-v2 trajectory (G arm).
SEQ_CKPTS = ["S0", "S1", "S2", "S3", "S4"]
G_CKPTS = ["S0", "G1", "G2", "G3", "G4"]
# Task trained at each SEQ stage (ScienceQA -> TextVQA -> Flickr30k -> VizWiz).
STAGE_TASKS = ["base", "SciQA", "TextVQA", "Flickr", "VizWiz"]

# Committed PILOT summaries (analysis/diag/method_comparison.json).  These are
# a DRIFT GUARD on the input file, not a claim: if the JSON is regenerated and
# any of them changes, the figure's printed numbers would be stale, so abort.
# Every range is over the arm's FULL window (S0..endpoint) -- like for like.
EXPECTED = {
    "SEQ c range":      0.778,
    "SEQ c endpoint":   0.313,
    "SEQ d' range":     0.193,
    "SEQ d' endpoint":  0.042,
    "anchor c range":   0.444,
    "anchor c endpoint": 0.337,
    "anchor d' range":  0.159,
    "anchor d' endpoint": -0.012,
}
DRIFT_TOL = 5e-3
# POPE is exactly balanced in this file (yes_rate == (H + FA) / 2), which is
# what licenses drawing c = 0 as "unbiased".  Verified, not assumed.
BALANCE_TOL = 1e-3


# ---------------------------------------------------------- strict data access

def die(msg):
    raise SystemExit("[make_fig_motivation] ERROR: %s" % msg)


def load_json(diag_dir, fname):
    path = os.path.join(diag_dir, fname)
    if not os.path.isfile(path):
        die("required diag file not found: %s" % path)
    with open(path) as f:
        try:
            return json.load(f)
        except ValueError as e:
            die("could not parse %s as JSON: %s" % (path, e))


def jnum(obj, keys, fname):
    """Strict nested numeric lookup; a missing/non-numeric key aborts loudly."""
    cur = obj
    for i, k in enumerate(keys):
        if not isinstance(cur, dict) or k not in cur:
            die("missing key '%s' in %s (no silent defaults; refusing to "
                "fabricate a value)" % (" -> ".join(str(x) for x in keys[:i + 1]),
                                        fname))
        cur = cur[k]
    try:
        return float(cur)
    except (TypeError, ValueError):
        die("key '%s' in %s is not a number (got %r)"
            % (" -> ".join(str(x) for x in keys), fname, cur))


def series(mcj, ckpts, col, fname):
    return [jnum(mcj, ["rows", c, col], fname) for c in ckpts]


def signed(v):
    """Signed 2dp with a typographic minus (U+2212), so a negative endpoint
    does not print with a stubby ASCII hyphen next to '+' values."""
    return ("%+.2f" % v).replace("-", "−")


def summarise(vals):
    """The TWO summaries the figure reports for EVERY series: full-window range
    and endpoint displacement.  Both panels use this same pair, so nothing in
    the figure ever compares a range against an endpoint delta."""
    return max(vals) - min(vals), vals[-1] - vals[0]


# ----------------------------------------------------------- (a) SDT schematic

def draw_schematic(ax):
    """Two distributions + a decision criterion in three ghosted positions.
    Purely illustrative geometry -- no data is read here; the accent stays
    muted/grayscale so the panel cannot be mistaken for measurements.  The d'
    bracket is LABELLED ONLY: this panel must not assert that d' is fixed."""
    GRAY_ABS = "#B4B4B4"     # object-absent (noise) distribution
    BLUE_PRES = "#8FC0E0"    # object-present (signal) distribution (muted sky)
    EDGE_ABS = "#808080"
    EDGE_PRES = "#3E7FA6"
    CRIT = "#1A1A1A"         # the neutral decision criterion
    GHOST = "#9A9A9A"        # drifted criterion positions
    ACC = "#555555"          # drift arrows / brackets

    mu_abs, mu_pres, sig = -1.15, 1.15, 1.0     # separation 2.3 sigma (schematic)
    x = np.linspace(-4.6, 4.6, 800)
    norm = 1.0 / (sig * math.sqrt(2 * math.pi))

    def g(mu):
        return norm * np.exp(-0.5 * ((x - mu) / sig) ** 2)

    ya, yp = g(mu_abs), g(mu_pres)
    peak = float(norm)                                # ~0.399

    # Faint evidence axis (curves rest on y=0); real spines are hidden.
    ax.axhline(0.0, color="#B0B0B0", lw=0.7, zorder=1)

    ax.fill_between(x, ya, color=GRAY_ABS, alpha=0.55, zorder=2, linewidth=0)
    ax.fill_between(x, yp, color=BLUE_PRES, alpha=0.55, zorder=2, linewidth=0)
    ax.plot(x, ya, color=EDGE_ABS, lw=1.0, zorder=3)
    ax.plot(x, yp, color=EDGE_PRES, lw=1.0, zorder=3)

    # Curve identity labels on the OUTER shoulders (clear of the center column).
    ax.text(-3.05, 0.205, "object\nabsent", ha="center", va="center",
            fontsize=6.8, color=EDGE_ABS, linespacing=0.95)
    ax.text(3.05, 0.205, "object\npresent", ha="center", va="center",
            fontsize=6.8, color=EDGE_PRES, linespacing=0.95)

    # d' = separation between the means.  Named, NOT claimed constant.
    y_d = peak + 0.190
    ax.annotate("", xy=(mu_pres, y_d), xytext=(mu_abs, y_d),
                arrowprops=dict(arrowstyle="<->", color=ACC, lw=0.9))
    ax.plot([mu_abs, mu_abs], [peak + 0.015, y_d], color=ACC, lw=0.5, ls=":")
    ax.plot([mu_pres, mu_pres], [peak + 0.015, y_d], color=ACC, lw=0.5, ls=":")
    ax.text(0.0, y_d + 0.018, "$d'$ (discrimination)",
            ha="center", va="bottom", fontsize=6.9, color="#2A2A2A")

    # Decision criterion: neutral (solid) + liberal/conservative ghosts; the
    # lines run from just below the baseline up to the criterion label.
    top = peak + 0.05
    for xc, ls, col, lw, z in ((-1.35, (0, (3, 2)), GHOST, 1.0, 3),
                               (0.0, "-", CRIT, 1.5, 4),
                               (1.35, (0, (3, 2)), GHOST, 1.0, 3)):
        ax.plot([xc, xc], [-0.055, top], color=col, lw=lw, ls=ls, zorder=z)
    ax.text(0.0, top + 0.012, "decision\ncriterion $c$", ha="center",
            va="bottom", fontsize=6.7, color=CRIT, linespacing=0.95)

    # Drift arrow + labels live in the clear zone BELOW the baseline.
    y_arrow = -0.085
    ax.annotate("", xy=(1.35, y_arrow), xytext=(-1.35, y_arrow),
                arrowprops=dict(arrowstyle="<->", color=ACC, lw=1.2))
    ax.text(-1.5, y_arrow, "liberal", ha="right", va="center",
            fontsize=6.6, color=ACC)
    ax.text(-1.5, y_arrow - 0.045, "(yes-biased)", ha="right", va="center",
            fontsize=5.7, color="#8A8A8A")
    ax.text(1.5, y_arrow, "conservative", ha="left", va="center",
            fontsize=6.6, color=ACC)
    ax.text(1.5, y_arrow - 0.045, "(no-biased)", ha="left", va="center",
            fontsize=5.7, color="#8A8A8A")
    ax.text(0.0, y_arrow - 0.052,
            "the criterion slides from\nstage to stage",
            ha="center", va="top", fontsize=6.8, color="#2A2A2A",
            linespacing=0.98)

    ax.set_xlim(-4.6, 4.6)
    ax.set_ylim(-0.255, peak + 0.34)
    ax.set_xlabel("internal evidence for “object present”")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.grid(False)
    for s in ("left", "top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title("(a) Signal-detection schematic (not data)", fontsize=7.6)


# ------------------------------------------------------------- (b) pilot data

# Shared geometry for the two stacked data panels.
DATA_PAD = 0.10     # padding around the data, as a fraction of the larger extent
HEADROOM = 0.42     # annotation band above the data, IN DATA UNITS (same for both)
X_LO, X_HI = -0.28, 4.34


def _panel_limits(vals, core, head):
    """Lower/upper y-limit centring `vals` in a `core`-wide band and adding the
    SAME `head` above.  Both panels therefore get an identical data-units span
    (identical units per inch) AND an identical annotation band."""
    mid = 0.5 * (max(vals) + min(vals))
    return mid - core / 2.0, mid + core / 2.0 + head


def draw_data(ax_c, ax_d, mcj, fname):
    """Pooled-POPE c (top) and d' (bottom) across S0..S4 for SEQ vs anchor v2.
    Both panels report the SAME two summaries (full-window range, endpoint
    displacement) for both arms; numbers are read from method_comparison.json
    and re-derived against the committed values as a data-drift guard."""
    seq_c = series(mcj, SEQ_CKPTS, "c", fname)
    seq_d = series(mcj, SEQ_CKPTS, "dprime", fname)
    g_c = series(mcj, G_CKPTS, "c", fname)
    g_d = series(mcj, G_CKPTS, "dprime", fname)

    # ---- like-for-like summaries computed from the data (never hardcoded).
    seq_c_rng, seq_c_end = summarise(seq_c)
    seq_d_rng, seq_d_end = summarise(seq_d)
    g_c_rng, g_c_end = summarise(g_c)
    g_d_rng, g_d_end = summarise(g_d)
    # The one comparative magnitude the figure states: range vs range.
    ratio_rng = seq_c_rng / seq_d_rng

    # ---- DATA-DRIFT GUARD (not a claim guard).  It asserts only that the
    # committed JSON still yields the numbers this figure prints, comparing
    # like windows with like; it takes no position on which arm is better.
    got = {
        "SEQ c range": seq_c_rng, "SEQ c endpoint": seq_c_end,
        "SEQ d' range": seq_d_rng, "SEQ d' endpoint": seq_d_end,
        "anchor c range": g_c_rng, "anchor c endpoint": g_c_end,
        "anchor d' range": g_d_rng, "anchor d' endpoint": g_d_end,
    }
    drift = ["%s: recomputed %+.4f, committed %+.4f" % (k, got[k], EXPECTED[k])
             for k in sorted(EXPECTED) if abs(got[k] - EXPECTED[k]) > DRIFT_TOL]
    if drift:
        die("method_comparison.json no longer yields the summaries this figure "
            "prints -- the rendered numbers would be stale. Refusing to render.\n"
            "  " + "\n  ".join(drift))

    # c = 0 is the unbiased point only because the POPE split is balanced.
    # Verify that in the file itself rather than assuming it.
    rows = mcj.get("rows", {})
    bal = max(abs(jnum(mcj, ["rows", k, "yes_rate"], fname)
                  - 0.5 * (jnum(mcj, ["rows", k, "H"], fname)
                           + jnum(mcj, ["rows", k, "FA"], fname)))
              for k in rows)
    if bal > BALANCE_TOL:
        die("POPE yes/no split is not balanced in %s (max |yes_rate - (H+FA)/2| "
            "= %.4f); c = 0 would not be the unbiased point, so the reference "
            "line this figure draws would be wrong" % (fname, bal))

    seq_c_lo, seq_c_hi = min(seq_c), max(seq_c)
    s0_d = seq_d[0]
    seq_st = FS.arm_line("SEQ")
    g_st = FS.arm_line("V2")
    xs = range(5)

    # Common vertical geometry: one data-units span for BOTH panels.
    core = max(max(seq_c + g_c) - min(seq_c + g_c),
               max(seq_d + g_d) - min(seq_d + g_d)) * (1.0 + 2.0 * DATA_PAD)
    c_lo_ax, c_hi_ax = _panel_limits(seq_c + g_c, core, HEADROOM)
    d_lo_ax, d_hi_ax = _panel_limits(seq_d + g_d, core, HEADROOM)

    # ---------------- top: criterion c --------------------------------------
    # c = 0 reference: balanced POPE (verified above) => unbiased responder.
    ax_c.axhline(0.0, color="#777777", ls="--", lw=0.7, zorder=1)
    ax_c.text(X_HI - 0.06, 0.035, "$c=0$ (unbiased)", ha="right", va="bottom",
              fontsize=5.8, color="#777777")
    # The SEQ range as a faint band (a dotted guide at the lower extreme would
    # sit ~0.03 units from the c = 0 line and read as a printing artifact).
    ax_c.axhspan(seq_c_lo, seq_c_hi, color=FS.SEQ_COLOR, alpha=0.055,
                 lw=0, zorder=0)
    ax_c.plot(xs, seq_c, label="SEQ", **seq_st)
    ax_c.plot(xs, g_c, label="anchor (causal probe)", **g_st)

    ax_c.set_ylim(c_lo_ax, c_hi_ax)
    ax_c.set_xlim(X_LO, X_HI)
    # Same two summaries, same window, for both arms.
    _stat_block(ax_c, c_hi_ax, core + HEADROOM, [
        ("SEQ $c$:  range %.2f,  endpoint %s" % (seq_c_rng, signed(seq_c_end)),
         FS.SEQ_COLOR),
        ("anchor $c$:  range %.2f,  endpoint %s" % (g_c_rng, signed(g_c_end)),
         FS.V2_COLOR),
    ])
    ax_c.set_ylabel("criterion $c$")

    # ---------------- bottom: d' (same data span => honest comparison) ------
    ax_d.axhline(s0_d, color="#888888", ls=":", lw=0.7, zorder=1)
    ax_d.plot(xs, seq_d, label="SEQ", **seq_st)
    ax_d.plot(xs, g_d, label="anchor (causal probe)", **g_st)

    ax_d.set_ylim(d_lo_ax, d_hi_ax)
    ax_d.set_xlim(X_LO, X_HI)
    _stat_block(ax_d, d_hi_ax, core + HEADROOM, [
        ("SEQ $d'$:  range %.2f,  endpoint %s" % (seq_d_rng, signed(seq_d_end)),
         FS.SEQ_COLOR),
        ("anchor $d'$:  range %.2f,  endpoint %s" % (g_d_rng, signed(g_d_end)),
         FS.V2_COLOR),
        # Says why d' LOOKS flat here: it is the shared span, not a null.
        ("both panels share one data-units span", "#6E6E6E"),
    ])
    ax_d.set_ylabel("$d'$")
    ax_d.set_xlabel("training stage (task)")
    ax_d.set_xticks(list(xs))
    ax_d.set_xticklabels(["%d\n%s" % (i, STAGE_TASKS[i]) for i in xs],
                         fontsize=6.0)
    ax_d.legend(loc="lower left", fontsize=5.9, ncol=2, handlelength=1.5,
                columnspacing=0.8, borderaxespad=0.3)

    ax_c.tick_params(labelbottom=False)
    # The headline is the LIKE-FOR-LIKE comparison, range against range.
    ax_c.set_title("(b) PILOT: $c$ range is %.1f$\\times$ the $d'$ range"
                   % ratio_rng, fontsize=7.6)

    return dict(seq_c_rng=seq_c_rng, seq_c_end=seq_c_end,
                seq_d_rng=seq_d_rng, seq_d_end=seq_d_end,
                g_c_rng=g_c_rng, g_c_end=g_c_end,
                g_d_rng=g_d_rng, g_d_end=g_d_end,
                ratio_rng=ratio_rng, balance=bal,
                seq_c_lo=seq_c_lo, seq_c_hi=seq_c_hi)


def _stat_block(ax, y_top, span, lines):
    """Two colour-coded summary lines in the panel's annotation headroom."""
    step = 0.085 * span
    for i, (txt, col) in enumerate(lines):
        ax.text(X_LO + 0.10, y_top - 0.035 * span - i * step, txt,
                ha="left", va="top", fontsize=6.0, color=col)


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diag", default="analysis/diag",
                    help="diagnostics dir (method_comparison.json)")
    ap.add_argument("--out", default="analysis/out",
                    help="output dir (paper graphicspath: ../analysis/out/)")
    args = ap.parse_args()

    fname = "method_comparison.json"
    mcj = load_json(args.diag, fname)
    os.makedirs(args.out, exist_ok=True)

    fig = plt.figure(figsize=(FS.FULL_W, FS.h_full(2.72)))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.06, 1.0], wspace=0.24,
                             left=0.045, right=0.985, top=0.90, bottom=0.155)
    ax_left = fig.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.14)
    ax_c = fig.add_subplot(right[0])
    ax_d = fig.add_subplot(right[1])

    draw_schematic(ax_left)
    f = draw_data(ax_c, ax_d, mcj, fname)

    rendered = []
    FS.save_fig(fig, args.out, "fig_motivation", rendered, "make_fig_motivation")
    print("[make_fig_motivation] PILOT, like-for-like summaries "
          "(full S0..endpoint window for every range):")
    print("[make_fig_motivation]   SEQ    c  range %.3f  endpoint %+.3f | "
          "d' range %.3f  endpoint %+.3f"
          % (f["seq_c_rng"], f["seq_c_end"], f["seq_d_rng"], f["seq_d_end"]))
    print("[make_fig_motivation]   anchor c  range %.3f  endpoint %+.3f | "
          "d' range %.3f  endpoint %+.3f"
          % (f["g_c_rng"], f["g_c_end"], f["g_d_rng"], f["g_d_end"]))
    print("[make_fig_motivation]   range ratio c/d' (SEQ) = %.2fx ; "
          "POPE balance residual %.1e" % (f["ratio_rng"], f["balance"]))
    print("[make_fig_motivation] DONE: %s" % ", ".join(rendered))


if __name__ == "__main__":
    main()
