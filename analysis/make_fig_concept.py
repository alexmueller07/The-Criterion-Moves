"""Motivation figure: why a hallucination score cannot say what changed.

Rebuilt 2026-09-14. The previous version was authored against the old two-column
geometry and, after the ICLR single-column resize, its text boxes collided badly --
panel titles ran through body text, the two formulas in panel (c) overlapped, and
the closing line was drawn twice. Nothing was recoverable by nudging, so the layout
is rewritten: fewer words, explicit axes rectangles, and every string placed in its
own reserved band so no two can overlap at the printed size.

The argument, in three panels:
  (a) the setting  -- continual instruction tuning, probed after every stage
  (b) the problem  -- one benchmark score, two incompatible causes
  (c) the fix      -- read two numbers instead of one

Panel (b) is NOT a cartoon. Both scenarios are drawn from actual Gaussian
parameters and both are scored by the same balanced-accuracy computation, which is
asserted to return the same value to three decimals before anything is drawn. If a
future edit breaks that equality the script fails rather than shipping a figure
whose premise is false.

Renders analysis/out/fig_concept.{pdf,png} at its final printed width.
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import fig_style as FS
    FS.apply_style()
    W = FS.FULL_W
except Exception:
    W = 4.95
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})

INK, MUT, FAINT = "#1a1a1a", "#5A6068", "#8A929E"
SIG, NOI, ACC = "#0072B2", "#8A929E", "#D55E00"
RULE, PANEL = "#D8DCE3", "#F2F4F7"


def phi(x, mu, sd):
    return math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


def Phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def balanced_acc(mu_n, mu_s, sd, c):
    """Hit rate and correct-rejection rate averaged: what a balanced probe scores."""
    H = 1 - Phi((c - mu_s) / sd)
    FA = 1 - Phi((c - mu_n) / sd)
    return 0.5 * (H + (1 - FA))


# --- the two scenarios ------------------------------------------------------
# A: the distributions move together -- the model genuinely sees worse.
# B: the distributions are untouched -- only the threshold moves.
A = dict(mu_n=-0.35, mu_s=0.35, sd=1.0, c=0.0)
sA = balanced_acc(**A)

# B keeps A's score by CONSTRUCTION rather than by hand-tuning: its distributions
# are far apart (good separation) and its threshold is solved for so the balanced
# accuracy matches A exactly. Bisection on a monotone branch -- pushing the
# threshold right lowers the score, so a root is bracketed between the signal mean
# and far into the tail.
B = dict(mu_n=-1.00, mu_s=1.00, sd=1.0, c=None)
lo, hi = B["mu_s"], B["mu_s"] + 6.0
for _ in range(200):
    mid = 0.5 * (lo + hi)
    if balanced_acc(B["mu_n"], B["mu_s"], B["sd"], mid) > sA:
        lo = mid
    else:
        hi = mid
B["c"] = 0.5 * (lo + hi)
sB = balanced_acc(**B)
assert abs(sA - sB) < 5e-4, (
    "the two scenarios must score the SAME for this figure's argument to hold; "
    "got %.4f and %.4f -- the bisection failed to bracket a root" % (sA, sB))
SCORE = "%.2f" % round(sA, 2)

FIG_H = 1.95
fig = plt.figure(figsize=(W, FIG_H))
fig.patch.set_facecolor("white")
# Only labels on the canvas: every sentence that explains the figure lives in
# the LaTeX caption. Each text element sits in space no curve or line enters.
for x, t in [(0.112, "(a)  the setting"), (0.480, "(b)  one score, two causes"),
             (0.848, "(c)  what we measure")]:
    fig.text(x, 0.975, t, ha="center", va="top", fontsize=7.6, color=INK,
             fontweight="bold")

# ============================ (a) the setting ==============================
ax = fig.add_axes([0.005, 0.04, 0.215, 0.83]); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
labels = ["base", "task 1", "task 2", "task $K$"]
ys = [0.80, 0.585, 0.37, 0.06]
for i, (lab, y) in enumerate(zip(labels, ys)):
    ax.add_patch(FancyBboxPatch((0.06, y), 0.50, 0.135,
                                boxstyle="round,pad=0.012,rounding_size=0.03",
                                fc="#E8EEF5" if i == 0 else PANEL,
                                ec=RULE, lw=0.7, transform=ax.transData))
    ax.text(0.31, y + 0.067, lab, ha="center", va="center", fontsize=6.9, color=INK)
    ax.annotate("", xy=(0.86, y + 0.067), xytext=(0.60, y + 0.067),
                arrowprops=dict(arrowstyle="-|>", color=ACC, lw=0.9,
                                shrinkA=0, shrinkB=0))
ax.plot([0.31, 0.31], [0.345, 0.215], color=MUT, lw=1.0, ls=(0, (1, 2.2)))
ax.text(0.905, 0.43, "probe", ha="left", va="center", fontsize=6.4,
        color=ACC, fontweight="bold", rotation=90)

# ====================== (b) one score, two causes ==========================
xs = [i * 0.02 - 4.0 for i in range(401)]
for row, (P, head) in enumerate([(A, "A   worse separation"), (B, "B   moved threshold")]):
    sub = fig.add_axes([0.300, 0.475 - row * 0.435, 0.360, 0.395])
    sub.set_xlim(-4.2, 4.2); sub.set_ylim(-0.11, 0.66)
    for sp in sub.spines.values():
        sp.set_visible(False)
    sub.set_yticks([]); sub.set_xticks([])
    sub.fill_between(xs, [phi(x, P["mu_n"], P["sd"]) for x in xs], color=NOI, alpha=0.30, lw=0)
    sub.plot(xs, [phi(x, P["mu_n"], P["sd"]) for x in xs], color=NOI, lw=1.1)
    sub.fill_between(xs, [phi(x, P["mu_s"], P["sd"]) for x in xs], color=SIG, alpha=0.24, lw=0)
    sub.plot(xs, [phi(x, P["mu_s"], P["sd"]) for x in xs], color=SIG, lw=1.3)
    # threshold: from the baseline up, so it never crosses the arrow below it
    sub.plot([P["c"], P["c"]], [0.0, 0.43], color=ACC, lw=1.5, solid_capstyle="butt")
    sub.text(P["c"], 0.445, "$c$", ha="center", va="bottom", fontsize=7.0,
             color=ACC, fontweight="bold")
    # row label above the curves (their peak is phi(0) = 0.399)
    sub.text(-4.15, 0.655, head, ha="left", va="top", fontsize=6.5,
             color=INK, fontweight="bold")
    sub.text(4.15, 0.30, "score\n%s" % SCORE, ha="right", va="center",
             fontsize=6.8, color=INK, fontweight="bold", linespacing=1.2,
             bbox=dict(boxstyle="round,pad=0.28", fc="#FCEFE7", ec=ACC, lw=0.7))
    # separation, drawn below the baseline where nothing else is
    sub.annotate("", xy=(P["mu_s"], -0.065), xytext=(P["mu_n"], -0.065),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8,
                                 shrinkA=0, shrinkB=0, mutation_scale=6))
    sub.plot(xs, [0.0] * len(xs), color=RULE, lw=0.6, zorder=0)

# ========================= (c) what we measure =============================
ax = fig.add_axes([0.700, 0.04, 0.295, 0.83]); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.add_patch(FancyBboxPatch((0.04, 0.72), 0.92, 0.17,
                            boxstyle="round,pad=0.015,rounding_size=0.03",
                            fc="#FCEFE7", ec=ACC, lw=0.8))
ax.text(0.50, 0.805, "hit rate $H$,  false alarms $FA$",
        ha="center", va="center", fontsize=6.6, color=INK)
for x0, col, sym, name in [(0.04, SIG, "$d'$", "separation"),
                           (0.53, ACC, "$c$", "threshold")]:
    ax.annotate("", xy=(x0 + 0.215, 0.475), xytext=(0.50, 0.70),
                arrowprops=dict(arrowstyle="-|>", color=MUT, lw=0.8))
    ax.add_patch(FancyBboxPatch((x0, 0.17), 0.43, 0.29,
                                boxstyle="round,pad=0.015,rounding_size=0.03",
                                fc="white", ec=col, lw=1.0))
    ax.text(x0 + 0.215, 0.355, sym, ha="center", va="center",
            fontsize=9.5, color=col, fontweight="bold")
    ax.text(x0 + 0.215, 0.235, name, ha="center", va="center",
            fontsize=6.0, color=MUT)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(out, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, "fig_concept." + ext), dpi=300, facecolor="white")
print("wrote %s/fig_concept.{pdf,png} at %.2f x %.2f in  (both scenarios score %s, B threshold at c=%.3f)"
      % (out, W, FIG_H, SCORE, B["c"]))
