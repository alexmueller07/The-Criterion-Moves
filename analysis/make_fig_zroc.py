"""Figure: do a cell's stages slide along ONE z-ROC, or scatter off it?

This is the assumption-free form of the paper's primary claim. "Only the
criterion moves" has an exact geometric meaning: the two evidence distributions
are fixed and the threshold slides, so every stage of a cell is a different
operating point ON THE SAME ROC curve.  Plotted in z-space an ROC is a straight
line, z(H) = a + b*z(FA), and the slope b is ESTIMATED, never assumed to be 1,
so no equal-variance assumption enters anywhere.

THE FIGURE IS THE CHART AND NOTHING ELSE (2026-09-15).  The previous version
drew a four-line explanatory paragraph on the canvas and a three-line statistics
block in every panel title.  All of that prose now belongs in the LaTeX caption;
see the suggested wording in the render note printed at the bottom of this file.
What is left on the canvas is axes, ticks, a one-line statistic per panel, one
legend entry, and the data.

WHAT THE CANVAS STILL SAYS, and why it is honest:
  * Panel headings carry the arm name only.  The single statistic line is
    mean R^2 with the WORST cell's R^2 beside it, because R^2 alone hides the
    spread and the worst cell is the whole anchor story (0.0576).
  * No panel gets extra ink.  Line width, alpha and marker size are IDENTICAL
    in all three panels, so the difference a reader sees -- sequential's fits
    collapsing into one pencil, the anchor's fanning out -- is produced by the
    data and not by styling.
  * Axes are shared and the aspect is EQUAL BY CONSTRUCTION (the axes box is
    sized to xlim/ylim below), so a slope of 0.55 and a slope of 0.80 look like
    what they are.  That matters here: slope IS the quantity under discussion.

The reading is ASYMMETRIC, exactly as analysis/zroc_coherence.py and the
pre-registration state it -- low R^2 REJECTS fixed distributions for that arm,
high R^2 is only CONSISTENT with them (six points per fit).  That asymmetry is
prose, so it lives in the caption, not on the canvas.

Every rendered number is recomputed from analysis/readout/fs_aggregate.json with
the same z()/fit() logic as analysis/zroc_coherence.py, then ASSERTED against the
certified analysis/readout/zroc_coherence.json before anything is drawn.  The
assertion covers the FULL summary (n, mean/min R^2, mean/sd d_a), which is a
superset of what the canvas now prints -- dropping a number from the figure must
never drop it from the check.

Renders analysis/out/fig_zroc.{pdf,png}.
"""
import json
import math
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import fig_style as FS
    FS.apply_style()
    TEXTWIDTH = getattr(FS, "TEXTWIDTH", 5.5)
except Exception:
    TEXTWIDTH = 5.5
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})

C_SEQ = "#0072B2"   # house palette (fig_style.SEQ_COLOR), so it matches Figure 2 on the same page
C_ANC = "#D55E00"   # fig_style.V2_COLOR
C_JNT = "#E69F00"   # fig_style.JOINT_COLOR
C_INK = "#1a1a1a"
C_MUT = "#8A8A8A"


def z(p):
    """Inverse standard normal CDF (Acklam); adequate over the range POPE gives."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl = 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > 1 - pl:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def fit(pts):
    """Least-squares z(H) = a + b*z(FA). Returns (a, b, r2, d_a).

    Identical to analysis/zroc_coherence.py's fit(), INCLUDING the d_a term --
    the earlier version of this script dropped d_a, which is the index the
    assumption-free framing exists to report.  d_a is no longer PRINTED on the
    canvas (it is a caption/table number), but it is still computed and still
    asserted against the certified readout.
    """
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    syy = sum((y - my) ** 2 for _, y in pts)
    if sxx <= 0 or syy <= 0:
        return None
    b = sxy / sxx
    a = my - b * mx
    r2 = 1 - sum((y - (a + b * x)) ** 2 for x, y in pts) / syy
    return a, b, r2, math.sqrt(2 / (1 + b * b)) * a


def cells_for(arms, arm):
    """Every usable cell of one arm, as dicts carrying its points and its fit."""
    rows = []
    for key, cell in sorted(arms.items()):
        if key.split("|")[0] != arm:
            continue
        pts = []
        for s in sorted((x for x in cell if x.isdigit()), key=int):
            p = cell[s].get("pope")
            if p and 0 < p["H"] < 1 and 0 < p["FA"] < 1:
                pts.append((z(p["FA"]), z(p["H"])))
        if len(pts) < 4:
            continue
        f = fit(pts)
        if not f:
            continue
        a, b, r2, d_a = f
        rows.append({"cell": key.split("|", 1)[1].replace("|", "/"),
                     "pts": pts, "a": a, "b": b, "r2": r2, "d_a": d_a})
    return rows


def summarize(arm, rows):
    """Cross-cell summary, matching zroc_coherence.py's aggregation exactly.

    Fails loudly rather than dividing by zero if an arm yields no usable cell.
    """
    if not rows:
        raise SystemExit(
            "fig_zroc: arm '%s' yielded 0 usable cells from "
            "readout/fs_aggregate.json (need >=4 stages with 0<H<1 and 0<FA<1 "
            "and a non-degenerate fit). Refusing to render a panel with no "
            "data -- rerun the aggregate, or drop the panel deliberately."
            % arm)
    n = len(rows)
    d_as = [r["d_a"] for r in rows]
    return {"n": n,
            "mean_r2": round(st.mean(r["r2"] for r in rows), 4),
            "min_r2": round(min(r["r2"] for r in rows), 4),
            "mean_d_a": round(st.mean(d_as), 4),
            "sd_d_a": round(st.pstdev(d_as) * math.sqrt(n / max(1, n - 1)), 4)}


def assert_certified(arm, got, cert_path, cert):
    """Every number on this figure must equal the certified readout.

    Deliberately checks the WHOLE summary, not just the two values the panel
    prints: if a future edit puts d_a back on the canvas it is already covered,
    and a readout that drifts anywhere still fails the render.
    """
    if arm not in cert:
        raise SystemExit("fig_zroc: arm '%s' missing from %s -- rerun "
                         "analysis/zroc_coherence.py." % (arm, cert_path))
    ref = cert[arm]
    bad = [(k, got[k], ref.get(k)) for k in sorted(got) if got[k] != ref.get(k)]
    if bad:
        raise SystemExit(
            "fig_zroc: recomputed stats for arm '%s' disagree with %s: %s. "
            "Refusing to render uncertified numbers -- rerun "
            "analysis/zroc_coherence.py so the readout and the figure agree."
            % (arm, cert_path, "; ".join("%s recomputed=%r certified=%r" % t
                                         for t in bad)))


here = os.path.dirname(os.path.abspath(__file__))
agg = json.load(open(os.path.join(here, "readout", "fs_aggregate.json")))
CERT_PATH = os.path.join(here, "readout", "zroc_coherence.json")
cert = json.load(open(CERT_PATH))
L = agg["backbones"]["llava15"]
arms, base = L["arms"], L["base"]["pope"]

# "Anchor (probe)", never "Anchor (ours)": the anchor is a causal probe whose
# endpoint POPE F1 (0.7963) is below sequential's (0.8625) in 9/9 cells.
panels = [("seq", "Sequential", C_SEQ), ("anchor", "Anchor (probe)", C_ANC),
          ("joint", "Joint (matched)", C_JNT)]

# --- geometry -----------------------------------------------------------------
# fig_style contract: figsize IS the printed size, so every point size below is
# the size on the page.  INCL_FRAC must equal the fraction in the \includegraphics
# line -- widened 2026-09-16 at two external reviewers' request to width=0.9\textwidth,
# the house full-width convention, so 0.90 of the 5.5in column => 4.95in.  If that line ever changes, change
# this with it, or LaTeX rescales the figure and every point size below silently
# stops being the size on the page.
# Height is the binding constraint in a 9-page body, so the axes box is derived
# from the data window rather than guessed: choosing ah/aw = yspan/xspan makes
# the aspect EQUAL BY CONSTRUCTION (no set_aspect, so subplots_adjust stays
# authoritative and nothing floats), which both shortens the figure and stops
# the panels from exaggerating slope -- the one quantity this figure is about.
INCL_FRAC = 0.88                             # must match the fig_zroc include in paper/main.tex (it is 0.88)
FIG_W = INCL_FRAC * TEXTWIDTH                # 3.19in

XLIM = (-2.28, -0.76)   # drawn data spans -2.2314..-0.8069 / 0.1960..1.3271
YLIM = (0.12, 1.38)     # (verified against every drawn segment, not eyeballed)
L_M, R_M, WSPACE = 0.115, 0.995, 0.07        # axes-area fractions of FIG_W
TOP_IN, BOT_IN = 0.340, 0.275                # inches reserved for title + statistic line / xlabel
TITLE_PAD = 13.5                             # pt; must clear the statistic line

_aw = (R_M - L_M) * FIG_W / (3 + 2 * WSPACE)                   # panel width, in
_ah = _aw * (YLIM[1] - YLIM[0]) / (XLIM[1] - XLIM[0])          # panel height, in
FIG_H = _ah + TOP_IN + BOT_IN

fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H), sharex=True, sharey=True)
fig.subplots_adjust(left=L_M, right=R_M, top=1 - TOP_IN / FIG_H,
                    bottom=BOT_IN / FIG_H, wspace=WSPACE)

# Identical in all three panels: the contrast between panels must come from the
# data, not from a thicker line on the arm we want the reader to notice.
LW, ALPHA, MS = 0.7, 0.55, 9.0

base_labels = []
for ax, (arm, title, col) in zip(axes, panels):
    rows = cells_for(arms, arm)
    s = summarize(arm, rows)
    assert_certified(arm, s, CERT_PATH, cert)
    for r in rows:
        a, b = r["a"], r["b"]
        xs = [p[0] for p in r["pts"]]
        lo, hi = min(xs) - 0.15, max(xs) + 0.15
        ax.plot([lo, hi], [a + b * lo, a + b * hi], color=col, lw=LW,
                alpha=ALPHA, zorder=2)
        ax.scatter(xs, [p[1] for p in r["pts"]], s=MS, color=col, alpha=0.85,
                   zorder=3, edgecolor="white", linewidth=0.3)
    # The untuned base: the same point in all three panels, sized to the data
    # markers (it is a reference point, not the finding). Labelled beside the
    # diamond in EVERY panel rather than by one legend in the middle panel,
    # which read as belonging to that panel alone (co-author feedback,
    # 2026-09-21). Above-left of the diamond is empty in all three panels.
    h = ax.scatter([z(base["FA"])], [z(base["H"])], marker="D", s=9,
                   color=C_INK, zorder=4, edgecolor="white", linewidth=0.3)
    base_labels.append(ax.annotate("Untuned base", xy=(z(base["FA"]), z(base["H"])),
                                   xytext=(-3.5, 2.5), textcoords="offset points",
                                   ha="right", va="bottom", fontsize=7, color=C_INK,
                                   zorder=5))

    # Heading = arm name only.  One statistic line beneath it: mean R^2 with the
    # worst cell beside it, because the spread is the finding and a mean alone
    # hides it.  Both numbers are certified above.
    FS.panel_title(ax, "abc"[list(axes).index(ax)], title, pad=TITLE_PAD)
    ax.text(0.0, 1.035, "Mean $R^2$ %.3f (min %.3f)" % (s["mean_r2"], s["min_r2"]),
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7, color=C_MUT)
    # Cell count, in the corner rather than the title: the joint bound is n=2
    # against n=9, and a reader must not have to take mean R^2 on trust.
    ax.text(0.975, 0.035, "%d runs" % s["n"], transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color=C_MUT)

    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([-2.0, -1.5, -1.0])
    # 1-decimal tick labels on purpose: 2 decimals widen the left margin
    # enough to push the y-label into them.
    ax.set_yticks([0.4, 0.8, 1.2])
    ax.set_xlabel("$z(\\mathrm{FA})$", color=C_INK, labelpad=1.6)
    ax.grid(color="#D8D5CE", lw=0.35, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(C_MUT)
        ax.spines[sp].set_linewidth(0.6)
    ax.tick_params(colors=C_INK, length=0, pad=1.8)

axes[0].set_ylabel("$z(H)$", color=C_INK, labelpad=3.0)
# Each "Untuned base" label must clear every data point and fit line in its
# panel; checked on the rendered geometry, not by eye.
fig.canvas.draw()
_r = fig.canvas.get_renderer()
for _ax, _lab in zip(axes, base_labels):
    _bb = _lab.get_window_extent(_r).expanded(1.04, 1.10)
    _hit = 0
    _base = (z(base["FA"]), z(base["H"]))
    for _coll in _ax.collections:
        for _xy in _coll.get_offsets():
            if abs(_xy[0] - _base[0]) < 1e-9 and abs(_xy[1] - _base[1]) < 1e-9:
                continue                     # the label's own diamond
            _hit += _bb.contains(*_ax.transData.transform(_xy))
    for _ln in _ax.get_lines():
        _xs, _ys = _ln.get_data()
        for _i in range(40):
            _t = _i / 39.0
            _x = _xs[0] + _t * (_xs[-1] - _xs[0]); _y = _ys[0] + _t * (_ys[-1] - _ys[0])
            _hit += _bb.contains(*_ax.transData.transform((_x, _y)))
    assert _hit == 0, "Untuned base label overlaps data in a panel (%d hits)" % _hit
    assert _ax.bbox.contains(_bb.x0, _bb.y0) and _ax.bbox.contains(_bb.x1, _bb.y1), "label leaves its panel"
    print("  label clear of data in panel %s" % "abc"[list(axes).index(_ax)])

out = os.path.join(here, "out")
os.makedirs(out, exist_ok=True)
# No bbox_inches="tight": figsize must stay the printed size, or the caption
# below it silently inflates the page.
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, "fig_zroc." + ext), dpi=300, facecolor="white")
print("wrote %s/fig_zroc.{pdf,png} at %.3f x %.3f in -- the PRINTED size at\n"
      "  \\includegraphics[width=%.2f\\textwidth] of a %.2fin text block.\n"
      "  panel axes %.3f x %.3f in; data aspect ratio %.3f (equal)."
      % (out, FIG_W, FIG_H, INCL_FRAC, TEXTWIDTH, _aw, _ah,
         (_aw / (XLIM[1] - XLIM[0])) / (_ah / (YLIM[1] - YLIM[0]))))
