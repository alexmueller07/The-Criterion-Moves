"""Shared publication style for the pilot figure scripts.

Imported by BOTH analysis/make_figures.py and analysis/make_diag_figures.py so
that every figure uses one font family, one palette, one set of line weights,
and IDENTICAL arm identities (SEQ/JOINT/ER/anchor-v1/anchor-v2).

Figures are BUILT AT THEIR FINAL PRINTED SIZE.  Because figsize already equals
the included width, every point size below is the size on the printed page -- do
NOT enlarge figsize "for quality"; that silently shrinks the effective fonts.

2026-09-12: RESIZED FOR ICLR 2027.  The paper moved from a hand-set two-column
layout (textwidth 6.75in, columnwidth 3.25in) to the official
iclr2027_conference.sty, which is SINGLE COLUMN with textwidth 5.5in.  In a
single-column document \\columnwidth == \\textwidth, so:

  COL_W    5.225in = 0.95 * 5.5   (was 3.09 -- included at 0.95\\columnwidth)
  COL_W90  4.950in = 0.90 * 5.5   (was 2.93 -- fig_forgetting_matrix)
  FULL_W   4.950in = 0.90 * 5.5   (was 6.08 -- included at 0.9\\textwidth)

Each still occupies the SAME FRACTION of the text block as before, so the visual
design is unchanged; only the page proportions moved.  Heights are hardcoded at
the call sites, so scale them with H_COL / H_COL90 / H_FULL below -- changing a
width without its height silently restretches the figure.

Arm identity (never change one without changing every figure):
  SEQ    Okabe-Ito blue      #0072B2  circle   solid
  JOINT  Okabe-Ito orange    #E69F00  square   dashed
  ER     Okabe-Ito green     #009E73  triangle solid
  v1     Okabe-Ito purple    #CC79A7  x        fine-dashed, thin (falsified arm)
  v2     Okabe-Ito vermillion#D55E00  diamond  solid, thick (emphasized arm)
  base   neutral gray        #999999  (shared S0 bar / reference lines)

matplotlib only; colorblind-safe Okabe-Ito palette; Python 3.9 compatible.
"""
import os

import matplotlib
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00",
             "#CC79A7", "#56B4E9", "#F0E442", "#000000"]

SEQ_COLOR = "#0072B2"    # blue
JOINT_COLOR = "#E69F00"  # orange
ER_COLOR = "#009E73"     # bluish green -- experience-replay control (E arm)
V1_COLOR = "#CC79A7"     # reddish purple -- anchor v1 (F arm, falsified)
V2_COLOR = "#D55E00"     # vermillion -- anchor v2 (G arm, emphasized)
BASE_COLOR = "#999999"   # shared-S0 / neutral gray

# Final printed widths (inches); see module docstring.
# ICLR 2027 is single column, so columnwidth == textwidth.
TEXTWIDTH = 5.5
COLUMNWIDTH = TEXTWIDTH
COL_W = 0.95 * COLUMNWIDTH    # 5.225
COL_W90 = 0.90 * COLUMNWIDTH  # 4.950
FULL_W = 0.90 * TEXTWIDTH     # 4.950

# --- height scalers, so aspect ratio survives the resize ---------------------
# The figure heights are hardcoded at each call site and were chosen against the
# OLD widths below. Scaling a width without its height restretches the plot: the
# fonts stay correct (figsize is still the printed size) but a line plot flattens
# and a matrix stops being square. Wrap every legacy height in the matching
# scaler and the design is carried over untouched.
_LEGACY_COL_W, _LEGACY_COL_W90, _LEGACY_FULL_W = 3.09, 2.93, 6.08


def h_col(h):
    """Scale a height authored against the old 3.09in single-column width."""
    return h * COL_W / _LEGACY_COL_W


def h_col90(h):
    """Scale a height authored against the old 2.93in forgetting-matrix width."""
    return h * COL_W90 / _LEGACY_COL_W90


def h_full(h):
    """Scale a height authored against the old 6.08in full-span width."""
    return h * FULL_W / _LEGACY_FULL_W

PNG_DPI = 300

# One line/marker spec per arm, reused verbatim by every line plot.
_ARM_LINE = {
    "SEQ":   dict(color=SEQ_COLOR, marker="o", linestyle="-",
                  linewidth=1.1, markersize=3.2, zorder=3.0),
    "JOINT": dict(color=JOINT_COLOR, marker="s", linestyle="--",
                  linewidth=1.1, markersize=3.2, zorder=2.8),
    "ER":    dict(color=ER_COLOR, marker="^", linestyle="-",
                  linewidth=1.1, markersize=3.5, zorder=2.6),
    "V1":    dict(color=V1_COLOR, marker="x", linestyle=(0, (2.2, 1.4)),
                  linewidth=0.9, markersize=3.5, markeredgewidth=1.0,
                  zorder=2.4),
    "V2":    dict(color=V2_COLOR, marker="D", linestyle="-",
                  linewidth=1.8, markersize=3.0, zorder=4.0),
}

ARM_LABEL = {"SEQ": "SEQ", "JOINT": "JOINT", "ER": "ER",
             "V1": "anchor v1", "V2": "anchor v2"}


def arm_line(key, mute=False, **overrides):
    """Line-plot kwargs for one arm. mute=True de-emphasizes a context arm
    (thinner, lighter) without changing its color/marker identity."""
    st = dict(_ARM_LINE[key])
    if mute:
        st["alpha"] = 0.6
        st["linewidth"] = 0.9
        st["markersize"] = 2.6
    st.update(overrides)
    return st


def arm_bar(key, **overrides):
    """Bar kwargs for one arm; v1 keeps its hatch everywhere it is a bar."""
    st = dict(color=_ARM_LINE[key]["color"], edgecolor="black", linewidth=0.5,
              zorder=3)
    if key == "V1":
        st["hatch"] = "///"
    st.update(overrides)
    return st


def apply_style():
    """Set rcParams once, at import time of each figure script."""
    matplotlib.rcParams.update({
        # One font family across all figures; embed as TrueType (Type 42)
        # so PDF text stays vector and editable, never Type 3 / rasterized.
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Point sizes AT FINAL PRINTED SIZE (figsize == included size).
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "figure.titlesize": 8.0,
        # Line weights tuned for ~3in-wide panels.
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "axes.labelpad": 2.5,
        "axes.titlepad": 4.0,
        # Compact legends, light grid.
        "legend.frameon": False,
        "legend.handlelength": 1.7,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
        "legend.labelspacing": 0.35,
        "legend.borderaxespad": 0.2,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
        "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
        "figure.autolayout": False,
        "savefig.dpi": PNG_DPI,
    })


def save_fig(fig, out_dir, name, rendered, log_tag):
    """Save <name>.png (300 dpi proof at final size) + <name>.pdf (vector,
    tight bounding box) and close the figure."""
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, name + "." + ext),
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    rendered.append(name)
    print("[%s] wrote %s.{png,pdf}" % (log_tag, os.path.join(out_dir, name)))
