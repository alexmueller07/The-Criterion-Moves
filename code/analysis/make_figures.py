"""Figure pipeline for the CL x hallucination pilot.

Usage:  python analysis/make_figures.py --results results/ --out figures/

Produces (each as .png at 300 dpi AND vector .pdf, built at final printed
size -- see fig_style.py for the size/font contract shared with
make_diag_figures.py):
  fig_trajectory        CHAIR_i and POPE-adversarial F1 vs checkpoint, SEQ vs JOINT
  fig_forgetting_matrix task-score heatmap, 4 tasks x S0..S4, values annotated
  fig_confounds         POPE yes-rate, CHAIR mean_new_tokens, CHAIR truncation_rate
  fig_pope_splits       F1 per POPE split (random/popular/adversarial) x checkpoint

Missing checkpoints/files are plotted as gaps (NaN) -- lines break instead of
interpolating across missing data; nothing is ever fabricated.

matplotlib only; colorblind-safe Okabe-Ito cycle; Python 3.9 compatible.
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_io import (CKPTS, SEQ_CKPTS, JOINT_CKPTS, STAGE_LABELS, TASKS,
                        TASK_METRIC, POPE_SPLITS, load_results,
                        chair_metric, pope_metric, task_score)
import fig_style as FS

FS.apply_style()

TASK_PRETTY = {"scienceqa": "ScienceQA", "textvqa": "TextVQA",
               "flickr": "Flickr30k", "vizwiz": "VizWiz"}
METRIC_PRETTY = {"mc_acc": "MC acc", "vqa_acc": "VQA acc",
                 "caption_uf1": "cap. uF1"}


def stage_values(table, ckpts, getter):
    """Values at each of the 5 stage positions; missing -> NaN (breaks the line)."""
    return [float("nan") if getter(table, c) is None else float(getter(table, c))
            for c in ckpts]


def has_data(vals):
    return any(not math.isnan(v) for v in vals)


def plot_arm_lines(ax, table, getter):
    """Plot SEQ + JOINT lines for one metric getter. Returns True if anything drawn."""
    drew = False
    seq = stage_values(table, SEQ_CKPTS, getter)
    joint = stage_values(table, JOINT_CKPTS, getter)
    if has_data(seq):
        # SEQ above JOINT: both visible at shared S0.
        ax.plot(range(5), seq, label="SEQ", **FS.arm_line("SEQ"))
        drew = True
    if has_data(joint):
        ax.plot(range(5), joint, label="JOINT", **FS.arm_line("JOINT"))
        drew = True
    if not drew:
        ax.text(0.5, 0.5, "no data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray")
    return drew


def style_stage_axis(ax):
    ax.set_xticks(range(5))
    ax.set_xticklabels(STAGE_LABELS)
    ax.set_xlabel("checkpoint (stage-matched)")


def save_both(fig, out_dir, name, rendered):
    FS.save_fig(fig, out_dir, name, rendered, "make_figures")


# ---------------------------------------------------------------- fig_trajectory

def fig_trajectory(table, out_dir, rendered):
    # figure* span: 0.9\textwidth. Captions live in the tex; ylabels carry the
    # metric + a minimal direction hint, nothing else.
    fig, axes = plt.subplots(1, 2, figsize=(FS.FULL_W, FS.h_full(2.0)))

    ax = axes[0]
    plot_arm_lines(ax, table, lambda t, c: chair_metric(t, c, "chair_i"))
    style_stage_axis(ax)
    ax.set_ylabel("CHAIR$_i$  ($\\uparrow$ worse)")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")

    ax = axes[1]
    plot_arm_lines(ax, table, lambda t, c: pope_metric(t, c, "adversarial", "f1"))
    style_stage_axis(ax)
    ax.set_ylabel("POPE-adv. F1  ($\\downarrow$ worse)")

    fig.tight_layout(w_pad=1.6)
    save_both(fig, out_dir, "fig_trajectory", rendered)


# --------------------------------------------------------- fig_forgetting_matrix

def fig_forgetting_matrix(table, out_dir, rendered):
    n_rows, n_cols = len(TASKS), len(SEQ_CKPTS)
    M = [[task_score(table, c, t) for c in SEQ_CKPTS] for t in TASKS]

    # 0.9\columnwidth in the tex.
    fig, ax = plt.subplots(figsize=(FS.COL_W90, FS.h_col90(2.25)))
    try:
        cmap = matplotlib.colormaps["viridis"].copy()
    except (AttributeError, KeyError):   # older matplotlib fallback
        cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#c8c8c8")

    import numpy as np  # matplotlib hard-dependency; used only for masking here
    arr = np.array([[float("nan") if v is None else v for v in row] for row in M],
                   dtype=float)
    # pcolormesh (not imshow) keeps the heatmap cells vector in the PDF.
    im = ax.pcolormesh(np.ma.masked_invalid(arr), cmap=cmap, vmin=0.0, vmax=1.0,
                       edgecolors="white", linewidth=0.4)
    ax.invert_yaxis()

    def cell_text_color(v):
        r, g, b, _ = cmap(v)
        return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.5 else "white"

    any_missing = False
    for i in range(n_rows):
        for j in range(n_cols):
            v = M[i][j]
            if v is None:
                ax.text(j + 0.5, i + 0.5, "n/a", ha="center", va="center",
                        fontsize=6.2, color="#555555")
                any_missing = True
            else:
                ax.text(j + 0.5, i + 0.5, "%.3f" % v, ha="center", va="center",
                        fontsize=6.2, color=cell_text_color(v))

    ax.set_xticks([j + 0.5 for j in range(n_cols)])
    ax.set_xticklabels(SEQ_CKPTS)
    ax.set_yticks([i + 0.5 for i in range(n_rows)])
    ax.set_yticklabels(["%s\n(%s)" % (TASK_PRETTY.get(t, t),
                                      METRIC_PRETTY.get(TASK_METRIC[t],
                                                        TASK_METRIC[t]))
                        for t in TASKS])
    ax.set_xlabel("SEQ checkpoint")
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label("task score", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5, width=0.6, length=2.4)
    cbar.outline.set_linewidth(0.6)
    if any_missing:
        # colorbar is the value legend; add a proxy only when n/a cells exist
        import matplotlib.patches as mpatches
        ax.legend(handles=[mpatches.Patch(facecolor="#c8c8c8", edgecolor="black",
                                          label="n/a = not evaluated")],
                  loc="upper left", bbox_to_anchor=(0.0, -0.22), fontsize=6.5,
                  frameon=False)
    fig.tight_layout()
    save_both(fig, out_dir, "fig_forgetting_matrix", rendered)


# ---------------------------------------------------------------- fig_confounds

def fig_confounds(table, out_dir, rendered):
    # is_rate: the quantity is a proportion in [0, 1], so the axis may never
    # show negative values.  Without this, matplotlib's singular-range handling
    # expands the identically-zero truncation series to about (-0.05, +0.05)
    # and the panel prints a NEGATIVE truncation rate.
    panels = [
        ("POPE yes-rate (all splits)", "yes-rate", True,
         lambda t, c: pope_metric(t, c, "all", "yes_rate")),
        ("CHAIR mean new tokens", "mean new tokens", False,
         lambda t, c: chair_metric(t, c, "mean_new_tokens")),
        ("CHAIR truncation rate (cap 512)", "truncation rate", True,
         lambda t, c: chair_metric(t, c, "truncation_rate")),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(FS.FULL_W, FS.h_full(1.95)))
    for k, (ax, (title, ylab, is_rate, getter)) in enumerate(zip(axes, panels)):
        plot_arm_lines(ax, table, getter)
        style_stage_axis(ax)
        ax.set_ylabel(ylab)
        # Shortened so the 3 titles do not run into each other at FULL_W/3
        # (~1.7in of axes); "(cap 512)" is the CHAIR generation budget and
        # moves into the panel body where it has room.
        ax.set_title(title.replace(" (cap 512)", ""), fontsize=8)
        if is_rate:
            vals = [v for v in stage_values(table, SEQ_CKPTS, getter)
                    + stage_values(table, JOINT_CKPTS, getter)
                    if v == v]                      # drop NaN gaps
            if vals and max(vals) - min(vals) < 1e-12:
                # Degenerate: a flat line on an autoscaled axis is unreadable
                # and invites a spurious negative range. Say the value instead.
                ax.set_ylim(0.0, 1.0)
                ax.text(0.5, 0.90, "%.3f at every checkpoint" % vals[0],
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=6.5, color="#444444")
            else:
                ax.set_ylim(bottom=max(0.0, ax.get_ylim()[0]))
        if k == 2:
            ax.text(0.5, 0.035, "generation budget: 512 new tokens",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=6.0, color="#666666")
        if k == 0 and ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best")
    fig.tight_layout(w_pad=1.4)
    save_both(fig, out_dir, "fig_confounds", rendered)


# -------------------------------------------------------------- fig_pope_splits

def fig_pope_splits(table, out_dir, rendered):
    fig, ax = plt.subplots(figsize=(FS.COL_W, FS.h_col(2.5)))
    split_colors = {"random": FS.OKABE_ITO[2], "popular": FS.OKABE_ITO[0],
                    "adversarial": FS.OKABE_ITO[3]}
    drew = False
    for split in POPE_SPLITS:
        getter = (lambda t, c, s=split: pope_metric(t, c, s, "f1"))
        seq = stage_values(table, SEQ_CKPTS, getter)
        joint = stage_values(table, JOINT_CKPTS, getter)
        if has_data(seq):
            ax.plot(range(5), seq,
                    **FS.arm_line("SEQ", color=split_colors[split]))
            drew = True
        if has_data(joint):
            ax.plot(range(5), joint,
                    **FS.arm_line("JOINT", color=split_colors[split]))
            drew = True
    if not drew:
        ax.text(0.5, 0.5, "no data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray")
    style_stage_axis(ax)
    ax.set_ylabel("POPE F1")
    ax.set_title("POPE F1 per split", fontsize=8)
    if drew:
        # Compact composite legend: 3 split colors + 2 arm line styles
        # (instead of 6 split-x-arm entries).
        handles = [Line2D([], [], color=split_colors[s], linewidth=1.2, label=s)
                   for s in POPE_SPLITS]
        handles += [Line2D([], [], color="#444444", label="SEQ",
                           **{k: v for k, v in FS.arm_line("SEQ").items()
                              if k in ("marker", "linestyle", "linewidth",
                                       "markersize")}),
                    Line2D([], [], color="#444444", label="JOINT",
                           **{k: v for k, v in FS.arm_line("JOINT").items()
                              if k in ("marker", "linestyle", "linewidth",
                                       "markersize")})]
        fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.5,
                   frameon=False, bbox_to_anchor=(0.5, 0.0),
                   columnspacing=0.8, handlelength=1.4)
        fig.tight_layout(rect=(0, 0.075, 1, 1))
    else:
        fig.tight_layout()
    save_both(fig, out_dir, "fig_pope_splits", rendered)


# ------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, help="results dir with S0..S4,J1..J4")
    ap.add_argument("--out", required=True, help="output dir for figures")
    args = ap.parse_args()

    table = load_results(args.results)
    os.makedirs(args.out, exist_ok=True)

    rendered = []
    fig_trajectory(table, args.out, rendered)
    fig_forgetting_matrix(table, args.out, rendered)
    fig_confounds(table, args.out, rendered)
    fig_pope_splits(table, args.out, rendered)

    present = [c for c in CKPTS if table.get(c)]
    print("[make_figures] DONE: %d figures (%s) from checkpoints [%s]"
          % (len(rendered), ", ".join(rendered), ", ".join(present)))


if __name__ == "__main__":
    main()
