"""The headline trajectory: the criterion moves across stages, discriminability does not.

Written 2026-09-16. Two independent readers of the submission made the same
point: the paper's central claim is a trajectory, and the main text never plotted
one. Figure 2's z-ROC pencils show the geometry but not the movement, and the
only criterion trajectories in the document were pilot-scale, in an appendix.

Everything here comes from analysis/readout/fs_aggregate.json -- the same file
behind every full-study number -- so this figure adds no data, only a view of it.

Left panel  : criterion c per stage, all 20 cells, three arms, c* and base marked.
Right panel : d' per stage on a SPAN-MATCHED axis, so "moves less" is visible
              rather than asserted. Both axes cover 1.30 units; that equality is
              asserted in code, because a reader comparing two panels will read
              relative movement off the ink and an unmatched pair would lie.
"""
import json, math, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import fig_style as FS
    FS.apply_style()
    W = FS.FULL_W
    SEQ, JOINT, ANCH, BASE = FS.SEQ_COLOR, FS.JOINT_COLOR, FS.V2_COLOR, "#999999"
except Exception:
    W = 4.95
    SEQ, JOINT, ANCH, BASE = "#0072B2", "#E69F00", "#D55E00", "#999999"
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42})

HERE = os.path.dirname(os.path.abspath(__file__))
AGG = os.path.join(HERE, "readout", "fs_aggregate.json")
CSTAR = 0.0878

bb = json.load(open(AGG))["backbones"]["llava15"]
arms, base = bb["arms"], bb["base"]["pope"]


def series(prefix, key):
    out = []
    for k, v in sorted(arms.items()):
        if not k.startswith(prefix + "|"):
            continue
        if not all(str(s) in v and "pope" in v[str(s)] for s in range(1, 7)):
            continue
        out.append([v[str(s)]["pope"][key] for s in range(1, 7)])
    return out


# Rendered at EXACTLY its included width (0.86\textwidth) so every point size
# prints at its stated size; the old 4.95 in render was shrunk to 96%.
W = 0.86 * FS.TEXTWIDTH
fig, axes = plt.subplots(1, 2, figsize=(W, 2.00))
stages = list(range(1, 7))

panels = [
    (axes[0], "c", ("a", "Decision criterion $c$"), CSTAR, base["c"]),
    (axes[1], "dprime", ("b", "Discriminability $d$′"), None, base["dprime"]),
]
spans = []
for ax, key, title, star, b in panels:
    allv = [b]
    for prefix, col, lw, z in (("seq", SEQ, 0.9, 3), ("anchor", ANCH, 0.9, 2), ("joint", JOINT, 1.4, 4)):
        for cell in series(prefix, key):
            allv += cell
            ax.plot(stages, cell, color=col, lw=lw, alpha=0.75, zorder=z,
                    solid_capstyle="round")
    lo, hi = min(allv), max(allv)
    mid, half = 0.5 * (lo + hi), 0.65          # 1.30-unit window on both panels
    ax.set_ylim(mid - half, mid + half)
    spans.append(2 * half)
    ax.axhline(b, color=BASE, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.text(6.12, b + 0.018, "Base", color=BASE, fontsize=7, va="bottom", ha="left")
    if star is not None:
        ax.axhline(star, color=JOINT, lw=0.8, ls=(0, (1, 1.6)), zorder=1)
        ax.text(6.12, star + 0.018, "$c^{*}$", color=JOINT, fontsize=8, va="bottom", ha="left")
    ax.set_xticks(stages)
    ax.set_xlabel("Stage")
    FS.panel_title(ax, *title)
    ax.set_xlim(0.8, 6.75)   # room right of stage 6 so the reference labels sit inside the axes
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

assert abs(spans[0] - spans[1]) < 1e-9, (
    "the two panels must cover the same number of units or the visual comparison "
    "of movement is meaningless; got %r" % (spans,))

h = [plt.Line2D([], [], color=SEQ, lw=1.4), plt.Line2D([], [], color=ANCH, lw=1.4),
     plt.Line2D([], [], color=JOINT, lw=1.6)]
# the d' panel is empty below its data band, so the legend goes there rather than over the anchor lines
axes[1].legend(h, ["Sequential (9 runs)", "Anchor (9)", "Joint (2)"],
               loc="lower left", handlelength=1.3, borderaxespad=0.2, labelspacing=0.25)

fig.subplots_adjust(left=0.085, right=0.985, bottom=0.175, top=0.90, wspace=0.24)
out = os.path.join(HERE, "out")
os.makedirs(out, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out, "fig_trajectory_main." + ext), dpi=300)
print("wrote %s/fig_trajectory_main.{pdf,png} at %.2f in wide; both panels span %.2f units"
      % (out, W, spans[0]))
