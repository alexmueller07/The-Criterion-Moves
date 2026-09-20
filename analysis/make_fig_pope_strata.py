"""RETRACTED FIGURE -- DO NOT PUT PANEL (a) IN THE PAPER. See the warning below.

!!! 2026-09-12 RETRACTION !!!
This figure's headline -- "d' tracks negative difficulty, criterion drift does not",
compared as LEVELS across the three strata -- is ARITHMETIC, not a finding. POPE's
strata share their positive items (verified byte-identical in the released
coco_pope_*.json), so the hit rate is constant across them and both indices are
affine in z(FA) alone, which forces

    delta d' = 2 * delta c    exactly, for any model whatsoever.

Measured on our own 9-cell seq readout, random -> adversarial: delta d' = -0.6816
against 2*delta c = -0.6820, residual +0.0004. Panel (a) therefore shows an identity
and is evidence for nothing. The figure is NOT referenced by main.tex and must not
be added. What survives is the PATH statistic (mean_path_c), a second difference
that the identity does not force; a figure built on that alone would be legitimate.
See analysis/pope_strata.py's docstring and the 2026-09-12 prereg entry.

Original docstring follows.

POPE-stratum double dissociation: d' tracks negative difficulty, criterion drift does not.

USAGE:  python analysis/make_fig_pope_strata.py
        [--readout analysis/readout/pope_strata_seq.json] [--out analysis/out]

WHY (2026-09-11).  Every measurement in the paper comes from POPE, so the obvious
reviewer question is whether the criterion account is an artifact of one way of
choosing negatives.  POPE is three benchmarks in one: `random`, `popular` and
`adversarial` differ ONLY in how the ABSENT object is drawn (uniformly / from the
most frequent classes / from those that most often co-occur with what is actually
in the image).  Task, images and prompt template are identical, so the three
strata form an increasingly hard demand on grounding with everything else held
fixed.

The result is a DOUBLE DISSOCIATION and it is the cleanest evidence in the paper
that c and d' are separate parameters rather than two views of one thing:

  (a) d' falls monotonically with negative difficulty -- the instrument responds
      to the thing POPE's design varies.
  (b) the criterion LEVEL falls with it too -- a harder negative set moves where
      the operating point sits, exactly as it should.
  (c) the criterion PATH LENGTH does NOT track difficulty at all, and is not even
      monotone -- the drift is a property of the decision rule, not of the
      sampling scheme.

Had the drift appeared only under easy negatives it would have been an artifact
and the paper would have had to say so.

ESTIMATOR DISCIPLINE (the rule this project keeps breaking).  All three panels
plot a quantity in the SAME units -- z units on the evidence axis -- and all
three are 9-cell means of a PER-CELL quantity, so nothing here pairs a path
length against a range or a per-cell number against a cross-cell average.  The
three panels are drawn on a COMMON y-axis SPAN so that the vertical excursion of
one panel is directly comparable to another's; only the centre differs.  The
number annotated on each panel is that panel's own cross-stratum spread
(max - min), the same statistic in all three.

Every number is read from analysis/readout/pope_strata_seq.json (produced by
analysis/pope_strata.py over the nine sequential UCIT cells).  A missing file or
key aborts with a message naming the key path -- no silent defaults, nothing is
ever fabricated, and nothing is transcribed from prose.

matplotlib only; house style from fig_style.py; Python 3.9 compatible.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_style as FS

FS.apply_style()

# POPE's three negative-sampling regimes, in increasing order of difficulty.
STRATA = ("random", "popular", "adversarial")
STRATUM_LABEL = {"random": "random", "popular": "popular",
                 "adversarial": "adversarial"}

# (readout key, axis label, panel title, value format)
#
# All three are per-cell statistics over the SAME six stages, averaged over the
# SAME nine cells: (a) and (b) are mean LEVELS, (c) is a PATH over the same
# stages.  |c| rather than c because c is signed and d' is not, so the absolute
# value is the level statistic that matches mean d'.
PANELS = (
    ("mean_dprime", "mean $d'$",
     "(a) discrimination: responds to difficulty", "%.3f"),
    ("mean_abs_c", "mean $|c|$",
     "(b) criterion LEVEL: responds to it too", "%.3f"),
    ("mean_path_c", "$\\Sigma|\\Delta c|$",
     "(c) criterion DRIFT: does not respond", "%.3f"),
)


def die(msg):
    raise SystemExit("[make_fig_pope_strata] ERROR: %s" % msg)


def jget(obj, keys, fname):
    """Strict nested lookup; a missing key aborts naming file + full key path."""
    cur = obj
    for i, k in enumerate(keys):
        if not isinstance(cur, dict) or k not in cur:
            die("missing key '%s' in %s (no silent defaults; refusing to "
                "fabricate a value)"
                % (" -> ".join(str(x) for x in keys[:i + 1]), fname))
        cur = cur[k]
    return cur


def jnum(obj, keys, fname):
    v = jget(obj, keys, fname)
    try:
        return float(v)
    except (TypeError, ValueError):
        die("key '%s' in %s is not a number (got %r)"
            % (" -> ".join(str(x) for x in keys), fname, v))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--readout",
                    default=os.path.join(here, "readout", "pope_strata_seq.json"),
                    help="per-stratum readout from analysis/pope_strata.py")
    ap.add_argument("--out", default=os.path.join(here, "out"),
                    help="output dir for figures")
    args = ap.parse_args()

    if not os.path.isfile(args.readout):
        die("readout not found: %s -- run analysis/pope_strata.py first "
            "(it re-scores the existing generations per stratum; no GPU)"
            % args.readout)
    fname = os.path.basename(args.readout)
    with open(args.readout) as f:
        try:
            rep = json.load(f)
        except ValueError as e:
            die("could not parse %s as JSON: %s" % (args.readout, e))

    arm = jget(rep, ["arm"], fname)
    n_cells = int(jnum(rep, ["n_cells"], fname))
    if arm != "seq":
        die("this figure states a property of the SEQUENTIAL arm; %s holds "
            "arm=%r. Refusing to relabel another arm's data." % (fname, arm))

    # Every panel's series, and the per-stratum cell count behind it.
    series = {}
    ns = []
    for key, _, _, _ in PANELS:
        series[key] = [jnum(rep, ["strata", s, key], fname) for s in STRATA]
    for s in STRATA:
        ns.append(int(jnum(rep, ["strata", s, "n"], fname)))
    if len(set(ns)) != 1:
        die("the three strata rest on different cell counts %r; a cross-stratum "
            "comparison would not be like-for-like" % (ns,))
    n_per = ns[0]
    if n_per != n_cells:
        die("per-stratum n=%d disagrees with n_cells=%d in %s"
            % (n_per, n_cells, fname))

    # One COMMON y-span for all three panels so the excursions are directly
    # comparable; each panel is centred on its own series.  The span is set by
    # the widest series plus headroom for that panel's value labels.
    spans = [max(v) - min(v) for v in series.values()]
    span = max(spans) * 1.85

    fig, axes = plt.subplots(3, 1, figsize=(FS.COL_W, FS.h_col(4.05)), sharex=True)
    xs = list(range(3))

    for ax, (key, ylab, title, vfmt) in zip(axes, PANELS):
        vals = series[key]
        st = FS.arm_line("SEQ", linewidth=1.4, markersize=4.2)
        ax.plot(xs, vals, **st)
        mid = 0.5 * (max(vals) + min(vals))
        ax.set_ylim(mid - span / 2.0, mid + span / 2.0)
        ax.set_xlim(-0.42, 2.42)

        # Value labels, placed on the side of the line with the most room.
        for x, v in zip(xs, vals):
            above = v <= mid
            ax.annotate(vfmt % v, xy=(x, v), xytext=(0, 6.0 if above else -6.5),
                        textcoords="offset points", ha="center",
                        va="bottom" if above else "top", fontsize=6.4,
                        color=FS.SEQ_COLOR, fontweight="bold")

        # The one statistic compared across panels: this panel's own spread.
        ax.text(0.985, 0.90, "spread %.3f" % (max(vals) - min(vals)),
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                color="#2a2a2a",
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#999999",
                          lw=0.5, alpha=0.92))
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=7.2)

    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels([STRATUM_LABEL[s] for s in STRATA])
    axes[-1].set_xlabel("POPE negative sampling  (easy $\\rightarrow$ hard)")

    fig.text(0.5, 0.006,
             "sequential arm, %d cells; all panels share one y-span, in $z$ units"
             % n_cells,
             ha="center", va="bottom", fontsize=6.0, color="#555555")

    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=0.85)
    rendered = []
    FS.save_fig(fig, args.out, "fig_pope_strata", rendered,
                "make_fig_pope_strata")
    print("[make_fig_pope_strata] DONE: %s (arm=%s, n=%d cells)"
          % (", ".join(rendered), arm, n_cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
