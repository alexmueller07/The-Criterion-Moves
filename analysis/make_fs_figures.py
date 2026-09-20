"""Full-study figure pipeline (FULLSTUDY_PREREG.md), layout-agnostic, headless.

USAGE:  python analysis/make_fs_figures.py [--agg AGG.json] [--out DIR]
            [--results ... --coco_gt ... --pope_dir ... --manifest_ucit ...
             --manifest_pilot ... --bench ...]   (same flags as fs_aggregate.py)
            [--backbones llava15,qwen25vl]
Defaults come from fs_common.find_layout (local repo or cluster); --agg defaults
to the layout's aggregate path, --out to <layout out_dir>/figures.

DATA SOURCE: the sibling fs_aggregate.json (primary). If it is absent, the
matrix is recomputed in-process with fs_aggregate.discover (identical scorers;
CHAIR only when --coco_gt resolves), so a figure can never disagree with the
aggregate by construction. Nothing is imputed: a partial matrix renders a
partial, clearly labelled figure.

Figures (each .png at 300 dpi + vector .pdf at final printed size; fig_style):
  fig_fs_generalization    UCIT: criterion c, SEQ vs anchor-v2, one panel per
                           backbone; thin = order x seed cell, bold = mean.
  fig_fs_mechanism_pilot   the same on the answer-statistics-varied pilot
                           suite (Qwen ps* arms; the mechanism suite).
  fig_fs_seed_consistency  E2: post-settling Sigma|dc| SEQ vs anchor, paired
                           per matched cell, every backbone x suite.
  fig_fs_dprime_falsifier  d' by stage for every arm per backbone (UCIT) with
                           the F1 band (base d' +/- 0.3) and its verdict.
  fig_fs_lamF_sweep        anchor-strength sweep: Sigma|dc| and endpoint
                           CHAIR_i@60 vs lambda_F (lamF_* arms + the prereg
                           anchor at lambda_F = 0.1), SEQ band for reference.
  fig_fs_scorecard         Sigma|dc| per arm (mean + per-cell dots) for every
                           backbone x suite with a scorecard.
matplotlib only; Okabe-Ito palette; Python 3.9.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import fs_common as FC     # noqa: E402
import fig_style as FS     # noqa: E402

FS.apply_style()
F1_MARGIN = 0.3   # pre-registered falsifier band half-width (FULLSTUDY_PREREG F1)

# arm token -> style key.  anchor-v1 = the CE-on-counterfactuals ablation (cecf).
ARM_KEY = {"seq": "SEQ", "joint": "JOINT", "er": "ER", "ewc": "EWC", "lwf": "LwF",
           "cecf": "V1", "anchor": "V2", "er500": "ER500"}
ARM_LABEL = dict(FS.ARM_LABEL)
ARM_LABEL.update({"EWC": "EWC", "LwF": "LwF", "ER500": "ER-500"})
_EXTRA_LINE = {
    "EWC": dict(color="#56B4E9", marker="v", linestyle="-", linewidth=1.1, markersize=3.2, zorder=2.5),
    "LwF": dict(color="#B8A400", marker="P", linestyle="-", linewidth=1.1, markersize=3.2, zorder=2.5),
    "ER500": dict(color="#009E73", marker="^", linestyle="--", linewidth=1.0, markersize=3.2, zorder=2.5),
}
_GENERIC = ["#000000", "#F0E442", "#CC79A7", "#56B4E9", "#E69F00", "#009E73"]
DPRIME_ARMS = ["seq", "joint", "er", "er500", "ewc", "lwf", "cecf", "anchor"]
BACKBONE_LABEL = FC.BACKBONE_LABEL
SUITE_LABEL = {"ucit": "UCIT", "pilot": "pilot suite"}


def arm_label(token):
    key = ARM_KEY.get(token)
    if key:
        return ARM_LABEL[key]
    if token.startswith("lamF_"):
        return "anchor $\\lambda_F$=%s" % token[5:]
    if token.startswith("mth_"):
        return "cand. %s" % token[4:]
    return token


def arm_style(token, **over):
    key = ARM_KEY.get(token)
    if key in FS._ARM_LINE:
        return FS.arm_line(key, **over)
    if key in _EXTRA_LINE:
        st = dict(_EXTRA_LINE[key])
    else:
        st = dict(color=_GENERIC[sum(map(ord, token)) % len(_GENERIC)], marker="s",
                  linestyle="-.", linewidth=1.0, markersize=3.0, zorder=2.2)
    st.update(over)
    return st


def die(msg):
    raise SystemExit("[make_fs_figures] ERROR: %s" % msg)


# ============================================================ data layer =====
# matrix[bb] = {"base": {c,dprime,yes,chair_i,chair_i60}|None,
#               "suites": {suite: {(token, order, seed): {stage:int -> {...}}}},
#               "scorecard": {suite: [rows]}}

def _slice(cell):
    p, ch = cell.get("pope"), cell.get("chair")
    if not p:
        return None
    return {"c": float(p["c"]), "dprime": float(p["dprime"]), "yes": float(p["yes_rate"]),
            "chair_i": (float(ch["chair_i"]) if ch else None),
            "chair_i60": (float(ch["chair_i60"]) if ch and "chair_i60" in ch else None)}


def _parse_key(armkey):
    tok, order, seed = armkey.split("|")
    return tok, order, int(seed.lstrip("sS"))


def load_from_aggregate(agg, backbones):
    matrix = {}
    n_stages = {s: v.get("n_stages") for s, v in agg.get("suites", {}).items()}
    for bb in backbones:
        entry = {"base": None, "suites": {}, "scorecard": {}}
        bd = agg.get("backbones", {}).get(bb)
        if bd:
            if bd.get("base"):
                entry["base"] = _slice(bd["base"])
            suites = bd.get("suites") or {"ucit": {"arms": bd.get("arms", {}),
                                                   "scorecard": []}}
            for suite, sd in suites.items():
                cells = {}
                for armkey, stages in sd.get("arms", {}).items():
                    try:
                        tok, order, seed = _parse_key(armkey)
                    except ValueError:
                        print("[make_fs_figures] skip unrecognized arm key %r" % armkey)
                        continue
                    st = {int(k): _slice(v) for k, v in stages.items()
                          if str(k).isdigit() and _slice(v)}
                    if st:
                        cells[(tok, order, seed)] = st
                entry["suites"][suite] = cells
                entry["scorecard"][suite] = sd.get("scorecard") or []
        matrix[bb] = entry
    return matrix, n_stages


def load_from_results(args, backbones):
    """Fallback: recompute with the aggregator's own discovery (no aggregate)."""
    import fs_aggregate as FA
    suites = {"ucit": FC.load_suite("ucit", args.manifest_ucit),
              "pilot": FC.load_suite("pilot", args.manifest_pilot)}
    coco_gt = FC.load_coco_gt(args.coco_gt)
    mx, _ip, _un, malformed, _notes = FA.discover(args.results, FC.parse_family_backbone(None),
                                                  coco_gt, suites, {})
    if malformed:
        print("[make_fs_figures] WARNING: %d malformed cells excluded" % len(malformed))
    matrix = {}
    for bb in backbones:
        entry = {"base": None, "suites": {}, "scorecard": {}}
        if bb in mx:
            if mx[bb]["base"]:
                entry["base"] = _slice(mx[bb]["base"])
            for suite, sd in mx[bb]["suites"].items():
                cells = {}
                for key, stages in sd["arms"].items():
                    st = {k: _slice(v) for k, v in stages.items() if _slice(v)}
                    if st:
                        cells[key] = st
                entry["suites"][suite] = cells
                entry["scorecard"][suite] = FA.build_scorecard(mx[bb]["base"], sd["arms"],
                                                               suites[suite])
        matrix[bb] = entry
    return matrix, {s: suites[s]["n_stages"] for s in suites}


def load_matrix(args, backbones):
    if args.agg and os.path.isfile(args.agg):
        with open(args.agg) as f:
            agg = json.load(f)
        print("[make_fs_figures] source: aggregate %s" % args.agg)
        m, n = load_from_aggregate(agg, backbones)
        return m, n, "aggregate"
    if not args.results or not os.path.isdir(args.results):
        die("no aggregate at %s and results dir %r missing -- nothing to render"
            % (args.agg, args.results))
    print("[make_fs_figures] source: direct recompute from %s (no aggregate)" % args.results)
    m, n = load_from_results(args, backbones)
    return m, n, "results_fs"


# ---------- helpers over the normalized matrix -------------------------------

def arm_cells(matrix, bb, suite, token, field):
    """{(order,seed): [(stage, value), ...]} with a leading stage-0 base point."""
    b = matrix[bb]
    out = {}
    base_v = b["base"][field] if b["base"] and b["base"].get(field) is not None else None
    for (tok, order, seed), st_map in b["suites"].get(suite, {}).items():
        if tok != token:
            continue
        pts = [(0, base_v)] if base_v is not None else []
        for stage in sorted(st_map):
            v = st_map[stage].get(field)
            if v is not None:
                pts.append((stage, v))
        if pts:
            out[(order, seed)] = pts
    return out


def mean_trajectory(cells):
    acc = {}
    for pts in cells.values():
        for s, v in pts:
            acc.setdefault(s, []).append(v)
    stages = sorted(acc)
    return stages, [float(np.mean(acc[s])) for s in stages], [len(acc[s]) for s in stages]


def _post_settle(pts):
    cs = [v for _, v in sorted(pts)]
    if len(cs) < 3:
        return None
    d = [abs(cs[i + 1] - cs[i]) for i in range(len(cs) - 1)]
    return sum(d[1:])


# ------------------------------------------- (1)/(2) c trajectory per suite

def fig_c_trajectory(matrix, backbones, suite, out_dir, rendered, name, title):
    bbs = [bb for bb in backbones if matrix[bb]["suites"].get(suite)] or backbones[:1]
    fig, axes = plt.subplots(1, len(bbs), figsize=(FS.FULL_W if len(bbs) > 1 else FS.COL_W,
                                     FS.h_full(2.55) if len(bbs) > 1 else FS.h_col(2.55)),
                             sharey=True, squeeze=False)
    axes = axes[0]
    any_data = False
    for ax, bb in zip(axes, bbs):
        base = matrix[bb]["base"]
        panel_has = False
        for tok in ("seq", "anchor"):
            cells = arm_cells(matrix, bb, suite, tok, "c")
            if not cells:
                print("[make_fs_figures] %s: %s/%s %s -- no cells yet" % (name, bb, suite, tok))
                continue
            panel_has = any_data = True
            thin = arm_style(tok, alpha=0.35, linewidth=0.7, markersize=2.2)
            thin.pop("zorder", None)
            for pts in cells.values():
                ax.plot([s for s, _ in pts], [v for _, v in pts], **thin)
            stages, means, _ = mean_trajectory(cells)
            ax.plot(stages, means, label="%s (mean of %d cell%s)" % (
                arm_label(tok), len(cells), "" if len(cells) == 1 else "s"),
                **arm_style(tok, linewidth=2.0, markersize=3.6, zorder=5))
        if base is not None:
            ax.axhline(base["c"], color=FS.BASE_COLOR, ls=":", lw=0.8, zorder=1, label="base $c$")
        ax.set_title("%s / %s" % (BACKBONE_LABEL.get(bb, bb), SUITE_LABEL.get(suite, suite)),
                     fontsize=8)
        ax.set_xlabel("stage (0 = base backbone)")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if not panel_has:
            ax.text(0.5, 0.5, "no cells yet", transform=ax.transAxes, ha="center",
                    va="center", fontsize=8, color="#999999")
        else:
            ax.legend(loc="best", fontsize=6.0)
    axes[0].set_ylabel("criterion $c$ (pooled POPE)")
    fig.suptitle(title, fontsize=8.5)
    if not any_data:
        print("[make_fs_figures] %s: NO cells on any backbone yet" % name)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FS.save_fig(fig, out_dir, name, rendered, "make_fs_figures")


# ------------------------------------------------------ (3) seed consistency

def fig_fs_seed_consistency(matrix, backbones, out_dir, rendered):
    fig, ax = plt.subplots(figsize=(FS.COL_W, FS.h_col(2.7)))
    markers = {"llava15": "o", "qwen25vl": "s"}
    pairs = []
    n_seq = n_anc = 0
    for bb in backbones:
        for suite in sorted(matrix[bb]["suites"]):
            seq = arm_cells(matrix, bb, suite, "seq", "c")
            anc = arm_cells(matrix, bb, suite, "anchor", "c")
            n_seq += len(seq)
            n_anc += len(anc)
            for key in sorted(set(seq) & set(anc)):
                s, a = _post_settle(seq[key]), _post_settle(anc[key])
                if s is not None and a is not None:
                    pairs.append((bb, suite, key, s, a))
    if not pairs:
        ax.text(0.5, 0.55, "no matched SEQ/anchor cells yet", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#999999")
        ax.text(0.5, 0.40, "SEQ cells: %d   anchor cells: %d   (need >=2 stages each)"
                % (n_seq, n_anc), transform=ax.transAxes, ha="center", va="center",
                fontsize=7, color="#999999")
        print("[make_fs_figures] seed_consistency: 0 matched pairs -- placeholder")
    n_down = 0
    seen = set()
    for bb, suite, key, s, a in pairs:
        down = a < s
        n_down += down
        col = FS.ER_COLOR if down else FS.V1_COLOR
        mk = markers.get(bb, "o")
        lbl = None
        tag = (bb, suite)
        if tag not in seen:
            lbl = "%s / %s" % (BACKBONE_LABEL.get(bb, bb), SUITE_LABEL.get(suite, suite))
            seen.add(tag)
        ls = "-" if suite == "ucit" else "--"
        ax.plot([0, 1], [s, a], ls, color=col, lw=0.8, alpha=0.75, zorder=2)
        ax.plot([0], [s], marker=mk, color=FS.SEQ_COLOR, markersize=4.0, linestyle="none",
                zorder=3, label=lbl, markerfacecolor=FS.SEQ_COLOR if suite == "ucit" else "white")
        ax.plot([1], [a], marker=mk, color=FS.V2_COLOR, markersize=4.0, linestyle="none",
                zorder=3, markerfacecolor=FS.V2_COLOR if suite == "ucit" else "white")
    ax.set_xlim(-0.35, 1.35)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["SEQ", "anchor v2"])
    ax.set_ylabel("post-settling $\\Sigma|\\Delta c|$")
    ax.set_title("Anchor suppresses criterion motion per cell (E2)", fontsize=8)
    if pairs:
        ax.text(0.03, 0.98, "anchor $<$ SEQ in %d / %d matched cells" % (n_down, len(pairs)),
                transform=ax.transAxes, ha="left", va="top", fontsize=7,
                color=FS.ER_COLOR if n_down == len(pairs) else "#333333")
        ax.legend(loc="lower left", fontsize=6.0, title="cell (open marker = pilot suite)",
                  title_fontsize=6.0)
        print("[make_fs_figures] seed_consistency: %d matched pairs, anchor<SEQ in %d"
              % (len(pairs), n_down))
    fig.tight_layout()
    FS.save_fig(fig, out_dir, "fig_fs_seed_consistency", rendered, "make_fs_figures")


# ------------------------------------------------------- (4) d' falsifier band

def fig_fs_dprime_falsifier(matrix, backbones, out_dir, rendered, suite="ucit"):
    n = len(backbones)
    fig, axes = plt.subplots(1, n, figsize=(FS.FULL_W, FS.h_full(2.7)), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, bb in zip(axes, backbones):
        base = matrix[bb]["base"]
        drew = False
        handles, labels = [], []
        if base is not None:
            b = base["dprime"]
            ax.axhspan(b - F1_MARGIN, b + F1_MARGIN, color="#DDDDDD", alpha=0.55, zorder=0)
            ax.axhline(b, color=FS.BASE_COLOR, ls=":", lw=0.8, zorder=1)
            ax.text(0.02, b + F1_MARGIN, "F1 band: base $d'\\pm%.1f$" % F1_MARGIN,
                    transform=ax.get_yaxis_transform(), va="bottom", ha="left",
                    fontsize=6.0, color="#666666")
        seq_cells = arm_cells(matrix, bb, suite, "seq", "dprime")
        seq_breach = None
        if seq_cells and base is not None:
            seq_breach = False
            thin = arm_style("seq", alpha=0.4, linewidth=0.7, markersize=2.2)
            thin.pop("zorder", None)
            for pts in seq_cells.values():
                ax.plot([s for s, _ in pts], [v for _, v in pts], **thin)
                for s, v in pts:
                    if s > 0 and abs(v - base["dprime"]) > F1_MARGIN:
                        seq_breach = True
        present = sorted({k[0] for k in matrix[bb]["suites"].get(suite, {})},
                         key=FC.arm_sort_key)
        for tok in [t for t in DPRIME_ARMS if t in present] + \
                [t for t in present if t not in DPRIME_ARMS and not t.startswith("single_")]:
            cells = arm_cells(matrix, bb, suite, tok, "dprime")
            if not cells:
                continue
            drew = True
            stages, means, _ = mean_trajectory(cells)
            st = arm_style(tok, linewidth=1.8 if tok == "seq" else 1.1,
                           markersize=3.4 if tok == "seq" else 3.0,
                           zorder=6 if tok == "seq" else 3)
            (ln,) = ax.plot(stages, means, **st)
            handles.append(ln)
            labels.append(arm_label(tok))
        ax.set_title("%s / %s" % (BACKBONE_LABEL.get(bb, bb), SUITE_LABEL[suite]), fontsize=8)
        ax.set_xlabel("stage (0 = base backbone)")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        if not drew:
            ax.text(0.5, 0.5, "no cells yet", transform=ax.transAxes, ha="center",
                    va="center", fontsize=8, color="#999999")
            continue
        if seq_breach is None:
            verdict, vcol = "F1: base d' not yet available", "#999999"
        elif seq_breach:
            verdict, vcol = "F1 BREACHED: SEQ d' left the band -- headline at risk", FS.V1_COLOR
        else:
            verdict, vcol = "F1 intact: SEQ d' within band", FS.ER_COLOR
        ax.text(0.5, 0.015, verdict, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=6.4, color=vcol, fontweight="bold")
        if handles:
            ax.legend(handles, labels, loc="upper right", fontsize=5.6, ncol=2,
                      columnspacing=0.8, handlelength=1.3)
        print("[make_fs_figures] dprime %s: arms=%s seq_breach=%s" % (bb, labels, seq_breach))
    axes[0].set_ylabel("$d'$ (pooled POPE)")
    fig.suptitle("Grounding ($d'$) stays flat -- pre-registered F1 falsifier", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FS.save_fig(fig, out_dir, "fig_fs_dprime_falsifier", rendered, "make_fs_figures")


# ------------------------------------------------------- (5) lambda_F sweep

def _sc_rows(matrix, bb, suite):
    return matrix[bb].get("scorecard", {}).get(suite) or []


def fig_fs_lamF_sweep(matrix, backbones, out_dir, rendered, suite="ucit"):
    fig, axes = plt.subplots(1, 2, figsize=(FS.FULL_W, FS.h_full(2.6)), squeeze=False)
    ax1, ax2 = axes[0]
    drew = False
    for bb in backbones:
        rows = _sc_rows(matrix, bb, suite)
        pts = []
        for r in rows:
            if r["arm"].startswith("lamF_") and r.get("faith_weight") is not None:
                pts.append((r["faith_weight"], r, "lamF"))
            elif r["arm"] == "anchor":
                pts.append((0.1, r, "anchor"))   # prereg anchor: lambda_F = 0.1
        seq = next((r for r in rows if r["arm"] == "seq"), None)
        pts = [p for p in pts if p[1].get("sum_abs_dc")]
        if not pts:
            continue
        drew = True
        mk = "o" if bb == "llava15" else "s"
        for ax, key, ylab in ((ax1, "sum_abs_dc", "$\\Sigma|\\Delta c|$ (base $\\to$ endpoint)"),
                              (ax2, "endpoint_chair_i60", "endpoint CHAIR$_i$@60")):
            xs, ys = [], []
            for w, r, kind in sorted(pts, key=lambda t: t[0]):
                v = r.get(key)
                if not v:
                    continue
                for cv in v["values"]:
                    ax.plot([w], [cv], marker=mk, linestyle="none", markersize=2.6,
                            color=FS.V2_COLOR, alpha=0.45, zorder=3)
                xs.append(w)
                ys.append(v["mean"])
                if kind == "anchor":
                    ax.plot([w], [v["mean"]], marker="D", linestyle="none", markersize=4.2,
                            color=FS.V2_COLOR, zorder=5,
                            label="prereg anchor ($\\lambda_F$=0.1)" if ax is ax1 else None)
            if xs:
                ax.plot(xs, ys, "-", color=FS.V2_COLOR, lw=1.2, zorder=4,
                        label="%s (mean per $\\lambda_F$)" % BACKBONE_LABEL.get(bb, bb) if ax is ax1 else None)
            if seq and seq.get(key):
                lo, hi = seq[key]["range"]
                ax.axhspan(lo, hi, color=FS.SEQ_COLOR, alpha=0.12, zorder=1)
                ax.axhline(seq[key]["mean"], color=FS.SEQ_COLOR, ls="--", lw=0.8, zorder=2,
                           label="SEQ (mean, band = cell range)" if ax is ax1 else None)
            ax.set_xscale("log")
            ax.set_xlabel("anchor strength $\\lambda_F$")
            ax.set_ylabel(ylab)
    if not drew:
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "no lamF_* / anchor cells yet", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="#999999")
        print("[make_fs_figures] lamF_sweep: no cells yet -- placeholder")
    else:
        ax1.legend(loc="best", fontsize=5.8)
    fig.suptitle("Anchor-strength sweep (calibration-theory prediction: $|\\Delta c|\\sim 1/\\lambda_F$)",
                 fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FS.save_fig(fig, out_dir, "fig_fs_lamF_sweep", rendered, "make_fs_figures")


# ------------------------------------------------------- (6) scorecard bars

def fig_fs_scorecard(matrix, backbones, out_dir, rendered):
    panels = [(bb, suite) for bb in backbones for suite in sorted(matrix[bb].get("scorecard", {}))
              if any(r.get("sum_abs_dc") for r in _sc_rows(matrix, bb, suite))]
    n = max(1, len(panels))
    fig, axes = plt.subplots(1, n, figsize=(min(FS.FULL_W, 2.2 * n + 1.0), FS.h_full(2.7)), squeeze=False)
    axes = axes[0]
    if not panels:
        axes[0].text(0.5, 0.5, "no complete cells yet", transform=axes[0].transAxes,
                     ha="center", va="center", fontsize=8, color="#999999")
        print("[make_fs_figures] scorecard: no complete cells -- placeholder")
    for ax, (bb, suite) in zip(axes, panels):
        rows = [r for r in _sc_rows(matrix, bb, suite) if r.get("sum_abs_dc")]
        ys = np.arange(len(rows))
        for y, r in zip(ys, rows):
            st = arm_style(r["arm"])
            ax.barh(y, r["sum_abs_dc"]["mean"], color=st["color"], edgecolor="black",
                    linewidth=0.4, height=0.62, zorder=2,
                    hatch="///" if r["arm"] == "cecf" else None)
            if r.get("post_settle"):
                ax.barh(y, r["post_settle"]["mean"], color="white", alpha=0.55,
                        height=0.62, zorder=3, edgecolor="none")
            for v in r["sum_abs_dc"]["values"]:
                ax.plot([v], [y], marker="|", color="black", markersize=5, zorder=4)
        ax.set_yticks(ys)
        ax.set_yticklabels([arm_label(r["arm"]) + " (n=%d)" % r["n_complete"] for r in rows],
                           fontsize=6.2)
        ax.invert_yaxis()
        ax.set_xlabel("$\\Sigma|\\Delta c|$ (bar = mean; ticks = cells; pale = post-settling)")
        ax.set_title("%s / %s" % (BACKBONE_LABEL.get(bb, bb), SUITE_LABEL.get(suite, suite)),
                     fontsize=8)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Method scorecard: criterion motion per arm (complete cells)", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FS.save_fig(fig, out_dir, "fig_fs_scorecard", rendered, "make_fs_figures")


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agg", default=None, help="fs_aggregate.json [layout agg_default]")
    FC.add_layout_args(ap)
    ap.add_argument("--out", default=None, help="output dir [layout out_dir/figures]")
    ap.add_argument("--backbones", default="llava15,qwen25vl", help="comma list; panel order")
    args = ap.parse_args()
    lay = FC.resolve_layout_args(args)
    if not args.agg:
        args.agg = str(lay["agg_default"])
    else:
        args.agg = os.path.abspath(os.path.expanduser(args.agg))
    out_dir = os.path.abspath(args.out) if args.out else str(lay["out_dir"] / "figures")
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    for b in backbones:
        if b not in FC.BASE_DIRS:
            die("unknown backbone %r (known: %s)" % (b, ", ".join(FC.BASE_DIRS)))
    os.makedirs(out_dir, exist_ok=True)

    matrix, n_stages, source = load_matrix(args, backbones)
    print("[make_fs_figures] matrix coverage (source=%s):" % source)
    for bb in backbones:
        base = "yes" if matrix[bb]["base"] is not None else "MISSING"
        segs = ["base=%s" % base]
        for suite, cells in sorted(matrix[bb]["suites"].items()):
            counts = {}
            for (tok, _o, _s) in cells:
                counts[tok] = counts.get(tok, 0) + 1
            segs.append("%s[%s]" % (suite, " ".join("%s=%d" % (t, counts[t])
                                                     for t in sorted(counts, key=FC.arm_sort_key))))
        print("  %-9s %s" % (bb, "  ".join(segs)))

    rendered = []
    fig_c_trajectory(matrix, backbones, "ucit", out_dir, rendered, "fig_fs_generalization",
                     "Criterion drift and anchor stabilization across the UCIT sequence")
    fig_c_trajectory(matrix, backbones, "pilot", out_dir, rendered, "fig_fs_mechanism_pilot",
                     "Mechanism suite (answer-statistics-varied): criterion trajectory")
    fig_fs_seed_consistency(matrix, backbones, out_dir, rendered)
    fig_fs_dprime_falsifier(matrix, backbones, out_dir, rendered)
    fig_fs_lamF_sweep(matrix, backbones, out_dir, rendered)
    fig_fs_scorecard(matrix, backbones, out_dir, rendered)
    print("[make_fs_figures] DONE: %d figure(s) in %s: %s" % (len(rendered), out_dir, ", ".join(rendered)))


if __name__ == "__main__":
    main()
