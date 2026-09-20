"""SOTA-comparison table generator (Park request #2), layout-agnostic.

USAGE:  python analysis/make_sota_table.py [--agg AGG.json] [--out DIR]
            [--results ... --bench ... --coco_gt ... --pope_dir ...
             --manifest_ucit ... --manifest_pilot ...]   (fs_aggregate.py flags)
            [--backbone llava15] [--suite ucit] [--order o1] [--seed 17]
Defaults come from fs_common.find_layout; --out defaults to <layout out_dir>/tables.

Emits <out>/sota.tex (booktabs) + sota_notes.tex + sota.csv for the canonical
cell (LLaVA-1.5-7B, UCIT, ordering o1, seed 17 -- the cell the endpoint
external-benchmark job runs on), and additionally sota_<backbone>_<suite>.*
for every other backbone x suite in the aggregate that has the same
order/seed cell. Rows: SEQ, JOINT, ER, EWC, LwF, anchor-v1 (cecf ablation),
anchor-v2, then every extra arm found in that cell (ER-500, lamF_*, mth_*),
plus the base backbone as a reference row (never bolded).

Columns (judge-free; direction drives which value is bolded as best):
  POPE-F1          exact 'all'-category F1 (aggregate pope.f1; raw fallback)
  CHAIR_i          COCO-500, per-mention, vendored synonym list
  CHAIR_i@60       the same over the first 60 words (length-controlled axis)
  ObjHal CHAIR_i   Object HalBench detail prompts (results_fs_bench objhal.json)
  non-COCO CHAIR_i Open Images CHAIR (noncoco.json; the domain-overlap control)
  MME-Hall         existence+count+position+color score (mme.json 'all' row)
  c range          max - min of criterion c over base + every stage
  plasticity       mean accuracy on the just-trained task (UCIT: closed-ended
                   tasks only; open-ended caption tasks excluded, footnoted)
'--' = not-yet-available (nothing imputed). The CSV carries raw numbers, empty
for missing, plus audit flags per row.

DATA SOURCE: fs_aggregate.json (primary; bench block included). Without it the
cell is recomputed in-process via fs_aggregate.discover (identical scorers) and
the bench dir is read directly.

AUDIT caveat (emitted as table notes): per this study's own finding, the raw
sequential-vs-joint endpoint hallucination gap is largely a CRITERION-DRIFT
artifact, not grounding loss; the c-range column is that decomposition axis and
the benchmark columns must not be read as pure grounding degradation.

Python 3.9; stdlib + the pilot scorers (no matplotlib).
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import fs_common as FC  # noqa: E402

METHODS = [("SEQ", "seq"), ("JOINT", "joint"), ("ER", "er"), ("EWC", "ewc"), ("LwF", "lwf"),
           ("anchor-v1", "cecf"), ("anchor-v2", "anchor")]
EXTRA_LABEL = {"er500": "ER-500"}
COLUMNS = [
    ("pope_f1",       "POPE-F1 $\\uparrow$",               +1, "%.3f"),
    ("chair_i",       "CHAIR$_i$ $\\downarrow$",           -1, "%.3f"),
    ("chair_i60",     "CHAIR$_i$@60 $\\downarrow$",        -1, "%.3f"),
    ("objhal_chair",  "ObjHal CHAIR$_i$ $\\downarrow$",    -1, "%.3f"),
    ("noncoco_chair", "non-COCO CHAIR$_i$ $\\downarrow$",  -1, "%.3f"),
    ("mme_hall",      "MME-Hall $\\uparrow$",              +1, "%.1f"),
    ("c_range",       "$c$ range $\\downarrow$",           -1, "%.3f"),
    ("plasticity",    "plasticity $\\uparrow$",            +1, "%.3f"),
]


def die(msg):
    raise SystemExit("[make_sota_table] ERROR: %s" % msg)


def extra_label(token):
    if token in EXTRA_LABEL:
        return EXTRA_LABEL[token]
    if token.startswith("lamF_"):
        return "anchor-v2 ($\\lambda_F$=%s)" % token[5:]
    if token.startswith("mth_"):
        return "cand.\\ %s" % token[4:].replace("_", "\\_")
    return token.replace("_", "\\_")


# ------------------------------------------------------------- data access

def load_aggregate_or_recompute(args):
    """-> (agg_like dict, source). Without an aggregate the matrix is rebuilt
    in-process with the aggregator's discovery so every number is the same."""
    if args.agg and os.path.isfile(args.agg):
        with open(args.agg) as f:
            agg = json.load(f)
        print("[make_sota_table] source: aggregate %s" % args.agg)
        return agg, "aggregate"
    if not args.results or not os.path.isdir(args.results):
        die("no aggregate at %s and results dir %r missing" % (args.agg, args.results))
    print("[make_sota_table] source: recompute from %s (no aggregate)" % args.results)
    import fs_aggregate as FA
    suites = {"ucit": FC.load_suite("ucit", args.manifest_ucit),
              "pilot": FC.load_suite("pilot", args.manifest_pilot)}
    coco_gt = FC.load_coco_gt(args.coco_gt)
    mx, _ip, _un, malformed, _notes = FA.discover(args.results, FC.parse_family_backbone(None),
                                                  coco_gt, suites, {})
    if malformed:
        print("[make_sota_table] WARNING: %d malformed cells excluded" % len(malformed))
    agg = {"suites": {s: {"n_stages": suites[s]["n_stages"]} for s in suites},
           "bench": {}, "backbones": {}}
    for bb, bd in mx.items():
        rep = {"base": FC.strip_private(bd["base"]) if bd["base"] else None, "suites": {}}
        for suite, sd in bd["suites"].items():
            rep["suites"][suite] = {
                "arms": {"%s|%s|s%d" % (a, o, s): {str(k): FC.strip_private(v) for k, v in st.items()}
                         for (a, o, s), st in sd["arms"].items()},
                "scorecard": FA.build_scorecard(bd["base"], sd["arms"], suites[suite])}
        agg["backbones"][bb] = rep
    try:
        agg["bench"] = {k: v for k, v in FC.discover_bench(args.bench).items() if v}
    except FC.MalformedResult as e:
        print("[make_sota_table] WARNING: bench unreadable: %s" % e)
    return agg, "results_fs"


def suite_block(agg, bb, suite):
    bd = agg.get("backbones", {}).get(bb) or {}
    suites = bd.get("suites")
    if suites is not None:
        return suites.get(suite)
    if suite == "ucit" and bd.get("arms") is not None:     # pre-suite aggregate
        return {"arms": bd["arms"], "scorecard": []}
    return None


def bench_for(agg, args, label):
    b = (agg.get("bench") or {}).get(label)
    if b is None and args.bench and os.path.isdir(args.bench):
        try:
            b = FC.load_bench(args.bench, label)
        except FC.MalformedResult as e:
            print("[make_sota_table] WARNING: %s" % e)
            b = None
    return b


def raw_pope_f1(results_dir, cell_dir):
    p = os.path.join(results_dir or "", cell_dir or "", FC.POPE_FILE)
    if not results_dir or not cell_dir or not os.path.isfile(p):
        return None
    try:
        return FC.score_pope(p)["f1"]
    except FC.MalformedResult:
        return None


def derived_pope_f1(pope):
    try:
        H, FA = float(pope["H"]), float(pope["FA"])
        nyes, nno = float(pope["n_gt_yes"]), float(pope["n_gt_no"])
    except (KeyError, TypeError, ValueError):
        return None
    tp, fp, fn = H * nyes, FA * nno, (1.0 - H) * nyes
    prec, rec = tp / max(1e-9, tp + fp), tp / max(1e-9, tp + fn)
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else None


# ------------------------------------------------------------- per-row values

def fill_bench(v, notes, b):
    if not b:
        return
    if b.get("objhal") and b["objhal"].get("chair_i") is not None:
        v["objhal_chair"] = float(b["objhal"]["chair_i"])
    if b.get("noncoco") and b["noncoco"].get("chair_i") is not None:
        v["noncoco_chair"] = float(b["noncoco"]["chair_i"])
    if b.get("mme") and b["mme"].get("score") is not None:
        v["mme_hall"] = float(b["mme"]["score"])
        if b["mme"].get("full_800_scale") is False:
            notes.add("mme_partial")
    if not b.get("done"):
        notes.add("bench_in_progress")


def method_values(agg, args, bb, suite, token, n_stages):
    v = {c[0]: None for c in COLUMNS}
    notes = set()
    sb = suite_block(agg, bb, suite)
    stages = (sb or {}).get("arms", {}).get("%s|%s|s%d" % (token, args.order, args.seed))
    if not stages:
        return v, notes, None
    ks = sorted(int(k) for k in stages if str(k).isdigit())
    end = stages[str(ks[-1])]
    if n_stages and len(FC.contiguous_prefix({k: 1 for k in ks})) < n_stages:
        notes.add("partial_cell")
    runtag = end.get("runtag") or "%s%s_%s_%s_s%d" % (
        "ps" if suite == "pilot" else "fs", "L" if bb == "llava15" else "Q", token,
        args.order, args.seed)
    fill_bench(v, notes, bench_for(agg, args, FC.bench_label(runtag)))

    p = end.get("pope") or {}
    f1 = p.get("f1")
    if f1 is None:
        f1 = raw_pope_f1(args.results, end.get("dir"))
    if f1 is None and p:
        f1 = derived_pope_f1(p)
        if f1 is not None:
            notes.add("f1_derived")
    v["pope_f1"] = f1
    ch = end.get("chair") or {}
    if ch.get("chair_i") is not None:
        v["chair_i"] = float(ch["chair_i"])
    if ch.get("chair_i60") is not None:
        v["chair_i60"] = float(ch["chair_i60"])

    base = (agg.get("backbones", {}).get(bb) or {}).get("base") or {}
    cs = [float(stages[str(k)]["pope"]["c"]) for k in ks
          if stages[str(k)].get("pope") and stages[str(k)]["pope"].get("c") is not None]
    if base.get("pope") and base["pope"].get("c") is not None:
        cs = [float(base["pope"]["c"])] + cs
    else:
        notes.add("base_absent")
    if len(cs) >= 2:
        v["c_range"] = max(cs) - min(cs)
    elif cs:
        v["c_range"] = 0.0
        notes.add("c_range_singlepoint")

    if token != "joint":
        accs, n_open = [], 0
        for k in ks:
            sbk = stages[str(k)]
            tt = sbk.get("task_trained")
            tinfo = (sbk.get("tasks") or {}).get(tt) if tt else None
            if not tinfo:
                continue
            if suite == "ucit" and tinfo.get("open_ended"):
                n_open += 1
                continue
            if tinfo.get("acc") is not None:
                accs.append(float(tinfo["acc"]))
        if accs:
            v["plasticity"] = sum(accs) / len(accs)
            if n_open:
                notes.add("plasticity_open_excluded")
    else:
        notes.add("joint_no_plasticity")
    return v, notes, runtag


def base_values(agg, args, bb):
    v = {c[0]: None for c in COLUMNS}
    notes = {"base_row"}
    base = (agg.get("backbones", {}).get(bb) or {}).get("base")
    if not base:
        return v, notes
    p = base.get("pope") or {}
    v["pope_f1"] = p.get("f1") if p.get("f1") is not None else (
        raw_pope_f1(args.results, base.get("dir")) or derived_pope_f1(p) if p else None)
    ch = base.get("chair") or {}
    v["chair_i"] = ch.get("chair_i")
    v["chair_i60"] = ch.get("chair_i60")
    fill_bench(v, notes, bench_for(agg, args, FC.BASE_DIRS[bb]))
    return v, notes


# ------------------------------------------------------------- table emission

def best_per_column(rows):
    best = {}
    for key, _hdr, direction, _fmt in COLUMNS:
        cand = [(label, vals[key]) for label, vals, n in rows
                if vals[key] is not None and "base_row" not in n]
        if cand:
            best[key] = (max if direction > 0 else min)(cand, key=lambda t: t[1])[0]
    return best


def build_notes(rows, meta):
    notes = ["All columns are judge-free: POPE-F1 and MME-Hall are exact-match accuracies; "
             "CHAIR$_i$ (COCO-500, @60, Object-HalBench and non-COCO columns) uses the "
             "repository's vendored CHAIR synonym list (documented deviation from the "
             "canonical list).",
             "Per this study's decomposition, the raw sequential-vs-joint hallucination gap is "
             "largely a criterion-drift artifact, not grounding loss; the $c$-range column is "
             "that decomposition axis and the benchmark columns should not be read as pure "
             "grounding change.",
             "Cell: %s, %s suite, ordering %s, seed %d (the canonical endpoint-benchmark cell); "
             "'--' marks a not-yet-available metric (nothing imputed); the base row is the raw "
             "backbone and is never bolded." % (meta["backbone_label"], meta["suite"],
                                                 meta["order"], meta["seed"]),
             "anchor-v1 is the CE-on-counterfactuals ablation (RUNTAG token cecf); anchor-v2 is "
             "the full anchor; lamF rows are the anchor-strength sweep.",
             "MME-Hall is the object-hallucination slice score (max 800 across existence/count/"
             "position/color); non-COCO CHAIR$_i$ is the Open Images control (images and "
             "annotation pipeline disjoint from COCO)."]
    flags = set().union(*[n for _l, _v, n in rows]) if rows else set()
    if "plasticity_open_excluded" in flags:
        notes.append("Plasticity is the mean of each just-trained task's normalized-EM accuracy "
                     "over the CLOSED-ended tasks; the open-ended caption tasks (Flickr30k, "
                     "VizWiz) are not exact-match scorable and are excluded from the mean.")
    if "joint_no_plasticity" in flags:
        notes.append("JOINT has no just-trained-task diagonal, so plasticity is '--' for it.")
    if "f1_derived" in flags:
        notes.append("A POPE-F1 marked derived was reconstructed from H/FA and GT counts.")
    if "mme_partial" in flags:
        notes.append("An MME-Hall value is on a partial subtask set (< all four); not on the "
                     "full 800 scale.")
    if "partial_cell" in flags:
        notes.append("A row marked partial has not reached its final stage yet; its endpoint "
                     "columns are the latest available stage.")
    if "bench_in_progress" in flags:
        notes.append("An external-benchmark job for this cell has not written BENCH_DONE.")
    if "c_range_singlepoint" in flags:
        notes.append("A $c$-range of 0.000 reflects a single available checkpoint.")
    return notes


def write_tex(rows, best, meta, path):
    L = ["% Auto-generated by analysis/make_sota_table.py -- do not edit by hand.",
         "%% Cell: %s, %s, ordering %s, seed %d (the endpoint-benchmark cell)."
         % (meta["backbone_label"], meta["suite"], meta["order"], meta["seed"]),
         "% '--' = not-yet-available (nothing imputed); best per column in bold.",
         "\\begin{tabular}{l%s}" % ("r" * len(COLUMNS)), "\\toprule",
         " & ".join(["Method"] + [h for _k, h, _d, _f in COLUMNS]) + " \\\\", "\\midrule"]
    for label, vals, n in rows:
        cells = [label]
        for key, _h, _d, fmt in COLUMNS:
            val = vals[key]
            if val is None:
                cells.append("--")
                continue
            s = fmt % val
            if best.get(key) == label and "base_row" not in n:
                s = "\\textbf{%s}" % s
            cells.append(s)
        L.append(" & ".join(cells) + " \\\\")
        if "base_row" in n:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    notes = build_notes(rows, meta)
    L.append("%% --- audit notes (judge-free; decomposition caveat) ---")
    L += ["%% %s" % nl for nl in notes]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    npath = os.path.splitext(path)[0] + "_notes.tex"
    with open(npath, "w") as f:
        f.write("%% Auto-generated audit notes for %s.\n{\\footnotesize\n" % os.path.basename(path))
        for nl in notes:
            f.write(nl + "\\\\\n")
        f.write("}\n")
    print("[make_sota_table] wrote %s (+ %s)" % (path, npath))


def write_csv(rows, meta, path):
    fields = ["method", "runtag"] + [k for k, _h, _d, _f in COLUMNS] + \
             ["backbone", "suite", "order", "seed", "notes"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for label, vals, n in rows:
            row = {"method": label.replace("\\", ""), "runtag": n.get("_runtag", "") if isinstance(n, dict) else "",
                   "backbone": meta["backbone"], "suite": meta["suite"],
                   "order": meta["order"], "seed": meta["seed"],
                   "notes": ";".join(sorted(x for x in n if not x.startswith("_")))}
            for key, _h, _d, _f in COLUMNS:
                row[key] = "" if vals[key] is None else repr(round(vals[key], 6))
            w.writerow(row)
    print("[make_sota_table] wrote %s" % path)


def emit_table(agg, args, bb, suite, out_dir, stem):
    n_stages = (agg.get("suites", {}).get(suite) or {}).get("n_stages")
    meta = {"backbone": bb, "backbone_label": FC.BACKBONE_LABEL[bb], "suite": suite,
            "order": args.order, "seed": args.seed}
    rows = []
    bv, bn = base_values(agg, args, bb)
    rows.append(("base (%s)" % FC.BACKBONE_LABEL[bb], bv, bn))
    sb = suite_block(agg, bb, suite) or {}
    present = {}
    for armkey in sb.get("arms", {}):
        tok, order, seed = armkey.split("|")
        if order == args.order and int(seed.lstrip("s")) == args.seed:
            present[tok] = armkey
    tokens = [t for _l, t in METHODS] + sorted(
        [t for t in present if t not in {x for _l, x in METHODS} and not t.startswith("single_")],
        key=FC.arm_sort_key)
    labels = dict((t, l) for l, t in METHODS)
    n_any = 0
    for tok in tokens:
        vals, flags, runtag = method_values(agg, args, bb, suite, tok, n_stages)
        flags = set(flags)
        if runtag:
            flags.add("_runtag=" + runtag)
        label = labels.get(tok) or extra_label(tok)
        rows.append((label, vals, flags))
        got = [k for k, _h, _d, _f in COLUMNS if vals[k] is not None]
        n_any += bool(got)
        print("  %-28s present: %s" % (label, ", ".join(got) or "(none)"))
    best = best_per_column(rows)
    # the CSV wants the runtag as a column, not a flag
    rows_csv = []
    for label, vals, flags in rows:
        rt = next((x[len("_runtag="):] for x in flags if x.startswith("_runtag=")), "")
        d = {x: True for x in flags if not x.startswith("_")}
        d["_runtag"] = rt
        rows_csv.append((label, vals, d))
    write_tex(rows, best, meta, os.path.join(out_dir, stem + ".tex"))
    write_csv(rows_csv, meta, os.path.join(out_dir, stem + ".csv"))
    print("[make_sota_table] %s: %d/%d method rows have at least one value"
          % (stem, n_any, len(tokens)))
    return n_any


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agg", default=None, help="fs_aggregate.json [layout agg_default]")
    FC.add_layout_args(ap)
    ap.add_argument("--out", default=None, help="output dir [layout out_dir/tables]")
    ap.add_argument("--backbone", default="llava15", choices=list(FC.BASE_DIRS))
    ap.add_argument("--suite", default="ucit", choices=["ucit", "pilot"])
    ap.add_argument("--order", default="o1")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--only_canonical", action="store_true",
                    help="skip the extra sota_<backbone>_<suite> tables")
    args = ap.parse_args()
    lay = FC.resolve_layout_args(args)
    args.agg = str(lay["agg_default"]) if not args.agg else os.path.abspath(os.path.expanduser(args.agg))
    out_dir = os.path.abspath(args.out) if args.out else str(lay["out_dir"] / "tables")
    os.makedirs(out_dir, exist_ok=True)
    if not args.bench or not os.path.isdir(args.bench):
        print("[make_sota_table] note: bench dir %s absent -> external columns come from the "
              "aggregate's bench block only" % args.bench)

    agg, _source = load_aggregate_or_recompute(args)
    print("[make_sota_table] canonical cell: %s / %s / %s / s%d"
          % (args.backbone, args.suite, args.order, args.seed))
    emit_table(agg, args, args.backbone, args.suite, out_dir, "sota")
    if not args.only_canonical:
        for bb in sorted(agg.get("backbones", {})):
            for suite in sorted((agg["backbones"][bb].get("suites") or {})):
                if (bb, suite) == (args.backbone, args.suite):
                    continue
                if not suite_block(agg, bb, suite) or not suite_block(agg, bb, suite).get("arms"):
                    continue
                print("[make_sota_table] extra table: %s / %s" % (bb, suite))
                emit_table(agg, args, bb, suite, out_dir, "sota_%s_%s" % (bb, suite))
    print("[make_sota_table] DONE -> %s" % out_dir)


if __name__ == "__main__":
    main()
