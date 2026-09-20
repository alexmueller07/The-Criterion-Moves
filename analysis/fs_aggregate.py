#!/usr/bin/env python3
"""Full-study aggregation harness -- results flow into the pre-registered
endpoints automatically as arms drain into results_fs/. Layout-agnostic: runs
unchanged from <repo>/analysis/ (scorers in <repo>/pilot/) and from the cluster
~/cl-halluc/code/analysis/ (scorers flat in ~/cl-halluc/code/).

    python analysis/fs_aggregate.py                      # local repo, auto-detected
    python ~/cl-halluc/code/analysis/fs_aggregate.py     # cluster, auto-detected
    python fs_aggregate.py --results R --coco_gt G --pope_dir P \
        --manifest_ucit U --manifest_pilot M --bench B --out agg.json --summary s.txt

Every flag defaults to the detected layout (fs_common.find_layout: $CLH_ROOT,
~/cl-halluc, the script's grandparent, /home/alexmueller/cl-halluc, else the
local repo); explicit flags always win; relative paths resolve against the CWD.

Scoring (fs_common; the audited pilot scorers are imported, never re-typed):
  POPE    pooled H/FA/d'/c/yes-rate (+ exact POPE-F1)   parse_yn
  CHAIR   CHAIR_i (per-mention), CHAIR_s, CHAIR_i@60    extract_object_mentions
  tasks   ucit: norm-containment-EM; pilot: mc_acc/vqa_acc/caption_uf1
          + refusal_rate on every task (leakage audit)

Result layout, per cell dir:
  <RUNTAG>_k<N>            sequential-arm stage N        <backbone>_base
  <RUNTAG>_ckpt_step<S>    joint checkpoints (stage index = sorted step order)
  <RUNTAG>_final           joint endpoint
  each holds pope_gen.jsonl, chair_gen.jsonl, <task>_gen.jsonl, gate_info.json,
  EVAL_DONE (empty marker => eval complete).
RUNTAG families (fs_common.parse_runtag):
  fs{L,Q}_<arm>_o<N>_s<seed>   UCIT suite, 6 stages        (fsL_seq_o1_s17)
  ps{L,Q}_<arm>_o<N>_s<seed>   pilot suite, 4 stages       (psQ_anchor_o2_s17)
  lamF_<w>_o<N>_s<seed>        anchor-strength sweep, UCIT (lamF_0.02_o1_s17)
  mth_<cand>_o<N>_s<seed>      method candidates, pilot    (mth_critp_o1_s17)
  lamF/mth carry no backbone letter: the backbone is read from each cell's
  gate_info.json argv (--backbone), else --family_backbone, else LLaVA.
  The suite is cross-checked against the task gen files actually present.

Partial-draining discipline:
  * a cell is aggregated only if its dir carries EVAL_DONE (else IN-PROGRESS);
  * a metric whose gen file is absent in an EVAL_DONE cell is ABSENT;
  * a present-but-malformed file FAILS LOUD: the cell is excluded, listed under
    MALFORMED in the summary and JSON, and the exit code is 3 -- never averaged;
  * endpoint math uses the contiguous stage prefix 1..m of a cell; the method
    scorecard uses complete cells only (all n_stages present).

Endpoints (FULLSTUDY_PREREG.md) in the multi-seed style guide language
(design_notes/review_statistics.md sec 2): per-cell values + [min-max] ranges,
sign-consistency counts, orderings as fixed strata, NO pooled cross-seed CIs.
  E1   dose-response where the frozen manifest has criterion-active stages
       (pilot suite): sign(dyes-rate) vs the stage's answer-prior direction per
       cell; zero-dose |dc| < smallest active |dc| in the same run. On the
       answer-statistics-flat UCIT stream this is the NULL side (E1-null):
       little drift predicted, drift reported as observed, never assumed.
  E2   anchor criterion-freeze: post-settling Sigma|dc|(ANCHOR) < (SEQ) per
       matched seed x order cell; sign count; pooled suppression ratio.
  E3   length-controlled mid-sequence rise: CHAIR_i@60 stage 1 -> stage T-1 in
       SEQ; sign per cell; paired per-cell image bootstrap (ratio of sums,
       B=--e3_boot, seed 17); prereg rule >=7/9 positive AND CI>0 in >=5/9.
  E4   memorylessness: each single-task control's endpoint c vs the seed
       min-max band of the SEQ stage that ends on the same task, per ordering.
  E5   refusal leakage: applies iff a stage's frozen refusal_frac >= 10%
       (pilot VizWiz); refusal_rate on the short-answer tasks per stage.
  F1   falsifier: any SEQ checkpoint with |d' - d'_base| > 0.30 damages the
       "criterion drift, not grounding" headline.
METHOD SCORECARD per backbone x suite (complete cells; per arm): Sigma|dc| per
cell, mean, [min-max]; post-settling Sigma|dc|; d' max-dev from base; endpoint
CHAIR_i and CHAIR_i@60; mean new-task accuracy (plasticity; UCIT closed-ended
only); endpoint task mean; paired sign-consistency vs SEQ on matched cells.

JSON schema is backward compatible: top-level {results_dir, in_progress,
unparsed, f1_tol, backbones{bb: {base, arms, present_cells, endpoints}}} keeps
its previous meaning (arms/endpoints = the UCIT suite); new content is additive
(backbones[bb].suites[suite].{arms,present_cells,endpoints,scorecard,coverage},
bench, malformed, layout, suites, E1_cross_backbone).

Python 3.9 + stdlib (numpy used for the bootstrap when importable).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import fs_common as FC  # noqa: E402

F1_DPRIME_TOL = 0.30
E3_RULE = {"n_cells": 9, "need_positive": 7, "need_ci": 5}   # prereg E3 (LLaVA cells)
E5_REFUSAL_MIN = 0.10
PLASTICITY_TOL = 0.02   # "not worse" tolerance for the plasticity sign count


# ---------------------------------------------------------------------------
# Cell scoring + discovery
# ---------------------------------------------------------------------------
def score_cell(cell_dir, suite, task_label, coco_gt, expected):
    """Score one EVAL_DONE dir. ABSENT (None) where a gen file is missing;
    raises MalformedResult on corrupt content."""
    a = {}
    out = {"dir": cell_dir.name, "task_trained": task_label}
    pope_p = cell_dir / FC.POPE_FILE
    out["pope"] = FC.score_pope(pope_p, a) if pope_p.exists() else None
    chair_p = cell_dir / FC.CHAIR_FILE
    if chair_p.exists() and coco_gt is not None:
        out["chair"] = FC.score_chair(chair_p, coco_gt, a)
    else:
        out["chair"] = None
        if chair_p.exists():
            a["chair_skipped_no_gt"] = True
    tasks = {}
    for t in FC.task_gen_files(cell_dir):
        tasks[t] = FC.score_task(cell_dir / (t + "_gen.jsonl"), t, suite, a)
    out["tasks"] = tasks
    mism = {}
    if out["pope"] and expected.get("pope") and out["pope"]["n_total"] != expected["pope"]:
        mism["pope"] = [out["pope"]["n_total"], expected["pope"]]
    if out["chair"] and expected.get("chair") and out["chair"]["n_captions"] != expected["chair"]:
        mism["chair"] = [out["chair"]["n_captions"], expected["chair"]]
    for t, r in tasks.items():
        exp = expected.get("tasks", {}).get(t)
        if exp and r["n"] != exp:
            mism[t] = [r["n"], exp]
    if mism:
        a["row_count_mismatch"] = mism
    out["audit"] = a
    return out


def expected_counts(args, coco_gt, suites):
    exp = {"pope": None, "chair": len(coco_gt) if coco_gt else None, "tasks": {}}
    if args.pope_dir and os.path.isfile(os.path.join(args.pope_dir, "prompts.jsonl")):
        exp["pope"] = FC.count_jsonl_ids(os.path.join(args.pope_dir, "prompts.jsonl"))
    for suite, mp in (("ucit", args.manifest_ucit), ("pilot", args.manifest_pilot)):
        if not mp or not os.path.isfile(mp):
            continue
        droot = os.path.dirname(os.path.abspath(mp))
        for t in suites[suite]["tasks"]:
            vp = os.path.join(droot, "tasks", t, "val.jsonl")
            if os.path.isfile(vp):
                exp["tasks"][t] = FC.count_jsonl_ids(vp)
    return exp


def _empty_bb():
    return {"base": None, "suites": {}}


def discover(results_dir, family_backbone, coco_gt, suites, expected):
    """-> (matrix, in_progress, unparsed, malformed, notes)
    matrix[bb] = {"base": cell|None,
                  "suites": {suite: {"arms": {(arm, order, seed): {k: cell}}}}}"""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise SystemExit("[fs_aggregate] --results dir not found: %s" % results_dir)
    matrix, in_progress, unparsed, malformed, notes = {}, [], [], [], []
    groups = {}
    for entry in sorted(results_dir.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if name in FC.BASE_DIRS.values():
            bb = [k for k, v in FC.BASE_DIRS.items() if v == name][0]
            matrix.setdefault(bb, _empty_bb())
            if not (entry / "EVAL_DONE").exists():
                in_progress.append(name)
                continue
            suite = FC.suite_of_task_files(FC.task_gen_files(entry)) or "ucit"
            try:
                cell = score_cell(entry, suite, "(base)", coco_gt, expected)
            except FC.MalformedResult as ex:
                malformed.append({"dir": name, "error": str(ex)})
                continue
            cell.update({"backbone": bb, "suite": suite, "k": 0})
            matrix[bb]["base"] = cell
            continue
        pc = FC.parse_cell_name(name)
        if pc is None:
            unparsed.append(name)
            continue
        runtag, kind, val = pc
        pr = FC.parse_runtag(runtag, family_backbone)
        if pr is None:
            unparsed.append(name)
            continue
        groups.setdefault(runtag, {"info": pr, "cells": []})["cells"].append((kind, val, entry))

    for runtag in sorted(groups):
        info, cells = groups[runtag]["info"], groups[runtag]["cells"]
        if info["arm"] == "joint":
            ordered = sorted(cells, key=lambda t: (float("inf") if t[0] == "final" else t[1]))
            staged = [(i + 1, kind, val, e) for i, (kind, val, e) in enumerate(ordered)]
        else:
            staged = []
            for kind, val, e in cells:
                if kind == "k":
                    staged.append((val, kind, val, e))
                else:
                    unparsed.append(e.name)
                    notes.append("%s: %s cell on a non-joint arm ignored" % (e.name, kind))
        bb, bb_src = info["backbone"], "runtag-letter"
        if info["family"] in ("lamF", "mth"):
            gi = None
            for _, _, _, e in staged:
                gi = FC.backbone_from_gate_info(e)
                if gi:
                    break
            if gi:
                bb, bb_src = gi, "gate_info"
            else:
                bb_src = "family_backbone-default"
        suite = info["suite"]
        done_dirs = [e for _, _, _, e in staged if (e / "EVAL_DONE").exists()]
        if done_dirs:
            s2 = FC.suite_of_task_files(FC.task_gen_files(done_dirs[0]))
            if s2 and s2 != suite:
                notes.append("%s: suite %s by family, %s by task files -> using %s"
                             % (runtag, suite, s2, s2))
                suite = s2
        sinfo = suites[suite]
        tlist = FC.order_tasks(sinfo, info["order"])
        arm = info["arm"]
        for k, kind, val, e in staged:
            if not (e / "EVAL_DONE").exists():
                in_progress.append(e.name)
                continue
            if arm.startswith("single_"):
                label = arm[len("single_"):]
            elif arm == "joint":
                label = "joint@%d" % k
            else:
                label = tlist[k - 1] if (tlist and 1 <= k <= len(tlist)) else "?"
            try:
                cell = score_cell(e, suite, label, coco_gt, expected)
            except FC.MalformedResult as ex:
                malformed.append({"dir": e.name, "error": str(ex)})
                continue
            cell.update({"runtag": runtag, "arm": arm, "order": info["order"],
                         "seed": info["seed"], "k": k, "kind": kind,
                         "step": (val if kind == "step" else None),
                         "suite": suite, "family": info["family"], "backbone": bb,
                         "backbone_source": bb_src, "faith_weight": info["faith_weight"]})
            matrix.setdefault(bb, _empty_bb())["suites"].setdefault(suite, {"arms": {}})
            matrix[bb]["suites"][suite]["arms"].setdefault(
                (arm, info["order"], info["seed"]), {})[k] = cell
    return matrix, in_progress, unparsed, malformed, notes


# ---------------------------------------------------------------------------
# Endpoint helpers
# ---------------------------------------------------------------------------
def series(base, stages, ks):
    """(cs, ds, ys) with the base prepended over stage indices ks, or None."""
    if base is None or not base.get("pope") or not ks:
        return None
    if any(not stages[k].get("pope") for k in ks):
        return None
    p0 = base["pope"]
    cs = [p0["c"]] + [stages[k]["pope"]["c"] for k in ks]
    ds = [p0["dprime"]] + [stages[k]["pope"]["dprime"] for k in ks]
    ys = [p0["yes_rate"]] + [stages[k]["pope"]["yes_rate"] for k in ks]
    return cs, ds, ys


def r4(x):
    return None if x is None else round(x, 4)


def sign_str(k, n):
    return "%d/%d" % (k, n)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
def eval_endpoints(base, arms, sinfo, boot_B):
    T = sinfo["n_stages"]
    base_c = base["pope"]["c"] if base and base.get("pope") else None
    base_d = base["pope"]["dprime"] if base and base.get("pope") else None
    seq = {(o, s): st for (a, o, s), st in arms.items() if a == "seq"}
    anc = {(o, s): st for (a, o, s), st in arms.items() if a == "anchor"}
    doses = {t: FC.dose_of((sinfo["tasks"].get(t) or {}).get("answer_stats"))
             for t in sinfo["tasks"]}
    active_tasks = sorted(t for t, d in doses.items() if d and d["active"])
    stats_missing = [t for t, d in doses.items() if d is None]

    # ---- E1 (dose-response or null side) + F1 walk the SEQ cells ----------
    e1_cells, e1n_cells, f1_cells = [], [], []
    e1n_by_order, e1_by_order = {}, {}
    for (order, seed), stages in sorted(seq.items()):
        ks = FC.contiguous_prefix(stages)
        tl = FC.order_tasks(sinfo, order)
        # F1 over every present checkpoint (any-cell claim)
        if base_d is not None:
            dd = [(k, round(stages[k]["pope"]["dprime"] - base_d, 4))
                  for k in sorted(stages) if stages[k].get("pope")]
            if dd:
                f1_cells.append({"order": order, "seed": seed, "k": [k for k, _ in dd],
                                 "dd": [v for _, v in dd],
                                 "max_abs_dd": round(max(abs(v) for _, v in dd), 4),
                                 "tripped": any(abs(v) > F1_DPRIME_TOL for _, v in dd)})
        ser = series(base, stages, ks)
        if ser is None:
            continue
        cs, ds, ys = ser
        sw = FC.swings(cs)
        dc_from_base = [round(c - base_c, 4) for c in cs[1:]]
        e1n_cells.append({"order": order, "seed": seed, "k_present": ks,
                          "complete": len(ks) == T,
                          "dc_from_base": dc_from_base,
                          "max_abs_dc": round(max(abs(x) for x in dc_from_base), 4),
                          "sum_abs_step": round(sum(sw), 4),
                          "endpoint_c": r4(cs[-1]), "base_c": r4(base_c)})
        e1n_by_order.setdefault(order, []).append((seed, max(abs(x) for x in dc_from_base)))
        per_stage = []
        for i, k in enumerate(ks):
            t = tl[k - 1] if (tl and k <= len(tl)) else None
            d = doses.get(t) if t else None
            pred = d["direction"] if d else 0
            dy = ys[i + 1] - ys[i]
            match = None if pred == 0 else bool((dy > 0) == (pred > 0))
            per_stage.append({"k": k, "task": t, "dc": round(cs[i + 1] - cs[i], 4),
                              "dyes": round(dy, 4), "dose_dir": pred,
                              "dose_mag": d["magnitude"] if d else None,
                              "sign_match": match})
        act = [abs(s["dc"]) for s in per_stage if s["dose_dir"] != 0]
        zero = [abs(s["dc"]) for s in per_stage if s["dose_dir"] == 0]
        e1_cells.append({"order": order, "seed": seed, "k_present": ks,
                         "complete": len(ks) == T, "stages": per_stage,
                         "n_active": len(act),
                         "n_match": sum(1 for s in per_stage if s["sign_match"]),
                         "zero_dose_below_active": (max(zero) < min(act))
                         if (act and zero) else None,
                         "min_active_abs_dc": r4(min(act)) if act else None,
                         "max_zero_abs_dc": r4(max(zero)) if zero else None})
        for s in per_stage:
            if s["sign_match"] is not None:
                e1_by_order.setdefault(order, {}).setdefault(s["k"], []).append(
                    (seed, s["sign_match"]))
    e1n_by_order_out = {o: {"seeds": sorted(s for s, _ in v),
                            "max_abs_dc_range": FC.rng_pair([m for _, m in v])}
                        for o, v in e1n_by_order.items()}
    if not e1n_cells:
        e1n_note = "no complete SEQ cell (with base) present -> ABSENT"
    elif active_tasks:
        e1n_note = ("suite has criterion-active stages %s: drift is expected, see E1 "
                    "dose-response; this block is the descriptive drift table" % active_tasks)
    else:
        mx = max(c["max_abs_dc"] for c in e1n_cells)
        e1n_note = ("zero-dose stream (all stages answer-statistics-flat): little criterion "
                    "drift is PREDICTED; observed max|dc| across cells = %.3f -- reported as "
                    "observed, not assumed flat" % mx)
    e1 = {"mode": "dose-response" if active_tasks else "null",
          "active_tasks": active_tasks, "answer_stats_missing": stats_missing,
          "dose_table": {t: d for t, d in doses.items() if d},
          "cells": e1_cells,
          "by_order": {o: {str(k): {"n_match": sum(1 for _, m in v if m), "n": len(v),
                                    "seeds": sorted(s for s, _ in v)}
                           for k, v in sorted(kv.items())}
                       for o, kv in e1_by_order.items()},
          "prereg_rule": ">=5 of 6 applicable seed x backbone cells per ordering match the "
                         "stage's answer-prior direction (cross-backbone tally at report "
                         "level); zero-dose |dc| < smallest active |dc| in the same run",
          "note": ("no criterion-active stage in this suite -> E1 dose-response NOT testable "
                   "here (null side reported in E1_null)" if not active_tasks else
                   "criterion-active stages: %s" % active_tasks)}

    # ---- E2: matched seq vs anchor, post-settling Sigma|dc| ----------------
    def post(stages):
        ks = FC.contiguous_prefix(stages)
        if len(ks) != T:
            return None
        ser = series(base, stages, ks)
        if ser is None:
            return None
        return round(FC.post_settle_sum(FC.swings(ser[0])), 4)
    matched = []
    for key in sorted(set(seq) & set(anc)):
        sq, an = post(seq[key]), post(anc[key])
        if sq is None or an is None:
            continue
        matched.append({"order": key[0], "seed": key[1], "seq": sq, "anchor": an,
                        "freeze": bool(an < sq),
                        "ratio": round(sq / an, 3) if an > 0 else None})
    e2_by_order = {}
    for m in matched:
        e2_by_order.setdefault(m["order"], []).append(m)
    e2 = {"matched": matched, "n_freeze": sum(m["freeze"] for m in matched),
          "n_matched": len(matched),
          "pooled_suppression_ratio": (round(sum(m["seq"] for m in matched)
                                             / sum(m["anchor"] for m in matched), 3)
                                       if matched and sum(m["anchor"] for m in matched) > 0
                                       else None),
          "by_order": {o: {"n_freeze": sum(m["freeze"] for m in v), "n": len(v),
                           "seq_range": FC.rng_pair([m["seq"] for m in v]),
                           "anchor_range": FC.rng_pair([m["anchor"] for m in v])}
                       for o, v in e2_by_order.items()},
          "note": ("no matched seed x order cell with BOTH seq and anchor complete "
                   "(seq complete: %s; anchor complete: %s) -> E2 ABSENT"
                   % (sorted(k for k in seq if post(seq[k]) is not None),
                      sorted(k for k in anc if post(anc[k]) is not None))
                   if not matched else
                   "prereg: anchor < seq in EVERY matched cell (18 per backbone-complete matrix)")}

    # ---- E3: CHAIR_i@60 stage 1 -> stage T-1 in SEQ ------------------------
    e3_cells = []
    for (order, seed), stages in sorted(seq.items()):
        a, b = stages.get(1), stages.get(T - 1)
        if T < 3 or not a or not b or not a.get("chair") or not b.get("chair"):
            continue
        ca, cb = a["chair"], b["chair"]
        boot = FC.boot_ratio_gap(ca.get("_per_image_k", {}), cb.get("_per_image_k", {}),
                                 B=boot_B) if boot_B > 0 else None
        delta = round(cb["chair_i60"] - ca["chair_i60"], 4)
        e3_cells.append({"order": order, "seed": seed, "k_from": 1, "k_to": T - 1,
                         "chair_i60_from": ca["chair_i60"], "chair_i60_to": cb["chair_i60"],
                         "delta": delta, "positive": delta > 0,
                         "boot": boot,
                         "ci_above_zero": bool(boot and boot["ci"][0] > 0),
                         "frac_over_budget": [ca["frac_over_budget"], cb["frac_over_budget"]]})
    n_pos = sum(c["positive"] for c in e3_cells)
    n_ci = sum(c["ci_above_zero"] for c in e3_cells)
    if not e3_cells:
        e3_verdict = "ABSENT (no SEQ cell has both stage 1 and stage %d CHAIR)" % (T - 1)
    elif len(e3_cells) < E3_RULE["n_cells"]:
        e3_verdict = ("PENDING: %d/%d cells; positive %s, CI>0 %s (rule needs >=%d/%d and >=%d/%d)"
                      % (len(e3_cells), E3_RULE["n_cells"], sign_str(n_pos, len(e3_cells)),
                         sign_str(n_ci, len(e3_cells)), E3_RULE["need_positive"],
                         E3_RULE["n_cells"], E3_RULE["need_ci"], E3_RULE["n_cells"]))
    elif n_pos >= E3_RULE["need_positive"] and n_ci >= E3_RULE["need_ci"]:
        e3_verdict = "CONFIRMATORY: positive %s, CI>0 %s" % (sign_str(n_pos, len(e3_cells)),
                                                              sign_str(n_ci, len(e3_cells)))
    else:
        e3_verdict = ("NOT CONFIRMED: positive %s, CI>0 %s"
                      % (sign_str(n_pos, len(e3_cells)), sign_str(n_ci, len(e3_cells))))
    e3_by_order = {}
    for c in e3_cells:
        e3_by_order.setdefault(c["order"], []).append(c)
    e3 = {"cells": e3_cells, "n_positive": n_pos, "n_ci_above_zero": n_ci,
          "n": len(e3_cells), "rule": E3_RULE, "verdict": e3_verdict,
          "by_order": {o: {"n_positive": sum(c["positive"] for c in v),
                           "n_ci": sum(c["ci_above_zero"] for c in v), "n": len(v),
                           "delta_range": FC.rng_pair([c["delta"] for c in v])}
                       for o, v in e3_by_order.items()}}

    # ---- E4: single-task controls vs SEQ same-last-task seed band ---------
    ctrls = []
    for (arm, order, seed), stages in sorted(arms.items()):
        if not arm.startswith("single_"):
            continue
        task = arm[len("single_"):]
        kk = max(stages)
        p = stages[kk].get("pope")
        if not p:
            continue
        entry = {"arm": arm, "task": task, "seed": seed, "c": p["c"], "dprime": p["dprime"],
                 "yes_rate": p["yes_rate"], "base_c": r4(base_c), "bands": []}
        for o, tl in sinfo["orders"].items():
            if task not in tl:
                continue
            k = tl.index(task) + 1
            vals = [(s, st[k]["pope"]["c"]) for (oo, s), st in seq.items()
                    if oo == o and k in st and st[k].get("pope")]
            band = FC.rng_pair([v for _, v in vals])
            inside = (band[0] <= p["c"] <= band[1]) if band else None
            end_vals = [st[T]["pope"]["c"] for (oo, _s), st in seq.items()
                        if oo == o and T in st and st[T].get("pope")]
            entry["bands"].append({"order": o, "seq_stage": k, "seeds": sorted(s for s, _ in vals),
                                   "seq_c_values": [round(v, 4) for _, v in vals],
                                   "band": band, "inside": inside,
                                   "seq_endpoint_band": FC.rng_pair(end_vals)})
        ctrls.append(entry)
    n_bands = sum(1 for c in ctrls for b in c["bands"] if b["inside"] is not None)
    n_inside = sum(1 for c in ctrls for b in c["bands"] if b["inside"])
    e4 = {"controls": sorted(c["arm"] for c in ctrls), "cells": ctrls,
          "n_inside": n_inside, "n_bands": n_bands,
          "note": ("no single-task control cell present -> E4 ABSENT" if not ctrls else
                   "no complete SEQ stage to form a band -> E4 ABSENT" if n_bands == 0 else
                   "control endpoint c inside the SEQ same-last-task seed band in %s "
                   "comparisons (prereg: within the band, per backbone)"
                   % sign_str(n_inside, n_bands))}

    # ---- E5: refusal leakage (conditional) ---------------------------------
    refusal_tasks = [t for t, d in doses.items() if d and d["refusal_frac"] >= E5_REFUSAL_MIN]
    caption = set(sinfo["caption_tasks"])
    leak_tasks = [t for t in sinfo["tasks"] if t not in refusal_tasks and t not in caption
                  and (doses.get(t) is None or doses[t]["refusal_frac"] < E5_REFUSAL_MIN)]
    e5_cells = []
    if refusal_tasks:
        for (order, seed), stages in sorted(seq.items()):
            tl = FC.order_tasks(sinfo, order) or []
            first_ref = min([tl.index(t) + 1 for t in refusal_tasks if t in tl] or [None])
            per = {}
            for k in sorted(stages):
                per[str(k)] = {t: stages[k]["tasks"][t]["refusal_rate"]
                               for t in leak_tasks if t in stages[k]["tasks"]}
            ks = sorted(stages)
            end = per[str(ks[-1])] if ks else {}
            pre = per.get(str(first_ref - 1)) if first_ref and first_ref > 1 else None
            e5_cells.append({"order": order, "seed": seed, "first_refusal_stage": first_ref,
                             "leak_by_stage": per, "endpoint": end,
                             "pre_refusal_stage": pre,
                             "excess": {t: round(end[t] - pre[t], 4) for t in end
                                        if pre and t in pre}})
    e5 = {"applies": bool(refusal_tasks), "refusal_tasks": refusal_tasks,
          "leak_tasks": leak_tasks, "cells": e5_cells,
          "note": ("not testable on this suite: no stage with >= %.0f%% refusal-class "
                   "training targets (frozen stats)" % (E5_REFUSAL_MIN * 100)
                   if not refusal_tasks else
                   "refusal_rate ('unanswerable' in the normalized output) on the short-"
                   "answer tasks %s per SEQ stage; excess = endpoint - pre-refusal stage"
                   % leak_tasks)}

    return {
        "E1_null": {"cells": e1n_cells, "by_order": e1n_by_order_out, "note": e1n_note},
        "E1": e1,
        "F1": {"cells": f1_cells, "any_tripped": any(c["tripped"] for c in f1_cells),
               "base_absent": base_d is None, "tol": F1_DPRIME_TOL},
        "E2": e2, "E3": e3, "E4": e4, "E5": e5,
    }


# ---------------------------------------------------------------------------
# Method scorecard
# ---------------------------------------------------------------------------
def cell_scorecard(base, stages, suite):
    T_present = len(stages)
    ks = FC.contiguous_prefix(stages)
    entry = {"n_stages_present": T_present, "k_present": ks}
    ser = series(base, stages, ks)
    if ser is not None:
        cs, ds, ys = ser
        sw = FC.swings(cs)
        entry.update({"sum_abs_dc": round(sum(sw), 4),
                      "post_settle": r4(FC.post_settle_sum(sw)),
                      "endpoint_c": r4(cs[-1]), "endpoint_dc": r4(cs[-1] - cs[0]),
                      "max_abs_dd": round(max(abs(d - ds[0]) for d in ds[1:]), 4),
                      "endpoint_yes_rate": r4(ys[-1]),
                      "dc_steps": [round(x, 4) for x in sw]})
    if ks:
        end = stages[ks[-1]]
        if end.get("chair"):
            entry["endpoint_chair_i"] = end["chair"]["chair_i"]
            entry["endpoint_chair_i60"] = end["chair"]["chair_i60"]
        closed = {t: v["acc"] for t, v in end["tasks"].items()
                  if not (suite == "ucit" and v.get("open_ended"))}
        if closed:
            entry["endpoint_task_mean"] = round(FC.mean(list(closed.values())), 4)
        accs = []
        for k in ks:
            tt = stages[k].get("task_trained")
            tv = stages[k]["tasks"].get(tt) if tt else None
            if tv is None or (suite == "ucit" and tv.get("open_ended")):
                continue
            accs.append(tv["acc"])
        if accs and not str(stages[ks[0]].get("task_trained", "")).startswith("joint"):
            entry["plasticity"] = round(FC.mean(accs), 4)
            entry["plasticity_n"] = len(accs)
    return entry


SC_KEYS = ["sum_abs_dc", "post_settle", "endpoint_c", "max_abs_dd", "endpoint_chair_i",
           "endpoint_chair_i60", "plasticity", "endpoint_task_mean"]


def build_scorecard(base, arms, sinfo):
    T = sinfo["n_stages"]
    suite = sinfo["name"]
    rows = {}
    for (arm, order, seed), stages in arms.items():
        if arm.startswith("single_"):
            continue
        e = cell_scorecard(base, stages, suite)
        e.update({"order": order, "seed": seed, "complete": len(e["k_present"]) == T})
        rows.setdefault(arm, []).append(e)
    seq_cells = {(c["order"], c["seed"]): c for c in rows.get("seq", []) if c["complete"]}
    out = []
    for arm in sorted(rows, key=FC.arm_sort_key):
        cells = sorted(rows[arm], key=lambda c: (c["order"], c["seed"]))
        comp = [c for c in cells if c["complete"]]
        fw = None
        if arm.startswith("lamF_"):
            try:
                fw = float(arm[5:])
            except ValueError:
                fw = None
        agg = {"arm": arm, "family": FC.arm_family(arm), "faith_weight": fw,
               "n_cells": len(cells), "n_complete": len(comp), "cells": cells}
        for key in SC_KEYS:
            vals = [c[key] for c in comp if c.get(key) is not None]
            agg[key] = ({"mean": round(FC.mean(vals), 4), "range": FC.rng_pair(vals),
                         "values": vals, "n": len(vals)} if vals else None)
        if arm != "seq":
            pairs = [(c, seq_cells[(c["order"], c["seed"])]) for c in comp
                     if (c["order"], c["seed"]) in seq_cells]

            def cnt(key, lower=True, tol=0.0):
                n = k = 0
                for c, s in pairs:
                    if c.get(key) is None or s.get(key) is None:
                        continue
                    n += 1
                    k += int(c[key] < s[key]) if lower else int(c[key] >= s[key] - tol)
                return [k, n]
            per = []
            for c, s in pairs:
                d = {"order": c["order"], "seed": c["seed"]}
                for key in ("sum_abs_dc", "post_settle", "endpoint_chair_i",
                            "endpoint_chair_i60", "plasticity", "max_abs_dd"):
                    if c.get(key) is not None and s.get(key) is not None:
                        d["d_" + key] = round(c[key] - s[key], 4)
                per.append(d)
            agg["vs_seq"] = {"n_matched": len(pairs),
                             "sum_abs_dc_lower": cnt("sum_abs_dc"),
                             "post_settle_lower": cnt("post_settle"),
                             "chair_i_lower": cnt("endpoint_chair_i"),
                             "chair_i60_lower": cnt("endpoint_chair_i60"),
                             "plasticity_not_worse": cnt("plasticity", lower=False,
                                                         tol=PLASTICITY_TOL),
                             "plasticity_tol": PLASTICITY_TOL, "per_cell": per}
        out.append(agg)
    return out


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def _f(v, spec):
    return "ABSENT" if v is None else format(v, spec)


def print_matrix(bb, suite, base, arms, sinfo, out):
    P = out.append
    P("")
    P("#" * 78)
    P("# BACKBONE %s   SUITE %s (%d stages; orders %s; source: %s)"
      % (bb, suite, sinfo["n_stages"], ",".join(sorted(sinfo["orders"])), sinfo["source"]))
    P("#" * 78)
    if base is None or not base.get("pope"):
        P("  base: ABSENT (no EVAL_DONE %s cell) -- drift/F1/scorecard need it" % FC.BASE_DIRS[bb])
    else:
        p, cj = base["pope"], base["chair"]
        P("  base: c=%+.3f  d'=%+.3f  yes=%4.1f%%  CHAIR_i=%s  CHAIR_i@60=%s  (parse_fail=%d)"
          % (p["c"], p["dprime"], p["yes_rate"] * 100,
             _f(cj["chair_i"] if cj else None, ".4f"), _f(cj["chair_i60"] if cj else None, ".4f"),
             p["parse_fail"]))
    bd = base["pope"]["dprime"] if base and base.get("pope") else None
    for key in sorted(arms, key=lambda k: (FC.arm_sort_key(k[0]), k[1], k[2])):
        arm, order, seed = key
        stages = arms[key]
        P("")
        P("  --- arm=%s  order=%s  seed=%d  (stages present: %s of %d) ---"
          % (arm, order, seed, sorted(stages), sinfo["n_stages"]))
        P("    %-6s%-13s%8s%8s%10s%7s%9s%10s%9s  %s"
          % ("stage", "task", "c", "d'", "|dd'|base", "yes%", "CHAIR_i", "CHAIR@60",
             "new-acc", "audit"))
        for k in sorted(stages):
            cell = stages[k]
            p, cj = cell["pope"], cell["chair"]
            tt = cell.get("task_trained") or "?"
            nt = cell["tasks"].get(tt) if tt in cell["tasks"] else None
            aud = []
            a = cell.get("audit") or {}
            if a.get("row_count_mismatch"):
                aud.append("ROWS!" + ",".join(a["row_count_mismatch"]))
            if a.get("recovered_lines"):
                aud.append("recov=%d" % a["recovered_lines"])
            if a.get("dup_ids"):
                aud.append("dup=%d" % a["dup_ids"])
            if p and p["parse_fail"]:
                aud.append("pfail=%d" % p["parse_fail"])
            if p is None:
                P("    k%-5d%-13s%8s" % (k, tt[:12], "ABSENT"))
                continue
            dd = abs(p["dprime"] - bd) if bd is not None else None
            P("    k%-5d%-13s%+8.3f%+8.3f%10s%7.1f%9s%10s%9s  %s"
              % (k, tt[:12], p["c"], p["dprime"], _f(dd, ".3f") if dd is not None else "NA",
                 p["yes_rate"] * 100, _f(cj["chair_i"] if cj else None, ".4f"),
                 _f(cj["chair_i60"] if cj else None, ".4f"),
                 _f(nt["acc"] if nt else None, ".3f"), " ".join(aud)))
        tnames = sorted({t for k in stages for t in stages[k]["tasks"]})
        if tnames:
            P("    per-task acc (%s; * = open-ended caption task%s):"
              % ("audited pilot scorers" if suite == "pilot" else "norm-containment-EM",
                 ", containment-EM is a floor" if suite == "ucit" else ""))
            P("      stage " + "".join("%13s" % (t[:11] + ("*" if t in sinfo["caption_tasks"] else ""))
                                    for t in tnames))
            for k in sorted(stages):
                row = stages[k]["tasks"]
                P("      k%-5d" % k + "".join("%13s" % (_f(row[t]["acc"], ".4f") if t in row else "--")
                                             for t in tnames))
            leak = {t for k in stages for t, v in stages[k]["tasks"].items()
                    if v.get("refusal_rate", 0) > 0}
            if leak:
                P("      refusal_rate>0 on: " + ", ".join(
                    "%s@k%d=%.3f" % (t, k, stages[k]["tasks"][t]["refusal_rate"])
                    for k in sorted(stages) for t in sorted(leak)
                    if t in stages[k]["tasks"] and stages[k]["tasks"][t]["refusal_rate"] > 0))


def endpoint_lines(bb, suite, ep, sinfo):
    L = []
    P = L.append
    T = sinfo["n_stages"]
    P("")
    P("  " + "=" * 74)
    P("  ENDPOINT READOUT  %s / %s  (per-cell values + [min-max]; orderings are strata; "
      "no pooled cross-seed CIs)" % (bb, suite))
    P("  " + "=" * 74)

    e1 = ep["E1"]
    e1n = ep["E1_null"]
    if e1["mode"] == "null":
        P("")
        P("  [E1 null-side] criterion drift on the zero-dose %s stream (SEQ)" % suite)
        if e1["answer_stats_missing"]:
            P("    (answer stats missing for %s -> zero-dose assumed from the prereg amendment)"
              % e1["answer_stats_missing"])
        if not e1n["cells"]:
            P("    no complete SEQ cell (with base) present -> ABSENT")
        else:
            for c in e1n["cells"]:
                P("    %s/s%d%s: dc-from-base=[%s]  max|dc|=%.3f  Sigma|dc|(step)=%.3f"
                  % (c["order"], c["seed"], "" if c["complete"] else " (partial k%s)" % c["k_present"],
                     ", ".join("%+.3f" % x for x in c["dc_from_base"]), c["max_abs_dc"],
                     c["sum_abs_step"]))
            for o, agg in sorted(e1n["by_order"].items()):
                P("    stratum %s: max|dc| across seeds=%s (seeds %s)"
                  % (o, agg["max_abs_dc_range"], agg["seeds"]))
        P("    NOTE: %s" % e1n["note"])
    else:
        P("")
        P("  [E1 dose-response] sign(dyes-rate) vs stage answer-prior direction (SEQ); "
          "active stages: %s" % e1["active_tasks"])
        for t, d in sorted(e1["dose_table"].items()):
            P("    dose %-12s dir=%+d mag=%.3f (yes %.3f / no %.3f / refusal %.3f)%s"
              % (t, d["direction"], d["magnitude"], d["yes_frac"], d["no_frac"],
                 d["refusal_frac"], "" if d["active"] else "  [zero-dose]"))
        if not e1["cells"]:
            P("    no complete SEQ cell (with base) present -> ABSENT")
        for c in e1["cells"]:
            st = "  ".join("k%d %s dyes=%+.3f dc=%+.3f %s"
                           % (s["k"], (s["task"] or "?")[:10], s["dyes"], s["dc"],
                              "-" if s["sign_match"] is None else ("OK" if s["sign_match"] else "MISS"))
                           for s in c["stages"])
            P("    %s/s%d: %s" % (c["order"], c["seed"], st))
            P("      sign matches %s of active stages; zero-dose |dc| below smallest active |dc|: %s"
              % (sign_str(c["n_match"], c["n_active"]), c["zero_dose_below_active"]))
        for o, kv in sorted(e1["by_order"].items()):
            P("    stratum %s: " % o + "  ".join("k%s match %s" % (k, sign_str(v["n_match"], v["n"]))
                                                for k, v in kv.items()))
        P("    RULE: %s" % e1["prereg_rule"])

    f1 = ep["F1"]
    P("")
    P("  [F1 falsifier] |d' - d'_base| > %.2f at any SEQ checkpoint?" % f1["tol"])
    if f1["base_absent"]:
        P("    base ABSENT -> F1 not evaluable")
    elif not f1["cells"]:
        P("    no SEQ checkpoint present -> ABSENT")
    else:
        for c in f1["cells"]:
            P("    %s/s%d: max|d'-base|=%.3f  per-stage=[%s]  -> %s"
              % (c["order"], c["seed"], c["max_abs_dd"],
                 ", ".join("%+.3f" % x for x in c["dd"]), "TRIPPED" if c["tripped"] else "ok"))
        P("    VERDICT: %s" % ("FALSIFIER TRIPPED -- headline damaged, must revise"
                               if f1["any_tripped"] else "not tripped -- headline intact so far"))

    e2 = ep["E2"]
    P("")
    P("  [E2] anchor criterion-freeze: post-settling Sigma|dc|(ANCHOR) < (SEQ), matched cells")
    if not e2["matched"]:
        P("    %s" % e2["note"])
    else:
        for c in e2["matched"]:
            P("    %s/s%d: seq=%.3f  anchor=%.3f  ratio=%s  anchor<seq=%s"
              % (c["order"], c["seed"], c["seq"], c["anchor"], _f(c["ratio"], ".2f"), c["freeze"]))
        for o, v in sorted(e2["by_order"].items()):
            P("    stratum %s: anchor<seq in %s; seq %s vs anchor %s"
              % (o, sign_str(v["n_freeze"], v["n"]), v["seq_range"], v["anchor_range"]))
        P("    sign-consistency: anchor<seq in %s matched cells; pooled suppression ratio %s"
          % (sign_str(e2["n_freeze"], e2["n_matched"]), _f(e2["pooled_suppression_ratio"], ".2f")))

    e3 = ep["E3"]
    P("")
    P("  [E3] length-controlled rise: CHAIR_i@60 stage 1 -> stage %d in SEQ" % (T - 1))
    for c in e3["cells"]:
        b = c["boot"]
        P("    %s/s%d: %.4f -> %.4f  delta=%+.4f  %s  over-budget=%s"
          % (c["order"], c["seed"], c["chair_i60_from"], c["chair_i60_to"], c["delta"],
             ("CI95 [%+.4f, %+.4f] %s (B=%d, %s)" % (b["ci"][0], b["ci"][1],
                                                    "excl 0" if b["excludes_zero"] else "incl 0",
                                                    b["B"], b["impl"])) if b else "no bootstrap",
             c["frac_over_budget"]))
    for o, v in sorted(e3["by_order"].items()):
        P("    stratum %s: positive %s, CI>0 %s, delta range %s"
          % (o, sign_str(v["n_positive"], v["n"]), sign_str(v["n_ci"], v["n"]), v["delta_range"]))
    P("    VERDICT: %s" % e3["verdict"])

    e4 = ep["E4"]
    P("")
    P("  [E4] memorylessness: single-task control endpoint c vs SEQ same-last-task seed band")
    for c in e4["cells"]:
        P("    %s (s%d): c=%+.3f d'=%+.3f yes=%.1f%% (base c=%s)"
          % (c["arm"], c["seed"], c["c"], c["dprime"], c["yes_rate"] * 100, _f(c["base_c"], "+.3f")))
        for b in c["bands"]:
            P("      vs SEQ %s stage %d (seeds %s): c values %s band %s -> %s; SEQ endpoint band %s"
              % (b["order"], b["seq_stage"], b["seeds"], b["seq_c_values"], b["band"],
                 "INSIDE" if b["inside"] else ("outside" if b["inside"] is not None else "no band"),
                 b["seq_endpoint_band"]))
    P("    %s" % e4["note"])

    e5 = ep["E5"]
    P("")
    P("  [E5] refusal leakage (conditional endpoint)")
    P("    %s" % e5["note"])
    for c in e5["cells"]:
        P("    %s/s%d: endpoint %s  excess vs pre-refusal stage %s: %s"
          % (c["order"], c["seed"],
             " ".join("%s=%.3f" % (t, v) for t, v in sorted(c["endpoint"].items())) or "--",
             c["first_refusal_stage"] - 1 if c["first_refusal_stage"] else "?",
             " ".join("%s=%+.3f" % (t, v) for t, v in sorted(c["excess"].items())) or "--"))
    return L


def _sc(v, spec=".3f"):
    if v is None:
        return "--"
    return "%s %s" % (format(v["mean"], spec), "[%s-%s]" % (format(v["range"][0], spec),
                                                          format(v["range"][1], spec)))


def scorecard_lines(bb, suite, sc, sinfo):
    L = []
    P = L.append
    P("")
    P("  " + "=" * 74)
    P("  METHOD SCORECARD  %s / %s  (complete cells only: all %d stages EVAL_DONE + base; "
      "mean [min-max]; vs-seq = paired sign counts on matched order x seed cells)"
      % (bb, suite, sinfo["n_stages"]))
    P("  " + "=" * 74)
    if not sc:
        P("    no arm cells present")
        return L
    hdr = ("  %-12s %5s %-18s %-18s %-8s %-18s %-18s %-18s %-18s %-24s"
           % ("arm", "cells", "Sum|dc|", "post-settle", "d'maxdev", "end CHAIR_i",
              "end CHAIR@60", "plasticity", "end task mean", "vs seq (dc|post|ch60|plast)"))
    P(hdr)
    for r in sc:
        vs = r.get("vs_seq")
        if vs and vs["n_matched"]:
            vss = "%s|%s|%s|%s n=%d" % (sign_str(*vs["sum_abs_dc_lower"]),
                                        sign_str(*vs["post_settle_lower"]),
                                        sign_str(*vs["chair_i60_lower"]),
                                        sign_str(*vs["plasticity_not_worse"]), vs["n_matched"])
        elif r["arm"] == "seq":
            vss = "(reference)"
        else:
            vss = "no matched seq cell"
        P("  %-12s %2d/%-2d %-18s %-18s %-8s %-18s %-18s %-18s %-18s %-24s"
          % (r["arm"][:12], r["n_complete"], r["n_cells"], _sc(r["sum_abs_dc"]),
             _sc(r["post_settle"]),
             "--" if r["max_abs_dd"] is None else "%.3f" % r["max_abs_dd"]["range"][1],
             _sc(r["endpoint_chair_i"], ".4f"), _sc(r["endpoint_chair_i60"], ".4f"),
             _sc(r["plasticity"]), _sc(r["endpoint_task_mean"]), vss))
    P("    Sum|dc| per cell:")
    for r in sc:
        cells = [c for c in r["cells"] if c.get("sum_abs_dc") is not None]
        if cells:
            P("      %-12s %s" % (r["arm"][:12], "  ".join(
                "%s/s%d=%.3f%s" % (c["order"], c["seed"], c["sum_abs_dc"], "" if c["complete"] else "*")
                for c in cells)))
    if any(not c["complete"] for r in sc for c in r["cells"]):
        P("    (* = partial cell, excluded from means)")
    P("    columns: Sum|dc| = sum_k |c_k - c_{k-1}| (c_0 = base); post-settle drops the first step; "
      "d'maxdev = max over cells of max_k |d'_k - d'_base| (F1 trips at %.2f); plasticity = mean "
      "accuracy on the just-trained task%s; vs seq = cells with lower Sum|dc| | lower post-settle | "
      "lower endpoint CHAIR_i@60 | plasticity not worse than seq - %.2f."
      % (F1_DPRIME_TOL, " (UCIT: closed-ended tasks only)" if suite == "ucit" else "",
         PLASTICITY_TOL))
    return L


def coverage_of(matrix, suites):
    cov = {}
    for bb in sorted(matrix):
        for suite, sd in sorted(matrix[bb]["suites"].items()):
            T = suites[suite]["n_stages"]
            arms = {}
            for (arm, order, seed), stages in sd["arms"].items():
                a = arms.setdefault(arm, {"complete": 0, "partial": 0, "cells": []})
                complete = len(FC.contiguous_prefix(stages)) == T or arm.startswith("single_")
                a["complete" if complete else "partial"] += 1
                a["cells"].append("%s/s%d:%d" % (order, seed, len(stages)))
            cov["%s/%s" % (bb, suite)] = {"base": bool(matrix[bb]["base"]), "arms": arms}
    return cov


def coverage_lines(cov, in_progress, malformed, unparsed, notes):
    L = ["COVERAGE (complete/partial cells per arm; base present?)"]
    for key, v in cov.items():
        L.append("  %-18s base=%s  %s" % (key, "yes" if v["base"] else "NO", "  ".join(
            "%s=%d/%d" % (a, d["complete"], d["complete"] + d["partial"])
            for a, d in sorted(v["arms"].items(), key=lambda kv: FC.arm_sort_key(kv[0])))))
    if in_progress:
        L.append("  IN-PROGRESS (no EVAL_DONE, skipped): %d dirs: %s"
                 % (len(in_progress), ", ".join(in_progress[:12]) + (" ..." if len(in_progress) > 12 else "")))
    if unparsed:
        L.append("  UNPARSED dir names (ignored): %s" % unparsed)
    for n in notes:
        L.append("  NOTE: %s" % n)
    if malformed:
        L.append("  " + "!" * 70)
        L.append("  MALFORMED cells EXCLUDED (never averaged; exit code 3):")
        for m in malformed:
            L.append("    %s: %s" % (m["dir"], m["error"]))
        L.append("  " + "!" * 70)
    return L


def bench_lines(bench):
    if not bench:
        return ["EXTERNAL BENCHMARKS: none found (results_fs_bench absent or empty)"]
    L = ["EXTERNAL BENCHMARKS (results_fs_bench; endpoint cells)"]
    L.append("  %-30s %-5s %10s %10s %10s %s" % ("label", "done", "ObjHal_i", "nonCOCO_i", "MME-Hall", "note"))
    for label, b in sorted(bench.items()):
        if b is None:
            continue
        mme = b["mme"]
        L.append("  %-30s %-5s %10s %10s %10s %s"
                 % (label, "yes" if b["done"] else "no",
                    _f(b["objhal"]["chair_i"] if b["objhal"] else None, ".4f"),
                    _f(b["noncoco"]["chair_i"] if b["noncoco"] else None, ".4f"),
                    _f(mme["score"] if mme else None, ".1f"),
                    ("" if not mme or mme.get("full_800_scale") in (True, None)
                     else "MME partial: %s" % mme.get("subtasks_present"))))
    return L


def cross_backbone_e1(report):
    """Per suite x order x active stage: sign matches summed over backbones
    (the prereg counts seed x backbone cells per ordering)."""
    out = {}
    for bb, bd in report["backbones"].items():
        for suite, sd in bd["suites"].items():
            e1 = sd["endpoints"]["E1"]
            if e1["mode"] != "dose-response":
                continue
            for o, kv in e1["by_order"].items():
                for k, v in kv.items():
                    slot = out.setdefault(suite, {}).setdefault(o, {}).setdefault(
                        k, {"n_match": 0, "n": 0, "backbones": []})
                    slot["n_match"] += v["n_match"]
                    slot["n"] += v["n"]
                    slot["backbones"].append(bb)
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    FC.add_layout_args(ap)
    ap.add_argument("--out", default=None, help="aggregate JSON [layout agg_default]")
    ap.add_argument("--summary", default=None,
                    help="write the endpoint + scorecard summary to this file too")
    ap.add_argument("--family_backbone", default=None,
                    help="fallback backbone for letterless families, e.g. lamF=llava15,mth=qwen25vl "
                         "(gate_info.json wins when present)")
    ap.add_argument("--e3_boot", type=int, default=10000,
                    help="paired bootstrap resamples for E3 (0 = skip)")
    ap.add_argument("--no_matrix", action="store_true", help="print endpoints/scorecards only")
    args = ap.parse_args()
    lay = FC.resolve_layout_args(args)
    out_path = os.path.abspath(args.out) if args.out else str(lay["agg_default"])
    fam_bb = FC.parse_family_backbone(args.family_backbone)
    t0 = time.time()

    suites = {"ucit": FC.load_suite("ucit", args.manifest_ucit),
              "pilot": FC.load_suite("pilot", args.manifest_pilot)}
    coco_gt = FC.load_coco_gt(args.coco_gt)
    expected = expected_counts(args, coco_gt, suites)

    lines = []
    P = lines.append
    P("=" * 78)
    P("FULL-STUDY AGGREGATION   layout=%s   results=%s" % (lay["name"], args.results))
    P("  bench=%s" % args.bench)
    P("  coco_gt=%s%s" % (args.coco_gt, "" if coco_gt else "  [ABSENT -> CHAIR skipped]"))
    P("  pope_dir=%s (expected POPE rows: %s)" % (args.pope_dir, expected["pope"]))
    P("  manifests: ucit=%s | pilot=%s" % (suites["ucit"]["source"], suites["pilot"]["source"]))
    P("=" * 78)

    matrix, in_progress, unparsed, malformed, notes = discover(
        args.results, fam_bb, coco_gt, suites, expected)
    bench = {}
    try:
        bench = FC.discover_bench(args.bench)
    except FC.MalformedResult as ex:
        malformed.append({"dir": "results_fs_bench", "error": str(ex)})

    cov = coverage_of(matrix, suites)
    lines += coverage_lines(cov, in_progress, malformed, unparsed, notes)
    lines += bench_lines(bench)
    print("\n".join(lines))
    summary = list(lines)

    report = {"results_dir": os.path.abspath(args.results), "bench_dir": args.bench,
              "layout": lay["name"], "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "in_progress": in_progress, "unparsed": unparsed, "malformed": malformed,
              "notes": notes, "f1_tol": F1_DPRIME_TOL,
              "e3": {"k_words": FC.CHAIR_K_WORDS, "boot_B": args.e3_boot},
              "expected_counts": expected,
              "suites": {s: {k: v for k, v in suites[s].items()} for s in suites},
              "coverage": cov,
              "bench": {k: v for k, v in bench.items() if v is not None},
              "backbones": {}}

    for bb in sorted(matrix):
        base = matrix[bb]["base"]
        bbrep = {"base": FC.strip_private(base) if base else None,
                 "arms": {}, "present_cells": {}, "endpoints": None, "suites": {}}
        for suite in sorted(matrix[bb]["suites"]):
            arms = matrix[bb]["suites"][suite]["arms"]
            sinfo = suites[suite]
            ep = eval_endpoints(base, arms, sinfo, args.e3_boot)
            sc = build_scorecard(base, arms, sinfo)
            mat = []
            if not args.no_matrix:
                print_matrix(bb, suite, base, arms, sinfo, mat)
                print("\n".join(mat))
            el = endpoint_lines(bb, suite, ep, sinfo) + scorecard_lines(bb, suite, sc, sinfo)
            print("\n".join(el))
            summary += el
            arms_ser = {"%s|%s|s%d" % (a, o, s): {str(k): FC.strip_private(v)
                                                for k, v in st.items()}
                        for (a, o, s), st in arms.items()}
            present = {"%s|%s|s%d" % (a, o, s): sorted(st) for (a, o, s), st in arms.items()}
            bbrep["suites"][suite] = {"arms": arms_ser, "present_cells": present,
                                      "endpoints": ep, "scorecard": sc,
                                      "coverage": cov.get("%s/%s" % (bb, suite))}
            if suite == "ucit":     # backward-compatible top-level view
                bbrep["arms"], bbrep["present_cells"], bbrep["endpoints"] = arms_ser, present, ep
        if bbrep["endpoints"] is None:
            bbrep["endpoints"] = eval_endpoints(base, {}, suites["ucit"], 0)
        report["backbones"][bb] = bbrep

    xb = cross_backbone_e1(report)
    report["E1_cross_backbone"] = xb
    if xb:
        xl = ["", "E1 cross-backbone tally (prereg: >=5 of 6 seed x backbone cells per ordering)"]
        for suite, od in sorted(xb.items()):
            for o, kv in sorted(od.items()):
                xl.append("  %s %s: " % (suite, o) + "  ".join(
                    "k%s %s (%s)" % (k, sign_str(v["n_match"], v["n"]), "+".join(v["backbones"]))
                    for k, v in sorted(kv.items())))
        print("\n".join(xl))
        summary += xl

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
    tail = ["", "=" * 78, "written %s  (%.1fs)" % (out_path, time.time() - t0)]
    if malformed:
        tail.append("MALFORMED cells were excluded -- see the banner above (exit 3)")
    print("\n".join(tail))
    summary += tail
    if args.summary:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary)) or ".", exist_ok=True)
        with open(args.summary, "w") as f:
            f.write("\n".join(summary) + "\n")
    sys.exit(3 if malformed else 0)


if __name__ == "__main__":
    main()
