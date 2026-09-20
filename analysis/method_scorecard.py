#!/usr/bin/env python3
"""Automatic scorecard every NEW method candidate is judged by.

Given a results_fs tree and a list of RUNTAG prefixes (arms), the script
discovers every (order, seed) cell that is complete (EVAL_DONE at every stage),
scores each stage with the audited pilot scorers, builds the per-arm scorecard,
pairs every candidate arm against a reference arm cell-by-cell, and emits a
PASS/FAIL verdict for the four pre-registered method predictions.

USAGE
  python analysis/method_scorecard.py \
      --results /abs/path/results_fs --suite pilot \
      --arms psQ_seq,psQ_anchor,mth_critp --ref psQ_seq \
      [--base qwen25vl_base] [--pope_dir ...] [--coco_gt ...] [--manifest ...] \
      --out /abs/path/out_dir
  python analysis/method_scorecard.py --write_reference analysis/reference_numbers.json

  Runs unchanged from the local repo (analysis/method_scorecard.py, scorers in
  pilot/) and from the cluster (~/cl-halluc/code/analysis/method_scorecard.py,
  scorers flat in ~/cl-halluc/code, PYTHONPATH=~/cl-halluc/code). Layout
  detection, RUNTAG parsing, the JSONL reader and the SDT primitives come from
  the sibling analysis/fs_common.py; nothing numerically load-bearing is
  re-implemented here.

PER-STAGE METRICS (per cell dir <RUNTAG>_k<K>/, stage 0 = <backbone>_base/)
  c, d', yes-rate      pooled POPE (parse_yn; clip to [1/2N, 1-1/2N] for the
                       z-transform only; raw H/FA kept). Mirrors
                       analysis/method_comparison.py::sdt.
  leakage              refusal leakage on the non-refusal short-answer task:
                       pilot suite = TextVQA outputs equal to "unanswerable"
                       (strip/lower; method_comparison.py::leakage);
                       ucit suite  = ABSENT (no such task on UCIT).
  CHAIR_i, CHAIR_i@60  per-mention CHAIR_i over full captions and over the
                       first 60 whitespace words (analysis/diag/
                       length_controlled_chair.py::chair_over).
  tokens               mean n_new_tokens of the CHAIR captions.
  new-task accuracy    accuracy on the task learned at that stage (plasticity):
                       pilot = the audited pilot scorers (mc_acc / vqa_acc /
                       caption_uf1); ucit = norm-containment-EM (fs_aggregate).

SCORECARD PER ARM (per cell + [min, max] across cells; orderings are strata)
  Sum|dc|  = sum_k |c_k - c_{k-1}|, c_0 = base      endpoint c
  max|d'-d'_base| (F1 flag at 0.30)                 endpoint leakage
  endpoint CHAIR_i@60                                mean plasticity

VERSUS THE REFERENCE ARM (matched (order, seed) cells only)
  paired per-cell differences arm - ref for Sum|dc|, endpoint CHAIR_i@60,
  mean plasticity; sign-consistency counts; paired eval-set bootstrap CI
  (images, ratio of sums, B=10000, random.Random(17)) for the CHAIR_i@60 gap.
  Style guide (design_notes/review_statistics.md sec 2): per-cell values and
  min-max ranges only, never a pooled cross-seed CI.

PASS/FAIL RULES (pre-registered method predictions)
  P1  Sum|dc|(arm) < Sum|dc|(ref) in EVERY matched cell (2026-09-08: '>=2/3' passed with p=0.5 under the null)
  P2  mean plasticity not more than 0.02 below ref in EVERY matched cell
      (one-sided: exceeding the reference never fails)
  P3  max_k |d'_k - d'_base| < 0.30 in EVERY complete cell of the arm
  P4  endpoint CHAIR_i@60 <= ref with the paired bootstrap CI excluding 0
      (upper bound < 0) in >= half of the matched cells
  A prediction whose inputs are missing (no base cell, no matched cell) is
  ABSENT, never PASS or FAIL. Overall: PASS if all four PASS, FAIL if any FAIL,
  otherwise INCOMPLETE.

OUTPUTS (in --out): scorecard.md, scorecard.json (with audit columns),
scorecard.png (matplotlib; skipped with a note when unavailable).

Python 3.9 + stdlib; matplotlib optional (PNG only).
"""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fs_common as FC  # noqa: E402

F1_DPRIME_TOL = 0.30
P2_PLASTICITY_TOL = 0.02
DEFAULT_K = 60
DEFAULT_B = 10_000
DEFAULT_SEED = 17
LEAK_TASK_BY_SUITE = {"pilot": "textvqa", "ucit": None}
LEAK_TOKEN = "unanswerable"

_HELPERS = None


# ---------------------------------------------------------------------------
# Helper imports: pilot scorers via fs_common; CHAIR@k + bootstrap from
# analysis/diag/length_controlled_chair.py when importable, else the verbatim
# local twins below (provenance recorded in scorecard.json).
# ---------------------------------------------------------------------------
def helpers():
    global _HELPERS
    if _HELPERS is not None:
        return _HELPERS
    sc = FC.import_pilot_scorers()
    src = "analysis/diag/length_controlled_chair.py"
    try:
        diag_dir = HERE / "diag"
        if str(diag_dir) not in sys.path:
            sys.path.append(str(diag_dir))
        import length_controlled_chair as LCC  # noqa: E402
        chair_over, boot_gap_ci, load_gt = LCC.chair_over, LCC.boot_gap_ci, LCC.load_gt
    except Exception as e:  # ImportError or a path/layout problem
        src = "local verbatim copy (diag import failed: %s)" % e
        chair_over, boot_gap_ci, load_gt = _chair_over_local, _boot_gap_ci_local, None
    if load_gt is None:
        def load_gt(path):
            raw = json.load(open(path))
            gt = {}
            for cid, rec in raw.items():
                objs = set(rec["objects"])
                for c in rec["captions"]:
                    objs |= set(sc["extract_object_mentions"](c))
                gt[cid] = objs
            return gt
    _HELPERS = dict(sc, chair_over=chair_over, boot_gap_ci=boot_gap_ci,
                    load_gt=load_gt, chair_helpers_source=src)
    return _HELPERS


def _chair_over_local(rows, gt, k=None):
    """Verbatim twin of length_controlled_chair.chair_over (used only when the
    diag module is not importable, e.g. an unsynced cluster checkout)."""
    ext = helpers()["extract_object_mentions"]
    n_caps = n_hal_caps = n_mentions = n_hal = 0
    n_at_budget = 0
    per_img = []
    for r in rows:
        text = r["output"]
        if k is not None:
            words = text.split()
            if len(words) > k:
                n_at_budget += 1
            text = " ".join(words[:k])
        mentions = ext(text)
        g = gt[str(r["coco_id"])]
        hal = [m for m in mentions if m not in g]
        n_caps += 1
        n_mentions += len(mentions)
        n_hal += len(hal)
        if hal:
            n_hal_caps += 1
        per_img.append((len(mentions), len(hal)))
    summ = {
        "chair_i": n_hal / max(1, n_mentions),
        "chair_s": n_hal_caps / max(1, n_caps),
        "mentions_per_caption": n_mentions / max(1, n_caps),
        "n_mentions": n_mentions,
        "n_hal_mentions": n_hal,
    }
    if k is not None:
        summ["frac_captions_over_budget"] = n_at_budget / max(1, n_caps)
    return summ, per_img


def _boot_gap_ci_local(per_a, per_b, b=DEFAULT_B, seed=DEFAULT_SEED):
    """Verbatim twin of length_controlled_chair.boot_gap_ci."""
    assert len(per_a) == len(per_b)
    rng = random.Random(seed)
    n = len(per_a)
    gaps = []
    for _ in range(b):
        idx = [rng.randrange(n) for _ in range(n)]
        ma = ha = mb = hb = 0
        for i in idx:
            ma += per_a[i][0]
            ha += per_a[i][1]
            mb += per_b[i][0]
            hb += per_b[i][1]
        gaps.append(ha / max(1, ma) - hb / max(1, mb))
    gaps.sort()
    return gaps[int(0.025 * b)], gaps[int(0.975 * b)]


# ---------------------------------------------------------------------------
# Per-metric scorers (assemble counts here; primitives imported)
# ---------------------------------------------------------------------------
def score_pope(path, audit):
    """Pooled POPE SDT. Mirrors method_comparison.sdt / fs_aggregate.pope_sdt."""
    parse_yn = helpers()["parse_yn"]
    hits = fas = nyes = nno = say = parsed = nfail = ntot = 0
    for r in FC.read_rows_dedup(path, audit):
        if "gt" not in r or "output" not in r:
            raise FC.MalformedResult("%s: POPE row missing 'gt'/'output'" % path)
        ntot += 1
        pred = parse_yn(r["output"])
        if pred is None:
            nfail += 1
            continue
        parsed += 1
        say += pred == "yes"
        if r["gt"] == "yes":
            nyes += 1
            hits += pred == "yes"
        elif r["gt"] == "no":
            nno += 1
            fas += pred == "yes"
        else:
            raise FC.MalformedResult("%s: unexpected POPE gt=%r" % (path, r["gt"]))
    if nyes == 0 or nno == 0:
        raise FC.MalformedResult("%s: degenerate POPE (nyes=%d, nno=%d)" % (path, nyes, nno))
    d, c, clipped = FC.sdt_from_counts(hits, nyes, fas, nno)
    return {"H": round(hits / nyes, 4), "FA": round(fas / nno, 4),
            "dprime": round(d, 4), "c": round(c, 4),
            "yes_rate": round(say / parsed, 4),
            "n_total": ntot, "n_parsed": parsed, "parse_fail": nfail,
            "n_gt_yes": nyes, "n_gt_no": nno, "clipped": clipped}


def score_leakage(path, audit):
    """Refusal leakage: fraction of outputs equal to 'unanswerable' after
    strip().lower(). Mirrors method_comparison.leakage exactly."""
    n = hit = 0
    for r in FC.read_rows_dedup(path, audit):
        if "output" not in r:
            raise FC.MalformedResult("%s: task row missing 'output'" % path)
        n += 1
        hit += r["output"].strip().lower() == LEAK_TOKEN
    if n == 0:
        raise FC.MalformedResult("%s: empty gen file" % path)
    return {"rate": round(hit / n, 4), "n": n, "n_leak": hit, "token": LEAK_TOKEN}


def score_chair(path, gt, k, audit):
    """Full CHAIR_i, CHAIR_i@k, mean tokens; per-image (mentions, hal) at k keyed
    by coco_id for the paired bootstrap."""
    H = helpers()
    rows = FC.read_rows_dedup(path, audit)
    if not rows:
        raise FC.MalformedResult("%s: empty CHAIR gen file" % path)
    for r in rows:
        if "coco_id" not in r or "output" not in r:
            raise FC.MalformedResult("%s: CHAIR row missing 'coco_id'/'output'" % path)
        if str(r["coco_id"]) not in gt:
            raise FC.MalformedResult("%s: coco_id %s absent from coco_gt" % (path, r["coco_id"]))
    full, _ = H["chair_over"](rows, gt, k=None)
    at_k, per_k = H["chair_over"](rows, gt, k=k)
    per_img = {str(r["coco_id"]): per_k[i] for i, r in enumerate(rows)}
    toks = [r.get("n_new_tokens") for r in rows]
    has_tok = all(t is not None for t in toks)
    trunc = [bool(r.get("truncated", False)) for r in rows]
    return {
        "chair_i": round(full["chair_i"], 4), "chair_s": round(full["chair_s"], 4),
        "mentions_per_caption": round(full["mentions_per_caption"], 3),
        "n_captions": len(rows), "n_mentions": full["n_mentions"],
        "n_hal_mentions": full["n_hal_mentions"],
        "k": k, "chair_i_at_k": round(at_k["chair_i"], 4),
        "chair_s_at_k": round(at_k["chair_s"], 4),
        "n_mentions_at_k": at_k["n_mentions"], "n_hal_at_k": at_k["n_hal_mentions"],
        "frac_over_budget": round(at_k["frac_captions_over_budget"], 4),
        "mean_new_tokens": round(sum(toks) / len(toks), 1) if has_tok else None,
        "truncation_rate": round(sum(trunc) / len(rows), 4),
        "mean_words": round(sum(len(r["output"].split()) for r in rows) / len(rows), 1),
        "_per_img_at_k": per_img,
    }


def score_task(path, task, suite, caption_tasks, audit):
    """New-task accuracy. pilot: the audited pilot scorer for that task
    (metrics_task.SCORERS); ucit: norm-containment-EM (fs_aggregate.task_acc).
    A pilot task whose rows lack the scorer's meta keys falls back to
    containment-EM with the fallback NAMED in 'metric' (never silent)."""
    H = helpers()
    rows = FC.read_rows_dedup(path, audit)
    if not rows:
        raise FC.MalformedResult("%s: empty task gen file" % path)
    for r in rows:
        if "output" not in r or "meta" not in r:
            raise FC.MalformedResult("%s: task row missing output/meta" % path)
    scorer = H["SCORERS"].get(task) if suite == "pilot" else None
    need = {"scienceqa": "answer_letter", "textvqa": "answers", "vizwiz": "answers",
            "flickr": "refs"}.get(task)
    fallback = None
    if scorer is not None and need is not None and any(need not in r["meta"] for r in rows):
        fallback = "meta.%s absent in some rows" % need
        scorer = None
    if scorer is not None:
        res = scorer(rows)
        out = {"acc": res["score"], "metric": res["metric"], "n": res["n"],
               "parse_fail_rate": res.get("parse_fail_rate", 0.0)}
    else:
        norm = H["norm_answer"]
        ok = empty = 0
        for r in rows:
            if "answers" not in r["meta"]:
                raise FC.MalformedResult("%s: task row missing meta.answers" % path)
            pred = norm(r["output"])
            if not pred:
                empty += 1
            anss = [norm(a) for a in r["meta"]["answers"]]
            if any(a and (pred == a or a in pred) for a in anss):
                ok += 1
        out = {"acc": round(ok / len(rows), 4), "n": len(rows), "empty_pred": empty,
               "metric": "norm-containment-EM" + (" (fallback: %s)" % fallback if fallback else "")}
    out["open_ended"] = task in caption_tasks
    return out


def pope_rowcount_expected(pope_dir):
    p = Path(pope_dir) / "prompts.jsonl" if pope_dir else None
    if p and p.is_file():
        return FC.count_jsonl_ids(p)
    return None


def infer_backbone(cell_dir):
    """Backbone recorded by eval_gen in gate_info.json argv (--backbone X)."""
    gi = Path(cell_dir) / "gate_info.json"
    if not gi.is_file():
        return None
    try:
        argv = json.load(open(gi)).get("argv", [])
        if "--backbone" in argv:
            return argv[argv.index("--backbone") + 1]
    except Exception:
        return None
    return None


def score_cell(cell_dir, suite, gt, k, leak_task, caption_tasks, task_learned,
               pope_expected):
    """Score one checkpoint dir. ABSENT (None) where a gen file is missing;
    raises MalformedResult on corrupt content."""
    cell_dir = Path(cell_dir)
    audit = {}
    out = {"dir": cell_dir.name, "task_learned": task_learned}
    p = cell_dir / "pope_gen.jsonl"
    out["pope"] = score_pope(p, audit) if p.is_file() else None
    if out["pope"] is not None and pope_expected is not None:
        out["pope"]["rowcount_expected"] = pope_expected
        out["pope"]["rowcount_ok"] = out["pope"]["n_total"] == pope_expected
    p = cell_dir / "chair_gen.jsonl"
    out["chair"] = score_chair(p, gt, k, audit) if p.is_file() else None
    if leak_task:
        p = cell_dir / ("%s_gen.jsonl" % leak_task)
        out["leak"] = score_leakage(p, audit) if p.is_file() else None
        out["leak_task"] = leak_task
    else:
        out["leak"] = None
        out["leak_task"] = None
    out["new_task"] = None
    if task_learned:
        p = cell_dir / ("%s_gen.jsonl" % task_learned)
        out["new_task"] = (score_task(p, task_learned, suite, caption_tasks, audit)
                           if p.is_file() else None)
    out["audit"] = audit
    return out


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_arm(results_dir, prefix, n_stages, family_backbone):
    """All cells of one arm prefix. A cell is (order, seed) -> {k: dir}.
    Returns dict(prefix, info, complete{(order,seed): {k: Path}},
                 incomplete{(order,seed): reason}, backbones{set})."""
    results_dir = Path(results_dir)
    cells = {}
    reasons = {}
    info = None
    backbones = set()
    for entry in sorted(results_dir.iterdir()):
        if not entry.is_dir():
            continue
        pc = FC.parse_cell_name(entry.name)
        if pc is None:
            continue
        runtag, kind, val = pc
        if not (runtag == prefix or runtag.startswith(prefix + "_o")):
            continue
        pr = FC.parse_runtag(runtag, family_backbone)
        if pr is None or not runtag.startswith(prefix + "_o"):
            continue
        info = info or pr
        key = (pr["order"], pr["seed"])
        if kind != "k":
            reasons[key] = "%s cell (%s) is not a staged sequential run; skipped" % (kind, entry.name)
            continue
        bb = infer_backbone(entry)
        if bb:
            backbones.add(bb)
        d = cells.setdefault(key, {})
        if (entry / "EVAL_DONE").exists():
            d[val] = entry
        else:
            reasons.setdefault(key, "IN-PROGRESS: %s lacks EVAL_DONE" % entry.name)
    complete, incomplete = {}, {}
    for key, stages in sorted(cells.items()):
        missing = [k for k in range(1, n_stages + 1) if k not in stages]
        extra = [k for k in stages if k > n_stages]
        if missing:
            why = reasons.get(key, "")
            incomplete[key] = ("missing EVAL_DONE stages %s" % missing) + ("; " + why if why else "")
        else:
            complete[key] = {k: stages[k] for k in range(1, n_stages + 1)}
            if extra:
                incomplete.setdefault(key, "")  # keep key visible
                incomplete[key] = "NOTE extra stages beyond suite length ignored: %s" % extra
    for key, why in reasons.items():
        if key not in cells:
            incomplete[key] = why
    return {"prefix": prefix, "info": info, "complete": complete,
            "incomplete": incomplete, "backbones": backbones}


def stage_task(suite_info, order, k):
    tl = FC.order_tasks(suite_info, order)
    if tl and 1 <= k <= len(tl):
        return tl[k - 1]
    return None


# ---------------------------------------------------------------------------
# Scorecard per cell / per arm
# ---------------------------------------------------------------------------
def cell_scorecard(base, stages):
    """base: scored base cell or None; stages: {k: scored cell}. Returns the
    per-cell scorecard with None where an input is absent."""
    ks = sorted(stages)
    pope_ok = base is not None and base["pope"] is not None and all(
        stages[k]["pope"] is not None for k in ks)
    sc = {"stages_present": ks}
    if pope_ok:
        cs = [base["pope"]["c"]] + [stages[k]["pope"]["c"] for k in ks]
        ds = [stages[k]["pope"]["dprime"] for k in ks]
        sw = [abs(cs[i + 1] - cs[i]) for i in range(len(ks))]
        dd = [d - base["pope"]["dprime"] for d in ds]
        sc.update({
            "c_traj": [round(x, 4) for x in cs],
            "abs_dc_per_stage": [round(x, 4) for x in sw],
            "sum_abs_dc": round(sum(sw), 4),
            "sum_abs_dc_post_settling": round(sum(sw[1:]), 4) if len(sw) > 1 else round(sum(sw), 4),
            "endpoint_c": round(cs[-1], 4),
            "dd_from_base": [round(x, 4) for x in dd],
            "max_abs_dd": round(max(abs(x) for x in dd), 4),
            "f1_tripped": bool(max(abs(x) for x in dd) > F1_DPRIME_TOL),
        })
    else:
        sc.update({"c_traj": None, "abs_dc_per_stage": None, "sum_abs_dc": None,
                   "sum_abs_dc_post_settling": None,
                   "endpoint_c": (stages[ks[-1]]["pope"]["c"] if ks and stages[ks[-1]]["pope"] else None),
                   "dd_from_base": None, "max_abs_dd": None, "f1_tripped": None})
    last = stages[ks[-1]] if ks else None
    sc["endpoint_leak"] = last["leak"]["rate"] if last and last["leak"] else None
    sc["endpoint_chair_at_k"] = last["chair"]["chair_i_at_k"] if last and last["chair"] else None
    sc["endpoint_chair_i"] = last["chair"]["chair_i"] if last and last["chair"] else None
    accs = [stages[k]["new_task"]["acc"] if stages[k]["new_task"] else None for k in ks]
    sc["plasticity_per_stage"] = accs
    sc["mean_plasticity"] = (round(sum(accs) / len(accs), 4)
                             if accs and all(a is not None for a in accs) else None)
    return sc


SCORE_KEYS = ["sum_abs_dc", "sum_abs_dc_post_settling", "endpoint_c", "max_abs_dd",
              "endpoint_leak", "endpoint_chair_at_k", "endpoint_chair_i", "mean_plasticity"]


def arm_summary(cells_sc):
    """min/max across cells for each scorecard key (None-aware)."""
    out = {}
    for key in SCORE_KEYS:
        vals = [sc[key] for sc in cells_sc.values() if sc.get(key) is not None]
        out[key] = {"n": len(vals), "range": FC.rng_pair(vals)}
    trips = [sc["f1_tripped"] for sc in cells_sc.values() if sc["f1_tripped"] is not None]
    out["f1_any_tripped"] = any(trips) if trips else None
    out["f1_n_evaluable"] = len(trips)
    out["f1_n_tripped"] = sum(1 for t in trips if t)
    out["n_cells"] = len(cells_sc)
    return out


# ---------------------------------------------------------------------------
# Pairing against the reference arm
# ---------------------------------------------------------------------------
def paired_chair_gap(arm_last, ref_last, b, seed):
    """Paired image bootstrap of CHAIR_i@k(arm) - CHAIR_i@k(ref) at the endpoint."""
    pa, pb = arm_last["chair"]["_per_img_at_k"], ref_last["chair"]["_per_img_at_k"]
    ids = sorted(set(pa) & set(pb))
    if not ids:
        return None
    la = [pa[i] for i in ids]
    lb = [pb[i] for i in ids]
    point = (sum(h for _, h in la) / max(1, sum(m for m, _ in la))
             - sum(h for _, h in lb) / max(1, sum(m for m, _ in lb)))
    lo, hi = helpers()["boot_gap_ci"](la, lb, b, seed)
    return {"point": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "n_pairs": len(ids), "n_unpaired": len(set(pa) ^ set(pb)),
            "excludes_zero": bool(lo > 0 or hi < 0), "B": b, "seed": seed}


def pair_against_ref(arm, ref, b, seed):
    """arm/ref: {'cells': {(o,s): {'sc':..., 'stages':...}}}. Returns per matched
    cell diffs + sign counts."""
    matched = sorted(set(arm["cells"]) & set(ref["cells"]))
    rows = []
    for key in matched:
        a, r = arm["cells"][key]["sc"], ref["cells"][key]["sc"]
        row = {"order": key[0], "seed": key[1]}
        for k in ("sum_abs_dc", "endpoint_chair_at_k", "mean_plasticity", "endpoint_c",
                  "endpoint_leak", "sum_abs_dc_post_settling"):
            row["d_" + k] = (round(a[k] - r[k], 4) if a[k] is not None and r[k] is not None else None)
            row["arm_" + k] = a[k]
            row["ref_" + k] = r[k]
        al = arm["cells"][key]["stages"][max(arm["cells"][key]["stages"])]
        rl = ref["cells"][key]["stages"][max(ref["cells"][key]["stages"])]
        row["chair_at_k_boot"] = (paired_chair_gap(al, rl, b, seed)
                                  if al["chair"] and rl["chair"] else None)
        rows.append(row)

    def count(pred, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return {"n": len(vals), "count": sum(1 for v in vals if pred(v))}

    signs = {
        "sum_abs_dc_lower": count(lambda v: v < 0, "d_sum_abs_dc"),
        "chair_at_k_lower": count(lambda v: v < 0, "d_endpoint_chair_at_k"),
        "chair_at_k_ci_excl0_lower": {
            "n": sum(1 for r in rows if r["chair_at_k_boot"] is not None),
            "count": sum(1 for r in rows if r["chair_at_k_boot"] is not None
                         and r["chair_at_k_boot"]["ci95"][1] < 0)},
        "plasticity_within_tol": count(lambda v: v >= -P2_PLASTICITY_TOL, "d_mean_plasticity"),
        "plasticity_abs_within_tol": count(lambda v: abs(v) <= P2_PLASTICITY_TOL, "d_mean_plasticity"),
    }
    return {"n_matched": len(matched), "matched": [list(k) for k in matched],
            "unmatched_arm_cells": [list(k) for k in sorted(set(arm["cells"]) - set(ref["cells"]))],
            "unmatched_ref_cells": [list(k) for k in sorted(set(ref["cells"]) - set(arm["cells"]))],
            "rows": rows, "signs": signs}


# ---------------------------------------------------------------------------
# PASS / FAIL
# ---------------------------------------------------------------------------
RULES = {
    "P1": "Sum|dc|(arm) < Sum|dc|(ref) in EVERY matched cell (c_0 = base); count/n reported so a 2/3 result reads as PARTIAL",
    "P2": "mean plasticity >= ref - %.2f in every matched cell (one-sided)" % P2_PLASTICITY_TOL,
    "P3": "max_k |d'_k - d'_base| < %.2f in every complete cell of the arm (F1 falsifier)" % F1_DPRIME_TOL,
    "P4": "endpoint CHAIR_i@k <= ref AND paired bootstrap CI95 upper < 0 in >= half of matched cells",
}


def _verdict(ok, n, need_frac=None, need_all=False):
    if n == 0:
        return "ABSENT"
    if need_all:
        return "PASS" if ok == n else "FAIL"
    return "PASS" if ok >= need_frac * n - 1e-9 else "FAIL"


def judge(arm_sum, pairing):
    s = pairing["signs"] if pairing else None
    out = {}
    if s:
        out["P1"] = {"count": s["sum_abs_dc_lower"]["count"], "n": s["sum_abs_dc_lower"]["n"],
                     "verdict": _verdict(s["sum_abs_dc_lower"]["count"], s["sum_abs_dc_lower"]["n"], need_all=True)}
        out["P2"] = {"count": s["plasticity_within_tol"]["count"], "n": s["plasticity_within_tol"]["n"],
                     "verdict": _verdict(s["plasticity_within_tol"]["count"],
                                         s["plasticity_within_tol"]["n"], need_all=True)}
        out["P4"] = {"count": s["chair_at_k_ci_excl0_lower"]["count"],
                     "n": s["chair_at_k_ci_excl0_lower"]["n"],
                     "verdict": _verdict(s["chair_at_k_ci_excl0_lower"]["count"],
                                         s["chair_at_k_ci_excl0_lower"]["n"], 0.5)}
    else:
        for p in ("P1", "P2", "P4"):
            out[p] = {"count": 0, "n": 0, "verdict": "ABSENT"}
    trips = arm_sum["f1_any_tripped"]
    n3 = arm_sum["f1_n_evaluable"]
    out["P3"] = {"count": n3 - arm_sum["f1_n_tripped"], "n": n3,
                 "verdict": "ABSENT" if trips is None else ("FAIL" if trips else "PASS")}
    vs = [out[p]["verdict"] for p in ("P1", "P2", "P3", "P4")]
    out["overall"] = "PASS" if all(v == "PASS" for v in vs) else (
        "FAIL" if any(v == "FAIL" for v in vs) else "INCOMPLETE")
    for p in ("P1", "P2", "P3", "P4"):
        out[p]["rule"] = RULES[p]
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _f(v, spec=".3f", absent="ABSENT"):
    return absent if v is None else format(v, spec)


def _pct(v):
    return "ABSENT" if v is None else "%.1f%%" % (100 * v)


def _cellname(key):
    return "%s/s%d" % (key[0], key[1])


def write_markdown(rep, path):
    L = []
    hdr = rep["header"]
    L.append("# Method scorecard: %s suite, ref %s" % (hdr["suite"], hdr["ref"]))
    L.append("")
    L.append("- results: `%s`" % hdr["results"])
    L.append("- base (stage 0): `%s` %s" % (hdr["base_dir"], "(EVAL_DONE)" if hdr["base_present"] else "**ABSENT** (Sum|dc|, max|d'-d'_base|, P1, P3 not evaluable)"))
    L.append("- suite: %s (%d stages; orders from %s)" % (hdr["suite"], hdr["n_stages"], hdr["suite_source"]))
    L.append("- length budget k=%d words; paired bootstrap B=%d, seed=%d; CHAIR helpers: %s" % (
        hdr["k"], hdr["B"], hdr["seed"], hdr["chair_helpers_source"]))
    L.append("- leakage task: %s" % (hdr["leak_task"] or "none on this suite (reported ABSENT)"))
    if hdr.get("warnings"):
        L.append("")
        for w in hdr["warnings"]:
            L.append("- **WARNING** %s" % w)
    L.append("")
    L.append("Style: per-cell values and [min, max] across cells; orderings are strata; "
             "no pooled cross-seed CIs (design_notes/review_statistics.md sec 2).")
    L.append("")

    # Discovery
    L.append("## Discovery")
    L.append("")
    L.append("| arm | complete cells | incomplete / skipped |")
    L.append("|---|---|---|")
    for arm in rep["arm_order"]:
        a = rep["arms"][arm]
        comp = ", ".join(_cellname(tuple(k)) for k in a["complete_cells"]) or "none"
        inc = "; ".join("%s: %s" % (_cellname(tuple(k)), v) for k, v in a["incomplete"]) or "-"
        L.append("| %s%s | %s | %s |" % (arm, " (ref)" if arm == hdr["ref"] else "", comp, inc))
    L.append("")

    # Per-stage tables
    L.append("## Per-stage metrics")
    L.append("")
    for arm in rep["arm_order"]:
        a = rep["arms"][arm]
        for key, cell in a["cells"].items():
            L.append("### %s  cell %s" % (arm, key))
            L.append("")
            L.append("| stage | task learned | c | d' | d'-base | yes%% | leak%% | CHAIR_i | CHAIR_i@%d | tok | new-task acc | metric | audit |" % hdr["k"])
            L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            rows = ([("0", rep["base"])] if rep["base"] else []) + [(str(k), s) for k, s in cell["stages"].items()]
            bd = rep["base"]["pope"]["dprime"] if rep["base"] and rep["base"]["pope"] else None
            for k, s in rows:
                p, ch, lk, nt = s["pope"], s["chair"], s["leak"], s["new_task"]
                dd = (p["dprime"] - bd) if (p and bd is not None and k != "0") else None
                aud = []
                if p:
                    aud.append("parse_fail=%d" % p["parse_fail"])
                    if p["clipped"] != "none":
                        aud.append("clip=%s" % p["clipped"])
                    if p.get("rowcount_ok") is False:
                        aud.append("ROWCOUNT %d!=%d" % (p["n_total"], p["rowcount_expected"]))
                if ch:
                    aud.append("over_budget=%.0f%%" % (100 * ch["frac_over_budget"]))
                    if ch["truncation_rate"]:
                        aud.append("trunc=%.1f%%" % (100 * ch["truncation_rate"]))
                for ak, av in s.get("audit", {}).items():
                    aud.append("%s=%s" % (ak, av))
                L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    k, s.get("task_learned") or ("base" if k == "0" else "?"),
                    _f(p["c"], "+.3f") if p else "ABSENT", _f(p["dprime"]) if p else "ABSENT",
                    _f(dd, "+.3f", "-"), _pct(p["yes_rate"]) if p else "ABSENT",
                    _pct(lk["rate"]) if lk else "ABSENT",
                    _f(ch["chair_i"], ".4f") if ch else "ABSENT",
                    _f(ch["chair_i_at_k"], ".4f") if ch else "ABSENT",
                    _f(ch["mean_new_tokens"], ".1f") if ch else "ABSENT",
                    _f(nt["acc"], ".4f") if nt else ("-" if k == "0" else "ABSENT"),
                    (nt["metric"] + (" *open-ended floor*" if nt.get("open_ended") else "")) if nt else "-",
                    ", ".join(aud) or "-"))
            L.append("")

    # Scorecard per arm
    L.append("## Scorecard per arm (per cell, then [min, max] across cells)")
    L.append("")
    L.append("| arm | cell | Sum abs dc | post-settle Sum | endpoint c | max abs d'-base | F1 | endpoint leak | endpoint CHAIR_i@%d | endpoint CHAIR_i | mean plasticity | plasticity per stage |" % hdr["k"])
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in rep["arm_order"]:
        a = rep["arms"][arm]
        for key, cell in a["cells"].items():
            sc = cell["scorecard"]
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                arm, key, _f(sc["sum_abs_dc"]), _f(sc["sum_abs_dc_post_settling"]),
                _f(sc["endpoint_c"], "+.3f"), _f(sc["max_abs_dd"]),
                "ABSENT" if sc["f1_tripped"] is None else ("TRIPPED" if sc["f1_tripped"] else "ok"),
                _pct(sc["endpoint_leak"]), _f(sc["endpoint_chair_at_k"], ".4f"),
                _f(sc["endpoint_chair_i"], ".4f"), _f(sc["mean_plasticity"], ".4f"),
                ", ".join(_f(x, ".3f") for x in sc["plasticity_per_stage"])))
        s = a["summary"]

        def rg(k, spec=".3f"):
            r = s[k]["range"]
            return "ABSENT" if r is None else "[%s, %s] n=%d" % (format(r[0], spec), format(r[1], spec), s[k]["n"])
        L.append("| %s | **range** | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            arm, rg("sum_abs_dc"), rg("sum_abs_dc_post_settling"), rg("endpoint_c", "+.3f"),
            rg("max_abs_dd"),
            "ABSENT" if s["f1_any_tripped"] is None else "%d/%d tripped" % (s["f1_n_tripped"], s["f1_n_evaluable"]),
            rg("endpoint_leak", ".3f"), rg("endpoint_chair_at_k", ".4f"), rg("endpoint_chair_i", ".4f"),
            rg("mean_plasticity", ".4f"), "-"))
    L.append("")

    # Versus reference
    L.append("## Versus reference arm %s (paired per matched cell; arm - ref)" % hdr["ref"])
    L.append("")
    for arm in rep["arm_order"]:
        if arm == hdr["ref"]:
            continue
        pr = rep["arms"][arm]["pairing"]
        L.append("### %s vs %s" % (arm, hdr["ref"]))
        L.append("")
        if not pr or pr["n_matched"] == 0:
            L.append("no matched (order, seed) cell complete in both arms -> ABSENT")
            if pr:
                L.append("(arm-only cells: %s; ref-only cells: %s)" % (pr["unmatched_arm_cells"], pr["unmatched_ref_cells"]))
            L.append("")
            continue
        L.append("| cell | d Sum abs dc | d post-settle | d endpoint c | d CHAIR_i@%d | CI95 (paired image bootstrap) | n pairs | d mean plasticity | d endpoint leak |" % hdr["k"])
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in pr["rows"]:
            bt = r["chair_at_k_boot"]
            L.append("| %s/s%d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r["order"], r["seed"], _f(r["d_sum_abs_dc"], "+.3f"),
                _f(r["d_sum_abs_dc_post_settling"], "+.3f"), _f(r["d_endpoint_c"], "+.3f"),
                _f(r["d_endpoint_chair_at_k"], "+.4f"),
                ("[%+.4f, %+.4f]%s" % (bt["ci95"][0], bt["ci95"][1], " excl0" if bt["excludes_zero"] else "")) if bt else "ABSENT",
                str(bt["n_pairs"]) if bt else "-",
                _f(r["d_mean_plasticity"], "+.4f"), _f(r["d_endpoint_leak"], "+.4f")))
        s = pr["signs"]
        L.append("")
        L.append("sign consistency: Sum|dc| lower %d/%d; CHAIR_i@%d lower %d/%d (CI excl 0 and lower %d/%d); "
                 "plasticity >= ref-%.2f %d/%d (|d| <= %.2f: %d/%d)" % (
                     s["sum_abs_dc_lower"]["count"], s["sum_abs_dc_lower"]["n"], hdr["k"],
                     s["chair_at_k_lower"]["count"], s["chair_at_k_lower"]["n"],
                     s["chair_at_k_ci_excl0_lower"]["count"], s["chair_at_k_ci_excl0_lower"]["n"],
                     P2_PLASTICITY_TOL, s["plasticity_within_tol"]["count"], s["plasticity_within_tol"]["n"],
                     P2_PLASTICITY_TOL, s["plasticity_abs_within_tol"]["count"], s["plasticity_abs_within_tol"]["n"]))
        L.append("")

    # PASS / FAIL
    L.append("## PASS / FAIL per pre-registered prediction")
    L.append("")
    L.append("| arm | P1 Sum abs dc | P2 plasticity | P3 d' falsifier | P4 CHAIR_i@%d | overall |" % hdr["k"])
    L.append("|---|---|---|---|---|---|")
    for arm in rep["arm_order"]:
        j = rep["arms"][arm]["verdicts"]
        if arm == hdr["ref"]:
            L.append("| %s (ref) | - | - | %s (%d/%d ok) | - | reference |" % (
                arm, j["P3"]["verdict"], j["P3"]["count"], j["P3"]["n"]))
            continue
        L.append("| %s | %s (%d/%d) | %s (%d/%d) | %s (%d/%d ok) | %s (%d/%d) | **%s** |" % (
            arm, j["P1"]["verdict"], j["P1"]["count"], j["P1"]["n"],
            j["P2"]["verdict"], j["P2"]["count"], j["P2"]["n"],
            j["P3"]["verdict"], j["P3"]["count"], j["P3"]["n"],
            j["P4"]["verdict"], j["P4"]["count"], j["P4"]["n"], j["overall"]))
    L.append("")
    for p in ("P1", "P2", "P3", "P4"):
        L.append("- %s: %s" % (p, RULES[p]))
    L.append("- ABSENT = inputs missing (no base cell / no matched cell); never counts as PASS or FAIL. "
             "Overall PASS needs all four PASS; any FAIL = FAIL; else INCOMPLETE.")
    L.append("")

    # Pilot references
    if rep.get("pilot_reference"):
        pr = rep["pilot_reference"]
        L.append("## Pilot-run reference values (context only; NOT used in any verdict)")
        L.append("")
        L.append(pr.get("label", ""))
        L.append("")
        L.append("| pilot arm | Sum abs dc | endpoint c | max abs d'-base | endpoint leak | endpoint CHAIR_i@60 | endpoint CHAIR_i | mean plasticity |")
        L.append("|---|---|---|---|---|---|---|---|")
        for name, a in pr.get("arms", {}).items():
            sc = a["scorecard"]
            L.append("| %s (%s) | %s | %s | %s | %s | %s | %s | %s |" % (
                name, a["label"], _f(sc["sum_abs_dc"]), _f(sc["endpoint_c"], "+.3f"),
                _f(sc["max_abs_dd"]), _pct(sc["endpoint_leak"]),
                _f(sc.get("endpoint_chair_at_60"), ".4f"), _f(sc["endpoint_chair_i"], ".4f"),
                _f(sc["mean_plasticity"], ".4f")))
        L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def strip_private(obj):
    if isinstance(obj, dict):
        return {k: strip_private(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [strip_private(v) for v in obj]
    if isinstance(obj, tuple):
        return list(obj)
    return obj


def make_png(rep, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # matplotlib absent on the cluster env
        return "PNG skipped: matplotlib unavailable (%s)" % e
    try:
        import fig_style as FS
        FS.apply_style()
        palette = [FS.V2_COLOR, FS.ER_COLOR, FS.JOINT_COLOR, FS.V1_COLOR, "#56B4E9", "#000000"]
        ref_color, base_color, full_w = FS.SEQ_COLOR, FS.BASE_COLOR, FS.FULL_W
    except Exception:
        palette = ["#D55E00", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#000000"]
        ref_color, base_color, full_w = "#0072B2", "#999999", 6.08
    hdr = rep["header"]
    arms = rep["arm_order"]
    colors = {}
    ci = 0
    for a in arms:
        if a == hdr["ref"]:
            colors[a] = ref_color
        else:
            colors[a] = palette[ci % len(palette)]
            ci += 1
    fig, axs = plt.subplots(2, 2, figsize=(full_w, 4.6))
    ax = axs[0, 0]
    for a in arms:
        for key, cell in rep["arms"][a]["cells"].items():
            sc = cell["scorecard"]
            if sc["c_traj"]:
                ax.plot(range(len(sc["c_traj"])), sc["c_traj"], color=colors[a], alpha=0.75,
                        linewidth=1.0, marker="o", markersize=2.5)
    if rep["base"] and rep["base"]["pope"]:
        ax.axhline(rep["base"]["pope"]["c"], color=base_color, linewidth=0.8, linestyle=":")
    ax.set_xlabel("stage (0 = base)")
    ax.set_ylabel("criterion c")
    ax.set_title("criterion trajectory per cell", fontsize=8)
    for a in arms:
        ax.plot([], [], color=colors[a], label=a + (" (ref)" if a == hdr["ref"] else ""))
    ax.legend(fontsize=6, frameon=False)

    ax = axs[0, 1]
    for a in arms:
        for key, cell in rep["arms"][a]["cells"].items():
            xs, ys = [], []
            for k, s in cell["stages"].items():
                if s["chair"]:
                    xs.append(int(k))
                    ys.append(s["chair"]["chair_i_at_k"])
            if rep["base"] and rep["base"]["chair"]:
                xs = [0] + xs
                ys = [rep["base"]["chair"]["chair_i_at_k"]] + ys
            ax.plot(xs, ys, color=colors[a], alpha=0.75, linewidth=1.0, marker="o", markersize=2.5)
    ax.set_xlabel("stage (0 = base)")
    ax.set_ylabel("CHAIR_i@%d" % hdr["k"])
    ax.set_title("length-controlled CHAIR per cell", fontsize=8)

    ax = axs[1, 0]
    xt, xl = [], []
    pos = 0
    for a in arms:
        if a == hdr["ref"]:
            continue
        pr = rep["arms"][a]["pairing"]
        if pr:
            d1 = [r["d_sum_abs_dc"] for r in pr["rows"] if r["d_sum_abs_dc"] is not None]
            d2 = [r["d_mean_plasticity"] for r in pr["rows"] if r["d_mean_plasticity"] is not None]
            ax.scatter([pos] * len(d1), d1, color=colors[a], s=14, zorder=3)
            ax.scatter([pos + 1] * len(d2), d2, color=colors[a], s=14, marker="s", zorder=3)
        xt += [pos, pos + 1]
        xl += ["%s\ndSum|dc|" % a, "%s\ndplast" % a]
        pos += 2.5
    ax.axhline(0, color=base_color, linewidth=0.8)
    ax.axhline(-P2_PLASTICITY_TOL, color=base_color, linewidth=0.6, linestyle=":")
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=6)
    ax.set_title("paired d vs ref per cell (Sum|dc|, plasticity)", fontsize=8)

    ax = axs[1, 1]
    pos = 0
    xt, xl = [], []
    for a in arms:
        if a == hdr["ref"]:
            continue
        pr = rep["arms"][a]["pairing"]
        if pr:
            for i, r in enumerate(pr["rows"]):
                bt = r["chair_at_k_boot"]
                if bt:
                    x = pos + 0.15 * i
                    ax.errorbar([x], [bt["point"]], yerr=[[bt["point"] - bt["ci95"][0]], [bt["ci95"][1] - bt["point"]]],
                                fmt="o", color=colors[a], markersize=3, capsize=2, linewidth=0.9)
        xt.append(pos)
        xl.append(a)
        pos += 1.0
    ax.axhline(0, color=base_color, linewidth=0.8)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=6)
    ax.set_title("d CHAIR_i@%d vs ref per cell (CI95, images)" % hdr["k"], fontsize=8)

    verdict = "; ".join("%s: %s" % (a, rep["arms"][a]["verdicts"]["overall"])
                        for a in arms if a != hdr["ref"]) or "no candidate arm"
    fig.suptitle("Method scorecard (%s suite, ref %s) -- %s" % (hdr["suite"], hdr["ref"], verdict), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return "wrote %s" % path


# ---------------------------------------------------------------------------
# Pilot-run reference numbers (analysis/reference_numbers.json)
# ---------------------------------------------------------------------------
PILOT_ARMS = {"S": ("SEQ", "plain sequential"), "G": ("anchor-v2", "symmetric decision-axis anchor"),
              "J": ("JOINT", "matched joint control"), "E": ("ER", "experience replay, 100 exemplars"),
              "F": ("anchor-v1", "one-sided anchor (falsified)")}
PILOT_STAGE_TASK = {1: "scienceqa", 2: "textvqa", 3: "flickr", 4: "vizwiz"}


def build_reference_numbers(diag_dir):
    mc = json.load(open(Path(diag_dir) / "method_comparison.json"))
    lcc_p = Path(diag_dir) / "length_controlled_chair.json"
    lcc = json.load(open(lcc_p)) if lcc_p.is_file() else None
    g4_p = Path(diag_dir) / "g4_bootstrap_cis.json"
    g4 = json.load(open(g4_p)) if g4_p.is_file() else None
    rows = mc["rows"]
    out = {
        "label": ("PILOT-RUN REFERENCES: LLaVA-1.5-7B, pilot suite, order o1 "
                  "(scienceqa, textvqa, flickr, vizwiz), one seed, one cell per arm; exploratory "
                  "post-hoc method results from analysis/method_comparison.py (2026-08-31). "
                  "Context only for full-study scorecards: different backbone/seed/cell, "
                  "never an input to any PASS/FAIL verdict."),
        "source": {"method_comparison": "analysis/diag/method_comparison.json",
                   "length_controlled_chair": str(lcc_p.name) if lcc else None,
                   "g4_bootstrap_cis": str(g4_p.name) if g4 else None},
        "base": {"ckpt": "S0", **{k: rows["S0"][k] for k in ("H", "FA", "dprime", "c", "yes_rate", "chair_i", "tok")},
                 "chair_i_at_60": (round(lcc["checkpoints"]["S0"]["at_k"]["60"]["chair_i"], 4) if lcc else None)},
        "arms": {},
    }
    for letter, (name, desc) in PILOT_ARMS.items():
        cks = mc["arms"][letter]
        stages = []
        for i, ck in enumerate(cks):
            r = rows[ck]
            st = {"stage": i, "ckpt": ck, "task_learned": PILOT_STAGE_TASK.get(i),
                  "c": r["c"], "dprime": r["dprime"], "yes_rate": r["yes_rate"],
                  "leak_textvqa": r["leak_tvqa"], "chair_i": r["chair_i"], "tok": r["tok"],
                  "new_task_acc": r["tasks"][PILOT_STAGE_TASK[i]] if i else None,
                  "chair_i_at_60": (round(lcc["checkpoints"][ck]["at_k"]["60"]["chair_i"], 4) if lcc else None)}
            stages.append(st)
        cs = [s["c"] for s in stages]
        sw = [abs(cs[i + 1] - cs[i]) for i in range(len(cs) - 1)]
        dd = [s["dprime"] - stages[0]["dprime"] for s in stages[1:]]
        pl = [s["new_task_acc"] for s in stages[1:]]
        sc = {"sum_abs_dc": round(sum(sw), 4), "abs_dc_per_stage": [round(x, 4) for x in sw],
              "sum_abs_dc_post_settling": round(sum(sw[1:]), 4),
              "endpoint_c": cs[-1], "max_abs_dd": round(max(abs(x) for x in dd), 4),
              "f1_tripped": bool(max(abs(x) for x in dd) > F1_DPRIME_TOL),
              "endpoint_leak": stages[-1]["leak_textvqa"],
              "endpoint_chair_i": stages[-1]["chair_i"],
              "endpoint_chair_at_60": stages[-1]["chair_i_at_60"],
              "mean_plasticity": round(sum(pl) / len(pl), 4)}
        out["arms"][name] = {"letter": letter, "label": desc, "checkpoints": cks,
                             "stages": stages, "scorecard": sc}
    if g4:
        out["endpoint_chair_at_60_paired_bootstrap"] = {
            "note": "G4 (anchor-v2) minus S4 / J4 at k=60; paired images, B=10000, seed 17",
            **g4}
    if lcc and "bootstrap" in lcc:
        out["length_controlled_bootstrap_measurement"] = lcc["bootstrap"]
    return out


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=["pilot", "ucit"], default=None,
                    help="default: inferred from the arm families (ps/mth -> pilot, fs/lamF -> ucit)")
    ap.add_argument("--arms", default=None, help="comma-separated RUNTAG prefixes, e.g. psQ_seq,psQ_anchor,mth_critp")
    ap.add_argument("--ref", default=None, help="reference arm prefix (default: the first --arms entry containing 'seq', else the first)")
    ap.add_argument("--base", default=None, help="base cell dir name (llava15_base|qwen25vl_base); default from the arms' backbone")
    ap.add_argument("--manifest", default=None, help="suite manifest json (overrides --manifest_pilot/--manifest_ucit)")
    ap.add_argument("--out", default=None, help="output dir (scorecard.md/json/png)")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="length budget in whitespace words")
    ap.add_argument("--B", type=int, default=DEFAULT_B)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--leak_task", default=None, help="override the leakage task gen name (default by suite)")
    ap.add_argument("--family_backbone", default=None, help="e.g. mth=qwen25vl (fs_common convention)")
    ap.add_argument("--reference", default=None, help="pilot reference json to echo (default: analysis/reference_numbers.json if present)")
    ap.add_argument("--no_png", action="store_true")
    ap.add_argument("--write_reference", default=None, metavar="PATH",
                    help="write the pilot-run reference numbers json from analysis/diag/ and exit")
    FC.add_layout_args(ap, ("results", "coco_gt", "pope_dir", "manifest_ucit", "manifest_pilot"))
    args = ap.parse_args(argv)

    if args.write_reference:
        ref = build_reference_numbers(HERE / "diag")
        Path(args.write_reference).parent.mkdir(parents=True, exist_ok=True)
        json.dump(ref, open(args.write_reference, "w"), indent=1)
        print("wrote %s (%d arms)" % (args.write_reference, len(ref["arms"])))
        return 0

    if not args.arms or not args.out:
        ap.error("--arms and --out are required (or --write_reference)")
    lay = FC.resolve_layout_args(args, ("results", "coco_gt", "pope_dir", "manifest_ucit", "manifest_pilot"))
    if not args.results or not Path(args.results).is_dir():
        raise SystemExit("--results dir not found: %r (layout=%s)" % (args.results, lay["name"]))
    if not args.coco_gt or not Path(args.coco_gt).is_file():
        raise SystemExit("--coco_gt not found: %r" % args.coco_gt)
    fb = FC.parse_family_backbone(args.family_backbone)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    ref = args.ref or next((a for a in arms if "seq" in a), arms[0])
    if ref not in arms:
        arms.insert(0, ref)
    warnings = []

    # suite
    fam_suites = {}
    for a in arms:
        pr = FC.parse_runtag(a + "_o1_s0", fb)
        fam_suites[a] = pr["suite"] if pr else None
    suite = args.suite
    if suite is None:
        ss = {s for s in fam_suites.values() if s}
        if len(ss) != 1:
            raise SystemExit("cannot infer --suite from arm families %s; pass --suite" % fam_suites)
        suite = ss.pop()
    for a, s in fam_suites.items():
        if s and s != suite:
            warnings.append("arm %s belongs to a %s-suite family but --suite %s was given; scored as %s" % (a, s, suite, suite))
    manifest = args.manifest or (args.manifest_pilot if suite == "pilot" else args.manifest_ucit)
    suite_info = FC.load_suite(suite, manifest)
    n_stages = suite_info["n_stages"]
    caption = set(suite_info["caption_tasks"])
    leak_task = args.leak_task if args.leak_task is not None else LEAK_TASK_BY_SUITE[suite]

    # discovery
    disc = {a: discover_arm(args.results, a, n_stages, fb) for a in arms}
    bbs = set()
    for a in arms:
        bbs |= disc[a]["backbones"]
        if not disc[a]["backbones"] and disc[a]["info"]:
            bbs.add(disc[a]["info"]["backbone"])
    if args.base:
        base_dir = Path(args.results) / args.base
        if len(bbs) > 1:
            warnings.append("arms span backbones %s; base forced to %s -- CROSS-BACKBONE PAIRING IS NOT VALID" % (sorted(bbs), args.base))
    else:
        if len(bbs) > 1:
            raise SystemExit("arms span more than one backbone %s (from gate_info.json / family defaults); "
                             "pass --family_backbone mth=<bb> or --base explicitly" % sorted(bbs))
        bb = bbs.pop() if bbs else None
        if bb is None:
            raise SystemExit("no cell found for any arm in %s; cannot infer the base dir (pass --base)" % args.results)
        base_dir = Path(args.results) / FC.BASE_DIRS.get(bb, bb + "_base")

    H = helpers()
    gt = H["load_gt"](args.coco_gt)
    pope_expected = pope_rowcount_expected(args.pope_dir)
    base_present = (base_dir / "EVAL_DONE").exists()
    base = (score_cell(base_dir, suite, gt, args.k, leak_task, caption, None, pope_expected)
            if base_present else None)
    if not base_present:
        warnings.append("base cell %s absent or without EVAL_DONE" % base_dir)

    report = {"header": {
        "results": str(args.results), "suite": suite, "n_stages": n_stages,
        "suite_source": suite_info["source"], "base_dir": str(base_dir), "base_present": base_present,
        "ref": ref, "arms": arms, "k": args.k, "B": args.B, "seed": args.seed,
        "leak_task": leak_task, "coco_gt": str(args.coco_gt), "pope_dir": args.pope_dir,
        "pope_rows_expected": pope_expected, "chair_helpers_source": H["chair_helpers_source"],
        "f1_tol": F1_DPRIME_TOL, "p2_tol": P2_PLASTICITY_TOL, "layout": lay["name"],
        "family_backbone": fb, "warnings": warnings},
        "base": base, "arm_order": arms, "arms": {}}

    scored = {}
    for a in arms:
        d = disc[a]
        cells = {}
        for key, stages in d["complete"].items():
            st = {}
            for k, path in stages.items():
                if path.name not in scored:
                    scored[path.name] = score_cell(path, suite, gt, args.k, leak_task, caption,
                                                   stage_task(suite_info, key[0], k), pope_expected)
                st[k] = scored[path.name]
            cells[key] = {"stages": st, "sc": cell_scorecard(base, st)}
        report["arms"][a] = {"cells": cells, "info": d["info"],
                             "complete_cells": [list(k) for k in cells],
                             "incomplete": [[list(k), v] for k, v in sorted(d["incomplete"].items())],
                             "backbones_seen": sorted(d["backbones"]),
                             "summary": arm_summary({k: v["sc"] for k, v in cells.items()})}
    for a in arms:
        A = report["arms"][a]
        A["pairing"] = (pair_against_ref(A, report["arms"][ref], args.B, args.seed)
                        if a != ref and A["cells"] and report["arms"][ref]["cells"] else None)
        A["verdicts"] = judge(A["summary"], A["pairing"])

    # echo pilot references (context only)
    ref_path = args.reference or (HERE / "reference_numbers.json")
    if Path(ref_path).is_file():
        report["pilot_reference"] = json.load(open(ref_path))

    # serialise: cell keys -> "o1/s17", stage keys -> str
    ser = {"header": report["header"], "base": strip_private(base), "arm_order": arms, "arms": {},
           "pilot_reference": report.get("pilot_reference")}
    for a in arms:
        A = report["arms"][a]
        ser["arms"][a] = {
            "info": A["info"], "complete_cells": A["complete_cells"], "incomplete": A["incomplete"],
            "backbones_seen": A["backbones_seen"], "summary": A["summary"],
            "pairing": strip_private(A["pairing"]), "verdicts": A["verdicts"],
            "cells": {_cellname(k): {"stages": {str(kk): strip_private(v) for kk, v in c["stages"].items()},
                                     "scorecard": c["sc"]} for k, c in A["cells"].items()}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(ser, open(out / "scorecard.json", "w"), indent=1)
    write_markdown(ser, out / "scorecard.md")
    png_msg = "PNG skipped (--no_png)" if args.no_png else make_png(ser, out / "scorecard.png")

    # console summary
    print("=" * 78)
    print("METHOD SCORECARD  suite=%s  ref=%s  base=%s%s" % (suite, ref, base_dir.name, "" if base_present else " (ABSENT)"))
    print("=" * 78)
    for w in warnings:
        print("WARNING:", w)
    for a in arms:
        A = ser["arms"][a]
        print("\n%s%s: complete cells %s" % (a, " (ref)" if a == ref else "", A["complete_cells"] or "none"))
        for k, why in A["incomplete"]:
            print("   incomplete %s: %s" % (_cellname(tuple(k)), why))
        for cn, c in A["cells"].items():
            sc = c["scorecard"]
            print("   %s: Sum|dc|=%s  endpoint c=%s  max|dd'|=%s%s  leak=%s  CHAIR@%d=%s  plast=%s" % (
                cn, _f(sc["sum_abs_dc"]), _f(sc["endpoint_c"], "+.3f"), _f(sc["max_abs_dd"]),
                " TRIPPED" if sc["f1_tripped"] else "", _pct(sc["endpoint_leak"]), args.k,
                _f(sc["endpoint_chair_at_k"], ".4f"), _f(sc["mean_plasticity"], ".4f")))
        j = A["verdicts"]
        if a != ref:
            print("   verdicts: " + "  ".join("%s=%s(%d/%d)" % (p, j[p]["verdict"], j[p]["count"], j[p]["n"])
                                              for p in ("P1", "P2", "P3", "P4")) + "  -> " + j["overall"])
    print("\nwrote %s, %s; %s" % (out / "scorecard.md", out / "scorecard.json", png_msg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
