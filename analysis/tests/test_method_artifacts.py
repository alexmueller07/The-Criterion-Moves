#!/usr/bin/env python3
"""Synthetic end-to-end test for analysis/make_method_artifacts.py.

Builds OBVIOUSLY FAKE inputs under analysis/tests/fixtures/results_method_artifacts_synth/
(marker file inside; regenerated on every run) and writes every artifact under
analysis/tests/out/method_artifacts_synth/ -- never into paper/tables/.

  * two results trees -> two REAL scorecard.json files via method_scorecard.main
      tree A  llava15_base + psL_seq (o1 s17/s23/s31, o2 s17), psL_anchor x3, psL_er x2,
              psL_ewc x1, psL_lwf (k4 without EVAL_DONE -> 0 complete cells = ABSENT row),
              psL_cecf x1, mth_critp x3, mth_critp1 x1, mth_anchorcrit x2,
              psL_seq_neu x3 + psL_seq_amp x1 (intervention arms), psL_joint (final only)
      tree B  qwen25vl_base + psQ_seq x2, psQ_anchor x2, psQ_er x1
  * two logit dumps (psL_seq o1 s17 / s23) -> REAL logit_bias_analysis JSONs
  * two blind dumps (psL_seq o1 s17; psL_anchor o1 s17 with k2 too-few-aligned and
    k3 ABSENT) -> REAL blind_prior_share JSONs
  * a hand-built fs_aggregate.json (current schema) whose seq/anchor cells copy the
    scorecard (cross-check passes) plus a JOINT arm; a tampered twin must abort
  * neutral / amplified manifests derived from fullstudy/pilot_manifest.json

Asserts table structure (column counts, escaping), ABSENT rows, bold placement,
CSV/tex agreement, manifest sha256s, intervention markers, prior-share readings,
the three fail-loud paths, and the run on the method_scorecard fixture.

Run:  python3 analysis/tests/test_method_artifacts.py
"""
import csv
import json
import math
import os
import random
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(HERE))
import method_scorecard as MS  # noqa: E402
import make_method_artifacts as MA  # noqa: E402
import test_method_scorecard as T  # noqa: E402  (synthetic cell writer)
import logit_bias_analysis as LB  # noqa: E402
import blind_prior_share as BP  # noqa: E402

FIX = HERE / "fixtures" / "results_method_artifacts_synth"
OUT = HERE / "out" / "method_artifacts_synth"
MANIFEST = ROOT / "fullstudy" / "pilot_manifest.json"
FLAT = [(0.78, 0.09)] * 4
FLAT2 = [(0.80, 0.08)] * 4
NEU_HF = [(0.82, 0.12), (0.80, 0.10), (0.81, 0.11), (0.79, 0.10)]
# SEQ criterion trajectory for the o1 stage order (scienceqa, textvqa, flickr,
# vizwiz). It has to be DOSE-CONSISTENT with the manifest the intervention table
# reads, i.e. the pilot's documented pattern (PILOT_FINDINGS.md T1: "TextVQA
# pushes liberal (dc -0.37/-0.30/-0.37), VizWiz pushes conservative past base
# (+0.57/+0.59/+0.71), zero-dose stage moves small in every seed").
# test_method_scorecard.REF_HF walks a path with the SAME summary numbers
# (Sum|dc| 2.639, post-settling 2.419, max|dd'| 0.225, endpoint c +1.027,
# base c +0.220) but takes the TextVQA step CONSERVATIVE and the zero-dose
# Flickr step large -- fine for the scorer's arithmetic, wrong for a table that
# marks each step against its own stage's dose.
#           k1 scienceqa    k2 textvqa    k3 flickr     k4 vizwiz
#   c  ->     0.000         -0.696        -0.515        +1.027
SEQ_HF = [(0.85, 0.15), (0.95, 0.40), (0.94, 0.30), (0.50, 0.02)]
# same shape, but k2 also trips the d' falsifier (d'=3.335 vs base 2.123)
SEQ_HF_TRIP = [(0.85, 0.15), (0.98, 0.10), (0.94, 0.30), (0.50, 0.02)]
# amplified prior: same first three stages, a harsher conservative VizWiz step
AMP_HF = [(0.85, 0.15), (0.95, 0.40), (0.94, 0.30), (0.40, 0.01)]


def cells(prefix, seeds, hf, chair, tok, accs, leak=0.0, order="o1", done=True, backbone="llava15", late_k=None):
    for s in seeds:
        for k in range(1, 5):
            T.write_cell("%s_%s_s%d_k%d" % (prefix, order, s, k), hf[k - 1][0], hf[k - 1][1], chair, tok, accs,
                         leak=leak if k == 4 else 0.0, late=(k == late_k), done=(done or k < 4), backbone=backbone)


def build_trees():
    if FIX.exists():
        shutil.rmtree(FIX)
    FIX.mkdir(parents=True)
    (FIX / "THIS_IS_SYNTHETIC_FIXTURE_DATA.txt").write_text("fake inputs for test_method_artifacts.py\n")
    gt = {str(1000 + i): {"objects": ["dog", "cat"], "captions": ["a dog and a cat"]} for i in range(T.N_IMG)}
    acc_m = {t: round(v - 0.01, 4) for t, v in T.REF_ACC.items()}
    # ---- tree A (LLaVA) ----
    T.FIX = FIX / "tree_llava"
    T.FIX.mkdir()
    json.dump(gt, open(T.FIX / "coco_gt.json", "w"))
    T.write_cell("llava15_base", 0.80, 0.10, 0.5, 100, T.REF_ACC, extra_fail=2)
    cells("psL_seq", [17, 31], SEQ_HF, 0.5, 101, T.REF_ACC, leak=0.2, late_k=3)
    cells("psL_seq", [23], SEQ_HF_TRIP, 0.5, 101, T.REF_ACC, leak=0.2)
    cells("psL_seq", [17], SEQ_HF, 0.5, 100, T.REF_ACC, order="o2")
    cells("psL_anchor", [17, 23, 31], FLAT, 0.1, 90, acc_m, leak=0.04)
    cells("psL_er", [17, 23], SEQ_HF, 0.55, 100, T.REF_ACC, leak=0.15)
    cells("psL_ewc", [17], [(0.82, 0.12)] * 4, 0.5, 100, T.REF_ACC)
    cells("psL_lwf", [17], FLAT, 0.4, 100, T.REF_ACC, done=False)          # k4 lacks EVAL_DONE
    cells("psL_cecf", [17], [(0.70, 0.05)] * 4, 0.3, 95, acc_m)
    cells("mth_critp", [17, 23, 31], FLAT, 0.1, 90, acc_m, leak=0.04)
    cells("mth_critp1", [17], FLAT, 0.15, 90, acc_m)
    cells("mth_anchorcrit", [17, 23], FLAT2, 0.08, 90, acc_m)
    cells("psL_seq_neu", [17, 23, 31], NEU_HF, 0.5, 100, T.REF_ACC, leak=0.02)
    cells("psL_seq_amp", [17], AMP_HF, 0.5, 100, T.REF_ACC, leak=0.3)
    T.write_cell("psL_joint_o1_s17_final", 0.8, 0.1, 0.5, 100, T.REF_ACC)
    # ---- tree B (Qwen) ----
    T.FIX = FIX / "tree_qwen"
    T.FIX.mkdir()
    json.dump(gt, open(T.FIX / "coco_gt.json", "w"))
    T.write_cell("qwen25vl_base", 0.80, 0.10, 0.5, 100, T.REF_ACC, backbone="qwen25vl")
    cells("psQ_seq", [17, 23], SEQ_HF, 0.5, 100, T.REF_ACC, leak=0.2, backbone="qwen25vl")
    cells("psQ_anchor", [17, 23], FLAT, 0.1, 90, acc_m, leak=0.04, backbone="qwen25vl")
    cells("psQ_er", [17], SEQ_HF, 0.55, 100, T.REF_ACC, backbone="qwen25vl")


def run_scorecard(tree, arms, ref, out):
    argv = ["--results", str(tree), "--suite", "pilot", "--arms", arms, "--ref", ref,
            "--coco_gt", str(tree / "coco_gt.json"), "--manifest", str(MANIFEST), "--out", str(out),
            "--B", "300", "--no_png"]
    assert MS.main(argv) == 0
    return out / "scorecard.json"


# ---------------------------------------------------------------- logits / blind
def write_logits(path, ids, gaps, gts):
    with open(path, "w") as f:
        for i, g, gt in zip(ids, gaps, gts):
            f.write(json.dumps({"id": i, "gt": gt, "gap": g, "z_yes": g, "z_no": 0.0,
                                "argmax_is": "yes" if g > 0 else "no", "category": "adversarial"}) + "\n")


def build_logits_and_blind():
    # POPE dumps are 9k rows; the calibration split is calib_frac=0.1 of that, so
    # n must stay large enough that b* is fitted on a meaningful sample. At n=400
    # the calibration split is 40 items (20 per class) and, at d'=3 where the
    # rates sit at .93/.07, z() amplifies binomial noise into a c0_calib swing of
    # ~+-0.35 -- the labeled correction then "restores" a criterion 0.2 off base
    # and the post-hoc row's Sum|dc| is calibration noise, not signal.
    n = 3000
    ids = ["pope_%d" % i for i in range(n)]
    gts = ["yes"] * (n // 2) + ["no"] * (n // 2)
    rng = random.Random(3)
    base_real = [rng.gauss(1.5 if g == "yes" else -1.5, 1.0) for g in gts]
    base_blind = [rng.gauss(0.0, 0.5) for _ in gts]          # no evidence: d' ~ 0
    shifts = [-0.3, 1.0, -0.2, 1.5]                            # class-independent (additive theory holds)
    LOG = FIX / "logits"
    BL = FIX / "blind"
    for d in (LOG, BL):
        d.mkdir()
    (LOG / "llava15_base").mkdir()
    write_logits(LOG / "llava15_base" / "pope_logits.jsonl", ids, base_real, gts)
    (BL / "llava15_base").mkdir()
    write_logits(BL / "llava15_base" / "pope_logits.jsonl", ids, base_real, gts)
    write_logits(BL / "llava15_base" / "pope_logits_blind.jsonl", ids, base_blind, gts)
    logit_jsons = []
    for seed, jitter in ((17, 0.0), (23, 0.05)):
        rt = "psL_seq_o1_s%d" % seed
        stage_args = []
        for k in range(1, 5):
            d = LOG / ("%s_k%d" % (rt, k))
            d.mkdir()
            # jitter separates the two seeds and must stay CLASS-INDEPENDENT: a
            # +-jitter split by gt would be a real non-additive (class-dependent)
            # shift, which is what additive_test is built to detect.
            g = [b + shifts[k - 1] + jitter + rng.gauss(0, 0.05) for b in base_real]
            write_logits(d / "pope_logits.jsonl", ids, g, gts)
            stage_args += ["--stage", "%s_k%d=%s" % (rt, k, d / "pope_logits.jsonl")]
        prefix = OUT / "logits" / rt
        prefix.parent.mkdir(parents=True, exist_ok=True)
        sys.argv = ["logit_bias_analysis.py", "--base", str(LOG / "llava15_base" / "pope_logits.jsonl")] + stage_args + \
                   ["--runtag", rt, "--out_prefix", str(prefix), "--n_boot", "200", "--n_splits", "3", "--no_fig"]
        LB.main()
        logit_jsons.append(str(prefix) + ".json")
    # blind dumps: seq rho ~ 0.9 (prior-borne), anchor rho ~ 0.2 (evidence-conditional)
    blind_jsons = []
    for rt, rho, present in (("psL_seq_o1_s17", 0.9, {1: n, 2: n, 3: n, 4: n}),
                             ("psL_anchor_o1_s17", 0.2, {1: n, 2: 50, 4: n})):
        for k in range(1, 5):
            if k not in present:
                continue
            m = present[k]
            d = BL / ("%s_k%d" % (rt, k))
            d.mkdir()
            g = [b + shifts[k - 1] + rng.gauss(0, 0.05) for b in base_real]
            gb = [b + rho * shifts[k - 1] + rng.gauss(0, 0.05) for b in base_blind]
            write_logits(d / "pope_logits.jsonl", ids[:m], g[:m], gts[:m])
            write_logits(d / "pope_logits_blind.jsonl", ids[:m], gb[:m], gts[:m])
        outp = OUT / "blind" / (rt + ".json")
        outp.parent.mkdir(parents=True, exist_ok=True)
        sys.argv = ["blind_prior_share.py", "--results", str(BL), "--runtag", rt, "--base", "llava15_base",
                    "--stages", "k1,k2,k3,k4", "--out", str(outp)]
        BP.main()
        blind_jsons.append(str(outp))
    return logit_jsons, blind_jsons


# ---------------------------------------------------------------- aggregate / manifests
def build_aggregate(sc_path):
    rep = json.load(open(sc_path))

    def cells_of(prefix):
        out = []
        for cn, c in rep["arms"][prefix]["cells"].items():
            o, s = cn.split("/")
            sc = c["scorecard"]
            out.append({"order": o, "seed": int(s[1:]), "complete": True, "sum_abs_dc": sc["sum_abs_dc"],
                        "post_settle": sc["sum_abs_dc_post_settling"], "endpoint_c": sc["endpoint_c"],
                        "max_abs_dd": sc["max_abs_dd"], "endpoint_chair_i60": sc["endpoint_chair_at_k"],
                        "plasticity": sc["mean_plasticity"], "dc_steps": sc["abs_dc_per_stage"]})
        return out
    joint_cells = [{"order": "o1", "seed": 17, "complete": True, "sum_abs_dc": 0.5, "post_settle": 0.3,
                    "endpoint_c": -0.10, "max_abs_dd": 0.05, "endpoint_chair_i60": 0.15, "dc_steps": [0.2, 0.1, 0.1, 0.1]},
                   {"order": "o1", "seed": 23, "complete": True, "sum_abs_dc": 0.6, "post_settle": 0.4,
                    "endpoint_c": -0.05, "max_abs_dd": 0.35, "endpoint_chair_i60": 0.16, "dc_steps": [0.2, 0.2, 0.1, 0.1]},
                   {"order": "o1", "seed": 31, "complete": False}]
    agg = {"f1_tol": MS.F1_DPRIME_TOL, "suites": {"pilot": {"n_stages": 4}},
           "backbones": {"llava15": {"base": {}, "suites": {"pilot": {"arms": {}, "scorecard": [
               {"arm": "seq", "cells": cells_of("psL_seq"), "n_complete": 4},
               {"arm": "anchor", "cells": cells_of("psL_anchor"), "n_complete": 3},
               {"arm": "joint", "cells": joint_cells, "n_complete": 2,
                "vs_seq": {"n_matched": 2, "sum_abs_dc_lower": [2, 2], "post_settle_lower": [2, 2],
                           "chair_i_lower": [1, 2], "chair_i60_lower": [1, 2], "plasticity_not_worse": [0, 0],
                           "plasticity_tol": 0.02, "per_cell": []}}]}}}}}
    ok = OUT / "agg_ok.json"
    json.dump(agg, open(ok, "w"), indent=1)
    bad = json.loads(json.dumps(agg))
    bad["backbones"]["llava15"]["suites"]["pilot"]["scorecard"][0]["cells"][0]["sum_abs_dc"] += 0.01
    badp = OUT / "agg_bad.json"
    json.dump(bad, open(badp, "w"), indent=1)
    return ok, badp


def build_manifests():
    base = json.load(open(MANIFEST))
    neu = json.loads(json.dumps(base))
    neu["tasks"]["textvqa"]["answer_stats"].update({"yes_frac": 0.015, "no_frac": 0.015, "refusal_frac": 0.019})
    neu["tasks"]["vizwiz"]["answer_stats"].update({"yes_frac": 0.024, "no_frac": 0.024, "refusal_frac": 0.019})
    amp = json.loads(json.dumps(base))
    amp["tasks"]["textvqa"]["answer_stats"].update({"yes_frac": 0.051, "no_frac": 0.013, "refusal_frac": 0.022})
    amp["tasks"]["vizwiz"]["answer_stats"].update({"yes_frac": 0.017, "no_frac": 0.020, "refusal_frac": 0.58})
    pn, pa = OUT / "neutral_manifest.json", OUT / "amplified_manifest.json"
    json.dump(neu, open(pn, "w"))
    json.dump(amp, open(pa, "w"))
    return pn, pa


# ---------------------------------------------------------------- tex helpers
def tex_rows(path):
    """-> (header cells, [row cells...], spans) with a structural check."""
    txt = Path(path).read_text()
    assert txt.count("\\begin{tabular}") == 1 and txt.count("\\end{tabular}") == 1, path
    lines = [l for l in txt.splitlines() if not l.startswith("%")]
    body = lines[lines.index("\\toprule") + 1: lines.index("\\bottomrule")]
    header = [c.strip() for c in body[0][:-3].split(" & ")]
    rows, spans = [], []
    for l in body[1:]:
        if l in ("\\midrule",):
            continue
        assert l.endswith(" \\\\"), l
        if l.startswith("\\multicolumn"):
            spans.append(l)
            continue
        cells = [c.strip() for c in l[:-3].split(" & ")]
        assert len(cells) == len(header), (path, len(cells), len(header), l)
        # no bare underscore outside math
        stripped = re.sub(r"\$[^$]*\$", "", l)
        assert re.search(r"(?<!\\)_", stripped) is None, ("unescaped underscore", l)
        assert l.count("{") == l.count("}"), ("unbalanced braces", l)
        rows.append(cells)
    return header, rows, spans


def rows_by_label(rows):
    out = {}
    for r in rows:
        out.setdefault(re.sub(r"\$\^\{[a-z]\}\$$", "", r[0]), []).append(r)
    return out


def sha(path):
    return MA.sha256(path)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    build_trees()
    scA = run_scorecard(FIX / "tree_llava",
                        "psL_seq,psL_anchor,psL_er,psL_ewc,psL_lwf,psL_cecf,mth_critp,mth_critp1,mth_anchorcrit,"
                        "psL_seq_neu,psL_seq_amp,psL_joint", "psL_seq", OUT / "scorecard_llava")
    scB = run_scorecard(FIX / "tree_qwen", "psQ_seq,psQ_anchor,psQ_er", "psQ_seq", OUT / "scorecard_qwen")
    repA = json.load(open(scA))
    assert repA["arms"]["psL_lwf"]["complete_cells"] == []
    assert repA["arms"]["psL_joint"]["complete_cells"] == []
    print("scorecards built")
    logit_jsons, blind_jsons = build_logits_and_blind()
    agg_ok, agg_bad = build_aggregate(scA)
    man_neu, man_amp = build_manifests()
    print("inputs built")

    tables, out = OUT / "tables", OUT / "out"
    argv = ["--scorecard", "pilot-llava=%s" % scA, str(scB), "--logits"] + logit_jsons + ["--blind"] + blind_jsons + \
           ["--aggregate", str(agg_ok), "--manifest_neutral", str(man_neu), "--manifest_amplified", str(man_amp),
            "--tables_dir", str(tables), "--out_dir", str(out), "--with_pilot_reference"]
    assert MA.main(argv) == 0
    for f in ("method_main.tex", "method_main_notes.tex", "method_main.csv", "method_main_cells.csv",
              "method_main_posthoc_stages.csv", "method_intervention.tex", "method_intervention.csv",
              "method_intervention_paired.tex", "method_intervention_paired.csv", "method_prior_share.tex",
              "method_prior_share.csv"):
        assert (tables / f).is_file() and (tables / f).stat().st_size > 0, f
    for f in ("fig_method_scorecard.pdf", "fig_method_scorecard.png", "fig_method_scorecard_caption.txt",
              "fig_method_scorecard_pilot_qwen2_5_vl_7b.pdf", "fig_method_scorecard_pilot_qwen2_5_vl_7b.png",
              "method_artifacts_manifest.json"):
        assert (out / f).is_file() and (out / f).stat().st_size > 0, f
    print("artifacts present")

    # ---- main table structure + content ----
    header, rows, spans = tex_rows(tables / "method_main.tex")
    assert len(header) == 10 and header[0] == "Method"
    assert len(spans) == 3, spans            # two blocks + pilot reference span
    assert "pilot-llava" in spans[0] and "Qwen2.5-VL-7B" in spans[1]
    # block A = rows up to the second span: find by cells column patterns
    csvrows = list(csv.DictReader(open(tables / "method_main.csv")))
    A = [r for r in csvrows if r["block"] == "pilot-llava"]
    B = [r for r in csvrows if r["block"] == "pilot / Qwen2.5-VL-7B"]
    assert A and B
    tokA = {r["token"]: r for r in A}
    tokB = {r["token"]: r for r in B}
    assert tokA["seq"]["n_cells"] == "4" and tokA["seq"]["orders"] == "o1;o2" and tokA["seq"]["is_ref"] == "1"
    assert tokA["lwf"]["n_cells"] == "0" and tokA["lwf"]["notes"].startswith("ABSENT")
    assert tokA["joint"]["source"] == "aggregate" and tokA["joint"]["n_cells"] == "2" and tokA["joint"]["plasticity_n"] == "0"
    assert tokA["joint"]["better_sum_abs_dc_k"] == "2" and tokA["joint"]["better_chair_k"] == "1"
    assert tokA["joint"]["f1_tripped_k"] == "1"       # max_abs_dd 0.35 > 0.30 in one aggregate cell
    assert tokB["joint"]["n_cells"] == "0"            # no aggregate block for qwen -> ABSENT
    assert tokA["__posthoc_labeled"]["n_cells"] == "2" and tokA["__posthoc_labeled"]["chair60_n"] == "0"
    assert tokA["__posthoc_labeled"]["P2"] == "ABSENT" and tokA["__posthoc_labeled"]["P4"] == "ABSENT"
    assert tokA["__posthoc_labeled"]["P1"] == "PASS", tokA["__posthoc_labeled"]   # corrected churn < uncorrected
    assert tokA["__posthoc_labelfree"]["P3"] == "PASS"
    assert tokB["__posthoc_labeled"]["n_cells"] == "0"
    assert tokA["critp"]["overall"] == "PASS" and tokA["critp"]["n_cells"] == "3"
    assert tokA["er"]["P1"] == "FAIL"                 # ER swings like SEQ
    assert tokA["ewc"]["n_cells"] == "1"
    assert "seq_neu" not in tokA and "seq_amp" not in tokA   # routed to the intervention table
    assert all(t in tokA for t in ("critp1", "anchorcrit", "cecf", "anchor"))
    assert tokB["critp"]["n_cells"] == "0" and tokB["anchor"]["n_cells"] == "2"
    # rows in the tex: the LwF row is all '--'; JOINT carries the aggregate mark; post-hoc rows carry l
    byl = rows_by_label(rows)
    lwf = [r for r in rows if r[0] == "LwF"]
    assert len(lwf) == 2 and all(c == "--" for c in lwf[0][1:]) and all(c == "--" for c in lwf[1][1:])
    joint = [r for r in rows if r[0].startswith("JOINT")]
    assert joint[0][0] == "JOINT$^{a}$" and joint[0][1] == "2 (o1)" and joint[0][7] == "2/2, 1/2" and joint[0][8] == "--/--/--/--"
    assert joint[1][0] == "JOINT" and joint[1][2] == "--"
    ph = [r for r in rows if r[0].startswith("post-hoc $b^{*}$")]
    assert ph[0][5] == "--" and ph[0][6] == "--" and ph[0][8].startswith("P/--/") and ph[0][8].endswith("/--")
    assert all(c == "--" for c in ph[1][1:])
    seq = [r for r in rows if r[0] == "SEQ"]
    assert seq[0][1] == "4 (o1,o2)" and seq[0][7] == "ref" and seq[0][9] == "ref" and "$^{\\dagger}$" in seq[0][4]
    # pilot reference rows present, marked p, never bolded, verdict columns '--'
    pil = [r for r in rows if r[0].endswith("$^{p}$")]
    assert len(pil) >= 4 and all("\\textbf" not in " ".join(r) for r in pil) and all(r[9] == "--" for r in pil)
    # bold: exactly one bolded mean per bolded column per block, on the arg-best complete row
    for blk in (A, B):
        comp = [r for r in blk if r["source"] == "scorecard" and all(r[m + "_n"] not in ("", "0") for m in MA.METRICS)]
        for m, d in MA.BOLD_DIR.items():
            bolded = [r["token"] for r in blk if m in r["bold"].split(";")]
            assert len(bolded) == 1, (m, bolded)
            best = (max if d > 0 else min)(comp, key=lambda r: float(r[m + "_mean"]))
            assert bolded[0] == best["token"], (m, bolded, best["token"])
        assert not any("endpoint_c" in r["bold"] for r in blk)
    # every scorecard/aggregate/logits row's mean appears in its tex row
    label_rows = {}
    for r in rows:
        label_rows.setdefault(re.sub(r"\$\^\{a\}\$$", "", r[0]), []).append(r)
    for r in A:
        if r["n_cells"] == "0" or r["source"] == "pilot_reference":
            continue
        lab = MA.label_of(r["token"])[0]
        cand = label_rows[lab]
        want = MA.FMT["sum_abs_dc"] % float(r["sum_abs_dc_mean"])
        assert any(want in c[2] for c in cand), (lab, want, [c[2] for c in cand])
    # per-cell CSV: one line per complete cell + post-hoc cells + aggregate joint cells
    cellrows = list(csv.DictReader(open(tables / "method_main_cells.csv")))
    n_sc = sum(len(repA["arms"][p]["cells"]) for p in repA["arm_order"] if p not in ("psL_seq_neu", "psL_seq_amp"))
    repB = json.load(open(scB))
    n_sc += sum(len(repB["arms"][p]["cells"]) for p in repB["arm_order"])
    assert len(cellrows) == n_sc + 2 * 2 + 2, (len(cellrows), n_sc)
    seqcell = next(r for r in cellrows if r["token"] == "seq" and r["cell"] == "o1/s17" and r["block"] == "pilot-llava")
    assert seqcell["c_traj"].count(";") == 4 and seqcell["f1_tripped"] == "0"
    crit = next(r for r in cellrows if r["token"] == "critp" and r["cell"] == "o1/s17")
    assert crit["chair_ci_excludes_zero"] == "1" and float(crit["d_vs_ref_sum_abs_dc"]) < 0
    print("method_main OK")

    # ---- post-hoc stage CSV: additive theory holds on the synthetic shifts ----
    st = list(csv.DictReader(open(tables / "method_main_posthoc_stages.csv")))
    assert len(st) == 8 and all(r["additive_dominant"] == "True" for r in st), [r["nonadd_classmean"] for r in st]
    assert all(abs(float(r["c_labeled"]) - float(r["c_base_test"])) < 0.15 for r in st)
    print("post-hoc rows OK")

    # ---- intervention ----
    h2, rows2, spans2 = tex_rows(tables / "method_intervention.tex")
    assert h2[7:11] == ["$\\Delta c_{1}$", "$\\Delta c_{2}$", "$\\Delta c_{3}$", "$\\Delta c_{4}$"]
    orig = [r for r in rows2 if r[0] == "SEQ (original)"]
    neu = [r for r in rows2 if r[0] == "SEQ-neutral"]
    amp = [r for r in rows2 if r[0] == "SEQ-amplified"]
    assert len(orig) == 4 and len(neu) == 3 and len(amp) == 1, (len(orig), len(neu), len(amp))
    # original manifest: textvqa (stage 2, o1) is liberal-active (dose .058, yes>no) -> dc<0 expected (checkmark);
    # vizwiz (stage 4) refusal-heavy -> conservative -> dc>0 (checkmark); scienceqa/flickr zero-dose -> circ
    o17 = next(r for r in orig if r[1] == "o1/s17")
    assert o17[7].endswith("$^{\\circ}$") and o17[9].endswith("$^{\\circ}$"), o17
    assert o17[8].startswith("-") and o17[8].endswith("$\\checkmark$"), o17
    assert o17[10].startswith("+") and o17[10].endswith("$\\checkmark$"), o17
    # neutral: every stage zero-dose (refusal .019 < DOSE_ACTIVE) -> circ / bullet markers only
    for r in neu:
        assert all(c.endswith("$^{\\circ}$") or c.endswith("$^{\\bullet}$") for c in r[7:11]), r
    # amplified: vizwiz refusal .58 -> conservative push -> stage 4 dc>0 checkmark
    assert amp[0][10].startswith("+") and amp[0][10].endswith("$\\checkmark$"), amp[0]
    assert amp[0][11] == "30.0\\%"
    rng_rows = [r for r in rows2 if r[0] == "\\quad range"]
    assert len(rng_rows) == 2                         # original (4 cells) + neutral (3 cells)
    h3, rows3, _ = tex_rows(tables / "method_intervention_paired.tex")
    assert len(rows3) == 4 and all(r[4] in ("below", "inside", "above") for r in rows3)
    assert all(r[4] == "below" for r in rows3 if r[0] == "SEQ-neutral")
    assert rows3[-1][0] == "SEQ-amplified" and rows3[-1][4] == "above"
    icsv = list(csv.DictReader(open(tables / "method_intervention.csv")))
    assert len(icsv) == 8 and all(r["manifest"] for r in icsv)
    print("intervention OK")

    # ---- prior share ----
    h4, rows4, spans4 = tex_rows(tables / "method_prior_share.tex")
    assert h4[2:6] == ["$\\rho_{k1}$", "$\\rho_{k2}$", "$\\rho_{k3}$", "$\\rho_{k4}$"]
    seqr = next(r for r in rows4 if r[0] == "SEQ")
    ancr = next(r for r in rows4 if r[0] == "anchor-v2")
    assert seqr[-1] == "prior-borne" and all(0.7 < float(v) < 1.1 for v in seqr[2:6]), seqr
    assert ancr[3] == "--$^{\\ddagger}$" and ancr[4] == "--" and ancr[-1] == "evidence-conditional", ancr
    pcsv = list(csv.DictReader(open(tables / "method_prior_share.csv")))
    assert pcsv[1]["status_k2"] == "TOO_FEW_ALIGNED" and pcsv[1]["rho_k3"] == ""
    print("prior share OK")

    # ---- manifest ----
    man = json.load(open(out / "method_artifacts_manifest.json"))
    assert len(man["artifacts"]) >= 17
    for a in man["artifacts"]:
        assert os.path.isfile(a["path"]) and sha(a["path"]) == a["sha256"], a["path"]
    for i in man["inputs"]:
        assert sha(i["path"]) == i["sha256"], i
    roles = [i["role"] for i in man["inputs"]]
    assert roles.count("scorecard") == 2 and roles.count("logits") == 2 and roles.count("blind") == 2
    assert "aggregate" in roles and "manifest_neutral" in roles and "manifest_amplified" in roles and "manifest_original" in roles
    cc = man["aggregate_crosscheck"]
    assert cc[0]["n_compared"] == 7 and cc[0]["n_missing"] > 0       # seq(4)+anchor(3) compared; others missing
    assert "no llava15/pilot" not in (cc[1].get("note") or "") and cc[1]["n_compared"] == 0
    status = {a["kind"]: a["status"] for a in man["artifacts"]}
    assert status["method_main"] == "partial"        # ABSENT rows exist
    assert status["method_intervention"] == "complete"
    assert status["method_prior_share"] == "partial"  # ABSENT / too-few stages
    assert status["figure"] == "complete"
    print("manifest OK")

    # ---- fail-loud paths ----
    bad_sc = OUT / "scorecard_bad.json"
    rb = json.load(open(scA))
    del rb["arms"]["psL_anchor"]["cells"]["o1/s17"]["scorecard"]["max_abs_dd"]
    json.dump(rb, open(bad_sc, "w"))
    for argv, what in (
            (["--scorecard", str(bad_sc), "--tables_dir", str(OUT / "t_bad1"), "--out_dir", str(OUT / "o_bad1"), "--no_fig"],
             "missing scorecard key"),
            (["--scorecard", str(scA), "--aggregate", str(agg_bad), "--tables_dir", str(OUT / "t_bad2"),
              "--out_dir", str(OUT / "o_bad2"), "--no_fig"], "aggregate disagreement"),
    ):
        try:
            MA.main(argv)
        except SystemExit as e:
            assert "ERROR" in str(e), e
            print("fail-loud OK:", what, "->", str(e)[:90])
        else:
            raise AssertionError("expected SystemExit for " + what)
    tb = json.load(open(blind_jsons[0]))
    tb["summary"]["reading"] = "mixed"
    bad_bl = OUT / "blind_bad.json"
    json.dump(tb, open(bad_bl, "w"))
    try:
        MA.main(["--scorecard", str(scA), "--blind", str(bad_bl), "--tables_dir", str(OUT / "t_bad3"),
                 "--out_dir", str(OUT / "o_bad3"), "--no_fig"])
    except SystemExit as e:
        assert "reading re-derived" in str(e), e
        print("fail-loud OK: blind reading mismatch")
    else:
        raise AssertionError("expected SystemExit for tampered blind reading")
    # two scorecards with the same label must be refused
    try:
        MA.main(["--scorecard", str(scA), str(scA), "--tables_dir", str(OUT / "t_bad4"), "--out_dir", str(OUT / "o_bad4"), "--no_fig"])
    except SystemExit as e:
        assert "share the label" in str(e), e
        print("fail-loud OK: duplicate block label")
    else:
        raise AssertionError("expected SystemExit for duplicate labels")

    # ---- the method_scorecard fixture (real scorer output) ----
    fx = HERE / "out" / "scorecard_synth" / "scorecard.json"
    if fx.is_file():
        tf, of = OUT / "fixture_tables", OUT / "fixture_out"
        assert MA.main(["--scorecard", str(fx), "--tables_dir", str(tf), "--out_dir", str(of)]) == 0
        h, r5, s5 = tex_rows(tf / "method_main.tex")
        crit = next(r for r in r5 if r[0] == "critp(0.1)")
        assert crit[8] == "P/P/P/P" and crit[9] == "\\textbf{PASS}" and crit[2].startswith("\\textbf{0.064}"), crit
        assert "Qwen2.5-VL-7B" in s5[0] and "psQ\\_seq" in s5[0]
        print("fixture scorecard OK")
    else:
        print("method_scorecard fixture not present (run test_method_scorecard.py first); skipped")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
