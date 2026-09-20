#!/usr/bin/env python3
"""Paper-ready method artifacts from the study's machine outputs.

USAGE
  python analysis/make_method_artifacts.py \
      --scorecard [LABEL=]/path/scorecard.json [[LABEL=]/path/scorecard.json ...] \
      [--logits  /path/logit_bias.json ...]     (analysis/logit_bias_analysis.py output)
      [--blind   /path/blind.json ...]          (analysis/blind_prior_share.py output)
      [--aggregate /path/fs_aggregate.json]     (analysis/fs_aggregate.py output)
      [--manifest_original P] [--manifest_neutral P] [--manifest_amplified P]
      [--tables_dir paper/tables] [--out_dir analysis/out]
      [--with_pilot_reference] [--no_fig]

Every scorecard.json is one BLOCK of the main table (one suite x backbone; the
scorecard already refuses cross-backbone pairing, so blocks are never merged
and "best per column" is decided inside a block).

OUTPUTS (tables -> --tables_dir, figure + manifest -> --out_dir)
  method_main.tex / .csv / _cells.csv / _notes.tex
      rows    = arms with >= 1 complete (order, seed) cell in the block, in the
                canonical order SEQ, JOINT, ER, ER-500, EWC, LwF, post-hoc scalar
                correction (labeled b*, label-free b_mean), anchor-v1 (cecf),
                anchor-v2, anchor-OOD, critp(0.1), critp(1.0), anchorcrit, CNP,
                CNP-ref, GCA, policy, policy-abst, policy-eos, then lamF_* /
                unknown tokens; an arm named in another block but not this one
                is a row of '--' (ABSENT), nothing imputed.
      columns = cells | Sum|dc| mean [min--max] | endpoint c | max|dd'| (+F1 flag)
                | CHAIR_i@k endpoint (+ k/n paired-CI marker vs ref) | plasticity
                | '< ref' counts (Sum|dc|, CHAIR@k) | P1-P4 | overall
      sources = scorecard.json for every arm row; logit_bias JSON for the two
                post-hoc rows (marked l); fs_aggregate.json for JOINT (marked a;
                the scorecard has no staged JOINT cells); reference_numbers.json
                pilot rows (marked p, --with_pilot_reference only, never bolded).
  method_intervention.tex / .csv, method_intervention_paired.tex / .csv
      SEQ original vs SEQ on the neutral / amplified answer prior (arm tokens
      seq_neu / seq_amp, design_notes/mechanism_intervention.md): Sum|dc| and
      post-settling Sum|dc| per seed, endpoint c and d', per-stage dc with its
      sign checked against the manipulated prior (FC.dose_of on the variant's
      manifest; no manifest -> no marker, stated), leakage, plasticity; then
      the per-seed pairing (delta post-settling, fraction of churn removed,
      position against the original seed range).
  method_prior_share.tex / .csv
      rho_k = b_k^blind / b_k per arm cell and stage from the blind JSONs with
      the pre-registered reading (prior-borne rho > 0.7 / mixed > 0.3 /
      evidence-conditional), re-derived here and checked against the JSON.
  fig_method_scorecard.{pdf,png} (+ _<slug> for further blocks, + _caption.txt)
      (a) Sum|dc| bars (mean over cells) with per-cell dots; (b) endpoint
      CHAIR_i@k bars with per-cell dots and the base level; (c) paired
      CHAIR_i@k gap vs ref per matched cell with its image-bootstrap CI95
      whiskers (filled = excludes 0). Only scorecard rows are drawn.
  method_artifacts_manifest.json
      every artifact with sha256 + size + status (complete / partial / absent),
      every input with sha256, the aggregate cross-check, generator provenance.

RULES
  * Per-cell values and [min--max] ranges over complete cells; orderings are
    strata; never a pooled cross-seed CI (design_notes/review_statistics.md
    sec 2). n=1 shows the value alone.
  * Partial inputs -> partial artifacts, clearly marked; nothing imputed.
  * Schema mismatch (missing key, unparsable runtag, inconsistent internal
    numbers, aggregate disagreeing with a scorecard cell) -> SystemExit.
  * Verdict rules and tolerances are imported from analysis/method_scorecard.py,
    never re-implemented.

Python 3.9 + stdlib; matplotlib only for the figure (skipped with a note).
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fs_common as FC  # noqa: E402
import method_scorecard as MS  # noqa: E402  (rules + constants; not re-implemented)

TAG = "[make_method_artifacts]"
ABSENT = "--"
REPO = HERE.parent

# Canonical row order and paper labels (token = RUNTAG arm with the mth_ prefix
# stripped; run_arm.py: critp = CRIT_WEIGHT default 0.1, critp1 = CRIT_WEIGHT=1.0).
ARM_ROWS = [
    ("seq", "SEQ"), ("joint", "JOINT"), ("er", "ER"), ("er500", "ER-500"),
    ("ewc", "EWC"), ("lwf", "LwF"),
    ("__posthoc_labeled", "post-hoc $b^{*}$ (labeled)$^{\\ell}$"),
    ("__posthoc_labelfree", "post-hoc $\\bar{b}$ (label-free)$^{\\ell}$"),
    ("cecf", "anchor-v1 (cecf)"), ("anchor", "anchor-v2"), ("anchorOOD", "anchor-OOD"),
    ("critp", "critp(0.1)"), ("critp1", "critp(1.0)"), ("anchorcrit", "anchorcrit"),
    ("cnp", "CNP"), ("cnp_ref", "CNP-ref"), ("gca", "GCA"),
    ("policy", "policy"), ("policy_abst", "policy-abst"), ("policy_eos", "policy-eos"),
]
ARM_LABEL = dict(ARM_ROWS)
ARM_INDEX = {t: i for i, (t, _l) in enumerate(ARM_ROWS)}
POSTHOC_TOKENS = ("__posthoc_labeled", "__posthoc_labelfree")
INTERVENTION_TOKENS = {"seq_neu": "neutral", "seq_amp": "amplified"}
# mechanism_intervention.md sec 4 P3: on a zero-dose stage every |dc| must stay
# below the smallest criterion-active |dc| of the matched original seed (0.30).
ZERO_DOSE_ABS_DC = 0.30
# blind_prior_share.py summary thresholds (rho mean): prior-borne / mixed / evidence-conditional
RHO_PRIOR_BORNE = 0.7
RHO_MIXED = 0.3
METRICS = ["sum_abs_dc", "endpoint_c", "max_abs_dd", "chair60", "plasticity"]
BOLD_DIR = {"sum_abs_dc": -1, "max_abs_dd": -1, "chair60": -1, "plasticity": +1}
FMT = {"sum_abs_dc": "%.3f", "endpoint_c": "%+.3f", "max_abs_dd": "%.3f",
       "chair60": "%.4f", "plasticity": "%.3f"}
SC_REQ = ["sum_abs_dc", "sum_abs_dc_post_settling", "endpoint_c", "max_abs_dd", "f1_tripped",
          "endpoint_leak", "endpoint_chair_at_k", "mean_plasticity", "c_traj",
          "abs_dc_per_stage", "plasticity_per_stage"]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def die(msg):
    raise SystemExit("%s ERROR: %s" % (TAG, msg))


def require(obj, keys, ctx):
    if not isinstance(obj, dict):
        die("%s: expected an object, got %s" % (ctx, type(obj).__name__))
    missing = [k for k in keys if k not in obj]
    if missing:
        die("%s: schema mismatch, missing key(s) %s" % (ctx, missing))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tex_escape(s):
    return str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%") \
        .replace("&", "\\&").replace("#", "\\#")


def mstat(pairs):
    """pairs: [(cellname, value)] with None dropped -> {mean,min,max,n,values} or None."""
    vals = [(c, v) for c, v in pairs if v is not None]
    if not vals:
        return None
    xs = [v for _c, v in vals]
    return {"mean": sum(xs) / len(xs), "min": min(xs), "max": max(xs), "n": len(xs),
            "values": vals}


def fmt_stat(st, fmt, bold=False):
    if st is None:
        return ABSENT
    m = fmt % st["mean"]
    if bold:
        m = "\\textbf{%s}" % m
    if st["n"] == 1:
        return m
    return "%s [%s--%s]" % (m, fmt % st["min"], fmt % st["max"])


def kn(pair):
    return ABSENT if pair is None else "%d/%d" % tuple(pair)


def short_verdict(v):
    return {"PASS": "P", "FAIL": "F", "ABSENT": ABSENT}.get(v, ABSENT)


def parse_cellname(cn):
    m = re.match(r"^(o\d+)/s(\d+)$", cn)
    if not m:
        die("unexpected cell name %r in scorecard (want o<N>/s<seed>)" % cn)
    return m.group(1), int(m.group(2))


def token_of(prefix, info):
    """RUNTAG arm prefix + scorecard info -> row token ('seq', 'critp', 'lamF_0.02')."""
    arm = info["arm"] if info else re.sub(r"^(fs[LQ]|ps[LQ]|mth)_", "", prefix)
    if info and info.get("family") == "lamF":
        return arm                       # lamF_0.02 (kept whole)
    if arm.startswith("mth_"):
        arm = arm[4:]
    return arm


def label_of(token):
    if token in ARM_LABEL:
        return ARM_LABEL[token], plain_label(token)
    if token.startswith("lamF_"):
        return "anchor-v2 ($\\lambda_F$=%s)" % token[5:], "anchor-v2 lamF=%s" % token[5:]
    return tex_escape(token), token


def plain_label(token):
    lab = ARM_LABEL.get(token, token)
    lab = re.sub(r"\$[^$]*\$", lambda m: m.group(0).strip("$").replace("\\", "").replace("{", "").replace("}", ""), lab)
    return lab.replace("^", "")


def row_sort_key(token):
    if token in ARM_INDEX:
        return (0, ARM_INDEX[token], 0.0, token)
    k = FC.arm_sort_key(token)
    return (1,) + tuple(k[1:]) if isinstance(k, tuple) else (1, 0, 0.0, token)


def backbone_of_block(rep):
    base_name = Path(rep["header"]["base_dir"]).name
    for bb, d in FC.BASE_DIRS.items():
        if d == base_name:
            return bb
    seen = set()
    for a in rep["arms"].values():
        seen |= set(a.get("backbones_seen") or [])
        if a.get("info"):
            seen.add(a["info"]["backbone"])
    if len(seen) == 1:
        return seen.pop()
    die("cannot determine the backbone of scorecard block (base_dir=%s, seen=%s)" % (base_name, sorted(seen)))


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


# ---------------------------------------------------------------------------
# scorecard.json -> block
# ---------------------------------------------------------------------------
def load_scorecard(spec):
    label = None
    path = spec
    if "=" in spec and not os.path.exists(spec):
        label, path = spec.split("=", 1)
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        die("scorecard not found: %s" % path)
    with open(path) as f:
        try:
            rep = json.load(f)
        except ValueError as e:
            die("%s: not JSON (%s)" % (path, e))
    require(rep, ["header", "base", "arm_order", "arms"], path)
    hdr = rep["header"]
    require(hdr, ["suite", "ref", "k", "B", "seed", "base_dir", "base_present", "n_stages",
                  "f1_tol", "p2_tol"], path + ":header")
    if abs(hdr["f1_tol"] - MS.F1_DPRIME_TOL) > 1e-12 or abs(hdr["p2_tol"] - MS.P2_PLASTICITY_TOL) > 1e-12:
        die("%s: scorecard tolerances (f1 %.3f, p2 %.3f) differ from method_scorecard.py (%.3f, %.3f)"
            % (path, hdr["f1_tol"], hdr["p2_tol"], MS.F1_DPRIME_TOL, MS.P2_PLASTICITY_TOL))
    for a in rep["arm_order"]:
        if a not in rep["arms"]:
            die("%s: arm_order names %s but arms lacks it" % (path, a))
        A = rep["arms"][a]
        require(A, ["cells", "summary", "verdicts", "complete_cells", "incomplete"], "%s:arms.%s" % (path, a))
        require(A["verdicts"], ["P1", "P2", "P3", "P4", "overall"], "%s:arms.%s.verdicts" % (path, a))
        require(A["summary"], ["f1_n_tripped", "f1_n_evaluable", "n_cells"], "%s:arms.%s.summary" % (path, a))
        for cn, c in A["cells"].items():
            require(c, ["stages", "scorecard"], "%s:arms.%s.cells.%s" % (path, a, cn))
            require(c["scorecard"], SC_REQ, "%s:arms.%s.cells.%s.scorecard" % (path, a, cn))
            parse_cellname(cn)
        if A["pairing"] is not None:
            require(A["pairing"], ["n_matched", "rows", "signs"], "%s:arms.%s.pairing" % (path, a))
    if hdr["ref"] not in rep["arms"]:
        die("%s: ref arm %s not among arms" % (path, hdr["ref"]))
    bb = backbone_of_block(rep)
    if label is None:
        label = "%s / %s" % (hdr["suite"], FC.BACKBONE_LABEL.get(bb, bb))
    return {"path": path, "sha256": sha256(path), "label": label, "rep": rep, "hdr": hdr,
            "suite": hdr["suite"], "backbone": bb, "ref": hdr["ref"],
            "ref_token": token_of(hdr["ref"], rep["arms"][hdr["ref"]].get("info")),
            "rows": [], "cells_long": [], "posthoc_cells": [], "warnings": []}


def row_from_arm(block, prefix):
    rep, hdr = block["rep"], block["hdr"]
    A = rep["arms"][prefix]
    token = token_of(prefix, A.get("info"))
    lab, plain = label_of(token)
    is_ref = prefix == hdr["ref"]
    row = {"token": token, "prefix": prefix, "label": lab, "plain": plain, "source": "scorecard",
           "is_ref": is_ref, "n_cells": len(A["cells"]), "orders": [], "cells": [],
           "metrics": {}, "f1": None, "chair_ci": None, "better": {"sum_abs_dc": None, "chair60": None},
           "verdicts": None, "marks": set(), "incomplete": A["incomplete"]}
    pair_rows = {}
    if A.get("pairing"):
        for r in A["pairing"]["rows"]:
            pair_rows[(r["order"], r["seed"])] = r
    for cn, c in A["cells"].items():
        sc = c["scorecard"]
        o, s = parse_cellname(cn)
        ks = sorted(int(k) for k in c["stages"])
        last = c["stages"][str(ks[-1])] if ks else {}
        pr = pair_rows.get((o, s))
        bt = pr["chair_at_k_boot"] if pr else None
        cell = {"cell": cn, "order": o, "seed": s,
                "sum_abs_dc": sc["sum_abs_dc"], "post_settle": sc["sum_abs_dc_post_settling"],
                "endpoint_c": sc["endpoint_c"], "max_abs_dd": sc["max_abs_dd"],
                "f1_tripped": sc["f1_tripped"], "endpoint_leak": sc["endpoint_leak"],
                "chair60": sc["endpoint_chair_at_k"], "plasticity": sc["mean_plasticity"],
                "c_traj": sc["c_traj"], "abs_dc_per_stage": sc["abs_dc_per_stage"],
                "plasticity_per_stage": sc["plasticity_per_stage"],
                "endpoint_dprime": (last.get("pope") or {}).get("dprime"),
                "d_vs_ref_sum_abs_dc": pr["d_sum_abs_dc"] if pr else None,
                "d_vs_ref_chair60": pr["d_endpoint_chair_at_k"] if pr else None,
                "d_vs_ref_plasticity": pr["d_mean_plasticity"] if pr else None,
                "chair_gap_point": bt["point"] if bt else None,
                "chair_ci_lo": bt["ci95"][0] if bt else None,
                "chair_ci_hi": bt["ci95"][1] if bt else None,
                "chair_ci_excludes_zero": bt["excludes_zero"] if bt else None,
                "chair_ci_n_pairs": bt["n_pairs"] if bt else None}
        row["cells"].append(cell)
        if o not in row["orders"]:
            row["orders"].append(o)
    row["orders"].sort()
    for m in METRICS:
        row["metrics"][m] = mstat([(c["cell"], c[m]) for c in row["cells"]])
    S = A["summary"]
    if S["f1_n_evaluable"]:
        row["f1"] = (S["f1_n_tripped"], S["f1_n_evaluable"])
    if A.get("pairing") and A["pairing"]["n_matched"]:
        sg = A["pairing"]["signs"]
        row["chair_ci"] = (sg["chair_at_k_ci_excl0_lower"]["count"], sg["chair_at_k_ci_excl0_lower"]["n"])
        row["better"]["sum_abs_dc"] = (sg["sum_abs_dc_lower"]["count"], sg["sum_abs_dc_lower"]["n"])
        row["better"]["chair60"] = (sg["chair_at_k_lower"]["count"], sg["chair_at_k_lower"]["n"])
    v = A["verdicts"]
    row["verdicts"] = {p: v[p]["verdict"] for p in ("P1", "P2", "P3", "P4")}
    row["verdict_counts"] = {p: (v[p]["count"], v[p]["n"]) for p in ("P1", "P2", "P3", "P4")}
    row["verdicts"]["overall"] = "ref" if is_ref else v["overall"]
    return row


def build_block_rows(block):
    rep = block["rep"]
    for prefix in rep["arm_order"]:
        row = row_from_arm(block, prefix)
        if row["token"] in INTERVENTION_TOKENS:
            block.setdefault("intervention", []).append(row)
            continue
        if row["token"].startswith("single_"):
            continue
        block["rows"].append(row)


# ---------------------------------------------------------------------------
# logit_bias_analysis.py JSON -> post-hoc rows
# ---------------------------------------------------------------------------
def load_logits(path):
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        die("logits JSON not found: %s" % path)
    with open(path) as f:
        try:
            res = json.load(f)
        except ValueError as e:
            die("%s: not JSON (%s)" % (path, e))
    require(res, ["runtag", "stages", "base_sdt", "calib_frac", "n_splits", "flags"], path)
    if not res["stages"]:
        die("%s: no stages" % path)
    for st in res["stages"]:
        require(st, ["label", "correction", "additive_vs_base", "sdt"], path + ":stages[]")
        require(st["correction"], ["splits", "summary"], path + ":correction")
        require(st["correction"]["summary"], ["c_corrected", "c_base", "b_star", "b_mean_labelfree"],
                path + ":correction.summary")
        if not st["correction"]["splits"]:
            die("%s: stage %s has no calibration splits" % (path, st["label"]))
        for sp in st["correction"]["splits"]:
            require(sp, ["base", "uncorrected", "corrected", "corrected_labelfree", "b_star",
                         "b_mean_labelfree", "split_seed"], path + ":correction.splits[]")
            for k in ("base", "uncorrected", "corrected", "corrected_labelfree"):
                require(sp[k], ["c", "dprime", "f1"], "%s:splits[].%s" % (path, k))
        require(st["additive_vs_base"], ["shift_yes", "shift_no", "diff", "diff_ci95", "b_common",
                                         "nonadd_classmean", "nonadd_z", "dc", "dd", "dc_undone"],
                path + ":additive_vs_base")
    pr = FC.parse_runtag(res["runtag"])
    if pr is None:
        die("%s: runtag %r is not a parsable RUNTAG (need <family>_<arm>_o<N>_s<seed>)" % (path, res["runtag"]))
    return {"path": path, "sha256": sha256(path), "res": res, "info": pr}


def _mean(xs):
    return sum(xs) / len(xs)


def posthoc_cell(L):
    """One logits JSON (one cell) -> per-variant trajectory statistics."""
    res, path = L["res"], L["path"]
    stages = res["stages"]
    seeds0 = [sp["split_seed"] for sp in stages[0]["correction"]["splits"]]
    base_c = base_d = None
    traj = {"uncorrected": {"c": [], "d": []}, "labeled": {"c": [], "d": []}, "labelfree": {"c": [], "d": []}}
    add = []
    for st in stages:
        sps = st["correction"]["splits"]
        if [sp["split_seed"] for sp in sps] != seeds0:
            die("%s: calibration split seeds differ across stages (%s vs %s)" % (path, seeds0, [sp["split_seed"] for sp in sps]))
        bc, bd = _mean([sp["base"]["c"] for sp in sps]), _mean([sp["base"]["dprime"] for sp in sps])
        if base_c is None:
            base_c, base_d = bc, bd
        elif abs(bc - base_c) > 1e-9 or abs(bd - base_d) > 1e-9:
            die("%s: base c/d' on the test split differ across stages (%.6f vs %.6f): the splits are not shared"
                % (path, bc, base_c))
        for var, key in (("uncorrected", "uncorrected"), ("labeled", "corrected"), ("labelfree", "corrected_labelfree")):
            traj[var]["c"].append(_mean([sp[key]["c"] for sp in sps]))
            traj[var]["d"].append(_mean([sp[key]["dprime"] for sp in sps]))
        S = st["correction"]["summary"]
        if abs(traj["labeled"]["c"][-1] - S["c_corrected"]["mean"]) > 1e-6 or abs(base_c - S["c_base"]["mean"]) > 1e-6:
            die("%s: stage %s summary.c_corrected/c_base disagree with the per-split values (internal inconsistency)"
                % (path, st["label"]))
        a = st["additive_vs_base"]
        fl = res["flags"].get(st["label"], {})
        add.append({"stage": st["label"], "shift_yes": a["shift_yes"], "shift_no": a["shift_no"],
                    "diff": a["diff"], "diff_ci_lo": a["diff_ci95"][0], "diff_ci_hi": a["diff_ci95"][1],
                    "b_common": a["b_common"], "nonadd_classmean": a["nonadd_classmean"],
                    "nonadd_z": a["nonadd_z"], "dc": a["dc"], "dd": a["dd"], "dc_undone": a["dc_undone"],
                    "b_star_mean": S["b_star"]["mean"], "b_star_sd": S["b_star"]["sd"],
                    "b_mean_mean": S["b_mean_labelfree"]["mean"], "b_mean_sd": S["b_mean_labelfree"]["sd"],
                    "f1_base": S.get("f1_base", {}).get("mean"), "f1_uncorrected": S.get("f1_uncorrected", {}).get("mean"),
                    "f1_corrected": S.get("f1_corrected", {}).get("mean"),
                    "f1_corrected_labelfree": S.get("f1_corrected_labelfree", {}).get("mean"),
                    "c_uncorrected": traj["uncorrected"]["c"][-1], "c_labeled": traj["labeled"]["c"][-1],
                    "c_labelfree": traj["labelfree"]["c"][-1], "c_base_test": base_c,
                    "d_uncorrected": traj["uncorrected"]["d"][-1], "d_labeled": traj["labeled"]["d"][-1],
                    "d_labelfree": traj["labelfree"]["d"][-1], "d_base_test": base_d,
                    "additive_dominant": fl.get("additive_dominant"), "diff_ci_covers_0": fl.get("diff_ci_covers_0"),
                    "correction_recovers_base_f1": fl.get("correction_recovers_base_f1")})
    out = {}
    for var in traj:
        cs = [base_c] + traj[var]["c"]
        sw = [abs(cs[i + 1] - cs[i]) for i in range(len(cs) - 1)]
        dd = [abs(d - base_d) for d in traj[var]["d"]]
        out[var] = {"sum_abs_dc": sum(sw), "post_settle": sum(sw[1:]) if len(sw) > 1 else sum(sw),
                    "endpoint_c": cs[-1], "max_abs_dd": max(dd), "f1_tripped": max(dd) > MS.F1_DPRIME_TOL,
                    "c_traj": cs, "abs_dc_per_stage": sw}
    info = L["info"]
    return {"cell": "%s/s%d" % (info["order"], info["seed"]), "order": info["order"], "seed": info["seed"],
            "runtag": res["runtag"], "arm": info["arm"], "suite": info["suite"], "backbone": info["backbone"],
            "path": path, "n_splits": res["n_splits"], "calib_frac": res["calib_frac"],
            "n_stages": len(stages), "variants": out, "additive": add}


def posthoc_rows(cells, ref_token):
    """cells: posthoc_cell() outputs of one block -> the two post-hoc rows."""
    rows = []
    for var, token in (("labeled", "__posthoc_labeled"), ("labelfree", "__posthoc_labelfree")):
        lab, plain = label_of(token)
        row = {"token": token, "prefix": None, "label": lab, "plain": plain, "source": "logits",
               "is_ref": False, "n_cells": len(cells), "orders": sorted({c["order"] for c in cells}),
               "cells": [], "metrics": {}, "f1": None, "chair_ci": None,
               "better": {"sum_abs_dc": None, "chair60": None}, "verdicts": None, "marks": {"l"},
               "incomplete": [], "corrected_arm": sorted({c["arm"] for c in cells})}
        n_better = n_trip = 0
        for c in cells:
            v, u = c["variants"][var], c["variants"]["uncorrected"]
            row["cells"].append({"cell": c["cell"], "order": c["order"], "seed": c["seed"],
                                 "sum_abs_dc": v["sum_abs_dc"], "post_settle": v["post_settle"],
                                 "endpoint_c": v["endpoint_c"], "max_abs_dd": v["max_abs_dd"],
                                 "f1_tripped": v["f1_tripped"], "endpoint_leak": None, "chair60": None,
                                 "plasticity": None, "c_traj": v["c_traj"], "abs_dc_per_stage": v["abs_dc_per_stage"],
                                 "plasticity_per_stage": None, "endpoint_dprime": None,
                                 "d_vs_ref_sum_abs_dc": v["sum_abs_dc"] - u["sum_abs_dc"],
                                 "d_vs_ref_chair60": None, "d_vs_ref_plasticity": None,
                                 "chair_gap_point": None, "chair_ci_lo": None, "chair_ci_hi": None,
                                 "chair_ci_excludes_zero": None, "chair_ci_n_pairs": None,
                                 "uncorrected_sum_abs_dc": u["sum_abs_dc"], "runtag": c["runtag"]})
            n_better += v["sum_abs_dc"] < u["sum_abs_dc"]
            n_trip += v["f1_tripped"]
        for m in METRICS:
            row["metrics"][m] = mstat([(c["cell"], c[m]) for c in row["cells"]])
        n = len(cells)
        row["f1"] = (n_trip, n)
        row["better"]["sum_abs_dc"] = (n_better, n)
        row["verdicts"] = {"P1": MS._verdict(n_better, n, 2 / 3), "P2": "ABSENT",
                           "P3": ("ABSENT" if n == 0 else ("FAIL" if n_trip else "PASS")), "P4": "ABSENT",
                           "overall": "INCOMPLETE"}
        row["verdict_counts"] = {"P1": (n_better, n), "P2": (0, 0), "P3": (n - n_trip, n), "P4": (0, 0)}
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# fs_aggregate.json: JOINT row + cross-check
# ---------------------------------------------------------------------------
def load_aggregate(path):
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        die("aggregate not found: %s" % path)
    with open(path) as f:
        try:
            agg = json.load(f)
        except ValueError as e:
            die("%s: not JSON (%s)" % (path, e))
    require(agg, ["backbones", "f1_tol"], path)
    for bb, bd in agg["backbones"].items():
        if "suites" not in bd:
            die("%s: backbone %s has no 'suites' block -- this is the pre-suite fs_aggregate schema; "
                "regenerate with the current analysis/fs_aggregate.py" % (path, bb))
        for suite, sd in bd["suites"].items():
            require(sd, ["scorecard", "arms"], "%s:backbones.%s.suites.%s" % (path, bb, suite))
            for e in sd["scorecard"]:
                require(e, ["arm", "cells", "n_complete"], "%s:scorecard[]" % path)
                for c in e["cells"]:
                    require(c, ["order", "seed", "complete"], "%s:scorecard[].cells[]" % path)
    return {"path": path, "sha256": sha256(path), "agg": agg}


def agg_entries(AG, bb, suite):
    sd = ((AG["agg"]["backbones"].get(bb) or {}).get("suites") or {}).get(suite)
    if not sd:
        return None
    return {e["arm"]: e for e in sd["scorecard"]}


def joint_row_from_aggregate(AG, block):
    ents = agg_entries(AG, block["backbone"], block["suite"])
    if not ents or "joint" not in ents:
        return None
    e = ents["joint"]
    comp = [c for c in e["cells"] if c["complete"]]
    lab, plain = label_of("joint")
    row = {"token": "joint", "prefix": "aggregate:joint", "label": lab, "plain": plain, "source": "aggregate",
           "is_ref": False, "n_cells": len(comp), "orders": sorted({c["order"] for c in comp}), "cells": [],
           "metrics": {}, "f1": None, "chair_ci": None, "better": {"sum_abs_dc": None, "chair60": None},
           "verdicts": None, "marks": {"a"}, "incomplete": [[[c["order"], c["seed"]], "partial in aggregate"]
                                                           for c in e["cells"] if not c["complete"]]}
    tol = AG["agg"]["f1_tol"]
    n_trip = 0
    for c in comp:
        mdd = c.get("max_abs_dd")
        trip = (mdd is not None and mdd > tol)
        n_trip += trip
        row["cells"].append({"cell": "%s/s%d" % (c["order"], c["seed"]), "order": c["order"], "seed": c["seed"],
                             "sum_abs_dc": c.get("sum_abs_dc"), "post_settle": c.get("post_settle"),
                             "endpoint_c": c.get("endpoint_c"), "max_abs_dd": mdd,
                             "f1_tripped": (trip if mdd is not None else None), "endpoint_leak": None,
                             "chair60": c.get("endpoint_chair_i60"), "plasticity": None,
                             "c_traj": None, "abs_dc_per_stage": c.get("dc_steps"), "plasticity_per_stage": None,
                             "endpoint_dprime": None, "d_vs_ref_sum_abs_dc": None, "d_vs_ref_chair60": None,
                             "d_vs_ref_plasticity": None, "chair_gap_point": None, "chair_ci_lo": None,
                             "chair_ci_hi": None, "chair_ci_excludes_zero": None, "chair_ci_n_pairs": None})
    for m in METRICS:
        row["metrics"][m] = mstat([(c["cell"], c[m]) for c in row["cells"]])
    evaluable = sum(1 for c in comp if c.get("max_abs_dd") is not None)
    row["f1"] = (n_trip, evaluable) if evaluable else None
    vs = e.get("vs_seq")
    if vs and vs.get("n_matched"):
        row["better"]["sum_abs_dc"] = tuple(vs["sum_abs_dc_lower"])
        row["better"]["chair60"] = tuple(vs["chair_i60_lower"])
    row["verdicts"] = {"P1": "ABSENT", "P2": "ABSENT", "P3": "ABSENT", "P4": "ABSENT", "overall": ABSENT}
    row["verdict_counts"] = {p: (0, 0) for p in ("P1", "P2", "P3", "P4")}
    return row


def crosscheck_aggregate(AG, block, tol):
    """Every complete scorecard cell must agree with the aggregate's cell (same
    scorers, same rounding). Disagreement is fatal; a cell missing from the
    aggregate is recorded, not fatal."""
    ents = agg_entries(AG, block["backbone"], block["suite"])
    out = {"block": block["label"], "n_compared": 0, "n_missing": 0, "missing": [], "compared": []}
    if ents is None:
        out["note"] = "aggregate has no %s/%s suite block" % (block["backbone"], block["suite"])
        return out
    keymap = {"sum_abs_dc": "sum_abs_dc", "endpoint_c": "endpoint_c"}
    if block["hdr"]["k"] == FC.CHAIR_K_WORDS:
        keymap["chair60"] = "endpoint_chair_i60"
    for row in block["rows"] + block.get("intervention", []):
        if row["source"] != "scorecard":
            continue
        info = block["rep"]["arms"][row["prefix"]].get("info")
        arm = info["arm"] if info else row["token"]
        e = ents.get(arm)
        for c in row["cells"]:
            ac = None
            if e:
                ac = next((x for x in e["cells"] if x["order"] == c["order"] and x["seed"] == c["seed"]
                           and x["complete"]), None)
            if ac is None:
                out["n_missing"] += 1
                out["missing"].append("%s %s" % (arm, c["cell"]))
                continue
            for k_sc, k_ag in keymap.items():
                a, b = c[k_sc], ac.get(k_ag)
                if a is None or b is None:
                    continue
                if abs(a - b) > tol:
                    die("aggregate disagrees with scorecard: %s %s %s scorecard=%.4f aggregate=%.4f (tol %g) "
                        "-- two pipelines, two numbers; refusing to emit" % (block["label"], arm, c["cell"], a, b, tol))
            out["n_compared"] += 1
            out["compared"].append("%s %s" % (arm, c["cell"]))
    return out


# ---------------------------------------------------------------------------
# pilot reference rows (context only)
# ---------------------------------------------------------------------------
def pilot_reference_rows(rep):
    pr = rep.get("pilot_reference")
    if not pr or not pr.get("arms"):
        return []
    rows = []
    for name, a in pr["arms"].items():
        sc = a["scorecard"]
        lab = "%s$^{p}$" % tex_escape(name)
        row = {"token": "pilot:" + name, "prefix": None, "label": lab, "plain": name + " (pilot)",
               "source": "pilot_reference", "is_ref": False, "n_cells": 1, "orders": ["o1"], "cells": [],
               "metrics": {}, "f1": (int(bool(sc.get("f1_tripped"))), 1), "chair_ci": None,
               "better": {"sum_abs_dc": None, "chair60": None}, "verdicts": None, "marks": {"p"}, "incomplete": []}
        cell = {"cell": "o1/pilot", "order": "o1", "seed": None, "sum_abs_dc": sc.get("sum_abs_dc"),
                "post_settle": sc.get("sum_abs_dc_post_settling"), "endpoint_c": sc.get("endpoint_c"),
                "max_abs_dd": sc.get("max_abs_dd"), "f1_tripped": sc.get("f1_tripped"),
                "endpoint_leak": sc.get("endpoint_leak"), "chair60": sc.get("endpoint_chair_at_60"),
                "plasticity": sc.get("mean_plasticity"), "c_traj": None, "abs_dc_per_stage": sc.get("abs_dc_per_stage"),
                "plasticity_per_stage": None, "endpoint_dprime": None, "d_vs_ref_sum_abs_dc": None,
                "d_vs_ref_chair60": None, "d_vs_ref_plasticity": None, "chair_gap_point": None,
                "chair_ci_lo": None, "chair_ci_hi": None, "chair_ci_excludes_zero": None, "chair_ci_n_pairs": None}
        row["cells"].append(cell)
        for m in METRICS:
            row["metrics"][m] = mstat([(cell["cell"], cell[m])])
        row["verdicts"] = {"P1": "ABSENT", "P2": "ABSENT", "P3": "ABSENT", "P4": "ABSENT", "overall": ABSENT}
        row["verdict_counts"] = {p: (0, 0) for p in ("P1", "P2", "P3", "P4")}
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# generic table emission
# ---------------------------------------------------------------------------
def write_tex_table(path, colspec, header, body, comments, notes):
    """body: list of items; each item is a list of cell strings (a row), the
    string 'midrule', or ('span', text) for a full-width italic label row."""
    ncol = len(header)
    L = ["% Auto-generated by analysis/make_method_artifacts.py -- do not edit by hand."]
    L += ["% " + c for c in comments]
    L += ["\\begin{tabular}{%s}" % colspec, "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for item in body:
        if item == "midrule":
            L.append("\\midrule")
        elif isinstance(item, tuple) and item[0] == "span":
            L.append("\\multicolumn{%d}{l}{\\textit{%s}} \\\\" % (ncol, item[1]))
        else:
            if len(item) != ncol:
                die("internal: row has %d cells, header %d (%s)" % (len(item), ncol, item[:2]))
            L.append(" & ".join(item) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "%% --- audit notes ---"]
    L += ["% " + n for n in notes]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    npath = os.path.splitext(path)[0] + "_notes.tex"
    with open(npath, "w") as f:
        f.write("%% Auto-generated audit notes for %s.\n{\\footnotesize\n" % os.path.basename(path))
        for n in notes:
            f.write(n + "\\\\\n")
        f.write("}\n")
    print("%s wrote %s (+ %s)" % (TAG, path, os.path.basename(npath)))
    return [path, npath]


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    print("%s wrote %s (%d rows)" % (TAG, path, len(rows)))
    return path


def num(v, nd=6):
    return "" if v is None else repr(round(float(v), nd))


# ---------------------------------------------------------------------------
# method_main
# ---------------------------------------------------------------------------
def complete_row(row):
    return row["source"] == "scorecard" and row["n_cells"] > 0 and all(row["metrics"][m] for m in METRICS)


def absent_note(row):
    """CSV note for an arm present in the block but with zero complete cells.

    Starts with 'ABSENT' like the missing-arm note, then names the incomplete
    cells and why, so the audit trail says which run is still pending rather
    than only that the row is blank. ';' is reserved as the note separator, so
    reasons carrying one are re-punctuated instead of splitting the field.
    """
    why = []
    for item in row.get("incomplete") or []:
        try:
            (o, s), reason = item
            why.append("%s/s%s: %s" % (o, s, reason))
        except (TypeError, ValueError):
            why.append(str(item))
    tail = " -- " + " | ".join(w.replace(";", ",") for w in why) if why else ""
    return "ABSENT: 0 complete cells in this scorecard (%d incomplete)%s" % (len(why), tail)


def best_in_block(rows):
    best = {}
    comp = [r for r in rows if complete_row(r)]
    for m, d in BOLD_DIR.items():
        cand = [(r["token"], r["metrics"][m]["mean"]) for r in comp]
        if len(cand) >= 2:
            best[m] = (max if d > 0 else min)(cand, key=lambda t: t[1])[0]
    return best


def main_table(blocks, tables_dir, k_words, opts):
    all_tokens = set()
    for b in blocks:
        all_tokens |= {r["token"] for r in b["rows"]}
    ordered = sorted(all_tokens, key=row_sort_key)
    ref_is_seq = all(b["ref_token"] == "seq" for b in blocks)
    ref_hdr = "SEQ" if ref_is_seq else "ref"
    header = ["Method", "cells", "$\\Sigma|\\Delta c|$ $\\downarrow$", "endpoint $c$",
              "max$|\\Delta d'|$ $\\downarrow$", "CHAIR$_i$@%d $\\downarrow$" % k_words,
              "plasticity $\\uparrow$", "$<$ %s ($\\Sigma|\\Delta c|$, CHAIR)" % ref_hdr, "P1--P4", "overall"]
    body, csv_rows, cell_rows = [], [], []
    marks_used, status = set(), "complete"
    for bi, b in enumerate(blocks):
        if bi:
            body.append("midrule")
        base_name = Path(b["hdr"]["base_dir"]).name
        span = "%s: ref %s; base %s%s; %d stages; $B$=%d, seed %d" % (
            tex_escape(b["label"]), tex_escape(b["ref"]), tex_escape(base_name),
            "" if b["hdr"]["base_present"] else " (ABSENT)", b["hdr"]["n_stages"], b["hdr"]["B"], b["hdr"]["seed"])
        body.append(("span", span))
        by_tok = {r["token"]: r for r in b["rows"]}
        best = best_in_block(b["rows"])
        for tok in ordered:
            r = by_tok.get(tok)
            if tok in POSTHOC_TOKENS:
                # the $\ell$ superscript is baked into the canonical label, so an
                # ABSENT post-hoc row still shows it and still needs its footnote
                marks_used.add("l")
            if r is None:
                lab, _plain = label_of(tok)
                body.append([lab] + [ABSENT] * 9)
                csv_rows.append({"block": b["label"], "suite": b["suite"], "backbone": b["backbone"],
                                 "method": _plain, "token": tok, "source": "", "n_cells": 0, "orders": "",
                                 "notes": "ABSENT: arm not in this scorecard"})
                status = "partial"
                continue
            if r["n_cells"] == 0:
                # Present in this scorecard but with zero complete (ordering, seed)
                # cells: an ABSENT row exactly like an arm missing from the block
                # -- every column '--', nothing imputed, and no mark registered
                # (no mark is rendered, so no footnote may claim one).
                status = "partial"
                body.append([r["label"]] + [ABSENT] * 9)
                rec = {"block": b["label"], "suite": b["suite"], "backbone": b["backbone"], "method": r["plain"],
                       "token": tok, "source": r["source"], "n_cells": 0, "orders": "",
                       "is_ref": int(r["is_ref"]), "bold": "", "notes": absent_note(r)}
                if r["verdicts"]:
                    rec.update({p: r["verdicts"][p] for p in ("P1", "P2", "P3", "P4", "overall")})
                for m in METRICS:
                    rec.update({m + "_mean": "", m + "_min": "", m + "_max": "", m + "_n": 0})
                csv_rows.append(rec)
                continue
            marks_used |= r["marks"]
            cells_txt = str(r["n_cells"]) + (" (%s)" % ",".join(r["orders"]) if r["orders"] else "")
            label = r["label"] + ("$^{a}$" if "a" in r["marks"] else "")
            cells = [label, cells_txt]
            for m in METRICS:
                bold = best.get(m) == tok and complete_row(r)
                s = fmt_stat(r["metrics"][m], FMT[m], bold)
                if m == "max_abs_dd" and r["f1"] and r["f1"][0] > 0 and s != ABSENT:
                    s += "$^{\\dagger}$"
                    marks_used.add("dagger")
                if m == "chair60" and r["chair_ci"] and s != ABSENT:
                    s += "$^{%d/%d}$" % r["chair_ci"]
                    marks_used.add("ci")
                cells.append(s)
            better = "%s, %s" % (kn(r["better"]["sum_abs_dc"]), kn(r["better"]["chair60"]))
            if r["is_ref"]:
                better = "ref"
            cells.append(better)
            v = r["verdicts"]
            cells.append("/".join(short_verdict(v[p]) for p in ("P1", "P2", "P3", "P4")))
            cells.append(tex_escape(v["overall"]) if v["overall"] not in ("PASS", "FAIL") else "\\textbf{%s}" % v["overall"])
            body.append(cells)
            rec = {"block": b["label"], "suite": b["suite"], "backbone": b["backbone"], "method": r["plain"],
                   "token": tok, "source": r["source"], "n_cells": r["n_cells"], "orders": ";".join(r["orders"]),
                   "is_ref": int(r["is_ref"]), "f1_tripped_k": r["f1"][0] if r["f1"] else "",
                   "f1_n": r["f1"][1] if r["f1"] else "",
                   "chair_ci_excl0_k": r["chair_ci"][0] if r["chair_ci"] else "",
                   "chair_ci_n": r["chair_ci"][1] if r["chair_ci"] else "",
                   "better_sum_abs_dc_k": r["better"]["sum_abs_dc"][0] if r["better"]["sum_abs_dc"] else "",
                   "better_sum_abs_dc_n": r["better"]["sum_abs_dc"][1] if r["better"]["sum_abs_dc"] else "",
                   "better_chair_k": r["better"]["chair60"][0] if r["better"]["chair60"] else "",
                   "better_chair_n": r["better"]["chair60"][1] if r["better"]["chair60"] else "",
                   "P1": v["P1"], "P2": v["P2"], "P3": v["P3"], "P4": v["P4"], "overall": v["overall"],
                   "bold": ";".join(m for m in METRICS if best.get(m) == tok and complete_row(r)),
                   "notes": ";".join(sorted(r["marks"])) + (";corrected_arm=" + "+".join(r["corrected_arm"]) if r.get("corrected_arm") else "")}
            for m in METRICS:
                st = r["metrics"][m]
                rec.update({m + "_mean": num(st["mean"]) if st else "", m + "_min": num(st["min"]) if st else "",
                            m + "_max": num(st["max"]) if st else "", m + "_n": st["n"] if st else 0})
            csv_rows.append(rec)
            for c in r["cells"]:
                cr = {"block": b["label"], "suite": b["suite"], "backbone": b["backbone"], "method": r["plain"],
                      "token": tok, "source": r["source"], "cell": c["cell"], "order": c["order"], "seed": c["seed"]}
                for k in ("sum_abs_dc", "post_settle", "endpoint_c", "endpoint_dprime", "max_abs_dd",
                          "endpoint_leak", "chair60", "plasticity", "d_vs_ref_sum_abs_dc", "d_vs_ref_chair60",
                          "d_vs_ref_plasticity", "chair_gap_point", "chair_ci_lo", "chair_ci_hi"):
                    cr[k] = num(c.get(k))
                cr["f1_tripped"] = "" if c.get("f1_tripped") is None else int(c["f1_tripped"])
                cr["chair_ci_excludes_zero"] = "" if c.get("chair_ci_excludes_zero") is None else int(c["chair_ci_excludes_zero"])
                cr["chair_ci_n_pairs"] = c.get("chair_ci_n_pairs") or ""
                cr["c_traj"] = ";".join("%.4f" % x for x in c["c_traj"]) if c.get("c_traj") else ""
                cr["abs_dc_per_stage"] = ";".join("%.4f" % x for x in c["abs_dc_per_stage"]) if c.get("abs_dc_per_stage") else ""
                cr["plasticity_per_stage"] = ";".join("" if x is None else "%.4f" % x for x in c["plasticity_per_stage"]) if c.get("plasticity_per_stage") else ""
                cr["uncorrected_sum_abs_dc"] = num(c.get("uncorrected_sum_abs_dc"))
                cell_rows.append(cr)
    # Pilot-run reference rows are context only: they are APPENDED after every
    # scorecard block (never wedged between two blocks, which would read as part
    # of the following block and split the comparable blocks apart).
    if opts.get("pilot_rows"):
        pb = next((b for b in blocks if b.get("rep", {}).get("pilot_reference")), None)
        prs = pilot_reference_rows(pb["rep"]) if pb else []
        if prs:
            body.append("midrule")
            body.append(("span", tex_escape(pb["rep"]["pilot_reference"].get("label", "pilot reference")[:110]) + "\\ldots"))
            for r in prs:
                marks_used |= r["marks"]
                cells = [r["label"], "1 (o1)"]
                for m in METRICS:
                    s = fmt_stat(r["metrics"][m], FMT[m])
                    if m == "max_abs_dd" and r["f1"] and r["f1"][0] and s != ABSENT:
                        s += "$^{\\dagger}$"
                        marks_used.add("dagger")
                    cells.append(s)
                cells += [ABSENT, ABSENT, ABSENT]
                body.append(cells)
                rec = {"block": pb["label"] + " (pilot reference)", "suite": "pilot", "backbone": "llava15",
                       "method": r["plain"], "token": r["token"], "source": r["source"], "n_cells": 1,
                       "orders": "o1", "notes": "pilot reference; context only"}
                for m in METRICS:
                    st = r["metrics"][m]
                    rec.update({m + "_mean": num(st["mean"]) if st else "", m + "_n": st["n"] if st else 0})
                csv_rows.append(rec)

    comments = ["Blocks = one scorecard.json each (suite x backbone); '--' = ABSENT (nothing imputed);",
                "bold = best per column among complete rows of the block; mean [min--max] over cells."]
    notes = [
        "Rows are arms with at least one complete (ordering, seed) cell in the block's scorecard; a row of "
        "'--' is an arm present in another block (or with zero complete cells here). Nothing is imputed.",
        "Each entry is the mean over complete cells with the [min--max] across cells (a single cell shows "
        "the value alone); cells are (ordering, seed) runs, orderings are strata, and no cross-seed "
        "confidence interval is computed (design\\_notes/review\\_statistics.md sec.~2). Per-cell values: "
        "method\\_main\\_cells.csv.",
        "Reference arm and base cell are named in each block header. $\\Sigma|\\Delta c|=\\sum_k |c_k-c_{k-1}|$ "
        "with $c_0$ = base (pooled POPE, clipped $z$); max$|\\Delta d'| = \\max_k |d'_k - d'_{\\mathrm{base}}|$",
        "CHAIR$_i$@%d = per-mention CHAIR$_i$ over the first %d words of the endpoint captions; the superscript "
        "$k/n$ counts matched cells whose paired image-bootstrap CI95 of arm $-$ ref excludes 0 on the reducing "
        "side ($B$ and seed in the block header)." % (k_words, k_words),
        "Plasticity = mean over stages of the just-trained task's score (pilot scorers mc\\_acc / vqa\\_acc / "
        "caption\\_uf1; the caption task is a trajectory signal, not an accuracy).",
        "'$<$ %s' = matched cells in which the arm's $\\Sigma|\\Delta c|$ (first) and endpoint CHAIR$_i$@%d "
        "(second) are below the reference." % (ref_hdr, k_words),
        "P1--P4 are the pre-registered predictions with the rules of analysis/method\\_scorecard.py "
        "(P1 $\\Sigma|\\Delta c|$ lower in $\\geq 2/3$ matched cells; P2 plasticity $\\geq$ ref $- %.2f$ in every "
        "cell; P3 max$|\\Delta d'| < %.2f$ in every cell; P4 CHAIR$_i$@%d CI excludes 0 below ref in $\\geq$ half); "
        "P = pass, F = fail, -- = inputs absent; overall PASS needs all four." % (MS.P2_PLASTICITY_TOL, MS.F1_DPRIME_TOL, k_words),
    ]
    if "dagger" in marks_used:
        notes.append("$\\dagger$: falsifier F1 tripped ($|\\Delta d'| > %.2f$) in at least one cell (count in the CSV)."
                     % MS.F1_DPRIME_TOL)
    if "l" in marks_used:
        notes.append("$\\ell$ rows: zero-training inference-time correction of the POPE answer-position logits "
                     "(analysis/logit\\_bias\\_analysis.py). Per stage a scalar is added to $g = z_{yes} - z_{no}$: "
                     "$b^{*}$ (labeled) matches the base criterion on a stratified calibration split, $\\bar b$ "
                     "(label-free) matches the mean gap; decisions at 0 on the held-out split; $c$, $d'$ are means "
                     "over calibration-split seeds. CHAIR and plasticity do not apply (captions and task outputs are "
                     "unchanged). '$<$ %s' and P1/P3 compare against the UNCORRECTED logit trajectory of the same "
                     "cell (same instrument and split), so they are not directly comparable with the text-parsed rows; "
                     "P2/P4 are not applicable." % ref_hdr)
    if "a" in marks_used:
        notes.append("$a$ row: JOINT from fs\\_aggregate.json (stage-boundary checkpoints of the joint run scored by "
                     "the same scorers); the scorecard has no staged JOINT cells, so no paired CI and no verdicts.")
    if "p" in marks_used:
        notes.append("$p$ rows: pilot-run reference (LLaVA-1.5-7B, ordering o1, one seed; analysis/reference\\_numbers.json) "
                     "-- context only, never an input to a verdict, never bolded.")
    stem = os.path.join(tables_dir, "method_main")
    files = write_tex_table(stem + ".tex", "ll" + "r" * 5 + "ccc", header, body, comments, notes)
    fields = ["block", "suite", "backbone", "method", "token", "source", "n_cells", "orders", "is_ref"]
    for m in METRICS:
        fields += [m + "_mean", m + "_min", m + "_max", m + "_n"]
    fields += ["f1_tripped_k", "f1_n", "chair_ci_excl0_k", "chair_ci_n", "better_sum_abs_dc_k", "better_sum_abs_dc_n",
               "better_chair_k", "better_chair_n", "P1", "P2", "P3", "P4", "overall", "bold", "notes"]
    files.append(write_csv(stem + ".csv", fields, csv_rows))
    cfields = ["block", "suite", "backbone", "method", "token", "source", "cell", "order", "seed", "sum_abs_dc",
               "post_settle", "endpoint_c", "endpoint_dprime", "max_abs_dd", "f1_tripped", "endpoint_leak", "chair60",
               "plasticity", "d_vs_ref_sum_abs_dc", "d_vs_ref_chair60", "d_vs_ref_plasticity", "chair_gap_point",
               "chair_ci_lo", "chair_ci_hi", "chair_ci_excludes_zero", "chair_ci_n_pairs", "c_traj",
               "abs_dc_per_stage", "plasticity_per_stage", "uncorrected_sum_abs_dc"]
    files.append(write_csv(stem + "_cells.csv", cfields, cell_rows))
    return files, status


def posthoc_stage_csv(blocks, tables_dir):
    rows = []
    for b in blocks:
        for c in b.get("posthoc_cells", []):
            for a in c["additive"]:
                r = {"block": b["label"], "runtag": c["runtag"], "cell": c["cell"], "arm": c["arm"],
                     "n_splits": c["n_splits"], "calib_frac": c["calib_frac"]}
                for k, v in a.items():
                    r[k] = (num(v) if isinstance(v, float) else ("" if v is None else v))
                rows.append(r)
    if not rows:
        return None
    fields = ["block", "runtag", "cell", "arm", "n_splits", "calib_frac"] + list(rows[0].keys())[6:]
    return write_csv(os.path.join(tables_dir, "method_main_posthoc_stages.csv"), fields, rows)


# ---------------------------------------------------------------------------
# method_intervention
# ---------------------------------------------------------------------------
def load_manifest_stats(path, suite):
    if not path:
        return None
    if not os.path.isfile(path):
        die("manifest not found: %s" % path)
    info = FC.load_suite(suite, path)
    return {"path": path, "sha256": sha256(path), "info": info,
            "dose": {t: FC.dose_of((info["tasks"].get(t) or {}).get("answer_stats")) for t in info["tasks"]}}


def stage_marker(dc, dose):
    """dc: signed c_k - c_{k-1}; dose: FC.dose_of(...) or None."""
    if dose is None:
        return "", ""
    if not dose["active"]:
        ok = abs(dc) < ZERO_DOSE_ABS_DC
        return ("$^{\\circ}$" if ok else "$^{\\bullet}$"), ("zero-dose ok" if ok else "zero-dose |dc|>=%.2f" % ZERO_DOSE_ABS_DC)
    pred = -dose["direction"]          # liberal push (+1) lowers c
    ok = (dc > 0) == (pred > 0) and dc != 0
    return ("$\\checkmark$" if ok else "$\\times$"), ("sign follows prior" if ok else "sign opposes prior")


def intervention_tables(blocks, tables_dir, manifests):
    variants = []   # (block, variant_name, row)
    originals = {}
    for b in blocks:
        for r in b.get("intervention", []):
            variants.append((b, INTERVENTION_TOKENS[r["token"]], r))
        ref = next((r for r in b["rows"] if r["is_ref"]), None)
        if ref:
            originals[b["label"]] = ref
    n_st = max([b["hdr"]["n_stages"] for b in blocks] or [4])
    header = ["arm", "cell", "$\\Sigma|\\Delta c|$", "post-settle", "endpoint $c$", "endpoint $d'$",
              "max$|\\Delta d'|$"] + ["$\\Delta c_{%d}$" % k for k in range(1, n_st + 1)] + ["leak@end", "plast."]
    body, csv_rows, pbody, pcsv = [], [], [], []
    stem = os.path.join(tables_dir, "method_intervention")
    pstem = stem + "_paired"
    notes = ["SEQ on the original answer prior vs SEQ on the neutral / amplified prior (arm tokens seq\\_neu / "
             "seq\\_amp; design\\_notes/mechanism\\_intervention.md). Per-seed values, then [min--max]; no pooled "
             "cross-seed CI; the amplified arm is a position statement against the original seed range.",
             "$\\Delta c_k = c_k - c_{k-1}$ (pooled POPE criterion, $c_0$ = base); post-settle = $\\Sigma|\\Delta c|$ "
             "over stages $\\geq 2$ (the E2 statistic); leak@end = exact-'unanswerable' rate on the leakage task "
             "at the endpoint; plast. = mean just-trained-task score.",
             "Stage markers from the variant's manifest answer statistics (fs\\_common.dose\\_of): $\\checkmark$/"
             "$\\times$ = sign of $\\Delta c_k$ follows / opposes the manipulated prior on a criterion-active stage "
             "(a liberal push lowers $c$); $\\circ$/$\\bullet$ = zero-dose stage with $|\\Delta c_k| <$ / $\\geq$ %.2f. "
             "No marker = no manifest given for that variant (stated in the manifest file)." % ZERO_DOSE_ABS_DC]
    if not variants:
        body.append(("span", "ABSENT: no seq\\_neu / seq\\_amp arm with a complete cell in any scorecard"))
        files = write_tex_table(stem + ".tex", "ll" + "r" * (5 + n_st + 2), header, body,
                                ["ABSENT -- no intervention arm present; placeholder so \\input does not break."], notes)
        files += write_tex_table(pstem + ".tex", "llrrrl", ["variant", "seed", "$\\Delta$ post-settle",
                                                             "churn removed", "position", "original range"],
                                 [("span", "ABSENT: no intervention arm present")], ["ABSENT placeholder."], notes[:1])
        files.append(write_csv(stem + ".csv", ["status"], [{"status": "ABSENT"}]))
        files.append(write_csv(pstem + ".csv", ["status"], [{"status": "ABSENT"}]))
        return files, "absent"
    status = "complete"
    seen_blocks = []
    for b, vname, vrow in variants:
        if b["label"] not in seen_blocks:
            seen_blocks.append(b["label"])
    first = True
    for bl in seen_blocks:
        b = next(x for x in blocks if x["label"] == bl)
        if not first:
            body.append("midrule")
            pbody.append("midrule")
        first = False
        body.append(("span", tex_escape(b["label"])))
        pbody.append(("span", tex_escape(b["label"])))
        orig = originals.get(bl)
        groups = [("original", orig, manifests.get("original"))] if orig else []
        for bb_, vname, vrow in variants:
            if bb_["label"] == bl:
                groups.append((vname, vrow, manifests.get(vname)))
        for vname, row, man in groups:
            if row is None:
                continue
            if man is None:
                status = "partial"
            for c in sorted(row["cells"], key=lambda x: (x["order"], x["seed"])):
                tl = FC.order_tasks(man["info"], c["order"]) if man else None
                cells = ["SEQ-%s" % vname if vname != "original" else "SEQ (original)", tex_escape(c["cell"]),
                         fmt_stat(mstat([(c["cell"], c["sum_abs_dc"])]), "%.3f"),
                         fmt_stat(mstat([(c["cell"], c["post_settle"])]), "%.3f"),
                         fmt_stat(mstat([(c["cell"], c["endpoint_c"])]), "%+.3f"),
                         fmt_stat(mstat([(c["cell"], c["endpoint_dprime"])]), "%.3f"),
                         fmt_stat(mstat([(c["cell"], c["max_abs_dd"])]), "%.3f")
                         + ("$^{\\dagger}$" if c["f1_tripped"] else "")]
                rec = {"block": bl, "variant": vname, "cell": c["cell"], "order": c["order"], "seed": c["seed"],
                       "sum_abs_dc": num(c["sum_abs_dc"]), "post_settle": num(c["post_settle"]),
                       "endpoint_c": num(c["endpoint_c"]), "endpoint_dprime": num(c["endpoint_dprime"]),
                       "max_abs_dd": num(c["max_abs_dd"]), "f1_tripped": int(bool(c["f1_tripped"])),
                       "endpoint_leak": num(c["endpoint_leak"]), "plasticity": num(c["plasticity"]),
                       "manifest": man["path"] if man else ""}
                ct = c["c_traj"] or []
                for k in range(1, n_st + 1):
                    if len(ct) > k:
                        dc = ct[k] - ct[k - 1]
                        dose = None
                        task = tl[k - 1] if tl and k <= len(tl) else None
                        if man and task:
                            dose = man["dose"].get(task)
                        mk, why = stage_marker(dc, dose)
                        cells.append("%+.3f%s" % (dc, mk))
                        rec["dc_%d" % k] = num(dc)
                        rec["dose_%d" % k] = "" if dose is None else "%s%.3f" % ("+" if dose["direction"] > 0 else "-" if dose["direction"] < 0 else "0", dose["magnitude"])
                        rec["marker_%d" % k] = why
                    else:
                        cells.append(ABSENT)
                cells.append(ABSENT if c["endpoint_leak"] is None else "%.1f\\%%" % (100 * c["endpoint_leak"]))
                cells.append(fmt_stat(mstat([(c["cell"], c["plasticity"])]), "%.3f"))
                body.append(cells)
                csv_rows.append(rec)
            if len(row["cells"]) > 1:
                rng = ["\\quad range", "n=%d" % len(row["cells"])]
                for k in ("sum_abs_dc", "post_settle", "endpoint_c", "endpoint_dprime", "max_abs_dd"):
                    st = mstat([(c["cell"], c[k]) for c in row["cells"]])
                    rng.append(ABSENT if st is None else "[%s--%s]" % ((FMT.get(k, "%.3f") % st["min"]), (FMT.get(k, "%.3f") % st["max"])))
                for k in range(1, n_st + 1):
                    dcs = [c["c_traj"][k] - c["c_traj"][k - 1] for c in row["cells"] if c["c_traj"] and len(c["c_traj"]) > k]
                    rng.append("[%+.2f--%+.2f]" % (min(dcs), max(dcs)) if dcs else ABSENT)
                lk = mstat([(c["cell"], c["endpoint_leak"]) for c in row["cells"]])
                rng.append(ABSENT if lk is None else "[%.1f--%.1f]\\%%" % (100 * lk["min"], 100 * lk["max"]))
                pl = mstat([(c["cell"], c["plasticity"]) for c in row["cells"]])
                rng.append(ABSENT if pl is None else "[%.3f--%.3f]" % (pl["min"], pl["max"]))
                body.append(rng)
        # paired section
        if orig:
            o_ps = [c["post_settle"] for c in orig["cells"] if c["post_settle"] is not None]
            o_rng = (min(o_ps), max(o_ps)) if o_ps else None
            o_by = {(c["order"], c["seed"]): c for c in orig["cells"]}
            for vname, row, _man in groups[1:]:
                for c in sorted(row["cells"], key=lambda x: (x["order"], x["seed"])):
                    oc = o_by.get((c["order"], c["seed"]))
                    if oc is None or oc["post_settle"] is None or c["post_settle"] is None:
                        pbody.append(["SEQ-%s" % vname, tex_escape(c["cell"]), ABSENT, ABSENT, "unmatched", ABSENT])
                        pcsv.append({"block": bl, "variant": vname, "cell": c["cell"], "status": "unmatched"})
                        status = "partial"
                        continue
                    d = c["post_settle"] - oc["post_settle"]
                    removed = (1 - c["post_settle"] / oc["post_settle"]) if oc["post_settle"] else None
                    pos = ABSENT
                    if o_rng:
                        pos = ("below" if c["post_settle"] < o_rng[0] else "above" if c["post_settle"] > o_rng[1] else "inside")
                    pbody.append(["SEQ-%s" % vname, tex_escape(c["cell"]), "%+.3f" % d,
                                  ABSENT if removed is None else "%.0f\\%%" % (100 * removed), pos,
                                  ABSENT if not o_rng else "[%.3f--%.3f] (n=%d)" % (o_rng[0], o_rng[1], len(o_ps))])
                    pcsv.append({"block": bl, "variant": vname, "cell": c["cell"], "order": c["order"], "seed": c["seed"],
                                 "post_settle_variant": num(c["post_settle"]), "post_settle_original": num(oc["post_settle"]),
                                 "delta_post_settle": num(d), "churn_removed": num(removed), "position_vs_original_range": pos,
                                 "original_range_min": num(o_rng[0]) if o_rng else "", "original_range_max": num(o_rng[1]) if o_rng else "",
                                 "original_n": len(o_ps), "status": "matched"})
        else:
            pbody.append(("span", "no original SEQ cells in this block -- pairing ABSENT"))
            status = "partial"
    comments = ["SEQ original vs neutral vs amplified answer prior; per seed then range; '--' = ABSENT."]
    files = write_tex_table(stem + ".tex", "ll" + "r" * (5 + n_st + 2), header, body, comments, notes)
    files += write_tex_table(pstem + ".tex", "llrrll", ["variant", "cell", "$\\Delta$ post-settle",
                                                        "churn removed", "position", "original range"], pbody,
                             ["Per-seed pairing of the variant against the original SEQ cell with the same (ordering, seed)."],
                             ["$\\Delta$ post-settle = variant $-$ original post-settling $\\Sigma|\\Delta c|$; churn removed "
                              "= $1 - \\Sigma_{\\mathrm{variant}}/\\Sigma_{\\mathrm{original}}$ (mechanism\\_intervention.md "
                              "sec.~5, partial clause); position = where the variant's value falls against the original "
                              "seed range [min--max]."])
    fields = ["block", "variant", "cell", "order", "seed", "sum_abs_dc", "post_settle", "endpoint_c", "endpoint_dprime",
              "max_abs_dd", "f1_tripped", "endpoint_leak", "plasticity", "manifest"]
    for k in range(1, n_st + 1):
        fields += ["dc_%d" % k, "dose_%d" % k, "marker_%d" % k]
    files.append(write_csv(stem + ".csv", fields, csv_rows))
    files.append(write_csv(pstem + ".csv", ["block", "variant", "cell", "order", "seed", "post_settle_variant",
                                            "post_settle_original", "delta_post_settle", "churn_removed",
                                            "position_vs_original_range", "original_range_min", "original_range_max",
                                            "original_n", "status"], pcsv))
    return files, status


# ---------------------------------------------------------------------------
# method_prior_share
# ---------------------------------------------------------------------------
def load_blind(path):
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        die("blind JSON not found: %s" % path)
    with open(path) as f:
        try:
            res = json.load(f)
        except ValueError as e:
            die("%s: not JSON (%s)" % (path, e))
    require(res, ["runtag", "base", "stages", "notes"], path)
    pr = FC.parse_runtag(res["runtag"])
    if pr is None:
        die("%s: runtag %r is not a parsable RUNTAG" % (path, res["runtag"]))
    rhos = []
    for k, v in res["stages"].items():
        if v == "ABSENT" or (isinstance(v, dict) and "status" in v and "rho_prior_share" not in v):
            continue
        require(v, ["b_real", "b_blind", "rho_prior_share", "blind", "real", "n_aligned"], "%s:stages.%s" % (path, k))
        rho = v["rho_prior_share"]
        if rho is not None and not (isinstance(rho, float) and math.isnan(rho)):
            rhos.append(rho)
    reading = None
    if rhos:
        m = sum(rhos) / len(rhos)
        reading = "prior-borne" if m > RHO_PRIOR_BORNE else "mixed" if m > RHO_MIXED else "evidence-conditional"
        if "summary" not in res:
            die("%s: stages carry rho values but no summary block" % path)
        jr = res["summary"].get("reading", "")
        if not jr.startswith(reading):
            die("%s: reading re-derived here (%s, rho mean %.3f) disagrees with the JSON's %r" % (path, reading, m, jr))
        if abs(res["summary"]["rho_mean"] - m) > 1e-9:
            die("%s: summary.rho_mean %.6f != mean of stage rhos %.6f" % (path, res["summary"]["rho_mean"], m))
    return {"path": path, "sha256": sha256(path), "res": res, "info": pr, "reading": reading,
            "rho_mean": (sum(rhos) / len(rhos)) if rhos else None, "n_rho": len(rhos)}


def prior_share_table(blinds, tables_dir):
    stem = os.path.join(tables_dir, "method_prior_share")
    notes = ["$\\rho_k = b_k^{\\mathrm{blind}} / b_k$ with $b_k$ = mean shift of $g = z_{yes}-z_{no}$ from base at stage $k$ on "
             "the real POPE images and $b_k^{\\mathrm{blind}}$ the same shift with every image replaced by constant gray "
             "(analysis/blind\\_prior\\_share.py). Pre-registered reading on the per-cell mean over stages: prior-borne "
             "($\\bar\\rho > %.1f$: the drift lives in the image-independent answer prior), mixed ($> %.1f$), "
             "evidence-conditional otherwise." % (RHO_PRIOR_BORNE, RHO_MIXED),
             "'--' = stage ABSENT (no blind dump); $^{\\ddagger}$ = too few aligned items (< 100); n/a = $|b_k| < 10^{-6}$ "
             "(ratio undefined). max$|d'_{\\mathrm{blind}}|$ is the sanity audit (should sit near 0: no evidence). "
             "One row per cell; per-arm summary = range over cells and the reading count."]
    if not blinds:
        files = write_tex_table(stem + ".tex", "llrrl", ["arm", "cell", "$\\rho$", "max$|d'_{\\mathrm{blind}}|$", "reading"],
                                [("span", "ABSENT: no blind JSON given")], ["ABSENT placeholder."], notes)
        files.append(write_csv(stem + ".csv", ["status"], [{"status": "ABSENT"}]))
        return files, "absent"
    stage_keys = []
    for B in blinds:
        for k in B["res"]["stages"]:
            if k not in stage_keys:
                stage_keys.append(k)
    stage_keys.sort(key=lambda s: (len(s), s))
    header = ["arm", "cell"] + ["$\\rho_{%s}$" % k for k in stage_keys] + ["max$|d'_{\\mathrm{blind}}|$", "$n$", "reading"]
    body, csv_rows = [], []
    status = "complete"
    by_arm = {}
    for B in blinds:
        tok = token_of(B["info"]["arm"], B["info"])
        by_arm.setdefault((B["info"]["suite"], B["info"]["backbone"], tok), []).append(B)
    first = True
    for (suite, bb, tok), Bs in sorted(by_arm.items(), key=lambda t: (t[0][0], t[0][1], row_sort_key(t[0][2]))):
        if not first:
            body.append("midrule")
        first = False
        lab, plain = label_of(tok)
        body.append(("span", "%s / %s" % (suite, tex_escape(FC.BACKBONE_LABEL.get(bb, bb)))))
        per_stage = {k: [] for k in stage_keys}
        readings = []
        for B in sorted(Bs, key=lambda x: (x["info"]["order"], x["info"]["seed"])):
            cell = "%s/s%d" % (B["info"]["order"], B["info"]["seed"])
            cells = [lab, tex_escape(cell)]
            rec = {"suite": suite, "backbone": bb, "method": plain, "token": tok, "runtag": B["res"]["runtag"], "cell": cell,
                   "reading": B["reading"] or "", "rho_mean": num(B["rho_mean"]), "n_rho": B["n_rho"], "path": B["path"]}
            maxd, nmin = None, None
            for k in stage_keys:
                v = B["res"]["stages"].get(k)
                if v is None or v == "ABSENT":
                    cells.append(ABSENT)
                    rec["rho_%s" % k] = ""
                    status = "partial"
                    continue
                if isinstance(v, dict) and "rho_prior_share" not in v:
                    cells.append("--$^{\\ddagger}$")
                    rec["rho_%s" % k] = ""
                    rec["status_%s" % k] = v.get("status", "")
                    status = "partial"
                    continue
                rho = v["rho_prior_share"]
                if rho is None or (isinstance(rho, float) and math.isnan(rho)):
                    cells.append("n/a")
                    rec["rho_%s" % k] = "nan"
                else:
                    cells.append("%+.2f" % rho)
                    rec["rho_%s" % k] = num(rho)
                    per_stage[k].append(rho)
                rec["b_real_%s" % k] = num(v["b_real"])
                rec["b_blind_%s" % k] = num(v["b_blind"])
                rec["dprime_blind_%s" % k] = num(v["blind"]["dprime"])
                rec["n_aligned_%s" % k] = v["n_aligned"]
                maxd = max(maxd or 0.0, abs(v["blind"]["dprime"]))
                nmin = v["n_aligned"] if nmin is None else min(nmin, v["n_aligned"])
            cells.append(ABSENT if maxd is None else "%.2f" % maxd)
            cells.append(ABSENT if nmin is None else str(nmin))
            cells.append(tex_escape(B["reading"]) if B["reading"] else ABSENT)
            rec["max_abs_dprime_blind"] = num(maxd)
            rec["n_aligned_min"] = "" if nmin is None else nmin
            if B["reading"]:
                readings.append(B["reading"])
            body.append(cells)
            csv_rows.append(rec)
        if len(Bs) > 1:
            rng = ["\\quad range", "n=%d" % len(Bs)]
            for k in stage_keys:
                xs = per_stage[k]
                rng.append("[%+.2f--%+.2f]" % (min(xs), max(xs)) if xs else ABSENT)
            rng += [ABSENT, ABSENT]
            cnt = {r: readings.count(r) for r in set(readings)}
            rng.append(", ".join("%d/%d %s" % (cnt[r], len(Bs), r) for r in sorted(cnt)) or ABSENT)
            body.append(rng)
    files = write_tex_table(stem + ".tex", "ll" + "r" * (len(stage_keys) + 2) + "l", header, body,
                            ["Language-prior share of criterion drift per arm cell and stage; '--' = ABSENT."], notes)
    fields = ["suite", "backbone", "method", "token", "runtag", "cell", "reading", "rho_mean", "n_rho",
              "max_abs_dprime_blind", "n_aligned_min"]
    for k in stage_keys:
        fields += ["rho_%s" % k, "b_real_%s" % k, "b_blind_%s" % k, "dprime_blind_%s" % k, "n_aligned_%s" % k, "status_%s" % k]
    fields.append("path")
    files.append(write_csv(stem + ".csv", fields, csv_rows))
    return files, status


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def make_figure(block, out_dir, name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import fig_style as FS
    except Exception as e:  # matplotlib absent
        print("%s figure skipped: %s" % (TAG, e))
        return [], "absent: matplotlib unavailable (%s)" % e
    FS.apply_style()
    rows = [r for r in block["rows"] if r["source"] == "scorecard" and r["n_cells"] > 0]
    if not rows:
        print("%s figure skipped for %s: no scorecard rows with complete cells" % (TAG, block["label"]))
        return [], "absent: no complete cells"
    ident = {"seq": FS.SEQ_COLOR, "joint": FS.JOINT_COLOR, "er": FS.ER_COLOR, "er500": FS.ER_COLOR,
             "cecf": FS.V1_COLOR, "anchor": FS.V2_COLOR}
    extra = ["#56B4E9", "#F0E442", "#000000", "#999999"]
    hatches = [None, "//", "..", "xx", "\\\\", "++"]
    colors, hatch = {}, {}
    ei = 0
    for r in rows:
        if r["token"] in ident:
            colors[r["token"]] = ident[r["token"]]
            hatch[r["token"]] = "///" if r["token"] == "cecf" else None
        else:
            colors[r["token"]] = extra[ei % len(extra)]
            hatch[r["token"]] = hatches[(ei // len(extra)) % len(hatches)]
            ei += 1
    orders = sorted({c["order"] for r in rows for c in r["cells"]})
    omark = {o: m for o, m in zip(orders, ["o", "s", "^", "D", "v", "P"])}
    ooff = {o: (i - (len(orders) - 1) / 2) * 0.16 for i, o in enumerate(orders)}
    k = block["hdr"]["k"]
    fig, axs = plt.subplots(1, 3, figsize=(FS.FULL_W, FS.h_full(2.45)),
                            gridspec_kw={"width_ratios": [1.25, 1.0, 1.0], "wspace": 0.42})
    x = list(range(len(rows)))
    labels = [r["plain"] for r in rows]

    def bars(ax, key, ylabel, title):
        for i, r in enumerate(rows):
            st = r["metrics"][key]
            if st is None:
                continue
            ax.bar([i], [st["mean"]], width=0.62, color=colors[r["token"]], edgecolor="black", linewidth=0.5,
                   hatch=hatch[r["token"]], zorder=2, alpha=0.85)
            for c in r["cells"]:
                if c[key] is not None:
                    ax.scatter([i + ooff[c["order"]]], [c[key]], s=11, marker=omark[c["order"]], color="black",
                               zorder=4, linewidths=0.4, edgecolors="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=6.0)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=7.2)

    bars(axs[0], "sum_abs_dc", "$\\Sigma|\\Delta c|$ (pooled POPE)", "(a) criterion churn: bar = mean, dots = cells")
    bars(axs[1], "chair60", "endpoint CHAIR$_i$@%d" % k, "(b) length-controlled endpoint hallucination")
    base = block["rep"].get("base") or {}
    if base.get("chair") and base["chair"].get("chair_i_at_k") is not None:
        axs[1].axhline(base["chair"]["chair_i_at_k"], color=FS.BASE_COLOR, ls=":", lw=0.8, zorder=1)
        axs[1].text(len(rows) - 0.55, base["chair"]["chair_i_at_k"], "base", fontsize=5.8, color="#666666", va="bottom", ha="right")
    ax = axs[2]
    xt, xl = [], []
    pos = 0
    for r in rows:
        if r["is_ref"]:
            continue
        pts = [c for c in r["cells"] if c["chair_gap_point"] is not None]
        for c in pts:
            xx = pos + ooff[c["order"]]
            filled = bool(c["chair_ci_excludes_zero"])
            ax.errorbar([xx], [c["chair_gap_point"]],
                        yerr=[[c["chair_gap_point"] - c["chair_ci_lo"]], [c["chair_ci_hi"] - c["chair_gap_point"]]],
                        fmt=omark[c["order"]], color=colors[r["token"]], markersize=3.2, capsize=1.6, linewidth=0.8,
                        markerfacecolor=colors[r["token"]] if filled else "white", markeredgewidth=0.7, zorder=3)
        xt.append(pos)
        xl.append(r["plain"])
        pos += 1
    ax.axhline(0, color=FS.BASE_COLOR, lw=0.8, zorder=1)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, rotation=35, ha="right", fontsize=6.0)
    ax.set_ylabel("$\\Delta$ CHAIR$_i$@%d vs %s" % (k, next((r["plain"] for r in rows if r["is_ref"]), "ref")))
    ax.set_title("(c) paired gap per cell, image-bootstrap CI95", fontsize=7.2)
    if orders and len(orders) > 1:
        for o in orders:
            axs[0].scatter([], [], marker=omark[o], color="black", s=11, label=o)
        axs[0].legend(fontsize=5.8, loc="upper right", title="ordering", title_fontsize=5.8)
    fig.suptitle("%s -- ref %s; %d stages; B=%d" % (block["label"], block["ref"], block["hdr"]["n_stages"], block["hdr"]["B"]),
                 fontsize=7.0, y=1.02)
    fig.tight_layout()
    rendered = []
    FS.save_fig(fig, out_dir, name, rendered, TAG.strip("[]"))
    cap = os.path.join(out_dir, name + "_caption.txt")
    n_cells = sum(r["n_cells"] for r in rows)
    with open(cap, "w") as f:
        f.write("Method scorecard, %s (reference arm %s; %d complete (ordering, seed) cells across %d arms). "
                "(a) Sum|dc| = sum_k |c_k - c_{k-1}| with c_0 = base; bar = mean over cells, dots = per-cell values "
                "(marker = ordering). (b) endpoint length-controlled CHAIR_i@%d, same convention; dotted line = base. "
                "(c) paired endpoint CHAIR_i@%d gap arm - ref per matched cell with its paired image-bootstrap CI95 "
                "(B=%d, seed %d); filled marker = CI excludes 0. Ranges across cells are the seed/ordering spread, "
                "not confidence intervals; no cross-seed CI is drawn. Post-hoc logit rows of the table are not drawn "
                "(different instrument).\n"
                % (block["label"], block["ref"], n_cells, len(rows), k, k, block["hdr"]["B"], block["hdr"]["seed"]))
    return [os.path.join(out_dir, name + ".pdf"), os.path.join(out_dir, name + ".png"), cap], "complete"


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scorecard", nargs="+", required=True, help="[LABEL=]path to scorecard.json (one block each)")
    ap.add_argument("--logits", nargs="*", default=[], help="logit_bias_analysis.py JSONs (one cell each)")
    ap.add_argument("--blind", nargs="*", default=[], help="blind_prior_share.py JSONs (one cell each)")
    ap.add_argument("--aggregate", default=None, help="fs_aggregate.json: JOINT row + cross-check of every cell")
    ap.add_argument("--agg_tol", type=float, default=2e-4, help="aggregate-vs-scorecard tolerance (both round to 4 dp)")
    ap.add_argument("--manifest_original", default=None, help="pilot_manifest.json (default: fullstudy/pilot_manifest.json)")
    ap.add_argument("--manifest_neutral", default=None, help="neutral_manifest.json (cluster data/tasks_neutral/)")
    ap.add_argument("--manifest_amplified", default=None, help="amplified_manifest.json")
    ap.add_argument("--tables_dir", default=str(REPO / "paper" / "tables"))
    ap.add_argument("--out_dir", default=str(REPO / "analysis" / "out"))
    ap.add_argument("--with_pilot_reference", action="store_true", help="append the pilot-run reference rows (context only)")
    ap.add_argument("--no_fig", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(args.tables_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    blocks = [load_scorecard(s) for s in args.scorecard]
    labels = [b["label"] for b in blocks]
    if len(set(labels)) != len(labels):
        die("two scorecards share the label %r; pass LABEL=path" % next(l for l in labels if labels.count(l) > 1))
    ks = {b["hdr"]["k"] for b in blocks}
    if len(ks) != 1:
        die("scorecards use different CHAIR length budgets k=%s; one table cannot mix them" % sorted(ks))
    k_words = ks.pop()
    for b in blocks:
        build_block_rows(b)

    inputs = [{"role": "scorecard", "label": b["label"], "path": b["path"], "sha256": b["sha256"]} for b in blocks]
    warnings = []

    # aggregate: JOINT row + cross-check
    AG = load_aggregate(args.aggregate) if args.aggregate else None
    crosscheck = []
    if AG:
        inputs.append({"role": "aggregate", "path": AG["path"], "sha256": AG["sha256"]})
        for b in blocks:
            crosscheck.append(crosscheck_aggregate(AG, b, args.agg_tol))
            if not any(r["token"] == "joint" and r["n_cells"] for r in b["rows"]):
                jr = joint_row_from_aggregate(AG, b)
                if jr:
                    b["rows"] = [r for r in b["rows"] if r["token"] != "joint"] + [jr]
                    b["rows"].sort(key=lambda r: row_sort_key(r["token"]))

    # logits -> post-hoc rows
    for p in args.logits:
        L = load_logits(p)
        inputs.append({"role": "logits", "path": L["path"], "sha256": L["sha256"], "runtag": L["res"]["runtag"]})
        pc = posthoc_cell(L)
        blk = next((b for b in blocks if b["suite"] == pc["suite"] and b["backbone"] == pc["backbone"]), None)
        if blk is None:
            blk = {"path": None, "sha256": None, "label": "%s / %s (post-hoc only)" % (pc["suite"], FC.BACKBONE_LABEL.get(pc["backbone"], pc["backbone"])),
                   "rep": {"arms": {}, "base": None}, "hdr": {"suite": pc["suite"], "ref": pc["arm"], "k": k_words, "B": 0, "seed": 0,
                                                              "base_dir": "n/a", "base_present": False, "n_stages": pc["n_stages"]},
                   "suite": pc["suite"], "backbone": pc["backbone"], "ref": pc["arm"], "ref_token": pc["arm"], "rows": [],
                   "cells_long": [], "posthoc_cells": [], "warnings": []}
            blocks.append(blk)
            warnings.append("logits %s: no scorecard block for %s/%s; emitted as a post-hoc-only block" % (p, pc["suite"], pc["backbone"]))
        blk["posthoc_cells"].append(pc)
    for b in blocks:
        if b["posthoc_cells"]:
            b["rows"] = [r for r in b["rows"] if r["token"] not in POSTHOC_TOKENS] + posthoc_rows(b["posthoc_cells"], b["ref_token"])
            b["rows"].sort(key=lambda r: row_sort_key(r["token"]))

    artifacts = []

    def add(paths, status, kind, notes=None, inp=None):
        for p in paths:
            if p is None:
                continue
            artifacts.append({"path": os.path.abspath(p), "kind": kind, "status": status, "bytes": os.path.getsize(p),
                              "sha256": sha256(p), "inputs": inp if inp is not None else [i["path"] for i in inputs],
                              "notes": notes or ""})

    files, st = main_table(blocks, args.tables_dir, k_words, {"pilot_rows": args.with_pilot_reference})
    add(files, st, "method_main")
    ps = posthoc_stage_csv(blocks, args.tables_dir)
    if ps:
        add([ps], "complete", "method_main_posthoc_stages", inp=[i["path"] for i in inputs if i["role"] == "logits"])

    man = {}
    orig = args.manifest_original or (str(REPO / "fullstudy" / "pilot_manifest.json")
                                      if (REPO / "fullstudy" / "pilot_manifest.json").is_file() else None)
    suite0 = blocks[0]["suite"]
    for name, path in (("original", orig), ("neutral", args.manifest_neutral), ("amplified", args.manifest_amplified)):
        m = load_manifest_stats(path, suite0)
        if m:
            man[name] = m
            inputs.append({"role": "manifest_" + name, "path": m["path"], "sha256": m["sha256"]})
    files, st = intervention_tables(blocks, args.tables_dir, man)
    add(files, st, "method_intervention",
        notes="" if st != "partial" else "partial: a variant lacks its manifest (no sign markers) or a cell is unmatched")

    blinds = [load_blind(p) for p in args.blind]
    for B in blinds:
        inputs.append({"role": "blind", "path": B["path"], "sha256": B["sha256"], "runtag": B["res"]["runtag"]})
    files, st = prior_share_table(blinds, args.tables_dir)
    add(files, st, "method_prior_share", inp=[B["path"] for B in blinds])

    if not args.no_fig:
        for i, b in enumerate(blocks):
            if b["path"] is None:
                continue
            name = "fig_method_scorecard" if i == 0 else "fig_method_scorecard_" + slug(b["label"])
            files, st = make_figure(b, args.out_dir, name)
            add(files, st, "figure", inp=[b["path"]])

    manifest = {
        "generator": {"script": str(Path(__file__).resolve()), "sha256": sha256(__file__), "argv": sys.argv[1:] if argv is None else argv,
                      "python": sys.version.split()[0], "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "method_scorecard_rules": {"F1_DPRIME_TOL": MS.F1_DPRIME_TOL, "P2_PLASTICITY_TOL": MS.P2_PLASTICITY_TOL},
                      "chair_k_words": k_words},
        "inputs": inputs,
        "blocks": [{"label": b["label"], "suite": b["suite"], "backbone": b["backbone"], "ref": b["ref"],
                    "base_dir": b["hdr"]["base_dir"], "base_present": b["hdr"]["base_present"],
                    "scorecard": b["path"],
                    "arms": [{"token": r["token"], "label": r["plain"], "source": r["source"], "n_cells": r["n_cells"],
                              "orders": r["orders"], "overall": r["verdicts"]["overall"] if r["verdicts"] else None}
                             for r in b["rows"]],
                    "intervention_arms": [{"token": r["token"], "n_cells": r["n_cells"]} for r in b.get("intervention", [])],
                    "posthoc_cells": [c["runtag"] for c in b["posthoc_cells"]]} for b in blocks],
        "aggregate_crosscheck": crosscheck,
        "warnings": warnings,
        "artifacts": artifacts,
        "style": "per-cell values and [min-max] over complete (ordering, seed) cells; orderings are strata; "
                 "no pooled cross-seed CI (design_notes/review_statistics.md sec 2); '--' = ABSENT, nothing imputed",
    }
    mpath = os.path.join(args.out_dir, "method_artifacts_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)
    print("%s wrote %s" % (TAG, mpath))
    for w in warnings:
        print("%s WARNING: %s" % (TAG, w))
    print("%s DONE: %d artifacts, %d blocks, %d logits cells, %d blind cells (%.1fs)"
          % (TAG, len(artifacts), len(blocks), sum(len(b["posthoc_cells"]) for b in blocks), len(blinds), time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
