#!/usr/bin/env python3
"""pcr_transfer.py -- is a train-time criterion method a METHOD, or train-time
post-hoc correction? (design_notes/review_method_design.md, threat M2;
fullstudy/FULLSTUDY_PREREG.md, 2026-09-08 "design red-team consequences" (b))

The threat. critp / anchorcrit / CNP regularize mean g = z_yes - z_no on a
label-free probe that is the POPE endpoint distribution minus its answer key
(COCO images x 80 COCO classes x the POPE question template). "The criterion did
not move" on that endpoint is the objective restated, and the zero-training
label-free post-hoc scalar -- PCR: delta_k = mean g_probe(base) - mean g_probe(k),
decide gap + delta_k > 0 -- already achieves it for free on the original template.
The only axis on which a TRAINED pin can beat the free scalar is template
transfer: a REPHRASED POPE template (same ids / images / GT, question rewritten)
that neither the probe nor PCR ever saw.

Decision rule (pre-registered, per matched cell = same suite / order / seed):
    the arm is a METHOD on this cell  iff
    sum_k |dc_rephr(k)| (arm, UNcorrected)
        <  sum_k |dc_rephr(k)| (SEQ reference, PCR-corrected with the reference's
                                 OWN probe delta_k, transferred from the original
                                 template to the rephrased one)
    otherwise it is reported as train-time PCR: a diagnostic, not a contribution.
The prereg requires the boolean to be True in 3/3 seeds; this script decides ONE
cell (it emits the boolean and the two numbers); aggregation across seeds is the
scorecard's job.

Per checkpoint k of an arm and for the base, sb_fs_eval.sbatch writes under
<results>/<label>/ (label = <RUNTAG>_k<K>, base = e.g. llava15_base):
    pope_logits.jsonl              real POPE, ORIGINAL template
    pope_rephr_logits.jsonl        same items, REPHRASED template
    probe_prompts_logits.jsonl     COCO probe (unlabeled: gt=None, category='probe')
    probe_oi_prompts_logits.jsonl  Open Images TRAIN probe
    probe_noise_prompts_logits.jsonl  content-free noise probe
Row schema (pilot/eval_gen.py --dump_logits): {id, gt, category, z_yes, z_no, gap,
argmax_is, ...}; greedy answers "yes" iff gap > 0.

What is computed, per stage k:
 (1) delta_k^probe = mean g_probe(base) - mean g_probe(k), for EACH available
     probe (coco / oi / noise), fitted on the probe only, never on POPE.
 (2) the decision rule gap + delta > 0 applied to the arm's POPE gaps on the
     ORIGINAL template and on the REPHRASED template: c, d', F1 (plus H, FA,
     yes-rate, accuracy) uncorrected vs PCR-corrected (per probe) vs base.
 (3) the arm's own criterion on the rephrased template, c_rephr(k), and
     sum_k |dc_rephr| across stages with base = stage 0 (full sum; the
     post-settling sum, stages 2..T, is reported alongside and never judged --
     design review M1).
 (4) with --ref_runtag: the DECISIVE comparison above, per probe; the boolean
     `arm_is_method` uses --primary_probe (default coco = the probe critp trains
     on, rendered with the anchor's / POPE's own template). Both trajectories are
     recomputed on the arm-and-reference JOINT id set. The same contrast on the
     original template is reported as context (there PCR wins by construction).
 (5) template-transfer gap: c_orig(k) - c_rephr(k) per stage, and the same minus
     the base's own gap (a method that only pins the trained template shows a
     large shift of this gap; a genuine prior fix does not).

SDT primitives mirror analysis/logit_bias_analysis.py sdt() exactly (rates
clipped to [1/(2n), 1-1/(2n)], NormalDist().inv_cdf, c = -(zH+zFA)/2,
d' = zH - zFA, F1 with the same guards); ids are aligned across dumps by exact
`id`; dedup on id (last occurrence wins, counted). ABSENT cells are reported,
never imputed; a sum over zero transitions is ABSENT, never 0.0. Malformed rows
(non-finite gap, POPE gt not in {yes,no}, gt disagreeing across dumps for one
id) abort loudly. Stdlib only (Python 3.9+).

usage:
  pcr_transfer.py --results <results_fs> --runtag <RT> --base <label>
                  [--stages k1,k2,...] [--ref_runtag <SEQ RT>]
                  [--primary_probe coco|oi|noise] [--min_join_frac 0.5] --out <json>
"""
import argparse
import json
import math
import os
import re
import sys
from statistics import NormalDist, mean

_z = NormalDist().inv_cdf

TEMPLATES = {"orig": "pope_logits.jsonl", "rephr": "pope_rephr_logits.jsonl"}
PROBES = {"coco": "probe_prompts_logits.jsonl",
          "oi": "probe_oi_prompts_logits.jsonl",
          "noise": "probe_noise_prompts_logits.jsonl"}
ABSENT = "ABSENT"
TAG = "[pcr_transfer]"

DECISION_RULE = ("arm is a METHOD on this cell iff sum|dc_rephr|(arm, uncorrected) < "
                 "sum|dc_rephr|(SEQ ref, PCR-corrected with the ref's own probe delta_k); "
                 "otherwise train-time PCR (diagnostic). delta_k = mean g_probe(base) - "
                 "mean g_probe(k), fitted on the probe only; decision gap + delta_k > 0.")


def die(msg):
    raise SystemExit(f"{TAG} {msg}")


# ---------------------------------------------------------------------------
# SDT on gaps (threshold 0) -- verbatim semantics of logit_bias_analysis.sdt
# ---------------------------------------------------------------------------
def clipped_rate(k, n):
    """Rate k/n clipped to [1/(2n), 1-1/(2n)]. Returns (rate_for_z, fired)."""
    raw = k / n
    lo = 1.0 / (2.0 * n)
    hi = 1.0 - lo
    if raw < lo:
        return lo, True
    if raw > hi:
        return hi, True
    return raw, False


def sdt(gaps, yes, delta=0.0):
    """c / d' / F1 of the decision (gap + delta) > 0 against gt. `gaps`: floats,
    `yes`: bools (gt == 'yes'), same length. Mirrors logit_bias_analysis.sdt on
    (gaps + delta) at threshold 0."""
    ny = sum(1 for y in yes if y)
    nn = len(yes) - ny
    if ny == 0 or nn == 0:
        die(f"degenerate split (n_yes={ny}, n_no={nn})")
    hits = fas = 0
    for g, y in zip(gaps, yes):
        if g + delta > 0.0:
            if y:
                hits += 1
            else:
                fas += 1
    Hc, hcl = clipped_rate(hits, ny)
    FAc, fcl = clipped_rate(fas, nn)
    zH, zFA = _z(Hc), _z(FAc)
    tp, fp, fn, tn = hits, fas, ny - hits, nn - fas
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {
        "H": hits / ny, "FA": fas / nn, "dprime": zH - zFA, "c": -0.5 * (zH + zFA),
        "yes_rate": (hits + fas) / (ny + nn), "accuracy": (tp + tn) / (ny + nn),
        "precision": prec, "recall": rec, "f1": 2 * prec * rec / max(1e-9, prec + rec),
        "n": ny + nn, "n_yes": ny, "n_no": nn,
        "clipped": ("H" if hcl else "") + ("FA" if fcl else "") or "none",
    }


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load_dump(path, labeled):
    """-> (rows_by_id | None, audit). None + status ABSENT when the file does not
    exist. Dedup on id, last occurrence wins (fs_common.read_rows_dedup
    convention), duplicates counted. `labeled` (POPE dumps): gt must be yes/no.
    Probe dumps carry gt=None; a probe dump with yes/no labels is noted, not
    fatal. Non-finite / missing gap aborts (a broken dump, not an absent one)."""
    audit = {"path": path}
    if not os.path.isfile(path):
        audit["status"] = ABSENT
        return None, audit
    rows = {}
    n_lines = n_blank = n_dup = n_noid = 0
    gt_counts = {}
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                n_blank += 1
                continue
            n_lines += 1
            try:
                r = json.loads(s)
            except json.JSONDecodeError as e:
                die(f"{path}:{ln}: bad JSON ({e})")
            rid = r.get("id")
            if rid is None:
                n_noid += 1
                continue
            rid = str(rid)
            g = r.get("gap")
            if isinstance(g, bool) or not isinstance(g, (int, float)) or not math.isfinite(g):
                die(f"{path}:{ln}: row {rid!r} has non-finite/missing gap {g!r}")
            gt = r.get("gt")
            gt_counts[str(gt)] = gt_counts.get(str(gt), 0) + 1
            if labeled and gt not in ("yes", "no"):
                die(f"{path}:{ln}: row {rid!r} gt={gt!r} (POPE dump must be yes/no)")
            if rid in rows:
                n_dup += 1
            rows[rid] = {"gap": float(g), "gt": gt, "argmax_is": r.get("argmax_is")}
    n_other = sum(1 for r in rows.values() if r["argmax_is"] == "other")
    audit.update({
        "status": "OK" if rows else "EMPTY",
        "n_lines": n_lines, "n_blank": n_blank, "n_rows": len(rows), "dup_ids": n_dup,
        "rows_without_id": n_noid, "gt_counts": gt_counts,
        "argmax_other_rate": (n_other / len(rows)) if rows else None,
    })
    if not rows:
        return None, audit
    if not labeled and any(k in ("yes", "no") for k in gt_counts):
        audit["note"] = "probe dump carries yes/no labels -- expected gt=None (probe_prompts.py)"
    return rows, audit


def common_ids(dumps):
    ids = None
    for rows in dumps.values():
        ids = set(rows) if ids is None else ids & set(rows)
    return ids or set()


def stage_label(runtag, k):
    return f"{runtag}_{k}"


def discover_stages(results, runtags):
    """All k for which <results>/<runtag>_k<N> exists, union over runtags,
    numerically sorted. Falls back to k1..k6 (all ABSENT) when none exist."""
    ks = set()
    for rt in runtags:
        pat = re.compile(r"^" + re.escape(rt) + r"_k(\d+)$")
        if os.path.isdir(results):
            for d in os.listdir(results):
                m = pat.match(d)
                if m and os.path.isdir(os.path.join(results, d)):
                    ks.add(int(m.group(1)))
    if not ks:
        return [f"k{i}" for i in range(1, 7)], False
    return [f"k{i}" for i in sorted(ks)], True


# ---------------------------------------------------------------------------
# per-arm blocks
# ---------------------------------------------------------------------------
def load_arm_template(results, base, runtag, stages, tname, min_join, notes):
    """Load base + stage dumps of one template for one arm; align on the common
    id set (base and every PRESENT stage); check gt agreement per id.
    -> (raw | None, audit_block). raw = {ids, yes, g0, gk: {k: gaps}, n_base}."""
    fname = TEMPLATES[tname]
    block = {"file": fname, "file_audits": {}}
    base_rows, aud = load_dump(os.path.join(results, base, fname), labeled=True)
    block["file_audits"][base] = aud
    if base_rows is None:
        block["status"] = ABSENT
        block["note"] = f"base dump {fname} absent/empty -> {tname} template not analysable"
        return None, block
    stage_rows = {}
    for k in stages:
        lab = stage_label(runtag, k)
        rows, aud = load_dump(os.path.join(results, lab, fname), labeled=True)
        block["file_audits"][lab] = aud
        if rows is not None:
            stage_rows[k] = rows
            if abs(len(rows) - len(base_rows)) > 0.01 * len(base_rows):
                notes.append(f"WARN_ROWCOUNT {lab}/{fname}: {len(rows)} rows vs base {len(base_rows)}")
    present = [k for k in stages if k in stage_rows]
    ids = common_ids({"base": base_rows, **stage_rows})
    n_base = len(base_rows)
    if len(ids) < min_join * n_base:
        die(f"{runtag}/{fname}: only {len(ids)}/{n_base} ids joined across base + stages "
            f"{present} (< {min_join:.0%}) -- truncated or mismatched dumps, refusing")
    for k in present:
        for i in ids:
            if stage_rows[k][i]["gt"] != base_rows[i]["gt"]:
                die(f"gt mismatch for id {i!r}: base={base_rows[i]['gt']!r} "
                    f"{stage_label(runtag, k)}={stage_rows[k][i]['gt']!r}")
    ids = sorted(ids)
    raw = {"ids": ids, "yes": [base_rows[i]["gt"] == "yes" for i in ids],
           "g0": [base_rows[i]["gap"] for i in ids],
           "gk": {k: [stage_rows[k][i]["gap"] for i in ids] for k in present},
           "n_base": n_base}
    block.update({
        "status": "OK", "n_ids": len(ids), "n_base_rows": n_base,
        "dropped_per_file": {base: n_base - len(ids),
                             **{stage_label(runtag, k): len(stage_rows[k]) - len(ids) for k in present}},
        "stages_present": present, "stages_absent": [k for k in stages if k not in stage_rows],
    })
    return raw, block


def load_arm_probe(results, base, runtag, stages, pname, min_join, notes):
    """delta_k = mean g_probe(base) - mean g_probe(k) on the probe's common ids."""
    fname = PROBES[pname]
    block = {"file": fname, "file_audits": {}}
    base_rows, aud = load_dump(os.path.join(results, base, fname), labeled=False)
    block["file_audits"][base] = aud
    if base_rows is None:
        block["status"] = ABSENT
        block["note"] = f"base probe dump {fname} absent/empty"
        return block
    stage_rows = {}
    for k in stages:
        lab = stage_label(runtag, k)
        rows, aud = load_dump(os.path.join(results, lab, fname), labeled=False)
        block["file_audits"][lab] = aud
        if rows is not None:
            stage_rows[k] = rows
    present = [k for k in stages if k in stage_rows]
    if not present:
        block["status"] = ABSENT
        block["note"] = f"no stage probe dumps ({fname}) for {runtag}"
        return block
    ids = common_ids({"base": base_rows, **stage_rows})
    n_base = len(base_rows)
    if len(ids) < min_join * n_base:
        die(f"{runtag}/{fname}: only {len(ids)}/{n_base} probe ids joined (< {min_join:.0%})")
    ids = sorted(ids)
    m0 = mean(base_rows[i]["gap"] for i in ids)
    block.update({
        "status": "OK", "n_ids": len(ids), "n_base_rows": n_base,
        "dropped_per_file": {base: n_base - len(ids),
                             **{stage_label(runtag, k): len(stage_rows[k]) - len(ids) for k in present}},
        "stages_present": present, "stages_absent": [k for k in stages if k not in stage_rows],
        "base_mean_gap": m0, "stages": {},
    })
    for k in stages:
        if k not in stage_rows:
            block["stages"][k] = ABSENT
            continue
        mk = mean(stage_rows[k][i]["gap"] for i in ids)
        block["stages"][k] = {"mean_gap": mk, "delta": m0 - mk, "n_aligned": len(ids)}
    if all(block["stages"][k]["delta"] == 0.0 for k in present):
        notes.append(f"WARN_ZERO_DELTA {runtag}/{fname}: every probe delta is exactly 0.0 -- "
                     "verify the stage dumps are not copies of the base dump")
    return block


def probe_delta(probe_block, k):
    if not isinstance(probe_block, dict) or probe_block.get("status") != "OK":
        return None
    st = probe_block["stages"].get(k)
    return st["delta"] if isinstance(st, dict) else None


def template_metrics(raw, stages, probes):
    """(2): base / per-stage uncorrected / per-probe PCR-corrected sdt blocks."""
    out = {"base": sdt(raw["g0"], raw["yes"]), "stages": {}}
    for k in stages:
        if k not in raw["gk"]:
            out["stages"][k] = ABSENT
            continue
        gk = raw["gk"][k]
        st = {"uncorrected": sdt(gk, raw["yes"]), "pcr": {}}
        for pname, pb in probes.items():
            d = probe_delta(pb, k)
            st["pcr"][pname] = ABSENT if d is None else {"delta": d, **sdt(gk, raw["yes"], delta=d)}
        out["stages"][k] = st
    return out


def trajectory(c_by_stage, c0, stages):
    """Chain of consecutive PRESENT stages, base = stage 0. Sums over zero
    transitions are ABSENT (never a silent 0.0). post_settling drops the first
    transition (base -> first present stage), per design review M1."""
    seq = [("base", c0)] + [(k, c_by_stage[k]) for k in stages if c_by_stage.get(k) is not None]
    trans = [{"from": a, "to": b, "dc": cb - ca} for (a, ca), (b, cb) in zip(seq, seq[1:])]
    used = [k for k, _ in seq[1:]]
    full = sum(abs(t["dc"]) for t in trans) if trans else None
    post = sum(abs(t["dc"]) for t in trans[1:]) if len(trans) > 1 else None
    dev = {k: abs(c_by_stage[k] - c0) for k in used}
    return {
        "c": {"base": c0, **{k: (c_by_stage[k] if c_by_stage.get(k) is not None else ABSENT) for k in stages}},
        "transitions": trans, "n_transitions": len(trans),
        "sum_abs_dc_full": ABSENT if full is None else full,
        "sum_abs_dc_post_settling": ABSENT if post is None else post,
        "abs_dev_from_base": dev,
        "max_abs_dev_from_base": max(dev.values()) if dev else ABSENT,
        "stages_used": used, "stages_absent": [k for k in stages if k not in used],
        "complete": len(used) == len(stages),
    }


def arm_trajectories(metrics, stages, probes):
    """(3): uncorrected + per-probe PCR-corrected trajectories from a template_metrics block."""
    c0 = metrics["base"]["c"]
    unc = {k: (metrics["stages"][k]["uncorrected"]["c"] if isinstance(metrics["stages"][k], dict) else None)
           for k in stages}
    out = {"uncorrected": trajectory(unc, c0, stages), "pcr": {}}
    for pname in probes:
        cp = {}
        for k in stages:
            st = metrics["stages"][k]
            cp[k] = st["pcr"][pname]["c"] if isinstance(st, dict) and isinstance(st["pcr"][pname], dict) else None
        out["pcr"][pname] = trajectory(cp, c0, stages) if any(v is not None for v in cp.values()) else ABSENT
    return out


def template_gap(m_orig, m_rephr, stages):
    """(5): c_orig(k) - c_rephr(k), uncorrected, and the same minus the base's own gap."""
    if m_orig is None or m_rephr is None:
        return {"status": ABSENT, "note": "needs both templates"}
    g0 = m_orig["base"]["c"] - m_rephr["base"]["c"]
    out = {"status": "OK", "base": {"c_orig": m_orig["base"]["c"], "c_rephr": m_rephr["base"]["c"], "gap": g0},
           "stages": {}}
    shifts = []
    for k in stages:
        a, b = m_orig["stages"][k], m_rephr["stages"][k]
        if not (isinstance(a, dict) and isinstance(b, dict)):
            out["stages"][k] = ABSENT
            continue
        co, cr = a["uncorrected"]["c"], b["uncorrected"]["c"]
        out["stages"][k] = {"c_orig": co, "c_rephr": cr, "gap": co - cr, "gap_minus_base": co - cr - g0}
        shifts.append(abs(co - cr - g0))
    out["max_abs_gap_shift"] = max(shifts) if shifts else ABSENT
    return out


def analyse_arm(results, base, runtag, stages, min_join, notes):
    """Everything single-arm: (1) probe deltas, (2) per-template metrics, (3)
    trajectories, (5) template gap. Returns (json_block, raw_by_template)."""
    probes = {p: load_arm_probe(results, base, runtag, stages, p, min_join, notes) for p in PROBES}
    raws, metrics, blocks = {}, {}, {}
    for t in TEMPLATES:
        raw, blk = load_arm_template(results, base, runtag, stages, t, min_join, notes)
        raws[t] = raw
        blocks[t] = blk
        metrics[t] = template_metrics(raw, stages, probes) if raw is not None else None
    # smoke: the rephrased dump must not be a byte-copy of the original one
    if raws["orig"] is not None and raws["rephr"] is not None:
        both = sorted(set(raws["orig"]["ids"]) & set(raws["rephr"]["ids"]))
        if both:
            io = {i: j for j, i in enumerate(raws["orig"]["ids"])}
            ir = {i: j for j, i in enumerate(raws["rephr"]["ids"])}
            if all(raws["orig"]["g0"][io[i]] == raws["rephr"]["g0"][ir[i]] for i in both):
                notes.append(f"WARN_IDENTICAL_TEMPLATES {base}: pope_rephr_logits gaps identical to "
                             "pope_logits on every common id -- the rephrased pass may not have run")
    block = {"runtag": runtag, "probes": probes, "templates": {}, "trajectory": {}}
    for t in TEMPLATES:
        blk = blocks[t]
        if metrics[t] is not None:
            blk["base"] = metrics[t]["base"]
            blk["stages"] = metrics[t]["stages"]
            block["trajectory"][t] = arm_trajectories(metrics[t], stages, probes)
        else:
            block["trajectory"][t] = ABSENT
        block["templates"][t] = blk
    block["template_gap"] = template_gap(metrics["orig"], metrics["rephr"], stages)
    return block, raws


# ---------------------------------------------------------------------------
# (4) the decisive comparison
# ---------------------------------------------------------------------------
def _contrast(tname, arm_raw, ref_raw, arm_probes, ref_probes, stages, min_join, notes, arm_rt, ref_rt):
    """Arm-uncorrected vs ref-PCR-corrected sum|dc| on template `tname`, both on
    the arm-and-ref JOINT id set, per probe. Returns the per-probe dict."""
    if arm_raw is None or ref_raw is None:
        return {"status": ABSENT, "note": f"{tname}: base/stage dumps absent for "
                + ("arm" if arm_raw is None else "ref")}
    ids = sorted(set(arm_raw["ids"]) & set(ref_raw["ids"]))
    n_base = arm_raw["n_base"]
    if len(ids) < min_join * n_base:
        die(f"{tname}: joint arm/ref id set {len(ids)}/{n_base} (< {min_join:.0%})")
    ia = {i: j for j, i in enumerate(arm_raw["ids"])}
    ir = {i: j for j, i in enumerate(ref_raw["ids"])}
    yes = [arm_raw["yes"][ia[i]] for i in ids]
    g0 = [arm_raw["g0"][ia[i]] for i in ids]
    for i in ids[: min(len(ids), 2000)]:
        if arm_raw["g0"][ia[i]] != ref_raw["g0"][ir[i]]:
            die(f"{tname}: base gaps differ between the arm and ref loads for id {i!r} -- "
                "the two analyses must share one base dump")
    c0 = sdt(g0, yes)["c"]
    pick = lambda raw, k, idx: [raw["gk"][k][idx[i]] for i in ids]
    out = {"status": "OK", "n_ids_joint": len(ids), "n_ids_arm": len(arm_raw["ids"]),
           "n_ids_ref": len(ref_raw["ids"]), "c_base": c0, "per_probe": {}}
    for pname in PROBES:
        pb = ref_probes[pname]
        if not (isinstance(pb, dict) and pb.get("status") == "OK"):
            out["per_probe"][pname] = {"status": ABSENT, "note": f"ref probe {pname} absent"}
            continue
        used = [k for k in stages if k in arm_raw["gk"] and k in ref_raw["gk"]
                and probe_delta(pb, k) is not None]
        if not used:
            out["per_probe"][pname] = {"status": ABSENT, "note": "no stage with arm dump + ref dump + ref probe delta"}
            continue
        arm_unc = {k: sdt(pick(arm_raw, k, ia), yes)["c"] for k in used}
        ref_unc = {k: sdt(pick(ref_raw, k, ir), yes)["c"] for k in used}
        ref_pcr = {k: sdt(pick(ref_raw, k, ir), yes, delta=probe_delta(pb, k))["c"] for k in used}
        ap = arm_probes.get(pname)
        arm_pcr = {k: sdt(pick(arm_raw, k, ia), yes, delta=probe_delta(ap, k))["c"]
                   for k in used if probe_delta(ap, k) is not None}
        ta, tr = trajectory(arm_unc, c0, used), trajectory(ref_pcr, c0, used)
        tru = trajectory(ref_unc, c0, used)
        tap = trajectory(arm_pcr, c0, used) if arm_pcr else ABSENT
        a, r = ta["sum_abs_dc_full"], tr["sum_abs_dc_full"]
        complete = used == list(stages)
        res = {
            "status": "DECISIVE" if complete else "PARTIAL",
            "stages_used": used, "stages_missing": [k for k in stages if k not in used],
            "ref_delta": {k: probe_delta(pb, k) for k in used},
            "sum_abs_dc": {"arm_uncorrected": a, "ref_pcr": r,
                           "ref_uncorrected": tru["sum_abs_dc_full"],
                           "arm_pcr_own_probe": tap["sum_abs_dc_full"] if tap != ABSENT else ABSENT},
            "sum_abs_dc_post_settling_not_judged": {
                "arm_uncorrected": ta["sum_abs_dc_post_settling"], "ref_pcr": tr["sum_abs_dc_post_settling"],
                "ref_uncorrected": tru["sum_abs_dc_post_settling"]},
            "margin_ref_pcr_minus_arm": r - a,
            "ratio_arm_over_ref_pcr": (a / r) if r > 0 else None,
            "arm_lower": bool(a < r),
            "trajectories": {"arm_uncorrected": ta, "ref_pcr": tr, "ref_uncorrected": tru,
                             "arm_pcr_own_probe": tap},
        }
        if not complete:
            notes.append(f"PARTIAL {tname}/{pname}: decisive sum over {used} only (missing {res['stages_missing']})")
        out["per_probe"][pname] = res
    return out


def decisive(arm_raws, ref_raws, arm_block, ref_block, stages, primary, min_join, notes, arm_rt, ref_rt):
    out = {"decision_rule": DECISION_RULE, "primary_probe": primary,
           "rephr": _contrast("rephr", arm_raws["rephr"], ref_raws["rephr"], arm_block["probes"],
                              ref_block["probes"], stages, min_join, notes, arm_rt, ref_rt),
           "orig_template_context_not_judged":
               _contrast("orig", arm_raws["orig"], ref_raws["orig"], arm_block["probes"],
                         ref_block["probes"], stages, min_join, notes, arm_rt, ref_rt)}
    R = out["rephr"]
    pp = R.get("per_probe", {}).get(primary) if R.get("status") == "OK" else None
    if not (isinstance(pp, dict) and pp.get("status") in ("DECISIVE", "PARTIAL")):
        out.update({"status": ABSENT, "arm_is_method": None,
                    "note": f"rephrased-template comparison with probe {primary!r} unavailable: "
                            + (R.get("note") or (pp or {}).get("note") or "see per_probe")})
        return out
    avail = {p: v for p, v in R["per_probe"].items() if isinstance(v, dict) and "arm_lower" in v}
    out.update({
        "status": pp["status"], "stages_used": pp["stages_used"], "n_ids_joint": R["n_ids_joint"],
        "sum_abs_dc_rephr_arm_uncorrected": pp["sum_abs_dc"]["arm_uncorrected"],
        "sum_abs_dc_rephr_ref_pcr": pp["sum_abs_dc"]["ref_pcr"],
        "sum_abs_dc_rephr_ref_uncorrected": pp["sum_abs_dc"]["ref_uncorrected"],
        "sum_abs_dc_rephr_arm_pcr_own_probe": pp["sum_abs_dc"]["arm_pcr_own_probe"],
        "margin_ref_pcr_minus_arm": pp["margin_ref_pcr_minus_arm"],
        "arm_is_method": pp["arm_lower"],
        "arm_is_method_all_available_probes": all(v["arm_lower"] for v in avail.values()),
        "probes_available": sorted(avail),
        "per_probe_arm_lower": {p: v["arm_lower"] for p, v in avail.items()},
    })
    return out


def verdict_line(res):
    D = res.get("decisive")
    rt, ref = res["runtag"], res.get("ref_runtag")
    if D is None:
        return f"{rt}: NO_REF -- no --ref_runtag given; arm-only diagnostics written, no method/PCR decision"
    if D["status"] == ABSENT:
        return f"{rt} vs {ref}: ABSENT -- {D['note']}"
    a, r = D["sum_abs_dc_rephr_arm_uncorrected"], D["sum_abs_dc_rephr_ref_pcr"]
    tag = "METHOD on this cell" if D["arm_is_method"] else "TRAIN-TIME PCR (diagnostic, not a contribution)"
    part = "" if D["status"] == "DECISIVE" else f" [PARTIAL: stages {D['stages_used']} only]"
    return (f"{rt} vs {ref}: {tag} -- sum|dc_rephr| arm(uncorrected)={a:.3f} "
            f"{'<' if D['arm_is_method'] else '>='} ref(PCR, probe={D['primary_probe']})={r:.3f} "
            f"(ref uncorrected={D['sum_abs_dc_rephr_ref_uncorrected']:.3f}; n={D['n_ids_joint']}){part}")


# ---------------------------------------------------------------------------
def fmt(v, spec="+.3f"):
    return format(v, spec) if isinstance(v, (int, float)) and not isinstance(v, bool) else ABSENT


def print_summary(res, primary):
    for who in ("arm", "ref"):
        B = res.get(who)
        if not B:
            continue
        T = B["templates"]
        print(f"{TAG} {who} {B['runtag']}: orig {T['orig']['status']} rephr {T['rephr']['status']} | probes "
              + ", ".join(f"{p}={B['probes'][p]['status']}" for p in PROBES))
        if T["orig"]["status"] != "OK" and T["rephr"]["status"] != "OK":
            continue
        print(f"  {'stage':6s} {'c_orig':>8s} {'c_orig+PCR':>10s} {'c_rephr':>8s} {'c_rephr+PCR':>11s} "
              f"{'F1_rephr':>8s} {'d_rephr':>7s} {'delta':>7s} {'gap-gap0':>8s}")
        bo = T["orig"].get("base"), T["rephr"].get("base")
        print(f"  {'base':6s} {fmt(bo[0]['c'] if bo[0] else None):>8s} {'-':>10s} "
              f"{fmt(bo[1]['c'] if bo[1] else None):>8s} {'-':>11s} "
              f"{fmt(bo[1]['f1'] if bo[1] else None, '.4f'):>8s} {fmt(bo[1]['dprime'] if bo[1] else None, '.3f'):>7s} "
              f"{'-':>7s} {'-':>8s}")
        for k in res["stages"]:
            so = T["orig"].get("stages", {}).get(k, ABSENT)
            sr = T["rephr"].get("stages", {}).get(k, ABSENT)
            d = probe_delta(B["probes"][primary], k)
            gp = B["template_gap"].get("stages", {}).get(k, ABSENT)
            row = [fmt(so["uncorrected"]["c"]) if isinstance(so, dict) else ABSENT,
                   fmt(so["pcr"][primary]["c"]) if isinstance(so, dict) and isinstance(so["pcr"][primary], dict) else ABSENT,
                   fmt(sr["uncorrected"]["c"]) if isinstance(sr, dict) else ABSENT,
                   fmt(sr["pcr"][primary]["c"]) if isinstance(sr, dict) and isinstance(sr["pcr"][primary], dict) else ABSENT,
                   fmt(sr["uncorrected"]["f1"], ".4f") if isinstance(sr, dict) else ABSENT,
                   fmt(sr["uncorrected"]["dprime"], ".3f") if isinstance(sr, dict) else ABSENT,
                   fmt(d), fmt(gp["gap_minus_base"]) if isinstance(gp, dict) else ABSENT]
            print(f"  {k:6s} {row[0]:>8s} {row[1]:>10s} {row[2]:>8s} {row[3]:>11s} {row[4]:>8s} {row[5]:>7s} "
                  f"{row[6]:>7s} {row[7]:>8s}")
        tr = B["trajectory"]["rephr"]
        if tr != ABSENT:
            pcr = tr["pcr"].get(primary)
            print(f"  sum|dc_rephr| uncorrected={fmt(tr['uncorrected']['sum_abs_dc_full'], '.3f')} "
                  f"PCR({primary})={fmt(pcr['sum_abs_dc_full'], '.3f') if isinstance(pcr, dict) else ABSENT} "
                  f"| sum|dc_orig| uncorrected="
                  f"{fmt(B['trajectory']['orig']['uncorrected']['sum_abs_dc_full'], '.3f') if B['trajectory']['orig'] != ABSENT else ABSENT}"
                  f" | max|gap shift|={fmt(B['template_gap'].get('max_abs_gap_shift'), '.3f')}")
    for n in res["notes"]:
        print(f"{TAG} NOTE {n}")
    print(f"{TAG} VERDICT {res['verdict']}")


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results_fs directory")
    ap.add_argument("--runtag", required=True, help="arm runtag, e.g. mth_critp_o1_s17")
    ap.add_argument("--base", required=True, help="base label, e.g. llava15_base")
    ap.add_argument("--stages", default=None, help="comma list (k1,k2,...); default: discovered <RT>_k<N> dirs")
    ap.add_argument("--ref_runtag", default=None, help="same-path SEQ reference, e.g. psL_seq_o1_s17")
    ap.add_argument("--primary_probe", default="coco", choices=sorted(PROBES))
    ap.add_argument("--min_join_frac", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    notes = []
    if a.stages:
        stages, discovered = [s.strip() for s in a.stages.split(",") if s.strip()], None
    else:
        stages, discovered = discover_stages(a.results, [a.runtag] + ([a.ref_runtag] if a.ref_runtag else []))
        if not discovered:
            notes.append(f"no <runtag>_k<N> dirs found under {a.results}; defaulting to k1..k6 (all ABSENT)")
    res = {"runtag": a.runtag, "ref_runtag": a.ref_runtag, "base": a.base, "results": os.path.abspath(a.results),
           "stages": stages, "stages_discovered": discovered, "primary_probe": a.primary_probe,
           "min_join_frac": a.min_join_frac, "decision_rule": DECISION_RULE, "notes": notes}
    arm_block, arm_raws = analyse_arm(a.results, a.base, a.runtag, stages, a.min_join_frac, notes)
    res["arm"] = arm_block
    if a.ref_runtag:
        ref_block, ref_raws = analyse_arm(a.results, a.base, a.ref_runtag, stages, a.min_join_frac, notes)
        res["ref"] = ref_block
        res["decisive"] = decisive(arm_raws, ref_raws, arm_block, ref_block, stages, a.primary_probe,
                                   a.min_join_frac, notes, a.runtag, a.ref_runtag)
    else:
        res["ref"] = None
        res["decisive"] = None
    res["verdict"] = verdict_line(res)
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(a.out + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(a.out + ".tmp", a.out)
    if not a.quiet:
        print_summary(res, a.primary_probe)
        print(f"{TAG} wrote {a.out}")
    return res


if __name__ == "__main__":
    run()
