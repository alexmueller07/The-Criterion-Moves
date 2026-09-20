#!/usr/bin/env python3
"""pcr_transfer.py -- is a train-time criterion method a METHOD, or train-time
post-hoc correction? (design_notes/review_method_design.md, threat M2;
fullstudy/FULLSTUDY_PREREG.md, 2026-09-08 "design red-team consequences" (b);
PCR-0 added 2026-09-12 to discharge the OWED item logged in the prereg's
2026-09-11 "balanced-probe target is an APPROXIMATION" entry)

The threat. critp / anchorcrit / CNP regularize mean g = z_yes - z_no on a
label-free probe that is the POPE endpoint distribution minus its answer key
(COCO images x 80 COCO classes x the POPE question template). "The criterion did
not move" on that endpoint is the objective restated, and the zero-training
label-free post-hoc scalar -- PCR: delta_k = mean g_probe(base) - mean g_probe(k),
decide gap + delta_k > 0 -- already achieves it for free on the original template.
The only axis on which a TRAINED pin can beat the free scalar is template
transfer: a REPHRASED POPE template (same ids / images / GT, question rewritten)
that neither the probe nor PCR ever saw.

TWO COMPETITORS, BOTH REPORTED. PCR as defined above aims the corrected model at
the FROZEN BASE's operating point. The base sits at c = +0.431 and that is itself
badly mis-placed (the JOINT arm's endpoint, the best-placed thing we have
measured, sits at c = +0.088). So PCR inherits exactly the mis-placement the
trained methods exist to remove, and a method that beats it has beaten a
handicapped opponent. PCR-0 removes the handicap:

    PCR    delta_k        = mean g_probe(base) - mean g_probe(k)
    PCR-0  delta_k^(0)    = T*              - mean g_probe(k)
                          = delta_k + s*,   s* = T* - mean g_probe(base)

i.e. the SAME estimator with a different target constant. T* is the probe value
whose corrected operating point lands on the target criterion c* instead of on
the base's. s* is found model-free, by sweeping the decision threshold over the
BASE's own empirical POPE gaps until the criterion equals c* -- no Gaussian, no
equal-variance assumption, one constant, no training, still a single scalar added
to the logits at inference. It is therefore the strongest honest form of the
"you did not need a method" objection, and it is what the paper's method-vs-
post-hoc claim must be quoted against.

WHERE c* COMES FROM, AND WHY NOT ZERO. c* is the JOINT arm's EMPIRICAL endpoint
criterion read out of analysis/readout/fs_aggregate.json (+0.088, n = 2 cells),
not a modelled c = 0. POPE is balanced 4500/4500, so c = 0 is the optimum only
under EQUAL VARIANCE, and this project's own z-ROC slopes are 0.55-0.72 (prereg,
2026-09-11) -- equal variance is an assumption we have already falsified on this
data, so a modelled zero would put the competitor on a target we know is wrong.
The script refuses to run PCR-0 against a target it cannot certify.

WHAT THIS SCRIPT WOULD CONCLUDE, AND WHAT WOULD FALSIFY IT.
  It concludes the arm is a METHOD on this cell only if its UNcorrected criterion
  path on the rephrased template is shorter than the path of the reference
  corrected by the free scalar -- separately against PCR (base-aimed) and against
  PCR-0 (target-aimed) -- AND, for PCR-0, only if the arm does not end up further
  from c* than the corrected reference does. Two things falsify a method claim
  here, and either one alone is enough: a shorter path for the PCR-0-corrected
  reference (the free scalar already does the job), or an equal/shorter path for
  the arm bought by parking the criterion further from c* than the competitor
  (drift traded for mis-placement, which is the failure already documented for
  the anchor and for the projector freeze). A win against PCR alone is NOT a
  method claim, and this script will say so in its verdict line.

Decision rule (pre-registered, per matched cell = same suite / order / seed):
    the arm is a METHOD on this cell  iff
    sum_k |dc_rephr(k)| (arm, UNcorrected)
        <  sum_k |dc_rephr(k)| (SEQ reference, PCR-corrected with the reference's
                                 OWN probe delta_k, transferred from the original
                                 template to the rephrased one)
    otherwise it is reported as train-time PCR: a diagnostic, not a contribution.
The prereg requires the boolean to be True in 3/3 seeds; this script decides ONE
cell (it emits the boolean and the two numbers); aggregation across seeds is the
scorecard's job. The PCR-0 contrast and the placement contrast are reported
ALONGSIDE that pre-registered boolean, never in place of it.

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
     yes-rate, accuracy) uncorrected vs corrected vs base, for each correction
     variant:
         pcr    delta_k                (aims at the base: the original competitor)
         pcr0   delta_k + s*_orig      (aims at c*; s* fitted on the base's
                                        ORIGINAL dump and TRANSFERRED, so the
                                        competitor never sees the rephrased
                                        template, exactly like PCR)
         pcr0t  delta_k + s*_t         (s* refit on the base dump of the template
                                        being scored: the ceiling of the
                                        target-aimed scalar, reported as a
                                        sensitivity, never judged)
     Each variant's trajectory starts from its OWN corrected stage 0, because a
     post-hoc scalar is applied at inference to every checkpoint including the
     base. For pcr the offset is 0 and the start point is the uncorrected base,
     so nothing about the pre-registered comparison changes; for pcr0 the start
     point is on the target. Referencing a target-aimed correction to the
     un-re-aimed base would charge it |c* - c_base| for doing the one thing it
     exists to do -- the mirror image of the rigging PCR-0 was added to remove.
 (3) the arm's own criterion on the rephrased template, c_rephr(k), and
     sum_k |dc_rephr| across stages with base = stage 0 (full sum; the
     post-settling sum, stages 2..T, is reported alongside and never judged --
     design review M1).
 (4) with --ref_runtag: the DECISIVE comparison above, per probe, against BOTH
     competitors; the boolean `arm_is_method` uses --primary_probe (default coco
     = the probe critp trains on, rendered with the anchor's / POPE's own
     template) and the PCR competitor, unchanged. `arm_survives_pcr0` is the new
     verdict: path AND placement against the target-aimed competitor. Both
     trajectories are recomputed on the arm-and-reference JOINT id set. The same
     contrast on the original template is reported as context (there PCR wins by
     construction).
 (5) template-transfer gap: c_orig(k) - c_rephr(k) per stage, and the same minus
     the base's own gap (a method that only pins the trained template shows a
     large shift of this gap; a genuine prior fix does not).
 (6) placement against c*: |c(k) - c*| per stage, at the endpoint and averaged
     over the stages used, for the arm and for each corrected reference.
 (7) a paired ITEM bootstrap over the joint POPE ids for the two margins and for
     the endpoint placement gap. The resample is a LIST with multiplicity; see
     bootstrap_margins() for why that sentence is in the source.

--placement_audit is a second, dumps-free mode: it reads only the certified
aggregate and reports, per arm and per cell, how far each arm's ENDPOINT
criterion sits from c* -- the axis on which a properly-aimed post-hoc scalar is
correct by construction. It exists so the placement half of the comparison can be
quoted from data already on disk while the per-item logit dumps for the method
arms are still on the cluster.

SDT primitives mirror analysis/logit_bias_analysis.py sdt() exactly (rates
clipped to [1/(2n), 1-1/(2n)], NormalDist().inv_cdf, c = -(zH+zFA)/2,
d' = zH - zFA, F1 with the same guards); ids are aligned across dumps by exact
`id`; dedup on id (last occurrence wins, counted). ABSENT cells are reported,
never imputed; a sum over zero transitions is ABSENT, never 0.0. Malformed rows
(non-finite gap, POPE gt not in {yes,no}, gt disagreeing across dumps for one
id) abort loudly. The base dump is CERTIFIED against fs_aggregate's recorded base
criterion before any correction is fitted from it: a mismatch on a real results
tree aborts, because a scalar fitted on a dump that is not the dump the paper's
numbers came from is a number about nothing. Stdlib only (Python 3.9+).

usage:
  pcr_transfer.py --results <results_fs> --runtag <RT> --base <label>
                  [--stages k1,k2,...] [--ref_runtag <SEQ RT>]
                  [--primary_probe coco|oi|noise] [--min_join_frac 0.5]
                  [--aggregate <fs_aggregate.json>] [--target_c <float>]
                  [--n_boot 1000] [--boot_seed 20260912] --out <json>
  pcr_transfer.py --placement_audit [--aggregate <fs_aggregate.json>] [--out <json>]
"""
import argparse
import json
import math
import os
import random
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

# correction variants; "pcr" is the original, base-aimed competitor and its JSON
# shape is unchanged. Order matters only for printing.
VARIANTS = ("pcr", "pcr0", "pcr0t")

# The certified target. c* is recomputed from the aggregate on every run and
# checked against this constant; the constant is the number the prereg records
# (2026-09-11, "reference the target to the JOINT arm's empirical endpoint
# criterion (+0.088)"). If the aggregate ever stops reproducing it the run dies
# rather than silently re-aiming the competitor.
PREREG_JOINT_ENDPOINT_C = 0.088
TARGET_DRIFT_TOL = 0.010          # aggregate vs prereg constant
BASE_CERT_TOL = 0.010             # base dump vs aggregate's recorded base c
DEFAULT_AGGREGATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "readout", "fs_aggregate.json")
SYNTH_MARKER = "THIS_IS_SYNTHETIC_FIXTURE_DATA.txt"

DECISION_RULE = ("arm is a METHOD on this cell iff sum|dc_rephr|(arm, uncorrected) < "
                 "sum|dc_rephr|(SEQ ref, PCR-corrected with the ref's own probe delta_k); "
                 "otherwise train-time PCR (diagnostic). delta_k = mean g_probe(base) - "
                 "mean g_probe(k), fitted on the probe only; decision gap + delta_k > 0.")

PCR0_RULE = ("PCR-0 is the same scalar aimed at the certified target c* (the JOINT arm's "
             "empirical endpoint criterion) instead of at the frozen base: delta_k + s*, "
             "s* the model-free offset that moves the BASE's own POPE criterion to c*. The "
             "arm SURVIVES PCR-0 only if it has BOTH the shorter rephrased-template path AND "
             "an endpoint no further from c* than the PCR-0-corrected reference.")


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


def c_from_counts(hits, nyes, fas, nno):
    """c = -(zH + zFA)/2 from counts, same clipping as sdt(). Used by the offset
    sweep and by the bootstrap, where building a full sdt dict per replicate
    would dominate the runtime."""
    Hc, _ = clipped_rate(hits, nyes)
    FAc, _ = clipped_rate(fas, nno)
    return -0.5 * (_z(Hc) + _z(FAc))


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
# the target-aimed scalar: model-free threshold sweep
# ---------------------------------------------------------------------------
def _operating_points(gaps, yes):
    """Every distinct operating point of the rule (g > t), from t above the
    largest gap (nothing yes) down to t below the smallest (everything yes).
    Yields (t, hits, fas) with t placed at the MIDPOINT of the open interval
    that produces the split, so a returned threshold never sits on a datum and
    ties cannot decide the answer. No distributional assumption: this is the
    empirical ROC of the dump, which is the whole point (analysis/
    threshold_sweep.py makes the same argument for the ceiling)."""
    n = len(gaps)
    ny = sum(1 for y in yes if y)
    nn = n - ny
    if ny == 0 or nn == 0:
        die(f"offset sweep: degenerate split (n_yes={ny}, n_no={nn})")
    order = sorted(range(n), key=lambda i: -gaps[i])
    gmax, gmin = gaps[order[0]], gaps[order[-1]]
    span = max(1.0, gmax - gmin)
    pts = [(gmax + 0.5 * span, 0, 0)]
    hits = fas = 0
    i = 0
    while i < n:
        g = gaps[order[i]]
        j = i
        while j < n and gaps[order[j]] == g:
            if yes[order[j]]:
                hits += 1
            else:
                fas += 1
            j += 1
        t = 0.5 * (g + gaps[order[j]]) if j < n else gmin - 0.5 * span
        pts.append((t, hits, fas))
        i = j
    return ny, nn, pts


def offset_for_target(gaps, yes, target_c=None, target_yes_rate=None):
    """The additive offset s whose corrected decision (g + s > 0) lands closest
    to the target. g + s > 0 <=> g > -s, so s = -t over the sweep above.
    Exactly one of target_c / target_yes_rate. Ties broken toward the smaller
    |s|, i.e. toward the smaller intervention."""
    if (target_c is None) == (target_yes_rate is None):
        die("offset_for_target: give exactly one of target_c / target_yes_rate")
    ny, nn, pts = _operating_points(gaps, yes)
    best = None
    for t, h, f in pts:
        if target_c is not None:
            val = c_from_counts(h, ny, f, nn)
            err = abs(val - target_c)
        else:
            val = (h + f) / (ny + nn)
            err = abs(val - target_yes_rate)
        s = -t
        key = (err, abs(s))
        if best is None or key < best[0]:
            best = (key, s, val, h, f)
    _, s, val, h, f = best
    tgt = target_c if target_c is not None else target_yes_rate
    return {"offset": s, "achieved": val, "target": tgt, "residual": val - tgt,
            "n": ny + nn, "n_yes": ny, "n_no": nn, "hits": h, "fas": f,
            "matched": "criterion" if target_c is not None else "yes_rate"}


# ---------------------------------------------------------------------------
# the certified target
# ---------------------------------------------------------------------------
def load_aggregate(path, required, notes):
    if not os.path.isfile(path):
        if required:
            die(f"aggregate readout {path} not found -- PCR-0 needs a CERTIFIED target "
                "(the JOINT arm's empirical endpoint criterion). Refusing to fall back to a "
                "modelled c = 0: our own z-ROC slopes are 0.55-0.72, so equal variance -- the "
                "only condition under which 0 is optimal -- is already falsified on this data. "
                "Pass --aggregate or --target_c explicitly.")
        notes.append(f"NO_AGGREGATE {path} absent; PCR-0 unavailable")
        return None
    with open(path) as f:
        return json.load(f)


def _cells(agg, backbone, arm):
    """[(cell_key, [(k, pope_block), ...])] for one arm, stages numerically
    sorted. Per cell, never pooled here -- aggregation is the caller's job and
    must happen after the per-cell quantity is formed."""
    arms = agg.get("backbones", {}).get(backbone, {}).get("arms", {})
    out = []
    for key in sorted(arms):
        if key.split("|")[0] != arm:
            continue
        cell = arms[key]
        stages = []
        for s in sorted((x for x in cell if x.isdigit()), key=int):
            p = cell[s].get("pope")
            if p is not None:
                stages.append((int(s), p))
        if stages:
            out.append((key, stages))
    return out


def derive_target(agg, backbone, target_c_override, notes):
    """c* and its provenance. Per JOINT cell first, then the mean over cells --
    never a stage-pooled average (estimator discipline: the per-cell endpoint is
    the quantity; the cross-cell mean is a summary of those, and the two are not
    interchangeable)."""
    block = {"prereg_constant": PREREG_JOINT_ENDPOINT_C, "backbone": backbone}
    if target_c_override is not None:
        block.update({"c_star": float(target_c_override), "provenance": "--target_c (manual)",
                      "target_yes_rate": None, "joint_cells": [], "n_joint_cells": 0})
        notes.append(f"TARGET_MANUAL c* = {target_c_override:+.4f} supplied on the command line; "
                     "the certified JOINT-endpoint derivation was NOT used")
        return block
    if agg is None:
        block.update({"status": ABSENT, "c_star": None, "target_yes_rate": None,
                      "note": "no aggregate -> no certified target -> PCR-0 unavailable"})
        return block
    cells = _cells(agg, backbone, "joint")
    if not cells:
        die(f"aggregate has no JOINT cells for backbone {backbone!r} -- cannot certify c*. "
            "PCR-0 will not be fitted against an uncertified target.")
    per_cell = []
    for key, stages in cells:
        k, p = stages[-1]
        per_cell.append({"cell": key, "endpoint_k": k, "c": p["c"],
                         "yes_rate": p.get("yes_rate"), "f1": p.get("f1"),
                         "n_parsed": p.get("n_parsed")})
    c_star = mean(x["c"] for x in per_cell)
    yr = [x["yes_rate"] for x in per_cell if isinstance(x["yes_rate"], (int, float))]
    drift = abs(c_star - PREREG_JOINT_ENDPOINT_C)
    if drift > TARGET_DRIFT_TOL:
        die(f"the aggregate's JOINT endpoint criterion is {c_star:+.4f}, "
            f"{drift:.4f} away from the pre-registered {PREREG_JOINT_ENDPOINT_C:+.4f} "
            f"(tol {TARGET_DRIFT_TOL}). The competitor's target moved. Update "
            "PREREG_JOINT_ENDPOINT_C deliberately, with a prereg entry, or fix the readout.")
    block.update({
        "status": "OK", "c_star": c_star, "provenance": "JOINT arm endpoint criterion, "
        f"mean over {len(per_cell)} cell(s) of {os.path.basename(DEFAULT_AGGREGATE)}",
        "target_yes_rate": mean(yr) if yr else None,
        "joint_cells": per_cell, "n_joint_cells": len(per_cell),
        "drift_from_prereg": c_star - PREREG_JOINT_ENDPOINT_C,
    })
    if len(per_cell) < 4:
        detail = ", ".join("%s=%+.4f" % (x["cell"], x["c"]) for x in per_cell)
        notes.append(f"TARGET_N_SMALL c* is the mean of {len(per_cell)} JOINT cell(s) "
                     f"({detail}) -- quote the per-cell values, not an interval")
    return block


def certified_base_c(agg, backbone):
    try:
        return agg["backbones"][backbone]["base"]["pope"]["c"]
    except (TypeError, KeyError):
        return None


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


def is_synthetic_tree(results):
    return os.path.isfile(os.path.join(results, SYNTH_MARKER))


# ---------------------------------------------------------------------------
# fitting s*: the target-aimed offset, on the BASE dump only
# ---------------------------------------------------------------------------
def fit_offsets(results, base, target, agg, backbone, notes):
    """s* per template, fitted on the BASE's own POPE dump. Returns
    (offsets, block) where offsets[variant][template] is a float or None.

      pcr    0.0 everywhere            -- the original, base-aimed competitor
      pcr0   s*_orig on BOTH templates -- fitted on the ORIGINAL template and
             transferred, so the target-aimed competitor is held to the same
             transfer discipline as PCR: it never sees the rephrased template.
      pcr0t  s*_t per template         -- refit on that template's base dump.
             This is the CEILING of the target-aimed scalar and is reported as a
             sensitivity, never judged, because fitting on the rephrased base
             dump leaks the held-out template to the competitor.

    Also certifies the base dump against the aggregate's recorded base criterion.
    On a real results tree a mismatch is fatal: a scalar fitted on a dump that is
    not the dump the paper's numbers came from is a number about nothing. On a
    synthetic fixture tree (marker file present) it is a note."""
    block = {"rule": PCR0_RULE, "per_template": {}, "base_certification": {}}
    offsets = {"pcr": {t: 0.0 for t in TEMPLATES},
               "pcr0": {t: None for t in TEMPLATES},
               "pcr0t": {t: None for t in TEMPLATES}}
    c_star = target.get("c_star")
    synth = is_synthetic_tree(results)
    block["synthetic_fixture_tree"] = synth
    base_c_obs = None
    for t, fname in TEMPLATES.items():
        rows, _ = load_dump(os.path.join(results, base, fname), labeled=True)
        if rows is None:
            block["per_template"][t] = {"status": ABSENT,
                                        "note": f"base dump {fname} absent -> no s* for {t}"}
            continue
        ids = sorted(rows)
        gaps = [rows[i]["gap"] for i in ids]
        yes = [rows[i]["gt"] == "yes" for i in ids]
        base_sdt = sdt(gaps, yes)
        if t == "orig":
            base_c_obs = base_sdt["c"]
        blk = {"status": "OK", "n_fit": len(ids), "base_c": base_sdt["c"],
               "base_yes_rate": base_sdt["yes_rate"]}
        if c_star is None:
            blk.update({"status": ABSENT, "note": "no certified c* -> s* not fitted"})
            block["per_template"][t] = blk
            continue
        fit_c = offset_for_target(gaps, yes, target_c=c_star)
        blk["s_star"] = fit_c
        tyr = target.get("target_yes_rate")
        if tyr is not None:
            fit_r = offset_for_target(gaps, yes, target_yes_rate=tyr)
            blk["s_star_label_free"] = fit_r
            blk["oracle_premium"] = fit_c["offset"] - fit_r["offset"]
        offsets["pcr0t"][t] = fit_c["offset"]
        block["per_template"][t] = blk
    s_orig = offsets["pcr0t"].get("orig")
    if s_orig is not None:
        for t in TEMPLATES:
            offsets["pcr0"][t] = s_orig
        block["s_star_transferred_from"] = "orig"
        block["s_star"] = s_orig
    else:
        block["s_star"] = None
        block["note"] = ("no ORIGINAL-template base dump -> the transferred PCR-0 offset "
                         "cannot be fitted; pcr0 is ABSENT (pcr0t may still exist)")
        notes.append("PCR0_ABSENT no original-template base dump to fit s* on")
    # --- certification against the readout the paper quotes -----------------
    cert = {"tol": BASE_CERT_TOL, "observed_base_c_orig": base_c_obs}
    ref_c = certified_base_c(agg, backbone) if agg is not None else None
    cert["certified_base_c"] = ref_c
    if ref_c is None or base_c_obs is None:
        cert["status"] = ABSENT
        cert["note"] = "no aggregate base criterion, or no original-template base dump"
    elif abs(base_c_obs - ref_c) <= BASE_CERT_TOL:
        cert["status"] = "CERTIFIED"
    else:
        cert["status"] = "MISMATCH"
        msg = (f"base dump criterion {base_c_obs:+.4f} != aggregate's certified base "
               f"{ref_c:+.4f} (|diff| {abs(base_c_obs - ref_c):.4f} > {BASE_CERT_TOL})")
        if synth:
            cert["note"] = msg + " -- synthetic fixture tree, not fatal"
            notes.append("SYNTHETIC_TREE " + msg)
        else:
            die(msg + ". The dump the scalar would be fitted on is not the dump the "
                      "paper's numbers came from. Refusing to fit a correction on it.")
    block["base_certification"] = cert
    return offsets, block


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


def variant_delta(probe_block, k, offsets, variant, tname):
    """delta for one correction variant = probe delta + the variant's offset.
    None when either half is missing -- never 0.0 by default, which would be a
    silent uncorrected series wearing a corrected label."""
    d = probe_delta(probe_block, k)
    if d is None:
        return None
    off = offsets.get(variant, {}).get(tname)
    if off is None:
        return None
    return d + off


def template_metrics(raw, stages, probes, offsets, tname):
    """(2): base / per-stage uncorrected / per-probe corrected sdt blocks, one
    block per correction variant."""
    # A post-hoc scalar is applied at inference to EVERY checkpoint, stage 0
    # included, so each variant's trajectory starts from its OWN corrected base.
    # For pcr the offset is 0 and delta_0 = mean g_probe(base) - mean g_probe(base)
    # = 0, so its start point is the uncorrected base and nothing changes. For
    # pcr0 the start point is base + s*, i.e. on the target. Referencing a
    # target-aimed correction to the un-re-aimed base would charge it a first
    # transition of |c* - c_base| for doing exactly the thing it exists to do --
    # the mirror image of the rigging this variant was added to remove.
    out = {"base": sdt(raw["g0"], raw["yes"]), "base_by_variant": {}, "stages": {}}
    for v in VARIANTS:
        off = offsets.get(v, {}).get(tname)
        out["base_by_variant"][v] = (ABSENT if off is None
                                     else {"offset": off, **sdt(raw["g0"], raw["yes"], delta=off)})
    for k in stages:
        if k not in raw["gk"]:
            out["stages"][k] = ABSENT
            continue
        gk = raw["gk"][k]
        st = {"uncorrected": sdt(gk, raw["yes"])}
        for v in VARIANTS:
            st[v] = {}
            for pname, pb in probes.items():
                d = variant_delta(pb, k, offsets, v, tname)
                if d is None:
                    st[v][pname] = ABSENT
                else:
                    st[v][pname] = {"delta": d, "probe_delta": probe_delta(pb, k),
                                    "offset": offsets[v][tname], **sdt(gk, raw["yes"], delta=d)}
        out["stages"][k] = st
    return out


def trajectory(c_by_stage, c0, stages, c_star=None):
    """Chain of consecutive PRESENT stages, base = stage 0. Sums over zero
    transitions are ABSENT (never a silent 0.0). post_settling drops the first
    transition (base -> first present stage), per design review M1. When c_star
    is given, the placement block is filled in as well: drift and placement are
    the two axes the prereg requires to be reported in the same table."""
    seq = [("base", c0)] + [(k, c_by_stage[k]) for k in stages if c_by_stage.get(k) is not None]
    trans = [{"from": a, "to": b, "dc": cb - ca} for (a, ca), (b, cb) in zip(seq, seq[1:])]
    used = [k for k, _ in seq[1:]]
    full = sum(abs(t["dc"]) for t in trans) if trans else None
    post = sum(abs(t["dc"]) for t in trans[1:]) if len(trans) > 1 else None
    dev = {k: abs(c_by_stage[k] - c0) for k in used}
    out = {
        "c": {"base": c0, **{k: (c_by_stage[k] if c_by_stage.get(k) is not None else ABSENT) for k in stages}},
        "transitions": trans, "n_transitions": len(trans),
        "sum_abs_dc_full": ABSENT if full is None else full,
        "sum_abs_dc_post_settling": ABSENT if post is None else post,
        "abs_dev_from_base": dev,
        "max_abs_dev_from_base": max(dev.values()) if dev else ABSENT,
        "stages_used": used, "stages_absent": [k for k in stages if k not in used],
        "complete": len(used) == len(stages),
    }
    if c_star is not None:
        if used:
            devs = [abs(c_by_stage[k] - c_star) for k in used]
            out["placement"] = {
                "c_star": c_star, "endpoint_stage": used[-1],
                "endpoint_c": c_by_stage[used[-1]],
                "abs_dev_endpoint": abs(c_by_stage[used[-1]] - c_star),
                "mean_abs_dev": mean(devs), "max_abs_dev": max(devs),
                "n_stages": len(used),
            }
        else:
            out["placement"] = {"status": ABSENT, "c_star": c_star,
                                "note": "no present stage -> no placement"}
    return out


def arm_trajectories(metrics, stages, probes, c_star=None):
    """(3): uncorrected + per-variant per-probe corrected trajectories."""
    c0 = metrics["base"]["c"]
    unc = {k: (metrics["stages"][k]["uncorrected"]["c"] if isinstance(metrics["stages"][k], dict) else None)
           for k in stages}
    out = {"uncorrected": trajectory(unc, c0, stages, c_star)}
    for v in VARIANTS:
        out[v] = {}
        bv = metrics.get("base_by_variant", {}).get(v)
        c0v = bv["c"] if isinstance(bv, dict) else c0
        for pname in probes:
            cp = {}
            for k in stages:
                st = metrics["stages"][k]
                cp[k] = (st[v][pname]["c"]
                         if isinstance(st, dict) and isinstance(st[v][pname], dict) else None)
            out[v][pname] = (trajectory(cp, c0v, stages, c_star)
                             if any(x is not None for x in cp.values()) else ABSENT)
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


def analyse_arm(results, base, runtag, stages, min_join, notes, offsets, c_star):
    """Everything single-arm: (1) probe deltas, (2) per-template metrics, (3)
    trajectories, (5) template gap. Returns (json_block, raw_by_template)."""
    probes = {p: load_arm_probe(results, base, runtag, stages, p, min_join, notes) for p in PROBES}
    raws, metrics, blocks = {}, {}, {}
    for t in TEMPLATES:
        raw, blk = load_arm_template(results, base, runtag, stages, t, min_join, notes)
        raws[t] = raw
        blocks[t] = blk
        metrics[t] = template_metrics(raw, stages, probes, offsets, t) if raw is not None else None
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
            block["trajectory"][t] = arm_trajectories(metrics[t], stages, probes, c_star)
        else:
            block["trajectory"][t] = ABSENT
        block["templates"][t] = blk
    block["template_gap"] = template_gap(metrics["orig"], metrics["rephr"], stages)
    return block, raws


# ---------------------------------------------------------------------------
# (7) paired item bootstrap
# ---------------------------------------------------------------------------
def bootstrap_margins(yes, decisions, c_star, n_boot, seed, notes):
    """Paired bootstrap over the joint POPE ids for the two margins and the
    endpoint placement gap.

    ESTIMATOR DISCIPLINE, stated in the source because this project has already
    paid for it: the resample is a LIST of indices WITH MULTIPLICITY. Wrapping a
    resample in set() turns it into a ~63.2% subsample of distinct items and
    narrows every interval -- that bug made nine intervals here about 24% too
    narrow. Nothing below may be passed through set(), and the drawn multiplicity
    is what the counts are built from.

    `decisions` = {series: {stage: [bool]}}, one boolean per joint id: "this
    checkpoint, under this correction, answers yes to this item". The key
    "__base__" is that series' own stage 0 (corrected by the same scalar, since a
    post-hoc scalar is applied at inference to the base too). The booleans are
    precomputed once, so a replicate only re-counts them; the corrections
    themselves are held fixed at their full-sample values, so the interval covers
    EVALUATION-SET noise and not the probe-estimation noise in delta_k. That is a
    stated limit of the interval, not an accident.
    """
    if not n_boot:
        return {"status": "OFF", "n_boot": 0,
                "note": "--n_boot 0: no interval computed (an absent interval, not a zero-width one)"}
    n = len(yes)
    yes_idx = [i for i, y in enumerate(yes) if y]
    no_idx = [i for i, y in enumerate(yes) if not y]
    if not yes_idx or not no_idx:
        return {"status": ABSENT, "note": "degenerate split -> no bootstrap"}
    # per (series, stage): the item indices that count as a hit and as a false alarm
    idx = {}
    for s, per_stage in decisions.items():
        idx[s] = {}
        for k, dec in per_stage.items():
            idx[s][k] = ([i for i in yes_idx if dec[i]], [i for i in no_idx if dec[i]])
    rnd = random.Random(seed)
    keys = sorted(decisions)
    reps = {kk: [] for kk in ("sum_pcr", "sum_pcr0", "sum_arm", "dev_arm", "dev_pcr0")}
    n_deg = 0
    stage_order = [k for k in decisions[keys[0]] if k != "__base__"]
    for _ in range(n_boot):
        res = [rnd.randrange(n) for _ in range(n)]   # LIST, with multiplicity. Never set().
        cnt = [0] * n
        for i in res:
            cnt[i] += 1
        ny = sum(cnt[i] for i in yes_idx)
        nn = n - ny
        if ny == 0 or nn == 0:
            n_deg += 1
            continue
        cs = {}
        for s in keys:
            cs[s] = {}
            for k, (hi, fi) in idx[s].items():
                cs[s][k] = c_from_counts(sum(cnt[i] for i in hi), ny,
                                         sum(cnt[i] for i in fi), nn)
        for s, tag in (("arm_uncorrected", "sum_arm"), ("ref_pcr", "sum_pcr"), ("ref_pcr0", "sum_pcr0")):
            if s not in cs:
                continue
            chain = [cs[s]["__base__"]] + [cs[s][k] for k in stage_order]
            reps[tag].append(sum(abs(b - a) for a, b in zip(chain, chain[1:])))
        if c_star is not None and stage_order:
            last = stage_order[-1]
            if "arm_uncorrected" in cs:
                reps["dev_arm"].append(abs(cs["arm_uncorrected"][last] - c_star))
            if "ref_pcr0" in cs:
                reps["dev_pcr0"].append(abs(cs["ref_pcr0"][last] - c_star))

    def ci(vals):
        if not vals:
            return None
        v = sorted(vals)
        lo = v[max(0, int(math.floor(0.025 * (len(v) - 1))))]
        hi = v[min(len(v) - 1, int(math.ceil(0.975 * (len(v) - 1))))]
        return [lo, hi]

    out = {"status": "OK", "n_boot": n_boot, "n_items": n, "seed": seed,
           "n_degenerate_resamples": n_deg,
           "resample_unit": "POPE item id, drawn as a LIST with multiplicity (never set())",
           "covers": "evaluation-set noise only; delta_k held at its full-sample value"}
    if reps["sum_arm"] and reps["sum_pcr"]:
        m = [p - a for a, p in zip(reps["sum_arm"], reps["sum_pcr"])]
        out["margin_ref_pcr_minus_arm"] = {"ci95": ci(m), "mean": mean(m),
                                           "frac_arm_lower": sum(1 for x in m if x > 0) / len(m)}
    if reps["sum_arm"] and reps["sum_pcr0"]:
        m = [p - a for a, p in zip(reps["sum_arm"], reps["sum_pcr0"])]
        out["margin_ref_pcr0_minus_arm"] = {"ci95": ci(m), "mean": mean(m),
                                            "frac_arm_lower": sum(1 for x in m if x > 0) / len(m)}
    if reps["dev_arm"] and reps["dev_pcr0"]:
        m = [a - p for a, p in zip(reps["dev_arm"], reps["dev_pcr0"])]
        out["endpoint_placement_arm_minus_pcr0"] = {
            "ci95": ci(m), "mean": mean(m),
            "frac_arm_closer": sum(1 for x in m if x < 0) / len(m)}
    return out


# ---------------------------------------------------------------------------
# (4) the decisive comparison
# ---------------------------------------------------------------------------
def _contrast(tname, arm_raw, ref_raw, arm_probes, ref_probes, stages, min_join, notes,
              arm_rt, ref_rt, offsets, c_star, primary, n_boot, boot_seed):
    """Arm-uncorrected vs corrected-reference sum|dc| on template `tname`, both on
    the arm-and-ref JOINT id set, per probe and per correction variant. Returns
    the per-probe dict."""
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
           "n_ids_ref": len(ref_raw["ids"]), "c_base": c0, "c_star": c_star, "per_probe": {}}
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
        arm_gaps = {k: pick(arm_raw, k, ia) for k in used}
        ref_gaps = {k: pick(ref_raw, k, ir) for k in used}
        arm_unc = {k: sdt(arm_gaps[k], yes)["c"] for k in used}
        ref_unc = {k: sdt(ref_gaps[k], yes)["c"] for k in used}
        ta = trajectory(arm_unc, c0, used, c_star)
        tru = trajectory(ref_unc, c0, used, c_star)
        corrected, traj, c0_var = {}, {}, {}
        for v in VARIANTS:
            dv = {k: variant_delta(pb, k, offsets, v, tname) for k in used}
            if any(d is None for d in dv.values()):
                corrected[v], traj[v], c0_var[v] = None, ABSENT, None
                continue
            corrected[v] = dv
            # each variant's trajectory starts from its own corrected stage 0 --
            # see template_metrics() for why referencing it to the un-re-aimed
            # base would be the mirror image of the original rigging.
            c0_var[v] = sdt(g0, yes, delta=offsets[v][tname])["c"]
            traj[v] = trajectory({k: sdt(ref_gaps[k], yes, delta=dv[k])["c"] for k in used},
                                 c0_var[v], used, c_star)
        ap = arm_probes.get(pname)
        arm_pcr = {k: sdt(arm_gaps[k], yes, delta=probe_delta(ap, k))["c"]
                   for k in used if probe_delta(ap, k) is not None}
        tap = trajectory(arm_pcr, c0, used, c_star) if arm_pcr else ABSENT
        tr = traj["pcr"]
        t0 = traj["pcr0"]
        a = ta["sum_abs_dc_full"]
        r = tr["sum_abs_dc_full"] if tr != ABSENT else None
        r0 = t0["sum_abs_dc_full"] if t0 != ABSENT else None
        complete = used == list(stages)
        placement = {"c_star": c_star, "endpoint_stage": used[-1],
                     "abs_dev_endpoint": {
                         "arm_uncorrected": ta["placement"]["abs_dev_endpoint"],
                         "ref_uncorrected": tru["placement"]["abs_dev_endpoint"],
                         "ref_pcr": tr["placement"]["abs_dev_endpoint"] if tr != ABSENT else ABSENT,
                         "ref_pcr0": t0["placement"]["abs_dev_endpoint"] if t0 != ABSENT else ABSENT},
                     "mean_abs_dev": {
                         "arm_uncorrected": ta["placement"]["mean_abs_dev"],
                         "ref_uncorrected": tru["placement"]["mean_abs_dev"],
                         "ref_pcr": tr["placement"]["mean_abs_dev"] if tr != ABSENT else ABSENT,
                         "ref_pcr0": t0["placement"]["mean_abs_dev"] if t0 != ABSENT else ABSENT},
                     } if c_star is not None else {"status": ABSENT, "note": "no certified c*"}
        arm_closer = (None if (c_star is None or t0 == ABSENT) else
                      bool(ta["placement"]["abs_dev_endpoint"] <= t0["placement"]["abs_dev_endpoint"]))
        arm_lower0 = None if r0 is None else bool(a < r0)
        res = {
            "status": "DECISIVE" if complete else "PARTIAL",
            "stages_used": used, "stages_missing": [k for k in stages if k not in used],
            "ref_delta": {k: probe_delta(pb, k) for k in used},
            "ref_delta_pcr0": ({k: corrected["pcr0"][k] for k in used}
                               if corrected["pcr0"] is not None else ABSENT),
            "pcr0_offset": offsets["pcr0"].get(tname, None),
            "pcr0t_offset": offsets["pcr0t"].get(tname, None),
            "sum_abs_dc": {"arm_uncorrected": a,
                           "ref_pcr": r if r is not None else ABSENT,
                           "ref_pcr0": r0 if r0 is not None else ABSENT,
                           "ref_pcr0t": (traj["pcr0t"]["sum_abs_dc_full"]
                                         if traj["pcr0t"] != ABSENT else ABSENT),
                           "ref_uncorrected": tru["sum_abs_dc_full"],
                           "arm_pcr_own_probe": tap["sum_abs_dc_full"] if tap != ABSENT else ABSENT},
            "sum_abs_dc_post_settling_not_judged": {
                "arm_uncorrected": ta["sum_abs_dc_post_settling"],
                "ref_pcr": tr["sum_abs_dc_post_settling"] if tr != ABSENT else ABSENT,
                "ref_pcr0": t0["sum_abs_dc_post_settling"] if t0 != ABSENT else ABSENT,
                "ref_uncorrected": tru["sum_abs_dc_post_settling"]},
            "margin_ref_pcr_minus_arm": (r - a) if r is not None else ABSENT,
            "margin_ref_pcr0_minus_arm": (r0 - a) if r0 is not None else ABSENT,
            "ratio_arm_over_ref_pcr": (a / r) if (r is not None and r > 0) else None,
            "ratio_arm_over_ref_pcr0": (a / r0) if (r0 is not None and r0 > 0) else None,
            "arm_lower": bool(a < r) if r is not None else None,
            "arm_lower_vs_pcr0": arm_lower0,
            "arm_endpoint_closer_to_target_than_pcr0": arm_closer,
            "arm_survives_pcr0": (None if (arm_lower0 is None or arm_closer is None)
                                  else bool(arm_lower0 and arm_closer)),
            "placement": placement,
            "trajectories": {"arm_uncorrected": ta, "ref_pcr": tr, "ref_pcr0": t0,
                             "ref_pcr0t": traj["pcr0t"], "ref_uncorrected": tru,
                             "arm_pcr_own_probe": tap},
        }
        if pname == primary:
            dec = {"arm_uncorrected": {"__base__": [g > 0.0 for g in g0],
                                       **{k: [g > 0.0 for g in arm_gaps[k]] for k in used}}}
            for v, name in (("pcr", "ref_pcr"), ("pcr0", "ref_pcr0")):
                if corrected[v] is None:
                    continue
                off = offsets[v][tname]
                dec[name] = {"__base__": [g + off > 0.0 for g in g0],
                             **{k: [g + corrected[v][k] > 0.0 for g in ref_gaps[k]] for k in used}}
            res["bootstrap"] = bootstrap_margins(yes, dec, c_star, n_boot, boot_seed, notes)
        if not complete:
            notes.append(f"PARTIAL {tname}/{pname}: decisive sum over {used} only (missing {res['stages_missing']})")
        out["per_probe"][pname] = res
    return out


def decisive(arm_raws, ref_raws, arm_block, ref_block, stages, primary, min_join, notes,
             arm_rt, ref_rt, offsets, c_star, n_boot, boot_seed):
    out = {"decision_rule": DECISION_RULE, "pcr0_rule": PCR0_RULE, "primary_probe": primary,
           "c_star": c_star,
           "rephr": _contrast("rephr", arm_raws["rephr"], ref_raws["rephr"], arm_block["probes"],
                              ref_block["probes"], stages, min_join, notes, arm_rt, ref_rt,
                              offsets, c_star, primary, n_boot, boot_seed),
           "orig_template_context_not_judged":
               _contrast("orig", arm_raws["orig"], ref_raws["orig"], arm_block["probes"],
                         ref_block["probes"], stages, min_join, notes, arm_rt, ref_rt,
                         offsets, c_star, primary, 0, boot_seed)}
    R = out["rephr"]
    pp = R.get("per_probe", {}).get(primary) if R.get("status") == "OK" else None
    if not (isinstance(pp, dict) and pp.get("status") in ("DECISIVE", "PARTIAL")):
        out.update({"status": ABSENT, "arm_is_method": None, "arm_survives_pcr0": None,
                    "note": f"rephrased-template comparison with probe {primary!r} unavailable: "
                            + (R.get("note") or (pp or {}).get("note") or "see per_probe")})
        return out
    avail = {p: v for p, v in R["per_probe"].items() if isinstance(v, dict) and v.get("arm_lower") is not None}
    avail0 = {p: v for p, v in R["per_probe"].items()
              if isinstance(v, dict) and v.get("arm_survives_pcr0") is not None}
    out.update({
        "status": pp["status"], "stages_used": pp["stages_used"], "n_ids_joint": R["n_ids_joint"],
        "sum_abs_dc_rephr_arm_uncorrected": pp["sum_abs_dc"]["arm_uncorrected"],
        "sum_abs_dc_rephr_ref_pcr": pp["sum_abs_dc"]["ref_pcr"],
        "sum_abs_dc_rephr_ref_pcr0": pp["sum_abs_dc"]["ref_pcr0"],
        "sum_abs_dc_rephr_ref_uncorrected": pp["sum_abs_dc"]["ref_uncorrected"],
        "sum_abs_dc_rephr_arm_pcr_own_probe": pp["sum_abs_dc"]["arm_pcr_own_probe"],
        "margin_ref_pcr_minus_arm": pp["margin_ref_pcr_minus_arm"],
        "margin_ref_pcr0_minus_arm": pp["margin_ref_pcr0_minus_arm"],
        "arm_is_method": pp["arm_lower"],
        "arm_lower_vs_pcr0": pp["arm_lower_vs_pcr0"],
        "arm_endpoint_closer_to_target_than_pcr0": pp["arm_endpoint_closer_to_target_than_pcr0"],
        "arm_survives_pcr0": pp["arm_survives_pcr0"],
        "placement": pp["placement"],
        "bootstrap": pp.get("bootstrap"),
        "arm_is_method_all_available_probes": all(v["arm_lower"] for v in avail.values()) if avail else None,
        "arm_survives_pcr0_all_available_probes": (all(v["arm_survives_pcr0"] for v in avail0.values())
                                                   if avail0 else None),
        "probes_available": sorted(avail),
        "per_probe_arm_lower": {p: v["arm_lower"] for p, v in avail.items()},
        "per_probe_arm_survives_pcr0": {p: v["arm_survives_pcr0"] for p, v in avail0.items()},
    })
    return out


def verdict_line(res):
    D = res.get("decisive")
    rt, ref = res["runtag"], res.get("ref_runtag")
    if D is None:
        return f"{rt}: NO_REF -- no --ref_runtag given; arm-only diagnostics written, no method/PCR decision"
    if D["status"] == ABSENT:
        return f"{rt} vs {ref}: ABSENT -- {D['note']}"
    a = D["sum_abs_dc_rephr_arm_uncorrected"]
    r = D["sum_abs_dc_rephr_ref_pcr"]
    r0 = D["sum_abs_dc_rephr_ref_pcr0"]
    part = "" if D["status"] == "DECISIVE" else f" [PARTIAL: stages {D['stages_used']} only]"
    tail = (f" (ref uncorrected={D['sum_abs_dc_rephr_ref_uncorrected']:.3f}; "
            f"n={D['n_ids_joint']}){part}")
    if isinstance(r, float):
        tag = ("METHOD on this cell" if D["arm_is_method"]
               else "TRAIN-TIME PCR (diagnostic, not a contribution)")
        head = (f"{rt} vs {ref}: vs PCR(base-aimed) {tag} -- sum|dc_rephr| arm(uncorrected)={a:.3f} "
                f"{'<' if D['arm_is_method'] else '>='} ref(PCR, probe={D['primary_probe']})={r:.3f}")
    else:
        head = (f"{rt} vs {ref}: vs PCR(base-aimed) ABSENT -- sum|dc_rephr| "
                f"arm(uncorrected)={a:.3f}, no PCR-corrected reference")
    if not isinstance(r0, float):
        return (head + tail + " || vs PCR-0(target-aimed): ABSENT -- "
                "no certified target or no base dump to fit s* on, so the fair comparison "
                "was NOT made and no method claim may be quoted from this run")
    surv = D["arm_survives_pcr0"]
    why = []
    if D["arm_lower_vs_pcr0"] is False:
        why.append("longer path")
    if D["arm_endpoint_closer_to_target_than_pcr0"] is False:
        why.append("endpoint further from c*")
    tag0 = ("SURVIVES the target-aimed competitor" if surv else
            "LOSES to the target-aimed competitor (" + ", ".join(why or ["see placement"]) + ")")
    return (head + tail + f" || vs PCR-0(target-aimed, c*={D['c_star']:+.4f}): "
            f"{tag0} -- ref(PCR-0)={r0:.3f}, "
            f"endpoint |c-c*| arm={D['placement']['abs_dev_endpoint']['arm_uncorrected']:.3f} "
            f"vs ref(PCR-0)={D['placement']['abs_dev_endpoint']['ref_pcr0']:.3f}")


# ---------------------------------------------------------------------------
def fmt(v, spec="+.3f"):
    return format(v, spec) if isinstance(v, (int, float)) and not isinstance(v, bool) else ABSENT


def print_summary(res, primary):
    T = res.get("target") or {}
    if T:
        print(f"{TAG} target c* = {fmt(T.get('c_star'), '+.4f')} "
              f"({T.get('provenance', '?')}; prereg {PREREG_JOINT_ENDPOINT_C:+.3f})")
        for c in T.get("joint_cells", []):
            print(f"    JOINT {c['cell']:16s} endpoint k{c['endpoint_k']} c={c['c']:+.4f} "
                  f"yes_rate={c['yes_rate']}")
    O = res.get("offsets") or {}
    if O:
        cert = O.get("base_certification", {})
        print(f"{TAG} base certification: {cert.get('status', '?')} "
              f"(dump c={fmt(cert.get('observed_base_c_orig'), '+.4f')}, "
              f"aggregate c={fmt(cert.get('certified_base_c'), '+.4f')})")
        for t in TEMPLATES:
            b = O.get("per_template", {}).get(t, {})
            if b.get("status") != "OK" or "s_star" not in b:
                print(f"    s*[{t}] {ABSENT} ({b.get('note', b.get('status', '?'))})")
                continue
            ss = b["s_star"]
            lf = b.get("s_star_label_free")
            extra = (f" | label-free (yes-rate matched) s*={lf['offset']:+.4f}, "
                     f"oracle premium {b['oracle_premium']:+.4f}") if lf else ""
            print(f"    s*[{t}] = {ss['offset']:+.4f} -> base c {b['base_c']:+.4f} "
                  f"becomes {ss['achieved']:+.4f} (target {ss['target']:+.4f}, "
                  f"residual {ss['residual']:+.4f}, n={ss['n']}){extra}")
    for who in ("arm", "ref"):
        B = res.get(who)
        if not B:
            continue
        Tm = B["templates"]
        print(f"{TAG} {who} {B['runtag']}: orig {Tm['orig']['status']} rephr {Tm['rephr']['status']} | probes "
              + ", ".join(f"{p}={B['probes'][p]['status']}" for p in PROBES))
        if Tm["orig"]["status"] != "OK" and Tm["rephr"]["status"] != "OK":
            continue
        print(f"  {'stage':6s} {'c_orig':>8s} {'c_orig+PCR':>10s} {'c_rephr':>8s} {'c_rephr+PCR':>11s} "
              f"{'+PCR-0':>8s} {'F1_rephr':>8s} {'d_rephr':>7s} {'delta':>7s} {'gap-gap0':>8s}")
        bo = Tm["orig"].get("base"), Tm["rephr"].get("base")
        print(f"  {'base':6s} {fmt(bo[0]['c'] if bo[0] else None):>8s} {'-':>10s} "
              f"{fmt(bo[1]['c'] if bo[1] else None):>8s} {'-':>11s} {'-':>8s} "
              f"{fmt(bo[1]['f1'] if bo[1] else None, '.4f'):>8s} {fmt(bo[1]['dprime'] if bo[1] else None, '.3f'):>7s} "
              f"{'-':>7s} {'-':>8s}")
        cell = lambda st, v: (fmt(st[v][primary]["c"])
                              if isinstance(st, dict) and isinstance(st[v][primary], dict) else ABSENT)
        for k in res["stages"]:
            so = Tm["orig"].get("stages", {}).get(k, ABSENT)
            sr = Tm["rephr"].get("stages", {}).get(k, ABSENT)
            d = probe_delta(B["probes"][primary], k)
            gp = B["template_gap"].get("stages", {}).get(k, ABSENT)
            row = [fmt(so["uncorrected"]["c"]) if isinstance(so, dict) else ABSENT,
                   cell(so, "pcr"),
                   fmt(sr["uncorrected"]["c"]) if isinstance(sr, dict) else ABSENT,
                   cell(sr, "pcr"), cell(sr, "pcr0"),
                   fmt(sr["uncorrected"]["f1"], ".4f") if isinstance(sr, dict) else ABSENT,
                   fmt(sr["uncorrected"]["dprime"], ".3f") if isinstance(sr, dict) else ABSENT,
                   fmt(d), fmt(gp["gap_minus_base"]) if isinstance(gp, dict) else ABSENT]
            print(f"  {k:6s} {row[0]:>8s} {row[1]:>10s} {row[2]:>8s} {row[3]:>11s} {row[4]:>8s} "
                  f"{row[5]:>8s} {row[6]:>7s} {row[7]:>7s} {row[8]:>8s}")
        tr = B["trajectory"]["rephr"]
        if tr != ABSENT:
            pcr = tr["pcr"].get(primary)
            pcr0 = tr["pcr0"].get(primary)
            print(f"  sum|dc_rephr| uncorrected={fmt(tr['uncorrected']['sum_abs_dc_full'], '.3f')} "
                  f"PCR({primary})={fmt(pcr['sum_abs_dc_full'], '.3f') if isinstance(pcr, dict) else ABSENT} "
                  f"PCR-0({primary})={fmt(pcr0['sum_abs_dc_full'], '.3f') if isinstance(pcr0, dict) else ABSENT} "
                  f"| sum|dc_orig| uncorrected="
                  f"{fmt(B['trajectory']['orig']['uncorrected']['sum_abs_dc_full'], '.3f') if B['trajectory']['orig'] != ABSENT else ABSENT}"
                  f" | max|gap shift|={fmt(B['template_gap'].get('max_abs_gap_shift'), '.3f')}")
    D = res.get("decisive")
    if isinstance(D, dict) and isinstance(D.get("bootstrap"), dict) and D["bootstrap"].get("status") == "OK":
        bs = D["bootstrap"]
        print(f"{TAG} bootstrap (n_items={bs['n_items']}, B={bs['n_boot']}, "
              f"resample unit = {bs['resample_unit']}):")
        for key in ("margin_ref_pcr_minus_arm", "margin_ref_pcr0_minus_arm",
                    "endpoint_placement_arm_minus_pcr0"):
            v = bs.get(key)
            if v:
                frac = v.get("frac_arm_lower", v.get("frac_arm_closer"))
                print(f"    {key:38s} mean {v['mean']:+.4f}  CI95 "
                      f"[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  frac favouring arm {frac:.3f}")
    for n in res["notes"]:
        print(f"{TAG} NOTE {n}")
    if isinstance(D, dict) and D.get("arm_is_method") is True and D.get("arm_survives_pcr0") is False:
        print(f"{TAG} *** THE PRE-REGISTERED WIN DOES NOT SURVIVE A PROPERLY-AIMED COMPETITOR ***")
        print(f"{TAG} *** the free scalar, aimed at c* instead of at the mis-placed base, ***")
        print(f"{TAG} *** matches or beats the trained arm on this cell. Report this.      ***")
    print(f"{TAG} VERDICT {res['verdict']}")


# ---------------------------------------------------------------------------
# --placement_audit: the dumps-free half of the same question
# ---------------------------------------------------------------------------
def _boot_cells(vals, n_boot, seed):
    """Bootstrap the mean of a per-cell quantity. Cells are resampled as a LIST
    with multiplicity -- see bootstrap_margins() for why that is spelled out.
    n < 4 returns no interval: with two JOINT cells an interval would be theatre,
    and the per-cell values are printed instead."""
    if len(vals) < 4:
        return {"status": "NO_INTERVAL", "n": len(vals), "values": list(vals),
                "note": "n < 4: per-cell values quoted instead of an interval"}
    rnd = random.Random(seed)
    means = []
    for _ in range(n_boot):
        res = [vals[rnd.randrange(len(vals))] for _ in range(len(vals))]  # LIST, multiplicity
        means.append(sum(res) / len(res))
    means.sort()
    return {"status": "OK", "n": len(vals), "n_boot": n_boot,
            "ci95": [means[int(math.floor(0.025 * (n_boot - 1)))],
                     means[int(math.ceil(0.975 * (n_boot - 1)))]]}


def placement_audit(agg_path, backbone, target_c_override, n_boot, seed, out_path, quiet):
    """How far does each arm's ENDPOINT criterion sit from c*?

    This is the half of the method-vs-post-hoc comparison that needs no per-item
    logit dumps, and therefore the half that can be settled from the certified
    aggregate today. A properly-aimed post-hoc scalar is at |c - c*| = 0 BY
    CONSTRUCTION on the template it is fitted on -- that is what "aimed at the
    target" means -- so any arm whose endpoint sits far from c* is losing this
    axis to a free scalar before the transfer question is even asked. What this
    mode CANNOT do is measure the scalar's transfer residual on a rephrased
    template; that needs the dumps, and it is the only thing keeping this from
    being the whole verdict.

    Per cell first, then the mean over cells, with n reported every time."""
    notes = []
    agg = load_aggregate(agg_path, required=True, notes=notes)
    target = derive_target(agg, backbone, target_c_override, notes)
    c_star = target["c_star"]
    base_c = certified_base_c(agg, backbone)
    res = {"mode": "placement_audit", "aggregate": os.path.abspath(agg_path),
           "backbone": backbone, "target": target, "base_c": base_c,
           "base_abs_dev": abs(base_c - c_star) if base_c is not None else None,
           "arms": {}, "notes": notes,
           "what_this_cannot_show": "the post-hoc scalar's TRANSFER residual on a rephrased "
                                    "template; that needs the per-item logit dumps and is the "
                                    "only reason this is not the whole verdict"}
    arm_names = sorted({k.split("|")[0]
                        for k in agg.get("backbones", {}).get(backbone, {}).get("arms", {})})
    for arm in arm_names:
        cells = _cells(agg, backbone, arm)
        per_cell = []
        for key, stages in cells:
            cs = [p["c"] for _, p in stages]
            chain = ([base_c] + cs) if base_c is not None else cs
            full = sum(abs(b - a) for a, b in zip(chain, chain[1:]))
            post = sum(abs(b - a) for a, b in zip(chain[1:], chain[2:]))
            end = cs[-1]
            per_cell.append({"cell": key, "n_stages": len(cs), "endpoint_k": stages[-1][0],
                             "endpoint_c": end, "abs_dev_endpoint": abs(end - c_star),
                             "sum_abs_dc_full": full, "sum_abs_dc_post_settling": post,
                             "endpoint_f1": stages[-1][1].get("f1"),
                             "endpoint_dprime": stages[-1][1].get("dprime")})
        if not per_cell:
            continue
        devs = [x["abs_dev_endpoint"] for x in per_cell]
        raw = [x["sum_abs_dc_full"] for x in per_cell]
        post = [x["sum_abs_dc_post_settling"] for x in per_cell]
        res["arms"][arm] = {
            "n_cells": len(per_cell),
            "mean_abs_dev_endpoint": mean(devs),
            "mean_endpoint_c": mean(x["endpoint_c"] for x in per_cell),
            # RAW and POST-SETTLING are different estimators and are reported on
            # separate lines, never mixed in one comparison (analysis/
            # TRADEOFF_PLACEMENT.md: a raw single-cell path quoted against a
            # post-settling nine-cell mean is what produced the trade-off
            # hypothesis that the same file then had to refute).
            "mean_sum_abs_dc_raw": mean(raw), "range_sum_abs_dc_raw": [min(raw), max(raw)],
            "mean_sum_abs_dc_post_settling": mean(post),
            "range_sum_abs_dc_post_settling": [min(post), max(post)],
            "mean_endpoint_f1": mean(x["endpoint_f1"] for x in per_cell
                                     if isinstance(x["endpoint_f1"], (int, float))),
            "boot_mean_abs_dev_endpoint": _boot_cells(devs, n_boot, seed),
            "boot_mean_sum_abs_dc_raw": _boot_cells(raw, n_boot, seed),
            "boot_mean_sum_abs_dc_post_settling": _boot_cells(post, n_boot, seed),
            "cells": per_cell,
        }
    if not quiet:
        print(f"{TAG} PLACEMENT AUDIT (no dumps needed) -- aggregate {res['aggregate']}")
        print(f"{TAG} c* = {c_star:+.4f} ({target['provenance']}), base c = {base_c:+.4f} "
              f"-> base is mis-placed by {res['base_abs_dev']:.4f}")
        print("  PLACEMENT axis -- |endpoint c - c*|, per cell then averaged")
        print(f"  {'arm':8s} {'n':>3s} {'mean end c':>11s} {'mean|c-c*|':>11s} {'CI95':>22s} "
              f"{'mean end F1':>12s}")
        for arm, b in sorted(res["arms"].items()):
            bt = b["boot_mean_abs_dev_endpoint"]
            ci = (f"[{bt['ci95'][0]:.4f}, {bt['ci95'][1]:.4f}]" if bt["status"] == "OK"
                  else "n<4: " + ", ".join(f"{v:.4f}" for v in bt["values"]))
            print(f"  {arm:8s} {b['n_cells']:3d} {b['mean_endpoint_c']:+11.4f} "
                  f"{b['mean_abs_dev_endpoint']:11.4f} {ci:>22s} {b['mean_endpoint_f1']:12.4f}")
        print(f"  {'PCR-0':8s} {'-':>3s} {c_star:+11.4f} {0.0:11.4f} {'by construction':>22s} "
              f"{'(transfer residual unmeasured here)':>12s}")
        print("\n  DRIFT axes -- RAW and POST-SETTLING are different estimators; never")
        print("  quote one arm's raw path against another arm's post-settling path.")
        print(f"  {'arm':8s} {'n':>3s} {'raw mean':>9s} {'raw range':>20s} "
              f"{'post mean':>10s} {'post range':>20s}")
        for arm, b in sorted(res["arms"].items()):
            rr, pr = b["range_sum_abs_dc_raw"], b["range_sum_abs_dc_post_settling"]
            print(f"  {arm:8s} {b['n_cells']:3d} {b['mean_sum_abs_dc_raw']:9.4f} "
                  f"{'[%.4f, %.4f]' % (rr[0], rr[1]):>20s} "
                  f"{b['mean_sum_abs_dc_post_settling']:10.4f} "
                  f"{'[%.4f, %.4f]' % (pr[0], pr[1]):>20s}")
        for n in notes:
            print(f"{TAG} NOTE {n}")
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path + ".tmp", "w") as f:
            json.dump(res, f, indent=1)
        os.replace(out_path + ".tmp", out_path)
        if not quiet:
            print(f"{TAG} wrote {out_path}")
    return res


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", help="results_fs directory")
    ap.add_argument("--runtag", help="arm runtag, e.g. mth_critp_o1_s17")
    ap.add_argument("--base", help="base label, e.g. llava15_base")
    ap.add_argument("--stages", default=None, help="comma list (k1,k2,...); default: discovered <RT>_k<N> dirs")
    ap.add_argument("--ref_runtag", default=None, help="same-path SEQ reference, e.g. psL_seq_o1_s17")
    ap.add_argument("--primary_probe", default="coco", choices=sorted(PROBES))
    ap.add_argument("--min_join_frac", type=float, default=0.5)
    ap.add_argument("--aggregate", default=DEFAULT_AGGREGATE,
                    help="certified fs_aggregate.json; source of the PCR-0 target c*")
    ap.add_argument("--backbone", default="llava15")
    ap.add_argument("--target_c", type=float, default=None,
                    help="override c* (recorded as manual provenance); default: the JOINT arm's "
                         "empirical endpoint criterion from --aggregate")
    ap.add_argument("--n_boot", type=int, default=1000,
                    help="paired item bootstrap replicates for the margins; 0 disables")
    ap.add_argument("--boot_seed", type=int, default=20260912)
    ap.add_argument("--placement_audit", action="store_true",
                    help="dumps-free mode: endpoint placement per arm from the aggregate only")
    ap.add_argument("--out", help="output json")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    if a.placement_audit:
        return placement_audit(a.aggregate, a.backbone, a.target_c, max(a.n_boot, 1000),
                               a.boot_seed, a.out, a.quiet)
    missing = [f"--{n}" for n in ("results", "runtag", "base", "out") if getattr(a, n) is None]
    if missing:
        ap.error("missing required argument(s): " + ", ".join(missing))

    notes = []
    if a.stages:
        stages, discovered = [s.strip() for s in a.stages.split(",") if s.strip()], None
    else:
        stages, discovered = discover_stages(a.results, [a.runtag] + ([a.ref_runtag] if a.ref_runtag else []))
        if not discovered:
            notes.append(f"no <runtag>_k<N> dirs found under {a.results}; defaulting to k1..k6 (all ABSENT)")
    agg = load_aggregate(a.aggregate, required=(a.target_c is None), notes=notes)
    target = derive_target(agg, a.backbone, a.target_c, notes)
    c_star = target.get("c_star")
    offsets, off_block = fit_offsets(a.results, a.base, target, agg, a.backbone, notes)
    res = {"runtag": a.runtag, "ref_runtag": a.ref_runtag, "base": a.base, "results": os.path.abspath(a.results),
           "stages": stages, "stages_discovered": discovered, "primary_probe": a.primary_probe,
           "min_join_frac": a.min_join_frac, "decision_rule": DECISION_RULE, "pcr0_rule": PCR0_RULE,
           "aggregate": os.path.abspath(a.aggregate) if os.path.isfile(a.aggregate) else ABSENT,
           "target": target, "offsets": off_block, "n_boot": a.n_boot, "boot_seed": a.boot_seed,
           "notes": notes}
    arm_block, arm_raws = analyse_arm(a.results, a.base, a.runtag, stages, a.min_join_frac, notes,
                                      offsets, c_star)
    res["arm"] = arm_block
    if a.ref_runtag:
        ref_block, ref_raws = analyse_arm(a.results, a.base, a.ref_runtag, stages, a.min_join_frac, notes,
                                          offsets, c_star)
        res["ref"] = ref_block
        res["decisive"] = decisive(arm_raws, ref_raws, arm_block, ref_block, stages, a.primary_probe,
                                   a.min_join_frac, notes, a.runtag, a.ref_runtag, offsets, c_star,
                                   a.n_boot, a.boot_seed)
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
