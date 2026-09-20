#!/usr/bin/env python3
"""Synthetic end-to-end test for analysis/pcr_transfer.py.

Builds OBVIOUSLY FAKE logit dumps in a tempdir (marker file inside) and drives
the real pcr_transfer.run() over them. Nothing is written into the repo.

Tree (results_fs layout, one directory per label):
    llava15_base/{pope,pope_rephr,probe_prompts}_logits.jsonl
    <runtag>_k1..k4/  same three files

Every arm is one constant per-stage shift added to the base gaps, separately on
the ORIGINAL POPE template, on the REPHRASED POPE template, and on the label-free
COCO probe. For equal-variance Gaussian gaps a shift of +s moves the criterion by
-s, so the arms below are readable straight off their shift vectors:

  psL_seq_o1_s17   (reference)  orig  s      probe  s      rephr  1.4*s
        drifts everywhere; its probe tracks the ORIGINAL template exactly, so the
        label-free PCR scalar delta_k = mean g_probe(base) - mean g_probe(k) = -s
        nearly erases the drift on the original template and leaves 0.4*s on the
        rephrased one. That residual is the bar every arm has to beat.
  mth_critp_o1_s17       orig 0      probe 0      rephr 0        -> METHOD
  mth_trainpcr_o1_s17    orig 0      probe 0      rephr 1.4*s    -> NOT a method
        (holds the criterion only on the template it was trained to pin)
  psL_seq_o1_s23         orig .9*s   probe .9*s   rephr 1.26*s   -> NOT a method
        (a second SEQ cell: drifts on both templates)

Asserts the decisive boolean in all three cases, that the correction uses the
REFERENCE's probe deltas (not the arm's), that PCR wins by construction on the
original template, the fail-loud paths, and ABSENT degradation when the
rephrased or probe dumps are missing (never a crash, never an imputed 0.0).

Run:  python3 analysis/tests/test_pcr_transfer.py
"""
import json
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import pcr_transfer as PT  # noqa: E402

ABSENT = PT.ABSENT
BASE = "llava15_base"
REF = "psL_seq_o1_s17"
STAGES = ["k1", "k2", "k3", "k4"]
N_POPE = 600            # 300 yes / 300 no
N_PROBE = 400
S = [-0.35, 0.95, -0.20, 1.40]      # the reference's original-template drift
RHO = 1.4                            # rephrased-template drift / original drift

ARMS = {
    REF:                    {"orig": S, "probe": S, "rephr": [RHO * s for s in S]},
    "mth_critp_o1_s17":     {"orig": [0.0] * 4, "probe": [0.0] * 4, "rephr": [0.0] * 4},
    "mth_trainpcr_o1_s17":  {"orig": [0.0] * 4, "probe": [0.0] * 4, "rephr": [RHO * s for s in S]},
    "psL_seq_o1_s23":       {"orig": [0.9 * s for s in S], "probe": [0.9 * s for s in S],
                             "rephr": [0.9 * RHO * s for s in S]},
}


# ---------------------------------------------------------------- fixture
def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _row(rid, gt, category, gap):
    # gap = z_yes - z_no is what the script reads; the halves keep the dump
    # self-consistent with pilot/eval_gen.py --dump_logits.
    return {"id": rid, "gt": gt, "category": category, "z_yes": gap / 2.0,
            "z_no": -gap / 2.0, "gap": gap, "argmax_is": "yes" if gap > 0 else "no"}


def dump_pope(path, ids, gts, gaps):
    write_jsonl(path, [_row(i, g, "adversarial", v) for i, g, v in zip(ids, gts, gaps)])


def dump_probe(path, pids, gaps):
    write_jsonl(path, [_row(i, None, "probe", v) for i, v in zip(pids, gaps)])


def build_tree(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "THIS_IS_SYNTHETIC_FIXTURE_DATA.txt").write_text("fake dumps for test_pcr_transfer.py\n")
    rng = random.Random(11)
    ids = ["pope_%04d" % i for i in range(N_POPE)]
    gts = ["yes"] * (N_POPE // 2) + ["no"] * (N_POPE // 2)
    pids = ["probe_%04d" % i for i in range(N_PROBE)]
    b_orig = [rng.gauss(1.6 if g == "yes" else -1.6, 1.0) for g in gts]
    # the rephrased template is the same items asked differently: a template
    # offset plus item-level re-ranking, never a copy (the script warns on that).
    b_rephr = [g + 0.25 + rng.gauss(0, 0.35) for g in b_orig]
    b_probe = [rng.gauss(0.4, 1.2) for _ in pids]
    d = root / BASE
    d.mkdir()
    dump_pope(d / "pope_logits.jsonl", ids, gts, b_orig)
    dump_pope(d / "pope_rephr_logits.jsonl", ids, gts, b_rephr)
    dump_probe(d / "probe_prompts_logits.jsonl", pids, b_probe)
    for rt, spec in ARMS.items():
        for j, k in enumerate(STAGES):
            d = root / ("%s_%s" % (rt, k))
            d.mkdir()
            dump_pope(d / "pope_logits.jsonl", ids, gts,
                      [g + spec["orig"][j] + rng.gauss(0, 0.05) for g in b_orig])
            dump_pope(d / "pope_rephr_logits.jsonl", ids, gts,
                      [g + spec["rephr"][j] + rng.gauss(0, 0.05) for g in b_rephr])
            dump_probe(d / "probe_prompts_logits.jsonl", pids,
                       [g + spec["probe"][j] + rng.gauss(0, 0.05) for g in b_probe])
    return root


MIS_ARM = "mth_anchorcrit_o1_s17"
MIS_BASE_C = 0.43          # the real llava15 base sits at +0.4314
MIS_S = [-0.30, 0.85, -0.25, 1.20]


def build_misplaced_tree(root):
    """A tree whose BASE is mis-placed exactly the way the real one is, and whose
    arm pins the criterion to it -- the anchor's failure mode.

    Equal-variance gaps with yes ~ N(d/2 + m, 1) and no ~ N(-d/2 + m, 1) sit at
    c = -m at threshold 0, so m = -MIS_BASE_C puts the base at c = +0.43 while
    leaving d' alone. The arm holds that criterion on BOTH templates (a real
    method by the pre-registered rule); the reference drifts. The question this
    tree exists to answer is whether pinning to a mis-placed base still counts as
    a method once the free scalar is aimed at the target instead of at the base.

    The rephrased template here is item-level re-ranking with NO template offset,
    and the reference's rephrased drift is only 1.05x its original drift, so the
    scalar's TRANSFER residual is small by construction. That is deliberate: it
    isolates the effect of the target the scalar is aimed at. Where the residual
    is large the re-aim can overshoot the target instead -- a real property of
    PCR-0, visible in the first tree above, that this test should not hide.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "THIS_IS_SYNTHETIC_FIXTURE_DATA.txt").write_text("fake dumps for test_pcr_transfer.py\n")
    rng = random.Random(4242)
    ids = ["pope_%04d" % i for i in range(N_POPE)]
    gts = ["yes"] * (N_POPE // 2) + ["no"] * (N_POPE // 2)
    pids = ["probe_%04d" % i for i in range(N_PROBE)]
    d_prime = 2.35
    b_orig = [rng.gauss((d_prime / 2 if g == "yes" else -d_prime / 2) - MIS_BASE_C, 1.0) for g in gts]
    b_rephr = [g + rng.gauss(0, 0.30) for g in b_orig]
    b_probe = [rng.gauss(-0.5, 1.1) for _ in pids]
    d = root / BASE
    d.mkdir()
    dump_pope(d / "pope_logits.jsonl", ids, gts, b_orig)
    dump_pope(d / "pope_rephr_logits.jsonl", ids, gts, b_rephr)
    dump_probe(d / "probe_prompts_logits.jsonl", pids, b_probe)
    arms = {REF: {"orig": MIS_S, "probe": MIS_S, "rephr": [1.05 * s for s in MIS_S]},
            MIS_ARM: {"orig": [0.0] * 4, "probe": [0.0] * 4, "rephr": [0.0] * 4}}
    for rt, spec in arms.items():
        for j, k in enumerate(STAGES):
            d = root / ("%s_%s" % (rt, k))
            d.mkdir()
            dump_pope(d / "pope_logits.jsonl", ids, gts,
                      [g + spec["orig"][j] + rng.gauss(0, 0.05) for g in b_orig])
            dump_pope(d / "pope_rephr_logits.jsonl", ids, gts,
                      [g + spec["rephr"][j] + rng.gauss(0, 0.05) for g in b_rephr])
            dump_probe(d / "probe_prompts_logits.jsonl", pids,
                       [g + spec["probe"][j] + rng.gauss(0, 0.05) for g in b_probe])
    return root


def check_reaim_path_invariance():
    """Does aiming the free scalar at c* instead of at the base change the PATH?

    It should not, and the reason is geometric: if a cell's stages lie on ONE
    z-ROC -- fixed evidence distributions, sliding threshold, which is the
    project's primary claim and holds for the sequential arm at mean R^2 = 0.957
    (analysis/readout/zroc_coherence.json) -- then c is linear in an added offset
    with a slope set by the (shared) evidence scales, so a CONSTANT re-aim moves
    every stage's criterion by the same amount and every |dc| is unchanged. Any
    residual is finite-sample discreteness in the empirical z-transform.

    This matters for reading the verdict: it says the rigging fixed by PCR-0 does
    NOT live on the pre-registered path axis, so the pre-registered boolean is
    largely unaffected by it. It lives entirely on the placement axis. If this
    check ever failed at POPE's n, that reading would be wrong and the path
    comparison would have to be re-run under both targets before being quoted.

    Exercised at POPE's real n (4500/4500) across z-ROC slopes spanning the fitted
    range 0.55-0.80, with an imperfect (85%) transport so the corrected path is
    non-zero the way a real one is.
    """
    zq = PT._z
    def c_of(gaps, yes, d):
        ny = sum(1 for y in yes if y); nn = len(yes) - ny
        h = sum(1 for g, y in zip(gaps, yes) if y and g + d > 0)
        f = sum(1 for g, y in zip(gaps, yes) if (not y) and g + d > 0)
        return -0.5 * (zq(PT.clipped_rate(h, ny)[0]) + zq(PT.clipped_rate(f, nn)[0]))
    shifts = [-0.30, 0.85, -0.25, 1.20, 0.40, -0.75]
    worst = 0.0
    for slope in (1.00, 0.70, 0.55):
        rng = random.Random(5)
        n_side = 4500
        yes = [True] * n_side + [False] * n_side
        base = [rng.gauss(1.175, 1.0) if y else rng.gauss(-1.175 * slope, slope) for y in yes]
        stages = [[g + s for g in base] for s in shifts]     # one ROC: threshold only
        deltas = [-s * 0.85 for s in shifts]                 # imperfect transport
        def path(off):
            cs = [c_of(base, yes, off)] + [c_of(st, yes, d + off) for st, d in zip(stages, deltas)]
            return sum(abs(b - a) for a, b in zip(cs, cs[1:]))
        p0, p1 = path(0.0), path(0.343)                      # 0.343 = our real |c_base - c*|
        worst = max(worst, abs(p1 - p0))
        print("   z-ROC slope %.2f: path(base-aimed)=%.4f  path(target-aimed)=%.4f  diff=%+.4f (%+.1f%%)"
              % (slope, p0, p1, p1 - p0, 100 * (p1 - p0) / p0))
        assert abs(p1 - p0) < 0.10 * p0, (slope, p0, p1)
    print("re-aiming the scalar leaves the PATH axis alone (worst |diff| %.4f)  OK" % worst)


def run_cell(root, runtag, out, ref=REF, stages=STAGES, extra=()):
    """stages=None omits --stages, exercising directory discovery."""
    argv = ["--results", str(root), "--runtag", runtag, "--base", BASE, "--out", str(out)]
    if stages:
        argv += ["--stages", ",".join(stages)]
    if ref:
        argv += ["--ref_runtag", ref]
    argv += list(extra)
    print("\n=== pcr_transfer %s vs %s ===" % (runtag, ref))
    return PT.run(argv)


def variant(root, work, name, drop):
    """Copy the tree and delete the (label, filename) pairs in `drop`."""
    dst = Path(work) / name
    shutil.copytree(str(root), str(dst))
    for label, fname in drop:
        p = dst / label / fname
        assert p.is_file(), p
        p.unlink()
    return dst


# ---------------------------------------------------------------- assertions
def close(a, b, tol, what):
    assert isinstance(a, float) and abs(a - b) <= tol, (what, a, b, tol)


def main():
    work = tempfile.mkdtemp(prefix="pcr_transfer_test_")
    ok = False
    try:
        root = build_tree(Path(work) / "results_fs")
        out = Path(work) / "out"
        out.mkdir()

        # ---- 1. a true method: holds c on BOTH templates -------------------
        r1 = run_cell(root, "mth_critp_o1_s17", out / "critp.json")
        D1 = r1["decisive"]
        assert D1["status"] == "DECISIVE", D1
        assert D1["arm_is_method"] is True, D1["verdict"] if "verdict" in D1 else D1
        a1, p1 = D1["sum_abs_dc_rephr_arm_uncorrected"], D1["sum_abs_dc_rephr_ref_pcr"]
        assert a1 < p1, (a1, p1)
        assert D1["margin_ref_pcr_minus_arm"] > 0
        assert "METHOD on this cell" in r1["verdict"], r1["verdict"]
        # the correction must use the REFERENCE's own probe deltas (delta = -s),
        # not the arm's (whose probe is pinned at 0).
        ref_delta = D1["rephr"]["per_probe"]["coco"]["ref_delta"]
        for j, k in enumerate(STAGES):
            close(ref_delta[k], -S[j], 0.15, "ref_delta " + k)
        arm_probe = r1["arm"]["probes"]["coco"]["stages"]
        for k in STAGES:
            close(arm_probe[k]["delta"], 0.0, 0.15, "arm delta " + k)
        # PCR on the ORIGINAL template is near-perfect by construction: reported
        # as context, never judged.
        oc = D1["orig_template_context_not_judged"]["per_probe"]["coco"]["sum_abs_dc"]
        assert oc["ref_pcr"] < 0.4 * oc["ref_uncorrected"], oc
        # ... and it does NOT transfer: the rephrased residual is much larger.
        assert p1 > 4 * oc["ref_pcr"], (p1, oc["ref_pcr"])
        assert D1["sum_abs_dc_rephr_ref_uncorrected"] > p1, D1
        # the arm holds the criterion on both templates
        tr = r1["arm"]["trajectory"]
        assert tr["rephr"]["uncorrected"]["sum_abs_dc_full"] < 0.5
        assert tr["orig"]["uncorrected"]["sum_abs_dc_full"] < 0.5
        assert tr["rephr"]["uncorrected"]["complete"] is True
        # nothing imputed: every stage present, no ABSENT in the c chain
        assert tr["rephr"]["uncorrected"]["stages_absent"] == []
        assert r1["arm"]["template_gap"]["status"] == "OK"
        assert not any(n.startswith("WARN_IDENTICAL_TEMPLATES") for n in r1["notes"]), r1["notes"]
        print("1. true method -> METHOD  OK  (arm %.3f < ref+PCR %.3f)" % (a1, p1))

        # ---- 2. train-time PCR: holds c only on the ORIGINAL template ------
        r2 = run_cell(root, "mth_trainpcr_o1_s17", out / "trainpcr.json")
        D2 = r2["decisive"]
        assert D2["status"] == "DECISIVE", D2
        assert D2["arm_is_method"] is False, D2
        a2 = D2["sum_abs_dc_rephr_arm_uncorrected"]
        assert a2 > D2["sum_abs_dc_rephr_ref_pcr"], D2
        assert D2["margin_ref_pcr_minus_arm"] < 0
        assert "TRAIN-TIME PCR" in r2["verdict"], r2["verdict"]
        # it really is indistinguishable from SEQ once the template is rephrased,
        # while looking perfectly pinned on the original one
        assert r2["arm"]["trajectory"]["orig"]["uncorrected"]["sum_abs_dc_full"] < 0.5
        assert r2["arm"]["trajectory"]["rephr"]["uncorrected"]["sum_abs_dc_full"] > 3.0
        # (5) the template-transfer gap moves a lot for a template-only pin
        assert r2["arm"]["template_gap"]["max_abs_gap_shift"] > \
            r1["arm"]["template_gap"]["max_abs_gap_shift"], "trainpcr should shift the template gap most"
        print("2. train-time PCR -> NOT a method  OK  (arm %.3f >= ref+PCR %.3f)"
              % (a2, D2["sum_abs_dc_rephr_ref_pcr"]))

        # ---- 3. a SEQ cell: drifts on both templates -----------------------
        r3 = run_cell(root, "psL_seq_o1_s23", out / "seq23.json")
        D3 = r3["decisive"]
        assert D3["status"] == "DECISIVE", D3
        assert D3["arm_is_method"] is False, D3
        assert D3["sum_abs_dc_rephr_arm_uncorrected"] > D3["sum_abs_dc_rephr_ref_pcr"], D3
        assert r3["arm"]["trajectory"]["orig"]["uncorrected"]["sum_abs_dc_full"] > 2.0
        assert r3["arm"]["trajectory"]["rephr"]["uncorrected"]["sum_abs_dc_full"] > 2.0
        # its own PCR helps on the original template but not on the rephrased one
        own = r3["arm"]["trajectory"]
        assert own["orig"]["pcr"]["coco"]["sum_abs_dc_full"] < \
            own["orig"]["uncorrected"]["sum_abs_dc_full"]
        assert "TRAIN-TIME PCR" in r3["verdict"], r3["verdict"]
        print("3. SEQ vs SEQ -> NOT a method  OK  (arm %.3f >= ref+PCR %.3f)"
              % (D3["sum_abs_dc_rephr_arm_uncorrected"], D3["sum_abs_dc_rephr_ref_pcr"]))

        # every decisive run agrees across the probes it actually has
        for r in (r1, r2, r3):
            D = r["decisive"]
            assert D["probes_available"] == ["coco"], D["probes_available"]
            assert D["arm_is_method_all_available_probes"] == D["arm_is_method"]
            assert D["rephr"]["per_probe"]["oi"]["status"] == ABSENT
            assert D["rephr"]["per_probe"]["noise"]["status"] == ABSENT

        # ---- ABSENT: the reference has no probe dump -----------------------
        v = variant(root, work, "no_ref_probe",
                    [(BASE, "probe_prompts_logits.jsonl")]
                    + [("%s_%s" % (rt, k), "probe_prompts_logits.jsonl")
                       for rt in ARMS for k in STAGES])
        r = run_cell(v, "mth_critp_o1_s17", out / "no_probe.json")
        assert r["decisive"]["status"] == ABSENT and r["decisive"]["arm_is_method"] is None, r["decisive"]
        assert "ABSENT" in r["verdict"], r["verdict"]
        assert r["arm"]["probes"]["coco"]["status"] == ABSENT
        # the arm-only trajectories still exist and are not imputed
        assert r["arm"]["trajectory"]["rephr"]["pcr"]["coco"] == ABSENT
        assert isinstance(r["arm"]["trajectory"]["rephr"]["uncorrected"]["sum_abs_dc_full"], float)
        print("ABSENT ref probe -> arm_is_method None  OK")

        # ---- ABSENT: the arm has no probe, the reference does --------------
        v = variant(root, work, "no_arm_probe",
                    [("mth_critp_o1_s17_%s" % k, "probe_prompts_logits.jsonl") for k in STAGES])
        r = run_cell(v, "mth_critp_o1_s17", out / "no_arm_probe.json")
        assert r["decisive"]["arm_is_method"] is True, r["decisive"]
        assert r["decisive"]["sum_abs_dc_rephr_arm_pcr_own_probe"] == ABSENT
        assert r["arm"]["probes"]["coco"]["status"] == ABSENT
        print("ABSENT arm probe -> still decisive on the ref's probe  OK")

        # ---- ABSENT: no rephrased stage dumps for the arm ------------------
        v = variant(root, work, "no_rephr_stage",
                    [("mth_critp_o1_s17_%s" % k, "pope_rephr_logits.jsonl") for k in STAGES])
        r = run_cell(v, "mth_critp_o1_s17", out / "no_rephr_stage.json")
        assert r["decisive"]["status"] == ABSENT and r["decisive"]["arm_is_method"] is None, r["decisive"]
        t = r["arm"]["trajectory"]["rephr"]["uncorrected"]
        assert t["sum_abs_dc_full"] == ABSENT and t["sum_abs_dc_post_settling"] == ABSENT, t
        assert t["n_transitions"] == 0 and t["complete"] is False
        assert r["arm"]["templates"]["rephr"]["stages_absent"] == STAGES
        # the original template is untouched and still analysable
        assert isinstance(r["arm"]["trajectory"]["orig"]["uncorrected"]["sum_abs_dc_full"], float)
        print("ABSENT rephrased stages -> ABSENT sums, no 0.0 imputed  OK")

        # ---- ABSENT: the rephrased BASE dump is gone -----------------------
        v = variant(root, work, "no_rephr_base", [(BASE, "pope_rephr_logits.jsonl")])
        r = run_cell(v, "mth_critp_o1_s17", out / "no_rephr_base.json")
        assert r["decisive"]["status"] == ABSENT and r["decisive"]["arm_is_method"] is None
        assert r["arm"]["templates"]["rephr"]["status"] == ABSENT
        assert r["arm"]["trajectory"]["rephr"] == ABSENT
        assert r["arm"]["template_gap"]["status"] == ABSENT
        print("ABSENT rephrased base -> whole rephrased block ABSENT  OK")

        # ---- ABSENT: the ORIGINAL base dump is gone (mirror image) ---------
        v = variant(root, work, "no_orig_base", [(BASE, "pope_logits.jsonl")])
        r = run_cell(v, "mth_critp_o1_s17", out / "no_orig_base.json")
        assert r["arm"]["templates"]["orig"]["status"] == ABSENT
        assert r["arm"]["trajectory"]["orig"] == ABSENT
        assert r["arm"]["template_gap"]["status"] == ABSENT
        # the rephrased template alone still decides the cell
        assert r["decisive"]["arm_is_method"] is True, r["decisive"]
        assert r["decisive"]["orig_template_context_not_judged"]["status"] == ABSENT
        print("ABSENT original base -> decided on the rephrased template alone  OK")

        # ---- PARTIAL: one stage missing on the rephrased template ----------
        v = variant(root, work, "partial", [("mth_critp_o1_s17_k2", "pope_rephr_logits.jsonl")])
        r = run_cell(v, "mth_critp_o1_s17", out / "partial.json")
        D = r["decisive"]
        assert D["status"] == "PARTIAL", D
        assert D["arm_is_method"] is True, D
        assert D["stages_used"] == ["k1", "k3", "k4"], D["stages_used"]
        assert D["rephr"]["per_probe"]["coco"]["stages_missing"] == ["k2"]
        assert "PARTIAL" in r["verdict"], r["verdict"]
        assert any(n.startswith("PARTIAL rephr/coco") for n in r["notes"]), r["notes"]
        print("PARTIAL one rephrased stage -> decided on 3 stages, flagged  OK")

        # ---- no reference at all -------------------------------------------
        r = run_cell(root, "mth_critp_o1_s17", out / "noref.json", ref=None)
        assert r["decisive"] is None and r["ref"] is None
        assert "NO_REF" in r["verdict"], r["verdict"]
        assert isinstance(r["arm"]["trajectory"]["rephr"]["uncorrected"]["sum_abs_dc_full"], float)
        print("no --ref_runtag -> NO_REF, arm-only diagnostics  OK")

        # ---- PARTIAL: the reference's probe pass failed on one checkpoint ---
        v = variant(root, work, "partial_probe", [("%s_k3" % REF, "probe_prompts_logits.jsonl")])
        r = run_cell(v, "mth_critp_o1_s17", out / "partial_probe.json")
        D = r["decisive"]
        assert D["status"] == "PARTIAL" and D["arm_is_method"] is True, D
        assert D["stages_used"] == ["k1", "k2", "k4"], D["stages_used"]
        assert r["ref"]["probes"]["coco"]["stages"]["k3"] == ABSENT
        assert r["ref"]["probes"]["coco"]["stages_absent"] == ["k3"]
        # the stage with no delta is dropped from the decision, not corrected by 0
        assert "k3" not in D["rephr"]["per_probe"]["coco"]["ref_delta"]
        print("PARTIAL one reference probe checkpoint -> stage dropped, not zero-corrected  OK")

        # ---- stage discovery from the <runtag>_k<N> directories -------------
        r = run_cell(root, "mth_critp_o1_s17", out / "discover.json", stages=None)
        assert r["stages_discovered"] is True and r["stages"] == STAGES, (r["stages_discovered"], r["stages"])
        assert r["decisive"]["arm_is_method"] is True and r["decisive"]["status"] == "DECISIVE"
        print("stage discovery from <runtag>_k<N> dirs  OK")

        # ---- the written JSON is the returned object ------------------------
        on_disk = json.load(open(out / "critp.json"))
        assert on_disk["decisive"]["arm_is_method"] is True
        assert on_disk["decision_rule"] == PT.DECISION_RULE
        assert on_disk["stages"] == STAGES and on_disk["base"] == BASE

        # ---- PCR-0: the competitor aimed at the target, not at the base ----
        mis = build_misplaced_tree(Path(work) / "results_misplaced")
        rm = run_cell(mis, MIS_ARM, out / "misplaced.json",
                      extra=["--n_boot", "400", "--boot_seed", "7"])
        # the target is the CERTIFIED JOINT endpoint, never a modelled zero
        T = rm["target"]
        assert T["status"] == "OK" and T["provenance"].startswith("JOINT arm endpoint"), T
        assert abs(T["c_star"] - PT.PREREG_JOINT_ENDPOINT_C) <= PT.TARGET_DRIFT_TOL, T
        assert T["n_joint_cells"] >= 1 and T["c_star"] != 0.0, T
        # s* moves the mis-placed base onto the target, and nowhere else
        O = rm["offsets"]
        assert O["synthetic_fixture_tree"] is True
        assert O["base_certification"]["status"] == "MISMATCH", O["base_certification"]
        assert any(n.startswith("SYNTHETIC_TREE") for n in rm["notes"]), rm["notes"]
        so = O["per_template"]["orig"]
        close(so["base_c"], MIS_BASE_C, 0.12, "mis-placed base c")
        close(so["s_star"]["achieved"], T["c_star"], 0.02, "s* achieved criterion")
        assert so["s_star"]["offset"] > 0, so["s_star"]   # base above target -> push yes-ward
        assert O["s_star_transferred_from"] == "orig"
        # the arm pins the criterion: a METHOD under the pre-registered rule ...
        D = rm["decisive"]
        assert D["status"] == "DECISIVE", D
        assert D["arm_is_method"] is True, D
        assert D["arm_lower_vs_pcr0"] is True, D      # the path comparison barely moves ...
        # ... because a constant re-aim shifts every stage alike; what changes is
        # WHERE the corrected reference lands.
        P = D["placement"]
        assert P["abs_dev_endpoint"]["ref_pcr0"] < P["abs_dev_endpoint"]["ref_pcr"], P
        assert P["abs_dev_endpoint"]["ref_pcr0"] < P["abs_dev_endpoint"]["arm_uncorrected"], P
        assert D["arm_endpoint_closer_to_target_than_pcr0"] is False, D
        assert D["arm_survives_pcr0"] is False, D
        assert "LOSES to the target-aimed competitor" in rm["verdict"], rm["verdict"]
        assert "endpoint further from c*" in rm["verdict"], rm["verdict"]
        # the interval exists, is not degenerate, and counts every drawn item
        B = D["bootstrap"]
        assert B["status"] == "OK" and B["n_boot"] == 400, B
        assert B["n_items"] == D["n_ids_joint"], B
        pm = B["endpoint_placement_arm_minus_pcr0"]
        assert pm["ci95"][0] < pm["ci95"][1] and pm["ci95"][0] > 0, pm   # arm strictly worse
        assert pm["frac_arm_closer"] <= 0.05, pm
        # pcr0t (refit on the scored template) is reported but never judged
        pp = D["rephr"]["per_probe"]["coco"]
        assert isinstance(pp["sum_abs_dc"]["ref_pcr0t"], float), pp["sum_abs_dc"]
        print("PCR-0 vs a mis-placed base -> pre-registered win does NOT survive  OK\n"
              "   (arm |c-c*|=%.3f, ref+PCR |c-c*|=%.3f, ref+PCR-0 |c-c*|=%.3f)"
              % (P["abs_dev_endpoint"]["arm_uncorrected"], P["abs_dev_endpoint"]["ref_pcr"],
                 P["abs_dev_endpoint"]["ref_pcr0"]))

        # ---- the re-aim moves PLACEMENT, not the path ----------------------
        check_reaim_path_invariance()

        # a manual target is allowed but is recorded as manual, never as certified
        rm2 = run_cell(mis, MIS_ARM, out / "misplaced_manual.json",
                       extra=["--target_c", "0.0", "--n_boot", "0"])
        assert rm2["target"]["provenance"].startswith("--target_c"), rm2["target"]
        assert any(n.startswith("TARGET_MANUAL") for n in rm2["notes"]), rm2["notes"]
        assert rm2["decisive"]["bootstrap"]["status"] == "OFF", rm2["decisive"]["bootstrap"]
        print("manual --target_c recorded as manual, --n_boot 0 -> OFF not zero-width  OK")

        # ---- fail-loud -----------------------------------------------------
        v = variant(root, work, "bad_gap", [])
        p = v / "mth_critp_o1_s17_k1" / "pope_logits.jsonl"
        lines = p.read_text().splitlines()
        r0 = json.loads(lines[0])
        r0["gap"] = float("nan")
        lines[0] = json.dumps(r0)
        p.write_text("\n".join(lines) + "\n")
        try:
            run_cell(v, "mth_critp_o1_s17", out / "bad_gap.json")
        except SystemExit as e:
            assert "non-finite" in str(e), e
            print("fail-loud OK: non-finite gap ->", str(e)[:80])
        else:
            raise AssertionError("expected SystemExit for a non-finite gap")

        v = variant(root, work, "bad_gt", [])
        p = v / "mth_critp_o1_s17_k1" / "pope_logits.jsonl"
        lines = p.read_text().splitlines()
        r0 = json.loads(lines[0])
        r0["gt"] = "no" if r0["gt"] == "yes" else "yes"
        lines[0] = json.dumps(r0)
        p.write_text("\n".join(lines) + "\n")
        try:
            run_cell(v, "mth_critp_o1_s17", out / "bad_gt.json")
        except SystemExit as e:
            assert "gt mismatch" in str(e), e
            print("fail-loud OK: gt disagreement ->", str(e)[:80])
        else:
            raise AssertionError("expected SystemExit for a gt disagreement")

        v = variant(root, work, "short_dump", [])
        p = v / "mth_critp_o1_s17_k1" / "pope_logits.jsonl"
        p.write_text("\n".join(p.read_text().splitlines()[:100]) + "\n")
        try:
            run_cell(v, "mth_critp_o1_s17", out / "short.json")
        except SystemExit as e:
            assert "joined across base" in str(e) or "ids joined" in str(e), e
            print("fail-loud OK: truncated dump ->", str(e)[:80])
        else:
            raise AssertionError("expected SystemExit for a truncated dump")

        print("\nALL TESTS PASSED")
        ok = True
    finally:
        if ok:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print("\n[kept for debugging] %s" % work, file=sys.stderr)


if __name__ == "__main__":
    main()
