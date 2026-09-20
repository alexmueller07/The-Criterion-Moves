#!/usr/bin/env python3
"""Synthetic end-to-end test for analysis/method_scorecard.py.

Builds an OBVIOUSLY FAKE results_fs tree under analysis/tests/fixtures/
results_scorecard_synth/ (marker file inside; regenerated on every run):

  qwen25vl_base/                       stage 0        H=.80 FA=.10 -> c=+0.220, d'=2.123
  psQ_seq_o1_s17_k1..k4  (ref)         c swings 0 / .696 / 0 / 1.027 ; leak 20% at k4;
                                       CHAIR@60 endpoint 0.200; k3 hides a hallucination
                                       AFTER word 60 (full 0.333 vs @60 0.000)
  psQ_seq_o1_s23_k1..k4  (ref)         same, but k2 trips the d' falsifier (d'=3.29)
  psQ_seq_o2_s31_k1..k4  (ref)         complete, no candidate match -> unmatched
  mth_critp_o1_s17_k1..k4 (candidate)  flat c=+0.284 ; leak 4% ; CHAIR@60 0.048 ;
                                       plasticity ref-0.01 (s17) / ref+0.03 (s23)
  mth_critp_o1_s23_k1..k4 (candidate)
  mth_critp_o2_s17_k1..k3 + k4 WITHOUT EVAL_DONE  -> incomplete, skipped
  psQ_joint_o1_s17_final, junk/         ignored (not staged / not ours)

Asserts discovery, every per-stage metric against hand-computed values,
pairing (n_matched=2, unmatched ref cell), and PASS on all four predictions
for mth_critp vs psQ_seq; then swaps ref/candidate and asserts P1..P4 all FAIL.
If fullstudy/results_fs/llava15_base exists, also runs the ucit suite on the
real partial tree and checks graceful ABSENT handling (leak None, missing
anchor arm -> pairing None, verdicts ABSENT).

Run:  python3 analysis/tests/test_method_scorecard.py
"""
import json
import os
import shutil
import sys
from pathlib import Path
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import method_scorecard as MS  # noqa: E402

FIX = HERE / "fixtures" / "results_scorecard_synth"
OUT = HERE / "out" / "scorecard_synth"
N_IMG = 80
N_TASK = 50
Z = NormalDist().inv_cdf
ORDERS = {"o1": ["scienceqa", "textvqa", "flickr", "vizwiz"],
          "o2": ["vizwiz", "flickr", "textvqa", "scienceqa"]}


def jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def pope_rows(H, FA, n=100, extra_fail=0):
    rows = []
    i = 0
    for gt, rate in (("yes", H), ("no", FA)):
        nyes = round(rate * n)
        for j in range(n):
            out = "Yes" if j < nyes else "No"
            rows.append({"id": "pope_adversarial_%d" % i, "prompt": "Is there a dog?", "gt": gt,
                         "category": "adversarial", "meta": {}, "output": out,
                         "n_new_tokens": 2, "truncated": False})
            i += 1
    for j in range(extra_fail):
        rows.append({"id": "pope_fail_%d" % j, "prompt": "?", "gt": "yes", "category": "adversarial",
                     "meta": {}, "output": "Maybe", "n_new_tokens": 2, "truncated": False})
    return rows


def chair_rows(frac_hal, tokens, late=False):
    rows = []
    n_hal = round(frac_hal * N_IMG)
    for i in range(N_IMG):
        base = "A dog sits next to a cat."
        if late:
            text = base + " " + " ".join(["very"] * 60) + " and a bicycle."
        elif i < n_hal:
            text = "A dog sits next to a cat near a table."
        else:
            text = base
        rows.append({"id": "chair_%d" % (1000 + i), "coco_id": 1000 + i,
                     "prompt": "Please describe this image in detail.", "output": text,
                     "n_new_tokens": tokens, "truncated": False})
    return rows


def task_rows(task, acc, leak=0.0):
    rows = []
    n_ok = round(acc * N_TASK)
    n_leak = round(leak * N_TASK)
    for i in range(N_TASK):
        if task == "scienceqa":
            meta, good, bad = {"answer_letter": "C"}, "C", "A"
        elif task in ("textvqa", "vizwiz"):
            meta, good, bad = {"answers": ["media"] * 10}, "media", "wrong"
        else:
            meta, good, bad = {"refs": ["a man rides a horse"]}, "a man rides a horse", "zzz qqq"
        if i < n_ok:
            out = good
        elif i < n_ok + n_leak:
            out = "Unanswerable"
        else:
            out = bad
        rows.append({"id": "%s_%d" % (task, i), "prompt": "q", "target": good, "meta": meta,
                     "output": out, "n_new_tokens": 3, "truncated": False})
    return rows


def write_cell(name, H, FA, chair_hal, tokens, accs, leak=0.0, late=False, done=True,
               backbone="qwen25vl", extra_fail=0):
    d = FIX / name
    d.mkdir(parents=True, exist_ok=True)
    jsonl(d / "pope_gen.jsonl", pope_rows(H, FA, extra_fail=extra_fail))
    jsonl(d / "chair_gen.jsonl", chair_rows(chair_hal, tokens, late))
    for t in ORDERS["o1"]:
        jsonl(d / ("%s_gen.jsonl" % t), task_rows(t, accs[t], leak if t == "textvqa" else 0.0))
    json.dump({"argv": ["eval_gen.py", "--backbone", backbone, "--out", name]}, open(d / "gate_info.json", "w"))
    if done:
        (d / "EVAL_DONE").touch()


def sdt(H, FA):
    return -0.5 * (Z(H) + Z(FA)), Z(H) - Z(FA)


REF_HF = [(0.85, 0.15), (0.60, 0.05), (0.85, 0.15), (0.50, 0.02)]
REF_HF_TRIP = [(0.85, 0.15), (0.95, 0.05), (0.85, 0.15), (0.50, 0.02)]
CAND_HF = [(0.78, 0.09)] * 4
REF_ACC = {"scienceqa": 0.80, "textvqa": 0.60, "flickr": 0.50, "vizwiz": 0.70}


def build_tree():
    if FIX.exists():
        shutil.rmtree(FIX)
    FIX.mkdir(parents=True)
    (FIX / "THIS_IS_SYNTHETIC_FIXTURE_DATA.txt").write_text("fake results for test_method_scorecard.py\n")
    gt = {str(1000 + i): {"objects": ["dog", "cat"], "captions": ["a dog and a cat"]} for i in range(N_IMG)}
    json.dump(gt, open(FIX / "coco_gt.json", "w"))
    write_cell("qwen25vl_base", 0.80, 0.10, 0.5, 100, REF_ACC, extra_fail=2)
    # ref cells
    for seed, hf in ((17, REF_HF), (23, REF_HF_TRIP)):
        for k in range(1, 5):
            write_cell("psQ_seq_o1_s%d_k%d" % (seed, k), hf[k - 1][0], hf[k - 1][1],
                       0.5, 100 + k, REF_ACC, leak=0.2 if k == 4 else 0.0, late=(k == 3))
    for k in range(1, 5):
        write_cell("psQ_seq_o2_s31_k%d" % k, REF_HF[k - 1][0], REF_HF[k - 1][1], 0.5, 100, REF_ACC)
    # candidate cells
    for seed, delta in ((17, -0.01), (23, +0.03)):
        accs = {t: round(v + delta, 4) for t, v in REF_ACC.items()}
        for k in range(1, 5):
            write_cell("mth_critp_o1_s%d_k%d" % (seed, k), CAND_HF[k - 1][0], CAND_HF[k - 1][1],
                       0.1, 90, accs, leak=0.04 if k == 4 else 0.0)
    for k in range(1, 5):
        write_cell("mth_critp_o2_s17_k%d" % k, 0.78, 0.09, 0.1, 90, REF_ACC, done=(k < 4))
    write_cell("psQ_joint_o1_s17_final", 0.8, 0.1, 0.5, 100, REF_ACC)
    (FIX / "junk").mkdir()


def run(arms, ref, out):
    argv = ["--results", str(FIX), "--suite", "pilot", "--arms", arms, "--ref", ref,
            "--coco_gt", str(FIX / "coco_gt.json"), "--manifest",
            str(ROOT / "fullstudy" / "pilot_manifest.json"), "--out", str(out), "--B", "2000"]
    rc = MS.main(argv)
    assert rc == 0
    return json.load(open(Path(out) / "scorecard.json"))


def close(a, b, tol=2e-3):
    assert a is not None and abs(a - b) <= tol, (a, b)


def main():
    build_tree()
    rep = run("psQ_seq,mth_critp", "psQ_seq", OUT)

    # ---- discovery ----
    seq, cand = rep["arms"]["psQ_seq"], rep["arms"]["mth_critp"]
    assert seq["complete_cells"] == [["o1", 17], ["o1", 23], ["o2", 31]], seq["complete_cells"]
    assert cand["complete_cells"] == [["o1", 17], ["o1", 23]], cand["complete_cells"]
    assert cand["incomplete"] and cand["incomplete"][0][0] == ["o2", 17] and "missing" in cand["incomplete"][0][1]
    assert rep["header"]["base_present"] and rep["header"]["base_dir"].endswith("qwen25vl_base")
    assert cand["backbones_seen"] == ["qwen25vl"]
    print("discovery OK")

    # ---- per-stage metrics ----
    c0, d0 = sdt(0.80, 0.10)
    close(rep["base"]["pope"]["c"], c0)
    close(rep["base"]["pope"]["dprime"], d0)
    assert rep["base"]["pope"]["parse_fail"] == 2 and rep["base"]["pope"]["n_parsed"] == 200
    cell = seq["cells"]["o1/s17"]
    for k in range(1, 5):
        c, d = sdt(*REF_HF[k - 1])
        st = cell["stages"][str(k)]
        close(st["pope"]["c"], c)
        close(st["pope"]["dprime"], d)
        assert st["task_learned"] == ORDERS["o1"][k - 1]
        close(st["new_task"]["acc"], REF_ACC[ORDERS["o1"][k - 1]])
        close(st["chair"]["mean_new_tokens"], 100 + k, 0.05)
    close(cell["stages"]["4"]["leak"]["rate"], 0.20)
    close(cell["stages"]["1"]["leak"]["rate"], 0.0)
    # length control: k3 hallucination sits after word 60
    st3 = cell["stages"]["3"]
    close(st3["chair"]["chair_i"], 80 / 240)
    close(st3["chair"]["chair_i_at_k"], 0.0)
    assert st3["chair"]["frac_over_budget"] == 1.0
    close(cell["stages"]["4"]["chair"]["chair_i_at_k"], 40 / 200)
    close(cand["cells"]["o1/s17"]["stages"]["4"]["chair"]["chair_i_at_k"], 8 / 168)
    assert cell["stages"]["1"]["new_task"]["metric"] == "mc_acc"
    assert cell["stages"]["3"]["new_task"]["metric"] == "caption_uf1"
    print("per-stage metrics OK")

    # ---- scorecard ----
    cs = [c0] + [sdt(*hf)[0] for hf in REF_HF]
    close(cell["scorecard"]["sum_abs_dc"], sum(abs(cs[i + 1] - cs[i]) for i in range(4)))
    close(cell["scorecard"]["endpoint_c"], cs[-1])
    close(cell["scorecard"]["max_abs_dd"], max(abs(sdt(*hf)[1] - d0) for hf in REF_HF))
    assert cell["scorecard"]["f1_tripped"] is False
    assert seq["cells"]["o1/s23"]["scorecard"]["f1_tripped"] is True
    close(cell["scorecard"]["mean_plasticity"], sum(REF_ACC.values()) / 4)
    cc = cand["cells"]["o1/s17"]["scorecard"]
    close(cc["sum_abs_dc"], abs(sdt(0.78, 0.09)[0] - c0))
    close(cc["mean_plasticity"], sum(REF_ACC.values()) / 4 - 0.01)
    close(cc["endpoint_leak"], 0.04)
    assert seq["summary"]["sum_abs_dc"]["n"] == 3 and seq["summary"]["f1_n_tripped"] == 1
    print("scorecard OK")

    # ---- pairing ----
    pr = cand["pairing"]
    assert pr["n_matched"] == 2 and pr["unmatched_ref_cells"] == [["o2", 31]], pr["matched"]
    r17 = [r for r in pr["rows"] if r["seed"] == 17][0]
    close(r17["d_endpoint_chair_at_k"], 8 / 168 - 40 / 200)
    assert r17["chair_at_k_boot"]["n_pairs"] == N_IMG and r17["chair_at_k_boot"]["ci95"][1] < 0
    close(r17["d_mean_plasticity"], -0.01)
    r23 = [r for r in pr["rows"] if r["seed"] == 23][0]
    close(r23["d_mean_plasticity"], +0.03)
    assert pr["signs"]["sum_abs_dc_lower"] == {"n": 2, "count": 2}
    assert pr["signs"]["chair_at_k_ci_excl0_lower"] == {"n": 2, "count": 2}
    assert pr["signs"]["plasticity_within_tol"] == {"n": 2, "count": 2}
    assert pr["signs"]["plasticity_abs_within_tol"] == {"n": 2, "count": 1}
    print("pairing OK")

    # ---- PASS/FAIL ----
    v = cand["verdicts"]
    assert [v[p]["verdict"] for p in ("P1", "P2", "P3", "P4")] == ["PASS"] * 4, v
    assert v["overall"] == "PASS"
    assert seq["verdicts"]["P3"]["verdict"] == "FAIL" and seq["verdicts"]["P1"]["verdict"] == "ABSENT"
    for f in ("scorecard.md", "scorecard.json", "scorecard.png"):
        assert (OUT / f).is_file(), f
    md = (OUT / "scorecard.md").read_text()
    assert "| mth_critp | PASS (2/2) | PASS (2/2) | PASS (2/2 ok) | PASS (2/2) | **PASS** |" in md, md
    print("PASS path OK")

    # swap: the reference becomes the candidate -> every prediction must FAIL
    rep2 = run("mth_critp,psQ_seq", "mth_critp", OUT.parent / "scorecard_synth_swap")
    v2 = rep2["arms"]["psQ_seq"]["verdicts"]
    assert [v2[p]["verdict"] for p in ("P1", "P2", "P3", "P4")] == ["FAIL"] * 4, v2
    assert v2["P1"]["count"] == 0 and v2["P2"]["count"] == 1 and v2["P4"]["count"] == 0
    assert v2["overall"] == "FAIL"
    print("FAIL path OK")

    # ---- real partial tree (ucit): graceful ABSENT handling ----
    fs = ROOT / "fullstudy" / "results_fs"
    if (fs / "llava15_base" / "EVAL_DONE").exists():
        out3 = OUT.parent / "scorecard_fs_partial_test"
        rc = MS.main(["--results", str(fs), "--suite", "ucit", "--arms", "fsL_seq,fsL_anchor",
                      "--ref", "fsL_seq", "--coco_gt", str(ROOT / "results" / "coco_gt.json"),
                      "--out", str(out3), "--B", "200"])
        assert rc == 0
        r3 = json.load(open(out3 / "scorecard.json"))
        assert r3["header"]["leak_task"] is None
        anc = r3["arms"]["fsL_anchor"]
        assert anc["complete_cells"] == [] and anc["pairing"] is None
        assert anc["verdicts"]["overall"] == "INCOMPLETE"
        assert all(anc["verdicts"][p]["verdict"] == "ABSENT" for p in ("P1", "P2", "P3", "P4"))
        for cn, c in r3["arms"]["fsL_seq"]["cells"].items():
            assert c["scorecard"]["endpoint_leak"] is None
            assert c["scorecard"]["sum_abs_dc"] is not None
            assert len(c["stages"]) == 6
        print("real partial tree (ucit) ABSENT handling OK")
    else:
        print("real partial tree not present; skipped")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
