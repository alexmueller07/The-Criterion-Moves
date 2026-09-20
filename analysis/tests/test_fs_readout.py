"""End-to-end tests for the full-study readout pipeline.

  python analysis/tests/test_fs_readout.py [--keep] [--scratch DIR]

1. LOCAL layout: fs_aggregate on fullstudy/results_fs (the real fsL_seq_o1_s17
   cells) -> figures -> SOTA table; regression-checks the numbers against the
   committed analysis/diag/fs_aggregate.json (c/d'/CHAIR_i must be unchanged).
2. CLUSTER-LIKE layout: a synthetic tree from make_fs_fixtures.py; the scripts
   are run from <root>/code/analysis/ with PYTHONPATH UNSET and CLH_ROOT
   pointing at the tree (the exact cluster import path), then once more with
   every flag explicit and no CLH_ROOT. Asserts family parsing (fsL/psQ/lamF/
   mth/joint/single), gate_info backbone resolution, suite cross-check,
   in-progress / malformed / recoverable handling (exit code 3), endpoint
   blocks E1/E2/E3/E4/E5/F1, the scorecard, the bench block, and that figures
   and tables are written.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PY = sys.executable
FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def run(cmd, env=None, cwd=None, ok_codes=(0,)):
    print("$ " + " ".join(cmd))
    p = subprocess.run(cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True)
    tail = "\n".join(p.stdout.splitlines()[-25:])
    if p.returncode not in ok_codes:
        print(p.stdout[-6000:])
    check(p.returncode in ok_codes, "%s -> rc=%d (expected %s)" % (cmd[1].split("/")[-1],
                                                                  p.returncode, ok_codes))
    return p.returncode, p.stdout, tail


def test_local(scratch):
    print("\n=== 1. local layout: fullstudy/results_fs ===")
    out = os.path.join(scratch, "local")
    os.makedirs(out, exist_ok=True)
    agg = os.path.join(out, "fs_aggregate.json")
    rc, so, _ = run([PY, os.path.join(REPO, "analysis", "fs_aggregate.py"),
                     "--results", os.path.join(REPO, "fullstudy", "results_fs"),
                     "--coco_gt", os.path.join(REPO, "results", "coco_gt.json"),
                     "--out", agg, "--summary", os.path.join(out, "summary.txt"),
                     "--e3_boot", "2000"])
    if rc != 0:
        return
    with open(agg) as f:
        A = json.load(f)
    with open(os.path.join(REPO, "analysis", "diag", "fs_aggregate.json")) as f:
        OLD = json.load(f)
    L = A["backbones"]["llava15"]
    check("seq|o1|s17" in L["arms"], "compat: backbones.llava15.arms carries the UCIT seq cell")
    for k in ("1", "3", "6"):
        new, old = L["arms"]["seq|o1|s17"][k], OLD["backbones"]["llava15"]["arms"]["seq|o1|s17"][k]
        for fld in ("c", "dprime", "yes_rate", "H", "FA"):
            check(abs(new["pope"][fld] - old["pope"][fld]) < 1e-9,
                  "regression k%s pope.%s %.4f == %.4f" % (k, fld, new["pope"][fld], old["pope"][fld]))
        check(abs(new["chair"]["chair_i"] - old["chair"]["chair_i"]) < 1e-9,
              "regression k%s chair_i %.4f" % (k, new["chair"]["chair_i"]))
        for t in old["tasks"]:
            check(abs(new["tasks"][t]["acc"] - old["tasks"][t]["acc"]) < 1e-9,
                  "regression k%s task %s acc" % (k, t))
    check(abs(L["base"]["pope"]["c"] - OLD["backbones"]["llava15"]["base"]["pope"]["c"]) < 1e-9,
          "regression base c")
    check("f1" in L["base"]["pope"] and 0 < L["base"]["pope"]["f1"] < 1, "pope.f1 added")
    check("chair_i60" in L["base"]["chair"], "chair_i60 added")
    ep = L["endpoints"]
    check(ep["E1"]["mode"] == "null", "UCIT E1 mode is null (answer-stat-flat)")
    check(len(ep["E1_null"]["cells"]) == 1, "E1_null one seq cell")
    check(ep["F1"]["any_tripped"] is False and len(ep["F1"]["cells"]) == 1, "F1 evaluated, not tripped")
    check(ep["E3"]["n"] == 1 and ep["E3"]["cells"][0]["boot"] is not None, "E3 one cell with bootstrap")
    check(ep["E5"]["applies"] is False, "E5 not testable on UCIT")
    sc = L["suites"]["ucit"]["scorecard"]
    check(len(sc) == 1 and sc[0]["arm"] == "seq" and sc[0]["n_complete"] == 1, "scorecard: seq complete")
    check(abs(sc[0]["sum_abs_dc"]["mean"] - OLD["backbones"]["llava15"]["endpoints"]["E1_null"]["cells"][0]["sum_abs_step"]) < 1e-6,
          "scorecard Sum|dc| equals the old E1_null sum_abs_step")
    check(sc[0]["plasticity"] is not None and sc[0]["plasticity"]["n"] == 1, "plasticity present (closed-ended diag)")
    check(sc[0]["cells"][0].get("plasticity_n") == 4, "plasticity averaged over the 4 closed-ended UCIT stages")
    figs = os.path.join(out, "figures")
    run([PY, os.path.join(REPO, "analysis", "make_fs_figures.py"), "--agg", agg, "--out", figs])
    for n in ("fig_fs_generalization", "fig_fs_seed_consistency", "fig_fs_dprime_falsifier",
              "fig_fs_lamF_sweep", "fig_fs_scorecard", "fig_fs_mechanism_pilot"):
        check(os.path.isfile(os.path.join(figs, n + ".pdf")) and os.path.isfile(os.path.join(figs, n + ".png")),
              "figure %s written" % n)
    tabs = os.path.join(out, "tables")
    run([PY, os.path.join(REPO, "analysis", "make_sota_table.py"), "--agg", agg, "--out", tabs,
         "--results", os.path.join(REPO, "fullstudy", "results_fs")])
    check(os.path.isfile(os.path.join(tabs, "sota.tex")), "sota.tex written")
    with open(os.path.join(tabs, "sota.csv")) as f:
        rows = list(__import__("csv").DictReader(f))
    seq = next(r for r in rows if r["method"] == "SEQ")
    check(seq["pope_f1"] != "" and seq["chair_i"] != "" and seq["c_range"] != "" and seq["plasticity"] != "",
          "SEQ row has POPE-F1/CHAIR_i/c-range/plasticity")
    check(seq["runtag"] == "fsL_seq_o1_s17", "SEQ row runtag recorded")
    # no aggregate -> in-process recompute path
    run([PY, os.path.join(REPO, "analysis", "make_fs_figures.py"), "--agg", os.path.join(out, "nope.json"),
         "--results", os.path.join(REPO, "fullstudy", "results_fs"),
         "--coco_gt", os.path.join(REPO, "results", "coco_gt.json"), "--out", os.path.join(out, "figs_fallback")])
    check(os.path.isfile(os.path.join(out, "figs_fallback", "fig_fs_generalization.pdf")), "fallback figure path")


def test_cluster_like(scratch):
    print("\n=== 2. cluster-like synthetic tree ===")
    root = os.path.join(scratch, "cluster")
    run([PY, os.path.join(HERE, "make_fs_fixtures.py"), "--out", root])
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CLH_ROOT")}
    env["CLH_ROOT"] = root
    env["MPLBACKEND"] = "Agg"
    code = os.path.join(root, "code", "analysis")
    out = os.path.join(root, "readout")
    agg = os.path.join(out, "fs_aggregate.json")
    rc, so, _ = run([PY, os.path.join(code, "fs_aggregate.py"), "--out", agg,
                     "--summary", os.path.join(out, "summary.txt"), "--e3_boot", "3000"],
                    env=env, cwd=scratch, ok_codes=(3,))
    if rc != 3:
        return
    with open(agg) as f:
        A = json.load(f)
    check(A["layout"] == "cluster", "layout detected as cluster via CLH_ROOT")
    check(sorted(A["backbones"]) == ["llava15", "qwen25vl"], "both backbones present")
    L, Q = A["backbones"]["llava15"], A["backbones"]["qwen25vl"]
    check(sorted(L["suites"]) == ["ucit"], "llava15 has the ucit suite only")
    check(sorted(Q["suites"]) == ["pilot", "ucit"], "qwen25vl has pilot + ucit suites")
    sys.path.insert(0, HERE)
    import make_fs_fixtures as MF
    check(A["expected_counts"]["pope"] == MF.N_POPE and A["expected_counts"]["chair"] == MF.N_CHAIR,
          "expected row counts from pope_dir + coco_gt (%d / %d)" % (MF.N_POPE, MF.N_CHAIR))
    check(not any(c.get("audit", {}).get("row_count_mismatch")
                  for st in L["suites"]["ucit"]["arms"].values() for c in st.values()),
          "no row-count mismatch flagged on well-formed cells")
    check(A["expected_counts"]["tasks"].get("ArxivQA") == 30 and A["expected_counts"]["tasks"].get("textvqa") == 30,
          "expected task val counts from both manifests")
    # families
    ua = L["suites"]["ucit"]["arms"]
    check("lamF_0.02|o1|s17" in ua and ua["lamF_0.02|o1|s17"]["1"]["faith_weight"] == 0.02, "lamF weight parsed")
    check(ua["lamF_0.02|o1|s17"]["1"]["backbone_source"] == "gate_info", "lamF backbone from gate_info")
    check(sorted(int(k) for k in ua["joint|o1|s17"]) == [1, 2, 3, 4, 5, 6], "joint ckpt_step/final -> stages 1..6")
    check(ua["joint|o1|s17"]["6"]["kind"] == "final" and ua["joint|o1|s17"]["2"]["step"] == 200, "joint kinds/steps kept")
    check(ua["single_ArxivQA|o1|s17"]["1"]["task_trained"] == "ArxivQA", "single-task control labelled")
    check("er500|o1|s17" in ua and "cecf|o1|s17" in ua, "er500 / cecf arms parsed")
    pa = Q["suites"]["pilot"]["arms"]
    check("mth_critp|o1|s17" in pa and pa["mth_critp|o1|s17"]["1"]["backbone"] == "qwen25vl",
          "mth_ candidate on Qwen via gate_info, pilot suite")
    check(pa["seq|o1|s17"]["2"]["tasks"]["textvqa"]["metric"] == "vqa_acc"
          and pa["seq|o1|s17"]["1"]["tasks"]["scienceqa"]["metric"] == "mc_acc"
          and pa["seq|o1|s17"]["3"]["tasks"]["flickr"]["metric"] == "caption_uf1",
          "pilot suite scored with the audited pilot scorers")
    check(ua["seq|o1|s17"]["1"]["tasks"]["ArxivQA"]["metric"] == "norm-containment-EM", "UCIT containment-EM")
    check(sorted(int(k) for k in Q["suites"]["ucit"]["arms"]["seq|o1|s17"]) == [1, 2], "fsQ partial cell kept")
    # discipline
    check("fsL_anchor_o1_s23_k3" in A["in_progress"], "in-progress cell skipped")
    check(any(m["dir"] == "fsL_ewc_o1_s17_k2" for m in A["malformed"]), "malformed cell listed + exit 3")
    check("ewc|o1|s17" in ua and "2" not in ua["ewc|o1|s17"], "malformed cell excluded, siblings kept")
    check(ua["seq|o1|s23"]["4"]["audit"].get("recovered_lines") == 1, "recoverable corruption recovered + audited")
    check("scratch_tmp" in A["unparsed"], "unparsed dir reported")
    # endpoints
    ep = L["suites"]["ucit"]["endpoints"]
    check(ep["E1"]["mode"] == "null" and len(ep["E1_null"]["cells"]) == 3, "UCIT E1-null over 3 seq cells")
    check(ep["E2"]["n_matched"] == 2 and ep["E2"]["n_freeze"] == 2, "E2: 2 matched cells, anchor<seq in both")
    check(ep["E2"]["pooled_suppression_ratio"] > 1, "E2 pooled suppression ratio > 1")
    check(ep["E3"]["n"] == 3 and ep["E3"]["n_positive"] >= 2, "E3: 3 seq cells, mid-sequence rise positive")
    check(all(c["boot"]["impl"] in ("numpy", "stdlib") for c in ep["E3"]["cells"]), "E3 bootstrap ran")
    check(len(ep["E4"]["cells"]) == 2 and ep["E4"]["n_bands"] >= 2, "E4: two controls with seq bands")
    b0 = ep["E4"]["cells"][0]["bands"][0]
    check(b0["seq_stage"] == 1 and b0["seeds"] == [17, 23] and b0["inside"] is not None
          and b0["band"][0] - 0.05 <= ep["E4"]["cells"][0]["c"] <= b0["band"][1] + 0.05,
          "E4: ArxivQA control matched to seq stage 1 over seeds 17/23, c at the band")
    b1 = next(b for c in ep["E4"]["cells"] if c["task"] == "IconQA" for b in c["bands"] if b["order"] == "o1")
    check(b1["seq_stage"] == 4 and b1["seq_endpoint_band"] is not None, "E4: IconQA control matched to seq stage 4")
    check(ep["F1"]["any_tripped"] is False, "F1 not tripped on synthetic d'")
    pe = Q["suites"]["pilot"]["endpoints"]
    check(pe["E1"]["mode"] == "dose-response" and pe["E1"]["active_tasks"] == ["textvqa", "vizwiz"],
          "pilot E1 dose-response with active textvqa/vizwiz")
    check(pe["E5"]["applies"] and pe["E5"]["refusal_tasks"] == ["vizwiz"], "pilot E5 applies (VizWiz refusal)")
    check(any(c["excess"].get("textvqa", 0) > 0 for c in pe["E5"]["cells"]), "E5 leakage excess measured")
    check(pe["E2"]["n_matched"] == 2, "pilot E2 matched 2 cells")
    check("pilot" in A["E1_cross_backbone"], "cross-backbone E1 tally present")
    # scorecard
    sc = {r["arm"]: r for r in L["suites"]["ucit"]["scorecard"]}
    check(set(sc) >= {"seq", "anchor", "ewc", "cecf", "er500", "joint", "lamF_0.02", "lamF_0.4"},
          "scorecard rows for every non-control arm")
    check("single_ArxivQA" not in sc, "single-task controls excluded from the scorecard")
    check(sc["seq"]["n_complete"] == 3 and sc["anchor"]["n_complete"] == 2 and sc["anchor"]["n_cells"] == 3,
          "scorecard complete/partial counts")
    check(sc["anchor"]["vs_seq"]["n_matched"] == 2 and sc["anchor"]["vs_seq"]["sum_abs_dc_lower"] == [2, 2],
          "anchor vs seq paired sign count 2/2")
    check(sc["ewc"]["n_complete"] == 0 and sc["ewc"]["sum_abs_dc"] is None, "ewc (malformed stage) has no complete cell")
    check(sc["joint"]["plasticity"] is None and sc["joint"]["endpoint_task_mean"] is not None, "joint: endpoint mean, no plasticity")
    check(sc["lamF_0.02"]["sum_abs_dc"]["mean"] > sc["lamF_0.4"]["sum_abs_dc"]["mean"], "lamF: weaker anchor drifts more")
    check(sc["seq"]["plasticity"] is not None and sc["seq"]["max_abs_dd"] is not None, "seq plasticity + d' maxdev")
    # bench
    check(A["bench"]["fsL_seq_o1_s17_bench"]["mme"]["score"] == 590.0 and A["bench"]["fsL_seq_o1_s17_bench"]["done"],
          "bench block: mme + done")
    check(A["bench"]["fsL_ewc_o1_s17_bench"]["mme"]["full_800_scale"] is False, "bench: partial MME flagged")
    check(A["bench"]["fsL_anchor_o1_s17_bench"]["noncoco"] is None, "bench: absent noncoco -> None")
    with open(os.path.join(out, "summary.txt")) as f:
        s = f.read()
    check("METHOD SCORECARD" in s and "MALFORMED" in s and "[E2]" in s and "[E3]" in s, "summary carries scorecard + banner")
    # figures + tables from the cluster tree, cluster import path
    run([PY, os.path.join(code, "make_fs_figures.py"), "--agg", agg, "--out", os.path.join(out, "figures")],
        env=env, cwd=scratch)
    for n in ("fig_fs_generalization", "fig_fs_mechanism_pilot", "fig_fs_seed_consistency",
              "fig_fs_dprime_falsifier", "fig_fs_lamF_sweep", "fig_fs_scorecard"):
        check(os.path.isfile(os.path.join(out, "figures", n + ".pdf")), "cluster figure %s" % n)
    run([PY, os.path.join(code, "make_sota_table.py"), "--agg", agg, "--out", os.path.join(out, "tables")],
        env=env, cwd=scratch)
    check(os.path.isfile(os.path.join(out, "tables", "sota.tex")), "cluster sota.tex")
    check(os.path.isfile(os.path.join(out, "tables", "sota_qwen25vl_pilot.tex")), "extra table qwen/pilot")
    with open(os.path.join(out, "tables", "sota.csv")) as f:
        rows = {r["method"]: r for r in __import__("csv").DictReader(f)}
    check(rows["SEQ"]["mme_hall"] == "590.0" and rows["SEQ"]["noncoco_chair"] != "", "sota SEQ bench columns")
    check(rows["EWC"]["mme_hall"] == "600.0" and "mme_partial" in rows["EWC"]["notes"], "sota EWC partial MME flagged")
    check("anchor-v2 (\\lambda_F=0.02)".replace("\\lambda_F", "$\\lambda_F$") in rows or any("0.02" in m for m in rows),
          "lamF extra rows present")
    check(rows["anchor-v2"]["objhal_chair"] == "0.21" and "bench_in_progress" in rows["anchor-v2"]["notes"],
          "anchor bench in progress flagged")
    # explicit-flag path, no CLH_ROOT, cwd elsewhere
    env2 = {k: v for k, v in env.items() if k != "CLH_ROOT"}
    out2 = os.path.join(scratch, "explicit")
    rc, so, _ = run([PY, os.path.join(REPO, "analysis", "fs_aggregate.py"),
                     "--results", os.path.join(root, "results_fs"),
                     "--bench", os.path.join(root, "results_fs_bench"),
                     "--coco_gt", os.path.join(root, "data", "chair", "coco_gt.json"),
                     "--pope_dir", os.path.join(root, "data", "pope"),
                     "--manifest_ucit", os.path.join(root, "data_ucit", "ucit_manifest.json"),
                     "--manifest_pilot", os.path.join(root, "data", "pilot_manifest.json"),
                     "--out", os.path.join(out2, "agg.json"), "--e3_boot", "0", "--no_matrix"],
                    env=env2, cwd=tempfile.gettempdir(), ok_codes=(3,))
    if rc == 3:
        with open(os.path.join(out2, "agg.json")) as f:
            A2 = json.load(f)
        check(A2["backbones"]["llava15"]["suites"]["ucit"]["scorecard"][0]["sum_abs_dc"] ==
              L["suites"]["ucit"]["scorecard"][0]["sum_abs_dc"], "explicit flags reproduce the CLH_ROOT run")
        check(all(c["boot"] is None for c in A2["backbones"]["llava15"]["suites"]["ucit"]["endpoints"]["E3"]["cells"]),
              "--e3_boot 0 skips the bootstrap")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--scratch", default=None)
    args = ap.parse_args()
    scratch = args.scratch or tempfile.mkdtemp(prefix="clh_readout_test_")
    os.makedirs(scratch, exist_ok=True)
    print("scratch: %s" % scratch)
    test_local(scratch)
    test_cluster_like(scratch)
    print("\n%s: %d failure(s)" % ("PASS" if not FAILS else "FAIL", len(FAILS)))
    for f in FAILS:
        print("  - " + f)
    if not args.keep and not args.scratch and not FAILS:
        shutil.rmtree(scratch, ignore_errors=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
