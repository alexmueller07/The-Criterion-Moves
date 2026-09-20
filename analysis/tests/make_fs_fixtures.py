"""Build a SYNTHETIC cluster-like tree for testing the full-study readout
pipeline (fs_aggregate -> make_fs_figures -> make_sota_table) end to end,
including the cluster import path (script under <root>/code/analysis/, pilot
scorers flat in <root>/code/).

Everything written is OBVIOUSLY FAKE (patterned values, 120-row POPE, 40-image
CHAIR) and lives only where --out points (default: a temp dir). Layout mirrors
vgi2:~/cl-halluc exactly:

  <root>/results_fs/<cell>/{pope_gen,chair_gen,<task>_gen}.jsonl gate_info.json EVAL_DONE
  <root>/results_fs_bench/<label>/{objhal,mme,noncoco}.json BENCH_DONE
  <root>/data/{pope/prompts.jsonl, chair/coco_gt.json, pilot_manifest.json, tasks/<t>/val.jsonl}
  <root>/data_ucit/{ucit_manifest.json, tasks/<T>/val.jsonl}
  <root>/code/{metrics_pope,metrics_chair,metrics_task}.py   <root>/code/analysis/*.py

Cells (families: fsL_ 6-stage UCIT, psQ_ 4-stage pilot, lamF_, mth_, joint,
single-task controls, an in-progress cell, a MALFORMED cell, a cell with the
one recoverable corruption, an unparsed dir) are listed in CELLS below with
the patterned criterion trajectory each one follows.

Usage:  python analysis/tests/make_fs_fixtures.py [--out DIR]
"""
import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from statistics import NormalDist

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
Phi = NormalDist().cdf

UCIT = ["ArxivQA", "CLEVR-Math", "Flickr30k", "IconQA", "ImageNet-R", "VizWiz"]
PILOT = ["scienceqa", "textvqa", "flickr", "vizwiz"]
N_POPE = 300         # 150 gt=yes / 150 gt=no over 3 categories (exact counts, no draws)
N_CHAIR = 60
N_TASK = 30
OBJS = ["dog", "cat", "chair", "car", "person", "bottle", "cup", "bench"]
BASE_C, BASE_D = 0.30, 2.30


def w(path, obj=None, rows=None, text=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        if rows is not None:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        elif text is not None:
            f.write(text)
        else:
            json.dump(obj, f, indent=1)


# ------------------------------------------------------------ trajectories
def traj_c(arm, k, seed, fw=None):
    """Patterned criterion c at stage k (1-based) for an arm token."""
    j = 0.01 * ((seed * 7 + k * 3) % 5 - 2)      # small deterministic seed jitter
    if arm == "seq":
        return BASE_C - 0.15 * k + 0.05 * (k % 2) + j
    if arm == "anchor":
        return BASE_C + 0.35 + 0.02 * (k % 2) + j
    if arm.startswith("lamF"):
        return BASE_C + 0.35 - (0.1 / fw) * 0.04 * k + j     # drift ~ 1/lambda_F
    if arm == "joint":
        return BASE_C - 0.03 * k + j
    if arm in ("er", "er500", "ewc", "lwf"):
        return BASE_C - 0.12 * k + 0.04 * (k % 2) + j
    if arm == "cecf":
        return BASE_C + 1.2 + 0.1 * (k % 2) + j
    if arm.startswith("mth"):
        return BASE_C + 0.10 + 0.01 * (k % 2) + j
    if arm.startswith("single"):
        return BASE_C - 0.15 * k + 0.05 * (k % 2)   # == seq stage k, seed-jitter free
    return BASE_C


def p_hal(arm, k, T):
    """CHAIR hallucination probability per caption (rises mid-sequence for SEQ)."""
    if arm == "seq":
        return 0.15 + 0.10 * min(k, T - 1) / (T - 1)
    if arm in ("anchor",) or arm.startswith("lamF") or arm.startswith("mth"):
        return 0.12
    return 0.18


# ------------------------------------------------------------ gen writers
def write_pope(path, c, d, rng, corrupt=None):
    """Exact hit / false-alarm COUNTS for the target (c, d'), so the realized
    criterion matches the pattern to rate quantization (1/150), not to noise."""
    H, FA = Phi(d / 2 - c), Phi(-d / 2 - c)
    n_yes = N_POPE // 2
    n_hits, n_fas = int(round(H * n_yes)), int(round(FA * n_yes))
    rows = []
    cats = ["adversarial", "popular", "random"]
    seen_yes = seen_no = 0
    for i in range(N_POPE):
        gt = "yes" if i % 2 == 0 else "no"
        if gt == "yes":
            say_yes = seen_yes < n_hits
            seen_yes += 1
        else:
            say_yes = seen_no < n_fas
            seen_no += 1
        rows.append({"id": "pope_%s_%d" % (cats[i % 3], i), "prompt": "Is there a dog?",
                     "gt": gt, "category": cats[i % 3], "output": "Yes" if say_yes else "No",
                     "n_new_tokens": 2, "truncated": False})
    if corrupt == "recoverable":
        # a partial record left by a killed job with a full record appended on the same line
        lines = [json.dumps(r) for r in rows]
        lines[5] = '{"id": "pope_popular_5", "prompt": "Is th' + lines[5]
        w(path, text="\n".join(lines) + "\n")
        return
    if corrupt == "malformed":
        lines = [json.dumps(r) for r in rows]
        lines[7] = '{"id": "broken", this is not json}'
        w(path, text="\n".join(lines) + "\n")
        return
    w(path, rows=rows)


def write_chair(path, gt, p, rng, long_tail=True):
    """Exactly round(p * N) captions hallucinate (a fixed image set); every
    other one of those puts the hallucination past the 60-word budget, so
    CHAIR_i@60 < CHAIR_i and both rise deterministically with p."""
    rows = []
    n_hal = int(round(p * len(gt)))
    for j, cid in enumerate(sorted(gt)):
        objs = gt[cid]["objects"]
        out = "The image shows a %s and a %s." % (objs[0], objs[-1])
        if j < n_hal:
            other = [o for o in OBJS if o not in objs]
            filler = (" and so on" * 40) if (long_tail and j % 2 == 1) else ""
            out += filler + " There is also a %s near a %s." % (other[0], other[1])
        rows.append({"id": "chair_%s" % cid, "coco_id": int(cid), "prompt": "Describe.",
                     "output": out, "n_new_tokens": len(out.split()), "truncated": False})
    w(path, rows=rows)


def write_task_ucit(path, task, acc, rng):
    rows = []
    for i in range(N_TASK):
        ans = "A" if task == "ArxivQA" else str(i % 7)
        if task in ("Flickr30k", "VizWiz"):
            ans = "a person standing next to a %s" % OBJS[i % len(OBJS)]
            out = "a photo of something"
        else:
            out = ans if rng.random() < acc else "Z"
        rows.append({"id": "%s_va%d" % (task, i), "prompt": "q", "target": ans,
                     "meta": {"answers": [ans]}, "output": out, "n_new_tokens": 2,
                     "truncated": False})
    w(path, rows=rows)


def write_task_pilot(path, task, acc, leak, rng):
    rows = []
    for i in range(N_TASK):
        if task == "scienceqa":
            r = {"meta": {"answer_letter": "B"}, "output": "B" if rng.random() < acc else "C"}
        elif task == "flickr":
            refs = ["a man rides a bike down the road", "someone on a bicycle", "a cyclist",
                    "a person biking", "a man on a bike"]
            r = {"meta": {"refs": refs},
                 "output": "a man rides a bike" if rng.random() < acc else "green field"}
        else:
            answers = ["media"] * 8 + ["plaza"] * 2
            if task == "vizwiz":
                answers = ["unanswerable"] * 6 + ["cake"] * 4
            x = rng.random()
            out = answers[0] if x < acc else ("Unanswerable" if x < acc + leak else "foo")
            r = {"meta": {"answers": answers}, "output": out}
        r.update({"id": "%s_%d" % (task, i), "prompt": "q", "target": "t", "n_new_tokens": 2,
                  "truncated": False})
        rows.append(r)
    w(path, rows=rows)


def gate_info(path, backbone, cell):
    w(path, obj={"host": "synthetic", "argv": ["eval_gen.py", "--backbone", backbone, "--out",
                                               "results_fs/%s/pope_gen.jsonl" % cell],
                 "slurm_job_id": "0"})


# ------------------------------------------------------------ cell table
# (dirname, arm_token, backbone, suite, k, order, seed, faith_weight, flags)
def cells():
    C = []

    def seqarm(prefix, arm, order, seed, T, bb, suite, fw=None, stages=None):
        # RUNTAG token = the arm minus its family prefix (lamF_0.02 -> 0.02, mth_critp -> critp)
        tok = arm.split("_", 1)[1] if arm.startswith(("lamF_", "mth_")) else arm
        for k in (stages or range(1, T + 1)):
            C.append(("%s_%s_%s_s%d_k%d" % (prefix, tok, order, seed, k),
                      arm, bb, suite, k, order, seed, fw, set()))
    # LLaVA UCIT core
    for seed in (17, 23):
        seqarm("fsL", "seq", "o1", seed, 6, "llava15", "ucit")
    seqarm("fsL", "seq", "o2", 17, 6, "llava15", "ucit")
    seqarm("fsL", "anchor", "o1", 17, 6, "llava15", "ucit")
    seqarm("fsL", "anchor", "o1", 23, 6, "llava15", "ucit", stages=[1, 2, 3])   # partial
    seqarm("fsL", "anchor", "o2", 17, 6, "llava15", "ucit")
    seqarm("fsL", "ewc", "o1", 17, 6, "llava15", "ucit")
    seqarm("fsL", "cecf", "o1", 17, 6, "llava15", "ucit")
    seqarm("fsL", "er500", "o1", 17, 6, "llava15", "ucit")
    # joint: ckpt_step cells + final
    for i, step in enumerate([100, 200, 300, 400, 500]):
        C.append(("fsL_joint_o1_s17_ckpt_step%d" % step, "joint", "llava15", "ucit", i + 1,
                  "o1", 17, None, set()))
    C.append(("fsL_joint_o1_s17_final", "joint", "llava15", "ucit", 6, "o1", 17, None, set()))
    # single-task controls
    C.append(("fsL_single_ArxivQA_o1_s17_k1", "single_ArxivQA", "llava15", "ucit", 1, "o1", 17, None, set()))
    C.append(("fsL_single_IconQA_o1_s17_k1", "single_IconQA", "llava15", "ucit", 4, "o1", 17, None, set()))
    # lamF sweep (LLaVA, UCIT; backbone via gate_info)
    for fw in (0.02, 0.4):
        seqarm("lamF", "lamF_%s" % fw, "o1", 17, 6, "llava15", "ucit", fw=fw)
    # Qwen pilot suite
    seqarm("psQ", "seq", "o1", 17, 4, "qwen25vl", "pilot")
    seqarm("psQ", "seq", "o1", 23, 4, "qwen25vl", "pilot")
    seqarm("psQ", "anchor", "o1", 17, 4, "qwen25vl", "pilot")
    seqarm("psQ", "anchor", "o1", 23, 4, "qwen25vl", "pilot")
    seqarm("psQ", "seq", "o2", 17, 4, "qwen25vl", "pilot")
    # method candidate on the pilot suite, Qwen backbone (only gate_info says so)
    seqarm("mth", "mth_critp", "o1", 17, 4, "qwen25vl", "pilot")
    # Qwen UCIT partial
    seqarm("fsQ", "seq", "o1", 17, 6, "qwen25vl", "ucit", stages=[1, 2])
    # flags
    for c in C:
        if c[0] == "fsL_seq_o1_s23_k4":
            c[8].add("recoverable")
        if c[0] == "fsL_ewc_o1_s17_k2":
            c[8].add("malformed")
        if c[0] == "fsL_anchor_o1_s23_k3":
            c[8].add("in_progress")
    return C


def build(root):
    rng = random.Random(17)
    # ---- data assets
    gt = {}
    for i in range(N_CHAIR):
        cid = str(100001 + i)
        objs = sorted(rng.sample(OBJS, 2))
        gt[cid] = {"objects": objs, "captions": ["A %s next to a %s." % (objs[0], objs[1])]}
    w(os.path.join(root, "data", "chair", "coco_gt.json"), obj=gt)
    w(os.path.join(root, "data", "pope", "prompts.jsonl"),
      rows=[{"id": "pope_%d" % i, "image": "x.jpg", "prompt": "q", "gt": "yes", "category": "adversarial"}
            for i in range(N_POPE)])
    with open(os.path.join(REPO, "fullstudy", "pilot_manifest.json")) as f:
        pm = json.load(f)
    w(os.path.join(root, "data", "pilot_manifest.json"), obj=pm)
    for t in PILOT:
        w(os.path.join(root, "data", "tasks", t, "val.jsonl"),
          rows=[{"id": "%s_%d" % (t, i)} for i in range(N_TASK)])
    o3 = list(UCIT)
    random.Random(41).shuffle(o3)
    um = {"benchmark": "MLLM-CL/UCIT (SYNTHETIC FIXTURE)", "seed": 17,
          "tasks": {t: {"n_train": 8000, "max_new_tokens": 16,
                        "answer_stats": {"yes_frac": 0.0, "no_frac": 0.0, "refusal_frac": 0.0,
                                         "mean_target_words": 1.0}} for t in UCIT},
          "orders": {"o1": UCIT, "o2": list(reversed(UCIT)), "o3": o3}}
    w(os.path.join(root, "data_ucit", "ucit_manifest.json"), obj=um)
    for t in UCIT:
        w(os.path.join(root, "data_ucit", "tasks", t, "val.jsonl"),
          rows=[{"id": "%s_va%d" % (t, i)} for i in range(N_TASK)])
    # ---- code tree (cluster import path)
    os.makedirs(os.path.join(root, "code", "analysis"), exist_ok=True)
    for f in ("metrics_pope.py", "metrics_chair.py", "metrics_task.py"):
        shutil.copy(os.path.join(REPO, "pilot", f), os.path.join(root, "code", f))
    for f in ("fs_common.py", "fs_aggregate.py", "make_fs_figures.py", "make_sota_table.py",
              "fig_style.py"):
        shutil.copy(os.path.join(REPO, "analysis", f), os.path.join(root, "code", "analysis", f))
    # ---- base cells
    R = os.path.join(root, "results_fs")
    for bb in ("llava15", "qwen25vl"):
        d = os.path.join(R, bb + "_base")
        write_pope(os.path.join(d, "pope_gen.jsonl"), BASE_C, BASE_D, rng)
        write_chair(os.path.join(d, "chair_gen.jsonl"), gt, 0.15, rng)
        for t in UCIT:
            write_task_ucit(os.path.join(d, t + "_gen.jsonl"), t, 0.3, rng)
        gate_info(os.path.join(d, "gate_info.json"), bb, bb + "_base")
        w(os.path.join(d, "EVAL_DONE"), text="")
    # ---- arm cells
    for name, arm, bb, suite, k, order, seed, fw, flags in cells():
        d = os.path.join(R, name)
        T = 6 if suite == "ucit" else 4
        c = traj_c(arm, k, seed, fw)
        dprime = BASE_D + 0.02 * ((seed + k) % 3 - 1)
        corrupt = "recoverable" if "recoverable" in flags else ("malformed" if "malformed" in flags else None)
        write_pope(os.path.join(d, "pope_gen.jsonl"), c, dprime, rng, corrupt=corrupt)
        write_chair(os.path.join(d, "chair_gen.jsonl"), gt, p_hal(arm, k, T), rng)
        tasks = UCIT if suite == "ucit" else PILOT
        orders = um["orders"] if suite == "ucit" else pm["orders"]
        tl = orders[order]
        for t in tasks:
            trained = (arm.startswith("single_") and t == arm[7:]) or \
                      (arm == "joint") or (t in tl and tl.index(t) < k)
            acc = 0.8 if trained else 0.3
            if arm == "anchor" or arm.startswith("lamF"):
                acc -= 0.03
            if suite == "ucit":
                write_task_ucit(os.path.join(d, t + "_gen.jsonl"), t, acc, rng)
            else:
                leak = 0.15 if (t == "textvqa" and "vizwiz" in tl[:k] and arm == "seq") else 0.0
                write_task_pilot(os.path.join(d, t + "_gen.jsonl"), t, acc, leak, rng)
        gate_info(os.path.join(d, "gate_info.json"), bb, name)
        if "in_progress" not in flags:
            w(os.path.join(d, "EVAL_DONE"), text="")
    os.makedirs(os.path.join(R, "scratch_tmp"), exist_ok=True)      # unparsed dir
    # ---- bench
    B = os.path.join(root, "results_fs_bench")

    def bench(label, chair_i, mme_score, done=True, noncoco=True, full=True):
        d = os.path.join(B, label)
        w(os.path.join(d, "objhal.json"), obj={"ckpt": label, "n_captions": 300, "chair_s": 0.4,
                                               "chair_i": chair_i, "mentions_per_caption": 5.0,
                                               "mean_new_tokens": 100.0, "truncation_rate": 0.0})
        if noncoco:
            w(os.path.join(d, "noncoco.json"), obj={"ckpt": label, "n_captions": 500, "chair_s": 0.3,
                                                    "chair_i": chair_i - 0.05,
                                                    "mentions_per_caption": 4.0,
                                                    "mean_new_tokens": 90.0, "truncation_rate": 0.0})
        subs = ["existence", "count", "position", "color"] if full else ["existence", "count"]
        rows = [{"ckpt": label, "subtask": s, "n_questions": 60, "n_images": 30, "acc": 0.8,
                 "acc_plus": 0.6, "score": mme_score / len(subs)} for s in subs]
        rows.append({"ckpt": label, "subtask": "all", "n_questions": 60 * len(subs),
                     "acc": 0.8, "score": mme_score, "subtasks_present": ",".join(subs),
                     "full_800_scale": full, "other_rate": 0.0})
        w(os.path.join(d, "mme.json"), obj=rows)
        if done:
            w(os.path.join(d, "BENCH_DONE"), text="")
    bench("llava15_base", 0.20, 640.0)
    bench("fsL_seq_o1_s17_bench", 0.26, 590.0)
    bench("fsL_anchor_o1_s17_bench", 0.21, 630.0, done=False, noncoco=False)
    bench("fsL_ewc_o1_s17_bench", 0.25, 600.0, full=False)
    bench("psQ_seq_o1_s17_bench", 0.24, 580.0)
    w(os.path.join(root, "THIS_IS_SYNTHETIC_FIXTURE_DATA.txt"),
      text="Every value under this directory is fake (analysis/tests/make_fs_fixtures.py).\n")
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="root dir (default: a fresh temp dir)")
    args = ap.parse_args()
    root = args.out or tempfile.mkdtemp(prefix="clh_fixture_")
    if args.out and os.path.isdir(root):
        shutil.rmtree(root)
    os.makedirs(root, exist_ok=True)
    build(root)
    n = len([d for d in os.listdir(os.path.join(root, "results_fs"))])
    print("[make_fs_fixtures] wrote %s (%d results_fs dirs)" % (root, n))
    print(root)


if __name__ == "__main__":
    main()
