"""Generate SYNTHETIC results fixtures for testing the analysis pipeline.

Everything written here is OBVIOUSLY FAKE (patterned values: chair_i is exactly
0.111, 0.222, ... per checkpoint) and lives ONLY under analysis/tests/fixtures/
so it can never be mistaken for real results. File names and schemas exactly
mirror what pilot/jobs/sb_eval.sbatch + pilot/metrics_*.py produce.

Creates three fixture results dirs:
  fixtures/results_synth/     all 9 checkpoints (S0..S4, J1..J4), complete files
  fixtures/results_s0_only/   only S0 (the realistic partial state mid-pilot)
  fixtures/results_gappy/     S0, S1, S3, J1 (tests mid-trajectory gap handling)

Usage:  python analysis/tests/make_fixtures.py
Then:   python analysis/make_figures.py --results analysis/tests/fixtures/results_synth \
                                        --out analysis/tests/out/synth
        python analysis/make_tables.py  --results analysis/tests/fixtures/results_synth \
                                        --out analysis/tests/out/synth
"""
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
FIXROOT = os.path.join(HERE, "fixtures")

CKPTS = ["S0", "S1", "S2", "S3", "S4", "J1", "J2", "J3", "J4"]
TASKS = ["scienceqa", "textvqa", "flickr", "vizwiz"]
TASK_METRIC = {"scienceqa": "mc_acc", "textvqa": "vqa_acc",
               "flickr": "caption_uf1", "vizwiz": "vqa_acc"}
POPE_SPLITS = ["adversarial", "popular", "random"]  # metrics_pope writes sorted cats

N_POPE_PER_SPLIT = 30   # tiny on purpose; summary values are NOT derived from these
N_CHAIR_IMGS = 60


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def make_ckpt(d, ckpt, i):
    """i = index of ckpt in CKPTS (0..8); all values are patterned fakes."""
    os.makedirs(d, exist_ok=True)

    # ---- chair.json (schema: metrics_chair.py summary)
    chair_i = round(0.111 * (i + 1), 3)          # 0.111, 0.222, ..., 0.999
    chair_s = round(0.101 * (i + 1), 3)          # 0.101, 0.202, ..., 0.909
    chair = {
        "ckpt": ckpt,
        "n_captions": N_CHAIR_IMGS,
        "chair_s": chair_s,
        "chair_i": chair_i,
        "mentions_per_caption": round(5.0 + 0.1 * i, 3),
        "mean_new_tokens": round(100.0 + 10.0 * i, 1),
        "truncation_rate": round(0.01 * i, 4),
    }
    write_json(os.path.join(d, "chair.json"), chair)

    # ---- chair_per_image.jsonl (schema: metrics_chair.py per-image rows)
    rows = []
    n_hal = int(round(chair_s * N_CHAIR_IMGS))
    for k in range(N_CHAIR_IMGS):
        hal = ["cat"] if k < n_hal else []
        rows.append({
            "id": "chair_%s_%03d" % (ckpt, k),
            "coco_id": str(100000 + k),
            "mentioned": sorted(set(["dog", "chair"] + hal)),
            "hallucinated": hal,
            "n_new_tokens": 100 + 10 * i,
            "truncated": bool(k < i),
        })
    write_jsonl(os.path.join(d, "chair_per_image.jsonl"), rows)

    # ---- pope.json (schema: metrics_pope.py rows, sorted by category incl. "all")
    def pope_row(cat, sidx, n):
        f1 = round(0.9 - 0.05 * i - 0.02 * sidx, 4)   # declining, split-offset
        return {
            "ckpt": ckpt, "category": cat, "n": n,
            "accuracy": round(f1 + 0.01, 4),
            "precision": round(f1 + 0.005, 4),
            "recall": round(f1 - 0.005, 4),
            "f1": f1,
            "yes_rate": round(0.5 + 0.02 * i, 4),
            "parse_fail_rate": round(0.005 * i, 4),
        }

    pope = [pope_row(c, s, N_POPE_PER_SPLIT)
            for s, c in enumerate(POPE_SPLITS)]
    pope.insert(1, pope_row("all", 1, 3 * N_POPE_PER_SPLIT))  # sorted: adv, all, pop, rand
    pope.sort(key=lambda r: r["category"])
    write_json(os.path.join(d, "pope.json"), pope)

    # ---- pope_gen.jsonl (schema: eval_gen.py output rows over POPE prompts)
    rows = []
    for cat in POPE_SPLITS:
        for k in range(N_POPE_PER_SPLIT):
            rows.append({
                "id": "pope_%s_%s_%03d" % (ckpt, cat, k),
                "category": cat,
                "gt": "yes" if k % 2 == 0 else "no",
                "prompt": "Is there a dog in the image?",
                "output": "Yes." if k % 2 == 0 else "No.",
                "n_new_tokens": 3,
                "truncated": bool(k < i),   # per-ckpt varying truncation audit
                "gen_s": 0.05,
            })
    write_jsonl(os.path.join(d, "pope_gen.jsonl"), rows)

    # ---- task_<t>.json (schema: metrics_task.py + ckpt/task/audit fields)
    for tidx, t in enumerate(TASKS):
        res = {
            "metric": TASK_METRIC[t],
            "score": round(0.9 - 0.1 * tidx - 0.05 * i, 3),  # patterned decline
            "n": 500,
            "parse_fail_rate": round(0.002 * i, 4),
            "ckpt": ckpt,
            "task": t,
            "mean_new_tokens": round(10.0 + 5.0 * tidx + i, 1),
            "truncation_rate": round(0.003 * i, 4),
        }
        write_json(os.path.join(d, "task_%s.json" % t), res)


def main():
    if os.path.isdir(FIXROOT):
        shutil.rmtree(FIXROOT)
    os.makedirs(FIXROOT)
    with open(os.path.join(FIXROOT, "THIS_IS_SYNTHETIC_FIXTURE_DATA.txt"), "w") as f:
        f.write("Every value under this directory is fake, generated by\n"
                "analysis/tests/make_fixtures.py for pipeline testing only.\n"
                "chair_i is literally 0.111, 0.222, ... Do not analyze as results.\n")

    variants = {
        "results_synth": CKPTS,
        "results_s0_only": ["S0"],
        "results_gappy": ["S0", "S1", "S3", "J1"],
    }
    for name, ckpts in variants.items():
        root = os.path.join(FIXROOT, name)
        for c in ckpts:
            make_ckpt(os.path.join(root, c), c, CKPTS.index(c))
        print("[make_fixtures] wrote %s (%s)" % (root, ", ".join(ckpts)))
    print("[make_fixtures] DONE")


if __name__ == "__main__":
    main()
