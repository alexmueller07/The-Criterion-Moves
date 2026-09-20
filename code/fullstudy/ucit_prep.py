"""UCIT full-study data prep: builds tasks/<name>/{train,val}.jsonl + images,
computes and FREEZES per-task answer statistics into ucit_manifest.json
(pre-registration blinding rule: stats are committed before any eval runs).

Schema (verified by probe on ArxivQA; asserted per-config here): rows carry
`conversations` (LLaVA-style list of {from, value} with "<image>" in the first
human turn), `images` (list, PIL-decodable), and optionally `problem`/`answer`.
Prompt = first human turn minus the image tag; target = first gpt turn.
"""
import argparse
import json
import os
import random
import re
from collections import Counter

SEED = 17
TRAIN_CAP = 8000
VAL_CAP = 500

ORDER_CANONICAL = ["ArxivQA", "CLEVR-Math", "Flickr30k", "IconQA", "ImageNet-R", "VizWiz"]


def extract(ex, cfg, idx):
    conv = ex.get("conversations")
    assert conv and isinstance(conv, list) and len(conv) >= 2, \
        f"{cfg}[{idx}]: unexpected conversations {type(conv)}"
    def val(turn):
        return turn.get("value") if isinstance(turn, dict) else None
    human = next((val(t) for t in conv if isinstance(t, dict)
                  and t.get("from") in ("human", "user")), None)
    gpt = next((val(t) for t in conv if isinstance(t, dict)
                and t.get("from") in ("gpt", "assistant")), None)
    assert human and gpt, f"{cfg}[{idx}]: missing human/gpt turns: {conv[:2]}"
    prompt = re.sub(r"<image>\s*", "", human).strip()
    target = gpt.strip()
    imgs = ex.get("images") or ([ex["image"]] if "image" in ex else None)
    assert imgs, f"{cfg}[{idx}]: no images field"
    return prompt, target, imgs[0]


def save_img(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if hasattr(img, "save"):
        img.convert("RGB").save(path, "JPEG", quality=92)
    else:
        raise AssertionError(f"non-PIL image of type {type(img)}")


def answer_stats(rows):
    yes = no = una = 0
    lens = []
    for r in rows:
        a = r["target"].strip().lower()
        yes += a.startswith("yes")
        no += a.startswith("no")
        una += "unanswerable" in a[:24]
        lens.append(len(a.split()))
    lens.sort()
    n = len(rows)
    return {"yes_frac": round(yes / n, 4), "no_frac": round(no / n, 4),
            "refusal_frac": round(una / n, 4),
            "mean_target_words": round(sum(lens) / n, 2),
            "p95_target_words": lens[int(0.95 * n)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    import datasets
    datasets.disable_progress_bars()
    from datasets import load_dataset, get_dataset_split_names

    manifest = {"benchmark": "MLLM-CL/UCIT", "seed": SEED, "tasks": {}, "orders": {}}
    for cfg in ORDER_CANONICAL:
        out = os.path.join(args.root, "tasks", cfg)
        done_marker = os.path.join(out, "PREP_DONE")
        splits = get_dataset_split_names("MLLM-CL/UCIT", cfg)
        train_split = "train" if "train" in splits else splits[0]
        eval_split = "test" if "test" in splits else train_split
        ds = load_dataset("MLLM-CL/UCIT", cfg, split=train_split)
        n_train = min(TRAIN_CAP, len(ds) - VAL_CAP) if eval_split == train_split \
            else min(TRAIN_CAP, len(ds))
        if args.smoke:
            n_train = 40
        idxs = list(range(len(ds)))
        random.Random(SEED).shuffle(idxs)
        train_idx = idxs[:n_train]
        if eval_split == train_split:
            val_idx = idxs[n_train:n_train + (VAL_CAP if not args.smoke else 10)]
            eval_ds, val_rows_src = ds, val_idx
        else:
            eds = load_dataset("MLLM-CL/UCIT", cfg, split=eval_split)
            vidx = list(range(len(eds)))
            random.Random(SEED).shuffle(vidx)
            eval_ds, val_rows_src = eds, vidx[:VAL_CAP if not args.smoke else 10]

        if os.path.exists(done_marker) and not args.smoke:
            print(f"[prep] {cfg}: already built, recomputing stats only", flush=True)
            rows = [json.loads(l) for l in open(os.path.join(out, "train.jsonl"))]
        else:
            rows = []
            for j, i in enumerate(train_idx):
                p, t, img = extract(ds[i], cfg, i)
                rel = os.path.join("tasks", cfg, "images", f"tr{j:06d}.jpg")
                save_img(img, os.path.join(args.root, rel))
                rows.append({"id": f"{cfg}_tr{j}", "image": rel, "prompt": p, "target": t})
            vrows = []
            for j, i in enumerate(val_rows_src):
                p, t, img = extract(eval_ds[i], cfg, i)
                rel = os.path.join("tasks", cfg, "images", f"va{j:06d}.jpg")
                save_img(img, os.path.join(args.root, rel))
                vrows.append({"id": f"{cfg}_va{j}", "image": rel, "prompt": p,
                              "target": t, "meta": {"answers": [t]}})
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "train.jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            with open(os.path.join(out, "val.jsonl"), "w") as f:
                for r in vrows:
                    f.write(json.dumps(r) + "\n")
            open(done_marker, "w").write("ok\n")
        st = answer_stats(rows)
        mnt = max(16, min(128, 2 * st["p95_target_words"] + 8))
        manifest["tasks"][cfg] = {"n_train": len(rows), "max_new_tokens": int(mnt),
                                  "answer_stats": st}
        print(f"[prep] {cfg}: n_train={len(rows)} stats={st} mnt={mnt}", flush=True)

    manifest["orders"]["o1"] = ORDER_CANONICAL
    manifest["orders"]["o2"] = list(reversed(ORDER_CANONICAL))
    o3 = list(ORDER_CANONICAL)
    random.Random(41).shuffle(o3)
    manifest["orders"]["o3"] = o3
    mpath = os.path.join(args.root, "ucit_manifest.json")
    json.dump(manifest, open(mpath, "w"), indent=1)
    print(f"[prep] manifest frozen at {mpath}; orders o3={o3}", flush=True)
    print("[prep] ALL OK", flush=True)


if __name__ == "__main__":
    main()
