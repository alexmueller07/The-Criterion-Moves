"""Build the pilot's train/eval assets on vgi2 from HF datasets.

Outputs under $ROOT/data:
  tasks/<task>/train.jsonl        {id, image, prompt, target}   (image = relative path)
  tasks/<task>/val.jsonl          500 held-out items, same schema + task-specific meta
  tasks/<task>/images/*.jpg
  pope/prompts.jsonl              {id, image, prompt, gt, category}
  pope/images/*.jpg
  chair/images.jsonl              {id, image, coco_id}
  chair/images/*.jpg              500 val2014 images (seed 17)
  chair/coco_gt.json              per-coco_id ground-truth object sets (80-cat space)
  prep_report.json                counts + audit fields; written LAST (success marker)

All sampling is seeded (SEED). Images are re-encoded to JPEG (RGB).
"""
import argparse
import io
import json
import os
import random
import sys
import zipfile
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.gate_checks import run_all_gates

SEED = 17
TRAIN_PER_TASK = 8000
VAL_PER_TASK = 500
VIZWIZ_TRAIN = 3800
SCIENCEQA_TRAIN = 5700


def save_img(img, path):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(path, "JPEG", quality=92)


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prep_task(root, name, hf_load, build_row, n_train, n_val, smoke):
    """Generic: sample rows, save images, emit train/val jsonl."""
    from datasets import load_dataset
    out = os.path.join(root, "tasks", name)
    imgdir = os.path.join(out, "images")
    os.makedirs(imgdir, exist_ok=True)
    ds = hf_load()
    n_total = len(ds)
    rng = random.Random(SEED)
    idx = list(range(n_total))
    rng.shuffle(idx)
    if smoke:
        n_train, n_val = 200, 50
    need = n_train + n_val
    rows, used, skipped = [], 0, 0
    for i in idx:
        if used >= need:
            break
        try:
            r = build_row(ds[i], f"{name}_{i}")
        except Exception:
            skipped += 1
            continue
        if r is None:
            skipped += 1
            continue
        img, rec = r
        rel = os.path.join("tasks", name, "images", rec["id"] + ".jpg")
        save_img(img, os.path.join(root, rel))
        rec["image"] = rel
        rows.append(rec)
        used += 1
    if used < need:
        raise RuntimeError(f"{name}: only {used}/{need} usable rows (skipped {skipped})")
    write_jsonl(os.path.join(out, "train.jsonl"), rows[:n_train])
    write_jsonl(os.path.join(out, "val.jsonl"), rows[n_train:need])
    print(f"[prep] {name}: train={n_train} val={need - n_train} skipped={skipped}", flush=True)
    return {"task": name, "train": n_train, "val": need - n_train, "skipped": skipped}


# ---------------- task builders ----------------

def build_scienceqa(ex, rid):
    if ex.get("image") is None:
        return None
    letters = "ABCDE"
    choices = ex["choices"]
    if not choices or len(choices) > 5:
        return None
    opts = "\n".join(f"{letters[j]}. {c}" for j, c in enumerate(choices))
    prompt = (f"{ex['question']}\n{opts}\n"
              "Answer with the option's letter from the given choices directly.")
    target = letters[int(ex["answer"])]
    return ex["image"], {"id": rid, "prompt": prompt, "target": target,
                         "meta": {"answer_letter": target}}


def build_textvqa(ex, rid):
    if ex.get("image") is None:
        return None
    answers = [a.strip() for a in ex["answers"] if a and a.strip()]
    if not answers:
        return None
    target = Counter(a.lower() for a in answers).most_common(1)[0][0]
    prompt = f"{ex['question']}\nAnswer the question using a single word or phrase."
    return ex["image"], {"id": rid, "prompt": prompt, "target": target,
                         "meta": {"answers": answers}}


def build_flickr(ex, rid):
    caps = ex.get("caption") or []
    if ex.get("image") is None or not caps:
        return None
    prompt = "Provide a one-sentence caption for the provided image."
    return ex["image"], {"id": rid, "prompt": prompt, "target": caps[0],
                         "meta": {"refs": caps[:5]}}


def build_vizwiz(ex, rid):
    if ex.get("image") is None:
        return None
    answers = [a.strip() for a in ex["answers"] if a and a.strip()]
    if not answers:
        return None
    target = Counter(a.lower() for a in answers).most_common(1)[0][0]
    prompt = (f"{ex['question']}\n"
              "When the provided information is insufficient, respond with 'Unanswerable'.\n"
              "Answer the question using a single word or phrase.")
    return ex["image"], {"id": rid, "prompt": prompt, "target": target,
                         "meta": {"answers": answers}}


# ---------------- POPE ----------------

def prep_pope(root, smoke):
    """POPE ships as three named splits (adversarial/popular/random) — verified
    by probe 2026-08-31. image_source (e.g. COCO_val2014_000000310196) is the
    per-image identifier used for dedup."""
    from datasets import load_dataset
    out = os.path.join(root, "pope")
    imgdir = os.path.join(out, "images")
    os.makedirs(imgdir, exist_ok=True)
    rows, seen_imgs = [], {}
    ds = load_dataset("lmms-lab/POPE", split="test")
    n = 60 if smoke else len(ds)
    cat_counts = Counter()
    for i in range(n):
        ex = ds[i]
        cat = str(ex["category"]).strip().lower()
        img_key = str(ex["image_source"])
        if img_key not in seen_imgs:
            rel = os.path.join("pope", "images", img_key + ".jpg")
            save_img(ex["image"], os.path.join(root, rel))
            seen_imgs[img_key] = rel
        cat_counts[cat] += 1
        rows.append({
            "id": f"pope_{cat}_{cat_counts[cat]}",
            "image": seen_imgs[img_key],
            "prompt": f"{ex['question']}\nAnswer the question using a single word or phrase.",
            "gt": str(ex["answer"]).strip().lower(),
            "category": cat,
            "meta": {"image_source": img_key,
                     "question_id": str(ex.get("question_id"))},
        })
    assert set(cat_counts) >= {"adversarial", "popular", "random"} or smoke, \
        f"POPE category column unexpected: {dict(cat_counts)}"
    write_jsonl(os.path.join(out, "prompts.jsonl"), rows)
    cats = Counter(r["category"] for r in rows)
    print(f"[prep] POPE: {len(rows)} prompts, {len(seen_imgs)} unique images, cats={dict(cats)}",
          flush=True)
    return {"pope_prompts": len(rows), "pope_images": len(seen_imgs), "cats": dict(cats)}


# ---------------- CHAIR (COCO val2014) ----------------

COCO_URLS = {
    "ann": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
    "val_imgs": "http://images.cocodataset.org/zips/val2014.zip",
}


def download(url, dest):
    import urllib.request
    if os.path.exists(dest) and os.path.getsize(dest) > 1e6:
        print(f"[prep] cached: {dest}", flush=True)
        return
    print(f"[prep] downloading {url}", flush=True)
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.rename(tmp, dest)


def prep_chair(root, smoke):
    out = os.path.join(root, "chair")
    imgdir = os.path.join(out, "images")
    dl = os.path.join(root, "downloads")
    os.makedirs(imgdir, exist_ok=True)
    os.makedirs(dl, exist_ok=True)
    n_imgs = 20 if smoke else 500

    ann_zip = os.path.join(dl, "annotations_trainval2014.zip")
    download(COCO_URLS["ann"], ann_zip)
    with zipfile.ZipFile(ann_zip) as z:
        inst = json.load(io.TextIOWrapper(z.open("annotations/instances_val2014.json"),
                                          encoding="utf-8"))
        caps = json.load(io.TextIOWrapper(z.open("annotations/captions_val2014.json"),
                                          encoding="utf-8"))

    cat_by_id = {c["id"]: c["name"] for c in inst["categories"]}
    gt_objs = {}
    for a in inst["annotations"]:
        gt_objs.setdefault(a["image_id"], set()).add(cat_by_id[a["category_id"]])
    gt_caps = {}
    for c in caps["annotations"]:
        gt_caps.setdefault(c["image_id"], []).append(c["caption"])

    img_meta = {im["id"]: im["file_name"] for im in inst["images"]}
    candidates = sorted(iid for iid in img_meta if iid in gt_objs and iid in gt_caps)
    rng = random.Random(SEED)
    chosen = rng.sample(candidates, n_imgs)

    val_zip = os.path.join(dl, "val2014.zip")
    download(COCO_URLS["val_imgs"], val_zip)
    rows = []
    with zipfile.ZipFile(val_zip) as z:
        for iid in chosen:
            fn = img_meta[iid]
            rel = os.path.join("chair", "images", fn)
            with z.open(f"val2014/{fn}") as src, \
                 open(os.path.join(root, rel), "wb") as dst:
                dst.write(src.read())
            rows.append({"id": f"chair_{iid}", "image": rel, "coco_id": iid,
                         "prompt": "Please describe this image in detail."})
    write_jsonl(os.path.join(out, "images.jsonl"), rows)
    with open(os.path.join(out, "coco_gt.json"), "w") as f:
        json.dump({str(iid): {"objects": sorted(gt_objs[iid]),
                              "captions": gt_caps[iid][:5]} for iid in chosen}, f)
    print(f"[prep] CHAIR: {len(rows)} images with GT", flush=True)
    return {"chair_images": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default="all",
                    help="comma list: scienceqa,textvqa,flickr,vizwiz,pope,chair")
    args = ap.parse_args()
    run_all_gates(args.root, min_free_gb=60.0, need_gpu=False)
    from datasets import load_dataset

    only = set(args.only.split(","))
    report = {"smoke": args.smoke, "seed": SEED, "parts": []}

    tasks = {
        "scienceqa": lambda: prep_task(
            args.root, "scienceqa",
            lambda: load_dataset("derek-thomas/ScienceQA", split="train"),
            build_scienceqa, SCIENCEQA_TRAIN, VAL_PER_TASK, args.smoke),
        "textvqa": lambda: prep_task(
            args.root, "textvqa",
            lambda: load_dataset("lmms-lab/textvqa", split="train"),
            build_textvqa, TRAIN_PER_TASK, VAL_PER_TASK, args.smoke),
        "flickr": lambda: prep_task(
            args.root, "flickr",
            lambda: load_dataset("nlphuji/flickr30k", split="test"),
            build_flickr, TRAIN_PER_TASK, VAL_PER_TASK, args.smoke),
        "vizwiz": lambda: prep_task(
            args.root, "vizwiz",
            lambda: load_dataset("lmms-lab/VizWiz-VQA", split="val"),
            build_vizwiz, VIZWIZ_TRAIN, VAL_PER_TASK, args.smoke),
        "pope": lambda: prep_pope(args.root, args.smoke),
        "chair": lambda: prep_chair(args.root, args.smoke),
    }
    failures = []
    for name, fn in tasks.items():
        if "all" not in only and name not in only:
            continue
        try:
            report["parts"].append(fn())
        except Exception as e:
            failures.append({"part": name, "error": repr(e)[:500]})
            print(f"[prep] FAILURE in {name}: {e!r}", flush=True)
    report["failures"] = failures
    with open(os.path.join(args.root, "prep_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    if failures:
        sys.exit(1)
    print("[prep] ALL OK", flush=True)


if __name__ == "__main__":
    main()
