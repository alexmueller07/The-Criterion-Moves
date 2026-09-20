"""Build the LABEL-FREE probe set for the criterion-preservation term
(method/crit_preserve.py, candidate M1).

Outputs under $ROOT (same root as data_prep.py / build_grounding_pairs.py):
  grounding/probe_images/probe_<id>.jpg   re-encoded RGB JPEG copies
  grounding/probe.jsonl                   {id, image, object}     <- NO labels
  grounding/probe_report.json             counts + audit; written LAST
(--out X.jsonl keys the images dir / report on X: X_images/, X_report.json)

Two image sources:
  (default)     COCO val2014 from downloads/val2014.zip (already fetched by
                data_prep.py), EXCLUDING every image in POPE's eval image set
                (pope/prompts.jsonl meta.image_source, val2014 ids) and, when
                present, the CHAIR pool (chair/images.jsonl coco_id). The
                grounding-pair images are NOT excluded (they are training-side
                anchor inputs, not an eval pool); their overlap is reported.
  --images_dir  any folder of images (non-COCO), sampled the same way.

Objects are drawn from the 80 COCO class NAMES only (the vocabulary, hardcoded
below; cross-checked against the annotation zip's category list when that zip
is present) with a balanced-uniform assignment: each class appears
floor/ceil(N/80) times. Per-image instance annotations are NEVER read, so no
row knows whether its object is present -- that is the point: the term
preserves a population moment of the decision statistic and needs no
supervision. The row schema is deliberately incompatible with the pairs
schema (no present/absent keys; CriterionPreserver rejects label-like keys).

All sampling is seeded (--seed, default 17).
"""
import argparse
import io
import json
import os
import random
import re
import sys
import zipfile
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.gate_checks import run_all_gates  # noqa: E402

from PIL import Image  # noqa: E402

SEED = 17
N_PROBE = 600
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VAL2014_NAME = re.compile(r"^val2014/COCO_val2014_(\d+)\.jpg$")
COCO_SOURCE = re.compile(r"^COCO_([A-Za-z0-9]+)_(\d+)$")

# The 80 COCO object categories (instances_*2014 order). Names only.
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)
assert len(COCO_CLASSES) == 80 and len(set(COCO_CLASSES)) == 80


def save_img(img, path):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(path, "JPEG", quality=92)


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_source(s):
    stem = os.path.splitext(os.path.basename(str(s)))[0]
    m = COCO_SOURCE.match(stem)
    assert m, f"unparseable COCO image source {s!r} - exclusion would silently break"
    return m.group(1), int(m.group(2))


def load_eval_pool_ids(root):
    """val2014 ids of the POPE eval images (REQUIRED) and the CHAIR pool
    (optional). Returns (pope_ids, chair_ids, n_pope_non_val2014)."""
    pope_path = os.path.join(root, "pope", "prompts.jsonl")
    assert os.path.exists(pope_path), (
        f"{pope_path} missing: cannot guarantee the probe set is disjoint from "
        "POPE's eval images - run data_prep.py (pope) first or pass --images_dir")
    pope_ids, pope_other = set(), 0
    with open(pope_path) as f:
        for l in f:
            r = json.loads(l)
            src = (r.get("meta") or {}).get("image_source") or r["image"]
            split, iid = parse_source(src)
            if split == "val2014":
                pope_ids.add(iid)
            else:
                pope_other += 1
    assert pope_ids, "POPE prompts carry no val2014 images - wrong --root?"
    chair_ids = set()
    chair_path = os.path.join(root, "chair", "images.jsonl")
    if os.path.exists(chair_path):
        with open(chair_path) as f:
            for l in f:
                chair_ids.add(int(json.loads(l)["coco_id"]))
    return pope_ids, chair_ids, pope_other


def grounding_pair_ids(root):
    """val2014 ids used by the anchor's grounding pairs (reported, not excluded)."""
    p = os.path.join(root, "grounding", "grounding_pairs.jsonl")
    ids = set()
    if os.path.exists(p):
        with open(p) as f:
            for l in f:
                m = re.search(r"ground_(\d+)", json.loads(l)["id"])
                if m:
                    ids.add(int(m.group(1)))
    return ids


def check_vocab_against_annotations(root):
    """If the annotation zip is around, assert the hardcoded vocabulary equals its
    category list. Reads ONLY the category names, never per-image annotations.
    Returns True if checked, False if the zip is absent."""
    ann_zip = os.path.join(root, "downloads", "annotations_trainval2014.zip")
    if not os.path.exists(ann_zip):
        return False
    with zipfile.ZipFile(ann_zip) as z:
        inst = json.load(io.TextIOWrapper(z.open("annotations/instances_val2014.json"),
                                          encoding="utf-8"))
    names = sorted(c["name"] for c in inst["categories"])
    assert names == sorted(COCO_CLASSES), (
        "hardcoded COCO_CLASSES differs from the annotation zip's categories: "
        f"{sorted(set(names) ^ set(COCO_CLASSES))}")
    return True


def balanced_objects(n, rng, classes=COCO_CLASSES):
    """n object names with each class used floor(n/80) or ceil(n/80) times, in a
    seeded random order (uniform marginal, minimum-variance coverage)."""
    out = []
    while len(out) < n:
        block = list(classes)
        rng.shuffle(block)
        out.extend(block)
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="data root (data_prep.py's --root)")
    ap.add_argument("--images_dir", default=None,
                    help="arbitrary (non-COCO) image folder instead of val2014.zip")
    ap.add_argument("--n", type=int, default=N_PROBE, help="number of probe rows")
    ap.add_argument("--objects_per_image", type=int, default=1,
                    help="questions per sampled image (rows = n; images = n / this)")
    ap.add_argument("--out", default=None,
                    help="probe jsonl path (default <root>/grounding/probe.jsonl)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true", help="n=30")
    args = ap.parse_args()
    run_all_gates(args.root, min_free_gb=5.0, need_gpu=False)
    rng = random.Random(args.seed)
    n_rows = 30 if args.smoke else int(args.n)
    per_img = max(1, int(args.objects_per_image))
    assert n_rows >= 1 and n_rows % per_img == 0, (
        f"--n {n_rows} must be a positive multiple of --objects_per_image {per_img}")
    n_imgs = n_rows // per_img

    out_jsonl = args.out or os.path.join(args.root, "grounding", "probe.jsonl")
    out_dir = os.path.dirname(os.path.abspath(out_jsonl))
    # Images dir and report are keyed on the jsonl STEM (probe.jsonl ->
    # probe_images/, probe_report.json) so a second probe set written next to
    # the first (e.g. an --images_dir OOD probe as probe_ood.jsonl) never
    # clobbers its images or its audit report.
    stem_out = os.path.splitext(os.path.basename(out_jsonl))[0]
    imgdir = os.path.join(out_dir, f"{stem_out}_images")
    report_path = os.path.join(out_dir, f"{stem_out}_report.json")
    os.makedirs(imgdir, exist_ok=True)
    # Row image paths are ROOT-relative (the trainer resolves them against
    # --data_root), so the probe dir must sit inside the root.
    rel_imgdir = os.path.relpath(imgdir, args.root)
    assert not rel_imgdir.startswith(".."), (
        f"probe images dir {imgdir} must be inside --root {args.root}")

    report = {"seed": args.seed, "smoke": args.smoke, "n_rows": n_rows,
              "objects_per_image": per_img, "vocab_size": len(COCO_CLASSES),
              "object_sampling": "balanced_uniform_over_80_coco_class_names",
              "labels_used": False}

    if args.images_dir:
        files = sorted(fn for fn in os.listdir(args.images_dir)
                       if fn.lower().endswith(IMG_EXT))
        assert files, f"no images in {args.images_dir}"
        if len(files) < n_imgs:
            print(f"[probe] WARNING: only {len(files)} images in {args.images_dir} "
                  f"for {n_imgs} requested; using all", flush=True)
            n_imgs = len(files)
            n_rows = n_imgs * per_img
        chosen = rng.sample(files, n_imgs)
        report.update({"source": "images_dir", "images_dir": os.path.abspath(args.images_dir),
                       "available": len(files), "eval_pool_exclusion": "n/a (non-COCO)"})

        def load(fn):
            return Image.open(os.path.join(args.images_dir, fn)).convert("RGB")

        def stem(fn):
            return re.sub(r"[^A-Za-z0-9_-]", "_", os.path.splitext(fn)[0])
    else:
        val_zip = os.path.join(args.root, "downloads", "val2014.zip")
        assert os.path.exists(val_zip), f"missing archive {val_zip} - run data_prep.py first"
        pope_ids, chair_ids, pope_other = load_eval_pool_ids(args.root)
        excluded = pope_ids | chair_ids
        vocab_checked = check_vocab_against_annotations(args.root)
        zf = zipfile.ZipFile(val_zip)
        by_id = {}
        for name in zf.namelist():
            m = VAL2014_NAME.match(name)
            if m:
                by_id[int(m.group(1))] = name
        assert by_id, f"no val2014/COCO_val2014_*.jpg entries in {val_zip}"
        candidates = sorted(iid for iid in by_id if iid not in excluded)
        assert len(candidates) >= n_imgs, (
            f"only {len(candidates)} val2014 images outside the eval pools for {n_imgs}")
        chosen = rng.sample(candidates, n_imgs)
        assert not set(chosen) & excluded
        pair_ids = grounding_pair_ids(args.root)
        report.update({"source": "coco_val2014", "val2014_total": len(by_id),
                       "excluded_pope_val2014": len(pope_ids),
                       "pope_non_val2014_sources": pope_other,
                       "excluded_chair": len(chair_ids),
                       "eligible_candidates": len(candidates),
                       "overlap_with_grounding_pairs": len(set(chosen) & pair_ids),
                       "vocab_checked_against_annotation_zip": vocab_checked})

        def load(iid):
            with zf.open(by_id[iid]) as src:
                return Image.open(io.BytesIO(src.read())).convert("RGB")

        def stem(iid):
            return str(iid)

    objects = balanced_objects(n_rows, rng)
    rows, ids = [], set()
    for j, key in enumerate(chosen):
        img = load(key)
        s = stem(key)
        rel = os.path.join(rel_imgdir, f"probe_{s}.jpg")
        save_img(img, os.path.join(args.root, rel))
        for q in range(per_img):
            obj = objects[j * per_img + q]
            rid = f"probe_{s}" if per_img == 1 else f"probe_{s}_{q}"
            assert rid not in ids, f"duplicate probe id {rid}"
            ids.add(rid)
            rows.append({"id": rid, "image": rel, "object": obj})
    assert len(rows) == n_rows
    for r in rows:  # the label-free contract CriterionPreserver enforces
        assert set(r) == {"id", "image", "object"}, r
    write_jsonl(out_jsonl, rows)

    hist = Counter(r["object"] for r in rows)
    report.update({"n_images": len(chosen), "out": os.path.abspath(out_jsonl),
                   "object_hist_min": min(hist.values()),
                   "object_hist_max": max(hist.values()),
                   "objects_covered": len(hist)})
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[probe] {len(rows)} label-free rows over {len(chosen)} images -> {out_jsonl} "
          f"(objects/class {report['object_hist_min']}-{report['object_hist_max']}, "
          f"source={report['source']})", flush=True)


if __name__ == "__main__":
    main()
