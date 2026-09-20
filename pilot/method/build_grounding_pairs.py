"""Build counterfactual grounding pairs for the faithfulness anchor (candidate C2).

Outputs under $ROOT (same root as data_prep.py):
  grounding/images/ground_<coco_id>.jpg   original val2014 image (re-encoded RGB JPEG)
  grounding/masked/ground_<coco_id>.jpg   every instance of the present class covered
                                          by a filled bbox in the dataset mean color
  grounding/grounding_pairs.jsonl         {id, image, masked_image, present, absent}
  grounding/report.json                   counts + audit fields; written LAST (success marker)

Pair images are val2014, DISJOINT from the CHAIR pool (chair/images.jsonl coco_id)
and the POPE pool (pope/prompts.jsonl meta.image_source). present = the category of
the single largest instance whose bbox covers >= MIN_AREA_FRAC of the image;
absent = an 80-category class with no instance in the image, sampled proportional
to its co-occurrence with the image's present classes across val2014 (hard
negative). Reuses the archives data_prep.py already downloaded; downloads nothing.
All sampling is seeded (SEED).
"""
import argparse
import io
import json
import os
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.gate_checks import run_all_gates

from PIL import Image, ImageDraw, ImageStat

SEED = 17
N_PAIRS = 2000
MIN_AREA_FRAC = 0.05

COCO_SOURCE = re.compile(r"^COCO_([A-Za-z0-9]+)_(\d+)$")


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
    assert m, f"unparseable COCO image source {s!r} - dedup would silently break"
    return m.group(1), int(m.group(2))


def load_excluded_ids(root):
    chair_ids = set()
    with open(os.path.join(root, "chair", "images.jsonl")) as f:
        for l in f:
            chair_ids.add(int(json.loads(l)["coco_id"]))
    pope_ids, pope_other = set(), 0
    with open(os.path.join(root, "pope", "prompts.jsonl")) as f:
        for l in f:
            r = json.loads(l)
            src = (r.get("meta") or {}).get("image_source") or r["image"]
            split, iid = parse_source(src)
            if split == "val2014":
                pope_ids.add(iid)
            else:
                pope_other += 1
    assert chair_ids and pope_ids, "empty exclusion pool - wrong --root?"
    return chair_ids, pope_ids, pope_other


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run_all_gates(args.root, min_free_gb=10.0, need_gpu=False)
    rng = random.Random(SEED)
    n_pairs = 40 if args.smoke else N_PAIRS

    ann_zip = os.path.join(args.root, "downloads", "annotations_trainval2014.zip")
    val_zip = os.path.join(args.root, "downloads", "val2014.zip")
    for p in (ann_zip, val_zip):
        assert os.path.exists(p), f"missing archive {p} - run data_prep.py first"
    with zipfile.ZipFile(ann_zip) as z:
        inst = json.load(io.TextIOWrapper(z.open("annotations/instances_val2014.json"),
                                          encoding="utf-8"))

    cat_name = {c["id"]: c["name"] for c in inst["categories"]}
    all_cats = sorted(set(cat_name.values()))
    assert len(all_cats) == 80, f"expected 80 categories, got {len(all_cats)}"
    img_meta = {im["id"]: im for im in inst["images"]}
    anns_by_img = defaultdict(list)
    for a in inst["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    cats_by_img = {iid: {cat_name[a["category_id"]] for a in anns}
                   for iid, anns in anns_by_img.items()}

    cooc = defaultdict(Counter)
    for cats in cats_by_img.values():
        for p in cats:
            for c in cats:
                if c != p:
                    cooc[p][c] += 1

    chair_ids, pope_ids, pope_other = load_excluded_ids(args.root)
    excluded = chair_ids | pope_ids

    candidates = []
    for iid in sorted(anns_by_img):
        if iid in excluded or iid not in img_meta:
            continue
        w_img, h_img = img_meta[iid]["width"], img_meta[iid]["height"]
        best_frac, best_cat = 0.0, None
        for a in anns_by_img[iid]:
            frac = (a["bbox"][2] * a["bbox"][3]) / (w_img * h_img)
            if frac >= MIN_AREA_FRAC and frac > best_frac:
                best_frac, best_cat = frac, cat_name[a["category_id"]]
        if best_cat is None or len(cats_by_img[iid]) >= len(all_cats):
            continue
        candidates.append((iid, best_cat, best_frac))
    assert len(candidates) >= n_pairs, (
        f"only {len(candidates)} eligible images for {n_pairs} pairs")
    rng.shuffle(candidates)

    pairs = []
    for iid, present, frac in candidates[:n_pairs]:
        present_cats = cats_by_img[iid]
        absent_pool = [c for c in all_cats if c not in present_cats]
        weights = [sum(cooc[p][c] for p in present_cats) for c in absent_pool]
        if sum(weights) > 0:
            absent = rng.choices(absent_pool, weights=weights, k=1)[0]
        else:
            absent = rng.choice(absent_pool)
        assert present in present_cats and absent not in present_cats
        pairs.append({"iid": iid, "present": present, "absent": absent, "frac": frac})

    out = os.path.join(args.root, "grounding")
    imgdir = os.path.join(out, "images")
    maskdir = os.path.join(out, "masked")
    os.makedirs(imgdir, exist_ok=True)
    os.makedirs(maskdir, exist_ok=True)

    ch_sum, n_px = [0.0, 0.0, 0.0], 0
    with zipfile.ZipFile(val_zip) as z:
        for p_ in pairs:
            fn = img_meta[p_["iid"]]["file_name"]
            with z.open(f"val2014/{fn}") as src:
                img = Image.open(io.BytesIO(src.read())).convert("RGB")
            s = ImageStat.Stat(img)
            for c in range(3):
                ch_sum[c] += s.sum[c]
            n_px += img.size[0] * img.size[1]
            rel = os.path.join("grounding", "images", f"ground_{p_['iid']}.jpg")
            save_img(img, os.path.join(args.root, rel))
            p_["image"] = rel
    mean_color = tuple(int(round(ch_sum[c] / n_px)) for c in range(3))

    for p_ in pairs:
        img = Image.open(os.path.join(args.root, p_["image"])).convert("RGB")
        draw = ImageDraw.Draw(img)
        n_boxes = 0
        for a in anns_by_img[p_["iid"]]:
            if cat_name[a["category_id"]] != p_["present"]:
                continue
            x, y, w, h = a["bbox"]
            draw.rectangle([x, y, x + w, y + h], fill=mean_color)
            n_boxes += 1
        assert n_boxes >= 1, f"no instance masked for {p_['iid']}/{p_['present']}"
        rel = os.path.join("grounding", "masked", f"ground_{p_['iid']}.jpg")
        save_img(img, os.path.join(args.root, rel))
        p_["masked_image"] = rel
        p_["n_boxes"] = n_boxes

    rows = [{"id": f"ground_{p_['iid']}", "image": p_["image"],
             "masked_image": p_["masked_image"], "present": p_["present"],
             "absent": p_["absent"]} for p_ in pairs]
    assert len({r["id"] for r in rows}) == len(rows) == n_pairs
    assert not {p_["iid"] for p_ in pairs} & excluded
    write_jsonl(os.path.join(out, "grounding_pairs.jsonl"), rows)

    report = {
        "smoke": args.smoke,
        "seed": SEED,
        "n_pairs": len(rows),
        "min_area_frac": MIN_AREA_FRAC,
        "mean_color_rgb": list(mean_color),
        "excluded_chair": len(chair_ids),
        "excluded_pope_val2014": len(pope_ids),
        "pope_non_val2014_sources": pope_other,
        "eligible_candidates": len(candidates),
        "mean_present_area_frac": round(sum(p_["frac"] for p_ in pairs) / len(pairs), 4),
        "mean_boxes_masked": round(sum(p_["n_boxes"] for p_ in pairs) / len(pairs), 2),
        "present_top": Counter(p_["present"] for p_ in pairs).most_common(10),
        "absent_top": Counter(p_["absent"] for p_ in pairs).most_common(10),
    }
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[pairs] {len(rows)} pairs, mean_color={mean_color}, "
          f"excluded {len(excluded)} eval-pool images", flush=True)


if __name__ == "__main__":
    main()
