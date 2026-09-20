"""Build a NON-COCO CHAIR-style generative-hallucination eval from Open Images V7
(validation split), in the EXACT schema our vendored scorer (pilot/metrics_chair.py)
consumes -- no scorer change needed.

WHY THIS EXISTS (the confound it closes).
  The anchor is trained on COCO grounding pairs (POPE-format probes on COCO images).
  Its one non-circular generative win (length-controlled CHAIR reduction) is measured
  on CHAIR, and the launched full-study generative benchmarks (Object HalBench, MME-Hal)
  are ALSO COCO-derived. So "the anchor just memorized COCO images / COCO object
  co-occurrence" (teaching-to-the-test) has no current resolution. AMBER -- the
  pre-registered non-COCO judge-free generative eval -- was dropped because its images
  are unsourceable. This script supplies the missing non-COCO generative eval from a
  primary source whose images are RELIABLY downloadable (verified below), so the
  anchor's grounding win can be re-tested on images from a different distribution and a
  different annotation pipeline that the model never saw in training. If the CHAIR
  reduction survives here, it is grounding improvement, not COCO memorization.

WHY OPEN IMAGES (verified at primary source 2026-09-05, not from a mirror).
  * Images: direct HTTPS from the CVDF public S3 bucket, no auth, no requester-pays --
    verified by actually downloading real JPEGs:
      https://open-images-dataset.s3.amazonaws.com/validation/<ImageID>.jpg  -> HTTP 200
    (This is the AMBER lesson: images must resolve at a stable primary source. They do.)
  * Ground truth: Open Images validation boxes are EXHAUSTIVELY, MANUALLY drawn for all
    instances of all positive image-level labels ("we provide exhaustive box annotation
    for all object instances, for all available positive image-level labels" --
    factsfigures_v7). This exhaustiveness is exactly what CHAIR validity needs: the GT
    object set over the scored vocabulary is complete, so a correct mention is not
    mis-scored as a hallucination. (COCO CHAIR unions caption-parsed objects precisely
    because COCO instance annotations are NOT exhaustive; here the boxes already are.)
  * Non-COCO: Open Images images are an independent Flickr collection, disjoint from
    COCO val2014 -- the images and their annotation pipeline are both non-COCO.

VOCABULARY MAPPING (Open Images 600 boxable classes -> the scorer's 80 COCO categories).
  metrics_chair.py scores exactly the 80 COCO categories (its SYN dict). We therefore
  map each Open Images boxable class name to its COCO category. 59 COCO categories match
  an OI class name case-insensitively (verified present); the remaining 21 are mapped
  by hand below (COCO_TO_OI), including one critical FALSE FRIEND: COCO "mouse" is the
  computer mouse, and OI has BOTH "Mouse" (the animal) and "Computer mouse" -- we map to
  the latter. Objects outside the 80 COCO categories are simply invisible to the scorer,
  exactly as in standard CHAIR. Every mapped OI name is gated against the official class
  list at runtime, so an upstream rename fails loudly instead of silently dropping a
  category.

OUTPUTS (house style, --out_dir):
  prompts.jsonl  {id, image, coco_id, prompt, prompt_group}  (eval_gen.py input)
                 prompt = "Please describe this image in detail." (identical to the
                 pilot CHAIR prompt in data_prep.prep_chair), image = "<ImageID>.jpg".
  gt.json        {str(ImageID): {"objects":[COCO cat names], "captions":[]}}
                 identical schema to data_prep.prep_chair's coco_gt.json; captions is
                 empty because the exhaustive boxes ARE the complete GT (no caption union
                 needed). metrics_chair.py's --coco_gt reads this unchanged.
  images/<ImageID>.jpg   the N eval images (unless --no_images)
  manifest.json  source URLs + pinned byte sizes + sha256 + counts + the full
                 COCO<-OI mapping + per-category GT frequency + seed.

USAGE (scored later with the EXISTING scorer, no new scorer needed):
  python fetch_noncoco_chair.py --out_dir oichair_data --n 500
  python eval_gen.py --prompts oichair_data/prompts.jsonl \
      --data_root oichair_data/images --out gen/OICHAIR_S2.jsonl --max_new_tokens 512
  python metrics_chair.py --gen gen/OICHAIR_S2.jsonl \
      --coco_gt oichair_data/gt.json --out_prefix results/S2_oichair --ckpt S2
"""
import argparse
import csv
import datetime
import hashlib
import io
import json
import os
import random
import sys
import urllib.request

# --- Open Images V7 primary-source URLs (verified 2026-09-05) -----------------
CLASS_DESC_URL = ("https://storage.googleapis.com/openimages/v7/"
                  "oidv7-class-descriptions-boxable.csv")
BOX_CSV_URL = ("https://storage.googleapis.com/openimages/v5/"
               "validation-annotations-bbox.csv")
IMG_LABELS_URL = ("https://storage.googleapis.com/openimages/v5/"
                  "validation-annotations-human-imagelabels-boxable.csv")
IMG_URL_TMPL = "https://open-images-dataset.s3.amazonaws.com/validation/{iid}.jpg"

# Exact byte sizes at the pinned file versions (verified by download 2026-09-05).
# A mismatch means the upstream file moved -- re-verify before trusting it, exactly
# as in fetch_objhalbench.py / fetch_amber_assets.py.
CLASS_DESC_BYTES = 12064
BOX_CSV_BYTES = 25105048

CHAIR_PROMPT = "Please describe this image in detail."  # == data_prep.prep_chair
SEED = 17  # == data_prep.SEED, so selection is reproducible in the same spirit

# --- COCO(80) <- Open Images boxable class map --------------------------------
# 59 categories match an OI class name case-insensitively at runtime; only the
# non-trivial 21 are listed here. Every DisplayName is gated against the official
# class list. Sources of each choice are documented in noncoco_generative_eval.md.
COCO_TO_OI = {
    "airplane": ["Aircraft", "Fixed-wing aircraft"],  # OI has no bare "Airplane"
    "cow": ["Cattle", "Bull"],                          # OI has no bare "Cow"
    "frisbee": ["Flying disc"],
    "skis": ["Ski"],
    "sports ball": ["Ball (Object)", "Cricket ball", "Football", "Golf ball",
                    "Rugby ball", "Tennis ball", "Volleyball (Ball)"],
    "cup": ["Coffee cup", "Mug", "Measuring cup"],      # OI has no bare "Cup"
    "orange": ["Orange (fruit)"],
    "donut": ["Doughnut"],
    "potted plant": ["Houseplant", "Flowerpot"],
    "dining table": ["Table", "Kitchen & dining room table", "Coffee table", "Desk"],
    "tv": ["Television"],
    "remote": ["Remote control"],
    "keyboard": ["Computer keyboard"],                  # not the musical keyboard
    "cell phone": ["Mobile phone", "Telephone", "Corded phone"],
    "microwave": ["Microwave oven"],
    "hair drier": ["Hair dryer"],
    # false friend: OI "Mouse" is the ANIMAL; COCO "mouse" is the computer mouse
    "mouse": ["Computer mouse"],
    # sense-tightened / enriched (COCO bear includes panda; broaden for completeness)
    "bear": ["Bear", "Brown bear", "Polar bear", "Panda"],
    "clock": ["Clock", "Wall clock", "Alarm clock", "Digital clock"],
    "knife": ["Knife", "Kitchen knife"],
    "couch": ["Couch", "Sofa bed", "Loveseat", "Studio couch"],
}

# The scorer's 80 COCO category names are the keys of its SYN dict. We import them
# from the scorer so the two can never drift apart.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics_chair import SYN  # noqa: E402
COCO80 = list(SYN.keys())
assert len(COCO80) == 80, f"expected 80 COCO categories, got {len(COCO80)}"


def _download_to(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "oichair-fetch"})
    h = hashlib.sha256()
    n = 0
    with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            h.update(chunk)
            f.write(chunk)
            n += len(chunk)
    return n, h.hexdigest()


def _download_bytes(url, retries=3):
    req = urllib.request.Request(url, headers={"User-Agent": "oichair-fetch"})
    last = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read(), r.status
        except Exception as e:  # noqa: BLE001
            last = e
    raise last


def build_label_map(class_desc_path):
    """Return (oi_label_to_coco, coco_to_labels, name_to_label).
    oi_label_to_coco: /m/... code -> COCO category name (only mapped classes).
    Gate: every COCO category resolves to >=1 valid OI /m/ code; every hand-mapped
    DisplayName exists in the official list."""
    name_to_label = {}
    with open(class_desc_path, encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for label, name in r:
            name_to_label[name.strip().lower()] = (label.strip(), name.strip())

    coco_to_labels = {}          # coco cat -> [(/m/code, DisplayName)]
    oi_label_to_coco = {}        # /m/code -> coco cat
    unresolved = []
    for c in COCO80:
        if c in COCO_TO_OI:
            names = COCO_TO_OI[c]
        else:
            names = [c]  # case-insensitive auto-match against the OI list
        got = []
        for nm in names:
            key = nm.lower()
            if key not in name_to_label:
                unresolved.append((c, nm))
                continue
            code, disp = name_to_label[key]
            got.append((code, disp))
            oi_label_to_coco[code] = c
        if not got:
            unresolved.append((c, "<no OI class resolved>"))
        coco_to_labels[c] = got

    if unresolved:
        lines = "\n".join(f"  {c!r} -> {nm!r} not found in OI class list"
                          for c, nm in unresolved)
        sys.exit("GATE FAIL: COCO<-OI mapping did not resolve (upstream rename?):\n"
                 + lines)
    return oi_label_to_coco, coco_to_labels, name_to_label


def parse_boxes(box_csv_path, oi_label_to_coco):
    """image_id -> set(COCO cats) from exhaustive validation boxes."""
    img_to_cats = {}
    with open(box_csv_path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        need = {"ImageID", "LabelName"}
        if not need.issubset(r.fieldnames):
            sys.exit(f"GATE FAIL: box CSV header {r.fieldnames} lacks {need}")
        for row in r:
            c = oi_label_to_coco.get(row["LabelName"])
            if c is not None:
                img_to_cats.setdefault(row["ImageID"], set()).add(c)
    return img_to_cats


def union_image_labels(labels_csv_path, oi_label_to_coco, img_to_cats):
    """Optionally union human-verified POSITIVE image-level labels (Confidence==1).
    Boxes are already exhaustive for positive labels, so this is belt-and-suspenders."""
    added = 0
    with open(labels_csv_path, encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not {"ImageID", "LabelName", "Confidence"}.issubset(r.fieldnames):
            sys.exit(f"GATE FAIL: image-labels CSV header {r.fieldnames} unexpected")
        for row in r:
            if row["Confidence"] != "1":
                continue
            c = oi_label_to_coco.get(row["LabelName"])
            if c is not None:
                s = img_to_cats.setdefault(row["ImageID"], set())
                if c not in s:
                    s.add(c)
                    added += 1
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=500,
                    help="number of eval images (pilot CHAIR uses 500)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--cache_dir", default=None,
                    help="dir to cache the OI CSVs (default: <out_dir>/_cache)")
    ap.add_argument("--no_download", action="store_true",
                    help="require cached CSVs; never fetch the CSVs")
    ap.add_argument("--no_images", action="store_true",
                    help="write prompts.jsonl + gt.json + manifest only, no pixels "
                         "(still verifies a few image URLs resolve)")
    ap.add_argument("--with_image_labels", action="store_true",
                    help="also union human-verified positive image-level labels "
                         "(unpinned file; boxes are already exhaustive for positives)")
    ap.add_argument("--img_retries", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache = args.cache_dir or os.path.join(args.out_dir, "_cache")
    os.makedirs(cache, exist_ok=True)

    # 1. CSVs (pinned byte sizes; a mismatch aborts) --------------------------
    prov = {}

    def _get_csv(url, name, expected_bytes):
        dest = os.path.join(cache, name)
        if os.path.exists(dest):
            with open(dest, "rb") as f:
                blob = f.read()
            nbytes, sha = len(blob), hashlib.sha256(blob).hexdigest()
            src = f"cache:{dest}"
        else:
            if args.no_download:
                sys.exit(f"--no_download but {name} not cached at {dest}")
            print(f"[oichair] downloading {name}\n          {url}", flush=True)
            nbytes, sha = _download_to(url, dest)
            src = url
        if expected_bytes is not None and nbytes != expected_bytes:
            sys.exit(f"GATE FAIL: {name} is {nbytes} bytes, expected "
                     f"{expected_bytes}; upstream file changed -- re-verify.")
        prov[name] = {"source": src, "bytes": nbytes, "sha256": sha,
                      "expected_bytes": expected_bytes}
        print(f"[oichair] {name}: {nbytes} bytes sha256={sha[:16]}...", flush=True)
        return dest

    class_desc = _get_csv(CLASS_DESC_URL, "oidv7-class-descriptions-boxable.csv",
                          CLASS_DESC_BYTES)
    box_csv = _get_csv(BOX_CSV_URL, "validation-annotations-bbox.csv", BOX_CSV_BYTES)

    # 2. mapping + GT ---------------------------------------------------------
    oi_label_to_coco, coco_to_labels, _ = build_label_map(class_desc)
    print(f"[oichair] mapped 80 COCO categories -> {len(oi_label_to_coco)} OI classes",
          flush=True)
    img_to_cats = parse_boxes(box_csv, oi_label_to_coco)
    print(f"[oichair] {len(img_to_cats)} validation images carry >=1 COCO-cat box",
          flush=True)
    if args.with_image_labels:
        labels_csv = _get_csv(IMG_LABELS_URL,
                              "validation-annotations-human-imagelabels-boxable.csv",
                              None)  # unpinned: gate header only
        added = union_image_labels(labels_csv, oi_label_to_coco, img_to_cats)
        print(f"[oichair] unioned +{added} positive image-level labels", flush=True)

    # 3. deterministic selection of N images with >=1 COCO-cat GT -------------
    candidates = sorted(iid for iid, cats in img_to_cats.items() if cats)
    if len(candidates) < args.n:
        sys.exit(f"GATE FAIL: only {len(candidates)} candidate images, need {args.n}")
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(candidates, args.n))

    # 4. write gt.json + prompts.jsonl (metrics_chair schema) -----------------
    gt = {str(iid): {"objects": sorted(img_to_cats[iid]), "captions": []}
          for iid in chosen}
    with open(os.path.join(args.out_dir, "gt.json"), "w") as f:
        json.dump(gt, f)
    prompt_rows = [{
        "id": f"oichair_{iid}",
        "image": f"{iid}.jpg",
        "coco_id": iid,             # OI ImageID; metrics_chair keys gt by str(coco_id)
        "prompt": CHAIR_PROMPT,
        "prompt_group": 0,
    } for iid in chosen]
    with open(os.path.join(args.out_dir, "prompts.jsonl"), "w") as f:
        for row in prompt_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 5. verify a few image URLs resolve, then download (unless --no_images) ---
    sample_check = chosen[:5]
    resolved = []
    for iid in sample_check:
        try:
            _, status = _download_bytes(IMG_URL_TMPL.format(iid=iid), retries=2)
            resolved.append((iid, status))
        except Exception as e:  # noqa: BLE001
            resolved.append((iid, f"FAIL {e!r}"))
    print(f"[oichair] URL resolve check (first 5): {resolved}", flush=True)
    bad = [x for x in resolved if x[1] != 200]
    if bad:
        sys.exit(f"GATE FAIL: image URLs did not resolve: {bad}")

    n_img_ok = len(sample_check)  # the 5 checked above count as verified
    if not args.no_images:
        imgdir = os.path.join(args.out_dir, "images")
        os.makedirs(imgdir, exist_ok=True)
        n_img_ok = 0
        missing = []
        for i, iid in enumerate(chosen):
            dest = os.path.join(imgdir, f"{iid}.jpg")
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                n_img_ok += 1
                continue
            try:
                blob, status = _download_bytes(IMG_URL_TMPL.format(iid=iid),
                                               retries=args.img_retries)
                if status == 200 and blob[:2] == b"\xff\xd8":  # JPEG magic
                    with open(dest, "wb") as f:
                        f.write(blob)
                    n_img_ok += 1
                else:
                    missing.append(iid)
            except Exception:  # noqa: BLE001
                missing.append(iid)
            if i % 100 == 0:
                print(f"[oichair] images {i}/{len(chosen)} ok={n_img_ok}", flush=True)
        frac = n_img_ok / len(chosen)
        if frac < 0.99:
            sys.exit(f"GATE FAIL: only {n_img_ok}/{len(chosen)} images downloaded "
                     f"({frac:.3f}); e.g. missing {missing[:5]}")
        print(f"[oichair] downloaded {n_img_ok}/{len(chosen)} images "
              f"({frac:.4f}) -> {imgdir}", flush=True)

    # 6. manifest -------------------------------------------------------------
    catfreq = {}
    for iid in chosen:
        for c in img_to_cats[iid]:
            catfreq[c] = catfreq.get(c, 0) + 1
    manifest = {
        "benchmark": "Open Images V7 validation, CHAIR-style (non-COCO generative)",
        "purpose": ("non-COCO judge-free generative hallucination eval; closes the "
                    "anchor<->COCO domain-overlap confound (AMBER replacement)."),
        "fetch_date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "image_url_template": IMG_URL_TMPL,
        "csv_provenance": prov,
        "seed": args.seed,
        "n_requested": args.n,
        "n_images": len(chosen),
        "n_candidate_images_with_coco_gt": len(candidates),
        "used_image_labels_union": bool(args.with_image_labels),
        "prompt": CHAIR_PROMPT,
        "scorer": "pilot/metrics_chair.py (judge-free; vendored 80-cat CHAIR list)",
        "coco_to_oi_map": {c: [d for _, d in coco_to_labels[c]] for c in COCO80},
        "gt_category_frequency": dict(sorted(catfreq.items(),
                                             key=lambda kv: -kv[1])),
        "note": ("Open Images validation boxes are exhaustively, manually annotated "
                 "for all positive image-level labels, so the 80-cat GT is complete "
                 "and CHAIR is valid without a caption union. Images are non-COCO and "
                 "downloaded from the CVDF public S3 bucket (no auth)."),
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[oichair] wrote prompts.jsonl ({len(prompt_rows)}), gt.json "
          f"({len(gt)} images), manifest.json to {args.out_dir}", flush=True)
    print(f"[oichair] images downloaded/verified: {n_img_ok}", flush=True)
    print("[oichair] OK", flush=True)


if __name__ == "__main__":
    main()
