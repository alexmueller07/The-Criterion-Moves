"""Build ANNOTATION-FREE, out-of-domain (OOD) grounding pairs for the
faithfulness anchor -- a drop-in replacement source for the COCO-instance-
supervised `build_grounding_pairs.py`.

Motivation (closes the "teaching-to-the-test" limitation admitted in METHOD.md
and FULLSTUDY_PREREG.md). The current anchor is COCO val2014, the SAME domain as
both hallucination evals (POPE, CHAIR are COCO-derived). A reviewer can argue the
method memorizes the eval distribution rather than learning grounding. This script
builds the anchor from a NON-COCO image source with NO COCO instance masks, so the
anchor is image-disjoint from the eval pools by construction and the domain-shift
becomes evidence *for* generalization (see design_notes/method_ood_anchor.md).

Output (identical schema + conventions to build_grounding_pairs.py) under $ROOT:
  <subdir>/images/ood_<n>.jpg        original OOD image (re-encoded RGB JPEG)
  <subdir>/masked/ood_<n>.jpg        every box of the `present` label filled with
                                     the dataset mean color (same op as the COCO
                                     builder -- box fill, not inpaint)
  <subdir>/ood_pairs.jsonl           {id, image, masked_image, present, absent}
  <subdir>/report.json               counts + audit; written LAST (success marker)

The emitted row schema is EXACTLY build_grounding_pairs.py's:
    {"id", "image", "masked_image", "present", "absent"}
so faith_loss.FaithfulnessAnchor consumes it unchanged as a --faith_pairs source
(image / masked_image are root-relative paths; present / absent are object nouns).

--------------------------------------------------------------------------------
Two ways to supply the per-image object labels + boxes (the ONLY thing that
replaces COCO's instance annotations). Neither uses COCO:

(A) --detector owlv2   [OPTIONAL heavy path; needs torch + transformers + a GPU]
    Runs an open-vocabulary detector (OWLv2, already shipped in `transformers`:
    Owlv2Processor / Owlv2ForObjectDetection) over --images_dir with the object
    nouns in --vocab. High-confidence detections (score >= --score_hi) give the
    `present` label and its box(es) for masking; any vocab noun the detector does
    NOT find anywhere in the image (no box above --score_lo) is a verified-absent
    candidate. Works on ANY images (Objects365, LAION, a web crawl) -> maximally
    out-of-domain, no dataset labels required at all.

(B) --manifest weak_labels.jsonl   [DEFAULT light path; needs only Pillow]
    Consumes a pre-computed weak-label manifest -- one row per image:
        {"image": <path>, "width": W, "height": H,
         "boxes": [{"label": str, "bbox": [x, y, w, h], "score": float?}, ...]}
    This is the "whatever weak labels it ships" route: point it at a native OOD
    detection dataset's shipped human boxes (e.g. Objects365, Flickr-sourced,
    image-disjoint from COCO, ~15.8 boxes/image densely annotated across 365
    categories -- Shao et al., ICCV 2019), or at a cached detector run from (A).
    No detector, no GPU, no network at build time.

If --detector is requested but torch/transformers are unavailable, the script
DEGRADES to (B) when --manifest is also given (clearly stamped in report.json as
source_mode="weak_labels_fallback"); otherwise it exits with an actionable message
rather than a stack trace. The masking + pair-emission core is shared by both paths
and is dependency-light (Pillow only), which is what the offline unit test drives.

bbox convention throughout is COCO-style [x, y, w, h] (same as build_grounding_
pairs.py's a["bbox"]); OWLv2's xyxy boxes are converted at manifest-build time.
All sampling is seeded (--seed, default 17).
"""
import argparse
import io
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.gate_checks import run_all_gates

from PIL import Image, ImageDraw, ImageStat

SEED = 17
N_PAIRS = 2000
MIN_AREA_FRAC = 0.05

# MUST stay byte-identical to the row keys emitted by build_grounding_pairs.py
# (verified by reading both, 2026-09-05). faith_loss.FaithfulnessAnchor asserts
# exactly these five keys, so any drift here silently breaks the drop-in contract.
SCHEMA_KEYS = ("id", "image", "masked_image", "present", "absent")


def save_img(img, path):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(path, "JPEG", quality=92)


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Weak-label manifest (path B, and the target format the detector path emits).
# ---------------------------------------------------------------------------
def load_manifest(path, images_root):
    """Return [{img_id, image_path, W, H, boxes:[{label,bbox,score}]}], validated.

    `image` in each row resolves against images_root when not absolute. img_id is
    the basename stem, deduped with a numeric suffix so ids are unique even if two
    images share a basename across subfolders.
    """
    rows, seen = [], {}
    with open(path) as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            assert "image" in r and "boxes" in r, f"manifest line {ln}: need image+boxes"
            ip = r["image"] if os.path.isabs(r["image"]) \
                else os.path.join(images_root, r["image"])
            stem = os.path.splitext(os.path.basename(r["image"]))[0]
            if stem in seen:
                seen[stem] += 1
                img_id = f"{stem}_{seen[stem]}"
            else:
                seen[stem] = 0
                img_id = stem
            boxes = []
            for b in r["boxes"]:
                assert "label" in b and "bbox" in b, f"manifest line {ln}: box needs label+bbox"
                x, y, w, h = b["bbox"]
                assert w > 0 and h > 0, f"manifest line {ln}: non-positive box {b['bbox']}"
                boxes.append({"label": str(b["label"]),
                              "bbox": [float(x), float(y), float(w), float(h)],
                              "score": float(b.get("score", 1.0))})
            rows.append({"img_id": img_id, "image_path": ip,
                         "W": r.get("width"), "H": r.get("height"), "boxes": boxes})
    assert rows, f"empty manifest {path}"
    return rows


# ---------------------------------------------------------------------------
# Open-vocabulary detector (path A) -- OPTIONAL, heavy deps guarded inside.
# Produces the exact manifest structure load_manifest returns, so the rest of
# the pipeline is detector-agnostic.
# ---------------------------------------------------------------------------
def detect_owlv2(images_dir, vocab, model_name, score_lo, device):
    """Run OWLv2 over every image in images_dir querying `vocab`; keep boxes with
    score > score_lo. Raises ImportError (caught by the caller) if the heavy stack
    is missing. API per the google/owlv2-base-patch16-ensemble model card:
    Owlv2Processor + Owlv2ForObjectDetection, post_process_object_detection."""
    import torch  # noqa: F401  (ImportError here triggers the documented fallback)
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    files = sorted(fn for fn in os.listdir(images_dir)
                   if fn.lower().endswith(exts))
    assert files, f"no images in {images_dir}"
    proc = Owlv2Processor.from_pretrained(model_name)
    model = Owlv2ForObjectDetection.from_pretrained(model_name).to(device).eval()
    queries = [f"a photo of a {v}" for v in vocab]  # OWLv2 prompt convention
    rows = []
    for fn in files:
        img = Image.open(os.path.join(images_dir, fn)).convert("RGB")
        W, H = img.size
        inputs = proc(text=[queries], images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        target = torch.tensor([[H, W]], device=device)
        res = proc.post_process_object_detection(
            outputs=out, target_sizes=target, threshold=score_lo)[0]
        boxes = []
        for box, score, label in zip(res["boxes"].tolist(),
                                     res["scores"].tolist(),
                                     res["labels"].tolist()):
            x0, y0, x1, y1 = box  # OWLv2 returns xyxy -> convert to COCO xywh
            boxes.append({"label": vocab[int(label)],
                          "bbox": [x0, y0, x1 - x0, y1 - y0],
                          "score": float(score)})
        rows.append({"img_id": os.path.splitext(fn)[0], "image_path":
                     os.path.join(images_dir, fn), "W": W, "H": H, "boxes": boxes})
    return rows


# ---------------------------------------------------------------------------
# Shared core: manifest rows -> pairs -> masked images -> schema jsonl.
# ---------------------------------------------------------------------------
def build_pairs(rows, vocab, n_pairs, min_area_frac, score_hi, scored, rng):
    """rows: manifest structure. vocab: global label universe (defines the absent
    pool). scored: whether `score` is meaningful (detector / scored weak labels)
    -> present requires score>=score_hi and absent requires the label be entirely
    undetected. Returns (pairs, all_labels) where each pair carries the chosen
    present/absent and the present-label boxes to mask."""
    all_labels = sorted(set(vocab))
    assert len(all_labels) >= 2, "need >=2 distinct object labels to form pairs"

    # per-image present-label set (score-gated when scores are meaningful)
    label_set = {}
    for r in rows:
        labs = {b["label"] for b in r["boxes"]
                if (not scored) or b["score"] >= score_hi}
        # a box below score_hi still counts as "detected" for absent-exclusion,
        # so an object weakly seen is never called absent (guards property 3).
        seen_any = {b["label"] for b in r["boxes"]}
        label_set[r["img_id"]] = (labs, seen_any)

    # corpus co-occurrence over the confident present sets (hard-negative absent)
    cooc = defaultdict(Counter)
    for labs, _ in label_set.values():
        for p in labs:
            for c in labs:
                if c != p:
                    cooc[p][c] += 1

    by_id = {r["img_id"]: r for r in rows}
    candidates = []
    for r in sorted(rows, key=lambda z: z["img_id"]):
        labs, _ = label_set[r["img_id"]]
        if not labs or len(labs) >= len(all_labels):
            continue
        W, H = r["W"], r["H"]
        if not (W and H):  # fall back to loading the image if size absent
            with Image.open(r["image_path"]) as im:
                W, H = im.size
            r["W"], r["H"] = W, H
        best_frac, best_lab = 0.0, None
        for b in r["boxes"]:
            if scored and b["score"] < score_hi:
                continue
            frac = (b["bbox"][2] * b["bbox"][3]) / float(W * H)
            if frac >= min_area_frac and frac > best_frac:
                best_frac, best_lab = frac, b["label"]
        if best_lab is None:
            continue
        candidates.append((r["img_id"], best_lab, best_frac))
    assert len(candidates) >= n_pairs, (
        f"only {len(candidates)} eligible OOD images for {n_pairs} pairs "
        f"(min_area_frac={min_area_frac}, score_hi={score_hi if scored else 'n/a'})")
    rng.shuffle(candidates)

    pairs = []
    for img_id, present, frac in candidates[:n_pairs]:
        present_cats, seen_any = label_set[img_id]
        # absent must be in NEITHER the confident set NOR anything weakly seen.
        absent_pool = [c for c in all_labels
                       if c not in present_cats and c not in seen_any]
        assert absent_pool, f"no absent candidate for {img_id}"
        weights = [sum(cooc[p][c] for p in present_cats) for c in absent_pool]
        if sum(weights) > 0:
            absent = rng.choices(absent_pool, weights=weights, k=1)[0]
        else:
            absent = rng.choice(absent_pool)
        assert present in present_cats and absent not in present_cats, (img_id, present, absent)
        present_boxes = [b["bbox"] for b in by_id[img_id]["boxes"]
                         if b["label"] == present]
        assert present_boxes, f"no box to mask for {img_id}/{present}"
        pairs.append({"img_id": img_id, "present": present, "absent": absent,
                      "frac": frac, "image_path": by_id[img_id]["image_path"],
                      "present_boxes": present_boxes})
    return pairs, all_labels


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="data root (same dir passed as --data_root to training)")
    ap.add_argument("--subdir", default="grounding_ood",
                    help="output subdir under --root")
    src = ap.add_argument_group("label source (choose one; detector falls back to manifest)")
    src.add_argument("--manifest", default=None,
                     help="weak-label JSONL: {image,width,height,boxes:[{label,bbox,score?}]}")
    src.add_argument("--images_root", default=None,
                     help="root for relative image paths in --manifest (default: --root)")
    src.add_argument("--detector", choices=["owlv2"], default=None,
                     help="optional open-vocab detector; needs torch+transformers+GPU")
    src.add_argument("--images_dir", default=None, help="images for --detector")
    src.add_argument("--vocab", default=None,
                     help="newline-separated object nouns; required for --detector, "
                          "optional in --manifest mode (defaults to the manifest's label union)")
    src.add_argument("--detector_model", default="google/owlv2-base-patch16-ensemble")
    src.add_argument("--device", default="cuda")
    ap.add_argument("--score_hi", type=float, default=0.3,
                    help="min detector score for a `present` label (scored sources)")
    ap.add_argument("--score_lo", type=float, default=0.1,
                    help="detector keep-threshold; a label with no box above this is absent")
    ap.add_argument("--n_pairs", type=int, default=N_PAIRS)
    ap.add_argument("--min_area_frac", type=float, default=MIN_AREA_FRAC)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true", help="40 pairs")
    ap.add_argument("--no_gates", action="store_true",
                    help="skip cluster env gates (offline unit tests only)")
    args = ap.parse_args()

    images_root = args.images_root or args.root
    n_pairs = 40 if args.smoke else args.n_pairs
    rng = random.Random(args.seed)

    if not args.no_gates:
        run_all_gates(args.root, min_free_gb=10.0, need_gpu=bool(args.detector))

    vocab = None
    if args.vocab:
        with open(args.vocab) as f:
            vocab = [w.strip() for w in f if w.strip()]
        assert vocab, f"empty vocab {args.vocab}"

    # ---- obtain per-image detections (the only COCO-annotation replacement) ----
    source_mode, detector_name, scored = None, None, None
    rows = None
    if args.detector:
        assert args.images_dir and vocab, "--detector needs --images_dir and --vocab"
        try:
            rows = detect_owlv2(args.images_dir, vocab, args.detector_model,
                                args.score_lo, args.device)
            source_mode, detector_name, scored = "detector", args.detector_model, True
        except ImportError as e:
            if args.manifest:
                print(f"[ood] detector deps unavailable ({e}); FALLING BACK to "
                      f"--manifest weak-label mode", flush=True)
            else:
                sys.exit(f"[ood] --detector requested but torch/transformers "
                         f"unavailable ({e}) and no --manifest to fall back to. "
                         f"Supply a pre-computed --manifest (weak-label mode) or "
                         f"install the detector stack.")
    if rows is None:
        assert args.manifest, "supply --manifest (weak-label mode) or --detector"
        rows = load_manifest(args.manifest, images_root)
        # scored iff every box carries a non-default score; a native GT dataset
        # (Objects365) is unscored -> absent = annotation-set complement.
        scored = any(b.get("score", 1.0) != 1.0 for r in rows for b in r["boxes"])
        source_mode = "weak_labels_fallback" if args.detector else "weak_labels"
        if detector_name is None:
            detector_name = None
        if vocab is None:
            vocab = sorted({b["label"] for r in rows for b in r["boxes"]})

    absent_verification = ("detector score<=score_lo (queried full vocab)" if scored
                           else "annotation-set complement (unscored source assumed "
                                "densely annotated within its label vocabulary)")

    pairs, all_labels = build_pairs(rows, vocab, n_pairs, args.min_area_frac,
                                    args.score_hi, scored, rng)

    out = os.path.join(args.root, args.subdir)
    imgdir, maskdir = os.path.join(out, "images"), os.path.join(out, "masked")
    os.makedirs(imgdir, exist_ok=True)
    os.makedirs(maskdir, exist_ok=True)

    # pass 1: save originals, accumulate dataset mean color (same recipe as the
    # COCO builder so the fill color is drawn from the anchor domain itself)
    ch_sum, n_px = [0.0, 0.0, 0.0], 0
    for i, p in enumerate(pairs):
        img = Image.open(p["image_path"]).convert("RGB")
        s = ImageStat.Stat(img)
        for c in range(3):
            ch_sum[c] += s.sum[c]
        n_px += img.size[0] * img.size[1]
        rel = os.path.join(args.subdir, "images", f"ood_{i}.jpg")
        save_img(img, os.path.join(args.root, rel))
        p["image"] = rel
    mean_color = tuple(int(round(ch_sum[c] / n_px)) for c in range(3))

    # pass 2: fill every `present`-label box with the mean color
    for i, p in enumerate(pairs):
        img = Image.open(os.path.join(args.root, p["image"])).convert("RGB")
        draw = ImageDraw.Draw(img)
        for x, y, w, h in p["present_boxes"]:
            draw.rectangle([x, y, x + w, y + h], fill=mean_color)
        rel = os.path.join(args.subdir, "masked", f"ood_{i}.jpg")
        save_img(img, os.path.join(args.root, rel))
        p["masked_image"] = rel
        p["n_boxes"] = len(p["present_boxes"])

    rows_out = [{"id": f"ood_{i}", "image": p["image"],
                 "masked_image": p["masked_image"], "present": p["present"],
                 "absent": p["absent"]} for i, p in enumerate(pairs)]
    for r in rows_out:  # hard schema guard -- the drop-in contract
        assert tuple(r.keys()) == SCHEMA_KEYS, (list(r.keys()), SCHEMA_KEYS)
    assert len({r["id"] for r in rows_out}) == len(rows_out) == n_pairs
    write_jsonl(os.path.join(out, "ood_pairs.jsonl"), rows_out)

    report = {
        "smoke": args.smoke,
        "seed": args.seed,
        "n_pairs": len(rows_out),
        "source_mode": source_mode,
        "detector": detector_name,
        "scored": bool(scored),
        "absent_verification": absent_verification,
        "min_area_frac": args.min_area_frac,
        "score_hi": args.score_hi if scored else None,
        "score_lo": args.score_lo if scored else None,
        "vocab_size": len(all_labels),
        "n_source_images": len(rows),
        "mean_color_rgb": list(mean_color),
        "mean_present_area_frac": round(sum(p["frac"] for p in pairs) / len(pairs), 4),
        "mean_boxes_masked": round(sum(p["n_boxes"] for p in pairs) / len(pairs), 2),
        "present_top": Counter(p["present"] for p in pairs).most_common(10),
        "absent_top": Counter(p["absent"] for p in pairs).most_common(10),
        "note": ("OOD anchor: image source is NON-COCO by construction; no COCO "
                 "instance masks used. Disjoint from POPE/CHAIR eval pools by "
                 "domain. See design_notes/method_ood_anchor.md."),
    }
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[ood-pairs] {len(rows_out)} pairs, source={source_mode}, "
          f"scored={bool(scored)}, mean_color={mean_color}, vocab={len(all_labels)}",
          flush=True)


if __name__ == "__main__":
    main()
