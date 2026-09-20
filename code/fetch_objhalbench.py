"""Build the Object HalBench 300-image / 8-detail-prompt evaluation set into the
exact format our vendored CHAIR scorer (pilot/metrics_chair.py) consumes.

Object HalBench has NO standalone repo: the 300-image protocol ships inside
RLAIF-V (github.com/RLHF-V/RLAIF-V), and the official scorer is the classic
Rohrbach-et-al. CHAIR with ground truth = COCO instance segmentations UNION
objects parsed from the 5 GT captions. Verified at source 2026-09-03:
  eval/eval_gpt_obj_halbench.py       -- combine_coco_instances() + combine_coco_captions();
                                         judge-free path is caption_to_words() over synonyms_refine.txt
  eval/summarize_gpt_obj_halbench_review.py -- CHAIRs / CHAIRi / CHAIRs_refine
  script/eval/eval_rlaifv_objhal.sh   -- q_file = eval/data/obj_halbench_300_with_image.jsonl,
                                         beam=3, temperature=0
  eval/data/obj_halbench_300_with_image.jsonl -- 300 lines, keys {org_idx,image_id,question,image};
                                         300 distinct COCO image_ids, EXACTLY 8 distinct detail prompts
All pinned to RLAIF-V commit 83d917b2e8118ca8e7877a796d72af9bb3e6b356
(default branch main, "Merge pull request #45", 2025-05-14).

Because our own vendored CHAIR is byte-for-byte the same construct (GT =
instances UNION caption-parsed objects, per-mention CHAIR_i, per-response
CHAIR_s), running these 300 detail-prompted captions through metrics_chair.py IS
the judge-free ("--use_gpt" off) Object HalBench protocol. It is the POPE-paper
detail-prompt robustness check: the 8 detail-eliciting instructions are the
"Instruction 2" regime that (per POPE, arXiv 2305.10355) can double CHAIR.

WHAT THIS SCRIPT DOES / DOES NOT DOWNLOAD.
  * It fetches the canonical 300-line spec ONCE (~64 MB, images embedded as
    base64) purely to read {image_id, question}; it DISCARDS every image byte
    and writes NO image pixels. Use --spec to point at an already-stripped local
    copy and skip the network entirely.
  * COCO images for scoring are NOT fetched here -- the cluster pipeline already
    stages COCO val2014 (data_prep.prep_chair). prompts.jsonl carries the
    val2014 filename; set eval_gen.py --data_root to the val2014 image dir.
  * COCO ground-truth ANNOTATIONS are read from a LOCAL --coco_ann_dir (the
    trainval2014 annotation zip or the loose JSONs the pilot already downloads);
    nothing COCO is fetched over the network.

Outputs (house style, --out_dir):
  prompts.jsonl   {id, image, coco_id, prompt, prompt_group}   (eval_gen.py input)
  coco_gt.json    {str(coco_id): {"objects":[cat names], "captions":[<=5]}}
                  (metrics_chair.py --coco_gt; identical schema to data_prep.prep_chair)
  manifest.json   source URLs + commit + sha256 + sizes + counts + COCO provenance

Usage:
  # scored later with the EXISTING scorer, no new scorer needed:
  python fetch_objhalbench.py --out_dir objhal_data \
      --coco_ann_dir /path/with/annotations_trainval2014.zip
  python eval_gen.py --prompts objhal_data/prompts.jsonl \
      --data_root /path/to/coco/val2014_images --out gen/OBJHAL_S2.jsonl --max_new_tokens 512
  python metrics_chair.py --gen gen/OBJHAL_S2.jsonl \
      --coco_gt objhal_data/coco_gt.json --out_prefix results/S2_objhal --ckpt S2
"""
import argparse
import datetime
import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile

RLAIFV_COMMIT = "83d917b2e8118ca8e7877a796d72af9bb3e6b356"
RLAIFV_REPO = "RLHF-V/RLAIF-V"
SPEC_REPO_PATH = "eval/data/obj_halbench_300_with_image.jsonl"
SPEC_URL = (f"https://raw.githubusercontent.com/{RLAIFV_REPO}/{RLAIFV_COMMIT}/"
            f"{SPEC_REPO_PATH}")
# Verified by full download 2026-09-03 (do not "update to match" a changed file;
# a size/hash mismatch means the upstream file moved and the protocol changed).
SPEC_EXPECTED_BYTES = 63537348
SPEC_EXPECTED_SHA256 = "e89c705f3cdebc35b7863878e059c88a97506767b87ad17014d58428ce6547a7"
EXPECTED_N = 300
EXPECTED_N_PROMPTS = 8

DEFAULT_IMAGE_TEMPLATE = "COCO_val2014_{image_id:012d}.jpg"


def _download_to(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "objhal-fetch"})
    h = hashlib.sha256()
    n = 0
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            h.update(chunk)
            f.write(chunk)
            n += len(chunk)
    return n, h.hexdigest()


def load_spec(spec_path, allow_download, work_dir):
    """Return (rows, provenance). rows: [{org_idx,image_id,question}] (NO images)."""
    prov = {"source_url": SPEC_URL, "repo": f"github.com/{RLAIFV_REPO}",
            "commit": RLAIFV_COMMIT, "repo_path": SPEC_REPO_PATH}
    if spec_path is None:
        if not allow_download:
            sys.exit("no --spec given and --no_download set; provide a local "
                     "obj_halbench_300_with_image.jsonl or allow download")
        spec_path = os.path.join(work_dir, "obj_halbench_300_with_image.jsonl")
        print(f"[objhal] downloading canonical spec (~64 MB, images stripped "
              f"after parse, kept: none)\n         {SPEC_URL}", flush=True)
        nbytes, sha = _download_to(SPEC_URL, spec_path)
        prov.update({"downloaded_bytes": nbytes, "sha256": sha,
                     "expected_bytes": SPEC_EXPECTED_BYTES,
                     "expected_sha256": SPEC_EXPECTED_SHA256})
        if nbytes != SPEC_EXPECTED_BYTES or sha != SPEC_EXPECTED_SHA256:
            sys.exit(f"GATE FAIL: spec bytes/sha mismatch "
                     f"({nbytes} vs {SPEC_EXPECTED_BYTES}; {sha}); upstream file "
                     "changed -- re-verify the protocol before trusting it")
        downloaded_here = True
    else:
        prov.update({"source_url": f"local:{spec_path}"})
        h = hashlib.sha256()
        nbytes = 0
        with open(spec_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
                nbytes += len(chunk)
        prov.update({"local_bytes": nbytes, "sha256": h.hexdigest()})
        downloaded_here = False

    rows = []
    with open(spec_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for k in ("image_id", "question"):
                if k not in r:
                    sys.exit(f"{spec_path}:{lineno}: missing key {k!r}")
            rows.append({"org_idx": r.get("org_idx"),
                         "image_id": int(r["image_id"]),
                         "question": r["question"]})  # 'image' base64 dropped
    if downloaded_here:
        os.remove(spec_path)  # never keep image bytes
    return rows, prov


def _open_coco_json(coco_ann_dir, split, kind):
    """kind in {'instances','captions'}; look for loose JSON, else the trainval zip."""
    name = f"{kind}_{split}2014.json"
    loose = os.path.join(coco_ann_dir, name)
    if os.path.exists(loose):
        with open(loose, encoding="utf-8") as f:
            return json.load(f), f"file:{loose}"
    loose2 = os.path.join(coco_ann_dir, "annotations", name)
    if os.path.exists(loose2):
        with open(loose2, encoding="utf-8") as f:
            return json.load(f), f"file:{loose2}"
    zpath = os.path.join(coco_ann_dir, "annotations_trainval2014.zip")
    if os.path.exists(zpath):
        with zipfile.ZipFile(zpath) as z:
            member = f"annotations/{name}"
            if member in z.namelist():
                with z.open(member) as m:
                    return json.load(io.TextIOWrapper(m, encoding="utf-8")), \
                        f"zip:{zpath}!{member}"
    return None, None


def build_coco_gt(image_ids, coco_ann_dir, merge_train):
    """GT per image_id: category names from instances UNION nothing else here;
    caption text kept raw so metrics_chair.py parses caption objects with ITS
    own extractor (exactly as the official get_annotations_from_captions does).
    Reads val2014 (required) and, if merge_train, train2014 -- matching the
    official combine_coco_instances/captions which concatenate train+val."""
    gt_objs, gt_caps = {}, {}
    provenance = []
    splits = ["val"] + (["train"] if merge_train else [])
    got_val = False
    for split in splits:
        inst, isrc = _open_coco_json(coco_ann_dir, split, "instances")
        caps, csrc = _open_coco_json(coco_ann_dir, split, "captions")
        if inst is None or caps is None:
            if split == "val":
                sys.exit(f"GATE FAIL: could not find instances_val2014.json + "
                         f"captions_val2014.json (loose or in "
                         f"annotations_trainval2014.zip) under {coco_ann_dir}")
            print(f"[objhal] note: {split}2014 annotations absent; skipping "
                  "(fine if all 300 ids are val images)", flush=True)
            continue
        if split == "val":
            got_val = True
        cat_by_id = {c["id"]: c["name"] for c in inst["categories"]}
        wanted = set(image_ids)
        # An image listed in this split but carrying NO instance annotations has a
        # legitimately EMPTY ground-truth object set -- it is not missing data. For
        # CHAIR that is the strictest case (any object mentioned is a hallucination),
        # so seed it explicitly; otherwise such images fall into `missing` and the
        # benchmark refuses to score. Two of Object HalBench's 300 (276731, 474398)
        # are exactly this: present in val2014 with captions, zero instances, and
        # absent from train2014 entirely (verified 2026-09-09).
        for im in inst.get("images", []):
            if im["id"] in wanted:
                gt_objs.setdefault(im["id"], set())
        for a in inst["annotations"]:
            iid = a["image_id"]
            if iid in wanted:
                gt_objs.setdefault(iid, set()).add(cat_by_id[a["category_id"]])
        for c in caps["annotations"]:
            iid = c["image_id"]
            if iid in wanted:
                gt_caps.setdefault(iid, []).append(c["caption"])
        provenance.append({"split": split, "instances": isrc, "captions": csrc})
    assert got_val
    missing = [iid for iid in image_ids
               if iid not in gt_objs or iid not in gt_caps]
    if missing:
        print(f"[objhal] WARNING: {len(missing)} of {len(image_ids)} image_ids "
              "have no COCO instance+caption GT in the provided annotations: "
              f"{missing[:10]}{'...' if len(missing) > 10 else ''}", flush=True)
        print("[objhal] if these are COCO train2014 images, re-run with "
              "--merge_train and a dir that also has the train2014 annotations; "
              "the RLHF-V paper states the 300 are sampled from COCO val, but the "
              "official scorer combines train+val -- no silent drops here.",
              flush=True)
    coco_gt = {}
    for iid in image_ids:
        if iid in gt_objs and iid in gt_caps:
            coco_gt[str(iid)] = {"objects": sorted(gt_objs[iid]),
                                 "captions": gt_caps[iid][:5]}
    return coco_gt, provenance, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--coco_ann_dir", required=True,
                    help="dir with instances_/captions_val2014.json (loose, in "
                         "an annotations/ subdir, or inside "
                         "annotations_trainval2014.zip)")
    ap.add_argument("--spec", default=None,
                    help="local obj_halbench_300_with_image.jsonl "
                         "(else downloaded from the pinned RLAIF-V commit)")
    ap.add_argument("--no_download", action="store_true",
                    help="require --spec; never touch the network")
    ap.add_argument("--merge_train", action="store_true",
                    help="also read train2014 annotations (official combines "
                         "train+val); needs them present under --coco_ann_dir")
    ap.add_argument("--image_template", default=DEFAULT_IMAGE_TEMPLATE,
                    help="prompts.jsonl 'image' field; {image_id} is the COCO id")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows, spec_prov = load_spec(args.spec, not args.no_download, args.out_dir)

    # sanity gates (no silent acceptance of a changed upstream file)
    ids = [r["image_id"] for r in rows]
    prompts = sorted({r["question"] for r in rows})
    if len(rows) != EXPECTED_N:
        sys.exit(f"GATE FAIL: {len(rows)} spec rows, expected {EXPECTED_N}")
    if len(set(ids)) != EXPECTED_N:
        sys.exit(f"GATE FAIL: {len(set(ids))} distinct image_ids, expected "
                 f"{EXPECTED_N}")
    if len(prompts) != EXPECTED_N_PROMPTS:
        sys.exit(f"GATE FAIL: {len(prompts)} distinct detail prompts, expected "
                 f"{EXPECTED_N_PROMPTS}")
    prompt_group = {q: i for i, q in enumerate(prompts)}

    coco_gt, coco_prov, missing = build_coco_gt(
        sorted(set(ids)), args.coco_ann_dir, args.merge_train)

    out_rows = []
    for r in rows:
        iid = r["image_id"]
        out_rows.append({
            "id": f"objhal_{iid}",
            "image": args.image_template.format(image_id=iid),
            "coco_id": iid,
            "prompt": r["question"],
            "prompt_group": prompt_group[r["question"]],
        })

    prompts_path = os.path.join(args.out_dir, "prompts.jsonl")
    with open(prompts_path, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    gt_path = os.path.join(args.out_dir, "coco_gt.json")
    with open(gt_path, "w") as f:
        json.dump(coco_gt, f)

    manifest = {
        "benchmark": "Object HalBench (RLAIF-V line, arXiv 2312.00849)",
        "fetch_date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "spec_provenance": spec_prov,
        "coco_gt_provenance": coco_prov,
        "n_prompts": len(out_rows),
        "n_images": len(set(ids)),
        "n_detail_prompts": len(prompts),
        "detail_prompts": prompts,
        "n_with_gt": len(coco_gt),
        "n_missing_gt": len(missing),
        "missing_gt_image_ids": missing,
        "image_template": args.image_template,
        "scorer": "pilot/metrics_chair.py (judge-free; vendored CHAIR list)",
        "note": ("official generation is beam=3 temp=0; our eval_gen.py is greedy "
                 "-- a documented protocol deviation. Use --max_new_tokens >= 512 "
                 "for the detail-prompt regime and report truncation_rate."),
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[objhal] wrote {prompts_path} ({len(out_rows)} prompts), "
          f"{gt_path} ({len(coco_gt)} images with GT), manifest.json", flush=True)
    if missing:
        print(f"[objhal] {len(missing)} image(s) lack GT -- see warning above", flush=True)


if __name__ == "__main__":
    main()
