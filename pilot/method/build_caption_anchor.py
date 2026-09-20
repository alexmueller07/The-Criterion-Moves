"""Build the caption anchor set for the policy anchor's EOS/stopping-hazard term
(method/policy_anchor.py, Term 3; design_notes/method_multiaxis.md).

Output under $ROOT (same root as build_grounding_pairs.py):
  grounding/caption_anchor.jsonl          {id, image, caption}
  grounding/caption_anchor_report.json    counts + audit fields; written LAST

Rows are the grounding-pair images (grounding/grounding_pairs.jsonl) joined to
their COCO val2014 ground-truth captions (annotations/captions_val2014.json in
the archive data_prep.py already downloaded). Those images are by construction
DISJOINT from the CHAIR and POPE pools (build_grounding_pairs.py filters them and
writes grounding/report.json as its success marker, required here), and AMBER is
not COCO, so the caption set is disjoint from every eval pool. Supervision stays
ground truth: the reference text is a human caption, the reference hazard is the
base model's own along it -- no previous-stage checkpoint enters.

Caption text per image (default --mode joined): all of the image's COCO captions
in annotation-id order, joined into one paragraph (~50-60 tokens, so the hazard
trajectory reaches the length regime CHAIR@60 reads). --mode single keeps only
the lowest-annotation-id caption (~12 tokens). Which images are kept is a seeded
shuffle (SEED) of the pairs; the caption choice itself is deterministic.
Downloads nothing.
"""
import argparse
import io
import json
import os
import random
import re
import sys
import zipfile
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.gate_checks import run_all_gates

SEED = 17
N_CAPS = 500
GROUND_ID = re.compile(r"^ground_(\d+)$")
MODES = ("joined", "single")


def clean(text):
    return " ".join(str(text).split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=N_CAPS,
                    help="caption rows to keep (seeded subset of the pairs)")
    ap.add_argument("--mode", default="joined", choices=MODES)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run_all_gates(args.root, min_free_gb=10.0, need_gpu=False)
    n_keep = 40 if args.smoke else args.n

    gdir = os.path.join(args.root, "grounding")
    pairs_path = os.path.join(gdir, "grounding_pairs.jsonl")
    report_path = os.path.join(gdir, "report.json")
    ann_zip = os.path.join(args.root, "downloads", "annotations_trainval2014.zip")
    for p in (pairs_path, report_path, ann_zip):
        assert os.path.exists(p), (
            f"missing {p} - run data_prep.py and build_grounding_pairs.py first "
            "(report.json is the pairs builder's success marker; without it the "
            "POPE/CHAIR disjointness of the pair images is not established)")

    with open(pairs_path) as f:
        pairs = [json.loads(l) for l in f if l.strip()]
    assert pairs, f"empty pairs file {pairs_path}"
    assert len(pairs) >= n_keep, f"only {len(pairs)} pairs for --n {n_keep}"

    with zipfile.ZipFile(ann_zip) as z:
        caps = json.load(io.TextIOWrapper(z.open("annotations/captions_val2014.json"),
                                          encoding="utf-8"))
    caps_by_img = defaultdict(list)
    for a in caps["annotations"]:
        caps_by_img[int(a["image_id"])].append((int(a["id"]), clean(a["caption"])))

    rng = random.Random(SEED)
    order = list(pairs)
    rng.shuffle(order)

    rows, n_caps_used = [], []
    for r in order:
        if len(rows) >= n_keep:
            break
        m = GROUND_ID.match(str(r["id"]))
        assert m, f"unparseable pair id {r['id']!r} (expected ground_<coco_id>)"
        iid = int(m.group(1))
        cands = sorted(c for c in caps_by_img.get(iid, []) if c[1])
        assert cands, f"no COCO caption for image {iid} (pair {r['id']})"
        if args.mode == "single":
            text = cands[0][1]
            n_caps_used.append(1)
        else:
            text = " ".join(c[1] for c in cands)
            n_caps_used.append(len(cands))
        assert os.path.exists(os.path.join(args.root, r["image"])), (
            f"missing pair image {r['image']}")
        rows.append({"id": r["id"], "image": r["image"], "caption": text})
    assert len(rows) == n_keep, (len(rows), n_keep)
    assert len({r["id"] for r in rows}) == len(rows)

    out = os.path.join(gdir, "caption_anchor.jsonl")
    with open(out + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(out + ".tmp", out)

    words = [len(r["caption"].split()) for r in rows]
    report = {
        "smoke": args.smoke, "seed": SEED, "mode": args.mode, "n": len(rows),
        "source_pairs": os.path.abspath(pairs_path),
        "pairs_report": json.load(open(report_path)),
        "mean_words": round(sum(words) / len(words), 2),
        "min_words": min(words), "max_words": max(words),
        "mean_captions_joined": round(sum(n_caps_used) / len(n_caps_used), 2),
    }
    with open(os.path.join(gdir, "caption_anchor_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[captions] {len(rows)} rows ({args.mode}), mean {report['mean_words']} "
          f"words -> {out}", flush=True)


if __name__ == "__main__":
    main()
