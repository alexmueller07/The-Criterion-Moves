#!/usr/bin/env python3
"""Build a BALANCED probe set, so the criterion target becomes reachable without labels.

WHY (2026-09-11). Our regularizers pin the decision statistic to the frozen base
model's value, and the base sits at c = +0.431. Pinning to a mis-placed reference
inherits the mis-placement: the anchor lands at c = +0.697 and is worse than doing
nothing at every stage. The target, not the mechanism, is what is broken.

The fix follows from an identity that holds on a balanced set. Writing H for the
hit rate and FA for the false-alarm rate,

    c = 0  <=>  z(H) = -z(FA)  <=>  H = 1 - FA  <=>  yes-rate = 0.500

so on a probe whose items are half present and half absent, "put the criterion at
the optimum" and "make the model say yes half the time" are the SAME instruction —
and the second needs no labels at all, only a count of how often the model says
yes. Verified on our own cells: yes-rate is exactly 0.5*(H+FA) in every one.

The existing probe (`grounding/probe.jsonl`) cannot be used for this. It was built
deliberately WITHOUT reading instance annotations, assigning object names to images
uniformly at random, so most of its items are absent and a yes-rate of 0.5 would be
the wrong target on it.

This script reuses those same images but pairs each with one PRESENT and one ABSENT
object, read from COCO instance annotations. That is a one-time construction cost.
It does NOT make the method supervised: the emitted rows carry no labels, the
training-time term only ever counts yes-answers, and nothing about the continual
stream is annotated. The labels live in the audit report so the balance is checkable.

Negative sampling follows POPE's "random" setting: an absent object is drawn
uniformly from the COCO classes not annotated in that image. COCO annotations are
incomplete, so a "absent" object is occasionally visible but unannotated; that is
inherited from POPE's own construction and is recorded as a limitation rather than
silently assumed away. --hard_negatives switches to co-occurrence-based negatives
for a harder set.

Usage:
  python3 build_balanced_probe.py --root <data dir> [--n 600] [--seed 17]
"""
import argparse
import collections
import json
import os
import random
import re
import sys
import zipfile

COCO80 = [
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
]


def load_instances(root):
    """{coco_id: set(category names)} from the annotations zip, streamed."""
    zp = os.path.join(root, "downloads", "annotations_trainval2014.zip")
    if not os.path.isfile(zp):
        raise SystemExit("missing %s" % zp)
    with zipfile.ZipFile(zp) as z:
        member = next((m for m in z.namelist() if m.endswith("instances_val2014.json")), None)
        if member is None:
            raise SystemExit("instances_val2014.json not in %s" % zp)
        with z.open(member) as f:
            d = json.load(f)
    cats = {c["id"]: c["name"] for c in d["categories"]}
    present = collections.defaultdict(set)
    for a in d["annotations"]:
        n = cats.get(a["category_id"])
        if n:
            present[a["image_id"]].add(n)
    return present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="data dir (holds grounding/ and downloads/)")
    ap.add_argument("--probe", default=None, help="source probe.jsonl (for its image list)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=600, help="total rows; half present, half absent")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--hard_negatives", action="store_true",
                    help="draw absent objects from the classes that most often "
                         "co-occur with what IS in the image (POPE 'popular'/"
                         "'adversarial' style) instead of uniformly")
    args = ap.parse_args()

    src = args.probe or os.path.join(args.root, "grounding", "probe.jsonl")
    out = args.out or os.path.join(args.root, "grounding", "probe_balanced.jsonl")
    rows = [json.loads(l) for l in open(src) if l.strip()]
    print("[probe] source rows: %d" % len(rows))

    present = load_instances(args.root)
    print("[probe] images with instance annotations: %d" % len(present))

    # co-occurrence, only needed for the hard-negative variant
    cooc = collections.Counter()
    if args.hard_negatives:
        for objs in present.values():
            for o in objs:
                cooc[o] += 1

    rnd = random.Random(args.seed)
    made, audit = [], []
    skipped_no_ann = skipped_no_pos = 0
    for r in rows:
        m = re.search(r"(\d+)", r["id"])
        if not m:
            continue
        cid = int(m.group(1))
        objs = present.get(cid)
        if objs is None:
            skipped_no_ann += 1
            continue
        pos = sorted(objs & set(COCO80))
        if not pos:
            skipped_no_pos += 1
            continue
        neg_pool = [c for c in COCO80 if c not in objs]
        if not neg_pool:
            continue
        p = rnd.choice(pos)
        if args.hard_negatives:
            neg_pool.sort(key=lambda c: -cooc.get(c, 0))
            n = rnd.choice(neg_pool[:20])
        else:
            n = rnd.choice(neg_pool)
        for obj, truth in ((p, "present"), (n, "absent")):
            # emitted row is LABEL-FREE on purpose; truth goes to the audit only
            made.append({"id": "%s_%s" % (r["id"], truth[:3]),
                         "image": r["image"], "object": obj})
            audit.append({"id": made[-1]["id"], "coco_id": cid,
                          "object": obj, "truth": truth})

    rnd.shuffle(made)
    if args.n and len(made) > args.n:
        keep = set()
        # keep the balance exactly while subsampling
        truth_of = {a["id"]: a["truth"] for a in audit}
        want = args.n // 2
        cnt = collections.Counter()
        sub = []
        for r in made:
            t = truth_of[r["id"]]
            if cnt[t] < want:
                sub.append(r); cnt[t] += 1; keep.add(r["id"])
            if len(sub) >= args.n:
                break
        made = sub
        audit = [a for a in audit if a["id"] in keep]

    bal = collections.Counter(a["truth"] for a in audit)
    with open(out, "w") as f:
        for r in made:
            f.write(json.dumps(r) + "\n")
    rep = out.replace(".jsonl", "_report.json")
    with open(rep, "w") as f:
        json.dump({"n": len(made), "balance": dict(bal),
                   "yes_fraction": round(bal["present"] / max(1, len(made)), 4),
                   "seed": args.seed, "hard_negatives": bool(args.hard_negatives),
                   "skipped_no_annotation": skipped_no_ann,
                   "skipped_no_positive": skipped_no_pos,
                   "source_probe": src,
                   "limitation": ("absent objects are drawn from classes not "
                                  "annotated in the image; COCO annotations are "
                                  "incomplete, so a small share of 'absent' items "
                                  "may be visible but unannotated. Inherited from "
                                  "POPE's own negative construction."),
                   "audit": audit}, f, indent=2)

    print("[probe] wrote %s  (%d rows: %d present / %d absent, yes-fraction %.4f)"
          % (out, len(made), bal["present"], bal["absent"],
             bal["present"] / max(1, len(made))))
    print("[probe] audit + limitation in %s" % rep)
    print("[probe] skipped: %d without annotations, %d with no COCO-80 object"
          % (skipped_no_ann, skipped_no_pos))
    if abs(bal["present"] / max(1, len(made)) - 0.5) > 0.01:
        print("BALANCED_PROBE_WARN: not within 1% of 0.5 -- the yes-rate target "
              "assumes balance and would be biased")
        return 2
    print("BALANCED_PROBE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
