"""Build AMBER's discriminative generation prompts -> amber/prompts_disc.jsonl.

Sibling of fullstudy/amber_images.py (which builds the GENERATIVE prompts.jsonl).
The discriminative scorer pilot/metrics_amber_disc.py already exists and reads a
gen JSONL of {id, amber_id, output, n_new_tokens, truncated}; what was missing is
the PROMPT set that eval_gen.py runs the model over to PRODUCE that gen JSONL.
This builder closes that gap.

INPUT  : <out>/query_discriminative.json  (fetched by pilot/fetch_amber_assets.py;
         14,216 rows, ids 1005-15220, schema {id, image, query}, verified at
         AMBER commit 534babf6bbfcce2e735c26289dedfb21cef3c939).
OUTPUT : <out>/prompts_disc.jsonl, rows {id, amber_id, image, prompt}
         -- the SAME schema fullstudy/amber_images.py emits, so eval_gen.py runs
         it unchanged and metrics_amber_disc.py scores the result unchanged.

Row construction (mirrors amber_images.py field-for-field):
  id       f"amber_{q['id']}"            (AMBER ids are globally unique across
                                          generative 1-1004 and discriminative
                                          1005-15220, so no collision)
  amber_id q['id']                       (the key metrics_amber_disc joins on)
  image    os.path.join("amber","image",q['image'])   (== amber_images.py)
  prompt   q['query'] + answer suffix

PROMPT SUFFIX (disclosed, auditable knob). AMBER's query_discriminative.json
carries the bare yes/no question ("Is the sky sunny in this image?"); its README
(commit 534babf...) states the expected discriminative response is exactly
"Yes"/"No" but mandates NO prompt template -- prompt phrasing is left to the
runner. To elicit a one-word yes/no answer (which the official-exact branch of
metrics_amber_disc.py needs, response == 'Yes'/'No'), we append a fixed
instruction. The default, " Please answer yes or no.", matches the phrasing that
the OFFICIAL MME data bakes into its own question text (see fetch_mme.py), so
AMBER-disc and MME-Hall prompts are structurally identical yes/no items. It is
fully configurable (--answer_suffix, empty string = raw query) and recorded in
the manifest so the exact prompt is reproducible and comparable across arms.

IMAGE DEPENDENCY (read this).  AMBER's images are distributed ONLY via a Google
Drive link in the AMBER README and BOTH Hugging Face mirror candidates were
verified ABSENT from the Hub (see fullstudy/PARK_REQUESTS.md, FULLSTUDY_PREREG.md
-- AMBER is currently DROPPED because its images are unsourceable). This builder
is cheap and correct and is written so the discriminative path is ready the
moment AMBER images resolve; it does NOT itself make the images appear. The
'image' paths point at <data_root>/amber/image/AMBER_*.jpg exactly like the
generative builder, so eval_gen.py will only succeed once those files exist.
--require_images turns the missing-image check into a hard failure (default: a
warning + coverage report, so the prompt file is still produced).

Usage:
  python build_amber_disc_prompts.py --out data_ucit/amber
  # (then, IF AMBER images resolve, eval_gen over prompts_disc.jsonl ->
  #  metrics_amber_disc.py, as wired in fullstudy/sb_fs_bench.sbatch.)
"""
import argparse
import datetime
import json
import os
import sys

# discriminative annotation type strings (must match metrics_amber_disc.TYPE_MAP
# / AMBER inference.py); used only for the optional annotations cross-check.
DISCRIMINATIVE_TYPES = {
    "discriminative-hallucination",
    "discriminative-attribute-state",
    "discriminative-attribute-number",
    "discriminative-attribute-action",
    "discriminative-relation",
    "relation",  # official 'else' bucket in inference.py
}
DEFAULT_SUFFIX = "Please answer yes or no."


def build_prompt(query, suffix):
    q = str(query).rstrip()
    s = suffix.strip()
    return f"{q} {s}" if s else q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="AMBER data dir (has query_discriminative.json); "
                         "prompts_disc.jsonl is written here")
    ap.add_argument("--query_file", default=None,
                    help="override path to query_discriminative.json")
    ap.add_argument("--answer_suffix", default=DEFAULT_SUFFIX,
                    help="instruction appended to each query "
                         "(empty string = raw query)")
    ap.add_argument("--annotations", default=None,
                    help="annotations.json to cross-check ids are discriminative "
                         "(default: <out>/annotations.json if present)")
    ap.add_argument("--require_images", action="store_true",
                    help="fail if <out>/image/ is missing referenced images "
                         "(default: warn + report coverage, still write prompts)")
    args = ap.parse_args()

    query_path = args.query_file or os.path.join(args.out,
                                                 "query_discriminative.json")
    if not os.path.exists(query_path):
        sys.exit(f"missing {query_path} (run fetch_amber_assets.py first)")
    queries = json.load(open(query_path, encoding="utf-8"))
    if not isinstance(queries, list) or not queries:
        sys.exit(f"{query_path}: expected a non-empty JSON list")

    # optional annotation cross-check (loud gate against a wrong-task query file)
    ann_path = args.annotations or os.path.join(args.out, "annotations.json")
    ann_by_id = None
    if os.path.exists(ann_path):
        ann_by_id = {a["id"]: a for a in json.load(open(ann_path,
                                                        encoding="utf-8"))}

    rows = []
    truth_counts = {"yes": 0, "no": 0, "unknown": 0}
    seen_ids = set()
    for q in queries:
        for k in ("id", "image", "query"):
            if k not in q:
                sys.exit(f"{query_path}: row missing key {k!r}: {q}")
        qid = q["id"]
        if qid in seen_ids:
            sys.exit(f"{query_path}: duplicate id {qid}")
        seen_ids.add(qid)
        if ann_by_id is not None:
            ann = ann_by_id.get(qid)
            if ann is None:
                sys.exit(f"{query_path}: id {qid} absent from annotations.json")
            if ann.get("type") not in DISCRIMINATIVE_TYPES:
                sys.exit(f"{query_path}: id {qid} type {ann.get('type')!r} is "
                         "NOT discriminative -- wrong query file for this "
                         "builder (metrics_amber_disc would reject it too)")
            truth_counts[ann.get("truth", "unknown")] = \
                truth_counts.get(ann.get("truth", "unknown"), 0) + 1
        rows.append({
            "id": f"amber_{qid}",
            "amber_id": qid,
            "image": os.path.join("amber", "image", q["image"]),
            "prompt": build_prompt(q["query"], args.answer_suffix),
        })

    # image-dependency report (AMBER images are unsourceable today; see header)
    img_dir = os.path.join(args.out, "image")
    needed_imgs = {q["image"] for q in queries}
    present = sum(1 for n in needed_imgs
                  if os.path.exists(os.path.join(img_dir, n)))
    coverage = present / max(1, len(needed_imgs))
    if coverage < 1.0:
        msg = (f"[amber-disc] IMAGE DEPENDENCY: {present}/{len(needed_imgs)} "
               f"referenced images present under {img_dir} (coverage "
               f"{coverage:.3f}). AMBER images are Google-Drive-only and both HF "
               "mirrors were verified absent; eval_gen over prompts_disc.jsonl "
               "will only run once these images resolve.")
        print(msg, flush=True)
        if args.require_images and coverage < 0.99:
            sys.exit("[amber-disc] --require_images set and coverage < 0.99; "
                     "aborting (prompts NOT written).")

    out_path = os.path.join(args.out, "prompts_disc.jsonl")
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "benchmark": "AMBER discriminative prompts (14,216 yes/no queries)",
        "build_date_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "query_file": query_path,
        "source": "github.com/junyangwang0410/AMBER data/query/"
                  "query_discriminative.json (pinned commit in "
                  "fetch_amber_assets.py)",
        "n_prompts": len(rows),
        "answer_suffix": args.answer_suffix,
        "prompt_format": "<query> + ' ' + <answer_suffix>",
        "schema": ["id", "amber_id", "image", "prompt"],
        "id_scheme": "amber_<amber_id>",
        "image_path_template": "amber/image/<AMBER_*.jpg> "
                               "(relative to eval_gen --data_root)",
        "scorer": "pilot/metrics_amber_disc.py (No is the positive class)",
        "annotations_crosscheck": (None if ann_by_id is None else
                                   {"checked": True,
                                    "truth_counts": truth_counts}),
        "image_dependency": {
            "referenced_images": len(needed_imgs),
            "present_on_disk": present,
            "coverage": round(coverage, 4),
            "note": "AMBER images unsourceable (Google-Drive-only; HF mirrors "
                    "absent). prompts_disc.jsonl is complete and correct; it "
                    "runs only once the AMBER image set resolves.",
        },
    }
    with open(os.path.join(args.out, "prompts_disc_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[amber-disc] wrote {out_path} ({len(rows)} prompts), "
          "prompts_disc_manifest.json", flush=True)
    if ann_by_id is not None:
        print(f"[amber-disc]   annotation cross-check OK; truth "
              f"yes={truth_counts.get('yes', 0)} no={truth_counts.get('no', 0)}",
              flush=True)
    print(f"[amber-disc]   answer_suffix={args.answer_suffix!r}", flush=True)
    print("[amber-disc] OK", flush=True)


if __name__ == "__main__":
    main()
