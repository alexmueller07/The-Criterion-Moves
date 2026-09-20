"""Build the MME-Hallucination prompt set (existence/count/position/color) from
the official MME eval tool, in the format eval_gen.py + metrics_mme.py consume.

Everything needed to SCORE (questions + ground-truth yes/no) ships in the 107 KB
official eval tool -- no image download here. Verified at source 2026-09-03:
  github.com/BradyFU/Awesome-Multimodal-Large-Language-Models, branch Evaluation
  @ dd2950902889cd614d4edf606827240166d29381, file tools/eval_tool.zip
  (sha256 b8125e2a7c3418e5761c12b3cfe4f1624b3c53ba44009e75b7d3f797d3d8acee).
  Inside: Your_Results/{existence,count,position,color}.txt, each 60 lines =
  30 images x 2 questions, tab-separated "image_name \t question \t gt_answer",
  balanced 120 Yes / 120 No across the four files.

Image layout (verified by reading the ZIP central directory of
MME_Benchmark_release_version.zip on huggingface.co/datasets/darkyarding/MME):
  MME_Benchmark_release_version/MME_Benchmark/<subtask>/<image_name>.jpg
so prompts.jsonl 'image' = "MME_Benchmark/<subtask>/<name>" and eval_gen.py
--data_root should be the extracted MME_Benchmark_release_version/ directory.
Bulk images are staged by the cluster pipeline, not here.

Outputs (--out_dir):
  prompts.jsonl   {id, image, prompt, subtask, gt, mme_image}
                  (image popped by eval_gen.py; subtask/gt/mme_image pass through
                   to metrics_mme.py; mme_image = basename, pairs the 2 questions)
  manifest.json   source URL + commit + sha256 + per-subtask counts

Usage:
  python fetch_mme.py --out_dir mme_data
  python eval_gen.py --prompts mme_data/prompts.jsonl \
      --data_root /path/to/MME_Benchmark_release_version --out gen/MME_S2.jsonl \
      --max_new_tokens 8
  python metrics_mme.py --gen gen/MME_S2.jsonl --out_prefix results/S2_mme --ckpt S2
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

MME_COMMIT = "dd2950902889cd614d4edf606827240166d29381"
MME_REPO = "BradyFU/Awesome-Multimodal-Large-Language-Models"
ZIP_REPO_PATH = "tools/eval_tool.zip"
ZIP_URL = (f"https://raw.githubusercontent.com/{MME_REPO}/{MME_COMMIT}/"
           f"{ZIP_REPO_PATH}")
ZIP_EXPECTED_BYTES = 107320
ZIP_EXPECTED_SHA256 = ("b8125e2a7c3418e5761c12b3cfe4f1624b3c53ba44009e75b7d3f797"
                       "d3d8acee")
SUBTASKS = ["existence", "count", "position", "color"]
DEFAULT_IMAGE_TEMPLATE = "MME_Benchmark/{subtask}/{name}"


def get_zip_bytes(zip_path, allow_download):
    if zip_path is not None:
        with open(zip_path, "rb") as f:
            blob = f.read()
        return blob, f"local:{zip_path}"
    if not allow_download:
        sys.exit("no --zip given and --no_download set")
    req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "mme-fetch"})
    with urllib.request.urlopen(req, timeout=120) as r:
        blob = r.read()
    return blob, ZIP_URL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--zip", default=None,
                    help="local eval_tool.zip (else download pinned commit)")
    ap.add_argument("--no_download", action="store_true")
    ap.add_argument("--image_template", default=DEFAULT_IMAGE_TEMPLATE,
                    help="prompts.jsonl 'image'; {subtask} and {name} substituted")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    blob, src = get_zip_bytes(args.zip, not args.no_download)
    sha = hashlib.sha256(blob).hexdigest()
    if args.zip is None:
        if len(blob) != ZIP_EXPECTED_BYTES or sha != ZIP_EXPECTED_SHA256:
            sys.exit(f"GATE FAIL: eval_tool.zip {len(blob)}B sha {sha} != "
                     f"expected {ZIP_EXPECTED_BYTES}B {ZIP_EXPECTED_SHA256}")

    rows = []
    per_sub_counts = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = z.namelist()
        for st in SUBTASKS:
            member = next((n for n in names
                           if n.endswith(f"Your_Results/{st}.txt")), None)
            if member is None:
                sys.exit(f"GATE FAIL: {st}.txt not found in eval_tool.zip")
            text = io.TextIOWrapper(z.open(member), encoding="utf-8").read()
            n_lines = 0
            seen_img_q = {}
            for lineno, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    sys.exit(f"{st}.txt:{lineno}: expected 3 tab fields "
                             f"(image, question, gt), got {len(parts)}")
                name, question, gt = parts[0], parts[1], parts[2].strip().lower()
                if gt not in ("yes", "no"):
                    sys.exit(f"{st}.txt:{lineno}: gt {parts[2]!r} not yes/no")
                base = os.path.splitext(name)[0]
                qi = seen_img_q.get((st, base), 0)
                seen_img_q[(st, base)] = qi + 1
                rows.append({
                    "id": f"mme_{st}_{base}_{qi}",
                    "image": args.image_template.format(subtask=st, name=name),
                    "prompt": question,
                    "subtask": st,
                    "gt": gt,
                    "mme_image": f"{st}/{base}",  # unique per (subtask,image)
                })
                n_lines += 1
            imgs = len({k for k in seen_img_q})
            per_sub_counts[st] = {"questions": n_lines, "images": imgs}
            bad = [k for k, v in seen_img_q.items() if v != 2]
            if bad:
                sys.exit(f"{st}.txt: images without exactly 2 questions: {bad[:5]}")

    prompts_path = os.path.join(args.out_dir, "prompts.jsonl")
    with open(prompts_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "benchmark": "MME-Hallucination (existence/count/position/color; "
                     "MME arXiv 2306.13394)",
        "fetch_date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_url": src,
        "repo": f"github.com/{MME_REPO}",
        "branch_commit": MME_COMMIT,
        "eval_tool_zip_sha256": sha,
        "eval_tool_zip_bytes": len(blob),
        "subtasks": per_sub_counts,
        "n_prompts": len(rows),
        "image_template": args.image_template,
        "scorer": "pilot/metrics_mme.py (judge-free acc/acc+; official prefix-4 parse)",
        "note": "score = acc*100 + acc+*100 per subtask (max 200); "
                "MME-Hallucination headline = sum of the 4 subtasks (max 800).",
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[mme] wrote {prompts_path} ({len(rows)} prompts across "
          f"{len(per_sub_counts)} subtasks), manifest.json", flush=True)
    for st, c in per_sub_counts.items():
        print(f"[mme]   {st}: {c['questions']} questions / {c['images']} images",
              flush=True)


if __name__ == "__main__":
    main()
