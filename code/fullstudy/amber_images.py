"""Acquire AMBER's 1,004 generative-query images and build amber/prompts.jsonl.

Sources are tried in order from amber_sources.json (filled from the dependency
probe's verified candidates — never guessed): each entry is either
  {"type": "hf_dataset", "repo": ..., "image_col": ..., "id_col": ...}
  {"type": "gdrive_zip", "file_id": ..., "inner_dir": ...}
After acquisition the set is VALIDATED against query_generative.json: every
queried image must exist on disk (>=99% required; failures listed). Then
prompts.jsonl rows: {id, amber_id, image, prompt} with the official generative
query text.
"""
import argparse
import json
import os
import sys


def from_hf(spec, img_dir, needed):
    import datasets
    datasets.disable_progress_bars()
    from datasets import load_dataset
    ds = load_dataset(spec["repo"], split=spec.get("split", "train"))
    got = 0
    for ex in ds:
        name = str(ex[spec["id_col"]])
        if not name.endswith(".jpg"):
            name = name + ".jpg"
        if name in needed:
            ex[spec["image_col"]].convert("RGB").save(os.path.join(img_dir, name),
                                                      "JPEG", quality=95)
            got += 1
    return got


def from_gdrive(spec, img_dir, needed):
    import subprocess
    import tempfile
    import zipfile
    with tempfile.TemporaryDirectory() as td:
        zp = os.path.join(td, "amber.zip")
        subprocess.run([sys.executable, "-m", "gdown", "--id", spec["file_id"],
                        "-O", zp], check=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(td)
        inner = os.path.join(td, spec.get("inner_dir", "image"))
        got = 0
        for name in needed:
            src = os.path.join(inner, name)
            if os.path.exists(src):
                os.replace(src, os.path.join(img_dir, name))
                got += 1
    return got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="amber data dir (has query_generative.json)")
    args = ap.parse_args()
    queries = json.load(open(os.path.join(args.out, "query_generative.json")))
    needed = {q["image"] for q in queries}
    img_dir = os.path.join(args.out, "image")
    os.makedirs(img_dir, exist_ok=True)
    missing = {n for n in needed if not os.path.exists(os.path.join(img_dir, n))}
    print(f"[amber] need {len(needed)} images, missing {len(missing)}", flush=True)

    if missing:
        sources = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "amber_sources.json")))
        for spec in sources:
            print(f"[amber] trying source: {spec['type']} {spec.get('repo', spec.get('file_id'))}",
                  flush=True)
            try:
                got = (from_hf if spec["type"] == "hf_dataset" else from_gdrive)(
                    spec, img_dir, missing)
                print(f"[amber] source yielded {got}", flush=True)
            except Exception as e:
                print(f"[amber] source failed: {e!r}", flush=True)
            missing = {n for n in needed if not os.path.exists(os.path.join(img_dir, n))}
            if not missing:
                break

    frac = 1 - len(missing) / len(needed)
    assert frac >= 0.99, f"AMBER images incomplete: {len(missing)} missing, e.g. {sorted(missing)[:5]}"
    rows = []
    for q in queries:
        if q["image"] in missing:
            continue
        rows.append({"id": f"amber_{q['id']}", "amber_id": q["id"],
                     "image": os.path.join("amber", "image", q["image"]),
                     "prompt": q["query"]})
    with open(os.path.join(args.out, "prompts.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[amber] prompts.jsonl: {len(rows)} rows; coverage {frac:.4f}", flush=True)
    print("[amber] OK", flush=True)


if __name__ == "__main__":
    main()
