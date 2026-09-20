"""Fetch AMBER's official scoring data files (no images) into a local dir.

Downloads the four data files the generative scorer (pilot/metrics_amber.py)
needs, from raw.githubusercontent.com pinned to a specific commit of
github.com/junyangwang0410/AMBER, verifies byte sizes, and writes a
manifest.json recording URL + sha256 + size + fetch date per file.

Pinned commit (read and verified 2026-09-02): 534babf6bbfcce2e735c26289dedfb21cef3c939
(repo HEAD, "Update README.md", 2024-01-15). Expected byte sizes below are the
exact blob sizes at that commit (from the GitHub git/trees API); with the
default --ref they are enforced exactly, with a different --ref only
non-emptiness is enforced (and a warning is printed).

Images are handled separately by the cluster pipeline (Google Drive link in
the AMBER README) — this script deliberately does NOT fetch images.

Usage:
  python fetch_amber_assets.py --out_dir /path/to/amber_data
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import urllib.request

PINNED_COMMIT = "534babf6bbfcce2e735c26289dedfb21cef3c939"
REPO = "junyangwang0410/AMBER"

# repo-relative path -> (local filename, exact byte size at PINNED_COMMIT)
# query_discriminative.json (14,216 yes/no queries, ids 1005-15220) added
# 2026-09-03 for the discriminative scorer (metrics_amber_disc.py); size verified
# via the GitHub contents API at PINNED_COMMIT. annotations.json already carries
# the discriminative truth/type, so the scorer needs only annotations.json; this
# query file is for building the discriminative generation prompts.
FILES = {
    "data/query/query_generative.json": ("query_generative.json", 106220),
    "data/query/query_discriminative.json": ("query_discriminative.json", 1738529),
    "data/annotations.json": ("annotations.json", 1813301),
    "data/relation.json": ("relation.json", 12699),
    "data/safe_words.txt": ("safe_words.txt", 57),
}


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "amber-asset-fetch"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ref", default=PINNED_COMMIT,
                    help="git ref to fetch from (default: pinned commit)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pinned = args.ref == PINNED_COMMIT
    if not pinned:
        print(f"WARNING: --ref {args.ref} != pinned commit {PINNED_COMMIT}; "
              "exact byte-size checks disabled (non-emptiness only).", flush=True)

    manifest = {
        "repo": f"github.com/{REPO}",
        "ref": args.ref,
        "pinned_commit": PINNED_COMMIT,
        "fetch_date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "files": {},
    }
    failures = []
    for repo_path, (local_name, expected_size) in FILES.items():
        url = f"https://raw.githubusercontent.com/{REPO}/{args.ref}/{repo_path}"
        try:
            blob = fetch(url)
        except Exception as e:
            failures.append(f"{repo_path}: fetch failed: {e}")
            continue
        size_ok = (len(blob) == expected_size) if pinned else (len(blob) > 0)
        if not size_ok:
            failures.append(f"{repo_path}: size {len(blob)} != expected {expected_size}")
            continue
        dest = os.path.join(args.out_dir, local_name)
        with open(dest, "wb") as f:
            f.write(blob)
        entry = {
            "url": url,
            "bytes": len(blob),
            "expected_bytes_at_pinned_commit": expected_size,
            "sha256": hashlib.sha256(blob).hexdigest(),
        }
        manifest["files"][local_name] = entry
        print(f"OK {local_name}: {len(blob)} bytes sha256={entry['sha256'][:16]}...", flush=True)

    if failures:
        for msg in failures:
            print("FAIL " + msg, file=sys.stderr, flush=True)
        sys.exit(1)

    mpath = os.path.join(args.out_dir, "manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {mpath}", flush=True)


if __name__ == "__main__":
    main()
