"""Prepare the DCL (domain continual learning) split of MLLM-CL for this harness.

DCL is the five-domain half of MLLM-CL (Zhao et al., arXiv:2506.05453): remote
sensing, medical, autonomous driving, science and finance. The HuggingFace repo
MLLM-CL/DCL is 67.9 GB across 151 parquet shards, which is far more than we
need: our UCIT tasks are subsampled to 8,000 train and 500 val each, so a few
shards per domain suffice.

This script therefore works SHARD BY SHARD and deletes each raw shard once its
rows are extracted, so peak disk is one shard rather than one domain. That
matters here: the node's free space oscillates by ~78 GB on other users'
activity, and the training gate aborts below 40 GB.

It does not hardcode the column layout. The repo's schema is discovered at
runtime -- the image column is whichever column holds bytes or a dict with a
'bytes' key, and the question/answer columns are matched by name against a
list of known conventions -- and the discovered mapping is printed so it can be
checked before any conversion happens.

    python3 dcl_prep.py --inspect Med              # schema only, one shard
    python3 dcl_prep.py --domain Med --n_train 8000 --n_val 500
"""
import argparse
import io
import json
import os
import shutil
import sys

REPO = "MLLM-CL/DCL"
DOMAINS = ["RS", "Med", "AD", "Sci", "Fin"]
Q_KEYS = ["question", "problem", "query", "instruction", "text", "conversations", "prompt"]
A_KEYS = ["answer", "solution", "response", "label", "gt", "target", "output"]


def load_api():
    from huggingface_hub import HfApi
    return HfApi()


def shards_for(api, domain, split):
    info = api.repo_info(REPO, repo_type="dataset", files_metadata=True)
    out = [(f.rfilename, f.size or 0) for f in info.siblings
           if f.rfilename.startswith(domain + "/") and split in f.rfilename]
    return sorted(out)


def find_columns(row):
    """Discover the image column and the question/answer columns from one row."""
    img = None
    for k, v in row.items():
        if isinstance(v, (bytes, bytearray)):
            img = k
            break
        if isinstance(v, dict) and "bytes" in v:
            img = k
            break
        # MLLM-CL/DCL stores images as a LIST of {bytes, path} dicts, one per
        # turn. Checking only the scalar shapes leaves img=None, after which
        # every row fails the `if not b: continue` test and the domain writes
        # zero rows while still exiting 0 -- which is exactly what happened on
        # the first attempt (job 21284).
        if isinstance(v, list) and v and isinstance(v[0], dict) and "bytes" in v[0]:
            img = k
            break
        if isinstance(v, list) and v and isinstance(v[0], (bytes, bytearray)):
            img = k
            break
    q = next((k for k in Q_KEYS if k in row), None)
    a = next((k for k in A_KEYS if k in row), None)
    if q is None or a is None:
        text = [k for k, v in row.items() if isinstance(v, str) and k != img]
        q = q or (text[0] if text else None)
        a = a or (text[1] if len(text) > 1 else None)
    return img, q, a


def as_text(v):
    """Conversation-style fields arrive as a list of role/content dicts."""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        parts = []
        for m in v:
            if isinstance(m, dict):
                parts.append(str(m.get("value") or m.get("content") or ""))
            else:
                parts.append(str(m))
        return "\n".join(p for p in parts if p)
    return "" if v is None else str(v)


def image_bytes(v):
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    if isinstance(v, dict) and v.get("bytes"):
        return bytes(v["bytes"])
    if isinstance(v, list) and v:
        return image_bytes(v[0])
    return None


def answer_stats(targets):
    yes = no = una = 0
    lens = []
    for t in targets:
        a = t.strip().lower()
        yes += a.startswith("yes")
        no += a.startswith("no")
        una += "unanswerable" in a[:24]
        lens.append(len(a.split()))
    lens.sort()
    n = len(targets)
    if not n:
        return None
    return {"yes_frac": round(yes / n, 4), "no_frac": round(no / n, 4),
            "refusal_frac": round(una / n, 4),
            "mean_target_words": round(sum(lens) / n, 2),
            "p95_target_words": lens[int(0.95 * n)]}


def build_manifest(out_root):
    """Assemble dcl_manifest.json from whatever domains finished preparing."""
    import random
    man = {"benchmark": REPO, "seed": 17, "tasks": {}, "orders": {}}
    have = []
    for d in DOMAINS:
        f = os.path.join(out_root, "tasks", d, "train.jsonl")
        if not os.path.exists(f):
            continue
        tg = [json.loads(l)["target"] for l in open(f)]
        st = answer_stats(tg)
        if st is None:
            continue
        man["tasks"][d] = {"n_train": len(tg),
                           "max_new_tokens": int(max(16, min(128, 2 * st["p95_target_words"] + 8))),
                           "answer_stats": st}
        have.append(d)
    if not have:
        sys.exit("no prepared domain has a train.jsonl; refusing to write a manifest")
    man["orders"]["d1"] = have
    man["orders"]["d2"] = list(reversed(have))
    o3 = list(have)
    random.Random(41).shuffle(o3)
    man["orders"]["d3"] = o3
    mp = os.path.join(out_root, "dcl_manifest.json")
    json.dump(man, open(mp, "w"), indent=1)
    print("manifest frozen at %s over %d domains: %s" % (mp, len(have), have))
    for d in have:
        print("  %-4s n_train=%-6d max_new=%-4d %s"
              % (d, man["tasks"][d]["n_train"], man["tasks"][d]["max_new_tokens"],
                 man["tasks"][d]["answer_stats"]))
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=DOMAINS)
    ap.add_argument("--inspect", choices=DOMAINS)
    ap.add_argument("--manifest", action="store_true",
                    help="build dcl_manifest.json from the already-prepared domains")
    ap.add_argument("--out_root", default=os.path.expanduser("~/cl-halluc/data_dcl"))
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--min_free_gb", type=float, default=45.0,
                    help="abort before downloading a shard if free space is below this")
    a = ap.parse_args()

    if a.manifest:
        build_manifest(a.out_root)
        return

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    api = load_api()

    if a.inspect:
        name, size = shards_for(api, a.inspect, "test")[0]
        print("inspecting %s (%.0f MB)" % (name, size / 1e6))
        p = hf_hub_download(REPO, name, repo_type="dataset")
        f = pq.ParquetFile(p)
        row = next(f.iter_batches(batch_size=1)).to_pylist()[0]
        img, q, ans = find_columns(row)
        print("rows in shard: %d" % f.metadata.num_rows)
        print("columns: %s" % [c.name for c in f.schema_arrow])
        print("discovered mapping -> image=%r question=%r answer=%r" % (img, q, ans))
        for k, v in row.items():
            d = ("<%d bytes>" % len(image_bytes(v) or b"")) if k == img else repr(v)[:200]
            print("  %-14s %s" % (k, d))
        return

    if not a.domain:
        sys.exit("give --domain or --inspect")

    free_gb = shutil.disk_usage(os.path.expanduser("~")).free / 1e9
    if free_gb < a.min_free_gb:
        sys.exit("ABORT: only %.1f GB free, below the %.1f GB floor. The training gate "
                 "aborts below 40 GB and running arms matter more than this." % (free_gb, a.min_free_gb))

    out = os.path.join(a.out_root, "tasks", a.domain)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    mapping = None
    written = {"train": 0, "val": 0}
    kept = {"train": [], "val": []}
    targets = {"train": a.n_train, "val": a.n_val}

    for split, hf_split in (("train", "train"), ("val", "test")):
        fh = open(os.path.join(out, "%s.jsonl" % split), "w")
        for name, size in shards_for(api, a.domain, hf_split):
            if written[split] >= targets[split]:
                break
            free_gb = shutil.disk_usage(os.path.expanduser("~")).free / 1e9
            if free_gb < a.min_free_gb:
                print("  stopping early: %.1f GB free" % free_gb)
                break
            print("  shard %s (%.0f MB), have %d/%d" % (name, size / 1e6, written[split], targets[split]))
            p = hf_hub_download(REPO, name, repo_type="dataset")
            try:
                pf = pq.ParquetFile(p)
                for batch in pf.iter_batches(batch_size=64):
                    for row in batch.to_pylist():
                        if written[split] >= targets[split]:
                            break
                        if mapping is None:
                            mapping = find_columns(row)
                            print("  discovered mapping -> image=%r prompt=%r target=%r" % mapping)
                            if mapping[0] is None or mapping[1] is None or mapping[2] is None:
                                sys.exit("ABORT: incomplete column mapping %r on columns %r. "
                                         "Writing rows under this mapping yields an empty or "
                                         "unusable task." % (mapping, sorted(row)))
                        ic, qc, ac = mapping
                        b = image_bytes(row.get(ic))
                        if not b:
                            continue
                        prompt = as_text(row.get(qc)).replace("<image>", "").strip()
                        target = as_text(row.get(ac)).strip()
                        if not prompt or not target:
                            continue          # unusable row; do not pad the count
                        j = written[split]
                        rid = "%s_%s%d" % (a.domain, "tr" if split == "train" else "va", j)
                        # Path is relative to the DATA ROOT (data_root/tasks/<D>/images/..),
                        # matching UCIT exactly -- the loader joins it onto data_root, so a
                        # task-relative path silently resolves to a missing file.
                        rel = os.path.join("tasks", a.domain, "images", "%s.jpg" % rid)
                        with open(os.path.join(a.out_root, rel), "wb") as im:
                            im.write(b)
                        rec = {"id": rid, "image": rel, "prompt": prompt, "target": target}
                        if split == "val":
                            rec["meta"] = {"answers": [target]}
                        fh.write(json.dumps(rec) + "\n")
                        kept[split].append(target)
                        written[split] += 1
            finally:
                # delete the raw shard immediately; peak disk stays at one shard
                try:
                    os.remove(p)
                except OSError:
                    pass
        fh.close()
        print("  %s: wrote %d" % (split, written[split]))

    meta = {"domain": a.domain, "mapping": {"image": mapping[0], "prompt": mapping[1],
                                            "target": mapping[2]} if mapping else None,
            "n_train": written["train"], "n_val": written["val"], "source": REPO,
            "answer_stats": answer_stats(kept["train"])}
    json.dump(meta, open(os.path.join(out, "prep_meta.json"), "w"), indent=1)
    if not written["train"]:
        sys.exit("ABORT: %s produced 0 training rows -- refusing to mark it prepared" % a.domain)
    open(os.path.join(out, "PREP_DONE"), "w").write("ok\n")
    print("done: %s" % json.dumps(meta))


if __name__ == "__main__":
    main()
