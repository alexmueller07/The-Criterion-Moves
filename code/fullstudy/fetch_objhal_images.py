"""Download the 300 COCO val2014 images Object HalBench needs.

fetch_objhalbench.py writes prompts whose `image` field is a BARE filename
(COCO_val2014_0000003397xx.jpg) but never fetches the pixels, so every bench run
silently skipped Object HalBench. Images come from the public COCO mirror; any
image already present in another local COCO dir is hard-linked instead of refetched.
usage: fetch_objhal_images.py --root <data_ucit> [--local_dirs d1,d2] [--workers 8]
"""
import argparse, json, os, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

URL = "http://images.cocodataset.org/val2014/{name}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir containing objhal/prompts.jsonl")
    ap.add_argument("--local_dirs", default="", help="comma list of dirs that may already hold COCO val2014 jpgs")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    pj = os.path.join(a.root, "objhal", "prompts.jsonl")
    out = os.path.join(a.root, "objhal", "images"); os.makedirs(out, exist_ok=True)
    names = sorted({json.loads(l)["image"] for l in open(pj)})
    locals_ = [d for d in a.local_dirs.split(",") if d and os.path.isdir(d)]
    linked = fetched = failed = 0
    def get(name):
        nonlocal linked, fetched, failed
        dst = os.path.join(out, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0: return
        for d in locals_:
            src = os.path.join(d, name)
            if os.path.exists(src):
                try: os.link(src, dst)
                except OSError: 
                    import shutil; shutil.copy2(src, dst)
                linked += 1; return
        for attempt in range(4):
            try:
                with urllib.request.urlopen(URL.format(name=name), timeout=60) as r, open(dst + ".tmp", "wb") as f:
                    f.write(r.read())
                os.replace(dst + ".tmp", dst); fetched += 1; return
            except (urllib.error.URLError, OSError):
                if attempt == 3: failed += 1
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(get, names))
    have = sum(1 for n in names if os.path.exists(os.path.join(out, n)))
    print(f"[objhal-img] wanted {len(names)} have {have} (linked {linked}, fetched {fetched}, failed {failed})")
    assert have == len(names), f"{len(names)-have} Object HalBench images missing"
    print("OBJHAL_IMAGES_OK")

if __name__ == "__main__":
    main()
