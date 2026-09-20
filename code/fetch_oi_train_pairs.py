"""Build the OUT-OF-DOMAIN faithfulness-anchor pair set from the Open Images V7
TRAIN split, so the GT anchor can be trained on non-COCO images that are disjoint
from every eval (POPE/CHAIR/ObjHal/MME are COCO; the non-COCO CHAIR eval is Open
Images VALIDATION, built by fetch_noncoco_chair.py).

WHY TRAIN, WHY OPEN IMAGES (design_notes/method_ood_anchor.md, ood_anchor_openimages.md).
  method_ood_anchor.md recommends "Strategy B": a native OOD detection dataset's
  human boxes. Its named source, Objects365, is registration-gated (Biendata/BAAI
  login; noncoco_generative_eval.md sec.3) -- the AMBER failure mode -- so it is
  NOT used. Open Images is the verified-downloadable replacement: images on the
  CVDF public S3 bucket, no auth (fetch_noncoco_chair.py proved the validation
  split; this script verifies the TRAIN prefix by downloading real JPEGs), boxes
  drawn by humans, images a Flickr collection independent of COCO.

DISJOINTNESS (the property the OOD anchor exists for).
  * vs the COCO evals: Open Images is non-COCO by construction (image-disjoint
    from COCO val2014; noncoco_generative_eval.md sec.3, verified 2026-09-05).
  * vs the non-COCO CHAIR eval: that eval uses the VALIDATION split; this anchor
    uses the TRAIN split. Open Images partitions ImageIDs across splits and the
    S3 bucket routes them by split prefix (train/<ID>.jpg vs validation/<ID>.jpg;
    a train ID 404s under /validation/ and vice versa -- verified 2026-09-08 and
    re-checked at runtime for 3 sample IDs). On top of the by-construction
    argument, --exclude_prompts <noncoco/prompts.jsonl> ASSERTS (a) no excluded
    ImageID occurs anywhere in the streamed TRAIN CSV and (b) no selected ImageID
    is in the exclusion set. A hit in (a) means the split assumption is broken
    and the build aborts.

PIPELINE.
  1. Class map: reuse fetch_noncoco_chair.build_label_map / COCO_TO_OI (the
     COCO-80 <-> Open Images mapping incl. the mouse -> "Computer mouse" false
     friend), gated against the official oidv7 class list (pinned 12,064 B).
  2. Stream the TRAIN box CSV (oidv6-train-annotations-bbox.csv, 2,258,447,590 B,
     ~14.6M rows) straight off HTTP -- never loaded into memory, optionally
     tee'd to --cache_csv. Rows are grouped per ImageID (the file is sorted by
     ImageID; monotone order is asserted, so a group can never be split).
     An image is ELIGIBLE iff it has >=1 box of a COCO-80-mapped class whose
     normalized area (XMax-XMin)*(YMax-YMin) >= --min_area_frac (the same
     dominant-instance rule build_ood_pairs.py applies in pixels), and it has no
     mapped IsDepiction / IsInside box (P1 ambiguity: a drawing of a cat, a photo
     from inside a car). IsGroupOf boxes are kept (they are masked, which is what
     P2 needs) unless --drop_group_images.
  3. Seeded reservoir sampling (Algorithm R, --seed) of ceil(N * --oversample)
     eligible images -- one pass, O(K) memory, deterministic for the pinned CSV.
  4. Download those images from https://open-images-dataset.s3.amazonaws.com/
     train/<ImageID>.jpg (JPEG-magic checked; >=99% must succeed) into
     <root>/<subdir>/_src/, read W,H with Pillow, and write the weak-label
     manifest in EXACTLY the format build_ood_pairs.py consumes:
        {"image", "width", "height", "boxes":[{"label", "bbox":[x,y,w,h]}]}
     with COCO-style pixel [x,y,w,h] boxes (converted from the normalized CSV).
  5. Invoke build_ood_pairs.main() (imported, not re-implemented) with
     --root <root> --subdir <subdir> --vocab <COCO-80> --n_pairs N: it picks the
     dominant instance as `present`, a co-occurrence-weighted hard negative with
     no box on the image as `absent`, fills every `present` box with the dataset
     mean color, and emits the anchor schema {id, image, masked_image, present,
     absent} with paths RELATIVE to <root> ("grounding_oi/images/<id>.jpg"),
     exactly how data/grounding/ is laid out for faith_loss.FaithfulnessAnchor.
  6. Write <root>/<subdir>/grounding_pairs.jsonl (the anchor file run_arm.py's
     FAITH_PAIRS env points at), provenance.jsonl (pair id -> OI ImageID, group
     flag, area), cooc_corpus.json (80x80 co-occurrence over ALL eligible train
     images, for audit) and report.json LAST (success marker).

USAGE (cluster; see design_notes/ood_anchor_openimages.md):
  python fetch_oi_train_pairs.py --root /home/alexmueller/cl-halluc/data \
      --exclude_prompts /home/alexmueller/cl-halluc/data_ucit/noncoco/prompts.jsonl \
      --n 2000 --workers 8
  FAITH_PAIRS=$D/grounding_oi/grounding_pairs.jsonl  (run_arm.py env override)
"""
import argparse
import csv
import datetime
import hashlib
import io
import json
import math
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                          # fetch_noncoco_chair, metrics_chair
sys.path.insert(0, os.path.join(HERE, "method"))  # build_ood_pairs
import fetch_noncoco_chair as fnc  # noqa: E402  (verified OI validation pipeline)
import build_ood_pairs as bop      # noqa: E402  (anchor-pair schema owner)

# --- Open Images V7 TRAIN split primary-source facts (verified 2026-09-08) ------
TRAIN_BOX_CSV_URL = ("https://storage.googleapis.com/openimages/v6/"
                     "oidv6-train-annotations-bbox.csv")
TRAIN_BOX_CSV_BYTES = 2258447590          # Content-Length, HEAD 2026-09-08
TRAIN_BOX_CSV_ETAG = "3c3e70cfaba5757ea5c2604b19cac3b2"  # informational (GCS md5)
TRAIN_IMG_URL_TMPL = "https://open-images-dataset.s3.amazonaws.com/train/{iid}.jpg"
VAL_IMG_URL_TMPL = fnc.IMG_URL_TMPL       # .../validation/{iid}.jpg (negative control)
CLASS_DESC_URL, CLASS_DESC_BYTES = fnc.CLASS_DESC_URL, fnc.CLASS_DESC_BYTES

# Header of the train box CSV (first 13 of 21 columns are what we read).
BOX_COLS = ("ImageID", "Source", "LabelName", "Confidence", "XMin", "XMax",
            "YMin", "YMax", "IsOccluded", "IsTruncated", "IsGroupOf",
            "IsDepiction", "IsInside")

SEED = 17                       # == data_prep.SEED == build_*_pairs.SEED
N_PAIRS = bop.N_PAIRS           # 2000, held fixed across anchor sources
MIN_AREA_FRAC = bop.MIN_AREA_FRAC
OVERSAMPLE = 1.15               # download slack for the >=99% image gate
SUBDIR = "grounding_oi"
SRC_SUBDIR = "_src"
COCO80 = fnc.COCO80
SCHEMA_KEYS = bop.SCHEMA_KEYS


class GateError(RuntimeError):
    """A loud, actionable abort (repo gate convention)."""


# ---------------------------------------------------------------------------
# CSV streaming
# ---------------------------------------------------------------------------
class _HashingReader(io.RawIOBase):
    """Binary read-through wrapper: counts bytes, sha256s them, optional tee."""

    def __init__(self, raw, tee=None):
        self.raw, self.tee = raw, tee
        self.nbytes, self.sha = 0, hashlib.sha256()

    def readable(self):
        return True

    def close(self):
        # TextIOWrapper -> BufferedReader -> here: release the HTTP response /
        # file handle too (RawIOBase.close alone only flips our own flag).
        try:
            if not self.closed:
                self.raw.close()
        finally:
            super().close()

    def readinto(self, b):
        chunk = self.raw.read(len(b))
        n = len(chunk)
        if n:
            b[:n] = chunk
            self.nbytes += n
            self.sha.update(chunk)
            if self.tee is not None:
                self.tee.write(chunk)
        return n


def open_box_csv(csv_path=None, url=TRAIN_BOX_CSV_URL, expected_bytes=TRAIN_BOX_CSV_BYTES,
                 cache_csv=None, allow_unpinned=False):
    """Return (text_stream, hashing_reader, provenance). Streams from HTTP unless
    csv_path is given. The byte-size pin is checked up front (Content-Length /
    file size) and again after the full read by the caller."""
    prov = {"url": url if csv_path is None else None, "path": csv_path,
            "expected_bytes": None if allow_unpinned else expected_bytes}
    tee = None
    if csv_path is not None:
        size = os.path.getsize(csv_path)
        if not allow_unpinned and size != expected_bytes:
            raise GateError(f"GATE FAIL: {csv_path} is {size} bytes, expected "
                            f"{expected_bytes}; upstream file changed -- re-verify.")
        raw = open(csv_path, "rb")
    else:
        req = urllib.request.Request(url, headers={"User-Agent": "oitrain-fetch"})
        raw = urllib.request.urlopen(req, timeout=600)
        clen = raw.headers.get("Content-Length")
        prov["content_length"] = int(clen) if clen else None
        prov["etag"] = (raw.headers.get("ETag") or "").strip('"')
        if not allow_unpinned and prov["content_length"] != expected_bytes:
            raise GateError(f"GATE FAIL: {url} Content-Length={clen}, expected "
                            f"{expected_bytes}; upstream file changed -- re-verify.")
        if cache_csv:
            os.makedirs(os.path.dirname(os.path.abspath(cache_csv)), exist_ok=True)
            tee = open(cache_csv + ".part", "wb")
    hr = _HashingReader(raw, tee)
    text = io.TextIOWrapper(io.BufferedReader(hr, buffer_size=1 << 20),
                            encoding="utf-8", newline="")
    return text, hr, prov


def iter_image_groups(reader):
    """Yield (ImageID, [row, ...]) for contiguous ImageID runs. Asserts the file
    is sorted by ImageID (monotone non-decreasing) so a group is never split;
    a violation aborts -- silently mis-grouped boxes would poison P2."""
    header = next(reader)
    if tuple(header[:len(BOX_COLS)]) != BOX_COLS:
        raise GateError(f"GATE FAIL: train box CSV header {header[:13]} != {BOX_COLS}")
    cur, rows, prev = None, [], None
    for row in reader:
        if not row:
            continue
        if len(row) != len(header):
            raise GateError(f"GATE FAIL: malformed CSV row (len {len(row)} != "
                            f"{len(header)}): {row[:3]}... (truncated stream?)")
        iid = row[0]
        if iid != cur:
            if cur is not None:
                yield cur, rows
                if iid < cur:
                    raise GateError(f"GATE FAIL: train box CSV not sorted by ImageID "
                                    f"({iid!r} after {cur!r}); per-image grouping "
                                    f"unsafe -- a two-pass build is required.")
            cur, rows = iid, []
        rows.append(row)
    if cur is not None:
        yield cur, rows


# ---------------------------------------------------------------------------
# Per-image filter (the only place the OI flags are interpreted)
# ---------------------------------------------------------------------------
def filter_image(iid, rows, oi_label_to_coco, min_area_frac, keep_depictions=False,
                 keep_inside=False, drop_group_images=False, stats=None):
    """Return an eligible-image record or None.

    record = {"iid", "boxes": [{"label", "nbox": [xmin, ymin, xmax, ymax],
              "group": bool, "source": str}], "labels": set, "dom_label",
              "dom_frac"}. Eligibility mirrors build_ood_pairs.build_pairs: the
       largest mapped box must cover >= min_area_frac of the image (normalized
       coords make that scale-free). Depiction / inside boxes of a mapped class
       reject the image (P1 ambiguity) unless kept; group boxes are kept and
       masked unless --drop_group_images."""
    st = stats if stats is not None else Counter()
    boxes = []
    reject = None
    for row in rows:
        cat = oi_label_to_coco.get(row[2])
        if cat is None:
            continue
        if row[3] != "1":              # train boxes are all Confidence 1; be strict
            st["skipped_confidence"] += 1
            continue
        xmin, xmax, ymin, ymax = (float(row[4]), float(row[5]),
                                  float(row[6]), float(row[7]))
        if not (0.0 <= xmin < xmax <= 1.0 and 0.0 <= ymin < ymax <= 1.0):
            st["degenerate_boxes"] += 1
            continue
        group, depiction, inside = row[10] == "1", row[11] == "1", row[12] == "1"
        if depiction and not keep_depictions:
            reject = reject or "depiction"
        if inside and not keep_inside:
            reject = reject or "inside"
        if group and drop_group_images:
            reject = reject or "group"
        if group:
            st["group_boxes"] += 1
        boxes.append({"label": cat, "nbox": [xmin, ymin, xmax, ymax],
                      "group": group, "source": row[1]})
    if not boxes:
        return None
    st["mapped_images"] += 1
    if reject:
        st[f"rejected_{reject}"] += 1
        return None
    dom_label, dom_frac = None, 0.0
    for b in boxes:
        x0, y0, x1, y1 = b["nbox"]
        frac = (x1 - x0) * (y1 - y0)
        if frac >= min_area_frac and frac > dom_frac:
            dom_label, dom_frac = b["label"], frac
    if dom_label is None:
        st["rejected_small"] += 1
        return None
    st["eligible"] += 1
    return {"iid": iid, "boxes": boxes, "labels": {b["label"] for b in boxes},
            "dom_label": dom_label, "dom_frac": dom_frac}


# ---------------------------------------------------------------------------
# Selection: one streaming pass, seeded reservoir, disjointness gate.
# ---------------------------------------------------------------------------
def select_images(reader, oi_label_to_coco, k, rng, min_area_frac=MIN_AREA_FRAC,
                  excluded=frozenset(), keep_depictions=False, keep_inside=False,
                  drop_group_images=False, log_every=200000, log=print):
    """Stream the grouped CSV once; return (reservoir, stats, cooc, excluded_hits).

    reservoir: k eligible records chosen by Algorithm R with `rng` (deterministic
    for a fixed CSV + seed). cooc: 80x80 label co-occurrence over ALL eligible
    images (audit). excluded_hits: ImageIDs from `excluded` seen in the CSV --
    any hit means a validation-split ID lives in the TRAIN file (split broken)."""
    stats = Counter()
    cooc = defaultdict(Counter)
    reservoir, n_elig, hits = [], 0, []
    t0 = time.time()
    for iid, rows in iter_image_groups(reader):
        stats["images"] += 1
        stats["rows"] += len(rows)
        if iid in excluded:
            hits.append(iid)
        rec = filter_image(iid, rows, oi_label_to_coco, min_area_frac,
                           keep_depictions, keep_inside, drop_group_images, stats)
        if stats["images"] % log_every == 0:
            log(f"[oitrain] {stats['images']} images / {stats['rows']} rows streamed, "
                f"eligible={stats['eligible']} ({time.time() - t0:.0f}s)")
        if rec is None:
            continue
        labs = rec["labels"]
        for p in labs:
            for c in labs:
                if c != p:
                    cooc[p][c] += 1
        n_elig += 1
        if len(reservoir) < k:
            reservoir.append(rec)
        else:
            j = rng.randrange(n_elig)
            if j < k:
                reservoir[j] = rec
    stats["eligible_total"] = n_elig
    return reservoir, stats, cooc, hits


def load_exclude_ids(paths):
    """ImageIDs to exclude. JSONL rows in the fetch_noncoco_chair prompts schema
    ({id:"oichair_<ID>", image:"<ID>.jpg", coco_id:"<ID>"}) or one ID per line."""
    ids = set()
    for p in paths:
        if not os.path.isfile(p):
            raise GateError(f"GATE FAIL: exclusion source {p} does not exist. For the "
                            f"validation-eval ids build it first: fetch_noncoco_chair.py "
                            f"--out_dir <data_ucit>/noncoco (writes prompts.jsonl).")
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    r = json.loads(line)
                    iid = (r.get("coco_id") or
                           os.path.splitext(os.path.basename(r.get("image", "")))[0] or
                           str(r.get("id", "")).split("_", 1)[-1])
                else:
                    iid = os.path.splitext(os.path.basename(line))[0]
                if not iid:
                    raise GateError(f"GATE FAIL: could not read an ImageID from {p}: {line[:80]}")
                ids.add(str(iid))
    return ids


def assert_disjoint(selected_ids, excluded, csv_hits):
    if csv_hits:
        raise GateError(f"GATE FAIL: {len(csv_hits)} excluded (validation-eval) ImageIDs "
                        f"occur in the TRAIN box CSV, e.g. {sorted(csv_hits)[:5]} -- the "
                        f"train/validation split assumption is broken; do not proceed.")
    overlap = set(selected_ids) & set(excluded)
    if overlap:
        raise GateError(f"GATE FAIL: selected train images overlap the exclusion set: "
                        f"{sorted(overlap)[:5]}")


# ---------------------------------------------------------------------------
# Images + weak-label manifest
# ---------------------------------------------------------------------------
def _fetch_one(iid, dest, retries, url_tmpl=TRAIN_IMG_URL_TMPL):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return iid, True, "cached"
    try:
        blob, status = fnc._download_bytes(url_tmpl.format(iid=iid), retries=retries)
    except Exception as e:  # noqa: BLE001
        return iid, False, repr(e)
    if status == 200 and blob[:2] == b"\xff\xd8":  # JPEG magic
        with open(dest + ".part", "wb") as f:
            f.write(blob)
        os.replace(dest + ".part", dest)
        return iid, True, "ok"
    return iid, False, f"status={status} magic={blob[:2]!r}"


def download_images(ids, src_dir, retries=4, workers=8, log=print):
    os.makedirs(src_dir, exist_ok=True)
    ok, missing = [], {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_fetch_one, iid, os.path.join(src_dir, f"{iid}.jpg"), retries)
                for iid in ids]
        for i, fu in enumerate(futs):
            iid, good, why = fu.result()
            (ok.append(iid) if good else missing.__setitem__(iid, why))
            if (i + 1) % 200 == 0:
                log(f"[oitrain] images {i + 1}/{len(ids)} ok={len(ok)}")
    return ok, missing


def url_resolve_check(ids, log=print):
    """3 sample train URLs must be HTTP 200; the same IDs under /validation/ must
    NOT be 200 (cross-split negative control; recorded, informational)."""
    res = []
    for iid in ids:
        try:
            blob, status = fnc._download_bytes(TRAIN_IMG_URL_TMPL.format(iid=iid), retries=2)
            train_ok = status == 200 and blob[:2] == b"\xff\xd8"
        except Exception as e:  # noqa: BLE001
            train_ok, status = False, repr(e)
        try:
            _, vstatus = fnc._download_bytes(VAL_IMG_URL_TMPL.format(iid=iid), retries=1)
        except urllib.error.HTTPError as e:
            vstatus = e.code
        except Exception as e:  # noqa: BLE001
            vstatus = repr(e)
        res.append({"iid": iid, "train_status": status, "train_jpeg": train_ok,
                    "validation_status": vstatus})
    log(f"[oitrain] URL resolve check: {res}")
    bad = [r for r in res if not r["train_jpeg"]]
    if bad:
        raise GateError(f"GATE FAIL: train image URLs did not resolve: {bad}")
    return res


def write_manifest(records, src_dir, manifest_path, image_rel_dir=SRC_SUBDIR):
    """Weak-label manifest in build_ood_pairs.load_manifest's format. Boxes go
    from normalized [xmin,ymin,xmax,ymax] to pixel COCO [x,y,w,h] using the
    downloaded image's real size. Extra keys (oi_image_id, is_group_of, source)
    are ignored by the builder and kept for audit."""
    from PIL import Image
    n = 0
    with open(manifest_path, "w") as f:
        for rec in records:
            path = os.path.join(src_dir, f"{rec['iid']}.jpg")
            with Image.open(path) as im:
                W, H = im.size
            boxes = []
            for b in rec["boxes"]:
                x0, y0, x1, y1 = b["nbox"]
                boxes.append({"label": b["label"],
                              "bbox": [x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H],
                              "is_group_of": bool(b["group"]), "source": b["source"]})
            f.write(json.dumps({"image": f"{image_rel_dir}/{rec['iid']}.jpg",
                                "width": W, "height": H, "boxes": boxes,
                                "oi_image_id": rec["iid"]}) + "\n")
            n += 1
    return n


# ---------------------------------------------------------------------------
# Builder invocation (build_ood_pairs.py owns the schema + masking)
# ---------------------------------------------------------------------------
def run_builder(root, subdir, manifest, images_root, vocab_path, n_pairs, seed,
                min_area_frac, no_gates):
    argv = ["build_ood_pairs.py", "--root", root, "--subdir", subdir,
            "--manifest", manifest, "--images_root", images_root,
            "--vocab", vocab_path, "--n_pairs", str(n_pairs),
            "--min_area_frac", str(min_area_frac), "--seed", str(seed)]
    if no_gates:
        argv.append("--no_gates")
    saved = sys.argv
    sys.argv = argv
    try:
        bop.main()
    finally:
        sys.argv = saved


def rederive_pairs(manifest, images_root, vocab, n_pairs, min_area_frac, seed):
    """Re-run the builder's pure selection (same rows, same seed) to recover the
    pair index -> source ImageID map that its emitted rows do not carry."""
    rows = bop.load_manifest(manifest, images_root)
    pairs, _ = bop.build_pairs(rows, vocab, n_pairs, min_area_frac,
                               score_hi=0.3, scored=False, rng=random.Random(seed))
    return pairs


def _dir_bytes(d):
    return sum(os.path.getsize(os.path.join(d, fn)) for fn in os.listdir(d)) \
        if os.path.isdir(d) else 0


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="data root (the --data_root training resolves pair paths against)")
    ap.add_argument("--subdir", default=SUBDIR, help="output subdir under --root")
    ap.add_argument("--n", type=int, default=N_PAIRS, help="pairs to emit (fixed 2000)")
    ap.add_argument("--oversample", type=float, default=OVERSAMPLE,
                    help="download ceil(n*oversample) images so the >=99%% gate has slack")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--min_area_frac", type=float, default=MIN_AREA_FRAC)
    ap.add_argument("--csv_path", default=None,
                    help="local copy of oidv6-train-annotations-bbox.csv (else streamed)")
    ap.add_argument("--cache_csv", default=None,
                    help="tee the streamed CSV to this path (2.26 GB) for reruns")
    ap.add_argument("--allow_unpinned", action="store_true",
                    help="skip the CSV byte-size pin (fixtures only)")
    ap.add_argument("--class_desc", default=None,
                    help="local oidv7-class-descriptions-boxable.csv (else downloaded)")
    ap.add_argument("--exclude_prompts", action="append", default=[],
                    help="fetch_noncoco_chair prompts.jsonl (validation eval); asserted disjoint")
    ap.add_argument("--exclude_ids", action="append", default=[],
                    help="extra ImageIDs to exclude, one per line")
    ap.add_argument("--keep_depictions", action="store_true",
                    help="keep images with IsDepiction boxes (default: reject)")
    ap.add_argument("--keep_inside", action="store_true",
                    help="keep images with IsInside boxes (default: reject)")
    ap.add_argument("--drop_group_images", action="store_true",
                    help="reject images with IsGroupOf boxes (default: keep + mask them)")
    ap.add_argument("--workers", type=int, default=8, help="parallel image downloads")
    ap.add_argument("--img_retries", type=int, default=4)
    ap.add_argument("--skip_download", action="store_true",
                    help="require images already in <out>/_src (offline / rerun)")
    ap.add_argument("--no_url_check", action="store_true",
                    help="skip the 3-sample train-URL resolve check (offline tests)")
    ap.add_argument("--rm_src", action="store_true",
                    help="delete <out>/_src after a successful build (saves disk)")
    ap.add_argument("--no_gates", action="store_true",
                    help="skip cluster env gates (offline unit tests only)")
    ap.add_argument("--smoke", action="store_true", help="40 pairs")
    args = ap.parse_args(argv)

    t_start = time.time()
    n_pairs = 40 if args.smoke else args.n
    k = int(math.ceil(n_pairs * args.oversample))
    out = os.path.join(args.root, args.subdir)
    src_dir = os.path.join(out, SRC_SUBDIR)
    cache = os.path.join(out, "_cache")
    os.makedirs(cache, exist_ok=True)
    os.makedirs(src_dir, exist_ok=True)
    if not args.no_gates:
        bop.run_all_gates(args.root, min_free_gb=10.0, need_gpu=False)

    # 1. class map (reused from the verified validation pipeline) --------------
    prov = {}
    if args.class_desc:
        class_desc = args.class_desc
    else:
        class_desc = os.path.join(cache, "oidv7-class-descriptions-boxable.csv")
        if not os.path.exists(class_desc):
            print(f"[oitrain] downloading class list {CLASS_DESC_URL}", flush=True)
            fnc._download_to(CLASS_DESC_URL, class_desc)
    with open(class_desc, "rb") as f:
        blob = f.read()
    if not args.allow_unpinned and len(blob) != CLASS_DESC_BYTES:
        raise GateError(f"GATE FAIL: class list is {len(blob)} bytes, expected "
                        f"{CLASS_DESC_BYTES}; upstream file changed -- re-verify.")
    prov["class_desc"] = {"path": class_desc, "bytes": len(blob),
                          "sha256": hashlib.sha256(blob).hexdigest()}
    oi_label_to_coco, coco_to_labels, _ = fnc.build_label_map(class_desc)
    print(f"[oitrain] mapped 80 COCO categories -> {len(oi_label_to_coco)} OI classes",
          flush=True)

    # 2. exclusion set (validation-split eval ids) ----------------------------
    excluded = load_exclude_ids(args.exclude_prompts + args.exclude_ids)
    print(f"[oitrain] exclusion set: {len(excluded)} ImageIDs from "
          f"{args.exclude_prompts + args.exclude_ids}", flush=True)
    if not excluded:
        print("[oitrain] WARNING: empty exclusion set -- the train/validation "
              "disjointness gate is VACUOUS for this build (pass --exclude_prompts "
              "<noncoco/prompts.jsonl> for a real build).", flush=True)

    # 3. stream the TRAIN box CSV once; reservoir-select k eligible images ------
    text, hr, csv_prov = open_box_csv(args.csv_path, cache_csv=args.cache_csv,
                                      allow_unpinned=args.allow_unpinned)
    print(f"[oitrain] streaming train boxes from "
          f"{csv_prov['path'] or csv_prov['url']}", flush=True)
    rng = random.Random(args.seed)
    try:
        reservoir, stats, cooc, hits = select_images(
            csv.reader(text), oi_label_to_coco, k, rng, args.min_area_frac,
            excluded, args.keep_depictions, args.keep_inside, args.drop_group_images)
    finally:
        text.close()
        if hr.tee is not None:
            hr.tee.close()
    csv_prov["bytes_read"] = hr.nbytes
    csv_prov["sha256"] = hr.sha.hexdigest()
    if not args.allow_unpinned and hr.nbytes != TRAIN_BOX_CSV_BYTES:
        raise GateError(f"GATE FAIL: read {hr.nbytes} CSV bytes, expected "
                        f"{TRAIN_BOX_CSV_BYTES} (truncated stream?)")
    if args.cache_csv and hr.tee is not None:
        os.replace(args.cache_csv + ".part", args.cache_csv)
        csv_prov["cached_to"] = args.cache_csv
    print(f"[oitrain] streamed {stats['rows']} rows / {stats['images']} images; "
          f"mapped={stats['mapped_images']} eligible={stats['eligible_total']} "
          f"(rejected: depiction={stats['rejected_depiction']} inside={stats['rejected_inside']} "
          f"group={stats['rejected_group']} small={stats['rejected_small']}; "
          f"degenerate boxes={stats['degenerate_boxes']}) sha256={csv_prov['sha256'][:16]}...",
          flush=True)
    if len(reservoir) < k:
        raise GateError(f"GATE FAIL: only {len(reservoir)} eligible train images, need {k}")
    selected_ids = [r["iid"] for r in reservoir]
    assert_disjoint(selected_ids, excluded, hits)
    print(f"[oitrain] DISJOINT: 0 excluded ids in train CSV, 0 overlap with "
          f"{len(selected_ids)} selected", flush=True)
    with open(os.path.join(out, "cooc_corpus.json"), "w") as f:
        json.dump({p: dict(c) for p, c in sorted(cooc.items())}, f)

    # 4. images (train prefix) + weak-label manifest ---------------------------
    url_check = None
    if not args.no_url_check:
        url_check = url_resolve_check(sorted(selected_ids)[:3])
    if args.skip_download:
        ok = [i for i in selected_ids
              if os.path.exists(os.path.join(src_dir, f"{i}.jpg"))]
        missing = {i: "absent (--skip_download)" for i in selected_ids if i not in ok}
    else:
        ok, missing = download_images(selected_ids, src_dir, args.img_retries, args.workers)
    frac_ok = len(ok) / len(selected_ids)
    print(f"[oitrain] images ok={len(ok)}/{len(selected_ids)} ({frac_ok:.4f}) -> {src_dir}",
          flush=True)
    if frac_ok < 0.99 or len(ok) < n_pairs:
        raise GateError(f"GATE FAIL: {len(ok)}/{len(selected_ids)} images ({frac_ok:.3f}); "
                        f"need >=99% and >={n_pairs}; e.g. {list(missing.items())[:5]}")
    ok_set = set(ok)
    records = [r for r in reservoir if r["iid"] in ok_set]
    manifest = os.path.join(out, "manifest_weak_labels.jsonl")
    n_manifest = write_manifest(records, src_dir, manifest)
    vocab_path = os.path.join(out, "vocab_coco80.txt")
    with open(vocab_path, "w") as f:
        f.write("\n".join(COCO80) + "\n")

    # 5. the anchor builder (imported): schema, masking, hard negatives --------
    run_builder(args.root, args.subdir, manifest, out, vocab_path, n_pairs,
                args.seed, args.min_area_frac, no_gates=True)  # gates ran above
    with open(os.path.join(out, "report.json")) as f:
        builder_report = json.load(f)
    rows_out = [json.loads(l) for l in open(os.path.join(out, "ood_pairs.jsonl"))]
    pairs = rederive_pairs(manifest, out, COCO80, n_pairs, args.min_area_frac, args.seed)
    assert len(rows_out) == len(pairs) == n_pairs, (len(rows_out), len(pairs), n_pairs)
    by_iid = {r["iid"]: r for r in records}
    prov_rows = []
    for i, (row, p) in enumerate(zip(rows_out, pairs)):
        assert tuple(row.keys()) == SCHEMA_KEYS, (list(row.keys()), SCHEMA_KEYS)
        assert row["id"] == f"ood_{i}", row
        assert row["image"] == f"{args.subdir}/images/{row['id']}.jpg", row
        assert row["masked_image"] == f"{args.subdir}/masked/{row['id']}.jpg", row
        assert (row["present"], row["absent"]) == (p["present"], p["absent"]), (
            "builder re-derivation mismatch (non-deterministic selection?)", i, row, p)
        rec = by_iid[p["img_id"]]
        assert row["present"] in rec["labels"] and row["absent"] not in rec["labels"], (row, rec)
        assert p["img_id"] not in excluded
        dom = max((b for b in rec["boxes"] if b["label"] == row["present"]),
                  key=lambda b: (b["nbox"][2] - b["nbox"][0]) * (b["nbox"][3] - b["nbox"][1]))
        prov_rows.append({"id": row["id"], "oi_image_id": p["img_id"],
                          "present": row["present"], "absent": row["absent"],
                          "present_area_frac": round(p["frac"], 4),
                          "present_is_group_of": bool(dom["group"]),
                          "present_box_source": dom["source"],
                          "n_present_boxes": len(p["present_boxes"]),
                          "train_url": TRAIN_IMG_URL_TMPL.format(iid=p["img_id"])})
    bop.write_jsonl(os.path.join(out, "grounding_pairs.jsonl"), rows_out)
    bop.write_jsonl(os.path.join(out, "provenance.jsonl"), prov_rows)

    if args.rm_src:
        for fn in os.listdir(src_dir):
            os.remove(os.path.join(src_dir, fn))
        os.rmdir(src_dir)

    # 6. report (written LAST = success marker) --------------------------------
    n_group = sum(r["present_is_group_of"] for r in prov_rows)
    report = {
        "anchor_source": "Open Images V7 TRAIN split (human boxes, non-COCO)",
        "purpose": ("out-of-domain faithfulness anchor: non-COCO images disjoint from "
                    "POPE/CHAIR (COCO) and from the non-COCO CHAIR eval (Open Images "
                    "VALIDATION split). See design_notes/ood_anchor_openimages.md."),
        "build_date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seconds": round(time.time() - t_start, 1),
        "seed": args.seed, "smoke": args.smoke,
        "n_pairs": len(rows_out), "n_reservoir": k, "oversample": args.oversample,
        "min_area_frac": args.min_area_frac,
        "train_box_csv": csv_prov,
        "train_box_csv_pin": {"url": TRAIN_BOX_CSV_URL, "bytes": TRAIN_BOX_CSV_BYTES,
                              "etag": TRAIN_BOX_CSV_ETAG},
        "class_desc": prov["class_desc"],
        "image_url_template": TRAIN_IMG_URL_TMPL,
        "url_resolve_check": url_check,
        "stream_stats": dict(stats),
        "filters": {"keep_depictions": args.keep_depictions, "keep_inside": args.keep_inside,
                    "drop_group_images": args.drop_group_images},
        "images_downloaded_ok": len(ok), "images_missing": len(missing),
        "manifest_rows": n_manifest,
        "disjointness": {
            "argument": ("train vs validation split of Open Images (ImageIDs partitioned "
                         "by split; S3 routes by split prefix); non-COCO by construction "
                         "vs the COCO evals"),
            "excluded_sources": args.exclude_prompts + args.exclude_ids,
            "n_excluded_ids": len(excluded),
            "excluded_ids_seen_in_train_csv": len(hits),
            "overlap_selected_vs_excluded": 0,
            # True only when there was something to assert against; an empty
            # exclusion set makes the gate vacuous and the report must say so.
            "asserted": bool(excluded),
        },
        "present_is_group_of_n": n_group,
        "present_box_source_hist": dict(Counter(r["present_box_source"] for r in prov_rows)),
        "coco_to_oi_map": {c: [d for _, d in coco_to_labels[c]] for c in COCO80},
        "cooc_scope_note": ("build_ood_pairs samples `absent` by co-occurrence over the "
                            "manifest pool (the downloaded images), not the full corpus; "
                            "cooc_corpus.json holds the corpus-level 80x80 counts for audit."),
        "disk_bytes": {"images": _dir_bytes(os.path.join(out, "images")),
                       "masked": _dir_bytes(os.path.join(out, "masked")),
                       "_src": _dir_bytes(src_dir)},
        "builder_report": builder_report,
        "files": ["grounding_pairs.jsonl", "ood_pairs.jsonl", "provenance.jsonl",
                  "manifest_weak_labels.jsonl", "vocab_coco80.txt", "cooc_corpus.json",
                  "images/", "masked/"],
    }
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[oitrain] wrote {len(rows_out)} pairs -> {out}/grounding_pairs.jsonl "
          f"(group-box present: {n_group}); report.json written; "
          f"{report['seconds']}s", flush=True)
    print("[oitrain] OK", flush=True)


if __name__ == "__main__":
    try:
        main()
    except GateError as e:
        sys.exit(str(e))
