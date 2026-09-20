"""Download and place the MME-Hallucination IMAGE set for the four object-
hallucination perception subtasks (existence / count / position / color).

WHY THIS EXISTS
---------------
pilot/fetch_mme.py ships everything needed to SCORE (questions + yes/no ground
truth, from the official 107 KB eval_tool.zip) but deliberately does NOT fetch
images. eval_gen.py needs the actual JPEGs on disk; the endpoint bench job
(fullstudy/sb_fs_bench.sbatch) looks for them at

    data_ucit/mme/MME_Benchmark/<subtask>/<name>.jpg

and skips MME when they are absent (the failure this script fixes). This script
downloads exactly those images and places them under

    <out_dir>/MME_Benchmark/<subtask>/<name>.jpg

so with --out_dir data_ucit/mme they land where the bench job expects.

SOURCE (verified at primary source 2026-09-05 -- NOT from search snippets)
--------------------------------------------------------------------------
MME's OFFICIAL images are access-gated: the official repo
(github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) distributes the
image set only via a request form / e-mail, so there is no official direct
download URL. The images ARE mirrored, un-gated and byte-stable, on the
Hugging Face Hub dataset `darkyarding/MME`:

  repo      : huggingface.co/datasets/darkyarding/MME   (public, gated=False)
  commit    : 7056f44dac19de35e510b62d734bb2f7a6f64739  (branch main, pinned)
  file      : MME_Benchmark_release_version.zip
  size      : 199,848,448 bytes
  sha256    : 1c4b79b494eae354df8fc0728c64fe4b89e46b4c609d36003be15dace6d8e3d7
              (the file's Git-LFS oid == sha256 of content, from the HF tree API;
               also served as the x-linked-etag header)
  layout    : MME_Benchmark_release_version/MME_Benchmark/<subtask>/<name>.jpg

VERIFICATION PERFORMED at pin time (all reproducible):
  * The zip's central directory (2,400 entries, no zip64) was read over HTTP
    range requests; every one of the 30 images per subtask
    (existence/count/position/color) is present as
    MME_Benchmark_release_version/MME_Benchmark/<subtask>/000000XXXXXX.jpg.
  * The set of image filenames referenced by the four Your_Results/<subtask>.txt
    files inside the OFFICIAL eval_tool.zip (sha256 b8125e2a...d3d8acee) matches
    the set of .jpg files in this mirror EXACTLY: 30 referenced == 30 present,
    0 missing, 0 extra, for all four subtasks (120 images total).
  * A sample image decoded to a valid JPEG (SOI ff d8 ff, EOI ff d9, size ==
    the central-directory uncompressed size).
These images are the COCO-sourced MME perception subtasks; filenames are the
bare 12-digit COCO ids (e.g. 000000006040.jpg), matching fetch_mme.py's prompts.

INTEGRITY
---------
Default (range mode): only the ~120 needed members are pulled (~5 MB, not
200 MB). Each is verified against the zip's own CRC-32 (from the central
directory) AND a JPEG magic-byte check -- a per-file cryptographic-strength
content check that does not require hashing the whole archive.
Whole-file modes (--zip / --full_download): the entire archive's sha256 is
checked against the pinned LFS oid above before extraction.

Usage
-----
  # efficient default: range-fetch only the 120 subtask images
  python fetch_mme_images.py --out_dir data_ucit/mme
  # then (or already done) build prompts and run the bench:
  python fetch_mme.py --out_dir data_ucit/mme \
      --image_template "mme/MME_Benchmark/{subtask}/{name}"
  # from a pre-downloaded local zip (no network):
  python fetch_mme_images.py --out_dir data_ucit/mme --zip /path/MME...zip
"""
import argparse
import datetime
import hashlib
import io
import json
import os
import struct
import sys
import urllib.request
import zipfile
import zlib

MME_REPO = "darkyarding/MME"
MME_COMMIT = "7056f44dac19de35e510b62d734bb2f7a6f64739"
MME_ZIP_NAME = "MME_Benchmark_release_version.zip"
MME_ZIP_URL = (f"https://huggingface.co/datasets/{MME_REPO}/resolve/"
               f"{MME_COMMIT}/{MME_ZIP_NAME}")
MME_ZIP_BYTES = 199848448
MME_ZIP_SHA256 = "1c4b79b494eae354df8fc0728c64fe4b89e46b4c609d36003be15dace6d8e3d7"
INNER_MARKER = "MME_Benchmark/"  # every wanted member path contains this segment
DEFAULT_SUBTASKS = ["existence", "count", "position", "color"]
UA = "mme-image-fetch"
N_PER_SUBTASK = 30  # 30 images x 2 questions each, per subtask (official)


# --------------------------------------------------------------------------- #
# remote-zip reader (stdlib only): read central directory + selected members
# over HTTP range requests, so we transfer ~5 MB instead of the full 200 MB.
# --------------------------------------------------------------------------- #
class RemoteZip:
    def __init__(self, url):
        self.url = url
        self._cdn = url  # resolved (possibly signed) URL, refreshed on failure
        self.size = None

    def _fetch(self, start=None, end=None, _retry=True):
        headers = {"User-Agent": UA}
        if start is not None:
            headers["Range"] = f"bytes={start}-{end}"
        req = urllib.request.Request(self._cdn, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                blob = r.read()
                self._cdn = getattr(r, "url", self._cdn)  # cache resolved URL
                cr = r.headers.get("Content-Range")
                return blob, cr
        except Exception:
            if _retry and self._cdn != self.url:
                self._cdn = self.url  # signed URL may have expired; re-resolve
                return self._fetch(start, end, _retry=False)
            raise

    def resolve_size(self):
        blob, cr = self._fetch(0, 0)
        if cr and "/" in cr:
            self.size = int(cr.rsplit("/", 1)[1])
        else:
            self.size = len(blob)  # server ignored range; fall back
        return self.size

    def central_dir(self):
        if self.size is None:
            self.resolve_size()
        tail_n = min(self.size, 1 << 18)  # 256 KB: EOCD + comment
        tail, _ = self._fetch(self.size - tail_n, self.size - 1)
        i = tail.rfind(b"PK\x05\x06")
        if i < 0:
            sys.exit("GATE FAIL: no EOCD in MME zip tail (zip64 or truncated?)")
        (_, _, _, _, n_total, cd_size, cd_off, _clen) = struct.unpack(
            "<IHHHHIIH", tail[i:i + 22])
        if n_total == 0xFFFF or cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
            sys.exit("GATE FAIL: MME zip is zip64; use --full_download/--zip.")
        cd = tail[cd_off - (self.size - tail_n):] if cd_off >= self.size - tail_n \
            else self._fetch(cd_off, cd_off + cd_size - 1)[0]
        return self._parse_cd(cd, n_total)

    @staticmethod
    def _parse_cd(cd, n_total):
        entries = {}
        off = 0
        for _ in range(n_total):
            if cd[off:off + 4] != b"PK\x01\x02":
                break
            method = struct.unpack("<H", cd[off + 10:off + 12])[0]
            crc = struct.unpack("<I", cd[off + 16:off + 20])[0]
            csize = struct.unpack("<I", cd[off + 20:off + 24])[0]
            usize = struct.unpack("<I", cd[off + 24:off + 28])[0]
            fn_len = struct.unpack("<H", cd[off + 28:off + 30])[0]
            ef_len = struct.unpack("<H", cd[off + 30:off + 32])[0]
            cm_len = struct.unpack("<H", cd[off + 32:off + 34])[0]
            lho = struct.unpack("<I", cd[off + 42:off + 46])[0]
            name = cd[off + 46:off + 46 + fn_len].decode("utf-8", "replace")
            entries[name] = {"method": method, "crc": crc, "csize": csize,
                             "usize": usize, "lho": lho}
            off += 46 + fn_len + ef_len + cm_len
        return entries

    def fetch_block(self, start, end):
        """Fetch a contiguous byte range [start, end] (inclusive)."""
        buf, _ = self._fetch(start, min(end, self.size - 1))
        return buf

    @staticmethod
    def member_from_buffer(buf, buf_start, e):
        """Decompress one member from a buffer, indexing by absolute local-header
        offset e['lho'] (works regardless of member ordering within the buffer)."""
        o = e["lho"] - buf_start
        if buf[o:o + 4] != b"PK\x03\x04":
            sys.exit("GATE FAIL: bad local header in MME zip block")
        lfn = struct.unpack("<H", buf[o + 26:o + 28])[0]
        lef = struct.unpack("<H", buf[o + 28:o + 30])[0]
        ds = o + 30 + lfn + lef
        comp = buf[ds:ds + e["csize"]]
        return zlib.decompress(comp, -15) if e["method"] == 8 else comp


# --------------------------------------------------------------------------- #
def _dest_rel(member_name):
    """Map a zip member path to its on-disk relative path under --out_dir:
    .../MME_Benchmark/existence/x.jpg -> MME_Benchmark/existence/x.jpg"""
    idx = member_name.find(INNER_MARKER)
    return member_name[idx:] if idx >= 0 else os.path.basename(member_name)


def _validate_jpeg(raw, where):
    if not (raw[:3] == b"\xff\xd8\xff" and raw[-2:] == b"\xff\xd9"):
        sys.exit(f"GATE FAIL: {where} is not a valid JPEG "
                 f"(magic {raw[:4].hex()}, tail {raw[-2:].hex()})")


def _want(name, subtasks):
    if INNER_MARKER not in name or not name.lower().endswith(".jpg"):
        return False
    tail = name.split(INNER_MARKER, 1)[1]  # <subtask>/<file>.jpg
    st = tail.split("/", 1)[0]
    return st in subtasks


def _place(raw, crc, out_dir, member_name):
    _validate_jpeg(raw, member_name)
    if (zlib.crc32(raw) & 0xFFFFFFFF) != crc:
        sys.exit(f"GATE FAIL: CRC mismatch for {member_name}")
    dest = os.path.join(out_dir, _dest_rel(member_name))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(raw)
    return dest


def from_local_zip(zip_path, out_dir, subtasks, verify_sha):
    with open(zip_path, "rb") as f:
        blob = f.read()
    if verify_sha:
        sha = hashlib.sha256(blob).hexdigest()
        if len(blob) != MME_ZIP_BYTES or sha != MME_ZIP_SHA256:
            sys.exit(f"GATE FAIL: local zip {len(blob)}B sha {sha} != pinned "
                     f"{MME_ZIP_BYTES}B {MME_ZIP_SHA256}")
    placed = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for info in z.infolist():
            if not _want(info.filename, subtasks):
                continue
            raw = z.read(info.filename)
            _place(raw, info.CRC, out_dir, info.filename)
            st = _dest_rel(info.filename).split("/")[1]
            placed.setdefault(st, 0)
            placed[st] += 1
    return placed, f"local:{zip_path}"


def from_full_download(out_dir, subtasks):
    print(f"[mme-img] downloading full zip {MME_ZIP_URL}", flush=True)
    req = urllib.request.Request(MME_ZIP_URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=600) as r:
        blob = r.read()
    sha = hashlib.sha256(blob).hexdigest()
    if len(blob) != MME_ZIP_BYTES or sha != MME_ZIP_SHA256:
        sys.exit(f"GATE FAIL: downloaded zip {len(blob)}B sha {sha} != pinned "
                 f"{MME_ZIP_BYTES}B {MME_ZIP_SHA256}")
    placed = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for info in z.infolist():
            if not _want(info.filename, subtasks):
                continue
            _place(z.read(info.filename), info.CRC, out_dir, info.filename)
            st = _dest_rel(info.filename).split("/")[1]
            placed[st] = placed.get(st, 0) + 1
    return placed, MME_ZIP_URL


def from_range(out_dir, subtasks, force):
    """Range-fetch only the wanted images. Each subtask's members form one tight
    (~1.5 MB) contiguous block in the archive, so we pull ONE range per subtask
    (~4 requests, ~6 MB) instead of two requests per file."""
    rz = RemoteZip(MME_ZIP_URL)
    size = rz.resolve_size()
    if size != MME_ZIP_BYTES:
        sys.exit(f"GATE FAIL: remote MME zip is {size}B, expected pinned "
                 f"{MME_ZIP_BYTES}B -- source changed; re-verify before use.")
    entries = rz.central_dir()
    wanted = {n: e for n, e in entries.items() if _want(n, subtasks)}
    if not wanted:
        sys.exit("GATE FAIL: no matching image members found in MME zip "
                 f"central directory for subtasks {subtasks}")
    by_sub = {}
    for name, e in wanted.items():
        st = _dest_rel(name).split("/")[1]
        by_sub.setdefault(st, []).append((name, e))

    placed = {}
    for st in sorted(by_sub):
        members = by_sub[st]
        if not force and all(
                os.path.exists(os.path.join(out_dir, _dest_rel(n)))
                for n, _ in members):
            placed[st] = len(members)
            print(f"[mme-img] {st}: {len(members)} already present, skip",
                  flush=True)
            continue
        start = min(e["lho"] for _, e in members)
        # +30 local header, +512 (name+extra headroom), +csize of the last-by-offset
        end = max(e["lho"] + 30 + 512 + e["csize"] for _, e in members)
        buf = rz.fetch_block(start, end)
        for name, e in members:
            raw = rz.member_from_buffer(buf, start, e)
            _place(raw, e["crc"], out_dir, name)
        placed[st] = len(members)
        print(f"[mme-img] {st}: {len(members)} images placed "
              f"({(end - start) / 1e6:.1f} MB range)", flush=True)
    return placed, MME_ZIP_URL


def verify_against_prompts(prompts_path, out_dir):
    """Every image referenced by prompts.jsonl must now exist on disk."""
    refs, missing = 0, []
    for line in open(prompts_path):
        line = line.strip()
        if not line:
            continue
        img = json.loads(line).get("image")
        if not img:
            continue
        refs += 1
        # 'image' may be '(mme/)MME_Benchmark/<subtask>/<name>.jpg'
        rel = _dest_rel(img)
        if not os.path.exists(os.path.join(out_dir, rel)):
            missing.append(rel)
    return refs, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True,
                    help="MME data dir; images placed under "
                         "<out_dir>/MME_Benchmark/<subtask>/<name>.jpg "
                         "(use data_ucit/mme so the bench job finds them)")
    ap.add_argument("--subtasks", default=",".join(DEFAULT_SUBTASKS),
                    help="comma-separated subtasks to fetch")
    ap.add_argument("--zip", default=None,
                    help="extract from a pre-downloaded MME zip (no network)")
    ap.add_argument("--full_download", action="store_true",
                    help="download the entire 200 MB zip then extract "
                         "(instead of the default range-fetch of ~120 members)")
    ap.add_argument("--no_download", action="store_true",
                    help="forbid network; requires --zip")
    ap.add_argument("--no_verify_sha", action="store_true",
                    help="with --zip: skip whole-file sha256 gate (NOT advised)")
    ap.add_argument("--prompts", default=None,
                    help="prompts.jsonl to cross-verify (default: "
                         "<out_dir>/prompts.jsonl if present)")
    ap.add_argument("--force", action="store_true",
                    help="re-fetch images even if already present on disk")
    args = ap.parse_args()

    subtasks = [s.strip() for s in args.subtasks.split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    if args.zip:
        placed, src = from_local_zip(args.zip, args.out_dir, subtasks,
                                     verify_sha=not args.no_verify_sha)
    elif args.no_download:
        sys.exit("--no_download set but no --zip given")
    elif args.full_download:
        placed, src = from_full_download(args.out_dir, subtasks)
    else:
        placed, src = from_range(args.out_dir, subtasks, args.force)

    total = sum(placed.get(st, 0) for st in subtasks)
    print(f"[mme-img] placed {total} images across {len(subtasks)} subtasks "
          f"from {src}", flush=True)
    short = [f"{st}:{placed.get(st, 0)}" for st in subtasks]
    print("[mme-img]   " + "  ".join(short), flush=True)

    # loud gate: expect exactly 30 per subtask (official MME perception count)
    bad = {st: placed.get(st, 0) for st in subtasks
           if placed.get(st, 0) != N_PER_SUBTASK}
    if bad:
        sys.exit(f"GATE FAIL: expected {N_PER_SUBTASK} images per subtask, got "
                 f"{bad}")

    # end-to-end check: referenced prompt images resolve on disk
    prompts_path = args.prompts or os.path.join(args.out_dir, "prompts.jsonl")
    verified = None
    if os.path.exists(prompts_path):
        refs, missing = verify_against_prompts(prompts_path, args.out_dir)
        if missing:
            sys.exit(f"GATE FAIL: {len(missing)} prompt-referenced images "
                     f"missing on disk, e.g. {missing[:5]}")
        verified = refs
        print(f"[mme-img] verified all {refs} prompt-referenced images resolve "
              f"({os.path.basename(prompts_path)})", flush=True)
    else:
        print(f"[mme-img] note: {prompts_path} not found; skipped prompt "
              "cross-check (run fetch_mme.py to build prompts).", flush=True)

    manifest = {
        "benchmark": "MME-Hallucination images (existence/count/position/color)",
        "fetch_date_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "source": {
            "hf_dataset": f"huggingface.co/datasets/{MME_REPO}",
            "commit": MME_COMMIT,
            "file": MME_ZIP_NAME,
            "url": MME_ZIP_URL,
            "zip_bytes": MME_ZIP_BYTES,
            "zip_sha256_lfs_oid": MME_ZIP_SHA256,
            "note": "un-gated HF mirror of MME's request-gated official images; "
                    "filenames byte-match the official eval_tool.zip references.",
        },
        "layout": "<out_dir>/MME_Benchmark/<subtask>/<name>.jpg",
        "mode": ("local_zip" if args.zip else
                 "full_download" if args.full_download else "http_range"),
        "subtasks": {st: placed.get(st, 0) for st in subtasks},
        "n_images": total,
        "per_file_integrity": "zip CRC-32 + JPEG magic (range/zip modes)",
        "prompts_referenced_verified": verified,
    }
    with open(os.path.join(args.out_dir, "images_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[mme-img] wrote {os.path.join(args.out_dir, 'images_manifest.json')}",
          flush=True)
    print("[mme-img] OK", flush=True)


if __name__ == "__main__":
    main()
