"""Offline unit tests for pilot/method/build_ood_pairs.py -- the annotation-free,
out-of-domain (OOD) grounding-pair builder.

Proven here WITHOUT GPU / network / a detector (the heavy OWLv2 path is guarded
and never touched): the weak-label manifest path is driven end-to-end on a tiny
synthetic fixture generated at runtime in a tempdir (no committed fixture files).

Checks:
  * SCHEMA CONTRACT -- build_ood_pairs emits EXACTLY the row keys that
    build_grounding_pairs.py emits and faith_loss.py asserts (read from all three
    source files, so the drop-in --faith_pairs contract can't silently drift).
  * FAITHFULNESS / DISJOINTNESS on the fixture:
      - present is the object genuinely in the image (recovered from the saved
        image's object color), absent is a different vocab label genuinely NOT in
        the image (singleton-label images -> any other label is truly absent);
      - the masked image has every `present` box filled with the dataset mean
        color, and background pixels are left unchanged (counterfactual is real);
      - ids unique, count == requested, every image/masked_image path exists.
  * MASK COVERS ALL INSTANCES -- an image with two same-label boxes has both filled.
  * SCORED ABSENT GUARD -- a weakly-detected object (score in [score_lo, score_hi))
    is never chosen as `present` and never chosen as `absent`.

Pillow is guarded: if it is missing the file skips cleanly (exit 0), the way
test_baselines.py guards torch.

    python3 pilot/tests/test_ood_pairs.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
METHOD = os.path.join(PILOT, "method")
PY = sys.executable

try:
    from PIL import Image
    HAVE_PIL = True
except ImportError:  # PIL-guarded: the masking core needs Pillow; skip cleanly.
    HAVE_PIL = False

# distinct, well-separated object colors so the present label is unambiguously
# recoverable from a saved image's object center even after JPEG q=92.
LABEL_COLOR = {
    "cat": (220, 20, 20), "dog": (20, 220, 20), "car": (20, 20, 220),
    "tree": (220, 220, 20), "boat": (20, 220, 220), "lamp": (220, 20, 220),
    "kite": (130, 130, 130), "bench": (240, 150, 30),
}
BG = (0, 0, 0)
SIZE = 100
BOX = (25, 25, 75, 75)  # centered dominant object; center (50,50), corner bg (5,5)


def _skip(name):
    print(f"SKIP {name}: Pillow not installed")


def _run(root, *args):
    cmd = [PY, os.path.join(METHOD, "build_ood_pairs.py"),
           "--root", root, "--no_gates", *map(str, args)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise AssertionError("build_ood_pairs.py failed "
                             f"({p.returncode}):\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return p


def _make_image(path, boxes):
    """boxes: [(label, (x0,y0,x1,y1))]; solid BG with a colored rect per box."""
    img = Image.new("RGB", (SIZE, SIZE), BG)
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    for label, (x0, y0, x1, y1) in boxes:
        d.rectangle([x0, y0, x1, y1], fill=LABEL_COLOR[label])
    img.save(path, "JPEG", quality=95)


def _nearest_label(rgb, labels):
    return min(labels, key=lambda L: sum((a - b) ** 2
                                         for a, b in zip(rgb, LABEL_COLOR[L])))


def _px(path, xy):
    with Image.open(path) as im:
        return im.convert("RGB").getpixel(xy)


def _close(a, b, tol):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# 1. Schema contract: three-way key agreement.
# ---------------------------------------------------------------------------
def test_schema_contract():
    if not HAVE_PIL:
        return _skip("test_schema_contract")
    sys.path.insert(0, METHOD)
    sys.path.insert(0, PILOT)
    import build_ood_pairs as bop
    schema = set(bop.SCHEMA_KEYS)
    assert schema == {"id", "image", "masked_image", "present", "absent"}, schema

    # keys faith_loss.py asserts on every pair row (the authoritative consumer)
    fl = open(os.path.join(METHOD, "faith_loss.py")).read()
    m = re.search(r'for key in \(([^)]*)\):', fl)
    assert m, "could not find faith_loss.py required-key tuple"
    consumer = set(re.findall(r'"(\w+)"', m.group(1)))
    assert consumer == schema, (consumer, schema)

    # keys the COCO builder emits in its rows literal (must be byte-identical)
    bg = open(os.path.join(METHOD, "build_grounding_pairs.py")).read()
    seg = bg[bg.index("rows = [{"):]
    seg = seg[:seg.index("} for")]
    coco_keys = set(re.findall(r'"(\w+)":', seg))
    assert coco_keys == schema, (coco_keys, schema)
    print("SCHEMA OK: build_ood_pairs == build_grounding_pairs == faith_loss "
          f"consumer keys {sorted(schema)}")


# ---------------------------------------------------------------------------
# 2. End-to-end build on a singleton-label fixture: schema + disjointness + mask.
# ---------------------------------------------------------------------------
def test_build_and_properties():
    if not HAVE_PIL:
        return _skip("test_build_and_properties")
    labels = list(LABEL_COLOR)  # 8 labels
    n_img = 16
    with tempfile.TemporaryDirectory() as root:
        src = os.path.join(root, "src_images")
        os.makedirs(src)
        vocab_path = os.path.join(root, "vocab.txt")
        with open(vocab_path, "w") as f:
            f.write("\n".join(labels) + "\n")
        manifest = os.path.join(root, "weak.jsonl")
        with open(manifest, "w") as mf:
            for i in range(n_img):
                lab = labels[i % len(labels)]
                fn = f"img_{i}.jpg"
                _make_image(os.path.join(src, fn), [(lab, BOX)])
                mf.write(json.dumps({
                    "image": os.path.join("src_images", fn),
                    "width": SIZE, "height": SIZE,
                    "boxes": [{"label": lab,
                               "bbox": [BOX[0], BOX[1], BOX[2] - BOX[0], BOX[3] - BOX[1]]}],
                }) + "\n")

        n_pairs = 10
        _run(root, "--manifest", manifest, "--images_root", root,
             "--vocab", vocab_path, "--n_pairs", n_pairs, "--seed", 17)

        out = os.path.join(root, "grounding_ood")
        rows = [json.loads(l) for l in open(os.path.join(out, "ood_pairs.jsonl"))]
        assert len(rows) == n_pairs, len(rows)
        assert len({r["id"] for r in rows}) == n_pairs, "ids not unique"

        report = json.load(open(os.path.join(out, "report.json")))
        assert report["source_mode"] == "weak_labels", report
        assert report["scored"] is False, report
        mean_color = tuple(report["mean_color_rgb"])

        center = ((BOX[0] + BOX[2]) // 2, (BOX[1] + BOX[3]) // 2)  # (50,50)
        corner = (5, 5)
        for r in rows:
            assert set(r.keys()) == set(("id", "image", "masked_image", "present", "absent"))
            ip = os.path.join(root, r["image"])
            mp = os.path.join(root, r["masked_image"])
            assert os.path.exists(ip) and os.path.exists(mp), r

            # present recovered from the saved original's object color == labeled present
            recovered = _nearest_label(_px(ip, center), labels)
            assert recovered == r["present"], (recovered, r["present"])
            # singleton-label image -> absent genuinely absent, and != present
            assert r["absent"] in labels and r["absent"] != r["present"], r

            # counterfactual is real: present box filled with mean color...
            assert _close(_px(mp, center), mean_color, 14), (_px(mp, center), mean_color)
            # ...and the object was actually there before masking (mask changed it)
            assert not _close(_px(ip, center), mean_color, 30), "object==mean? bad fixture"
            # background left untouched by the mask
            assert _close(_px(mp, corner), _px(ip, corner), 14), "background changed"
        print(f"BUILD OK: {n_pairs} OOD pairs, schema+disjointness+mask verified, "
              f"mean_color={mean_color}")


# ---------------------------------------------------------------------------
# 3. Masking must cover ALL instances of the present label.
# ---------------------------------------------------------------------------
def test_mask_covers_all_instances():
    if not HAVE_PIL:
        return _skip("test_mask_covers_all_instances")
    with tempfile.TemporaryDirectory() as root:
        src = os.path.join(root, "src_images")
        os.makedirs(src)
        vocab_path = os.path.join(root, "vocab.txt")
        open(vocab_path, "w").write("cat\ndog\n")
        # two cat boxes (each 30x30, frac 0.09 >= min_area_frac) + black bg
        b1, b2 = (10, 10, 40, 40), (60, 60, 90, 90)
        _make_image(os.path.join(src, "two.jpg"), [("cat", b1), ("cat", b2)])
        manifest = os.path.join(root, "weak.jsonl")
        open(manifest, "w").write(json.dumps({
            "image": os.path.join("src_images", "two.jpg"),
            "width": SIZE, "height": SIZE,
            "boxes": [{"label": "cat", "bbox": [b1[0], b1[1], 30, 30]},
                      {"label": "cat", "bbox": [b2[0], b2[1], 30, 30]}],
        }) + "\n")

        _run(root, "--manifest", manifest, "--images_root", root,
             "--vocab", vocab_path, "--n_pairs", 1, "--seed", 17)
        out = os.path.join(root, "grounding_ood")
        rows = [json.loads(l) for l in open(os.path.join(out, "ood_pairs.jsonl"))]
        assert len(rows) == 1 and rows[0]["present"] == "cat", rows
        assert rows[0]["absent"] == "dog", rows
        mean_color = tuple(json.load(open(os.path.join(out, "report.json")))["mean_color_rgb"])
        mp = os.path.join(root, rows[0]["masked_image"])
        for c in ((25, 25), (75, 75)):  # centers of BOTH cat boxes
            assert _close(_px(mp, c), mean_color, 16), (c, _px(mp, c), mean_color)
        print("MASK OK: both same-label instances filled with mean color")


# ---------------------------------------------------------------------------
# 4. Scored source: a weakly-seen label is never `present` and never `absent`.
# ---------------------------------------------------------------------------
def test_scored_absent_guard():
    if not HAVE_PIL:
        return _skip("test_scored_absent_guard")
    with tempfile.TemporaryDirectory() as root:
        src = os.path.join(root, "src_images")
        os.makedirs(src)
        vocab_path = os.path.join(root, "vocab.txt")
        open(vocab_path, "w").write("cat\ndog\ncar\n")
        # strong cat (dominant) + a weak dog detection; car never appears
        _make_image(os.path.join(src, "a.jpg"),
                    [("cat", BOX), ("dog", (5, 5, 20, 20))])
        manifest = os.path.join(root, "weak.jsonl")
        open(manifest, "w").write(json.dumps({
            "image": os.path.join("src_images", "a.jpg"),
            "width": SIZE, "height": SIZE,
            "boxes": [{"label": "cat", "bbox": [25, 25, 50, 50], "score": 0.90},
                      {"label": "dog", "bbox": [5, 5, 15, 15], "score": 0.20}],
        }) + "\n")

        _run(root, "--manifest", manifest, "--images_root", root,
             "--vocab", vocab_path, "--n_pairs", 1,
             "--score_hi", 0.3, "--score_lo", 0.1, "--seed", 17)
        out = os.path.join(root, "grounding_ood")
        rows = [json.loads(l) for l in open(os.path.join(out, "ood_pairs.jsonl"))]
        report = json.load(open(os.path.join(out, "report.json")))
        assert report["scored"] is True and report["score_hi"] == 0.3, report
        r = rows[0]
        assert r["present"] == "cat", r          # dog (0.20 < score_hi) not present
        assert r["absent"] == "car", r           # dog weakly seen -> excluded from absent
        print("SCORED OK: weak dog neither present nor absent; present=cat absent=car")


TESTS = [test_schema_contract, test_build_and_properties,
         test_mask_covers_all_instances, test_scored_absent_guard]


def main():
    if not HAVE_PIL:
        print("SKIP: Pillow not installed; build_ood_pairs masking needs Pillow. "
              "The end-to-end path is exercised by the cluster smoke gate.")
        return
    for t in TESTS:
        t()
    print(f"\nOOD-PAIRS OK: {len(TESTS)} checks passed "
          "(schema contract; disjointness+mask; all-instance mask; scored absent guard)")


if __name__ == "__main__":
    main()
