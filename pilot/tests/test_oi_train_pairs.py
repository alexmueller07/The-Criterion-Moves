"""OFFLINE fixture tests for pilot/fetch_oi_train_pairs.py -- the Open Images
TRAIN-split builder of the out-of-domain faithfulness-anchor pairs.

No network, no GPU, no pytest required. Everything below runs on a tiny
synthetic oidv6 box CSV + class list generated in a tempdir at runtime:

  * CSV STREAMING FILTER -- the real 21-column oidv6-train-annotations-bbox.csv
    header (verified against the live file 2026-09-08), contiguous per-ImageID
    grouping, the sorted-order / header / malformed-row GateErrors, the byte-size
    pin, and the hashing read-through wrapper (nbytes + sha256 == the file's).
  * PER-IMAGE FILTER -- dominant-box area gate, IsDepiction / IsInside rejection,
    IsGroupOf kept-by-default vs --drop_group_images, Confidence!=1 skipped,
    degenerate boxes dropped, unmapped classes invisible.
  * CLASS MAPPING -- all 80 COCO categories resolve through fetch_noncoco_chair.
    build_label_map, the animal "Mouse" (/m/04rmv) is NOT mapped while
    "Computer mouse" (/m/020lf) -> COCO "mouse"; "Musical keyboard" decoy unmapped.
  * DISJOINTNESS -- load_exclude_ids reads the fetch_noncoco_chair prompts.jsonl
    schema; assert_disjoint passes when disjoint, aborts on overlap, and aborts
    when an excluded (validation) id occurs anywhere in the TRAIN CSV.
  * RESERVOIR -- seeded Algorithm R is deterministic; k > eligible returns all.
  * OUTPUT SCHEMA / PATHS (end-to-end via subprocess, --skip_download on
    pre-placed fixture JPEGs): grounding_pairs.jsonl rows carry EXACTLY
    {id, image, masked_image, present, absent}, image == "grounding_oi/images/
    ood_<i>.jpg", masked_image == "grounding_oi/masked/ood_<i>.jpg", files
    exist under --root, present is in the source image, absent is not, the
    present box is filled with the mean color, provenance maps back to train
    ImageIDs, report.json is the success marker and records the disjointness
    assertion as real (non-vacuous).

Pillow is guarded (skips cleanly if missing), like tests/test_ood_pairs.py.

    python3 pilot/tests/test_oi_train_pairs.py
(also collectable by pytest -- each check is a test_* function.)
"""
import csv
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
PY = sys.executable
sys.path.insert(0, PILOT)
sys.path.insert(0, os.path.join(PILOT, "method"))

import fetch_oi_train_pairs as m      # noqa: E402
import fetch_noncoco_chair as fnc     # noqa: E402

try:
    from PIL import Image, ImageDraw
    HAVE_PIL = True
except ImportError:  # PIL-guarded: only the end-to-end build needs Pillow.
    HAVE_PIL = False

# The live oidv6-train-annotations-bbox.csv header (fetched 2026-09-08).
REAL_HEADER = ("ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,"
               "IsTruncated,IsGroupOf,IsDepiction,IsInside,XClick1X,XClick2X,XClick3X,"
               "XClick4X,XClick1Y,XClick2Y,XClick3Y,XClick4Y").split(",")

# /m/ codes for the fixture class list. These particular codes were checked
# against the official oidv7-class-descriptions-boxable.csv on 2026-09-08, but
# the tests do not depend on them being real -- they are labels in a synthetic
# CSV. Everything else gets a synthetic /m/fx### code.
REAL_CODES = {
    "cat": "/m/01yrx", "dog": "/m/0bt9lr", "bird": "/m/015p6", "chair": "/m/01mzpv",
    "person": "/m/01g317", "computer mouse": "/m/020lf", "mouse": "/m/04rmv",
    "computer keyboard": "/m/01m2v", "musical keyboard": "/m/057cc",
    "aircraft": "/m/0k5j", "fixed-wing aircraft": "/m/0cmf2", "tree": "/m/07j7r",
}
DECOYS = [("mouse", "Mouse"), ("musical keyboard", "Musical keyboard"), ("tree", "Tree")]

# fixture image ids: real-looking 16-hex strings, already in sorted order.
IDS = [f"{i:016x}" for i in range(1, 15)]
A, B, C, D, E, F, G, H, I, J, K, L, M, N = IDS
BIG = (0.2, 0.8, 0.2, 0.8)          # area 0.36 >= MIN_AREA_FRAC
SMALL = (0.40, 0.50, 0.40, 0.50)    # area 0.01 <  MIN_AREA_FRAC
# (iid, coco_label_or_decoy_name, (xmin,xmax,ymin,ymax), conf, group, dep, inside)
FIXTURE_ROWS = [
    (A, "cat", BIG, "1", 0, 0, 0),                 # eligible, dom cat
    (B, "cat", SMALL, "1", 0, 0, 0),               # rejected_small
    (C, "dog", BIG, "1", 0, 1, 0),                 # rejected_depiction
    (D, "dog", BIG, "1", 0, 0, 1),                 # rejected_inside
    (E, "Mouse", BIG, "1", 0, 0, 0),               # animal mouse: unmapped -> ignored
    (F, "mouse", BIG, "1", 0, 0, 0),               # Computer mouse -> eligible, dom mouse
    (G, "person", BIG, "1", 1, 0, 0),              # IsGroupOf person (kept by default)
    (G, "chair", (0.1, 0.5, 0.1, 0.35), "1", 0, 0, 0),   # area 0.10
    (H, "Tree", BIG, "1", 0, 0, 0),                # unmapped -> ignored
    (I, "cat", BIG, "0", 0, 0, 0),                 # Confidence 0 -> skipped
    (J, "bird", BIG, "1", 0, 0, 0),                # eligible, dom bird
    (J, "bird", (0.5, 0.5, 0.1, 0.2), "1", 0, 0, 0),     # degenerate (xmin==xmax)
    (K, "dog", BIG, "1", 0, 0, 0),                 # eligible (exclusion-hit test)
    (L, "chair", (0.1, 0.9, 0.1, 0.9), "1", 0, 0, 0),    # area 0.64, dom chair
    (L, "cat", (0.02, 0.10, 0.02, 0.10), "1", 0, 0, 0),  # small co-occurring cat
    (M, "bird", BIG, "1", 0, 0, 0),                # eligible
    (N, "person", BIG, "1", 0, 0, 0),              # eligible
]
ELIGIBLE = {A, F, G, J, K, L, M, N}
LABELS = {A: {"cat"}, B: {"cat"}, C: {"dog"}, D: {"dog"}, E: set(), F: {"mouse"},
          G: {"person", "chair"}, H: set(), I: set(), J: {"bird"}, K: {"dog"},
          L: {"chair", "cat"}, M: {"bird"}, N: {"person"}}
DOM = {A: "cat", F: "mouse", G: "person", J: "bird", K: "dog", L: "chair",
       M: "bird", N: "person"}
VAL_IDS = ["ffffffffffffff01", "ffffffffffffff02"]  # validation-like, NOT in the CSV

LABEL_COLOR = {"cat": (220, 20, 20), "dog": (20, 220, 20), "mouse": (20, 20, 220),
               "person": (220, 220, 20), "chair": (20, 220, 220), "bird": (220, 20, 220)}
SIZE = 100


# ---------------------------------------------------------------------------
# fixture writers
# ---------------------------------------------------------------------------
def _write_class_list(path):
    """Synthetic oidv7-class-descriptions-boxable.csv resolving all 80 COCO
    categories through fnc.COCO_TO_OI / auto-match, plus the three decoys."""
    n = 0
    rows = [("LabelName", "DisplayName")]
    for c in fnc.COCO80:
        names = fnc.COCO_TO_OI.get(c, [c[:1].upper() + c[1:]])
        for nm in names:
            code = REAL_CODES.get(nm.lower())
            if code is None:
                code, n = f"/m/fx{n:03d}", n + 1
            rows.append((code, nm))
    for key, disp in DECOYS:
        rows.append((REAL_CODES[key], disp))
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def _code_for(name_to_label, label):
    """COCO label (or decoy DisplayName) -> /m/ code via the fixture list."""
    if label in ("Mouse", "Tree"):
        return name_to_label[label.lower()][0]
    names = fnc.COCO_TO_OI.get(label, [label])
    return name_to_label[names[0].lower()][0]


def _bbox_rows(name_to_label, rows=FIXTURE_ROWS):
    out = [REAL_HEADER]
    for iid, lab, (x0, x1, y0, y1), conf, grp, dep, ins in rows:
        out.append([iid, "xclick", _code_for(name_to_label, lab), conf,
                    f"{x0:.6f}", f"{x1:.6f}", f"{y0:.6f}", f"{y1:.6f}",
                    "0", "0", str(grp), str(dep), str(ins)] + ["0.000000"] * 8)
    return out


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def _fixture(root):
    cls = os.path.join(root, "classes.csv")
    _write_class_list(cls)
    m2c, c2l, n2l = fnc.build_label_map(cls)
    bbox = os.path.join(root, "train_bbox.csv")
    _write_csv(bbox, _bbox_rows(n2l))
    return cls, bbox, m2c, c2l, n2l


def _stream(bbox_path, **kw):
    text, hr, _ = m.open_box_csv(csv_path=bbox_path, allow_unpinned=True)
    try:
        return m.select_images(csv.reader(text), kw.pop("m2c"), kw.pop("k"),
                               random.Random(kw.pop("seed", 17)), log_every=10 ** 9, **kw)
    finally:
        text.close()


def _write_prompts(path, ids):
    """fetch_noncoco_chair prompts.jsonl schema (the exclusion source)."""
    with open(path, "w") as f:
        for iid in ids:
            f.write(json.dumps({"id": f"oichair_{iid}", "image": f"{iid}.jpg",
                                "coco_id": iid, "prompt": fnc.CHAIR_PROMPT,
                                "prompt_group": 0}) + "\n")


def _make_images(src_dir):
    """One 100x100 JPEG per fixture id: black bg, each valid mapped box drawn
    in its label color (so present/mask can be verified from pixels)."""
    os.makedirs(src_dir, exist_ok=True)
    for iid in IDS:
        img = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
        d = ImageDraw.Draw(img)
        for r_iid, lab, (x0, x1, y0, y1), conf, *_ in FIXTURE_ROWS:
            if r_iid != iid or lab not in LABEL_COLOR or conf != "1" or x0 >= x1:
                continue
            d.rectangle([x0 * SIZE, y0 * SIZE, x1 * SIZE, y1 * SIZE], fill=LABEL_COLOR[lab])
        img.save(os.path.join(src_dir, f"{iid}.jpg"), "JPEG", quality=95)


def _dom_center(iid, label):
    best, ctr = -1.0, None
    for r_iid, lab, (x0, x1, y0, y1), conf, *_ in FIXTURE_ROWS:
        if r_iid == iid and lab == label and x0 < x1 and y0 < y1:
            a = (x1 - x0) * (y1 - y0)
            if a > best:
                best, ctr = a, (int((x0 + x1) / 2 * SIZE), int((y0 + y1) / 2 * SIZE))
    return ctr


def _px(path, xy):
    with Image.open(path) as im:
        return im.convert("RGB").getpixel(xy)


def _close(a, b, tol):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def _expect_gate(fn, *needles):
    try:
        fn()
    except m.GateError as e:
        for nd in needles:
            assert nd in str(e), (nd, str(e))
        return str(e)
    raise AssertionError(f"expected GateError containing {needles}")


# ---------------------------------------------------------------------------
# 1. class mapping incl. the mouse false friend
# ---------------------------------------------------------------------------
def test_class_mapping_mouse_false_friend():
    with tempfile.TemporaryDirectory() as root:
        cls = os.path.join(root, "classes.csv")
        _write_class_list(cls)
        m2c, c2l, _ = fnc.build_label_map(cls)
    assert set(c2l) == set(fnc.COCO80) and all(c2l[c] for c in fnc.COCO80)
    n_expected = sum(len(fnc.COCO_TO_OI.get(c, [c])) for c in fnc.COCO80)
    assert len(m2c) == n_expected == 106, (len(m2c), n_expected)
    assert m2c[REAL_CODES["computer mouse"]] == "mouse"
    assert REAL_CODES["mouse"] not in m2c, "animal Mouse must not map to COCO mouse"
    assert [d for _, d in c2l["mouse"]] == ["Computer mouse"]
    assert m2c[REAL_CODES["computer keyboard"]] == "keyboard"
    assert REAL_CODES["musical keyboard"] not in m2c
    assert REAL_CODES["tree"] not in m2c
    assert m2c[REAL_CODES["aircraft"]] == m2c[REAL_CODES["fixed-wing aircraft"]] == "airplane"
    print(f"MAPPING OK: 80 COCO -> {len(m2c)} OI codes; /m/04rmv (Mouse) unmapped, "
          "/m/020lf (Computer mouse) -> mouse")


# ---------------------------------------------------------------------------
# 2. CSV streaming: header/sort/malformed gates, byte pin, hashing wrapper
# ---------------------------------------------------------------------------
def test_stream_gates_and_pin():
    with tempfile.TemporaryDirectory() as root:
        cls, bbox, m2c, _, n2l = _fixture(root)
        size = os.path.getsize(bbox)
        sha = hashlib.sha256(open(bbox, "rb").read()).hexdigest()

        # header check uses the live 21-column header
        assert tuple(REAL_HEADER[:len(m.BOX_COLS)]) == m.BOX_COLS

        # byte pin: wrong size aborts, right size streams and hashes the file
        _expect_gate(lambda: m.open_box_csv(csv_path=bbox, expected_bytes=size + 1),
                     "GATE FAIL", "bytes, expected")
        text, hr, prov = m.open_box_csv(csv_path=bbox, expected_bytes=size)
        groups = list(m.iter_image_groups(csv.reader(text)))
        text.close()
        assert hr.nbytes == size and hr.sha.hexdigest() == sha, (hr.nbytes, size)
        assert prov["expected_bytes"] == size and prov["path"] == bbox
        assert hr.closed and hr.raw.closed, "underlying file not released"

        # grouping: one group per fixture image, contiguous, all rows kept
        assert [g[0] for g in groups] == IDS
        assert sum(len(g[1]) for g in groups) == len(FIXTURE_ROWS)
        assert len(dict(groups)[G]) == 2 and len(dict(groups)[L]) == 2

        # unsorted -> GateError (swap the blocks of two images)
        rows = _bbox_rows(n2l)
        swapped = [rows[0]] + [r for r in rows[1:] if r[0] == B] + \
                  [r for r in rows[1:] if r[0] != B]
        p = os.path.join(root, "unsorted.csv")
        _write_csv(p, swapped)
        _expect_gate(lambda: list(m.iter_image_groups(csv.reader(open(p, newline="")))),
                     "GATE FAIL", "not sorted")

        # wrong header -> GateError
        bad = [list(rows[0])] + rows[1:]
        bad[0][2] = "Label"
        p = os.path.join(root, "badhdr.csv")
        _write_csv(p, bad)
        _expect_gate(lambda: list(m.iter_image_groups(csv.reader(open(p, newline="")))),
                     "GATE FAIL", "header")

        # truncated row -> GateError
        trunc = rows[:-1] + [rows[-1][:9]]
        p = os.path.join(root, "trunc.csv")
        _write_csv(p, trunc)
        _expect_gate(lambda: list(m.iter_image_groups(csv.reader(open(p, newline="")))),
                     "GATE FAIL", "malformed")
        print(f"STREAM OK: {len(groups)} image groups; pin/sort/header/malformed gates fire")


# ---------------------------------------------------------------------------
# 3. per-image filter + reservoir + co-occurrence
# ---------------------------------------------------------------------------
def test_filter_and_reservoir():
    with tempfile.TemporaryDirectory() as root:
        cls, bbox, m2c, _, _ = _fixture(root)
        res, st, cooc, hits = _stream(bbox, m2c=m2c, k=100)
        assert {r["iid"] for r in res} == ELIGIBLE, sorted(r["iid"] for r in res)
        assert st["images"] == 14 and st["rows"] == len(FIXTURE_ROWS)
        assert st["mapped_images"] == 11, dict(st)   # E, H unmapped; I skipped
        assert st["eligible_total"] == 8
        assert st["rejected_small"] == 1 and st["rejected_depiction"] == 1
        assert st["rejected_inside"] == 1 and st["rejected_group"] == 0
        assert st["group_boxes"] == 1 and st["degenerate_boxes"] == 1
        assert st["skipped_confidence"] == 1
        assert hits == []
        for r in res:
            assert r["labels"] == LABELS[r["iid"]], r
            assert r["dom_label"] == DOM[r["iid"]], r
        byid = {r["iid"]: r for r in res}
        assert abs(byid[A]["dom_frac"] - 0.36) < 1e-9
        assert abs(byid[L]["dom_frac"] - 0.64) < 1e-9
        assert [b["group"] for b in byid[G]["boxes"] if b["label"] == "person"] == [True]
        assert len(byid[J]["boxes"]) == 1, "degenerate box must be dropped"
        assert cooc["chair"]["cat"] == 1 and cooc["cat"]["chair"] == 1
        assert cooc["person"]["chair"] == 1 and cooc["bird"] == {}

        # --drop_group_images rejects G
        res2, st2, _, _ = _stream(bbox, m2c=m2c, k=100, drop_group_images=True)
        assert {r["iid"] for r in res2} == ELIGIBLE - {G} and st2["rejected_group"] == 1
        # --keep_depictions / --keep_inside admit C / D
        res3, _, _, _ = _stream(bbox, m2c=m2c, k=100, keep_depictions=True, keep_inside=True)
        assert {r["iid"] for r in res3} == ELIGIBLE | {C, D}

        # seeded reservoir: deterministic, size k, subset of eligible
        r1, _, _, _ = _stream(bbox, m2c=m2c, k=3, seed=17)
        r2, _, _, _ = _stream(bbox, m2c=m2c, k=3, seed=17)
        assert [r["iid"] for r in r1] == [r["iid"] for r in r2] and len(r1) == 3
        assert {r["iid"] for r in r1} <= ELIGIBLE

        # an excluded (validation) id present in the TRAIN csv is reported
        _, _, _, hits_k = _stream(bbox, m2c=m2c, k=100, excluded=frozenset({K, VAL_IDS[0]}))
        assert hits_k == [K]
        print("FILTER OK: 8/14 eligible, rejections counted, group/depiction/inside "
              "flags honoured, reservoir deterministic")


# ---------------------------------------------------------------------------
# 4. exclusion source + disjointness assertion
# ---------------------------------------------------------------------------
def test_exclusion_and_disjointness():
    with tempfile.TemporaryDirectory() as root:
        pr = os.path.join(root, "prompts.jsonl")
        _write_prompts(pr, VAL_IDS)
        assert m.load_exclude_ids([pr]) == set(VAL_IDS)
        plain = os.path.join(root, "ids.txt")
        open(plain, "w").write("abc123.jpg\n\ndef456\n")
        assert m.load_exclude_ids([plain]) == {"abc123", "def456"}
        assert m.load_exclude_ids([pr, plain]) == set(VAL_IDS) | {"abc123", "def456"}
        _expect_gate(lambda: m.load_exclude_ids([os.path.join(root, "missing.jsonl")]),
                     "GATE FAIL", "does not exist")

    m.assert_disjoint([A, F], set(VAL_IDS), [])                        # disjoint: passes
    _expect_gate(lambda: m.assert_disjoint([A, F], {A}, []), "overlap")
    _expect_gate(lambda: m.assert_disjoint([A, F], {K}, [K]),
                 "occur in the TRAIN box CSV", "split assumption is broken")
    print("DISJOINT OK: prompts.jsonl ids parsed; overlap and csv-hit both abort")


# ---------------------------------------------------------------------------
# 5. manifest conversion (normalized -> pixel COCO xywh)
# ---------------------------------------------------------------------------
def test_manifest_conversion():
    if not HAVE_PIL:
        return print("SKIP test_manifest_conversion: Pillow not installed")
    with tempfile.TemporaryDirectory() as root:
        cls, bbox, m2c, _, _ = _fixture(root)
        src = os.path.join(root, "_src")
        _make_images(src)
        res, _, _, _ = _stream(bbox, m2c=m2c, k=100)
        rec = {r["iid"]: r for r in res}
        mp = os.path.join(root, "manifest.jsonl")
        n = m.write_manifest([rec[A], rec[L]], src, mp)
        rows = [json.loads(l) for l in open(mp)]
        assert n == 2 and len(rows) == 2
        ra = rows[0]
        assert ra["image"] == f"_src/{A}.jpg" and (ra["width"], ra["height"]) == (SIZE, SIZE)
        assert ra["oi_image_id"] == A and ra["boxes"][0]["label"] == "cat"
        assert all(abs(v - e) < 1e-6 for v, e in zip(ra["boxes"][0]["bbox"], [20, 20, 60, 60]))
        assert ra["boxes"][0]["is_group_of"] is False and ra["boxes"][0]["source"] == "xclick"
        rl = rows[1]
        assert {b["label"] for b in rl["boxes"]} == {"chair", "cat"}
        # the builder must accept exactly this manifest (its own loader)
        import build_ood_pairs as bop
        loaded = bop.load_manifest(mp, root)
        assert [r["img_id"] for r in loaded] == [A, L]
        assert loaded[0]["image_path"] == os.path.join(root, "_src", f"{A}.jpg")
        print("MANIFEST OK: normalized boxes -> pixel [x,y,w,h]; build_ood_pairs loads it")


# ---------------------------------------------------------------------------
# 6. end-to-end (subprocess, offline): schema, relative paths, provenance,
#    mask, disjointness recorded; then the csv-hit abort; then the vacuous warning.
# ---------------------------------------------------------------------------
def _run_main(root, *extra):
    cmd = [PY, os.path.join(PILOT, "fetch_oi_train_pairs.py"), "--root", root,
           "--csv_path", os.path.join(root, "train_bbox.csv"), "--allow_unpinned",
           "--class_desc", os.path.join(root, "classes.csv"),
           "--no_gates", "--no_url_check", "--skip_download", "--seed", "17",
           *map(str, extra)]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_end_to_end_offline():
    if not HAVE_PIL:
        return print("SKIP test_end_to_end_offline: Pillow not installed")
    n_pairs = 4
    with tempfile.TemporaryDirectory() as root:
        _fixture(root)
        out = os.path.join(root, m.SUBDIR)
        _make_images(os.path.join(out, m.SRC_SUBDIR))
        pr = os.path.join(root, "noncoco_prompts.jsonl")
        _write_prompts(pr, VAL_IDS)

        p = _run_main(root, "--exclude_prompts", pr, "--n", n_pairs, "--oversample", 1.25)
        assert p.returncode == 0, f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
        assert "[oitrain] OK" in p.stdout and "DISJOINT: 0 excluded ids" in p.stdout
        assert "WARNING: empty exclusion set" not in p.stdout

        rows = [json.loads(l) for l in open(os.path.join(out, "grounding_pairs.jsonl"))]
        assert len(rows) == n_pairs
        prov = [json.loads(l) for l in open(os.path.join(out, "provenance.jsonl"))]
        assert len(prov) == n_pairs
        report = json.load(open(os.path.join(out, "report.json")))
        mean_color = tuple(report["builder_report"]["mean_color_rgb"])

        for i, (r, pv) in enumerate(zip(rows, prov)):
            assert tuple(r.keys()) == m.SCHEMA_KEYS == (
                "id", "image", "masked_image", "present", "absent"), list(r.keys())
            assert r["id"] == f"ood_{i}" == pv["id"]
            assert r["image"] == f"grounding_oi/images/ood_{i}.jpg", r
            assert r["masked_image"] == f"grounding_oi/masked/ood_{i}.jpg", r
            ip, mp_ = os.path.join(root, r["image"]), os.path.join(root, r["masked_image"])
            assert os.path.exists(ip) and os.path.exists(mp_), r
            iid = pv["oi_image_id"]
            assert iid in ELIGIBLE and iid not in VAL_IDS, pv
            assert r["present"] in LABELS[iid] and r["absent"] not in LABELS[iid], (r, pv)
            assert r["present"] in m.COCO80 and r["absent"] in m.COCO80
            assert (pv["present"], pv["absent"]) == (r["present"], r["absent"])
            assert pv["train_url"] == m.TRAIN_IMG_URL_TMPL.format(iid=iid)
            assert pv["present_is_group_of"] == (iid == G and r["present"] == "person")
            # pixels: present object really there, and really removed in the mask
            c = _dom_center(iid, r["present"])
            assert _close(_px(ip, c), LABEL_COLOR[r["present"]], 30), (_px(ip, c), r)
            assert _close(_px(mp_, c), mean_color, 14), (_px(mp_, c), mean_color, r)
            assert _close(_px(mp_, (1, 1)), _px(ip, (1, 1)), 14), "background changed"
        assert len({pv["oi_image_id"] for pv in prov}) == n_pairs, "duplicate source image"

        assert report["n_pairs"] == n_pairs and report["seed"] == 17
        assert report["n_reservoir"] == 5 and report["stream_stats"]["eligible_total"] == 8
        dj = report["disjointness"]
        assert dj["asserted"] is True and dj["n_excluded_ids"] == 2
        assert dj["excluded_ids_seen_in_train_csv"] == 0 and dj["overlap_selected_vs_excluded"] == 0
        assert dj["excluded_sources"] == [pr]
        assert report["builder_report"]["source_mode"] == "weak_labels"
        assert report["builder_report"]["scored"] is False
        assert report["url_resolve_check"] is None          # --no_url_check
        assert report["train_box_csv"]["path"].endswith("train_bbox.csv")
        assert report["train_box_csv"]["expected_bytes"] is None   # --allow_unpinned
        assert report["coco_to_oi_map"]["mouse"] == ["Computer mouse"]
        assert os.path.exists(os.path.join(out, "cooc_corpus.json"))
        assert os.path.exists(os.path.join(out, "manifest_weak_labels.jsonl"))
        assert open(os.path.join(out, "vocab_coco80.txt")).read().split("\n")[:-1] == m.COCO80
        print(f"E2E OK: {n_pairs} pairs; schema + grounding_oi/... relative paths + "
              f"provenance + mask verified; mean_color={mean_color}")

    # csv-hit: a "validation" id that is actually in the train csv must abort the build
    with tempfile.TemporaryDirectory() as root:
        _fixture(root)
        out = os.path.join(root, m.SUBDIR)
        _make_images(os.path.join(out, m.SRC_SUBDIR))
        pr = os.path.join(root, "noncoco_prompts.jsonl")
        _write_prompts(pr, VAL_IDS + [K])
        p = _run_main(root, "--exclude_prompts", pr, "--n", 4, "--oversample", 1.25)
        assert p.returncode != 0
        assert "GATE FAIL" in p.stderr and "occur in the TRAIN box CSV" in p.stderr, p.stderr
        assert not os.path.exists(os.path.join(out, "report.json")), "success marker on abort"
        assert not os.path.exists(os.path.join(out, "grounding_pairs.jsonl"))
        print("E2E ABORT OK: excluded id found in train csv -> GATE FAIL, no report.json")

    # no exclusion source: build succeeds but the gate is flagged vacuous
    with tempfile.TemporaryDirectory() as root:
        _fixture(root)
        out = os.path.join(root, m.SUBDIR)
        _make_images(os.path.join(out, m.SRC_SUBDIR))
        p = _run_main(root, "--n", 4, "--oversample", 1.25)
        assert p.returncode == 0, p.stderr
        assert "WARNING: empty exclusion set" in p.stdout
        report = json.load(open(os.path.join(out, "report.json")))
        assert report["disjointness"]["asserted"] is False
        print("E2E VACUOUS OK: no --exclude_prompts -> warning + asserted=false in report")


TESTS = [test_class_mapping_mouse_false_friend, test_stream_gates_and_pin,
         test_filter_and_reservoir, test_exclusion_and_disjointness,
         test_manifest_conversion, test_end_to_end_offline]


def main():
    for t in TESTS:
        t()
    print(f"\nOI-TRAIN-PAIRS OK: {len(TESTS)} checks passed (mapping; stream gates+pin; "
          "filter+reservoir; disjointness; manifest; end-to-end schema/paths"
          + ("" if HAVE_PIL else " [PIL-dependent checks skipped]") + ")")


if __name__ == "__main__":
    main()
