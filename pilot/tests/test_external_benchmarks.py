"""Fixture tests for the external hallucination-benchmark scorers added 2026-09-03:
  * metrics_mme.py           (MME-Hallucination acc/acc+, judge-free)
  * metrics_amber_disc.py    (AMBER discriminative Acc/P/R/F1, judge-free)
  * fetch_objhalbench.py -> metrics_chair.py  (Object HalBench via vendored CHAIR)

Every expected number below is hand-computed in the comments; the scorers must
reproduce them. No network, no GPU, no pytest required:

    python3 pilot/tests/test_external_benchmarks.py

(also collectable by pytest -- each check is a test_* function.)
"""
import base64
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
PY = sys.executable


def _run(script, *args):
    cmd = [PY, os.path.join(PILOT, script), *map(str, args)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise AssertionError(f"{script} failed ({p.returncode}):\n"
                             f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return p


def _approx(a, b, tol=1e-4):
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------------------
# MME-Hallucination
# ---------------------------------------------------------------------------
def test_mme():
    """Fixture (pilot/tests/mme_fixture/gen.jsonl), hand-computed:
    existence (imgA both correct; imgB q1 'No'->wrong, q2 'Maybe'->other/wrong):
      acc=2/4=0.5, acc+=1/2=0.5, score=100, yes_rate=1/4=0.25,
      other_rate=1/4=0.25; P/R/F1 (yes pos, non-other): tp=1,fp=0,fn=1 ->
      precision=1.0, recall=0.5, f1=0.6667.
    count (imgC q1 'yes' ok, q2 'Yes there are two'->yes but gt no -> wrong):
      acc=0.5, acc+=0, score=50.
    position/color: both correct -> acc=1, acc+=1, score=200 each.
    all: score = 100+50+200+200 = 550; full_800_scale True; overall acc =
      (2+1+2+2)/10 = 0.7; other_rate = 1/10 = 0.1."""
    with tempfile.TemporaryDirectory() as d:
        pre = os.path.join(d, "mme")
        _run("metrics_mme.py", "--gen",
             os.path.join(HERE, "mme_fixture", "gen.jsonl"),
             "--out_prefix", pre, "--ckpt", "FX")
        rows = json.load(open(pre + ".json"))
        by = {r["subtask"]: r for r in rows}

        ex = by["existence"]
        assert _approx(ex["acc"], 0.5) and _approx(ex["acc_plus"], 0.5), ex
        assert _approx(ex["score"], 100.0), ex
        assert _approx(ex["yes_rate"], 0.25) and _approx(ex["other_rate"], 0.25), ex
        assert _approx(ex["precision"], 1.0) and _approx(ex["recall"], 0.5), ex
        assert _approx(ex["f1"], 0.6667), ex
        assert ex["other_num"] == 1, ex

        assert _approx(by["count"]["score"], 50.0), by["count"]
        assert _approx(by["position"]["score"], 200.0), by["position"]
        assert _approx(by["color"]["score"], 200.0), by["color"]

        allr = by["all"]
        assert _approx(allr["score"], 550.0), allr
        assert allr["full_800_scale"] is True, allr
        assert _approx(allr["acc"], 0.7), allr
        assert _approx(allr["other_rate"], 0.1), allr

        # per-image bootstrap file: 5 (subtask,image) rows (imgA,imgB,C,E,F),
        # 3 acc+ hits (imgA, position imgE, color imgF)
        pim = [json.loads(l) for l in open(pre + "_per_image.jsonl")]
        assert len(pim) == 5, len(pim)
        assert sum(r["acc_plus_hit"] for r in pim) == 3, pim
    print("MME OK: acc/acc+ per subtask + 800-scale headline match hand calc")


# ---------------------------------------------------------------------------
# AMBER discriminative
# ---------------------------------------------------------------------------
def test_amber_disc():
    """Fixture (amber_disc_fixture/), No is the positive class. Robust primary:
    all (7 items): correct = 2001,2002,2003,2004,2006,2007 = 6 -> acc=6/7=0.8571;
      pred=no:{2002,2003,2006}=3, truth=no:{2002,2003,2005,2006}=4, tp_no=3 ->
      precision_no=1.0, recall_no=0.75, f1_no=0.8571; yes_rate=3/7=0.4286;
      parse_fail_rate=1/7 (2005 'Maybe')=0.1429.
    existence (2001 yes/'Yes', 2002 no/'No, there is no dog.'): both parse right ->
      acc=1.0, precision_no=recall_no=f1_no=1.0, parse_fail=0.
    Official exact (brittle response=='Yes'/'No'):
      existence: 2002 'No, there is no dog.' != 'No' -> recall_no=0 -> f1=0.0,
        acc_official=50.0.  all: acc_official = 3/7.001*100 = 42.9
        (only 2001,2003,2007 exact-correct). Demonstrates format brittleness."""
    with tempfile.TemporaryDirectory() as d:
        pre = os.path.join(d, "ad")
        _run("metrics_amber_disc.py", "--gen",
             os.path.join(HERE, "amber_disc_fixture", "gen.jsonl"),
             "--amber_data", os.path.join(HERE, "amber_disc_fixture"),
             "--out_prefix", pre, "--ckpt", "FX")
        rows = json.load(open(pre + ".json"))
        by = {r["dimension"]: r for r in rows}

        a = by["all"]
        assert _approx(a["accuracy"], 0.8571), a
        assert _approx(a["precision_no"], 1.0), a
        assert _approx(a["recall_no"], 0.75), a
        assert _approx(a["f1_no"], 0.8571), a
        assert _approx(a["yes_rate"], 0.4286), a
        assert _approx(a["parse_fail_rate"], 0.1429), a
        assert a["positive_class"] == "no", a

        e = by["existence"]
        assert _approx(e["accuracy"], 1.0) and _approx(e["f1_no"], 1.0), e

        at = by["attribute"]
        assert _approx(at["accuracy"], 0.6667), at          # 2/3
        assert _approx(at["parse_fail_rate"], 0.3333), at   # 2005

        r = by["relation"]
        assert _approx(r["accuracy"], 1.0), r               # 2006,2007

        # official-exact audit columns show the brittleness gap
        assert _approx(e["accuracy_official_exact"], 50.0), e
        assert _approx(e["f1_no_official_exact"], 0.0), e
        assert _approx(a["accuracy_official_exact"], 42.9), a
        assert a["accuracy_official_exact"] < a["accuracy"] * 100, a

        # sub-dimensions present
        assert "attribute:state" in by and "attribute:action" in by, list(by)
        per = [json.loads(l) for l in open(pre + "_per_item.jsonl")]
        assert len(per) == 7, len(per)
    print("AMBER-disc OK: No-positive Acc/P/R/F1 + official-exact audit match "
          "hand calc; brittleness gap demonstrated")


# ---------------------------------------------------------------------------
# Object HalBench: fetch (spec-strip + coco_gt) -> vendored CHAIR
# ---------------------------------------------------------------------------
def _write_objhal_synth(d):
    """300-line canonical-format spec (8 distinct prompts, 300 distinct ids) +
    synthetic COCO val2014 annotations (each image has instance cat 'cat' and 5
    'A cat sits on a mat.' captions)."""
    prompts8 = [f"Detail prompt number {k}." for k in range(8)]
    ids = list(range(900001, 900301))
    fake_img = base64.b64encode(b"not-a-real-image").decode()
    spec = os.path.join(d, "spec.jsonl")
    with open(spec, "w") as f:
        for i, iid in enumerate(ids):
            f.write(json.dumps({"org_idx": i, "image_id": iid,
                                "question": prompts8[i % 8],
                                "image": fake_img}) + "\n")
    ann_dir = os.path.join(d, "coco_ann")
    os.makedirs(ann_dir)
    instances = {
        "categories": [{"id": 1, "name": "person"}, {"id": 17, "name": "cat"},
                       {"id": 18, "name": "dog"}],
        "images": [{"id": iid, "file_name": f"COCO_val2014_{iid:012d}.jpg"}
                   for iid in ids],
        "annotations": [{"image_id": iid, "category_id": 17, "id": iid}
                        for iid in ids],
    }
    captions = {"annotations": []}
    cid = 1
    for iid in ids:
        for _ in range(5):
            captions["annotations"].append(
                {"image_id": iid, "id": cid, "caption": "A cat sits on a mat."})
            cid += 1
    json.dump(instances, open(os.path.join(ann_dir, "instances_val2014.json"), "w"))
    json.dump(captions, open(os.path.join(ann_dir, "captions_val2014.json"), "w"))
    return spec, ann_dir, ids


def test_objhalbench():
    """fetch_objhalbench builds prompts.jsonl (300) + coco_gt.json (300, each
    {'objects':['cat'],...}); then vendored CHAIR on a 3-row synthetic gen:
      900001 'There is a cat and a dog.' -> mentions cat,dog; gt {cat}; hal dog
      900002 'A cat on a mat.'           -> mentions cat; no hal
      900003 'A person walks.'           -> mentions person; hal person
      chair_s = 2/3 = 0.6667 (900001,900003 hallucinate)
      chair_i = 2/4 = 0.5   (hal mentions dog,person over 4 total mentions)"""
    with tempfile.TemporaryDirectory() as d:
        spec, ann_dir, ids = _write_objhal_synth(d)
        out = os.path.join(d, "objhal_data")
        _run("fetch_objhalbench.py", "--out_dir", out, "--coco_ann_dir", ann_dir,
             "--spec", spec, "--no_download")

        prompts = [json.loads(l) for l in open(os.path.join(out, "prompts.jsonl"))]
        assert len(prompts) == 300, len(prompts)
        r0 = prompts[0]
        assert set(r0) >= {"id", "image", "coco_id", "prompt", "prompt_group"}, r0
        assert r0["image"] == f"COCO_val2014_{r0['coco_id']:012d}.jpg", r0
        assert len({p["prompt"] for p in prompts}) == 8, "8 detail prompts"

        coco_gt = json.load(open(os.path.join(out, "coco_gt.json")))
        assert len(coco_gt) == 300, len(coco_gt)
        assert coco_gt["900001"]["objects"] == ["cat"], coco_gt["900001"]
        assert coco_gt["900001"]["captions"][:1] == ["A cat sits on a mat."]

        manifest = json.load(open(os.path.join(out, "manifest.json")))
        assert manifest["n_images"] == 300 and manifest["n_missing_gt"] == 0, manifest

        # feed 3 synthetic captions through the EXISTING vendored CHAIR scorer
        gen = os.path.join(d, "gen.jsonl")
        with open(gen, "w") as f:
            for iid, text in [(900001, "There is a cat and a dog."),
                              (900002, "A cat on a mat."),
                              (900003, "A person walks.")]:
                f.write(json.dumps({"id": f"objhal_{iid}", "coco_id": iid,
                                    "output": text, "n_new_tokens": 8,
                                    "truncated": False}) + "\n")
        pre = os.path.join(d, "chair")
        _run("metrics_chair.py", "--gen", gen,
             "--coco_gt", os.path.join(out, "coco_gt.json"),
             "--out_prefix", pre, "--ckpt", "FX")
        summ = json.load(open(pre + ".json"))
        assert summ["n_captions"] == 3, summ
        assert _approx(summ["chair_s"], 0.6667), summ
        assert _approx(summ["chair_i"], 0.5), summ
    print("Object HalBench OK: fetch builds 300 prompts + coco_gt in "
          "metrics_chair format; vendored CHAIR reproduces hand-calc CHAIR_s/i")


def main():
    test_mme()
    test_amber_disc()
    test_objhalbench()
    print("\nALL EXTERNAL-BENCHMARK FIXTURES PASS")


if __name__ == "__main__":
    main()
