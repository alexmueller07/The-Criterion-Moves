"""Fixture tests for fullstudy/neutralize_prior.py (the interventional answer-
prior manipulation of the pilot suite, 2026-09-08).

Proves, with NO network / NO GPU / NO pytest required (Python 3.9 compatible):
  * neutral: yes and no counts are equal, refusal-class rows <= 2% of n,
    n_train preserved, refills come only from answerable non-yes/no rows;
  * amplified: refusal-bearing task reaches 60% refusal, yes/no-bearing task
    reaches yes:no = 4:1 by subsampling 'no', n_train preserved;
  * a task with no yes/no/refusal content is written BYTE-IDENTICAL;
  * every written line is a byte-identical source line (selection only), and
    the unique kept rows keep source order;
  * DETERMINISM across processes with different PYTHONHASHSEED (all train
    files and the manifest byte-identical);
  * the manifest's n_train / answer_stats equal an INDEPENDENT recount of the
    written files with the pilot definition (exact yes/no/unanswerable), and
    the recipe counts are internally consistent;
  * --refill shrink never duplicates and records the smaller n_train;
  * asset links: images/val resolve to the source, ER dirs and manifests are
    NOT linked (a replay arm must rebuild buffers from the variant files);
  * refuses to overwrite a frozen variant without --force;
  * pilot-shaped class counts (3800/8000-row VizWiz/TextVQA) give the recipe
    numbers quoted in design_notes/mechanism_intervention.md.

Run:
    python3 pilot/tests/test_neutralize_prior.py
(also collectable by pytest -- each check is a test_* function.)
"""
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
FS = os.path.join(REPO, "fullstudy")
SCRIPT = os.path.join(FS, "neutralize_prior.py")
PY = sys.executable

sys.path.insert(0, FS)
from neutralize_prior import (  # noqa: E402
    answer_stats, classify, select_rows, CLASSES)

# ---------------------------------------------------------------------------
# synthetic fixture: three tasks with known class counts
# ---------------------------------------------------------------------------
# qa_yn : n=200  yes 40 | no 16 | refusal 8 | other 136   (yes:no 2.5:1)
# qa_ref: n=100  yes 3  | no 4  | refusal 40 (38 'unanswerable' + 2
#                'unsuitable image') | other 53             (refusal 40%)
# mc    : n=50   all option letters -> no yes/no/refusal content
FIX = {
    "qa_yn": [("yes", 40), ("no", 16), ("unanswerable", 8), ("other", 136)],
    "qa_ref": [("yes", 3), ("no", 4), ("unanswerable", 38),
               ("unsuitable image", 2), ("other", 53)],
    "mc": [("letter", 50)],
}
OTHER_WORDS = ["cat", "north", "nothing", "red", "two", "yes please not",
               "a dog", "unknown brand name", "3", "blue car"]


def _rows(task, spec):
    rows, k = [], 0
    for target, count in spec:
        for j in range(count):
            if target == "other":
                t = OTHER_WORDS[j % len(OTHER_WORDS)]
            elif target == "letter":
                t = "ABCDE"[j % 5]
            else:
                t = target
            rows.append({"id": f"{task}_{k}", "image": f"tasks/{task}/images/{k}.jpg",
                         "prompt": f"question {k} for {task}?", "target": t,
                         "meta": {"k": k}})
            k += 1
    # interleave classes so "source order" is a real thing to preserve
    random.Random(f"fixture:{task}").shuffle(rows)
    return rows


def _write_fixture(root):
    man = {"benchmark": "fixture", "seed": 17, "tasks": {}, "orders": {"o1": []}}
    for task, spec in FIX.items():
        d = os.path.join(root, "tasks", task)
        os.makedirs(os.path.join(d, "images"), exist_ok=True)
        rows = _rows(task, spec)
        with open(os.path.join(d, "train.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(os.path.join(d, "val.jsonl"), "w") as f:
            f.write(json.dumps({"id": f"{task}_va0", "image": "x.jpg",
                                "prompt": "p", "target": "t"}) + "\n")
        open(os.path.join(d, "images", "0.jpg"), "wb").write(b"\xff\xd8placeholder")
        man["tasks"][task] = {"n_train": len(rows), "max_new_tokens": 16,
                              "answer_stats": answer_stats([r["target"] for r in rows])}
        man["orders"]["o1"].append(task)
    os.makedirs(os.path.join(root, "grounding"), exist_ok=True)
    open(os.path.join(root, "grounding", "grounding_pairs.jsonl"), "w").write("{}\n")
    os.makedirs(os.path.join(root, "pope"), exist_ok=True)
    os.makedirs(os.path.join(root, "er100"), exist_ok=True)
    open(os.path.join(root, "er100", "qa_yn_100.jsonl"), "w").write("{}\n")
    mpath = os.path.join(root, "pilot_manifest.json")
    json.dump(man, open(mpath, "w"), indent=1)
    return mpath


def _run(root, mpath, variant, out_root, extra=(), env_extra=None, check=True):
    env = dict(os.environ)
    env.update(env_extra or {})
    cmd = [PY, SCRIPT, "--data_root", root, "--manifest", mpath,
           "--variant", variant, "--out_root", out_root] + list(extra)
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if check:
        assert p.returncode == 0, f"{' '.join(cmd)}\nSTDOUT:{p.stdout}\nSTDERR:{p.stderr}"
    return p


def _lines(path):
    return [l.rstrip("\n") for l in open(path, encoding="utf-8") if l.strip()]


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _class_counts(lines):
    c = {k: 0 for k in CLASSES}
    for l in lines:
        c[classify(json.loads(l)["target"])] += 1
    return c


def _independent_stats(lines):
    """Pilot definition, written independently of the script's helper."""
    ts = [json.loads(l)["target"].strip().lower() for l in lines]
    n = len(ts)
    return {"yes_frac": round(sum(t == "yes" for t in ts) / n, 4),
            "no_frac": round(sum(t == "no" for t in ts) / n, 4),
            "refusal_frac": round(sum(t == "unanswerable" for t in ts) / n, 4),
            "mean_target_words": round(sum(len(t.split()) for t in ts) / n, 2)}


def _check_selection_only(src_lines, out_lines, n_unique):
    """Every output line is a source line; the first n_unique are distinct and
    in source order; the rest are duplicates of source lines."""
    pos = {l: i for i, l in enumerate(src_lines)}
    assert len(pos) == len(src_lines), "fixture rows must be unique"
    for l in out_lines:
        assert l in pos, f"written line is not a source line: {l[:80]}"
    head = out_lines[:n_unique]
    assert len(set(head)) == n_unique, "unique block contains a duplicate"
    idx = [pos[l] for l in head]
    assert idx == sorted(idx), "unique kept rows are not in source order"


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_classify():
    assert classify("yes") == "yes" and classify("Yes.") == "yes"
    assert classify("No") == "no" and classify("no.") == "no"
    assert classify("north") == "other", "startswith('no') trap"
    assert classify("nothing") == "other"
    assert classify("yes please not") == "other"
    assert classify("Unanswerable") == "refusal"
    assert classify("unanswerable.") == "refusal"
    assert classify("unsuitable image") == "refusal"
    assert classify("the unknown") == "refusal"
    assert classify("unknown brand name") == "other"
    assert classify("A") == "other" and classify("a dog") == "other"
    print("classify OK")


def test_neutral_priors():
    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, "data")
        mpath = _write_fixture(root)
        out = os.path.join(root, "tasks_neutral")
        _run(root, mpath, "neutral", out)
        man = json.load(open(os.path.join(out, "neutral_manifest.json")))
        assert man["variant"] == "neutral"
        for task in FIX:
            src = _lines(os.path.join(root, "tasks", task, "train.jsonl"))
            got = _lines(os.path.join(out, "tasks", task, "train.jsonl"))
            rec = man["recipe"][task]
            assert len(got) == len(src) == rec["n_written"], (task, len(got), len(src))
            assert man["tasks"][task]["n_train"] == len(got)
            c = _class_counts(got)
            assert c["yes"] == c["no"], (task, c)
            assert c["refusal"] <= 0.02 * len(got), (task, c)
            _check_selection_only(src, got, rec["n_unique_written"])
            # refills only from answerable non-yes/no rows
            assert rec["duplicated"]["yes"] == rec["duplicated"]["no"] == 0
            assert rec["duplicated"]["refusal"] == 0
        # exact expected recipe for qa_yn: yes 40->16, no 16, refusal 8->4,
        # other 136 + 28 duplicates
        r = man["recipe"]["qa_yn"]
        assert r["counts_written"] == {"yes": 16, "no": 16, "refusal": 4, "other": 164}, r
        assert r["duplicated"]["other"] == 28 and r["n_unique_written"] == 172, r
        assert r["removed"] == {"yes": 24, "no": 0, "refusal": 4, "other": 0}, r
        r = man["recipe"]["qa_ref"]
        assert r["counts_written"] == {"yes": 3, "no": 3, "refusal": 2, "other": 92}, r
        # untouched task is byte-identical
        assert man["recipe"]["mc"]["unchanged"] is True
        assert _sha(os.path.join(out, "tasks", "mc", "train.jsonl")) == \
            _sha(os.path.join(root, "tasks", "mc", "train.jsonl"))
        assert man["recipe"]["mc"]["rule"] == "unchanged"
    print("neutral priors OK (yes==no, refusal<=2%, n preserved, selection only)")


def test_amplified_priors():
    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, "data")
        mpath = _write_fixture(root)
        out = os.path.join(root, "tasks_amplified")
        _run(root, mpath, "amplified", out)
        man = json.load(open(os.path.join(out, "amplified_manifest.json")))
        assert man["variant"] == "amplified"
        # refusal-bearing task -> 60% refusal (40 unique + 20 duplicates)
        src = _lines(os.path.join(root, "tasks", "qa_ref", "train.jsonl"))
        got = _lines(os.path.join(out, "tasks", "qa_ref", "train.jsonl"))
        r = man["recipe"]["qa_ref"]
        assert r["rule"] == "amplified:refusal", r
        assert len(got) == 100 and man["tasks"]["qa_ref"]["n_train"] == 100
        c = _class_counts(got)
        assert c["refusal"] == 60, c
        assert r["duplicated"]["refusal"] == 20 and r["n_unique_written"] == 80, r
        _check_selection_only(src, got, r["n_unique_written"])
        # yes/no-bearing task -> yes:no = 4:1 by subsampling 'no'; yes untouched
        src = _lines(os.path.join(root, "tasks", "qa_yn", "train.jsonl"))
        got = _lines(os.path.join(out, "tasks", "qa_yn", "train.jsonl"))
        r = man["recipe"]["qa_yn"]
        assert r["rule"] == "amplified:yes_ratio", r
        assert len(got) == 200
        c = _class_counts(got)
        assert c["yes"] == 40 and c["no"] == 10, c
        assert c["refusal"] == 8, c          # refusal rows untouched here
        assert r["duplicated"] == {"yes": 0, "no": 0, "refusal": 0, "other": 6}, r
        _check_selection_only(src, got, r["n_unique_written"])
        assert man["recipe"]["mc"]["unchanged"] is True
    print("amplified priors OK (refusal 60%, yes:no 4:1, n preserved)")


def test_determinism_across_processes():
    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, "data")
        mpath = _write_fixture(root)
        outs = []
        for i, hs in enumerate(("0", "4242")):
            out = os.path.join(d, f"v{i}")
            _run(root, mpath, "neutral", out, env_extra={"PYTHONHASHSEED": hs})
            outs.append(out)
            out = os.path.join(d, f"a{i}")
            _run(root, mpath, "amplified", out, env_extra={"PYTHONHASHSEED": hs})
            outs.append(out)
        for a, b in ((outs[0], outs[2]), (outs[1], outs[3])):
            for task in FIX:
                assert _sha(os.path.join(a, "tasks", task, "train.jsonl")) == \
                    _sha(os.path.join(b, "tasks", task, "train.jsonl")), (a, b, task)
            ma = [f for f in os.listdir(a) if f.endswith("_manifest.json")][0]
            assert _sha(os.path.join(a, ma)) == _sha(os.path.join(b, ma)), (a, b)
    print("determinism OK (byte-identical across PYTHONHASHSEED values)")


def test_manifest_stats_and_recipe_consistency():
    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, "data")
        mpath = _write_fixture(root)
        src_man = json.load(open(mpath))
        for variant in ("neutral", "amplified"):
            out = os.path.join(root, f"tasks_{variant}")
            _run(root, mpath, variant, out)
            man = json.load(open(os.path.join(out, f"{variant}_manifest.json")))
            assert man["orders"] == src_man["orders"]
            assert man["seed"] == 17 and man["refill_policy"] == "dup"
            assert set(man["recipe"]) == set(FIX)
            for task in FIX:
                got = _lines(os.path.join(out, "tasks", task, "train.jsonl"))
                t = man["tasks"][task]
                assert t["n_train"] == len(got)
                assert t["n_train_source"] == src_man["tasks"][task]["n_train"]
                assert t["max_new_tokens"] == src_man["tasks"][task]["max_new_tokens"]
                assert t["answer_stats"] == _independent_stats(got), (variant, task)
                r = man["recipe"][task]
                assert r["answer_stats_source"] == src_man["tasks"][task]["answer_stats"]
                assert sum(r["counts_written"].values()) == r["n_written"] == len(got)
                assert r["n_unique_written"] + sum(r["duplicated"].values()) == r["n_written"]
                for c in CLASSES:
                    assert r["counts_kept_unique"][c] + r["removed"][c] == r["counts_source"][c]
                    assert r["counts_written"][c] == r["counts_kept_unique"][c] + r["duplicated"][c]
                assert r["counts_written"] == _class_counts(got), (variant, task)
                assert r["sha256_written"] == _sha(os.path.join(out, "tasks", task, "train.jsonl"))
                assert r["n_shortfall"] == 0
    print("manifest stats + recipe consistency OK")


def test_shrink_refill_never_duplicates():
    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, "data")
        mpath = _write_fixture(root)
        out = os.path.join(root, "tasks_neutral")
        _run(root, mpath, "neutral", out, extra=["--refill", "shrink"])
        man = json.load(open(os.path.join(out, "neutral_manifest.json")))
        r = man["recipe"]["qa_yn"]
        assert sum(r["duplicated"].values()) == 0
        assert r["n_written"] == 172 == man["tasks"]["qa_yn"]["n_train"], r
        assert r["n_shortfall"] == 28, r
        got = _lines(os.path.join(out, "tasks", "qa_yn", "train.jsonl"))
        assert len(got) == len(set(got)) == 172
        out2 = os.path.join(root, "tasks_amplified")
        _run(root, mpath, "amplified", out2, extra=["--refill", "shrink"])
        man = json.load(open(os.path.join(out2, "amplified_manifest.json")))
        r = man["recipe"]["qa_ref"]
        assert sum(r["duplicated"].values()) == 0
        assert r["n_written"] == 66 == man["tasks"]["qa_ref"]["n_train"], r
        got = _lines(os.path.join(out2, "tasks", "qa_ref", "train.jsonl"))
        c = _class_counts(got)
        assert c["refusal"] == 40 and len(got) == 66, c
        assert r["targets"]["n_shrunk_to"] == 66 and r["n_shortfall"] == 34, r
    print("shrink refill OK (no duplicates, smaller n_train recorded)")


def test_asset_links_make_a_dropin_root():
    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, "data")
        mpath = _write_fixture(root)
        out = os.path.join(root, "tasks_neutral")
        _run(root, mpath, "neutral", out)
        for task in FIX:
            for name in ("images", "val.jsonl"):
                dst = os.path.join(out, "tasks", task, name)
                assert os.path.islink(dst), dst
                assert os.path.realpath(dst) == os.path.realpath(
                    os.path.join(root, "tasks", task, name)), dst
            # the row's relative image path resolves through the link
            r = json.loads(_lines(os.path.join(out, "tasks", task, "train.jsonl"))[0])
            assert os.path.isdir(os.path.dirname(os.path.join(out, r["image"])))
        assert os.path.realpath(os.path.join(out, "grounding")) == \
            os.path.realpath(os.path.join(root, "grounding"))
        assert os.path.isdir(os.path.join(out, "pope"))
        assert not os.path.lexists(os.path.join(out, "er100")), "ER buffers must not be linked"
        assert not os.path.lexists(os.path.join(out, "pilot_manifest.json"))
        assert not os.path.lexists(os.path.join(out, "tasks_neutral")), "no self-link"
        man = json.load(open(os.path.join(out, "neutral_manifest.json")))
        assert set(man["linked_top_level"]) == {"grounding", "pope"}, man["linked_top_level"]
        # run_arm.py's hard-coded path shape resolves
        assert os.path.isfile(os.path.join(out, "tasks", "qa_yn", "train.jsonl"))
    print("asset links OK (drop-in data root; ER dirs and manifests excluded)")


def test_refuses_overwrite_without_force():
    with tempfile.TemporaryDirectory() as d:
        root = os.path.join(d, "data")
        mpath = _write_fixture(root)
        out = os.path.join(root, "tasks_neutral")
        _run(root, mpath, "neutral", out)
        p = _run(root, mpath, "neutral", out, check=False)
        assert p.returncode != 0 and "--force" in (p.stdout + p.stderr)
        _run(root, mpath, "neutral", out, extra=["--force"])
        p = _run(root, mpath, "neutral", os.path.join(d, "dry"), extra=["--dry_run"])
        assert not os.path.exists(os.path.join(d, "dry"))
    print("overwrite guard + dry run OK")


def _pilot_classes(n, yes, no, refusal):
    cl = ["yes"] * yes + ["no"] * no + ["refusal"] * refusal
    cl += ["other"] * (n - len(cl))
    random.Random("pilot-shape").shuffle(cl)
    return cl


def test_pilot_shaped_recipes():
    """Class counts implied by fullstudy/pilot_manifest.json (fractions x n,
    rounded): these are the numbers quoted in mechanism_intervention.md."""
    vw = _pilot_classes(3800, 91, 106, 1642)     # VizWiz  .024/.028/.432
    tv = _pilot_classes(8000, 408, 120, 176)     # TextVQA .051/.015/.022
    sel, r = select_rows(vw, "neutral", random.Random("t"))
    assert len(sel) == 3800
    assert r["counts_written"] == {"yes": 91, "no": 91, "refusal": 76, "other": 3542}, r
    assert r["duplicated"]["other"] == 1581 and r["n_unique_written"] == 2219, r
    sel, r = select_rows(vw, "amplified", random.Random("t"))
    assert len(sel) == 3800
    assert r["counts_written"]["refusal"] == 2280 and r["duplicated"]["refusal"] == 638, r
    assert sum(r["counts_written"][c] for c in ("yes", "no", "other")) == 1520, r
    sel, r = select_rows(tv, "neutral", random.Random("t"))
    assert len(sel) == 8000
    assert r["counts_written"] == {"yes": 120, "no": 120, "refusal": 160, "other": 7600}, r
    assert r["duplicated"]["other"] == 304, r
    sel, r = select_rows(tv, "amplified", random.Random("t"))
    assert len(sel) == 8000
    assert r["counts_written"] == {"yes": 408, "no": 102, "refusal": 176, "other": 7314}, r
    assert r["duplicated"]["other"] == 18, r
    # shrink variants (no duplicates)
    sel, r = select_rows(vw, "neutral", random.Random("t"), refill="shrink")
    assert len(sel) == 2219 and r["n_shortfall"] == 1581
    sel, r = select_rows(vw, "amplified", random.Random("t"), refill="shrink")
    assert len(sel) == 2736 and r["counts_written"]["refusal"] == 1642, r
    # answer_stats helper reproduces the pilot manifest numbers
    st = answer_stats(["yes"] * 91 + ["no"] * 106 + ["unanswerable"] * 1642
                      + ["cat"] * 1961)
    assert st["yes_frac"] == 0.0239 and st["refusal_frac"] == 0.4321, st
    print("pilot-shaped recipes OK (numbers match mechanism_intervention.md)")


def main():
    test_classify()
    test_neutral_priors()
    test_amplified_priors()
    test_determinism_across_processes()
    test_manifest_stats_and_recipe_consistency()
    test_shrink_refill_never_duplicates()
    test_asset_links_make_a_dropin_root()
    test_refuses_overwrite_without_force()
    test_pilot_shaped_recipes()
    print("\nALL NEUTRALIZE-PRIOR FIXTURES PASS")


if __name__ == "__main__":
    main()
