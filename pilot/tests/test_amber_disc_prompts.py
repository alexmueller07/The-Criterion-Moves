"""Fixture tests for pilot/build_amber_disc_prompts.py (the AMBER-discriminative
generation-PROMPT builder added 2026-09-05).

Proves, with NO network / NO GPU / NO pytest required:
  * the builder emits rows in EXACTLY the {id, amber_id, image, prompt} schema
    that fullstudy/amber_images.py uses (so eval_gen.py runs it unchanged);
  * the image path and id scheme match the generative builder field-for-field;
  * the default answer suffix is appended, and --answer_suffix "" disables it;
  * the optional annotations cross-check rejects a non-discriminative query file;
  * ROUND TRIP: the produced prompts, once given model outputs, are scored by the
    real pilot/metrics_amber_disc.py to a hand-computed number -- i.e. the two
    halves of the AMBER-disc path actually fit together.

Run:
    python3 pilot/tests/test_amber_disc_prompts.py
(also collectable by pytest -- each check is a test_* function.)
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
PY = sys.executable

# ids 1005-1008 are all discriminative in real AMBER; truths chosen for the
# round-trip hand calc below.
QUERIES = [
    {"id": 1005, "image": "AMBER_1.jpg", "query": "Is there a dog in this image?"},
    {"id": 1006, "image": "AMBER_1.jpg", "query": "Is the sky blue in this image?"},
    {"id": 1007, "image": "AMBER_2.jpg",
     "query": "Are there two cats in this image?"},
    {"id": 1008, "image": "AMBER_2.jpg",
     "query": "Is the man running in this image?"},
]
ANNOTATIONS = [
    {"id": 1005, "type": "discriminative-hallucination", "truth": "yes"},
    {"id": 1006, "type": "discriminative-attribute-state", "truth": "no"},
    {"id": 1007, "type": "discriminative-attribute-number", "truth": "no"},
    {"id": 1008, "type": "discriminative-attribute-action", "truth": "yes"},
]
DEFAULT_SUFFIX = "Please answer yes or no."


def _run(script, *args, expect_fail=False):
    cmd = [PY, os.path.join(PILOT, script), *map(str, args)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if expect_fail:
        if p.returncode == 0:
            raise AssertionError(f"{script} unexpectedly SUCCEEDED\n"
                                 f"STDOUT:\n{p.stdout}")
        return p
    if p.returncode != 0:
        raise AssertionError(f"{script} failed ({p.returncode}):\n"
                             f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return p


def _approx(a, b, tol=1e-4):
    return abs(float(a) - float(b)) <= tol


def _write_amber_dir(d, queries=QUERIES, annotations=ANNOTATIONS):
    os.makedirs(d, exist_ok=True)
    json.dump(queries, open(os.path.join(d, "query_discriminative.json"), "w"))
    if annotations is not None:
        json.dump(annotations, open(os.path.join(d, "annotations.json"), "w"))
    return d


def test_schema_and_prompt():
    """Builder output must match amber_images.py schema exactly + default suffix."""
    with tempfile.TemporaryDirectory() as d:
        _write_amber_dir(d)
        _run("build_amber_disc_prompts.py", "--out", d)
        rows = [json.loads(l)
                for l in open(os.path.join(d, "prompts_disc.jsonl"))]
        assert len(rows) == 4, len(rows)
        for r, q in zip(rows, QUERIES):
            assert set(r.keys()) == {"id", "amber_id", "image", "prompt"}, r
            assert r["amber_id"] == q["id"], r
            assert r["id"] == f"amber_{q['id']}", r
            # image path is EXACTLY what fullstudy/amber_images.py emits
            assert r["image"] == os.path.join("amber", "image", q["image"]), r
            assert r["prompt"] == f"{q['query']} {DEFAULT_SUFFIX}", r
        # manifest records the auditable knobs + image dependency
        man = json.load(open(os.path.join(d, "prompts_disc_manifest.json")))
        assert man["n_prompts"] == 4, man
        assert man["answer_suffix"] == DEFAULT_SUFFIX, man
        assert man["schema"] == ["id", "amber_id", "image", "prompt"], man
        assert man["image_dependency"]["present_on_disk"] == 0, man
        assert _approx(man["image_dependency"]["coverage"], 0.0), man
        assert man["annotations_crosscheck"]["checked"] is True, man
        assert man["annotations_crosscheck"]["truth_counts"]["yes"] == 2, man
        assert man["annotations_crosscheck"]["truth_counts"]["no"] == 2, man
    print("AMBER-disc builder OK: {id,amber_id,image,prompt} schema, image path "
          "+ id scheme match amber_images.py, default suffix appended, manifest "
          "records image dependency")


def test_suffix_empty():
    """--answer_suffix '' yields the raw query as the prompt."""
    with tempfile.TemporaryDirectory() as d:
        _write_amber_dir(d)
        _run("build_amber_disc_prompts.py", "--out", d, "--answer_suffix", "")
        rows = [json.loads(l)
                for l in open(os.path.join(d, "prompts_disc.jsonl"))]
        for r, q in zip(rows, QUERIES):
            assert r["prompt"] == q["query"], r
    print("AMBER-disc builder OK: empty --answer_suffix leaves query verbatim")


def test_rejects_generative_query_file():
    """The annotations cross-check must reject a query whose id is generative."""
    with tempfile.TemporaryDirectory() as d:
        bad_q = QUERIES + [{"id": 1, "image": "AMBER_1.jpg",
                            "query": "Describe this image."}]
        bad_a = ANNOTATIONS + [{"id": 1, "type": "generative",
                                "truth": ["dog"]}]
        _write_amber_dir(d, queries=bad_q, annotations=bad_a)
        p = _run("build_amber_disc_prompts.py", "--out", d, expect_fail=True)
        assert "NOT discriminative" in (p.stdout + p.stderr), p.stdout + p.stderr
        assert not os.path.exists(os.path.join(d, "prompts_disc.jsonl")), \
            "prompts must NOT be written when the gate fails"
    print("AMBER-disc builder OK: non-discriminative query id triggers a loud "
          "gate and writes nothing")


def test_builds_without_annotations():
    """Annotations are optional; the builder still produces valid prompts."""
    with tempfile.TemporaryDirectory() as d:
        _write_amber_dir(d, annotations=None)
        _run("build_amber_disc_prompts.py", "--out", d)
        rows = [json.loads(l)
                for l in open(os.path.join(d, "prompts_disc.jsonl"))]
        assert len(rows) == 4, len(rows)
        man = json.load(open(os.path.join(d, "prompts_disc_manifest.json")))
        assert man["annotations_crosscheck"] is None, man
    print("AMBER-disc builder OK: works without annotations.json (cross-check "
          "skipped)")


def test_roundtrip_through_scorer():
    """The built prompts, given model outputs, are scored by the REAL
    metrics_amber_disc.py. Outputs (Yes,No,Yes,Yes) vs truths (yes,no,no,yes):
      1005 yes/'Yes' ok, 1006 no/'No' ok, 1007 no/'Yes' wrong, 1008 yes/'Yes' ok
      -> accuracy all = 3/4 = 0.75; yes_rate = 3/4 = 0.75; parse_fail = 0;
         positive_class = 'no'."""
    outputs = {1005: "Yes", 1006: "No", 1007: "Yes", 1008: "Yes"}
    with tempfile.TemporaryDirectory() as d:
        _write_amber_dir(d)
        _run("build_amber_disc_prompts.py", "--out", d)
        prompts = [json.loads(l)
                   for l in open(os.path.join(d, "prompts_disc.jsonl"))]
        # synthesize a gen JSONL exactly as eval_gen.py would (image popped,
        # output/n_new_tokens/truncated added), straight from the built prompts.
        gen = os.path.join(d, "gen.jsonl")
        with open(gen, "w") as f:
            for r in prompts:
                f.write(json.dumps({
                    "id": r["id"], "amber_id": r["amber_id"],
                    "prompt": r["prompt"], "output": outputs[r["amber_id"]],
                    "n_new_tokens": 1, "truncated": False}) + "\n")
        pre = os.path.join(d, "score")
        _run("metrics_amber_disc.py", "--gen", gen, "--amber_data", d,
             "--out_prefix", pre, "--ckpt", "FX")
        rows = json.load(open(pre + ".json"))
        by = {r["dimension"]: r for r in rows}
        a = by["all"]
        assert a["n"] == 4, a
        assert _approx(a["accuracy"], 0.75), a
        assert _approx(a["yes_rate"], 0.75), a
        assert _approx(a["parse_fail_rate"], 0.0), a
        assert a["positive_class"] == "no", a
    print("AMBER-disc builder OK: round-trips through metrics_amber_disc.py to "
          "hand-computed accuracy=0.75 (schema is end-to-end compatible)")


def main():
    test_schema_and_prompt()
    test_suffix_empty()
    test_rejects_generative_query_file()
    test_builds_without_annotations()
    test_roundtrip_through_scorer()
    print("\nALL AMBER-DISC PROMPT-BUILDER FIXTURES PASS")


if __name__ == "__main__":
    main()
