"""probe.jsonl -> probe_prompts.jsonl: render each label-free probe row {id,image,object}
as an eval_gen prompt row {id, image, prompt} using the anchor's exact question template,
so `eval_gen.py --dump_logits` can dump the decision statistic g on the PROBE set at every
checkpoint. That dump is what makes "PCR transferred from the probe" possible: the label-free
post-hoc offset delta_k = mean g_probe(base) - mean g_probe(k) is fitted on the probe (never on
POPE) and applied to POPE — the zero-training baseline every train-time criterion method
must beat on a DIFFERENT template (design_notes/review_method_design.md, M2).

usage: probe_prompts.py --probe <probe.jsonl> --out <probe_prompts.jsonl> [--template ...]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from faith_loss import QUESTION  # "Is there a {obj} in the image?" — the anchor's template
except Exception:  # pragma: no cover
    QUESTION = "Is there a {obj} in the image?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--template", default=QUESTION,
                    help="question template with {obj}; default = the anchor's QUESTION")
    a = ap.parse_args()
    n = 0
    with open(a.probe) as f, open(a.out + ".tmp", "w") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            assert {"id", "image", "object"} <= set(r), r
            g.write(json.dumps({"id": r["id"], "image": r["image"],
                                "prompt": a.template.format(obj=r["object"]),
                                "object": r["object"], "gt": None,
                                "category": "probe"}, ensure_ascii=False) + "\n")
            n += 1
    os.replace(a.out + ".tmp", a.out)
    print(f"[probe_prompts] wrote {n} rows -> {a.out} (template: {a.template!r})")


if __name__ == "__main__":
    main()
