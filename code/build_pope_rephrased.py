"""Rephrase every POPE prompt onto a DIFFERENT question template, keeping id / image /
gt / category. Design red-team M2: a criterion method that only holds c on the exact POPE
template is indistinguishable from a post-hoc offset; template TRANSFER is the axis where a
train-time method can beat post-hoc correction. Fails loud if >1% of questions don't parse.
usage: build_pope_rephrased.py --pope <pope dir with prompts.jsonl> [--template "..."]
"""
import argparse, json, os, re
PAT = re.compile(r"^Is there (a|an) (.+?) in the image\?\s*$")
SUFFIX = "\nAnswer the question using a single word or phrase."
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pope", required=True)
    ap.add_argument("--template", default="Does the image contain {art} {obj}?")
    a = ap.parse_args()
    src = os.path.join(a.pope, "prompts.jsonl"); dst = os.path.join(a.pope, "prompts_rephrased.jsonl")
    n = bad = 0
    with open(src) as f, open(dst + ".tmp", "w") as g:
        for line in f:
            r = json.loads(line)
            q = r["prompt"].split("\n", 1)[0]
            m = PAT.match(q)
            if not m:
                bad += 1; continue
            r2 = dict(r); r2["prompt"] = a.template.format(art=m.group(1), obj=m.group(2)) + SUFFIX
            r2["template"] = "rephrased"; g.write(json.dumps(r2, ensure_ascii=False) + "\n"); n += 1
    assert bad <= 0.01 * max(1, n + bad), f"{bad} unparsed POPE questions"
    os.replace(dst + ".tmp", dst)
    print(f"[rephrase] {n} rows -> {dst} ({bad} unparsed) template={a.template!r}")
if __name__ == "__main__":
    main()
