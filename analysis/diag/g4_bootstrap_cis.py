"""Committed provenance for the G4 endpoint length-controlled CHAIR CIs quoted
in METHOD_COMPARISON.md and annotated in fig_method_endpoint.

Paired bootstrap (images, B=10000, seed 17) on CHAIR_i over 60-word caption
prefixes: G4-S4 and G4-J4. Run from the repo root. Expected (2026-08-31):
  G4-S4@60: point=-0.0128  CI95=[-0.0228,-0.0028]
  G4-J4@60: point=-0.0138  CI95=[-0.0257,-0.0019]
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "pilot"))
from metrics_chair import extract_object_mentions, extract_objects

B = 10000
SEED = 17
RES = "results"


def per_image(ck, gt):
    out = {}
    for line in open(os.path.join(RES, ck, "chair_gen.jsonl")):
        r = json.loads(line)
        cid = str(r["coco_id"])
        gto = set(gt[cid]["objects"])
        for c in gt[cid]["captions"]:
            gto |= extract_objects(c)
        m = extract_object_mentions(" ".join(r["output"].split()[:60]))
        out[cid] = (len(m), sum(1 for x in m if x not in gto))
    return out


def main():
    gt = json.load(open(os.path.join(RES, "coco_gt.json")))
    arms = {c: per_image(c, gt) for c in ("G4", "S4", "J4")}
    ids = sorted(set(arms["G4"]) & set(arms["S4"]) & set(arms["J4"]))
    rng = random.Random(SEED)

    def chair_i(sample, t):
        m = sum(t[i][0] for i in sample)
        h = sum(t[i][1] for i in sample)
        return h / max(1, m)

    results = {}
    for a, b in (("G4", "S4"), ("G4", "J4")):
        point = chair_i(ids, arms[a]) - chair_i(ids, arms[b])
        gaps = []
        for _ in range(B):
            s = [ids[rng.randrange(len(ids))] for _ in ids]
            gaps.append(chair_i(s, arms[a]) - chair_i(s, arms[b]))
        gaps.sort()
        lo, hi = gaps[int(0.025 * B)], gaps[min(B - 1, int(0.975 * B))]
        results[f"{a}_minus_{b}_at60"] = {
            "point": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "n_pairs": len(ids), "excludes_zero": bool(lo > 0 or hi < 0)}
        print(f"{a}-{b}@60: point={point:+.4f}  CI95=[{lo:+.4f},{hi:+.4f}]  "
              f"excludes0={lo > 0 or hi < 0}")
    json.dump(results, open("analysis/diag/g4_bootstrap_cis.json", "w"), indent=1)
    print("written analysis/diag/g4_bootstrap_cis.json")


if __name__ == "__main__":
    main()
