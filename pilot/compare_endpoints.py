"""Pre-registered gate evaluation. Committed BEFORE any results exist.

Inputs: results/<CKPT>/ dirs produced by sb_eval.sbatch for S0,S1..S4,J1..J4.
Outputs: trajectory table (all metrics x checkpoints), gate verdicts, and
paired-bootstrap CIs for the SEQ-vs-JOINT endpoint gaps.

Gate (from PREREGISTRATION.md):
 1. Sustained worsening in SEQ: CHAIR_i(S4)>CHAIR_i(S1) and >=2 of {S2,S3,S4}
    worse than S1 on CHAIR_i; OR the mirrored pattern on POPE-adversarial F1.
 2. Endpoint gap vs matched joint: CHAIR_i(S4)-CHAIR_i(J4) > 0 with 95% paired
    bootstrap CI excluding 0; OR F1_adv(J4)-F1_adv(S4) > 0 with CI excluding 0.
Both must hold. Confound panel is always printed alongside.
"""
import argparse
import json
import os
import random

from metrics_pope import parse_yn

B = 10000
SEED = 17


def load_json(path):
    with open(path) as f:
        return json.load(f)


def chair_per_image(results, ckpt):
    rows = []
    p = os.path.join(results, ckpt, "chair_per_image.jsonl")
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            m = r.get("n_mentions", len(r["mentioned"]))
            h = r.get("n_hal_mentions", len(r["hallucinated"]))
            rows.append((r["coco_id"], m, h))
    return {cid: (m, h) for cid, m, h in rows}


def boot_chair_gap(results):
    s4 = chair_per_image(results, "S4")
    j4 = chair_per_image(results, "J4")
    ids = sorted(set(s4) & set(j4))
    if len(ids) < 50:
        raise RuntimeError(f"paired CHAIR ids too few: {len(ids)}")
    rng = random.Random(SEED)

    def chair_i(sample, table):
        m = sum(table[i][0] for i in sample)
        h = sum(table[i][1] for i in sample)
        return h / max(1, m)

    point = chair_i(ids, s4) - chair_i(ids, j4)
    gaps = []
    for _ in range(B):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        gaps.append(chair_i(sample, s4) - chair_i(sample, j4))
    gaps.sort()
    lo, hi = gaps[int(0.025 * B)], gaps[min(B - 1, int(0.975 * B))]
    return {"point": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "n_pairs": len(ids), "excludes_zero": bool(lo > 0 or hi < 0)}


def pope_adv_rows(results, ckpt):
    out = {}
    p = os.path.join(results, ckpt, "pope_gen.jsonl")
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            if r.get("category") != "adversarial":
                continue
            out[r["id"]] = (parse_yn(r["output"]), r["gt"])
    return out


def f1_of(pairs):
    tp = fp = fn = 0
    for pred, gt in pairs:
        if pred is None:
            continue
        if pred == "yes" and gt == "yes":
            tp += 1
        elif pred == "yes" and gt == "no":
            fp += 1
        elif pred == "no" and gt == "yes":
            fn += 1
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return 2 * prec * rec / max(1e-9, prec + rec)


def boot_pope_gap(results):
    s4 = pope_adv_rows(results, "S4")
    j4 = pope_adv_rows(results, "J4")
    ids = sorted(set(s4) & set(j4))
    if len(ids) < 200:
        raise RuntimeError(f"paired POPE-adv ids too few: {len(ids)}")
    rng = random.Random(SEED)
    point = f1_of([j4[i] for i in ids]) - f1_of([s4[i] for i in ids])
    gaps = []
    for _ in range(B):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        gaps.append(f1_of([j4[i] for i in sample]) - f1_of([s4[i] for i in sample]))
    gaps.sort()
    lo, hi = gaps[int(0.025 * B)], gaps[min(B - 1, int(0.975 * B))]
    return {"point": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "n_pairs": len(ids), "excludes_zero": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpts = ["S0", "S1", "S2", "S3", "S4", "J1", "J2", "J3", "J4"]
    table = {}
    for c in ckpts:
        d = os.path.join(args.results, c)
        row = {}
        try:
            row["chair"] = load_json(os.path.join(d, "chair.json"))
        except FileNotFoundError:
            row["chair"] = None
        try:
            pope = load_json(os.path.join(d, "pope.json"))
            row["pope"] = {r["category"]: r for r in pope}
        except FileNotFoundError:
            row["pope"] = None
        row["tasks"] = {}
        for t in ("scienceqa", "textvqa", "flickr", "vizwiz"):
            try:
                row["tasks"][t] = load_json(os.path.join(d, f"task_{t}.json"))
            except FileNotFoundError:
                row["tasks"][t] = None
        table[c] = row

    def chair_i(c):
        return table[c]["chair"]["chair_i"] if table[c]["chair"] else None

    def adv_f1(c):
        p = table[c]["pope"]
        return p["adversarial"]["f1"] if p and "adversarial" in p else None

    report = {"trajectory": table}

    # A branch that could not be evaluated (missing data) contributes None,
    # never a definitive False (audit finding M5: bool(None or X) collapsed
    # unevaluated branches into FAILs).
    s_vals_c = [chair_i(c) for c in ("S1", "S2", "S3", "S4")]
    s_vals_f = [adv_f1(c) for c in ("S1", "S2", "S3", "S4")]
    g1_branches = []
    if all(v is not None for v in s_vals_c):
        worse = sum(v > s_vals_c[0] for v in s_vals_c[1:])
        gate1_chair = (s_vals_c[3] > s_vals_c[0]) and worse >= 2
        g1_branches.append(gate1_chair)
        report["gate1_chair"] = {"S1..S4": s_vals_c, "passes": gate1_chair}
    else:
        report["gate1_chair"] = {"S1..S4": s_vals_c, "passes": None,
                                 "note": "missing checkpoints - not evaluated"}
    if all(v is not None for v in s_vals_f):
        worse = sum(v < s_vals_f[0] for v in s_vals_f[1:])
        gate1_pope = (s_vals_f[3] < s_vals_f[0]) and worse >= 2
        g1_branches.append(gate1_pope)
        report["gate1_pope_adv"] = {"S1..S4": s_vals_f, "passes": gate1_pope}
    else:
        report["gate1_pope_adv"] = {"S1..S4": s_vals_f, "passes": None,
                                    "note": "missing checkpoints - not evaluated"}
    gate1 = (any(g1_branches) if g1_branches else None)

    g2_branches = []
    try:
        g = boot_chair_gap(args.results)
        report["gate2_chair_gap"] = g
        g2_branches.append(g["point"] > 0 and g["excludes_zero"])
    except Exception as e:
        report["gate2_chair_gap"] = {"error": repr(e)[:300], "passes": None}
    try:
        g = boot_pope_gap(args.results)
        report["gate2_pope_gap"] = g
        g2_branches.append(g["point"] > 0 and g["excludes_zero"])
    except Exception as e:
        report["gate2_pope_gap"] = {"error": repr(e)[:300], "passes": None}
    gate2 = (any(g2_branches) if g2_branches else None)

    if gate1 is None or gate2 is None:
        report["GATE_PASSES"] = None
        report["gate_note"] = "one or both gates could not be evaluated - NO-DATA, not a fail"
    else:
        report["GATE_PASSES"] = bool(gate1 and gate2)

    confounds = {}
    for c in ckpts:
        row = {}
        if table[c]["chair"]:
            row["truncation_rate"] = table[c]["chair"]["truncation_rate"]
            row["mean_new_tokens"] = table[c]["chair"]["mean_new_tokens"]
            row["mentions_per_caption"] = table[c]["chair"]["mentions_per_caption"]
        if table[c]["pope"] and "all" in table[c]["pope"]:
            row["pope_yes_rate"] = table[c]["pope"]["all"]["yes_rate"]
            row["pope_parse_fail"] = table[c]["pope"]["all"]["parse_fail_rate"]
        confounds[c] = row
    report["confound_panel"] = confounds

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "trajectory"}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
