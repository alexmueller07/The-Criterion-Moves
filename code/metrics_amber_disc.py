"""AMBER discriminative metrics (14,216 yes/no queries) -- vendored, judge-free.

Sibling of metrics_amber.py (generative CHAIR/Cover/Hal/Cog). The discriminative
protocol differs enough from the generative one -- different queries (ids
1005-15220), different metrics (Accuracy/Precision/Recall/F1), and a different
positive class -- that it lives in its own scorer rather than bolting onto the
generative path.

Verified at source 2026-09-03, pinned commit
534babf6bbfcce2e735c26289dedfb21cef3c939 of github.com/junyangwang0410/AMBER:
  inference.py (discriminative branch + print block), data/annotations.json,
  data/metrics.txt.
Discriminative annotation inventory counted directly from annotations.json at
that commit (15,220 total; 1,004 generative; 14,216 discriminative):
  existence  : type 'discriminative-hallucination'          4,924
  attribute  : 'discriminative-attribute-state'  4,764
               'discriminative-attribute-number' 2,072
               'discriminative-attribute-action'   792       (= 7,628)
  relation   : 'discriminative-relation' 975 + 'relation' 689 (= 1,664)
  truth: 4,789 yes / 9,427 no.
Each entry: {id, type, truth in {'yes','no'}}. Query text is in
data/query/query_discriminative.json (id, image, query); not needed to score.

KEY CONVENTION (reproduced faithfully): AMBER's Precision/Recall/F1 use **"No" as
the positive class** -- i.e. they measure the model's ability to correctly answer
"No" (the queried attribute/object is absent). This is the OPPOSITE polarity from
POPE/metrics_pope.py, where "Yes" is positive. Accuracy is overall (either
answer). The official formulas (inference.py print block):
  Accuracy  = correct / N
  Precision = (#pred==No & truth==no) / (#pred==No)
  Recall    = (#pred==No & truth==no) / (#truth==no)
  F1        = 2PR/(P+R+eps)      eps: 0.001 for existence, 0.0001 elsewhere
The official answer test is EXACT string equality: response == 'Yes' / 'No'
(case-sensitive, no strip) -- brittle ("Yes." scores as neither).

This scorer emits BOTH:
  * PRIMARY (house style): a robust word-boundary yes/no parse (same as
    metrics_pope.parse_yn), No-positive Accuracy/Precision/Recall/F1 per
    dimension + overall, plus yes_rate and parse_fail_rate (never silently
    coerced). Use these for the within-study trajectory.
  * AUDIT (`*_official_exact`): the official exact-'Yes'/'No' metrics, with the
    0.001-initialised denominators of data/metrics.txt and the official per-dim
    F1 epsilons and 1-decimal rounding -- byte-comparable to running
    inference.py --evaluation_type a/de/da/dr. Use these for AMBER-leaderboard
    comparability. The gap between the two exposes format/verbosity drift.

Input --gen JSONL rows: {id, amber_id, output, n_new_tokens, truncated}
  amber_id = AMBER query id (1005-15220 for discriminative).
Outputs (house style):
  <out_prefix>.json / .csv        one row per dimension (existence, attribute,
                                  attribute:state/number/action, relation, all)
  <out_prefix>_per_item.jsonl     per-query rows for bootstrap
Usage:
  python metrics_amber_disc.py --gen gen/AMBERD_S2.jsonl \
      --amber_data amber_data/ --out_prefix results/S2_amber_disc --ckpt S2
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

# dimension (headline) and subtype per official type string.
TYPE_MAP = {
    "discriminative-hallucination": ("existence", "existence"),
    "discriminative-attribute-state": ("attribute", "state"),
    "discriminative-attribute-number": ("attribute", "number"),
    "discriminative-attribute-action": ("attribute", "action"),
    "discriminative-relation": ("relation", "relation"),
    "relation": ("relation", "relation"),  # official 'else' bucket
}
EXIST_F1_EPS = 0.001      # official hallucination_F1 uses +0.001
OTHER_F1_EPS = 0.0001     # every other official F1 uses +0.0001


def parse_yn(text):
    """Word-boundary yes/no, identical policy to metrics_pope.parse_yn."""
    words = re.findall(r"[a-z]+", str(text).strip().lower())
    if not words:
        return None
    if words[0] in ("yes", "yeah", "yep"):
        return "yes"
    if words[0] in ("no", "nope"):
        return "no"
    for w in words[:3]:
        if w == "yes":
            return "yes"
        if w == "no":
            return "no"
    return None


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def robust_metrics(items):
    """items: list of (truth, pred) with pred in {'yes','no',None}. No-positive."""
    n = len(items)
    correct = pred_no = actual_no = tp_no = fails = yes_pred = 0
    for truth, pred in items:
        if pred is None:
            fails += 1
        if pred == "yes":
            yes_pred += 1
        if pred == truth:
            correct += 1
        if truth == "no":
            actual_no += 1
        if pred == "no":
            pred_no += 1
            if truth == "no":
                tp_no += 1
    prec = tp_no / max(1, pred_no)
    rec = tp_no / max(1, actual_no)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {
        "n": n,
        "accuracy": round(correct / max(1, n), 4),
        "precision_no": round(prec, 4),
        "recall_no": round(rec, 4),
        "f1_no": round(f1, 4),
        "yes_rate": round(yes_pred / max(1, n), 4),
        "parse_fail_rate": round(fails / max(1, n), 4),
    }


def official_metrics(items, f1_eps):
    """items: list of (truth, exact_response_str). Byte-faithful to inference.py:
    exact 'Yes'/'No', denominators initialised at 0.001, F1 epsilon, round to 1."""
    correct_num = no_num = ans_no_num = 0.001
    correct_score = no_score = ans_no_score = 0
    for truth, resp in items:
        correct_num += 1
        if truth == "yes":
            if resp == "Yes":
                correct_score += 1
        else:  # truth == 'no'
            no_num += 1
            if resp == "No":
                correct_score += 1
                no_score += 1
        if resp == "No":
            ans_no_num += 1
            if truth == "no":
                ans_no_score += 1
    acc = round(correct_score / correct_num * 100, 1)
    prec = round(ans_no_score / ans_no_num * 100, 1)
    rec = round(no_score / no_num * 100, 1)
    f1 = round(2 * (prec / 100) * (rec / 100)
               / ((prec / 100) + (rec / 100) + f1_eps) * 100, 1)
    return {"accuracy_official_exact": acc, "precision_no_official_exact": prec,
            "recall_no_official_exact": rec, "f1_no_official_exact": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True,
                    help="JSONL: {id, amber_id, output, n_new_tokens, truncated}")
    ap.add_argument("--amber_data", required=True,
                    help="dir with annotations.json (see fetch_amber_assets.py)")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    ann_path = os.path.join(args.amber_data, "annotations.json")
    ann_by_id = {a["id"]: a for a in json.load(open(ann_path, encoding="utf-8"))}

    # buckets keyed by dimension label; 'all' aggregates every discriminative item.
    robust_b = defaultdict(list)   # label -> [(truth, pred)]
    official_b = defaultdict(list)  # label -> [(truth, exact_resp)]
    per_item = []
    tot_len = tot_trunc = n = 0
    for lineno, line in enumerate(open(args.gen), 1):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        for k in ("id", "amber_id", "output", "n_new_tokens", "truncated"):
            if k not in r:
                sys.exit(f"{args.gen}:{lineno}: missing key {k!r}")
        ann = ann_by_id.get(r["amber_id"])
        if ann is None:
            sys.exit(f"{args.gen}:{lineno}: amber_id {r['amber_id']} not in "
                     "annotations.json")
        typ = ann["type"]
        if typ == "generative" or typ not in TYPE_MAP:
            sys.exit(f"{args.gen}:{lineno}: amber_id {r['amber_id']} has type "
                     f"{typ!r}, not a discriminative type -- wrong scorer")
        truth = ann["truth"]
        if truth not in ("yes", "no"):
            sys.exit(f"annotation {r['amber_id']}: truth {truth!r} not yes/no")
        dim, sub = TYPE_MAP[typ]
        pred = parse_yn(r["output"])
        exact = r["output"]  # exact string as generated (already .strip()ed upstream)

        labels = ["all", dim]
        if dim == "attribute":
            labels.append(f"attribute:{sub}")
        for lab in labels:
            robust_b[lab].append((truth, pred))
            official_b[lab].append((truth, exact))

        n += 1
        tot_len += r["n_new_tokens"]
        tot_trunc += int(r["truncated"])
        per_item.append({"id": r["id"], "amber_id": r["amber_id"],
                         "type": typ, "dim": dim, "sub": sub, "truth": truth,
                         "pred": pred, "correct": int(pred == truth),
                         "n_new_tokens": r["n_new_tokens"],
                         "truncated": r["truncated"]})

    if n == 0:
        sys.exit(f"{args.gen}: no discriminative rows")

    order = ["existence", "attribute", "attribute:state", "attribute:number",
             "attribute:action", "relation", "all"]
    ann_sha = sha256_file(ann_path)
    rows = []
    for lab in order:
        if lab not in robust_b:
            continue
        f1_eps = EXIST_F1_EPS if lab == "existence" else OTHER_F1_EPS
        rm = robust_metrics(robust_b[lab])
        om = official_metrics(official_b[lab], f1_eps)
        # per-dim token audit
        items_here = [p for p in per_item
                      if lab == "all" or p["dim"] == lab.split(":")[0]
                      and (":" not in lab or p["sub"] == lab.split(":")[1])]
        mlen = round(sum(p["n_new_tokens"] for p in items_here)
                     / max(1, len(items_here)), 1)
        mtr = round(sum(int(p["truncated"]) for p in items_here)
                    / max(1, len(items_here)), 4)
        row = {"ckpt": args.ckpt, "dimension": lab}
        row.update(rm)
        row.update(om)
        row.update({"mean_new_tokens": mlen, "truncation_rate": mtr,
                    "parse": "word_boundary(primary)+official_exact(audit)",
                    "positive_class": "no",
                    "annotations_sha256": ann_sha})
        rows.append(row)

    with open(args.out_prefix + "_per_item.jsonl", "w") as f:
        for row in per_item:
            f.write(json.dumps(row) + "\n")
    with open(args.out_prefix + ".json", "w") as f:
        json.dump(rows, f, indent=2)
    fieldnames = list(rows[0].keys())
    with open(args.out_prefix + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r, flush=True)


if __name__ == "__main__":
    main()
