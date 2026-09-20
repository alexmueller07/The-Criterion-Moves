"""Score POPE generations: per-category accuracy/P/R/F1, yes-rate, parse-fail rate.

Yes is the positive class. Unparseable answers are counted as failures and
reported — never silently coerced (non-discriminating-check rule).
"""
import argparse
import csv
import json
from collections import defaultdict


import re


def parse_yn(text):
    # Word-boundary matching only. Substring fallbacks ("snowboarding"->"no",
    # "eyes"->"yes") biased toward spurious answers exactly when verbosity
    # drift begins, while suppressing the parse-fail alarm (red-team audit
    # finding C1, 2026-08-31).
    words = re.findall(r"[a-z]+", text.strip().lower())
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--ckpt", required=True, help="checkpoint label, e.g. S2")
    args = ap.parse_args()

    buckets = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0,
                                   "yes": 0, "fail": 0, "n": 0})
    for line in open(args.gen):
        r = json.loads(line)
        cat = r.get("category", "unknown")
        for c in (cat, "all"):
            b = buckets[c]
            b["n"] += 1
            pred = parse_yn(r["output"])
            gt = r["gt"]
            if pred is None:
                b["fail"] += 1
                continue
            if pred == "yes":
                b["yes"] += 1
            if pred == "yes" and gt == "yes":
                b["tp"] += 1
            elif pred == "yes" and gt == "no":
                b["fp"] += 1
            elif pred == "no" and gt == "no":
                b["tn"] += 1
            else:
                b["fn"] += 1

    rows = []
    for cat, b in sorted(buckets.items()):
        parsed = b["n"] - b["fail"]
        prec = b["tp"] / max(1, b["tp"] + b["fp"])
        rec = b["tp"] / max(1, b["tp"] + b["fn"])
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        rows.append({
            "ckpt": args.ckpt, "category": cat, "n": b["n"],
            "accuracy": round((b["tp"] + b["tn"]) / max(1, parsed), 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4),
            "yes_rate": round(b["yes"] / max(1, parsed), 4),
            "parse_fail_rate": round(b["fail"] / max(1, b["n"]), 4),
        })

    with open(args.out_prefix + ".json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(args.out_prefix + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r, flush=True)


if __name__ == "__main__":
    main()
