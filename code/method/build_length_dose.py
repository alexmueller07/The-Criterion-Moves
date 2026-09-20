#!/usr/bin/env python3
"""Length-dose variants of one task: the controlled test of the format mechanism.

WHY (2026-09-11). Across the six UCIT tasks, how far a task pulls the yes/no
criterion is predicted by how long its ANSWERS are -- long-answer tasks push
conservative, short-answer tasks push liberal -- even though not one of the six
contains any yes/no training answers at all (yes/no fraction 0.0000 everywhere).

But that evidence is weak in two specific ways, both recorded in the
pre-registration. The answer lengths are BIMODAL (four tasks at ~1 word, two at
~12), so it is a two-group contrast with an exact permutation p of 1/15 = 0.067,
the smallest that design can produce. And answer length is perfectly confounded
with task identity: "long-answer tasks" and "Flickr30k and VizWiz specifically"
are, on that data, the same hypothesis.

This script removes both problems at once. It takes ONE task and emits variants
whose answers are truncated to k words, for several k. Same images, same prompts,
same example count, same task -- the ONLY thing that varies is answer length. So:

  * length becomes a graded dose instead of two clumps, and
  * length is decoupled from task identity, because the task is held fixed.

Each variant is then trained as a single stage from the base model
(`--arm single:<name>`), which isolates one task's effect rather than confounding
it with a sequence, and costs ~25 GPU-minutes instead of ~2.5 hours.

Prediction under the format account, pre-registered here before the runs: the
criterion shift should be MONOTONE in k, negative at k=1 and positive at the
untruncated length. If the shift is flat in k, the account is wrong and what
mattered was the task's identity, not its answer format.

Usage:
  python3 build_length_dose.py --root <data_ucit> --task Flickr30k --ks 1,2,4,8
"""
import argparse
import json
import os
import statistics as st
import sys

TARGET_FIELDS = ("target", "answer", "output", "response", "caption")


def target_field(row):
    for f in TARGET_FIELDS:
        if isinstance(row.get(f), str) and row[f].strip():
            return f
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="data_ucit dir (contains tasks/)")
    ap.add_argument("--task", default="Flickr30k")
    ap.add_argument("--ks", default="1,2,4,8", help="comma-separated word budgets")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    src = os.path.join(args.root, "tasks", args.task, "train.jsonl")
    if not os.path.isfile(src):
        raise SystemExit("missing %s" % src)
    rows = [json.loads(l) for l in open(src) if l.strip()]
    if not rows:
        raise SystemExit("empty %s" % src)

    tf = target_field(rows[0])
    if tf is None:
        raise SystemExit("no target-like field in %s; saw keys %s"
                         % (src, sorted(rows[0])))
    full = [len(r[tf].split()) for r in rows if r.get(tf)]
    print("[dose] %s: %d rows, target field %r, mean %.2f words, median %.0f"
          % (args.task, len(rows), tf, st.mean(full), st.median(full)))

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    made = []
    for k in ks:
        name = "%s_w%d" % (args.task, k)
        out_dir = os.path.join(args.root, "tasks", name)
        n_trunc = 0
        new_rows = []
        for r in rows:
            r2 = dict(r)
            w = str(r.get(tf, "")).split()
            if len(w) > k:
                n_trunc += 1
            r2[tf] = " ".join(w[:k])
            new_rows.append(r2)
        lens = [len(r[tf].split()) for r in new_rows]
        print("[dose]   %-18s mean %5.2f words, %5.1f%% of rows truncated"
              % (name, st.mean(lens), 100.0 * n_trunc / len(new_rows)))
        if not args.dry_run:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "train.jsonl"), "w") as f:
                for r in new_rows:
                    f.write(json.dumps(r) + "\n")
            # carry any sibling files (val/test) through untouched so the variant
            # is a drop-in for the original task directory
            for extra in ("val.jsonl", "test.jsonl"):
                p = os.path.join(args.root, "tasks", args.task, extra)
                if os.path.isfile(p) and not os.path.exists(os.path.join(out_dir, extra)):
                    os.symlink(p, os.path.join(out_dir, extra))
        made.append({"name": name, "k": k, "mean_words": round(st.mean(lens), 3),
                     "frac_truncated": round(n_trunc / len(new_rows), 4),
                     "n_rows": len(new_rows)})

    if not args.dry_run:
        rep = os.path.join(args.root, "tasks", "%s_lengthdose_report.json" % args.task)
        json.dump({"source_task": args.task, "target_field": tf,
                   "source_mean_words": round(st.mean(full), 3),
                   "source_median_words": st.median(full),
                   "n_rows": len(rows), "variants": made,
                   "design": ("same images, same prompts, same row count; only the "
                              "answer is truncated. Length is therefore decoupled "
                              "from task identity, which it is not across the six "
                              "UCIT tasks."),
                   "prediction": ("criterion shift monotone in k: negative at k=1, "
                                  "positive at the untruncated length. A flat "
                                  "profile refutes the format account.")},
                  open(rep, "w"), indent=2)
        print("[dose] wrote %s" % rep)
    print("LENGTH_DOSE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
