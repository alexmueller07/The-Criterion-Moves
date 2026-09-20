"""MME-Hallucination scorer: judge-free acc / acc+ over the four object-
hallucination perception subtasks (existence, count, position, color).

Faithful reimplementation of the official MME tool's scoring, verified at source
2026-09-03 from tools/eval_tool.zip (sha256
b8125e2a7c3418e5761c12b3cfe4f1624b3c53ba44009e75b7d3f797d3d8acee) of
github.com/BradyFU/Awesome-Multimodal-Large-Language-Models, branch Evaluation
@ dd2950902889cd614d4edf606827240166d29381 (calculation.py + Your_Results/).

Exact official conventions reproduced (calculation.py):
  * Answer parsing (parse_pred_ans): lowercase the response; if it is exactly
    "yes"/"no" use it; else look at the FIRST 4 CHARACTERS -- "yes" in prefix ->
    yes, elif "no" in prefix -> no, else "other". Faithful quirks: this maps a
    leading "not"/"none" to "no" (prefix "not "/"none" contains "no"), and
    "other" is never correct. We keep this verbatim because acc/acc+ are DEFINED
    by it -- swapping in our POPE word-boundary parser would change the numbers
    and break MME-leaderboard comparability. (yes_rate/other_rate are emitted so
    a parser-driven degeneracy is still visible.)
  * acc  = correct / total_questions  ("other" counts as wrong; official uses
           sklearn accuracy_score over labels {yes:1,no:0,other:-1}).
  * acc+ = images where BOTH questions are correct / images.  <-- the metric MME
           is known for. Each image has exactly two questions (one gt=yes, one
           gt=no).
  * subtask score = acc*100 + acc+*100   (each subtask max 200; official prints
           this as "score").
  * precision/recall over the NON-"other" predictions only, yes as the positive
    class (sklearn precision_score/recall_score, average='binary'); reported here
    but NOT part of the MME score.

We pair the two questions of an image by IMAGE IDENTITY (subtask, mme_image),
not by the official's file-adjacency (divide_chunks of 2). On well-formed data
this is identical to the official pairing but does not silently depend on row
order; we hard-require exactly two questions per image (loud error otherwise).

MME-Hallucination headline = existence + count + position + color subtask scores
(max 800), the object-hallucination slice of MME's Perception split used
throughout the LVLM-hallucination literature.

Input --gen JSONL rows (produced by eval_gen.py from fetch_mme.py's prompts):
  {id, output, subtask, gt, mme_image, n_new_tokens, truncated}
  gt in {"yes","no"} (case-insensitive); subtask in the four names below.
Outputs, house style:
  <out_prefix>.json / .csv        one row per subtask + an "all" summary row
  <out_prefix>_per_image.jsonl    per-(subtask,image) rows for bootstrap
Usage:
  python metrics_mme.py --gen gen/MME_S2.jsonl --out_prefix results/S2_mme --ckpt S2
"""
import argparse
import csv
import json
import sys
from collections import defaultdict

HALL_SUBTASKS = ["existence", "count", "position", "color"]


def parse_pred_ans(pred_ans):
    """Verbatim port of the official MME calculation.py parse_pred_ans."""
    pred_ans = pred_ans.lower()
    if pred_ans in ["yes", "no"]:
        return pred_ans
    prefix = pred_ans[:4]
    if "yes" in prefix:
        return "yes"
    if "no" in prefix:
        return "no"
    return "other"


def score_subtask(rows, ckpt, subtask):
    by_img = defaultdict(list)
    for r in rows:
        by_img[r["mme_image"]].append(r)

    n_q = correct = yes_pred = other = 0
    n_img = both_correct = 0
    tp = fp = fn = tn = 0  # yes = positive
    tot_len = tot_trunc = 0
    per_img = []
    for img, items in sorted(by_img.items()):
        if len(items) != 2:
            sys.exit(f"MME {subtask}: image {img!r} has {len(items)} questions, "
                     "expected exactly 2 (one gt=yes, one gt=no)")
        n_img += 1
        img_correct = 0
        for r in items:
            gt = str(r["gt"]).strip().lower()
            if gt not in ("yes", "no"):
                sys.exit(f"MME {subtask}: gt {r['gt']!r} for id {r.get('id')} "
                         "not in {yes,no}")
            pred = parse_pred_ans(str(r["output"]))
            n_q += 1
            tot_len += r.get("n_new_tokens", 0)
            tot_trunc += int(r.get("truncated", False))
            if pred == "yes":
                yes_pred += 1
            if pred == "other":
                other += 1
            if pred == gt:
                correct += 1
                img_correct += 1
            # precision/recall (yes positive) over non-other predictions only
            if pred in ("yes", "no"):
                if pred == "yes" and gt == "yes":
                    tp += 1
                elif pred == "yes" and gt == "no":
                    fp += 1
                elif pred == "no" and gt == "yes":
                    fn += 1
                else:
                    tn += 1
        if img_correct == 2:
            both_correct += 1
        per_img.append({"ckpt": ckpt, "subtask": subtask, "mme_image": img,
                        "img_correct": img_correct,
                        "acc_plus_hit": int(img_correct == 2)})

    acc = correct / max(1, n_q)
    acc_plus = both_correct / max(1, n_img)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    row = {
        "ckpt": ckpt, "subtask": subtask,
        "n_questions": n_q, "n_images": n_img,
        "acc": round(acc, 4), "acc_plus": round(acc_plus, 4),
        "score": round(acc * 100 + acc_plus * 100, 2),
        "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
        "yes_rate": round(yes_pred / max(1, n_q), 4),
        "other_rate": round(other / max(1, n_q), 4),
        "other_num": other,
        "mean_new_tokens": round(tot_len / max(1, n_q), 1),
        "truncation_rate": round(tot_trunc / max(1, n_q), 4),
        "parse": "official_prefix4",
    }
    return row, per_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--ckpt", required=True, help="checkpoint label, e.g. S2")
    args = ap.parse_args()

    by_sub = defaultdict(list)
    for lineno, line in enumerate(open(args.gen), 1):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        for k in ("output", "subtask", "gt", "mme_image"):
            if k not in r:
                sys.exit(f"{args.gen}:{lineno}: missing key {k!r}")
        st = r["subtask"]
        if st not in HALL_SUBTASKS:
            sys.exit(f"{args.gen}:{lineno}: subtask {st!r} not in {HALL_SUBTASKS}")
        by_sub[st].append(r)

    present = [s for s in HALL_SUBTASKS if s in by_sub]
    if not present:
        sys.exit(f"{args.gen}: no rows in any of {HALL_SUBTASKS}")
    if present != HALL_SUBTASKS:
        print(f"[mme] WARNING: only subtasks {present} present; the "
              "MME-Hallucination headline (max 800) needs all four "
              f"{HALL_SUBTASKS} -- reporting a partial sum.", flush=True)

    rows, per_img_all = [], []
    tot_q = tot_correct = tot_other = 0
    score_sum = 0.0
    for st in present:
        row, per_img = score_subtask(by_sub[st], args.ckpt, st)
        rows.append(row)
        per_img_all.extend(per_img)
        score_sum += row["score"]
        tot_q += row["n_questions"]
        tot_correct += round(row["acc"] * row["n_questions"])
        tot_other += row["other_num"]

    rows.append({
        "ckpt": args.ckpt, "subtask": "all",
        "n_questions": tot_q, "n_images": sum(r["n_images"] for r in rows),
        "acc": round(tot_correct / max(1, tot_q), 4), "acc_plus": "",
        "score": round(score_sum, 2),  # MME-Hallucination headline (max 800 if 4)
        "precision": "", "recall": "", "f1": "",
        "yes_rate": "", "other_rate": round(tot_other / max(1, tot_q), 4),
        "other_num": tot_other,
        "mean_new_tokens": "", "truncation_rate": "",
        "parse": "official_prefix4",
        "subtasks_present": ",".join(present),
        "full_800_scale": len(present) == 4,
    })

    with open(args.out_prefix + "_per_image.jsonl", "w") as f:
        for row in per_img_all:
            f.write(json.dumps(row) + "\n")
    with open(args.out_prefix + ".json", "w") as f:
        json.dump(rows, f, indent=2)
    fieldnames = sorted({k for r in rows for k in r})
    with open(args.out_prefix + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    for r in rows:
        print(r, flush=True)


if __name__ == "__main__":
    main()
