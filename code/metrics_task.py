"""Task-accuracy scorers for the forgetting matrix.

scienceqa: MC letter match (robust extraction, parse-fail reported)
textvqa/vizwiz: VQA accuracy = min(matching_annotator_answers / 3, 1)
flickr: max unigram-F1 against the 5 reference captions (trajectory signal only)
"""
import argparse
import csv
import json
import re
import string


def norm_answer(s):
    s = s.strip().lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def parse_mc_letter(output):
    # A bare \b[A-E]\b matches the article "A" in caption-style outputs, so
    # format collapse would read as answer A with parse_fail ~ 0 (red-team
    # audit finding C2, 2026-08-31). Accept only: a bare letter (with optional
    # punctuation), letter-then-punctuation-then-text, or an explicit
    # "answer is X"/"option X" phrase.
    s = output.strip()
    m = re.match(r"^\(?([A-Ea-e])\)?[.:,]?$", s)
    if m:
        return m.group(1).upper()
    m = re.match(r"^\(?([A-Ea-e])\)?[.:,)]\s", s)
    if m:
        return m.group(1).upper()
    m = re.search(r"(?:answer is|answer:|option)\s*\(?([A-Ea-e])\)?\b", s, re.I)
    if m:
        return m.group(1).upper()
    return None


def score_scienceqa(rows):
    n = ok = fail = 0
    for r in rows:
        n += 1
        letter = parse_mc_letter(r["output"])
        if letter is None:
            fail += 1
            continue
        if letter == r["meta"]["answer_letter"]:
            ok += 1
    return {"metric": "mc_acc", "score": round(ok / max(1, n - fail), 4),
            "n": n, "parse_fail_rate": round(fail / max(1, n), 4)}


NUM_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
             "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
             "ten": "10"}


def score_vqa(rows):
    # Official VQA accuracy averages min(matches/3, 1) over the 10
    # leave-one-out annotator subsets; for m exact matches out of 10 that is
    # [m*min((m-1)/3,1) + (10-m)*min(m/3,1)] / 10 (audit finding M3).
    n, total = 0, 0.0
    for r in rows:
        n += 1
        pred = norm_answer(r["output"])
        pred = " ".join(NUM_WORDS.get(w, w) for w in pred.split())
        answers = [norm_answer(a) for a in r["meta"]["answers"]]
        answers = [" ".join(NUM_WORDS.get(w, w) for w in a.split()) for a in answers]
        m = sum(a == pred for a in answers)
        k = len(answers)
        if k == 10:
            total += (m * min((m - 1) / 3.0, 1.0) + (k - m) * min(m / 3.0, 1.0)) / k
        else:
            total += min(m / 3.0, 1.0)
    return {"metric": "vqa_acc", "score": round(total / max(1, n), 4),
            "n": n, "parse_fail_rate": 0.0}


def score_caption(rows):
    def uf1(pred, ref):
        p = set(re.findall(r"[a-z]+", pred.lower()))
        q = set(re.findall(r"[a-z]+", ref.lower()))
        if not p or not q:
            return 0.0
        inter = len(p & q)
        prec, rec = inter / len(p), inter / len(q)
        return 2 * prec * rec / max(1e-9, prec + rec)

    n, total = 0, 0.0
    for r in rows:
        n += 1
        total += max(uf1(r["output"], ref) for ref in r["meta"]["refs"])
    return {"metric": "caption_uf1", "score": round(total / max(1, n), 4),
            "n": n, "parse_fail_rate": 0.0}


SCORERS = {"scienceqa": score_scienceqa, "textvqa": score_vqa,
           "vizwiz": score_vqa, "flickr": score_caption}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--task", required=True, choices=list(SCORERS))
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.gen)]
    res = SCORERS[args.task](rows)
    res.update({"ckpt": args.ckpt, "task": args.task,
                "mean_new_tokens": round(sum(r["n_new_tokens"] for r in rows) / max(1, len(rows)), 1),
                "truncation_rate": round(sum(r["truncated"] for r in rows) / max(1, len(rows)), 4)})
    with open(args.out_prefix + ".json", "w") as f:
        json.dump(res, f, indent=2)
    with open(args.out_prefix + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res.keys()))
        w.writeheader()
        w.writerow(res)
    print(res, flush=True)


if __name__ == "__main__":
    main()
