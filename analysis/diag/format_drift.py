"""A9/A10 diagnostics: format-collapse and answer-drift audit.

Pre-specified in design_notes/analysis_ideas.md (A9: SEFE-style format audit;
A10: forgetting <-> hallucination decoupling needs strict vs lenient forgetting).

Computes, per checkpoint (S0..S4, J1..J4) x task (scienceqa, textvqa, flickr, vizwiz):
  1. answer-length distribution (words) + drift vs S0
  2. strict vs lenient task scores
       scienceqa strict   = parse_mc_letter, parse-fail counts WRONG (ok/n)
       scienceqa reported = parse_mc_letter over parsed rows only (the pilot's
                            task_scienceqa.json number, ok/(n-fail))
       scienceqa lenient  = strict-correct OR the correct option's TEXT appears
                            (normalized contiguous-token containment) in the output.
                            Option texts are recovered from the prompt ("A. <text>"
                            lines) because meta carries only answer_letter; recovery
                            verified for all rows (assert below).
       scienceqa lenient_excl = as lenient, but a text match is rejected when any
                            OTHER option's text also appears (multi-option guard).
       textvqa/vizwiz strict  = the pilot's VQA accuracy (exact normalized match,
                            leave-one-out formula, metrics_task.score_vqa semantics)
       textvqa/vizwiz lenient = same formula, but an annotator answer "matches" when
                            norm_answer(ann) is a contiguous token subsequence of
                            norm_answer(output)
       textvqa/vizwiz lenient_ng = lenient with a negation guard: a containment
                            match is rejected when a negation token occurs within
                            3 tokens before the matched span (A9's named divergence
                            regime: "not a cat" contains "cat")
       flickr             = caption uF1 (no strict/lenient distinction; copied into
                            both columns of the forgetting matrix)
  3. 'unanswerable' rate on ALL four tasks: eq (norm_answer(output)=='unanswerable')
     and contains (case-insensitive substring); VizWiz GT unanswerable rate as anchor
  4. caption-style-output rate on non-caption tasks: fraction of outputs > 15 words
  5. forgetting matrix F_t(k) = score_t(S_stage(t)) - score_t(S_k), k > stage(t),
     strict vs lenient side by side, with paired-bootstrap SE over the 500 shared
     val items (ids verified aligned across all 9 checkpoints)

Outputs: markdown tables to stdout, full numbers to analysis/diag/format_drift.json.
Usage: python3 analysis/diag/format_drift.py
"""
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pilot"))
from metrics_task import NUM_WORDS, norm_answer, parse_mc_letter  # noqa: E402

RESULTS = ROOT / "results"
SEQ_CKPTS = ["S0", "S1", "S2", "S3", "S4"]
JOINT_CKPTS = ["J1", "J2", "J3", "J4"]
CKPTS = SEQ_CKPTS + JOINT_CKPTS
TASKS = ["scienceqa", "textvqa", "flickr", "vizwiz"]
# stage at which each task is trained in the sequential arm
STAGE = {"scienceqa": 1, "textvqa": 2, "flickr": 3, "vizwiz": 4}
CAPTION_WORDS = 15  # ">15 words" = caption-style output on a non-caption task
N_BOOT = 2000
RNG_SEED = 20260830

NEGATION_TOKENS = {"no", "not", "never", "cannot", "cant", "dont", "doesnt",
                   "didnt", "isnt", "arent", "wasnt", "werent", "wont",
                   "without", "none", "neither", "nor"}


# ---------------------------------------------------------------- data loading

def load(ckpt, task):
    with open(RESULTS / ckpt / f"{task}_gen.jsonl") as f:
        return [json.loads(line) for line in f]


# ---------------------------------------------------------------- normalizers

def norm_tokens(s):
    """metrics_task normalization (norm_answer + number-word map) -> token list."""
    s = norm_answer(s)
    return [NUM_WORDS.get(w, w) for w in s.split()]


def find_subseq(hay, needle):
    """First start index of needle as a contiguous subsequence of hay, else -1.
    Empty needle matches only an empty hay (exact-empty), guarding answers that
    normalize away entirely (e.g. 'the')."""
    if not needle:
        return 0 if not hay else -1
    n = len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i:i + n] == needle:
            return i
    return -1


def negated_at(hay, start):
    """Negation token within the 3 tokens before position start."""
    return any(t in NEGATION_TOKENS for t in hay[max(0, start - 3):start])


# ------------------------------------------------------- per-row score vectors

def parse_choices(prompt):
    """Recover letter -> option text from the prompt's 'A. <text>' lines."""
    return dict(re.findall(r"^([A-E])\.\s*(.+)$", prompt, re.M))


def rows_scienceqa(rows):
    """Per-row dict of indicator scores + format flags."""
    out = []
    for r in rows:
        choices = parse_choices(r["prompt"])
        gold = r["meta"]["answer_letter"]
        assert gold in choices, f"choice recovery failed for {r['id']}"
        letter = parse_mc_letter(r["output"])
        strict = int(letter == gold)
        parse_fail = int(letter is None)
        out_toks = norm_tokens(r["output"])
        text_hits = {L: find_subseq(out_toks, norm_tokens(txt)) >= 0
                     for L, txt in choices.items()}
        gold_text = text_hits[gold]
        other_text = any(v for L, v in text_hits.items() if L != gold)
        lenient = int(strict or gold_text)
        lenient_excl = int(strict or (gold_text and not other_text))
        bare_letter = int(re.match(r"^\(?[A-Ea-e]\)?[.:,]?$", r["output"].strip())
                          is not None)
        out.append(dict(strict=strict, lenient=lenient, lenient_excl=lenient_excl,
                        parse_fail=parse_fail, bare_letter=bare_letter,
                        text_rescued=int(parse_fail and gold_text)))
    return out


def vqa_formula(m, k):
    """metrics_task.score_vqa audit-M3 semantics for m matches out of k answers."""
    if k == 10:
        return (m * min((m - 1) / 3.0, 1.0) + (k - m) * min(m / 3.0, 1.0)) / k
    return min(m / 3.0, 1.0)


def rows_vqa(rows):
    out = []
    for r in rows:
        pred_toks = norm_tokens(r["output"])
        pred = " ".join(pred_toks)
        anns = [norm_tokens(a) for a in r["meta"]["answers"]]
        k = len(anns)
        m_strict = sum(" ".join(a) == pred for a in anns)
        m_len = m_ng = 0
        for a in anns:
            pos = find_subseq(pred_toks, a)
            if pos >= 0:
                m_len += 1
                if not negated_at(pred_toks, pos):
                    m_ng += 1
        out.append(dict(strict=vqa_formula(m_strict, k),
                        lenient=vqa_formula(m_len, k),
                        lenient_ng=vqa_formula(m_ng, k)))
    return out


def rows_flickr(rows):
    def uf1(pred, ref):
        p = set(re.findall(r"[a-z]+", pred.lower()))
        q = set(re.findall(r"[a-z]+", ref.lower()))
        if not p or not q:
            return 0.0
        inter = len(p & q)
        prec, rec = inter / len(p), inter / len(q)
        return 2 * prec * rec / max(1e-9, prec + rec)
    return [dict(strict=max(uf1(r["output"], ref) for ref in r["meta"]["refs"]))
            for r in rows]


# ---------------------------------------------------------------- aggregates

def length_stats(rows):
    lens = [len(r["output"].split()) for r in rows]
    return dict(mean=statistics.fmean(lens), median=statistics.median(lens),
                p90=float(np.percentile(lens, 90)), max=max(lens),
                one_word=sum(v == 1 for v in lens) / len(lens),
                gt15=sum(v > CAPTION_WORDS for v in lens) / len(lens),
                mean_new_tokens=statistics.fmean(r["n_new_tokens"] for r in rows),
                trunc=statistics.fmean(float(r["truncated"]) for r in rows))


def unanswerable_stats(rows):
    eq = sum(norm_answer(r["output"]) == "unanswerable" for r in rows)
    contains = sum("unanswerable" in r["output"].lower() for r in rows)
    n = len(rows)
    return dict(eq=eq / n, contains=contains / n, eq_n=eq, contains_n=contains)


def vizwiz_gt_unanswerable(rows):
    fracs = [statistics.fmean(norm_answer(a) == "unanswerable"
                              for a in r["meta"]["answers"]) for r in rows]
    return dict(majority=statistics.fmean(f >= 0.5 for f in fracs),
                mean_annotator_frac=statistics.fmean(fracs))


def boot_diff_se(a, b, rng):
    """Paired bootstrap SE of mean(a) - mean(b) over shared items."""
    a, b = np.asarray(a), np.asarray(b)
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    return float(np.std(a[idx].mean(axis=1) - b[idx].mean(axis=1)))


# ---------------------------------------------------------------- main

def main():
    rng = np.random.default_rng(RNG_SEED)
    per_row = {}   # (ckpt, task) -> list of per-row dicts
    agg = {}       # (ckpt, task) -> aggregate dict
    for ck in CKPTS:
        for t in TASKS:
            rows = load(ck, t)
            assert len(rows) == 500, (ck, t, len(rows))
            pr = (rows_scienceqa(rows) if t == "scienceqa"
                  else rows_flickr(rows) if t == "flickr" else rows_vqa(rows))
            per_row[ck, t] = pr
            n = len(pr)
            a = dict(length=length_stats(rows), unans=unanswerable_stats(rows))
            a["strict"] = statistics.fmean(r["strict"] for r in pr)
            if t == "flickr":
                a["lenient"] = a["strict"]
            else:
                a["lenient"] = statistics.fmean(r["lenient"] for r in pr)
            if t == "scienceqa":
                fails = sum(r["parse_fail"] for r in pr)
                a["parse_fail_rate"] = fails / n
                a["reported"] = (sum(r["strict"] for r in pr) / max(1, n - fails))
                a["lenient_excl"] = statistics.fmean(r["lenient_excl"] for r in pr)
                a["bare_letter"] = statistics.fmean(r["bare_letter"] for r in pr)
                a["text_rescued_n"] = sum(r["text_rescued"] for r in pr)
            if t in ("textvqa", "vizwiz"):
                a["lenient_ng"] = statistics.fmean(r["lenient_ng"] for r in pr)
            if t == "vizwiz":
                a["gt_unans"] = vizwiz_gt_unanswerable(rows)
            agg[ck, t] = a

    # cross-check strict/reported scores against the pilot's task_*.json
    for ck in CKPTS:
        for t in TASKS:
            ref = json.load(open(RESULTS / ck / f"task_{t}.json"))
            mine = agg[ck, t]["reported" if t == "scienceqa" else "strict"]
            assert abs(mine - ref["score"]) < 5e-4, (ck, t, mine, ref["score"])

    # forgetting matrix (sequential arm), strict vs lenient, paired bootstrap SE
    forgetting = []
    for t in TASKS:
        peak = f"S{STAGE[t]}"
        for k in range(STAGE[t] + 1, 5):
            ck = f"S{k}"
            cell = dict(task=t, peak=peak, ckpt=ck)
            for kind in (("strict", "lenient") if t != "flickr" else ("strict",)):
                a = [r[kind] for r in per_row[peak, t]]
                b = [r[kind] for r in per_row[ck, t]]
                F = statistics.fmean(a) - statistics.fmean(b)
                se = boot_diff_se(a, b, rng)
                cell[kind] = dict(F=F, se=se, sig=bool(abs(F) > 2 * se))
            if t == "flickr":
                cell["lenient"] = cell["strict"]
            forgetting.append(cell)

    out = {
        "aggregates": {f"{ck}/{t}": agg[ck, t] for ck in CKPTS for t in TASKS},
        "forgetting_matrix": forgetting,
        "config": dict(n_boot=N_BOOT, seed=RNG_SEED,
                       caption_words_threshold=CAPTION_WORDS,
                       negation_tokens=sorted(NEGATION_TOKENS)),
    }
    with open(ROOT / "analysis" / "diag" / "format_drift.json", "w") as f:
        json.dump(out, f, indent=1)

    # ------------------------------------------------------------- markdown
    def row(cells):
        print("| " + " | ".join(str(c) for c in cells) + " |")

    print("### 1. Answer length (words) per checkpoint x task")
    row(["ckpt"] + [f"{t} mean/med/p90 (d-mean vs S0)" for t in TASKS])
    row(["---"] * 5)
    for ck in CKPTS:
        cells = [ck]
        for t in TASKS:
            L = agg[ck, t]["length"]
            d = L["mean"] - agg["S0", t]["length"]["mean"]
            cells.append(f"{L['mean']:.2f}/{L['median']:.0f}/{L['p90']:.0f} ({d:+.2f})")
        row(cells)

    print("\n### 1b. Single-word share / >15-word (caption-style) share")
    row(["ckpt"] + [f"{t} 1w% / >15w%" for t in TASKS])
    row(["---"] * 5)
    for ck in CKPTS:
        row([ck] + [f"{agg[ck, t]['length']['one_word']*100:.1f} / "
                    f"{agg[ck, t]['length']['gt15']*100:.1f}" for t in TASKS])

    print("\n### 2. Strict vs lenient scores")
    print("\nScienceQA:")
    row(["ckpt", "reported (parsed-only)", "strict (fail=wrong)", "parse_fail%",
         "lenient (text OK)", "lenient_excl", "bare-letter%", "text-rescued n"])
    row(["---"] * 8)
    for ck in CKPTS:
        a = agg[ck, "scienceqa"]
        row([ck, f"{a['reported']:.4f}", f"{a['strict']:.4f}",
             f"{a['parse_fail_rate']*100:.1f}", f"{a['lenient']:.4f}",
             f"{a['lenient_excl']:.4f}", f"{a['bare_letter']*100:.1f}",
             a["text_rescued_n"]])
    for t in ("textvqa", "vizwiz"):
        print(f"\n{t}:")
        row(["ckpt", "strict", "lenient (containment)", "lenient_negguard",
             "gap G = lenient-strict"])
        row(["---"] * 5)
        for ck in CKPTS:
            a = agg[ck, t]
            row([ck, f"{a['strict']:.4f}", f"{a['lenient']:.4f}",
                 f"{a['lenient_ng']:.4f}", f"{a['lenient']-a['strict']:+.4f}"])
    print("\nflickr (uF1, single scorer):")
    row(["ckpt", "uF1"])
    row(["---"] * 2)
    for ck in CKPTS:
        row([ck, f"{agg[ck, 'flickr']['strict']:.4f}"])

    print("\n### 3. 'Unanswerable' rate per checkpoint on ALL tasks "
          "(eq% / contains%, n=500 each)")
    row(["ckpt"] + TASKS + ["vizwiz GT majority-unans%"])
    row(["---"] * 6)
    for ck in CKPTS:
        cells = [ck]
        for t in TASKS:
            u = agg[ck, t]["unans"]
            cells.append(f"{u['eq']*100:.1f} / {u['contains']*100:.1f}"
                         f" (n={u['contains_n']})")
        cells.append(f"{agg[ck, 'vizwiz']['gt_unans']['majority']*100:.1f}")
        row(cells)

    print("\n### 5. Forgetting matrix F_t(k) = score_t(peak) - score_t(S_k), "
          "strict vs lenient (paired bootstrap SE, * = |F| > 2 SE)")
    row(["task", "peak", "ckpt", "F_strict (SE)", "F_lenient (SE)"])
    row(["---"] * 5)
    for c in forgetting:
        def fmt(d):
            return f"{d['F']:+.4f} ({d['se']:.4f}){'*' if d['sig'] else ''}"
        row([c["task"], c["peak"], c["ckpt"], fmt(c["strict"]), fmt(c["lenient"])])

    print("\nWrote analysis/diag/format_drift.json")


if __name__ == "__main__":
    main()
