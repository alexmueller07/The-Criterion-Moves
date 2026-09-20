"""Diagnostic A3 (design_notes/analysis_ideas.md): length-controlled CHAIR.

Pre-specified 2026-08-30. Scope as directed for the pilot diagnostic run:
  1. Reproduce full CHAIR_i / CHAIR_s per checkpoint from chair_gen.jsonl +
     coco_gt.json and assert they match results/*/chair.json (self-check).
  2. Length-controlled CHAIR: truncate every caption to its first k whitespace
     words, k in {30, 60, 90}, re-extract mentions, score against the same GT.
     (The A3 note phrases budgets in extractor word-tokens; per the task spec
     this run uses whitespace words. The two token streams differ only on
     punctuation/digit splitting, so budgets are comparable.)
  3. Mention-position quartiles: hallucination rate per quartile of relative
     token position within the caption (EOS-literature "late hallucination"
     signature).
  4. Assertion rate: object mentions per 100 generated whitespace words.

Machinery is imported from pilot/metrics_chair.py (extractor + synonym map are
NOT re-vendored). The only new walker (position-tagged extraction) mirrors
extract_object_mentions and is asserted equal to it on every caption.

Stdlib only. Paired bootstrap (images, B=10,000, seed 17) for the two
load-bearing gaps, mirroring the gate's boot_chair_gap convention.

Run:  python3 analysis/diag/length_controlled_chair.py
Writes analysis/diag/length_controlled_chair.json and prints tables.
"""
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pilot"))
from metrics_chair import WORD2CAT, _norm_token, extract_object_mentions  # noqa: E402

CKPTS = ["S0", "S1", "S2", "S3", "S4", "J1", "J2", "J3", "J4", "E1", "E2", "E3", "E4", "F1", "F2", "F3", "F4", "G1", "G2", "G3", "G4"]
KS = [30, 60, 90]
B = 10_000
SEED = 17


def extract_mentions_with_pos(text):
    """Position-tagged twin of metrics_chair.extract_object_mentions.

    Returns ([(category, token_index)], n_tokens) where token_index is the
    index of the mention's first token in re.findall(r"[a-z]+", text.lower()).
    Control flow mirrors the original exactly (incl. the for/else bigram
    retry); equality of the category sequence is asserted by the caller.
    """
    toks = [_norm_token(w) for w in re.findall(r"[a-z]+", text.lower())]
    out = []
    i = 0
    while i < len(toks):
        if i + 1 < len(toks):
            for second in (toks[i + 1], _norm_token(toks[i + 1])):
                bigram = toks[i] + " " + second
                if bigram in WORD2CAT:
                    out.append((WORD2CAT[bigram], i))
                    i += 2
                    break
            else:
                if toks[i] in WORD2CAT:
                    out.append((WORD2CAT[toks[i]], i))
                i += 1
            continue
        if toks[i] in WORD2CAT:
            out.append((WORD2CAT[toks[i]], i))
        i += 1
    return out, len(toks)


def load_gt(path):
    raw = json.load(open(path))
    gt = {}
    for cid, rec in raw.items():
        objs = set(rec["objects"])
        for c in rec["captions"]:
            objs |= set(extract_object_mentions(c))
        gt[cid] = objs
    return gt


def chair_over(rows, gt, k=None):
    """CHAIR over (optionally k-whitespace-word-truncated) captions.

    Returns summary dict + per-image (n_mentions, n_hal) list aligned to rows.
    """
    n_caps = n_hal_caps = n_mentions = n_hal = 0
    n_at_budget = 0
    per_img = []
    for r in rows:
        text = r["output"]
        if k is not None:
            words = text.split()
            if len(words) > k:
                n_at_budget += 1
            text = " ".join(words[:k])
        mentions = extract_object_mentions(text)
        g = gt[str(r["coco_id"])]
        hal = [m for m in mentions if m not in g]
        n_caps += 1
        n_mentions += len(mentions)
        n_hal += len(hal)
        if hal:
            n_hal_caps += 1
        per_img.append((len(mentions), len(hal)))
    summ = {
        "chair_i": n_hal / max(1, n_mentions),
        "chair_s": n_hal_caps / max(1, n_caps),
        "mentions_per_caption": n_mentions / max(1, n_caps),
        "n_mentions": n_mentions,
        "n_hal_mentions": n_hal,
    }
    if k is not None:
        summ["frac_captions_over_budget"] = n_at_budget / max(1, n_caps)
    return summ, per_img


def position_quartiles(rows, gt):
    """Pooled hallucination rate per quartile of relative token position."""
    cnt = [0, 0, 0, 0]
    hal = [0, 0, 0, 0]
    for r in rows:
        tagged, n_tok = extract_mentions_with_pos(r["output"])
        # self-check: position walker must reproduce the vendored extractor
        assert [c for c, _ in tagged] == extract_object_mentions(r["output"]), r["id"]
        if n_tok == 0:
            continue
        g = gt[str(r["coco_id"])]
        for cat, idx in tagged:
            q = min(3, int(4 * idx / n_tok))
            cnt[q] += 1
            if cat not in g:
                hal[q] += 1
    return {
        "n_mentions": cnt,
        "n_hal": hal,
        "hal_rate": [h / max(1, c) for h, c in zip(hal, cnt)],
    }


def boot_gap_ci(per_a, per_b, b=B, seed=SEED):
    """Paired-image bootstrap CI for CHAIR_i(A) - CHAIR_i(B) (ratio of sums)."""
    assert len(per_a) == len(per_b)
    rng = random.Random(seed)
    n = len(per_a)
    gaps = []
    for _ in range(b):
        idx = [rng.randrange(n) for _ in range(n)]
        ma = ha = mb = hb = 0
        for i in idx:
            ma += per_a[i][0]
            ha += per_a[i][1]
            mb += per_b[i][0]
            hb += per_b[i][1]
        gaps.append(ha / max(1, ma) - hb / max(1, mb))
    gaps.sort()
    return gaps[int(0.025 * b)], gaps[int(0.975 * b)]


def main():
    gt = load_gt(ROOT / "results" / "coco_gt.json")
    out = {"ks": KS, "checkpoints": {}}
    per_img_store = {}  # (ckpt, k) -> aligned per-image lists
    row_order = None

    for c in CKPTS:
        rows = [json.loads(l) for l in open(ROOT / "results" / c / "chair_gen.jsonl")]
        ids = [r["coco_id"] for r in rows]
        if row_order is None:
            row_order = ids
        assert ids == row_order, f"{c}: image order differs; pairing would break"

        # (1) full CHAIR, asserted against the stored summary
        full, per_full = chair_over(rows, gt, k=None)
        stored = json.load(open(ROOT / "results" / c / "chair.json"))
        assert round(full["chair_i"], 4) == stored["chair_i"], (c, full["chair_i"], stored["chair_i"])
        assert round(full["chair_s"], 4) == stored["chair_s"], (c, full["chair_s"], stored["chair_s"])
        assert round(full["mentions_per_caption"], 3) == stored["mentions_per_caption"], c
        per_img_store[(c, None)] = per_full

        # (2) length-controlled
        at_k = {}
        for k in KS:
            summ, per_k = chair_over(rows, gt, k=k)
            at_k[k] = summ
            per_img_store[(c, k)] = per_k

        # (3) position quartiles
        quart = position_quartiles(rows, gt)

        # (4) assertion rate + length stats
        wlens = sorted(len(r["output"].split()) for r in rows)
        n_words = sum(wlens)
        out["checkpoints"][c] = {
            "full": full,
            "at_k": at_k,
            "position_quartiles": quart,
            "mean_words": n_words / len(rows),
            "median_words": wlens[len(wlens) // 2],
            "mean_new_tokens": stored["mean_new_tokens"],
            "mentions_per_100_words": 100 * full["n_mentions"] / max(1, n_words),
        }
        print(f"{c}: reproduced chair_i={stored['chair_i']} chair_s={stored['chair_s']} OK")

    # bootstrap the two load-bearing gaps, full and k=60
    out["bootstrap"] = {}
    for name, a, bnm in [("S4_minus_J4", "S4", "J4"), ("S3_minus_S1", "S3", "S1")]:
        for k in [None, 60]:
            lo, hi = boot_gap_ci(per_img_store[(a, k)], per_img_store[(bnm, k)])
            pa = per_img_store[(a, k)]
            pb = per_img_store[(bnm, k)]
            point = (sum(h for _, h in pa) / max(1, sum(m for m, _ in pa))
                     - sum(h for _, h in pb) / max(1, sum(m for m, _ in pb)))
            out["bootstrap"][f"{name}@{'full' if k is None else k}"] = {
                "gap": point, "ci95": [lo, hi]}

    dst = Path(__file__).resolve().parent / "length_controlled_chair.json"
    json.dump(out, open(dst, "w"), indent=1)
    print(f"\nwrote {dst}\n")

    # ---- printed tables ----
    def f(x, nd=4):
        return f"{x:.{nd}f}"

    print("CHAIR_i: full and length-controlled (first k whitespace words)")
    print(f"{'ckpt':<5}{'full':>8}{'k=30':>8}{'k=60':>8}{'k=90':>8}"
          f"{'meanW':>8}{'medW':>7}{'m/100w':>8}")
    for c in CKPTS:
        d = out["checkpoints"][c]
        print(f"{c:<5}{f(d['full']['chair_i']):>8}"
              + "".join(f"{f(d['at_k'][k]['chair_i']):>8}" for k in KS)
              + f"{d['mean_words']:>8.1f}{d['median_words']:>7}"
              + f"{d['mentions_per_100_words']:>8.2f}")

    print("\nCHAIR_s: full and length-controlled")
    print(f"{'ckpt':<5}{'full':>8}{'k=30':>8}{'k=60':>8}{'k=90':>8}"
          f"{'%cap>60w':>10}")
    for c in CKPTS:
        d = out["checkpoints"][c]
        print(f"{c:<5}{f(d['full']['chair_s']):>8}"
              + "".join(f"{f(d['at_k'][k]['chair_s']):>8}" for k in KS)
              + f"{100*d['at_k'][60]['frac_captions_over_budget']:>10.1f}")

    print("\nHallucination rate by quartile of relative mention position (pooled)")
    print(f"{'ckpt':<5}{'Q1':>8}{'Q2':>8}{'Q3':>8}{'Q4':>8}{'Q4/Q1':>8}{'Q4 share':>9}")
    for c in CKPTS:
        q = out["checkpoints"][c]["position_quartiles"]
        r = q["hal_rate"]
        tot_h = sum(q["n_hal"])
        print(f"{c:<5}" + "".join(f"{f(x):>8}" for x in r)
              + f"{r[3]/max(1e-9, r[0]):>8.2f}"
              + f"{q['n_hal'][3]/max(1, tot_h):>9.3f}")

    print("\nPaired bootstrap (images, B=10000, seed 17), CHAIR_i gaps:")
    for k, v in out["bootstrap"].items():
        print(f"  {k:<22} gap={v['gap']:+.4f}  95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
