#!/usr/bin/env python3
"""A1 diagnostic: equal-variance signal-detection decomposition of POPE trajectories.

Pre-specified in design_notes/analysis_ideas.md section A1. Per checkpoint x
category (adversarial/popular/random, plus pooled "all"), over parsed rows only:

    H  = P(pred=yes | gt=yes)          (hit rate)
    FA = P(pred=yes | gt=no)           (false-alarm rate)
    d' = z(H) - z(FA)                  (sensitivity)
    c  = -0.5 * (z(H) + z(FA))         (criterion; + = conservative/no-biased,
                                        - = liberal/yes-biased)
    bal_acc = (H + 1 - FA) / 2

Rates are clipped to [1/(2N), 1 - 1/(2N)] before the z-transform (standard
correction for 0/1 rates), with N the number of parsed signal (gt=yes) or
noise (gt=no) trials respectively. Raw unclipped H/FA are what is reported;
clipping only ever affects the z-transform and a `clipped` audit flag records
when it fired. Parse handling: rows where metrics_pope.parse_yn returns None
are excluded and counted (never coerced) — exclusion counts are audit columns.

Divergence regime (name-the-divergence rule, carried from A1b): d' assumes an
equal-variance Gaussian latent; under greedy decoding it is a monotone
recoding of the (H, FA) pair. Treat d'/c as descriptive coordinates; (H, FA)
is the primitive and is always reported alongside.

Deterministic, Python 3.9 stdlib only. Run from anywhere:

    python3 analysis/diag/signal_detection.py

Reads  results/{S0,S1..S4,J1..J4}/pope_gen.jsonl
Writes analysis/diag/signal_detection.csv (machine-parseable, audit columns)
Prints the full table plus the pairwise contrasts quoted in
analysis/diag_signal_detection.md.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pilot"))
from metrics_pope import parse_yn  # noqa: E402  single source of truth for parsing

CKPTS = ["S0", "S1", "S2", "S3", "S4", "J1", "J2", "J3", "J4"]
CATS = ["adversarial", "popular", "random", "all"]
_z = NormalDist().inv_cdf


def clipped_rate(k, n):
    """Rate k/n clipped to [1/(2n), 1-1/(2n)]. Returns (rate_for_z, fired)."""
    raw = k / n
    lo = 1.0 / (2.0 * n)
    hi = 1.0 - lo
    if raw < lo:
        return lo, True
    if raw > hi:
        return hi, True
    return raw, False


def score_checkpoint(ckpt):
    """Return {category: audit-dict} for one checkpoint's pope_gen.jsonl."""
    counts = defaultdict(lambda: {"n": 0, "parse_fail": 0,
                                  "n_gt_yes": 0, "n_gt_no": 0,
                                  "hits": 0, "fas": 0})
    path = ROOT / "results" / ckpt / "pope_gen.jsonl"
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            cat = r.get("category", "unknown")
            pred = parse_yn(r["output"])
            gt = r["gt"]
            for c in (cat, "all"):
                b = counts[c]
                b["n"] += 1
                if pred is None:
                    b["parse_fail"] += 1
                    continue
                if gt == "yes":
                    b["n_gt_yes"] += 1
                    if pred == "yes":
                        b["hits"] += 1
                else:
                    b["n_gt_no"] += 1
                    if pred == "yes":
                        b["fas"] += 1

    out = {}
    for cat, b in counts.items():
        H_raw = b["hits"] / b["n_gt_yes"]
        FA_raw = b["fas"] / b["n_gt_no"]
        H_z, h_clip = clipped_rate(b["hits"], b["n_gt_yes"])
        FA_z, fa_clip = clipped_rate(b["fas"], b["n_gt_no"])
        zH, zFA = _z(H_z), _z(FA_z)
        out[cat] = {
            "ckpt": ckpt, "category": cat,
            "n": b["n"], "parse_fail": b["parse_fail"],
            "n_parsed": b["n"] - b["parse_fail"],
            "n_gt_yes": b["n_gt_yes"], "n_gt_no": b["n_gt_no"],
            "hits": b["hits"], "false_alarms": b["fas"],
            "H": round(H_raw, 4), "FA": round(FA_raw, 4),
            "dprime": round(zH - zFA, 4),
            "c": round(-0.5 * (zH + zFA), 4),
            "bal_acc": round((H_raw + 1.0 - FA_raw) / 2.0, 4),
            "clipped": ("H" if h_clip else "") + ("FA" if fa_clip else "") or "none",
        }
    return out


def main():
    table = {ck: score_checkpoint(ck) for ck in CKPTS}

    fields = ["ckpt", "category", "n", "parse_fail", "n_parsed",
              "n_gt_yes", "n_gt_no", "hits", "false_alarms",
              "H", "FA", "dprime", "c", "bal_acc", "clipped"]
    out_csv = ROOT / "analysis" / "diag" / "signal_detection.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ck in CKPTS:
            for cat in CATS:
                w.writerow(table[ck][cat])
    print(f"wrote {out_csv}\n")

    hdr = ("ckpt cat          n   pfail    H      FA     d'      c     balacc clip")
    print(hdr)
    for ck in CKPTS:
        for cat in CATS:
            r = table[ck][cat]
            print(f"{r['ckpt']:<4} {r['category']:<11} {r['n']:>5} "
                  f"{r['parse_fail']:>5}  {r['H']:.4f} {r['FA']:.4f} "
                  f"{r['dprime']:>7.4f} {r['c']:>7.4f} {r['bal_acc']:.4f} "
                  f"{r['clipped']}")
        print()

    def delta(a, b, cat, key):
        return round(table[b][cat][key] - table[a][cat][key], 4)

    print("Pairwise contrasts (delta = second minus first):")
    for cat in ("adversarial", "all"):
        print(f"  [{cat}]")
        for a, b, label in [("S3", "S4", "S3->S4 (the F1 drop)"),
                            ("S1", "S2", "S1->S2 (the precision drop)"),
                            ("S0", "S4", "S0->S4 (endpoint, SEQ)"),
                            ("J1", "J4", "J1->J4 (endpoint, JOINT)"),
                            ("J4", "S4", "J4 vs S4 (between-arm endpoint)")]:
            print(f"    {label:<32} dd'={delta(a, b, cat, 'dprime'):+.4f}  "
                  f"dc={delta(a, b, cat, 'c'):+.4f}  "
                  f"dH={delta(a, b, cat, 'H'):+.4f}  "
                  f"dFA={delta(a, b, cat, 'FA'):+.4f}")
    print("\nRange of d' across checkpoints (pooled 'all'): "
          f"min={min(table[ck]['all']['dprime'] for ck in CKPTS):.4f} "
          f"max={max(table[ck]['all']['dprime'] for ck in CKPTS):.4f}")
    print("Range of c  across checkpoints (pooled 'all'): "
          f"min={min(table[ck]['all']['c'] for ck in CKPTS):.4f} "
          f"max={max(table[ck]['all']['c'] for ck in CKPTS):.4f}")


if __name__ == "__main__":
    main()
