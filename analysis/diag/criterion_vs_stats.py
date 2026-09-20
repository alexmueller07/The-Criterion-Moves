"""Diagnostic A11-extended: criterion drift vs training answer statistics.

Pre-specified in design_notes/analysis_ideas.md (A11, extended with the A1
hit/false-alarm primitives). Diagnostic only — cannot rescue a gate fail.

Computes, per checkpoint C in {S0, S1..S4, J1..J4} from results/C/:
  1. POPE pooled yes-rate, hit rate H = P(yes|gt=yes), false-alarm rate
     FA = P(yes|gt=no) over parsed rows (parse_yn imported from
     pilot/metrics_pope.py — same parser as the gate); descriptive d'/c.
  2. Per-stage Delta yes-rate (S_k - S_{k-1}) paired with that stage's
     TRAINING-SET answer statistics (verified at source on the cluster from
     the actual train.jsonl files, 2026-08-30 — treated as ground truth here).
  3. JOINT-arm internal trajectory J1..J4 (yes-rate, adv-F1, CHAIR_i),
     flatness after J2, J1 anomaly quantification, J1-vs-S4 conservatism
     signature in (H, FA) space; auxiliary VizWiz abstention-output rate.
  4. Tuning-volume vs sequencing decomposition: S4-S0 vs J4-S0 vs S4-J4.
  5. Dose-response table (stage answer statistics vs behavioral deltas).

Stdlib only. Outputs: markdown tables on stdout + machine-parseable JSON at
analysis/diag/criterion_vs_stats.json (deliverable-integrity convention:
audit columns n_rows / n_parsed / parse_fail carried on every rate).

n = 1 seed, 1 task order: all deltas are descriptive; with n=4 stages no
correlation coefficient is reported (pattern description only).
"""
import json
import sys
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pilot"))
from metrics_pope import parse_yn  # noqa: E402  (the gate's own parser)

RESULTS = ROOT / "results"
CKPTS = ["S0", "S1", "S2", "S3", "S4", "J1", "J2", "J3", "J4"]
SEQ = ["S0", "S1", "S2", "S3", "S4"]
JNT = ["J1", "J2", "J3", "J4"]

# Optimizer steps for the four checkpoints of each arm (stage boundaries for
# SEQ; identical steps for JOINT per sb_train_joint.sbatch --checkpoint_steps
# 90,215,340 + final). Final step 399 per analysis_ideas.md A11.
STEPS = {1: 90, 2: 215, 3: 340, 4: 399}

# VERIFIED training-set answer statistics, computed at source from the actual
# train.jsonl files on the cluster (ground-truth inputs; do not recompute here
# — the train files are not on this machine).
#   yes/no/una = fraction of training targets that are exactly yes / no /
#   unanswerable; mean_words = mean target length in words.
TRAIN_STATS = {
    "S1": {"task": "scienceqa", "n": 5700, "yes": 0.000, "no": 0.000,
           "una": 0.000, "mean_words": 1.00},
    "S2": {"task": "textvqa", "n": 8000, "yes": 0.051, "no": 0.015,
           "una": 0.022, "mean_words": 1.56},
    "S3": {"task": "flickr", "n": 8000, "yes": 0.000, "no": 0.000,
           "una": 0.000, "mean_words": 17.93},
    "S4": {"task": "vizwiz", "n": 3800, "yes": 0.024, "no": 0.028,
           "una": 0.432, "mean_words": 1.41},
}

Z = NormalDist().inv_cdf


def pope_stats(ckpt):
    """Pooled + adversarial signal-detection stats from pope_gen.jsonl."""
    n = parsed = fail = 0
    tp = fp = tn = fn = 0          # pooled (all categories)
    a_tp = a_fp = a_tn = a_fn = 0  # adversarial split
    a_fail = a_n = 0
    for line in open(RESULTS / ckpt / "pope_gen.jsonl"):
        r = json.loads(line)
        n += 1
        adv = r.get("category") == "adversarial"
        if adv:
            a_n += 1
        pred = parse_yn(r["output"])
        if pred is None:
            fail += 1
            if adv:
                a_fail += 1
            continue
        parsed += 1
        gt = r["gt"]
        if pred == "yes" and gt == "yes":
            tp += 1
            if adv:
                a_tp += 1
        elif pred == "yes" and gt == "no":
            fp += 1
            if adv:
                a_fp += 1
        elif pred == "no" and gt == "no":
            tn += 1
            if adv:
                a_tn += 1
        else:
            fn += 1
            if adv:
                a_fn += 1

    n_yes_gt = tp + fn
    n_no_gt = fp + tn
    hit = tp / n_yes_gt
    fa = fp / n_no_gt
    # A1 clipping for the z-transform only (descriptive; greedy decoding).
    h_c = min(max(hit, 1 / (2 * n_yes_gt)), 1 - 1 / (2 * n_yes_gt))
    f_c = min(max(fa, 1 / (2 * n_no_gt)), 1 - 1 / (2 * n_no_gt))
    prec = a_tp / max(1, a_tp + a_fp)
    rec = a_tp / max(1, a_tp + a_fn)
    return {
        "ckpt": ckpt,
        "n_rows": n, "n_parsed": parsed, "parse_fail": fail,
        "yes_rate": (tp + fp) / parsed,
        "hit": hit, "fa": fa,
        "dprime": Z(h_c) - Z(f_c),
        "criterion_c": -(Z(h_c) + Z(f_c)) / 2,
        "balanced_acc": (hit + 1 - fa) / 2,
        "adv_f1": 2 * prec * rec / max(1e-9, prec + rec),
        "adv_n": a_n, "adv_parse_fail": a_fail,
    }


def chair_stats(ckpt):
    ch = json.load(open(RESULTS / ckpt / "chair.json"))
    return {"chair_i": ch["chair_i"], "chair_s": ch["chair_s"],
            "mentions_per_caption": ch["mentions_per_caption"],
            "chair_mean_new_tokens": ch["mean_new_tokens"],
            "chair_truncation_rate": ch["truncation_rate"]}


def vizwiz_abstain_rate(ckpt):
    """Auxiliary (J1-anomaly hypothesis check): fraction of VizWiz val
    outputs containing 'unanswerable' (case-insensitive substring)."""
    n = una = 0
    for line in open(RESULTS / ckpt / "vizwiz_gen.jsonl"):
        r = json.loads(line)
        n += 1
        if "unanswerable" in r["output"].lower():
            una += 1
    return {"vizwiz_n": n, "vizwiz_abstain_rate": una / n}


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def table(headers, rows, title):
    print(f"\n### {title}\n")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        print("| " + " | ".join(fmt(v) for v in row) + " |")


def main():
    per = {}
    for c in CKPTS:
        per[c] = pope_stats(c)
        per[c].update(chair_stats(c))
        per[c].update(vizwiz_abstain_rate(c))

    # ---- (1) per-checkpoint table -------------------------------------
    table(
        ["ckpt", "n", "parse_fail", "yes_rate", "hit", "FA", "d'", "c",
         "adv_F1", "CHAIR_i", "CHAIR_s"],
        [[per[c]["ckpt"], per[c]["n_rows"], per[c]["parse_fail"],
          per[c]["yes_rate"], per[c]["hit"], per[c]["fa"], per[c]["dprime"],
          per[c]["criterion_c"], per[c]["adv_f1"], per[c]["chair_i"],
          per[c]["chair_s"]] for c in CKPTS],
        "Per-checkpoint POPE signal-detection + CHAIR (pooled, parsed rows)")

    # ---- (2)+(5) dose-response: stage stats vs behavioral deltas ------
    dose = []
    for k in range(1, 5):
        cur, prev = SEQ[k], SEQ[k - 1]
        ts = TRAIN_STATS[cur]
        d = {
            "stage": cur, "task": ts["task"], "n_train": ts["n"],
            "yes_pct": ts["yes"], "no_pct": ts["no"],
            "net_yes": ts["yes"] - ts["no"], "una_pct": ts["una"],
            "mean_words": ts["mean_words"],
            "d_yes_rate": per[cur]["yes_rate"] - per[prev]["yes_rate"],
            "d_fa": per[cur]["fa"] - per[prev]["fa"],
            "d_hit": per[cur]["hit"] - per[prev]["hit"],
            "d_adv_f1": per[cur]["adv_f1"] - per[prev]["adv_f1"],
            "d_chair_i": per[cur]["chair_i"] - per[prev]["chair_i"],
        }
        dose.append(d)
    table(
        ["stage", "task", "yes%", "no%", "yes-no", "una%", "mean_words",
         "dYes", "dFA", "dHit", "dAdvF1", "dCHAIR_i"],
        [[d["stage"], d["task"], d["yes_pct"], d["no_pct"], d["net_yes"],
          d["una_pct"], d["mean_words"], d["d_yes_rate"], d["d_fa"],
          d["d_hit"], d["d_adv_f1"], d["d_chair_i"]] for d in dose],
        "Answer-statistics dose-response (stage train stats vs Delta on "
        "S_k - S_{k-1}); n=4 stages — pattern only, no correlation")

    # ---- (3) JOINT internal trajectory --------------------------------
    table(
        ["ckpt", "step", "frac_train", "yes_rate", "hit", "FA", "adv_F1",
         "CHAIR_i", "vizwiz_abstain"],
        [[c, STEPS[k + 1], STEPS[k + 1] / STEPS[4], per[c]["yes_rate"],
          per[c]["hit"], per[c]["fa"], per[c]["adv_f1"], per[c]["chair_i"],
          per[c]["vizwiz_abstain_rate"]] for k, c in enumerate(JNT)],
        "JOINT-arm internal trajectory (identical optimizer steps to SEQ)")

    j234 = {m: sum(per[c][m] for c in JNT[1:]) / 3
            for m in ("yes_rate", "hit", "fa", "adv_f1", "chair_i")}
    flat = {m: max(abs(per[c][m] - j234[m]) for c in JNT[1:])
            for m in j234}
    j1dev = {m: per["J1"][m] - j234[m] for m in j234}
    table(
        ["metric", "J1", "mean(J2..J4)", "J1 - mean(J2..J4)",
         "max |J2..J4 dev|"],
        [[m, per["J1"][m], j234[m], j1dev[m], flat[m]] for m in j234],
        "J1 anomaly vs post-J2 flatness (J2..J4 spread = within-arm "
        "stability scale)")

    table(
        ["metric", "J1", "S4", "|J1 - S4|", "S0 (base)"],
        [[m, per["J1"][m], per["S4"][m], abs(per["J1"][m] - per["S4"][m]),
          per["S0"][m]]
         for m in ("yes_rate", "hit", "fa", "criterion_c", "dprime",
                   "adv_f1", "vizwiz_abstain_rate")],
        "Conservatism signature: J1 vs S4 in (H, FA) space")

    # ---- (4) volume-vs-sequencing decomposition -----------------------
    decomp = []
    for m in ("yes_rate", "hit", "fa", "criterion_c", "dprime", "adv_f1",
              "chair_i", "chair_s"):
        total = per["S4"][m] - per["S0"][m]
        vol = per["J4"][m] - per["S0"][m]      # shared-with-joint (volume)
        seqspec = per["S4"][m] - per["J4"][m]  # sequential-specific
        same_sign = (total != 0) and (vol * total > 0) and (seqspec * total > 0)
        decomp.append({
            "metric": m, "S4_minus_S0": total, "J4_minus_S0": vol,
            "S4_minus_J4": seqspec,
            "vol_share": vol / total if same_sign else None,
            "note": "" if same_sign else "components oppose; shares undefined",
        })
    table(
        ["metric", "S4-S0 (total)", "J4-S0 (volume)", "S4-J4 (seq-specific)",
         "volume share", "note"],
        [[d["metric"], d["S4_minus_S0"], d["J4_minus_S0"], d["S4_minus_J4"],
          "-" if d["vol_share"] is None else fmt(d["vol_share"], 2),
          d["note"]] for d in decomp],
        "Tuning-volume vs sequencing decomposition "
        "(S4-S0 = (J4-S0) + (S4-J4); shares only reported when both "
        "components share the total's sign)")

    # Approximate eval-noise scale (binomial, ignoring clustering) so the
    # deltas above can be read against measurement noise.
    n_p = per["S0"]["n_parsed"]
    se_yes = (0.5 * 0.5 / n_p) ** 0.5
    n_mentions = per["S0"]["mentions_per_caption"] * 500
    p = per["S0"]["chair_i"]
    se_chair = (p * (1 - p) / n_mentions) ** 0.5
    print(f"\nApprox eval-noise scale (descriptive): SE(yes_rate) ~ "
          f"{se_yes:.4f} (n={n_p}); SE(CHAIR_i) ~ {se_chair:.4f} "
          f"(~{n_mentions:.0f} mentions, clustering ignored). "
          f"n=1 seed: trajectory-level noise floor unknown (prereg lim. 1).")

    out = {
        "per_checkpoint": [per[c] for c in CKPTS],
        "train_stats_verified_at_source": TRAIN_STATS,
        "dose_response": dose,
        "joint_flatness_j2_j4_maxdev": flat,
        "j1_minus_mean_j2_j4": j1dev,
        "decomposition": decomp,
        "steps": STEPS,
        "se_approx": {"yes_rate": se_yes, "chair_i": se_chair},
    }
    out_path = Path(__file__).resolve().parent / "criterion_vs_stats.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
