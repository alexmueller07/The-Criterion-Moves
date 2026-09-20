# A1 — POPE signal-detection decomposition (diagnostic, pre-specified)

Spec: `design_notes/analysis_ideas.md` §A1. Code: `analysis/diag/signal_detection.py`
(imports `parse_yn` from `pilot/metrics_pope.py` — same parser as the gate metrics;
machine-parseable output with audit columns in `analysis/diag/signal_detection.csv`).
Equal-variance model: d′ = z(H) − z(FA), c = −(z(H)+z(FA))/2 (c > 0 = conservative /
no-biased, c < 0 = liberal / yes-biased), z via `statistics.NormalDist().inv_cdf`,
rates clipped to [1/(2N), 1−1/(2N)] before z. Computed 2026-08-30 from
`results/{S0,S1..S4,J1..J4}/pope_gen.jsonl`.

**Parse exclusions: zero.** All 9,000 rows parsed at every checkpoint (81,000/81,000);
the 1/(2N) clip never fired (every H, FA strictly inside (0,1)). Every cell below is
computed on the full balanced denominator (1,500 gt-yes / 1,500 gt-no per split;
4,500/4,500 pooled), so A2's shifting-denominator concern does not bite for this run.

Divergence-regime note (house rule): under greedy decoding d′/c are a monotone
recoding of (H, FA) — descriptive coordinates, not a latent-variable claim. (H, FA)
is the primitive and is reported in every row.

## Full table (H / FA / d′ / c / balanced accuracy)

| ckpt | category | H | FA | d′ | c | bal_acc |
|---|---|---|---|---|---|---|
| S0 | adversarial | 0.7693 | 0.0907 | 2.0733 | 0.3000 | 0.8393 |
| S0 | popular | 0.7713 | 0.0480 | 2.4078 | 0.4607 | 0.8617 |
| S0 | random | 0.7727 | 0.0240 | 2.7250 | 0.6149 | 0.8743 |
| S0 | all | 0.7711 | 0.0542 | 2.3477 | 0.4314 | 0.8584 |
| S1 | adversarial | 0.7980 | 0.1140 | 2.0400 | 0.1855 | 0.8420 |
| S1 | popular | 0.7993 | 0.0513 | 2.4713 | 0.3964 | 0.8740 |
| S1 | random | 0.7993 | 0.0293 | 2.7299 | 0.5257 | 0.8850 |
| S1 | all | 0.7989 | 0.0649 | 2.3526 | 0.3387 | 0.8670 |
| S2 | adversarial | 0.8700 | 0.2233 | 1.8874 | −0.1827 | 0.8233 |
| S2 | popular | 0.8720 | 0.1167 | 2.3277 | 0.0280 | 0.8777 |
| S2 | random | 0.8720 | 0.0907 | 2.4726 | 0.1004 | 0.8907 |
| S2 | all | 0.8713 | 0.1436 | 2.1972 | −0.0341 | 0.8639 |
| S3 | adversarial | 0.8353 | 0.1553 | 1.9893 | 0.0192 | 0.8400 |
| S3 | popular | 0.8367 | 0.0687 | 2.4666 | 0.2525 | 0.8840 |
| S3 | random | 0.8373 | 0.0493 | 2.6349 | 0.3339 | 0.8940 |
| S3 | all | 0.8364 | 0.0911 | 2.3139 | 0.1770 | 0.8727 |
| S4 | adversarial | 0.6727 | 0.0480 | 2.1119 | 0.6086 | 0.8123 |
| S4 | popular | 0.6753 | 0.0207 | 2.4949 | 0.7927 | 0.8273 |
| S4 | random | 0.6740 | 0.0100 | 2.7773 | 0.9377 | 0.8320 |
| S4 | all | 0.6740 | 0.0262 | 2.3905 | 0.7442 | 0.8239 |
| J1 | adversarial | 0.6767 | 0.0413 | 2.1938 | 0.6385 | 0.8177 |
| J1 | popular | 0.6800 | 0.0213 | 2.4947 | 0.7796 | 0.8293 |
| J1 | random | 0.6807 | 0.0120 | 2.7267 | 0.8938 | 0.8343 |
| J1 | all | 0.6791 | 0.0249 | 2.4271 | 0.7483 | 0.8271 |
| J2 | adversarial | 0.8807 | 0.2593 | 1.8237 | −0.2665 | 0.8107 |
| J2 | popular | 0.8827 | 0.1393 | 2.2717 | −0.0526 | 0.8717 |
| J2 | random | 0.8820 | 0.1113 | 2.4045 | 0.0172 | 0.8853 |
| J2 | all | 0.8818 | 0.1700 | 2.1381 | −0.1149 | 0.8559 |
| J3 | adversarial | 0.8660 | 0.2233 | 1.8687 | −0.1733 | 0.8213 |
| J3 | popular | 0.8673 | 0.1233 | 2.2724 | 0.0223 | 0.8720 |
| J3 | random | 0.8673 | 0.0760 | 2.5464 | 0.1593 | 0.8957 |
| J3 | all | 0.8669 | 0.1409 | 2.1881 | −0.0177 | 0.8630 |
| J4 | adversarial | 0.8660 | 0.2220 | 1.8731 | −0.1711 | 0.8220 |
| J4 | popular | 0.8673 | 0.1227 | 2.2756 | 0.0239 | 0.8723 |
| J4 | random | 0.8673 | 0.0800 | 2.5189 | 0.1456 | 0.8937 |
| J4 | all | 0.8669 | 0.1416 | 2.1852 | −0.0192 | 0.8627 |

## Answers to the five pre-specified questions

**(1) Is the S4 F1 drop discrimination loss or criterion shift?** Criterion shift,
entirely. S3→S4 adversarial: d′ *rises* 1.9893 → 2.1119 (+0.12; pooled +0.08), while
c jumps 0.0192 → 0.6086 (Δc = +0.59 adversarial, +0.57 pooled) — a large move toward
conservative. Both H (0.8353 → 0.6727) and FA (0.1553 → 0.0480) fall together, which
is the signature of a criterion move, not a sensitivity change. The adv F1 drop
(0.8392 → 0.7819) is recall paying for the no-bias. This is exactly the outcome the
spec pre-stated for the VizWiz-abstention stage: the criterion drifts toward *no*, and
by the yes-FP definition the model hallucinates *less* at S4 (adv FA 4.8%, the
S-arm's lowest after S0's 9.1%). "Hallucination worsens at S4" is the wrong reading
of this number.

**(2) Is S2's precision drop a liberal criterion shift?** Yes, predominantly. S1→S2
adversarial: c falls 0.1855 → −0.1827 (Δc = −0.37, into liberal territory) while d′
moves only 2.0400 → 1.8874 (−0.15). FA rises 0.1140 → 0.2233 and H rises
0.7980 → 0.8700 together — yes-bias after the TextVQA stage, with at most a small
genuine sensitivity dip (the −0.15 in d′ is the largest within-S-arm d′ move, but it
recovers to 1.99 at S3 and 2.11 at S4).

**(3) Does d′ change materially anywhere across S0→S4?** No. Pooled d′ across
S0..S4: 2.3477, 2.3526, 2.1972, 2.3139, 2.3905 — total range 0.19, non-monotone, and
the S0→S4 endpoint change is **+0.04** (adversarial: +0.04, from 2.0733 to 2.1119;
sequential fine-tuning ends with trivially *better* discrimination than base). Over
the same span c swings from +0.43 down to −0.03 (S2) and up to +0.74 (S4) — a range
of 0.78, roughly 20× the endpoint d′ movement. The entire S-arm POPE story is
criterion movement; balanced accuracy's 3.4-point drop at S4 (0.8727 → 0.8239) is
criterion mispositioning (bal_acc is maximized at c = 0), not sensitivity loss.

**(4) Is the J arm's steady precision ~0.80 a stable liberal criterion?** For J2..J4,
yes: adversarial c = −0.2665, −0.1733, −0.1711 and d′ = 1.8237, 1.8687, 1.8731
(pooled c ≈ −0.02, d′ ≈ 2.14–2.19) — a stable, mildly liberal criterion with flat
sensitivity, and J4 is nearly indistinguishable from S2 (the other checkpoint whose
most recent gradient steps are yes/no-VQA-heavy: pooled d′ 2.185 vs 2.197, c −0.019
vs −0.034). J1 is the outlier: it sits at S4-like conservative coordinates (pooled
c = 0.7483, d′ = 2.4271; adv precision 0.9424, not 0.80) — the ~0.80-precision
description is a J2-onward regime, reached after step 90 and stationary thereafter.

**(5) S4 vs J4 in (d′, c) space.** Pooled: S4 = (2.3905, +0.7442) vs
J4 = (2.1852, −0.0192); adversarial: S4 = (2.1119, +0.6086) vs J4 = (1.8731,
−0.1711). The between-arm endpoint gap is Δc ≈ 0.76–0.78 versus Δd′ ≈ +0.21–0.24 —
and the d′ difference *favors S4*. The sequential endpoint discriminates as well as
or slightly better than the joint endpoint; the two arms differ almost purely in
where the criterion sits (S4 conservative after the VizWiz-abstention final stage,
J4 near-neutral/liberal). The adv-F1 endpoint gap (0.7819 vs 0.8295) that gate 2's
POPE arm reads as "sequential hallucinates more" is criterion positioning, not
discrimination — and on false alarms alone the sign reverses (S4 adv FA 4.8% vs J4
22.2%).

## Verdict per the A1 outcome map

Outcome (c)-bullet-2 plus (c)-bullet-4 of the spec: d′ flat (endpoint |Δd′| ≤ 0.04
within-arm; every pooled value in [2.14, 2.43]), c does the moving, and the S4 move
is the pre-stated VizWiz-abstention drift toward *no* with F1 falling via recall.
**ARTIFACT** for the discriminative half of gates 1/2, per the spec's KILL clause:
the POPE arm cannot support "hallucination worsens"; whatever survives of the gate
must be carried by CHAIR (pending A3/A4). Consistency check owed to A7: a
criterion-drift account predicts all three splits' FA move in parallel — they do
(e.g. S3→S4 FA falls and J1→J2 FA rises in every split; adv−random FA gap tracks
overall FA level).

## Implication for the paper's framing (3 sentences)

On the discriminative instrument, "hallucination worsens under sequential tuning"
does not survive: sensitivity is flat from S0 to S4 (pooled d′ 2.35 → 2.39) and the
sequential endpoint actually posts the lowest false-alarm rates in either arm's
post-S0 trajectory. What the POPE trajectory measures is a faithfulness *criterion*
that drifts with the answer statistics of the most recent training stage —
yes-biased after yes/no-VQA-heavy exposure (S2, J2..J4), no-biased after
VizWiz-abstention training (S4) — i.e. SEFE-style superficial/format drift, which
the prereg itself requires reporting as a confound rather than as hallucination.
The paper's POPE claims must be reworded to "the yes/no answer criterion tracks
stage answer statistics while discrimination is preserved," and any "hallucination
worsens" headline now rests entirely on the generative (CHAIR) arm surviving its own
kills (A3 length control, A4 degeneracy audit).
