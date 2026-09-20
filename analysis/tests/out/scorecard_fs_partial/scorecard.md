# Method scorecard: ucit suite, ref fsL_seq

- results: `/Users/alexandermueller/Research/cl-hallucination-vlm/fullstudy/results_fs`
- base (stage 0): `/Users/alexandermueller/Research/cl-hallucination-vlm/fullstudy/results_fs/llava15_base` 
- suite: ucit (6 stages; orders from inline (ucit_prep.py ORDER_CANONICAL; o3 = Random(41).shuffle))
- length budget k=60 words; paired bootstrap B=500, seed=17; CHAIR helpers: analysis/diag/length_controlled_chair.py
- leakage task: none on this suite (reported ABSENT)

Style: per-cell values and [min, max] across cells; orderings are strata; no pooled cross-seed CIs (design_notes/review_statistics.md sec 2).

## Discovery

| arm | complete cells | incomplete / skipped |
|---|---|---|
| fsL_seq (ref) | o1/s17 | - |
| fsL_anchor | none | - |

## Per-stage metrics

### fsL_seq  cell o1/s17

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.431 | 2.348 | - | 41.3% | ABSENT | 0.1564 | 0.1052 | 108.4 | - | - | parse_fail=0, over_budget=86% |
| 1 | ArxivQA | +0.212 | 2.279 | -0.069 | 45.6% | ABSENT | 0.1613 | 0.1036 | 113.9 | 0.7380 | norm-containment-EM | parse_fail=0, over_budget=88% |
| 2 | CLEVR-Math | +0.195 | 2.282 | -0.066 | 45.9% | ABSENT | 0.1680 | 0.1032 | 124.3 | 0.6540 | norm-containment-EM | parse_fail=0, over_budget=95% |
| 3 | Flickr30k | +0.318 | 2.349 | +0.001 | 43.6% | ABSENT | 0.0757 | 0.0690 | 22.9 | 0.0000 | norm-containment-EM *open-ended floor* | parse_fail=0, over_budget=5% |
| 4 | IconQA | +0.029 | 2.214 | -0.134 | 49.4% | ABSENT | 0.0775 | 0.0669 | 24.1 | 0.6720 | norm-containment-EM | parse_fail=0, over_budget=5% |
| 5 | ImageNet-R | -0.009 | 2.144 | -0.204 | 50.0% | ABSENT | 0.0514 | 0.0448 | 23.1 | 0.8740 | norm-containment-EM | parse_fail=66, over_budget=3% |
| 6 | VizWiz | +0.068 | 2.188 | -0.160 | 48.5% | ABSENT | 0.1015 | 0.0846 | 30.8 | 0.0720 | norm-containment-EM *open-ended floor* | parse_fail=0, over_budget=11% |

## Scorecard per arm (per cell, then [min, max] across cells)

| arm | cell | Sum abs dc | post-settle Sum | endpoint c | max abs d'-base | F1 | endpoint leak | endpoint CHAIR_i@60 | endpoint CHAIR_i | mean plasticity | plasticity per stage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fsL_seq | o1/s17 | 0.764 | 0.545 | +0.068 | 0.204 | ok | ABSENT | 0.0846 | 0.1015 | 0.5017 | 0.738, 0.654, 0.000, 0.672, 0.874, 0.072 |
| fsL_seq | **range** | [0.764, 0.764] n=1 | [0.545, 0.545] n=1 | [+0.068, +0.068] n=1 | [0.204, 0.204] n=1 | 0/1 tripped | ABSENT | [0.0846, 0.0846] n=1 | [0.1015, 0.1015] n=1 | [0.5017, 0.5017] n=1 | - |
| fsL_anchor | **range** | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT | - |

## Versus reference arm fsL_seq (paired per matched cell; arm - ref)

### fsL_anchor vs fsL_seq

no matched (order, seed) cell complete in both arms -> ABSENT

## PASS / FAIL per pre-registered prediction

| arm | P1 Sum abs dc | P2 plasticity | P3 d' falsifier | P4 CHAIR_i@60 | overall |
|---|---|---|---|---|---|
| fsL_seq (ref) | - | - | PASS (1/1 ok) | - | reference |
| fsL_anchor | ABSENT (0/0) | ABSENT (0/0) | ABSENT (0/0 ok) | ABSENT (0/0) | **INCOMPLETE** |

- P1: Sum|dc|(arm) < Sum|dc|(ref) in >= 2/3 of matched cells (c_0 = base)
- P2: mean plasticity >= ref - 0.02 in every matched cell (one-sided)
- P3: max_k |d'_k - d'_base| < 0.30 in every complete cell of the arm (F1 falsifier)
- P4: endpoint CHAIR_i@k <= ref AND paired bootstrap CI95 upper < 0 in >= half of matched cells
- ABSENT = inputs missing (no base cell / no matched cell); never counts as PASS or FAIL. Overall PASS needs all four PASS; any FAIL = FAIL; else INCOMPLETE.

## Pilot-run reference values (context only; NOT used in any verdict)

PILOT-RUN REFERENCES: LLaVA-1.5-7B, pilot suite, order o1 (scienceqa, textvqa, flickr, vizwiz), one seed, one cell per arm; exploratory post-hoc method results from analysis/method_comparison.py (2026-08-31). Context only for full-study scorecards: different backbone/seed/cell, never an input to any PASS/FAIL verdict.

| pilot arm | Sum abs dc | endpoint c | max abs d'-base | endpoint leak | endpoint CHAIR_i@60 | endpoint CHAIR_i | mean plasticity |
|---|---|---|---|---|---|---|---|
| SEQ (plain sequential) | 1.243 | +0.744 | 0.151 | 14.6% | 0.0894 | 0.1273 | 0.6592 |
| anchor-v2 (symmetric decision-axis anchor) | 0.791 | +0.768 | 0.147 | 16.4% | 0.0766 | 0.1091 | 0.6558 |
| JOINT (matched joint control) | 1.278 | -0.019 | 0.210 | 2.2% | 0.0904 | 0.1561 | 0.6339 |
| ER (experience replay, 100 exemplars) | 1.358 | +0.853 | 0.116 | 12.6% | 0.0957 | 0.1482 | 0.6569 |
| anchor-v1 (one-sided anchor (falsified)) | 3.430 | +1.427 | 0.477 | 17.8% | 0.0856 | 0.1258 | 0.6575 |

