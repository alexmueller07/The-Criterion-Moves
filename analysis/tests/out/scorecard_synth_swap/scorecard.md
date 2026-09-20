# Method scorecard: pilot suite, ref mth_critp

- results: `/Users/alexandermueller/Research/cl-hallucination-vlm/analysis/tests/fixtures/results_scorecard_synth`
- base (stage 0): `/Users/alexandermueller/Research/cl-hallucination-vlm/analysis/tests/fixtures/results_scorecard_synth/qwen25vl_base` (EVAL_DONE)
- suite: pilot (4 stages; orders from /Users/alexandermueller/Research/cl-hallucination-vlm/fullstudy/pilot_manifest.json)
- length budget k=60 words; paired bootstrap B=2000, seed=17; CHAIR helpers: analysis/diag/length_controlled_chair.py
- leakage task: textvqa

Style: per-cell values and [min, max] across cells; orderings are strata; no pooled cross-seed CIs (design_notes/review_statistics.md sec 2).

## Discovery

| arm | complete cells | incomplete / skipped |
|---|---|---|
| mth_critp (ref) | o1/s17, o1/s23 | o2/s17: missing EVAL_DONE stages [4]; IN-PROGRESS: mth_critp_o2_s17_k4 lacks EVAL_DONE |
| psQ_seq | o1/s17, o1/s23, o2/s31 | - |

## Per-stage metrics

### mth_critp  cell o1/s17

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | scienceqa | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0476 | 0.0476 | 90.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |
| 2 | textvqa | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0476 | 0.0476 | 90.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 3 | flickr | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0476 | 0.0476 | 90.0 | 0.4800 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=0% |
| 4 | vizwiz | +0.284 | 2.113 | -0.010 | 43.5% | 4.0% | 0.0476 | 0.0476 | 90.0 | 0.6800 | vqa_acc | parse_fail=0, over_budget=0% |

### mth_critp  cell o1/s23

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | scienceqa | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0476 | 0.0476 | 90.0 | 0.8400 | mc_acc | parse_fail=0, over_budget=0% |
| 2 | textvqa | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0476 | 0.0476 | 90.0 | 0.6400 | vqa_acc | parse_fail=0, over_budget=0% |
| 3 | flickr | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0476 | 0.0476 | 90.0 | 0.5200 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=0% |
| 4 | vizwiz | +0.284 | 2.113 | -0.010 | 43.5% | 4.0% | 0.0476 | 0.0476 | 90.0 | 0.7200 | vqa_acc | parse_fail=0, over_budget=0% |

### psQ_seq  cell o1/s17

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | scienceqa | -0.000 | 2.073 | -0.050 | 50.0% | 0.0% | 0.2000 | 0.2000 | 101.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |
| 2 | textvqa | +0.696 | 1.898 | -0.225 | 32.5% | 0.0% | 0.2000 | 0.2000 | 102.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 3 | flickr | -0.000 | 2.073 | -0.050 | 50.0% | 0.0% | 0.3333 | 0.0000 | 103.0 | 0.5000 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=100% |
| 4 | vizwiz | +1.027 | 2.054 | -0.070 | 26.0% | 20.0% | 0.2000 | 0.2000 | 104.0 | 0.7000 | vqa_acc | parse_fail=0, over_budget=0% |

### psQ_seq  cell o1/s23

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | scienceqa | -0.000 | 2.073 | -0.050 | 50.0% | 0.0% | 0.2000 | 0.2000 | 101.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |
| 2 | textvqa | +0.000 | 3.290 | +1.166 | 50.0% | 0.0% | 0.2000 | 0.2000 | 102.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 3 | flickr | -0.000 | 2.073 | -0.050 | 50.0% | 0.0% | 0.3333 | 0.0000 | 103.0 | 0.5000 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=100% |
| 4 | vizwiz | +1.027 | 2.054 | -0.070 | 26.0% | 20.0% | 0.2000 | 0.2000 | 104.0 | 0.7000 | vqa_acc | parse_fail=0, over_budget=0% |

### psQ_seq  cell o2/s31

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | vizwiz | -0.000 | 2.073 | -0.050 | 50.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.7000 | vqa_acc | parse_fail=0, over_budget=0% |
| 2 | flickr | +0.696 | 1.898 | -0.225 | 32.5% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.5000 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=0% |
| 3 | textvqa | -0.000 | 2.073 | -0.050 | 50.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 4 | scienceqa | +1.027 | 2.054 | -0.070 | 26.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |

## Scorecard per arm (per cell, then [min, max] across cells)

| arm | cell | Sum abs dc | post-settle Sum | endpoint c | max abs d'-base | F1 | endpoint leak | endpoint CHAIR_i@60 | endpoint CHAIR_i | mean plasticity | plasticity per stage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mth_critp | o1/s17 | 0.064 | 0.000 | +0.284 | 0.010 | ok | 4.0% | 0.0476 | 0.0476 | 0.6400 | 0.800, 0.600, 0.480, 0.680 |
| mth_critp | o1/s23 | 0.064 | 0.000 | +0.284 | 0.010 | ok | 4.0% | 0.0476 | 0.0476 | 0.6800 | 0.840, 0.640, 0.520, 0.720 |
| mth_critp | **range** | [0.064, 0.064] n=2 | [0.000, 0.000] n=2 | [+0.284, +0.284] n=2 | [0.010, 0.010] n=2 | 0/2 tripped | [0.040, 0.040] n=2 | [0.0476, 0.0476] n=2 | [0.0476, 0.0476] n=2 | [0.6400, 0.6800] n=2 | - |
| psQ_seq | o1/s17 | 2.639 | 2.418 | +1.027 | 0.225 | ok | 20.0% | 0.2000 | 0.2000 | 0.6500 | 0.800, 0.600, 0.500, 0.700 |
| psQ_seq | o1/s23 | 1.247 | 1.027 | +1.027 | 1.167 | TRIPPED | 20.0% | 0.2000 | 0.2000 | 0.6500 | 0.800, 0.600, 0.500, 0.700 |
| psQ_seq | o2/s31 | 2.639 | 2.418 | +1.027 | 0.225 | ok | 0.0% | 0.2000 | 0.2000 | 0.6500 | 0.700, 0.500, 0.600, 0.800 |
| psQ_seq | **range** | [1.247, 2.639] n=3 | [1.027, 2.418] n=3 | [+1.027, +1.027] n=3 | [0.225, 1.167] n=3 | 1/3 tripped | [0.000, 0.200] n=3 | [0.2000, 0.2000] n=3 | [0.2000, 0.2000] n=3 | [0.6500, 0.6500] n=3 | - |

## Versus reference arm mth_critp (paired per matched cell; arm - ref)

### psQ_seq vs mth_critp

| cell | d Sum abs dc | d post-settle | d endpoint c | d CHAIR_i@60 | CI95 (paired image bootstrap) | n pairs | d mean plasticity | d endpoint leak |
|---|---|---|---|---|---|---|---|---|
| o1/s17 | +2.574 | +2.418 | +0.743 | +0.1524 | [+0.1139, +0.1895] excl0 | 80 | +0.0100 | +0.1600 |
| o1/s23 | +1.183 | +1.027 | +0.743 | +0.1524 | [+0.1139, +0.1895] excl0 | 80 | -0.0300 | +0.1600 |

sign consistency: Sum|dc| lower 0/2; CHAIR_i@60 lower 0/2 (CI excl 0 and lower 0/2); plasticity >= ref-0.02 1/2 (|d| <= 0.02: 1/2)

## PASS / FAIL per pre-registered prediction

| arm | P1 Sum abs dc | P2 plasticity | P3 d' falsifier | P4 CHAIR_i@60 | overall |
|---|---|---|---|---|---|
| mth_critp (ref) | - | - | PASS (2/2 ok) | - | reference |
| psQ_seq | FAIL (0/2) | FAIL (1/2) | FAIL (2/3 ok) | FAIL (0/2) | **FAIL** |

- P1: Sum|dc|(arm) < Sum|dc|(ref) in EVERY matched cell (c_0 = base); count/n reported so a 2/3 result reads as PARTIAL
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

