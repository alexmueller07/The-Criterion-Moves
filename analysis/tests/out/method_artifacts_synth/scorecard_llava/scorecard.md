# Method scorecard: pilot suite, ref psL_seq

- results: `/Users/alexandermueller/Research/cl-hallucination-vlm/analysis/tests/fixtures/results_method_artifacts_synth/tree_llava`
- base (stage 0): `/Users/alexandermueller/Research/cl-hallucination-vlm/analysis/tests/fixtures/results_method_artifacts_synth/tree_llava/llava15_base` (EVAL_DONE)
- suite: pilot (4 stages; orders from /Users/alexandermueller/Research/cl-hallucination-vlm/fullstudy/pilot_manifest.json)
- length budget k=60 words; paired bootstrap B=300, seed=17; CHAIR helpers: analysis/diag/length_controlled_chair.py
- leakage task: textvqa

Style: per-cell values and [min, max] across cells; orderings are strata; no pooled cross-seed CIs (design_notes/review_statistics.md sec 2).

## Discovery

| arm | complete cells | incomplete / skipped |
|---|---|---|
| psL_seq (ref) | o1/s17, o1/s23, o1/s31, o2/s17 | - |
| psL_anchor | o1/s17, o1/s23, o1/s31 | - |
| psL_er | o1/s17, o1/s23 | - |
| psL_ewc | o1/s17 | - |
| psL_lwf | none | o1/s17: missing EVAL_DONE stages [4]; IN-PROGRESS: psL_lwf_o1_s17_k4 lacks EVAL_DONE |
| psL_cecf | o1/s17 | - |
| mth_critp | o1/s17, o1/s23, o1/s31 | - |
| mth_critp1 | o1/s17 | - |
| mth_anchorcrit | o1/s17, o1/s23 | - |
| psL_seq_neu | o1/s17, o1/s23, o1/s31 | - |
| psL_seq_amp | o1/s17 | - |
| psL_joint | none | o1/s17: final cell (psL_joint_o1_s17_final) is not a staged sequential run; skipped |

## Per-stage metrics

### psL_seq  cell o1/s17

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | scienceqa | -0.000 | 2.073 | -0.050 | 50.0% | 0.0% | 0.2000 | 0.2000 | 101.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |
| 2 | textvqa | -0.696 | 1.898 | -0.225 | 67.5% | 0.0% | 0.2000 | 0.2000 | 101.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 3 | flickr | -0.515 | 2.079 | -0.044 | 62.0% | 0.0% | 0.3333 | 0.0000 | 101.0 | 0.5000 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=100% |
| 4 | vizwiz | +1.027 | 2.054 | -0.070 | 26.0% | 20.0% | 0.2000 | 0.2000 | 101.0 | 0.7000 | vqa_acc | parse_fail=0, over_budget=0% |

### psL_seq  cell o1/s23

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | scienceqa | -0.000 | 2.073 | -0.050 | 50.0% | 0.0% | 0.2000 | 0.2000 | 101.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |
| 2 | textvqa | -0.386 | 3.335 | +1.212 | 54.0% | 0.0% | 0.2000 | 0.2000 | 101.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 3 | flickr | -0.515 | 2.079 | -0.044 | 62.0% | 0.0% | 0.2000 | 0.2000 | 101.0 | 0.5000 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=0% |
| 4 | vizwiz | +1.027 | 2.054 | -0.070 | 26.0% | 20.0% | 0.2000 | 0.2000 | 101.0 | 0.7000 | vqa_acc | parse_fail=0, over_budget=0% |

### psL_seq  cell o1/s31

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | scienceqa | -0.000 | 2.073 | -0.050 | 50.0% | 0.0% | 0.2000 | 0.2000 | 101.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |
| 2 | textvqa | -0.696 | 1.898 | -0.225 | 67.5% | 0.0% | 0.2000 | 0.2000 | 101.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 3 | flickr | -0.515 | 2.079 | -0.044 | 62.0% | 0.0% | 0.3333 | 0.0000 | 101.0 | 0.5000 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=100% |
| 4 | vizwiz | +1.027 | 2.054 | -0.070 | 26.0% | 20.0% | 0.2000 | 0.2000 | 101.0 | 0.7000 | vqa_acc | parse_fail=0, over_budget=0% |

### psL_seq  cell o2/s17

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | vizwiz | -0.000 | 2.073 | -0.050 | 50.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.7000 | vqa_acc | parse_fail=0, over_budget=0% |
| 2 | flickr | -0.696 | 1.898 | -0.225 | 67.5% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.5000 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=0% |
| 3 | textvqa | -0.515 | 2.079 | -0.044 | 62.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 4 | scienceqa | +1.027 | 2.054 | -0.070 | 26.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |

### psL_anchor  cell o1/s17

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | scienceqa | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0476 | 0.0476 | 90.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |
| 2 | textvqa | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0476 | 0.0476 | 90.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 3 | flickr | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0476 | 0.0476 | 90.0 | 0.4800 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=0% |
| 4 | vizwiz | +0.284 | 2.113 | -0.010 | 43.5% | 4.0% | 0.0476 | 0.0476 | 90.0 | 0.6800 | vqa_acc | parse_fail=0, over_budget=0% |

### psL_anchor  cell o1/s23

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | scienceqa | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0476 | 0.0476 | 90.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |
| 2 | textvqa | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0476 | 0.0476 | 90.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 3 | flickr | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0476 | 0.0476 | 90.0 | 0.4800 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=0% |
| 4 | vizwiz | +0.284 | 2.113 | -0.010 | 43.5% | 4.0% | 0.0476 | 0.0476 | 90.0 | 0.6800 | vqa_acc | parse_fail=0, over_budget=0% |

### psL_anchor  cell o1/s31

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | scienceqa | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0476 | 0.0476 | 90.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |
| 2 | textvqa | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0476 | 0.0476 | 90.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 3 | flickr | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0476 | 0.0476 | 90.0 | 0.4800 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=0% |
| 4 | vizwiz | +0.284 | 2.113 | -0.010 | 43.5% | 4.0% | 0.0476 | 0.0476 | 90.0 | 0.6800 | vqa_acc | parse_fail=0, over_budget=0% |

### psL_er  cell o1/s17

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | scienceqa | -0.000 | 2.073 | -0.050 | 50.0% | 0.0% | 0.2157 | 0.2157 | 100.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |
| 2 | textvqa | -0.696 | 1.898 | -0.225 | 67.5% | 0.0% | 0.2157 | 0.2157 | 100.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 3 | flickr | -0.515 | 2.079 | -0.044 | 62.0% | 0.0% | 0.2157 | 0.2157 | 100.0 | 0.5000 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=0% |
| 4 | vizwiz | +1.027 | 2.054 | -0.070 | 26.0% | 16.0% | 0.2157 | 0.2157 | 100.0 | 0.7000 | vqa_acc | parse_fail=0, over_budget=0% |

### psL_er  cell o1/s23

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | scienceqa | -0.000 | 2.073 | -0.050 | 50.0% | 0.0% | 0.2157 | 0.2157 | 100.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |
| 2 | textvqa | -0.696 | 1.898 | -0.225 | 67.5% | 0.0% | 0.2157 | 0.2157 | 100.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 3 | flickr | -0.515 | 2.079 | -0.044 | 62.0% | 0.0% | 0.2157 | 0.2157 | 100.0 | 0.5000 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=0% |
| 4 | vizwiz | +1.027 | 2.054 | -0.070 | 26.0% | 16.0% | 0.2157 | 0.2157 | 100.0 | 0.7000 | vqa_acc | parse_fail=0, over_budget=0% |

### psL_ewc  cell o1/s17

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | scienceqa | +0.130 | 2.090 | -0.033 | 47.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |
| 2 | textvqa | +0.130 | 2.090 | -0.033 | 47.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 3 | flickr | +0.130 | 2.090 | -0.033 | 47.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.5000 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=0% |
| 4 | vizwiz | +0.130 | 2.090 | -0.033 | 47.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.7000 | vqa_acc | parse_fail=0, over_budget=0% |

### psL_cecf  cell o1/s17

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | scienceqa | +0.560 | 2.169 | +0.046 | 37.5% | 0.0% | 0.1304 | 0.1304 | 95.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |
| 2 | textvqa | +0.560 | 2.169 | +0.046 | 37.5% | 0.0% | 0.1304 | 0.1304 | 95.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 3 | flickr | +0.560 | 2.169 | +0.046 | 37.5% | 0.0% | 0.1304 | 0.1304 | 95.0 | 0.4800 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=0% |
| 4 | vizwiz | +0.560 | 2.169 | +0.046 | 37.5% | 0.0% | 0.1304 | 0.1304 | 95.0 | 0.6800 | vqa_acc | parse_fail=0, over_budget=0% |

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
| 1 | scienceqa | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0476 | 0.0476 | 90.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |
| 2 | textvqa | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0476 | 0.0476 | 90.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 3 | flickr | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0476 | 0.0476 | 90.0 | 0.4800 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=0% |
| 4 | vizwiz | +0.284 | 2.113 | -0.010 | 43.5% | 4.0% | 0.0476 | 0.0476 | 90.0 | 0.6800 | vqa_acc | parse_fail=0, over_budget=0% |

### mth_critp  cell o1/s31

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | scienceqa | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0476 | 0.0476 | 90.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |
| 2 | textvqa | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0476 | 0.0476 | 90.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 3 | flickr | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0476 | 0.0476 | 90.0 | 0.4800 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=0% |
| 4 | vizwiz | +0.284 | 2.113 | -0.010 | 43.5% | 4.0% | 0.0476 | 0.0476 | 90.0 | 0.6800 | vqa_acc | parse_fail=0, over_budget=0% |

### mth_critp1  cell o1/s17

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | scienceqa | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0698 | 0.0698 | 90.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |
| 2 | textvqa | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0698 | 0.0698 | 90.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 3 | flickr | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0698 | 0.0698 | 90.0 | 0.4800 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=0% |
| 4 | vizwiz | +0.284 | 2.113 | -0.010 | 43.5% | 0.0% | 0.0698 | 0.0698 | 90.0 | 0.6800 | vqa_acc | parse_fail=0, over_budget=0% |

### mth_anchorcrit  cell o1/s17

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | scienceqa | +0.282 | 2.247 | +0.123 | 44.0% | 0.0% | 0.0361 | 0.0361 | 90.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |
| 2 | textvqa | +0.282 | 2.247 | +0.123 | 44.0% | 0.0% | 0.0361 | 0.0361 | 90.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 3 | flickr | +0.282 | 2.247 | +0.123 | 44.0% | 0.0% | 0.0361 | 0.0361 | 90.0 | 0.4800 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=0% |
| 4 | vizwiz | +0.282 | 2.247 | +0.123 | 44.0% | 0.0% | 0.0361 | 0.0361 | 90.0 | 0.6800 | vqa_acc | parse_fail=0, over_budget=0% |

### mth_anchorcrit  cell o1/s23

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | scienceqa | +0.282 | 2.247 | +0.123 | 44.0% | 0.0% | 0.0361 | 0.0361 | 90.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |
| 2 | textvqa | +0.282 | 2.247 | +0.123 | 44.0% | 0.0% | 0.0361 | 0.0361 | 90.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 3 | flickr | +0.282 | 2.247 | +0.123 | 44.0% | 0.0% | 0.0361 | 0.0361 | 90.0 | 0.4800 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=0% |
| 4 | vizwiz | +0.282 | 2.247 | +0.123 | 44.0% | 0.0% | 0.0361 | 0.0361 | 90.0 | 0.6800 | vqa_acc | parse_fail=0, over_budget=0% |

### psL_seq_neu  cell o1/s17

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | scienceqa | +0.130 | 2.090 | -0.033 | 47.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |
| 2 | textvqa | +0.220 | 2.123 | +0.000 | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 3 | flickr | +0.174 | 2.104 | -0.019 | 46.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.5000 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=0% |
| 4 | vizwiz | +0.238 | 2.088 | -0.035 | 44.5% | 2.0% | 0.2000 | 0.2000 | 100.0 | 0.7000 | vqa_acc | parse_fail=0, over_budget=0% |

### psL_seq_neu  cell o1/s23

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | scienceqa | +0.130 | 2.090 | -0.033 | 47.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |
| 2 | textvqa | +0.220 | 2.123 | +0.000 | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 3 | flickr | +0.174 | 2.104 | -0.019 | 46.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.5000 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=0% |
| 4 | vizwiz | +0.238 | 2.088 | -0.035 | 44.5% | 2.0% | 0.2000 | 0.2000 | 100.0 | 0.7000 | vqa_acc | parse_fail=0, over_budget=0% |

### psL_seq_neu  cell o1/s31

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | scienceqa | +0.130 | 2.090 | -0.033 | 47.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |
| 2 | textvqa | +0.220 | 2.123 | +0.000 | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 3 | flickr | +0.174 | 2.104 | -0.019 | 46.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.5000 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=0% |
| 4 | vizwiz | +0.238 | 2.088 | -0.035 | 44.5% | 2.0% | 0.2000 | 0.2000 | 100.0 | 0.7000 | vqa_acc | parse_fail=0, over_budget=0% |

### psL_seq_amp  cell o1/s17

| stage | task learned | c | d' | d'-base | yes% | leak% | CHAIR_i | CHAIR_i@60 | tok | new-task acc | metric | audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | base | +0.220 | 2.123 | - | 45.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | - | - | parse_fail=2, over_budget=0% |
| 1 | scienceqa | -0.000 | 2.073 | -0.050 | 50.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.8000 | mc_acc | parse_fail=0, over_budget=0% |
| 2 | textvqa | -0.696 | 1.898 | -0.225 | 67.5% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.6000 | vqa_acc | parse_fail=0, over_budget=0% |
| 3 | flickr | -0.515 | 2.079 | -0.044 | 62.0% | 0.0% | 0.2000 | 0.2000 | 100.0 | 0.5000 | caption_uf1 *open-ended floor* | parse_fail=0, over_budget=0% |
| 4 | vizwiz | +1.290 | 2.073 | -0.050 | 20.5% | 30.0% | 0.2000 | 0.2000 | 100.0 | 0.7000 | vqa_acc | parse_fail=0, over_budget=0% |

## Scorecard per arm (per cell, then [min, max] across cells)

| arm | cell | Sum abs dc | post-settle Sum | endpoint c | max abs d'-base | F1 | endpoint leak | endpoint CHAIR_i@60 | endpoint CHAIR_i | mean plasticity | plasticity per stage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| psL_seq | o1/s17 | 2.639 | 2.418 | +1.027 | 0.225 | ok | 20.0% | 0.2000 | 0.2000 | 0.6500 | 0.800, 0.600, 0.500, 0.700 |
| psL_seq | o1/s23 | 2.277 | 2.057 | +1.027 | 1.212 | TRIPPED | 20.0% | 0.2000 | 0.2000 | 0.6500 | 0.800, 0.600, 0.500, 0.700 |
| psL_seq | o1/s31 | 2.639 | 2.418 | +1.027 | 0.225 | ok | 20.0% | 0.2000 | 0.2000 | 0.6500 | 0.800, 0.600, 0.500, 0.700 |
| psL_seq | o2/s17 | 2.639 | 2.418 | +1.027 | 0.225 | ok | 0.0% | 0.2000 | 0.2000 | 0.6500 | 0.700, 0.500, 0.600, 0.800 |
| psL_seq | **range** | [2.277, 2.639] n=4 | [2.057, 2.418] n=4 | [+1.027, +1.027] n=4 | [0.225, 1.212] n=4 | 1/4 tripped | [0.000, 0.200] n=4 | [0.2000, 0.2000] n=4 | [0.2000, 0.2000] n=4 | [0.6500, 0.6500] n=4 | - |
| psL_anchor | o1/s17 | 0.064 | 0.000 | +0.284 | 0.010 | ok | 4.0% | 0.0476 | 0.0476 | 0.6400 | 0.800, 0.600, 0.480, 0.680 |
| psL_anchor | o1/s23 | 0.064 | 0.000 | +0.284 | 0.010 | ok | 4.0% | 0.0476 | 0.0476 | 0.6400 | 0.800, 0.600, 0.480, 0.680 |
| psL_anchor | o1/s31 | 0.064 | 0.000 | +0.284 | 0.010 | ok | 4.0% | 0.0476 | 0.0476 | 0.6400 | 0.800, 0.600, 0.480, 0.680 |
| psL_anchor | **range** | [0.064, 0.064] n=3 | [0.000, 0.000] n=3 | [+0.284, +0.284] n=3 | [0.010, 0.010] n=3 | 0/3 tripped | [0.040, 0.040] n=3 | [0.0476, 0.0476] n=3 | [0.0476, 0.0476] n=3 | [0.6400, 0.6400] n=3 | - |
| psL_er | o1/s17 | 2.639 | 2.418 | +1.027 | 0.225 | ok | 16.0% | 0.2157 | 0.2157 | 0.6500 | 0.800, 0.600, 0.500, 0.700 |
| psL_er | o1/s23 | 2.639 | 2.418 | +1.027 | 0.225 | ok | 16.0% | 0.2157 | 0.2157 | 0.6500 | 0.800, 0.600, 0.500, 0.700 |
| psL_er | **range** | [2.639, 2.639] n=2 | [2.418, 2.418] n=2 | [+1.027, +1.027] n=2 | [0.225, 0.225] n=2 | 0/2 tripped | [0.160, 0.160] n=2 | [0.2157, 0.2157] n=2 | [0.2157, 0.2157] n=2 | [0.6500, 0.6500] n=2 | - |
| psL_ewc | o1/s17 | 0.090 | 0.000 | +0.130 | 0.033 | ok | 0.0% | 0.2000 | 0.2000 | 0.6500 | 0.800, 0.600, 0.500, 0.700 |
| psL_ewc | **range** | [0.090, 0.090] n=1 | [0.000, 0.000] n=1 | [+0.130, +0.130] n=1 | [0.033, 0.033] n=1 | 0/1 tripped | [0.000, 0.000] n=1 | [0.2000, 0.2000] n=1 | [0.2000, 0.2000] n=1 | [0.6500, 0.6500] n=1 | - |
| psL_lwf | **range** | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT | - |
| psL_cecf | o1/s17 | 0.340 | 0.000 | +0.560 | 0.046 | ok | 0.0% | 0.1304 | 0.1304 | 0.6400 | 0.800, 0.600, 0.480, 0.680 |
| psL_cecf | **range** | [0.340, 0.340] n=1 | [0.000, 0.000] n=1 | [+0.560, +0.560] n=1 | [0.046, 0.046] n=1 | 0/1 tripped | [0.000, 0.000] n=1 | [0.1304, 0.1304] n=1 | [0.1304, 0.1304] n=1 | [0.6400, 0.6400] n=1 | - |
| mth_critp | o1/s17 | 0.064 | 0.000 | +0.284 | 0.010 | ok | 4.0% | 0.0476 | 0.0476 | 0.6400 | 0.800, 0.600, 0.480, 0.680 |
| mth_critp | o1/s23 | 0.064 | 0.000 | +0.284 | 0.010 | ok | 4.0% | 0.0476 | 0.0476 | 0.6400 | 0.800, 0.600, 0.480, 0.680 |
| mth_critp | o1/s31 | 0.064 | 0.000 | +0.284 | 0.010 | ok | 4.0% | 0.0476 | 0.0476 | 0.6400 | 0.800, 0.600, 0.480, 0.680 |
| mth_critp | **range** | [0.064, 0.064] n=3 | [0.000, 0.000] n=3 | [+0.284, +0.284] n=3 | [0.010, 0.010] n=3 | 0/3 tripped | [0.040, 0.040] n=3 | [0.0476, 0.0476] n=3 | [0.0476, 0.0476] n=3 | [0.6400, 0.6400] n=3 | - |
| mth_critp1 | o1/s17 | 0.064 | 0.000 | +0.284 | 0.010 | ok | 0.0% | 0.0698 | 0.0698 | 0.6400 | 0.800, 0.600, 0.480, 0.680 |
| mth_critp1 | **range** | [0.064, 0.064] n=1 | [0.000, 0.000] n=1 | [+0.284, +0.284] n=1 | [0.010, 0.010] n=1 | 0/1 tripped | [0.000, 0.000] n=1 | [0.0698, 0.0698] n=1 | [0.0698, 0.0698] n=1 | [0.6400, 0.6400] n=1 | - |
| mth_anchorcrit | o1/s17 | 0.062 | 0.000 | +0.282 | 0.123 | ok | 0.0% | 0.0361 | 0.0361 | 0.6400 | 0.800, 0.600, 0.480, 0.680 |
| mth_anchorcrit | o1/s23 | 0.062 | 0.000 | +0.282 | 0.123 | ok | 0.0% | 0.0361 | 0.0361 | 0.6400 | 0.800, 0.600, 0.480, 0.680 |
| mth_anchorcrit | **range** | [0.062, 0.062] n=2 | [0.000, 0.000] n=2 | [+0.282, +0.282] n=2 | [0.123, 0.123] n=2 | 0/2 tripped | [0.000, 0.000] n=2 | [0.0361, 0.0361] n=2 | [0.0361, 0.0361] n=2 | [0.6400, 0.6400] n=2 | - |
| psL_seq_neu | o1/s17 | 0.289 | 0.199 | +0.238 | 0.035 | ok | 2.0% | 0.2000 | 0.2000 | 0.6500 | 0.800, 0.600, 0.500, 0.700 |
| psL_seq_neu | o1/s23 | 0.289 | 0.199 | +0.238 | 0.035 | ok | 2.0% | 0.2000 | 0.2000 | 0.6500 | 0.800, 0.600, 0.500, 0.700 |
| psL_seq_neu | o1/s31 | 0.289 | 0.199 | +0.238 | 0.035 | ok | 2.0% | 0.2000 | 0.2000 | 0.6500 | 0.800, 0.600, 0.500, 0.700 |
| psL_seq_neu | **range** | [0.289, 0.289] n=3 | [0.199, 0.199] n=3 | [+0.238, +0.238] n=3 | [0.035, 0.035] n=3 | 0/3 tripped | [0.020, 0.020] n=3 | [0.2000, 0.2000] n=3 | [0.2000, 0.2000] n=3 | [0.6500, 0.6500] n=3 | - |
| psL_seq_amp | o1/s17 | 2.901 | 2.681 | +1.290 | 0.225 | ok | 30.0% | 0.2000 | 0.2000 | 0.6500 | 0.800, 0.600, 0.500, 0.700 |
| psL_seq_amp | **range** | [2.901, 2.901] n=1 | [2.681, 2.681] n=1 | [+1.290, +1.290] n=1 | [0.225, 0.225] n=1 | 0/1 tripped | [0.300, 0.300] n=1 | [0.2000, 0.2000] n=1 | [0.2000, 0.2000] n=1 | [0.6500, 0.6500] n=1 | - |
| psL_joint | **range** | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT | - |

## Versus reference arm psL_seq (paired per matched cell; arm - ref)

### psL_anchor vs psL_seq

| cell | d Sum abs dc | d post-settle | d endpoint c | d CHAIR_i@60 | CI95 (paired image bootstrap) | n pairs | d mean plasticity | d endpoint leak |
|---|---|---|---|---|---|---|---|---|
| o1/s17 | -2.574 | -2.418 | -0.743 | -0.1524 | [-0.1909, -0.1134] excl0 | 80 | -0.0100 | -0.1600 |
| o1/s23 | -2.213 | -2.057 | -0.743 | -0.1524 | [-0.1909, -0.1134] excl0 | 80 | -0.0100 | -0.1600 |
| o1/s31 | -2.574 | -2.418 | -0.743 | -0.1524 | [-0.1909, -0.1134] excl0 | 80 | -0.0100 | -0.1600 |

sign consistency: Sum|dc| lower 3/3; CHAIR_i@60 lower 3/3 (CI excl 0 and lower 3/3); plasticity >= ref-0.02 3/3 (|d| <= 0.02: 3/3)

### psL_er vs psL_seq

| cell | d Sum abs dc | d post-settle | d endpoint c | d CHAIR_i@60 | CI95 (paired image bootstrap) | n pairs | d mean plasticity | d endpoint leak |
|---|---|---|---|---|---|---|---|---|
| o1/s17 | +0.000 | +0.000 | +0.000 | +0.0157 | [+0.0037, +0.0312] excl0 | 80 | +0.0000 | -0.0400 |
| o1/s23 | +0.361 | +0.361 | +0.000 | +0.0157 | [+0.0037, +0.0312] excl0 | 80 | +0.0000 | -0.0400 |

sign consistency: Sum|dc| lower 0/2; CHAIR_i@60 lower 0/2 (CI excl 0 and lower 0/2); plasticity >= ref-0.02 2/2 (|d| <= 0.02: 2/2)

### psL_ewc vs psL_seq

| cell | d Sum abs dc | d post-settle | d endpoint c | d CHAIR_i@60 | CI95 (paired image bootstrap) | n pairs | d mean plasticity | d endpoint leak |
|---|---|---|---|---|---|---|---|---|
| o1/s17 | -2.548 | -2.418 | -0.897 | +0.0000 | [+0.0000, +0.0000] | 80 | +0.0000 | -0.2000 |

sign consistency: Sum|dc| lower 1/1; CHAIR_i@60 lower 0/1 (CI excl 0 and lower 0/1); plasticity >= ref-0.02 1/1 (|d| <= 0.02: 1/1)

### psL_lwf vs psL_seq

no matched (order, seed) cell complete in both arms -> ABSENT

### psL_cecf vs psL_seq

| cell | d Sum abs dc | d post-settle | d endpoint c | d CHAIR_i@60 | CI95 (paired image bootstrap) | n pairs | d mean plasticity | d endpoint leak |
|---|---|---|---|---|---|---|---|---|
| o1/s17 | -2.298 | -2.418 | -0.467 | -0.0696 | [-0.1007, -0.0410] excl0 | 80 | -0.0100 | -0.2000 |

sign consistency: Sum|dc| lower 1/1; CHAIR_i@60 lower 1/1 (CI excl 0 and lower 1/1); plasticity >= ref-0.02 1/1 (|d| <= 0.02: 1/1)

### mth_critp vs psL_seq

| cell | d Sum abs dc | d post-settle | d endpoint c | d CHAIR_i@60 | CI95 (paired image bootstrap) | n pairs | d mean plasticity | d endpoint leak |
|---|---|---|---|---|---|---|---|---|
| o1/s17 | -2.574 | -2.418 | -0.743 | -0.1524 | [-0.1909, -0.1134] excl0 | 80 | -0.0100 | -0.1600 |
| o1/s23 | -2.213 | -2.057 | -0.743 | -0.1524 | [-0.1909, -0.1134] excl0 | 80 | -0.0100 | -0.1600 |
| o1/s31 | -2.574 | -2.418 | -0.743 | -0.1524 | [-0.1909, -0.1134] excl0 | 80 | -0.0100 | -0.1600 |

sign consistency: Sum|dc| lower 3/3; CHAIR_i@60 lower 3/3 (CI excl 0 and lower 3/3); plasticity >= ref-0.02 3/3 (|d| <= 0.02: 3/3)

### mth_critp1 vs psL_seq

| cell | d Sum abs dc | d post-settle | d endpoint c | d CHAIR_i@60 | CI95 (paired image bootstrap) | n pairs | d mean plasticity | d endpoint leak |
|---|---|---|---|---|---|---|---|---|
| o1/s17 | -2.574 | -2.418 | -0.743 | -0.1302 | [-0.1660, -0.0958] excl0 | 80 | -0.0100 | -0.2000 |

sign consistency: Sum|dc| lower 1/1; CHAIR_i@60 lower 1/1 (CI excl 0 and lower 1/1); plasticity >= ref-0.02 1/1 (|d| <= 0.02: 1/1)

### mth_anchorcrit vs psL_seq

| cell | d Sum abs dc | d post-settle | d endpoint c | d CHAIR_i@60 | CI95 (paired image bootstrap) | n pairs | d mean plasticity | d endpoint leak |
|---|---|---|---|---|---|---|---|---|
| o1/s17 | -2.577 | -2.418 | -0.745 | -0.1639 | [-0.1983, -0.1262] excl0 | 80 | -0.0100 | -0.2000 |
| o1/s23 | -2.216 | -2.057 | -0.745 | -0.1639 | [-0.1983, -0.1262] excl0 | 80 | -0.0100 | -0.2000 |

sign consistency: Sum|dc| lower 2/2; CHAIR_i@60 lower 2/2 (CI excl 0 and lower 2/2); plasticity >= ref-0.02 2/2 (|d| <= 0.02: 2/2)

### psL_seq_neu vs psL_seq

| cell | d Sum abs dc | d post-settle | d endpoint c | d CHAIR_i@60 | CI95 (paired image bootstrap) | n pairs | d mean plasticity | d endpoint leak |
|---|---|---|---|---|---|---|---|---|
| o1/s17 | -2.349 | -2.219 | -0.789 | +0.0000 | [+0.0000, +0.0000] | 80 | +0.0000 | -0.1800 |
| o1/s23 | -1.988 | -1.858 | -0.789 | +0.0000 | [+0.0000, +0.0000] | 80 | +0.0000 | -0.1800 |
| o1/s31 | -2.349 | -2.219 | -0.789 | +0.0000 | [+0.0000, +0.0000] | 80 | +0.0000 | -0.1800 |

sign consistency: Sum|dc| lower 3/3; CHAIR_i@60 lower 0/3 (CI excl 0 and lower 0/3); plasticity >= ref-0.02 3/3 (|d| <= 0.02: 3/3)

### psL_seq_amp vs psL_seq

| cell | d Sum abs dc | d post-settle | d endpoint c | d CHAIR_i@60 | CI95 (paired image bootstrap) | n pairs | d mean plasticity | d endpoint leak |
|---|---|---|---|---|---|---|---|---|
| o1/s17 | +0.263 | +0.263 | +0.263 | +0.0000 | [+0.0000, +0.0000] | 80 | +0.0000 | +0.1000 |

sign consistency: Sum|dc| lower 0/1; CHAIR_i@60 lower 0/1 (CI excl 0 and lower 0/1); plasticity >= ref-0.02 1/1 (|d| <= 0.02: 1/1)

### psL_joint vs psL_seq

no matched (order, seed) cell complete in both arms -> ABSENT

## PASS / FAIL per pre-registered prediction

| arm | P1 Sum abs dc | P2 plasticity | P3 d' falsifier | P4 CHAIR_i@60 | overall |
|---|---|---|---|---|---|
| psL_seq (ref) | - | - | FAIL (3/4 ok) | - | reference |
| psL_anchor | PASS (3/3) | PASS (3/3) | PASS (3/3 ok) | PASS (3/3) | **PASS** |
| psL_er | FAIL (0/2) | PASS (2/2) | PASS (2/2 ok) | FAIL (0/2) | **FAIL** |
| psL_ewc | PASS (1/1) | PASS (1/1) | PASS (1/1 ok) | FAIL (0/1) | **FAIL** |
| psL_lwf | ABSENT (0/0) | ABSENT (0/0) | ABSENT (0/0 ok) | ABSENT (0/0) | **INCOMPLETE** |
| psL_cecf | PASS (1/1) | PASS (1/1) | PASS (1/1 ok) | PASS (1/1) | **PASS** |
| mth_critp | PASS (3/3) | PASS (3/3) | PASS (3/3 ok) | PASS (3/3) | **PASS** |
| mth_critp1 | PASS (1/1) | PASS (1/1) | PASS (1/1 ok) | PASS (1/1) | **PASS** |
| mth_anchorcrit | PASS (2/2) | PASS (2/2) | PASS (2/2 ok) | PASS (2/2) | **PASS** |
| psL_seq_neu | PASS (3/3) | PASS (3/3) | PASS (3/3 ok) | FAIL (0/3) | **FAIL** |
| psL_seq_amp | FAIL (0/1) | PASS (1/1) | PASS (1/1 ok) | FAIL (0/1) | **FAIL** |
| psL_joint | ABSENT (0/0) | ABSENT (0/0) | ABSENT (0/0 ok) | ABSENT (0/0) | **INCOMPLETE** |

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

