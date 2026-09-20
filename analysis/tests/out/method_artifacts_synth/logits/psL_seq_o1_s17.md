# Logit-level additive-bias test: psL_seq_o1_s17

Base: `/Users/alexandermueller/Research/cl-hallucination-vlm/analysis/tests/fixtures/results_method_artifacts_synth/logits/llava15_base/pope_logits.jsonl`; stages: 4; joined ids: 3000 (base rows 3000; per-file drops: {'base': 0, 'psL_seq_o1_s17_k1': 0, 'psL_seq_o1_s17_k2': 0, 'psL_seq_o1_s17_k3': 0, 'psL_seq_o1_s17_k4': 0}). Bootstrap n=200, seed=0.

Theory (Claim 1): a stage shifts g by one class-independent constant -> `diff` ~ 0 (CI covers 0 or |diff| << |b_common|), `nonadd(class)` and `nonadd(z)` small, `dd'` ~ 0, H and FA co-move, class sds unchanged. Failure: `shift_yes` != `shift_no` (separation changes = a d' move). `nonadd(class)` large with `nonadd(z)` small means the class SPREAD changed (scale, not location; see the sd table): the equal-variance location model failed, not d'-invariance. `dc undone` is the feasibility check for (b), not a theory test: on balanced data the pooled offset restores the midpoint c even when the separation changed.

## (a) Additive-bias test vs base (gap units = logits)

| stage | shift_yes | shift_no | diff [95% CI] | b_common | R2_item | nonadd(class) | dc undone | c_gap | dc | d'_gap | dd' | nonadd(z) | H,FA co-move |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | - | - | - | - | - | - | - | -0.042 | - | 2.949 | - | - | - |
| psL_seq_o1_s17_k1 | -0.301 | -0.300 | -0.001 [-0.005, +0.003] | -0.300 | 0.97 | 0.4% | 99% | +0.287 | +0.330 | 3.007 | +0.057 | 14.9% | yes |
| psL_seq_o1_s17_k2 | +0.999 | +0.999 | -0.000 [-0.003, +0.004] | +0.999 | 1.00 | 0.0% | 99% | -1.018 | -0.976 | 2.987 | +0.038 | 3.8% | yes |
| psL_seq_o1_s17_k3 | -0.203 | -0.202 | -0.002 [-0.006, +0.002] | -0.202 | 0.94 | 0.8% | 99% | +0.212 | +0.254 | 3.033 | +0.083 | 24.7% | yes |
| psL_seq_o1_s17_k4 | +1.500 | +1.500 | -0.000 [-0.003, +0.003] | +1.500 | 1.00 | 0.0% | 100% | -1.417 | -1.374 | 2.923 | -0.026 | 1.9% | yes |

Stage-to-stage (k-1 -> k):

| transition | shift_yes | shift_no | diff [95% CI] | b_common | nonadd(class) | dc undone | dc | dd' |
|---|---|---|---|---|---|---|---|---|
| base -> psL_seq_o1_s17_k1 | -0.301 | -0.300 | -0.001 [-0.004, +0.002] | -0.300 | 0.4% | 99% | +0.330 | +0.057 |
| psL_seq_o1_s17_k1 -> psL_seq_o1_s17_k2 | +1.300 | +1.299 | +0.001 [-0.004, +0.006] | +1.300 | 0.1% | 100% | -1.306 | -0.019 |
| psL_seq_o1_s17_k2 -> psL_seq_o1_s17_k3 | -1.202 | -1.201 | -0.002 [-0.007, +0.004] | -1.202 | 0.1% | 99% | +1.230 | +0.045 |
| psL_seq_o1_s17_k3 -> psL_seq_o1_s17_k4 | +1.703 | +1.702 | +0.001 [-0.003, +0.006] | +1.702 | 0.1% | 98% | -1.628 | -0.109 |

Class-conditional spread (equal-variance check [A2]) and the logit-level 'parse failure' (argmax is neither ' Yes' nor ' No'):

| stage | mean g|yes | mean g|no | sd g|yes | sd g|no | yes-rate(g>0) | argmax other | (g>0)==argmax |
|---|---|---|---|---|---|---|---|
| base | +1.481 | -1.501 | 0.989 | 1.012 | 0.506 | 0.00% | 1.0000 |
| psL_seq_o1_s17_k1 | +1.180 | -1.801 | 0.990 | 1.013 | 0.462 | 0.00% | 1.0000 |
| psL_seq_o1_s17_k2 | +2.480 | -0.502 | 0.990 | 1.014 | 0.656 | 0.00% | 1.0000 |
| psL_seq_o1_s17_k3 | +1.278 | -1.703 | 0.990 | 1.013 | 0.473 | 0.00% | 1.0000 |
| psL_seq_o1_s17_k4 | +2.980 | -0.001 | 0.990 | 1.012 | 0.740 | 0.00% | 1.0000 |

## (b) Post-hoc scalar correction (10% stratified calibration split, 3 split seeds; metrics on the other 90%; mean +- sd over splits)

| stage | b* | F1 base | F1 uncorr | F1 corr | dF1 [95% CI, split 0] | F1 corr (label-free b_mean) | c uncorr | c corr | c base | d' uncorr | d' corr | acc uncorr | acc corr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| psL_seq_o1_s17_k1 | +0.302+-0.019 | 0.9321 | 0.9246+-0.0012 | 0.9304+-0.0006 | +0.0047 [-0.0014, +0.0121] | 0.9311 | +0.289 | -0.039 | -0.038 | 3.036 | 2.955 | 0.9274 | 0.9300 |
| psL_seq_o1_s17_k2 | -1.008+-0.033 | 0.9321 | 0.8607+-0.0012 | 0.9331+-0.0008 | +0.0710 [+0.0583, +0.0826] | 0.9339 | -1.027 | -0.018 | -0.038 | 3.010 | 3.000 | 0.8390 | 0.9330 |
| psL_seq_o1_s17_k3 | +0.174+-0.004 | 0.9321 | 0.9312+-0.0012 | 0.9313+-0.0012 | -0.0010 [-0.0062, +0.0057] | 0.9307 | +0.217 | -0.020 | -0.038 | 3.069 | 2.970 | 0.9331 | 0.9311 |
| psL_seq_o1_s17_k4 | -1.504+-0.011 | 0.9321 | 0.8056+-0.0013 | 0.9319+-0.0023 | +0.1266 [+0.1117, +0.1403] | 0.9323 | -1.418 | -0.035 | -0.038 | 2.938 | 2.980 | 0.7591 | 0.9316 |

Reading: a training-time method that fixes criterion drift must beat `F1 corr` (one scalar, 10% labeled calibration, zero training). If `F1 corr` ~ `F1 base` the drift is fully bias-removable at inference time; the gap `F1 base - F1 corr` is the non-additive (d') damage no scalar can undo.
