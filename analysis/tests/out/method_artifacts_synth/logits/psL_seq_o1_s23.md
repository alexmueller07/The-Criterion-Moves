# Logit-level additive-bias test: psL_seq_o1_s23

Base: `/Users/alexandermueller/Research/cl-hallucination-vlm/analysis/tests/fixtures/results_method_artifacts_synth/logits/llava15_base/pope_logits.jsonl`; stages: 4; joined ids: 3000 (base rows 3000; per-file drops: {'base': 0, 'psL_seq_o1_s23_k1': 0, 'psL_seq_o1_s23_k2': 0, 'psL_seq_o1_s23_k3': 0, 'psL_seq_o1_s23_k4': 0}). Bootstrap n=200, seed=0.

Theory (Claim 1): a stage shifts g by one class-independent constant -> `diff` ~ 0 (CI covers 0 or |diff| << |b_common|), `nonadd(class)` and `nonadd(z)` small, `dd'` ~ 0, H and FA co-move, class sds unchanged. Failure: `shift_yes` != `shift_no` (separation changes = a d' move). `nonadd(class)` large with `nonadd(z)` small means the class SPREAD changed (scale, not location; see the sd table): the equal-variance location model failed, not d'-invariance. `dc undone` is the feasibility check for (b), not a theory test: on balanced data the pooled offset restores the midpoint c even when the separation changed.

## (a) Additive-bias test vs base (gap units = logits)

| stage | shift_yes | shift_no | diff [95% CI] | b_common | R2_item | nonadd(class) | dc undone | c_gap | dc | d'_gap | dd' | nonadd(z) | H,FA co-move |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | - | - | - | - | - | - | - | -0.042 | - | 2.949 | - | - | - |
| psL_seq_o1_s23_k1 | -0.249 | -0.251 | +0.002 [-0.001, +0.005] | -0.250 | 0.96 | 0.8% | 96% | +0.248 | +0.290 | 3.021 | +0.072 | 19.9% | yes |
| psL_seq_o1_s23_k2 | +1.050 | +1.051 | -0.001 [-0.004, +0.002] | +1.050 | 1.00 | 0.1% | 99% | -1.071 | -1.029 | 2.964 | +0.015 | 1.4% | yes |
| psL_seq_o1_s23_k3 | -0.149 | -0.151 | +0.002 [-0.002, +0.005] | -0.150 | 0.90 | 1.0% | 93% | +0.107 | +0.149 | 2.989 | +0.039 | 20.9% | yes |
| psL_seq_o1_s23_k4 | +1.550 | +1.548 | +0.002 [-0.002, +0.006] | +1.549 | 1.00 | 0.1% | 100% | -1.516 | -1.474 | 2.975 | +0.026 | 1.7% | yes |

Stage-to-stage (k-1 -> k):

| transition | shift_yes | shift_no | diff [95% CI] | b_common | nonadd(class) | dc undone | dc | dd' |
|---|---|---|---|---|---|---|---|---|
| base -> psL_seq_o1_s23_k1 | -0.249 | -0.251 | +0.002 [-0.001, +0.006] | -0.250 | 0.8% | 96% | +0.290 | +0.072 |
| psL_seq_o1_s23_k1 -> psL_seq_o1_s23_k2 | +1.298 | +1.302 | -0.003 [-0.009, +0.002] | +1.300 | 0.2% | 99% | -1.319 | -0.057 |
| psL_seq_o1_s23_k2 -> psL_seq_o1_s23_k3 | -1.199 | -1.202 | +0.003 [-0.003, +0.008] | -1.200 | 0.2% | 100% | +1.178 | +0.024 |
| psL_seq_o1_s23_k3 -> psL_seq_o1_s23_k4 | +1.699 | +1.699 | +0.000 [-0.005, +0.006] | +1.699 | 0.0% | 100% | -1.623 | -0.013 |

Class-conditional spread (equal-variance check [A2]) and the logit-level 'parse failure' (argmax is neither ' Yes' nor ' No'):

| stage | mean g|yes | mean g|no | sd g|yes | sd g|no | yes-rate(g>0) | argmax other | (g>0)==argmax |
|---|---|---|---|---|---|---|---|
| base | +1.481 | -1.501 | 0.989 | 1.012 | 0.506 | 0.00% | 1.0000 |
| psL_seq_o1_s23_k1 | +1.232 | -1.752 | 0.989 | 1.013 | 0.468 | 0.00% | 1.0000 |
| psL_seq_o1_s23_k2 | +2.530 | -0.450 | 0.991 | 1.014 | 0.668 | 0.00% | 1.0000 |
| psL_seq_o1_s23_k3 | +1.331 | -1.652 | 0.990 | 1.013 | 0.486 | 0.00% | 1.0000 |
| psL_seq_o1_s23_k4 | +3.031 | +0.047 | 0.990 | 1.014 | 0.755 | 0.00% | 1.0000 |

## (b) Post-hoc scalar correction (10% stratified calibration split, 3 split seeds; metrics on the other 90%; mean +- sd over splits)

| stage | b* | F1 base | F1 uncorr | F1 corr | dF1 [95% CI, split 0] | F1 corr (label-free b_mean) | c uncorr | c corr | c base | d' uncorr | d' corr | acc uncorr | acc corr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| psL_seq_o1_s23_k1 | +0.248+-0.032 | 0.9321 | 0.9278+-0.0020 | 0.9320+-0.0021 | +0.0047 [-0.0027, +0.0120] | 0.9324 | +0.250 | -0.027 | -0.038 | 3.047 | 2.983 | 0.9301 | 0.9317 |
| psL_seq_o1_s23_k2 | -1.028+-0.055 | 0.9321 | 0.8528+-0.0021 | 0.9304+-0.0011 | +0.0766 [+0.0640, +0.0873] | 0.9309 | -1.074 | -0.045 | -0.038 | 2.983 | 2.958 | 0.8283 | 0.9300 |
| psL_seq_o1_s23_k3 | +0.138+-0.013 | 0.9321 | 0.9322+-0.0015 | 0.9312+-0.0015 | -0.0010 [-0.0058, +0.0037] | 0.9318 | +0.111 | -0.020 | -0.038 | 3.020 | 2.969 | 0.9332 | 0.9310 |
| psL_seq_o1_s23_k4 | -1.556+-0.016 | 0.9321 | 0.7963+-0.0014 | 0.9334+-0.0012 | +0.1362 [+0.1217, +0.1505] | 0.9332 | -1.533 | -0.032 | -0.038 | 3.016 | 3.000 | 0.7444 | 0.9331 |

Reading: a training-time method that fixes criterion drift must beat `F1 corr` (one scalar, 10% labeled calibration, zero training). If `F1 corr` ~ `F1 base` the drift is fully bias-removable at inference time; the gap `F1 base - F1 corr` is the non-additive (d') damage no scalar can undo.
