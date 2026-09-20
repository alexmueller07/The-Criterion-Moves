# Four-Arm Method Comparison (pilot scale) — 2026-08-31

Arms, identical data/steps/schedule: SEQ (plain sequential), JOINT (matched joint control),
ER (Experience Replay, 100 exemplars/prior task — MCITlib convention), F (faithfulness anchor v1,
λ=0.1, k=4, margin=2.0). All 17 checkpoints scored with the identical audited scorers.
Full numbers: analysis/diag/method_comparison.json, computed by analysis/method_comparison.py.

## ER control — the kill-condition did NOT fire (the key method-side result)

Prediction registered in PILOT_FINDINGS.md: if replay fixes the faithfulness phenomena, they are
forgetting-like and the "distinct axis" claim dies. Result: ER tracks SEQ within noise on every
faithfulness metric — criterion swings Σ|Δc| 1.358 vs SEQ 1.243 (same shape, same VizWiz-stage
blowup), stage-4 refusal leakage 12.6% vs 14.6%, CHAIR_i trajectory equivalent, plasticity equal
(0.657 vs 0.659). Replay — the standard remedy for forgetting — leaves criterion drift and
behavior leakage intact. Caveat: 100-exemplar buffer only; a larger-buffer arm is a full-study
robustness item.

## Anchor v1 — predictions falsified; failure mode diagnosed

Predictions (i) shrink criterion swings and (ii) suppress leakage FAILED — inverted: the F arm
swings hyper-conservative immediately (F1: c=+2.02, yes-rate 7.0% vs S1's 0.34 / 43.2%),
Σ|Δc| = 3.430 (2.8× SEQ), F4 leakage 17.8%. Prediction (iv) plasticity HELD (0.657 ≈ SEQ).
CHAIR_i is lower at most stages but with yes-rates of 4-22% this is conservatism, not grounding.
d′ is erratic across stages (1.87-2.73), so the anchor perturbs the decision axis rather than
stabilizing it.

**Diagnosis (visible in the v1 equations):** the absent-object term boosts z_no directly, but the
present-object term is only a real-vs-masked relative margin on z_yes — nothing ever enforces
z_yes > z_no on real present objects. The gradient pressure on the decision axis is one-sided
toward "No". The pilot measured exactly that.

**v2 (implemented, launching):** symmetric decision-axis margins — softplus(m − (z_yes − z_no))
on real/present + softplus(m − (z_no − z_yes)) on real/absent — plus the counterfactual
sensitivity term softplus(m − (z_yes(real) − z_yes(masked))) retained separately. Same data,
same λ/k/m initially. v2 checkpoints labeled G1-G4.

## Anchor v2 (G arm) — criterion stabilization achieved; length-robust generative reduction

Same data/λ/k/m as v1, symmetric decision-axis loss (see faith_loss.py v2 note). Results:
- **Prediction (i) PASS — criterion frozen.** After a one-time settling step to the anchor's
  operating point (S0 c=0.431 → G1 0.838), per-stage |Δc| = 0.120 / 0.157 / 0.107 vs SEQ's
  0.373 / 0.211 / 0.567 and ER's 0.368 / 0.100 / 0.790: the TextVQA yes-push suppressed ~3×,
  the VizWiz abstention-push ~5×. Σ|Δc| 0.791 (G) vs 1.243 (S) / 1.358 (E) / 3.430 (F-v1).
  d′ healthiest of any arm (2.34-2.50). Yes-rate parks at ~33-36% (the m=2.0 equilibrium is
  slightly conservative of base — an offset, not an instability; margin tuning is a
  full-study knob).
- **Generative reduction is real, not length:** length-controlled CHAIR_i@60 at endpoint:
  G4 0.0766 vs S4 0.0894 (gap −0.0128, paired-bootstrap CI95 [−0.0228, −0.0028]) and vs
  J4 0.0904 (−0.0138, CI [−0.0257, −0.0019]) — both exclude zero. G is lowest at every stage
  except G3 (the captioning stage pulls it to 0.1004, matching the S-arm's peak — the anchor
  protects less during caption-heavy training; noted).
- **Prediction (iv) PASS — plasticity intact** (mean new-task 0.656 vs SEQ 0.659; best VizWiz
  score of any arm, 0.723).
- **Prediction (ii) FAIL — leakage untouched** (G4 TextVQA "Unanswerable" rate 16.4% ≈ SEQ's
  14.6%): the anchor supervises the yes/no axis, not the abstention token. Scoped limitation;
  candidate fix is an abstention-aware anchor term (future work / full study).

All method results are exploratory (post-hoc of the measurement gate; 1 seed, 1 ordering).
v1-vs-v2 doubles as the ablation showing the symmetric decision-axis design is what matters.

## Standing verdicts after this round

- Measurement claims are UNAFFECTED (they never depended on the method): criterion drift is real,
  sequencing-specific, dose-dependent; leakage is real; the length-controlled mid-sequence CHAIR
  rise stands; and ER's null now shows all of it is robust to replay.
- Method status: open. v1 falsified (a useful negative + ablation for the paper); v2 is the
  live candidate. If v2 also fails, honest fallbacks: report the measurement + ER-null as the
  contribution with the method as negative results, or explore the C6 EOS/termination-calibration
  term (still novelty-clear per the dossier).
