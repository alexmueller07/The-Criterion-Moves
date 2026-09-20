# Diagnostic A11-extended: criterion drift vs training answer statistics

**Run 2026-08-30** by `analysis/diag/criterion_vs_stats.py` (machine-parseable
numbers in `analysis/diag/criterion_vs_stats.json`). Pre-specified in
`design_notes/analysis_ideas.md` (A11, extended with A1's hit/false-alarm
primitives). Diagnostic only: nothing here touches the gate
(`compare_endpoints.py`), and per the asymmetry rule these findings can
invalidate a gate-passed claim but cannot rescue a failed one.

**Instrument audit.** All POPE quantities computed from `pope_gen.jsonl` with
the gate's own parser (`pilot/metrics_pope.parse_yn`). Parse-fail count is **0
at every one of the 9 checkpoints** (9,000/9,000 parsed), so no
shifting-denominator or imputation-bound caveats apply (A2 is trivially clean
for this analysis). Recomputed adversarial F1 matches `pope.json` to 4 decimals
at all checkpoints. GT is balanced 1,500 yes / 1,500 no per split.

**Ground-truth inputs.** Training-set answer statistics were computed at
source from the actual `train.jsonl` files on the cluster (verified
2026-08-30) and are hard-coded in the script: scienceqa n=5,700, yes/no/una
0/0/0%, mean target 1.00 words (bare MC letters); textvqa n=8,000,
yes 5.1% / no 1.5% / una 2.2%, 1.56 words; flickr n=8,000, 0/0/0%, 17.93 words
(captions); vizwiz n=3,800, yes 2.4% / no 2.8% / una 43.2%, 1.41 words.
Stage order: scienceqa(S1) → textvqa(S2) → flickr(S3) → vizwiz(S4).

**Scope caveats.** n = 1 seed, 1 task order; all deltas descriptive.
Approximate eval-noise scale (binomial, clustering ignored): SE(yes_rate)
≈ 0.0053 (n=9,000), SE(CHAIR_i) ≈ 0.0059 (~3,740 mentions). d′/c are
descriptive monotone recodings of the (H, FA) pair (A1's
name-the-divergence note); (H, FA) is the primitive throughout.

---

## 1. Per-checkpoint signal-detection panel

| ckpt | n | parse_fail | yes_rate | hit | FA | d' | c | adv_F1 | CHAIR_i | CHAIR_s |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 | 9000 | 0 | 0.4127 | 0.7711 | 0.0542 | 2.3477 | 0.4314 | 0.8272 | 0.1564 | 0.4840 |
| S1 | 9000 | 0 | 0.4319 | 0.7989 | 0.0649 | 2.3526 | 0.3387 | 0.8347 | 0.1372 | 0.4820 |
| S2 | 9000 | 0 | 0.5074 | 0.8713 | 0.1436 | 2.1972 | -0.0341 | 0.8312 | 0.1443 | 0.4940 |
| S3 | 9000 | 0 | 0.4638 | 0.8364 | 0.0911 | 2.3139 | 0.1770 | 0.8392 | 0.1504 | 0.5040 |
| S4 | 9000 | 0 | 0.3501 | 0.6740 | 0.0262 | 2.3905 | 0.7442 | 0.7819 | 0.1273 | 0.4560 |
| J1 | 9000 | 0 | 0.3520 | 0.6791 | 0.0249 | 2.4271 | 0.7483 | 0.7877 | 0.1494 | 0.5020 |
| J2 | 9000 | 0 | 0.5259 | 0.8818 | 0.1700 | 2.1381 | -0.1149 | 0.8231 | 0.1650 | 0.5480 |
| J3 | 9000 | 0 | 0.5039 | 0.8669 | 0.1409 | 2.1881 | -0.0177 | 0.8290 | 0.1501 | 0.4840 |
| J4 | 9000 | 0 | 0.5042 | 0.8669 | 0.1416 | 2.1852 | -0.0192 | 0.8295 | 0.1561 | 0.5020 |

**Finding.** Across all 9 checkpoints, sensitivity is nearly constant — d′
spans 2.14–2.43 (range 0.29) and is *highest* at S4 and J1 — while the
criterion c swings from −0.11 to +0.74. Every large move in yes_rate is a
joint move of hit and FA in the same direction (criterion shift), never a
hit-down/FA-up move (sensitivity loss). The S4 adv-F1 drop (0.827 → 0.782) is
a recall drop from a conservative criterion, with FA *halving* (0.054 →
0.026): by the yes-false-positive definition the endpoint model hallucinates
*less* on POPE, exactly A1's outcome (d). CHAIR_i at S4 is also the lowest
value in the table (0.1273 vs S0's 0.1564), alongside the shortest captions
(mean 104.8 new tokens vs 108–117 elsewhere), consistent with the same
conservatism reaching the generative instrument.

## 2. Answer-statistics dose-response (per-stage deltas, SEQ arm)

Δ columns are S_k − S_{k−1}; train-stat columns describe stage k's diet.
n = 4 stages: **pattern description only, no correlation coefficient.**

| stage | task | yes% | no% | yes−no | una% | mean_words | ΔYes | ΔFA | ΔHit | ΔAdvF1 | ΔCHAIR_i |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | scienceqa | 0.000 | 0.000 | 0.000 | 0.000 | 1.00 | +0.0192 | +0.0107 | +0.0278 | +0.0075 | -0.0192 |
| S2 | textvqa | 0.051 | 0.015 | +0.036 | 0.022 | 1.56 | +0.0756 | +0.0787 | +0.0724 | -0.0035 | +0.0071 |
| S3 | flickr | 0.000 | 0.000 | 0.000 | 0.000 | 17.93 | -0.0437 | -0.0524 | -0.0349 | +0.0080 | +0.0061 |
| S4 | vizwiz | 0.024 | 0.028 | -0.004 | 0.432 | 1.41 | -0.1137 | -0.0649 | -0.1624 | -0.0574 | -0.0231 |

**Observed relationship, stated plainly.** The eval-time criterion moves in
the direction of the current stage's answer statistics, and by an amount
ordered like the strength of that stage's yes/no-relevant supervision:

- The only stage with a net **yes** tilt (textvqa, yes:no ≈ 3.4:1) produces
  the only large **upward** yes-rate move (+0.076, ~14× SE).
- The stage with a slight **no** tilt plus a massive abstention mass (vizwiz,
  una 43.2%) produces the largest move in the table, **downward** (−0.114,
  ~21× SE), landing 6 points *below* the base model.
- The two stages with **zero** yes/no/unanswerable content move the criterion
  least: scienceqa +0.019, and flickr −0.044 — the flickr move is a *decay
  back toward baseline* of the textvqa push (it gives back ~58% of S2's
  +0.076), which answer statistics alone (0 = 0 for both stages) do not
  distinguish; the natural reading is that criterion pressure requires
  ongoing yes/no supervision and relaxes without it. This is the one part of
  the ordering (|S3| > |S1|) the dose variable itself does not predict.
- Magnitude ordering |ΔYes|: vizwiz (0.114) > textvqa (0.076) > flickr
  (0.044) > scienceqa (0.019), matching the ordering of
  (una% + |yes−no|): 0.436 > 0.058 > 0 = 0 (with the S3/S1 tie broken by the
  decay dynamic above).
- Target length shows no comparable relationship: the 17.9-word caption stage
  produces a smaller criterion move than either short-answer stage with
  yes/no content, so mean target length is not the driver; the yes/no/una
  composition is.
- ΔHit and ΔFA co-move (same sign) at every stage — all four transitions are
  criterion shifts, none is a sensitivity event.

## 3. JOINT-arm internal trajectory

Identical optimizer steps to the SEQ boundaries
(`sb_train_joint.sbatch --checkpoint_steps 90,215,340` + final ≈ 399;
J1 = step 90 ≈ 22.6% of training, the "~25%" checkpoint).

| ckpt | step | frac_train | yes_rate | hit | FA | adv_F1 | CHAIR_i | vizwiz_abstain* |
|---|---|---|---|---|---|---|---|---|
| J1 | 90 | 0.226 | 0.3520 | 0.6791 | 0.0249 | 0.7877 | 0.1494 | 0.650 |
| J2 | 215 | 0.539 | 0.5259 | 0.8818 | 0.1700 | 0.8231 | 0.1650 | 0.530 |
| J3 | 340 | 0.852 | 0.5039 | 0.8669 | 0.1409 | 0.8290 | 0.1501 | 0.556 |
| J4 | 399 | 1.000 | 0.5042 | 0.8669 | 0.1416 | 0.8295 | 0.1561 | 0.566 |

\* auxiliary: fraction of the 500 VizWiz val outputs containing
"unanswerable" (substring proxy; S0 = 0.556, S4 = 0.574).

**Flat after J2 — yes.** Max deviation from the J2–J4 mean: yes_rate 0.0146
(~2.7 SE), hit 0.0099, FA 0.0192, adv_F1 0.0041, CHAIR_i 0.0079 (~1.3 SE).
From half of training onward the joint model holds one stable mixed-diet
criterion (c ≈ −0.05 ± 0.05) with essentially unchanged sensitivity and
CHAIR. The mixed diet's pooled answer statistics (item-weighted: yes ≈ 2.0%,
no ≈ 0.9%, una ≈ 7.1% — a mild net-yes tilt in the yes/no pair) sit on the yes side, and the
settled criterion sits correspondingly slightly yes-of-base (0.504 vs 0.413).

**J1 is strongly anomalous — and it is S4's signature.** J1 sits
−0.159 yes-rate (~30× SE), −0.193 hit, −0.126 FA, −0.039 adv-F1 below the
J2–J4 mean, but only −0.008 on CHAIR_i (~1.3 SE — CHAIR barely participates).
Compared with S4 directly:

| metric | J1 | S4 | \|J1−S4\| | S0 (base) |
|---|---|---|---|---|
| yes_rate | 0.3520 | 0.3501 | 0.0019 | 0.4127 |
| hit | 0.6791 | 0.6740 | 0.0051 | 0.7711 |
| FA | 0.0249 | 0.0262 | 0.0013 | 0.0542 |
| criterion c | 0.7483 | 0.7442 | 0.0041 | 0.4314 |
| d′ | 2.4271 | 2.3905 | 0.0366 | 2.3477 |
| adv_F1 | 0.7877 | 0.7819 | 0.0059 | 0.8272 |
| vizwiz_abstain | 0.650 | 0.574 | 0.076 | 0.556 |

J1 reproduces S4's conservatism signature in (H, FA) space to within 0.006 on
every POPE quantity — two checkpoints from different arms, different data
diets, and different training durations land on the same (low-H, very-low-FA,
high-c, base-level-d′) point. **Hypothesis assessment (exploratory, n=1):**

- *Pure warmup / undertraining* predicts J1 between S0 and the settled J2
  state (interpolation). J1 instead lies **outside** the S0–J2 interval on
  every metric — an overshoot below base — so undertraining alone is
  inconsistent with the data.
- *Early exposure to VizWiz-style conservative targets* predicts an overshoot
  in exactly the observed direction: short, high-frequency, low-entropy
  targets ("no", "unanswerable", one-word answers) are the fastest to fit,
  so early joint training transiently over-weights the abstention/no mode
  before grounded yes-answers are learned and the criterion relaxes to the
  diet's net tilt. The auxiliary column supports this: J1's VizWiz
  abstention-output rate (0.650) is the highest of all 9 checkpoints —
  above S4 itself (0.574), base (0.556), and J2–J4 (0.530–0.566).
- The two are not mutually exclusive (LR warmup means early-step gradients
  dominate the step-90 weights, which is *how* the fast-learned conservative
  mode gets expressed), but the direction and the abstention spike say the
  content of the transient is VizWiz-style conservatism, not generic
  under-adaptation. Full study: an extra joint checkpoint at ~step 45 and a
  joint arm with VizWiz held out would separate these cleanly.

## 4. Tuning-volume vs sequencing decomposition (A11)

S4−S0 = (J4−S0) + (S4−J4); "volume" = shared-with-joint component,
"seq-specific" = what only the sequential arm does. Shares are reported only
when both components carry the total's sign.

| metric | S4−S0 (total) | J4−S0 (volume) | S4−J4 (seq-specific) | volume share | note |
|---|---|---|---|---|---|
| yes_rate | -0.0626 | +0.0916 | -0.1541 | – | components oppose; shares undefined |
| hit | -0.0971 | +0.0958 | -0.1929 | – | components oppose |
| FA | -0.0280 | +0.0873 | -0.1153 | – | components oppose |
| criterion c | +0.3129 | -0.4506 | +0.7635 | – | components oppose |
| d′ | +0.0427 | -0.1626 | +0.2053 | – | components oppose |
| adv_F1 | -0.0454 | +0.0023 | -0.0476 | – | volume ≈ 0 (opposite sign) |
| CHAIR_i | -0.0291 | -0.0003 | -0.0288 | 0.01 | ~99% seq-specific |
| CHAIR_s | -0.0280 | +0.0180 | -0.0460 | – | components oppose |

**Finding.** Sequential drift is not a scaled-up version of joint drift — on
the criterion metrics the two components have **opposite signs**. Tuning
volume alone (J4−S0) pushes the criterion toward *yes* (+0.092 yes-rate,
tracking the mixed diet's net yes tilt); the sequential-specific component
(−0.154) overrides it in the *no* direction, tracking the final stage's
abstention-heavy statistics. Adv-F1 decomposes as ~0 volume, ~100%
sequential-specific; CHAIR_i as 1% / 99%. So on this pilot, essentially
nothing of the sequential endpoint movement is a tuning-volume effect — but
the sequential-specific component is itself criterion placement (d′ moves by
+0.04 total while c moves by +0.31), i.e. **last-stage dominance**, not
accumulated grounding damage. Note the direction: the endpoint moved
*conservative*, so both "hallucination" instruments improve at S4
(FA 0.054→0.026, CHAIR_i 0.156→0.127); the A11 novelty-kill scenario
("J rises like S") is absent, but so is the hoped-for sequential rise —
what the sequence manipulates is the criterion, in either direction.

## Paper-framing implication (4 sentences)

The pilot's discriminative trajectory is not a grounding trajectory: d′ is
flat to slightly rising across all nine checkpoints while the answer
criterion swings by ~0.9 in c-units, and every per-stage move is predicted in
sign (and largely in magnitude ordering) by the current stage's yes/no/
unanswerable answer statistics. This motivates naming the third axis of the
stability–plasticity trade-off as **criterion stability** — a VLM in a task
sequence carries its most recent stage's answer priors into every subsequent
faithfulness measurement, so POPE-style hallucination scores measure "where
the last diet put the criterion" rather than cumulative perceptual
degradation, and can just as easily *improve* (as at S4) as worsen. The joint
arm makes the mechanism legible: a mixed diet holds one stable criterion from
mid-training onward, while its early checkpoint transiently reproduces the
VizWiz-conservatism signature to within 0.006, showing that answer-prior
drift is fast, content-specific, and reversible rather than a slow structural
degradation. For the full study this yields a falsifiable prediction —
reordering tasks should move the endpoint criterion to the new final stage's
answer statistics while leaving d′ unchanged — and a design obligation: any
claimed hallucination trajectory must be reported alongside (H, FA) so that
criterion motion cannot masquerade as grounding change in either direction.

---

*Relation to other pre-specified analyses:* this instantiates A1's outcome
(d) (criterion drifts toward *no* at S4 via VizWiz abstention training —
ARTIFACT for a "hallucination worsens" reading of adv-F1) and A11's outcome
(a)-with-a-twist (J flat ⇒ volume does not explain SEQ movement, but the SEQ
movement is criterion, not grounding). A7 (split-resolved FPR) and A9's
VizWiz abstention-drift panel should be read next for convergence; the
CHAIR-side interpretation (shorter captions at S4) awaits A3's
length-controlled CHAIR before any generative claim is scoped.
