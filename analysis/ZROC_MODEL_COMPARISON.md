# Replacing the R² proxy: a nested model comparison and an ML binormal fit

**Date:** 2026-09-13. **Scope:** LLaVA-1.5-7B, UCIT, 9 sequential + 9 anchor + 2 JOINT cells,
six checkpoints each. **Compute:** zero GPU — everything here runs off
`analysis/readout/fs_aggregate.json`.

**Scripts:** `analysis/zroc_model_comparison.py` (Task 1),
`analysis/zroc_binormal_ml.py` (Task 2, and the shared likelihood machinery).
**Machine output:** `analysis/readout/zroc_model_comparison.json`,
`analysis/readout/zroc_binormal_ml.json`. Nothing under `paper/` was touched.

Both scripts re-derive the certified OLS fit and **assert** it against
`readout/zroc_coherence.json` before emitting anything, reconstruct the integer
binomial counts and assert they round-trip to the published rates, and
cross-check the fast Newton inner solve against a bounded-Brent reference.

---

## Headline

1. **The one-ROC model survives the comparison for sequential, and the anchor is
   not convicted by it.** By AIC, M0 (one ROC) beats M1 (per-checkpoint
   sensitivity) in **9/9 sequential, 2/2 JOINT, 6/9 anchor** cells. The
   likelihood-ratio test rejects M0 in **0/20 cells**, anchor included.
2. **The arm contrast is real and survives.** Mean G² is 2.750 (sequential) vs
   5.827 (anchor), a difference of **+3.077, 95% CI [+1.017, +5.146]** — the
   same direction as the R² gap, now against a named alternative with a null
   distribution attached.
3. **The sharpest discriminator is the directed test**, not the saturated one:
   sensitivity drift along the sequence is significant in **0/9 sequential** and
   **3/9 anchor** cells (arm-level exceedance **p = 0.0084**).
4. **OLS attenuation is confirmed in 20/20 cells and is small for sequential**
   (−0.0128, ~1.8% of the slope). OLS is conservative exactly as the design note
   claims.
5. **⚠ The anchor's slope is not identified.** Its ML slope moves +0.373 on
   average — **4.7× more than attenuation explains** — and its 95% profile
   interval excludes 1 in only 1/9 cells, running to the search ceiling in one.
   `b = 0.546` and `b = 1.79` are both unquotable for the anchor.
6. **⚠ The anchor's d_a deficit is an OLS artifact and must be retracted.** Under
   ML all three arms sit level (2.165 / 2.175 / 2.187). The *spread* — which the
   canon note already says is what carries the argument — survives intact.

---

## Task 1 — nested model comparison

### The models, and why M1 is the saturated model

Each checkpoint contributes exactly two free binomial cells (one hit rate, one
false-alarm rate), so a cell of K = 6 checkpoints carries **2K = 12 free
probabilities**.

| model | description | parameters (K=6) |
|---|---|---|
| **M0** — one ROC | single shared `(a, b)`; each checkpoint its own criterion | `a, b, x_1..x_K` = **K+2 = 8** |
| **M1** — drifting sensitivity | per-checkpoint sensitivity; points need not lie on one curve | **2K = 12** |
| **M1_trend** — directed | `a_k = a0 + g·(k−k̄)`, shared `b`, own criteria | **K+3 = 9** |

Letting sensitivity vary per checkpoint **saturates the design**. Writing
per-checkpoint `(a_k, b_k, x_k)` would be 3K = 18 parameters for 12 observations;
the extra 6 are not estimable, and the fitted probabilities collapse to the
observed rates. So M1 *is* the saturated model at 2K parameters. With one
operating point per checkpoint there is no identified intermediate — which is
itself worth stating in the paper, because it is the precise sense in which six
checkpoints cannot separate "criterion slid" from "everything changed" without a
structural assumption.

Nesting is genuine: M0's fitted probabilities are a point in the same 12-dim
space M1 ranges over freely, and M0 ⊂ M1_trend ⊂ M1. Hence **df = 2K−(K+2) = 4**
for M1 vs M0 and **df = 1** for M1_trend vs M0.

M1_trend exists because the saturated alternative spends four degrees of freedom
on arbitrary scatter, while the substantive worry is specifically that
sensitivity **decays along the task sequence**. One degree of freedom aimed at
that is a far more powerful test.

### Is χ²₄ the right reference? (checked, not assumed)

Parametric bootstrap under the fitted M0, 300 replicates per cell:

| arm | simulated null mean G² (df = 4 expected) | simulated 95th pct (χ²₄ = 9.488) |
|---|---|---|
| sequential | 3.986 | 9.138 |
| anchor | 3.804 | 8.974 |
| JOINT | 3.808 | 8.541 |

χ²₄ is well calibrated here and very slightly **conservative**, so the asymptotic
p-values below understate significance marginally. The asymptotics are in trials
per operating point (~9000), not in the number of operating points, which is why
six points is not a problem for the *test* even though it is a weak basis for R².

### Results

| arm | n | AIC favours M0 | M0 rejected (p<.05) | mean G² | 95% CI | mean R² (OLS) |
|---|---|---|---|---|---|---|
| sequential | 9 | **9/9** | 0/9 | 2.750 | [1.542, 4.058] | 0.9569 |
| anchor | 9 | **6/9** | 0/9 | 5.827 | [4.238, 7.435] | 0.5857 |
| JOINT | 2 | **2/2** | 0/2 | 0.394 | *per-cell 0.511, 0.276 — no interval at n=2* | 0.9446 |

The three anchor cells where AIC prefers M1 are `o2/s31` (ΔAIC −1.46, G² 9.46,
p = 0.051), `o3/s17` (−1.09, p = 0.059) and `o3/s23` (−0.30, p = 0.081) — all
marginal, none rejecting at 5%.

**Directed test (M1_trend, df = 1):**

| arm | cells with significant sensitivity drift | arm-level exceedance p |
|---|---|---|
| sequential | **0/9** | 1.0000 |
| anchor | **3/9** (`o2/s17` p=.018, `o2/s31` p=.007, `o3/s23` p=.040) | **0.0084** |
| JOINT | 0/2 | 1.0000 |

Under the null each cell rejects with probability 0.05, so the count is
Binomial(n, 0.05); 3/9 sits at p = 0.0084 and 0/9 is exactly what the null
predicts. This is the cleanest statement in Task 1: **sequential shows no
sensitivity drift at all, and the anchor does.**

### What this does and does not license

- **It defuses the circularity objection.** The null and the alternative are now
  explicit models, so "the points lie on one ROC" is a claim the data could have
  refused. For sequential it did not refuse.
- **It does *not* convict the anchor per cell.** The LR test rejects one ROC in
  0/9 anchor cells. Any sentence of the form "the anchor's points scatter off any
  single curve" is **not supported** and should be cut. What is supported: the
  anchor's departure from one ROC is about **2.1× sequential's**, with a
  difference CI excluding zero, and its sensitivity drift is arm-level
  significant.

---

## Task 2 — maximum-likelihood binormal fit

### The estimator

OLS regresses `z(H)` on `z(FA)`, but `z(FA)` is itself estimated, so the
regressor carries error and the slope is attenuated toward zero (Pesce & Metz
2007). The ML fit treats each criterion `x_k = z(FA_k)` as a free parameter and
maximises the two binomial log-likelihoods jointly over `(a, b, x_1..x_K)`:

```
FA_k = Φ(x_k)          H_k = Φ(a + b·x_k)
```

Because `x_k` is informed by **both** rates rather than read off `z(FA_k)` alone,
this is the errors-in-variables fit OLS approximates. It is **not** the
Dorfman–Alf rating-category MLE and must not be called that — we have no latent
rating table, only K independent (FA, Hit) pairs. It is the MLE for *our* design.

### Attenuation: direction confirmed, magnitude measured

Simulating from each cell's own fit at its own sample sizes and operating range
(4000 replicates/cell; ground truth = the OLS fit with criteria at the observed
`z(FA)`, which makes this a **lower bound** on attenuation, since the observed
spread is already noise-inflated):

| arm | n | mean OLS bias | 95% CI (bootstrap over cells) | negative in |
|---|---|---|---|---|
| sequential | 9 | **−0.0128** | [−0.0176, −0.0086] | 9/9 |
| anchor | 9 | **−0.0789** | [−0.1086, −0.0545] | 9/9 |
| JOINT | 2 | −0.0855 | *no interval at n=2* | 2/2 |

**OLS is biased downward in 20/20 cells.** Since the reading turns on `b < 1`,
**OLS is conservative**, quantitatively: for sequential the bias is ~1.8% of the
slope, so the design note's claim is correct and cheap to state.

### OLS vs ML slopes, per arm

| arm | mean b_OLS | mean b_ML | mean Δ | Δ ÷ attenuation | slope identified | CI excludes 1 |
|---|---|---|---|---|---|---|
| sequential | 0.7155 [0.664, 0.758] | 0.7289 [0.678, 0.771] | **+0.0134** [0.006, 0.023] | **1.05×** | 9/9 | **7/9** |
| anchor | 0.5457 [0.415, 0.666] | 0.9187 [0.680, 1.200] | **+0.373** [0.135, 0.717] | **4.73×** | 8/9 | **1/9** |
| JOINT | 0.574 / 0.765 *(per cell)* | 0.589 / 0.775 *(per cell)* | +0.015 / +0.011 | 0.15× | 2/2 | 0/2 |

**Sequential: the correction is exactly attenuation.** The observed OLS→ML shift
(+0.0134) matches the independently simulated attenuation (−0.0128) to within 5%.
That is a strong mutual validation of both estimates, and it means the
sequential slopes need no revision: `b` moves from 0.716 to 0.729, and the
profile interval excludes 1 in 7/9 cells. **The `b < 1` reading survives the
better estimator.**

**⚠ The anchor: non-identification, not correction.** Its shift is 4.7× larger
than attenuation can explain, and the profile intervals say why:

| cell | b_OLS | b_ML | 95% profile-LR CI |
|---|---|---|---|
| o1/s17 | 0.490 | 0.748 | [0.36, 2.11] |
| o1/s31 | 0.677 | 1.349 | **[0.66, >6.00]** — unbounded above |
| o2/s17 | 0.598 | 0.870 | [0.50, 1.89] |
| o2/s31 | **0.192** | **1.791** | [0.49, 5.13] |
| o3/s23 | 0.478 | 0.674 | [0.39, 1.25] |

The anchor's operating range is compressed enough that **no estimator can recover
its slope from these counts**. `b_ML = 1.791` is not an estimate to quote; it is
the likelihood wandering on a near-flat ridge. The honest statement is:

> The anchor's z-ROC slope is not estimable from six operating points at this
> operating range. Neither 0.546 nor 1.79 may be quoted as its variance ratio.

This is a *stronger* result than "the anchor's geometry is distorted", and it
reaches the paper's existing conclusion by a different and more defensible route:
the anchor arm is not merely off one curve, it is **under-determined**.

### ⚠ Consequence: the anchor's d_a deficit does not survive

| arm | d_a (OLS) | d_a (ML) |
|---|---|---|
| sequential | 2.1587 ± 0.0402 | 2.1649 ± 0.0383 |
| anchor | **1.7884 ± 0.3618** | **2.1745 ± 0.3536** |
| JOINT | 2.1812 ± 0.0423 | 2.1869 ± 0.0395 |

`d_a = √(2/(1+b²))·a` inherits the slope's instability. Under ML the anchor's
**mean** d_a is level with the other arms (2.175 vs 2.165 and 2.187) — the
apparent sensitivity deficit was an artifact of an attenuated slope. The
**spread** (0.354 vs 0.038, ~9×) survives the change of estimator essentially
unchanged.

This is a clean split, and it happens to vindicate `ZROC_CANON_CONSTRAINTS.md`
item 4, which already insists the SD and not the mean is what carries the
isosensitivity argument. Concretely:

- **Retract** any reading of anchor mean d_a = 1.788 as lower sensitivity.
- **Keep** the d_a spread contrast; it is estimator-robust.
- This **reinforces the existing hold** on recoverable-headroom numbers: the
  anchor's corrected ceiling is a function of `b`, and `b` is not identified for
  the anchor, so neither direction is quotable. The project-level hold stands
  for a second, independent reason.

### A side finding: R² is fragile where it is load-bearing

Re-deriving the identical OLS fit from exact counts rather than the published
4-dp rates moves R² by up to **0.00527 on the anchor** (vs 0.00097 sequential).
A fifth of a percent of the R² gap the paper leans on is rounding noise, because
the anchor's `SS_tot` is small. Slopes are unaffected (<5e-3, asserted). This is
a minor point but it is one more reason not to let R² carry the claim alone.

---

## Verdict: replace or supplement?

**Supplement in the appendix, but re-lead the main text with the model
comparison.** Specifically:

1. **Lead the primary claim with the nested comparison, not R².** "AIC favours a
   single shared ROC in 9/9 sequential and 2/2 JOINT cells; the likelihood-ratio
   test never rejects it" is the sentence that answers the circularity objection,
   because it names the alternative. R² cannot do that at any value.
2. **Quote the arm contrast with its interval** (ΔG² = +3.077, [+1.017, +5.146]),
   and the directed trend result (0/9 vs 3/9, arm p = 0.0084), which is the
   sharpest single discriminator we have.
3. **Keep R² as a descriptive readout** beside residual RMSE (already project
   policy). Readers know it, and dropping it would look evasive. But it should be
   labelled a descriptive goodness-of-fit summary, with the model comparison
   cited as the inferential statement.
4. **Weaken the anchor sentence.** The LR test does not reject one ROC for any
   anchor cell. Replace "the anchor's points leave a single curve" with the
   non-identification finding, which is both true and stronger.
5. **Report ML alongside OLS for the slopes**, stating that OLS attenuates,
   that the attenuation is measured at −0.013 for sequential, and that it is
   conservative for a `b < 1` claim. Two lines, and it pre-empts the sharpest
   available methodological hit.
6. **Retract the anchor's mean d_a deficit** and keep the spread.

The honest summary for the paper is that the model comparison **supports the
primary claim more defensibly than R² did** while **weakening the secondary
anchor claim** — and the ML fit **leaves sequential's numbers essentially intact**
while showing the anchor's were never estimable.

---

## Limitations

- **The independence assumption is violated in two directions and the net cannot
  be signed.** The binomial likelihood treats a checkpoint's 9000 POPE answers as
  9000 independent Bernoulli trials. (a) *Within* a checkpoint, POPE's three
  splits share their positive items — verified byte-identical in this project's
  own stratum audit, which is what retracted the stratum "double dissociation" —
  so the 4500 positive trials are roughly 1500 distinct items asked three times,
  and the true variance of H exceeds Binomial(4500, ·). That **inflates** G².
  (b) *Across* checkpoints, all six are scored on the **same** 9000 items, so
  their item-level idiosyncrasies are common-mode and the six points scatter
  around a common curve **less** than independent binomials would. That
  **deflates** G². Settling this needs per-item responses, which are not on disk
  for these cells: **NO-DATA, not clearance.** Prefer the arm contrast, which is
  internal and carries both flaws equally, to any single cell's p-value.
- **JOINT is n = 2 cells.** Every JOINT number here is quoted per cell. No
  interval is given for it anywhere, and none should be.
- **Six checkpoints, three models.** M0 at 8 parameters against 12 observations
  is not a lavish design; the LR test is well calibrated (checked above) but the
  *power* against non-trend alternatives is modest. A non-rejection is not proof
  of a single ROC — it is failure to refute one.
- **The design remains non-standard.** These are six different models, not one
  observer sliding a criterion. The scope sentence in
  `ZROC_CANON_CONSTRAINTS.md` still applies verbatim and nothing here relaxes it.
- **Bootstraps resample cells as a list with multiplicity** (20 000 reps,
  asserted in code — a `set()` here previously cost this project nine intervals
  that were ~24% too narrow).
