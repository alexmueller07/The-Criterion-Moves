# Does the criterion drift cost anything measurable at the endpoint? — 2026-09-13

**Script:** `analysis/does_drift_cost.py` · **Machine output:** `analysis/readout/does_drift_cost.json`
**Data:** `analysis/readout/fs_aggregate.json` (generated 2026-09-09T07:32, all 20 LLaVA/UCIT cells). Zero GPU.

Reproduce:

```
python3 analysis/does_drift_cost.py
```

---

## Verdict up front

**Drift costs about 1.2 POPE F1 points, in 9/9 sequential cells, and the cost is real against
both noise floors. So "drift is free" is false and the reframe that provoked this test does not
survive.** But the cost is small, it is *mid-sequence* rather than at the endpoint, and the claim
it was supposed to rescue — "sequential's destination does not move" — turns out to be quoted
**32× more precisely than the data supports**.

Neither the paper's existing framing nor the proposed reframe comes through intact:

1. **The drift is not free.** At each cell's worst-drifted mid-sequence stage, POPE F1 sits
   **0.0118 below that cell's own endpoint F1** (95% CI [+0.0079, +0.0162], positive in **9/9**
   cells, sign test p = 0.004). That is ~2.6× the item-sampling floor (0.0042) and ~2.6× the
   seed-to-seed floor (0.0046). Small, but measured and consistent.
2. **The "destination does not move" claim is an averaging artifact at the precision it is
   quoted.** Sequential's endpoint c is +0.0903 against JOINT's +0.0878 — a gap of 0.0025 — but
   the **mean per-cell absolute error is 0.0811**, a **cancellation ratio of 32.4**. Individual
   endpoints span **[−0.0143, +0.2607]**, a width of 0.275, which is **80% of the base's entire
   mis-placement** (0.3436). Four cells land above c*, five below.
3. **But the migration itself is real.** The stage-index control shows stage 6 genuinely is the
   best-placed stage index: mean |c − c*| falls **0.2012 (k1) → 0.0811 (k6)**, and k6 is the only
   stage where cells straddle c* rather than sitting almost uniformly conservative. The endpoint
   beats its own cell's *typical* earlier stage in **9/9** cells.
4. **It returns; it does not converge.** 0/9 cells approach monotonically; the path is 2.5× the
   net displacement; the endpoint is interior to its own visited range in 6/9 cells; and the
   endpoint beats the *best* earlier stage in only **3/9**. In 6/9 cells the criterion passed
   closer to the target earlier and then moved away again.
5. **The residual endpoint deficit to JOINT is not criterion placement.** Endpoint criterion gap
   +0.0025 (≈0) but endpoint F1 gap −0.0042, with 7/9 sequential cells below JOINT's lower cell.
   With the criterion matched, separability is the only remaining carrier on this axis.

**The framing the data actually supports** is neither "a drift to be fixed" nor "a drift that
self-corrects". It is: *the criterion wanders, ends closer to the multitask value than it
typically was but not at an attractor, costs about a point of F1 on the way, and lands in a
distribution 0.275 wide.* The reason to care is **not** endpoint quality — it is that **you cannot
predict where any single run will land**. See "What this means for how the paper is framed".

---

## Positive controls (all PASS; the script aborts and emits nothing if any fails)

| check | result |
|---|---|
| raw path Σ\|Δc\| == certified `E1_null.sum_abs_step`, 9 seq cells | PASS, max abs diff 0.000000 |
| endpoint c == certified `E1_null.endpoint_c`, 9 seq cells | PASS, max abs diff 0.000000 |
| F1 reconstructed from (H, FA, n_gt_yes, n_gt_no) == recorded `f1` | PASS, 120 stage-points, max diff 0.000093 |
| c* recomputed from JOINT endpoints == prereg +0.088 | PASS, +0.0878 |
| base criterion == recorded +0.4314 | PASS |

Conventions match `analysis/tradeoff_placement.py` / `fs_common.py` exactly: raw path = base → k1 →
… → k6; post-settling drops the first (settling) step. **c\* = +0.0878, the mean of the JOINT arm's
two empirical endpoint criteria (`joint|o1|s17` +0.0623, `joint|o1|s23` +0.1134). n = 2, so those
two values are quoted and no interval is. Never a modelled c = 0.**

A second, unplanned positive control fell out of Q1: the cross-arm correlations below reproduce
`TRADEOFF_PLACEMENT.md`'s numbers (−0.467 pooled, +0.682 with the anchor family dropped) to three
decimals from an independent code path.

---

## Q1 — Does path length predict endpoint quality?

### Within the sequential arm (n = 9 cells)

| x | y | Pearson r | Spearman ρ | 95% CI (cells resampled with multiplicity) | permutation p |
|---|---|---|---|---|---|
| raw Σ\|Δc\| | \|endpoint c − c*\| | **+0.614** | +0.600 | [+0.066, +0.928] | 0.082 |
| post-settling Σ\|Δc\| | \|endpoint c − c*\| | **+0.569** | +0.450 | [−0.081, +0.898] | 0.110 |
| raw Σ\|Δc\| | endpoint POPE F1 | **−0.488** | −0.483 | [−0.939, +0.368] | 0.182 |
| post-settling Σ\|Δc\| | endpoint POPE F1 | **−0.629** | −0.583 | [−0.931, +0.079] | 0.071 |

**All four point the same way: a longer path goes with a worse-placed, lower-F1 endpoint.** None
reaches significance. At n = 9 the critical |r| for p < .05 is **0.666**, and 80% power requires a
true ρ ≥ **0.816**. The observed 0.49–0.63 sit exactly in the band this design cannot resolve.

**This is "underpowered", not "no relationship".** The point estimates are moderate-to-large and
sign-consistent across four related tests. One bootstrap CI (raw path vs endpoint error) excludes
zero, but its permutation p is 0.082; percentile bootstrap CIs for correlations are liberal at
n = 9, so the permutation p is treated as the inferential statistic here and the relationship is
reported as **not established**. What can be said is that the data do **not** support independence
of path and endpoint — if anything they lean the other way, against the free-drift reading.

Per-cell values behind the table:

| cell | raw path | post-settling | endpoint c | \|endpoint c − c*\| | endpoint F1 |
|---|---|---|---|---|---|
| seq\|o1\|s17 | 0.7643 | 0.5451 | +0.0679 | 0.0199 | 0.8604 |
| seq\|o1\|s23 | 0.7376 | 0.7140 | +0.1030 | 0.0151 | 0.8680 |
| seq\|o1\|s31 | 1.2021 | 1.0846 | +0.2607 | 0.1729 | 0.8547 |
| seq\|o2\|s17 | 1.0071 | 0.8998 | −0.0111 | 0.0989 | 0.8632 |
| seq\|o2\|s23 | 1.0193 | 0.9794 | +0.1617 | 0.0739 | 0.8610 |
| seq\|o2\|s31 | 1.1935 | 1.1568 | +0.2021 | 0.1143 | 0.8567 |
| seq\|o3\|s17 | 0.6020 | 0.4295 | +0.0138 | 0.0741 | 0.8641 |
| seq\|o3\|s23 | 0.7861 | 0.5079 | −0.0143 | 0.1021 | 0.8651 |
| seq\|o3\|s31 | 1.1159 | 0.7153 | +0.0289 | 0.0590 | 0.8691 |

### Within the anchor arm (n = 9): flat

r = −0.046 / −0.088 (path vs endpoint error) and +0.226 / +0.326 (path vs endpoint F1); every
permutation p ≥ 0.38, every CI spanning most of [−1, +1]. Uninformative at this n.

### Across arms — and why the pooled number must not be read as n = 20

| set | n cells | post-settling vs \|endpoint c − c*\| | CI | perm p |
|---|---|---|---|---|
| all arms | 20 | **−0.467** | [−0.748, −0.176] | 0.037 |
| anchor family dropped | 11 | **+0.682** | [+0.247, +0.885] | 0.022 |
| JOINT dropped (seq + anchor) | 18 | **−0.678** | [−0.847, −0.501] | 0.001 |
| arm identity partialled out (within-arm centred) | 20 | **+0.247** | [−0.203, +0.689] | 0.298 |

This is **n = 20 cells but n = 3 arms**, and dropping the anchor family does not weaken the pooled
correlation — it **reverses** it. That is the two-group separation `TRADEOFF_PLACEMENT.md` already
refuted, reproduced here from an independent code path. **The pooled −0.467 is not evidence that
drift buys placement.** With arm identity removed, the within-arm relationship is +0.247, CI
spanning zero — i.e. nothing, at this n.

JOINT is dropped in one row because c* is the mean of its two endpoints, so JOINT's placement error
is partly by construction.

### Are "endpoint F1" and "endpoint placement" two readings of one quantity?

Within the sequential arm, endpoint c vs endpoint F1: **r = −0.747** (R² = 0.56, perm p = 0.022).
Substantially shared but not identical — about half the cross-cell variance in endpoint F1 is
endpoint criterion placement. They should not be presented as independent confirmations of each
other.

---

## Q2 — Is the endpoint convergence real, or an artifact of averaging?

### Sequential endpoint criterion, per cell (n = 9)

| cell | endpoint c | signed err (c − c*) | abs err |
|---|---|---|---|
| seq\|o1\|s17 | +0.0679 | −0.0199 | 0.0199 |
| seq\|o1\|s23 | +0.1030 | +0.0151 | 0.0151 |
| seq\|o1\|s31 | **+0.2607** | +0.1729 | **0.1729** |
| seq\|o2\|s17 | −0.0111 | −0.0989 | 0.0989 |
| seq\|o2\|s23 | +0.1617 | +0.0739 | 0.0739 |
| seq\|o2\|s31 | +0.2021 | +0.1143 | 0.1143 |
| seq\|o3\|s17 | +0.0138 | −0.0741 | 0.0741 |
| seq\|o3\|s23 | **−0.0143** | −0.1021 | 0.1021 |
| seq\|o3\|s31 | +0.0289 | −0.0590 | 0.0590 |

- mean endpoint c **+0.0903**, sd **0.0988**, range **[−0.0143, +0.2607]**, width **0.2750**
- **|mean − c*| = 0.0025** but **mean |c − c*| = 0.0811** → **cancellation ratio 32.4**
- 4 cells above c*, 5 below
- 95% CI on the mean endpoint c (cells resampled as a list with multiplicity): **[+0.0331, +0.1545]**
- 95% CI on mean |c − c*|: **[+0.0518, +0.1122]** — excludes zero decisively

**The agreement "+0.0903 vs +0.0878" is 32× tighter than the typical cell.** It exists because
signed errors cancel, not because cells land on the bound. The endpoint spread (mean abs error
0.0811) is **24% of the base's entire distance from the target**, and the full range (0.275) is
**80%** of it. The CI on the mean is 0.121 wide — it contains c*, but it also contains endpoints
0.055 below and 0.067 above it. The mean is **not** pinned to the bound; it is merely consistent
with it.

For comparison, JOINT's two cells (+0.0623, +0.1134) have mean abs error 0.0255 — about 3× tighter
than sequential — but **n = 2, so this is two numbers, not a replicated spread.**

### The stage-index control: is stage 6 actually special?

If every stage index were about as close to c* on average, "the destination does not move" would be
a statement about the arm's mean, not about the endpoint. It is not:

| stage | mean c | sd c | \|mean − c*\| | **mean \|c − c*\|** | n above c* |
|---|---|---|---|---|---|
| 1 | +0.2763 | 0.1265 | 0.1885 | **0.2012** | 8/9 |
| 2 | +0.2182 | 0.0804 | 0.1304 | **0.1441** | 8/9 |
| 3 | +0.1508 | 0.2052 | *0.0630* | **0.1884** | 6/9 |
| 4 | +0.1739 | 0.0660 | 0.0860 | **0.0991** | 8/9 |
| 5 | +0.1733 | 0.1619 | 0.0854 | **0.1646** | 6/9 |
| **6** | **+0.0903** | 0.0988 | **0.0025** | **0.0811** | **4/9** |

**Stage 6 is genuinely distinguished.** It has the smallest mean absolute error of any stage index
(0.0811 vs 0.099–0.201) and it is the only stage where the cells straddle c* rather than sitting
almost uniformly conservative. Mean |c − c*| falls by a factor of **2.5** from k1 to k6. **The
migration toward the multitask value is real.**

Stage 3 is the cautionary case and belongs in the paper as one: its |mean − c*| is 0.0630, nearly
as small as stage 6's 0.0025 — while its mean |c − c*| is 0.1884, the second *worst* of any stage.
It is a textbook demonstration that |mean − c*| is the wrong statistic for this question.

The anchor migrates too, from much further out and far more slowly: mean |c − c*| 0.7018 (k1) →
0.6094 (k6), a factor of 1.15 against sequential's 2.5, with 9/9 cells above c* at every stage.

---

## Q3 — Does the criterion return, or does it happen to end near the bound?

| cell | raw path | net displacement | net/gross | crossings of its endpoint level | endpoint rank | endpoint err | best earlier err |
|---|---|---|---|---|---|---|---|
| seq\|o1\|s17 | 0.7643 | 0.3635 | 0.476 | 1 | 3/6 | 0.0199 | 0.0590 (k4) |
| seq\|o1\|s23 | 0.7376 | 0.3284 | 0.445 | 1 | 2/6 | 0.0151 | 0.0746 (k5) |
| seq\|o1\|s31 | 1.2021 | 0.1707 | 0.142 | 3 | 4/6 | 0.1729 | 0.1497 (k4) |
| seq\|o2\|s17 | 1.0071 | 0.4425 | 0.439 | 2 | 2/6 | 0.0989 | 0.0709 (k4) |
| seq\|o2\|s23 | 1.0193 | 0.2697 | 0.265 | 2 | 2/6 | 0.0739 | 0.0742 (k4) |
| seq\|o2\|s31 | 1.1935 | 0.2293 | 0.192 | 2 | 3/6 | 0.1143 | 0.0619 (k2) |
| seq\|o3\|s17 | 0.6020 | 0.4176 | 0.694 | 0 | 1/6 | 0.0741 | 0.0613 (k3) |
| seq\|o3\|s23 | 0.7861 | 0.4457 | 0.567 | 0 | 1/6 | 0.1021 | 0.0654 (k1) |
| seq\|o3\|s31 | 1.1159 | 0.4025 | 0.361 | 0 | 1/6 | 0.0590 | 0.0570 (k1) |

Summary, sequential (n = 9):

- **mean net-to-gross 0.398** — the path is 2.5× the straight-line displacement it achieves
- **monotone approach: 0/9**
- **endpoint interior to the cell's own visited range: 6/9**; **crossed its own endpoint level: 6/9**
- **endpoint beats every earlier stage on |c − c*|: 3/9** (mean endpoint err 0.0811 vs mean
  best-earlier 0.0749)
- **endpoint beats the *typical* earlier stage: 9/9** (0.0811 vs 0.1595 averaged over k1–k5),
  sign test p = 0.004

The "best earlier" column is a minimum over five stages and is therefore selection-biased low; that
is exactly why the typical-earlier comparison is reported alongside it and is the unbiased one.

**Reading.** The two comparisons say different things and both are needed:

- **Against a typical stage, the endpoint is reliably better — 9/9.** There is genuine restoring
  movement toward c*. This is the half of "self-correcting" that survives.
- **Against the best point on its own path, the endpoint wins in only 3/9.** In **6/9 cells the
  criterion passed closer to the target at some earlier stage and then moved away again.** The
  endpoint is not an attractor the trajectory settles into.

Together with 0/9 monotone approaches, a 0.398 net-to-gross ratio, and a 6/9 crossing rate, the
signature is a **wandering trajectory with a weak restoring drift**, not convergence. The practical
consequence is sharp: **stage 6's proximity to the bound carries no guarantee about stage 7.** A
longer task sequence has no reason to stay there, and the paper cannot claim it would.

JOINT (n = 2, per-cell values only): net/gross 0.601 and 0.530, endpoint interior 2/2, endpoint
beats all earlier 2/2, endpoint err 0.0255 / 0.0256. Directionally the tidiest arm, but two cells.

---

## Q4 — How much accuracy is the mid-sequence drift actually worth?

Per cell: the worst-drifted **mid-sequence** stage (argmax over k < 6 of |c_k − c*|), and the POPE
F1 there against that same cell's endpoint F1.

### Sequential (n = 9)

| cell | worst stage k* | c at k* | \|c−c*\| at k* | F1 at k* | endpoint F1 | **F1 gap** |
|---|---|---|---|---|---|---|
| seq\|o1\|s17 | 3 | +0.3183 | 0.2305 | 0.8591 | 0.8604 | +0.0013 |
| seq\|o1\|s23 | 1 | +0.4078 | 0.3200 | 0.8421 | 0.8680 | +0.0259 |
| seq\|o1\|s31 | 3 | +0.4034 | 0.3155 | 0.8446 | 0.8547 | +0.0101 |
| seq\|o2\|s17 | 1 | +0.3241 | 0.2363 | 0.8481 | 0.8632 | +0.0151 |
| seq\|o2\|s23 | 1 | +0.3915 | 0.3037 | 0.8464 | 0.8610 | +0.0146 |
| seq\|o2\|s31 | 1 | +0.3947 | 0.3069 | 0.8439 | 0.8567 | +0.0128 |
| seq\|o3\|s17 | 1 | +0.2589 | 0.1711 | 0.8545 | 0.8641 | +0.0096 |
| seq\|o3\|s23 | 5 | +0.2921 | 0.2043 | 0.8577 | 0.8651 | +0.0074 |
| seq\|o3\|s31 | 5 | +0.2804 | 0.1925 | 0.8597 | 0.8691 | +0.0094 |

- **mean F1 gap +0.0118**, sd 0.0067, range [+0.0013, +0.0259], **positive in 9/9 cells**
- **95% CI on the mean gap (cells resampled as a list with multiplicity): [+0.0079, +0.0162]**
- item-sampling SD of F1 at that stage: **0.0042** (binomial resample of hits/false alarms at the
  recorded n, conditional on the checkpoint)
- seed-to-seed SD of *endpoint* F1 within a fixed task order, n = 3 seeds: **0.0046**

**The mean gap is ~2.6× either noise floor and positive in every cell.** So the answer to "does the
drift cost anything measurable?" is **yes — about 1.2 POPE F1 points at the worst-drifted stage**.
It is small: for scale, the anchor's endpoint deficit versus sequential is 6.6 points, and the base
model's F1 (0.8449) is *below* sequential's endpoint (0.8625).

Two caveats on the supporting statistics, stated before they get quoted:

- Within-cell correlation between |c − c*| and F1 across stages: mean **−0.691**, negative in
  **9/9** cells. This is close to a mechanical consequence of the arm's near-fixed evidence
  distributions (z-ROC R² = 0.957) — it confirms the SDT model translates placement into F1 at the
  expected rate. **It is a consistency check, not independent evidence that drift costs accuracy.**
- "Does a bigger excursion buy a bigger loss?" Sequential r = +0.589 (CI [−0.540, +0.973], perm
  p = 0.093) — **underpowered**. For the anchor it is r = +0.855 (perm p = 0.016), which is
  informative because the anchor's excursions are far larger.

### The exchange rate is strongly non-linear — Σ|Δc| must not be converted to F1

| arm | mean excess \|c−c*\| at worst mid stage | mean F1 gap | implied F1 per criterion unit |
|---|---|---|---|
| joint (n=2, per-cell values) | 0.0469 | +0.0003 | 0.006 |
| seq | 0.1723 | +0.0118 | 0.069 |
| anchor | 0.1255 | +0.0266 | 0.212 |

The anchor pays **more** F1 for a **smaller** excursion, because it is operating far out on the ROC
where F1 falls steeply. **The cost of drift depends on the maximum *displacement* from c*, not on
the path length.** The paper's Σ|Δc| = 0.94 is a path and **cannot be multiplied by any of these
slopes** to produce an F1 cost. Anchor (n = 9) mean F1 gap: **+0.0266**, 9/9 positive, CI
[+0.0159, +0.0411]. JOINT (n = 2): +0.0003 (cells −0.0011, +0.0017) — the arm that never drifts far
pays nothing, which is the internal consistency one would want.

---

## Q5 — The residual endpoint gap to JOINT: criterion, or separability?

Sequential's endpoint criterion matches JOINT's. Its endpoint F1 does not quite. Where does the
remainder live?

| quantity | sequential (n = 9) | JOINT (n = 2, cells quoted individually) | gap |
|---|---|---|---|
| endpoint criterion c | mean +0.0903 | +0.0623, +0.1134 | **+0.0025** |
| endpoint POPE F1 | mean 0.8625 | 0.8682, 0.8652 | **−0.0042** |
| endpoint d′ | mean 2.2266 (sd 0.0324) | 2.2567, 2.2525 | **−0.0280** |

Sequential endpoint F1 per cell: 0.8604, 0.8680, 0.8547, 0.8632, 0.8610, 0.8567, 0.8641, 0.8651,
0.8691.

- **7/9 sequential cells fall below JOINT's *lower* cell (0.8652); 1/9 rises above its upper (0.8682).**
- The endpoint d′ gap of 0.0280 is **0.86 sd** of the sequential cell distribution.

**With the endpoint criterion gap at +0.0025, the residual 0.42-point F1 deficit cannot be a
criterion-placement cost.** On this axis separability is the only remaining carrier — i.e. what is
left at the endpoint is **ordinary forgetting, not criterion drift**. That is a meaningful finding
for the paper: it says the criterion story, whatever else it explains, does **not** explain
sequential's residual endpoint deficit.

**Two limits on how hard this may be pushed, both real:**

1. **JOINT has n = 2.** Seven-of-nine below a single reference cell is suggestive, not a test.
2. **This uses single-point d′ — the exact estimator this project retracted a claim over**
   (2026-09-09, the corrected-ceiling reversal). The assumption-free index is d_a, but d_a is
   fitted across a cell's six stages and is therefore a *trajectory-level* quantity that cannot
   arbitrate an *endpoint* comparison. For what it is worth, d_a points the same way on the means
   (seq 2.1587, JOINT 2.1513 / 2.2111) but **JOINT's two d_a values straddle the sequential mean**,
   and only 2/9 sequential cells fall below JOINT's lower d_a cell. So the direction is consistent
   and the magnitude is **not established**. The claim that survives is the negative one: *the
   residual endpoint gap is not criterion placement.*

---

## What each outcome would mean for how the paper is framed

### The outcome that did **not** happen

Had the drift cost nothing measurable, the honest move would have been to reframe the paper from
"here is a drift to fix" to "here is a drift that turns out to be self-correcting, which is itself
the finding" — and the method half's failure would have become much less damaging, because it would
have been trying to fix something that did not need fixing. **That reframe is not available**, on
three independent counts: the cost is 9/9 and outside both noise floors (Q4); the trajectory does
not converge (Q3); and the endpoint agreement is an averaging artifact at the precision quoted (Q2).
It should not be pursued.

### The outcome that did happen, and what follows

**1. The existing motivation is vindicated, but only weakly, and the paper must quote the size.**
Drift costs ~1.2 F1 points mid-sequence. That is a real cost and it is worth reporting with its CI.
It is also not a large one, and the paper should not imply otherwise: the base model is *worse* at
the endpoint than the sequentially tuned one, and the anchor's own endpoint deficit is 5× the entire
drift cost. **"Reducing Σ|Δc| is worth about one F1 point in this setting" is the honest headline**,
and it is a weaker motivation than the paper currently implies.

**2. The claim "sequential's endpoint is already at the JOINT bound" must be requoted.** In its
current form — +0.0903 against +0.0878 — it is 32× more precise than the cells support and will not
survive a reviewer who asks for the spread. The defensible statement is:

> Sequential's endpoint criterion migrates to within 0.081 of the multitask value (mean absolute
> error, 95% CI [0.052, 0.112], n = 9 cells), from 0.344 at the untuned base and 0.201 at stage 1.
> Individual cells span [−0.014, +0.261], so no single run may be assumed to land on the bound.

**3. "Self-correcting" may be used only in the weak sense, and "converges" not at all.** The
endpoint beats a typical earlier stage in 9/9 cells — that is real and worth stating. The endpoint
beats the *best* earlier stage in only 3/9, no cell approaches monotonically, and the path is 2.5×
its net displacement. The mechanism is a wandering criterion with a weak restoring drift. **Stage 6
is not a fixed point**, and any sentence implying a longer sequence would remain there is
unsupported.

**4. The strongest available motivation for the method half is dispersion, not endpoint quality.**
The data point this way and the paper does not currently make the argument: sequential's endpoint
lands anywhere in a 0.275-wide window, which is 80% of the base's entire mis-placement. A
practitioner running one sequence has no way to know which end they got. **That is a defensible
reason to want criterion control even though the endpoint *mean* needs no fixing** — and it is
consistent with, rather than contradicted by, the negative method result.

It should be offered with the caveat that the anchor does not deliver it either: the anchor's
endpoint criterion sd is 0.0834 against sequential's 0.0988 — a modest 0.84× dispersion reduction
bought at a bias of 0.61 criterion units. Textbook bias–variance, on the wrong side of the trade.

**5. One claim is strengthened and should be stated: the residual endpoint deficit is not the
criterion's fault.** Sequential ends 0.42 F1 points below JOINT with its criterion matched, so the
remainder is separability — ordinary forgetting. This cleanly separates the two stories the paper
tells and is a better-posed statement than the current implicit attribution. Bounded by JOINT's
n = 2 and by single-point d′ being a retracted-estimator family, as recorded above.

**6. Nothing here revives the structural trade-off, and nothing here rescues the method.** The
cross-arm correlation reproduces `TRADEOFF_PLACEMENT.md` exactly, anchor family included and
excluded; the reversal on dropping the anchor is intact. `PCR0_VERDICT.md` is unaffected.

---

## What would change these conclusions

1. **More cells.** Every within-arm correlation here is underpowered: n = 9 cannot resolve a true
   ρ below 0.816 at 80% power, and the observed 0.49–0.63 sit inside that blind spot. If the
   sequential path-vs-endpoint relationship is real at r ≈ 0.6, ~20 cells would establish it, and
   it would then partially restore the paper's original framing on the endpoint axis.
2. **More JOINT cells.** c*, the endpoint spread comparison, and the separability decomposition all
   rest on n = 2. A third and fourth JOINT cell would do more per GPU-hour for this paper's
   inferential claims than another method arm.
3. **A seventh stage.** The single sharpest test of "attractor versus wandering" is whether stage 7
   stays near c*. Q3 predicts it would not, in roughly 6/9 cells. Nothing on disk can answer it.
4. **A trajectory noise floor.** The 0.0046 figure is seed-to-seed at the endpoint only. No
   configuration on disk is re-run end to end, so the run-to-run variability of the *path* is
   unmeasured, and the 1.2-point F1 cost is quoted against an item-sampling floor and an endpoint
   seed floor rather than against a true trajectory floor.
