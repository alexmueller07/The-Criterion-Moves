# Is "less criterion drift costs endpoint placement" a real trade-off? — 2026-09-12

**Verdict up front: NO, not on the data we have. The pattern is the anchor family's
signature, not a trade-off, and it cannot carry a reframing of the method section.**

> **CORRECTED 2026-09-13.** This write-up originally carried `fsL_locproj_o1_s17` at
> Σ|Δc| = 0.2237 with endpoint c = +0.2739. Those numbers were read **while the arm was
> still in eval**, over fewer than six transitions, and a shorter path is mechanically
> smaller; they were withdrawn in full (`FULLSTUDY_PREREG.md`, 2026-09-12 CORRECTION and
> 2026-09-13 SECOND CORRECTION). At 6/6 the arm is **0.9895 / +0.0041** — *more* drift
> than the sequential control and, against c\* = +0.0878, *worse* placement. Every
> raw-axis number below is regenerated from the corrected value; the post-settling axis
> is untouched, because `locproj` was never on it. **The verdict is unchanged, but two of
> the arguments behind it move: one gets stronger, one is withdrawn. Both are marked in
> place rather than quietly rewritten.**

Script: `analysis/tradeoff_placement.py` (no GPU, no new runs).
Machine output: `analysis/readout/tradeoff_placement.json`.
Reproduce: `python3 analysis/tradeoff_placement.py --check-premise --out analysis/readout/tradeoff_placement.json`

---

## What was tested

The hypothesis: across all arms we have, reducing the criterion drift path Σ|Δc|
systematically costs endpoint criterion placement — so the method section becomes
"a family of interventions is defeated by a characterisable trade-off" rather than
"our candidate failed".

Placement error is `|c_endpoint − c_ref|` with **c_ref = +0.088, the JOINT arm's
empirical endpoint criterion**, not a modelled 0 (balanced accuracy peaks at c = 0
only under equal variance; our z-ROC slopes are 0.55–0.72). A trade-off predicts
**r < 0** between drift and placement error.

Everything is computed **per cell** and only then averaged. The raw path
(base → k1 → … → k6) and the post-settling path (first step dropped — the
pre-registered endpoint) are kept on separate axes and never mixed.

## The premise table mixes two estimators — fix this first

The motivating table is not estimator-matched, and that matters before anything else:

| arm | cells | raw Σ\|Δc\| | post-settling Σ\|Δc\| |
|---|---|---|---|
| seq | 9 | 0.9364 [0.6020–1.2021] | 0.7814 [0.4295–1.1568] |
| anchor | 9 | 0.7224 [0.5209–1.0154] | 0.3641 [0.1878–0.5519] |
| joint | 2 | 0.6074 [0.6002–0.6145] | 0.2278 [0.2157–0.2400] |
| locproj | 1 | 0.9895 (single cell) | **not recoverable** |

The quoted "sequential 0.7643" is the **raw** path of one cell (o1/s17); "anchor
0.364" is a **post-settling** 9-cell mean; "locproj 0.2237" is **raw**, single cell,
**and retracted** — read mid-eval over fewer than six transitions. Its 6/6 value is
**0.9895**, which is worse than the sequential cell it was quoted against, not 71% better.
Matched on the raw axis the anchor's advantage over sequential is small
(0.7224 vs 0.9364) and the per-cell ranges overlap almost completely — the anchor's
drift reduction is largely a post-settling phenomenon. Matched on the same cell
(o1/s17) the anchor's raw path is **0.7834, slightly worse than sequential's 0.7643**.

`fsL_locproj_o1_s17`'s per-stage criteria were never recorded, so it cannot be
placed on the pre-registered post-settling axis at all with what is on disk. The
script carries it as `TEXT-RECORDED` — now the **2026-09-12 CORRECTION** table as
re-scored against c\* on 2026-09-13, with the superseded numbers kept beside it in the
JSON under `supersedes_retracted` — and drops it from that axis rather than
approximating it. Its `n_stages: 6` is flagged `n_stages_verified: false`: the arm has
no cell under `results_fs`, and `fs_aggregate.json` (2026-09-09) predates it, so 6/6 is
a human assertion and not a recorded fact.

## The measured relationship, with n stated honestly

The units are **cells**, and cells are clustered inside arms, so these n's are not
independent observations of the hypothesis. The honest n for "is this a trade-off"
is the number of **arms**: 3, or 4 with locproj.

**Post-settling axis (pre-registered endpoint), 20 cells from 3 arms:**

| set | n | Pearson r | 95% CI (cell bootstrap) |
|---|---|---|---|
| all cells | 20 | **−0.467** | [−0.760, −0.169] |
| **anchor family dropped** | 11 | **+0.682** | [+0.245, +0.886] |
| joint dropped (circularity) | 18 | −0.678 | [−0.849, −0.496] |
| anchor + joint dropped (= seq) | 9 | +0.569 | [−0.086, +0.895] |
| **arm identity partialled out** | 20 | **+0.247** | [−0.198, +0.682] |
| **arm level** | **3 arms** | **−0.197** | no interval — n = 3 |

**Raw axis, 21 cells from 4 arms (locproj included):**

| set | n | Pearson r | 95% CI (cell bootstrap) |
|---|---|---|---|
| all cells | 21 | −0.317 | [−0.635, +0.025] |
| **anchor family dropped** | 12 | **+0.711** | [+0.338, +0.914] |
| joint dropped | 19 | −0.4755 | [−0.752, −0.153] |
| anchor + joint dropped (= seq + locproj) | 10 | +0.614 | [+0.068, +0.928] |
| **arm identity partialled out** | 20 | **+0.208** | [−0.335, +0.766] |
| **arm level** | **4 arms** | **−0.246** | no interval — n = 4 |

The two rows that move most on this axis, +0.073 → **+0.711** and −0.095 → **+0.614**,
both turn on a **single hand-transcribed cell out of 12 and 10**. That is enormous
leverage for one point, and the direction it now points (strongly positive = the
opposite of a trade-off) should be read as *one cell's worth* of evidence, not as an
n = 12 result. Dropping `locproj` entirely leaves the pooled raw r at −0.294, exactly
where it was before the correction.

The bootstrap resamples cells as a **list with multiplicity** (verified: 12.83 of 20
distinct indices per resample, ≈ 0.632n; a `set()`-deduped version returns intervals
88% as wide). Those intervals cover cell-to-cell noise **within the arms we ran**.
They do not cover arm-to-arm uncertainty — the question actually being asked — which
3–4 arms cannot estimate. **Read every interval above as a lower bound on the true width.**

## Does it survive dropping the anchor family? No — it reverses.

On the pre-registered axis the pooled r = −0.467 becomes **r = +0.682** with
anchor / critp / anchorlo / anchorOOD removed. The sign flips. The raw axis now does
the same thing and harder: −0.317 → **+0.711**, an interval that excludes zero
([+0.338, +0.914]) — though see the leverage caveat above, since that swing is one
transcribed cell. Leave-one-arm-out on the pre-registered axis:

| arm dropped | n cells | r |
|---|---|---|
| anchor | 11 | **+0.682** |
| joint | 18 | −0.678 |
| seq | 11 | +0.440 |

Only the anchor's presence produces a negative relationship. Dropping **either**
non-anchor arm leaves a positive one.

Three further checks say the same thing:

1. **Arm identity partialled out** (subtract each arm's mean from both variables —
   the direct test of whether the relationship is continuous or a two-group
   separation): r = **+0.247** post-settling, **+0.208** raw. There is **no
   within-arm trade-off**. Within the sequential arm alone the sign is positive
   (r = +0.569 post-settling): cells that drifted *more* landed *further* from the
   reference, the opposite of the hypothesis. The pooled negative correlation is the
   anchor cluster sitting in one corner of the plane wearing an n = 20 costume.
2. **Pairwise arm directions.** Raw axis: **2 of 6** arm pairs go the trade-off way —
   below chance (it was 3 of 6 before the `locproj` correction). Post-settling: **1 of 3**. The single trade-off-consistent pair on
   the pre-registered axis is anchor vs seq, which is the observation we started from.
3. ~~**The intervention-vs-intervention comparison goes the wrong way.**~~
   **WITHDRAWN 2026-09-13 — this argument reverses.** It read: the only comparison
   between two *interventions* (not against the control) is locproj vs anchor, matched
   at cell o1/s17, where locproj had far less raw drift (0.2237 vs 0.7834) **and** far
   better placement (err 0.186 vs 0.675). With locproj corrected to 6/6 the same
   matched pair now runs the other way — **anchor has the shorter path (0.7834 vs
   0.9895) and the worse placement (err 0.675 vs 0.084)**, which is trade-off-consistent.
   One pair is not evidence in either direction (chance is a half), but this line no
   longer supports the verdict and is not counted toward it.

## The counterexample that kills "structural"

**The joint arm already has both.** On the pre-registered axis it has the *lowest*
drift of any arm (0.2278, below the anchor's 0.3641) *and* the best placement
(err 0.0255, vs seq 0.0811 and anchor 0.6094). It dominates the anchor on both
objectives and it dominates sequential on both. A configuration with short path and
correct endpoint therefore exists in our own data, which refutes "you cannot have
both" as a structural law.

The fair objection: joint is a non-sequential oracle, not an intervention applied to
sequential training, so one could restrict the claim to sequential interventions.
That restriction leaves **{seq (control), anchor, locproj}** — two interventions and
a control. **Corrected 2026-09-13:** the one intervention-vs-intervention pair inside
it (anchor vs locproj) now goes the *trade-off* way, so the restricted sample no longer
refutes the claim by that route. What the restricted sample does contain is a dominance
pair involving the control — `seq` has both less drift than `locproj` (0.9364 vs 0.9895)
and better placement (0.0811 vs 0.0837) — but those margins, **0.053 and 0.003**, sit far
inside cell-to-cell noise: seq's own nine cells span raw path [0.602, 1.202] and
|c − c\*| [0.015, 0.173]. **That is a dominance pair by arithmetic and nothing by
measurement, and it is not leaned on here.** Under the restriction the honest position is
that two arms, one of them a single transcribed cell, cannot decide the question either
way — not that the sample refutes the claim.

**Robustness.** Re-running against a modelled c_ref = 0 instead of the joint endpoint
changes nothing qualitative: pooled −0.441 → **+0.594** with the anchor family
dropped, arm-identity-partialled r = +0.525, joint still dominates both other arms.

## What would falsify this write-up (i.e. what would revive the hypothesis)

1. **More arms, scored the same way.** The decisive missing data is arms that reduce
   drift by mechanisms unrelated to the anchor. `fsL_critbal_o1_s17` is the direct
   test — it corrects the *target* rather than damping the *path*. If critbal,
   locproj's second seed, locearly and loclate all land at low drift and poor
   placement, the arm-level n rises to 6–7 with three distinct mechanisms and the
   arm-level correlation becomes worth computing. It is not worth computing at n = 3.
2. **locproj on the pre-registered axis.** Pull `fsL_locproj_o1_s17_k*/pope_gen.jsonl`
   into `results_fs` and rerun — the script rescores it with `fs_common.score_pope`
   automatically and the `TEXT-RECORDED` fallback drops out. `mechanism_readout.py` now
   refuses any cell lacking `EVAL_DONE` or with POPE `n_total != 9000`, reports every
   refusal, and dumps the per-stage criterion vector it used to discard — so the same
   pull also *verifies* the 6/6 stage count, which today is asserted in prose and
   recorded by no artifact in this repo. Until then the only arm that motivated the
   hypothesis cannot be compared on the axis the paper pre-registered.
3. **A within-arm relationship.** If, with more cells, the arm-identity-partialled
   correlation turned negative, the trade-off would be continuous rather than a
   between-group artifact. It is currently **positive** on both axes.
4. **Ruling the joint arm out of scope on a principled basis** — stated before seeing
   the numbers, not after. Post-hoc exclusion of the one counterexample is not available.

## Verdict on the reframing

**Too thin. Do not reframe the method section on this.**

The honest summary is: *n = 3–4 arms; the pooled cell-level correlation is negative on
the pre-registered axis (−0.467) but reverses to +0.682 when the anchor family is
dropped, is +0.247 with arm identity partialled out, is −0.197 at the arm level with
no interval, and is contradicted by the joint arm, which has both the shortest path and
the best placement.* Three arms is not a trade-off, and here two of the three
observations (anchor, critp) are one mechanism, while the third (locproj) is a single
cell, hand-transcribed, whose only comparable number is on the wrong axis — and whose
corrected value shows **more** drift than the control, so it never belonged in the
motivating pattern in the first place.

What the data *does* support, and what is already written:

- The anchor family buys post-settling drift reduction at a large placement cost
  (Σ|Δc| 0.364 vs 0.781, endpoint c +0.697 vs +0.090). Established, 9/9 cells.
- The projector freeze does **not** repeat that shape — corrected 2026-09-13. At 6/6
  it has *more* raw drift than the sequential control (0.9895 vs the 0.9364 arm mean,
  and vs 0.7643 at the matched cell) and *worse* placement against c\* (0.0837 vs
  0.0811), i.e. it is behind the control on both axes, by margins well inside the
  control's own cell-to-cell spread. It is not a third observation of the trade-off
  shape, and the localization dissociation it belonged to was withdrawn on 2026-09-13.
  Still a single hand-transcribed cell, still unmeasurable on the pre-registered axis.
- Nothing licenses "a family of interventions is defeated by a trade-off". The
  claim needs the arms in item 1 above before it can be made, and the joint
  counterexample means that even then it would have to be scoped to sequential
  interventions and stated with that scope visible.

If critbal lands and *also* buys drift with placement, that is a third mechanism and
the arm-level picture is worth recomputing — this script takes it automatically. If
critbal reduces drift *and* lands near +0.088, the hypothesis is dead and the method
section has a positive result instead.
