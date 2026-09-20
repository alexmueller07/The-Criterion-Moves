# What moves the criterion besides answer statistics — a variance accounting

Date: 2026-09-13. Zero GPU; every number below comes from readouts already on disk.

Script: `analysis/dose_residual.py` · Machine output: `analysis/readout/dose_residual.json`
Inputs: `analysis/readout/fs_aggregate.json` (UCIT, 2026-09-09), `results/S0..S4` +
`results/seq_s{2,3}_k*` + `results/seq_rev_k*` (pilot, rescored here with the audited
`fs_common.score_pope`), `fullstudy/pilot_manifest.json`, `fullstudy/results_fs/llava15_base/*_gen.jsonl`.

**Nothing under `paper/` was touched.**

## 0. Pipeline validation before any conclusion

The script re-derives, from scratch, five numbers the pre-registration already records. All
five match exactly, so a pipeline error would have surfaced here rather than inside a claim.

| quantity | this script | `FULLSTUDY_PREREG.md` |
|---|---|---|
| zero-dose max\|c − base\| across 9 cells | 0.403–0.550, mean 0.472 | 0.403–0.550, mean 0.472 |
| SEQ η² task identity (post-settling) | 0.7311 | 0.731 |
| SEQ η² sequence depth (post-settling) | 0.0488 | 0.049 |
| ANCHOR η² task / depth (post-settling) | 0.3523 / 0.0526 | 0.352 / 0.053 |
| per-task pull, post-settling, all six | −0.2511, −0.1839, −0.1092, +0.0049, +0.1391, +0.1629 | identical |

---

## 1. The headline

**Nothing on disk explains the zero-dose drift, and the reason is structural, not statistical
bad luck: the UCIT design has ~0.14 of Δc variance available for every non-task explanation
combined, and inside that 0.14 adjacency, depth and ordering are the same factor.**

The one thing that *is* established — task identity carries the drift (η² 0.683, p < 0.0001) —
was already known. The new content is (a) a hard ceiling on what any further stream-level
variable could ever explain here, (b) seven candidates ruled out (§5), and (c) one candidate
that survives at ~5–9% of variance and is a *hypothesis*, not a finding (§6).

---

## 2. Step 1 — the answer-statistic account's residual is 100% of the data

The frozen `ucit_manifest.json` (committed at data-prep time, before any eval ran) gives all six
UCIT tasks yes = 0.0000, no = 0.0000 (VizWiz 0.0005), refusal = 0.0000. Dose =
refusal + |yes − no| has **standard deviation 0.000186 across the six tasks.**

*Provenance, stated because it matters:* the manifest itself lives on the cluster and is not in
the repo. The values used are transcribed into `dose_residual.py` from `FULLSTUDY_PREREG.md`
(lines ~86, ~399, ~1504), which quotes it in three places. `--manifest_ucit <path>` overrides
the transcription with the real file and the script prints which source it used. The
`mean_target_words` column is independently corroborated against the on-disk **val** targets
(`fullstudy/results_fs/llava15_base/*_gen.jsonl`, 500 rows/task), which reproduce four of six
train values to 2 decimals; the two that differ are ArxivQA (train 1.58 vs val 1.11) and
Flickr30k (train 12.29 vs val 17.85), and both split-level measurements give the same answer
for every analysis here (`answer length (val mean words)` is reported as its own row in §4.3 and
lands within 0.01 of the train version).

The instruction was "fit Δc on dose and keep the residuals". On UCIT that fit is degenerate and
saying so is the honest result:

- OLS Δc ~ dose returns R² = 0.1009. **This number is meaningless** — it is a regression on a
  regressor with sd 2×10⁻⁴, driven entirely by VizWiz's 0.0005 acting as a task-identity dummy.
  It is reported only so nobody later "discovers" it.
- Scored against what the account actually *predicts* — Δc = 0 where dose = 0 —
  R²_predictive = **−0.098**. Predicting the sample mean beats predicting zero.
- **Variance the answer-statistic account explains on UCIT: 0.0%. The residual is Δc itself.**

Observed Δc over 54 stage-cells: mean −0.0568, mean |Δc| 0.1561, sd 0.1835.

---

## 3. The design ceiling — read this before any candidate

Within one ordering, position determines *both* the task trained and its predecessor. So
`ordering × position` (18 groups of 3 seeds) is the **finest partition any stream-level
variable can induce**, and its η² is a hard ceiling on task, depth, predecessor, transition,
interference and every interaction among them.

| | share of Δc variance |
|---|---|
| ceiling: ordering × position (η², 18 levels) | **0.823** |
| task identity alone (η², 6 levels) | **0.683** |
| headroom for *everything else combined* | **0.140** |
| irreducible seed-to-seed noise | **0.177** |

η² with 18 levels over 54 points sits near 0.32 on pure noise, so the level-count-corrected
ω² is quoted alongside everywhere below (ceiling ω² = 0.736, task ω² = 0.645).

**Consequence.** Any answer of the form "X explains the zero-dose drift" is capped at 14% of
Δc variance unless X *is* task identity. A candidate reported at 40% would be an error.

### And inside that 0.140, the candidates are not separable

Within one task, the three orderings supply exactly three (position, predecessor) pairs, one
per ordering. **For a given task, position, predecessor and ordering induce the identical
3-way split of its 9 cells.** Whatever sits above task identity is the task × ordering
interaction; this design cannot say whether it is adjacency, depth, or ordering-specific
anything. The only reason the three differ numerically at all is how their labels align
*across* tasks.

Orderings (from the data, not assumed): o1 = ArxivQA, CLEVR-Math, Flickr30k, IconQA,
ImageNet-R, VizWiz; o2 = reverse of o1; o3 = ImageNet-R, VizWiz, ArxivQA, CLEVR-Math,
Flickr30k, IconQA.

---

## 4. Variance accounting

All-stages, LLaVA SEQ, 9 cells × 6 stages = 54 Δc. η² CIs from resampling the 9 **cells as a
list with multiplicity**, B = 10,000, seed 17. Permutation p over 20,000 shuffles.

### 4.1 Factors on the raw residual

| factor | levels | η² | ω² | perm p | η² 95% CI |
|---|---|---|---|---|---|
| task identity | 6 | 0.683 | 0.645 | <0.0001 | [0.666, 0.796] |
| sequence depth (position) | 6 | 0.101 | 0.007 | 0.380 | [0.057, 0.536] |
| predecessor task | 7 | 0.172 | 0.065 | 0.161 | [0.097, 0.674] |
| transition (prev → cur) | 14 | 0.808 | 0.742 | <0.0001 | [0.769, 0.937] |
| ordering | 3 | 0.003 | −0.036 | 0.926 | [0.000, 0.008] |
| seed | 3 | 0.003 | −0.036 | 0.932 | [0.000, 0.009] |
| *ceiling*: ordering × position | 18 | 0.823 | 0.736 | <0.0001 | [0.807, 0.958] |

Transition's 0.808 looks like it beats task's 0.683. **It does not mean what it looks like.**
Transition is a 14-level refinement of the task factor with only 3–9 observations per level;
its apparent gain is almost entirely level count plus the task means it contains. The
incremental test is the next table.

### 4.2 Each candidate *beyond* task identity

Null: reshuffle Δc **within each task** — the exchangeability H₀ actually asserts, and far more
appropriate than the global shuffle, which destroys the task effect we already know is real.
"of TOTAL" = η² on the task-residual × 0.317 (the task residual's share of total variance).

| factor \| task | levels | η² | ω² | p (within-task) | of TOTAL |
|---|---|---|---|---|---|
| transition (prev → cur) | 14 | 0.395 | 0.195 | 0.0090 | 0.125 |
| predecessor task | 7 | 0.280 | 0.185 | **0.0015** | 0.089 |
| sequence depth (position) | 6 | 0.126 | 0.035 | 0.178 | 0.040 |
| *ceiling*: task × ordering | 18 | 0.443 | 0.177 | 0.0327 | 0.140 |

Post-settling (stage 1 dropped, removing the BASE predecessor level, which is a settling
contrast rather than an adjacency one), the same ordering holds and shrinks:
transition 0.079 of total (p = 0.011), predecessor **0.055 of total (p = 0.0057)**,
depth 0.027 (p = 0.246), ceiling 0.097.

**Replication on the ANCHOR arm** (same stream, different training objective, 9 more cells),
post-settling: predecessor beyond task η² = 0.175, p = 0.0245; depth p = 0.780. Same ordering,
independently.

### 4.3 Which property of the task

Every candidate here is a per-task constant, so each is *nested inside* the task factor and
capped by its 0.683. "of TOTAL" = (r² on the 6 task means) × 0.683. Slopes are fitted **within
each cell and then aggregated over the 9 cells**.

| per-task property | within-cell mean r | cells w/ same-sign slope | r² on 6 task means | of TOTAL |
|---|---|---|---|---|
| answer length, train mean words | +0.705 | 9/9 | 0.683 | **0.466** |
| answer length, val mean words (independent measurement) | +0.714 | 9/9 | 0.696 | 0.475 |
| log₁₀ answer length | +0.696 | 9/9 | 0.659 | 0.450 |
| is long-answer (≥2 words) | +0.699 | 9/9 | 0.674 | 0.460 |
| answer FORM (letter/digit/word/sentence, 4 levels) | — | — | η² 0.497, ω² 0.462, p<0.0001 | 0.497 |
| answer-statistic dose | +0.319 | 8/9 | 0.148 | 0.101 |
| **prompt length** (val mean words; 13.0–99.9 across tasks) | −0.198 | 3/9 | 0.063 | **0.043** |
| **n_train / optimizer steps** | — | — | **constant** | **0.000** |

Per-task pull, post-settling, per cell then aggregated:

| task | train words | post-settling mean Δc | +/n cells | 95% CI |
|---|---|---|---|---|
| IconQA | 1.00 | −0.2511 | 0/9 | [−0.300, −0.204] |
| ImageNet-R | 1.28 | −0.1839 | 0/6 | [−0.283, −0.090] |
| ArxivQA | 1.58 | −0.1092 | 0/6 | [−0.160, −0.058] |
| CLEVR-Math | 1.00 | +0.0049 | 5/9 | [−0.053, +0.061] |
| VizWiz | 11.61 | +0.1391 | 5/6 | [+0.036, +0.247] |
| Flickr30k | 12.29 | +0.1629 | 9/9 | [+0.107, +0.226] |

Long-vs-short contrast, computed per cell then aggregated: **+0.2616, 95% CI [+0.2095,
+0.3168], same sign in 9/9 cells.** Exact permutation over all 15 two-vs-four splits:
rank 1 of 15, **p = 0.0667 — which is this design's floor.** No 2-vs-4 split of six tasks can
produce a smaller p. The effect is maximal and the design is underpowered at the same time.
This reproduces the 2026-09-11 prereg entry with the per-cell discipline applied and the
conclusion unchanged.

---

## 5. What is ruled out

**Stage magnitude / training steps — RULED OUT for UCIT, by construction.**
`fullstudy/ucit_prep.py` caps every task at `TRAIN_CAP = 8000`, and every UCIT source split is
≥ 23,998 rows (`design_notes/full_study_options.md`, Table 5 of arXiv:2503.12941v2). With
`--epochs 1.0 --eff_batch 64` (`fullstudy/run_arm.py`), **every stage of every cell runs exactly
125 optimizer steps on exactly 8,000 examples.** A regressor with zero variance explains zero
variance. The mundane "it just scales with how much the model was updated" explanation is
unavailable on UCIT — it is not merely unsupported, it is inapplicable.

Caveat, and it is real: *gradient and loss magnitude* are a different question, and they are
**NO-DATA, not cleared.** No training log, loss curve or grad-norm record is on disk locally;
`fullstudy/results_fs/*/` holds only generations and `gate_info.json` (host, argv, job id, free
disk, GPU). Do not write "training magnitude is ruled out" without that qualifier.

**Sequence depth — RULED OUT as a standalone factor.** η² 0.101, p = 0.380 raw; 0.040 of total
beyond task, p = 0.178; post-settling 0.027, p = 0.246; anchor arm p = 0.780. Four looks, no
signal. (Also: it is not separable from adjacency within a task — §3 — so this is a statement
about how well its labels align across tasks, nothing stronger.)

**Prompt / image distribution — RULED OUT as far as this design can see it.** Prompt length
varies enormously across tasks (ArxivQA 99.9 words, which include a full rationale, versus
Flickr30k and VizWiz at 13.0) and explains 0.043 of Δc variance, with the slope's sign
inconsistent across cells (3/9). This is the only *structural* per-task variable recorded that
is not answer-side, and it does not track the criterion.

**Ordering and seed as main effects — RULED OUT.** η² 0.003 each, ω² negative, p > 0.92.

**Mean reversion toward a fixed point — RULED OUT.** Δc regressed on the incoming criterion
level gives slope −0.922, r = −0.707 — but Δc = c_k − c_prev contains c_prev, so this is the
Galton artefact, and slope −1 is what *independent* c_k and c_prev produce, not what an
attractor produces. The Galton-free statistic is the level autocorrelation:
**r(c_k, c_prev) = +0.111, 95% CI [−0.192, +0.360]** — indistinguishable from zero. The
criterion after a stage has essentially forgotten where it was. That is memorylessness (the
pre-registered E4 property), not a new explanatory variable, and it re-states rather than
supplements §4.1.

**Δd′ as an explanation — RULED OUT as circular.** Δc correlates with Δd′ at within-cell mean
r = +0.909, which is the largest correlation anywhere in this analysis and means nothing.
c = −½(z(H) + z(FA)) and d′ = z(H) − z(FA) are orthogonal *contrasts of the same two z-scores*.
When FA moves more than H — observed sd ratio z(FA):z(H) = **1.41** — then Δc → −½Δz(FA) and
Δd′ → −Δz(FA), so r → +1 arithmetically. It restates *where* the movement lives (the
false-alarm side); it is not an independent variable. Incidentally: max |d′ − base| = 0.204
across all 54 stage-cells, inside the pre-registered F1 damage threshold of 0.30.

**Generation-length state — NOT SUPPORTED.** Δ(CHAIR caption mean_words) gives within-cell mean
r = −0.288, CI [−0.405, −0.169]; the level correlates at −0.081, CI spanning zero. And POPE's
own `mean_new_tokens` is **2.0 in every stage of every cell** — the criterion moves while the
answer format on the measured task is frozen. Whatever the format account is, it is not "the
model started writing longer POPE answers".

---

## 6. What survives — and it is a hypothesis

**Predecessor identity, beyond the task being trained.** 0.089 of total Δc variance all-stages
(p = 0.0015 within-task), 0.055 post-settling (p = 0.0057), replicated on the anchor arm
(p = 0.0245). It survives additionally removing the c_prev slope (0.071 of total, p = 0.0067).

Mean task-residual Δc by predecessor:

| predecessor | n | mean residual Δc | +/n | current tasks it precedes |
|---|---|---|---|---|
| BASE (untuned) | 9 | −0.0692 | 2/9 | ArxivQA, ImageNet-R, VizWiz |
| CLEVR-Math | 9 | −0.0558 | 0/9 | ArxivQA, Flickr30k |
| ArxivQA | 6 | −0.0468 | 2/6 | CLEVR-Math |
| ImageNet-R | 9 | +0.0356 | 6/9 | IconQA, VizWiz |
| Flickr30k | 9 | +0.0402 | 6/9 | CLEVR-Math, IconQA |
| VizWiz | 6 | +0.0405 | 5/6 | ArxivQA, ImageNet-R |
| IconQA | 6 | +0.0801 | 5/6 | Flickr30k, ImageNet-R |

Range ±0.08, about one-third the size of the task effect (range 0.41).

Why this one is estimable when depth is not: the predecessor label is shared *across* tasks —
three different current-tasks at three different depths can all be "preceded by Flickr30k" —
so a consistent predecessor effect is a 6-df constraint inside the 18-df task × ordering space,
and it fits better than an arbitrary interaction would. That is genuinely informative about
adjacency.

It is **not** the length account one step back: r(predecessor's own answer length, predecessor
residual) = +0.338 over six predecessors, and IconQA (1.00 words) is the most positive
predecessor while the two long-answer tasks sit mid-table.

**Labelled honestly:**

- This was chosen after looking at the data. It is a **hypothesis**, not a finding.
- Four factors were tested beyond task; p = 0.0015 survives Bonferroni ×4 (0.006), p = 0.0057
  post-settling does not comfortably (0.023) — quote the post-settling number.
- It buys **5–9% of Δc variance.** The zero-dose drift is 0.40–0.55 units; this is nowhere near
  an explanation of it.
- The within-task permutation holds the task means fixed but cannot know that predecessor and
  position are the same split within a task. Its p is evidence that the *predecessor labelling*
  aligns across tasks better than the *position labelling* does — which is what an adjacency
  effect looks like, and also what a lucky alignment looks like at n = 9 cells.

---

## 7. The pilot suite — the only place dose actually varies

4 cells (o1 × seeds 17/23/31, o2 × seed 17) × 4 stages = 16 Δc, rescored from raw generations
with the audited scorer (verified: S4 → c = 0.7442, matching `diag/criterion_vs_stats.json`).
Here dose and answer length are near-orthogonal: flickr is long/zero-dose, vizwiz is
short/high-dose.

| task | dose | mean words | n_train | mean Δc |
|---|---|---|---|---|
| vizwiz | 0.436 | 1.41 | 3,800 | +0.5789 |
| textvqa | 0.058 | 1.56 | 8,000 | −0.3534 |
| scienceqa | 0.000 | 1.00 | 5,700 | +0.0284 |
| flickr | 0.000 | 17.93 | 8,000 | −0.0345 |

Residualising in the order asked for: Δc ~ dose R² = **0.578**; residual ~ log length adds
**0.024**; combined 0.588. Reversed: Δc ~ length R² = 0.023; residual ~ dose adds 0.516.
**On the suite where dose is measurable, dose is the variable and length adds almost nothing.**
Per cell then aggregated, dose gives within-cell mean r = +0.777, CI [+0.711, +0.842], 4/4
cells positive; length gives −0.140, CI [−0.527, +0.151], 1/4.

Warning attached: pilot `n_train` varies (3,800–8,000) and correlates with Δc at r = −0.835,
4/4 cells. With four tasks this is confounded one-to-one with task identity and cannot be
untangled — it is a caution about reading anything from four points, not a finding. On UCIT the
same variable is constant.

### 7.1 Out-of-sample test of the answer-length account — INCONCLUSIVE

Flickr30k captioning appears in both suites with comparable targets (UCIT frozen train mean
12.29 words; the same task's val targets measure 17.85, and the pilot's 17.93), so the pilot is
a free out-of-sample test of the account fitted on UCIT.

| | mean Δc | +/n | 95% CI |
|---|---|---|---|
| UCIT Flickr30k (9 cells) | +0.1629 | 9/9 | [+0.107, +0.226] |
| pilot flickr (4 cells) | −0.0345 | 3/4 | [−0.351, +0.175] |

The pooled pilot mean hides everything. By ordering: **o1 (3 cells) +0.1187, positive 3/3** —
agreeing in sign and roughly in size with UCIT. **o2 (1 cell) −0.4941.** The pilot's o2 puts
flickr at position 2, immediately after vizwiz, the 43.6%-refusal +0.58 stage.

**Verdict: neither confirmed nor refuted.** The 4-cell interval contains the UCIT value; the
sign flip rests on a single cell in a single ordering; and that cell is itself an adjacency
observation, which is the §6 hypothesis appearing again rather than independent evidence
against length. Anyone quoting this as a refutation would be over-reading one cell.

**Do not confuse this with the paper's own pilot Flickr number.** `paper/main.tex` L1516 reports
Flickr at **−0.044, which is Δyes-rate**, not Δc, and it is seed 17 / o1 only. In Δc terms that
same cell is **+0.2111** — same direction as UCIT's Flickr30k, because a falling yes-rate is a
rising (more conservative) criterion. So the paper's "Flickr relaxation" and the answer-format
candidate are **not** in conflict on o1; they point the same way. This closes, in the direction
of agreement, the `paper/ADVERSARIAL_REVIEW.md` item "Reconcile the Flickr relaxation reading
with the format candidate — they cannot both be the explanation." The only tension left is the
single o2 cell.

---

## 8. The open question, stated precisely

> A continual instruction-tuning stream whose training targets contain no yes/no and no refusal
> content displaces the POPE criterion by 0.40–0.55 units in 9 of 9 cells. Task identity
> predicts 68% of the per-stage variance of that displacement; of the six per-task properties
> recorded, only answer length/format tracks it (46–50% of total variance, but at the design's
> p-floor of 0.067 and perfectly confounded with "Flickr30k and VizWiz specifically"); training
> magnitude is constant and cannot contribute; prompt distribution, depth, ordering, seed,
> mean reversion and generation-length state are all ruled out. **We do not know which property
> of a task sets its criterion pull, and this design cannot find out: with 6 tasks and 9 cells
> every candidate property is a per-task constant, so all of them are the same hypothesis as
> "task identity" wearing different labels.**

Not a power problem that more seeds would fix. Adding seeds shrinks the 0.177 noise floor and
leaves the confound untouched.

## 9. Cheapest experiment that would settle it

**Already specified, already built, and it costs 5 single-stage runs.**
`analysis/mechanism_readout.py --mode dose` is written and reads arms
`fsL_dose{w1,w2,w4,w8,full}_o1_s17`: one stage on **Flickr30k truncated to k ∈ {1, 2, 4, 8,
untruncated} words**. No results for those arms are on disk locally (`fs_aggregate.json`
predates them; `fullstudy/results_fs` holds none; `fullstudy/queue.txt` has no dose lines) —
**check the cluster before re-queuing**, they may already be running.
`paper/ADVERSARIAL_REVIEW.md` independently asks for the same experiment with
k ∈ {1, 3, 6, 12}; either grid works, the point is intermediate lengths. Same images, same prompts, same row count, same 125 optimizer steps —
only answer length varies. That single change does three things at once:

1. Breaks the answer-length / task-identity confound, which no analysis of the existing 6×9
   matrix can break.
2. Turns the bimodal two-group contrast (four tasks at ~1 word, two at ~12) into a genuine
   five-point dose–response, so the p-floor of 0.067 stops binding.
3. The pre-registered prediction is already recorded in
   `data_ucit/tasks/Flickr30k_lengthdose_report.json`: Δc **monotone in k, negative at k = 1,
   positive untruncated.** A flat profile refutes the format account and is to be reported as a
   refutation, not reinterpreted.

Cost: 5 single-stage LoRA runs (8,000 rows, 125 steps each) plus POPE eval — roughly one stage
of one cell, ×5. Seeds 23 and 31 would cost 10 more and are what makes the endpoint
cell-bootstrappable; run seed 17 first as a gate.

Second, much cheaper, and it tests §6 rather than §3: a **fourth ordering** o4 chosen so that
the predecessor→current pairs are disjoint from o1/o2/o3. The predecessor effect predicts the
per-task residuals shift by the table in §6; task identity alone predicts they do not. One
ordering × 3 seeds = 18 stage-runs, and it is the only out-of-sample test the adjacency
hypothesis can get from this benchmark.

Third, free, and it should be read before either: the single-task controls `fsL_single_<task>`
(already queued per the prereg) estimate each task's pull with **no predecessor at all**. If
those reproduce the §4.3 task means, the predecessor component is small and §6 shrinks further;
if they do not, the gap is the adjacency effect measured directly.

## 10. What must not be written

- "Answer statistics explain X% of the zero-dose drift." They explain 0.0%; the dose regressor
  has no variance on UCIT. The R² = 0.101 in the JSON is a degenerate fit, kept only so it is
  not rediscovered later.
- "Transitions explain 81% of Δc." That η² is level count plus the task means it contains. The
  incremental figure is 0.125 all-stages, 0.079 post-settling.
- "Δc tracks d′, so the criterion and sensitivity move together." Arithmetic, not a result.
- "Adjacency/interference is the mechanism." 5–9% of variance, chosen post-hoc, not separable
  from depth or ordering within a task, and not yet tested out of sample.
- "Training magnitude is ruled out." Optimizer steps and example count are ruled out for UCIT.
  Gradient and loss magnitude are NO-DATA.
- "The paper's pilot Flickr −0.044 contradicts the format account." That is Δyes-rate; the
  Δc for the same cell is +0.2111 and agrees with UCIT.

## 11. Side effects on open review items

Two analysis asks in `paper/ADVERSARIAL_REVIEW.md` are answered here in passing:

- *"Report ω² instead of η², and state it as 'depth adds no detectable variance once task is in
  the model'."* — §4.1 reports ω² throughout; §4.2 is exactly the once-task-is-in-the-model
  test, with the within-task permutation null. Depth: ω² 0.035, p = 0.178 (0.011, p = 0.246
  post-settling). The statement is supported.
- *"Run the depth test including stage 1, sequential arm only."* — §4.1 and §4.2 are the
  all-stages version; §4.2's post-settling rows are the comparison. Depth explains nothing in
  either, so the objection does die.
