# Pilot Findings — Faithfulness Dynamics under Continual Adaptation of an MLLM

Date: 2026-08-31. Pilot: LLaVA-1.5-7B LoRA, ScienceQA→TextVQA→Flickr30k→VizWiz (5,700/8,000/8,000/3,800),
SEQ vs data/step-matched JOINT control, evals at every stage boundary (POPE 9k ×3 splits, CHAIR 500,
4×500 task vals). All scorers red-team-audited and fixed pre-readout; all 9 checkpoints scored
identically; 0 parse failures on POPE (81,000/81,000). One seed, one ordering — pilot-grade.
Diagnostics A1/A3/A9-A10/A11 were pre-specified in design_notes/analysis_ideas.md before any
result file existed.

## Headline

**Catastrophic forgetting is nearly absent at pilot scale — yet faithfulness behavior swings
dramatically.** The pre-registered gate formally passed, but the pre-specified diagnostics show
both of its endpoint branches are artifacts, and reveal three real phenomena that endpoint
evaluation (the field's standard practice) cannot see:

### 1. Criterion drift: fast, recency-driven, dose-dependent, reversible (A1 + A11)

- Discrimination is FLAT: pooled d′ 2.348 (S0) → 2.391 (S4); endpoint change +0.04. The model
  never gets worse at telling present from absent.
- The answer criterion c swings 0.78 units across the sequence — 4× d′'s range, 7× its endpoint displacement [ratio corrected 2026-08-31 per the statistics review; originally stated as ~20× by pairing c's range with d′'s endpoint change]:
  liberal after TextVQA (yes-rate 41→55%, adv precision .895→.796 — the classic hallucination
  *signature* without any grounding loss), conservative after VizWiz (yes-rate →35%,
  recall .87→.67, precision .933 = best in the run).
- Dose-response: each stage moves c in the direction of its own training answer statistics,
  magnitude ordered by supervision strength (VizWiz 43.2% "unanswerable" → largest move,
  −0.114 yes-rate, overshooting below base; TextVQA yes:no 3.4:1 → only large upward move,
  +0.076; zero-yes/no stages barely move it — and Flickr RELAXES ~58% of TextVQA's push back).
- **Sequencing, not volume**: the volume-vs-sequencing decomposition attributes ~100% of the
  endpoint criterion displacement to the sequential arm specifically (JOINT is flat after J2);
  this is a continual-learning phenomenon, not the known single-stage SFT effect.

### 2. Behavior leakage across tasks (A9)

VizWiz's refusal behavior leaks into TextVQA at S4: 14.6% of TextVQA answers become the exact
string "Unanswerable" vs 2.2% in the volume-matched joint arm (6.6×; recency, not exposure).
Prompt-format-conditional (0/500 on MC and caption prompts). Accounts for 44% of the apparent
TextVQA "forgetting" at S4. Classic format collapse is otherwise ABSENT (answer lengths frozen,
caption-style leakage ≤1/500 everywhere).

### 3. A genuine, forgetting-decoupled generative hallucination rise mid-sequence (A3 + A10)

- Length-controlled CHAIR_i (fixed 60-word budget) rises S1→S3: 0.087→0.102,
  paired-bootstrap CI [+0.0023, +0.0279] excluding zero, with near-identical caption lengths.
- Over the same S1→S3 window, NO task shows significant forgetting under lenient scoring —
  the rise is decoupled from knowledge loss. This is the cleanest "beyond forgetting" evidence
  in the pilot, stronger in kind than the pre-registered endpoint would have been.
- Total genuine ("essential") forgetting across the whole pilot: ≈1.8pp on one task (TextVQA
  residual at S4). ScienceQA's apparent forgetting dissolves under lenient scoring (option-text
  answers, a format effect).

## Why the pre-registered endpoints misled (and were caught)

- Gate branch POPE (passed): S4-vs-J4 ΔF1 = 0.048 (CI excludes 0) — but A1 shows it is entirely
  criterion placement (Δc ≈ 0.77, Δd′ ≈ +0.21 *favoring* SEQ). F1 conflates policy with grounding.
- Gate branch CHAIR (opposite direction): S4's advantage over J4 (−0.029, CI excludes 0)
  evaporates at fixed length (−0.001, CI straddles 0): S4 asserts less (stops earlier), with
  *higher* hallucination density per 100 words than J4. Raw CHAIR conflates length with faithfulness.
- Both artifacts flow from answer-policy drift (#1), which is itself a real finding.
- Late-caption hazard (5–9× first-quartile) is present at S0 — a model property, not tuning-induced.

## Paper framing (revised measurement claim)

Faithfulness under continual adaptation is not a monotone degradation; it decomposes into
(a) answer-criterion/assertion-policy drift — large, fast, recency-driven, dose-dependent on
stage answer statistics, reversible, sequencing-specific — and (b) a smaller genuine grounding
erosion (length-controlled generative hallucination rise) that accumulates mid-sequence while
task accuracy is fully retained. Standard endpoint metrics conflate the two, in opposite
directions on discriminative vs generative instruments. The third axis therefore needs
criterion-corrected instruments (d′, length-controlled CHAIR, leakage rates), which we provide.

**Method predictions (falsifiable, for the SEQ+F anchor arm):** the two-sided GT margin should
(i) shrink criterion swings (|Δc| per stage), (ii) suppress refusal leakage on non-VizWiz prompts,
(iii) reduce the length-controlled S1→S3 rise, (iv) leave new-task accuracy (plasticity) intact.
ER control predictions: if replay fixes (i)-(iii), the phenomena are forgetting-like after all —
that is the honest kill condition for the "distinct axis" claim.

## Caveats

One seed, one ordering, one backbone, 4 tasks, LoRA-only: every number above is pilot-grade.
The S1→S3 CI is an eval-set bootstrap (does not cover run-to-run variance — see the
trajectory-variance-floor risk). Full study (UCIT, 3 seeds × 3 orderings, Qwen2.5-VL second
backbone, AMBER) is required before any of this is a claim.

## Reframe novelty status (resolved 2026-08-31 — design_notes/reframe_novelty_check.md)

Two-vocabulary sweep complete (30 queries, 15 primary sources). Claims CLEAR: the SDT
criterion/discrimination decomposition of hallucination metrics across a tuning trajectory (1),
and refusal-behavior acquisition-then-leakage under CL (3). Claim PARTIAL: the general
"sequential-tuning degradation is reversible answer bias" headline is owned by arXiv 2510.08564v2
("How to Teach Large Multimodal Models New Skills") — zero POPE/hallucination/abstention/SDT
content in its full text, so our delta survives strictly as the conjunction (faithfulness
instruments + signed dose-response vs labeled answer statistics + sequencing-vs-volume via the
matched joint arm + d′-flat). MUST-CITE with in-paragraph differentiation; never claim the
general sentence. Watched papers 2607.02020 / 2602.18055 remain v1, grep-clean of
criterion/bias content.

## Robustness readout (2026-08-31) — seeds, reverse ordering, single-stage controls

Runs: 3 forward seeds × {SEQ, G (anchor v2)}, 1 reverse-ordering run of each
(VizWiz→Flickr→TextVQA→ScienceQA), and single-stage controls vw_only / tv_only at matched
steps. Aggregation: analysis/robustness_aggregation.py → analysis/diag/robustness_aggregation.json.
Reporting per the pre-stated multi-seed rules (design_notes/review_statistics.md §2): seed
values and min–max ranges with sign counts; no cross-seed CIs; controls (n=1 each) are
position statements only.

- **T1 — dose-response criterion trajectory replicates across seeds.** Per-stage pattern
  identical in 3/3 SEQ seeds: TextVQA pushes liberal (Δc −0.37/−0.30/−0.37), VizWiz pushes
  conservative past base (+0.57/+0.59/+0.71), zero-dose stage moves small in every seed.
  SEQ post-settling Σ|Δc| (stages 2–4) range 0.958–1.160.
- **T2/T1-rev — reverse ordering confirms recency.** The same four stages reversed put the
  endpoints on opposite sides of base (c=+0.431): forward S4 c=+0.744 (conservative) vs
  SEQ-REV end c=+0.217 (near-neutral after zero-dose ScienceQA). VizWiz-first spikes
  immediately to +0.871; the TextVQA stage drives c to 0.000. d′ stays in [2.18, 2.50]
  across all 26 robustness checkpoints — the pre-stated reordering prediction (endpoint
  criterion follows the new final stage's answer statistics, d′ unchanged) realized at
  pilot scale.
- **T3 — anchor criterion-freeze replicates 3/3.** G post-settling Σ|Δc| 0.178–0.385 vs
  SEQ 0.958–1.160 across seeds — non-overlapping ranges — and holds under reverse ordering
  (G-REV 0.351 vs SEQ-REV 1.087).
- **T4 — S1→S3 length-controlled rise sign-consistent 3/3.** Deltas +0.0147/+0.0179/+0.0137
  (chair_i@60 stage-1 0.0870/0.0827/0.0830 → stage-3 0.1016/0.1005/0.0968); each delta
  exceeds the cross-seed spread at fixed checkpoint (0.0043 at stage 1, 0.0049 at stage 3).
  Status upgraded from "exploratory, pending confirmation" to "replicated across seeds at
  pilot scale"; the full-study confirmatory endpoint is unchanged.
- **T5 — single-stage controls: the criterion is NEAR-MEMORYLESS (reinterpretation
  adopted).** VizWiz-only lands at c=+0.848 vs S4 +0.744 (inside the SEQ seed range
  [0.744, 0.871]); TextVQA-only at c=+0.003 vs S2 −0.034 (inside the seed range
  [−0.034, +0.110]). Single-stage arms reproduce the sequential endpoint criteria almost
  exactly, so the criterion is set by the most recent criterion-active supervision with
  history contributing only a small residual (gaps 0.104 and 0.037). The headline
  "Sequencing, not volume" above is hereby refined to **recency, not accumulation**: what
  JOINT shows is that MIXED supervision prevents drift; what the controls show is that
  history barely buffers it. The continual-learning relevance is perpetual criterion churn
  under sequential updates — each update re-sets the criterion — which the anchor prevents
  (and its freeze is what replicates in T3). The paper (2026-08-31 integration pass)
  adopts this sharpened claim at every site that previously said "sequencing-specific".
