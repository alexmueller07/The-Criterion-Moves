# Pilot Pre-registration — committed BEFORE any training run

Date: 2026-08-30. Per the quote-the-endpoint rule: the endpoint below is what gets reported,
regardless of which round/stage looks best afterward.

## Setup (fixed before launch)

- **Model:** llava-hf/llava-1.5-7b-hf, bf16. LoRA (r=64, alpha=128, dropout 0.05) on LLM
  attention+MLP projections; multimodal projector trainable; vision tower frozen.
- **Sequence (pilot):** ScienceQA → TextVQA → Flickr30k(captioning) → VizWiz(VQA).
  8,000 train examples per task, subsampled with seed 17. One epoch per stage,
  effective batch 64, lr 1e-4 cosine per stage, max seq len 2048.
  [Amended 2026-08-31 pre-launch, before any training run: the verified VizWiz source
  (lmms-lab/VizWiz-VQA) has no train split — only val (4,319) and test. Stage 4 therefore
  trains on 3,800 examples carved from val, with its 500 held-out items disjoint by
  construction. Second pre-launch amendment, same day: ScienceQA's train split contains only
  6,218 image-bearing rows (the rest are text-only), so stage 1 trains on 5,700 with 500
  held out. Stage sizes are therefore 5,700/8,000/8,000/3,800; the SEQ↔JOINT matching is
  unaffected (joint uses the identical 25,500-example union with checkpoints at the exact
  stage-boundary steps). No results existed at either amendment time.]
- **Arms:**
  - SEQ: train tasks in order, continuously updating the same adapter; checkpoint after each stage (S1..S4).
  - JOINT: the identical example union shuffled (seed 17; 25,500 examples after the VizWiz and
    ScienceQA amendments below), same effective batch/lr schedule over the same total steps;
    checkpoints at the exact cumulative sequential stage boundaries in optimizer steps —
    90/215/340 — plus J4 = the joint run's final checkpoint (399 steps). Jk is the
    data/step-matched twin of Sk. [Endpoint bookkeeping note, added 2026-08-31 after the SEQ
    arm launched but before any JOINT training step and before any eval result existed:
    per-stage partial batches make SEQ total 400 steps while the single-stream union rounds
    to 399, so an exact step-400 joint checkpoint cannot exist; J4 therefore trails S4 by at
    most one optimizer step (≤64 examples, ~0.25% of the data). No metric was computed
    before this note.] [Amended 2026-08-31
    pre-launch together with the VizWiz amendment: originally "32,000 examples, checkpoints at
    25/50/75/100%"; with unequal stage sizes the match must be at stage-boundary steps, not
    round fractions. Step numbers updated in the same pre-launch window when ScienceQA's
    real image-bearing size forced stage sizes 5,700/8,000/8,000/3,800.]
  - S0 = base model (no tuning), evaluated identically.
- **Evals at every checkpoint (S0, S1..S4, J1..J4):**
  - POPE (random/popular/adversarial, full 9k): accuracy, F1, **yes-rate**, parse-failure rate.
  - CHAIR on 500 COCO val2014 images sampled uniformly with seed 17 (GT = instance annotations
    ∪ objects extracted from the 5 GT captions, both mapped into the 80-category space via the
    repo-vendored synonym list): CHAIR_s, CHAIR_i, mean caption length, objects/caption,
    **truncation rate** at max_new_tokens=512.
    [Amended 2026-08-30 pre-launch, before any training run: original draft tied the image set
    to the POPE pool; switched to seeded-random val2014 with full COCO GT — closer to the
    Rohrbach protocol. No results existed at amendment time.]
  - Task accuracy on 500-example held-out sets per task (forgetting matrix), format-robust scorers
    with parse-failure rates reported, never silently dropped.
- Greedy decoding everywhere. All generations logged as JSONL with audit fields
  (length, truncated, parse_ok). Machine-parseable summary CSV per checkpoint.

## Scorer amendment (2026-08-31, before any SEQ/JOINT metric existed)

An independent red-team audit of the scoring code (design_notes/eval_code_audit.md),
commissioned before results, found 2 critical + 6 moderate defects. Fixes adopted:
word-boundary POPE yes/no parsing (no substring matches); strict MC-letter parsing
(article-"A" no longer parses as an answer); official per-mention CHAIR_i; plural
compound synonyms; official 10-choose-9 VQA averaging + number-word normalization;
gate branches that cannot be evaluated report NO-DATA instead of FAIL; EOS-aware
truncation flag; adapter provenance in generation rows. Timing disclosure: at adoption
time the ONLY results in existence were the S0 base-model baseline (which had been
viewed) and eval S1's in-progress generations; no SEQ-vs-JOINT comparison or trajectory
existed. All checkpoints, including S0 (rescored from its logged generations), are
scored with the identical fixed scorers. Generation-side audit fields (adapter,
truncated flag) for S0/S1 were produced under pre-fix semantics; the truncated-flag
difference matters only when generation stops exactly at the token budget with an EOS,
and is reported if it is ever nonzero.

## Primary endpoint

Hallucination trajectory = (CHAIR_i, POPE-adversarial F1) across S1→S4, compared point-wise
against J1→J4.

**The pilot "effect exists" gate passes iff BOTH:**
1. Worsening trajectory in SEQ: CHAIR_i(S4) > CHAIR_i(S1) AND at least 2 of {S2,S3,S4} are worse
   than S1 on CHAIR_i, or the same pattern on POPE-adversarial F1 (decreasing) — i.e., sustained,
   not a single-stage blip.
2. Sequential exceeds matched joint at the end: CHAIR_i(S4) − CHAIR_i(J4) > 0 and the gap's 95%
   bootstrap CI over eval images excludes 0 (10,000 resamples); or equivalently for POPE-adv F1
   (J4 − S4 > 0, CI excludes 0).

If the gate FAILS, the result goes to Alex and Prof. Park as a finding (with the full trajectory
plots), and scaling is halted pending discussion. It does not get quietly rerun with different
knobs. Confound checks reported alongside either way: yes-rate drift, truncation rate,
caption-length drift, parse-failure rates (a "hallucination increase" explainable entirely by
yes-bias or length inflation must be reported as such, not as grounding failure).

## Secondary / exploratory (reported, no gate)

- Correlation between per-task forgetting (accuracy drop from its own stage to S4) and
  hallucination growth; the accuracy-retained-but-hallucination-grown cell.
- POPE yes-rate and response-length trajectories as candidate mechanisms.
- Per-split POPE (random vs popular vs adversarial) divergence.

## Known limitations accepted for the pilot (fixed in full study, not silently)

- n=1 seed, 1 task order → no run-to-run noise floor yet; bootstrap CIs cover eval-set noise only.
- CHAIR image set: 500 seeded-random val2014 images (per the 2026-08-30 pre-launch amendment above); [stale text removed 2026-08-31: an earlier draft line still described the superseded POPE-pool tie].
- Task data budgets equalized at 8k (not CoIN's full sizes).
- LLaVA-1.5's own pretraining saw TextVQA-adjacent and COCO-domain data; S0 anchors this.
