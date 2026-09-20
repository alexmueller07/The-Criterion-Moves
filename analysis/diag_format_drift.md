# A9/A10 diagnostic: format-collapse and answer-drift audit

Run 2026-08-30 by `analysis/diag/format_drift.py` (full numbers in
`analysis/diag/format_drift.json`). Data: `results/{S0..S4,J1..J4}/{task}_gen.jsonl`,
500 rows per task per checkpoint, ids verified aligned across all 9 checkpoints
(paired bootstrap valid, N=2000, seed 20260830). The script re-derives every strict
score from raw generations and asserts equality with the pilot's `task_*.json`
summaries — all 36 cells match.

**Scoring definitions** (A9 spec):

- ScienceQA — *reported* = the pilot's parsed-only accuracy (ok/(n−fail));
  *strict* = parse fails count wrong (ok/n); *lenient* = strict-correct OR the
  correct option's text appears (normalized contiguous-token containment) in the
  output; *lenient_excl* additionally rejects a text match when another option's
  text also appears.
- TextVQA/VizWiz — *strict* = the pilot's VQA accuracy (exact normalized match,
  leave-one-out formula); *lenient* = same formula with token-subsequence
  containment; *lenient_ng* adds A9's negation guard (reject a match with a
  negation token within 3 tokens before the span).
- Flickr — uF1 only (no strict/lenient distinction).

**Deviation from the tasking note:** the letter→choice-text map is *not* in
`meta` (only `answer_letter`), but it is fully recoverable from the prompt's
`"A. <text>"` lines — verified by assertion for **all 4,500 ScienceQA rows**
(9 checkpoints × 500), so the option-TEXT lenient scorer was used; the
"any standalone A–E letter anywhere" fallback was never needed. Named divergence
regimes for the lenient scorers, both empirically inert here: (i) multi-option
outputs — `lenient_excl` equals `lenient` at every checkpoint, so no output ever
matched two option texts; (ii) negated containment ("not a cat" ⊃ "cat") —
`lenient_ng` equals `lenient` at every checkpoint, so zero negated matches occurred.

---

## 1. Answer length (words) and drift vs S0

| ckpt | scienceqa mean/med/p90 (Δmean) | textvqa (Δmean) | flickr (Δmean) | vizwiz (Δmean) |
| --- | --- | --- | --- | --- |
| S0 | 1.00/1/1 (+0.00) | 1.38/1/2 (+0.00) | 11.67/11/15 (+0.00) | 1.15/1/1 (+0.00) |
| S1 | 1.00/1/1 (+0.00) | 1.36/1/2 (−0.02) | 12.23/12/15 (+0.55) | 1.15/1/1 (−0.00) |
| S2 | 1.00/1/1 (+0.00) | 1.41/1/2 (+0.03) | 10.78/10/14 (−0.90) | 1.09/1/1 (−0.06) |
| S3 | 1.00/1/1 (+0.00) | 1.43/1/2 (+0.05) | 16.12/15/22 (+4.45) | 1.11/1/1 (−0.04) |
| S4 | 1.01/1/1 (+0.01) | 1.30/1/2 (−0.08) | 13.45/13/18 (+1.78) | 1.19/1/2 (+0.04) |
| J1 | 1.00/1/1 (+0.00) | 1.44/1/2 (+0.06) | 15.50/15/20 (+3.83) | 1.15/1/2 (+0.00) |
| J2 | 1.00/1/1 (+0.00) | 1.39/1/2 (+0.01) | 15.74/15/21 (+4.06) | 1.21/1/2 (+0.06) |
| J3 | 1.00/1/1 (+0.00) | 1.40/1/2 (+0.02) | 15.82/15/21 (+4.15) | 1.21/1/2 (+0.06) |
| J4 | 1.00/1/1 (+0.00) | 1.40/1/2 (+0.02) | 15.94/15/21 (+4.27) | 1.22/1/2 (+0.07) |

**Plain answer:** answer-length distributions on the three QA tasks are
essentially frozen across the whole sequence (Δmean ≤ 0.08 words; truncation 0
everywhere except TextVQA S0 at 0.2%). The only length movement is *within*
Flickr: captions lengthen at S3 (11.7 → 16.1 words) — toward the Flickr30k
training style, and the joint arm lands in the same place (J4 = 15.9), so this is
target-style learning, not sequential contamination.

## 2. Strict vs lenient scores

ScienceQA:

| ckpt | reported (parsed-only) | strict (fail=wrong) | parse-fail % | lenient (text OK) | lenient_excl | bare-letter % | text-rescued n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | 0.5840 | 0.5840 | 0.0 | 0.5860 | 0.5860 | 100.0 | 0 |
| S1 | 0.8180 | 0.8180 | 0.0 | 0.8200 | 0.8200 | 100.0 | 0 |
| S2 | 0.8213 | 0.8180 | 0.4 | 0.8220 | 0.8220 | 99.6 | 2 |
| S3 | 0.8280 | 0.8280 | 0.0 | 0.8280 | 0.8280 | 100.0 | 0 |
| S4 | 0.8041 | 0.7880 | 2.0 | 0.8040 | 0.8040 | 98.0 | 8 |
| J1 | 0.7280 | 0.7280 | 0.0 | 0.7300 | 0.7300 | 100.0 | 0 |
| J2 | 0.7700 | 0.7700 | 0.0 | 0.7720 | 0.7720 | 100.0 | 0 |
| J3 | 0.7840 | 0.7840 | 0.0 | 0.7840 | 0.7840 | 100.0 | 0 |
| J4 | 0.7900 | 0.7900 | 0.0 | 0.7900 | 0.7900 | 100.0 | 0 |

TextVQA / VizWiz (G = lenient − strict):

| ckpt | textvqa strict | lenient | lenient_ng | G | vizwiz strict | lenient | lenient_ng | G |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | 0.5162 | 0.5296 | 0.5296 | +0.0134 | 0.5574 | 0.5624 | 0.5624 | +0.0050 |
| S1 | 0.5012 | 0.5134 | 0.5134 | +0.0122 | 0.5746 | 0.5764 | 0.5764 | +0.0018 |
| S2 | 0.5710 | 0.5926 | 0.5926 | +0.0216 | 0.5622 | 0.5668 | 0.5668 | +0.0046 |
| S3 | 0.5828 | 0.6000 | 0.6000 | +0.0172 | 0.5612 | 0.5630 | 0.5630 | +0.0018 |
| S4 | 0.5386 | 0.5576 | 0.5576 | +0.0190 | 0.7112 | 0.7306 | 0.7306 | +0.0194 |
| J1 | 0.5430 | 0.5710 | 0.5710 | +0.0280 | 0.6976 | 0.7132 | 0.7132 | +0.0156 |
| J2 | 0.5628 | 0.5844 | 0.5844 | +0.0216 | 0.6914 | 0.7126 | 0.7126 | +0.0212 |
| J3 | 0.5630 | 0.5798 | 0.5798 | +0.0168 | 0.7068 | 0.7338 | 0.7338 | +0.0270 |
| J4 | 0.5630 | 0.5778 | 0.5778 | +0.0148 | 0.7116 | 0.7398 | 0.7398 | +0.0282 |

Flickr uF1: S0 0.5257, S1 0.5294, S2 0.5342, S3 0.5365, S4 0.5517;
J1 0.5333, J2 0.5406, J3 0.5331, J4 0.5357.

**Plain answers:**

- ScienceQA never leaves letter format wholesale: bare-letter share stays ≥ 98%.
  The one real format event is at S4, where 10 outputs (2.0%) are unparseable and
  **8 of the 10 are the correct option's TEXT** ("a liquid", "a gymnosperm", …).
  Lenient scoring recovers exactly those 8 rows: strict 0.7880 → lenient 0.8040,
  which coincides with the reported parsed-only number. A9's parsed-only-denominator
  risk is confirmed in direction but tiny in size at pilot scale (reported flatters
  strict by 1.6 pp at S4, 0 elsewhere on the S arm).
- TextVQA/VizWiz format gap G is small (+0.002 to +0.028) and **not growing with
  stage distance** — SEFE's pattern is absent. Verbosity is not eating VQA scores
  (consistent with the frozen length distributions in §1). The negation guard
  never fires.

## 3. "Unanswerable" rate on ALL four tasks (eq% / contains%, n = 500 each)

| ckpt | scienceqa | textvqa | flickr | vizwiz | vizwiz GT majority-unans % |
| --- | --- | --- | --- | --- | --- |
| S0 | 0.0 / 0.0 (n=0) | 0.0 / 0.0 (n=0) | 0.0 / 0.0 (n=0) | 55.6 / 55.6 (n=278) | 34.6 |
| S1 | 0.0 / 0.0 (n=0) | 0.0 / 0.0 (n=0) | 0.0 / 0.0 (n=0) | 55.4 / 55.4 (n=277) | 34.6 |
| S2 | 0.0 / 0.0 (n=0) | 1.0 / 1.0 (n=5) | 0.0 / 0.0 (n=0) | 84.6 / 84.6 (n=423) | 34.6 |
| S3 | 0.0 / 0.0 (n=0) | 0.0 / 0.0 (n=0) | 0.0 / 0.0 (n=0) | 85.0 / 85.0 (n=425) | 34.6 |
| S4 | 0.0 / 0.0 (n=0) | **14.6 / 14.6 (n=73)** | 0.0 / 0.0 (n=0) | 57.4 / 57.4 (n=287) | 34.6 |
| J1 | 0.0 / 0.0 (n=0) | 3.6 / 3.6 (n=18) | 0.0 / 0.0 (n=0) | 65.0 / 65.0 (n=325) | 34.6 |
| J2 | 0.0 / 0.0 (n=0) | 1.8 / 1.8 (n=9) | 0.0 / 0.0 (n=0) | 52.8 / 53.0 (n=265) | 34.6 |
| J3 | 0.0 / 0.0 (n=0) | 2.4 / 2.4 (n=12) | 0.0 / 0.0 (n=0) | 55.4 / 55.6 (n=278) | 34.6 |
| J4 | 0.0 / 0.0 (n=0) | 2.2 / 2.2 (n=11) | 0.0 / 0.0 (n=0) | 56.4 / 56.6 (n=283) | 34.6 |

**Plain answer — the leakage is real, large, TextVQA-specific, and mostly
sequential.** After the VizWiz stage, 73/500 TextVQA outputs (14.6%) are exactly
"Unanswerable" (all 73 are pure abstentions — eq = contains), vs 0 at S0/S1/S3
and 5 at S2. The joint arm, trained on the *same* VizWiz data at matched volume,
leaks only 2.2–3.6% (J4 = 11/500), so the sequential excess is **12.4 pp,
~6.6× the joint arm** — a recency effect, not merely exposure to VizWiz data.
The leakage is confined to TextVQA, whose prompt ("Answer the question using a
single word or phrase.") is the VizWiz prompt minus the Unanswerable instruction;
ScienceQA (MC format) and Flickr (caption format) show 0/500 at every checkpoint.
So this is prompt-format-conditional abstention transfer, not a global refusal mode.

Decomposition of the 73 leaked rows (script cross-section, in
`format_drift.json` + §5 below): they were already the hard slice (strict on
those rows at peak S2 = 0.207 vs 0.633 on the rest) — the model abstains
preferentially where it was wrong anyway; at S4 they score 0.110 (16/73 rows
have ≥1 TextVQA annotator answering "unanswerable", mean annotator-unans
fraction 0.040, so abstention occasionally earns partial credit).

**Bonus finding (feeds A1's criterion story):** VizWiz's *flat* pre-stage score
hides a large abstention-criterion swing. Split by GT majority answerability
(173 unans / 327 answerable):

| ckpt | abstain % | strict all | strict GT-unans | strict GT-answerable | abst-recall (GT-unans) | abst-FPR (GT-ans) |
| --- | --- | --- | --- | --- | --- | --- |
| S0 | 55.6 | 0.5574 | 0.8075 | 0.4251 | 75.1 | 45.3 |
| S1 | 55.4 | 0.5746 | 0.8399 | 0.4343 | 78.0 | 43.4 |
| S2 | 84.6 | 0.5622 | 0.9399 | 0.3624 | 93.6 | 79.8 |
| S3 | 85.0 | 0.5612 | 0.9584 | 0.3511 | 93.6 | 80.4 |
| S4 | 57.4 | 0.7112 | 0.9324 | 0.5942 | 89.6 | 40.4 |
| J1 | 65.0 | 0.6976 | 0.9526 | 0.5627 | 93.6 | 49.8 |
| J2 | 52.8 | 0.6914 | 0.9185 | 0.5713 | 88.4 | 33.9 |
| J3 | 55.4 | 0.7068 | 0.9428 | 0.5820 | 89.6 | 37.3 |
| J4 | 56.4 | 0.7116 | 0.9474 | 0.5869 | 90.8 | 38.2 |

At S2/S3 (after TextVQA/Flickr stages, before VizWiz training) the model abstains
on 85% of VizWiz — false-abstention on answerable questions nearly doubles
(45% → 80%) while abstention recall rises (75% → 94%), and the two movements
cancel to a flat headline 0.56. The aggregate VQA score is criterion-blind here;
any VizWiz claim in the paper must use the split, not the aggregate.

## 4. Caption-style-output rate on non-caption tasks (> 15 words)

From §1b of the script output: ScienceQA 0.0% at every checkpoint; TextVQA 0.2%
at S0 and 0.0% everywhere else; VizWiz 0.2% at S3 (1/500) and 0.0% everywhere
else. Flickr itself: 6.2% (S0) → 49.8% (S3) → 21.4% (S4); joint arm ≈ 45–50%.

**Plain answer:** the hypothesized Flickr-stage style leakage into the QA tasks at
S3 **did not happen** — caption-style outputs on non-caption tasks are ≤ 1 row per
500 at every checkpoint. The cross-task contamination in this pilot travels
through *abstention* (VizWiz → TextVQA at S4, §3), not through verbosity.

## 5. Forgetting matrix, strict vs lenient side by side

F_t(k) = score_t(S_stage(t)) − score_t(S_k), paired bootstrap SE (2000 resamples
over the 500 shared items); `*` = |F| > 2·SE.

| task | peak | ckpt | F_strict (SE) | F_lenient (SE) |
| --- | --- | --- | --- | --- |
| scienceqa | S1 | S2 | +0.0000 (0.0159) | −0.0020 (0.0160) |
| scienceqa | S1 | S3 | −0.0100 (0.0145) | −0.0080 (0.0145) |
| scienceqa | S1 | S4 | **+0.0300 (0.0150)\*** | +0.0160 (0.0137) |
| textvqa | S2 | S3 | −0.0118 (0.0108) | −0.0074 (0.0107) |
| textvqa | S2 | S4 | **+0.0324 (0.0139)\*** | **+0.0350 (0.0141)\*** |
| flickr | S3 | S4 | **−0.0153 (0.0043)\*** | (uF1, same) |

**Does any forgetting appear/disappear with leniency?** Yes, one cell each way
of the ledger:

- **ScienceQA@S4 disappears.** Significant under strict (+3.0 pp, 2·SE = 3.0),
  not significant under lenient (+1.6 pp, 2·SE = 2.7). Roughly half the strict
  drop is the 8 letter→text format rows; the remainder is inside the noise at
  n = 500. Under the pilot's own reported (parsed-only) convention F would be
  just +1.4 pp — i.e., even its *significance* under "strict" is a
  scoring-convention artifact, exactly the A9 mechanism.
- **TextVQA@S4 survives leniency** (+3.2 pp strict, +3.5 pp lenient) — because
  the damage is abstention, which containment cannot rescue. Decomposing on the
  73 leaked rows: they contribute **+1.42 pp of the +3.24 pp strict forgetting
  (44%)**; the non-leaked 427 rows drop 0.633 → 0.612 and contribute +1.82 pp
  (56%). (Divergence regime, named: this split conditions on S4 behavior; the
  unobservable counterfactual is what abstaining rows would have scored had they
  answered — vs the S3 checkpoint instead of S2 the leaked-row share is the same
  44%, so the attribution is stable.)
- No other cell is significantly forgotten under either scorer; Flickr *improves*
  at S4 (anti-forgetting, −1.5 pp uF1). At S2 and S3, **no task is significantly
  forgotten under lenient scoring** — relevant to A10's money cell: if the
  CHAIR/POPE trajectories (out of scope for this file) show hallucination growth
  at S2/S3, that growth occurs with essential forgetting statistically at zero.

## Verdict

**How much of the task-accuracy story is format drift vs knowledge change?**

- **Classic SEFE-style format collapse is absent.** Length distributions frozen
  on QA tasks, bare-letter rate ≥ 98%, VQA containment gap G small
  (≤ +2.8 pp) and not growing with stage distance, zero truncation, zero
  caption-style leakage into QA tasks. The forgetting numbers are *not* verbosity
  artifacts, and A10 may use the lenient matrix as its essential-forgetting input.
- **What drift exists is behavioral (answer-selection), concentrated at S4, and
  splits cleanly:** (i) ScienceQA's only significant forgetting cell is ~half
  letter→text format drift and fully non-significant once text answers are
  credited — treat ScienceQA as *not* essentially forgotten; (ii) TextVQA's
  forgetting is real under any scorer but **44% of it is VizWiz abstention
  leakage**, leaving ≈ +1.8 pp genuine decline; (iii) VizWiz's flat S2/S3 score
  conceals an 80% false-abstention rate — criterion drift the aggregate metric
  cannot see.
- **S4 cross-task "unanswerable" leakage: real, large, and sequence-specific.**
  TextVQA 14.6% (73/500 pure "Unanswerable" outputs) vs 0% at S0/S1/S3 and
  2.2% in the volume-matched joint arm at the same step (J4 = 11/500):
  a 12.4 pp sequential excess (~6.6×). It is prompt-format-conditional
  (short-answer VQA prompts only; 0/500 on ScienceQA and Flickr at every
  checkpoint), and it lands preferentially on questions the model was already
  getting wrong (peak strict 0.21 on the leaked slice vs 0.63 elsewhere).
- **Consequence for the A10 decomposition:** essential forgetting at pilot scale
  reduces to a single ≈ 1.8 pp TextVQA decline at S4; everything else that looked
  like forgetting is scoring convention, abstention transfer, or noise. The
  forgetting axis is therefore much weaker than the raw matrix suggests — which
  *sharpens* the independent-axis test: any hallucination growth at S2/S3 (per
  A3/A10, from the CHAIR/POPE files) cannot be attributed to knowledge loss on
  these tasks. Conversely, any "VizWiz recovery/gain" narrative at S4 must be
  quoted with the criterion split from §3.

Repro: `python3 analysis/diag/format_drift.py` (reads `results/`, writes
`analysis/diag/format_drift.json`, prints these tables; imports parsers and
normalizers from `pilot/metrics_task.py`).
