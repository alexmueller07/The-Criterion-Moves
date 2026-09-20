# Diagnostic A3 — length-controlled CHAIR

Pre-specified in `design_notes/analysis_ideas.md` §A3 (KILL-class for the CHAIR arm of
gates 1/2). Code: `analysis/diag/length_controlled_chair.py`; numbers:
`analysis/diag/length_controlled_chair.json`. Extraction and synonym machinery imported
from `pilot/metrics_chair.py` (nothing re-vendored; the position-tagged walker is
asserted equal to `extract_object_mentions` on every caption).

**Self-check:** recomputed full CHAIR_i / CHAIR_s / mentions-per-caption from
`chair_gen.jsonl` + `results/coco_gt.json` match `results/*/chair.json` exactly for all
9 checkpoints (assertions, not eyeballing).

**Deviation from the A3 note, stated:** budgets k are counted in *whitespace words*
(per the diagnostic task spec), not the extractor's `[a-z]+` tokens the note mentions;
the two streams differ only on punctuation/digit splitting. Truncation = first k
whitespace words, mentions re-extracted from the prefix, hallucination scored against
the identical GT sets (instances ∪ caption-extracted, exactly as `metrics_chair.py`).
Paired bootstrap = images, B=10,000, seed 17, ratio-of-sums, mirroring the gate.

## 1. CHAIR_i, full and length-controlled

| ckpt | full | k=30 | k=60 | k=90 | mean words | median words | mentions/100w |
|---|---|---|---|---|---|---|---|
| S0 | 0.1564 | 0.0463 | 0.1052 | 0.1549 | 83.4 | 86 | 8.97 |
| S1 | 0.1372 | 0.0418 | 0.0870 | 0.1297 | 88.6 | 91 | 8.55 |
| S2 | 0.1443 | 0.0427 | 0.0936 | 0.1375 | 88.3 | 91 | 8.67 |
| S3 | 0.1504 | 0.0485 | 0.1016 | 0.1415 | 89.3 | 90 | 8.61 |
| S4 | 0.1273 | 0.0382 | 0.0894 | 0.1212 | 81.0 | 84 | 8.81 |
| J1 | 0.1494 | 0.0413 | 0.1000 | 0.1435 | 88.7 | 90 | 8.57 |
| J2 | 0.1650 | 0.0449 | 0.1022 | 0.1601 | 89.7 | 92 | 8.57 |
| J3 | 0.1501 | 0.0461 | 0.0909 | 0.1413 | 89.9 | 91 | 8.60 |
| J4 | 0.1561 | 0.0441 | 0.0904 | 0.1467 | 90.2 | 92 | 8.62 |

CHAIR_s shows the same pattern (S4 full 0.456 vs J4 0.502; at k=60 S4 0.314 vs J4
0.300 — S4 is nominally *worse* caption-level at matched length). Budget bindingness:
k=60 truncates 81–93% of captions in every checkpoint (clean shared budget); k=90 binds
asymmetrically (S4 median 84 words < 90, J4 median 92 > 90), so k=90 partially
re-admits the length difference and is not a clean control — k=30/k=60 are.

Bootstrap on the two load-bearing gaps:

| gap | full | k=60 |
|---|---|---|
| S4 − J4 | **−0.0288** [−0.0433, −0.0144] | **−0.0010** [−0.0134, +0.0113] |
| S3 − S1 | +0.0132 [−0.0010, +0.0273] | **+0.0147** [+0.0023, +0.0279] |

## 2. Position of hallucinations (quartiles of relative token position, pooled)

| ckpt | Q1 | Q2 | Q3 | Q4 | Q4/Q1 | Q4 share of hal |
|---|---|---|---|---|---|---|
| S0 | 0.0368 | 0.1111 | 0.2455 | 0.2805 | 7.6 | 0.330 |
| S1 | 0.0376 | 0.1109 | 0.2268 | 0.2096 | 5.6 | 0.287 |
| S2 | 0.0385 | 0.0959 | 0.2436 | 0.2448 | 6.3 | 0.320 |
| S3 | 0.0464 | 0.0981 | 0.2578 | 0.2369 | 5.1 | 0.313 |
| S4 | 0.0259 | 0.0961 | 0.2045 | 0.2259 | 8.7 | 0.335 |
| J1 | 0.0325 | 0.1060 | 0.2666 | 0.2336 | 7.2 | 0.294 |
| J2 | 0.0401 | 0.1128 | 0.2848 | 0.2725 | 6.8 | 0.306 |
| J3 | 0.0400 | 0.1090 | 0.2523 | 0.2456 | 6.1 | 0.310 |
| J4 | 0.0373 | 0.1129 | 0.2562 | 0.2661 | 7.1 | 0.334 |

## Answers

**(a) Does S4's CHAIR_i advantage over J4 survive at fixed k=60?** No. The full-caption
gap −0.0288 (CI excludes 0) collapses to −0.0010 at k=60 (CI [−0.0134, +0.0113],
straddles 0); at k=30 it is −0.0059 and at CHAIR_s@60 the sign flips against S4. The
advantage is entirely explained by S4's shorter captions (81.0 vs 90.2 mean words;
104.8 vs 116.8 mean tokens): S4's assertion *density* is not lower — 8.81 mentions/100
words vs J4's 8.62, and mentions/caption 7.13 vs 7.78 — it simply stops earlier, which
mechanically deletes the late, high-hazard portion of the caption (Q3/Q4 hazards are
5–9× Q1 at every checkpoint). This is A3 outcome 2: **ARTIFACT** as a grounding claim
for the endpoint; per the pre-specified severity, the S4-vs-J4 CHAIR arm of gate 2
cannot be claimed as "SEQ more faithful".

**(b) Is the S1→S3 rise present under length control?** Yes, and it sharpens. Full:
0.1372→0.1504 (+0.0132, CI [−0.0010, +0.0273], marginal). At k=60: 0.0870→0.1016
(+0.0147, CI [+0.0023, +0.0279], excludes 0); at k=30: 0.0418→0.0485; at k=90:
0.1297→0.1415. S1 and S3 have nearly identical lengths (88.6 vs 89.3 words), so no
length confound is available to explain it. Mid-sequence per-mention grounding
genuinely degrades — an AXIS-compatible signal (subject to the n=1-seed caveat: no
trajectory noise floor exists at pilot scale).

**(c) Do hallucinations concentrate late, and does that shift across stages?** Strongly
late at *every* checkpoint including base S0: pooled hazard rises from ~0.03–0.05 in Q1
to ~0.20–0.28 in Q3/Q4 (Q4/Q1 ≈ 5–9), and Q3+Q4 carry ~60% of all hallucinated
mentions. This is the EOS-literature signature (2402.14545), but it is a property of
the model family, not something sequential tuning creates. Across stages the shifts are
modest and non-monotone: early-position hazard creeps up S1→S3 (0.0376→0.0464, mildly
against a pure-EOS account for the mid-sequence rise), and S4's high Q4/Q1 = 8.7 comes
from a *drop* in early hazard (Q1 0.0259), not a rise in late hazard. No systematic
SEQ-vs-JOINT difference in concentration.

## Implication (generative story)

The endpoint story is "SEQ just asserts less": S4's raw CHAIR advantage over J4 is
produced by earlier termination (plausibly a residue of the Flickr one-sentence stage
plus VizWiz short answers) that truncates the late high-hazard region, and at matched
length S4 and J4 are indistinguishable (gap −0.001). What survives length control is
the mid-sequence S1→S3 per-mention degradation (+0.0147 at k=60, CI excludes 0), so the
defensible generative claim is a within-sequence faithfulness decline, not endpoint
superiority of sequential over joint training. Per A3's KILL scope, any gate-2 CHAIR
endpoint claim must be reworded as length/EOS-mediated, and the trajectory claim should
lean on the length-controlled numbers (with the single-seed caveat carried through).
