# PCR-0: the method-vs-post-hoc test against a properly-aimed competitor — 2026-09-12

**Verdict up front, in three parts, because the honest answer is not one sentence:**

1. **On the axis we can measure today, the method loses, and it loses to both
   competitors — including the handicapped one.** The anchor family's endpoint
   criterion sits further from the target than a free scalar's landing point in
   **9 of 9 cells against the target-aimed competitor and 9 of 9 against the
   base-aimed one**. It is also beaten 9/9 by doing nothing at all (plain
   sequential). This was not new information in kind — the prereg has said the
   anchor is mis-placed since 2026-09-11 — but it is the first time the *competitor*
   has been placed on the same axis, and the competitor wins on it outright.
2. **The pre-registered boolean itself is NOT decidable on this machine, and this
   analysis does not decide it.** It needs per-item POPE / rephrased-POPE / probe
   logit dumps, which are on the cluster; `fullstudy/results_fs` holds generations
   only. The script now fails loudly with `ABSENT` there rather than emitting a
   number. **NO-DATA, not a negative.**
3. **The rigging turns out not to have distorted the pre-registered boolean much —
   it distorted which axis got reported.** Re-aiming the scalar is a *constant*
   added to every stage, and a constant shifts every stage's criterion equally when
   a cell's stages lie on one z-ROC (which the sequential reference does, mean
   R² = 0.957). So the path comparison is near-invariant to the target; the
   placement comparison is not invariant at all. The fix therefore neither rescues
   the method nor overturns the old boolean. It shows the old boolean was never
   sufficient on its own.

Scripts: `analysis/pcr_transfer.py` (rewritten), `analysis/tests/test_pcr_transfer.py`
(extended). Machine output: `analysis/readout/pcr0_placement_audit.json`.

Reproduce:

```
python3 analysis/pcr_transfer.py --placement_audit --out analysis/readout/pcr0_placement_audit.json
python3 analysis/tests/test_pcr_transfer.py
```

---

## What was wrong, and what PCR-0 is

`pcr_transfer.py` asks whether a trained criterion method beats a free post-hoc
scalar. The scalar was **PCR**: `delta_k = mean g_probe(base) − mean g_probe(k)`,
which aims the corrected model at the **frozen base**. The base sits at
`c = +0.4314` and the target is `c* = +0.0878`, so the competitor was being aimed
**0.3436 away from where it should be**, inheriting exactly the mis-placement the
method exists to remove. Logged as OWED in `FULLSTUDY_PREREG.md` (2026-09-11).

**PCR-0** is the same estimator with a different target constant:

```
PCR     delta_k      = mean g_probe(base) − mean g_probe(k)
PCR-0   delta_k^(0)  = T*                 − mean g_probe(k)   =  delta_k + s*
```

`s*` is found **model-free**, by sweeping the decision threshold over the base's own
empirical POPE gaps until the criterion equals `c*` — no Gaussian, no equal-variance
assumption, one constant, no training, still a single scalar added to the logits at
inference. It is the strongest honest form of "you did not need a method".

**The target is the JOINT arm's empirical endpoint criterion, `c* = +0.0878`** (per
cell: `joint|o1|s17` +0.0623, `joint|o1|s23` +0.1134; n = 2, so the per-cell values
are quoted and no interval is), read out of the certified `fs_aggregate.json` and
checked against the prereg's +0.088 on every run. **Not a modelled c = 0**: balanced
accuracy peaks at c = 0 only under equal variance and our fitted z-ROC slopes are
0.55–0.80, so a modelled zero is an assumption this project has already falsified on
this data. The script **refuses to run** if it cannot certify the target — it will not
silently fall back to zero.

Two extra guards worth naming:

- **The base dump is certified against the readout before any scalar is fitted from
  it.** If its criterion disagrees with `fs_aggregate`'s recorded base c by more than
  0.010 the run aborts on a real results tree (a scalar fitted on a dump that is not
  the dump the paper's numbers came from is a number about nothing). On a synthetic
  fixture tree it degrades to a loud note.
- **A correction is applied to stage 0 too.** A post-hoc scalar is applied at
  inference to every checkpoint, the base included, so each variant's trajectory
  starts from its *own* corrected base. For PCR the offset is 0 and nothing changes;
  for PCR-0 the start point is on the target. Referencing PCR-0 to the un-re-aimed
  base would charge it a first transition of `|c* − c_base|` for doing the one thing
  it exists to do — the mirror image of the original rigging. I had this wrong in the
  first draft of the script and fixed it; it moved the fixture's PCR-0 path from
  1.262 to 0.246.

Both competitors are reported side by side. The pre-registered boolean
(`arm_is_method`, path vs PCR) is unchanged in definition and in JSON shape.

---

## Comparison 1 — endpoint placement, |c_endpoint − c*|. REAL DATA, n = 20 cells.

Per cell first, then the mean over cells. Intervals bootstrap **cells as a list with
multiplicity**; n < 4 gets the per-cell values instead of an interval.

| who | n cells | mean end c | mean \|c − c*\| | 95% CI (cell bootstrap) | mean end F1 |
|---|---|---|---|---|---|
| **PCR-0** (target-aimed scalar) | — | +0.0878 | **0.0000** | by construction | not measurable here |
| **PCR** (base-aimed scalar) | — | +0.4314 | **0.3436** | by construction | not measurable here |
| joint | 2 | +0.0878 | 0.0255 | per cell: 0.0255, 0.0256 | 0.8667 |
| seq (control) | 9 | +0.0903 | 0.0811 | [0.0523, 0.1142] | 0.8625 |
| **anchor** (the trained criterion method) | 9 | **+0.6972** | **0.6094** | [0.5565, 0.6605] | 0.7963 |

Per cell, the anchor:

| cell | end c | \|c − c*\| | beats PCR (0.3436)? | beats PCR-0 (0.0)? |
|---|---|---|---|---|
| anchor\|o1\|s17 | +0.7626 | 0.6747 | no | no |
| anchor\|o1\|s23 | +0.6701 | 0.5823 | no | no |
| anchor\|o1\|s31 | +0.7405 | 0.6527 | no | no |
| anchor\|o2\|s17 | +0.8284 | 0.7406 | no | no |
| anchor\|o2\|s23 | +0.7277 | 0.6399 | no | no |
| anchor\|o2\|s31 | +0.7179 | 0.6300 | no | no |
| anchor\|o3\|s17 | +0.5414 | 0.4536 | no | no |
| anchor\|o3\|s23 | +0.6380 | 0.5502 | no | no |
| anchor\|o3\|s31 | +0.6485 | 0.5606 | no | no |

**0/9 against either competitor.** The worst anchor cell is 0.7406 from the target;
the best is 0.4536, still worse than the *handicapped* competitor's 0.3436. Plain
sequential beats the base-aimed competitor in 9/9 (all its errors are below 0.3436)
and loses to the target-aimed one in 9/9 (all are above 0).

**The caveat that keeps this from being the whole verdict, stated plainly.** `0.0000`
for PCR-0 is exact for the *base* checkpoint, where `s*` is fitted directly. For a
trained checkpoint k it is `delta_k + s*`, which lands on `c*` only to the extent the
label-free probe transports — and the **transport residual cannot be measured without
the probe dumps**. So the honest reading of the PCR-0 row is: *the target-aimed
scalar starts on the target and drifts off it by an unmeasured residual*, while the
anchor starts 0.61 away. For the anchor to win this axis the residual would have to
exceed 0.61 criterion units — larger than the entire base mis-placement it is
correcting, and larger than any residual in the fixtures. Possible, not plausible,
and **untested**.

---

## Comparison 2 — criterion path, Σ|Δc| on the rephrased template. NO DATA.

This is the pre-registered comparison and **it cannot be run on this machine.** The
per-item dumps (`pope_logits.jsonl`, `pope_rephr_logits.jsonl`,
`probe_*_logits.jsonl`) exist only on the cluster. Attempted against
`fullstudy/results_fs`:

```
[pcr_transfer] base certification: ABSENT (dump c=ABSENT, aggregate c=+0.4314)
    s*[orig] ABSENT (base dump pope_logits.jsonl absent -> no s* for orig)
[pcr_transfer] NOTE PCR0_ABSENT no original-template base dump to fit s* on
[pcr_transfer] VERDICT ... ABSENT -- rephrased-template comparison with probe 'coco'
               unavailable: rephr: base/stage dumps absent for arm
```

No number is emitted, and the verdict line says in full sentences that the fair
comparison was **not made** and no method claim may be quoted from that run. When the
dumps land:

```
python3 analysis/pcr_transfer.py --results <results_fs> --runtag <METHOD_RT> \
        --base llava15_base --ref_runtag <SEQ_RT> --out analysis/readout/pcr_<cell>.json
```

which now emits `arm_is_method` (vs PCR, pre-registered, unchanged),
`arm_lower_vs_pcr0`, `arm_endpoint_closer_to_target_than_pcr0`, and
`arm_survives_pcr0` = both, with a paired item bootstrap (default B = 1000,
resampling POPE ids as a **list with multiplicity**) on both margins and on the
endpoint placement gap.

**What is separately reported for the drift axes**, because the two are different
estimators and the last time they were mixed in this project it produced a trade-off
hypothesis that `TRADEOFF_PLACEMENT.md` then had to refute:

| arm | n | raw Σ\|Δc\| mean | raw range | post-settling mean | post-settling range |
|---|---|---|---|---|---|
| seq | 9 | 0.9364 | [0.6020, 1.2021] | 0.7814 | [0.4295, 1.1568] |
| anchor | 9 | 0.7224 | [0.5209, 1.0154] | 0.3641 | [0.1878, 0.5519] |
| joint | 2 | 0.6074 | [0.6002, 0.6145] | 0.2278 | [0.2157, 0.2400] |

(These reproduce `tradeoff_placement.py`'s premise table exactly, from an independent
code path — a positive control that the per-cell chain arithmetic agrees between the
two scripts.) The anchor does genuinely shorten the post-settling path. It has no
competitor on that axis yet.

---

## Why the rigging mattered less than expected for the boolean, and more for the paper

Re-aiming the scalar adds a **constant** to every stage. If a cell's stages lie on one
z-ROC — fixed evidence distributions, sliding threshold, which is this project's
primary claim and holds for the sequential reference at mean R² = 0.957, min 0.844 —
then c is linear in an added offset with a slope set by the shared evidence scales, so
a constant re-aim moves every stage's criterion by the same amount and **every |Δc| is
unchanged**. Measured at POPE's real n (4500/4500), with imperfect 85% transport, over
the fitted slope range, by `check_reaim_path_invariance()` in the test suite:

| z-ROC slope | path (base-aimed) | path (target-aimed) | diff |
|---|---|---|---|
| 1.00 | 0.8752 | 0.8412 | −0.0341 (−3.9%) |
| 0.70 | 1.0388 | 1.0397 | +0.0008 (+0.1%) |
| 0.55 | 1.2054 | 1.1947 | −0.0107 (−0.9%) |

Sub-4%, no systematic sign — finite-sample discreteness in the empirical z-transform,
not a real effect. This is now an assertion in the test suite, so if it ever stops
holding the reading below becomes wrong and the path comparison has to be re-run under
both targets before being quoted.

**So the consequence of fitting the competitor against the base was not that the
competitor was handicapped on the metric being judged.** It is that the metric being
judged — path only — was the one metric on which the two competitors are nearly
identical, and the axis where a free scalar is unbeatable was not in the test at all.
A reviewer asking "why not just add a scalar?" is asking about *placement*, and the
old test had no answer to give.

---

## Verdict: does the method survive a properly-aimed competitor?

**On placement, no, and not by a small margin: 0 of 9 cells, against both
competitors, with the trained arm also losing 9/9 to doing nothing.** If the paper
quotes the anchor family as a criterion-correction method, a reviewer with one scalar
and no GPU beats it on the axis that motivated the method.

**On the pre-registered path axis, undecided — NO-DATA, not a negative.** It needs the
dumps. And the actual method candidates (`critp`, `critbal`, `CNP`) have **no scored
cells at all**; everything above is the anchor family, which is the nearest thing with
data. Nothing here forecloses `fsL_critbal_o1_s17`, which corrects the *target* rather
than damping the *path* and is the one queued arm that could land near `c*` with a
short path. If it does, the method section has a positive result and this document's
placement column is the one it should be argued on.

I am not leaning on the structural-trade-off framing: it is refuted
(`TRADEOFF_PLACEMENT.md`), and nothing in this analysis revives it. The joint arm
still has both the shortest post-settling path and the best placement.

### What this means for the paper's method section

1. **The method-vs-post-hoc claim may not be quoted from the path axis alone.** Both
   competitors and both axes go in the same table — the prereg already binds drift
   wins to their placement cost, and this is the same rule applied to the competitor.
2. **State the competitor as PCR-0, not PCR.** Quoting a win over a scalar aimed at
   `c = +0.431` invites exactly the reviewer objection the test exists to pre-empt.
   The margin against PCR-0 is what should appear; against PCR it is context.
3. **The honest current claim is negative and should stay negative.** "A trained
   criterion regularizer aimed at the frozen base lands 0.61 from the empirical
   optimum in 9/9 cells, worse than an untouched sequential model and worse than a
   single free scalar" is a clean, defensible, useful negative result. It is stronger
   writing than a path-only win would have been, and it costs nothing that was
   actually established.
4. **Do not describe PCR-0's placement as free of assumptions.** It is exact on the
   base and approximate on a trained checkpoint by an unmeasured transport residual.
   Say so in the same sentence that quotes 0.0000.
5. **The target is `c* = +0.088`, empirical, n = 2 JOINT cells, never a modelled 0**,
   and the paper should carry that n. Two cells is thin for a yardstick; it is still
   better than a number equal variance would have to justify.

### What would falsify this write-up

1. **The dumps land and the trained arm's rephrased-template path beats PCR-0's by
   more than the placement gap costs it.** That is a real possible outcome and the
   script now measures it in one run.
2. **A transport residual above 0.61 criterion units.** That would make the
   target-aimed scalar worse than the anchor in practice despite starting on target.
   Measurable the moment the probe dumps exist; currently unmeasured, and the
   placement row above should be read with that gap visible.
3. **`check_reaim_path_invariance()` failing at POPE's n.** Then the re-aim does move
   the path axis, the pre-registered boolean *was* distorted by the rigging, and the
   old numbers have to be recomputed rather than reinterpreted.
4. **`critbal` landing near `c*` with a short path.** Then a trained target-correction
   does beat the scalar on both axes and the method section is positive — which is the
   outcome this document would most like to be wrong in favour of.
