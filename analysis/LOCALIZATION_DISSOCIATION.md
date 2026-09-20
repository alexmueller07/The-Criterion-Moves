# Drift localization: is the path/endpoint dissociation real?

Date: 2026-09-13. Zero GPU. Scope: audit only — nothing under `paper/` touched, nothing appended
to `fullstudy/FULLSTUDY_PREREG.md`. The pre-registration entry in section 6 is a **proposal**.

**Verdict up front.** The dissociation is **not real yet**, and the reason is not only that the arms
are single cells. Under the project's own adopted reference point (`c* = +0.0878`, not `c = 0`), the
projector-frozen arm does not have the best placement of any arm measured — it has an **ordinary**
one: its placement error 0.0837 sits almost exactly on the sequential arm's own mean per-cell
placement error 0.0811 (95% CI [0.0518, 0.1122], `analysis/readout/does_drift_cost.json`). Both
freeze arms land inside the sequential arm's nine-cell distribution on the drift-path axis, and
`locproj` lands inside it on the placement axis too. The comparison is also **not
capacity-matched**: the two arms whose endpoints supposedly dissociate differ by 59.0M trainable
parameters as well as by which component is frozen. And the per-stage vectors needed to
characterise the path do not exist anywhere — they were computed and thrown away.

There is one thing here worth a follow-up paper's worth of work, and it is cheap to find out.
Section 7 ranks the runs.

---

## 1. What I could and could not verify

### 1.1 Verified, locally, from raw generations

`analysis/mechanism_readout.py --mode loc --results fullstudy/results_fs` runs on this machine and
reproduces the baseline row **exactly**, by rescoring `pope_gen.jsonl` with the audited scorer
(`fs_common.score_pope`) — an independent code path from the aggregate JSON:

```
untuned base:  c = +0.4314   d' = 2.3477   F1 = 0.8449

arm                    what adapts               sum|dc|  endpoint c     end d'
fsL_seq_o1_s17         baseline (all adapt)       0.7643      0.0679     2.1878  (+0% vs baseline)
fsL_locproj_o1_s17     projector FROZEN         (not finished)
fsL_locearly_o1_s17    layers 0-15 only         (not finished)
fsL_loclate_o1_s17     layers 16-31 only        (not finished)
```

Independently, `fs_common.score_pope('.../fsL_seq_o1_s17_k6/pope_gen.jsonl')` gives
`c = 0.0679, dprime = 2.1878, H = 0.8476, FA = 0.1227, n_total = 9000, parse_fail = 0`, matching
`analysis/readout/fs_aggregate.json` to four decimals. **Baseline row: confirmed.**

### 1.2 NOT verifiable here — the two freeze arms

`(not finished)` is not a partial-eval warning. **There is no `locproj`, `locearly` or `loclate`
data in this repository at all.** Established by exhaustive search:

| where I looked | what is there |
|---|---|
| `fullstudy/results_fs/` | only `fsL_seq_o1_s17_k1..k6` and `llava15_base` |
| `analysis/readout/fs_aggregate.json` | stamped `2026-09-09T07:32:40`; arms = 9 seq + 9 anchor + 2 joint. **No loc arms** — it predates them |
| `analysis/readout/` (every JSON) | no loc arm appears except the retracted text-recorded one (below) |
| `results/` | pilot suite only |
| grep for `0.9895`, `0.2953`, `2.2298`, `0.7237` | only inside synthetic test fixtures; coincidental digit matches |
| `ssh vgi-server` | connection to 163.152.163.85:59344 times out from this machine |

So the three numbers in the brief's table for `locproj` and `loclate` are **text-recorded only** —
they exist in the 2026-09-12 correction entry of `FULLSTUDY_PREREG.md` and nowhere else. Everything
below that uses them is conditional on that transcription being right. I could not falsify them and
I could not confirm them.

Two provenance gaps follow, both worth fixing before the numbers are used again:

1. **`EVAL_DONE = 6/6` is asserted by a human, not recorded by an artifact.** `mechanism_readout.py`
   still prints no stage count in the `loc` table (only the optional `--out` JSON carries
   `n_stages`), and no `--out` JSON was written. The correction entry's own reusable lesson —
   "any readout of a running arm must report its stage count beside its numbers" — is **not yet
   implemented in the code**, so the same failure can recur unchanged.
2. **`analysis/readout/tradeoff_placement.json` still carries the retracted number.** Its cell
   `fsL_locproj_o1_s17` is recorded as `raw_path 0.2237, endpoint_c 0.2739, n_stages 6,
   source "TEXT-RECORDED"` — the partial-run values, stamped with a stage count of 6 that was never
   true. Anything that reloads that file inherits the withdrawn reading.

### 1.3 The per-stage vectors do not exist anywhere

`mechanism_readout.py` computes each arm's per-stage criterion vector inside its loop and then
**discards it**, keeping only the aggregate:

```python
report["rows"].append({"arm": tag, "what": what, "sum_abs_dc": round(path, 4),
                       "endpoint_c": end["c"], "endpoint_dprime": end["dprime"],
                       "n_stages": len(ks)})
```

`NEXT_EXPERIMENTS.md` already flags this ("`locproj`'s per-stage criteria were never recorded …
only its raw path exists"). The important consequence for planning is that this is **not gated on
GPU**: the `pope_gen.jsonl` files still exist on vgi2, so the whole per-stage vector for all three
arms is recoverable by re-running the readout on a CPU debug job. See section 7, Tier 0.

---

## 2. The numbers, re-scored against the reference this project actually adopted

On 2026-09-11 the project retracted `c = 0` as the placement reference and replaced it with
`c* = +0.0878`, the JOINT arm's empirical endpoint (`FULLSTUDY_PREREG.md` line 1684;
`TRADEOFF_PLACEMENT.md` line 19; `PCR0_VERDICT.md` "never a modelled c = 0";
`does_drift_cost.json` `target_provenance`). The 2026-09-12 correction entry ranks placement by
closeness to zero — "the best placement of any arm measured: c = +0.0041, against sequential's
+0.0679 and JOINT's +0.088". That is the retracted reference, used one day after it was retracted.

Re-scored against `c*`:

| arm | trainable params | Σ\|Δc\| (path) | endpoint c | **\|c − c*\|** | endpoint d′ |
|---|---|---|---|---|---|
| `fsL_seq_o1_s17` (verified) | 180,887,552 | 0.7643 | +0.0679 | **0.0199** | 2.1878 |
| `fsL_locproj_o1_s17` (text) | 159,907,840 | 0.9895 | +0.0041 | **0.0837** | 2.2298 |
| `fsL_loclate_o1_s17` (text) | 100,933,632 † | 0.7237 | +0.2953 | **0.2075** | 2.2345 |

† assuming `loclate` left the projector trainable — see section 5.2; the alternative is 79,953,920.

**The ordering reverses.** Under `c*` the *baseline* has the best placement; `locproj` is 4.2×
worse and `loclate` 10.4× worse. The crossover is at `c* = 0.0360`: for any reference above that,
`locproj` is worse than doing nothing. The adopted reference is 0.0878 and JOINT's two cells
bracket it at [+0.0623, +0.1134], so the conclusion is robust across the whole empirical range of
`c*` — it flips only if you go back to a modelled zero.

**So `locproj` shows no dissociation at all.** It is worse on the path axis *and* worse on the
placement axis. Only `loclate` has the trade-off shape (slightly shorter path, much worse
placement), and that shape is the anchor family's already-documented signature, not a new
phenomenon.

### d′ does hold up

Base d′ = 2.3477. The nine sequential endpoint d′ values span [2.1878, 2.2732] (sd 0.0458). Both
freeze arms (2.2298, 2.2345) sit inside that band, and |d′ − base| is 0.118 and 0.113, far under
the pre-registered F1 falsifier of 0.3. **"What moves is placement, not discriminability" survives
unchanged** — it is the one part of the reading that does.

---

## 3. Is the difference larger than cell-to-cell noise? No.

This is the part that decides whether the observation is worth a paper. It is answerable *now*,
with zero GPU, because the sequential arm has nine cells and three of them are the exact matched
stratum (order o1, seeds 17/23/31).

### 3.1 The baseline's own spread

Order o1, arm `seq`, three seeds — computed from `fs_aggregate.json`:

| statistic | s17 | s23 | s31 | mean | sd | range |
|---|---|---|---|---|---|---|
| raw Σ\|Δc\| | 0.7643 | 0.7376 | 1.2021 | 0.9013 | 0.2608 | [0.7376, 1.2021] |
| endpoint c | +0.0679 | +0.1030 | +0.2607 | +0.1439 | 0.1027 | [+0.0679, +0.2607] |
| \|c − c*\| | 0.0199 | 0.0151 | 0.1729 | 0.0693 | 0.0897 | [0.0151, 0.1729] |
| endpoint d′ | 2.1878 | 2.2732 | 2.2592 | 2.2401 | 0.0458 | [2.1878, 2.2732] |

Across all nine sequential cells (three orders × three seeds): path [0.6020, 1.2021] mean 0.9364;
endpoint c [−0.0143, +0.2607]; |c − c*| mean **0.0811**, bootstrap 95% CI **[0.0518, 0.1122]**,
max 0.1729.

### 3.2 Where the freeze arms fall

| quantity | `locproj` | vs baseline distribution | `loclate` | vs baseline distribution |
|---|---|---|---|---|
| path 0.9895 / 0.7237 | 0.9895 | **inside** [0.7376, 1.2021]; 5th of 10 if inserted into the nine seq cells — dead centre | 0.7237 | 0.0139 below the o1 min; **inside** the nine-cell range; 2nd of 10 |
| \|c − c*\| | 0.0837 | **inside** the seq mean's 95% CI [0.0518, 0.1122]; 6th of 10; just above the seq median 0.0741 | 0.2075 | 0.0346 **above** the seq max 0.1729 — the only statistic that leaves the envelope, by 0.35 sd |
| endpoint d′ | 2.2298 | inside [2.1878, 2.2732] | 2.2345 | inside |

**Every headline percentage in the correction entry is an artifact of comparing to one baseline
cell.** `fsL_seq_o1_s17` is below-median on path (3rd lowest of nine) and 2nd best of nine on
placement. Against the sequential arm's *mean* path 0.9364 instead of that one cell:

- `locproj`: **+5.7%**, not +29%.
- `loclate`: **−22.7%**, not −5%.

The baseline's own three-seed spread is −18% / +33% around its mean. A 29% path difference is
smaller than the difference between two seeds of the same arm.

### 3.3 The two noise floors are different, and the smaller one is irrelevant

Binomial SE of `c` at 4,500 yes / 4,500 no is **0.0185** at base, ~0.017 mean over stage cells
(clustering by image ignored, so this is a floor on a floor). Seed-to-seed sd of the endpoint
criterion is **0.1027** — **6× larger**. Eval noise is not the binding constraint; trajectory
variance is. Scored against measurement noise the `locproj`–baseline endpoint gap of 0.0638 looks
like 2.6σ; scored against the floor that actually applies it is **0.62 seed-sd**, and `loclate`'s
0.2274 gap is 2.2 seed-sd against a single comparison cell but 1.47 against the seed mean.

With n = 1 per arm none of these is a test. The honest statement is: **the arms differ; the
differences are not larger than the baseline arm's own cell-to-cell variation, with the single
partial exception of `loclate`'s placement error, which clears the sequential maximum by about a
third of a standard deviation.**

---

## 4. Per-stage characterisation: cannot be done, and does not need GPU to fix

The brief asks whether the split is visible per stage, where the arms diverge, and whether
`locproj` wanders and returns while `loclate` goes somewhere and stays. **None of this is
answerable from anything that exists** — section 1.3. What can be derived from the two recorded
scalars plus the verified base is an identity, not evidence:

| arm | path | endpoint | net displacement | wander = path/\|net\| | excess = path − \|net\| |
|---|---|---|---|---|---|
| `seq` s17 | 0.7643 | +0.0679 | −0.3635 | 2.10 | 0.4008 |
| `locproj` | 0.9895 | +0.0041 | −0.4273 | 2.32 | 0.5622 |
| `loclate` | 0.7237 | +0.2953 | −0.1361 | **5.32** | 0.5876 |

Read literally this **inverts the brief's framing**: it is `loclate`, not `locproj`, that moves the
most non-productively — the largest excess path and much the largest wander ratio, i.e. it does not
"go somewhere and stay", it churns and ends up barely displaced. `locproj` is the one that travels
furthest net.

But this separates nothing. Sequential's own wander ratios span [1.44, 7.04] across nine cells
(2.10 / 2.25 / 7.04 at o1 alone), and today's entry already reports "the path is 2.5× the net
displacement" as the sequential norm. Both freeze arms are inside. Two scalars cannot distinguish
"wandered and returned" from "went straight and stopped short" — only the stage vector can.

**Recovering it costs no GPU.** Tier 0 in section 7.

---

## 5. The capacity confound. The comparison is not matched, and the unmatched pair is the headline one.

### 5.1 What each arm actually trains

The freeze knobs are `CLH_LORA_BAND` and `CLH_LORA_SAVE`, handled by `_lora_overrides()` in
`pilot/model_zoo.py`. I ran the real function against a LLaVA-1.5-7B module-name universe and
counted PEFT-matched modules per setting, then costed them at LoRA r = 64 (α = 2r):

| arm | env | matched modules | layers | LoRA params | projector | **trainable** | vs `seq` |
|---|---|---|---|---|---|---|---|
| `seq` | (none) | 224 | 0–31 (32) | 159,907,840 | 20,979,712 | **180,887,552** | 100.0% |
| `locproj` | `CLH_LORA_SAVE=none` | 224 | 0–31 (32) | 159,907,840 | 0 | **159,907,840** | **88.4%** |
| `locearly` | `CLH_LORA_BAND=early` | 112 | 0–15 (16) | 79,953,920 | 20,979,712 | **100,933,632** | **55.8%** |
| `loclate` | `CLH_LORA_BAND=late` | 112 | 16–31 (16) | 79,953,920 | 20,979,712 | **100,933,632** | **55.8%** |
| `loclate` if also `SAVE=none` | both | 112 | 16–31 (16) | 79,953,920 | 0 | **79,953,920** | **44.2%** |

The band regexes were executed, not eyeballed: `early` matches layers 0–15 and `late` matches
16–31, 112 modules each, under both plausible `language_model` module-name layouts. Projector
params use the llava-hf `multi_modal_projector` (`linear_1` 1024→4096 + `linear_2` 4096→4096, with
biases) = 20,979,712. Architecture constants (32 layers, hidden 4096, intermediate 11008, MHA so
k/v are square, CLIP-L/14-336 vision hidden 1024) are the published LLaVA-1.5-7B / Llama-2-7B
config values, taken from architecture knowledge — **no local copy of the model config exists on
this machine to read them from**. Confirmed by exhaustive search: the HF cache holds only
`Qwen2.5-0.5B/1.5B-Instruct`, `Qwen3-4B`, `gpt2` and `clip-vit-base-patch32` (the base/patch32
model, hidden 768 — not the ViT-L/14-336 that LLaVA-1.5 uses), and no `llava` `config.json` exists
anywhere under `~`. The conclusions below are insensitive to modest errors in these constants — the
capacity gaps at issue are 12% and 37%, not 1% — but the counts are **derived, not read**, and the
Tier 0 job in section 7 should settle them from the two authoritative sources at once: the cluster's
own `$ROOT/hf_home` copy of the config, and the `[train] trainable params` line each arm already
printed to its slurm log. If those two disagree with the table above, the table is wrong.

### 5.2 Nothing in the repository records this

`grep` finds **no trainable-parameter count anywhere in the repo** for any arm. The count is
printed once per stage to the cluster training log (`[train] trainable params: …M / …B`,
`pilot/train_lora.py:373`) and the logs are not in the repo. Worse, the loc arms' queue lines are
not in `fullstudy/queue.txt` either, so **it is not recorded locally whether `loclate` also set
`CLH_LORA_SAVE=none`**. Its trainable count is either 100.9M or 80.0M and the repository cannot say
which. Both are ways of saying the same thing: the capacity match was never checked because the
numbers to check it against were never written down.

There is also a guard gap. `train_lora.py:374` asserts `1e7 < trainable < 1e9`. A band regex that
silently matched nothing would leave only the projector trainable at 20.98M — **inside the assert
window**, so the guard would pass. (PEFT itself would likely raise on zero matches, so this is a
latent hole rather than a live bug, but the assert is not the thing protecting you.)

### 5.3 The verdict on matching

- **`locearly` vs `loclate` is exactly matched** — 100,933,632 trainable parameters each, identical
  module count, identical rank. The docstring's claim that "both bands adapt the same number of
  modules, so the contrast isolates location, not capacity" is **correct and verified**. This is the
  design's one clean contrast, and it is **the leg that never ran**.
- **`locproj` vs `seq` is not matched** — 11.6% fewer trainable parameters, and the missing 21.0M
  are *full-rank* parameters sitting directly on the vision-language interface, not rank-64 factors
  spread over 32 layers.
- **`locproj` vs `loclate` is not matched by a wide margin** — 159.9M vs 100.9M, a 59.0M / 37%
  difference. **These are the two arms the entire dissociation is drawn between.** They differ in
  which component is frozen *and* in how much capacity they have *and* in whether adaptation spans
  all layers. There is no way to attribute the endpoint difference to the projector.
- **`loclate` vs `seq` is not matched** — 44% fewer trainable parameters.

Training steps, data, order, seed and hyperparameters are identical across arms, so "trained less"
here means strictly "had fewer parameters to train with", not fewer updates. That is a narrower
confound than it could be — but it is exactly the one that "the projector carries placement" is
competing against, and on the present evidence it wins on parsimony: the arm with the most
trainable parameters (`seq`, 180.9M) has the best placement, the middle one (`locproj`, 159.9M) is
second, and the smallest (`loclate`, 100.9M) is worst. **Placement error is currently monotone in
trainable parameter count across all three arms.** One cell each, so this is a pattern in three
points, not a finding — but it is a simpler story than the localization one and it must be excluded
before the localization one can be told.

---

## 6. Proposed pre-registration entry

Written before any further arm runs, in the style of the existing `FULLSTUDY_PREREG.md` entries.
**Not appended — this is a draft for Alex to paste, edit or reject.** If any of the runs below have
already started when it is pasted, it is not a pre-registration and must be re-dated and labelled.

> ## 2026-09-13 — PRE-REGISTERED: does the drift localize, or is it just capacity?
>
> The 2026-09-12 correction left one cell per arm on a four-leg design, and a re-audit
> (`analysis/LOCALIZATION_DISSOCIATION.md`) found three defects that must be fixed before any
> localization sentence is written: the placement comparison used the retracted `c = 0` reference;
> the percentages were computed against a single below-median baseline cell; and the two arms the
> dissociation is drawn between differ by 59.0M trainable parameters. This entry pre-registers what
> would settle it.
>
> **Arms (LLaVA-1.5-7B, UCIT, order o1, seeds 17/23/31, all else identical to `seq`).**
> `locproj` (`CLH_LORA_SAVE=none`, 159,907,840 params); `locearly` / `loclate`
> (`CLH_LORA_BAND=early|late`, 100,933,632 each); capacity controls `locr56` (r = 56, all 32 layers,
> projector trainable, 160,899,072 = `locproj` + 0.62%) and `locr32` (r = 32, all 32 layers,
> projector trainable, 100,933,632 = `locearly` = `loclate` **exactly**).
>
> **Preconditions, gating every arm before it may be quoted.** `EVAL_DONE` present in all six cells
> AND `pope.n_total == 9000` in all six (the marker file is a proxy; the row count is the
> condition). The per-stage criterion vector is dumped to JSON. The arm's `[train] trainable params`
> line is copied out of the slurm log into the entry. Any arm failing a precondition is reported as
> not-run, never as a number.
>
> **Analysis rules, fixed here.** Placement error is `|c_endpoint − c*|` with `c* = +0.0878`, never
> a modelled 0. Comparison is against the **sequential o1 three-seed envelope** (path
> [0.7376, 1.2021]; `|c − c*|` [0.0151, 0.1729]; nine-cell mean 0.0811, 95% CI [0.0518, 0.1122]),
> never against a single seq cell. Per-seed values and min–max ranges; sign-consistency is the
> currency; no pooled cross-seed CIs. Endpoint is k6, quoted regardless of which stage looks best.
>
> - **L1 — the projector carries endpoint placement (primary).** Prediction: `locproj`'s placement
>   error is below its matched `seq` cell's in **3/3 seeds**, AND its three-seed mean is below
>   0.0518 (the lower bound of the sequential mean's bootstrap CI).
>   **Fail:** ≤2/3 sign consistency, or a three-seed mean inside [0.0518, 0.1122] — i.e. the arm is
>   an ordinary sequential cell. *The one cell in hand already reads as a fail on the second clause
>   (0.0837 sits inside the CI); L1 is pre-registered as the test that would have to overturn it.*
> - **L2 — the path/endpoint dissociation.** Prediction: `locproj` has a **higher** raw Σ|Δc| than
>   its matched `seq` cell in 3/3 seeds **while** having a **lower** placement error in 3/3.
>   **Falsifier:** both statistics move in the same direction in ≥2/3 seeds. Then there is no
>   dissociation, only a uniformly worse or uniformly better arm, and the framing is withdrawn.
> - **L3 — location within the decoder (the only capacity-matched contrast).** `locearly` vs
>   `loclate`, identical parameter counts. Prediction: placement error(`loclate`) >
>   placement error(`locearly`) in 3/3 seeds and the three-seed ranges do not overlap.
>   **Falsifier:** <3/3, or overlapping ranges. Then location within the LM does not carry placement
>   and the series reports a negative.
> - **L4 — capacity control (gates L1–L3 regardless of their outcome).** `locr56` must **not**
>   reproduce `locproj`: required |mean placement error(`locr56`) − mean(`locproj`)| > 0.05.
>   `locr32` must **not** reproduce `loclate`: same threshold.
>   **Falsifier:** either control lands within 0.03 of the arm it controls for. Then the
>   corresponding effect is trainable-parameter count, not component identity, and L1/L2 (or L3) are
>   withdrawn **even if they passed**. Pre-committed: this is a veto, not a tie-breaker.
> - **L5 — placement, not discriminability (inherited falsifier F1).** Any loc checkpoint with
>   |d′ − base| > 0.3 damages the "placement, not discriminability" reading and must be reported.
>
> **What kills the whole line.** If, with 3 seeds per arm, every arm's placement error and raw path
> fall inside the sequential o1 three-seed envelope, the localization series is a **negative
> result**: the criterion drift is distributed, no single component carries it, and freezing any one
> component is indistinguishable from rerunning the baseline with a different seed. That is a
> reportable outcome and will be written as one, not reinterpreted.
>
> **Stated limitation of the capacity controls, in advance.** There is no perfect capacity control.
> Matching parameter count by lowering LoRA rank also lowers the dimension of the adapted subspace;
> matching it by dropping layers also changes location. `locr56`/`locr32` answer only "is a deficit
> of this many trainable parameters, on its own, enough to produce this endpoint?" They do not
> establish that rank and component are otherwise interchangeable, and no wording will imply they do.

---

## 7. Ranked runs, with cost

**Cost basis.** One arm = 6 training stages + one 6-checkpoint eval job on a single RTX 4090
(`sb_fs_arm.sbatch`: `--gres=gpu:rtx_4090:1`, `-c 8`, `--mem 28G`). Eval is **measured** at 2.0 h
from the local cell timestamps (`fsL_seq_o1_s17_k1` first write 15:42:53 → `_k6` last write
17:44:12). Training wall time is **not recorded anywhere in this repo**; the SLURM budget used for
LLaVA `seq` on UCIT is 12:00:00 (`launch_matrix.sh`). So **≤ 14 GPU-h per arm**, and the training
half of that is a budget, not a measurement.

| # | run | GPU cost | what it rules out / establishes |
|---|---|---|---|
| **0** | **Zero-GPU recovery.** Re-run `mechanism_readout.py --mode loc --out …` on vgi2 (CPU debug job, minutes) after patching it to (a) dump the per-stage `c` vector, (b) refuse any cell lacking `EVAL_DONE` or with `pope.n_total != 9000`, (c) print `n_stages` in the table. Then `grep '\[train\] trainable params' ~/cl-halluc/logs/clh_fsL_loc*` and copy the counts and the queue lines into the prereg; in the same job `cat` the LLaVA `config.json` under `$ROOT/hf_home` to replace section 5.1's derived parameter counts with read ones. | **0** | Everything in section 4, the whole per-stage characterisation, and the section 5 capacity question for `loclate` specifically. **Blocks all of the below and costs nothing.** Also stops the partial-run failure from recurring. |
| **1** | `locr56_o1_s17` — capacity control for `locproj` (160.9M vs 159.9M) | ~14 h | Whether "the projector carries placement" is just "21M fewer trainable parameters". If `locr56` reproduces `locproj`'s endpoint, L1 and L2 are dead and runs 3–5 are not worth buying. **Highest information per GPU-hour: one arm can kill the line.** |
| **2** | `locproj_o1_s23` + `loclate_o1_s23` | ~28 h | Whether either arm's numbers are stable at all. n = 2 gives a range, not an interval — it cannot establish that the arms differ, only whether the single cells were flukes. Cheapest way to find out that there is nothing here. |
| **3** | `locearly_o1_s17` — the missing leg | ~14 h | The design's only exactly capacity-matched contrast (100,933,632 vs 100,933,632). Establishes whether *location within the decoder* moves placement at all. |
| **4** | `locr32_o1_s17` — capacity control for the band arms | ~14 h | Whether the band arms differ from `seq` because of *where* they adapt or because they adapt with half the parameters. Exact match (100,933,632), so no residual to argue about. |
| **5** | `locproj`, `loclate`, `locearly` at s31; `locr56`, `locr32` at s23/s31 | ~70 h | Takes every arm to 3 seeds, matching the baseline's stratum. **This is the first point at which "the difference exceeds cell-to-cell noise" becomes sayable**, and only via sign-consistency across matched seeds — not via an interval. |
| — | **complete design** (5 arms × 3 seeds, minus the one cell in hand) | **~182 h** | 13 new arms. At n = 3 per arm with the baseline's endpoint sd of 0.1027, an effect below ~0.10 in placement error will still not separate. That is a real limit of the design, not of the budget, and is worth knowing before spending the 182 hours. |

**Ops notes, from this repo's own recorded traps.** Pin `--gres=gpu:rtx_4090:1` (vgi2's Blackwell
card cannot run the pinned torch). Pass an explicit `--time` (a wrapper without one takes the 14-day
max and gets zero backfill). Adding `locr32`/`locr56` needs a `CLH_LORA_R` knob — `run_arm.py` never
passes `--lora_r`, so it is stuck at the default 64, and `CLH_LORA_TARGETS` is unusable from
`queue.txt` because that file is pipe-delimited and every target regex contains `|`. The knob is
safe to add while arms are queued **only because unset env leaves the config byte-identical**,
exactly like the existing overrides; verify that property before adding it, since a queued job runs
whatever is on disk when it *starts*.

---

## 8. Bottom line

**Is this the seed of a follow-up paper?** The *design* might be — a four-leg freeze series with two
exactly capacity-matched arms is a clean instrument for asking where a decision criterion's drift
lives. **The result is not.** What exists is
three single cells, of which one is verified and two are transcriptions; scored against the
project's own adopted reference the headline reverses; every statistic except one sits inside the
baseline arm's own nine-cell distribution; the per-stage evidence does not exist; and the two arms
the dissociation is drawn between differ by 37% of their trainable parameters.

The cheapest path to knowing is Tier 0 (free) followed by one 14-GPU-hour capacity control that can
kill the line outright. **Do those two before buying seeds.**

Until then: one cell is one cell, and none of these arms may carry an interval, a percentage against
a single baseline cell, or the word "dissociation".

**Novelty is NOT cleared and was not checked here.** Parameter-group freezing as a localization
instrument already appears in this project's own `related_work.md`: Kaplan/Gekhman 2604.15574
("freezing parameter groups = suppressing factual plasticity", text-only, single-stage),
Model-Dowser 2602.04509 (data-free importance freezing), and — most directly — POPEv2/Obliviate
2508.04567, which **localizes hallucination to the LM head**. Our series excludes the LM head by
construction (it is frozen in every arm including the baseline), so the nearest published
localization result is about the one component this design cannot speak to. Before any of this is
framed as a contribution it needs the full two-vocabulary sweep, with the method-shaped candidates
checked at equation level — not a reading of these three notes.
