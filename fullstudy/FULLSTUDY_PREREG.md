# Full-Study Pre-registration — committed BEFORE any full-study training job

Date: 2026-09-02. Authorized by Alex ("run the full study, queue the whole thing").
Park sign-off is on tomorrow's meeting agenda; if the design changes there, amendments will be
dated here and queued jobs cancelled rather than silently repointed. Quote-the-endpoint rule
applies: the endpoints below are what gets reported.

## Design

- **Benchmark:** UCIT (MLLM-CL/UCIT parquet; exact task list, splits, and per-task train sizes
  recorded by the dependency probe and frozen in `ucit_manifest.json` at data-prep time).
  Budget cap: min(full task train size, 8,000) per stage, seed-17 subsample; 500 held out per
  task. The pilot's 4-task suite is retained as the leaky-benchmark contrast appendix (already run).
- **Backbones:** LLaVA-1.5-7B (primary; identical config to pilot) and Qwen2.5-VL-7B-Instruct
  (generalizability; LoRA on LM projections, visual tower frozen, merger trainable, dynamic-res
  capped — exact config frozen in the harness before its smoke).
- **Orderings (fixed strata, never pooled):** O1 = UCIT canonical order; O2 = exact reverse;
  O3 = the permutation produced by random.Random(41).shuffle on the canonical list (committed in
  `ucit_manifest.json`).
- **Seeds:** 17, 23, 31 (training randomness; data composition pinned to 17 everywhere).
- **Arms per backbone — core wave:**
  SEQ 3 seeds × 3 orderings = 9; ANCHOR-v2 (λ=0.1, k=4, m=2.0, pilot values) 9;
  JOINT (shuffled union, stage-boundary-step checkpoints) 3 (one per seed; ordering-invariant);
  ER-100 (100 exemplars/prior task) 3 (O1 × seeds); single-task controls 1 per task (seed 17).
- **Ablation wave (queued behind core):** ER-500 ×3; CE-on-counterfactuals anchor ablation ×3
  (O1 × seeds); out-of-domain anchor arm — DEFERRED until a non-COCO box-annotated source is
  built; scheduled within the full study, reported as pending if unfinished at write-up.
- **Evals at every stage boundary:** POPE 9k (3 splits), CHAIR 500 (vendored scorer, fixed
  synonym list), AMBER generative (1,004; vendored judge-free scorer), per-task held-out 500,
  refusal-rate audit on every short-answer task. Greedy decoding; identical audited scorers for
  every checkpoint; all local rescoring from logged JSONL as in the pilot.

## Stage answer statistics — blinding rule

Per-task training answer statistics (yes/no/unanswerable fractions, mean target length) are
computed at data-prep time and committed to `ucit_manifest.json` BEFORE any evaluation runs.
Endpoint E1 is scored against those frozen numbers.

## Confirmatory endpoints (style guide: seed values + min–max ranges; sign-consistency is the
confirmatory currency; orderings are strata; no pooled cross-seed CIs)

- **E1 — dose-response.** For every criterion-active stage (frozen answer stats decide which),
  the sign of the post-stage pooled-POPE yes-rate change matches the stage's answer-prior
  direction in ≥5 of 6 applicable seed×backbone cells per ordering. Zero-dose stages predicted
  |Δc| < the smallest criterion-active |Δc| in the same run.
- **E2 — anchor criterion-freeze.** Post-settling Σ|Δc|(ANCHOR) < Σ|Δc|(SEQ) in every matched
  seed×ordering cell (18 cells per backbone-complete matrix); pooled suppression ratio reported
  with per-cell values.
- **E3 — length-controlled mid-sequence rise.** CHAIR_i@60 (stage 1 → stage T−1) in SEQ:
  sign across all 9 LLaVA cells reported; confirmatory if ≥7/9 positive AND the paired
  per-cell bootstrap excludes zero in ≥5/9. (This is the pilot's promoted diagnostic getting
  its pre-registered confirmatory test.)
- **E4 — memorylessness.** Each single-task control's criterion falls within the min–max band
  of the matched sequential endpoints across seeds, per backbone.
- **E5 — refusal leakage.** Conditional endpoint: applies iff UCIT contains a stage whose
  training answers are ≥10% refusal-class (frozen stats decide). If none: reported as
  not-testable-on-UCIT; the pilot result stands alone and is labeled single-suite.
- **Falsifier F1 — policy-not-perception.** If pooled d′ moves by >0.3 from base at any SEQ
  checkpoint (any seed/ordering), the "criterion drift, not grounding" headline is damaged and
  must be revised — this is a pre-committed vulnerability, not a robustness claim.

## Secondary (reported, no gate)

AMBER trajectories (all arms); forgetting matrices strict+lenient; Qwen-vs-LLaVA qualitative
comparison of every endpoint (generalizability is claimed only where sign-consistency holds on
both backbones); settling-step magnitude vs anchor margin m.

## Amendments (dated, before any full-study training result existed)

- **2026-09-03, GPU pinning.** The dependency probe found vgi2 now carries 4x RTX 4090 +
  1x RTX PRO 6000 Blackwell (sm_120); the pinned torch 2.13.0+cu126 has no sm_120 kernels, so a
  generic `--gres=gpu:1` that lands on the Blackwell crashes on the first forward. All jobs are
  pinned to `--gres=gpu:rtx_4090:1`, keeping the exact 4090 + torch the pilot ran on — this
  PRESERVES pilot-vs-full-study numeric comparability (an env rebuild was deliberately rejected as
  it would confound the two). Recorded before any arm ran.
- **2026-09-03, Qwen config asymmetry.** Qwen2.5-VL adapter freezes the entire vision tower
  including the visual merger (LoRA on LM projections only, no modules_to_save); LLaVA trains its
  multi_modal_projector. Intentional: the generalizability claim concerns the LM-side criterion
  phenomenon and whether the anchor fixes it, both unaffected by a frozen visual merger; freezing
  it is the cleaner and lower-VRAM choice. The asymmetry is disclosed wherever the two backbones
  are compared. Three Qwen chat-format conventions the probe's crashed forward could not confirm
  (eos terminator, target leading-space rule, merger freeze) are gated by the smoke's collator
  boundary assert + pre-train forward-loss check — no real Qwen run proceeds on an unvalidated guess.

- **2026-09-03, UCIT is answer-statistics-flat (blinding rule fired as designed).** The frozen
  `ucit_manifest.json` shows ALL SIX UCIT tasks have ~0% yes/no and 0% refusal training content
  (ArxivQA=MC letters, CLEVR-Math/IconQA=numbers, ImageNet-R=class names, Flickr30k/VizWiz=captions;
  UCIT's "VizWiz" is captioning, NOT the yes/no VizWiz-VQA the pilot used). Consequences for the
  endpoints, decided from the frozen stats before any eval:
  - **E1 (dose-response) and E5 (leakage) are NOT testable on UCIT** — no task supplies the
    answer-statistics variation whose *presence* drives criterion drift or the refusal terminator
    whose *presence* drives leakage. Per the prereg's own conditional-endpoint rule, these are
    reported not-testable-on-UCIT; the mechanism evidence stands on the answer-statistics-varied
    pilot suite (ScienceQA/TextVQA/Flickr/VizWiz-VQA).
  - **What UCIT DOES test, validly and importantly:** (a) the *null side* of dose-response — an
    all-zero-dose stream predicts little criterion drift in SEQ (E1's contrapositive: no dose, no
    push); (b) E2/E3/E4/F1 — whether the anchor's criterion-stabilization, the length-controlled
    hallucination axis, memorylessness, and d'-flatness generalize to a standard, harder,
    leakage-hygienic 6-task benchmark and a second backbone.
  - **Resulting paper structure (strengthened, not weakened):** mechanism on the pilot suite
    (answer-stat-varied, both backbones); generalization of the phenomenon + method on standard
    UCIT (both backbones). Two suites, each doing what it is suited for. The full-rigor pilot-suite
    runs on the second backbone (Qwen) are therefore part of the full study, not optional.
  - Flagged for Alex to ratify; proceeding with the UCIT method-generalization matrix (valid) and
    setting up the pilot-suite second-backbone mechanism runs.

## Execution honesty rules

Same as pilot: smoke gate green before any wave launches; failed jobs reported, never silently
rerun with different knobs; every amendment dated; scorer changes prohibited after first eval
unless red-team-grade, then applied to all checkpoints with disclosure; shared-cluster courtesy —
all jobs time-limited, chained by dependency, and interruptible between stages.

## LAUNCHED 2026-09-05

Full matrix submitted after a green end-to-end smoke (18315) that took 7 iterations, each
catching a real integration bug: log paths, grounding root, Qwen forward interface, anchor VRAM
(pair-chunking + Qwen anchor-image downsize to 448px), and O-LoRA self-Gram explosion. Two honest
scoping outcomes, both documented and defensible: AMBER dropped (images unsourceable from any
mirror); O-LoRA dropped (needs per-task adapter blocks, incompatible with our single continuously-
resumed adapter). Baselines: EWC, LwF, ER, JOINT.

Submitted (~72 train arms + chained evals + endpoint-benchmark jobs, gated afterok on 18315):
- LLaVA: seq/anchor 3 orders x 3 seeds (18); joint/er 3 seeds; 6 single-task controls; er500/cecf
  3 seeds; ewc/lwf 3 seeds. Endpoint bench (ObjHalBench+MME) on the canonical o1/s17 cell of each.
- Qwen2.5-VL: seq/anchor 3 orders x 3 seeds (18); joint/er 3 seeds; 6 single-task controls.
- Base evals + base bench for both backbones.
Drains over days of shared-cluster time; results pulled/rescored/aggregated per the endpoints
as arms complete. First verification target: the first full LLaVA arm's eval produces sane
POPE/CHAIR/task numbers (confirms the UCIT eval path end-to-end).

## Amendment 2026-09-08 — main run: rerun after disk-gate stall; new method candidates; intervention

Recorded BEFORE any of the new arms below produced a result.

- **Operational.** The 2026-09-05 launch stalled when trained adapters filled the shared disk and the
  40 GB gate correctly refused new jobs; the 20 LLaVA SEQ/anchor/joint cells had already been trained and
  evaluated (results intact). Every Qwen arm had died earlier from CUDA OOM at micro-batch 2 (fixed:
  micro-batch 1 for Qwen). Evals now delete adapters after scoring; benches keep the final adapter until
  they run. A disk-safe wave scheduler with bounded retries replaces the monolithic launch.
- **First full-study LLaVA readout (UCIT, 9 cells each):** SEQ Σ|Δc| mean 0.94 (0.60–1.20); anchor 0.72
  (0.52–1.02); anchor tighter in 7/9 matched cells (NOT 9/9). E2 as pre-registered ("every matched cell")
  is therefore NOT met on the answer-statistics-flat UCIT stream; reported as a partial effect whose
  magnitude tracks the available drift. F1 falsifier intact on every cell checked so far.
- **New method candidates (pre-registered predictions in their design notes; scored by
  analysis/method_scorecard.py P1–P4 against the same-path SEQ reference):**
  - `critp` — label-free criterion preservation (design_notes/method_crit_preserve.md): population-mean
    (optionally std) of the yes/no decision statistic on an unlabeled probe set pinned to the frozen base.
    Prediction: Σ|Δc| < SEQ without labels; d′ and plasticity intact; CHAIR@60 expected NOT to move.
    Weights 0.1 and 1.0 (λ sweep is part of the test, not a post-hoc choice).
  - `anchorcrit` — GT anchor + critp. Prediction: anchor's freeze minus its conservative offset.
  - `lamF_{0.02,0.05,0.2,0.4}` — anchor-strength sweep testing the calibration theory's inverse-strength law.
  - Post-hoc scalar criterion correction (analysis/logit_bias_analysis.py) is a zero-training BASELINE every
    method must beat, and the logit dump tests the additive-bias model directly (shift on GT-yes vs GT-no).
- **Mechanism intervention (design_notes/mechanism_intervention.md):** SEQ on prior-NEUTRAL vs
  prior-AMPLIFIED pilot-suite variants. Pre-registered: neutral Σ|Δc| well below original (toward the joint
  floor); amplified above; d′ unchanged. Falsifier: neutral drifts as much as original ⇒ drift is intrinsic to
  instruction tuning, not the answer prior.
- **Same-execution-path references:** `psL_seq`/`psL_anchor` (LLaVA, pilot suite, run_arm.py path) are the
  comparators for all pilot-suite method tests; the 2026-08 pilot numbers are context only.
- **Non-COCO generative eval (Open Images V7 val, exhaustively boxed) added to every benchmark cell** as the
  anchor domain-overlap control; Objects365 dropped (gated); OOD anchor pairs to come from Open Images TRAIN.
- **2026-09-08 (later), M1 framing + degeneracy check (vocab-B sweep).** critp is a self-anchored
  generalized-expectation criterion (Mann & McCallum 2007/2010; XR 1909.00430) — cite and differentiate.
  Added pre-registered check for every critp/anchorcrit checkpoint: per-item dispersion of g − g⁰ (std, IQR)
  from the logit dump must stay comparable to SEQ's and d′ non-inferior; collapse demotes M1 to a per-item KD
  variant. Version-watch 2609.01888v1 (faithfulness–informativeness–capability trade-off framing).
- **2026-09-08 (later), policy-anchor ablations.** Arms `policy` (GT anchor + abstention 0.1 + EOS 0.1),
  `policy_abst`, `policy_eos` (design_notes/method_multiaxis.md run plan). Pre-registered: the abstention term
  lowers TextVQA refusal-leakage vs the anchor WITHOUT a VizWiz accuracy loss (the honest risk: it fights VizWiz's
  own 'unanswerable' labels — a leakage drop bought with VizWiz accuracy is a FAIL, reported as such); the EOS term
  lowers stopping-hazard drift vs base and is not expected to move CHAIR@60 (it is completeness, not a lever).
  Scored by the P1–P4 scorecard against psL_anchor as the reference.
- **2026-09-08 (later), blind-probe diagnostic (pre-registered reading rule).** For every method-test arm the
  eval dumps POPE logits twice per checkpoint: real images and a constant-gray BLIND twin. With b_k = Δmean g
  (real) and b_k^blind = Δmean g (blind) vs base, ρ_k = b_k^blind / b_k is the language-prior share of the
  criterion drift (analysis/blind_prior_share.py). Reading rule fixed now: mean ρ > 0.7 ⇒ drift is prior-borne
  (a blind/label-free probe is a valid image-domain-free estimator; the anchor–eval image-overlap objection
  dissolves); 0.3–0.7 ⇒ mixed (report both components); < 0.3 ⇒ evidence-conditional (blind probes cannot
  see the drift; M1 must be run with real images and the image-overlap control becomes mandatory).
  d′_blind is expected ≈ 0 at every checkpoint (no evidence) and serves as the instrument's sanity check.
- **2026-09-08 (later), out-of-domain anchor arms.** `mth_anchorOOD_o1_s{17,23,31}`: the GT anchor trained on
  Open Images TRAIN pairs (design_notes/ood_anchor_openimages.md; split-disjoint from the Open Images VALIDATION
  non-COCO eval, asserted at build time). Pre-registered: (i) criterion suppression Σ|Δc| within the COCO anchor's
  seed range (the criterion mechanism is image-domain-free); (ii) on the NON-COCO generative eval, anchor-OOD's
  CHAIR ≤ anchor-COCO's within noise ⇒ the anchor's generative win is grounding, not COCO memorization;
  (iii) if anchor-OOD loses the generative win that anchor-COCO shows ⇒ the domain-overlap confound is REAL and
  the generative claim is demoted to in-domain only. Scored by the P1–P4 scorecard vs psL_seq, paired with
  psL_anchor.
- **2026-09-08 (later), design red-team consequences (design_notes/review_method_design.md) — BEFORE any method cell.**
  (a) P1 is now "Σ|Δc|(arm) < Σ|Δc|(ref) in EVERY matched cell"; a 2/3 outcome is PARTIAL, never PASS (the old
  ≥2/3 gate passed with p=0.5 under the null). (b) critp's default probe (COCO images × 80 COCO classes × the POPE
  template) is the endpoint distribution minus labels; "criterion held" on it is the objective restated and is
  matched for free by the label-free post-hoc offset. Therefore: critp is reported as a METHOD only if, in 3/3
  seeds, it holds c on a REPHRASED POPE template ("Does the image contain …?") better than PCR transferred from the
  probe (δ_k fitted on probe logits, applied to POPE) does for SEQ; otherwise it is reported as train-time PCR
  — a diagnostic. (c) Added arms: `mth_critpOI_o1_s17` (Open Images TRAIN probe) and `mth_critpNZ_o1_s17`
  (content-free noise probe) — the additive-bias claim predicts the noise probe works; either outcome is a
  finding. (d) Every method table carries PCR (label-free, probe-fitted) and the labeled b* rows.
  (e) Every mth_/psL_/lamF_ checkpoint eval dumps: real + blind POPE logits, rephrased-POPE generations + logits,
  and probe-set logits (COCO / OI / noise probes when present).
- **2026-09-08 (later), CNP (design_notes/method_cnp.md).** Criterion-null-space projection of raw gradients
  before AdamW (one-sided; deadband; optional refusal direction), arms `cnp`, `cnp_ref`. Pre-registered:
  Σ|Δc| on the PROBE ≈ 0 by construction — NOT evidence; the tests are d′ (non-inferior), plasticity (within
  0.02; expected cost 0.5–2 points), the rephrased-template transfer, and leakage under `cnp_ref` (likely inert
  with the derived COCO refusal probe — a TextVQA-format probe is prerequisite for any leakage claim). The
  update-space (Adam-NSCL-style) variant is the pre-planned fix if the raw-gradient constraint leaks.
  Honest framing: CNP = SafeGrad with the alignment loss replaced by the label-free SDT statistic, continual, VLM.
- **2026-09-08 (later), GCA (design_notes/method_gca.md).** Generative counterfactual token anchor: softplus margins
  on the caption-slot log-odds ℓ_t = log p(o⁺|J,y<t) − log p(o⁻|J,y<t) at the earliest CHAIR-countable mention,
  on the real image and across the GT-masked counterfactual; arms `gca` (λ_g 0.1), `gca` at λ_g 1.0, `anchorgca`.
  Honest label: a recombination of HALVA's ratio and TPO's raw-vs-corrupted token quantity on our GT-masked pairs.
  Pre-registered: CHAIR_i@60 S1→S3 rise reduced vs SEQ AND vs the yes/no anchor; c/d′ unchanged; Flickr plasticity
  held; a COCO-CHAIR gain is classified as vocabulary HOMOLOGY until it replicates on the Open Images (non-COCO)
  eval with a COCO-noun mention-rate control. Expected honest outcome: a small COCO-only reduction that does not
  survive on Open Images (demoted to an in-domain paragraph; the generative axis stays open). Vocab-B novelty
  sweep pending; TPO's equation to be verified at full text before citation.
- **2026-09-08 (later), vocab-B novelty on CNP and GCA (both weaker than vocab A recorded).** CNP's
  projection operator is CROWDED (AEGIS: label-free frozen-base-anchored statistic, gradient projected
  against, never in the loss, conflict-conditional, pre-optimizer, VLM backbone); CNP's delta narrows to
  the output-space DECISION statistic + continual setting + refusal direction + dead-band, and is framed as
  the hard rung of the PCR→M1→CNP ladder, not a new operator. GCA's Term 2 is CROWDED by See-or-Guess
  (ACM MM 2024 — the same GT-masked counterfactual and teacher-forced entity-token log-prob difference in
  the same loss role; vocab A's TPO attribution was wrong); GCA is framed as a reformulation whose claim
  rests entirely on the pre-registered non-COCO homology rule. Both corrections are recorded BEFORE any
  cell of either arm has run. Neither is dropped: crowded-operator candidates still test the ladder and
  the generative axis, and a negative result on either is reported as such.
- **2026-09-09, anchor third-term normalization (arm `anchorlo`; design_notes/method_calibration_theory.md §8).**
  The counterfactual term compares z_yes across two images, which is not normalization-free (IC-VCO's
  partition-function critique). `anchorlo` replaces it with the within-image log-odds difference
  g(I) − g(I∖o⁺). The default (`--faith_cf_form logit`) is byte-identical to the replicated v2 anchor, so
  no existing result moves. THEORY: under the log-odds form, §2's *assumption* that the counterfactual term
  does not constrain the decision axis becomes a consequence of the additive-bias claim (a class-independent
  shift cancels in the difference), so b*, κ, properness and the v1 analysis are unchanged; under the raw-logit
  form that projection is uncontrolled. PRE-REGISTERED: (1) primary, a prediction of NO effect — Σ|Δc|(anchorlo)
  inside the anchor's range (mean 0.72, 0.52–1.02) in ≥7/9 matched cells; a violation falsifies bias-neutrality;
  (2) d′ unchanged; (3) secondary one-sided — CHAIR_i@60 ≤ anchor's, sign-consistent; (4) monitor — the
  mean g(I) − g(I∖o⁺) gap must widen, else the arm did nothing and the endpoints are uninformative.
  OBLIGATION: a clear anchorlo win means adopting the log-odds form as the method and RE-RUNNING the headline
  anchor cells (all seeds × orderings × both backbones, plus anchorcrit / policy* / anchorgca / anchorOOD,
  which inherit term 3) — not swapping a number in.

## Full readout — LLaVA / UCIT, all 9 SEQ + 9 anchor cells complete (2026-09-09)

Scored by the pre-registered endpoint definitions (analysis/fs_aggregate.py), not by the ad-hoc script
used in the 2026-09-08 entry. **Correction to that entry:** it reported "anchor tighter in 7/9 cells"
using RAW Σ|Δc|, which includes the anchor's one-time settling step at k1. The pre-registered endpoint is
POST-SETTLING Σ|Δc|. On the pre-registered measure:

- **E2 (anchor criterion-freeze): 8/9 matched cells**, pooled suppression ratio **2.15×**, per-cell ratios
  0.78–5.28. By ordering: o1 3/3, o2 3/3, o3 2/3 (the single miss is o3/s17, where SEQ itself drifts least,
  0.4295 — the smallest SEQ value in the matrix). Post-settle means: SEQ 0.781 [0.430–1.157] vs anchor
  0.364 [0.188–0.552]. The prereg's strict "every matched cell" is therefore NOT met (8/9); reported as
  such, with the honest reading that the anchor's advantage scales with how much drift exists to suppress.
- **F1 falsifier: not tripped in any of the 9 cells** (max |Δd′| overall 0.204 < 0.30). The
  "criterion drifts, not grounding" headline holds across every seed and ordering. The anchor is also
  *more* d′-stable than SEQ (mean max|Δd′| 0.100 vs 0.175).
- **Plasticity: no cost.** anchor 0.7277 [0.7235–0.7320] vs SEQ 0.7248 [0.7140–0.7345].
- **E3 (length-controlled CHAIR rise): NOT CONFIRMED on UCIT** — 4/9 positive, 2/9 with CI above zero
  (rule needs 7/9 and 5/9). It is strongly ORDERING-DEPENDENT: o1 0/3 positive (all three CIs *below* zero,
  i.e. CHAIR@60 falls), o2 3/3 positive (2 CIs above zero), o3 1/3. The pilot's mid-sequence rise does not
  replicate as a main effect on UCIT; what replicates is an ordering-conditional effect. Reported as a
  negative result for the pre-registered form of E3.
- **Endpoint CHAIR@60: no anchor benefit on UCIT** — anchor 0.0983 vs SEQ 0.0977 (means). The anchor's
  generative win in the pilot does not appear here; consistent with the honest expectation that the
  criterion term is not a generative lever.
- E4 (memorylessness) ABSENT (single-task controls not yet run); E5 not testable on UCIT (no refusal-class
  stage, as pre-registered); JOINT cells not scored (its checkpoints are step-named, not k-named — the
  aggregator lists them under `unparsed`; fix before the JOINT column is claimed).
- **Qwen pilot suite (psQ_seq_o1_s17, 3 of 4 stages):** E1 dose-response sign MATCHES at the active
  (TextVQA) stage, zero-dose stages stay below the smallest active |Δc|, and F1 max|Δd′| = 0.068. First
  evidence the *mechanism* generalizes to a second backbone; the 4th stage and the anchor arm are pending.

### JOINT control scored (2026-09-09) — the anchor recovers 75% of the achievable range

The matched JOINT arm was silently unscored until today (its checkpoints are labelled `<RUNTAG>_step<N>`
because sb_fs_eval strips the `ckpt_` prefix, and the aggregator's regex demanded `ckpt_step<N>`; 10 cells
sat in `unparsed`). With it scored, post-settling Σ|Δc| on LLaVA/UCIT reads:

| arm | cells | post-settle Σ\|Δc\| mean [min–max] | endpoint c | max\|Δd′\| | CHAIR@60 |
|---|---|---|---|---|---|
| SEQ | 9 | 0.781 [0.429–1.157] | +0.090 | 0.204 | 0.0977 |
| anchor | 9 | 0.364 [0.188–0.552] | **+0.697** | 0.160 | 0.0983 |
| JOINT (matched upper bound) | 2 | 0.228 [0.216–0.240] | +0.088 | 0.106 | 0.1095 |

**The anchor closes 75% of the SEQ→JOINT gap in criterion drift** — without replay and without joint access.
JOINT is the data- and step-matched bound (same data, shuffled union, checkpoints at the same optimizer
steps), so this is the fraction of the *achievable* stabilization that a per-stage loss term recovers.

**But stability is not correctness, and the numbers say so plainly.** SEQ and JOINT both end near c ≈ +0.09;
the anchor ends at c = **+0.697**, i.e. it holds the criterion steady at a markedly CONSERVATIVE operating
point rather than at the joint-training one. This is exactly the margin behaviour the calibration theory
derives (a margin confines the criterion to an interval; it does not pin it to the calibrated value), and it
is a real limitation to state, not a footnote: the anchor buys stability at the cost of a biased operating
point. It also sharpens the motivation for the newer candidates — `critp` pins the decision statistic to the
BASE model's value rather than to a margin, and CNP constrains the update direction, so both predict the
same suppression WITHOUT the conservative offset. That comparison (endpoint c, not just Σ|Δc|) is hereby
pre-registered as the discriminating endpoint between the anchor and the pin/projection family.
Caveat: JOINT n=2 cells (o1/s17, s23); the third seed is queued.
- **2026-09-09, intervention reframed as an instrument (vocab-B sweep).** The neutral/amplified answer-prior
  builder is NOT claimed as a method ("prior-neutral curriculum" is unavailable: per-stage balancing against
  task-recency bias is a standard class-incremental family — EEIL's balanced training, BiC, BGS). It is an
  INSTRUMENT for the causal test only. Nearest published design: 2303.11863v1 (dose-swept, content-preserving,
  distribution-only manipulation of a continual stream with a neutral reference) — same shape, but input-side
  spurious attributes in image classification with accuracy outcomes, not the answer prior with a criterion
  outcome at fixed d′. New must-cites: VQA-CP (1712.00377v2), EEIL (1807.09536v2), CIL survey (2010.15277v3),
  2303.11863v1. The pilot's recency finding must be named as TASK-RECENCY BIAS and cited, not described afresh.
- **2026-09-09, GCA's preemptors run as baselines (vocab-B obligation).** Arms `gca_t1` (Term 1 only —
  the HALVA-style slot-decision baseline) and `gca_te` (Term 2 only — the See-or-Guess-style GT-masked
  counterfactual baseline, i.e. the term vocab B found is CROWDED), both in OUR continual setting via
  `--gca_terms {both,decision,cf}`. Pre-registered: GCA (both terms) counts as a contribution over its
  preemptors ONLY if it beats BOTH single-term arms on endpoint CHAIR_i@60 with the paired CI excluding
  zero, on the non-COCO Open Images eval as well as COCO (the homology rule). If `gca_te` alone matches
  GCA, the honest report is that the published counterfactual term already suffices in the continual
  setting and our reformulation adds nothing — written as such, not omitted.

### Why E3 failed: it was mis-specified as a DEPTH effect (2026-09-09, post-hoc analysis, labelled as such)

E3 asked whether length-controlled CHAIR_i@60 RISES from stage 1 to stage 5 — a depth/accumulation claim.
It came back 4/9. Examining the 54 (cell x stage) SEQ observations shows the failure is not noise but a
mis-specification, and the three orderings (which the single-ordering pilot could not do) are what reveal it:

- **No accumulation.** By depth, mean CHAIR@60 is 0.0932, 0.1070, 0.0959, 0.0954, 0.0848, 0.0977 for stages
  1-6: non-monotone, maximum at stage 2, and stage 6 ~ stage 1. The accumulation hypothesis makes a
  directional prediction and it fails outright.
- **Task identity dominates.** Grouping the same 54 observations by the task JUST TRAINED gives
  eta^2 = 0.276 versus 0.134 by depth — roughly twice the variance explained. Task means (n=9 each):
  CLEVR-Math 0.1112 > ArxivQA 0.1048 > VizWiz 0.0932 > IconQA 0.0916 > Flickr30k 0.0895 > ImageNet-R 0.0836.
- **The E3 sign flips are exactly the endpoint tasks.** o1 is ArxivQA -> ImageNet-R (high -> low: all three
  seeds negative), o2 is VizWiz -> CLEVR-Math (low -> high: all three positive), o3 is ImageNet-R ->
  Flickr30k (near-equal: ~0). A task-mean model predicts the nine per-cell deltas with r = 0.955 and 7/9
  sign agreement, the two misses being o3 where the predicted effect (+0.006) is within noise.
- **CIRCULARITY, stated:** those task means were fit on the same 54 observations, so r = 0.955 is NOT an
  out-of-sample test and must never be reported as one. It is a description of structure. The load-bearing
  evidence is the failed depth prediction and the seed-stable sign flips.
- **Pre-registered out-of-sample test (arms already queued):** the single-task controls `fsL_single_<task>`
  train on ONE task only, so they estimate each task's CHAIR@60 with no sequence at all. If those values
  reproduce the task means above (rank order preserved, and the o1/o2 endpoint gaps recovered within the
  seed band), the recency reading is confirmed out-of-sample; if they do not, this analysis is withdrawn.

**Consequence for the paper.** This is the same TASK-RECENCY BIAS (cilsurvey; eeil) that the pilot's
single-task controls found for the criterion, now appearing on the generative axis: what a checkpoint
hallucinates tracks the task it was last trained on, not how much tuning it has accumulated. The pilot's
"mid-sequence rise" was measured on ONE ordering, where depth and task identity are perfectly confounded;
with three orderings they separate, and task identity wins. E3 is reported as a FAILED pre-registered
endpoint whose failure is explained, not as a positive result.

### External benchmarks: first real scores, and a domain-shift result (2026-09-09)

Every `BENCH_DONE` directory before today contained only a gate file — the benchmark battery had been
silently scoring nothing for the whole run (Object HalBench images never fetched; MME image path mismatch;
a bench that marked itself done regardless). Repaired, with the bench now failing loudly on zero scores.
First real endpoint numbers, LLaVA sequential arm (o1/s17):

- **Non-COCO generative eval (Open Images V7 val, 500 captions): CHAIR_i = 0.2605** (CHAIR_s 0.284,
  1.81 mentions/caption).
- **MME-Hallucination existence: acc 0.983, score 195/200, yes-rate 0.483.**

**Length confound checked before use, and it is not one.** The non-COCO rate is ~2.5x the COCO CHAIR rate
(~0.10), which would be meaningless if the two evals elicited different caption lengths. They do not: both
use the identical prompt ("Please describe this image in detail.") and mean generated length is comparable
(COCO 30.3 tokens, non-COCO 25.7, Object HalBench 21.5). The gap is therefore a genuine DOMAIN effect —
the model hallucinates markedly more on images outside its adaptation domain — which is precisely why this
eval is the anchor's domain-overlap control. Any anchor CHAIR gain that appears on COCO must be shown here.

**Object HalBench:** scored 298/300 then died on `KeyError '474398'` — two of its 300 ids are COCO
TRAIN2014 images while our GT was built from val2014 only (the official scorer combines train+val). Rebuilt
with `--merge_train` and asserted that every prompt id now has GT before scoring; no silent drops.

### Free result: two independent instruments agree on the size of the domain gap (2026-09-09)

Noticed while verifying that the out-of-domain anchor arm had loaded the right pairs. The base model's
pre-training anchor loss is a deterministic, training-free measurement of how far it already is from the
anchor's grounding margins. Same backbone (LLaVA-1.5-7B), same loss, same hyperparameters (k=4, m=2.0),
same 2000 pairs, differing ONLY in image source:

| anchor pair set | pre-train faith loss |
|---|---|
| COCO (`data/grounding`) | **1.5770** (bit-identical across seeds 17/23/31, as it must be — no training has happened) |
| Open Images TRAIN (`data/grounding_oi`) | **4.1618** |

**Ratio 2.64×.** Independently, the generative instrument gives non-COCO CHAIR_i 0.2605 vs COCO ~0.10,
a ratio of ~2.5×, with prompts identical and caption lengths comparable. Two instruments that share no
machinery — a discriminative yes/no logit margin on grounding pairs, and generative caption-level CHAIR —
agree to within ~5% on how much worse the model's grounding is off-domain. That mutual corroboration is
what licenses treating the Open Images eval as a genuine domain-shift probe rather than a metric artifact.

**Honest scope.** The two pair sets differ in more than image distribution: object vocabulary, box
annotation convention (COCO instance masks vs Open Images boxes) and per-image object density all differ,
so "domain gap" here bundles those together and should be worded as such, not as a pure image-distribution
effect. It also raises the OOD anchor arm's difficulty: it starts far from satisfied margins, so a smaller
suppression there is expected and must not be read as the method failing.

### E1 null-side: FAILED, and it bounds our own mechanism claim (2026-09-09)

Surfaced by an internal consistency audit (design_notes/review_consistency_2026-09-09.md) which found this
pre-registered result had run and was NOT being reported. Recording it prominently because it cuts against us.

The UCIT stream is certified zero-dose by the blinding rule, frozen BEFORE any eval: every one of the six
tasks has yes = 0.000, no = 0.000 (VizWiz 0.0005), refusal = 0.000. §4 pre-registered the null side of
dose-response: **a zero-dose stream should show little criterion drift.** Observed, LLaVA SEQ, 9/9 complete
cells (base c = 0.4314):

| ordering | max \|Δc from base\| |
|---|---|
| o1 | 0.418–0.529 |
| o2 | 0.517–0.550 |
| o3 | 0.403–0.446 |

Across all nine cells max|Δc| = 0.403–0.550 (mean 0.472), and the endpoint displacement is the SAME
(liberal) direction in 9/9 cells, −0.171 to −0.446. For scale, the pilot's largest single dose-driven push
was +0.567 and its headline swing 0.78.

**The prediction is NOT met, and the honest consequence is a bound on the answer-statistics mechanism:**
answer statistics are not the whole story. Continual instruction tuning displaces the criterion by an
amount comparable to a strong dose even when the training targets contain no yes/no/refusal signal at all.
What the dose-response evidence continues to support (pilot suite, and the Qwen pilot-suite stage) is that
the SIGN of a stage's criterion movement tracks that stage's answer prior; what it does not support is that
answer statistics explain the MAGNITUDE of drift, since a flat stream produces ~0.47 units of it.
"Not testable on UCIT" was the wrong report: the signed dose-response is untestable there, but the
null-side magnitude is exactly what UCIT does test, and it failed.

### Statistical spec recorded (2026-09-09) — F26/F27 from the consistency audit

Recorded so these are not post-hoc choices. Both were already implemented in the code that produced the
landed numbers (`analysis/fs_aggregate.json`); this entry states them.

- **E3 bootstrap.** Paired bootstrap over the CHAIR evaluation set with B = 10,000 resamples, seed 17, the
  resampling unit being the 500 CHAIR **images** (not captions or mentions). CIs are per-cell; they are never
  pooled across seeds or orderings, per the multi-seed style guide.
- **Generation-budget (truncation) audit.** Every CHAIR@k figure is reported with the fraction of captions
  that hit the generation budget, because a fixed `max_new_tokens` turns "longer outputs" into an apparent
  metric change. Standing rule, applied to the landed matrix: a CHAIR@60 DIFFERENCE between two checkpoints
  whose truncation fractions differ materially is not a calibrated effect size. It does not reach signs,
  seed-consistency, or the criterion / d' / plasticity endpoints (which are not length-budgeted).
  **Consequence already carried in the paper:** under o1/o2 this binds exactly the stage-1-vs-stage-5
  contrast, and the depth-vs-recency reanalysis built on those cells inherits the caveat. Truncation
  fraction also covaries with depth (86.8-88.2% of stage-1 captions truncated vs 1.2-14.0% at stage 5 under
  o1), so part of the depth profile could be budget rather than model — an artifact that would work IN
  FAVOUR of the reanalysis's own conclusion, which is a reason to hold it loosely, not to prefer it.

### Benchmark arm comparison is LENGTH-CONFOUNDED as currently computed (2026-09-09)

Four bench cells have landed (LLaVA base, SEQ, anchor; Qwen base). The raw numbers invite a wrong reading
and must not be used for an arm comparison until length-controlled:

| cell | non-COCO CHAIR_i | CHAIR_s | mentions/caption | mean tokens | MME total | MME existence | ex yes-rate |
|---|---|---|---|---|---|---|---|
| llava15 base | 0.3401 | 0.7100 | 6.28 | 104.2 | 1249.98 | 190 | 0.4667 |
| fsL_seq o1/s17 | 0.2605 | 0.2840 | 1.81 | 26.5 | 1206.68 | 195 | 0.4833 |
| fsL_anchor o1/s17 | 0.3276 | 0.4040 | 3.13 | 52.1 | 1193.34 | 175 | 0.4167 |
| qwen25vl base | 0.2010 | 0.4940 | 5.58 | 186.8 | 1410.00 | 200 | 0.5000 |

**Two findings and one blocker.**

1. **Continual tuning collapses caption length**: base 104.2 tokens / 6.28 mentions → SEQ 26.5 / 1.81. The
   anchor partially preserves it (52.1 / 3.13). This is the format/length-collapse confound the pilot
   already flagged, now visible on an out-of-domain generative eval.
2. **BLOCKER for any arm comparison here:** CHAIR_s tracks mentions-per-caption almost monotonically across
   all four rows (0.710/6.28, 0.404/3.13, 0.284/1.81, 0.494/5.58), so the raw bench CHAIR compares arms that
   differ 2–4× in output length. Read naively the anchor looks WORSE than SEQ (0.3276 vs 0.2605) while
   producing twice the text — precisely the artifact our length-controlled CHAIR@k exists to remove. The
   bench currently computes RAW CHAIR only. **Required before any bench-based arm claim:** compute
   length-controlled CHAIR@60 on the non-COCO eval (and report per-arm truncation/length), exactly as the
   in-domain CHAIR pipeline does. Until then these cells support only within-arm and base-vs-tuned
   statements, not anchor-vs-SEQ.
3. **Independent corroboration of the conservative offset:** on MME existence the anchor's yes-rate is
   0.4167 vs SEQ 0.4833 and base 0.4667 — the anchor answers "yes" less often on a benchmark it was never
   tuned against, which is the same conservative displacement the criterion endpoint shows (c = +0.697).
   Its existence score is correspondingly lower (175 vs 195). That is the cost of the offset, visible on an
   external instrument.

### RESOLVED: the bench length confound manufactures BOTH a false effect and a sign inversion (2026-09-09)

Rescored the existing bench generations (no regeneration) with the audited in-domain scorer
`analysis/fs_common.py::score_chair` via the new `analysis/bench_lc_chair.py`, sweeping the word budget k
and running a paired image-level bootstrap (4000 reps, resample = LIST of image ids with multiplicity;
CHAIR_i re-summed as a ratio per replicate, never averaged over per-image rates).

**Matching criterion (pre-registered here, before reading the pairs):** two cells are comparable only when
their mentions@k per caption differ by <10%. Mentions, not words, are CHAIR_i's denominator and therefore
the opportunity count that must be held fixed. Any pair outside that band is reported LENGTH-LIMITED and
may not be quoted as an effect.

non-COCO Open Images, anchor vs SEQ:

| k | mentions@k ratio | ΔCHAIR_i | 95% CI | reading |
|---|---|---|---|---|
| 60 | 1.580 | +0.0450 | [+0.0197, +0.0713] | LENGTH-LIMITED — CI excludes 0 but the arms differ 1.6× in opportunity |
| 30 | 1.234 | — | — | LENGTH-LIMITED |
| 20 | 1.088 | — | — | borderline |
| **12** | **0.993** | **−0.0024** | **[−0.0180, +0.0131]** | **matched — NULL** |

**The "significant" anchor deficit at k=60 is manufactured entirely by caption length.** At matched
opportunity the difference is −0.002 with a CI straddling zero. The gap shrinks monotonically as the
mention ratio closes (0.045 → 0.007 → 0.008 → −0.002), which is the signature of an opportunity artifact.

**Sign inversion on the tuned-vs-base comparison.** Same data, same images, two budgets:

| comparison | k=60 (ratio 0.35, unmatched) | k=12 (ratio 1.015, matched) |
|---|---|---|
| SEQ − base, non-COCO | **−0.0498** [−0.0782, −0.0214] | **+0.0292** [+0.0065, +0.0529] |

Raw CHAIR says continual tuning REDUCES out-of-domain hallucination; the matched comparison says it
INCREASES it, both with CIs excluding zero. The raw number is not merely noisy, it points the wrong way,
because sequential tuning collapses caption length (81 → 21 words) and short captions have fewer chances
to err. This is the same failure class as the format-collapse confound already on file, on a new axis.

**Consequences.**
1. **This is a measurement-side contribution, not a nuisance.** Endpoint POPE + raw CHAIR after the stream
   is becoming standard practice in CL-for-MLLM (MCITlib, D-MoLE, LLaVA-c, FCIT). Raw CHAIR at
   max_new_tokens=512 sits squarely in the k=60+ regime where we just showed the sign can invert. Any
   CL-for-MLLM paper comparing raw CHAIR across arms that differ in verbosity is reporting an artifact.
2. **Established finding, matched and bootstrapped:** continual instruction tuning raises out-of-domain
   generative hallucination by ~0.03 CHAIR_i at matched opportunity (SEQ +0.0292 [+0.0065, +0.0529] vs
   base), and **the anchor does not repair it** (anchor vs base +0.0268 [+0.0033, +0.0504]; anchor vs SEQ
   null). This CORROBORATES, on an out-of-domain benchmark, the already-reported in-domain result that the
   anchor buys criterion stability with no generative benefit (UCIT 0.0983 vs 0.0977). Two independent
   generative evals now agree.
3. **Limitation of the k=12 matched point:** it truncates the base to ~15% of its caption, so it compares
   hallucination among EARLY mentions only, and hallucination is known to concentrate later in a caption.
   The matched comparison is therefore specific to early mentions and likely conservative. The
   mention-index-matched analysis (hallucination rate at mention #1, #2, #3 …) removes the budget choice
   entirely and is the pre-registered follow-up.
4. Object HalBench is NOT yet matched at k=12 (ratios 1.197 / 1.155) and none of its pairs may be quoted
   until a matched budget is found.

### Mention-index control: anchor-vs-SEQ null CONFIRMED; tuned-vs-base WEAKENED (2026-09-09)

`analysis/mention_index_chair.py` removes the budget choice entirely. For each caption we walk its object
mentions in order and record whether mention #i is hallucinated, then compare arms AT THE SAME INDEX.
Index i is equal opportunity by construction: every caption reaching index i contributes exactly one
observation, in every arm. No truncation, no post-hoc k.

**Result 1 — anchor vs SEQ is null under a second, independent control.** Paired bootstrap at matched
index, non-COCO: #1 −0.007 [−0.018, +0.004]; #2 +0.006 [−0.034, +0.045]; #3 0.000 [−0.096, +0.096];
#4 −0.097 [−0.290, +0.097]. Every CI straddles zero. Together with the k=12 matched result (−0.002
[−0.018, +0.013]) the raw-CHAIR "anchor is worse" reading is now refuted by two controls that share no
assumptions. **This is settled.**

**Result 2 — the tuned-vs-base increase does NOT survive this control, and I am downgrading it.** The
pooled k=12 estimate was SEQ − base = +0.029 [+0.007, +0.053], CI excluding zero. Per mention index the
effect localizes to index #2 only (+0.082 [+0.016, +0.147]); indices #1, #3, #4 are all null (+0.015
[−0.004, +0.037]; −0.013 [−0.115, +0.090]; +0.029 [−0.143, +0.200]). One significant result out of four
tested indices would not survive a multiplicity correction (Bonferroni α = 0.0125; the index-#2 interval's
lower bound sits at +0.016, i.e. nowhere near that). **Status: SUGGESTIVE, NOT ESTABLISHED.** The prereg
entry above is amended accordingly — the sentence "continual instruction tuning raises out-of-domain
generative hallucination by ~0.03 CHAIR_i at matched opportunity" must NOT be written as a finding until
the additional seeds land. What IS established is the anchor-vs-SEQ null and the sign-inversion
demonstration, neither of which depends on this.

**Result 3 — POSITIONAL HALLUCINATION HAZARD (new, and the mechanism behind the whole confound).**
Restricting to the FIXED set of captions with ≥5 mentions (identical caption set at every index, so the
"longer captions depict busier scenes" selection effect is gone), the hallucination rate rises
monotonically with mention position:

| arm (fixed set) | #1 | #2 | #3 | #4 | #5 | rise |
|---|---|---|---|---|---|---|
| llava15 base (n=344) | 0.125 | 0.230 | 0.311 | 0.387 | 0.442 | 3.5× |
| anchor (n=115) | 0.157 | 0.296 | 0.339 | 0.339 | 0.443 | 2.8× |
| qwen25vl base (n=319) | 0.129 | 0.157 | 0.169 | 0.197 | 0.213 | 1.65× |
| SEQ (n=29) | 0.241 | 0.276 | 0.276 | 0.310 | 0.310 | — **n too small, DO NOT READ** |

This explains *why* raw CHAIR is length-confounded rather than merely asserting that it is: mentions are
not exchangeable, so adding mentions adds disproportionately hallucination-prone ones. Secondary
observation: the stronger backbone (Qwen) has a markedly flatter hazard (1.65× vs 3.5×).

**Caveats, stated before the claim hardens.** (i) The fixed set is itself the subset of captions with ≥5
mentions, i.e. long captions; the hazard for short captions is not measured and may differ. (ii) SEQ's
n=29 is far too small and its row is excluded from every reading. (iii) The hazard is a within-caption
association, not a causal claim about generation order. (iv) NOVELTY IS UNRESOLVED — a positional
hallucination effect may already exist in the VLM literature (LURE arXiv 2310.00754 relates hallucination
to object position and caption length; the EOS-decision paper relates it to sequence length; "snowballing"
is the text-only analogue). A check is running. **Nothing in Result 3 may be written as ours until that
check returns.**

### CHAIR IS PRECISION-ONLY: the anchor HAS a generative benefit, hidden by the metric (2026-09-09)

CHAIR_i = hallucinated mentions / total mentions = 1 − precision over object mentions. It contains no
recall term, so it is gamed by saying less. Our arms do exactly that: sequential tuning takes LLaVA from
7.53 mentions/caption to 1.79 on Object HalBench, and CHAIR duly "improves" 0.171 → 0.067.

`analysis/object_prf.py` scores the SAME generations with precision, recall and F1. GT per image is built
identically to `fs_common.score_chair` (instance objects ∪ objects extracted from the reference captions),
and the precision column reproduces 1 − chair_i exactly, so this is not a redefinition.

**Object HalBench (n = 300 images, single cell o1/s17):**

| arm | precision | recall | F1 | ment/cap | CHAIR_i |
|---|---|---|---|---|---|
| llava15 base | 0.8292 | 0.7549 | 0.7903 | 7.53 | 0.1708 |
| SEQ | **0.9330** | **0.4600** | 0.6162 | 1.79 | **0.0670** |
| anchor | 0.8798 | 0.5826 | **0.7010** | 3.44 | 0.1202 |

Paired image bootstrap, 4000 reps: **anchor − SEQ = +0.0848 F1 [+0.0593, +0.1120]**, CI excludes zero;
SEQ − base = −0.1741 [−0.2051, −0.1439]; anchor − base = −0.0893 [−0.1147, −0.0648].

**Reading.** Sequential continual tuning degrades generative object coverage severely (recall 0.755 →
0.460) while *improving* the metric the field reports. The anchor closes 0.0848 of the 0.1741 F1 gap,
i.e. **~49% of the generative degradation** — a benefit that CHAIR not only misses but reverses in sign
(CHAIR says the anchor is worse, 0.1202 vs 0.0670). This directly amends the standing limitation "the
anchor shows no generative benefit": that statement was an artifact of a precision-only metric.

**What this does NOT yet establish, stated before the claim hardens.**
1. **Single cell.** o1/s17 only. The bootstrap CI is over IMAGES WITHIN ONE RUN and therefore quantifies
   image-sampling noise, NOT run-to-run variance. Per the standing noise-floor rule an image-level interval
   may not be presented as if it were a seed-level one. **Replication across orderings/seeds is required
   before this is written as a result.**
2. **It does not replicate on the non-COCO eval**, where every F1 pair is null (anchor − SEQ −0.0130
   [−0.0323, +0.0058]). A plausible reason is a ceiling: non-COCO base recall is 0.9197 vs 0.7549 on
   Object HalBench, leaving far less headroom. That explanation is POST-HOC and is not evidence.
3. Recall is measured against a GT set that unions instance annotations with caption-extracted objects, so
   it is a relative quantity comparable across arms on the same images, not an absolute coverage rate.

**Consequence for the paper.** Reporting CHAIR alone is not merely incomplete, it is directionally wrong
for continual learning, because continual tuning's characteristic failure (output shortening) moves
precision and recall in opposite directions. Any CL-for-MLLM result that reports endpoint CHAIR without a
coverage term can be satisfied by degeneration. **Pre-registered next step:** bench cells on additional
orderings/seeds, then the anchor F1 recovery is either confirmed as a method result or withdrawn.

## CRITICAL (2026-09-09): the anchor's ENDPOINT is worse than doing nothing — 9/9, no overlap

Computed directly from `analysis/readout/fs_aggregate.json` (generated 2026-09-09T07:32, all 20 cells).
This was NOT in the pre-registered endpoint set; it is the first thing a reviewer will compute.

| arm | n | endpoint c | d′ | POPE F1 | balanced acc |
|---|---|---|---|---|---|
| llava15 base | — | +0.431 | 2.348 | 0.8449 | 0.8585 |
| SEQ | 9 | **+0.090** | 2.227 | **0.8625** | 0.8652 |
| anchor (ours) | 9 | **+0.697** | 2.345 | **0.7963** | 0.8255 |
| JOINT bound | 2 | +0.088 | 2.255 | 0.8667 | 0.8692 |

**The separation is complete: max anchor F1 = 0.8167 (o3/s17) < min SEQ F1 = 0.8547 (o1/s31).** Nine
versus nine, no overlap, so this is not sampling noise. The anchor costs **6.6 POPE F1 points and 4.0
points of balanced accuracy relative to running no method at all.**

**Two facts that reframe the paper's method story.**

1. **d′ is flat across every arm** (2.23 / 2.35 / 2.25 vs base 2.35). The entire F1 deficit is criterion
   placement. The anchor does not damage the model's ability to tell present from absent objects; it
   answers "no" too often. That is consistent with everything else we report, and it means the deficit is
   in principle *removable by a scalar*, which is the opening for the corrected-target method below.
2. **Sequential's ENDPOINT is already essentially optimal** (c = +0.090 vs JOINT's +0.088; F1 0.8625 vs
   0.8667). What sequential does badly is the PATH, not the destination. Our Σ|Δc| headline measures path
   roughness, and it is a real effect, but it must never be worded so as to imply the sequential endpoint
   is bad. It is not.

**Why the anchor lands wrong, and it is diagnosable.** The anchor and `critp` both regularize toward the
FROZEN BASE's statistics, and the base itself sits at c = +0.431 — conservative. Anchoring to a
mis-placed target inherits and amplifies the mis-placement (+0.431 → +0.697). This is a design flaw in the
target, not in the mechanism.

**POPE is exactly balanced** (n_gt_yes = n_gt_no = 4500 in every cell). The Bayes-optimal criterion is
therefore c = 0 and is DERIVABLE, not a values choice. `design_notes/method_calibration_theory.md` §4
currently frames the target as "a values choice"; that framing is wrong for this benchmark and must be
corrected.

**Consequences, binding.**
- The claim "the anchor closes 75% of the SEQ→JOINT gap" stays true for post-settling Σ|Δc| but MUST be
  reported alongside the endpoint cost in the same table. Reporting the trajectory win without the
  endpoint loss would be selective reporting.
- The anchor's status changes: it is a **causal probe demonstrating criterion drift is controllable**, not
  a deployable method. Written as a method with these numbers it is indefensible.
- **This defines the method target precisely:** reduce drift AND land at c ≈ 0, label-free. A method that
  achieves both beats every arm we have, including JOINT, on the endpoint. That is now the method program.

### NOVELTY RETRACTION: length-controlled CHAIR is PREEMPTED; my mechanism argument was also wrong (2026-09-09)

A two-vocabulary sweep with primary-source verification came back **PREEMPTED** on the claim recorded
earlier today that "raw CHAIR comparisons across arms of differing verbosity are invalid" is our
methodological contribution. It is established prior art, at strong venues:

- **LeHaCE (NeurIPS 2024 main track**, DOI 10.52202/079017-3538, not on arXiv; venue verified at
  proceedings.neurips.cc). Length-controlled CHAIR is its ENTIRE contribution: it evaluates hallucination
  "at a uniform image description length to mitigate the effect of description lengths." Its Table 1 shows
  **rank reversal under matched length** (InstructBLIP best at CHAIR_i@20, worst-ranked by @80). Its one
  stated gap: it does not control length by truncating generated descriptions.
- **CCEval / HallE-Control (arXiv 2310.01779v1)**: explicitly holds average sentence length and object
  count constant, and states that comparing object hallucination "is impractical when there is a
  significant disparity in average sentence length and number of objects" — our "significant difference
  becomes null", already published. Venue NO-DATA (DBLP/S2 blocked), NOT cleared as unpublished.
- **POPE (EMNLP 2023, 2305.10355)** has a subsection "Disadvantages of CHAIR" reporting that it is "biased
  to short captions" and that model ORDER changes.
- **OPERA (CVPR 2024, 2311.17911v3)** restricts max-new-tokens to two fixed budgets "for fair evaluation".

**My mechanism argument was ALSO wrong and is corrected here.** I justified matching by "opportunity":
more mentions means more chances to err. That explains CHAIR_s, which is a per-caption rate, but CHAIR_i
is ALREADY per-mention, so opportunity alone cannot move it. What actually drives our effect is the
**position-dependent hazard** measured today (rate rises from 0.125 at mention #1 to 0.442 at #5 on a
fixed caption set). The correct statement is: mentions are not exchangeable, later ones are far more
error-prone, so a verbose arm's mention pool is drawn from a worse part of the hazard curve.

**And the positional hazard is itself partly prior art**, which retracts a second claim made today:
- **Rohrbach et al. (CHAIR's own paper, 1809.02156v2, EMNLP 2018)**: "hallucinated objects tend to be
  mentioned towards the end of the sentence (on average at position 6, with average sentence length 9)."
  The same paper reports "no obvious correlation between the average length of the generated captions and
  the hallucination rate" — the OPPOSITE of our premise, though measured on ~9-word LSTM captions, a scope
  limitation rather than a refutation. Both must be addressed in the paper; reviewers will raise it.
- **AMBER (2311.07397v2)**: "hallucinatory objects frequently occur in the middle and latter parts."

**What actually survives as ours (claim ONLY this):**
1. Post-hoc truncation of FIXED generations to a budget calibrated so mentions-per-caption match within
   10% — LeHaCE explicitly declines truncation, CCEval declines strict enforcement, OPERA matches a token
   budget rather than mentions.
2. Paired image-level bootstrap intervals on the matched contrast; none of the above report intervals.
3. **Mention-index-matched hallucination rate** — not found anywhere in the sweep.
4. The application to WITHIN-MODEL CHECKPOINT SEQUENCES (continual learning) rather than cross-model
   comparison. All four preemptors compare different models at one time point.
5. The pre-registered severity outcome: our own second arm going null (+0.045 → −0.002).

The framing "raw CHAIR comparisons are invalid" must be stated as ESTABLISHED, citing POPE, CCEval,
LeHaCE and OPERA, with our contribution positioned as the enforced, interval-estimated instrument applied
along a training trajectory.

**CONCURRENT-WORK ALERT (highest risk on file).** arXiv **2609.01888v1**, posted ~2 Sept 2026 — one week
ago — "Does Playing it Safe Count as Faithfulness?", reported to show CHAIR_s vs recall r = 0.73 and that
protocols "may overestimate progress by rewarding conservative generation." That is the same insight as
today's precision/recall result. A primary-source verification is running. Until it returns, the
precision/recall finding must NOT be written as novel.

**Repo correction required:** `design_notes/method_extension_novelty.md:112` records HallE-Control as
"not length/EOS … orthogonal to length". That is true of its ε control knob but FALSE of CCEval, the
paper's own benchmark, whose stated motivation is exactly length/object-count disparity. Fix that line so
it stops hiding our closest preemptor.

### Second independent sweep: mention-index is ALSO preempted (LURE); metrology angle is closed (2026-09-09)

A second sweep, run in a different vocabulary and verified at primary source with per-document positive
controls, confirms the LeHaCE/CCEval/POPE/OPERA preemption above and adds one more that removes the last
metrology item I had listed as surviving:

- **LURE (arXiv 2310.00754v2, ICLR 2024)** §2.3 defines **PoScore_{s,i} = Index(o_{s,i}) / N_s**, a
  normalized object-mention index, and reports that "dominant hallucinations occur in the latter part of
  the descriptions." Our "mention-index-matched hallucination rate" is therefore a re-derivation of an
  existing diagnostic, not a new one. **Retract item 3 of the surviving list.**
- **AMBER (2311.07397v2)** §4.4 already truncates responses at different lengths to trace the
  hallucination/coverage trade-off, though only at *relative* (within-model) lengths, and its main
  evaluation is explicitly untruncated.
- **MMHal-Bench (2309.14525v1, Findings of ACL 2024)** already documents the mirror-image bias in its
  limitations: "short or evasive responses can inadvertently attain high scores."

**Useful for us rather than against us:** LURE's own Table 17 shows average description length falling for
all six models it is applied to (e.g. 67.08 → 56.63, 102.8 → 96.39) while the paper describes this as
"minor changes to the description length." LURE is thus a live published example of a CHAIR improvement
confounded with a length reduction — a concrete demonstration target for the instrument, and a much
stronger use of our tooling than claiming the confound itself.

**DECISION: the CHAIR-metrology angle is closed as a contribution.** It is now a METHODS section that
cites POPE, CCEval, LeHaCE, OPERA, Rohrbach, AMBER and LURE, states the confound as established, and
contributes only the enforced mention-matched instrument with bootstrap intervals applied along a training
trajectory. No further effort is to be spent trying to make it a headline. Both independent sweeps
converged on the same conclusion about where our ground actually is: **criterion drift across a continual
task sequence is unpreempted**, and that is the paper.

**Outstanding NO-DATA on this topic, recorded so it is not mistaken for clearance:** the disjoint
vocabulary "exposure / rate denominator / base rate / per-mention risk / hazard" was NOT swept, and LeHaCE
was invisible to arXiv search entirely (no arXiv version), so the non-arXiv proceedings channel needs a
second pass before submission. "Multi-Object Hallucination in Vision Language Models" (NeurIPS 2024) was
spotted in the same proceedings index and is unread.

### REPLICATION FAILURE: the anchor's F1 advantage does NOT hold across the 9-cell matrix (2026-09-09)

The single-cell Object HalBench result (+0.0848 F1, anchor over SEQ) was replicated at zero compute cost
by running the same precision/recall decomposition on the IN-DOMAIN CHAIR generations, which already exist
for all 18 full-study cells (`analysis/fs_object_prf.py`). Unit of replication = the cell, matching the
rest of this pre-registration.

**Primary result: NULL.** F1 favours the anchor in 5/9 cells, mean ΔF1 = 0.0219 (sd 0.0360), exact
two-sided sign test **p = 1.0**. CHAIR and F1 also agree in direction here (both 4/9 for SEQ), so the
sign-inversion seen on the bench does NOT occur in domain. **The claim that the anchor recovers generative
coverage is NOT established and must not be written as a result.**

| cell | SEQ ment/cap | ΔF1 (SEQ − anchor; negative = anchor better) |
|---|---|---|
| o1_s17 | 2.46 | −0.0766 |
| o1_s31 | 2.45 | −0.0860 |
| o1_s23 | 4.62 | −0.0260 |
| o3_s31 | 3.84 | −0.0230 |
| o3_s17 | 4.34 | +0.0048 |
| o3_s23 | 5.59 | +0.0127 |
| o2_s17 | 6.92 | −0.0071 |
| o2_s23 | 7.18 | +0.0021 |
| o2_s31 | 7.26 | +0.0018 |

**Post-hoc structure, flagged as post-hoc.** The anchor's advantage tracks how far sequential tuning
collapses output: Pearson r(SEQ mentions/caption, ΔF1) = **0.809**, t(7) = 3.64. Where sequential keeps
its verbosity (ordering o2, 7.12 mentions) the anchor buys nothing (mean ΔF1 −0.0011); where sequential
collapses (ordering o1, 3.18 mentions) the anchor recovers a lot (mean ΔF1 −0.0629). Mechanistically
coherent: there is nothing to restore unless the reference arm degenerates.

**Why I am not treating that as a rescue.** Spearman ρ = 0.633 against Pearson 0.809 says the relation is
carried by the two extreme o1 cells; n = 9; the moderator (SEQ's own mention count) was chosen AFTER
seeing which cells moved; and I went looking for it precisely because it would save a claim that had just
failed. That is the motivated-data-request pattern this project has recorded before, and it gets the same
scrutiny as the claim it rescues. **Status: hypothesis for out-of-sample test, not a finding.**

**Pre-registered test that would settle it, written before running:** ordering o1 is the collapse regime.
If the anchor's coverage benefit is real and conditional on collapse, then on the additional o1 cells and
on the Object HalBench bench cells for o2 (where SEQ does not collapse) the anchor's ΔF1 must be ≈ 0,
while remaining clearly negative on o1. If the anchor also shows a large benefit on a non-collapsing
ordering, the moderator is wrong and the whole coverage story is withdrawn.

**Standing correction to the earlier entry today:** the sentence "the anchor closes ~49% of the generative
degradation" describes ONE cell on ONE benchmark and is now known not to generalize in domain. It is
withdrawn as a claim and retained only as the o1/s17 observation that motivated the matrix test.

### THE METHOD STORY: the anchor protects d′ best and is one scalar away from beating JOINT (2026-09-09)

The endpoint deficit recorded above is entirely criterion placement, and the anchor has the HIGHEST d′ of
any arm. Under equal-variance Gaussian SDT — the same model that already defines the c and d′ this paper
reports, so no new assumption — balanced accuracy at the optimal criterion c=0 is Φ(d′/2):

| arm | n | d′ | balanced acc NOW | at c = 0 | gain |
|---|---|---|---|---|---|
| SEQ | 9 | 2.2266 | 0.8652 | 0.8672 | +0.0020 |
| **anchor** | 9 | **2.3454** | 0.8255 | **0.8794** | **+0.0539** |
| JOINT | 2 | 2.2546 | 0.8692 | 0.8702 | +0.0010 |
| base | — | 2.3477 | 0.8585 | 0.8798 | +0.0213 |

**Paired per cell, the anchor's corrected ceiling exceeds SEQ's in 9/9 cells** (diffs +0.0026 … +0.0321,
every one positive), and the anchor's corrected mean (0.8794) exceeds the JOINT bound's (0.8702).

**Reading.** The anchor is not a worse model wearing a bad threshold by accident — it is the arm that best
PRESERVES the model's discriminative information (d′ 2.3454 vs base 2.3477, i.e. near-perfect retention,
while SEQ decays to 2.2266). It converts that preserved information into worse decisions only because it
parks the criterion at +0.697. Everything the anchor protects is recoverable by one scalar. This is a
coherent three-part story: criterion drifts while d′ does not (measurement) → a criterion anchor protects
d′ but mis-places the threshold (mechanism) → one label-free scalar unlocks it (method).

**What this is NOT, stated plainly.**
1. **Analytic, not measured.** Φ(d′/2) is a model-based ceiling. Equal variance cannot be tested from a
   single operating point per cell, and unequal variance would move it. **Required before any claim:**
   sweep the decision threshold on real per-item POPE logits and read the empirical best balanced accuracy.
   The four running method-arm evaluations dump per-item POPE logits, so this check arrives on its own.
   **Until it does, the table above is a projection, not a result.**
2. **An ORACLE correction.** It is the best achievable scalar, not one we can currently obtain. A
   label-free estimator will do worse, and how much worse is exactly the open question. The
   polarity-antisymmetry diagnostic (job 19494) tests whether such an estimator exists on this backbone;
   its pre-registered FAIL condition is a real possibility, not a formality.
3. **It does not rescue the generative axis.** The coverage/F1 claim remains withdrawn (null at 5/9).
4. It also raises the honest question the paper must answer: **if the whole effect is one scalar, why
   train at all rather than correct post-hoc?** `analysis/pcr_transfer.py` is the pre-registered test, and
   a defensible answer may be that post-hoc correction IS the right deliverable, with the anchor's role
   being to protect d′ so that the correction has more to recover (+0.054 vs SEQ's +0.002). That framing
   is permitted only if the empirical threshold sweep confirms the ceiling.

### Third sweep: positional hazard PREEMPTED 4×, and CHAIR-precision-only is THRONE (CVPR 2024) (2026-09-09)

All read from arXiv HTML or LaTeX e-print source, zero PDF extraction, positive control on every document.

**(a) Positional hallucination hazard — PREEMPTED, repeatedly.** Retract it entirely as a finding.
- **Deng, Chen, Hooi, CLIP-Guided Decoding (2402.15300v2, ICLR 2024 R2-FM workshop)** has our instrument
  including the anti-selection guard: it defines the risk set 𝒴ᵢ = responses of length ≥ i and reports
  "sentences generated in the later part are more prone to hallucination, with surprisingly consistent
  increasing pattern across multiple LVLMs", on LLaVA-1.5 among others. It goes FURTHER: an R_first(i)
  variant isolating first-time hallucinations shows "positional bias is not exclusively a result of error
  propagation."
- **LURE (2310.00754v2, ICLR 2024)** §2.3 is titled "Object Position in Generated Descriptions", sells it
  in the abstract, and already ran our length control (Appendix C.1.1): "high-density areas of
  hallucinatory objects predominantly appear towards the end of the sequence, regardless of the length of
  the descriptions."
- **OPERA (2311.17911v3, CVPR 2024)** computes CHAIR per positional split — an object-level rate
  stratified by position, the same quantity we computed.
- **Rohrbach 2018** has it in the metric's founding paper (position 6 of 9).
- **FActScore (2305.14251v2, EMNLP 2023)** is the text-only analogue: "the later part of the generation
  has significantly worse precision."
- **Caution for our own numbers:** SENTINEL (2507.12455v3) reports that in the last ~10% of tokens both
  hallucinated AND real objects decrease, so extending past mention #5 should expect non-monotonicity. Our
  curve stops at #5 and must not be extrapolated.

**(b) CHAIR-is-precision-only — PREEMPTED by THRONE (2405.05256v2, CVPR 2024)**, which states it directly:
"by lacking recall measurements, CHAIR_i may assign high scores to short and incomplete captions which are
not comprehensive in detailing the image." The EOS paper (2402.14545v2, ACL 2024) additionally runs
truncation and length-penalty BASELINES precisely because shortening lowers CHAIR, and reports they
"effectively reduce hallucinations at the cost of Recall." **Our precision/recall decomposition is a
replication, not a discovery** — independent of whatever arXiv 2609.01888 turns out to say.

**(c) Cross-model hazard slope — PARTIALLY PREEMPTED.** LeHaCE already makes the slope a cross-model
metric ("the curve slope as an innovative hallucination evaluation metric"). Deng frames the cross-model
finding as SAMENESS ("surprisingly consistent"), and 2607.18292v3 reports the OPPOSITE sign on an adjacent
axis (degradation grows with scale). Our n=2 backbone comparison cannot carry this regardless. **Drop it.**

**BINDING WORDING RULES.** The paper may NOT contain "we identify a positional hallucination hazard" or
"we show CHAIR is confounded by caption length" or "we show CHAIR ignores recall". Each is prior art at a
top venue. These analyses stay ONLY as a control/diagnostic for the continual-learning result, cited to
LURE, Deng, OPERA, POPE, LeHaCE, THRONE and AMBER. `analysis/diag_length_chair.md` already frames it
correctly ("a property of the model family, not something sequential tuning creates") — match that
framing everywhere.

**SERIOUS CITATION GAP (actionable now).** LURE, OPERA, POPE, Deng, THRONE, LeHaCE and AMBER are cited
NOWHERE in `related_work.md` or `paper/main.tex`. `related_work.md` was scoped to the CL×hallucination
axis and never swept hallucination metrology. A submission missing POPE and LURE in this area would be
desk-rejected by any informed reviewer. Being fixed in this session.

**NO-DATA on this sweep, recorded so it is not read as clearance:** WebSearch was exhausted session-wide
before the lane began, so discovery ran on the arXiv API, which is METADATA-ONLY — an arXiv-API zero here
is not full-text clearance (demonstrated: a query returned 0 for a phrase present in Deng's body text).
Pre-2022 captioning and non-arXiv venues are thinly covered. No published per-index NUMBERS were
extractable (all figures), so whether an existing curve matches our 0.125→0.442 is unknown.

### RESOLUTION: the confound is OLD in metrology but UNEXAMINED in continual learning (2026-09-09)

A fourth sweep full-text-grepped **42 papers** covering the CL / sequential-instruction-tuning × MLLM
literature (2023–2026) plus its hallucination intersection, all via arXiv HTML with a per-document positive
control, no PDF extraction. Corpus-wide result:

- `AMBER` as a metric: **0/42**. `Object HalBench` as a metric: **0/42**. `CHAIR` as a metric: **1/42**.
- `verbosity`: **1/42**. `confound` co-occurring with length/tokens/words: **0/42**.
- Sign-flip vocabulary co-occurring with length vocabulary: **0 matches across the entire corpus**.
- **No CL-MLLM paper matches a word budget, length-matches arms, reports words-per-caption across
  checkpoints, or reports any conclusion inverting under length control.**

**This settles the positioning.** The confound is established in the METROLOGY literature (POPE, LeHaCE,
THRONE, CCEval, OPERA — see `related_work.md` §9) and must be cited as such. But it has **never been
applied in continual learning**, where output length demonstrably changes across the stream — our SEQ arm
goes from the base's 81 words to 21. So the contribution is: *import a known measurement problem into a
literature that has never applied it, and show it changes that literature's conclusions.* That is
defensible, honest, and does not require claiming the confound.

**Two concrete demonstration targets, both published, both uncontrolled:**
1. **LLaCA (arXiv 2410.10868v5, ICML 2025 — venue verified via OpenReview camera-ready).** In the SAME
   appendix section it claims its method "can spontaneously suppress the occurrence of hallucinations in
   the continual instruction tuning" AND separately observes "the generated answers of our method are more
   concise and to the point." It reports reduced hallucination and shorter outputs side by side and never
   connects them. **This is our confound, live and unnoticed, in a published ICML paper.**
2. **DSCA (arXiv 2604.07965v1, arXiv-only).** The only sequential-adaptation paper using a generative
   object-hallucination metric: CHAIR-H 15.9 vs LiveEdit's 21.1 over 1,000 sequential edits. Its Table 5
   carries a "Richness ≈" column that is **never defined anywhere in the paper**, and the paper contains
   zero occurrences of `length`, `word` or `verbos` (controls on the same file: "the" 380, "editing" 48).
   An uncontrolled raw-CHAIR claim with a gesture at richness and no actual control.

**Version-watch:** MLLM-CTBench (2508.08275v3, arXiv-only, "under review") is the only CL paper naming
verbosity drift across checkpoints as a scoring problem — and that sentence is **new in v3** (v1 has zero
occurrences). It is framed as LLM-judge calibration, not as a metric confound, and it has no hallucination
metric at all. Closest mover on this axis; re-check before submission.

**Also useful:** "When More Words Say Less: Decoupling Length and Specificity in Image Description
Evaluation" (2601.04609v2) controls length in VLM description evaluation and finds "controlling for length
alone cannot account for differences in specificity" — a caution that length control is necessary but not
sufficient, which our own k-sweep independently showed (matching at k=60 left the arms 1.6× apart).

**Caveats recorded, not clearance:** the arXiv API is metadata-only, so a paper discussing length only in
body text would not surface; the 42-paper full-text sweep is broad but not exhaustive. DBLP is behind an
anti-bot wall in this environment and Semantic Scholar 429s, so four "arXiv-only" labels mean "no venue
evidence found", NOT verified-unpublished.

### THE SURVIVING METROLOGY CLAIM, sharply: CHAIR_i is assumed length-robust, and is not (2026-09-09)

Two further sweeps (ACL Anthology 39,091 records across 17 volumes 2024–2026; CVF 13,739 titles across
CVPR 2024/25/26, ICCV 2025, WACV 2024/25/26; plus ~102 OpenReview queries) found no paper claiming that
matching on mention opportunity inverts a VLM hallucination comparison. More usefully, they found the
specific published ASSUMPTION our data falsifies:

- **Geigle, Timofte, Glavaš, "Does Object Grounding Really Reduce Hallucination …?" (EMNLP 2024 main,
  2024.emnlp-main.159, arXiv 2406.14492v1)** states its metric choice outright: "CHAIR_s is less than
  ideal for longer captions as they are more likely to contain at least one hallucination … **Because of
  this, we adopt only CHAIR_i in this work.**" They adopt CHAIR_i *precisely because* they take it to be
  the length-robust variant.
- **DSCC (arXiv 2608.12746v3)** asserts the same in general terms: "Only the third kind is genuinely
  independent of object density: **precision per mention, 1 − CHAIR_I, normalises by the total number of
  mentions.**"

**Our data falsify that assumption.** Because mentions are NOT exchangeable — the positional hazard runs
0.125 → 0.442 from mention #1 to #5 on a fixed caption set — normalizing by mention count does not remove
the length dependence. On the non-COCO eval, CHAIR_i moves with the budget in every arm (SEQ 0.2605 →
0.1771, anchor 0.3276 → 0.1747 from k=∞ to k=12), and the anchor-vs-SEQ contrast inverts from +0.045
[+0.020, +0.071] to −0.002 [−0.018, +0.013].

**This is the claim to make, and it is much narrower than what I wrote this morning.** Not "CHAIR is
confounded by length" (POPE/LeHaCE/THRONE own that), but: *the per-mention normalization that the field
relies on to make CHAIR_i length-robust does not work, because the hazard is positional; therefore
CHAIR_i comparisons across arms of differing verbosity require explicit mention matching.* It has two
named targets that state the assumption in print, a mechanism, and a measured inversion with intervals.

**One more near-miss to cite and differentiate (a reviewer will raise it):** "Measuring the Measurers"
(arXiv 2406.17115v3, retitled from "Evaluating the Quality of Hallucination Benchmarks"; **REJECTED at
ICLR 2025** per OpenReview forum kjVgyR3RFr; no other venue found) contains a genuine length-driven
ranking reversal in its Table III (AMBER-g: InstructBLIP 0.151@104.7 words → 0.037@10.4; Otter
0.102@47.2 → 0.128@63.5, so which model looks better flips). But it induces length change by PARAPHRASING
THE PROMPT and reads the result as benchmark unreliability, proposing a new benchmark. It never matches
length, never treats matching as an estimator, reports no intervals, and is base-vs-base rather than
base-vs-tuned. **Version-watch it.**

**Unresolved lead, chase before submission:** a search snippet referenced an ICLR 2026 ACCEPTED paper
(PDF hash a2b956bd4931ec122228eb0baf0375cefdbf2bbd) claiming paraphrased instructions produced "doubled
CHAIR metric values, and the performance order of some models changed". The PDF is CAPTCHA-gated and the
hash matched none of 587 hallucination-related notes via the API. **Unverified search-engine text, not a
finding** — but it must be identified before submission.

**Residual gap, recorded so this is not read as clearance:** none of these sweeps had a full-text keyword
index over ACL/CVF PDFs (WebSearch exhausted; OpenReview indexes title+abstract only; OpenReview PDFs are
bot-gated). A length-matched control living in an unrelated paper's appendix would not have been caught.
Eight most-likely papers were full-text read to mitigate this. It remains a real gap.

### SELF-CORRECTION: the positional hazard is NOT monotone (2026-09-09)

Earlier today I wrote that the hallucination rate "rises monotonically with mention position". A sweep
flagged that no published curve is monotone — HaloProbe's fraction plateaus flat from ~position 90 to 160,
M3ID's histogram dips mid-caption, CGD's curve dips at index 2 — and invoked our own unrun-experiment
rule: a ready-to-run falsification blocks the claim rather than sitting in a limitations section.

Ran it. Object HalBench, fixed caption set (≥10 mentions, identical captions at every index):

| arm | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 |
|---|---|---|---|---|---|---|---|---|---|---|
| llava15 base (n=79) | 0.025 | 0.063 | 0.101 | **0.076** | 0.101 | 0.165 | 0.228 | 0.316 | **0.291** | 0.354 |
| qwen25vl base (n=101) | 0.020 | 0.040 | 0.059 | **0.050** | 0.069 | **0.030** | 0.089 | **0.069** | 0.099 | 0.149 |

**Monotonicity is FALSE.** LLaVA dips at #4 and #9; Qwen dips at #4, #6 and #8. The earlier appearance of
monotonicity was an artifact of stopping at index 5 on a single eval. **The claim is corrected to: the
hazard has a strong INCREASING TREND (LLaVA 0.025 → 0.354, ~14× across ten mentions) but is not monotone.**
Any wording implying monotonicity is withdrawn.

The anchor row (n=23) and especially SEQ (n=1 at this threshold) are far too small to read and are
excluded. Note also that the absolute hazard on Object HalBench (0.025→0.354) sits well below the non-COCO
eval's (0.125→0.442), so the level is benchmark-specific and only the trend transfers.

This does not affect the analyses that depend on the hazard, which need only non-exchangeability of
mentions, not monotonicity. It does affect how it may be described.

### Positional-hazard measurement: FOUR more preemptors, with the exact quantity plotted

- **SumGD (2410.13321v3, Findings of NAACL 2025** — ACL Anthology verified): Fig 3(b) plots "Hallucination
  Ratio" against "Object Position" with a fitted linear trend. Caption: "Object hallucination ratio at
  each generated token position."
- **OPERA (2311.17911v3, CVPR 2024)**: Fig 4(b)(c) plots CHAIR_s and **CHAIR_i** against "Split of
  Generated Text" for four LVLMs — a per-object-mention rate by position, the closest match to ours.
  *Citation trap:* OPERA's text SWAPS the CHAIR_s / CHAIR_i definitions relative to Rohrbach and to its own
  results table. Do not copy its equations.
- **CGD (2402.15300v2, venue unresolved)**: defines the rate conditional on "responses with at least i
  sentences" — our risk-set guard — and reports "surprisingly consistent increasing pattern across
  multiple LVLMs", plus a first-time-hallucination variant ruling out error propagation.
- **HaloProbe (2604.06165v2, arXiv-only; the "ICML" string in it is a LaTeX template keyword, NOT an
  acceptance)**: Fig 5 plots the proportion of correct vs hallucinated object tokens by token position on
  5K COCO samples — literally P(hallucinated | mention at position t).

**What survives is one methodological sliver:** every one of these lets the caption set shrink as the
index grows; none fixes the caption population. Our fixed-set estimate removes that selection effect. That
is a corrective note on a published phenomenon, not a finding, and the honest version of the note now has
to report non-monotonicity and saturation rather than a clean rise.

## RETRACTION (2026-09-09, same day): the scalar-correction ceiling REVERSES under the correct variance model

Earlier today I recorded, and put into the paper, that the anchor's oracle-corrected balanced accuracy
would reach 0.8794 — above SEQ (0.8672) and above the JOINT bound (0.8702), in 9/9 paired cells. That
rested on Φ(d′/2), i.e. **equal-variance** Gaussian SDT, which I flagged as "internally consistent, no new
assumption" because it is the model that defines the c and d′ we already report. **I then tested it, and
it is false.**

**The test.** Each cell's six stages give six (H, FA) operating points. Fitting z(H) = a + b·z(FA) across
them estimates the variance ratio b = σ_noise/σ_signal; equal variance means b = 1.

| arm | n | z-ROC slope b | bal acc now | EQUAL-var ceiling | UNEQUAL-var ceiling |
|---|---|---|---|---|---|
| SEQ | 9 | 0.715 | 0.8652 | 0.8672 | 0.8664 |
| anchor | 9 | **0.546** | 0.8255 | **0.8794** | **0.8427** |
| JOINT | 2 | 0.669 | 0.8692 | 0.8702 | 0.8717 |

Mean b across all 20 cells = 0.634 (sd 0.163). Every arm violates equal variance, and the **anchor
violates it most**. Recomputing the ceiling properly (noise ~ N(0,1), signal ~ N(a/b, 1/b²), maximizing
balanced accuracy over the criterion numerically) gives:

**anchor > SEQ in 0/9 cells, every difference negative (−0.0026 … −0.0462).** The equal-variance model
said 9/9 in favour of the anchor. The correct model says 0/9 against it. **The assumption was doing all
the work**, and the direction of the entire method story flipped with it.

**Why it flips.** Single-point d′ overstates separability when the signal distribution is wider than the
noise distribution, and it overstates it most where b is smallest — which is the anchor (0.546). The
anchor's apparent d′ advantage (2.345 vs SEQ's 2.227) is largely an artifact of reading a two-parameter
geometry off one operating point.

**Actions taken.**
1. `paper/main.tex` abstract and contributions edited in the same session that introduced the claim; the
   0.879 figure and the "9/9 paired" wording are removed and replaced with the reversal and an explicit
   statement that the recoverable headroom is UNRESOLVED. **No recoverable-headroom number is reported.**
2. The prereg entry "THE METHOD STORY: the anchor … is one scalar away from beating JOINT" is **RETRACTED**
   in full. It may not be cited, quoted or built on.

**What still stands, unaffected:** the endpoint deficit itself (POPE F1 0.7963 vs 0.8625, 9/9, no overlap)
is a direct measurement and does not depend on any SDT model. The anchor is worse than doing nothing at
the endpoint either way; what changed is the claim that a scalar would more than repair it.

**Limits of this test, stated so it is not over-read in the other direction.** The six operating points
come from six TRAINING STAGES, which differ in d′ as well as in criterion, so this is not a clean z-ROC
(those need one model rated at several criteria). The slope estimate is therefore confounded and b = 0.634
should not be quoted as a measured variance ratio. What the test does establish is that **equal variance
cannot be assumed**, which is enough to invalidate the ceiling either way. **The decisive experiment is an
empirical threshold sweep on per-item POPE logits** — no distributional assumption at all — and the four
running method-arm evaluations dump exactly those. Nothing about recoverable headroom, in either
direction, may be written until that sweep runs.

### DECISIVE: empirical threshold sweep confirms the retraction, on no assumption at all (2026-09-09)

`analysis/threshold_sweep.py` reads the per-item POPE logit dumps (`pope_logits.jsonl`, 9000 rows/cell,
carrying z_yes, z_no and gt) and sweeps the decision threshold exactly — sorting once and evaluating every
reachable partition, so the maximum is exact, not grid-approximated. No Gaussians, no variance ratio, no
d′. `bal@0` is the checkpoint's own operating point; `ceiling` is the best ANY scalar criterion correction
could reach; `t*` is the size of the correction required.

| cell | bal@0 | ceiling | gain | t* |
|---|---|---|---|---|
| psL_seq k1 | 0.8670 | **0.8747** | 0.0077 | −0.81 |
| psL_seq k2 | 0.8201 | **0.8724** | 0.0523 | +1.53 |
| psL_seq k3 | 0.8698 | **0.8728** | 0.0030 | +0.19 |
| psL_anchor k1 | 0.8051 | 0.8702 | 0.0651 | **−6.56** |
| psL_anchor k2 | 0.8421 | 0.8614 | 0.0193 | **−3.41** |
| psL_anchor k3 | 0.8212 | 0.8547 | 0.0334 | **−4.41** |
| psL_anchor k4 | 0.8218 | 0.8628 | 0.0410 | **−4.08** |
| mth_critp k1 | 0.8619 | 0.8663 | 0.0044 | −0.56 |
| mth_critp k2 | 0.8679 | 0.8717 | 0.0038 | +0.53 |
| mth_critp1 k1 | 0.8417 | 0.8566 | 0.0149 | −0.94 |
| mth_critp1 k2 | 0.8563 | 0.8597 | 0.0033 | +0.41 |
| mth_critp1 k3 | 0.8211 | 0.8407 | 0.0196 | −1.09 |

**Two findings, both assumption-free.**

1. **The mis-threshold diagnosis is CONFIRMED and is large.** The anchor needs a correction of −3.4 to
   −6.6 logits; sequential and both critp arms need |t*| < 1.6. The anchor is displaced ~4× further than
   any other arm, in the conservative direction, exactly as the criterion measurement says.
2. **But correcting it does NOT make the anchor competitive.** At every matched stage sequential's ceiling
   exceeds the anchor's, and the gap GROWS along the sequence: k1 0.8747 vs 0.8702 (+0.0045), k2 0.8724 vs
   0.8614 (+0.0110), k3 0.8728 vs 0.8547 (+0.0181). So the anchor does not merely sit at the wrong
   threshold on equally good evidence — **its evidence is genuinely less separable**, and increasingly so.

**This settles the question the two analytic routes disagreed about.** Equal-variance SDT predicted the
anchor's corrected ceiling would beat sequential's in 9/9 cells; unequal-variance predicted 0/9. The
empirical sweep, which assumes nothing, agrees with the unequal-variance prediction. **The retraction was
correct, and the "anchor is one scalar from beating JOINT" story is dead on assumption-free grounds.**
Single-point d′ was misleading here precisely because it reads a two-parameter geometry off one point.

**Scope and the control still owed.**
- These are the pilot-suite same-path cells (psL/mth, 4 stages, one ordering/seed), not the 9-cell UCIT
  matrix. The direction is consistent with the UCIT unequal-variance analysis but the full-study version
  needs the fsL logit dumps.
- Cells are at different completion depths (seq has k1–k3, anchor k1–k4), so only matched-k comparisons
  above are used; the unmatched k4 row is reported but not compared.
- **Positive control still owed:** H@0/FA@0 are internally consistent (bal@0 = ½(H + 1 − FA) reproduces
  exactly), but they have NOT yet been checked against the generation-scored POPE H/FA for the same cell.
  If the logit readout and the generated answer disagree, this sweep measures a different decision rule
  than the paper's c and d′. Run that check before the numbers enter the paper.

**Consequence for the method program.** The target is unchanged — land at the correct criterion,
label-free — but the anchor is now known to be the wrong vehicle: it degrades separability while
displacing the threshold. A corrected-target method must be built so that it does NOT inherit that, which
is an argument for the antisymmetry term (`pilot/method/crit_antisym.py`, gated on job 19494) over any
further variation on base-anchoring.

## ⚠ CRITICAL INSTRUMENT + METHOD BUG: answer-token casing drifts by stage (2026-09-09)

Found while discharging the positive control owed by the threshold sweep. The control FAILED, and the
cause is worse than a scoring mismatch.

**The observation.** `pilot/method/faith_loss.py::resolve_answer_token_ids` resolves the answer tokens as
`" Yes"` / `" No"` (capitalized) once, in context, at construction. But the model's ACTUAL emitted token
alternates by stage, identically in every arm:

| stage | emitted tokens | `argmax_is` | instrument correct? |
|---|---|---|---|
| k1 | `Yes` / `No` | yes/no | ✔ |
| **k2** | **`yes` / `no`** | **`other` on 100% of rows** | ✘ |
| k3 | `Yes` / `No` | yes/no | ✔ |
| **k4** | **`yes` / `no`** | **`other` on 100% of rows** | ✘ |

Verified across `psL_seq`, `psL_anchor`, `mth_critp`, `mth_critp1` — every arm, same pattern. The pilot
sequence is ScienceQA → TextVQA → Flickr30k → VizWiz, so the casing tracks the answer format of the task
most recently trained. **This is task-recency drift in the answer token itself**, the same recency the
paper already reports on the criterion and generative axes, now visible in the vocabulary.

**Consequence 1 — the logit instrument is wrong at alternating stages.** At k2/k4 the dumps record
z(" Yes") and z(" No") while the model is deciding between "yes" and "no". The proxy g = z_yes − z_no
agrees with the realized argmax only **90.7%** of the time, and generation-scored vs logit-scored rates
diverge badly and UNEQUALLY across arms (psL_seq k2: FA 0.150 generated vs 0.284 from logits, Δ = +0.134;
mth_critp k2: ΔH = +0.064, ΔFA = +0.064; psL_anchor k2: ΔH = +0.020 only).

**Consequence 2 — the METHOD regularizes the wrong vocabulary rows at those stages.** `FaithfulnessAnchor`
and `CriterionPreserver` both call the same resolver and both pin a population moment of
g = z(" Yes") − z(" No"). At any stage where the model has migrated to lowercase, the regularizer is
constraining a statistic that has decoupled from the decision the model actually makes. **This is a
candidate mechanistic explanation for the anchor's conservative displacement (c = +0.697) and for its
degraded separability in the threshold sweep** — it is pinning a shadow of the decision variable, not the
decision variable. Note the pilot ENDPOINT (k4) is a lowercase stage, so the anchor's reported endpoint
criterion is measured at a stage where its own regularizer was mis-targeted.

**What is NOT affected.** The paper's headline c, d′, F1 and yes-rate come from GENERATION scoring
(`metrics_pope.py` on `pope_gen.jsonl`), which parses the answer text case-insensitively — `parse_fail = 0`
and `n_parsed = 9000` in every cell. **The measurement contribution stands.** The full-study fsL cells have
no logit dumps at all, so nothing in the 9-cell matrix is computed from the broken statistic.

**What IS affected and must be re-examined before use:** `analysis/threshold_sweep.py` (run today),
`analysis/blind_prior_share.py`, `analysis/logit_bias_analysis.py`, `analysis/pcr_transfer.py`,
`analysis/theory_sanitycheck.py`, and the ρ_k / probe-dump analyses — every one consumes these dumps.

**DOWNGRADE of today's threshold-sweep entry.** The entry above headed "DECISIVE: empirical threshold
sweep …" is **NOT decisive** and is downgraded to provisional. Its ceiling comparison is invariant to a
constant per-arm offset (a max over thresholds absorbs any shift), which protects it somewhat, but the
readout is demonstrably not the realized decision rule and its error differs per arm, so the anchor-vs-SEQ
ceiling gap cannot be trusted at face value. **The qualitative conclusion that the equal-variance ceiling
was unsafe still stands** — that rests on the z-ROC slopes, computed from generation-scored H/FA, not from
these dumps.

**THE FIX (to implement, not yet done).** Resolve the answer tokens as SETS rather than single ids and
pool them:  g = logsumexp{z(t) : t ∈ yes-variants} − logsumexp{z(t) : t ∈ no-variants}, with variants
covering casing and leading-space forms. This is casing-robust, keeps the statistic differentiable for the
training loss, and reduces to the current definition when only one variant carries mass. The dumps must
also record the realized argmax id (they already do) so this class of failure is detectable in future
without a separate investigation. Until the fix lands, **no new claim may rest on the logit dumps.**

### SCOPING the casing bug: the FULL STUDY is clean, the METHOD ARMS are not (2026-09-09)

I overstated the reach of the casing bug when I recorded it. Checked directly by reading the generated
answer's first token across every full-study cell and stage (`pope_gen.jsonl`, 2000 rows sampled per cell):

**UCIT full study: 0.0% lowercase in EVERY cell and EVERY stage.** All 54+ cells across anchor / seq / ewc,
three orderings x three seeds x six stages, emit `Yes` / `No` capitalized. Occasional stray first tokens
appear (`Hotdog`, `Hen`, `Seagull`, 1–4 rows per cell) but no casing drift at all.

**Consequences, correcting the earlier entry:**
1. **The full-study 9-cell matrix is UNAFFECTED.** The token resolver was correct at every UCIT stage, so
   the criterion regularizer pinned the right vocabulary rows throughout. **The anchor's conservative
   displacement (c = +0.697) is NOT explained by the casing bug** — I floated that explanation when I
   recorded the bug and it does not apply here. Withdraw it for the full study.
2. **The PILOT-SUITE arms are affected**, and that matters more than it first appears: the pilot sequence
   is ScienceQA → TextVQA → Flickr30k → VizWiz, and TextVQA and VizWiz have lowercase gold answers, which
   is what pulls the emitted casing down at stages 2 and 4. **Every method arm runs on the pilot suite**
   (`psL_seq`, `psL_anchor`, `mth_critp`, `mth_critp1`, and all queued candidates carry `|pilot|` in the
   queue). So the entire method-comparison substrate sits on the affected sequence.
3. Therefore the criterion regularizer in every method arm is mis-targeted at 2 of 4 stages, INCLUDING the
   endpoint (k4 = VizWiz, lowercase). The method verdicts that `analysis/method_scorecard.py` and
   `analysis/pcr_transfer.py` will produce are computed on that substrate.
4. Today's threshold sweep ran on psL/mth cells and is therefore inside the affected set — consistent with
   the downgrade already recorded.

**This may be good news for the method rather than bad.** The candidates have been regularizing a
statistic that decouples from the decision at half the pilot stages, including the one the endpoint is
read from. If a candidate looks weak on the current substrate, that result is confounded with the bug and
does not licence abandoning the candidate. **Pre-registered position: no method candidate may be declared
FAILED on pilot-suite evidence collected under the single-id statistic.** Re-running the method arms under
the pooled statistic (`pilot/method/answer_tokens.py`) is now a priority, and is the cleanest test of
whether the criterion family works at all.

**Open question worth one cheap check:** UCIT also contains VizWiz, yet shows no lowercase drift. Either
the UCIT VizWiz variant formats answers differently or the six-task mix dilutes the effect. Worth
resolving, because it decides whether the pilot-vs-UCIT difference is about the task or about the
sequence, and that bears on the task-recency account the paper already advances.

## The drift endpoint measures stability of a mis-placed threshold (2026-09-10)

Raised by Alex reading the trajectory figure: the anchor sits at c ≈ 0.7 while sequential and JOINT sit
near the optimum, so is the anchor simply the worst arm? Checked across all nine cells. **Yes.**

Distance from the optimum, |c|, over the six stages of each cell:

| arm | n | mean \|c\| | best stage | worst stage |
|---|---|---|---|---|
| sequential | 9 | 0.196 | 0.034 | 0.341 |
| **anchor (ours)** | 9 | **0.731** | **0.647** | **0.823** |
| JOINT bound | 2 | 0.099 | 0.039 | 0.160 |

**Paired head-to-head: sequential's WORST stage is better placed than the anchor's BEST stage in 9/9
cells** (seq worst 0.259–0.408 against anchor best 0.541–0.696). No overlap, no exception. So on endpoint,
mean, best-case AND worst-case placement, the anchor is worse than running no method.

**This is a problem with the pre-registered ENDPOINT, not only with the method.** Σ|Δc| measures
step-to-step change. The anchor wins it decisively (0.364 vs 0.781) by holding a threshold that is
uniformly mis-placed. We never established that low drift is worth wanting independently of placement,
and this result is direct evidence that it is not: a practitioner offered "wanders around 0.20" versus
"stable at 0.73" takes the wanderer, because every checkpoint of it is better calibrated than every
checkpoint of the alternative.

**Consequences.**
1. The headline "closes 75% of the SEQ→JOINT drift gap" is true and, on its own, misleading. It may not
   appear without the placement table above in the same view. The earlier framing of this as "a 6.6 F1
   cost at the endpoint" understated it — the deficit is not endpoint-specific, it holds at every stage.
2. **Σ|Δc| should not be the paper's primary endpoint.** A defensible primary is placement-based — mean
   |c| or endpoint |c| against the JOINT bound — with Σ|Δc| reported as a secondary describing the path.
   Changing a pre-registered endpoint after seeing results is exactly what pre-registration forbids, so
   the honest move is to report BOTH, state that the pre-registered one was the wrong choice, and say why.
   That is a limitation, not a silent substitution.
3. The anchor's status narrows again: it demonstrates the criterion is *controllable*, and simultaneously
   demonstrates that controlling it toward the base's value is harmful. Both halves are informative and
   neither is a method.
4. This strengthens the case for the balanced-probe target (NEXT_EXPERIMENTS item 1): the failure is
   entirely in the target, and a target of yes-rate = 0.5 is both correct and label-free.

**What is unaffected:** the measurement itself. "The criterion moves while d′ does not" is a property of
sequential fine-tuning, established on the SEQ and JOINT arms, and does not depend on the anchor at all.

## OUT-OF-SAMPLE TEST PASSES: criterion drift is task-driven, not depth-driven (2026-09-11)

The recency account was fitted post-hoc on the observations it explains, and the pre-registration recorded
an out-of-sample test as OWED. **The three orderings are that test and it cost nothing to run.** Each UCIT
task sits at a different position in each ordering, so task identity and sequence depth are CROSSED rather
than confounded. `analysis/order_effects.py` decomposes the one-step criterion change
dc_k = c_k − c_{k−1} by each factor, with a 20,000-shuffle permutation null (both factors have six levels,
so the η² comparison is like-for-like, and the permutation removes the level-count bias regardless).

Post-settling (stage 1 dropped, matching the paper's own endpoint convention — the anchor's one-time
settling jump of +0.358 otherwise loads onto "depth" and masquerades as a sequence effect):

| arm | η² task identity | η² sequence depth | ratio |
|---|---|---|---|
| sequential | **0.731** (p < 0.0001) | 0.049 (p = 0.73) | 15.0 |
| anchor | **0.352** (p = 0.0034) | 0.053 (p = 0.69) | 6.7 |

**Depth explains nothing in either arm.** The recency account is CONFIRMED out-of-sample and the prereg's
withdrawal condition does not fire. This is far stronger than the post-hoc η² 0.276 vs 0.134 the paper
currently reports on the generative axis, and it is on the criterion axis where the measurement lives.

**Second finding, and it is the first genuinely positive result for the anchor.** Per-task criterion pull,
post-settling mean dc when that task is trained:

| task | sequential | anchor | shrink |
|---|---|---|---|
| IconQA | −0.2511 | −0.0844 | 3.0× |
| ImageNet-R | −0.1839 | +0.0162 | 11.3× |
| Flickr30k | +0.1629 | +0.0449 | 3.6× |
| VizWiz | +0.1391 | −0.0591 | 2.4× |
| ArxivQA | −0.1092 | +0.0321 | 3.4× |
| CLEVR-Math | +0.0049 | −0.0458 | 0.1× |
| **mean \|dc\|** | **0.1418** | **0.0471** | **3.0×** |

**The anchor decouples the criterion from task identity**: η² for task falls 0.731 → 0.352 and the mean
per-task pull falls 3.0×. That is a specific mechanistic claim, much sharper than "it reduces drift", and
it is what the anchor actually does. It remains true that it does so at a mis-placed operating point
(mean |c| 0.731 vs sequential's 0.196), so the correct summary is: **the anchor removes the task-specific
component of criterion movement and adds a large constant offset.** Both halves belong in the paper.

**Caveats.** (i) CLEVR-Math inverts (0.1×) because its sequential pull is already ~0, so the ratio is
meaningless there; the mean is the number to quote, not the per-task ratios. (ii) η² is an in-sample
variance share; the permutation p is the inferential statistic. (iii) Tasks differ in dataset size and
answer format, which are not controlled here — this establishes THAT task identity predicts the shift, not
WHICH property of the task causes it. The natural follow-up is whether the pull is predicted by the task's
answer-format statistics, which the answer-statistics instrument can test on existing data.

**Paper consequence:** this replaces the post-hoc recency paragraph with a pre-registered, crossed,
out-of-sample result, and it supplies the anchor with a real mechanism. Both are upgrades.

## SELF-CORRECTION: the c-vs-d′ ratio I have been quoting mixed two estimators (2026-09-11)

Raised by the pre-submission novelty sweep and verified. **This is my error and it has propagated into
the slide deck, the explainer artifact and my verbal summaries, so it is recorded prominently.**

I have repeatedly written that "the criterion swings by 0.79 while d′ stays inside a band of 0.12". Those
two numbers are not the same kind of statistic. 0.781 is Σ|Δc|, a PATH LENGTH summed over stages. 0.12 is
a RANGE, and it is computed on the 9-cell AVERAGE per stage, so cross-cell variation is averaged away
before the range is taken. Pairing them is the estimator-width confound already on file in this project's
env notes, committed by me.

**Measured the same way on the same cells (sequential arm):**

| summary | c | d′ | ratio |
|---|---|---|---|
| path length Σ\|Δ\| | 0.936 | 0.380 | **2.46×** |
| within-cell range | 0.382 | 0.144 | **2.66×** |

(anchor: 2.27× by path, 1.34× by range.) **The defensible full-study figure is ~2.5×, not the near-total
separation I have been presenting.** Every future statement uses one estimator for both quantities.

**Consequence for the headline: "d′ never moves" is FALSE and must be withdrawn.** Sequential's endpoint
d′ falls by 0.121 with SE 0.0108 across nine cells — small, but ~11 SE from zero, so a real decline. The
correct claim is *relative*: the criterion moves about two and a half times as much as discriminability,
and the discriminability change is bounded.

**The bound is now positively established, which is an upgrade.** The pre-registered F1 falsifier was a
threshold ("max |Δd′| < 0.30"), i.e. only a failure-to-trip. A two-one-sided-tests equivalence test
against that same ±0.30 margin now PASSES for both arms:

| arm | mean Δd′ | SE | TOST t (lower / upper) | verdict |
|---|---|---|---|---|
| sequential | −0.1211 | 0.0108 | +16.55 / +38.98 | equivalent to 0 within ±0.30 |
| anchor | −0.0023 | 0.0267 | +11.14 / +11.32 | equivalent to 0 within ±0.30 |

So: d′ declines slightly and that decline is statistically equivalent to zero at the pre-registered margin,
while c's movement is not. That is the claim the data support, stated with both halves.

**A separate internal inconsistency in the paper, found while checking.** `main.tex:975-978` says d′
"stays within $[2.14, 2.43]$" — a range of 0.29 — and two lines later calls d′'s range 0.19. Both cannot
be right. The pilot paragraph's own ratio ("four times", 0.78 vs 0.19) is at least range-against-range and
so is internally like-for-like, unlike my full-study version; but the 0.19/0.29 discrepancy must be
resolved before submission, not papered over.

**Also raised by the sweep and NOT yet addressed — the sharpest reviewer objection we have.** Our z-ROC
slopes (0.55–0.72) say the equal-variance model is wrong. We use that to reverse the anchor's ceiling
verdict, but we rely on single-point d′ to carry the PRIMARY claim. A reviewer will ask why single-point
d′ is unreliable enough to overturn the method result and reliable enough to support the headline. An
independent report (arXiv 2603.14893) finds slopes 0.52–0.84 in LLMs and instruct models MORE extreme than
base, i.e. tuning may move the variance ratio itself — in which case d′ across stages is not a constant
quantity. This needs either d_a (the unequal-variance index) reported alongside d′, or an explicit
argument for why the primary claim survives. `main.tex` currently contains zero occurrences of d_a, UVSD
or z-ROC outside the anchor-ceiling passage.

## THE ASSUMPTION-FREE FORM OF THE PRIMARY CLAIM — and it is stronger (2026-09-11)

The pre-submission sweep raised the sharpest objection on file: we invoke unequal variance to reverse the
anchor's corrected-ceiling verdict while relying on single-point d′ to carry the primary claim, and an
independent report (arXiv 2603.14893) finds z-ROC slopes 0.52–0.84 in LLMs with instruct models *more*
extreme than base — so tuning may move the variance ratio itself, in which case d′ across stages is not a
constant quantity.

`analysis/zroc_coherence.py` sidesteps the argument entirely. "Only the criterion moves" has an exact
geometric meaning: the two evidence distributions are fixed and the threshold slides, so every stage of a
cell must be a different operating point **on the same ROC curve**. Fit z(H) = a + b·z(FA) across a cell's
six stages and read the residuals. No equal-variance assumption anywhere — b is estimated, not assumed.

| arm | n | mean R² | min R² | mean d_a | sd d_a |
|---|---|---|---|---|---|
| sequential | 9 | **0.9569** | 0.8435 | 2.1587 | 0.0401 |
| JOINT bound | 2 | 0.9446 | 0.9190 | 2.1812 | 0.0423 |
| **anchor** | 9 | **0.5857** | **0.0576** | **1.7884** | 0.3618 |

**Two results, and they point in opposite directions.**

1. **The primary claim SURVIVES and gets stronger.** A sequential cell's six stages lie on one z-ROC with
   R² = 0.957, and d_a is near-constant across cells (sd 0.040). That is exactly what "the distributions
   are fixed and only the criterion moves" means, established without assuming equal variance and without
   relying on single-point d′. **This should replace "d′ is flat" as the paper's primary statement** — it
   is the same claim, assumption-free, and it answers the reviewer objection in advance rather than
   defending against it. The JOINT bound behaves identically (R² 0.945), as it should.

2. **The anchor does NOT merely mis-place the threshold — it distorts the evidence geometry.** R² 0.586,
   one cell as low as 0.058, and d_a scatter 9× the sequential arm's. Its d_a is also genuinely lower
   (1.788 vs 2.159). So the anchor is not "the same model wearing a bad threshold": it degrades
   separability and adds an offset. This independently explains why its corrected ceiling sits below
   sequential's in the threshold sweep, and it is the third line of evidence for that conclusion (the
   first two being the unequal-variance ceiling recomputation and the empirical sweep).

**Caveats, stated before the claim hardens.** Six points per fit is modest. A high R² is CONSISTENT with
fixed distributions rather than proof of them — some other trajectory could in principle trace a line. A
low R², however, does reject fixed distributions for that arm, so the anchor result is the more secure of
the two directions. The fit also pools stages that differ in training data, which is the point, but it
means R² measures coherence of the whole trajectory rather than of any single transition.

**Action:** report d_a alongside d′ throughout, and lead the measurement section with the coherence result
rather than with the flatness of a single-point index. `main.tex` currently contains zero occurrences of
d_a, UVSD or z-ROC outside the anchor-ceiling passage.

## CONFOUND AUDIT PASSED: the criterion swing is not a parsing artifact (2026-09-11)

The non-arXiv proceedings sweep raised a genuine threat to the primary result, not a citation issue.
**SEFE (ICML 2025)** documents that multimodal continual instruction tuning causes *answer-style drift* —
"superficial forgetting", where the knowledge is intact and the output FORMAT deviates. POPE is scored by
string-matching "Yes"/"No". If the format drifted across our six-task stream, the parser could manufacture
a criterion swing out of nothing, and the reported shape (large c movement, small d′ movement) is exactly
what that artifact would look like. The sweep's wording was blunt and correct: a reviewer will kill the
paper on this if it is not addressed.

**Audited over every UCIT row we have: 1,026,000 generations, all cells, all six stages.**

| | |
|---|---|
| rows with a non-Yes/No first token | **677 of 1,026,000 = 0.066%** |
| by stage | 0.113%, 0.049%, 0.002%, 0.024%, 0.208%, 0.000% |
| worst single cell | fsL_seq_o1_s23_k5, 101/9000 = 1.12% |

**There is no drift and no trend.** The highest-rate stage is ImageNet-R (0.208%), which is where a
classification task would plausibly leak a class name, and it is still two parts in a thousand.

**The bound.** Take the worst case: every one of those 677 rows is silently mis-assigned to the same side.
The yes-rate then shifts by at most 0.00066, and near yes-rate 0.5 the criterion moves by
0.00066 / φ(0) = **0.0017 units**. The observed per-cell criterion range is **0.382 units** — a margin of
**231×**. A parser that fails on 0.066% of rows cannot produce this effect, and the margin is large enough
that the conclusion does not depend on the exact worst-case assumption.

`parse_fail = 0` and `n_parsed = 9000` in every cell, so no row was dropped either. **Audit passed. Report
the per-stage rates and this bound in the paper** — it is the cheapest possible answer to the objection and
it converts a paper-killer into a passed check.

### Also from the same sweep — three items that change what we may claim

1. **Claim 2 (the SDT decomposition itself) is PREEMPTED.** arXiv 2603.14893 does full parametric SDT on
   LLMs: unequal-variance fitting, criterion estimation, z-ROC, 168,000 trials. The move is not ours. It
   is text-only factual QA with temperature as the manipulation, no VLM and no trajectory — and notably
   its result *refutes* a clean dissociation (temperature moved both parameters) where we report one.
   Its OSF pre-registration (10.17605/osf.io/qpk9a) states the claim-1 sentence shape with temperature
   substituted for continual tuning.
2. **Claim 1's SHAPE is established in continual learning**, largely at non-arXiv venues: WACV 2025
   ("the primary contributor ... is the linear head"), TPAMI 2025 PASS++ ("representation bias and
   classifier bias"), **ICML 2025 SEFE** ("superficial" vs "essential" forgetting, in our exact MCIT
   setting), CoLLAs 2024. "Readout moves, representation doesn't" is known. **Our novelty must therefore
   rest explicitly on SDT as the instrument on a generative yes/no benchmark where there is no
   classification head to blame** — the paper must say this in those words.
3. **Terminology collision to pre-empt in one sentence:** "Criterion-Conditional In-Context Learning:
   Evaluating Criterion-Shift Adaptation in Vision-Language Models" (ICML 2026, accepted; arXiv
   2607.02575) ships metrics named *Criterion Invariance* and *Criterion Sensitivity* for VLMs. Their
   "criterion" is an in-context task rule, not SDT's c.

**Correction to an earlier note:** LeHaCE IS on proceedings.neurips.cc and was never invisible to a
proceedings sweep — only to arXiv. The §9 claim that it "was invisible to every sweep" overstates it.

**Process note:** a WebSearch backend fabricated a quote attributed to 2607.25196 ("yes-bias masked by
stable F1/Accuracy"); the string "yes-bias" occurs zero times in that paper. Do not cite that phrasing.

## MECHANISM CANDIDATE: answer FORMAT, not answer content, moves the criterion (2026-09-11)

The order-effects result established that criterion drift is task-driven (η² 0.731 vs 0.049 for depth).
The obvious next question is *which property of a task* does the pulling. Measured directly from the UCIT
training files:

| task | mean answer words | criterion pull (post-settling) |
|---|---|---|
| CLEVR-Math | 1.00 | +0.0049 |
| IconQA | 1.00 | −0.2511 |
| ImageNet-R | 1.28 | −0.1839 |
| ArxivQA | 1.58 | −0.1092 |
| VizWiz | 11.61 | +0.1391 |
| Flickr30k | 12.29 | +0.1629 |

**The single most important number here: the yes/no fraction of every task's training answers is 0.0000.**
Not one of the six tasks teaches the model to say "yes" or "no". Yet the yes/no criterion moves
systematically, and the direction is predicted by how LONG the task's answers are: the two long-answer
tasks push it conservative, the four short-answer tasks push it liberal or leave it alone.

**This unifies the pre-registered E1 failure rather than sitting beside it.** The paper reports, as a
failure, that a stream certified to contain no yes/no content still moves the criterion 0.40–0.55 units.
That was recorded as "answer statistics explain the direction of drift but not its magnitude". The result
above says why: **the criterion responds to answer FORMAT statistics, not to yes/no CONTENT.** A zero-dose
stream is zero-dose only in content; it still has a format. So E1's null side failed because the dose was
measured on the wrong variable.

**Stated honestly, because the design is weak.** The answer lengths are BIMODAL — four tasks at ~1 word
and two at ~12 — so this is a two-group contrast, not a dose-response, and reporting it as a correlation
(Pearson r = 0.866, exact permutation p = 0.0278 over all 720 pairings) overstates what the design
supports. The correct test is the two-group one:

- long-answer mean pull **+0.1510**, short-answer **−0.1348**, difference **+0.2858**
- exact permutation over all 15 two-vs-four splits: **p = 1/15 = 0.067**, one-sided

The observed split is the most extreme of the fifteen, so **0.067 is the smallest p this design can
produce**. The effect is maximal and the design is underpowered simultaneously. That is not a significant
result at 0.05 and must not be written as one.

**Pre-registered follow-up, before this is claimed:** the test needs tasks with INTERMEDIATE answer
lengths to become a dose-response rather than a two-group difference. Candidates already in the data
directory could be re-purposed, or the existing tasks truncated to controlled answer lengths, which would
also break the confound between answer length and task identity — currently the two are inseparable, so
"long-answer tasks" and "Flickr30k and VizWiz specifically" are the same hypothesis. Until that runs, this
is a mechanism CANDIDATE with a p of 0.067 and a known confound, not a finding.

## CORRECTIONS to my own 2026-09-11 entries, found while writing them up (2026-09-11, later)

Four of the entries above contain errors. All were caught by re-deriving every number from source rather
than transcribing, which is the only reason they surfaced. Corrected here; the original entries stand as
written so the record shows what was claimed when.

**1. "parse_fail = 0 and n_parsed = 9000 in every cell" is FALSE.** I asserted this in the parsing-audit
entry and in earlier summaries. In fact **23 of the 120 LLaVA/UCIT stage-cells have parse_fail between 1
and 101**, totalling **602 dropped rows**, with n_parsed ranging 8,899–9,000. Rows are *dropped*, not
merely mis-assigned, which is a different failure mode than the one I audited. The audit's conclusion
survives — see item 2 — but the premise as I stated it was wrong.

**2. The "231× margin" violates the like-for-like rule I had just written.** It divides a MATRIX-POOLED
anomaly rate (0.066%) into a PER-CELL criterion range (0.382). That is the same class of error as the
path-length-versus-range mistake corrected earlier the same day, committed again within hours. The
like-for-like bounds are:

| comparison | bound |
|---|---|
| worst affected cell (1.12%) → Δc ≤ 0.028, against that cell's own range 0.394 | **14×** |
| same bound against the SMALLEST per-cell range in the matrix (0.245) | **8.7×** |

**The audit still passes** — 8.7× is a comfortable margin — but the number to quote is 8.7–14×, not 231×.
231× may appear only in a footnote, labelled as not like-for-like.

**3. "The highest-rate stage is ImageNet-R" conflates position with task.** The per-stage rates are
indexed by POSITION, and ImageNet-R occupies position 5 under ordering 1, position 2 under o2 and
position 1 under o3 — which is exactly why those positions spike. Regrouped by the task actually trained:
**548 of 602 exclusions are ImageNet-R (0.338%), 45 are IconQA, and exactly 0 come from ArxivQA,
CLEVR-Math, Flickr30k or VizWiz.** This is stronger evidence than what I wrote, because it shows a task
effect rather than a drift with depth, and it removes an apparent depth trend that was an artifact of the
orderings.

**4. "Both factors have six levels, so the η² comparison is like-for-like" is wrong post-settling.**
Dropping stage 1 leaves POSITION with five levels and TASK with six. η² is biased upward by level count,
so the bias runs **toward the factor we claim wins**. The permutation test (fixed group sizes, relabelling)
is unaffected and carries the comparison; any raw η²-ratio must not be quoted as if the two were
comparable. The recency conclusion is unchanged — depth's permutation p is 0.73, nowhere near
significance — but the framing in the original entry overstated the symmetry.

**Three smaller ones.** (a) "Zero percent yes/no in every task" — VizWiz's *no*-fraction is 0.0005, not
exactly zero. (b) The anchor's "6.6 F1 points worse than running no method" is versus SEQUENTIAL; versus
the frozen base it is 4.9 points. Both belong in the text. (c) The 2.46× path-length row is the RAW path
(base → stage 6, c 0.936), not the post-settling endpoint 0.781 — stated together they read as a
contradiction unless the caption says which is which. (d) The 677-row and 602-row counts come from
different cell sets (19 vs 20 cells) and must not be presented as the same number.

**One substantive claim superseded.** "d′ is flat, so the entire endpoint F1 deficit is criterion
placement" no longer holds: the z-ROC entry gives the anchor d_a = 1.788 against sequential's 2.159, so
the anchor **does** lose separability. The deficit is placement *plus* a genuine loss of separability, and
the paper must say so rather than attributing it wholly to the threshold.

**Process note.** Every one of these came from a re-derivation pass, not from review of the prose. The
lesson is the one already on file: a number is not checked until it has been recomputed from the readout
in the same estimator as the thing it is compared against.

## GENERALIZATION WITHIN POPE: a double dissociation across negative-sampling regimes (2026-09-11)

Every measurement in the paper comes from POPE, so the obvious objection is that the criterion account is
an artifact of one benchmark or of one way of choosing negatives. The second half of that is answerable
for free, because POPE is three benchmarks: `random`, `popular` and `adversarial` differ ONLY in how the
absent object is drawn — uniformly, from the most frequent classes, or from those that most often co-occur
with what is actually in the image. Task, images and prompt are identical.

`analysis/pope_strata.py` re-scores the existing generations per stratum with the same
`fs_common.sdt_from_counts` (same loglinear edge correction, same clipping flag) so only the row subset
differs. Sequential arm, 9 cells:

| stratum | mean d′ | Σ\|Δc\| | mean \|c\| | endpoint c |
|---|---|---|---|---|
| random | 2.6191 | 0.9683 | 0.3641 | +0.2712 |
| popular | 2.3487 | 0.8874 | 0.2365 | +0.1428 |
| adversarial | 1.9375 | 0.9964 | 0.1203 | −0.0698 |

Untuned base, for reference: d′ 2.725 / 2.408 / 2.073, c +0.615 / +0.461 / +0.300.

**This is a double dissociation, and it is the cleanest evidence in the paper that c and d′ are separate
parameters rather than two views of one thing.**

1. **d′ tracks negative difficulty exactly as POPE's design intends**: 2.619 → 2.349 → 1.938, monotone,
   a spread of 0.68. Harder negatives genuinely are harder to discriminate. The instrument responds to the
   thing it is supposed to measure.
2. **Criterion DRIFT does not track difficulty at all**: 0.968 / 0.887 / 0.996, a spread of 0.109 — about
   a sixth of d′'s spread, and not monotone. The criterion moves by the same amount whichever way the
   negatives were drawn.

So the drift is a property of the decision rule, not of the sampling scheme. Had it appeared only under
easy negatives it would have been an artifact and the paper would have had to say so.

**A second, smaller point worth reporting.** The criterion LEVEL does shift across strata (mean |c| 0.364
/ 0.237 / 0.120; endpoint c +0.271 / +0.143 / −0.070), i.e. adversarial negatives pull the operating point
toward and past zero. That is expected — a harder negative set changes where the optimal threshold sits —
and it is the level, not the drift, that responds. The two behave differently, which is the dissociation.

**Limits.** This generalizes across NEGATIVE SAMPLING, not across benchmarks: all three strata share
POPE's images, prompt template and yes/no format. A genuinely independent discriminative benchmark is
still owed, and MME-existence is now enabled (BENCH=1) on the method and second-backbone arms so that
gap closes as those land. The strata also share rows with the pooled analysis, so these numbers are not
independent evidence from the headline — they are a decomposition of it.

## The balanced-probe target is an APPROXIMATION, not a derivation (2026-09-11, evening)

An adversarial-review pass found a real error in the method's premise, and it is mine. I wrote, in
`NEXT_EXPERIMENTS.md`, in `crit_balanced.py` and to Alex directly, that on a balanced probe "c = 0 is
DERIVABLE, not a values choice". **That holds only under equal variance, and our own z-ROC fits reject
equal variance.**

Balanced accuracy is maximised where the ROC slope equals 1, which coincides with c = 0 only when the
z-ROC slope b = 1. Verified numerically at realistic d_a:

| z-ROC slope b | optimal yes-rate | cost of targeting 0.5 instead |
|---|---|---|
| 1.000 (equal variance) | 0.5000 | 0 |
| 0.800 | 0.4787 | 0.0012 |
| 0.715 (our sequential) | 0.4693 | 0.0026 |
| 0.550 (our anchor) | 0.4530 | 0.0072 |

**The approximation survives, but the justification does not.** Targeting 0.5 costs at most 0.007
balanced accuracy at the most extreme slope we observe, against the 0.040 the anchor gives up by parking
at c = +0.697 — an order of magnitude smaller than the problem it corrects. So the method is defensible
as an approximation with a stated bound, and indefensible as "correct by logic". `crit_balanced.py` has
been corrected; the paper must never call it derivable.

**Two further problems found in the same pass, both acted on.**

1. **τ = 1.0 is very likely the wrong default, and the arms were queued with it.** Expanding the sigmoid,
   σ(u/τ) = ½ + u/(4τ) + O(τ⁻³): LARGE τ collapses the "rate" mode toward mean-matching — that is, toward
   the `mean` ablation it exists to beat — and only τ → 0 gives the median-matching constraint the identity
   actually asks for. At τ = 1 it sits between the two and dominates neither. Small τ is not free either:
   it concentrates the gradient on items near the boundary and inflates the variance of a k = 8 estimate.
   **Queued τ = 0.3 and τ = 0.1 arms alongside the default so this is swept rather than assumed.**
2. **The identity is exact on the full item set but not on the SCORED subset.** Parse failures drop rows
   asymmetrically (602 rows across 23 of 120 stage-cells), so 46 of 246 cells are not exactly balanced
   once scored. Max prevalence deviation 0.0049; max |yes-rate − ½(H+FA)| 0.0036. That is ~50× smaller
   than the smallest per-cell criterion range, so it does not threaten the method, but it is a bound and
   the docstring's "exactly, in every one" was wrong.

**Also flagged and OWED before the method can be compared honestly:** `analysis/pcr_transfer.py` fits its
post-hoc scalar against the BASE model, so the "free scalar correction" competitor inherits precisely the
mis-placement the method removes. As written, the method-vs-post-hoc comparison is rigged in the method's
favour. A PCR-0 variant that fits the scalar against the balanced target must exist before that verdict
is quoted.

**A separate correction with the same root cause:** the placement analysis references c = 0 throughout
(mean |c|, "the base is mis-placed by 0.431"). Under unequal variance the reference point is not exactly
0. The model-free fix is to reference the JOINT arm's endpoint criterion (+0.088) instead — it is an
empirical optimum from data rather than a modelled one, and every conclusion is unchanged because the
anchor sits 0.6 away from it either way.

---

## 2026-09-12 — Drift localization, first leg: the projector freeze REPEATS the anchor's failure

`fsL_locproj_o1_s17` finished and was scored by `analysis/mechanism_readout.py --mode loc`. It freezes the
multimodal projector (`CLH_LORA_SAVE=none`) and trains the same UCIT sequence otherwise unchanged.

| arm | what adapts | Σ\|Δc\| | endpoint c | endpoint d′ |
|---|---|---|---|---|
| `fsL_seq_o1_s17` | baseline, all adapt | 0.7643 | **+0.0679** | 2.1878 |
| `fsL_locproj_o1_s17` | projector FROZEN | 0.2237 (−71%) | **+0.2739** | 2.2955 |

**Read the endpoint column, not the drift column.** The freeze removes 71% of the criterion path and lands
the criterion **4× further from zero** than simply doing nothing. This is the anchor's failure mode exactly:
Σ|Δc| improves, placement degrades. d′ is unchanged to within the usual range (2.19 → 2.30), so once again
the whole difference is placement, not discriminability.

**Status: SINGLE CELL (o1/s17), not a claim.** One arm against one baseline cell. `fsL_locproj_o1_s23` is
queued for a second seed. Nothing about localization may be written as a finding until it replicates, and
if it is written up, the endpoint cost belongs in the same table as the drift reduction — the same rule
already binding the anchor.

**Why this matters more than the arm itself.** Three independent interventions now show the same pattern:
the anchor (regularizes toward the frozen base), critp (same), and a projector freeze (no reference model
at all). The third is the informative one, because it shares no mechanism with the first two — it cannot be
explained by "they inherit the base's mis-placed c = +0.431." That the pattern survives an intervention with
no reference model suggests the trade-off between drift reduction and endpoint placement may be structural
rather than an artifact of choosing the wrong anchor target.

**Consequence for the queue (acted on).** `fsL_critbal_o1_s17` was promoted to queue position 1. It is the
only queued arm that corrects the TARGET rather than damping the PATH, so it is the direct test of whether
the trade-off above is structural or merely a bad-reference artifact. If critbal also buys Σ|Δc| with
placement, the structural reading is strongly supported and the paper's method section should say so
plainly rather than presenting a fourth failed candidate.

**Cluster reality check (2026-09-12 ~01:40 cluster time).** vgi1 and vgi2 are both memory-saturated
(5.4G and 11.2G free of 122G). jiwoong holds 152G across 3 jobs, two of which do not release for 26+ hours;
yunseoc holds 24G for ~3 days. Our account holds 46G across 5 jobs, 4 of them another lane's. We are not
the blocking account, but essentially the only memory freeing before morning is our own, so expect 1–2
further arms to land, not 20.

## 2026-09-12 — critbal would have died at startup; caught before it got a GPU slot

`fsL_critbal_*` had never been run, and a pre-flight check of the dispatch path found a fatal path
mismatch. `fullstudy/run_arm.py` resolves the probe as `os.path.join(D, "grounding",
"probe_balanced.jsonl")`, and for `SUITE=ucit` the scheduler sets `D = $ROOT/data_ucit`
(`wave_sched.sh:76`). The probe built by `build_balanced_probe.py` was written to
`$ROOT/data/grounding/`. `data_ucit/grounding/` existed — it holds the anchor's `caption_anchor.jsonl`
and `grounding_pairs.jsonl` — but had no `probe_balanced.jsonl`, so every critbal arm would have
crashed on load the moment it was allocated a GPU, after waiting hours in the queue.

**Fixed by placing the probe under `data_ucit/grounding/`** (copied, not moved, so pilot-suite paths
still resolve). Verified after placing:

- 600 rows, `balance = {present: 300, absent: 300}`, `yes_fraction` 0.5, seed 17.
- Keys are exactly `id`, `image`, `object` — **no label-like key**, so the parent `CriterionPreserver`'s
  guard against being handed a labelled file passes, and the arm stays label-free as designed.
- Every referenced image resolves under both `data_ucit/` and `data/`.
- The cluster's `code/method/answer_tokens.py` exports `resolve_answer_token_sets`, so the pooled
  casing-variant decision statistic is the one that will actually be used. (Its Sep 9 mtime looked stale
  against the local fix; the content matches.)

**Deliberately NOT done: no fallback path added to `run_arm.py` tonight.** Five arms are already queued,
and a queued job picks up whatever is on disk when it *starts*, so editing the shared harness now would
silently change all of them to fix a problem the file placement has already fixed. The fallback (resolve
the probe from the suite dir, else from `data/grounding/`) is a follow-up for after the queue drains.

**Process note.** This is the second time a run was saved by checking the dispatch path rather than the
module: the module was correct in isolation and the wiring was wrong. Pre-flight belongs on the resolved
path, not on the code.

## 2026-09-12 — RETRACTION, same night: the "structural trade-off" reading is refuted

Earlier tonight I logged the projector-freeze result and argued that because it shares no mechanism with
the anchor or critp, a drift-versus-placement trade-off might be **structural**. That reading is wrong and
is withdrawn. `analysis/tradeoff_placement.py` tested it directly against every cell on disk; the report is
`analysis/TRADEOFF_PLACEMENT.md`.

**1. The relationship is the anchor family's signature, not a law.** Correlation between drift and
placement error (reference = JOINT's empirical endpoint +0.088; a trade-off predicts r < 0):

| set | n | r (post-settling) |
|---|---|---|
| all cells | 20 | −0.467, CI [−0.760, −0.169] |
| **anchor family dropped** | 11 | **+0.682**, CI [+0.245, +0.886] |
| arm identity partialled out | 20 | +0.247 |

Dropping the anchor family does not weaken the effect, it **reverses** it. Leave-one-arm-out agrees: only
the anchor's presence produces a negative relationship. The pooled −0.467 is a two-group separation with
n = 20 cells but **n = 3 arms**.

**2. The JOINT arm is a standing counterexample in our own data.** It has the lowest drift of any arm
(0.2278, below the anchor's 0.3641) *and* the best placement (0.0255). It dominates both anchor and
sequential on both objectives at once, so "you cannot have both" is false here.

**3. My premise table mixed estimators — the error this project already has a standing rule against.**
I paired `sequential 0.7643` (raw, single cell) with `anchor ~0.364` (post-settling, 9-cell mean) and
`locproj 0.2237` (raw, single cell). Matched raw-to-raw **at the same cell o1/s17, the anchor's path is
0.7834 versus sequential's 0.7643 — worse, not better.** The anchor's drift reduction is largely a
post-settling phenomenon. locproj's per-stage criteria were never recorded, so it cannot be placed on the
pre-registered axis at all with what is on disk.

**4. The one intervention-versus-intervention comparison runs the wrong way.** Matched at o1/s17, locproj
has both less raw drift (0.2237 vs 0.7834) and better placement (0.186 vs 0.675) than the anchor.

Robust to referencing c = 0 instead of +0.088 (−0.441 → +0.594 with the anchor family dropped).

**What survives untouched**, because each is a direct measurement rather than a cross-arm generalization:
the anchor's endpoint deficit versus sequential (POPE F1 0.7963 vs 0.8625, 9/9 cells, no overlap);
sequential's endpoint already sitting at the JOINT bound; and the single-cell observation that the
projector freeze lands the criterion further from zero than doing nothing.

**Process note, and the reason this is logged rather than quietly dropped.** The trade-off reading was
attractive because it would have converted a negative method result into a headline finding, and I
generated it from three numbers that a single estimator-matching check would have rejected. The standing
rule exists precisely for the case where the mismatched comparison flatters us. It took twenty minutes to
refute and would have been a fatal reviewer objection had it reached the paper.

---

## 2026-09-12 — The OWED PCR-0 variant exists, and the method loses to a properly-aimed competitor

The 2026-09-11 entry recorded that `analysis/pcr_transfer.py` fitted its post-hoc scalar competitor
against the FROZEN BASE (c = +0.431), handing the method a competitor aimed at the wrong target, and
that no method-vs-post-hoc verdict could be quoted until a properly-aimed variant existed. **It now
exists** (`analysis/pcr_transfer.py`, rewritten; `analysis/PCR0_VERDICT.md`). The OWED item is
discharged. The verdict has three parts and none of them favours the method.

**1. On placement, the method loses 9/9 — to BOTH competitors, and to doing nothing.**

| series | endpoint distance from target c\* = +0.0878 |
|---|---|
| anchor family | **0.6094** [0.5565, 0.6605] |
| plain sequential (no method) | 0.0811 |
| JOINT | 0.0255 |

The anchor's *best* cell (0.4536) is still worse than the **handicapped, base-aimed** competitor's
landing point (0.3436). The target is the JOINT arm's empirical endpoint criterion, not a modelled
c = 0; the script **dies rather than falling back** to a modelled zero, citing the falsified
equal-variance premise.

**2. The pre-registered boolean is NO-DATA, not a negative.** It needs per-item POPE / rephrased-POPE /
probe logit dumps; `fullstudy/results_fs` holds generations only. The script now emits `ABSENT` and
refuses to produce a number. **No method claim may be quoted from the path comparison.**

**3. The rigging did not distort the boolean — it distorted which axis was reported.** Re-aiming the
scalar adds a *constant*, and a constant shifts every stage's criterion equally when a cell's stages lie
on one z-ROC (sequential mean R² = 0.957). Verified at n = 9000 across the fitted slope range: the path
moves under 4% with no systematic sign. So the path axis is where the two competitors are nearly
identical, and the axis on which a free scalar is unbeatable was **not in the pre-registered test at
all**. The fix neither rescues the method nor overturns the old boolean; it shows the boolean was never
sufficient on its own.

**A rigging found inside the fix itself, and corrected.** The first draft referenced a corrected
trajectory to the *uncorrected* base, charging PCR-0 a first transition of |c\* − c_base| for doing
precisely the thing it exists to do. Each variant now starts from its own corrected stage 0 (a no-op for
PCR). That moved the fixture's PCR-0 path from 1.262 to 0.246. Recorded because it is the same class of
error as the original rigging and was caught only by looking for it deliberately.

**Caveat kept visible.** PCR-0's exactness holds for the base and is approximate for a trained
checkpoint by an **unmeasured transport residual**. It would have to exceed 0.61 criterion units to flip
this result, which is implausible but untested.

**Consequence for the paper.** The method section's honest claim is negative and should stay negative.
`critbal` remains the one queued arm that could make it positive, and nothing here forecloses it.

---

## 2026-09-12 — The answer-token casing FLIPS between stages: measured, and it invalidates the existing logit dumps

Checked directly while investigating whether the cluster's existing per-item logit dumps could settle the
pre-registered method-vs-post-hoc boolean without new GPU time. They cannot, and the reason is measurable.

Argmax answer token at the POPE answer position, `psL_seq_o1_s17`, n = 9000 rows per stage:

| stage | dominant tokens | casing |
|---|---|---|
| k1 | `No`(1939) 57% / `Yes`(3869) 43% | **capitalized** |
| k2 | `yes`(4874) 51% / `no`(694) 49% | **lowercase** |
| k3 | `No`(1939) 53% / `Yes`(3869) 47% | **capitalized** |
| k4 | `no`(694) 65% / `yes`(4874) 35% | **lowercase** |
| k4 (anchor) | `no`(694) 67% / `yes`(4874) 33% | **lowercase** |

In every stage the top two ids cover ~100% of rows, and **which pair** they are flips with the stage.

**Consequence 1 — the existing dumps are unusable for any logit-based criterion analysis.** They were
written 2026-09-09 17:48; `method/answer_tokens.py`, which pools over casing variants, is dated
2026-09-09 20:41 — about three hours later. So `z_yes`/`z_no` in those files are single-casing values,
and at roughly half the stages they read vocabulary rows the model is not emitting. Settling the
pre-registered boolean requires **re-dumping** with the pooled resolver, not reusing these files.

**Consequence 2 — no live claim is affected, verified rather than assumed.** The four consumers of
`pope_logits.jsonl` are `blind_prior_share.py`, `logit_bias_analysis.py`, `pcr_transfer.py` and
`threshold_sweep.py`. The only readout any of them has produced is `pcr0_placement_audit.json`, which is
computed from the certified aggregate and explicitly emits `ABSENT` for the logit-dependent half. The
body quotes no logit-derived number. The recoverable-headroom embargo already standing since 2026-09-09
remains correct and is now independently justified.

**Consequence 3 — this strengthens the parsing-confound audit rather than threatening it.** Text scoring
lowercases before matching (`out.strip().lower().startswith("yes")`), so a casing flip is absorbed
entirely and the reported 81,000/81,000 parse rate stands. The asymmetry is the point worth stating in
the instruments section: **a string-matched parser is invariant to this drift; a fixed-token-id logit
readout is not.** That is precisely why the decision statistic pools over casing variants, and it is a
concrete measured instance of the SEFE-style answer-format drift the paper audits for.

---

## 2026-09-12 — RETRACTION: the POPE-stratum "double dissociation" is arithmetic, not a finding

Flagged by the pre-submission novelty sweep and **verified independently here on our own readout** before
acting. It is a correctness problem, not a priority one, and the claim must not ship as written.

**The mechanism.** POPE's three strata differ *only* in how the ABSENT objects are sampled; they share
their positive items. So the hit rate is constant across strata (0.8480 / 0.8480 / 0.8467), and with H
fixed both indices collapse onto a single free variable:

    d' = z(H) - z(FA)          c = -0.5*[z(H) + z(FA)]
    with z(H) constant  =>  Δd' = -Δz(FA)  and  Δc = -0.5*Δz(FA)
    =>  **Δd' = 2Δc, identically.**

**The check on our data** (`analysis/readout/pope_strata_seq.json`, seq arm, 9 cells), random →
adversarial:

| quantity | value |
|---|---|
| Δd′ | −0.6816 |
| 2Δc | −0.6820 |
| **residual** | **+0.0004** (0.06% of the effect) |

So "d′ falls with negative difficulty while the criterion does not track that fall" — compared as
**levels across strata** — is forced by construction. It would hold for any model whatsoever, including
one with no criterion drift at all. It is not evidence for anything.

**What survives, and it is the part worth keeping.** The criterion **path length across stages within
each stratum** — 0.9683 / 0.8874 / 0.9964 — is a *second* difference and is NOT forced by the identity.
That the drift has nearly the same magnitude in all three strata does support the claim the section was
really after: criterion drift is a property of the decision rule, not an artifact of how the negatives
were drawn. That claim stands; the levels comparison does not.

**A further caution from the same sweep, not yet independently checked:** popular and adversarial share
their negative object set almost completely, so the three strata are **not three independent conditions**
and must not be treated as n = 3.

**Action.** The levels/"double dissociation" framing is withdrawn. `analysis/pope_strata.py` gains a
docstring warning so the identity is not rediscovered as a result, and the paper section is being
rewritten around the path claim only.

---

## 2026-09-12 — The range-restriction objection to the z-ROC claim: raised, tested, answered

A pre-submission sweep of the ROC literature flagged that **R² is range-dependent**: since
R² = 1 − SS_res/SS_tot, compressing the spread of z(FA) shrinks SS_tot and depresses R² for the *same*
residual scatter. An anchor regularizer is precisely the kind of intervention that would compress the
operating range, so "the anchor's R² is low" is open to the reading that we measured compression rather
than incoherence. Unanswered this is fatal to the secure half of the primary claim.

**Tested** (`analysis/zroc_range_check.py`, reusing the certified `z()`/`fit()` logic):

| arm | n | mean R² | mean z(FA) range | mean residual RMSE |
|---|---|---|---|---|
| seq | 9 | 0.9569 | 0.4441 | 0.0183 |
| anchor | 9 | 0.5857 | **0.2121** | 0.0316 |
| JOINT | 2 | 0.9446 | **0.1417** | 0.0072 |

**The objection's premise is TRUE: the anchor's operating range is compressed**, to 48% of sequential's.
So R² alone is not a safe statistic here and the paper must not lean on it unaccompanied.

**But the objection's conclusion is FALSE, and the JOINT arm is the internal control that shows it.**
JOINT's range is *smaller still* — 67% of the anchor's — yet JOINT's R² is 0.9446 against the anchor's
0.5857, because JOINT's residuals are **4.4× smaller** (0.0072 vs 0.0316). A narrow operating range
therefore does not produce a low R² in this data. The anchor's low R² comes from genuine scatter about
its own fitted line: its residual RMSE is 1.73× sequential's.

**Reporting rule adopted.** Residual RMSE in z-units is the range-free statistic and must be reported
**alongside** R² everywhere the coherence result appears, with JOINT named as the control. This converts
the vulnerability into a strengthening: the result now rests on a statistic the objection cannot touch.

**Two further methodological cautions from the same sweep, not yet acted on:**
1. **OLS on a z-ROC is the estimator the source field moved away from** (Pesce & Metz 2007, PMC2693394,
   on maximum-likelihood binormal fitting) because regression on z-transformed rates is attenuated. A ML
   fit as a robustness check would close this; we currently report OLS of z(H) on z(FA) and must at least
   say so explicitly.
2. **A nested model comparison would be a better statistic than R²** — one shared (a, b) across the six
   checkpoints versus per-checkpoint d_a, giving a likelihood-ratio/AIC test of "one ROC" directly rather
   than a goodness-of-fit proxy. It would also defuse the circularity objection, because the null and the
   alternative are then stated as models rather than as a threshold on a fit statistic.

---

## 2026-09-12 — The difficulty-rides-position confound: checked, and the design controls it

The ordering literature supplies the confound that would destroy the order-effects headline. From
arXiv 2608.18066: *"The default order exhibits an implicit easy-to-hard curriculum... In general, tasks of
different difficulties are not distributed evenly in the default order."* If the orderings do not balance
task **difficulty** against **position**, then η²(task) silently absorbs difficulty and η²(depth) is
suppressed — which is exactly the pattern we report (0.731 vs 0.049).

**This is answerable from the design alone, before any result.** `analysis/design_crossing_check.py`:

| task | p1 | p2 | p3 | p4 | p5 | p6 | scored depths (p ≥ 2) |
|---|---|---|---|---|---|---|---|
| ArxivQA | 1 | . | 1 | . | . | 1 | 3, 6 |
| CLEVR-Math | . | 1 | . | 1 | 1 | . | 2, 4, 5 |
| Flickr30k | . | . | 1 | 1 | 1 | . | 3, 4, 5 |
| IconQA | . | . | 1 | 1 | . | 1 | 3, 4, 6 |
| ImageNet-R | 1 | 1 | . | . | 1 | . | 2, 5 |
| VizWiz | 1 | 1 | . | . | . | 1 | 2, 6 |

**Every task occupies three distinct positions and is scored at two or three distinct depths.** No task is
pinned to one depth, so η²(task) cannot be a relabelling of η²(depth) and the confound is controlled by
construction rather than by argument. The script **asserts** this condition rather than printing it, so a
future change to the orderings that reintroduced the confound would fail loudly.

**Claim exactly this and no more.** It is a *partial* crossing — 3 of the 6! = 720 possible orderings —
sufficient to separate the two factors, not a complete factorial design.

**Two related limits to state in the paper, both from the same sweep:**
1. **η² here is descriptive over purposefully selected levels.** The nearest methodological neighbour
   (arXiv 2605.09041, BiAxisBias — prespecified factorial η² with resampling, applied to LLM answering
   bias) hedges its own version in exactly the terms that apply to us: *"descriptive effect sizes for
   purposefully selected levels, not row-independent ANOVA tests or estimates of interaction
   prevalence."* Our three orderings are purposefully selected levels; the partition is not an inference
   about the population of orderings. **We should say this ourselves.**
2. **The statistic is not the contribution.** η² over a factorial design applied to model answering bias
   was published in August 2026 (BiAxisBias), and fANOVA-style attribution in ML goes back to 2014. What
   is unclaimed, verified across ~20 opened papers, is **crossing task identity with sequence position
   over a training sequence**. Lead with the design, not the statistic.

---

## 2026-09-12 — CORRECTION: the projector-freeze entry above was read off a PARTIAL run and is wrong

The 2026-09-12 entry "Drift localization, first leg" recorded `fsL_locproj_o1_s17` as Σ|Δc| = 0.2237
(−71% vs baseline) with endpoint c = +0.2739, and drew the conclusion that freezing the projector buys
path stability at the cost of endpoint placement — the anchor's failure mode again. **That arm was still
in eval when the number was read.** It is now 6/6 and both halves of the reading reverse:

| arm | what adapts | Σ\|Δc\| | endpoint c | endpoint d′ |
|---|---|---|---|---|
| `fsL_seq_o1_s17` | baseline, all adapt | 0.7643 | +0.0679 | 2.1878 |
| `fsL_locproj_o1_s17` | projector FROZEN | **0.9895 (+29%)** | **+0.0041** | 2.2298 |
| `fsL_loclate_o1_s17` | layers 16–31 only | 0.7237 (−5%) | +0.2953 | 2.2345 |
| `fsL_locearly_o1_s17` | layers 0–15 only | not finished | — | — |

**So the earlier conclusion is withdrawn in full.** Freezing the projector does not reduce drift — it
*increases* it by 29% — and it does not worsen endpoint placement, it produces **the best placement of any
arm measured**: c = +0.0041, against sequential's +0.0679 and JOINT's +0.088. Freezing the late layers
does the opposite: slightly less path, markedly worse endpoint (+0.2953).

**What this does and does not license.** It is a genuine dissociation between the path and the endpoint,
and it points at the projector as carrying endpoint mis-placement rather than path movement. It is also
**one cell per arm, with two of four legs missing** (`locearly` never ran, `locproj_o1_s23` never ran), so
it is not a claim and nothing about localization may enter the paper. d′ is unchanged across all three
(2.19–2.23), so whatever moves is placement, not discriminability.

**The process failure, recorded because it is the reusable lesson.** `mechanism_readout.py` scores whatever
stages exist and prints a number without saying how many it found. A partially evaluated arm therefore
reads as a finished one, and its Σ|Δc| is a path over fewer transitions — mechanically smaller — which is
exactly why the truncated run looked like a large drift reduction. **Any readout of a running arm must
report its stage count beside its numbers, and no arm may be quoted before `EVAL_DONE` = 6/6.** The
overnight loop's own output had "(not finished)" markers for other arms and none for this one, which is
what made the partial number look trustworthy.

---

## 2026-09-13 — Does the drift cost anything? Yes. And "the endpoint is already optimal" was 32× too precise

Run to test whether the paper's motivation survives its own data: sequential's endpoint criterion appeared
to sit at the JOINT bound, which would mean the drift is self-correcting and there is nothing to fix.
**The reframe does not survive, and a live claim needed correcting.** (`analysis/does_drift_cost.py`,
`analysis/DOES_DRIFT_COST.md`.)

### The correction: a signed mean that cancels

| quantity | value |
|---|---|
| mean endpoint c (sequential, 9 cells) | +0.0903 |
| \|mean − c*\| — what the paper quoted | **0.0025** |
| **mean \|c − c*\| — the honest per-cell statistic** | **0.0811** |
| cancellation ratio | **32.4×** |
| per-cell spread | [−0.0143, +0.2607], range 0.275, sd 0.0988 |
| cells above / below the target | 4 / 5 |

The agreement between +0.0903 and c* = +0.0878 is **positive and negative deviations cancelling**, not
cells landing on the target. This is the estimator-matching failure again, in its third distinct form:
averaging a signed quantity and then taking the absolute value, rather than averaging the absolute value.
The body text has been corrected; the results table already carried the honest interval `[-0.01, 0.26]`.

**What survives, and it is real.** Endpoints *do* migrate toward the target with depth — mean absolute
error 0.2012 at k1 falling to 0.0811 at k6, and k6 is the only stage whose cells straddle c*. But they do
not converge: **0/9 cells are monotone**, the path is 2.5× the net displacement, the endpoint is interior
to its own visited range in 6/9, and while the endpoint beats a *typical* earlier stage 9/9 it beats the
*best* earlier stage only **3/9**. Stage 6 is not a fixed point.

### The drift does cost accuracy

At each cell's worst-drifted mid-sequence stage, POPE F1 sits **0.0118 below that cell's own endpoint F1**
(95% CI [0.0079, 0.0162]), positive in **9/9 cells**, sign test p = 0.004. That is ~2.6× both noise floors
(item-sampling 0.0042, seed-to-seed 0.0046). Real and measured — and small: the anchor's endpoint deficit
is 6.6 points, more than fifty times larger.

### The residual endpoint gap is ordinary forgetting, not criterion drift

With the criterion effectively matched to JOINT (gap +0.0025), sequential still trails JOINT on endpoint
F1 by 0.0042, with **7/9** cells below JOINT's lower cell. Criterion placement cannot carry that, so
separability does. **Bounded honestly:** JOINT is n=2, and this rests on single-point d′ — the estimator
family this project already retracted a claim over. d_a agrees in sign, but JOINT's two d_a values
straddle the sequential mean, so this is suggestive rather than established.

### Power, stated rather than hidden

All four sequential path-versus-endpoint correlations carry the sign the paper predicts (+0.614, +0.569,
−0.488, −0.629) and **none is significant**: at n = 9 the critical |r| is 0.666 and 80% power needs
ρ ≥ 0.816. The estimates sit exactly in the blind spot. **This is underpowered, not null**, and must be
reported as such. An unplanned positive control: the cross-arm correlations reproduced
`TRADEOFF_PLACEMENT.md`'s −0.467 pooled and +0.682 anchor-dropped to three decimals from an independent
code path.

### Consequence for how the paper is motivated

The strongest honest motivation for the method half is **dispersion, not endpoint quality**: a single run
lands anywhere in a 0.275-wide window and a practitioner cannot know which end they drew. That survives
the negative method result intact. It carries its own caveat — **the anchor does not deliver it either**
(endpoint sd 0.0834 vs sequential's 0.0988, a 0.84× variance gain bought at 0.61 units of bias).

---

## 2026-09-13 — SECOND CORRECTION to the localization entry: I used the reference this project already retracted

The 2026-09-12 correction entry reported that freezing the projector produces "the best endpoint placement
of any arm measured (c = +0.0041)". **That ranked placement by closeness to c = 0 — the reference this
project retracted on 2026-09-11** in favour of the JOINT arm's empirical endpoint c* = +0.0878, because
c = 0 is optimal only under equal variance and our slopes are 0.55–0.80. Re-scored against our own
reference:

| arm | path Σ\|Δc\| | endpoint c | **\|c − c*\|** |
|---|---|---|---|
| `fsL_seq` (baseline) | 0.7643 | +0.0679 | **0.0199** |
| `fsL_locproj` | 0.9895 | +0.0041 | **0.0837** — 4.2× worse |
| `fsL_loclate` | 0.7237 | +0.2953 | **0.2075** |

**The baseline has the best placement, and `locproj` is worse on BOTH axes.** There is no dissociation for
it. The crossover sits at c* = 0.036, so the reversal holds across JOINT's entire empirical range — it is
not sensitive to the exact reference value. Only `loclate` has the trade-off shape, which is the anchor
family's known signature rather than anything new.

**This is the third time the same reference error has been made in this project.** It is not a slip to
note and move past: any statement ranking criterion placement must use |c − c*| with c* = +0.0878, and
any text quoting closeness to zero is wrong by construction.

### Three further reasons the localization line is not ready

1. **Nothing exceeds baseline cell-to-cell noise.** Sequential's own o1 seeds span path [0.7376, 1.2021]
   and |c − c*| [0.0151, 0.1729], endpoint c sd 0.1027. `locproj`'s 0.9895 and 0.0837 are **inside** that
   envelope — 0.0837 sits inside the bootstrap CI of the sequential nine-cell mean, 0.0811 [0.0518,
   0.1122]. It is an ordinary sequential cell. Only `loclate` leaves the envelope, and only by 0.35 sd.
2. **The "+29% / −5%" headline compares one cell to one below-median cell.** Against the sequential arm
   *mean* path the figures become **+5.7% / −22.7%**. Estimator matching again, in yet another form.
3. **Capacity is not matched, and the unmatched pair is the headline one.** Trainable parameters, computed
   by running the project's own `_lora_overrides()` against the LLaVA-1.5-7B module universe: seq
   180,887,552; `locproj` 159,907,840 (−11.6%); the two bands 100,933,632 (−44.2%, and exactly matched to
   each other). **`locproj` and `loclate` differ by 59.0M parameters (37%) as well as by component**, and
   placement error is currently **monotone in trainable-parameter count** across all three arms. A
   simpler story — "this arm trained less" — is not excluded, and nothing in the repo records a trainable
   count for any arm.

### Provenance problem worth fixing regardless

The two freeze arms have **no data in this repository** — not in `results_fs/`, not in `fs_aggregate.json`
(stamped 2026-09-09, predating them). Their numbers exist only as prose in the prereg. `EVAL_DONE = 6/6`
was asserted by hand; the loc readout still prints no stage count, which is exactly what produced the
first wrong entry. **`analysis/readout/tradeoff_placement.json` still carries the retracted 0.2237/0.2739
values labelled `n_stages: 6`** and must be regenerated.

### Verdict

**The localization dissociation is withdrawn as a finding and is not a follow-up paper seed in its current
form.** The cheapest path to knowing is Tier 0 and costs no GPU: gate the readout on `EVAL_DONE` *and*
`n_total == 9000`, dump the per-stage criterion vector it currently computes and discards, and recover
trainable counts from the SLURM logs. The first GPU arm worth running is a **capacity control**
(`r=56` + projector = 160.9M, matching `locproj`) which can kill the line on its own. Novelty is also not
cleared: `related_work.md` already contains parameter-group-freezing localization work, including
POPEv2/Obliviate localizing hallucination to the LM head — the one component frozen in every arm here.

---

## 2026-09-13 — The dose-response residual: the account is not weak on UCIT, it is inapplicable there

Run to chase the unexplained fact that a certified zero-dose stream still moves the criterion 0.40–0.55 in
9/9 cells. (`analysis/dose_residual.py`, `analysis/DOSE_RESIDUAL.md`.) The pipeline first re-derives five
numbers the prereg already records — the zero-dose headline (0.403–0.550, mean 0.472), SEQ η² task/depth
post-settling (0.7311 / 0.0488), the anchor's (0.3523 / 0.0526), and all six per-task pulls — all matching
exactly, so a pipeline error would have surfaced before any claim.

### The finding: on UCIT the dose variable does not vary

**The dose regressor has sd 2×10⁻⁴ — all six UCIT tasks sit at zero.** Scored against the account's own
prediction (Δc = 0), R²_predictive = **−0.098**. Answer statistics explain **0.0%** of Δc variance on the
full study; the residual *is* Δc.

This reframes the pre-registered failure. It is not that the answer-statistic account is weak on UCIT —
**it is inapplicable there**, because UCIT contains no dose contrast to explain anything with. The
account's support comes from the **pilot suite**, which is the only design on disk where dose varies
(Δc ~ dose R² = 0.578, with length adding only 0.024). Any sentence in the paper attributing UCIT drift to
answer statistics must be scoped to the pilot, or dropped.

### A hard design ceiling, previously uncomputed

Within an ordering, position fixes both the task and its predecessor, so `ordering × position` is the
**finest partition any stream-level variable can induce**: η² = 0.823. Task alone is 0.683.

- **Headroom for every non-task explanation combined: 0.140.**
- **Irreducible seed noise: 0.177.**

So the headroom is smaller than the noise. Inside a task, position, predecessor and ordering induce the
*identical* three-way split of its nine cells — **adjacency, depth and ordering are not separable in this
design**. No reanalysis of these runs can settle the question; it needs a different design.

### Ruled out, with the reason

| candidate | verdict |
|---|---|
| optimizer steps | **inapplicable**, not merely unsupported — every task trains exactly 125 steps (`TRAIN_CAP=8000`, all sources ≥ 23,998 rows). Gradient/loss magnitude is NO-DATA, not cleared. |
| prompt distribution | η² 0.043, sign inconsistent 3/9, despite prompt length spanning 13.0–99.9 words |
| sequence depth | p = 0.178 → 0.246 → 0.780 across three looks |
| mean reversion | the Δc-on-c_prev slope of −0.922 is the **Galton artifact**; level autocorrelation r(c_k, c_prev) = +0.111 [−0.19, +0.36] — memorylessness, not a new variable |
| Δd′ | r = +0.909 is **arithmetic**: c and d′ are orthogonal contrasts of the same z-scores, with sd z(FA):z(H) = 1.41 |

### Survives as a hypothesis, not a finding

**Predecessor identity beyond task**: 0.089 of total variance (within-task p = 0.0015), 0.055
post-settling (p = 0.0057), and it **replicates on the anchor arm** (p = 0.0245). It is not the
answer-length account one step back (r = +0.338; IconQA at 1.0 words is the most positive predecessor).
Chosen after looking at the data, so it is a hypothesis with an owed out-of-sample test.

### Checked and NOT an error

The pilot Flickr figure of −0.044 in the body is labelled `|Δyes-rate|`, which is correct — the same cell
is **+0.2111 in Δc**, and a falling yes-rate with a rising criterion is consistent. The "Flickr relaxation"
reading and the answer-format candidate agree on o1 and were never in conflict. The one free out-of-sample
test (Flickr30k appears in both suites) is **inconclusive**: o1's three cells give +0.1187 (3/3, agreeing
with UCIT's +0.163), and the apparent sign flip rests entirely on the single o2 cell (−0.4941), where
Flickr sits immediately after the +0.58 high-dose stage.

### Cheapest experiment that would settle it

The already-written `--mode dose` arms: five single-stage Flickr30k runs truncated to 1/2/4/8/untruncated
words. They break the length↔identity confound and lift the p-floor of 0.067 that the current design
cannot beat. None have results on disk.

---

## 2026-09-13 — The coherence result REPLICATES on a second task sequence (zero GPU)

The primary claim rested entirely on UCIT, which leaves open whether it is a property of continual tuning
or of that benchmark. The pilot suite answers it for free: a **different task sequence**
(ScienceQA → TextVQA → Flickr30k → VizWiz), same backbone, same instrument, run months earlier for a
different purpose. (`analysis/zroc_pilot_replication.py`.)

**Recovering the operating points.** The pilot's `pope.csv` stores recall and yes-rate rather than hit and
false-alarm. POPE is balanced 50/50, so H = recall and FA = 2·yes_rate − H. The identity is **verified
against the reported precision** H/(H+FA) on every row before the row is used; a mismatch drops the cell
rather than fitting it. Checked exactly on S1: predicted precision 0.92487 against reported 0.9249.

The pilot suite's known answer-token casing drift does **not** apply — `pope.csv` is text-scored after
lowercasing, so it absorbs the flip entirely. Only logit-based readouts are affected.

| arm | cells | mean R² | min R² | mean d_a | **sd d_a** |
|---|---|---|---|---|---|
| sequential | 4 | **0.9951** | 0.9879 | 2.2051 | **0.0222** |
| anchor (v2) | 4 | **0.5332** | 0.0826 | 1.6241 | **0.7947** |
| JOINT | 1 | 0.9995 | — | 2.1584 | — |
| ER | 1 | 0.9967 | — | 2.2278 | — |
| anchor v1 | 1 | 0.5273 | — | 2.6328 | — |

**It replicates, in both directions, on a second benchmark.** Sequential fits one ROC (0.9951 against
UCIT's 0.9569) and holds d_a tight (0.0222 against 0.0402); the anchor breaks coherence (0.5332 against
0.5857) and its d_a spread is **larger than UCIT's** (0.7947 against 0.3618). The two independent
task sequences agree on the sign, the ordering and the rough magnitude.

**What this is NOT.** Same backbone — so it is a second *benchmark*, not a second *model*, and it does not
answer the generality-across-models question that the Qwen arms were meant to answer. It must never be
presented as if it did.

**Bounds on it.** Four operating points per cell against UCIT's six, so the fits are weaker; four cells per
arm against nine; ER, JOINT and anchor-v1 have a single cell each and are quoted without intervals. The
between-arm contrast carries this, not any absolute R².

**Consequence.** The paper can state that the coherence result holds on two independent task sequences.
That is a materially stronger position than one benchmark, and it costs nothing — but it does not
substitute for the second backbone, and the limitation stands.

---

## 2026-09-14 — The second backbone is not achievable on UCIT, at any Qwen scale

Zero-shot exact-match on UCIT val, n = 20 per task, with a grey-image **blind** control. Both models
verified loaded (7B = 5 checkpoint shards, 3B = 2).

| task | 7B sighted | 7B blind | 3B sighted | 3B blind |
|---|---|---|---|---|
| ArxivQA | 95% | **95%** | 90% | **95%** |
| CLEVR-Math | 100% | 20% | 90% | 20% |
| IconQA | 70% | 30% | 35% | 10% |
| ImageNet-R | 25% | 0% | 15% | 0% |
| Flickr30k | 0% | 0% | 0% | 0% |
| VizWiz | 0% | 0% | 0% | 0% |

**Two caveats on our own probe, stated before the conclusion.** Exact-match scoring is inappropriate for
captioning (Flickr30k) and for VQA with many acceptable answers (VizWiz), so **those two rows are
uninformative** — the 0% is our metric, not the model. And n = 20 per task is small; these are coarse
readings, adequate for a ceiling/no-ceiling call and nothing finer.

**The finding.** ArxivQA and CLEVR-Math are at ceiling for **both** Qwen scales, and they are exactly
stages 1 and 2 of ordering o1 — which is precisely where the training gate fired. Reducing model scale
from 7B to 3B does **not** fix it: ArxivQA stays at 90–95%. This is a property of the Qwen family on these
tasks, not of model capacity, so no smaller Qwen rescues the design.

**ArxivQA is 95% solvable BLIND at both scales** — the model does not need the image. That is a leakage
result about the benchmark, not about the model.

### Decision: ship with one backbone

Running the arms anyway would mean a continual-learning sequence whose first two stages teach the model
nothing, which is scientifically empty for those stages and indefensible under review. Switching families
(InternVL) needs a new backbone class written and validated with eleven days left — the wrong risk.
**The single-backbone limitation stands and the paper states it plainly.**

Partial mitigation already in hand: the coherence result **replicates on a second task sequence** (the
pilot suite, 2026-09-13 entry). That is a second benchmark, not a second model, and must be scoped as such.

### This is a contribution, not only a setback

UCIT was selected for this project because it has **low zero-shot leakage for LLaVA-class models**. It does
not have that property for Qwen. So **a continual-learning benchmark's suitability is a property of the
model–benchmark pair, not of the benchmark**, and a suite chosen for low leakage against one family can
leave a stronger family at ceiling — at which point "continual tuning" updates nothing and any forgetting
metric measures noise. We have the measurement to support that, including the blind control showing one
task needs no image at all.

**Procedure adopted:** run the ceiling probe on any candidate backbone **before** queueing a training arm.
Twenty minutes against eighteen hours; it would have saved two days here.

---

## 2026-09-15 — Running the calibrated-target arm (`fsL_critbal_o1_s17`, job 20654)

**Why this arm and not a new one.** The open question the paper leaves is whether criterion drift can be
suppressed *without* sacrificing placement — the anchor achieves the first and fails the second. The
natural test is to anchor toward a calibrated target rather than the frozen base. `critbal` already is
that arm: it drives the yes-rate on a balanced probe to 0.5, which corresponds to c ≈ 0, against the
JOINT endpoint of c* = +0.0878.

| target | criterion | corresponding yes-rate at d_a = 2.159 | distance from c* |
|---|---|---|---|
| frozen base (what the anchor uses) | +0.4314 | 0.4035 | **0.344** |
| `critbal` (balanced probe) | ≈ 0 | 0.5000 | **0.088** |
| JOINT endpoint c* | +0.0878 | 0.4804 | — |

So `critbal`'s target sits **four times closer** to the empirical optimum than the anchor's. No new arm was
needed, and building one would have meant new untested code on a scarce GPU slot.

**Two bugs fixed before it could run.** It had failed twice, each inside 20 seconds:
1. A bare sibling import (fixed earlier).
2. `train_lora.py` unpacked `crit.base_summary()` into two values. That contract holds for the
   base-anchored preservers, which return `(mean, std)` of cached base statistics — but
   `CriterionBalanced` has **no base statistics at all**, since its target comes from the probe's 50/50
   balance rather than the frozen model, and it returns a descriptive dict. `ValueError: too many values
   to unpack`. The call site now accepts both shapes.

   The same edit fixes a second latent fault: the stage-1 identity assertion ("live and base statistics
   disagree") is meaningful only when the target *is* the frozen base. Under a balanced-probe target the
   base sits at c = +0.431, so a non-zero stage-1 penalty is correct rather than a failure. That check is
   now conditioned on the arm being base-anchored.

**Resource note.** The disk guard (55G minimum) correctly refused the submission at 37G free. Space was
reclaimed by deleting 39G of Qwen2.5-VL weights across two caches — a backbone abandoned on measured
evidence (2026-09-14 entry) — after confirming zero open file handles on all three trees. LLaVA, the
UCIT data and the balanced probe were verified intact afterwards. 75G free at submission.

**What the outcome means either way.** If the criterion lands near c* with drift suppressed, then
stability and correctness are jointly attainable and the paper's open problem is answered rather than
merely posed. If it suppresses drift and still lands badly, the difficulty is not the target but the
mechanism, which is a stronger negative result than the one currently reported. Both are worth having;
neither is assumed.

---

## 2026-09-16 — Experiments queued in response to the external review

An outside review scored the draft 4–5 and identified the two dominant weaknesses as **no
continual-learning baselines** and **one backbone**. Both are compute, not writing. Queued:

| arm | job | what it decides |
|---|---|---|
| `fsL_critbal_o1_s17` | 20654 (running) | whether a calibrated target suppresses drift *without* mis-placing the criterion |
| `qwen_loss` probe | 20678 | which UCIT tasks Qwen can actually learn, measured as pre-train loss |
| `fsL_ewc_o1_s17` | 20673 | a standard CL baseline's criterion trajectory |
| `fsL_lwf_o1_s17` | 20675 | the baseline closest in spirit to the anchor |
| `fsL_ewc_o1_s23` | queued | replication |

**On the second backbone, the earlier abandonment was too quick.** The 2026-09-14 entry
concluded Qwen was unreachable "at any scale", which is right for the full six-task suite but
does not follow for a reduced one. The review's suggestion — a different suite or dropping
the ceiling tasks — is better than dropping the backbone.

The earlier ceiling probe also had a defect worth recording: it scored **exact-match
accuracy**, which is meaningless for captioning (Flickr30k) and for VQA with many acceptable
answers (VizWiz). Those two rows read 0% and were treated as uninformative, correctly, but
that left the viability of a four-task sequence unknown. Job 20678 measures **pre-train
per-token loss per task instead**, which is the quantity the training gate actually tests, so
it predicts directly whether a reduced-suite Qwen run would clear the gate rather than
inferring it from accuracy.

**What the review got right that we had not seen.** The z-ROC coherence result is *not*
continual-learning-specific: the matched joint arm coheres too (R² 0.945), which the paper
itself says "must" be so "if the property belongs to the measurement rather than to
sequential training". So coherence is a property of the instrument, and the only CL-specific
result is the task-recency finding of Section 4. That section has been promoted into the
abstract accordingly.

**And the title was indefensible.** The paper documents sequential endpoint d′ falling 0.121
at eleven standard errors, negative in 9/9 cells, and Section 3 withdraws "d′ never moves" —
while the title asserted the curve does not move. Retitled to *The Criterion Moves More Than
the Curve*, and the abstract now states the d′ drift before the equivalence bound rather than
behind it.

---

## 2026-09-16 — The calibrated-target arm lands: `fsL_critbal_o1_s17`, 6/6 evaluated

Job 20654 trained all six stages (losses 5.30 / 12.48 / 53.71 / 13.24 / 8.99 / 58.31, tracking the
completed anchor cell's 6.86 / 12.73 / 54.36 / 14.04 / 9.63 / 58.69 within a few percent at every
stage) and job 20655 evaluated all six. **EVAL_DONE = 6/6, so it is quotable.**

Matched comparison, all three on the same cell (o1/s17), placement scored as |c − c*| against the
joint arm's empirical endpoint c* = +0.0878:

| arm | path Σ\|Δc\| | post-settling | endpoint c | **\|c − c*\|** | endpoint F1 |
|---|---|---|---|---|---|
| sequential (no intervention) | 0.7643 | 0.5451 | +0.0679 | **0.0199** | **0.8604** |
| anchor (frozen-base target) | 0.7834 | 0.4618 | +0.7626 | 0.6748 | 0.7844 |
| **critbal (calibrated target)** | **0.3030** | **0.2836** | +0.2456 | 0.1578 | 0.8380 |

### What it establishes

**Correcting the target recovers most of the anchor's placement failure.** Placement error falls from
0.6748 to 0.1578, a factor of **4.3**. The 2026-09-09 diagnosis — that the anchor fails because it
inherits the frozen base's mis-placed c = +0.431, not because the mechanism is broken — is supported.

**And it produces the lowest drift of any arm measured.** Post-settling 0.2836 against sequential's
0.5451 and the anchor's 0.4618; the path is 0.3030 against 0.7643 and 0.7834. Drift suppression is
real and larger than the anchor's.

### What it does not establish

**It still does not beat doing nothing.** Sequential places 7.9× better (0.0199 against 0.1578) and
retains a higher endpoint F1 (0.8604 against 0.8380). So the paper's thesis stands, with a sharper
and better-controlled demonstration: an intervention can secure stability of the threshold and still
fail to secure its placement, even when the target has been corrected.

**Why the residual gap is not a surprise.** critbal targets a yes-rate of 0.5, which equals c = 0
only under equal variance; our fitted z-ROC slopes are 0.55–0.80. The realized endpoint is +0.2456
rather than ~0, and the module docstring recorded this approximation before the arm ran. Part of the
remaining 0.158 is the approximation in the target itself, which is a stated and checkable cause
rather than an unexplained residual.

**n = 1 cell.** `fsL_critbal_o1_s23` is queued. Nothing here may be given an interval, and the
comparison is legitimate only because all three arms are the same cell.

---

## 2026-09-16 — correction to the entry above, and a CL baseline that was already on disk

### Correction: critbal does NOT have "the lowest drift of any arm"

The entry above says critbal "produces the lowest drift of any arm measured." **That is wrong and is
withdrawn.** It compared critbal only against sequential and the anchor. Scored on the same basis, both
JOINT cells drift less: `fsL_joint_o1_s17` **0.2400** and `fsL_joint_o1_s23` **0.2157**, against critbal's
0.2836. What is true, and all that may be said: **critbal has the lowest drift of any arm that sees the
stream sequentially**, and it comes within 0.044 of the JOINT bound on the matched cell — closing
**85.7%** of the SEQ→JOINT gap on that cell against the anchor's 27.3%.

### All numbers in this entry come from the canonical scorer, with a passing positive control

Everything below is `fs_common.score_pope` (the same function behind every published number), run over
`pope_gen.jsonl`, filed with provenance at `analysis/readout/matched_cells_critbal_ewc.json`. A first
ad-hoc scorer written this session disagreed by ~0.01 on path sums; the cause was that it coerced
**parse failures** to "no" instead of excluding them (stage k5 carries 66 failures in seq, 27 in anchor,
3 in critbal). Its numbers were discarded. The canonical scorer reproduces the published values exactly
— seq o1/s17 **0.5451**, anchor o1/s17 **0.4618**, seq o1/s31 **1.0846**, JOINT mean **0.2279** against
the published 0.228 — so the readout is trusted only because that control passed.

### `fsL_ewc_o1_s31` has been complete since 2026-09-09 and was never scored

Job 19027, 6/6 evaluated, `ewc.py` unchanged since 2026-09-04. It appears in the 2026-09-13 casing audit
("all 54+ cells across anchor / seq / ewc") and then was never read out. The paper currently says the
pre-registered baselines "are not evaluated here." **One of them was.** Matched seq and anchor cells for
the same ordering and seed both exist, so this is a legitimate matched triple at n=1.

| o1 / seed 31 | post-settling | endpoint $c$ | $\|c-c^*\|$ | endpoint $F_1$ | $\max\|\Delta d'\|$ |
|---|---|---|---|---|---|
| sequential | 1.0846 | +0.2607 | 0.1729 | 0.8547 | 0.0982 |
| **EWC** | **0.9166** | **+0.1982** | **0.1104** | **0.8573** | **0.0766** |
| anchor | 0.4099 | +0.7405 | 0.6527 | 0.7896 | 0.0926 |

**The finding is not that EWC fails. It is that EWC does not see the criterion at all.** Stage for stage
its trajectory lies on top of the untreated one:

```
seq  c: [0.3139, 0.2455, 0.4034, 0.2375, -0.0971, 0.2607]
ewc  c: [0.3352, 0.2722, 0.3927, 0.2227, -0.0711, 0.1982]
```

Pearson $r = 0.9794$; mean absolute difference **0.0270**, which is **5.4%** of that cell's own criterion
range (0.5005). Same shape, same sign change at stage 5, same recovery. For scale, the masked-image
anchor on the identical cell decorrelates the trajectory completely ($r = 0.044$) and displaces it by
**107%** of the range. EWC buys a 15.5% drift reduction as a side effect and leaves placement, sensitivity
and endpoint $F_1$ where they were.

This is the cleanest statement of the paper's thesis yet, and it comes from a standard baseline rather
than from our own probe: **weight-space regularization protects what it was designed to protect, and the
decision criterion is not in that set.**

### What this does not license

**n = 1 cell.** No interval, no "EWC does not control drift" in general — one ordering, one seed. Job
20673 (`fsL_ewc_o1_s17`) is training now and makes it n=2; `lwf` is queued behind it. A cross-seed
comparison is NOT admissible here and one was briefly made and caught in this session: reading EWC's
0.9166 (s31) against sequential's 0.5451 (s17) suggested EWC drifts *more* than doing nothing. Matched,
it drifts *less*. Only same-cell contrasts may be quoted.

### 2026-09-16, same day — a mismatched-statistic error caught by checking the draft against the JSON

The draft said the calibrated-target arm "recovers $4.3$ of the anchor's $6.6$-point $F_1$ deficit."
**Both numbers were wrong together**, and wrong in the project's recurring way: $6.6$ points is the
*nine-cell mean* deficit (SEQ 0.8625 against anchor 0.7963, Table 2), while the recovery is a *single
cell*. On the matched o1/s17 cell the deficit is **7.6** points (0.8604 against 0.7844) and the arm
recovers **5.4** of them (0.8380). Corrected in the paper and in the note for Prof. Park.

This is the same failure the paper warns about in its own $2.5\times$ discussion --- a quantity computed
after averaging cells compared against one computed within a cell. It was found only because every
number in the new table and its prose was checked programmatically against
`analysis/readout/matched_cells_critbal_ewc.json` rather than re-read. The table itself was correct;
the prose was not. **Check prose against the readout, not only tables.**

Also corrected: the caption claimed no row's $\max|\Delta d'|$ exceeds $0.134$; the true maximum is
$0.1343$ (sequential o1/s17). Now stated as $0.135$. The pre-registered falsifier is $0.30$, so the
conclusion is unaffected, but the bound as written was false.

---

## 2026-09-16 — the Qwen feasibility probe returns, and it contradicts our own limitation

Job 20678, `clh_qwen_loss`, COMPLETED in 3m22s. It measures Qwen2.5-VL-7B's **pre-training per-token
loss on each UCIT task** — the exact quantity the training gate tests, rather than the accuracy and
blind-image evidence the earlier abandonment rested on.

| task | median | min | max | verdict |
|---|---|---|---|---|
| ArxivQA | 0.13616 | 0.00969 | 0.54333 | trainable |
| CLEVR-Math | **0.00043** | 0.00002 | 0.00106 | **AT CEILING (gate fires)** |
| Flickr30k | 4.80413 | 2.51867 | 5.45080 | trainable |
| IconQA | 0.51627 | 0.07472 | 5.84173 | trainable |
| ImageNet-R | 2.28884 | 1.77843 | 4.01358 | trainable |
| VizWiz | 4.70048 | 3.21168 | 6.04454 | trainable |

**Five of six tasks are trainable. Only CLEVR-Math fires the gate.**

### This falsifies the limitation as written

The paper says Qwen "is already at ceiling on **two** of this suite's tasks --- the **first two stages**
of our ordering --- so a run there begins with two stages that teach nothing." The canonical o1 ordering
is ArxivQA → CLEVR-Math → Flickr30k → IconQA → ImageNet-R → VizWiz, and ArxivQA's median loss is
**0.136**, nowhere near the 0.02 gate. So the claim is wrong in both of its parts: one task is at
ceiling, not two, and it is the second stage, not the first two.

The honest consequence is worse for us and is now what the paper says: **a five-task Qwen sequence is
viable and we did not run it.** The second backbone is open, not foreclosed. Dropping CLEVR-Math gives
ArxivQA → Flickr30k → IconQA → ImageNet-R → VizWiz.

### Caveat on the probe itself

The log carries three non-fatal `CUDACachingAllocator` OOM-retry warnings. The job exited `0:0` and
produced a complete six-task table with plausible spreads, so the measurement stands, but it was taken
under memory pressure and a rerun before any Qwen arm is launched would be cheap insurance.

**What this does not license.** It says the tasks are *learnable*, not that the criterion result
replicates on Qwen. No cross-model claim may be made from a loss probe.

---

## 2026-09-16 — two gaps between what the paper promises and what the release contains

Found by checking the public repository against the Reproducibility Statement rather than assuming.
`gh api .../git/trees/main?recursive=1` lists **153 files**: `analysis/` 68, `pilot/` 56, `fullstudy/`
20, `docs/` 5. Last push **2026-09-14 15:36 UTC**.

### 1. The generation logs are promised and are not there

Appendix item (iii) promises "per-checkpoint generation logs (JSONL with audit fields) and summary CSVs
for all pilot checkpoints (21 primary-arm plus 26 robustness) and all 120 LLaVA/UCIT full-study
stage-cells." **The repository contains zero `*_gen.jsonl` files.** At ~3.5 MB per POPE stage-cell this
is roughly half a gigabyte, which is plausibly why they were left out, but the paper states them as
released. Before submission this needs either a host (LFS, Zenodo, a release asset) or a narrowed
claim. It is checkable by any reviewer once the named version is public, so it should not be left.

The named readouts in item (v) --- `fs_aggregate.json`, `zroc_coherence.json`, `order_effects_*.json`,
`pope_strata_seq.json` --- **are** all present, so the CPU-rederivation claim holds for the full-study
matrix.

### 2. The repository predates tonight's results

Fourteen tracked files have changed since the last push, including the readout behind every new number
(`analysis/readout/matched_cells_critbal_ewc.json`), the `train_lora.py` interface fix that let the
calibrated-target arm run at all, `mechanism_readout.py`, and this file. Until it is pushed, "the
machine readouts behind every number here" is false for the EWC and calibrated-target numbers.

**Not pushed tonight, deliberately.** The original export excluded internal notes under rules this
session does not know, and this file now carries candid correction logs. Whether they go public is
Alex's call, not a default.

### 3. A stale source comment nearly cost us a verified citation

`main.tex` carried the note "the invented \citet{cha2025diverse} was removed on 2026-09-11". That was
true for about an hour: the key was fabricated, the real paper was then found, and `refs.bib` now holds
the canonical PMLR record verbatim (CoLLAs 2025, PMLR v274, `proceedings.mlr.press/v274/cha25a.html`,
HTTP 200, same day). The citation is legitimate and is cited three times. The comment has been replaced
with an explicit do-not-delete note. **A correction log that stops at the retraction and omits the
repair is worse than none.**

### The release's inclusion rules, recovered (there is no export script)

Nothing in the tree builds the public repository, so the rules were reconstructed by diffing its file
list against `git ls-files`. **153 released against 2333 tracked.** Worth writing down because the next
sync would otherwise be guesswork.

*Excluded wholesale:* `paper/` (all 30 files), `design_notes/`, `results/` (660), `CLAUDE.md`,
`DECISIONS.md`, `NEXT_EXPERIMENTS.md`, `WRITING_PHASE_HANDOFF.md`, `ICLR2027_SUBMISSION_CONSTRAINTS.md`,
`related_work.md`, and 1387 of `analysis/` (everything under `analysis/diag/` and `analysis/out/`).
*Included:* the core `analysis/` scripts plus all of `analysis/readout/`, `pilot/`, and part of
`fullstudy/`.

*Added at export time, existing nowhere locally:* `LICENSE`, `requirements.txt`, and five `docs/*.md`
— including **`docs/PRE_REGISTRATION.md`**, which is a derived, cleaned form of this file.

**That last point is the whole reason tonight's work was not pushed.** Syncing means regenerating
`docs/PRE_REGISTRATION.md` from a file that now contains candid correction logs — the withdrawn critbal
overstatement, the mismatched-statistic error, the cross-seed comparison that reversed a sign. Some of
that is exactly what a reproducibility-minded reader should see, and some of it is internal. Which is
which is Alex's call, and it cannot be made by a script.

Minimum sync for the paper's claims to be true: add `analysis/readout/matched_cells_critbal_ewc.json`,
refresh `analysis/mechanism_readout.py` and `pilot/train_lora.py`, and regenerate
`docs/PRE_REGISTRATION.md` under whatever rule is chosen.

---

## 2026-09-16 — every headline number recomputed from `fs_aggregate.json`, independently

Not re-read from the readout files that produced them: reimplemented from the per-stage POPE record and
compared. All of it passes.

| claim | paper | recomputed | |
|---|---|---|---|
| criterion vs $d'$, full path | $0.936 / 0.380 = 2.46\times$ | $0.936 / 0.380 = 2.46\times$ | ✓ |
| criterion vs $d'$, within-cell range | $0.382 / 0.144 = 2.66\times$ | $0.382 / 0.144 = 2.66\times$ | ✓ |
| anchor closes SEQ→JOINT gap | $75\%$ | $75.4\%$ | ✓ |
| per-ordering closure | $62$–$86\%$ | $62.0 / 67.6 / 86.4\%$ | ✓ |
| Table 2 post-settling | $0.781 / 0.364 / 0.228$ | identical | ✓ |
| Table 2 mean $|c|$ | $0.196 / 0.731 / 0.099$ | identical | ✓ |
| $\eta^2$ task identity | $0.731$ | $0.731$ | ✓ |
| $\eta^2$ sequence depth | $0.049$ | $0.049$ | ✓ |
| permutation $p$ (depth) | $0.73$ | $0.724$ | ✓ |

The $\eta^2$ pair was rebuilt from scratch --- 45 post-settling transitions regrouped by `task_trained`
and by stage index, sums of squares and a fresh 5,000-shuffle permutation null --- rather than read from
`order_effects_seq_ps.json`. It reproduces both shares exactly. $p(\text{task})$ comes back at the
$1/(B{+}1)$ floor, consistent with the paper's $<10^{-4}$ at $B = 20{,}000$.

### One thing this clarified about the $2.5\times$ figure

The path numbers only reproduce when the **base → $k_1$ step is included on both quantities**; the
post-settling path gives $0.781 / 0.303 = 2.58\times$ instead. Both are defensible, and the paper's is
the like-for-like one it claims to be --- one summary applied to $c$ and $d'$ over the same cells. Worth
recording because $0.936$ and the endpoint's $0.781$ are both "the sequential path" under different
conventions, and mixing them would produce a wrong ratio silently.

---

## 2026-09-16 — the calibrated-target arm trips the sensitivity margin, and I had hidden it behind the wrong statistic

Found while recomputing Table 2's last row. It reads "Mean per-cell $\max|\Delta d'|$", which looks like
a stage-to-stage change. It is not. Recomputing stage-to-stage gives $0.119/0.107/0.045$ against the
published $0.175/0.100/0.102$; recomputing as $\max_k |d'_k - d'_{\mathrm{base}}|$ reproduces all three
exactly. **The published row is displacement from base** --- correctly, since the pre-registered F1
falsifier is "pooled $d'$ moving more than $0.30$ \emph{from base} at any sequential checkpoint", and
SEQ's $0.204$ is exactly the number the paper quotes for it.

### The error

The Table 3 I added last night computed `max_abs_ddprime` **stage-to-stage**, and its caption then said
"No row's $\max|\Delta d'|$ exceeds $0.135$, inside the pre-registered $0.30$ falsifier." Same notation,
different quantity, and a falsifier invoked on a statistic it is not defined over.

On the correct quantity:

| row | stage-to-stage | **from base** |
|---|---|---|
| seq o1/s17 | 0.1343 | 0.2041 |
| anchor o1/s17 | 0.0770 | 0.0771 |
| **critbal o1/s17** | 0.0931 | **0.3305** |
| JOINT o1/s17 | 0.0528 | 0.1061 |
| seq o1/s31 | 0.0982 | 0.1722 |
| EWC o1/s31 | 0.0766 | 0.1794 |
| anchor o1/s31 | 0.0926 | 0.0687 |

**The calibrated-target arm reaches $0.3305$ at stage 5 --- outside the $\pm0.30$ margin.** Its $d'$ at
that stage is $2.0172$ against a base of $2.3477$. Every other row is comfortably inside.

### What it changes

Not the placement result: $0.675 \to 0.158$ stands, computed on the criterion. What it changes is the
reading. Last night's entry said the arm suppresses drift and improves placement with sensitivity
untouched. **That was true only under the statistic I had chosen and false under the pre-registered
one.** The arm buys its placement partly by moving discrimination, and the paper and appendix now say so
in both places. Its mean $d'$ over the cell ($2.107$ against sequential's $2.243$) points the same way.

This is the project's own estimator-matching rule broken by the person who wrote it down, one day after
recording a mismatched-statistic error of exactly the same shape. **The rule is not "match estimators";
it is "when reusing a notation, recompute what the original meant before quoting a threshold against
it."**

---

## 2026-09-16 — the EWC replication lands, and it qualifies the first cell rather than confirming it

`fsL_ewc_o1_s17`: training 4h29m (job 20673), evaluation 2h14m (job 20674), both `COMPLETED 0:0`,
**EVAL_DONE 6/6**. Scored with `fs_common.score_pope`; the matched seq and anchor cells reproduce their
published values exactly, so the control passes.

| o1 / seed 17 | post-settling | endpoint $c$ | $\|c-c^*\|$ | endpoint $F_1$ |
|---|---|---|---|---|
| sequential | 0.5451 | +0.068 | 0.020 | 0.8604 |
| **EWC** | **0.4963** | +0.134 | 0.046 | 0.8622 |
| anchor | 0.4618 | +0.763 | 0.675 | 0.7844 |

### What replicates

The direction, and the separation from the probe. EWC tracks the untreated trajectory in both cells
($r = 0.979$, $0.913$) while the anchor does not on the same cells ($r = 0.044$, $0.051$). Drift
reduction is modest in both (15.5%, 9.0%). Endpoint $F_1$ moves by under 0.003 either way. Sensitivity
stays inside the pre-registered margin in both.

### What does NOT replicate

**The tightness.** Mean absolute displacement is 5.4% of the cell's criterion range at seed 31 and
**21.6%** at seed 17 — a factor of four. Yesterday's entry called the trajectory reproduced "almost
exactly" and quoted 5.4% as if it characterised EWC. It characterised one cell.

**The placement direction flips.** EWC is better placed than sequential at seed 31 (0.110 against
0.173) and **worse** at seed 17 (0.046 against 0.020). With n=2 and opposite signs, the honest claim is
**no placement effect in either direction**, which is what the paper now says. This is why the earlier
single-cell "EWC also places better" reading had to go.

### The claim that survives at n=2

Not "EWC reproduces the trajectory almost exactly." Rather: **EWC leaves the shape of the criterion
trajectory substantially intact and separates sharply from an intervention that targets the criterion
directly**, buying only 9–16% drift reduction with no consistent placement effect. The qualitative
contrast — $r > 0.91$ against $r \approx 0.05$, displacement under 22% against over 100% — is the part
that is stable across both cells, and it is enough for the reading that a weight-space penalty does not
see the decision threshold.

Two cells of **one ordering** (o1). Nothing here speaks to o2 or o3.

---

## 2026-09-16 — LwF lands, and it is the arm that does both

`fsL_lwf_o1_s17`: training 4h00m (job 20675), evaluation 2h15m (job 20676), both `COMPLETED 0:0`,
**EVAL_DONE 6/6**. Matched seq and anchor cells reproduce their published values, so the control passes.
**The queue is now empty.**

| o1 / seed 17 | drift | endpoint $c$ | $\|c-c^*\|$ | endpoint $F_1$ |
|---|---|---|---|---|
| JOINT | 0.2400 | +0.062 | 0.0255 | 0.8682 |
| calibrated target | 0.2836 | +0.246 | 0.1578 | 0.8380 |
| **LwF** | **0.3819** | +0.092 | **0.0040** | 0.8618 |
| anchor | 0.4618 | +0.763 | 0.6748 | 0.7844 |
| EWC | 0.4963 | +0.134 | 0.0458 | 0.8622 |
| sequential | 0.5451 | +0.068 | 0.0199 | 0.8604 |

**LwF cuts drift 29.9% and lands 0.0040 from $c^*$ — nearer than any other arm measured, the joint
bound included — with sensitivity untouched** ($\max_k|d'_k-d'_{\mathrm{base}}| = 0.161$; mean $d'$
2.260 against sequential's 2.243). It is the first arm here that suppresses drift *and* does not
mis-place the criterion.

### What it licenses

Existence, and it strengthens rather than threatens the paper's thesis. The paper argues the anchor's
placement failure is a property of **its target** and not a price stabilization must pay, citing the
joint arm as proof both are jointly attainable. LwF is a second proof, and a better one: the joint arm
is an upper bound that sees the whole stream, while LwF runs on the stream in order. A standard
distillation baseline does what our masked-image probe was built to do and does not.

### What it does not license

**One cell.** And placement is the half to be careful with: sequential endpoints span $[-0.014,
+0.261]$ across the nine cells, so landing 0.004 from $c^*$ on one draw is weak evidence about
placement in general. The 29.9% drift reduction is the firmer half. No claim that LwF beats the anchor
in general, and nothing about o2 or o3.

**Final experimental state.** Nine SEQ, nine anchor, two JOINT, two EWC, one LwF, one calibrated-target
cell, plus the Qwen loss probe. ER and the second backbone remain unrun.
