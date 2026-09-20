# Faithfulness Anchor (C2) — Masked-Image Contrastive Faithfulness Loss

Formal writeup of the method prototype implemented in `faith_loss.py` +
`build_grounding_pairs.py`. Candidate C2 from
`design_notes/method_differentiation.md` (the only candidate besides C6 judged
DISTINCT from all three comparators at the equation level). Status: prototype;
enters the pipeline only after the measurement gate (`pilot/PREREGISTRATION.md`)
passes.

## Loss

Anchor set `\mathcal{P}=\{(I_i, I_i^{\setminus o^{+}_i}, o^{+}_i, o^{-}_i)\}_{i=1}^{N}`
(`N=2000`), built by `build_grounding_pairs.py`: `I` a COCO val2014 image disjoint
from the POPE and CHAIR eval pools; `o^{+}` a class genuinely present (largest
instance bbox `\geq 5\%` of image area); `I^{\setminus o^{+}}` the counterfactual with
every instance bbox of `o^{+}` filled with the dataset mean color; `o^{-}` a class
with no instance in `I`, sampled preferring high co-occurrence with the image's
present classes (hard negative, motivated by the co-occurrence-bias finding of
arXiv:2508.04567).

For query prompt `q_{o}` = `"USER: <image>\nIs there a {o} in the image? ASSISTANT:"`,
let `z^{\rm yes}_{\theta}(I,q_{o})` and `z^{\rm no}_{\theta}(I,q_{o})` be the model's
next-token logits of the single tokens `" Yes"` / `" No"` at the last prompt
position. With `\mathrm{sp}(u)=\log(1+e^{u})` and margin `m`:

```latex
\mathcal{L}_{\rm faith}(\theta)=\mathbb{E}_{(I,I^{\setminus o^{+}},o^{+},o^{-})\sim\mathcal{P}}\Big[
\mathrm{sp}\big(m-\big[z^{\rm yes}_{\theta}(I,q_{o^{+}})-z^{\rm yes}_{\theta}(I^{\setminus o^{+}},q_{o^{+}})\big]\big)
+\mathrm{sp}\big(m-\big[z^{\rm no}_{\theta}(I,q_{o^{-}})-z^{\rm yes}_{\theta}(I,q_{o^{-}})\big]\big)\Big]
```

estimated per optimizer step by `k` pairs sampled without replacement (seeded).
Full objective during each CL stage:

```latex
\min_{\phi}\;\mathcal{L}_{\rm task}(\phi)+\lambda_{\rm faith}\,\mathcal{L}_{\rm faith}(\phi)
```

with `\phi` the LoRA + projector parameters (identical trainable set as the task
loss; no extra parameters, no frozen copies).

Term 1 (counterfactual sensitivity): the yes-logit must drop by at least `m` when
the present object is masked out. Term 2 (hard-negative discrimination): on the
untouched real image, the no-logit must exceed the yes-logit by at least `m` for
the absent co-occurring object. Both terms saturate at `\mathrm{sp}(m-\Delta)\to 0`
once the margin `\Delta` exceeds `m`, so a model that is already grounded pays
approximately nothing — the loss is an anchor, not a permanent drag on plasticity.

Relation to the C2 form in the dossier: the dossier wrote
`-\log\sigma(\Delta)=\mathrm{sp}(-\Delta)`; the implemented
`\mathrm{sp}(m-\Delta)` is its margin-shifted generalization (`m=0` recovers it
exactly), and the absent-object term is an addition. Neither change introduces a
teacher, a reference policy, or a preference pair, so the differentiation
verdicts below are unchanged.

## What it anchors, against what reference

- **Anchored quantity:** the behavioral grounding margin of the yes/no answer
  logits (LM-head outputs) on counterfactual image pairs — exactly the quantity
  POPE-style discriminative evals measure, made differentiable.
- **Reference:** ground-truth COCO instance annotations. The reference is
  *data-side*, not a model: no previous-stage checkpoint, no base-model teacher,
  no early-run snapshot appears anywhere in the loss. This is the key
  equation-level distinction from all three comparators.
- **Data:** a fixed, stage-independent anchor set, disjoint by construction from
  the POPE image_sources and the CHAIR image ids (asserted in
  `build_grounding_pairs.py`), so the training signal never touches the
  measurement pools.

**Deviation from the dossier's C2 data spec.** The dossier sketched "a small
probe set built from current-stage images." Implemented instead: a fixed COCO
val2014 anchor set. Reasons: (i) the pilot's stage datasets
(ScienceQA/TextVQA/Flickr30k/VizWiz) carry no instance annotations, so
ground-truth counterfactuals are not constructible from them; (ii) a fixed set
keeps the faithfulness axis constant across stages, which is what
"stability–plasticity–faithfulness" needs for a clean third axis; (iii)
eval-disjointness is enforceable once, globally. Cost of the choice: the anchor
is COCO-domain, the same domain as both hallucination evals — gains may be
partly domain-specific. The AMBER extension in the full study
(`design_notes/generative_metrics_extension.md`) is the transfer check.

## Hyperparameters (ALL TUNE-ME — priors, no sweep has been run)

| Name | Default | Role | Note |
|---|---|---|---|
| `k_pairs` | 4 | pairs sampled per optimizer step (3k forwards) | tune-me; compute/variance trade-off |
| `margin` (`m`) | 2.0 | logit-margin target | tune-me; sets where the loss saturates |
| `\lambda_{\rm faith}` | 0.1 | weight vs task loss | **tune-me first** — no evidence yet that 0.1 is even the right order of magnitude |
| `N` pairs | 2000 | anchor-set size | fixed by build script |
| `MIN_AREA_FRAC` | 0.05 | present-object min bbox area | build-time |
| mask fill | dataset mean color | counterfactual construction | build-time; see limitations |

## Differentiation (reusing the verdicts of `design_notes/method_differentiation.md`)

**vs RCL's `\mathcal{L}_{\rm rel}` (arXiv:2607.02020v1, their Eq. 7) — DISTINCT.**
`\mathcal{L}_{\rm rel}` is a JS divergence between softmax-normalized
counterfactual channel-reliance simplex vectors, gated by teacher confidence,
with the *previous-stage checkpoint* `\theta^{-}` as reference, on
*current-stage* samples; it is explicitly preservation-symmetric across channels
("does not impose a fixed preference for visual, textual, or OCR evidence").
`\mathcal{L}_{\rm faith}` shares none of these: the quantity is a
ground-truth-supervised answer-logit margin (no simplex, no JS, no
normalization over channels), the reference is the annotation (no model
reference at all, hence nothing to gate), and the data is a fixed eval-disjoint
anchor set. The deeper contrast: RCL masks channels to *measure* reliance and
then preserves whatever profile the teacher had; we mask objects to *construct
labels* for a directional grounding constraint. It is also not RCL's Eq. 8
(`\mathcal{L}_{\rm pred}`: token-level KD to `\theta^{-}` — we distill nothing)
nor its Eq. 9 `\lambda_{\rm reg}` term (parameter-space L2 — we penalize no
parameter drift).

**vs φ-DPO (arXiv:2602.22601v2, their Eq. 14/17) — DISTINCT.** φ-DPO's loss is
a focal-reweighted *sequence-level preference log-ratio* between one preferred
and one dispreferred full response, normalized by the previous-task policy
`\pi_{t-1}`; its negatives are LLM-fabricated, image-unverified "forgotten
response" simulations. Ours is a pointwise two-logit margin on a binary visual
query with *no* reference policy, and the negatives are *images* (counterfactual
masks) plus annotation-verified absent objects, not fabricated responses. There
is no `\pi_{t}/\pi_{t-1}` ratio anywhere to reweight.

**vs self-distillation of arXiv:2604.15574v1 (their Eq. 2) — DISTINCT.** Their
regularizer is temperature-scaled KD of the *full next-token distribution at
every non-pad position* to a frozen 1-epoch snapshot of the same run (or, in
their §3 variant, a hard parameter-subset freeze). We have no snapshot teacher,
no distillation target, no frozen subset; the anchored quantity is a 2-logit
margin against ground truth on held-out images, not the model's own earlier
distribution on the training batch.

**Scope of these verdicts.** The dossier clears only the three named comparators
plus arXiv:2508.04567 (whose POPEv2 is the *evaluation* analogue of our masking
construction — cite it as such; Obliviate is unlearning on mined spans, LM-head
only, not a margin loss). The nearest non-comparator line is
counterfactual-data-augmentation debiasing for *static* VLMs; the
CL-stage-coupled training use is the claimed slot. The two-disjoint-vocabulary
novelty sweep against the full literature is still owed at design freeze and
before submission (standing rule), and both comparator arXiv ids remain on
version-watch.

## Known limitations / risks

1. **Box-artifact cue.** Mean-color bbox fill removes the object but leaves a
   visible rectangle; the model could learn "filled box ⇒ say No" instead of
   grounding. Term 2 partially decouples this (its "No" supervision uses
   untouched images), but the risk stands; inpainting-based masking is the
   full-study upgrade if ablations show artifact reliance.
2. **Annotation noise.** "Absent" means unannotated, not verified-absent; COCO
   misses small/background instances. Hard-negative sampling makes collisions
   with co-occurring unannotated objects *more* likely, not less — accepted for
   the prototype, worth quantifying later.
3. **Answer-bias vs grounding** (ablation-instrument memory): a loss on yes/no
   logits can be satisfied partly by moving the global yes/no prior. The two
   terms pull in opposite directions on real images (yes for present, no for
   absent), which constrains a pure prior shift, but the yes-rate confound panel
   in the evals remains the arbiter.
4. **Stochasticity.** LoRA dropout is active during anchor forwards (train
   mode, deliberately un-mutated), so per-step values are noisy — a training
   signal, not a monitoring metric. Trajectories of `\mathcal{L}_{\rm faith}`
   should be read smoothed.
5. **Single prompt template.** One fixed query phrasing; template overfitting
   (margin satisfied only for this phrasing) is untested. POPE uses the same
   phrasing family, which cuts both ways: eval-aligned, but the AMBER/CHAIR
   generative metrics are the uncontaminated check.


## v2 amendment (2026-08-31)

The two-term loss above is the v1 formulation, falsified at pilot scale (see
analysis/METHOD_COMPARISON.md): its terms are asymmetric on the yes/no decision
axis and the F-arm swung hyper-conservative. The implemented method is now the
v2 three-term symmetric loss in faith_loss.py:

  L = softplus(m - (z_yes - z_no)) on real/present
    + softplus(m - (z_no - z_yes)) on real/absent
    + softplus(m - (z_yes(real) - z_yes(masked))) on present  (counterfactual sensitivity)

v1 is retained in the paper as the ablation demonstrating the symmetric
decision-axis design is load-bearing. Hyperparameters unchanged (lambda=0.1,
k=4, m=2.0).
