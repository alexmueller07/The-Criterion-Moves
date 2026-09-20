"""Label-free criterion CORRECTION via polarity antisymmetry (method candidate M6).

WHY THIS EXISTS (2026-09-09). The full study showed our anchor reduces criterion
DRIFT (post-settling Σ|Δc| 0.781 → 0.364) but lands at c = +0.697 while the
Bayes-optimal criterion on balanced POPE is 0. The consequence is severe and
uniform: endpoint POPE F1 0.7963 vs plain sequential's 0.8625, worse in 9/9 cells
with no overlap, while d' is FLAT across arms (2.35 vs 2.23). The method costs 6.6
F1 points relative to running nothing, purely through threshold placement.

The diagnosis is the target, not the mechanism. `crit_preserve.CriterionPreserver`
regularizes toward the FROZEN BASE's population mean of

    g(I, o; θ) = z_yes - z_no

and the base itself sits at c = +0.431. Pinning to a mis-placed reference inherits
and amplifies the mis-placement. Every label-free anchoring method in the vocab-A
sweep has this same structural defect: they can only PRESERVE a reference, never
CORRECT it, so none of them can be right unless the reference already was.

THE FIX. Ask each probe item in both polarities. If the model is logically
consistent, the truth signal reverses while a polarity-independent answer bias
does not:

    g_i^native  =  t_i + b
    g_i^flipped = -t_i + b
    =>  r_i = (g_i^native + g_i^flipped) / 2  =  b        <- NO LABELS USED

so b is estimable on unlabeled images, and the target b = 0 is correct BY LOGIC
rather than assumed from a reference that may itself be wrong. The penalty is

    mean mode:  L = ( mean_{i in S} r_i )^2          <- the criterion term
    item mode:  L = mean_{i in S} r_i^2              <- ABLATION: per-item
                                                       consistency, a strictly
                                                       stronger constraint that
                                                       also touches d'

Mean mode keeps critp's defining property, which is what differentiates this
family from LwF / RCL L_pred / LLaVA-c UIR: the gradient

    dL/dg_i^native = dL/dg_i^flipped = (1/k) * mean_j r_j

is IDENTICAL for every item in the batch. The term never asks any individual
decision to stay put; it only opposes a common polarity-independent offset.

THE PRE-REGISTERED FAILURE MODE, AND WHY IT IS MONITORED, NOT ASSUMED AWAY.
If the model does not process the polarity flip then g^flipped ≈ g^native, so
r_i ≈ g_i, and driving r to zero drives the whole decision statistic to zero --
destroying d' instead of centring the criterion. This is invisible in the loss
value itself: the penalty falls happily either way. So:
  * `analysis/antisymmetry_diag.py` gates this method on the BASE model before it
    is ever trained (PASS needs flip corr <= -0.30 with a positive control >= +0.60);
  * and this class records corr(g^native, g^flipped) into `history` at every
    logged step, so a run that starts healthy and then collapses toward polarity-
    blindness is detectable AFTER the fact rather than silently producing a
    d'-destroyed model that looks like a converged one.
A run whose flip correlation rises toward 0 during training is INVALID even if
its loss curve is beautiful. Check `history` before reporting any result.

Differentiation obligations (see related_work.md §9):
  * CCS (2212.03827, ICLR 2023) owns the antisymmetry CONSTRAINT ("a statement and
    its negation have opposite truth values") but trains a separate probe on frozen
    activations and mean-centres each branch -- which subtracts exactly the constant
    b we are trying to estimate. It never touches the output head and is not a
    training regularizer for the model itself.
  * PriDe (2309.03882, ICLR 2024) owns "use a task symmetry to estimate the answer
    prior label-free", but for MCQ option permutation, which does not exist for
    binary yes/no.
  * AdaPrior (CVPR 2026 Highlight) owns "track a model-induced prior online in
    continual learning" -- but over per-class softmax priors on CNN classification,
    with no decision criterion, no SDT, and no vision-language component.
  * VTI (ICLR 2025 Spotlight) adds a fixed direction to hidden states, which with a
    frozen head is a literal constant offset -- the nearest neighbour of the ANCHOR
    arm, not of this one, since ours estimates the offset from data rather than
    precomputing it.

Inherits every memory/serialization constraint from CriterionPreserver by
subclassing it: chunked forwards, fp32 statistic, in-context answer-token ids,
backbone-agnostic processor use. `_g_chunk` is overridden rather than refactored
in the parent ON PURPOSE -- ~85 queued cluster jobs will import
crit_preserve.py as it exists on disk WHEN THEY START, so editing that file
would silently change arms that are already queued.
"""
import json

import torch

from method.answer_tokens import pooled_decision_stat, resolve_answer_token_sets
from method.crit_preserve import CHUNK_IMAGES, CriterionPreserver

# Native polarity must match faith_loss.QUESTION exactly, so g^native here is the
# same statistic the anchor, critp and every measurement in the paper use.
NATIVE_TEMPLATE = "Is there a {obj} in the image?"
# Flip templates, in preference order. The diagnostic picks the winner on the base
# model; training uses whichever is passed in.
FLIP_TEMPLATES = {
    "neg": "Is there no {obj} in the image?",
    "absent": "Is the {obj} absent from the image?",
}
MODES = ("mean", "item")


def antisym_residual(g_native, g_flipped):
    """r_i = (g_i^native + g_i^flipped)/2 -- the label-free bias estimate."""
    assert g_native.shape == g_flipped.shape, (g_native.shape, g_flipped.shape)
    return 0.5 * (g_native + g_flipped)


def antisym_penalty(g_native, g_flipped, mode="mean"):
    """L from the residuals. fp32 in, fp32 out; see the module docstring."""
    assert mode in MODES, f"mode must be one of {MODES}, got {mode!r}"
    r = antisym_residual(g_native.float(), g_flipped.float())
    if mode == "mean":
        return r.mean().pow(2)
    return r.pow(2).mean()


def flip_correlation(g_native, g_flipped):
    """Pearson corr between the two polarities. The health statistic.

    Strongly NEGATIVE (<= -0.30) means the model inverts on polarity and r is a
    genuine bias estimate. Near zero or positive means it does not, and the
    penalty is silently attacking d' instead of the criterion. Returns None when
    either side is constant (correlation undefined) rather than a fake 0.0.
    """
    a = g_native.detach().float()
    b = g_flipped.detach().float()
    if a.numel() < 3:
        return None
    a = a - a.mean()
    b = b - b.mean()
    da = torch.sqrt((a * a).sum())
    db = torch.sqrt((b * b).sum())
    if float(da) <= 0 or float(db) <= 0:
        return None
    return float((a * b).sum() / (da * db))


class CriterionAntisymmetry(CriterionPreserver):
    """Corrected-target criterion regularizer. Needs NO base statistics.

    Unlike CriterionPreserver this class does not call init_base_stats(): its
    target is b = 0, which is derived rather than measured, so there is nothing
    to cache from theta0. `init_base_stats` is accepted and made a no-op so the
    training harness can call it uniformly for either regularizer.
    """

    def __init__(self, processor, probe_jsonl, data_root, k=8, device="cuda",
                 backbone=None, mode="mean", flip="neg", flip_template=None):
        # Parent validates the probe set, resolves answer-token ids in context and
        # rejects label-like keys; reuse all of it. mode is validated separately
        # here because the parent's MODES differ from ours.
        super().__init__(processor, probe_jsonl, data_root, k=k, device=device,
                         backbone=backbone, mode="mean")
        assert mode in MODES, f"mode must be one of {MODES}, got {mode!r}"
        self.mode = mode
        if flip_template is None:
            assert flip in FLIP_TEMPLATES, (
                f"flip must be one of {sorted(FLIP_TEMPLATES)} or an explicit "
                f"flip_template, got {flip!r}")
            flip_template = FLIP_TEMPLATES[flip]
        assert "{obj}" in flip_template, (
            f"flip_template must contain '{{obj}}', got {flip_template!r}")
        self.flip = flip
        self.flip_template = flip_template
        self.native_template = NATIVE_TEMPLATE
        self._tmpl = NATIVE_TEMPLATE     # which template _g_chunk renders
        self.history = []                # per-logged-step health record
        self.base_source = "none (target b=0 is derived, not measured)"
        # CASING FIX (2026-09-09). The parent resolves a SINGLE pair of ids for
        # " Yes"/" No" (3869/1939 on LLaVA). The model's emitted answer token
        # drifts with the task -- at TextVQA and VizWiz stages it emits 'yes'/'no'
        # (4874/694), which that pair does not contain, so the parent's statistic
        # decouples from the actual decision at those stages. Pool over casing
        # variants instead; verified on the real tokenizer to capture both forms.
        self.yes_ids, self.no_ids = resolve_answer_token_sets(processor.tokenizer,
                                                              self.bb)

    # -- template-parametrized forward -------------------------------------
    # Body mirrors CriterionPreserver._g_chunk exactly except for the template.
    # Duplicated deliberately: see the module docstring on queued jobs.

    def _g_chunk(self, model, rows):
        assert 1 <= len(rows) <= CHUNK_IMAGES, len(rows)
        param = next(model.parameters())
        images = [self._load(img) for img, _ in rows]
        texts = [self.bb.build_prompt(self._tmpl.format(obj=obj)) for _, obj in rows]
        enc = self.p(text=texts, images=images, return_tensors="pt",
                     padding=True, truncation=True, max_length=2048)
        model_inputs = {}
        for name, v in enc.items():
            if hasattr(v, "to"):
                v = v.to(self.device, param.dtype) if v.dtype.is_floating_point \
                    else v.to(self.device)
            model_inputs[name] = v
        out = model(use_cache=False, **model_inputs)
        logits = out.logits
        attn = model_inputs["attention_mask"]
        assert logits.shape[1] == attn.shape[1], (
            f"logit length {logits.shape[1]} != input length {attn.shape[1]}"
            " - image-token expansion mismatch; last-position index would be wrong")
        assert logits.shape[-1] > max(max(self.yes_ids), max(self.no_ids))
        last = attn.sum(dim=1) - 1
        idx = last.view(-1, 1, 1).expand(-1, 1, logits.shape[-1])
        step = logits.gather(1, idx).squeeze(1).float()
        # Pooled over casing variants -- see the CASING FIX note in __init__.
        return pooled_decision_stat(step, self.yes_ids, self.no_ids)

    def _g_rows_with(self, model, rows, template):
        prev = self._tmpl
        self._tmpl = template
        try:
            return self._g_rows(model, rows)
        finally:
            self._tmpl = prev

    # -- no theta0 needed ---------------------------------------------------

    def init_base_stats(self, model, cache_path=None, fallback_path=None,
                        **kwargs):
        """No-op. The target is b = 0, derived; there is no reference to cache.

        Kept so the training harness can treat this interchangeably with
        CriterionPreserver without a branch.
        """
        self.g0_by_id = {}
        return {"base_src": self.base_source, "n_probe": len(self.rows)}

    def base_summary(self):
        return {"base_src": self.base_source, "target": 0.0,
                "flip": self.flip, "flip_template": self.flip_template,
                "mode": self.mode, "k": self.k}

    # -- the penalty --------------------------------------------------------

    def loss(self, model, record=False):
        """Sample k probe items, forward BOTH polarities, penalize the residual.

        Costs two forwards per step instead of one; k is the knob for that.
        `record=True` appends a health entry to `history` -- always pass it on the
        harness's logging steps, because the flip correlation is the only signal
        that separates a working run from one that is quietly collapsing d'.
        """
        sampled = self.rng.sample(self.rows, self.k)
        rows = [(r["image"], r["object"]) for r in sampled]
        g_nat = self._g_rows_with(model, rows, self.native_template)
        g_flip = self._g_rows_with(model, rows, self.flip_template)
        val = antisym_penalty(g_nat, g_flip, mode=self.mode)
        assert bool(torch.isfinite(val.detach())), "antisym loss non-finite"
        if record:
            r = antisym_residual(g_nat.detach().float(), g_flip.detach().float())
            self.history.append({
                "loss": float(val.detach()),
                "mean_r": float(r.mean()),          # the bias estimate itself
                "sd_r": float(r.std()) if r.numel() > 1 else 0.0,
                "mean_g_native": float(g_nat.detach().float().mean()),
                "mean_g_flip": float(g_flip.detach().float().mean()),
                # THE health statistic. Must stay clearly negative. Rising toward
                # 0 => the model is going polarity-blind and the run is INVALID.
                "flip_corr": flip_correlation(g_nat, g_flip),
                "k": self.k, "mode": self.mode, "flip": self.flip,
            })
        return val

    def dump_history(self, path):
        with open(path, "w") as f:
            json.dump({"summary": self.base_summary(),
                       "history": self.history}, f, indent=2)
