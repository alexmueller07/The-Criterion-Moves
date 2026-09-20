"""Criterion correction by targeting the YES-RATE on a balanced probe (candidate M7).

WHY THIS EXISTS (2026-09-11). Every label-free regularizer in this project so far
pins the decision statistic to the FROZEN BASE's value. The base sits at
c = +0.431, so they inherit a mis-placed target: the anchor lands at c = +0.697
and is worse than doing nothing at every stage (mean |c| 0.731 vs sequential's
0.196; sequential's worst stage beats the anchor's best in 9/9 cells). The
mechanism works. The target is wrong.

THE IDENTITY THIS RESTS ON, AND ITS LIMIT. On a probe whose items are half
present and half absent, with H the hit rate and FA the false-alarm rate,

    c = 0  <=>  z(H) = -z(FA)  <=>  H = 1 - FA  <=>  yes-rate = 0.500

so aiming the criterion at zero and making the model say "yes" half the time are
the SAME instruction, and the second requires no labels, only a count of yes
answers. The identity is exact on the full item set, because POPE is 4500/4500.
It is very slightly inexact on what is actually SCORED: parse failures drop rows
asymmetrically (602 rows across 23 of 120 stage-cells), so 46 of 246 cells are
not exactly balanced on the parsed subset. Max prevalence deviation 0.0049, max
|yes-rate - 0.5*(H+FA)| 0.0036 -- about 50x smaller than the smallest per-cell
criterion range it is used to reason about, but it is a bound, not an identity.

BUT c = 0 is optimal only under EQUAL VARIANCE, and our own data reject that.
Balanced accuracy is maximised where the ROC slope equals 1, which coincides with
c = 0 only when the z-ROC slope b equals 1. Our fitted slopes are 0.55-0.72, and
there the optimum sits near a yes-rate of 0.47, not 0.50:

    b = 1.000   optimum at yes-rate 0.5000   cost of targeting 0.5 = 0
    b = 0.715   optimum at yes-rate 0.4693   cost = 0.0026 balanced accuracy
    b = 0.550   optimum at yes-rate 0.4530   cost = 0.0072

So targeting 0.5 is an APPROXIMATION, not a derivation, and neither this module
nor the paper may describe it as correct by logic. The approximation is
defensible because its error (<= 0.007) is an order of magnitude smaller than the
mis-placement it corrects: the anchor gives up 0.040 balanced accuracy by parking
the criterion at +0.697. `target` is exposed so a run can aim at an empirically
estimated optimum once one exists.
The probe is `grounding/probe_balanced.jsonl` (300 present / 300 absent by
construction, built once from COCO instance annotations by
build_balanced_probe.py). Building it reads annotations; USING it does not. No
row carries a label, the loss only ever counts yes-ness, and nothing in the
continual stream is annotated. So the method stays label-free in the sense that
matters.

WHAT THE MODEL'S DECISION ACTUALLY IS. The model answers yes iff g > 0, where
g = logsumexp(yes-token logits) - logsumexp(no-token logits) pooled over casing
variants (method/answer_tokens.py -- the single-id version reads rows the model
stops emitting at some stages). So yes-rate = P(g > 0), and the target is
    median(g) = 0
not mean(g) = 0. Those differ whenever g is skewed, which is why this class
offers two modes. Neither is unconditionally correct at the default tau -- see
the tau note below the mode list:

  mode="rate"  L = ( mean_i sigmoid(g_i / tau) - 0.5 )^2
               A smooth surrogate for P(g>0): as tau -> 0 the sigmoid approaches
               a step and the mean approaches the true yes-rate. Its gradient
               weights items by sigmoid'(g_i/tau), i.e. by how close they are to
               the decision boundary, so it is NOT item-invariant.

  mode="mean"  L = ( mean_i g_i )^2
               Keeps the item-invariant gradient that differentiates this family
               from the per-item (LwF / RCL L_pred / LLaVA-c UIR) family: every
               item in the batch receives the same push, so individual decisions
               stay free and only a common offset is opposed. But mean(g)=0 equals
               median(g)=0 only when g is symmetric, so this is an APPROXIMATION
               of the target, exact only under that assumption. Reported as an
               ablation, not as the method.

TAU IS A LIVE DESIGN CHOICE, NOT A DETAIL. Expanding the sigmoid,
sigma(u/tau) = 1/2 + u/(4 tau) + O(tau^-3), so LARGE tau makes "rate" collapse
toward mean-matching -- i.e. toward the `mean` ablation it is supposed to beat --
and only tau -> 0 makes it the median-matching constraint the identity actually
asks for. At the default tau = 1.0 it sits between the two and does not clearly
dominate either. Small tau is not free: it concentrates the gradient on the few
items nearest the boundary and inflates the variance of a k = 8 estimate. Treat
tau as a swept hyperparameter, not a constant.

Neither mode needs base statistics, so init_base_stats is a no-op: the target
comes from the probe's balance rather than from the base model.

Differentiation obligations (related_work.md sec 9): AdaPrior (CVPR 2026
Highlight) tracks a model-induced prior online in continual learning, but over
per-class softmax priors in CNN classification, with no decision criterion, no
SDT, and no vision-language component. CCS (2212.03827) and PriDe (2309.03882)
estimate a label-free bias from a task symmetry rather than from a balanced
reference set. The nearest thing to this specific move is post-hoc prior
correction / logit adjustment, which is applied at inference rather than as a
training-time constraint along a continual stream.
"""
import json

import torch

from method.answer_tokens import pooled_decision_stat, resolve_answer_token_sets
from method.crit_preserve import CHUNK_IMAGES, CriterionPreserver

MODES = ("rate", "mean")
DEFAULT_TAU = 1.0
TARGET_YES_RATE = 0.5


def soft_yes_rate(g, tau=DEFAULT_TAU):
    """Smooth surrogate for P(g > 0). tau -> 0 recovers the hard fraction."""
    assert tau > 0, tau
    return torch.sigmoid(g.float() / tau).mean()


def balanced_penalty(g, mode="rate", tau=DEFAULT_TAU, target=TARGET_YES_RATE):
    """L from the pooled decision statistic; see the module docstring."""
    assert mode in MODES, f"mode must be one of {MODES}, got {mode!r}"
    if mode == "mean":
        return g.float().mean().pow(2)
    return (soft_yes_rate(g, tau) - target).pow(2)


class CriterionBalanced(CriterionPreserver):
    """Pushes the balanced probe's yes-rate to 0.5, i.e. the criterion to zero.

    Subclasses CriterionPreserver only to reuse its validated machinery: probe
    loading, chunked backbone-agnostic forwards, fp32 statistics, in-context
    token resolution. The target and the penalty are different.
    """

    # the balanced probe carries no label columns; a label-like key would mean
    # the wrong file was passed, so keep the parent's guard active
    def __init__(self, processor, probe_jsonl, data_root, k=8, device="cuda",
                 backbone=None, mode="rate", tau=DEFAULT_TAU,
                 target=TARGET_YES_RATE):
        super().__init__(processor, probe_jsonl, data_root, k=k, device=device,
                         backbone=backbone, mode="mean")
        assert mode in MODES, f"mode must be one of {MODES}, got {mode!r}"
        self.mode = mode
        self.tau = float(tau)
        self.target = float(target)
        self.history = []
        self.base_source = "none (target yes-rate=0.5 fixed a priori; exact only at z-ROC slope 1)"
        # pooled over casing variants: the single-id statistic reads vocabulary
        # rows the model stops emitting once a lowercase-answer task is trained
        self.yes_ids, self.no_ids = resolve_answer_token_sets(processor.tokenizer,
                                                              self.bb)

    def _g_chunk(self, model, rows):
        assert 1 <= len(rows) <= CHUNK_IMAGES, len(rows)
        param = next(model.parameters())
        images = [self._load(img) for img, _ in rows]
        texts = [self.bb.build_prompt(self.QUESTION.format(obj=obj))
                 for _, obj in rows]
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
        last = attn.sum(dim=1) - 1
        idx = last.view(-1, 1, 1).expand(-1, 1, logits.shape[-1])
        step = logits.gather(1, idx).squeeze(1).float()
        return pooled_decision_stat(step, self.yes_ids, self.no_ids)

    # the anchor's question template, kept identical so g is the same statistic
    # the rest of the paper measures
    QUESTION = "Is there a {obj} in the image?"

    def init_base_stats(self, model, cache_path=None, fallback_path=None, **kw):
        """No-op: the target comes from the probe's balance, not from the base model."""
        self.g0_by_id = {}
        return {"base_src": self.base_source, "n_probe": len(self.rows)}

    def base_summary(self):
        return {"base_src": self.base_source, "target_yes_rate": self.target,
                "mode": self.mode, "tau": self.tau, "k": self.k,
                "n_probe": len(self.rows)}

    def loss(self, model, record=False):
        """Sample k probe items, forward once, penalize the yes-rate gap.

        One forward per step, unlike the antisymmetry term's two. `record=True`
        stores the realized soft yes-rate so a run that is driving the statistic
        to zero (rather than centring it) is visible afterwards: a collapsing
        |g| with a yes-rate stuck at 0.5 is the degenerate solution.
        """
        sampled = self.rng.sample(self.rows, self.k)
        g = self._g_rows(model, [(r["image"], r["object"]) for r in sampled])
        val = balanced_penalty(g, mode=self.mode, tau=self.tau, target=self.target)
        assert bool(torch.isfinite(val.detach())), "balanced crit loss non-finite"
        if record:
            gd = g.detach().float()
            self.history.append({
                "loss": float(val.detach()),
                "soft_yes_rate": float(soft_yes_rate(gd, self.tau)),
                "hard_yes_rate": float((gd > 0).float().mean()),
                "mean_g": float(gd.mean()),
                "median_g": float(gd.median()),
                # the degeneracy watch: centring the criterion should NOT shrink
                # the spread of g. If this falls while yes-rate sits at 0.5, the
                # model is being flattened, not calibrated.
                "sd_g": float(gd.std()) if gd.numel() > 1 else 0.0,
                "k": self.k, "mode": self.mode, "tau": self.tau,
            })
        return val

    def dump_history(self, path):
        with open(path, "w") as f:
            json.dump({"summary": self.base_summary(), "history": self.history},
                      f, indent=2)
