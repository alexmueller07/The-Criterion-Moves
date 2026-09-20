"""Label-free CRITERION-preservation regularizer (method candidate M1).

The measurement (analysis/PILOT_FINDINGS.md) and the theory
(design_notes/method_calibration_theory.md, Claim 1) say a continual-tuning
stage acts, to first order, as a class-INDEPENDENT additive bias b on the
yes/no decision statistic

    g(I, o; theta) = z_yes(I, o; theta) - z_no(I, o; theta)

at the answer position of "Is there a {o} in the image?": d' is flat, the
criterion c moves one-for-one with b, and b is re-set by every stage's answer
statistics. The criterion IS the population mean offset of g. So the thing to
preserve is a POPULATION MOMENT of g, and it can be estimated WITHOUT LABELS:
for a probe set P of (image, object-name) questions with no ground truth,

    mean:     L_crit = ( mean_{i in P} g_i(theta) - mean_{i in P} g_i(theta0) )^2
    meanvar:  L_crit = mean-term + ( std_i g_i(theta) - std_i g_i(theta0) )^2
    item:     L_crit = mean_i softplus( |g_i(theta) - g_i(theta0)| - delta )
              [ABLATION ONLY -- this is the per-item family (LwF / PFCL /
               RCL L_pred / LLaVA-c UIR) the method is defined AGAINST]

with theta0 the ORIGINAL base model (never the previous stage). Per step the
mean is estimated on k sampled probe items, PAIRED with the cached base value
of the same items, i.e. L = (mean_{i in S} [g_i(theta) - g_i(theta0)])^2.
Pairing matters: the unpaired estimator (batch mean vs full-set base mean)
carries the Var_P(g)/k sampling noise of which items were drawn, which would
push the sampled subset toward the population mean instead of pinning the
shift.

Equation-level differentiation from every label-free base-anchoring method
the vocab-A sweep found (design_notes/method_candidates_novelty_M1_vocabA.md):
those pin PER-ITEM outputs, hidden states or per-prompt marginals, so their
gradient on item i is a function of item i's own deviation. Here

    dL_mean / d g_i = (2/k) * mean_j d_j        (d_j = g_j(theta) - g_j(theta0))

is IDENTICAL for every item in the batch: the term never asks any individual
decision to stay where the base put it, it only opposes a common shift of the
whole batch. Per-item decisions may move freely (and must, if the new task
teaches new grounding); only the criterion is held. tests/test_crit_preserve.py
pins exactly this (zero-mean per-item perturbations cost 0 in "mean" mode and
> 0 in "item" mode; the mean-mode gradient is item-invariant).

Design constraints inherited VERBATIM from faith_loss.FaithfulnessAnchor so it
fits 24GB on Qwen2.5-VL: the FULL processor is run on (text, image) together per
forward and every processor output is passed to the model (backbone-agnostic;
no assumption that input_ids are image-independent); forwards are chunked to
<= 3 images and concatenated (peak vision memory never grows with k); the
statistic is computed in fp32 regardless of model dtype; answer-token ids are
resolved once, IN CONTEXT, via backbone.anchor_stub(); probe images go through
backbone.anchor_image() (Qwen downsizes; LLaVA no-op); logits are gathered at
attention_mask.sum(1) - 1.

The penalty MATH is factored into pure tensor functions (decision_stat /
population_moments / crit_penalty) unit-tested on CPU; the class wires them to
model forwards and is exercised end-to-end by the cluster smoke gate.
"""
import json
import os
import random

import torch
import torch.nn.functional as F
from PIL import Image

from method.faith_loss import QUESTION  # "Is there a {obj} in the image?"

SEED = 17
MODES = ("mean", "meanvar", "item")
CHUNK_IMAGES = 3   # faith_loss forwards ONE pair = 3 images at a time; same bound
DEFAULT_DELTA = 1.0  # item-mode dead zone (logits); ablation only


# ---------------------------------------------------------------------------
# Pure penalty math (torch tensors in/out; no model, no I/O). Unit-tested.
# ---------------------------------------------------------------------------

def decision_stat(z_yes, z_no):
    """g = z_yes - z_no in fp32 (the 1-D decision statistic; yes iff g > 0)."""
    return z_yes.float() - z_no.float()


def population_moments(g):
    """(mean, std) of a batch of decision statistics. Population std (ddof=0)
    so a 1-item batch is 0, not NaN; std of a constant batch is exactly 0."""
    g = g.float()
    mean = g.mean()
    if g.numel() < 2:
        return mean, g.new_zeros(())
    return mean, g.std(unbiased=False)


def crit_penalty(g, g0, mode="mean", delta=DEFAULT_DELTA):
    """Criterion-preservation penalty between the live statistics g (k,) and
    the cached base statistics g0 (k,) of the SAME k probe items.

      mean     ( mean(g) - mean(g0) )^2
      meanvar  ( mean(g) - mean(g0) )^2 + ( std(g) - std(g0) )^2
      item     mean_i softplus( |g_i - g0_i| - delta )      [ablation]

    g0 is treated as a constant (detached). Returns a 0-d fp32 tensor.
    """
    assert mode in MODES, f"mode must be one of {MODES}, got {mode!r}"
    g = g.float().reshape(-1)
    g0 = g0.detach().float().reshape(-1)
    assert g.shape == g0.shape, (
        f"live/base statistic shape mismatch {tuple(g.shape)} vs {tuple(g0.shape)}")
    assert g.numel() >= 1, "empty probe batch"
    if mode == "item":
        assert delta >= 0.0, f"delta must be >= 0, got {delta}"
        val = F.softplus((g - g0).abs() - float(delta)).mean()
    else:
        m, s = population_moments(g)
        m0, s0 = population_moments(g0)
        val = (m - m0) ** 2
        if mode == "meanvar":
            val = val + (s - s0) ** 2
    assert bool(torch.isfinite(val.detach())), "crit penalty non-finite"
    return val


# ---------------------------------------------------------------------------
# Base-statistic cache (plain JSON; no torch needed to read it).
# ---------------------------------------------------------------------------

def save_base_stats(path, g0_by_id, meta):
    """Write {meta..., "g0": {id: float}} atomically (tmp + rename)."""
    payload = dict(meta)
    payload["n"] = len(g0_by_id)
    vals = list(g0_by_id.values())
    payload["mean"] = sum(vals) / max(1, len(vals))
    if len(vals) > 1:
        mu = payload["mean"]
        payload["std"] = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
    else:
        payload["std"] = 0.0
    payload["g0"] = {str(k): float(v) for k, v in g0_by_id.items()}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)


def load_base_stats(path, expected_ids):
    """Read a cache written by save_base_stats and check it covers EXACTLY the
    probe ids in use (a stale cache from another probe file must fail loudly,
    not silently anchor to the wrong population)."""
    with open(path) as f:
        payload = json.load(f)
    g0 = payload.get("g0")
    assert isinstance(g0, dict) and g0, f"no g0 map in {path}"
    have, want = set(g0), set(str(i) for i in expected_ids)
    assert have == want, (
        f"base-stat cache {path} covers {len(have)} ids, probe set has {len(want)} "
        f"(missing {len(want - have)}, extra {len(have - want)}) - stale cache")
    return {k: float(v) for k, v in g0.items()}, payload


# ---------------------------------------------------------------------------
# The regularizer.
# ---------------------------------------------------------------------------

class CriterionPreserver:
    """Label-free criterion-preservation term. `.loss(model)` returns the fp32
    penalty for k freshly sampled probe items; `.init_base_stats(model, ...)`
    must be called once (model on device) to cache g(theta0) for every item.

    Args
      processor    : HF processor for the backbone (tokenizer + image proc).
      probe_jsonl  : {id, image, object} rows (build_probe_set.py); image
                     paths resolve against data_root. NO labels are read; a
                     row carrying any label-like key is rejected so the
                     label-free claim is enforced by construction.
      data_root    : root the probe image paths resolve against.
      k            : probe items sampled per step (default 8).
      device       : forward device.
      backbone     : model_zoo adapter (default llava15, as faith_loss).
      mode         : "mean" (the method) | "meanvar" | "item" (ablation).
      delta        : item-mode dead zone in logits (ignored otherwise).
    """

    FORBIDDEN_KEYS = ("label", "gt", "answer", "present", "absent", "target")

    def __init__(self, processor, probe_jsonl, data_root, k=8, device="cuda",
                 backbone=None, mode="mean", delta=DEFAULT_DELTA):
        assert mode in MODES, f"mode must be one of {MODES}, got {mode!r}"
        self.mode = mode
        self.delta = float(delta)
        if backbone is None:  # same default as faith_loss
            from model_zoo import get_backbone
            backbone = get_backbone("llava15")
        self.bb = backbone
        self.p = processor
        self.data_root = data_root
        self.k = int(k)
        self.device = torch.device(device)
        self.rng = random.Random(SEED)
        self.probe_path = probe_jsonl

        with open(probe_jsonl) as f:
            self.rows = [json.loads(l) for l in f if l.strip()]
        assert self.rows, f"empty probe set {probe_jsonl}"
        assert 1 <= self.k <= len(self.rows), (
            f"k={self.k} but only {len(self.rows)} probe items in {probe_jsonl}")
        ids = set()
        for r in self.rows:
            for key in ("id", "image", "object"):
                assert key in r, f"probe row missing {key!r}: {r}"
            bad = [key for key in self.FORBIDDEN_KEYS if key in r]
            assert not bad, (
                f"probe row {r.get('id')!r} carries label-like keys {bad}; the "
                "criterion term is label-free by construction - strip them")
            assert isinstance(r["object"], str) and r["object"].strip(), r
            path = os.path.join(data_root, r["image"])
            assert os.path.exists(path), f"missing probe image {path}"
            assert r["id"] not in ids, f"duplicate probe id {r['id']!r}"
            ids.add(r["id"])

        # Answer-token ids resolved IN CONTEXT (faith_loss's trap-guard, copied):
        # " Yes" in isolation tokenizes to [space-marker, "Yes"] under Llama; the
        # id we need is the first token by which tok(stub + answer) extends
        # tok(stub).
        tok = processor.tokenizer
        stub = self.bb.anchor_stub()
        stub_ids = tok(stub, add_special_tokens=False)["input_ids"]
        for text, attr in ((" Yes", "yes_id"), (" No", "no_id")):
            full_ids = tok(stub + text, add_special_tokens=False)["input_ids"]
            assert full_ids[:len(stub_ids)] == stub_ids, (
                f"context tokenization unstable for {text!r}: "
                f"{full_ids[:len(stub_ids)]} vs {stub_ids}")
            assert len(full_ids) > len(stub_ids), f"no answer token for {text!r}"
            setattr(self, attr, full_ids[len(stub_ids)])
        assert self.yes_id != self.no_id

        self.g0_by_id = None      # filled by init_base_stats
        self.base_source = None   # audit string: how theta0 stats were obtained

    # -- forwards (faith_loss pattern, verbatim) ----------------------------

    def _load(self, rel):
        img = Image.open(os.path.join(self.data_root, rel)).convert("RGB")
        return self.bb.anchor_image(img)

    def _g_chunk(self, model, rows):
        """Forward <= CHUNK_IMAGES rows [(image_rel, obj), ...]; return the (n,)
        fp32 decision statistic g = z_yes - z_no at each row's last real token."""
        assert 1 <= len(rows) <= CHUNK_IMAGES, len(rows)
        param = next(model.parameters())
        images = [self._load(img) for img, _ in rows]
        texts = [self.bb.build_prompt(QUESTION.format(obj=obj)) for _, obj in rows]
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
        assert logits.shape[-1] > max(self.yes_id, self.no_id)
        last = attn.sum(dim=1) - 1
        idx = last.view(-1, 1, 1).expand(-1, 1, logits.shape[-1])
        step = logits.gather(1, idx).squeeze(1).float()
        return decision_stat(step[:, self.yes_id], step[:, self.no_id])

    def _g_rows(self, model, rows):
        """(n,) g for any number of rows, forwarded CHUNK_IMAGES at a time and
        concatenated (identical to one batched forward; bounded vision memory)."""
        parts = []
        for i in range(0, len(rows), CHUNK_IMAGES):
            parts.append(self._g_chunk(model, rows[i:i + CHUNK_IMAGES]))
        return torch.cat(parts, dim=0)

    # -- theta0 reference (once) -------------------------------------------

    @torch.no_grad()
    def _all_g_eval(self, model):
        """g for EVERY probe item, deterministic (eval mode: dropout off)."""
        was_training = model.training
        model.eval()
        try:
            rows = [(r["image"], r["object"]) for r in self.rows]
            g = self._g_rows(model, rows).detach().float().cpu()
        finally:
            if was_training:
                model.train()
        assert g.shape == (len(self.rows),), g.shape
        return g

    @staticmethod
    def _adapter_disabler(model):
        """PEFT's `with model.disable_adapter():` context manager if the model
        exposes one (PeftModel), else None. Only the CONTEXT-MANAGER form is
        accepted: transformers' PeftAdapterMixin has a plural
        `disable_adapters()` that toggles state instead and must not be
        confused with it."""
        fn = getattr(model, "disable_adapter", None)
        if fn is None or not callable(fn):
            return None
        try:
            ctx = fn()
        except Exception:
            return None
        if not (hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__")):
            return None
        return ctx

    def init_base_stats(self, model, cache_path=None, fallback_path=None,
                        is_stage1=False):
        """Cache g_i(theta0) for every probe item. Routes, in order:

          1. PEFT `with model.disable_adapter():` available -> compute with the
             adapter switched off. This is the TRUE base even when a previous
             stage's adapter is resumed (LoRA layers fall back to the frozen
             weights; modules_to_save fall back to their original modules).
          2. else, at stage 1 (`is_stage1`, no resumed adapter): a fresh LoRA has
             B = 0, so the live model IS the base -> compute directly.
          3. else, `fallback_path` (the previous stage's cache) -> load.

        The result is always written to `cache_path` when given, so the next
        stage has a fallback and the audit trail records theta0's statistics.
        Returns the audit string stored in self.base_source.
        """
        ctx = self._adapter_disabler(model)
        if ctx is not None:
            with ctx:
                g = self._all_g_eval(model)
            source = "disable_adapter"
        elif is_stage1:
            g = self._all_g_eval(model)
            source = "stage1_fresh_lora_is_base"
        elif fallback_path and os.path.exists(fallback_path):
            g0, payload = load_base_stats(fallback_path, [r["id"] for r in self.rows])
            self.g0_by_id = g0
            source = f"loaded:{fallback_path}"
            self.base_source = source
            if cache_path:
                meta = {k: v for k, v in payload.items() if k != "g0"}
                meta["source"] = source
                save_base_stats(cache_path, g0, meta)
            return source
        else:
            raise RuntimeError(
                "cannot obtain base-model probe statistics: model has no PEFT "
                "disable_adapter() context, this is not stage 1, and no cached "
                f"probe_base_stats.json was found at {fallback_path!r}")
        self.g0_by_id = {r["id"]: float(v) for r, v in zip(self.rows, g.tolist())}
        self.base_source = source
        if cache_path:
            save_base_stats(cache_path, self.g0_by_id, {
                "source": source, "probe": os.path.abspath(self.probe_path),
                "backbone": getattr(self.bb, "name", None), "mode": self.mode,
                "yes_id": int(self.yes_id), "no_id": int(self.no_id)})
        return source

    # -- public API --------------------------------------------------------

    def base_summary(self):
        """(mean, std) of the cached base statistics over the WHOLE probe set."""
        assert self.g0_by_id is not None, "call init_base_stats first"
        g0 = torch.tensor(list(self.g0_by_id.values()), dtype=torch.float32)
        m, s = population_moments(g0)
        return float(m), float(s)

    def loss(self, model):
        """Sample k probe items, forward them (<= 3 images per forward), and
        return the fp32 criterion-preservation penalty vs the cached theta0
        statistics of the SAME items."""
        assert self.g0_by_id is not None, (
            "CriterionPreserver.init_base_stats(model) must run before loss()")
        sampled = self.rng.sample(self.rows, self.k)
        g = self._g_rows(model, [(r["image"], r["object"]) for r in sampled])
        g0 = torch.tensor([self.g0_by_id[r["id"]] for r in sampled],
                          dtype=torch.float32, device=g.device)
        val = crit_penalty(g, g0, mode=self.mode, delta=self.delta)
        assert bool(torch.isfinite(val.detach())), "crit loss non-finite"
        return val
