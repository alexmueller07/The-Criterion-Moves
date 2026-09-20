"""Multi-axis answer-POLICY stabilizer (method candidate C2 + C6 + abstention).

The measurement (analysis/PILOT_FINDINGS.md) decomposes faithfulness drift under
continual adaptation into THREE answer-policy components:

  (1) yes/no CRITERION drift   c  (dose-dependent, recency-driven, d'-flat)
  (2) refusal/ABSTENTION leakage   ("Unanswerable" bleeds onto grounded questions)
  (3) EOS/LENGTH drift             (the late-caption stopping-hazard shifts)

The v2 faithfulness anchor (faith_loss.FaithfulnessAnchor) stabilizes ONLY (1):
its three softplus margins act on the yes/no logits. METHOD_COMPARISON.md records
the two open axes as scoped limitations -- prediction (ii) (leakage) FAILED for
the anchor because "the anchor supervises the yes/no axis, not the abstention
token", and the EOS/length axis is only touched indirectly (via a
length-controlled CHAIR reduction that "protects less during caption-heavy
training").

This module makes the METHOD the true counterpart of the MEASUREMENT: it
stabilizes the WHOLE answer policy, one loss term per measured drift component.
It does NOT edit faith_loss.py -- it *composes* with it, importing
FaithfulnessAnchor for the criterion term (1) and adding two new,
ground-truth-supervised terms for (2) and (3):

  L_policy = lambda_c * L_criterion   (imported v2 anchor: yes/no margins)
           + lambda_a * L_abstain     (NEW: refusal-token suppression, evidence present)
           + lambda_e * L_eos         (NEW: stopping-hazard stability vs a fixed reference)

and the training objective is L_task + L_policy. Each term is independently
ablatable (its lambda, or its use_* flag). With lambda_a = lambda_e = 0 the module
reproduces the v2 anchor exactly (scaled by lambda_c), so it is a strict superset.

Design constraints inherited from faith_loss.py (so it fits 24GB on Qwen2.5-VL):
  * the FULL processor is run on (text, image) together per forward and every
    processor output is passed to the model -- backbone-agnostic via model_zoo
    (llava15 / qwen25vl), no assumption that input_ids are image-independent;
  * forwards are chunked to ONE image at a time and concatenated, so peak vision
    memory never grows with k;
  * all margins / log-probs are computed in fp32 regardless of model dtype;
  * answer/refusal token ids and the stop-token id are resolved once, in context,
    at init (the leading-space / cross-boundary-merge trap that faith_loss guards).

The penalty MATH is factored into pure tensor functions at the top of the file
(abstain_term / eos_stability_term / eos_explen_term / compose_policy_loss); those
are unit-tested on CPU in tests/test_policy_anchor.py the way tests/test_baselines.py
tests the ewc/olora/lwf math. The class wires them to model forwards and is
exercised end-to-end by the cluster smoke gate.
"""
import json
import os
import random

import torch
import torch.nn.functional as F
from PIL import Image

from method.crit_preserve import CriterionPreserver
from method.faith_loss import FaithfulnessAnchor, QUESTION

SEED = 17
EOS_REF_CACHE = "pol_eos_ref.json"  # per-stage theta0 stopping-hazard cache

# Default TextVQA-style wrapper. The measured leakage (A9) is FORMAT-CONDITIONAL:
# "Unanswerable" bleeds into the open single-word/phrase VQA format, never into
# the yes/no or MC formats. So the abstention term poses the grounding question
# INSIDE that leakage-prone wrapper -- otherwise it would be inert (the refusal
# token is already ~0 in the plain yes/no format and there would be nothing to
# suppress). Set abstain_suffix="" to co-locate it with the plain anchor format.
ABSTAIN_SUFFIX = " Answer the question using a single word or phrase."

# First-token candidates for the refusal/abstention behavior. The pilot's measured
# leakage string is exactly "Unanswerable"; the others are cheap extra coverage.
# Aggregated by logsumexp (a smooth "strongest refusal"), so unresolved phrases
# are simply dropped and a single resolved phrase reduces to its plain logit.
REFUSAL_PHRASES = (" Unanswerable", " Unknown", " Sorry", " Unclear")

# Captioning instruction for the EOS/length anchor set. The late-caption hazard is
# a *generative* (captioning) phenomenon, so the stopping-hazard anchor is measured
# on a caption prompt, not on the yes/no query.
CAPTION_PROMPT = "Describe the image in detail."


# ---------------------------------------------------------------------------
# Pure penalty math (torch tensors in/out; no model, no I/O). Unit-tested.
# ---------------------------------------------------------------------------

def abstain_term(z_answer, z_refuse, margin):
    """Abstention-calibration penalty (drift component #2).

    On an image where the queried object is GENUINELY PRESENT (resp. absent), the
    grounded answer token is " Yes" (resp. " No") and abstaining is *wrong* -- the
    question is answerable. This term supervises the refusal-token logit DOWN
    relative to the correct grounded answer:

        L_abstain = softplus( margin - ( z_answer - logsumexp_r z_refuse[r] ) )

    z_answer : (...,)   logit of the correct grounded answer token (Yes / No)
    z_refuse : (..., R) logits of the R refusal first-tokens; aggregated by
               logsumexp into a single "strongest refusal" score.
    margin   : how far the grounded answer must beat the refusal score.

    softplus (matching faith_loss's soft-hinge philosophy) so a model that already
    answers confidently pays ~0 (asymptotically) and the term is an anchor, not a
    permanent drag. Returns the elementwise penalty (reduce with .mean() outside).

    NON-REDUNDANCY vs the v2 anchor: the anchor's present/absent terms constrain
    z_yes vs z_no. The refusal token is a DIFFERENT vocabulary entry than " No";
    nothing in faith_loss ever touches it. Suppressing z_no does not move z_refuse.
    """
    agg = torch.logsumexp(z_refuse, dim=-1)
    return F.softplus(margin - (z_answer - agg))


def eos_stability_term(eos_logprob, eos_logprob_ref):
    """EOS/length stability penalty (drift component #3), primary form.

    Constrains the whole per-position stopping-hazard TRAJECTORY of the model to a
    fixed reference (the base / pre-CL model), formalizing the late-caption-hazard
    mechanism:

        L_eos = mean_t ( log p_theta(EOS | y_<t, x) - log p_theta0(EOS | y_<t, x) )^2

    eos_logprob     : (T,) current-model log p(stop | prefix + y_<=t), t = 0..T-1,
                      teacher-forced along a reference caption of length T-1 (+EOS).
    eos_logprob_ref : (T,) the same, cached once from the reference model theta0.

    Exactly 0 when the current stopping hazard equals the reference at every
    position; > 0 for any drift. Reference-anchored (not reference-free) because
    "drift" is only defined against a reference; theta0's hazard on the anchor set
    is cached once at init (the EWC / LwF snapshot pattern), so per-step cost is one
    live forward.
    """
    return ((eos_logprob - eos_logprob_ref) ** 2).mean()


def expected_length(eos_logprob):
    """Differentiable expected generated length under the teacher-forced hazard.

    h_t = exp(eos_logprob_t) is the probability of stopping after producing t
    caption tokens. Survival S_t = prod_{s<t}(1 - h_s); E[L] = sum_t S_t. Any fixed
    additive offset cancels when differenced against a reference (see below), so the
    off-by-one at the boundary is immaterial -- what matters is the SHAPE.
    """
    h = eos_logprob.exp().clamp(min=1e-6, max=1.0 - 1e-6)
    surv = torch.cumprod(1.0 - h, dim=-1)            # surv[t] = prod_{s<=t}(1-h_s)
    reach = torch.cat([torch.ones(1, dtype=surv.dtype, device=surv.device),
                       surv[:-1]])                    # P(reach token t), t=0..T-1
    return reach.sum()


def eos_explen_term(eos_logprob, eos_logprob_ref):
    """EOS/length stability penalty, SCALAR (expected-length) variant.

        L_eos^len = ( E_theta[L] - E_theta0[L] )^2

    Weaker than eos_stability_term: it constrains only the mean of the stopping
    distribution, so a model can redistribute hazard across positions and keep E[L]
    fixed. Offered as an interpretable alternative ("stops earlier / later"); the
    per-position form is the default. Exactly 0 when the hazards match.
    """
    return (expected_length(eos_logprob) - expected_length(eos_logprob_ref)) ** 2


def compose_policy_loss(l_criterion, l_abstain, l_eos,
                        lam_c, lam_a, lam_e):
    """The composite anchor loss L_policy (L_task is added by the training loop):

        L_policy = lam_c * L_criterion + lam_a * L_abstain + lam_e * L_eos

    Any lambda = 0 drops its term exactly (the term is also skipped upstream so no
    forward is spent). Kept as a pure function so the "composite == weighted sum"
    invariant is unit-testable without a model.
    """
    return lam_c * l_criterion + lam_a * l_abstain + lam_e * l_eos


# ---------------------------------------------------------------------------
# Token-resolution helper (in-context first-token id, faith_loss's trap-guard).
# ---------------------------------------------------------------------------

def _first_token_in_context(tok, stub, stub_ids, text):
    """Id of the first token by which tok(stub + text) extends tok(stub), or None
    if the context tokenization is unstable (a cross-boundary BPE merge). Mirrors
    FaithfulnessAnchor's yes/no resolution so " Unanswerable" etc. resolve to the
    id the model actually emits first after "ASSISTANT:"."""
    full_ids = tok(stub + text, add_special_tokens=False)["input_ids"]
    if full_ids[:len(stub_ids)] != stub_ids:
        return None
    if len(full_ids) <= len(stub_ids):
        return None
    return full_ids[len(stub_ids)]


def _stop_token_id(backbone, tok):
    """The token whose emission ends generation for this backbone: the chat-turn
    terminator when the backbone defines one (Qwen: <|im_end|>), else the
    tokenizer EOS (LLaVA). This is the token whose hazard the EOS term stabilizes."""
    turn = getattr(backbone, "TURN_EOS", None)
    if turn is not None:
        tid = tok.convert_tokens_to_ids(turn)
        if tid is not None and tid >= 0:
            return tid
    assert tok.eos_token_id is not None, "tokenizer has no eos token for the EOS term"
    return tok.eos_token_id


# ---------------------------------------------------------------------------
# theta0 stopping-hazard cache (plain JSON; no torch needed to read it). Mirrors
# crit_preserve.save_base_stats / load_base_stats: written every stage as the
# next stage's fallback + audit trail, and a cache for a DIFFERENT caption set
# must fail loudly, never silently anchor the hazard to the wrong rows.
# ---------------------------------------------------------------------------

def save_eos_reference(path, ref_by_id, meta):
    """Write {meta..., "n", "ref": {id: [log p(EOS) per stopping position]}}
    atomically (tmp + rename)."""
    payload = dict(meta)
    payload["n"] = len(ref_by_id)
    payload["ref"] = {str(k): [float(x) for x in v] for k, v in ref_by_id.items()}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def load_eos_reference(path, expected_ids):
    """Read a cache written by save_eos_reference and check it covers EXACTLY
    the caption ids in use (stale-cache rejection, as load_base_stats)."""
    with open(path) as f:
        payload = json.load(f)
    ref = payload.get("ref")
    assert isinstance(ref, dict) and ref, f"no ref map in {path}"
    have, want = set(ref), set(str(i) for i in expected_ids)
    assert have == want, (
        f"EOS reference cache {path} covers {len(have)} ids, caption set has "
        f"{len(want)} (missing {len(want - have)}, extra {len(have - want)}) - stale cache")
    out = {}
    for k, v in ref.items():
        assert isinstance(v, list) and v, f"empty reference trajectory for {k!r} in {path}"
        out[k] = [float(x) for x in v]
    return out, payload


# ---------------------------------------------------------------------------
# The composite stabilizer.
# ---------------------------------------------------------------------------

class PolicyAnchor:
    """Multi-axis answer-policy stabilizer. Wraps a FaithfulnessAnchor (criterion
    term) and adds the abstention and EOS/length terms. `.loss(model)` returns the
    weighted composite L_policy; `.loss_components(model)` returns the per-term
    breakdown for logging.

    Args
      processor, pairs_jsonl, data_root : as FaithfulnessAnchor (grounding pairs).
      k_pairs, margin, backbone, criterion_mode : forwarded to FaithfulnessAnchor
        (criterion_mode -> its `mode`, "margin" | "ce").
      lambda_c / lambda_a / lambda_e : per-term weights (each ablatable; 0 disables
        AND skips the term's forwards).
      use_criterion / use_abstain / use_eos : hard on/off switches (default: on iff
        the term's lambda != 0).
      margin_a : abstention margin (grounded answer must beat refusal by this).
      abstain_suffix : format wrapper appended to the grounding question for the
        abstention forward (default ABSTAIN_SUFFIX, the leakage-prone VQA format;
        "" co-locates it with the plain anchor format -- see the honest note in
        design_notes/method_multiaxis.md about why the wrapper matters).
      abstain_include_absent : also supervise refusal-down on absent-object rows
        (correct answer " No"); default False (present-only is the lean form).
      refusal_phrases : refusal first-token candidates (default REFUSAL_PHRASES).
      caption_jsonl / caption_prompt / eos_mode : EOS anchor set + prompt + form
        ("logprob" per-position squared, default; "explen" scalar expected-length).
      k_caps : caption rows sampled per step for the EOS term.
      ref_model : frozen reference model (base / pre-CL) used ONCE at init to cache
        the reference stopping hazard. Alternatively pass ref_eos_logprobs (a list
        of 1-D tensors aligned with the caption rows) to skip the reference forward.
        Or pass NEITHER and call `.init_eos_reference(model, ...)` once the model
        is on device: that route obtains theta0 through PEFT's disable_adapter()
        context (the true base even on a resumed adapter), the stage-1 fresh-LoRA
        identity, or the previous stage's cache -- crit_preserve's pattern.
      device : forward device (default "cuda").

    Ground-truth supervision only: presence/absence comes from COCO annotations
    (the pairs file); the EOS reference is the base model's own hazard. No
    previous-stage checkpoint or preference pairs enter any term.
    """

    def __init__(self, processor, pairs_jsonl, data_root,
                 k_pairs=4, margin=2.0, device="cuda", backbone=None,
                 criterion_mode="margin",
                 lambda_c=0.1, lambda_a=0.1, lambda_e=0.1,
                 use_criterion=None, use_abstain=None, use_eos=None,
                 margin_a=2.0, abstain_suffix=ABSTAIN_SUFFIX,
                 abstain_include_absent=False, refusal_phrases=REFUSAL_PHRASES,
                 caption_jsonl=None, caption_prompt=CAPTION_PROMPT,
                 eos_mode="logprob", k_caps=4,
                 ref_model=None, ref_eos_logprobs=None):
        assert criterion_mode in ("margin", "ce"), criterion_mode
        assert eos_mode in ("logprob", "explen"), eos_mode
        self.lambda_c = float(lambda_c)
        self.lambda_a = float(lambda_a)
        self.lambda_e = float(lambda_e)
        self.use_criterion = (self.lambda_c != 0.0) if use_criterion is None else bool(use_criterion)
        self.use_abstain = (self.lambda_a != 0.0) if use_abstain is None else bool(use_abstain)
        self.use_eos = (self.lambda_e != 0.0) if use_eos is None else bool(use_eos)
        self.margin_a = float(margin_a)
        assert self.margin_a >= 0.0, self.margin_a
        self.abstain_suffix = abstain_suffix
        self.abstain_include_absent = bool(abstain_include_absent)
        self.eos_mode = eos_mode
        self.k_caps = int(k_caps)
        self.device = torch.device(device)

        # Criterion term = the imported v2 anchor (composition, not a re-write). It
        # also owns the processor/backbone/pairs/_load and the yes/no token ids we
        # reuse, so nothing is duplicated.
        self.crit = FaithfulnessAnchor(processor, pairs_jsonl, data_root,
                                       k_pairs=k_pairs, margin=margin,
                                       device=device, backbone=backbone,
                                       mode=criterion_mode)
        self.bb = self.crit.bb
        self.p = self.crit.p
        self.pairs = self.crit.pairs
        self.yes_id = self.crit.yes_id
        self.no_id = self.crit.no_id
        self.k_pairs = self.crit.k
        self.rng_a = random.Random(SEED + 1)
        self.rng_e = random.Random(SEED + 2)

        # Refusal first-token ids, resolved in context exactly like yes/no.
        tok = processor.tokenizer
        stub = self.bb.anchor_stub()
        stub_ids = tok(stub, add_special_tokens=False)["input_ids"]
        self.refuse_ids, dropped = [], []
        for phrase in refusal_phrases:
            tid = _first_token_in_context(tok, stub, stub_ids, phrase)
            if tid is None:
                dropped.append(phrase)
            elif tid not in self.refuse_ids:
                self.refuse_ids.append(tid)
        if self.use_abstain:
            assert self.refuse_ids, (
                f"no refusal phrase resolved to a stable in-context token "
                f"(tried {list(refusal_phrases)}); abstention term cannot run")
        self._dropped_refusals = dropped

        # EOS/length term: caption anchor set + cached reference hazard.
        self.captions = []
        self.ref_eos = None
        self.ref_source = None   # audit string: how theta0's hazard was obtained
        self.stop_id = None
        self.caption_path = caption_jsonl
        if self.use_eos:
            assert caption_jsonl is not None, (
                "use_eos requires caption_jsonl ({id,image,caption} rows, images "
                "relative to data_root, disjoint from the eval pools)")
            self.caption_prompt = caption_prompt
            self.eos_text = tok.eos_token  # backbones that ignore it use TURN_EOS
            self.stop_id = _stop_token_id(self.bb, tok)
            with open(caption_jsonl) as f:
                self.captions = [json.loads(l) for l in f if l.strip()]
            assert self.captions, f"empty caption anchor set {caption_jsonl}"
            ids = set()
            for r in self.captions:
                for key in ("id", "image", "caption"):
                    assert key in r, f"caption row missing {key!r}: {r}"
                path = os.path.join(data_root, r["image"])
                assert os.path.exists(path), f"missing caption image {path}"
                assert r["id"] not in ids, f"duplicate caption id {r['id']!r}"
                ids.add(r["id"])
            assert 1 <= self.k_caps <= len(self.captions), (
                f"k_caps={self.k_caps} but {len(self.captions)} caption rows")
            if ref_eos_logprobs is not None:
                assert len(ref_eos_logprobs) == len(self.captions), (
                    "ref_eos_logprobs must align 1:1 with the caption rows")
                self.ref_eos = [t.detach().float().cpu() for t in ref_eos_logprobs]
                self.ref_source = "ctor:precomputed"
            elif ref_model is not None:
                self.ref_eos = self._all_eos_eval(ref_model)
                self.ref_source = "ctor:ref_model"
            # else: deferred -> init_eos_reference(model, ...) must run before loss().

    # -- reference caching (once) ------------------------------------------

    @torch.no_grad()
    def _all_eos_eval(self, model):
        """theta0's per-position EOS log-prob for EVERY caption row, as detached
        CPU tensors aligned with self.captions; eval mode (dropout off) so the
        reference is deterministic."""
        was_training = model.training
        model.eval()
        try:
            ref = [self._eos_logprobs(model, r).detach().float().cpu()
                   for r in self.captions]
        finally:
            if was_training:
                model.train()
        assert len(ref) == len(self.captions)
        return ref

    def _cache_meta(self, source):
        return {"source": source,
                "captions": os.path.abspath(self.caption_path) if self.caption_path else None,
                "backbone": getattr(self.bb, "name", None),
                "eos_mode": self.eos_mode, "stop_id": int(self.stop_id),
                "caption_prompt": self.caption_prompt}

    def init_eos_reference(self, model, cache_path=None, fallback_path=None,
                           is_stage1=False):
        """Obtain theta0's stopping hazard on the caption anchor set. Routes, in
        order (crit_preserve.CriterionPreserver.init_base_stats, verbatim):

          1. PEFT `with model.disable_adapter():` available -> compute with the
             adapter switched off. The TRUE base even when a previous stage's
             adapter is resumed.
          2. else, at stage 1 (`is_stage1`): a fresh LoRA has B = 0, so the live
             model IS the base -> compute directly.
          3. else, `fallback_path` (the previous stage's cache) -> load.

        The result is always written to `cache_path` when given (next stage's
        fallback + audit trail). Returns the audit string stored in ref_source.
        No-op (returns None) when the EOS term is disabled.
        """
        if not self.use_eos:
            return None
        ctx = CriterionPreserver._adapter_disabler(model)
        if ctx is not None:
            with ctx:
                _was_training = model.training
                model.eval()  # reference in eval mode (no dropout), restored below
                try:
                    ref = self._all_eos_eval(model)
                finally:
                    if _was_training:
                        model.train()
            source = "disable_adapter"
        elif is_stage1:
            _was_training = model.training
            model.eval()  # reference in eval mode (no dropout), restored below
            try:
                ref = self._all_eos_eval(model)
            finally:
                if _was_training:
                    model.train()
            source = "stage1_fresh_lora_is_base"
        elif fallback_path and os.path.exists(fallback_path):
            ref_by_id, payload = load_eos_reference(
                fallback_path, [r["id"] for r in self.captions])
            ref = [torch.tensor(ref_by_id[str(r["id"])], dtype=torch.float32)
                   for r in self.captions]
            source = f"loaded:{fallback_path}"
            self.ref_eos, self.ref_source = ref, source
            if cache_path:
                meta = {k: v for k, v in payload.items() if k != "ref"}
                meta["source"] = source
                save_eos_reference(cache_path, ref_by_id, meta)
            return source
        else:
            raise RuntimeError(
                "cannot obtain base-model stopping hazard: model has no PEFT "
                "disable_adapter() context, this is not stage 1, and no cached "
                f"{EOS_REF_CACHE} was found at {fallback_path!r}")
        self.ref_eos, self.ref_source = ref, source
        if cache_path:
            save_eos_reference(cache_path,
                               {r["id"]: t.tolist() for r, t in zip(self.captions, ref)},
                               self._cache_meta(source))
        return source

    # -- forwards ----------------------------------------------------------

    def _to_model(self, enc, dtype):
        model_inputs = {}
        for name, v in enc.items():
            if hasattr(v, "to"):
                v = v.to(self.device, dtype) if v.dtype.is_floating_point \
                    else v.to(self.device)
            model_inputs[name] = v
        return model_inputs

    def _last_logits(self, model, image_rel, prompt_text):
        """Full (V,) logit vector at the last real token position for one
        (image, prompt) row. One image per forward -> bounded vision memory."""
        param = next(model.parameters())
        img = self.crit._load(image_rel)
        text = self.bb.build_prompt(prompt_text)
        enc = self.p(text=[text], images=[img], return_tensors="pt",
                     padding=True, truncation=True, max_length=2048)
        model_inputs = self._to_model(enc, param.dtype)
        out = model(use_cache=False, **model_inputs)
        logits = out.logits
        attn = model_inputs["attention_mask"]
        assert logits.shape[1] == attn.shape[1], (
            f"logit length {logits.shape[1]} != input length {attn.shape[1]}"
            " - image-token expansion mismatch; last-position index would be wrong")
        assert logits.shape[-1] > max(self.yes_id, self.no_id, *self.refuse_ids)
        last = int(attn.sum().item()) - 1
        return logits[0, last].float()  # (V,)

    def _abstain_prompt(self, obj):
        return QUESTION.format(obj=obj) + self.abstain_suffix

    def _abstain_loss(self, model):
        k = min(self.k_pairs, len(self.pairs))
        sampled = self.rng_a.sample(self.pairs, k)
        refuse = self.refuse_ids
        parts = []
        for r in sampled:
            # Present object -> correct grounded answer is " Yes"; refusal is wrong.
            z = self._last_logits(model, r["image"], self._abstain_prompt(r["present"]))
            parts.append(abstain_term(z[self.yes_id], z[refuse], self.margin_a))
            if self.abstain_include_absent:
                # Absent object -> correct grounded answer is " No"; refusal wrong too
                # (the question is answerable: the object is absent).
                za = self._last_logits(model, r["image"],
                                       self._abstain_prompt(r["absent"]))
                parts.append(abstain_term(za[self.no_id], za[refuse], self.margin_a))
        val = torch.stack(parts).mean()
        assert bool(torch.isfinite(val.detach())), "abstain loss non-finite"
        return val

    def _eos_logprobs(self, model, row):
        """(T,) log p(stop | prefix + y_<=t) at the T stopping positions of the
        teacher-forced caption. T = (caption+eos token count); positions are read
        from the SEQUENCE TAIL via attention_mask, so image-token expansion in the
        prefix never shifts them (the same tail-indexing faith_loss relies on)."""
        param = next(model.parameters())
        img = self.crit._load(row["image"])
        full_text, prefix_text, _frag = self.bb.build_train_text(
            self.caption_prompt, row["caption"], self.eos_text)
        tok = self.p.tokenizer
        # n_cap = tokens the caption+eos fragment ADDS beyond the prefix (text-only
        # tokenization; robust to cross-boundary merges, image-independent).
        n_full = len(tok(full_text, add_special_tokens=False)["input_ids"])
        n_pre = len(tok(prefix_text, add_special_tokens=False)["input_ids"])
        n_cap = n_full - n_pre
        assert n_cap >= 1, f"empty caption fragment for {row.get('id')!r}"
        enc = self.p(text=[full_text], images=[img], return_tensors="pt",
                     padding=True, truncation=True, max_length=2048)
        model_inputs = self._to_model(enc, param.dtype)
        out = model(use_cache=False, **model_inputs)
        logits = out.logits
        attn = model_inputs["attention_mask"]
        assert logits.shape[1] == attn.shape[1], (
            f"logit length {logits.shape[1]} != input length {attn.shape[1]}")
        L_true = int(attn.sum().item())
        # Stopping positions: the token BEFORE the first caption token (predicts an
        # immediate stop) through the last-but-one caption token. n_cap positions,
        # predicting y_1..y_{n_cap}; each carries the EOS hazard at that step.
        start = L_true - n_cap - 1
        assert start >= 0, (
            f"prefix shorter than expected for {row.get('id')!r} "
            f"(L_true={L_true}, n_cap={n_cap})")
        pos = torch.arange(start, start + n_cap, device=logits.device)
        lp = F.log_softmax(logits[0, pos].float(), dim=-1)  # (n_cap, V)
        return lp[:, self.stop_id]                          # (n_cap,)

    def _eos_loss(self, model):
        assert self.ref_eos is not None, (
            "EOS term has no theta0 reference: pass ref_model/ref_eos_logprobs at "
            "init or call init_eos_reference(model) before loss()")
        k = min(self.k_caps, len(self.captions))
        idxs = self.rng_e.sample(range(len(self.captions)), k)
        parts = []
        for i in idxs:
            g = self._eos_logprobs(model, self.captions[i])
            g0 = self.ref_eos[i].to(g.device)
            assert g.shape == g0.shape, (
                f"caption {i}: live hazard {tuple(g.shape)} != reference "
                f"{tuple(g0.shape)} (tokenization drifted)")
            if self.eos_mode == "explen":
                parts.append(eos_explen_term(g, g0))
            else:
                parts.append(eos_stability_term(g, g0))
        val = torch.stack(parts).mean()
        assert bool(torch.isfinite(val.detach())), "eos loss non-finite"
        return val

    # -- public API --------------------------------------------------------

    def loss_components(self, model):
        """Dict of raw (unweighted) term values as 0-d tensors; disabled terms are
        a fresh zero tensor (no forward spent). Used for logging and by loss()."""
        zero = torch.zeros((), device=self.device)
        crit = self.crit.loss(model) if self.use_criterion else zero
        abst = self._abstain_loss(model) if self.use_abstain else zero
        eos = self._eos_loss(model) if self.use_eos else zero
        return {"criterion": crit, "abstain": abst, "eos": eos}

    def loss(self, model):
        """Weighted composite L_policy (the training loop adds L_task):

            L_policy = lambda_c*L_criterion + lambda_a*L_abstain + lambda_e*L_eos
        """
        c = self.loss_components(model)
        val = compose_policy_loss(c["criterion"], c["abstain"], c["eos"],
                                  self.lambda_c, self.lambda_a, self.lambda_e)
        assert bool(torch.isfinite(val.detach())), "policy loss non-finite"
        return val
