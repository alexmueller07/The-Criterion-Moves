"""Faithfulness anchor loss (method candidate C2): ground-truth-supervised
yes/no logit margins on counterfactual image pairs.

For a pair (I, I_masked, present, absent) from build_grounding_pairs.py and the
prompt backbone.build_prompt("Is there a {obj} in the image?") — for the
default llava15 backbone that is byte-identical to the previously hardcoded
"USER: <image>\\nIs there a {obj} in the image? ASSISTANT:" — with z_yes /
z_no the next-token logits of " Yes" / " No":

  L = mean over k sampled pairs of                        [v1, arm F; SUPERSEDED]
        softplus(margin - [z_yes(I, present) - z_yes(I_masked, present)])
      + softplus(margin - [z_no(I, absent)   - z_yes(I, absent)])

The LIVE loss is v2 (2026-08-31): three terms, the two symmetric decision
margins on the real image plus the counterfactual-sensitivity term; written out
in loss() below and in faith_margin_terms().

Supervision is the COCO instance annotation itself (present is genuinely visible
in I and covered in I_masked; absent is genuinely unannotated in I) - no teacher
checkpoint, no reference policy, no preference pairs. Gradients flow into the
model; the model is never mutated. Answer-token ids and per-object prompt
encodings are resolved once at init; margins are computed in fp32 regardless of
model dtype.

Counterfactual-term form (cf_form; ABLATION, added 2026-09-08 for R1 of
design_notes/method_ideas_lit1.md, motivated by IC-VCO arXiv 2605.31312v1):

  cf_form="logit"   (DEFAULT, unchanged; the replicated v2 anchor)
      cf = sp(m - [ z_yes(I) - z_yes(I_masked) ])
  cf_form="logodds" (ablation arm "anchorlo")
      cf = sp(m - [ g(I)     - g(I_masked)     ]),  g := z_yes - z_no

z_yes alone is a RAW logit and the two images are two different softmax
contexts with their own log-partition functions, so the "logit" difference can
be moved by a context-wide shift that carries z_no along with it; the
within-image log-odds g is normalization-free (the log-partition cancels
exactly inside each image), so the "logodds" difference measures only what
masking o+ out of the image did to the yes/no decision. Terms 1 and 2 (the
symmetric decision margins, whose criterion-freeze property is DERIVED in
design_notes/method_calibration_theory.md sec.2-3) are IDENTICAL under both
forms; only term 3 changes. The default is byte-identical to the pre-flag
implementation so the pilot and full-study anchor cells stay reproducible.
"""
import json
import os
import random

import torch
import torch.nn.functional as F
from PIL import Image

SEED = 17
QUESTION = "Is there a {obj} in the image?"
CF_FORMS = ("logit", "logodds")


def faith_margin_terms(z_yes, z_no, margin, cf_form="logit"):
    """The v2 three-term symmetric margin loss, elementwise, on the (k, 3)
    answer logits laid out by loss(): column 0 = (real image, present object),
    1 = (GT-masked image, present object), 2 = (real image, absent object).

        present_decision = sp(m - [z_yes - z_no](real, present))
        absent_decision  = sp(m - [z_no - z_yes](real, absent))
        cf_sensitivity   = sp(m - D),  D = the counterfactual gap, see below

    cf_form selects D ONLY (terms 1-2 are the same for both):
        "logit"   D = z_yes(real, present) - z_yes(masked, present)     [default]
        "logodds" D = g(real, present)     - g(masked, present),  g = z_yes - z_no

    Returns the three (k,) tensors; reduce with (a + b + c).mean() outside.
    Pure math: no model, no I/O, unit-tested on CPU in
    tests/test_faith_cf_form.py."""
    assert cf_form in CF_FORMS, f"cf_form must be one of {CF_FORMS}, got {cf_form!r}"
    assert margin >= 0.0, f"margin must be >= 0, got {margin}"
    present_decision = F.softplus(margin - (z_yes[:, 0] - z_no[:, 0]))
    absent_decision = F.softplus(margin - (z_no[:, 2] - z_yes[:, 2]))
    if cf_form == "logodds":
        # Normalization-free: the per-image log-partition cancels inside each g.
        cf_sensitivity = F.softplus(
            margin - ((z_yes[:, 0] - z_no[:, 0]) - (z_yes[:, 1] - z_no[:, 1])))
    else:
        cf_sensitivity = F.softplus(margin - (z_yes[:, 0] - z_yes[:, 1]))
    return present_decision, absent_decision, cf_sensitivity


def resolve_answer_token_ids(tokenizer, backbone):
    """(yes_id, no_id): the first token the model emits for ' Yes' / ' No'
    after the backbone's generation prefix, resolved IN CONTEXT.

    ' Yes' in isolation tokenizes to [space-marker, "Yes"] under the Llama
    tokenizer (caught by the original assert in the first smoke run). The id we
    need is the first token by which tok(stub + answer) extends tok(stub),
    where stub = backbone.anchor_stub() ends exactly like build_prompt() does.
    Shared by FaithfulnessAnchor.__init__ and eval_gen.py --dump_logits so the
    training loss and the logit instrument read the SAME two vocabulary rows.
    Torch-free."""
    stub = backbone.anchor_stub()
    stub_ids = tokenizer(stub, add_special_tokens=False)["input_ids"]
    ids = {}
    for text in (" Yes", " No"):
        full_ids = tokenizer(stub + text, add_special_tokens=False)["input_ids"]
        assert full_ids[:len(stub_ids)] == stub_ids, (
            f"context tokenization unstable for {text!r}: "
            f"{full_ids[:len(stub_ids)]} vs {stub_ids}")
        assert len(full_ids) > len(stub_ids), f"no answer token for {text!r}"
        ids[text] = full_ids[len(stub_ids)]
    assert ids[" Yes"] != ids[" No"]
    return ids[" Yes"], ids[" No"]


class FaithfulnessAnchor:
    def __init__(self, processor, pairs_jsonl, data_root, k_pairs=4, margin=2.0,
                 device="cuda", backbone=None, mode="margin", cf_form="logit"):
        # mode="margin": the v2 three-term symmetric margin loss (the method).
        # mode="ce": CE-on-counterfactuals ablation (CSS/MUTANT-style baseline a
        # reviewer will ask for): plain 2-way cross-entropy over the yes/no
        # logits with targets yes on (real, present), no on (masked, present),
        # no on (real, absent). Same pairs, same positions, no margins.
        assert mode in ("margin", "ce"), mode
        self.mode = mode
        # cf_form selects the QUANTITY compared across the two images in the
        # third (counterfactual) term only; "logit" reproduces the replicated v2
        # anchor exactly, "logodds" is the R1 ablation (arm "anchorlo"). See the
        # module docstring and faith_margin_terms(). No effect in mode="ce".
        assert cf_form in CF_FORMS, f"cf_form must be one of {CF_FORMS}, got {cf_form!r}"
        self.cf_form = cf_form
        if backbone is None:  # default preserves pre-refactor behavior exactly
            from model_zoo import get_backbone
            backbone = get_backbone("llava15")
        self.bb = backbone
        self.p = processor
        self.data_root = data_root
        self.k = int(k_pairs)
        self.margin = float(margin)
        self.device = torch.device(device)
        self.rng = random.Random(SEED)
        assert self.margin >= 0.0, f"margin must be >= 0, got {self.margin}"

        with open(pairs_jsonl) as f:
            self.pairs = [json.loads(l) for l in f]
        assert 1 <= self.k <= len(self.pairs), (
            f"k_pairs={self.k} but only {len(self.pairs)} pairs in {pairs_jsonl}")
        for r in self.pairs:
            for key in ("id", "image", "masked_image", "present", "absent"):
                assert key in r, f"pair row missing {key!r}: {r}"
            for key in ("image", "masked_image"):
                path = os.path.join(data_root, r[key])
                assert os.path.exists(path), f"missing pair image {path}"

        # Answer-token ids must be resolved IN CONTEXT (see
        # resolve_answer_token_ids; hoisted 2026-09-08 so eval_gen.py
        # --dump_logits reads the same two vocabulary rows -- identical asserts
        # and results to the inline version it replaces).
        tok = processor.tokenizer
        self.yes_id, self.no_id = resolve_answer_token_ids(tok, self.bb)
        self.pad_id = tok.pad_token_id
        if self.pad_id is None:
            self.pad_id = tok.eos_token_id
        assert self.pad_id is not None, "tokenizer has neither pad nor eos token"
        # NOTE: earlier versions cached per-object text encodings and computed
        # pixel_values separately, assuming input_ids are image-independent
        # (true for llava-hf). That breaks on Qwen2.5-VL, whose image-token
        # count varies with resolution and whose forward needs image_grid_thw
        # (smoke 2026-09-05: 'NoneType' is not iterable). loss() now runs the
        # full processor on (text, image) together per forward and passes every
        # processor output to the model -- backbone-agnostic. For llava-hf this
        # is numerically identical: input_ids are image-invariant and
        # pixel_values are computed from the same images either way.

    def _load(self, rel):
        img = Image.open(os.path.join(self.data_root, rel)).convert("RGB")
        return self.bb.anchor_image(img)  # backbone hook (Qwen downsizes; LLaVA no-op)

    def _answer_logits(self, model, rows):
        """Forward `rows` [(image_rel, obj), ...] and return the (n,) tensors of
        ' Yes' / ' No' logits at each row's last real token position."""
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
        return step[:, self.yes_id], step[:, self.no_id]

    def loss(self, model):
        sampled = self.rng.sample(self.pairs, self.k)
        # Forward ONE pair (3 images) at a time and concatenate: peak vision
        # memory is bounded to 3 images regardless of k. Qwen2.5-VL (8.45B) OOMs
        # on a 24GB card if all k*3 images go through the vision tower at once
        # (smoke 2026-09-05). Result is identical to a single batched forward.
        yes_parts, no_parts = [], []
        for r in sampled:
            rows = [(r["image"], r["present"]),
                    (r["masked_image"], r["present"]),
                    (r["image"], r["absent"])]
            zy, zn = self._answer_logits(model, rows)
            yes_parts.append(zy)
            no_parts.append(zn)
        z_yes = torch.stack(yes_parts, dim=0)  # (k, 3)
        z_no = torch.stack(no_parts, dim=0)    # (k, 3)

        # v2 (2026-08-31): v1's two terms were asymmetric on the yes/no decision
        # axis - the absent term boosted "No" directly while the present term
        # was only a real-vs-masked relative margin, so net pressure said No:
        # the pilot F-arm swung hyper-conservative (c=2.02 at stage 1, yes-rate
        # 7%). v2 balances the decision axis (one yes-margin, one no-margin on
        # real images) and keeps the counterfactual sensitivity term separately.
        if self.mode == "ce":
            two = torch.stack([torch.stack([z_yes[:, i], z_no[:, i]], dim=1)
                               for i in range(3)], dim=1)  # (k, 3, 2)
            tgt = torch.tensor([0, 1, 1], device=two.device)  # yes, no, no
            val = F.cross_entropy(two.reshape(-1, 2),
                                  tgt.repeat(self.k)).mean()
        else:
            present_decision, absent_decision, cf_sensitivity = faith_margin_terms(
                z_yes, z_no, self.margin, self.cf_form)
            val = (present_decision + absent_decision + cf_sensitivity).mean()
        assert bool(torch.isfinite(val.detach())), "faith loss non-finite"
        return val
