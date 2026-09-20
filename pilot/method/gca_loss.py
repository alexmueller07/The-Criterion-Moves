"""Generative counterfactual token anchor (method candidate GCA;
design_notes/method_gca.md, design_notes/method_ideas_lit1.md sec. 2.2).

The GT anchor (faith_loss.FaithfulnessAnchor) constrains the yes/no head at the
answer position of a QUESTION. The generative instrument (length-controlled
CHAIR_i@60) reads a different decision: at the caption position where an object
name is emitted, which object wins. Nothing on the table constrains that
decision, and it is the one that rises S1->S3 while d' stays flat. GCA puts the
anchor's GT supervision (present / GT-masked / co-occurring absent) on that
caption-slot decision.

For a grounding pair (I, I^{\\o+}, o+, o-) and a GT COCO caption y of I that
mentions o+, let t be the FIRST token position of o+ in y (earliest CHAIR
surface form of o+; first subword). Teacher-force the caption prefix y_<t under
the caption prompt on image J and read the two vocabulary rows at the slot:

    l_t(J) = log p(o+_t | J, y_<t) - log p(o-_t | J, y_<t) = z(o+) - z(o-)

(the within-context log-odds; the partition function cancels, so no
log-softmax and no cross-image normalization issue). With sp = softplus:

    L_GCA = mean_k [ sp(m - l_t(I))                          # Term 1 "decision"
                   + sp(m - [l_t(I) - l_t(I^{\\o+})]) ]      # Term 2 "cf"

Term 1: at the slot, the present object must beat the co-occurring absent object
by a margin (the generative twin of the anchor's present/absent decision terms;
HALVA 2405.18654v3's GT phrase ratio in margin form at one GT slot). Term 2:
masking o+ out of the image must lower that log-odds by a margin (the twin of
the anchor's cf_sensitivity; TPO 2412.14487v4's raw-vs-corrupted token quantity
with a GT instance mask and a margin TARGET instead of a reward weight).
Honest label: a RECOMBINATION of those two ingredients on the anchor's GT
pairs, reference-free, inside the CL loop; its claim to exist is that it is the
first term aimed at the length-controlled CHAIR rise.

Supervision is COCO ground truth only (instance annotations for present/absent
and the mask; human captions for the slot). No teacher, no reference policy,
no preference pairs; the model is never mutated.

Design constraints inherited from faith_loss.py so it fits 24GB on Qwen2.5-VL:
full processor call on (text, image) together per forward, every processor
output passed to the model (backbone-agnostic); <= CHUNK_IMAGES (2) images per
forward -- one pair = (real, masked) with the SAME prefix text -- and results
concatenated; slot logits read at attention_mask.sum(1) - 1 (sequence tail, so
image-token expansion never shifts them); fp32 for the two gathered logits
only; slot token ids resolved once, IN CONTEXT, at init (the leading-space /
cross-boundary-merge trap resolve_answer_token_ids guards); images go through
backbone.anchor_image().

The slot-finding and the penalty math are pure functions (find_object_slot /
slot_texts / resolve_slot_ids / build_slot_rows / gca_terms / gca_penalty),
unit-tested on CPU with synthetic tokenizers in tests/test_gca_loss.py; the
class wires them to model forwards and is exercised by the cluster smoke gate.
"""
import json
import os
import random
import re
from collections import Counter

import torch
import torch.nn.functional as F
from PIL import Image

from metrics_chair import SYN as CHAIR_SYN  # the surface forms CHAIR counts as a mention
from method.policy_anchor import CAPTION_PROMPT  # "Describe the image in detail." (shared with the EOS term)

SEED = 17
CHUNK_IMAGES = 2   # one pair = (real, masked) per forward; bounded vision memory
MAX_LEN = 2048


# ---------------------------------------------------------------------------
# Slot finding (pure string / tokenizer functions; torch-free). Unit-tested.
# ---------------------------------------------------------------------------

def find_object_slot(caption, obj, synonyms=None):
    """Earliest mention of `obj` in `caption`: (char_start, surface) or None.

    Surface forms = the object's CHAIR synonym list (so the slot is exactly a
    position CHAIR would score as a mention of `obj`; the class name itself is
    always included), matched case-insensitively at word boundaries with an
    optional plural suffix (s/es). Ties at the same start prefer the longest
    match ("fire hydrant" over "fire"). `synonyms=None` -> class name only."""
    forms = list((synonyms or {}).get(obj, []))
    if obj not in forms:
        forms.insert(0, obj)
    best = None
    for w in forms:
        pat = re.compile(r"\b" + re.escape(w) + r"(?:s|es)?\b", re.IGNORECASE)
        m = pat.search(caption)
        if m is None:
            continue
        cand = (m.start(), m.group(0))
        if best is None or cand[0] < best[0] or (cand[0] == best[0] and len(cand[1]) > len(best[1])):
            best = cand
    return best


def slot_texts(backbone, caption_prompt, caption, start, surface, absent):
    """(pre_text, pos_text, neg_text) for a slot.

    pre_text : caption prompt + the caption prefix y_<t, ending at the last
               character BEFORE the whitespace that precedes the object (so the
               object token carries its leading space, as in generation; the
               prefix never ends with a space -- the smoke lesson of 2026-08-31).
    pos_text : pre_text + lead + surface   == prompt + sep + caption[:start+len(surface)]
    neg_text : pre_text + lead + absent (casing copied from `surface`)

    `sep` (the backbone's prefix->target separator: " " for LLaVA, "" for Qwen)
    is read off build_train_text so the teacher-forced text is byte-identical to
    what the train collator would feed for this caption. `lead` is the actual
    whitespace run before the object (== sep when the object opens the caption).
    """
    full, prefix, frag = backbone.build_train_text(caption_prompt, caption, "")
    idx = frag.find(caption)
    assert idx >= 0, f"build_train_text fragment does not contain the caption: {frag[:40]!r}"
    sep = frag[:idx]
    assert surface and caption[start:start + len(surface)] == surface, (start, surface, caption)
    before = sep + caption[:start]
    before_stripped = before.rstrip()
    lead = before[len(before_stripped):]
    neg_word = absent
    if surface[0].isupper() and absent:
        neg_word = absent[0].upper() + absent[1:]
    pre_text = prefix + before_stripped
    pos_text = pre_text + lead + surface
    neg_text = pre_text + lead + neg_word
    assert pos_text == prefix + sep + caption[:start + len(surface)]
    return pre_text, pos_text, neg_text


def resolve_slot_ids(tok, pre_text, pos_text, neg_text):
    """(pos_id, neg_id, n_prefix): the first token by which tok(pos_text) /
    tok(neg_text) extend tok(pre_text) -- the in-context resolution of
    resolve_answer_token_ids, applied at a caption slot. None if either context
    tokenization is unstable (a cross-boundary BPE merge changed the prefix
    tokens) or adds no token: such rows are SKIPPED, never guessed."""
    pre = list(tok(pre_text, add_special_tokens=False)["input_ids"])
    out = []
    for text in (pos_text, neg_text):
        ids = list(tok(text, add_special_tokens=False)["input_ids"])
        if ids[:len(pre)] != pre or len(ids) <= len(pre):
            return None
        out.append(int(ids[len(pre)]))
    return out[0], out[1], len(pre)


def build_slot_rows(pairs, captions_by_id, tok, backbone, caption_prompt=CAPTION_PROMPT,
                    synonyms=None):
    """Join grounding pairs to their captions and resolve one slot per pair.

    Returns (rows, report). Each row carries the pair fields plus caption,
    surface, slot_char, slot_word (0-based word index of the slot in the
    caption), pre_text, pos_id, neg_id, n_prefix. Pairs are skipped (and
    counted in `report`) when: no caption row (no_caption); the object never
    appears in the caption (no_mention); the context tokenization is unstable
    (unstable_tokenization); the present and absent objects share their first
    subword, e.g. "baseball bat" vs "baseball glove" (same_first_token)."""
    rows, report = [], Counter()
    for r in pairs:
        cap = captions_by_id.get(r["id"])
        if cap is None:
            report["no_caption"] += 1
            continue
        hit = find_object_slot(cap, r["present"], synonyms)
        if hit is None:
            report["no_mention"] += 1
            continue
        start, surface = hit
        pre_text, pos_text, neg_text = slot_texts(backbone, caption_prompt, cap,
                                                  start, surface, r["absent"])
        ids = resolve_slot_ids(tok, pre_text, pos_text, neg_text)
        if ids is None:
            report["unstable_tokenization"] += 1
            continue
        pos_id, neg_id, n_prefix = ids
        if pos_id == neg_id:
            report["same_first_token"] += 1
            continue
        row = dict(r)
        row.update({"caption": cap, "surface": surface, "slot_char": start,
                    "slot_word": len(cap[:start].split()), "pre_text": pre_text,
                    "pos_id": pos_id, "neg_id": neg_id, "n_prefix": n_prefix})
        rows.append(row)
        report["joined"] += 1
    return rows, dict(report)


# ---------------------------------------------------------------------------
# Pure penalty math (torch tensors in/out; no model, no I/O). Unit-tested.
# ---------------------------------------------------------------------------

def slot_log_odds(z_pos, z_neg):
    """l = z(o+) - z(o-) in fp32: the within-context log-odds at the slot.
    Equals log p(o+) - log p(o-) exactly (the log-partition cancels)."""
    return z_pos.float() - z_neg.float()


def gca_terms(ell_real, ell_masked, margin):
    """Elementwise (decision, cf) terms for slot log-odds on the real image
    l(I) and on the GT-masked image l(I \\ o+):

        decision = sp(m - l(I))                  present beats absent at the slot
        cf       = sp(m - [l(I) - l(I \\ o+)])     masking o+ out lowers the log-odds

    Both ~0 (asymptotically; softplus soft hinge as in faith_loss) once the
    margins are met, > 0 otherwise. Reduce with .mean() outside."""
    assert margin >= 0.0, f"margin must be >= 0, got {margin}"
    ell_real = ell_real.float()
    ell_masked = ell_masked.float()
    assert ell_real.shape == ell_masked.shape, (ell_real.shape, ell_masked.shape)
    decision = F.softplus(margin - ell_real)
    cf = F.softplus(margin - (ell_real - ell_masked))
    return decision, cf


GCA_TERMS = ("both", "decision", "cf")


def gca_penalty(ell_real, ell_masked, margin, terms="both"):
    """L_GCA = mean(decision) + mean(cf); 0-d fp32 tensor.

    `terms` selects which of the two terms is active, so the published
    neighbours can be run as baselines IN OUR CONTINUAL SETTING rather than
    argued against in prose (vocab-B: Term 2 is crowded by See-or-Guess, and
    Term 1 is HALVA's ratio in margin form):
      "decision" -> Term 1 only  (the HALVA-style slot-decision baseline)
      "cf"       -> Term 2 only  (the See-or-Guess-style counterfactual baseline)
      "both"     -> GCA as proposed.
    Any claim that GCA beats its own preemptors must be read off these arms."""
    assert terms in GCA_TERMS, f"terms must be one of {GCA_TERMS}, got {terms!r}"
    d, c = gca_terms(ell_real, ell_masked, margin)
    if terms == "decision":
        val = d.mean()
    elif terms == "cf":
        val = c.mean()
    else:
        val = d.mean() + c.mean()
    assert bool(torch.isfinite(val.detach())), "gca penalty non-finite"
    return val


# ---------------------------------------------------------------------------
# The anchor.
# ---------------------------------------------------------------------------

class GenerativeCounterfactualAnchor:
    """Caption-slot counterfactual margin anchor. `.loss(model)` returns the
    fp32 L_GCA over k freshly sampled joined pairs; `.loss_components(model)`
    returns the per-term breakdown (decision, cf, and the detached monitors
    ell_real, ell_gap) for logging.

    Args
      processor     : HF processor for the backbone (tokenizer + image proc).
      pairs_jsonl   : {id, image, masked_image, present, absent} rows
                      (build_grounding_pairs.py); paths resolve against data_root.
      captions_jsonl: {id, image, caption} rows (build_caption_anchor.py) joined
                      to the pairs on `id`; a caption row whose image differs
                      from its pair's is rejected (stale/mismatched file).
      data_root     : root the image paths resolve against.
      k             : pairs sampled per step (2 images each, forwarded per pair).
      margin        : m (logits) for both terms.
      device        : forward device.
      backbone      : model_zoo adapter (default llava15, as faith_loss).
      caption_prompt: instruction the caption is teacher-forced under (default
                      the EOS term's "Describe the image in detail.").
      synonyms      : surface-form table for slot finding (default the vendored
                      CHAIR list; pass {} for class-name-only matching).
    """

    def __init__(self, processor, pairs_jsonl, captions_jsonl, data_root, k=4,
                 margin=2.0, device="cuda", backbone=None,
                 caption_prompt=CAPTION_PROMPT, synonyms=CHAIR_SYN):
        if backbone is None:  # same default as faith_loss
            from model_zoo import get_backbone
            backbone = get_backbone("llava15")
        self.bb = backbone
        self.p = processor
        self.data_root = data_root
        self.k = int(k)
        self.margin = float(margin)
        assert self.margin >= 0.0, f"margin must be >= 0, got {self.margin}"
        self.device = torch.device(device)
        self.rng = random.Random(SEED)
        self.caption_prompt = caption_prompt
        self.pairs_path, self.captions_path = pairs_jsonl, captions_jsonl

        with open(pairs_jsonl) as f:
            self.pairs = [json.loads(l) for l in f if l.strip()]
        assert self.pairs, f"empty pairs file {pairs_jsonl}"
        for r in self.pairs:
            for key in ("id", "image", "masked_image", "present", "absent"):
                assert key in r, f"pair row missing {key!r}: {r}"
            for key in ("image", "masked_image"):
                path = os.path.join(data_root, r[key])
                assert os.path.exists(path), f"missing pair image {path}"

        with open(captions_jsonl) as f:
            caps = [json.loads(l) for l in f if l.strip()]
        assert caps, f"empty caption file {captions_jsonl}"
        image_of_pair = {r["id"]: r["image"] for r in self.pairs}
        captions_by_id = {}
        for c in caps:
            for key in ("id", "image", "caption"):
                assert key in c, f"caption row missing {key!r}: {c}"
            assert c["id"] not in captions_by_id, f"duplicate caption id {c['id']!r}"
            if c["id"] in image_of_pair:
                assert c["image"] == image_of_pair[c["id"]], (
                    f"caption row {c['id']!r} points at {c['image']!r} but its pair at "
                    f"{image_of_pair[c['id']]!r} - caption file does not belong to this pair set")
            captions_by_id[c["id"]] = c["caption"]

        tok = processor.tokenizer
        self.rows, report = build_slot_rows(self.pairs, captions_by_id, tok, self.bb,
                                            caption_prompt=caption_prompt,
                                            synonyms=synonyms)
        report.update({"n_pairs": len(self.pairs), "n_captions": len(caps)})
        if self.rows:
            sw = [r["slot_word"] for r in self.rows]
            npre = [r["n_prefix"] for r in self.rows]
            report.update({"mean_slot_word": round(sum(sw) / len(sw), 2),
                           "max_slot_word": max(sw),
                           "mean_prefix_tokens": round(sum(npre) / len(npre), 1)})
        self.report = report
        assert 1 <= self.k <= len(self.rows), (
            f"gca k={self.k} but only {len(self.rows)} pairs joined to a caption slot "
            f"(report {report}) - captions must be built from THIS pair set "
            "(build_caption_anchor.py) and mention the present object")

    # -- forwards (faith_loss pattern) ---------------------------------------

    def _load(self, rel):
        img = Image.open(os.path.join(self.data_root, rel)).convert("RGB")
        return self.bb.anchor_image(img)  # backbone hook (Qwen downsizes; LLaVA no-op)

    def _slot_logits(self, model, rows):
        """Forward <= CHUNK_IMAGES rows [(image_rel, pre_text, pos_id, neg_id)]
        and return the (n,) fp32 logits of the present and absent slot tokens
        at each row's last real position (the slot's next-token distribution)."""
        assert 1 <= len(rows) <= CHUNK_IMAGES, len(rows)
        param = next(model.parameters())
        images = [self._load(img) for img, _, _, _ in rows]
        texts = [t for _, t, _, _ in rows]
        enc = self.p(text=texts, images=images, return_tensors="pt",
                     padding=True, truncation=True, max_length=MAX_LEN)
        assert enc["input_ids"].shape[1] < MAX_LEN, (
            "slot prefix hit max_length; the tail position would not be the slot")
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
        pos = torch.tensor([p for _, _, p, _ in rows], device=logits.device)
        neg = torch.tensor([n for _, _, _, n in rows], device=logits.device)
        assert logits.shape[-1] > int(max(pos.max(), neg.max()))
        last = attn.sum(dim=1) - 1
        idx = last.view(-1, 1, 1).expand(-1, 1, logits.shape[-1])
        step = logits.gather(1, idx).squeeze(1).float()  # (n, V) fp32
        z_pos = step.gather(1, pos.view(-1, 1)).squeeze(1)
        z_neg = step.gather(1, neg.view(-1, 1)).squeeze(1)
        return z_pos, z_neg

    # -- public API --------------------------------------------------------

    def loss_components(self, model):
        """Sample k joined pairs; per pair ONE forward of (real, masked) with the
        same teacher-forced prefix. Returns 0-d tensors: decision / cf (the two
        loss terms, mean over pairs, with grad) and ell_real / ell_gap (detached
        monitors: mean l(I) and mean l(I) - l(I \\ o+))."""
        sampled = self.rng.sample(self.rows, self.k)
        ell_real, ell_masked = [], []
        for r in sampled:
            zp, zn = self._slot_logits(model, [
                (r["image"], r["pre_text"], r["pos_id"], r["neg_id"]),
                (r["masked_image"], r["pre_text"], r["pos_id"], r["neg_id"])])
            ell = slot_log_odds(zp, zn)  # (2,) = [l(I), l(I \ o+)]
            ell_real.append(ell[0])
            ell_masked.append(ell[1])
        ell_real = torch.stack(ell_real)
        ell_masked = torch.stack(ell_masked)
        d, c = gca_terms(ell_real, ell_masked, self.margin)
        return {"decision": d.mean(), "cf": c.mean(),
                "ell_real": ell_real.detach().mean(),
                "ell_gap": (ell_real - ell_masked).detach().mean()}

    def loss(self, model):
        """L_GCA = mean decision + mean cf (fp32, 0-d, with grad)."""
        c = self.loss_components(model)
        val = c["decision"] + c["cf"]
        assert bool(torch.isfinite(val.detach())), "gca loss non-finite"
        return val
