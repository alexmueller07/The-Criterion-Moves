"""Backbone adapter registry for the pilot train/eval harness.

One adapter class per backbone. Every backbone-specific string, model class,
dtype/attention setting, freeze rule, and LoRA target lives here;
train_lora.py, eval_gen.py, and method/faith_loss.py route through
get_backbone(name) and are otherwise backbone-agnostic.

Adapter interface (every backbone class exposes exactly this):
  model_id                                   default HF id for the backbone
  load(model_id, device_map=None)         -> (model, processor)
  build_prompt(prompt_text)               -> generation-ready prefix string
  build_train_text(prompt_text, target_text, eos)
                                          -> (full_text, prefix_text,
                                              target_fragment)
  freeze_policy(model)                       freezes the vision tower in place
  lora_kwargs(r)                          -> plain dict of LoraConfig fields
                                             (torch-free, for parity tests)
  lora_config(r)                          -> peft.LoraConfig built from those
  anchor_stub()                           -> in-context stub string used by
                                             FaithfulnessAnchor token resolution
  generation_kwargs(processor)            -> generation defaults passthrough
                                             for model.generate(**...)

Behavior contract for "llava15": byte-identical to the strings/configs that
were hardcoded in train_lora.py / eval_gen.py / method/faith_loss.py before
this module existed. Verified by tests/test_backbone_parity.py; do not edit
the llava15 literals without re-running that test.

This module must stay importable WITHOUT torch/transformers/peft (the parity
test runs at pure string/config level on python3.9), so all heavy imports are
deferred into the methods that need them. Keep the file 3.9-importable: no
match statements, no PEP 604 unions.
"""


def _lora_overrides(backbone_name, default_targets, default_save):
    """Apply the optional drift-localization env overrides to (targets, save).

    Module-level ON PURPOSE: each backbone class defines its own lora_kwargs, so
    putting this logic in one class silently leaves the other unaffected -- which
    is how the first version of this override shipped, working for LLaVA and
    doing nothing at all for Qwen.

      CLH_LORA_BAND   "early" | "late"  restrict adaptation to layers 0-15 / 16-31.
                      A NAME, not a regex: queue.txt is pipe-delimited and every
                      target regex contains "|". Both bands adapt the same number
                      of modules, so the contrast isolates location, not capacity.
      CLH_LORA_TARGETS  replace target_modules outright; CLH_LORA_BAND wins.
      CLH_LORA_SAVE     replace modules_to_save; "none"/"" means freeze them.

    Unset env leaves both arguments exactly as passed, so every arm that does not
    opt in is numerically unchanged.

    Why these two knobs and not a third: the LM head is in neither target_modules
    nor modules_to_save for either backbone, so it is already frozen everywhere,
    and the criterion drifts regardless. That rules the head's weights out and
    leaves the projector and the adapted decoder layers as the candidates.
    """
    import os
    BANDS = {"early": r"([0-9]|1[0-5])", "late": r"(1[6-9]|2[0-9]|3[01])"}
    band = os.environ.get("CLH_LORA_BAND", "").strip().lower()
    targets = os.environ.get("CLH_LORA_TARGETS") or default_targets
    if band:
        if band not in BANDS:
            raise SystemExit("CLH_LORA_BAND must be one of %s, got %r"
                             % (sorted(BANDS), band))
        if backbone_name == "llava15":
            targets = (r".*language_model.*\.layers\.%s\..*\."
                       r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
                       % BANDS[band])
        else:
            targets = (r".*model\.layers\.%s\.(self_attn\.(q_proj|k_proj|v_proj|"
                       r"o_proj)|mlp\.(gate_proj|up_proj|down_proj))$" % BANDS[band])
    sav = os.environ.get("CLH_LORA_SAVE")
    if sav is None:
        save = list(default_save)
    else:
        save = [] if sav.strip().lower() in ("none", "") else [m for m in sav.split(",") if m]
    return targets, save


class Llava15Backbone:
    """llava-hf/llava-1.5-7b-hf -- the pilot's original backbone."""

    name = "llava15"
    model_id = "llava-hf/llava-1.5-7b-hf"

    # LoRA literals, copied verbatim from the original train_lora.py.
    # lora_alpha is always 2*r; dropout and targets are fixed.
    LORA_TARGET_MODULES = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
    LORA_MODULES_TO_SAVE = ("multi_modal_projector",)
    LORA_DROPOUT = 0.05
    LORA_TASK_TYPE = "CAUSAL_LM"

    def load(self, model_id, device_map=None):
        """(model, processor) with this backbone's dtype/attn settings.

        device_map is an optional passthrough (eval loads with
        device_map="cuda:0"; training loads on CPU and moves later --
        identical to the pre-refactor call sites). Caller-side concerns stay
        caller-side: tokenizer padding_side (right for training, left for
        generation) and model.config.use_cache.
        """
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_id)
        kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        if device_map is not None:
            kwargs["device_map"] = device_map
        model = LlavaForConditionalGeneration.from_pretrained(model_id, **kwargs)
        return model, processor

    def build_prompt(self, prompt_text):
        """Full generation-ready prefix. NOTE: must end WITHOUT a trailing
        space -- "ASSISTANT: " + "A" merges the space into the answer token
        ("_A"), the prefix-length mask then swallows the whole answer and only
        EOS gets supervised (loss ~0 from step 1; caught by the smoke run on
        2026-08-31). The leading space belongs to the target fragment."""
        return f"USER: <image>\n{prompt_text} ASSISTANT:"

    def build_train_text(self, prompt_text, target_text, eos):
        """(full_text, prefix_text, target_fragment) for the train collator.

        Byte-identical to the original collator construction:
          prefix          = "USER: <image>\\n{prompt} ASSISTANT:"
          target_fragment = " " + target + eos      (leading space mandatory)
          full            = prefix + target_fragment
        """
        prefix = self.build_prompt(prompt_text)
        target_fragment = " " + target_text + eos
        return prefix + target_fragment, prefix, target_fragment

    def freeze_policy(self, model):
        """Freeze the vision tower (LLaVA: model.vision_tower), in place."""
        for p in model.vision_tower.parameters():
            p.requires_grad_(False)

    def lora_kwargs(self, r):
        """LoraConfig fields as a plain dict (torch/peft-free; parity-tested).

        Optional localization overrides are applied by _lora_overrides(); with no
        env set the result is byte-identical to the pre-2026-09-11 config.
        """
        tgt, save = _lora_overrides(self.name, self.LORA_TARGET_MODULES,
                                    self.LORA_MODULES_TO_SAVE)
        return dict(
            r=r,
            lora_alpha=r * 2,
            lora_dropout=self.LORA_DROPOUT,
            target_modules=tgt,
            modules_to_save=save,
            task_type=self.LORA_TASK_TYPE,
        )

    def lora_config(self, r):
        from peft import LoraConfig
        return LoraConfig(**self.lora_kwargs(r))

    def anchor_stub(self):
        """In-context stub for FaithfulnessAnchor answer-token resolution:
        the answer id is the first token by which tok(stub + " Yes") extends
        tok(stub). Must end exactly like build_prompt() output does
        ("ASSISTANT:" with no trailing space)."""
        return "USER: hi ASSISTANT:"

    def anchor_image(self, img):
        """Preprocess a grounding-pair image for the anchor forward. LLaVA's
        CLIP tower resizes to 336 internally, so pass through unchanged (keeps
        the pilot anchor numerically identical)."""
        return img

    def generation_kwargs(self, processor):
        """Generation defaults passthrough for model.generate(**enc, ...).
        Greedy decoding; pad id from the tokenizer (pre-refactor literals)."""
        return dict(do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id)


class Qwen25VLBackbone:
    """Qwen2.5-VL-7B-Instruct. Facts filled from the dependency probe
    (clh_fsprobe_18078, 2026-09-03): model class Qwen2_5_VLForConditionalGeneration,
    chat-template render captured verbatim, LM projections at model.layers.N,
    vision tower at `visual.`, Qwen2VLImageProcessor (max_pixels default
    12845056 -> capped to bound visual tokens on a 24GB 4090).

    THREE conventions the probe's forward pass could NOT confirm (it crashed on
    the Blackwell card before generating): the assistant-turn eos terminator,
    the leading-space rule for the target fragment, and whether the visual
    merger must stay trainable. Best-informed choices are made below and left
    for the smoke gate's hard asserts to validate -- the collator's
    unmasked==target-token boundary assert and the pre-train forward-loss check
    fail loudly if any is wrong, so no real run proceeds on a bad guess. Config
    asymmetry vs LLaVA (visual merger frozen here, projector trained there) is
    intentional and recorded in fullstudy/FULLSTUDY_PREREG.md."""

    name = "qwen25vl"
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    # Anchored to `model.layers.` so the frozen vision tower (`visual.*`) can
    # never match; fullmatch semantics, leading .* for any wrapper prefix.
    LORA_TARGET_MODULES = (r".*model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|"
                           r"v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$")
    LORA_MODULES_TO_SAVE = ()  # visual merger stays frozen (see class docstring)
    LORA_DROPOUT = 0.05
    LORA_TASK_TYPE = "CAUSAL_LM"
    TURN_EOS = "<|im_end|>"     # assistant-turn terminator in Qwen chat format
    MAX_PIXELS = 768 * 28 * 28  # <=768 visual tokens/image; probe value

    def load(self, model_id, device_map=None):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(model_id)
        processor.image_processor.max_pixels = self.MAX_PIXELS
        self._guard_tiny_images(processor.image_processor)
        kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa")
        if device_map is not None:
            kwargs["device_map"] = device_map
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        return model, processor

    MIN_SIDE = 28  # Qwen2.5-VL patch factor: smart_resize raises below this

    @classmethod
    def _guard_tiny_images(cls, image_processor):
        """Qwen2VLImageProcessor.smart_resize raises ValueError for any image
        with a side < 28px ("height:26 or width:392 must be larger than
        factor:28"; hit by a real eval image, 2026-09-08). Wrap preprocess to
        upscale such images (aspect preserved) before the processor sees them.
        Patching the processor covers EVERY call site (train collator, evals,
        anchor / probe forwards) without touching them."""
        from PIL import Image

        def _up(im):
            if isinstance(im, Image.Image):
                w, h = im.size
                m = min(w, h)
                if m < cls.MIN_SIDE:
                    s = cls.MIN_SIDE / m
                    return im.resize((max(cls.MIN_SIDE, round(w * s)),
                                      max(cls.MIN_SIDE, round(h * s))))
            return im

        orig = image_processor.preprocess

        def preprocess(images, *a, **k):
            if isinstance(images, (list, tuple)):
                images = [[_up(i) for i in x] if isinstance(x, (list, tuple)) else _up(x)
                          for x in images]
            else:
                images = _up(images)
            return orig(images, *a, **k)

        image_processor.preprocess = preprocess

    def build_prompt(self, prompt_text):
        # Verbatim from the probe's apply_chat_template render (one user turn,
        # one image, add_generation_prompt=True). Ends at "assistant\n" so
        # generation starts at the answer position.
        return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                f"{prompt_text}<|im_end|>\n<|im_start|>assistant\n")

    def build_train_text(self, prompt_text, target_text, eos):
        # eos param (tokenizer.eos_token) is ignored: the chat-format turn
        # terminator is <|im_end|>, a recognized special token, so
        # tok("...<|im_end|>", add_special_tokens=False) yields its single id
        # and the prefix/target boundary stays clean (no cross-boundary BPE
        # merge after "assistant\n"). No leading space: Qwen BPE does not
        # prepend a word-boundary marker the way Llama SentencePiece does.
        prefix = self.build_prompt(prompt_text)
        target_fragment = target_text + self.TURN_EOS
        return prefix + target_fragment, prefix, target_fragment

    def freeze_policy(self, model):
        # Freeze the entire vision tower incl. the merger (see class docstring
        # for the intentional asymmetry vs LLaVA's trainable projector).
        for name, p in model.named_parameters():
            if name.startswith("visual.") or ".visual." in name:
                p.requires_grad_(False)

    def lora_kwargs(self, r):
        """See Llava15Backbone.lora_kwargs; overrides applied by _lora_overrides()."""
        tgt, save = _lora_overrides(self.name, self.LORA_TARGET_MODULES,
                                    self.LORA_MODULES_TO_SAVE)
        return dict(
            r=r,
            lora_alpha=r * 2,
            lora_dropout=self.LORA_DROPOUT,
            target_modules=tgt,
            modules_to_save=save,
            task_type=self.LORA_TASK_TYPE,
        )

    def lora_config(self, r):
        from peft import LoraConfig
        return LoraConfig(**self.lora_kwargs(r))

    def anchor_stub(self):
        # Text-only chat render (no image) ending at the generation position,
        # so tok(stub + " Yes") extends tok(stub) by the answer token.
        return ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")

    # 448 fixed the first Qwen anchor OOM; stage 2 (TextVQA, high-res task images)
    # still hit 23.38/23.52 GiB with the anchor's extra forwards on top, so cap
    # further: 336px -> ~144 vision tokens/image (~44% less anchor vision memory).
    # Object presence/absence is a coarse signal, so this costs little; LLaVA is
    # unaffected (its anchor_image is a no-op). No Qwen anchor result predates this.
    ANCHOR_MAX_SIDE = 336

    def anchor_image(self, img):
        """Downsize grounding-pair images for the anchor forward: Qwen2.5-VL
        (8.45B) OOMs on a 24GB card when the vision tower processes several
        full-resolution images at once (smoke 2026-09-05). Object presence /
        absence is coarse, so a 448px cap is ample and cuts vision tokens ~9x.
        Only affects the anchor's grounding images, never task training."""
        w, h = img.size
        m = max(w, h)
        if m <= self.ANCHOR_MAX_SIDE:
            return img
        s = self.ANCHOR_MAX_SIDE / m
        return img.resize((max(1, round(w * s)), max(1, round(h * s))))

    def generation_kwargs(self, processor):
        tok = processor.tokenizer
        pad = tok.pad_token_id
        if pad is None:
            pad = tok.eos_token_id
        kw = dict(do_sample=False, pad_token_id=pad)
        imend = tok.convert_tokens_to_ids(self.TURN_EOS)
        if imend is not None and imend >= 0:
            kw["eos_token_id"] = imend  # stop at assistant-turn end
        return kw


_REGISTRY = {
    Llava15Backbone.name: Llava15Backbone,
    Qwen25VLBackbone.name: Qwen25VLBackbone,
}


def available_backbones():
    """Sorted list of registered backbone names (for argparse choices)."""
    return sorted(_REGISTRY)


def get_backbone(name):
    """Return an adapter instance for `name` ("llava15" or "qwen25vl")."""
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown backbone {name!r}; available: {available_backbones()}")
    return _REGISTRY[name]()
