"""Byte-level parity test: model_zoo "llava15" adapter vs the strings/configs
that were hardcoded in train_lora.py / eval_gen.py / method/faith_loss.py
before the backbone refactor (2026-09-03).

Pure string/config level — importing model_zoo must NOT pull in torch,
transformers, or peft. Python-3.9 compatible; no pytest required:

    python3 pilot/tests/test_backbone_parity.py

(also collectable by pytest — every check is a test_* function).

The OLD_* literals below are copied verbatim from the pre-refactor sources and
must never be "updated to match" the adapter; if this test fails, the adapter
changed LLaVA behavior.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_HEAVY = ("torch", "transformers", "peft")
_PRELOADED = {m for m in _HEAVY if m in sys.modules}

from model_zoo import available_backbones, get_backbone  # noqa: E402


# ---------------------------------------------------------------------------
# Pre-refactor literals (verbatim copies — do not edit).
# ---------------------------------------------------------------------------

# eval_gen.py:  f"USER: <image>\n{r['prompt']} ASSISTANT:"
def old_eval_prompt(prompt):
    return f"USER: <image>\n{prompt} ASSISTANT:"


# train_lora.py Collator:
#   prefix = f"USER: <image>\n{ex['prompt']} ASSISTANT:"
#   texts.append(prefix + " " + ex["target"] + self.p.tokenizer.eos_token)
#   targets.append(" " + ex["target"] + self.p.tokenizer.eos_token)
def old_collator_strings(prompt, target, eos):
    prefix = f"USER: <image>\n{prompt} ASSISTANT:"
    full = prefix + " " + target + eos
    target_fragment = " " + target + eos
    return full, prefix, target_fragment


# method/faith_loss.py:
OLD_FAITH_PROMPT = "USER: <image>\nIs there a {obj} in the image? ASSISTANT:"
OLD_ANCHOR_STUB = "USER: hi ASSISTANT:"

# train_lora.py LoraConfig literals:
OLD_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
OLD_LORA_TARGET_MODULES = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
OLD_LORA_MODULES_TO_SAVE = ["multi_modal_projector"]
OLD_LORA_DROPOUT = 0.05
OLD_LORA_TASK_TYPE = "CAUSAL_LM"


# ---------------------------------------------------------------------------
# Sample prompts/targets, including edge cases.
# ---------------------------------------------------------------------------

PROMPTS = [
    "Describe the image.",
    "What is unusual about this scene?",          # trailing punctuation
    "Is there a dog in the image?",
    "A",                                          # minimal
    "Café — naïve résumé; 中文提示 🍜?",            # unicode + emoji
    "line one\nline two",                         # embedded newline
    "prompt that itself contains ASSISTANT: and <image> markers",
    "trailing space ",                            # trailing whitespace kept
]

TARGETS = [
    "",                                           # empty-ish target
    " ",                                          # whitespace-only target
    "yes",
    "Yes.",                                       # trailing punctuation
    "Two cats and a naïve café ☕️ 🐈‍⬛",            # unicode incl. ZWJ emoji
    "ends with newline\n",
    "  leading spaces preserved",
]

EOS_TOKENS = ["</s>"]  # llava-1.5 / llama tokenizer eos


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------

def test_import_is_torch_free():
    loaded = [m for m in _HEAVY if m in sys.modules and m not in _PRELOADED]
    assert not loaded, (
        f"importing model_zoo pulled in heavy modules {loaded}; the string-"
        "level parity test must run without torch/transformers/peft")


def test_registry():
    assert available_backbones() == ["llava15", "qwen25vl"]
    assert get_backbone("llava15").name == "llava15"
    assert get_backbone("qwen25vl").name == "qwen25vl"
    try:
        get_backbone("nope")
    except KeyError:
        pass
    else:
        raise AssertionError("get_backbone('nope') did not raise KeyError")


def test_model_id():
    assert get_backbone("llava15").model_id == OLD_MODEL_ID


def test_build_prompt_parity():
    bb = get_backbone("llava15")
    for p in PROMPTS:
        got = bb.build_prompt(p)
        want = old_eval_prompt(p)
        assert got == want, f"build_prompt mismatch for {p!r}:\n{got!r}\n{want!r}"
        assert not got.endswith(" "), (
            f"prefix must not end with a space (smoke lesson 2026-08-31): {got!r}")


def test_build_train_text_parity():
    bb = get_backbone("llava15")
    n = 0
    for eos in EOS_TOKENS:
        for p in PROMPTS:
            for t in TARGETS:
                got = bb.build_train_text(p, t, eos)
                want = old_collator_strings(p, t, eos)
                assert got == want, (
                    f"build_train_text mismatch for prompt={p!r} target={t!r}:"
                    f"\n got={got!r}\nwant={want!r}")
                full, prefix, target_fragment = got
                assert full == prefix + target_fragment
                assert target_fragment.startswith(" "), (
                    "target fragment must carry the leading space")
                assert target_fragment.endswith(eos)
                n += 1
    assert n == len(EOS_TOKENS) * len(PROMPTS) * len(TARGETS)
    return n


def test_anchor_stub_parity():
    assert get_backbone("llava15").anchor_stub() == OLD_ANCHOR_STUB


def test_faith_prompt_parity():
    # faith_loss.py now builds bb.build_prompt(QUESTION.format(obj=obj));
    # must be byte-equal to the old hardcoded PROMPT.format(obj=obj).
    bb = get_backbone("llava15")
    for obj in ["cat", "fire hydrant", "naïve café", "teddy bear"]:
        got = bb.build_prompt(f"Is there a {obj} in the image?")
        want = OLD_FAITH_PROMPT.format(obj=obj)
        assert got == want, f"faith prompt mismatch for {obj!r}:\n{got!r}\n{want!r}"


def test_lora_config_field_parity():
    bb = get_backbone("llava15")
    for r in (64, 32, 8):
        kw = bb.lora_kwargs(r)
        assert kw["r"] == r
        assert kw["lora_alpha"] == r * 2, kw
        assert kw["lora_dropout"] == OLD_LORA_DROPOUT, kw
        assert kw["target_modules"] == OLD_LORA_TARGET_MODULES, kw
        assert kw["modules_to_save"] == OLD_LORA_MODULES_TO_SAVE, kw
        assert kw["task_type"] == OLD_LORA_TASK_TYPE, kw
        assert set(kw) == {"r", "lora_alpha", "lora_dropout", "target_modules",
                           "modules_to_save", "task_type"}, (
            f"unexpected/missing LoraConfig fields: {sorted(kw)}")


def test_generation_kwargs_parity():
    # eval_gen.py generate() call passed exactly: do_sample=False,
    # pad_token_id=processor.tokenizer.pad_token_id
    class _Tok(object):
        pad_token_id = 32001

    class _Proc(object):
        tokenizer = _Tok()

    kw = get_backbone("llava15").generation_kwargs(_Proc())
    assert kw == {"do_sample": False, "pad_token_id": 32001}, kw


def test_qwen25vl_strings():
    """Qwen adapter implemented 2026-09-03 from the dependency probe. Only the
    torch-free string/config methods are checked here; the runtime path
    (load/freeze/forward) is validated by the cluster smoke gate, since its
    conventions (eos terminator, leading-space rule, merger freeze) could not
    be confirmed on the probe's Blackwell card and are asserted by the
    collator boundary check + pre-train forward-loss check."""
    import re
    qb = get_backbone("qwen25vl")
    p = qb.build_prompt("Is there a dog in the image?")
    assert p.endswith("<|im_start|>assistant\n") and "<|image_pad|>" in p
    f, pre, t = qb.build_train_text("Q", "A", "<ignored-eos>")
    assert f == pre + t and t == "A<|im_end|>" and pre.endswith("assistant\n")
    assert qb.anchor_stub().endswith("<|im_start|>assistant\n")
    lk = qb.lora_kwargs(64)
    assert lk["r"] == 64 and lk["lora_alpha"] == 128 and lk["modules_to_save"] == []
    rx = lk["target_modules"]
    assert re.fullmatch(rx, "model.layers.0.self_attn.q_proj")
    assert re.fullmatch(rx, "base_model.model.model.layers.12.mlp.gate_proj")
    assert not re.fullmatch(rx, "visual.blocks.0.attn.qkv")
    assert not re.fullmatch(rx, "visual.blocks.0.mlp.gate_proj")


def main():
    test_import_is_torch_free()
    test_registry()
    test_model_id()
    test_build_prompt_parity()
    n_train = test_build_train_text_parity()
    test_anchor_stub_parity()
    test_faith_prompt_parity()
    test_lora_config_field_parity()
    test_generation_kwargs_parity()
    test_qwen25vl_strings()
    print("PARITY OK: llava15 adapter byte-equals pre-refactor literals "
          f"({len(PROMPTS)} prompts, {n_train} train-text combos, "
          "anchor stub, faith prompt, LoRA fields, generation kwargs); "
          "qwen25vl string/config methods verified (runtime via cluster smoke).")


if __name__ == "__main__":
    main()
