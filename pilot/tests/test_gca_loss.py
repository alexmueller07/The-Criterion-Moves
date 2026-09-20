"""CPU tests for the generative counterfactual token anchor (method/gca_loss.py)
and its train_lora.py / run_arm.py wiring.

Self-contained and CPU-only, like tests/test_crit_preserve.py: no GPU, no model
download, no forward pass, no transformers/peft. The class is exercised
end-to-end by the cluster smoke gate; here we pin

  * slot finding on synthetic captions: earliest CHAIR surface form of the
    present object, plural-tolerant, longest match at a tie, None when the
    object never appears;
  * slot texts: the teacher-forced prefix ends WITHOUT a trailing space, pos_text
    is byte-identical to the collator's text through the object, the absent word
    copies the present word's casing, on both backbones' string conventions;
  * in-context token resolution on a synthetic SentencePiece-like tokenizer,
    including the skip paths (unstable prefix tokenization; present/absent sharing
    their first subword) and the first-subword rule for multi-word objects;
  * the two-term math: ~0 when both margins are met, > 0 otherwise, exact
    softplus values, gradient flows with the right signs on all four logits;
  * train_lora flags default OFF (gca not requested -> byte-identical arms),
    add_gca_term == loss + accum*weight*(decision+cf) with per-term logging;
  * fullstudy/run_arm.py exposes gca / anchorgca + GCA_CAPTIONS and keeps its
    pinned MICRO_BATCH / FAITH_PAIRS lines.

Torch is guarded: on a torch-free (or PIL-free) interpreter the torch tests skip
cleanly (exit 0); the string-level and source-level checks still run.

    python3 pilot/tests/test_gca_loss.py
"""
import contextlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
sys.path.insert(0, PILOT)

_HEAVY = ("transformers", "peft")
_PRELOADED = {m for m in _HEAVY if m in sys.modules}

from model_zoo import get_backbone  # noqa: E402  (torch-free)

try:
    import torch
    import torch.nn.functional as F
    import train_lora
    from method.gca_loss import (
        CAPTION_PROMPT,
        CHAIR_SYN,
        build_slot_rows,
        find_object_slot,
        gca_penalty,
        gca_terms,
        resolve_slot_ids,
        slot_log_odds,
        slot_texts,
    )
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


def _skip(name):
    print(f"SKIP {name}: torch/PIL (gca_loss deps) not installed")


LLAVA = get_backbone("llava15")
QWEN = get_backbone("qwen25vl")


# ---------------------------------------------------------------------------
# Synthetic tokenizer: SentencePiece-like. Every maximal run of
# "optional whitespace + non-whitespace" is one token, so " horse" and "horse"
# are DIFFERENT ids (the leading-space trap), and a prefix's tokens are a prefix
# of any extension that starts at a whitespace boundary.
# ---------------------------------------------------------------------------

class WordTok:
    PIECE = re.compile(r"\s*\S+")

    def __init__(self):
        self.vocab = {}

    def _id(self, piece):
        if piece not in self.vocab:
            self.vocab[piece] = len(self.vocab) + 100
        return self.vocab[piece]

    def pieces(self, text):
        return self.PIECE.findall(text)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self._id(p) for p in self.pieces(text)]}


class MergingTok(WordTok):
    """Merges 'a' + ' <word>' into one token whenever the word follows ' a':
    the prefix '... riding a' tokenizes differently once the object is appended
    (a cross-boundary merge) -> resolve_slot_ids must return None."""

    def pieces(self, text):
        ps = super().pieces(text)
        out, i = [], 0
        while i < len(ps):
            if i + 1 < len(ps) and ps[i].strip() == "a":
                out.append(ps[i] + ps[i + 1])
                i += 2
            else:
                out.append(ps[i])
                i += 1
        return out


# ---------------------------------------------------------------------------
# Slot finding (string level; needs the module -> guarded).
# ---------------------------------------------------------------------------

def test_imports_stay_light():
    if not HAVE_DEPS:
        return _skip("test_imports_stay_light")
    leaked = [m for m in _HEAVY if m in sys.modules and m not in _PRELOADED]
    assert not leaked, (
        f"importing gca_loss/train_lora pulled in heavy modules {leaked}; the CPU "
        "test must run without transformers/peft")
    # The caption prompt is the EOS term's, so both generative terms share the
    # instruction the slot is teacher-forced under.
    from method.policy_anchor import CAPTION_PROMPT as POL_PROMPT
    assert CAPTION_PROMPT == POL_PROMPT == "Describe the image in detail."


def test_find_slot_class_name_and_plural():
    if not HAVE_DEPS:
        return _skip("test_find_slot_class_name_and_plural")
    assert find_object_slot("A man riding a horse on a beach.", "horse") == (15, "horse")
    assert find_object_slot("Two horses graze in a field.", "horse") == (4, "horses")
    assert find_object_slot("Three buses parked at a depot.", "bus") == (6, "buses")
    # Case-insensitive, and the surface keeps the caption's casing.
    assert find_object_slot("Horse and rider in a parade.", "horse") == (0, "Horse")
    # Word boundary: "cat" must not match inside "catalog".
    assert find_object_slot("A catalog on a table.", "cat") is None


def test_find_slot_uses_chair_synonyms_earliest_then_longest():
    if not HAVE_DEPS:
        return _skip("test_find_slot_uses_chair_synonyms_earliest_then_longest")
    # "person" never appears literally in COCO captions; its CHAIR surface forms do.
    assert find_object_slot("A woman walks a dog.", "person") is None  # class-name only
    hit = find_object_slot("A woman walks a dog.", "person", CHAIR_SYN)
    assert hit == (2, "woman"), hit                  # earliest form; not "man" inside "woman"
    hit = find_object_slot("A man and a woman.", "person", CHAIR_SYN)
    assert hit == (2, "man"), hit
    # Multi-word class: literal match wins the tie over its own first word.
    hit = find_object_slot("A red fire hydrant on the curb.", "fire hydrant", CHAIR_SYN)
    assert hit == (6, "fire hydrant"), hit
    # Synonym that is not the class name ("hydrant" alone).
    hit = find_object_slot("A dog next to a hydrant.", "fire hydrant", CHAIR_SYN)
    assert hit == (16, "hydrant"), hit
    # Object absent from the caption -> None (the skip path).
    assert find_object_slot("A bowl of fruit on a table.", "dog", CHAIR_SYN) is None


# ---------------------------------------------------------------------------
# Slot texts on both backbones' string conventions.
# ---------------------------------------------------------------------------

def test_slot_texts_llava_mid_caption():
    if not HAVE_DEPS:
        return _skip("test_slot_texts_llava_mid_caption")
    cap = "A man riding a horse on a beach."
    pre, pos, neg = slot_texts(LLAVA, CAPTION_PROMPT, cap, 15, "horse", "dog")
    prompt = LLAVA.build_prompt(CAPTION_PROMPT)
    assert pre == prompt + " A man riding a", pre
    assert not pre.endswith(" "), "teacher-forced prefix must not end with a space"
    assert pos == prompt + " A man riding a horse", pos
    assert neg == prompt + " A man riding a dog", neg
    # pos_text == the collator's full text truncated right after the object.
    full, _, _ = LLAVA.build_train_text(CAPTION_PROMPT, cap, "</s>")
    assert full.startswith(pos)


def test_slot_texts_caption_initial_object_and_casing():
    if not HAVE_DEPS:
        return _skip("test_slot_texts_caption_initial_object_and_casing")
    cap = "Horses graze in a field."
    pre, pos, neg = slot_texts(LLAVA, CAPTION_PROMPT, cap, 0, "Horses", "cow")
    prompt = LLAVA.build_prompt(CAPTION_PROMPT)
    assert pre == prompt, pre                      # prefix is the bare prompt
    assert pos == prompt + " Horses", pos          # LLaVA sep " " carried as the lead
    assert neg == prompt + " Cow", neg             # casing copied from the surface
    # Qwen: no separator, prefix ends at "assistant\n" (rstrip must not eat the newline).
    pre_q, pos_q, neg_q = slot_texts(QWEN, CAPTION_PROMPT, cap, 0, "Horses", "cow")
    prompt_q = QWEN.build_prompt(CAPTION_PROMPT)
    assert pre_q == prompt_q and prompt_q.endswith("assistant\n"), pre_q
    assert pos_q == prompt_q + "Horses" and neg_q == prompt_q + "Cow", (pos_q, neg_q)


def test_slot_texts_qwen_mid_caption():
    if not HAVE_DEPS:
        return _skip("test_slot_texts_qwen_mid_caption")
    cap = "A man riding a horse."
    pre, pos, neg = slot_texts(QWEN, CAPTION_PROMPT, cap, 15, "horse", "sheep")
    prompt = QWEN.build_prompt(CAPTION_PROMPT)
    assert pre == prompt + "A man riding a", pre
    assert pos == prompt + "A man riding a horse" and neg == prompt + "A man riding a sheep"
    full, _, _ = QWEN.build_train_text(CAPTION_PROMPT, cap, "<ignored>")
    assert full.startswith(pos)


# ---------------------------------------------------------------------------
# In-context token resolution + the join.
# ---------------------------------------------------------------------------

def test_resolve_slot_ids_stable_and_first_subword():
    if not HAVE_DEPS:
        return _skip("test_resolve_slot_ids_stable_and_first_subword")
    tok = WordTok()
    cap = "A man riding a horse on a beach."
    pre, pos, neg = slot_texts(LLAVA, CAPTION_PROMPT, cap, 15, "horse", "dog")
    got = resolve_slot_ids(tok, pre, pos, neg)
    assert got is not None
    pos_id, neg_id, n_pre = got
    assert pos_id == tok._id(" horse") and neg_id == tok._id(" dog"), got
    assert pos_id != tok._id("horse"), "must be the leading-space token, not the bare word"
    assert n_pre == len(tok(pre)["input_ids"])
    # Multi-word present AND absent objects: FIRST subword of each.
    cap2 = "A red fire hydrant on the curb."
    pre2, pos2, neg2 = slot_texts(LLAVA, CAPTION_PROMPT, cap2, 6, "fire hydrant", "stop sign")
    p2, n2, _ = resolve_slot_ids(tok, pre2, pos2, neg2)
    assert p2 == tok._id(" fire") and n2 == tok._id(" stop"), (p2, n2)


def test_resolve_slot_ids_unstable_tokenization_returns_none():
    if not HAVE_DEPS:
        return _skip("test_resolve_slot_ids_unstable_tokenization_returns_none")
    tok = MergingTok()
    cap = "A man riding a horse on a beach."
    pre, pos, neg = slot_texts(LLAVA, CAPTION_PROMPT, cap, 15, "horse", "dog")
    # ' a' + ' horse' merge into one token, so tok(pos)[:len(tok(pre))] != tok(pre).
    assert resolve_slot_ids(tok, pre, pos, neg) is None
    # No extension at all -> None as well.
    assert resolve_slot_ids(WordTok(), pre, pre, pre) is None


def test_build_slot_rows_join_and_skip_reasons():
    if not HAVE_DEPS:
        return _skip("test_build_slot_rows_join_and_skip_reasons")
    tok = WordTok()
    pairs = [
        {"id": "ground_1", "image": "i1.jpg", "masked_image": "m1.jpg",
         "present": "horse", "absent": "dog"},
        {"id": "ground_2", "image": "i2.jpg", "masked_image": "m2.jpg",
         "present": "person", "absent": "bicycle"},          # via synonym "woman"
        {"id": "ground_3", "image": "i3.jpg", "masked_image": "m3.jpg",
         "present": "dog", "absent": "cat"},                 # object not in caption
        {"id": "ground_4", "image": "i4.jpg", "masked_image": "m4.jpg",
         "present": "baseball bat", "absent": "baseball glove"},  # same first subword
        {"id": "ground_5", "image": "i5.jpg", "masked_image": "m5.jpg",
         "present": "cat", "absent": "dog"},                 # no caption row
    ]
    caps = {"ground_1": "A man riding a horse on a beach.",
            "ground_2": "A woman walks along the street.",
            "ground_3": "A bowl of fruit on a table.",
            "ground_4": "A boy swings a baseball bat at a ball.",
            "ground_9": "An unrelated caption with a horse."}   # not a pair id -> ignored
    rows, report = build_slot_rows(pairs, caps, tok, LLAVA, synonyms=CHAIR_SYN)
    assert [r["id"] for r in rows] == ["ground_1", "ground_2"], [r["id"] for r in rows]
    assert report == {"joined": 2, "no_mention": 1, "same_first_token": 1,
                      "no_caption": 1}, report
    r1, r2 = rows
    assert r1["surface"] == "horse" and r1["slot_char"] == 15 and r1["slot_word"] == 4
    assert r1["pos_id"] == tok._id(" horse") and r1["neg_id"] == tok._id(" dog")
    assert r1["pre_text"].endswith("ASSISTANT: A man riding a")
    assert r1["masked_image"] == "m1.jpg" and r1["caption"] == caps["ground_1"]
    assert r2["surface"] == "woman" and r2["slot_word"] == 1
    assert r2["pos_id"] == tok._id(" woman") and r2["neg_id"] == tok._id(" bicycle")
    # Class-name-only matching (synonyms={}): the "person" pair is now a no_mention.
    rows0, report0 = build_slot_rows(pairs, caps, tok, LLAVA, synonyms={})
    assert [r["id"] for r in rows0] == ["ground_1"] and report0["no_mention"] == 2, report0
    # Unstable tokenization is its own skip reason.
    rows_m, report_m = build_slot_rows(pairs[:1], caps, MergingTok(), LLAVA, synonyms=CHAIR_SYN)
    assert rows_m == [] and report_m == {"unstable_tokenization": 1}, report_m


# ---------------------------------------------------------------------------
# The two-term math.
# ---------------------------------------------------------------------------

def test_terms_near_zero_when_margins_met():
    if not HAVE_DEPS:
        return _skip("test_terms_near_zero_when_margins_met")
    # l(I) = 30 >> m, l(I) - l(I\o+) = 30 - (-5) = 35 >> m -> both softplus(-28..-33) ~ 0.
    ell_real = torch.tensor([30.0, 25.0])
    ell_masked = torch.tensor([-5.0, 0.0])
    d, c = gca_terms(ell_real, ell_masked, margin=2.0)
    assert d.shape == c.shape == (2,)
    assert float(d.max()) < 1e-9 and float(c.max()) < 1e-9, (d, c)
    assert float(gca_penalty(ell_real, ell_masked, 2.0)) < 1e-8


def test_terms_positive_and_exact_when_violated():
    if not HAVE_DEPS:
        return _skip("test_terms_positive_and_exact_when_violated")
    # Decision violated (absent beats present: l(I) = -1), cf satisfied.
    d, c = gca_terms(torch.tensor([-1.0]), torch.tensor([-40.0]), margin=2.0)
    assert abs(float(d) - float(F.softplus(torch.tensor(3.0)))) < 1e-6, float(d)
    assert float(c) < 1e-9
    # Decision satisfied, cf violated: masking changes nothing (l equal on both
    # images -> the slot is decided by the language prior, not the image).
    d, c = gca_terms(torch.tensor([30.0]), torch.tensor([30.0]), margin=2.0)
    assert float(d) < 1e-9
    assert abs(float(c) - float(F.softplus(torch.tensor(2.0)))) < 1e-6, float(c)
    # Masked image RAISES the present object's odds: cf penalised harder still.
    _, c_worse = gca_terms(torch.tensor([30.0]), torch.tensor([33.0]), margin=2.0)
    assert float(c_worse) > float(c)
    # Bad margin rejected.
    try:
        gca_terms(torch.tensor([0.0]), torch.tensor([0.0]), margin=-0.5)
    except AssertionError:
        pass
    else:
        raise AssertionError("negative margin must be rejected")


def test_gradient_flows_with_correct_signs():
    if not HAVE_DEPS:
        return _skip("test_gradient_flows_with_correct_signs")
    # Four logits, both terms active (ell values near the margin).
    zp_r = torch.tensor([1.0], requires_grad=True)   # z(o+ | I)
    zn_r = torch.tensor([0.5], requires_grad=True)   # z(o- | I)
    zp_m = torch.tensor([0.8], requires_grad=True)   # z(o+ | I \ o+)
    zn_m = torch.tensor([0.4], requires_grad=True)   # z(o- | I \ o+)
    ell_real = slot_log_odds(zp_r, zn_r)
    ell_masked = slot_log_odds(zp_m, zn_m)
    assert ell_real.dtype == torch.float32
    val = gca_penalty(ell_real, ell_masked, margin=2.0)
    val.backward()
    # Descent raises the present object's logit on the real image (both terms
    # push the same way) and lowers the absent object's logit there.
    assert float(zp_r.grad) < 0 and float(zn_r.grad) > 0, (zp_r.grad, zn_r.grad)
    # On the masked image the cf term wants the present object LOWER and the
    # absent object HIGHER (the log-odds must drop when o+ is masked out).
    assert float(zp_m.grad) > 0 and float(zn_m.grad) < 0, (zp_m.grad, zn_m.grad)
    # Real-image gradient carries both terms; masked carries only the cf term.
    assert abs(float(zp_r.grad)) > abs(float(zp_m.grad))
    # bf16 inputs are lifted to fp32 before the margin math.
    ell = slot_log_odds(torch.tensor([2.0], dtype=torch.bfloat16),
                        torch.tensor([0.5], dtype=torch.bfloat16))
    assert ell.dtype == torch.float32 and abs(float(ell) - 1.5) < 1e-6


# ---------------------------------------------------------------------------
# train_lora wiring (flags + composition path with a mocked anchor).
# ---------------------------------------------------------------------------

REQUIRED = ["--data", "x.jsonl", "--data_root", "/r", "--out", "/o"]


def test_flags_default_off_and_parse():
    if not HAVE_DEPS:
        return _skip("test_flags_default_off_and_parse")
    args = train_lora.build_parser().parse_args(REQUIRED)
    assert args.gca_pairs is None and args.gca_captions is None, vars(args)
    assert args.gca_weight == 0.0 and args.gca_k == 4 and args.gca_margin == 2.0, vars(args)
    assert train_lora.gca_requested(args) is False
    # Pre-existing flag surface untouched.
    assert args.faith_pairs is None and args.faith_weight == 0.1 and args.faith_k == 4
    assert args.crit_probe is None and args.crit_weight == 0.0
    assert args.pol_weight_abstain == 0.0 and args.pol_weight_eos == 0.0
    a = train_lora.build_parser().parse_args(REQUIRED + [
        "--gca_pairs", "p.jsonl", "--gca_captions", "c.jsonl", "--gca_weight", "0.1",
        "--gca_k", "2", "--gca_margin", "1.5"])
    assert a.gca_pairs == "p.jsonl" and a.gca_captions == "c.jsonl"
    assert a.gca_weight == 0.1 and a.gca_k == 2 and a.gca_margin == 1.5
    assert train_lora.gca_requested(a) is True
    try:
        with open(os.devnull, "w") as sink, contextlib.redirect_stderr(sink):
            train_lora.build_parser().parse_args(REQUIRED + ["--gca_k", "two"])
    except SystemExit:
        pass
    else:
        raise AssertionError("--gca_k two must be rejected")


class MockGCA:
    def __init__(self, decision, cf):
        self.comps = {"decision": decision, "cf": cf,
                      "ell_real": torch.tensor(3.5), "ell_gap": torch.tensor(1.25)}
        self.calls = 0

    def loss_components(self, model):
        self.calls += 1
        return self.comps


def test_add_gca_term_composes_and_logs():
    if not HAVE_DEPS:
        return _skip("test_add_gca_term_composes_and_logs")
    dec = torch.tensor(0.7, requires_grad=True)
    cf = torch.tensor(0.25, requires_grad=True)
    gca = MockGCA(dec, cf)
    base = torch.tensor(2.0, requires_grad=True)
    hist = []
    w, accum = 0.1, 4
    out = train_lora.add_gca_term(base, gca, model=None, weight=w, accum=accum, history=hist)
    assert abs(float(out.detach()) - (2.0 + accum * w * (0.7 + 0.25))) < 1e-6, float(out.detach())
    assert gca.calls == 1
    assert len(hist) == 1 and set(hist[0]) == {"decision", "cf", "ell_real", "ell_gap"}, hist
    assert abs(hist[0]["decision"] - 0.7) < 1e-6 and abs(hist[0]["cf"] - 0.25) < 1e-6  # raw
    assert abs(hist[0]["ell_real"] - 3.5) < 1e-6 and abs(hist[0]["ell_gap"] - 1.25) < 1e-6
    out.backward()
    assert abs(float(dec.grad) - accum * w) < 1e-6 and abs(float(cf.grad) - accum * w) < 1e-6
    assert float(base.grad) == 1.0
    # Weight 0 (monitor only): value unchanged, still logged, no gradient.
    dec0 = torch.tensor(0.7, requires_grad=True)
    cf0 = torch.tensor(0.25, requires_grad=True)
    hist0 = []
    out0 = train_lora.add_gca_term(torch.tensor(1.0), MockGCA(dec0, cf0), None, 0.0, 2, hist0)
    assert float(out0.detach()) == 1.0 and len(hist0) == 1
    out0.backward()
    assert float(dec0.grad) == 0.0 and float(cf0.grad) == 0.0


# ---------------------------------------------------------------------------
# Arm driver surface (module-level script: checked at source level; torch-free).
# ---------------------------------------------------------------------------

def test_run_arm_exposes_gca_arms_and_keeps_pinned_lines():
    src = open(os.path.join(PILOT, "..", "fullstudy", "run_arm.py")).read()
    for needle in ('args.arm == "gca"', 'args.arm == "anchorgca"',
                   'os.environ.get("GCA_CAPTIONS"', 'os.environ.get("GCA_WEIGHT", "0.1")',
                   '"--gca_pairs"', '"--gca_captions"', '"--gca_weight"', '"--gca_k"',
                   '"--gca_margin"', 'extra = anchor_extra + gca_extra',
                   'os.path.join(D, "grounding", "caption_anchor.jsonl")'):
        assert needle in src, f"run_arm.py missing {needle!r}"
    # Lines the integration must preserve verbatim.
    assert 'MICRO_BATCH = 1 if args.backbone == "qwen25vl" else 2' in src
    assert ('_FP = os.environ.get("FAITH_PAIRS", os.path.join(D, "grounding", '
            '"grounding_pairs.jsonl"))') in src
    assert 'args.arm == "critp"' in src and 'args.arm == "policy"' in src


TESTS = [
    test_imports_stay_light,
    test_find_slot_class_name_and_plural,
    test_find_slot_uses_chair_synonyms_earliest_then_longest,
    test_slot_texts_llava_mid_caption,
    test_slot_texts_caption_initial_object_and_casing,
    test_slot_texts_qwen_mid_caption,
    test_resolve_slot_ids_stable_and_first_subword,
    test_resolve_slot_ids_unstable_tokenization_returns_none,
    test_build_slot_rows_join_and_skip_reasons,
    test_terms_near_zero_when_margins_met,
    test_terms_positive_and_exact_when_violated,
    test_gradient_flows_with_correct_signs,
    test_flags_default_off_and_parse,
    test_add_gca_term_composes_and_logs,
    test_run_arm_exposes_gca_arms_and_keeps_pinned_lines,
]


def main():
    if not HAVE_DEPS:
        print("SKIP: torch/PIL not installed; gca-loss tests need them "
              "(the arm-driver source check still runs).")
        test_run_arm_exposes_gca_arms_and_keeps_pinned_lines()
        return
    for t in TESTS:
        t()
    print(f"GCA-LOSS OK: {len(TESTS)} checks passed "
          "(slot finding via CHAIR surface forms, plural/casing/word-boundary; slot "
          "texts on llava15 + qwen25vl with no trailing space; in-context first-subword "
          "resolution with unstable/same-token/no-mention/no-caption skips; two-term "
          "math ~0 when met / exact softplus when violated / signed gradients on all "
          "four logits; flags default off; add_gca_term composition + logging; "
          "run_arm arms + pinned lines).")


if __name__ == "__main__":
    main()
