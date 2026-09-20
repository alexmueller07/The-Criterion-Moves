"""CPU tests for the faithfulness anchor's counterfactual-term FORM
(method/faith_loss.py `cf_form`, R1 of design_notes/method_ideas_lit1.md sec.2.0)
and its train_lora.py / run_arm.py wiring.

R1 is an ABLATION, not a repair: the v2 anchor's criterion freeze is a
replicated result (3/3 seeds), so the default path must stay exactly what it
was. The refinement is IC-VCO's (arXiv 2605.31312v1) partition-function
critique applied to term 3: comparing the RAW logit z_yes across the real and
GT-masked images compares two different softmax contexts, each with its own
log-partition; the within-image log-odds g = z_yes - z_no is
normalization-free.

Self-contained and CPU-only, like tests/test_crit_preserve.py and
tests/test_gca_loss.py: no GPU, no model download, no forward pass. We pin

  * (a) cf_form="logit" is BITWISE identical to the pre-change implementation
    (an inlined verbatim copy of the three replaced lines) AND matches
    independently pinned literal values computed with stdlib softplus -- so the
    two copies cannot drift together;
  * (b) cf_form="logodds" computes exactly sp(m - [g(I) - g(I\\o+)]) with
    g = z_yes - z_no, pinned the same two ways, and is the ONLY term that
    changes (terms 1-2 are bitwise equal across the forms);
  * the property that motivates the refinement: adding a constant to the whole
    masked-image logit vector leaves "logodds" EXACTLY unchanged and moves
    "logit" one-for-one;
  * (c) each form is 0 (to softplus's exponential tail) when its own margins
    are met and strictly positive when they are not, including a pair of cases
    where the two forms DISAGREE about satisfaction;
  * (d) gradients flow to all FOUR logits (z_yes/z_no on both images) under
    "logodds", with the right signs, while "logit" leaves z_no(masked) with
    exactly zero gradient;
  * FaithfulnessAnchor's kwarg: default "logit", stored, bad values rejected;
  * train_lora.py --faith_cf_form defaults to "logit", rejects other values,
    and does not disturb the other faith flags;
  * fullstudy/run_arm.py exposes arm "anchorlo" + FAITH_CF_FORM, leaves the
    "anchor" arm's command line free of --faith_cf_form, and keeps its pinned
    MICRO_BATCH / FAITH_PAIRS / CRIT_PROBE / critp / cnp / gca / policy lines.

Torch is guarded: on a torch-free (or PIL-free) interpreter the torch tests
skip cleanly (exit 0); the run_arm source check still runs.

    python3 pilot/tests/test_faith_cf_form.py
"""
import contextlib
import json
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
sys.path.insert(0, PILOT)

_HEAVY = ("transformers", "peft")
_PRELOADED = {m for m in _HEAVY if m in sys.modules}

try:
    import torch
    import torch.nn.functional as F
    import train_lora
    from method.faith_loss import CF_FORMS, FaithfulnessAnchor, faith_margin_terms
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


def _skip(name):
    print(f"SKIP {name}: torch/PIL (faith_loss deps) not installed")


# ---------------------------------------------------------------------------
# Fixed synthetic logits. Columns follow loss()'s row order:
#   0 = (real image, present object)
#   1 = (GT-masked image, present object)
#   2 = (real image, absent object)
# Chosen so all three terms are active (none saturated to 0) and so the two
# cf forms give visibly different numbers.
# ---------------------------------------------------------------------------
Z_YES = [[3.0, 0.5, -1.0],
         [1.0, 0.0, 0.5],
         [-0.5, -0.25, 2.0]]
Z_NO = [[-1.0, 1.5, 2.0],
        [0.5, -0.5, 1.0],
        [0.25, 0.75, 0.5]]
MARGIN = 2.0

# Pinned expectations (see _sp below for the stdlib recomputation; the literals
# are the fp32 values the pre-change implementation produced on these inputs).
WANT_PRESENT = [0.12692801654338837, 1.7014132738113403, 2.811967611312866]
WANT_ABSENT = [0.3132616877555847, 1.7014132738113403, 3.529750347137451]
WANT_CF_LOGIT = [0.4740769863128662, 1.31326162815094, 2.3502066135406494]
WANT_CF_LOGODDS = [0.04858734831213951, 2.1269280910491943, 1.910224199295044]
WANT_LOSS_LOGIT = 4.774093151092529
WANT_LOSS_LOGODDS = 4.756824493408203


def _sp(x):
    """softplus in pure stdlib, numerically stable. Deliberately NOT
    F.softplus: this is the independent recomputation of the pins."""
    return math.log1p(math.exp(-abs(x))) + max(x, 0.0)


def _tensors():
    return torch.tensor(Z_YES), torch.tensor(Z_NO)


def _pre_change_reference(z_yes, z_no, margin):
    """VERBATIM copy of the three lines faith_loss.loss() used before the
    cf_form flag existed (repo state 2026-09-08, pre-R1). The default path must
    stay bitwise equal to this forever."""
    present_decision = F.softplus(margin - (z_yes[:, 0] - z_no[:, 0]))
    absent_decision = F.softplus(margin - (z_no[:, 2] - z_yes[:, 2]))
    cf_sensitivity = F.softplus(margin - (z_yes[:, 0] - z_yes[:, 1]))
    return present_decision, absent_decision, cf_sensitivity


# ---------------------------------------------------------------------------
# (a) the default path is unchanged
# ---------------------------------------------------------------------------

def test_imports_stay_light():
    if not HAVE_DEPS:
        return _skip("test_imports_stay_light")
    leaked = [m for m in _HEAVY if m in sys.modules and m not in _PRELOADED]
    assert not leaked, (
        f"importing faith_loss pulled in heavy modules {leaked}; the CPU "
        "penalty-math test must run without transformers/peft")
    assert CF_FORMS == ("logit", "logodds"), CF_FORMS


def test_logit_form_bitwise_identical_to_pre_change():
    if not HAVE_DEPS:
        return _skip("test_logit_form_bitwise_identical_to_pre_change")
    zy, zn = _tensors()
    got = faith_margin_terms(zy, zn, MARGIN)          # default cf_form
    got_explicit = faith_margin_terms(zy, zn, MARGIN, cf_form="logit")
    want = _pre_change_reference(zy, zn, MARGIN)
    for name, g, ge, w in zip(("present", "absent", "cf"), got, got_explicit, want):
        assert torch.equal(g, w), f"{name}: default cf_form drifted: {g} vs {w}"
        assert torch.equal(ge, w), f"{name}: cf_form='logit' drifted: {ge} vs {w}"
    # Same reduction loss() applies, bitwise.
    assert torch.equal(sum(got).mean(), sum(want).mean())
    # ... on several margins, not just the operating point.
    for m in (0.0, 0.5, 2.0, 5.0):
        for a, b in zip(faith_margin_terms(zy, zn, m), _pre_change_reference(zy, zn, m)):
            assert torch.equal(a, b), m


def test_logit_form_matches_pinned_literals_and_stdlib():
    if not HAVE_DEPS:
        return _skip("test_logit_form_matches_pinned_literals_and_stdlib")
    zy, zn = _tensors()
    pres, absent, cf = faith_margin_terms(zy, zn, MARGIN, cf_form="logit")
    for i in range(3):
        # stdlib recomputation from the raw inputs (independent of torch).
        want_p = _sp(MARGIN - (Z_YES[i][0] - Z_NO[i][0]))
        want_a = _sp(MARGIN - (Z_NO[i][2] - Z_YES[i][2]))
        want_c = _sp(MARGIN - (Z_YES[i][0] - Z_YES[i][1]))
        assert abs(float(pres[i]) - want_p) < 1e-6, (i, float(pres[i]), want_p)
        assert abs(float(absent[i]) - want_a) < 1e-6, (i, float(absent[i]), want_a)
        assert abs(float(cf[i]) - want_c) < 1e-6, (i, float(cf[i]), want_c)
        # frozen literals (the regression pin proper)
        assert abs(float(pres[i]) - WANT_PRESENT[i]) < 1e-6, (i, float(pres[i]))
        assert abs(float(absent[i]) - WANT_ABSENT[i]) < 1e-6, (i, float(absent[i]))
        assert abs(float(cf[i]) - WANT_CF_LOGIT[i]) < 1e-6, (i, float(cf[i]))
    total = float((pres + absent + cf).mean())
    assert abs(total - WANT_LOSS_LOGIT) < 1e-6, total


# ---------------------------------------------------------------------------
# (b) the logodds form computes the stated quantity
# ---------------------------------------------------------------------------

def test_logodds_form_computes_stated_quantity():
    if not HAVE_DEPS:
        return _skip("test_logodds_form_computes_stated_quantity")
    zy, zn = _tensors()
    pres, absent, cf = faith_margin_terms(zy, zn, MARGIN, cf_form="logodds")
    for i in range(3):
        g_real = Z_YES[i][0] - Z_NO[i][0]
        g_masked = Z_YES[i][1] - Z_NO[i][1]
        want = _sp(MARGIN - (g_real - g_masked))
        assert abs(float(cf[i]) - want) < 1e-6, (i, float(cf[i]), want)
        assert abs(float(cf[i]) - WANT_CF_LOGODDS[i]) < 1e-6, (i, float(cf[i]))
    total = float((pres + absent + cf).mean())
    assert abs(total - WANT_LOSS_LOGODDS) < 1e-6, total
    # Written the other way round: sp(m - (g_real - g_masked)) built from the
    # decision statistic itself (crit_preserve.decision_stat's quantity).
    g = zy - zn
    want_cf = F.softplus(MARGIN - (g[:, 0] - g[:, 1]))
    assert torch.equal(cf, want_cf), (cf, want_cf)


def test_decision_terms_are_identical_across_forms():
    if not HAVE_DEPS:
        return _skip("test_decision_terms_are_identical_across_forms")
    # The symmetric-margin design (terms 1-2) is load-bearing: its
    # criterion-freeze property is derived, and R1 must not touch it.
    zy, zn = _tensors()
    p1, a1, c1 = faith_margin_terms(zy, zn, MARGIN, cf_form="logit")
    p2, a2, c2 = faith_margin_terms(zy, zn, MARGIN, cf_form="logodds")
    assert torch.equal(p1, p2) and torch.equal(a1, a2), (p1, p2, a1, a2)
    assert not torch.equal(c1, c2), "the two cf forms must differ on these inputs"


def test_logodds_is_normalization_free_logit_is_not():
    if not HAVE_DEPS:
        return _skip("test_logodds_is_normalization_free_logit_is_not")
    # THE property R1 buys. Shift the masked image's WHOLE logit vector by a
    # constant (a different log-partition, no change in that image's decision):
    # the log-odds gap is invariant; the raw-z_yes gap moves one-for-one.
    zy, zn = _tensors()
    for shift in (0.75, -1.5, 4.0):
        zy_s = zy.clone()
        zn_s = zn.clone()
        zy_s[:, 1] += shift
        zn_s[:, 1] += shift
        _, _, cf_lo_base = faith_margin_terms(zy, zn, MARGIN, cf_form="logodds")
        _, _, cf_lo_shift = faith_margin_terms(zy_s, zn_s, MARGIN, cf_form="logodds")
        assert torch.allclose(cf_lo_base, cf_lo_shift, atol=1e-6), (shift, cf_lo_base, cf_lo_shift)
        _, _, cf_lg_base = faith_margin_terms(zy, zn, MARGIN, cf_form="logit")
        _, _, cf_lg_shift = faith_margin_terms(zy_s, zn_s, MARGIN, cf_form="logit")
        assert not torch.allclose(cf_lg_base, cf_lg_shift, atol=1e-3), shift
        # exactly the shifted argument, item by item
        for i in range(3):
            want = _sp(MARGIN - (Z_YES[i][0] - (Z_YES[i][1] + shift)))
            assert abs(float(cf_lg_shift[i]) - want) < 1e-5, (shift, i)


# ---------------------------------------------------------------------------
# (c) zero when the margins are met, positive otherwise
# ---------------------------------------------------------------------------

def test_both_forms_zero_when_margins_met_positive_otherwise():
    if not HAVE_DEPS:
        return _skip("test_both_forms_zero_when_margins_met_positive_otherwise")
    # Grounded model, huge margins on every term and under BOTH cf forms:
    # real present is confidently yes, masked present is confidently no, real
    # absent is confidently no.
    zy = torch.tensor([[30.0, -30.0, -30.0]])
    zn = torch.tensor([[-30.0, 30.0, 30.0]])
    for form in CF_FORMS:
        p, a, c = faith_margin_terms(zy, zn, MARGIN, cf_form=form)
        assert float(p.max()) < 1e-9 and float(a.max()) < 1e-9, (form, p, a)
        assert float(c.max()) < 1e-9, (form, c)
        assert float((p + a + c).mean()) < 1e-8, form
    # Every term violated (all three gaps below the margin) -> all positive.
    zy_bad = torch.tensor([[0.0, 0.0, 0.0]])
    zn_bad = torch.tensor([[0.0, 0.0, 0.0]])
    for form in CF_FORMS:
        p, a, c = faith_margin_terms(zy_bad, zn_bad, MARGIN, cf_form=form)
        # all gaps are exactly 0 -> sp(m) each
        want = _sp(MARGIN)
        for t in (p, a, c):
            assert abs(float(t) - want) < 1e-6, (form, float(t), want)
        assert float((p + a + c).mean()) > 0.0

    # The forms genuinely disagree about satisfaction, in both directions.
    # (i) logit satisfied, logodds violated: masking drops z_yes by 30, but it
    #     drops z_no by 30 as well -- a pure context shift. The DECISION on the
    #     masked image is unchanged (g = 10 on both images), so the raw-logit
    #     term is silent while the log-odds term charges in full.
    zy = torch.tensor([[10.0, -20.0, -10.0]])
    zn = torch.tensor([[0.0, -30.0, 10.0]])
    _, _, cf_lg = faith_margin_terms(zy, zn, MARGIN, cf_form="logit")
    _, _, cf_lo = faith_margin_terms(zy, zn, MARGIN, cf_form="logodds")
    assert float(cf_lg) < 1e-9, float(cf_lg)                 # gap 30 >> m: quiet
    assert abs(float(cf_lo) - _sp(MARGIN)) < 1e-6, float(cf_lo)  # gap 0: full charge
    # (ii) logodds satisfied, logit violated: z_yes barely moves under masking
    #      but z_no rises sharply, so the decision does collapse.
    zy = torch.tensor([[20.0, 19.5, -10.0]])
    zn = torch.tensor([[0.0, 19.0, 10.0]])
    _, _, cf_lg = faith_margin_terms(zy, zn, MARGIN, cf_form="logit")
    _, _, cf_lo = faith_margin_terms(zy, zn, MARGIN, cf_form="logodds")
    assert abs(float(cf_lg) - _sp(MARGIN - 0.5)) < 1e-6, float(cf_lg)
    assert float(cf_lo) < 1e-6, float(cf_lo)                 # gap 20 - 0.5 = 19.5


def test_bad_cf_form_and_negative_margin_rejected():
    if not HAVE_DEPS:
        return _skip("test_bad_cf_form_and_negative_margin_rejected")
    zy, zn = _tensors()
    for bad in ("logits", "log_odds", "LOGODDS", "", None):
        try:
            faith_margin_terms(zy, zn, MARGIN, cf_form=bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"cf_form={bad!r} must be rejected")
    try:
        faith_margin_terms(zy, zn, -0.5)
    except AssertionError:
        pass
    else:
        raise AssertionError("negative margin must be rejected")


# ---------------------------------------------------------------------------
# (d) gradients
# ---------------------------------------------------------------------------

def test_gradients_flow_to_all_four_logits_under_logodds():
    if not HAVE_DEPS:
        return _skip("test_gradients_flow_to_all_four_logits_under_logodds")
    # One pair, cf term active (gap below the margin) on both forms.
    zy = torch.tensor([[1.0, 0.5, -1.0]], requires_grad=True)
    zn = torch.tensor([[0.5, 0.25, 1.0]], requires_grad=True)
    p, a, c = faith_margin_terms(zy, zn, MARGIN, cf_form="logodds")
    (p + a + c).mean().backward()
    gy, gn = zy.grad, zn.grad
    # All FOUR logits that enter the two images' decisions get gradient.
    for name, val in (("z_yes(real)", gy[0, 0]), ("z_no(real)", gn[0, 0]),
                      ("z_yes(masked)", gy[0, 1]), ("z_no(masked)", gn[0, 1])):
        assert abs(float(val)) > 1e-6, f"{name} got no gradient: {val}"
    # Signs: descent raises z_yes and lowers z_no on the real image (terms 1
    # and 3 push the same way there), and does the opposite on the masked image
    # (its log-odds must FALL when the present object is masked out).
    assert float(gy[0, 0]) < 0 and float(gn[0, 0]) > 0, (gy, gn)
    assert float(gy[0, 1]) > 0 and float(gn[0, 1]) < 0, (gy, gn)
    # Absent column: push toward "No".
    assert float(gy[0, 2]) > 0 and float(gn[0, 2]) < 0, (gy, gn)
    # The cf term's pull on the masked pair is equal and opposite between the
    # two tokens (it is a pure log-odds difference).
    assert abs(float(gy[0, 1]) + float(gn[0, 1])) < 1e-6, (gy, gn)

    # Contrast: under "logit" the masked image's z_no is not in the graph.
    zy2 = torch.tensor([[1.0, 0.5, -1.0]], requires_grad=True)
    zn2 = torch.tensor([[0.5, 0.25, 1.0]], requires_grad=True)
    p2, a2, c2 = faith_margin_terms(zy2, zn2, MARGIN, cf_form="logit")
    (p2 + a2 + c2).mean().backward()
    assert float(zn2.grad[0, 1]) == 0.0, zn2.grad
    assert abs(float(zy2.grad[0, 1])) > 1e-6, zy2.grad
    # Terms 1-2 gradients are untouched by the form change on the absent column.
    assert abs(float(zy2.grad[0, 2]) - float(gy[0, 2])) < 1e-6
    assert abs(float(zn2.grad[0, 2]) - float(gn[0, 2])) < 1e-6

    # The cast to fp32 lives upstream in _answer_logits (`step[...].float()`),
    # not here; pin that so the margin math never sees bf16 from loss().
    fl_src = open(os.path.join(PILOT, "method", "faith_loss.py")).read()
    assert "step = logits.gather(1, idx).squeeze(1).float()" in fl_src


# ---------------------------------------------------------------------------
# FaithfulnessAnchor kwarg (constructor only; no forwards).
# ---------------------------------------------------------------------------

class _Tok:
    """Minimal SentencePiece-like tokenizer: one token per
    'optional whitespace + non-whitespace' run, so a prefix's ids are a prefix
    of any extension starting at a whitespace boundary."""
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=False):
        import re
        ids = []
        for piece in re.findall(r"\s*\S+", text):
            if piece not in self.vocab:
                self.vocab[piece] = len(self.vocab) + 100
            ids.append(self.vocab[piece])
        return {"input_ids": ids}


class _Proc:
    def __init__(self):
        self.tokenizer = _Tok()


def _fixture(tmp):
    root = os.path.join(tmp, "root")
    os.makedirs(os.path.join(root, "img"), exist_ok=True)
    rows = []
    for i in range(2):
        rel, rel_m = f"img/{i}.jpg", f"img/{i}_masked.jpg"
        for r in (rel, rel_m):
            open(os.path.join(root, r), "wb").close()  # init only checks existence
        rows.append({"id": f"p{i}", "image": rel, "masked_image": rel_m,
                     "present": "dog", "absent": "cat"})
    pairs = os.path.join(tmp, "pairs.jsonl")
    with open(pairs, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return pairs, root


def test_anchor_kwarg_defaults_to_logit_and_validates():
    if not HAVE_DEPS:
        return _skip("test_anchor_kwarg_defaults_to_logit_and_validates")
    from model_zoo import get_backbone
    bb = get_backbone("llava15")
    with tempfile.TemporaryDirectory() as tmp:
        pairs, root = _fixture(tmp)
        a = FaithfulnessAnchor(_Proc(), pairs, root, k_pairs=2, device="cpu",
                               backbone=bb)
        assert a.cf_form == "logit", a.cf_form          # default unchanged
        assert a.mode == "margin"
        lo = FaithfulnessAnchor(_Proc(), pairs, root, k_pairs=2, device="cpu",
                                backbone=bb, cf_form="logodds")
        assert lo.cf_form == "logodds"
        # cf_form is orthogonal to mode (no effect in the CE ablation, but the
        # kwarg must still validate).
        ce = FaithfulnessAnchor(_Proc(), pairs, root, k_pairs=2, device="cpu",
                                backbone=bb, mode="ce", cf_form="logodds")
        assert ce.mode == "ce" and ce.cf_form == "logodds"
        try:
            FaithfulnessAnchor(_Proc(), pairs, root, k_pairs=2, device="cpu",
                               backbone=bb, cf_form="log_odds")
        except AssertionError:
            pass
        else:
            raise AssertionError("bad cf_form must be rejected at init")


# ---------------------------------------------------------------------------
# train_lora wiring (flag surface).
# ---------------------------------------------------------------------------

REQUIRED = ["--data", "x.jsonl", "--data_root", "/r", "--out", "/o"]


def test_train_lora_flag_defaults_to_logit():
    if not HAVE_DEPS:
        return _skip("test_train_lora_flag_defaults_to_logit")
    args = train_lora.build_parser().parse_args(REQUIRED)
    assert args.faith_cf_form == "logit", vars(args)
    # Pre-existing faith surface untouched.
    assert args.faith_pairs is None and args.faith_weight == 0.1
    assert args.faith_k == 4 and args.faith_margin == 2.0
    assert args.faith_ce_mode is False
    a = train_lora.build_parser().parse_args(REQUIRED + ["--faith_cf_form", "logodds"])
    assert a.faith_cf_form == "logodds"
    for bad in ("logits", "log_odds", "raw"):
        try:
            with open(os.devnull, "w") as sink, contextlib.redirect_stderr(sink):
                train_lora.build_parser().parse_args(REQUIRED + ["--faith_cf_form", bad])
        except SystemExit:
            pass
        else:
            raise AssertionError(f"--faith_cf_form {bad} must be rejected")


def test_train_lora_passes_cf_form_to_the_anchor():
    if not HAVE_DEPS:
        return _skip("test_train_lora_passes_cf_form_to_the_anchor")
    src = open(os.path.join(PILOT, "train_lora.py")).read()
    assert "cf_form=args.faith_cf_form" in src, "anchor construction drops the flag"
    assert 'ap.add_argument("--faith_cf_form", default="logit", choices=CF_FORMS' in src


# ---------------------------------------------------------------------------
# Arm driver surface (module-level script: checked at source level; torch-free).
# ---------------------------------------------------------------------------

def test_run_arm_exposes_anchorlo_and_keeps_pinned_lines():
    src = open(os.path.join(PILOT, "..", "fullstudy", "run_arm.py")).read()
    for needle in ('args.arm == "anchorlo"', 'extra = anchorlo_extra',
                   'os.environ.get("FAITH_CF_FORM", "logodds")',
                   'anchorlo_extra = anchor_extra + ["--faith_cf_form", _CF]'):
        assert needle in src, f"run_arm.py missing {needle!r}"
    # The anchor arm's own command line must stay free of the new flag.
    assert ('anchor_extra = ["--faith_pairs", _FP,\n'
            '                "--faith_weight", _FW, "--faith_k", "4", '
            '"--faith_margin", _FM]') in src, "anchor_extra was modified"
    # Lines earlier lanes pinned; the integration must preserve them verbatim.
    assert 'MICRO_BATCH = 1 if args.backbone == "qwen25vl" else 2' in src
    assert ('_FP = os.environ.get("FAITH_PAIRS", os.path.join(D, "grounding", '
            '"grounding_pairs.jsonl"))') in src
    assert 'os.environ.get("CRIT_PROBE"' in src
    for arm in ('"critp"', '"anchorcrit"', '"cnp"', '"cnp_ref"', '"policy"',
                '"policy_abst"', '"policy_eos"', '"gca"', '"anchorgca"',
                '"cecf"', '"ewc"', '"lwf"'):
        assert f"args.arm == {arm}" in src, f"run_arm.py lost arm {arm}"


TESTS = [
    test_imports_stay_light,
    test_logit_form_bitwise_identical_to_pre_change,
    test_logit_form_matches_pinned_literals_and_stdlib,
    test_logodds_form_computes_stated_quantity,
    test_decision_terms_are_identical_across_forms,
    test_logodds_is_normalization_free_logit_is_not,
    test_both_forms_zero_when_margins_met_positive_otherwise,
    test_bad_cf_form_and_negative_margin_rejected,
    test_gradients_flow_to_all_four_logits_under_logodds,
    test_anchor_kwarg_defaults_to_logit_and_validates,
    test_train_lora_flag_defaults_to_logit,
    test_train_lora_passes_cf_form_to_the_anchor,
    test_run_arm_exposes_anchorlo_and_keeps_pinned_lines,
]


def main():
    if not HAVE_DEPS:
        print("SKIP: torch/PIL not installed; faith cf_form tests need them "
              "(the arm-driver source check still runs).")
        test_run_arm_exposes_anchorlo_and_keeps_pinned_lines()
        return
    for t in TESTS:
        t()
    print(f"FAITH-CF-FORM OK: {len(TESTS)} checks passed "
          "(cf_form='logit' bitwise identical to the pre-change implementation and "
          "matched to stdlib-recomputed literal pins; 'logodds' = sp(m - [g(I) - "
          "g(I\\o+)]) pinned the same two ways; terms 1-2 bitwise equal across "
          "forms; log-odds invariant under a masked-image logit shift, raw-logit "
          "form not; both forms 0 when their margins are met and positive when not, "
          "including the two cases where they disagree; gradients to all four "
          "logits under logodds with equal-and-opposite pull on the masked pair, "
          "zero on z_no(masked) under logit; anchor kwarg default+validation; "
          "--faith_cf_form default 'logit'; run_arm anchorlo + pinned lines).")


if __name__ == "__main__":
    main()
