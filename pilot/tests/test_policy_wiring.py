"""CPU wiring test for the multi-axis policy anchor in train_lora.py.

No GPU, no model download, no transformers/peft (train_lora imports those lazily
inside main()). Pins the parts of the integration that can be checked without a
model, the way tests/test_baselines.py pins the penalty math:

  * the --pol_* flags parse with the documented defaults (both weights 0, k 8) and
    `policy_requested` is False at the defaults -> no PolicyAnchor is built, so
    every existing arm is byte-identical when the flags are unset;
  * `add_policy_term` (the compute_loss step) equals
        loss + accum * (w_abstain * L_abstain + w_eos * L_eos)
    with the criterion weight pinned to 0 (the criterion term is the --faith_pairs
    anchor, added separately), logs the raw per-term values, and routes gradient
    only through the enabled terms -- exercised with a MOCKED anchor;
  * the theta0 EOS-reference routing (disable_adapter -> stage-1 identity ->
    previous-stage cache -> error) and the JSON cache round-trip / stale
    rejection, exercised on a stub PolicyAnchor whose forwards are replaced;
  * fullstudy/run_arm.py exposes the three arms and env overrides and still
    carries its MICRO_BATCH / FAITH_PAIRS lines.

    python3 pilot/tests/test_policy_wiring.py
"""
import contextlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
sys.path.insert(0, PILOT)

try:
    import torch
    import train_lora
    from method.policy_anchor import (
        EOS_REF_CACHE,
        PolicyAnchor,
        load_eos_reference,
        save_eos_reference,
    )
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


def _skip(name):
    print(f"SKIP {name}: torch/PIL (train_lora deps) not installed")


REQUIRED = ["--data", "x.jsonl", "--data_root", "/r", "--out", "/o"]


# ---------------------------------------------------------------------------
# Flag parsing.
# ---------------------------------------------------------------------------

def test_flags_default_off():
    if not HAVE_DEPS:
        return _skip("test_flags_default_off")
    args = train_lora.build_parser().parse_args(REQUIRED)
    assert args.pol_weight_abstain == 0.0 and args.pol_weight_eos == 0.0, vars(args)
    assert args.pol_k == 8 and args.pol_captions is None, vars(args)
    assert args.pol_margin == 2.0 and args.pol_eos_mode == "logprob", vars(args)
    assert train_lora.policy_requested(args) is False
    # Pre-existing flag surface untouched.
    assert args.faith_pairs is None and args.faith_weight == 0.1 and args.faith_k == 4
    assert args.crit_probe is None and args.crit_weight == 0.0 and args.crit_k == 8


def test_flags_parse_overrides():
    if not HAVE_DEPS:
        return _skip("test_flags_parse_overrides")
    args = train_lora.build_parser().parse_args(REQUIRED + [
        "--pol_weight_abstain", "0.1", "--pol_weight_eos", "0.05", "--pol_k", "4",
        "--pol_captions", "cap.jsonl", "--pol_margin", "1.5", "--pol_eos_mode", "explen"])
    assert args.pol_weight_abstain == 0.1 and args.pol_weight_eos == 0.05
    assert args.pol_k == 4 and args.pol_captions == "cap.jsonl"
    assert args.pol_margin == 1.5 and args.pol_eos_mode == "explen"
    assert train_lora.policy_requested(args) is True
    # Either term alone requests the anchor (the single-term ablation arms).
    for flag in ("--pol_weight_abstain", "--pol_weight_eos"):
        a = train_lora.build_parser().parse_args(REQUIRED + [flag, "0.1"])
        assert train_lora.policy_requested(a) is True, flag


def test_flags_reject_bad_eos_mode():
    if not HAVE_DEPS:
        return _skip("test_flags_reject_bad_eos_mode")
    try:
        with open(os.devnull, "w") as sink, contextlib.redirect_stderr(sink):
            train_lora.build_parser().parse_args(REQUIRED + ["--pol_eos_mode", "foo"])
    except SystemExit:
        pass
    else:
        raise AssertionError("--pol_eos_mode foo must be rejected")


# ---------------------------------------------------------------------------
# Loss composition path with a mocked anchor.
# ---------------------------------------------------------------------------

class MockAnchor:
    """Stands in for PolicyAnchor: returns fixed per-term tensors."""

    def __init__(self, crit, abst, eos):
        self.comps = {"criterion": crit, "abstain": abst, "eos": eos}
        self.calls = 0

    def loss_components(self, model):
        self.calls += 1
        return self.comps


def test_add_policy_term_composes_and_logs():
    if not HAVE_DEPS:
        return _skip("test_add_policy_term_composes_and_logs")
    crit = torch.tensor(99.0, requires_grad=True)   # must be IGNORED (weight pinned 0)
    abst = torch.tensor(0.7, requires_grad=True)
    eos = torch.tensor(0.25, requires_grad=True)
    pol = MockAnchor(crit, abst, eos)
    base = torch.tensor(2.0, requires_grad=True)
    hist = []
    wa, we, accum = 0.1, 0.05, 4
    out = train_lora.add_policy_term(base, pol, model=None, w_abstain=wa, w_eos=we,
                                     accum=accum, history=hist)
    expect = 2.0 + accum * (wa * 0.7 + we * 0.25)
    assert abs(float(out) - expect) < 1e-4, (float(out), expect)
    assert pol.calls == 1
    assert len(hist)==1 and abs(hist[0]["abstain"]-0.7)<1e-5 and abs(hist[0]["eos"]-0.25)<1e-5, hist   # raw, unweighted
    out.backward()
    assert abs(float(abst.grad) - accum * wa) < 1e-6, abst.grad
    assert abs(float(eos.grad) - accum * we) < 1e-6, eos.grad
    assert crit.grad is None or float(crit.grad) == 0.0, crit.grad
    assert float(base.grad) == 1.0, base.grad


def test_add_policy_term_zero_weight_drops_term():
    if not HAVE_DEPS:
        return _skip("test_add_policy_term_zero_weight_drops_term")
    # policy_abst arm: eos weight 0 -> no gradient through the eos term, and the
    # value is exactly the abstain contribution (composite == weighted sum).
    abst = torch.tensor(0.7, requires_grad=True)
    eos = torch.tensor(5.0, requires_grad=True)
    pol = MockAnchor(torch.tensor(0.0), abst, eos)
    hist = []
    out = train_lora.add_policy_term(torch.tensor(1.0), pol, None, 0.1, 0.0, 2, hist)
    assert abs(float(out) - (1.0 + 2 * 0.1 * 0.7)) < 1e-6, float(out)
    out.backward()
    assert float(eos.grad) == 0.0, eos.grad
    assert abs(float(abst.grad) - 0.2) < 1e-6, abst.grad
    assert hist[0]["eos"] == 5.0   # still logged (monitor value) even at weight 0


# ---------------------------------------------------------------------------
# theta0 EOS-reference routing + cache (stub anchor; forwards replaced).
# ---------------------------------------------------------------------------

class _Disabler:
    def __init__(self, model):
        self.m = model

    def __enter__(self):
        self.m.adapter_disabled = True

    def __exit__(self, *exc):
        self.m.adapter_disabled = False
        return False


class PeftLikeModel:
    """Has PEFT's context-manager disable_adapter(); tracks whether forwards
    happened with the adapter off."""

    def __init__(self):
        self.training = True
        self.adapter_disabled = False
        self.seen_disabled = []

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def disable_adapter(self):
        return _Disabler(self)


class PlainModel:
    def __init__(self):
        self.training = True
        self.adapter_disabled = False
        self.seen_disabled = []

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


CAPTIONS = [{"id": "ground_1", "image": "a.jpg", "caption": "a cat"},
            {"id": "ground_2", "image": "b.jpg", "caption": "a dog on a rug"}]
TRAJ = {"ground_1": [-3.0, -2.0, -0.5], "ground_2": [-4.0, -3.5, -2.0, -1.0, -0.2]}


class StubAnchor(PolicyAnchor):
    """PolicyAnchor with the model forward replaced by a table lookup. Built via
    __new__ (no processor / images); only the attributes the EOS routing and
    _eos_loss touch are set."""

    def _eos_logprobs(self, model, row):
        model.seen_disabled.append(model.adapter_disabled)
        if self.ref_eos is None:  # reference computation; the live loss path runs in train mode by design
            assert not model.training, "reference must be computed in eval mode"
        return torch.tensor(TRAJ[row["id"]], dtype=torch.float32)


def _stub(use_eos=True, k_caps=2):
    import random as _random
    a = StubAnchor.__new__(StubAnchor)
    a.use_eos = use_eos
    a.captions = list(CAPTIONS)
    a.caption_path = "/nonexistent/caption_anchor.jsonl"
    a.caption_prompt = "Describe the image in detail."
    a.eos_mode = "logprob"
    a.stop_id = 2
    a.k_caps = k_caps
    a.ref_eos = None
    a.ref_source = None
    a.rng_e = _random.Random(19)
    a.bb = type("BB", (), {"name": "stub"})()
    return a


def test_eos_reference_cache_roundtrip_and_stale_rejection():
    if not HAVE_DEPS:
        return _skip("test_eos_reference_cache_roundtrip_and_stale_rejection")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "nested", EOS_REF_CACHE)
        save_eos_reference(path, TRAJ, {"source": "unit", "stop_id": 2})
        back, payload = load_eos_reference(path, ["ground_2", "ground_1"])
        assert back == TRAJ, back
        assert payload["source"] == "unit" and payload["n"] == 2
        assert json.load(open(path))["stop_id"] == 2
        for bad in (["ground_1"], ["ground_1", "ground_2", "ground_3"]):
            try:
                load_eos_reference(path, bad)
            except AssertionError:
                pass
            else:
                raise AssertionError(f"cache with wrong id set {bad} must be rejected")


def test_routing_disable_adapter_then_cache_then_loss():
    if not HAVE_DEPS:
        return _skip("test_routing_disable_adapter_then_cache_then_loss")
    with tempfile.TemporaryDirectory() as d:
        s1 = os.path.join(d, "S1", EOS_REF_CACHE)
        # Route 1: PEFT disable_adapter context -> forwards run with adapter OFF.
        a = _stub()
        m = PeftLikeModel()
        src = a.init_eos_reference(m, cache_path=s1, fallback_path=None, is_stage1=True)
        assert src == "disable_adapter" and a.ref_source == src, src
        assert m.seen_disabled == [True, True], m.seen_disabled
        assert m.adapter_disabled is False and m.training is True  # restored
        assert all(torch.allclose(t.float(), torch.tensor(TRAJ[k]), atol=1e-5) for t, k in zip(a.ref_eos, ["ground_1", "ground_2"])), [t.tolist() for t in a.ref_eos]
        assert os.path.exists(s1)
        meta = json.load(open(s1))
        assert meta["source"] == "disable_adapter" and meta["n"] == 2 and meta["stop_id"] == 2
        # Live == reference -> the EOS term is exactly 0 (the stage-1 gate).
        val = a._eos_loss(m)
        assert float(val) == 0.0, float(val)

        # Route 2: no disabler, stage 1 -> the fresh LoRA IS the base.
        b = _stub()
        src = b.init_eos_reference(PlainModel(), cache_path=None, is_stage1=True)
        assert src == "stage1_fresh_lora_is_base", src

        # Route 3: no disabler, not stage 1, previous stage's cache -> loaded, and
        # re-written to this stage's cache with the audit source updated.
        c = _stub()
        s2 = os.path.join(d, "S2", EOS_REF_CACHE)
        src = c.init_eos_reference(PlainModel(), cache_path=s2, fallback_path=s1,
                                   is_stage1=False)
        assert src.startswith("loaded:") and s1 in src, src
        assert all(torch.allclose(t.float(), torch.tensor(TRAJ[k]), atol=1e-5) for t, k in zip(c.ref_eos, ["ground_1", "ground_2"])), [t.tolist() for t in c.ref_eos]
        assert json.load(open(s2))["source"] == src
        assert float(c._eos_loss(PlainModel())) == 0.0

        # Route 4: nothing available -> loud failure, never a silent reference.
        e = _stub()
        try:
            e.init_eos_reference(PlainModel(), cache_path=None,
                                 fallback_path=os.path.join(d, "missing.json"),
                                 is_stage1=False)
        except RuntimeError:
            pass
        else:
            raise AssertionError("must refuse to run without a theta0 reference")

        # Disabled term: no-op, no forward.
        f = _stub(use_eos=False)
        mm = PeftLikeModel()
        assert f.init_eos_reference(mm, cache_path=s1, is_stage1=True) is None
        assert mm.seen_disabled == []


def test_eos_loss_refuses_without_reference_and_sees_drift():
    if not HAVE_DEPS:
        return _skip("test_eos_loss_refuses_without_reference_and_sees_drift")
    a = _stub()
    try:
        a._eos_loss(PlainModel())
    except AssertionError:
        pass
    else:
        raise AssertionError("_eos_loss must refuse to run before the reference exists")
    # Drifted reference -> positive penalty (per-position squared delta mean).
    a.ref_eos = [torch.tensor(TRAJ["ground_1"]) - 1.0, torch.tensor(TRAJ["ground_2"])]
    a.k_caps = 2
    val = a._eos_loss(PlainModel())
    assert abs(float(val) - 0.5) < 1e-6, float(val)   # mean over rows of (1.0, 0.0)


# ---------------------------------------------------------------------------
# Arm driver surface (module-level script: checked at source level).
# ---------------------------------------------------------------------------

def test_run_arm_exposes_policy_arms_and_keeps_pinned_lines():
    src = open(os.path.join(PILOT, "..", "fullstudy", "run_arm.py")).read()
    for needle in ('args.arm == "policy"', 'args.arm == "policy_abst"',
                   'args.arm == "policy_eos"', 'os.environ.get("POL_W_ABST"',
                   'os.environ.get("POL_W_EOS"', "--pol_weight_abstain",
                   "--pol_weight_eos", "--pol_captions"):
        assert needle in src, f"run_arm.py missing {needle!r}"
    # Lines the integration must preserve verbatim.
    assert 'MICRO_BATCH = 1 if args.backbone == "qwen25vl" else 2' in src
    assert ('_FP = os.environ.get("FAITH_PAIRS", os.path.join(D, "grounding", '
            '"grounding_pairs.jsonl"))') in src


TESTS = [
    test_flags_default_off,
    test_flags_parse_overrides,
    test_flags_reject_bad_eos_mode,
    test_add_policy_term_composes_and_logs,
    test_add_policy_term_zero_weight_drops_term,
    test_eos_reference_cache_roundtrip_and_stale_rejection,
    test_routing_disable_adapter_then_cache_then_loss,
    test_eos_loss_refuses_without_reference_and_sees_drift,
    test_run_arm_exposes_policy_arms_and_keeps_pinned_lines,
]


def main():
    if not HAVE_DEPS:
        print("SKIP: torch/PIL not installed; policy-wiring tests need them "
              "(the arm-driver source check still runs).")
        test_run_arm_exposes_policy_arms_and_keeps_pinned_lines()
        return
    for t in TESTS:
        t()
    print(f"POLICY-WIRING OK: {len(TESTS)} checks passed "
          "(flags default off / parse / reject; add_policy_term == loss + "
          "accum*(w_a*L_a + w_e*L_e) with criterion pinned 0, per-term log, "
          "gradient only via enabled terms; theta0 routing disable_adapter -> "
          "stage1 -> cache -> error, cache round-trip + stale rejection; "
          "run_arm arms + pinned lines).")


if __name__ == "__main__":
    main()
