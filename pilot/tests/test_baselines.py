"""CPU unit tests for the CL baseline penalty math (ewc.py / olora.py / lwf.py).

Self-contained and CPU-only: no GPU, no model download, no transformers/peft. The
heavy model stack is exercised end-to-end by the cluster smoke gate; here we pin the
*math* of the new penalties:

  * EWC penalty is exactly 0 at stage 1 (empty Fisher) and > 0 with a nonzero
    Fisher, is 0 when theta == theta*, and actually produces gradient on `theta`
    (the CoIN `p.data` bug is fixed).
  * O-LoRA orthogonality term is exactly 0 when the LoRA-A subspaces are orthogonal
    and > 0 otherwise; the model-level penalty is 0 at stage 1 (no snapshot).
  * LwF top-k KD is None when no example carries teacher data, finite/positive
    otherwise, and smaller when the student matches the teacher than when it does
    not.

Torch is guarded: if torch is unavailable the tests skip cleanly (exit 0) rather
than erroring, so the file stays importable/collectable in a torch-free CI the way
tests/test_backbone_parity.py runs torch-free.

    python3 pilot/tests/test_baselines.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    HAVE_TORCH = True
except ImportError:  # torch-guarded: skip on a torch-free interpreter
    HAVE_TORCH = False

_HEAVY = ("transformers", "peft")
_PRELOADED = {m for m in _HEAVY if m in sys.modules}

if HAVE_TORCH:
    import ewc as ewc_mod
    import lwf as lwf_mod
    import olora as olora_mod


def _skip(name):
    print(f"SKIP {name}: torch not installed")


# ---------------------------------------------------------------------------
# Import hygiene: the baseline math must not drag in transformers / peft.
# ---------------------------------------------------------------------------

def test_imports_stay_light():
    if not HAVE_TORCH:
        return _skip("test_imports_stay_light")
    leaked = [m for m in _HEAVY if m in sys.modules and m not in _PRELOADED]
    assert not leaked, (
        f"importing ewc/olora/lwf pulled in heavy modules {leaked}; the CPU "
        "penalty-math test must run without transformers/peft")


# ---------------------------------------------------------------------------
# EWC.
# ---------------------------------------------------------------------------

def test_ewc_penalty_zero_at_stage1():
    if not HAVE_TORCH:
        return _skip("test_ewc_penalty_zero_at_stage1")
    # Stage 1: no Fisher loaded -> empty dict -> penalty exactly 0.0, no grad.
    params = [("w", torch.tensor([1.0, 2.0, 3.0]))]
    pen = ewc_mod.ewc_penalty(params, fisher={}, optpar={}, lam=0.5)
    assert float(pen) == 0.0, f"stage-1 EWC penalty must be 0, got {float(pen)}"


def test_ewc_penalty_positive_with_nonzero_fisher():
    if not HAVE_TORCH:
        return _skip("test_ewc_penalty_positive_with_nonzero_fisher")
    p = torch.tensor([1.0, 2.0])
    fisher = {"w": torch.tensor([1.0, 1.0])}
    optpar = {"w": torch.tensor([0.0, 0.0])}
    # lambda * sum F*(theta-theta*)^2 = 0.5 * (1*1 + 1*4) = 2.5  (CoIN form, no 1/2)
    pen = ewc_mod.ewc_penalty([("w", p)], fisher, optpar, lam=0.5)
    assert float(pen) > 0.0, "EWC penalty must be > 0 with a nonzero Fisher"
    assert abs(float(pen) - 2.5) < 1e-6, f"expected 2.5, got {float(pen)}"


def test_ewc_penalty_zero_when_theta_equals_star():
    if not HAVE_TORCH:
        return _skip("test_ewc_penalty_zero_when_theta_equals_star")
    # The step-0 sanity property the runtime gate relies on: theta == theta* -> 0.
    p = torch.tensor([5.0, -7.0])
    fisher = {"w": torch.tensor([3.0, 9.0])}
    optpar = {"w": p.clone()}
    pen = ewc_mod.ewc_penalty([("w", p)], fisher, optpar, lam=0.5)
    assert abs(float(pen)) < 1e-9, f"penalty must be 0 at theta==theta*, got {pen}"


def test_ewc_penalty_has_gradient():
    if not HAVE_TORCH:
        return _skip("test_ewc_penalty_has_gradient")
    # The CoIN bug is using p.data (detached) -> zero gradient. Our form uses p, so
    # the penalty must produce a real gradient on theta.
    p = torch.tensor([1.0, 2.0], requires_grad=True)
    fisher = {"w": torch.tensor([1.0, 1.0])}
    optpar = {"w": torch.tensor([0.0, 0.0])}
    pen = ewc_mod.ewc_penalty([("w", p)], fisher, optpar, lam=0.5)
    pen.backward()
    assert p.grad is not None and torch.any(p.grad != 0), (
        "EWC penalty produced no gradient on theta (regression of the p.data bug)")
    # d/dtheta [0.5 * F*(theta-star)^2] = F*(theta-star) = [1*1, 1*2] = [1, 2]
    assert torch.allclose(p.grad, torch.tensor([1.0, 2.0]), atol=1e-6), p.grad


def test_ewc_names_not_in_fisher_are_ignored():
    if not HAVE_TORCH:
        return _skip("test_ewc_names_not_in_fisher_are_ignored")
    # Frozen / unmatched params (name absent from Fisher) contribute nothing.
    params = [("w", torch.tensor([9.0])), ("frozen", torch.tensor([100.0]))]
    fisher = {"w": torch.tensor([2.0])}
    optpar = {"w": torch.tensor([0.0])}
    pen = ewc_mod.ewc_penalty(params, fisher, optpar, lam=1.0)
    assert abs(float(pen) - (2.0 * 81.0)) < 1e-5, float(pen)


# ---------------------------------------------------------------------------
# O-LoRA.
# ---------------------------------------------------------------------------

class _Tiny(object):
    """Minimal stand-in exposing named_parameters() with a LoRA-A param, so the
    olora helpers can be tested without peft."""

    def __init__(self, a):
        self._a = torch.nn.Parameter(a) if not isinstance(a, torch.nn.Parameter) else a

    def named_parameters(self):
        yield "base.layer.lora_A.default.weight", self._a


def test_olora_term_zero_when_orthogonal():
    if not HAVE_TORCH:
        return _skip("test_olora_term_zero_when_orthogonal")
    a_cur = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])   # rows span x,y
    a_prev = torch.tensor([[0.0, 0.0, 1.0]])                    # row spans z
    term = olora_mod.orthogonality_term(a_cur, a_prev)
    assert float(term) == 0.0, f"orthogonal subspaces must give 0, got {float(term)}"


def test_olora_term_positive_when_not_orthogonal():
    if not HAVE_TORCH:
        return _skip("test_olora_term_positive_when_not_orthogonal")
    a_cur = torch.tensor([[1.0, 0.0]])
    a_prev = torch.tensor([[1.0, 1.0]])   # inner product 1 -> |.|=1
    term = olora_mod.orthogonality_term(a_cur, a_prev)
    assert float(term) > 0.0, "non-orthogonal subspaces must give > 0"
    assert abs(float(term) - 1.0) < 1e-6, float(term)


def test_olora_penalty_model_level():
    if not HAVE_TORCH:
        return _skip("test_olora_penalty_model_level")
    prev_model = _Tiny(torch.tensor([[1.0, 1.0]]))
    prev_snap = olora_mod.snapshot_lora_A(prev_model)
    assert prev_snap, "snapshot must capture the LoRA-A param"
    # Orthogonal live A -> 0.
    ortho = _Tiny(torch.tensor([[1.0, -1.0]]))  # <[1,-1],[1,1]> = 0
    assert float(olora_mod.olora_penalty(ortho, prev_snap).detach()) == 0.0
    # Non-orthogonal live A -> > 0.
    para = _Tiny(torch.tensor([[2.0, 0.0]]))    # <[2,0],[1,1]> = 2
    pen = float(olora_mod.olora_penalty(para, prev_snap).detach())
    assert pen > 0.0 and abs(pen - 2.0) < 1e-6, pen


def test_olora_penalty_zero_at_stage1():
    if not HAVE_TORCH:
        return _skip("test_olora_penalty_zero_at_stage1")
    # Stage 1: no resumed adapter -> empty snapshot -> penalty exactly 0.0.
    live = _Tiny(torch.tensor([[3.0, 4.0]]))
    assert float(olora_mod.olora_penalty(live, {}).detach()) == 0.0


def test_olora_penalty_has_gradient():
    if not HAVE_TORCH:
        return _skip("test_olora_penalty_has_gradient")
    live = _Tiny(torch.tensor([[2.0, 0.0]]))
    prev = {"base.layer.lora_A.default.weight": torch.tensor([[1.0, 1.0]])}
    pen = olora_mod.olora_penalty(live, prev)
    pen.backward()
    g = next(live.named_parameters())[1].grad
    assert g is not None and torch.any(g != 0), "O-LoRA penalty produced no gradient"


# ---------------------------------------------------------------------------
# LwF (top-k KD).
# ---------------------------------------------------------------------------

def test_lwf_kd_none_when_no_teacher():
    if not HAVE_TORCH:
        return _skip("test_lwf_kd_none_when_no_teacher")
    student = torch.zeros(1, 3, 5)
    assert lwf_mod.kd_loss(student, [None], temp=2.0) is None


def test_lwf_kd_finite_and_ordered():
    if not HAVE_TORCH:
        return _skip("test_lwf_kd_finite_and_ordered")
    V = 6
    student = torch.zeros(1, 4, V)
    kd = {"q": torch.tensor([2]),
          "idx": torch.tensor([[0, 1]]),
          "logit": torch.tensor([[4.0, 0.0]])}  # teacher strongly prefers id 0
    # Student matching the teacher's preference.
    student_match = student.clone()
    student_match[0, 2, 0] = 4.0
    student_match[0, 2, 1] = 0.0
    kd_match = lwf_mod.kd_loss(student_match, [kd], temp=2.0)
    # Student with the opposite preference.
    student_bad = student.clone()
    student_bad[0, 2, 0] = 0.0
    student_bad[0, 2, 1] = 4.0
    kd_bad = lwf_mod.kd_loss(student_bad, [kd], temp=2.0)
    assert kd_match is not None and torch.isfinite(kd_match)
    assert float(kd_match) > 0.0
    assert float(kd_bad) > float(kd_match), (
        f"KD should penalise the mismatched student more: "
        f"match={float(kd_match):.4f} bad={float(kd_bad):.4f}")


TESTS = [
    test_imports_stay_light,
    test_ewc_penalty_zero_at_stage1,
    test_ewc_penalty_positive_with_nonzero_fisher,
    test_ewc_penalty_zero_when_theta_equals_star,
    test_ewc_penalty_has_gradient,
    test_ewc_names_not_in_fisher_are_ignored,
    test_olora_term_zero_when_orthogonal,
    test_olora_term_positive_when_not_orthogonal,
    test_olora_penalty_model_level,
    test_olora_penalty_zero_at_stage1,
    test_olora_penalty_has_gradient,
    test_lwf_kd_none_when_no_teacher,
    test_lwf_kd_finite_and_ordered,
]


def main():
    if not HAVE_TORCH:
        print("SKIP: torch not installed; baseline penalty-math tests need torch. "
              "The math is validated end-to-end by the cluster smoke gate.")
        return
    for t in TESTS:
        t()
    print(f"BASELINES OK: {len(TESTS)} checks passed "
          "(EWC penalty stage-1=0 / >0 with Fisher / grad flows; "
          "O-LoRA orthogonality 0 iff orthogonal; LwF top-k KD ordered).")


if __name__ == "__main__":
    main()
