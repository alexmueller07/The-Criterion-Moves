"""CPU unit tests for the multi-axis policy-anchor penalty math (policy_anchor.py).

Self-contained and CPU-only, like tests/test_baselines.py: no GPU, no model
download, no forward pass. The full class (PolicyAnchor) is exercised end-to-end by
the cluster smoke gate; here we pin the *math* of the three composable terms:

  * abstain_term  is ~0 when the grounded answer dominates the refusal score
    (trivially satisfied), > 0 when refusal is competitive, and produces gradient
    on both the answer logit and every refusal logit. It is softplus-based
    (matching faith_loss's soft hinge), so "satisfied" means asymptotically 0.
  * eos_stability_term is EXACTLY 0 when the live stopping hazard equals the
    reference at every position, > 0 for any drift, and produces gradient.
  * eos_explen_term / expected_length: expected length is monotone-decreasing in
    the stopping hazard; the term is exactly 0 when hazards match, > 0 otherwise.
  * compose_policy_loss equals the weighted sum exactly, a zeroed lambda drops its
    term exactly, and gradient flows only through the enabled terms.

Torch is guarded: on a torch-free (or PIL-free, since importing policy_anchor
pulls in faith_loss) interpreter the tests skip cleanly (exit 0) rather than
erroring, so the file stays importable/collectable in a light CI.

    python3 pilot/tests/test_policy_anchor.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn.functional as F
    # Importing policy_anchor pulls in faith_loss -> torch + PIL. Guard the whole
    # chain: a missing heavy dep skips, never errors (test_baselines convention).
    from method.policy_anchor import (
        abstain_term,
        eos_stability_term,
        eos_explen_term,
        expected_length,
        compose_policy_loss,
    )
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


def _skip(name):
    print(f"SKIP {name}: torch/PIL (policy_anchor deps) not installed")


# ---------------------------------------------------------------------------
# Abstention term (drift component #2).
# ---------------------------------------------------------------------------

def test_abstain_zero_when_answer_dominates():
    if not HAVE_DEPS:
        return _skip("test_abstain_zero_when_answer_dominates")
    # Grounded answer logit 12, single refusal logit -12 -> gap 24 >> margin 2 ->
    # softplus(2 - 24) ~ 0 (asymptotic; the soft-hinge analogue of "exactly 0").
    z_answer = torch.tensor(12.0)
    z_refuse = torch.tensor([-12.0])
    val = abstain_term(z_answer, z_refuse, margin=2.0)
    assert float(val) < 1e-4, f"trivially-satisfied abstain must be ~0, got {float(val)}"


def test_abstain_positive_when_refusal_competitive():
    if not HAVE_DEPS:
        return _skip("test_abstain_positive_when_refusal_competitive")
    # Equal logits -> gap 0 -> softplus(margin) = softplus(2) > 0.
    z_answer = torch.tensor(0.0)
    z_refuse = torch.tensor([0.0])
    val = abstain_term(z_answer, z_refuse, margin=2.0)
    assert float(val) > 0.0, "competitive refusal must give a positive penalty"
    assert abs(float(val) - float(F.softplus(torch.tensor(2.0)))) < 1e-6, float(val)


def test_abstain_monotone_in_refusal():
    if not HAVE_DEPS:
        return _skip("test_abstain_monotone_in_refusal")
    # Raising the refusal logit (more spurious abstention) must raise the penalty.
    z_answer = torch.tensor(1.0)
    low = abstain_term(z_answer, torch.tensor([-3.0]), margin=2.0)
    high = abstain_term(z_answer, torch.tensor([3.0]), margin=2.0)
    assert float(high) > float(low), (
        f"penalty must increase with refusal strength: {float(low)} -> {float(high)}")


def test_abstain_logsumexp_aggregates_refusals():
    if not HAVE_DEPS:
        return _skip("test_abstain_logsumexp_aggregates_refusals")
    # Multiple refusal tokens aggregate by logsumexp (>= the strongest single one),
    # so more refusal mass never lowers the penalty vs one token alone.
    z_answer = torch.tensor(1.0)
    one = abstain_term(z_answer, torch.tensor([0.0]), margin=2.0)
    many = abstain_term(z_answer, torch.tensor([0.0, 0.0, 0.0]), margin=2.0)
    assert float(many) >= float(one) - 1e-6, (float(one), float(many))
    # logsumexp([0,0,0]) = log 3, so the aggregate refusal is strictly larger.
    assert float(many) > float(one), (float(one), float(many))


def test_abstain_gradient_flows():
    if not HAVE_DEPS:
        return _skip("test_abstain_gradient_flows")
    z_answer = torch.tensor(0.5, requires_grad=True)
    z_refuse = torch.tensor([0.3, -0.2], requires_grad=True)
    val = abstain_term(z_answer, z_refuse, margin=2.0)
    val.backward()
    assert z_answer.grad is not None and float(z_answer.grad) != 0.0, z_answer.grad
    assert z_refuse.grad is not None and torch.any(z_refuse.grad != 0), z_refuse.grad
    # Pushing the answer up lowers the loss: d/dz_answer < 0.
    assert float(z_answer.grad) < 0.0, z_answer.grad
    # Pushing any refusal up raises the loss: d/dz_refuse > 0.
    assert torch.all(z_refuse.grad > 0), z_refuse.grad


# ---------------------------------------------------------------------------
# EOS/length stability term (drift component #3), per-position form.
# ---------------------------------------------------------------------------

def test_eos_stability_zero_when_matched():
    if not HAVE_DEPS:
        return _skip("test_eos_stability_zero_when_matched")
    lp = torch.log(torch.tensor([0.1, 0.2, 0.05, 0.6]))
    val = eos_stability_term(lp, lp.clone())
    assert float(val) == 0.0, f"matched hazard must give exactly 0, got {float(val)}"


def test_eos_stability_positive_on_drift():
    if not HAVE_DEPS:
        return _skip("test_eos_stability_positive_on_drift")
    lp = torch.log(torch.tensor([0.1, 0.2, 0.05, 0.6]))
    lp_ref = torch.log(torch.tensor([0.1, 0.2, 0.05, 0.3]))  # drifted last position
    val = eos_stability_term(lp, lp_ref)
    assert float(val) > 0.0, "any hazard drift must give a positive penalty"
    # mean of squared per-position log-prob deltas; only the last differs.
    expect = ((lp - lp_ref) ** 2).mean()
    assert abs(float(val) - float(expect)) < 1e-9, float(val)


def test_eos_stability_gradient_flows():
    if not HAVE_DEPS:
        return _skip("test_eos_stability_gradient_flows")
    lp = torch.log(torch.tensor([0.1, 0.2, 0.05, 0.6])).requires_grad_(True)
    lp_ref = torch.log(torch.tensor([0.2, 0.2, 0.05, 0.3]))
    val = eos_stability_term(lp, lp_ref)
    val.backward()
    assert lp.grad is not None and torch.any(lp.grad != 0), lp.grad


# ---------------------------------------------------------------------------
# EOS/length stability term, expected-length (scalar) variant.
# ---------------------------------------------------------------------------

def test_expected_length_monotone_in_hazard():
    if not HAVE_DEPS:
        return _skip("test_expected_length_monotone_in_hazard")
    # A model that stops sooner (higher EOS hazard everywhere) has a shorter
    # expected length -- the late-caption-hazard mechanism, sanity-checked.
    T = 6
    low_hazard = torch.log(torch.full((T,), 0.01))   # rarely stops -> long
    high_hazard = torch.log(torch.full((T,), 0.5))   # stops fast   -> short
    assert float(expected_length(low_hazard)) > float(expected_length(high_hazard))


def test_eos_explen_zero_when_matched_and_positive_on_drift():
    if not HAVE_DEPS:
        return _skip("test_eos_explen_zero_when_matched_and_positive_on_drift")
    lp = torch.log(torch.tensor([0.05, 0.05, 0.05, 0.05, 0.7]))
    assert float(eos_explen_term(lp, lp.clone())) == 0.0
    lp_short = torch.log(torch.tensor([0.4, 0.4, 0.4, 0.4, 0.7]))  # stops earlier
    assert float(eos_explen_term(lp_short, lp)) > 0.0


def test_eos_explen_gradient_flows():
    if not HAVE_DEPS:
        return _skip("test_eos_explen_gradient_flows")
    lp = torch.log(torch.tensor([0.2, 0.2, 0.2, 0.4])).requires_grad_(True)
    lp_ref = torch.log(torch.tensor([0.05, 0.05, 0.05, 0.4]))
    val = eos_explen_term(lp, lp_ref)
    val.backward()
    assert lp.grad is not None and torch.any(lp.grad != 0), lp.grad


# ---------------------------------------------------------------------------
# Composite.
# ---------------------------------------------------------------------------

def test_compose_matches_weighted_sum():
    if not HAVE_DEPS:
        return _skip("test_compose_matches_weighted_sum")
    lc, la, le = torch.tensor(1.5), torch.tensor(0.4), torch.tensor(2.0)
    lam_c, lam_a, lam_e = 0.1, 0.2, 0.05
    val = compose_policy_loss(lc, la, le, lam_c, lam_a, lam_e)
    expect = lam_c * 1.5 + lam_a * 0.4 + lam_e * 2.0
    assert abs(float(val) - expect) < 1e-6, (float(val), expect)


def test_compose_zero_lambda_drops_term():
    if not HAVE_DEPS:
        return _skip("test_compose_zero_lambda_drops_term")
    # With lambda_a = lambda_e = 0 the composite is exactly lambda_c * criterion --
    # the property that makes PolicyAnchor a strict superset of the v2 anchor and
    # each axis independently ablatable.
    lc, la, le = torch.tensor(1.5), torch.tensor(99.0), torch.tensor(99.0)
    val = compose_policy_loss(lc, la, le, 0.1, 0.0, 0.0)
    assert abs(float(val) - 0.1 * 1.5) < 1e-6, float(val)


def test_compose_gradient_only_through_enabled_terms():
    if not HAVE_DEPS:
        return _skip("test_compose_gradient_only_through_enabled_terms")
    lc = torch.tensor(1.5, requires_grad=True)
    la = torch.tensor(0.4, requires_grad=True)
    le = torch.tensor(2.0, requires_grad=True)
    # Enable criterion + eos, ablate abstain.
    val = compose_policy_loss(lc, la, le, lam_c=0.1, lam_a=0.0, lam_e=0.05)
    val.backward()
    assert abs(float(lc.grad) - 0.1) < 1e-6, lc.grad
    assert float(la.grad) == 0.0, la.grad          # ablated term gets no gradient
    assert abs(float(le.grad) - 0.05) < 1e-6, le.grad


TESTS = [
    test_abstain_zero_when_answer_dominates,
    test_abstain_positive_when_refusal_competitive,
    test_abstain_monotone_in_refusal,
    test_abstain_logsumexp_aggregates_refusals,
    test_abstain_gradient_flows,
    test_eos_stability_zero_when_matched,
    test_eos_stability_positive_on_drift,
    test_eos_stability_gradient_flows,
    test_expected_length_monotone_in_hazard,
    test_eos_explen_zero_when_matched_and_positive_on_drift,
    test_eos_explen_gradient_flows,
    test_compose_matches_weighted_sum,
    test_compose_zero_lambda_drops_term,
    test_compose_gradient_only_through_enabled_terms,
]


def main():
    if not HAVE_DEPS:
        print("SKIP: torch/PIL not installed; policy-anchor math tests need them. "
              "The math is validated end-to-end by the cluster smoke gate.")
        return
    for t in TESTS:
        t()
    print(f"POLICY-ANCHOR OK: {len(TESTS)} checks passed "
          "(abstain ~0 when answered / >0 when refusal competitive / grad flows; "
          "EOS stability exactly 0 iff hazard matched, monotone expected-length; "
          "composite == weighted sum, zeroed lambda drops its term).")


if __name__ == "__main__":
    main()
