"""CPU unit tests for the antisymmetry criterion penalty (pilot/method/crit_antisym.py).

Pins the properties the method's argument actually rests on, so a refactor cannot
quietly break them:

  1. r = (g+ + g-)/2 recovers the injected bias b exactly under ideal antisymmetry.
  2. "mean" mode is blind to zero-mean per-item disagreement -- it constrains the
     CRITERION only, which is what differentiates this family from the per-item
     (LwF / RCL L_pred) family.
  3. The "mean"-mode gradient is IDENTICAL for every item in the batch. This is
     the load-bearing claim in the module docstring.
  4. "item" mode is NOT blind to per-item disagreement (it is the stronger
     ablation, and would also move d').
  5. flip_correlation returns None rather than a fake 0.0 on a constant input,
     and correctly separates the healthy case from the polarity-blind failure.
  6. THE FAILURE MODE: when the model ignores polarity (g- == g+), the penalty
     reduces to a penalty on g itself, i.e. it attacks d'. Pinned explicitly so
     nobody later mistakes a falling loss for a working method.
"""
import os
import sys
import unittest

import torch

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
# Two layouts must both work. In the REPO the modules live under pilot/ and
# pilot/method/; on the CLUSTER the repo's pilot/ is flattened into code/, so the
# same files sit at code/ and code/method/. Probe for whichever exists rather
# than hard-coding one -- a wrong guess here shows up as a confusing
# ModuleNotFoundError on crit_preserve, not on the module under test.
# crit_preserve imports `method.faith_loss`, so the PARENT of method/ must be on
# the path as well as method/ itself.
for _parent in (os.path.join(_ROOT, "pilot"), _ROOT):
    if os.path.isfile(os.path.join(_parent, "method", "crit_preserve.py")):
        sys.path.insert(0, _parent)
        sys.path.insert(0, os.path.join(_parent, "method"))
        break
else:
    raise SystemExit("cannot locate method/crit_preserve.py under %s" % _ROOT)
from crit_antisym import (antisym_penalty, antisym_residual,  # noqa: E402
                          flip_correlation)


class TestAntisymResidual(unittest.TestCase):
    def test_recovers_injected_bias_exactly(self):
        t = torch.tensor([2.0, -1.5, 0.7, -3.0, 4.2])
        for b in (0.0, 0.431, -1.2):
            g_pos = t + b
            g_neg = -t + b
            r = antisym_residual(g_pos, g_neg)
            self.assertTrue(torch.allclose(r, torch.full_like(r, b), atol=1e-6),
                            f"bias {b} not recovered: {r}")

    def test_zero_bias_gives_zero_penalty(self):
        t = torch.tensor([1.0, -2.0, 3.0, 0.5])
        val = antisym_penalty(t, -t, mode="mean")
        self.assertAlmostEqual(float(val), 0.0, places=6)

    def test_penalty_is_squared_bias(self):
        t = torch.tensor([1.0, -2.0, 3.0, 0.5])
        b = 0.7
        val = antisym_penalty(t + b, -t + b, mode="mean")
        self.assertAlmostEqual(float(val), b * b, places=5)


class TestModeSemantics(unittest.TestCase):
    def test_mean_mode_ignores_zero_mean_disagreement(self):
        """Per-item antisymmetry violations that cancel cost nothing in mean mode.

        This is the differentiator from the per-item family: individual decisions
        may move, only a COMMON offset is opposed.
        """
        t = torch.tensor([1.0, -2.0, 3.0, 0.5])
        eps = torch.tensor([0.6, -0.6, 0.9, -0.9])   # sums to zero
        val = antisym_penalty(t + eps, -t + eps, mode="mean")
        self.assertAlmostEqual(float(val), 0.0, places=6)

    def test_item_mode_does_not_ignore_it(self):
        t = torch.tensor([1.0, -2.0, 3.0, 0.5])
        eps = torch.tensor([0.6, -0.6, 0.9, -0.9])
        val = antisym_penalty(t + eps, -t + eps, mode="item")
        self.assertGreater(float(val), 0.1)

    def test_mean_mode_gradient_is_item_invariant(self):
        """dL/dg_i must be the same scalar for every i -- the docstring's claim."""
        t = torch.tensor([1.0, -2.0, 3.0, 0.5, -0.25])
        b = 0.4
        g_pos = (t + b).clone().requires_grad_(True)
        g_neg = (-t + b).clone().requires_grad_(True)
        antisym_penalty(g_pos, g_neg, mode="mean").backward()
        for grad in (g_pos.grad, g_neg.grad):
            self.assertTrue(torch.allclose(grad, grad[0].expand_as(grad),
                                           atol=1e-7),
                            f"gradient not item-invariant: {grad}")
        # and both polarities receive the same push
        self.assertTrue(torch.allclose(g_pos.grad, g_neg.grad, atol=1e-7))

    def test_item_mode_gradient_is_not_item_invariant(self):
        t = torch.tensor([1.0, -2.0, 3.0, 0.5])
        eps = torch.tensor([0.6, -0.6, 0.9, -0.9])
        g_pos = (t + eps).clone().requires_grad_(True)
        g_neg = (-t + eps).clone().requires_grad_(True)
        antisym_penalty(g_pos, g_neg, mode="item").backward()
        self.assertFalse(torch.allclose(g_pos.grad,
                                        g_pos.grad[0].expand_as(g_pos.grad),
                                        atol=1e-7))


class TestFlipCorrelation(unittest.TestCase):
    def test_healthy_case_is_strongly_negative(self):
        torch.manual_seed(0)
        t = torch.randn(200)
        c = flip_correlation(t + 0.4, -t + 0.4)
        self.assertIsNotNone(c)
        self.assertLess(c, -0.95)

    def test_polarity_blind_case_is_strongly_positive(self):
        """The pre-registered FAILURE: model ignores the flip, g- ~ g+."""
        torch.manual_seed(0)
        t = torch.randn(200)
        c = flip_correlation(t, t + 0.01 * torch.randn(200))
        self.assertIsNotNone(c)
        self.assertGreater(c, 0.9)

    def test_constant_input_returns_none_not_zero(self):
        """A fake 0.0 here would read as 'no relationship' instead of 'undefined'."""
        const = torch.full((50,), 1.5)
        self.assertIsNone(flip_correlation(const, torch.randn(50)))
        self.assertIsNone(flip_correlation(torch.randn(50), const))

    def test_too_few_points_returns_none(self):
        self.assertIsNone(flip_correlation(torch.randn(2), torch.randn(2)))


class TestFailureMode(unittest.TestCase):
    def test_polarity_blind_penalty_attacks_the_statistic_itself(self):
        """If g- == g+ then r == g, so minimizing the penalty shrinks g -- i.e. it
        destroys d' rather than centring the criterion. Pinned so that a falling
        loss is never mistaken for a working method."""
        g = torch.tensor([2.0, 1.0, 3.0, -1.0])
        r = antisym_residual(g, g)
        self.assertTrue(torch.allclose(r, g, atol=1e-7))
        # the gradient pushes every g_i toward zero, not toward a common offset
        gp = g.clone().requires_grad_(True)
        antisym_penalty(gp, gp, mode="item").backward()
        self.assertTrue(torch.all(torch.sign(gp.grad) == torch.sign(g)),
                        "polarity-blind item-mode gradient should shrink |g|")


if __name__ == "__main__":
    unittest.main(verbosity=2)
