"""CPU unit tests for the label-free criterion-preservation penalty math
(method/crit_preserve.py).

Self-contained and CPU-only, like tests/test_baselines.py / test_policy_anchor.py:
no GPU, no model download, no forward pass. The class (CriterionPreserver) is
exercised end-to-end by the cluster smoke gate; here we pin the *math*, and in
particular the ONE property that separates the method from per-item distillation
(LwF / PFCL / RCL L_pred / LLaVA-c UIR):

  * "mean" mode is EXACTLY 0 for any zero-mean per-item perturbation of the base
    statistics -- individual decisions may move, only the criterion is held --
    while "item" mode (the per-item ablation) is strictly larger for the same
    perturbation.
  * Under the calibration theory's additive-bias model g = g0 + b, "mean" mode
    equals b^2 exactly, independent of which items were sampled.
  * The "mean"-mode gradient is IDENTICAL across items ((2/k) * mean shift):
    it never asks item i to return to its own base value. "item" mode's is not.
  * "meanvar" is 0 under any permutation of the base statistics (mean and std
    preserved, every item changed) and > 0 when the dispersion changes.
  * The base-stat JSON cache round-trips and refuses a stale id set.

Torch is guarded: on a torch-free (or PIL-free) interpreter the tests skip
cleanly (exit 0) rather than erroring.

    python3 pilot/tests/test_crit_preserve.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_HEAVY = ("transformers", "peft")
_PRELOADED = {m for m in _HEAVY if m in sys.modules}

try:
    import torch
    import torch.nn.functional as F
    from method.crit_preserve import (
        MODES,
        crit_penalty,
        decision_stat,
        load_base_stats,
        population_moments,
        save_base_stats,
    )
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


def _skip(name):
    print(f"SKIP {name}: torch/PIL (crit_preserve deps) not installed")


G0 = [3.0, -5.0, -7.5, 1.0, -2.0, -4.0, 0.5, -9.0]  # a plausible base batch


def test_imports_stay_light():
    if not HAVE_DEPS:
        return _skip("test_imports_stay_light")
    leaked = [m for m in _HEAVY if m in sys.modules and m not in _PRELOADED]
    assert not leaked, (
        f"importing crit_preserve pulled in heavy modules {leaked}; the CPU "
        "penalty-math test must run without transformers/peft")


def test_decision_stat_fp32():
    if not HAVE_DEPS:
        return _skip("test_decision_stat_fp32")
    zy = torch.tensor([2.0, -1.0], dtype=torch.bfloat16)
    zn = torch.tensor([0.5, 0.5], dtype=torch.bfloat16)
    g = decision_stat(zy, zn)
    assert g.dtype == torch.float32, g.dtype
    assert torch.allclose(g, torch.tensor([1.5, -1.5])), g


def test_mean_zero_when_paired_equal():
    if not HAVE_DEPS:
        return _skip("test_mean_zero_when_paired_equal")
    g0 = torch.tensor(G0)
    for mode in ("mean", "meanvar"):
        val = crit_penalty(g0.clone(), g0, mode=mode)
        assert float(val) == 0.0, f"{mode}: identical stats must give exactly 0, got {float(val)}"


def test_mean_ignores_zero_mean_per_item_perturbation_item_does_not():
    if not HAVE_DEPS:
        return _skip("test_mean_ignores_zero_mean_per_item_perturbation_item_does_not")
    # THE differentiating property. Every item moves (by +-2 logits), the batch
    # mean does not: the criterion is untouched, so "mean" mode must be 0.
    g0 = torch.tensor(G0)
    eps = torch.tensor([2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0])
    assert float(eps.sum()) == 0.0
    g = g0 + eps
    val_mean = crit_penalty(g, g0, mode="mean")
    assert float(val_mean) < 1e-10, f"mean mode must ignore zero-mean per-item motion, got {float(val_mean)}"
    # The per-item ablation charges for exactly this motion.
    base_item = crit_penalty(g0.clone(), g0, mode="item", delta=1.0)
    moved_item = crit_penalty(g, g0, mode="item", delta=1.0)
    assert float(moved_item) > float(base_item) + 0.1, (
        f"item mode must penalise per-item motion: {float(base_item)} -> {float(moved_item)}")
    # softplus(|2|-1) per item vs softplus(-1): exact values.
    want_moved = float(F.softplus(torch.tensor(1.0)))
    want_base = float(F.softplus(torch.tensor(-1.0)))
    assert abs(float(moved_item) - want_moved) < 1e-6, (float(moved_item), want_moved)
    assert abs(float(base_item) - want_base) < 1e-6, (float(base_item), want_base)


def test_mean_equals_b_squared_under_additive_bias():
    if not HAVE_DEPS:
        return _skip("test_mean_equals_b_squared_under_additive_bias")
    # Calibration theory Claim 1: a CL stage is a class-independent shift b of
    # g. The penalty must read that shift exactly, whatever items were drawn.
    g0 = torch.tensor(G0)
    for b in (0.5, -1.25, 3.0):
        for idx in ([0, 1, 2], [3, 5, 7, 1], list(range(8))):
            sub = g0[idx]
            val = crit_penalty(sub + b, sub, mode="mean")
            assert abs(float(val) - b * b) < 1e-4, (b, idx, float(val))
            # meanvar adds nothing under a pure shift (std unchanged).
            val2 = crit_penalty(sub + b, sub, mode="meanvar")
            assert abs(float(val2) - b * b) < 1e-4, (b, idx, float(val2))


def test_mean_gradient_item_invariant_and_restoring():
    if not HAVE_DEPS:
        return _skip("test_mean_gradient_item_invariant_and_restoring")
    g0 = torch.tensor(G0)
    # Heterogeneous per-item deviations; only their mean should matter.
    d = torch.tensor([3.0, -1.0, 0.0, 2.0, -0.5, 1.5, 4.0, -1.0])
    g = (g0 + d).requires_grad_(True)
    val = crit_penalty(g, g0, mode="mean")
    val.backward()
    k = g.numel()
    expect = 2.0 * float(d.mean()) / k
    assert torch.allclose(g.grad, torch.full((k,), expect), atol=1e-6), (
        f"mean-mode gradient must be identical across items ({expect}), got {g.grad}")
    # Restoring: the batch mean drifted UP (+1.0), so every gradient is > 0
    # (descent pushes the common shift back down), including on items whose
    # OWN deviation is negative or zero -- no per-item pull.
    assert float(d.mean()) > 0 and torch.all(g.grad > 0), g.grad


def test_item_gradient_depends_on_own_deviation():
    if not HAVE_DEPS:
        return _skip("test_item_gradient_depends_on_own_deviation")
    g0 = torch.tensor(G0)
    d = torch.tensor([3.0, -3.0, 0.0, 2.0, -0.5, 1.5, 4.0, -1.0])
    g = (g0 + d).requires_grad_(True)
    val = crit_penalty(g, g0, mode="item", delta=1.0)
    val.backward()
    # Per-item: sign of the gradient follows the sign of the item's own deviation.
    assert float(g.grad[0]) > 0 and float(g.grad[1]) < 0, g.grad
    assert float(g.grad[2]) == 0.0, g.grad  # |d|=0: abs has zero subgradient at 0
    assert not torch.allclose(g.grad, torch.full_like(g.grad, float(g.grad[0]))), (
        "item-mode gradient must NOT be item-invariant")


def test_meanvar_zero_under_permutation_positive_under_dispersion_change():
    if not HAVE_DEPS:
        return _skip("test_meanvar_zero_under_permutation_positive_under_dispersion_change")
    g0 = torch.tensor(G0)
    perm = torch.tensor([7, 0, 6, 1, 5, 2, 4, 3])
    g = g0[perm]
    assert torch.any(g != g0)  # every position changed
    val = crit_penalty(g, g0, mode="meanvar")
    assert float(val) < 1e-8, f"permutation preserves both moments, got {float(val)}"
    assert float(crit_penalty(g, g0, mode="item", delta=0.0)) > 0.5  # item mode objects
    # Dispersion doubled around the same mean: mean-term 0, std-term > 0.
    m0 = g0.mean()
    g_wide = m0 + 2.0 * (g0 - m0)
    assert abs(float(crit_penalty(g_wide, g0, mode="mean"))) < 1e-8
    val_w = crit_penalty(g_wide, g0, mode="meanvar")
    s0 = float(g0.std(unbiased=False))
    assert abs(float(val_w) - s0 * s0) < 1e-4, (float(val_w), s0 * s0)  # (2s-s)^2


def test_meanvar_gradient_flows():
    if not HAVE_DEPS:
        return _skip("test_meanvar_gradient_flows")
    g0 = torch.tensor(G0)
    g = (g0 * 1.5 + 0.3).requires_grad_(True)
    val = crit_penalty(g, g0, mode="meanvar")
    val.backward()
    assert g.grad is not None and torch.any(g.grad != 0), g.grad


def test_item_dead_zone_monotone_in_delta():
    if not HAVE_DEPS:
        return _skip("test_item_dead_zone_monotone_in_delta")
    g0 = torch.tensor(G0)
    g = g0 + torch.tensor([1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.0, 0.0])
    vals = [float(crit_penalty(g, g0, mode="item", delta=dl)) for dl in (0.0, 0.5, 1.0, 3.0)]
    assert vals == sorted(vals, reverse=True) and vals[0] > vals[-1], vals


def test_population_moments_single_item():
    if not HAVE_DEPS:
        return _skip("test_population_moments_single_item")
    m, s = population_moments(torch.tensor([4.0]))
    assert float(m) == 4.0 and float(s) == 0.0 and torch.isfinite(s)
    m, s = population_moments(torch.tensor([1.0, 3.0]))
    assert float(m) == 2.0 and float(s) == 1.0  # population std (ddof=0)


def test_bad_inputs_raise():
    if not HAVE_DEPS:
        return _skip("test_bad_inputs_raise")
    g0 = torch.tensor(G0)
    for bad in ("kd", "MEAN", ""):
        try:
            crit_penalty(g0, g0, mode=bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"mode {bad!r} must be rejected")
    try:
        crit_penalty(g0[:3], g0, mode="mean")
    except AssertionError:
        pass
    else:
        raise AssertionError("live/base shape mismatch must be rejected")
    assert set(MODES) == {"mean", "meanvar", "item"}


def test_base_stats_cache_roundtrip_and_stale_rejection():
    if not HAVE_DEPS:
        return _skip("test_base_stats_cache_roundtrip_and_stale_rejection")
    g0 = {"probe_1": 2.5, "probe_2": -3.0, "probe_3": 0.5}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "nested", "probe_base_stats.json")
        save_base_stats(path, g0, {"source": "unit", "mode": "mean"})
        back, payload = load_base_stats(path, ["probe_3", "probe_1", "probe_2"])
        assert back == g0, back
        assert payload["source"] == "unit" and payload["n"] == 3
        assert abs(payload["mean"] - 0.0) < 1e-12, payload["mean"]
        raw = json.load(open(path))
        assert set(raw["g0"]) == set(g0)
        # A cache for a different probe set must fail loudly.
        try:
            load_base_stats(path, ["probe_1", "probe_2"])
        except AssertionError:
            pass
        else:
            raise AssertionError("stale/oversized cache must be rejected")
        try:
            load_base_stats(path, ["probe_1", "probe_2", "probe_3", "probe_4"])
        except AssertionError:
            pass
        else:
            raise AssertionError("incomplete cache must be rejected")


TESTS = [
    test_imports_stay_light,
    test_decision_stat_fp32,
    test_mean_zero_when_paired_equal,
    test_mean_ignores_zero_mean_per_item_perturbation_item_does_not,
    test_mean_equals_b_squared_under_additive_bias,
    test_mean_gradient_item_invariant_and_restoring,
    test_item_gradient_depends_on_own_deviation,
    test_meanvar_zero_under_permutation_positive_under_dispersion_change,
    test_meanvar_gradient_flows,
    test_item_dead_zone_monotone_in_delta,
    test_population_moments_single_item,
    test_bad_inputs_raise,
    test_base_stats_cache_roundtrip_and_stale_rejection,
]


def main():
    if not HAVE_DEPS:
        print("SKIP: torch/PIL not installed; crit-preserve math tests need them. "
              "The math is validated end-to-end by the cluster smoke gate.")
        return
    for t in TESTS:
        t()
    print(f"CRIT-PRESERVE OK: {len(TESTS)} checks passed "
          "(mean mode 0 under zero-mean per-item motion / == b^2 under a shift / "
          "item-invariant restoring gradient; item ablation per-item; meanvar 0 under "
          "permutation; cache round-trip + stale rejection).")


if __name__ == "__main__":
    main()
