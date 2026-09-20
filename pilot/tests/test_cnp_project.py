"""CPU unit tests for the criterion-null-space projection math and wiring
(method/cnp_project.py, train_lora.py --cnp_* flags, fullstudy/run_arm.py arms).

Self-contained and CPU-only, like tests/test_crit_preserve.py: no GPU, no model
download, no forward pass, no transformers/peft. The class is exercised
end-to-end by the cluster smoke gate (design_notes/method_cnp.md sec 7); here we
pin the *rule* on synthetic parameter/gradient tensors:

  * two-sided projection removes EXACTLY the u-component (<G',u> = 0,
    G' = G - <G,u>/<u,u> u, removed ratio = |cos(G,u)|), written back in place;
  * one-sided: a gradient whose descent step moves the criterion TOWARD its base
    value is left byte-identical; one that pushes it away is projected -- checked
    by actually taking the step on a toy quadratic and reading |s - s0|; the dead
    band makes the rule two-sided at base;
  * two directions: the result equals N (N^T N)^-1 N^T (least squares) and the
    Gram-Schmidt/second-pass logic activates a direction whose motion flips after
    the first removal (and only then); collinear directions are flagged degenerate
    and still give the right projection;
  * norm-ratio logging fields, None/zero gradients, bf16 write-back dtype, the
    list-of-tensors vector equals the flat vector;
  * refusal_stat is the refusal log-odds and is shift-invariant;
  * step() cadence (refresh at 0, T, 2T, ...), history length, in-place grads on a
    tiny module, via a stub whose direction estimate is replaced;
  * train_lora flags default OFF (byte-identical arms) and crit_term_active's
    semantics; run_arm.py exposes cnp / cnp_ref + CNP_T / CNP_K and keeps its
    pinned MICRO_BATCH / FAITH_PAIRS / critp / policy lines.

    python3 pilot/tests/test_cnp_project.py
"""
import contextlib
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
sys.path.insert(0, PILOT)

_HEAVY = ("transformers", "peft")
_PRELOADED = {m for m in _HEAVY if m in sys.modules}

try:
    import torch
    from method.cnp_project import (
        CRIT,
        REF,
        CriterionNullSpace,
        _dropout_off,
        away_from_base,
        gram_matrix,
        project_grads,
        refusal_stat,
        solve_projection_coefficients,
        vaxpy_,
        vdot,
        vnorm,
    )
    import train_lora
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


def _skip(name):
    print(f"SKIP {name}: torch/PIL (cnp_project deps) not installed")


def _flat(vec):
    return torch.cat([t.reshape(-1).double() for t in vec if t is not None])


def _split(flat, shapes):
    out, i = [], 0
    for s in shapes:
        n = int(math.prod(s)) if s else 1
        out.append(flat[i:i + n].reshape(s).clone())
        i += n
    assert i == flat.numel()
    return out


SHAPES = [(2, 3), (4,), (1, 1, 2)]   # a 12-dim "parameter vector" in three tensors


def test_imports_stay_light():
    if not HAVE_DEPS:
        return _skip("test_imports_stay_light")
    leaked = [m for m in _HEAVY if m in sys.modules and m not in _PRELOADED]
    assert not leaked, (
        f"importing cnp_project/train_lora pulled in heavy modules {leaked}; the CPU "
        "projection-math test must run without transformers/peft")


# ---------------------------------------------------------------------------
# Vector helpers.
# ---------------------------------------------------------------------------

def test_vdot_list_vector_equals_flat_and_skips_none():
    if not HAVE_DEPS:
        return _skip("test_vdot_list_vector_equals_flat_and_skips_none")
    g = torch.Generator().manual_seed(0)
    a = _split(torch.randn(12, generator=g, dtype=torch.float64), SHAPES)
    b = _split(torch.randn(12, generator=g, dtype=torch.float64), SHAPES)
    want = float(_flat(a) @ _flat(b))
    assert abs(vdot(a, b) - want) < 1e-5, (vdot(a, b), want)
    assert abs(vnorm(a) - float(_flat(a).norm())) < 1e-5
    # None = zero contribution.
    a2 = [a[0], None, a[2]]
    want2 = float(_flat([a[0]]) @ _flat([b[0]]) + _flat([a[2]]) @ _flat([b[2]]))
    assert abs(vdot(a2, b) - want2) < 1e-5
    assert vdot([None, None, None], b) == 0.0
    # axpy in place, None skipped.
    y = [t.clone() if t is not None else None for t in a2]
    vaxpy_(y, -2.0, b)
    assert y[1] is None
    assert torch.allclose(y[0], a[0] - 2.0 * b[0]) and torch.allclose(y[2], a[2] - 2.0 * b[2])
    M = gram_matrix([a, b])
    assert M.dtype == torch.float64 and M.shape == (2, 2)
    assert abs(float(M[0, 1]) - want) < 1e-5 and abs(float(M[1, 0]) - want) < 1e-5


# ---------------------------------------------------------------------------
# The rule.
# ---------------------------------------------------------------------------

def test_away_from_base_truth_table():
    if not HAVE_DEPS:
        return _skip("test_away_from_base_truth_table")
    # s(theta - eta G) - s0 = disp - eta <G,u>: |disp| grows iff <G,u>*disp < 0.
    assert away_from_base(dot_gu=-1.0, disp=+2.0) is True     # above base, step raises s
    assert away_from_base(dot_gu=+1.0, disp=+2.0) is False    # above base, step lowers s
    assert away_from_base(dot_gu=+1.0, disp=-2.0) is True     # below base, step lowers s
    assert away_from_base(dot_gu=-1.0, disp=-2.0) is False    # below base, step raises s
    assert away_from_base(dot_gu=+1.0, disp=0.0) is True      # exactly at base: any motion is away
    assert away_from_base(dot_gu=0.0, disp=+2.0) is False     # no first-order motion
    assert away_from_base(dot_gu=+1.0, disp=+0.05, deadband=0.1) is True   # inside dead band
    assert away_from_base(dot_gu=+1.0, disp=+0.5, deadband=0.1) is False   # outside it


def test_two_sided_removes_exactly_u_component_in_place():
    if not HAVE_DEPS:
        return _skip("test_two_sided_removes_exactly_u_component_in_place")
    g = torch.Generator().manual_seed(1)
    G0 = torch.randn(12, generator=g, dtype=torch.float64)
    u0 = torch.randn(12, generator=g, dtype=torch.float64)
    grads = _split(G0.float(), SHAPES)
    u = _split(u0.float(), SHAPES)
    ids = [id(t) for t in grads]
    for disp in (+3.0, -3.0, 0.0):   # two-sided ignores the sign entirely
        grads = _split(G0.float(), SHAPES)
        info = project_grads(grads, {CRIT: u}, {CRIT: disp}, one_sided=False)
        want = G0 - (G0 @ u0) / (u0 @ u0) * u0
        got = _flat(grads)
        assert torch.allclose(got, want, atol=1e-5), (got, want)
        assert abs(vdot(grads, u)) < 1e-4 * float(u0.norm()) * float(G0.norm()), vdot(grads, u)
        assert info["active"] == [CRIT] and abs(info["dot_after"][CRIT]) < 1e-6
        cos = abs(float(G0 @ u0) / (G0.norm() * u0.norm()))
        assert abs(info["removed_ratio"] - cos) < 1e-5, (info["removed_ratio"], cos)
        assert abs(info["removed_norm"] - float((G0 - want).norm())) < 1e-5
        assert abs(info["grad_norm"] - float(G0.norm())) < 1e-5
        assert abs(info["alpha"][CRIT] - float((G0 @ u0) / (u0 @ u0))) < 1e-6
        assert info["degenerate"] is False
    # In place: the SAME tensors were modified (param.grad write-back contract).
    grads = _split(G0.float(), SHAPES)
    ids = [id(t) for t in grads]
    project_grads(grads, {CRIT: u}, {CRIT: 1.0}, one_sided=False)
    assert [id(t) for t in grads] == ids


def test_one_sided_leaves_toward_steps_untouched_and_projects_away_steps():
    if not HAVE_DEPS:
        return _skip("test_one_sided_leaves_toward_steps_untouched_and_projects_away_steps")
    # Toy: theta in R^3, criterion s(theta) = <u, theta> with u = e0, base s0 = 0.
    u = [torch.tensor([1.0, 0.0, 0.0])]
    eta = 0.1
    for disp in (+2.0, -2.0):
        theta = torch.tensor([disp, 5.0, -1.0])           # s(theta) - s0 = disp
        # (a) descent step moves s TOWARD base: G's u-component has the sign of disp.
        G_toward = torch.tensor([math.copysign(1.0, disp), 1.0, -0.5])
        grads = [G_toward.clone()]
        info = project_grads(grads, {CRIT: u}, {CRIT: disp}, one_sided=True)
        assert torch.equal(grads[0], G_toward), "toward-base gradient must be untouched"
        assert info["active"] == [] and info["removed_ratio"] == 0.0 and info["alpha"] == {}
        assert abs(float(theta[0] - eta * grads[0][0])) < abs(disp)   # |s - s0| shrank
        # (b) descent step moves s AWAY: projected, first-order motion of s is 0.
        G_away = torch.tensor([-math.copysign(1.0, disp), 1.0, -0.5])
        grads = [G_away.clone()]
        info = project_grads(grads, {CRIT: u}, {CRIT: disp}, one_sided=True)
        assert info["active"] == [CRIT]
        assert torch.allclose(grads[0], torch.tensor([0.0, 1.0, -0.5]), atol=1e-7), grads[0]
        assert abs(float(theta[0] - eta * grads[0][0]) - disp) < 1e-7  # s unchanged
        assert abs(info["removed_ratio"] - 1.0 / math.sqrt(1 + 1 + 0.25)) < 1e-6
        # Two-sided would have projected case (a) too.
        grads = [G_toward.clone()]
        project_grads(grads, {CRIT: u}, {CRIT: disp}, one_sided=False)
        assert float(grads[0][0]) == 0.0


def test_deadband_makes_rule_two_sided_at_base():
    if not HAVE_DEPS:
        return _skip("test_deadband_makes_rule_two_sided_at_base")
    u = [torch.tensor([1.0, 0.0])]
    G = torch.tensor([1.0, 1.0])
    # disp exactly 0 (stage-1 start): any motion is away -> projected even with deadband 0.
    grads = [G.clone()]
    info = project_grads(grads, {CRIT: u}, {CRIT: 0.0}, one_sided=True, deadband=0.0)
    assert info["active"] == [CRIT] and float(grads[0][0]) == 0.0
    # Small positive disp, toward-base gradient: untouched without a dead band...
    grads = [G.clone()]
    info = project_grads(grads, {CRIT: u}, {CRIT: 0.05}, one_sided=True, deadband=0.0)
    assert info["active"] == [] and torch.equal(grads[0], G)
    # ...projected inside the dead band.
    grads = [G.clone()]
    info = project_grads(grads, {CRIT: u}, {CRIT: 0.05}, one_sided=True, deadband=0.1)
    assert info["active"] == [CRIT] and float(grads[0][0]) == 0.0
    try:
        away_from_base(1.0, 1.0, deadband=-0.1)
    except AssertionError:
        pass
    else:
        raise AssertionError("negative dead band must be rejected")


def test_two_directions_equal_normal_equations_and_collinear_is_degenerate():
    if not HAVE_DEPS:
        return _skip("test_two_directions_equal_normal_equations_and_collinear_is_degenerate")
    g = torch.Generator().manual_seed(2)
    G0 = torch.randn(12, generator=g, dtype=torch.float64)
    u0 = torch.randn(12, generator=g, dtype=torch.float64)
    n0 = torch.randn(12, generator=g, dtype=torch.float64) * 0.01   # badly scaled vs u
    N = torch.stack([u0, n0], dim=1)                                 # (12, 2)
    beta = torch.linalg.lstsq(N, G0.unsqueeze(1)).solution.squeeze(1)
    want = G0 - N @ beta                                             # G - N (N^T N)^-1 N^T G
    grads = _split(G0.float(), SHAPES)
    info = project_grads(grads, {CRIT: _split(u0.float(), SHAPES), REF: _split(n0.float(), SHAPES)},
                         {CRIT: 1.0, REF: 1.0}, one_sided=False)
    got = _flat(grads)
    assert torch.allclose(got, want, atol=1e-4), (got - want).abs().max()
    assert abs(vdot(grads, _split(u0.float(), SHAPES))) < 1e-3
    assert abs(vdot(grads, _split(n0.float(), SHAPES))) < 1e-5
    assert info["active"] == [CRIT, REF] and info["degenerate"] is False
    assert abs(info["alpha"][CRIT] - float(beta[0])) < 1e-3 and abs(info["alpha"][REF] - float(beta[1])) < 1e-1 * abs(float(beta[1])) + 1e-3
    assert abs(info["removed_norm"] - float((N @ beta).norm())) < 1e-4
    # Collinear second direction: flagged, projection == single-direction projection.
    grads = _split(G0.float(), SHAPES)
    info = project_grads(grads, {CRIT: _split(u0.float(), SHAPES), REF: _split((2.0 * u0).float(), SHAPES)},
                         {CRIT: 1.0, REF: 1.0}, one_sided=False)
    want1 = G0 - (G0 @ u0) / (u0 @ u0) * u0
    assert torch.allclose(_flat(grads), want1, atol=1e-4)
    assert info["degenerate"] is True and info["active"] == [CRIT, REF]
    # Zero direction: coefficients 0, flagged, other direction still removed.
    zero = [torch.zeros(s) for s in SHAPES]
    grads = _split(G0.float(), SHAPES)
    info = project_grads(grads, {CRIT: _split(u0.float(), SHAPES), REF: zero},
                         {CRIT: 1.0, REF: 1.0}, one_sided=False)
    assert torch.allclose(_flat(grads), want1, atol=1e-4)
    assert info["degenerate"] is True and info["alpha"][REF] == 0.0
    alpha, degen = solve_projection_coefficients(torch.tensor([[4.0, 0.0], [0.0, 0.0]]),
                                                 torch.tensor([2.0, 1.0]))
    assert degen and abs(float(alpha[0]) - 0.5) < 1e-12 and float(alpha[1]) == 0.0


def test_one_sided_second_pass_activates_flipped_direction_only_when_it_flips():
    if not HAVE_DEPS:
        return _skip("test_one_sided_second_pass_activates_flipped_direction_only_when_it_flips")
    u = [torch.tensor([1.0, 0.0, 0.0])]
    n = [torch.tensor([1.0, 1.0, 0.0]) / math.sqrt(2.0)]
    disps = {CRIT: +1.0, REF: -1.0}
    # Case A: crit initially TOWARD (<G,u> = 0.2 > 0 with disp +1); ref AWAY
    # (<G,n> > 0 with disp -1). Removing n turns <G',u> negative -> crit flips to
    # away -> second pass removes it too -> G'' orthogonal to span{e0, e1}.
    grads = [torch.tensor([0.2, 1.0, 1.0])]
    info = project_grads(grads, {CRIT: u, REF: n}, disps, one_sided=True)
    assert info["active"] == [CRIT, REF], info["active"]
    assert torch.allclose(grads[0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6), grads[0]
    assert abs(info["dot_after"][CRIT]) < 1e-9 and abs(info["dot_after"][REF]) < 1e-9
    assert abs(info["dot_before"][CRIT] - 0.2) < 1e-6
    assert abs(info["removed_ratio"] - math.sqrt(0.04 + 1.0) / math.sqrt(0.04 + 2.0)) < 1e-6
    # Case B: same directions/disps, G = (1.5, 1, 1): ref away, crit toward, and
    # after removing n the u-motion is still toward (0.25 > 0) -> crit stays
    # inactive, G' = (0.25, -0.25, 1).
    grads = [torch.tensor([1.5, 1.0, 1.0])]
    info = project_grads(grads, {CRIT: u, REF: n}, disps, one_sided=True)
    assert info["active"] == [REF], info["active"]
    assert torch.allclose(grads[0], torch.tensor([0.25, -0.25, 1.0]), atol=1e-6), grads[0]
    assert abs(info["dot_after"][CRIT] - 0.25) < 1e-6 and abs(info["dot_after"][REF]) < 1e-9
    assert not away_from_base(info["dot_after"][CRIT], disps[CRIT])
    # Case C: nothing away -> untouched, both inactive.
    G = torch.tensor([1.0, -3.0, 1.0])   # <G,u>=1>0 (toward, disp +1); <G,n><0 (toward, disp -1)
    grads = [G.clone()]
    info = project_grads(grads, {CRIT: u, REF: n}, disps, one_sided=True)
    assert info["active"] == [] and torch.equal(grads[0], G) and info["removed_ratio"] == 0.0


def test_none_and_zero_gradients_are_safe():
    if not HAVE_DEPS:
        return _skip("test_none_and_zero_gradients_are_safe")
    u = [torch.ones(2, 3), torch.ones(4), torch.ones(1, 1, 2)]
    # A parameter without gradient (None) is skipped, the others projected.
    grads = [torch.ones(2, 3), None, torch.ones(1, 1, 2)]
    info = project_grads(grads, {CRIT: u}, {CRIT: 0.0}, one_sided=True)
    assert grads[1] is None
    # <G,u> over the 8 live entries = 8; <u,u> over all 12 = 12; alpha = 2/3.
    assert abs(info["alpha"][CRIT] - 8.0 / 12.0) < 1e-6
    assert torch.allclose(grads[0], torch.full((2, 3), 1.0 - 8.0 / 12.0))
    assert 0.0 <= info["removed_ratio"] <= 1.0
    # All-zero gradient: no division by zero, ratio 0, finite fields.
    grads = [torch.zeros(2, 3), torch.zeros(4), torch.zeros(1, 1, 2)]
    info = project_grads(grads, {CRIT: u}, {CRIT: 0.0}, one_sided=False)
    assert info["removed_ratio"] == 0.0 and info["grad_norm"] == 0.0
    assert all(math.isfinite(v) for v in (info["removed_norm"], info["alpha"][CRIT]))
    assert all(float(t.abs().sum()) == 0.0 for t in grads)


def test_bf16_grad_writeback_keeps_dtype():
    if not HAVE_DEPS:
        return _skip("test_bf16_grad_writeback_keeps_dtype")
    G0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    u0 = torch.tensor([1.0, 0.0, 1.0, 0.0])
    grads = [G0.to(torch.bfloat16)]
    info = project_grads(grads, {CRIT: [u0]}, {CRIT: 0.0}, one_sided=False)
    assert grads[0].dtype == torch.bfloat16
    want = G0 - (G0 @ u0) / (u0 @ u0) * u0      # (-1, 2, 1, 4)
    assert torch.allclose(grads[0].float(), want, atol=0.05), (grads[0], want)
    assert info["active"] == [CRIT]


def test_refusal_stat_is_logodds_and_shift_invariant():
    if not HAVE_DEPS:
        return _skip("test_refusal_stat_is_logodds_and_shift_invariant")
    g = torch.Generator().manual_seed(3)
    z = torch.randn(4, 7, generator=g) * 3.0
    uid = 5
    r = refusal_stat(z, uid)
    assert r.shape == (4,) and r.dtype == torch.float32
    lp = torch.log_softmax(z, dim=-1)
    p = lp[:, uid].exp()
    want = torch.log(p) - torch.log1p(-p)
    assert torch.allclose(r, want, atol=1e-4), (r, want)
    assert torch.allclose(refusal_stat(z + 11.0, uid), r, atol=1e-4)   # per-row shift cancels
    assert torch.allclose(refusal_stat(z.to(torch.bfloat16), uid), r, atol=0.2)
    # Raising only the refusal logit raises the statistic on that row alone.
    z2 = z.clone()
    z2[1, uid] += 2.0
    r2 = refusal_stat(z2, uid)
    assert float(r2[1]) > float(r[1]) and torch.allclose(r2[[0, 2, 3]], r[[0, 2, 3]])
    for bad in (-1, 7):
        try:
            refusal_stat(z, bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"out-of-range token id {bad} must be rejected")


def test_dropout_off_context_restores_state():
    if not HAVE_DEPS:
        return _skip("test_dropout_off_context_restores_state")
    m = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5), torch.nn.Dropout(0.1))
    m.train()
    m[2].eval()   # one already off: must be restored to eval, not train
    with _dropout_off(m):
        assert m.training is True and m[0].training is True    # model/train-mode untouched
        assert m[1].training is False and m[2].training is False
    assert m[1].training is True and m[2].training is False


# ---------------------------------------------------------------------------
# step() cadence and in-place grads on a tiny module (stub direction estimate).
# ---------------------------------------------------------------------------

class StubCNP(CriterionNullSpace):
    """CriterionNullSpace with the model-forward direction estimate replaced by
    a fixed direction; built via __new__ (no preserver/processor)."""

    def refresh(self, model, global_step):
        params = self.bind(model)
        self.dirs = {CRIT: [torch.zeros_like(p) for p in params]}
        self.dirs[CRIT][0][0, 0] = 1.0               # u = e_(first coord of param 0)
        self.disps = {CRIT: self.disp_schedule[global_step]}
        self.gram = gram_matrix([self.dirs[CRIT]])
        self.last_refresh_step = int(global_step)
        self.last_refresh_stats = {CRIT: {"s_live": 0.0, "s_base": 0.0, "dir_norm": 1.0}}
        self.n_refresh += 1
        self.refresh_calls.append(int(global_step))


def _stub(T, disp_schedule):
    c = StubCNP.__new__(StubCNP)
    c.T = T
    c.one_sided = True
    c.deadband = 0.0
    c.params = None
    c.param_names = None
    c.dirs, c.disps, c.gram = {}, {}, None
    c.last_refresh_step = None
    c.last_refresh_stats = None
    c.n_refresh = 0
    c.history = []
    c.refresh_calls = []
    c.disp_schedule = disp_schedule
    return c


def test_step_cadence_history_and_inplace_param_grads():
    if not HAVE_DEPS:
        return _skip("test_step_cadence_history_and_inplace_param_grads")
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    for p in model[1].parameters():
        p.requires_grad_(False)          # frozen params must not be part of the vector
    T = 3
    disp = {0: +1.0, 3: +1.0, 6: -1.0}   # refresh steps -> displacement then in force
    c = _stub(T, disp)
    for step in range(8):
        for p in model.parameters():
            p.grad = None
        loss = model(torch.randn(5, 3)).pow(2).mean()
        loss.backward()
        w = model[0].weight
        w.grad[0, 0] = -1.0 if step < 6 else +1.0   # with disp +1 / -1: always AWAY -> projected
        gid = id(w.grad)
        raw = [p.grad.clone() for p in model.parameters() if p.requires_grad]
        info = c.step(model, step)
        assert id(w.grad) == gid, "grad must be projected in place, not replaced"
        assert float(w.grad[0, 0]) == 0.0, (step, w.grad[0, 0])
        assert info["active"] == [CRIT] and info["step"] == step
        # Every other coordinate untouched (u is a single coordinate).
        live = [p.grad for p in model.parameters() if p.requires_grad]
        assert torch.equal(live[1], raw[1])
        assert torch.equal(live[0][1:], raw[0][1:]) and torch.equal(live[0][0, 1:], raw[0][0, 1:])
        assert info["disp"][CRIT] == disp[(step // T) * T], (step, info["disp"])
        assert info["refresh"] is (step % T == 0)
        if info["refresh"]:
            assert "refresh_stats" in info and abs(info["dot_after_audit"][CRIT]) < 1e-9
    assert c.refresh_calls == [0, 3, 6] and c.n_refresh == 3
    assert len(c.history) == 8
    assert c.param_names == ["0.weight", "0.bias"] and len(c.params) == 2
    # A toward-base gradient (disp -1 in force at step 6+, <G,u> < 0) is untouched.
    for p in model.parameters():
        p.grad = None
    model(torch.randn(5, 3)).pow(2).mean().backward()
    w.grad[0, 0] = -0.75      # exactly representable in fp32 (-0.7 is not)
    info = c.step(model, 8)
    assert info["active"] == [], info["active"]
    assert float(w.grad[0, 0]) == -0.75 and info["refresh"] is False, (w.grad[0, 0], info)
    # Calling without gradients is a wiring error, never a silent no-op -- and it
    # must fail BEFORE any refresh forward is spent (step 9 would refresh; the
    # stub schedule has no entry for it, so a refresh-first order would KeyError
    # instead of hitting the wiring assert).
    for p in model.parameters():
        p.grad = None
    n_ref = c.n_refresh
    try:
        c.step(model, 9)
    except AssertionError:
        pass
    else:
        raise AssertionError("step() without grads must fail loudly")
    assert c.n_refresh == n_ref and c.refresh_calls == [0, 3, 6]


# ---------------------------------------------------------------------------
# The REAL refresh path (_direction / refresh / pretrain_check / step) on a fake
# preserver whose "forward" reads the statistic off a tiny model's parameters,
# so u has a closed form: g_i = w.x_i + b  =>  grad_(w,b) mean_S g = (mean_S x, 1).
# ---------------------------------------------------------------------------

class FakeCrit:
    def __init__(self, n_rows=6):
        self.rows = [{"id": f"p{i}", "image": f"img{i}", "object": f"obj{i}"} for i in range(n_rows)]
        self.device = torch.device("cpu")
        g = torch.Generator().manual_seed(5)
        self.X = torch.randn(n_rows, 3, generator=g)
        self.g0_by_id = None

    def _g_chunk(self, model, rows):
        idx = [int(img[3:]) for img, _ in rows]
        return model(self.X[idx]).squeeze(-1).float()


def _real(crit, T=2, k=4):
    import random as _random
    c = CriterionNullSpace.__new__(CriterionNullSpace)
    c.crit = crit
    c.T, c.k = T, k
    c.one_sided, c.deadband = True, 0.0
    c.use_ref, c.dir_dtype = False, torch.float32
    c.rng = _random.Random(17 + 3)
    c.device = crit.device
    c.unans_id, c.ref_rows, c.ref_derived = None, [], False
    c.ref0_by_id, c.ref_base_source = None, None
    c.params = c.param_names = None
    c.dirs, c.disps, c.gram = {}, {}, None
    c.last_refresh_step = c.last_refresh_stats = None
    c.n_refresh, c.history = 0, []
    return c


def test_real_refresh_path_matches_analytic_gradient():
    if not HAVE_DEPS:
        return _skip("test_real_refresh_path_matches_analytic_gradient")
    torch.manual_seed(1)
    # Dropout in front: in train mode the statistic would be stochastic unless
    # the estimate switches dropout off (it must; ewc.compute_fisher convention).
    model = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(3, 1))
    model.train()
    crit = FakeCrit()
    with torch.no_grad():
        model.eval()
        g0 = crit._g_chunk(model, [(r["image"], r["object"]) for r in crit.rows])
        model.train()
    crit.g0_by_id = {r["id"]: float(v) for r, v in zip(crit.rows, g0)}
    c = _real(crit)
    # Stage-1 gate passes at theta == theta0 (train mode restored afterwards).
    d0 = c.pretrain_check(model, is_stage1=True)
    assert abs(d0[CRIT]) < 1e-5 and model.training is True, d0
    # Perturb w by +0.5 on input 0 -> disp = 0.5 * mean_S x_0 exactly.
    with torch.no_grad():
        model[1].weight[0, 0] += 0.5
    # Task gradient, then the hook.
    for p in model.parameters():
        p.grad = None
    model[1](torch.randn(4, 3)).pow(2).mean().backward()
    raw = [p.grad.clone() for p in model.parameters()]
    info = c.step(model, 0)
    assert info["refresh"] is True and c.n_refresh == 1 and model.training is True
    assert model[0].training is True   # dropout module restored to train after the estimate
    # Which items were sampled: recover S from the same RNG stream.
    import random as _random
    idx = _random.Random(17 + 3).sample(range(len(crit.rows)), c.k)
    xS = crit.X[idx]
    u = c.dirs[CRIT]
    assert c.param_names == ["1.weight", "1.bias"]
    assert torch.allclose(u[0], xS.mean(0, keepdim=True), atol=1e-6), (u[0], xS.mean(0))
    assert torch.allclose(u[1], torch.ones(1), atol=1e-6), u[1]
    want_disp = 0.5 * float(xS[:, 0].mean())
    assert abs(c.disps[CRIT] - want_disp) < 1e-5, (c.disps[CRIT], want_disp)
    assert abs(float(c.gram[0, 0]) - (float(xS.mean(0).pow(2).sum()) + 1.0)) < 1e-5
    st = info["refresh_stats"][CRIT]
    assert abs(st["s_live"] - st["s_base"] - want_disp) < 1e-5
    assert abs(st["dir_norm"] - math.sqrt(float(c.gram[0, 0]))) < 1e-6
    # The rule was applied to the real grads (in place) with the right sign.
    dot = float(sum((r * d).sum() for r, d in zip(raw, u)))
    expect_active = away_from_base(dot, want_disp)
    assert (info["active"] == [CRIT]) is expect_active, (info, dot, want_disp)
    live = [p.grad for p in model.parameters()]
    if expect_active:
        assert abs(info["dot_after_audit"][CRIT]) < 1e-6
        assert abs(float(sum((l * d).sum() for l, d in zip(live, u)))) < 1e-6
    else:
        assert all(torch.equal(l, r) for l, r in zip(live, raw))
    # Second call one step later: no refresh, same direction object reused.
    for p in model.parameters():
        p.grad = None
    model[1](torch.randn(4, 3)).pow(2).mean().backward()
    info2 = c.step(model, 1)
    assert info2["refresh"] is False and c.dirs[CRIT] is u and len(c.history) == 2
    # Stage-1 gate refuses a live/base mismatch > STAGE1_GATE; passes when not stage 1.
    with torch.no_grad():
        model[1].bias += 5.0
    try:
        c.pretrain_check(model, is_stage1=True)
    except AssertionError:
        pass
    else:
        raise AssertionError("stage-1 gate must reject a 5-logit displacement")
    d = c.pretrain_check(model, is_stage1=False)
    # pretrain_check samples with its OWN stream (SEED+4) so the step-0 sample is
    # unaffected by whether the check ran -> expected value on THAT sample.
    idx_chk = _random.Random(17 + 4).sample(range(len(crit.rows)), c.k)
    want_chk = 5.0 + 0.5 * float(crit.X[idx_chk][:, 0].mean())
    assert abs(d[CRIT] - want_chk) < 1e-4, (d, want_chk)


# ---------------------------------------------------------------------------
# train_lora flag surface + run_arm arms (source level).
# ---------------------------------------------------------------------------

REQUIRED = ["--data", "x.jsonl", "--data_root", "/r", "--out", "/o"]


def test_flags_default_off_and_crit_term_active():
    if not HAVE_DEPS:
        return _skip("test_flags_default_off_and_crit_term_active")
    args = train_lora.build_parser().parse_args(REQUIRED)
    assert args.cnp is False and args.cnp_T == 5 and args.cnp_k == 16, vars(args)
    assert args.cnp_refusal is False and args.cnp_two_sided is False, vars(args)
    assert args.cnp_deadband == 0.0 and args.cnp_ref_probe is None, vars(args)
    # Pre-existing surface untouched.
    assert args.crit_probe is None and args.crit_weight == 0.0 and args.crit_k == 8
    assert args.faith_pairs is None and args.pol_weight_abstain == 0.0
    # crit_term_active: unchanged pre-CNP semantics whenever --cnp is off.
    assert train_lora.crit_term_active(args) is False                         # no probe
    a = train_lora.build_parser().parse_args(REQUIRED + ["--crit_probe", "p.jsonl"])
    assert train_lora.crit_term_active(a) is True                             # weight 0 = monitor (old behaviour)
    a = train_lora.build_parser().parse_args(REQUIRED + ["--crit_probe", "p.jsonl", "--crit_weight", "0.1"])
    assert train_lora.crit_term_active(a) is True
    a = train_lora.build_parser().parse_args(REQUIRED + ["--crit_probe", "p.jsonl", "--cnp"])
    assert train_lora.crit_term_active(a) is False                            # projection-only arm
    a = train_lora.build_parser().parse_args(REQUIRED + ["--crit_probe", "p.jsonl", "--cnp",
                                                         "--crit_weight", "0.1"])
    assert train_lora.crit_term_active(a) is True                             # M1 + CNP compose
    a = train_lora.build_parser().parse_args(REQUIRED + [
        "--crit_probe", "p.jsonl", "--cnp", "--cnp_T", "1", "--cnp_k", "4", "--cnp_refusal",
        "--cnp_two_sided", "--cnp_deadband", "0.2", "--cnp_ref_probe", "r.jsonl"])
    assert a.cnp and a.cnp_T == 1 and a.cnp_k == 4 and a.cnp_refusal and a.cnp_two_sided
    assert a.cnp_deadband == 0.2 and a.cnp_ref_probe == "r.jsonl"
    try:
        with open(os.devnull, "w") as sink, contextlib.redirect_stderr(sink):
            train_lora.build_parser().parse_args(REQUIRED + ["--cnp_T", "five"])
    except SystemExit:
        pass
    else:
        raise AssertionError("--cnp_T five must be rejected")


def test_run_arm_exposes_cnp_arms_and_keeps_pinned_lines():
    src = open(os.path.join(PILOT, "..", "fullstudy", "run_arm.py")).read()
    for needle in ('args.arm == "cnp"', 'args.arm == "cnp_ref"', 'os.environ.get("CNP_T"',
                   'os.environ.get("CNP_K"', '"--cnp"', "--cnp_refusal", '"--crit_weight", "0"'):
        assert needle in src, f"run_arm.py missing {needle!r}"
    # Lines the integration must preserve verbatim (MICRO_BATCH / FAITH_PAIRS /
    # critp / policy).
    assert 'MICRO_BATCH = 1 if args.backbone == "qwen25vl" else 2' in src
    assert ('_FP = os.environ.get("FAITH_PAIRS", os.path.join(D, "grounding", '
            '"grounding_pairs.jsonl"))') in src
    for needle in ('args.arm == "critp"', 'args.arm == "anchorcrit"', 'os.environ.get("CRIT_WEIGHT"',
                   'args.arm == "policy"', 'args.arm == "policy_abst"', 'args.arm == "policy_eos"',
                   'os.environ.get("POL_W_ABST"', 'os.environ.get("POL_W_EOS"'):
        assert needle in src, f"run_arm.py lost {needle!r}"
    tl = open(os.path.join(PILOT, "train_lora.py")).read()
    assert "on_pre_optimizer_step" in tl and "crit_term_active(args)" in tl
    assert '"cnp_history"' in tl


TESTS = [
    test_imports_stay_light,
    test_vdot_list_vector_equals_flat_and_skips_none,
    test_away_from_base_truth_table,
    test_two_sided_removes_exactly_u_component_in_place,
    test_one_sided_leaves_toward_steps_untouched_and_projects_away_steps,
    test_deadband_makes_rule_two_sided_at_base,
    test_two_directions_equal_normal_equations_and_collinear_is_degenerate,
    test_one_sided_second_pass_activates_flipped_direction_only_when_it_flips,
    test_none_and_zero_gradients_are_safe,
    test_bf16_grad_writeback_keeps_dtype,
    test_refusal_stat_is_logodds_and_shift_invariant,
    test_dropout_off_context_restores_state,
    test_step_cadence_history_and_inplace_param_grads,
    test_real_refresh_path_matches_analytic_gradient,
    test_flags_default_off_and_crit_term_active,
    test_run_arm_exposes_cnp_arms_and_keeps_pinned_lines,
]


def main():
    if not HAVE_DEPS:
        print("SKIP: torch/PIL not installed; CNP math tests need them "
              "(the arm-driver source check still runs).")
        test_run_arm_exposes_cnp_arms_and_keeps_pinned_lines()
        return
    for t in TESTS:
        t()
    print(f"CNP-PROJECT OK: {len(TESTS)} checks passed "
          "(two-sided removes exactly the u-component in place; one-sided leaves "
          "toward-base steps untouched and zeroes first-order motion on away steps; "
          "dead band; two directions == N(N^T N)^-1 N^T with second-pass activation; "
          "collinear/zero/None/bf16 safe; refusal log-odds; step cadence + history; "
          "flags default off; run_arm arms + pinned lines).")


if __name__ == "__main__":
    main()
