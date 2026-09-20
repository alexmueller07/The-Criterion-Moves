"""Criterion-null-space update projection (method candidate CNP): the HARD-
constraint twin of the soft label-free criterion regularizer M1
(method/crit_preserve.py). design_notes/method_cnp.md is the design note;
design_notes/method_ideas_lit2.md sec 2.1 is the spec this implements.

The statistic is M1's: on the unlabeled probe P of (image, object) questions,

    g_i(theta) = z_yes,i - z_no,i          (fp32, answer position)
    s(theta)   = mean_{i in S} g_i(theta),  S subset of P, |S| = k'

and Claim 1 (method_calibration_theory.md) says a continual-tuning step moves the
criterion by <u_t, Delta_t> to first order, u_t := grad_theta s(theta_t) over the
TRAINABLE parameters (LoRA A/B + modules_to_save). M1 penalizes the displacement
(s - s0)^2 and has a weight knob; CNP removes the component of every task
gradient along u_t instead, so the criterion cannot move at first order and there
is no lambda:

    two-sided        G <- G - (<G,u>/<u,u>) u
    one-sided        the same, applied ONLY when a descent step -eta*G would push
    (default)        s further from its base value s0:
                         s(theta - eta G) - s0  =  (s - s0) - eta <u, G>
                     so |s - s0| grows iff <G,u> * (s - s0) < 0  (or |s - s0| is
                     inside the dead band, where any motion is "away").
                     Steps that move the criterion back toward s0 pass untouched
                     (estimation drift can be corrected, never accumulated).
    refusal handle   a second direction n_ref = grad_theta s_ref(theta),
    (optional)         s_ref = mean_S [ z_unans - LSE_{v != unans} z_v ]
                     (log-odds of the leakage token " Unanswerable" as the FIRST
                     answer token) on the leakage-prone open-VQA prompt format;
                     G is projected off span{u, n_ref} = N (N^T N)^-1 N^T G,
                     one-sided per direction with a second pass (removing one
                     direction can flip the other's first-order motion).

Where it acts: on the RAW accumulated task gradients (param.grad) after HF's
gradient clipping and immediately before optimizer.step() -- the
TrainerCallback.on_pre_optimizer_step event wired in train_lora.py. It is NOT a
loss. Adam's per-coordinate preconditioning then re-scales the projected
gradient, so the constraint is exact on G, approximate on the realised update
(Adam-NSCL projects the update instead; see the design note, risk R3). The
s(theta_t) logged at every refresh is the ground-truth monitor of whether the
criterion actually holds.

Memory/compute: u is estimated every T optimizer steps on k' probe items,
forwarded <= 3 images at a time (CriterionPreserver._g_chunk, verbatim), with
ONE torch.autograd.grad per chunk accumulated into a per-parameter fp32 buffer
(peak = one chunk graph; param.grad is never touched by the estimate). The
"flattened vector" is kept as a list of per-parameter tensors, so no extra
640 MB concatenation is ever materialised; per step the projection costs
|dirs| big dot products and |active| axpy's, and all the small linear algebra
(Gram matrix, coefficients, the one-sided second pass) is scalar float64 on CPU.
Dropout is switched off for the estimate (train mode kept so gradient
checkpointing stays active), as ewc.compute_fisher does.

The projection MATH (vdot / gram_matrix / away_from_base / project_grads /
refusal_stat) is pure tensor code unit-tested on CPU in
tests/test_cnp_project.py; the class wires it to model forwards and is
exercised end-to-end by the cluster smoke gate.
"""
import json
import math
import os
import random

import torch

from method.crit_preserve import (
    CHUNK_IMAGES,
    CriterionPreserver,
    load_base_stats,
    save_base_stats,
)
from method.faith_loss import QUESTION
from method.policy_anchor import ABSTAIN_SUFFIX, _first_token_in_context

SEED = 17
CRIT = "crit"    # direction name: yes/no criterion statistic s = mean g
REF = "ref"      # direction name: refusal log-odds statistic s_ref
REFUSAL_PHRASE = " Unanswerable"       # the pilot's measured leakage string
REFUSAL_BASE_CACHE = "probe_base_refusal.json"  # per-stage theta0 cache (audit + fallback)
STAGE1_GATE = 1.0   # |s(theta) - s(theta0)| bound at stage 1 (theta == theta0)


# ---------------------------------------------------------------------------
# Pure math on list-of-tensors "vectors" (the flattened trainable-parameter
# vector, stored per parameter). None entries are zeros. Unit-tested.
# ---------------------------------------------------------------------------

def vdot(a, b):
    """<a, b> as a python float, accumulated in fp32 per tensor and summed in
    fp64. `a`/`b` are lists of same-shaped tensors; a None entry contributes 0
    (a parameter with no gradient)."""
    assert len(a) == len(b), (len(a), len(b))
    acc = None
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        assert x.shape == y.shape, (tuple(x.shape), tuple(y.shape))
        t = torch.dot(x.reshape(-1).float(), y.reshape(-1).float()).double()
        acc = t if acc is None else acc + t
    return 0.0 if acc is None else float(acc)


def vnorm(a):
    return math.sqrt(max(0.0, vdot(a, a)))


def vaxpy_(y, alpha, x):
    """y <- y + alpha * x in place, per tensor; None entries of y are skipped
    (no gradient to project). Mixed dtypes (bf16 grad, fp32 direction) are
    combined in fp32 and cast back, so the direction is never bf16-rounded."""
    assert len(y) == len(x), (len(y), len(x))
    for yt, xt in zip(y, x):
        if yt is None or xt is None:
            continue
        if yt.dtype == xt.dtype:
            yt.add_(xt, alpha=alpha)
        else:
            yt.copy_((yt.float() + alpha * xt.float()).to(yt.dtype))


def gram_matrix(vecs):
    """(n, n) float64 CPU Gram matrix M_ij = <v_i, v_j> of list-of-tensors vectors."""
    n = len(vecs)
    M = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        for j in range(i, n):
            M[i, j] = M[j, i] = vdot(vecs[i], vecs[j])
    return M


def away_from_base(dot_gu, disp, deadband=0.0):
    """True iff a descent step -eta*G moves the statistic FURTHER from its base
    value to first order. With disp = s(theta) - s0 and dot_gu = <G, u>,

        s(theta - eta G) - s0 = disp - eta * dot_gu,

    so |disp| grows iff dot_gu * disp < 0. Inside the dead band (|disp| <=
    deadband, default 0 -> exactly at base) every nonzero motion is away, so the
    rule is two-sided there; that is also what makes the sign well defined at
    stage 1, where disp is bf16 noise around 0."""
    assert deadband >= 0.0, deadband
    if abs(disp) <= deadband:
        return True
    return dot_gu * disp < 0.0


def solve_projection_coefficients(M, d, rtol=1e-8):
    """alpha with sum_i alpha_i v_i = orthogonal projection of G onto span{v_i},
    given the Gram matrix M (n,n) and d_i = <G, v_i>: alpha = M^+ d, solved on
    the cosine-normalised Gram so a badly scaled pair (|u| >> |n_ref|) is not
    mistaken for a collinear one. Returns (alpha float64 (n,), degenerate) where
    degenerate flags a (near-)collinear or zero direction (pinv then yields the
    minimum-norm coefficients; the projection itself is still correct)."""
    M = M.to(torch.float64).cpu()
    d = d.to(torch.float64).cpu()
    n = M.shape[0]
    assert M.shape == (n, n) and d.shape == (n,), (M.shape, d.shape)
    diag = M.diagonal().clamp_min(0.0)
    zero = diag <= 0.0
    scale = diag.sqrt()
    scale = torch.where(zero, torch.ones_like(scale), scale)
    Mn = M / (scale[:, None] * scale[None, :])
    Mn = 0.5 * (Mn + Mn.T)
    sv = torch.linalg.svdvals(Mn)
    degenerate = bool(zero.any()) or bool(sv.min() < 1e-6 * sv.max())
    alpha_n = torch.linalg.pinv(Mn, rtol=rtol) @ (d / scale)
    alpha = alpha_n / scale
    alpha = torch.where(zero, torch.zeros_like(alpha), alpha)
    return alpha, degenerate


def project_grads(grads, dirs, disps, gram=None, one_sided=True, deadband=0.0):
    """The CNP rule, in place on `grads` (list of per-parameter grad tensors or
    None). `dirs` is an ordered dict name -> list-of-tensors direction aligned
    with `grads`; `disps` name -> s(theta) - s0 for that direction's statistic;
    `gram` the directions' Gram matrix (computed if None; pass the cached one).

    Two-sided: every direction is active. One-sided: a direction is active iff
    the descent step moves its statistic away from base (away_from_base); after
    projecting, the remaining directions are re-tested with the scalar-updated
    dots <G', d_j> = <G, d_j> - sum_i alpha_i M_ij (removing one direction can
    flip another's motion), until no new direction is away. The active set A
    is then removed in ONE vector update  G <- G - sum_{i in A} alpha_i d_i  with
    alpha = M_AA^+ d_A, i.e. G - N_A (N_A^T N_A)^-1 N_A^T G.

    Returns a JSON-serialisable info dict: grad_norm, dot_before/dot_after (per
    direction; after = scalar prediction, ~0 on active ones), disp, active,
    alpha, removed_norm, removed_ratio = |G - G'| / |G| in [0, 1], degenerate.
    """
    names = list(dirs)
    assert names, "project_grads: no directions"
    for n in names:
        assert len(dirs[n]) == len(grads), (n, len(dirs[n]), len(grads))
        assert n in disps, f"no displacement for direction {n!r}"
    if gram is None:
        gram = gram_matrix([dirs[n] for n in names])
    gram = gram.to(torch.float64).cpu()
    assert gram.shape == (len(names), len(names)), gram.shape

    dots0 = torch.tensor([vdot(grads, dirs[n]) for n in names], dtype=torch.float64)
    gnorm = vnorm(grads)
    disp_v = [float(disps[n]) for n in names]

    active, remaining = [], list(range(len(names)))
    dots_cur = dots0.clone()
    alpha_full = torch.zeros(len(names), dtype=torch.float64)
    degenerate = False
    while remaining:
        new = [j for j in remaining
               if (not one_sided) or away_from_base(float(dots_cur[j]), disp_v[j], deadband)]
        if not new:
            break
        active = sorted(active + new)
        remaining = [j for j in remaining if j not in new]
        alpha, degen = solve_projection_coefficients(gram[active][:, active], dots0[active])
        degenerate = degenerate or degen
        alpha_full.zero_()
        alpha_full[active] = alpha
        dots_cur = dots0 - gram @ alpha_full

    if active:
        for i in active:
            coef = float(alpha_full[i])
            if coef != 0.0:
                vaxpy_(grads, -coef, dirs[names[i]])
    removed2 = float(alpha_full @ gram @ alpha_full)
    removed = math.sqrt(max(0.0, removed2))
    ratio = removed / gnorm if gnorm > 0.0 else 0.0
    ratio = min(1.0, max(0.0, ratio))
    return {
        "grad_norm": gnorm,
        "dot_before": {n: float(dots0[i]) for i, n in enumerate(names)},
        "dot_after": {n: float(dots_cur[i]) for i, n in enumerate(names)},
        "disp": {n: disp_v[i] for i, n in enumerate(names)},
        "active": [names[i] for i in active],
        "alpha": {names[i]: float(alpha_full[i]) for i in active},
        "removed_norm": removed,
        "removed_ratio": ratio,
        "degenerate": bool(degenerate),
    }


def refusal_stat(z, unans_id):
    """(n,) refusal log-odds from (n, V) last-position logits:
    z_unans - logsumexp_{v != unans} z_v = log p(unans) - log(1 - p(unans)).
    Invariant to a per-row additive logit shift (unlike the raw logit), fp32."""
    z = z.float()
    assert z.dim() == 2, tuple(z.shape)
    assert 0 <= int(unans_id) < z.shape[-1], (unans_id, z.shape)
    mask = torch.zeros(z.shape[-1], dtype=torch.bool, device=z.device)
    mask[int(unans_id)] = True
    others = z.masked_fill(mask, float("-inf"))
    return z[:, int(unans_id)] - torch.logsumexp(others, dim=-1)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

class _dropout_off:
    """Switch every nn.Dropout to eval for the direction estimate while leaving
    model.training untouched (gradient checkpointing keys on it). ewc.py's
    Fisher pass uses the same trick; a noisy direction is a worse constraint."""

    def __init__(self, model):
        self.model = model
        self.saved = []

    def __enter__(self):
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                self.saved.append((m, m.training))
                m.training = False
        return self

    def __exit__(self, *exc):
        for m, was in self.saved:
            m.training = was
        return False


def _chunks(rows, size=CHUNK_IMAGES):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


# ---------------------------------------------------------------------------
# The projector.
# ---------------------------------------------------------------------------

class CriterionNullSpace:
    """Hard criterion constraint on the task update (see module docstring).

    Args
      probe_preserver : an initialised method.crit_preserve.CriterionPreserver
                        (probe rows, chunked forward, yes/no ids, cached g0 --
                        init_base_stats must have run before the first refresh).
      every_T         : refresh the direction(s) every T optimizer steps
                        (global_step % T == 0; the first refresh is at step 0).
      k_probe         : probe items per refresh (forwarded <= 3 images at a time).
      one_sided       : ratchet rule (default) vs two-sided projection.
      use_refusal_dir : add the refusal-token direction n_ref.
      deadband        : |s - s0| <= deadband counts as "at base" (two-sided there).
      ref_probe_jsonl : optional {id, image, prompt} rows for the refusal
                        statistic (e.g. TextVQA-format questions on non-VizWiz
                        images; answers never read). Default None: reuse the
                        criterion probe items with the grounding question posed in
                        the leakage-prone open-VQA wrapper (policy_anchor's
                        ABSTAIN_SUFFIX) -- the same items as the criterion sample.
      ref_suffix      : wrapper appended to the derived refusal prompt.
      refusal_phrase  : the leakage string whose in-context FIRST token defines
                        s_ref (resolved like the yes/no ids; must be stable).
      dir_dtype       : storage dtype of the directions (fp32 default; bf16 halves
                        the ~640 MB per direction on LLaVA r=64).
    """

    FORBIDDEN_KEYS = CriterionPreserver.FORBIDDEN_KEYS

    def __init__(self, probe_preserver, every_T=5, k_probe=16, one_sided=True,
                 use_refusal_dir=False, deadband=0.0, ref_probe_jsonl=None,
                 ref_suffix=ABSTAIN_SUFFIX, refusal_phrase=REFUSAL_PHRASE,
                 dir_dtype=torch.float32):
        assert isinstance(probe_preserver, CriterionPreserver), type(probe_preserver)
        self.crit = probe_preserver
        self.T = int(every_T)
        self.k = int(k_probe)
        assert self.T >= 1, f"every_T must be >= 1, got {every_T}"
        assert 1 <= self.k <= len(self.crit.rows), (
            f"k_probe={self.k} but only {len(self.crit.rows)} probe items")
        self.one_sided = bool(one_sided)
        self.deadband = float(deadband)
        assert self.deadband >= 0.0, self.deadband
        self.use_ref = bool(use_refusal_dir)
        self.dir_dtype = dir_dtype
        self.rng = random.Random(SEED + 3)         # own stream: never consumes M1's
        self.device = self.crit.device

        # Refusal statistic: token id + probe rows.
        self.unans_id = None
        self.ref_rows = []          # [{id, image, prompt}]
        self.ref_derived = False    # True: ref rows are the criterion rows re-prompted
        self.ref_probe_path = ref_probe_jsonl
        self.ref0_by_id = None
        self.ref_base_source = None
        if self.use_ref:
            tok = self.crit.p.tokenizer
            stub = self.crit.bb.anchor_stub()
            stub_ids = tok(stub, add_special_tokens=False)["input_ids"]
            tid = _first_token_in_context(tok, stub, stub_ids, refusal_phrase)
            assert tid is not None, (
                f"refusal phrase {refusal_phrase!r} has no stable in-context first "
                "token; the refusal direction cannot be defined")
            assert tid not in (self.crit.yes_id, self.crit.no_id), tid
            self.unans_id = int(tid)
            self.refusal_phrase = refusal_phrase
            if ref_probe_jsonl is None:
                self.ref_derived = True
                self.ref_rows = [{"id": r["id"], "image": r["image"],
                                  "prompt": QUESTION.format(obj=r["object"]) + ref_suffix}
                                 for r in self.crit.rows]
            else:
                with open(ref_probe_jsonl) as f:
                    self.ref_rows = [json.loads(l) for l in f if l.strip()]
                assert self.ref_rows, f"empty refusal probe {ref_probe_jsonl}"
                ids = set()
                for r in self.ref_rows:
                    for key in ("id", "image", "prompt"):
                        assert key in r, f"refusal probe row missing {key!r}: {r}"
                    bad = [key for key in self.FORBIDDEN_KEYS if key in r]
                    assert not bad, (
                        f"refusal probe row {r.get('id')!r} carries label-like keys "
                        f"{bad}; the statistic is label-free by construction")
                    assert isinstance(r["prompt"], str) and r["prompt"].strip(), r
                    path = os.path.join(self.crit.data_root, r["image"])
                    assert os.path.exists(path), f"missing refusal probe image {path}"
                    assert r["id"] not in ids, f"duplicate refusal probe id {r['id']!r}"
                    ids.add(r["id"])
                assert self.k <= len(self.ref_rows), (
                    f"k_probe={self.k} but only {len(self.ref_rows)} refusal probe rows")

        # Bound at the first step (model on device, PEFT-wrapped).
        self.params = None
        self.param_names = None
        # Current constraint (set by refresh()).
        self.dirs = {}
        self.disps = {}
        self.gram = None
        self.last_refresh_step = None
        self.last_refresh_stats = None
        self.n_refresh = 0
        self.history = []

    # -- parameters ---------------------------------------------------------

    def bind(self, model):
        """Fix the trainable-parameter list (the flattened vector's layout)."""
        if self.params is None:
            named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            assert named, "CNP: model has no trainable parameters"
            self.param_names = [n for n, _ in named]
            self.params = [p for _, p in named]
        return self.params

    # -- forwards -----------------------------------------------------------

    def _ref_chunk(self, model, rows):
        """Forward <= CHUNK_IMAGES rows [(image_rel, prompt_text), ...]; return
        the (n,) fp32 refusal log-odds at each row's last real token. Mirrors
        CriterionPreserver._g_chunk (full processor call, every output passed
        to the model, last position from attention_mask)."""
        crit = self.crit
        assert 1 <= len(rows) <= CHUNK_IMAGES, len(rows)
        param = next(model.parameters())
        images = [crit._load(img) for img, _ in rows]
        texts = [crit.bb.build_prompt(pt) for _, pt in rows]
        enc = crit.p(text=texts, images=images, return_tensors="pt",
                     padding=True, truncation=True, max_length=2048)
        model_inputs = {}
        for name, v in enc.items():
            if hasattr(v, "to"):
                v = v.to(crit.device, param.dtype) if v.dtype.is_floating_point \
                    else v.to(crit.device)
            model_inputs[name] = v
        out = model(use_cache=False, **model_inputs)
        logits = out.logits
        attn = model_inputs["attention_mask"]
        assert logits.shape[1] == attn.shape[1], (
            f"logit length {logits.shape[1]} != input length {attn.shape[1]}"
            " - image-token expansion mismatch; last-position index would be wrong")
        assert logits.shape[-1] > self.unans_id
        last = attn.sum(dim=1) - 1
        idx = last.view(-1, 1, 1).expand(-1, 1, logits.shape[-1])
        z = logits.gather(1, idx).squeeze(1).float()          # (n, V)
        return refusal_stat(z, self.unans_id)

    def _crit_chunk(self, model, rows):
        return self.crit._g_chunk(model, rows)

    @staticmethod
    def _crit_key(r):
        return (r["image"], r["object"])

    @staticmethod
    def _ref_key(r):
        return (r["image"], r["prompt"])

    @torch.no_grad()
    def _all_ref_eval(self, model):
        was_training = model.training
        model.eval()
        try:
            parts = [self._ref_chunk(model, [self._ref_key(r) for r in ch]).detach().float().cpu()
                     for ch in _chunks(self.ref_rows)]
        finally:
            if was_training:
                model.train()
        v = torch.cat(parts)
        assert v.shape == (len(self.ref_rows),), v.shape
        return v

    def init_refusal_base(self, model, cache_path=None, fallback_path=None,
                          is_stage1=False):
        """Cache s_ref,i(theta0) for every refusal probe row, routed exactly like
        CriterionPreserver.init_base_stats (disable_adapter -> stage-1 identity
        -> previous stage's cache -> error). No-op (None) without the refusal
        direction. Returns the audit string stored in ref_base_source."""
        if not self.use_ref:
            return None
        ctx = CriterionPreserver._adapter_disabler(model)
        if ctx is not None:
            with ctx:
                v = self._all_ref_eval(model)
            source = "disable_adapter"
        elif is_stage1:
            v = self._all_ref_eval(model)
            source = "stage1_fresh_lora_is_base"
        elif fallback_path and os.path.exists(fallback_path):
            ref0, payload = load_base_stats(fallback_path, [r["id"] for r in self.ref_rows])
            self.ref0_by_id = ref0
            source = f"loaded:{fallback_path}"
            self.ref_base_source = source
            if cache_path:
                meta = {k: val for k, val in payload.items() if k != "g0"}
                meta["source"] = source
                save_base_stats(cache_path, ref0, meta)
            return source
        else:
            raise RuntimeError(
                "cannot obtain base-model refusal statistics: model has no PEFT "
                "disable_adapter() context, this is not stage 1, and no cached "
                f"{REFUSAL_BASE_CACHE} was found at {fallback_path!r}")
        self.ref0_by_id = {r["id"]: float(x) for r, x in zip(self.ref_rows, v.tolist())}
        self.ref_base_source = source
        if cache_path:
            save_base_stats(cache_path, self.ref0_by_id, {
                "source": source, "statistic": "refusal_logodds",
                "probe": os.path.abspath(self.ref_probe_path) if self.ref_probe_path
                else f"derived:{os.path.abspath(self.crit.probe_path)}",
                "backbone": getattr(self.crit.bb, "name", None),
                "unans_id": int(self.unans_id), "refusal_phrase": self.refusal_phrase})
        return source

    # -- sampling -------------------------------------------------------------

    def _sample(self, rng):
        """(crit_rows, ref_rows) for one refresh. Derived refusal rows are the
        SAME items as the criterion sample (one draw); a separate refusal probe
        is sampled independently."""
        idx = rng.sample(range(len(self.crit.rows)), self.k)
        crit_rows = [self.crit.rows[i] for i in idx]
        ref_rows = []
        if self.use_ref:
            if self.ref_derived:
                ref_rows = [self.ref_rows[i] for i in idx]
            else:
                ref_rows = [self.ref_rows[i] for i in rng.sample(range(len(self.ref_rows)), self.k)]
        return crit_rows, ref_rows

    # -- direction estimate ---------------------------------------------------

    def _direction(self, model, params, chunk_fn, keys, base_vals):
        """u = grad_theta mean_S stat(theta) accumulated chunk by chunk (one
        autograd.grad per chunk; param.grad untouched). Returns
        (u as list of per-param tensors in dir_dtype, s_live, s_base)."""
        acc = [torch.zeros_like(p, dtype=self.dir_dtype) for p in params]
        vals = []
        for ch in _chunks(keys):
            st = chunk_fn(model, ch)
            assert st.requires_grad, "probe statistic has no graph to the trainable params"
            grads = torch.autograd.grad(st.sum(), params, allow_unused=True)
            for a, g in zip(acc, grads):
                if g is not None:
                    a.add_(g.to(a.dtype))
            vals.append(st.detach().float().cpu())
            del grads, st
        k = float(len(keys))
        for a in acc:
            a.div_(k)
        live = torch.cat(vals)
        assert live.shape == (len(keys),), live.shape
        assert bool(torch.isfinite(live).all()), "probe statistic non-finite"
        base = torch.tensor(base_vals, dtype=torch.float32)
        return acc, float(live.mean()), float(base.mean())

    def refresh(self, model, global_step):
        """Re-estimate the direction(s) and displacement(s) at theta_t."""
        assert self.crit.g0_by_id is not None, (
            "CriterionPreserver.init_base_stats(model) must run before CNP")
        params = self.bind(model)
        crit_rows, ref_rows = self._sample(self.rng)
        dirs, disps, stats = {}, {}, {}
        with _dropout_off(model):
            u, s_live, s_base = self._direction(
                model, params, self._crit_chunk, [self._crit_key(r) for r in crit_rows],
                [self.crit.g0_by_id[r["id"]] for r in crit_rows])
            dirs[CRIT] = u
            disps[CRIT] = s_live - s_base
            stats[CRIT] = {"s_live": s_live, "s_base": s_base}
            if self.use_ref:
                assert self.ref0_by_id is not None, (
                    "init_refusal_base(model) must run before CNP with the refusal direction")
                n, r_live, r_base = self._direction(
                    model, params, self._ref_chunk, [self._ref_key(r) for r in ref_rows],
                    [self.ref0_by_id[r["id"]] for r in ref_rows])
                dirs[REF] = n
                disps[REF] = r_live - r_base
                stats[REF] = {"s_live": r_live, "s_base": r_base}
        self.dirs, self.disps = dirs, disps
        self.gram = gram_matrix([dirs[nm] for nm in dirs])
        names = list(dirs)
        for i, nm in enumerate(names):
            stats[nm]["dir_norm"] = math.sqrt(max(0.0, float(self.gram[i, i])))
            assert math.isfinite(stats[nm]["dir_norm"]), nm
        if len(names) == 2:
            d = math.sqrt(max(1e-30, float(self.gram[0, 0]) * float(self.gram[1, 1])))
            stats["cos_dirs"] = float(self.gram[0, 1]) / d
        self.last_refresh_step = int(global_step)
        self.last_refresh_stats = stats
        self.n_refresh += 1
        return stats

    # -- pre-train check --------------------------------------------------------

    @torch.no_grad()
    def pretrain_check(self, model, is_stage1):
        """No-grad displacement check on one k'-sample (own RNG so the step-0
        sample is unaffected): at stage 1 theta == theta0, so every
        |s(theta) - s(theta0)| must be ~0 (bf16 chunk/padding noise; bound
        STAGE1_GATE) -- wrong images/ids/adapter state fail here, before any
        step. Returns {name: disp}. Run with the model on device."""
        assert self.crit.g0_by_id is not None, "init_base_stats first"
        crit_rows, ref_rows = self._sample(random.Random(SEED + 4))
        was_training = model.training
        model.eval()
        try:
            out = {}
            live = torch.cat([self._crit_chunk(model, [self._crit_key(r) for r in ch]).float().cpu()
                              for ch in _chunks(crit_rows)])
            base = torch.tensor([self.crit.g0_by_id[r["id"]] for r in crit_rows])
            out[CRIT] = float(live.mean() - base.mean())
            if self.use_ref:
                assert self.ref0_by_id is not None, "init_refusal_base first"
                lr = torch.cat([self._ref_chunk(model, [self._ref_key(r) for r in ch]).float().cpu()
                                for ch in _chunks(ref_rows)])
                br = torch.tensor([self.ref0_by_id[r["id"]] for r in ref_rows])
                out[REF] = float(lr.mean() - br.mean())
        finally:
            if was_training:
                model.train()
        for nm, d in out.items():
            assert d == d, f"CNP pre-train displacement for {nm} is NaN"
            if is_stage1:
                assert abs(d) < STAGE1_GATE, (
                    f"stage-1 CNP displacement for {nm} is {d:.3f}, not ~0: live and "
                    "base statistics disagree on identical weights (wrong images/ids "
                    "or adapter state)")
        return out

    # -- the per-step hook ------------------------------------------------------

    def step(self, model, global_step):
        """Call between backward (grads accumulated, clipped) and
        optimizer.step(). Refreshes the direction(s) when due, projects
        param.grad in place, appends and returns the per-step info dict."""
        params = self.bind(model)
        grads = [p.grad for p in params]
        # Wiring check BEFORE the (expensive) refresh forward: no gradients means
        # the hook is not between backward and optimizer.step().
        assert any(g is not None for g in grads), (
            "CNP.step called with no gradients: the hook is not between backward "
            "and optimizer.step()")
        refreshed = False
        if self.last_refresh_step is None or (
                int(global_step) % self.T == 0 and int(global_step) != self.last_refresh_step):
            self.refresh(model, global_step)
            refreshed = True
        assert self.dirs, "CNP: no directions after refresh"
        with torch.no_grad():
            info = project_grads(grads, self.dirs, self.disps, gram=self.gram,
                                 one_sided=self.one_sided, deadband=self.deadband)
            info["step"] = int(global_step)
            info["refresh"] = refreshed
            if refreshed:
                info["refresh_stats"] = self.last_refresh_stats
                # Identifiability audit on refresh steps only (two big dots):
                # the realised <G', d> must match the scalar prediction (~0 on
                # active directions).
                info["dot_after_audit"] = {nm: vdot(grads, self.dirs[nm]) for nm in self.dirs}
        self.history.append(info)
        return info
