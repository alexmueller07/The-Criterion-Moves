"""Casing-robust resolution of the yes/no answer tokens, and a pooled decision statistic.

WHY THIS EXISTS (2026-09-09). `faith_loss.resolve_answer_token_ids` resolves the
answer tokens ONCE, as " Yes" / " No", and every criterion instrument and every
criterion regularizer in this project reads those two vocabulary rows. But the
model's emitted answer token DRIFTS WITH THE TASK:

    stage k1 (ScienceQA)  -> "Yes" / "No"    resolver correct
    stage k2 (TextVQA)    -> "yes" / "no"    resolver WRONG on 100% of rows
    stage k3 (Flickr30k)  -> "Yes" / "No"    resolver correct
    stage k4 (VizWiz)     -> "yes" / "no"    resolver WRONG on 100% of rows

identically in all four arms measured (psL_seq, psL_anchor, mth_critp,
mth_critp1). This is task-recency drift in the answer vocabulary itself. The
consequences are that (a) the logit instrument records logits for tokens the
model never emits at those stages -- the single-id proxy agrees with the realized
argmax only 90.7% of the time -- and (b) the criterion regularizer PINS a
statistic that has decoupled from the decision the model actually makes, which is
a candidate explanation for the anchor's conservative displacement, since the
pilot endpoint is one of the affected stages.

THE FIX. Resolve yes-ness and no-ness as SETS of vocabulary rows covering the
casing and leading-space variants, then pool within each set before differencing:

    g = logsumexp_{t in YES} z_t  -  logsumexp_{t in NO} z_t

logsumexp rather than max because it is smooth (the statistic is used inside a
training loss and must stay differentiable everywhere) and because it is the
correct pooling for "probability the answer is any yes-form": it is exactly
log sum_t exp(z_t), the unnormalized log-mass of the yes set. When only one
variant carries mass it reduces to that variant's logit, so this agrees with the
old definition wherever the old definition was right, and fixes it where it was
not.

This module is NEW rather than a patch to faith_loss.py on purpose. ~85 arms are
queued on the cluster and a pending job runs whatever is on disk WHEN IT STARTS,
so editing the shared resolver would silently change queued arms and make them
incomparable with the arms already completed under the old resolver. Whether to
re-run the affected arms with pooled tokens is a separate, deliberate decision.
"""
import torch

# Casing / leading-space variants. Resolved IN CONTEXT (stub + variant), because
# " Yes" in isolation tokenizes to [space-marker, "Yes"] under the Llama
# tokenizer -- the id we need is the first token by which tok(stub + v) extends
# tok(stub). Order is irrelevant; duplicates are collapsed.
YES_VARIANTS = (" Yes", " yes", "Yes", "yes", " YES", "YES")
NO_VARIANTS = (" No", " no", "No", "no", " NO", "NO")


def _first_extending_id(tokenizer, stub_ids, stub, text):
    """The first token by which tok(stub + text) extends tok(stub), or None.

    Returns None instead of raising when the context tokenization is unstable for
    this variant (some variants legitimately re-tokenize the stub's tail); the
    caller needs the variants that DO resolve cleanly, not an exception on the
    ones that do not.
    """
    full = tokenizer(stub + text, add_special_tokens=False)["input_ids"]
    if full[:len(stub_ids)] != stub_ids or len(full) <= len(stub_ids):
        return None
    return full[len(stub_ids)]


def resolve_answer_token_sets(tokenizer, backbone):
    """(yes_ids, no_ids) as sorted disjoint lists of vocabulary rows.

    Fails loud if either set is empty (nothing resolved -> the statistic would be
    meaningless) or if the two sets intersect (a token counted as both yes and no
    would silently cancel).
    """
    stub = backbone.anchor_stub()
    stub_ids = tokenizer(stub, add_special_tokens=False)["input_ids"]
    yes, no = set(), set()
    for text in YES_VARIANTS:
        tid = _first_extending_id(tokenizer, stub_ids, stub, text)
        if tid is not None:
            yes.add(tid)
    for text in NO_VARIANTS:
        tid = _first_extending_id(tokenizer, stub_ids, stub, text)
        if tid is not None:
            no.add(tid)
    assert yes, "no yes-variant resolved in context; check backbone.anchor_stub()"
    assert no, "no no-variant resolved in context; check backbone.anchor_stub()"
    overlap = yes & no
    assert not overlap, (
        f"yes/no token sets overlap on {sorted(overlap)}; pooling would cancel")
    return sorted(yes), sorted(no)


def pooled_decision_stat(step_logits, yes_ids, no_ids):
    """g = logsumexp(z over yes rows) - logsumexp(z over no rows).

    `step_logits` is (n, vocab) fp32 at the answer position. Returns (n,) fp32.
    Differentiable in every argument, unlike a max-pool over variants.
    """
    assert step_logits.dim() == 2, step_logits.shape
    v = step_logits.shape[-1]
    assert max(yes_ids) < v and max(no_ids) < v, "token id outside vocabulary"
    z = step_logits.float()
    y = torch.logsumexp(z[:, yes_ids], dim=-1)
    n = torch.logsumexp(z[:, no_ids], dim=-1)
    return y - n


def realized_side(argmax_id, yes_ids, no_ids):
    """'yes' / 'no' / 'other' for a realized argmax id. For instrument audits.

    Lets a dump record whether the emitted token was in either pooled set, so the
    casing-drift failure is detectable directly from the dump in future rather
    than requiring a separate investigation.
    """
    if argmax_id in set(yes_ids):
        return "yes"
    if argmax_id in set(no_ids):
        return "no"
    return "other"
