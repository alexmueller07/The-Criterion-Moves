"""O-LoRA (orthogonal LoRA) orthogonality penalty for the pilot harness.

Convention anchor: MCITlib (github.com/Ghy0501/MCITlib @ a3a622c), read at source in
design_notes/baselines_plan.md sec 1.4:

  * orthogonality loss :  sum |A_cur . A_prev^T|   (abs-sum of the cross-Gram matrix)
                          between the current task's LoRA-A and each previous task's
                          LoRA-A, with previous blocks detached.
  * weight             :  0.05, hardcoded in source
                          (`loss_total = loss + 0.05 * orthogonal_loss`,
                          LLaVA/OLoRA/llava/train/llava_trainer.py line 193).

DELIBERATE ADAPTATION (must be stated in the paper's baseline description):
MCITlib's O-LoRA keeps a SEPARATE LoRA block per task (A_1..A_t) and constrains the
new block A_t against the concatenation of previous blocks. Our harness trains a
SINGLE LoRA adapter resumed across stages (baselines_plan.md sec 2 flags this
topology as fundamentally different from per-task-block O-LoRA). We therefore
implement the *orthogonality-loss principle* rather than the vendored per-task-block
mechanism: at stage start we SNAPSHOT the resumed adapter's LoRA-A weights
(= "previous stages' accumulated subspace") and freeze that snapshot; during
training the penalty pushes the live LoRA-A off that frozen subspace. This is an
O-LoRA-*style* regulariser adapted to a continuously-trained adapter, NOT a
byte-faithful reimplementation of MCITlib's O-LoRA; the orthogonality *loss form*
(abs-sum cross-Gram) and the source weight (0.05) are matched exactly.

Stage 1 has no resumed adapter -> no snapshot -> no penalty (skip cleanly).

Torch-only module (no transformers / peft / PIL): the orthogonality math is
unit-testable on CPU (pilot/tests/test_baselines.py).
"""
import torch

# Substring identifying a LoRA down-projection ("A") weight in PEFT param names,
# e.g. "base_model.model.<...>.lora_A.default.weight".
LORA_A_MARK = "lora_A"


def orthogonality_term(a_cur, a_prev):
    """abs-sum of the cross-Gram matrix between two LoRA-A weight matrices.

        term = sum_ij | (A_cur @ A_prev^T)_ij |

    A_cur, A_prev are [r, in_features] LoRA-A weights. Entry (i, j) is the inner
    product of current row i with previous row j; the value is exactly 0 iff every
    current row is orthogonal to every previous row (the subspaces are orthogonal),
    and strictly positive otherwise. Computed in fp32 regardless of input dtype.
    """
    a_cur = a_cur.float()
    a_prev = a_prev.float()
    return (a_cur @ a_prev.t()).abs().sum()


def snapshot_lora_A(model):
    """Detached fp32 clones of every trainable LoRA-A weight, keyed by param name.

    Call once at stage start, on the freshly-resumed (untrained) adapter, to capture
    the previous stages' accumulated LoRA-A subspace as the frozen orthogonality
    anchor. Returns {} when the model has no LoRA-A params (nothing to anchor).
    """
    snap = {}
    for n, p in model.named_parameters():
        if LORA_A_MARK in n and p.requires_grad:
            snap[n] = p.detach().float().clone()
    return snap


def olora_penalty(model, prev_A):
    """Sum of orthogonality_term(live A, frozen previous A) over matched LoRA-A params.

    `prev_A` is the dict returned by snapshot_lora_A at stage start. Only names
    present in both the live model and the snapshot contribute. Returns a scalar
    fp32 tensor; when `prev_A` is empty (stage 1) the result is exactly 0.0.
    """
    if not prev_A:
        return torch.zeros((), dtype=torch.float32)
    total = None
    for n, p in model.named_parameters():
        if n in prev_A:
            term = orthogonality_term(p, prev_A[n])
            total = term if total is None else total + term
    if total is None:
        return torch.zeros((), dtype=torch.float32)
    return total
