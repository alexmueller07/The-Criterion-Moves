"""LwF (Learning without Forgetting) for the pilot harness -- offline top-k
teacher-logit distillation.

FEASIBILITY (24GB RTX 4090): a live second 7B teacher does NOT fit
(baselines_plan.md sec 2). This module implements the *offline cached-logits*
variant instead, which does fit trivially: top-k=20 logits over ~100 examples is
~single-digit MB. Because our harness resumes ONE adapter per stage, the teacher
(the previous stage's model) is exactly the model we just loaded, BEFORE any
training step -- so we cache in the SAME process, right after resume:

  * teacher = the freshly-resumed adapter (pre-training weights).
  * cache pass: forward the teacher over the first N current-task examples, store
    the top-k vocab logits at each supervised (target) position, keyed by dataset
    index.
  * train pass: the student (same weights) trains; a KD term is added on exactly
    those N examples wherever they appear in the shuffled epoch.

This SAME-PROCESS design removes the correctness trap baselines_plan.md flagged for
the CoIN offline route: CoIN aligns teacher/student by truncating to the min shared
sequence length (silently misaligns under dynamic padding). Here teacher and
student see byte-identical per-example tokenisation, and with right padding the
supervised positions have identical ABSOLUTE indices in the padded batch row, so
KD positions line up exactly -- no id matching, no min-length truncation.

Convention anchor: CoIN (github.com/zackschen/CoIN @ 41411ab), design_notes/
baselines_plan.md sec 1.3:
  * KD weight (alpha) : 0.1   (CoIN LWF_lambda default)
  * temperature T     : 2     (CoIN smooth(probs, T)=softmax(logits/T))
  * subset size       : 100 current-task examples
  * KD loss           : teacher-weighted CE, -mean sum old*log(new) on softened probs

Declared deviations (documented so the baseline description is honest):
  1. TOP-K (default 20) instead of CoIN's full vocab. Full-vocab logits for 100
     examples are ~4.5GB bf16; top-k=20 is ~MB and is the route baselines_plan.md
     sec 2 named as feasible. KD is computed over the teacher's top-k indices,
     renormalised within that support on BOTH teacher and student (a standard
     top-k KD approximation).
  2. Standard T^2 KD scaling is applied (keeps the KD gradient magnitude
     temperature-invariant); CoIN's raw form omits it. alpha absorbs the constant.
  3. First-N deterministic slice instead of CoIN's random.sample(.,100), for
     reproducibility and clean index alignment with the cache.

Torch-only module (no transformers / peft): the KD math is unit-testable on CPU.
The dataset / collator wrappers are thin and reuse the harness's base collator.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def build_teacher_cache(model, dataset, collator, n_examples, micro_batch,
                        top_k, device="cuda"):
    """Cache the teacher's top-k logits at supervised positions for the first N
    examples. Returns {dataset_index -> {"q", "idx", "logit"}} of CPU tensors.

    For each example, supervised (target) token positions p are labels != -100; the
    predicting logit index is q = p - 1. We store q (so training gathers student
    logits at the same q), the teacher's top-k vocab ids at q, and their logits.
    Dropout is frozen for a stable teacher signal.
    """
    cache = {}
    was_training = model.training
    model.train()
    saved = []
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            saved.append((m, m.training))
            m.training = False
    try:
        n_examples = min(n_examples, len(dataset))
        for s in range(0, n_examples, micro_batch):
            idx = list(range(s, min(s + micro_batch, n_examples)))
            batch = collator([dataset[j] for j in idx])
            labels = batch["labels"]
            out = model(**{k: v.to(device) for k, v in batch.items()})
            logits = out.logits  # [B, L, V]
            for b, gidx in enumerate(idx):
                lab = labels[b]
                target_pos = (lab != -100).nonzero(as_tuple=False).squeeze(-1)
                # predicting-logit index q = p - 1; drop any p == 0 (no predictor).
                q = target_pos[target_pos > 0] - 1
                if q.numel() == 0:
                    continue
                step = logits[b, q, :].float()  # [P, V]
                topk = torch.topk(step, k=min(top_k, step.shape[-1]), dim=-1)
                cache[gidx] = {
                    "q": q.detach().cpu(),
                    "idx": topk.indices.detach().cpu(),
                    "logit": topk.values.detach().cpu(),
                }
    finally:
        for m, tr in saved:
            m.training = tr
        model.train(was_training)
    return cache


def kd_loss(student_logits, kd_list, temp):
    """Top-k teacher-weighted KD over the cached positions.

    student_logits: [B, L, V] (from the student's forward on the training batch).
    kd_list       : length-B list; each entry is None or {"q","idx","logit"}.
    temp          : distillation temperature T.

    Returns a scalar fp32 tensor averaged over contributing examples, or None when
    no example in the batch carries teacher data (so the caller adds nothing).
    """
    dev = student_logits.device
    terms = []
    for b, kd in enumerate(kd_list):
        if kd is None:
            continue
        q = kd["q"].to(dev)
        t_idx = kd["idx"].to(dev)
        t_logit = kd["logit"].to(dev).float()
        s_step = student_logits[b, q, :].float()            # [P, V]
        s_topk = torch.gather(s_step, 1, t_idx)             # [P, k] at teacher ids
        t_prob = F.softmax(t_logit / temp, dim=-1)          # teacher, top-k support
        s_logprob = F.log_softmax(s_topk / temp, dim=-1)    # student, same support
        per_pos = -(t_prob * s_logprob).sum(dim=-1)         # [P]
        terms.append(per_pos.mean() * (temp * temp))
    if not terms:
        return None
    return torch.stack(terms).mean()


class LwFDataset(torch.utils.data.Dataset):
    """Wraps the base JSONL dataset, attaching each example's cached teacher data
    (or None) under "_lwf". Length and item shape are otherwise unchanged, so the
    Trainer's sampler and step counts match the base dataset exactly."""

    def __init__(self, base, cache):
        self.base = base
        self.cache = cache

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = self.base[i]
        item["_lwf"] = self.cache.get(i)
        return item


class LwFCollator:
    """Wraps the base collator: builds the normal encoding, then appends the batch's
    per-example teacher data as a plain list under "lwf_kd". The base collator
    ignores the "_lwf" key. compute_loss pops "lwf_kd" before the model forward, so
    the model never sees the extra key."""

    def __init__(self, base):
        self.base = base

    def __call__(self, batch):
        enc = self.base(batch)
        enc["lwf_kd"] = [ex.get("_lwf") for ex in batch]
        return enc
