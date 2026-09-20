"""EWC (Elastic Weight Consolidation) for the pilot harness -- new baseline arm.

Convention anchor: CoIN (github.com/zackschen/CoIN @ 41411ab), read at source in
design_notes/baselines_plan.md sec 1.2 / 3.2:

  * penalty form  :  lambda * sum_n F_n * (theta_n - theta*_n)^2     (NO 1/2 factor)
  * lambda        :  0.5  (CoIN EWC_lambda default; run scripts name dirs CoIN_EWC_0.5)
  * Fisher        :  diagonal empirical Fisher over the *first N examples* of the
                     just-finished stage (deterministic slice, not random),
                     accumulate per-micro-batch grad^2, divide by the example count
  * anchor        :  previous stage only (chain-EWC), not a sum over all past tasks

Three declared, deliberate deviations from CoIN source (each verified-at-source in
baselines_plan.md, repeated here so the module is self-documenting):

  1. Penalty is computed on `p` (the live Parameter), NOT `p.data`. CoIN's published
     code uses `(p.data - optpar).pow(2)`; `p.data` is detached from the autograd
     graph, so CoIN's EWC term contributes ZERO gradient -- it only inflates the
     logged loss and never regularises. We use `p` so the penalty actually
     constrains drift. (CoIN's lambda=0.5 was therefore never validated end-to-end;
     baselines_plan.md sec 3.2 pre-registers a coarse sweep {0.05, 0.5, 5}.)
  2. Dropout is disabled during the Fisher pass (CoIN left lora_dropout=0.05 active).
     Dropout injects noise into F and breaks exact-reproducibility checks
     (cf. memory env_pg_ratio_exact_check).
  3. Fisher / optpar are kept in fp32 (CoIN also cast to cpu float).

NOTE ON THE 1/2 FACTOR: the textbook Kirkpatrick-2017 EWC penalty is
(lambda/2) * sum F (theta-theta*)^2. This module follows the CoIN convention with
NO 1/2 factor, because lambda=0.5 is the *CoIN-verified* value under the no-1/2
form and baselines_plan.md directs us to match CoIN's lambda. Folding a 1/2 in
would silently redefine what "lambda=0.5" means relative to the convention anchor.

Torch-only module: it does NOT import transformers / peft / PIL, so the penalty
math is unit-testable on CPU (pilot/tests/test_baselines.py) without the heavy
model stack. All functions take the model / dataset / collator as arguments.
"""
import os

import torch


def ewc_penalty(named_parameters, fisher, optpar, lam):
    """CoIN chain-EWC quadratic penalty, corrected to be a real regulariser.

        penalty = lam * sum_{n in fisher} F_n * (theta_n - theta*_n)^2      (no 1/2)

    Args:
      named_parameters: iterable of (name, Parameter) -- pass
                        `model.named_parameters()` at call sites. Only names present
                        in `fisher` contribute (that key set is exactly the trainable
                        params recorded by compute_fisher), so `p` is used directly
                        (grad flows) rather than `p.data`.
      fisher : dict name -> fp32 tensor (diagonal Fisher), same device as params.
      optpar : dict name -> fp32 tensor (stage-end reference params theta*).
      lam    : float penalty weight (CoIN: 0.5).

    Returns a scalar fp32 tensor. When `fisher` is empty (stage 1) the sum is over
    nothing and the result is exactly 0.0 -- the "skip cleanly at stage 1" contract.
    """
    total = None
    for n, p in named_parameters:
        if n not in fisher:
            continue
        f = fisher[n]
        star = optpar[n]
        term = (f * (p.float() - star).pow(2)).sum()
        total = term if total is None else total + term
    if total is None:
        # No overlapping params (stage 1: empty fisher) -> zero penalty, no grad.
        return torch.zeros((), dtype=torch.float32)
    return lam * total


@torch.no_grad()
def _restore_dropout(model, saved):
    for m, was_training in saved:
        m.training = was_training


def compute_fisher(model, dataset, collator, n_batches, micro_batch, device="cuda"):
    """Diagonal empirical Fisher over the FIRST (n_batches * micro_batch) examples.

    Matches CoIN's "first 1,000 examples" convention when called with
    n_batches * micro_batch == 1000 (run_arm passes 500 batches at micro_batch 2).
    Deterministic first-N slice (not random); grad^2 accumulated per micro-batch;
    normalised by the example count actually seen. Dropout is frozen (deviation 2).

    Returns (fisher, optpar) as dicts of CPU fp32 tensors, ready for torch.save.
    """
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    fisher = {n: torch.zeros_like(p, dtype=torch.float32, device=device)
              for n, p in named}
    optpar = {n: p.detach().float().clone() for n, p in named}

    # Freeze dropout for a noise-free Fisher (deviation 2) but keep train mode so
    # the gradient-checkpointing path stays valid.
    was_training = model.training
    model.train()
    saved = []
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            saved.append((m, m.training))
            m.training = False

    n_examples = min(n_batches * micro_batch, len(dataset))
    seen = 0
    try:
        for s in range(0, n_examples, micro_batch):
            idx = list(range(s, min(s + micro_batch, n_examples)))
            batch = collator([dataset[j] for j in idx])
            model.zero_grad(set_to_none=True)
            out = model(**{k: v.to(device) for k, v in batch.items()})
            out.loss.backward()
            for n, p in named:
                if p.grad is not None:
                    fisher[n] += p.grad.detach().float().pow(2)
            seen += len(idx)
    finally:
        model.zero_grad(set_to_none=True)
        _restore_dropout(model, saved)
        model.train(was_training)

    denom = float(max(1, seen))
    for n in fisher:
        fisher[n] /= denom  # CoIN normalisation: divide by example count
    fisher = {n: v.cpu() for n, v in fisher.items()}
    optpar = {n: v.cpu() for n, v in optpar.items()}
    return fisher, optpar


def save_ewc(out_dir, fisher, optpar):
    """Persist stage-end Fisher + reference params for the next stage to load."""
    os.makedirs(out_dir, exist_ok=True)
    torch.save({k: v.cpu() for k, v in fisher.items()},
               os.path.join(out_dir, "fisher.pt"))
    torch.save({k: v.cpu() for k, v in optpar.items()},
               os.path.join(out_dir, "optpar.pt"))


def load_ewc(load_dir, device="cuda"):
    """Load fisher.pt / optpar.pt written by a previous stage onto `device`."""
    fisher = torch.load(os.path.join(load_dir, "fisher.pt"), map_location=device)
    optpar = torch.load(os.path.join(load_dir, "optpar.pt"), map_location=device)
    fisher = {k: v.to(device).float() for k, v in fisher.items()}
    optpar = {k: v.to(device).float() for k, v in optpar.items()}
    return fisher, optpar
