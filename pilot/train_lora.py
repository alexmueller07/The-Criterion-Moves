"""LoRA training on instruction JSONL data (backbone via --backbone/model_zoo).

Arms:
  SEQ stage k:  --data tasks/<task_k>/train.jsonl --resume_adapter <stage k-1 dir>
  JOINT:        --data t1,t2,t3,t4 (shuffled union, seed 17) --checkpoint_steps 90,215,340

Prompt format for the default llava15 backbone:
  "USER: <image>\n{prompt} ASSISTANT: {target}</s>"
Labels mask everything up to and including "ASSISTANT: ". All backbone-specific
strings/classes (prompt build, freeze rule, LoRA targets, dtype/attn) live in
model_zoo.py adapters; parity with the pre-refactor hardcoded LLaVA values is
enforced by tests/test_backbone_parity.py.
"""
import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ewc as ewc_mod
import lwf as lwf_mod
import olora as olora_mod
from common.gate_checks import run_all_gates
from method.cnp_project import REFUSAL_BASE_CACHE, CriterionNullSpace
from method.crit_preserve import MODES as CRIT_MODES, CriterionPreserver
from method.faith_loss import CF_FORMS, FaithfulnessAnchor
from method.gca_loss import GenerativeCounterfactualAnchor
from method.policy_anchor import EOS_REF_CACHE, PolicyAnchor, compose_policy_loss
from model_zoo import available_backbones, get_backbone

import torch
from PIL import Image
from torch.utils.data import Dataset

SEED = 17


class JsonlVLDataset(Dataset):
    def __init__(self, paths, data_root, shuffle):
        self.rows = []
        for p in paths:
            with open(p) as f:
                self.rows.extend(json.loads(l) for l in f)
        if shuffle:
            random.Random(SEED).shuffle(self.rows)
        self.data_root = data_root

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(os.path.join(self.data_root, r["image"])).convert("RGB")
        return {"image": img, "prompt": r["prompt"], "target": r["target"]}


class Collator:
    def __init__(self, processor, max_len, backbone):
        self.p = processor
        self.bb = backbone
        self.max_len = max_len
        self.n_truncated = 0
        self.n_seen = 0

    def __call__(self, batch):
        # Prefix must end WITHOUT a trailing space: "ASSISTANT: " + "A" merges
        # the space into the answer token ("_A"), the prefix-length mask then
        # swallows the whole answer and only EOS gets supervised (loss ~0 from
        # step 1 — caught by the smoke run on 2026-08-31). The string
        # construction now lives in model_zoo.build_train_text (parity-tested).
        texts, prefixes, targets, images = [], [], [], []
        for ex in batch:
            full, prefix, target_fragment = self.bb.build_train_text(
                ex["prompt"], ex["target"], self.p.tokenizer.eos_token)
            texts.append(full)
            prefixes.append(prefix)
            targets.append(target_fragment)
            images.append(ex["image"])
        enc = self.p(text=texts, images=images, return_tensors="pt",
                     padding=True, truncation=True, max_length=self.max_len)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for j, prefix in enumerate(prefixes):
            pre = self.p(text=prefix, images=images[j], return_tensors="pt",
                         truncation=True, max_length=self.max_len)
            plen = pre["input_ids"].shape[1]
            labels[j, :plen] = -100
            n_unmasked = int((labels[j] != -100).sum())
            n_target = len(self.p.tokenizer(targets[j],
                                            add_special_tokens=False)["input_ids"])
            truncated = enc["input_ids"].shape[1] >= self.max_len
            if not truncated:
                assert n_unmasked >= 2 and abs(n_unmasked - n_target) <= 1, (
                    f"label boundary broken: {n_unmasked} unmasked vs "
                    f"{n_target} target tokens for target {targets[j][:60]!r}")
            self.n_seen += 1
            if truncated:
                self.n_truncated += 1
        enc["labels"] = labels
        return enc


def build_parser():
    """The CLI. Factored out of main() (2026-09-08, pure refactor) so the flag
    surface is testable on CPU (tests/test_policy_wiring.py)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="comma-separated jsonl paths")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume_adapter", default=None)
    ap.add_argument("--joint_shuffle", action="store_true")
    ap.add_argument("--checkpoint_fracs", default="")
    ap.add_argument("--checkpoint_steps", default="",
                    help="comma list of exact global steps to snapshot (takes precedence)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--micro_batch", type=int, default=2)
    ap.add_argument("--eff_batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--lora_r", type=int, default=64)
    ap.add_argument("--backbone", default="llava15",
                    choices=available_backbones(),
                    help="model backbone adapter from model_zoo.py")
    ap.add_argument("--faith_pairs", default=None,
                    help="grounding_pairs.jsonl (relative paths inside resolve "
                         "against --data_root); enables the faithfulness anchor")
    ap.add_argument("--faith_weight", type=float, default=0.1)
    ap.add_argument("--faith_k", type=int, default=4)
    ap.add_argument("--faith_margin", type=float, default=2.0)
    ap.add_argument("--faith_ce_mode", action="store_true",
                    help="CE-on-counterfactuals ablation instead of margin loss")
    # R1 ablation (design_notes/method_ideas_lit1.md sec.2.0; IC-VCO's
    # partition-function critique). Selects the quantity the THIRD
    # (counterfactual) anchor term compares across the real and GT-masked
    # images: "logit" = z_yes (default, byte-identical to the replicated v2
    # anchor), "logodds" = the normalization-free g = z_yes - z_no. Terms 1-2
    # are untouched either way. Arm "anchorlo" in fullstudy/run_arm.py.
    ap.add_argument("--faith_cf_form", default="logit", choices=CF_FORMS,
                    help="counterfactual-term quantity: raw z_yes across the two "
                         "images (logit, default) or the within-image log-odds "
                         "z_yes - z_no (logodds; R1 ablation)")
    # Label-free criterion-preservation term (method/crit_preserve.py;
    # design_notes/method_crit_preserve.md). Enabled by --crit_probe; weight
    # defaults to 0 so every existing arm is byte-identical when unset.
    ap.add_argument("--crit_probe", default=None,
                    help="probe.jsonl ({id,image,object}, NO labels; images resolve "
                         "against --data_root); enables the criterion-preservation term")
    ap.add_argument("--crit_weight", type=float, default=0.0)
    ap.add_argument("--crit_k", type=int, default=8,
                    help="probe items sampled per step (forwarded <=3 images at a time)")
    ap.add_argument("--crit_mode", default="mean", choices=CRIT_MODES,
                    help="population moment(s) of g=z_yes-z_no held at the base "
                         "value; 'item' is the per-item ablation")
    ap.add_argument("--crit_delta", type=float, default=1.0,
                    help="item-mode dead zone in logits (ignored by mean/meanvar)")
    # Which TARGET the criterion term aims at. "base" is the original behaviour
    # (hold g at the frozen base's value). "balanced" aims at yes-rate 0.5 on a
    # balanced probe, which on such a probe IS the c=0 optimum and needs no
    # labels -- it exists because anchoring to the base inherits the base's own
    # mis-calibration (c=+0.431) and lands the anchor at +0.697.
    ap.add_argument("--crit_target", default="base", choices=("base", "balanced"),
                    help="criterion target: 'base' = frozen-base value (original); "
                         "'balanced' = yes-rate 0.5 on a balanced probe")
    ap.add_argument("--crit_bal_mode", default="rate", choices=("rate", "mean"),
                    help="balanced target only: 'rate' = soft yes-rate (correct "
                         "target); 'mean' = mean(g) (item-invariant gradient, "
                         "exact only if g is symmetric)")
    ap.add_argument("--crit_tau", type=float, default=1.0,
                    help="balanced 'rate' mode: sigmoid temperature; smaller is a "
                         "sharper surrogate for the hard yes-rate")
    # Criterion-null-space update projection (method/cnp_project.py;
    # design_notes/method_cnp.md): the HARD-constraint twin of the --crit_* term
    # on the SAME probe set (needs --crit_probe). Not a loss: it projects the
    # accumulated task gradients off grad_theta mean_S g right before
    # optimizer.step() (TrainerCallback.on_pre_optimizer_step). Default OFF ->
    # every existing arm is byte-identical when unset. With --crit_weight 0 the
    # M1 term is skipped entirely (projection-only arm); with a nonzero weight
    # the two compose (M1 restores, CNP prevents).
    ap.add_argument("--cnp", action="store_true",
                    help="enable criterion-null-space projection of the task update")
    ap.add_argument("--cnp_T", type=int, default=5,
                    help="re-estimate the projection direction(s) every T optimizer steps")
    ap.add_argument("--cnp_k", type=int, default=16,
                    help="probe items per direction estimate (forwarded <=3 images at a time)")
    ap.add_argument("--cnp_refusal", action="store_true",
                    help="add the refusal-token (' Unanswerable' first-token log-odds) "
                         "direction; projects off span{u, n_ref}")
    ap.add_argument("--cnp_two_sided", action="store_true",
                    help="always remove the component (default: one-sided ratchet, only "
                         "when the step would push the statistic away from its base value)")
    ap.add_argument("--cnp_deadband", type=float, default=0.0,
                    help="|s - s0| (logits) inside which the one-sided rule is two-sided")
    ap.add_argument("--cnp_ref_probe", default=None,
                    help="optional {id,image,prompt} rows for the refusal statistic (no "
                         "labels); default derives them from --crit_probe in the open-VQA wrapper")
    # Multi-axis answer-policy anchor (method/policy_anchor.py;
    # design_notes/method_multiaxis.md). Adds the ABSTENTION and EOS/stopping-
    # hazard terms on top of the --faith_pairs anchor (which stays the criterion
    # term, so the policy arm's criterion term is byte-identical to the anchor
    # arm's). Both weights default to 0 -> nothing is built, every existing arm
    # is byte-identical when unset. Reuses --faith_pairs for the grounding rows;
    # the EOS term needs --pol_captions ({id,image,caption} rows from
    # method/build_caption_anchor.py) and anchors to theta0 (base, adapter
    # disabled) cached per stage in <out>/pol_eos_ref.json.
    ap.add_argument("--pol_weight_abstain", type=float, default=0.0,
                    help="lambda_a: refusal-token suppression on answerable "
                         "grounding questions (0 = term off)")
    ap.add_argument("--pol_weight_eos", type=float, default=0.0,
                    help="lambda_e: per-position EOS log-prob stability vs the "
                         "base model on the caption anchor set (0 = term off)")
    ap.add_argument("--pol_k", type=int, default=8,
                    help="grounding pairs (abstain) and caption rows (eos) sampled "
                         "per step; each forwarded ONE image at a time")
    ap.add_argument("--pol_captions", default=None,
                    help="caption_anchor.jsonl ({id,image,caption}; images resolve "
                         "against --data_root); required iff --pol_weight_eos != 0")
    ap.add_argument("--pol_margin", type=float, default=2.0,
                    help="abstention margin m_a (grounded answer must beat the "
                         "logsumexp refusal score by this)")
    ap.add_argument("--pol_eos_mode", default="logprob", choices=("logprob", "explen"),
                    help="EOS term form: per-position squared log-prob delta "
                         "(default) or scalar expected-length delta")
    # Generative counterfactual token anchor (method/gca_loss.py;
    # design_notes/method_gca.md): the anchor's GT supervision moved to the
    # CAPTION-SLOT decision -- softplus margins on the log-odds z(o+) - z(o-)
    # at the first caption token of the present object, on the real image and
    # as a real-minus-masked difference. Reference-free; COCO GT only. Enabled
    # by --gca_pairs (+ --gca_captions); weight defaults to 0 so every existing
    # arm is byte-identical when unset (weight 0 with pairs set = monitor only,
    # the --crit_* convention).
    ap.add_argument("--gca_pairs", default=None,
                    help="grounding_pairs.jsonl for the caption-slot anchor (paths "
                         "resolve against --data_root); enables the GCA term")
    ap.add_argument("--gca_captions", default=None,
                    help="caption_anchor.jsonl ({id,image,caption}; joined to "
                         "--gca_pairs on id); required with --gca_pairs")
    ap.add_argument("--gca_weight", type=float, default=0.0,
                    help="lambda_g on L_GCA = sp(m - l(I)) + sp(m - [l(I) - l(I\\o+)])")
    ap.add_argument("--gca_k", type=int, default=4,
                    help="joined pairs sampled per step (one (real, masked) forward each)")
    ap.add_argument("--gca_terms", default="both", choices=("both", "decision", "cf"),
                    help="which GCA terms are active: both (proposed), decision (HALVA-style\n"
                         "slot-decision baseline), cf (See-or-Guess-style counterfactual baseline)")
    ap.add_argument("--gca_margin", type=float, default=2.0,
                    help="m_g (logits) for both GCA terms")
    # --- CL baseline arms (new code paths; all default OFF so SEQ/anchor/joint/
    #     er/cecf behavior stays byte-identical when unset). See ewc.py/olora.py/
    #     lwf.py and design_notes/baselines_plan.md for the at-source conventions.
    # EWC (CoIN chain-EWC, corrected):
    ap.add_argument("--ewc_lambda", type=float, default=0.0,
                    help=">0 enables the EWC penalty lambda*sum(F*(theta-theta*)^2) "
                         "(CoIN convention, no 1/2 factor; CoIN lambda=0.5)")
    ap.add_argument("--ewc_fisher_batches", type=int, default=500,
                    help="stage-end Fisher over the first (batches*micro_batch) "
                         "examples; 500*2=1000 matches CoIN's first-1000-examples")
    ap.add_argument("--ewc_load", default="",
                    help="dir with fisher.pt/optpar.pt from the previous stage "
                         "(omit at stage 1 -> penalty skipped cleanly)")
    ap.add_argument("--ewc_fisher_out", default="",
                    help="if set, compute+save stage-end fisher.pt/optpar.pt here "
                         "for the next stage")
    # O-LoRA (orthogonality-loss variant adapted to a single resumed adapter):
    ap.add_argument("--olora_weight", type=float, default=0.0,
                    help=">0 enables the O-LoRA orthogonality penalty "
                         "sum|A_cur.A_prev^T| vs the resumed adapter's LoRA-A "
                         "(source weight 0.05); no-op at stage 1 (no resume)")
    # LwF (offline top-k teacher-logit distillation):
    ap.add_argument("--lwf_alpha", type=float, default=0.0,
                    help=">0 enables LwF KD toward the resumed (teacher) adapter's "
                         "cached top-k logits (CoIN LWF_lambda=0.1); needs a "
                         "resumed adapter -> no-op at stage 1")
    ap.add_argument("--lwf_temp", type=float, default=2.0,
                    help="LwF distillation temperature (CoIN T=2)")
    ap.add_argument("--lwf_cache_n", type=int, default=100,
                    help="number of current-task examples to distill (CoIN: 100)")
    ap.add_argument("--lwf_top_k", type=int, default=20,
                    help="teacher logits cached per position (top-k KD approx)")
    ap.add_argument("--run_seed", type=int, default=SEED,
                    help="training randomness (LoRA init, dataloader order); "
                         "data composition stays pinned to SEED=17")
    return ap


def policy_requested(args):
    """True iff either policy-anchor term has a nonzero weight. With both at
    their default 0 no PolicyAnchor is built and compute_loss never sees it."""
    return args.pol_weight_abstain != 0.0 or args.pol_weight_eos != 0.0


def crit_term_active(args):
    """True iff the per-step M1 term (crit.loss inside compute_loss) runs.
    Unchanged from before CNP existed whenever --cnp is off: any --crit_probe
    runs the term, weight 0 meaning "monitor only". With --cnp AND weight 0 the
    term is skipped, so the projection-only arm pays no per-step M1 forward
    (CNP has its own s(theta) monitor at every refresh)."""
    if args.crit_probe is None:
        return False
    return not (args.cnp and args.crit_weight == 0.0)


def gca_requested(args):
    """True iff --gca_pairs is set. With the default None no anchor is built and
    compute_loss never sees it (every existing arm byte-identical)."""
    return bool(args.gca_pairs)


def add_gca_term(loss, gca, model, weight, accum, history, terms="both"):
    """One GCA step inside compute_loss (module-level so the composition path
    is CPU-testable with a mocked anchor):

        loss <- loss + accum * weight * (L_decision + L_cf)

    Same once-per-window, accum-scaled convention as faith/crit/policy. Appends
    the raw (unweighted) per-term values and the detached log-odds monitors to
    `history`; weight 0 still logs (monitor only) but adds nothing."""
    comps = gca.loss_components(model)
    history.append({k: float(v.detach()) for k, v in comps.items()})
    if terms == "decision":
        term_sum = comps["decision"]
    elif terms == "cf":
        term_sum = comps["cf"]
    else:
        term_sum = comps["decision"] + comps["cf"]
    return loss + accum * weight * term_sum


def add_policy_term(loss, pol, model, w_abstain, w_eos, accum, history):
    """One policy-anchor step inside compute_loss (module-level so the
    composition path is CPU-testable with a mocked anchor).

        loss <- loss + accum * (w_abstain * L_abstain + w_eos * L_eos)

    The criterion weight is pinned to 0 here: the criterion term is the
    --faith_pairs anchor, added separately in compute_loss exactly as in the
    anchor arm. Same once-per-window, accum-scaled convention as faith/crit.
    Appends the raw (unweighted) per-term values to `history`."""
    comps = pol.loss_components(model)
    history.append({"abstain": float(comps["abstain"].detach()),
                    "eos": float(comps["eos"].detach())})
    p = compose_policy_loss(comps["criterion"], comps["abstain"], comps["eos"],
                            0.0, w_abstain, w_eos)
    return loss + accum * p


def main():
    args = build_parser().parse_args()

    run_all_gates(args.out, min_free_gb=40.0)
    torch.manual_seed(args.run_seed)
    random.seed(args.run_seed)

    from peft import PeftModel, get_peft_model
    from transformers import Trainer, TrainerCallback, TrainingArguments

    backbone = get_backbone(args.backbone)
    model, processor = backbone.load(backbone.model_id)
    processor.tokenizer.padding_side = "right"
    model.config.use_cache = False

    backbone.freeze_policy(model)

    if args.resume_adapter:
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
        print(f"[train] resumed adapter from {args.resume_adapter}", flush=True)
    else:
        model = get_peft_model(model, backbone.lora_config(args.lora_r))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train] trainable params: {trainable/1e6:.1f}M / {total/1e9:.2f}B", flush=True)
    assert 1e7 < trainable < 1e9, "trainable-parameter count out of expected LoRA range"

    ds = JsonlVLDataset(args.data.split(","), args.data_root, shuffle=args.joint_shuffle)
    collator = Collator(processor, args.max_len, backbone)
    accum = max(1, args.eff_batch // args.micro_batch)

    anchor = None
    if args.faith_pairs:
        anchor = FaithfulnessAnchor(processor, args.faith_pairs, args.data_root,
                                    k_pairs=args.faith_k,
                                    margin=args.faith_margin, device="cuda",
                                    backbone=backbone,
                                    mode="ce" if args.faith_ce_mode else "margin",
                                    cf_form=args.faith_cf_form)
        print(f"[train] faith anchor: {len(anchor.pairs)} pairs k={anchor.k} "
              f"margin={anchor.margin} weight={args.faith_weight} "
              f"cf_form={anchor.cf_form}", flush=True)

    crit = None
    if args.crit_probe:
        if args.crit_target == "balanced":
            from method.crit_balanced import CriterionBalanced
            crit = CriterionBalanced(processor, args.crit_probe, args.data_root,
                                     k=args.crit_k, device="cuda",
                                     backbone=backbone, mode=args.crit_bal_mode,
                                     tau=args.crit_tau)
        else:
            crit = CriterionPreserver(processor, args.crit_probe, args.data_root,
                                      k=args.crit_k, device="cuda", backbone=backbone,
                                      mode=args.crit_mode, delta=args.crit_delta)
        print(f"[train] crit preserver: {len(crit.rows)} probe items k={crit.k} "
              f"mode={crit.mode} weight={args.crit_weight}"
              + (" (weight 0: monitor only)" if args.crit_weight == 0 and not args.cnp else "")
              + (" (weight 0 + --cnp: M1 term OFF, probe serves CNP)"
                 if args.crit_weight == 0 and args.cnp else ""),
              flush=True)
    crit_on = crit_term_active(args)

    # Criterion-null-space projection: built on the SAME probe/preserver; acts on
    # gradients in a callback (below), never on the loss. None unless --cnp.
    cnp = None
    if args.cnp:
        assert crit is not None, (
            "--cnp needs --crit_probe (the label-free probe set the projection "
            "direction is estimated on)")
        cnp = CriterionNullSpace(crit, every_T=args.cnp_T, k_probe=args.cnp_k,
                                 one_sided=not args.cnp_two_sided,
                                 use_refusal_dir=args.cnp_refusal,
                                 deadband=args.cnp_deadband,
                                 ref_probe_jsonl=args.cnp_ref_probe)
        print(f"[train] cnp: T={cnp.T} k'={cnp.k} one_sided={cnp.one_sided} "
              f"deadband={cnp.deadband} refusal_dir={cnp.use_ref}"
              + (f" unans_id={cnp.unans_id} ref_rows={len(cnp.ref_rows)}"
                 f"{' (derived from probe)' if cnp.ref_derived else ''}" if cnp.use_ref else ""),
              flush=True)

    # Generative counterfactual token anchor: tokenizer-only init (slot
    # resolution), no forwards, so it is built here like the faith anchor.
    gca = None
    if gca_requested(args):
        assert args.gca_captions, "--gca_pairs requires --gca_captions"
        gca = GenerativeCounterfactualAnchor(processor, args.gca_pairs, args.gca_captions,
                                             args.data_root, k=args.gca_k,
                                             margin=args.gca_margin, device="cuda",
                                             backbone=backbone)
        print(f"[train] gca anchor: {len(gca.rows)} slot rows joined "
              f"(report {gca.report}) k={gca.k} margin={gca.margin} "
              f"weight={args.gca_weight}"
              + (" (weight 0: monitor only)" if args.gca_weight == 0 else ""),
              flush=True)

    # Policy anchor is constructed AFTER the model is on cuda (its theta0 EOS
    # reference needs forwards); None until then and None forever when unset.
    pol = None
    if policy_requested(args):
        assert args.faith_pairs, (
            "--pol_weight_abstain/--pol_weight_eos need --faith_pairs (the policy "
            "anchor reuses the grounding pairs for its abstention rows)")
        if args.pol_weight_eos != 0.0:
            assert args.pol_captions, "--pol_weight_eos != 0 requires --pol_captions"

    steps_per_epoch = math.ceil(len(ds) / args.eff_batch)
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)

    callbacks = []
    marks = {}
    if args.checkpoint_steps:
        marks = {int(s): f"step{int(s):06d}" for s in args.checkpoint_steps.split(",")}
    elif args.checkpoint_fracs:
        fracs = [float(x) for x in args.checkpoint_fracs.split(",")]
        marks = {max(1, int(total_steps * fr)): f"frac{int(fr*100):03d}" for fr in fracs}
    if marks:
        bad = [s for s in marks if s > total_steps]
        assert not bad, f"checkpoint steps {bad} exceed total_steps={total_steps}"

        class MarkCkpt(TrainerCallback):
            def on_step_end(self, targs, state, control, **kw):
                if state.global_step in marks:
                    d = os.path.join(args.out, f"ckpt_{marks[state.global_step]}")
                    kw["model"].save_pretrained(d)
                    print(f"[train] checkpoint step={state.global_step} -> {d}", flush=True)
        callbacks.append(MarkCkpt())

    # NOTE: with this transformers+peft version the LOGGED loss is inflated by
    # the grad-accumulation factor (PEFT wrapper hides num_items_in_batch), so
    # logged values are only guarded against NaN/explosion. The real
    # plausibility check is the direct pre-training forward pass below.
    class LossGuard(TrainerCallback):
        def on_log(self, targs, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                v = logs["loss"]
                assert v == v and v < 25.0 * accum, (
                    f"logged loss {v} NaN or exploded (accum={accum})")
    callbacks.append(LossGuard())

    if cnp is not None:
        # Insertion point: HF's loop (transformers 4.49) runs, per optimizer
        # step, backward over the accumulation window -> clip_grad_norm_ ->
        # on_pre_optimizer_step -> optimizer.step() -> on_optimizer_step ->
        # zero_grad. Projecting param.grad in on_pre_optimizer_step therefore
        # sees the full accumulated, clipped task gradient and nothing else
        # touches it before Adam reads it. Fail loudly on a transformers
        # without the event rather than silently training an unconstrained arm.
        assert hasattr(TrainerCallback, "on_pre_optimizer_step"), (
            "this transformers has no TrainerCallback.on_pre_optimizer_step; CNP "
            "needs a hook between backward/clipping and optimizer.step()")

        class CNPProject(TrainerCallback):
            def on_pre_optimizer_step(self, targs, state, control, **kw):
                cnp.step(kw.get("model", model), state.global_step)
        callbacks.append(CNPProject())

    class FaithTrainer(Trainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._micro = 0
            self.faith_history = []
            self.crit_history = []
            self.pol_history = []
            self.gca_history = []
            # CL-baseline state (set after construction by the arm setup below;
            # all None/empty here so non-baseline arms take the original path).
            self.fisher = None
            self.optpar = None
            self.olora_prev = None
            self.lwf_active = False
            self.ewc_history = []
            self.olora_history = []
            self.lwf_history = []

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            # Pop the LwF side-channel BEFORE the model forward (the model must not
            # receive it as a kwarg). Absent for every non-LwF arm -> returns None,
            # so this line is a no-op there.
            kd_list = inputs.pop("lwf_kd", None)
            loss, outputs = super().compute_loss(model, inputs,
                                                 return_outputs=True, **kwargs)
            if anchor is not None and self._micro % accum == 0:
                f = anchor.loss(model)
                self.faith_history.append(float(f.detach()))
                loss = loss + args.faith_weight * accum * f
            # Criterion preservation: label-free, data-independent; same
            # once-per-window, accum-scaled convention as the faith term.
            # crit_on is True whenever --cnp is off (pre-CNP behaviour).
            if crit is not None and crit_on and self._micro % accum == 0:
                c = crit.loss(model)
                self.crit_history.append(float(c.detach()))
                loss = loss + args.crit_weight * accum * c
            # Multi-axis policy anchor: abstention + EOS terms ONLY (the
            # criterion term is the faith anchor above). Same once-per-window,
            # accum-scaled convention; pol is None unless a weight is set.
            if pol is not None and self._micro % accum == 0:
                loss = add_policy_term(loss, pol, model, args.pol_weight_abstain,
                                       args.pol_weight_eos, accum, self.pol_history)
            # Generative counterfactual token anchor: caption-slot margins on
            # the GT pairs; same once-per-window, accum-scaled convention. gca
            # is None unless --gca_pairs is set.
            if gca is not None and self._micro % accum == 0:
                loss = add_gca_term(loss, gca, model, args.gca_weight, accum,
                                    self.gca_history, terms=args.gca_terms)
            # EWC: data-independent penalty; add once per accumulation window scaled
            # by accum (same convention as the faith term) so it is not shrunk by
            # HF's grad-accumulation normalization.
            if self.fisher is not None and self._micro % accum == 0:
                pen = ewc_mod.ewc_penalty(model.named_parameters(),
                                          self.fisher, self.optpar, args.ewc_lambda)
                self.ewc_history.append(float(pen.detach()))
                loss = loss + accum * pen
            # O-LoRA: data-independent orthogonality penalty; same once-per-window,
            # weight 0.05 (from source).
            if self.olora_prev is not None and self._micro % accum == 0:
                open_ = olora_mod.olora_penalty(model, self.olora_prev)
                self.olora_history.append(float(open_.detach()))
                loss = loss + args.olora_weight * accum * open_
            # LwF: per-example KD; data-dependent, so it rides with the base CE at
            # the same per-micro-batch normalization (no accum multiply).
            if self.lwf_active and kd_list is not None:
                kd = lwf_mod.kd_loss(outputs.logits, kd_list, args.lwf_temp)
                if kd is not None:
                    self.lwf_history.append(float(kd.detach()))
                    loss = loss + args.lwf_alpha * kd
            self._micro += 1
            return (loss, outputs) if return_outputs else loss

    targs = TrainingArguments(
        output_dir=os.path.join(args.out, "trainer"),
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=accum,
        max_steps=total_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=4,
        seed=args.run_seed,
    )

    trainer = FaithTrainer(model=model, args=targs, train_dataset=ds,
                           data_collator=collator, callbacks=callbacks)

    with torch.no_grad():
        sample = collator([ds[i] for i in range(2)])
        n_lab = int((sample["labels"] != -100).sum())
        n_tok = int(sample["attention_mask"].sum())
        assert 0 < n_lab < n_tok, f"label masking broken: {n_lab}/{n_tok}"
        print(f"[train] label-mask check ok: {n_lab} target tokens / {n_tok} total", flush=True)
        model.to("cuda")
        out = model(**{k: v.to("cuda") for k, v in sample.items()})
        l0 = float(out.loss)
        assert 0.02 < l0 < 15.0, (
            f"pre-train per-token loss {l0} implausible: near-zero means degenerate "
            "label masking, huge means structural batch corruption")
        print(f"[train] pre-train forward per-token loss {l0:.4f}", flush=True)

    if anchor is not None:
        with torch.no_grad():
            f0 = float(anchor.loss(model))
        assert 0.0 < f0 < 50.0, f"pre-train faith loss {f0} implausible"
        print(f"[train] pre-train faith loss {f0:.4f}", flush=True)

    if gca is not None:
        # Reference-free (no theta0), so the only gate is plausibility: both
        # terms finite, the sum in the faith anchor's band. Model is on cuda.
        with torch.no_grad():
            g0 = {k: float(v) for k, v in gca.loss_components(model).items()}
        gsum = g0["decision"] + g0["cf"]
        assert gsum == gsum and 0.0 < gsum < 50.0, f"pre-train gca loss {g0} implausible"
        print(f"[train] pre-train gca decision {g0['decision']:.4f} cf {g0['cf']:.4f} "
              f"(mean slot log-odds l(I) {g0['ell_real']:.3f}, real-masked gap "
              f"{g0['ell_gap']:.3f})", flush=True)

    if crit is not None:
        # theta0 probe statistics: computed with the adapter DISABLED (the true
        # base even on a resumed adapter); falls back to the previous stage's
        # cache. Always written to <out>/probe_base_stats.json (next stage's
        # fallback + audit trail). Model is on cuda here.
        prev_cache = None
        if args.resume_adapter:
            prev_cache = os.path.join(os.path.dirname(args.resume_adapter.rstrip("/")),
                                      "probe_base_stats.json")
        src = crit.init_base_stats(
            model, cache_path=os.path.join(args.out, "probe_base_stats.json"),
            fallback_path=prev_cache, is_stage1=not args.resume_adapter)
        # base_summary() returns (mean, std) for the base-anchored preservers,
        # but CriterionBalanced has NO base statistics -- its target comes from
        # the probe's 50/50 balance rather than the frozen model -- and returns a
        # descriptive dict. Unpacking blindly killed every critbal arm at 18s.
        summary = crit.base_summary()
        anchored_to_base = isinstance(summary, tuple)
        if anchored_to_base:
            m0, s0 = summary
            stats_line = "mean g0=%.3f std g0=%.3f" % (m0, s0)
        else:
            stats_line = "target yes-rate=%.4f mode=%s tau=%s" % (
                summary.get("target_yes_rate", float("nan")),
                summary.get("mode"), summary.get("tau"))
        with torch.no_grad():
            c0 = float(crit.loss(model))
        assert c0 == c0 and c0 < 1e4, f"pre-train crit loss {c0} implausible"
        # The stage-1 identity check applies only when the target IS the frozen
        # base: then theta == theta0 and the paired penalty must vanish. A
        # balanced-probe target has no such identity, since the base sits at
        # c = +0.431 and its yes-rate penalty is legitimately non-zero.
        if anchored_to_base and not args.resume_adapter and crit.mode != "item":
            # Stage 1: theta == theta0 up to bf16 chunk/padding noise, so the
            # paired moment penalty must be ~0 (EWC step-0 gate analogue).
            assert c0 < 1.0, (
                f"stage-1 crit loss {c0} not ~0: live and base statistics disagree "
                "on identical weights (wrong images/ids or adapter state)")
        print(f"[train] crit base stats ({src}): {stats_line} "
              f"over {len(crit.rows)} probes; pre-train crit loss {c0:.4f}", flush=True)

    if cnp is not None:
        # Refusal-statistic theta0 reference (same routing as the crit base
        # stats; cached to <out>/probe_base_refusal.json), then a no-grad
        # displacement check: at stage 1 every |s - s0| must be ~0. The
        # direction itself is first estimated at step 0 inside training (with
        # gradient checkpointing active), not here.
        src_r = None
        if cnp.use_ref:
            prev_ref = None
            if args.resume_adapter:
                prev_ref = os.path.join(os.path.dirname(args.resume_adapter.rstrip("/")),
                                        REFUSAL_BASE_CACHE)
            src_r = cnp.init_refusal_base(
                model, cache_path=os.path.join(args.out, REFUSAL_BASE_CACHE),
                fallback_path=prev_ref, is_stage1=not args.resume_adapter)
        disp0 = cnp.pretrain_check(model, is_stage1=not args.resume_adapter)
        print(f"[train] cnp pre-train displacements {disp0}"
              + (f"; refusal base ({src_r})" if cnp.use_ref else ""), flush=True)

    if policy_requested(args):
        # Criterion term pinned OFF inside the composite (lambda_c=0,
        # use_criterion=False): it is the --faith_pairs anchor above. theta0 for
        # the EOS term follows crit_preserve's routing (adapter DISABLED ->
        # stage-1 identity -> previous stage's cache); always cached to
        # <out>/pol_eos_ref.json. Model is on cuda here.
        pol = PolicyAnchor(processor, args.faith_pairs, args.data_root,
                           k_pairs=args.pol_k, margin=args.faith_margin,
                           device="cuda", backbone=backbone,
                           lambda_c=0.0, use_criterion=False,
                           lambda_a=args.pol_weight_abstain,
                           lambda_e=args.pol_weight_eos,
                           margin_a=args.pol_margin,
                           caption_jsonl=args.pol_captions,
                           eos_mode=args.pol_eos_mode, k_caps=args.pol_k)
        prev_cache = None
        if args.resume_adapter:
            prev_cache = os.path.join(os.path.dirname(args.resume_adapter.rstrip("/")),
                                      EOS_REF_CACHE)
        src = pol.init_eos_reference(
            model, cache_path=os.path.join(args.out, EOS_REF_CACHE),
            fallback_path=prev_cache, is_stage1=not args.resume_adapter)
        with torch.no_grad():
            comps0 = {k: float(v) for k, v in pol.loss_components(model).items()}
        a0, e0 = comps0["abstain"], comps0["eos"]
        assert a0 == a0 and 0.0 <= a0 < 50.0, f"pre-train abstain loss {a0} implausible"
        assert e0 == e0 and e0 < 1e4, f"pre-train eos loss {e0} implausible"
        if pol.use_eos and not args.resume_adapter:
            # Stage 1: theta == theta0 up to bf16 noise, so the hazard-stability
            # term must be ~0 (the crit step-0 gate analogue).
            assert e0 < 1.0, (
                f"stage-1 eos loss {e0} not ~0: live and base stopping hazards "
                "disagree on identical weights (wrong captions/ids or adapter state)")
        print(f"[train] policy anchor: {len(pol.pairs)} pairs, "
              f"{len(pol.captions)} captions, k={args.pol_k}; "
              f"abstain w={args.pol_weight_abstain} refusal_ids={pol.refuse_ids} "
              f"(dropped {pol._dropped_refusals}); eos w={args.pol_weight_eos} "
              f"mode={args.pol_eos_mode} ref={src}; pre-train abstain {a0:.4f} "
              f"eos {e0:.4f}", flush=True)

    # --- CL baseline arm setup (model is on cuda here; runs before training so the
    #     resumed adapter still holds the previous stage's weights = the anchor /
    #     teacher). Each block is a no-op unless its flag enables it.
    if args.ewc_lambda > 0:
        if args.ewc_load:
            fisher, optpar = ewc_mod.load_ewc(args.ewc_load, device="cuda")
            live = {n for n, p in model.named_parameters() if p.requires_grad}
            cov = len(live & set(fisher)) / max(1, len(live))
            assert cov > 0.99, (
                f"fisher/param name mismatch: coverage {cov:.2%} "
                f"(fisher {len(fisher)} vs live {len(live)})")
            trainer.fisher, trainer.optpar = fisher, optpar
            # Step-0 sanity: we resumed the exact adapter that produced optpar, so
            # theta == theta* and the penalty must be ~0 (plan sec 3.2 gate).
            with torch.no_grad():
                pen0 = float(ewc_mod.ewc_penalty(model.named_parameters(),
                                                 fisher, optpar, args.ewc_lambda))
            assert pen0 < 1e-2, (
                f"EWC step-0 penalty {pen0} not ~0: theta != theta*, the resumed "
                "adapter does not match the saved optpar")
            print(f"[train] EWC on: lambda={args.ewc_lambda} coverage={cov:.2%} "
                  f"step0_penalty={pen0:.2e}", flush=True)
        else:
            print("[train] EWC: no --ewc_load (stage 1) -> penalty skipped",
                  flush=True)

    if args.olora_weight > 0:
        if args.resume_adapter:
            trainer.olora_prev = olora_mod.snapshot_lora_A(model)
            assert trainer.olora_prev, "olora: no LoRA-A params found to anchor"
            print(f"[train] O-LoRA on: weight={args.olora_weight} "
                  f"anchored {len(trainer.olora_prev)} LoRA-A matrices", flush=True)
        else:
            print("[train] O-LoRA: no --resume_adapter (stage 1) -> penalty skipped",
                  flush=True)

    if args.lwf_alpha > 0:
        if args.resume_adapter:
            cache = lwf_mod.build_teacher_cache(
                model, ds, collator, args.lwf_cache_n, args.micro_batch,
                args.lwf_top_k, device="cuda")
            assert cache, "lwf: teacher cache empty (no supervised positions?)"
            trainer.train_dataset = lwf_mod.LwFDataset(ds, cache)
            trainer.data_collator = lwf_mod.LwFCollator(collator)
            trainer.lwf_active = True
            print(f"[train] LwF on: alpha={args.lwf_alpha} T={args.lwf_temp} "
                  f"cached {len(cache)}/{args.lwf_cache_n} examples "
                  f"top_k={args.lwf_top_k}", flush=True)
        else:
            print("[train] LwF: no --resume_adapter (stage 1) -> KD skipped",
                  flush=True)

    result = trainer.train()

    if cnp is not None:
        # The hook must have fired exactly once per optimizer step; a silent
        # no-fire would be an unconstrained SEQ arm labelled "cnp".
        n_steps = int(trainer.state.global_step)
        assert n_steps > 0 and len(cnp.history) == n_steps, (
            f"CNP hook fired {len(cnp.history)} times for {n_steps} optimizer steps")
        n_proj = sum(1 for h in cnp.history if h["active"])
        print(f"[train] cnp: {cnp.n_refresh} refreshes, projected on {n_proj}/{n_steps} "
              f"steps, mean removed ratio "
              f"{sum(h['removed_ratio'] for h in cnp.history) / n_steps:.3f}", flush=True)

    if args.ewc_fisher_out:
        fisher, optpar = ewc_mod.compute_fisher(
            model, ds, collator, args.ewc_fisher_batches, args.micro_batch,
            device="cuda")
        ewc_mod.save_ewc(args.ewc_fisher_out, fisher, optpar)
        print(f"[train] EWC fisher saved: {len(fisher)} tensors over "
              f"{min(args.ewc_fisher_batches * args.micro_batch, len(ds))} "
              f"examples -> {args.ewc_fisher_out}", flush=True)

    final_dir = os.path.join(args.out, "adapter_final")
    model.save_pretrained(final_dir)
    log = {
        "args": vars(args),
        "total_steps": total_steps,
        "train_loss": result.training_loss,
        "n_examples": len(ds),
        "collator_truncated": collator.n_truncated,
        "collator_seen": collator.n_seen,
        "faith_history": trainer.faith_history,
        "crit_history": trainer.crit_history,
        "crit_base_source": crit.base_source if crit is not None else None,
        "pol_history": trainer.pol_history,
        "pol_eos_source": pol.ref_source if pol is not None else None,
        "gca_history": trainer.gca_history,
        "gca_join_report": gca.report if gca is not None else None,
        "cnp_history": cnp.history if cnp is not None else None,
        "cnp_n_refresh": cnp.n_refresh if cnp is not None else None,
        "cnp_refusal_base_source": cnp.ref_base_source if cnp is not None else None,
        "cnp_param_names": cnp.param_names if cnp is not None else None,
        "ewc_history": trainer.ewc_history,
        "olora_history": trainer.olora_history,
        "lwf_history": trainer.lwf_history,
        "loss_history": [
            {"step": h.get("step"), "loss": h.get("loss")}
            for h in trainer.state.log_history if "loss" in h
        ],
    }
    with open(os.path.join(args.out, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"[train] DONE loss={result.training_loss:.4f} -> {final_dir}", flush=True)


if __name__ == "__main__":
    main()
