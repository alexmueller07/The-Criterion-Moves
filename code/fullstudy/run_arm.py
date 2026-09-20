"""Full-study arm driver: runs one arm's full stage sequence by invoking the
pilot harness (train_lora.py) once per stage with the right data, resume, and
loss flags. Stage order comes from ucit_manifest.json; every behavior matches
the pilot conventions (skip-if-done per stage, joint stage-boundary
checkpoint steps computed from actual stage sizes, ER buffers seeded at 17).

Arms: seq, joint, single:<task>, anchor, anchorlo (the R1 counterfactual-term
ablation: same anchor with its third term on the normalization-free log-odds
g = z_yes - z_no; design_notes/method_ideas_lit1.md sec.2.0), cecf, er[N], the label-free
criterion-preservation arms critp / anchorcrit (design_notes/
method_crit_preserve.md), the criterion-null-space projection arms cnp /
cnp_ref (design_notes/method_cnp.md; hard twin of critp on the same probe),
the multi-axis policy-anchor arms policy / policy_abst / policy_eos
(design_notes/method_multiaxis.md), the generative counterfactual token-anchor
arms gca / anchorgca (design_notes/method_gca.md), and the CL baselines ewc /
olora / lwf (design_notes/baselines_plan.md; see train_lora.py flags).
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--data_root", required=True)
ap.add_argument("--out_root", required=True)
ap.add_argument("--backbone", required=True)
ap.add_argument("--order", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--arm", required=True)
args = ap.parse_args()

man = json.load(open(args.manifest))
tasks = man["orders"][args.order]
sizes = {t: man["tasks"][t]["n_train"] for t in tasks}
CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TRAIN = os.path.join(CODE, "train_lora.py")
D = args.data_root
EFF_BATCH = 64
# Qwen2.5-VL (8.45B) OOMs at micro_batch 2 on high-resolution tasks (TextVQA
# stage 2, 2026-09-08; the same cause killed every earlier fsQ arm). The smoke
# validated Qwen at micro_batch 1; eff_batch is unchanged (accumulation adapts).
MICRO_BATCH = 1 if args.backbone == "qwen25vl" else 2


def train(out, data, resume=None, extra=None):
    if os.path.exists(os.path.join(out, "adapter_final", "adapter_config.json")):
        print(f"[arm] skip done stage: {out}", flush=True)
        return
    cmd = [sys.executable, TRAIN, "--data", data, "--data_root", D, "--out", out,
           "--backbone", args.backbone, "--run_seed", str(args.seed),
           "--epochs", "1.0", "--micro_batch", str(MICRO_BATCH), "--eff_batch", str(EFF_BATCH),
           "--lr", "1e-4"]
    if resume:
        cmd += ["--resume_adapter", resume]
    cmd += extra or []
    print("[arm] RUN:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def er_buffers(k):
    """Build (once) and return buffer files for tasks before stage index k."""
    n = int(args.arm[2:]) if args.arm.startswith("er") and len(args.arm) > 2 else 100
    paths = []
    for t in tasks[:k]:
        p = os.path.join(D, f"er{n}", f"{t}_{n}.jsonl")
        if not os.path.exists(p):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            rows = [l for l in open(os.path.join(D, "tasks", t, "train.jsonl"))]
            random.Random(17).shuffle(rows)
            with open(p + ".tmp", "w") as f:
                f.writelines(rows[:n])
            os.replace(p + ".tmp", p)
        paths.append(p)
    return paths


_FW = os.environ.get("FAITH_WEIGHT", "0.1")
_FM = os.environ.get("FAITH_MARGIN", "2.0")
# FAITH_PAIRS env: point the anchor at a different pair set (e.g. the Open Images
# out-of-domain pairs at data/grounding_oi/grounding_pairs.jsonl) without a new arm.
_FP = os.environ.get("FAITH_PAIRS", os.path.join(D, "grounding", "grounding_pairs.jsonl"))
anchor_extra = ["--faith_pairs", _FP,
                "--faith_weight", _FW, "--faith_k", "4", "--faith_margin", _FM]
cecf_extra = anchor_extra + ["--faith_ce_mode"]
# R1 ablation arm (design_notes/method_ideas_lit1.md sec.2.0; IC-VCO
# 2605.31312v1's partition-function critique): the anchor with its THIRD
# (counterfactual) term on the normalization-free within-image log-odds
# g = z_yes - z_no instead of the raw z_yes. Terms 1-2 (the symmetric decision
# margins) are untouched, so the criterion-freeze derivation carries over
# unchanged. The "anchor" arm never passes --faith_cf_form (train_lora defaults
# to "logit"), so its command line stays byte-identical to the replicated cells.
# FAITH_CF_FORM overrides this arm's form (e.g. =logit to re-run it as anchor).
_CF = os.environ.get("FAITH_CF_FORM", "logodds")
anchorlo_extra = anchor_extra + ["--faith_cf_form", _CF]
# Label-free criterion-preservation term (method/crit_preserve.py). The probe
# set comes from method/build_probe_set.py, which writes <root>/grounding/
# probe.jsonl by default; data_ucit/grounding is the symlink to the pilot
# grounding dir (sb_fs_prep.sbatch), so one file serves both data roots.
_CW = os.environ.get("CRIT_WEIGHT", "0.1")
_CK = os.environ.get("CRIT_K", "8")
_CM = os.environ.get("CRIT_MODE", "mean")
crit_extra = ["--crit_probe", os.environ.get("CRIT_PROBE", os.path.join(D, "grounding", "probe.jsonl")),
              "--crit_weight", _CW, "--crit_k", _CK, "--crit_mode", _CM]
# Criterion-null-space projection (method/cnp_project.py; design_notes/
# method_cnp.md): the HARD twin of critp on the SAME probe file. --crit_weight
# is pinned to 0 so the M1 term is skipped and the arm is projection-only (the
# soft/hard ladder is then two arms differing in one mechanism). CNP_T / CNP_K
# set the refresh period and probe items per direction estimate.
_CT = os.environ.get("CNP_T", "5")
_CKP = os.environ.get("CNP_K", "16")
cnp_extra = ["--crit_probe", os.path.join(D, "grounding", "probe.jsonl"),
             "--crit_weight", "0", "--crit_k", _CK,
             "--cnp", "--cnp_T", _CT, "--cnp_k", _CKP]
# Multi-axis policy anchor (method/policy_anchor.py; design_notes/
# method_multiaxis.md): the GT anchor stays the criterion term (anchor_extra,
# byte-identical to the anchor arm) and the abstention / EOS terms ride on top.
# The caption anchor set comes from method/build_caption_anchor.py, written
# next to the pairs file (<root>/grounding/caption_anchor.jsonl, so it follows
# FAITH_PAIRS); POL_CAPTIONS overrides it. POL_W_ABST / POL_W_EOS set the
# weights for the "policy" arm; the single-term arms pin the other weight to 0.
_PA = os.environ.get("POL_W_ABST", "0.1")
_PE = os.environ.get("POL_W_EOS", "0.1")
_PK = os.environ.get("POL_K", "8")
_PC = os.environ.get("POL_CAPTIONS",
                     os.path.join(os.path.dirname(_FP), "caption_anchor.jsonl"))


def pol_extra(w_abst, w_eos):
    ex = anchor_extra + ["--pol_weight_abstain", w_abst, "--pol_weight_eos", w_eos,
                         "--pol_k", _PK]
    if float(w_eos) != 0.0:
        ex += ["--pol_captions", _PC]
    return ex


# Generative counterfactual token anchor (method/gca_loss.py; design_notes/
# method_gca.md): caption-slot log-odds margins on the GT pairs. Pairs follow
# FAITH_PAIRS (GCA_PAIRS overrides); the caption set defaults to the pilot
# grounding dir's caption_anchor.jsonl (method/build_caption_anchor.py) and
# GCA_CAPTIONS overrides it. The join is on pair id, so a pair set whose ids
# are not in the caption file (e.g. the Open Images pairs) fails loudly at init.
_GW = os.environ.get("GCA_WEIGHT", "0.1")
_GK = os.environ.get("GCA_K", "4")
_GM = os.environ.get("GCA_MARGIN", "2.0")
_GP = os.environ.get("GCA_PAIRS", _FP)
_GC = os.environ.get("GCA_CAPTIONS", os.path.join(D, "grounding", "caption_anchor.jsonl"))
gca_extra = ["--gca_pairs", _GP, "--gca_captions", _GC,
             "--gca_weight", _GW, "--gca_k", _GK, "--gca_margin", _GM]


if args.arm == "joint":
    union = ",".join(os.path.join(D, "tasks", t, "train.jsonl") for t in tasks)
    steps, cum = 0, []
    for t in tasks:
        steps += math.ceil(sizes[t] / EFF_BATCH)
        cum.append(steps)
    marks = ",".join(str(s) for s in cum[:-1])
    train(os.path.join(args.out_root, "joint"), union,
          extra=["--joint_shuffle", "--checkpoint_steps", marks])
elif args.arm.startswith("single:"):
    t = args.arm.split(":", 1)[1]
    train(os.path.join(args.out_root, "S1"), os.path.join(D, "tasks", t, "train.jsonl"))
else:
    prev = None
    for k, t in enumerate(tasks):
        out = os.path.join(args.out_root, f"S{k + 1}")
        data = os.path.join(D, "tasks", t, "train.jsonl")
        extra = None
        if args.arm == "anchor":
            extra = anchor_extra
        elif args.arm == "anchorlo":
            # R1: anchor, counterfactual term on the log-odds (see above). The
            # anchor's replicated result is NOT touched; this is a sibling arm.
            extra = anchorlo_extra
        elif args.arm == "cecf":
            extra = cecf_extra
        elif args.arm == "critbal":
            # Criterion CORRECTION, not preservation: aim the yes-rate at 0.5 on
            # a balanced probe, which on a balanced probe is exactly c=0 and
            # needs no labels. Replaces the frozen-base target that lands the
            # anchor at c=+0.697 (worse than no method at every stage).
            extra = ["--crit_probe", os.environ.get(
                         "CRITBAL_PROBE",
                         os.path.join(D, "grounding", "probe_balanced.jsonl")),
                     "--crit_weight", os.environ.get("CRIT_WEIGHT", "1.0"),
                     "--crit_k", os.environ.get("CRIT_K", "8"),
                     "--crit_target", "balanced",
                     "--crit_bal_mode", os.environ.get("CRITBAL_MODE", "rate"),
                     "--crit_tau", os.environ.get("CRITBAL_TAU", "1.0")]
        elif args.arm == "critp":
            # Criterion preservation ONLY: no ground-truth anchor, no labels.
            extra = crit_extra
        elif args.arm == "anchorcrit":
            # GT anchor + criterion preservation (method_calibration_theory.md
            # sec.4: the margin protects d', the centering term removes the
            # anchor's residual conservative offset).
            extra = anchor_extra + crit_extra
        elif args.arm == "cnp":
            # Hard criterion constraint only: one-sided projection off the
            # criterion-statistic gradient; no labels, no loss term.
            extra = cnp_extra
        elif args.arm == "cnp_ref":
            # + the refusal-token direction (the leakage handle): projects off
            # span{u, n_ref}, one-sided per direction.
            extra = cnp_extra + ["--cnp_refusal"]
        elif args.arm == "policy":
            # Full composite: anchor (criterion) + abstention + EOS terms.
            extra = pol_extra(_PA, _PE)
        elif args.arm == "policy_abst":
            # Ablation: anchor + abstention term only (EOS weight pinned to 0).
            extra = pol_extra(_PA, "0")
        elif args.arm == "policy_eos":
            # Ablation: anchor + EOS/stopping-hazard term only (abstain pinned to 0).
            extra = pol_extra("0", _PE)

        elif args.arm == "gca_t1":

            # HALVA-style baseline IN OUR CL SETTING: GCA Term 1 only (slot decision).

            extra = gca_extra + ["--gca_terms", "decision"]

        elif args.arm == "gca_te":

            # See-or-Guess-style baseline IN OUR CL SETTING: GCA Term 2 only

            # (the GT-masked counterfactual term that vocab B found is crowded).

            extra = gca_extra + ["--gca_terms", "cf"]
        elif args.arm == "gca":
            # Generative counterfactual token anchor ONLY (no yes/no anchor):
            # isolates the caption-slot term's effect on CHAIR_i@60.
            extra = gca_extra
        elif args.arm == "anchorgca":
            # GT yes/no anchor + caption-slot anchor: the composite the design
            # note pre-registers against the anchor arm.
            extra = anchor_extra + gca_extra
        elif args.arm.startswith("er"):
            bufs = er_buffers(k)
            if bufs:
                data = data + "," + ",".join(bufs)
        elif args.arm == "ewc":
            # CoIN chain-EWC (corrected): lambda=0.5, Fisher over first 1000
            # examples (500 batches * micro_batch 2), anchored to the previous
            # stage only. fisher.pt/optpar.pt written into this stage's dir for
            # the next stage; stage 1 has no --ewc_load so the penalty is skipped.
            extra = ["--ewc_lambda", "0.5", "--ewc_fisher_batches", "500",
                     "--ewc_fisher_out", out]
            if prev:
                extra += ["--ewc_load", os.path.dirname(prev)]
        elif args.arm == "olora":
            # O-LoRA DROPPED from the study (2026-09-05): the orthogonal-subspace
            # mechanism requires per-task LoRA blocks. With a single adapter
            # resumed across stages, A_cur==A_prev at stage start, so the
            # orthogonality penalty is the adapter's self-Gram (~1590 in smoke)
            # and is meaningless. The flag/code remain for reference; the arm is
            # not launched. Baselines: EWC, LwF, ER, JOINT.
            extra = ["--olora_weight", "0.05"]
        elif args.arm == "lwf":
            # LwF offline top-k distillation toward the resumed teacher adapter;
            # CoIN alpha=0.1, T=2, 100 examples; no-op at stage 1 (no teacher).
            extra = ["--lwf_alpha", "0.1", "--lwf_temp", "2.0",
                     "--lwf_cache_n", "100", "--lwf_top_k", "20"]
        elif args.arm != "seq":
            raise SystemExit(f"unknown arm {args.arm}")
        train(out, data, resume=prev, extra=extra)
        prev = os.path.join(out, "adapter_final")
print("[arm] ALL STAGES DONE", flush=True)
