# The Criterion Moves More Than the Curve

Code, pre-registration and machine readouts for *The Criterion Moves More Than the
Curve: A Signal-Detection Account of Hallucination Under Continual Instruction
Tuning* (ICLR 2027 submission).

A hallucination benchmark reports one number, and that number cannot say what
changed: a model that has become worse at telling present objects from absent
ones and a model that has only changed how readily it answers "yes" can produce
the same score. This project reads signal detection theory's discriminability
`d'` and decision criterion `c` after every stage of a continual instruction
tuning sequence, and separates the two.

## Start here: verify the numbers yourself

Everything in the paper's result tables is re-derivable on a CPU in seconds, with
no GPU and no model checkpoints.

```bash
pip install -r requirements.txt
python3 analysis/independent_headline_checks.py
```

That script re-derives 189 quantities from the released readouts — the z-ROC
fits and the model comparison, the task-versus-depth decomposition over the
independent ordering-by-position units, the single-task controls, the second
benchmark (MLLM-CL DCL), the replay and third joint cells, both main tables, and
the parser audit — using its own z-transform, its own OLS and its own sums,
sharing no code with the scripts that produced them. It prints the
paper's value beside the recomputed one for each, and exits non-zero on any
mismatch.

Figures regenerate the same way, from the same readouts:

```bash
python3 analysis/make_fig_concept.py             # Figure 1 (also writes an editable fig_concept.pptx)
python3 analysis/make_fig_trajectory_main.py     # Figure 2
python3 analysis/make_fig_zroc.py                # Figure 3
python3 analysis/make_paper_figures.py --diag analysis/diag --out analysis/out
```

The remaining appendix figures (`analysis/make_figures.py`, and the robustness
figure in `analysis/make_diag_figures.py`) recompute from the raw per-checkpoint
generation logs, which are available on request (see below).

## What is in here

| Path | Contents |
|---|---|
| `code/` | Training and evaluation harness: LoRA fine-tuning, per-stage generation, the scorers, the EWC and LwF baselines, the anchor probe, and the gate checks that refuse to run on a degenerate setup |
| `analysis/` | Analysis and figure scripts, the machine readouts in `analysis/readout/`, per-diagnostic readouts in `analysis/diag/`, the figures in `analysis/out/`, and the test suite with its fixtures |
| `fullstudy/` | The full-study pre-registration with its dated amendment log, plus launch and readout tooling |
| `pilot/` | The pilot-suite pre-registration and its scorers |

Two scripts are deliberately independent re-implementations rather than reuses:
`analysis/independent_headline_checks.py` and
`analysis/independent_zroc_modelcomp_check.py`. They exist so that the reported
values are checked by code that could disagree with the pipeline, and they are
released so that a reader can run that check rather than take it on trust.

## What is not in here

The raw per-item generation logs (roughly 0.5 GB of JSONL, one record per POPE
question per checkpoint, with audit fields) are archived separately and available
on request. The readouts in `analysis/readout/` are the distilled form and are
sufficient to re-derive every number in the paper.

Model weights and adapters are not included. The backbones are the public
`llava-hf/llava-1.5-7b-hf` and `Qwen/Qwen2.5-VL-7B-Instruct`.

## The pre-registration is part of the record

`fullstudy/FULLSTUDY_PREREG.md` is a dated log, not a tidied-up document. It
contains the endpoints as they were registered before the study ran, every
amendment with its date, and the claims that were withdrawn along with the
evidence that killed them. Several results in the paper are reported as failures
of their own pre-registered endpoints, and the log is where that is auditable.

Claims that were advanced and later withdrawn are listed in the paper's appendix
with their corrections, rather than removed.

## Reproducing the runs

The analysis above needs no GPU. Reproducing the runs themselves does:
`code/train_lora.py` and `code/eval_gen.py` drive one stage of training and one
checkpoint of evaluation respectively; `fullstudy/launch_matrix.sh` shows how the
cells were enumerated. Each cell is roughly 6.5 GPU-hours on one RTX 4090 for the
six-task sequence, about 4 to 4.5 hours of training plus 2.2 hours of evaluation.

Scoring is deliberately format-robust and never coerces an unparseable answer to
a label; the exclusion counts are reported rather than hidden, and
`analysis/score_q5_cells.py` refuses to report a criterion at all when the
parseable share of answers falls below 95 percent, because a criterion computed
from the answers that still complied with the expected format is a criterion of
a selected subsample.

## Citation

Please cite the paper. A BibTeX entry will be added here once the submission has
a public record.
