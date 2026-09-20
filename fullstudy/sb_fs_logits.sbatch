#!/bin/bash
#SBATCH -J clh_fs_logits
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 12:00:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
set -euo pipefail

# Logit-level instrument for the criterion-drift mechanism: for each listed
# checkpoint, ONE teacher-forced forward per POPE prompt (no generation),
# recording z_yes / z_no / gap at the answer position (eval_gen.py
# --dump_logits). Then, per RUNTAG, the additive-bias test + post-hoc
# correction baseline (analysis/logit_bias_analysis.py) against the text-based
# results_fs cells.
#
# Env (set in the shell, submit with --export=ALL; values contain spaces, and
# sbatch's --export splits NAME=value pairs on commas, so do NOT inline lists):
#   BACKBONE    llava15 | qwen25vl                                     (required)
#   CKPT_LIST   whitespace-separated  label=adapter_path  entries; adapter_path
#               "none" = the raw backbone, e.g.
#               "llava15_base=none fsL_seq_o1_s17_k3=/home/alexmueller/cl-halluc/runs_fs/fsL_seq_o1_s17/S3/adapter_final"
#   RUNTAGS     optional whitespace-separated RUNTAGs; each expands to
#               <RUNTAG>_k<K>=runs_fs/<RUNTAG>/S<K>/adapter_final for every
#               stage K whose adapter OR eval cell exists (deleted adapters are
#               reported and skipped, never silently dropped).
#   DUMP_BATCH  optional batch override (default: backbone-aware, see below).
#   NO_ANALYSIS=1  skip the analysis step.
# Outputs: results_fs_logits/<label>/pope_logits.jsonl (+ DUMP_DONE marker;
# resumable), results_fs_logits/analysis/<RUNTAG>.{json,md,_fig.png}.
#
# Adapters are deleted after their eval job (sb_fs_eval.sbatch) unless the cell
# is an endpoint-bench cell (KEEP_FINAL) -- the canonical fsL_seq_o1_s17 /
# fsL_anchor_o1_s17 / fsL_joint_o1_s17 keep theirs until their bench runs.
# Missing adapters are SKIPPED with a message; the job still succeeds for the
# rest, so run this BEFORE the bench job cleans them.
ROOT=$HOME/cl-halluc
source $ROOT/env/bin/activate
export HF_HOME=$ROOT/hf_home
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DP=$ROOT/data                 # POPE prompts (pilot assets, COCO instrument)
CODE=$ROOT/code
OUTROOT=$ROOT/results_fs_logits
PROMPTS=$DP/pope/prompts.jsonl
: "${BACKBONE:?set BACKBONE=llava15|qwen25vl}"
[ -f "$PROMPTS" ] || { echo "GATE FAIL: no POPE prompts at $PROMPTS"; exit 1; }

# Backbone-aware batch: a single forward over a ~600-token (LLaVA: 576 image
# tokens) / <=800-token (Qwen, 768 visual cap) prompt keeps full-vocab logits
# for every position, so the batch is bounded by (B x T x V) logits, not by
# generation; eval_gen's OOM-halving self-recovers from any overshoot.
if [ "$BACKBONE" = "qwen25vl" ]; then DB=8; else DB=16; fi
DB=${DUMP_BATCH:-$DB}

# --- expand RUNTAGS into label=adapter entries ---------------------------------
ENTRIES="${CKPT_LIST:-}"
for RT in ${RUNTAGS:-}; do
  R=$ROOT/runs_fs/$RT
  K=1
  while [ -d $R/S$K ] || [ -d $ROOT/results_fs/${RT}_k$K ]; do
    ENTRIES="$ENTRIES ${RT}_k$K=$R/S$K/adapter_final"
    K=$((K+1))
  done
  [ $K -eq 1 ] && echo "WARN: RUNTAG $RT has neither runs_fs/$RT/S1 nor results_fs/${RT}_k1 -- nothing to dump"
done
[ -n "$ENTRIES" ] || { echo "GATE FAIL: nothing to do (set CKPT_LIST and/or RUNTAGS)"; exit 1; }

# --- dump ------------------------------------------------------------------------
N_OK=0; N_SKIP=0
for E in $ENTRIES; do
  LABEL=${E%%=*}; ADAPTER=${E#*=}
  OUT=$OUTROOT/$LABEL
  if [ -f $OUT/DUMP_DONE ]; then echo "skip done $LABEL"; N_OK=$((N_OK+1)); continue; fi
  AD=""
  if [ "$ADAPTER" != "none" ]; then
    if [ ! -f "$ADAPTER/adapter_config.json" ]; then
      echo "SKIP $LABEL: adapter missing at $ADAPTER (deleted after its eval job? only KEEP_FINAL/bench cells keep adapters)"
      N_SKIP=$((N_SKIP+1)); continue
    fi
    AD="--adapter $ADAPTER"
  fi
  mkdir -p $OUT
  echo "=== logits $LABEL adapter=$ADAPTER batch=$DB ==="
  python $CODE/eval_gen.py --dump_logits --backbone $BACKBONE --prompts $PROMPTS \
    --data_root $DP $AD --out $OUT/pope_logits.jsonl --batch $DB
  N_EXP=$(wc -l < $PROMPTS); N_GOT=$(wc -l < $OUT/pope_logits.jsonl)
  if [ "$N_GOT" -lt "$N_EXP" ]; then
    echo "GATE FAIL: $LABEL wrote $N_GOT/$N_EXP rows"; exit 1
  fi
  touch $OUT/DUMP_DONE; N_OK=$((N_OK+1))
done
echo "dumps: $N_OK done, $N_SKIP skipped (missing adapters)"

# --- analysis per RUNTAG (non-fatal; needs the base dump + text cells) -------------
if [ "${NO_ANALYSIS:-0}" != "1" ] && [ -n "${RUNTAGS:-}" ]; then
  mkdir -p $OUTROOT/analysis
  set +e
  for RT in ${RUNTAGS:-}; do
    if [ ! -f $OUTROOT/${BACKBONE}_base/DUMP_DONE ]; then
      echo "ANALYSIS_SKIPPED ($RT): no base dump at $OUTROOT/${BACKBONE}_base (add ${BACKBONE}_base=none to CKPT_LIST)"; continue
    fi
    python $CODE/analysis/logit_bias_analysis.py --logits_dir $OUTROOT \
      --results_fs $ROOT/results_fs --backbone $BACKBONE --runtag $RT \
      --out_prefix $OUTROOT/analysis/$RT \
      && echo "ANALYSIS_OK ($RT)" || echo "ANALYSIS_FAILED ($RT)"
  done
  set -e
fi
echo FS_LOGITS_OK_$BACKBONE
