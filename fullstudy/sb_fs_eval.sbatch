#!/bin/bash
#SBATCH -J clh_fs_eval
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 8
#SBATCH --mem=28G
#SBATCH -t 24:00:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
# --mem right-sized 2026-09-09: RAM (not GPUs) is the binding constraint on vgi2
# (122G, ~90G held by other lanes). The prep smokes ran the identical LLaVA
# training path -- model load + steps + anchor/crit/gca losses -- at 24G, so 32G
# is the evidence-backed ask with headroom. The [PEAKRSS] sampler logs the truth.
set -euo pipefail

# Evaluates EVERY checkpoint of one full-study run in a single job.
# Env: RUNTAG (runs_fs/<RUNTAG>), BACKBONE. Optional BASE_ONLY=1 evaluates the
# raw backbone once into results_fs/<BACKBONE>_base.
ROOT=$HOME/cl-halluc
source $ROOT/env/bin/activate
export HF_HOME=$ROOT/hf_home
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DP=$ROOT/data          # pilot assets: POPE + CHAIR (COCO instruments, reused)
DU=${DATADIR:-$ROOT/data_ucit}   # task-val assets (data_ucit or pilot data via DATADIR)
MANIFEST=${MANIFEST:-$DU/ucit_manifest.json}
CODE=$ROOT/code

# Backbone-aware generation batch sizes: Qwen2.5-VL (8.45B) OOMs on 24GB at the
# LLaVA batch sizes for long-caption (512-token) generation (baseQ eval,
# 2026-09-06). Short yes/no gen tolerates a larger batch.
if [ "$BACKBONE" = "qwen25vl" ]; then GB_SHORT=6; GB_LONG=2; else GB_SHORT=12; GB_LONG=6; fi

eval_ckpt () {
  local LABEL=$1
  local ADAPTER=$2
  local OUT=$ROOT/results_fs/$LABEL
  mkdir -p $OUT
  if [ -f $OUT/EVAL_DONE ]; then
    echo "skip done $LABEL"
    return
  fi
  local AD_ARGS=""
  if [ "$ADAPTER" != "none" ]; then
    AD_ARGS="--adapter $ADAPTER"
  fi
  echo "=== fs-eval $LABEL adapter=$ADAPTER ==="
  python $CODE/eval_gen.py --backbone $BACKBONE --prompts $DP/pope/prompts.jsonl \
    --data_root $DP $AD_ARGS --out $OUT/pope_gen.jsonl --max_new_tokens 16 --batch $GB_SHORT
  # Method-test arms (and the canonical LLaVA cells): dump POPE yes/no logits per
  # checkpoint -> additive-bias test + GE-degeneracy check (per-item dispersion of
  # g - g0) via analysis/logit_bias_analysis.py. Non-fatal; runs before cleanup.
  case "$RUNTAG" in mth_*|psL_*|lamF_*|fsL_seq_o1_s*|fsL_anchor_o1_s*)
    python $CODE/eval_gen.py --backbone $BACKBONE --prompts $DP/pope/prompts.jsonl \
      --data_root $DP $AD_ARGS --out $OUT/pope_logits.jsonl --dump_logits --batch $GB_SHORT \
      || echo "LOGITS_SKIPPED $LABEL"
    # blind twin (constant-gray images): the language-prior share of the criterion
    python $CODE/eval_gen.py --backbone $BACKBONE --prompts $DP/pope/prompts.jsonl \
      --data_root $DP $AD_ARGS --out $OUT/pope_logits_blind.jsonl --dump_logits --blind --batch $GB_SHORT \
      || echo "BLIND_LOGITS_SKIPPED $LABEL"
    # rephrased POPE template (the transfer test where train-time can beat post-hoc)
    if [ -f $DP/pope/prompts_rephrased.jsonl ]; then
      python $CODE/eval_gen.py --backbone $BACKBONE --prompts $DP/pope/prompts_rephrased.jsonl \
        --data_root $DP $AD_ARGS --out $OUT/pope_rephr_gen.jsonl --max_new_tokens 16 --batch $GB_SHORT \
        || echo "REPHR_SKIPPED $LABEL"
      python $CODE/eval_gen.py --backbone $BACKBONE --prompts $DP/pope/prompts_rephrased.jsonl \
        --data_root $DP $AD_ARGS --out $OUT/pope_rephr_logits.jsonl --dump_logits --batch $GB_SHORT \
        || echo "REPHR_LOGITS_SKIPPED $LABEL"
    fi
    # probe-set logit dumps (PCR transferred FROM the probe, never fitted on POPE)
    for PP in probe_prompts probe_oi_prompts probe_noise_prompts; do
      [ -f $DP/grounding/$PP.jsonl ] && { python $CODE/eval_gen.py --backbone $BACKBONE --prompts $DP/grounding/$PP.jsonl \
        --data_root $DP $AD_ARGS --out $OUT/${PP}_logits.jsonl --dump_logits --batch $GB_SHORT \
        || echo "PROBE_LOGITS_SKIPPED $LABEL $PP"; }
    done;;
  esac
  python $CODE/eval_gen.py --backbone $BACKBONE --prompts $DP/chair/images.jsonl \
    --data_root $DP $AD_ARGS --out $OUT/chair_gen.jsonl --max_new_tokens 512 --batch $GB_LONG
  if [ -f $DU/amber/prompts.jsonl ]; then
    python $CODE/eval_gen.py --backbone $BACKBONE --prompts $DU/amber/prompts.jsonl \
      --data_root $DU $AD_ARGS --out $OUT/amber_gen.jsonl --max_new_tokens 512 --batch $GB_LONG
  fi
  for T in $(python -c "import json;print(' '.join(json.load(open('$MANIFEST'))['tasks']))"); do
    python $CODE/eval_gen.py --backbone $BACKBONE --prompts $DU/tasks/$T/val.jsonl \
      --data_root $DU $AD_ARGS --out $OUT/${T}_gen.jsonl \
      --max_new_tokens $(python -c "import json;print(json.load(open('$MANIFEST'))['tasks']['$T']['max_new_tokens'])") \
      --batch $GB_SHORT
  done
  touch $OUT/EVAL_DONE
}

if [ "${BASE_ONLY:-0}" = "1" ]; then
  eval_ckpt ${BACKBONE}_base none
  echo FS_EVAL_OK_base
  exit 0
fi

R=$ROOT/runs_fs/$RUNTAG
if [ -d $R/joint ]; then
  for CK in $R/joint/ckpt_step*; do
    [ -d "$CK" ] || continue
    eval_ckpt ${RUNTAG}_$(basename $CK | sed s/ckpt_//) $CK
  done
  eval_ckpt ${RUNTAG}_final $R/joint/adapter_final
else
  K=1
  while [ -d $R/S$K ]; do
    eval_ckpt ${RUNTAG}_k$K $R/S$K/adapter_final
    K=$((K+1))
  done
fi
# Adapters are extracted into results_fs; delete this arm's ~1-2GB of LoRA
# weights now that every checkpoint is evaluated. Prevents the runs_fs pile-up
# that filled the shared disk (2026-09-08). Guarded: only remove if the last
# eval dir carries EVAL_DONE, and never touch anything outside this arm.
# NOTE: `ls a b` exits NON-ZERO if ANY argument is missing, even when another
# matched. Staged arms never have a _final marker, so the old combined test always
# failed and this self-cleanup NEVER fired -- runs_fs reached 97G of already-scored
# adapters and stalled the scheduler's disk gate (2026-09-09). Test each separately.
if ls $ROOT/results_fs/${RUNTAG}_k*/EVAL_DONE >/dev/null 2>&1 \
   || ls $ROOT/results_fs/${RUNTAG}_final/EVAL_DONE >/dev/null 2>&1; then
  if [ "${KEEP_FINAL:-0}" = "1" ]; then
    # Endpoint-bench cell: keep ONLY the final adapter (the bench job deletes
    # it when done via CLEAN_AFTER=1); drop every earlier stage checkpoint.
    if [ -d $R/joint ]; then
      find $R/joint -mindepth 1 -maxdepth 1 -name 'ckpt_step*' -exec rm -rf {} + 2>/dev/null
    else
      LAST=$((K-1))
      for ((J=1; J<LAST; J++)); do rm -rf $R/S$J; done
    fi
    echo "kept final adapter only for bench: $R"
  else
    rm -rf $R && echo "cleaned adapters $R"
  fi
fi
echo FS_EVAL_OK_$RUNTAG
