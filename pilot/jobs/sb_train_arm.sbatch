#!/bin/bash
#SBATCH -J clh_arm
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 30:00:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
set -euo pipefail

ROOT=$HOME/cl-halluc
source $ROOT/env/bin/activate
export HF_HOME=$ROOT/hf_home
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUNTAG=${RUNTAG:?set RUNTAG}
TASKS=${TASKS:-scienceqa textvqa flickr vizwiz}
RUN_SEED=${RUN_SEED:-17}
FAITH=${FAITH:-0}

D=$ROOT/data
R=$ROOT/runs/$RUNTAG
mkdir -p $R

FAITH_ARGS=""
if [ "$FAITH" = "1" ]; then
  FAITH_ARGS="--faith_pairs $D/grounding/grounding_pairs.jsonl --faith_weight 0.1 --faith_k 4 --faith_margin 2.0"
fi

PREV=""
K=0
for T in $TASKS; do
  K=$((K+1))
  OUT=$R/S$K
  if [ -f $OUT/adapter_final/adapter_config.json ]; then
    PREV=$OUT/adapter_final
    continue
  fi
  RESUME_ARGS=""
  if [ -n "$PREV" ]; then
    RESUME_ARGS="--resume_adapter $PREV"
  fi
  echo "=== ARM $RUNTAG stage $K task $T seed $RUN_SEED faith=$FAITH ==="
  python $ROOT/code/train_lora.py \
    --data $D/tasks/$T/train.jsonl \
    --data_root $D \
    --out $OUT \
    $RESUME_ARGS \
    $FAITH_ARGS \
    --run_seed $RUN_SEED \
    --epochs 1.0 --micro_batch 2 --eff_batch 64 --lr 1e-4
  PREV=$OUT/adapter_final
done
echo ARM_OK_$RUNTAG
