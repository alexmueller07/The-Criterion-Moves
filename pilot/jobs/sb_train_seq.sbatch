#!/bin/bash
#SBATCH -J clh_seq
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 48:00:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
set -euo pipefail

ROOT=$HOME/cl-halluc
source $ROOT/env/bin/activate
export HF_HOME=$ROOT/hf_home
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

D=$ROOT/data
R=$ROOT/runs/seq
mkdir -p $R

TASKS="scienceqa textvqa flickr vizwiz"
PREV=""
K=0
for T in $TASKS; do
  K=$((K+1))
  OUT=$R/S$K
  if [ -f $OUT/adapter_final/adapter_config.json ]; then
    echo "S$K already done, skipping"
    PREV=$OUT/adapter_final
    continue
  fi
  RESUME_ARGS=""
  if [ -n "$PREV" ]; then
    RESUME_ARGS="--resume_adapter $PREV"
  fi
  echo "=== SEQ stage $K task $T ==="
  python $ROOT/code/train_lora.py \
    --data $D/tasks/$T/train.jsonl \
    --data_root $D \
    --out $OUT \
    $RESUME_ARGS \
    --epochs 1.0 --micro_batch 2 --eff_batch 64 --lr 1e-4
  PREV=$OUT/adapter_final
done
echo SEQ_OK
