#!/bin/bash
#SBATCH -J clh_smoke
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 04:00:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
set -euo pipefail

ROOT=$HOME/cl-halluc
source $ROOT/env/bin/activate
export HF_HOME=$ROOT/hf_home
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SD=$ROOT/smokedata
RD=$ROOT/runs/smoke
rm -rf $RD
mkdir -p $SD $RD

python $ROOT/code/data_prep.py --root $SD --smoke --only scienceqa,pope

echo "=== smoke train stage1 ==="
python $ROOT/code/train_lora.py \
  --data $SD/tasks/scienceqa/train.jsonl \
  --data_root $SD \
  --out $RD/stage1 \
  --max_steps 8 --micro_batch 2 --eff_batch 8

echo "=== smoke train stage2 resume ==="
python $ROOT/code/train_lora.py \
  --data $SD/tasks/scienceqa/train.jsonl \
  --data_root $SD \
  --out $RD/stage2 \
  --resume_adapter $RD/stage1/adapter_final \
  --max_steps 4 --micro_batch 2 --eff_batch 8

echo "=== smoke eval gen ==="
python $ROOT/code/eval_gen.py \
  --prompts $SD/pope/prompts.jsonl \
  --data_root $SD \
  --adapter $RD/stage2/adapter_final \
  --out $RD/pope_gen.jsonl \
  --max_new_tokens 16 --batch 8 --limit 24

python $ROOT/code/metrics_pope.py --gen $RD/pope_gen.jsonl \
  --out_prefix $RD/pope_metrics --ckpt SMOKE

echo "=== smoke S0 eval gen (no adapter) ==="
python $ROOT/code/eval_gen.py \
  --prompts $SD/pope/prompts.jsonl \
  --data_root $SD \
  --out $RD/pope_gen_s0.jsonl \
  --max_new_tokens 16 --batch 8 --limit 24

python $ROOT/code/metrics_pope.py --gen $RD/pope_gen_s0.jsonl \
  --out_prefix $RD/pope_metrics_s0 --ckpt SMOKE_S0

echo SMOKE_OK
