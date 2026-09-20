#!/bin/bash
#SBATCH -J clh_fsmoke
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 03:00:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
set -euo pipefail
ROOT=$HOME/cl-halluc
source $ROOT/env/bin/activate
export HF_HOME=$ROOT/hf_home
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf $ROOT/runs/faith_smoke0 $ROOT/runs/faith_smoke1
python $ROOT/code/train_lora.py \
  --data $ROOT/data/tasks/scienceqa/train.jsonl --data_root $ROOT/data \
  --out $ROOT/runs/faith_smoke0 --max_steps 10 --micro_batch 2 --eff_batch 8
python $ROOT/code/train_lora.py \
  --data $ROOT/data/tasks/scienceqa/train.jsonl --data_root $ROOT/data \
  --out $ROOT/runs/faith_smoke1 --max_steps 10 --micro_batch 2 --eff_batch 8 \
  --faith_pairs $ROOT/data/grounding/grounding_pairs.jsonl \
  --faith_weight 0.1 --faith_k 4 --faith_margin 2.0
python $ROOT/code/method/check_faith_smoke.py $ROOT/runs/faith_smoke0/training_log.json $ROOT/runs/faith_smoke1/training_log.json
echo FSMOKE_OK
