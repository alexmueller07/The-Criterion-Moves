#!/bin/bash
#SBATCH -J clh_joint
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
R=$ROOT/runs/joint
mkdir -p $R

python $ROOT/code/train_lora.py \
  --data $D/tasks/scienceqa/train.jsonl,$D/tasks/textvqa/train.jsonl,$D/tasks/flickr/train.jsonl,$D/tasks/vizwiz/train.jsonl \
  --data_root $D \
  --out $R \
  --joint_shuffle \
  --checkpoint_steps 90,215,340 \
  --epochs 1.0 --micro_batch 2 --eff_batch 64 --lr 1e-4
echo JOINT_OK
