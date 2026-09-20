#!/bin/bash
#SBATCH -J clh_eval
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 16:00:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
set -euo pipefail

ROOT=$HOME/cl-halluc
source $ROOT/env/bin/activate
export HF_HOME=$ROOT/hf_home
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CKPT=${CKPT:?set CKPT label e.g. S0 S1 J2}
ADAPTER=${ADAPTER:-none}

D=$ROOT/data
OUT=$ROOT/results/$CKPT
mkdir -p $OUT

AD_ARGS=""
if [ "$ADAPTER" != "none" ]; then
  AD_ARGS="--adapter $ADAPTER"
fi

echo "=== eval $CKPT adapter=$ADAPTER ==="

python $ROOT/code/eval_gen.py --prompts $D/pope/prompts.jsonl --data_root $D \
  $AD_ARGS --out $OUT/pope_gen.jsonl --max_new_tokens 16 --batch 12
python $ROOT/code/metrics_pope.py --gen $OUT/pope_gen.jsonl \
  --out_prefix $OUT/pope --ckpt $CKPT

python $ROOT/code/eval_gen.py --prompts $D/chair/images.jsonl --data_root $D \
  $AD_ARGS --out $OUT/chair_gen.jsonl --max_new_tokens 512 --batch 6
python $ROOT/code/metrics_chair.py --gen $OUT/chair_gen.jsonl \
  --coco_gt $D/chair/coco_gt.json --out_prefix $OUT/chair --ckpt $CKPT

for T in scienceqa textvqa flickr vizwiz; do
  MNT=32
  if [ "$T" = "flickr" ]; then
    MNT=64
  fi
  python $ROOT/code/eval_gen.py --prompts $D/tasks/$T/val.jsonl --data_root $D \
    $AD_ARGS --out $OUT/${T}_gen.jsonl --max_new_tokens $MNT --batch 12
  python $ROOT/code/metrics_task.py --gen $OUT/${T}_gen.jsonl --task $T \
    --out_prefix $OUT/task_$T --ckpt $CKPT
done

echo EVAL_OK_$CKPT
