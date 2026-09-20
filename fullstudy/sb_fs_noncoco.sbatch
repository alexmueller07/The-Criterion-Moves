#!/bin/bash
#SBATCH -J clh_fs_noncoco
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 4:00:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
set -euo pipefail
# Non-COCO (Open Images) generative CHAIR eval on one arm's final adapter —
# the confound control: images the model never saw, non-COCO. Env: RUNTAG, BACKBONE.
ROOT=$HOME/cl-halluc; source $ROOT/env/bin/activate
export HF_HOME=$ROOT/hf_home PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NC=$ROOT/data_ucit/noncoco
if [ ! -f $NC/prompts.jsonl ]; then echo "NONCOCO_DATA_ABSENT"; exit 0; fi
if [ "${BASE_ONLY:-0}" = "1" ]; then LABEL=${BACKBONE}_base; AD="";
else R=$ROOT/runs_fs/$RUNTAG
  if [ -d $R/joint ]; then A=$R/joint/adapter_final; else K=1; while [ -d $R/S$((K+1)) ]; do K=$((K+1)); done; A=$R/S$K/adapter_final; fi
  LABEL=${RUNTAG}_nc; AD="--adapter $A"; fi
OUT=$ROOT/results_fs_noncoco/$LABEL; mkdir -p $OUT
[ -f $OUT/NC_DONE ] && { echo "skip $LABEL"; exit 0; }
python $ROOT/code/eval_gen.py --backbone $BACKBONE --prompts $NC/prompts.jsonl \
  --data_root $NC $AD --out $OUT/nc_gen.jsonl --max_new_tokens 512 --batch 6
python $ROOT/code/metrics_chair.py --gen $OUT/nc_gen.jsonl --coco_gt $NC/gt.json \
  --out_prefix $OUT/nc --ckpt $LABEL
touch $OUT/NC_DONE; echo FS_NONCOCO_OK_$LABEL
