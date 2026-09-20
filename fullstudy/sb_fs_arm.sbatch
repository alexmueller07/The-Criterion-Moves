#!/bin/bash
#SBATCH -J clh_fs_arm
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 8
#SBATCH --mem=28G
#SBATCH -t 20:00:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
# --mem right-sized 2026-09-09: RAM (not GPUs) is the binding constraint on vgi2
# (122G, ~90G held by other lanes). The prep smokes ran the identical LLaVA
# training path -- model load + steps + anchor/crit/gca losses -- at 24G, so 32G
# is the evidence-backed ask with headroom. The [PEAKRSS] sampler logs the truth.
set -euo pipefail

# Full-study arm runner. Required env:
#   RUNTAG   unique run name (also results/runs namespace), e.g. fsL_seq_o1_s17
#   BACKBONE llava15 | qwen25vl
#   ORDER    o1 | o2 | o3            (task ordering from ucit_manifest.json)
#   SEED     17 | 23 | 31
#   ARM      seq | anchor | joint | er | er500 | cecf | single:<task>
ROOT=$HOME/cl-halluc
source $ROOT/env/bin/activate
export HF_HOME=$ROOT/hf_home
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

D=${DATADIR:-$ROOT/data_ucit}
MANIFEST=${MANIFEST:-$D/ucit_manifest.json}
R=$ROOT/runs_fs/$RUNTAG
mkdir -p $R

# Peak-RSS sampler: SLURM accounting does not record MaxRSS here, and --mem is
# the binding constraint on this node (48G asks starve the queue). Log the real
# peak so the request can be right-sized from evidence instead of guessed.
( PEAK=0; while :; do
    U=$(ps -o rss= -u $USER 2>/dev/null | awk '{s+=$1} END{print int(s/1024)}')
    [ -n "$U" ] && [ "$U" -gt "$PEAK" ] && { PEAK=$U; echo "[PEAKRSS] ${PEAK}MB"; }
    sleep 60
  done ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

python $ROOT/code/fullstudy/run_arm.py \
  --manifest $MANIFEST \
  --data_root $D \
  --out_root $R \
  --backbone $BACKBONE \
  --order "$ORDER" \
  --seed $SEED \
  --arm "$ARM"
echo FS_ARM_OK_$RUNTAG
