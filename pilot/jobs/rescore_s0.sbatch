#!/bin/bash
#SBATCH -J clh_rescore_S0
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
set -euo pipefail
ROOT=$HOME/cl-halluc
source $ROOT/env/bin/activate
export PYTHONUNBUFFERED=1
OUT=$ROOT/results/S0
python $ROOT/code/metrics_pope.py --gen $OUT/pope_gen.jsonl --out_prefix $OUT/pope --ckpt S0
python $ROOT/code/metrics_chair.py --gen $OUT/chair_gen.jsonl --coco_gt $ROOT/data/chair/coco_gt.json --out_prefix $OUT/chair --ckpt S0
for T in scienceqa textvqa flickr vizwiz; do
  python $ROOT/code/metrics_task.py --gen $OUT/${T}_gen.jsonl --task $T --out_prefix $OUT/task_$T --ckpt S0
done
echo RESCORE_S0_OK
