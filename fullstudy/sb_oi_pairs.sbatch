#!/bin/bash
#SBATCH -J clh_oi_pairs
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH -c 8
#SBATCH --mem=12G
#SBATCH -t 2:00:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
# Out-of-domain anchor pairs from Open Images TRAIN (disjoint from the Open Images
# VALIDATION non-COCO eval; asserted against data_ucit/noncoco/prompts.jsonl).
# Output: data/grounding_oi/{grounding_pairs.jsonl,images/,masked/,report.json}.
set -euo pipefail
ROOT=$HOME/cl-halluc; cd $ROOT && source env/bin/activate
[ -f data_ucit/noncoco/prompts.jsonl ] || { echo "noncoco prompts missing (build the non-COCO eval first)"; exit 1; }
[ -f data/grounding_oi/report.json ] && { echo "already built"; exit 0; }
python code/fetch_oi_train_pairs.py --root $ROOT/data --subdir grounding_oi \
  --exclude_prompts $ROOT/data_ucit/noncoco/prompts.jsonl --n 2000 --seed 17 --workers 8 --rm_src
wc -l data/grounding_oi/grounding_pairs.jsonl; python3 -c "import json;r=json.load(open('data/grounding_oi/report.json'));print({k:r[k] for k in list(r)[:8]})"
echo OI_PAIRS_OK
