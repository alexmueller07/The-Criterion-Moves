#!/bin/bash
#SBATCH -J clh_mth2_prep
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 1:30:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
# Smoke for the CNP and GCA candidates (LLaVA, 4 steps each). Builds the caption anchor if absent.
set -euo pipefail
ROOT=$HOME/cl-halluc; cd $ROOT && source env/bin/activate
export HF_HOME=$ROOT/hf_home PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=$ROOT/data; SD=$ROOT/smoke_mth2; mkdir -p $SD; head -64 $D/tasks/scienceqa/train.jsonl > $SD/slice.jsonl
[ -f $D/grounding/caption_anchor.jsonl ] || python3 code/method/build_caption_anchor.py --root $D --n 600
[ -f $D/grounding/probe.jsonl ] || python3 code/method/build_probe_set.py --root $D --images_dir $D/grounding/images --n 600
echo "=== smoke CNP (crit probe, T=1, k=4, one-sided) ==="
python code/train_lora.py --backbone llava15 --data $SD/slice.jsonl --data_root $D --out $SD/cnp --max_steps 4 --micro_batch 2 --eff_batch 8 --lr 1e-4 \
  --crit_probe $D/grounding/probe.jsonl --crit_weight 0 --cnp --cnp_T 1 --cnp_k 4
python3 -c "import json;l=json.load(open('$SD/cnp/training_log.json'));h=l.get('cnp_history');assert h,'no cnp_history';print('cnp_history',h[:2])"
echo "=== smoke GCA ==="
python code/train_lora.py --backbone llava15 --data $SD/slice.jsonl --data_root $D --out $SD/gca --max_steps 4 --micro_batch 2 --eff_batch 8 --lr 1e-4 \
  --gca_pairs $D/grounding/grounding_pairs.jsonl --gca_captions $D/grounding/caption_anchor.jsonl --gca_weight 0.1 --gca_k 2
python3 -c "import json;l=json.load(open('$SD/gca/training_log.json'));h=l.get('gca_history');assert h,'no gca_history';print('gca_history',h[:2],'join',l.get('gca_join_report'))"
rm -rf $SD/cnp $SD/gca
echo MTH2_PREP_OK
