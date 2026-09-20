#!/bin/bash
#SBATCH -J clh_policy_prep
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 1:30:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
# Prep + smoke for the multi-axis policy-anchor arms (policy / policy_abst /
# policy_eos): build the caption anchor set the EOS term needs, then a 4-step
# LLaVA smoke of the full policy composition. Emits POLICY_PREP_OK on success;
# the local waiter inserts fullstudy/queue_policy_lines.txt into the queue then.
set -euo pipefail
ROOT=$HOME/cl-halluc; cd $ROOT && source env/bin/activate
export HF_HOME=$ROOT/hf_home PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=$ROOT/data
echo "=== 1/2 caption anchor set (grounding images x COCO captions; written last = report) ==="
[ -f $D/grounding/caption_anchor.jsonl ] || python3 code/method/build_caption_anchor.py --root $D --n 600
wc -l $D/grounding/caption_anchor.jsonl; head -c 300 $D/grounding/caption_anchor.jsonl; echo
echo "=== 2/2 smoke: policy = anchor + abstention + EOS (LLaVA, 4 steps) ==="
SD=$ROOT/smoke_pol; mkdir -p $SD; head -64 $D/tasks/scienceqa/train.jsonl > $SD/slice.jsonl
python code/train_lora.py --backbone llava15 --data $SD/slice.jsonl --data_root $D --out $SD/policy \
  --max_steps 4 --micro_batch 2 --eff_batch 8 --lr 1e-4 \
  --faith_pairs $D/grounding/grounding_pairs.jsonl --faith_weight 0.1 --faith_k 2 --faith_margin 2.0 \
  --pol_weight_abstain 0.1 --pol_weight_eos 0.1 --pol_k 2 --pol_captions $D/grounding/caption_anchor.jsonl
python3 -c "import json;l=json.load(open('$SD/policy/training_log.json'));h=l.get('pol_history');assert h,'no pol_history';print('pol_history',h[:3])"
rm -rf $SD/policy
echo POLICY_PREP_OK
