#!/bin/bash
#SBATCH -J clh_seq_er
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
set -euo pipefail
ROOT=$HOME/cl-halluc
source $ROOT/env/bin/activate
export HF_HOME=$ROOT/hf_home
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
D=$ROOT/data
R=$ROOT/runs/seq_er
mkdir -p $R $D/er
python - <<PYEOF
import json, random
for t in ("scienceqa","textvqa","flickr","vizwiz"):
    rows=[json.loads(l) for l in open(f"$D/tasks/{t}/train.jsonl")]
    random.Random(17).shuffle(rows)
    with open(f"$D/er/{t}_100.jsonl","w") as f:
        for r in rows[:100]:
            f.write(json.dumps(r)+"\n")
print("er buffers built")
PYEOF
TASKS="scienceqa textvqa flickr vizwiz"
PREV=""
K=0
BUF=""
for T in $TASKS; do
  K=$((K+1))
  OUT=$R/S$K
  DATA=$D/tasks/$T/train.jsonl$BUF
  if [ -f $OUT/adapter_final/adapter_config.json ]; then
    PREV=$OUT/adapter_final
    BUF=$BUF,$D/er/${T}_100.jsonl
    continue
  fi
  RESUME_ARGS=""
  if [ -n "$PREV" ]; then
    RESUME_ARGS="--resume_adapter $PREV"
  fi
  echo "=== SEQ-ER stage $K task $T data=$DATA ==="
  python $ROOT/code/train_lora.py \
    --data $DATA \
    --data_root $D \
    --out $OUT \
    $RESUME_ARGS \
    --epochs 1.0 --micro_batch 2 --eff_batch 64 --lr 1e-4
  PREV=$OUT/adapter_final
  BUF=$BUF,$D/er/${T}_100.jsonl
done
echo SEQ_ER_OK
