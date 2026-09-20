#!/bin/bash
#SBATCH -J clh_fs_smoke
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 3:00:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
set -euo pipefail

ROOT=$HOME/cl-halluc
source $ROOT/env/bin/activate
export HF_HOME=$ROOT/hf_home
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_DISABLE_PROGRESS_BARS=1

D=$ROOT/data_ucit
SD=$ROOT/smoke_fs
rm -rf $SD
mkdir -p $SD

echo "=== smoke 1/6: UCIT smoke slice ==="
python $ROOT/code/fullstudy/ucit_prep.py --root $SD/data --smoke

echo "=== smoke 2/6: LLaVA 8-step train on UCIT ArxivQA slice ==="
python $ROOT/code/train_lora.py --backbone llava15 \
  --data $SD/data/tasks/ArxivQA/train.jsonl --data_root $SD/data \
  --out $SD/llava_stage --max_steps 8 --micro_batch 2 --eff_batch 8 --lr 1e-4

echo "=== smoke 3/6: Qwen 8-step train on same slice ==="
python $ROOT/code/train_lora.py --backbone qwen25vl \
  --data $SD/data/tasks/ArxivQA/train.jsonl --data_root $SD/data \
  --out $SD/qwen_stage --max_steps 8 --micro_batch 1 --eff_batch 8 --lr 1e-4

echo "=== smoke 4/6: Qwen anchor 4-step (token resolution + margin loss) ==="
# Grounding images resolve against --data_root (paths are "grounding/images/...").
# Training data and grounding must share ONE root: symlink grounding into the
# smoke data root so both resolve. (Full study does the same in sb_fs_prep.)
ln -sfn $ROOT/data/grounding $SD/data/grounding
python $ROOT/code/train_lora.py --backbone qwen25vl \
  --data $SD/data/tasks/ArxivQA/train.jsonl --data_root $SD/data \
  --out $SD/qwen_anchor --max_steps 4 --micro_batch 1 --eff_batch 8 --lr 1e-4 \
  --faith_pairs $SD/data/grounding/grounding_pairs.jsonl \
  --faith_weight 0.1 --faith_k 2 --faith_margin 2.0
python - <<'PYEOF'
import json
log = json.load(open("/home/alexmueller/cl-halluc/smoke_fs/qwen_anchor/training_log.json"))
print("qwen anchor log ok:", list(log.keys())[:5])
PYEOF

echo "=== smoke 5/6: Qwen eval_gen micro (6 POPE prompts) ==="
head -6 $ROOT/data/pope/prompts.jsonl > $SD/pope6.jsonl
python $ROOT/code/eval_gen.py --backbone qwen25vl --prompts $SD/pope6.jsonl \
  --data_root $ROOT/data --out $SD/pope6_gen.jsonl --max_new_tokens 16 --batch 3
python - <<'PYEOF'
rows = [l for l in open("/home/alexmueller/cl-halluc/smoke_fs/pope6_gen.jsonl")]
assert len(rows) == 6, f"expected 6 gen rows, got {len(rows)}"
import json
outs = [json.loads(l)["output"] for l in rows]
print("qwen pope outputs:", outs)
assert any(o.strip() for o in outs), "all-empty generations"
PYEOF

echo "=== smoke 6/6: AMBER scorer on 5 real generations (NON-FATAL) ==="
set +e
if [ ! -f $D/amber/prompts.jsonl ]; then
  echo "AMBER prompts absent - skipping AMBER smoke (scorer is fixture-tested separately)"
else
head -5 $D/amber/prompts.jsonl > $SD/amber5.jsonl
python $ROOT/code/eval_gen.py --backbone llava15 --prompts $SD/amber5.jsonl \
  --data_root $D --out $SD/amber5_gen.jsonl --max_new_tokens 128 --batch 5
python $ROOT/code/metrics_amber.py --gen $SD/amber5_gen.jsonl \
  --amber_data $D/amber --out_prefix $SD/amber5 --ckpt SMOKE
cat $SD/amber5.json 2>/dev/null || cat $SD/amber5*.json
fi
set -e

echo "=== smoke 7/9: EWC cross-stage (Fisher save at S1, penalty load at S2) ==="
python $ROOT/code/train_lora.py --backbone llava15 \
  --data $SD/data/tasks/ArxivQA/train.jsonl --data_root $SD/data \
  --out $SD/ewc_s1 --max_steps 8 --micro_batch 2 --eff_batch 8 --lr 1e-4 \
  --ewc_fisher_out $SD/ewc_s1 --ewc_fisher_batches 4
test -f $SD/ewc_s1/fisher.pt && test -f $SD/ewc_s1/optpar.pt
python $ROOT/code/train_lora.py --backbone llava15 \
  --data $SD/data/tasks/CLEVR-Math/train.jsonl --data_root $SD/data \
  --out $SD/ewc_s2 --resume_adapter $SD/ewc_s1/adapter_final \
  --max_steps 8 --micro_batch 2 --eff_batch 8 --lr 1e-4 \
  --ewc_lambda 0.5 --ewc_load $SD/ewc_s1 --ewc_fisher_out $SD/ewc_s2 --ewc_fisher_batches 4

# O-LoRA DROPPED (2026-09-05): its orthogonal-subspace mechanism needs per-task
# LoRA blocks; with a single continuously-resumed adapter A_cur==A_prev at stage
# start, so the penalty is the adapter's self-Gram (exploded to ~1590 in smoke)
# and is meaningless. Architecturally incompatible; excluded and documented.
echo "=== smoke 8/8: LwF cross-stage (top-k KD toward resumed teacher) ==="
python $ROOT/code/train_lora.py --backbone llava15 \
  --data $SD/data/tasks/CLEVR-Math/train.jsonl --data_root $SD/data \
  --out $SD/lwf_s2 --resume_adapter $SD/ewc_s1/adapter_final \
  --max_steps 8 --micro_batch 2 --eff_batch 8 --lr 1e-4 \
  --lwf_alpha 0.1 --lwf_temp 2.0 --lwf_cache_n 8 --lwf_top_k 20
python - <<'PYEOF'
import json
for arm in ("ewc_s2", "lwf_s2"):
    log = json.load(open(f"/home/alexmueller/cl-halluc/smoke_fs/{arm}/training_log.json"))
    hk = [k for k in log if k.endswith("_history")]
    print(f"{arm}: history keys {hk}")
print("baseline arms cross-stage OK")
PYEOF

echo FS_SMOKE_OK
