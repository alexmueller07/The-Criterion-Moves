#!/bin/bash
#SBATCH -J clh_fs_prep
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
set -euo pipefail

ROOT=$HOME/cl-halluc
source $ROOT/env/bin/activate
export HF_HOME=$ROOT/hf_home
export PYTHONUNBUFFERED=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=1

D=$ROOT/data_ucit
mkdir -p $D

echo "=== 1/4 UCIT tasks + frozen manifest ==="
python $ROOT/code/fullstudy/ucit_prep.py --root $D

echo "=== 2/4 AMBER assets + images + prompts (NON-BLOCKING: secondary metric) ==="
set +e
python $ROOT/code/fetch_amber_assets.py --out $D/amber &&   python $ROOT/code/fullstudy/amber_images.py --out $D/amber   && echo AMBER_READY || echo "AMBER_SKIPPED (secondary metric; confirmatory endpoints unaffected)"
set -e

echo "=== 3/4 Qwen snapshot download (CPU) ==="
python - <<'PYEOF'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct")
print("qwen snapshot at", p)
PYEOF

echo "=== 3b/4 External hallucination benchmarks (NON-BLOCKING; endpoint SOTA table) ==="
set +e
# Object HalBench: self-contained (COCO GT from the pilot chair dir, no image dl).
python $ROOT/code/fetch_objhalbench.py --out_dir $D/objhal --coco_ann_dir $ROOT/data/chair \
  && echo OBJHAL_READY || echo "OBJHAL_SKIPPED"
# MME-Hallucination: data ships in the release zip.
python $ROOT/code/fetch_mme.py --out_dir $D/mme && echo MME_READY || echo "MME_SKIPPED"
# AMBER discriminative query is fetched with the generative assets in step 2/4;
# the discriminative generation-prompt builder ($D/amber/prompts_disc.jsonl) is
# NOT yet implemented, so the bench job skips AMBER-disc until it exists.
set -e

echo "=== 4/4 grounding pairs: symlink into data_ucit so anchor images resolve ==="
# faith_loss resolves grounding image paths ("grounding/images/...") against the
# training --data_root, which for the full study is data_ucit. Symlink the pilot
# grounding dir in so data_ucit/grounding/{grounding_pairs.jsonl,images,masked}
# all resolve (every anchor arm needs this; smoke 2026-09-03 caught its absence).
ln -sfn $ROOT/data/grounding $D/grounding
test -f $ROOT/data/grounding/grounding_pairs.jsonl
wc -l $ROOT/data/grounding/grounding_pairs.jsonl
df -h /home | tail -1
echo FS_PREP_OK
