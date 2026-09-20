#!/bin/bash
#SBATCH -J clh_probe
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 06:00:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
set -euo pipefail

ROOT=$HOME/cl-halluc
source $ROOT/env/bin/activate
export HF_HOME=$ROOT/hf_home
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python $ROOT/code/probe_datasets.py --out $ROOT/data

python - <<'EOF'
import os
from huggingface_hub import snapshot_download
p = snapshot_download("llava-hf/llava-1.5-7b-hf")
total = 0
for dp, _, fns in os.walk(p):
    for fn in fns:
        total += os.path.getsize(os.path.join(dp, fn))
print("model snapshot at", p, f"{total/1e9:.1f} GB")
EOF
echo PROBE_OK
