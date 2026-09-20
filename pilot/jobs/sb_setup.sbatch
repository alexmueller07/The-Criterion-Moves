#!/bin/bash
#SBATCH -J clh_setup
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH -c 8
#SBATCH --mem=8G
#SBATCH -t 03:00:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
set -euo pipefail

ROOT=$HOME/cl-halluc
mkdir -p $ROOT/hf_home $ROOT/data $ROOT/runs $ROOT/results $ROOT/logs
df -h / | tail -1

if [ ! -f $ROOT/env/bin/activate ]; then
  rm -rf $ROOT/env
  python3 -m venv --without-pip $ROOT/env
fi
source $ROOT/env/bin/activate
if ! python -m pip --version 2>/dev/null; then
  curl -sS https://bootstrap.pypa.io/get-pip.py -o $ROOT/get-pip.py
  python $ROOT/get-pip.py
fi
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install "transformers==4.49.0" "peft==0.14.0" "datasets==3.2.0" "accelerate==1.3.0" pillow sentencepiece protobuf

python - <<'EOF'
import torch, transformers, peft, datasets
print("torch", torch.__version__, "cuda_available_at_setup", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("datasets", datasets.__version__)
EOF
echo SETUP_OK
