#!/bin/bash
#SBATCH -J clh_prep
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
set -euo pipefail

ROOT=$HOME/cl-halluc
source $ROOT/env/bin/activate
export HF_HOME=$ROOT/hf_home
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

ONLY=${ONLY:-all}
ONLY=${ONLY//+/,}
python $ROOT/code/data_prep.py --root $ROOT/data --only $ONLY
echo PREP_OK
