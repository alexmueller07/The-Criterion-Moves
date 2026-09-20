#!/bin/bash
#SBATCH -J clh_ground
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
set -euo pipefail
ROOT=$HOME/cl-halluc
source $ROOT/env/bin/activate
export PYTHONUNBUFFERED=1
python $ROOT/code/method/build_grounding_pairs.py --root $ROOT/data
echo GROUND_OK
