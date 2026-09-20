#!/bin/bash
# Run on vgi1 from ~/clh_jobs after both training arms complete.
# Submits the 9 checkpoint evals (S0, S1-4, J1-4). SLURM queues them across
# vgi2's GPUs. Usage: bash submit_evals.sh [only_labels...]
set -euo pipefail

R=/home/alexmueller/cl-halluc/runs

declare -A ADAPTERS=(
  [S0]=none
  [S1]=$R/seq/S1/adapter_final
  [S2]=$R/seq/S2/adapter_final
  [S3]=$R/seq/S3/adapter_final
  [S4]=$R/seq/S4/adapter_final
  [J1]=$R/joint/ckpt_step000090
  [J2]=$R/joint/ckpt_step000215
  [J3]=$R/joint/ckpt_step000340
  [J4]=$R/joint/adapter_final
)

LABELS="${*:-S0 S1 S2 S3 S4 J1 J2 J3 J4}"
for L in $LABELS; do
  JID=$(sbatch --parsable -J clh_eval_$L \
    --export=ALL,CKPT=$L,ADAPTER=${ADAPTERS[$L]} sb_eval.sbatch)
  echo "eval $L -> job $JID (adapter ${ADAPTERS[$L]})"
done
