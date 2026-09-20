#!/bin/bash
# Run from the Mac, inside pilot/. Ships code to vgi2 (no shared fs, no
# inter-node ssh: tarball piped through srun stdin), sbatch files to vgi1,
# and optionally submits the setup->probe chain.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=== ship sbatch files to vgi1 ==="
ssh vgi-server 'mkdir -p ~/clh_jobs'
scp -q "$HERE"/jobs/*.sbatch vgi-server:clh_jobs/

echo "=== ship code to vgi2 via srun stdin pipe ==="
tar czf - -C "$HERE" common method data_prep.py probe_datasets.py train_lora.py \
    eval_gen.py metrics_pope.py metrics_chair.py metrics_task.py \
  | ssh vgi-server 'unset SLURM_JOB_ID SLURM_JOBID SLURM_NODELIST SLURM_JOB_NODELIST; srun -p debug -w vgi2 --mem=2G -c 2 -t 10 bash -c "mkdir -p ~/cl-halluc/code && tar xzf - -C ~/cl-halluc/code && find ~/cl-halluc/code -name \"*.py\" | sort"'

if [ "${1:-}" = "submit_setup" ]; then
  echo "=== submit setup -> probe chain ==="
  ssh vgi-server 'cd ~/clh_jobs && SID=$(sbatch --parsable sb_setup.sbatch) && echo setup=$SID && sbatch --parsable --dependency=afterok:$SID sb_probe.sbatch | sed "s/^/probe=/"'
fi
