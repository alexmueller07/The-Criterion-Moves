#!/bin/bash
# Ship the readout pipeline code to vgi2:~/cl-halluc/code/analysis/ (the pilot
# scorers it imports already live flat in ~/cl-halluc/code/ via
# pilot/sync_and_submit.sh). Tar-over-srun-stdin: no shared fs, no inter-node ssh.
#
#   bash fullstudy/push_readout_code.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
A="$HERE/../analysis"
for f in fs_common.py fs_aggregate.py make_fs_figures.py make_sota_table.py fig_style.py; do
  [ -f "$A/$f" ] || { echo "missing $A/$f"; exit 1; }
done
python3 -m py_compile "$A"/fs_common.py "$A"/fs_aggregate.py "$A"/make_fs_figures.py \
  "$A"/make_sota_table.py "$A"/fig_style.py
tar czf - -C "$A" fs_common.py fs_aggregate.py make_fs_figures.py make_sota_table.py fig_style.py \
  | ssh vgi-server 'unset SLURM_JOB_ID SLURM_JOBID SLURM_NODELIST SLURM_JOB_NODELIST; srun -n1 -p debug -w vgi2 --mem=1G -c1 -t5 bash -c "mkdir -p ~/cl-halluc/code/analysis && tar xzf - -C ~/cl-halluc/code/analysis && ls -la ~/cl-halluc/code/analysis && ls ~/cl-halluc/code/metrics_pope.py ~/cl-halluc/code/metrics_chair.py ~/cl-halluc/code/metrics_task.py"'
echo "pushed readout code -> vgi2:~/cl-halluc/code/analysis/"
