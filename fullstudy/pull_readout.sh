#!/bin/bash
# Pull ONLY the small readout dir (aggregate JSON, summary, figures, tables)
# back from vgi2 into analysis/readout/ -- tar-over-srun, since vgi2 shares no
# filesystem with the login node and allows no inter-node ssh.
#
#   bash fullstudy/pull_readout.sh            # -> <repo>/analysis/readout/
#   bash fullstudy/pull_readout.sh /some/dir  # -> /some/dir/readout/
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST=${1:-"$HERE/../analysis"}
mkdir -p "$DEST"
ssh vgi-server 'unset SLURM_JOB_ID SLURM_JOBID SLURM_NODELIST SLURM_JOB_NODELIST; srun -n1 -p debug -w vgi2 --mem=1G -c1 -t5 tar czf - -C /home/alexmueller/cl-halluc readout' \
  | tar xzf - -C "$DEST"
echo "pulled -> $DEST/readout  (stamp $(cat "$DEST/readout/STAMP" 2>/dev/null || echo '?'))"
ls -la "$DEST/readout" | head -20
