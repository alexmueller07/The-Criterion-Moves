#!/bin/bash
# One-command full-study readout ON THE CLUSTER.
#
#   ssh vgi-server 'bash -s' < fullstudy/readout.sh        # from the Mac
#   bash readout.sh                                        # from the vgi1 login node
#
# Runs, inside ONE srun job on vgi2 (the node that holds results_fs; no shared
# fs with the login node):  fs_aggregate -> make_fs_figures -> make_sota_table
# into /home/alexmueller/cl-halluc/readout/ and prints the endpoint + scorecard
# summary. The code is ~/cl-halluc/code/analysis/{fs_common,fs_aggregate,
# make_fs_figures,make_sota_table,fig_style}.py (ship it with
# fullstudy/push_readout_code.sh); the pilot scorers are flat in ~/cl-halluc/code.
# Pull the small readout dir back with fullstudy/pull_readout.sh.
#
# Env knobs: READOUT_MEM (8G), READOUT_MIN (40), E3_BOOT (10000), READOUT_NO_FIGS=1.
set -uo pipefail
unset SLURM_JOB_ID SLURM_JOBID SLURM_NODELIST SLURM_JOB_NODELIST 2>/dev/null || true
MEM=${READOUT_MEM:-8G}
MIN=${READOUT_MIN:-40}
srun -n1 -p debug -w vgi2 --mem=$MEM -c2 -t $MIN --job-name=clh_readout \
  --export=ALL,E3_BOOT=${E3_BOOT:-10000},READOUT_NO_FIGS=${READOUT_NO_FIGS:-0} \
  bash -c '
set -uo pipefail
ROOT=$HOME/cl-halluc
OUT=$ROOT/readout
CODE=$ROOT/code
mkdir -p $OUT/figures $OUT/tables
if [ -f $ROOT/env/bin/activate ]; then source $ROOT/env/bin/activate; fi
export PYTHONPATH=$CODE:$CODE/analysis${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR=$OUT/.mplconfig
export CLH_ROOT=$ROOT
for f in fs_common.py fs_aggregate.py make_fs_figures.py make_sota_table.py fig_style.py; do
  [ -f $CODE/analysis/$f ] || { echo "MISSING $CODE/analysis/$f -- run fullstudy/push_readout_code.sh first"; exit 2; }
done
for f in metrics_pope.py metrics_chair.py metrics_task.py; do
  [ -f $CODE/$f ] || { echo "MISSING $CODE/$f (pilot scorer)"; exit 2; }
done
echo "== readout on $(hostname) $(date "+%F %T")  results_fs cells: $(ls -d $ROOT/results_fs/*/ 2>/dev/null | wc -l)  EVAL_DONE: $(ls $ROOT/results_fs/*/EVAL_DONE 2>/dev/null | wc -l)"
STAMP=$(date +%Y%m%d_%H%M%S)
echo "$STAMP" > $OUT/STAMP

python $CODE/analysis/fs_aggregate.py --out $OUT/fs_aggregate.json --summary $OUT/summary.txt \
  --e3_boot ${E3_BOOT} > $OUT/aggregate_full.txt 2>&1
RC_AGG=$?
if [ $RC_AGG -ne 0 ] && [ $RC_AGG -ne 3 ]; then
  echo "AGGREGATE FAILED (rc=$RC_AGG); tail of $OUT/aggregate_full.txt:"; tail -40 $OUT/aggregate_full.txt; exit $RC_AGG
fi

RC_FIG=0
if [ "${READOUT_NO_FIGS}" != "1" ]; then
  python $CODE/analysis/make_fs_figures.py --agg $OUT/fs_aggregate.json --out $OUT/figures \
    > $OUT/figures.log 2>&1 || { RC_FIG=$?; echo "FIGURES FAILED (rc=$RC_FIG); tail of $OUT/figures.log:"; tail -20 $OUT/figures.log; }
fi
RC_SOTA=0
python $CODE/analysis/make_sota_table.py --agg $OUT/fs_aggregate.json --out $OUT/tables \
  > $OUT/sota.log 2>&1 || { RC_SOTA=$?; echo "SOTA TABLE FAILED (rc=$RC_SOTA); tail of $OUT/sota.log:"; tail -20 $OUT/sota.log; }

cat $OUT/summary.txt
echo
echo "== figures: $(ls $OUT/figures 2>/dev/null | grep -c pdf) pdf   tables: $(ls $OUT/tables 2>/dev/null | tr "\n" " ")"
echo "== full per-stage matrix: $OUT/aggregate_full.txt   JSON: $OUT/fs_aggregate.json   stamp $STAMP"
if [ $RC_AGG -eq 3 ]; then echo "== WARNING: malformed cells were excluded (see MALFORMED banner above)"; fi
[ $RC_FIG -eq 0 ] && [ $RC_SOTA -eq 0 ] && [ $RC_AGG -eq 0 ]
'
