#!/bin/bash
# Disk-safe wave scheduler for the full study. Run ON THE LOGIN NODE (vgi1)
# from ~/clh_jobs; call it repeatedly (a monitor loop does, every ~20 min).
#
# Reads queue.txt: one arm per line
#   RUNTAG|BACKBONE|ORDER|SEED|ARM|SUITE|TIMELIMIT|BENCH|EXTRAENV
#   SUITE = ucit | pilot ; BENCH = 1 to chain the endpoint benchmark ; EXTRAENV
#   is a comma-joined KEY=VAL list for the train job (e.g. FAITH_WEIGHT=0.2).
# A submitted line is rewritten as  DONE:<trainjobid>:<retries>|<original>.
# Behavior each pass:
#   * requeue any submitted train that FAILED (<= MAX_RETRY times), logging why
#   * if vgi2 free disk > MIN_FREE_GB and my in-flight train arms < MAX_INFLIGHT,
#     submit the next pending lines (train -> eval(afterok) -> bench(afterok))
#   * print one status line; append details to sched.log
# Everything here is idempotent; a crash mid-pass loses nothing (queue.txt is
# rewritten atomically only after successful sbatch).
set -uo pipefail
unset SLURM_JOB_ID SLURM_JOBID SLURM_NODELIST SLURM_JOB_NODELIST 2>/dev/null

cd "$(dirname "$0")"
Q=queue.txt; LOG=sched.log
MAX_INFLIGHT=${MAX_INFLIGHT:-7}
MIN_FREE_GB=${MIN_FREE_GB:-55}
MAX_RETRY=${MAX_RETRY:-2}
ROOT=/home/alexmueller/cl-halluc
ts() { date "+%m-%d %H:%M"; }
log() { echo "[$(ts)] $*" >> $LOG; }

[ -f $Q ] || { echo "no $Q"; exit 0; }

# --- disk on the compute node (no shared fs) ---------------------------------
FREE=$(timeout 60 srun -n1 -p debug -w vgi2 --mem=512M -c1 -t2 bash -c \
       'df -BG /home | tail -1 | awk "{print \$4}" | tr -d G' 2>/dev/null </dev/null | tail -1)
[[ "$FREE" =~ ^[0-9]+$ ]] || FREE=-1

# --- retry failed trains ------------------------------------------------------
tmp=$(mktemp)
requeued=0
while IFS= read -r line; do
  if [[ "$line" == DONE:* ]]; then
    meta=${line%%|*}; orig=${line#*|}
    jid=$(echo "$meta" | cut -d: -f2); rt=$(echo "$meta" | cut -d: -f3)
    st=$(sacct -j "$jid" -n -X --format=State 2>/dev/null </dev/null | head -1 | tr -d ' ')
    case "$st" in
      FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|CANCELLED*)
        RT1=${orig%%|*}
        rt=$(awk -v r="$RT1" '$1==r{print $2}' retries.txt 2>/dev/null | tail -1); rt=${rt:-0}
        if [ "$rt" -lt "$MAX_RETRY" ]; then
          reason=$(timeout 45 srun -n1 -p debug -w vgi2 --mem=256M -c1 -t2 bash -c \
                   "grep -hoE 'GATE FAIL[^.]*|OutOfMemoryError|must be larger than factor|No space left|[A-Za-z]*Error: [^\n]{0,80}' $ROOT/logs/*_${jid}.out 2>/dev/null | tail -1" 2>/dev/null </dev/null | tail -1)
          echo "$RT1 $((rt+1))" >> retries.txt
          log "RETRY $RT1 (job $jid $st, retry $((rt+1))/$MAX_RETRY; reason: ${reason:-unknown})"
          echo "$orig" >> $tmp; requeued=$((requeued+1)); continue
        else
          log "GIVEUP $RT1 after $MAX_RETRY retries (job $jid $st)"
        fi;;
    esac
  fi
  echo "$line" >> $tmp
done < $Q
mv $tmp $Q

# --- how many of my train arms are in flight ---------------------------------
INFL=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -cE '^clh_(fsL|fsQ|psQ|psL|lamF|mth)_' )
PEND=$(grep -cvE '^(DONE:|#|$)' $Q)

submitted=0
if [ "$FREE" -ge "$MIN_FREE_GB" ] && [ "$INFL" -lt "$MAX_INFLIGHT" ]; then
  slots=$((MAX_INFLIGHT - INFL))
  tmp=$(mktemp)
  while IFS= read -r line; do
    if [ $slots -gt 0 ] && [[ "$line" != DONE:* ]] && [[ "$line" != \#* ]] && [ -n "$line" ]; then
      IFS='|' read -r RT BK ORD SEED ARM SUITE TL BENCH EXTRA <<< "$line"
      if [ "$SUITE" = "pilot" ]; then DD=$ROOT/data; MF=$ROOT/data/pilot_manifest.json
      elif [[ "$SUITE" == pilot_* ]]; then V=${SUITE#pilot_}; DD=$ROOT/data/tasks_$V; MF=$DD/${V}_manifest.json
      else DD=$ROOT/data_ucit; MF=$ROOT/data_ucit/ucit_manifest.json; fi
      EXP="ALL,RUNTAG=$RT,BACKBONE=$BK,ORDER=$ORD,SEED=$SEED,ARM=$ARM,DATADIR=$DD,MANIFEST=$MF"
      [ -n "${EXTRA:-}" ] && EXP="$EXP,$EXTRA"
      TID=$(sbatch --parsable -t "$TL" -J clh_$RT --export="$EXP" sb_fs_arm.sbatch 2>>$LOG </dev/null)
      if [[ "$TID" =~ ^[0-9]+$ ]]; then
        KF=0; [ "${BENCH:-0}" = "1" ] && KF=1
        EID=$(sbatch --parsable --dependency=afterok:$TID -J clh_ev_$RT \
              --export="ALL,RUNTAG=$RT,BACKBONE=$BK,DATADIR=$DD,MANIFEST=$MF,KEEP_FINAL=$KF" sb_fs_eval.sbatch 2>>$LOG </dev/null)
        if [ "${BENCH:-0}" = "1" ] && [[ "$EID" =~ ^[0-9]+$ ]]; then
          sbatch --parsable --dependency=afterok:$EID -J clh_bn_$RT \
            --export="ALL,RUNTAG=$RT,BACKBONE=$BK,CLEAN_AFTER=1" sb_fs_bench.sbatch >/dev/null 2>>$LOG </dev/null
        fi
        RC=$(awk -v r="$RT" '$1==r{print $2}' retries.txt 2>/dev/null | tail -1); RC=${RC:-0}
        echo "DONE:$TID:$RC|$line" >> $tmp
        log "SUBMIT $RT train=$TID eval=$EID bench=${BENCH:-0}"
        submitted=$((submitted+1)); slots=$((slots-1)); continue
      else
        log "SBATCH-FAIL $RT (kept pending)"
      fi
    fi
    echo "$line" >> $tmp
  done < $Q
  mv $tmp $Q
fi

DONE_N=$(grep -c '^DONE:' $Q); PEND=$(grep -cvE '^(DONE:|#|$)' $Q)
echo "[$(ts)] free=${FREE}G inflight=$INFL submitted=$submitted requeued=$requeued pending=$PEND submitted_total=$DONE_N"
log "PASS free=${FREE}G inflight=$INFL submitted=$submitted requeued=$requeued pending=$PEND"
