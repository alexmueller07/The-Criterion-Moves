#!/bin/bash
#SBATCH -J clh_diskclean
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH -t 00:20:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
# Reap adapters ONLY for arms that are COMPLETELY scored and have no eval in
# flight.
#
# 2026-09-09 incident: the first version deleted an arm as soon as ANY cell had
# EVAL_DONE. An eval in progress writes k1's marker and then still needs k2..kN,
# so this deleted adapters out from under a RUNNING eval and killed
# mth_anchorOOD_o1_s17 after 95 minutes. Two guards now: (1) every expected stage
# must be scored, (2) no clh_ev_<tag> job may be queued or running.
set -uo pipefail
cd /home/alexmueller/cl-halluc
echo "before: $(df -BG /home | tail -1 | awk '{print $4}')"
LIVE=$(squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -E "^clh_ev_" | sed "s/^clh_ev_//" | sort -u)
freed=0; kept=0
for d in runs_fs/*/; do
  tag=$(basename "$d")
  case "$tag" in psL_*|psQ_*|mth_*|lamF_*) exp=4;; *) exp=6;; esac
  n_done=$(ls -d results_fs/${tag}_k*/EVAL_DONE 2>/dev/null | wc -l | tr -d " ")
  final_done=0; ls results_fs/${tag}_final/EVAL_DONE >/dev/null 2>&1 && final_done=1
  if echo "$LIVE" | grep -qx "$tag"; then
    kept=$((kept+1)); echo "  keep (eval in flight): $tag"; continue
  fi
  if [ "$n_done" -ge "$exp" ] || [ "$final_done" = "1" ]; then
    sz=$(du -sm "$d" 2>/dev/null | cut -f1); rm -rf "$d" && freed=$((freed+sz)) && echo "  reaped $tag (${sz}MB, $n_done/$exp scored)"
  else
    kept=$((kept+1)); echo "  keep (only $n_done/$exp scored): $tag"
  fi
done
echo "freed_MB=$freed kept_arms=$kept"
echo "after: $(df -BG /home | tail -1 | awk '{print $4}')"
echo DISKCLEAN_OK
