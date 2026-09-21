#!/bin/bash
# DCL arms: the second benchmark. POPE/CHAIR stay the measuring instrument
# (sb_fs_eval.sbatch pins DP=$ROOT/data, the COCO assets), so only the training
# data and the manifest change. AMBER is guarded by a -f test in the eval and is
# simply skipped for DCL, which has no amber/ directory.
#
# Only seq and joint are launched. The anchor/critp arms need D/grounding/*,
# which DCL has no counterpart for, and the paper's DCL claim is about whether
# criterion drift reproduces on a second benchmark -- that needs the sequential
# trajectory and its matched joint bound, nothing else.
set -u
J=$HOME/clh_jobs
DD=$HOME/cl-halluc/data_dcl
MF=$DD/dcl_manifest.json

sub () {
  local out id
  out=$(sbatch -J "$2" --export=ALL,"$3" "$1" 2>&1)
  id=$(echo "$out" | grep -oE "[0-9]{4,}" | tail -1)
  [ -z "$id" ] && { echo "SUBMIT FAILED for $2: $out" >&2; return 1; }
  echo "$id"
}
launch () {   # launch <runtag> <arm> <order> <seed>
  local tag=$1 arm=$2 ord=$3 seed=$4 a e
  # No commas inside any value: sbatch --export truncates a value at its first
  # comma, which silently runs a fraction of the intended work.
  a=$(sub $J/sb_fs_arm.sbatch "tr_$tag" \
      "RUNTAG=$tag,BACKBONE=llava15,ORDER=$ord,SEED=$seed,ARM=$arm,DATADIR=$DD,MANIFEST=$MF") || return 1
  e=$(sbatch -J "ev_$tag" --dependency=afterok:$a \
      --export=ALL,"RUNTAG=$tag,BACKBONE=llava15,DATADIR=$DD,MANIFEST=$MF" \
      $J/sb_fs_eval.sbatch 2>&1 | grep -oE "[0-9]{4,}" | tail -1)
  printf "  %-26s train=%s eval=%s  (arm=%s order=%s)\n" "$tag" "$a" "$e" "$arm" "$ord"
}

echo "=== DCL: sequential trajectory and its matched joint bound ==="
launch "fsD_seq_d1_s17"   seq   d1 17
launch "fsD_joint_d1_s17" joint d1 17
