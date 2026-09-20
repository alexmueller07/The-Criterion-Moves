#!/bin/bash
# Full-study launch matrix. Run ON vgi1 from ~/clh_jobs AFTER smoke is green.
# Submits every arm with its eval chained afterok, waves gated by dependencies.
# Usage: bash launch_matrix.sh <smoke_job_id>
set -euo pipefail
SMOKE=${1:?pass the green smoke job id}
DEP="--dependency=afterok:$SMOKE"

submit_arm () {
  # arg7 BENCH (optional): "bench" also queues the endpoint external-benchmark
  # job (Object HalBench + MME + AMBER-disc) for the SOTA comparison table.
  local BACKBONE=$1 ORDER=$2 SEED=$3 ARM=$4 TLIMIT=$5 EXTRA_DEP=$6 BENCH=${7:-}
  local BTAG=L
  if [ "$BACKBONE" = "qwen25vl" ]; then BTAG=Q; fi
  local ATAG=${ARM/:/_}
  local RUNTAG="fs${BTAG}_${ATAG}_${ORDER}_s${SEED}"
  local TID
  TID=$(sbatch --parsable $EXTRA_DEP -t $TLIMIT -J clh_${RUNTAG} \
    --export=ALL,RUNTAG=$RUNTAG,BACKBONE=$BACKBONE,ORDER=$ORDER,SEED=$SEED,ARM=$ARM \
    sb_fs_arm.sbatch)
  sbatch --parsable --dependency=afterok:$TID -J clh_ev_${RUNTAG} \
    --export=ALL,RUNTAG=$RUNTAG,BACKBONE=$BACKBONE sb_fs_eval.sbatch > /dev/null
  if [ "$BENCH" = "bench" ]; then
    sbatch --parsable --dependency=afterok:$TID -J clh_bn_${RUNTAG} \
      --export=ALL,RUNTAG=$RUNTAG,BACKBONE=$BACKBONE sb_fs_bench.sbatch > /dev/null
  fi
  echo "$RUNTAG train=$TID${BENCH:+ +bench}"
}

# Bench flag helper: the SOTA endpoint-benchmark table uses one canonical cell
# per method (LLaVA, canonical order o1, seed 17). bcell echoes "bench" there.
bcell () { [ "$1" = "o1" ] && [ "$2" = "17" ] && echo bench || echo ""; }

echo "=== base evals + base bench (once per backbone) ==="
sbatch --parsable $DEP -J clh_ev_baseL --export=ALL,BASE_ONLY=1,BACKBONE=llava15 sb_fs_eval.sbatch
sbatch --parsable $DEP -J clh_ev_baseQ --export=ALL,BASE_ONLY=1,BACKBONE=qwen25vl sb_fs_eval.sbatch
sbatch --parsable $DEP -J clh_bn_baseL --export=ALL,BASE_ONLY=1,BACKBONE=llava15 sb_fs_bench.sbatch
sbatch --parsable $DEP -J clh_bn_baseQ --export=ALL,BASE_ONLY=1,BACKBONE=qwen25vl sb_fs_bench.sbatch

echo "=== WAVE A: LLaVA core (SEQ + ANCHOR, 3 orders x 3 seeds) ==="
for ORDER in o1 o2 o3; do
  for SEED in 17 23 31; do
    submit_arm llava15 $ORDER $SEED seq    12:00:00 "$DEP" "$(bcell $ORDER $SEED)"
    submit_arm llava15 $ORDER $SEED anchor 16:00:00 "$DEP" "$(bcell $ORDER $SEED)"
  done
done

echo "=== WAVE B: LLaVA joint (per seed) + ER + single-task controls ==="
for SEED in 17 23 31; do
  submit_arm llava15 o1 $SEED joint 12:00:00 "$DEP" "$(bcell o1 $SEED)"
  submit_arm llava15 o1 $SEED er    12:00:00 "$DEP" "$(bcell o1 $SEED)"
done
# UCIT task list (frozen by the schema probe 2026-09-03); the manifest itself
# lives on vgi2 and is read by the jobs, but this launcher runs on the vgi1
# login node which shares no filesystem with vgi2, so the list is inlined.
TASKS="ArxivQA CLEVR-Math Flickr30k IconQA ImageNet-R VizWiz"
for T in $TASKS; do
  submit_arm llava15 o1 17 single:$T 4:00:00 "$DEP"
done

echo "=== WAVE C: Qwen generalizability (gated on Qwen smoke inside same smoke job) ==="
for ORDER in o1 o2 o3; do
  for SEED in 17 23 31; do
    submit_arm qwen25vl $ORDER $SEED seq    16:00:00 "$DEP"
    submit_arm qwen25vl $ORDER $SEED anchor 20:00:00 "$DEP"
  done
done
for SEED in 17 23 31; do
  submit_arm qwen25vl o1 $SEED joint 16:00:00 "$DEP"
  submit_arm qwen25vl o1 $SEED er    16:00:00 "$DEP"
done
for T in $TASKS; do
  submit_arm qwen25vl o1 17 single:$T 6:00:00 "$DEP"
done

echo "=== WAVE D: ablations (LLaVA) ==="
for SEED in 17 23 31; do
  submit_arm llava15 o1 $SEED er500 12:00:00 "$DEP" "$(bcell o1 $SEED)"
  submit_arm llava15 o1 $SEED cecf  16:00:00 "$DEP" "$(bcell o1 $SEED)"
done

echo "=== WAVE E: SOTA-table CL baselines (LLaVA, canonical order, 3 seeds) ==="
for SEED in 17 23 31; do
  submit_arm llava15 o1 $SEED ewc   14:00:00 "$DEP" "$(bcell o1 $SEED)"
  submit_arm llava15 o1 $SEED lwf   14:00:00 "$DEP" "$(bcell o1 $SEED)"
  # O-LoRA excluded: incompatible with single-resumed-adapter CL (see run_arm.py)
done

echo "MATRIX SUBMITTED"
squeue -u $USER -h | wc -l
