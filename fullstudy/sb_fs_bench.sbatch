#!/bin/bash
#SBATCH -J clh_fs_bench
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 8
#SBATCH --mem=28G
#SBATCH -t 6:00:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
# --mem right-sized 2026-09-09: RAM (not GPUs) is the binding constraint on vgi2
# (122G, ~90G held by other lanes). The prep smokes ran the identical LLaVA
# training path -- model load + steps + anchor/crit/gca losses -- at 24G, so 32G
# is the evidence-backed ask with headroom. The [PEAKRSS] sampler logs the truth.
set -euo pipefail

# Endpoint external-benchmark eval for the SOTA comparison table (Park request).
# Runs Object HalBench + MME-Hallucination + AMBER-discriminative on ONE
# checkpoint (an arm's final adapter), not per-stage. Judge-free; no API cost.
# Env: RUNTAG (runs_fs/<RUNTAG>), BACKBONE. Optional BASE_ONLY=1 -> raw backbone.
ROOT=$HOME/cl-halluc
source $ROOT/env/bin/activate
export HF_HOME=$ROOT/hf_home
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DU=$ROOT/data_ucit
CODE=$ROOT/code

if [ "${BASE_ONLY:-0}" = "1" ]; then
  LABEL=${BACKBONE}_base; ADAPTER=none
else
  R=$ROOT/runs_fs/$RUNTAG
  if [ -d $R/joint ]; then ADAPTER=$R/joint/adapter_final; else
    K=1; while [ -d $R/S$((K+1)) ]; do K=$((K+1)); done; ADAPTER=$R/S$K/adapter_final
  fi
  LABEL=${RUNTAG}_bench
fi
OUT=$ROOT/results_fs_bench/$LABEL
mkdir -p $OUT
if [ -f $OUT/BENCH_DONE ]; then echo "skip done $LABEL"; exit 0; fi

AD=""; [ "$ADAPTER" != "none" ] && AD="--adapter $ADAPTER"
if [ "$BACKBONE" = "qwen25vl" ]; then GB_SHORT=6; GB_LONG=2; else GB_SHORT=12; GB_LONG=6; fi
echo "=== bench $LABEL adapter=$ADAPTER ==="
# Each benchmark is INDEPENDENT and NON-FATAL: one benchmark's missing images
# (MME needs its own image set, not shipped in the eval-tool zip; AMBER-disc
# needs a prompt-builder) must not kill the others. ObjHal uses COCO (present).
set +e
NOK=0

# Object HalBench (detailed captions -> our vendored CHAIR)
if [ -f $DU/objhal/prompts.jsonl ] && [ -d $DU/objhal/images ]; then
  python $CODE/eval_gen.py --backbone $BACKBONE --prompts $DU/objhal/prompts.jsonl \
    --data_root $DU/objhal/images $AD --out $OUT/objhal_gen.jsonl --max_new_tokens 512 --batch $GB_LONG && \
  python $CODE/metrics_chair.py --gen $OUT/objhal_gen.jsonl \
    --coco_gt $DU/objhal/coco_gt.json --out_prefix $OUT/objhal --ckpt $LABEL \
    && { echo OBJHAL_OK; NOK=$((NOK+1)); } || echo "OBJHAL_SKIPPED ($LABEL)"
fi

# Open Images CHAIR: the NON-COCO generative eval (anchor domain-overlap
# confound control; built by pilot/fetch_noncoco_chair.py). Skip if absent.
if [ -f $DU/noncoco/prompts.jsonl ] && [ -f $DU/noncoco/gt.json ]; then
  python $CODE/eval_gen.py --backbone $BACKBONE --prompts $DU/noncoco/prompts.jsonl \
    --data_root $DU/noncoco/images $AD --out $OUT/noncoco_gen.jsonl --max_new_tokens 512 --batch $GB_LONG && \
  python $CODE/metrics_chair.py --gen $OUT/noncoco_gen.jsonl \
    --coco_gt $DU/noncoco/gt.json --out_prefix $OUT/noncoco --ckpt $LABEL \
    && { echo NONCOCO_OK; NOK=$((NOK+1)); } || echo "NONCOCO_SKIPPED ($LABEL)"
else
  echo "NONCOCO_SKIPPED ($LABEL): data absent"
fi

# MME-Hallucination (needs its own images; skip if absent)
if [ -f $DU/mme/prompts.jsonl ] && [ -d $DU/mme/MME_Benchmark ]; then
  python $CODE/eval_gen.py --backbone $BACKBONE --prompts $DU/mme/prompts.jsonl \
    --data_root $DU $AD --out $OUT/mme_gen.jsonl --max_new_tokens 16 --batch $GB_SHORT && \
  python $CODE/metrics_mme.py --gen $OUT/mme_gen.jsonl \
    --out_prefix $OUT/mme --ckpt $LABEL && { echo MME_OK; NOK=$((NOK+1)); } || echo "MME_SKIPPED ($LABEL)"
else
  echo "MME_SKIPPED ($LABEL): images absent"
fi

# AMBER discriminative (prompt-builder is an open gap; skip if absent)
if [ -f $DU/amber/prompts_disc.jsonl ]; then
  python $CODE/eval_gen.py --backbone $BACKBONE --prompts $DU/amber/prompts_disc.jsonl \
    --data_root $DU $AD --out $OUT/amberdisc_gen.jsonl --max_new_tokens 16 --batch $GB_SHORT && \
  python $CODE/metrics_amber_disc.py --gen $OUT/amberdisc_gen.jsonl \
    --amber_data $DU/amber --out_prefix $OUT/amberdisc --ckpt $LABEL \
    && { echo AMBERDISC_OK; NOK=$((NOK+1)); } || echo "AMBERDISC_SKIPPED ($LABEL)"
fi

set -e
# A bench that scored NOTHING is a failure, not a result: no BENCH_DONE, no
# adapter cleanup, non-zero exit so the scheduler/monitor sees it (junk
# BENCH_DONE dirs with only a partial gen file were locked in this way).
if [ "$NOK" -eq 0 ]; then echo "FS_BENCH_FAILED_$LABEL: no benchmark produced a score"; exit 1; fi
touch $OUT/BENCH_DONE
# The bench is the last consumer of this arm's adapter; free it when asked
# (the scheduler passes CLEAN_AFTER=1; default keeps it for safety).
if [ "${CLEAN_AFTER:-0}" = "1" ] && [ "$ADAPTER" != "none" ]; then
  rm -rf $ROOT/runs_fs/$RUNTAG && echo "cleaned adapters after bench: $RUNTAG"
fi
echo FS_BENCH_OK_$LABEL
