#!/bin/bash
#SBATCH -J clh_antisym
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 6
#SBATCH --mem=20G
#SBATCH -t 1:00:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
# Polarity-antisymmetry diagnostic (2026-09-09).
#
# Tests whether the criterion offset b is estimable WITHOUT LABELS via
#   r_i = (g_i^native + g_i^flipped)/2 ,  g = z(" Yes") - z(" No")
# which equals b exactly when the model inverts its truth signal under a polarity
# flip. This is the gate for a corrected-target method: our anchor lands at
# c=+0.697 because it regularizes toward a base that itself sits at c=+0.431, and
# a label-free estimate of b would replace that mis-placed target with one that is
# correct by logic.
#
# The run is DIAGNOSTIC ONLY -- no training, no adapter is written. It dumps
# teacher-forced answer-position logits for four templates on the frozen base and
# scores them with a PRE-REGISTERED decision rule (analysis/antisymmetry_diag.py).
#
# The positive control is not optional. A weak flip correlation is uninterpretable
# unless we can show the same pipeline reproduces a paraphrase on the same probe;
# without it a null is NO-DATA, which is a mistake this project has made before.
set -euo pipefail
ROOT=$HOME/cl-halluc; cd $ROOT && source env/bin/activate
export HF_HOME=$ROOT/hf_home PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$ROOT/code:$ROOT/code/analysis:$ROOT/code/method
D=$ROOT/data
OUT=$ROOT/results_antisym
mkdir -p $OUT

BACKBONE=${BACKBONE:-llava15}
ADAPTER=${ADAPTER:-none}
TAG=${TAG:-base}
AD=""; [ "$ADAPTER" != "none" ] && AD="--adapter $ADAPTER"

echo "=== 1/3 build the four probe-prompt templates ==="
# native  : the anchor's own question (baseline decision statistic)
# neg     : negation flip           -- truth signal should invert
# absent  : lexical-absence flip    -- second, independent flip wording
# pos_ctrl: paraphrase, SAME polarity -- POSITIVE CONTROL, must correlate +
build () {  # $1 = name, $2 = template
  if [ ! -f $OUT/probe_$1.jsonl ]; then
    python3 $ROOT/code/method/probe_prompts.py \
      --probe $D/grounding/probe.jsonl --out $OUT/probe_$1.jsonl --template "$2"
  fi
  echo "  $1: $(wc -l < $OUT/probe_$1.jsonl) rows"
}
build native   "Is there a {obj} in the image?"
build neg      "Is there no {obj} in the image?"
build absent   "Is the {obj} absent from the image?"
build pos_ctrl "Can you see a {obj} in the image?"

echo "=== 2/3 teacher-forced answer-position logit dumps ($BACKBONE, adapter=$ADAPTER) ==="
for T in native neg absent pos_ctrl; do
  if [ -f $OUT/${TAG}_${T}_dump.jsonl ]; then echo "  skip $T (exists)"; continue; fi
  # POOLED dumper, not eval_gen --dump_logits: the shared instrument resolves a
  # single " Yes"/" No" id pair that the model stops emitting at TextVQA/VizWiz
  # stages (argmax_is "other" on 100% of rows there), so it would measure a
  # statistic decoupled from the actual decision. See method/answer_tokens.py.
  python3 $ROOT/code/method/dump_pooled_logits.py --backbone $BACKBONE \
    --prompts $OUT/probe_$T.jsonl --data_root $D $AD \
    --out $OUT/${TAG}_${T}_dump.jsonl --batch 12
  n=$(wc -l < $OUT/${TAG}_${T}_dump.jsonl)
  # A short dump means rows were silently dropped; that would bias every
  # correlation below, so fail loudly rather than scoring a partial file.
  [ "$n" -ge 500 ] || { echo "DUMP_TOO_SHORT $T ($n rows)"; exit 1; }
  echo "  $T: $n rows"
done

echo "=== 3/3 pre-registered diagnostic ==="
python3 $ROOT/code/analysis/antisymmetry_diag.py \
  --dumps native=$OUT/${TAG}_native_dump.jsonl \
          pos_control=$OUT/${TAG}_pos_ctrl_dump.jsonl \
          neg=$OUT/${TAG}_neg_dump.jsonl \
          absent=$OUT/${TAG}_absent_dump.jsonl \
  ${MEASURED_C:+--measured_c $MEASURED_C} \
  --out $ROOT/readout/antisym_${TAG}.json

echo ANTISYM_DIAG_OK
