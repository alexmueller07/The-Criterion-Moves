#!/bin/bash
#SBATCH -J clh_probe_prep
#SBATCH -p batch
#SBATCH -w vgi2
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 1:00:00
#SBATCH -o /home/alexmueller/cl-halluc/logs/%x_%j.out
# Red-team M2 instruments: rephrased POPE; noise probe; Open Images probe (if OI pairs exist);
# prompt renderings of every probe (for the PCR-transferred-from-probe baseline). CPU only.
set -euo pipefail
ROOT=$HOME/cl-halluc; cd $ROOT && source env/bin/activate; D=$ROOT/data
python code/build_pope_rephrased.py --pope $D/pope
[ -d $D/noise_images ] || python code/method/make_noise_images.py --out_dir $D/noise_images --n 600
[ -f $D/grounding/probe_noise.jsonl ] || python code/method/build_probe_set.py --root $D --images_dir $D/noise_images --n 600 --out $D/grounding/probe_noise.jsonl
if [ -d $D/grounding_oi/images ] && [ ! -f $D/grounding/probe_oi.jsonl ]; then
  python code/method/build_probe_set.py --root $D --images_dir $D/grounding_oi/images --n 600 --out $D/grounding/probe_oi.jsonl
fi
for P in probe probe_noise probe_oi; do
  [ -f $D/grounding/$P.jsonl ] && python code/method/probe_prompts.py --probe $D/grounding/$P.jsonl --out $D/grounding/${P}_prompts.jsonl
done
ls -la $D/pope/prompts_rephrased.jsonl $D/grounding/probe*_prompts.jsonl
echo PROBE_PREP_OK
