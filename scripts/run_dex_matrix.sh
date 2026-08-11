#!/usr/bin/env bash
set -uo pipefail

# Sequential DEX variant runner for one GPU.
#   usage: run_dex_matrix.sh GPU LR "variant:seed variant:seed ..."
# Every job uses the identical data, budget, optimiser, schedule, head plan and
# lambda schedule; only the variant name and the seed change.

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GPU LR 'variant:seed [variant:seed ...]'" >&2
  exit 2
fi

gpu="$1"; lr="$2"; jobs="$3"
repo=/work/mingze/delta-mem
py=/work/mingze/miniconda3/envs/deltamem/bin/python
plan="$repo/out_dex/head_plan_qwen3_4b.json"

cd "$repo" || exit 1
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for job in $jobs; do
  variant="${job%%:*}"
  seed="${job##*:}"
  tag="${DEX_TAG_PREFIX:-}dex_${variant}_lr${lr}_s${seed}"
  if [[ -f "out_dex/${tag}.json" ]]; then
    echo "[matrix] skip $tag (already done)"
    continue
  fi
  steps=156
  extra=()
  if [[ "$variant" == "base" ]]; then
    steps=1                      # no optimizer: the loop exits immediately
  fi
  echo "[matrix] $(date +%H:%M:%S) start $tag on GPU $gpu"
  "$py" scripts/dex_train_qasper.py \
    --variant "$variant" --seed "$seed" --steps "$steps" --lr "$lr" \
    --head-plan "$plan" --head-selection entropy_high --heads-per-layer -1 \
    --lambda-init-mode diff_depth --lambda-learn-init 0.0 --lambda-anneal-steps 78 \
    --grad-accum 16 --batch-size 1 --warmup-ratio 0.03 --grad-clip 1.0 \
    --weight-decay 0.0 --grad-checkpointing true \
    --val-every 26 --log-every 4 --val-loss-examples 32 --eval-at-start true \
    --output-dir out_dex --tag "$tag" "${extra[@]}" \
    > "out_dex/${tag}.log" 2>&1
  echo "[matrix] $(date +%H:%M:%S) done  $tag rc=$?"
done
