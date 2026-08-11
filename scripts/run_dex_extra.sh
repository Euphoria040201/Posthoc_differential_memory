#!/usr/bin/env bash
set -uo pipefail
# Follow-up conditions for the DEX control study.
#   usage: run_dex_extra.sh GPU LR "cond:seed cond:seed ..."
# conditions:
#   mirror_plus       dex_plus initialised at -W_D  (mirror of dex_minus at the same seed)
#   fix_minus         dex_minus with lambda pinned to the depth-aware lambda_init (always on)
#   fix_plus          dex_plus   with lambda pinned to the depth-aware lambda_init (always on)
#   fix_adapteronly   adapter_only with lambda pinned on (attention frozen)
# Everything else is byte-identical to the main matrix.

if [[ $# -lt 3 ]]; then echo "usage: $0 GPU LR 'cond:seed ...'" >&2; exit 2; fi
gpu="$1"; lr="$2"; jobs="$3"
repo=/work/mingze/delta-mem
py=/work/mingze/miniconda3/envs/deltamem/bin/python
plan="$repo/out_dex/head_plan_qwen3_4b.json"
cd "$repo" || exit 1
export CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$repo" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for job in $jobs; do
  cond="${job%%:*}"; seed="${job##*:}"
  case "$cond" in
    mirror_plus)     variant=dex_plus;     extra=(--negate-fd-init true --lambda-anneal-steps 78) ;;
    fix_minus)       variant=dex_minus;    extra=(--lambda-anneal-steps 0 --lambda-learnable false --allow-no-anneal true) ;;
    fix_plus)        variant=dex_plus;     extra=(--lambda-anneal-steps 0 --lambda-learnable false --allow-no-anneal true) ;;
    fix_adapteronly) variant=adapter_only; extra=(--lambda-anneal-steps 0 --lambda-learnable false --allow-no-anneal true) ;;
    *) echo "unknown condition $cond" >&2; continue ;;
  esac
  tag="${DEX_TAG_PREFIX:-}dexx_${cond}_lr${lr}_s${seed}"
  [[ -f "out_dex/${tag}.json" ]] && { echo "[extra] skip $tag"; continue; }
  echo "[extra] $(date +%H:%M:%S) start $tag on GPU $gpu"
  "$py" scripts/dex_train_qasper.py \
    --variant "$variant" --seed "$seed" --steps 156 --lr "$lr" \
    --head-plan "$plan" --head-selection entropy_high --heads-per-layer -1 \
    --lambda-init-mode diff_depth --lambda-learn-init 0.0 \
    --grad-accum 16 --batch-size 1 --warmup-ratio 0.03 --grad-clip 1.0 \
    --weight-decay 0.0 --grad-checkpointing true \
    --val-every 26 --log-every 4 --val-loss-examples 32 --eval-at-start true \
    --output-dir out_dex --tag "$tag" "${extra[@]}" \
    > "out_dex/${tag}.log" 2>&1
  echo "[extra] $(date +%H:%M:%S) done  $tag rc=$?"
done
