#!/bin/bash
# Train the FULL 4500-ctx spread12 config on 2x16GB by splitting ONE run across both cards
# (pipeline / device_map balanced) + gradient checkpointing. Verified on 2x RTX 4080 SUPER.
# Uses BOTH GPUs for a SINGLE run -> run seeds sequentially (not 2-in-parallel).
set -e; cd "$(dirname "$0")"
PY=${PY:-~/miniconda3/envs/deltamem/bin/python}
SL="0,3,6,9,12,15,18,21,24,27,30,33"
SEEDS=(${SEEDS:-1 2})
for s in "${SEEDS[@]}"; do
  echo "=== seed $s (both GPUs, 4500 ctx) ==="
  PYTHONPATH=.:scripts CUDA_VISIBLE_DEVICES=${GPUS:-0,1} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY scripts/qasper_prefix_steer.py \
    --model-path Qwen/Qwen3-4B-Instruct-2507 --data qasper --train-papers 800 --val-papers 1 \
    --max-ctx-tok 4500 --batch-size 1 --grad-accum 16 --eval-every 999999 --max-new-tokens 8 --lr 5e-4 \
    --steer-mode deltamem --steer-gain 0.1 --delta-heads qkvo --sliding-window-size 256 --backbone-window 0 \
    --memory-mode dynamic --prefix-lr 1e-2 --train-mode ctx --num-prefix-tokens 64 --prefix-write true \
    --write-ctx-only true --delta-rank 0 --mem-head-dim 64 --mem-num-heads 1 --pool-reads true \
    --steer-layers $SL --steps 156 --save-steps 100,130 --train-target-n 935 --data-compose-seed 42 \
    --max-yesno-frac 0.03 --device-map balanced --grad-checkpointing --dtype bfloat16 \
    --output-dir out16_4500 --tag s45_s$s
done
echo "done -> out16_4500/"
