#!/bin/bash
# Train the spread12 config on 16GB cards (e.g. RTX 4080 SUPER). Gradient checkpointing +
# max-ctx-tok 3500 → peak ~15.2GB. Launches ONE seed per GPU (2 cards = 2 seeds in parallel).
set -e; cd "$(dirname "$0")"
PY=${PY:-~/miniconda3/envs/deltamem/bin/python}
SL="0,3,6,9,12,15,18,21,24,27,30,33"
COMMON="--model-path Qwen/Qwen3-4B-Instruct-2507 --data qasper --train-papers 800 --val-papers 1 \
  --max-ctx-tok 3500 --batch-size 1 --grad-accum 16 --eval-every 999999 --max-new-tokens 8 --lr 5e-4 \
  --steer-mode deltamem --steer-gain 0.1 --delta-heads qkvo --sliding-window-size 256 --backbone-window 0 \
  --memory-mode dynamic --prefix-lr 1e-2 --train-mode ctx --num-prefix-tokens 64 --prefix-write true \
  --write-ctx-only true --delta-rank 0 --mem-head-dim 64 --mem-num-heads 1 --pool-reads true \
  --steer-layers $SL --steps 156 --save-steps 100,130 --train-target-n 935 --data-compose-seed 42 \
  --max-yesno-frac 0.03 --grad-checkpointing --dtype bfloat16"
mkdir -p out16
GPUS=(${GPUS:-0 1}); SEEDS=(${SEEDS:-1 2}); i=0
for s in "${SEEDS[@]}"; do g=${GPUS[$i]}; i=$((i+1))
  echo "seed $s -> GPU $g"
  PYTHONPATH=.:scripts CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup $PY scripts/qasper_prefix_steer.py $COMMON --seed $s --output-dir out16 --tag s16_s$s \
    > out16/s16_s$s.log 2>&1 &
done
wait; echo "all training done -> out16/"
