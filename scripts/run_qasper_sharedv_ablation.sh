#!/usr/bin/env bash
set -euo pipefail

# Reproducible single-GPU launcher for the no-prefix/shared-V Qasper ablation.
# Usage:
#   bash scripts/run_qasper_sharedv_ablation.sh \
#     <physical_gpu> <tag> <trainable|main_v> <fixed|rms_match|cosine> \
#     <seed> [heads] [gain] [prefix_tokens]
#
# PREFIX_TOKENS defaults to 0 (window-only memory).  Passing P>0 restores the
# written prefix slots so the same protocol can measure what the prefix adds.

if [[ $# -lt 5 || $# -gt 8 ]]; then
  echo "usage: $0 GPU TAG VALUE_SOURCE OUTPUT_FUSION SEED [HEADS] [GAIN] [PREFIX_TOKENS]" >&2
  exit 2
fi

physical_gpu="$1"
run_tag="$2"
value_source="$3"
output_fusion="$4"
run_seed="$5"
mem_heads="${6:-1}"
prefix_tokens="${8:-0}"
if [[ $# -ge 7 ]]; then
  steer_gain="$7"
elif [[ "$output_fusion" == "fixed" ]]; then
  steer_gain="0.1"
else
  # Norm-based branches are defined at unit branch strength.  For rms_match
  # this gives equal main/delta RMS followed by the sqrt(2) energy divisor.
  steer_gain="1.0"
fi

case "$value_source" in
  trainable|main_v) ;;
  *) echo "invalid VALUE_SOURCE: $value_source" >&2; exit 2 ;;
esac
case "$output_fusion" in
  fixed|rms_match|cosine) ;;
  *) echo "invalid OUTPUT_FUSION: $output_fusion" >&2; exit 2 ;;
esac
case "$mem_heads" in
  1|2|4|8) ;;
  *) echo "HEADS must divide Qwen's 8 KV heads: $mem_heads" >&2; exit 2 ;;
esac
if ! [[ "$prefix_tokens" =~ ^[0-9]+$ ]]; then
  echo "PREFIX_TOKENS must be a non-negative integer: $prefix_tokens" >&2
  exit 2
fi
# main_v reuses the frozen backbone's own v_proj over the READ tokens, and the
# prefix slots are not backbone tokens, so that branch is prefix-free by
# construction (the trainer asserts the same).  P>0 needs a trainable mem_v.
if (( prefix_tokens > 0 )) && [[ "$value_source" == "main_v" ]]; then
  echo "PREFIX_TOKENS>0 requires VALUE_SOURCE=trainable (main_v has no prefix V)" >&2
  exit 2
fi

# P=0 is the window-only branch: no written slots, so nothing to write and no
# context-only WRITE pass.  P>0 restores the Qasper WRITE (context-only probes).
if (( prefix_tokens > 0 )); then
  prefix_write="true"
  write_ctx_only="true"
else
  prefix_write="false"
  write_ctx_only="false"
fi

repo_dir="/work/mingze/delta-mem"
python_bin="/work/mingze/miniconda3/envs/deltamem/bin/python"
model_dir="/work/mingze/models/Qwen3-4B-Instruct-2507"
output_dir="${repo_dir}/out_swa_sharedv"

mkdir -p "$output_dir"
cd "$repo_dir"

export CUDA_VISIBLE_DEVICES="$physical_gpu"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"

# Qwen3-4B has eight 128-d KV heads.  heads=1 is the compact branch (the frozen
# V heads are grouped by an adapter-free mean); heads=8 is exact per-head reuse.
exec "$python_bin" scripts/qasper_prefix_steer.py \
  --model-path "$model_dir" \
  --attn-impl sdpa \
  --dtype bfloat16 \
  --device cuda \
  --seed "$run_seed" \
  --data qasper \
  --train-mode ctx \
  --num-prefix-tokens "$prefix_tokens" \
  --prefix-write "$prefix_write" \
  --write-ctx-only "$write_ctx_only" \
  --memory-mode dynamic \
  --read-prefix-only false \
  --pool-reads false \
  --pool-gate false \
  --sliding-window-size 256 \
  --mem-num-heads "$mem_heads" \
  --mem-head-dim 128 \
  --memory-value-source "$value_source" \
  --steer-mode deltamem \
  --delta-heads o \
  --steer-gain "$steer_gain" \
  --output-fusion "$output_fusion" \
  --output-fusion-eps 1e-6 \
  --output-fusion-scale-max 10 \
  --steer-layers 0,3,6,9,12,15,18,21,24,27,30,33 \
  --backbone-window 0 \
  --max-chunk-tok 256 \
  --max-ctx-tok 4500 \
  --max-ans-tok 24 \
  --train-papers 800 \
  --val-papers 75 \
  --data-compose-seed 42 \
  --train-target-n 935 \
  --mix-temporal-n 0 \
  --max-yesno-frac 0.03 \
  --lr 5e-4 \
  --steps 156 \
  --batch-size 1 \
  --grad-accum 16 \
  --eval-every 156 \
  --max-new-tokens 24 \
  --save-steps 100,130 \
  --log-gradnorm \
  --output-dir "$output_dir" \
  --tag "$run_tag"
