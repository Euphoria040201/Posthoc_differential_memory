#!/usr/bin/env bash
set -euo pipefail

# Train one Qasper ablation and immediately run the same checkpoint on the
# HotpotQA-500 screening split and the 10-conversation LoCoMo screening set.
# The winning setting is promoted to the full HotpotQA/LoCoMo runs separately.

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
steer_gain="${7:-}"
prefix_tokens="${8:-}"

repo_dir="/work/mingze/delta-mem"
python_bin="/work/mingze/miniconda3/envs/deltamem/bin/python"
model_dir="/work/mingze/models/Qwen3-4B-Instruct-2507"
output_dir="${repo_dir}/out_swa_sharedv"
ckpt="${output_dir}/${run_tag}_ckpt.pt"

export CUDA_VISIBLE_DEVICES="$physical_gpu"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH="${repo_dir}${PYTHONPATH:+:${PYTHONPATH}}"

train_args=(
  "$physical_gpu" "$run_tag" "$value_source" "$output_fusion" "$run_seed" "$mem_heads"
)
if [[ -n "$prefix_tokens" && -z "$steer_gain" ]]; then
  echo "PREFIX_TOKENS requires GAIN to be given explicitly" >&2
  exit 2
fi
if [[ -n "$steer_gain" ]]; then
  train_args+=("$steer_gain")
fi
if [[ -n "$prefix_tokens" ]]; then
  train_args+=("$prefix_tokens")
fi
bash "${repo_dir}/scripts/run_qasper_sharedv_ablation.sh" "${train_args[@]}"

cd "$repo_dir"

"$python_bin" scripts/eval_ours_hotpotqa.py \
  --model-path "$model_dir" \
  --ckpt "$ckpt" \
  --attn-impl sdpa \
  --dtype bfloat16 \
  --max-samples 500 \
  --seed 42 \
  --max-new-tokens 32 \
  --conds base,ours \
  --write-pass inline \
  --output "${output_dir}/${run_tag}_hotpot500.json"

"$python_bin" scripts/eval_ours_locomo.py \
  --model-path "$model_dir" \
  --ckpt "$ckpt" \
  --attn-impl sdpa \
  --dtype bfloat16 \
  --data-file data/locomo10.json \
  --categories 1 2 3 4 \
  --max-conversations 10 \
  --max-questions-per-conversation 20 \
  --max-context-tokens 8000 \
  --seed 42 \
  --conds base,ours \
  --output "${output_dir}/${run_tag}_locomo10x20.json"
