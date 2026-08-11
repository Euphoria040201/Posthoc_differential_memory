#!/usr/bin/env bash
set -uo pipefail

# Formal PersonaMem-v2 dev arm: 719 train personas / 80 unseen dev personas,
# official Qwen-VeRL reader prompt, four-choice CE + identity contrast,
# 300 optimizer updates x 64 labels = 19,200 label exposures for every arm.
# Only the memory architecture changes between arms.
#
# usage: run_personamem_formal_arm.sh GPU ARM [SEED]
#   ARM in {poolsteer, pool, prefixonly, standard, hybridpart_pooldrop05,
#           hybridpart_g1}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 GPU ARM [SEED]" >&2
  exit 2
fi

gpu="$1"
arm="$2"
seed="${3:-1}"

repo_dir="/work/mingze/delta-mem"
python_bin="/work/mingze/miniconda3/envs/deltamem/bin/python"
model_dir="/work/mingze/models/Qwen3-4B-PersonaMem-SFT"
out_dir="${repo_dir}/out_personamem"

case "$arm" in
  poolsteer)
    arch=(--memory-mode pooled_steer --P 0 --head-dim 84 --history-pool attn
          --read-mode broadcast)
    tag="poolsteer_d84"
    ;;
  pool)
    arch=(--memory-mode prefix --P 64 --head-dim 64 --read-mode pool)
    tag="p64_pool"
    ;;
  prefixonly)
    arch=(--memory-mode prefix --P 64 --head-dim 64 --read-mode prefix_only)
    tag="p64_prefixonly"
    ;;
  standard)
    # Qasper-native reader: one softmax over [written prefix slots ; local
    # sliding window of the READ sequence] plus the max-prefix bonus term.
    arch=(--memory-mode prefix --P 64 --head-dim 64 --read-mode standard
          --sliding-window 256)
    tag="p64_standard"
    ;;
  hybridpart_g1)
    arch=(--memory-mode hybrid --P 64 --head-dim 64 --history-pool attn
          --read-mode pooled_plus_prefix --prefix-write-layout partitioned
          --hybrid-prefix-gate-mode fixed --hybrid-prefix-gate-init 1.0)
    tag="hybridpart_g1"
    ;;
  hybridpart_pooldrop05)
    # Branch dropout: the pooled summary is removed for a whole correct+donor
    # microstep with p=0.5 so the optimizer cannot route everything through it.
    arch=(--memory-mode hybrid --P 64 --head-dim 64 --history-pool attn
          --read-mode pooled_plus_prefix --prefix-write-layout partitioned
          --hybrid-prefix-gate-mode fixed --hybrid-prefix-gate-init 1.0
          --hybrid-pool-drop-prob 0.5)
    tag="hybridpart_g1_pooldrop05"
    ;;
  *)
    echo "unknown ARM: $arm" >&2
    exit 2
    ;;
esac

run_name="official_dev_sft_${tag}_idc10_m1_offprompt_budget19200_k4_s${seed}"
ckpt="${out_dir}/${run_name}.pt"
log="${out_dir}/${run_name}.log"

cd "$repo_dir"
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

resume=()
if [[ -f "${ckpt%.pt}.resume.pt" ]]; then
  resume=(--resume-checkpoint "${ckpt%.pt}.resume.pt")
fi

exec "$python_bin" scripts/personamem_prefix_steer.py \
  --model-path "$model_dir" \
  --data-root data/personamem_v2 \
  --eval-split val \
  --persona-holdout-size 80 \
  --dev-source train+val \
  --max-personas 0 \
  --max-queries 0 \
  --max-history-tokens 37000 \
  --history-truncation tail \
  --queries-per-write 4 \
  --train-sampler cyclic_label_budget \
  --labels-per-update 64 \
  --num-swap-derangements 3 \
  --reader-protocol official_qwen \
  --task-loss four_choice \
  --identity-contrast-lambda 10 \
  --identity-margin 1 \
  --steps 300 \
  --eval-every 0 \
  --save-every 25 \
  --lr 1e-4 \
  --prefix-lr 1e-3 \
  --seed "$seed" \
  "${arch[@]}" \
  "${resume[@]}" \
  --output "$ckpt" \
  >> "$log" 2>&1
