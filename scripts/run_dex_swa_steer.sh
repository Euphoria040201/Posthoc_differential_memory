#!/usr/bin/env bash
set -uo pipefail

# swa_steer runner: frozen backbone + trained SWA memory sidecar, as the
# parameter-efficient counterpart of attn_only.
#   usage: run_dex_swa_steer.sh GPU "seed [seed ...]"
#
# Everything except the trainable set is held at the DEX main-table protocol
# (935 Qasper examples, 156 updates, grad-accum 16, cosine + 3% warmup, greedy
# 24-token eval over all 187 val examples), so the F1 lands in the same table.
#
# Two deliberate deviations, both forced by what is being trained:
#   * the sidecar LR is 5e-4, not the backbone's 2e-5.  2e-5 was tuned on
#     attn_only, i.e. on pretrained weights; the sidecar is randomly initialised
#     and this repo's SWA line tuned 5e-4 on this exact module.
#   * the sidecar config reproduces out_swa_sharedv/p0_mainv_h1_fixed_g01_s1
#     (P=0 window-only, main_v, delta_heads=o, gain 0.1, every third layer),
#     which scored .2968 on this same val protocol at seed 1.

if [[ $# -lt 2 ]]; then
  echo "usage: $0 GPU 'seed [seed ...]'" >&2
  exit 2
fi

gpu="$1"; seeds="$2"
repo=/work/mingze/delta-mem
py=/work/mingze/miniconda3/envs/deltamem/bin/python
steer_lr="${STEER_LR:-5e-4}"

cd "$repo" || exit 1
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for seed in $seeds; do
  tag="${DEX_TAG_PREFIX:-}dex_swa_steer_${STEER_FUSION:-fixed}_${STEER_SCHED:-constant}_slr${steer_lr}_s${seed}"
  if [[ -f "out_dex/${tag}.json" ]]; then
    echo "[swa_steer] skip $tag (already done)"
    continue
  fi
  echo "[swa_steer] $(date +%H:%M:%S) start $tag on GPU $gpu"
  "$py" scripts/dex_train_qasper.py \
    --variant swa_steer --seed "$seed" --steps 156 \
    --steer-lr-schedule "${STEER_SCHED:-constant}" \
    --lr 2e-5 --steer-lr "$steer_lr" \
    --steer-layers 0,3,6,9,12,15,18,21,24,27,30,33 \
    --steer-gain 0.1 --steer-delta-heads o --steer-output-fusion "${STEER_FUSION:-fixed}" \
    --steer-value-source main_v --steer-window 256 \
    --steer-mem-heads 1 --steer-mem-head-dim 128 --steer-prefix-tokens 0 \
    --grad-accum 16 --batch-size 1 --warmup-ratio 0.03 --grad-clip 1.0 \
    --weight-decay 0.0 --grad-checkpointing true \
    --val-every 26 --log-every 4 --val-loss-examples 32 --eval-at-start false \
    --output-dir out_dex --tag "$tag" \
    > "out_dex/${tag}.log" 2>&1
  echo "[swa_steer] $(date +%H:%M:%S) done  $tag rc=$?"
done
