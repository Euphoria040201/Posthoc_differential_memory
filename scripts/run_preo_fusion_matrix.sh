#!/bin/bash
# Pre-O fusion study: train the SWA sidecar at o_fusion_position=pre_o (3 seeds),
# then run the stage-1 fusion arms on each FROZEN checkpoint, plus the strict
# post_o_projected control (same checkpoint, mathematically identical fusion for
# the linear modes -- any difference is implementation path, not math).
#
# The post-o reference table is NOT rerun: it already exists as
# out_dex_fusion/stage1_{base,fixed_add,fixed_sub,learned_diff,variance_diff}_s{0,1,2}
# (see dex_control_report.md section 13) and this study must not overwrite it.
#
# Protocol is copied verbatim from the post-o line:
#   train : dex_train_qasper.py --variant swa_steer, 156 updates, steer-lr 5e-4
#           constant, gain 0.1, delta_heads o, main_v, layers 0,3,...,33
#   stage1: same data composition, 187-example val F1, learned_diff trains
#           156 lambda steps at lr 1e-2, variance_diff calibrates 64 batches
#
# Resumable: a job is skipped when its output json already exists.
# Usage: bash scripts/run_preo_fusion_matrix.sh [SEEDS] [TRAIN_GPUS] [STAGE_GPUS]
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/work/mingze/miniconda3/envs/deltamem/bin/python}
OUT=${OUT:-out_dex_fusion}
SEEDS=(${1:-0 1 2})
TRAIN_GPUS=(${2:-0 1 2})
STAGE_GPUS=(${3:-0 1 2 3 4 5 6 7})
export HF_HUB_OFFLINE=1
mkdir -p "$OUT"

train_one() {  # seed gpu
  local s=$1 g=$2 tag="preo_swa_steer_s$1"
  [ -f "$OUT/${tag}.json" ] && { echo "[skip] $tag"; return 0; }
  CUDA_VISIBLE_DEVICES=$g $PY scripts/dex_train_qasper.py \
    --variant swa_steer --seed "$s" --steps 156 \
    --steer-lr-schedule constant --lr 2e-5 --steer-lr 5e-4 \
    --steer-layers 0,3,6,9,12,15,18,21,24,27,30,33 \
    --steer-gain 0.1 --steer-delta-heads o \
    --steer-output-fusion fixed --steer-o-fusion-position pre_o \
    --steer-value-source main_v --steer-window 256 \
    --steer-mem-heads 1 --steer-mem-head-dim 128 --steer-prefix-tokens 0 \
    --grad-accum 16 --batch-size 1 --warmup-ratio 0.03 --grad-clip 1.0 \
    --weight-decay 0.0 --grad-checkpointing true --val-every 26 --log-every 4 \
    --val-loss-examples 32 --eval-at-start false \
    --output-dir "$OUT" --tag "$tag" > "$OUT/${tag}.log" 2>&1
}

echo "== phase A: pre_o sidecar training, seeds ${SEEDS[*]} =="
i=0
for s in "${SEEDS[@]}"; do
  g=${TRAIN_GPUS[$((i % ${#TRAIN_GPUS[@]}))]}
  train_one "$s" "$g" &
  i=$((i + 1))
done
wait
for s in "${SEEDS[@]}"; do
  [ -f "$OUT/preo_swa_steer_s${s}_steer.pt" ] || { echo "FATAL: seed $s ckpt missing"; exit 1; }
done

echo "== phase B: stage-1 fusion arms on the frozen pre_o checkpoints =="
JOBS=()
for s in "${SEEDS[@]}"; do
  ck="$OUT/preo_swa_steer_s${s}_steer.pt"
  # pre_o arms (position taken from the checkpoint)
  JOBS+=("--steer-ckpt $ck --arm base          --seed $s --tag s1preo_base_s$s")
  JOBS+=("--steer-ckpt $ck --arm fixed_add     --seed $s --tag s1preo_fixed_add_s$s")
  JOBS+=("--steer-ckpt $ck --arm fixed_sub     --seed $s --tag s1preo_fixed_sub_s$s")
  JOBS+=("--steer-ckpt $ck --arm learned_diff  --seed $s --fusion-steps 156 --fusion-lr 1e-2 --tag s1preo_learned_diff_s$s")
  JOBS+=("--steer-ckpt $ck --arm variance_diff --seed $s --calibrate-batches 64 --tag s1preo_variance_diff_s$s")
  # strict control: SAME checkpoint, fused as post_o_projected
  JOBS+=("--steer-ckpt $ck --arm fixed_add     --seed $s --o-fusion-position post_o_projected --tag s1ctlproj_fixed_add_s$s")
  JOBS+=("--steer-ckpt $ck --arm fixed_sub     --seed $s --o-fusion-position post_o_projected --tag s1ctlproj_fixed_sub_s$s")
  JOBS+=("--steer-ckpt $ck --arm learned_diff  --seed $s --fusion-steps 156 --fusion-lr 1e-2 --o-fusion-position post_o_projected --tag s1ctlproj_learned_diff_s$s")
done

run_queue() {  # gpu; consumes jobs whose index % ngpus matches this queue slot
  local slot=$1 g=$2 n=${#JOBS[@]}
  for ((j = slot; j < n; j += ${#STAGE_GPUS[@]})); do
    local args=(${JOBS[$j]})
    local tag=${args[-1]}
    [ -f "$OUT/${tag}.json" ] && { echo "[skip] $tag"; continue; }
    echo "[gpu$g] $tag"
    CUDA_VISIBLE_DEVICES=$g $PY scripts/dex_stage1_fusion.py \
      ${JOBS[$j]} --output-dir "$OUT" > "$OUT/${tag}.log" 2>&1 \
      || echo "[gpu$g] FAILED $tag (rc=$?)"
  done
}

slot=0
for g in "${STAGE_GPUS[@]}"; do
  run_queue "$slot" "$g" &
  slot=$((slot + 1))
done
wait
echo "== done; results in $OUT/s1preo_*.json and $OUT/s1ctlproj_*.json =="
