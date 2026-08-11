#!/usr/bin/env bash
set -uo pipefail

# Stage-1 fusion matrix: one frozen SWA control, five ways of using it.
#   usage: run_dex_stage1.sh GPU SEED "arm [arm ...]"
#
# learned_diff is the only arm that trains (one scalar per steer layer, 12 total);
# it is given the same 156-update budget as the sidecar itself so its lambda has a
# fair chance to move.  lambda is UNCONSTRAINED in sign: if additive is optimal it
# can learn a negative value and reproduce fixed_add.  Every other arm is eval-only,
# which is what makes this comparison cheap and exactly matched -- all five share one
# checkpoint, one backbone and one eval protocol.

if [[ $# -lt 3 ]]; then
  echo "usage: $0 GPU SEED 'arm [arm ...]'" >&2
  exit 2
fi

gpu="$1"; seed="$2"; arms="$3"
repo=/work/mingze/delta-mem
py=/work/mingze/miniconda3/envs/deltamem/bin/python
ckpt="$repo/out_dex/dex_swa_steer_constant_slr5e-4_s${seed}_steer.pt"

cd "$repo" || exit 1
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$repo"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -f "$ckpt" ]]; then
  echo "missing checkpoint: $ckpt" >&2
  exit 2
fi

for arm in $arms; do
  tag="stage1_${arm}_s${seed}"
  if [[ -f "out_dex_fusion/${tag}.json" ]]; then
    echo "[stage1] skip $tag (already done)"
    continue
  fi
  extra=()
  if [[ "$arm" == "learned_diff" ]]; then
    extra+=(--fusion-steps 156 --fusion-lr 1e-2)
  fi
  if [[ "$arm" == "variance_diff" ]]; then
    extra+=(--calibrate-batches 64)
  fi
  echo "[stage1] $(date +%H:%M:%S) start $tag on GPU $gpu"
  "$py" scripts/dex_stage1_fusion.py \
    --steer-ckpt "$ckpt" --arm "$arm" --seed "$seed" \
    --output-dir out_dex_fusion --tag "$tag" "${extra[@]}" \
    > "out_dex_fusion/${tag}.log" 2>&1
  echo "[stage1] $(date +%H:%M:%S) done  $tag rc=$?"
done
