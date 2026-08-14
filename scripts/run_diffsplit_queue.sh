#!/usr/bin/env bash
# Serial per-GPU queue runner for the diff-split comparison.
# Usage: run_diffsplit_queue.sh <GPU> <queuefile>
# queuefile: one line per job:  TAG<TAB>EXTRA_ARGS
set -u
GPU="$1"; QF="$2"
REPO=/work/mingze/Posthoc_differential_memory
O=$REPO/out_diffsplit_20260814
PY=/work/mingze/miniconda3/envs/deltamem/bin/python
LEDGER=$REPO/diff_split_gpu_job_ledger.tsv
COMMON="--model-path /work/mingze/models/Qwen3-4B-Instruct-2507 --data qasper \
--train-papers 800 --val-papers 75 --max-ctx-tok 4500 --train-target-n 935 \
--data-compose-seed 42 --steps 156 --batch-size 1 --grad-accum 16 \
--steer-lr 5e-4 --steer-lr-schedule constant --steer-layers 0,3,6,9,12,15,18,21,24,27,30,33 \
--grad-checkpointing true --val-every 52 --log-every 8 --val-loss-examples 32 \
--eval-at-start false --output-dir $O"
while IFS=$'\t' read -r TAG EXTRA; do
  [ -z "${TAG:-}" ] && continue
  case "$TAG" in \#*) continue;; esac
  if [ -f "$O/$TAG.json" ]; then echo "[gpu$GPU] SKIP $TAG (exists)"; continue; fi
  LOG="$O/logs/$TAG.log"
  echo "[gpu$GPU] START $TAG"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$TAG" "$$" "gpu$GPU" "$(date -u +%FT%TZ)" "RUNNING" "$LOG" >> "$LEDGER"
  CUDA_VISIBLE_DEVICES=$GPU $PY $REPO/scripts/dex_train_qasper.py $COMMON --tag "$TAG" $EXTRA > "$LOG" 2>&1
  RC=$?
  M=$($PY -c "import json,sys;d=json.load(open('$O/$TAG.json'));print('F1=%.4f val_loss=%.4f'%(d['final']['qa']['F1'],d['final']['val_loss']))" 2>/dev/null || echo "no-metrics")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$TAG" "$$" "gpu$GPU" "$(date -u +%FT%TZ)" "DONE(rc=$RC) $M" "$LOG" >> "$LEDGER"
  echo "[gpu$GPU] DONE $TAG rc=$RC $M"
done < "$QF"
echo "[gpu$GPU] QUEUE EMPTY"
