#!/bin/bash
# Resumable GPU job queue for the downstream audit.
#
# Every job is one line in a TSV queue file:  <tag>\t<command...>
# A job is SKIPPED when out/<tag>.json already exists, so the whole queue can be
# re-run after a crash, a node reboot or a deadline stop without repeating work.
# Placement waits for a GPU with enough free memory instead of assuming one is
# idle — the OOM this guards against came from launching onto a card that was
# already at 78/80 GB.
#
# Usage:
#   bash scripts/run_downstream_queue.sh queue.tsv [MIN_FREE_MIB] [POLL_SEC]
set -u
QUEUE=${1:?usage: run_downstream_queue.sh <queue.tsv> [min_free_mib] [poll_sec]}
MIN_FREE=${2:-30000}
POLL=${3:-60}
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$REPO/out_downstream_audit_20260812}
PY=${PY:-/work/mingze/miniconda3/envs/deltamem/bin/python}
export PYTHONPATH=$REPO:$REPO/scripts
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
LEDGER=$OUT/gpu_job_ledger.tsv
[ -f "$LEDGER" ] || printf 'utc\tgpu\ttag\tstatus\n' > "$LEDGER"

pick_gpu() {  # echoes a gpu index with >= MIN_FREE MiB free, or nothing
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits |
    awk -F', ' -v need="$MIN_FREE" '($2-$3) >= need {print ($2-$3), $1}' |
    sort -rn | head -1 | awk '{print $2}'
}

while IFS=$'\t' read -r tag cmd; do
  [ -z "${tag:-}" ] && continue
  case "$tag" in \#*) continue ;; esac
  if [ -f "$OUT/$tag.json" ]; then
    echo "[skip] $tag (result exists)"
    printf '%s\t-\t%s\tSKIP\n' "$(date -u +%FT%TZ)" "$tag" >> "$LEDGER"
    continue
  fi
  gpu=""
  while [ -z "$gpu" ]; do
    gpu=$(pick_gpu)
    [ -z "$gpu" ] && { echo "[wait] no GPU with ${MIN_FREE}MiB free; sleeping ${POLL}s"; sleep "$POLL"; }
  done
  echo "[gpu$gpu] $tag"
  printf '%s\t%s\t%s\tSTART\n' "$(date -u +%FT%TZ)" "$gpu" "$tag" >> "$LEDGER"
  CUDA_VISIBLE_DEVICES=$gpu $PY $cmd > "$OUT/$tag.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && [ -f "$OUT/$tag.json" ]; then
    : > "$OUT/$tag.done"
    printf '%s\t%s\t%s\tDONE\n' "$(date -u +%FT%TZ)" "$gpu" "$tag" >> "$LEDGER"
  else
    printf '%s\t%s\t%s\tFAIL(rc=%s)\n' "$(date -u +%FT%TZ)" "$gpu" "$tag" "$rc" >> "$LEDGER"
    echo "[gpu$gpu] FAILED $tag rc=$rc — see $OUT/$tag.log"
  fi
done < "$QUEUE"
echo "queue complete: $QUEUE"
