#!/usr/bin/env bash
# Run one CPT arm on one GPU, recording a machine-readable ledger row.
# Usage: cpt_run_job.sh <repo_dir> <gpu> <tag> <args...>
set -uo pipefail

REPO="$1"; GPU="$2"; TAG="$3"; shift 3
PY=/work/mingze/miniconda3/envs/deltamem/bin/python
LEDGER="$REPO/out_cpt_20260817/cpt_job_ledger.tsv"
LOGDIR="$REPO/out_cpt_20260817/logs"
LOG="$LOGDIR/$TAG.log"
mkdir -p "$LOGDIR"

if [ ! -f "$LEDGER" ]; then
  printf 'tag\thost\tgpu\tpid\tstart_utc\tend_utc\tsecs\trc\tstatus\tartifact\tlog\tcmd\n' > "$LEDGER"
fi

START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
T0=$(date +%s)
CMD="$PY $REPO/scripts/cpt_train.py --tag $TAG --data-dir $REPO/out_cpt_20260817 --out-dir $REPO/out_cpt_20260817 $*"

cd "$REPO" || exit 1
CUDA_VISIBLE_DEVICES="$GPU" $CMD > "$LOG" 2>&1 &
PID=$!
printf '%s\t%s\t%s\t%s\t%s\t\t\t\tRUNNING\t%s\t%s\t%s\n' \
  "$TAG" "$(hostname)" "$GPU" "$PID" "$START" \
  "$REPO/out_cpt_20260817/$TAG.json" "$LOG" "$CMD" >> "$LEDGER"

wait $PID
RC=$?
END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
SECS=$(( $(date +%s) - T0 ))
NLL=$(grep -oE 'FINAL nll=[0-9.]+' "$LOG" | tail -1 | cut -d= -f2)
STATUS="DONE(rc=$RC)"
[ -n "${NLL:-}" ] && STATUS="$STATUS nll=$NLL"
[ "$RC" -ne 0 ] && STATUS="FAILED(rc=$RC)"

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$TAG" "$(hostname)" "$GPU" "$PID" "$START" "$END" "$SECS" "$RC" "$STATUS" \
  "$REPO/out_cpt_20260817/$TAG.json" "$LOG" "$CMD" >> "$LEDGER"
exit $RC
