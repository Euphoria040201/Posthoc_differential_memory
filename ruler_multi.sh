#!/bin/bash
# RULER on node12: given config length, all 13 tasks, per-task N, sharded across GPUS + merge.
# GPU quota is 0-3 (2026-05-22+); default to those. Override: GPUS="0 1" bash ruler_multi.sh 16384 100
set -u; cd /work/mingze/delta-mem; PY=/work/mingze/miniconda3/envs/deltamem/bin/python
CFG=${1:-16384}; PT=${2:-100}
GPUS=(${GPUS:-0 1 2 3}); NS=${#GPUS[@]}
OUT=/tmp/ruler_$CFG; rm -rf $OUT; mkdir -p $OUT
for i in "${!GPUS[@]}"; do
  env PYTHONPATH=.:scripts CUDA_VISIBLE_DEVICES=${GPUS[$i]} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup $PY scripts/eval_ours_ruler.py --ckpt ckpts/spread12_swa_ckpt.pt --config $CFG \
    --per-task $PT --num-shards $NS --shard $i --conds base,ours \
    --output $OUT/shard$i.json > $OUT/shard$i.log 2>&1 &
done
wait; echo "SHARDS_DONE_$CFG"
$PY - "$OUT" "$CFG" <<'PY'
import json, collections, glob, sys
out, cfg = sys.argv[1], sys.argv[2]
recs=[]
for f in glob.glob(out+"/shard*.json"): recs+=json.load(open(f))["records"]
conds=["base","ours"]; agg={c:collections.defaultdict(list) for c in conds}
for r in recs:
    for c in conds: agg[c][r["task"]].append(r[c])
by={c:{t:sum(v)/len(v) for t,v in agg[c].items()} for c in conds}
ov={c:sum(by[c].values())/max(1,len(by[c])) for c in conds}
print(f"\n=== RULER-{cfg} (recall, mean over tasks; n={len(recs)}) ===")
print(f"{'task':>16}   base    ours")
for t in sorted(agg["base"]): print(f"{t:>16} {by['base'][t]:>7.3f} {by['ours'][t]:>7.3f}")
print(f"{'OVERALL':>16} {ov['base']:>7.3f} {ov['ours']:>7.3f}")
PY
echo "RULER_${CFG}_MERGED"
