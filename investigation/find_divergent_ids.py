"""Regenerate deg_ids.json: HotpotQA question ids where the WORST seed (ms3) degenerates
to yes/no but the CHAMPION (spread12 seed0, 0.6815) gives a real answer. These are the
docs where the L33-prefix-collapse pathology bites (see deg_test.py). Needs the two eval
record dirs (champion + ms3) under ../eval_records/."""
import json, glob, sys
from pathlib import Path
sys.path.insert(0, "..")  # deltamem importable when run from investigation/
from deltamem.eval.benchmark_compare import normalize_hotpotqa_answer as N, load_hotpotqa

def load(pat):
    d = {}
    for f in sorted(glob.glob(pat)):
        for r in json.load(open(f)).get("records", []):
            d[r["id"]] = r
    return d

champ = load("../eval_records/champion_spread12/*.json")
worst = load("../eval_records/worst_ms3/*.json")
deg = [i for i in set(champ) & set(worst)
       if N(worst[i]["ours"]) in ("yes", "no") and N(champ[i]["ours"]) not in ("yes", "no")
       and N(worst[i]["gold"]) not in ("yes", "no")]
print(f"champion degenerate-divergent ids: {len(deg)}")
data = load_hotpotqa(cache_dir=Path.home()/".cache/huggingface/datasets", max_samples=100000,
                     seed=42, local_files_only=True)
byid = {(it.get("id") or it.get("_id")): it for it in data}
sel = [i for i in deg if i in byid][:8]
json.dump({"ids": sel}, open("deg_ids.json", "w"))
print(f"wrote deg_ids.json with {len(sel)} ids")
