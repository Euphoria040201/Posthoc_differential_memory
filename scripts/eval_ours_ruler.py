"""RULER (official 13-task suite, via simonjegou/ruler) eval for OUR prefix-steer ckpt vs base.
At 4096 ctx a 4B model fits 16GB. Official RULER protocol: prompt = context + question +
answer_prefix (the model continues), metric = string-match RECALL of the gold answer list
(mean over gold answers of `ans.lower() in prediction.lower()`), averaged per task then overall.
"""
import argparse, sys, json, ast
from pathlib import Path
import torch
sys.path.insert(0, "scripts")
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval_ours_hotpotqa import load_ours
from deltamem.core.prefix_steer import set_steer_segments, set_steer_enabled, set_mem_cache, set_window_only
from deltamem.core.global_prefix import SEG_CTX, SEG_ANS
from deltamem.core.diff_split import set_diff_enabled as _set_diff_enabled


def get_dtype(n): return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[n]


def parse_gold(a):
    if isinstance(a, list): return [str(x) for x in a]
    try:
        v = ast.literal_eval(a)
        return [str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)]
    except Exception:
        return [str(a)]


def recall(pred, gold_list, task=""):
    """OFFICIAL RULER string match: qa_1/qa_2 use string_match_part (ANY gold alias in the
    prediction scores 1.0); all synthetic tasks use string_match_all (mean over golds).
    The old uniform mean-over-golds under-scored qa (aliases averaged instead of any-of)."""
    p = pred.lower()
    hits = [1.0 if str(g).lower() in p else 0.0 for g in gold_list]
    if not hits:
        return 0.0
    return max(hits) if task.startswith("qa") else sum(hits) / len(hits)


@torch.no_grad()
def generate(model, tok, input_ids, dev, max_new, eos, steer):
    # KV-cached generation (prefill once, then decode) -- fast AND correct. NO newline early-stop:
    # multi-answer RULER tasks (multivalue/multiquery) and base's "colon\n\nanswer" formatting put
    # the answer AFTER a newline, so any newline-stop truncates unfairly. Generate to max_new / eos.
    set_steer_enabled(model, steer)
    # set_steer_enabled() does not reach DiffSplitAttention; without this the
    # base condition silently re-runs the split model as its own baseline.
    _set_diff_enabled(model, steer)
    set_mem_cache(model, steer)
    L = input_ids.shape[1]
    set_steer_segments(model, torch.full((1, L), SEG_CTX, dtype=torch.long, device=dev),
                       torch.ones(1, L, dtype=torch.bool, device=dev))
    out = []
    o = model(input_ids=input_ids, use_cache=True); pkv = o.past_key_values; lg = o.logits[0, -1]
    for _ in range(max_new):
        nx = int(lg.argmax())
        if nx == eos: break
        out.append(nx)
        o = model(input_ids=torch.tensor([[nx]], device=dev), past_key_values=pkv, use_cache=True)
        pkv = o.past_key_values; lg = o.logits[0, -1]
    set_mem_cache(model, False)
    return tok.decode(out, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="4096", help="RULER context length config (4096/8192/...)")
    ap.add_argument("--data-dir", default="",
                    help="directory of OFFICIALLY generated per-task test.jsonl "
                         "(NVIDIA/RULER generator via NeMo-Skills prepare.py). "
                         "Overrides the simonjegou/ruler HF mirror.")
    ap.add_argument("--attn-impl", default="sdpa"); ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--device-map", default="")
    ap.add_argument("--tasks", default="", help="comma subset of tasks (empty = all 13)")
    ap.add_argument("--per-task", type=int, default=0, help="cap samples per task (0 = all 500)")
    ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--conds", default="base,ours")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    eos = tok.eos_token_id
    kw = dict(dtype=get_dtype(args.dtype), attn_implementation=args.attn_impl, local_files_only=True)
    if args.device_map: kw["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kw)
    if not args.device_map: model = model.to(args.device)
    load_ours(model, args.ckpt); model.eval()
    dev = str(next(model.parameters()).device) if args.device_map else args.device

    if args.data_dir:
        # OFFICIAL generation: one test.jsonl per task, produced by NVIDIA/RULER's own
        # generator via NeMo-Skills `nemo_skills/dataset/ruler/prepare.py` with OUR
        # tokenizer.  Samples, prompts, gold answers and answer_prefix are untouched.
        import json as _json
        from pathlib import Path as _P
        data = []
        for task_dir in sorted(_P(args.data_dir).iterdir()):
            if not (task_dir / "test.jsonl").exists():
                continue
            for line in open(task_dir / "test.jsonl"):
                row = _json.loads(line)
                row["task"] = task_dir.name
                # the official generator names the gold list "outputs"; the HF mirror
                # calls it "answer".  Normalise WITHOUT touching the values.
                if "answer" not in row and "outputs" in row:
                    row["answer"] = row["outputs"]
                data.append(row)
        print(f"[ruler] loaded {len(data)} OFFICIAL samples from {args.data_dir}", flush=True)
    else:
        from datasets import load_dataset
        data = load_dataset("simonjegou/ruler", args.config, split="test")
    keep = set(x.strip() for x in args.tasks.split(",") if x.strip())
    idx_by_task = {}
    rows = []
    for i, s in enumerate(data):
        t = s["task"]
        if keep and t not in keep: continue
        idx_by_task.setdefault(t, 0)
        if args.per_task and idx_by_task[t] >= args.per_task: continue
        idx_by_task[t] += 1
        rows.append(s)
    rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    print(f"[ruler-{args.config}] {len(rows)} samples over {len(idx_by_task)} tasks", flush=True)

    conds = args.conds.split(",")
    agg = {c: {} for c in conds}   # task -> [recalls]
    recs = []
    for k, s in enumerate(rows):
        gold = parse_gold(s["answer"])
        # HF mirror splits the prompt into context/question; the official generator
        # emits ONE `input` field that already contains the question.  Use each
        # source's own field(s) verbatim -- never re-wrap the official prompt.
        user = (s["input"] if "input" in s
                else s["context"] + "\n\n" + s["question"])
        chat = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True)["input_ids"]
        ap_ids = tok(s["answer_prefix"], add_special_tokens=False, return_tensors="pt")["input_ids"]
        ids = torch.cat([chat, ap_ids], dim=1).to(dev)   # prefill the answer_prefix
        # OFFICIAL per-task generation budget from the dataset (niah 128 / cwe 120 / fwe 50 /
        # vt 30 / qa 32). The old min(mnt,48) cap truncated verbose multi-answer formats and
        # manufactured a fake base-vs-ours gap on multiquery (base=0.500 exactly, 100/100).
        mnt = int(s.get("max_new_tokens", 128) or 128)
        row = {"task": s["task"], "gold": gold}
        for c in conds:
            set_window_only(model, c.endswith("window_only"))
            pred = generate(model, tok, ids, dev, mnt, eos, steer=(c != "base"))
            r = recall(pred, gold, s["task"])
            agg[c].setdefault(s["task"], []).append(r)
            row[c] = round(r, 3); row[c + "_pred"] = pred[:60]
        recs.append(row)
        if (k + 1) % 20 == 0:
            m = {c: sum(v for vs in agg[c].values() for v in vs)/max(1, sum(len(vs) for vs in agg[c].values())) for c in conds}
            print(f"  {k+1}/{len(rows)}  " + " ".join(f"{c}={m[c]:.3f}" for c in conds), flush=True)
    set_steer_enabled(model, True); _set_diff_enabled(model, True); set_window_only(model, False)

    out = {"config": args.config, "by_task": {}, "overall": {}, "records": recs}
    for c in conds:
        pt = {t: sum(v)/len(v) for t, v in agg[c].items()}
        out["by_task"][c] = pt
        out["overall"][c] = sum(pt.values())/max(1, len(pt))   # RULER = mean over tasks
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"))
    print(f"\n=== RULER-{args.config} (recall, mean over tasks) ===")
    tasks = sorted(agg[conds[0]].keys())
    print(f"{'task':>16} " + " ".join(f"{c:>8}" for c in conds))
    for t in tasks:
        print(f"{t:>16} " + " ".join(f"{out['by_task'][c].get(t,0):>8.3f}" for c in conds))
    print(f"{'OVERALL':>16} " + " ".join(f"{out['overall'][c]:>8.3f}" for c in conds))


if __name__ == "__main__":
    main()
