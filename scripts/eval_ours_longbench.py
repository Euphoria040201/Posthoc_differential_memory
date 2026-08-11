"""LongBench-v2 (MCQ A/B/C/D) eval for OUR prefix-steer ckpt vs base, on small GPUs.
Reuses load_ours/gen_ours (eval_ours_hotpotqa) + the official prompt/truncation/letter-extraction
(deltamem.eval.longbench_v2_official). Contexts are truncated head+tail to --max-tokens so a 4B
model fits 16GB; set --max-tokens lower for tighter cards / --length-filter for the short subset.
"""
import argparse, sys, json, time
from pathlib import Path
import torch
sys.path.insert(0, "scripts")
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval_ours_hotpotqa import load_ours, gen_ours
from deltamem.core.prefix_steer import set_steer_enabled
from deltamem.eval.longbench_v2_official import PROMPT_SUFFIX, SYS, extract_letter, get_dtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--device-map", default="")
    ap.add_argument("--max-tokens", type=int, default=8000, help="official head+tail truncation length")
    ap.add_argument("--reserve", type=int, default=2000)
    ap.add_argument("--max-new-tokens", type=int, default=12)
    ap.add_argument("--length-filter", default="", help="comma list of LongBench-v2 'length' to keep (short/medium/long)")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all")
    ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--conds", default="base,ours")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    eos = tok.eos_token_id
    kw = dict(dtype=get_dtype(args.dtype), attn_implementation=args.attn_impl, local_files_only=True)
    if args.device_map:
        kw["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kw)
    if not args.device_map:
        model = model.to(args.device)
    load_ours(model, args.ckpt); model.eval()
    dev = str(next(model.parameters()).device) if args.device_map else args.device

    from datasets import load_dataset
    data = load_dataset("THUDM/LongBench-v2", split="train")
    keep = set(x.strip() for x in args.length_filter.split(",") if x.strip())
    tasks = [s for s in data if (not keep or s.get("length") in keep)]
    tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    if args.max_samples: tasks = tasks[:args.max_samples]
    print(f"[lbv2-ours] {len(tasks)} samples (filter={keep or 'all'}), max_tokens={args.max_tokens}", flush=True)

    budget = args.max_tokens - args.reserve
    conds = args.conds.split(",")
    res = {c: {"ok": 0, "n": 0} for c in conds}
    recs = []
    for k, s in enumerate(tasks):
        ctx_ids = tok(s["context"], add_special_tokens=False).input_ids
        if len(ctx_ids) > budget:
            half = budget // 2
            ctx_text = tok.decode(ctx_ids[:half] + ctx_ids[-half:], skip_special_tokens=True)
        else:
            ctx_text = s["context"]
        suffix = PROMPT_SUFFIX.format(q=s["question"], a=s["choice_A"], b=s["choice_B"],
                                      c=s["choice_C"], d=s["choice_D"])
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": ctx_text + suffix}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True)["input_ids"].to(dev)
        gold = str(s["answer"]).strip().upper()
        row = {"_id": s["_id"], "gold": gold, "length": s.get("length")}
        for c in conds:
            set_steer_enabled(model, c != "base")
            gen = gen_ours(model, tok, ids, dev, args.max_new_tokens, eos)
            pred = extract_letter(gen)
            row[c] = pred
            res[c]["n"] += 1; res[c]["ok"] += int(pred == gold)
        recs.append(row)
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(tasks)}  " + " ".join(f"{c}={res[c]['ok']/max(1,res[c]['n']):.3f}" for c in conds), flush=True)
    set_steer_enabled(model, True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"acc": {c: res[c]["ok"]/max(1,res[c]["n"]) for c in conds}, "n": len(recs), "records": recs},
              open(args.output, "w"))
    print("\n=== LongBench-v2 accuracy ===")
    for c in conds:
        print(f"  {c:>6}: {res[c]['ok']/max(1,res[c]['n']):.4f}  ({res[c]['ok']}/{res[c]['n']})")


if __name__ == "__main__":
    main()
