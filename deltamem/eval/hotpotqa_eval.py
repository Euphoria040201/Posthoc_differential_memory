"""HotpotQA (distractor / validation) eval for base / SFT / delta-mem.

Reuses the project's OFFICIAL hotpot prompt + EM/F1 metric from benchmark_compare,
in the same 8-way-shardable, resumable harness style as the LongBench-v2 scripts.

base/SFT: plain model (no --adapter-dir). delta-mem: + adapter, online state reset
before each question (each prompt re-read independently).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deltamem.core.delta import (
    HFDeltaMemConfig,
    attach_delta_mem,
    load_delta_mem_adapter,
    reset_delta_mem_states,
)
from deltamem.eval.benchmark_compare import (
    HOTPOTQA_PROMPT_TEMPLATE,
    build_hotpotqa_context,
    extract_first_line,
    hotpotqa_exact_match,
    hotpotqa_f1,
    load_hotpotqa,
)


def get_dtype(name):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--adapter-dir", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-implementation", default="flash_attention_2")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--max-samples", type=int, default=None, help="cap dataset size (None=full 7405)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", default=str(Path.home() / ".cache/huggingface/datasets"))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--output", required=True)
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=get_dtype(args.dtype),
        attn_implementation=args.attn_implementation, device_map={"": args.device},
    )
    model.eval()
    dev = next(model.parameters()).device

    is_delta = bool(args.adapter_dir)
    if is_delta:
        cfg = HFDeltaMemConfig.from_pretrained(args.adapter_dir)
        attach_delta_mem(model, cfg)
        load_delta_mem_adapter(model, args.adapter_dir)
        print("[hotpot] delta-mem adapter attached", flush=True)
    print(f"[hotpot] {'DELTA' if is_delta else 'BASE'} model on {dev}", flush=True)

    data = load_hotpotqa(cache_dir=Path(args.cache_dir), max_samples=args.max_samples,
                         seed=args.seed, local_files_only=True)
    print(f"[hotpot] loaded {len(data)} samples", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        if args.no_resume:
            out_path.rename(str(out_path) + ".bak." + str(int(time.time())))
        else:
            for line in open(out_path):
                line = line.strip()
                if line:
                    try:
                        done.add(str(json.loads(line)["id"]))
                    except Exception:
                        pass
    fout = open(out_path, "a")
    pad = tok.pad_token_id or tok.eos_token_id

    n = n_em = 0
    f1_sum = 0.0
    t0 = time.time()
    for idx, item in enumerate(data):
        if args.num_shards > 1 and idx % args.num_shards != args.shard:
            continue
        if str(item["id"]) in done:
            continue
        prompt = HOTPOTQA_PROMPT_TEMPLATE.format(
            context=build_hotpotqa_context(item), question=str(item["question"]).strip(),
        )
        enc = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )
        input_ids = enc["input_ids"].to(dev)
        attn = enc.get("attention_mask")
        attn = attn.to(dev) if attn is not None else torch.ones_like(input_ids)
        if is_delta:
            reset_delta_mem_states(model)
        t1 = time.time()
        try:
            with torch.inference_mode():
                out = model.generate(input_ids=input_ids, attention_mask=attn,
                                     max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=pad)
            pred = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        except torch.cuda.OutOfMemoryError:
            pred = "[OOM]"
            torch.cuda.empty_cache()
        gen_s = round(time.time() - t1, 2)
        first = extract_first_line(pred)
        gold = str(item["answer"]).strip()
        em = hotpotqa_exact_match(first, gold)
        f1 = hotpotqa_f1(first, gold)
        n += 1
        n_em += int(em)
        f1_sum += f1
        rec = {"idx": idx, "id": item["id"], "level": item.get("level"), "type": item.get("type"),
               "question": item["question"], "answer": gold, "prediction": pred,
               "extracted": first, "em": em, "f1": round(f1, 4), "correct": em, "gen_s": gen_s}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        if n % 50 == 0:
            print(f"[{idx}] n={n} EM={100*n_em/n:.1f}% F1={100*f1_sum/n:.1f}%", flush=True)

    mins = (time.time() - t0) / 60
    print(f"=== done: n={n} EM={100*n_em/max(n,1):.2f}% F1={100*f1_sum/max(n,1):.2f}%  ({mins:.1f} min) ===", flush=True)


if __name__ == "__main__":
    main()
