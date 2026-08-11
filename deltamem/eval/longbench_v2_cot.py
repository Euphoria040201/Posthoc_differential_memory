"""LongBench-v2 eval with the OFFICIAL prompt + OFFICIAL 2-stage CoT
(THUDM/LongBench pred.py + prompts/0shot_cot*.txt).

Stage 1 (0shot_cot): doc + question + choices + "Let's think step by step:"  -> CoT (<=1024 tok)
Stage 2 (0shot_cot_ans): NO doc, question + choices + CoT -> "The correct answer is (X)" (<=128 tok)
Answer extraction + truncation (head+tail) are copied verbatim from the official pred.py.

Supports base/SFT (no adapter) and delta-mem (--adapter-dir). For delta-mem the online
state is reset before each generate() call (each prefill re-reads its prompt).
"""
from __future__ import annotations

import argparse
import json
import re
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

# ---- official templates (THUDM/LongBench/prompts) ----
TEMPLATE_COT = """Please read the following text and answer the questions below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Let’s think step by step:"""

TEMPLATE_COT_ANS = """Please read the following text and answer the questions below.

The text is too long and omitted here.

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Let’s think step by step: $COT$

Based on the above, what is the single, most likely answer choice? Format your response as follows: "The correct answer is (insert answer here)"."""


def get_dtype(name):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def fill(template, item, cot=None):
    s = (template.replace("$DOC$", item["context"].strip())
                 .replace("$Q$", item["question"].strip())
                 .replace("$C_A$", item["choice_A"].strip())
                 .replace("$C_B$", item["choice_B"].strip())
                 .replace("$C_C$", item["choice_C"].strip())
                 .replace("$C_D$", item["choice_D"].strip()))
    if cot is not None:
        s = s.replace("$COT$", cot)
    return s


def extract_answer(response):
    response = response.replace("*", "")
    m = re.search(r"The correct answer is \(([A-D])\)", response)
    if m:
        return m.group(1)
    m = re.search(r"The correct answer is ([A-D])", response)
    if m:
        return m.group(1)
    return None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--adapter-dir", default=None)
    p.add_argument("--data", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-implementation", default="flash_attention_2")
    p.add_argument("--max-tokens", type=int, default=128000, help="official truncation length (full prompt)")
    p.add_argument("--cot-new-tokens", type=int, default=1024)
    p.add_argument("--ans-new-tokens", type=int, default=128)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
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
        print("[lbv2-cot] delta-mem adapter attached", flush=True)
    print(f"[lbv2-cot] {'DELTA' if is_delta else 'BASE'} model on {dev}", flush=True)

    data = json.load(open(args.data))
    end = args.end if args.end is not None else len(data)
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
                        done.add(str(json.loads(line)["_id"]))
                    except Exception:
                        pass
    fout = open(out_path, "a")

    half = args.max_tokens // 2
    pad = tok.pad_token_id or tok.eos_token_id

    def generate(user_text, max_new):
        # official truncation on the full prompt text
        ids = tok(user_text, add_special_tokens=False).input_ids
        orig = len(ids)
        truncated = orig > args.max_tokens
        if truncated:
            ids = ids[:half] + ids[-half:]
            user_text = tok.decode(ids, skip_special_tokens=True)
        enc = tok.apply_chat_template(
            [{"role": "user", "content": user_text}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )
        input_ids = enc["input_ids"].to(dev)
        attn = enc.get("attention_mask")
        attn = attn.to(dev) if attn is not None else torch.ones_like(input_ids)
        if is_delta:
            reset_delta_mem_states(model)
        try:
            with torch.inference_mode():
                out = model.generate(input_ids=input_ids, attention_mask=attn,
                                     max_new_tokens=max_new, do_sample=False, pad_token_id=pad)
            txt = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        except torch.cuda.OutOfMemoryError:
            txt = "[OOM]"
            torch.cuda.empty_cache()
        return txt, orig, truncated

    n_done = n_ok = 0
    t0 = time.time()
    for idx in range(args.start, end):
        if args.num_shards > 1 and idx % args.num_shards != args.shard:
            continue
        s = data[idx]
        if str(s["_id"]) in done:
            continue
        t1 = time.time()
        cot, ctx_tokens, truncated = generate(fill(TEMPLATE_COT, s), args.cot_new_tokens)
        resp, _, _ = generate(fill(TEMPLATE_COT_ANS, s, cot=cot), args.ans_new_tokens)
        gen_s = round(time.time() - t1, 1)
        pred = extract_answer(resp)
        gold = str(s["answer"]).strip().upper()
        ok = pred == gold
        n_done += 1
        n_ok += int(ok)
        rec = {"idx": idx, "_id": s["_id"], "domain": s["domain"], "length": s["length"],
               "ctx_tokens": ctx_tokens, "truncated": truncated, "gold": gold, "pred": pred,
               "cot": cot, "resp": resp, "correct": ok, "gen_s": gen_s}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[{idx}] {ctx_tokens//1000}K{'T' if truncated else ' '} {s['length']} "
              f"gold={gold} pred={pred} {'OK' if ok else 'x'} ({gen_s}s) "
              f"acc={n_ok}/{n_done}={100*n_ok/n_done:.1f}%", flush=True)

    mins = (time.time() - t0) / 60
    acc = (100 * n_ok / n_done) if n_done else 0.0
    print(f"=== done: {n_ok}/{n_done} = {acc:.1f}%  ({mins:.1f} min) ===", flush=True)


if __name__ == "__main__":
    main()
