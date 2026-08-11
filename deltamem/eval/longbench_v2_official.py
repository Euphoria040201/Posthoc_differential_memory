"""LongBench-v2 direct-inference eval for delta-mem vs base.

Reconstructed from longbench_v2.pyc, with ONE behavioral change:
the old "skip samples whose context >= --max-tokens" is replaced by the
OFFICIAL LongBench-v2 truncation (THUDM/LongBench pred.py):

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]

i.e. keep the first half and last half of the document, drop the middle.
Nothing is skipped any more, so the long-context bucket is actually evaluated.

Prompt / chat template / answer extraction / output record format are kept
byte-identical to the original eval so results stay comparable with prior runs.
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

PROMPT_SUFFIX = (
    "\n\n{q}\n\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n\n"
    "choose the correct answer, make sure you only output the LETTER "
    "(for example: Therefore, the answer is X)"
)
SYS = "You are a helpful assistant. Read the document and answer the multiple-choice question."


def get_dtype(name: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def extract_letter(text: str) -> str:
    t = (text or "").replace("*", "")
    m = re.search(r"\b([A-D])\b", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    ms = re.findall(r"[A-D]", t.upper())
    return ms[-1] if ms else ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--adapter-dir", default=None, help="attach delta-mem adapter (delta condition)")
    p.add_argument("--data", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default=None, help="e.g. 'auto' to span GPUs for long ctx")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-implementation", default="flash_attention_2")
    # NOTE: this is now the OFFICIAL truncation length (max_len), NOT a skip threshold.
    p.add_argument("--max-tokens", type=int, default=128000, help="official truncation length (head+tail)")
    p.add_argument("--reserve", type=int, default=2000, help="tokens reserved for question/choices/template/answer")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--shard", type=int, default=0, help="this process index for stride sharding")
    p.add_argument("--num-shards", type=int, default=1, help="total processes; sample taken when idx %% num_shards == shard")
    p.add_argument("--output", required=True)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--done-file", default=None, help="skip _ids already present in this jsonl (for merge/parallel)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_device = args.device

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model_kwargs = dict(dtype=get_dtype(args.dtype), attn_implementation=args.attn_implementation)
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    else:
        model_kwargs["device_map"] = {"": args.device}
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval()
    in_device = next(model.parameters()).device

    is_delta = bool(args.adapter_dir)
    if is_delta:
        cfg = HFDeltaMemConfig.from_pretrained(args.adapter_dir)
        attach_delta_mem(model, cfg)
        load_delta_mem_adapter(model, args.adapter_dir)
        print("[lbv2] delta-mem adapter attached", flush=True)
    print(f"[lbv2] {'DELTA' if is_delta else 'BASE'} model on {in_device}", flush=True)

    data = json.load(open(args.data))
    end = args.end if args.end is not None else len(data)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    # skip ids from an external done-file (e.g. the previous run's results)
    if args.done_file and Path(args.done_file).exists():
        for line in open(args.done_file):
            line = line.strip()
            if line:
                try:
                    done.add(str(json.loads(line)["_id"]))
                except Exception:
                    pass
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
    budget = args.max_tokens - args.reserve
    n_done = 0
    n_ok = 0
    t0 = time.time()

    for idx in range(args.start, end):
        if args.num_shards > 1 and idx % args.num_shards != args.shard:
            continue
        s = data[idx]
        if str(s["_id"]) in done:
            continue

        ctx_ids = tok(s["context"], add_special_tokens=False).input_ids
        orig_len = len(ctx_ids)

        # ---- OFFICIAL LongBench-v2 truncation: head half + tail half ----
        truncated = orig_len > budget
        if truncated:
            half = budget // 2
            kept = ctx_ids[:half] + ctx_ids[-half:]
            ctx_text = tok.decode(kept, skip_special_tokens=True)
        else:
            ctx_text = s["context"]

        suffix = PROMPT_SUFFIX.format(
            q=s["question"], a=s["choice_A"], b=s["choice_B"], c=s["choice_C"], d=s["choice_D"]
        )
        messages = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": ctx_text + suffix},
        ]
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        input_ids = enc["input_ids"].to(in_device)
        attn_mask = enc.get("attention_mask")
        attn_mask = attn_mask.to(in_device) if attn_mask is not None else torch.ones_like(input_ids)

        if is_delta:
            reset_delta_mem_states(model)

        t1 = time.time()
        try:
            with torch.inference_mode():
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            gen = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        except torch.cuda.OutOfMemoryError:
            gen = "[OOM]"
            torch.cuda.empty_cache()
        gen_s = round(time.time() - t1, 1)

        pred = extract_letter(gen)
        gold = str(s["answer"]).strip().upper()
        ok = pred == gold
        n_done += 1
        n_ok += int(ok)

        rec = {
            "idx": idx,
            "_id": s["_id"],
            "domain": s["domain"],
            "length": s["length"],
            "ctx_tokens": orig_len,
            "truncated": truncated,
            "gold": gold,
            "pred": pred,
            "raw": gen,
            "correct": ok,
            "gen_s": gen_s,
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        tt = "T" if truncated else " "
        print(
            f"[{idx}] {orig_len // 1000}K{tt} {s['length']} gold={gold} pred={pred} "
            f"{'OK' if ok else 'x'} ({gen_s}s) run_acc={n_ok}/{n_done}={100*n_ok/n_done:.1f}%",
            flush=True,
        )

    mins = (time.time() - t0) / 60
    acc = (100 * n_ok / n_done) if n_done else 0.0
    print(f"=== done: {n_ok}/{n_done} = {acc:.1f}%  ({mins:.1f} min) ===", flush=True)


if __name__ == "__main__":
    main()
