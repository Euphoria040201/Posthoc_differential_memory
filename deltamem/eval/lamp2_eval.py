"""LaMP-2 (personalized movie tagging) eval for base / SFT / delta-mem.

Each question gives the user's tagging history (profile) + a new movie description;
predict one of 15 tags. Personalization signal = the profile, placed in context.
Metric: accuracy + macro-F1. Same 8-way-shardable harness as the other evals.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deltamem.core.delta import (
    HFDeltaMemConfig, attach_delta_mem, load_delta_mem_adapter, reset_delta_mem_states,
)

TAGS = ["comedy", "sci-fi", "violence", "classic", "twist ending", "based on a book",
        "true story", "action", "dark comedy", "romance", "dystopia",
        "thought-provoking", "social commentary", "psychology", "fantasy"]


def get_dtype(name):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def clean(s):
    s = str(s).strip()
    return s[2:].strip() if s.startswith("x ") else s


def build_prompt(item, max_profile):
    prof = item["profile"][:max_profile]
    hist = "\n".join(f'- "{clean(p["description"])[:300]}" -> {p["tag"]}' for p in prof)
    tags = ", ".join(TAGS)
    return (
        "Based on a user's movie-tagging history, assign exactly ONE tag to a new movie.\n"
        f"Allowed tags: {tags}.\n\n"
        f"The user's tagging history:\n{hist}\n\n"
        f'New movie description: "{clean(item["input"])}"\n'
        "Answer with exactly one tag from the allowed list, nothing else."
    )


def extract_tag(text):
    t = text.strip().lower()
    # exact tag appearing in output (prefer longest match to disambiguate comedy/dark comedy)
    found = [tag for tag in TAGS if re.search(r"\b" + re.escape(tag) + r"\b", t)]
    if found:
        return max(found, key=len)
    return t.split("\n")[0].strip()[:30]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--adapter-dir", default=None)
    p.add_argument("--questions", required=True)
    p.add_argument("--outputs", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-implementation", default="flash_attention_2")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--max-profile", type=int, default=11)
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
        attn_implementation=args.attn_implementation, device_map={"": args.device})
    model.eval()
    dev = next(model.parameters()).device
    is_delta = bool(args.adapter_dir)
    if is_delta:
        cfg = HFDeltaMemConfig.from_pretrained(args.adapter_dir)
        attach_delta_mem(model, cfg)
        load_delta_mem_adapter(model, args.adapter_dir)
        print("[lamp2] delta-mem adapter attached", flush=True)
    print(f"[lamp2] {'DELTA' if is_delta else 'BASE'} model on {dev}", flush=True)

    q = json.load(open(args.questions))
    gmap = {str(g["id"]): g["output"] for g in json.load(open(args.outputs))["golds"]}

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
    n = correct = 0
    for idx, item in enumerate(q):
        if args.num_shards > 1 and idx % args.num_shards != args.shard:
            continue
        qid = str(item["id"])
        if qid in done or qid not in gmap:
            continue
        prompt = build_prompt(item, args.max_profile)
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_tensors="pt", return_dict=True)
        input_ids = enc["input_ids"].to(dev)
        attn = enc.get("attention_mask")
        attn = attn.to(dev) if attn is not None else torch.ones_like(input_ids)
        if is_delta:
            reset_delta_mem_states(model)
        try:
            with torch.inference_mode():
                o = model.generate(input_ids=input_ids, attention_mask=attn,
                                   max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=pad)
            pred_raw = tok.decode(o[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        except torch.cuda.OutOfMemoryError:
            pred_raw = "[OOM]"; torch.cuda.empty_cache()
        pred = extract_tag(pred_raw)
        gold = gmap[qid]
        ok = pred == gold
        n += 1; correct += int(ok)
        fout.write(json.dumps({"id": qid, "gold": gold, "pred": pred, "raw": pred_raw, "correct": ok}, ensure_ascii=False) + "\n")
        fout.flush()
        if n % 25 == 0:
            print(f"[{idx}] n={n} acc={100*correct/n:.1f}%", flush=True)
    print(f"=== done: acc={100*correct/max(n,1):.2f}% (n={n}) ===", flush=True)


if __name__ == "__main__":
    main()
