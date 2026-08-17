#!/usr/bin/env python
"""Audit issue 1: measure TOTAL per-sequence inference state, not just the KV cache.

The 2026-08-14 report leaned on "KV cache unchanged / KV-cache-free".  That is
true and is a real property, but it is not the same claim as "no extra inference
state": `DiffSplitAttention` keeps `_read_h` (w-1 hidden states) and `_read_v`
(w-1 backbone V rows) per layer for the decode path.  This script measures both
methods' real footprint on the actual Qwen3-4B so the report can state the number
instead of an adjective.

Writes out_cpt_20260817/inference_state.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def cache_bytes(cache):
    n = 0
    for layer in cache.layers:
        n += layer.keys.numel() * layer.keys.element_size()
        n += layer.values.numel() * layer.values.element_size()
    return n


def module_state_bytes(model, cls):
    """Bytes of per-sequence tensor state held on the wrapper modules themselves."""
    total, detail = 0, {}
    for name, m in model.named_modules():
        if not isinstance(m, cls):
            continue
        for k, v in vars(m).items():
            if torch.is_tensor(v):
                b = v.numel() * v.element_size()
                total += b
                detail.setdefault(k, {"tensors": 0, "bytes": 0, "shape": list(v.shape)})
                detail[k]["tensors"] += 1
                detail[k]["bytes"] += b
    return total, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/work/mingze/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--layers", default="0,3,6,9,12,15,18,21,24,27,30,33")
    ap.add_argument("--rank", type=int, default=177)
    ap.add_argument("--read-dim", type=int, default=128)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--out", default="/work/mingze/Posthoc_differential_memory/out_cpt_20260817/inference_state.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, DynamicCache
    from deltamem.core.diff_split import DiffSplitAttention, attach_diff_split
    from deltamem.core.lowrank_split import LowRankSplitAttention, attach_lowrank_split

    layers = [int(x) for x in args.layers.split(",")]
    dev = "cuda"
    x = torch.randint(0, 1000, (1, args.seq_len), device=dev)
    res = {"model": args.model, "seq_len": args.seq_len, "layers": layers,
           "window": args.window, "rank": args.rank}

    def run(build, label, cls):
        m = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
        if build:
            build(m)
        cache = DynamicCache()
        with torch.no_grad():
            m(input_ids=x, past_key_values=cache, use_cache=True)
            # one decode step, which is what populates any private reader cache
            m(input_ids=x[:, -1:], past_key_values=cache, use_cache=True)
        kv = cache_bytes(cache)
        mod, detail = module_state_bytes(m, cls) if cls else (0, {})
        res[label] = {"kv_cache_bytes": kv, "module_state_bytes": mod,
                      "total_state_bytes": kv + mod,
                      "module_state_detail": detail,
                      "overhead_vs_kv_pct": round(100.0 * mod / kv, 3) if kv else None}
        print(f"[{label}] kv={kv:,}B module={mod:,}B total={kv+mod:,}B", flush=True)
        del m
        torch.cuda.empty_cache()

    run(None, "base", None)
    run(lambda m: attach_diff_split(m, layers, read_dim=args.read_dim,
                                    window=args.window, gamma=1.0),
        "localreader_split", DiffSplitAttention)
    run(lambda m: attach_lowrank_split(m, layers, rank=args.rank, gamma=1.0),
        "lowrank_split", LowRankSplitAttention)

    res["summary"] = (
        f"At seq_len {args.seq_len}, LocalRead adds "
        f"{res['localreader_split']['module_state_bytes']:,} B of per-sequence "
        f"reader state ({res['localreader_split']['overhead_vs_kv_pct']}% of its KV "
        f"cache) on top of an unchanged KV cache; the token-local low-rank split "
        f"adds {res['lowrank_split']['module_state_bytes']:,} B.")
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("\n" + res["summary"])


if __name__ == "__main__":
    main()
