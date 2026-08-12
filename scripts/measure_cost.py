#!/usr/bin/env python
"""Cost accounting required by the reporting spec: tokens, state size, latency.

Measures, on real HotpotQA items with the real wrapper:
  * context tokens, tokens the WRITER actually consumes, tokens the READER sees
  * whether anything was truncated
  * state size in slots and bytes
  * write latency, per-query read latency
  * amortized latency when one state serves 1 / 5 / 20 queries
  * trainable parameter count and effective correction norm ||g*C|| / ||Z||
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

from eval_ours_hotpotqa import load_ours  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from deltamem.core.prefix_steer import (  # noqa: E402
    clear_frozen_memory, collect_fusion_norms, iter_steer_modules,
    set_collect_fusion_norms, set_steer_enabled, set_steer_segments,
    set_write_freeze, is_steer_param_name,
)
from deltamem.core.global_prefix import SEG_CTX  # noqa: E402
from deltamem.eval.benchmark_compare import (  # noqa: E402
    HOTPOTQA_PROMPT_TEMPLATE, build_hotpotqa_context, load_hotpotqa,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-path", default="/work/mingze/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True).to("cuda").eval()
    cfg = load_ours(model, args.ckpt)
    mods = list(iter_steer_modules(model))
    trainable = sum(p.numel() for n, p in model.named_parameters() if is_steer_param_name(n))

    data = load_hotpotqa(cache_dir=Path.home() / ".cache/huggingface/datasets",
                         max_samples=args.n, seed=1234, local_files_only=True)

    def setseg(L):
        set_steer_segments(model, torch.full((1, L), SEG_CTX, dtype=torch.long, device="cuda"),
                           torch.ones(1, L, dtype=torch.bool, device="cuda"))

    def write(ctx_text):
        ids = tok(ctx_text, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
        clear_frozen_memory(model); set_write_freeze(model, True); setseg(ids.shape[1])
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        set_write_freeze(model, False)
        return ids.shape[1], dt

    def read(ctx_text, q, steer=True):
        set_steer_enabled(model, steer)
        p = HOTPOTQA_PROMPT_TEMPLATE.format(context=ctx_text, question=q)
        ids = tok.apply_chat_template([{"role": "user", "content": p}],
                                      add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True)["input_ids"].cuda()
        n_in = ids.shape[1]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(args.max_new):
            setseg(ids.shape[1])
            with torch.no_grad():
                lg = model(input_ids=ids, use_cache=False).logits[0, -1]
            nx = int(lg.argmax())
            if nx == tok.eos_token_id:
                break
            ids = torch.cat([ids, torch.tensor([[nx]], device="cuda")], dim=1)
        torch.cuda.synchronize()
        return n_in, time.perf_counter() - t0

    rows = []
    for it in data:
        ctx = build_hotpotqa_context(it)
        q = str(it["question"]).strip()
        w_tok, w_dt = write(ctx)
        r_tok_state, r_dt_state = read("", q)            # state-only read
        r_tok_full, r_dt_full = read(ctx, q)             # full-context read
        clear_frozen_memory(model)
        _, b_dt = read(ctx, q, steer=False)              # frozen base, full context
        state_elems = sum(m._frozen_prefix.numel() for m in mods
                          if getattr(m, "_frozen_prefix", None) is not None)
        rows.append({
            "context_tokens": w_tok, "writer_tokens": w_tok,
            "reader_tokens_state_only": r_tok_state,
            "reader_tokens_fullctx": r_tok_full,
            "write_s": w_dt, "read_state_only_s": r_dt_state,
            "read_fullctx_s": r_dt_full, "base_fullctx_s": b_dt,
        })

    # state size: written prefix across all steered layers
    ctx = build_hotpotqa_context(data[0])
    write(ctx)
    state_elems = sum(m._frozen_prefix.numel() for m in mods
                      if getattr(m, "_frozen_prefix", None) is not None)
    state_bytes = sum(m._frozen_prefix.numel() * m._frozen_prefix.element_size()
                      for m in mods if getattr(m, "_frozen_prefix", None) is not None)
    # effective correction norm over a real forward
    set_collect_fusion_norms(model, True)
    read(ctx, str(data[0]["question"]).strip())
    norms = collect_fusion_norms(model).get("mean", {})
    set_collect_fusion_norms(model, False)
    g = cfg.steer_gain if hasattr(cfg, "steer_gain") else None
    if g is not None and "norm_C" in norms and "norm_Z" in norms:
        norms["effective_gC_over_Z"] = g * norms["norm_C"] / max(norms["norm_Z"], 1e-9)
    if g is not None and "norm_WC" in norms and "norm_WZ" in norms:
        norms["effective_gWC_over_WZ"] = g * norms["norm_WC"] / max(norms["norm_WZ"], 1e-9)

    mean = lambda k: sum(r[k] for r in rows) / len(rows)  # noqa: E731
    w, rs, rf, bf = mean("write_s"), mean("read_state_only_s"), mean("read_fullctx_s"), mean("base_fullctx_s")
    payload = {
        "ckpt": args.ckpt, "n": len(rows),
        "trainable_params": trainable,
        "num_prefix_tokens": getattr(cfg, "num_prefix_tokens", None),
        "steer_layers": len(mods),
        "steer_gain": g,
        "tokens": {
            "context_mean": mean("context_tokens"),
            "writer_sees_mean": mean("writer_tokens"),
            "reader_sees_state_only_mean": mean("reader_tokens_state_only"),
            "reader_sees_fullctx_mean": mean("reader_tokens_fullctx"),
            "truncation": "none - full official 10-paragraph context is written and read",
        },
        "state": {"elements": state_elems, "bytes": state_bytes,
                  "bytes_per_layer": state_bytes // max(len(mods), 1)},
        "latency_s": {"write": w, "read_state_only": rs, "read_fullctx": rf,
                      "base_fullctx": bf,
                      "ours_fullctx_overhead_vs_base": rf - bf},
        "amortized_state_only_s_per_query": {
            "1_query": w + rs, "5_queries": w / 5 + rs, "20_queries": w / 20 + rs,
            "base_fullctx_per_query": bf,
        },
        "fusion_norms": norms,
        "per_example": rows,
    }
    json.dump(payload, open(args.output, "w"), indent=2)
    print(json.dumps({k: v for k, v in payload.items() if k != "per_example"}, indent=2))


if __name__ == "__main__":
    main()
