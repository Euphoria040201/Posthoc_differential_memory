"""Zero-shot eval of our (Qasper-trained) prefix-steer ckpt on LoCoMo.

Uses the repo's OFFICIAL LoCoMo protocol (prompt, history construction, metric).
Our backbone reads the full conversation history in-prompt (capped at
--max-context-tokens) with the memory steer active; then we score with the
official per-category metric.  Shard by global question index.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deltamem.eval.locomo_delta import load_locomo_samples
from deltamem.eval.locomo_protocol import (
    prepare_locomo_question, build_official_full_history_messages,
    canonicalize_locomo_prediction, score_locomo_prediction,
    OFFICIAL_MAX_NEW_TOKENS, OFFICIAL_ANSWER_RESERVE_TOKENS,
)
from deltamem.eval.steer_checkpoint import (
    load_steer_state_strict,
    restore_prefix_steer_config,
)
from deltamem.core.prefix_steer import (
    attach_prefix_steer, freeze_backbone_keep_steer,
    set_steer_segments, set_steer_enabled, set_mem_cache, set_window_only,
)
from deltamem.core.global_prefix import SEG_CTX, SEG_ANS
from deltamem.core.diff_split import set_diff_enabled as _set_diff_enabled


def get_dtype(n):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[n]


def load_ours(model, ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ck, dict) and "diff_config" in ck:
        # diff_split checkpoints are a different module family; reuse the single
        # loader in eval_ours_hotpotqa so all three evaluators stay in step.
        from eval_ours_hotpotqa import load_ours as _load_any
        return _load_any(model, ckpt_path)
    if not isinstance(ck, dict) or "cfg" not in ck or "state" not in ck:
        raise ValueError("prefix checkpoint must contain cfg and state mappings")
    cfg = restore_prefix_steer_config(ck["cfg"])
    attach_prefix_steer(model, cfg)
    freeze_backbone_keep_steer(model)
    load_steer_state_strict(model, ck["state"], label="ours-locomo")
    print(
        f"[ours-locomo] loaded {len(ck['state'])} steer tensors; cfg={cfg}",
        flush=True,
    )
    return cfg


@torch.no_grad()
def gen_ours(model, tok, input_ids, dev, max_new, eos, steer):
    # KV-cached generation: prefill once, then decode with cache (backbone + memory).
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
        nxt = int(lg.argmax())
        if nxt == eos:
            break
        out.append(nxt)
        o = model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=pkv, use_cache=True)
        pkv = o.past_key_values; lg = o.logits[0, -1]
    set_mem_cache(model, False)
    return tok.decode(out, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--attn-impl", default="sdpa"); ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--data-file", default="data/locomo10.json")
    ap.add_argument("--categories", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--max-conversations", type=int, default=None)
    ap.add_argument("--max-questions-per-conversation", type=int, default=None)
    ap.add_argument("--max-context-tokens", type=int, default=8000)
    ap.add_argument("--max-new-tokens", type=int, default=OFFICIAL_MAX_NEW_TOKENS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--conds", default="ours")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    # backbone_window is NOT in cfg; AUTO-restore from the ckpt's training args so a bw-trained
    # ckpt runs on the SAME bounded backbone at eval (else memory is silently bypassed).
    _train_bw = int((torch.load(args.ckpt, map_location="cpu").get("args", {}) or {}).get("backbone_window", 0) or 0)
    _kw = {}
    if _train_bw > 0:
        from transformers import AutoConfig
        _bc = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
        _tc = _bc.get_text_config() if hasattr(_bc, "get_text_config") else _bc
        _tc.sliding_window = _train_bw
        _tc.layer_types = ["sliding_attention"] * _tc.num_hidden_layers
        _kw["config"] = _bc
        print(f"[eval] BOUNDED backbone: window={_train_bw}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=get_dtype(args.dtype), attn_implementation=args.attn_impl,
        local_files_only=True, **_kw).to(dev).eval()
    load_ours(model, args.ckpt)
    eos = tok.eos_token_id

    samples = load_locomo_samples(Path(args.data_file), max_conversations=args.max_conversations,
                                  max_questions_per_conversation=args.max_questions_per_conversation,
                                  categories=args.categories)
    # flat list of (sample, qa, qidx)
    tasks = []
    for s in samples:
        for qi, qa in enumerate(s["qa"]):
            tasks.append((s, qa, qi))
    shard = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    print(f"[shard {args.shard}] {len(shard)}/{len(tasks)} questions", flush=True)

    conds = args.conds.split(",")
    agg = {c: {} for c in conds}  # cat -> list of scores
    recs = []
    for k, (s, qa, qi) in enumerate(shard):
        spec = prepare_locomo_question(qa, sample_id=s.get("sample_id", "s"), question_index=qi, seed=args.seed)
        msgs = build_official_full_history_messages(s, tok, spec, max_context_tokens=args.max_context_tokens,
                                                    answer_reserve_tokens=OFFICIAL_ANSWER_RESERVE_TOKENS)
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        input_ids = enc["input_ids"].to(dev)
        row = {"cat": int(qa["category"]), "q": qa["question"][:60], "gold": str(qa.get("answer", ""))}
        for c in conds:
            set_window_only(model, c.endswith("window_only"))   # ours_window_only = steer w/o prefix read
            raw = gen_ours(model, tok, input_ids, dev, args.max_new_tokens, eos, steer=(c != "base"))
            canon = canonicalize_locomo_prediction(raw, spec)
            sc = score_locomo_prediction(qa, canon)
            row[c + "_pred"] = canon  # full prediction for content/format analysis
            agg[c].setdefault(int(qa["category"]), []).append(sc)
            row[c] = canon[:40]
        recs.append(row)
        if (k + 1) % 25 == 0:
            m = {c: round(sum(v for vs in agg[c].values() for v in vs) / max(1, sum(len(vs) for vs in agg[c].values())), 4) for c in conds}
            print(f"[shard {args.shard}] {k+1}/{len(shard)} overall={m}", flush=True)

    out = {"shard": args.shard, "num_shards": args.num_shards, "by_cat": {}, "records": recs}
    for c in conds:
        flat = [v for vs in agg[c].values() for v in vs]
        out["by_cat"][c] = {str(cat): {"n": len(vs), "sum": sum(vs)} for cat, vs in agg[c].items()}
        out["by_cat"][c]["overall"] = {"n": len(flat), "sum": sum(flat)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"))
    for c in conds:
        o = out["by_cat"][c]["overall"]
        print(f"[shard {args.shard}] {c} overall={o['sum']/max(1,o['n']):.4f} (n={o['n']})", flush=True)


if __name__ == "__main__":
    main()
