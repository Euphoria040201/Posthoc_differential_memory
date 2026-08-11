"""FAITHFUL delta-Mem memory-agent eval for MemoryAgentBench (write-then-drop).

Unlike official_memory_agent_bench.py (which puts the FULL context in every answer
prompt -> memory is redundant), this mirrors the PAPER / training protocol:
  1. WRITE: ingest the context into the delta-Mem recurrent state (one forward),
     then DROP the raw context (KV cache discarded; only the O(r^2) state kept).
  2. READ: for each query, load the state and generate from a QUERY-ONLY prompt
     (no context in the prompt) -> the delta state is the only carrier of memory.

This is the protocol where memory is load-bearing, so it is the correct setting to
test whether delta-Mem's memory actually works / reproduces the paper's MAB gains.

base / plainsft models have no delta state -> they are NOT run here; use the
long-context port (official_memory_agent_bench.py) for those baselines.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from deltamem.core.delta import (
    load_delta_mem_online_state,
    reset_delta_mem_states,
    set_delta_mem_write_enabled,
)
from deltamem.runtime.session import DeltaMemChatSession
from deltamem.eval.common import load_delta_model_and_tokenizer
from deltamem.eval.official_eval_utils import (
    DEFAULT_MEMORY_AGENT_BENCH_ROOT,
    truncate_text_by_tokens,
)
from deltamem.eval.official_memory_agent_bench import (
    OFFICIAL_SOURCE_CONFIGS,
    build_context_chunks,
    build_dataset_config,
    build_memorized_context,
    build_query_answer_pairs,
    load_mab_eval_utils,
    load_source_rows,
    resolve_source_config,
    summarize_metrics,
)
from deltamem.eval.official_memory_agent_bench_templates import get_template


def parse_args():
    p = argparse.ArgumentParser(description="Faithful write-then-drop delta-Mem MAB eval")
    p.add_argument("--split", required=True, choices=["Accurate_Retrieval", "Test_Time_Learning", "Long_Range_Understanding", "Conflict_Resolution"])
    p.add_argument("--source", required=True, choices=sorted(OFFICIAL_SOURCE_CONFIGS))
    p.add_argument("--model-path", required=True)
    p.add_argument("--delta-adapter-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-implementation", default="flash_attention_2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-test-samples", type=int, default=0)
    p.add_argument("--generation-max-length", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=0)
    p.add_argument("--context-max-length", type=int, default=0, help="cap context tokens written to state (0=use source default, capped at write-window)")
    p.add_argument("--write-window", type=int, default=131072, help="max tokens ingested in a single forward; longer contexts are chunked")
    p.add_argument("--hub-cache-dir", default=None)
    p.add_argument("--datasets-cache-dir", default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--external-memory-agent-bench-root", type=Path, default=DEFAULT_MEMORY_AGENT_BENCH_ROOT)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-jsonl", type=Path)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--no-chat-template", action="store_true")
    return p.parse_args()


@torch.inference_mode()
def write_context_to_state(session, model, tokenizer, device, memorized_context, *, write_window):
    """Ingest memorized_context into the delta state in <=write_window-token chunks,
    accumulating the online state across chunks and DROPPING the KV cache between
    chunks (only the O(r^2) recurrent state is retained). Returns delta_state dict."""
    reset_delta_mem_states(model)
    ids = tokenizer(memorized_context, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    total = ids.size(1)
    set_delta_mem_write_enabled(model, True)
    start = 0
    while start < total:
        chunk = ids[:, start:start + write_window]
        # fresh session per chunk so KV does not accumulate; delta online-state persists
        # in the model across forwards (we never reset between chunks).
        sess = DeltaMemChatSession(model=model, tokenizer=tokenizer, device=device)
        sess.past_key_values = None
        sess.processed_input_ids = None
        sess._ingest_full_ids(chunk)
        start += write_window
    state = {name: t.detach().clone() for name, t in _online_state(model).items()}
    return state, total


def _online_state(model):
    from deltamem.core.delta import get_delta_mem_online_state
    return get_delta_mem_online_state(model)


@torch.inference_mode()
def answer_from_state(model, tokenizer, device, delta_state, messages, *, max_new_tokens):
    reset_delta_mem_states(model)
    load_delta_mem_online_state(model, delta_state)
    set_delta_mem_write_enabled(model, False)
    try:
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(device)
        out = model.generate(
            input_ids=enc.input_ids, attention_mask=enc.attention_mask,
            do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
    finally:
        set_delta_mem_write_enabled(model, True)
    gen = out[:, enc.input_ids.shape[1]:]
    text = tokenizer.decode(gen[0], skip_special_tokens=True).strip()
    return text, int(enc.input_ids.shape[1]), int(gen.shape[1])


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    cfg = resolve_source_config(args)
    if args.max_test_samples > 0:
        cfg["max_test_samples"] = args.max_test_samples
    dataset_config = build_dataset_config(args, cfg)
    eval_utils = load_mab_eval_utils(args.external_memory_agent_bench_root)

    rows = load_source_rows(args, cfg)
    context_chunks = build_context_chunks(rows, chunk_size=cfg["chunk_size"], eval_utils_module=eval_utils)
    qa_pairs_all = [build_query_answer_pairs(r, source=args.source) for r in rows]

    model, tokenizer, _ = load_delta_model_and_tokenizer(
        model_path=args.model_path, adapter_dir=Path(args.delta_adapter_dir),
        device=args.device, dtype=args.dtype, attn_implementation=args.attn_implementation,
    )
    model.eval()
    system_message = get_template(args.source, "system", "Long_context_agent_deltamem")
    cap = args.context_max_length if args.context_max_length > 0 else cfg["context_max_length"]

    session = DeltaMemChatSession(model=model, tokenizer=tokenizer, device=args.device)
    metrics = defaultdict(list)
    results = []
    records = []
    if args.output_jsonl:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        jl = open(args.output_jsonl, "w")
    else:
        jl = None

    for cidx, (chunks, qa_pairs) in enumerate(zip(context_chunks, qa_pairs_all)):
        memorized = build_memorized_context(args.source, chunks)
        if cap > 0:
            memorized = truncate_text_by_tokens(memorized, tokenizer=tokenizer, max_tokens=cap, keep="tail")
        t0 = time.time()
        delta_state, written = write_context_to_state(
            session, model, tokenizer, args.device, memorized, write_window=args.write_window)
        write_time = time.time() - t0
        print(f"[ctx {cidx}] wrote {written} tokens to state in {write_time:.1f}s, {len(qa_pairs)} queries", flush=True)
        for qidx, (query, answer, qa_pair_id) in enumerate(qa_pairs):
            msgs = [{"role": "system", "content": system_message}, {"role": "user", "content": query}]
            pred, in_len, out_len = answer_from_state(model, tokenizer, args.device, delta_state, msgs,
                                     max_new_tokens=cfg["generation_max_length"])
            before = len(results)
            metrics, results = eval_utils.metrics_summarization(
                {"output": pred, "input_len": in_len, "output_len": out_len,
                 "memory_construction_time": write_time, "query_time_len": 0},
                query, answer, dataset_config, metrics, results,
                query_id=before, qa_pair_id=qa_pair_id)
            rec = results[-1]
            rec.update({"context_id": cidx, "question_index": qidx, "query": query,
                        "split": args.split, "source": args.source, "memory_write_time": write_time,
                        "written_tokens": written})
            records.append(rec)
            if jl:
                jl.write(json.dumps(rec, ensure_ascii=False) + "\n"); jl.flush()
    if jl:
        jl.close()

    keep = {"exact_match", "f1", "substring_exact_match", "rougeL_f1", "rougeL_recall",
            "eventqa_recall", "recsys_recall@1", "recsys_recall@5", "recsys_recall@10"}
    merged = defaultdict(list)
    for r in records:
        for k, v in r.items():
            if k in keep and isinstance(v, (int, float, bool)):
                merged[k].append(v)
    payload = {
        "agent_config": {"agent_name": "delta_mem_write_then_drop", "write_window": args.write_window, "context_cap": cap},
        "dataset_config": dataset_config,
        "data": records,
        "metrics": dict(merged),
        "averaged_metrics": summarize_metrics(merged),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print("averaged_metrics:", json.dumps(payload["averaged_metrics"]), flush=True)


if __name__ == "__main__":
    main()
