"""Prefix-Memory Steering on Qasper (full-document context).

Backbone = FROZEN Qwen3-4B with FULL attention (unchanged main path).  An attached
SWA+prefix memory module (per selected layer) reads the context and steers the
backbone (δ-mem-style Q/K/V/O corrections OR a residual add).  Only prefix + memory
+ steer projections train.  This is the δ-mem interface with the delta-rule state
replaced by trainable prefix memory tokens read via sliding-window attention.

The ``memory_value_source=main_v`` ablation is deliberately prefix-free: a parallel
causal SWA keeps independent trainable Q/K, reuses grouped frozen Qwen V heads, and
injects only delta-O.  The frozen backbone Q/K/V and attention path remain unchanged.

Evaluates three conditions on the SAME model:
  base         : steer OFF   -> frozen backbone (baseline to beat)
  method       : steer ON    -> our prefix-memory steer
  method_nomem : steer ON, memory READS zeroed (reads := 0 => delta(0) = 0, so this
                 degenerates to base for bias-free deltas). It is a NO-MEMORY control,
                 NOT "zero the learned prefix but keep mem-from-hidden".

Eval protocol follows --train-mode:
  ctx     : generate from [context + question]
  noctx   : method writes memory from the context, context is REMOVED, generation is
            question-only; base/method_nomem generate question-only too (same footing).
  ctxmask : method writes memory from the FULL context, then ALL conditions generate over
            the SAME deterministically-masked context (--eval-context-mask-ratio); results
            are reported as ctxmask_method / ctxmask_base / ctxmask_nomem.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deltamem.core.prefix_steer import (
    PrefixSteerConfig, attach_prefix_steer, freeze_backbone_keep_steer,
    set_steer_segments, set_steer_zero_prefix, set_steer_enabled, iter_steer_modules,
)
from deltamem.core.global_prefix import SEG_CTX, SEG_QRY, SEG_ANS
from deltamem.kv_binding.qasper_episodes import build_fulldoc_episodes

SYS = "Answer the question using the context. Give a short answer."


def _norm(s):
    s = s.lower(); s = re.sub(r"\b(a|an|the)\b", " ", s); s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def f1_em(pred, gold):
    p, g = _norm(pred).split(), _norm(gold).split()
    em = float(_norm(pred) == _norm(gold))
    if not p or not g:
        return float(p == g), em
    ns = sum((Counter(p) & Counter(g)).values())
    if ns == 0:
        return 0.0, em
    prec, rec = ns / len(p), ns / len(g)
    return 2 * prec * rec / (prec + rec), em


def _ctx_with_spans(chunks, tok, max_ctx_tok):
    """Tokenize "Context:\\n" + "\\n".join(chunks) in ONE tokenizer call on the joined string --
    c_ids is byte-identical to the old code (per-chunk tokenization would move BPE merges at
    the joins) -- and derive each chunk's token span in c_ids from the fast tokenizer's char
    offsets. Boundary rule (stable, documented): chunk j starts at the FIRST token whose char
    span covers the chunk's first character, so a BPE token straddling a chunk boundary is
    assigned to the LATER chunk; the "\\n" joiner token (when not merged) stays with the
    EARLIER chunk. Tokens before chunk 0 are the "Context:\\n" header and belong to no chunk.
    Spans are clipped by the max_ctx_tok truncation; fully-truncated chunks are not recorded."""
    import bisect
    doc = "Context:\n" + "\n".join(chunks)
    enc = tok(doc, add_special_tokens=False, return_offsets_mapping=True)
    c_ids = enc["input_ids"][:max_ctx_tok]
    ends = [e for _, e in enc["offset_mapping"]]
    starts_char, pos = [], len("Context:\n")
    for ch in chunks:
        starts_char.append(pos)
        pos += len(ch) + 1                                  # +1: the "\n" joiner
    bounds = [bisect.bisect_right(ends, s) for s in starts_char] + [len(ends)]
    spans = []
    for j in range(len(chunks)):
        s, e = min(bounds[j], len(c_ids)), min(bounds[j + 1], len(c_ids))
        if e > s:
            spans.append((s, e))
    return c_ids, spans


def mask_context(ex, ratio, mode, rng, mask_block_tokens):
    """READ-stage context for ctxmask training: a SUBSEQUENCE of ex['ctx_ids'] with ~ratio of
    the context tokens DELETED (never replaced -- Qwen has no trained [MASK] token; deletion
    also avoids leaving fact fragments the way independent token dropout would).
      ratio 0.0 -> the full context, ratio 1.0 -> [] (phase 2 == noctx question-only).
      chunk mode : delete whole original chunks until >= ratio of the chunk TOKENS are gone
                   (falls back to span mode when ctx_chunk_spans is unavailable).
      span mode  : delete randomly-chosen contiguous mask_block_tokens-sized blocks.
    Kept tokens preserve their original order. The "Context:\\n" header (no-chunk tokens)
    survives unless every chunk is deleted, in which case the result is fully empty."""
    cid = list(ex["ctx_ids"])
    if ratio <= 0.0:
        return cid
    if ratio >= 1.0:
        return []
    spans = ex.get("ctx_chunk_spans") or []
    if mode == "chunk" and spans:
        total = sum(e - s for s, e in spans)
        target = ratio * total
        order = list(range(len(spans)))
        rng.shuffle(order)
        dropped, keep = 0, [True] * len(cid)
        for j in order:
            if dropped >= target:
                break
            s, e = spans[j]
            for t in range(s, e):
                keep[t] = False
            dropped += e - s
        if dropped >= total:                                # every chunk gone -> drop header too
            return []
        return [c for c, k in zip(cid, keep) if k]
    B = max(1, int(mask_block_tokens))
    nb = (len(cid) + B - 1) // B
    # ratio > 0 here (early-returned above) -> drop AT LEAST one block: with few blocks
    # round() would otherwise silently no-op (nb=1, ratio=0.5 -> round(0.5)=0). Short docs
    # can only be masked at coarse discrete steps; floor at 1 rather than at 0.
    k = min(nb, max(1, int(round(ratio * nb))))
    if k >= nb:
        return []
    drop = set(rng.sample(range(nb), k))
    return [c for b in range(nb) if b not in drop for c in cid[b * B:(b + 1) * B]]


def stable_mask_seed(ctx_ids, base_seed, epoch):
    """Per-(document, epoch) deterministic mask seed. Keying by document content makes the
    example->mask mapping independent of model-init --seed and of train order (both feed
    random.shuffle); all queries of the same paper share the mask within an epoch, and eval
    uses epoch=-1 so its masks never collide with any training epoch and are identical
    across evaluate() calls and val-set compositions."""
    import hashlib
    raw = f"{base_seed}:{epoch}:" + ",".join(map(str, ctx_ids))
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "big")


def _episode_to_examples(chunks, queries, tok, max_ctx_tok, max_ans_tok):
    c_ids, c_spans = _ctx_with_spans(chunks, tok, max_ctx_tok)
    out = []
    for q in queries:
        q_ids = tok(f"\n\n{SYS}\nQuestion: {q['question']}\nAnswer:", add_special_tokens=False)["input_ids"]
        a_ids = tok(" " + q["answer"], add_special_tokens=False)["input_ids"][:max_ans_tok]
        ids = c_ids + q_ids + a_ids
        seg = [SEG_CTX] * len(c_ids) + [SEG_QRY] * len(q_ids) + [SEG_ANS] * len(a_ids)
        out.append({"ids": ids, "seg": seg, "labels": [-100] * (len(c_ids) + len(q_ids)) + list(a_ids),
                    "prompt_ids": c_ids + q_ids, "prompt_seg": [SEG_CTX] * len(c_ids) + [SEG_QRY] * len(q_ids),
                    "ctx_ids": c_ids, "ctx_chunk_spans": c_spans, "qa_ids": q_ids + a_ids,
                    "qa_seg": [SEG_QRY] * len(q_ids) + [SEG_ANS] * len(a_ids),
                    "qa_labels": [-100] * len(q_ids) + list(a_ids), "answer": q["answer"]})
    return out


def build_examples(split, max_papers, tok, max_chunk_tok, max_ctx_tok, max_ans_tok, data="qasper",
                   max_yesno_frac=1.0, yesno_seed=0, train_target_n=0, mix_temporal_n=0):
    if data == "memalpha":
        from deltamem.kv_binding.memalpha_episodes import build_memalpha_episodes
        # split passes through: "train"/"validation" -> 90/10 of official train; never test.
        # max_chunk_tok=None: keep MemAlpha's multi-thousand-token chunks WHOLE (the Qasper 256
        # cap would gut them). Budget by TOTAL context via skip-over, and short-answer filter.
        eps, ma_stats = build_memalpha_episodes(
            split=split, max_papers=max_papers, tokenizer=tok, max_chunk_tok=None,
            max_ctx_tok=max_ctx_tok, max_ans_tok=max_ans_tok, return_stats=True)
        print(f"[memalpha:{split}] kept={ma_stats['kept']} ctx_over_dropped={ma_stats['drop_ctx_over']} "
              f"({ma_stats['ctx_over_by_source']}) no_qa_dropped={ma_stats['drop_no_qa']}", flush=True)
    else:
        eps = build_fulldoc_episodes(split, max_papers=max_papers, tokenizer=tok, max_chunk_tok=max_chunk_tok)
    out = []
    for ep in eps:
        c_ids, c_spans = _ctx_with_spans(ep["chunks"], tok, max_ctx_tok)
        for q in ep["queries"]:
            q_ids = tok(f"\n\n{SYS}\nQuestion: {q['question']}\nAnswer:", add_special_tokens=False)["input_ids"]
            a_ids = tok(" " + q["answer"], add_special_tokens=False)["input_ids"][:max_ans_tok]
            ids = c_ids + q_ids + a_ids
            seg = [SEG_CTX] * len(c_ids) + [SEG_QRY] * len(q_ids) + [SEG_ANS] * len(a_ids)
            labels = [-100] * (len(c_ids) + len(q_ids)) + list(a_ids)
            out.append({"ids": ids, "seg": seg, "labels": labels,
                        "prompt_ids": c_ids + q_ids,
                        "prompt_seg": [SEG_CTX] * len(c_ids) + [SEG_QRY] * len(q_ids),
                        # pieces for write->drop->read (no-context) training: memory is
                        # written from the context, then the answer is produced from the
                        # question ALONE, so the loss can only fall by storing the doc in memory.
                        "ctx_ids": c_ids,
                        # per-chunk token spans inside ctx_ids (for ctxmask chunk deletion)
                        "ctx_chunk_spans": c_spans,
                        # gold-evidence-only WRITE source (--write-gold-only): header + the
                        # query's gold chunks. Truncation only drops TAIL chunks, so span
                        # index j still addresses chunk j for j < len(c_spans).
                        "gold_ctx_ids": (
                            (c_ids[:c_spans[0][0]] +
                             [t for j in sorted(set(q.get("gold") or [])) if j < len(c_spans)
                              for t in c_ids[c_spans[j][0]:c_spans[j][1]]])
                            if (q.get("gold") and c_spans) else None),
                        "qa_ids": q_ids + a_ids,
                        "qa_seg": [SEG_QRY] * len(q_ids) + [SEG_ANS] * len(a_ids),
                        "qa_labels": [-100] * len(q_ids) + list(a_ids),
                        "answer": q["answer"]})
    # OPTIONAL: control the yes/no fraction. Qasper has ~12.9% yes/no and "Yes" is the single most
    # frequent answer (8.8%) -> the model learns a strong yes/no prior and DEGENERATES to yes/no on
    # hard multi-hop HotpotQA bridge questions (degeneration rate tracks F1). Two modes:
    #   train_target_n>0 : COMPOSE exactly N examples with yes/no ~= max_yesno_frac, backfilling with
    #                      non-yes/no from a larger paper pool -> keeps TRAIN COUNT CONSTANT (so the
    #                      effect is the RATIO, not fewer examples). Needs max_papers big enough.
    #   else             : just DROP yes/no down to <= max_yesno_frac (train count shrinks).
    if train_target_n or max_yesno_frac < 1.0:
        import random as _r
        yn = [e for e in out if e["answer"].strip().lower() in ("yes", "no")]
        other = [e for e in out if e["answer"].strip().lower() not in ("yes", "no")]
        _r.Random(yesno_seed).shuffle(yn); _r.Random(yesno_seed + 2).shuffle(other)
        n_yn0, n_other0 = len(yn), len(other)
        if train_target_n:
            N = train_target_n
            frac = max_yesno_frac if max_yesno_frac < 1.0 else n_yn0 / max(1, len(out))
            n_yn = min(len(yn), int(round(frac * N)))
            n_other = min(len(other), N - n_yn)
            out = other[:n_other] + yn[:n_yn]
        else:
            keep = int(max_yesno_frac * len(other) / max(1e-9, 1.0 - max_yesno_frac))
            out = other + yn[:keep]
        _r.Random(yesno_seed + 1).shuffle(out)
        n_yn_now = sum(1 for e in out if e["answer"].strip().lower() in ("yes", "no"))
        print(f"[compose] total={len(out)} (target_n={train_target_n or '-'}) yes/no={n_yn_now} "
              f"({100*n_yn_now/max(1,len(out)):.1f}%) ; pool yn={n_yn0} other={n_other0}", flush=True)
    if mix_temporal_n > 0 and split == "train":
        # ADD synthetic LoCoMo-style temporal QA (clean-date extraction) to teach cat2 answering.
        from deltamem.kv_binding.synth_temporal import build_synthetic_temporal_episodes
        teps = build_synthetic_temporal_episodes(n_episodes=max(1, mix_temporal_n // 4),
                                                 queries_per_ep=4, seed=yesno_seed)
        tex = []
        for ep in teps:
            tex += _episode_to_examples(ep["chunks"], ep["queries"], tok, max_ctx_tok, max_ans_tok)
        tex = tex[:mix_temporal_n]
        out = out + tex
        import random as _r
        _r.Random(yesno_seed + 3).shuffle(out)
        print(f"[mix-temporal] added {len(tex)} synthetic temporal examples -> total {len(out)}", flush=True)
    return out


def collate(batch, pad_id, device):
    maxlen = max(len(b["ids"]) for b in batch)
    B = len(batch)
    ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    seg = torch.full((B, maxlen), SEG_ANS, dtype=torch.long)
    lab = torch.full((B, maxlen), -100, dtype=torch.long)
    val = torch.zeros((B, maxlen), dtype=torch.bool)
    for i, b in enumerate(batch):
        L = len(b["ids"])
        ids[i, :L] = torch.tensor(b["ids"]); seg[i, :L] = torch.tensor(b["seg"])
        lab[i, :L] = torch.tensor(b["labels"]); val[i, :L] = True
    return ids.to(device), seg.to(device), val.to(device), lab.to(device)


def get_dtype(n):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[n]


@torch.no_grad()
def generate(model, tok, ex, device, max_new_tokens, eos, noctx=False, prompt=None):
    if prompt is not None:
        # explicit (ids, seg) prompt -- used by the ctxmask eval to hand every condition
        # the SAME masked context
        ids, seg = list(prompt[0]), list(prompt[1])
    elif noctx:
        # question-only prompt: the context is REMOVED from the input entirely
        nc = len(ex["ctx_ids"])
        ids = list(ex["prompt_ids"][nc:]); seg = [SEG_QRY] * len(ids)
    else:
        ids = list(ex["prompt_ids"]); seg = list(ex["prompt_seg"])
    gen = []
    for _ in range(max_new_tokens):
        iid = torch.tensor([ids], device=device)
        sgt = torch.tensor([seg], device=device)
        val = torch.ones_like(iid, dtype=torch.bool)
        set_steer_segments(model, sgt, val)
        logits = model(input_ids=iid, use_cache=False).logits
        nxt = int(logits[0, -1].argmax())
        gen.append(nxt); ids.append(nxt); seg.append(SEG_ANS)
        if eos is not None and nxt == eos:
            break
    return tok.decode(gen, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-prefix-tokens", type=int, default=64)
    ap.add_argument("--prefix-lr", type=float, default=1e-2,
                    help="separate lr for prefix embeddings. Defaults to 1e-2 (20x --lr): "
                         "prompt-tuning practice is 1e-2..3e-2, and at the shared 5e-4 the "
                         "prefix barely leaves its init. Pass --prefix-lr 5e-4 to tie them.")
    ap.add_argument("--prefix-wd", type=float, default=0.0,
                    help="weight decay for prefix embeddings (AdamW default 0.01 shrinks them)")
    ap.add_argument("--write-ctx-only", default="false", choices=["true","false"],
                    help="WRITE: prefix attends context ONLY (strict document memory)")
    ap.add_argument("--memory-mode", default="residual", choices=["residual","dynamic"],
                    help="dynamic = READ memory is ONLY write_proj(ctx), no static shortcut")
    ap.add_argument("--pool-reads", default="true", choices=["true","false"],
                    help="DEFAULT method: reads_t = R_t + amax_p(alpha_{t,p} * v_p) -- "
                         "TOKEN-SPECIFIC max over the weighted prefix contributions, added "
                         "to the standard SWA read (requires --read-prefix-only false)")
    ap.add_argument("--pool-gate", default="false", choices=["true","false"],
                    help="learnable token-conditioned gate on M_t: reads_t = R_t + "
                         "sigmoid(W_g h_t) (.) M_t; trained jointly (needs --pool-reads true)")
    ap.add_argument("--pool-gate-bias", type=float, default=0.0,
                    help="gate bias init: 0=>0.5 start; 4=>~0.98 (start OPEN = plain-pool, "
                         "learn to close). Use positive to avoid the 0.5-init handicap.")
    ap.add_argument("--pool-gate-input", default="rm", choices=["rm","hidden"],
                    help="gate input: rm=[R_t;M_t] (sees token AND memory); hidden=h_t only")
    ap.add_argument("--pool-gate-max", type=float, default=2.0,
                    help="gate range: 1=[0,1] can only attenuate M (<=plain pool); "
                         "2=[0,2] starts at plain pool (bias0) and can push M up or down.")
    ap.add_argument("--gate-lr", type=float, default=None,
                    help="separate lr for the mgate (like --prefix-lr). NOTE AdamW already "
                         "normalizes gradient SCALE, so 'gate grad is 5000x smaller' does NOT "
                         "justify a 5000x lr; 1e-2 is likely too big. Try 1e-4 / 5e-4.")
    ap.add_argument("--init-ckpt", default="",
                    help="load these trained steer weights before training (strict=False, so a "
                         "newly-added mgate stays at its g=1 init). Use to START from a trained "
                         "plain-pool ckpt and only fine-tune the gate.")
    ap.add_argument("--gate-only", default="false", choices=["true","false"],
                    help="freeze EVERYTHING except mgate -- the clean test of 'can a gate "
                         "improve a fixed plain-pool solution' (no joint-training confound, "
                         "no shared-weight drift, gate grad never enters others' clip norm).")
    ap.add_argument("--train-mode", default="ctx", choices=["ctx","noctx","ctxmask"],
                    help="ctx=standard (context present); noctx=write->drop->read "
                         "(memory written from context, answer from question alone); "
                         "ctxmask=write from the FULL context, then answer over a randomly "
                         "MASKED context (backbone+SWA see only the masked ctx; the full "
                         "doc reaches the answer only through the written prefix memory).")
    ap.add_argument("--context-mask-mode", default="chunk", choices=["chunk", "span"],
                    help="ctxmask deletion unit: chunk=whole original chunks (needs the saved "
                         "chunk boundaries); span=contiguous --mask-block-tokens blocks. Never "
                         "independent token dropout (it leaves fact fragments in place).")
    ap.add_argument("--context-mask-ratios", default="0.0,0.5,1.0",
                    help="comma list of mask ratios sampled per training example "
                         "(0=keep full ctx, 0.5=delete ~half, 1=delete all)")
    ap.add_argument("--context-mask-weights", default="0.2,0.5,0.3",
                    help="comma list, same length as --context-mask-ratios; normalized into "
                         "the sampling distribution over the ratios")
    ap.add_argument("--context-mask-seed", type=int, default=-1,
                    help="seed of the DEDICATED mask RNG (decoupled from model init and data "
                         "composition). -1 = use --seed.")
    ap.add_argument("--mask-block-tokens", type=int, default=256,
                    help="block size for --context-mask-mode span (and the fallback when an "
                         "example carries no reliable chunk boundaries)")
    ap.add_argument("--eval-context-mask-ratio", type=float, default=0.5,
                    help="fixed mask ratio used by the ctxmask evaluation (all conditions "
                         "share the same deterministic masked context per example)")
    ap.add_argument("--swap-contrast-lambda", type=float, default=0.0,
                    help="ctxmask only. >0 adds a swap-contrastive term: per sample also "
                         "compute the answer CE with a WRONG document's written memory and "
                         "penalize relu(margin + CE_correct - CE_swap). This is the ONLY "
                         "pressure that forces the written memory to be DOC-SPECIFIC -- "
                         "plain CE lets the writer collapse to a doc-agnostic bias (measured: "
                         "swap ~= correct on every plain writer). ~2x step cost.")
    ap.add_argument("--swap-margin", type=float, default=0.5,
                    help="hinge margin (nats) for --swap-contrast-lambda: wrong-memory CE "
                         "should exceed correct-memory CE by at least this much")
    ap.add_argument("--wo-contrast-lambda", type=float, default=0.0,
                    help="ctxmask only. >0 adds relu(margin + CE_correct - CE_window_only): "
                         "the same read WITHOUT the prefix must be WORSE, i.e. the prefix "
                         "content must actively lower the answer CE. The swap hinge alone is "
                         "satisfiable by making WRONG memory confusing (CE_swap up) without "
                         "the correct memory ever helping (measured: swap_gap +1.15 nats yet "
                         "generation F1 unchanged); this term closes that loophole.")
    ap.add_argument("--wo-margin", type=float, default=0.2,
                    help="hinge margin (nats) for --wo-contrast-lambda")
    ap.add_argument("--mix-ambig-k", type=int, default=0,
                    help="ctxmask only. Mix K synthetic AMBIGUOUS bindings into training: K "
                         "names x 2 equally-frequent city variants ('X lives in Paris/Tokyo'), "
                         "same question -> the weights cap at 50%% and only a memory READ can "
                         "reduce the loss. Ports the toy-ladder success condition (the ONLY "
                         "regime where doc-specific lookup provably gets learned) into real "
                         "training to keep the lookup circuit alive. 0 = off.")
    ap.add_argument("--mix-ambig-rep", type=int, default=8,
                    help="copies of each synthetic (name,variant) instance in the train list")
    ap.add_argument("--write-gold-only", default="false", choices=["true", "false"],
                    help="ctxmask only, qasper only. WRITE from the query's GOLD evidence "
                         "chunks (header + 1-3 paragraphs, ~300-700 tok) instead of the full "
                         "4500-tok document. Fact-level curriculum: the toy ladder proves the "
                         "memory binds ~16 ambiguous facts cleanly; full papers are orders of "
                         "magnitude past that, which is where doc-specificity collapses.")
    ap.add_argument("--wo-detach", type=str2bool, default=False,
                    help="detach CE_window_only in the wo hinge: the gradient can then ONLY "
                         "satisfy the hinge by pulling CE_correct down (below the no-prefix "
                         "reference), never by sabotaging the window path (measured: without "
                         "detach the model blows CE_wo up to +10 nats -- garbage-on-masked -- "
                         "instead of making the prefix useful)")
    ap.add_argument("--backbone-window", type=int, default=0,
                    help="if >0, BOUND the frozen backbone to this sliding window (all layers). "
                         "This is the regime where a memory can actually add information: each "
                         "h_i is local, but the memory WRITE aggregates over the whole sequence.")
    ap.add_argument("--read-prefix-only", default="false", choices=["true","false"],
                    help="LEGACY bottleneck: normal tokens read the PREFIX ONLY. Conflicts "
                         "with --pool-gate true (the default method reads prefix + SWA window).")
    ap.add_argument("--prefix-write", default="true", choices=["true","false"],
                    help="two-stage memory: prefix READS the context (WRITE) then normal "
                         "tokens read it. false = old static learned KV prior (not a memory).")
    ap.add_argument("--prefix-init-data", type=int, default=0,
                    help="DATA-DRIVEN prefix init: if >0, initialize each layer's P write-probes "
                         "from REAL post-input_layernorm hidden states captured over this many "
                         "training contexts (instead of random). The probes start as actual "
                         "document token vectors, so they query the context meaningfully from step 0.")
    ap.add_argument("--prefix-init-dist", default="normal", choices=["normal","uniform","orthogonal"],
                    help="init distribution of the prefix WRITE-probes: normal (default), "
                         "uniform (same variance), orthogonal (maximally-diverse probes, scaled "
                         "to match). In dynamic+write mode the prefix is only the write query, so "
                         "probe DIVERSITY (init) sets how much distinct document content it stores.")
    ap.add_argument("--prefix-init-std", type=float, default=0.7,
                    help="std of prefix init; must match the POST-input_layernorm hidden RMS "
                         "the memory actually sees (measured 0.704 @ Qwen3-4B layer 20). "
                         "0.02 = embedding scale, correct only for layer-0 prompt tuning.")
    ap.add_argument("--sliding-window-size", type=int, default=256)
    ap.add_argument("--steer-mode", default="deltamem", choices=["deltamem", "residual"])
    ap.add_argument("--mem-num-heads", type=int, default=8)
    ap.add_argument("--mem-head-dim", type=int, default=64)
    ap.add_argument("--steer-gain", type=float, default=1.0)
    ap.add_argument("--share-qkv", type=str2bool, default=False)
    ap.add_argument(
        "--memory-value-source",
        default="trainable",
        choices=["trainable", "main_v"],
        help="memory-attention V source: trainable=legacy mem_v; main_v=reuse the "
             "current frozen Qwen v_proj, grouping its GQA KV heads to --mem-num-heads "
             "(requires --mem-head-dim equal to the backbone head dim)",
    )
    ap.add_argument("--delta-heads", default="qkvo")
    ap.add_argument("--delta-rank", type=int, default=0)
    ap.add_argument("--read-proj-dim", type=int, default=0)
    ap.add_argument(
        "--output-fusion",
        default="fixed",
        choices=["fixed", "rms_match", "cosine"],
        help="delta-O fusion: fixed=legacy out+gain*delta; rms_match=detached per-token "
             "RMS matching with energy normalization; cosine=(1+detached cosine)/2 gate",
    )
    ap.add_argument("--output-fusion-eps", type=float, default=1e-6)
    ap.add_argument("--output-fusion-scale-max", type=float, default=10.0)
    ap.add_argument("--save-steps", default="", help="comma steps to also save intermediate ckpts for best-ckpt selection")
    ap.add_argument("--steer-layers", default="", help="comma list; empty=all")
    ap.add_argument("--prefix-layers", default="", help="comma list of steer-layers that GET the prefix (P>0); others get P=0 pure-SWA. empty=all get prefix")
    ap.add_argument("--normal-attends-prefix", type=str2bool, default=True)
    ap.add_argument("--prefix-sees-query", type=str2bool, default=True)
    ap.add_argument("--max-chunk-tok", type=int, default=256)
    ap.add_argument("--max-ctx-tok", type=int, default=4500)
    ap.add_argument("--max-ans-tok", type=int, default=24)
    ap.add_argument("--train-papers", type=int, default=300)
    ap.add_argument("--val-papers", type=int, default=30)
    ap.add_argument("--data-compose-seed", type=int, default=-1,
                    help="seed for the yes/no cap / target-n COMPOSITION only, DECOUPLED from "
                         "the model-init --seed. Set to a FIXED value to hold the training set "
                         "identical across model seeds (isolates data recipe from init noise). "
                         "-1 = use --seed (old behavior).")
    ap.add_argument("--train-target-n", type=int, default=0,
                    help="COMPOSE exactly this many training examples (backfill non-yes/no from a "
                         "larger --train-papers pool) so the yes/no-ratio experiment holds TRAIN "
                         "COUNT CONSTANT (isolates ratio from 'fewer examples'). 0 = off.")
    ap.add_argument("--mix-temporal-n", type=int, default=0,
                    help="mix this many synthetic LoCoMo-style temporal QA (clean-date extraction) "
                         "into the training set to teach cat2 temporal answering (0 = off).")
    ap.add_argument("--max-yesno-frac", type=float, default=1.0,
                    help="cap the fraction of yes/no training answers (Qasper is 12.9%%, 'Yes' is "
                         "the single most frequent answer -> model degenerates to yes/no on hard "
                         "HotpotQA bridge questions). e.g. 0.03 subsamples yes/no down to <=3%%.")
    ap.add_argument("--data", default="qasper", choices=["qasper", "memalpha"],
                    help="training corpus: qasper (default) or memalpha (long-context memory "
                         "QA/ICL; hotpotqa+squad+booksum sources dropped to avoid eval leak)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--log-gradnorm", action="store_true",
                    help="print per-component grad norms every 20 updates (prefix / mem_qkv / "
                         "write_proj / delta / mgate) -- shows which memory parts get learning signal")
    ap.add_argument("--device-map", default="",
                    help="split the backbone across GPUs to keep full ctx on small cards, e.g. "
                         "'auto' or 'balanced' (needs accelerate). ONE run uses all visible GPUs "
                         "(pipeline). Empty = single device (--device).")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="recompute the frozen backbone in backward instead of storing the full "
                         "activation graph -> fits ~16GB cards (peak ~10-12GB @ 4500 ctx vs ~38GB). "
                         "~30%% slower. Backbone is frozen so this also enables input_require_grads.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    if args.memory_value_source == "main_v":
        # This named branch is the exact experiment requested here, not a loose collection
        # of partially-compatible flags.  Fail before allocating the 4B model if any prefix,
        # writer, q/k/v-delta, or two-forward-memory path is accidentally left enabled.
        assert args.num_prefix_tokens == 0, "main_v Qasper branch requires --num-prefix-tokens 0"
        assert args.prefix_write == "false", "main_v Qasper branch requires --prefix-write false"
        assert args.pool_reads == "false", "main_v Qasper branch requires --pool-reads false"
        assert args.pool_gate == "false", "main_v Qasper branch requires --pool-gate false"
        assert args.read_prefix_only == "false", (
            "main_v Qasper branch requires --read-prefix-only false"
        )
        assert args.prefix_init_data == 0 and not args.prefix_layers, (
            "main_v Qasper branch has no prefix to initialize or route"
        )
        assert not args.share_qkv, (
            "main_v keeps independent trainable side-SWA Q/K; use --share-qkv false"
        )
        assert args.steer_mode == "deltamem" and set(args.delta_heads) == {"o"}, (
            "main_v Qasper branch only permits delta-O: use --steer-mode deltamem "
            "--delta-heads o"
        )
        assert args.train_mode == "ctx", (
            "prefix-free causal SWA has no persistent WRITE state; train it inline with "
            "--train-mode ctx"
        )
    if args.train_mode in ("noctx", "ctxmask"):
        # these branches read train[i] only (per-example two-forward write->read) --
        # batch_size > 1 would SILENTLY drop (batch_size-1)/batch_size of the data
        assert args.batch_size == 1, f"--train-mode {args.train_mode} requires --batch-size 1"
        # BOTH two-forward modes keep _frozen_prefix alive across forwards, while
        # non-reentrant checkpointing RE-RUNS layer forwards at backward time with the module
        # state as it is THEN (frozen memory cleared / write branch flipped on
        # _frozen_prefix) -> the recompute diverges from the original forward and the
        # gradients are silently wrong.
        assert not args.grad_checkpointing, (
            f"--train-mode {args.train_mode} is incompatible with --grad-checkpointing: "
            "frozen-memory state mutates between the write and read forwards, so the "
            "checkpoint recompute re-enters the write branch under different state")
    mask_ratios = mask_weights = None; mask_seed = 0
    if args.train_mode == "ctxmask":
        # ctxmask is only meaningful when the memory CONTENT is a pure function of the
        # written document. In residual mode M_D = P_static + Write(D): a broken write path
        # can hide behind a static prefix that still receives gradient, so the first-update
        # grad check would pass while the document is never actually stored.
        assert args.prefix_write == "true", "ctxmask requires --prefix-write true"
        assert args.memory_mode == "dynamic", "ctxmask requires --memory-mode dynamic"
        assert args.write_ctx_only == "true", "ctxmask requires --write-ctx-only true"
        mask_ratios = [float(x) for x in args.context_mask_ratios.split(",") if x.strip()]
        mask_weights = [float(x) for x in args.context_mask_weights.split(",") if x.strip()]
        assert mask_ratios and len(mask_ratios) == len(mask_weights), (
            f"--context-mask-weights must match --context-mask-ratios in length "
            f"({len(mask_weights)} vs {len(mask_ratios)})")
        assert all(0.0 <= r <= 1.0 for r in mask_ratios), "mask ratios must be in [0,1]"
        assert all(w >= 0.0 for w in mask_weights), "mask weights must be non-negative"
        _tw = sum(mask_weights)
        assert _tw > 0, "--context-mask-weights must sum > 0"
        mask_weights = [w / _tw for w in mask_weights]
        # masks are keyed per (document, epoch) via stable_mask_seed -- NOT a stateful RNG
        # stream. A stream is only stream-independent: train order comes from --seed's
        # random.shuffle, so under a different model seed the same stream would re-pair
        # masks with different documents. Keyed masks give the same doc the same mask at
        # the same epoch across model seeds.
        mask_seed = args.context_mask_seed if args.context_mask_seed >= 0 else args.seed
        print(f"[{args.tag}] ctxmask: mode={args.context_mask_mode} ratios={mask_ratios} "
              f"weights={[round(w,3) for w in mask_weights]} mask_seed={mask_seed} (per-doc keyed) "
              f"block={args.mask_block_tokens} eval_ratio={args.eval_context_mask_ratio}", flush=True)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if args.backbone_window > 0:
        from transformers import AutoConfig
        _bc = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
        _tc = _bc.get_text_config() if hasattr(_bc, "get_text_config") else _bc
        _tc.sliding_window = args.backbone_window
        _tc.layer_types = ["sliding_attention"] * _tc.num_hidden_layers
        print(f"[{args.tag}] BOUNDED backbone: window={args.backbone_window}, all {_tc.num_hidden_layers} layers sliding")
    _kw = {"config": _bc} if args.backbone_window > 0 else {}
    if args.device_map:
        # PIPELINE across GPUs: split the frozen backbone's layers over the visible cards so ONE
        # run keeps the full 4500 ctx on e.g. 2x16GB. Each attached steer module already moves its
        # seg/valid/prefix to the local layer device in forward, so the custom attach is device-safe.
        base = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=get_dtype(args.dtype),
            attn_implementation=args.attn_impl, local_files_only=True,
            device_map=args.device_map, **_kw)
        # inputs must start on the input-embedding's device; HF/accelerate hooks move the rest.
        args.device = str(base.get_input_embeddings().weight.device)
        print(f"[{args.tag}] device_map='{args.device_map}' -> {base.hf_device_map if hasattr(base,'hf_device_map') else '?'}; inputs on {args.device}")
        # accelerate quietly offloads layers that don't fit -> not an error but a crawl (or a
        # backward-time failure). Training must be fully GPU-resident; fail fast instead.
        _bad = {k: v for k, v in getattr(base, "hf_device_map", {}).items()
                if str(v) in ("cpu", "disk")}
        assert not _bad, f"training model has CPU/disk offload: {_bad}"
    else:
        base = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=get_dtype(args.dtype),
            attn_implementation=args.attn_impl, local_files_only=True, **_kw).to(args.device)
    if args.backbone_window > 0:
        # ASSERT, don't trust the flag: the config used to be built and then thrown away
        # (config= was never passed), so the backbone stayed FULL attention while the log
        # cheerfully printed "BOUNDED backbone".
        _lt = base.config.get_text_config().layer_types if hasattr(base.config, "get_text_config") else base.config.layer_types
        _sw = base.config.get_text_config().sliding_window if hasattr(base.config, "get_text_config") else base.config.sliding_window
        assert set(_lt) == {"sliding_attention"} and _sw == args.backbone_window, (
            f"backbone_window={args.backbone_window} did NOT take effect: "
            f"sliding_window={_sw}, layer_types[0]={_lt[0]}")
        print(f"[{args.tag}] VERIFIED bounded backbone: sliding_window={_sw}, all layers {_lt[0]}")

    steer_layers = tuple(int(x) for x in args.steer_layers.split(",") if x.strip()) if args.steer_layers else ()
    cfg = PrefixSteerConfig(
        num_prefix_tokens=args.num_prefix_tokens, prefix_init_std=args.prefix_init_std,
        prefix_init_dist=args.prefix_init_dist,
        prefix_write=(args.prefix_write == "true"),
        read_prefix_only=(args.read_prefix_only == "true"), memory_mode=args.memory_mode,
        write_ctx_only=(args.write_ctx_only == "true"), pool_reads=(args.pool_reads == "true"),
        pool_gate=(args.pool_gate == "true"), pool_gate_bias=args.pool_gate_bias, pool_gate_max=args.pool_gate_max, pool_gate_input=args.pool_gate_input,
        sliding_window_size=args.sliding_window_size,
        mem_num_heads=args.mem_num_heads, mem_head_dim=args.mem_head_dim,
        steer_mode=args.steer_mode, normal_attends_prefix=args.normal_attends_prefix,
        prefix_sees_query=args.prefix_sees_query, steer_layers=steer_layers,
        steer_gain=args.steer_gain, share_qkv=args.share_qkv, delta_heads=args.delta_heads,
        delta_rank=args.delta_rank, read_proj_dim=args.read_proj_dim,
        memory_value_source=args.memory_value_source,
        output_fusion=args.output_fusion,
        output_fusion_eps=args.output_fusion_eps,
        output_fusion_scale_max=args.output_fusion_scale_max,
        prefix_layers=tuple(int(x) for x in args.prefix_layers.split(",") if x.strip()))
    replaced = attach_prefix_steer(base, cfg)
    freeze_backbone_keep_steer(base)
    if args.grad_checkpointing:
        # Fit small (e.g. 16GB) cards: recompute the frozen backbone's activations in the
        # backward instead of storing the whole 4500-token graph (the ~30GB hog). Weights stay
        # ~8GB; activation storage drops ~5-10x. The write path in ctx-mode is a pure function of
        # the inputs (no state cached in training mode), so recomputation is exact.
        base.config.use_cache = False
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # backbone is FROZEN -> its input embeddings don't require grad, so checkpointing would
        # store nothing / not activate. This hook makes the embedding output require grad so the
        # graph (and thus the attached steer params' gradients) flows through the recomputed layers.
        base.enable_input_require_grads()
        print(f"[{args.tag}] gradient checkpointing ON (use_reentrant=False, input_require_grads)")
    if args.init_ckpt:
        ick = torch.load(args.init_ckpt, map_location="cpu")
        miss, unexp = base.load_state_dict(ick["state"], strict=False)
        from deltamem.core.prefix_steer import is_steer_param_name
        miss_steer = [n for n in miss if is_steer_param_name(n)]
        print(f"[{args.tag}] init-ckpt {args.init_ckpt}: loaded {len(ick['state'])}; "
              f"steer params NOT in ckpt (stay at code init, e.g. mgate g=1): {len(miss_steer)}")
    if args.gate_only == "true":
        # freeze EVERYTHING except mgate -> the clean 'can a gate improve a fixed plain pool'
        for n, p in base.named_parameters():
            p.requires_grad_(".mgate." in n)
        assert any(p.requires_grad for _, p in base.named_parameters()), "gate-only froze everything"
    ntr = sum(p.numel() for p in base.parameters() if p.requires_grad)
    print(f"[{args.tag}] patched {len(replaced)} layers, trainable={ntr:,} (gate_only={args.gate_only})")

    print("loading data...")
    _compose_seed = args.data_compose_seed if args.data_compose_seed >= 0 else args.seed
    train = build_examples("train", args.train_papers, tok, args.max_chunk_tok, args.max_ctx_tok, args.max_ans_tok, data=args.data, max_yesno_frac=args.max_yesno_frac, yesno_seed=_compose_seed, train_target_n=args.train_target_n, mix_temporal_n=args.mix_temporal_n)
    val = build_examples("validation", args.val_papers, tok, args.max_chunk_tok, args.max_ctx_tok, args.max_ans_tok, data=args.data)
    print(f"[{args.tag}] train={len(train)} val={len(val)}")
    if args.train_mode == "ctxmask" and args.mix_ambig_k > 0:
        _ar = random.Random(1234)
        _F = ["Alice","Bob","Carol","David","Emma","Frank","Grace","Henry","Ivy","Jack","Kate",
              "Leo","Mia","Noah","Olga","Paul","Quinn","Rosa","Sam","Tina","Uma","Vera","Will",
              "Xena","Yuri","Zoe"]
        _L = ["Anders","Brooks","Chen","Diaz","Evans","Fischer","Garcia","Hayes","Ito","Jones",
              "Kim","Lopez","Meyer","Novak","Owens","Park","Qureshi","Rossi","Silva","Tanaka",
              "Ueda","Vogel","Weber","Xu"]
        _C = ["Paris","Tokyo","Cairo","Lima","Oslo","Delhi","Rome","Seoul","Quito","Hanoi",
              "Lagos","Berlin","Madrid","Athens","Dublin","Vienna","Prague","Havana","Nairobi",
              "Manila","Bogota","Warsaw","Lisbon","Helsinki","Brussels","Zagreb","Riga","Tunis"]
        _names = _ar.sample([f"{a} {b}" for a in _F for b in _L], args.mix_ambig_k)
        _added = []
        for _nm in _names:
            for _ct in _ar.sample(_C, 2):
                _cid = tok(f"Context:\n{_nm} lives in {_ct}. They work at a small bakery near "
                           f"the old bridge.", add_special_tokens=False)["input_ids"]
                _qid = tok(f"\n\n{SYS}\nQuestion: Where does {_nm} live?\nAnswer:",
                           add_special_tokens=False)["input_ids"]
                _aid = tok(" " + _ct, add_special_tokens=False)["input_ids"]
                _ex = {"ids": _cid + _qid + _aid,
                       "seg": [SEG_CTX] * len(_cid) + [SEG_QRY] * len(_qid) + [SEG_ANS] * len(_aid),
                       "labels": [-100] * (len(_cid) + len(_qid)) + list(_aid),
                       "prompt_ids": _cid + _qid,
                       "prompt_seg": [SEG_CTX] * len(_cid) + [SEG_QRY] * len(_qid),
                       "ctx_ids": _cid, "ctx_chunk_spans": [(2, len(_cid))], "gold_ctx_ids": None,
                       "qa_ids": _qid + _aid,
                       "qa_seg": [SEG_QRY] * len(_qid) + [SEG_ANS] * len(_aid),
                       "qa_labels": [-100] * len(_qid) + list(_aid), "answer": _ct}
                _added += [_ex] * args.mix_ambig_rep
        train = train + _added
        print(f"[{args.tag}] mix-ambig: +{len(_added)} synthetic instances "
              f"(K={args.mix_ambig_k} names x2 variants x{args.mix_ambig_rep}) -> train={len(train)}", flush=True)
    if args.train_mode == "ctxmask" and args.write_gold_only == "true":
        assert args.data == "qasper", "--write-gold-only needs qasper gold evidence"
        _gl = [len(e["gold_ctx_ids"]) for e in train if e.get("gold_ctx_ids")]
        print(f"[{args.tag}] gold-write: {len(_gl)}/{len(train)} examples carry gold spans "
              f"(mean {sum(_gl)/max(1,len(_gl)):.0f} tok; full-doc fallback otherwise)", flush=True)

    if args.prefix_init_data > 0:
        # DATA-DRIVEN init: capture each steered layer's INPUT hidden states (post-input_LN --
        # exactly what mem_q/mem_k see) over a few real contexts, then set each layer's P prefix
        # write-probes to sampled real hidden vectors so they start as meaningful document queries.
        from deltamem.core.prefix_steer import iter_steer_modules
        mods = [m for m in iter_steer_modules(base) if m.cfg.num_prefix_tokens > 0]
        caps = {id(m): [] for m in mods}
        hooks = []
        for m in mods:
            def pre(mod, a_, kw_, mm=m):
                hs = kw_.get("hidden_states", a_[0] if a_ else None)
                if hs is not None:
                    caps[id(mm)].append(hs.detach()[0].float().cpu())
            hooks.append(m.register_forward_pre_hook(pre, with_kwargs=True))
        base.eval(); set_steer_enabled(base, False)
        with torch.no_grad():
            for ex in train[:args.prefix_init_data]:
                cids = torch.tensor([ex["ctx_ids"]], device=args.device)
                set_steer_segments(base, torch.full_like(cids, SEG_CTX),
                                   torch.ones_like(cids, dtype=torch.bool))
                base(input_ids=cids, use_cache=False)
        for h in hooks: h.remove()
        set_steer_enabled(base, True)
        g = torch.Generator().manual_seed(args.seed)
        for m in mods:
            pool = torch.cat(caps[id(m)], dim=0)               # [total_ctx_tokens, hidden]
            P = m.cfg.num_prefix_tokens
            idx = torch.randint(0, pool.shape[0], (P,), generator=g)
            m.prefix.data.copy_(pool[idx].to(dtype=m.prefix.dtype, device=m.prefix.device))
        print(f"[{args.tag}] data-init: prefix set from real ctx hiddens "
              f"({len(mods)} layers, pool ~{pool.shape[0]} tok/layer)", flush=True)
    pad_id = tok.pad_token_id; eos = tok.eos_token_id

    # prefix embeddings need their own (much higher) lr: their grad reaches them only
    # through the zero-init delta_o / 1e-3 delta_qkv path * steer_gain, so at a shared
    # lr they never leave their random init (measured: std 0.0200 -> 0.021 after 2500 steps).
    pref_p = [p for n, p in base.named_parameters()
              if p.requires_grad and p.numel() > 0 and n.endswith(".prefix")]
    # mgate gets its OWN (high) lr: its gradient is ~5000x smaller than delta_o (M_t is small
    # x steer_gain), so at the shared lr it never leaves its init (gate frozen at sigmoid(bias)).
    gate_p = [p for n, p in base.named_parameters() if p.requires_grad and ".mgate." in n]
    gate_ids = {id(p) for p in gate_p}
    rest_p = [p for n, p in base.named_parameters() if p.requires_grad
              and not n.endswith(".prefix") and id(p) not in gate_ids]
    plr = args.prefix_lr if args.prefix_lr is not None else args.lr
    glr = args.gate_lr if args.gate_lr is not None else args.lr
    groups = [{"params": rest_p, "lr": args.lr}]
    if pref_p:
        groups.append({"params": pref_p, "lr": plr, "weight_decay": args.prefix_wd})
    if gate_p:
        groups.append({"params": gate_p, "lr": glr})
    opt = torch.optim.AdamW(groups, lr=args.lr)
    print(f"[{args.tag}] optim: {len(rest_p)} tensors @ lr={args.lr}; "
          f"{len(pref_p)} prefix @ lr={plr}; {len(gate_p)} gate @ lr={glr}")

    def evaluate_ctxmask():
        """ctxmask eval: per example, ONE deterministic masked context is shared by ALL
        conditions (per-document keyed via stable_mask_seed(ctx_ids, mask_seed, epoch=-1),
        so every evaluate() call across training -- and any val-set composition -- sees
        identical masks):
          ctxmask_method      : memory WRITTEN from the full ctx, generation over masked ctx
          ctxmask_base        : steer OFF, no write, SAME masked ctx
          ctxmask_window_only : steer ON, memory written, but the prefix is REMOVED from the
                                READ (window SWA kept) -> method - window_only isolates the
                                written prefix's marginal value over pure SWA steering
          ctxmask_nomem       : steer ON, ALL memory reads zeroed, SAME masked ctx.
        NOTE ctxmask_nomem zeroes the WHOLE read (prefix pool AND window SWA) -- it is a
        no-memory control and CANNOT attribute prefix vs SWA; the prefix attribution is
        method - window_only, not method - nomem."""
        base.eval()
        from deltamem.core.prefix_steer import set_write_freeze, clear_frozen_memory, set_window_only
        conds = {"ctxmask_base": dict(steer=False, nomem=False, wo=False, write=False),
                 "ctxmask_method": dict(steer=True, nomem=False, wo=False, write=True),
                 "ctxmask_window_only": dict(steer=True, nomem=False, wo=True, write=True),
                 "ctxmask_nomem": dict(steer=True, nomem=True, wo=False, write=False)}
        acc = {n: ([], []) for n in conds}
        for vi, ex in enumerate(val):
            mrng = random.Random(stable_mask_seed(ex["ctx_ids"], mask_seed, -1))
            mctx = mask_context(ex, args.eval_context_mask_ratio, args.context_mask_mode,
                                mrng, args.mask_block_tokens)
            q_ids = ex["prompt_ids"][len(ex["ctx_ids"]):]
            pids = mctx + list(q_ids)
            pseg = [SEG_CTX] * len(mctx) + [SEG_QRY] * len(q_ids)
            for name, c in conds.items():
                set_steer_enabled(base, c["steer"]); set_steer_zero_prefix(base, c["nomem"])
                set_window_only(base, c["wo"])
                clear_frozen_memory(base)
                if c["write"]:
                    cids = torch.tensor([ex["ctx_ids"]], device=args.device)
                    set_write_freeze(base, True)
                    set_steer_segments(base, torch.full_like(cids, SEG_CTX),
                                       torch.ones_like(cids, dtype=torch.bool))
                    with torch.no_grad():
                        base(input_ids=cids, use_cache=False)
                    set_write_freeze(base, False)      # stop writing, KEEP the memory
                pred = generate(base, tok, ex, args.device, args.max_new_tokens, eos,
                                prompt=(pids, pseg))
                clear_frozen_memory(base)
                f, e = f1_em(pred, ex["answer"])
                acc[name][0].append(f); acc[name][1].append(e)
        set_steer_enabled(base, True); set_steer_zero_prefix(base, False); set_window_only(base, False)
        return {n: {"F1": round(sum(f) / len(f), 4), "EM": round(sum(e) / len(e), 4)}
                for n, (f, e) in acc.items()}

    def evaluate():
        if args.train_mode == "ctxmask":
            return evaluate_ctxmask()
        base.eval()
        from deltamem.core.prefix_steer import set_write_freeze, clear_frozen_memory
        noctx = args.train_mode == "noctx"
        res = {}
        conds = {"base": dict(steer=False, nomem=False),
                 "method": dict(steer=True, nomem=False),
                 "method_nomem": dict(steer=True, nomem=True)}
        for name, c in conds.items():
            set_steer_enabled(base, c["steer"]); set_steer_zero_prefix(base, c["nomem"])
            f1s, ems = [], []
            for ex in val:
                clear_frozen_memory(base)
                if noctx and c["steer"] and not c["nomem"]:
                    # method: WRITE the memory from the context, freeze it, then the
                    # context is REMOVED -- the same write->drop->read protocol as training.
                    cids = torch.tensor([ex["ctx_ids"]], device=args.device)
                    set_write_freeze(base, True)
                    set_steer_segments(base, torch.full_like(cids, SEG_CTX),
                                       torch.ones_like(cids, dtype=torch.bool))
                    with torch.no_grad():
                        base(input_ids=cids, use_cache=False)
                    set_write_freeze(base, False)      # stop writing, KEEP the memory
                # noctx: base/method_nomem generate question-only WITHOUT any write, so all
                # three conditions answer from the same context-free prompt.
                pred = generate(base, tok, ex, args.device, args.max_new_tokens, eos, noctx=noctx)
                clear_frozen_memory(base)
                f, e = f1_em(pred, ex["answer"]); f1s.append(f); ems.append(e)
            res[name] = {"F1": round(sum(f1s) / len(f1s), 4), "EM": round(sum(ems) / len(ems), 4)}
        set_steer_enabled(base, True); set_steer_zero_prefix(base, False)
        return res

    save_steps = set(int(x) for x in args.save_steps.split(",") if x.strip()) if args.save_steps else set()
    def save_ckpt(suffix):
        from deltamem.core.prefix_steer import is_steer_param_name
        # save ALL steer params (not just trainable) so gate-only ckpts carry the frozen
        # plain-pool weights too and are self-contained for eval
        st = {n: p.detach().cpu() for n, p in base.named_parameters() if is_steer_param_name(n)}
        torch.save({"state": st, "cfg": vars(cfg) if not isinstance(cfg, dict) else cfg,
                    "args": {k: getattr(args, k) for k in vars(args)}}, out / f"{args.tag}{suffix}_ckpt.pt")
        print(f"[{args.tag}] saved ckpt{suffix} ({len(st)} tensors)", flush=True)

    # NOTE: `step` counts OPTIMIZER UPDATES, not microbatches. It used to count microbatches,
    # so `--steps 2500 --grad-accum 16` performed only 2500/16 = 156 updates -- 16x fewer than
    # intended. That alone explains why the prefix never moved (156 updates @ lr 5e-4 is a
    # ~0.006 random walk; the measured drift was ~0.001).
    log = []; step = 0; micro = 0; t0 = time.time(); opt.zero_grad()
    done = False; last_mask = (None, 0, 0); last_swap_gap = None; last_wo_gap = None; epoch = 0
    swap_skips = 0
    while not done:
        random.shuffle(train)
        epoch += 1                                  # 1-based; eval reserves epoch=-1
        for i in range(0, len(train), args.batch_size):
            base.train()
            if args.train_mode == "noctx":
                # write->drop->read: FORWARD 1 writes memory from the context (frozen while
                # training so the graph is KEPT), FORWARD 2 answers from the question ALONE.
                from deltamem.core.prefix_steer import set_write_freeze, clear_frozen_memory, iter_steer_modules
                ex = train[i]
                dev = args.device
                cids = torch.tensor([ex["ctx_ids"]], device=dev)
                clear_frozen_memory(base); set_write_freeze(base, True)
                set_steer_segments(base, torch.full_like(cids, SEG_CTX), torch.ones_like(cids, dtype=torch.bool))
                base(input_ids=cids, use_cache=False)               # writes + freezes memory
                for m in iter_steer_modules(base): m._freeze_write = False
                qa = torch.tensor([ex["qa_ids"]], device=dev)
                qseg = torch.tensor([ex["qa_seg"]], device=dev)
                qlab = torch.tensor([ex["qa_labels"]], device=dev)
                set_steer_segments(base, qseg, torch.ones_like(qa, dtype=torch.bool))
                loss = base(input_ids=qa, labels=qlab, use_cache=False).loss  # context is GONE
                clear_frozen_memory(base)
            elif args.train_mode == "ctxmask":
                # write->MASK->read: FORWARD 1 writes the memory from the FULL context
                # (graph KEPT: _frozen_prefix is not detached in training mode, so the
                # answer CE backprops through the written memory into prefix / write_proj /
                # the write-stage mem_qkv). FORWARD 2 answers over a MASKED context: the
                # backbone attention and the SWA read see ONLY [masked ctx ; question ;
                # answer] hidden states -- the full document reaches forward 2 through the
                # written prefix memory and through nothing else (use_cache=False in both
                # forwards, no KV/mem cache; set_write_freeze(True) also dropped _mem_kv).
                from deltamem.core.prefix_steer import set_write_freeze, clear_frozen_memory
                ex = train[i]
                dev = args.device
                lrng = random.Random(stable_mask_seed(ex["ctx_ids"], mask_seed, epoch))
                ratio = lrng.choices(mask_ratios, weights=mask_weights, k=1)[0]
                mctx = mask_context(ex, ratio, args.context_mask_mode, lrng, args.mask_block_tokens)
                ids2 = mctx + ex["qa_ids"]
                seg2 = [SEG_CTX] * len(mctx) + ex["qa_seg"]
                lab2 = [-100] * len(mctx) + ex["qa_labels"]
                t2 = torch.tensor([ids2], device=dev)
                seg2_t = torch.tensor([seg2], device=dev)
                lab2_t = torch.tensor([lab2], device=dev)

                def write_then_ce(ctx_ids):
                    cids = torch.tensor([ctx_ids], device=dev)
                    clear_frozen_memory(base); set_write_freeze(base, True)
                    set_steer_segments(base, torch.full_like(cids, SEG_CTX),
                                       torch.ones_like(cids, dtype=torch.bool))
                    base(input_ids=cids, use_cache=False)   # WRITE: freezes graph-connected memory
                    set_write_freeze(base, False)           # stop writing, KEEP the memory
                    set_steer_segments(base, seg2_t, torch.ones_like(t2, dtype=torch.bool))
                    return base(input_ids=t2, labels=lab2_t, use_cache=False).loss

                def wsrc(e):
                    # gold-evidence write source when enabled and available; the READ side
                    # and the doc-identity checks keep using the full ctx_ids
                    if args.write_gold_only == "true" and e.get("gold_ctx_ids"):
                        return e["gold_ctx_ids"]
                    return e["ctx_ids"]
                ce_c = write_then_ce(wsrc(ex))
                loss = ce_c
                if args.wo_contrast_lambda > 0:
                    # prefix-usefulness hinge: with the SAME frozen memory, mask the prefix
                    # out of the read (window-only) and demand that this be WORSE by wo_margin.
                    # Unlike the swap hinge this cannot be satisfied by sabotaging wrong
                    # memories -- only by the prefix content lowering the correct-answer CE.
                    from deltamem.core.prefix_steer import set_window_only
                    set_window_only(base, True)
                    set_steer_segments(base, seg2_t, torch.ones_like(t2, dtype=torch.bool))
                    ce_wo = base(input_ids=t2, labels=lab2_t, use_cache=False).loss
                    set_window_only(base, False)
                    _ref = ce_wo.detach() if args.wo_detach else ce_wo
                    loss = loss + args.wo_contrast_lambda * torch.relu(
                        args.wo_margin + ce_c - _ref)
                    last_wo_gap = float(ce_wo.item() - ce_c.item())
                if args.swap_contrast_lambda > 0:
                    # DOC-SPECIFICITY pressure: the same masked-ctx+qa read, but the memory
                    # was written from a DIFFERENT document. Plain CE lets the writer collapse
                    # to a doc-agnostic bias (swap == correct, measured on every plain writer);
                    # the hinge relu(margin + CE_correct - CE_swap) is minimized only when the
                    # WRONG memory is genuinely worse -- i.e. the content depends on the doc.
                    # NOTE: hinges reference the PURE ce_c, never the penalized loss.
                    j = (i - 1) % len(train)
                    if train[j]["ctx_ids"] == ex["ctx_ids"]:
                        j = (i - 2) % len(train)            # neighbor query of the SAME paper
                    if train[j]["ctx_ids"] != ex["ctx_ids"]:
                        ce_swap = write_then_ce(wsrc(train[j]))
                        loss = loss + args.swap_contrast_lambda * torch.relu(
                            args.swap_margin + ce_c - ce_swap)
                        last_swap_gap = float(ce_swap.item() - ce_c.item())
                    else:
                        # both fallback neighbors were the SAME paper -> hinge skipped;
                        # a silent skip must be visible, not invisible
                        swap_skips += 1
                        if swap_skips in (1, 10, 100):
                            print(f"[{args.tag}] WARN swap hinge skipped (same-doc neighbors) "
                                  f"x{swap_skips}", flush=True)
                # only the PYTHON REFERENCE to the written memory is dropped here; the autograd
                # graph built through it in forward 2 holds its own references, so backward is intact.
                clear_frozen_memory(base)
                last_mask = (ratio, len(ex["ctx_ids"]), len(mctx))
            else:
                ids, seg, val_m, lab = collate(train[i:i + args.batch_size], pad_id, args.device)
                set_steer_segments(base, seg, val_m)
                loss = base(input_ids=ids, labels=lab, use_cache=False).loss
            (loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum != 0:
                continue
            if args.train_mode == "ctxmask" and step == 0:
                # write->read graph sanity at the FIRST update. prefix>0 alone is NOT enough:
                # it proves the prefix is USED, not that the document is WRITTEN into it (in
                # residual mode a static prefix would pass; dynamic mode is asserted above,
                # and here every stage of Writer(D_full) must carry signal):
                #   prefix     = the write QUERIES        (probe what to extract)
                #   mem_q      = write-query projection   (document-side attention)
                #   write_proj = read->hidden map         (the written content itself)
                def _bucket(mk):
                    ps = [p for n, p in base.named_parameters()
                          if p.requires_grad and (n.endswith(mk) if mk == ".prefix" else mk in n)]
                    none = all(p.grad is None for p in ps)
                    return none, sum(float(p.grad.pow(2).sum()) for p in ps if p.grad is not None) ** 0.5
                p_none, pn = _bucket(".prefix")
                _, wn = _bucket(".write_proj.")
                _, mqn = _bucket(".mem_q.")
                print(f"[{args.tag}] ctxmask FIRST-UPDATE grad check: prefix_grad_is_none={p_none} "
                      f"prefix_grad_norm={pn:.3e} write_proj_grad_norm={wn:.3e} "
                      f"mem_q_grad_norm={mqn:.3e}", flush=True)
                if p_none or pn == 0.0 or wn == 0.0 or mqn == 0.0:
                    raise RuntimeError(
                        f"ctxmask write path invalid: prefix_grad={pn:.3e} (none={p_none}), "
                        f"write_proj_grad={wn:.3e}, mem_q_grad={mqn:.3e} -- the document->"
                        "memory write is not receiving learning signal (detach/no_grad in the "
                        "write path, or frozen memory cleared before the read forward)")
            if args.log_gradnorm and step % 20 == 0:
                # per-group grad norms BEFORE clip: shows which parts of the memory receive
                # learning signal (a near-zero norm => that component is not being shaped).
                import math as _math
                buckets = {"prefix": ".prefix", "mem_q": ".mem_q.", "mem_k": ".mem_k.",
                           "mem_v": ".mem_v.", "write_proj": ".write_proj.",
                           "delta_o": ".delta_o.", "delta_q": ".delta_q.", "mgate": ".mgate."}
                gn = {}
                for bn, mk in buckets.items():
                    s2 = sum(float(p.grad.pow(2).sum()) for n, p in base.named_parameters()
                             if p.requires_grad and p.grad is not None and mk in n)
                    if s2 > 0: gn[bn] = _math.sqrt(s2)
                print(f"[{args.tag}] gradnorm@{step}: " + " ".join(f"{k}={v:.2e}" for k, v in gn.items()), flush=True)
            torch.nn.utils.clip_grad_norm_([p for p in base.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            step += 1                                   # one OPTIMIZER UPDATE
            if step % 20 == 0 or (args.train_mode == "ctxmask" and step <= 2):
                extra = ""
                if args.train_mode == "ctxmask" and last_mask[0] is not None:
                    _r, _fc, _mc = last_mask
                    extra = (f" mask_ratio={_r:.2f} full_ctx_tok={_fc} masked_ctx_tok={_mc} "
                             f"retained={_mc/max(1,_fc):.2f}")
                    if last_swap_gap is not None:
                        extra += f" swap_gap={last_swap_gap:+.3f}"
                    if last_wo_gap is not None:
                        extra += f" wo_gap={last_wo_gap:+.3f}"
                print(f"[{args.tag}] step {step}/{args.steps} (micro {micro}) loss {loss.item():.4f} "
                      f"{(time.time()-t0)/step:.2f}s/update{extra}", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                r = evaluate(); r["step"] = step; log.append(r)
                print(f"[{args.tag}] EVAL {step}: " + " ".join(f"{k}={v['F1']:.3f}" for k, v in r.items() if isinstance(v, dict)))
            if step in save_steps:
                save_ckpt(f"_step{step}")
            if step >= args.steps:
                done = True; break

    final = evaluate()
    result = {"tag": args.tag,
              "config": {k: getattr(args, k) for k in ["num_prefix_tokens", "sliding_window_size",
                         "steer_mode", "steer_layers", "mem_num_heads", "steps", "lr", "train_papers", "val_papers",
                          "prefix_write", "prefix_init_std", "prefix_lr", "backbone_window",
                          "memory_value_source", "output_fusion", "output_fusion_eps",
                          "output_fusion_scale_max", "delta_heads",
                          "read_prefix_only", "train_mode", "context_mask_mode", "context_mask_ratios",
                          "context_mask_weights", "context_mask_seed", "mask_block_tokens",
                          "eval_context_mask_ratio"]},
              "num_patched_layers": len(replaced), "trainable": ntr, "final": final, "history": log}
    with open(out / f"{args.tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    # save trainable steer weights (prefix + mem_* + delta_*/res_*) so we can reload
    from deltamem.core.prefix_steer import is_steer_param_name
    steer_state = {n: p.detach().cpu() for n, p in base.named_parameters() if is_steer_param_name(n)}
    torch.save({"state": steer_state, "cfg": vars(cfg) if not isinstance(cfg, dict) else cfg,
                "args": {k: getattr(args, k) for k in vars(args)}}, out / f"{args.tag}_ckpt.pt")
    print(f"[{args.tag}] saved ckpt ({len(steer_state)} tensors) -> {out/f'{args.tag}_ckpt.pt'}")
    print(f"[{args.tag}] DONE -> {out/f'{args.tag}.json'}\n{json.dumps(final, indent=2)}")


def str2bool(v):
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


if __name__ == "__main__":
    main()
