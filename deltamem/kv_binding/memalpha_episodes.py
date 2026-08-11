"""MemAlpha full-document episodes, in the SAME shape as qasper_episodes.build_fulldoc_episodes.

Each MemAlpha instance -> one episode:
  chunks   : list[str]  the instance's dialogue/document chunks (memory content), KEPT WHOLE
  queries  : list[{question, answer}]  a capped, shuffled sample of its short-answer QA pairs

The HF dataset `YuWangX/Memalpha` stores `chunks` and `questions_and_answers` as JSON STRINGS
(not lists) -- they must be json.loads'd.  data_source is a MIX of tasks; we DROP `hotpotqa`
and `squad` because they are our zero-shot EVAL benchmarks (training on them would be eval-set
leakage), and `booksum` by default because its "answer" is a long summary rather than a short
QA target.  Everything else (perltqa / lme_train / icl_* / pubmed-rct) is kept.

Anti-truncation contract (see the session critique):
  * chunks are NOT per-chunk truncated by default -- a MemAlpha chunk is a multi-thousand-token
    dialogue, so the Qasper default max_chunk_tok=256 would gut it.  Pass max_chunk_tok=None.
  * total context is bounded by max_ctx_tok via SKIP-OVER, not truncation: an instance whose
    full "Context:\n"+chunks exceeds the budget is DROPPED whole, so no QA is ever asked about
    evidence that was silently cut off.  Report what was dropped.
  * QA pairs are filtered to answers of <= max_ans_tok tokens, so the training target is never a
    truncated prefix of a long answer (train/eval target mismatch).

Splits: MemAlpha ships only `train` (567) and `test` (458) -- there is NO official validation.
We carve an instance-level 90/10 split out of `train` (fixed split seed, independent of the
episode seed) so hyperparameter selection never touches the official `test`.
"""
from __future__ import annotations

import json
import os
import random

# sources that would leak into our zero-shot eval, or aren't short-answer QA
_LEAK = {"hotpotqa", "squad"}
_NON_QA = {"booksum"}
DEFAULT_EXCLUDE = _LEAK | _NON_QA

_REPO = "YuWangX/Memalpha"     # official casing
_SPLIT_SEED = 1234             # fixes the train/val partition, independent of the episode seed
_CTX_PREFIX = "Context:\n"     # must match build_examples' doc = "Context:\n" + "\n".join(chunks)
_CHUNK_JOIN = "\n"


def build_memalpha_episodes(split="train", max_papers=None, tokenizer=None, max_chunk_tok=None,
                            max_chunks=60, seed=0, exclude_sources=None, include_sources=None,
                            max_queries_per_doc=8, max_ctx_tok=None, max_ans_tok=None,
                            return_stats=False):
    """Return [{chunks, queries:[{question,answer}], data_source}] mirroring build_fulldoc_episodes.

    split: "train" / "validation" (90/10 of the official train) / "test" (official test).
    max_ctx_tok: SKIP (never truncate) any instance whose full context exceeds this.
    max_ans_tok: drop QA whose answer tokenizes to more than this (short-answer only).
    """
    import datasets

    exclude = set(DEFAULT_EXCLUDE if exclude_sources is None else exclude_sources)
    keep = set(include_sources) if include_sources else None
    cache = os.path.expanduser("~/.cache/huggingface/datasets")
    raw = datasets.load_dataset(_REPO, cache_dir=cache)

    # instance-level 90/10 split of the official train (fixed seed); test -> official test
    train_ds = raw["train"]
    idx = list(range(len(train_ds)))
    random.Random(_SPLIT_SEED).shuffle(idx)
    n_val = max(1, len(idx) // 10)
    if split == "test":
        ds, sel = raw["test"], list(range(len(raw["test"])))
    elif split == "validation":
        ds, sel = train_ds, idx[:n_val]
    else:
        ds, sel = train_ds, idx[n_val:]

    rng = random.Random(seed)
    sel = list(sel)
    rng.shuffle(sel)

    stats = {"seen": 0, "kept": 0, "drop_source": 0, "drop_ctx_over": 0, "drop_no_qa": 0,
             "drop_parse": 0, "ctx_over_by_source": {}}
    eps = []
    for i in sel:
        if max_papers and len(eps) >= max_papers:
            break
        r = ds[i]
        src = r["data_source"]
        stats["seen"] += 1
        if src in exclude or (keep is not None and src not in keep):
            stats["drop_source"] += 1
            continue
        try:
            chunks = json.loads(r["chunks"])
            qas = json.loads(r["questions_and_answers"])
        except (json.JSONDecodeError, TypeError):
            stats["drop_parse"] += 1
            continue
        chunks = [str(c) for c in chunks][:max_chunks]
        if tokenizer is not None and max_chunk_tok:  # OFF by default for MemAlpha
            chunks = [tokenizer.decode(tokenizer(c, add_special_tokens=False)["input_ids"][:max_chunk_tok])
                      for c in chunks]
        # total-context budget: SKIP the whole instance if it does not fit (no mid-doc truncation)
        if tokenizer is not None and max_ctx_tok:
            full = _CTX_PREFIX + _CHUNK_JOIN.join(chunks)
            if len(tokenizer(full, add_special_tokens=False)["input_ids"]) > max_ctx_tok:
                stats["drop_ctx_over"] += 1
                stats["ctx_over_by_source"][src] = stats["ctx_over_by_source"].get(src, 0) + 1
                continue
        # short-answer QA only (avoid truncated-target train/eval mismatch)
        cand = []
        for q in qas:
            question = str(q.get("question", "")).strip()
            answer = str(q.get("answer", "")).strip()
            if not question or not answer:
                continue
            if tokenizer is not None and max_ans_tok:
                if len(tokenizer(" " + answer, add_special_tokens=False)["input_ids"]) > max_ans_tok:
                    continue
            cand.append({"question": question, "answer": answer})
        if not cand:
            stats["drop_no_qa"] += 1
            continue
        rng.shuffle(cand)
        eps.append({"chunks": chunks, "queries": cand[:max_queries_per_doc], "data_source": src})
        stats["kept"] += 1
    if return_stats:
        return eps, stats
    return eps


if __name__ == "__main__":
    from transformers import AutoTokenizer
    import collections
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", local_files_only=True)
    for sp in ("train", "validation"):
        e, st = build_memalpha_episodes(split=sp, max_papers=None, tokenizer=tok,
                                        max_chunk_tok=None, max_ctx_tok=20000, max_ans_tok=24,
                                        return_stats=True)
        src = collections.Counter(x["data_source"] for x in e)
        nq = sum(len(x["queries"]) for x in e)
        print(f"[{sp}] episodes={len(e)} queries={nq} sources={dict(src)}")
        print(f"      stats={st}")
    print("ex query:", e[0]["queries"][0])


def build_temporal_episodes(tokenizer, max_ctx_tok=4500, max_ans_tok=24, max_per_doc=6, seed=0):
    """TEMPORAL episodes from MemAlpha personal-dialogue sources (perltqa/lme_train): keep only
    'when / how long / how often / what year/date' QA with a temporal answer. Teaches clean
    temporal answering (LoCoMo cat2 style) -- our model dumps raw timestamps ('date: 7:18pm on
    27 may') instead of clean dates. Chunks kept whole, instance SKIPPED (not truncated) if >budget."""
    import re
    wh = re.compile(r'\bwhen\b|how long|how often|how many (weeks|days|months|years|times)|what (year|date|day|month)|which (year|month)', re.I)
    tans = re.compile(r'\b(19|20)\d{2}\b|january|february|march|april|may|june|july|august|september|october|november|december|monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekend|\bweeks?\b|\bmonths?\b|\byears?\b|\bdays?\b|ago|times a', re.I)
    # per-chunk truncate so long dialogues fit the ctx budget; the dialogue TIMESTAMP is in each
    # chunk's header ("[Dialogue ... on 2024-01-01 ...]") so it survives truncation.
    _mct = max(1, max_ctx_tok // 12)
    eps = build_memalpha_episodes(split="train", max_papers=None, tokenizer=tokenizer,
                                  max_chunk_tok=_mct, max_ctx_tok=max_ctx_tok, max_ans_tok=max_ans_tok,
                                  include_sources=("perltqa", "lme_train"), max_queries_per_doc=100000)
    out = []
    for ep in eps:
        tq = [q for q in ep["queries"] if wh.search(q["question"]) and tans.search(q["answer"])]
        if tq:
            out.append({"chunks": ep["chunks"], "queries": tq[:max_per_doc]})
    return out
