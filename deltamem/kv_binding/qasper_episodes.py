"""Qasper episodes for o_proj KV-binding training (gold-evidence-first).

One episode = one paper:
  chunks   : list[str]   memory slots = gold evidence paragraphs (union over the paper's usable
                         questions), optionally + distractor paragraphs from the paper.
  queries  : list[{question, answer, pos:[chunk_idx,...]}]  each question's gold-evidence slots.

Usable question = answerable AND has non-empty gold evidence AND a resolvable short answer.
"""
from __future__ import annotations
import random
from datasets import load_dataset as _hf_load_dataset


def load_dataset(path, split=None, **kw):
    """datasets>=3 refuses qasper's loader script; fall back to the hub's parquet
    auto-conversion branch (same rows/schema/order, so episodes are unchanged)."""
    try:
        return _hf_load_dataset(path, split=split, **kw)
    except RuntimeError as e:
        if "script" not in str(e).lower():
            raise
        return _hf_load_dataset(path, split=split, revision="refs/convert/parquet", **kw)


def resolve_answer(a):
    if a["yes_no"] is not None:
        return "Yes" if a["yes_no"] else "No"
    if a["free_form_answer"]:
        return a["free_form_answer"].strip()
    if a["extractive_spans"]:
        return ", ".join(a["extractive_spans"]).strip()
    if a["unanswerable"]:
        return "No information available"
    return None


def _clean_ev(ev):
    return [e.strip() for e in ev if e and e.strip() and not e.startswith("FLOAT SELECTED")]


def paper_paragraphs(paper):
    out = []
    ft = paper["full_text"]                      # {"section_name": [...], "paragraphs": [[...], ...]}
    for sec_paras in ft["paragraphs"]:
        for para in sec_paras:
            if para and para.strip():
                out.append(para.strip())
    return out


def build_fulldoc_episodes(split="train", max_papers=None, tokenizer=None, max_chunk_tok=256,
                           max_chunks=60, seed=0):
    """Full-document episodes: chunks = ALL paper paragraphs; each query keeps gold-chunk indices
    (for retrieval supervision / diagnostics only, NOT used to restrict the memory)."""
    ds = load_dataset("allenai/qasper", split=split)
    eps = []
    for paper in ds:
        if max_papers and len(eps) >= max_papers:
            break
        paras = paper_paragraphs(paper)[:max_chunks]
        if len(paras) < 3:
            continue
        chunks = paras
        if tokenizer is not None:
            chunks = [tokenizer.decode(tokenizer(p).input_ids[:max_chunk_tok])
                      if len(tokenizer(p).input_ids) > max_chunk_tok else p for p in paras]
        qas = paper["qas"]; queries = []
        for k in range(len(qas["question"])):
            a = qas["answers"][k]["answer"][0]; ans = resolve_answer(a); ev = _clean_ev(a.get("evidence", []))
            if not ans or a["unanswerable"] or not ev:
                continue
            gold = [i for i, c in enumerate(paras) if any(e[:80] in c or c[:80] in e for e in ev)]
            if gold:
                queries.append({"question": qas["question"][k].strip(), "answer": ans, "gold": sorted(set(gold))})
        if queries:
            eps.append({"paper_id": paper["id"], "chunks": chunks, "queries": queries})
    return eps


def build_fulldoc_episodes(split="train", max_papers=None, tokenizer=None, max_chunk_tok=256,
                           max_chunks=60, seed=0):
    """Full-document episodes: chunks = ALL paper paragraphs; each query keeps gold-chunk indices
    (for retrieval supervision / diagnostics only, NOT used to restrict the memory)."""
    ds = load_dataset("allenai/qasper", split=split)
    eps = []
    for paper in ds:
        if max_papers and len(eps) >= max_papers:
            break
        paras = paper_paragraphs(paper)[:max_chunks]
        if len(paras) < 3:
            continue
        chunks = paras
        if tokenizer is not None:
            chunks = [tokenizer.decode(tokenizer(p).input_ids[:max_chunk_tok])
                      if len(tokenizer(p).input_ids) > max_chunk_tok else p for p in paras]
        qas = paper["qas"]; queries = []
        for k in range(len(qas["question"])):
            a = qas["answers"][k]["answer"][0]; ans = resolve_answer(a); ev = _clean_ev(a.get("evidence", []))
            if not ans or a["unanswerable"] or not ev:
                continue
            gold = [i for i, c in enumerate(paras) if any(e[:80] in c or c[:80] in e for e in ev)]
            if gold:
                queries.append({"question": qas["question"][k].strip(), "answer": ans, "gold": sorted(set(gold))})
        if queries:
            eps.append({"paper_id": paper["id"], "chunks": chunks, "queries": queries})
    return eps


def build_episodes(split="train", max_papers=None, max_chunk_tok=256, tokenizer=None,
                   n_distractors=0, min_queries=1, seed=0):
    ds = load_dataset("allenai/qasper", split=split)
    rng = random.Random(seed)
    episodes = []
    for pi, paper in enumerate(ds):
        if max_papers and len(episodes) >= max_papers:
            break
        qas = paper["qas"]
        slots, slot_key = [], {}          # dedup evidence text -> slot idx
        queries = []
        for k in range(len(qas["question"])):
            a = qas["answers"][k]["answer"][0]
            ans = resolve_answer(a)
            ev = _clean_ev(a.get("evidence", []))
            if not ans or not ev or a["unanswerable"]:
                continue
            pos = []
            for e in ev:
                if tokenizer is not None and len(tokenizer(e).input_ids) > max_chunk_tok:
                    e = tokenizer.decode(tokenizer(e).input_ids[:max_chunk_tok])
                if e not in slot_key:
                    slot_key[e] = len(slots); slots.append(e)
                pos.append(slot_key[e])
            if pos:
                queries.append({"question": qas["question"][k].strip(), "answer": ans,
                                "pos": sorted(set(pos))})
        if len(queries) < min_queries:
            continue
        if n_distractors > 0:
            paras = [p for p in paper_paragraphs(paper) if p not in slot_key]
            rng.shuffle(paras)
            for p in paras[:n_distractors]:
                if tokenizer is not None and len(tokenizer(p).input_ids) > max_chunk_tok:
                    p = tokenizer.decode(tokenizer(p).input_ids[:max_chunk_tok])
                slots.append(p)
        episodes.append({"paper_id": paper["id"], "chunks": slots, "queries": queries})
    return episodes


def build_refusal_episodes(split="train", max_papers=None, tokenizer=None, max_chunk_tok=256,
                           max_chunks=60, max_per_paper=3):
    """REFUSAL episodes: Qasper UNANSWERABLE questions (answer='No information available') paired
    with the paper's paragraphs as context. Teaches the model to REFUSE when the answer is not in
    the context -- directly targets LoCoMo cat5 (adversarial), where our answer-happy model scores
    ~0.15 (base ~0.36) because it never saw refusal training (build_fulldoc_episodes drops these)."""
    ds = load_dataset("allenai/qasper", split=split)
    eps = []
    for paper in ds:
        if max_papers and len(eps) >= max_papers:
            break
        paras = paper_paragraphs(paper)[:max_chunks]
        if len(paras) < 3:
            continue
        chunks = paras
        if tokenizer is not None:
            chunks = [tokenizer.decode(tokenizer(p, add_special_tokens=False).input_ids[:max_chunk_tok])
                      for p in paras]
        qas = paper["qas"]; queries = []
        for k in range(len(qas["question"])):
            a = qas["answers"][k]["answer"][0]
            if a.get("unanswerable"):
                queries.append({"question": qas["question"][k].strip(),
                                "answer": "No information available"})
                if len(queries) >= max_per_paper:
                    break
        if queries:
            eps.append({"paper_id": paper["id"], "chunks": chunks, "queries": queries})
    return eps
