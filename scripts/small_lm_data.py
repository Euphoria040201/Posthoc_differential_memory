#!/usr/bin/env python
"""Build a deterministic tokenized manifest from the cached C4 corpus.

Chain B needs every arm to see *the same tokens in the same order*, so the
manifest is built once, hashed, and then memory-mapped read-only by each run.
No synthetic text: the source is the standard allenai/c4 `en` split already in
the HF cache, and the tokenizer is the project's locked Qwen3-4B tokenizer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="/work/mingze/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--out-dir", default="/work/mingze/Posthoc_differential_memory/out_smalllm_20260814")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--train-tokens", type=int, default=2_200_000_000)
    ap.add_argument("--val-tokens", type=int, default=4_194_304)
    ap.add_argument("--num-proc", type=int, default=16)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id
    print(f"[data] tokenizer={args.tokenizer} vocab={tok.vocab_size} eos={eos}", flush=True)

    # Read the ALREADY-CACHED c4 shards directly.  load_dataset("allenai/c4","en")
    # would start a 1024-file download of the full corpus; the cache holds a
    # 356,318-document subset which is what every arm will share.
    from datasets import Dataset, concatenate_datasets
    cache = Path("/home/mingze/.cache/huggingface/datasets/allenai___c4")
    # Several cache dirs hold BYTE-IDENTICAL copies of the same shards; taking the
    # glob as-is doubles the document count and would silently make the tail of a
    # large token budget an exact repeat of the head.  Deduplicate by basename.
    def _dedup(paths):
        seen, keep = set(), []
        for f in sorted(paths):
            if f.name not in seen:
                seen.add(f.name)
                keep.append(f)
        return keep

    train_files = _dedup(cache.glob("*/*/*/c4-train-*.arrow"))
    val_files = _dedup(cache.glob("*/*/*/c4-validation.arrow"))
    assert train_files, "no cached c4 train shards"
    print(f"[data] cached train shards: {[f.name for f in train_files]}", flush=True)
    ds = {
        "train": concatenate_datasets([Dataset.from_file(str(f)) for f in train_files]),
        "validation": concatenate_datasets([Dataset.from_file(str(f)) for f in val_files]),
    }
    print(f"[data] docs={ {k: len(v) for k, v in ds.items()} }", flush=True)

    def build(split: str, target: int, path: Path) -> dict:
        if path.exists():
            arr = np.load(path, mmap_mode="r")
            print(f"[data] {path.name} exists ({len(arr):,} tokens), reusing", flush=True)
            return {"tokens": int(len(arr)), "path": str(path)}
        buf = np.empty(target + 65536, dtype=np.uint32)
        n = 0
        # `split` order is the dataset's own on-disk order -- deterministic, no shuffle.
        for i, ex in enumerate(ds[split]):
            ids = tok(ex["text"], add_special_tokens=False)["input_ids"]
            ids.append(eos)
            k = min(len(ids), len(buf) - n)
            buf[n:n + k] = ids[:k]
            n += k
            if n >= target:
                break
            if i % 20000 == 0:
                print(f"[data] {split}: doc {i:,} -> {n:,}/{target:,} tokens", flush=True)
        arr = buf[:target if n >= target else n]
        np.save(path, arr)
        print(f"[data] wrote {path} ({len(arr):,} tokens)", flush=True)
        return {"tokens": int(len(arr)), "path": str(path)}

    man = {"tokenizer": args.tokenizer, "seq_len": args.seq_len,
           "corpus": "allenai/c4:en (HF cache, on-disk order, no shuffle)"}
    man["val"] = build("validation", args.val_tokens, out / "c4_val.npy")
    man["train"] = build("train", args.train_tokens, out / "c4_train.npy")

    for k in ("train", "val"):
        p = Path(man[k]["path"])
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        man[k]["sha256"] = h.hexdigest()
        print(f"[data] {k} sha256 = {man[k]['sha256']}", flush=True)

    (out / "data_manifest.json").write_text(json.dumps(man, indent=1))
    print(f"[data] manifest -> {out/'data_manifest.json'}")


if __name__ == "__main__":
    main()
