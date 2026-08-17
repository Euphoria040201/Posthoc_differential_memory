#!/usr/bin/env python
"""Build a real long-document corpus (PG19 books) for the continue-pretrain study.

Why PG19 and not the existing C4 stream: the chain-B C4 corpus is web text packed
at seq_len 1024, where almost no sequence carries a dependency longer than a few
hundred tokens.  A Differential Transformer's claimed benefit is cancellation of
attention noise over long context, so a corpus with no long-range structure is
the one setting where it provably cannot show an advantage.  PG19 books are
~70k tokens each, so every packed window can be drawn from ONE book.

Packing contract (this is what makes position-stratified NLL meaningful):
  * only books with >= `--min-book-tokens` tokens are used;
  * a sequence NEVER crosses a book boundary — window i of a book is
    tokens [i*L, (i+1)*L) of that book;
  * therefore position p inside a sequence has exactly p tokens of *same-book*
    context, and NLL(p) measures how well the model uses p tokens of real
    long-range context.
  * train and val use DISJOINT books (val comes from the official pg19 test split).

Outputs <out>/pg19_{train,val}.npy (uint32 token ids, [n_seq, seq_len]),
<out>/pg19_val_bookid.npy and a manifest with sha256 of every array.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def build_split(files, tok, seq_len, min_book_tokens, target_tokens, eos_id, label):
    """Tokenize books shard by shard, emit only whole in-book windows."""
    import pyarrow.parquet as pq

    seqs, book_ids, n_books, n_tok = [], [], 0, 0
    t0 = time.time()
    for fi, f in enumerate(files):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=16, columns=["text"]):
            texts = batch.column("text").to_pylist()
            enc = tok(texts, add_special_tokens=False)["input_ids"]
            for ids in enc:
                if len(ids) < min_book_tokens:
                    continue
                ids = ids + [eos_id]
                n_windows = len(ids) // seq_len
                if n_windows == 0:
                    continue
                arr = np.asarray(ids[: n_windows * seq_len], dtype=np.uint32)
                seqs.append(arr.reshape(n_windows, seq_len))
                book_ids.append(np.full(n_windows, n_books, dtype=np.int32))
                n_books += 1
                n_tok += n_windows * seq_len
            if n_tok >= target_tokens:
                break
        print(f"[{label}] shard {fi+1}/{len(files)} books={n_books} "
              f"tokens={n_tok:,} ({time.time()-t0:.0f}s)", flush=True)
        if n_tok >= target_tokens:
            break
    X = np.concatenate(seqs, 0)
    B = np.concatenate(book_ids, 0)
    # trim to the exact target so every arm sees an identical token budget
    n_keep = min(len(X), target_tokens // seq_len)
    return X[:n_keep], B[:n_keep], n_books


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/mingze/Posthoc_differential_memory/out_cpt_20260817")
    ap.add_argument("--tokenizer", default="/work/mingze/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--min-book-tokens", type=int, default=8192)
    ap.add_argument("--train-tokens", type=int, default=900_000_000)
    ap.add_argument("--val-tokens", type=int, default=8_388_608)
    ap.add_argument("--train-shards", type=int, default=8)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, list_repo_files
    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tok.eos_token_id

    files = list_repo_files("emozilla/pg19", repo_type="dataset")
    train_files = sorted(f for f in files if f.startswith("data/train-") and f.endswith(".parquet"))
    test_files = sorted(f for f in files if f.startswith("data/test-") and f.endswith(".parquet"))
    train_files = train_files[: args.train_shards]
    print(f"train shards: {len(train_files)}  val(test) shards: {len(test_files)}", flush=True)

    def fetch(rel):
        return hf_hub_download("emozilla/pg19", rel, repo_type="dataset")

    t0 = time.time()
    local_val = [fetch(f) for f in test_files]
    Xv, Bv, nbv = build_split(local_val, tok, args.seq_len, args.min_book_tokens,
                              args.val_tokens, eos_id, "val")
    np.save(out / "pg19_val.npy", Xv)
    np.save(out / "pg19_val_bookid.npy", Bv)
    print(f"VAL {Xv.shape} from {nbv} books", flush=True)

    local_train = []
    for f in train_files:
        local_train.append(fetch(f))
        print(f"downloaded {f} ({time.time()-t0:.0f}s)", flush=True)
    Xt, Bt, nbt = build_split(local_train, tok, args.seq_len, args.min_book_tokens,
                              args.train_tokens, eos_id, "train")
    np.save(out / "pg19_train.npy", Xt)
    print(f"TRAIN {Xt.shape} from {nbt} books", flush=True)

    man = {
        "corpus": "emozilla/pg19 (parquet mirror of deepmind/pg19), books",
        "packing": "intra-book windows only; a sequence never crosses a book boundary",
        "tokenizer": args.tokenizer,
        "seq_len": args.seq_len,
        "min_book_tokens": args.min_book_tokens,
        "eos_id": eos_id,
        "train_shards": train_files,
        "val_shards": test_files,
        "train": {"path": str(out / "pg19_train.npy"), "shape": list(Xt.shape),
                  "tokens": int(Xt.size), "books": nbt,
                  "sha256": sha256_file(out / "pg19_train.npy")},
        "val": {"path": str(out / "pg19_val.npy"), "shape": list(Xv.shape),
                "tokens": int(Xv.size), "books": nbv,
                "sha256": sha256_file(out / "pg19_val.npy")},
        "val_bookid": {"path": str(out / "pg19_val_bookid.npy"),
                       "sha256": sha256_file(out / "pg19_val_bookid.npy")},
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_seconds": round(time.time() - t0, 1),
    }
    (out / "pg19_manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: v for k, v in man.items() if k in ("train", "val")}, indent=1))


if __name__ == "__main__":
    main()
