#!/usr/bin/env python
"""Freeze one evidence-visible, EOS-supervised Qasper subset for every experiment arm."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--train-papers", type=int, default=0, help="0 = all official training papers")
    ap.add_argument("--val-papers", type=int, default=0)
    ap.add_argument("--test-papers", type=int, default=0)
    ap.add_argument("--train-target-n", type=int, default=935)
    ap.add_argument("--calibration-papers", type=int, default=32)
    ap.add_argument("--max-yesno-frac", type=float, default=0.03)
    ap.add_argument("--max-chunk-tok", type=int, default=0, help="0 preserves whole paragraphs")
    ap.add_argument("--max-ctx-tok", type=int, default=4500)
    ap.add_argument("--max-ans-tok", type=int, default=24, help="includes EOS")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.train_target_n < 1 or args.calibration_papers < 1:
        ap.error("train-target-n and calibration-papers must be positive")
    if not 0 <= args.max_yesno_frac <= 1:
        ap.error("max-yesno-frac must be in [0,1]")
    from transformers import AutoTokenizer
    from qasper_prefix_steer import build_examples
    from deltamem.kv_binding.qa_protocol import write_manifest

    tok = AutoTokenizer.from_pretrained(args.model_path)
    splits, audits = {}, {}
    for split, limit in (("train", args.train_papers), ("validation", args.val_papers),
                         ("test", args.test_papers)):
        audit = {}
        examples = build_examples(split, limit, tok, args.max_chunk_tok, args.max_ctx_tok,
                                  args.max_ans_tok, audit_stats=audit)
        # Every arm uses this same groupable population, including the ordinary QA controls.
        kept = [ex for ex in examples if len(ex["ctx_chunk_spans"]) >= 2]
        audit["dropped_ungroupable"] = len(examples) - len(kept)
        audit["groupable"] = len(kept)
        splits[split], audits[split] = kept, audit

    pool = splits["train"]
    papers = sorted({ex["paper_id"] for ex in pool})
    random.Random(args.seed).shuffle(papers)
    if len(papers) < args.calibration_papers + 1:
        raise ValueError("not enough eligible training papers for a disjoint calibration holdout")
    held_out = set(papers[:args.calibration_papers])
    splits["calibration"] = [ex for ex in pool if ex["paper_id"] in held_out]
    pool = [ex for ex in pool if ex["paper_id"] not in held_out]
    yesno = [ex for ex in pool if ex["answer"].strip().lower() in ("yes", "no")]
    other = [ex for ex in pool if ex["answer"].strip().lower() not in ("yes", "no")]
    random.Random(args.seed).shuffle(yesno)
    random.Random(args.seed + 2).shuffle(other)
    ny = min(len(yesno), int(args.train_target_n * args.max_yesno_frac))
    train = other[:args.train_target_n - ny] + yesno[:ny]
    if len(train) != args.train_target_n:
        raise ValueError(f"eligible train pool cannot supply {args.train_target_n} examples; "
                         f"available non-yes/no={len(other)}, yes/no={len(yesno)}. "
                         "Choose a shared smaller budget or a larger context budget before running any arm.")
    random.Random(args.seed + 1).shuffle(train)
    splits["train"] = train
    paper_sets = {split: {ex["paper_id"] for ex in examples}
                  for split, examples in splits.items()}
    for split, ids in paper_sets.items():
        for other_split, other_ids in paper_sets.items():
            if split != other_split and ids & other_ids:
                raise ValueError(f"paper leakage between {split} and {other_split}")
    metadata = {"settings": vars(args), "audits": audits,
                "counts": {k: len(v) for k, v in splits.items()},
                "calibration_paper_ids": sorted(held_out),
                "benchmark_label": "Qasper evidence-visible short-answer subset, first annotation; not official full Qasper"}
    payload = write_manifest(args.output, tok, splits, metadata)
    print(json.dumps({"manifest_sha256": payload["sha256"], **metadata}, indent=2))


if __name__ == "__main__":
    main()
