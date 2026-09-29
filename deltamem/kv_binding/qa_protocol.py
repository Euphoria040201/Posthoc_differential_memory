"""Versioned short-answer protocol and immutable, tokenized experiment manifests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROTOCOL_VERSION = "qasper_short_evidence_eos_v2"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def answer_ids(tok, answer, max_ans_tok):
    """The budget includes EOS. Reject overlong targets instead of changing gold."""
    if tok.eos_token_id is None:
        raise ValueError("short-answer training requires tokenizer.eos_token_id")
    if max_ans_tok < 2:
        raise ValueError("max_ans_tok must reserve at least one answer token and EOS")
    ids = tok(" " + answer, add_special_tokens=False)["input_ids"]
    if not ids or len(ids) + 1 > max_ans_tok:
        return None
    return list(ids) + [int(tok.eos_token_id)]


def evidence_is_visible(evidence, context):
    """All annotated text evidence must survive both paragraph and total truncation."""
    norm = lambda s: " ".join(s.split())
    context = norm(context)
    return bool(evidence) and all(norm(e) in context for e in evidence)


def example_id(data, split, paper_id, question_id, question, answer):
    return digest([data, split, paper_id, question_id, question, answer])[:24]


def examples_fingerprint(examples):
    return digest([{k: ex[k] for k in ("sample_id", "ids", "labels", "answer")}
                   for ex in examples])


def tokenizer_fingerprint(tok):
    return digest({"vocab": tok.get_vocab(), "eos": tok.eos_token_id,
                   "pad": tok.pad_token_id,
                   "backend": tok.backend_tokenizer.to_str()})


def write_manifest(path, tok, splits, metadata):
    payload = {"protocol_version": PROTOCOL_VERSION,
               "tokenizer_sha256": tokenizer_fingerprint(tok),
               "eos_token_id": tok.eos_token_id, "metadata": metadata,
               "splits": splits,
               "fingerprints": {k: examples_fingerprint(v) for k, v in splits.items()}}
    payload["sha256"] = digest(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != payload:
            raise FileExistsError(f"refusing to replace a different data manifest: {path}")
        return payload
    path.write_text(json.dumps(payload, ensure_ascii=False))
    return payload


def read_manifest(path, tok=None):
    payload = json.loads(Path(path).read_text())
    expected = payload.pop("sha256")
    if digest(payload) != expected:
        raise ValueError("data manifest checksum mismatch")
    payload["sha256"] = expected
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("data manifest uses a different QA protocol")
    if tok is not None and payload["tokenizer_sha256"] != tokenizer_fingerprint(tok):
        raise ValueError("data manifest tokenizer differs from the model tokenizer")
    for split, examples in payload["splits"].items():
        if not examples:
            raise ValueError(f"empty manifest split: {split}")
        if examples_fingerprint(examples) != payload["fingerprints"][split]:
            raise ValueError(f"example fingerprint mismatch: {split}")
        for ex in examples:
            if ex["labels"][-1] != payload["eos_token_id"]:
                raise ValueError(f"missing EOS target in {split}")
    return payload


def add_manifest_arguments(parser):
    parser.add_argument("--data-manifest", default="",
                        help="immutable tokenized v2 data; overrides data construction flags")
    parser.add_argument("--eval-split", choices=["validation", "test"], default="validation")


def data_from_manifest(args, tok):
    payload = read_manifest(args.data_manifest, tok)
    return payload["splits"]["train"], payload["splits"][args.eval_split], {
        "protocol_version": PROTOCOL_VERSION, "manifest_sha256": payload["sha256"],
        "eval_split": args.eval_split, "fingerprints": payload["fingerprints"],
        "metadata": payload["metadata"],
    }
