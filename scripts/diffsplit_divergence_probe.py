#!/usr/bin/env python
"""Measure how far the negative branch actually moved, on real Qasper val data.

This separates "the weights changed" from "the attention output changed".  A
trained `delta_q` with a large norm still proves nothing if the softmax washes
it out -- what matters is ||O+ - O-|| / ||O+|| and cos(O+, O-) at the point
where the differential is taken, i.e. immediately before `o_proj`.

Reported per layer and averaged, for a checkpoint and for the zero-init state.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deltamem.core.diff_split import (  # noqa: E402
    attach_diff_split, collect_diff_stats, iter_diff_modules, set_diff_stats,
    set_read_control,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/work/mingze/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-examples", type=int, default=24)
    ap.add_argument("--val-papers", type=int, default=75)
    ap.add_argument("--max-ctx-tok", type=int, default=4500)
    ap.add_argument("--max-chunk-tok", type=int, default=256)
    ap.add_argument("--max-ans-tok", type=int, default=24)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from qasper_prefix_steer import build_examples, collate, get_dtype

    blob = torch.load(args.ckpt, map_location="cpu")
    state, dcfg = blob["state"], blob["diff_config"]
    print(f"[probe] ckpt={args.ckpt}\n[probe] diff_config={dcfg}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=get_dtype("bfloat16"), attn_implementation="sdpa",
    ).to("cuda")
    model.config.use_cache = False
    attach_diff_split(model, tuple(dcfg["layers"]), read_dim=dcfg["read_dim"],
                      window=dcfg["window"], gamma=dcfg["gamma"],
                      dynamic_gate=dcfg["dynamic_gate"])

    val = build_examples("validation", args.val_papers, tok, args.max_chunk_tok,
                         args.max_ctx_tok, args.max_ans_tok, data="qasper")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    batches = [collate([e], pad, "cuda") for e in val[: args.n_examples]]
    print(f"[probe] {len(batches)} real val examples", flush=True)

    def sweep(label: str, *, zero=False, shuffle=False, shuffle_causal=False) -> dict:
        set_read_control(model, zero=zero, shuffle=shuffle,
                         shuffle_causal=shuffle_causal)
        set_diff_stats(model, True)
        per_layer: dict[int, list[dict]] = {}
        for ids, _seg, valid, _lab in batches:
            with torch.no_grad():
                model(input_ids=ids, attention_mask=valid.long())
            for li, m in zip(dcfg["layers"], iter_diff_modules(model)):
                if m.last_stats:
                    per_layer.setdefault(li, []).append(dict(m.last_stats))
        set_diff_stats(model, False)
        set_read_control(model, zero=False, shuffle=False, shuffle_causal=False)
        rows = {}
        for li, rs in per_layer.items():
            rows[li] = {k: sum(r[k] for r in rs) / len(rs) for k in rs[0]}
        keys = sorted(next(iter(rows.values())))
        mean = {k: sum(r[k] for r in rows.values()) / len(rows) for k in keys}
        print(f"\n=== {label} ===")
        print("layer  " + "  ".join(f"{k:>15s}" for k in keys))
        for li in sorted(rows):
            print(f"{li:5d}  " + "  ".join(f"{rows[li][k]:15.6f}" for k in keys))
        print("MEAN   " + "  ".join(f"{mean[k]:15.6f}" for k in keys))
        return {"per_layer": rows, "mean": mean}

    res = {"ckpt": args.ckpt, "diff_config": dcfg, "n_examples": len(batches)}
    res["zero_init"] = sweep("ZERO-INIT (delta_q = 0, must be exactly 0 divergence)")

    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected[:5]
    # fail-closed: every expected diff tensor must be present (audit issue 5)
    from deltamem.core.diff_split import is_diff_param_name as _isd
    _exp = {n for n, _ in model.named_parameters() if _isd(n)}
    _absent = sorted(_exp - set(state))
    if _absent:
        raise RuntimeError(f"probe: {len(_absent)}/{len(_exp)} diff tensors absent "
                           f"from ckpt, would stay zero-init: {_absent[:4]}")
    loaded = [k for k in state]
    print(f"\n[probe] loaded {len(loaded)} trained tensors", flush=True)

    res["trained"] = sweep("TRAINED")
    res["trained_zero_window"] = sweep("TRAINED + zeroed local window", zero=True)
    # audit issue 2 (2026-08-17): the old `shuffle=True` sweep permuted the key
    # axis AFTER causal masking and could move past probability mass onto FUTURE
    # keys, so it could not support a claim about the reader's use of the window.
    # It is replaced by a permutation restricted to each query's causal window.
    res["trained_shuffled_window_causal"] = sweep(
        "TRAINED + causally-permuted local window", shuffle_causal=True)

    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"\n[probe] wrote {args.out}")


if __name__ == "__main__":
    main()
