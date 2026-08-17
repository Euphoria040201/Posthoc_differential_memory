#!/usr/bin/env python
"""Hard gates for the token-local low-rank split on the REAL Qwen3-4B.

Tiny-model unit tests cannot catch head-count / kernel-selection problems that
only appear at production shapes (head_dim 128, 32q/8kv, sdpa picking a
different kernel).  This runs the same contract against the real checkpoint and
writes one JSON artifact with full provenance.

    python scripts/lowrank_realgate.py --layers 0,3,6,9,12,15,18,21,24,27,30,33 --rank 177
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deltamem.core.lowrank_split import (  # noqa: E402
    attach_lowrank_split, expected_split_param_count, freeze_backbone_keep_split,
    is_split_param_name, iter_split_modules, load_split_state_dict,
    set_split_enabled, split_param_names, split_state_dict,
    unfreeze_split_layer_attention,
)


def backbone_hash(model) -> str:
    """sha256 over every non-split tensor, to prove the backbone never moved.

    Attaching the wrapper renames `...self_attn.q_proj` to `...self_attn.base.q_proj`,
    so the `.base.` segment is stripped before hashing: otherwise the hash changes
    for a purely cosmetic reason and the gate reports a backbone drift that did
    not happen.
    """
    h = hashlib.sha256()
    rows = []
    for n, p in model.named_parameters():
        if is_split_param_name(n):
            continue
        rows.append((n.replace(".base.", "."), p))
    for n, p in sorted(rows, key=lambda r: r[0]):
        h.update(n.encode())
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def cache_summary(cache):
    ks, vs, nbytes = [], [], 0
    for layer in cache.layers:
        ks.append(list(layer.keys.shape))
        vs.append(list(layer.values.shape))
        nbytes += layer.keys.numel() * layer.keys.element_size()
        nbytes += layer.values.numel() * layer.values.element_size()
    return {"key_shapes": ks, "value_shapes": vs, "bytes": nbytes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/work/mingze/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--layers", default="0,3,6,9,12,15,18,21,24,27,30,33")
    ap.add_argument("--rank", type=int, default=177)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--out", default="/work/mingze/Posthoc_differential_memory/out_cpt_20260817/realgate_lowrank.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    layers = [int(x) for x in args.layers.split(",")]
    dev = "cuda"
    res = {"model": args.model, "layers": layers, "rank": args.rank,
           "gamma": args.gamma, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        res["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
            text=True).strip()
    except Exception:
        pass

    tok = AutoTokenizer.from_pretrained(args.model)
    x = tok(["The differential transformer cancels attention noise by subtracting "
             "a second softmax map from the first, which "] * 2,
            return_tensors="pt").input_ids[:, : args.seq_len].to(dev)

    # ---------------------------------------------------------------- FP32 gate
    print("[gate] loading fp32 ...", flush=True)
    m32 = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, attn_implementation="sdpa").to(dev).eval()
    cfg = m32.config
    res["config"] = {"hidden_size": cfg.hidden_size, "heads": cfg.num_attention_heads,
                     "kv_heads": cfg.num_key_value_heads, "head_dim": cfg.head_dim,
                     "n_layers": cfg.num_hidden_layers}
    with torch.no_grad():
        ref32 = m32(input_ids=x).logits.clone()
    h_before = backbone_hash(m32)

    attach_lowrank_split(m32, layers, rank=args.rank, gamma=args.gamma)
    n_split = sum(p.numel() for n, p in m32.named_parameters() if is_split_param_name(n))
    per_layer = args.rank * (cfg.hidden_size + cfg.num_attention_heads * cfg.head_dim)
    res["trainable_params"] = {
        "measured": n_split, "formula": per_layer * len(layers),
        "expected_helper": expected_split_param_count(m32),
        "per_layer": per_layer, "n_tensors": len(split_param_names(m32))}
    assert n_split == per_layer * len(layers) == expected_split_param_count(m32)

    with torch.no_grad():
        got32 = m32(input_ids=x).logits
    res["fp32_parity_max_abs"] = float((ref32 - got32).abs().max())
    res["fp32_parity_bit_exact"] = bool(torch.equal(ref32, got32))
    print(f"[gate] fp32 parity max_abs={res['fp32_parity_max_abs']:.3e} "
          f"bit_exact={res['fp32_parity_bit_exact']}", flush=True)

    # --------------------------------------------------- gradients / freezing
    freeze_backbone_keep_split(m32)
    m32.train()
    m32(input_ids=x, labels=x).loss.backward()
    bb_grads = [n for n, p in m32.named_parameters()
                if not is_split_param_name(n) and p.grad is not None]
    mod0 = next(iter_split_modules(m32))
    res["gradients"] = {
        "backbone_params_with_grad": len(bb_grads),
        "B_grad_nonzero_at_step0": float(mod0.lr_B.weight.grad.abs().max()) > 0,
        "A_grad_exactly_zero_at_step0": float(mod0.lr_A.weight.grad.abs().max()) == 0.0,
    }
    with torch.no_grad():                       # move B off zero, re-check A
        for mm in iter_split_modules(m32):
            mm.lr_B.weight.normal_(0, 0.01)
    m32.zero_grad(set_to_none=True)
    m32(input_ids=x, labels=x).loss.backward()
    res["gradients"]["A_grad_nonzero_after_B_moves"] = \
        float(mod0.lr_A.weight.grad.abs().max()) > 0
    m32.zero_grad(set_to_none=True)
    m32.eval()

    # the backbone must be byte-identical after a real backward pass
    res["backbone_sha256_before"] = h_before
    res["backbone_sha256_after"] = backbone_hash(m32)
    res["backbone_unchanged"] = h_before == backbone_hash(m32)

    # ------------------------------------------------ enabled/disabled switch
    n_off = set_split_enabled(m32, False)
    with torch.no_grad():
        off = m32(input_ids=x).logits
    set_split_enabled(m32, True)
    with torch.no_grad():
        on = m32(input_ids=x).logits
    res["switch"] = {
        "modules_switched": n_off,
        "disabled_equals_base_bit_exact": bool(torch.equal(ref32, off)),
        "enabled_differs_from_base": float((on - ref32).abs().max()),
    }

    # --------------------------------------------- strict checkpoint round trip
    st = split_state_dict(m32)
    info = load_split_state_dict(m32, st)
    fail_closed = {}
    for label, bad in (("missing", {k: v for k, v in list(st.items())[:-1]}),
                       ("unexpected", {**st, "model.layers.0.self_attn.lr_C.weight": torch.zeros(1)}),
                       ("backbone_key", {**st, "model.embed_tokens.weight": torch.zeros(1)})):
        try:
            load_split_state_dict(m32, bad)
            fail_closed[label] = "ACCEPTED (BUG)"
        except KeyError:
            fail_closed[label] = "rejected"
    res["checkpoint"] = {"round_trip": info, "fail_closed": fail_closed}

    # arm D bookkeeping: how many params the unfreeze arm actually trains
    newly = unfreeze_split_layer_attention(m32)
    unfrozen = sum(p.numel() for n, p in m32.named_parameters()
                   if p.requires_grad and not is_split_param_name(n))
    res["arm_D_unfreeze"] = {"n_tensors": len(newly), "backbone_params_unfrozen": unfrozen,
                             "total_trainable": unfrozen + n_split}
    del m32
    torch.cuda.empty_cache()

    # ---------------------------------------------------------------- BF16 gate
    for impl in ("eager", "sdpa"):
        print(f"[gate] bf16 {impl} ...", flush=True)
        mb = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation=impl).to(dev).eval()
        with torch.no_grad():
            r = mb(input_ids=x).logits.float().clone()
        attach_lowrank_split(mb, layers, rank=args.rank, gamma=args.gamma)
        with torch.no_grad():
            g = mb(input_ids=x).logits.float()
        res[f"bf16_{impl}"] = {
            "max_abs": float((r - g).abs().max()),
            "mean_abs": float((r - g).abs().mean()),
            "rel_max": float((r - g).abs().max() / r.abs().max()),
            "greedy_token_match": float((r.argmax(-1) == g.argmax(-1)).float().mean()),
            "bit_exact": bool(torch.equal(r, g)),
        }
        print(f"[gate] bf16 {impl}: {res[f'bf16_{impl}']}", flush=True)

        if impl == "sdpa":
            from transformers import DynamicCache
            # KV cache identity + decode==prefill, with a NON-zero delta
            def decode_gap():
                """bf16 prefill vs cached incremental decode, same model/state."""
                with torch.no_grad():
                    full = mb(input_ids=x).logits.float()
                    c = DynamicCache()
                    mb(input_ids=x[:, :-4], past_key_values=c, use_cache=True)
                    steps = [mb(input_ids=x[:, t:t + 1], past_key_values=c,
                                use_cache=True).logits[:, -1].float()
                             for t in range(x.shape[1] - 4, x.shape[1])]
                    return float((full[:, -4:] - torch.stack(steps, 1)).abs().max())

            # The base model itself is not prefill/decode bit-exact in bf16 (the
            # kernels differ by sequence length), so the meaningful gate is
            # "the split does not make it worse", not "the split is exact".
            set_split_enabled(mb, False)
            base_gap = decode_gap()
            set_split_enabled(mb, True)
            zero_gap = decode_gap()                    # split on, dQ still 0
            with torch.no_grad():
                for mm in iter_split_modules(mb):
                    mm.lr_B.weight.normal_(0, 0.01)
            res["decode_vs_prefill"] = {
                "base_bf16": base_gap, "split_zero_delta": zero_gap,
                "split_nonzero_delta": decode_gap(),
                "note": "bf16 kernel-level gap; base is the reference, not 0",
            }
            res["decode_vs_prefill_max_abs"] = res["decode_vs_prefill"]["split_nonzero_delta"]

            with torch.no_grad():
                set_split_enabled(mb, False)
                c_base = DynamicCache()
                mb(input_ids=x, past_key_values=c_base, use_cache=True)
                set_split_enabled(mb, True)
                c2 = DynamicCache()
                mb(input_ids=x, past_key_values=c2, use_cache=True)
            sb, ss = cache_summary(c_base), cache_summary(c2)
            res["kv_cache"] = {"base": sb, "split": ss, "identical": sb == ss}

            # no private inference state on the module
            leaks = {}
            for i, mm in enumerate(iter_split_modules(mb)):
                t = {k: list(v.shape) for k, v in vars(mm).items() if torch.is_tensor(v)}
                if t:
                    leaks[i] = t
            res["private_inference_cache"] = {"tensor_attrs": leaks,
                                              "clean": not leaks}
        del mb
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

    ok = (res["fp32_parity_bit_exact"] and res["backbone_unchanged"]
          and res["gradients"]["backbone_params_with_grad"] == 0
          and res["switch"]["disabled_equals_base_bit_exact"]
          and res["kv_cache"]["identical"] and res["private_inference_cache"]["clean"]
          # split must not degrade the base model's own bf16 decode consistency
          and res["decode_vs_prefill"]["split_zero_delta"] <= max(
              3 * res["decode_vs_prefill"]["base_bf16"], 1e-3)
          and all(v == "rejected" for v in fail_closed.values()))
    print("\nREALGATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
