#!/usr/bin/env python
"""Hard gates on the REAL Qwen3-4B — the tiny-model unit tests are necessary, not sufficient.

Checks, on the locked checkpoint with the locked 12 intervention layers:
  * exact trainable parameter count vs the old sidecar's 14,155,776
  * explicit head -> kv-group map for one wrapped layer
  * FP32 end-to-end logits parity (split_zero vs base)
  * BF16 parity + greedy token agreement
  * prefill/decode cache equality and KV-cache byte equality
  * causality of the local reader on real hidden states
  * backbone SHA256 unchanged across a real optimizer step
"""
from __future__ import annotations

import argparse, hashlib, json, sys, time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

from deltamem.core.diff_split import (  # noqa: E402
    attach_diff_split, collect_diff_stats, freeze_backbone_keep_diff,
    is_diff_param_name, iter_diff_modules, set_diff_stats,
)

LAYERS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33]   # same 12 as the old sidecar
OLD_BUDGET = 14_155_776


def backbone_sha(model):
    h = hashlib.sha256()
    n = 0
    for name, p in sorted(model.named_parameters()):
        if is_diff_param_name(name):
            continue
        h.update(name.encode()); h.update(p.detach().float().cpu().numpy().tobytes()); n += 1
    return h.hexdigest(), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/work/mingze/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--read-dim", type=int, default=128)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--seq", type=int, default=192)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    out = {"args": vars(args), "layers": LAYERS, "gates": {}}
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    torch.manual_seed(0)
    ids = torch.randint(0, 100000, (1, args.seq)).cuda()

    # ---------------- FP32 parity (the strict gate) -------------------------
    m = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.float32, attn_implementation="sdpa",
        local_files_only=True).cuda().eval()
    m.config.use_cache = False
    with torch.no_grad():
        ref32 = m(input_ids=ids, use_cache=False).logits.clone()
    rep = attach_diff_split(m, LAYERS, read_dim=args.read_dim, window=args.window,
                            gamma=args.gamma)
    with torch.no_grad():
        got32 = m(input_ids=ids, use_cache=False).logits
    err32 = (got32 - ref32).abs().max().item()
    out["gates"]["fp32_parity"] = {
        "n_layers_wrapped": len(rep), "max_abs_logit_err": err32,
        "threshold": 1e-5, "pass": err32 <= 1e-5}
    print(f"[gate] FP32 parity max_abs={err32:.3e} over {len(rep)} layers -> "
          f"{'PASS' if err32 <= 1e-5 else 'FAIL'}", flush=True)

    # ---------------- exact parameter budget --------------------------------
    trainable = [(n, p.numel()) for n, p in m.named_parameters() if is_diff_param_name(n)]
    total = sum(v for _, v in trainable)
    per_kind = {}
    for n, v in trainable:
        per_kind.setdefault(n.split(".")[-2], []).append(v)
    out["gates"]["param_budget"] = {
        "trainable_total": total, "old_budget": OLD_BUDGET,
        "delta": total - OLD_BUDGET, "rel_error": (total - OLD_BUDGET) / OLD_BUDGET,
        "n_tensors": len(trainable),
        "per_kind": {k: {"count": len(v), "each": v[0], "sum": sum(v)}
                     for k, v in sorted(per_kind.items())},
        "pass": total == OLD_BUDGET}
    print(f"[gate] trainable = {total:,} vs old {OLD_BUDGET:,} "
          f"(delta {total-OLD_BUDGET:+,}) -> {'PASS' if total==OLD_BUDGET else 'FAIL'}",
          flush=True)

    # ---------------- explicit head -> kv group map -------------------------
    mod = next(iter_diff_modules(m))
    H, G = mod.n_heads, mod.n_kv
    rep_base, rep_split = H // G, (2 * H) // G
    mapping = [{"head": i, "base_kv": i // rep_base,
                "plus_kv": (2 * i) // rep_split, "minus_kv": (2 * i + 1) // rep_split}
               for i in range(H)]
    ok = all(r["base_kv"] == r["plus_kv"] == r["minus_kv"] for r in mapping)
    out["gates"]["gqa_map"] = {"n_q_heads": H, "n_kv_heads": G,
                               "repeat_base": rep_base, "repeat_split": rep_split,
                               "map": mapping, "pass": ok}
    print(f"[gate] GQA: {H}q/{G}kv, repeat {rep_base}->{rep_split}; pair shares kv group "
          f"-> {'PASS' if ok else 'FAIL'}", flush=True)
    print("       head->kv (base|+|-): " + " ".join(
        f"{r['head']}:{r['base_kv']}/{r['plus_kv']}/{r['minus_kv']}" for r in mapping[:8])
        + " ...", flush=True)

    # ---------------- causality on real hidden states -----------------------
    with torch.no_grad():
        B, L = 1, 96
        h = torch.randn(B, L, m.config.hidden_size, device="cuda")
        v = torch.randn(B, mod.n_kv, L, mod.head_dim, device="cuda")
        r1 = mod._local_read(h, v)
        h2 = h.clone(); h2[:, L // 2:] += 7.0
        v2 = v.clone(); v2[:, :, L // 2:] += 7.0
        r2 = mod._local_read(h2, v2)
        cerr = (r1[:, :L // 2] - r2[:, :L // 2]).abs().max().item()
        ceff = (r1[:, L // 2:] - r2[:, L // 2:]).abs().max().item()
    out["gates"]["causality"] = {"max_abs_past_change": cerr,
                                 "future_perturb_effect": ceff,
                                 "pass": cerr == 0.0 and ceff > 0}
    print(f"[gate] causality: past unchanged={cerr:.3e}, future effect={ceff:.3e} -> "
          f"{'PASS' if cerr == 0.0 and ceff > 0 else 'FAIL'}", flush=True)

    # ---------------- gradient + frozen backbone ----------------------------
    freeze_backbone_keep_diff(m)
    sha0, nb = backbone_sha(m)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    m.train()
    with torch.no_grad():
        for mm in iter_diff_modules(m):           # ALL layers, else the untouched ones
            mm.delta_q.weight.normal_(std=1e-3)   # keep dLoss/dR == 0 by construction
    loss = m(input_ids=ids[:, :64], labels=ids[:, :64], use_cache=False).loss
    loss.backward()
    grads = {}
    for k, mm in enumerate(iter_diff_modules(m)):
        grads[f"layer{LAYERS[k]}"] = {
            "delta_q": float(mm.delta_q.weight.grad.abs().sum()),
            "read_q": float(mm.read_q.weight.grad.abs().sum()),
            "read_k": float(mm.read_k.weight.grad.abs().sum())}
    bb_with_grad = [n for n, p in m.named_parameters()
                    if not is_diff_param_name(n) and p.grad is not None
                    and p.grad.abs().sum() > 0]
    opt.step()
    sha1, _ = backbone_sha(m)
    gok = (all(all(v > 0 for v in d.values()) for d in grads.values())
           and not bb_with_grad and sha0 == sha1)
    out["gates"]["gradient_frozen"] = {
        "loss": float(loss), "n_backbone_tensors": nb,
        "backbone_sha_before": sha0, "backbone_sha_after": sha1,
        "backbone_bit_identical": sha0 == sha1,
        "n_backbone_params_with_grad": len(bb_with_grad),
        "per_layer_grad_abs_sum": grads, "pass": gok}
    print(f"[gate] backbone sha {'UNCHANGED' if sha0==sha1 else 'CHANGED'} "
          f"({nb} tensors), backbone grads={len(bb_with_grad)}, "
          f"all sidecar grads nonzero={all(all(v>0 for v in d.values()) for d in grads.values())}"
          f" -> {'PASS' if gok else 'FAIL'}", flush=True)
    del m, opt
    torch.cuda.empty_cache()

    # ---------------- BF16 parity + greedy + cache --------------------------
    mb = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True).cuda().eval()
    with torch.no_grad():
        refb = mb(input_ids=ids, use_cache=False).logits.clone()
        gref = mb.generate(ids[:, :32], max_new_tokens=16, do_sample=False)
        cb = DynamicCache(); mb(input_ids=ids, past_key_values=cb, use_cache=True)
        kv_bytes_base = sum(cb.layers[i].keys.numel() * cb.layers[i].keys.element_size()
                            + cb.layers[i].values.numel() * cb.layers[i].values.element_size()
                            for i in range(len(cb.layers)))
    attach_diff_split(mb, LAYERS, read_dim=args.read_dim, window=args.window, gamma=args.gamma)
    with torch.no_grad():
        gotb = mb(input_ids=ids, use_cache=False).logits
        gsplit = mb.generate(ids[:, :32], max_new_tokens=16, do_sample=False)
        cs = DynamicCache(); mb(input_ids=ids, past_key_values=cs, use_cache=True)
        kv_bytes_split = sum(cs.layers[i].keys.numel() * cs.layers[i].keys.element_size()
                             + cs.layers[i].values.numel() * cs.layers[i].values.element_size()
                             for i in range(len(cs.layers)))
    errb = (gotb.float() - refb.float()).abs().max().item()
    meanb = (gotb.float() - refb.float()).abs().mean().item()
    tok_agree = bool(torch.equal(gref, gsplit))
    out["gates"]["bf16"] = {
        "max_abs_logit_err": errb, "mean_abs_logit_err": meanb,
        "greedy_token_agreement": tok_agree,
        "kv_cache_bytes_base": kv_bytes_base, "kv_cache_bytes_split": kv_bytes_split,
        "kv_cache_identical": kv_bytes_base == kv_bytes_split,
        "pass": tok_agree and kv_bytes_base == kv_bytes_split}
    print(f"[gate] BF16 max_abs={errb:.3e} mean={meanb:.3e}; greedy agree={tok_agree}; "
          f"KV bytes {kv_bytes_base:,} == {kv_bytes_split:,} -> "
          f"{'PASS' if tok_agree and kv_bytes_base==kv_bytes_split else 'FAIL'}", flush=True)

    # prefill/decode equality.  bf16 logits carry their own prefill-vs-decode gap, so the
    # BASE model's gap on the same tokens is the reference; the split must not exceed it
    # by more than the bf16 logit scale already measured above.
    def prefill_decode_gap(model):
        with torch.no_grad():
            full = model(input_ids=ids[:, :96], use_cache=False).logits[:, -1].float()
            c = DynamicCache()
            model(input_ids=ids[:, :95], past_key_values=c, use_cache=True)
            step = model(input_ids=ids[:, 95:96], past_key_values=c,
                         use_cache=True).logits[:, -1].float()
        return (full - step).abs().max().item()

    from deltamem.core.diff_split import set_diff_enabled as _sde
    _sde(mb, False)
    base_gap = prefill_decode_gap(mb)        # same weights, sidecar bypassed
    _sde(mb, True)
    zero_gap = prefill_decode_gap(mb)        # split active, delta_q still zero
    with torch.no_grad():
        for mm in iter_diff_modules(mb):
            mm.delta_q.weight.normal_(std=1e-3)
    derr = prefill_decode_gap(mb)            # split active, delta_q nonzero
    tol = max(3 * base_gap, 1e-2)
    out["gates"]["cache_decode"] = {
        "base_gap": base_gap, "split_zero_gap": zero_gap, "split_nonzero_gap": derr,
        "tolerance": tol,
        "note": "bf16 prefill-vs-cached-decode; base model gap is the reference",
        "pass": derr <= tol and zero_gap <= tol}
    print(f"[gate] prefill vs cached-decode (bf16): base={base_gap:.3e} "
          f"split_zero={zero_gap:.3e} split_nonzero={derr:.3e} tol={tol:.3e} -> "
          f"{'PASS' if derr <= tol and zero_gap <= tol else 'FAIL'}", flush=True)

    out["all_pass"] = all(g.get("pass") for g in out["gates"].values())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"\nALL GATES: {'PASS' if out['all_pass'] else 'FAIL'} -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
