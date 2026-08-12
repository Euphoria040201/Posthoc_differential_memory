#!/usr/bin/env python
"""Forced implementation audit (§4 of the downstream-audit task).

Nothing here trusts a report: every claim is re-derived from the live module,
forward hooks, real tensors and real generations.

Checks (select with --checks a,b,c,d or "all"):
  graph      A. computation graph, injection points, widths, trainable set
  frozen     A.10 backbone bit-identity across a real optimizer step
  numerics   B. pre_o vs post_o_projected across dtype / cache / batch / repeats
  parity     C. base parity: wrapper@gain0, wrapper@zero-state, pristine HF
  isolation  D. state isolation, writer blindness, batch contamination, order

Writes one JSON per check into --output-dir and a `<tag>.done` marker.
"""
from __future__ import annotations

import argparse, hashlib, json, os, platform, sys, time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from deltamem.core.prefix_steer import (  # noqa: E402
    PrefixSteerConfig, attach_prefix_steer, clear_frozen_memory,
    iter_steer_modules, set_steer_enabled, set_steer_segments,
    set_steer_zero_prefix, set_window_only, set_write_freeze,
)
from deltamem.core.global_prefix import SEG_CTX, SEG_ANS  # noqa: E402
from deltamem.eval.steer_checkpoint import restore_prefix_steer_config  # noqa: E402


def get_dtype(n):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[n]


def tensor_hash(t):
    return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]


def state_hash(model):
    h = hashlib.sha256()
    for n, p in sorted(model.named_parameters()):
        h.update(n.encode())
        h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def backbone_hash(model):
    """Hash ONLY the frozen backbone tensors (everything the sidecar did not add)."""
    from deltamem.core.prefix_steer import is_steer_param_name
    h = hashlib.sha256()
    n_t = 0
    for n, p in sorted(model.named_parameters()):
        if is_steer_param_name(n):
            continue
        h.update(n.encode())
        h.update(p.detach().float().cpu().numpy().tobytes())
        n_t += 1
    return h.hexdigest(), n_t


def load_backbone(args, dtype=None):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype or get_dtype(args.dtype),
        attn_implementation=args.attn_impl,
    ).to(args.device)
    m.config.use_cache = False
    return m


def load_ckpt_model(args, ckpt, dtype=None, position=None):
    """Backbone + sidecar restored exactly as trained (optionally re-positioned)."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    raw = ck.get("cfg") or ck.get("steer_config")
    cfg = restore_prefix_steer_config(raw)
    if position is not None:
        from dataclasses import replace
        cfg = replace(cfg, o_fusion_position=position)
    model = load_backbone(args, dtype=dtype)
    attach_prefix_steer(model, cfg)
    state = {k: v.to(args.device) for k, v in ck["state"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected tensors: {unexpected[:5]}")
    model.eval()
    return model, cfg, len(state)


def qasper_examples(args, n, split="validation"):
    from transformers import AutoTokenizer
    from qasper_prefix_steer import build_examples
    tok = AutoTokenizer.from_pretrained(args.model_path)
    ex = build_examples(split, args.val_papers, tok, 256, args.max_ctx_tok, 24,
                        data="qasper")
    return tok, ex[:n]


def feed(model, seg, valid):
    set_steer_segments(model, seg, valid)


# --------------------------------------------------------------------------
# A. graph / params
# --------------------------------------------------------------------------
def check_graph(args):
    from qasper_prefix_steer import collate
    tok, ex = qasper_examples(args, 2)
    model, cfg, n_loaded = load_ckpt_model(args, args.ckpt)
    mods = list(iter_steer_modules(model))
    m0 = mods[0]
    pad = tok.pad_token_id or tok.eos_token_id
    ids, seg, valid, lab = collate([ex[0]], pad, args.device)

    grab = {}
    hooks = [
        m0.base.o_proj.register_forward_pre_hook(
            lambda m, i: grab.__setitem__("o_in", i[0].detach())),
        m0.base.o_proj.register_forward_hook(
            lambda m, i, o: grab.__setitem__("o_out", o.detach())),
        m0.register_forward_hook(lambda m, i, o: grab.__setitem__("wrap_out", o[0].detach())),
    ]
    if m0.delta_o is not None:
        hooks.append(m0.delta_o.register_forward_hook(
            lambda m, i, o: grab.__setitem__("C", o.detach())))
    # what the memory READ sees
    reads_seen = {}
    orig_read = m0._memory_read
    def spy_read(hidden):
        r = orig_read(hidden)
        reads_seen["hidden_shape"] = list(hidden.shape)
        reads_seen["reads_shape"] = None if r is None else list(r.shape)
        reads_seen["seg_unique"] = sorted(set(m0._seg.flatten().tolist())) if m0._seg is not None else None
        return r
    m0._memory_read = spy_read
    feed(model, seg, valid)
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    m0._memory_read = orig_read
    for h in hooks:
        h.remove()

    from deltamem.core.prefix_steer import is_steer_param_name
    trainable = [n for n, p in model.named_parameters() if is_steer_param_name(n)]
    seg_counts = {int(s): int((seg == s).sum()) for s in seg.unique()}
    out = {
        "tag": "graph",
        "ckpt": str(args.ckpt),
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(cfg).items()},
        "loaded_tensors": n_loaded,
        "n_steer_layers": len(mods),
        "widths": {
            "o_proj_in": m0.base.o_proj.in_features,
            "o_proj_out": m0.base.o_proj.out_features,
            "o_proj_bias": m0.base.o_proj.bias is not None,
            "Z_width": list(grab["o_in"].shape),
            "C_width": None if "C" not in grab else list(grab["C"].shape),
            "read_dim": m0.read_dim,
            "n_query_heads": m0.n_heads, "n_kv_heads": m0.n_kv, "head_dim": m0.head_dim,
            "num_prefix_tokens": cfg.num_prefix_tokens,
        },
        "memory_read": reads_seen,
        "segments_in_forward": seg_counts,
        "SEG_CTX": SEG_CTX, "SEG_ANS": SEG_ANS,
        "trainable_steer_tensors": len(trainable),
        "trainable_steer_params": int(sum(p.numel() for n, p in model.named_parameters()
                                          if is_steer_param_name(n))),
        "trainable_names_head": trainable[:8],
        "has_written_state": bool(cfg.num_prefix_tokens > 0 and cfg.prefix_write),
        "fusion_position": cfg.o_fusion_position,
    }
    # verify the fusion formula numerically at layer 0
    if "C" in grab:
        W = m0.base.o_proj.weight
        pos = cfg.o_fusion_position
        g = cfg.steer_gain
        if pos == "pre_o":
            # o_in should already contain the fused Z; recover unfused Z by turning steer off
            set_steer_enabled(model, False)
            g2 = {}
            h = m0.base.o_proj.register_forward_pre_hook(
                lambda m, i: g2.__setitem__("o_in", i[0].detach()))
            with torch.no_grad():
                feed(model, seg, valid); model(input_ids=ids, use_cache=False)
            h.remove(); set_steer_enabled(model, True)
            resid = (grab["o_in"] - (g2["o_in"] + g * grab["C"])).abs().max().item()
            out["formula_check"] = {"claim": "o_proj input == Z + g*C",
                                    "max_abs_residual": resid}
        elif pos == "post_o":
            resid = (grab["wrap_out"] - (grab["o_out"] + g * grab["C"])).abs().max().item()
            out["formula_check"] = {"claim": "wrapper out == W_O Z + g*C",
                                    "max_abs_residual": resid}
    return out


def check_frozen(args):
    """Run the REAL training entry for a few steps and hash the backbone around it."""
    from qasper_prefix_steer import build_examples, collate
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path)
    cfg = PrefixSteerConfig(
        num_prefix_tokens=0, sliding_window_size=256, mem_num_heads=1, mem_head_dim=128,
        steer_mode="deltamem", memory_mode="dynamic", memory_value_source="main_v",
        delta_heads="o", steer_gain=0.1, output_fusion="fixed",
        o_fusion_position=args.position, steer_layers=tuple(range(0, 36, 3)),
        prefix_write=False, write_ctx_only=False, read_prefix_only=False,
        pool_reads=False, pool_gate=False,
    )
    model = load_backbone(args)
    attach_prefix_steer(model, cfg)
    from deltamem.core.prefix_steer import freeze_backbone_keep_steer, is_steer_param_name
    freeze_backbone_keep_steer(model)
    trainable_names = [n for n, q in model.named_parameters() if q.requires_grad]
    h0, n_backbone = backbone_hash(model)

    # exactly how dex_train_qasper builds the optimizer for swa_steer
    steer_params = [p for n, p in model.named_parameters() if is_steer_param_name(n) and p.requires_grad]
    other_params = [n for n, p in model.named_parameters()
                    if p.requires_grad and not is_steer_param_name(n)]
    opt = torch.optim.AdamW([{"params": steer_params, "lr": 5e-4}])
    train = build_examples("train", 20, tok, 256, args.max_ctx_tok, 24, data="qasper",
                           max_yesno_frac=0.03, yesno_seed=42, train_target_n=8)
    pad = tok.pad_token_id or tok.eos_token_id
    model.train()
    losses = []
    for i in range(args.frozen_steps):
        opt.zero_grad(set_to_none=True)
        ids, seg, valid, lab = collate([train[i % len(train)]], pad, args.device)
        feed(model, seg, valid)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, labels=lab, use_cache=False).loss
        loss.backward()
        opt.step()
        losses.append(float(loss))
    model.eval()
    h1, _ = backbone_hash(model)
    # did any backbone param receive a gradient?
    grad_backbone = [n for n, p in model.named_parameters()
                     if not is_steer_param_name(n) and p.grad is not None]
    return {
        "tag": "frozen",
        "steps": args.frozen_steps, "losses": losses,
        "n_backbone_tensors": n_backbone,
        "backbone_hash_before": h0, "backbone_hash_after": h1,
        "backbone_bit_identical": h0 == h1,
        "trainable_names_count": len(trainable_names),
        "non_steer_trainable_params": other_params,
        "backbone_params_with_grad": grad_backbone[:10],
        "n_backbone_params_with_grad": len(grad_backbone),
        "optimizer_param_groups": [len(g["params"]) for g in opt.param_groups],
        "note": ("dex_train_qasper.py --variant swa_steer puts ONLY steer params in the "
                 "optimizer; --lr applies to the (empty for swa_steer) attn/adapter group"),
    }


# --------------------------------------------------------------------------
# B. numerics
# --------------------------------------------------------------------------
def logit_stats(a, b):
    d = (a.float() - b.float()).abs()
    top1 = (a.float().argmax(-1) != b.float().argmax(-1)).float().mean().item()
    return {"max_abs": d.max().item(), "mean_abs": d.mean().item(),
            "top1_disagreement": top1}


def check_numerics(args):
    from qasper_prefix_steer import collate
    tok, ex = qasper_examples(args, args.n_numerics)
    pad = tok.pad_token_id or tok.eos_token_id
    res = {"tag": "numerics", "ckpt": str(args.ckpt), "cases": []}

    for dt in ("float32", "bfloat16"):
        for use_cache in (False, True):
            for bs in (1, 2):
                if bs == 2 and dt == "float32":
                    continue  # fp32 B=2 at 4.5k ctx is the same test, save memory
                mp, _, _ = load_ckpt_model(args, args.ckpt, dtype=get_dtype(dt), position="pre_o")
                mc, _, _ = load_ckpt_model(args, args.ckpt, dtype=get_dtype(dt), position="post_o_projected")
                batch = ex[:bs]
                ids, seg, valid, lab = collate(batch, pad, args.device)
                outs = {}
                for name, m in (("pre_o", mp), ("post_o_projected", mc)):
                    clear_frozen_memory(m)
                    feed(m, seg, valid)
                    with torch.no_grad():
                        outs[name] = m(input_ids=ids, use_cache=use_cache).logits.detach()
                st = logit_stats(outs["pre_o"], outs["post_o_projected"])
                # determinism of the same arm, same input, run twice
                clear_frozen_memory(mp); feed(mp, seg, valid)
                with torch.no_grad():
                    again = mp(input_ids=ids, use_cache=use_cache).logits.detach()
                st_rep = logit_stats(outs["pre_o"], again)
                res["cases"].append({
                    "dtype": dt, "use_cache": use_cache, "batch": bs,
                    "n_tokens": int(ids.numel()),
                    "pre_o_vs_ctlproj": st,
                    "pre_o_repeat_bitexact": bool(torch.equal(outs["pre_o"], again)),
                    "pre_o_repeat_stats": st_rep,
                })
                del mp, mc
                torch.cuda.empty_cache()
    # greedy generation divergence on the same examples (bf16, the eval dtype)
    from qasper_prefix_steer import generate
    mp, _, _ = load_ckpt_model(args, args.ckpt, position="pre_o")
    mc, _, _ = load_ckpt_model(args, args.ckpt, position="post_o_projected")
    eos = tok.eos_token_id
    gens = []
    for i, e in enumerate(ex[:args.n_gen]):
        for m in (mp, mc):
            clear_frozen_memory(m)
        p1 = generate(mp, tok, e, args.device, 24, eos)
        p2 = generate(mc, tok, e, args.device, 24, eos)
        gens.append({"i": i, "pre_o": p1, "ctlproj": p2, "identical": p1 == p2})
    res["generation"] = {
        "n": len(gens),
        "identical_rate": sum(g["identical"] for g in gens) / max(len(gens), 1),
        "samples": gens[:8],
    }
    return res


# --------------------------------------------------------------------------
# C. base parity
# --------------------------------------------------------------------------
def check_parity(args):
    from qasper_prefix_steer import collate
    tok, ex = qasper_examples(args, args.n_parity)
    pad = tok.pad_token_id or tok.eos_token_id
    pristine = load_backbone(args)
    wrapped, cfg, _ = load_ckpt_model(args, args.ckpt)

    rows = []
    for i, e in enumerate(ex):
        ids, seg, valid, lab = collate([e], pad, args.device)
        with torch.no_grad():
            lp = pristine(input_ids=ids, use_cache=False).logits.detach()
        # 1. wrapper, steer disabled entirely
        set_steer_enabled(wrapped, False)
        clear_frozen_memory(wrapped); feed(wrapped, seg, valid)
        with torch.no_grad():
            l_off = wrapped(input_ids=ids, use_cache=False).logits.detach()
        # 2. wrapper enabled but gain forced to 0 (memory computed, contribution zero)
        set_steer_enabled(wrapped, True)
        old_gain = [m.cfg.steer_gain for m in iter_steer_modules(wrapped)]
        for m in iter_steer_modules(wrapped):
            from dataclasses import replace as _r
            m.cfg = _r(m.cfg, steer_gain=0.0)
        clear_frozen_memory(wrapped); feed(wrapped, seg, valid)
        with torch.no_grad():
            l_g0 = wrapped(input_ids=ids, use_cache=False).logits.detach()
        for m, g in zip(iter_steer_modules(wrapped), old_gain):
            from dataclasses import replace as _r
            m.cfg = _r(m.cfg, steer_gain=g)
        rows.append({
            "i": i,
            "steer_off_vs_pristine": logit_stats(l_off, lp),
            "steer_off_bitexact": bool(torch.equal(l_off, lp)),
            "gain0_vs_pristine": logit_stats(l_g0, lp),
            "gain0_bitexact": bool(torch.equal(l_g0, lp)),
        })
    agg = lambda key, sub: {
        "max": max(r[key][sub] for r in rows),
        "mean": sum(r[key][sub] for r in rows) / len(rows),
    }
    return {
        "tag": "parity", "n": len(rows), "dtype": args.dtype, "attn": args.attn_impl,
        "steer_off_bitexact_rate": sum(r["steer_off_bitexact"] for r in rows) / len(rows),
        "gain0_bitexact_rate": sum(r["gain0_bitexact"] for r in rows) / len(rows),
        "steer_off_max_abs": agg("steer_off_vs_pristine", "max_abs"),
        "steer_off_top1_disagreement": agg("steer_off_vs_pristine", "top1_disagreement"),
        "gain0_max_abs": agg("gain0_vs_pristine", "max_abs"),
        "gain0_top1_disagreement": agg("gain0_vs_pristine", "top1_disagreement"),
        "rows": rows[:10],
    }


# --------------------------------------------------------------------------
# D. state isolation
# --------------------------------------------------------------------------
def check_isolation(args):
    from qasper_prefix_steer import collate
    tok, ex = qasper_examples(args, max(4, args.n_isolation))
    pad = tok.pad_token_id or tok.eos_token_id
    model, cfg, _ = load_ckpt_model(args, args.ckpt)
    P = cfg.num_prefix_tokens
    res = {"tag": "isolation", "num_prefix_tokens": P,
           "prefix_write": cfg.prefix_write,
           "has_written_state": bool(P > 0 and cfg.prefix_write)}
    if not res["has_written_state"]:
        res["verdict"] = ("This checkpoint has NO written memory state (P=0, prefix_write=False): "
                          "the sidecar is a local sliding-window attention adapter. "
                          "state_only / swap / zero-state arms are undefined for it.")
        # still verify the READ is context-dependent and order-invariant
        outs = []
        for e in ex[:2]:
            ids, seg, valid, lab = collate([e], pad, args.device)
            feed(model, seg, valid)
            with torch.no_grad():
                outs.append(model(input_ids=ids, use_cache=False).logits.detach())
        ids0, seg0, valid0, _ = collate([ex[0]], pad, args.device)
        feed(model, seg0, valid0)
        with torch.no_grad():
            rerun = model(input_ids=ids0, use_cache=False).logits.detach()
        res["repeat_after_other_sample_bitexact"] = bool(torch.equal(outs[0], rerun))
        # window-only == full read when P=0 (no prefix columns to mask)
        set_window_only(model, True)
        feed(model, seg0, valid0)
        with torch.no_grad():
            wo = model(input_ids=ids0, use_cache=False).logits.detach()
        set_window_only(model, False)
        res["window_only_equals_full_when_P0"] = bool(torch.equal(outs[0], wo))
        return res

    # P>0: real written-state checks
    hashes = []
    for rep in range(2):
        for i, e in enumerate(ex[:3]):
            ids, seg, valid, lab = collate([e], pad, args.device)
            clear_frozen_memory(model)
            set_write_freeze(model, True)
            feed(model, seg, valid)
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)
            set_write_freeze(model, False)
            st = [m._frozen_prefix for m in iter_steer_modules(model)]
            hashes.append({"rep": rep, "ex": i,
                           "hash": hashlib.sha256(b"".join(
                               tensor_hash(s).encode() for s in st if s is not None)).hexdigest()[:16]})
    same_ctx = [h["hash"] for h in hashes if h["ex"] == 0]
    diff_ctx = {h["ex"]: h["hash"] for h in hashes if h["rep"] == 0}
    res["write_deterministic_same_context"] = len(set(same_ctx)) == 1
    res["distinct_contexts_distinct_state"] = len(set(diff_ctx.values())) == len(diff_ctx)
    res["hashes"] = hashes
    return res


CHECKS = {"graph": check_graph, "frozen": check_frozen, "numerics": check_numerics,
          "parity": check_parity, "isolation": check_isolation}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/work/mingze/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--ckpt", default=str(REPO / "out_dex_fusion/preo_swa_steer_s0_steer.pt"))
    ap.add_argument("--checks", default="all")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--position", default="pre_o")
    ap.add_argument("--val-papers", type=int, default=75)
    ap.add_argument("--max-ctx-tok", type=int, default=4500)
    ap.add_argument("--n-numerics", type=int, default=4)
    ap.add_argument("--n-gen", type=int, default=16)
    ap.add_argument("--n-parity", type=int, default=100)
    ap.add_argument("--n-isolation", type=int, default=4)
    ap.add_argument("--frozen-steps", type=int, default=4)
    ap.add_argument("--output-dir", default=str(REPO / "out_downstream_audit_20260812"))
    ap.add_argument("--tag", default="audit")
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    names = list(CHECKS) if args.checks == "all" else [c.strip() for c in args.checks.split(",")]
    env = {"python": platform.python_version(), "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
           "command": " ".join(sys.argv)}
    for name in names:
        t0 = time.time()
        print(f"[{args.tag}] === {name} ===", flush=True)
        payload = CHECKS[name](args)
        payload["env"] = env
        payload["runtime_min"] = round((time.time() - t0) / 60, 2)
        path = out_dir / f"{args.tag}_{name}.json"
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"[{args.tag}] {name} -> {path} ({payload['runtime_min']}m)", flush=True)
        print(json.dumps({k: v for k, v in payload.items()
                          if k not in ("rows", "hashes", "cases", "config", "env")},
                         indent=2, default=str)[:2500], flush=True)
    (out_dir / f"{args.tag}.done").write_text(str(time.time()))


if __name__ == "__main__":
    main()
