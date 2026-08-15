"""Zero-shot eval of our (Qasper-trained) prefix-steer ckpt on HotpotQA.

Loads our steer ckpt (trained ONLY on Qasper), attaches it to a frozen Qwen3-4B,
and evaluates ZERO-SHOT on HotpotQA using the project's OFFICIAL prompt + F1/EM.
Also runs `base` (steer off) on the same samples as the paired control.
Same 500 samples (seed 42) as the base/δ-mem targets; shard by index.
"""

from __future__ import annotations

import argparse, json, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deltamem.eval.benchmark_compare import (
    HOTPOTQA_PROMPT_TEMPLATE, build_hotpotqa_context, extract_first_line,
    hotpotqa_f1, hotpotqa_exact_match, load_hotpotqa,
)
from deltamem.eval.steer_checkpoint import (
    has_persistent_history_writer,
    load_steer_state_strict,
    restore_prefix_steer_config,
)
from deltamem.core.prefix_steer import (
    attach_prefix_steer, freeze_backbone_keep_steer,
    set_steer_segments, set_steer_enabled, set_steer_zero_prefix,
    set_write_freeze, clear_frozen_memory, set_gate_off, set_window_only,
)
from deltamem.core.global_prefix import SEG_CTX, SEG_ANS
from deltamem.core.diff_split import set_diff_enabled as _set_diff_enabled


def get_dtype(n):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[n]


def load_ours(model, ckpt_path, steer_gain_override=None, pool_override=None):
    ck = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ck, dict) and "diff_config" in ck and "state" in ck:
        # post-hoc differential head split: a different module family than the
        # prefix sidecar, so it is attached and loaded through its own helpers.
        from deltamem.core.diff_split import (
            attach_diff_split, freeze_backbone_keep_diff)
        dc = ck["diff_config"]
        attach_diff_split(model, tuple(dc["layers"]), read_dim=dc["read_dim"],
                          window=dc["window"], gamma=dc["gamma"],
                          dynamic_gate=dc["dynamic_gate"])
        freeze_backbone_keep_diff(model)
        missing, unexpected = model.load_state_dict(ck["state"], strict=False)
        assert not unexpected, unexpected[:5]
        loaded = [k for k in ck["state"] if k in dict(model.named_parameters())]
        assert len(loaded) == len(ck["state"]), (len(loaded), len(ck["state"]))
        print(f"[ours] loaded {len(ck['state'])} diff-split tensors; cfg={dc}",
              flush=True)
        # The split has P=0 and no WRITE path at all, so the prefix-sidecar
        # branches downstream (write pass, pooling) must all resolve to "off".
        from types import SimpleNamespace
        return SimpleNamespace(prefix_write=False, write_ctx_only=False,
                               num_prefix_tokens=0, history_pool_mode="none",
                               diff_config=dc)
    if not isinstance(ck, dict) or "cfg" not in ck or "state" not in ck:
        raise ValueError("prefix checkpoint must contain cfg and state mappings")
    cfg = restore_prefix_steer_config(
        ck["cfg"],
        steer_gain_override=steer_gain_override,
        pool_reads_override=pool_override,
    )
    attach_prefix_steer(model, cfg)
    freeze_backbone_keep_steer(model)
    load_steer_state_strict(model, ck["state"], label="ours")
    print(f"[ours] loaded {len(ck['state'])} steer tensors; prefix_write={cfg.prefix_write}; cfg={cfg}", flush=True)
    return cfg


def should_run_context_write(config, *, condition: str, write_pass: str) -> bool:
    """Whether this eval cell needs the separate document WRITE forward."""

    return (
        write_pass == "ctx"
        and condition != "base"
        and has_persistent_history_writer(config)
    )


@torch.no_grad()
def gen_ours(model, tok, input_ids, dev, max_new, eos):
    ids = input_ids[0].tolist()
    seg = [SEG_CTX] * len(ids)
    out = []
    for _ in range(max_new):
        iid = torch.tensor([ids], device=dev)
        sgt = torch.tensor([seg], device=dev)
        set_steer_segments(model, sgt, torch.ones_like(iid, dtype=torch.bool))
        nxt = int(model(input_ids=iid, use_cache=False).logits[0, -1].argmax())
        out.append(nxt); ids.append(nxt); seg.append(SEG_ANS)
        if nxt == eos or "\n" in tok.decode([nxt]):
            break
    return tok.decode(out, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=32)  # match official hotpotqa_eval
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--conds", default="base,ours",
                    help="base=steer off; ours=full memory; ours_window_only=prefix MASKED from "
                         "the read (SWA window kept, M_t=0) -> the REAL prefix ablation; "
                         "ours_zero_read=whole memory read:=0 (== base for zero-init delta) -- "
                         "this is a WHOLE-READ ablation, NOT a prefix ablation. F1_ours - "
                         "F1_window_only = PREFIX value; F1_window_only - F1_base = SWA-steer value.")
    ap.add_argument("--backbone-window", type=int, default=0,
                    help="must match training: bound the frozen backbone to this window")
    ap.add_argument("--steer-gain", type=float, default=None,
                    help="override ckpt steer_gain (e.g. -0.1 = negative/anti steering)")
    ap.add_argument("--gate-off", action="store_true",
                    help="force the learned pool gate to g==1 (reads=R+M) with the GATED "
                         "training weights -- isolates 'gate learned badly' from 'gate "
                         "training moved everything to a worse basin'")
    ap.add_argument("--pool-alpha", type=float, default=1.0,
                    help="reads_t = R_t + alpha * M_t (diagnostic; 1.0 = trained behavior, "
                         "0 = exactly the no-pool computation)")
    ap.add_argument("--pool-override", default="none", choices=["none", "true", "false"],
                    help="force pool_reads on/off regardless of the ckpt cfg (pooling has "
                         "no params, so e.g. a nopool-trained ckpt can be run WITH pooling)")
    ap.add_argument("--write-pass", default="ctx", choices=["inline", "ctx"],
                    help="ctx = two-pass: WRITE the memory from the raw context tokens only "
                         "(seg=SEG_CTX, question absent), freeze it, then generate with the "
                         "full prompt. Matches ctx-mode training, where write_ctx_only masks "
                         "the write to SEG_CTX -- 'inline' would label the whole eval prompt "
                         "SEG_CTX and leak the question into the write.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    dev = "cuda"
    # backbone_window is NOT stored in cfg and NOT auto-restored -> a ckpt trained with a
    # BOUNDED backbone (bw>0) but evaluated at the default bw=0 would silently run on a
    # DIFFERENT backbone architecture. Verify against the ckpt's saved training args.
    _train_bw = int((torch.load(args.ckpt, map_location="cpu").get("args", {}) or {}).get("backbone_window", 0) or 0)
    if args.backbone_window != _train_bw:
        raise ValueError(f"backbone_window mismatch: ckpt was TRAINED with backbone_window="
                         f"{_train_bw} but eval got --backbone-window {args.backbone_window}. "
                         f"Pass --backbone-window {_train_bw} to match the training backbone.")
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    _kw = {}
    if args.backbone_window > 0:
        from transformers import AutoConfig
        _bc = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
        _tc = _bc.get_text_config() if hasattr(_bc, "get_text_config") else _bc
        _tc.sliding_window = args.backbone_window
        _tc.layer_types = ["sliding_attention"] * _tc.num_hidden_layers
        _kw["config"] = _bc
        print(f"[eval] BOUNDED backbone: window={args.backbone_window}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=get_dtype(args.dtype),
        attn_implementation=args.attn_impl, local_files_only=True, **_kw).to(dev).eval()
    _po = {"none": None, "true": True, "false": False}[args.pool_override]
    _cfg = load_ours(model, args.ckpt, steer_gain_override=args.steer_gain, pool_override=_po)
    if _cfg.prefix_write and _cfg.write_ctx_only and args.write_pass == "inline":
        raise ValueError("a dynamic + write_ctx_only ckpt writes memory from the DOCUMENT only; "
                         "--write-pass inline would relabel the whole eval prompt SEG_CTX and "
                         "re-WRITE the memory from context+question+partial-answer every step "
                         "(train/eval mismatch). Use --write-pass ctx.")
    model.eval()   # attach_prefix_steer() adds NEW modules that default to
                   # training=True; the earlier .eval() does not cover them.
    _has_writer = has_persistent_history_writer(_cfg)
    if args.write_pass == "ctx" and not _has_writer:
        print(
            "[eval] checkpoint has no persistent history writer; "
            "skipping redundant context WRITE pass",
            flush=True,
        )
    eos = tok.eos_token_id
    from deltamem.core.prefix_steer import iter_steer_modules, set_write_freeze
    mods = list(iter_steer_modules(model))
    for m_ in mods:
        m_._pool_alpha = args.pool_alpha
    if args.gate_off:
        set_gate_off(model, True); print("[eval] gate FORCED to g==1", flush=True)
    if args.pool_alpha != 1.0:
        print(f"[eval] pool_alpha = {args.pool_alpha}", flush=True)

    data = load_hotpotqa(cache_dir=Path.home() / ".cache/huggingface/datasets",
                         max_samples=args.max_samples, seed=args.seed, local_files_only=True)
    shard = [(i, it) for i, it in enumerate(data) if i % args.num_shards == args.shard]
    print(f"[shard {args.shard}] {len(shard)} examples", flush=True)

    # (the old document-level pooled-vector interventions were removed together with the
    # document-level pooling itself; M_t is token-specific and cannot be swapped wholesale)

    conds = args.conds.split(",")
    res = {c: {"f1": 0.0, "em": 0.0, "n": 0} for c in conds}
    recs = []
    t0 = time.time()
    for idx, item in shard:
        ctx_str = build_hotpotqa_context(item)
        prompt = HOTPOTQA_PROMPT_TEMPLATE.format(
            context=ctx_str, question=str(item["question"]).strip())
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                      add_generation_prompt=True, return_tensors="pt", return_dict=True)
        input_ids = enc["input_ids"].to(dev)
        gold = str(item["answer"]).strip()
        row = {"id": str(item["id"]), "gold": gold}
        for c in conds:
            # base             : steer off entirely
            # ours             : full memory (prefix + SWA window + pool)
            # ours_window_only : prefix MASKED from the read (SWA window kept, M_t=0) -- the REAL
            #                    prefix ablation. ours - ours_window_only = PREFIX value.
            # ours_zero_read   : whole memory read := 0. For zero-init delta_o this == base, so
            #                    this is a WHOLE-READ-branch ablation, NOT a prefix ablation.
            #                    (Any past "ours_zp = prefix ablation" claim is really this.)
            set_steer_enabled(model, c != "base")
            # set_steer_enabled() only reaches PrefixMemSteerAttention.  A
            # diff_split checkpoint installs a DIFFERENT module family, which it
            # leaves untouched -- so without this the "base" condition silently
            # re-runs the split model and reports it as the baseline.
            _set_diff_enabled(model, c != "base")
            set_steer_zero_prefix(model, c == "ours_zero_read")
            set_window_only(model, c == "ours_window_only")
            clear_frozen_memory(model)
            if should_run_context_write(
                _cfg, condition=c, write_pass=args.write_pass
            ):
                # two-pass: WRITE from the raw document tokens only, freeze, then generate.
                # NOTE: NOT byte-identical to the training writer input -- training saw
                # "Context:\n"+chunks (Qasper), here ctx_str is HotpotQA "Passage N -...".
                # Cross-domain zero-shot already shifts this; keep ctx_str for consistency
                # with prior numbers. (A stricter re-baseline could prepend "Context:\n".)
                with torch.no_grad():
                    ci = tok(ctx_str, add_special_tokens=False,
                             return_tensors="pt")["input_ids"].to(dev)
                    set_write_freeze(model, True)
                    set_steer_segments(model,
                        torch.full_like(ci, SEG_CTX),
                        torch.ones_like(ci, dtype=torch.bool))
                    model(input_ids=ci, use_cache=False)
                    set_write_freeze(model, False)   # stop writing, KEEP the memory
            pred = gen_ours(model, tok, input_ids, dev, args.max_new_tokens, eos)
            set_window_only(model, False)   # reset so it never leaks into the next condition
            first = extract_first_line(pred)
            f1 = hotpotqa_f1(first, gold); em = float(hotpotqa_exact_match(first, gold))
            res[c]["f1"] += f1; res[c]["em"] += em; res[c]["n"] += 1
            row[c] = first
        recs.append(row)
        if len(recs) % 20 == 0:
            print(f"[shard {args.shard}] {len(recs)}/{len(shard)} "
                  + " ".join(f"{c}F1={res[c]['f1']/res[c]['n']:.3f}" for c in conds)
                  + f" ({(time.time()-t0)/60:.1f}m)", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"shard": args.shard, "res": res, "records": recs}, open(args.output, "w"))
    print(f"[shard {args.shard}] DONE " + " ".join(
        f"{c}: F1={res[c]['f1']/res[c]['n']:.4f} EM={res[c]['em']/res[c]['n']:.4f}" for c in conds), flush=True)


if __name__ == "__main__":
    main()
