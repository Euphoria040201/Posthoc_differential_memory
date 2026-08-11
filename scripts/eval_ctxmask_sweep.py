"""Ratio-sweep diagnostic for ctxmask ckpts: base / window_only / method at several eval
mask ratios on the Qasper val set. Protocol identical to evaluate_ctxmask in
qasper_prefix_steer.py (per-doc keyed masks via stable_mask_seed(ctx_ids, seed, -1); ONE
masked ctx shared by all conditions). Answers WHERE the written prefix matters:
method - window_only per ratio is the prefix's marginal value when X% of the doc is gone."""
import argparse, json, sys, random

sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qasper_prefix_steer import build_examples, mask_context, stable_mask_seed, generate, f1_em
from eval_ours_hotpotqa import load_ours
from deltamem.core.prefix_steer import (
    set_steer_segments, set_steer_enabled, set_steer_zero_prefix,
    set_window_only, set_write_freeze, clear_frozen_memory)
from deltamem.core.global_prefix import SEG_CTX, SEG_QRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--ratios", default="0.75,1.0")
    ap.add_argument("--val-papers", type=int, default=15)
    ap.add_argument("--mask-seed", type=int, default=1,
                    help="must match the ckpt's training mask_seed so eval masks line up")
    ap.add_argument("--mask-mode", default="chunk")
    ap.add_argument("--mask-block-tokens", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True).to(args.device)
    load_ours(model, args.ckpt)
    model.eval()
    eos = tok.eos_token_id
    val = build_examples("validation", args.val_papers, tok, 256, 4500, 24)
    print(f"[sweep] {args.ckpt}: {len(val)} val queries", flush=True)

    conds = {"base": dict(steer=False, wo=False, write=False),
             "window_only": dict(steer=True, wo=True, write=True),
             "swap": dict(steer=True, wo=False, write=True, swap=True),
             "method": dict(steer=True, wo=False, write=True)}
    out = {}
    for ratio in [float(x) for x in args.ratios.split(",")]:
        acc = {n: [] for n in conds}
        for vi, ex in enumerate(val):
            # swap cond: memory written from a DIFFERENT paper's context (in-domain
            # doc-specificity control; method - swap is the memory CONTENT's value).
            # val is ordered by paper with 2-4 queries each, so a fixed -1/-2 offset can
            # land on the SAME paper -- scan until the document actually differs.
            swap_ctx = None
            for _off in range(1, len(val)):
                cand = val[(vi - _off) % len(val)]
                if cand["ctx_ids"] != ex["ctx_ids"]:
                    swap_ctx = cand["ctx_ids"]
                    break
            assert swap_ctx is not None and swap_ctx != ex["ctx_ids"], \
                "no different-document swap candidate found in val"
            mrng = random.Random(stable_mask_seed(ex["ctx_ids"], args.mask_seed, -1))
            mctx = mask_context(ex, ratio, args.mask_mode, mrng, args.mask_block_tokens)
            q_ids = ex["prompt_ids"][len(ex["ctx_ids"]):]
            pids = mctx + list(q_ids)
            pseg = [SEG_CTX] * len(mctx) + [SEG_QRY] * len(q_ids)
            for name, c in conds.items():
                set_steer_enabled(model, c["steer"]); set_steer_zero_prefix(model, False)
                set_window_only(model, c["wo"]); clear_frozen_memory(model)
                if c["write"]:
                    wsrc = swap_ctx if c.get("swap") else ex["ctx_ids"]
                    cids = torch.tensor([wsrc], device=args.device)
                    set_write_freeze(model, True)
                    set_steer_segments(model, torch.full_like(cids, SEG_CTX),
                                       torch.ones_like(cids, dtype=torch.bool))
                    with torch.no_grad():
                        model(input_ids=cids, use_cache=False)
                    set_write_freeze(model, False)     # stop writing, KEEP the memory
                pred = generate(model, tok, ex, args.device, args.max_new_tokens, eos,
                                prompt=(pids, pseg))
                clear_frozen_memory(model)
                acc[name].append(f1_em(pred, ex["answer"])[0])
        res = {n: round(sum(v) / len(v), 4) for n, v in acc.items()}
        out[str(ratio)] = res
        print(f"[sweep] ratio={ratio}: " + " ".join(f"{n}={v}" for n, v in res.items()), flush=True)
    set_steer_enabled(model, True); set_window_only(model, False)
    json.dump(out, open(args.output, "w"))
    print("[sweep] DONE ->", args.output)


if __name__ == "__main__":
    main()
