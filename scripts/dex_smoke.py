#!/usr/bin/env python
"""Phase-4 smoke test: one forward/backward/optimizer step per DEX variant.

Checks, per variant, on the real backbone:
  * finite loss;
  * gradients exist exactly on the intended parameters and nowhere else;
  * lambda receives gradient when it is trainable;
  * selected heads are modified and non-selected heads are bit-identical;
  * reports trainable names/count, initial loss, grad norm, lambda,
    adapter output norm and ||lambda f_D(O)|| / ||O||.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from deltamem.core.dex import (  # noqa: E402
    VARIANTS,
    AttentionOutputAdapter,
    DexConfig,
    DexOutputProjection,
    attach_dex,
    collect_dex_stats,
    is_dex_param_name,
    load_head_plan,
    set_dex_stats,
    set_dex_step,
    set_trainable,
    trainable_report,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/work/mingze/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--head-plan", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--seq-len", type=int, default=0, help="0 => full example (tail-truncate otherwise)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--anneal-steps", type=int, default=78)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--output", default=str(REPO / "out_dex" / "smoke.json"))
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from qasper_prefix_steer import build_examples, collate, get_dtype

    tok = AutoTokenizer.from_pretrained(args.model_path)
    plan = load_head_plan(args.head_plan)
    examples = build_examples("train", 8, tok, 256, 4500, 24, data="qasper")
    ex = dict(examples[0])
    if args.seq_len > 0:
        # keep the TAIL: the supervised answer tokens sit at the end of the
        # sequence, so a head-truncation would leave only -100 labels and a NaN loss.
        for k in ("ids", "seg", "labels"):
            ex[k] = ex[k][-args.seq_len:]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    results = {}
    for variant in args.variants.split(","):
        variant = variant.strip()
        if not variant:
            continue
        t0 = time.time()
        torch.manual_seed(args.seed)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=get_dtype(args.dtype), attn_implementation=args.attn_impl,
        ).to(args.device)
        model.config.use_cache = False
        cfg = DexConfig(variant=variant, head_selection="entropy_high",
                        lambda_anneal_steps=args.anneal_steps).resolve()
        report = attach_dex(model, cfg, plan=plan)
        names = set_trainable(model, cfg)
        for _, p in model.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()
        tr = trainable_report(model)

        # mid-annealing so lambda(t) > 0 and the adapter is actually active
        set_dex_step(model, max(1, args.anneal_steps // 2))
        set_dex_stats(model, True)

        # capture the first wrapped o_proj's input and the adapter's output
        captured = {}
        first = next(m for m in model.modules() if isinstance(m, DexOutputProjection))

        def hook(mod, inp, out):  # noqa: ANN001
            x = inp[0]
            heads_in = x.view(*x.shape[:-1], mod.num_heads, mod.head_dim).detach().float()
            captured["in"] = heads_in
            if mod.adapter is not None:
                with torch.no_grad():
                    captured["out"] = mod.adapter(
                        x.view(*x.shape[:-1], mod.num_heads, mod.head_dim)
                    ).detach().float()

        h = first.register_forward_hook(hook)

        ids, seg, valid, lab = collate([ex], pad_id, args.device)
        params = [p for p in model.parameters() if p.requires_grad]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, labels=lab, use_cache=False)
        loss = out.loss
        if params:                      # variant 0 (base) trains nothing
            loss.backward()
        h.remove()
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0)) if params else 0.0
        stats = collect_dex_stats(model)

        grad_ok, grad_missing = [], []
        for n, p in model.named_parameters():
            if p.requires_grad:
                if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0:
                    grad_missing.append(n)
                else:
                    grad_ok.append(n)
        frozen_with_grad = [n for n, p in model.named_parameters()
                            if not p.requires_grad and p.grad is not None]

        head_check = {}
        if cfg.adapter_enabled and "out" in captured:
            sel = set(report["selected_heads"]["0"])
            changed, unchanged_bad = [], []
            for hh in range(report["num_heads"]):
                same = torch.equal(captured["in"][..., hh, :], captured["out"][..., hh, :])
                if hh in sel and not same:
                    changed.append(hh)
                if hh not in sel and not same:
                    unchanged_bad.append(hh)
            head_check = {
                "selected": sorted(sel),
                "selected_modified": changed,
                "non_selected_modified": unchanged_bad,
                "all_selected_modified": len(changed) == len(sel),
                "non_selected_bit_identical": not unchanged_bad,
            }

        if params:
            opt = torch.optim.AdamW(params, lr=args.lr)
            before = params[0].detach().clone()
            opt.step()
            moved = float((params[0] - before).abs().max().item())
        else:
            moved = 0.0

        lam_vals = [float(m.current_lambda().item())
                    for m in model.modules() if isinstance(m, AttentionOutputAdapter)]
        lam_grad = [float(m.lambda_learn.grad.abs().item())
                    for m in model.modules()
                    if isinstance(m, AttentionOutputAdapter)
                    and isinstance(m.lambda_learn, torch.nn.Parameter)
                    and m.lambda_learn.grad is not None]

        results[variant] = {
            "trainable_param_count": tr["trainable_param_count"],
            "adapter_param_count": tr["adapter_param_count"],
            "attn_param_count": tr["attn_param_count"],
            "trainable_tensor_count": tr["trainable_tensor_count"],
            "trainable_names_sample": names[:6],
            "loss": float(loss.item()),
            "loss_finite": bool(torch.isfinite(loss).item()),
            "grad_norm": gnorm,
            "n_params_with_grad": len(grad_ok),
            "n_trainable_without_grad": len(grad_missing),
            "trainable_without_grad_sample": grad_missing[:6],
            "frozen_params_with_grad": len(frozen_with_grad),
            "lambda_layer0": lam_vals[0] if lam_vals else None,
            "lambda_mean": (sum(lam_vals) / len(lam_vals)) if lam_vals else None,
            "lambda_grad_layers": len(lam_grad),
            "lambda_grad_mean": (sum(lam_grad) / len(lam_grad)) if lam_grad else None,
            "dex_stats": stats,
            "head_check": head_check,
            "first_param_moved_after_step": moved,
            "seconds": round(time.time() - t0, 1),
        }
        print(f"[smoke] {variant}: " + json.dumps(results[variant], indent=2), flush=True)
        del model, out, loss, params
        torch.cuda.empty_cache()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"[smoke] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
