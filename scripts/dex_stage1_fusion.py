#!/usr/bin/env python
"""Stage-1 mechanism test: one frozen SWA control, four ways of fusing it.

The DEX study showed that ``O - lambda f_D(O)`` cannot be a meaningful subtraction,
because ``f_D`` is a free linear map of ``O`` itself and ``(W -> -W)`` makes minus
and plus the same function class.  The nuisance probe then showed the closed-form
``lambda* = Cov/Var`` variant is unmotivated *with that control*: even with the
branch forced on, AntiAlign stayed at ~1e-3, i.e. ``f_D(O)`` does not explain the
nuisance residual of ``O``.

This script replaces the control.  ``C`` is now the output of the SWA/prefix memory
sidecar -- a separate context-aggregation path, not a linear map of ``O`` -- taken
from a trained checkpoint and FROZEN.  Freezing is what makes the sign testable: a
jointly trained control can flip its own output, which would collapse the minus
back into a reparameterisation.

Arms (all from the same checkpoint, same eval protocol):

    base            Y                       sidecar disabled
    fixed_add       Y + g C                 the additive baseline the ckpt was trained as
    fixed_sub       Y - g C                 same magnitude, opposite sign
    learned_diff    Y - lambda C            one scalar per layer, the only thing trained
    variance_diff   Y - lambda* (C - mu_C)  closed-form, no parameters at all

Only ``learned_diff`` trains anything (one scalar per steer layer).  ``variance_diff``
needs a calibration pass to populate mu_Y/mu_C but has no parameters.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from deltamem.core.prefix_steer import (  # noqa: E402
    DIFFERENTIAL_FUSIONS,
    O_FUSION_POSITIONS,
    OUTPUT_FUSIONS,
    PrefixSteerConfig,
    attach_prefix_steer,
    collect_fusion_norms,
    collect_fusion_stats,
    freeze_steer_keep_fusion,
    is_fusion_param_name,
    iter_steer_modules,
    set_collect_fusion_norms,
    set_fusion_calibrating,
    set_steer_enabled,
    set_steer_segments,
)

ARMS = ("base",) + OUTPUT_FUSIONS


def str2bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/work/mingze/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--steer-ckpt", required=True,
                    help="a *_steer.pt written by dex_train_qasper.py --variant swa_steer")
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    # data / eval protocol: identical to the DEX main table
    ap.add_argument("--data", default="qasper")
    ap.add_argument("--train-papers", type=int, default=800)
    ap.add_argument("--val-papers", type=int, default=75)
    ap.add_argument("--max-chunk-tok", type=int, default=256)
    ap.add_argument("--max-ctx-tok", type=int, default=4500)
    ap.add_argument("--max-ans-tok", type=int, default=24)
    ap.add_argument("--train-target-n", type=int, default=935)
    ap.add_argument("--mix-temporal-n", type=int, default=0)
    ap.add_argument("--max-yesno-frac", type=float, default=0.03)
    ap.add_argument("--data-compose-seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--eval-examples", type=int, default=0)
    ap.add_argument("--val-loss-examples", type=int, default=32)
    # fusion knobs
    ap.add_argument("--o-fusion-position", default="ckpt",
                    choices=["ckpt"] + list(O_FUSION_POSITIONS),
                    help="'ckpt' (default) uses the position the sidecar was trained "
                         "with (old checkpoints deserialize to post_o). An explicit "
                         "override is only allowed between pre_o and post_o_projected, "
                         "whose delta_o shapes are identical -- that pair is the strict "
                         "math-vs-implementation control.")
    ap.add_argument("--fusion-lambda-init", type=float, default=0.1)
    ap.add_argument("--fusion-lambda-max", type=float, default=1.0)
    ap.add_argument("--fusion-ema-momentum", type=float, default=0.99)
    ap.add_argument("--calibrate-batches", type=int, default=64,
                    help="forward-only passes that populate mu_Y/mu_C (variance_diff)")
    ap.add_argument("--fusion-steps", type=int, default=0,
                    help="optimizer updates for learned_diff; 0 => evaluate the init")
    ap.add_argument("--fusion-lr", type=float, default=1e-2)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--output-dir", default=str(REPO / "out_dex_fusion"))
    ap.add_argument("--tag", required=True)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from qasper_prefix_steer import build_examples, collate, f1_em, generate, get_dtype

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ckpt = torch.load(args.steer_ckpt, map_location="cpu", weights_only=False)
    saved = ckpt["steer_config"]
    if saved is None:
        raise SystemExit(f"{args.steer_ckpt} carries no steer_config")

    # Rebuild the sidecar EXACTLY as trained, overriding only the fusion.  Any other
    # difference would make the arms incomparable to the checkpoint they come from.
    fusion = "fixed_add" if args.arm == "base" else args.arm
    # Old checkpoints predate o_fusion_position and deserialize to post_o (the exact
    # behavior they were trained with).  An override is only allowed within the
    # {pre_o, post_o_projected} pair: same delta_o shape, mathematically identical
    # for the linear fusions -- the strict math-vs-implementation control.
    saved_pos = saved.get("o_fusion_position", "post_o")
    pos = saved_pos if args.o_fusion_position == "ckpt" else args.o_fusion_position
    zspace = {"pre_o", "post_o_projected"}
    if pos != saved_pos and not (pos in zspace and saved_pos in zspace):
        raise SystemExit(
            f"--o-fusion-position {pos} is shape-incompatible with the checkpoint's "
            f"{saved_pos} delta_o (only pre_o <-> post_o_projected may be swapped)"
        )
    steer_cfg = PrefixSteerConfig(**{
        **{k: (tuple(v) if isinstance(v, list) else v) for k, v in saved.items()},
        "output_fusion": fusion,
        "o_fusion_position": pos,
        "fusion_lambda_init": args.fusion_lambda_init,
        "fusion_lambda_max": args.fusion_lambda_max,
        "fusion_ema_momentum": args.fusion_ema_momentum,
    })

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=get_dtype(args.dtype), attn_implementation=args.attn_impl,
    ).to(args.device)
    model.config.use_cache = False
    attach_prefix_steer(model, steer_cfg)

    state = {k: v.to(args.device) for k, v in ckpt["state"].items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded = [k for k in state if k not in set(unexpected)]
    if not loaded:
        raise SystemExit("no steer tensor from the checkpoint matched the model")
    if unexpected:
        raise SystemExit(f"checkpoint tensors did not match the rebuilt sidecar: "
                         f"{unexpected[:5]}")
    # The memory path is frozen; only a learned_diff lambda may train.
    trainable = freeze_steer_keep_fusion(model)
    if args.arm != "learned_diff" and trainable:
        raise SystemExit(f"arm {args.arm} must have nothing to train, got {trainable}")

    if args.arm == "base":
        set_steer_enabled(model, False)          # frozen backbone, control switched off

    print(f"[{args.tag}] arm={args.arm} fusion={fusion}@{pos} loaded={len(loaded)} "
          f"tensors trainable={len(trainable)} "
          f"layers={len(list(iter_steer_modules(model)))}", flush=True)

    train = build_examples(
        "train", args.train_papers, tok, args.max_chunk_tok, args.max_ctx_tok,
        args.max_ans_tok, data=args.data, max_yesno_frac=args.max_yesno_frac,
        yesno_seed=args.data_compose_seed, train_target_n=args.train_target_n,
        mix_temporal_n=args.mix_temporal_n,
    )
    val = build_examples(
        "validation", args.val_papers, tok, args.max_chunk_tok, args.max_ctx_tok,
        args.max_ans_tok, data=args.data,
    )
    eval_set = val if args.eval_examples <= 0 else val[: args.eval_examples]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos = tok.eos_token_id
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731
    rng = random.Random(args.seed)

    def feed(seg, valid):
        set_steer_segments(model, seg, valid)

    # ---- calibration: populate mu_Y / mu_C on TRAIN data, then freeze ----
    calib = {}
    if args.arm == "variance_diff":
        model.eval()
        set_fusion_calibrating(model, True)
        order = list(range(len(train)))
        rng.shuffle(order)
        with torch.no_grad(), amp():
            for i in range(min(args.calibrate_batches, len(order))):
                ids, seg, valid, lab = collate([train[order[i]]], pad_id, args.device)
                feed(seg, valid)
                model(input_ids=ids, use_cache=False)
        set_fusion_calibrating(model, False)
        calib = collect_fusion_stats(model)
        print(f"[{args.tag}] calibrated on {args.calibrate_batches} batches: {calib}",
              flush=True)

    # ---- optional: train ONLY the fusion lambda --------------------------
    hist = []
    if args.arm == "learned_diff" and args.fusion_steps > 0:
        params = [p for n, p in model.named_parameters() if is_fusion_param_name(n)]
        opt = torch.optim.AdamW(params, lr=args.fusion_lr)
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        cursor = 0
        for step in range(args.fusion_steps):
            opt.zero_grad(set_to_none=True)
            for _ in range(args.grad_accum):
                if cursor >= len(order):
                    rng.shuffle(order)
                    cursor = 0
                ids, seg, valid, lab = collate([train[order[cursor]]], pad_id, args.device)
                cursor += 1
                feed(seg, valid)
                with amp():
                    out = model(input_ids=ids, labels=lab, use_cache=False)
                (out.loss / args.grad_accum).backward()
            opt.step()
            if step % 4 == 0 or step == args.fusion_steps - 1:
                st = collect_fusion_stats(model)
                hist.append({"step": step + 1, "loss": float(out.loss), **st})
                print(f"[{args.tag}] step {step + 1}/{args.fusion_steps} "
                      f"loss {float(out.loss):.4f} lambda {st.get('lambda', float('nan')):+.4f}",
                      flush=True)
        model.eval()

    # ---- evaluation ------------------------------------------------------
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad(), amp():
        for ex in val[: args.val_loss_examples]:
            ids, seg, valid, lab = collate([ex], pad_id, args.device)
            feed(seg, valid)
            tot += float(model(input_ids=ids, labels=lab, use_cache=False).loss)
            n += 1
    v_loss = tot / max(n, 1)

    f1s, ems, per_example = [], [], []
    with amp():
        for i, ex in enumerate(eval_set):
            pred = generate(model, tok, ex, args.device, args.max_new_tokens, eos)
            f, e = f1_em(pred, ex["answer"])
            f1s.append(f)
            ems.append(e)
            per_example.append({"i": i, "f1": f, "em": e, "pred": pred[:200]})

    # branch strength on a fixed example, so |delta|/|Y| is comparable across arms
    set_collect_fusion_norms(model, True)
    with torch.no_grad(), amp():
        ids, seg, valid, lab = collate([val[0]], pad_id, args.device)
        feed(seg, valid)
        model(input_ids=ids, use_cache=False)
    final_stats = collect_fusion_stats(model)
    final_norms = collect_fusion_norms(model)
    set_collect_fusion_norms(model, False)
    # per-layer learned lambdas (learned_diff only): init, final, zero crossings
    lambdas = [float(m.fusion_lambda.detach())
               for m in iter_steer_modules(model)
               if getattr(m, "fusion_lambda", None) is not None]
    lambda_report = {}
    if lambdas:
        lambda_report = {
            "init": args.fusion_lambda_init,
            "per_layer": lambdas,
            "mean": sum(lambdas) / len(lambdas),
            "min": min(lambdas), "max": max(lambdas),
            "n_negative": sum(l < 0 for l in lambdas),
            "crossed_zero": any(
                (l < 0) != (args.fusion_lambda_init < 0) for l in lambdas
            ),
        }

    payload = {
        "tag": args.tag,
        "arm": args.arm,
        "fusion": fusion,
        "o_fusion_position": pos,
        "o_fusion_position_ckpt": saved_pos,
        "lambda_report": lambda_report,
        "final_fusion_norms": final_norms,
        "steer_ckpt": str(args.steer_ckpt),
        "steer_config": {k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in vars(steer_cfg).items()},
        "args": vars(args),
        "loaded_tensors": len(loaded),
        "trainable_names": trainable,
        "calibration": calib,
        "fusion_history": hist,
        "final_fusion_stats": final_stats,
        "final": {
            "val_loss": v_loss,
            "qa": {"F1": round(sum(f1s) / len(f1s), 4),
                   "EM": round(sum(ems) / len(ems), 4),
                   "n": len(f1s), "per_example": per_example},
        },
        "env": {
            "python": platform.python_version(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "command": " ".join(sys.argv),
        },
        "runtime_min": round((time.time() - t_start) / 60, 2),
    }
    with open(out_dir / f"{args.tag}.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[{args.tag}] FINAL arm={args.arm}@{pos} F1={payload['final']['qa']['F1']} "
          f"val_loss={v_loss:.4f} stats={final_stats} "
          f"lambda={lambda_report.get('mean')} "
          f"norms={final_norms.get('mean', {})} "
          f"({payload['runtime_min']:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
