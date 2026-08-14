#!/usr/bin/env python
"""Controlled DEX experiment on the repo's Qasper adaptation setup.

Every variant of arXiv:2505.16333 and its controls runs through this single
script; only ``--variant`` (and the seed) changes between runs.  See
``deltamem/core/dex.py`` for the variant table and ``dex_control_report.md`` for
the experimental design.

The backbone, data, tokenisation, optimiser, schedule, step budget and
evaluation protocol are shared with the repo's existing Qasper pipeline
(``scripts/qasper_prefix_steer.py``), so the DEX numbers sit on the same scale as
the memory-sidecar numbers already in ``out_swa_sharedv/``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from deltamem.core.prefix_steer import OUTPUT_FUSIONS as _OUTPUT_FUSIONS  # noqa: E402
from deltamem.core.prefix_steer import O_FUSION_POSITIONS as _O_FUSION_POSITIONS  # noqa: E402
from deltamem.core.dex import (  # noqa: E402
    VARIANTS,
    DexConfig,
    attach_dex,
    collect_dex_stats,
    load_head_plan,
    set_dex_stats,
    set_dex_step,
    set_trainable,
    trainable_report,
)


def str2bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    # --- backbone / data (kept identical to the repo's Qasper runs) ------
    ap.add_argument("--model-path", default="/work/mingze/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--device", default="cuda")
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
    # --- variant ---------------------------------------------------------
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--head-plan", default="")
    ap.add_argument("--head-selection", default="entropy_high")
    ap.add_argument("--heads-per-layer", type=int, default=-1)
    ap.add_argument("--lambda-init-mode", default="diff_depth", choices=["diff_depth", "fixed"])
    ap.add_argument("--lambda-init", type=float, default=0.8)
    ap.add_argument("--lambda-learn-init", type=float, default=0.0)
    ap.add_argument("--lambda-learnable", type=str2bool, default=True)
    ap.add_argument("--lambda-anneal-steps", type=int, default=-1,
                    help="T in Eq. (4); -1 => 0.5 * --steps, 0 disables annealing")
    ap.add_argument("--allow-no-anneal", type=str2bool, default=False,
                    help="explicitly run a paper-style variant without Eq. (4) annealing")
    ap.add_argument("--fd-init", default="linear_default", choices=["linear_default", "zeros"])
    ap.add_argument("--fd-bias", type=str2bool, default=False)
    ap.add_argument("--negate-fd-init", type=str2bool, default=False,
                    help="diagnostic: initialise W_D as -W_D (paired minus/plus check)")
    # --- optimisation (shared by every variant) --------------------------
    ap.add_argument("--steps", type=int, default=156, help="optimizer updates")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--adapter-lr", type=float, default=-1.0,
                    help="-1 => same as --lr (paper uses one LR for all trained params)")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    # --- swa_steer sidecar ------------------------------------------------
    # Defaults reproduce the winning configuration of this repo's own SWA line
    # (out_swa_sharedv/p0_mainv_h1_fixed_g01_s1: P=0 window-only memory, frozen
    # main V reused as the memory value, output-side steer only, gain 0.1, every
    # third layer).  That run scored .2968 on the same 187-example Qasper val
    # protocol at the same 156 updates, against attn_only's .2933 -- reproducing
    # it inside this harness is the point of the variant.
    ap.add_argument("--steer-lr", type=float, default=5e-4,
                    help="LR for the sidecar. The backbone LR (2e-5) was tuned for "
                         "pretrained weights; the sidecar is randomly initialised and "
                         "the SWA line tuned 5e-4 for it.")
    ap.add_argument("--steer-lr-schedule", default="constant",
                    choices=["constant", "cosine"],
                    help="'constant' matches scripts/qasper_prefix_steer.py, which never "
                         "modifies the LR after creating the optimiser. DEX's cosine-to-0 "
                         "(+3%% warmup) is tuned for finetuning PRETRAINED weights; applying "
                         "it to a randomly initialised sidecar wastes the tail of the run "
                         "(lr is 2.7%% of peak by update 141) and measured -0.022 F1.")
    ap.add_argument("--steer-layers", default="0,3,6,9,12,15,18,21,24,27,30,33",
                    help="comma-separated layer indices carrying the sidecar")
    ap.add_argument("--steer-gain", type=float, default=0.1)
    ap.add_argument("--steer-delta-heads", default="o",
                    help="which of q/k/v/o receive the additive steer")
    ap.add_argument("--steer-output-fusion", default="fixed",
                    choices=list(_OUTPUT_FUSIONS),
                    help="differential modes train the sidecar UNDER subtraction from "
                         "step 0, which is a different experiment from reusing an "
                         "additive-trained checkpoint subtractively")
    ap.add_argument("--steer-o-fusion-position", default="post_o",
                    choices=list(_O_FUSION_POSITIONS),
                    help="post_o: Y = fuse(W_O Z, C) (historical). pre_o: Y = W_O fuse(Z, C) "
                         "-- delta_o then lives in the o_proj INPUT space, so its "
                         "checkpoints are not shape-compatible with post_o ones. "
                         "post_o_projected: Y = fuse(W_O Z, W_O C), the strict pre_o "
                         "control (identical for linear fusions).")
    ap.add_argument("--diff-read-dim", type=int, default=128,
                    help="local-reader width; 128 reproduces the old sidecar's exact "
                         "14,155,776 parameter budget on Qwen3-4B")
    ap.add_argument("--diff-window", type=int, default=256,
                    help="causal local window w for the reader")
    ap.add_argument("--diff-gamma", type=float, default=1.0,
                    help="fixed gamma in O~ = O+ + gamma*(O+ - O-)")
    ap.add_argument("--diff-dynamic-gate", type=str2bool, default=False,
                    help="Stage B only: gamma_{t,h} = 2*sigmoid(gate(R_t))")
    ap.add_argument("--steer-value-source", default="main_v",
                    choices=["trainable", "main_v"])
    ap.add_argument("--steer-window", type=int, default=256)
    ap.add_argument("--steer-mem-heads", type=int, default=1)
    ap.add_argument("--steer-mem-head-dim", type=int, default=128)
    ap.add_argument("--steer-prefix-tokens", type=int, default=0,
                    help="0 => window-only memory (no written prefix slots)")
    ap.add_argument("--grad-checkpointing", type=str2bool, default=True,
                    help="exact recompute; identical maths, ~40%% less activation memory")
    ap.add_argument("--seed", type=int, default=0)
    # --- evaluation ------------------------------------------------------
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--eval-examples", type=int, default=0, help="0 => all val examples")
    ap.add_argument("--val-loss-examples", type=int, default=32)
    ap.add_argument("--val-every", type=int, default=26)
    ap.add_argument("--log-every", type=int, default=4)
    ap.add_argument("--eval-at-start", type=str2bool, default=False)
    ap.add_argument("--skip-final-eval", type=str2bool, default=False)
    # --- io --------------------------------------------------------------
    ap.add_argument("--output-dir", default=str(REPO / "out_dex"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--save-adapter", type=str2bool, default=True)
    ap.add_argument("--save-attn", type=str2bool, default=False,
                    help="also checkpoint the trained W_K/W_V/W_O (bf16, ~1.1GB) so the "
                         "run can be reloaded for diagnostics without retraining")
    return ap


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "not-a-git-repo"


def lr_at(update: int, total: int, warmup: int, peak: float) -> float:
    """Linear warmup then cosine decay to 0 (App. E.1: cosine, warmup 0.03)."""
    if update < warmup:
        return peak * (update + 1) / max(warmup, 1)
    progress = (update - warmup) / max(total - warmup, 1)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main() -> None:
    args = build_parser().parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from qasper_prefix_steer import build_examples, collate, f1_em, generate, get_dtype

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    if args.lambda_anneal_steps < 0:
        anneal_steps = max(1, args.steps // 2)
    else:
        anneal_steps = args.lambda_anneal_steps

    cfg = DexConfig(
        variant=args.variant,
        head_selection=args.head_selection,
        heads_per_layer=args.heads_per_layer,
        head_plan_path=args.head_plan,
        fd_bias=args.fd_bias,
        fd_init=args.fd_init,
        lambda_init_mode=args.lambda_init_mode,
        lambda_init=args.lambda_init,
        lambda_learn_init=args.lambda_learn_init,
        lambda_learnable=args.lambda_learnable,
        lambda_anneal_steps=anneal_steps,
        allow_no_anneal=args.allow_no_anneal,
    ).resolve()

    # ---- determinism ----------------------------------------------------
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    data_rng = random.Random(args.seed)          # data order: seed only, never variant

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=get_dtype(args.dtype), attn_implementation=args.attn_impl,
    ).to(args.device)
    model.config.use_cache = False

    if args.grad_checkpointing:
        # use_reentrant=False so parameters inside a checkpointed block still get
        # gradients when the block's *input* does not require grad (frozen embeddings).
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # The sidecar is attached BEFORE the DEX wrapper on purpose.  Attaching it
    # runs `.to(dtype=q_proj.dtype)` over the whole wrapped attention, which
    # would demote an already-installed DEX lambda from fp32 to bf16 (the bug
    # recorded as review finding #6).  In this order the DEX wrapper is applied
    # to the inner Qwen3Attention afterwards and keeps its own dtypes; the
    # sidecar's forward reaches it through `base.o_proj`, which DexOutputProjection
    # duck-types as an nn.Linear.
    steer_cfg = None
    if cfg.train_steer:
        from deltamem.core.prefix_steer import PrefixSteerConfig, attach_prefix_steer
        steer_layers = tuple(int(x) for x in args.steer_layers.split(",") if x.strip())
        steer_cfg = PrefixSteerConfig(
            num_prefix_tokens=args.steer_prefix_tokens,
            sliding_window_size=args.steer_window,
            mem_num_heads=args.steer_mem_heads,
            mem_head_dim=args.steer_mem_head_dim,
            steer_mode="deltamem",
            memory_mode="dynamic",
            memory_value_source=args.steer_value_source,
            delta_heads=args.steer_delta_heads,
            steer_gain=args.steer_gain,
            output_fusion=args.steer_output_fusion,
            o_fusion_position=args.steer_o_fusion_position,
            steer_layers=steer_layers,
            # P=0 is the window-only branch: nothing is written, so the
            # context-only WRITE pass and the pooled read are both off.
            prefix_write=args.steer_prefix_tokens > 0,
            write_ctx_only=args.steer_prefix_tokens > 0,
            read_prefix_only=False,
            pool_reads=False,
            pool_gate=False,
        )
        patched = attach_prefix_steer(model, steer_cfg)
        print(f"[{args.tag}] steer sidecar on {len(patched)} layers "
              f"(delta_heads={args.steer_delta_heads} gain={args.steer_gain} "
              f"P={args.steer_prefix_tokens} value={args.steer_value_source} "
              f"fusion={args.steer_output_fusion}@{args.steer_o_fusion_position})",
              flush=True)

    diff_cfg = None
    if args.variant == "diff_split":
        from deltamem.core.diff_split import (
            attach_diff_split, freeze_backbone_keep_diff, is_diff_param_name)
        diff_layers = tuple(int(x) for x in args.steer_layers.split(",") if x.strip())
        diff_cfg = {"layers": list(diff_layers), "read_dim": args.diff_read_dim,
                    "window": args.diff_window, "gamma": args.diff_gamma,
                    "dynamic_gate": args.diff_dynamic_gate}
        patched = attach_diff_split(model, diff_layers, read_dim=args.diff_read_dim,
                                    window=args.diff_window, gamma=args.diff_gamma,
                                    dynamic_gate=args.diff_dynamic_gate)
        freeze_backbone_keep_diff(model)
        n_tr = sum(p.numel() for n, p in model.named_parameters() if is_diff_param_name(n))
        print(f"[{args.tag}] diff-split on {len(patched)} layers "
              f"(read_dim={args.diff_read_dim} w={args.diff_window} "
              f"gamma={args.diff_gamma} dyn={args.diff_dynamic_gate}) "
              f"trainable={n_tr:,}", flush=True)

    plan = load_head_plan(args.head_plan) if args.head_plan else None
    report = attach_dex(model, cfg, plan=plan)
    trainable_names = set_trainable(model, cfg)
    if args.variant == "diff_split":
        # set_trainable() only knows about DEX adapter / attention params and would
        # leave EVERYTHING frozen for this variant, silently producing a run whose
        # delta_q never leaves zero (i.e. a base-parity run reported as a result).
        # Re-apply the split's own freeze AFTER it and assert the result.
        from deltamem.core.diff_split import (
            freeze_backbone_keep_diff as _fbkd, is_diff_param_name as _isd)
        trainable_names = _fbkd(model)
        n_tr = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
        assert trainable_names and all(_isd(n) for n in trainable_names), trainable_names[:5]
        assert n_tr == args.diff_read_dim * (2 * model.config.hidden_size
                                             + model.config.num_attention_heads
                                             * getattr(model.config, "head_dim",
                                                       model.config.hidden_size
                                                       // model.config.num_attention_heads)
                                             ) * len(diff_cfg["layers"]), n_tr
        print(f"[{args.tag}] diff-split trainable after set_trainable: {n_tr:,} "
              f"in {len(trainable_names)} tensors", flush=True)

    if args.negate_fd_init:
        from deltamem.core.dex import AttentionOutputAdapter
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, AttentionOutputAdapter):
                    m.proj.weight.mul_(-1.0)

    # fp32 master weights for everything that is trained: bf16 cannot absorb
    # updates of relative size ~1e-4 (bf16 eps ~ 4e-3), so a pure-bf16 optimiser
    # would silently no-op the attention finetuning.
    for _, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()

    tr = trainable_report(model)
    print(f"[{args.tag}] variant={cfg.variant} sign={cfg.sign} adapter={cfg.adapter_enabled} "
          f"train_attn={cfg.train_attn} layers={len(report['layers'])} "
          f"heads/layer={len(report['selected_heads'].get('0', []))} "
          f"trainable={tr['trainable_param_count']:,} "
          f"(adapter={tr['adapter_param_count']:,}, attn={tr['attn_param_count']:,}"
          f", steer={tr['steer_param_count']:,}) "
          f"anneal_T={anneal_steps} lr={args.lr}"
          f"{f' steer_lr={args.steer_lr}' if cfg.train_steer else ''} "
          f"seed={args.seed}", flush=True)

    # ---- data (identical across variants and seeds-per-order) -----------
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
    print(f"[{args.tag}] train={len(train)} val={len(val)}", flush=True)
    val_loss_set = val[: args.val_loss_examples]
    eval_set = val if args.eval_examples <= 0 else val[: args.eval_examples]

    params = [p for p in model.parameters() if p.requires_grad]
    groups = []
    if params:
        from deltamem.core.dex import is_dex_param_name
        from deltamem.core.prefix_steer import is_steer_param_name
        adapter_params = [p for n, p in model.named_parameters()
                          if p.requires_grad and is_dex_param_name(n)]
        from deltamem.core.diff_split import is_diff_param_name as _isdiff
        steer_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and (is_steer_param_name(n) or _isdiff(n))]
        attn_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not is_dex_param_name(n)
                       and not is_steer_param_name(n) and not _isdiff(n)]
        a_lr = args.lr if args.adapter_lr < 0 else args.adapter_lr
        if attn_params:
            groups.append({"params": attn_params, "lr": args.lr, "peak_lr": args.lr})
        if adapter_params:
            groups.append({"params": adapter_params, "lr": a_lr, "peak_lr": a_lr})
        if steer_params:
            # randomly initialised sidecar: the backbone's 2e-5 (tuned on pretrained
            # weights) barely moves it; the SWA line tuned 5e-4 on this exact module.
            groups.append({"params": steer_params, "lr": args.steer_lr,
                           "peak_lr": args.steer_lr,
                           "schedule": args.steer_lr_schedule})
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay) if groups else None

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos = tok.eos_token_id
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731

    if cfg.train_steer:
        from deltamem.core.prefix_steer import set_steer_segments

        def feed_segments(seg, valid):
            """The sidecar's window/validity mask must track the current batch.

            ``generate`` (imported from qasper_prefix_steer) already does this per
            decode step, so only the teacher-forced forwards need it here.
            """
            set_steer_segments(model, seg, valid)
    else:
        def feed_segments(seg, valid):
            return None

    def val_loss() -> float:
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad(), amp():
            for ex in val_loss_set:
                ids, seg, valid, lab = collate([ex], pad_id, args.device)
                feed_segments(seg, valid)
                out = model(input_ids=ids, labels=lab, use_cache=False)
                tot += float(out.loss.item())
                n += 1
        model.train()
        return tot / max(n, 1)

    def qa_eval() -> dict:
        """Greedy QA eval; keeps per-example scores so seeds can be paired later."""
        model.eval()
        f1s, ems, per_example = [], [], []
        with amp():
            for i, ex in enumerate(eval_set):
                pred = generate(model, tok, ex, args.device, args.max_new_tokens, eos)
                f, e = f1_em(pred, ex["answer"])
                f1s.append(f)
                ems.append(e)
                per_example.append({"i": i, "f1": f, "em": e,
                                    "pred": pred[:200], "gold": str(ex["answer"])[:200]})
        model.train()
        return {"F1": round(sum(f1s) / len(f1s), 4), "EM": round(sum(ems) / len(ems), 4),
                "n": len(f1s), "per_example": per_example}

    log: list[dict] = []
    warmup = max(1, int(round(args.warmup_ratio * args.steps)))
    total_micro = args.steps * args.grad_accum

    if args.eval_at_start:
        v0 = val_loss()
        print(f"[{args.tag}] step=0 val_loss={v0:.4f} (pre-training)", flush=True)
        log.append({"step": 0, "val_loss": v0})

    model.train()
    order = list(range(len(train)))
    data_rng.shuffle(order)
    cursor = 0
    running, running_n = 0.0, 0
    t0 = time.time()

    for update in range(args.steps):
        if opt is None:
            break
        for g in opt.param_groups:
            # groups without an explicit schedule keep DEX's cosine, so every
            # pre-existing variant is bit-identical to its published run
            g["lr"] = (g["peak_lr"] if g.get("schedule") == "constant"
                       else lr_at(update, args.steps, warmup, g["peak_lr"]))
        set_dex_step(model, update)
        will_log = (update % args.log_every == 0) or (update == args.steps - 1)
        opt.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
            if cursor >= len(order):
                data_rng.shuffle(order)
                cursor = 0
            batch = []
            for _ in range(args.batch_size):
                if cursor >= len(order):
                    data_rng.shuffle(order)
                    cursor = 0
                batch.append(train[order[cursor]])
                cursor += 1
            ids, seg, valid, lab = collate(batch, pad_id, args.device)
            feed_segments(seg, valid)
            set_dex_stats(model, will_log and micro == 0)
            with amp():
                out = model(input_ids=ids, labels=lab, use_cache=False)
            loss = out.loss / args.grad_accum
            loss.backward()
            set_dex_stats(model, False)
            running += float(out.loss.item())
            running_n += 1
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip))
        opt.step()

        if will_log:
            stats = collect_dex_stats(model)
            row = {
                "step": update + 1,
                "train_loss": running / max(running_n, 1),
                "grad_norm": gnorm,
                "lr": opt.param_groups[0]["lr"],
                "elapsed_min": round((time.time() - t0) / 60, 2),
            }
            row.update({f"dex_{k}": v for k, v in stats.items()})
            log.append(row)
            msg = (f"[{args.tag}] step {update + 1}/{args.steps} loss {row['train_loss']:.4f} "
                   f"gnorm {gnorm:.3f} lr {row['lr']:.2e}")
            if stats:
                msg += (f" lambda {stats['lambda']:+.4f} |dO|/|O| {stats['corr_ratio']:.4f}"
                        f" cos {stats['cos_delta_o']:+.3f}")
            print(msg + f" ({row['elapsed_min']:.1f}m)", flush=True)
            running, running_n = 0.0, 0

        if args.val_every and ((update + 1) % args.val_every == 0):
            vl = val_loss()
            log.append({"step": update + 1, "val_loss": vl})
            print(f"[{args.tag}] step {update + 1} val_loss {vl:.4f}", flush=True)

    set_dex_step(model, args.steps)
    final = {}
    if not args.skip_final_eval:
        final["val_loss"] = val_loss()
        final["qa"] = qa_eval()
        print(f"[{args.tag}] FINAL val_loss={final['val_loss']:.4f} "
              f"F1={final['qa']['F1']} EM={final['qa']['EM']} n={final['qa']['n']}", flush=True)

    set_dex_stats(model, True)
    with torch.no_grad(), amp():
        ids, seg, valid, lab = collate([val[0]], pad_id, args.device)
        feed_segments(seg, valid)
        model(input_ids=ids, use_cache=False)
    final_stats = collect_dex_stats(model)
    set_dex_stats(model, False)

    payload = {
        "tag": args.tag,
        "variant": cfg.variant,
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(cfg).items()},
        "args": vars(args),
        "attach_report": report,
        "diff_config": diff_cfg,
        "steer_config": (
            {k: (list(v) if isinstance(v, tuple) else v)
             for k, v in vars(steer_cfg).items()} if steer_cfg is not None else None
        ),
        "trainable": tr,
        "trainable_names_sample": trainable_names[:12],
        "log": log,
        "final": final,
        "final_dex_stats": final_stats,
        "env": {
            "git_hash": git_hash(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "command": " ".join(sys.argv),
        },
        "runtime_min": round((time.time() - t_start) / 60, 2),
    }
    with open(out_dir / f"{args.tag}.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    if args.save_adapter and cfg.adapter_enabled:
        from deltamem.core.dex import is_dex_param_name
        state = {n: p.detach().cpu() for n, p in model.named_parameters()
                 if is_dex_param_name(n)}
        torch.save({"state": state, "config": payload["config"], "args": vars(args)},
                   out_dir / f"{args.tag}_adapter.pt")
    if cfg.train_steer:
        # Always checkpointed: the sidecar is ~12M parameters, so unlike the 566M
        # attention finetune (which report section 11 lists as unrecoverable) this
        # run CAN be reloaded and re-probed without retraining.  Reload order is
        # backbone -> attach_prefix_steer(steer_config) -> load_state_dict(strict=False).
        from deltamem.core.prefix_steer import is_steer_param_name
        steer_state = {n: p.detach().to(torch.bfloat16).cpu()
                       for n, p in model.named_parameters() if is_steer_param_name(n)}
        torch.save({"state": steer_state,
                    # "cfg" is the key the eval ecosystem expects
                    # (deltamem/eval/steer_checkpoint.py, eval_ours_locomo.py,
                    # eval_ours_hotpotqa.py); "steer_config" is kept as the
                    # descriptive alias used by dex_stage1_fusion.py.
                    "cfg": payload["steer_config"],
                    "steer_config": payload["steer_config"],
                    "config": payload["config"], "args": vars(args)},
                   out_dir / f"{args.tag}_steer.pt")
        print(f"[{args.tag}] saved {len(steer_state)} steer tensors", flush=True)
    if args.variant == "diff_split":
        from deltamem.core.diff_split import is_diff_param_name as _isdiff2
        dstate = {n: p.detach().to(torch.bfloat16).cpu()
                  for n, p in model.named_parameters() if _isdiff2(n)}
        torch.save({"state": dstate, "diff_config": diff_cfg,
                    "config": payload["config"], "args": vars(args)},
                   out_dir / f"{args.tag}_diff.pt")
        print(f"[{args.tag}] saved {len(dstate)} diff-split tensors", flush=True)
    if args.save_attn:
        # keys are post-wrap names (``...o_proj.base.weight``); reload order is
        # backbone -> attach_dex(config) -> load_state_dict(strict=False)
        from deltamem.core.prefix_steer import is_steer_param_name as _is_steer
        attn = {n: p.detach().to(torch.bfloat16).cpu()
                for n, p in model.named_parameters()
                if p.requires_grad and not is_dex_param_name(n) and not _is_steer(n)}
        torch.save({"state": attn, "config": payload["config"], "args": vars(args)},
                   out_dir / f"{args.tag}_attn.pt")
        print(f"[{args.tag}] saved {len(attn)} attention tensors", flush=True)
    print(f"[{args.tag}] DONE in {payload['runtime_min']:.1f} min -> {out_dir / (args.tag + '.json')}",
          flush=True)


if __name__ == "__main__":
    main()
