#!/usr/bin/env python
"""Arm 4: train the SWA branch to BE a nuisance estimate, then subtract it.

Everything before this arm tested subtraction against a control that was never
trained to be subtractable:

  * DEX's own control was ``f_D(O)``, a free per-head linear map of the tensor
    being corrected -- so minus and plus were the same function class.
  * ``C_SWA`` from an additive run carries information COMPLEMENTARY to Y (that
    is why adding it works, +.049 F1), hence nothing to remove: the closed-form
    coefficient came out at |cov|/var ~ 1e-3 on every seed, and a free lambda
    walked from +0.1 to -0.15 on every seed, i.e. back to addition.
  * Training the same architecture from scratch under a minus sign does not fix
    this either, because ``C_theta = W_C R(X)`` can absorb the sign in ``W_C``.

The only way the subtraction can carry meaning is if C is DEFINED as the thing
that should be removed.  This script does that.

Construction.  A nuisance group is one Qasper (paper, question, answer) rendered
K ways: the SAME context chunks in K different orders.  Evidence content, the
question and the answer are identical across the group; only which filler
surrounds the evidence and how deep it sits change.  Reordering permutes whole
chunk spans, so every variant has an identical token count and the K sequences
are aligned position-by-position -- the group mean is well defined without any
padding or masking.

Objective, per steer layer l:

    r_l   = Y_l - mean_k Y_l                      (group-centred nuisance residual)
    L_nui = || C_l - stopgrad(r_l) ||^2           (C must predict it)
    L_inv = Var_k [ Y_l - lambda C_l ]            (and removing it must flatten Y)

    L = L_QA + beta * L_nui + gamma * L_inv

``L_QA`` keeps the branch from destroying the task; ``L_inv`` is the property we
actually want (a corrected representation that stops moving with nuisance) rather
than the proxy; ``L_nui`` gives C a dense, well-posed target to start from.

If AntiAlign/VRR/NSR still do not improve after this, the differential route is
finished for real: C will have been handed the residual on a plate and still not
have cancelled it.
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
    PrefixSteerConfig,
    attach_prefix_steer,
    collect_fusion_tensors,
    freeze_backbone_keep_steer,
    is_steer_param_name,
    set_collect_fusion_tensors,
    set_steer_segments,
)


def str2bool(v: str) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "t"}


def build_nuisance_group(ex, k: int, rng: random.Random):
    """K permutations of one example's context chunks; identical token count.

    Returns a list of K dicts shaped like the trainer's collate input.  Variant 0
    is deliberately the ORIGINAL order, so a group always contains the ordering the
    additive runs were trained on.
    """
    spans = ex["ctx_chunk_spans"]
    if not spans or len(spans) < 2:
        return None
    c_ids = ex["ctx_ids"]
    head = c_ids[: spans[0][0]]                       # document header, never moved
    chunks = [c_ids[a:b] for a, b in spans]
    tail_start = spans[-1][1]
    tail = c_ids[tail_start:]                         # anything after the last span

    qa_ids = ex["qa_ids"]
    qa_seg = ex["qa_seg"]
    qa_labels = ex["qa_labels"]

    variants = []
    for ki in range(k):
        idx = list(range(len(chunks)))
        if ki > 0:
            rng.shuffle(idx)
        new_ctx = list(head)
        for j in idx:
            new_ctx.extend(chunks[j])
        new_ctx.extend(tail)
        assert len(new_ctx) == len(c_ids), "reordering must preserve token count"
        from deltamem.core.global_prefix import SEG_CTX
        variants.append({
            "ids": new_ctx + qa_ids,
            "seg": [SEG_CTX] * len(new_ctx) + qa_seg,
            "labels": [-100] * len(new_ctx) + qa_labels,
            "order": idx,
        })
    return variants


def stack_group(variants, device):
    """[K, T] tensors; every variant has the same length by construction."""
    lens = {len(v["ids"]) for v in variants}
    if len(lens) != 1:
        raise RuntimeError(f"nuisance variants must align, got lengths {lens}")
    ids = torch.tensor([v["ids"] for v in variants], device=device)
    seg = torch.tensor([v["seg"] for v in variants], device=device)
    lab = torch.tensor([v["labels"] for v in variants], device=device)
    valid = torch.ones_like(ids, dtype=torch.bool)
    return ids, seg, valid, lab


def nuisance_losses(model, lam: float):
    """L_nui and L_inv summed over steer layers, from the last forward's (Y, C).

    Y is detached inside the target (stopgrad) but NOT inside L_inv: flattening the
    corrected representation is a property of C, and C is the only trainable thing
    here, so the gradient path through C is the one that matters.
    """
    rows = collect_fusion_tensors(model)
    if not rows:
        raise RuntimeError("no fusion tensors captured; set_collect_fusion_tensors first")
    l_nui = 0.0
    l_inv = 0.0
    for _, y, c in rows:
        y32, c32 = y.float(), c.float()
        r = y32 - y32.mean(dim=0, keepdim=True)           # group-centred, over K
        l_nui = l_nui + (c32 - r.detach()).pow(2).mean()
        corrected = y32.detach() - lam * c32
        l_inv = l_inv + corrected.var(dim=0, unbiased=False).mean()
    n = len(rows)
    return l_nui / n, l_inv / n, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/work/mingze/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    # data (identical composition to every other arm)
    ap.add_argument("--data", default="qasper")
    ap.add_argument("--train-papers", type=int, default=800)
    ap.add_argument("--val-papers", type=int, default=75)
    ap.add_argument("--max-chunk-tok", type=int, default=256)
    ap.add_argument("--max-ctx-tok", type=int, default=4500)
    ap.add_argument("--max-ans-tok", type=int, default=24)
    ap.add_argument("--train-target-n", type=int, default=935)
    ap.add_argument("--max-yesno-frac", type=float, default=0.03)
    ap.add_argument("--data-compose-seed", type=int, default=42)
    # nuisance objective
    ap.add_argument("--group-k", type=int, default=4, help="variants per nuisance group")
    ap.add_argument("--beta", type=float, default=1.0, help="weight on L_nuisance")
    ap.add_argument("--gamma", type=float, default=1.0, help="weight on L_invariance")
    ap.add_argument("--qa-weight", type=float, default=1.0)
    ap.add_argument("--fusion-lambda", type=float, default=0.1,
                    help="fixed lambda; the sign question is settled, this arm asks "
                         "whether a nuisance-trained C is subtractable at all")
    # optimisation
    ap.add_argument("--steps", type=int, default=156)
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="groups per update; each group is K forwards")
    ap.add_argument("--steer-lr", type=float, default=5e-4)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--grad-checkpointing", type=str2bool, default=True)
    # sidecar (same as the additive runs)
    ap.add_argument("--steer-layers", default="0,3,6,9,12,15,18,21,24,27,30,33")
    ap.add_argument("--steer-gain", type=float, default=0.1)
    ap.add_argument("--steer-window", type=int, default=256)
    ap.add_argument("--steer-mem-heads", type=int, default=1)
    ap.add_argument("--steer-mem-head-dim", type=int, default=128)
    # eval
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--eval-examples", type=int, default=0)
    ap.add_argument("--val-loss-examples", type=int, default=32)
    ap.add_argument("--log-every", type=int, default=4)
    ap.add_argument("--output-dir", default=str(REPO / "out_dex_fusion"))
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from qasper_prefix_steer import build_examples, collate, f1_em, generate, get_dtype

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=get_dtype(args.dtype), attn_implementation=args.attn_impl,
    ).to(args.device)
    model.config.use_cache = False
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    steer_cfg = PrefixSteerConfig(
        num_prefix_tokens=0, sliding_window_size=args.steer_window,
        mem_num_heads=args.steer_mem_heads, mem_head_dim=args.steer_mem_head_dim,
        steer_mode="deltamem", memory_mode="dynamic", memory_value_source="main_v",
        delta_heads="o", steer_gain=args.steer_gain,
        # the branch is applied subtractively at a FIXED lambda: this arm is about
        # what C is trained to be, not about who picks the coefficient
        output_fusion="fixed_sub",
        steer_layers=tuple(int(x) for x in args.steer_layers.split(",") if x.strip()),
        prefix_write=False, write_ctx_only=False, read_prefix_only=False,
        pool_reads=False, pool_gate=False,
    )
    attach_prefix_steer(model, steer_cfg)
    freeze_backbone_keep_steer(model)
    for _, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{args.tag}] trainable={n_tr:,} lambda={args.fusion_lambda} "
          f"K={args.group_k} beta={args.beta} gamma={args.gamma}", flush=True)

    train = build_examples(
        "train", args.train_papers, tok, args.max_chunk_tok, args.max_ctx_tok,
        args.max_ans_tok, data=args.data, max_yesno_frac=args.max_yesno_frac,
        yesno_seed=args.data_compose_seed, train_target_n=args.train_target_n,
    )
    val = build_examples(
        "validation", args.val_papers, tok, args.max_chunk_tok, args.max_ctx_tok,
        args.max_ans_tok, data=args.data,
    )
    groupable = [ex for ex in train if ex.get("ctx_chunk_spans")
                 and len(ex["ctx_chunk_spans"]) >= 2]
    print(f"[{args.tag}] train={len(train)} groupable={len(groupable)} val={len(val)}",
          flush=True)
    if not groupable:
        raise SystemExit("no example has >=2 context chunks; cannot form nuisance groups")

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos = tok.eos_token_id
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.steer_lr)

    order = list(range(len(groupable)))
    rng.shuffle(order)
    cursor = 0
    log = []
    model.train()
    t0 = time.time()

    for update in range(args.steps):
        opt.zero_grad(set_to_none=True)
        acc = {"qa": 0.0, "nui": 0.0, "inv": 0.0}
        for _ in range(args.grad_accum):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            ex = groupable[order[cursor]]
            cursor += 1
            variants = build_nuisance_group(ex, args.group_k, rng)
            if variants is None:
                continue
            ids, seg, valid, lab = stack_group(variants, args.device)
            set_steer_segments(model, seg, valid)
            set_collect_fusion_tensors(model, True)
            with amp():
                out = model(input_ids=ids, labels=lab, use_cache=False)
            l_nui, l_inv, n_layers = nuisance_losses(model, args.fusion_lambda)
            set_collect_fusion_tensors(model, False)
            loss = (args.qa_weight * out.loss
                    + args.beta * l_nui
                    + args.gamma * l_inv) / args.grad_accum
            loss.backward()
            acc["qa"] += float(out.loss)
            acc["nui"] += float(l_nui)
            acc["inv"] += float(l_inv)
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip))
        opt.step()
        if update % args.log_every == 0 or update == args.steps - 1:
            row = {"step": update + 1, "grad_norm": gnorm,
                   **{k: v / args.grad_accum for k, v in acc.items()},
                   "elapsed_min": round((time.time() - t0) / 60, 2)}
            log.append(row)
            print(f"[{args.tag}] step {update + 1}/{args.steps} qa {row['qa']:.4f} "
                  f"nui {row['nui']:.5f} inv {row['inv']:.5f} gnorm {gnorm:.3f} "
                  f"({row['elapsed_min']:.1f}m)", flush=True)

    # ---- eval, same protocol as every other arm --------------------------
    model.eval()
    eval_set = val if args.eval_examples <= 0 else val[: args.eval_examples]
    tot, n = 0.0, 0
    with torch.no_grad(), amp():
        for ex in val[: args.val_loss_examples]:
            ids, seg, valid, lab = collate([ex], pad_id, args.device)
            set_steer_segments(model, seg, valid)
            tot += float(model(input_ids=ids, labels=lab, use_cache=False).loss)
            n += 1
    v_loss = tot / max(n, 1)
    f1s, ems, per_example = [], [], []
    with amp():
        for i, ex in enumerate(eval_set):
            pred = generate(model, tok, ex, args.device, args.max_new_tokens, eos)
            f, e = f1_em(pred, ex["answer"])
            f1s.append(f); ems.append(e)
            per_example.append({"i": i, "f1": f, "em": e, "pred": pred[:200]})

    payload = {
        "tag": args.tag, "arm": "nuisance_subtractive", "args": vars(args),
        "steer_config": {k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in vars(steer_cfg).items()},
        "trainable_param_count": n_tr, "log": log,
        "final": {"val_loss": v_loss,
                  "qa": {"F1": round(sum(f1s) / len(f1s), 4),
                         "EM": round(sum(ems) / len(ems), 4),
                         "n": len(f1s), "per_example": per_example}},
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "command": " ".join(sys.argv)},
        "runtime_min": round((time.time() - t_start) / 60, 2),
    }
    with open(out_dir / f"{args.tag}.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    steer_state = {n_: p.detach().to(torch.bfloat16).cpu()
                   for n_, p in model.named_parameters() if is_steer_param_name(n_)}
    torch.save({"state": steer_state, "cfg": payload["steer_config"],
                "steer_config": payload["steer_config"], "args": vars(args)},
               out_dir / f"{args.tag}_steer.pt")
    print(f"[{args.tag}] FINAL F1={payload['final']['qa']['F1']} val_loss={v_loss:.4f} "
          f"({payload['runtime_min']:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
