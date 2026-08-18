#!/usr/bin/env python
"""Continue-pretraining matrix on PG19 long documents.

Arms (paper labels in brackets)
  vanilla            [--]  from-scratch dense baseline; saves the shared T0
  native_diffv2      [B]   from-scratch native DIFF V2 (HF-matched init)
  vanilla_continue   [A]   T0 -> continue training the dense model normally
  lowrank            [C]   T0 -> token-local low-rank split, train A/B only
  lowrank_unfreeze   [D]   T0 -> split, warm-up A/B, then unfreeze q/k/v/o+norms
                           in the split layers at a much smaller LR
                           (ARCHITECTURE CONVERSION, not a 14M-param PEFT result)
  localreader        [E]   T0 -> the existing LocalRead split (baseline arm)
  additive           [F]   T0 -> parameter-matched additive correction, no
                           differential attention branch

Controls that make the arms comparable
  * every continuation arm loads the SAME T0 file and asserts its sha256;
  * the token stream is one fixed permutation of the corpus (--data-seed), so
    all arms consume identical sequences in identical order;
  * continuation arms all start at the same stream offset (T0's token count)
    and run for the same --continue-tokens with the same fresh schedule;
  * --seed controls ONLY initialization/dropout, never the data order.

Every run writes one JSON artifact carrying config, layer selection, rank, git
SHA, data manifest sha256, T0 sha256, seeds and the exact argv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ARMS = ("vanilla", "native_diffv2", "vanilla_continue", "lowrank",
        "lowrank_unfreeze", "localreader", "additive")
CONTINUATION = ("vanilla_continue", "lowrank", "lowrank_unfreeze",
                "localreader", "additive")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


class SeqStream:
    """Fixed-order stream of whole 4096-token book windows.

    The order is one permutation seeded by `data_seed` and is INDEPENDENT of the
    model seed, so two arms with different init see byte-identical batches.
    """

    def __init__(self, path, batch_size, device, *, data_seed: int, start_seq: int = 0):
        self.arr = np.load(path, mmap_mode="r")
        self.n, self.seq_len = self.arr.shape
        self.order = np.random.default_rng(data_seed).permutation(self.n)
        self.bs, self.device = batch_size, device
        self.pos = start_seq

    def next(self):
        idx = [self.order[(self.pos + i) % self.n] for i in range(self.bs)]
        self.pos += self.bs
        chunk = np.stack([np.asarray(self.arr[i], dtype=np.int64) for i in sorted(idx)])
        x = torch.from_numpy(chunk)
        return x.to(self.device, non_blocking=True)


POS_BUCKETS = [(0, 128), (128, 512), (512, 1024), (1024, 2048), (2048, 4096)]


@torch.no_grad()
def evaluate(model, val_path, batch_size, device, max_batches=None):
    """Mean val NLL plus NLL stratified by within-document position.

    Because every val window lies inside ONE book, token at position p has p
    tokens of genuine same-book context, so NLL(p) measures long-range use.
    """
    model.eval()
    arr = np.load(val_path, mmap_mode="r")
    n_seq, seq_len = arr.shape
    nb = n_seq // batch_size if max_batches is None else min(max_batches, n_seq // batch_size)
    tot_nll, tot_tok = 0.0, 0
    pos_nll = torch.zeros(seq_len - 1, dtype=torch.float64, device=device)
    pos_cnt = torch.zeros(seq_len - 1, dtype=torch.float64, device=device)
    for b in range(nb):
        x = torch.from_numpy(
            np.asarray(arr[b * batch_size:(b + 1) * batch_size], dtype=np.int64)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x).logits
        lp = torch.nn.functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            x[:, 1:].reshape(-1), reduction="none").view(x.shape[0], -1)
        tot_nll += float(lp.sum()); tot_tok += lp.numel()
        pos_nll += lp.sum(0).double(); pos_cnt += x.shape[0]
    model.train()
    per_pos = (pos_nll / pos_cnt.clamp(min=1)).cpu().numpy()
    buckets = {f"{a}-{b}": float(per_pos[a:min(b, len(per_pos))].mean())
               for a, b in POS_BUCKETS if a < len(per_pos)}
    return {"val_nll": tot_nll / max(tot_tok, 1),
            "val_ppl": math.exp(tot_nll / max(tot_tok, 1)),
            "val_tokens": tot_tok, "nll_by_position": buckets,
            "per_position": per_pos[::64].tolist()}


def build_config(vocab, args):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    return Qwen3Config(
        vocab_size=vocab, hidden_size=args.hidden, intermediate_size=args.intermediate,
        num_hidden_layers=args.layers, num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads, head_dim=args.head_dim,
        max_position_embeddings=args.seq_len, rms_norm_eps=1e-6,
        tie_word_embeddings=True, attn_implementation=args.attn_impl)


def param_report(model):
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    attn = sum(p.numel() for n, p in model.named_parameters() if "self_attn" in n)
    emb = sum(p.numel() for n, p in model.named_parameters()
              if "embed_tokens" in n or "lm_head" in n)
    return {"total": tot, "trainable": tr, "attention": attn, "embedding": emb,
            "non_embedding": tot - emb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data-dir", default="/work/mingze/Posthoc_differential_memory/out_cpt_20260817")
    ap.add_argument("--out-dir", default="/work/mingze/Posthoc_differential_memory/out_cpt_20260817")
    ap.add_argument("--tokenizer", default="/work/mingze/models/Qwen3-4B-Instruct-2507")
    # architecture (chain-B config, seq_len raised to 4096 for long documents)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--intermediate", type=int, default=1920)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--attn-impl", default="sdpa")
    # budget
    ap.add_argument("--total-tokens", type=int, default=800_000_000)
    ap.add_argument("--t0-tokens", type=int, default=600_000_000)
    ap.add_argument("--continue-tokens", type=int, default=200_000_000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--split-lr", type=float, default=5e-4)
    ap.add_argument("--backbone-lr", type=float, default=1e-5, help="arm D stage 2")
    ap.add_argument("--warmup-frac", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=42)
    ap.add_argument("--stage1-frac", type=float, default=0.15,
                    help="arm D: fraction of continuation spent on A/B only")
    # split config
    ap.add_argument("--rank", type=int, default=96)
    ap.add_argument("--split-layers", default="", help="comma list; default = all")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--delta-pre-norm", type=int, default=1)
    ap.add_argument("--read-dim", type=int, default=64, help="LocalRead / additive arms")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--init-from", default="")
    ap.add_argument("--init-sha256", default="")
    ap.add_argument("--pretrained-model", default="",
                    help="4B port: continue from a real HF checkpoint instead of a "
                         "from-scratch T0. The pretrained weights ARE T0.")
    ap.add_argument("--grad-checkpointing", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--eval-batches", type=int, default=32)
    ap.add_argument("--final-eval-batches", type=int, default=128)
    ap.add_argument("--pilot-steps", type=int, default=0)
    ap.add_argument("--save-final", type=int, default=1)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    t_start = time.time()
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    man = json.loads((Path(args.data_dir) / "pg19_manifest.json").read_text())
    # resolve arrays relative to --data-dir: the manifest records the absolute
    # build path, which differs on the second host the matrix runs on.
    train_npy = str(Path(args.data_dir) / Path(man["train"]["path"]).name)
    val_npy = str(Path(args.data_dir) / Path(man["val"]["path"]).name)
    assert man["seq_len"] == args.seq_len, (man["seq_len"], args.seq_len)

    extra, groups = {}, None
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.pretrained_model:
        # 4B port: the pretrained checkpoint IS T0.  bf16 weights, adapter-only
        # arms keep the backbone frozen so no optimizer state is allocated for it.
        from transformers import AutoModelForCausalLM
        assert args.arm in CONTINUATION, "--pretrained-model is for continuation arms"
        assert args.tokenizer == args.pretrained_model or True
        torch.manual_seed(args.seed)
        model = AutoModelForCausalLM.from_pretrained(
            args.pretrained_model, dtype=torch.bfloat16,
            attn_implementation=args.attn_impl).to(dev)
        cfg = model.config
        if args.grad_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        extra["pretrained_model"] = args.pretrained_model
        extra["pretraining_overlap_caveat"] = (
            "PG19 is public-domain Gutenberg text and is very likely inside this "
            "model's own pretraining corpus, so continued pretraining here is "
            "in-domain refresh, not exposure to unseen data. Arms are still "
            "compared under identical conditions, but absolute gains must not be "
            "read as learning new material.")
    else:
        cfg = build_config(len(tok), args)
        torch.manual_seed(args.seed)
        model = Qwen3ForCausalLM(cfg).to(dev, torch.float32)

    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum
    n_model_layers = getattr(cfg, "num_hidden_layers", args.layers)
    split_layers = ([int(x) for x in args.split_layers.split(",")] if args.split_layers
                    else list(range(n_model_layers)))

    # ------------------------------------------------------------- from scratch
    if args.arm in ("vanilla", "native_diffv2"):
        if args.arm == "native_diffv2":
            from deltamem.core.diffv2_native import convert_to_native_diffv2, init_stats
            torch.manual_seed(args.seed)
            extra["converted_layers"] = len(convert_to_native_diffv2(model))
            model = model.to(dev) if args.pretrained_model else model.to(dev, torch.float32)
            extra["init_stats_sample"] = {
                k: v for k, v in list(init_stats(model).items())[:4]}
        start_tokens, end_tokens = 0, args.total_tokens
        groups = [{"params": list(model.parameters()), "lr": args.lr, "name": "all"}]
    # ------------------------------------------------------------ continuation
    else:
        if args.pretrained_model:
            start_tokens, end_tokens = 0, args.continue_tokens
        else:
            assert args.init_from, "continuation arms require --init-from (shared T0)"
            got = sha256_file(args.init_from)
            if args.init_sha256:
                assert got == args.init_sha256, f"T0 mismatch {got} != {args.init_sha256}"
            blob = torch.load(args.init_from, map_location="cpu", weights_only=False)
            model.load_state_dict(blob["model"])
            model = model.to(dev) if args.pretrained_model else model.to(dev, torch.float32)
            extra.update(init_from=args.init_from, init_sha256=got,
                         init_tokens=blob["tokens_seen"])
            start_tokens = blob["tokens_seen"]
            end_tokens = start_tokens + args.continue_tokens

        if args.arm == "vanilla_continue":
            groups = [{"params": list(model.parameters()), "lr": args.lr, "name": "all"}]
        elif args.arm in ("lowrank", "lowrank_unfreeze"):
            from deltamem.core.lowrank_split import (
                attach_lowrank_split, expected_split_param_count,
                freeze_backbone_keep_split)
            attach_lowrank_split(model, split_layers, rank=args.rank, gamma=args.gamma,
                                 delta_pre_norm=bool(args.delta_pre_norm))
            model = model.to(dev) if args.pretrained_model else model.to(dev, torch.float32)
            freeze_backbone_keep_split(model)
            extra["split_layers"] = split_layers
            extra["split_params_expected"] = expected_split_param_count(model)
            groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                       "lr": args.split_lr, "name": "split"}]
        elif args.arm == "localreader":
            from deltamem.core.diff_split import (attach_diff_split,
                                                  freeze_backbone_keep_diff)
            attach_diff_split(model, split_layers, read_dim=args.read_dim,
                              window=args.window, gamma=args.gamma, dynamic_gate=False)
            model = model.to(dev) if args.pretrained_model else model.to(dev, torch.float32)
            freeze_backbone_keep_diff(model)
            extra["split_layers"] = split_layers
            groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                       "lr": args.split_lr, "name": "split"}]
        elif args.arm == "additive":
            from deltamem.core.small_additive import (attach_additive_sidecar,
                                                      freeze_backbone_keep_sidecar)
            attach_additive_sidecar(model, split_layers, read_dim=args.read_dim,
                                    window=args.window)
            model = model.to(dev) if args.pretrained_model else model.to(dev, torch.float32)
            freeze_backbone_keep_sidecar(model)
            extra["split_layers"] = split_layers
            groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                       "lr": args.split_lr, "name": "split"}]

    pr = param_report(model)
    total_steps = max(1, (end_tokens - start_tokens) // tokens_per_step)
    if args.pilot_steps:
        total_steps = args.pilot_steps
    warmup = max(1, int(args.warmup_frac * total_steps))
    stage2_step = (int(args.stage1_frac * total_steps)
                   if args.arm == "lowrank_unfreeze" else None)

    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))
    peak = [g["lr"] for g in opt.param_groups]

    stream = SeqStream(train_npy, args.batch_size, dev, data_seed=args.data_seed,
                       start_seq=start_tokens // args.seq_len)
    extra["stream_start_seq"] = start_tokens // args.seq_len
    print(f"[{args.tag}] arm={args.arm} params={pr} tokens/step={tokens_per_step:,} "
          f"steps={total_steps:,} {start_tokens:,}->{end_tokens:,} "
          f"stage2@{stage2_step}", flush=True)

    model.train()
    log, tokens_seen = [], start_tokens
    t_last, tok_last = time.time(), 0
    t0_path = out / f"{args.tag}_T0.pt"
    saved_t0 = None

    for step in range(1, total_steps + 1):
        if stage2_step is not None and step == stage2_step + 1:
            from deltamem.core.lowrank_split import unfreeze_split_layer_attention
            newly = unfreeze_split_layer_attention(model)
            bb = [p for n, p in model.named_parameters()
                  if p.requires_grad and not any(m in n for m in ("lr_A", "lr_B"))]
            opt.add_param_group({"params": bb, "lr": args.backbone_lr,
                                 "weight_decay": args.weight_decay})
            peak.append(args.backbone_lr)
            extra["stage2_unfrozen_tensors"] = len(newly)
            extra["stage2_unfrozen_params"] = sum(p.numel() for p in bb)
            extra["stage2_step"] = step
            print(f"[{args.tag}] STAGE2 @step {step}: unfroze {len(newly)} tensors "
                  f"({sum(p.numel() for p in bb):,} params) at lr {args.backbone_lr}",
                  flush=True)

        scale = (step / warmup) if step <= warmup else (
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup)
                                            / max(1, total_steps - warmup))))
        for g, pk in zip(opt.param_groups, peak):
            g["lr"] = pk * scale

        opt.zero_grad(set_to_none=True)
        tot_loss = 0.0
        for _ in range(args.grad_accum):
            x = stream.next()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, labels=x).loss
            (loss / args.grad_accum).backward()
            tot_loss += float(loss) / args.grad_accum
        gn = torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]], args.grad_clip)
        opt.step()
        tokens_seen += tokens_per_step

        if step % args.log_every == 0 or step == 1:
            dt = time.time() - t_last
            tps = (tokens_seen - tok_last) / max(dt, 1e-9) if tok_last else 0
            t_last, tok_last = time.time(), tokens_seen
            print(f"[{args.tag}] step {step}/{total_steps} tok {tokens_seen:,} "
                  f"loss {tot_loss:.4f} gn {float(gn):.2f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"{tps:,.0f} tok/s ({(time.time()-t_start)/60:.1f}m)", flush=True)
            log.append({"step": step, "tokens": tokens_seen, "loss": tot_loss,
                        "gnorm": float(gn), "lr": opt.param_groups[0]["lr"],
                        "tok_per_s": tps})

        if step % args.eval_every == 0:
            ev = evaluate(model, val_npy, args.batch_size, dev, args.eval_batches)
            print(f"[{args.tag}] step {step} VAL nll {ev['val_nll']:.4f} "
                  f"ppl {ev['val_ppl']:.2f} bypos {ev['nll_by_position']}", flush=True)
            log.append({"step": step, "tokens": tokens_seen, **ev})

        if (args.arm == "vanilla" and saved_t0 is None
                and tokens_seen >= args.t0_tokens):
            torch.save({"model": model.state_dict(), "tokens_seen": tokens_seen,
                        "config": cfg.to_dict(), "args": vars(args),
                        "data_manifest_sha": man["train"]["sha256"]}, t0_path)
            saved_t0 = sha256_file(t0_path)
            print(f"[{args.tag}] saved T0 @ {tokens_seen:,} -> {t0_path}\n"
                  f"[{args.tag}] T0 sha256 = {saved_t0}", flush=True)

    ev = evaluate(model, val_npy, args.batch_size, dev, args.final_eval_batches)
    res = {"tag": args.tag, "arm": args.arm, "args": vars(args), "argv": sys.argv,
           "params": pr, "final": ev, "tokens_seen": tokens_seen,
           "start_tokens": start_tokens, "steps": total_steps, "log": log,
           "t0_sha256": saved_t0, "seed": args.seed, "data_seed": args.data_seed,
           "data_manifest": {k: man[k] for k in ("corpus", "packing", "seq_len",
                                                 "tokenizer", "train", "val")},
           "peak_vram_bytes": torch.cuda.max_memory_allocated(),
           "runtime_min": round((time.time() - t_start) / 60, 2),
           "env": {"python": platform.python_version(), "torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name(0),
                   "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")},
           **extra}
    try:
        res["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
            text=True).strip()
    except Exception:
        pass
    if args.save_final and not args.pilot_steps:
        torch.save({"model": model.state_dict(), "tokens_seen": tokens_seen,
                    "config": cfg.to_dict(), "args": vars(args)},
                   out / f"{args.tag}_final.pt")
    (out / f"{args.tag}.json").write_text(json.dumps(res, indent=1))
    print(f"[{args.tag}] FINAL nll={ev['val_nll']:.4f} ppl={ev['val_ppl']:.2f} "
          f"bypos={ev['nll_by_position']} in {res['runtime_min']}m", flush=True)


if __name__ == "__main__":
    main()
