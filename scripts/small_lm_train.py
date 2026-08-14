#!/usr/bin/env python
"""Chain B: controlled small-model LM pretraining.

Arms
  vanilla   plain Qwen3 decoder, random init, trained 0 -> T (checkpoint at T0)
  diffv2    official-style DIFF V2, random init, trained 0 -> T
  ours      fork of vanilla@T0: function-preserving split, backbone frozen,
            only the new branch trained over T0 -> T
  additive  fork of the SAME vanilla@T0: parameter-matched additive sidecar,
            backbone frozen, no real differential attention

`ours` and `additive` MUST fork from a bit-identical vanilla@T0, so both load
the same checkpoint file and the loader asserts its sha256.

The model implementation is HuggingFace's own Qwen3 (plus the audited DIFF V2
module); only the data pipeline and the training loop are local, because
neither repo contains an LM pretraining trainer to reuse.
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


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def build_config(tok_vocab: int, args):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    return Qwen3Config(
        vocab_size=tok_vocab,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        head_dim=args.head_dim,
        max_position_embeddings=args.seq_len,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        attn_implementation=args.attn_impl,
    )


def param_report(model) -> dict:
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    attn = sum(p.numel() for n, p in model.named_parameters() if "self_attn" in n)
    mlp = sum(p.numel() for n, p in model.named_parameters() if ".mlp." in n)
    emb = sum(p.numel() for n, p in model.named_parameters()
              if "embed_tokens" in n or "lm_head" in n)
    return {"total": tot, "trainable": tr, "attention": attn, "ffn": mlp,
            "embedding": emb, "non_embedding": tot - emb}


class Batches:
    """Deterministic contiguous windows over the shared token manifest."""

    def __init__(self, path, seq_len, batch_size, device, start_token=0):
        self.arr = np.load(path, mmap_mode="r")
        self.seq_len, self.bs, self.device = seq_len, batch_size, device
        self.span = seq_len * batch_size
        # The forked arms resume at T0 tokens, which is past the end of the
        # manifest once the run spans several epochs.  Wrap so a fork lands on
        # exactly the window vanilla would have seen next, instead of silently
        # restarting at token 0 and training on different data than the control.
        self.pos = start_token % len(self.arr)

    def next(self):
        if self.pos + self.span + 1 > len(self.arr):
            self.pos = 0
        chunk = np.asarray(self.arr[self.pos: self.pos + self.span + 1], dtype=np.int64)
        self.pos += self.span
        x = torch.from_numpy(chunk[:-1]).view(self.bs, self.seq_len)
        y = torch.from_numpy(chunk[1:]).view(self.bs, self.seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)


@torch.no_grad()
def evaluate(model, path, seq_len, batch_size, device, max_batches):
    model.eval()
    b = Batches(path, seq_len, batch_size, device)
    tot, n = 0.0, 0
    for _ in range(max_batches):
        x, _y = b.next()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=x, labels=x).loss
        tot += float(loss); n += 1
    model.train()
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["vanilla", "diffv2", "ours", "additive"])
    ap.add_argument("--data-dir", default="/work/mingze/Posthoc_differential_memory/out_smalllm_20260814")
    ap.add_argument("--out-dir", default="/work/mingze/Posthoc_differential_memory/out_smalllm_20260814")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--tokenizer", default="/work/mingze/models/Qwen3-4B-Instruct-2507")
    # architecture (derived from Qwen3-4B ratios; see report)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=2)      # GQA 4:1 like Qwen3-4B
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--intermediate", type=int, default=1920)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--attn-impl", default="sdpa")
    # optimisation
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--total-tokens", type=int, default=160_000_000)
    ap.add_argument("--t0-tokens", type=int, default=112_000_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--branch-lr", type=float, default=5e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=24)
    # post-hoc arms
    ap.add_argument("--init-from", default="")
    ap.add_argument("--init-sha256", default="")
    ap.add_argument("--diff-read-dim", type=int, default=64)
    ap.add_argument("--diff-window", type=int, default=256)
    ap.add_argument("--diff-gamma", type=float, default=1.0)
    ap.add_argument("--pilot-steps", type=int, default=0,
                    help="if >0, run only this many steps and report throughput")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    t_start = time.time()

    man = json.loads((Path(args.data_dir) / "data_manifest.json").read_text())
    train_npy = man["train"]["path"]; val_npy = man["val"]["path"]

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    cfg = build_config(len(tok), args)
    torch.manual_seed(args.seed)
    model = Qwen3ForCausalLM(cfg).to(dev, torch.float32)

    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum
    extra = {}

    if args.arm == "diffv2":
        from deltamem.core.small_diffv2 import convert_to_diffv2
        torch.manual_seed(args.seed)
        conv = convert_to_diffv2(model)
        model = model.to(dev, torch.float32)
        extra["converted_layers"] = len(conv)
        start_tokens, total_tokens = 0, args.total_tokens
        params = list(model.parameters())
        groups = [{"params": params, "lr": args.lr}]
    elif args.arm == "vanilla":
        start_tokens, total_tokens = 0, args.total_tokens
        groups = [{"params": list(model.parameters()), "lr": args.lr}]
    else:
        assert args.init_from, "--init-from is required for post-hoc arms"
        got = sha256_file(args.init_from)
        if args.init_sha256:
            assert got == args.init_sha256, f"T0 checkpoint mismatch: {got} != {args.init_sha256}"
        blob = torch.load(args.init_from, map_location="cpu")
        model.load_state_dict(blob["model"])
        model = model.to(dev, torch.float32)
        extra["init_from"] = args.init_from
        extra["init_sha256"] = got
        extra["init_tokens"] = blob["tokens_seen"]
        start_tokens, total_tokens = blob["tokens_seen"], args.total_tokens
        layers = tuple(range(args.layers))
        if args.arm == "ours":
            from deltamem.core.diff_split import (
                attach_diff_split, freeze_backbone_keep_diff)
            attach_diff_split(model, layers, read_dim=args.diff_read_dim,
                              window=args.diff_window, gamma=args.diff_gamma,
                              dynamic_gate=False)
        else:
            from deltamem.core.small_additive import (
                attach_additive_sidecar as attach_diff_split_like)
            attach_diff_split_like(model, layers, read_dim=args.diff_read_dim,
                                   window=args.diff_window)
            from deltamem.core.small_additive import freeze_backbone_keep_sidecar
            freeze_backbone_keep_diff = freeze_backbone_keep_sidecar
        model = model.to(dev, torch.float32)
        names = freeze_backbone_keep_diff(model)
        assert names, "no trainable branch parameters"
        groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                   "lr": args.branch_lr}]

    pr = param_report(model)
    total_steps = max(1, (total_tokens - start_tokens) // tokens_per_step)
    if args.pilot_steps:
        total_steps = args.pilot_steps
    warmup = max(1, int(args.warmup_frac * total_steps))
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))
    peak = {id(g["params"][0]): g["lr"] for g in groups}

    print(f"[{args.tag}] arm={args.arm} params={pr} tokens/step={tokens_per_step:,} "
          f"steps={total_steps:,} start_tokens={start_tokens:,} "
          f"total_tokens={total_tokens:,}", flush=True)

    batches = Batches(train_npy, args.seq_len, args.batch_size, dev,
                      start_token=start_tokens)
    model.train()
    log, t_last, tok_last = [], time.time(), 0
    tokens_seen = start_tokens
    t0_path = out / f"{args.tag}_T0.pt"
    saved_t0 = None

    for step in range(1, total_steps + 1):
        frac = step / total_steps
        scale = (step / warmup) if step <= warmup else \
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
        for g in groups:
            g["lr"] = peak[id(g["params"][0])] * scale
        opt.zero_grad(set_to_none=True)
        tot_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = batches.next()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # let HF compute the LM loss: it uses a chunked/fused CE, which
                # matters here because the 151k vocab makes a materialised fp32
                # logits tensor larger than the rest of training combined.
                # HF shifts internally (logits[:-1] vs labels[1:]), so the label
                # argument is the UNSHIFTED sequence -- passing `y` would shift twice.
                loss = model(input_ids=x, labels=x).loss
            (loss / args.grad_accum).backward()
            tot_loss += float(loss) / args.grad_accum
        gn = torch.nn.utils.clip_grad_norm_(
            [p for g in groups for p in g["params"]], args.grad_clip)
        opt.step()
        tokens_seen += tokens_per_step

        if step % args.log_every == 0 or step == 1:
            dt = time.time() - t_last
            tps = (tokens_seen - tok_last) / max(dt, 1e-9) if tok_last else 0
            t_last, tok_last = time.time(), tokens_seen
            print(f"[{args.tag}] step {step}/{total_steps} tokens {tokens_seen:,} "
                  f"loss {tot_loss:.4f} gnorm {float(gn):.3f} lr {groups[0]['lr']:.2e} "
                  f"{tps:,.0f} tok/s ({(time.time()-t_start)/60:.1f}m)", flush=True)
            log.append({"step": step, "tokens": tokens_seen, "loss": tot_loss,
                        "gnorm": float(gn), "lr": groups[0]["lr"], "tok_per_s": tps})

        if step % args.eval_every == 0:
            vl = evaluate(model, val_npy, args.seq_len, args.batch_size, dev,
                          args.eval_batches)
            print(f"[{args.tag}] step {step} tokens {tokens_seen:,} "
                  f"VAL loss {vl:.4f} ppl {math.exp(vl):.2f}", flush=True)
            log.append({"step": step, "tokens": tokens_seen, "val_loss": vl,
                        "val_ppl": math.exp(vl)})

        if (args.arm == "vanilla" and saved_t0 is None
                and tokens_seen >= args.t0_tokens):
            torch.save({"model": model.state_dict(), "tokens_seen": tokens_seen,
                        "config": cfg.to_dict(), "args": vars(args)}, t0_path)
            saved_t0 = sha256_file(t0_path)
            print(f"[{args.tag}] saved T0 @ {tokens_seen:,} tokens -> {t0_path}\n"
                  f"[{args.tag}] T0 sha256 = {saved_t0}", flush=True)

    vl = evaluate(model, val_npy, args.seq_len, args.batch_size, dev, args.eval_batches * 2)
    res = {"tag": args.tag, "arm": args.arm, "args": vars(args), "params": pr,
           "final_val_loss": vl, "final_val_ppl": math.exp(vl),
           "tokens_seen": tokens_seen, "steps": total_steps, "log": log,
           "t0_sha256": saved_t0, "data_manifest": man,
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
    if not args.pilot_steps:
        torch.save({"model": model.state_dict(), "tokens_seen": tokens_seen,
                    "config": cfg.to_dict(), "args": vars(args)},
                   out / f"{args.tag}_final.pt")
    (out / f"{args.tag}.json").write_text(json.dumps(res, indent=1))
    print(f"[{args.tag}] FINAL val_loss={vl:.4f} ppl={math.exp(vl):.2f} "
          f"tokens={tokens_seen:,} in {res['runtime_min']}m -> {out/f'{args.tag}.json'}",
          flush=True)


if __name__ == "__main__":
    main()
