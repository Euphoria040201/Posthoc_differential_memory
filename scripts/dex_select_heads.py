#!/usr/bin/env python
"""Data-driven head selection for DEX (arXiv:2505.16333, Sec. 3.2).

Computes, for every layer and head of the frozen backbone:

* ``entropy``    -- mean Shannon entropy of the softmax attention rows, averaged
                    over query positions >= --min-position and over the
                    calibration batch.  DEX selects the top-k *highest*-entropy
                    heads per layer.
* ``importance`` -- |dL / d xi_h| head-gate sensitivity (Michel et al. /
                    Molchanov et al. style): the gradient of the LM loss w.r.t.
                    a multiplicative gate on head h's output, held at 1.  DEX's
                    low-importance strategy selects the top-k *lowest* scores.

The plan is written once and reused verbatim by every variant and seed, so head
selection is never a source of variance between runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


class _HeadGate(nn.Module):
    """o_proj wrapper multiplying each head's output by a gate held at 1."""

    def __init__(self, base: nn.Module, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.base = base
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.gate = nn.Parameter(torch.ones(num_heads))

    def forward(self, x):
        shape = x.shape
        heads = x.view(*shape[:-1], self.num_heads, self.head_dim)
        heads = heads * self.gate.to(heads.dtype).view(*([1] * (heads.dim() - 2)), -1, 1)
        return self.base(heads.reshape(*shape))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/work/mingze/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--min-position", type=int, default=32,
                    help="skip the first positions, whose entropy is trivially low")
    ap.add_argument("--data", default="qasper")
    ap.add_argument("--train-papers", type=int, default=64)
    ap.add_argument("--max-chunk-tok", type=int, default=256)
    ap.add_argument("--max-ctx-tok", type=int, default=4500)
    ap.add_argument("--max-ans-tok", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from qasper_prefix_steer import build_examples, get_dtype

    torch.manual_seed(args.seed)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=get_dtype(args.dtype), attn_implementation="eager",
    ).to(args.device).eval()
    text_cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    n_layers = text_cfg.num_hidden_layers
    n_heads = text_cfg.num_attention_heads
    head_dim = getattr(text_cfg, "head_dim", None) or text_cfg.hidden_size // n_heads

    examples = build_examples(
        "train", args.train_papers, tok, args.max_chunk_tok, args.max_ctx_tok,
        args.max_ans_tok, data=args.data,
    )
    rng = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(examples), generator=rng)[: args.num_samples].tolist()
    batches = []
    for i in idx:
        ids = examples[i]["ids"][: args.seq_len]
        if len(ids) >= args.min_position + 8:
            batches.append(torch.tensor([ids], device=args.device))
    print(f"[head-select] {len(batches)} calibration sequences, "
          f"L={n_layers} H={n_heads} Dh={head_dim}", flush=True)

    ent_sum = torch.zeros(n_layers, n_heads, dtype=torch.float64)
    ent_n = 0
    for b, ids in enumerate(batches):
        with torch.no_grad():
            out = model(input_ids=ids, output_attentions=True, use_cache=False)
        for li, att in enumerate(out.attentions):
            a = att[0].float()[:, args.min_position:, :]
            ent = -(a.clamp_min(1e-12) * a.clamp_min(1e-12).log()).sum(-1)
            ent_sum[li] += ent.mean(dim=-1).double().cpu()
        ent_n += 1
        del out
        torch.cuda.empty_cache()
        print(f"[head-select] entropy {b + 1}/{len(batches)}", flush=True)
    entropy = (ent_sum / max(ent_n, 1)).tolist()

    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    gates = []
    for module in model.modules():
        if isinstance(module, Qwen3Attention):
            # NB: never call .to(dtype=...) on the wrapper -- that converts the
            # wrapped o_proj too and breaks the bf16 backbone.  Only the gate is fp32.
            g = _HeadGate(module.o_proj, n_heads, head_dim)
            g.gate.data = g.gate.data.float().to(module.o_proj.weight.device)
            module.o_proj = g
            gates.append((module.layer_idx, g))
    gates.sort(key=lambda kv: kv[0])
    for p in model.parameters():
        p.requires_grad_(False)
    for _, g in gates:
        g.gate.requires_grad_(True)

    imp_sum = torch.zeros(n_layers, n_heads, dtype=torch.float64)
    imp_n = 0
    for b, ids in enumerate(batches):
        model.zero_grad(set_to_none=True)
        out = model(input_ids=ids, labels=ids, use_cache=False)
        out.loss.backward()
        for li, g in gates:
            if g.gate.grad is not None:
                imp_sum[li] += g.gate.grad.detach().abs().double().cpu()
        imp_n += 1
        print(f"[head-select] importance {b + 1}/{len(batches)} loss={out.loss.item():.4f}", flush=True)
    importance = (imp_sum / max(imp_n, 1)).tolist()

    payload = {
        "criterion": "entropy",
        "scores": {str(i): entropy[i] for i in range(n_layers)},
        "entropy": {str(i): entropy[i] for i in range(n_layers)},
        "importance": {str(i): importance[i] for i in range(n_layers)},
        "meta": {
            "model_path": args.model_path,
            "num_layers": n_layers,
            "num_heads": n_heads,
            "head_dim": head_dim,
            "num_samples": len(batches),
            "seq_len": args.seq_len,
            "min_position": args.min_position,
            "data": args.data,
            "seed": args.seed,
            "dtype": args.dtype,
            "elapsed_s": round(time.time() - t0, 1),
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[head-select] wrote {out_path} in {payload['meta']['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
