#!/usr/bin/env python
"""Evidence for layer placement, measured on TRAINING data only.

The task forbids choosing layers by final test results, so the selection signal
has to be intrinsic.  Two statistics are collected per layer from the shared T0
checkpoint:

  attn_entropy   mean Shannon entropy of the attention distribution, in nats.
                 This is the statistic the DIFF-V2 paper itself selects on (it
                 picks the highest-entropy heads): a high-entropy row is a
                 diffuse, noisy attention map, which is exactly what a
                 differential second branch is supposed to cancel.

  qproj_grad     L2 norm of dL/d(q_proj.weight) under the LM loss, normalised by
                 the weight norm.  A layer whose query map is already receiving
                 large relative gradient is a layer where changing the query
                 geometry matters.

Both are computed on windows drawn from the TRAIN array (offset well past the
tokens T0 consumed is unnecessary — these are diagnostics of the model, not a
performance estimate — but the val file is never touched).

Writes out_cpt_20260817/layer_evidence_<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="/work/mingze/Posthoc_differential_memory/out_cpt_20260817")
    ap.add_argument("--n-windows", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--entropy-seq", type=int, default=1024,
                    help="entropy needs materialised attention; use a shorter window")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Qwen3Config(**blob["config"])
    model = Qwen3ForCausalLM(cfg)
    model.load_state_dict(blob["model"])
    model = model.to("cuda", torch.float32).eval()
    n_layers = cfg.num_hidden_layers

    man = json.loads((Path(args.data_dir) / "pg19_manifest.json").read_text())
    arr = np.load(str(Path(args.data_dir) / Path(man["train"]["path"]).name), mmap_mode="r")
    rng = np.random.default_rng(1234)
    rows = rng.integers(0, arr.shape[0], args.n_windows)

    # ---------------------------------------------------- attention entropy
    model.config._attn_implementation = "eager"
    ent = np.zeros(n_layers)
    n_ent = 0
    with torch.no_grad():
        for r in rows[: max(1, args.n_windows // 2)]:
            x = torch.from_numpy(np.asarray(arr[r][: args.entropy_seq],
                                            dtype=np.int64))[None].cuda()
            out = model(input_ids=x, output_attentions=True)
            for li, a in enumerate(out.attentions):
                p = a.float().clamp_min(1e-12)          # [B, H, L, L]
                e = -(p * p.log()).sum(-1)              # entropy per query row
                ent[li] += float(e.mean())
            n_ent += 1
            del out
            torch.cuda.empty_cache()
    ent /= max(n_ent, 1)

    # ------------------------------------------------------- q_proj gradient
    model.config._attn_implementation = "sdpa"
    model.train()
    grad = np.zeros(n_layers)
    wnorm = np.zeros(n_layers)
    n_g = 0
    for r in rows:
        x = torch.from_numpy(np.asarray(arr[r], dtype=np.int64))[None].cuda()
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(input_ids=x, labels=x).loss.backward()
        for li, layer in enumerate(model.model.layers):
            w = layer.self_attn.q_proj.weight
            if w.grad is not None:
                grad[li] += float(w.grad.detach().float().norm())
                wnorm[li] = float(w.detach().float().norm())
        n_g += 1
    grad /= max(n_g, 1)
    model.zero_grad(set_to_none=True)

    rel = grad / np.maximum(wnorm, 1e-9)
    res = {
        "ckpt": args.ckpt, "n_layers": n_layers,
        "n_windows_grad": int(n_g), "n_windows_entropy": int(n_ent),
        "attn_entropy_nats": [round(float(x), 5) for x in ent],
        "qproj_grad_l2": [round(float(x), 6) for x in grad],
        "qproj_grad_rel": [round(float(x), 8) for x in rel],
        "rank_by_entropy": [int(i) for i in np.argsort(-ent)],
        "rank_by_grad_rel": [int(i) for i in np.argsort(-rel)],
    }
    # combined evidence rank: mean of the two z-scores
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)  # noqa: E731
    comb = z(ent) + z(rel)
    res["combined_z"] = [round(float(x), 5) for x in comb]
    res["rank_combined"] = [int(i) for i in np.argsort(-comb)]
    res["selected_half"] = sorted(int(i) for i in np.argsort(-comb)[: n_layers // 2])
    res["note"] = ("selection uses training-data statistics only; no validation or "
                   "test result informs it")

    out = args.out or str(Path(args.data_dir) / f"layer_evidence_{Path(args.ckpt).stem}.json")
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "note"}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
