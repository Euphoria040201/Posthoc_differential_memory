#!/usr/bin/env python
"""Re-evaluate any CPT checkpoint with ONE evaluator, dumping per-sequence NLL.

The training loop's inline eval reports a scalar.  Every claim in the report
needs a confidence interval, and a CI needs per-example values, so all arms are
re-scored here by the same code path from their saved `*_final.pt`.

Per-sequence NLL also makes the comparison PAIRED: arm X and arm Y see the same
val windows in the same order, so the paired difference removes between-window
variance, which dominates raw NLL.

    python scripts/cpt_eval.py --ckpt out_cpt_20260817/van_s0_final.pt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def rebuild(ckpt_path, device="cuda"):
    """Rebuild the exact architecture an arm used, then load its weights."""
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Qwen3Config(**blob["config"])
    a = blob["args"]
    arm = a["arm"]
    torch.manual_seed(a.get("seed", 0))
    model = Qwen3ForCausalLM(cfg)

    layers = ([int(x) for x in a["split_layers"].split(",")] if a.get("split_layers")
              else list(range(cfg.num_hidden_layers)))
    if arm == "native_diffv2":
        from deltamem.core.diffv2_native import convert_to_native_diffv2
        convert_to_native_diffv2(model)
    elif arm in ("lowrank", "lowrank_unfreeze"):
        from deltamem.core.lowrank_split import attach_lowrank_split
        attach_lowrank_split(model, layers, rank=a["rank"], gamma=a["gamma"],
                             delta_pre_norm=bool(a.get("delta_pre_norm", 1)))
    elif arm == "localreader":
        from deltamem.core.diff_split import attach_diff_split
        attach_diff_split(model, layers, read_dim=a["read_dim"], window=a["window"],
                          gamma=a["gamma"], dynamic_gate=False)
    elif arm == "additive":
        from deltamem.core.small_additive import attach_additive_sidecar
        attach_additive_sidecar(model, layers, read_dim=a["read_dim"],
                                window=a["window"])

    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint/architecture mismatch for arm={arm}: "
                           f"missing={list(missing)[:4]} unexpected={list(unexpected)[:4]}")
    return model.to(device, torch.float32).eval(), blob, arm


@torch.no_grad()
def evaluate(model, val_path, batch_size, device, max_seqs=None):
    arr = np.load(val_path, mmap_mode="r")
    n_seq, seq_len = arr.shape
    if max_seqs:
        n_seq = min(n_seq, max_seqs)
    per_seq, pos_sum, pos_cnt = [], None, 0
    for b in range(0, n_seq - batch_size + 1, batch_size):
        x = torch.from_numpy(np.asarray(arr[b:b + batch_size], dtype=np.int64)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=x).logits
        nll = torch.nn.functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            x[:, 1:].reshape(-1), reduction="none").view(x.shape[0], -1)
        per_seq.extend(nll.mean(-1).double().cpu().tolist())
        pos_sum = nll.sum(0).double() if pos_sum is None else pos_sum + nll.sum(0).double()
        pos_cnt += x.shape[0]
    per_pos = (pos_sum / pos_cnt).cpu().numpy()
    buckets = {f"{a}-{b}": float(per_pos[a:min(b, len(per_pos))].mean())
               for a, b in [(0, 128), (128, 512), (512, 1024), (1024, 2048), (2048, 4096)]
               if a < len(per_pos)}
    mean = float(np.mean(per_seq))
    return {"val_nll": mean, "val_ppl": math.exp(mean), "n_seq": len(per_seq),
            "per_seq_nll": per_seq, "nll_by_position": buckets,
            "per_position_every64": per_pos[::64].tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default="/work/mingze/Posthoc_differential_memory/out_cpt_20260817")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-seqs", type=int, default=2048)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    man = json.loads((Path(args.data_dir) / "pg19_manifest.json").read_text())
    val = str(Path(args.data_dir) / Path(man["val"]["path"]).name)
    model, blob, arm = rebuild(args.ckpt)
    # Distinguish variants that share an --arm value.  Layout-screen runs, the
    # post-norm ablation and the 4B port are all `--arm lowrank`; pooling them
    # into one "lowrank" group would silently corrupt that arm's mean and its
    # seed spread, which is what the noise floor is computed from.
    a = blob["args"]
    raw_arm = arm
    if a.get("pretrained_model"):
        arm = f"{arm}_4b"
    if arm.startswith("lowrank") and not a.get("delta_pre_norm", 1):
        arm = f"{arm}_postnorm"
    tag = a["tag"]
    if tag.startswith("layout_"):
        arm = tag.rsplit("_s", 1)[0]          # layout_last4_s0 -> layout_last4
    else:
        # General rule for any other variant that reuses an --arm value: if the
        # tag carries extra tokens beyond the arm name (e.g. lowrank_r192_s0),
        # keep them.  Without this the capacity probe is pooled into `lowrank`,
        # which shifts that arm's mean AND its seed spread -- and the spread is
        # what the noise floor, hence the recovery-ratio gate, is computed from.
        stem = tag.rsplit("_s", 1)[0] if tag.rsplit("_s", 1)[-1].isdigit() else tag
        if stem != raw_arm and raw_arm in stem:
            arm = stem + ("_4b" if a.get("pretrained_model") else "")
    res = evaluate(model, val, args.batch_size, "cuda", args.max_seqs)
    res.update(ckpt=args.ckpt, arm=arm, raw_arm=raw_arm, tag=blob["args"]["tag"],
               seed=blob["args"].get("seed"), tokens_seen=blob.get("tokens_seen"),
               val_sha256=man["val"]["sha256"], data_seed=blob["args"].get("data_seed"))
    out = args.out or str(Path(args.ckpt).with_suffix("")) + "_eval.json"
    Path(out).write_text(json.dumps(res, indent=1))
    print(f"[{res['tag']}] arm={arm} nll={res['val_nll']:.5f} ppl={res['val_ppl']:.2f} "
          f"n={res['n_seq']} bypos={res['nll_by_position']} -> {out}")


if __name__ == "__main__":
    main()
