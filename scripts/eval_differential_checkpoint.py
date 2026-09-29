#!/usr/bin/env python
"""Strict checkpoint reload followed by paired clean/permuted-context QA evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def evaluate(model, tok, examples, device, max_new_tokens, group_k, perturb_seed):
    import torch
    from qasper_prefix_steer import generate, f1_em
    from dex_nuisance_train import build_nuisance_group
    from deltamem.core.prefix_steer import set_steer_segments, set_collect_fusion_tensors, collect_fusion_tensors
    model.eval()
    rows = []
    for ex in examples:
        seed = int(hashlib.sha256(f"{perturb_seed}:{ex['sample_id']}".encode()).hexdigest()[:16], 16)
        variants = build_nuisance_group(ex, group_k, random.Random(seed))
        if variants is None:
            raise ValueError("evaluation population contains an ungroupable example")
        predictions, states = [], []
        prompt_len = len(ex["prompt_ids"])
        for variant in variants:
            prompt = variant["ids"][:prompt_len], variant["seg"][:prompt_len]
            pred = generate(model, tok, ex, device, max_new_tokens, tok.eos_token_id, prompt=prompt)
            f1, em = f1_em(pred, ex["answer"])
            predictions.append({"f1": f1, "em": em, "pred": pred,
                                "order": variant["order"], **ex["last_generation"]})
            # Mechanism diagnostics are sampled at the same answer-prediction position.
            # No gold answer tokens enter this forward. Base/DEX runs can have no fusion rows.
            ids = torch.tensor([prompt[0]], device=device)
            seg = torch.tensor([prompt[1]], device=device)
            valid = torch.ones_like(ids, dtype=torch.bool)
            set_steer_segments(model, seg, valid)
            set_collect_fusion_tensors(model, True)
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)
            captured = []
            from deltamem.core.prefix_steer import iter_steer_modules
            mods = list(iter_steer_modules(model))
            for layer, y, c in collect_fusion_tensors(model):
                m = mods[layer]
                y, c = y[0, -1].float(), c[0, -1].float()
                if m.cfg.output_fusion in ("fixed", "fixed_add"):
                    corrected = y + m.cfg.steer_gain * c
                elif m.cfg.output_fusion == "fixed_sub":
                    corrected = y - m.cfg.steer_gain * c
                elif m.cfg.output_fusion == "learned_diff":
                    corrected = y - m.fusion_lambda.detach().float() * c
                elif m.cfg.output_fusion == "variance_diff":
                    corrected = y - m.fusion_coefficient.float() * (c - m.fusion_mu_c.float())
                else:
                    raise ValueError("unsupported mechanism fusion")
                captured.append((y.cpu(), corrected.cpu()))
            states.append(captured)
            set_collect_fusion_tensors(model, False)
        metrics = {}
        if all(states):
            ys = torch.stack([torch.stack([y for y, _ in row]) for row in states])
            corrected = torch.stack([torch.stack([z for _, z in row]) for row in states])
            before = ys.var(dim=0, unbiased=False).mean().item()
            after = corrected.var(dim=0, unbiased=False).mean().item()
            metrics = {"nuisance_variance_before": before, "nuisance_variance_after": after,
                       "variance_reduction_ratio": 1 - after / (before + 1e-12)}
        scores = [p["f1"] for p in predictions]
        rows.append({"sample_id": ex["sample_id"], "paper_id": ex.get("paper_id", ""),
                     "gold": ex["answer"], "clean_f1": scores[0],
                     "mean_f1": sum(scores) / len(scores), "worst_f1": min(scores),
                     "permuted_f1": sum(scores[1:]) / (len(scores) - 1),
                     "predictions": predictions, **metrics})
    count = len(rows)
    if not count:
        raise ValueError("empty evaluation split")
    summary = {key: sum(row[key] for row in rows) / count
               for key in ("clean_f1", "mean_f1", "worst_f1", "permuted_f1")}
    summary["clean_em"] = sum(r["predictions"][0]["em"] for r in rows) / count
    summary["n"] = count
    pred_rows = [p for r in rows for p in r["predictions"]]
    summary["eos_rate"] = sum(p["stopped_on_eos"] for p in pred_rows) / len(pred_rows)
    summary["mean_generated_tokens"] = sum(p["generated_tokens"] for p in pred_rows) / len(pred_rows)
    return {"summary": summary, "per_example": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-manifest", required=True)
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--group-k", type=int, default=4)
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--perturb-seed", type=int, default=20260929)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.group_k < 2:
        ap.error("group-k must be >= 2")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from qasper_prefix_steer import get_dtype
    from deltamem.kv_binding.qa_protocol import read_manifest
    from deltamem.core.prefix_steer import attach_prefix_steer, set_steer_enabled
    from deltamem.eval.steer_checkpoint import restore_prefix_steer_config, load_steer_state_strict
    from deltamem.eval.dex_checkpoint import restore_dex_config, load_dex_state_strict
    from deltamem.core.dex import attach_dex

    started = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_path)
    data = read_manifest(args.data_manifest, tok)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("data_protocol", {}).get("manifest_sha256") != data["sha256"]:
        raise ValueError("checkpoint was not trained with this exact v2 data manifest")
    if ckpt.get("args", {}).get("model_path") != args.model_path:
        raise ValueError("model-path must match the checkpoint's backbone")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=get_dtype(args.dtype), attn_implementation=args.attn_impl).to(args.device)
    if ckpt.get("steer_config"):
        attach_prefix_steer(model, restore_prefix_steer_config(ckpt["steer_config"]))
    if "config" in ckpt:
        cfg = restore_dex_config(ckpt["config"])
        attach_dex(model, cfg)
        load_dex_state_strict(model, ckpt["state"], cfg)
    else:
        if not ckpt.get("steer_config"):
            attach_prefix_steer(model, restore_prefix_steer_config(ckpt["cfg"]))
        load_steer_state_strict(model, ckpt["state"])
        for p in model.parameters():
            p.requires_grad_(False)
    set_steer_enabled(model, ckpt.get("steer_enabled", True))
    examples = data["splits"][args.split]
    if args.max_examples > 0:
        examples = examples[:args.max_examples]
    if args.max_new_tokens < max(len(ex["ids"]) - len(ex["prompt_ids"]) for ex in examples):
        raise ValueError("max-new-tokens must cover the full supervised answer including EOS")
    with torch.autocast(torch.device(args.device).type, dtype=get_dtype(args.dtype),
                        enabled=args.dtype != "float32"):
        result = evaluate(model, tok, examples, args.device, args.max_new_tokens,
                          args.group_k, args.perturb_seed)
    result.update({"args": vars(args), "manifest_sha256": data["sha256"],
                   "protocol_version": data["protocol_version"],
                   "runtime_min": (time.time() - started) / 60})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    temporary.replace(output)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
