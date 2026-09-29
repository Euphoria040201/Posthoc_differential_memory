#!/usr/bin/env python
"""Offline CPU integration experiment. Random tiny Qwen; NOT a downstream quality result."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default="runs/differential_cpu_smoke")
    args = ap.parse_args()
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from qasper_prefix_steer import _episode_to_examples
    from deltamem.kv_binding.qa_protocol import write_manifest, example_id
    torch.set_num_threads(2)
    torch.manual_seed(123)
    words = ["<pad>", "<eos>", "<unk>", "Context", ":", "alpha", "beta", "value", "red", "blue",
             "filler", "one", "two", "what", "?", "Question", "Answer", "."]
    tokenizer = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<eos>",
                                 pad_token="<pad>", unk_token="<unk>")
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                     head_dim=8, max_position_embeddings=256,
                     bos_token_id=None, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    model = Qwen3ForCausalLM(cfg)
    model_dir = root / "tiny_model"
    model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    splits = {}
    for split in ("train", "calibration", "validation", "test"):
        examples = []
        for i, answer in enumerate(("red", "blue")):
            rows = _episode_to_examples([f"alpha value {answer}", "filler one", "filler two"],
                                         [{"question": "what alpha value?", "answer": answer}], tok, 128, 8)
            ex = rows[0]
            ex.update(sample_id=example_id("synthetic", split, i, i, "what alpha value?", answer),
                      paper_id=f"{split}_{i}", question="what alpha value?")
            examples.append(ex)
        splits[split] = examples
    manifest_path = root / "data.json"
    write_manifest(manifest_path, tok, splits, {"synthetic_smoke_only": True,
                                               "settings": {"max_ans_tok": 8}})
    command = [sys.executable, str(REPO / "scripts/run_differential_v2.py"), "run",
               "--model-path", str(model_dir), "--data-manifest", str(manifest_path),
               "--output-root", str(root), "--phase", "pilot", "--device", "cpu",
               "--dtype", "float32", "--attn-impl", "eager", "--steps", "2",
               "--grad-accum", "1", "--group-k", "2", "--steer-layers", "0,1",
               "--steer-mem-head-dim", "8", "--steer-window", "8", "--eval-examples", "2",
               "--robust-examples", "2", "--eval-group-k", "2", "--max-new-tokens", "8"]
    subprocess.run(command, check=True, cwd=REPO)
    common = ["--model-path", str(model_dir), "--data-manifest", str(manifest_path),
              "--steer-ckpt", str(root / "pilot/sub_nui_s0_steer.pt"),
              "--device", "cpu", "--dtype", "float32", "--attn-impl", "eager",
              "--eval-examples", "2", "--max-new-tokens", "8",
              "--calibrate-batches", "2", "--output-dir", str(root / "mechanism")]
    for arm in ("variance_diff", "learned_diff"):
        subprocess.run([sys.executable, str(REPO / "scripts/dex_stage1_fusion.py"), *common,
                        "--arm", arm, "--tag", arm, "--fusion-steps", "2", "--grad-accum", "1"],
                       check=True, cwd=REPO)
    subprocess.run([sys.executable, str(REPO / "scripts/summarize_differential_v2.py"),
                    "--directory", str(root / "pilot"), "--output", str(root / "summary.json")],
                   check=True, cwd=REPO)
    print(json.dumps({"status": "PASS", "kind": "offline CPU integration only",
                      "quality_claim": False, "output_root": str(root)}))


if __name__ == "__main__":
    main()
