"""Point-6 + point-5 checks.
[6] seg A/B: same prompt labelled all-CTX vs proper CTX/QRY split must give IDENTICAL
    logits in the read phase (mask is provably seg-invariant for normal rows when
    normal_attends_prefix=True; this verifies it empirically).
[5] deep-read causal finite difference: toggle _window_only on ONE layer at a time and
    measure the final-logit change. Nonzero deep-layer effect while _dbg prefR/R says 0
    would mean the diagnostic (not the architecture) is wrong. Test at ~3000 tok (dense
    path) AND ~4500 tok (chunked path) to rule out chunk-branch artifacts.
"""
import sys, torch
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval_ours_hotpotqa import load_ours
from qasper_prefix_steer import build_examples
from deltamem.core.prefix_steer import (set_steer_segments, set_write_freeze,
    clear_frozen_memory, iter_steer_modules)
from deltamem.core.global_prefix import SEG_CTX, SEG_QRY

DEV = "cuda"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", local_files_only=True)
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16,
    attn_implementation="sdpa", local_files_only=True).to(DEV)
load_ours(m, "out_ctxmask/nct_r4_ckpt.pt"); m.eval()
val = build_examples("validation", 4, tok, 256, 4500, 24)

print("=== [6] seg labelling A/B (read phase, frozen memory) ===")
worst = 0.0
for ex in val[:3]:
    cids = torch.tensor([ex["ctx_ids"]], device=DEV)
    clear_frozen_memory(m); set_write_freeze(m, True)
    set_steer_segments(m, torch.full_like(cids, SEG_CTX), torch.ones_like(cids, dtype=torch.bool))
    with torch.no_grad(): m(input_ids=cids, use_cache=False)
    set_write_freeze(m, False)
    ids = torch.tensor([ex["prompt_ids"]], device=DEV)
    nc = len(ex["ctx_ids"])
    seg_proper = torch.tensor([[SEG_CTX] * nc + [SEG_QRY] * (ids.shape[1] - nc)], device=DEV)
    seg_allctx = torch.full_like(ids, SEG_CTX)
    outs = {}
    for name, seg in [("proper", seg_proper), ("allctx", seg_allctx)]:
        set_steer_segments(m, seg, torch.ones_like(ids, dtype=torch.bool))
        with torch.no_grad():
            outs[name] = m(input_ids=ids, use_cache=False).logits[0, -1].float()
    d = float((outs["proper"] - outs["allctx"]).abs().max())
    worst = max(worst, d)
    print(f"  len={ids.shape[1]}: max|logit diff| proper-vs-allCTX = {d:.6f} "
          f"top1_same={int(outs['proper'].argmax()) == int(outs['allctx'].argmax())}")
print(f"[6] worst diff = {worst:.6f}  ({'SEG-INVARIANT ✓' if worst < 1e-2 else 'SEG MATTERS ✗'})")

print("=== [5] per-layer causal finite difference (remove prefix from READ at ONE layer) ===")
mods = list(iter_steer_modules(m))
LAY = [x.base.layer_idx for x in mods]
for cap, tag in [(3000, "dense-path"), (4500, "chunked-path")]:
    ex = next(e for e in val if len(e["ctx_ids"]) >= 4400)
    ids_l = ex["ctx_ids"][:cap] + ex["prompt_ids"][len(ex["ctx_ids"]):]
    ids = torch.tensor([ids_l], device=DEV)
    seg = torch.tensor([[SEG_CTX] * cap + [SEG_QRY] * (ids.shape[1] - cap)], device=DEV)
    def fwd():
        # ctx-mode single forward: write+read happen in-pass, like training/eval with context
        clear_frozen_memory(m)
        set_steer_segments(m, seg, torch.ones_like(ids, dtype=torch.bool))
        with torch.no_grad():
            return m(input_ids=ids, use_cache=False).logits[0, -1].float()
    base_lg = fwd()
    diffs = []
    for x in mods:
        x._window_only = True
        lg = fwd()
        x._window_only = False
        diffs.append(float((lg - base_lg).abs().max()))
    print(f"  [{tag} L={ids.shape[1]}] per-layer max|Δlogit| when prefix removed from that layer's read:")
    print("   " + " ".join(f"L{l}={d:.3f}" for l, d in zip(LAY, diffs)))
print("DONE")
