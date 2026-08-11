"""WHY prefix hurts on LoCoMo: decompose the prefix contribution at the answer position on
LONG (LoCoMo) vs SHORT (HotpotQA) contexts, on the REAL eval paths, and test whether the
WRITTEN prefix is doc-independent (lossy 65k->64 attention-pool write).

Metrics per steer layer, at the answer position:
  prefR/R : ||sum_p a_p V_p|| / ||R_t||   -- weighted prefix share of the SWA read
  pfx_mass: sum_p a_p                       -- attention mass on prefix (vs window)
  M/R     : ||M_t|| / ||R_t||               -- pooled (max) prefix term relative to read
Doc-independence of the WRITTEN prefix (dynamic write from context):
  cos_across : mean pairwise cosine of the written prefix across DIFFERENT conversations
               (high => write ignores the doc => injected signal is ~constant noise)
"""
import torch, sys, argparse, statistics as st
from pathlib import Path
sys.path.insert(0, "scripts")
from eval_ours_hotpotqa import load_ours
from transformers import AutoModelForCausalLM, AutoTokenizer
from deltamem.core.prefix_steer import (iter_steer_modules, set_steer_segments,
    set_write_freeze, clear_frozen_memory, set_steer_enabled, set_mem_cache)
from deltamem.core.global_prefix import SEG_CTX
from deltamem.eval.benchmark_compare import (HOTPOTQA_PROMPT_TEMPLATE, build_hotpotqa_context,
    load_hotpotqa)
from deltamem.eval.locomo_delta import load_locomo_samples
from deltamem.eval.locomo_protocol import (prepare_locomo_question,
    build_official_full_history_messages)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--max-context-tokens", type=int, default=65000)
args = ap.parse_args()

def rms(x): return float(x.float().pow(2).mean().sqrt())

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", local_files_only=True)
_bw = int((torch.load(args.ckpt, map_location="cpu").get("args", {}) or {}).get("backbone_window", 0) or 0)
_kw = {}
if _bw > 0:
    from transformers import AutoConfig
    _bc = AutoConfig.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", local_files_only=True)
    _tc = _bc.get_text_config() if hasattr(_bc, "get_text_config") else _bc
    _tc.sliding_window = _bw; _tc.layer_types = ["sliding_attention"] * _tc.num_hidden_layers
    _kw["config"] = _bc; print(f"[diag] BOUNDED backbone window={_bw}")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16,
    attn_implementation="sdpa", local_files_only=True, **_kw).to("cuda").eval()
cfg = load_ours(m, args.ckpt); m.eval()
assert cfg.pool_reads, "need a pool_reads (with-prefix) ckpt"
mods = list(iter_steer_modules(m)); LAY = list(cfg.steer_layers)
set_steer_enabled(m, True)

def setseg(L):
    set_steer_segments(m, torch.full((1, L), SEG_CTX, dtype=torch.long, device="cuda"),
                       torch.ones(1, L, dtype=torch.bool, device="cuda"))

def decomp(ids):
    """run one steered forward over `ids`, capture prefix decomposition @last pos per layer."""
    for x in mods: x._debug_read = True; x._debug_write = True
    setseg(ids.shape[1])
    with torch.no_grad(): m(input_ids=ids, use_cache=False)
    for x in mods: x._debug_read = False; x._debug_write = False
    row = []
    for x in mods:
        a  = x._dbg_RP.float()[0, :, -1, :]           # [h,P] alphas @answer
        Rt = x._dbg_RH.float()[0, -1]; Mt = x._dbg_pooled.float()[0, -1]
        prR = x._dbg_pref_read.float()[0, -1]
        wp = getattr(x, "_dbg_written", None)         # written prefix [P,hidden] if captured
        row.append(dict(prR=rms(prR)/max(rms(Rt),1e-9), mass=float(a.sum(-1).mean()),
                        MR=rms(Mt)/max(rms(Rt),1e-9),
                        wp=(wp.float()[0].flatten() if wp is not None else None)))
    return row

def run(name, items, build_ids, do_write):
    agg = [{"prR":[], "mass":[], "MR":[]} for _ in mods]
    wp_by_layer = [[] for _ in mods]                   # written-prefix vectors across docs
    for it in items:
        clear_frozen_memory(m)
        ids = build_ids(it)
        if ids is None or ids.shape[1] < 8: continue
        if do_write is not None: do_write(it, ids)
        r = decomp(ids)
        for k in range(len(mods)):
            for kk in ("prR","mass","MR"): agg[k][kk].append(r[k][kk])
            if r[k]["wp"] is not None: wp_by_layer[k].append(r[k]["wp"])
        clear_frozen_memory(m)
    print(f"\n=== {name}  (n={len(agg[0]['prR'])} docs) ===")
    print(f"{'lay':>4}{'prefR/R':>10}{'pfx_mass':>11}{'M/R':>9}{'cos_across':>12}")
    for k in range(len(mods)):
        cos = float('nan')
        W = wp_by_layer[k]
        if len(W) >= 2:
            M = torch.stack(W); M = M / M.norm(dim=1, keepdim=True).clamp_min(1e-9)
            S = M @ M.t(); n = S.shape[0]
            cos = float((S.sum() - n) / (n*(n-1)))     # mean off-diagonal cosine
        print(f"{LAY[k]:>4}{st.mean(agg[k]['prR']):>10.4f}{st.mean(agg[k]['mass']):>11.2e}"
              f"{st.mean(agg[k]['MR']):>9.4f}{cos:>12.4f}")
    return wp_by_layer

# ---------- HotpotQA (SHORT ctx, write-pass ctx like the real eval) ----------
hq = load_hotpotqa(cache_dir=Path.home()/".cache/huggingface/datasets", max_samples=args.n,
                   seed=42, local_files_only=True)
def hq_ids(it):
    ctx = build_hotpotqa_context(it); q = str(it["question"]).strip()
    p = HOTPOTQA_PROMPT_TEMPLATE.format(context=ctx, question=q)
    return tok.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True,
                                   return_tensors="pt", return_dict=True)["input_ids"].to("cuda")
def hq_write(it, ids):
    ctx = build_hotpotqa_context(it)
    ci = tok(ctx, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
    set_write_freeze(m, True); setseg(ci.shape[1])
    with torch.no_grad(): m(input_ids=ci, use_cache=False)
    set_write_freeze(m, False)
run("HotpotQA (short, write-pass ctx)", hq, hq_ids, hq_write)

# ---------- LoCoMo (LONG ctx, NO write pass -- dynamic write during forward, exactly like eval) ----------
samples = load_locomo_samples(Path("data/locomo10.json"), categories=[1,2,3,4],
                              max_conversations=None, max_questions_per_conversation=None)
lc_items = []
for s in samples:
    for qi, qa in enumerate(s["qa"]):
        lc_items.append((s, qa, qi))
        if len(lc_items) >= args.n*3: break
    if len(lc_items) >= args.n*3: break
def lc_ids(it):
    s, qa, qi = it
    spec = prepare_locomo_question(qa, sample_id=s.get("sample_id","s"), question_index=qi, seed=42)
    msgs = build_official_full_history_messages(s, tok, spec, max_context_tokens=args.max_context_tokens,
                                                answer_reserve_tokens=64)
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    return enc["input_ids"].to("cuda")
run("LoCoMo (long, dynamic write in-forward = real eval)", lc_items[:args.n], lc_ids, None)
