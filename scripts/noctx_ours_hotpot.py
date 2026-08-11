"""NO-CONTEXT test of our prefix-steer config: WRITE the doc into the prefix memory, then DROP
the context from the answer prompt (question only) and read from the frozen prefix. If the write
stored doc content, ours_noctx should beat base_noctx (the backbone's parametric HotpotQA prior).
Prediction from diag (cos_across=1.0 doc-independent): ~no lift.
Conditions: base_ctx (sanity), ours_ctx (sanity, memory+context), base_noctx (parametric floor),
ours_noctx (write->drop->read frozen prefix).
"""
import torch, sys, argparse, statistics as st
from pathlib import Path
sys.path.insert(0, "scripts")
from eval_ours_hotpotqa import load_ours
from transformers import AutoModelForCausalLM, AutoTokenizer
from deltamem.core.prefix_steer import (set_steer_segments, set_write_freeze,
    clear_frozen_memory, set_steer_enabled, set_window_only)
from deltamem.core.global_prefix import SEG_CTX
from deltamem.eval.benchmark_compare import (HOTPOTQA_PROMPT_TEMPLATE, build_hotpotqa_context,
    load_hotpotqa, hotpotqa_f1, extract_first_line)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--n", type=int, default=200)
ap.add_argument("--gold-write", action="store_true",
                help="write ONLY the 2 gold supporting passages (~250 tok) instead of the "
                     "full 10-passage context -- matches the gold-evidence training "
                     "distribution and removes the write-length shift as a confound")
args = ap.parse_args()


def build_gold_context(item):
    gold = set(item["supporting_facts"]["title"])
    ctx = item["context"]
    parts = [f"Passage {i} - {t}:\n" + " ".join(str(s).strip() for s in ss if str(s).strip())
             for i, (t, ss) in enumerate(zip(ctx["title"], ctx["sentences"]), start=1) if t in gold]
    return "\n\n".join(parts) if parts else "No passages provided."

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", local_files_only=True)
_bw = int((torch.load(args.ckpt, map_location="cpu").get("args", {}) or {}).get("backbone_window", 0) or 0)
_kw = {}
if _bw > 0:
    from transformers import AutoConfig
    _bc = AutoConfig.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", local_files_only=True)
    _tc = _bc.get_text_config() if hasattr(_bc, "get_text_config") else _bc
    _tc.sliding_window = _bw; _tc.layer_types = ["sliding_attention"] * _tc.num_hidden_layers
    _kw["config"] = _bc
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16,
    attn_implementation="sdpa", local_files_only=True, **_kw).to("cuda").eval()
cfg = load_ours(m, args.ckpt); m.eval()
data = load_hotpotqa(cache_dir=Path.home()/".cache/huggingface/datasets", max_samples=args.n,
                     seed=42, local_files_only=True)

def setseg(L):
    set_steer_segments(m, torch.full((1, L), SEG_CTX, dtype=torch.long, device="cuda"),
                       torch.ones(1, L, dtype=torch.bool, device="cuda"))

def write(ctx):
    ci = tok(ctx, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda")
    clear_frozen_memory(m); set_write_freeze(m, True); setseg(ci.shape[1])
    with torch.no_grad(): m(input_ids=ci, use_cache=False)
    set_write_freeze(m, False)

def gen(ctx_text, q, steer, max_new=24):
    set_steer_enabled(m, steer)
    p = HOTPOTQA_PROMPT_TEMPLATE.format(context=ctx_text, question=q)
    ids = tok.apply_chat_template([{"role":"user","content":p}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)["input_ids"].to("cuda")
    out = []
    for _ in range(max_new):  # no-cache greedy: mem read rebuilds its mask each full forward
        setseg(ids.shape[1])
        with torch.no_grad(): lg = m(input_ids=ids, use_cache=False).logits[0, -1]
        nxt = int(lg.argmax())
        if nxt == tok.eos_token_id: break
        out.append(nxt); ids = torch.cat([ids, torch.tensor([[nxt]], device="cuda")], dim=1)
    return extract_first_line(tok.decode(out, skip_special_tokens=True))

R = {"base_ctx":[], "ours_ctx":[], "base_noctx":[], "ours_noctx":[], "wo_noctx":[], "swap_noctx":[]}
wsrc = build_gold_context if args.gold_write else build_hotpotqa_context
prev_ctx = wsrc(data[-1])                     # rotating wrong-doc source (last sample's ctx)
for it in data:
    ctx = build_hotpotqa_context(it); q = str(it["question"]).strip(); gold = str(it["answer"]).strip()
    wctx = wsrc(it)                            # WRITE source (gold-only when --gold-write)
    # no-context conditions: write doc to prefix ONCE, then answer with EMPTY context.
    # ours_noctx = prefix+steer; wo_noctx = SAME written memory but the prefix is masked
    # out of the READ (window-only SWA = steer without memory access) -> off-distribution
    # intervention. swap_noctx = memory written from a DIFFERENT sample's doc (previous
    # item), everything else identical -> ON-distribution control; the paired
    # ours_noctx - swap_noctx is the DOC-SPECIFIC value of the written memory, immune to
    # the trained answer-style prior that contaminates the window_only comparison.
    write(wctx)
    R["ours_noctx"].append(hotpotqa_f1(gen("", q, True), gold))
    set_window_only(m, True)
    R["wo_noctx"].append(hotpotqa_f1(gen("", q, True), gold))
    set_window_only(m, False)
    assert prev_ctx != wctx, "swap source is the SAME document"
    write(prev_ctx)                            # WRONG document's memory
    R["swap_noctx"].append(hotpotqa_f1(gen("", q, True), gold))
    clear_frozen_memory(m)
    R["base_noctx"].append(hotpotqa_f1(gen("", q, False), gold))
    prev_ctx = wctx
    # context conditions (sanity)
    write(ctx)
    R["ours_ctx"].append(hotpotqa_f1(gen(ctx, q, True), gold))
    clear_frozen_memory(m)
    R["base_ctx"].append(hotpotqa_f1(gen(ctx, q, False), gold))

def paired(a, b):
    d = [x - y for x, y in zip(R[a], R[b])]
    mu = st.mean(d); sd = st.stdev(d) if len(d) > 1 else 0.0
    t = mu / (sd / len(d) ** 0.5) if sd > 0 else float("inf")
    return mu, t

print(f"\n=== NO-CONTEXT test  (n={len(data)})  ckpt={Path(args.ckpt).name} bw={_bw} ===")
for k in ["base_ctx","ours_ctx","base_noctx","wo_noctx","swap_noctx","ours_noctx"]:
    print(f"  {k:>12}  F1={st.mean(R[k]):.4f}")
for a, b, tag in [("ours_noctx","base_noctx","prefix+steer - base"),
                  ("wo_noctx","base_noctx","steer-only  - base"),
                  ("ours_noctx","wo_noctx","prefix marginal(wo)"),
                  ("ours_noctx","swap_noctx","DOC-SPECIFIC memory")]:
    mu, t = paired(a, b)
    print(f"  >>> {tag} = {mu:+.4f}  (paired t={t:+.2f})")
