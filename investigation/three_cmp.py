import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from deltamem.core.prefix_steer import (PrefixSteerConfig, attach_prefix_steer, freeze_backbone_keep_steer,
    set_steer_segments, set_write_freeze, clear_frozen_memory, set_steer_enabled, iter_steer_modules)
from deltamem.core.global_prefix import SEG_CTX
from deltamem.eval.benchmark_compare import (HOTPOTQA_PROMPT_TEMPLATE, build_hotpotqa_context, load_hotpotqa,
    extract_first_line)
torch.set_num_threads(16)
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507",local_files_only=True)
data=load_hotpotqa(cache_dir=Path.home()/".cache/huggingface/datasets",max_samples=5,seed=42,local_files_only=True)
def load(ckpt):
    ck=torch.load(ckpt,map_location="cpu"); cfg=PrefixSteerConfig(**{k:(tuple(v) if isinstance(v,list) else v) for k,v in ck["cfg"].items()})
    m=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507",dtype=torch.float32,attn_implementation="sdpa",local_files_only=True)
    attach_prefix_steer(m,cfg); freeze_backbone_keep_steer(m); m.load_state_dict(ck["state"],strict=False); m.eval(); set_steer_enabled(m,True)
    mods=list(iter_steer_modules(m)); LAY=list(cfg.steer_layers)
    return m,mods[LAY.index(33)]
def setseg(m,L): set_steer_segments(m,torch.full((1,L),SEG_CTX,dtype=torch.long),torch.ones(1,L,dtype=torch.bool))
def run(m,l33,ctx,q,mask_l33,max_new=6):
    ci=tok(ctx,add_special_tokens=False,return_tensors="pt")["input_ids"]
    clear_frozen_memory(m); set_write_freeze(m,True); setseg(m,ci.shape[1])
    with torch.no_grad(): m(input_ids=ci,use_cache=False)
    set_write_freeze(m,False)
    l33._window_only=mask_l33
    p=HOTPOTQA_PROMPT_TEMPLATE.format(context=ctx,question=q)
    ids=tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True,return_tensors="pt",return_dict=True)["input_ids"]
    out=[]
    for _ in range(max_new):
        setseg(m,ids.shape[1])
        with torch.no_grad(): lg=m(input_ids=ids,use_cache=False).logits[0,-1]
        nx=int(lg.argmax())
        if nx==tok.eos_token_id: break
        out.append(nx); ids=torch.cat([ids,torch.tensor([[nx]])],1)
    l33._window_only=False; clear_frozen_memory(m)
    return extract_first_line(tok.decode(out,skip_special_tokens=True))
print("加载冠军(0.6815)...",flush=True); mB,l33B=load("ckpts/spread12_swa_ckpt.pt")
best=[run(mB,l33B,build_hotpotqa_context(it),str(it["question"]).strip(),False) for it in data]; del mB
print("加载最差(ms3=0.618)...",flush=True); mW,l33W=load("ckpts/s0d_ms3_swa_ckpt.pt")
worst=[run(mW,l33W,build_hotpotqa_context(it),str(it["question"]).strip(),False) for it in data]
worst_no33=[run(mW,l33W,build_hotpotqa_context(it),str(it["question"]).strip(),True) for it in data]; del mW
print("\n=== 三条件原始输出对比 ===")
for i,it in enumerate(data):
    g=str(it["answer"]).strip()
    same="同" if worst[i]==worst_no33[i] else "★变了★"
    print(f"\ndoc{i} gold={g!r}")
    print(f"  冠军      : {best[i]!r}")
    print(f"  最差      : {worst[i]!r}")
    print(f"  最差-noL33: {worst_no33[i]!r}   [最差 vs 最差-noL33: {same}]")
print("DONE",flush=True)
