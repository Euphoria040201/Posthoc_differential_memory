import json, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from deltamem.core.prefix_steer import (PrefixSteerConfig, attach_prefix_steer, freeze_backbone_keep_steer,
    set_steer_segments, set_write_freeze, clear_frozen_memory, set_steer_enabled, iter_steer_modules)
from deltamem.core.global_prefix import SEG_CTX
from deltamem.eval.benchmark_compare import (HOTPOTQA_PROMPT_TEMPLATE, build_hotpotqa_context, load_hotpotqa,
    extract_first_line)
torch.set_num_threads(16)
SCR="."
sel=json.load(open(f"{SCR}/deg_ids.json"))["ids"]
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507",local_files_only=True)
data=load_hotpotqa(cache_dir=Path.home()/".cache/huggingface/datasets",max_samples=100000,seed=42,local_files_only=True)
byid={(it.get("id") or it.get("_id")):it for it in data}
ck=torch.load("ckpts/s0d_ms3_swa_ckpt.pt",map_location="cpu")
cfg=PrefixSteerConfig(**{k:(tuple(v) if isinstance(v,list) else v) for k,v in ck["cfg"].items()})
m=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507",dtype=torch.float32,attn_implementation="sdpa",local_files_only=True)
attach_prefix_steer(m,cfg); freeze_backbone_keep_steer(m); m.load_state_dict(ck["state"],strict=False); m.eval(); set_steer_enabled(m,True)
mods=list(iter_steer_modules(m)); LAY=list(cfg.steer_layers); l33=mods[LAY.index(33)]
def setseg(L): set_steer_segments(m,torch.full((1,L),SEG_CTX,dtype=torch.long),torch.ones(1,L,dtype=torch.bool))
def run(ctx,q,mask,max_new=8):
    ci=tok(ctx,add_special_tokens=False,return_tensors="pt")["input_ids"]
    clear_frozen_memory(m); set_write_freeze(m,True); setseg(ci.shape[1])
    with torch.no_grad(): m(input_ids=ci,use_cache=False)
    set_write_freeze(m,False); l33._window_only=mask
    p=HOTPOTQA_PROMPT_TEMPLATE.format(context=ctx,question=q)
    ids=tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True,return_tensors="pt",return_dict=True)["input_ids"]
    out=[]
    for _ in range(max_new):
        setseg(ids.shape[1])
        with torch.no_grad(): lg=m(input_ids=ids,use_cache=False).logits[0,-1]
        nx=int(lg.argmax())
        if nx==tok.eos_token_id: break
        out.append(nx); ids=torch.cat([ids,torch.tensor([[nx]])],1)
    l33._window_only=False; clear_frozen_memory(m)
    return extract_first_line(tok.decode(out,skip_special_tokens=True))
print("=== 最差模型在退化题上: full vs mask-L33 ===",flush=True)
for iid in sel:
    it=byid[iid]; ctx=build_hotpotqa_context(it); q=str(it["question"]).strip(); g=str(it["answer"]).strip()
    full=run(ctx,q,False); masked=run(ctx,q,True)
    chg="★变了★" if full.strip()!=masked.strip() else "同"
    print(f"gold={g[:22]!r:24} full={full[:22]!r:24} maskL33={masked[:22]!r:24} [{chg}]",flush=True)
print("DONE",flush=True)
