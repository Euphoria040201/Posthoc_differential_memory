import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from deltamem.core.prefix_steer import (PrefixSteerConfig, attach_prefix_steer, freeze_backbone_keep_steer,
    set_steer_segments, set_write_freeze, clear_frozen_memory, set_steer_enabled, iter_steer_modules)
from deltamem.core.global_prefix import SEG_CTX
from deltamem.eval.benchmark_compare import HOTPOTQA_PROMPT_TEMPLATE, build_hotpotqa_context, load_hotpotqa
torch.set_num_threads(16)
tok=AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507",local_files_only=True)
data=load_hotpotqa(cache_dir=Path.home()/".cache/huggingface/datasets",max_samples=2,seed=42,local_files_only=True)
def rms(x): return float(x.float().pow(2).mean().sqrt())
# seed -> (ckpt, F1)
MODELS=[("seed0冠军",0.6815,"ckpts/spread12_swa_ckpt.pt"),
        ("ms1",0.6679,"ckpts/s0d_ms1_swa_ckpt.pt"),
        ("ms2",0.6625,"ckpts/s0d_ms2_swa_ckpt.pt"),
        ("ms4",0.6616,"ckpts/s0d_ms4_swa_ckpt.pt"),
        ("ms5",0.6371,"ckpts/s0d_ms5_swa_ckpt.pt"),
        ("ms6",0.6352,"ckpts/s0d_ms6_swa_ckpt.pt"),
        ("ms3最差",0.6180,"ckpts/s0d_ms3_swa_ckpt.pt")]
def setseg(m,L): set_steer_segments(m,torch.full((1,L),SEG_CTX,dtype=torch.long),torch.ones(1,L,dtype=torch.bool))
def probe(ckpt):
    ck=torch.load(ckpt,map_location="cpu"); cfg=PrefixSteerConfig(**{k:(tuple(v) if isinstance(v,list) else v) for k,v in ck["cfg"].items()})
    m=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507",dtype=torch.float32,attn_implementation="sdpa",local_files_only=True)
    attach_prefix_steer(m,cfg); freeze_backbone_keep_steer(m); m.load_state_dict(ck["state"],strict=False); m.eval(); set_steer_enabled(m,True)
    mods=list(iter_steer_modules(m)); LAY=list(cfg.steer_layers); i33=LAY.index(33)
    prR33=[]; inj33=[]; deepmax=[]
    for it in data:
        ctx=build_hotpotqa_context(it); q=str(it["question"]).strip()
        ci=tok(ctx,add_special_tokens=False,return_tensors="pt")["input_ids"]
        clear_frozen_memory(m); set_write_freeze(m,True); setseg(m,ci.shape[1])
        with torch.no_grad(): m(input_ids=ci,use_cache=False)
        set_write_freeze(m,False)
        p=HOTPOTQA_PROMPT_TEMPLATE.format(context=ctx,question=q)
        ids=tok.apply_chat_template([{"role":"user","content":p}],add_generation_prompt=True,return_tensors="pt",return_dict=True)["input_ids"]
        for x in mods: x._debug_read=True
        setseg(m,ids.shape[1])
        with torch.no_grad(): m(input_ids=ids,use_cache=False)
        for x in mods: x._debug_read=False
        x=mods[i33]; Rt=x._dbg_RH.float()[0,-1]; prR=x._dbg_pref_read.float()[0,-1]
        prR33.append(rms(prR)/max(rms(Rt),1e-9))
        with torch.no_grad(): inj33.append(rms(x.delta_o(Rt[None,None,:]))*cfg.steer_gain)
        # 深层(>=12) 最大 prefix mass
        dm=max(mods[LAY.index(L)]._dbg_RP.float()[0,:,-1,:].sum(-1).mean().item() for L in LAY if L>=12)
        deepmax.append(dm)
        clear_frozen_memory(m)
    import statistics as st
    del m
    return st.mean(prR33), st.mean(inj33), st.mean(deepmax)
print(f"{'seed':>10}{'F1':>8}{'L33 prefR/R':>13}{'L33注入':>10}{'深层maxMass':>12}",flush=True)
for name,f1,ck in MODELS:
    a,b,c=probe(ck)
    print(f"{name:>10}{f1:>8.4f}{a:>13.4f}{b:>10.4f}{c:>12.4f}",flush=True)
print("DONE",flush=True)
