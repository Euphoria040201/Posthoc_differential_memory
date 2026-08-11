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
def load(ckpt):
    ck=torch.load(ckpt,map_location="cpu"); cfg=PrefixSteerConfig(**{k:(tuple(v) if isinstance(v,list) else v) for k,v in ck["cfg"].items()})
    m=AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507",dtype=torch.float32,attn_implementation="sdpa",local_files_only=True)
    attach_prefix_steer(m,cfg); freeze_backbone_keep_steer(m); m.load_state_dict(ck["state"],strict=False); m.eval(); set_steer_enabled(m,True)
    return m,list(iter_steer_modules(m)),list(cfg.steer_layers),cfg.steer_gain
def setseg(m,L): set_steer_segments(m,torch.full((1,L),SEG_CTX,dtype=torch.long),torch.ones(1,L,dtype=torch.bool))
def probe(tag,ckpt):
    m,mods,LAY,gain=load(ckpt)
    agg=[{"Vp":[], "prR":[], "read":[], "dO":[]} for _ in mods]
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
        for k,x in enumerate(mods):
            Vp=x._dbg_Vp.float()[0]              # [h,P,hd] prefix values
            Rt=x._dbg_RH.float()[0,-1]           # 末token的SWA读 R_t
            prR=x._dbg_pref_read.float()[0,-1]   # prefix加权贡献
            agg[k]["Vp"].append(rms(Vp))
            agg[k]["read"].append(rms(Rt))
            agg[k]["prR"].append(rms(prR)/max(rms(Rt),1e-9))
            with torch.no_grad():
                dO=x.delta_o((Rt).to(torch.float32)[None,None,:])   # 注入修正(未乘gain)
            agg[k]["dO"].append(rms(dO)*gain)
        clear_frozen_memory(m)
    import statistics as st
    print(f"\n=== {tag} (gain={gain}) ===")
    print(f"{'层':>4}{'Vp_RMS':>9}{'read_RMS':>10}{'prefR/R':>9}{'注入|gain*dO|':>13}")
    for k in range(len(mods)):
        r={kk:st.mean(vv) for kk,vv in agg[k].items()}
        print(f"{LAY[k]:>4}{r['Vp']:>9.3f}{r['read']:>10.3f}{r['prR']:>9.4f}{r['dO']:>13.4f}")
    del m
probe("冠军(0.6815)","ckpts/spread12_swa_ckpt.pt")
print("DONE",flush=True)
