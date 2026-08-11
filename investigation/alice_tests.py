"""Minimal-pair unit tests for the prefix-memory write->read path (Doc A: Alice lives in
Paris / Doc B: Alice lives in Tokyo, SAME question). Isolates WHERE document information
dies:
  T1 writer  : rel_diff(M_A, M_B) per layer, untrained AND trained ckpt. ~0 => writer/swap bug.
  T2 reader  : hand-set _frozen_prefix = +c / -c, same question-only prompt; logits must move.
  T3 overfit : train steer params on the 2-sample task (noctx write->drop->read). The ONLY
               way to answer differently is memory content. Can't overfit => real bottleneck.
  T3b manual : same overfit but _frozen_prefix is hand-set (bypasses writer). Works while T3
               fails => WRITER bottleneck; also fails => READOUT bottleneck.
"""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from transformers import AutoModelForCausalLM, AutoTokenizer
from deltamem.core.prefix_steer import (PrefixSteerConfig, attach_prefix_steer,
    freeze_backbone_keep_steer, set_steer_segments, set_steer_enabled,
    set_write_freeze, clear_frozen_memory, iter_steer_modules)
from deltamem.core.global_prefix import SEG_CTX, SEG_QRY, SEG_ANS

DEV = "cuda"
SYS = "Answer the question using the context. Give a short answer."
DOC_A = "Context:\nAlice lives in Paris. She works at a small bakery near the old bridge."
DOC_B = "Context:\nAlice lives in Tokyo. She works at a small bakery near the old bridge."
QUES = f"\n\n{SYS}\nQuestion: Where does Alice live?\nAnswer:"

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", local_files_only=True)
A_IDS = tok(DOC_A, add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEV)
B_IDS = tok(DOC_B, add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEV)
Q_IDS = tok(QUES, add_special_tokens=False)["input_ids"]
ANS = {"A": tok(" Paris", add_special_tokens=False)["input_ids"],
       "B": tok(" Tokyo", add_special_tokens=False)["input_ids"]}

CFG = PrefixSteerConfig(num_prefix_tokens=64, sliding_window_size=256, mem_num_heads=1,
    mem_head_dim=64, steer_mode="deltamem", prefix_write=True, memory_mode="dynamic",
    write_ctx_only=True, pool_reads=True, steer_gain=0.1, delta_heads="qkvo",
    steer_layers=(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33))


def fresh(ckpt=None):
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507",
        dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).to(DEV)
    if ckpt:
        from eval_ours_hotpotqa import load_ours
        load_ours(m, ckpt)
    else:
        attach_prefix_steer(m, CFG); freeze_backbone_keep_steer(m)
    return m.eval()


def write(m, cids):
    clear_frozen_memory(m); set_write_freeze(m, True)
    set_steer_segments(m, torch.full_like(cids, SEG_CTX), torch.ones_like(cids, dtype=torch.bool))
    with torch.no_grad():
        m(input_ids=cids, use_cache=False)
    set_write_freeze(m, False)


def grab_mem(m):
    return [x._frozen_prefix.detach().float().clone() for x in iter_steer_modules(m)]


def rel_diff(a, b):
    return float((a - b).norm() / a.norm().clamp_min(1e-8))


def qonly_logits(m):
    ids = torch.tensor([Q_IDS], device=DEV)
    seg = torch.full_like(ids, SEG_QRY)
    set_steer_segments(m, seg, torch.ones_like(ids, dtype=torch.bool))
    with torch.no_grad():
        return m(input_ids=ids, use_cache=False).logits[0, -1].float()


def gen_qonly(m, n=3):
    ids = list(Q_IDS); seg = [SEG_QRY] * len(ids); out = []
    for _ in range(n):
        t = torch.tensor([ids], device=DEV)
        set_steer_segments(m, torch.tensor([seg], device=DEV), torch.ones_like(t, dtype=torch.bool))
        with torch.no_grad():
            nxt = int(m(input_ids=t, use_cache=False).logits[0, -1].argmax())
        out.append(nxt); ids.append(nxt); seg.append(SEG_ANS)
    return tok.decode(out)


def t1(m, tag):
    write(m, A_IDS); MA = grab_mem(m)
    write(m, B_IDS); MB = grab_mem(m)
    d = [rel_diff(a, b) for a, b in zip(MA, MB)]
    print(f"[T1:{tag}] rel_diff(M_A,M_B) per layer min={min(d):.4f} mean={sum(d)/len(d):.4f} max={max(d):.4f}")
    return MA, MB


def t2(m):
    mods = list(iter_steer_modules(m))
    for sgn, name in [(+1.0, "+c"), (-1.0, "-c")]:
        clear_frozen_memory(m)
        for x in mods:
            x._frozen_prefix = sgn * 0.7 * torch.ones(1, x.cfg.num_prefix_tokens, x.hidden_size,
                                                      dtype=torch.bfloat16, device=DEV)
        lg = qonly_logits(m)
        if sgn > 0: lp = lg
    kl = float(F.kl_div(F.log_softmax(lg, -1), F.softmax(lp, -1), reduction="sum"))
    print(f"[T2] logits(+c) vs (-c): max|diff|={float((lp-lg).abs().max()):.4f} KL={kl:.4f} "
          f"top1_changed={int(lp.argmax()) != int(lg.argmax())}")
    clear_frozen_memory(m)


def overfit(m, manual=False, steps=300, tag="T3"):
    mods = list(iter_steer_modules(m))
    if manual:
        MAN = {k: [torch.randn(1, x.cfg.num_prefix_tokens, x.hidden_size, device=DEV,
                               dtype=torch.bfloat16) * 0.7 for x in mods] for k in ("A", "B")}
    pref = [p for n, p in m.named_parameters() if p.requires_grad and n.endswith(".prefix")]
    rest = [p for n, p in m.named_parameters() if p.requires_grad and not n.endswith(".prefix")]
    opt = torch.optim.AdamW([{"params": rest, "lr": 1e-3}, {"params": pref, "lr": 1e-2}])
    qa = torch.tensor([Q_IDS + ANS["A"]], device=DEV)  # same length for A/B (both 2-token answers)
    for step in range(steps):
        k = "A" if step % 2 == 0 else "B"
        m.train()
        if manual:
            clear_frozen_memory(m)
            for x, mm in zip(mods, MAN[k]):
                x._frozen_prefix = mm
        else:
            cids = A_IDS if k == "A" else B_IDS
            clear_frozen_memory(m); set_write_freeze(m, True)
            set_steer_segments(m, torch.full_like(cids, SEG_CTX), torch.ones_like(cids, dtype=torch.bool))
            m(input_ids=cids, use_cache=False)
            set_write_freeze(m, False)
        ids = torch.tensor([Q_IDS + ANS[k]], device=DEV)
        seg = torch.tensor([[SEG_QRY] * len(Q_IDS) + [SEG_ANS] * len(ANS[k])], device=DEV)
        lab = torch.tensor([[-100] * len(Q_IDS) + ANS[k]], device=DEV)
        set_steer_segments(m, seg, torch.ones_like(ids, dtype=torch.bool))
        loss = m(input_ids=ids, labels=lab, use_cache=False).loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
        opt.step()
        if (step + 1) % 100 == 0 or step == steps - 1:
            m.eval(); outs = {}
            for kk in ("A", "B"):
                if manual:
                    clear_frozen_memory(m)
                    for x, mm in zip(mods, MAN[kk]):
                        x._frozen_prefix = mm
                else:
                    write(m, A_IDS if kk == "A" else B_IDS)
                outs[kk] = gen_qonly(m).strip()
            ok = outs["A"].startswith("Paris") and outs["B"].startswith("Tokyo")
            print(f"[{tag}] step {step+1}: loss={loss.item():.4f} genA={outs['A']!r} genB={outs['B']!r} "
                  f"{'*** DIFFERENTIATED ***' if ok else ''}", flush=True)
            if ok:
                return True
    return False


if __name__ == "__main__":
    print("== untrained model ==")
    m = fresh()
    t1(m, "untrained")
    t2(m)
    print("== trained ckpt (nct_r4) ==")
    del m; torch.cuda.empty_cache()
    m = fresh("out_ctxmask/nct_r4_ckpt.pt")
    t1(m, "nct_r4")
    t2(m)
    print("== T3: 2-sample overfit via WRITER (the decisive test) ==")
    del m; torch.cuda.empty_cache()
    m = fresh()
    ok3 = overfit(m, manual=False, steps=400, tag="T3")
    print("== T3b: 2-sample overfit with MANUAL prefixes (bypasses writer) ==")
    del m; torch.cuda.empty_cache()
    m = fresh()
    ok3b = overfit(m, manual=True, steps=400, tag="T3b")
    print(f"\nVERDICT: writer-overfit={'PASS' if ok3 else 'FAIL'}  manual-overfit={'PASS' if ok3b else 'FAIL'}")
    print("  both PASS  -> implementation fine; real-data failure = capacity/distribution")
    print("  T3 FAIL + T3b PASS -> WRITER bottleneck")
    print("  both FAIL  -> READOUT bottleneck (delta/gain/routing)")
