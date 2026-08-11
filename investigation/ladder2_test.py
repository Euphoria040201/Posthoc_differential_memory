"""Ambiguity-corrected capacity ladder. K names x 2 city-variants each: the question
("Where does {name} live?") is UNANSWERABLE from weights -- both variants are trained
equally often, so accuracy above 50% is attributable ONLY to the written memory.
This fixes ladder_test.py, whose per-doc unique names let the delta weights memorize the
name->city mapping (swap diag showed question-driven=1.0). Usage: ladder2_test.py K [steps] [lr]."""
import sys, random, torch
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from transformers import AutoModelForCausalLM, AutoTokenizer
from deltamem.core.prefix_steer import (PrefixSteerConfig, attach_prefix_steer,
    freeze_backbone_keep_steer, set_steer_segments, set_write_freeze, clear_frozen_memory)
from deltamem.core.global_prefix import SEG_CTX, SEG_QRY, SEG_ANS

K = int(sys.argv[1]); MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 100 * K
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 3e-4
TSEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
torch.manual_seed(TSEED)   # attach init is torch-RNG-dependent; solvability is init-sensitive
DEV = "cuda"
SYS = "Answer the question using the context. Give a short answer."
rng = random.Random(11)
FIRST = ["Alice","Bob","Carol","David","Emma","Frank","Grace","Henry","Ivy","Jack","Kate","Leo",
         "Mia","Noah","Olga","Paul","Quinn","Rosa","Sam","Tina","Uma","Vera","Will","Xena","Yuri","Zoe"]
LAST = ["Anders","Brooks","Chen","Diaz","Evans","Fischer","Garcia","Hayes","Ito","Jones","Kim","Lopez",
        "Meyer","Novak","Owens","Park","Qureshi","Rossi","Silva","Tanaka","Ueda","Vogel","Weber","Xu"]
CITY = ["Paris","Tokyo","Cairo","Lima","Oslo","Delhi","Rome","Seoul","Quito","Hanoi","Lagos","Berlin",
        "Madrid","Athens","Dublin","Vienna","Prague","Havana","Nairobi","Manila","Bogota","Warsaw",
        "Lisbon","Helsinki","Brussels","Zagreb","Riga","Tunis","Amman","Dakar","Accra","Kyiv"]
names = rng.sample([f"{a} {b}" for a in FIRST for b in LAST], K)
pairs = [rng.sample(CITY, 2) for _ in range(K)]           # 2 distinct cities per name

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", local_files_only=True)
DOC, Q, A, CT = {}, {}, {}, {}
for k in range(K):
    for v in (0, 1):
        ct = pairs[k][v]
        DOC[(k, v)] = tok(f"Context:\n{names[k]} lives in {ct}. They work at a small bakery "
                          f"near the old bridge.", add_special_tokens=False,
                          return_tensors="pt")["input_ids"].to(DEV)
        A[(k, v)] = tok(" " + ct, add_special_tokens=False)["input_ids"]
        CT[(k, v)] = ct
    Q[k] = tok(f"\n\n{SYS}\nQuestion: Where does {names[k]} live?\nAnswer:",
               add_special_tokens=False)["input_ids"]

CFG = PrefixSteerConfig(num_prefix_tokens=64, sliding_window_size=256, mem_num_heads=1,
    mem_head_dim=64, steer_mode="deltamem", prefix_write=True, memory_mode="dynamic",
    write_ctx_only=True, pool_reads=True, steer_gain=0.1, delta_heads="qkvo",
    steer_layers=(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33))
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16,
    attn_implementation="sdpa", local_files_only=True).to(DEV)
attach_prefix_steer(m, CFG); freeze_backbone_keep_steer(m)

def write(k, v):
    cids = DOC[(k, v)]
    clear_frozen_memory(m); set_write_freeze(m, True)
    set_steer_segments(m, torch.full_like(cids, SEG_CTX), torch.ones_like(cids, dtype=torch.bool))
    m(input_ids=cids, use_cache=False)
    set_write_freeze(m, False)

def gen(k, ntok=3):
    ids = list(Q[k]); seg = [SEG_QRY] * len(ids); out = []
    for _ in range(ntok):
        t = torch.tensor([ids], device=DEV)
        set_steer_segments(m, torch.tensor([seg], device=DEV), torch.ones_like(t, dtype=torch.bool))
        with torch.no_grad():
            nx = int(m(input_ids=t, use_cache=False).logits[0, -1].argmax())
        out.append(nx); ids.append(nx); seg.append(SEG_ANS)
    return tok.decode(out).strip()

def evaluate(kmax=32):
    ks = list(range(K)) if K <= kmax else rng.sample(range(K), kmax)
    m.eval(); hit = tot = 0
    with torch.no_grad():
        for k in ks:
            for v in (0, 1):
                write(k, v); tot += 1
                if gen(k).startswith(CT[(k, v)]): hit += 1
    clear_frozen_memory(m)
    return hit / tot                                       # chance = ~0.5 (weights can't disambiguate)

pref = [p for n, p in m.named_parameters() if p.requires_grad and n.endswith(".prefix")]
rest = [p for n, p in m.named_parameters() if p.requires_grad and not n.endswith(".prefix")]
opt = torch.optim.AdamW([{"params": rest, "lr": LR}, {"params": pref, "lr": 10 * LR}])
EV = max(100, MAX_STEPS // 10)
print(f"[ladder2 K={K} ({2*K} docs)] steps={MAX_STEPS} lr={LR} eval_every={EV} chance=0.5", flush=True)
for step in range(MAX_STEPS):
    k, v = rng.randrange(K), rng.randrange(2)
    m.train(); write(k, v)
    ids = torch.tensor([Q[k] + A[(k, v)]], device=DEV)
    seg = torch.tensor([[SEG_QRY] * len(Q[k]) + [SEG_ANS] * len(A[(k, v)])], device=DEV)
    lab = torch.tensor([[-100] * len(Q[k]) + A[(k, v)]], device=DEV)
    set_steer_segments(m, seg, torch.ones_like(ids, dtype=torch.bool))
    loss = m(input_ids=ids, labels=lab, use_cache=False).loss
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
    opt.step()
    if (step + 1) % EV == 0 or step == MAX_STEPS - 1:
        acc = evaluate()
        print(f"[ladder2 K={K}] step {step+1}: loss={loss.item():.4f} variant_acc={acc:.3f}", flush=True)
        if acc >= 0.95:
            print(f"[ladder2 K={K}] SOLVED at step {step+1}", flush=True)
            break

# ---- explicit ablations on the solved model (all should collapse to ~0.5 chance if the
# variant flip is carried by the prefix READ of the written memory and nothing else) ----
if evaluate() < 0.8:
    print("[ablate] SKIPPED: model did not solve the task -- ablations on a broken model "
          "are uninformative", flush=True)
    sys.exit(0)
from deltamem.core.prefix_steer import set_window_only
m.eval()
with torch.no_grad():
    ks = list(range(K))
    # (a) NO-WRITE: clear memory, question only -> no way to know the variant
    hit = 0
    for k in ks:
        clear_frozen_memory(m)
        o = gen(k)
        hit += o.startswith(CT[(k, 0)]) or o.startswith(CT[(k, 1)])  # says either city at all?
    print(f"[ablate] no-write: says-one-of-the-two-cities rate={hit/len(ks):.3f} "
          f"(variant acc undefined without a write; chance a fixed guess matches a given "
          f"variant = 0.5)", flush=True)
    # (a') NO-WRITE variant acc: score fixed no-memory output against each variant
    hit = tot = 0
    for k in ks:
        clear_frozen_memory(m)
        o = gen(k)
        for v in (0, 1):
            tot += 1
            hit += o.startswith(CT[(k, v)])
    print(f"[ablate] no-write variant_acc={hit/tot:.3f}  (expect ~0.5 if memory carried it)", flush=True)
    # (b) WINDOW-ONLY: write normally but mask the prefix out of the READ
    set_window_only(m, True)
    hit = tot = 0
    for k in ks:
        for v in (0, 1):
            write(k, v); tot += 1
            hit += gen(k).startswith(CT[(k, v)])
    set_window_only(m, False)
    print(f"[ablate] window_only variant_acc={hit/tot:.3f}  (expect ~0.5: prefix read is the carrier)", flush=True)
    # (c) CROSS-NAME swap: write doc(k2,v), ask about k1 -> does output track the WRITTEN city?
    trk_w = trk_q = tot = 0
    for k in ks:
        k2 = (k + 1) % K
        for v in (0, 1):
            write(k2, v); tot += 1
            o = gen(k)
            if o.startswith(CT[(k2, v)]): trk_w += 1
            elif o.startswith(CT[(k, 0)]) or o.startswith(CT[(k, 1)]): trk_q += 1
    print(f"[ablate] cross-name: tracks-WRITTEN-city={trk_w/tot:.3f} tracks-questioned-name={trk_q/tot:.3f}", flush=True)
    clear_frozen_memory(m)
print(f"[ladder2 K={K}] DONE", flush=True)
