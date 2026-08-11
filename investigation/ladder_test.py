"""N-doc capacity ladder: at what scale does doc-specific memory die?
Synthetic minimal docs ("{Name} lives in {City}."), noctx write->drop->read training on N
docs. Reports, per eval: correct-write accuracy (write doc_i, ask about name_i -> city_i)
and swap behavior (write doc_j: how often the output tracks the WRITTEN doc's city =
memory-driven, vs the QUESTIONED name's city = leaked-into-weights). N=2 already passes
(alice_tests). Usage: ladder_test.py N [max_steps]."""
import sys, random, torch
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from transformers import AutoModelForCausalLM, AutoTokenizer
from deltamem.core.prefix_steer import (PrefixSteerConfig, attach_prefix_steer,
    freeze_backbone_keep_steer, set_steer_segments, set_write_freeze,
    clear_frozen_memory)
from deltamem.core.global_prefix import SEG_CTX, SEG_QRY, SEG_ANS

N = int(sys.argv[1]); MAX_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 40 * N
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
PLR = float(sys.argv[4]) if len(sys.argv) > 4 else 10 * LR
DEV = "cuda"
SYS = "Answer the question using the context. Give a short answer."
rng = random.Random(7)
FIRST = ["Alice","Bob","Carol","David","Emma","Frank","Grace","Henry","Ivy","Jack","Kate","Leo",
         "Mia","Noah","Olga","Paul","Quinn","Rosa","Sam","Tina","Uma","Vera","Will","Xena","Yuri","Zoe"]
LAST = ["Anders","Brooks","Chen","Diaz","Evans","Fischer","Garcia","Hayes","Ito","Jones","Kim","Lopez",
        "Meyer","Novak","Owens","Park","Qureshi","Rossi","Silva","Tanaka","Ueda","Vogel","Weber","Xu","Yang","Zhang"]
CITY = ["Paris","Tokyo","Cairo","Lima","Oslo","Delhi","Rome","Seoul","Quito","Hanoi","Lagos","Berlin",
        "Madrid","Athens","Dublin","Vienna","Prague","Havana","Nairobi","Manila","Bogota","Warsaw",
        "Lisbon","Helsinki","Brussels","Zagreb","Riga","Tunis","Amman","Dakar","Accra","Kyiv"]
names = rng.sample([f"{a} {b}" for a in FIRST for b in LAST], N)
cities = [CITY[i % len(CITY)] for i in range(N)]  # cities repeat when N>32: binding name->city is the memory task
rng.shuffle(cities)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", local_files_only=True)
DOCS, QAS = [], []
for nm, ct in zip(names, cities):
    DOCS.append(tok(f"Context:\n{nm} lives in {ct}. They work at a small bakery near the old bridge.",
                    add_special_tokens=False, return_tensors="pt")["input_ids"].to(DEV))
    q = tok(f"\n\n{SYS}\nQuestion: Where does {nm} live?\nAnswer:", add_special_tokens=False)["input_ids"]
    a = tok(" " + ct, add_special_tokens=False)["input_ids"]
    QAS.append((q, a, ct))

CFG = PrefixSteerConfig(num_prefix_tokens=64, sliding_window_size=256, mem_num_heads=1,
    mem_head_dim=64, steer_mode="deltamem", prefix_write=True, memory_mode="dynamic",
    write_ctx_only=True, pool_reads=True, steer_gain=0.1, delta_heads="qkvo",
    steer_layers=(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33))
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16,
    attn_implementation="sdpa", local_files_only=True).to(DEV)
attach_prefix_steer(m, CFG); freeze_backbone_keep_steer(m)

def write(i):
    cids = DOCS[i]
    clear_frozen_memory(m); set_write_freeze(m, True)
    set_steer_segments(m, torch.full_like(cids, SEG_CTX), torch.ones_like(cids, dtype=torch.bool))
    m(input_ids=cids, use_cache=False)
    set_write_freeze(m, False)

def gen(qi, ntok=3):
    q, _, _ = QAS[qi]
    ids = list(q); seg = [SEG_QRY] * len(ids); out = []
    for _ in range(ntok):
        t = torch.tensor([ids], device=DEV)
        set_steer_segments(m, torch.tensor([seg], device=DEV), torch.ones_like(t, dtype=torch.bool))
        with torch.no_grad():
            nx = int(m(input_ids=t, use_cache=False).logits[0, -1].argmax())
        out.append(nx); ids.append(nx); seg.append(SEG_ANS)
    return tok.decode(out).strip()

def evaluate(k=None):
    idx = list(range(N)) if k is None else rng.sample(range(N), k)
    m.eval(); correct = 0; sw_mem = 0; sw_q = 0; nsw = 0
    with torch.no_grad():
        for i in idx:
            write(i)
            if gen(i).startswith(QAS[i][2]): correct += 1
            j = rng.choice([x for x in idx if x != i and QAS[x][2] != QAS[i][2]] or [i])
            if j != i:
                write(j); o = gen(i); nsw += 1
                if o.startswith(QAS[j][2]): sw_mem += 1     # tracks WRITTEN doc = memory-driven
                elif o.startswith(QAS[i][2]): sw_q += 1     # tracks QUESTION = leaked to weights
    clear_frozen_memory(m)
    return correct / len(idx), (sw_mem / max(1, nsw), sw_q / max(1, nsw))

pref = [p for n, p in m.named_parameters() if p.requires_grad and n.endswith(".prefix")]
rest = [p for n, p in m.named_parameters() if p.requires_grad and not n.endswith(".prefix")]
opt = torch.optim.AdamW([{"params": rest, "lr": LR}, {"params": pref, "lr": PLR}])
EV = max(100, MAX_STEPS // 10)
print(f"[ladder N={N}] max_steps={MAX_STEPS} lr={LR}/{PLR} eval_every={EV}", flush=True)
for step in range(MAX_STEPS):
    i = rng.randrange(N)
    m.train(); write(i)
    q, a, _ = QAS[i]
    ids = torch.tensor([q + a], device=DEV)
    seg = torch.tensor([[SEG_QRY] * len(q) + [SEG_ANS] * len(a)], device=DEV)
    lab = torch.tensor([[-100] * len(q) + a], device=DEV)
    set_steer_segments(m, seg, torch.ones_like(ids, dtype=torch.bool))
    loss = m(input_ids=ids, labels=lab, use_cache=False).loss
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
    opt.step()
    if (step + 1) % EV == 0 or step == MAX_STEPS - 1:
        acc, (swm, swq) = evaluate(k=min(N, 64))
        print(f"[ladder N={N}] step {step+1}: loss={loss.item():.4f} correct_acc={acc:.3f} "
              f"swap: mem-driven={swm:.3f} question-driven={swq:.3f}", flush=True)
        if acc >= 0.95:
            print(f"[ladder N={N}] SOLVED at step {step+1}", flush=True)
            break
print(f"[ladder N={N}] DONE", flush=True)
