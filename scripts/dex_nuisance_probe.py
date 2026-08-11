#!/usr/bin/env python
"""Grouped long-context nuisance probe.

Question this answers, before any new method is written: **does the DEX
correction actually target long-context nuisance variance?**

Construction.  For each semantic group ``g`` we fix the query ``q`` and the
long-range evidence ``e`` and vary only nuisance ``n``: which filler paragraphs
surround the evidence, their order, the evidence's absolute depth, and the
distractor needles.  Every group is instantiated K times, and twice over:
once with evidence value ``v1`` and once with ``v2`` (evidence swap).

Readout.  A *fixed* scalar per context, independent of which evidence was
inserted::

    g(x) = log p(v1 | x) - log p(v2 | x)

Using the paper-style "gold margin" instead would compare different targets in
the two evidence conditions and could not separate signal from noise; this
log-odds readout keeps the target fixed, so

    V_nuis = mean_g Var_k[ g(x_{g,k}) ]            (within one evidence value)
    S_evid = mean_g ( E_k[g | v1] - E_k[g | v2] )^2
    NSR    = V_nuis / (S_evid + eps)

Hidden-state metrics, per layer and head, on the probe tokens, after centering
within a group (Y = per-head attention output at the DEX insertion point,
Delta = the signed update the adapter actually applies, Ytil = Y + Delta)::

    VRR       = 1 - sum|Ytil - mean_k Ytil|^2 / sum|Y - mean_k Y|^2
    AntiAlign = - <r_Delta, r_Y> / (|r_Delta| |r_Y|)
    MeanShift = |mean_k Ytil - mean_k Y| / (|mean_k Y| + eps)

VRR > 0 means the correction removes nuisance variance; AntiAlign > 0 means it
points against the nuisance fluctuation; MeanShift says how much of what it does
is just a constant translation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from deltamem.core.dex import (  # noqa: E402
    DexConfig,
    DexOutputProjection,
    attach_dex,
    load_head_plan,
    set_dex_step,
)

FACT = "The {subject} achieves a BLEU score of {value} on the WMT14 English-German test set."
QUESTION = "What BLEU score does the {subject} achieve on the WMT14 English-German test set?"
SUBJECTS = [
    "proposed model", "hierarchical encoder", "retrieval-augmented decoder",
    "sparse mixture model", "adapter-tuned transformer", "multilingual baseline",
    "distilled student model", "curriculum-trained system",
]
DISTRACTOR_SUBJECTS = [
    "earlier prototype", "ablated variant", "reference implementation",
    "internal replication", "unpublished predecessor", "shared-encoder variant",
    "frozen-embedding baseline", "two-stage pipeline", "single-pass decoder",
    "reranked ensemble", "byte-level system", "character-level system",
]


def build_groups(tok, chunks: list[str], args) -> list[dict]:
    """K nuisance instantiations x 2 evidence assignments per group.

    A *binding* probe: both candidate values are always present in the context.
    The queried subject owns ``v1`` in condition ``v1`` and ``v2`` in condition
    ``v2``, and a foil subject owns the other one, so the value multiset is
    identical in both conditions.  Only the subject->value binding changes, which
    is what the fixed log-odds readout measures.  Nuisance = filler paragraphs,
    needle order, needle depths.
    """
    rng = random.Random(args.seed)
    groups = []
    for gi in range(args.groups):
        subject = SUBJECTS[gi % len(SUBJECTS)]
        foil = DISTRACTOR_SUBJECTS[gi % len(DISTRACTOR_SUBJECTS)]
        v1, v2 = f"{rng.uniform(20, 49):.1f}", f"{rng.uniform(20, 49):.1f}"
        while abs(float(v1) - float(v2)) < 3.0:
            v2 = f"{rng.uniform(20, 49):.1f}"
        others = [s for s in DISTRACTOR_SUBJECTS if s != foil]
        distractors = [
            (ds, f"{rng.uniform(20, 49):.1f}")
            for ds in rng.sample(others, min(args.distractors, len(others)))
        ]
        variants = []
        for k in range(args.k):
            pool = rng.sample(chunks, min(len(chunks), args.filler_chunks))
            n_needles = len(distractors) + 2
            variants.append({
                "filler": pool,
                "depths": [rng.uniform(0.02, 0.98) for _ in range(n_needles)],
                "order": rng.sample(range(n_needles), n_needles),
            })
        groups.append({"gi": gi, "subject": subject, "foil": foil,
                       "query": QUESTION.format(subject=subject),
                       "v1": v1, "v2": v2, "distractors": distractors,
                       "variants": variants})
    return groups


def render(group: dict, variant: dict, which: str, tok, args) -> list[int]:
    """Assemble one long context; ``which`` picks the subject->value binding."""
    target_v, foil_v = ((group["v1"], group["v2"]) if which == "v1"
                        else (group["v2"], group["v1"]))
    needles = [FACT.format(subject=group["subject"], value=target_v),
               FACT.format(subject=group["foil"], value=foil_v)]
    needles += [FACT.format(subject=s, value=v) for s, v in group["distractors"]]
    needles = [needles[i] for i in variant["order"]]
    depths = [variant["depths"][i] for i in variant["order"]]

    body_ids: list[int] = []
    for chunk in variant["filler"]:
        body_ids += tok(chunk + "\n\n", add_special_tokens=False)["input_ids"]
        if len(body_ids) > args.ctx_tokens:
            break
    body_ids = body_ids[: args.ctx_tokens]

    def insert(ids: list[int], sentence: str, frac: float) -> list[int]:
        s_ids = tok(" " + sentence + "\n\n", add_special_tokens=False)["input_ids"]
        pos = int(len(ids) * frac)
        return ids[:pos] + s_ids + ids[pos:]

    for needle, frac in sorted(zip(needles, depths), key=lambda t: -t[1]):
        body_ids = insert(body_ids, needle, frac)

    tail = tok(f"\n\nQuestion: {group['query']}\nAnswer:", add_special_tokens=False)["input_ids"]
    return body_ids + tail


@torch.no_grad()
def candidate_logprob(model, prompt_ids: list[int], answer: str, tok, device) -> float:
    ans_ids = tok(" " + answer, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([prompt_ids + ans_ids], device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids, use_cache=False).logits.float()
    lp = torch.log_softmax(logits[0, len(prompt_ids) - 1: -1], dim=-1)
    tgt = torch.tensor(ans_ids, device=device)
    return float(lp.gather(-1, tgt[:, None]).sum().item())


def load_variant(name: str, path: str, args, plan):
    """Backbone -> attach_dex(saved config) -> load adapter and/or attention."""
    from transformers import AutoModelForCausalLM
    from qasper_prefix_steer import get_dtype

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=get_dtype("bfloat16"), attn_implementation="sdpa"
    ).to(args.device)
    model.config.use_cache = False
    if name == "base":
        attach_dex(model, DexConfig(variant="base"), plan=plan)
        return model, DexConfig(variant="base").resolve()

    adapter_ckpt = Path(f"{path}_adapter.pt")
    attn_ckpt = Path(f"{path}_attn.pt")
    src = adapter_ckpt if adapter_ckpt.exists() else attn_ckpt
    saved = torch.load(src, map_location="cpu", weights_only=False)
    saved_cfg = saved["config"]
    cfg = DexConfig(
        variant=saved_cfg["variant"],
        head_selection=saved_cfg["head_selection"],
        heads_per_layer=saved_cfg["heads_per_layer"],
        lambda_init_mode=saved_cfg["lambda_init_mode"],
        lambda_init=saved_cfg["lambda_init"],
        lambda_learn_init=saved_cfg["lambda_learn_init"],
        lambda_learnable=saved_cfg["lambda_learnable"],
        lambda_anneal_steps=saved_cfg["lambda_anneal_steps"],
        allow_no_anneal=True,
    )
    attach_dex(model, cfg, plan=plan)
    state = {}
    if adapter_ckpt.exists():
        state.update(torch.load(adapter_ckpt, map_location="cpu", weights_only=False)["state"])
    if attn_ckpt.exists():
        state.update(torch.load(attn_ckpt, map_location="cpu", weights_only=False)["state"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:4]}"
    loaded = len(state)
    print(f"[probe] {name}: loaded {loaded} tensors from {src.name}", flush=True)
    # lambda(t) at the end of training
    set_dex_step(model, saved["args"]["steps"])
    return model, cfg.resolve()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/work/mingze/models/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--head-plan", default="out_dex/head_plan_qwen3_4b.json")
    ap.add_argument("--variants", default="base:,attn_only:out_dex/ckpt_attn_only_s0,"
                                          "dex_minus:out_dex/ckpt_dex_minus_s0,"
                                          "dex_plus:out_dex/ckpt_dex_plus_s0,"
                                          "adapter_only:out_dex/ckpt_adapter_only_s0")
    ap.add_argument("--groups", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--ctx-tokens", type=int, default=3200)
    ap.add_argument("--filler-chunks", type=int, default=24)
    ap.add_argument("--distractors", type=int, default=3)
    ap.add_argument("--depths", default="0,25,50,75")
    ap.add_argument("--probe-tokens", type=int, default=16)
    ap.add_argument("--layers", default="", help="comma list; default = every wrapped layer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="out_dex/nuisance_probe.json")
    args = ap.parse_args()
    args.depths = [float(x) for x in args.depths.split(",")]

    from transformers import AutoTokenizer
    from deltamem.kv_binding.qasper_episodes import build_fulldoc_episodes

    tok = AutoTokenizer.from_pretrained(args.model_path)
    plan = load_head_plan(args.head_plan)
    eps = build_fulldoc_episodes("validation", max_papers=40, tokenizer=tok, max_chunk_tok=256)
    chunks = [c for ep in eps for c in ep["chunks"] if len(c.split()) > 40]
    print(f"[probe] filler pool: {len(chunks)} paragraphs", flush=True)

    groups = build_groups(tok, chunks, args)
    prompts = {}   # (gi, k, which) -> ids
    for g in groups:
        for k, v in enumerate(g["variants"]):
            for which in ("v1", "v2"):
                prompts[(g["gi"], k, which)] = render(g, v, which, tok, args)
    lens = [len(v) for v in prompts.values()]
    print(f"[probe] {len(prompts)} contexts, {min(lens)}-{max(lens)} tokens", flush=True)

    results = {}
    for spec in args.variants.split(","):
        name, _, path = spec.partition(":")
        name = name.strip()
        if not name:
            continue
        model, cfg = load_variant(name, path, args, plan)
        model.eval()

        captures: dict = {}
        head_masks: dict = {}
        hooks = []
        wrapped = [m for m in model.modules() if isinstance(m, DexOutputProjection)]
        want_layers = ([int(x) for x in args.layers.split(",") if x.strip()]
                       if args.layers else list(range(len(wrapped))))

        capture_on = {"flag": False}

        def make_hook(li):
            def hook(mod, inp, out):  # noqa: ANN001
                if not capture_on["flag"]:
                    return                     # candidate-scoring forwards must not capture
                x = inp[0]
                y = x.reshape(*x.shape[:-1], mod.num_heads, mod.head_dim)
                probe = y[0, -args.probe_tokens:].float().mean(0)          # [H, Dh]
                if mod.adapter is not None:
                    yt = mod.adapter(y)
                    dt = (yt - y)[0, -args.probe_tokens:].float().mean(0)   # signed update
                else:
                    dt = torch.zeros_like(probe)
                captures.setdefault(li, []).append(
                    (probe.cpu().numpy(), dt.cpu().numpy()))
            return hook

        for li, m in enumerate(wrapped):
            if li in want_layers:
                hooks.append(m.register_forward_hook(make_hook(li)))
                if m.adapter is not None:
                    head_masks[li] = m.adapter.head_mask.bool().cpu().numpy()
                else:
                    # no adapter: use the plan's would-be selection so that the
                    # selected-head statistics stay comparable across variants
                    from deltamem.core.dex import select_heads_for_layer
                    ref = DexConfig(head_selection="entropy_high", heads_per_layer=-1,
                                    allow_no_anneal=True).resolve()
                    sel = select_heads_for_layer(li, m.num_heads, ref, plan)
                    mask = np.zeros(m.num_heads, dtype=bool)
                    mask[list(sel)] = True
                    head_masks[li] = mask

        order = []
        readout = {}
        for key, ids in prompts.items():
            capture_on["flag"] = True
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(input_ids=torch.tensor([ids], device=args.device), use_cache=False)
            capture_on["flag"] = False
            order.append(key)
            g = next(x for x in groups if x["gi"] == key[0])
            lp1 = candidate_logprob(model, ids, g["v1"], tok, args.device)
            lp2 = candidate_logprob(model, ids, g["v2"], tok, args.device)
            readout[key] = lp1 - lp2
        for h in hooks:
            h.remove()

        # ---- scalar signal / noise ------------------------------------
        v_nuis, s_evid, acc = [], [], []
        for g in groups:
            r1 = np.array([readout[(g["gi"], k, "v1")] for k in range(args.k)])
            r2 = np.array([readout[(g["gi"], k, "v2")] for k in range(args.k)])
            v_nuis.append(0.5 * (r1.var(ddof=1) + r2.var(ddof=1)))
            s_evid.append((r1.mean() - r2.mean()) ** 2)
            acc.append(float((r1 > 0).mean() * 0.5 + (r2 < 0).mean() * 0.5))
        v_nuis, s_evid = float(np.mean(v_nuis)), float(np.mean(s_evid))

        # ---- hidden-state metrics -------------------------------------
        idx = {key: i for i, key in enumerate(order)}
        for li, rows in captures.items():
            assert len(rows) == len(order), (
                f"layer {li}: captured {len(rows)} rows for {len(order)} prompts")
        hidden = {}
        for li, rows in captures.items():
            Y_all = np.stack([rows[idx[key]][0] for key in order])      # [N, H, Dh]
            D_all = np.stack([rows[idx[key]][1] for key in order])
            hmask = head_masks[li]
            Y, D = Y_all[:, hmask, :], D_all[:, hmask, :]   # adapted heads only
            vrr_num = vrr_den = 0.0
            aa_num = aa_dy = aa_dd = 0.0
            ms, vy, vy_all = [], [], []
            for g in groups:
                for which in ("v1", "v2"):
                    sel = [idx[(g["gi"], k, which)] for k in range(args.k)]
                    y, d = Y[sel], D[sel]
                    vy_all.append(float(((Y_all[sel] - Y_all[sel].mean(0, keepdims=True)) ** 2).mean()))
                    yt = y + d
                    ry = y - y.mean(0, keepdims=True)
                    rt = yt - yt.mean(0, keepdims=True)
                    rd = d - d.mean(0, keepdims=True)
                    vrr_den += float((ry ** 2).sum())
                    vrr_num += float((rt ** 2).sum())
                    aa_num += float((rd * ry).sum())
                    aa_dy += float((ry ** 2).sum())
                    aa_dd += float((rd ** 2).sum())
                    ms.append(float(np.linalg.norm(yt.mean(0) - y.mean(0))
                                    / (np.linalg.norm(y.mean(0)) + 1e-9)))
                    vy.append(float((ry ** 2).mean()))
            hidden[li] = {
                "VRR": 1.0 - vrr_num / max(vrr_den, 1e-12),
                "AntiAlign": (-aa_num / max((aa_dy * aa_dd) ** 0.5, 1e-12)) if aa_dd > 0 else 0.0,
                "MeanShift": float(np.mean(ms)),
                "hidden_var": float(np.mean(vy)),            # adapted heads
                "hidden_var_all_heads": float(np.mean(vy_all)),
                "n_adapted_heads": int(hmask.sum()),
            }

        results[name] = {
            "variant_cfg": {k: (list(v) if isinstance(v, tuple) else v)
                            for k, v in vars(cfg).items()},
            "V_nuis": v_nuis,
            "S_evid": s_evid,
            "NSR": v_nuis / (s_evid + 1e-9),
            "readout_accuracy": float(np.mean(acc)),
            "hidden_mean": {
                m: float(np.mean([h[m] for h in hidden.values()]))
                for m in ("VRR", "AntiAlign", "MeanShift", "hidden_var",
                          "hidden_var_all_heads")
            },
            "hidden_per_layer": hidden,
        }
        print(f"[probe] {name}: V_nuis={v_nuis:.4f} S_evid={s_evid:.4f} "
              f"NSR={results[name]['NSR']:.4f} acc={results[name]['readout_accuracy']:.3f} "
              f"VRR={results[name]['hidden_mean']['VRR']:+.4f} "
              f"AntiAlign={results[name]['hidden_mean']['AntiAlign']:+.4f} "
              f"MeanShift={results[name]['hidden_mean']['MeanShift']:.4f} "
              f"hidden_var={results[name]['hidden_mean']['hidden_var']:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump({"args": vars(args), "results": results}, fh, indent=2)
    print(f"[probe] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
