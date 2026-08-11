"""Train/evaluate prefix steering as an episodic associative memory.

Every episode randomly reassigns cities to names, so the question alone is
deliberately ambiguous across episodes.  A task adapter can learn the answer
format but cannot know the current assignment.  The document is written once,
removed, and then queried:

    correct : write this episode, ask its question
    window  : same write, but remove prefix columns from the read
    swap    : write another episode, ask the original question
    base    : steer disabled, no write

The clean success criterion is correct >> max(window, swap), on held-out random
assignments.  Unlike a fixed finite lookup table, this measures an algorithmic
write/read capability rather than memorization of training documents.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from deltamem.core.global_prefix import SEG_ANS, SEG_CTX, SEG_QRY
from deltamem.core.prefix_steer import (
    PrefixSteerConfig,
    attach_prefix_steer,
    clear_frozen_memory,
    freeze_backbone_keep_steer,
    is_steer_param_name,
    set_steer_enabled,
    set_steer_segments,
    set_window_only,
    set_write_freeze,
)


SYS = "Answer the question using the stored document. Give only the city name."
FIRST = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Ivy", "Jack", "Kate", "Leo", "Mia", "Noah", "Olga", "Paul",
]
LAST = [
    "Anders", "Brooks", "Chen", "Diaz", "Evans", "Fischer", "Garcia",
    "Hayes", "Ito", "Jones", "Kim", "Lopez", "Meyer", "Novak", "Owens",
    "Park",
]
CITIES = [
    "Paris", "Tokyo", "Cairo", "Lima", "Oslo", "Delhi", "Rome", "Seoul",
    "Quito", "Hanoi", "Lagos", "Berlin", "Madrid", "Athens", "Dublin",
    "Vienna",
]


@dataclass
class Episode:
    doc_ids: list[int]
    q_ids: list[int]
    a_ids: list[int]
    answer: str
    target_name: str
    mapping: dict[str, str]


def make_episode(
    tok,
    rng: random.Random,
    facts: int,
    name_pool: list[str] | None = None,
    city_pool: list[str] | None = None,
) -> Episode:
    pool = name_pool or [f"{a} {b}" for a in FIRST for b in LAST]
    names = rng.sample(pool, facts)
    cities = rng.sample(city_pool or CITIES, facts)
    mapping = dict(zip(names, cities))
    target = rng.choice(names)
    # Shuffle fact order independently of target selection.
    rows = [f"{name} lives in {mapping[name]}." for name in names]
    rng.shuffle(rows)
    doc = "Context:\n" + "\n".join(rows)
    question = f"\n\n{SYS}\nQuestion: Where does {target} live?\nAnswer:"
    answer = mapping[target]
    return Episode(
        doc_ids=tok(doc, add_special_tokens=False)["input_ids"],
        q_ids=tok(question, add_special_tokens=False)["input_ids"],
        a_ids=tok(" " + answer, add_special_tokens=False)["input_ids"],
        answer=answer,
        target_name=target,
        mapping=mapping,
    )


def make_episode_group(
    tok,
    rng: random.Random,
    facts: int,
    queries_per_write: int,
    name_pool: list[str] | None = None,
    city_pool: list[str] | None = None,
) -> list[Episode]:
    """Create several target queries that share one randomly written document."""
    pool = name_pool or [f"{a} {b}" for a in FIRST for b in LAST]
    names = rng.sample(pool, facts)
    cities = rng.sample(city_pool or CITIES, facts)
    mapping = dict(zip(names, cities))
    rows = [f"{name} lives in {mapping[name]}." for name in names]
    rng.shuffle(rows)
    doc_ids = tok(
        "Context:\n" + "\n".join(rows), add_special_tokens=False
    )["input_ids"]
    targets = rng.sample(names, min(max(1, queries_per_write), facts))
    episodes = []
    for target in targets:
        question = f"\n\n{SYS}\nQuestion: Where does {target} live?\nAnswer:"
        answer = mapping[target]
        episodes.append(Episode(
            doc_ids=doc_ids,
            q_ids=tok(question, add_special_tokens=False)["input_ids"],
            a_ids=tok(" " + answer, add_special_tokens=False)["input_ids"],
            answer=answer,
            target_name=target,
            mapping=mapping,
        ))
    return episodes


def make_counterfactual_group(tok, episodes: list[Episode]) -> list[Episode]:
    """Keep the same keys/questions but rotate every value to a wrong one."""
    if not episodes:
        raise ValueError("episodes must be non-empty")
    original = episodes[0].mapping
    names = list(original)
    values = [original[name] for name in names]
    if len(values) == 1:
        replacements = [next(city for city in CITIES if city != values[0])]
    else:
        replacements = values[1:] + values[:1]
    mapping = dict(zip(names, replacements))
    rows = [f"{name} lives in {mapping[name]}." for name in names]
    doc_ids = tok(
        "Context:\n" + "\n".join(rows), add_special_tokens=False
    )["input_ids"]
    return [
        Episode(
            doc_ids=doc_ids,
            q_ids=ep.q_ids,
            a_ids=ep.a_ids,
            answer=ep.answer,
            target_name=ep.target_name,
            mapping=mapping,
        )
        for ep in episodes
    ]


def make_contrast_group(
    tok,
    rng: random.Random,
    facts: int,
    queries_per_write: int,
    name_pool: list[str] | None = None,
    city_pool: list[str] | None = None,
) -> tuple[list[Episode], list[Episode]]:
    """Two documents with identical keys/questions but conflicting value mappings."""
    pool = name_pool or [f"{a} {b}" for a in FIRST for b in LAST]
    values = city_pool or CITIES
    names = rng.sample(pool, facts)
    cities_a = rng.sample(values, facts)
    if facts == 1:
        cities_b = [rng.choice([city for city in values if city != cities_a[0]])]
    else:
        cities_b = cities_a[1:] + cities_a[:1]
    targets = rng.sample(names, min(max(1, queries_per_write), facts))

    def build(cities: list[str]) -> list[Episode]:
        mapping = dict(zip(names, cities))
        rows = [f"{name} lives in {mapping[name]}." for name in names]
        rng.shuffle(rows)
        doc_ids = tok(
            "Context:\n" + "\n".join(rows), add_special_tokens=False
        )["input_ids"]
        episodes = []
        for target in targets:
            question = f"\n\n{SYS}\nQuestion: Where does {target} live?\nAnswer:"
            answer = mapping[target]
            episodes.append(Episode(
                doc_ids=doc_ids,
                q_ids=tok(question, add_special_tokens=False)["input_ids"],
                a_ids=tok(" " + answer, add_special_tokens=False)["input_ids"],
                answer=answer,
                target_name=target,
                mapping=mapping,
            ))
        return episodes

    return build(cities_a), build(cities_b)


def make_contrast_pair(
    tok,
    rng: random.Random,
    facts: int,
    name_pool: list[str] | None = None,
    city_pool: list[str] | None = None,
) -> tuple[Episode, Episode]:
    """Same names and same question, but two conflicting random assignments."""
    pool = name_pool or [f"{a} {b}" for a in FIRST for b in LAST]
    names = rng.sample(pool, facts)
    target = rng.choice(names)
    values = city_pool or CITIES
    cities_a = rng.sample(values, facts)
    cities_b = rng.sample(values, facts)
    while cities_b[names.index(target)] == cities_a[names.index(target)]:
        cities_b = rng.sample(values, facts)

    def build(cities: list[str]) -> Episode:
        mapping = dict(zip(names, cities))
        rows = [f"{name} lives in {mapping[name]}." for name in names]
        rng.shuffle(rows)
        doc = "Context:\n" + "\n".join(rows)
        question = f"\n\n{SYS}\nQuestion: Where does {target} live?\nAnswer:"
        answer = mapping[target]
        return Episode(
            doc_ids=tok(doc, add_special_tokens=False)["input_ids"],
            q_ids=tok(question, add_special_tokens=False)["input_ids"],
            a_ids=tok(" " + answer, add_special_tokens=False)["input_ids"],
            answer=answer,
            target_name=target,
            mapping=mapping,
        )

    return build(cities_a), build(cities_b)


def set_segments(model, seg: torch.Tensor) -> None:
    set_steer_segments(model, seg, torch.ones_like(seg, dtype=torch.bool))


def city_hit(prediction: str, gold: str) -> float:
    """Accept repetition/punctuation after the city, but not glued junk (e.g. LagosHuman)."""
    pat = rf"^{re.escape(gold)}(?:\b|[\s,.])"
    return float(re.match(pat, prediction.strip(), flags=re.IGNORECASE) is not None)


def write_episode(model, ep: Episode, device: str, *, grad: bool) -> None:
    ids = torch.tensor([ep.doc_ids], device=device)
    clear_frozen_memory(model)
    set_write_freeze(model, True)
    set_segments(model, torch.full_like(ids, SEG_CTX))
    if grad:
        model(input_ids=ids, use_cache=False)
    else:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
    set_write_freeze(model, False)


def answer_loss(model, ep: Episode, device: str) -> torch.Tensor:
    """Answer CE using the currently frozen memory."""
    ids = torch.tensor([ep.q_ids + ep.a_ids], device=device)
    seg = torch.tensor(
        [[SEG_QRY] * len(ep.q_ids) + [SEG_ANS] * len(ep.a_ids)], device=device
    )
    labels = torch.tensor(
        [[-100] * len(ep.q_ids) + ep.a_ids], device=device
    )
    set_segments(model, seg)
    return model(input_ids=ids, labels=labels, use_cache=False).loss


def train_loss(
    model,
    ep: Episode,
    device: str,
    *,
    wo_contrast_lambda: float,
    wo_margin: float,
) -> tuple[torch.Tensor, float | None]:
    write_episode(model, ep, device, grad=True)
    ce = answer_loss(model, ep, device)
    loss = ce
    gap = None
    if wo_contrast_lambda > 0:
        set_window_only(model, True)
        ce_wo = answer_loss(model, ep, device)
        set_window_only(model, False)
        # Detached reference: the hinge can only make the correct-prefix path
        # better; it cannot manufacture a gap by sabotaging window-only.
        loss = loss + wo_contrast_lambda * torch.relu(
            wo_margin + ce - ce_wo.detach()
        )
        gap = float(ce_wo.item() - ce.item())
    clear_frozen_memory(model)
    return loss, gap


def train_group_loss(
    model,
    episodes: list[Episode],
    device: str,
    *,
    wo_contrast_lambda: float,
    wo_margin: float,
) -> tuple[torch.Tensor, float | None]:
    """Average multiple query losses while reusing exactly one written prefix."""
    if not episodes:
        raise ValueError("episodes must be non-empty")
    write_episode(model, episodes[0], device, grad=True)
    ces = [answer_loss(model, ep, device) for ep in episodes]
    ce = sum(ces) / len(ces)
    loss = ce
    gap = None
    if wo_contrast_lambda > 0:
        set_window_only(model, True)
        ces_wo = [answer_loss(model, ep, device) for ep in episodes]
        set_window_only(model, False)
        ce_wo = sum(ces_wo) / len(ces_wo)
        loss = loss + wo_contrast_lambda * torch.relu(
            wo_margin + ce - ce_wo.detach()
        )
        gap = float(ce_wo.item() - ce.item())
    clear_frozen_memory(model)
    return loss, gap


@torch.no_grad()
def generate(model, tok, ep: Episode, device: str, max_new_tokens: int = 3) -> str:
    ids = list(ep.q_ids)
    seg = [SEG_QRY] * len(ids)
    out: list[int] = []
    for _ in range(max_new_tokens):
        iid = torch.tensor([ids], device=device)
        sgt = torch.tensor([seg], device=device)
        set_segments(model, sgt)
        nxt = int(model(input_ids=iid, use_cache=False).logits[0, -1].argmax())
        if nxt == tok.eos_token_id:
            break
        out.append(nxt)
        ids.append(nxt)
        seg.append(SEG_ANS)
    return tok.decode(out, skip_special_tokens=True).strip()


@torch.no_grad()
def evaluate(model, tok, groups: list[list[Episode]], device: str) -> dict:
    """Evaluate several questions after one correct or counterfactual write per group."""
    model.eval()
    scores = {k: [] for k in ("base", "window", "swap", "correct")}
    records = []
    for group_id, episodes in enumerate(groups):
        counterfactual = make_counterfactual_group(tok, episodes)

        clear_frozen_memory(model)
        set_steer_enabled(model, False)
        base_predictions = [generate(model, tok, ep, device) for ep in episodes]

        set_steer_enabled(model, True)
        write_episode(model, episodes[0], device, grad=False)
        window_predictions = []
        correct_predictions = []
        for ep in episodes:
            set_window_only(model, True)
            window_predictions.append(generate(model, tok, ep, device))
            set_window_only(model, False)
            correct_predictions.append(generate(model, tok, ep, device))

        write_episode(model, counterfactual[0], device, grad=False)
        swap_predictions = [generate(model, tok, ep, device) for ep in episodes]
        clear_frozen_memory(model)

        for ep, pred_base, pred_window, pred_swap, pred_correct in zip(
            episodes,
            base_predictions,
            window_predictions,
            swap_predictions,
            correct_predictions,
        ):
            row = {
                "group": group_id,
                "name": ep.target_name,
                "gold": ep.answer,
                "counterfactual": counterfactual[0].mapping[ep.target_name],
                "base": pred_base,
                "window": pred_window,
                "swap": pred_swap,
                "correct": pred_correct,
            }
            for cond in scores:
                scores[cond].append(city_hit(row[cond], ep.answer))
            records.append(row)
    set_window_only(model, False)
    set_steer_enabled(model, True)
    n = len(records)
    return {
        "protocol": {
            "num_writes": len(groups),
            "num_queries": n,
            "mean_queries_per_write": n / len(groups),
            "strong_swap_same_keys": True,
        },
        "accuracy": {k: sum(v) / n for k, v in scores.items()},
        "paired": {
            "correct_minus_window": sum(
                a - b for a, b in zip(scores["correct"], scores["window"])
            ) / n,
            "correct_minus_swap": sum(
                a - b for a, b in zip(scores["correct"], scores["swap"])
            ) / n,
        },
        "records": records,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--facts", type=int, default=4)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-n", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--prefix-lr", type=float, default=3e-3)
    ap.add_argument("--num-prefix-tokens", type=int, default=64)
    ap.add_argument("--mem-num-heads", type=int, default=1)
    ap.add_argument("--mem-head-dim", type=int, default=64)
    ap.add_argument(
        "--read-mode",
        choices=["pool", "standard", "prefix_only", "steer_only"],
        default="pool",
    )
    ap.add_argument(
        "--queries-per-write",
        type=int,
        default=1,
        help="distinct targets supervised from one shared document write",
    )
    ap.add_argument("--paired-train", action="store_true")
    ap.add_argument("--wo-contrast-lambda", type=float, default=0.0)
    ap.add_argument("--wo-margin", type=float, default=0.5)
    ap.add_argument("--steer-gain", type=float, default=0.1)
    ap.add_argument("--init-ckpt", default="")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--eval-seed", type=int, default=-1)
    ap.add_argument(
        "--heldout-names",
        action="store_true",
        help="train on FIRST[:12] names and evaluate only FIRST[12:] names",
    )
    ap.add_argument(
        "--heldout-values",
        action="store_true",
        help="train on CITIES[:12] and evaluate only unseen CITIES[12:]",
    )
    ap.add_argument("--steer-layers", default="0,3,6,9,12,15,18,21,24,27,30,33")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.facts < 1:
        ap.error("--facts must be >= 1")
    if args.eval_n < 1:
        ap.error("--eval-n must be >= 1")
    if args.queries_per_write < 1:
        ap.error("--queries-per-write must be >= 1")
    if args.read_mode == "steer_only":
        args.num_prefix_tokens = 0

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(args.device)

    prefix_only = args.read_mode == "prefix_only"
    steer_only = args.read_mode == "steer_only"
    cfg = PrefixSteerConfig(
        num_prefix_tokens=args.num_prefix_tokens,
        sliding_window_size=256,
        mem_num_heads=args.mem_num_heads,
        mem_head_dim=args.mem_head_dim,
        steer_mode="deltamem",
        prefix_write=not steer_only,
        memory_mode="dynamic",
        write_ctx_only=not steer_only,
        read_prefix_only=prefix_only,
        pool_reads=args.read_mode == "pool",
        steer_gain=args.steer_gain,
        delta_heads="qkvo",
        steer_layers=tuple(int(x) for x in args.steer_layers.split(",") if x),
    )
    attach_prefix_steer(model, cfg)
    freeze_backbone_keep_steer(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(ck["state"], strict=False)
        steer_names = {
            n for n, _ in model.named_parameters() if is_steer_param_name(n)
        }
        missing_steer = sorted(n for n in missing if n in steer_names)
        dropped = sorted(n for n in unexpected if n in ck["state"])
        if missing_steer or dropped:
            raise RuntimeError(
                f"checkpoint/config mismatch: missing={missing_steer[:4]} "
                f"dropped={dropped[:4]}"
            )
        print(f"[episodic] initialized from {args.init_ckpt}", flush=True)

    pref = [
        p for n, p in model.named_parameters()
        if p.requires_grad and n.endswith(".prefix")
    ]
    pref_ids = {id(p) for p in pref}
    rest = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in pref_ids
    ]
    groups = [{"params": rest, "lr": args.lr}]
    if pref:
        groups.append(
            {"params": pref, "lr": args.prefix_lr, "weight_decay": 0.0}
        )
    opt = torch.optim.AdamW(groups)

    train_rng = random.Random(10_000 + args.seed)
    eval_seed = args.eval_seed if args.eval_seed >= 0 else 90_000 + args.seed
    eval_rng = random.Random(eval_seed)
    train_names = [f"{a} {b}" for a in FIRST[:12] for b in LAST]
    eval_names = [f"{a} {b}" for a in FIRST[12:] for b in LAST]
    if not args.heldout_names:
        train_names = eval_names = [f"{a} {b}" for a in FIRST for b in LAST]
    train_cities = CITIES[:12] if args.heldout_values else CITIES
    eval_cities = CITIES[12:] if args.heldout_values else CITIES
    eval_groups = []
    remaining = args.eval_n
    while remaining > 0:
        group = make_episode_group(
            tok,
            eval_rng,
            args.facts,
            min(args.queries_per_write, remaining),
            eval_names,
            eval_cities,
        )
        eval_groups.append(group)
        remaining -= len(group)
    history = []
    t0 = time.time()
    actual_queries_per_write = min(args.queries_per_write, args.facts)
    print(
        f"[episodic] mode={args.read_mode} facts={args.facts} steps={args.steps} "
        f"heads={args.mem_num_heads} head_dim={args.mem_head_dim} "
        f"P={args.num_prefix_tokens} q/write={actual_queries_per_write} "
        f"trainable={trainable:,} seed={args.seed}",
        flush=True,
    )
    if args.eval_only:
        final = evaluate(model, tok, eval_groups, args.device)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "args": vars(args),
            "config": asdict(cfg),
            "final": final,
        }, indent=2))
        a = final["accuracy"]
        print(
            f"[episodic] EVAL-ONLY n={args.eval_n} seed={eval_seed} "
            f"base={a['base']:.3f} window={a['window']:.3f} "
            f"swap={a['swap']:.3f} correct={a['correct']:.3f} "
            f"dW={final['paired']['correct_minus_window']:+.3f} "
            f"dS={final['paired']['correct_minus_swap']:+.3f} -> {out}",
            flush=True,
        )
        return

    for step in range(1, args.steps + 1):
        model.train()
        opt.zero_grad()
        if args.paired_train:
            paired_groups = make_contrast_group(
                tok,
                train_rng,
                args.facts,
                args.queries_per_write,
                train_names,
                train_cities,
            )
            results = [
                train_group_loss(
                    model,
                    group,
                    args.device,
                    wo_contrast_lambda=args.wo_contrast_lambda,
                    wo_margin=args.wo_margin,
                )
                for group in paired_groups
            ]
        else:
            eps = make_episode_group(
                tok,
                train_rng,
                args.facts,
                args.queries_per_write,
                train_names,
                train_cities,
            )
            results = [
                train_group_loss(
                    model,
                    eps,
                    args.device,
                    wo_contrast_lambda=args.wo_contrast_lambda,
                    wo_margin=args.wo_margin,
                )
            ]
        losses = [li for li, _ in results]
        gaps = [gap for _, gap in results if gap is not None]
        loss = sum(losses) / len(losses)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(model, tok, eval_groups, args.device)
            row = {
                "step": step,
                "loss": float(loss.item()),
                "accuracy": result["accuracy"],
                "paired": result["paired"],
            }
            history.append(row)
            a = result["accuracy"]
            gap_text = (
                f"train_wo_gap={sum(gaps)/len(gaps):+.3f} " if gaps else ""
            )
            print(
                f"[episodic] step={step} loss={loss.item():.4f} "
                f"base={a['base']:.3f} window={a['window']:.3f} "
                f"swap={a['swap']:.3f} correct={a['correct']:.3f} "
                f"dW={result['paired']['correct_minus_window']:+.3f} "
                f"dS={result['paired']['correct_minus_swap']:+.3f} "
                f"{gap_text}"
                f"elapsed={(time.time()-t0)/60:.1f}m",
                flush=True,
            )

    final = evaluate(model, tok, eval_groups, args.device)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "args": vars(args),
        "config": asdict(cfg),
        "history": history,
        "final": final,
    }
    out.write_text(json.dumps(payload, indent=2))
    state = {
        n: p.detach().cpu()
        for n, p in model.named_parameters()
        if is_steer_param_name(n)
    }
    torch.save(
        {"state": state, "cfg": asdict(cfg), "args": vars(args)},
        out.with_suffix(".pt"),
    )
    print(f"[episodic] DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
