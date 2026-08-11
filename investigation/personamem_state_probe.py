#!/usr/bin/env python3
"""Probe whether PersonaMem written state is persona-specific and causally read.

The probe loads a trained ``personamem_prefix_steer.py`` checkpoint, writes a
small deterministic set of PersonaMem histories, and reports:

* cross-persona cosine of the complete written state, per layer and overall;
* within-persona prefix-slot cosine, entropy effective rank, and state norms;
* for fixed future queries, correct-history versus swapped-history A/B/C/D
  logits, accuracy, and prediction-change rate.

Both dynamic prefix memory and the P=0 history-conditioned pooled-steer baseline
are supported.  A correct history is written once and shared by all K queries
for that persona.  The swapped condition writes one fixed deranged donor history
and evaluates exactly the same queries and option ordering.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deltamem.core.prefix_steer import (  # noqa: E402
    PrefixSteerConfig,
    attach_prefix_steer,
    clear_frozen_memory,
    freeze_backbone_keep_steer,
    iter_steer_modules,
    set_steer_enabled,
    set_window_only,
)
from scripts.personamem_prefix_steer import (  # noqa: E402
    KNOWN_TARGET_OVERLAP_SAMPLE_IDS,
    _load_checkpoint,
    _load_dataset,
    collate_reader_batch,
    encode_history,
    final_reader_logits,
    make_swap_derangements,
    parse_persona_ids,
    resolve_letter_token_ids,
    write_persona_memory,
)
from scripts.personamem_v2_data import (  # noqa: E402
    PersonaEpisode,
    resolve_split_csv,
)


def _finite_float(value: torch.Tensor | float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"diagnostic produced a non-finite metric: {result}")
    return result


def cosine_stats(vectors: torch.Tensor) -> dict[str, Any]:
    """Pairwise off-diagonal cosine statistics for ``[N,D]`` vectors."""
    if vectors.ndim != 2:
        raise ValueError(f"expected [N,D] vectors, got {tuple(vectors.shape)}")
    count = vectors.shape[0]
    if count < 2:
        return {
            "num_vectors": count,
            "num_pairs": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    normalised = torch.nn.functional.normalize(
        vectors.float(), dim=-1, eps=1e-12
    )
    rows, cols = torch.triu_indices(count, count, offset=1)
    values = (normalised[rows] * normalised[cols]).sum(dim=-1)
    return {
        "num_vectors": count,
        "num_pairs": int(values.numel()),
        "mean": _finite_float(values.mean()),
        "std": _finite_float(values.std(unbiased=False)),
        "min": _finite_float(values.min()),
        "max": _finite_float(values.max()),
    }


def entropy_effective_rank(matrix: torch.Tensor) -> float:
    """Effective rank ``exp(H(singular-value energy))`` in ``[1,min(S,D)]``."""
    if matrix.ndim != 2:
        raise ValueError(f"expected a matrix, got {tuple(matrix.shape)}")
    singular_values = torch.linalg.svdvals(matrix.float())
    energy = singular_values.square()
    total = energy.sum()
    if total <= 0:
        return 0.0
    probabilities = energy / total
    probabilities = probabilities[probabilities > 0]
    entropy = -(probabilities * probabilities.log()).sum()
    return _finite_float(entropy.exp())


def slot_cosine(state: torch.Tensor) -> float | None:
    """Mean off-diagonal cosine between rows of one ``[slots,dim]`` state."""
    if state.ndim != 2:
        raise ValueError(f"expected [slots,dim] state, got {tuple(state.shape)}")
    if state.shape[0] < 2:
        return None
    return cosine_stats(state)["mean"]


def summarize_layer_states(
    states_by_persona: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Summarize aligned written states for one layer."""
    if not states_by_persona:
        raise ValueError("cannot summarize an empty state collection")
    ordered = sorted(states_by_persona)
    states = [states_by_persona[persona_id].float() for persona_id in ordered]
    shapes = {tuple(state.shape) for state in states}
    if len(shapes) != 1:
        raise ValueError(f"persona state shapes differ within a layer: {sorted(shapes)}")
    shape = tuple(states[0].shape)
    if len(shape) != 2:
        raise ValueError(f"written state must be [slots,dim], got {shape}")

    flattened = torch.stack([state.flatten() for state in states])
    slot_cosines = [
        value for state in states if (value := slot_cosine(state)) is not None
    ]
    effective_ranks = [entropy_effective_rank(state) for state in states]
    fro_norms = [state.norm() for state in states]
    rms_norms = [state.square().mean().sqrt() for state in states]
    per_slot_norms = torch.cat([state.norm(dim=-1) for state in states])
    return {
        "num_personas": len(states),
        "slots": shape[0],
        "state_dim": shape[1],
        "cross_persona_cosine": cosine_stats(flattened),
        "slot_cosine_mean": (
            sum(slot_cosines) / len(slot_cosines) if slot_cosines else None
        ),
        "effective_rank_mean": sum(effective_ranks) / len(effective_ranks),
        "effective_rank_min": min(effective_ranks),
        "effective_rank_max": max(effective_ranks),
        "fro_norm_mean": _finite_float(torch.stack(fro_norms).mean()),
        "rms_norm_mean": _finite_float(torch.stack(rms_norms).mean()),
        "slot_norm_mean": _finite_float(per_slot_norms.mean()),
        "slot_norm_std": _finite_float(per_slot_norms.std(unbiased=False)),
    }


def summarize_all_states(
    states_by_layer: Mapping[int, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    """Return per-layer metrics plus cosine after concatenating every layer."""
    if not states_by_layer:
        raise ValueError("no layer states were captured")
    layer_ids = sorted(states_by_layer)
    persona_sets = [set(states_by_layer[layer]) for layer in layer_ids]
    if any(personas != persona_sets[0] for personas in persona_sets[1:]):
        raise ValueError("captured persona IDs differ across layers")
    persona_ids = sorted(persona_sets[0])
    per_layer = {
        str(layer): summarize_layer_states(states_by_layer[layer])
        for layer in layer_ids
    }
    concatenated = torch.stack(
        [
            torch.cat(
                [
                    states_by_layer[layer][persona_id].float().flatten()
                    for layer in layer_ids
                ]
            )
            for persona_id in persona_ids
        ]
    )

    scalar_fields = (
        "slot_cosine_mean",
        "effective_rank_mean",
        "fro_norm_mean",
        "rms_norm_mean",
        "slot_norm_mean",
    )
    layer_means: dict[str, float | None] = {}
    for field in scalar_fields:
        values = [
            metrics[field]
            for metrics in per_layer.values()
            if metrics[field] is not None
        ]
        layer_means[field] = sum(values) / len(values) if values else None
    return {
        "per_layer": per_layer,
        "overall": {
            "num_layers": len(layer_ids),
            "num_personas": len(persona_ids),
            "concatenated_cross_persona_cosine": cosine_stats(concatenated),
            "mean_of_layer_metrics": layer_means,
        },
    }


def summarize_logit_pairs(
    correct_logits: torch.Tensor,
    swap_logits: torch.Tensor,
    gold_indices: torch.Tensor,
) -> dict[str, Any]:
    """Aggregate paired correct/swap A-D logit diagnostics."""
    if correct_logits.shape != swap_logits.shape or correct_logits.ndim != 2:
        raise ValueError("correct and swap logits must have the same [N,4] shape")
    if correct_logits.shape[1] != 4 or gold_indices.shape != (correct_logits.shape[0],):
        raise ValueError("expected four-way logits and one gold index per row")
    correct = correct_logits.float()
    swap = swap_logits.float()
    gold = gold_indices.long()
    row = torch.arange(correct.shape[0])
    difference = correct - swap
    correct_prediction = correct.argmax(dim=-1)
    swap_prediction = swap.argmax(dim=-1)
    return {
        "num_queries": int(correct.shape[0]),
        "mean_abs_ad_logit_delta": _finite_float(difference.abs().mean()),
        "rms_ad_logit_delta": _finite_float(difference.square().mean().sqrt()),
        "mean_signed_gold_logit_delta": _finite_float(
            (correct[row, gold] - swap[row, gold]).mean()
        ),
        "prediction_change_rate": _finite_float(
            (correct_prediction != swap_prediction).float().mean()
        ),
        "correct_history_accuracy": _finite_float(
            (correct_prediction == gold).float().mean()
        ),
        "swap_history_accuracy": _finite_float(
            (swap_prediction == gold).float().mean()
        ),
    }


def _checkpoint_config(checkpoint: Mapping[str, Any]) -> PrefixSteerConfig:
    raw = checkpoint.get("cfg") or checkpoint.get("config")
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint has no PrefixSteerConfig under 'cfg' or 'config'")
    valid_names = {field.name for field in fields(PrefixSteerConfig)}
    values = {key: value for key, value in raw.items() if key in valid_names}
    for tuple_field in ("steer_layers", "prefix_layers"):
        if tuple_field in values:
            values[tuple_field] = tuple(int(value) for value in values[tuple_field])
    return PrefixSteerConfig(**values)


def _memory_kind(config: PrefixSteerConfig) -> str:
    if config.history_pool_mode != "none":
        return f"pooled_steer:{config.history_pool_mode}"
    if config.num_prefix_tokens > 0 and config.prefix_write:
        return "prefix"
    return "none"


def _select_episodes(
    episodes: Sequence[PersonaEpisode],
    *,
    persona_ids: set[str],
    num_personas: int,
    queries_per_persona: int,
    seed: int,
) -> tuple[PersonaEpisode, ...]:
    available = {episode.persona_id: episode for episode in episodes}
    if persona_ids:
        missing = sorted(persona_ids - set(available))
        if missing:
            raise ValueError(f"requested persona IDs are missing: {missing}")
        selected_ids = sorted(persona_ids)
    else:
        selected_ids = sorted(available)
        random.Random(seed).shuffle(selected_ids)
        selected_ids = selected_ids[:num_personas]
    selected: list[PersonaEpisode] = []
    for persona_id in selected_ids:
        episode = available[persona_id]
        questions = tuple(
            sorted(episode.questions, key=lambda question: question.sample_id)[
                :queries_per_persona
            ]
        )
        if questions:
            from dataclasses import replace

            selected.append(replace(episode, questions=questions))
    if len(selected) < 2:
        raise ValueError("state/swap probe requires at least two selected personas")
    return tuple(selected)


def _capture_written_states(
    modules: Sequence[Any],
    *,
    memory_kind: str,
) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    for module in modules:
        layer = int(module.base.layer_idx)
        if memory_kind.startswith("pooled_steer:"):
            state = module._frozen_history_pool
        else:
            # Prefix debug capture is produced by the exact WRITE path.  Frozen prefix is a
            # safe fallback and should be identical in WRITE-only evaluation.
            state = module._dbg_written
            if state is None:
                state = module._frozen_prefix
        if state is None:
            raise RuntimeError(f"layer {layer} did not expose a written state")
        if state.shape[0] != 1:
            raise RuntimeError(
                f"expected one history at a time, layer {layer} has batch {state.shape[0]}"
            )
        captured[layer] = state[0].detach().float().cpu().clone()
    return captured


@torch.no_grad()
def _ad_logits(
    model: Any,
    tokenizer: Any,
    episode: PersonaEpisode,
    letter_token_ids: Sequence[int],
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = collate_reader_batch(
        tokenizer, episode.questions, letter_token_ids, device=device
    )
    logits = final_reader_logits(model, batch)
    letters = torch.tensor(letter_token_ids, dtype=torch.long, device=device)
    return (
        logits.index_select(-1, letters).detach().float().cpu(),
        batch.target_indices.detach().cpu(),
    )


def _load_clean_episodes(
    *,
    data_root: Path,
    csv_path: Path,
    split: str,
    window: str,
    option_seed: int,
    clean_manifest: Path,
) -> tuple[PersonaEpisode, ...]:
    manifest = json.loads(clean_manifest.read_text(encoding="utf-8"))
    excluded_personas = {
        str(value)
        for value in manifest.get("exclude_persona_ids_from_train_and_val", [])
    }
    excluded_samples = set(KNOWN_TARGET_OVERLAP_SAMPLE_IDS)
    excluded_samples.update(
        str(value) for value in manifest.get("exclude_sample_ids_all_windows", [])
    )
    return _load_dataset(
        csv_path=csv_path,
        split=split,
        data_root=data_root,
        window=window,
        option_seed=option_seed,
        excluded_ids=excluded_personas,
        excluded_sample_ids=excluded_samples,
        overlap_policy="drop",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", "--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", default="data/personamem_v2", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument(
        "--split", default="auto", choices=["auto", "train", "val", "benchmark"]
    )
    parser.add_argument("--window", default="auto", choices=["auto", "32k", "128k"])
    parser.add_argument(
        "--clean-manifest",
        default=REPO_ROOT / "configs" / "personamem_v2_clean_v1.json",
        type=Path,
    )
    parser.add_argument("--persona-ids", default="")
    parser.add_argument("--num-personas", type=int, default=8)
    parser.add_argument("--queries-per-persona", type=int, default=4)
    parser.add_argument("--selection-seed", type=int, default=20260730)
    parser.add_argument("--swap-seed", type=int, default=4242)
    parser.add_argument(
        "--max-history-tokens",
        type=int,
        default=0,
        help="0 inherits the checkpoint training value",
    )
    parser.add_argument(
        "--history-truncation",
        default="auto",
        choices=["auto", "head", "tail"],
    )
    parser.add_argument("--model-path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
    )
    parser.add_argument("--attn-impl", default="")
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.num_personas < 2:
        raise SystemExit("--num-personas must be >= 2")
    if args.queries_per_persona < 1:
        raise SystemExit("--queries-per-persona must be positive")

    checkpoint_path = args.ckpt.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    checkpoint_args = checkpoint.get("args") or {}
    if not isinstance(checkpoint_args, Mapping):
        checkpoint_args = {}
    config = _checkpoint_config(checkpoint)
    memory_kind = _memory_kind(config)
    if memory_kind == "none":
        raise ValueError(
            "checkpoint has no persistent written state; use a prefix or pooled-steer model"
        )

    split = (
        str(checkpoint_args.get("eval_split", "benchmark"))
        if args.split == "auto"
        else args.split
    )
    window = (
        str(checkpoint_args.get("window", "32k"))
        if args.window == "auto"
        else args.window
    )
    data_root = args.data_root.expanduser().resolve()
    csv_path = (
        args.csv.expanduser().resolve()
        if args.csv
        else resolve_split_csv(data_root, split)
    )
    clean_manifest = args.clean_manifest.expanduser().resolve()
    option_seed = int(checkpoint_args.get("eval_option_seed", 1618))
    episodes = _load_clean_episodes(
        data_root=data_root,
        csv_path=csv_path,
        split=split,
        window=window,
        option_seed=option_seed,
        clean_manifest=clean_manifest,
    )
    selected = _select_episodes(
        episodes,
        persona_ids=parse_persona_ids(args.persona_ids, ()),
        num_personas=args.num_personas,
        queries_per_persona=args.queries_per_persona,
        seed=args.selection_seed,
    )

    model_path = args.model_path or str(
        checkpoint_args.get("model_path", "Qwen/Qwen3-4B-Instruct-2507")
    )
    dtype_name = (
        str(checkpoint_args.get("dtype", "bfloat16"))
        if args.dtype == "auto"
        else args.dtype
    )
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]
    attn_impl = args.attn_impl or str(checkpoint_args.get("attn_impl", "sdpa"))
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    letter_token_ids = resolve_letter_token_ids(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        local_files_only=args.local_files_only,
    ).to(args.device)
    attach_prefix_steer(model, config)
    freeze_backbone_keep_steer(model)
    _load_checkpoint(model, str(checkpoint_path))
    model.config.use_cache = False
    model.eval()
    modules = list(iter_steer_modules(model))
    if not modules:
        raise RuntimeError("checkpoint config attached no prefix-steer modules")
    set_steer_enabled(model, True)
    set_window_only(model, False)

    max_history_tokens = (
        args.max_history_tokens
        if args.max_history_tokens > 0
        else int(checkpoint_args.get("max_history_tokens", 4096))
    )
    truncation = (
        str(checkpoint_args.get("history_truncation", "tail"))
        if args.history_truncation == "auto"
        else args.history_truncation
    )
    histories = {
        episode.persona_id: encode_history(
            tokenizer,
            episode.writer,
            persona_id=episode.persona_id,
            max_history_tokens=max_history_tokens,
            truncation=truncation,
        )
        for episode in selected
    }
    derangement = make_swap_derangements(
        [episode.persona_id for episode in selected],
        count=1,
        seed=args.swap_seed,
    )[0]

    states_by_layer: dict[int, dict[str, torch.Tensor]] = {
        int(module.base.layer_idx): {} for module in modules
    }
    correct_logits: dict[str, torch.Tensor] = {}
    gold_indices: dict[str, torch.Tensor] = {}
    for episode in selected:
        for module in modules:
            module._debug_write = True
            module._dbg_written = None
        write_persona_memory(
            model,
            histories[episode.persona_id],
            device=args.device,
            grad=False,
            prefix_enabled=True,
        )
        captured = _capture_written_states(modules, memory_kind=memory_kind)
        for module in modules:
            module._debug_write = False
        for layer, state in captured.items():
            states_by_layer[layer][episode.persona_id] = state
        correct_logits[episode.persona_id], gold_indices[episode.persona_id] = (
            _ad_logits(
                model,
                tokenizer,
                episode,
                letter_token_ids,
                device=args.device,
            )
        )
        clear_frozen_memory(model)

    swap_logits: dict[str, torch.Tensor] = {}
    for episode in selected:
        donor_id = derangement[episode.persona_id]
        write_persona_memory(
            model,
            histories[donor_id],
            device=args.device,
            grad=False,
            prefix_enabled=True,
        )
        swap_logits[episode.persona_id], swap_gold = _ad_logits(
            model,
            tokenizer,
            episode,
            letter_token_ids,
            device=args.device,
        )
        if not torch.equal(swap_gold, gold_indices[episode.persona_id]):
            raise AssertionError("correct and swap query batches changed gold labels")
        clear_frozen_memory(model)

    all_correct = torch.cat(
        [correct_logits[episode.persona_id] for episode in selected]
    )
    all_swap = torch.cat([swap_logits[episode.persona_id] for episode in selected])
    all_gold = torch.cat([gold_indices[episode.persona_id] for episode in selected])
    logit_summary = summarize_logit_pairs(all_correct, all_swap, all_gold)
    records: list[dict[str, Any]] = []
    cursor = 0
    for episode in selected:
        persona_correct = correct_logits[episode.persona_id]
        persona_swap = swap_logits[episode.persona_id]
        persona_gold = gold_indices[episode.persona_id]
        for index, question in enumerate(episode.questions):
            c_logits = persona_correct[index]
            s_logits = persona_swap[index]
            c_prediction = int(c_logits.argmax())
            s_prediction = int(s_logits.argmax())
            records.append(
                {
                    "sample_id": question.sample_id,
                    "persona_id": episode.persona_id,
                    "swap_persona_id": derangement[episode.persona_id],
                    "gold_index": int(persona_gold[index]),
                    "gold_letter": chr(ord("A") + int(persona_gold[index])),
                    "correct_ad_logits": c_logits.tolist(),
                    "swap_ad_logits": s_logits.tolist(),
                    "correct_minus_swap_ad_logits": (c_logits - s_logits).tolist(),
                    "correct_prediction": c_prediction,
                    "swap_prediction": s_prediction,
                    "correct_prediction_letter": chr(ord("A") + c_prediction),
                    "swap_prediction_letter": chr(ord("A") + s_prediction),
                    "prediction_changed": c_prediction != s_prediction,
                }
            )
            cursor += 1
    if cursor != all_correct.shape[0]:
        raise AssertionError("record/logit row count mismatch")

    state_summary = summarize_all_states(states_by_layer)
    payload = {
        "checkpoint": str(checkpoint_path),
        "memory_kind": memory_kind,
        "config": {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in vars(config).items()
        },
        "protocol": {
            "split": split,
            "window": window,
            "csv": str(csv_path),
            "persona_ids": [episode.persona_id for episode in selected],
            "num_personas": len(selected),
            "queries_per_persona_cap": args.queries_per_persona,
            "num_queries": len(records),
            "selection_seed": args.selection_seed,
            "swap_seed": args.swap_seed,
            "swap_mapping": derangement,
            "max_history_tokens": max_history_tokens,
            "history_truncation": truncation,
            "option_seed": option_seed,
            "one_correct_write_per_persona": True,
            "one_swap_write_per_target_persona": True,
        },
        "written_state": state_summary,
        "correct_vs_swap_logits": {
            "summary": logit_summary,
            "records": records,
        },
    }
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else checkpoint_path.with_suffix(".state_probe.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"[state-probe] memory={memory_kind} personas={len(selected)} "
        f"queries={len(records)} output={output_path}"
    )
    print(
        f"{'layer':>6} {'slots':>7} {'dim':>7} {'cos/persona':>13} "
        f"{'slot_cos':>10} {'eff_rank':>10} {'rms':>10}"
    )
    for layer, metrics in state_summary["per_layer"].items():
        cross_cosine = metrics["cross_persona_cosine"]["mean"]
        slot_cos = metrics["slot_cosine_mean"]
        print(
            f"{layer:>6} {metrics['slots']:>7} {metrics['state_dim']:>7} "
            f"{cross_cosine if cross_cosine is not None else float('nan'):>13.4f} "
            f"{slot_cos if slot_cos is not None else float('nan'):>10.4f} "
            f"{metrics['effective_rank_mean']:>10.3f} "
            f"{metrics['rms_norm_mean']:>10.4f}"
        )
    overall_cosine = state_summary["overall"][
        "concatenated_cross_persona_cosine"
    ]["mean"]
    print(
        "[state-probe] "
        f"overall_cross_persona_cos={overall_cosine:.4f} "
        f"correct_acc={logit_summary['correct_history_accuracy']:.3f} "
        f"swap_acc={logit_summary['swap_history_accuracy']:.3f} "
        f"prediction_change={logit_summary['prediction_change_rate']:.3f} "
        f"mean_abs_AD_delta={logit_summary['mean_abs_ad_logit_delta']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
