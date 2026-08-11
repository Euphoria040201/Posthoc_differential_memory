#!/usr/bin/env python3
"""Frozen full-context forced-choice baseline for PersonaMem-v2.

This is the deliberately simple, independently runnable oracle baseline:

* load one leakage-audited PersonaMem-v2 text split;
* prepend that persona's chat history to every future MCQ;
* append the current MCQ as a new user turn plus the assistant generation cue;
* score exactly the single-token choices A/B/C/D with a frozen causal LM.

The default remains the original ``correct_history``-only baseline.  For causal
controls, ``--conditions query_only correct_history swapped_history`` evaluates
the identical query/options under no history, its own history, and one or more
fixed different-persona history derangements.  ``--num-swaps 3`` requests three
distinct swapped histories per query.

The implementation repeats the full history for every query.  It does **not**
claim cached-KV write-once branching: that optimization would need a separate,
carefully validated implementation because position handling, padding, and cache
ownership can otherwise change the baseline.  The JSON protocol records this
fact explicitly.

Token truncation is also explicit.  The current MCQ and generation cue are never
truncated.  When the complete prompt exceeds ``--max-context-tokens``, only the
history token prefix is shortened: ``tail`` keeps the most recent history tokens
and ``head`` keeps the oldest history tokens.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.personamem_v2_data import (
        AuditTags,
        LoadedPersonaDataset,
        MCQExample,
        PersonaEpisode,
        load_personamem_text,
        resolve_split_csv,
    )
except ModuleNotFoundError:  # Direct invocation: python scripts/this_file.py
    from personamem_v2_data import (  # type: ignore[no-redef]
        AuditTags,
        LoadedPersonaDataset,
        MCQExample,
        PersonaEpisode,
        load_personamem_text,
        resolve_split_csv,
    )


DEFAULT_MANIFEST = REPO_ROOT / "configs" / "personamem_v2_clean_v1.json"
VALID_CONDITIONS = ("query_only", "correct_history", "swapped_history")
SUBGROUP_FIELDS = (
    "pref_type",
    "who",
    "updated",
    "sensitive_info",
    "conversation_scenario",
    "topic_query",
    "topic_preference",
)


@dataclass(frozen=True)
class EncodedFullContext:
    sample_id: str
    persona_id: str
    input_ids: tuple[int, ...]
    gold_index: int
    gold_letter: str
    tags: AuditTags
    history_tokens_total: int
    history_tokens_kept: int
    reader_suffix_tokens: int
    condition: str = "correct_history"
    history_persona_id: str | None = None
    swap_index: int | None = None

    @property
    def context_tokens(self) -> int:
        return len(self.input_ids)

    @property
    def history_tokens_truncated(self) -> int:
        return self.history_tokens_total - self.history_tokens_kept


def parse_csv_set(value: str, repeated: Iterable[str] = ()) -> set[str]:
    output = {str(item).strip() for item in repeated}
    output.update(item.strip() for item in value.split(","))
    output.discard("")
    return output


def _flatten_token_ids(value: Any) -> list[int]:
    """Normalize tokenizer/chat-template outputs across transformers versions."""
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError("tokenizer output has no input_ids")
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("expected one tokenized sequence")
        value = value[0]
    return [int(token_id) for token_id in value]


def resolve_letter_token_ids(tokenizer: Any) -> tuple[int, int, int, int]:
    """Resolve one consistent single-token spelling of A/B/C/D."""
    for prefix in ("", " "):
        encoded = [
            _flatten_token_ids(
                tokenizer(prefix + letter, add_special_tokens=False)["input_ids"]
            )
            for letter in "ABCD"
        ]
        if all(len(token_ids) == 1 for token_ids in encoded):
            result = tuple(token_ids[0] for token_ids in encoded)
            if len(set(result)) != 4:
                raise ValueError(f"A/B/C/D have non-unique token IDs: {result}")
            return result  # type: ignore[return-value]
    raise ValueError(
        "Tokenizer has no consistent one-token A/B/C/D spelling; this baseline "
        "intentionally uses one-token forced-choice scoring"
    )


def _chat_ids(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    try:
        encoded = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            "The full-context baseline requires a working tokenizer chat template"
        ) from error
    return _flatten_token_ids(encoded)


def encode_full_context(
    tokenizer: Any,
    episode: PersonaEpisode,
    question: MCQExample,
    *,
    max_context_tokens: int,
    truncation: str,
    condition: str = "correct_history",
    history_episode: PersonaEpisode | None = None,
    swap_index: int | None = None,
) -> EncodedFullContext:
    """Encode history + current MCQ while reserving the complete reader suffix.

    The separately rendered history must be an exact token prefix of the complete
    chat template.  This invariant holds for Qwen's turn-delimited template and
    prevents a seemingly convenient string/token splice from silently changing
    special-token boundaries.
    """
    if max_context_tokens < 1:
        raise ValueError("max_context_tokens must be positive")
    if truncation not in {"head", "tail"}:
        raise ValueError("truncation must be head or tail")
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    if condition == "query_only":
        if history_episode is not None:
            raise ValueError("query_only must not receive a history episode")
        source_episode = None
    elif condition == "correct_history":
        source_episode = history_episode or episode
        if source_episode.persona_id != episode.persona_id:
            raise ValueError("correct_history must use the target persona history")
    else:
        if history_episode is None:
            raise ValueError("swapped_history requires a source history episode")
        if history_episode.persona_id == episode.persona_id:
            raise ValueError("swapped_history must use a different persona")
        if swap_index is None or swap_index < 0:
            raise ValueError("swapped_history requires a non-negative swap_index")
        source_episode = history_episode

    history_messages: list[dict[str, str]]
    if source_episode is None:
        history_messages = []
    else:
        source_episode.writer.assert_safe_schema()
        history_messages = source_episode.writer.to_messages()
    reader_message = {"role": "user", "content": question.reader.to_prompt()}

    history_ids = (
        _chat_ids(tokenizer, history_messages, add_generation_prompt=False)
        if history_messages
        else []
    )
    full_ids = _chat_ids(
        tokenizer,
        [*history_messages, reader_message],
        add_generation_prompt=True,
    )
    if full_ids[: len(history_ids)] != history_ids:
        raise ValueError(
            "Chat template does not preserve the rendered history as a token prefix; "
            "refusing an ambiguous truncation splice"
        )
    suffix_ids = full_ids[len(history_ids) :]
    if not suffix_ids:
        raise ValueError("chat template produced no current-query/generation suffix")
    if len(suffix_ids) > max_context_tokens:
        raise ValueError(
            f"sample {question.sample_id}: current MCQ plus generation cue needs "
            f"{len(suffix_ids)} tokens, exceeding budget {max_context_tokens}; "
            "the current query/options are never truncated"
        )

    keep_history = min(len(history_ids), max_context_tokens - len(suffix_ids))
    if keep_history == len(history_ids):
        kept_history_ids = history_ids
    elif truncation == "tail":
        kept_history_ids = history_ids[-keep_history:] if keep_history else []
    else:
        kept_history_ids = history_ids[:keep_history]
    input_ids = (*kept_history_ids, *suffix_ids)
    if len(input_ids) > max_context_tokens:
        raise AssertionError("context budgeting invariant failed")
    return EncodedFullContext(
        sample_id=question.sample_id,
        persona_id=episode.persona_id,
        input_ids=tuple(input_ids),
        gold_index=question.correct_index,
        gold_letter=question.correct_letter,
        tags=question.tags,
        history_tokens_total=len(history_ids),
        history_tokens_kept=keep_history,
        reader_suffix_tokens=len(suffix_ids),
        condition=condition,
        history_persona_id=(
            source_episode.persona_id if source_episode is not None else None
        ),
        swap_index=swap_index if condition == "swapped_history" else None,
    )


def filter_clean_episodes(
    dataset: LoadedPersonaDataset,
    *,
    excluded_sample_ids: set[str],
    detected_overlap_sample_ids: set[str],
) -> tuple[PersonaEpisode, ...]:
    """Drop configured/detected target overlaps while retaining safe co-user MCQs."""
    excluded = excluded_sample_ids | detected_overlap_sample_ids
    clean = tuple(
        replace(
            episode,
            questions=tuple(
                question
                for question in episode.questions
                if question.sample_id not in excluded
            ),
        )
        for episode in dataset.episodes
    )
    clean = tuple(episode for episode in clean if episode.questions)
    if not clean:
        raise ValueError("every question was removed by clean filtering")
    return clean


def limit_episodes(
    episodes: Sequence[PersonaEpisode],
    *,
    include_persona_ids: set[str],
    max_personas: int,
    max_queries: int,
    selection_seed: int,
    shuffle_personas: bool,
) -> tuple[PersonaEpisode, ...]:
    selected = list(episodes)
    if include_persona_ids:
        available = {episode.persona_id for episode in selected}
        missing = sorted(include_persona_ids - available)
        if missing:
            raise ValueError(f"requested persona IDs are missing: {missing}")
        selected = [
            episode
            for episode in selected
            if episode.persona_id in include_persona_ids
        ]
    if shuffle_personas:
        random.Random(selection_seed).shuffle(selected)
    if max_personas > 0:
        selected = selected[:max_personas]

    remaining = max_queries
    limited: list[PersonaEpisode] = []
    for episode in selected:
        if max_queries > 0:
            if remaining <= 0:
                break
            questions = episode.questions[:remaining]
            remaining -= len(questions)
        else:
            questions = episode.questions
        if questions:
            limited.append(replace(episode, questions=tuple(questions)))
    if not limited:
        raise ValueError("episode limits selected no questions")
    return tuple(limited)


def build_deranged_swap_assignments(
    episodes: Sequence[PersonaEpisode],
    *,
    num_swaps: int,
    seed: int,
) -> dict[str, tuple[PersonaEpisode, ...]]:
    """Build fixed, distinct cyclic derangements over the selected personas."""
    if num_swaps < 1:
        raise ValueError("num_swaps must be >= 1")
    persona_ids = [episode.persona_id for episode in episodes]
    if len(set(persona_ids)) != len(persona_ids):
        raise ValueError("swap construction needs exactly one episode per persona")
    if len(episodes) < 2:
        raise ValueError("swapped_history needs at least two personas")
    if num_swaps > len(episodes) - 1:
        raise ValueError(
            f"num_swaps={num_swaps} exceeds the {len(episodes) - 1} unique "
            "different-persona histories available per query"
        )

    ordered = sorted(episodes, key=lambda episode: episode.persona_id)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    offsets = list(range(1, len(ordered)))
    rng.shuffle(offsets)
    offsets = offsets[:num_swaps]
    assignments: dict[str, tuple[PersonaEpisode, ...]] = {}
    for target_index, target in enumerate(ordered):
        sources = tuple(
            ordered[(target_index + offset) % len(ordered)] for offset in offsets
        )
        if any(source.persona_id == target.persona_id for source in sources):
            raise AssertionError("derangement construction produced a fixed point")
        assignments[target.persona_id] = sources
    return assignments


def load_clean_eval_split(
    *,
    csv_path: Path,
    split: str,
    data_root: Path,
    window: str,
    option_seed: int,
    manifest: Mapping[str, Any],
    extra_excluded_persona_ids: set[str],
    extra_excluded_sample_ids: set[str],
    content_overlap_policy: str,
    include_persona_ids: set[str],
    max_personas: int,
    max_queries: int,
    selection_seed: int,
    shuffle_personas: bool,
) -> tuple[tuple[PersonaEpisode, ...], dict[str, Any]]:
    manifest_personas = {
        str(value)
        for value in manifest.get("exclude_persona_ids_from_train_and_val", [])
    }
    # Apply the same clean-v1 behavior as the prefix pilot.  In particular,
    # persona 78 is removed from benchmark too so benchmark users are unseen.
    excluded_personas = manifest_personas | extra_excluded_persona_ids
    configured_samples = {
        str(value) for value in manifest.get("exclude_sample_ids_all_windows", [])
    } | extra_excluded_sample_ids
    loader_policy = "warn" if content_overlap_policy == "drop" else content_overlap_policy
    dataset = load_personamem_text(
        csv_path,
        split=split,
        window=window,
        data_root=data_root,
        shuffle_seed=option_seed,
        shuffle_round=0,
        content_overlap_policy=loader_policy,
        exclude_persona_ids=excluded_personas,
    )
    detected = (
        {
            warning.split(":", 1)[0]
            for warning in dataset.content_overlap_warnings
        }
        if content_overlap_policy == "drop"
        else set()
    )
    clean = filter_clean_episodes(
        dataset,
        excluded_sample_ids=configured_samples,
        detected_overlap_sample_ids=detected,
    )
    selected = limit_episodes(
        clean,
        include_persona_ids=include_persona_ids,
        max_personas=max_personas,
        max_queries=max_queries,
        selection_seed=selection_seed,
        shuffle_personas=shuffle_personas,
    )
    return selected, {
        "csv_path": str(csv_path),
        "split": split,
        "window": window,
        "option_shuffle": {"seed": option_seed, "round": 0},
        "excluded_persona_ids": sorted(excluded_personas),
        "configured_excluded_sample_ids": sorted(configured_samples),
        "detected_overlap_sample_ids": sorted(detected),
        "loader_rows_seen": dataset.rows_seen,
        "loader_invalid_mcqs_skipped": dataset.rows_skipped_invalid_mcq,
        "selected_persona_ids": sorted(
            episode.persona_id for episode in selected
        ),
        "selected_personas": len(selected),
        "selected_queries": sum(len(episode.questions) for episode in selected),
    }


def iter_encoded_examples(
    tokenizer: Any,
    episodes: Sequence[PersonaEpisode],
    *,
    max_context_tokens: int,
    truncation: str,
) -> Iterator[EncodedFullContext]:
    # Re-encode history per query by design: this is the repeat-full-context oracle.
    for episode in episodes:
        for question in episode.questions:
            yield encode_full_context(
                tokenizer,
                episode,
                question,
                max_context_tokens=max_context_tokens,
                truncation=truncation,
            )


def iter_condition_examples(
    tokenizer: Any,
    episodes: Sequence[PersonaEpisode],
    *,
    conditions: Sequence[str],
    swap_assignments: Mapping[str, Sequence[PersonaEpisode]],
    max_context_tokens: int,
    truncation: str,
) -> Iterator[EncodedFullContext]:
    """Yield all requested interventions over exactly the same MCQ objects."""
    for episode in episodes:
        for question in episode.questions:
            for condition in conditions:
                if condition == "query_only":
                    yield encode_full_context(
                        tokenizer,
                        episode,
                        question,
                        max_context_tokens=max_context_tokens,
                        truncation=truncation,
                        condition=condition,
                    )
                elif condition == "correct_history":
                    yield encode_full_context(
                        tokenizer,
                        episode,
                        question,
                        max_context_tokens=max_context_tokens,
                        truncation=truncation,
                        condition=condition,
                        history_episode=episode,
                    )
                elif condition == "swapped_history":
                    try:
                        source_episodes = swap_assignments[episode.persona_id]
                    except KeyError as error:
                        raise ValueError(
                            f"no swap assignment for persona {episode.persona_id}"
                        ) from error
                    for swap_index, source_episode in enumerate(source_episodes):
                        yield encode_full_context(
                            tokenizer,
                            episode,
                            question,
                            max_context_tokens=max_context_tokens,
                            truncation=truncation,
                            condition=condition,
                            history_episode=source_episode,
                            swap_index=swap_index,
                        )
                else:
                    raise ValueError(f"unknown condition: {condition}")


def buffered_length_batches(
    examples: Iterable[EncodedFullContext],
    *,
    batch_size: int,
    sort_buffer_size: int,
) -> Iterator[list[EncodedFullContext]]:
    """Locally sort by context length to reduce padding without global materialization."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if sort_buffer_size < batch_size:
        raise ValueError("sort_buffer_size must be >= batch_size")
    iterator = iter(examples)
    while True:
        buffer: list[EncodedFullContext] = []
        for _ in range(sort_buffer_size):
            try:
                buffer.append(next(iterator))
            except StopIteration:
                break
        if not buffer:
            return
        buffer.sort(key=lambda item: item.context_tokens)
        for start in range(0, len(buffer), batch_size):
            yield buffer[start : start + batch_size]


def _pad_batch(
    examples: Sequence[EncodedFullContext],
    *,
    pad_token_id: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = max(example.context_tokens for example in examples)
    input_ids = torch.full(
        (len(examples), max_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    last_indices = torch.empty(len(examples), dtype=torch.long, device=device)
    for row, example in enumerate(examples):
        length = example.context_tokens
        input_ids[row, :length] = torch.tensor(
            example.input_ids, dtype=torch.long, device=device
        )
        attention_mask[row, :length] = 1
        last_indices[row] = length - 1
    return input_ids, attention_mask, last_indices


@torch.inference_mode()
def forced_choice_batch(
    model: Any,
    examples: Sequence[EncodedFullContext],
    letter_token_ids: Sequence[int],
    *,
    pad_token_id: int,
    device: str | torch.device,
) -> list[int]:
    """Choose among A/B/C/D logits only; the global vocabulary argmax is irrelevant."""
    if not examples:
        return []
    input_ids, attention_mask, last_indices = _pad_batch(
        examples, pad_token_id=pad_token_id, device=device
    )
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    rows = torch.arange(len(examples), device=input_ids.device)
    last_logits = outputs.logits[rows, last_indices]
    choices = torch.as_tensor(
        letter_token_ids, dtype=torch.long, device=last_logits.device
    )
    return last_logits.index_select(-1, choices).argmax(dim=-1).cpu().tolist()


def _tag_dict(tags: AuditTags) -> dict[str, Any]:
    return asdict(tags)


def distance_bin(distance: int | None) -> str:
    if distance is None:
        return "missing"
    if distance < 0:
        return "negative"
    for upper in (4096, 8192, 16384, 32768):
        if distance < upper:
            lower = 0 if upper == 4096 else upper // 2
            return f"[{lower},{upper})"
    return "[32768,inf)"


def _group_accuracy(
    records: Sequence[Mapping[str, Any]], field: str
) -> dict[str, dict[str, int | float]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        value = record["tags"][field]
        groups[str(value)].append(record)
    return {
        value: {
            "n": len(rows),
            "correct": sum(bool(row["correct"]) for row in rows),
            "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        }
        for value, rows in sorted(groups.items())
    }


def _quantile(numbers: Sequence[float | int], fraction: float) -> float | None:
    if not numbers:
        return None
    ordered = sorted(float(value) for value in numbers)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def numeric_summary(numbers: Iterable[int | float | None]) -> dict[str, Any]:
    values = [float(value) for value in numbers if value is not None]
    if not values:
        return {
            "n": 0,
            "min": None,
            "p50": None,
            "p95": None,
            "mean": None,
            "max": None,
        }
    return {
        "n": len(values),
        "min": min(values),
        "p50": statistics.median(values),
        "p95": _quantile(values, 0.95),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize zero records")
    by_persona: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_persona[str(record["persona_id"])].append(record)
    per_persona = {
        persona_id: {
            "n": len(rows),
            "correct": sum(bool(row["correct"]) for row in rows),
            "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        }
        for persona_id, rows in sorted(by_persona.items())
    }
    correct = sum(bool(record["correct"]) for record in records)

    distance_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        distance_groups[
            distance_bin(record["tags"]["distance_to_related_snippet"])
        ].append(record)
    return {
        "num_queries": len(records),
        "num_personas": len(by_persona),
        "correct": correct,
        "accuracy_micro": correct / len(records),
        "accuracy_persona_macro": statistics.fmean(
            row["accuracy"] for row in per_persona.values()
        ),
        "per_persona": per_persona,
        "subgroups": {
            field: _group_accuracy(records, field) for field in SUBGROUP_FIELDS
        },
        "distance": {
            "official_tokens_summary": numeric_summary(
                record["tags"]["distance_to_related_snippet"]
                for record in records
            ),
            "bins": {
                label: {
                    "n": len(rows),
                    "correct": sum(bool(row["correct"]) for row in rows),
                    "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
                }
                for label, rows in sorted(distance_groups.items())
            },
        },
    }


def _paired_prediction_metrics(
    reference_records: Sequence[Mapping[str, Any]],
    intervention_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pair by target sample while allowing repeated swap interventions."""
    reference_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in reference_records:
        key = (str(record["persona_id"]), str(record["sample_id"]))
        if key in reference_by_key:
            raise ValueError(f"duplicate reference record for {key}")
        reference_by_key[key] = record

    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for intervention in intervention_records:
        key = (
            str(intervention["persona_id"]),
            str(intervention["sample_id"]),
        )
        if key not in reference_by_key:
            raise ValueError(f"intervention has no matching reference query: {key}")
        pairs.append((reference_by_key[key], intervention))
    if not pairs:
        raise ValueError("paired metrics need at least one pair")

    reference_correct = [bool(reference["correct"]) for reference, _ in pairs]
    intervention_correct = [
        bool(intervention["correct"]) for _, intervention in pairs
    ]
    flips = [
        int(reference["prediction_index"]) != int(intervention["prediction_index"])
        for reference, intervention in pairs
    ]
    persona_differences: dict[str, list[int]] = defaultdict(list)
    for (reference, intervention), left_correct, right_correct in zip(
        pairs, reference_correct, intervention_correct
    ):
        persona_differences[str(reference["persona_id"])].append(
            int(left_correct) - int(right_correct)
        )
    return {
        "num_pairs": len(pairs),
        "reference_accuracy": sum(reference_correct) / len(pairs),
        "intervention_accuracy": sum(intervention_correct) / len(pairs),
        "reference_minus_intervention_micro": (
            sum(reference_correct) - sum(intervention_correct)
        )
        / len(pairs),
        "reference_minus_intervention_persona_macro": statistics.fmean(
            statistics.fmean(values) for values in persona_differences.values()
        ),
        "prediction_flip_count": sum(flips),
        "prediction_flip_rate": sum(flips) / len(pairs),
        "same_prediction_rate": 1.0 - sum(flips) / len(pairs),
        "reference_correct_to_intervention_incorrect_count": sum(
            left and not right
            for left, right in zip(reference_correct, intervention_correct)
        ),
        "reference_correct_to_intervention_incorrect_rate": sum(
            left and not right
            for left, right in zip(reference_correct, intervention_correct)
        )
        / len(pairs),
        "reference_incorrect_to_intervention_correct_count": sum(
            not left and right
            for left, right in zip(reference_correct, intervention_correct)
        ),
        "reference_incorrect_to_intervention_correct_rate": sum(
            not left and right
            for left, right in zip(reference_correct, intervention_correct)
        )
        / len(pairs),
        "both_correct_rate": sum(
            left and right
            for left, right in zip(reference_correct, intervention_correct)
        )
        / len(pairs),
        "both_incorrect_rate": sum(
            not left and not right
            for left, right in zip(reference_correct, intervention_correct)
        )
        / len(pairs),
    }


def summarize_condition_records(
    records: Sequence[Mapping[str, Any]],
    *,
    conditions: Sequence[str],
) -> dict[str, Any]:
    """Report each intervention plus paired correct-history counterfactuals."""
    by_condition_records = {
        condition: [
            record for record in records if record["condition"] == condition
        ]
        for condition in conditions
    }
    by_condition = {
        condition: summarize_records(rows)
        for condition, rows in by_condition_records.items()
    }
    if "swapped_history" in by_condition:
        swap_rows = by_condition_records["swapped_history"]
        swap_indices = sorted(
            {int(record["swap_index"]) for record in swap_rows}
        )
        by_condition["swapped_history"]["by_swap_index"] = {
            str(index): {
                key: value
                for key, value in summarize_records(
                    [
                        record
                        for record in swap_rows
                        if int(record["swap_index"]) == index
                    ]
                ).items()
                if key
                in {
                    "num_queries",
                    "num_personas",
                    "correct",
                    "accuracy_micro",
                    "accuracy_persona_macro",
                }
            }
            for index in swap_indices
        }

    primary_condition = (
        "correct_history"
        if "correct_history" in by_condition
        else conditions[0]
    )
    output = dict(by_condition[primary_condition])
    output["primary_condition"] = primary_condition
    output["by_condition"] = by_condition
    paired: dict[str, Any] = {}
    correct_rows = by_condition_records.get("correct_history")
    if correct_rows and by_condition_records.get("query_only"):
        paired["correct_history_vs_query_only"] = _paired_prediction_metrics(
            correct_rows, by_condition_records["query_only"]
        )
    if correct_rows and by_condition_records.get("swapped_history"):
        swap_rows = by_condition_records["swapped_history"]
        correct_vs_swap = _paired_prediction_metrics(correct_rows, swap_rows)
        correct_vs_swap["by_swap_index"] = {
            str(index): _paired_prediction_metrics(
                correct_rows,
                [
                    record
                    for record in swap_rows
                    if int(record["swap_index"]) == index
                ],
            )
            for index in sorted(
                {int(record["swap_index"]) for record in swap_rows}
            )
        }
        paired["correct_history_vs_swapped_history"] = correct_vs_swap
    output["paired"] = paired
    return output


def _cuda_synchronize(device: str | torch.device) -> None:
    parsed = torch.device(device)
    if parsed.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(parsed)


def evaluate(
    model: Any,
    tokenizer: Any,
    episodes: Sequence[PersonaEpisode],
    letter_token_ids: Sequence[int],
    *,
    device: str | torch.device,
    max_context_tokens: int,
    truncation: str,
    batch_size: int,
    sort_buffer_size: int,
    save_records: bool,
    conditions: Sequence[str] = ("correct_history",),
    swap_assignments: Mapping[str, Sequence[PersonaEpisode]] | None = None,
) -> dict[str, Any]:
    model.eval()
    if hasattr(model, "parameters"):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer needs pad_token_id before evaluation")

    parsed_device = torch.device(device)
    if parsed_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(parsed_device)
    records: list[dict[str, Any]] = []
    forward_latencies: list[float] = []
    tokenization_seconds = 0.0
    evaluation_start = time.perf_counter()

    def timed_examples() -> Iterator[EncodedFullContext]:
        nonlocal tokenization_seconds
        iterator = iter_condition_examples(
            tokenizer,
            episodes,
            conditions=conditions,
            swap_assignments=swap_assignments or {},
            max_context_tokens=max_context_tokens,
            truncation=truncation,
        )
        while True:
            start = time.perf_counter()
            try:
                encoded = next(iterator)
            except StopIteration:
                return
            tokenization_seconds += time.perf_counter() - start
            yield encoded

    for batch in buffered_length_batches(
        timed_examples(),
        batch_size=batch_size,
        sort_buffer_size=sort_buffer_size,
    ):
        _cuda_synchronize(device)
        forward_start = time.perf_counter()
        predictions = forced_choice_batch(
            model,
            batch,
            letter_token_ids,
            pad_token_id=pad_token_id,
            device=device,
        )
        _cuda_synchronize(device)
        forward_latencies.append(time.perf_counter() - forward_start)
        for example, prediction in zip(batch, predictions):
            records.append(
                {
                    "sample_id": example.sample_id,
                    "persona_id": example.persona_id,
                    "gold_index": example.gold_index,
                    "gold_letter": example.gold_letter,
                    "prediction_index": int(prediction),
                    "prediction_letter": chr(ord("A") + int(prediction)),
                    "correct": int(prediction) == example.gold_index,
                    "condition": example.condition,
                    "history_persona_id": example.history_persona_id,
                    "swap_index": example.swap_index,
                    "context_tokens": example.context_tokens,
                    "history_tokens_total": example.history_tokens_total,
                    "history_tokens_kept": example.history_tokens_kept,
                    "history_tokens_truncated": example.history_tokens_truncated,
                    "reader_suffix_tokens": example.reader_suffix_tokens,
                    "tags": _tag_dict(example.tags),
                }
            )
    _cuda_synchronize(device)
    total_seconds = time.perf_counter() - evaluation_start
    metrics = summarize_condition_records(records, conditions=conditions)

    if parsed_device.type == "cuda":
        peak_allocated: int | None = torch.cuda.max_memory_allocated(parsed_device)
        peak_reserved: int | None = torch.cuda.max_memory_reserved(parsed_device)
    else:
        peak_allocated = None
        peak_reserved = None
    result = {
        "metrics": metrics,
        "context": {
            "max_context_tokens": max_context_tokens,
            "history_truncation": truncation,
            "model_inputs_by_condition": {
                condition: sum(
                    record["condition"] == condition for record in records
                )
                for condition in conditions
            },
            "queries_with_truncated_history": sum(
                record["history_tokens_truncated"] > 0 for record in records
            ),
            "context_tokens": numeric_summary(
                record["context_tokens"] for record in records
            ),
            "history_tokens_total": numeric_summary(
                record["history_tokens_total"] for record in records
            ),
            "history_tokens_kept": numeric_summary(
                record["history_tokens_kept"] for record in records
            ),
            "history_tokens_truncated": numeric_summary(
                record["history_tokens_truncated"] for record in records
            ),
            "reader_suffix_tokens": numeric_summary(
                record["reader_suffix_tokens"] for record in records
            ),
        },
        "latency": {
            "end_to_end_seconds": total_seconds,
            "tokenization_seconds": tokenization_seconds,
            "forward_seconds": sum(forward_latencies),
            "batch_forward_seconds": numeric_summary(forward_latencies),
            "num_batches": len(forward_latencies),
            "num_model_inputs": len(records),
            "queries_per_second_end_to_end": len(records) / total_seconds,
            "queries_per_second_forward_only": (
                len(records) / sum(forward_latencies)
                if sum(forward_latencies) > 0
                else None
            ),
        },
        "memory": {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "measurement_scope": "model-loaded evaluation",
        },
    }
    if save_records:
        result["records"] = records
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/personamem_v2"))
    parser.add_argument("--csv", type=Path)
    parser.add_argument(
        "--split", default="benchmark", choices=["train", "val", "benchmark"]
    )
    parser.add_argument("--window", default="32k", choices=["32k", "128k"])
    parser.add_argument("--clean-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--exclude-persona-ids", default="")
    parser.add_argument("--exclude-persona-id", action="append", default=[])
    parser.add_argument("--exclude-sample-ids", default="")
    parser.add_argument("--exclude-sample-id", action="append", default=[])
    parser.add_argument("--include-persona-ids", default="")
    parser.add_argument(
        "--content-overlap-policy",
        default="drop",
        choices=["error", "drop", "off"],
    )
    parser.add_argument("--option-seed", type=int, default=1618)
    parser.add_argument("--selection-seed", type=int, default=31415)
    parser.add_argument(
        "--shuffle-personas", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--max-personas", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=VALID_CONDITIONS,
        default=["correct_history"],
        help="default preserves the original correct-history-only baseline; use "
        "`--conditions query_only correct_history swapped_history` for causal controls",
    )
    parser.add_argument(
        "--num-swaps",
        type=int,
        default=1,
        help="number of distinct fixed deranged histories per query",
    )
    parser.add_argument(
        "--swap-seed",
        type=int,
        default=20260730,
        help="seed for the fixed persona derangements",
    )

    parser.add_argument("--model-path", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-context-tokens", type=int, default=32768)
    parser.add_argument(
        "--history-truncation", default="tail", choices=["head", "tail"]
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--sort-buffer-size",
        type=int,
        default=16,
        help="locally sort encoded prompts by length to reduce padding",
    )
    parser.add_argument(
        "--save-records", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.max_context_tokens < 1:
        parser.error("--max-context-tokens must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.sort_buffer_size < args.batch_size:
        parser.error("--sort-buffer-size must be >= --batch-size")
    if args.max_personas < 0 or args.max_queries < 0:
        parser.error("--max-personas/--max-queries must be >= 0")
    if len(set(args.conditions)) != len(args.conditions):
        parser.error("--conditions must not contain duplicates")
    if args.num_swaps < 1:
        parser.error("--num-swaps must be >= 1")
    if (
        "swapped_history" in args.conditions
        and "correct_history" not in args.conditions
    ):
        parser.error(
            "swapped_history requires correct_history so paired correct-swap "
            "metrics are well-defined"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    data_root = args.data_root.expanduser().resolve()
    csv_path = (
        args.csv.expanduser().resolve()
        if args.csv
        else resolve_split_csv(data_root, args.split)
    )
    manifest_path = args.clean_manifest.expanduser().resolve()
    if not manifest_path.is_file():
        parser.error(f"--clean-manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes, data_metadata = load_clean_eval_split(
        csv_path=csv_path,
        split=args.split,
        data_root=data_root,
        window=args.window,
        option_seed=args.option_seed,
        manifest=manifest,
        extra_excluded_persona_ids=parse_csv_set(
            args.exclude_persona_ids, args.exclude_persona_id
        ),
        extra_excluded_sample_ids=parse_csv_set(
            args.exclude_sample_ids, args.exclude_sample_id
        ),
        content_overlap_policy=args.content_overlap_policy,
        include_persona_ids=parse_csv_set(args.include_persona_ids),
        max_personas=args.max_personas,
        max_queries=args.max_queries,
        selection_seed=args.selection_seed,
        shuffle_personas=args.shuffle_personas,
    )
    swap_assignments = (
        build_deranged_swap_assignments(
            episodes,
            num_swaps=args.num_swaps,
            seed=args.swap_seed,
        )
        if "swapped_history" in args.conditions
        else {}
    )
    serialized_swap_assignments = {
        target_persona_id: [
            source.persona_id for source in source_episodes
        ]
        for target_persona_id, source_episodes in sorted(swap_assignments.items())
    }

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    letter_token_ids = resolve_letter_token_ids(tokenizer)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,
        local_files_only=args.local_files_only,
    ).to(args.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if hasattr(model, "config"):
        model.config.use_cache = False
    model_load_seconds = time.perf_counter() - load_start

    print(
        "[personamem-fullcontext] "
        f"split={args.split}/{args.window} "
        f"personas={len(episodes)} "
        f"queries={sum(len(episode.questions) for episode in episodes)} "
        f"budget={args.max_context_tokens}({args.history_truncation}) "
        f"batch={args.batch_size} conditions={','.join(args.conditions)} "
        f"swaps={args.num_swaps if swap_assignments else 0}; "
        "repeated full history per model input",
        flush=True,
    )
    result = evaluate(
        model,
        tokenizer,
        episodes,
        letter_token_ids,
        device=args.device,
        max_context_tokens=args.max_context_tokens,
        truncation=args.history_truncation,
        batch_size=args.batch_size,
        sort_buffer_size=args.sort_buffer_size,
        save_records=args.save_records,
        conditions=args.conditions,
        swap_assignments=swap_assignments,
    )
    result["latency"]["model_and_tokenizer_load_seconds"] = model_load_seconds
    payload = {
        "protocol": {
            "name": "personamem-v2-frozen-full-context-forced-choice",
            "implementation": "repeated_full_context_per_query",
            "frozen_model": True,
            "forced_choice_letters": "ABCD",
            "one_token_per_choice": True,
            "cached_kv_write_once_branching": False,
            "history_reencoded_per_query": True,
            "current_mcq_never_truncated": True,
            "conditions": list(args.conditions),
            "num_fixed_deranged_swaps": (
                args.num_swaps if swap_assignments else 0
            ),
            "same_query_and_options_across_conditions": True,
            "note": (
                "Cached-KV write-once branching is not implemented; every query "
                "repeats its persona history."
            ),
        },
        "args": {
            **vars(args),
            "data_root": str(args.data_root),
            "csv": str(args.csv) if args.csv else None,
            "clean_manifest": str(args.clean_manifest),
            "output": str(args.output),
        },
        "model": {
            "path": args.model_path,
            "dtype": args.dtype,
            "attention_implementation": args.attn_impl,
            "device": args.device,
            "letter_token_ids": dict(zip("ABCD", letter_token_ids)),
        },
        "data": {
            **data_metadata,
            "clean_manifest_path": str(manifest_path),
            "clean_manifest": manifest,
            "fixed_deranged_swap_history_persona_ids": serialized_swap_assignments,
        },
        "result": result,
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    metrics = result["metrics"]
    print(
        "[personamem-fullcontext] "
        f"micro={metrics['accuracy_micro']:.4f} "
        f"persona_macro={metrics['accuracy_persona_macro']:.4f} "
        f"time={result['latency']['end_to_end_seconds']:.1f}s "
        f"saved={output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
