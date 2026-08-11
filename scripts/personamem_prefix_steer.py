#!/usr/bin/env python3
"""Leakage-safe PersonaMem-v2 pilot for prefix-memory steering.

Protocol
--------
* The frozen Qwen3 backbone writes one persona history once.
* K future MCQ queries are read in one batch from the same frozen prefix state.
* The writer receives only ``PersonaEpisode.writer``.  It never receives a query,
  answer, option, label, or CSV annotation.
* Training uses either full-vocabulary next-token CE (legacy default) or
  four-choice CE.  Evaluation is forced choice over exactly those four protocol
  token IDs.
* Optional identity contrast reads the same K queries under one deterministic
  wrong-persona history and margins the gold A-D log-probability below correct
  memory.
* Train options are deterministically reshuffled every update; eval option order
  is fixed by one seed and ``shuffle_round=0``.
* The same checkpoint is evaluated as base, window-only, swapped-persona memory,
  and correct-persona memory.

``--memory-mode none --P 0`` is an independently trainable query-only steer model.
``--memory-mode pooled_steer --read-mode broadcast --P 0`` is the strong P=0
history-conditioned baseline: each layer pools the history once to one
query-independent persona vector and broadcasts it directly to the same delta heads.
``--memory-mode hybrid --read-mode pooled_plus_prefix --P 64`` is the additive
paper method: the same WRITE produces that pooled vector and P written slots, then
the query read sent to the same delta heads is
``pooled + gate * Attn(query, written_prefix_only)``.  Its runtime
``prefix_off`` condition is the exact same-weights pooled branch.
Identity-contrast pilots should add ``--task-loss four_choice`` together with
non-zero ``--identity-contrast-lambda``; the default task loss remains
``full_vocab`` solely for compatibility with earlier checkpoints and runs.

This is intentionally a short-context pilot.  WRITE-only mode keeps the history
forward on the frozen backbone path while retaining the written prefix graph for
the query loss.  Gradient checkpointing must not be enabled across those two
stateful forwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PosixPath
from typing import Any, Iterable, Iterator, Sequence

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Keep direct ``python scripts/...py`` invocation independent of editable installs.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.personamem_v2_data import (
        MCQExample,
        PersonaEpisode,
        WriterInput,
        load_personamem_text,
        resolve_split_csv,
    )
except ModuleNotFoundError:  # Direct invocation: python scripts/personamem_prefix_steer.py
    from personamem_v2_data import (  # type: ignore[no-redef]
        MCQExample,
        PersonaEpisode,
        WriterInput,
        load_personamem_text,
        resolve_split_csv,
    )

from deltamem.core.global_prefix import SEG_CTX, SEG_QRY
from deltamem.core.prefix_steer import (
    PrefixSteerConfig,
    attach_prefix_steer,
    clear_frozen_memory,
    freeze_backbone_keep_steer,
    has_frozen_memory,
    is_steer_param_name,
    set_hybrid_pool_off,
    set_hybrid_prefix_off,
    set_steer_enabled,
    set_steer_segments,
    set_window_only,
    set_write_only,
)


DEFAULT_LAYERS = "0,3,6,9,12,15,18,21,24,27,30,33"
# ``correct`` is retained for every existing log consumer. ``correct_full`` is its
# explicit paper-facing alias; ``prefix_off`` is populated only by the additive
# hybrid and is otherwise null.
CONDITIONS = (
    "base",
    "window",
    "swap",
    "correct",
    "correct_full",
    "prefix_off",
)
TASK_LOSSES = ("full_vocab", "four_choice")
READER_PROTOCOLS = ("legacy", "official_qwen")
# The official generation prompt requests reasoning followed by a boxed answer.
# This cheap auxiliary classifier supplies a valid empty reasoning block and the
# label-independent boxed prefix, then supervises only the next lowercase letter.
# It is intentionally not described as full generative SFT.
OFFICIAL_BOXED_CLASSIFICATION_PREFIX = "<think></think>\n\n\\boxed{"
DISTANCE_BUCKETS = (
    "0-4k",
    "4-8k",
    "8-16k",
    "16-32k",
    "32k+",
    "unknown",
)
DEFAULT_HOLDOUT_SALT = "personamem-v2-paper-dev-v1"
# Union over official text train/val/benchmark at 32k and 128k.  All seven are
# user-query matches; the audit found no answer-option matches.
KNOWN_TARGET_OVERLAP_SAMPLE_IDS = frozenset(
    {
        "30d8d4ed5a8f40ae3258",
        "4fac4f3b381bae48762d",
        "5a84d2826cf5ad207002",
        "7ea6851663c471fe0077",
        "a82a0de2841dab3fe44b",
        "e415bfca48efb86b50bb",
        "e9b64566c65886df1b3d",
    }
)


@dataclass(frozen=True)
class ReaderBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    segments: torch.Tensor
    last_indices: torch.Tensor
    target_token_ids: torch.Tensor
    target_indices: torch.Tensor


@dataclass(frozen=True)
class TrainStepResult:
    """Differentiable objective plus detached-at-report-time diagnostics."""

    loss: torch.Tensor
    ce_loss: torch.Tensor
    identity_contrast_loss: torch.Tensor | None
    gold_log_probability_gap: torch.Tensor | None
    gold_probability_gap: torch.Tensor | None
    target_persona_id: str
    donor_persona_id: str | None
    query_count: int
    sample_ids: tuple[str, ...]

    def scalar_metrics(self) -> dict[str, float | str | int | None]:
        """Return JSON-safe scalars without retaining either writer graph."""

        def scalar(value: torch.Tensor | None) -> float | None:
            return (
                None
                if value is None
                else float(value.detach().float().cpu().item())
            )

        return {
            "loss": scalar(self.loss),
            "ce_loss": scalar(self.ce_loss),
            "identity_contrast_loss": scalar(self.identity_contrast_loss),
            "gold_log_probability_gap": scalar(
                self.gold_log_probability_gap
            ),
            "gold_probability_gap": scalar(self.gold_probability_gap),
            "target_persona_id": self.target_persona_id,
            "donor_persona_id": self.donor_persona_id,
            "query_count": self.query_count,
        }


TRAIN_DIAGNOSTIC_KEYS = (
    "loss",
    "ce_loss",
    "identity_contrast_loss",
    "gold_log_probability_gap",
    "gold_probability_gap",
)


def mean_train_diagnostics(
    rows: Sequence[Mapping[str, float | str | int | None]],
    *,
    weight_by_query_count: bool = False,
) -> dict[str, float | int | None]:
    """Average numeric objective diagnostics over one reporting interval."""

    summary: dict[str, float | int | None] = {"steps": len(rows)}
    for key in TRAIN_DIAGNOSTIC_KEYS:
        values = [
            (
                float(row[key]),
                (
                    int(row.get("query_count", 0) or 0)
                    if weight_by_query_count
                    else 1
                ),
            )
            for row in rows
            if row.get(key) is not None
        ]
        total_weight = sum(weight for _, weight in values)
        summary[key] = (
            sum(value * weight for value, weight in values) / total_weight
            if total_weight
            else None
        )
    summary["identity_steps"] = sum(
        row.get("identity_contrast_loss") is not None for row in rows
    )
    return summary


def chunks(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    if size < 1:
        raise ValueError("chunk size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


class HybridPoolDropoutSchedule:
    """Dedicated deterministic RNG for hybrid pooled-branch dropout.

    This stream is intentionally independent of example sampling, option shuffling,
    PyTorch dropout, and CUDA RNGs.  Exactly one decision is drawn per training
    microstep, and its full state is checkpointed so a resumed run makes the same
    next decision as an uninterrupted run.
    """

    def __init__(self, probability: float, *, seed: int) -> None:
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("hybrid pool-drop probability must be in [0,1]")
        self.probability = float(probability)
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self.draw_count = 0
        self.drop_count = 0

    def next(self) -> bool:
        dropped = self._rng.random() < self.probability
        self.draw_count += 1
        self.drop_count += int(dropped)
        return dropped

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "probability": self.probability,
            "seed": self.seed,
            "rng_state": self._rng.getstate(),
            "draw_count": self.draw_count,
            "drop_count": self.drop_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state.get("version", -1)) != 1:
            raise ValueError("unsupported hybrid pool-drop checkpoint version")
        if float(state.get("probability", -1.0)) != self.probability:
            raise ValueError(
                "hybrid pool-drop checkpoint probability does not match"
            )
        if int(state.get("seed", -1)) != self.seed:
            raise ValueError("hybrid pool-drop checkpoint seed does not match")
        draw_count = int(state.get("draw_count", -1))
        drop_count = int(state.get("drop_count", -1))
        if draw_count < 0 or not 0 <= drop_count <= draw_count:
            raise ValueError("invalid hybrid pool-drop checkpoint counters")
        self._rng.setstate(state["rng_state"])
        self.draw_count = draw_count
        self.drop_count = drop_count


class CyclicChunkSampler:
    """Deterministic persona and within-persona no-replacement cycle sampler.

    Personas are visited once per shuffled persona cycle.  For each selected
    persona, questions are consumed from a shuffled permutation without
    replacement.  If a requested chunk crosses the end of that permutation, a
    new permutation starts only after every question in the prior cycle was
    consumed.  This permits an exact K-label microbatch even when a persona has
    fewer than K questions while retaining auditable no-replacement cycles.
    """

    def __init__(
        self, episodes: Sequence[PersonaEpisode], *, seed: int
    ) -> None:
        if not episodes:
            raise ValueError("cyclic chunk sampler needs at least one episode")
        if len({episode.persona_id for episode in episodes}) != len(episodes):
            raise ValueError("cyclic chunk sampler needs unique persona IDs")
        if any(not episode.questions for episode in episodes):
            raise ValueError("cyclic chunk sampler found an empty persona")
        self._episodes = tuple(
            sorted(episodes, key=lambda episode: episode.persona_id)
        )
        self._rng = random.Random(seed)
        self._persona_order: list[int] = []
        self._persona_offset = 0
        self._question_orders: dict[str, list[int]] = {}
        self._question_offsets: dict[str, int] = {}
        self._last_question_index: dict[str, int] = {}
        self.persona_cycles_started = 0
        self.question_cycles_started: Counter[str] = Counter()

    def _start_persona_cycle(self) -> None:
        self._persona_order = list(range(len(self._episodes)))
        self._rng.shuffle(self._persona_order)
        self._persona_offset = 0
        self.persona_cycles_started += 1

    def _start_question_cycle(self, episode: PersonaEpisode) -> None:
        order = list(range(len(episode.questions)))
        self._rng.shuffle(order)
        previous = self._last_question_index.get(episode.persona_id)
        if len(order) > 1 and previous is not None and order[0] == previous:
            order = order[1:] + order[:1]
        self._question_orders[episode.persona_id] = order
        self._question_offsets[episode.persona_id] = 0
        self.question_cycles_started[episode.persona_id] += 1

    def next_chunk(
        self,
        count: int,
        *,
        cross_question_cycle: bool = True,
    ) -> tuple[PersonaEpisode, tuple[MCQExample, ...]]:
        if count < 1:
            raise ValueError("cyclic chunk size must be positive")
        if self._persona_offset >= len(self._persona_order):
            self._start_persona_cycle()
        episode = self._episodes[self._persona_order[self._persona_offset]]
        self._persona_offset += 1

        if not cross_question_cycle:
            offset = self._question_offsets.get(episode.persona_id, 0)
            order = self._question_orders.get(episode.persona_id, [])
            if offset >= len(order):
                self._start_question_cycle(episode)
                offset = 0
                order = self._question_orders[episode.persona_id]
            take = min(count, len(order) - offset)
            indices = order[offset : offset + take]
            self._question_offsets[episode.persona_id] = offset + take
            self._last_question_index[episode.persona_id] = indices[-1]
            return (
                episode,
                tuple(episode.questions[index] for index in indices),
            )

        selected: list[MCQExample] = []
        while len(selected) < count:
            offset = self._question_offsets.get(episode.persona_id, 0)
            order = self._question_orders.get(episode.persona_id, [])
            if offset >= len(order):
                self._start_question_cycle(episode)
                offset = 0
                order = self._question_orders[episode.persona_id]
            take = min(count - len(selected), len(order) - offset)
            indices = order[offset : offset + take]
            selected.extend(episode.questions[index] for index in indices)
            self._question_offsets[episode.persona_id] = offset + take
            self._last_question_index[episode.persona_id] = indices[-1]
        return episode, tuple(selected)

    def state_dict(self) -> dict[str, Any]:
        """Return a complete, CPU/pickle-safe deterministic resume state."""

        return {
            "version": 1,
            "episode_sample_ids": {
                episode.persona_id: tuple(
                    question.sample_id for question in episode.questions
                )
                for episode in self._episodes
            },
            "rng_state": self._rng.getstate(),
            "persona_order": list(self._persona_order),
            "persona_offset": self._persona_offset,
            "question_orders": {
                persona_id: list(order)
                for persona_id, order in self._question_orders.items()
            },
            "question_offsets": dict(self._question_offsets),
            "last_question_index": dict(self._last_question_index),
            "persona_cycles_started": self.persona_cycles_started,
            "question_cycles_started": dict(self.question_cycles_started),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a sampler state, rejecting a changed dataset or corruption."""

        if int(state.get("version", -1)) != 1:
            raise ValueError("unsupported cyclic sampler checkpoint version")
        expected_samples = {
            episode.persona_id: tuple(
                question.sample_id for question in episode.questions
            )
            for episode in self._episodes
        }
        checkpoint_samples = {
            str(persona_id): tuple(str(value) for value in sample_ids)
            for persona_id, sample_ids in dict(
                state.get("episode_sample_ids", {})
            ).items()
        }
        if checkpoint_samples != expected_samples:
            raise ValueError(
                "cyclic sampler checkpoint dataset/order does not match "
                "the selected training episodes"
            )

        persona_order = [int(value) for value in state["persona_order"]]
        if sorted(persona_order) not in (
            [],
            list(range(len(self._episodes))),
        ):
            raise ValueError("invalid cyclic sampler persona permutation")
        persona_offset = int(state["persona_offset"])
        if not 0 <= persona_offset <= len(persona_order):
            raise ValueError("invalid cyclic sampler persona offset")

        question_orders = {
            str(persona_id): [int(value) for value in order]
            for persona_id, order in dict(state["question_orders"]).items()
        }
        question_offsets = {
            str(persona_id): int(value)
            for persona_id, value in dict(state["question_offsets"]).items()
        }
        last_question_index = {
            str(persona_id): int(value)
            for persona_id, value in dict(
                state["last_question_index"]
            ).items()
        }
        episode_sizes = {
            episode.persona_id: len(episode.questions)
            for episode in self._episodes
        }
        for persona_id, order in question_orders.items():
            if persona_id not in episode_sizes:
                raise ValueError(
                    f"unknown cyclic sampler persona {persona_id!r}"
                )
            if sorted(order) != list(range(episode_sizes[persona_id])):
                raise ValueError(
                    f"invalid question permutation for persona {persona_id!r}"
                )
            offset = question_offsets.get(persona_id)
            if offset is None or not 0 <= offset <= len(order):
                raise ValueError(
                    f"invalid question offset for persona {persona_id!r}"
                )
        if set(question_offsets) != set(question_orders):
            raise ValueError(
                "cyclic sampler question offsets/orders have different personas"
            )
        for persona_id, index in last_question_index.items():
            if (
                persona_id not in episode_sizes
                or not 0 <= index < episode_sizes[persona_id]
            ):
                raise ValueError(
                    f"invalid last question index for persona {persona_id!r}"
                )

        self._rng.setstate(state["rng_state"])
        self._persona_order = persona_order
        self._persona_offset = persona_offset
        self._question_orders = question_orders
        self._question_offsets = question_offsets
        self._last_question_index = last_question_index
        self.persona_cycles_started = int(state["persona_cycles_started"])
        self.question_cycles_started = Counter(
            {
                str(persona_id): int(value)
                for persona_id, value in dict(
                    state["question_cycles_started"]
                ).items()
            }
        )


def simulate_cyclic_label_budget(
    episodes: Sequence[PersonaEpisode],
    *,
    seed: int,
    max_queries_per_write: int,
    labels_per_update: int,
    optimizer_updates: int,
) -> dict[str, Any]:
    """Preflight the exact variable-chunk schedule without touching training."""

    sampler = CyclicChunkSampler(episodes, seed=seed)
    seen_personas: set[str] = set()
    seen_samples: set[str] = set()
    persona_microsteps: Counter[str] = Counter()
    micro_steps = 0
    labels = 0
    for _ in range(optimizer_updates):
        update_labels = 0
        while update_labels < labels_per_update:
            request = min(
                max_queries_per_write,
                labels_per_update - update_labels,
            )
            episode, questions = sampler.next_chunk(
                request, cross_question_cycle=False
            )
            query_count = len(questions)
            if not 0 < query_count <= request:
                raise AssertionError(
                    "cyclic label-budget sampler returned an invalid chunk"
                )
            micro_steps += 1
            labels += query_count
            update_labels += query_count
            seen_personas.add(episode.persona_id)
            seen_samples.update(question.sample_id for question in questions)
            persona_microsteps[episode.persona_id] += 1
    return {
        "optimizer_updates": optimizer_updates,
        "micro_steps": micro_steps,
        "label_exposures": labels,
        "seen_persona_count": len(seen_personas),
        "seen_sample_count": len(seen_samples),
        "persona_microsteps_min": min(persona_microsteps.values()),
        "persona_microsteps_max": max(persona_microsteps.values()),
    }


def parse_layers(value: str) -> tuple[int, ...]:
    layers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not layers:
        raise ValueError("--layers must contain at least one layer index")
    if len(set(layers)) != len(layers) or min(layers) < 0:
        raise ValueError("--layers must be unique non-negative integers")
    return layers


def resolve_memory_mode(mode: str, num_prefix_tokens: int) -> str:
    """Resolve the backwards-compatible CLI shorthand to an explicit architecture."""
    if mode == "auto":
        return "prefix" if num_prefix_tokens > 0 else "none"
    return mode


def build_prefix_config(args: argparse.Namespace) -> PrefixSteerConfig:
    """Translate validated CLI arguments into one checkpointed architecture config."""
    memory_mode = args.resolved_memory_mode
    prefix_enabled = memory_mode in ("prefix", "hybrid")
    hybrid_enabled = memory_mode == "hybrid"
    return PrefixSteerConfig(
        num_prefix_tokens=args.num_prefix_tokens,
        sliding_window_size=args.sliding_window,
        mem_num_heads=args.mem_num_heads,
        mem_head_dim=args.head_dim,
        steer_mode="deltamem",
        normal_attends_prefix=prefix_enabled,
        prefix_sees_query=False,
        prefix_init_std=args.prefix_init_std,
        prefix_init_dist=args.prefix_init_dist,
        prefix_write=prefix_enabled,
        read_prefix_only=(
            memory_mode == "prefix" and args.read_mode == "prefix_only"
        ),
        memory_mode="dynamic",
        write_ctx_only=prefix_enabled,
        prefix_write_layout=args.prefix_write_layout,
        prefix_write_overlap_tokens=args.prefix_write_overlap_tokens,
        pool_reads=memory_mode == "prefix" and args.read_mode == "pool",
        history_pool_mode=(
            args.history_pool
            if memory_mode in ("pooled_steer", "hybrid")
            else "none"
        ),
        hybrid_read_mode=(
            "pooled_plus_prefix" if hybrid_enabled else "none"
        ),
        hybrid_prefix_gate_mode=args.hybrid_prefix_gate_mode,
        hybrid_prefix_gate_init=args.hybrid_prefix_gate_init,
        steer_layers=args.steer_layers,
        steer_gain=args.steer_gain,
        delta_heads=args.delta_heads,
    )


def parse_persona_ids(csv_value: str, repeated: Iterable[str]) -> set[str]:
    values = set(repeated)
    values.update(item.strip() for item in csv_value.split(","))
    return {value for value in values if value}


def parse_sample_ids(csv_value: str, repeated: Iterable[str]) -> set[str]:
    values = set(repeated)
    values.update(item.strip() for item in csv_value.split(","))
    return {value for value in values if value}


def _stable_seed(seed: int, *parts: str) -> int:
    payload = ":".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def select_identity_donor(
    target_persona_id: str,
    episodes: Sequence[PersonaEpisode],
    *,
    seed: int,
    step: int,
) -> PersonaEpisode:
    """Select a deterministic, order-independent non-self history donor."""

    donors = sorted(
        (
            episode
            for episode in episodes
            if episode.persona_id != target_persona_id
        ),
        key=lambda episode: episode.persona_id,
    )
    if not donors:
        raise ValueError(
            "identity contrast requires at least two distinct training personas"
        )
    index = _stable_seed(
        seed, "identity-donor", str(step), str(target_persona_id)
    ) % len(donors)
    donor = donors[index]
    if donor.persona_id == target_persona_id:
        raise AssertionError("identity donor selection produced a self-swap")
    return donor


def limit_episodes(
    episodes: Sequence[PersonaEpisode],
    *,
    max_personas: int,
    max_queries: int,
    selection_seed: int,
    shuffle_personas: bool,
) -> tuple[PersonaEpisode, ...]:
    """Deterministically select personas, then cap total queries without splitting history."""
    selected = list(episodes)
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


def include_persona_episodes(
    episodes: Sequence[PersonaEpisode],
    persona_ids: set[str],
    *,
    label: str,
) -> tuple[PersonaEpisode, ...]:
    """Restrict an experimental split to explicit IDs and reject silent misses."""
    if not persona_ids:
        return tuple(episodes)
    available = {episode.persona_id for episode in episodes}
    missing = sorted(persona_ids - available)
    if missing:
        raise ValueError(f"{label}: requested persona IDs are missing: {missing}")
    return tuple(
        episode for episode in episodes if episode.persona_id in persona_ids
    )


def resolve_letter_token_ids(
    tokenizer: Any, *, reader_protocol: str = "legacy"
) -> tuple[int, int, int, int]:
    """Find one consistent one-token spelling for the protocol's four labels.

    Legacy prompts predict A-D directly.  ``official_qwen`` appends
    ``OFFICIAL_BOXED_CLASSIFICATION_PREFIX`` and predicts a-d, matching the
    official boxed-answer spelling.  The latter also verifies that separately
    tokenizing the fixed prefix and target letter has the same boundary as
    tokenizing their concatenation.
    """
    if reader_protocol == "official_qwen":
        prefix_ids = _flatten_token_ids(
            tokenizer(
                OFFICIAL_BOXED_CLASSIFICATION_PREFIX,
                add_special_tokens=False,
            )["input_ids"]
        )
        token_ids: list[int] = []
        for letter in "abcd":
            combined_ids = _flatten_token_ids(
                tokenizer(
                    OFFICIAL_BOXED_CLASSIFICATION_PREFIX + letter,
                    add_special_tokens=False,
                )["input_ids"]
            )
            if combined_ids[: len(prefix_ids)] != prefix_ids:
                raise ValueError(
                    "official boxed prefix changes tokenization at the label boundary"
                )
            suffix = combined_ids[len(prefix_ids) :]
            if len(suffix) != 1:
                raise ValueError(
                    f"official boxed label {letter!r} is not one token"
                )
            token_ids.append(int(suffix[0]))
        if len(set(token_ids)) != 4:
            raise ValueError(
                f"a/b/c/d map to non-unique token IDs: {tuple(token_ids)}"
            )
        return tuple(token_ids)  # type: ignore[return-value]
    if reader_protocol != "legacy":
        raise ValueError(
            f"unknown reader protocol {reader_protocol!r}; "
            f"expected one of {READER_PROTOCOLS}"
        )
    for prefix in ("", " "):
        encoded = [
            tokenizer(prefix + letter, add_special_tokens=False)["input_ids"]
            for letter in "ABCD"
        ]
        if all(len(ids) == 1 for ids in encoded):
            token_ids = tuple(int(ids[0]) for ids in encoded)
            if len(set(token_ids)) != 4:
                raise ValueError(f"A/B/C/D map to non-unique token IDs: {token_ids}")
            return token_ids  # type: ignore[return-value]
    raise ValueError(
        "Tokenizer has no consistent single-token spelling for A/B/C/D; "
        "this script deliberately requires letter-token CE"
    )


def _flatten_token_ids(value: Any) -> list[int]:
    # transformers>=5 may return BatchEncoding from apply_chat_template even when
    # tokenize=True and return_dict was not explicitly requested.
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


def encode_history(
    tokenizer: Any,
    writer: WriterInput,
    *,
    persona_id: str = "",
    max_history_tokens: int,
    truncation: str,
) -> list[int]:
    """Tokenize only a schema-checked WriterInput; no episode/MCQ is accepted."""
    writer.assert_safe_schema()
    messages = writer.to_messages()
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        ids = _flatten_token_ids(encoded)
    except (AttributeError, TypeError, ValueError):
        ids = _flatten_token_ids(
            tokenizer(writer.to_text(), add_special_tokens=False)["input_ids"]
        )
    if not ids:
        raise ValueError(f"persona {persona_id!r} has an empty tokenized history")
    if max_history_tokens > 0 and len(ids) > max_history_tokens:
        if truncation == "tail":
            ids = ids[-max_history_tokens:]
        elif truncation == "head":
            ids = ids[:max_history_tokens]
        else:
            raise ValueError(f"unknown history truncation: {truncation}")
    return ids


def reader_prompt(question: MCQExample) -> str:
    # Labels and correct_index are intentionally absent from prompt construction.
    return question.reader.to_prompt() + "\nAnswer:"


def official_reader_messages(question: MCQExample) -> list[dict[str, str]]:
    """Build the exact official Qwen system and MCQ user messages."""
    from scripts.personamem_official_hf_eval import (
        OFFICIAL_MCQ_TEMPLATE,
        OFFICIAL_SYSTEM_PROMPT,
        OFFICIAL_THINK_INSTRUCTION,
    )

    option_lines = "\n".join(
        f"({chr(ord('a') + index)}) {option}"
        for index, option in enumerate(question.reader.options)
    )
    user_prompt = (
        question.reader.query
        + OFFICIAL_MCQ_TEMPLATE.format(options_text=option_lines)
        + OFFICIAL_THINK_INSTRUCTION
    )
    return [
        {"role": "system", "content": OFFICIAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def encode_reader_prompt(
    tokenizer: Any,
    question: MCQExample,
    *,
    reader_protocol: str = "legacy",
) -> list[int]:
    """Encode a query at the selected protocol's auxiliary answer position."""
    if reader_protocol == "official_qwen":
        try:
            encoded = tokenizer.apply_chat_template(
                official_reader_messages(question),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError as exc:
            raise TypeError(
                "official_qwen requires a Qwen3-compatible chat template "
                "accepting enable_thinking"
            ) from exc
        ids = _flatten_token_ids(encoded)
        prefix_ids = _flatten_token_ids(
            tokenizer(
                OFFICIAL_BOXED_CLASSIFICATION_PREFIX,
                add_special_tokens=False,
            )["input_ids"]
        )
        return ids + prefix_ids
    if reader_protocol != "legacy":
        raise ValueError(
            f"unknown reader protocol {reader_protocol!r}; "
            f"expected one of {READER_PROTOCOLS}"
        )
    prompt = reader_prompt(question)
    try:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        ids = _flatten_token_ids(encoded)
    except (AttributeError, TypeError, ValueError):
        ids = _flatten_token_ids(
            tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )
    return ids


def collate_reader_batch(
    tokenizer: Any,
    questions: Sequence[MCQExample],
    letter_token_ids: Sequence[int],
    *,
    device: str | torch.device,
    reader_protocol: str = "legacy",
) -> ReaderBatch:
    if not questions:
        raise ValueError("questions must be non-empty")
    prompt_ids = [
        encode_reader_prompt(
            tokenizer, question, reader_protocol=reader_protocol
        )
        for question in questions
    ]
    if any(not ids for ids in prompt_ids):
        raise ValueError("reader prompt tokenized to an empty sequence")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer needs pad_token_id or eos_token_id")
    batch_size = len(questions)
    max_length = max(map(len, prompt_ids))
    input_ids = torch.full(
        (batch_size, max_length), int(pad_id), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros(
        (batch_size, max_length), dtype=torch.bool, device=device
    )
    segments = torch.full(
        (batch_size, max_length), SEG_QRY, dtype=torch.long, device=device
    )
    last_indices = torch.empty(batch_size, dtype=torch.long, device=device)
    for row, ids in enumerate(prompt_ids):
        length = len(ids)
        input_ids[row, :length] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row, :length] = True
        last_indices[row] = length - 1

    target_indices = torch.tensor(
        [question.correct_index for question in questions],
        dtype=torch.long,
        device=device,
    )
    letter_ids = torch.tensor(letter_token_ids, dtype=torch.long, device=device)
    target_token_ids = letter_ids[target_indices]
    return ReaderBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        segments=segments,
        last_indices=last_indices,
        target_token_ids=target_token_ids,
        target_indices=target_indices,
    )


def final_reader_logits(model: Any, batch: ReaderBatch) -> torch.Tensor:
    set_steer_segments(model, batch.segments, batch.attention_mask)
    output = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask.long(),
        use_cache=False,
    )
    row_indices = torch.arange(
        batch.input_ids.shape[0], dtype=torch.long, device=batch.input_ids.device
    )
    return output.logits[row_indices, batch.last_indices]


def letter_ce_loss(
    model: Any,
    tokenizer: Any,
    questions: Sequence[MCQExample],
    letter_token_ids: Sequence[int],
    *,
    device: str | torch.device,
    task_loss: str = "full_vocab",
    reader_protocol: str = "legacy",
) -> torch.Tensor:
    batch = collate_reader_batch(
        tokenizer,
        questions,
        letter_token_ids,
        device=device,
        reader_protocol=reader_protocol,
    )
    logits = final_reader_logits(model, batch)
    return task_ce_loss_from_logits(
        logits,
        batch,
        letter_token_ids,
        task_loss=task_loss,
    )


def task_ce_loss_from_logits(
    logits: torch.Tensor,
    batch: ReaderBatch,
    letter_token_ids: Sequence[int],
    *,
    task_loss: str,
) -> torch.Tensor:
    """Compute legacy full-vocabulary CE or evaluation-aligned A-D CE."""

    if task_loss == "full_vocab":
        return F.cross_entropy(logits.float(), batch.target_token_ids)
    if task_loss == "four_choice":
        letter_ids = torch.tensor(
            letter_token_ids, dtype=torch.long, device=logits.device
        )
        letter_logits = logits.float().index_select(-1, letter_ids)
        return F.cross_entropy(
            letter_logits, batch.target_indices.to(logits.device)
        )
    raise ValueError(
        f"unknown task loss {task_loss!r}; expected one of {TASK_LOSSES}"
    )


def identity_contrast_terms(
    correct_logits: torch.Tensor,
    wrong_memory_logits: torch.Tensor,
    batch: ReaderBatch,
    letter_token_ids: Sequence[int],
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Contrast the gold A-D score under correct and wrong persona memories.

    The optimization term uses a smooth margin on gold log-probabilities after
    an A-D-only log-softmax.  This removes the raw-logit common-shift shortcut and
    matches forced-choice evaluation.  The probability gap is also reported.
    Non-letter vocabulary logits are deliberately irrelevant to these terms;
    they may still be covered by the independently selected primary task loss.
    """

    if correct_logits.shape != wrong_memory_logits.shape:
        raise ValueError(
            "correct and wrong-memory logits must have identical shapes"
        )
    if correct_logits.ndim != 2:
        raise ValueError("reader logits must have shape [queries, vocabulary]")
    if correct_logits.shape[0] != batch.target_indices.shape[0]:
        raise ValueError("reader logits and targets have different batch sizes")

    letter_ids = torch.tensor(
        letter_token_ids,
        dtype=torch.long,
        device=correct_logits.device,
    )
    target_indices = batch.target_indices.to(correct_logits.device)
    correct_letters = correct_logits.float().index_select(-1, letter_ids)
    wrong_letters = wrong_memory_logits.float().index_select(-1, letter_ids)
    correct_log_probabilities = correct_letters.log_softmax(dim=-1)
    wrong_log_probabilities = wrong_letters.log_softmax(dim=-1)
    gold_correct_log_probability = correct_log_probabilities.gather(
        -1, target_indices.unsqueeze(-1)
    ).squeeze(-1)
    gold_wrong_log_probability = wrong_log_probabilities.gather(
        -1, target_indices.unsqueeze(-1)
    ).squeeze(-1)
    log_probability_gaps = (
        gold_correct_log_probability - gold_wrong_log_probability
    )
    contrast_loss = F.softplus(
        log_probability_gaps.new_tensor(float(margin))
        - log_probability_gaps
    ).mean()

    # Reporting-only diagnostic: avoid retaining a second pair of softmax graphs.
    with torch.no_grad():
        gold_correct_probabilities = gold_correct_log_probability.exp()
        gold_wrong_probabilities = gold_wrong_log_probability.exp()
        probability_gap = (
            gold_correct_probabilities - gold_wrong_probabilities
        ).mean()
    return contrast_loss, log_probability_gaps.mean(), probability_gap


@torch.no_grad()
def forced_choice_predictions(
    model: Any,
    tokenizer: Any,
    questions: Sequence[MCQExample],
    letter_token_ids: Sequence[int],
    *,
    device: str | torch.device,
    batch_size: int,
    reader_protocol: str = "legacy",
) -> list[int]:
    predictions: list[int] = []
    letter_ids = torch.tensor(letter_token_ids, dtype=torch.long, device=device)
    for question_batch in chunks(questions, batch_size):
        batch = collate_reader_batch(
            tokenizer,
            question_batch,
            letter_token_ids,
            device=device,
            reader_protocol=reader_protocol,
        )
        logits = final_reader_logits(model, batch)
        predictions.extend(
            logits.index_select(-1, letter_ids).argmax(dim=-1).cpu().tolist()
        )
    return [int(prediction) for prediction in predictions]


def write_persona_memory(
    model: Any,
    history_ids: Sequence[int],
    *,
    device: str | torch.device,
    grad: bool,
    prefix_enabled: bool,
) -> None:
    """Write history through ``model.model`` so the vocabulary lm_head is not run."""
    clear_frozen_memory(model)
    if not prefix_enabled:
        return
    input_ids = torch.tensor([history_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    segments = torch.full_like(input_ids, SEG_CTX)
    set_window_only(model, False)
    set_steer_enabled(model, True)
    set_write_only(model, True)
    set_steer_segments(model, segments, attention_mask)
    context = nullcontext() if grad else torch.no_grad()
    try:
        with context:
            # Deliberately avoid model(...).logits: the writer has no vocabulary loss.
            writer_output = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask.long(),
                use_cache=False,
            )
            del writer_output
    finally:
        # False stops WRITE-only mode but intentionally preserves the new state.
        set_write_only(model, False)
    frozen = has_frozen_memory(model)
    if not frozen or not all(frozen):
        clear_frozen_memory(model)
        raise RuntimeError(
            "history forward did not populate every prefix-steer layer's frozen memory"
        )


def _questions_for_step(
    episode: PersonaEpisode,
    *,
    count: int,
    rng: random.Random,
    option_shuffle_seed: int,
    option_shuffle_round: int,
    reader_protocol: str = "legacy",
    selected_questions: Sequence[MCQExample] | None = None,
) -> tuple[MCQExample, ...]:
    if selected_questions is not None:
        if not selected_questions:
            raise ValueError("selected_questions must be non-empty")
        selected = tuple(selected_questions)
    elif count <= 0 or count >= len(episode.questions):
        selected = episode.questions
    else:
        selected = tuple(rng.sample(list(episode.questions), count))
    if reader_protocol == "official_qwen":
        # The official Qwen preprocessing fixes option order once from the
        # unfiltered source CSV row.  Never reshuffle it across updates.
        return tuple(selected)
    if reader_protocol != "legacy":
        raise ValueError(
            f"unknown reader protocol {reader_protocol!r}; "
            f"expected one of {READER_PROTOCOLS}"
        )
    shuffled: list[MCQExample] = []
    for question in selected:
        permutation = list(range(len(question.reader.options)))
        random.Random(
            _stable_seed(
                option_shuffle_seed,
                str(option_shuffle_round),
                question.sample_id,
            )
        ).shuffle(permutation)
        old_correct = question.correct_index
        shuffled.append(
            replace(
                question,
                reader=replace(
                    question.reader,
                    options=tuple(
                        question.reader.options[index] for index in permutation
                    ),
                ),
                correct_index=permutation.index(old_correct),
            )
        )
    return tuple(shuffled)


def train_step(
    model: Any,
    tokenizer: Any,
    episode: PersonaEpisode,
    history_ids: Sequence[int],
    letter_token_ids: Sequence[int],
    *,
    device: str | torch.device,
    queries_per_write: int,
    prefix_enabled: bool,
    rng: random.Random,
    option_shuffle_seed: int,
    option_shuffle_round: int,
    reader_protocol: str = "legacy",
    task_loss: str = "full_vocab",
    identity_contrast_lambda: float = 0.0,
    identity_margin: float = 0.0,
    donor_history_ids: Sequence[int] | None = None,
    donor_persona_id: str | None = None,
    selected_questions: Sequence[MCQExample] | None = None,
) -> TrainStepResult:
    """Train on correct memory and optionally contrast one wrong-persona memory.

    The target history is written exactly once, followed by one K-query reader
    forward.  With identity contrast enabled, one distinct donor history is then
    written exactly once and the *same* collated K-query batch is read again.
    Neither writer call receives a query, option, answer, or label.
    """
    if task_loss not in TASK_LOSSES:
        raise ValueError(
            f"unknown task loss {task_loss!r}; expected one of {TASK_LOSSES}"
        )
    if (
        not math.isfinite(identity_contrast_lambda)
        or identity_contrast_lambda < 0
    ):
        raise ValueError("identity_contrast_lambda must be finite and non-negative")
    if not math.isfinite(identity_margin) or identity_margin < 0:
        raise ValueError("identity_margin must be finite and non-negative")
    use_identity_contrast = identity_contrast_lambda > 0
    if use_identity_contrast:
        if not prefix_enabled:
            raise ValueError(
                "identity contrast requires persistent prefix or pooled memory"
            )
        if donor_history_ids is None or donor_persona_id is None:
            raise ValueError(
                "identity contrast requires donor history and persona ID"
            )
        if donor_persona_id == episode.persona_id:
            raise ValueError("identity contrast donor must differ from target persona")

    set_steer_enabled(model, True)
    set_window_only(model, False)
    write_persona_memory(
        model,
        history_ids,
        device=device,
        grad=True,
        prefix_enabled=prefix_enabled,
    )
    questions = _questions_for_step(
        episode,
        count=queries_per_write,
        rng=rng,
        option_shuffle_seed=option_shuffle_seed,
        option_shuffle_round=option_shuffle_round,
        reader_protocol=reader_protocol,
        selected_questions=selected_questions,
    )
    if not use_identity_contrast:
        ce_loss = letter_ce_loss(
            model,
            tokenizer,
            questions,
            letter_token_ids,
            device=device,
            task_loss=task_loss,
            reader_protocol=reader_protocol,
        )
        return TrainStepResult(
            loss=ce_loss,
            ce_loss=ce_loss,
            identity_contrast_loss=None,
            gold_log_probability_gap=None,
            gold_probability_gap=None,
            target_persona_id=episode.persona_id,
            donor_persona_id=None,
            query_count=len(questions),
            sample_ids=tuple(question.sample_id for question in questions),
        )

    batch = collate_reader_batch(
        tokenizer,
        questions,
        letter_token_ids,
        device=device,
        reader_protocol=reader_protocol,
    )
    correct_logits = final_reader_logits(model, batch)
    ce_loss = task_ce_loss_from_logits(
        correct_logits,
        batch,
        letter_token_ids,
        task_loss=task_loss,
    )

    write_persona_memory(
        model,
        donor_history_ids,
        device=device,
        grad=True,
        prefix_enabled=prefix_enabled,
    )
    wrong_memory_logits = final_reader_logits(model, batch)
    contrast_loss, gold_log_probability_gap, gold_probability_gap = (
        identity_contrast_terms(
            correct_logits,
            wrong_memory_logits,
            batch,
            letter_token_ids,
            margin=identity_margin,
        )
    )
    loss = ce_loss + float(identity_contrast_lambda) * contrast_loss
    return TrainStepResult(
        loss=loss,
        ce_loss=ce_loss,
        identity_contrast_loss=contrast_loss,
        gold_log_probability_gap=gold_log_probability_gap,
        gold_probability_gap=gold_probability_gap,
        target_persona_id=episode.persona_id,
        donor_persona_id=donor_persona_id,
        query_count=len(questions),
        sample_ids=tuple(question.sample_id for question in questions),
    )


def _distance_bucket(distance: int | None) -> str:
    if distance is None or distance < 0:
        return "unknown"
    if distance < 4 * 1024:
        return "0-4k"
    if distance < 8 * 1024:
        return "4-8k"
    if distance < 16 * 1024:
        return "8-16k"
    if distance < 32 * 1024:
        return "16-32k"
    return "32k+"


def _tag_values(question: MCQExample) -> dict[str, Any]:
    tags = question.tags
    return {
        "pref_type": str(tags.pref_type),
        "who": str(tags.who),
        "updated": str(tags.updated),
        "sensitive_info": str(tags.sensitive_info),
        "distance_to_related_snippet": tags.distance_to_related_snippet,
        "distance_bucket": _distance_bucket(tags.distance_to_related_snippet),
    }


def make_swap_derangements(
    persona_ids: Sequence[str],
    *,
    count: int,
    seed: int,
) -> tuple[dict[str, str], ...]:
    """Return fixed, bijective persona swaps with no persona mapped to itself."""
    if count < 1:
        raise ValueError("swap derangement count must be positive")
    canonical_ids = sorted(str(persona_id) for persona_id in persona_ids)
    if len(set(canonical_ids)) != len(canonical_ids):
        raise ValueError("evaluation persona IDs must be unique")
    if len(canonical_ids) < 2:
        return ()

    # One seeded canonical cycle plus non-zero cyclic shifts gives deterministic
    # derangements.  The first min(count, n-1) mappings are guaranteed distinct.
    cycle = canonical_ids.copy()
    random.Random(_stable_seed(seed, "persona-swap-cycle")).shuffle(cycle)
    derangements: list[dict[str, str]] = []
    for run_index in range(count):
        shift = 1 + (run_index % (len(cycle) - 1))
        derangements.append(
            {
                persona_id: cycle[(position + shift) % len(cycle)]
                for position, persona_id in enumerate(cycle)
            }
        )
    return tuple(derangements)


def _condition_accuracy(
    records: Sequence[dict[str, Any]],
    condition: str,
    *,
    swap_run_index: int | None = None,
) -> float | None:
    outcomes: list[int] = []
    for record in records:
        if condition == "swap":
            predictions = record["swap_predictions"]
            if swap_run_index is None:
                selected = predictions
            elif swap_run_index < len(predictions):
                selected = [predictions[swap_run_index]]
            else:
                selected = []
            outcomes.extend(
                int(int(prediction) == record["gold_index"])
                for prediction in selected
            )
        else:
            prediction = record["predictions"][condition]
            if prediction is None:
                continue
            outcomes.append(int(int(prediction) == record["gold_index"]))
    return sum(outcomes) / len(outcomes) if outcomes else None


def _persona_macro_accuracy(
    records: Sequence[dict[str, Any]],
    condition: str,
    *,
    swap_run_index: int | None = None,
) -> float | None:
    by_persona: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_persona.setdefault(record["persona_id"], []).append(record)
    persona_scores = [
        score
        for rows in by_persona.values()
        if (
            score := _condition_accuracy(
                rows, condition, swap_run_index=swap_run_index
            )
        )
        is not None
    ]
    return (
        sum(persona_scores) / len(persona_scores)
        if persona_scores
        else None
    )


def _subgroup_accuracy(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in (
        "pref_type",
        "who",
        "updated",
        "sensitive_info",
        "distance_bucket",
    ):
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            groups.setdefault(record["tags"][field], []).append(record)
        group_names = (
            sorted(groups, key=DISTANCE_BUCKETS.index)
            if field == "distance_bucket"
            else sorted(groups)
        )
        output[field] = {
            group: {
                "n": len(groups[group]),
                **{
                    condition: _condition_accuracy(groups[group], condition)
                    for condition in CONDITIONS
                },
            }
            for group in group_names
        }
    return output


@torch.no_grad()
def evaluate(
    model: Any,
    tokenizer: Any,
    episodes: Sequence[PersonaEpisode],
    history_cache: dict[str, list[int]],
    letter_token_ids: Sequence[int],
    *,
    device: str | torch.device,
    query_batch_size: int,
    prefix_enabled: bool,
    hybrid_prefix_ablation: bool = False,
    num_swap_derangements: int = 1,
    swap_seed: int = 4242,
    reader_protocol: str = "legacy",
) -> dict[str, Any]:
    """Evaluate causal memory conditions with fixed persona derangements.

    Additive hybrids additionally evaluate ``prefix_off`` without changing any
    weights or re-writing memory.  This is their exact pooled-steer branch.
    """
    if not episodes:
        raise ValueError("evaluation needs at least one episode")
    # Training may deliberately drop the pooled branch for a microstep.  Evaluation
    # is always the complete method (or the explicit prefix_off pooled-only ablation).
    if hybrid_prefix_ablation:
        set_hybrid_pool_off(model, False)
    model.eval()
    records: list[dict[str, Any]] = []
    derangements = make_swap_derangements(
        [episode.persona_id for episode in episodes],
        count=num_swap_derangements,
        seed=swap_seed,
    )

    for episode in episodes:
        questions = episode.questions

        clear_frozen_memory(model)
        set_window_only(model, False)
        if hybrid_prefix_ablation:
            set_hybrid_prefix_off(model, False)
        set_steer_enabled(model, False)
        pred_base = forced_choice_predictions(
            model,
            tokenizer,
            questions,
            letter_token_ids,
            device=device,
            batch_size=query_batch_size,
            reader_protocol=reader_protocol,
        )

        set_steer_enabled(model, True)
        write_persona_memory(
            model,
            history_cache[episode.persona_id],
            device=device,
            grad=False,
            prefix_enabled=prefix_enabled,
        )
        set_window_only(model, False)
        pred_correct = forced_choice_predictions(
            model,
            tokenizer,
            questions,
            letter_token_ids,
            device=device,
            batch_size=query_batch_size,
            reader_protocol=reader_protocol,
        )
        if hybrid_prefix_ablation:
            set_hybrid_prefix_off(model, True)
            pred_prefix_off: list[int | None] = list(
                forced_choice_predictions(
                    model,
                    tokenizer,
                    questions,
                    letter_token_ids,
                    device=device,
                    batch_size=query_batch_size,
                    reader_protocol=reader_protocol,
                )
            )
            set_hybrid_prefix_off(model, False)
            # Hybrid has deliberately no local-SWA query branch.  Its historical
            # "window" intervention and the explicit prefix-off intervention are
            # therefore the same pooled-only read; do not spend a duplicate pass.
            pred_window = [int(value) for value in pred_prefix_off]
        else:
            pred_prefix_off = [None] * len(questions)
            set_window_only(model, True)
            pred_window = forced_choice_predictions(
                model,
                tokenizer,
                questions,
                letter_token_ids,
                device=device,
                batch_size=query_batch_size,
                reader_protocol=reader_protocol,
            )
            set_window_only(model, False)

        swap_persona_ids: list[str] = []
        swap_predictions: list[list[int]] = []
        for derangement in derangements:
            swap_persona_id = derangement[episode.persona_id]
            if swap_persona_id == episode.persona_id:
                raise AssertionError("persona swap derangement contains a fixed point")
            write_persona_memory(
                model,
                history_cache[swap_persona_id],
                device=device,
                grad=False,
                prefix_enabled=prefix_enabled,
            )
            swap_persona_ids.append(swap_persona_id)
            swap_predictions.append(
                forced_choice_predictions(
                    model,
                    tokenizer,
                    questions,
                    letter_token_ids,
                    device=device,
                    batch_size=query_batch_size,
                    reader_protocol=reader_protocol,
                )
            )
        clear_frozen_memory(model)

        for question_index, (
            question,
            base,
            window,
            correct,
            prefix_off,
        ) in enumerate(
            zip(
                questions,
                pred_base,
                pred_window,
                pred_correct,
                pred_prefix_off,
            )
        ):
            question_swap_predictions = [
                int(predictions[question_index])
                for predictions in swap_predictions
            ]
            records.append(
                {
                    "sample_id": question.sample_id,
                    "persona_id": episode.persona_id,
                    "swap_persona_id": (
                        swap_persona_ids[0] if swap_persona_ids else None
                    ),
                    "swap_persona_ids": swap_persona_ids,
                    "gold_index": question.correct_index,
                    "gold_letter": question.correct_letter,
                    "predictions": {
                        "base": int(base),
                        "window": int(window),
                        # Retained for one-swap consumers; aggregate reporting below
                        # always uses every value in ``swap_predictions``.
                        "swap": (
                            question_swap_predictions[0]
                            if question_swap_predictions
                            else None
                        ),
                        "correct": int(correct),
                        "correct_full": int(correct),
                        "prefix_off": (
                            int(prefix_off)
                            if prefix_off is not None
                            else None
                        ),
                    },
                    "swap_predictions": question_swap_predictions,
                    "prediction_letters": {
                        "base": chr(ord("A") + int(base)),
                        "window": chr(ord("A") + int(window)),
                        "swap": (
                            chr(ord("A") + question_swap_predictions[0])
                            if question_swap_predictions
                            else None
                        ),
                        "swap_runs": [
                            chr(ord("A") + prediction)
                            for prediction in question_swap_predictions
                        ],
                        "correct": chr(ord("A") + int(correct)),
                        "correct_full": chr(ord("A") + int(correct)),
                        "prefix_off": (
                            chr(ord("A") + int(prefix_off))
                            if prefix_off is not None
                            else None
                        ),
                    },
                    "tags": _tag_values(question),
                }
            )

    set_window_only(model, False)
    if hybrid_prefix_ablation:
        set_hybrid_prefix_off(model, False)
    set_steer_enabled(model, True)
    clear_frozen_memory(model)
    accuracy = {
        condition: _condition_accuracy(records, condition)
        for condition in CONDITIONS
    }
    persona_macro_accuracy = {
        condition: _persona_macro_accuracy(records, condition)
        for condition in CONDITIONS
    }
    swap_runs = []
    for run_index, derangement in enumerate(derangements):
        run_accuracy = _condition_accuracy(
            records, "swap", swap_run_index=run_index
        )
        run_macro = _persona_macro_accuracy(
            records, "swap", swap_run_index=run_index
        )
        swap_runs.append(
            {
                "index": run_index,
                "persona_mapping": derangement,
                "accuracy": run_accuracy,
                "persona_macro_accuracy": run_macro,
                "correct_minus_swap": (
                    accuracy["correct"] - run_accuracy
                    if accuracy["correct"] is not None
                    and run_accuracy is not None
                    else None
                ),
            }
        )
    swap_available = bool(derangements)
    return {
        "protocol": {
            "num_personas": len(episodes),
            "num_queries": len(records),
            "queries_per_reader_batch": query_batch_size,
            "correct_history_writes": len(episodes) if prefix_enabled else 0,
            "swap_history_writes": (
                len(episodes) * len(derangements) if prefix_enabled else 0
            ),
            "distinct_swap_persona": swap_available,
            "swap_available": swap_available,
            "requested_swap_derangements": num_swap_derangements,
            "effective_swap_derangements": len(derangements),
            "swap_seed": swap_seed,
            "forced_choice": True,
            "reader_protocol": reader_protocol,
            "hybrid_prefix_ablation": hybrid_prefix_ablation,
            "reader_objective": (
                "official_boxed_lowercase_next_token_auxiliary_classification"
                if reader_protocol == "official_qwen"
                else "legacy_uppercase_next_token_classification"
            ),
            "distance_bucket_edges_tokens": [0, 4096, 8192, 16384, 32768],
            "distance_bucket_labels": list(DISTANCE_BUCKETS),
        },
        "accuracy": accuracy,
        "persona_macro_accuracy": persona_macro_accuracy,
        "paired": {
            "correct_minus_base": accuracy["correct"] - accuracy["base"],
            "correct_minus_window": accuracy["correct"] - accuracy["window"],
            "correct_full_minus_prefix_off": (
                accuracy["correct_full"] - accuracy["prefix_off"]
                if accuracy["prefix_off"] is not None
                else None
            ),
            "correct_minus_swap": (
                accuracy["correct"] - accuracy["swap"]
                if accuracy["swap"] is not None
                else None
            ),
        },
        "persona_macro_paired": {
            "correct_minus_base": (
                persona_macro_accuracy["correct"]
                - persona_macro_accuracy["base"]
            ),
            "correct_minus_window": (
                persona_macro_accuracy["correct"]
                - persona_macro_accuracy["window"]
            ),
            "correct_full_minus_prefix_off": (
                persona_macro_accuracy["correct_full"]
                - persona_macro_accuracy["prefix_off"]
                if persona_macro_accuracy["prefix_off"] is not None
                else None
            ),
            "correct_minus_swap": (
                persona_macro_accuracy["correct"]
                - persona_macro_accuracy["swap"]
                if persona_macro_accuracy["swap"] is not None
                else None
            ),
        },
        "swap_derangements": {
            "requested": num_swap_derangements,
            "effective": len(derangements),
            "seed": swap_seed,
            "available": swap_available,
            "runs": swap_runs,
            "mean_accuracy": accuracy["swap"],
            "mean_persona_macro_accuracy": persona_macro_accuracy["swap"],
        },
        "hybrid_same_weights": {
            "available": hybrid_prefix_ablation,
            "correct_full": accuracy["correct_full"],
            "prefix_off_pooled": accuracy["prefix_off"],
            "swap_full": accuracy["swap"],
            "prefix_gain_over_pooled": (
                accuracy["correct_full"] - accuracy["prefix_off"]
                if accuracy["prefix_off"] is not None
                else None
            ),
            "correct_full_minus_swap": (
                accuracy["correct_full"] - accuracy["swap"]
                if accuracy["swap"] is not None
                else None
            ),
        },
        "subgroups": _subgroup_accuracy(records),
        "records": records,
    }


def _encode_history_cache(
    tokenizer: Any,
    episodes: Sequence[PersonaEpisode],
    *,
    max_history_tokens: int,
    truncation: str,
) -> dict[str, list[int]]:
    cache: dict[str, list[int]] = {}
    for episode in episodes:
        if episode.persona_id not in cache:
            cache[episode.persona_id] = encode_history(
                tokenizer,
                episode.writer,
                persona_id=episode.persona_id,
                max_history_tokens=max_history_tokens,
                truncation=truncation,
            )
    return cache


def _load_dataset(
    *,
    csv_path: Path,
    split: str,
    data_root: Path,
    window: str,
    option_seed: int,
    excluded_ids: set[str],
    excluded_sample_ids: set[str],
    overlap_policy: str,
    reader_protocol: str = "legacy",
) -> tuple[PersonaEpisode, ...]:
    # Eval always uses shuffle_round=0.  It never depends on model seed or call order.
    loader_overlap_policy = "warn" if overlap_policy == "drop" else overlap_policy
    dataset = load_personamem_text(
        csv_path,
        split=split,
        window=window,
        data_root=data_root,
        shuffle_seed=option_seed,
        shuffle_round=0,
        option_shuffle_protocol=(
            "official_qwen"
            if reader_protocol == "official_qwen"
            else "stable"
        ),
        content_overlap_policy=loader_overlap_policy,
        exclude_persona_ids=excluded_ids,
    )
    overlapping_sample_ids: set[str] = set()
    if overlap_policy == "drop":
        overlapping_sample_ids = {
            warning.split(":", 1)[0]
            for warning in dataset.content_overlap_warnings
        }
    all_excluded_samples = overlapping_sample_ids | excluded_sample_ids
    if not all_excluded_samples:
        return dataset.episodes
    # Warning entries are ``sample_id:field``.  Drop the whole MCQ whenever its
    # current query or any answer option occurs verbatim in the history.  Keeping
    # the history is safe for the persona's other, non-overlapping future queries.
    clean = tuple(
        replace(
            episode,
            questions=tuple(
                question
                for question in episode.questions
                if question.sample_id not in all_excluded_samples
            ),
        )
        for episode in dataset.episodes
    )
    clean = tuple(episode for episode in clean if episode.questions)
    if not clean:
        raise ValueError(f"{split}: every question was removed by overlap filtering")
    before = dataset.num_questions
    after = sum(len(episode.questions) for episode in clean)
    print(
        f"[personamem:data] {split}: dropped {before - after} excluded/leaking MCQs "
        f"(configured sample IDs={len(excluded_sample_ids)}, detected overlap="
        f"{len(overlapping_sample_ids)})",
        flush=True,
    )
    return clean


def merge_persona_episodes(
    *episode_groups: Sequence[PersonaEpisode],
    include_persona_ids: set[str] | None = None,
) -> tuple[PersonaEpisode, ...]:
    """Merge question sets while retaining exactly one schema-safe history per persona."""
    merged: dict[str, PersonaEpisode] = {}
    seen_samples: dict[str, set[str]] = {}
    for episodes in episode_groups:
        for episode in episodes:
            if (
                include_persona_ids is not None
                and episode.persona_id not in include_persona_ids
            ):
                continue
            current = merged.get(episode.persona_id)
            if current is None:
                merged[episode.persona_id] = episode
                seen_samples[episode.persona_id] = {
                    question.sample_id for question in episode.questions
                }
                continue
            if current.writer != episode.writer:
                raise ValueError(
                    f"persona {episode.persona_id!r} has different writer histories "
                    "across the merged splits"
                )
            known = seen_samples[episode.persona_id]
            new_questions = tuple(
                question
                for question in episode.questions
                if question.sample_id not in known
            )
            known.update(question.sample_id for question in new_questions)
            merged[episode.persona_id] = replace(
                current, questions=current.questions + new_questions
            )
    return tuple(merged.values())


def persona_holdout_ids(
    train_episodes: Sequence[PersonaEpisode],
    *,
    size: int,
    fraction: float,
    salt: str,
    seed: int,
) -> set[str]:
    """Select lowest SHA256 scores; default score is exactly ``salt + ':' + pid``."""
    persona_ids = {episode.persona_id for episode in train_episodes}
    if size < 0:
        raise ValueError("holdout size must be >= 0")
    if not 0.0 <= fraction < 1.0:
        raise ValueError("holdout fraction must be in [0, 1)")
    count = size if size > 0 else int(round(fraction * len(persona_ids)))
    if count == 0:
        return set()
    if count >= len(persona_ids):
        raise ValueError("persona holdout must leave at least one training persona")
    prefix = salt if seed == 0 else f"{salt}seed={seed}:"

    def score(persona_id: str) -> tuple[bytes, str]:
        separator = "" if prefix.endswith(":") else ":"
        payload = f"{prefix}{separator}{persona_id}".encode("utf-8")
        return hashlib.sha256(payload).digest(), persona_id

    return set(sorted(persona_ids, key=score)[:count])


def deterministic_persona_holdout(
    train_episodes: Sequence[PersonaEpisode],
    eval_episodes: Sequence[PersonaEpisode],
    *,
    size: int = 0,
    fraction: float,
    salt: str = DEFAULT_HOLDOUT_SALT,
    seed: int,
    dev_source: str = "train+val",
) -> tuple[tuple[PersonaEpisode, ...], tuple[PersonaEpisode, ...], set[str]]:
    """Build an unseen-persona dev split without adding ordinary val rows to train.

    Holdout IDs are selected from official train.  They are completely removed
    from training.  Dev can aggregate those users' official train+val questions
    (default) or use val questions only.  All non-heldout val questions are
    ignored; they never leak into training.
    """
    heldout = persona_holdout_ids(
        train_episodes,
        size=size,
        fraction=fraction,
        salt=salt,
        seed=seed,
    )
    if not heldout:
        return tuple(train_episodes), tuple(eval_episodes), set()
    train_kept = tuple(
        episode for episode in train_episodes if episode.persona_id not in heldout
    )
    if dev_source == "train+val":
        eval_kept = merge_persona_episodes(
            train_episodes, eval_episodes, include_persona_ids=heldout
        )
    elif dev_source == "val":
        eval_kept = tuple(
            episode for episode in eval_episodes if episode.persona_id in heldout
        )
    else:
        raise ValueError(f"unknown dev source: {dev_source}")
    if not train_kept or not eval_kept:
        raise ValueError("persona holdout removed every train or eval episode")
    return train_kept, eval_kept, heldout


def _save_checkpoint(
    model: Any,
    path: Path,
    *,
    config: PrefixSteerConfig,
    args: argparse.Namespace,
    metadata: Mapping[str, Any] | None = None,
    training_state: Mapping[str, Any] | None = None,
) -> None:
    state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if is_steer_param_name(name)
    }
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload = {
        "state": state,
        "cfg": asdict(config),
        "args": serializable_args,
        "metadata": dict(metadata or {}),
    }
    if training_state is not None:
        payload["training_state"] = dict(training_state)
    # A killed process must leave either the prior complete checkpoint or the
    # new complete checkpoint, never a partially written file.
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def _load_checkpoint(
    model: Any,
    path: str,
    *,
    config: PrefixSteerConfig | None = None,
) -> Mapping[str, Any]:
    # Older local checkpoints stored pathlib.PosixPath values in their metadata.
    # Keep PyTorch 2.6+'s weights-only loader enabled and allowlist only that
    # inert metadata type rather than falling back to unsafe arbitrary unpickling.
    with torch.serialization.safe_globals([PosixPath]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    raw_cfg = checkpoint.get("cfg") if isinstance(checkpoint, Mapping) else None
    if config is not None and isinstance(raw_cfg, Mapping):
        # These switches can alter behavior without altering most tensor shapes, so a
        # state_dict-only compatibility check is insufficient. Missing keys resolve to
        # the exact legacy behavior for old checkpoints.
        metadata_defaults = {
            "prefix_write_layout": "global",
            "prefix_write_overlap_tokens": 0,
            "hybrid_read_mode": "none",
            "hybrid_prefix_gate_mode": "fixed",
            "hybrid_prefix_gate_init": 0.1,
        }
        mismatches = []
        for key, default in metadata_defaults.items():
            saved = raw_cfg.get(key, default)
            expected = getattr(config, key)
            if saved != expected:
                mismatches.append(
                    f"{key}: checkpoint={saved!r}, current={expected!r}"
                )
        if mismatches:
            raise RuntimeError(
                "checkpoint/config metadata mismatch: " + "; ".join(mismatches)
            )
    state = checkpoint["state"] if "state" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    steer_names = {
        name for name, _ in model.named_parameters() if is_steer_param_name(name)
    }
    missing_steer = sorted(name for name in missing if name in steer_names)
    if missing_steer or unexpected:
        raise RuntimeError(
            f"checkpoint/config mismatch: missing steer={missing_steer[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    return checkpoint


def _training_data_fingerprint(
    episodes: Sequence[PersonaEpisode],
) -> str:
    """Fingerprint every WRITE history and supervised MCQ in training order."""

    digest = hashlib.sha256()
    for episode in episodes:
        digest.update(episode.persona_id.encode("utf-8"))
        digest.update(b"\0history\0")
        digest.update(episode.writer.to_text().encode("utf-8"))
        for question in episode.questions:
            digest.update(b"\0question\0")
            digest.update(question.sample_id.encode("utf-8"))
            digest.update(str(question.correct_index).encode("ascii"))
            for option in question.reader.options:
                digest.update(b"\0option\0")
                digest.update(option.encode("utf-8"))
    return digest.hexdigest()


RESUME_CRITICAL_ARGS = (
    "model_path",
    "dtype",
    "attn_impl",
    "num_prefix_tokens",
    "mem_num_heads",
    "head_dim",
    "layers",
    "sliding_window",
    "memory_mode",
    "resolved_memory_mode",
    "history_pool",
    "read_mode",
    "prefix_write_layout",
    "prefix_write_overlap_tokens",
    "hybrid_prefix_gate_mode",
    "hybrid_prefix_gate_init",
    "hybrid_pool_drop_prob",
    "steer_gain",
    "delta_heads",
    "seed",
    "queries_per_write",
    "grad_accum_steps",
    "labels_per_update",
    "train_sampler",
    "reader_protocol",
    "task_loss",
    "identity_contrast_lambda",
    "identity_margin",
    "identity_donor_seed",
    "train_option_seed",
    "max_history_tokens",
    "history_truncation",
    "lr",
    "prefix_lr",
    "weight_decay",
    "grad_clip",
)
RESUME_LEGACY_DEFAULTS = {
    "prefix_write_layout": "global",
    "prefix_write_overlap_tokens": 0,
    "hybrid_prefix_gate_mode": "fixed",
    "hybrid_prefix_gate_init": 0.1,
    "hybrid_pool_drop_prob": 0.0,
}


def _validate_resume_configuration(
    checkpoint: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    train_data_fingerprint: str,
) -> Mapping[str, Any]:
    """Return training state after rejecting an inexact training continuation."""

    training_state = checkpoint.get("training_state")
    if not isinstance(training_state, Mapping):
        raise ValueError(
            "--resume-checkpoint requires a checkpoint containing training_state; "
            "older/model-only checkpoints must use --init-checkpoint"
        )
    if int(training_state.get("version", -1)) != 1:
        raise ValueError("unsupported training checkpoint version")
    if training_state.get("train_data_fingerprint") != train_data_fingerprint:
        raise ValueError(
            "resume checkpoint training data/history fingerprint does not match"
        )
    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, Mapping):
        raise ValueError("resume checkpoint has no argument manifest")
    mismatches = []
    current_args = vars(args)
    for key in RESUME_CRITICAL_ARGS:
        saved = checkpoint_args.get(key, RESUME_LEGACY_DEFAULTS.get(key))
        if saved != current_args.get(key):
            mismatches.append(
                f"{key}: checkpoint={saved!r}, "
                f"current={current_args.get(key)!r}"
            )
    if mismatches:
        raise ValueError(
            "resume checkpoint configuration mismatch: " + "; ".join(mismatches)
        )
    completed = int(training_state.get("completed_optimizer_updates", -1))
    if completed < 0:
        raise ValueError("invalid completed optimizer update count in checkpoint")
    if not args.eval_only and args.steps < completed:
        raise ValueError(
            f"--steps={args.steps} is below checkpoint update {completed}; "
            "--steps is the total target, not additional updates"
        )
    return training_state


def _make_training_state(
    *,
    optimizer: torch.optim.Optimizer,
    train_data_fingerprint: str,
    optimizer_updates: int,
    micro_steps: int,
    actual_label_exposures: int,
    train_rng: random.Random,
    hybrid_pool_dropout: HybridPoolDropoutSchedule,
    cyclic_sampler: CyclicChunkSampler | None,
    seen_train_personas: set[str],
    persona_microstep_exposures: Counter[str],
    persona_query_exposures: Counter[str],
    sample_query_exposures: Counter[str],
    interval_train_metrics: Sequence[
        Mapping[str, float | str | int | None]
    ],
    history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Capture all mutable state needed to continue at the next update."""

    return {
        "version": 1,
        "train_data_fingerprint": train_data_fingerprint,
        "completed_optimizer_updates": optimizer_updates,
        "micro_steps": micro_steps,
        "actual_label_exposures": actual_label_exposures,
        "optimizer": optimizer.state_dict(),
        "train_rng_state": train_rng.getstate(),
        "hybrid_pool_dropout": hybrid_pool_dropout.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "cyclic_sampler": (
            cyclic_sampler.state_dict()
            if cyclic_sampler is not None
            else None
        ),
        "seen_train_personas": sorted(seen_train_personas),
        "persona_microstep_exposures": dict(persona_microstep_exposures),
        "persona_query_exposures": dict(persona_query_exposures),
        "sample_query_exposures": dict(sample_query_exposures),
        "interval_train_metrics": [
            dict(metrics) for metrics in interval_train_metrics
        ],
        "history": [dict(row) for row in history],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", default="data/personamem_v2", type=Path
    )
    parser.add_argument(
        "--clean-manifest",
        type=Path,
        default=REPO_ROOT / "configs" / "personamem_v2_clean_v1.json",
        help="reproducibility manifest supplying cross-window sample exclusions, "
        "persona exclusions, and the paper-dev hash split",
    )
    parser.add_argument("--train-split", default="train", choices=["train"])
    parser.add_argument(
        "--eval-split", default="val", choices=["val", "benchmark"]
    )
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--eval-csv", type=Path)
    parser.add_argument("--window", default="32k", choices=["32k", "128k"])
    parser.add_argument(
        "--exclude-persona-ids",
        default="78",
        help="comma-separated IDs; defaults to 78, which overlaps benchmark in the "
        "current official snapshot",
    )
    parser.add_argument(
        "--exclude-persona-id",
        action="append",
        default=[],
        help="repeatable additional persona ID exclusion",
    )
    parser.add_argument(
        "--include-persona-ids",
        default="",
        help="comma-separated exact persona subset applied to both train and eval; "
        "intended for reproducible smoke/overfit diagnostics",
    )
    parser.add_argument(
        "--exclude-sample-ids",
        default="",
        help="comma-separated sample IDs in addition to --clean-manifest",
    )
    parser.add_argument(
        "--exclude-sample-id",
        action="append",
        default=[],
        help="repeatable additional sample ID exclusion",
    )
    parser.add_argument(
        "--content-overlap-policy",
        default="drop",
        choices=["error", "drop", "off"],
        help="drop (default) removes any MCQ whose current query/option occurs "
        "verbatim in its writer history; error aborts; off disables the audit",
    )
    parser.add_argument("--max-history-tokens", type=int, default=4096)
    parser.add_argument(
        "--history-truncation", default="tail", choices=["head", "tail"]
    )
    parser.add_argument(
        "--max-personas",
        type=int,
        default=4,
        help="cap personas independently in train and eval; <=0 means all",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=64,
        help="cap total queries independently in train and eval; <=0 means all",
    )
    parser.add_argument("--queries-per-write", type=int, default=4)
    parser.add_argument(
        "--train-sampler",
        default="random_persona",
        choices=(
            "random_persona",
            "cyclic_chunks",
            "cyclic_label_budget",
        ),
        help="random_persona preserves the pilot sampler; cyclic_chunks visits "
        "personas cyclically and consumes each persona's shuffled questions "
        "without replacement before starting another question cycle; "
        "cyclic_label_budget uses variable no-cycle-crossing chunks and "
        "--labels-per-update for exact global effective batches",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="history-write/query microsteps accumulated per optimizer update; "
        "--steps counts optimizer updates (fixed samplers only)",
    )
    parser.add_argument(
        "--labels-per-update",
        type=int,
        default=0,
        help="exact number of query labels accumulated per optimizer update for "
        "--train-sampler cyclic_label_budget; each microstep is weighted by "
        "its query count, and --grad-accum-steps is not used",
    )
    parser.add_argument(
        "--reader-protocol",
        default="legacy",
        choices=READER_PROTOCOLS,
        help="legacy uses the pilot A-D prompt; official_qwen uses the official "
        "PersonaMem-v2 Qwen system/MCQ/think prompt and fixed row-index shuffle, "
        "then trains a clearly labelled boxed-lowercase next-token auxiliary "
        "classifier (not full generative SFT)",
    )
    parser.add_argument(
        "--task-loss",
        default="full_vocab",
        choices=TASK_LOSSES,
        help="primary correct-memory CE: legacy full vocabulary (default), or "
        "four protocol-label choices aligned with forced-choice evaluation; use "
        "four_choice for identity-contrast pilots",
    )
    parser.add_argument(
        "--identity-contrast-lambda",
        type=float,
        default=0.0,
        help="weight on wrong-persona identity contrast; 0 disables the donor "
        "write/reader forward and preserves ordinary CE training",
    )
    parser.add_argument(
        "--identity-margin",
        type=float,
        default=0.0,
        help="desired correct-minus-wrong gold A-D log-probability margin "
        "inside softplus",
    )
    parser.add_argument(
        "--identity-donor-seed",
        type=int,
        default=7331,
        help="fixed seed for deterministic, non-self training-history donors",
    )
    parser.add_argument("--selection-seed", type=int, default=31415)
    parser.add_argument(
        "--persona-holdout-size",
        type=int,
        default=None,
        help="for eval-split=val, exact number of lowest-hash official-train "
        "personas to reserve; default comes from --clean-manifest (80)",
    )
    parser.add_argument(
        "--persona-holdout-fraction",
        type=float,
        default=0.0,
        help="fallback fractional holdout when --persona-holdout-size is 0",
    )
    parser.add_argument(
        "--persona-holdout-salt",
        default="",
        help="SHA256 salt; default comes from --clean-manifest",
    )
    parser.add_argument(
        "--persona-holdout-seed",
        type=int,
        default=0,
        help="optional alternate split seed (0 preserves the manifest's exact hash)",
    )
    parser.add_argument(
        "--dev-source",
        default="train+val",
        choices=["train+val", "val"],
        help="questions for held-out personas; non-heldout val rows are never trained",
    )
    parser.add_argument("--train-option-seed", type=int, default=2718)
    parser.add_argument(
        "--eval-option-seed",
        type=int,
        default=1618,
        help="fixed option shuffle; eval always uses shuffle_round=0",
    )
    parser.add_argument(
        "--num-swap-derangements",
        type=int,
        default=1,
        help="number of fixed no-self persona permutations used for swap evaluation; "
        "use 3 for paper runs",
    )
    parser.add_argument(
        "--swap-seed",
        type=int,
        default=4242,
        help="fixed seed for swap-persona derangements",
    )

    parser.add_argument(
        "--model-path", default="Qwen/Qwen3-4B-Instruct-2507"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument(
        "--P", "--num-prefix-tokens", dest="num_prefix_tokens", type=int, default=64
    )
    parser.add_argument("--mem-num-heads", type=int, default=1)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--sliding-window", type=int, default=256)
    parser.add_argument(
        "--memory-mode",
        default="auto",
        choices=["auto", "prefix", "pooled_steer", "hybrid", "none"],
        help="persistent history architecture; auto preserves legacy behavior "
        "(P>0 => prefix, P=0 => none); hybrid is additive pooled steer + "
        "query-conditioned written prefix",
    )
    parser.add_argument(
        "--history-pool",
        default="attn",
        choices=["attn", "mean"],
        help="WRITE-time query-independent pooling for --memory-mode pooled_steer",
    )
    parser.add_argument(
        "--read-mode",
        default="pool",
        choices=[
            "pool",
            "standard",
            "prefix_only",
            "broadcast",
            "pooled_plus_prefix",
        ],
        help="prefix reader; broadcast selects single-vector pooled steer; "
        "pooled_plus_prefix selects the additive hybrid reader",
    )
    parser.add_argument(
        "--prefix-write-layout",
        choices=["global", "partitioned"],
        default="global",
        help="global lets every prefix probe read the full valid history; "
        "partitioned routes ordered contiguous chunks to different slots",
    )
    parser.add_argument(
        "--prefix-write-overlap-tokens",
        type=int,
        default=0,
        help="partitioned WRITE overlap on each side, measured in valid history tokens",
    )
    parser.add_argument(
        "--hybrid-prefix-gate-mode",
        choices=["fixed", "learned_scalar", "learned_channel"],
        default="fixed",
    )
    parser.add_argument(
        "--hybrid-prefix-gate-init",
        type=float,
        default=0.1,
        help="initial additive prefix weight; fixed accepts (0,1], learned gates (0,1)",
    )
    parser.add_argument(
        "--hybrid-pool-drop-prob",
        type=float,
        default=0.0,
        help="training-only probability of dropping the pooled branch for one "
        "whole hybrid correct+donor microstep; evaluation always uses the pool",
    )
    parser.add_argument("--steer-gain", type=float, default=0.1)
    parser.add_argument("--delta-heads", default="qkvo")
    parser.add_argument("--prefix-init-std", type=float, default=0.7)
    parser.add_argument(
        "--prefix-init-dist",
        default="normal",
        choices=["normal", "uniform", "orthogonal"],
    )

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--prefix-lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help="load steer weights only and start a fresh optimizer/sampler run",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default="",
        help="resume model, optimizer, RNG, cyclic-sampler, counters, and update "
        "number from a checkpoint produced with --save-every or a final checkpoint",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="atomically refresh OUTPUT.resume.pt every N optimizer updates; "
        "0 disables periodic recovery checkpoints",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.num_prefix_tokens < 0:
        parser.error("--P must be >= 0")
    if args.prefix_write_overlap_tokens < 0:
        parser.error("--prefix-write-overlap-tokens must be non-negative")
    if (
        args.prefix_write_layout == "global"
        and args.prefix_write_overlap_tokens != 0
    ):
        parser.error(
            "--prefix-write-overlap-tokens requires "
            "--prefix-write-layout partitioned"
        )
    if args.mem_num_heads < 1 or args.head_dim < 1:
        parser.error("--mem-num-heads and --head-dim must be positive")
    if args.max_history_tokens < 1:
        parser.error("--max-history-tokens must be positive")
    if args.persona_holdout_size is not None and args.persona_holdout_size < 0:
        parser.error("--persona-holdout-size must be >= 0")
    if not 0.0 <= args.persona_holdout_fraction < 1.0:
        parser.error("--persona-holdout-fraction must be in [0,1)")
    if args.queries_per_write < 1:
        parser.error("--queries-per-write must be positive")
    if args.grad_accum_steps < 1:
        parser.error("--grad-accum-steps must be positive")
    if args.labels_per_update < 0:
        parser.error("--labels-per-update must be >= 0")
    if args.train_sampler == "cyclic_label_budget":
        if args.labels_per_update < 1:
            parser.error(
                "--train-sampler cyclic_label_budget requires "
                "--labels-per-update > 0"
            )
        if args.grad_accum_steps != 1:
            parser.error(
                "cyclic_label_budget uses variable microsteps; leave "
                "--grad-accum-steps at 1"
            )
    elif args.labels_per_update != 0:
        parser.error(
            "--labels-per-update is only valid with "
            "--train-sampler cyclic_label_budget"
        )
    if args.reader_protocol not in READER_PROTOCOLS:
        parser.error(
            f"--reader-protocol must be one of {READER_PROTOCOLS}"
        )
    if (
        not math.isfinite(args.identity_contrast_lambda)
        or args.identity_contrast_lambda < 0
    ):
        parser.error("--identity-contrast-lambda must be finite and >= 0")
    if not math.isfinite(args.identity_margin) or args.identity_margin < 0:
        parser.error("--identity-margin must be finite and >= 0")
    if args.num_swap_derangements < 1:
        parser.error("--num-swap-derangements must be positive")
    if args.steps < 0:
        parser.error("--steps must be >= 0")
    if args.save_every < 0:
        parser.error("--save-every must be >= 0")
    if args.init_checkpoint and args.resume_checkpoint:
        parser.error(
            "--init-checkpoint and --resume-checkpoint are mutually exclusive"
        )
    if not args.eval_only and args.steps < 1:
        parser.error("training requires --steps >= 1")
    if args.eval_only and not (
        args.init_checkpoint or args.resume_checkpoint
    ):
        parser.error(
            "--eval-only requires --init-checkpoint or --resume-checkpoint"
        )
    resolved_memory_mode = resolve_memory_mode(
        args.memory_mode, args.num_prefix_tokens
    )
    if resolved_memory_mode == "prefix":
        if args.num_prefix_tokens == 0:
            parser.error("--memory-mode prefix requires --P > 0")
        if args.read_mode in ("broadcast", "pooled_plus_prefix"):
            parser.error(
                "--memory-mode prefix requires a prefix read-mode "
                "(pool, standard, or prefix_only)"
            )
    elif resolved_memory_mode == "pooled_steer":
        if args.num_prefix_tokens != 0:
            parser.error("--memory-mode pooled_steer requires --P 0")
        if args.read_mode != "broadcast":
            parser.error(
                "--memory-mode pooled_steer requires --read-mode broadcast"
            )
        if args.prefix_write_layout != "global":
            parser.error(
                "--prefix-write-layout is only active for prefix or hybrid memory"
            )
    elif resolved_memory_mode == "hybrid":
        if args.num_prefix_tokens == 0:
            parser.error("--memory-mode hybrid requires --P > 0")
        if args.read_mode != "pooled_plus_prefix":
            parser.error(
                "--memory-mode hybrid requires --read-mode pooled_plus_prefix"
            )
        if args.history_pool != "attn":
            parser.error("--memory-mode hybrid requires --history-pool attn")
    else:
        if args.num_prefix_tokens != 0:
            parser.error("--memory-mode none requires --P 0")
        if args.read_mode in ("prefix_only", "broadcast", "pooled_plus_prefix"):
            parser.error(
                "--memory-mode none has no persistent memory; use read-mode pool or standard"
            )
        if args.identity_contrast_lambda > 0:
            parser.error(
                "--identity-contrast-lambda > 0 requires prefix or pooled_steer memory"
            )
        if args.prefix_write_layout != "global":
            parser.error(
                "--prefix-write-layout is only active for prefix or hybrid memory"
            )
    gate_init_valid = (
        0.0 < args.hybrid_prefix_gate_init <= 1.0
        if args.hybrid_prefix_gate_mode == "fixed"
        else 0.0 < args.hybrid_prefix_gate_init < 1.0
    )
    if not gate_init_valid:
        parser.error(
            "--hybrid-prefix-gate-init must be in (0,1] for fixed or (0,1) "
            "for a learned gate"
        )
    if (
        not math.isfinite(args.hybrid_pool_drop_prob)
        or not 0.0 <= args.hybrid_pool_drop_prob <= 1.0
    ):
        parser.error("--hybrid-pool-drop-prob must be finite and in [0,1]")
    if (
        resolved_memory_mode != "hybrid"
        and args.hybrid_pool_drop_prob != 0.0
    ):
        parser.error(
            "--hybrid-pool-drop-prob is only configurable with "
            "--memory-mode hybrid"
        )
    if (
        resolved_memory_mode != "hybrid"
        and (
            args.hybrid_prefix_gate_mode != "fixed"
            or args.hybrid_prefix_gate_init != 0.1
        )
    ):
        parser.error(
            "--hybrid-prefix-gate-* options are only configurable with "
            "--memory-mode hybrid"
        )
    args.resolved_memory_mode = resolved_memory_mode
    try:
        args.steer_layers = parse_layers(args.layers)
    except ValueError as error:
        parser.error(str(error))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_path.with_suffix(".pt")
    recovery_checkpoint_path = output_path.with_suffix(".resume.pt")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_root = args.data_root.expanduser().resolve()
    train_csv = (
        args.train_csv.expanduser().resolve()
        if args.train_csv
        else resolve_split_csv(data_root, args.train_split)
    )
    eval_csv = (
        args.eval_csv.expanduser().resolve()
        if args.eval_csv
        else resolve_split_csv(data_root, args.eval_split)
    )
    manifest_path = args.clean_manifest.expanduser().resolve()
    if not manifest_path.is_file():
        parser.error(f"--clean-manifest does not exist: {manifest_path}")
    clean_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    excluded_ids = parse_persona_ids(
        args.exclude_persona_ids, args.exclude_persona_id
    )
    excluded_ids.update(
        str(value)
        for value in clean_manifest.get(
            "exclude_persona_ids_from_train_and_val", []
        )
    )
    excluded_sample_ids = parse_sample_ids(
        args.exclude_sample_ids, args.exclude_sample_id
    )
    excluded_sample_ids.update(
        str(value)
        for value in clean_manifest.get("exclude_sample_ids_all_windows", [])
    )
    paper_dev = clean_manifest.get("paper_dev", {})
    holdout_size = (
        args.persona_holdout_size
        if args.persona_holdout_size is not None
        else int(paper_dev.get("num_personas", 80))
    )
    holdout_salt = (
        args.persona_holdout_salt
        or str(paper_dev.get("salt", DEFAULT_HOLDOUT_SALT))
    )
    train_pool = _load_dataset(
        csv_path=train_csv,
        split=args.train_split,
        data_root=data_root,
        window=args.window,
        option_seed=args.train_option_seed,
        excluded_ids=excluded_ids,
        excluded_sample_ids=excluded_sample_ids,
        overlap_policy=args.content_overlap_policy,
        reader_protocol=args.reader_protocol,
    )
    eval_pool = _load_dataset(
        csv_path=eval_csv,
        split=args.eval_split,
        data_root=data_root,
        window=args.window,
        option_seed=args.eval_option_seed,
        excluded_ids=excluded_ids,
        excluded_sample_ids=excluded_sample_ids,
        overlap_policy=args.content_overlap_policy,
        reader_protocol=args.reader_protocol,
    )
    included_ids = parse_persona_ids(args.include_persona_ids, ())
    if included_ids:
        train_pool = include_persona_episodes(
            train_pool, included_ids, label="train"
        )
        eval_pool = include_persona_episodes(
            eval_pool, included_ids, label="eval"
        )
    heldout_personas: set[str] = set()
    if args.eval_split == "val" and (
        holdout_size > 0 or args.persona_holdout_fraction > 0
    ):
        train_pool, eval_pool, heldout_personas = deterministic_persona_holdout(
            train_pool,
            eval_pool,
            size=holdout_size,
            fraction=args.persona_holdout_fraction,
            salt=holdout_salt,
            seed=args.persona_holdout_seed,
            dev_source=args.dev_source,
        )
    train_episodes = limit_episodes(
        train_pool,
        max_personas=args.max_personas,
        max_queries=args.max_queries,
        selection_seed=args.selection_seed,
        shuffle_personas=True,
    )
    eval_episodes = limit_episodes(
        eval_pool,
        max_personas=args.max_personas,
        max_queries=args.max_queries,
        selection_seed=args.selection_seed + 1,
        shuffle_personas=True,
    )
    train_personas = {episode.persona_id for episode in train_episodes}
    eval_personas = {episode.persona_id for episode in eval_episodes}
    persona_overlap = sorted(train_personas & eval_personas)
    if args.identity_contrast_lambda > 0 and len(train_personas) < 2:
        parser.error(
            "--identity-contrast-lambda > 0 requires at least two selected "
            "training personas"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    letter_token_ids = resolve_letter_token_ids(
        tokenizer, reader_protocol=args.reader_protocol
    )
    train_histories = _encode_history_cache(
        tokenizer,
        train_episodes,
        max_history_tokens=args.max_history_tokens,
        truncation=args.history_truncation,
    )
    eval_histories = _encode_history_cache(
        tokenizer,
        eval_episodes,
        max_history_tokens=args.max_history_tokens,
        truncation=args.history_truncation,
    )

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
    memory_mode = args.resolved_memory_mode
    prefix_enabled = memory_mode in ("prefix", "hybrid")
    hybrid_enabled = memory_mode == "hybrid"
    memory_enabled = memory_mode != "none"
    config = build_prefix_config(args)
    replaced = attach_prefix_steer(model, config)
    freeze_backbone_keep_steer(model)
    model.config.use_cache = False
    resume_checkpoint: Mapping[str, Any] | None = None
    if args.init_checkpoint:
        _load_checkpoint(model, args.init_checkpoint, config=config)
    elif args.resume_checkpoint:
        resume_checkpoint = _load_checkpoint(
            model, args.resume_checkpoint, config=config
        )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    write_query_params = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (
            name.endswith(".prefix")
            or name.endswith(".history_pool_query")
            or name.endswith(".hybrid_prefix_gate_logit")
        )
    ]
    write_query_param_ids = {id(parameter) for parameter in write_query_params}
    other_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in write_query_param_ids
    ]
    parameter_groups: list[dict[str, Any]] = [
        {
            "params": other_params,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
    ]
    if write_query_params:
        parameter_groups.append(
            {
                "params": write_query_params,
                "lr": args.prefix_lr,
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups)
    planned_label_budget_schedule = (
        simulate_cyclic_label_budget(
            train_episodes,
            seed=20_000 + args.seed,
            max_queries_per_write=args.queries_per_write,
            labels_per_update=args.labels_per_update,
            optimizer_updates=args.steps,
        )
        if (
            not args.eval_only
            and args.train_sampler == "cyclic_label_budget"
        )
        else None
    )
    if (
        planned_label_budget_schedule is not None
        and planned_label_budget_schedule["seen_persona_count"]
        != len(train_episodes)
    ):
        parser.error(
            "cyclic_label_budget schedule does not cover every selected train "
            f"persona: {planned_label_budget_schedule['seen_persona_count']}/"
            f"{len(train_episodes)}; increase --steps/--labels-per-update or "
            "decrease --queries-per-write"
        )

    print(
        "[personamem] "
        f"memory={memory_mode} read="
        f"{'broadcast-' + args.history_pool if memory_mode == 'pooled_steer' else args.read_mode} "
        f"P={args.num_prefix_tokens} head_dim={args.head_dim} "
        f"write_layout={args.prefix_write_layout}"
        f"+{args.prefix_write_overlap_tokens} "
        f"hybrid_gate="
        f"{args.hybrid_prefix_gate_mode}:{args.hybrid_prefix_gate_init:g} "
        f"hybrid_pool_drop={args.hybrid_pool_drop_prob:g} "
        f"layers={len(replaced)} trainable={trainable:,} "
        f"train={len(train_episodes)}p/{sum(len(e.questions) for e in train_episodes)}q "
        f"eval={len(eval_episodes)}p/{sum(len(e.questions) for e in eval_episodes)}q "
        f"history<={args.max_history_tokens}({args.history_truncation}) "
        f"K={args.queries_per_write} excluded={sorted(excluded_ids)} "
        f"excluded_samples={len(excluded_sample_ids)} "
        f"holdout={len(heldout_personas)}p "
        f"reader={args.reader_protocol} "
        f"sampler={args.train_sampler} "
        f"K<={args.queries_per_write} "
        f"batching="
        f"{'label_budget=' + str(args.labels_per_update) if args.train_sampler == 'cyclic_label_budget' else 'accum=' + str(args.grad_accum_steps)} "
        f"planned_labels="
        f"{args.steps * (args.labels_per_update if args.train_sampler == 'cyclic_label_budget' else args.queries_per_write * args.grad_accum_steps)} "
        f"task_loss={args.task_loss} "
        f"identity_lambda={args.identity_contrast_lambda:g} "
        f"identity_margin={args.identity_margin:g} "
        f"donor_seed={args.identity_donor_seed}",
        flush=True,
    )
    if planned_label_budget_schedule is not None:
        print(
            "[personamem] cyclic-label-budget preflight "
            f"personas={planned_label_budget_schedule['seen_persona_count']}/"
            f"{len(train_episodes)} samples="
            f"{planned_label_budget_schedule['seen_sample_count']}/"
            f"{sum(len(episode.questions) for episode in train_episodes)} "
            f"micro={planned_label_budget_schedule['micro_steps']} "
            f"labels={planned_label_budget_schedule['label_exposures']} "
            f"persona_microsteps="
            f"{planned_label_budget_schedule['persona_microsteps_min']}-"
            f"{planned_label_budget_schedule['persona_microsteps_max']}",
            flush=True,
        )
    if persona_overlap:
        print(
            "[personamem] NOTE: selected train/eval personas overlap "
            f"({len(persona_overlap)} IDs; first={persona_overlap[:8]}). "
            "This is expected for official train/val, but use benchmark for held-out-persona "
            "reporting.",
            flush=True,
        )

    train_data_fingerprint = _training_data_fingerprint(train_episodes)
    history: list[dict[str, Any]] = []
    start_time = time.time()
    train_rng = random.Random(10_000 + args.seed)
    hybrid_pool_dropout = HybridPoolDropoutSchedule(
        args.hybrid_pool_drop_prob,
        seed=30_000 + args.seed,
    )
    interval_train_metrics: list[dict[str, float | str | int | None]] = []
    cyclic_sampler = (
        CyclicChunkSampler(train_episodes, seed=20_000 + args.seed)
        if args.train_sampler in {
            "cyclic_chunks",
            "cyclic_label_budget",
        }
        else None
    )
    optimizer_updates = 0
    micro_steps = 0
    actual_label_exposures = 0
    seen_train_personas: set[str] = set()
    persona_microstep_exposures: Counter[str] = Counter()
    persona_query_exposures: Counter[str] = Counter()
    sample_query_exposures: Counter[str] = Counter()
    if resume_checkpoint is not None:
        resume_state = _validate_resume_configuration(
            resume_checkpoint,
            args,
            train_data_fingerprint=train_data_fingerprint,
        )
        optimizer_state = resume_state.get("optimizer")
        if not isinstance(optimizer_state, Mapping):
            raise ValueError("resume checkpoint has no optimizer state")
        optimizer.load_state_dict(dict(optimizer_state))
        train_rng.setstate(resume_state["train_rng_state"])
        dropout_state = resume_state.get("hybrid_pool_dropout")
        if dropout_state is None:
            if args.hybrid_pool_drop_prob != 0.0:
                raise ValueError(
                    "resume checkpoint has no hybrid pool-drop RNG state"
                )
        elif not isinstance(dropout_state, Mapping):
            raise ValueError("invalid hybrid pool-drop RNG checkpoint state")
        else:
            hybrid_pool_dropout.load_state_dict(dropout_state)
        checkpoint_sampler_state = resume_state.get("cyclic_sampler")
        if cyclic_sampler is None:
            if checkpoint_sampler_state is not None:
                raise ValueError(
                    "resume checkpoint has cyclic state but current sampler is random"
                )
        else:
            if not isinstance(checkpoint_sampler_state, Mapping):
                raise ValueError(
                    "cyclic resume requires a cyclic sampler checkpoint state"
                )
            cyclic_sampler.load_state_dict(checkpoint_sampler_state)
        optimizer_updates = int(
            resume_state["completed_optimizer_updates"]
        )
        micro_steps = int(resume_state["micro_steps"])
        actual_label_exposures = int(
            resume_state["actual_label_exposures"]
        )
        if dropout_state is None:
            # Backward-compatible p=0 resume: reconstruct the otherwise inert
            # dedicated stream at the exact microstep boundary.
            for _ in range(micro_steps):
                hybrid_pool_dropout.next()
        elif hybrid_pool_dropout.draw_count != micro_steps:
            raise ValueError(
                "resume checkpoint hybrid pool-drop draw count is inconsistent "
                "with micro_steps"
            )
        seen_train_personas = {
            str(value)
            for value in resume_state.get("seen_train_personas", [])
        }
        persona_microstep_exposures = Counter(
            {
                str(key): int(value)
                for key, value in dict(
                    resume_state.get("persona_microstep_exposures", {})
                ).items()
            }
        )
        persona_query_exposures = Counter(
            {
                str(key): int(value)
                for key, value in dict(
                    resume_state.get("persona_query_exposures", {})
                ).items()
            }
        )
        sample_query_exposures = Counter(
            {
                str(key): int(value)
                for key, value in dict(
                    resume_state.get("sample_query_exposures", {})
                ).items()
            }
        )
        interval_train_metrics = [
            dict(value)
            for value in resume_state.get("interval_train_metrics", [])
        ]
        history = [
            dict(value) for value in resume_state.get("history", [])
        ]
        if args.train_sampler == "cyclic_label_budget":
            if actual_label_exposures != (
                optimizer_updates * args.labels_per_update
            ):
                raise ValueError(
                    "resume checkpoint label-budget/update counts are inconsistent"
                )
        elif micro_steps != optimizer_updates * args.grad_accum_steps:
            raise ValueError(
                "resume checkpoint microstep/update counts are inconsistent"
            )
        if micro_steps != sum(persona_microstep_exposures.values()):
            raise ValueError(
                "resume checkpoint persona microstep counters are inconsistent"
            )
        if actual_label_exposures != sum(
            persona_query_exposures.values()
        ) or actual_label_exposures != sum(
            sample_query_exposures.values()
        ):
            raise ValueError(
                "resume checkpoint label exposure counters are inconsistent"
            )
        if seen_train_personas != set(persona_microstep_exposures):
            raise ValueError(
                "resume checkpoint seen-persona counters are inconsistent"
            )
        if not args.eval_only:
            torch.set_rng_state(resume_state["torch_rng_state"])
            saved_cuda_states = list(
                resume_state.get("cuda_rng_states", [])
            )
            if torch.cuda.is_available():
                if len(saved_cuda_states) != torch.cuda.device_count():
                    raise ValueError(
                        "resume checkpoint CUDA RNG device count does not match"
                    )
                torch.cuda.set_rng_state_all(saved_cuda_states)
            elif saved_cuda_states:
                raise ValueError(
                    "resume checkpoint was made with CUDA but CUDA is unavailable"
                )
        print(
            "[personamem] resumed exact training state at "
            f"update={optimizer_updates} micro={micro_steps} "
            f"labels={actual_label_exposures}",
            flush=True,
        )

    def current_training_state() -> dict[str, Any]:
        return _make_training_state(
            optimizer=optimizer,
            train_data_fingerprint=train_data_fingerprint,
            optimizer_updates=optimizer_updates,
            micro_steps=micro_steps,
            actual_label_exposures=actual_label_exposures,
            train_rng=train_rng,
            hybrid_pool_dropout=hybrid_pool_dropout,
            cyclic_sampler=cyclic_sampler,
            seen_train_personas=seen_train_personas,
            persona_microstep_exposures=persona_microstep_exposures,
            persona_query_exposures=persona_query_exposures,
            sample_query_exposures=sample_query_exposures,
            interval_train_metrics=interval_train_metrics,
            history=history,
        )

    last_evaluation_result: dict[str, Any] | None = None
    last_evaluation_update: int | None = None
    if not args.eval_only:
        for step in range(optimizer_updates + 1, args.steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            update_micro_metrics: list[
                dict[str, float | str | int | None]
            ] = []
            update_label_count = 0
            accumulation_index = 0
            label_budget_mode = (
                args.train_sampler == "cyclic_label_budget"
            )
            while (
                update_label_count < args.labels_per_update
                if label_budget_mode
                else accumulation_index < args.grad_accum_steps
            ):
                requested_queries = (
                    min(
                        args.queries_per_write,
                        args.labels_per_update - update_label_count,
                    )
                    if label_budget_mode
                    else args.queries_per_write
                )
                accumulation_index += 1
                micro_steps += 1
                if cyclic_sampler is not None:
                    episode, selected_questions = cyclic_sampler.next_chunk(
                        requested_queries,
                        cross_question_cycle=not label_budget_mode,
                    )
                else:
                    episode = train_rng.choice(train_episodes)
                    selected_questions = None
                donor = (
                    select_identity_donor(
                        episode.persona_id,
                        train_episodes,
                        seed=args.identity_donor_seed,
                        step=micro_steps,
                    )
                    if args.identity_contrast_lambda > 0
                    else None
                )
                # One dedicated draw covers the complete correct+donor contrast
                # cycle inside train_step.  Reset immediately afterward so
                # validation and every non-training caller always keep the pool.
                pool_dropped = hybrid_pool_dropout.next()
                set_hybrid_pool_off(
                    model, hybrid_enabled and pool_dropped
                )
                try:
                    train_result = train_step(
                        model,
                        tokenizer,
                        episode,
                        train_histories[episode.persona_id],
                        letter_token_ids,
                        device=args.device,
                        queries_per_write=requested_queries,
                        prefix_enabled=memory_enabled,
                        rng=train_rng,
                        option_shuffle_seed=args.train_option_seed,
                        option_shuffle_round=micro_steps,
                        reader_protocol=args.reader_protocol,
                        task_loss=args.task_loss,
                        identity_contrast_lambda=args.identity_contrast_lambda,
                        identity_margin=args.identity_margin,
                        donor_history_ids=(
                            train_histories[donor.persona_id]
                            if donor is not None
                            else None
                        ),
                        donor_persona_id=(
                            donor.persona_id if donor is not None else None
                        ),
                        selected_questions=selected_questions,
                    )
                finally:
                    set_hybrid_pool_off(model, False)
                train_metrics = train_result.scalar_metrics()
                train_metrics["hybrid_pool_dropped"] = int(
                    hybrid_enabled and pool_dropped
                )
                query_count = train_result.query_count
                seen_train_personas.add(episode.persona_id)
                persona_microstep_exposures[episode.persona_id] += 1
                persona_query_exposures[episode.persona_id] += query_count
                sample_query_exposures.update(train_result.sample_ids)
                actual_label_exposures += query_count
                if label_budget_mode:
                    if query_count > (
                        args.labels_per_update - update_label_count
                    ):
                        raise AssertionError(
                            "label-budget microstep exceeded remaining labels"
                        )
                    update_label_count += query_count
                    loss_scale = query_count / float(
                        args.labels_per_update
                    )
                else:
                    update_label_count += query_count
                    loss_scale = 1.0 / float(args.grad_accum_steps)
                (train_result.loss * loss_scale).backward()
                del train_result
                clear_frozen_memory(model)
                update_micro_metrics.append(train_metrics)
                interval_train_metrics.append(train_metrics)
            if (
                label_budget_mode
                and update_label_count != args.labels_per_update
            ):
                raise AssertionError(
                    "label-budget update did not reach its exact target"
                )
            update_mean = mean_train_diagnostics(
                update_micro_metrics,
                weight_by_query_count=label_budget_mode,
            )
            train_metrics = dict(update_micro_metrics[-1])
            for key in TRAIN_DIAGNOSTIC_KEYS:
                train_metrics[key] = update_mean[key]
            train_metrics["query_count"] = sum(
                int(row["query_count"]) for row in update_micro_metrics
            )
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    args.grad_clip,
                )
            optimizer.step()
            optimizer_updates += 1
            clear_frozen_memory(model)

            should_eval = (
                step == args.steps
                or (args.eval_every > 0 and step % args.eval_every == 0)
            )
            if should_eval:
                result = evaluate(
                    model,
                    tokenizer,
                    eval_episodes,
                    eval_histories,
                    letter_token_ids,
                    device=args.device,
                    query_batch_size=args.queries_per_write,
                    prefix_enabled=memory_enabled,
                    hybrid_prefix_ablation=hybrid_enabled,
                    num_swap_derangements=args.num_swap_derangements,
                    swap_seed=args.swap_seed,
                    reader_protocol=args.reader_protocol,
                )
                last_evaluation_result = result
                last_evaluation_update = optimizer_updates
                row = {
                    "step": step,
                    **train_metrics,
                    "optimizer_update": optimizer_updates,
                    "micro_steps": micro_steps,
                    "actual_label_exposures": actual_label_exposures,
                    "grad_accum_steps": args.grad_accum_steps,
                    "labels_per_update": (
                        args.labels_per_update if label_budget_mode else None
                    ),
                    "actual_microsteps_this_update": len(
                        update_micro_metrics
                    ),
                    "hybrid_pool_drop_count": sum(
                        int(row["hybrid_pool_dropped"])
                        for row in update_micro_metrics
                    ),
                    "task_loss": args.task_loss,
                    "train_interval_mean": mean_train_diagnostics(
                        interval_train_metrics,
                        weight_by_query_count=label_budget_mode,
                    ),
                    "accuracy": result["accuracy"],
                    "persona_macro_accuracy": result["persona_macro_accuracy"],
                    "paired": result["paired"],
                    "swap_derangements": result["swap_derangements"],
                    "hybrid_same_weights": result["hybrid_same_weights"],
                }
                history.append(row)
                interval_train_metrics.clear()
                accuracy = result["accuracy"]
                paired = result["paired"]
                swap_text = (
                    f"{accuracy['swap']:.3f}"
                    if accuracy["swap"] is not None
                    else "n/a"
                )
                delta_swap_text = (
                    f"{paired['correct_minus_swap']:+.3f}"
                    if paired["correct_minus_swap"] is not None
                    else "n/a"
                )
                prefix_off_text = (
                    f"{accuracy['prefix_off']:.3f}"
                    if accuracy["prefix_off"] is not None
                    else "n/a"
                )
                prefix_gain_text = (
                    f"{paired['correct_full_minus_prefix_off']:+.3f}"
                    if paired["correct_full_minus_prefix_off"] is not None
                    else "n/a"
                )
                contrast_text = (
                    f"{train_metrics['identity_contrast_loss']:.4f}"
                    if train_metrics["identity_contrast_loss"] is not None
                    else "off"
                )
                gap_text = (
                    f"{train_metrics['gold_log_probability_gap']:+.4f}"
                    if train_metrics["gold_log_probability_gap"] is not None
                    else "n/a"
                )
                probability_gap_text = (
                    f"{train_metrics['gold_probability_gap']:+.4f}"
                    if train_metrics["gold_probability_gap"] is not None
                    else "n/a"
                )
                print(
                    f"[personamem] step={step} "
                    f"micro={micro_steps} labels={actual_label_exposures} "
                    f"loss={float(train_metrics['loss']):.4f} "
                    f"ce={float(train_metrics['ce_loss']):.4f} "
                    f"idc={contrast_text} logpgap={gap_text} "
                    f"pgap={probability_gap_text} "
                    f"base={accuracy['base']:.3f} window={accuracy['window']:.3f} "
                    f"swap={swap_text} correct={accuracy['correct']:.3f} "
                    f"prefix_off={prefix_off_text} dPrefix={prefix_gain_text} "
                    f"dW={paired['correct_minus_window']:+.3f} "
                    f"dS={delta_swap_text} "
                    f"elapsed={(time.time() - start_time) / 60:.1f}m",
                    flush=True,
                )
            if args.save_every > 0 and step % args.save_every == 0:
                _save_checkpoint(
                    model,
                    recovery_checkpoint_path,
                    config=config,
                    args=args,
                    metadata={
                        "checkpoint_kind": "periodic_training_resume",
                        "completed_optimizer_updates": optimizer_updates,
                        "actual_label_exposures": actual_label_exposures,
                        "hybrid_pool_drop_draws": (
                            hybrid_pool_dropout.draw_count
                        ),
                        "hybrid_pool_drop_count": (
                            hybrid_pool_dropout.drop_count
                        ),
                    },
                    training_state=current_training_state(),
                )
                print(
                    "[personamem] saved recovery checkpoint "
                    f"update={optimizer_updates} -> "
                    f"{recovery_checkpoint_path}",
                    flush=True,
                )

    final = (
        last_evaluation_result
        if (
            last_evaluation_result is not None
            and last_evaluation_update == optimizer_updates
        )
        else evaluate(
            model,
            tokenizer,
            eval_episodes,
            eval_histories,
            letter_token_ids,
            device=args.device,
            query_batch_size=args.queries_per_write,
            prefix_enabled=memory_enabled,
            hybrid_prefix_ablation=hybrid_enabled,
            num_swap_derangements=args.num_swap_derangements,
            swap_seed=args.swap_seed,
            reader_protocol=args.reader_protocol,
        )
    )
    payload = {
        "args": {
            **vars(args),
            "data_root": str(args.data_root),
            "clean_manifest": str(args.clean_manifest),
            "train_csv": str(args.train_csv) if args.train_csv else None,
            "eval_csv": str(args.eval_csv) if args.eval_csv else None,
            "output": str(args.output),
            "steer_layers": list(args.steer_layers),
        },
        "config": asdict(config),
        "data": {
            "train_persona_ids": sorted(train_personas),
            "eval_persona_ids": sorted(eval_personas),
            "train_eval_persona_overlap": persona_overlap,
            "deterministic_holdout_persona_ids": sorted(heldout_personas),
            "holdout_hash": {
                "algorithm": "sha256",
                "salt": holdout_salt,
                "seed": args.persona_holdout_seed,
                "size": holdout_size,
                "dev_source": args.dev_source,
            },
            "excluded_persona_ids": sorted(excluded_ids),
            "included_persona_ids": sorted(included_ids),
            "excluded_sample_ids": sorted(excluded_sample_ids),
            "clean_manifest": clean_manifest,
            "eval_option_shuffle": {
                "protocol": (
                    "random.Random(42 + original_zero_based_csv_row_index)"
                    if args.reader_protocol == "official_qwen"
                    else "stable_sha256_seed_round_sample_id"
                ),
                "seed": (
                    42
                    if args.reader_protocol == "official_qwen"
                    else args.eval_option_seed
                ),
                "round": (
                    None if args.reader_protocol == "official_qwen" else 0
                ),
            },
            "history_token_lengths": {
                "train": {
                    key: len(value) for key, value in train_histories.items()
                },
                "eval": {
                    key: len(value) for key, value in eval_histories.items()
                },
            },
        },
        "trainable_parameters": trainable,
        "training_exposure": {
            "steps_semantics": "optimizer_updates",
            "resumed_from": (
                str(args.resume_checkpoint) if args.resume_checkpoint else None
            ),
            "periodic_checkpoint_every_updates": args.save_every,
            "train_data_fingerprint_sha256": train_data_fingerprint,
            "requested_optimizer_updates": (
                0 if args.eval_only else args.steps
            ),
            "actual_optimizer_updates": optimizer_updates,
            "grad_accum_steps": args.grad_accum_steps,
            "labels_per_update": (
                args.labels_per_update
                if args.train_sampler == "cyclic_label_budget"
                else None
            ),
            "actual_micro_steps": micro_steps,
            "hybrid_pool_drop_probability": args.hybrid_pool_drop_prob,
            "hybrid_pool_drop_draws": hybrid_pool_dropout.draw_count,
            "hybrid_pool_drop_count": hybrid_pool_dropout.drop_count,
            "max_queries_per_write": args.queries_per_write,
            "planned_label_exposures": (
                0
                if args.eval_only
                else args.steps
                * (
                    args.labels_per_update
                    if args.train_sampler == "cyclic_label_budget"
                    else (
                        args.grad_accum_steps
                        * args.queries_per_write
                    )
                )
            ),
            "actual_label_exposures": actual_label_exposures,
            "train_sampler": args.train_sampler,
            "planned_label_budget_schedule": planned_label_budget_schedule,
            "seen_persona_count": len(seen_train_personas),
            "seen_persona_ids": sorted(seen_train_personas),
            "persona_microstep_exposures": dict(
                sorted(persona_microstep_exposures.items())
            ),
            "persona_query_exposures": dict(
                sorted(persona_query_exposures.items())
            ),
            "sample_query_exposures": dict(
                sorted(sample_query_exposures.items())
            ),
            "cyclic_sampler": (
                {
                    "persona_cycles_started": (
                        cyclic_sampler.persona_cycles_started
                    ),
                    "question_cycles_started_by_persona": dict(
                        sorted(
                            cyclic_sampler.question_cycles_started.items()
                        )
                    ),
                    "within_persona_policy": (
                        (
                            "variable chunk ends at the current shuffled "
                            "question-cycle boundary; query-count-weighted "
                            "microsteps fill an exact global label budget"
                        )
                        if args.train_sampler == "cyclic_label_budget"
                        else (
                            "consume full shuffled permutation before repeat; "
                            "a fixed-K chunk may cross a cycle boundary"
                        )
                    ),
                }
                if cyclic_sampler is not None
                else None
            ),
        },
        "letter_token_ids": dict(
            zip(
                "abcd" if args.reader_protocol == "official_qwen" else "ABCD",
                letter_token_ids,
            )
        ),
        "train_objective": {
            "primary": args.task_loss,
            "primary_definitions": {
                "full_vocab": "next_token_ce_over_full_vocabulary",
                "four_choice": "cross_entropy_over_four_protocol_label_logits_only",
            },
            "identity_contrast": (
                "mean_softplus(identity_margin - "
                "(gold_correct_memory_AD_log_probability - "
                "gold_wrong_memory_AD_log_probability))"
            ),
            "identity_contrast_lambda": args.identity_contrast_lambda,
            "identity_margin": args.identity_margin,
            "identity_donor_seed": args.identity_donor_seed,
            "reader_protocol": args.reader_protocol,
            "reader_prompt": (
                "official_personamem_qwen_verl_system_mcq_think_verbatim"
                if args.reader_protocol == "official_qwen"
                else "legacy_pilot_mcq"
            ),
            "supervision_mode": (
                "auxiliary_next_token_after_fixed_<think></think>_and_boxed_prefix"
                if args.reader_protocol == "official_qwen"
                else "direct_next_token_letter"
            ),
            "is_full_generative_sft": False,
            "fixed_assistant_prefix": (
                OFFICIAL_BOXED_CLASSIFICATION_PREFIX
                if args.reader_protocol == "official_qwen"
                else None
            ),
            "forced_choice_tokens": (
                "abcd" if args.reader_protocol == "official_qwen" else "ABCD"
            ),
            "writer_inputs": "history_only_no_query_option_answer_or_label",
        },
        "history": history,
        "final": final,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _save_checkpoint(
        model,
        checkpoint_path,
        config=config,
        args=args,
        metadata={
            "training_exposure": payload["training_exposure"],
            "train_objective": payload["train_objective"],
        },
        training_state=(
            None if args.eval_only else current_training_state()
        ),
    )
    accuracy = final["accuracy"]
    swap_text = (
        f"{accuracy['swap']:.3f}"
        if accuracy["swap"] is not None
        else "n/a"
    )
    prefix_off_text = (
        f"{accuracy['prefix_off']:.3f}"
        if accuracy["prefix_off"] is not None
        else "n/a"
    )
    prefix_gain = final["paired"]["correct_full_minus_prefix_off"]
    prefix_gain_text = (
        f"{prefix_gain:+.3f}" if prefix_gain is not None else "n/a"
    )
    print(
        f"[personamem] DONE base={accuracy['base']:.3f} "
        f"window={accuracy['window']:.3f} swap={swap_text} "
        f"correct_full={accuracy['correct_full']:.3f} "
        f"prefix_off={prefix_off_text} dPrefix={prefix_gain_text} "
        f"-> {output_path} / {checkpoint_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
