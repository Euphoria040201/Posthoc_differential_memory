#!/usr/bin/env python3
"""Leakage-safe PersonaMem-v2 text loader and audit CLI.

The central API deliberately separates the two model inputs:

* ``PersonaEpisode.writer`` is built only from the selected official chat-history
  JSON.  Its schema contains messages and nothing from the benchmark CSV.
* ``MCQExample.reader`` contains the future query and shuffled answer options.
  The label and subgroup tags stay outside ``ReaderInput``.

In particular, CSV annotations such as ``preference``,
``related_conversation_snippet``, ``correct_answer``, and ``prev_pref`` are never
accepted by the writer-building function.  A related conversation can still occur
naturally inside the official history; that is the intended implicit signal, not
an annotation appended by this loader.

The module uses only the Python standard library so it can be imported by training
code without pandas/datasets.  Run it directly to validate a local download:

    python scripts/personamem_v2_data.py \
      --data-root /path/to/PersonaMem-v2 --split train --window 32k
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence


Split = Literal["train", "val", "benchmark"]
Window = Literal["32k", "128k"]
ContentOverlapPolicy = Literal["error", "warn", "off"]

SPLIT_FILENAMES: dict[Split, str] = {
    "train": "train.csv",
    "val": "val.csv",
    "benchmark": "benchmark.csv",
}

WINDOW_COLUMNS: dict[Window, dict[str, str]] = {
    "32k": {
        "link": "chat_history_32k_link",
        "tokens": "total_tokens_in_chat_history_32k",
        "distance": "distance_from_related_snippet_to_query_32k",
        "relevant_tokens": "num_persona_relevant_tokens_32k",
        "irrelevant_tokens": "num_persona_irrelevant_tokens_32k",
    },
    "128k": {
        "link": "chat_history_128k_link",
        "tokens": "total_tokens_in_chat_history_128k",
        "distance": "distance_from_related_snippet_to_query_128k",
        "relevant_tokens": "num_persona_relevant_tokens_128k",
        "irrelevant_tokens": "num_persona_irrelevant_tokens_128k",
    },
}

REQUIRED_BASE_COLUMNS = frozenset(
    {
        "persona_id",
        "user_query",
        "correct_answer",
        "incorrect_answers",
    }
)

# These names may exist in the CSV for labels/auditing, but they are forbidden in
# the writer payload and forbidden as extra keys on individual history messages.
WRITER_FORBIDDEN_FIELDS = frozenset(
    {
        "answer",
        "answers",
        "correct_answer",
        "expanded_persona",
        "ground_truth",
        "groundtruth_preference",
        "incorrect_answers",
        "label",
        "preference",
        "prev_pref",
        "query",
        "raw_persona_file",
        "related_conversation_snippet",
        "related_snippet",
        "short_persona",
        "target",
        "user_query",
    }
)

_WHITESPACE_RE = re.compile(r"\s+")
_VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})


class PersonaMemDataError(ValueError):
    """Base class for malformed PersonaMem input."""


class SchemaError(PersonaMemDataError):
    """CSV or JSON does not match the expected schema."""


class LeakageError(PersonaMemDataError):
    """A forbidden field or target-content overlap reached writer history."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class WriterInput:
    """The complete payload that may be tokenized for the memory writer."""

    messages: tuple[ChatMessage, ...]

    def to_messages(self) -> list[dict[str, str]]:
        """Return an OpenAI/Qwen-style message list with only role and content."""
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]

    def to_text(self) -> str:
        """Return an explicit-role text form for tokenizers without chat templates."""
        return "\n".join(f"{msg.role}: {msg.content}" for msg in self.messages)

    def assert_safe_schema(self) -> None:
        writer_fields = {item.name for item in fields(self)}
        if writer_fields != {"messages"}:
            raise LeakageError(f"WriterInput schema unexpectedly changed: {writer_fields}")
        for index, message in enumerate(self.messages):
            message_fields = {item.name for item in fields(message)}
            if message_fields != {"role", "content"}:
                raise LeakageError(
                    f"Writer message {index} has unexpected schema: {message_fields}"
                )


@dataclass(frozen=True, slots=True)
class ReaderInput:
    """Query-time input; labels are intentionally not part of this object."""

    query: str
    options: tuple[str, ...]

    def to_prompt(self) -> str:
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        option_lines = "\n".join(
            f"({labels[index]}) {option}" for index, option in enumerate(self.options)
        )
        return (
            f"{self.query}\n\n"
            "Choose the best personalized response from the options below.\n"
            f"{option_lines}\n\n"
            "Answer with only the option letter."
        )


@dataclass(frozen=True, slots=True)
class AuditTags:
    """Non-writer metadata retained for benchmark subgroup reporting."""

    topic_query: str
    topic_preference: str
    conversation_scenario: str
    pref_type: str
    who: str
    updated: bool
    sensitive_info: bool
    total_tokens: int | None
    distance_to_related_snippet: int | None
    persona_relevant_tokens: int | None
    persona_irrelevant_tokens: int | None


@dataclass(frozen=True, slots=True)
class MCQExample:
    sample_id: str
    reader: ReaderInput
    correct_index: int
    tags: AuditTags
    # Zero-based index in the original CSV (header excluded).  PersonaMem-v2's
    # Qwen preprocessing seeds each option shuffle with ``42 + row_index``.
    # Keep this outside ReaderInput so it cannot enter either model input.
    source_row_index: int = -1

    @property
    def correct_letter(self) -> str:
        return chr(ord("A") + self.correct_index)


@dataclass(frozen=True, slots=True)
class PersonaEpisode:
    """One history write followed by one or more future personalized queries."""

    persona_id: str
    split: Split
    window: Window
    history_path: Path
    writer: WriterInput
    questions: tuple[MCQExample, ...]


@dataclass(frozen=True, slots=True)
class LoadedPersonaDataset:
    csv_path: Path
    split: Split
    window: Window
    shuffle_seed: int
    shuffle_round: int
    episodes: tuple[PersonaEpisode, ...]
    rows_seen: int
    rows_skipped_missing_history: int
    rows_skipped_invalid_mcq: int
    rows_skipped_excluded_persona: int
    excluded_persona_ids: tuple[str, ...]
    content_overlap_warnings: tuple[str, ...]

    def iter_questions(self) -> Iterator[tuple[PersonaEpisode, MCQExample]]:
        for episode in self.episodes:
            for question in episode.questions:
                yield episode, question

    @property
    def num_questions(self) -> int:
        return sum(len(episode.questions) for episode in self.episodes)


@dataclass(slots=True)
class _EpisodeBuilder:
    history_link: str
    history_path: Path
    writer: WriterInput
    normalised_history: str
    questions: list[MCQExample]


def _normalise_split(value: str) -> Split:
    value = value.strip().lower()
    if value == "validation":
        value = "val"
    if value not in SPLIT_FILENAMES:
        raise ValueError(f"split must be train, val, or benchmark; got {value!r}")
    return value  # type: ignore[return-value]


def _normalise_window(value: str) -> Window:
    value = value.strip().lower()
    if value not in WINDOW_COLUMNS:
        raise ValueError(f"window must be 32k or 128k; got {value!r}")
    return value  # type: ignore[return-value]


def _normalise_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip().casefold()


def _nonempty(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "none", "null"} else text


def _nonempty_preserve_whitespace(value: Any) -> str:
    """Validate like ``_nonempty`` while retaining official prompt bytes."""

    if value is None:
        return ""
    text = str(value)
    stripped = text.strip()
    return "" if stripped.casefold() in {"", "nan", "none", "null"} else text


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _nonempty(value).casefold()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"", "0", "false", "no", "n"}:
        return False
    raise SchemaError(f"Cannot parse boolean value {value!r}")


def _parse_optional_int(value: Any, field_name: str) -> int | None:
    text = _nonempty(value)
    if not text:
        return None
    try:
        # Some CSV writers render integral values as "123.0".
        number = float(text)
        if not number.is_integer():
            raise ValueError
        return int(number)
    except ValueError as exc:
        raise SchemaError(f"{field_name} must be an integer, got {value!r}") from exc


def _parse_string_list(
    value: Any,
    field_name: str,
    *,
    preserve_whitespace: bool = False,
) -> list[str]:
    if isinstance(value, (list, tuple)):
        parsed: Any = value
    else:
        text = _nonempty(value)
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError) as exc:
                raise SchemaError(
                    f"{field_name} is not a JSON/Python list: {text[:120]!r}"
                ) from exc
    if not isinstance(parsed, (list, tuple)):
        raise SchemaError(f"{field_name} must decode to a list, got {type(parsed).__name__}")
    cleaner = (
        _nonempty_preserve_whitespace
        if preserve_whitespace
        else _nonempty
    )
    result = [cleaner(item) for item in parsed]
    if any(not item for item in result):
        raise SchemaError(f"{field_name} contains an empty answer")
    return result


def _extract_query(
    value: Any, *, preserve_whitespace: bool = False
) -> str:
    """Handle both plain strings and official JSON-like role/content strings."""
    cleaner = (
        _nonempty_preserve_whitespace
        if preserve_whitespace
        else _nonempty
    )
    if isinstance(value, Mapping):
        query = cleaner(value.get("content"))
        if not query:
            raise SchemaError("user_query mapping has no non-empty content")
        return query
    text = _nonempty(value)
    if not text:
        raise SchemaError("user_query is empty")
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, Mapping) and "content" in parsed:
            query = cleaner(parsed["content"])
            if not query:
                raise SchemaError("user_query mapping has empty content")
            return query
    return cleaner(value) if preserve_whitespace else text


def _stable_sample_id(
    split: Split,
    persona_id: str,
    query: str,
    correct_answer: str,
    incorrect_answers: Sequence[str],
) -> str:
    raw = json.dumps(
        [split, persona_id, query, correct_answer, list(incorrect_answers)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def _shuffle_mcq(
    correct_answer: str,
    incorrect_answers: Sequence[str],
    *,
    sample_id: str,
    seed: int,
    shuffle_round: int,
    row_index: int,
    protocol: str,
) -> tuple[tuple[str, ...], int]:
    options = [correct_answer, *incorrect_answers]
    normalised = [_normalise_text(option) for option in options]
    if len(set(normalised)) != len(normalised):
        raise SchemaError(f"{sample_id}: answer options are not distinct")
    if len(options) != 4:
        raise SchemaError(
            f"{sample_id}: expected one correct plus three incorrect options, got {len(options)}"
        )
    permutation = list(range(len(options)))
    if protocol == "stable":
        stable_key = f"{seed}\0{shuffle_round}\0{sample_id}".encode("utf-8")
        stable_seed = int.from_bytes(
            hashlib.sha256(stable_key).digest()[:16], "big"
        )
        random.Random(stable_seed).shuffle(permutation)
    elif protocol == "official_qwen":
        # Exact PersonaMem-v2 Qwen/VERL data_preprocess_rft.py convention.
        # ``row_index`` is zero based and refers to the unfiltered source CSV.
        random.Random(42 + row_index).shuffle(permutation)
    else:
        raise ValueError(
            "option shuffle protocol must be stable or official_qwen"
        )
    shuffled = tuple(options[index] for index in permutation)
    return shuffled, permutation.index(0)


def _infer_data_root(csv_path: Path) -> Path:
    # Official layout: ROOT/benchmark/text/{train,val,benchmark}.csv
    parents = csv_path.resolve().parents
    if len(parents) >= 3 and parents[0].name == "text" and parents[1].name == "benchmark":
        return parents[2]
    return csv_path.resolve().parent


def resolve_history_path(
    history_link: str,
    *,
    csv_path: Path,
    data_root: Path | None,
) -> Path:
    """Resolve official relative links without silently guessing a different file."""
    link = _nonempty(history_link).replace("\\", "/")
    if not link:
        raise SchemaError("Selected chat-history link is empty")

    raw_path = Path(link).expanduser()
    root = (data_root or _infer_data_root(csv_path)).expanduser().resolve()
    candidates: list[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                root / raw_path,
                csv_path.resolve().parent / raw_path,
                Path.cwd() / raw_path,
            ]
        )
    # Some generated CSVs preserve a machine-specific prefix before "data/".
    # Re-root only that unambiguous suffix under the explicit dataset root.
    parts = raw_path.parts
    if "data" in parts:
        data_index = parts.index("data")
        candidates.append(root / Path(*parts[data_index:]))

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if str(resolved) not in seen:
            seen.add(str(resolved))
            unique_candidates.append(resolved)
        if resolved.is_file():
            return resolved
    shown = "\n  ".join(str(path) for path in unique_candidates)
    raise FileNotFoundError(f"History file {link!r} not found. Tried:\n  {shown}")


def _load_writer_input(history_path: Path) -> WriterInput:
    """Load only official chat messages; this function cannot receive a CSV row."""
    try:
        with history_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"Invalid JSON in {history_path}: {exc}") from exc

    if isinstance(payload, list):
        raw_messages = payload
    elif isinstance(payload, Mapping):
        if isinstance(payload.get("chat_history"), list):
            raw_messages = payload["chat_history"]
        elif isinstance(payload.get("messages"), list):
            raw_messages = payload["messages"]
        else:
            raise SchemaError(
                f"{history_path}: expected top-level list or dict containing chat_history/messages"
            )
    else:
        raise SchemaError(
            f"{history_path}: expected list/dict, got {type(payload).__name__}"
        )

    messages: list[ChatMessage] = []
    for index, raw_message in enumerate(raw_messages):
        if not isinstance(raw_message, Mapping):
            raise SchemaError(
                f"{history_path}: message {index} is {type(raw_message).__name__}, not a mapping"
            )
        keys = {str(key).casefold() for key in raw_message}
        forbidden = keys & WRITER_FORBIDDEN_FIELDS
        if forbidden:
            raise LeakageError(
                f"{history_path}: message {index} contains forbidden writer keys "
                f"{sorted(forbidden)}"
            )
        role = _nonempty(raw_message.get("role")).casefold()
        content = raw_message.get("content")
        if role not in _VALID_ROLES:
            raise SchemaError(
                f"{history_path}: message {index} has invalid role {role!r}"
            )
        # A small number of official text histories contain role-only turns.
        # Treat their absent/null content as the empty string; they carry no target
        # information, and preserving the role boundary is more faithful than inventing text.
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise SchemaError(
                f"{history_path}: text-only message {index} needs string content"
            )
        # Extra non-forbidden message metadata is intentionally discarded.
        messages.append(ChatMessage(role=role, content=content))

    if not messages:
        raise SchemaError(f"{history_path}: chat history is empty")
    writer = WriterInput(messages=tuple(messages))
    writer.assert_safe_schema()
    return writer


def _find_target_content_overlaps(
    normalised_history: str,
    *,
    sample_id: str,
    query: str,
    options: Sequence[str],
    min_chars: int,
) -> list[str]:
    """Detect a current query/complete answer accidentally included in history."""
    checks = [("user_query", query)]
    checks.extend((f"answer_option_{index}", option) for index, option in enumerate(options))
    hits: list[str] = []
    for field_name, value in checks:
        needle = _normalise_text(value)
        if len(needle) >= min_chars and needle in normalised_history:
            hits.append(f"{sample_id}:{field_name}")
    return hits


def _make_audit_tags(row: Mapping[str, Any], window: Window) -> AuditTags:
    columns = WINDOW_COLUMNS[window]
    return AuditTags(
        topic_query=_nonempty(row.get("topic_query")),
        topic_preference=_nonempty(row.get("topic_preference")),
        conversation_scenario=_nonempty(row.get("conversation_scenario")),
        pref_type=_nonempty(row.get("pref_type")),
        who=_nonempty(row.get("who")),
        updated=_parse_bool(row.get("updated")),
        sensitive_info=_parse_bool(row.get("sensitive_info")),
        total_tokens=_parse_optional_int(row.get(columns["tokens"]), columns["tokens"]),
        distance_to_related_snippet=_parse_optional_int(
            row.get(columns["distance"]), columns["distance"]
        ),
        persona_relevant_tokens=_parse_optional_int(
            row.get(columns["relevant_tokens"]), columns["relevant_tokens"]
        ),
        persona_irrelevant_tokens=_parse_optional_int(
            row.get(columns["irrelevant_tokens"]), columns["irrelevant_tokens"]
        ),
    )


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _read_csv_rows(csv_path: Path) -> tuple[tuple[str, ...], Iterator[dict[str, str]]]:
    """Return field names plus an iterator whose file lifetime is safely enclosed."""
    _raise_csv_field_limit()

    def iterator() -> Iterator[dict[str, str]]:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield dict(row)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
    return fieldnames, iterator()


def load_personamem_text(
    csv_path: str | Path,
    *,
    split: str,
    window: str = "32k",
    data_root: str | Path | None = None,
    shuffle_seed: int = 42,
    shuffle_round: int = 0,
    option_shuffle_protocol: str = "stable",
    max_rows: int | None = None,
    content_overlap_policy: ContentOverlapPolicy = "error",
    content_overlap_min_chars: int = 24,
    skip_missing_history: bool = False,
    exclude_persona_ids: Iterable[str] = (),
) -> LoadedPersonaDataset:
    """Load and group PersonaMem-v2 text MCQs by persona.

    Under the default ``stable`` option shuffle, ``shuffle_round`` can be set to
    the epoch number to obtain a different but reproducible order.  The
    ``official_qwen`` protocol instead exactly fixes each row with
    ``random.Random(42 + original_zero_based_csv_row_index)``.
    """
    split_name = _normalise_split(split)
    window_name = _normalise_window(window)
    if content_overlap_policy not in {"error", "warn", "off"}:
        raise ValueError("content_overlap_policy must be error, warn, or off")
    if option_shuffle_protocol not in {"stable", "official_qwen"}:
        raise ValueError(
            "option_shuffle_protocol must be stable or official_qwen"
        )
    if content_overlap_min_chars < 1:
        raise ValueError("content_overlap_min_chars must be positive")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive when provided")

    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    root_path = Path(data_root).expanduser().resolve() if data_root else None
    excluded = {_nonempty(value) for value in exclude_persona_ids}
    excluded.discard("")

    fieldnames, rows = _read_csv_rows(path)
    required = set(REQUIRED_BASE_COLUMNS) | {WINDOW_COLUMNS[window_name]["link"]}
    missing_columns = required - set(fieldnames)
    if missing_columns:
        raise SchemaError(f"{path}: missing required columns {sorted(missing_columns)}")

    builders: dict[str, _EpisodeBuilder] = {}
    sample_ids: set[str] = set()
    skipped_personas: set[str] = set()
    overlap_warnings: list[str] = []
    rows_seen = 0
    rows_skipped_missing_history = 0
    rows_skipped_invalid_mcq = 0
    rows_skipped_excluded_persona = 0

    for row_number, row in enumerate(rows, start=2):
        if max_rows is not None and rows_seen >= max_rows:
            break
        rows_seen += 1
        persona_id = _nonempty(row.get("persona_id"))
        if not persona_id:
            raise SchemaError(f"{path}:{row_number}: persona_id is empty")
        if persona_id in excluded:
            rows_skipped_excluded_persona += 1
            continue
        if persona_id in skipped_personas:
            rows_skipped_missing_history += 1
            continue

        history_link = _nonempty(row.get(WINDOW_COLUMNS[window_name]["link"]))
        builder = builders.get(persona_id)
        if builder is None:
            try:
                history_path = resolve_history_path(
                    history_link, csv_path=path, data_root=root_path
                )
                # Writer construction receives a path only, never the benchmark row.
                writer = _load_writer_input(history_path)
            except FileNotFoundError:
                if not skip_missing_history:
                    raise
                skipped_personas.add(persona_id)
                rows_skipped_missing_history += 1
                continue
            builder = _EpisodeBuilder(
                history_link=history_link,
                history_path=history_path,
                writer=writer,
                normalised_history=_normalise_text(writer.to_text()),
                questions=[],
            )
            builders[persona_id] = builder
        elif history_link != builder.history_link:
            raise SchemaError(
                f"{path}:{row_number}: persona {persona_id!r} has multiple {window_name} "
                f"history links: {builder.history_link!r} vs {history_link!r}"
            )

        preserve_official_text = option_shuffle_protocol == "official_qwen"
        query = _extract_query(
            row.get("user_query"),
            preserve_whitespace=preserve_official_text,
        )
        correct_answer = (
            _nonempty_preserve_whitespace(row.get("correct_answer"))
            if preserve_official_text
            else _nonempty(row.get("correct_answer"))
        )
        if not correct_answer:
            raise SchemaError(f"{path}:{row_number}: correct_answer is empty")
        try:
            incorrect_answers = _parse_string_list(
                row.get("incorrect_answers"),
                "incorrect_answers",
                preserve_whitespace=preserve_official_text,
            )
        except SchemaError:
            # Malformed/empty distractors cannot define a valid four-way item.
            rows_skipped_invalid_mcq += 1
            continue
        if len(incorrect_answers) != 3:
            # The official 2026-07 snapshot has one train row with no distractors.
            # It cannot define the four-way benchmark objective, so exclude it and
            # expose the count in the audit instead of silently fabricating options.
            rows_skipped_invalid_mcq += 1
            continue
        # Keep the existing canonical sample identity independent of display-only
        # leading/trailing whitespace.  Official prompt text itself remains exact.
        sample_id = _stable_sample_id(
            split_name,
            persona_id,
            query.strip(),
            correct_answer.strip(),
            [answer.strip() for answer in incorrect_answers],
        )
        if sample_id in sample_ids:
            raise SchemaError(
                f"{path}:{row_number}: duplicate full QA record (sample_id={sample_id})"
            )
        try:
            options, correct_index = _shuffle_mcq(
                correct_answer,
                incorrect_answers,
                sample_id=sample_id,
                seed=shuffle_seed,
                shuffle_round=shuffle_round,
                row_index=row_number - 2,
                protocol=option_shuffle_protocol,
            )
        except SchemaError:
            rows_skipped_invalid_mcq += 1
            continue
        sample_ids.add(sample_id)

        if content_overlap_policy != "off":
            hits = _find_target_content_overlaps(
                builder.normalised_history,
                sample_id=sample_id,
                query=query,
                options=options,
                min_chars=content_overlap_min_chars,
            )
            if hits and content_overlap_policy == "error":
                raise LeakageError(
                    f"{path}:{row_number}: current target content occurs verbatim in writer "
                    f"history: {hits}"
                )
            overlap_warnings.extend(hits)

        question = MCQExample(
            sample_id=sample_id,
            reader=ReaderInput(query=query, options=options),
            correct_index=correct_index,
            tags=_make_audit_tags(row, window_name),
            source_row_index=row_number - 2,
        )
        builder.questions.append(question)

    episodes = tuple(
        PersonaEpisode(
            persona_id=persona_id,
            split=split_name,
            window=window_name,
            history_path=builder.history_path,
            writer=builder.writer,
            questions=tuple(builder.questions),
        )
        for persona_id, builder in builders.items()
        if builder.questions
    )
    if not episodes:
        raise SchemaError(f"{path}: no loadable questions")
    return LoadedPersonaDataset(
        csv_path=path,
        split=split_name,
        window=window_name,
        shuffle_seed=shuffle_seed,
        shuffle_round=shuffle_round,
        episodes=episodes,
        rows_seen=rows_seen,
        rows_skipped_missing_history=rows_skipped_missing_history,
        rows_skipped_invalid_mcq=rows_skipped_invalid_mcq,
        rows_skipped_excluded_persona=rows_skipped_excluded_persona,
        excluded_persona_ids=tuple(sorted(excluded)),
        content_overlap_warnings=tuple(overlap_warnings),
    )


def resolve_split_csv(data_root: str | Path, split: str) -> Path:
    split_name = _normalise_split(split)
    return (
        Path(data_root).expanduser().resolve()
        / "benchmark"
        / "text"
        / SPLIT_FILENAMES[split_name]
    )


def load_persona_ids(csv_path: str | Path) -> set[str]:
    path = Path(csv_path).expanduser().resolve()
    fieldnames, rows = _read_csv_rows(path)
    if "persona_id" not in fieldnames:
        raise SchemaError(f"{path}: missing persona_id")
    ids = {_nonempty(row.get("persona_id")) for row in rows}
    ids.discard("")
    return ids


def audit_split_disjointness(
    split_csvs: Mapping[str, str | Path],
) -> dict[str, Any]:
    ids_by_split = {
        _normalise_split(split): load_persona_ids(csv_path)
        for split, csv_path in split_csvs.items()
    }
    overlaps: dict[str, list[str]] = {}
    names = sorted(ids_by_split)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            shared = sorted(ids_by_split[left] & ids_by_split[right])
            overlaps[f"{left}__{right}"] = shared
    return {
        "persona_counts": {
            split: len(persona_ids) for split, persona_ids in ids_by_split.items()
        },
        "overlap_counts": {pair: len(shared) for pair, shared in overlaps.items()},
        "overlap_persona_ids": overlaps,
        "all_disjoint": all(not shared for shared in overlaps.values()),
    }


def _numeric_summary(values: Iterable[int | None]) -> dict[str, int | float | None]:
    numbers = [value for value in values if value is not None]
    if not numbers:
        return {"count": 0, "min": None, "p50": None, "mean": None, "max": None}
    return {
        "count": len(numbers),
        "min": min(numbers),
        "p50": statistics.median(numbers),
        "mean": round(statistics.fmean(numbers), 3),
        "max": max(numbers),
    }


def _counter_dict(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def build_audit_report(dataset: LoadedPersonaDataset) -> dict[str, Any]:
    questions = list(dataset.iter_questions())
    q_per_persona = [len(episode.questions) for episode in dataset.episodes]
    message_counts = [len(episode.writer.messages) for episode in dataset.episodes]
    history_chars = [len(episode.writer.to_text()) for episode in dataset.episodes]
    labels = [question.correct_letter for _, question in questions]
    tags = [question.tags for _, question in questions]

    return {
        "source": {
            "csv_path": str(dataset.csv_path),
            "split": dataset.split,
            "window": dataset.window,
            "shuffle_seed": dataset.shuffle_seed,
            "shuffle_round": dataset.shuffle_round,
            "excluded_persona_ids": list(dataset.excluded_persona_ids),
        },
        "counts": {
            "csv_rows_seen": dataset.rows_seen,
            "questions_loaded": dataset.num_questions,
            "personas_loaded": len(dataset.episodes),
            "histories_loaded": len(dataset.episodes),
            "rows_skipped_missing_history": dataset.rows_skipped_missing_history,
            "rows_skipped_invalid_mcq": dataset.rows_skipped_invalid_mcq,
            "rows_skipped_excluded_persona": dataset.rows_skipped_excluded_persona,
            "duplicate_sample_ids": dataset.num_questions
            - len({question.sample_id for _, question in questions}),
        },
        "grouping": {
            "questions_per_persona": _numeric_summary(q_per_persona),
            "history_messages_per_persona": _numeric_summary(message_counts),
            "history_characters_per_persona": _numeric_summary(history_chars),
            "one_history_per_persona": True,
        },
        "mcq": {
            "option_count_histogram": _counter_dict(
                len(question.reader.options) for _, question in questions
            ),
            "correct_letter_histogram": _counter_dict(labels),
            "all_options_distinct": all(
                len({_normalise_text(item) for item in question.reader.options})
                == len(question.reader.options)
                for _, question in questions
            ),
        },
        "subgroups": {
            "updated": _counter_dict(tag.updated for tag in tags),
            "who": _counter_dict(tag.who for tag in tags),
            "sensitive_info": _counter_dict(tag.sensitive_info for tag in tags),
            "pref_type": _counter_dict(tag.pref_type for tag in tags),
            "conversation_scenario": _counter_dict(
                tag.conversation_scenario for tag in tags
            ),
            "topic_query_unique": len({tag.topic_query for tag in tags if tag.topic_query}),
            "topic_preference_unique": len(
                {tag.topic_preference for tag in tags if tag.topic_preference}
            ),
        },
        "official_token_metadata": {
            "total_tokens": _numeric_summary(tag.total_tokens for tag in tags),
            "distance_to_related_snippet": _numeric_summary(
                tag.distance_to_related_snippet for tag in tags
            ),
            "persona_relevant_tokens": _numeric_summary(
                tag.persona_relevant_tokens for tag in tags
            ),
            "persona_irrelevant_tokens": _numeric_summary(
                tag.persona_irrelevant_tokens for tag in tags
            ),
        },
        "leakage_audit": {
            "writer_schema_fields": [item.name for item in fields(WriterInput)],
            "writer_message_schema_fields": [item.name for item in fields(ChatMessage)],
            "csv_annotation_fields_excluded_from_writer": sorted(
                WRITER_FORBIDDEN_FIELDS
            ),
            "target_content_overlap_warning_count": len(
                dataset.content_overlap_warnings
            ),
            "target_content_overlap_warnings": list(dataset.content_overlap_warnings),
            "writer_inputs_schema_safe": all(
                episode.writer.assert_safe_schema() is None
                for episode in dataset.episodes
            ),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and audit a local PersonaMem-v2 text-only split."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Explicit local CSV. Defaults to DATA_ROOT/benchmark/text/SPLIT.csv.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Root of the downloaded PersonaMem-v2 Hugging Face repository.",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=("train", "val", "validation", "benchmark"),
    )
    parser.add_argument("--window", default="32k", choices=("32k", "128k"))
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument(
        "--shuffle-round",
        type=int,
        default=0,
        help="Deterministic reshuffle index; use epoch number during training.",
    )
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--content-overlap",
        choices=("error", "warn", "off"),
        default="error",
        help="Policy if a complete current query/answer occurs verbatim in writer history.",
    )
    parser.add_argument("--content-overlap-min-chars", type=int, default=24)
    parser.add_argument(
        "--skip-missing-history",
        action="store_true",
        help="Skip all rows for personas whose selected history file is missing.",
    )
    parser.add_argument(
        "--exclude-persona-id",
        action="append",
        default=[],
        help="Persona ID to exclude; repeat for multiple IDs.",
    )
    parser.add_argument(
        "--check-all-splits",
        action="store_true",
        help="Also verify train/val/benchmark persona IDs are disjoint under DATA_ROOT.",
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        help="Optional path to save the same JSON report printed to stdout.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    split = _normalise_split(args.split)
    if args.csv is None:
        if args.data_root is None:
            raise SystemExit("--data-root is required when --csv is omitted")
        csv_path = resolve_split_csv(args.data_root, split)
    else:
        csv_path = args.csv

    dataset = load_personamem_text(
        csv_path,
        split=split,
        window=args.window,
        data_root=args.data_root,
        shuffle_seed=args.shuffle_seed,
        shuffle_round=args.shuffle_round,
        max_rows=args.max_rows,
        content_overlap_policy=args.content_overlap,
        content_overlap_min_chars=args.content_overlap_min_chars,
        skip_missing_history=args.skip_missing_history,
        exclude_persona_ids=args.exclude_persona_id,
    )
    report = build_audit_report(dataset)
    if args.check_all_splits:
        if args.data_root is None:
            raise SystemExit("--check-all-splits requires --data-root")
        report["split_disjointness"] = audit_split_disjointness(
            {
                split_name: resolve_split_csv(args.data_root, split_name)
                for split_name in SPLIT_FILENAMES
            }
        )
        if not report["split_disjointness"]["all_disjoint"]:
            raise LeakageError("Persona IDs overlap across official splits")

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.audit_json is not None:
        args.audit_json.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.expanduser().resolve().write_text(
            rendered + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
