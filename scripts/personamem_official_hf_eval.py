#!/usr/bin/env python3
"""Official-format PersonaMem-v2 MCQ evaluation with local Hugging Face models.

This harness is deliberately separate from ``personamem_prefix_steer.py``.  The
training/pilot script uses one-token forced choice; this file reproduces the
generative 32k MCQ protocol used by PersonaMem-v2's Qwen/VERL evaluation:

* option permutation is ``random.seed(42 + original_csv_row_index)``;
* options are labelled ``(a)`` through ``(d)``;
* the official system, MCQ, and ``<think>`` instructions are used verbatim;
* Qwen's chat template is called with ``enable_thinking=True``;
* validation is greedy and may generate up to 2048 tokens; and
* the last boxed a-d answer after the last ``</think>`` is scored.

An answer without a boxed a-d choice is counted wrong and exposed through the
format-rate metric.  In particular, this harness never loads the official
Azure/OpenAI embedding fallback, which would introduce an external model into a
supposedly local HF comparison.

Three backends share the same data, prompt, generation, and scoring path:

``base``
    A frozen local HF model sees system + full history + current MCQ.
``sft``
    Identical to ``base``; ``--model-path`` points at the official SFT model.
``prefix``
    A frozen base plus a local prefix-steer checkpoint.  History is written
    separately (query/answers never enter WRITE), then system + current MCQ are
    generated with the frozen memory active.

Results are appended one JSON object per line and flushed after every example,
so a pre-empted shard can be resumed safely with ``--resume``.

``--history-condition`` adds paper-grade causal controls without changing the
default official run: ``correct`` uses the query persona's history,
``query_only`` removes history, and ``swapped`` uses a fixed, seeded,
no-self persona derangement.  Separate swapped runs select one of the fixed
derangements with ``--swap-index``.

``--protocol frontier_api`` retains the older top-level ``inference.py`` message
and extraction format as an explicitly labelled compatibility run.  Paper Qwen
numbers must use the default ``qwen_verl`` protocol.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path, PosixPath
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.personamem_retrieval_plan import (
    RetrievalPlan,
    load_retrieval_plan,
    plan_metadata,
    plan_record_sha256,
    render_plan_selection,
)

OFFICIAL_REPO_ROOT = REPO_ROOT / "third_party" / "PersonaMem-v2-official"
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "personamem_v2"
DEFAULT_CLEAN_MANIFEST = REPO_ROOT / "configs" / "personamem_v2_clean_v1.json"
DEFAULT_MODEL_PATH = Path("/work/mingze/models/Qwen3-4B-Instruct-2507")

OFFICIAL_SYSTEM_PROMPT = (
    "You are a helpful assistant that provides personalized responses based on "
    "the user's preferences in conversation history."
)
OFFICIAL_MCQ_TEMPLATE = (
    "\n\nYou are performing a multiple-choice question task. You must choose the "
    "best response from the following options to answer the user query above:\n"
    "{options_text}\n\nProvide your answer in the format: \\boxed{{a}}, "
    "\\boxed{{b}}, \\boxed{{c}}, or \\boxed{{d}}."
)
OFFICIAL_THINK_INSTRUCTION = (
    " Always perform your reasoning inside <think> and </think> tags before your "
    "final answer."
)
OFFICIAL_MAX_PROMPT_TOKENS_32K = 37_000
OFFICIAL_MAX_NEW_TOKENS = 2_048
OFFICIAL_OPTION_SEED = 42
EXPECTED_SUBSET_SIZES = {"official5000": 5_000, "clean4992": 4_992}
HISTORY_CONDITIONS = ("correct", "query_only", "swapped")
DEFAULT_SWAP_SEED = 4_242
HISTORY_RETRIEVAL_MODES = ("full", "bm25", "plan")
DEFAULT_RAG_CHUNK_SIZE = 6
DEFAULT_RAG_CHUNK_OVERLAP = 2
DEFAULT_RAG_TOP_K = 10
BM25_K1 = 1.5
BM25_B = 0.75
BM25_EPSILON = 0.25

# Persona 78 is the sole official train/benchmark identity collision in the
# pinned snapshot.  The clean paper protocol removes it from benchmark as well
# as the seven target-content-overlap samples listed in the manifest.
CLEAN_BENCHMARK_EXCLUDED_PERSONAS = frozenset({"78"})

_BOXED_PATTERNS = (
    re.compile(r"\\boxed\{([a-dA-D])\}"),
    re.compile(r"\\boxed\{\(([a-dA-D])\)\}"),
    re.compile(r"boxed\{([a-dA-D])\}"),
)
_FRONTIER_ANSWER_PATTERNS = (
    re.compile(r"\$\\boxed\{([A-D])\}\$", re.IGNORECASE),
    re.compile(r"\\boxed\{([A-D])\}", re.IGNORECASE),
    re.compile(r"Final Answer:\s*([A-D])", re.IGNORECASE),
    re.compile(r"Answer:\s*([A-D])", re.IGNORECASE),
    re.compile(r"final answer is\s*\$?\\boxed\{([A-D])\}\$?", re.IGNORECASE),
    re.compile(r"final answer is\s*([A-D])", re.IGNORECASE),
    re.compile(r"the answer is\s*\$?\\boxed\{([A-D])\}\$?", re.IGNORECASE),
    re.compile(r"the answer is\s*([A-D])", re.IGNORECASE),
    re.compile(r"\b([A-D])\.\s*$", re.IGNORECASE | re.MULTILINE),
)


@dataclass(frozen=True, slots=True)
class OfficialMCQItem:
    """One benchmark row after the official deterministic option shuffle."""

    row_index: int
    sample_id: str
    persona_id: str
    history_link: str
    query: str
    options: tuple[str, str, str, str]
    correct_index: int
    correct_letter: str
    correct_answer: str
    tags: dict[str, str]

    def options_text(self, *, labels: str = "lower_parenthesized") -> str:
        if labels == "lower_parenthesized":
            render = lambda index: f"({chr(ord('a') + index)})"
        elif labels == "upper_dotted":
            render = lambda index: f"{chr(ord('A') + index)}."
        else:
            raise ValueError(f"unknown option label style {labels!r}")
        return "\n".join(
            f"{render(index)} {option}"
            for index, option in enumerate(self.options)
        )

    @property
    def user_prompt(self) -> str:
        return (
            self.query
            + OFFICIAL_MCQ_TEMPLATE.format(
                options_text=self.options_text(labels="lower_parenthesized")
            )
            + OFFICIAL_THINK_INSTRUCTION
        )

    @property
    def official_correct_answer(self) -> str:
        return f"({self.correct_letter}) {self.correct_answer}"


@dataclass(frozen=True, slots=True)
class HistoryChunk:
    """One message-indexed retrieval chunk, matching the official RAG layout."""

    start_idx: int
    messages: tuple[dict[str, Any], ...]
    text: str


def _nonempty(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "none", "null"} else text


def parse_user_query(value: Any) -> str:
    """Parse the official JSON/Python-literal user message into its content."""

    if isinstance(value, Mapping):
        query = _nonempty(value.get("content"))
        if not query:
            raise ValueError("user_query mapping has empty content")
        return query
    text = _nonempty(value)
    if not text:
        raise ValueError("user_query is empty")
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, Mapping):
            query = _nonempty(parsed.get("content"))
            if not query:
                raise ValueError("user_query mapping has empty content")
            return query
    return text


def parse_incorrect_answers(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        parsed = value
    else:
        text = _nonempty(value)
        if not text:
            raise ValueError("incorrect_answers is empty")
        parsed = None
        for parser in (json.loads, ast.literal_eval):
            try:
                candidate = parser(text)
            except (ValueError, SyntaxError, json.JSONDecodeError, TypeError):
                continue
            if isinstance(candidate, (list, tuple)):
                parsed = candidate
                break
        if parsed is None:
            raise ValueError("incorrect_answers is not a list")
    answers = [_nonempty(answer) for answer in parsed]
    if len(answers) != 3 or any(not answer for answer in answers):
        raise ValueError("official MCQ needs exactly three non-empty distractors")
    return answers


def stable_sample_id(
    persona_id: str,
    query: str,
    correct_answer: str,
    incorrect_answers: Sequence[str],
) -> str:
    """Match ``scripts.personamem_v2_data`` benchmark sample IDs."""

    raw = json.dumps(
        [
            "benchmark",
            persona_id,
            query,
            correct_answer,
            list(incorrect_answers),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def official_shuffle_options(
    correct_answer: str,
    incorrect_answers: Sequence[str],
    *,
    row_index: int,
) -> tuple[tuple[str, str, str, str], int]:
    """Exact option shuffle from official ``data_preprocess_rft.py``."""

    if len(incorrect_answers) != 3:
        raise ValueError("official option shuffle needs three distractors")
    options = [str(correct_answer), *(str(value) for value in incorrect_answers)]
    if len(set(options)) != 4:
        raise ValueError("MCQ answer options are not distinct")
    random.Random(OFFICIAL_OPTION_SEED + row_index).shuffle(options)
    correct_index = options.index(str(correct_answer))
    return tuple(options), correct_index  # type: ignore[return-value]


def frontier_shuffle_options(
    correct_answer: str,
    incorrect_answers: Sequence[str],
    *,
    persona_id: str,
    query: str,
) -> tuple[tuple[str, str, str, str], int]:
    """Match top-level ``inference.py`` (including its process-hash seed)."""

    options = [str(correct_answer), *(str(value) for value in incorrect_answers)]
    modified_query = (
        query
        + " Please recall my related preferences from our conversation history to "
        "give personalized responses."
    )
    row_seed = hash(f"{persona_id}_{modified_query}") % 2**32
    random.Random(row_seed).shuffle(options)
    correct_index = options.index(str(correct_answer))
    return tuple(options), correct_index  # type: ignore[return-value]


def build_official_item(
    row: Mapping[str, Any], *, row_index: int, protocol: str = "qwen_verl"
) -> OfficialMCQItem:
    persona_id = _nonempty(row.get("persona_id"))
    history_link = _nonempty(row.get("chat_history_32k_link"))
    correct_answer = _nonempty(row.get("correct_answer"))
    if not persona_id or not history_link or not correct_answer:
        raise ValueError(
            f"row {row_index}: persona_id/history/correct_answer must be non-empty"
        )
    query = parse_user_query(row.get("user_query"))
    incorrect_answers = parse_incorrect_answers(row.get("incorrect_answers"))
    sample_id = stable_sample_id(
        persona_id, query, correct_answer, incorrect_answers
    )
    if protocol == "qwen_verl":
        options, correct_index = official_shuffle_options(
            correct_answer, incorrect_answers, row_index=row_index
        )
    elif protocol == "frontier_api":
        options, correct_index = frontier_shuffle_options(
            correct_answer,
            incorrect_answers,
            persona_id=persona_id,
            query=query,
        )
    else:
        raise ValueError(f"unknown protocol {protocol!r}")
    tag_names = (
        "topic_query",
        "topic_preference",
        "conversation_scenario",
        "pref_type",
        "who",
        "updated",
        "sensitive_info",
        "distance_from_related_snippet_to_query_32k",
    )
    return OfficialMCQItem(
        row_index=row_index,
        sample_id=sample_id,
        persona_id=persona_id,
        history_link=history_link,
        query=query,
        options=options,
        correct_index=correct_index,
        correct_letter=chr(ord("a") + correct_index),
        correct_answer=correct_answer,
        tags={name: _nonempty(row.get(name)) for name in tag_names},
    )


def load_official_items(
    benchmark_csv: str | Path,
    *,
    subset: str,
    clean_manifest: str | Path = DEFAULT_CLEAN_MANIFEST,
    protocol: str = "qwen_verl",
) -> tuple[OfficialMCQItem, ...]:
    """Load official5000 or the pinned clean4992 without opening histories."""

    if subset not in EXPECTED_SUBSET_SIZES:
        raise ValueError(f"unknown subset {subset!r}")
    excluded_personas: set[str] = set()
    excluded_samples: set[str] = set()
    if subset == "clean4992":
        manifest = json.loads(Path(clean_manifest).read_text(encoding="utf-8"))
        excluded_personas.update(CLEAN_BENCHMARK_EXCLUDED_PERSONAS)
        excluded_personas.update(
            str(value)
            for value in manifest.get(
                "exclude_persona_ids_from_benchmark_clean", []
            )
        )
        excluded_samples.update(
            str(value)
            for value in manifest.get("exclude_sample_ids_all_windows", [])
        )

    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    items: list[OfficialMCQItem] = []
    seen: set[str] = set()
    with Path(benchmark_csv).open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            item = build_official_item(
                row, row_index=row_index, protocol=protocol
            )
            if item.sample_id in seen:
                raise ValueError(f"duplicate sample ID {item.sample_id}")
            seen.add(item.sample_id)
            if (
                item.persona_id in excluded_personas
                or item.sample_id in excluded_samples
            ):
                continue
            items.append(item)
    return tuple(items)


def contiguous_shard(
    items: Sequence[OfficialMCQItem], *, shard_index: int, num_shards: int
) -> tuple[OfficialMCQItem, ...]:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    start = len(items) * shard_index // num_shards
    end = len(items) * (shard_index + 1) // num_shards
    return tuple(items[start:end])


def _stable_seed(seed: int, *parts: str) -> int:
    payload = ":".join((str(seed), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def persona_history_items(
    items: Sequence[OfficialMCQItem],
) -> dict[str, OfficialMCQItem]:
    """Return one auditable history source per persona.

    Every official MCQ for one persona must name the same 32k history.  Failing
    on a mismatch is important for swapped-history evaluation: silently taking
    the first row would otherwise make the intervention ambiguous.
    """

    result: dict[str, OfficialMCQItem] = {}
    for item in items:
        previous = result.get(item.persona_id)
        if previous is not None and previous.history_link != item.history_link:
            raise ValueError(
                f"persona {item.persona_id!r} has multiple 32k histories: "
                f"{previous.history_link!r} and {item.history_link!r}"
            )
        result.setdefault(item.persona_id, item)
    return result


def make_history_derangements(
    persona_ids: Sequence[str],
    *,
    num_swaps: int,
    seed: int,
) -> tuple[dict[str, str], ...]:
    """Build stable, distinct, bijective, no-self persona-history mappings."""

    if num_swaps < 1:
        raise ValueError("num_swaps must be positive")
    canonical_ids = sorted(str(persona_id) for persona_id in persona_ids)
    if len(set(canonical_ids)) != len(canonical_ids):
        raise ValueError("persona IDs must be unique")
    if len(canonical_ids) < 2:
        raise ValueError("swapped history requires at least two personas")
    if num_swaps > len(canonical_ids) - 1:
        raise ValueError(
            f"num_swaps={num_swaps} exceeds the {len(canonical_ids) - 1} "
            "distinct no-self cyclic derangements available"
        )

    cycle = canonical_ids.copy()
    random.Random(_stable_seed(seed, "persona-swap-cycle")).shuffle(cycle)
    mappings: list[dict[str, str]] = []
    for swap_index in range(num_swaps):
        shift = swap_index + 1
        mapping = {
            persona_id: cycle[(position + shift) % len(cycle)]
            for position, persona_id in enumerate(cycle)
        }
        if any(target == source for target, source in mapping.items()):
            raise AssertionError("history derangement contains a fixed point")
        if len(set(mapping.values())) != len(mapping):
            raise AssertionError("history derangement is not bijective")
        mappings.append(mapping)
    return tuple(mappings)


def history_source_item(
    item: OfficialMCQItem,
    *,
    condition: str,
    histories: Mapping[str, OfficialMCQItem],
    swap_mapping: Mapping[str, str] | None = None,
) -> OfficialMCQItem | None:
    """Select the history intervention for one fixed query item."""

    if condition == "query_only":
        return None
    if condition == "correct":
        source_persona = item.persona_id
    elif condition == "swapped":
        if swap_mapping is None:
            raise ValueError("swapped condition requires a persona mapping")
        source_persona = swap_mapping[item.persona_id]
        if source_persona == item.persona_id:
            raise AssertionError("swapped condition selected the query history")
    else:
        raise ValueError(f"unknown history condition {condition!r}")
    try:
        return histories[source_persona]
    except KeyError as exc:
        raise ValueError(
            f"no history item for source persona {source_persona!r}"
        ) from exc


def selection_digest(items: Sequence[OfficialMCQItem]) -> str:
    """Fingerprint ordered samples so resume/merge cannot mix selections."""

    payload = "\n".join(item.sample_id for item in items).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def official_messages(
    item: OfficialMCQItem,
    history_messages: Sequence[Mapping[str, Any]] = (),
    *,
    protocol: str = "qwen_verl",
) -> list[dict[str, str]]:
    """Construct the selected official message protocol."""

    # Do not reduce history dictionaries to role/content here.  A handful of
    # malformed official JSON turns contain extra keys, and the released MCQ
    # parquet preserves them byte-for-byte in ``prompt``.  Qwen's template uses
    # role/content only, while retaining the raw dictionaries keeps this
    # reconstructed protocol exactly auditable against the release.
    history = [dict(message) for message in history_messages]
    if protocol == "qwen_verl":
        return [
            {"role": "system", "content": OFFICIAL_SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": item.user_prompt},
        ]
    if protocol == "frontier_api":
        query = (
            item.query
            + " Please recall my related preferences from our conversation history "
            "to give personalized responses."
        )
        instruction = (
            "Please choose the best answer from the following options:\n\n"
            + item.options_text(labels="upper_dotted")
            + "\n\nThink step by step about which answer best fits the user's query "
            "and conversation context. Provide your reasoning first, then give your "
            "final answer as 'Final Answer: [Letter]'"
        )
        return [
            *history,
            {"role": "user", "content": query},
            {"role": "system", "content": instruction},
        ]
    raise ValueError(f"unknown protocol {protocol!r}")


def extract_solution(response: str) -> str:
    """Match official reward code: score only content after the last think block."""

    if not response:
        return ""
    last_think_end = response.rfind("</think>")
    if last_think_end != -1:
        return (
            response[last_think_end + len("</think>") :]
            .replace("</think>", "")
            .strip()
        )
    return response.strip()


def extract_official_boxed_letter(response: str) -> str | None:
    """Match official pattern order and take the last match for that pattern."""

    solution = extract_solution(response)
    for pattern in _BOXED_PATTERNS:
        matches = pattern.findall(solution)
        if matches:
            return str(matches[-1]).lower()
    return None


def extract_frontier_letter(response: str) -> str | None:
    if not response:
        return None
    for pattern in _FRONTIER_ANSWER_PATTERNS:
        match = pattern.search(response)
        if match:
            return str(match.group(1)).lower()
    return None


def score_response(
    response: str,
    item: OfficialMCQItem,
    *,
    missing_box_policy: str,
    protocol: str = "qwen_verl",
) -> tuple[str | None, bool, bool]:
    """Return prediction, correctness, and an always-false legacy fallback flag."""

    prediction = (
        extract_official_boxed_letter(response)
        if protocol == "qwen_verl"
        else extract_frontier_letter(response)
    )
    if prediction is not None:
        return prediction, prediction == item.correct_letter, False
    if missing_box_policy != "wrong":
        raise ValueError(f"unknown missing-box policy {missing_box_policy!r}")
    return None, False, False


def load_history_messages(
    item: OfficialMCQItem, *, benchmark_csv: Path, data_root: Path
) -> tuple[list[dict[str, Any]], Path]:
    """Resolve and return the raw official message dictionaries."""

    from scripts.personamem_v2_data import resolve_history_path

    history_path = resolve_history_path(
        item.history_link, csv_path=benchmark_csv, data_root=data_root
    )
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("chat_history", payload.get("messages"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{history_path}: history must be a non-empty message list")
    messages: list[dict[str, Any]] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, Mapping) or "role" not in raw:
            raise ValueError(f"{history_path}: invalid message {index}")
        content = raw.get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError(f"{history_path}: non-text message {index}")
        messages.append(dict(raw))
    return messages, history_path


_BM25_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def _retrieval_text_content(content: Any) -> str:
    """Mirror the official RAG serializer, including its multimodal fallback."""

    if isinstance(content, list):
        return " ".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
        )
    return "" if content is None else str(content)


def chunk_history_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    chunk_size: int = DEFAULT_RAG_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_RAG_CHUNK_OVERLAP,
) -> tuple[HistoryChunk, ...]:
    """Apply the released RAG baseline's message chunking exactly.

    The first history ``system`` message is excluded, as in
    ``third_party/PersonaMem-v2-official/verl_custom/rag.py``.  Remaining
    messages are split into overlapping, message-counted chunks.  Chunk text is
    used only for retrieval; the selected raw messages are sent to Qwen.
    """

    if chunk_size < 1:
        raise ValueError("RAG chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("RAG chunk_overlap must be in [0, chunk_size)")

    conversation: list[dict[str, Any]] = []
    removed_first_system = False
    for raw in messages:
        message = dict(raw)
        if message.get("role") == "system" and not removed_first_system:
            removed_first_system = True
            continue
        conversation.append(message)
    if not conversation:
        return ()

    stride = chunk_size - chunk_overlap
    chunks: list[HistoryChunk] = []
    for start in range(0, len(conversation), stride):
        end = min(start + chunk_size, len(conversation))
        chunk_messages = conversation[start:end]
        text = "\n".join(
            f"{str(message.get('role', 'unknown')).capitalize()}: "
            f"{_retrieval_text_content(message.get('content', ''))}"
            for message in chunk_messages
        )
        chunks.append(
            HistoryChunk(
                start_idx=start,
                messages=tuple(dict(message) for message in chunk_messages),
                text=text,
            )
        )
        if end >= len(conversation):
            break
    return tuple(chunks)


def bm25_tokenize(text: str) -> tuple[str, ...]:
    """Deterministic, dependency-free tokenization for the local BM25 baseline."""

    return tuple(_BM25_TOKEN_PATTERN.findall(str(text).casefold()))


def bm25_okapi_scores(
    query: str,
    chunks: Sequence[HistoryChunk],
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
    epsilon: float = BM25_EPSILON,
) -> tuple[float, ...]:
    """Score chunks with the standard ``rank_bm25.BM25Okapi`` equations."""

    if k1 <= 0:
        raise ValueError("BM25 k1 must be positive")
    if not 0 <= b <= 1:
        raise ValueError("BM25 b must be in [0, 1]")
    if epsilon < 0:
        raise ValueError("BM25 epsilon must be non-negative")
    if not chunks:
        return ()

    documents = [bm25_tokenize(chunk.text) for chunk in chunks]
    frequencies = [Counter(document) for document in documents]
    document_frequency: Counter[str] = Counter()
    for frequency in frequencies:
        document_frequency.update(frequency.keys())

    num_documents = len(documents)
    idf = {
        token: math.log(num_documents - count + 0.5)
        - math.log(count + 0.5)
        for token, count in document_frequency.items()
    }
    average_idf = sum(idf.values()) / len(idf) if idf else 0.0
    epsilon_idf = epsilon * average_idf
    for token, value in tuple(idf.items()):
        if value < 0:
            idf[token] = epsilon_idf

    lengths = [len(document) for document in documents]
    average_length = sum(lengths) / num_documents
    if average_length == 0:
        average_length = 1.0
    query_tokens = bm25_tokenize(query)
    scores: list[float] = []
    for length, frequency in zip(lengths, frequencies):
        score = 0.0
        normalization = k1 * (1 - b + b * length / average_length)
        for token in query_tokens:
            term_frequency = frequency.get(token, 0)
            if not term_frequency:
                continue
            score += idf.get(token, 0.0) * (
                term_frequency * (k1 + 1)
                / (term_frequency + normalization)
            )
        scores.append(float(score))
    return tuple(scores)


def retrieve_bm25_history(
    messages: Sequence[Mapping[str, Any]],
    query: str,
    *,
    chunk_size: int = DEFAULT_RAG_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_RAG_CHUNK_OVERLAP,
    top_k: int = DEFAULT_RAG_TOP_K,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retrieve BM25 chunks, restore chronology, and deduplicate overlaps."""

    if top_k < 1:
        raise ValueError("RAG top_k must be positive")
    chunks = chunk_history_messages(
        messages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    scores = bm25_okapi_scores(query, chunks)
    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda index: (-scores[index], chunks[index].start_idx),
    )
    selected_indices = sorted(
        ranked_indices[: min(top_k, len(chunks))],
        key=lambda index: chunks[index].start_idx,
    )

    selected_messages: list[dict[str, Any]] = []
    selected_message_indices: list[int] = []
    added_indices: set[int] = set()
    for chunk_index in selected_indices:
        chunk = chunks[chunk_index]
        for offset, message in enumerate(chunk.messages):
            message_index = chunk.start_idx + offset
            if message_index in added_indices:
                continue
            added_indices.add(message_index)
            selected_message_indices.append(message_index)
            selected_messages.append(dict(message))

    audit = {
        "candidate_chunks": len(chunks),
        "retrieved_chunk_starts": [
            chunks[index].start_idx for index in selected_indices
        ],
        "retrieved_chunk_scores": [scores[index] for index in selected_indices],
        "retrieved_message_indices": selected_message_indices,
        "retrieved_messages": len(selected_messages),
    }
    return selected_messages, audit


def _flatten_input_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("expected one tokenized prompt")
        value = value[0]
    return [int(token_id) for token_id in value]


def encode_official_prompt(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    enable_thinking: bool,
) -> list[int]:
    """Apply the official Qwen generation template without silent fallback."""

    try:
        encoded = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError as exc:
        raise TypeError(
            "tokenizer chat template does not accept enable_thinking; official "
            "Qwen evaluation requires a Qwen3-compatible tokenizer"
        ) from exc
    ids = _flatten_input_ids(encoded)
    if not ids:
        raise ValueError("official prompt tokenized to an empty sequence")
    return ids


def _torch_dtype(name: str) -> Any:
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash one immutable paper artifact without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prefix_checkpoint_file_identity(path: str | Path) -> dict[str, Any]:
    """Return the shard/merge identity for one exact checkpoint file."""

    checkpoint_path = Path(path)
    return {
        "resolved_path": str(checkpoint_path.resolve()),
        "sha256": file_sha256(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
    }


def _checkpoint_manifest(
    checkpoint: Mapping[str, Any],
    *,
    config: Any,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Extract auditable architecture/training provenance from a checkpoint."""

    state = checkpoint["state"]
    if not isinstance(state, Mapping):
        raise ValueError("prefix checkpoint state must be a tensor mapping")
    try:
        adapter_parameter_count = sum(
            int(tensor.numel()) for tensor in state.values()
        )
    except AttributeError as exc:
        raise ValueError("prefix checkpoint state contains a non-tensor value") from exc
    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, Mapping):
        checkpoint_args = {}
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    training_exposure = metadata.get("training_exposure")
    if not isinstance(training_exposure, Mapping):
        training_exposure = {}
    train_objective = metadata.get("train_objective")
    if not isinstance(train_objective, Mapping):
        train_objective = {}
    config_dict = asdict(config)
    return {
        **prefix_checkpoint_file_identity(checkpoint_path),
        "adapter_tensor_count": len(state),
        "adapter_parameter_count": adapter_parameter_count,
        "prefix_config_sha256": _canonical_json_sha256(config_dict),
        "training_model_path": checkpoint_args.get("model_path"),
        "training_reader_protocol": checkpoint_args.get("reader_protocol"),
        "training_max_history_tokens": checkpoint_args.get(
            "max_history_tokens"
        ),
        "training_history_truncation": checkpoint_args.get(
            "history_truncation"
        ),
        "training_task_loss": checkpoint_args.get("task_loss"),
        "training_optimizer_updates": training_exposure.get(
            "actual_optimizer_updates"
        ),
        "training_label_exposures": training_exposure.get(
            "actual_label_exposures"
        ),
        "training_data_fingerprint_sha256": training_exposure.get(
            "train_data_fingerprint_sha256"
        ),
        "training_objective_reader_protocol": train_objective.get(
            "reader_protocol"
        ),
    }


def validate_prefix_checkpoint_contract(
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
) -> None:
    """Reject a checkpoint evaluated on a different training contract."""

    mismatches: list[str] = []
    declared_model = manifest.get("training_model_path")
    if declared_model is not None and (
        Path(str(declared_model)).resolve() != Path(args.model_path).resolve()
    ):
        mismatches.append(
            "model_path: checkpoint="
            f"{str(declared_model)!r}, evaluation={str(args.model_path)!r}"
        )
    expected_reader_protocol = (
        "official_qwen" if args.protocol == "qwen_verl" else None
    )
    declared_reader_protocol = manifest.get("training_reader_protocol")
    if (
        expected_reader_protocol is not None
        and declared_reader_protocol is not None
        and declared_reader_protocol != expected_reader_protocol
    ):
        mismatches.append(
            "reader_protocol: checkpoint="
            f"{declared_reader_protocol!r}, evaluation requires "
            f"{expected_reader_protocol!r}"
        )
    declared_history_tokens = manifest.get("training_max_history_tokens")
    if (
        declared_history_tokens is not None
        and int(declared_history_tokens) != args.max_history_tokens
    ):
        mismatches.append(
            "max_history_tokens: checkpoint="
            f"{declared_history_tokens!r}, evaluation={args.max_history_tokens!r}"
        )
    declared_truncation = manifest.get("training_history_truncation")
    if (
        declared_truncation is not None
        and declared_truncation != args.history_truncation
    ):
        mismatches.append(
            "history_truncation: checkpoint="
            f"{declared_truncation!r}, evaluation={args.history_truncation!r}"
        )
    if mismatches and not args.allow_checkpoint_training_mismatch:
        raise ValueError(
            "prefix checkpoint/evaluation contract mismatch: "
            + "; ".join(mismatches)
            + ". Use --allow-checkpoint-training-mismatch only for a labelled "
            "diagnostic ablation."
        )


def load_prefix_checkpoint(
    model: Any, checkpoint_path: str | Path
) -> tuple[Any, dict[str, Any]]:
    """Attach the exact checkpoint-declared steer architecture and load it."""

    import torch

    from deltamem.core.prefix_steer import (
        PrefixSteerConfig,
        attach_prefix_steer,
        freeze_backbone_keep_steer,
        is_steer_param_name,
    )

    with torch.serialization.safe_globals([PosixPath]):
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    if not isinstance(checkpoint, Mapping) or "state" not in checkpoint:
        raise ValueError("prefix checkpoint must contain state/cfg metadata")
    raw_cfg = dict(checkpoint.get("cfg") or {})
    if not raw_cfg:
        raise ValueError("prefix checkpoint has no cfg metadata")
    tuple_fields = {"steer_layers", "prefix_layers"}
    known = {field.name for field in fields(PrefixSteerConfig)}
    config_values = {
        key: tuple(value) if key in tuple_fields and value is not None else value
        for key, value in raw_cfg.items()
        if key in known
    }
    # Legacy checkpoints predate these non-parameterized reader/layout switches.
    # Their absence must resolve to the behavior they were trained with, not to a
    # future dataclass default.
    config_values.setdefault("pool_reads", False)
    config_values.setdefault("history_pool_mode", "none")
    config_values.setdefault("hybrid_read_mode", "none")
    config_values.setdefault("hybrid_prefix_gate_mode", "fixed")
    config_values.setdefault("hybrid_prefix_gate_init", 0.1)
    config_values.setdefault("prefix_write_layout", "global")
    config_values.setdefault("prefix_write_overlap_tokens", 0)
    config = PrefixSteerConfig(**config_values)
    attach_prefix_steer(model, config)
    freeze_backbone_keep_steer(model)
    missing, unexpected = model.load_state_dict(checkpoint["state"], strict=False)
    steer_names = {
        name
        for name, _ in model.named_parameters()
        if is_steer_param_name(name)
    }
    missing_steer = sorted(name for name in missing if name in steer_names)
    if missing_steer or unexpected:
        raise RuntimeError(
            "prefix checkpoint/config mismatch: "
            f"missing steer={missing_steer[:8]}, unexpected={unexpected[:8]}"
        )
    manifest = _checkpoint_manifest(
        checkpoint,
        config=config,
        checkpoint_path=checkpoint_path,
    )
    loaded_adapter_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if is_steer_param_name(name)
    )
    if loaded_adapter_parameters != manifest["adapter_parameter_count"]:
        raise RuntimeError(
            "loaded steer parameter count differs from checkpoint state: "
            f"loaded={loaded_adapter_parameters}, "
            f"checkpoint={manifest['adapter_parameter_count']}"
        )
    return config, manifest


def load_model_and_tokenizer(
    args: argparse.Namespace,
) -> tuple[Any, Any, Any, dict[str, Any] | None]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=args.local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=_torch_dtype(args.dtype),
        attn_implementation=args.attn_impl,
        local_files_only=args.local_files_only,
    ).to(args.device)
    prefix_config = None
    prefix_checkpoint_manifest = None
    if args.backend == "prefix":
        prefix_config, prefix_checkpoint_manifest = load_prefix_checkpoint(
            model, args.checkpoint
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    model.config.use_cache = True
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return model, tokenizer, prefix_config, prefix_checkpoint_manifest


def configure_prefix_contribution(
    model: Any,
    prefix_config: Any,
    mode: str,
) -> None:
    """Select full hybrid or its exact same-weights pooled-only branch."""
    from deltamem.core.prefix_steer import set_hybrid_prefix_off

    if mode not in ("full", "off"):
        raise ValueError(f"unknown prefix contribution mode {mode!r}")
    is_hybrid = (
        prefix_config is not None
        and prefix_config.hybrid_read_mode == "pooled_plus_prefix"
    )
    if mode == "off" and not is_hybrid:
        raise ValueError(
            "--prefix-contribution off requires a checkpoint with "
            "hybrid_read_mode='pooled_plus_prefix'"
        )
    set_hybrid_prefix_off(model, mode == "off")


def eos_token_ids(model: Any, tokenizer: Any) -> set[int]:
    """Return the stop IDs used by the official VERL validation path.

    ``ray_trainer.py`` passes only ``tokenizer.eos_token_id`` to the rollout
    worker.  In particular, Qwen's packaged ``generation_config.json`` also
    lists ``<|endoftext|>`` (151643), but the official validation does *not*
    treat that token as EOS.  Merging the generation-config IDs here made many
    long-context Base responses stop at their first token.
    """

    del model  # kept in the public helper signature for checkpoint compatibility
    values: list[Any] = [getattr(tokenizer, "eos_token_id", None)]
    result: set[int] = set()
    for value in values:
        if isinstance(value, (list, tuple, set)):
            result.update(int(item) for item in value if item is not None)
        elif value is not None:
            result.add(int(value))
    return result


def greedy_generate(
    model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    *,
    device: str,
    max_new_tokens: int,
    prefix_memory: bool,
) -> tuple[str, int]:
    """Greedy HF generation with the steer branch's incremental memory cache."""

    import torch

    from deltamem.core.global_prefix import SEG_QRY
    from deltamem.core.prefix_steer import (
        set_mem_cache,
        set_steer_enabled,
        set_steer_segments,
    )

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    if prefix_memory:
        set_steer_enabled(model, True)
        set_mem_cache(model, True)
        segments = torch.full_like(input_ids, SEG_QRY)
        set_steer_segments(model, segments, torch.ones_like(input_ids).bool())
    stops = eos_token_ids(model, tokenizer)
    generated: list[int] = []
    try:
        with torch.inference_mode():
            output = model(input_ids=input_ids, use_cache=True)
            past = output.past_key_values
            logits = output.logits[0, -1]
            for _ in range(max_new_tokens):
                next_token = int(logits.argmax().item())
                if next_token in stops:
                    break
                generated.append(next_token)
                output = model(
                    input_ids=torch.tensor(
                        [[next_token]], dtype=torch.long, device=device
                    ),
                    past_key_values=past,
                    use_cache=True,
                )
                past = output.past_key_values
                logits = output.logits[0, -1]
    finally:
        if prefix_memory:
            set_mem_cache(model, False)
    return tokenizer.decode(generated, skip_special_tokens=True), len(generated)


def write_prefix_history(
    model: Any,
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    persona_id: str,
    device: str,
    max_history_tokens: int,
    history_truncation: str,
) -> int:
    """Schema-safe WRITE using the same code path as prefix training."""

    from scripts.personamem_prefix_steer import encode_history, write_persona_memory
    from scripts.personamem_v2_data import ChatMessage, WriterInput

    writer = WriterInput(
        messages=tuple(
            ChatMessage(
                role=str(message["role"]),
                content=(
                    ""
                    if message.get("content") is None
                    else str(message["content"])
                ),
            )
            for message in messages
        )
    )
    history_ids = encode_history(
        tokenizer,
        writer,
        persona_id=persona_id,
        max_history_tokens=max_history_tokens,
        truncation=history_truncation,
    )
    write_persona_memory(
        model,
        history_ids,
        device=device,
        grad=False,
        prefix_enabled=True,
    )
    return len(history_ids)


def summarize_records(
    records: Sequence[Mapping[str, Any]], *, expected: int | None = None
) -> dict[str, Any]:
    total = len(records)
    correct = sum(bool(record.get("is_correct")) for record in records)
    parsed = sum(record.get("predicted_letter") is not None for record in records)
    fallback = sum(bool(record.get("embedding_fallback_used")) for record in records)
    by_persona: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_persona[str(record["persona_id"])].append(record)
    persona_scores = [
        sum(bool(row.get("is_correct")) for row in rows) / len(rows)
        for rows in by_persona.values()
    ]

    subgroups: dict[str, Any] = {}
    tag_names = (
        "pref_type",
        "who",
        "updated",
        "sensitive_info",
        "topic_query",
    )
    for tag_name in tag_names:
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for record in records:
            groups[str(record.get("tags", {}).get(tag_name, ""))].append(record)
        subgroups[tag_name] = {
            name: {
                "n": len(rows),
                "accuracy": (
                    sum(bool(row.get("is_correct")) for row in rows) / len(rows)
                ),
            }
            for name, rows in sorted(groups.items())
        }
    return {
        "n": total,
        "expected_n": expected,
        "complete": expected is None or total == expected,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "boxed_format_rate": parsed / total if total else None,
        "embedding_fallback_count": fallback,
        "num_personas": len(by_persona),
        "persona_macro_accuracy": (
            sum(persona_scores) / len(persona_scores) if persona_scores else None
        ),
        "subgroups": subgroups,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSONL record"
                    ) from exc
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=DEFAULT_DATA_ROOT / "benchmark" / "text" / "benchmark.csv",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--clean-manifest", type=Path, default=DEFAULT_CLEAN_MANIFEST
    )
    parser.add_argument(
        "--subset",
        choices=tuple(EXPECTED_SUBSET_SIZES),
        default="official5000",
    )
    parser.add_argument(
        "--protocol",
        choices=("qwen_verl", "frontier_api"),
        default="qwen_verl",
        help="paper Qwen runs use qwen_verl; frontier_api is compatibility only",
    )
    parser.add_argument("--backend", choices=("base", "sft", "prefix"), required=True)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--allow-checkpoint-training-mismatch",
        action="store_true",
        help="diagnostic only: permit a prefix checkpoint trained with a "
        "different backbone, reader protocol, history cap, or truncation",
    )
    parser.add_argument(
        "--prefix-contribution",
        choices=("full", "off"),
        default="full",
        help="prefix backend only: full hybrid, or the exact same-checkpoint "
        "pooled-steer branch with only the additive prefix contribution disabled",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--attn-impl", default="sdpa")
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=OFFICIAL_MAX_PROMPT_TOKENS_32K,
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=OFFICIAL_MAX_NEW_TOKENS
    )
    parser.add_argument("--max-history-tokens", type=int, default=32_768)
    parser.add_argument(
        "--history-truncation", choices=("head", "tail"), default="tail"
    )
    parser.add_argument(
        "--history-condition",
        choices=HISTORY_CONDITIONS,
        default="correct",
        help=(
            "correct keeps the official full history; query_only removes it; "
            "swapped uses a fixed different-persona history derangement"
        ),
    )
    parser.add_argument(
        "--history-retrieval",
        choices=HISTORY_RETRIEVAL_MODES,
        default="full",
        help=(
            "full keeps the selected history; bm25 applies local lexical "
            "retrieval; plan loads immutable precomputed message indices or "
            "facts. All modes retain the common Qwen generation/scorer"
        ),
    )
    parser.add_argument(
        "--retrieval-plan",
        type=Path,
        help=(
            "content-addressed JSONL plan; required only with "
            "--history-retrieval plan (manifest is PATH.manifest.json)"
        ),
    )
    parser.add_argument(
        "--rag-chunk-size",
        type=int,
        default=DEFAULT_RAG_CHUNK_SIZE,
        help="messages per retrieval chunk (official RAG default: 6)",
    )
    parser.add_argument(
        "--rag-chunk-overlap",
        type=int,
        default=DEFAULT_RAG_CHUNK_OVERLAP,
        help="overlapping messages between chunks (official RAG default: 2)",
    )
    parser.add_argument(
        "--rag-top-k",
        type=int,
        default=DEFAULT_RAG_TOP_K,
        help="retrieved chunks per query (official RAG default: 10)",
    )
    parser.add_argument(
        "--num-swaps",
        type=int,
        default=1,
        help="number of fixed distinct persona derangements to define",
    )
    parser.add_argument(
        "--swap-index",
        type=int,
        default=0,
        help="zero-based derangement selected by this swapped-condition run",
    )
    parser.add_argument(
        "--swap-seed",
        type=int,
        default=DEFAULT_SWAP_SEED,
        help="stable seed used to construct persona-history derangements",
    )
    parser.add_argument(
        "--missing-box-policy",
        choices=("wrong",),
        default="wrong",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--max-items", "--max-queries", dest="max_items", type=int, default=0
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-subset-size-mismatch",
        action="store_true",
        help="diagnostic only; paper runs should keep the exact 5000/4992 assertion",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.backend == "prefix" and not args.checkpoint:
        parser.error("--backend prefix requires --checkpoint")
    if (
        args.backend == "prefix"
        and args.checkpoint
        and not args.checkpoint.is_file()
    ):
        parser.error(f"--checkpoint is not a file: {args.checkpoint}")
    if args.backend != "prefix" and args.checkpoint:
        parser.error("--checkpoint is only valid for --backend prefix")
    if args.backend != "prefix" and args.prefix_contribution != "full":
        parser.error(
            "--prefix-contribution off is only valid for --backend prefix"
        )
    if args.max_prompt_tokens < 1 or args.max_new_tokens < 1:
        parser.error("prompt/response token limits must be positive")
    if args.max_history_tokens < 1:
        parser.error("--max-history-tokens must be positive")
    if args.rag_chunk_size < 1:
        parser.error("--rag-chunk-size must be positive")
    if (
        args.rag_chunk_overlap < 0
        or args.rag_chunk_overlap >= args.rag_chunk_size
    ):
        parser.error("--rag-chunk-overlap must be in [0, --rag-chunk-size)")
    if args.rag_top_k < 1:
        parser.error("--rag-top-k must be positive")
    if (
        args.history_retrieval in ("bm25", "plan")
        and args.backend == "prefix"
    ):
        parser.error(
            f"--history-retrieval {args.history_retrieval} is a base/sft "
            "retrieval baseline"
        )
    if (
        args.history_retrieval in ("bm25", "plan")
        and args.history_condition == "query_only"
    ):
        parser.error(
            f"--history-retrieval {args.history_retrieval} requires a history; use "
            "--history-condition query_only as its own baseline"
        )
    if args.history_retrieval == "plan" and args.retrieval_plan is None:
        parser.error("--history-retrieval plan requires --retrieval-plan")
    if args.history_retrieval != "plan" and args.retrieval_plan is not None:
        parser.error("--retrieval-plan is only valid with --history-retrieval plan")
    if args.history_retrieval != "bm25" and (
        args.rag_chunk_size != DEFAULT_RAG_CHUNK_SIZE
        or args.rag_chunk_overlap != DEFAULT_RAG_CHUNK_OVERLAP
        or args.rag_top_k != DEFAULT_RAG_TOP_K
    ):
        parser.error(
            "--rag-* options are only configurable with "
            "--history-retrieval bm25"
        )
    if args.max_items < 0:
        parser.error("--max-items must be non-negative")
    if args.num_swaps < 1:
        parser.error("--num-swaps must be positive")
    if args.swap_index < 0 or args.swap_index >= args.num_swaps:
        parser.error("--swap-index must be in [0, --num-swaps)")
    if args.history_condition != "swapped" and (
        args.num_swaps != 1
        or args.swap_index != 0
        or args.swap_seed != DEFAULT_SWAP_SEED
    ):
        parser.error(
            "--num-swaps/--swap-index/--swap-seed are only configurable with "
            "--history-condition swapped"
        )
    try:
        contiguous_shard(
            (), shard_index=args.shard_index, num_shards=args.num_shards
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not args.enable_thinking:
        parser.error(
            "official Qwen protocol requires thinking; --no-enable-thinking is "
            "reserved for explicit ablation runs"
        )
    if args.output.exists() and not args.resume:
        parser.error(f"--output already exists; use --resume: {args.output}")


def _run_metadata(
    args: argparse.Namespace,
    *,
    total_items: int,
    shard_items: int,
    selected_items: Sequence[OfficialMCQItem],
) -> dict[str, Any]:
    swapped = args.history_condition == "swapped"
    metadata = {
        "protocol": "PersonaMem-v2 official Qwen/VERL text MCQ",
        "prompt_protocol": args.protocol,
        "backend": args.backend,
        "subset": args.subset,
        "history_condition": args.history_condition,
        "swap_seed": args.swap_seed if swapped else None,
        "swap_index": args.swap_index if swapped else None,
        "num_swaps": args.num_swaps if swapped else 0,
        "swap_algorithm": (
            "sha256-seeded canonical persona cycle, cyclic shift index+1"
            if swapped
            else None
        ),
        "model_path": str(args.model_path),
        "tokenizer_path": str(args.tokenizer_path or args.model_path),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "dtype": args.dtype,
        "attn_implementation": args.attn_impl,
        "device": str(args.device),
        "local_files_only": bool(args.local_files_only),
        "benchmark_csv": str(args.benchmark_csv.resolve()),
        "data_root": str(args.data_root.resolve()),
        "clean_manifest": str(args.clean_manifest.resolve()),
        "option_seed_rule": "42 + original zero-based CSV row index",
        "enable_thinking": args.enable_thinking,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "max_history_tokens": args.max_history_tokens,
        "history_truncation": args.history_truncation,
        "max_items": args.max_items,
        "do_sample": False,
        "temperature": 0,
        "top_p": 1.0,
        "missing_box_policy": args.missing_box_policy,
        "python_hash_seed": __import__("os").environ.get("PYTHONHASHSEED"),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "global_subset_items": total_items,
        "shard_items": shard_items,
        "ordered_shard_sample_sha256": selection_digest(selected_items),
    }
    # Keep Base/SFT metadata byte-for-byte schema-compatible with official
    # shards launched before adapter provenance was added.
    if args.backend == "prefix":
        metadata.update(
            {
                "checkpoint_file": prefix_checkpoint_file_identity(
                    args.checkpoint
                ),
                "prefix_contribution": getattr(
                    args, "prefix_contribution", "full"
                ),
                "allow_checkpoint_training_mismatch": bool(
                    getattr(
                        args, "allow_checkpoint_training_mismatch", False
                    )
                ),
            }
        )
    # Keep the default/full metadata byte-for-byte schema-compatible with
    # official shards launched before retrieval baselines were added.
    history_retrieval = getattr(args, "history_retrieval", "full")
    if history_retrieval == "bm25":
        metadata.update(
            {
                "history_retrieval": history_retrieval,
                "rag_chunk_size": args.rag_chunk_size,
                "rag_chunk_overlap": args.rag_chunk_overlap,
                "rag_top_k": args.rag_top_k,
                "rag_chunk_unit": "messages",
                "rag_drop_first_history_system": True,
                "rag_retrieval_query": "raw current user query",
                "rag_prompt_instruction": None,
                "rag_reorder_selected_chunks": "ascending original start_idx",
                "rag_overlap_deduplication": "original message index",
                "bm25_variant": "BM25Okapi",
                "bm25_tokenizer": "unicode word regex + Unicode casefold",
                "bm25_k1": BM25_K1,
                "bm25_b": BM25_B,
                "bm25_epsilon": BM25_EPSILON,
            }
        )
    elif history_retrieval == "plan":
        retrieval_metadata = getattr(args, "_retrieval_plan_metadata", None)
        if not isinstance(retrieval_metadata, Mapping):
            raise ValueError(
                "retrieval plan must be loaded before constructing metadata"
            )
        metadata.update(
            {
                "history_retrieval": history_retrieval,
                **dict(retrieval_metadata),
            }
        )
    return metadata


def validate_resume_state(
    *,
    records: Sequence[Mapping[str, Any]],
    prior_metadata: Mapping[str, Any],
    expected_metadata: Mapping[str, Any],
    items: Sequence[OfficialMCQItem],
    history_personas: Mapping[str, str | None],
    retrieval_plan_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Reject any resume that could combine different experimental cells."""

    for key, expected in expected_metadata.items():
        actual = prior_metadata.get(key)
        if actual != expected:
            raise ValueError(
                f"resume metadata mismatch for {key!r}: "
                f"expected {expected!r}, found {actual!r}"
            )

    selected = {item.sample_id: item for item in items}
    record_ids = [str(record.get("sample_id", "")) for record in records]
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("resume JSONL contains duplicate sample IDs")
    expected_prefix = [item.sample_id for item in items[: len(records)]]
    if record_ids != expected_prefix:
        raise ValueError(
            "resume JSONL must be an ordered prefix of this exact shard"
        )
    for index, record in enumerate(records):
        sample_id = str(record.get("sample_id", ""))
        if not sample_id or sample_id not in selected:
            raise ValueError(
                f"resume record {index} sample {sample_id!r} is not in this shard"
            )
        item = selected[sample_id]
        expected_fields = {
            "backend": expected_metadata["backend"],
            "subset": expected_metadata["subset"],
            "history_condition": expected_metadata["history_condition"],
            "query_persona": item.persona_id,
            "history_persona": history_personas[sample_id],
            "swap_seed": expected_metadata["swap_seed"],
            "swap_index": expected_metadata["swap_index"],
            "num_swaps": expected_metadata["num_swaps"],
            "prefix_contribution": expected_metadata.get(
                "prefix_contribution"
            ),
        }
        history_retrieval = expected_metadata.get("history_retrieval")
        if history_retrieval == "bm25":
            expected_fields.update(
                {
                    "history_retrieval": history_retrieval,
                    "rag_chunk_size": expected_metadata["rag_chunk_size"],
                    "rag_chunk_overlap": expected_metadata[
                        "rag_chunk_overlap"
                    ],
                    "rag_top_k": expected_metadata["rag_top_k"],
                }
            )
        elif history_retrieval == "plan":
            if retrieval_plan_records is None:
                raise ValueError(
                    "plan resume validation requires retrieval plan records"
                )
            try:
                plan_record = retrieval_plan_records[sample_id]
            except KeyError as exc:
                raise ValueError(
                    f"retrieval plan has no row for {sample_id}"
                ) from exc
            expected_fields.update(
                {
                    "history_retrieval": history_retrieval,
                    "retrieval_plan_sha256": expected_metadata[
                        "retrieval_plan_file"
                    ]["sha256"],
                    "retrieval_plan_record_sha256": plan_record_sha256(
                        plan_record
                    ),
                    "retrieval_plan_selection_kind": plan_record[
                        "selection_kind"
                    ],
                    "retrieval_history_sha256": plan_record[
                        "history_sha256"
                    ],
                }
            )
        for key, expected in expected_fields.items():
            if record.get(key) != expected:
                raise ValueError(
                    f"resume record {index} mismatch for {key!r}: "
                    f"expected {expected!r}, found {record.get(key)!r}"
                )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)

    all_items = load_official_items(
        args.benchmark_csv,
        subset=args.subset,
        clean_manifest=args.clean_manifest,
        protocol=args.protocol,
    )
    expected_global = EXPECTED_SUBSET_SIZES[args.subset]
    if len(all_items) != expected_global and not args.allow_subset_size_mismatch:
        parser.error(
            f"{args.subset} must contain {expected_global} rows, found {len(all_items)}; "
            "dataset snapshot or clean manifest differs"
        )
    histories = persona_history_items(all_items)
    swap_mapping: Mapping[str, str] | None = None
    if args.history_condition == "swapped":
        try:
            swap_mapping = make_history_derangements(
                tuple(histories),
                num_swaps=args.num_swaps,
                seed=args.swap_seed,
            )[args.swap_index]
        except ValueError as exc:
            parser.error(str(exc))
    items = contiguous_shard(
        all_items, shard_index=args.shard_index, num_shards=args.num_shards
    )
    if args.max_items:
        items = items[: args.max_items]

    source_items = {
        item.sample_id: history_source_item(
            item,
            condition=args.history_condition,
            histories=histories,
            swap_mapping=swap_mapping,
        )
        for item in items
    }
    history_personas = {
        sample_id: source.persona_id if source is not None else None
        for sample_id, source in source_items.items()
    }
    retrieval_plan: RetrievalPlan | None = None
    if args.history_retrieval == "plan":
        try:
            retrieval_plan = load_retrieval_plan(
                args.retrieval_plan,
                items=all_items,
                benchmark_csv=args.benchmark_csv.resolve(),
                subset=args.subset,
                prompt_protocol=args.protocol,
            )
        except ValueError as exc:
            parser.error(str(exc))
        args._retrieval_plan_metadata = plan_metadata(retrieval_plan)

    meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    metadata = _run_metadata(
        args,
        total_items=len(all_items),
        shard_items=len(items),
        selected_items=items,
    )
    previous = read_jsonl(args.output) if args.resume else []
    prior_metadata: dict[str, Any] | None = None
    if args.resume:
        if args.output.exists() != meta_path.exists():
            parser.error(
                "resume requires the JSONL and its .meta.json sidecar to either "
                "both exist or both be absent"
            )
        if meta_path.exists():
            try:
                prior_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                parser.error(f"invalid resume metadata JSON: {exc}")
            try:
                validate_resume_state(
                    records=previous,
                    prior_metadata=prior_metadata,
                    expected_metadata=metadata,
                    items=items,
                    history_personas=history_personas,
                    retrieval_plan_records=(
                        retrieval_plan.records
                        if retrieval_plan is not None
                        else None
                    ),
                )
            except ValueError as exc:
                parser.error(str(exc))
    completed = {str(record["sample_id"]) for record in previous}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if prior_metadata is None:
        meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(
        f"[official] subset={args.subset} global={len(all_items)} "
        f"shard={args.shard_index}/{args.num_shards} n={len(items)} "
        f"resume={len(previous)} backend={args.backend} "
        f"history={args.history_condition} "
        f"retrieval={args.history_retrieval} "
        f"prefix_contribution={args.prefix_contribution}",
        flush=True,
    )
    (
        model,
        tokenizer,
        prefix_config,
        prefix_checkpoint_manifest,
    ) = load_model_and_tokenizer(args)
    if prefix_config is not None:
        try:
            assert prefix_checkpoint_manifest is not None
            validate_prefix_checkpoint_contract(
                args, prefix_checkpoint_manifest
            )
            configure_prefix_contribution(
                model, prefix_config, args.prefix_contribution
            )
        except ValueError as exc:
            parser.error(str(exc))
        metadata["prefix_config"] = asdict(prefix_config)
        metadata["prefix_checkpoint_manifest"] = prefix_checkpoint_manifest
        metadata["adapter_parameter_count"] = prefix_checkpoint_manifest[
            "adapter_parameter_count"
        ]
        metadata["model_parameter_count_total"] = sum(
            parameter.numel() for parameter in model.parameters()
        )
        metadata["backbone_parameter_count"] = (
            metadata["model_parameter_count_total"]
            - metadata["adapter_parameter_count"]
        )
        if prior_metadata is not None:
            prior_prefix_config = prior_metadata.get("prefix_config")
            if prior_prefix_config is None and previous:
                parser.error(
                    "resume metadata with existing prefix records lacks "
                    "prefix_config"
                )
            if (
                prior_prefix_config is not None
                and prior_prefix_config != metadata["prefix_config"]
            ):
                parser.error(
                    "resume prefix checkpoint architecture metadata changed"
                )
            prior_manifest = prior_metadata.get(
                "prefix_checkpoint_manifest"
            )
            if prior_manifest is None and previous:
                parser.error(
                    "resume metadata with existing prefix records lacks "
                    "prefix_checkpoint_manifest"
                )
            if (
                prior_manifest is not None
                and prior_manifest != metadata["prefix_checkpoint_manifest"]
            ):
                parser.error(
                    "resume prefix checkpoint provenance metadata changed"
                )
        meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.backend == "prefix" and args.history_condition == "query_only":
        from deltamem.core.prefix_steer import clear_frozen_memory

        clear_frozen_memory(model)

    loaded_history_persona: str | None = None
    history_messages: list[dict[str, Any]] = []
    history_path: Path | None = None
    history_tokens_written: int | None = None
    pending = [item for item in items if item.sample_id not in completed]
    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as output_handle:
        for progress, item in enumerate(pending, start=1):
            source_item = source_items[item.sample_id]
            source_persona = (
                source_item.persona_id if source_item is not None else None
            )
            if source_item is None:
                history_messages = []
                history_path = None
                history_tokens_written = None
            elif source_persona != loaded_history_persona:
                history_messages, history_path = load_history_messages(
                    source_item,
                    benchmark_csv=args.benchmark_csv.resolve(),
                    data_root=args.data_root.resolve(),
                )
                loaded_history_persona = source_persona
                history_tokens_written = None
                if args.backend == "prefix":
                    history_tokens_written = write_prefix_history(
                        model,
                        tokenizer,
                        history_messages,
                        persona_id=source_persona,
                        device=args.device,
                        max_history_tokens=args.max_history_tokens,
                        history_truncation=args.history_truncation,
                    )

            retrieval_audit: dict[str, Any] | None = None
            prompt_history_messages = history_messages
            if source_item is not None and args.history_retrieval == "bm25":
                prompt_history_messages, retrieval_audit = retrieve_bm25_history(
                    history_messages,
                    item.query,
                    chunk_size=args.rag_chunk_size,
                    chunk_overlap=args.rag_chunk_overlap,
                    top_k=args.rag_top_k,
                )
            elif source_item is not None and args.history_retrieval == "plan":
                assert retrieval_plan is not None
                try:
                    prompt_history_messages, retrieval_audit = (
                        render_plan_selection(
                            retrieval_plan.records[item.sample_id],
                            item=item,
                            history_persona=source_persona,
                            history_messages=history_messages,
                        )
                    )
                except ValueError as exc:
                    raise RuntimeError(
                        f"{item.sample_id}: invalid retrieval plan selection: {exc}"
                    ) from exc
            prompt_messages = official_messages(
                item,
                () if args.backend == "prefix" else prompt_history_messages,
                protocol=args.protocol,
            )
            prompt_ids = encode_official_prompt(
                tokenizer,
                prompt_messages,
                enable_thinking=args.enable_thinking,
            )
            if len(prompt_ids) > args.max_prompt_tokens:
                raise RuntimeError(
                    f"{item.sample_id}: prompt has {len(prompt_ids)} tokens, exceeds "
                    f"official cap {args.max_prompt_tokens}; do not silently truncate"
                )
            response, response_tokens = greedy_generate(
                model,
                tokenizer,
                prompt_ids,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                prefix_memory=args.backend == "prefix",
            )
            prediction, is_correct, fallback_used = score_response(
                response,
                item,
                missing_box_policy=args.missing_box_policy,
                protocol=args.protocol,
            )
            record = {
                "backend": args.backend,
                "subset": args.subset,
                "row_index": item.row_index,
                "sample_id": item.sample_id,
                "persona_id": item.persona_id,
                "query_persona": item.persona_id,
                "history_persona": source_persona,
                "history_condition": args.history_condition,
                "prefix_contribution": (
                    args.prefix_contribution
                    if args.backend == "prefix"
                    else None
                ),
                "swap_seed": (
                    args.swap_seed
                    if args.history_condition == "swapped"
                    else None
                ),
                "swap_index": (
                    args.swap_index
                    if args.history_condition == "swapped"
                    else None
                ),
                "num_swaps": (
                    args.num_swaps
                    if args.history_condition == "swapped"
                    else 0
                ),
                "history_path": (
                    str(history_path) if history_path is not None else None
                ),
                "history_tokens_written": history_tokens_written,
                "prompt_tokens": len(prompt_ids),
                "response_tokens": response_tokens,
                "correct_letter": item.correct_letter,
                "predicted_letter": prediction,
                "is_correct": bool(is_correct),
                "embedding_fallback_used": fallback_used,
                "response": response,
                "tags": item.tags,
            }
            if args.history_retrieval == "bm25":
                record.update(
                    {
                        "history_retrieval": args.history_retrieval,
                        "rag_chunk_size": args.rag_chunk_size,
                        "rag_chunk_overlap": args.rag_chunk_overlap,
                        "rag_top_k": args.rag_top_k,
                        "rag_audit": retrieval_audit,
                    }
                )
            elif args.history_retrieval == "plan":
                assert retrieval_plan is not None
                plan_record = retrieval_plan.records[item.sample_id]
                record.update(
                    {
                        "history_retrieval": args.history_retrieval,
                        "retrieval_plan_sha256": retrieval_plan.file_identity[
                            "sha256"
                        ],
                        "retrieval_plan_record_sha256": plan_record_sha256(
                            plan_record
                        ),
                        "retrieval_plan_selection_kind": plan_record[
                            "selection_kind"
                        ],
                        "retrieval_history_sha256": plan_record[
                            "history_sha256"
                        ],
                        "retrieval_plan_audit": retrieval_audit,
                    }
                )
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_handle.flush()
            previous.append(record)
            if progress == 1 or progress % 10 == 0 or progress == len(pending):
                running = summarize_records(previous)
                print(
                    f"[official] {progress}/{len(pending)} sample={item.sample_id} "
                    f"acc={running['accuracy']:.4f} "
                    f"boxed={running['boxed_format_rate']:.4f}",
                    flush=True,
                )

    expected_shard = len(items)
    summary = {
        "metadata": metadata,
        **summarize_records(previous, expected=expected_shard),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[official] done n={summary['n']} accuracy={summary['accuracy']:.4f} "
        f"-> {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
