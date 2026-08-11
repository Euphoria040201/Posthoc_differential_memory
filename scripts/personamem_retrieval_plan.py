#!/usr/bin/env python3
"""Immutable, content-addressed retrieval plans for PersonaMem-v2.

A retrieval plan is deliberately model-agnostic.  One JSONL record is keyed by
the benchmark ``sample_id`` and contains either:

* monotonically increasing indices into the *raw* history message list; or
* memory facts, rendered by the evaluator with one fixed serialization.

The adjacent ``.manifest.json`` binds the plan to the exact benchmark file,
ordered sample set, prompt protocol, and retriever provenance.  Per-record
hashes bind every selection to the raw user query, shuffled MCQ sample, and
source history.  The retrieval input is always the raw current user query;
answer options and labels are neither serialized nor accepted as plan fields.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PLAN_SCHEMA = "personamem_retrieval_plan_v1"
PLAN_MANIFEST_SCHEMA = "personamem_retrieval_plan_manifest_v1"
SELECTION_KINDS = ("message_indices", "facts")
RETRIEVAL_INPUT = "raw_current_user_query_only"
FACTS_PROMPT_HEADING = "Retrieved memory facts:"
MANIFEST_SUFFIX = ".manifest.json"

_COMMON_RECORD_KEYS = frozenset(
    {
        "schema",
        "sample_id",
        "sample_sha256",
        "query_sha256",
        "retrieval_input_sha256",
        "history_persona",
        "history_sha256",
        "history_message_count",
        "selection_kind",
    }
)
_SELECTION_KEYS = {
    "message_indices": frozenset({"message_indices"}),
    "facts": frozenset({"facts"}),
}
_FORBIDDEN_KEY_PARTS = (
    "answer",
    "correct",
    "gold",
    "label",
    "option",
    "prediction",
)
_EXPLICIT_LABEL_LEAK = re.compile(
    r"(?:\\boxed\s*\{\s*\(?[a-dA-D]\)?\s*\}"
    r"|(?:correct|gold|final)\s+(?:answer|label)\s*(?:is|:)\s*\(?[a-dA-D]\)?"
    r"|answer\s*:\s*\(?[a-dA-D]\)?)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    records: dict[str, dict[str, Any]]
    file_identity: dict[str, Any]
    manifest_identity: dict[str, Any]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    artifact = Path(path)
    return {
        "resolved_path": str(artifact.resolve()),
        "sha256": file_sha256(artifact),
        "size_bytes": artifact.stat().st_size,
    }


def manifest_path_for(plan_path: str | Path) -> Path:
    return Path(str(Path(plan_path)) + MANIFEST_SUFFIX)


def query_sha256(query: str) -> str:
    return hashlib.sha256(str(query).encode("utf-8")).hexdigest()


def sample_payload(item: Any) -> dict[str, Any]:
    """Canonical benchmark row payload without exposing it in plan JSONL."""

    return {
        "row_index": int(item.row_index),
        "sample_id": str(item.sample_id),
        "persona_id": str(item.persona_id),
        "history_link": str(item.history_link),
        "query": str(item.query),
        "options": [str(value) for value in item.options],
        "correct_index": int(item.correct_index),
    }


def sample_sha256(item: Any) -> str:
    return canonical_sha256(sample_payload(item))


def dataset_sha256(items: Sequence[Any]) -> str:
    """Bind order and every benchmark field that can affect evaluation."""

    return canonical_sha256(
        {
            "sample_count": len(items),
            "ordered_sample_sha256": [
                sample_sha256(item) for item in items
            ],
        }
    )


def history_sha256(messages: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256([dict(message) for message in messages])


def plan_record_sha256(record: Mapping[str, Any]) -> str:
    return canonical_sha256(dict(record))


def build_plan_record(
    *,
    item: Any,
    history_persona: str,
    history_messages: Sequence[Mapping[str, Any]],
    message_indices: Sequence[int] | None = None,
    facts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build and validate one plan row.

    Exactly one of ``message_indices`` and ``facts`` must be supplied.
    """

    supplied = int(message_indices is not None) + int(facts is not None)
    if supplied != 1:
        raise ValueError("plan row needs exactly one selection representation")
    selection_kind = (
        "message_indices" if message_indices is not None else "facts"
    )
    record: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "sample_id": str(item.sample_id),
        "sample_sha256": sample_sha256(item),
        "query_sha256": query_sha256(item.query),
        "retrieval_input_sha256": query_sha256(item.query),
        "history_persona": str(history_persona),
        "history_sha256": history_sha256(history_messages),
        "history_message_count": len(history_messages),
        "selection_kind": selection_kind,
    }
    if message_indices is not None:
        record["message_indices"] = [int(value) for value in message_indices]
    else:
        record["facts"] = [str(value) for value in facts or ()]
    validate_plan_record(
        record,
        item=item,
        history_persona=history_persona,
        history_messages=history_messages,
    )
    return record


def _validate_fact_text(text: Any, *, sample_id: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{sample_id}: every retrieved fact must be non-empty text")
    if _EXPLICIT_LABEL_LEAK.search(text):
        raise ValueError(
            f"{sample_id}: retrieved fact contains an explicit MCQ answer label"
        )
    return text


def validate_plan_record(
    record: Mapping[str, Any],
    *,
    item: Any,
    history_persona: str | None = None,
    history_messages: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Strictly validate sample/query provenance and, when supplied, history."""

    sample_id = str(item.sample_id)
    if not isinstance(record, Mapping):
        raise ValueError(f"{sample_id}: retrieval plan record must be an object")
    selection_kind = record.get("selection_kind")
    if selection_kind not in SELECTION_KINDS:
        raise ValueError(
            f"{sample_id}: unknown retrieval selection_kind {selection_kind!r}"
        )
    allowed = _COMMON_RECORD_KEYS | _SELECTION_KEYS[str(selection_kind)]
    extra = set(record) - allowed
    if extra:
        leakage_keys = sorted(
            key
            for key in extra
            if any(part in key.casefold() for part in _FORBIDDEN_KEY_PARTS)
        )
        if leakage_keys:
            raise ValueError(
                f"{sample_id}: option/label leakage fields are forbidden: "
                f"{leakage_keys}"
            )
        raise ValueError(
            f"{sample_id}: unsupported retrieval plan fields: {sorted(extra)}"
        )
    missing = allowed - set(record)
    if missing:
        raise ValueError(
            f"{sample_id}: retrieval plan fields are missing: {sorted(missing)}"
        )
    if record.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"{sample_id}: retrieval plan schema mismatch")
    if record.get("sample_id") != sample_id:
        raise ValueError(
            f"{sample_id}: plan is keyed by {record.get('sample_id')!r}"
        )
    if record.get("sample_sha256") != sample_sha256(item):
        raise ValueError(f"{sample_id}: benchmark sample hash mismatch")
    expected_query_hash = query_sha256(item.query)
    if record.get("query_sha256") != expected_query_hash:
        raise ValueError(f"{sample_id}: raw query hash mismatch")
    if record.get("retrieval_input_sha256") != expected_query_hash:
        raise ValueError(
            f"{sample_id}: retrieval input is not the raw query alone"
        )
    if history_persona is not None and (
        str(record.get("history_persona")) != str(history_persona)
    ):
        raise ValueError(
            f"{sample_id}: history persona mismatch: expected "
            f"{history_persona!r}, found {record.get('history_persona')!r}"
        )
    try:
        history_message_count = int(record.get("history_message_count"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{sample_id}: invalid history_message_count") from exc
    if history_message_count < 1:
        raise ValueError(f"{sample_id}: history_message_count must be positive")
    if history_messages is not None:
        if history_message_count != len(history_messages):
            raise ValueError(f"{sample_id}: history message count mismatch")
        if record.get("history_sha256") != history_sha256(history_messages):
            raise ValueError(f"{sample_id}: source history hash mismatch")
    else:
        history_hash = record.get("history_sha256")
        if (
            not isinstance(history_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", history_hash)
        ):
            raise ValueError(f"{sample_id}: invalid source history SHA256")

    if selection_kind == "message_indices":
        indices = record.get("message_indices")
        if (
            not isinstance(indices, list)
            or not indices
            or any(type(index) is not int for index in indices)
        ):
            raise ValueError(
                f"{sample_id}: message_indices must be a non-empty integer list"
            )
        if indices != sorted(set(indices)):
            raise ValueError(
                f"{sample_id}: message_indices must be unique and chronological"
            )
        if indices[0] < 0 or indices[-1] >= history_message_count:
            raise ValueError(
                f"{sample_id}: message index is outside the raw history"
            )
    else:
        facts = record.get("facts")
        if not isinstance(facts, list) or not facts:
            raise ValueError(f"{sample_id}: facts must be a non-empty text list")
        for fact in facts:
            _validate_fact_text(fact, sample_id=sample_id)


def render_plan_selection(
    record: Mapping[str, Any],
    *,
    item: Any,
    history_persona: str,
    history_messages: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate and turn one plan row into evaluator history messages."""

    validate_plan_record(
        record,
        item=item,
        history_persona=history_persona,
        history_messages=history_messages,
    )
    kind = str(record["selection_kind"])
    if kind == "message_indices":
        indices = [int(value) for value in record["message_indices"]]
        messages = [dict(history_messages[index]) for index in indices]
        audit = {
            "selection_kind": kind,
            "retrieved_message_indices": indices,
            "retrieved_messages": len(messages),
            "source_history_sha256": record["history_sha256"],
            "plan_record_sha256": plan_record_sha256(record),
        }
        return messages, audit

    facts = [str(value) for value in record["facts"]]
    rendered = {
        "role": "system",
        "content": FACTS_PROMPT_HEADING
        + "\n"
        + "\n".join(f"- {fact}" for fact in facts),
    }
    audit = {
        "selection_kind": kind,
        "retrieved_facts": len(facts),
        "fact_rendering": "single system message; fixed heading and dash list",
        "source_history_sha256": record["history_sha256"],
        "plan_record_sha256": plan_record_sha256(record),
    }
    return [rendered], audit


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing retrieval plan manifest: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def _read_plan_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"retrieval plan is not a file: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected one object")
            records.append(value)
    return records


def load_retrieval_plan(
    plan_path: str | Path,
    *,
    items: Sequence[Any],
    benchmark_csv: str | Path,
    subset: str,
    prompt_protocol: str,
) -> RetrievalPlan:
    """Load a complete plan and bind it to the exact benchmark selection."""

    path = Path(plan_path)
    manifest_path = manifest_path_for(path)
    manifest = _read_json_object(manifest_path)
    identity = file_identity(path)
    manifest_identity = file_identity(manifest_path)

    required_manifest = {
        "schema",
        "plan_file",
        "benchmark_file",
        "subset",
        "prompt_protocol",
        "dataset_sha256",
        "ordered_sample_id_sha256",
        "sample_count",
        "selection_kind",
        "retrieval_input",
        "history_hash",
        "sample_hash",
        "retriever",
    }
    missing = required_manifest - set(manifest)
    if missing:
        raise ValueError(
            f"{manifest_path}: required fields are missing: {sorted(missing)}"
        )
    if manifest.get("schema") != PLAN_MANIFEST_SCHEMA:
        raise ValueError(f"{manifest_path}: manifest schema mismatch")
    if manifest.get("plan_file") != identity:
        raise ValueError(f"{manifest_path}: plan file identity mismatch")
    benchmark_identity = file_identity(benchmark_csv)
    if manifest.get("benchmark_file") != benchmark_identity:
        raise ValueError(f"{manifest_path}: benchmark file hash mismatch")
    if manifest.get("subset") != subset:
        raise ValueError(f"{manifest_path}: subset mismatch")
    if manifest.get("prompt_protocol") != prompt_protocol:
        raise ValueError(f"{manifest_path}: prompt protocol mismatch")
    expected_dataset_hash = dataset_sha256(items)
    if manifest.get("dataset_sha256") != expected_dataset_hash:
        raise ValueError(f"{manifest_path}: dataset hash mismatch")
    ordered_ids_hash = canonical_sha256(
        [str(item.sample_id) for item in items]
    )
    if manifest.get("ordered_sample_id_sha256") != ordered_ids_hash:
        raise ValueError(f"{manifest_path}: ordered sample hash mismatch")
    if manifest.get("sample_count") != len(items):
        raise ValueError(f"{manifest_path}: sample count mismatch")
    if manifest.get("selection_kind") not in SELECTION_KINDS:
        raise ValueError(f"{manifest_path}: invalid selection_kind")
    if manifest.get("retrieval_input") != RETRIEVAL_INPUT:
        raise ValueError(
            f"{manifest_path}: options/labels are forbidden retrieval inputs"
        )
    if manifest.get("history_hash") != "canonical_json_sha256(raw_messages)":
        raise ValueError(f"{manifest_path}: unsupported history hash contract")
    if manifest.get("sample_hash") != (
        "canonical_json_sha256(row_index,sample_id,persona_id,history_link,"
        "query,shuffled_options,correct_index)"
    ):
        raise ValueError(f"{manifest_path}: unsupported sample hash contract")

    raw_records = _read_plan_jsonl(path)
    if len(raw_records) != len(items):
        raise ValueError(
            f"{path}: expected {len(items)} records, found {len(raw_records)}"
        )
    by_id: dict[str, dict[str, Any]] = {}
    item_by_id = {str(item.sample_id): item for item in items}
    for record in raw_records:
        sample_id = str(record.get("sample_id", ""))
        if not sample_id or sample_id not in item_by_id:
            raise ValueError(f"{path}: unknown sample_id {sample_id!r}")
        if sample_id in by_id:
            raise ValueError(f"{path}: duplicate sample_id {sample_id!r}")
        item = item_by_id[sample_id]
        validate_plan_record(record, item=item)
        if record.get("selection_kind") != manifest.get("selection_kind"):
            raise ValueError(
                f"{sample_id}: record selection kind differs from manifest"
            )
        by_id[sample_id] = dict(record)
    expected_order = [str(item.sample_id) for item in items]
    actual_order = [str(record["sample_id"]) for record in raw_records]
    if actual_order != expected_order:
        raise ValueError(f"{path}: records are not in exact benchmark order")
    return RetrievalPlan(
        path=path.resolve(),
        manifest_path=manifest_path.resolve(),
        manifest=manifest,
        records=by_id,
        file_identity=identity,
        manifest_identity=manifest_identity,
    )


def plan_metadata(plan: RetrievalPlan) -> dict[str, Any]:
    """Compact provenance copied into every evaluator shard sidecar."""

    return {
        "retrieval_plan_file": plan.file_identity,
        "retrieval_plan_manifest_file": plan.manifest_identity,
        "retrieval_plan_schema": plan.manifest["schema"],
        "retrieval_plan_dataset_sha256": plan.manifest["dataset_sha256"],
        "retrieval_plan_selection_kind": plan.manifest["selection_kind"],
        "retrieval_plan_retriever": plan.manifest["retriever"],
        "retrieval_plan_retrieval_input": plan.manifest["retrieval_input"],
        "retrieval_plan_fact_rendering": (
            "single system message; fixed heading and dash list"
            if plan.manifest["selection_kind"] == "facts"
            else None
        ),
    }


def write_retrieval_plan(
    plan_path: str | Path,
    *,
    items: Sequence[Any],
    records: Sequence[Mapping[str, Any]],
    benchmark_csv: str | Path,
    subset: str,
    prompt_protocol: str,
    selection_kind: str,
    retriever: Mapping[str, Any],
) -> RetrievalPlan:
    """Create, never overwrite, then reload one immutable plan + manifest."""

    path = Path(plan_path)
    manifest_path = manifest_path_for(path)
    if path.exists() or manifest_path.exists():
        existing = [str(value) for value in (path, manifest_path) if value.exists()]
        raise ValueError(
            "refusing to overwrite retrieval plan artifact: "
            + ", ".join(existing)
        )
    if selection_kind not in SELECTION_KINDS:
        raise ValueError(f"unknown selection_kind {selection_kind!r}")
    if len(records) != len(items):
        raise ValueError("plan must contain exactly one row per benchmark sample")
    for item, record in zip(items, records):
        validate_plan_record(record, item=item)
        if record.get("selection_kind") != selection_kind:
            raise ValueError("record selection kind differs from plan")
        if record.get("sample_id") != item.sample_id:
            raise ValueError("plan records must be in exact benchmark order")

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            for record in records:
                handle.write(canonical_json_bytes(dict(record)).decode("utf-8"))
                handle.write("\n")
        identity = file_identity(path)
        manifest = {
            "schema": PLAN_MANIFEST_SCHEMA,
            "plan_file": identity,
            "benchmark_file": file_identity(benchmark_csv),
            "subset": subset,
            "prompt_protocol": prompt_protocol,
            "dataset_sha256": dataset_sha256(items),
            "ordered_sample_id_sha256": canonical_sha256(
                [str(item.sample_id) for item in items]
            ),
            "sample_count": len(items),
            "selection_kind": selection_kind,
            "retrieval_input": RETRIEVAL_INPUT,
            "history_hash": "canonical_json_sha256(raw_messages)",
            "sample_hash": (
                "canonical_json_sha256(row_index,sample_id,persona_id,"
                "history_link,query,shuffled_options,correct_index)"
            ),
            "retriever": dict(retriever),
        }
        # ``x`` plus embedded content hashes makes accidental replacement or
        # cross-run reuse fail closed in the evaluator and merger.
        with manifest_path.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
                + "\n"
            )
    except BaseException:
        # Never silently delete an existing artifact.  Only clean the plan file
        # that this call just created if manifest creation did not complete.
        if path.exists() and not manifest_path.exists():
            try:
                os.unlink(path)
            except OSError:
                pass
        raise
    return load_retrieval_plan(
        path,
        items=items,
        benchmark_csv=benchmark_csv,
        subset=subset,
        prompt_protocol=prompt_protocol,
    )

