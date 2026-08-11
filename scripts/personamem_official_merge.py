#!/usr/bin/env python3
"""Safely merge complete PersonaMem-v2 official-evaluation JSONL shards.

The merger is intentionally strict.  It refuses missing or duplicate shard
indices, duplicate/missing samples, incomplete shard summaries, diagnostic
``--max-items`` runs, and any mismatch in model, checkpoint, prompt, generation,
subset, or causal-history configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.personamem_official_hf_eval import (
    contiguous_shard,
    file_sha256,
    history_source_item,
    load_official_items,
    make_history_derangements,
    persona_history_items,
    read_jsonl,
    selection_digest,
    summarize_records,
    validate_resume_state,
)
from scripts.personamem_retrieval_plan import (
    file_identity as retrieval_file_identity,
    load_retrieval_plan,
    plan_metadata as retrieval_plan_metadata,
)


# Placement and shard-selection fields may differ without changing the
# experimental cell.  Every other metadata field must match exactly.
SHARD_LOCAL_METADATA = frozenset(
    {
        "device",
        "shard_index",
        "shard_items",
        "ordered_shard_sample_sha256",
    }
)


def sidecar_path(path: Path, kind: str) -> Path:
    return path.with_suffix(path.suffix + f".{kind}.json")


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"missing required sidecar: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def comparable_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in SHARD_LOCAL_METADATA
    }


def validate_checkpoint_artifact(metadata: Mapping[str, Any]) -> None:
    """Require the exact adapter artifact used by every prefix shard."""

    if metadata.get("backend") != "prefix":
        return
    identity = metadata.get("checkpoint_file")
    manifest = metadata.get("prefix_checkpoint_manifest")
    if not isinstance(identity, Mapping) or not isinstance(manifest, Mapping):
        raise ValueError(
            "prefix paper shards require checkpoint_file and "
            "prefix_checkpoint_manifest provenance"
        )
    if {
        key: manifest.get(key)
        for key in ("resolved_path", "sha256", "size_bytes")
    } != {
        key: identity.get(key)
        for key in ("resolved_path", "sha256", "size_bytes")
    }:
        raise ValueError(
            "prefix checkpoint identity differs between metadata manifests"
        )
    checkpoint_path = Path(str(identity.get("resolved_path", "")))
    if not checkpoint_path.is_file():
        raise ValueError(
            f"prefix checkpoint artifact is missing: {checkpoint_path}"
        )
    if checkpoint_path.stat().st_size != identity.get("size_bytes"):
        raise ValueError(
            f"prefix checkpoint size changed since evaluation: {checkpoint_path}"
        )
    if file_sha256(checkpoint_path) != identity.get("sha256"):
        raise ValueError(
            f"prefix checkpoint SHA256 changed since evaluation: {checkpoint_path}"
        )
    if metadata.get("adapter_parameter_count") != manifest.get(
        "adapter_parameter_count"
    ):
        raise ValueError(
            "adapter parameter count differs between metadata manifests"
        )


def validate_retrieval_plan_artifact(metadata: Mapping[str, Any]) -> None:
    """Require an unchanged plan and manifest for precomputed retrieval shards."""

    if metadata.get("history_retrieval") != "plan":
        return
    plan_identity = metadata.get("retrieval_plan_file")
    manifest_identity = metadata.get("retrieval_plan_manifest_file")
    if not isinstance(plan_identity, Mapping) or not isinstance(
        manifest_identity, Mapping
    ):
        raise ValueError(
            "plan retrieval shards require plan and manifest provenance"
        )
    for name, identity in (
        ("retrieval plan", plan_identity),
        ("retrieval plan manifest", manifest_identity),
    ):
        artifact = Path(str(identity.get("resolved_path", "")))
        if not artifact.is_file():
            raise ValueError(f"{name} artifact is missing: {artifact}")
        if retrieval_file_identity(artifact) != dict(identity):
            raise ValueError(f"{name} artifact changed since evaluation: {artifact}")


def merge_shards(
    inputs: Sequence[Path],
    *,
    output: Path,
) -> dict[str, Any]:
    """Validate and merge one complete set of official shards."""

    paths = tuple(Path(path) for path in inputs)
    if not paths:
        raise ValueError("at least one input shard is required")
    if len(set(path.resolve() for path in paths)) != len(paths):
        raise ValueError("the same input shard was provided more than once")
    output_sidecars = (
        output,
        sidecar_path(output, "meta"),
        sidecar_path(output, "summary"),
    )
    existing = [path for path in output_sidecars if path.exists()]
    if existing:
        raise ValueError(
            "refusing to overwrite existing merge outputs: "
            + ", ".join(str(path) for path in existing)
        )

    loaded: list[
        tuple[Path, dict[str, Any], dict[str, Any], list[dict[str, Any]]]
    ] = []
    for path in paths:
        if not path.exists():
            raise ValueError(f"missing input shard: {path}")
        metadata = _read_json_object(sidecar_path(path, "meta"))
        summary = _read_json_object(sidecar_path(path, "summary"))
        records = read_jsonl(path)
        if summary.get("metadata") != metadata:
            raise ValueError(f"{path}: summary metadata differs from .meta.json")
        if not bool(summary.get("complete")):
            raise ValueError(f"{path}: shard summary is incomplete")
        if summary.get("n") != len(records):
            raise ValueError(
                f"{path}: summary n={summary.get('n')!r} but JSONL has "
                f"{len(records)} records"
            )
        if metadata.get("shard_items") != len(records):
            raise ValueError(
                f"{path}: metadata shard_items={metadata.get('shard_items')!r} "
                f"but JSONL has {len(records)} records"
            )
        if int(metadata.get("max_items", 0)) != 0:
            raise ValueError(
                f"{path}: diagnostic max_items runs cannot be paper-shard merged"
            )
        validate_checkpoint_artifact(metadata)
        validate_retrieval_plan_artifact(metadata)
        loaded.append((path, metadata, summary, records))

    reference = comparable_metadata(loaded[0][1])
    for path, metadata, _, _ in loaded[1:]:
        if comparable_metadata(metadata) != reference:
            raise ValueError(
                f"{path}: experimental configuration differs from the first shard"
            )

    num_shards = int(loaded[0][1].get("num_shards", 0))
    if num_shards < 1:
        raise ValueError("metadata num_shards must be positive")
    by_index: dict[
        int, tuple[Path, dict[str, Any], dict[str, Any], list[dict[str, Any]]]
    ] = {}
    for entry in loaded:
        path, metadata, _, _ = entry
        try:
            shard_index = int(metadata["shard_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: invalid shard_index") from exc
        if shard_index in by_index:
            raise ValueError(f"duplicate shard index {shard_index}")
        by_index[shard_index] = entry
    expected_indices = set(range(num_shards))
    actual_indices = set(by_index)
    if actual_indices != expected_indices:
        missing = sorted(expected_indices - actual_indices)
        extra = sorted(actual_indices - expected_indices)
        raise ValueError(f"shard set mismatch: missing={missing}, extra={extra}")

    metadata0 = loaded[0][1]
    all_items = load_official_items(
        metadata0["benchmark_csv"],
        subset=metadata0["subset"],
        clean_manifest=metadata0["clean_manifest"],
        protocol=metadata0["prompt_protocol"],
    )
    if len(all_items) != int(metadata0["global_subset_items"]):
        raise ValueError(
            "dataset now differs from shard metadata global_subset_items"
        )
    histories = persona_history_items(all_items)
    condition = str(metadata0["history_condition"])
    retrieval_plan = None
    if metadata0.get("history_retrieval") == "plan":
        plan_identity = metadata0["retrieval_plan_file"]
        retrieval_plan = load_retrieval_plan(
            plan_identity["resolved_path"],
            items=all_items,
            benchmark_csv=metadata0["benchmark_csv"],
            subset=metadata0["subset"],
            prompt_protocol=metadata0["prompt_protocol"],
        )
        expected_plan_metadata = retrieval_plan_metadata(retrieval_plan)
        actual_plan_metadata = {
            key: metadata0.get(key) for key in expected_plan_metadata
        }
        if actual_plan_metadata != expected_plan_metadata:
            raise ValueError(
                "retrieval plan metadata differs from the immutable artifacts"
            )
    swap_mapping: Mapping[str, str] | None = None
    if condition == "swapped":
        swap_mapping = make_history_derangements(
            tuple(histories),
            num_swaps=int(metadata0["num_swaps"]),
            seed=int(metadata0["swap_seed"]),
        )[int(metadata0["swap_index"])]

    merged_records: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    for shard_index in range(num_shards):
        path, metadata, _, records = by_index[shard_index]
        expected_items = contiguous_shard(
            all_items, shard_index=shard_index, num_shards=num_shards
        )
        expected_ids = [item.sample_id for item in expected_items]
        actual_ids = [str(record.get("sample_id", "")) for record in records]
        if actual_ids != expected_ids:
            raise ValueError(
                f"{path}: sample order/content is not the exact official shard"
            )
        if metadata.get("ordered_shard_sample_sha256") != selection_digest(
            expected_items
        ):
            raise ValueError(f"{path}: ordered shard sample digest mismatch")
        history_personas = {
            item.sample_id: (
                source.persona_id
                if (
                    source := history_source_item(
                        item,
                        condition=condition,
                        histories=histories,
                        swap_mapping=swap_mapping,
                    )
                )
                is not None
                else None
            )
            for item in expected_items
        }
        validate_resume_state(
            records=records,
            prior_metadata=metadata,
            expected_metadata=metadata,
            items=expected_items,
            history_personas=history_personas,
            retrieval_plan_records=(
                retrieval_plan.records
                if retrieval_plan is not None
                else None
            ),
        )
        overlap = seen_samples.intersection(actual_ids)
        if overlap:
            raise ValueError(
                f"{path}: duplicate samples across shards: {sorted(overlap)[:3]}"
            )
        seen_samples.update(actual_ids)
        merged_records.extend(records)

    expected_all_ids = [item.sample_id for item in all_items]
    if [str(record["sample_id"]) for record in merged_records] != expected_all_ids:
        raise ValueError("merged samples do not exactly cover the global subset")

    merged_metadata = dict(metadata0)
    merged_metadata.update(
        {
            "merged": True,
            "source_shards": [
                str(by_index[index][0].resolve()) for index in range(num_shards)
            ],
            "source_devices": [
                by_index[index][1].get("device") for index in range(num_shards)
            ],
            "shard_index": None,
            "shard_items": len(merged_records),
            "ordered_shard_sample_sha256": selection_digest(all_items),
        }
    )
    merged_summary = {
        "metadata": merged_metadata,
        **summarize_records(merged_records, expected=len(all_items)),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for record in merged_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    sidecar_path(output, "meta").write_text(
        json.dumps(merged_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    sidecar_path(output, "summary").write_text(
        json.dumps(merged_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return merged_summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        summary = merge_shards(args.inputs, output=args.output)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(
        f"[official-merge] n={summary['n']} "
        f"accuracy={summary['accuracy']:.4f} -> {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
