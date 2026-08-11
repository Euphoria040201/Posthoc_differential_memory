from __future__ import annotations

import csv
import json
import random
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, patch

import pytest

from scripts import personamem_official_hf_eval as official
from scripts import personamem_official_merge as official_merge


def make_row(
    *,
    persona_id: str = "7",
    query: str = "What should I cook?",
    correct: str = "Make lentil soup.",
    distractors: tuple[str, str, str] = (
        "Order steak.",
        "Skip dinner.",
        "Only eat candy.",
    ),
) -> dict[str, str]:
    return {
        "persona_id": persona_id,
        "chat_history_32k_link": "data/chat_history_32k/persona7.json",
        "user_query": repr({"role": "user", "content": query}),
        "correct_answer": correct,
        "incorrect_answers": json.dumps(list(distractors)),
        "topic_query": "food",
        "topic_preference": "vegetarian",
        "conversation_scenario": "recommendation",
        "pref_type": "neutral_preferences",
        "who": "self",
        "updated": "False",
        "sensitive_info": "False",
        "distance_from_related_snippet_to_query_32k": "1024",
    }


def test_official_option_shuffle_is_seed_42_plus_original_row_index() -> None:
    row = make_row()
    item = official.build_official_item(row, row_index=13)
    expected = [
        row["correct_answer"],
        *json.loads(row["incorrect_answers"]),
    ]
    random.Random(42 + 13).shuffle(expected)
    assert list(item.options) == expected
    assert item.correct_letter == chr(
        ord("a") + expected.index(row["correct_answer"])
    )


def test_official_prompt_text_and_message_roles_are_exact() -> None:
    item = official.build_official_item(make_row(), row_index=0)
    history = [
        {"role": "user", "content": "I avoid meat."},
        {"role": "assistant", "content": "Understood."},
    ]
    messages = official.official_messages(item, history)
    assert messages[0] == {
        "role": "system",
        "content": official.OFFICIAL_SYSTEM_PROMPT,
    }
    assert messages[1:3] == history
    assert messages[-1]["role"] == "user"
    assert "You are performing a multiple-choice question task." in item.user_prompt
    assert "Provide your answer in the format: \\boxed{a}" in item.user_prompt
    assert item.user_prompt.endswith(official.OFFICIAL_THINK_INSTRUCTION)
    assert "(a) " in item.options_text()
    assert "(d) " in item.options_text()


def test_box_extraction_occurs_after_last_think_and_uses_last_match() -> None:
    assert (
        official.extract_official_boxed_letter(
            "<think>tempting \\\\boxed{a}</think> reasoning done \\\\boxed{C}"
        )
        == "c"
    )
    assert (
        official.extract_official_boxed_letter(
            "</think> first \\\\boxed{a}, revised \\\\boxed{d}"
        )
        == "d"
    )
    assert official.extract_official_boxed_letter("</think> Final answer: b") is None


def test_score_response_marks_missing_box_wrong_without_external_api() -> None:
    item = official.build_official_item(make_row(), row_index=1)
    prediction, correct, fallback = official.score_response(
        "I choose option a.",
        item,
        missing_box_policy="wrong",
    )
    assert prediction is None
    assert correct is False
    assert fallback is False


def test_frontier_api_protocol_remains_explicit_compatibility_mode() -> None:
    item = official.build_official_item(
        make_row(), row_index=0, protocol="frontier_api"
    )
    messages = official.official_messages(
        item,
        [{"role": "user", "content": "old history"}],
        protocol="frontier_api",
    )
    assert messages[0] == {"role": "user", "content": "old history"}
    assert messages[-1]["role"] == "system"
    assert "A. " in messages[-1]["content"]
    assert "Final Answer: [Letter]" in messages[-1]["content"]
    assert official.extract_frontier_letter("Final Answer: C") == "c"


class FakeQwenTokenizer:
    def __init__(self) -> None:
        self.kwargs = None
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return {"input_ids": [[11, 12, 13]]}


def test_encode_prompt_requires_official_qwen_thinking_flag() -> None:
    tokenizer = FakeQwenTokenizer()
    ids = official.encode_official_prompt(
        tokenizer,
        [{"role": "user", "content": "question"}],
        enable_thinking=True,
    )
    assert ids == [11, 12, 13]
    assert tokenizer.kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": True,
    }


def test_clean_subset_filters_collision_and_manifest_samples(tmp_path: Path) -> None:
    rows = [
        make_row(persona_id="1", query="keep"),
        make_row(persona_id="78", query="collision"),
        make_row(persona_id="2", query="leaking"),
    ]
    leaking = official.build_official_item(rows[2], row_index=2).sample_id
    csv_path = tmp_path / "benchmark.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = tmp_path / "clean.json"
    manifest.write_text(
        json.dumps({"exclude_sample_ids_all_windows": [leaking]}),
        encoding="utf-8",
    )

    full = official.load_official_items(
        csv_path, subset="official5000", clean_manifest=manifest
    )
    clean = official.load_official_items(
        csv_path, subset="clean4992", clean_manifest=manifest
    )
    assert [item.persona_id for item in full] == ["1", "78", "2"]
    assert [item.persona_id for item in clean] == ["1"]


def test_contiguous_shards_cover_every_item_once() -> None:
    items = tuple(
        official.build_official_item(
            make_row(persona_id=str(index), query=f"query {index}"),
            row_index=index,
        )
        for index in range(11)
    )
    shards = [
        official.contiguous_shard(items, shard_index=index, num_shards=4)
        for index in range(4)
    ]
    assert tuple(item for shard in shards for item in shard) == items
    assert [len(shard) for shard in shards] == [2, 3, 3, 3]


def test_history_derangements_are_stable_bijective_and_no_self() -> None:
    persona_ids = ("10", "2", "7", "99")
    first = official.make_history_derangements(
        persona_ids, num_swaps=3, seed=73
    )
    second = official.make_history_derangements(
        tuple(reversed(persona_ids)), num_swaps=3, seed=73
    )
    assert first == second
    assert len({tuple(sorted(mapping.items())) for mapping in first}) == 3
    for mapping in first:
        assert set(mapping) == set(persona_ids)
        assert set(mapping.values()) == set(persona_ids)
        assert all(target != source for target, source in mapping.items())


def test_history_conditions_select_correct_none_or_swapped_source() -> None:
    target = official.build_official_item(
        make_row(persona_id="7", query="q7"), row_index=0
    )
    donor = official.build_official_item(
        {
            **make_row(persona_id="8", query="q8"),
            "chat_history_32k_link": "data/chat_history_32k/persona8.json",
        },
        row_index=1,
    )
    histories = official.persona_history_items((target, donor))
    assert (
        official.history_source_item(
            target, condition="correct", histories=histories
        )
        == target
    )
    assert (
        official.history_source_item(
            target, condition="query_only", histories=histories
        )
        is None
    )
    assert (
        official.history_source_item(
            target,
            condition="swapped",
            histories=histories,
            swap_mapping={"7": "8", "8": "7"},
        )
        == donor
    )


def test_official_rag_chunking_drops_first_system_and_matches_message_overlap() -> None:
    history = [
        {"role": "system", "content": "private persona profile"},
        {"role": "user", "content": "message zero"},
        {"role": "assistant", "content": "message one"},
        {"role": "user", "content": "message two"},
        {"role": "assistant", "content": "message three"},
        {"role": "user", "content": "message four"},
    ]
    chunks = official.chunk_history_messages(
        history, chunk_size=3, chunk_overlap=1
    )
    assert [chunk.start_idx for chunk in chunks] == [0, 2]
    assert [
        message["content"] for message in chunks[0].messages
    ] == ["message zero", "message one", "message two"]
    assert [
        message["content"] for message in chunks[1].messages
    ] == ["message two", "message three", "message four"]
    assert all("private persona profile" not in chunk.text for chunk in chunks)


def test_bm25_retrieval_is_query_specific_and_deduplicates_overlap() -> None:
    history = [
        {"role": "system", "content": "private persona profile"},
        {"role": "user", "content": "I enjoy apples."},
        {"role": "assistant", "content": "I will remember that."},
        {"role": "user", "content": "Saffron is my favorite spice."},
        {"role": "assistant", "content": "Saffron is a distinctive choice."},
        {"role": "user", "content": "I avoid refined sugar."},
        {"role": "assistant", "content": "Understood."},
    ]
    retrieved, audit = official.retrieve_bm25_history(
        history,
        "Which saffron spice do I favor?",
        chunk_size=2,
        chunk_overlap=0,
        top_k=1,
    )
    assert [message["content"] for message in retrieved] == [
        "Saffron is my favorite spice.",
        "Saffron is a distinctive choice.",
    ]
    assert audit["candidate_chunks"] == 3
    assert audit["retrieved_chunk_starts"] == [2]

    all_messages, all_audit = official.retrieve_bm25_history(
        history,
        "unseen query terms",
        chunk_size=2,
        chunk_overlap=1,
        top_k=10,
    )
    assert [message["content"] for message in all_messages] == [
        message["content"] for message in history[1:]
    ]
    assert all_audit["retrieved_message_indices"] == list(range(6))


def test_bm25_keeps_common_official_prompt_without_rag_instruction() -> None:
    item = official.build_official_item(make_row(), row_index=0)
    retrieved = [{"role": "user", "content": "I avoid meat."}]
    messages = official.official_messages(item, retrieved)
    assert messages[0] == {
        "role": "system",
        "content": official.OFFICIAL_SYSTEM_PROMPT,
    }
    assert messages[1] == retrieved[0]
    assert messages[-1]["content"] == item.user_prompt
    assert all("relevant excerpts" not in message["content"] for message in messages)


def test_resume_rejects_condition_or_history_persona_mixing() -> None:
    item = official.build_official_item(make_row(), row_index=0)
    metadata = {
        "backend": "base",
        "subset": "clean4992",
        "history_condition": "swapped",
        "swap_seed": 17,
        "swap_index": 0,
        "num_swaps": 2,
    }
    record = {
        **metadata,
        "sample_id": item.sample_id,
        "query_persona": item.persona_id,
        "history_persona": "8",
    }
    official.validate_resume_state(
        records=[record],
        prior_metadata=metadata,
        expected_metadata=metadata,
        items=[item],
        history_personas={item.sample_id: "8"},
    )
    with pytest.raises(ValueError, match="history_condition"):
        official.validate_resume_state(
            records=[record],
            prior_metadata={**metadata, "history_condition": "correct"},
            expected_metadata=metadata,
            items=[item],
            history_personas={item.sample_id: "8"},
        )
    with pytest.raises(ValueError, match="history_persona"):
        official.validate_resume_state(
            records=[{**record, "history_persona": "9"}],
            prior_metadata=metadata,
            expected_metadata=metadata,
            items=[item],
            history_personas={item.sample_id: "8"},
        )


def test_eos_ids_match_official_tokenizer_only_stop_rule() -> None:
    tokenizer = SimpleNamespace(eos_token_id=151645)
    model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[151645, 151643])
    )
    assert official.eos_token_ids(model, tokenizer) == {151645}


def test_official_prefix_contribution_requires_hybrid_and_toggles_runtime() -> None:
    hybrid = SimpleNamespace(hybrid_read_mode="pooled_plus_prefix")
    legacy = SimpleNamespace(hybrid_read_mode="none")
    with patch(
        "deltamem.core.prefix_steer.set_hybrid_prefix_off"
    ) as setter:
        official.configure_prefix_contribution(object(), hybrid, "full")
        setter.assert_called_once_with(ANY, False)
    with patch(
        "deltamem.core.prefix_steer.set_hybrid_prefix_off"
    ) as setter:
        model = object()
        official.configure_prefix_contribution(model, hybrid, "off")
        setter.assert_called_once_with(model, True)
    with pytest.raises(ValueError, match="pooled_plus_prefix"):
        official.configure_prefix_contribution(object(), legacy, "off")


def test_checkpoint_file_identity_is_content_addressed(tmp_path: Path) -> None:
    checkpoint = tmp_path / "method.pt"
    checkpoint.write_bytes(b"first immutable checkpoint")
    first = official.prefix_checkpoint_file_identity(checkpoint)
    assert first["resolved_path"] == str(checkpoint.resolve())
    assert first["size_bytes"] == len(b"first immutable checkpoint")
    checkpoint.write_bytes(b"second checkpoint")
    second = official.prefix_checkpoint_file_identity(checkpoint)
    assert second["sha256"] != first["sha256"]


def test_checkpoint_contract_rejects_backbone_protocol_and_history_drift(
    tmp_path: Path,
) -> None:
    expected_model = tmp_path / "sft"
    expected_model.mkdir()
    args = Namespace(
        model_path=str(expected_model),
        protocol="qwen_verl",
        max_history_tokens=37_000,
        history_truncation="tail",
        allow_checkpoint_training_mismatch=False,
    )
    matching = {
        "training_model_path": str(expected_model),
        "training_reader_protocol": "official_qwen",
        "training_max_history_tokens": 37_000,
        "training_history_truncation": "tail",
    }
    official.validate_prefix_checkpoint_contract(args, matching)
    mismatched = {
        **matching,
        "training_model_path": str(tmp_path / "base"),
        "training_reader_protocol": "legacy",
        "training_max_history_tokens": 32_768,
        "training_history_truncation": "head",
    }
    with pytest.raises(ValueError, match="contract mismatch"):
        official.validate_prefix_checkpoint_contract(args, mismatched)
    args.allow_checkpoint_training_mismatch = True
    official.validate_prefix_checkpoint_contract(args, mismatched)


def test_merge_revalidates_exact_prefix_checkpoint_artifact(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "method.pt"
    checkpoint.write_bytes(b"paper checkpoint")
    identity = official.prefix_checkpoint_file_identity(checkpoint)
    metadata = {
        "backend": "prefix",
        "checkpoint_file": identity,
        "prefix_checkpoint_manifest": {
            **identity,
            "adapter_parameter_count": 123,
        },
        "adapter_parameter_count": 123,
    }
    official_merge.validate_checkpoint_artifact(metadata)
    checkpoint.write_bytes(b"overwritten checkpoint")
    with pytest.raises(ValueError, match="size changed|SHA256 changed"):
        official_merge.validate_checkpoint_artifact(metadata)


def test_official_cli_rejects_prefix_off_for_non_prefix_backend(
    tmp_path: Path,
) -> None:
    parser = official.build_arg_parser()
    args = parser.parse_args(
        [
            "--backend",
            "base",
            "--prefix-contribution",
            "off",
            "--output",
            str(tmp_path / "out.jsonl"),
        ]
    )
    with pytest.raises(SystemExit):
        official.validate_args(parser, args)


def test_summary_reports_micro_macro_format_and_completion() -> None:
    records = [
        {
            "persona_id": "large",
            "is_correct": True,
            "predicted_letter": "a",
            "embedding_fallback_used": False,
            "tags": {"pref_type": "x"},
        },
        {
            "persona_id": "large",
            "is_correct": True,
            "predicted_letter": "b",
            "embedding_fallback_used": False,
            "tags": {"pref_type": "x"},
        },
        {
            "persona_id": "small",
            "is_correct": False,
            "predicted_letter": None,
            "embedding_fallback_used": True,
            "tags": {"pref_type": "y"},
        },
    ]
    summary = official.summarize_records(records, expected=3)
    assert summary["accuracy"] == pytest.approx(2 / 3)
    assert summary["persona_macro_accuracy"] == pytest.approx(0.5)
    assert summary["boxed_format_rate"] == pytest.approx(2 / 3)
    assert summary["embedding_fallback_count"] == 1
    assert summary["complete"] is True


def _write_merge_fixture(
    root: Path,
    *,
    num_shards: int = 2,
    history_retrieval: str = "full",
) -> tuple[list[Path], Path]:
    rows = [
        {
            **make_row(persona_id=str(index + 1), query=f"query {index}"),
            "chat_history_32k_link": (
                f"data/chat_history_32k/persona{index + 1}.json"
            ),
        }
        for index in range(4)
    ]
    csv_path = root / "benchmark.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    clean_manifest = root / "clean.json"
    clean_manifest.write_text("{}", encoding="utf-8")
    items = official.load_official_items(
        csv_path,
        subset="official5000",
        clean_manifest=clean_manifest,
    )
    shard_paths: list[Path] = []
    for shard_index in range(num_shards):
        shard_items = official.contiguous_shard(
            items, shard_index=shard_index, num_shards=num_shards
        )
        shard_path = root / f"shard{shard_index}.jsonl"
        args = Namespace(
            protocol="qwen_verl",
            backend="base",
            subset="official5000",
            history_condition="correct",
            swap_seed=official.DEFAULT_SWAP_SEED,
            swap_index=0,
            num_swaps=1,
            model_path="/model",
            tokenizer_path="",
            checkpoint=None,
            dtype="bfloat16",
            attn_impl="sdpa",
            device=f"cuda:{shard_index}",
            local_files_only=True,
            benchmark_csv=csv_path,
            data_root=root,
            clean_manifest=clean_manifest,
            enable_thinking=True,
            max_prompt_tokens=37000,
            max_new_tokens=2048,
            max_history_tokens=32768,
            history_truncation="tail",
            history_retrieval=history_retrieval,
            rag_chunk_size=official.DEFAULT_RAG_CHUNK_SIZE,
            rag_chunk_overlap=official.DEFAULT_RAG_CHUNK_OVERLAP,
            rag_top_k=official.DEFAULT_RAG_TOP_K,
            max_items=0,
            missing_box_policy="wrong",
            num_shards=num_shards,
            shard_index=shard_index,
        )
        metadata = official._run_metadata(
            args,
            total_items=len(items),
            shard_items=len(shard_items),
            selected_items=shard_items,
        )
        records = [
            {
                "backend": "base",
                "subset": "official5000",
                "history_condition": "correct",
                "swap_seed": None,
                "swap_index": None,
                "num_swaps": 0,
                "sample_id": item.sample_id,
                "persona_id": item.persona_id,
                "query_persona": item.persona_id,
                "history_persona": item.persona_id,
                "is_correct": True,
                "predicted_letter": item.correct_letter,
                "embedding_fallback_used": False,
                "tags": item.tags,
                **(
                    {
                        "history_retrieval": "bm25",
                        "rag_chunk_size": official.DEFAULT_RAG_CHUNK_SIZE,
                        "rag_chunk_overlap": (
                            official.DEFAULT_RAG_CHUNK_OVERLAP
                        ),
                        "rag_top_k": official.DEFAULT_RAG_TOP_K,
                        "rag_audit": {},
                    }
                    if history_retrieval == "bm25"
                    else {}
                ),
            }
            for item in shard_items
        ]
        shard_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        official_merge.sidecar_path(shard_path, "meta").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        summary = {
            "metadata": metadata,
            **official.summarize_records(records, expected=len(records)),
        }
        official_merge.sidecar_path(shard_path, "summary").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        shard_paths.append(shard_path)
    return shard_paths, root / "merged.jsonl"


def test_safe_merge_requires_all_shards_and_exact_config(tmp_path: Path) -> None:
    shard_paths, output = _write_merge_fixture(tmp_path)
    with pytest.raises(ValueError, match="missing=\\[1\\]"):
        official_merge.merge_shards(shard_paths[:1], output=output)

    metadata_path = official_merge.sidecar_path(shard_paths[1], "meta")
    summary_path = official_merge.sidecar_path(shard_paths[1], "summary")
    changed_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    changed_metadata["max_new_tokens"] = 17
    metadata_path.write_text(json.dumps(changed_metadata), encoding="utf-8")
    changed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    changed_summary["metadata"] = changed_metadata
    summary_path.write_text(json.dumps(changed_summary), encoding="utf-8")
    with pytest.raises(ValueError, match="configuration differs"):
        official_merge.merge_shards(shard_paths, output=output)


def test_safe_merge_rejects_duplicate_sample_and_merges_exact_cover(
    tmp_path: Path,
) -> None:
    shard_paths, output = _write_merge_fixture(tmp_path)
    duplicate = json.loads(
        shard_paths[0].read_text(encoding="utf-8").splitlines()[0]
    )
    second_records = official.read_jsonl(shard_paths[1])
    second_records[0] = duplicate
    shard_paths[1].write_text(
        "".join(json.dumps(record) + "\n" for record in second_records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sample order/content"):
        official_merge.merge_shards(shard_paths, output=output)

    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    clean_shards, clean_output = _write_merge_fixture(clean_root)
    summary = official_merge.merge_shards(clean_shards, output=clean_output)
    assert summary["n"] == 4
    assert summary["complete"] is True
    assert len(official.read_jsonl(clean_output)) == 4


def test_bm25_metadata_is_resume_and_merge_compatible(tmp_path: Path) -> None:
    shard_paths, output = _write_merge_fixture(
        tmp_path, history_retrieval="bm25"
    )
    metadata = json.loads(
        official_merge.sidecar_path(shard_paths[0], "meta").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["history_retrieval"] == "bm25"
    assert metadata["rag_chunk_size"] == 6
    assert metadata["rag_chunk_overlap"] == 2
    assert metadata["rag_top_k"] == 10
    assert metadata["rag_prompt_instruction"] is None

    summary = official_merge.merge_shards(shard_paths, output=output)
    assert summary["n"] == 4
    assert summary["metadata"]["history_retrieval"] == "bm25"


def test_full_history_metadata_schema_remains_legacy_compatible(
    tmp_path: Path,
) -> None:
    shard_paths, _ = _write_merge_fixture(tmp_path)
    metadata = json.loads(
        official_merge.sidecar_path(shard_paths[0], "meta").read_text(
            encoding="utf-8"
        )
    )
    assert "history_retrieval" not in metadata
    assert "rag_chunk_size" not in metadata
    assert "prefix_contribution" not in metadata
    assert "checkpoint_file" not in metadata
