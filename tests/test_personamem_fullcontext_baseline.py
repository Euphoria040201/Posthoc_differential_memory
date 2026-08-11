from __future__ import annotations

import types
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from scripts import personamem_fullcontext_baseline as baseline
from scripts.personamem_v2_data import (
    AuditTags,
    ChatMessage,
    LoadedPersonaDataset,
    MCQExample,
    PersonaEpisode,
    ReaderInput,
    WriterInput,
)


def make_tags(
    *,
    pref_type: str = "neutral_preferences",
    distance: int | None = 100,
) -> AuditTags:
    return AuditTags(
        topic_query="lifestyle",
        topic_preference="gardening",
        conversation_scenario="chat_message",
        pref_type=pref_type,
        who="self",
        updated=False,
        sensitive_info=False,
        total_tokens=32_000,
        distance_to_related_snippet=distance,
        persona_relevant_tokens=1_000,
        persona_irrelevant_tokens=31_000,
    )


def make_question(
    sample_id: str, correct_index: int = 0, *, distance: int | None = 100
) -> MCQExample:
    return MCQExample(
        sample_id=sample_id,
        reader=ReaderInput(
            query=f"future query {sample_id}",
            options=("option zero", "option one", "option two", "option three"),
        ),
        correct_index=correct_index,
        tags=make_tags(distance=distance),
    )


def make_episode(
    persona_id: str = "7", questions: tuple[MCQExample, ...] | None = None
) -> PersonaEpisode:
    return PersonaEpisode(
        persona_id=persona_id,
        split="benchmark",
        window="32k",
        history_path=Path(f"/tmp/persona-{persona_id}.json"),
        writer=WriterInput(
            messages=(
                ChatMessage(role="user", content="history with ten tokens"),
            )
        ),
        questions=questions or (make_question(f"{persona_id}-0"),),
    )


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        if text in "ABCD":
            return {"input_ids": [30 + ord(text) - ord("A")]}
        if text in {" A", " B", " C", " D"}:
            return {"input_ids": [40 + ord(text[-1]) - ord("A")]}
        return {"input_ids": [1, 2]}

    def apply_chat_template(
        self, messages, *, tokenize: bool, add_generation_prompt: bool
    ):
        assert tokenize is True
        output: list[int] = []
        for message in messages:
            assert set(message) == {"role", "content"}
            if message["content"] == "history with ten tokens":
                output.extend(range(10, 20))
            elif message["content"] == "alternate history":
                output.extend(range(50, 56))
            else:
                output.extend([70, 71])
        if add_generation_prompt:
            output.append(99)
        return {"input_ids": output}


class FakeModel:
    def __call__(self, *, input_ids, attention_mask, use_cache):
        assert use_cache is False
        logits = torch.full((*input_ids.shape, 80), -10.0)
        logits[..., 79] = 100.0  # Must be ignored by forced-choice scoring.
        for row in range(input_ids.shape[0]):
            last = int(attention_mask[row].sum()) - 1
            prediction = int(input_ids[row, 0]) % 4
            logits[row, last, 30 + prediction] = 8.0
        return types.SimpleNamespace(logits=logits)


class FullContextBaselineTest(unittest.TestCase):
    def test_head_and_tail_truncate_only_history_and_preserve_reader_suffix(self) -> None:
        tokenizer = FakeTokenizer()
        episode = make_episode()
        question = episode.questions[0]
        head = baseline.encode_full_context(
            tokenizer,
            episode,
            question,
            max_context_tokens=8,
            truncation="head",
        )
        tail = baseline.encode_full_context(
            tokenizer,
            episode,
            question,
            max_context_tokens=8,
            truncation="tail",
        )
        self.assertEqual(head.input_ids, (10, 11, 12, 13, 14, 70, 71, 99))
        self.assertEqual(tail.input_ids, (15, 16, 17, 18, 19, 70, 71, 99))
        self.assertEqual(head.history_tokens_total, 10)
        self.assertEqual(head.history_tokens_kept, 5)
        self.assertEqual(head.reader_suffix_tokens, 3)
        self.assertEqual(head.history_tokens_truncated, 5)

    def test_reader_suffix_larger_than_budget_is_rejected_not_truncated(self) -> None:
        with self.assertRaisesRegex(ValueError, "current MCQ"):
            baseline.encode_full_context(
                FakeTokenizer(),
                make_episode(),
                make_episode().questions[0],
                max_context_tokens=2,
                truncation="tail",
            )

    def test_query_only_and_swapped_history_keep_the_same_mcq_suffix(self) -> None:
        tokenizer = FakeTokenizer()
        target = make_episode("target")
        source = replace(
            make_episode("source"),
            writer=WriterInput(
                messages=(
                    ChatMessage(role="user", content="alternate history"),
                )
            ),
        )
        question = target.questions[0]
        query_only = baseline.encode_full_context(
            tokenizer,
            target,
            question,
            max_context_tokens=32,
            truncation="tail",
            condition="query_only",
        )
        correct = baseline.encode_full_context(
            tokenizer,
            target,
            question,
            max_context_tokens=32,
            truncation="tail",
            condition="correct_history",
            history_episode=target,
        )
        swapped = baseline.encode_full_context(
            tokenizer,
            target,
            question,
            max_context_tokens=32,
            truncation="tail",
            condition="swapped_history",
            history_episode=source,
            swap_index=0,
        )
        self.assertEqual(query_only.input_ids, (70, 71, 99))
        self.assertEqual(correct.input_ids[-3:], query_only.input_ids)
        self.assertEqual(swapped.input_ids[-3:], query_only.input_ids)
        self.assertEqual(
            {item.sample_id for item in (query_only, correct, swapped)},
            {question.sample_id},
        )
        self.assertEqual(swapped.history_persona_id, "source")
        self.assertEqual(swapped.persona_id, "target")
        self.assertEqual(swapped.swap_index, 0)
        with self.assertRaisesRegex(ValueError, "different persona"):
            baseline.encode_full_context(
                tokenizer,
                target,
                question,
                max_context_tokens=32,
                truncation="tail",
                condition="swapped_history",
                history_episode=target,
                swap_index=0,
            )

    def test_fixed_swap_assignments_are_deterministic_unique_derangements(self) -> None:
        episodes = tuple(make_episode(str(index)) for index in range(4))
        first = baseline.build_deranged_swap_assignments(
            episodes, num_swaps=3, seed=81
        )
        second = baseline.build_deranged_swap_assignments(
            tuple(reversed(episodes)), num_swaps=3, seed=81
        )
        first_ids = {
            target: tuple(source.persona_id for source in sources)
            for target, sources in first.items()
        }
        second_ids = {
            target: tuple(source.persona_id for source in sources)
            for target, sources in second.items()
        }
        self.assertEqual(first_ids, second_ids)
        for target, sources in first_ids.items():
            self.assertEqual(len(sources), 3)
            self.assertEqual(len(set(sources)), 3)
            self.assertNotIn(target, sources)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            baseline.build_deranged_swap_assignments(
                episodes[:2], num_swaps=2, seed=1
            )

    def test_forced_choice_ignores_global_vocabulary_argmax(self) -> None:
        tags = make_tags()
        examples = [
            baseline.EncodedFullContext(
                sample_id=str(index),
                persona_id="p",
                input_ids=(index, 5, 6),
                gold_index=index,
                gold_letter=chr(ord("A") + index),
                tags=tags,
                history_tokens_total=2,
                history_tokens_kept=2,
                reader_suffix_tokens=1,
            )
            for index in range(4)
        ]
        predictions = baseline.forced_choice_batch(
            FakeModel(),
            examples,
            (30, 31, 32, 33),
            pad_token_id=0,
            device="cpu",
        )
        self.assertEqual(predictions, [0, 1, 2, 3])

    def test_metrics_include_micro_persona_macro_subgroups_and_distance(self) -> None:
        records = [
            {
                "persona_id": "p1",
                "correct": True,
                "tags": {
                    **baseline._tag_dict(make_tags(distance=100)),
                },
            },
            {
                "persona_id": "p1",
                "correct": True,
                "tags": {
                    **baseline._tag_dict(make_tags(distance=5_000)),
                },
            },
            {
                "persona_id": "p2",
                "correct": False,
                "tags": {
                    **baseline._tag_dict(make_tags(distance=40_000)),
                },
            },
        ]
        metrics = baseline.summarize_records(records)
        self.assertAlmostEqual(metrics["accuracy_micro"], 2 / 3)
        self.assertAlmostEqual(metrics["accuracy_persona_macro"], 0.5)
        self.assertEqual(metrics["subgroups"]["pref_type"]["neutral_preferences"]["n"], 3)
        self.assertEqual(metrics["distance"]["bins"]["[0,4096)"]["n"], 1)
        self.assertEqual(metrics["distance"]["bins"]["[4096,8192)"]["n"], 1)
        self.assertEqual(metrics["distance"]["bins"]["[32768,inf)"]["n"], 1)

    def test_condition_metrics_pair_correct_with_each_fixed_swap_and_count_flips(
        self,
    ) -> None:
        tags = baseline._tag_dict(make_tags(distance=100))

        def record(
            sample: str,
            persona: str,
            condition: str,
            prediction: int,
            gold: int,
            swap_index: int | None = None,
        ):
            return {
                "sample_id": sample,
                "persona_id": persona,
                "condition": condition,
                "prediction_index": prediction,
                "correct": prediction == gold,
                "swap_index": swap_index,
                "tags": tags,
            }

        records = [
            record("q1", "p1", "correct_history", 0, 0),
            record("q2", "p2", "correct_history", 1, 1),
            record("q1", "p1", "query_only", 2, 0),
            record("q2", "p2", "query_only", 1, 1),
            record("q1", "p1", "swapped_history", 2, 0, 0),
            record("q2", "p2", "swapped_history", 1, 1, 0),
            record("q1", "p1", "swapped_history", 0, 0, 1),
            record("q2", "p2", "swapped_history", 3, 1, 1),
        ]
        metrics = baseline.summarize_condition_records(
            records,
            conditions=("query_only", "correct_history", "swapped_history"),
        )
        self.assertEqual(metrics["primary_condition"], "correct_history")
        self.assertEqual(metrics["accuracy_micro"], 1.0)
        self.assertEqual(
            metrics["by_condition"]["swapped_history"]["accuracy_micro"], 0.5
        )
        paired = metrics["paired"]["correct_history_vs_swapped_history"]
        self.assertEqual(paired["num_pairs"], 4)
        self.assertEqual(paired["reference_minus_intervention_micro"], 0.5)
        self.assertEqual(paired["prediction_flip_count"], 2)
        self.assertEqual(paired["prediction_flip_rate"], 0.5)
        self.assertEqual(
            paired["by_swap_index"]["0"]["prediction_flip_rate"], 0.5
        )
        self.assertEqual(
            paired["by_swap_index"]["1"]["prediction_flip_rate"], 0.5
        )

    def test_filtering_drops_only_excluded_questions_not_entire_safe_history(self) -> None:
        episode = make_episode(
            questions=(make_question("drop-me"), make_question("keep-me"))
        )
        dataset = LoadedPersonaDataset(
            csv_path=Path("/tmp/benchmark.csv"),
            split="benchmark",
            window="32k",
            shuffle_seed=1,
            shuffle_round=0,
            episodes=(episode,),
            rows_seen=2,
            rows_skipped_missing_history=0,
            rows_skipped_invalid_mcq=0,
            rows_skipped_excluded_persona=0,
            excluded_persona_ids=(),
            content_overlap_warnings=("drop-me:user_query",),
        )
        clean = baseline.filter_clean_episodes(
            dataset,
            excluded_sample_ids=set(),
            detected_overlap_sample_ids={"drop-me"},
        )
        self.assertEqual(len(clean), 1)
        self.assertEqual(
            [question.sample_id for question in clean[0].questions],
            ["keep-me"],
        )

    def test_length_buffer_batches_reduce_local_padding(self) -> None:
        tags = make_tags()
        examples = [
            baseline.EncodedFullContext(
                sample_id=str(length),
                persona_id="p",
                input_ids=tuple(range(length)),
                gold_index=0,
                gold_letter="A",
                tags=tags,
                history_tokens_total=length - 1,
                history_tokens_kept=length - 1,
                reader_suffix_tokens=1,
            )
            for length in (9, 2, 8, 3)
        ]
        batches = list(
            baseline.buffered_length_batches(
                examples, batch_size=2, sort_buffer_size=4
            )
        )
        self.assertEqual(
            [[item.context_tokens for item in batch] for batch in batches],
            [[2, 3], [8, 9]],
        )


if __name__ == "__main__":
    unittest.main()
