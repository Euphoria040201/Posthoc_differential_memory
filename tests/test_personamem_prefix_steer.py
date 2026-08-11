from __future__ import annotations

import types
import unittest
import random
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from scripts import personamem_prefix_steer as pilot
from scripts.personamem_v2_data import (
    AuditTags,
    ChatMessage,
    MCQExample,
    PersonaEpisode,
    ReaderInput,
    WriterInput,
)


def make_question(
    sample_id: str,
    correct_index: int,
    *,
    distance: int | None = 10,
) -> MCQExample:
    return MCQExample(
        sample_id=sample_id,
        reader=ReaderInput(
            query=f"future query {sample_id}",
            options=("option zero", "option one", "option two", "option three"),
        ),
        correct_index=correct_index,
        tags=AuditTags(
            topic_query="topic",
            topic_preference="preference topic",
            conversation_scenario="chat_message",
            pref_type="neutral_preferences",
            who="self",
            updated=False,
            sensitive_info=False,
            total_tokens=100,
            distance_to_related_snippet=distance,
            persona_relevant_tokens=20,
            persona_irrelevant_tokens=80,
        ),
    )


def make_episode(persona_id: str, question_count: int = 4) -> PersonaEpisode:
    return PersonaEpisode(
        persona_id=persona_id,
        split="train",
        window="32k",
        history_path=Path(f"/tmp/persona-{persona_id}.json"),
        writer=WriterInput(
            messages=(
                ChatMessage(role="user", content=f"history only for {persona_id}"),
            )
        ),
        questions=tuple(
            make_question(f"{persona_id}-{index}", index % 4)
            for index in range(question_count)
        ),
    )


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        letter_ids = {" A": 11, " B": 12, " C": 13, " D": 14}
        if text in letter_ids:
            return {"input_ids": [letter_ids[text]]}
        if text in "ABCD":
            return {"input_ids": [30 + ord(text) - ord("A")]}
        # Variable lengths exercise right-padding and per-row last-logit gathering.
        length = 2 + (sum(text.encode("utf-8")) % 3)
        return {"input_ids": list(range(1, length + 1))}

    def apply_chat_template(
        self, messages, *, tokenize: bool, add_generation_prompt: bool
    ):
        assert tokenize is True
        assert set(messages[0]) == {"role", "content"}
        if add_generation_prompt:
            return {"input_ids": [201, 202, 203]}
        return {"input_ids": [101, 102, 103, 104, 105]}


class OfficialFakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def __init__(self) -> None:
        self.messages = None
        self.enable_thinking = None

    def __call__(self, text: str, *, add_special_tokens: bool = False):
        assert add_special_tokens is False
        prefix = pilot.OFFICIAL_BOXED_CLASSIFICATION_PREFIX
        if text == prefix:
            return {"input_ids": [91, 92]}
        if text.startswith(prefix) and text[len(prefix) :] in "abcd":
            return {
                "input_ids": [
                    91,
                    92,
                    60 + ord(text[-1]) - ord("a"),
                ]
            }
        if text in "abcd":
            return {"input_ids": [60 + ord(text) - ord("a")]}
        return {"input_ids": [7, 8]}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ):
        assert tokenize is True
        assert add_generation_prompt is True
        self.messages = messages
        self.enable_thinking = enable_thinking
        return {"input_ids": [201, 202, 203]}


class FakeReaderModel:
    def __init__(self):
        self.batch_sizes: list[int] = []

    def __call__(self, *, input_ids, attention_mask, use_cache):
        assert use_cache is False
        self.batch_sizes.append(input_ids.shape[0])
        logits = torch.full((*input_ids.shape, 40), -4.0)
        # A huge non-letter logit must not affect forced-choice evaluation.
        logits[..., 39] = 100.0
        for row in range(input_ids.shape[0]):
            last = int(attention_mask[row].sum()) - 1
            logits[row, last, 30 + (row % 4)] = 5.0
        return types.SimpleNamespace(logits=logits)


class WriterBackbone:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(last_hidden_state=torch.zeros(1))


class WriterWrapper:
    def __init__(self):
        self.model = WriterBackbone()
        self.full_model_calls = 0

    def __call__(self, **kwargs):
        self.full_model_calls += 1
        raise AssertionError("history write must not call the CausalLM/lm_head")


class PersonaMemPilotTest(unittest.TestCase):
    def test_distance_buckets_have_exact_boundaries_and_overflow(self) -> None:
        cases = {
            None: "unknown",
            -1: "unknown",
            0: "0-4k",
            4095: "0-4k",
            4096: "4-8k",
            8191: "4-8k",
            8192: "8-16k",
            16383: "8-16k",
            16384: "16-32k",
            32767: "16-32k",
            32768: "32k+",
        }
        self.assertEqual(
            {value: pilot._distance_bucket(value) for value in cases}, cases
        )

    def test_fixed_swap_derangements_are_bijective_and_have_no_self_swap(
        self,
    ) -> None:
        persona_ids = ["4", "1", "3", "0", "2"]
        first = pilot.make_swap_derangements(
            persona_ids, count=3, seed=123
        )
        second = pilot.make_swap_derangements(
            list(reversed(persona_ids)), count=3, seed=123
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        for mapping in first:
            self.assertEqual(set(mapping), set(persona_ids))
            self.assertEqual(set(mapping.values()), set(persona_ids))
            self.assertTrue(
                all(source != target for source, target in mapping.items())
            )
        self.assertEqual(
            len({tuple(sorted(mapping.items())) for mapping in first}), 3
        )

    def test_persona_macro_accuracy_does_not_overweight_large_persona(
        self,
    ) -> None:
        records = [
            {
                "persona_id": "large",
                "gold_index": 0,
                "predictions": {"correct": 0},
                "swap_predictions": [],
            }
            for _ in range(3)
        ]
        records.append(
            {
                "persona_id": "small",
                "gold_index": 0,
                "predictions": {"correct": 1},
                "swap_predictions": [],
            }
        )
        self.assertEqual(
            pilot._condition_accuracy(records, "correct"), 0.75
        )
        self.assertEqual(
            pilot._persona_macro_accuracy(records, "correct"), 0.5
        )

    def test_evaluate_reports_each_derangement_and_their_mean(self) -> None:
        episodes = tuple(make_episode(str(index), 2) for index in range(3))
        histories = {
            episode.persona_id: [int(episode.persona_id)]
            for episode in episodes
        }
        model = types.SimpleNamespace(current_persona=None)
        model.eval = lambda: None

        def clear(current_model):
            current_model.current_persona = None

        def write(current_model, history_ids, **_kwargs):
            current_model.current_persona = str(history_ids[0])

        def predict(current_model, _tokenizer, questions, _letters, **_kwargs):
            predictions = []
            for question in questions:
                source_persona = question.sample_id.split("-", maxsplit=1)[0]
                if current_model.current_persona == source_persona:
                    predictions.append(question.correct_index)
                else:
                    predictions.append((question.correct_index + 1) % 4)
            return predictions

        with (
            patch.object(pilot, "clear_frozen_memory", side_effect=clear),
            patch.object(pilot, "set_window_only"),
            patch.object(pilot, "set_steer_enabled"),
            patch.object(pilot, "write_persona_memory", side_effect=write) as writer,
            patch.object(
                pilot, "forced_choice_predictions", side_effect=predict
            ),
        ):
            result = pilot.evaluate(
                model,
                object(),
                episodes,
                histories,
                (11, 12, 13, 14),
                device="cpu",
                query_batch_size=4,
                prefix_enabled=True,
                num_swap_derangements=3,
                swap_seed=99,
            )

        self.assertEqual(result["accuracy"]["correct"], 1.0)
        self.assertEqual(result["persona_macro_accuracy"]["correct"], 1.0)
        self.assertEqual(result["accuracy"]["swap"], 0.0)
        self.assertEqual(
            result["swap_derangements"]["mean_accuracy"], 0.0
        )
        self.assertEqual(len(result["swap_derangements"]["runs"]), 3)
        self.assertEqual(writer.call_count, 3 + 3 * 3)
        self.assertEqual(result["protocol"]["swap_history_writes"], 9)
        for record in result["records"]:
            self.assertEqual(len(record["swap_predictions"]), 3)
            self.assertTrue(
                all(
                    persona_id != record["persona_id"]
                    for persona_id in record["swap_persona_ids"]
                )
            )
            self.assertIn(
                "distance_to_related_snippet", record["tags"]
            )
            self.assertEqual(record["tags"]["distance_bucket"], "0-4k")

    def test_single_persona_marks_swap_unavailable(self) -> None:
        episode = make_episode("7", 1)
        model = types.SimpleNamespace(eval=lambda: None)
        with (
            patch.object(pilot, "clear_frozen_memory"),
            patch.object(pilot, "set_window_only"),
            patch.object(pilot, "set_steer_enabled"),
            patch.object(pilot, "write_persona_memory") as writer,
            patch.object(
                pilot,
                "forced_choice_predictions",
                return_value=[episode.questions[0].correct_index],
            ),
        ):
            result = pilot.evaluate(
                model,
                object(),
                (episode,),
                {"7": [7]},
                (11, 12, 13, 14),
                device="cpu",
                query_batch_size=1,
                prefix_enabled=True,
                num_swap_derangements=3,
            )
        self.assertFalse(result["protocol"]["swap_available"])
        self.assertEqual(result["protocol"]["effective_swap_derangements"], 0)
        self.assertEqual(result["protocol"]["swap_history_writes"], 0)
        self.assertIsNone(result["accuracy"]["swap"])
        self.assertIsNone(result["paired"]["correct_minus_swap"])
        self.assertEqual(result["swap_derangements"]["runs"], [])
        self.assertIsNone(result["records"][0]["predictions"]["swap"])
        writer.assert_called_once()

    def test_memory_mode_resolution_and_pooled_steer_cli_validation(self) -> None:
        self.assertEqual(pilot.resolve_memory_mode("auto", 64), "prefix")
        self.assertEqual(pilot.resolve_memory_mode("auto", 0), "none")
        self.assertEqual(
            pilot.resolve_memory_mode("pooled_steer", 0), "pooled_steer"
        )

        parser = pilot.build_arg_parser()
        valid = parser.parse_args(
            [
                "--memory-mode",
                "pooled_steer",
                "--read-mode",
                "broadcast",
                "--P",
                "0",
                "--output",
                "/tmp/pooled.json",
            ]
        )
        pilot.validate_args(parser, valid)
        self.assertEqual(valid.resolved_memory_mode, "pooled_steer")

        invalid = parser.parse_args(
            [
                "--memory-mode",
                "pooled_steer",
                "--P",
                "0",
                "--output",
                "/tmp/invalid.json",
            ]
        )
        with self.assertRaises(SystemExit):
            pilot.validate_args(parser, invalid)

    def test_hybrid_cli_builds_additive_partitioned_checkpoint_config(
        self,
    ) -> None:
        parser = pilot.build_arg_parser()
        args = parser.parse_args(
            [
                "--memory-mode",
                "hybrid",
                "--read-mode",
                "pooled_plus_prefix",
                "--P",
                "64",
                "--head-dim",
                "64",
                "--prefix-write-layout",
                "partitioned",
                "--prefix-write-overlap-tokens",
                "8",
                "--hybrid-prefix-gate-mode",
                "learned_scalar",
                "--hybrid-prefix-gate-init",
                "0.1",
                "--hybrid-pool-drop-prob",
                "0.5",
                "--output",
                "/tmp/hybrid.json",
            ]
        )
        pilot.validate_args(parser, args)
        config = pilot.build_prefix_config(args)
        self.assertEqual(args.resolved_memory_mode, "hybrid")
        self.assertEqual(config.hybrid_read_mode, "pooled_plus_prefix")
        self.assertEqual(config.history_pool_mode, "attn")
        self.assertTrue(config.prefix_write)
        self.assertTrue(config.write_ctx_only)
        self.assertFalse(config.pool_reads)
        self.assertFalse(config.read_prefix_only)
        self.assertEqual(config.prefix_write_layout, "partitioned")
        self.assertEqual(config.prefix_write_overlap_tokens, 8)
        self.assertEqual(config.hybrid_prefix_gate_mode, "learned_scalar")
        self.assertEqual(args.hybrid_pool_drop_prob, 0.5)
        self.assertIn("hybrid_pool_drop_prob", pilot.RESUME_CRITICAL_ARGS)

        wrong_reader = parser.parse_args(
            [
                "--memory-mode",
                "hybrid",
                "--read-mode",
                "pool",
                "--P",
                "64",
                "--output",
                "/tmp/hybrid-wrong-reader.json",
            ]
        )
        with self.assertRaises(SystemExit):
            pilot.validate_args(parser, wrong_reader)

        non_hybrid_dropout = parser.parse_args(
            [
                "--memory-mode",
                "pooled_steer",
                "--read-mode",
                "broadcast",
                "--P",
                "0",
                "--hybrid-pool-drop-prob",
                "0.5",
                "--output",
                "/tmp/non-hybrid-dropout.json",
            ]
        )
        with self.assertRaises(SystemExit):
            pilot.validate_args(parser, non_hybrid_dropout)

    def test_hybrid_evaluation_reports_exact_same_weights_prefix_off(self) -> None:
        episodes = tuple(make_episode(str(index), 2) for index in range(3))
        histories = {
            episode.persona_id: [int(episode.persona_id)]
            for episode in episodes
        }
        model = types.SimpleNamespace(
            current_persona=None,
            prefix_off=False,
            pool_off=True,
            eval=lambda: None,
        )

        def clear(current_model):
            current_model.current_persona = None

        def write(current_model, history_ids, **_kwargs):
            current_model.current_persona = str(history_ids[0])

        def prefix_off(current_model, flag):
            current_model.prefix_off = bool(flag)

        def pool_off(current_model, flag):
            current_model.pool_off = bool(flag)

        def predict(current_model, _tokenizer, questions, _letters, **_kwargs):
            self.assertFalse(current_model.pool_off)
            predictions = []
            for question in questions:
                source = question.sample_id.split("-", maxsplit=1)[0]
                correct_memory = current_model.current_persona == source
                if correct_memory and not current_model.prefix_off:
                    predictions.append(question.correct_index)
                else:
                    predictions.append((question.correct_index + 1) % 4)
            return predictions

        with (
            patch.object(pilot, "clear_frozen_memory", side_effect=clear),
            patch.object(pilot, "set_window_only"),
            patch.object(pilot, "set_hybrid_prefix_off", side_effect=prefix_off),
            patch.object(pilot, "set_hybrid_pool_off", side_effect=pool_off),
            patch.object(pilot, "set_steer_enabled"),
            patch.object(pilot, "write_persona_memory", side_effect=write),
            patch.object(
                pilot, "forced_choice_predictions", side_effect=predict
            ),
        ):
            result = pilot.evaluate(
                model,
                object(),
                episodes,
                histories,
                (11, 12, 13, 14),
                device="cpu",
                query_batch_size=4,
                prefix_enabled=True,
                hybrid_prefix_ablation=True,
                num_swap_derangements=1,
            )

        self.assertEqual(result["accuracy"]["correct"], 1.0)
        self.assertEqual(result["accuracy"]["correct_full"], 1.0)
        self.assertEqual(result["accuracy"]["prefix_off"], 0.0)
        self.assertEqual(result["accuracy"]["window"], 0.0)
        self.assertEqual(result["accuracy"]["swap"], 0.0)
        self.assertEqual(
            result["paired"]["correct_full_minus_prefix_off"], 1.0
        )
        self.assertEqual(
            result["hybrid_same_weights"],
            {
                "available": True,
                "correct_full": 1.0,
                "prefix_off_pooled": 0.0,
                "swap_full": 0.0,
                "prefix_gain_over_pooled": 1.0,
                "correct_full_minus_swap": 1.0,
            },
        )
        self.assertTrue(
            all(
                record["predictions"]["correct"]
                == record["predictions"]["correct_full"]
                for record in result["records"]
            )
        )

    def test_checkpoint_loader_rejects_non_tensor_hybrid_layout_mismatch(
        self,
    ) -> None:
        class EmptyModel:
            def named_parameters(self):
                return ()

            def load_state_dict(self, _state, strict=False):
                self.strict = strict
                return (), ()

        config = pilot.PrefixSteerConfig(
            hybrid_read_mode="pooled_plus_prefix",
            prefix_write_layout="partitioned",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(
                {
                    "state": {},
                    "cfg": {
                        "hybrid_read_mode": "none",
                        "prefix_write_layout": "global",
                    },
                },
                path,
            )
            with self.assertRaisesRegex(
                RuntimeError, "checkpoint/config metadata mismatch"
            ):
                pilot._load_checkpoint(
                    EmptyModel(), str(path), config=config
                )

    def test_letter_ids_reader_ce_and_forced_choice_are_batched(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeReaderModel()
        questions = [make_question(str(index), index) for index in range(4)]
        letter_ids = pilot.resolve_letter_token_ids(tokenizer)
        self.assertEqual(letter_ids, (30, 31, 32, 33))

        with patch.object(pilot, "set_steer_segments"):
            loss = pilot.letter_ce_loss(
                model, tokenizer, questions, letter_ids, device="cpu"
            )
            predictions = pilot.forced_choice_predictions(
                model,
                tokenizer,
                questions,
                letter_ids,
                device="cpu",
                batch_size=4,
            )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(predictions, [0, 1, 2, 3])
        self.assertEqual(model.batch_sizes, [4, 4])

    def test_task_ce_supports_legacy_full_vocab_and_forced_four_choice(
        self,
    ) -> None:
        batch = pilot.ReaderBatch(
            input_ids=torch.ones((2, 1), dtype=torch.long),
            attention_mask=torch.ones((2, 1), dtype=torch.bool),
            segments=torch.ones((2, 1), dtype=torch.long),
            last_indices=torch.zeros(2, dtype=torch.long),
            target_token_ids=torch.tensor([0, 1]),
            target_indices=torch.tensor([0, 1]),
        )
        logits = torch.tensor(
            [
                [4.0, 0.0, -1.0, -2.0, 20.0],
                [0.0, 4.0, -1.0, -2.0, 20.0],
            ]
        )
        full_vocab = pilot.task_ce_loss_from_logits(
            logits,
            batch,
            (0, 1, 2, 3),
            task_loss="full_vocab",
        )
        four_choice = pilot.task_ce_loss_from_logits(
            logits,
            batch,
            (0, 1, 2, 3),
            task_loss="four_choice",
        )
        self.assertTrue(
            torch.allclose(
                full_vocab,
                torch.nn.functional.cross_entropy(
                    logits, batch.target_token_ids
                ),
            )
        )
        self.assertTrue(
            torch.allclose(
                four_choice,
                torch.nn.functional.cross_entropy(
                    logits[:, :4], batch.target_indices
                ),
            )
        )
        # A dominant irrelevant vocabulary item hurts legacy CE but is outside
        # both the paper metric and the optional four-choice training objective.
        self.assertGreater(full_vocab.item(), 10.0)
        self.assertLess(four_choice.item(), 0.1)

    def test_reader_uses_chat_generation_template(self) -> None:
        ids = pilot.encode_reader_prompt(FakeTokenizer(), make_question("chat", 0))
        self.assertEqual(ids, [201, 202, 203])

    def test_official_reader_reuses_exact_prompt_and_boxed_lowercase_target(
        self,
    ) -> None:
        from scripts.personamem_official_hf_eval import (
            OFFICIAL_MCQ_TEMPLATE,
            OFFICIAL_SYSTEM_PROMPT,
            OFFICIAL_THINK_INSTRUCTION,
        )

        question = make_question("official", 2)
        tokenizer = OfficialFakeTokenizer()
        ids = pilot.encode_reader_prompt(
            tokenizer, question, reader_protocol="official_qwen"
        )
        option_lines = "\n".join(
            f"({letter}) {option}"
            for letter, option in zip("abcd", question.reader.options)
        )
        self.assertEqual(
            tokenizer.messages,
            [
                {"role": "system", "content": OFFICIAL_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        question.reader.query
                        + OFFICIAL_MCQ_TEMPLATE.format(
                            options_text=option_lines
                        )
                        + OFFICIAL_THINK_INSTRUCTION
                    ),
                },
            ],
        )
        self.assertTrue(tokenizer.enable_thinking)
        self.assertEqual(ids, [201, 202, 203, 91, 92])
        self.assertEqual(
            pilot.resolve_letter_token_ids(
                tokenizer, reader_protocol="official_qwen"
            ),
            (60, 61, 62, 63),
        )
        batch = pilot.collate_reader_batch(
            tokenizer,
            [question],
            (60, 61, 62, 63),
            device="cpu",
            reader_protocol="official_qwen",
        )
        # Causal-LM logits at the final fixed-prefix token supervise the next
        # (lowercase answer) token; the target itself is not appended to input.
        self.assertEqual(batch.last_indices.tolist(), [len(ids) - 1])
        self.assertEqual(
            batch.input_ids[0, : len(ids)].tolist(), ids
        )
        self.assertEqual(batch.input_ids[0, len(ids) - 1].item(), 92)
        self.assertEqual(batch.target_token_ids.tolist(), [62])

    def test_official_reader_never_reshuffles_options_across_updates(self) -> None:
        episode = make_episode("fixed", question_count=4)
        first = pilot._questions_for_step(
            episode,
            count=4,
            rng=random.Random(1),
            option_shuffle_seed=1,
            option_shuffle_round=1,
            reader_protocol="official_qwen",
        )
        second = pilot._questions_for_step(
            episode,
            count=4,
            rng=random.Random(1),
            option_shuffle_seed=999,
            option_shuffle_round=999,
            reader_protocol="official_qwen",
        )
        self.assertEqual(first, second)
        self.assertEqual(first, episode.questions)

    def test_cyclic_chunk_sampler_exhausts_each_question_cycle_before_repeat(
        self,
    ) -> None:
        episodes = (
            make_episode("a", question_count=5),
            make_episode("b", question_count=5),
        )
        first = pilot.CyclicChunkSampler(episodes, seed=17)
        second = pilot.CyclicChunkSampler(
            tuple(reversed(episodes)), seed=17
        )
        observed = []
        replayed = []
        for _ in range(4):
            episode, questions = first.next_chunk(3)
            replay_episode, replay_questions = second.next_chunk(3)
            observed.append(
                (episode.persona_id, tuple(q.sample_id for q in questions))
            )
            replayed.append(
                (
                    replay_episode.persona_id,
                    tuple(q.sample_id for q in replay_questions),
                )
            )
        self.assertEqual(observed, replayed)
        self.assertTrue(all(len(sample_ids) == 3 for _, sample_ids in observed))
        by_persona: dict[str, list[str]] = {}
        for persona_id, sample_ids in observed:
            by_persona.setdefault(persona_id, []).extend(sample_ids)
        # Each persona has two exact-K chunks.  Its first five outputs exhaust
        # all five questions before the sixth starts a new shuffled cycle.
        for persona_id, sample_ids in by_persona.items():
            self.assertEqual(
                set(sample_ids[:5]),
                {f"{persona_id}-{index}" for index in range(5)},
            )
            self.assertNotEqual(sample_ids[4], sample_ids[5])

    def test_cyclic_chunk_sampler_resume_is_bit_exact_mid_cycles(self) -> None:
        episodes = (
            make_episode("a", question_count=5),
            make_episode("b", question_count=7),
            make_episode("c", question_count=4),
        )
        uninterrupted = pilot.CyclicChunkSampler(episodes, seed=29)
        for _ in range(5):
            uninterrupted.next_chunk(3)
        state = uninterrupted.state_dict()

        restored = pilot.CyclicChunkSampler(
            tuple(reversed(episodes)), seed=999
        )
        restored.load_state_dict(state)
        for _ in range(12):
            expected_episode, expected_questions = uninterrupted.next_chunk(3)
            actual_episode, actual_questions = restored.next_chunk(3)
            self.assertEqual(actual_episode.persona_id, expected_episode.persona_id)
            self.assertEqual(
                tuple(question.sample_id for question in actual_questions),
                tuple(question.sample_id for question in expected_questions),
            )
        self.assertEqual(restored.state_dict(), uninterrupted.state_dict())

        changed = (
            make_episode("a", question_count=6),
            episodes[1],
            episodes[2],
        )
        with self.assertRaisesRegex(ValueError, "dataset/order"):
            pilot.CyclicChunkSampler(changed, seed=29).load_state_dict(state)

    def test_hybrid_pool_dropout_has_independent_bit_exact_resume(self) -> None:
        uninterrupted = pilot.HybridPoolDropoutSchedule(0.5, seed=301)
        prefix = [uninterrupted.next() for _ in range(17)]
        state = uninterrupted.state_dict()

        # Unrelated RNG traffic cannot perturb the dedicated branch schedule.
        random.Random(301).random()
        torch.rand(9)
        expected_tail = [uninterrupted.next() for _ in range(31)]

        restored = pilot.HybridPoolDropoutSchedule(0.5, seed=301)
        restored.load_state_dict(state)
        actual_tail = [restored.next() for _ in range(31)]
        self.assertEqual(actual_tail, expected_tail)
        self.assertEqual(restored.state_dict(), uninterrupted.state_dict())
        self.assertEqual(len(prefix), state["draw_count"])

        with self.assertRaisesRegex(ValueError, "probability"):
            pilot.HybridPoolDropoutSchedule(
                0.25, seed=301
            ).load_state_dict(state)

    def test_cyclic_label_budget_uses_variable_chunks_and_exact_labels(
        self,
    ) -> None:
        episodes = (
            make_episode("a", question_count=2),
            make_episode("b", question_count=5),
            make_episode("c", question_count=7),
        )
        plan = pilot.simulate_cyclic_label_budget(
            episodes,
            seed=13,
            max_queries_per_write=4,
            labels_per_update=8,
            optimizer_updates=2,
        )
        self.assertEqual(plan["label_exposures"], 16)
        self.assertEqual(plan["seen_persona_count"], 3)
        self.assertEqual(plan["seen_sample_count"], 14)
        self.assertGreater(plan["micro_steps"], 2)

        weighted = pilot.mean_train_diagnostics(
            [
                {"loss": 2.0, "query_count": 2},
                {"loss": 4.0, "query_count": 6},
            ],
            weight_by_query_count=True,
        )
        self.assertEqual(weighted["loss"], 3.5)

    def test_history_write_uses_model_model_once_and_never_lm_head(self) -> None:
        model = WriterWrapper()
        events: list[tuple[str, bool]] = []
        with ExitStack() as stack:
            stack.enter_context(patch.object(pilot, "clear_frozen_memory"))
            stack.enter_context(patch.object(pilot, "set_window_only"))
            stack.enter_context(patch.object(pilot, "set_steer_enabled"))
            stack.enter_context(patch.object(pilot, "set_steer_segments"))
            stack.enter_context(
                patch.object(
                    pilot,
                    "set_write_only",
                    side_effect=lambda _model, flag: events.append(("write_only", flag)),
                )
            )
            stack.enter_context(
                patch.object(pilot, "has_frozen_memory", return_value=[True, True])
            )
            pilot.write_persona_memory(
                model,
                [7, 8, 9],
                device="cpu",
                grad=True,
                prefix_enabled=True,
            )
        self.assertEqual(model.full_model_calls, 0)
        self.assertEqual(len(model.model.calls), 1)
        self.assertEqual(model.model.calls[0]["input_ids"].tolist(), [[7, 8, 9]])
        self.assertEqual(events, [("write_only", True), ("write_only", False)])

    def test_train_step_writes_once_then_passes_k_queries_together(self) -> None:
        episode = make_episode("5", question_count=5)
        fake_loss = torch.tensor(1.25, requires_grad=True)
        with (
            patch.object(pilot, "set_steer_enabled"),
            patch.object(pilot, "set_window_only"),
            patch.object(pilot, "write_persona_memory") as write,
            patch.object(pilot, "letter_ce_loss", return_value=fake_loss) as read,
        ):
            result = pilot.train_step(
                object(),
                FakeTokenizer(),
                episode,
                [1, 2, 3],
                (11, 12, 13, 14),
                device="cpu",
                queries_per_write=3,
                prefix_enabled=True,
                rng=__import__("random").Random(1),
                option_shuffle_seed=5,
                option_shuffle_round=2,
            )
        self.assertIs(result.loss, fake_loss)
        self.assertIs(result.ce_loss, fake_loss)
        self.assertIsNone(result.identity_contrast_loss)
        self.assertIsNone(result.donor_persona_id)
        write.assert_called_once()
        self.assertEqual(len(read.call_args.args[2]), 3)
        self.assertEqual(read.call_args.kwargs["task_loss"], "full_vocab")

    def test_identity_donor_is_deterministic_order_independent_and_non_self(
        self,
    ) -> None:
        episodes = tuple(make_episode(str(index)) for index in range(6))
        first = pilot.select_identity_donor(
            "2", episodes, seed=17, step=9
        )
        second = pilot.select_identity_donor(
            "2", tuple(reversed(episodes)), seed=17, step=9
        )
        self.assertEqual(first.persona_id, second.persona_id)
        self.assertNotEqual(first.persona_id, "2")
        with self.assertRaisesRegex(ValueError, "at least two distinct"):
            pilot.select_identity_donor(
                "2", (make_episode("2"),), seed=17, step=9
            )

    def test_identity_contrast_terms_use_gold_forced_choice_margin(self) -> None:
        batch = pilot.ReaderBatch(
            input_ids=torch.ones((2, 1), dtype=torch.long),
            attention_mask=torch.ones((2, 1), dtype=torch.bool),
            segments=torch.ones((2, 1), dtype=torch.long),
            last_indices=torch.zeros(2, dtype=torch.long),
            target_token_ids=torch.tensor([1, 2]),
            target_indices=torch.tensor([0, 1]),
        )
        correct = torch.tensor(
            [
                [3.0, 0.0, -1.0, -2.0, 100.0],
                [0.0, 2.0, -1.0, -2.0, 100.0],
            ],
            requires_grad=True,
        )
        wrong = torch.tensor(
            [
                [1.0, 0.0, -1.0, -2.0, -100.0],
                [0.0, 3.0, -1.0, -2.0, -100.0],
            ],
            requires_grad=True,
        )
        contrast, log_probability_gap, probability_gap = (
            pilot.identity_contrast_terms(
                correct, wrong, batch, (0, 1, 2, 3), margin=0.5
            )
        )
        correct_log_probabilities = correct[:, :4].log_softmax(dim=-1)
        wrong_log_probabilities = wrong[:, :4].log_softmax(dim=-1)
        expected_gaps = torch.stack(
            (
                correct_log_probabilities[0, 0]
                - wrong_log_probabilities[0, 0],
                correct_log_probabilities[1, 1]
                - wrong_log_probabilities[1, 1],
            )
        )
        expected_contrast = torch.nn.functional.softplus(
            0.5 - expected_gaps
        ).mean()
        self.assertTrue(torch.allclose(contrast, expected_contrast))
        self.assertTrue(
            torch.allclose(log_probability_gap, expected_gaps.mean())
        )
        self.assertTrue(torch.isfinite(probability_gap))

        shifted_correct = correct.detach().clone()
        shifted_wrong = wrong.detach().clone()
        shifted_correct[:, :4] += 100.0
        shifted_wrong[:, :4] -= 70.0
        shifted = pilot.identity_contrast_terms(
            shifted_correct,
            shifted_wrong,
            batch,
            (0, 1, 2, 3),
            margin=0.5,
        )
        # Raw gold logits would change by +170 here.  A-D log-softmax removes
        # that common-shift shortcut exactly.
        self.assertTrue(torch.allclose(contrast, shifted[0], atol=1e-6))
        self.assertTrue(
            torch.allclose(log_probability_gap, shifted[1], atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(probability_gap, shifted[2], atol=1e-6)
        )
        contrast.backward()
        self.assertIsNotNone(correct.grad)
        self.assertIsNotNone(wrong.grad)
        # The non-letter logit is ignored by the forced-choice contrast.
        self.assertEqual(float(correct.grad[:, 4].abs().sum()), 0.0)
        self.assertEqual(float(wrong.grad[:, 4].abs().sum()), 0.0)

    def test_identity_train_step_writes_target_and_donor_once_for_same_batch(
        self,
    ) -> None:
        episode = make_episode("target", question_count=3)
        batch = pilot.ReaderBatch(
            input_ids=torch.ones((2, 1), dtype=torch.long),
            attention_mask=torch.ones((2, 1), dtype=torch.bool),
            segments=torch.ones((2, 1), dtype=torch.long),
            last_indices=torch.zeros(2, dtype=torch.long),
            target_token_ids=torch.tensor([0, 1]),
            target_indices=torch.tensor([0, 1]),
        )
        correct_logits = torch.tensor(
            [[2.0, 0.0, -1.0, -2.0, 9.0], [0.0, 2.0, -1.0, -2.0, 9.0]],
            requires_grad=True,
        )
        wrong_logits = torch.tensor(
            [[0.0, 1.0, -1.0, -2.0, 9.0], [1.0, 0.0, -1.0, -2.0, 9.0]],
            requires_grad=True,
        )
        with (
            patch.object(pilot, "set_steer_enabled"),
            patch.object(pilot, "set_window_only"),
            patch.object(pilot, "write_persona_memory") as write,
            patch.object(
                pilot, "collate_reader_batch", return_value=batch
            ) as collate,
            patch.object(
                pilot,
                "final_reader_logits",
                side_effect=[correct_logits, wrong_logits],
            ) as reader,
            patch.object(pilot, "letter_ce_loss") as ordinary_ce,
        ):
            result = pilot.train_step(
                object(),
                FakeTokenizer(),
                episode,
                [1, 2, 3],
                (0, 1, 2, 3),
                device="cpu",
                queries_per_write=2,
                prefix_enabled=True,
                rng=__import__("random").Random(1),
                option_shuffle_seed=5,
                option_shuffle_round=2,
                task_loss="four_choice",
                identity_contrast_lambda=0.7,
                identity_margin=0.5,
                donor_history_ids=[8, 9],
                donor_persona_id="donor",
            )
        self.assertEqual(write.call_count, 2)
        self.assertEqual(write.call_args_list[0].args[1], [1, 2, 3])
        self.assertEqual(write.call_args_list[1].args[1], [8, 9])
        self.assertEqual(collate.call_count, 1)
        self.assertEqual(reader.call_count, 2)
        self.assertIs(reader.call_args_list[0].args[1], batch)
        self.assertIs(reader.call_args_list[1].args[1], batch)
        ordinary_ce.assert_not_called()
        self.assertEqual(result.target_persona_id, "target")
        self.assertEqual(result.donor_persona_id, "donor")
        self.assertIsNotNone(result.identity_contrast_loss)
        self.assertIsNotNone(result.gold_log_probability_gap)
        self.assertIsNotNone(result.gold_probability_gap)
        expected_ce = torch.nn.functional.cross_entropy(
            correct_logits[:, :4], batch.target_indices
        )
        self.assertTrue(torch.allclose(result.ce_loss, expected_ce))
        result.loss.backward()
        # Correct memory is trained by both four-choice CE and contrast;
        # wrong memory is trained only by contrast.
        self.assertIsNotNone(correct_logits.grad)
        self.assertIsNotNone(wrong_logits.grad)

    def test_identity_train_step_rejects_self_donor_before_writer(self) -> None:
        episode = make_episode("same")
        with patch.object(pilot, "write_persona_memory") as write:
            with self.assertRaisesRegex(ValueError, "must differ"):
                pilot.train_step(
                    object(),
                    FakeTokenizer(),
                    episode,
                    [1],
                    (0, 1, 2, 3),
                    device="cpu",
                    queries_per_write=1,
                    prefix_enabled=True,
                    rng=__import__("random").Random(1),
                    option_shuffle_seed=5,
                    option_shuffle_round=2,
                    identity_contrast_lambda=1.0,
                    donor_history_ids=[2],
                    donor_persona_id="same",
                )
        write.assert_not_called()

    def test_identity_train_step_rejects_missing_persistent_memory(self) -> None:
        episode = make_episode("target")
        with patch.object(pilot, "write_persona_memory") as write:
            with self.assertRaisesRegex(ValueError, "persistent"):
                pilot.train_step(
                    object(),
                    FakeTokenizer(),
                    episode,
                    [1],
                    (0, 1, 2, 3),
                    device="cpu",
                    queries_per_write=1,
                    prefix_enabled=False,
                    rng=__import__("random").Random(1),
                    option_shuffle_seed=5,
                    option_shuffle_round=2,
                    identity_contrast_lambda=1.0,
                    donor_history_ids=[2],
                    donor_persona_id="donor",
                )
        write.assert_not_called()

    def test_training_option_shuffle_changes_round_but_preserves_gold(self) -> None:
        episode = make_episode("shuffle", question_count=4)
        first = pilot._questions_for_step(
            episode,
            count=4,
            rng=__import__("random").Random(1),
            option_shuffle_seed=55,
            option_shuffle_round=1,
        )
        second = pilot._questions_for_step(
            episode,
            count=4,
            rng=__import__("random").Random(1),
            option_shuffle_seed=55,
            option_shuffle_round=2,
        )
        for original, shuffled in zip(episode.questions, first):
            self.assertEqual(
                shuffled.reader.options[shuffled.correct_index],
                original.reader.options[original.correct_index],
            )
        self.assertNotEqual(
            [item.reader.options for item in first],
            [item.reader.options for item in second],
        )

    def test_history_encoding_accepts_only_writer_and_truncates_tail(self) -> None:
        episode = make_episode("6")
        tokenizer = FakeTokenizer()
        ids = pilot.encode_history(
            tokenizer,
            episode.writer,
            persona_id=episode.persona_id,
            max_history_tokens=3,
            truncation="tail",
        )
        self.assertEqual(ids, [103, 104, 105])

    def test_deterministic_unseen_persona_holdout(self) -> None:
        train = tuple(make_episode(str(index)) for index in range(20))
        val = tuple(make_episode(str(index)) for index in range(2, 20))
        first_train, first_val, first_ids = pilot.deterministic_persona_holdout(
            train, val, fraction=0.25, seed=123
        )
        second_train, second_val, second_ids = pilot.deterministic_persona_holdout(
            train, val, fraction=0.25, seed=123
        )
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(len(first_ids), 5)
        self.assertFalse(
            {episode.persona_id for episode in first_train}
            & {episode.persona_id for episode in first_val}
        )
        self.assertEqual(
            {episode.persona_id for episode in first_val}, first_ids
        )
        self.assertEqual(first_train, second_train)
        self.assertEqual(first_val, second_val)

    def test_limit_episodes_caps_personas_and_total_queries(self) -> None:
        episodes = tuple(make_episode(str(index), question_count=4) for index in range(5))
        selected = pilot.limit_episodes(
            episodes,
            max_personas=3,
            max_queries=6,
            selection_seed=9,
            shuffle_personas=False,
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(sum(len(item.questions) for item in selected), 6)
        self.assertEqual([item.persona_id for item in selected], ["0", "1"])

    def test_explicit_persona_subset_rejects_missing_ids(self) -> None:
        episodes = tuple(make_episode(str(index)) for index in range(4))
        selected = pilot.include_persona_episodes(
            episodes, {"1", "3"}, label="train"
        )
        self.assertEqual([item.persona_id for item in selected], ["1", "3"])
        with self.assertRaisesRegex(ValueError, "requested persona IDs are missing"):
            pilot.include_persona_episodes(
                episodes, {"1", "missing"}, label="eval"
            )

    def test_cli_defaults_exclude_known_overlap_and_fix_eval_shuffle(self) -> None:
        parser = pilot.build_arg_parser()
        args = parser.parse_args(["--output", "/tmp/result.json"])
        self.assertEqual(args.exclude_persona_ids, "78")
        self.assertEqual(args.persona_holdout_fraction, 0.0)
        self.assertIsNone(args.persona_holdout_size)
        self.assertEqual(args.eval_option_seed, 1618)
        self.assertEqual(args.num_swap_derangements, 1)
        self.assertEqual(args.swap_seed, 4242)
        self.assertEqual(args.identity_contrast_lambda, 0.0)
        self.assertEqual(args.identity_margin, 0.0)
        self.assertEqual(args.identity_donor_seed, 7331)
        self.assertEqual(args.task_loss, "full_vocab")
        self.assertEqual(args.reader_protocol, "legacy")
        self.assertEqual(args.train_sampler, "random_persona")
        self.assertEqual(args.grad_accum_steps, 1)
        self.assertEqual(args.save_every, 0)
        self.assertEqual(args.resume_checkpoint, "")
        self.assertEqual(args.labels_per_update, 0)
        self.assertEqual(args.hybrid_pool_drop_prob, 0.0)

    def test_cyclic_label_budget_cli_is_explicit_and_grad_accum_is_rejected(
        self,
    ) -> None:
        parser = pilot.build_arg_parser()
        valid = parser.parse_args(
            [
                "--train-sampler",
                "cyclic_label_budget",
                "--labels-per-update",
                "64",
                "--queries-per-write",
                "32",
                "--output",
                "/tmp/label-budget.json",
            ]
        )
        pilot.validate_args(parser, valid)
        invalid = parser.parse_args(
            [
                "--train-sampler",
                "cyclic_label_budget",
                "--labels-per-update",
                "64",
                "--grad-accum-steps",
                "2",
                "--output",
                "/tmp/label-budget-invalid.json",
            ]
        )
        with self.assertRaises(SystemExit):
            pilot.validate_args(parser, invalid)

    def test_identity_contrast_cli_requires_persistent_memory(self) -> None:
        parser = pilot.build_arg_parser()
        no_memory = parser.parse_args(
            [
                "--memory-mode",
                "none",
                "--P",
                "0",
                "--identity-contrast-lambda",
                "1",
                "--output",
                "/tmp/no-memory.json",
            ]
        )
        with self.assertRaises(SystemExit):
            pilot.validate_args(parser, no_memory)

        prefix = parser.parse_args(
            [
                "--memory-mode",
                "prefix",
                "--P",
                "64",
                "--identity-contrast-lambda",
                "0.5",
                "--identity-margin",
                "1.0",
                "--task-loss",
                "four_choice",
                "--output",
                "/tmp/prefix.json",
            ]
        )
        pilot.validate_args(parser, prefix)
        self.assertEqual(prefix.resolved_memory_mode, "prefix")
        self.assertEqual(prefix.task_loss, "four_choice")

        pooled = parser.parse_args(
            [
                "--memory-mode",
                "pooled_steer",
                "--read-mode",
                "broadcast",
                "--P",
                "0",
                "--identity-contrast-lambda",
                "0.5",
                "--output",
                "/tmp/pooled-contrast.json",
            ]
        )
        pilot.validate_args(parser, pooled)
        self.assertEqual(pooled.resolved_memory_mode, "pooled_steer")


if __name__ == "__main__":
    unittest.main()
