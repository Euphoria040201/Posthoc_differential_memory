from __future__ import annotations

import csv
import importlib.util
import io
import json
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import fields
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "personamem_v2_data.py"
)
SPEC = importlib.util.spec_from_file_location("personamem_v2_data", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
pm = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pm
SPEC.loader.exec_module(pm)


CSV_FIELDS = [
    "persona_id",
    "chat_history_32k_link",
    "chat_history_128k_link",
    "user_query",
    "correct_answer",
    "incorrect_answers",
    "topic_query",
    "preference",
    "topic_preference",
    "conversation_scenario",
    "pref_type",
    "related_conversation_snippet",
    "who",
    "updated",
    "prev_pref",
    "sensitive_info",
    "total_tokens_in_chat_history_32k",
    "total_tokens_in_chat_history_128k",
    "distance_from_related_snippet_to_query_32k",
    "distance_from_related_snippet_to_query_128k",
    "num_persona_relevant_tokens_32k",
    "num_persona_irrelevant_tokens_32k",
    "num_persona_relevant_tokens_128k",
    "num_persona_irrelevant_tokens_128k",
]


class PersonaMemLoaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "benchmark" / "text").mkdir(parents=True)
        (self.root / "data" / "chat_history_32k").mkdir(parents=True)
        (self.root / "data" / "chat_history_128k").mkdir(parents=True)
        self.csv_path = self.root / "benchmark" / "text" / "train.csv"
        self.history_32k = self.root / "data" / "chat_history_32k" / "persona7.json"
        self.history_128k = self.root / "data" / "chat_history_128k" / "persona7.json"
        self._write_history(
            self.history_32k,
            [
                {"role": "user", "content": "Please rewrite this note about my garden."},
                {"role": "assistant", "content": "Here is a polished version of the note."},
            ],
            wrapper_extra={"preference": "must never enter WriterInput"},
        )
        self._write_history(
            self.history_128k,
            [
                {"role": "user", "content": "This is the distinct 128k history."},
                {"role": "assistant", "content": "Understood."},
            ],
        )
        self.rows = [
            self._row(
                query="Which weekend activity would suit me best this month?",
                correct="A quiet morning tending a small herb garden would suit you well.",
                incorrect=[
                    "Attend a crowded motorsport festival for the entire weekend.",
                    "Spend the morning shopping for industrial power tools.",
                    "Book a loud nightclub tour that lasts until sunrise.",
                ],
            ),
            self._row(
                query="What kind of low-key present is likely to appeal to me?",
                correct="A compact set of heirloom herb seeds is a thoughtful choice.",
                incorrect=[
                    "A season ticket for professional drag-racing events.",
                    "A high-powered speaker system for large dance parties.",
                    "A collection of heavy machinery repair manuals.",
                ],
                updated="True",
            ),
        ]
        self._write_csv(self.csv_path, self.rows)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_history(self, path: Path, messages: list[dict], wrapper_extra=None) -> None:
        payload = {"chat_history": messages}
        payload.update(wrapper_extra or {})
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _row(
        self,
        *,
        query: str,
        correct: str,
        incorrect: list[str],
        updated: str = "False",
    ) -> dict[str, str]:
        return {
            "persona_id": "7",
            "chat_history_32k_link": "data/chat_history_32k/persona7.json",
            "chat_history_128k_link": "data/chat_history_128k/persona7.json",
            "user_query": json.dumps({"role": "user", "content": query}),
            "correct_answer": correct,
            "incorrect_answers": json.dumps(incorrect),
            "topic_query": "Lifestyle",
            "preference": "Likes small herb gardens",
            "topic_preference": "Gardening",
            "conversation_scenario": "personal_email",
            "pref_type": "neutral_preference",
            "related_conversation_snippet": "[annotation intentionally excluded]",
            "who": "self",
            "updated": updated,
            "prev_pref": "",
            "sensitive_info": "False",
            "total_tokens_in_chat_history_32k": "31000",
            "total_tokens_in_chat_history_128k": "125000",
            "distance_from_related_snippet_to_query_32k": "9000",
            "distance_from_related_snippet_to_query_128k": "100000",
            "num_persona_relevant_tokens_32k": "31000",
            "num_persona_irrelevant_tokens_32k": "0",
            "num_persona_relevant_tokens_128k": "31000",
            "num_persona_irrelevant_tokens_128k": "94000",
        }

    def _write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_groups_queries_and_writer_has_no_annotation_fields(self) -> None:
        dataset = pm.load_personamem_text(
            self.csv_path, split="train", window="32k", data_root=self.root
        )
        self.assertEqual(len(dataset.episodes), 1)
        episode = dataset.episodes[0]
        self.assertEqual(len(episode.questions), 2)
        self.assertEqual({item.name for item in fields(pm.WriterInput)}, {"messages"})
        self.assertEqual(
            set(episode.writer.to_messages()[0]),
            {"role", "content"},
        )
        writer_blob = episode.writer.to_text()
        self.assertNotIn("must never enter WriterInput", writer_blob)
        self.assertNotIn("Likes small herb gardens", writer_blob)
        self.assertNotIn("[annotation intentionally excluded]", writer_blob)

    def test_mcq_shuffle_is_deterministic_and_labels_remain_correct(self) -> None:
        first = pm.load_personamem_text(
            self.csv_path,
            split="train",
            window="32k",
            data_root=self.root,
            shuffle_seed=913,
            shuffle_round=4,
        )
        second = pm.load_personamem_text(
            self.csv_path,
            split="train",
            window="32k",
            data_root=self.root,
            shuffle_seed=913,
            shuffle_round=4,
        )
        first_questions = first.episodes[0].questions
        second_questions = second.episodes[0].questions
        self.assertEqual(first_questions, second_questions)
        for original_row, question in zip(self.rows, first_questions):
            self.assertEqual(
                question.reader.options[question.correct_index],
                original_row["correct_answer"],
            )

    def test_official_qwen_shuffle_uses_42_plus_original_csv_row(self) -> None:
        dataset = pm.load_personamem_text(
            self.csv_path,
            split="train",
            window="32k",
            data_root=self.root,
            option_shuffle_protocol="official_qwen",
            # These must be ignored by the fixed official protocol.
            shuffle_seed=999,
            shuffle_round=123,
        )
        for row_index, (row, question) in enumerate(
            zip(self.rows, dataset.episodes[0].questions)
        ):
            expected = [
                row["correct_answer"],
                *json.loads(row["incorrect_answers"]),
            ]
            random.Random(42 + row_index).shuffle(expected)
            self.assertEqual(list(question.reader.options), expected)
            self.assertEqual(question.source_row_index, row_index)
            self.assertEqual(
                question.reader.options[question.correct_index],
                row["correct_answer"],
            )

    def test_official_qwen_preserves_prompt_whitespace_but_sample_id_is_stable(
        self,
    ) -> None:
        rows = [dict(self.rows[0])]
        query = "  Preserve this query exactly.  "
        rows[0]["user_query"] = json.dumps(
            {"role": "user", "content": query}
        )
        rows[0]["correct_answer"] = " correct answer "
        rows[0]["incorrect_answers"] = json.dumps(
            [" wrong zero ", "wrong one", "wrong two "]
        )
        self._write_csv(self.csv_path, rows)

        official = pm.load_personamem_text(
            self.csv_path,
            split="train",
            window="32k",
            data_root=self.root,
            option_shuffle_protocol="official_qwen",
            content_overlap_policy="off",
        ).episodes[0].questions[0]
        legacy = pm.load_personamem_text(
            self.csv_path,
            split="train",
            window="32k",
            data_root=self.root,
            option_shuffle_protocol="stable",
            content_overlap_policy="off",
        ).episodes[0].questions[0]

        self.assertEqual(official.reader.query, query)
        self.assertIn(" correct answer ", official.reader.options)
        self.assertIn(" wrong zero ", official.reader.options)
        self.assertEqual(legacy.reader.query, query.strip())
        self.assertIn("correct answer", legacy.reader.options)
        # IDs remain canonical so clean-manifest exclusions do not depend on
        # whether a run uses the legacy or official presentation protocol.
        self.assertEqual(official.sample_id, legacy.sample_id)

    def test_window_selects_128k_history_and_metadata(self) -> None:
        dataset = pm.load_personamem_text(
            self.csv_path, split="train", window="128k", data_root=self.root
        )
        episode = dataset.episodes[0]
        self.assertIn("distinct 128k history", episode.writer.to_text())
        tags = episode.questions[0].tags
        self.assertEqual(tags.total_tokens, 125000)
        self.assertEqual(tags.persona_irrelevant_tokens, 94000)

    def test_current_query_in_history_is_a_hard_leakage_error(self) -> None:
        leaked_query = json.loads(self.rows[0]["user_query"])["content"]
        self._write_history(
            self.history_32k,
            [{"role": "user", "content": leaked_query}],
        )
        with self.assertRaises(pm.LeakageError):
            pm.load_personamem_text(
                self.csv_path, split="train", window="32k", data_root=self.root
            )

    def test_forbidden_key_on_message_is_a_hard_leakage_error(self) -> None:
        self._write_history(
            self.history_32k,
            [
                {
                    "role": "user",
                    "content": "Benign message text.",
                    "correct_answer": "annotation",
                }
            ],
        )
        with self.assertRaises(pm.LeakageError):
            pm.load_personamem_text(
                self.csv_path, split="train", window="32k", data_root=self.root
            )

    def test_empty_string_message_content_is_allowed(self) -> None:
        self._write_history(
            self.history_32k,
            [
                {"role": "user", "content": ""},
                {"role": "assistant"},
            ],
        )
        dataset = pm.load_personamem_text(
            self.csv_path, split="train", window="32k", data_root=self.root
        )
        self.assertEqual(dataset.episodes[0].writer.messages[0].content, "")
        self.assertEqual(dataset.episodes[0].writer.messages[1].content, "")

    def test_non_four_way_official_row_is_counted_and_skipped(self) -> None:
        invalid = dict(
            self.rows[0],
            user_query=json.dumps(
                {"role": "user", "content": "This row has no distractors."}
            ),
            incorrect_answers="[]",
        )
        self._write_csv(self.csv_path, [invalid, self.rows[1]])
        dataset = pm.load_personamem_text(
            self.csv_path, split="train", window="32k", data_root=self.root
        )
        self.assertEqual(dataset.num_questions, 1)
        self.assertEqual(dataset.rows_skipped_invalid_mcq, 1)
        self.assertEqual(
            pm.build_audit_report(dataset)["counts"]["rows_skipped_invalid_mcq"],
            1,
        )

    def test_explicit_persona_exclusion_is_counted(self) -> None:
        retained = dict(self.rows[0], persona_id="8")
        self._write_csv(self.csv_path, [*self.rows, retained])
        dataset = pm.load_personamem_text(
            self.csv_path,
            split="train",
            window="32k",
            data_root=self.root,
            exclude_persona_ids={"999", "7"},
        )
        self.assertEqual(dataset.num_questions, 1)
        self.assertEqual(dataset.rows_skipped_excluded_persona, 2)
        self.assertEqual(dataset.excluded_persona_ids, ("7", "999"))

    def test_audit_and_cross_split_persona_overlap(self) -> None:
        dataset = pm.load_personamem_text(
            self.csv_path, split="train", window="32k", data_root=self.root
        )
        report = pm.build_audit_report(dataset)
        self.assertEqual(report["counts"]["questions_loaded"], 2)
        self.assertEqual(report["counts"]["personas_loaded"], 1)
        self.assertEqual(
            report["leakage_audit"]["target_content_overlap_warning_count"], 0
        )
        val_path = self.root / "benchmark" / "text" / "val.csv"
        benchmark_path = self.root / "benchmark" / "text" / "benchmark.csv"
        val_row = dict(self.rows[0], persona_id="8")
        benchmark_row = dict(self.rows[0], persona_id="9")
        self._write_csv(val_path, [val_row])
        self._write_csv(benchmark_path, [benchmark_row])
        disjoint = pm.audit_split_disjointness(
            {"train": self.csv_path, "val": val_path, "benchmark": benchmark_path}
        )
        self.assertTrue(disjoint["all_disjoint"])
        self._write_csv(benchmark_path, [dict(benchmark_row, persona_id="7")])
        overlap = pm.audit_split_disjointness(
            {"train": self.csv_path, "val": val_path, "benchmark": benchmark_path}
        )
        self.assertFalse(overlap["all_disjoint"])
        self.assertEqual(overlap["overlap_persona_ids"]["benchmark__train"], ["7"])

    def test_cli_prints_and_saves_audit_json(self) -> None:
        audit_path = self.root / "audit" / "train32k.json"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            status = pm.main(
                [
                    "--data-root",
                    str(self.root),
                    "--split",
                    "train",
                    "--window",
                    "32k",
                    "--audit-json",
                    str(audit_path),
                ]
            )
        self.assertEqual(status, 0)
        printed = json.loads(stdout.getvalue())
        saved = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertEqual(printed, saved)
        self.assertEqual(saved["counts"]["questions_loaded"], 2)


if __name__ == "__main__":
    unittest.main()
