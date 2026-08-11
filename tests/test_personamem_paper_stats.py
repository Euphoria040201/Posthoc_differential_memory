from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import personamem_paper_stats as stats


def write_run(
    path: Path,
    rows: list[tuple[str, str, str, bool]],
) -> Path:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "persona_id": persona_id,
                    "correct_letter": gold,
                    "is_correct": is_correct,
                }
            )
            + "\n"
            for sample_id, persona_id, gold, is_correct in rows
        ),
        encoding="utf-8",
    )
    return path


def test_pairing_uses_sample_id_not_jsonl_order_and_mcnemar_is_exact(
    tmp_path: Path,
) -> None:
    reference_path = write_run(
        tmp_path / "reference.jsonl",
        [
            ("q1", "p1", "a", True),
            ("q2", "p1", "b", True),
            ("q3", "p2", "c", True),
            ("q4", "p2", "d", True),
            ("q5", "p3", "a", False),
        ],
    )
    comparison_path = write_run(
        tmp_path / "comparison.jsonl",
        [
            ("q5", "p3", "a", False),
            ("q3", "p2", "c", False),
            ("q1", "p1", "a", False),
            ("q4", "p2", "d", False),
            ("q2", "p1", "b", False),
        ],
    )
    result = stats.compare_runs(
        stats.load_outcome_run(reference_path),
        [stats.load_outcome_run(comparison_path)],
        bootstrap_replicates=200,
        bootstrap_seed=3,
    )

    assert result["accuracy"]["reference_micro"] == pytest.approx(0.8)
    assert result["accuracy"]["comparison_micro_mean_over_runs"] == 0.0
    assert result["accuracy"]["difference_micro"] == pytest.approx(-0.8)
    mcnemar = result["mcnemar_exact_by_comparison"][0]
    assert mcnemar["reference_correct_comparison_wrong"] == 4
    assert mcnemar["reference_wrong_comparison_correct"] == 0
    assert mcnemar["discordant_pairs"] == 4
    assert mcnemar["p_value"] == pytest.approx(0.125)


def test_default_bootstrap_is_mean_of_paired_persona_effects(
    tmp_path: Path,
) -> None:
    reference_path = write_run(
        tmp_path / "reference.jsonl",
        [
            ("small", "p1", "a", False),
            ("large1", "p2", "b", True),
            ("large2", "p2", "c", True),
            ("large3", "p2", "d", True),
        ],
    )
    comparison_path = write_run(
        tmp_path / "comparison.jsonl",
        [
            ("large3", "p2", "d", False),
            ("small", "p1", "a", True),
            ("large1", "p2", "b", False),
            ("large2", "p2", "c", False),
        ],
    )
    result = stats.compare_runs(
        stats.load_outcome_run(reference_path),
        [stats.load_outcome_run(comparison_path)],
        bootstrap_replicates=500,
        bootstrap_seed=7,
    )

    # Micro weights p2 three times: (1 - 3) / 4 = -0.5.
    assert result["accuracy"]["difference_micro"] == pytest.approx(-0.5)
    # Persona effects are +1 for p1 and -1 for p2: macro effect is zero.
    assert result["accuracy"]["difference_persona_macro"] == pytest.approx(0.0)
    bootstrap = result["paired_bootstrap"]
    assert bootstrap["unit"] == "persona"
    assert bootstrap["num_units"] == 2
    assert bootstrap["point_estimate"] == pytest.approx(0.0)
    assert bootstrap["ci_low"] == pytest.approx(-1.0)
    assert bootstrap["ci_high"] == pytest.approx(1.0)


def test_three_swaps_average_within_persona_before_bootstrap(
    tmp_path: Path,
) -> None:
    reference_path = write_run(
        tmp_path / "correct.jsonl",
        [
            ("q1", "p1", "a", True),
            ("q2", "p1", "b", False),
            ("q3", "p2", "c", True),
        ],
    )
    swap_paths = [
        write_run(
            tmp_path / "swap0.jsonl",
            [
                ("q1", "p1", "a", True),
                ("q2", "p1", "b", True),
                ("q3", "p2", "c", False),
            ],
        ),
        write_run(
            tmp_path / "swap1.jsonl",
            [
                ("q1", "p1", "a", False),
                ("q2", "p1", "b", True),
                ("q3", "p2", "c", False),
            ],
        ),
        write_run(
            tmp_path / "swap2.jsonl",
            [
                ("q1", "p1", "a", True),
                ("q2", "p1", "b", False),
                ("q3", "p2", "c", True),
            ],
        ),
    ]
    result = stats.compare_runs(
        stats.load_outcome_run(reference_path),
        [stats.load_outcome_run(path) for path in swap_paths],
        bootstrap_replicates=300,
        bootstrap_seed=11,
    )

    assert result["validation"]["num_comparison_runs"] == 3
    # Across all 9 comparison decisions: 5/9 vs reference 2/3.
    assert result["accuracy"]["difference_micro"] == pytest.approx(-1 / 9)
    # p1: 2/3 - 1/2 = 1/6; p2: 1/3 - 1 = -2/3.
    assert result["paired_bootstrap"]["point_estimate"] == pytest.approx(-0.25)
    assert result["accuracy"]["difference_persona_macro"] == pytest.approx(-0.25)
    assert len(result["mcnemar_exact_by_comparison"]) == 3
    assert "not pooled" in result["mcnemar_aggregation"]


@pytest.mark.parametrize(
    ("mutated_rows", "message"),
    [
        (
            [
                ("q1", "p1", "a", True),
                ("different", "p1", "b", False),
            ],
            "sample_id set differs",
        ),
        (
            [
                ("q1", "p1", "d", True),
                ("q2", "p1", "b", False),
            ],
            "gold label mismatch",
        ),
        (
            [
                ("q1", "other", "a", True),
                ("q2", "p1", "b", False),
            ],
            "persona_id mismatch",
        ),
    ],
)
def test_strictly_rejects_unpaired_or_changed_annotations(
    tmp_path: Path,
    mutated_rows: list[tuple[str, str, str, bool]],
    message: str,
) -> None:
    reference = stats.load_outcome_run(
        write_run(
            tmp_path / "reference.jsonl",
            [
                ("q1", "p1", "a", True),
                ("q2", "p1", "b", False),
            ],
        )
    )
    comparison = stats.load_outcome_run(
        write_run(tmp_path / "comparison.jsonl", mutated_rows)
    )
    with pytest.raises(ValueError, match=message):
        stats.compare_runs(reference, [comparison], bootstrap_replicates=10)


def test_loader_rejects_duplicate_samples_and_non_boolean_correctness(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "sample_id": "q",
                        "persona_id": "p",
                        "correct_letter": "a",
                        "is_correct": True,
                    }
                ),
                json.dumps(
                    {
                        "sample_id": "q",
                        "persona_id": "p",
                        "correct_letter": "a",
                        "is_correct": False,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate sample_id"):
        stats.load_outcome_run(duplicate)

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(
        json.dumps(
            {
                "sample_id": "q",
                "persona_id": "p",
                "correct_letter": "a",
                "is_correct": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="JSON boolean"):
        stats.load_outcome_run(invalid)


def test_cli_writes_json_and_refuses_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reference = write_run(
        tmp_path / "reference.jsonl", [("q", "p", "a", False)]
    )
    comparison = write_run(
        tmp_path / "comparison.jsonl", [("q", "p", "a", True)]
    )
    output = tmp_path / "stats.json"
    assert (
        stats.main(
            [
                "--reference",
                str(reference),
                "--comparison",
                str(comparison),
                "--bootstrap-replicates",
                "20",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    rendered = json.loads(output.read_text(encoding="utf-8"))
    assert rendered["accuracy"]["difference_micro"] == 1.0
    assert json.loads(capsys.readouterr().out) == rendered

    with pytest.raises(SystemExit, match="File exists"):
        stats.main(
            [
                "--reference",
                str(reference),
                "--comparison",
                str(comparison),
                "--bootstrap-replicates",
                "20",
                "--output",
                str(output),
            ]
        )
