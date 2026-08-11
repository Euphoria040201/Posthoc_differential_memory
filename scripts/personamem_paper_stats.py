#!/usr/bin/env python3
"""Paired paper statistics for PersonaMem-v2 per-example JSONL results.

The reference and every comparison file must contain exactly one record for
every ``sample_id``.  The script refuses duplicate/missing samples, changed
gold labels, changed persona assignments, and non-boolean correctness values.

The reported effect is always ``comparison - reference``.  By default, the
paired bootstrap estimand is the mean of per-persona accuracy differences.
When several comparison files are supplied (for example, the three fixed
history swaps), comparison accuracy is averaged across runs *within each
persona* before personas are resampled.  Exact McNemar tests remain separate
for each comparison run because pooling correlated swaps would not be an exact
McNemar test.

Example:

    python scripts/personamem_paper_stats.py \
      --reference steer_only.jsonl \
      --comparison steer_prefix.jsonl \
      --output paper_stats/steer_prefix_vs_steer.json

    python scripts/personamem_paper_stats.py \
      --reference correct_history.jsonl \
      --comparison swap0.jsonl swap1.jsonl swap2.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence


DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_730
DEFAULT_CONFIDENCE_LEVEL = 0.95


@dataclass(frozen=True, slots=True)
class Outcome:
    sample_id: str
    persona_id: str
    gold_label: Any
    gold_key: str
    is_correct: bool


@dataclass(frozen=True, slots=True)
class OutcomeRun:
    path: Path
    sha256: str
    outcomes: Mapping[str, Outcome]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"gold label is not a finite JSON value: {value!r}") from exc


def _required_string(
    record: Mapping[str, Any],
    field: str,
    *,
    path: Path,
    line_number: int,
) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{path}:{line_number}: {field} must be a non-empty string"
        )
    return value


def load_outcome_run(path: Path) -> OutcomeRun:
    """Load one strict, one-record-per-sample PersonaMem result file."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"result JSONL does not exist: {path}")

    outcomes: dict[str, Outcome] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise ValueError(f"{path}:{line_number}: blank JSONL record")
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: malformed JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"{path}:{line_number}: each JSONL record must be an object"
                )

            sample_id = _required_string(
                record, "sample_id", path=path, line_number=line_number
            )
            persona_id = _required_string(
                record, "persona_id", path=path, line_number=line_number
            )
            if "correct_letter" not in record or record["correct_letter"] is None:
                raise ValueError(
                    f"{path}:{line_number}: missing non-null correct_letter"
                )
            gold_label = record["correct_letter"]
            gold_key = _canonical_json(gold_label)
            is_correct = record.get("is_correct")
            if type(is_correct) is not bool:
                raise ValueError(
                    f"{path}:{line_number}: is_correct must be a JSON boolean"
                )
            if sample_id in outcomes:
                raise ValueError(
                    f"{path}:{line_number}: duplicate sample_id {sample_id!r}"
                )
            outcomes[sample_id] = Outcome(
                sample_id=sample_id,
                persona_id=persona_id,
                gold_label=gold_label,
                gold_key=gold_key,
                is_correct=is_correct,
            )

    if not outcomes:
        raise ValueError(f"result JSONL is empty: {path}")
    return OutcomeRun(path=path, sha256=file_sha256(path), outcomes=outcomes)


def _short_ids(values: set[str], limit: int = 5) -> list[str]:
    return sorted(values)[:limit]


def validate_paired_runs(
    reference: OutcomeRun,
    comparisons: Sequence[OutcomeRun],
) -> None:
    """Require exact sample, gold-label, and persona agreement."""

    if not comparisons:
        raise ValueError("at least one comparison JSONL is required")
    resolved = [run.path for run in comparisons]
    if len(set(resolved)) != len(resolved):
        raise ValueError("the same comparison JSONL was provided more than once")
    if reference.path in set(resolved):
        raise ValueError("reference JSONL cannot also be a comparison JSONL")

    reference_ids = set(reference.outcomes)
    for run in comparisons:
        comparison_ids = set(run.outcomes)
        if comparison_ids != reference_ids:
            missing = reference_ids - comparison_ids
            extra = comparison_ids - reference_ids
            raise ValueError(
                f"{run.path}: sample_id set differs from reference; "
                f"missing_count={len(missing)} missing={_short_ids(missing)}, "
                f"extra_count={len(extra)} extra={_short_ids(extra)}"
            )
        label_mismatches: list[str] = []
        persona_mismatches: list[str] = []
        for sample_id in reference_ids:
            left = reference.outcomes[sample_id]
            right = run.outcomes[sample_id]
            if left.gold_key != right.gold_key:
                label_mismatches.append(sample_id)
            if left.persona_id != right.persona_id:
                persona_mismatches.append(sample_id)
        if label_mismatches:
            raise ValueError(
                f"{run.path}: gold label mismatch for "
                f"{len(label_mismatches)} sample(s), including "
                f"{sorted(label_mismatches)[:5]}"
            )
        if persona_mismatches:
            raise ValueError(
                f"{run.path}: persona_id mismatch for "
                f"{len(persona_mismatches)} sample(s), including "
                f"{sorted(persona_mismatches)[:5]}"
            )


def exact_mcnemar(
    reference_correct: Sequence[bool],
    comparison_correct: Sequence[bool],
) -> dict[str, Any]:
    """Return the exact two-sided conditional-binomial McNemar test."""

    if len(reference_correct) != len(comparison_correct):
        raise ValueError("McNemar inputs must have equal lengths")
    if not reference_correct:
        raise ValueError("McNemar inputs cannot be empty")
    ref_correct_cmp_wrong = sum(
        left and not right
        for left, right in zip(reference_correct, comparison_correct)
    )
    ref_wrong_cmp_correct = sum(
        not left and right
        for left, right in zip(reference_correct, comparison_correct)
    )
    discordant = ref_correct_cmp_wrong + ref_wrong_cmp_correct
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(ref_correct_cmp_wrong, ref_wrong_cmp_correct)
        numerator = sum(math.comb(discordant, k) for k in range(tail + 1))
        # Keep the factor of two in integer arithmetic.  Converting a
        # 5,000-pair binomial numerator to float before division can overflow.
        p_value = min(1.0, (2 * numerator) / (1 << discordant))
    return {
        "test": "exact two-sided McNemar (conditional binomial)",
        "reference_correct_comparison_wrong": ref_correct_cmp_wrong,
        "reference_wrong_comparison_correct": ref_wrong_cmp_correct,
        "discordant_pairs": discordant,
        "p_value": p_value,
    }


def _persona_sample_ids(reference: OutcomeRun) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for sample_id, outcome in reference.outcomes.items():
        grouped.setdefault(outcome.persona_id, []).append(sample_id)
    return grouped


def _accuracy(values: Sequence[bool]) -> float:
    if not values:
        raise ValueError("accuracy requires at least one value")
    return fmean(int(value) for value in values)


def _per_unit_effects(
    reference: OutcomeRun,
    comparisons: Sequence[OutcomeRun],
    *,
    unit: str,
) -> list[float]:
    """Comparison-minus-reference effects after within-unit averaging."""

    if unit == "sample":
        return [
            fmean(
                int(run.outcomes[sample_id].is_correct) for run in comparisons
            )
            - int(reference.outcomes[sample_id].is_correct)
            for sample_id in reference.outcomes
        ]
    if unit != "persona":
        raise ValueError(f"unsupported bootstrap unit: {unit!r}")

    effects: list[float] = []
    for sample_ids in _persona_sample_ids(reference).values():
        reference_accuracy = fmean(
            int(reference.outcomes[sample_id].is_correct)
            for sample_id in sample_ids
        )
        comparison_accuracy = fmean(
            int(run.outcomes[sample_id].is_correct)
            for run in comparisons
            for sample_id in sample_ids
        )
        effects.append(comparison_accuracy - reference_accuracy)
    return effects


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must be in [0, 1]")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return (
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def paired_bootstrap(
    effects: Sequence[float],
    *,
    unit: str,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Percentile CI from resampling already-paired sample/persona effects."""

    if not effects:
        raise ValueError("paired bootstrap requires at least one unit")
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must be strictly between 0 and 1")

    rng = random.Random(seed)
    count = len(effects)
    draws = sorted(
        sum(effects[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(replicates)
    )
    alpha = 1.0 - confidence_level
    return {
        "method": "paired percentile bootstrap",
        "unit": unit,
        "estimand": (
            "mean per-persona comparison-minus-reference accuracy difference"
            if unit == "persona"
            else "mean per-sample comparison-minus-reference accuracy difference"
        ),
        "num_units": count,
        "replicates": replicates,
        "seed": seed,
        "confidence_level": confidence_level,
        "point_estimate": fmean(effects),
        "ci_low": _quantile(draws, alpha / 2.0),
        "ci_high": _quantile(draws, 1.0 - alpha / 2.0),
    }


def _run_provenance(run: OutcomeRun) -> dict[str, Any]:
    return {
        "path": str(run.path),
        "sha256": run.sha256,
        "n": len(run.outcomes),
    }


def compare_runs(
    reference: OutcomeRun,
    comparisons: Sequence[OutcomeRun],
    *,
    bootstrap_unit: str = "persona",
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Validate paired runs and compute paper-facing statistics."""

    validate_paired_runs(reference, comparisons)
    sample_ids = list(reference.outcomes)
    personas = _persona_sample_ids(reference)
    reference_values = [
        reference.outcomes[sample_id].is_correct for sample_id in sample_ids
    ]
    comparison_values_by_run = [
        [run.outcomes[sample_id].is_correct for sample_id in sample_ids]
        for run in comparisons
    ]

    reference_micro = _accuracy(reference_values)
    comparison_micro = fmean(
        _accuracy(values) for values in comparison_values_by_run
    )
    reference_persona_macro = fmean(
        _accuracy(
            [reference.outcomes[sample_id].is_correct for sample_id in ids]
        )
        for ids in personas.values()
    )
    comparison_persona_macro = fmean(
        fmean(
            int(run.outcomes[sample_id].is_correct)
            for run in comparisons
            for sample_id in ids
        )
        for ids in personas.values()
    )

    effects = _per_unit_effects(
        reference, comparisons, unit=bootstrap_unit
    )
    mcnemar_by_run = []
    for run, values in zip(comparisons, comparison_values_by_run):
        mcnemar_by_run.append(
            {
                "comparison_path": str(run.path),
                **exact_mcnemar(reference_values, values),
            }
        )

    return {
        "schema_version": 1,
        "effect_direction": "comparison_minus_reference",
        "provenance": {
            "reference": _run_provenance(reference),
            "comparisons": [_run_provenance(run) for run in comparisons],
        },
        "validation": {
            "paired_by": "sample_id",
            "gold_label_field": "correct_letter",
            "correctness_field": "is_correct",
            "exact_sample_sets": True,
            "exact_gold_labels": True,
            "exact_persona_assignments": True,
            "num_samples": len(sample_ids),
            "num_personas": len(personas),
            "num_comparison_runs": len(comparisons),
        },
        "accuracy": {
            "reference_micro": reference_micro,
            "comparison_micro_mean_over_runs": comparison_micro,
            "difference_micro": comparison_micro - reference_micro,
            "reference_persona_macro": reference_persona_macro,
            "comparison_persona_macro_mean_over_runs": (
                comparison_persona_macro
            ),
            "difference_persona_macro": (
                comparison_persona_macro - reference_persona_macro
            ),
        },
        "paired_bootstrap": paired_bootstrap(
            effects,
            unit=bootstrap_unit,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            confidence_level=confidence_level,
        ),
        "mcnemar_exact_by_comparison": mcnemar_by_run,
        "mcnemar_aggregation": (
            "single exact paired test"
            if len(comparisons) == 1
            else (
                "reported separately per comparison run; correlated repeated "
                "swaps are not pooled"
            )
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--comparison",
        type=Path,
        nargs="+",
        required=True,
        help="one method result, or multiple repeated/swap result JSONLs",
    )
    parser.add_argument(
        "--bootstrap-unit",
        choices=("persona", "sample"),
        default="persona",
        help="paired resampling unit (default: persona cluster)",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED
    )
    parser.add_argument(
        "--confidence-level", type=float, default=DEFAULT_CONFIDENCE_LEVEL
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON output; existing files are never overwritten",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        reference = load_outcome_run(args.reference)
        comparisons = [
            load_outcome_run(path)
            for path in args.comparison
        ]
        result = compare_runs(
            reference,
            comparisons,
            bootstrap_unit=args.bootstrap_unit,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            confidence_level=args.confidence_level,
        )
        rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        if args.output is not None:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                handle.write(rendered)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
