from __future__ import annotations

import pytest
import torch

from investigation.personamem_state_probe import (
    cosine_stats,
    entropy_effective_rank,
    summarize_all_states,
    summarize_logit_pairs,
)


def test_cosine_stats_and_entropy_effective_rank() -> None:
    identical = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    assert cosine_stats(identical)["mean"] == pytest.approx(1.0)
    orthogonal = torch.eye(2)
    assert cosine_stats(orthogonal)["mean"] == pytest.approx(0.0)
    assert entropy_effective_rank(orthogonal) == pytest.approx(2.0)
    assert entropy_effective_rank(identical) == pytest.approx(1.0)


def test_state_summary_reports_layer_and_concatenated_persona_cosine() -> None:
    states = {
        3: {
            "a": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "b": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        },
        7: {
            "a": torch.tensor([[1.0, 1.0]]),
            "b": torch.tensor([[-1.0, -1.0]]),
        },
    }
    summary = summarize_all_states(states)
    assert set(summary["per_layer"]) == {"3", "7"}
    assert summary["per_layer"]["3"]["slots"] == 2
    assert summary["per_layer"]["3"]["effective_rank_mean"] == pytest.approx(2.0)
    assert summary["per_layer"]["7"]["slot_cosine_mean"] is None
    assert summary["overall"]["num_personas"] == 2


def test_logit_pair_summary_uses_forced_choice_predictions() -> None:
    correct = torch.tensor(
        [[4.0, 1.0, 0.0, -1.0], [0.0, 3.0, 1.0, -2.0]]
    )
    swapped = torch.tensor(
        [[0.0, 5.0, 1.0, -1.0], [0.0, 2.0, 4.0, -2.0]]
    )
    gold = torch.tensor([0, 1])
    summary = summarize_logit_pairs(correct, swapped, gold)
    assert summary["correct_history_accuracy"] == pytest.approx(1.0)
    assert summary["swap_history_accuracy"] == pytest.approx(0.0)
    assert summary["prediction_change_rate"] == pytest.approx(1.0)
    assert summary["mean_signed_gold_logit_delta"] == pytest.approx(2.5)
