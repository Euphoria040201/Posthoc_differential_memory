from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from deltamem.core.global_prefix import SEG_CTX, SEG_QRY
from deltamem.core.prefix_steer import (
    PrefixMemSteerAttention,
    PrefixSteerConfig,
    clear_frozen_memory,
    has_frozen_memory,
    set_steer_enabled,
    set_steer_segments,
    set_window_only,
    set_write_only,
)


class _TinyBaseAttention(nn.Module):
    def __init__(self, hidden_size: int = 8, num_heads: int = 2) -> None:
        super().__init__()
        self.head_dim = hidden_size // num_heads
        self.num_key_value_groups = 1
        self.layer_idx = 0
        self.attention_dropout = 0.0
        self.scaling = self.head_dim**-0.5
        self.sliding_window = None
        self.config = SimpleNamespace(_attn_implementation="eager")
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()


def _make_attention(pool_mode: str = "attn") -> PrefixMemSteerAttention:
    torch.manual_seed(23)
    base = _TinyBaseAttention()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    config = PrefixSteerConfig(
        num_prefix_tokens=0,
        sliding_window_size=4,
        mem_num_heads=2,
        mem_head_dim=2,
        steer_mode="deltamem",
        normal_attends_prefix=False,
        prefix_write=False,
        read_prefix_only=False,
        pool_reads=False,
        history_pool_mode=pool_mode,
    )
    return PrefixMemSteerAttention(base, config, hidden_size=8)


def _segments(
    attention: PrefixMemSteerAttention,
    segment: int,
    batch: int,
    length: int,
) -> None:
    seg = torch.full((batch, length), segment, dtype=torch.long)
    set_steer_segments(attention, seg, torch.ones_like(seg, dtype=torch.bool))


def _forward(attention: PrefixMemSteerAttention, hidden: torch.Tensor) -> torch.Tensor:
    batch, length, _ = hidden.shape
    cos = torch.ones(batch, length, attention.head_dim)
    sin = torch.zeros_like(cos)
    mask = torch.zeros(batch, 1, length, length)
    output, _ = attention(
        hidden,
        position_embeddings=(cos, sin),
        attention_mask=mask,
    )
    return output


def _write(attention: PrefixMemSteerAttention, history: torch.Tensor) -> torch.Tensor:
    _segments(attention, SEG_CTX, history.shape[0], history.shape[1])
    set_write_only(attention, True)
    _forward(attention, history)
    set_write_only(attention, False)
    assert attention._frozen_history_pool is not None
    return attention._frozen_history_pool


def test_history_attention_pool_writes_once_on_pure_base_path_and_backprops() -> None:
    attention = _make_attention("attn")
    attention.train()
    assert attention.prefix.numel() == 0
    assert attention.write_proj is None
    assert attention.history_pool_query is not None

    history = torch.randn(1, 7, 8)
    _segments(attention, SEG_CTX, 1, 7)
    set_steer_enabled(attention, False)
    expected_base = _forward(attention, history)
    set_steer_enabled(attention, True)

    project_calls = 0
    original_project = attention._project_mem

    def counted_project(sequence: torch.Tensor):
        nonlocal project_calls
        project_calls += 1
        return original_project(sequence)

    attention._project_mem = counted_project
    _segments(attention, SEG_CTX, 1, 7)
    set_write_only(attention, True)
    history_output = _forward(attention, history)
    set_write_only(attention, False)
    pooled = attention._frozen_history_pool
    assert pooled is not None
    # Exactly one learned-query projection plus one history projection; no prefix READ.
    assert project_calls == 2
    torch.testing.assert_close(history_output, expected_base, rtol=0.0, atol=0.0)
    assert pooled.shape == (1, 1, attention.read_dim)
    assert pooled.requires_grad and pooled.grad_fn is not None
    assert has_frozen_memory(attention) == [True]

    query = torch.randn(3, 4, 8)
    _segments(attention, SEG_QRY, 3, 4)
    query_reads = attention._memory_read(query)
    torch.testing.assert_close(query_reads[:, :1], query_reads[:, 1:2])
    torch.testing.assert_close(query_reads[0], query_reads[1])
    _forward(attention, query).square().sum().backward()

    assert attention.history_pool_query.grad is not None
    assert torch.count_nonzero(attention.history_pool_query.grad).item() > 0
    for projection in (attention.mem_q, attention.mem_k, attention.mem_v):
        assert projection.weight.grad is not None
        assert torch.count_nonzero(projection.weight.grad).item() > 0


def test_history_pool_correct_swap_and_window_only_are_causal() -> None:
    attention = _make_attention("attn")
    attention.eval()
    first = _write(attention, torch.randn(1, 6, 8)).clone()
    assert has_frozen_memory(attention) == [True]

    query = torch.randn(2, 3, 8)
    _segments(attention, SEG_QRY, 2, 3)
    correct = attention._memory_read(query)
    torch.testing.assert_close(correct[0, 0], first[0, 0])

    second = _write(attention, torch.randn(1, 6, 8) + 4.0).clone()
    _segments(attention, SEG_QRY, 2, 3)
    swapped = attention._memory_read(query)
    assert not torch.allclose(first, second)
    assert not torch.allclose(correct, swapped)

    set_window_only(attention, True)
    torch.testing.assert_close(attention._memory_read(query), torch.zeros_like(swapped))
    set_window_only(attention, False)
    clear_frozen_memory(attention)
    assert has_frozen_memory(attention) == [False]
    with pytest.raises(RuntimeError, match="no memory"):
        attention._memory_read(query)


def test_mean_pool_masks_invalid_history_tokens() -> None:
    attention = _make_attention("mean")
    attention.eval()
    history = torch.randn(1, 4, 8)
    seg = torch.full((1, 4), SEG_CTX, dtype=torch.long)
    valid = torch.tensor([[True, True, False, False]])
    set_steer_segments(attention, seg, valid)
    set_write_only(attention, True)
    _forward(attention, history)
    set_write_only(attention, False)

    _, _, values = attention._project_mem(history)
    expected = values[:, :, :2].mean(dim=2).reshape(1, 1, attention.read_dim)
    torch.testing.assert_close(attention._frozen_history_pool, expected)
    assert attention.history_pool_query is None


def test_default_mode_adds_no_state_and_invalid_prefix_combination_fails() -> None:
    default = PrefixMemSteerAttention(
        _TinyBaseAttention(),
        PrefixSteerConfig(
            num_prefix_tokens=0,
            prefix_write=False,
            pool_reads=False,
            normal_attends_prefix=False,
        ),
        hidden_size=8,
    )
    assert default.cfg.history_pool_mode == "none"
    assert default.history_pool_query is None
    assert not any("history_pool" in key for key in default.state_dict())

    with pytest.raises(ValueError, match="P=0"):
        PrefixMemSteerAttention(
            _TinyBaseAttention(),
            PrefixSteerConfig(
                num_prefix_tokens=2,
                prefix_write=True,
                pool_reads=False,
                history_pool_mode="attn",
            ),
            hidden_size=8,
        )
