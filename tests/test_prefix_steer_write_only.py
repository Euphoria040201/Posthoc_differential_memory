from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from deltamem.core.global_prefix import SEG_CTX, SEG_QRY
from deltamem.core.prefix_steer import (
    PrefixMemSteerAttention,
    PrefixSteerConfig,
    set_steer_enabled,
    set_steer_segments,
    set_write_freeze,
    set_write_only,
)


class _TinyBaseAttention(nn.Module):
    """CPU-only stand-in exposing the Qwen3Attention attributes the wrapper uses."""

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


def _make_attention(
    write_layout: str = "global",
) -> PrefixMemSteerAttention:
    torch.manual_seed(17)
    base = _TinyBaseAttention()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    config = PrefixSteerConfig(
        num_prefix_tokens=3,
        sliding_window_size=4,
        mem_num_heads=2,
        mem_head_dim=2,
        steer_mode="residual",
        prefix_sees_query=False,
        prefix_write=True,
        read_prefix_only=True,
        memory_mode="dynamic",
        write_ctx_only=True,
        prefix_write_layout=write_layout,
        pool_reads=False,
    )
    return PrefixMemSteerAttention(base, config, hidden_size=8)


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


def _set_segment(
    attention: PrefixMemSteerAttention, segment: int, length: int
) -> None:
    seg = torch.full((1, length), segment, dtype=torch.long)
    set_steer_segments(attention, seg, torch.ones_like(seg, dtype=torch.bool))


def test_write_only_is_pure_base_and_query_backpropagates_into_writer() -> None:
    attention = _make_attention()
    attention.train()
    history = torch.randn(1, 6, 8)
    _set_segment(attention, SEG_CTX, history.shape[1])

    # The disabled-steer wrapper is the exact frozen-base reference path.
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
    set_write_only(attention, True)
    history_output = _forward(attention, history)

    # Strict writer uses exactly two projections (prefix Q, history K/V).  A READ
    # would add a third projection over [written prefix ; history].
    assert project_calls == 2
    torch.testing.assert_close(history_output, expected_base, rtol=0.0, atol=0.0)
    written = attention._frozen_prefix
    assert written is not None
    assert written.grad_fn is not None
    assert written.requires_grad

    set_write_only(attention, False)
    assert attention._frozen_prefix is written
    query = torch.randn(1, 4, 8)
    _set_segment(attention, SEG_QRY, query.shape[1])
    query_output = _forward(attention, query)
    query_output.square().sum().backward()

    assert attention.write_proj is not None
    assert attention.write_proj.weight.grad is not None
    assert torch.count_nonzero(attention.write_proj.weight.grad).item() > 0
    assert attention.prefix.grad is not None
    assert torch.count_nonzero(attention.prefix.grad).item() > 0


def test_write_only_is_default_off_runtime_state_and_clears_stale_memory() -> None:
    attention = _make_attention()
    assert attention._write_only is False
    assert not any("write_only" in key for key in attention.state_dict())

    attention._frozen_prefix = torch.randn(1, 3, 8)
    dummy = torch.randn(1, 2, 1, 2)
    attention._mem_kv = (dummy, dummy, dummy, dummy)
    attention._mem_norm_len = 9

    set_write_only(attention, True)
    assert attention._write_only is True
    assert attention._frozen_prefix is None
    assert attention._mem_kv is None
    assert attention._mem_norm_len == 0

    fresh = torch.randn(1, 3, 8, requires_grad=True)
    attention._frozen_prefix = fresh
    set_write_only(attention, False)
    assert attention._write_only is False
    assert attention._frozen_prefix is fresh


def test_default_read_path_and_strict_checkpoint_loading_are_unchanged() -> None:
    source = _make_attention()
    checkpoint = source.state_dict()
    restored = _make_attention()
    restored.load_state_dict(checkpoint, strict=True)

    restored.train()
    history = torch.randn(1, 5, 8)
    _set_segment(restored, SEG_CTX, history.shape[1])
    project_calls = 0
    original_project = restored._project_mem

    def counted_project(sequence: torch.Tensor):
        nonlocal project_calls
        project_calls += 1
        return original_project(sequence)

    restored._project_mem = counted_project
    set_write_freeze(restored, True)
    output = _forward(restored, history)

    # Legacy/default mode still performs WRITE (two projections) followed by READ
    # (one projection), and still supports the existing set_write_freeze protocol.
    assert restored._write_only is False
    assert project_calls == 3
    assert output.shape == history.shape
    assert restored._frozen_prefix is not None


def test_legacy_prefix_only_writer_uses_partitioned_layout() -> None:
    attention = _make_attention("partitioned")
    attention.eval()
    history = torch.zeros(1, 6, 8)
    history[0, 0:2, 0] = 1.0
    history[0, 2:4, 0] = 10.0
    history[0, 4:6, 0] = 100.0
    with torch.no_grad():
        attention.mem_q.weight.zero_()
        attention.mem_k.weight.zero_()
        attention.mem_v.weight.zero_()
        attention.write_proj.weight.zero_()
        for index in range(attention.read_dim):
            attention.mem_v.weight[index, index] = 1.0
            attention.write_proj.weight[index, index] = 1.0

    _set_segment(attention, SEG_CTX, 6)
    set_write_only(attention, True)
    _forward(attention, history)
    set_write_only(attention, False)
    torch.testing.assert_close(
        attention._frozen_prefix[0, :, 0],
        torch.tensor([1.0, 10.0, 100.0]),
        rtol=0.0,
        atol=0.0,
    )
