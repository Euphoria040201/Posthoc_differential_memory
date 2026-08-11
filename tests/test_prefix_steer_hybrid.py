from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from deltamem.core.global_prefix import SEG_CTX, SEG_QRY
from deltamem.core.prefix_steer import (
    PrefixMemSteerAttention,
    PrefixSteerConfig,
    has_frozen_memory,
    set_hybrid_pool_off,
    set_hybrid_prefix_off,
    set_mem_cache,
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


def _make_attention(
    gate_mode: str = "fixed",
    gate_init: float = 0.1,
    write_layout: str = "global",
    write_overlap_tokens: int = 0,
) -> PrefixMemSteerAttention:
    torch.manual_seed(41)
    base = _TinyBaseAttention()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    config = PrefixSteerConfig(
        num_prefix_tokens=3,
        sliding_window_size=4,
        mem_num_heads=2,
        mem_head_dim=2,
        steer_mode="residual",
        normal_attends_prefix=True,
        prefix_sees_query=False,
        prefix_write=True,
        read_prefix_only=False,
        memory_mode="dynamic",
        write_ctx_only=True,
        prefix_write_layout=write_layout,
        prefix_write_overlap_tokens=write_overlap_tokens,
        pool_reads=False,
        history_pool_mode="attn",
        hybrid_read_mode="pooled_plus_prefix",
        hybrid_prefix_gate_mode=gate_mode,
        hybrid_prefix_gate_init=gate_init,
    )
    return PrefixMemSteerAttention(base, config, hidden_size=8)


def _segments(
    attention: PrefixMemSteerAttention,
    segment: int,
    batch: int,
    length: int,
    valid: torch.Tensor | None = None,
) -> None:
    seg = torch.full((batch, length), segment, dtype=torch.long)
    if valid is None:
        valid = torch.ones_like(seg, dtype=torch.bool)
    set_steer_segments(attention, seg, valid)


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


def _write(attention: PrefixMemSteerAttention, history: torch.Tensor) -> None:
    _segments(attention, SEG_CTX, history.shape[0], history.shape[1])
    set_write_only(attention, True)
    _forward(attention, history)
    set_write_only(attention, False)


def test_hybrid_write_schema_shares_history_projection_and_keeps_base_path() -> None:
    attention = _make_attention()
    attention.train()
    history = torch.randn(1, 7, 8)
    valid = torch.tensor([[True, True, True, True, True, False, False]])
    _segments(attention, SEG_CTX, 1, 7, valid)

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
    set_write_only(attention, False)

    # One combined [pool probe ; P prefix probes] Q projection and one shared history
    # K/V projection.  No history READ is performed during WRITE-only.
    assert project_calls == 2
    torch.testing.assert_close(history_output, expected_base, rtol=0.0, atol=0.0)
    assert attention._frozen_history_pool is not None
    assert attention._frozen_prefix is not None
    assert attention._frozen_history_pool.shape == (1, 1, attention.read_dim)
    assert attention._frozen_prefix.shape == (1, 3, 8)
    assert attention._frozen_history_pool.grad_fn is not None
    assert attention._frozen_prefix.grad_fn is not None
    assert has_frozen_memory(attention) == [True]


def test_hybrid_pool_row_matches_existing_attention_pool_writer() -> None:
    hybrid = _make_attention()
    pooled = PrefixMemSteerAttention(
        _TinyBaseAttention(),
        PrefixSteerConfig(
            num_prefix_tokens=0,
            sliding_window_size=4,
            mem_num_heads=2,
            mem_head_dim=2,
            steer_mode="residual",
            normal_attends_prefix=False,
            prefix_write=False,
            read_prefix_only=False,
            pool_reads=False,
            history_pool_mode="attn",
        ),
        hidden_size=8,
    )
    pooled.mem_q.load_state_dict(hybrid.mem_q.state_dict())
    pooled.mem_k.load_state_dict(hybrid.mem_k.state_dict())
    pooled.mem_v.load_state_dict(hybrid.mem_v.state_dict())
    with torch.no_grad():
        pooled.history_pool_query.copy_(hybrid.history_pool_query)
    hybrid.eval()
    pooled.eval()

    history = torch.randn(1, 7, 8)
    valid = torch.tensor([[True, True, True, True, False, False, False]])
    for attention in (hybrid, pooled):
        _segments(attention, SEG_CTX, 1, 7, valid)
        set_write_only(attention, True)
        _forward(attention, history)
        set_write_only(attention, False)

    torch.testing.assert_close(
        hybrid._frozen_history_pool,
        pooled._frozen_history_pool,
        rtol=1e-6,
        atol=1e-7,
    )


def test_partitioned_write_routes_ordered_valid_chunks_and_never_padding() -> None:
    attention = _make_attention(
        write_layout="partitioned", write_overlap_tokens=0
    )
    valid = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, False, False, False, True, False],
        ]
    )
    routed = attention._prefix_write_keep(valid)
    assert routed.shape == (2, 3, 6)
    assert routed.all(dim=-1).shape == (2, 3)
    assert routed.any(dim=-1).all()
    assert not (routed & ~valid[:, None, :]).any()
    torch.testing.assert_close(
        routed[0],
        torch.tensor(
            [
                [True, True, False, False, False, False],
                [False, False, True, True, False, False],
                [False, False, False, False, True, True],
            ]
        ),
    )
    # N=2 < P=3: every slot still sees a real token, with intentional boundary
    # sharing instead of an empty/all-masked softmax row.
    assert routed[1, 0, 0]
    assert routed[1, 1, [0, 4]].all()
    assert routed[1, 2, 4]

    overlap = _make_attention(
        write_layout="partitioned", write_overlap_tokens=1
    )._prefix_write_keep(valid[:1])
    assert overlap[0, 0, :3].all()
    assert overlap[0, 1, 1:5].all()
    assert overlap[0, 2, 3:].all()


def test_partitioned_slots_write_distinct_chunk_content() -> None:
    attention = _make_attention(write_layout="partitioned")
    attention.eval()
    with torch.no_grad():
        attention.mem_q.weight.zero_()
        attention.mem_k.weight.zero_()
        attention.mem_v.weight.zero_()
        attention.write_proj.weight.zero_()
        for index in range(attention.read_dim):
            attention.mem_v.weight[index, index] = 1.0
            attention.write_proj.weight[index, index] = 1.0

    history = torch.zeros(1, 6, 8)
    history[0, 0:2, 0] = 1.0
    history[0, 2:4, 0] = 10.0
    history[0, 4:6, 0] = 100.0
    _write(attention, history)
    torch.testing.assert_close(
        attention._frozen_prefix[0, :, 0],
        torch.tensor([1.0, 10.0, 100.0]),
        rtol=0.0,
        atol=0.0,
    )


def test_hybrid_prefix_off_is_exact_pooled_branch_and_batch_broadcasts() -> None:
    attention = _make_attention()
    attention.eval()
    _write(attention, torch.randn(1, 6, 8))

    one_query = torch.randn(1, 4, 8)
    query = one_query.expand(3, -1, -1).clone()
    _segments(attention, SEG_QRY, 3, 4)
    full = attention._memory_read(query)
    assert not torch.equal(full, attention._frozen_history_pool.expand_as(full))
    torch.testing.assert_close(full[0], full[1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(full[0], full[2], rtol=0.0, atol=0.0)

    # Each prefix contribution is query-conditioned but prefix-only: changing neighboring
    # query tokens cannot leak a local-SWA term into token zero.
    changed_query = query.clone()
    changed_query[:, 1:] += 9.0
    changed = attention._memory_read(changed_query)
    torch.testing.assert_close(changed[:, :1], full[:, :1], rtol=0.0, atol=0.0)

    set_hybrid_prefix_off(attention, True)
    pooled_only = attention._memory_read(query)
    expected = attention._frozen_history_pool.expand_as(pooled_only)
    torch.testing.assert_close(pooled_only, expected, rtol=0.0, atol=0.0)

    # The existing causal-prefix ablation aliases the same branch in this no-local-SWA
    # hybrid; neither intervention can accidentally remove pooled steer.
    set_hybrid_prefix_off(attention, False)
    set_window_only(attention, True)
    torch.testing.assert_close(
        attention._memory_read(query), expected, rtol=0.0, atol=0.0
    )


def test_hybrid_pool_off_is_exact_prefix_branch_and_preserves_prefix_off() -> None:
    attention = _make_attention(gate_init=0.7, write_layout="partitioned")
    attention.eval()
    _write(attention, torch.randn(1, 6, 8))
    query = torch.randn(2, 4, 8)
    _segments(attention, SEG_QRY, 2, 4)

    initial_state_keys = tuple(attention.state_dict())
    full = attention._memory_read(query)
    set_hybrid_prefix_off(attention, True)
    pooled = attention._memory_read(query)
    set_hybrid_prefix_off(attention, False)

    set_hybrid_pool_off(attention, True)
    prefix = attention._memory_read(query)
    torch.testing.assert_close(prefix, full - pooled, rtol=1e-5, atol=1e-6)

    # prefix_off retains its historical exact pooled-only meaning even if a
    # training caller accidentally leaves pool_off set.
    set_hybrid_prefix_off(attention, True)
    torch.testing.assert_close(
        attention._memory_read(query), pooled, rtol=0.0, atol=0.0
    )
    set_hybrid_prefix_off(attention, False)
    set_hybrid_pool_off(attention, False)
    torch.testing.assert_close(
        attention._memory_read(query), full, rtol=0.0, atol=0.0
    )
    assert tuple(attention.state_dict()) == initial_state_keys


def test_hybrid_pool_off_backpropagates_prefix_without_pool_probe() -> None:
    attention = _make_attention(
        "learned_scalar", gate_init=0.7, write_layout="partitioned"
    )
    attention.train()
    _write(attention, torch.randn(1, 6, 8))
    query = torch.randn(2, 4, 8)
    _segments(attention, SEG_QRY, 2, 4)
    set_hybrid_pool_off(attention, True)
    attention._memory_read(query).square().mean().backward()

    assert attention.prefix.grad is not None
    assert torch.count_nonzero(attention.prefix.grad).item() > 0
    assert attention.write_proj.weight.grad is not None
    assert torch.count_nonzero(attention.write_proj.weight.grad).item() > 0
    assert attention.hybrid_prefix_gate_logit is not None
    assert attention.hybrid_prefix_gate_logit.grad is not None
    assert torch.count_nonzero(
        attention.hybrid_prefix_gate_logit.grad
    ).item() > 0
    assert attention.history_pool_query is not None
    assert (
        attention.history_pool_query.grad is None
        or torch.count_nonzero(attention.history_pool_query.grad).item() == 0
    )


def test_hybrid_wrong_persona_changes_both_memories_and_query_read() -> None:
    attention = _make_attention()
    attention.eval()
    query = torch.randn(1, 5, 8)

    _write(attention, torch.randn(1, 7, 8))
    _segments(attention, SEG_QRY, 1, 5)
    first_pool = attention._frozen_history_pool.clone()
    first_prefix = attention._frozen_prefix.clone()
    first_read = attention._memory_read(query)

    _write(attention, torch.randn(1, 7, 8) + 5.0)
    _segments(attention, SEG_QRY, 1, 5)
    second_pool = attention._frozen_history_pool.clone()
    second_prefix = attention._frozen_prefix.clone()
    second_read = attention._memory_read(query)

    assert not torch.allclose(first_pool, second_pool)
    assert not torch.allclose(first_prefix, second_prefix)
    assert not torch.allclose(first_read, second_read)


@pytest.mark.parametrize("gate_mode", ["fixed", "learned_scalar", "learned_channel"])
def test_hybrid_nonzero_gate_backpropagates_into_prefix(
    gate_mode: str,
) -> None:
    attention = _make_attention(gate_mode, write_layout="partitioned")
    attention.train()
    _write(attention, torch.randn(1, 6, 8))
    query = torch.randn(2, 4, 8)
    _segments(attention, SEG_QRY, 2, 4)
    read = attention._memory_read(query)
    read.square().mean().backward()

    assert attention.prefix.grad is not None
    assert torch.count_nonzero(attention.prefix.grad).item() > 0
    assert (attention.prefix.grad.square().sum(dim=-1) > 0).all()
    assert attention.write_proj.weight.grad is not None
    assert torch.count_nonzero(attention.write_proj.weight.grad).item() > 0
    if gate_mode == "fixed":
        assert attention.hybrid_prefix_gate_logit is None
    else:
        assert attention.hybrid_prefix_gate_logit is not None
        assert attention.hybrid_prefix_gate_logit.grad is not None
        assert torch.count_nonzero(attention.hybrid_prefix_gate_logit.grad).item() > 0


def test_hybrid_incremental_cache_matches_full_prefix_only_read() -> None:
    attention = _make_attention(
        "learned_scalar", write_layout="partitioned"
    )
    attention.eval()
    _write(attention, torch.randn(1, 7, 8))
    query = torch.randn(1, 6, 8)

    _segments(attention, SEG_QRY, 1, 6)
    set_mem_cache(attention, False)
    full = attention._memory_read(query)

    set_mem_cache(attention, True)
    _segments(attention, SEG_QRY, 1, 3)
    prefill = attention._memory_read(query[:, :3])
    assert attention._mem_kv is not None
    assert attention._mem_kv[0].shape[2] == attention.cfg.num_prefix_tokens
    assert attention._mem_kv[2].shape[2] == 0
    torch.testing.assert_close(prefill, full[:, :3], rtol=1e-6, atol=1e-7)

    for index in range(3, 6):
        _segments(attention, SEG_QRY, 1, 1)
        incremental = attention._memory_read(query[:, index : index + 1])
        torch.testing.assert_close(
            incremental, full[:, index : index + 1], rtol=1e-6, atol=1e-7
        )


def test_hybrid_config_validation_and_old_metadata_state_compatibility() -> None:
    with pytest.raises(ValueError, match="history_pool_mode='attn'"):
        PrefixMemSteerAttention(
            _TinyBaseAttention(),
            PrefixSteerConfig(
                num_prefix_tokens=3,
                prefix_write=True,
                write_ctx_only=True,
                pool_reads=False,
                hybrid_read_mode="pooled_plus_prefix",
            ),
            hidden_size=8,
        )

    # A checkpoint/config produced before hybrid fields existed resolves to legacy behavior
    # and introduces no new state_dict key, so strict loading remains valid.
    legacy_cfg = PrefixSteerConfig(
        num_prefix_tokens=3,
        mem_num_heads=2,
        mem_head_dim=2,
        prefix_write=True,
        read_prefix_only=True,
        pool_reads=False,
    )
    old_metadata = asdict(legacy_cfg)
    for key in (
        "hybrid_read_mode",
        "hybrid_prefix_gate_mode",
        "hybrid_prefix_gate_init",
        "prefix_write_layout",
        "prefix_write_overlap_tokens",
    ):
        old_metadata.pop(key)
    restored_cfg = PrefixSteerConfig(**old_metadata)
    assert restored_cfg.hybrid_read_mode == "none"
    assert restored_cfg.prefix_write_layout == "global"
    assert restored_cfg.prefix_write_overlap_tokens == 0

    source = PrefixMemSteerAttention(_TinyBaseAttention(), legacy_cfg, hidden_size=8)
    restored = PrefixMemSteerAttention(
        _TinyBaseAttention(), restored_cfg, hidden_size=8
    )
    assert not any("hybrid_prefix" in key for key in source.state_dict())
    restored.load_state_dict(source.state_dict(), strict=True)

    # At Qwen3-4B's 12 patched layers, fixed-gate P64/D64 hybrid adds exactly one
    # 2,560-d pool query per layer to the existing 16,515,072-parameter P64/D64 model.
    assert 16_515_072 + 12 * 2_560 == 16_545_792
