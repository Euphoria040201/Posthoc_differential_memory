from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from deltamem.core.global_prefix import SEG_ANS, SEG_CTX
from deltamem.core.prefix_steer import (
    PrefixMemSteerAttention,
    PrefixSteerConfig,
    freeze_backbone_keep_steer,
    set_mem_cache,
    set_steer_enabled,
    set_steer_segments,
)


class _TinyGQABaseAttention(nn.Module):
    """CPU Qwen-attention stand-in with four Q heads and two GQA KV heads."""

    def __init__(self) -> None:
        super().__init__()
        hidden_size = 16
        self.head_dim = 4
        self.num_key_value_groups = 2
        self.layer_idx = 0
        self.attention_dropout = 0.0
        self.scaling = self.head_dim**-0.5
        self.sliding_window = None
        self.config = SimpleNamespace(_attn_implementation="eager")
        self.q_proj = nn.Linear(hidden_size, 16, bias=False)
        self.k_proj = nn.Linear(hidden_size, 8, bias=False)
        self.v_proj = nn.Linear(hidden_size, 8, bias=False)
        self.o_proj = nn.Linear(16, hidden_size, bias=False)
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()


def _config(
    *,
    mem_heads: int = 1,
    fusion: str = "fixed",
    window: int = 3,
) -> PrefixSteerConfig:
    return PrefixSteerConfig(
        num_prefix_tokens=0,
        sliding_window_size=window,
        mem_num_heads=mem_heads,
        mem_head_dim=4,
        steer_mode="deltamem",
        normal_attends_prefix=False,
        prefix_write=False,
        read_prefix_only=False,
        pool_reads=False,
        share_qkv=False,
        memory_value_source="main_v",
        delta_heads="o",
        output_fusion=fusion,
        output_fusion_eps=1e-6,
        output_fusion_scale_max=10.0,
    )


def _make(
    *,
    mem_heads: int = 1,
    fusion: str = "fixed",
    window: int = 3,
) -> PrefixMemSteerAttention:
    torch.manual_seed(101)
    base = _TinyGQABaseAttention()
    attention = PrefixMemSteerAttention(
        base, _config(mem_heads=mem_heads, fusion=fusion, window=window), 16
    )
    holder = nn.Module()
    holder.attention = attention
    freeze_backbone_keep_steer(holder)
    return attention


def _segments(
    attention: PrefixMemSteerAttention, segment: int, batch: int, length: int
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


def _grouped_main_values(
    attention: PrefixMemSteerAttention, hidden: torch.Tensor
) -> torch.Tensor:
    batch, length, _ = hidden.shape
    raw = attention.base.v_proj(hidden).view(
        batch, length, attention.n_kv, attention.head_dim
    )
    raw = raw.transpose(1, 2)
    group = attention.n_kv // attention.cfg.mem_num_heads
    if group != 1:
        raw = raw.reshape(
            batch,
            attention.cfg.mem_num_heads,
            group,
            length,
            attention.head_dim,
        ).mean(dim=2)
    return raw


@pytest.mark.parametrize("mem_heads", [1, 2])
def test_main_v_reuses_grouped_frozen_gqa_values_without_value_adapter(
    mem_heads: int,
) -> None:
    attention = _make(mem_heads=mem_heads)
    hidden = torch.randn(2, 5, 16)
    queries, keys, values = attention._project_mem(hidden)

    assert queries.shape == (2, mem_heads, 5, 4)
    assert keys.shape == queries.shape
    torch.testing.assert_close(values, _grouped_main_values(attention, hidden))
    assert attention.read_dim == mem_heads * attention.head_dim
    assert attention.mem_v is None
    assert not any(".mem_v." in f".{key}." for key in attention.state_dict())


def test_main_v_validates_head_shape_and_gqa_grouping() -> None:
    with pytest.raises(ValueError, match="backbone head_dim"):
        PrefixMemSteerAttention(
            _TinyGQABaseAttention(),
            PrefixSteerConfig(
                num_prefix_tokens=0,
                mem_num_heads=1,
                mem_head_dim=2,
                prefix_write=False,
                pool_reads=False,
                memory_value_source="main_v",
                delta_heads="o",
            ),
            16,
        )
    with pytest.raises(ValueError, match="positive divisor"):
        PrefixMemSteerAttention(
            _TinyGQABaseAttention(),
            PrefixSteerConfig(
                num_prefix_tokens=0,
                mem_num_heads=3,
                mem_head_dim=4,
                prefix_write=False,
                pool_reads=False,
                memory_value_source="main_v",
                delta_heads="o",
            ),
            16,
        )
    with pytest.raises(ValueError, match="conflicts"):
        PrefixMemSteerAttention(
            _TinyGQABaseAttention(),
            PrefixSteerConfig(
                num_prefix_tokens=0,
                mem_num_heads=1,
                mem_head_dim=4,
                prefix_write=False,
                pool_reads=False,
                share_qkv=True,
                memory_value_source="main_v",
                delta_heads="o",
            ),
            16,
        )


def test_p_zero_memory_mask_is_exact_causal_sliding_window() -> None:
    attention = _make(mem_heads=1, window=3)
    attention.eval()
    with torch.no_grad():
        attention.mem_q.weight.zero_()
        attention.mem_k.weight.zero_()

    hidden = torch.randn(1, 6, 16)
    _segments(attention, SEG_CTX, 1, 6)
    reads = attention._memory_read(hidden)
    values = _grouped_main_values(attention, hidden)[:, 0]
    expected = torch.stack(
        [
            values[:, max(0, token - 2) : token + 1].mean(dim=1)
            for token in range(hidden.shape[1])
        ],
        dim=1,
    )
    torch.testing.assert_close(reads, expected, rtol=1e-6, atol=1e-7)
    assert attention.prefix.numel() == 0
    assert attention.write_proj is None


def test_main_v_cached_decode_matches_full_p_zero_swa() -> None:
    cached = _make(mem_heads=1, window=3).eval()
    full = _make(mem_heads=1, window=3).eval()
    full.load_state_dict(cached.state_dict(), strict=True)
    hidden = torch.randn(1, 5, 16)

    _segments(cached, SEG_CTX, 1, 4)
    set_mem_cache(cached, True)
    cached._memory_read(hidden[:, :4])
    _segments(cached, SEG_ANS, 1, 1)
    incremental = cached._memory_read(hidden[:, 4:5])

    _segments(full, SEG_CTX, 1, 5)
    expected = full._memory_read(hidden)[:, -1:]
    torch.testing.assert_close(incremental, expected, rtol=1e-6, atol=1e-7)
    assert cached._mem_kv is not None
    # No prefix K/V, exactly W normal-token K/V after appending the decode token.
    assert cached._mem_kv[0].shape[2] == 0
    assert cached._mem_kv[2].shape[2] == 3


@pytest.mark.parametrize("fusion", ["fixed", "rms_match", "cosine"])
def test_delta_o_zero_init_is_exact_base_and_has_live_gradient(fusion: str) -> None:
    attention = _make(mem_heads=1, fusion=fusion)
    attention.train()
    hidden = torch.randn(1, 5, 16)
    _segments(attention, SEG_CTX, 1, 5)

    set_steer_enabled(attention, False)
    expected = _forward(attention, hidden)
    set_steer_enabled(attention, True)
    actual = _forward(attention, hidden)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    actual.square().sum().backward()
    assert attention.delta_o is not None
    assert attention.delta_o.weight.grad is not None
    assert torch.count_nonzero(attention.delta_o.weight.grad).item() > 0


def test_shared_v_architecture_trains_only_mem_q_mem_k_and_delta_o() -> None:
    attention = _make(mem_heads=1, fusion="rms_match")
    nonempty_trainable = {
        name
        for name, parameter in attention.named_parameters()
        if parameter.requires_grad and parameter.numel() > 0
    }
    assert nonempty_trainable == {
        "mem_q.weight",
        "mem_k.weight",
        "delta_o.weight",
    }
    assert attention.delta_q is None
    assert attention.delta_k is None
    assert attention.delta_v is None
    assert attention.base.v_proj.weight.requires_grad is False

    # At zero-init only delta_o receives a gradient.  Once delta_o is non-zero, learning
    # reaches the independent side Q/K through the reused frozen main-V values.
    with torch.no_grad():
        attention.delta_o.weight.normal_(std=0.05)
    hidden = torch.randn(1, 5, 16)
    _segments(attention, SEG_CTX, 1, 5)
    _forward(attention, hidden).square().sum().backward()
    for projection in (attention.mem_q, attention.mem_k):
        assert projection.weight.grad is not None
        assert torch.count_nonzero(projection.weight.grad).item() > 0
    assert attention.base.v_proj.weight.grad is None


def test_cosine_fusion_coefficient_and_zero_vector_definition() -> None:
    attention = _make(fusion="cosine")
    out = torch.zeros(1, 4, 16)
    delta = torch.zeros_like(out)
    out[0, :, 0] = 1.0
    delta[0, 0, 0] = 1.0       # cosine +1 -> coefficient 1
    delta[0, 1, 0] = -1.0      # cosine -1 -> coefficient 0
    delta[0, 2, 1] = 1.0       # cosine  0 -> coefficient 1/2
    # delta[0,3] remains zero: defined cosine 0 -> coefficient 1/2, but adds zero.
    fused = attention._fuse_delta_o(out, delta)
    torch.testing.assert_close(fused[0, 0], out[0, 0] + delta[0, 0])
    torch.testing.assert_close(fused[0, 1], out[0, 1])
    torch.testing.assert_close(fused[0, 2], out[0, 2] + 0.5 * delta[0, 2])
    torch.testing.assert_close(fused[0, 3], out[0, 3])


def test_rms_fusion_is_stable_at_zero_and_becomes_sqrt2_normalized() -> None:
    attention = _make(fusion="rms_match")
    out = torch.randn(2, 3, 16)
    zero = torch.zeros_like(out)
    torch.testing.assert_close(
        attention._fuse_delta_o(out, zero), out, rtol=0.0, atol=0.0
    )

    # Equal non-zero RMS means scale~=1 and relative~=1, hence sqrt(2) energy
    # normalization (up to the documented epsilon).
    delta = out.clone()
    fused = attention._fuse_delta_o(out, delta)
    out_rms = out.float().square().mean(dim=-1, keepdim=True).sqrt()
    eps = attention.cfg.output_fusion_eps
    scale = ((out_rms + eps) / (out_rms + eps)).clamp(
        max=attention.cfg.output_fusion_scale_max
    )
    relative = scale * out_rms / (out_rms + eps)
    expected = (out + scale.to(out.dtype) * delta) / torch.sqrt(
        1.0 + relative.square()
    ).to(out.dtype)
    torch.testing.assert_close(fused, expected)


def test_legacy_config_defaults_and_strict_state_schema_are_unchanged() -> None:
    legacy = PrefixSteerConfig(
        num_prefix_tokens=2,
        mem_num_heads=2,
        mem_head_dim=2,
        prefix_write=True,
        read_prefix_only=True,
        pool_reads=False,
    )
    old_metadata = asdict(legacy)
    for field in (
        "memory_value_source",
        "output_fusion",
        "output_fusion_eps",
        "output_fusion_scale_max",
    ):
        old_metadata.pop(field)
    restored_config = PrefixSteerConfig(**old_metadata)
    assert restored_config.memory_value_source == "trainable"
    assert restored_config.output_fusion == "fixed"
    assert restored_config.output_fusion_eps == 1e-6
    assert restored_config.output_fusion_scale_max == 10.0

    source = PrefixMemSteerAttention(_TinyGQABaseAttention(), legacy, 16)
    restored = PrefixMemSteerAttention(
        _TinyGQABaseAttention(), restored_config, 16
    )
    assert source.mem_v is not None
    assert source.delta_q is not None
    assert source.delta_k is not None
    assert source.delta_v is not None
    restored.load_state_dict(source.state_dict(), strict=True)
