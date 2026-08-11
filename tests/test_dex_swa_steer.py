"""Wiring tests for the ``swa_steer`` variant.

The variant composes two wrappers that were written independently: the memory
sidecar (``PrefixMemSteerAttention``, which REPLACES the attention module) and
the DEX output wrapper (``DexOutputProjection``, which replaces ``o_proj``
inside it).  What has to hold is that the backbone stays frozen, that the
sidecar is the entire trainable set, and that gradients actually reach it.
"""
from __future__ import annotations

import pytest
import torch

from deltamem.core.dex import DexConfig, attach_dex, set_trainable, trainable_report
from deltamem.core.prefix_steer import (
    PrefixSteerConfig,
    attach_prefix_steer,
    is_steer_param_name,
    set_steer_segments,
)
from tests.test_dex import HEAD_DIM, tiny_model


def steer_config(**kw):
    base = dict(
        num_prefix_tokens=0,          # window-only branch, as in the SWA line
        sliding_window_size=8,
        mem_num_heads=1,
        mem_head_dim=HEAD_DIM,
        steer_mode="deltamem",
        memory_mode="dynamic",
        memory_value_source="main_v",
        delta_heads="o",
        steer_gain=0.1,
        output_fusion="fixed",
        steer_layers=(0, 1),
        prefix_write=False,
        write_ctx_only=False,
        read_prefix_only=False,
        pool_reads=False,
        pool_gate=False,
    )
    base.update(kw)
    return PrefixSteerConfig(**base)


def build(variant="swa_steer", **steer_kw):
    model = tiny_model()
    attach_prefix_steer(model, steer_config(**steer_kw))
    cfg = DexConfig(variant=variant, allow_no_anneal=True).resolve()
    attach_dex(model, cfg)
    names = set_trainable(model, cfg)
    return model, cfg, names


def feed(model, ids):
    seg = torch.zeros_like(ids)
    set_steer_segments(model, seg, torch.ones_like(ids, dtype=torch.bool))


# --------------------------------------------------------------------------
# config resolution
# --------------------------------------------------------------------------


def test_variant_resolves_to_frozen_backbone_and_trained_sidecar():
    cfg = DexConfig(variant="swa_steer").resolve()
    assert cfg.train_steer is True
    assert cfg.train_attn is False       # the whole point: no 566M finetune
    assert cfg.adapter_enabled is False
    assert cfg.sign == 0.0


def test_steer_and_attn_training_are_mutually_exclusive():
    with pytest.raises(ValueError, match="instead of"):
        DexConfig(variant="attn_only", train_steer=True).resolve()


def test_set_trainable_refuses_when_no_sidecar_attached():
    model = tiny_model()
    cfg = DexConfig(variant="swa_steer").resolve()
    attach_dex(model, cfg)
    with pytest.raises(RuntimeError, match="attach_prefix_steer must run"):
        set_trainable(model, cfg)


# --------------------------------------------------------------------------
# trainable set
# --------------------------------------------------------------------------


def test_every_trainable_parameter_is_a_steer_parameter():
    model, _, names = build()
    assert names, "nothing was unfrozen"
    assert all(is_steer_param_name(n) for n in names), [
        n for n in names if not is_steer_param_name(n)
    ]


def test_no_backbone_tensor_requires_grad():
    model, _, _ = build()
    for n, p in model.named_parameters():
        if not is_steer_param_name(n):
            assert not p.requires_grad, n


def test_report_counts_steer_separately_from_attn():
    model, _, _ = build()
    tr = trainable_report(model)
    assert tr["steer_param_count"] > 0
    # the headline contrast against attn_only: zero backbone attention params
    assert tr["attn_param_count"] == 0
    assert tr["adapter_param_count"] == 0
    assert tr["trainable_param_count"] == tr["steer_param_count"]


def test_attn_only_report_is_unchanged_by_the_new_field():
    model = tiny_model()
    cfg = DexConfig(variant="attn_only", allow_no_anneal=True).resolve()
    attach_dex(model, cfg)
    set_trainable(model, cfg)
    tr = trainable_report(model)
    assert tr["steer_param_count"] == 0
    assert tr["attn_param_count"] == tr["trainable_param_count"]


# --------------------------------------------------------------------------
# forward / backward through both wrappers
# --------------------------------------------------------------------------


def test_forward_runs_through_both_wrappers():
    model, _, _ = build()
    ids = torch.randint(0, 64, (1, 12))
    feed(model, ids)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    assert torch.isfinite(out.logits).all()


def test_gradients_reach_the_sidecar_and_nothing_else():
    model, _, _ = build()
    ids = torch.randint(0, 64, (1, 12))
    feed(model, ids)
    model(input_ids=ids, labels=ids, use_cache=False).loss.backward()
    got = [n for n, p in model.named_parameters()
           if p.grad is not None and p.grad.abs().sum() > 0]
    assert got, "no parameter received a gradient"
    assert all(is_steer_param_name(n) for n in got), [
        n for n in got if not is_steer_param_name(n)
    ]


def test_dex_wrapper_is_reachable_through_the_sidecar():
    """The sidecar calls ``base.o_proj``; with swa_steer that is a DEX wrapper."""
    from deltamem.core.dex import DexOutputProjection
    from deltamem.core.prefix_steer import PrefixMemSteerAttention

    model, _, _ = build()
    seen = 0
    for m in model.modules():
        if isinstance(m, PrefixMemSteerAttention):
            assert isinstance(m.base.o_proj, DexOutputProjection)
            assert m.base.o_proj.adapter is None      # swa_steer carries no adapter
            seen += 1
    assert seen == 2


def test_zero_gain_reproduces_the_frozen_backbone():
    """steer_gain=0 must give back the untouched model, bit for bit.

    This is the check that the wrappers are additive rather than perturbing:
    any difference here would mean the composition itself changes the forward.
    """
    ids = torch.randint(0, 64, (1, 12))
    ref = tiny_model()
    with torch.no_grad():
        expect = ref(input_ids=ids, use_cache=False).logits

    model, _, _ = build(steer_gain=0.0)
    feed(model, ids)
    with torch.no_grad():
        got = model(input_ids=ids, use_cache=False).logits
    assert torch.equal(got, expect)
