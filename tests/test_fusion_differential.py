"""Tests for the differential output-fusion family.

These check the algebra of the fusion itself, independently of whether the memory
path produces a useful control.  The load-bearing one is
``test_variance_diff_cancels_a_covarying_control``: on a construction where the
control genuinely carries a component of the signal, the closed-form coefficient
must remove it -- that is the entire premise the DEX study could not test because
its control was a free linear map of the signal.
"""
from __future__ import annotations

import pytest
import torch

from deltamem.core.prefix_steer import (
    PrefixSteerConfig,
    attach_prefix_steer,
    collect_fusion_stats,
    freeze_steer_keep_fusion,
    is_fusion_param_name,
    is_steer_param_name,
    iter_steer_modules,
    set_fusion_calibrating,
    set_steer_segments,
)
from tests.test_dex import HEAD_DIM, tiny_model


def cfg(fusion, **kw):
    base = dict(
        num_prefix_tokens=0, sliding_window_size=8, mem_num_heads=1,
        mem_head_dim=HEAD_DIM, steer_mode="deltamem", memory_mode="dynamic",
        memory_value_source="main_v", delta_heads="o", steer_gain=0.1,
        output_fusion=fusion, steer_layers=(0, 1), prefix_write=False,
        write_ctx_only=False, read_prefix_only=False, pool_reads=False,
        pool_gate=False,
    )
    base.update(kw)
    return PrefixSteerConfig(**base)


def build(fusion, **kw):
    model = tiny_model()
    attach_prefix_steer(model, cfg(fusion, **kw))
    return model


def run(model, ids=None):
    ids = torch.randint(0, 64, (1, 12)) if ids is None else ids
    set_steer_segments(model, torch.zeros_like(ids),
                       torch.ones_like(ids, dtype=torch.bool))
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits


def a_module(model):
    return next(iter(iter_steer_modules(model)))


# --------------------------------------------------------------------------
# fusion algebra, exercised directly on _fuse_delta_o
# --------------------------------------------------------------------------


def test_fixed_add_is_unchanged_and_fixed_sub_is_its_mirror():
    y = torch.randn(2, 5, 32)
    c = torch.randn(2, 5, 32)
    add = a_module(build("fixed_add"))._fuse_delta_o(y, c)
    sub = a_module(build("fixed_sub"))._fuse_delta_o(y, c)
    torch.testing.assert_close(add, y + 0.1 * c)
    torch.testing.assert_close(sub, y - 0.1 * c)
    # legacy name must stay bit-identical to the new explicit alias
    legacy = a_module(build("fixed"))._fuse_delta_o(y, c)
    assert torch.equal(legacy, add)


def test_learned_diff_uses_its_parameter_and_is_trainable():
    model = build("learned_diff", fusion_lambda_init=0.25)
    mod = a_module(model)
    y, c = torch.randn(2, 5, 32), torch.randn(2, 5, 32)
    torch.testing.assert_close(mod._fuse_delta_o(y, c), y - 0.25 * c)
    names = [n for n, _ in model.named_parameters() if is_fusion_param_name(n)]
    assert len(names) == 2, names            # one scalar per steer layer
    assert all(is_steer_param_name(n) for n in names), "fusion must count as steer"


def test_variance_diff_cancels_a_covarying_control():
    """Y = S + a*C_true; the control sees C_true, so lambda* must recover `a`."""
    torch.manual_seed(0)
    mod = a_module(build("variance_diff", fusion_lambda_max=4.0))
    a = 0.8
    signal = torch.randn(64, 32)
    control = torch.randn(64, 32) + 3.0            # non-zero mean, so centring matters
    y = signal + a * (control - control.mean(0))

    mod.train()
    for _ in range(50):                            # calibrate the running means
        mod._fuse_delta_o(y, control)
    mod.eval()
    out = mod._fuse_delta_o(y, control)

    lam = mod.last_fusion_stats["lambda"]
    assert lam == pytest.approx(a, abs=0.15), mod.last_fusion_stats
    # the co-varying part is gone: residual correlation with the control collapses
    before = torch.corrcoef(torch.stack([
        (y - y.mean(0)).flatten(), (control - control.mean(0)).flatten()]))[0, 1]
    after = torch.corrcoef(torch.stack([
        (out - out.mean(0)).flatten(), (control - control.mean(0)).flatten()]))[0, 1]
    assert abs(after) < abs(before) / 3, (float(before), float(after))


def test_variance_diff_leaves_an_uncorrelated_control_alone():
    """No shared variance => lambda* ~ 0 => the output is essentially untouched."""
    torch.manual_seed(0)
    mod = a_module(build("variance_diff"))
    y = torch.randn(256, 32)
    c = torch.randn(256, 32)
    mod.train()
    for _ in range(50):
        mod._fuse_delta_o(y, c)
    mod.eval()
    out = mod._fuse_delta_o(y, c)
    assert abs(mod.last_fusion_stats["lambda"]) < 0.1, mod.last_fusion_stats
    assert (out - y).abs().max() < 0.1 * y.abs().max()


def test_variance_diff_clamps_lambda():
    torch.manual_seed(0)
    mod = a_module(build("variance_diff", fusion_lambda_max=0.5))
    c = torch.randn(64, 32)
    y = 5.0 * (c - c.mean(0))                      # raw lambda* would be ~5
    mod.train()
    for _ in range(20):
        mod._fuse_delta_o(y, c)
    assert mod.last_fusion_stats["lambda"] == pytest.approx(0.5)
    assert mod.last_fusion_stats["raw_lambda"] > 1.0


def test_variance_diff_falls_back_to_additive_before_calibration():
    mod = a_module(build("variance_diff"))
    mod.eval()                                     # never trained, never calibrated
    y, c = torch.randn(2, 5, 32), torch.randn(2, 5, 32)
    torch.testing.assert_close(mod._fuse_delta_o(y, c), y + 0.1 * c)


def test_running_means_freeze_at_eval():
    mod = a_module(build("variance_diff"))
    y, c = torch.randn(8, 32), torch.randn(8, 32)
    mod.train()
    mod._fuse_delta_o(y, c)
    seen = int(mod.fusion_ema_seen)
    mu = mod.fusion_mu_c.clone()
    mod.eval()
    mod._fuse_delta_o(torch.randn(8, 32), torch.randn(8, 32) + 50.0)
    assert int(mod.fusion_ema_seen) == seen
    assert torch.equal(mod.fusion_mu_c, mu)


def test_calibration_flag_updates_means_outside_training():
    model = build("variance_diff")
    mod = a_module(model)
    mod.eval()
    assert set_fusion_calibrating(model, True) == 2
    mod._fuse_delta_o(torch.randn(8, 32), torch.randn(8, 32))
    assert int(mod.fusion_ema_seen) == 1
    set_fusion_calibrating(model, False)
    mod._fuse_delta_o(torch.randn(8, 32), torch.randn(8, 32))
    assert int(mod.fusion_ema_seen) == 1


# --------------------------------------------------------------------------
# end-to-end through the model, and the stage-1 freezing contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fusion",
                         ["fixed_add", "fixed_sub", "learned_diff", "variance_diff"])
def test_every_fusion_runs_end_to_end(fusion):
    model = build(fusion)
    logits = run(model)
    assert torch.isfinite(logits).all()


def wake_delta_o(model, seed=0):
    """Give the zero-initialised delta_o a non-trivial weight.

    Stands in for a trained sidecar.  Without this every fusion is a no-op, which
    is itself a contract worth pinning down -- see the test below.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in iter_steer_modules(model):
            w = m.delta_o.weight
            w.copy_(torch.randn(w.shape, generator=g) * 0.05)


def test_untrained_sidecar_makes_every_fusion_a_no_op():
    """delta_o is zero-init, so an UNTRAINED sidecar collapses all four arms to base.

    This is why a stage-1 comparison must load a trained sidecar: otherwise
    add/sub/learned/variance are not four conditions, they are four copies of the
    frozen backbone.
    """
    ids = torch.randint(0, 64, (1, 12))
    ref = run(build("fixed_add"), ids)
    for fusion in ("fixed_sub", "learned_diff", "variance_diff"):
        assert torch.equal(run(build(fusion), ids), ref), fusion


def test_add_and_sub_differ_once_the_control_is_non_zero():
    """With a trained-like delta_o the sign reaches the output."""
    ids = torch.randint(0, 64, (1, 12))
    m_add, m_sub = build("fixed_add"), build("fixed_sub")
    wake_delta_o(m_add)
    wake_delta_o(m_sub)                      # same seed => same control
    add, sub = run(m_add, ids), run(m_sub, ids)
    assert not torch.allclose(add, sub)


def test_learned_diff_at_zero_lambda_equals_the_frozen_backbone():
    """lambda=0 must give back the untouched model even with a live control."""
    ids = torch.randint(0, 64, (1, 12))
    ref = tiny_model()
    with torch.no_grad():
        expect = ref(input_ids=ids, use_cache=False).logits
    model = build("learned_diff", fusion_lambda_init=0.0)
    wake_delta_o(model)
    assert torch.equal(run(model, ids), expect)


def test_stage1_freezing_leaves_only_the_fusion_trainable():
    model = build("learned_diff")
    names = freeze_steer_keep_fusion(model)
    assert names, "nothing trainable"
    assert all(is_fusion_param_name(n) for n in names), names
    # the memory path must be frozen: that is what makes the sign meaningful
    frozen = [n for n, p in model.named_parameters()
              if is_steer_param_name(n) and not is_fusion_param_name(n)]
    assert frozen and all(
        not dict(model.named_parameters())[n].requires_grad for n in frozen)


def test_stats_are_collected_per_layer():
    model = build("variance_diff")
    set_fusion_calibrating(model, True)
    run(model)
    stats = collect_fusion_stats(model)
    assert stats["n_layers"] == 2
    assert "lambda" in stats and "raw_lambda" in stats
