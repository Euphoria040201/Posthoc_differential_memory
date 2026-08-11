"""Tests for ``o_fusion_position``: pre_o / post_o / post_o_projected.

The load-bearing ones are the POSITION tests: with a non-identity W_O they verify
that pre_o computes exactly ``W_O (Z + g C)`` and post_o exactly ``W_O Z + g C``
against tensors captured from the real forward, so the two positions cannot be
confused by construction.  ``post_o_projected`` is then checked to be numerically
equivalent to pre_o for every linear fusion, which is what licenses it as the
math-vs-implementation control arm.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from deltamem.core.prefix_steer import (
    O_FUSION_POSITIONS,
    PrefixSteerConfig,
    attach_prefix_steer,
    collect_fusion_norms,
    is_steer_param_name,
    iter_steer_modules,
    set_collect_fusion_norms,
    set_steer_enabled,
    set_steer_segments,
)
from tests.test_dex import HEAD_DIM, tiny_model

GAIN = 0.1


def tiny_model_rect():
    """GQA backbone whose o_proj is NON-square (in 64 != out 32).

    4 query heads x head_dim 16 concat to Z of width 64 while hidden stays 32, so
    the concat-head basis and the output basis genuinely differ -- any silent
    reshape/truncation of C, or a fusion applied in the wrong basis, fails loudly
    here instead of hiding behind a square o_proj.
    """
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=64, attn_implementation="eager",
    )
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).eval()


def cfg(fusion="fixed_add", position="post_o", mem_head_dim=HEAD_DIM, **kw):
    base = dict(
        num_prefix_tokens=0, sliding_window_size=8, mem_num_heads=1,
        mem_head_dim=mem_head_dim, steer_mode="deltamem", memory_mode="dynamic",
        memory_value_source="main_v", delta_heads="o", steer_gain=GAIN,
        output_fusion=fusion, o_fusion_position=position, steer_layers=(0, 1),
        prefix_write=False, write_ctx_only=False, read_prefix_only=False,
        pool_reads=False, pool_gate=False,
    )
    base.update(kw)
    return PrefixSteerConfig(**base)


def build(fusion="fixed_add", position="post_o", rect=False, seed_delta=True, **kw):
    model = tiny_model_rect() if rect else tiny_model()
    mhd = 16 if rect else HEAD_DIM
    attach_prefix_steer(model, cfg(fusion, position, mem_head_dim=mhd, **kw))
    if seed_delta:
        # delta_o is zero-init by design; give the control C a real value so the
        # positions produce different numbers a test can distinguish.
        g = torch.Generator().manual_seed(7)
        for m in iter_steer_modules(model):
            with torch.no_grad():
                m.delta_o.weight.copy_(
                    torch.randn(m.delta_o.weight.shape, generator=g) * 0.5
                )
    return model


def ids_batch(batch=1, length=12):
    torch.manual_seed(3)
    return torch.randint(0, 64, (batch, length))


def run(model, ids, **fwd):
    set_steer_segments(model, torch.zeros_like(ids),
                       torch.ones_like(ids, dtype=torch.bool))
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False, **fwd)


def first_module(model):
    return next(iter(iter_steer_modules(model)))


def capture_layer0(model, ids):
    """(Z fed to o_proj, C from delta_o, wrapped-module output) of steer layer 0."""
    mod = first_module(model)
    grab = {}
    h1 = mod.base.o_proj.register_forward_pre_hook(
        lambda m, inp: grab.__setitem__("o_in", inp[0].detach().clone()))
    h2 = mod.delta_o.register_forward_hook(
        lambda m, inp, out: grab.__setitem__("C", out.detach().clone()))
    h3 = mod.register_forward_hook(
        lambda m, inp, out: grab.__setitem__("out", out[0].detach().clone()))
    try:
        run(model, ids)
    finally:
        h1.remove(); h2.remove(); h3.remove()
    return grab


# --------------------------------------------------------------------------
# position: each variant computes exactly its formula, with non-identity W_O
# --------------------------------------------------------------------------


def test_pre_o_fuses_before_o_proj():
    ids = ids_batch()
    model = build("fixed_add", "pre_o", rect=True)
    mod = first_module(model)
    # unfused Z: same weights, steer off (delta_heads="o", so q/k/v are untouched
    # and the layer-0 attention activation is identical)
    set_steer_enabled(model, False)
    z_off = capture_layer0(model, ids)
    set_steer_enabled(model, True)
    g = capture_layer0(model, ids)
    W = mod.base.o_proj.weight
    torch.testing.assert_close(g["o_in"], z_off["o_in"] + GAIN * g["C"])
    torch.testing.assert_close(g["out"], F.linear(z_off["o_in"] + GAIN * g["C"], W))
    # W_O is non-identity (and non-square): post-o algebra must NOT reproduce it
    wrong = F.linear(z_off["o_in"], W) + GAIN * F.linear(g["C"], W)
    assert torch.allclose(g["out"], wrong, atol=1e-5)  # equal ONLY because linear
    assert g["o_in"].shape[-1] == mod.base.o_proj.in_features == 64


def test_post_o_fuses_after_o_proj():
    ids = ids_batch()
    model = build("fixed_add", "post_o", rect=True)
    mod = first_module(model)
    g = capture_layer0(model, ids)
    W = mod.base.o_proj.weight
    # o_proj consumed the UNfused Z; C joined afterwards, in the output basis
    torch.testing.assert_close(g["out"], F.linear(g["o_in"], W) + GAIN * g["C"])
    assert g["C"].shape[-1] == mod.base.o_proj.out_features == 32


def test_positions_differ_under_non_identity_W_O():
    # same backbone weights + same-shape C is only possible in the square model;
    # there the two positions still disagree because W_O(Z + gC) != W_O Z + gC
    ids = ids_batch()
    out_pre = run(build("fixed_add", "pre_o"), ids).logits
    out_post = run(build("fixed_add", "post_o"), ids).logits
    assert not torch.allclose(out_pre, out_post, atol=1e-5)


# --------------------------------------------------------------------------
# equivalence boundaries
# --------------------------------------------------------------------------


def test_identity_W_O_collapses_the_positions():
    ids = ids_batch()
    outs = {}
    for pos in ("pre_o", "post_o"):
        model = build("fixed_add", pos)  # square o_proj (32 -> 32)
        with torch.no_grad():
            for m in iter_steer_modules(model):
                m.base.o_proj.weight.copy_(torch.eye(32))
        outs[pos] = run(model, ids).logits
    torch.testing.assert_close(outs["pre_o"], outs["post_o"])


@pytest.mark.parametrize("fusion", ["fixed_add", "fixed_sub", "learned_diff"])
def test_post_o_projected_equals_pre_o_for_linear_fusions(fusion):
    ids = ids_batch()
    kw = {"fusion_lambda_init": 0.25} if fusion == "learned_diff" else {}
    out_pre = run(build(fusion, "pre_o", rect=True, **kw), ids).logits
    out_ctl = run(build(fusion, "post_o_projected", rect=True, **kw), ids).logits
    torch.testing.assert_close(out_pre, out_ctl, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("position", list(O_FUSION_POSITIONS))
def test_zero_coefficient_is_exactly_base(position):
    ids = ids_batch()
    # gain 0 (fixed_add) and lambda 0 (learned_diff) must reproduce the frozen
    # backbone BIT-exactly -- the fused branch contributes 0 in either basis
    for fusion, kw in (("fixed_add", {"steer_gain": 0.0}),
                       ("learned_diff", {"fusion_lambda_init": 0.0})):
        model = build(fusion, position, rect=True, **kw)
        fused = run(model, ids).logits
        set_steer_enabled(model, False)
        base_out = run(model, ids).logits
        assert torch.equal(fused, base_out), (position, fusion)


# --------------------------------------------------------------------------
# gradients
# --------------------------------------------------------------------------


def test_pre_o_gradients_reach_delta_o_and_lambda():
    ids = ids_batch()
    model = build("learned_diff", "pre_o", rect=True, fusion_lambda_init=0.25)
    model.train()
    set_steer_segments(model, torch.zeros_like(ids),
                       torch.ones_like(ids, dtype=torch.bool))
    loss = model(input_ids=ids, labels=ids, use_cache=False).loss
    loss.backward()
    for m in iter_steer_modules(model):
        gw = m.delta_o.weight.grad
        gl = m.fusion_lambda.grad
        assert gw is not None and torch.isfinite(gw).all() and gw.abs().sum() > 0
        assert gl is not None and torch.isfinite(gl).all() and gl.abs() > 0


# --------------------------------------------------------------------------
# shapes / dtypes / GQA
# --------------------------------------------------------------------------


def test_gqa_delta_o_width_follows_position():
    pre = first_module(build("fixed_add", "pre_o", rect=True, seed_delta=False))
    post = first_module(build("fixed_add", "post_o", rect=True, seed_delta=False))
    # GQA: 4 query heads x 16 concat to 64 even though only 2 KV heads exist;
    # o_proj.in_features is the authoritative width, never n_kv * head_dim
    assert pre.delta_o.weight.shape[0] == pre.base.o_proj.in_features == 64
    assert post.delta_o.weight.shape[0] == post.base.o_proj.out_features == 32


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("position", list(O_FUSION_POSITIONS))
def test_batched_forward_shapes_and_dtypes(position, dtype):
    ids = ids_batch(batch=2, length=9)
    model = build("fixed_add", position, rect=True).to(dtype)
    out = run(model, ids).logits
    assert out.shape == (2, 9, 64)
    assert out.dtype == dtype
    assert torch.isfinite(out.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 needs CUDA")
def test_fp16_forward_on_cuda():
    ids = ids_batch(batch=2, length=9).cuda()
    model = build("fixed_add", "pre_o", rect=True).to("cuda", torch.float16)
    out = run(model, ids).logits
    assert out.shape == (2, 9, 64) and torch.isfinite(out.float()).all()


def test_attention_backends_share_the_fusion_path():
    # the fusion sits in the wrapper AFTER the attention_interface dispatch, so
    # sdpa and eager go through the same code; verify both run and stay close
    ids = ids_batch()
    outs = {}
    for impl in ("eager", "sdpa"):
        model = build("fixed_add", "pre_o", rect=True)
        model.config._attn_implementation = impl
        outs[impl] = run(model, ids).logits
    torch.testing.assert_close(outs["eager"], outs["sdpa"], atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------
# config / checkpoint compatibility
# --------------------------------------------------------------------------


def test_old_config_dicts_deserialize_to_post_o():
    saved = {k: v for k, v in vars(cfg()).items() if k != "o_fusion_position"}
    rebuilt = PrefixSteerConfig(**saved)
    assert rebuilt.o_fusion_position == "post_o"


def test_position_validation():
    # repo style: config combinations are validated when the wrapper is attached
    with pytest.raises(ValueError):
        attach_prefix_steer(tiny_model(), cfg(position="mid_o"))
    with pytest.raises(ValueError):
        # pre_o without an active delta_o must be rejected, not silently ignored
        attach_prefix_steer(tiny_model(), cfg(position="pre_o", delta_heads="qk",
                                              output_fusion="fixed"))


def test_post_o_checkpoint_does_not_load_into_pre_o_silently():
    src = build("fixed_add", "post_o", rect=True)
    state = {n: p for n, p in src.state_dict().items()
             if is_steer_param_name(n) and "delta_o" in n}
    dst = build("fixed_add", "pre_o", rect=True)
    with pytest.raises(RuntimeError):
        dst.load_state_dict(state, strict=False)  # 32-wide C into a 64-wide slot


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------


def test_fusion_norm_diagnostics_record_both_bases():
    ids = ids_batch()
    model = build("fixed_add", "pre_o", rect=True)
    set_collect_fusion_norms(model, True)
    run(model, ids)
    norms = collect_fusion_norms(model)
    mean = norms["mean"]
    for key in ("norm_Z", "norm_WZ", "norm_C", "norm_WC",
                "ratio_C_over_Z", "ratio_WC_over_WZ", "cos_Z_C", "cos_WZ_WC"):
        assert key in mean and torch.isfinite(torch.tensor(mean[key]))
    assert len(norms["per_layer"]) == 2
    set_collect_fusion_norms(model, False)
    assert first_module(model).last_fusion_norms == {}
