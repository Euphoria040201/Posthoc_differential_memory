"""Hard correctness gates for the token-local low-rank query split.

Run:  pytest -q tests/test_lowrank_split.py

These are the pre-training gates.  Nothing in the CPT study may launch before
every test here passes, because several of the defects found in the previous
session (silently frozen adapters, fail-open loaders, a condition switch that
reached zero modules) would have produced *plausible numbers* rather than a
crash.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deltamem.core.lowrank_split import (  # noqa: E402
    LowRankSplitAttention, attach_lowrank_split, collect_split_stats,
    expected_split_param_count, freeze_backbone_keep_split, is_split_param_name,
    iter_split_modules, load_split_state_dict, set_split_enabled, set_split_stats,
    split_param_names, split_state_dict, unfreeze_split_layer_attention,
)

HIDDEN, HEADS, KV, HEAD_DIM, LAYERS, VOCAB = 64, 4, 2, 16, 3, 128
RANK = 8
LAYER_IDS = (0, 2)


def build_model(attn_impl="eager", dtype=torch.float32, seed=0):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=128,
        num_hidden_layers=LAYERS, num_attention_heads=HEADS,
        num_key_value_heads=KV, head_dim=HEAD_DIM, max_position_embeddings=256,
        rms_norm_eps=1e-6, tie_word_embeddings=True, attn_implementation=attn_impl,
    )
    torch.manual_seed(seed)
    return Qwen3ForCausalLM(cfg).to(dtype).eval()


def randomize_split(model, scale=0.05, seed=1):
    """Take lr_B off zero so the split actually differs from the base."""
    g = torch.Generator().manual_seed(seed)
    for m in iter_split_modules(model):
        with torch.no_grad():
            m.lr_B.weight.copy_(torch.randn(m.lr_B.weight.shape, generator=g) * scale)


def ids(b=2, L=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (b, L), generator=g)


# ---------------------------------------------------------------- construction
def test_param_count_formula():
    m = build_model()
    attach_lowrank_split(m, LAYER_IDS, rank=RANK)
    per_layer = RANK * (HIDDEN + HEADS * HEAD_DIM)
    want = per_layer * len(LAYER_IDS)
    got = sum(p.numel() for n, p in m.named_parameters() if is_split_param_name(n))
    assert got == want == expected_split_param_count(m), (got, want)
    # no biases or gates were silently added
    assert len(split_param_names(m)) == 2 * len(LAYER_IDS)


def test_attach_rejects_unknown_layer():
    m = build_model()
    with pytest.raises(ValueError):
        attach_lowrank_split(m, (0, 99), rank=RANK)


def test_output_shapes_and_batch_gt_1():
    m = build_model()
    attach_lowrank_split(m, LAYER_IDS, rank=RANK)
    randomize_split(m)
    for b, L in ((1, 8), (3, 16), (5, 7)):
        out = m(input_ids=ids(b, L)).logits
        assert out.shape == (b, L, VOCAB)
        assert torch.isfinite(out).all()


# ------------------------------------------------------- function preservation
def test_fp32_logits_parity_at_zero_init():
    """B=0 must reproduce the base model EXACTLY in fp32, not approximately."""
    base = build_model()
    x = ids(2, 24)
    with torch.no_grad():
        ref = base(input_ids=x).logits.clone()
    attach_lowrank_split(base, LAYER_IDS, rank=RANK)
    with torch.no_grad():
        got = base(input_ids=x).logits
    assert torch.equal(ref, got), (ref - got).abs().max().item()


def test_bf16_difference_is_small_but_reported_honestly():
    """bf16 is NOT bit-exact; assert a bound and expose the numbers."""
    base = build_model(dtype=torch.bfloat16)
    x = ids(2, 24)
    with torch.no_grad():
        ref = base(input_ids=x).logits.float().clone()
    attach_lowrank_split(base, LAYER_IDS, rank=RANK)
    with torch.no_grad():
        got = base(input_ids=x).logits.float()
    max_abs = (ref - got).abs().max().item()
    rel = max_abs / ref.abs().max().item()
    print(f"\n[bf16] max_abs={max_abs:.3e} rel={rel:.3e} "
          f"greedy_match={(ref.argmax(-1) == got.argmax(-1)).float().mean().item():.4f}")
    assert max_abs < 5e-2, max_abs          # bounded, explicitly NOT zero


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("impl", ["eager", "sdpa"])
def test_bf16_cuda_difference_reported_per_backend(impl):
    """The honest bf16 gate: the REAL path is sdpa on CUDA, not eager on CPU.

    At zero init dQ is exactly 0.0, so Q- and Q+ are bit-identical and any
    residual difference comes purely from the attention kernel running over 2H
    heads instead of H (different blocking / reduction order).  Eager keeps the
    per-head reduction order and is bit-exact; sdpa need not be.  Report both
    rather than generalising from whichever one happens to be zero.
    """
    base = build_model(attn_impl=impl).to("cuda", torch.bfloat16)
    x = ids(2, 64).to("cuda")
    with torch.no_grad():
        ref = base(input_ids=x).logits.float().clone()
    attach_lowrank_split(base, LAYER_IDS, rank=RANK)
    with torch.no_grad():
        got = base(input_ids=x).logits.float()
    max_abs = (ref - got).abs().max().item()
    denom = ref.abs().max().item()
    match = (ref.argmax(-1) == got.argmax(-1)).float().mean().item()
    print(f"\n[bf16-cuda-{impl}] max_abs={max_abs:.3e} rel={max_abs/denom:.3e} "
          f"greedy_match={match:.6f} bit_exact={max_abs == 0.0}")
    assert max_abs < 5e-2, max_abs


def test_disabled_matches_base_even_with_nonzero_delta():
    """The condition switch must genuinely switch, not silently stay on."""
    base = build_model()
    x = ids(2, 20)
    with torch.no_grad():
        ref = base(input_ids=x).logits.clone()
    attach_lowrank_split(base, LAYER_IDS, rank=RANK)
    randomize_split(base)
    with torch.no_grad():
        on = base(input_ids=x).logits.clone()
    n = set_split_enabled(base, False)
    assert n == len(LAYER_IDS)              # the old bug reached ZERO modules
    with torch.no_grad():
        off = base(input_ids=x).logits
    assert torch.equal(ref, off)            # exactly base again
    assert not torch.allclose(ref, on, atol=1e-6)   # and it really was different


def test_nonzero_dq_changes_output():
    m = build_model()
    x = ids(2, 20)
    attach_lowrank_split(m, LAYER_IDS, rank=RANK)
    with torch.no_grad():
        before = m(input_ids=x).logits.clone()
    randomize_split(m)
    with torch.no_grad():
        after = m(input_ids=x).logits
    assert (before - after).abs().max() > 1e-4


# ------------------------------------------------------------ GQA / reference
@pytest.mark.parametrize("impl", ["eager", "sdpa"])
def test_matches_two_pass_reference_exactly(impl):
    """Strongest gate: an independent reference for the WHOLE construction.

    Computes O+ and O- with two separate H-head attention calls through the
    frozen base module, then combines them.  If the interleaving, the GQA repeat
    factor, the RoPE application or the branch de-interleaving were wrong, this
    could not match.
    """
    import transformers.models.qwen3.modeling_qwen3 as _q

    m = build_model(attn_impl=impl)
    attach_lowrank_split(m, LAYER_IDS, rank=RANK)
    randomize_split(m, scale=0.2)
    mod = next(iter_split_modules(m))
    base = mod.base

    B, L = 2, 16
    torch.manual_seed(3)
    h = torch.randn(B, L, HIDDEN)
    pos = torch.arange(L)[None].expand(B, L)
    rotary = m.model.rotary_emb
    pe = rotary(h, pos)

    with torch.no_grad():
        got, _ = mod(h, pe, None)

        hs = (B, L, -1, HEAD_DIM)
        q_pre = base.q_proj(h)
        dq = mod.lr_B(mod.lr_A(h))
        q_plus = base.q_norm(q_pre.view(hs)).transpose(1, 2)
        q_minus = base.q_norm((q_pre + dq).view(hs)).transpose(1, 2)
        k = base.k_norm(base.k_proj(h).view(hs)).transpose(1, 2)
        v = base.v_proj(h).view(hs).transpose(1, 2)
        cos, sin = pe
        qp, k_r = _q.apply_rotary_pos_emb(q_plus, k, cos, sin)
        qm, _ = _q.apply_rotary_pos_emb(q_minus, k, cos, sin)

        fn = _q.ALL_ATTENTION_FUNCTIONS.get_interface(impl, _q.eager_attention_forward)
        o_p, _ = fn(base, qp, k_r, v, None, dropout=0.0, scaling=base.scaling,
                    sliding_window=base.sliding_window)
        o_m, _ = fn(base, qm, k_r, v, None, dropout=0.0, scaling=base.scaling,
                    sliding_window=base.sliding_window)
        o_t = o_p + mod.gamma * (o_p - o_m)
        want = base.o_proj(o_t.reshape(B, L, -1).contiguous())

    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max().item()


def test_pair_shares_original_kv_group():
    """Head 2i and 2i+1 must both map to the kv group original head i used."""
    m = build_model()
    attach_lowrank_split(m, LAYER_IDS, rank=RANK)
    mod = next(iter_split_modules(m))
    repeat = (2 * mod.n_heads) // mod.n_kv
    base_repeat = mod.n_heads // mod.n_kv
    for i in range(mod.n_heads):
        assert (2 * i) // repeat == i // base_repeat
        assert (2 * i + 1) // repeat == i // base_repeat


# --------------------------------------------------------------- cache / decode
def test_kv_cache_shape_and_size_unchanged():
    from transformers import DynamicCache

    x = ids(2, 24)
    base = build_model()
    c1 = DynamicCache()
    with torch.no_grad():
        base(input_ids=x, past_key_values=c1, use_cache=True)
    split = build_model()
    attach_lowrank_split(split, LAYER_IDS, rank=RANK)
    randomize_split(split)
    c2 = DynamicCache()
    with torch.no_grad():
        split(input_ids=x, past_key_values=c2, use_cache=True)

    def summarize(c):
        ks, vs, n = [], [], 0
        for layer in c.layers:
            ks.append(tuple(layer.keys.shape)); vs.append(tuple(layer.values.shape))
            n += layer.keys.numel() * layer.keys.element_size()
            n += layer.values.numel() * layer.values.element_size()
        return ks, vs, n

    assert summarize(c1) == summarize(c2)


def test_cached_decode_matches_full_forward():
    """Token-local dQ means decode must equal prefill with no reader state."""
    from transformers import DynamicCache

    m = build_model()
    attach_lowrank_split(m, LAYER_IDS, rank=RANK)
    randomize_split(m)
    x = ids(2, 20)
    with torch.no_grad():
        full = m(input_ids=x).logits

        cache = DynamicCache()
        m(input_ids=x[:, :10], past_key_values=cache, use_cache=True)
        step = []
        for t in range(10, x.shape[1]):
            out = m(input_ids=x[:, t:t + 1], past_key_values=cache, use_cache=True)
            step.append(out.logits[:, -1])
        step = torch.stack(step, 1)
    assert torch.allclose(full[:, 10:], step, atol=2e-5), \
        (full[:, 10:] - step).abs().max().item()


def test_no_private_inference_cache():
    """The LocalRead module stores _read_h/_read_v; this one must store nothing."""
    m = build_model()
    attach_lowrank_split(m, LAYER_IDS, rank=RANK)
    randomize_split(m)
    with torch.no_grad():
        m(input_ids=ids(2, 16))
    for mod in iter_split_modules(m):
        stateful = {k: type(v).__name__ for k, v in vars(mod).items()
                    if torch.is_tensor(v)}
        assert not stateful, f"unexpected tensor state on the module: {stateful}"
        assert not list(mod.buffers(recurse=False))


# ------------------------------------------------------------------- padding
@pytest.mark.parametrize("side", ["left", "right"])
def test_padding_does_not_change_real_positions(side):
    """Token-local dQ cannot leak across padding; verify end to end."""
    m = build_model()
    attach_lowrank_split(m, LAYER_IDS, rank=RANK)
    randomize_split(m)
    real = ids(1, 12, seed=5)
    pad_len = 5
    pad = torch.zeros(1, pad_len, dtype=torch.long)
    if side == "left":
        padded = torch.cat([pad, real], 1)
        mask = torch.cat([torch.zeros(1, pad_len), torch.ones(1, real.shape[1])], 1).long()
        sl = slice(pad_len, None)
    else:
        padded = torch.cat([real, pad], 1)
        mask = torch.cat([torch.ones(1, real.shape[1]), torch.zeros(1, pad_len)], 1).long()
        sl = slice(0, real.shape[1])
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        ref = m(input_ids=real).logits
        got = m(input_ids=padded, attention_mask=mask, position_ids=pos).logits[:, sl]
    assert torch.allclose(ref, got, atol=2e-4), (ref - got).abs().max().item()


# ------------------------------------------------------------------ gradients
def test_backbone_gets_no_gradient_in_adapter_only_mode():
    m = build_model()
    attach_lowrank_split(m, LAYER_IDS, rank=RANK)
    names = freeze_backbone_keep_split(m)
    assert set(names) == set(split_param_names(m))
    x = ids(2, 16)
    m(input_ids=x, labels=x).loss.backward()
    for n, p in m.named_parameters():
        if is_split_param_name(n):
            continue
        assert p.grad is None, f"backbone parameter {n} received a gradient"


def test_B_gets_gradient_at_step0_and_A_only_after_B_moves():
    """With B=0, dL/dA = 0 exactly (chain rule through B). Assert both facts."""
    m = build_model()
    attach_lowrank_split(m, LAYER_IDS, rank=RANK)
    freeze_backbone_keep_split(m)
    x = ids(2, 16)
    m(input_ids=x, labels=x).loss.backward()
    mod = next(iter_split_modules(m))
    assert mod.lr_B.weight.grad.abs().max() > 0, "B must receive gradient at step 0"
    assert mod.lr_A.weight.grad.abs().max() == 0, "with B=0 the A gradient must be exactly 0"

    m.zero_grad(set_to_none=True)
    randomize_split(m)                      # B leaves zero
    m(input_ids=x, labels=x).loss.backward()
    assert mod.lr_A.weight.grad.abs().max() > 0, "A must receive gradient once B != 0"


def test_unfreeze_arm_touches_only_split_layers():
    m = build_model()
    attach_lowrank_split(m, LAYER_IDS, rank=RANK)
    freeze_backbone_keep_split(m)
    newly = unfreeze_split_layer_attention(m)
    assert newly, "nothing was unfrozen"
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    for n in trainable:
        if is_split_param_name(n):
            continue
        li = int(n.split("layers.")[1].split(".")[0])
        assert li in LAYER_IDS, f"unfroze a parameter outside the split layers: {n}"
        assert ".self_attn." in n, f"unfroze a non-attention parameter: {n}"
    # mlp / embeddings stay frozen
    assert not any(".mlp." in n for n in trainable)


# ------------------------------------------------------------------ checkpoint
def test_strict_checkpoint_round_trip():
    a = build_model()
    attach_lowrank_split(a, LAYER_IDS, rank=RANK)
    randomize_split(a, seed=7)
    state = split_state_dict(a)
    x = ids(2, 16)
    with torch.no_grad():
        want = a(input_ids=x).logits.clone()

    b = build_model()                       # same seed -> same backbone
    attach_lowrank_split(b, LAYER_IDS, rank=RANK)
    info = load_split_state_dict(b, state)
    assert info["loaded"] == info["expected"] == 2 * len(LAYER_IDS)
    with torch.no_grad():
        assert torch.equal(want, b(input_ids=x).logits)


def test_loader_is_fail_closed():
    m = build_model()
    attach_lowrank_split(m, LAYER_IDS, rank=RANK)
    good = split_state_dict(m)

    with pytest.raises(KeyError):           # missing key
        load_split_state_dict(m, {k: v for k, v in list(good.items())[:-1]})
    with pytest.raises(KeyError):           # unexpected key
        load_split_state_dict(m, {**good, "model.layers.0.self_attn.lr_C.weight": torch.zeros(1)})
    with pytest.raises(KeyError):           # backbone key inside a split-only ckpt
        load_split_state_dict(m, {**good, "model.embed_tokens.weight": torch.zeros(1)})
    with pytest.raises(ValueError):         # shape mismatch
        k0 = sorted(good)[0]
        load_split_state_dict(m, {**good, k0: torch.zeros(3, 3)})


# ------------------------------------------------------------ output_attentions
def test_output_attentions_rejects_incompatible_tensor():
    m = build_model(attn_impl="eager")
    attach_lowrank_split(m, LAYER_IDS, rank=RANK)
    randomize_split(m)
    with pytest.raises(NotImplementedError):
        m(input_ids=ids(1, 8), output_attentions=True)


def test_effective_attention_reconstructs_output():
    """A_eff = (1+g)A+ - g A- must satisfy O~ = A_eff @ V and sum to 1."""
    import transformers.models.qwen3.modeling_qwen3 as _q

    m = build_model(attn_impl="eager")
    attach_lowrank_split(m, LAYER_IDS, rank=RANK)
    randomize_split(m, scale=0.2)
    mod = next(iter_split_modules(m))
    base = mod.base
    B, L = 1, 12
    torch.manual_seed(11)
    h = torch.randn(B, L, HIDDEN)
    pe = m.model.rotary_emb(h, torch.arange(L)[None])
    causal = torch.full((L, L), float("-inf")).triu(1)[None, None]

    with torch.no_grad():
        y, aw = mod(h, pe, causal, output_attentions=True, effective_attention=True)
        assert aw.shape == (B, HEADS, L, L), aw.shape
        assert torch.allclose(aw.sum(-1), torch.ones(B, HEADS, L), atol=1e-5)

        v = base.v_proj(h).view(B, L, -1, HEAD_DIM).transpose(1, 2)
        v = _q.repeat_kv(v, mod.n_heads // mod.n_kv)
        o = torch.matmul(aw, v)             # [B, H, L, D]
        want = base.o_proj(o.transpose(1, 2).reshape(B, L, -1).contiguous())
    assert torch.allclose(y, want, atol=1e-5), (y - want).abs().max().item()


def test_stats_collection():
    m = build_model()
    attach_lowrank_split(m, LAYER_IDS, rank=RANK)
    randomize_split(m, scale=0.2)
    set_split_stats(m, True)
    with torch.no_grad():
        m(input_ids=ids(2, 16))
    st = collect_split_stats(m)
    assert st["n_layers"] == len(LAYER_IDS)
    assert st["branch_div_rel"] > 0 and st["dq_rel"] > 0
