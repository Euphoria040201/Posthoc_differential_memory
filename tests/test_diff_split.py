"""Hard gates for post-hoc differential head splitting.

Gate A shapes/GQA · B function equivalence · C gradients · D cache · E causality.
Every gate must pass before any training run is allowed to start.
"""
from __future__ import annotations

import pytest
import torch

from deltamem.core.diff_split import (
    DiffSplitAttention, attach_diff_split, collect_diff_stats,
    freeze_backbone_keep_diff, is_diff_param_name, iter_diff_modules,
    set_diff_enabled, set_diff_stats, set_read_control,
)

HID, HEADS, KV, HD = 64, 8, 2, 8          # 4 query heads per kv group, like Qwen3-4B's 32/8


def tiny_model(dtype=torch.float32, attn="eager"):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=97, hidden_size=HID, intermediate_size=128,
                      num_hidden_layers=3, num_attention_heads=HEADS,
                      num_key_value_heads=KV, head_dim=HD,
                      max_position_embeddings=256, attn_implementation=attn)
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).to(dtype).eval()


def ids(b=1, l=16, seed=3):
    torch.manual_seed(seed)
    return torch.randint(0, 97, (b, l))


# ----------------------------------------------------------------- Gate A
def test_A_gqa_pair_shares_kv_group():
    """The interleaved pair (2i, 2i+1) must land in the SAME kv group as head i."""
    H, G = HEADS, KV
    rep_base, rep_split = H // G, (2 * H) // G
    mapping = []
    for i in range(H):
        base_g = i // rep_base
        plus_g = (2 * i) // rep_split
        minus_g = (2 * i + 1) // rep_split
        mapping.append((i, base_g, plus_g, minus_g))
        assert plus_g == base_g == minus_g, (i, base_g, plus_g, minus_g)
    print("\nhead -> kv group (base | + | -):")
    for i, b, p, m in mapping:
        print(f"  head {i:2d}: base kv{b}  +kv{p}  -kv{m}")
    # a front/back split would NOT satisfy this; assert that explicitly
    bad = [(i, i // rep_base, (i + H) // rep_split) for i in range(H)]
    assert any(b != s for _, b, s in bad), "front/back split must break the mapping"


@pytest.mark.parametrize("b,l", [(1, 1), (1, 5), (2, 16), (3, 33)])
def test_A_shapes(b, l):
    m = tiny_model()
    attach_diff_split(m, [1], read_dim=HD, window=4)
    mod = next(iter_diff_modules(m))
    assert mod.n_heads == HEADS and mod.n_kv == KV
    assert mod.delta_q.out_features == HEADS * HD
    assert mod.base.o_proj.in_features == HEADS * HD      # o_proj input NOT doubled
    out = m(input_ids=ids(b, l), use_cache=False).logits
    assert out.shape == (b, l, 97) and torch.isfinite(out).all()


# ----------------------------------------------------------------- Gate B
def test_B_fp32_parity_full_forward():
    x = ids(2, 24)
    m = tiny_model(torch.float32)
    with torch.no_grad():
        ref = m(input_ids=x, use_cache=False).logits.clone()
    attach_diff_split(m, [0, 1, 2], read_dim=HD, window=8, gamma=1.0)
    with torch.no_grad():
        got = m(input_ids=x, use_cache=False).logits
    err = (got - ref).abs().max().item()
    print(f"\nFP32 split_zero max_abs_logit_err = {err:.3e}")
    assert err <= 1e-5, err


def test_B_disabled_takes_base_path():
    x = ids(2, 20)
    m = tiny_model(torch.float32)
    with torch.no_grad():
        ref = m(input_ids=x, use_cache=False).logits.clone()
    attach_diff_split(m, [0, 1, 2], read_dim=HD, window=8)
    set_diff_enabled(m, False)
    with torch.no_grad():
        got = m(input_ids=x, use_cache=False).logits
    assert (got - ref).abs().max().item() <= 1e-5


@pytest.mark.parametrize("gamma", [0.25, 0.5, 1.0, 3.0])
def test_B_parity_holds_for_any_fixed_gamma(gamma):
    """delta_q = 0  =>  O+ == O-  =>  O~ == O+ regardless of gamma."""
    x = ids(1, 18)
    m = tiny_model(torch.float32)
    with torch.no_grad():
        ref = m(input_ids=x, use_cache=False).logits.clone()
    attach_diff_split(m, [0, 1, 2], read_dim=HD, window=8, gamma=gamma)
    with torch.no_grad():
        got = m(input_ids=x, use_cache=False).logits
    assert (got - ref).abs().max().item() <= 1e-5


def test_B_bf16_greedy_agreement():
    x = ids(1, 20)
    m = tiny_model(torch.float32)
    with torch.no_grad():
        ref = m.generate(x, max_new_tokens=12, do_sample=False)
    mb = tiny_model(torch.float32).to(torch.bfloat16)
    attach_diff_split(mb, [0, 1, 2], read_dim=HD, window=8)
    with torch.no_grad():
        got = mb.generate(x, max_new_tokens=12, do_sample=False)
    mref = tiny_model(torch.float32).to(torch.bfloat16)
    with torch.no_grad():
        base_bf16 = mref.generate(x, max_new_tokens=12, do_sample=False)
    assert torch.equal(got, base_bf16), "bf16 greedy tokens must match bf16 base exactly"


def test_B_nonzero_delta_changes_output():
    """Sanity: the split is not a no-op once delta_q is non-zero."""
    x = ids(1, 20)
    m = tiny_model(torch.float32)
    attach_diff_split(m, [1], read_dim=HD, window=8)
    with torch.no_grad():
        a = m(input_ids=x, use_cache=False).logits.clone()
    mod = next(iter_diff_modules(m))
    with torch.no_grad():
        mod.delta_q.weight.normal_(std=0.5)
    with torch.no_grad():
        b = m(input_ids=x, use_cache=False).logits
    assert (a - b).abs().max().item() > 1e-3


# ----------------------------------------------------------------- Gate C
def test_C_gradients_and_frozen_backbone():
    m = tiny_model(torch.float32)
    attach_diff_split(m, [0, 1, 2], read_dim=HD, window=8)
    trainable = freeze_backbone_keep_diff(m)
    assert trainable, "no trainable params selected"
    assert all(is_diff_param_name(n) for n in trainable)
    before = {n: p.detach().clone() for n, p in m.named_parameters()
              if not is_diff_param_name(n)}
    x = ids(2, 16)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    m.train()
    loss = m(input_ids=x, labels=x, use_cache=False).loss
    loss.backward()
    # STEP 0: delta_q must receive a real gradient.  The reader necessarily gets
    # EXACTLY zero, because dLoss/dR = delta_q.weight^T @ dLoss/d(dq) and the
    # weight is zero-initialised.  This is the same structural property as a
    # zero-initialised LoRA B matrix and is asserted, not waived.
    for mod in iter_diff_modules(m):
        assert mod.delta_q.weight.grad is not None
        assert torch.isfinite(mod.delta_q.weight.grad).all()
        assert mod.delta_q.weight.grad.abs().sum() > 0, "delta_q got no gradient at step 0"
        for nm, p in (("read_q", mod.read_q.weight), ("read_k", mod.read_k.weight)):
            assert p.grad.abs().sum() == 0, (
                f"{nm} should be exactly zero at step 0 with zero-init delta_q")
    for n, p in m.named_parameters():
        if not is_diff_param_name(n):
            assert p.grad is None or p.grad.abs().sum() == 0, n
    opt.step()
    for n, p in m.named_parameters():
        if not is_diff_param_name(n):
            assert torch.equal(p.detach(), before[n]), f"backbone changed: {n}"
    assert all(p in set(id(q) for q in m.parameters() if q.requires_grad)
               for p in [id(g) for grp in opt.param_groups for g in grp["params"]])


def test_C_reader_receives_gradient_after_first_step():
    """Once delta_q is non-zero (i.e. from step 1 on) the local reader must learn."""
    m = tiny_model(torch.float32)
    attach_diff_split(m, [0, 1, 2], read_dim=HD, window=8)
    freeze_backbone_keep_diff(m)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    x = ids(2, 16)
    m.train()
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        m(input_ids=x, labels=x, use_cache=False).loss.backward()
        if step == 1:
            for mod in iter_diff_modules(m):
                for nm, p in (("read_q", mod.read_q.weight),
                              ("read_k", mod.read_k.weight)):
                    assert p.grad is not None and torch.isfinite(p.grad).all(), nm
                    assert p.grad.abs().sum() > 0, f"{nm} still zero at step 1"
        opt.step()
    for mod in iter_diff_modules(m):
        assert mod.delta_q.weight.abs().sum() > 0, "delta_q never moved off zero"


def test_C_gamma_is_not_a_parameter():
    m = tiny_model()
    attach_diff_split(m, [0], read_dim=HD, window=8, gamma=1.0)
    mod = next(iter_diff_modules(m))
    assert isinstance(mod.gamma, float)
    assert "gamma" not in dict(mod.named_parameters())


# ----------------------------------------------------------------- Gate D
def test_D_cache_prefill_decode_matches_full_forward():
    x = ids(1, 18)
    m = tiny_model(torch.float32)
    attach_diff_split(m, [0, 1, 2], read_dim=HD, window=8)
    with torch.no_grad():
        mod = next(iter_diff_modules(m))
        mod.delta_q.weight.normal_(std=0.1)          # exercise the real (non-zero) path
        full = m(input_ids=x, use_cache=False).logits[:, -1]
    from transformers import DynamicCache
    with torch.no_grad():
        cache = DynamicCache()
        m(input_ids=x[:, :-1], past_key_values=cache, use_cache=True)
        step = m(input_ids=x[:, -1:], past_key_values=cache, use_cache=True).logits[:, -1]
    err = (full - step).abs().max().item()
    print(f"\ncached-decode vs full-forward max_abs = {err:.3e}")
    assert err <= 2e-4, err


def test_D_kv_cache_shape_matches_base():
    from transformers import DynamicCache
    x = ids(2, 12)
    mb = tiny_model(torch.float32)
    with torch.no_grad():
        cb = DynamicCache(); mb(input_ids=x, past_key_values=cb, use_cache=True)
    ms = tiny_model(torch.float32)
    attach_diff_split(ms, [0, 1, 2], read_dim=HD, window=8)
    with torch.no_grad():
        cs = DynamicCache(); ms(input_ids=x, past_key_values=cs, use_cache=True)
    for li in range(3):
        kb, vb = cb.layers[li].keys, cb.layers[li].values
        ks, vs = cs.layers[li].keys, cs.layers[li].values
        assert ks.shape == kb.shape, (li, ks.shape, kb.shape)
        assert vs.shape == vb.shape
        assert ks.dtype == kb.dtype
        assert ks.shape[1] == KV, "negative-branch queries must not enter the cache"
        assert ks.numel() * ks.element_size() == kb.numel() * kb.element_size()


# ----------------------------------------------------------------- Gate E
def test_E_local_reader_is_causal():
    """Perturbing hidden states at t+1.. must not change R_t nor the output at t."""
    m = tiny_model(torch.float32)
    attach_diff_split(m, [1], read_dim=HD, window=4)
    mod = next(iter_diff_modules(m))
    with torch.no_grad():
        mod.delta_q.weight.normal_(std=0.3)
    B, L = 1, 14
    torch.manual_seed(5)
    h = torch.randn(B, L, HID)
    v = torch.randn(B, KV, L, HD)
    with torch.no_grad():
        r1 = mod._local_read(h, v)
        h2 = h.clone(); h2[:, 9:] += 5.0                 # perturb the future only
        v2 = v.clone(); v2[:, :, 9:] += 5.0
        r2 = mod._local_read(h2, v2)
    err = (r1[:, :9] - r2[:, :9]).abs().max().item()
    print(f"\ncausality: max |R_t(past) - R_t(future-perturbed)| for t<9 = {err:.3e}")
    assert err == 0.0, err
    assert (r1[:, 9:] - r2[:, 9:]).abs().max().item() > 0, "perturbation had no effect at all"


def test_E_window_limits_reach():
    m = tiny_model(torch.float32)
    attach_diff_split(m, [1], read_dim=HD, window=3)
    mod = next(iter_diff_modules(m))
    B, L = 1, 12
    torch.manual_seed(7)
    h = torch.randn(B, L, HID); v = torch.randn(B, KV, L, HD)
    with torch.no_grad():
        r1 = mod._local_read(h, v)
        h2 = h.clone(); h2[:, 0:2] += 9.0                # outside the window of t>=5
        v2 = v.clone(); v2[:, :, 0:2] += 9.0
        r2 = mod._local_read(h2, v2)
    assert (r1[:, 5:] - r2[:, 5:]).abs().max().item() == 0.0


# ----------------------------------------------------------------- extras
def test_param_count_formula_matches_old_sidecar():
    """d_r=128 on Qwen3-4B geometry must reproduce 14,155,776 over 12 layers."""
    hid, heads, hd, dr, layers = 2560, 32, 128, 128, 12
    per = hid * dr + hid * dr + dr * heads * hd
    assert per == 1_179_648, per
    assert per * layers == 14_155_776


def test_read_controls_change_output():
    x = ids(1, 20)
    m = tiny_model(torch.float32)
    attach_diff_split(m, [1], read_dim=HD, window=8)
    mod = next(iter_diff_modules(m))
    with torch.no_grad():
        mod.delta_q.weight.normal_(std=0.3)
        a = m(input_ids=x, use_cache=False).logits.clone()
        set_read_control(m, zero=True)
        b = m(input_ids=x, use_cache=False).logits.clone()
        set_read_control(m, zero=False)
        c = m(input_ids=x, use_cache=False).logits.clone()
    assert (a - b).abs().max() > 1e-4, "zero-read control had no effect"
    assert torch.equal(a, c)


def test_stats_are_zero_at_init():
    x = ids(1, 16)
    m = tiny_model(torch.float32)
    attach_diff_split(m, [0, 1, 2], read_dim=HD, window=8)
    set_diff_stats(m, True)
    with torch.no_grad():
        m(input_ids=x, use_cache=False)
    st = collect_diff_stats(m)
    print("\ninit stats:", st)
    assert st["branch_div_rel"] == 0.0 and st["correction_rel"] == 0.0
