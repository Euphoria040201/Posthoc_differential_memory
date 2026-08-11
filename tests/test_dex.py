"""Unit tests for the DEX adapter and its control variants (arXiv:2505.16333)."""

from __future__ import annotations

import math

import pytest
import torch

from deltamem.core.dex import (
    AttentionOutputAdapter,
    DexConfig,
    DexOutputProjection,
    attach_dex,
    collect_dex_stats,
    diff_lambda_init,
    is_attn_trainable_param_name,
    is_dex_param_name,
    select_heads_for_layer,
    set_dex_stats,
    set_dex_step,
    set_trainable,
    trainable_report,
)


NUM_HEADS = 4
HEAD_DIM = 8


def make_adapter(sign: float, *, seed: int = 0, **kw) -> AttentionOutputAdapter:
    torch.manual_seed(seed)
    params = dict(
        head_dim=HEAD_DIM,
        num_heads=NUM_HEADS,
        selected_heads=(0, 2),
        sign=sign,
        lambda_init=0.8,
        lambda_learn_init=0.0,
        lambda_learnable=True,
        lambda_anneal_steps=0,
    )
    params.update(kw)
    return AttentionOutputAdapter(**params)


def tiny_model():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).eval()


# --------------------------------------------------------------------------
# lambda schedule (Eq. 4)
# --------------------------------------------------------------------------


def test_lambda_schedule_matches_equation_4():
    a = make_adapter(-1.0, lambda_anneal_steps=100, lambda_learn_init=0.25)
    for t in (0, 25, 50, 99, 100, 250):
        a.set_step(t)
        alpha = min(1.0, t / 100)
        want = (1.0 - alpha) * (t / 100) * 0.8 + alpha * 0.25
        assert a.current_lambda().item() == pytest.approx(want, abs=1e-6)


def test_lambda_schedule_starts_at_zero_and_ends_at_learnable():
    a = make_adapter(-1.0, lambda_anneal_steps=10, lambda_learn_init=0.3)
    a.set_step(0)
    assert a.current_lambda().item() == pytest.approx(0.0)
    a.set_step(10)
    assert a.current_lambda().item() == pytest.approx(0.3)


def test_no_anneal_uses_lambda_learn_directly():
    a = make_adapter(-1.0, lambda_anneal_steps=0, lambda_learn_init=0.7)
    a.set_step(5)
    assert a.current_lambda().item() == pytest.approx(0.7)


def test_diff_depth_lambda_init_is_zero_based_like_the_official_impl():
    # microsoft/unilm Diff-Transformer: lambda_init_fn(depth), depth = layer index
    assert diff_lambda_init(0) == pytest.approx(0.2)
    for layer in (0, 5, 16):
        assert diff_lambda_init(layer) == pytest.approx(0.8 - 0.6 * math.exp(-0.3 * layer))


def test_attach_uses_zero_based_lambda_init_per_layer():
    model = tiny_model()
    rep = attach_dex(model, DexConfig(variant="dex_minus", allow_no_anneal=True),
                     plan=_plan_for(model))
    assert rep["lambda_init"]["0"] == pytest.approx(0.2)
    assert rep["lambda_init"]["1"] == pytest.approx(0.8 - 0.6 * math.exp(-0.3))


# --------------------------------------------------------------------------
# the central fairness check: minus(W_D) == plus(-W_D)
# --------------------------------------------------------------------------


def test_minus_and_plus_are_equivalent_under_weight_negation():
    torch.manual_seed(7)
    x = torch.randn(2, 5, NUM_HEADS, HEAD_DIM, dtype=torch.float64)

    minus = make_adapter(-1.0, seed=3).double()
    plus = make_adapter(+1.0, seed=3).double()
    with torch.no_grad():
        plus.proj.weight.copy_(-minus.proj.weight)
        plus.lambda_learn.copy_(minus.lambda_learn)
    for a in (minus, plus):
        with torch.no_grad():
            a.lambda_learn.fill_(0.42)

    om, op = minus(x), plus(x)
    max_abs = (om - op).abs().max().item()
    mean_abs = (om - op).abs().mean().item()
    print(f"[sign-flip diagnostic] max_abs_err={max_abs:.3e} mean_abs_err={mean_abs:.3e}")
    assert max_abs == pytest.approx(0.0, abs=1e-12)
    assert mean_abs == pytest.approx(0.0, abs=1e-12)


def test_minus_and_plus_have_identical_parameter_shapes_and_counts():
    minus, plus = make_adapter(-1.0, seed=1), make_adapter(+1.0, seed=1)
    pm = {n: tuple(p.shape) for n, p in minus.named_parameters()}
    pp = {n: tuple(p.shape) for n, p in plus.named_parameters()}
    assert pm == pp
    assert sum(p.numel() for p in minus.parameters()) == sum(p.numel() for p in plus.parameters())


def test_same_seed_gives_identical_initialisation_for_both_signs():
    minus, plus = make_adapter(-1.0, seed=11), make_adapter(+1.0, seed=11)
    assert torch.equal(minus.proj.weight, plus.proj.weight)


# --------------------------------------------------------------------------
# head selection isolation
# --------------------------------------------------------------------------


def test_non_selected_heads_are_bit_identical():
    torch.manual_seed(5)
    x = torch.randn(3, 4, NUM_HEADS, HEAD_DIM)
    a = make_adapter(-1.0, selected_heads=(1,))
    with torch.no_grad():
        a.lambda_learn.fill_(0.9)
    y = a(x)
    for h in range(NUM_HEADS):
        if h == 1:
            assert not torch.equal(y[..., h, :], x[..., h, :])
        else:
            assert torch.equal(y[..., h, :], x[..., h, :])


def test_head_plan_selection_high_and_low():
    plan = {"criterion": "entropy", "scores": {"0": [0.1, 0.9, 0.5, 0.7]}}
    cfg = DexConfig(head_selection="entropy_high", heads_per_layer=2,
                    allow_no_anneal=True).resolve()
    assert select_heads_for_layer(0, 4, cfg, plan) == (1, 3)
    cfg_low = DexConfig(head_selection="entropy_low", heads_per_layer=2,
                        allow_no_anneal=True).resolve()
    assert select_heads_for_layer(0, 4, cfg_low, plan) == (0, 2)
    cfg_all = DexConfig(head_selection="all", allow_no_anneal=True).resolve()
    assert select_heads_for_layer(0, 4, cfg_all, None) == (0, 1, 2, 3)


def test_default_k_is_half_the_heads():
    plan = {"criterion": "entropy", "scores": {"0": [0.1, 0.9, 0.5, 0.7]}}
    cfg = DexConfig(head_selection="entropy_high", heads_per_layer=-1,
                    allow_no_anneal=True).resolve()
    assert len(select_heads_for_layer(0, 4, cfg, plan)) == 2


# --------------------------------------------------------------------------
# variant wiring on a real (tiny) Qwen3
# --------------------------------------------------------------------------


def _plan_for(model):
    text_cfg = model.config
    n_layers = text_cfg.num_hidden_layers
    return {
        "criterion": "entropy",
        "scores": {str(i): [float(h) for h in range(NUM_HEADS)] for i in range(n_layers)},
    }


@pytest.mark.parametrize(
    "variant,want_adapter,want_attn",
    [
        ("base", False, False),
        ("dex_minus", True, True),
        ("dex_plus", True, True),
        ("ungated_adapter", True, True),
        ("attn_only", False, True),
        ("adapter_only", True, False),
    ],
)
def test_variant_trainable_sets(variant, want_adapter, want_attn):
    model = tiny_model()
    cfg = DexConfig(variant=variant, head_selection="entropy_high", allow_no_anneal=True)
    attach_dex(model, cfg, plan=_plan_for(model))
    names = set_trainable(model, cfg)
    rep = trainable_report(model)

    has_adapter = any(is_dex_param_name(n) for n in names)
    has_attn = any(is_attn_trainable_param_name(n) for n in names)
    assert has_adapter is want_adapter
    assert has_attn is want_attn
    if not want_adapter and not want_attn:
        assert names == []
        assert rep["trainable_param_count"] == 0

    # nothing outside the intended set may be trainable
    for n in names:
        assert is_dex_param_name(n) or is_attn_trainable_param_name(n)
    # q_proj / MLP / embeddings must stay frozen in every variant
    for n, p in model.named_parameters():
        if ".q_proj." in n or ".mlp." in n or "embed_tokens" in n:
            assert not p.requires_grad


def test_attn_trainable_set_is_kvo_not_q():
    model = tiny_model()
    cfg = DexConfig(variant="attn_only")
    attach_dex(model, cfg, plan=_plan_for(model))
    names = set_trainable(model, cfg)
    assert all(
        (".k_proj." in n) or (".v_proj." in n) or (".o_proj.base." in n) for n in names
    ), names
    # 2 layers x {k, v, o}
    assert len(names) == 6


def test_base_and_attn_only_forward_match_unpatched_model():
    ids = torch.randint(0, 64, (1, 6))
    ref = tiny_model()
    with torch.no_grad():
        want = ref(input_ids=ids).logits

    for variant in ("base", "attn_only"):
        model = tiny_model()
        cfg = DexConfig(variant=variant)
        attach_dex(model, cfg, plan=_plan_for(model))
        with torch.no_grad():
            got = model(input_ids=ids).logits
        assert torch.allclose(want, got, atol=0.0), variant


def test_dex_changes_the_forward_once_lambda_is_nonzero():
    ids = torch.randint(0, 64, (1, 6))
    model = tiny_model()
    with torch.no_grad():
        before = model(input_ids=ids).logits.clone()
    cfg = DexConfig(variant="dex_minus", lambda_anneal_steps=0, allow_no_anneal=True)
    attach_dex(model, cfg, plan=_plan_for(model))
    with torch.no_grad():
        at_zero = model(input_ids=ids).logits
    # lambda_learn starts at 0 -> DEX is a no-op at init (paper's stability argument)
    assert torch.allclose(before, at_zero, atol=0.0)
    for a in model.modules():
        if isinstance(a, AttentionOutputAdapter):
            with torch.no_grad():
                a.lambda_learn.fill_(0.5)
    with torch.no_grad():
        after = model(input_ids=ids).logits
    assert not torch.allclose(before, after, atol=1e-6)


def test_minus_and_plus_models_match_under_weight_negation():
    ids = torch.randint(0, 64, (1, 7))
    plan = None
    m_minus = tiny_model()
    plan = _plan_for(m_minus)
    attach_dex(m_minus, DexConfig(variant="dex_minus", allow_no_anneal=True), plan=plan)
    m_plus = tiny_model()
    attach_dex(m_plus, DexConfig(variant="dex_plus", allow_no_anneal=True), plan=plan)

    mods_minus = [m for m in m_minus.modules() if isinstance(m, AttentionOutputAdapter)]
    mods_plus = [m for m in m_plus.modules() if isinstance(m, AttentionOutputAdapter)]
    assert len(mods_minus) == len(mods_plus) == 2
    with torch.no_grad():
        for am, ap in zip(mods_minus, mods_plus):
            ap.proj.weight.copy_(-am.proj.weight)
            am.lambda_learn.fill_(0.6)
            ap.lambda_learn.fill_(0.6)
            assert am.selected_heads == ap.selected_heads

    with torch.no_grad():
        lm = m_minus(input_ids=ids).logits
        lp = m_plus(input_ids=ids).logits
    max_abs = (lm - lp).abs().max().item()
    mean_abs = (lm - lp).abs().mean().item()
    print(f"[model-level sign-flip] max_abs_err={max_abs:.3e} mean_abs_err={mean_abs:.3e}")
    assert max_abs < 1e-6


def test_ungated_adapter_equals_plus_with_unit_lambda():
    ids = torch.randint(0, 64, (1, 5))
    plan = None
    m_res = tiny_model()
    plan = _plan_for(m_res)
    attach_dex(m_res, DexConfig(variant="ungated_adapter"), plan=plan)
    m_plus = tiny_model()
    attach_dex(m_plus, DexConfig(variant="dex_plus", lambda_anneal_steps=0, allow_no_anneal=True), plan=plan)
    with torch.no_grad():
        for ar, ap in zip(
            (m for m in m_res.modules() if isinstance(m, AttentionOutputAdapter)),
            (m for m in m_plus.modules() if isinstance(m, AttentionOutputAdapter)),
        ):
            ap.proj.weight.copy_(ar.proj.weight)
            ap.lambda_learn.fill_(1.0)
    with torch.no_grad():
        a = m_res(input_ids=ids).logits
        b = m_plus(input_ids=ids).logits
    assert torch.allclose(a, b, atol=1e-6)


def test_ungated_adapter_lambda_is_fixed_at_one_and_not_trainable():
    model = tiny_model()
    cfg = DexConfig(variant="ungated_adapter")
    attach_dex(model, cfg, plan=_plan_for(model))
    names = set_trainable(model, cfg)
    assert not any(n.endswith("lambda_learn") for n in names)
    for m in model.modules():
        if isinstance(m, AttentionOutputAdapter):
            assert m.current_lambda().item() == pytest.approx(1.0)


def test_adapter_and_attention_parameter_counts_are_matched_across_variants():
    counts = {}
    for variant in ("dex_minus", "dex_plus", "ungated_adapter"):
        model = tiny_model()
        cfg = DexConfig(variant=variant, allow_no_anneal=True)
        attach_dex(model, cfg, plan=_plan_for(model))
        set_trainable(model, cfg)
        rep = trainable_report(model)
        counts[variant] = (rep["adapter_param_count"], rep["attn_param_count"])
    # minus and plus must match exactly; residual has no learnable lambda (-1 param/layer)
    assert counts["dex_minus"] == counts["dex_plus"]
    assert counts["ungated_adapter"][1] == counts["dex_minus"][1]
    assert counts["ungated_adapter"][0] == counts["dex_minus"][0] - 2  # 2 layers x lambda


# --------------------------------------------------------------------------
# gradients and diagnostics
# --------------------------------------------------------------------------


def test_gradients_reach_adapter_and_lambda_only_where_expected():
    ids = torch.randint(0, 64, (1, 6))
    model = tiny_model()
    cfg = DexConfig(variant="dex_minus", lambda_anneal_steps=4)
    attach_dex(model, cfg, plan=_plan_for(model))
    set_trainable(model, cfg)
    set_dex_step(model, 2)  # inside the annealing window: lambda(t) > 0
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()

    grads = {n: (p.grad is not None and p.grad.abs().sum().item() > 0)
             for n, p in model.named_parameters() if p.requires_grad}
    assert any(k.endswith("adapter.proj.weight") and v for k, v in grads.items())
    assert any(k.endswith("adapter.lambda_learn") and v for k, v in grads.items())
    for n, p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None, n


def test_adapter_only_variant_has_no_attention_gradients():
    ids = torch.randint(0, 64, (1, 6))
    model = tiny_model()
    cfg = DexConfig(variant="adapter_only", lambda_anneal_steps=4)
    attach_dex(model, cfg, plan=_plan_for(model))
    set_trainable(model, cfg)
    set_dex_step(model, 2)
    model(input_ids=ids, labels=ids).loss.backward()
    for n, p in model.named_parameters():
        if ".k_proj." in n or ".v_proj." in n or ".o_proj.base." in n:
            assert p.grad is None, n


def test_stats_collection_reports_norm_ratio_and_lambda():
    ids = torch.randint(0, 64, (1, 6))
    model = tiny_model()
    cfg = DexConfig(variant="dex_minus", lambda_anneal_steps=0, allow_no_anneal=True)
    attach_dex(model, cfg, plan=_plan_for(model))
    for m in model.modules():
        if isinstance(m, AttentionOutputAdapter):
            with torch.no_grad():
                m.lambda_learn.fill_(0.5)
    set_dex_stats(model, True)
    with torch.no_grad():
        model(input_ids=ids)
    stats = collect_dex_stats(model)
    assert stats["n_layers"] == 2
    assert stats["lambda"] == pytest.approx(0.5)
    assert stats["corr_ratio"] > 0.0
    assert "cos_delta_o" in stats and "corr_mean" in stats and "corr_std" in stats


def test_o_proj_wrapper_preserves_linear_interface():
    model = tiny_model()
    attach_dex(model, DexConfig(variant="dex_minus", allow_no_anneal=True), plan=_plan_for(model))
    proj = model.model.layers[0].self_attn.o_proj
    assert isinstance(proj, DexOutputProjection)
    assert proj.in_features == NUM_HEADS * HEAD_DIM
    assert proj.out_features == model.config.hidden_size
    assert proj.weight.shape == (model.config.hidden_size, NUM_HEADS * HEAD_DIM)


def test_double_attach_is_rejected():
    model = tiny_model()
    plan = _plan_for(model)
    attach_dex(model, DexConfig(variant="dex_minus", allow_no_anneal=True), plan=plan)
    with pytest.raises(RuntimeError):
        attach_dex(model, DexConfig(variant="dex_minus", allow_no_anneal=True), plan=plan)


def test_fixed_lambda_without_annealing_uses_layer_lambda_init():
    """T=0 + non-learnable lambda pins lambda to this layer's lambda_init."""
    a = make_adapter(-1.0, lambda_anneal_steps=0, lambda_learnable=False,
                     lambda_init=0.55, lambda_learn_init=0.0)
    for t in (0, 7, 500):
        a.set_step(t)
        assert a.current_lambda().item() == pytest.approx(0.55)


def test_fixed_lambda_leaves_residual_adapter_semantics_unchanged():
    model = tiny_model()
    cfg = DexConfig(variant="ungated_adapter")
    attach_dex(model, cfg, plan=_plan_for(model))
    for m in model.modules():
        if isinstance(m, AttentionOutputAdapter):
            assert m.current_lambda().item() == pytest.approx(1.0)
            assert not isinstance(m.lambda_learn, torch.nn.Parameter)


# --------------------------------------------------------------------------
# regressions for the review findings
# --------------------------------------------------------------------------


def test_paper_style_variant_refuses_to_run_without_annealing():
    for variant in ("dex_minus", "dex_plus", "adapter_only"):
        with pytest.raises(ValueError, match="lambda_anneal_steps"):
            DexConfig(variant=variant).resolve()
        DexConfig(variant=variant, lambda_anneal_steps=10).resolve()      # fine
        DexConfig(variant=variant, allow_no_anneal=True).resolve()        # explicit opt-in


def test_dead_configuration_is_rejected():
    with pytest.raises(ValueError, match="dead configuration"):
        DexConfig(variant="dex_minus", fd_init="zeros", lambda_learn_init=0.0,
                  lambda_anneal_steps=0, allow_no_anneal=True).resolve()


def test_layer_subset_restricts_the_unfrozen_attention_too():
    model = tiny_model()
    cfg = DexConfig(variant="dex_minus", lambda_anneal_steps=4, layers=(1,))
    rep = attach_dex(model, cfg, plan=_plan_for(model))
    assert rep["layers"] == [1]
    names = set_trainable(model, cfg)
    assert names, "layer 1 must still be trainable"
    for n in names:
        assert ".layers.1." in n, f"layer-0 parameter unfrozen by a layer-1 run: {n}"


def test_head_plan_criterion_mismatch_raises():
    entropy_plan = {"criterion": "entropy", "scores": {"0": [0.1, 0.9, 0.5, 0.7]}}
    cfg = DexConfig(head_selection="importance_low", heads_per_layer=2,
                    allow_no_anneal=True).resolve()
    with pytest.raises(ValueError, match="importance"):
        select_heads_for_layer(0, 4, cfg, entropy_plan)


def test_importance_scores_are_read_from_the_importance_block():
    plan = {
        "criterion": "entropy",
        "scores": {"0": [0.1, 0.9, 0.5, 0.7]},          # entropy
        "importance": {"0": [0.9, 0.1, 0.7, 0.5]},      # different ordering
    }
    cfg = DexConfig(head_selection="importance_low", heads_per_layer=2,
                    allow_no_anneal=True).resolve()
    assert select_heads_for_layer(0, 4, cfg, plan) == (1, 3)


def test_lambda_stays_fp32_when_the_backbone_is_bf16():
    model = tiny_model().to(torch.bfloat16)
    attach_dex(model, DexConfig(variant="dex_minus", lambda_anneal_steps=4),
               plan=_plan_for(model))
    for m in model.modules():
        if isinstance(m, AttentionOutputAdapter):
            assert m.lambda_learn.dtype == torch.float32
            assert m.proj.weight.dtype == torch.bfloat16


def test_raw_correction_cosine_is_the_sign_free_paper_quantity():
    ids = torch.randint(0, 64, (1, 6))
    stats = {}
    for variant in ("dex_minus", "dex_plus"):
        model = tiny_model()
        attach_dex(model, DexConfig(variant=variant, lambda_anneal_steps=0,
                                    allow_no_anneal=True), plan=_plan_for(model))
        for m in model.modules():
            if isinstance(m, AttentionOutputAdapter):
                with torch.no_grad():
                    m.lambda_learn.fill_(0.5)
        set_dex_stats(model, True)
        with torch.no_grad():
            model(input_ids=ids)
        stats[variant] = collect_dex_stats(model)
        # layer 0 only: deeper layers see different inputs once the sign differs
        first = next(m for m in model.modules() if isinstance(m, AttentionOutputAdapter))
        stats[variant + "@0"] = dict(first.last_stats)
    # the applied update flips sign with the variant ...
    assert stats["dex_minus"]["cos_delta_o"] == pytest.approx(
        -stats["dex_minus"]["cos_raw_corr_o"], abs=1e-6)
    # ... while cos(O, lambda f_D(O)) is sign-free: identical W_D and identical
    # layer-0 input give the same value for both signs
    assert stats["dex_minus@0"]["cos_raw_corr_o"] == pytest.approx(
        stats["dex_plus@0"]["cos_raw_corr_o"], abs=1e-6)
    assert stats["dex_plus@0"]["cos_delta_o"] == pytest.approx(
        +stats["dex_plus@0"]["cos_raw_corr_o"], abs=1e-6)


def test_o_proj_wrapper_rejects_a_head_shape_mismatch():
    from deltamem.core.dex import DexOutputProjection
    lin = torch.nn.Linear(NUM_HEADS * HEAD_DIM, 16, bias=False)
    with pytest.raises(ValueError, match="in_features"):
        DexOutputProjection(lin, None, NUM_HEADS + 1, HEAD_DIM)


def test_head_plan_without_a_scores_block_still_resolves():
    """plan.get(want, plan["scores"]) used to KeyError on a named-block-only plan."""
    plan = {"criterion": "entropy", "entropy": {"0": [0.1, 0.9, 0.5, 0.7]}}
    cfg = DexConfig(head_selection="entropy_high", heads_per_layer=2,
                    allow_no_anneal=True).resolve()
    assert select_heads_for_layer(0, 4, cfg, plan) == (1, 3)


def test_head_plan_missing_every_block_raises_keyerror():
    cfg = DexConfig(head_selection="entropy_high", heads_per_layer=2,
                    allow_no_anneal=True).resolve()
    with pytest.raises(KeyError):
        select_heads_for_layer(0, 4, cfg, {"criterion": "entropy"})


def test_adapterless_variants_are_not_hit_by_the_dead_config_check():
    for variant in ("base", "attn_only"):
        cfg = DexConfig(variant=variant, fd_init="zeros", lambda_learn_init=0.0,
                        lambda_anneal_steps=0).resolve()
        assert cfg.adapter_enabled is False


def test_ungated_adapter_with_zero_init_is_not_dead():
    # lambda resolves to 1, so W_D=0 still receives gradient
    cfg = DexConfig(variant="ungated_adapter", fd_init="zeros").resolve()
    assert cfg.lambda_init == 1.0
