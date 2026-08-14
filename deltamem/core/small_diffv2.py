"""Official-style Differential Transformer V2 attention, for FROM-SCRATCH training.

This is chain B's `small_diffv2_from_scratch` arm.  It is deliberately NOT the
post-hoc split in `diff_split.py`: there the negative query is a zero-init delta
on top of a frozen pretrained q_proj, here the doubled query projection is
trained from random init exactly as in the official V2 reference
(microsoft/unilm @ 833df7e7832e5064a281131ee64a481afa8e5b95,
Diff-Transformer/Diff-Transformer-V2/multihead_flashdiffv2.py):

    self.num_q_heads = 2 * self.num_heads          # query heads doubled
    self.k_proj      = Linear(d, num_kv_heads*hd)  # KV heads UNCHANGED
    self.o_proj      = Linear(num_heads*hd, d)     # o_proj input NOT doubled
    attn1, attn2 = attn[:, :, 0::2], attn[:, :, 1::2]        # interleaved
    attn = attn1 - sigmoid(lambda_val).unsqueeze(-1) * attn2 # before o_proj

V2 has no post-differential per-head RMSNorm (that was V1) and the gate is a
plain input-dependent sigmoid, so both are reproduced as-is.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.qwen3 import modeling_qwen3 as _q
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention


class _GQAProxy:
    """Delegates to the real module but reports the DOUBLED gqa repeat factor.

    `eager_attention_forward` / sdpa read `module.num_key_value_groups` to
    expand K/V; with 2H query heads that must be 2H/G, not H/G.
    """

    def __init__(self, base, groups):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "num_key_value_groups", groups)

    def __getattr__(self, k):
        return getattr(object.__getattribute__(self, "_base"), k)


class DiffV2Attention(nn.Module):
    """Drop-in replacement for Qwen3Attention with doubled query heads."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim",
                                config.hidden_size // config.num_attention_heads)
        self.n_heads = config.num_attention_heads          # output heads H
        self.n_kv = config.num_key_value_heads             # G, UNCHANGED
        self.num_key_value_groups = self.n_heads // self.n_kv
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.sliding_window = None

        d = config.hidden_size
        # query heads doubled; K/V untouched; o_proj input NOT doubled
        self.q_proj = nn.Linear(d, 2 * self.n_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(d, self.n_kv * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(d, self.n_kv * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.q_norm = _q.Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = _q.Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # input-dependent, per-output-head sigmoid gate
        self.lambda_proj = nn.Linear(d, self.n_heads, bias=False)
        nn.init.zeros_(self.lambda_proj.weight)   # sigmoid(0) = 0.5 at init

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        B, L = input_shape
        D = self.head_dim

        q = self.q_norm(self.q_proj(hidden_states).view(B, L, 2 * self.n_heads, D)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(B, L, self.n_kv, D)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, self.n_kv, D).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = _q.apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)

        fn = _q.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, _q.eager_attention_forward)
        proxy = _GQAProxy(self, (2 * self.n_heads) // self.n_kv)
        attn, w = fn(proxy, q, k, v, attention_mask,
                     dropout=0.0 if not self.training else self.attention_dropout,
                     scaling=self.scaling, sliding_window=self.sliding_window, **kwargs)
        # attention backends return [B, seq, heads, head_dim]
        assert attn.shape[1] == L and attn.shape[2] == 2 * self.n_heads, attn.shape

        a1 = attn[:, :, 0::2, :]                                  # [B, L, H, D]
        a2 = attn[:, :, 1::2, :]
        lam = torch.sigmoid(self.lambda_proj(hidden_states))       # [B, L, H]
        out = a1 - lam.unsqueeze(-1) * a2
        return self.o_proj(out.reshape(B, L, -1).contiguous()), w


def convert_to_diffv2(model) -> list[str]:
    """Replace every Qwen3Attention with DiffV2Attention (random init)."""
    cfg = model.config.get_text_config()
    done = []
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, Qwen3Attention):
            continue
        parent, attr = model, None
        parts = name.split(".")
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        attr = parts[-1]
        new = DiffV2Attention(cfg, mod.layer_idx)
        setattr(parent, attr, new.to(next(mod.parameters()).device,
                                     next(mod.parameters()).dtype))
        done.append(name)
    return done
