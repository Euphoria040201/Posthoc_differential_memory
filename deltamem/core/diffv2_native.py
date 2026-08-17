"""Native DIFF-V2 attention with an initialization matched to the HF baseline.

This is the corrected chain-B reference (audit issue 9).  `small_diffv2.py` is
kept byte-identical as the module that produced the 2026-08-14 artifacts; do not
use it for new comparisons.

The defect it fixes
-------------------
`small_diffv2.convert_to_diffv2` builds fresh `nn.Linear` modules AFTER the HF
model has been constructed and initialised.  A fresh `nn.Linear` uses PyTorch's
default `kaiming_uniform_(a=sqrt(5))`, i.e. `U(-1/sqrt(fan_in), +1/sqrt(fan_in))`,
while every weight in the vanilla control was initialised by HF's
`_init_weights` as `normal_(0, config.initializer_range)`.  For hidden=512 that
is std 0.0255 uniform versus std 0.02 normal -- a ~27% larger init on exactly
the tensors under test, and a different distribution shape.  Any loss gap
between "vanilla" and "diffv2" then confounds architecture with initialization,
which is fatal when the gap being measured is ~0.005 nats.

Here every new tensor is initialised the way the HF baseline initialises the
tensor it replaces:
  * Linear weights  -> normal_(0, initializer_range), bias -> 0
  * RMSNorm weights -> 1.0
  * lambda_proj     -> 0, so sigmoid(0)=0.5 at step 0 (the V2 reference value)

Architecture is otherwise the official V2 form (microsoft/unilm @ 833df7e,
Diff-Transformer-V2/multihead_flashdiffv2.py): query heads doubled, K/V heads
unchanged, o_proj input NOT doubled, interleaved 0::2 / 1::2 branch split, and
`attn1 - sigmoid(lambda) * attn2` applied before o_proj.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.qwen3 import modeling_qwen3 as _q
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention


class _GQAProxy:
    def __init__(self, base, groups):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "num_key_value_groups", groups)

    def __getattr__(self, k):
        return getattr(object.__getattribute__(self, "_base"), k)


class NativeDiffV2Attention(nn.Module):
    """Official-style DIFF V2 attention, HF-matched initialization."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim",
                                config.hidden_size // config.num_attention_heads)
        self.n_heads = config.num_attention_heads
        self.n_kv = config.num_key_value_heads
        self.num_key_value_groups = self.n_heads // self.n_kv
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.sliding_window = None

        d = config.hidden_size
        self.q_proj = nn.Linear(d, 2 * self.n_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(d, self.n_kv * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(d, self.n_kv * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.q_norm = _q.Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = _q.Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.lambda_proj = nn.Linear(d, self.n_heads, bias=False)
        self.reset_parameters_like_hf(config)

    def reset_parameters_like_hf(self, config):
        std = getattr(config, "initializer_range", 0.02)
        for lin in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.normal_(lin.weight, mean=0.0, std=std)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)
        for nrm in (self.q_norm, self.k_norm):
            nn.init.ones_(nrm.weight)
        nn.init.zeros_(self.lambda_proj.weight)      # sigmoid(0) = 0.5 at init

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
        assert attn.shape[1] == L and attn.shape[2] == 2 * self.n_heads, attn.shape

        a1, a2 = attn[:, :, 0::2, :], attn[:, :, 1::2, :]
        lam = torch.sigmoid(self.lambda_proj(hidden_states))
        out = a1 - lam.unsqueeze(-1) * a2
        return self.o_proj(out.reshape(B, L, -1).contiguous()), w


def convert_to_native_diffv2(model) -> list[str]:
    """Replace every Qwen3Attention with NativeDiffV2Attention (HF-matched init)."""
    cfg = model.config.get_text_config()
    done = []
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, Qwen3Attention):
            continue
        parent, parts = model, name.split(".")
        for p in parts[:-1]:
            parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
        new = NativeDiffV2Attention(cfg, mod.layer_idx)
        setattr(parent, parts[-1], new.to(next(mod.parameters()).device,
                                          next(mod.parameters()).dtype))
        done.append(name)
    return done


def init_stats(model) -> dict:
    """Per-tensor init statistics, so the matched-init claim is auditable."""
    import numpy as np
    rows = {}
    for n, p in model.named_parameters():
        if "self_attn" not in n:
            continue
        a = p.detach().float().cpu().numpy()
        rows[n] = {"std": float(a.std()), "mean": float(a.mean()),
                   "min": float(a.min()), "max": float(a.max()),
                   "numel": int(a.size)}
    return rows
