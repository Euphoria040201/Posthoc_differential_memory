"""Parameter-matched additive sidecar (chain B's `small_param_matched_additive`).

Identical to `diff_split.DiffSplitAttention` in every respect -- same causal
local reader, same read_dim, same tensor shapes, same zero-init, same
pre-`o_proj` injection point -- with exactly one difference:

    diff_split :  Q- = Q+ + delta(R) ; O-=Attn(Q-,K,V) ; O~ = O+ + g(O+ - O-)
    this       :  O~ = O+ + delta(R)                       <-- no second attention

So the negative branch never passes through attention.  Any gap between the two
arms is attributable to the differential attention itself rather than to the
extra trainable capacity, which is what makes this the decisive control.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.qwen3 import modeling_qwen3 as _q
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from deltamem.core.diff_split import DiffSplitAttention, _parent


class AdditiveSidecarAttention(DiffSplitAttention):
    """Reuses the split's reader/cache verbatim; replaces only the fusion."""

    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, **kwargs):
        base = self.base
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, base.head_dim)

        q = base.q_norm(base.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = base.k_norm(base.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = base.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = _q.apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, base.layer_idx)

        fn = _q.ALL_ATTENTION_FUNCTIONS.get_interface(
            base.config._attn_implementation, _q.eager_attention_forward)
        o_plus, w = fn(base, q, k, v, attention_mask,
                       dropout=0.0 if not self.training else base.attention_dropout,
                       scaling=base.scaling, sliding_window=base.sliding_window,
                       **kwargs)                                   # [B, L, H, D]

        # the SAME reader as the split arm, on the same frozen V
        if past_key_values is not None:
            past_len = past_key_values.get_seq_length(base.layer_idx)
            R = self._local_read_cached(hidden_states, v, past_len)
        else:
            R = self._local_read(hidden_states, v)
        # zero-init delta_q => correction is exactly 0 at step 0, same as the split
        corr = self.delta_q(R).view(*input_shape, self.n_heads, self.head_dim)
        o_tilde = o_plus + corr

        if self.collect_stats:
            with torch.no_grad():
                a = o_plus.detach().float()
                na = a.norm(dim=-1).mean().clamp_min(1e-9)
                self.last_stats = {
                    "correction_rel": float(corr.detach().float().norm(dim=-1).mean() / na),
                }
        return base.o_proj(o_tilde.reshape(*input_shape, -1).contiguous()), w


def attach_additive_sidecar(model, layers, *, read_dim=64, window=256) -> list[str]:
    cfg = model.config.get_text_config()
    want, done = set(layers), []
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, Qwen3Attention) or mod.layer_idx not in want:
            continue
        parent, attr = _parent(model, name)
        setattr(parent, attr, AdditiveSidecarAttention(
            mod, cfg.hidden_size, read_dim=read_dim, window=window,
            gamma=1.0, dynamic_gate=False).to(
                next(mod.parameters()).device, next(mod.parameters()).dtype))
        done.append(name)
    return done


def freeze_backbone_keep_sidecar(model):
    from deltamem.core.diff_split import is_diff_param_name
    for n, p in model.named_parameters():
        p.requires_grad_(is_diff_param_name(n))
    return [n for n, p in model.named_parameters() if p.requires_grad]
