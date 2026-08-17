"""Token-local low-rank query splitting: a faithful post-hoc DiffV2 approximation.

Motivation
----------
`diff_split.py` (kept unchanged as the LocalRead baseline) builds the negative
query from a *window read*:

    Q- = Q+ + delta_q(LocalRead_phi(H[t-w:t]))

A natively pretrained Differential Transformer has no such reader.  It has two
independently learned query projections applied to the SAME token hidden state.
So the post-hoc conversion that actually approximates it is a low-rank
parameterization of a second query matrix:

    Wq_minus = Wq_plus + Delta_W,     Delta_W = B A,   rank r

    Q-_pre = Wq_plus h_t + B(A(h_t)) = q_proj(h_t) + dQ_t
    Q-     = q_norm(Q-_pre)                          <-- norm AFTER the delta

Placing the delta *before* q_norm is what makes this a reparameterization of a
single query matrix rather than an additive patch on a normalized vector: with
`delta_pre_norm=True` (default) the module is exactly `q_norm(W h)` for
`W = Wq_plus + BA`, which is the native DiffV2 query path.  `delta_pre_norm=False`
reproduces the older post-norm behaviour and exists only for ablation.

    O+, O- = Attn(interleave(Q+, Q-), same K, same V, same RoPE, same mask)
    O~     = O+ + gamma * (O+ - O-)  = (1+gamma) O+ - gamma O-
    Y      = o_proj(O~)

B is zero-initialised, so dQ = 0, Q- == Q+, O~ == O+ and the converted model is
*exactly* the base model in FP32 at conversion time.

Properties this construction has and LocalRead does not
-------------------------------------------------------
* **token-local**: dQ_t depends only on h_t, so there is no reader state, no
  private inference cache, and decode is trivially identical to prefill.
* **padding-safe**: nothing is aggregated across positions, so attention_mask /
  left or right padding cannot change dQ.
* **KV cache untouched**: K/V are the frozen base projections; both query
  branches read the one cache.  Cache shape and byte size are unchanged.

GQA correctness
---------------
Branches are concatenated INTERLEAVED `[+,-,+,-,...]`.  With 2H query heads over
G kv heads the repeat factor is 2H/G, so new heads 2i and 2i+1 both map to kv
group (2i)//(2H/G) = i//(H/G) — the group original head i used.  A front/back
concatenation maps the pair into two different kv groups and silently breaks both
GQA and function preservation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import transformers.models.qwen3.modeling_qwen3 as _q


class _GQAProxy:
    """Delegates to the frozen base attention but reports the DOUBLED gqa repeat.

    `eager/sdpa/flash_attention_forward` read `module.num_key_value_groups` to
    expand K/V.  With 2H query heads over G kv heads the correct repeat is 2H/G,
    not the base module's H/G.
    """

    def __init__(self, base, groups):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "num_key_value_groups", groups)

    def __getattr__(self, k):
        return getattr(object.__getattribute__(self, "_base"), k)


class LowRankSplitAttention(nn.Module):
    """Wrap a Qwen3Attention with a low-rank second query projection.

    Trainable by default: ``lr_A`` (hidden -> r) and ``lr_B`` (r -> H*head_dim).
    Everything reached through ``self.base`` is the frozen pretrained module and
    is used verbatim.
    """

    SPLIT_MARKERS = ("lr_A", "lr_B")

    def __init__(self, base: nn.Module, hidden_size: int, *, rank: int,
                 gamma: float = 1.0, delta_pre_norm: bool = True,
                 enabled: bool = True, a_init_std: float = 0.02):
        super().__init__()
        self.base = base
        self.hidden_size = hidden_size
        self.head_dim = base.head_dim
        self.n_heads = base.q_proj.out_features // base.head_dim
        self.n_kv = base.k_proj.out_features // base.head_dim
        self.rank = int(rank)
        self.gamma = float(gamma)
        self.delta_pre_norm = bool(delta_pre_norm)
        self.enabled = bool(enabled)

        q_out = self.n_heads * self.head_dim
        self.lr_A = nn.Linear(hidden_size, self.rank, bias=False)
        self.lr_B = nn.Linear(self.rank, q_out, bias=False)
        nn.init.normal_(self.lr_A.weight, std=a_init_std)
        nn.init.zeros_(self.lr_B.weight)          # dQ = 0 => exact function preservation

        self.collect_stats = False
        self.last_stats: dict[str, float] = {}

    # ------------------------------------------------------------------ helpers
    def split_param_count(self) -> int:
        return self.rank * (self.hidden_size + self.n_heads * self.head_dim)

    def extra_repr(self) -> str:
        return (f"rank={self.rank}, gamma={self.gamma}, "
                f"delta_pre_norm={self.delta_pre_norm}, enabled={self.enabled}")

    # ----------------------------------------------------------------- forward
    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, **kwargs):
        base = self.base
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, base.head_dim)

        want_attn = bool(kwargs.get("output_attentions", False))
        if want_attn and self.enabled:
            # The backend would return a 2H-head tensor whose head axis is the
            # interleaved +/- branches, which is NOT interchangeable with the
            # base model's H-head attention.  Returning it silently is the bug
            # this branch exists to prevent.  `effective_attention=True` opts in
            # to the documented H-head tensor A_eff = (1+g)A+ - g A-, which
            # satisfies O~ = A_eff @ V exactly and whose rows still sum to 1.
            if not kwargs.pop("effective_attention", False):
                raise NotImplementedError(
                    "LowRankSplitAttention: output_attentions=True returns 2H "
                    "interleaved branch heads, not base-comparable H-head "
                    "attention. Pass effective_attention=True to receive the "
                    "documented H-head tensor A_eff=(1+gamma)A_plus - gamma*A_minus, "
                    "or disable the split (set_split_enabled(model, False)).")
        else:
            kwargs.pop("effective_attention", None)

        q_pre = base.q_proj(hidden_states)                      # [B, L, H*D]

        if not self.enabled:
            q_plus = base.q_norm(q_pre.view(hidden_shape)).transpose(1, 2)
            key_states = base.k_norm(base.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = base.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            q_r, k_r = _q.apply_rotary_pos_emb(q_plus, key_states, cos, sin)
            if past_key_values is not None:
                k_r, value_states = past_key_values.update(k_r, value_states, base.layer_idx)
            fn = _q.ALL_ATTENTION_FUNCTIONS.get_interface(
                base.config._attn_implementation, _q.eager_attention_forward)
            out, w = fn(base, q_r, k_r, value_states, attention_mask,
                        dropout=0.0 if not self.training else base.attention_dropout,
                        scaling=base.scaling, sliding_window=base.sliding_window, **kwargs)
            return base.o_proj(out.reshape(*input_shape, -1).contiguous()), w

        # ---- token-local low-rank query delta -------------------------------
        dq = self.lr_B(self.lr_A(hidden_states))                # [B, L, H*D]

        if self.delta_pre_norm:
            # faithful: q_norm(  (Wq + BA) h  ) -- one query matrix, then norm
            q_plus = base.q_norm(q_pre.view(hidden_shape)).transpose(1, 2)
            q_minus = base.q_norm((q_pre + dq).view(hidden_shape)).transpose(1, 2)
        else:
            # ablation: delta added to the already-normalised query
            q_plus = base.q_norm(q_pre.view(hidden_shape)).transpose(1, 2)
            q_minus = q_plus + dq.view(hidden_shape).transpose(1, 2)

        key_states = base.k_norm(base.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = base.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # ---- interleave [+,-,+,-,...] so pair (2i, 2i+1) shares head i's kv group
        B_, H, L, D = q_plus.shape
        q_cat = torch.stack((q_plus, q_minus), dim=2).reshape(B_, 2 * H, L, D)

        # ---- SAME RoPE, SAME position ids, SAME cache for both branches -------
        cos, sin = position_embeddings
        q_cat, key_states = _q.apply_rotary_pos_emb(q_cat, key_states, cos, sin)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, base.layer_idx)

        fn = _q.ALL_ATTENTION_FUNCTIONS.get_interface(
            base.config._attn_implementation, _q.eager_attention_forward)
        proxy = _GQAProxy(base, (2 * self.n_heads) // self.n_kv)
        attn_cat, attn_w = fn(
            proxy, q_cat, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else base.attention_dropout,
            scaling=base.scaling, sliding_window=base.sliding_window, **kwargs)
        # transformers' eager/sdpa/flash interfaces all return [B, seq, heads, dim]
        # (they end with `.transpose(1, 2).contiguous()`).  Do NOT infer the layout
        # from the shape: when seq == 2H the two layouts are indistinguishable.
        assert attn_cat.shape[-1] == D and attn_cat.shape[2] == 2 * H, attn_cat.shape
        o_plus = attn_cat[:, :, 0::2, :]
        o_minus = attn_cat[:, :, 1::2, :]
        o_tilde = o_plus + self.gamma * (o_plus - o_minus)

        if attn_w is not None:
            # documented H-head effective attention (see the output_attentions guard)
            aw_plus, aw_minus = attn_w[:, 0::2], attn_w[:, 1::2]
            attn_w = (1.0 + self.gamma) * aw_plus - self.gamma * aw_minus

        if self.collect_stats:
            with torch.no_grad():
                a = o_plus.detach().float()
                d = (o_plus - o_minus).detach().float()
                na = a.norm(dim=-1).mean().clamp_min(1e-9)
                self.last_stats = {
                    "branch_cos": float(torch.nn.functional.cosine_similarity(
                        a, o_minus.detach().float(), dim=-1).mean()),
                    "branch_div_rel": float(d.norm(dim=-1).mean() / na),
                    "dq_rel": float(dq.detach().float().norm(dim=-1).mean()
                                    / q_pre.detach().float().norm(dim=-1).mean().clamp_min(1e-9)),
                }

        out = o_tilde.reshape(*input_shape, -1).contiguous()
        return base.o_proj(out), attn_w


# --------------------------------------------------------------------------
# attach / freeze / switch helpers
# --------------------------------------------------------------------------
def _parent(root, name):
    parts = name.split(".")
    p = root
    for part in parts[:-1]:
        p = p[int(part)] if part.isdigit() else getattr(p, part)
    return p, parts[-1]


def attach_lowrank_split(model, layers, *, rank: int, gamma: float = 1.0,
                         delta_pre_norm: bool = True, a_init_std: float = 0.02):
    """Replace Qwen3Attention with LowRankSplitAttention on the given layer indices."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    text_cfg = (model.config.get_text_config()
                if hasattr(model.config, "get_text_config") else model.config)
    hidden = text_cfg.hidden_size
    want = set(int(x) for x in layers)
    replaced = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, Qwen3Attention) or module.layer_idx not in want:
            continue
        parent, attr = _parent(model, name)
        dev = next(module.parameters()).device
        dt = next(module.parameters()).dtype
        setattr(parent, attr, LowRankSplitAttention(
            module, hidden, rank=rank, gamma=gamma, delta_pre_norm=delta_pre_norm,
            a_init_std=a_init_std).to(dev, dt))
        replaced.append(name)
    missing = want - {m.base.layer_idx for m in iter_split_modules(model)}
    if missing:
        raise ValueError(f"requested layers not found / not Qwen3Attention: {sorted(missing)}")
    return replaced


def iter_split_modules(model):
    for m in model.modules():
        if isinstance(m, LowRankSplitAttention):
            yield m


def is_split_param_name(name: str) -> bool:
    return any(f".{m}." in name for m in LowRankSplitAttention.SPLIT_MARKERS)


def split_param_names(model):
    return [n for n, _ in model.named_parameters() if is_split_param_name(n)]


def expected_split_param_count(model) -> int:
    return sum(m.split_param_count() for m in iter_split_modules(model))


def freeze_backbone_keep_split(model):
    """Adapter-only mode: only lr_A / lr_B receive gradient."""
    for n, p in model.named_parameters():
        p.requires_grad_(is_split_param_name(n))
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not names:
        raise RuntimeError("no trainable split parameters found")
    return names


# attention submodules unfrozen by arm D (progressive attention unfreeze)
ATTN_UNFREEZE_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")


def unfreeze_split_layer_attention(model, *, include_norms: bool = True):
    """Arm D: additionally unfreeze q/k/v/o (+ q_norm/k_norm) INSIDE split layers.

    Returns the list of newly-unfrozen parameter names.  This is an architecture
    conversion, not a PEFT result, and must be labelled as such.
    """
    sufs = ATTN_UNFREEZE_SUFFIXES if include_norms else ATTN_UNFREEZE_SUFFIXES[:4]
    newly = []
    for mod in iter_split_modules(model):
        for n, p in mod.base.named_parameters():
            if any(n.startswith(s + ".") or n == s + ".weight" or f".{s}." in n
                   or n.split(".")[0] == s for s in sufs):
                if not p.requires_grad:
                    p.requires_grad_(True)
                    newly.append(n)
    return newly


def set_split_enabled(model, flag: bool):
    """Genuinely switch between base behaviour and the split.

    Returns the number of modules switched so a caller can assert it is non-zero
    (the old `set_steer_enabled` silently reached zero split modules).
    """
    n = 0
    for m in iter_split_modules(model):
        m.enabled = bool(flag)
        n += 1
    return n


def set_split_stats(model, flag: bool):
    for m in iter_split_modules(model):
        m.collect_stats = bool(flag)
        if not flag:
            m.last_stats = {}


def collect_split_stats(model) -> dict:
    rows = [m.last_stats for m in iter_split_modules(model) if m.last_stats]
    if not rows:
        return {}
    keys = set(rows[0])
    for r in rows:
        keys &= set(r)
    out = {k: float(sum(r[k] for r in rows) / len(rows)) for k in sorted(keys)}
    out["n_layers"] = len(rows)
    return out


# --------------------------------------------------------------------------
# strict checkpoint round trip
# --------------------------------------------------------------------------
def split_state_dict(model) -> dict:
    return {n: p.detach().cpu().clone()
            for n, p in model.named_parameters() if is_split_param_name(n)}


def load_split_state_dict(model, state: dict, *, strict: bool = True):
    """Fail-CLOSED loader.

    Rejects (a) any missing expected split key, (b) any unexpected key, and
    (c) any backbone key present in a split-only checkpoint.  The old diff
    loader was fail-open: a checkpoint that silently contained nothing loaded
    cleanly and training reported base numbers as a result.
    """
    expected = set(split_param_names(model))
    got = set(state)
    backbone_like = {k for k in got if not is_split_param_name(k)}
    missing = expected - got
    unexpected = got - expected
    if strict and (missing or unexpected or backbone_like):
        raise KeyError(
            f"strict split checkpoint mismatch: missing={sorted(missing)[:6]} "
            f"unexpected={sorted(unexpected - backbone_like)[:6]} "
            f"backbone_keys_in_split_ckpt={sorted(backbone_like)[:6]}")
    own = dict(model.named_parameters())
    for k, v in state.items():
        if k in own:
            if tuple(own[k].shape) != tuple(v.shape):
                raise ValueError(f"shape mismatch for {k}: {tuple(own[k].shape)} vs {tuple(v.shape)}")
            with torch.no_grad():
                own[k].copy_(v.to(own[k].device, own[k].dtype))
    return {"loaded": len(state), "expected": len(expected)}
