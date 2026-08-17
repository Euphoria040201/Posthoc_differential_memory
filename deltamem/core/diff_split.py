"""Post-hoc function-preserving differential head splitting for a trained Qwen3.

Converts a pretrained dense attention layer into a DIFF-V2-style differential
attention *without retraining the backbone and without changing the function at
initialisation*.

Construction (one wrapped layer):

    Q+_{t,h} = q_norm(q_proj(h_t))_h                     (bit-identical to base)
    R_t      = LocalRead_phi(H_{t-w..t})                  (causal, window w)
    dQ_{t,h} = B_h R_t                                    (zero-init  =>  dQ = 0)
    Q-_{t,h} = Q+_{t,h} + dQ_{t,h}

    O^±      = Attn(Q^±, K, V)     same K, V, mask, RoPE, position_ids
    O~_{t,h} = O+_{t,h} + gamma * (O+_{t,h} - O-_{t,h})
    Y_t      = W_O concat_h(O~_{t,h})

At init dQ = 0, so O+ = O- and O~ = O+ for ANY fixed gamma: the layer is exactly
the base layer. gamma = 1 gives O~ = 2 O+ - O-, the (normalised) lambda_0 = 0.5
differential form, with no dynamic denominator.

GQA correctness. The two branches are concatenated INTERLEAVED, `[+,-,+,-,...]`,
exactly as official DIFF V2 does with `attn[..., 0::2] / attn[..., 1::2]`. With
2H query heads over G kv heads the repeat factor is 2H/G, so new head 2i and new
head 2i+1 both map to kv head (2i)//(2H/G) = i//(H/G) — the *same* kv group the
original head i used. A front/back concatenation would map the pair into two
different kv groups and silently break both GQA and function preservation.

Nothing is added to the KV cache: K/V are computed and cached exactly as the base
layer does, and both query branches read that one cache.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers.models.qwen3.modeling_qwen3 as _q


class _GQAProxy:
    """Delegates to the frozen base attention but reports the DOUBLED gqa repeat.

    `eager/sdpa/flash_attention_forward` all read `module.num_key_value_groups`
    to expand K/V.  With 2H query heads over G kv heads the correct repeat is
    2H/G, not the base module's H/G.  Everything else (config, is_causal,
    training, layer_idx, ...) is forwarded to the real module unchanged.
    """

    def __init__(self, base, groups):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "num_key_value_groups", groups)

    def __getattr__(self, k):
        return getattr(object.__getattribute__(self, "_base"), k)


def _local_causal_mask(L: int, w: int, dtype, device):
    """[L, L] additive mask: token t may read t-w+1 .. t (inclusive), nothing later."""
    idx = torch.arange(L, device=device)
    rel = idx[:, None] - idx[None, :]           # >0 past, 0 self, <0 future
    keep = (rel >= 0) & (rel < w)
    m = torch.zeros(L, L, dtype=dtype, device=device)
    return m.masked_fill(~keep, torch.finfo(dtype).min)


class DiffSplitAttention(nn.Module):
    """Wrap a Qwen3Attention with a second, learned-offset query branch.

    Trainable: ``read_q``, ``read_k``, ``delta_q`` (and ``gate`` in dynamic mode).
    Everything reached through ``self.base`` stays frozen and is used verbatim.
    """

    def __init__(self, base: nn.Module, hidden_size: int, *, read_dim: int = 128,
                 window: int = 256, gamma: float = 1.0, dynamic_gate: bool = False,
                 enabled: bool = True):
        super().__init__()
        self.base = base
        self.hidden_size = hidden_size
        self.head_dim = base.head_dim
        self.n_heads = base.q_proj.out_features // base.head_dim
        self.n_kv = base.k_proj.out_features // base.head_dim
        self.read_dim = read_dim
        self.window = window
        self.gamma = float(gamma)
        self.enabled = enabled
        self.dynamic_gate = dynamic_gate

        # audit issue 3 (2026-08-17): the reader's values are the frozen backbone V
        # averaged over kv heads, i.e. a vector of width head_dim, and are consumed
        # by delta_q whose in_features is read_dim.  The module therefore only makes
        # sense when read_dim == head_dim; with any other value `_read_core` returns
        # a [.., head_dim] tensor into a [read_dim -> ..] projection and the failure
        # is a shape error far from its cause (or, worse, silently valid when the
        # two happen to be broadcastable).  Enforce it at construction.
        if read_dim != self.head_dim:
            raise ValueError(
                f"DiffSplitAttention requires read_dim == head_dim "
                f"(reader values are backbone V of width {self.head_dim}); "
                f"got read_dim={read_dim}. This constraint was implicit before "
                f"2026-08-17 and unenforced.")

        # --- local reader: one attention head over the last `window` hidden states.
        # Values are the FROZEN backbone v_proj output (memory_value_source="main_v"
        # in the old sidecar), so the reader adds no value projection of its own and
        # the parameter shapes match the old sidecar exactly.
        self.read_q = nn.Linear(hidden_size, read_dim, bias=False)
        self.read_k = nn.Linear(hidden_size, read_dim, bias=False)
        nn.init.normal_(self.read_q.weight, std=0.02)
        nn.init.normal_(self.read_k.weight, std=0.02)

        # --- query offset, ZERO-INIT so the split is function-preserving at step 0
        self.delta_q = nn.Linear(read_dim, self.n_heads * self.head_dim, bias=False)
        nn.init.zeros_(self.delta_q.weight)

        self.gate = None
        if dynamic_gate:
            # gamma_{t,h} = 2*sigmoid(gate(R_t)); zero-init weight => sigmoid(0)=.5 => gamma=1
            self.gate = nn.Linear(read_dim, self.n_heads, bias=False)
            nn.init.zeros_(self.gate.weight)

        # runtime diagnostics, filled by forward when requested
        self.collect_stats = False
        self.last_stats: dict[str, float] = {}
        # causality probe hook: force the reader to see only up to this many tokens
        self._read_override: torch.Tensor | None = None
        self._zero_read = False
        self._shuffle_read = False
        self._shuffle_causal = False
        # audit issue 1 (2026-08-17): these two tensors ARE per-sequence inference
        # state.  The KV cache is unchanged, but the module is NOT state-free, so
        # "KV-cache-free" must not be read as "no extra inference state".  Total
        # state is measured by scripts/measure_inference_state.py.
        # audit issue 4: `_read_core` applies only the causal-window mask and
        # ignores attention_mask, so PADDED positions are attended by the reader
        # and left-padded batches change dQ at real positions.  The training and
        # evaluation paths used packed, unpadded sequences, so no reported number
        # is affected, but the module must not be used with padded batches.
        self._read_h = None
        self._read_v = None

    # ------------------------------------------------------------------ reader
    def reset_read_cache(self):
        self._read_h = None
        self._read_v = None

    def _local_read_cached(self, hidden_states, v_main, past_len: int):
        """Reader for incremental decoding: prepend up to w-1 cached hidden/V states.

        Without this the reader would see only the newly generated token and the
        decode path would silently disagree with the full forward (measured 3.3e-3
        before this was added).
        """
        if past_len == 0:                      # a fresh prefill starts a new sequence
            self.reset_read_cache()
        h_prev, v_prev = self._read_h, self._read_v
        if h_prev is not None and h_prev.shape[0] != hidden_states.shape[0]:
            h_prev = v_prev = None             # batch changed -> cache is not ours
        h_all = hidden_states if h_prev is None else torch.cat([h_prev, hidden_states], 1)
        v_all = v_main if v_prev is None else torch.cat([v_prev, v_main], 2)
        C = 0 if h_prev is None else h_prev.shape[1]
        R = self._read_core(h_all, v_all, q_offset=C, n_query=hidden_states.shape[1])
        keep = self.window - 1
        if keep > 0:
            self._read_h = h_all[:, -keep:].detach()
            self._read_v = v_all[:, :, -keep:].detach()
        else:
            self.reset_read_cache()
        return R

    def _read_core(self, h_all, v_all, *, q_offset: int, n_query: int):
        """R for the last `n_query` positions of `h_all`, causal window `w`."""
        B, T, _ = h_all.shape
        if self._zero_read:
            return torch.zeros(B, n_query, self.read_dim, dtype=h_all.dtype,
                               device=h_all.device)
        q = self.read_q(h_all[:, q_offset:])                     # [B, n_query, dr]
        k = self.read_k(h_all)                                   # [B, T, dr]
        v = v_all.mean(dim=1)                                    # [B, T, hd]
        qi = torch.arange(q_offset, q_offset + n_query, device=h_all.device)
        kj = torch.arange(T, device=h_all.device)
        rel = qi[:, None] - kj[None, :]
        keep = (rel >= 0) & (rel < self.window)
        mask = torch.zeros(n_query, T, dtype=q.dtype, device=q.device)
        mask = mask.masked_fill(~keep, torch.finfo(q.dtype).min)
        scores = torch.baddbmm(mask.expand(B, n_query, T), q, k.transpose(1, 2),
                               alpha=self.read_dim ** -0.5, beta=1.0)
        probs = torch.softmax(scores, dim=-1)
        if self._shuffle_read:
            # audit issue 2 (2026-08-17): INVALID.  `probs[:, :, perm]` permutes
            # the whole key axis AFTER causal masking, so probability mass that
            # belonged to a past key lands on a FUTURE key.  The ablation could
            # therefore leak future tokens and cannot support any claim about the
            # reader's use of window content.  Use shuffle_causal instead.
            raise RuntimeError(
                "diff_split: the post-softmax full-axis shuffle is not a valid "
                "causal ablation (it can move probability mass onto future "
                "positions). Use set_read_control(shuffle_causal=True).")
        if self._shuffle_causal:
            probs = self._permute_within_window(probs, qi, kj, keep)
        return torch.bmm(probs, v)

    @staticmethod
    def _permute_within_window(probs, qi, kj, keep):
        """Valid ablation: permute probabilities only WITHIN each query's window.

        For query t the allowed key set is exactly {t-w+1..t}.  Permuting inside
        that set destroys which position in the window a probability was assigned
        to, while making it impossible for mass to reach a key the causal mask
        forbids: masked entries are zero before and after, so no future token can
        contribute.  This is the ablation the shuffle claim needed.
        """
        n_query, T = keep.shape
        out = torch.zeros_like(probs)
        # random permutation of the allowed indices, per query row
        noise = torch.rand(n_query, T, device=probs.device).masked_fill(~keep, 2.0)
        order = noise.argsort(dim=-1)                    # allowed idx first, shuffled
        n_allowed = keep.sum(-1)                         # [n_query]
        src_pos = keep.float().cumsum(-1).long() - 1     # rank of each allowed key
        valid = keep & (src_pos >= 0)
        rows = torch.arange(n_query, device=probs.device)[:, None].expand(n_query, T)
        tgt = order.gather(1, src_pos.clamp(min=0))      # where each key's mass goes
        out[:, rows[valid], tgt[valid]] = probs[:, rows[valid], kj.expand(n_query, T)[valid]]
        # Mass conservation is checked in fp32: summing thousands of bf16 terms
        # carries far more error than any meaningful tolerance on the raw dtype.
        s_out, s_in = out.float().sum(-1), probs.float().sum(-1)
        assert torch.allclose(s_out, s_in, atol=1e-2, rtol=1e-3), \
            f"shuffle_causal lost mass: max|d|={float((s_out - s_in).abs().max()):.3e}"
        # Causality is exact and must stay exact: nothing outside the window.
        assert float((out * (~keep).to(out.dtype)).abs().sum()) == 0.0, \
            "shuffle_causal leaked mass outside the causal window"
        _ = n_allowed
        return out

    def _local_read(self, hidden_states: torch.Tensor, v_main: torch.Tensor) -> torch.Tensor:
        """R_t over H[t-w+1..t]; values are the frozen backbone V (mean over kv heads).

        v_main: [B, n_kv, L, head_dim] as produced by the base v_proj (pre-RoPE, V has
        no RoPE in Qwen3). Averaged over kv heads to a single read head of width
        head_dim == read_dim.
        """
        return self._read_core(hidden_states, v_main, q_offset=0,
                               n_query=hidden_states.shape[1])

    # ----------------------------------------------------------------- forward
    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, **kwargs):
        base = self.base
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, base.head_dim)

        # ---- identical to base up to and including q_norm / k_norm -------------
        q_plus = base.q_norm(base.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = base.k_norm(base.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = base.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if not self.enabled:
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

        # ---- negative branch query: Q- = Q+ + B R_t (zero-init => equal at step 0)
        if self._read_override is not None:
            R = self._read_override
        elif past_key_values is not None:
            past_len = past_key_values.get_seq_length(base.layer_idx)
            R = self._local_read_cached(hidden_states, value_states, past_len)
        else:
            R = self._local_read(hidden_states, value_states)           # [B, L, dr]
        dq = self.delta_q(R).view(*input_shape, self.n_heads, self.head_dim).transpose(1, 2)
        q_minus = q_plus + dq

        # ---- interleave [+,-,+,-,...] so pair (2i, 2i+1) shares head i's kv group
        B_, H, L, D = q_plus.shape
        q_cat = torch.stack((q_plus, q_minus), dim=2).reshape(B_, 2 * H, L, D)

        # ---- SAME RoPE, SAME position ids, SAME cache for both branches ---------
        cos, sin = position_embeddings
        q_cat, key_states = _q.apply_rotary_pos_emb(q_cat, key_states, cos, sin)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, base.layer_idx)

        fn = _q.ALL_ATTENTION_FUNCTIONS.get_interface(
            base.config._attn_implementation, _q.eager_attention_forward)
        # the attention interface derives its GQA repeat from q/kv head counts
        proxy = _GQAProxy(base, (2 * self.n_heads) // self.n_kv)
        attn_cat, attn_w = fn(
            proxy, q_cat, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else base.attention_dropout,
            scaling=base.scaling, sliding_window=base.sliding_window, **kwargs)
        # transformers' eager/sdpa/flash interfaces all return [B, seq, heads, head_dim]
        # (they end with `.transpose(1, 2).contiguous()`).  Do NOT infer this from the
        # shape: when seq == 2H the two layouts are indistinguishable.
        assert attn_cat.shape[-1] == D and attn_cat.shape[2] == 2 * H, attn_cat.shape
        o_plus = attn_cat[:, :, 0::2, :]                    # [B, L, H, D]
        o_minus = attn_cat[:, :, 1::2, :]

        if self.gate is not None:
            g = 2.0 * torch.sigmoid(self.gate(R)).unsqueeze(-1)          # [B,L,H,1]
        else:
            g = self.gamma
        o_tilde = o_plus + g * (o_plus - o_minus)

        if self.collect_stats:
            with torch.no_grad():
                a = o_plus.detach().float()
                d = (o_plus - o_minus).detach().float()
                t = (o_tilde - o_plus).detach().float()
                na = a.norm(dim=-1).mean().clamp_min(1e-9)
                cos = torch.nn.functional.cosine_similarity(
                    a, o_minus.detach().float(), dim=-1)
                self.last_stats = {
                    "branch_cos": float(cos.mean()),
                    "branch_div_rel": float(d.norm(dim=-1).mean() / na),
                    "correction_rel": float(t.norm(dim=-1).mean() / na),
                    "delta_q_rel": float(dq.detach().float().norm(dim=-1).mean()
                                         / q_plus.detach().float().norm(dim=-1).mean().clamp_min(1e-9)),
                    "gamma_mean": float(g.mean()) if torch.is_tensor(g) else float(g),
                }

        out = o_tilde.reshape(*input_shape, -1).contiguous()
        return base.o_proj(out), attn_w


# --------------------------------------------------------------------------
# attach / control helpers
# --------------------------------------------------------------------------
def _parent(root, name):
    parts = name.split(".")
    p = root
    for part in parts[:-1]:
        p = getattr(p, part)
    return p, parts[-1]


def attach_diff_split(model, layers, *, read_dim=128, window=256, gamma=1.0,
                      dynamic_gate=False):
    """Replace Qwen3Attention with DiffSplitAttention on the given layer indices."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    text_cfg = (model.config.get_text_config()
                if hasattr(model.config, "get_text_config") else model.config)
    hidden = text_cfg.hidden_size
    want = set(layers)
    replaced = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, Qwen3Attention) or module.layer_idx not in want:
            continue
        parent, attr = _parent(model, name)
        setattr(parent, attr, DiffSplitAttention(
            module, hidden, read_dim=read_dim, window=window, gamma=gamma,
            dynamic_gate=dynamic_gate).to(
                next(module.parameters()).device, next(module.parameters()).dtype))
        replaced.append(name)
    return replaced


def iter_diff_modules(model):
    for m in model.modules():
        if isinstance(m, DiffSplitAttention):
            yield m


DIFF_PARAM_MARKERS = ("read_q", "read_k", "delta_q", "gate")


def is_diff_param_name(name: str) -> bool:
    return any(f".{m}." in name for m in DIFF_PARAM_MARKERS)


def freeze_backbone_keep_diff(model):
    for n, p in model.named_parameters():
        p.requires_grad_(is_diff_param_name(n))
    return [n for n, p in model.named_parameters() if p.requires_grad]


def set_diff_enabled(model, flag: bool):
    for m in iter_diff_modules(model):
        m.enabled = bool(flag)


def set_diff_stats(model, flag: bool):
    for m in iter_diff_modules(model):
        m.collect_stats = bool(flag)
        if not flag:
            m.last_stats = {}


def collect_diff_stats(model) -> dict:
    rows = [m.last_stats for m in iter_diff_modules(model) if m.last_stats]
    if not rows:
        return {}
    keys = set(rows[0])
    for r in rows:
        keys &= set(r)
    out = {k: float(sum(r[k] for r in rows) / len(rows)) for k in sorted(keys)}
    out["n_layers"] = len(rows)
    return out


def set_read_control(model, *, zero=False, shuffle=False, shuffle_causal=False):
    """Ablation switches for the local reader.

    zero            reader output forced to 0
    shuffle         DEPRECATED/INVALID -- raises when used (see audit issue 2)
    shuffle_causal  permute probabilities within each query's causal window only
    """
    for m in iter_diff_modules(model):
        m._zero_read = bool(zero)
        m._shuffle_read = bool(shuffle)
        m._shuffle_causal = bool(shuffle_causal)
