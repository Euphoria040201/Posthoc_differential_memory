"""Prefix-Memory Steering: attach an SWA+prefix memory module to a FROZEN
full-attention backbone, and steer its readout back into the backbone.

This follows δ-mem's philosophy (frozen full-attention backbone + attached memory
whose readout produces corrections) but replaces the delta-rule associative state
with **trainable global prefix memory tokens read via sliding-window attention**.

Per patched layer:
  1. backbone runs its normal FULL causal attention (unchanged, frozen).
  2. an attached memory attention over ``[prefix ; hidden]`` with the SWA+prefix
     mask (prefix globally reads context, normal tokens SWA + read prefix) produces
     a per-token readout ``reads``.
  3. ``reads`` steers the backbone via one of:
       * ``deltamem``  : low-rank delta_q/k/v (added before attention) + delta_o
                         (added to the attention output)  -- δ-mem interface.
       * ``residual``  : ``h += gate * Proj(reads)`` on the attention output.

Only the prefix tokens + memory/steer projections train; backbone stays frozen.
NOTE: only delta_o is zero-init; delta_q/k/v use std=1e-3, so step-0 is NOT byte-exactly
the frozen backbone (ΔQ,ΔK,ΔV are ~1e-4 at steer_gain=0.1, tiny but nonzero). For a strictly
exact start, zero-init the delta_q/k/v outputs too (changes the training trajectory, so do it
as a separate re-trained comparison, not mid-run).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers.models.qwen3.modeling_qwen3 as _q

from deltamem.core.global_prefix import build_prefix_attention_mask, SEG_PREFIX, SEG_CTX


# Additive family (historical) and the differential family added for the
# control-variable experiment.  "fixed" is kept as the legacy name of "fixed_add".
ADDITIVE_FUSIONS = ("fixed", "fixed_add", "rms_match", "cosine")
DIFFERENTIAL_FUSIONS = ("fixed_sub", "learned_diff", "variance_diff")
OUTPUT_FUSIONS = ADDITIVE_FUSIONS + DIFFERENTIAL_FUSIONS


@dataclass
class PrefixSteerConfig:
    num_prefix_tokens: int = 64
    sliding_window_size: int = 256
    mem_num_heads: int = 8
    mem_head_dim: int = 64
    steer_mode: str = "deltamem"          # "deltamem" | "residual"
    normal_attends_prefix: bool = True
    prefix_sees_query: bool = True
    # std of the prefix init. It must match the scale of the tensor the memory ACTUALLY
    # sees: `hidden_states` as passed into forward(), which is POST-input_layernorm.
    # Measured on Qwen3-4B @ layer 20: post-LN RMS = 0.704 (the *residual stream* is 6.5,
    # and the embeddings are 0.022 -- both are the wrong reference).
    # The old 0.02 default (embedding scale, correct only for layer-0 prompt tuning) left
    # the prefix 34x too small: near-zero mem_q(prefix) => every slot attends the context
    # near-uniformly => all slots write the same mean summary (measured eff. rank 3.4/64,
    # slot cosine 0.999) and the grad vanishes (4.5e-06, 56000x below delta_o).
    prefix_init_std: float = 0.7
    prefix_init_dist: str = "normal"   # "normal" | "uniform" | "orthogonal" (diversity of write probes)
    # Two-stage memory. False (old) = prefix is a STATIC learned KV prior: it never reads
    # the context (its attention rows ctx[:, :P, :] were computed then DISCARDED), so it is
    # input-independent and structurally cannot hold document memory. True = wire the write
    # path: prefix attends to the context (WRITE), the result residually updates the prefix,
    # and only then do the normal tokens read it (READ). This is what makes it a real memory.
    prefix_write: bool = True
    # READ-side bottleneck (LEGACY, conflicts with pool_reads). True = normal tokens read the
    # PREFIX ONLY. Kept for loading old ckpts; the pool_reads architecture reads
    # [prefix ; SWA window] instead, so this now defaults to False.
    read_prefix_only: bool = False
    memory_mode: str = "residual"   # residual = static_prefix + write; dynamic = write only
    write_ctx_only: bool = False    # WRITE: prefix queries attend CONTEXT only (no self/query/answer)
    # WRITE routing for strict context-only writers. "global" preserves the original
    # behavior (every probe sees every valid history token). "partitioned" divides valid
    # history tokens, in order, into P contiguous ranges and restricts each slot to its own
    # range.  This prevents all slots from converging to the same global mean.  When P is
    # larger than the valid history length, ranges intentionally share tokens so every slot
    # still sees at least one valid key.
    prefix_write_layout: str = "global"   # "global" | "partitioned"
    prefix_write_overlap_tokens: int = 0  # expand each partition on both sides in valid-token order
    # TOKEN-SPECIFIC pooled-prefix READ (DEFAULT method). The normal-token memory attention
    # runs over [prefix ; window] with ONE softmax (real SWA competition):
    #     alpha_{t,i} = softmax_i(q_t k_i / sqrt(d)),  i in {P prefix slots} + {W window}
    #     R_t  = sum_i alpha_{t,i} v_i                  (standard SWA read, unchanged)
    #     M_t  = amax_p( alpha_{t,p} * v_p )            (per-token max over the P WEIGHTED
    #                                                    prefix contributions; depends on q_t)
    #     reads_t = R_t + alpha_scale * M_t             (alpha_scale = _pool_alpha, 1.0)
    # and `reads` goes into delta_q/k/v/o as before. M_t is TOKEN-SPECIFIC (M_t1 != M_t2):
    # it is NOT the old document-level amax over the prefix QUERY rows' outputs (that
    # variant broadcast one near-constant vector to every token and was verified harmful).
    # No parameters. Needs manual attention (SDPA does not expose probs); the max is
    # computed via chunked no-grad argmax + differentiable gather so the [B,h,L,P,hd]
    # contribution tensor is NEVER materialized. Decode recomputes M_t per new token
    # (it depends on q_t) -- only prefix/window K/V are cached.
    # (Ckpt loaders must pass ckpt values explicitly -- old ckpts have pool_reads absent
    # and load with pool_reads=False via c.get("pool_reads", False).)
    pool_reads: bool = True
    # Learnable token-conditioned GATE on the pooled contribution M_t:
    #     reads_t = R_t + g_max*sigmoid(W_g [gate_input] + b_g) (.) M_t   [per-channel/token]
    # The gate lets the model LEARN, per token and per channel, how much prefix-memory pool
    # to admit -- trained jointly. Requires pool_reads=True.
    pool_gate: bool = False
    # gate INPUT. "rm" (default) = W_g[R_t ; M_t]: the gate sees BOTH the token's SWA read
    # AND what was retrieved from the prefix, so it can decide the admit-ratio from the
    # token<->memory MATCH (two docs giving different M_t at the same h_t get different
    # gates). "hidden" = W_g h_t: gate ignores M_t (weaker -- same token always same gate).
    pool_gate_input: str = "rm"    # "rm" | "hidden"
    # gate RANGE. gate = pool_gate_max * sigmoid(...). max=1 => [0,1], can only ATTENUATE M,
    # so at best MATCHES plain pool (never beats it). max=2 => [0,2] and, at bias=0 with
    # zero-init W_g, STARTS EXACTLY at plain pool (2*0.5=1.0) and can push M UP (<=2) or
    # DOWN (>=0) per channel -- a strict superset. Default 2.0 (start == plain pool).
    pool_gate_max: float = 2.0
    # gate BIAS init. With max=2: 0 => start EXACTLY at plain pool (the good baseline).
    pool_gate_bias: float = 0.0
    steer_layers: tuple[int, ...] = ()    # empty => all layers
    # PREFIX subset: layers in steer_layers but NOT in prefix_layers get P=0 (pure SWA steer:
    # delta_qkvo still active, but no prefix tokens / write / pool). Empty => all steer layers
    # get the prefix (current behavior). Use to keep the SWA steer everywhere while removing
    # the prefix from e.g. the middle layers (where healthy seeds ignore it and crashed seeds
    # explode) -- isolates "mid-layer prefix" from "mid-layer steer".
    prefix_layers: tuple[int, ...] = ()
    steer_gain: float = 1.0               # fixed multiplier on the memory correction (δ-mem uses ~0.05)
    share_qkv: bool = False               # reuse the frozen backbone q/k/v for the memory attention (no mem_* params)
    # Value source for the attached memory attention.  "trainable" is the historical
    # mem_q/mem_k/mem_v implementation.  "main_v" keeps independent trainable mem_q/mem_k
    # but reuses the FROZEN backbone v_proj with no value adapter:
    #
    #   V_main = reshape(base.v_proj(h), [B, n_kv, T, base_head_dim])
    #
    # The n_kv Qwen GQA heads are divided into mem_num_heads contiguous equal groups and
    # averaged.  Thus mem_num_heads=n_kv is exact per-head reuse, while mem_num_heads=1 is
    # the compact mean over all backbone KV heads.  This requires mem_head_dim to equal the
    # backbone head_dim and mem_num_heads to divide n_kv.
    memory_value_source: str = "trainable"  # "trainable" | "main_v"
    delta_heads: str = "qkvo"             # which heads get a delta correction (δ-mem uses "qo")
    delta_rank: int = 0                   # >0 => low-rank bottleneck (read_dim->r->out) to cut params
    read_proj_dim: int = 0                # >0 => trainable Linear compressing the memory readout before delta
    # How delta_o is fused with the frozen main-attention output.  "fixed" is the exact
    # historical out + steer_gain * delta_o behavior.  "rms_match" matches each token's
    # detached delta RMS to its main-output RMS, with a bounded scale, then divides by the
    # measured two-branch energy (sqrt(2) once RMS matching is active at gain=1).  At the
    # zero-initialized delta_o it is exactly the frozen output and still has non-zero delta
    # gradient.  "cosine" uses detached c_t=(1+cos(out,delta))/2 in [0,1]; if either vector
    # is zero cos is defined as 0, hence c_t=1/2 (again exact output and live gradient at
    # zero-init).
    # "fixed" (== "fixed_add") | "rms_match" | "cosine" are the historical additive
    # fusions.  The differential family below treats the memory readout as a CONTROL
    # VARIABLE to be subtracted rather than a correction to be added:
    #
    #   fixed_add     out = Y + g C            (historical; the additive baseline)
    #   fixed_sub     out = Y - g C            (same magnitude, opposite sign)
    #   learned_diff  out = Y - lambda C       (lambda: one learned scalar per layer)
    #   variance_diff out = Y - lambda* r_C,   r_C = C - mu_C,  r_Y = Y - mu_Y,
    #                 lambda* = <r_Y, r_C> / (<r_C, r_C> + eps), clamped to
    #                 [0, fusion_lambda_max] -- the closed-form regression coefficient
    #                 that removes the component of Y co-varying with the control.
    #
    # Why this is not the DEX sign question again: DEX's control was f_D(O), a free
    # per-head linear map of the very tensor being corrected, so (W -> -W) made minus
    # and plus the same function class and the sign carried no information.  Here C is
    # produced by a SEPARATE context-aggregation path (the SWA/prefix memory).  The
    # sign is only meaningful while that path is FROZEN: if C is trained jointly with
    # a subtractive fusion it can simply flip its own output and the minus degenerates
    # into the same reparameterisation.  Stage-1 experiments must load a trained
    # sidecar and freeze it.
    output_fusion: str = "fixed"
    output_fusion_eps: float = 1e-6
    output_fusion_scale_max: float = 10.0
    fusion_lambda_init: float = 0.1       # learned_diff: init of the per-layer scalar
    fusion_lambda_max: float = 1.0        # variance_diff: upper clamp on lambda*
    # EMA momentum for the mu_Y / mu_C running means used by variance_diff.  The probe
    # in the DEX study could centre within a nuisance group; real inference has no such
    # group, so the deployable estimator is a running mean maintained during training
    # (or during an explicit calibration pass) and frozen at eval.
    fusion_ema_momentum: float = 0.99
    # Strong P=0 baseline: summarize the WRITE-only history into one per-layer,
    # query-independent vector, then broadcast that vector directly into the existing
    # delta-q/k/v/o steering heads on every subsequent query.  This is persistent,
    # history-conditioned memory without P prefix slots or a query-conditioned memory
    # read.  "attn" uses one learned pooling query with the existing mem_q/k/v;
    # "mean" uses a validity-masked mean of mem_v(history).  Default "none" creates no
    # parameter and preserves old state_dicts and runtime behavior exactly.
    history_pool_mode: str = "none"       # "none" | "attn" | "mean"
    # Additive hybrid used to test whether prefix memory adds value over the strong
    # history-conditioned pooled-steer baseline:
    #
    #   WRITE(history) -> (one pooled vector, P written prefix slots)
    #   READ(query)    -> pooled + gate * Attn(query, written_prefix_only)
    #
    # The prefix contribution deliberately contains NO local-SWA read.  Consequently,
    # disabling it at runtime is an exact same-weights pooled-steer ablation, rather than
    # changing/removing the baseline branch.  "pooled_plus_prefix" currently requires the
    # attention history pool and the strict WRITE-only protocol.
    hybrid_read_mode: str = "none"        # "none" | "pooled_plus_prefix"
    # The additive prefix gate is either a fixed config scalar, a learned bounded scalar,
    # or a learned bounded per-read-channel vector.  Learned gates store logits and use a
    # sigmoid, so the contribution remains in [0,1].  A small non-zero default starts close
    # to pooled steer while preserving gradients into every prefix/write component.
    hybrid_prefix_gate_mode: str = "fixed"  # "fixed" | "learned_scalar" | "learned_channel"
    hybrid_prefix_gate_init: float = 0.1


class PrefixMemSteerAttention(nn.Module):
    """Wrap a base Qwen3Attention with an attached prefix-memory steer branch."""

    def __init__(self, base: nn.Module, config: PrefixSteerConfig, hidden_size: int):
        super().__init__()
        self.base = base
        self.cfg = config
        self.hidden_size = hidden_size
        P = config.num_prefix_tokens
        self.share_qkv = config.share_qkv
        self.head_dim = base.head_dim
        self.n_heads = base.q_proj.out_features // base.head_dim
        self.n_kv = base.k_proj.out_features // base.head_dim
        if config.memory_value_source not in ("trainable", "main_v"):
            raise ValueError(
                "memory_value_source must be 'trainable' or 'main_v', got "
                f"{config.memory_value_source!r}"
            )
        if config.output_fusion not in OUTPUT_FUSIONS:
            raise ValueError(
                f"output_fusion must be one of {OUTPUT_FUSIONS}, got "
                f"{config.output_fusion!r}"
            )
        if config.output_fusion_eps <= 0:
            raise ValueError("output_fusion_eps must be > 0")
        if config.output_fusion_scale_max <= 0:
            raise ValueError("output_fusion_scale_max must be > 0")
        if config.fusion_lambda_max <= 0:
            raise ValueError("fusion_lambda_max must be > 0")
        if not 0.0 <= config.fusion_ema_momentum < 1.0:
            raise ValueError("fusion_ema_momentum must be in [0, 1)")
        if config.output_fusion in DIFFERENTIAL_FUSIONS and config.delta_heads != "o":
            # the differential family is defined on the attention OUTPUT; a q/k/v
            # steer is not a control variable for O and must not silently ride along
            raise ValueError(
                f"output_fusion={config.output_fusion!r} requires delta_heads='o', "
                f"got {config.delta_heads!r}"
            )
        if config.output_fusion != "fixed" and (
            config.steer_mode != "deltamem" or "o" not in set(config.delta_heads)
        ):
            raise ValueError(
                f"output_fusion={config.output_fusion!r} requires "
                "steer_mode='deltamem' and an active delta_o"
            )
        if config.prefix_write_layout not in ("global", "partitioned"):
            raise ValueError(
                "prefix_write_layout must be 'global' or 'partitioned', got "
                f"{config.prefix_write_layout!r}"
            )
        if config.prefix_write_overlap_tokens < 0:
            raise ValueError("prefix_write_overlap_tokens must be non-negative")
        if config.prefix_write_layout == "partitioned" and (
            P <= 0 or not config.prefix_write or not config.write_ctx_only
        ):
            raise ValueError(
                "partitioned prefix WRITE requires num_prefix_tokens > 0, "
                "prefix_write=True, and write_ctx_only=True"
            )
        self.memory_value_source = config.memory_value_source
        self._main_v_group_size = None
        if self.share_qkv and self.memory_value_source == "main_v":
            raise ValueError(
                "share_qkv=True already reuses all backbone Q/K/V projections and conflicts "
                "with memory_value_source='main_v' (independent mem_q/mem_k + shared V)"
            )
        if self.share_qkv:
            # reuse the frozen backbone q/k/v; readout is in query head-space
            self.read_dim = base.q_proj.out_features
        elif self.memory_value_source == "main_v":
            if config.mem_head_dim != self.head_dim:
                raise ValueError(
                    "memory_value_source='main_v' requires mem_head_dim to equal the "
                    f"backbone head_dim ({self.head_dim}), got {config.mem_head_dim}"
                )
            if config.mem_num_heads <= 0 or self.n_kv % config.mem_num_heads != 0:
                raise ValueError(
                    "memory_value_source='main_v' requires mem_num_heads to be a positive "
                    f"divisor of the backbone's {self.n_kv} KV heads, got "
                    f"{config.mem_num_heads}"
                )
            self.read_dim = config.mem_num_heads * self.head_dim
            self._main_v_group_size = self.n_kv // config.mem_num_heads
            self.mem_q = nn.Linear(hidden_size, self.read_dim, bias=False)
            self.mem_k = nn.Linear(hidden_size, self.read_dim, bias=False)
            # Deliberately no trainable value projection.  Keeping the attribute at None
            # makes introspection explicit while adding no state_dict tensor.
            self.mem_v = None
        else:
            self.read_dim = config.mem_num_heads * config.mem_head_dim
            self.mem_q = nn.Linear(hidden_size, self.read_dim, bias=False)
            self.mem_k = nn.Linear(hidden_size, self.read_dim, bias=False)
            self.mem_v = nn.Linear(hidden_size, self.read_dim, bias=False)

        if config.hybrid_read_mode not in ("none", "pooled_plus_prefix"):
            raise ValueError(
                "hybrid_read_mode must be 'none' or 'pooled_plus_prefix', got "
                f"{config.hybrid_read_mode!r}"
            )
        self._hybrid_pool_prefix = config.hybrid_read_mode == "pooled_plus_prefix"

        self.history_pool_query = None
        if config.history_pool_mode not in ("none", "attn", "mean"):
            raise ValueError(
                "history_pool_mode must be 'none', 'attn', or 'mean', got "
                f"{config.history_pool_mode!r}"
            )
        if config.history_pool_mode != "none":
            if P != 0 and not self._hybrid_pool_prefix:
                raise ValueError("history_pool_mode is a P=0 baseline; set num_prefix_tokens=0")
            if (
                not self._hybrid_pool_prefix
                and (config.prefix_write or config.pool_reads or config.read_prefix_only)
            ):
                raise ValueError(
                    "history_pool_mode cannot create/read prefix slots: set prefix_write=False, "
                    "pool_reads=False, and read_prefix_only=False"
                )
            if config.history_pool_mode == "attn":
                # One learned, query-independent WRITE probe per layer.  It is initialized at
                # the same hidden-state scale as a prefix write probe, but its attention output
                # stays in read_dim and feeds delta-q/k/v/o directly (no write_proj).
                self.history_pool_query = nn.Parameter(torch.empty(1, hidden_size))
                nn.init.normal_(self.history_pool_query, std=config.prefix_init_std)
        if self._hybrid_pool_prefix:
            if config.history_pool_mode != "attn":
                raise ValueError(
                    "hybrid pooled_plus_prefix requires history_pool_mode='attn'"
                )
            if P <= 0:
                raise ValueError(
                    "hybrid pooled_plus_prefix requires num_prefix_tokens > 0"
                )
            if not config.prefix_write or not config.write_ctx_only:
                raise ValueError(
                    "hybrid pooled_plus_prefix requires prefix_write=True and "
                    "write_ctx_only=True"
                )
            if config.pool_reads or config.read_prefix_only:
                raise ValueError(
                    "hybrid pooled_plus_prefix owns its prefix-only additive reader; set "
                    "pool_reads=False and read_prefix_only=False"
                )
            if config.hybrid_prefix_gate_mode not in (
                "fixed",
                "learned_scalar",
                "learned_channel",
            ):
                raise ValueError(
                    "hybrid_prefix_gate_mode must be 'fixed', 'learned_scalar', or "
                    f"'learned_channel', got {config.hybrid_prefix_gate_mode!r}"
                )
            gate_init = config.hybrid_prefix_gate_init
            valid_gate_init = (
                0.0 < gate_init <= 1.0
                if config.hybrid_prefix_gate_mode == "fixed"
                else 0.0 < gate_init < 1.0
            )
            if not valid_gate_init:
                raise ValueError(
                    "hybrid_prefix_gate_init must be in (0,1] for a fixed gate or (0,1) "
                    "for a learned sigmoid gate, preserving a bounded non-zero prefix "
                    "gradient at initialization"
                )

        # per-layer trainable prefix memory tokens (in hidden space). In dynamic+write mode
        # these are used ONLY as the WRITE queries (64 probes that extract summaries from the
        # document), so their init DISTRIBUTION sets how diverse the probes start -- a collapsed
        # init (all slots alike) makes every probe read the same summary (low effective rank).
        self.prefix = nn.Parameter(torch.zeros(P, hidden_size))
        _dist = getattr(config, "prefix_init_dist", "normal")
        _std = config.prefix_init_std
        if _dist == "normal":
            nn.init.normal_(self.prefix, std=_std)
        elif _dist == "uniform":
            _a = _std * (3 ** 0.5)                      # Uniform(-a,a) has the same variance as N(0,std)
            nn.init.uniform_(self.prefix, -_a, _a)
        elif _dist == "orthogonal":
            # maximally-diverse probes: P orthonormal rows, scaled to match a normal row's norm
            nn.init.orthogonal_(self.prefix)
            with torch.no_grad():
                self.prefix.mul_(_std * (hidden_size ** 0.5))
        else:
            raise ValueError(f"unknown prefix_init_dist: {_dist}")
        # WRITE path: maps what the prefix read from the context back into hidden space,
        # so the prefix becomes input-dependent (a real per-document memory).
        self.write_proj = None
        if config.prefix_write and P > 0:
            self.write_proj = nn.Linear(self.read_dim, hidden_size, bias=False)
            nn.init.normal_(self.write_proj.weight, std=0.02)
        self.hybrid_prefix_gate_logit = None
        if self._hybrid_pool_prefix and config.hybrid_prefix_gate_mode != "fixed":
            gate_shape = (
                ()
                if config.hybrid_prefix_gate_mode == "learned_scalar"
                else (self.read_dim,)
            )
            init = torch.tensor(config.hybrid_prefix_gate_init).logit().item()
            self.hybrid_prefix_gate_logit = nn.Parameter(
                torch.full(gate_shape, init, dtype=torch.float32)
            )
        # pooled-prefix READ has NO parameters of its own (amax over the SWA output of the
        # prefix rows, added to the token reads) -- only validate the config combination.
        if config.pool_reads:
            if config.read_prefix_only:
                raise ValueError("pool_reads uses the [prefix ; SWA window] read; it conflicts "
                                 "with the read_prefix_only bottleneck -- set read_prefix_only=False")
            if P <= 0:
                raise ValueError("pool_reads pools the weighted prefix contributions; it needs "
                                 "num_prefix_tokens > 0")
            if not config.normal_attends_prefix:
                raise ValueError("pool_reads needs normal_attends_prefix=True: with the prefix "
                                 "masked out, alpha_{t,p} ~= 0 and M_t is meaningless")
        # learnable per-channel gate on M_t (trained jointly). zero-init W_g => the gate is
        # g_max*sigmoid(bias) everywhere at start (max=2,bias=0 => 1.0 == plain pool exactly).
        self.mgate = None
        if config.pool_gate:
            if not config.pool_reads:
                raise ValueError("pool_gate gates the pooled M_t; it needs pool_reads=True")
            if config.pool_gate_input not in ("rm", "hidden"):
                raise ValueError(f"pool_gate_input must be 'rm' or 'hidden', got {config.pool_gate_input}")
            gin = 2 * self.read_dim if config.pool_gate_input == "rm" else hidden_size
            self.mgate = nn.Linear(gin, self.read_dim, bias=True)
            nn.init.zeros_(self.mgate.weight)
            nn.init.constant_(self.mgate.bias, config.pool_gate_bias)

        q_out = base.q_proj.out_features
        k_out = base.k_proj.out_features
        v_out = base.v_proj.out_features
        o_out = base.o_proj.out_features
        self.delta_heads = set(config.delta_heads)
        r = config.delta_rank
        # optional trainable projection compressing the memory readout before delta
        self.read_proj = None
        delta_in = self.read_dim
        if config.read_proj_dim and config.read_proj_dim > 0:
            self.read_proj = nn.Linear(self.read_dim, config.read_proj_dim, bias=False)
            nn.init.normal_(self.read_proj.weight, std=0.02)
            delta_in = config.read_proj_dim
        def mk(out, zero_out):
            # full-rank Linear, or a low-rank bottleneck delta_in->r->out to cut params
            if r and r > 0:
                down = nn.Linear(delta_in, r, bias=False)
                up = nn.Linear(r, out, bias=False)
                nn.init.normal_(down.weight, std=0.02)
                (nn.init.zeros_ if zero_out else lambda w: nn.init.normal_(w, std=1e-3))(up.weight)
                return nn.Sequential(down, up)
            lin = nn.Linear(delta_in, out, bias=False)
            (nn.init.zeros_ if zero_out else lambda w: nn.init.normal_(w, std=1e-3))(lin.weight)
            return lin
        if config.steer_mode == "deltamem":
            if self.memory_value_source == "main_v":
                # The shared-V experiment is a genuinely delta-O-only architecture when
                # delta_heads="o": do not instantiate (and therefore do not optimize/save)
                # unused q/k/v correction modules.  Historical configs retain the old
                # all-four-module state schema for strict checkpoint compatibility.
                self.delta_q = mk(q_out, False) if "q" in self.delta_heads else None
                self.delta_k = mk(k_out, False) if "k" in self.delta_heads else None
                self.delta_v = mk(v_out, False) if "v" in self.delta_heads else None
                self.delta_o = mk(o_out, True) if "o" in self.delta_heads else None
            else:
                self.delta_q = mk(q_out, False)
                self.delta_k = mk(k_out, False)
                self.delta_v = mk(v_out, False)
                self.delta_o = mk(o_out, True)  # zero-init output so we start ~= frozen backbone
        else:  # residual
            self.res_proj = nn.Linear(self.read_dim, o_out, bias=False)
            nn.init.normal_(self.res_proj.weight, std=1e-3)
            self.res_gate = nn.Parameter(torch.ones(1))

        # --- differential fusion state ---------------------------------------
        # Only instantiated for the fusion mode that needs it, so an additive run
        # keeps its historical state_dict exactly.
        if config.output_fusion == "learned_diff":
            self.fusion_lambda = nn.Parameter(
                torch.tensor(float(config.fusion_lambda_init))
            )
        if config.output_fusion == "variance_diff":
            self.register_buffer("fusion_mu_y", torch.zeros(o_out), persistent=True)
            self.register_buffer("fusion_mu_c", torch.zeros(o_out), persistent=True)
            # 0 until the first update, so the first forward seeds the means from the
            # batch instead of centring on a spurious zero (cf. BatchNorm running stats).
            self.register_buffer("fusion_ema_seen",
                                 torch.zeros((), dtype=torch.long), persistent=True)
        # Set True to update the EMAs without training anything -- the calibration
        # pass used when the sidecar is frozen and the fusion has no parameters.
        self._fusion_calibrating = False
        self.last_fusion_stats: dict[str, float] = {}
        # Set True to expose (Y, C) from the last fusion WITH their autograd graph.
        # Needed to train C as a nuisance estimator: the target is the group-centred
        # residual of Y, which only exists outside this module.
        self.collect_fusion_tensors = False
        self.last_fusion_tensors: tuple | None = None

        # runtime state set by the top-level wrapper before each forward
        self._seg = None        # [B, L] segment ids for the text tokens
        self._valid = None      # [B, L] validity mask
        self._zero_prefix = False
        self._steer_enabled = True
        # WRITE-ONLY protocol for long histories.  The history forward still runs the
        # writer and keeps its graph-connected result in `_frozen_prefix`, but it skips
        # the memory READ and every delta/residual injection.  Thus the wrapped backbone
        # follows its frozen base path while the written memory can be consumed by a
        # subsequent query forward.  Runtime-only (not in state_dict), default-off for
        # backwards compatibility with old checkpoints and callers.
        self._write_only = False
        self._mem_cache_on = False      # enable memory KV cache for incremental decode
        self._mem_kv = None             # cached (Km, Vm) over [prefix ; context]
        # NO-CONTEXT protocol: write the memory from the context, then FREEZE it so it can be
        # injected into a later forward that does NOT contain the context at all. This is the
        # only setting where the memory can add information a full-attention backbone lacks.
        self._freeze_write = False
        self._frozen_prefix = None
        self._frozen_history_pool = None
        self._debug_write = False   # capture static/written prefix from the real path
        self._dbg_static = None; self._dbg_written = None
        # pool diagnostics (inference-time; pooling has no params so these are safe):
        self._pool_alpha = 1.0      # reads_t = R_t + alpha * M_t   (1.0 = trained behavior)
        self._pool_roll = 0         # >0: roll M_t along the TOKEN dim (shuffle intervention:
                                    # token t receives another token's M) -- diagnostics only
        self._debug_read = False    # capture prefix-probs / R_t / M_t from the real READ path
        self._dbg_RP = None; self._dbg_RH = None; self._dbg_pooled = None; self._dbg_gate = None
        self._dbg_Vp = None; self._dbg_pref_read = None
        self._gate_off = False      # force g==1 at inference (ablate the learned gate)
        # Same-weights hybrid ablation: remove ONLY the additive query-conditioned prefix
        # contribution.  The written/history pooled branch remains active and unchanged.
        self._hybrid_prefix_off = False
        # Training-only branch dropout for the additive hybrid.  Unlike prefix_off, this
        # removes ONLY the pooled branch and leaves gate * prefix_read active.  It is a
        # runtime switch rather than checkpointed architecture state; callers must reset it
        # for evaluation.
        self._hybrid_pool_off = False
        self._window_only = False   # causal: remove the prefix from the READ (mask its columns,
                                    # re-softmax over the window only, M_t=0) -- KEEPS the window
                                    # SWA read. Isolates the prefix's causal value vs window-only,
                                    # unlike _zero_prefix which kills the WHOLE read (=> base).
        self._mem_norm_len = 0          # number of normal tokens in the cache

    # -- runtime setters ---------------------------------------------------
    def set_segments(self, seg, valid):
        self._seg = seg
        self._valid = valid

    def set_zero_prefix(self, flag: bool):
        self._zero_prefix = flag

    def set_steer_enabled(self, flag: bool):
        self._steer_enabled = flag

    def set_write_only(self, flag: bool):
        self._write_only = bool(flag)
        if flag:
            # Arm a fresh history write.  As with set_write_freeze(True), stale memory or
            # decode K/V from a previous persona must never survive into the new write.
            self._frozen_prefix = None
            self._frozen_history_pool = None
            self._mem_kv = None
            self._mem_norm_len = 0

    def set_mem_cache(self, flag: bool):
        self._mem_cache_on = flag
        self._mem_kv = None
        self._mem_norm_len = 0

    # -- memory read -------------------------------------------------------
    def _project_mem(self, seq):
        """Project a [B,T,d] sequence to memory (Qm,Km,Vm) as [B,h,T,hd]."""
        B, T, _ = seq.shape
        if self.share_qkv:
            base, hd = self.base, self.head_dim
            Qm = base.q_norm(base.q_proj(seq).view(B, T, self.n_heads, hd)).transpose(1, 2)
            Km = base.k_norm(base.k_proj(seq).view(B, T, self.n_kv, hd)).transpose(1, 2)
            Vm = base.v_proj(seq).view(B, T, self.n_kv, hd).transpose(1, 2)
            if self.n_kv != self.n_heads:
                rep = self.n_heads // self.n_kv
                Km = Km.repeat_interleave(rep, dim=1); Vm = Vm.repeat_interleave(rep, dim=1)
        elif self.memory_value_source == "main_v":
            h, hd = self.cfg.mem_num_heads, self.head_dim
            Qm = self.mem_q(seq).view(B, T, h, hd).transpose(1, 2)
            Km = self.mem_k(seq).view(B, T, h, hd).transpose(1, 2)
            # Reuse the current layer's frozen Qwen value projection.  Qwen stores one V
            # per KV head; compact side attention groups contiguous GQA KV heads without
            # introducing a learned adapter.  This exact helper is used by prefill, WRITE,
            # and one-token cached decode.
            Vm = self.base.v_proj(seq).view(
                B, T, self.n_kv, hd
            ).transpose(1, 2)
            group = self._main_v_group_size
            if group != 1:
                Vm = Vm.reshape(B, h, group, T, hd).mean(dim=2)
        else:
            h, hd = self.cfg.mem_num_heads, self.cfg.mem_head_dim
            Qm = self.mem_q(seq).view(B, T, h, hd).transpose(1, 2)
            Km = self.mem_k(seq).view(B, T, h, hd).transpose(1, 2)
            Vm = self.mem_v(seq).view(B, T, h, hd).transpose(1, 2)
        return Qm, Km, Vm

    def _variance_diff(self, out, delta_o, eps):
        """out - lambda* (C - mu_C) with the closed-form regression coefficient.

        lambda* = <r_Y, r_C> / (<r_C, r_C> + eps) is the least-squares coefficient of
        the control on the signal, i.e. exactly the scalar that removes the component
        of Y that co-varies with C and nothing else.  It is DETACHED for the same
        reason the rms_match/cosine coefficients are: an attached coefficient could be
        driven by shrinking its own denominator rather than by explaining Y.

        mu_Y / mu_C are running means, updated while training or calibrating and frozen
        at eval.  The DEX nuisance probe could centre within a nuisance group; a
        deployed model sees one sequence at a time and has no group to average over, so
        the running mean is the estimator that actually survives deployment.
        """
        y32 = out.detach().float()
        c32 = delta_o.detach().float()
        flat_y = y32.reshape(-1, y32.shape[-1])
        flat_c = c32.reshape(-1, c32.shape[-1])

        if self.training or self._fusion_calibrating:
            with torch.no_grad():
                m = self.cfg.fusion_ema_momentum
                by, bc = flat_y.mean(0), flat_c.mean(0)
                if int(self.fusion_ema_seen) == 0:
                    self.fusion_mu_y.copy_(by)      # seed, do not decay towards zero
                    self.fusion_mu_c.copy_(bc)
                else:
                    self.fusion_mu_y.mul_(m).add_(by, alpha=1.0 - m)
                    self.fusion_mu_c.mul_(m).add_(bc, alpha=1.0 - m)
                self.fusion_ema_seen += 1

        if int(self.fusion_ema_seen) == 0:
            # never calibrated: fall back to the additive baseline rather than
            # silently subtracting against an all-zero mean
            return out + self.cfg.steer_gain * delta_o

        r_y = flat_y - self.fusion_mu_y.float()
        r_c = flat_c - self.fusion_mu_c.float()
        cov = (r_y * r_c).mean()
        var = r_c.square().mean()
        lam = (cov / (var + eps)).clamp(0.0, self.cfg.fusion_lambda_max)
        self.last_fusion_stats = {
            "lambda": float(lam), "cov": float(cov), "var": float(var),
            "raw_lambda": float(cov / (var + eps)),
        }
        centred = delta_o - self.fusion_mu_c.to(dtype=delta_o.dtype)
        return out - lam.to(dtype=delta_o.dtype) * centred

    def _fuse_delta_o(self, out, delta_o):
        """Fuse one raw delta-O branch into the frozen main-attention output.

        Norms and cosine coefficients are intentionally detached: the adaptive coefficient
        controls branch scale but cannot be gamed by changing only its denominator.  All
        statistics are computed in float32 and cast back to the model dtype.
        """
        if self.collect_fusion_tensors:
            # keep the graph: C is what a nuisance objective trains
            self.last_fusion_tensors = (out, delta_o)
        mode = self.cfg.output_fusion
        gain = self.cfg.steer_gain
        if mode in ("fixed", "fixed_add"):
            # Keep this expression identical to the historical implementation.
            return out + gain * delta_o

        eps = self.cfg.output_fusion_eps
        if mode == "fixed_sub":
            # Same branch, same magnitude, opposite sign.  Only meaningful against a
            # frozen control path (see the config note).
            return out - gain * delta_o
        if mode == "learned_diff":
            self.last_fusion_stats = {"lambda": float(self.fusion_lambda.detach())}
            return out - self.fusion_lambda.to(dtype=delta_o.dtype) * delta_o
        if mode == "variance_diff":
            return self._variance_diff(out, delta_o, eps)
        out_stats = out.detach().float()
        delta_stats = delta_o.detach().float()
        if mode == "rms_match":
            out_rms = out_stats.square().mean(dim=-1, keepdim=True).sqrt()
            delta_rms = delta_stats.square().mean(dim=-1, keepdim=True).sqrt()
            # (+eps)/(+eps) defines the both-zero case as scale=1.  With a normal non-zero
            # backbone output and zero-init delta this clamps to scale_max, preserving a live
            # gradient into delta_o while the actual added branch remains exactly zero.
            scale = ((out_rms + eps) / (delta_rms + eps)).clamp(
                max=self.cfg.output_fusion_scale_max
            )
            relative = (
                abs(gain) * scale * delta_rms / (out_rms + eps)
            )
            scale = scale.to(dtype=delta_o.dtype)
            denom = torch.sqrt(1.0 + relative.square()).to(dtype=out.dtype)
            return (out + gain * scale * delta_o) / denom

        # Per-token cosine gate.  Define cos=0 whenever either vector is zero; therefore
        # coeff=1/2 at zero-init.  The output is still exactly `out` because delta_o=0, but
        # d(output)/d(delta_o)=gain/2 is non-zero.
        out_norm = out_stats.norm(dim=-1, keepdim=True)
        delta_norm = delta_stats.norm(dim=-1, keepdim=True)
        valid = (out_norm > eps) & (delta_norm > eps)
        denom = (out_norm * delta_norm).clamp_min(eps)
        cosine = (out_stats * delta_stats).sum(dim=-1, keepdim=True) / denom
        cosine = torch.where(valid, cosine.clamp(-1.0, 1.0), torch.zeros_like(cosine))
        coeff = (0.5 * (1.0 + cosine)).to(dtype=delta_o.dtype)
        return out + gain * coeff * delta_o

    def _prefix_embeds(self, B, dtype, device):
        p = self.prefix.to(dtype=dtype, device=device)
        if self._zero_prefix:
            p = torch.zeros_like(p)
        return p.unsqueeze(0).expand(B, -1, -1)

    def _prefix_write_keep(self, ctx_valid):
        """Return ``[B,P,L]`` valid-key routing for strict prefix WRITE.

        Partition boundaries are computed in *valid-token order*, not padded tensor
        coordinates.  ``floor(p*N/P):ceil((p+1)*N/P)`` guarantees a non-empty range
        for every slot even when ``N < P``; optional overlap expands in that same
        valid-token order and can never admit padding.
        """
        if ctx_valid.ndim != 2:
            raise ValueError("ctx_valid must have shape [B,L]")
        ctx_valid = ctx_valid.bool()
        B, L = ctx_valid.shape
        P = self.cfg.num_prefix_tokens
        if (~ctx_valid.any(dim=1)).any():
            raise RuntimeError(
                "prefix WRITE needs at least one valid history token in every batch row"
            )
        if self.cfg.prefix_write_layout == "global":
            return ctx_valid[:, None, :].expand(B, P, L)

        routed = torch.zeros(B, P, L, dtype=torch.bool, device=ctx_valid.device)
        overlap = self.cfg.prefix_write_overlap_tokens
        for batch_index in range(B):
            positions = torch.nonzero(
                ctx_valid[batch_index], as_tuple=False
            ).flatten()
            count = int(positions.numel())
            for slot in range(P):
                start = (slot * count) // P
                # Integer ceil((slot+1)*count/P), and at least start+1.
                end = ((slot + 1) * count + P - 1) // P
                start = max(0, start - overlap)
                end = min(count, max(start + 1, end + overlap))
                routed[batch_index, slot, positions[start:end]] = True
        if (~routed.any(dim=-1)).any():
            raise AssertionError("partitioned prefix WRITE produced an empty slot")
        return routed

    def _apply_gate(self, nf, mf, hidden_states):
        """g_t = g_max * sigmoid(W_g [gate_input]); return g_t (.) M_t. gate_input is
        [R_t ; M_t] ('rm', default -- gate sees token AND retrieved memory) or h_t ('hidden').
        nf=R_t, mf=M_t, both [B, L, read_dim]."""
        if self.mgate is None or self._gate_off:
            return mf     # _gate_off: force g==1 (reads = R + M), i.e. plain pool with the
                          # GATED-training weights -- isolates "gate learned badly" from
                          # "gate-training moved everything to a worse basin".
        gin = torch.cat([nf, mf], dim=-1) if self.cfg.pool_gate_input == "rm" else hidden_states
        g = self.cfg.pool_gate_max * torch.sigmoid(self.mgate(gin))
        if self._debug_read:
            self._dbg_gate = g.detach()
        return g * mf

    def _hybrid_gate(self, *, dtype, device):
        """Return the additive-prefix gate, broadcastable over ``[B,L,read_dim]``."""
        if self.hybrid_prefix_gate_logit is None:
            return torch.as_tensor(
                self.cfg.hybrid_prefix_gate_init, dtype=dtype, device=device
            )
        gate = torch.sigmoid(
            self.hybrid_prefix_gate_logit.to(dtype=dtype, device=device)
        )
        if gate.ndim == 1:
            gate = gate.view(1, 1, self.read_dim)
        return gate

    @staticmethod
    def _broadcast_frozen(memory, B, *, name, dtype, device):
        """Move a frozen WRITE result to the query and broadcast a single writer batch."""
        if memory is None:
            raise RuntimeError(
                f"hybrid pooled_plus_prefix query read has no {name}; run one WRITE-only "
                "history forward before the query"
            )
        memory = memory.to(dtype=dtype, device=device)
        if memory.shape[0] != B:
            if memory.shape[0] != 1:
                raise RuntimeError(
                    f"cannot broadcast hybrid {name} batch {memory.shape[0]} to query batch {B}"
                )
            memory = memory.expand(B, -1, -1)
        return memory

    def _write_hybrid_pool_prefix(self, hidden_states):
        """WRITE one history into both branches using shared history K/V projections.

        The learned pool query and P learned prefix probes are concatenated into one query
        projection.  All P+1 probes attend the exact same validity-masked history K/V:
        row zero becomes the strong pooled-steer vector, and the remaining rows are mapped
        through ``write_proj`` into P persistent prefix slots.
        """
        B, L, _ = hidden_states.shape
        dtype, device = hidden_states.dtype, hidden_states.device
        P = self.cfg.num_prefix_tokens

        if self._seg is not None:
            seg = self._seg.to(device)
            valid = (
                self._valid.to(device).bool()
                if self._valid is not None
                else torch.ones(B, L, dtype=torch.bool, device=device)
            )
            keep = (seg == SEG_CTX) & valid
            no_ctx = ~keep.any(dim=1)
            keep = torch.where(no_ctx[:, None], valid, keep)
        else:
            keep = torch.ones(B, L, dtype=torch.bool, device=device)

        static_prefix = self._prefix_embeds(B, dtype, device)
        pool_query = self.history_pool_query.to(dtype=dtype, device=device)
        pool_query = pool_query.unsqueeze(0).expand(B, -1, -1)
        writer_queries = torch.cat([pool_query, static_prefix], dim=1)
        queries, _, _ = self._project_mem(writer_queries)
        _, keys, values = self._project_mem(hidden_states)
        writer_keep = torch.cat(
            [keep[:, None, :], self._prefix_write_keep(keep)], dim=1
        )
        mask = torch.zeros(B, 1, P + 1, L, dtype=queries.dtype, device=device)
        mask = mask.masked_fill(
            ~writer_keep[:, None, :, :], torch.finfo(queries.dtype).min
        )
        writer_read = F.scaled_dot_product_attention(
            queries, keys, values, attn_mask=mask
        )
        writer_read = writer_read.transpose(1, 2).reshape(B, P + 1, self.read_dim)

        pooled = writer_read[:, :1, :].to(dtype)
        written = self.write_proj(writer_read[:, 1:, :]).to(dtype)
        prefix = written if self.cfg.memory_mode == "dynamic" else static_prefix + written

        if self._debug_write:
            self._dbg_static = static_prefix.detach()
            self._dbg_written = prefix.detach()
        # Keep both writer graphs for two-forward training; eval does not need them.
        self._frozen_history_pool = pooled if self.training else pooled.detach()
        self._frozen_prefix = prefix if self.training else prefix.detach()

    def _hybrid_pool_prefix_memory(self, hidden_states):
        """Additive pooled-steer + query-conditioned prefix-only READ."""
        B, L, _ = hidden_states.shape
        dtype, device = hidden_states.dtype, hidden_states.device
        P = self.cfg.num_prefix_tokens

        if self._write_only:
            self._write_hybrid_pool_prefix(hidden_states)
            return None

        # Preserve the legacy set_write_freeze protocol for callers that write and read a
        # context in one forward.  The paper-facing long-history path uses WRITE-only.
        if (
            self._freeze_write
            and (
                self._frozen_history_pool is None
                or self._frozen_prefix is None
            )
        ):
            self._write_hybrid_pool_prefix(hidden_states)

        if self._zero_prefix:
            return torch.zeros(B, L, self.read_dim, dtype=dtype, device=device)

        pooled = self._broadcast_frozen(
            self._frozen_history_pool,
            B,
            name="history pool",
            dtype=dtype,
            device=device,
        ).expand(B, L, self.read_dim)

        # This is the decisive same-weights ablation: no re-normalization and no alternate
        # reader.  Return the pooled branch byte-for-byte as computed above.
        if self._hybrid_prefix_off or self._window_only:
            return pooled

        prefix = self._broadcast_frozen(
            self._frozen_prefix,
            B,
            name="written prefix",
            dtype=dtype,
            device=device,
        )

        if self._mem_cache_on and self._mem_kv is not None and L == 1:
            pK, pV, _, _ = self._mem_kv
            Qn, _, _ = self._project_mem(hidden_states)
        else:
            seq = torch.cat([prefix, hidden_states], dim=1)
            Qm, Km, Vm = self._project_mem(seq)
            Qn = Qm[:, :, P:, :]
            pK, pV = Km[:, :, :P, :], Vm[:, :, :P, :]
            if self._mem_cache_on:
                empty_k = pK[:, :, :0, :]
                empty_v = pV[:, :, :0, :]
                self._mem_kv = (pK, pV, empty_k, empty_v)

        prefix_read = F.scaled_dot_product_attention(Qn, pK, pV)
        prefix_read = prefix_read.transpose(1, 2).reshape(B, L, self.read_dim)
        gate = self._hybrid_gate(dtype=dtype, device=device)
        prefix_contribution = gate * prefix_read
        if self._debug_read:
            self._dbg_RH = pooled
            self._dbg_pref_read = prefix_read
            self._dbg_pooled = prefix_contribution
            self._dbg_gate = (
                gate.detach()
                if isinstance(gate, torch.Tensor)
                else torch.as_tensor(gate, device=device)
            )
        if self._hybrid_pool_off:
            return prefix_contribution
        return pooled + prefix_contribution

    def _history_pool_memory(self, hidden_states):
        """WRITE/read the single-vector history-conditioned pooled-steer baseline.

        WRITE-only history produces one graph-connected ``[B,1,read_dim]`` vector per
        patched layer.  Query forwards broadcast that same vector over tokens/batches and
        feed it directly to the existing steer heads.  The reader never sees prefix slots,
        normal-token SWA, or the query when choosing the memory content.
        """
        B, L, _ = hidden_states.shape
        dtype, device = hidden_states.dtype, hidden_states.device

        if self._write_only:
            if self._seg is not None:
                seg = self._seg.to(device)
                valid = self._valid.to(device).bool()
                keep = (seg == SEG_CTX) & valid
                # Writer-only callers normally mark every token SEG_CTX.  Avoid an all-masked
                # softmax for generic callers while still excluding non-context fields when
                # context markers are present.
                no_ctx = ~keep.any(dim=1)
                keep = torch.where(no_ctx[:, None], valid, keep)
            else:
                keep = torch.ones(B, L, dtype=torch.bool, device=device)

            if self.cfg.history_pool_mode == "mean":
                _, _, values = self._project_mem(hidden_states)  # [B,h,L,hd]
                weights = keep[:, None, :, None].to(values.dtype)
                pooled = (values * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1)
            else:
                query_h = self.history_pool_query.to(dtype=dtype, device=device)
                query_h = query_h.unsqueeze(0).expand(B, -1, -1)
                queries, _, _ = self._project_mem(query_h)
                _, keys, values = self._project_mem(hidden_states)
                mask = torch.zeros(B, 1, 1, L, dtype=queries.dtype, device=device)
                mask = mask.masked_fill(
                    ~keep[:, None, None, :], torch.finfo(queries.dtype).min
                )
                pooled = F.scaled_dot_product_attention(
                    queries, keys, values, attn_mask=mask
                ).squeeze(2)

            frozen = pooled.reshape(B, 1, self.read_dim).to(dtype)
            # Preserve the writer graph during training; evaluation does not need it.
            self._frozen_history_pool = frozen if self.training else frozen.detach()
            return None

        if self._zero_prefix or self._window_only:
            return torch.zeros(B, L, self.read_dim, dtype=dtype, device=device)
        if self._frozen_history_pool is None:
            raise RuntimeError(
                "history_pool_mode query read has no memory; run one WRITE-only history "
                "forward before the query"
            )
        pooled = self._frozen_history_pool.to(dtype=dtype, device=device)
        if pooled.shape[0] != B:
            if pooled.shape[0] != 1:
                raise RuntimeError(
                    f"cannot broadcast history pool batch {pooled.shape[0]} to query batch {B}"
                )
            pooled = pooled.expand(B, -1, -1)
        return pooled.expand(B, L, self.read_dim)

    def _token_pool(self, pp, Vp):
        """M_t = amax_p( alpha_{t,p} * v_p ) without materializing [B,h,L,P,hd].

        pp : [B,h,L,P] prefix attention probs (full-softmax alphas, prefix columns)
        Vp : [B,h,P,hd] prefix values
        ->   [B,h,L,hd]

        Two passes: (1) chunked no-grad argmax over slots (transient [B,h,L,chunk,hd]);
        (2) differentiable gather of the winners -- exactly the max subgradient, so
        training gradients route to alpha (=> q/k) and v of the winning slot only.
        """
        B, h, L, P = pp.shape
        hd = Vp.shape[-1]
        with torch.no_grad():
            best, idx = None, None
            for s in range(0, P, 16):
                c = pp[..., s:s+16].unsqueeze(-1) * Vp[:, :, s:s+16, :].unsqueeze(2)
                cb, ci = c.max(dim=3)                       # [B,h,L,hd]
                ci = ci + s
                if best is None:
                    best, idx = cb, ci
                else:
                    take = cb > best
                    best = torch.where(take, cb, best)
                    idx = torch.where(take, ci, idx)
        a = pp.gather(3, idx)                               # alpha of the winning slot per dim
        v = Vp.unsqueeze(2).expand(B, h, L, P, hd).gather(3, idx.unsqueeze(3)).squeeze(3)
        return a * v

    def _memory_read(self, hidden_states):
        B, L, d = hidden_states.shape
        device, dtype = hidden_states.device, hidden_states.dtype
        P = self.cfg.num_prefix_tokens
        W = self.cfg.sliding_window_size

        if self._hybrid_pool_prefix:
            return self._hybrid_pool_prefix_memory(hidden_states)

        if self.cfg.history_pool_mode != "none":
            return self._history_pool_memory(hidden_states)

        # TRUE no-memory ablation: kill the memory SIGNAL, keeping the steer architecture
        # and its params identical. Zeroing only the *initial* prefix is NOT an ablation once
        # prefix_write=True -- the write path (prefix += write_proj(attend(context))) still
        # injects document-dependent content, so method_zp would still have a real memory.
        if self._zero_prefix:
            return torch.zeros(B, L, self.read_dim, dtype=dtype, device=device)

        # ---- TRUE-SWA incremental decode (L==1, cache primed) ----
        # cache = (pK, pV, nK, nV): prefix KV (fixed) + a ROLLING window of the last
        # W normal-token KV. Because the answer token attends only to [prefix ; last W
        # normal], we drop older keys -> attention is over P+W keys (O(W), constant),
        # not O(N). For normal_attends_prefix=True no mask is needed (all keys allowed).
        if self._mem_cache_on and self._mem_kv is not None and L == 1:
            pK, pV, nK, nV = self._mem_kv
            Qn, Kn, Vn = self._project_mem(hidden_states)          # new token only
            if self.cfg.read_prefix_only:
                # decode must honour the bottleneck too -- the ANSWER tokens (the ones the
                # F1 is actually computed on) are generated here, so letting them read the
                # normal-hidden window would reopen the bypass for exactly those tokens.
                ctx = F.scaled_dot_product_attention(Qn, pK, pV)
                return ctx.transpose(1, 2).reshape(B, 1, self.read_dim)
            nK = torch.cat([nK, Kn], dim=2)[:, :, -W:, :]         # keep last W normal (O(W) copy)
            nV = torch.cat([nV, Vn], dim=2)[:, :, -W:, :]
            self._mem_kv = (pK, pV, nK, nV)
            Km = torch.cat([pK, nK], dim=2)                       # P + (<=W) keys
            Vm = torch.cat([pV, nV], dim=2)
            attn_mask = None
            if not self.cfg.normal_attends_prefix:                # prefix disallowed for normal rows
                m = torch.zeros(Km.shape[2], dtype=Qn.dtype, device=device)
                m[:P] = torch.finfo(Qn.dtype).min
                attn_mask = m.view(1, 1, 1, -1)
            if self.cfg.pool_reads and P > 0:
                # token-specific pool: M_t depends on the NEW token's query through
                # alpha[t, p], so it must be recomputed for every decode step (only the
                # prefix/window K/V are cached). Manual attention exposes the probs.
                scores = (Qn @ Km.transpose(-2, -1)) * (Qn.shape[-1] ** -0.5)
                if attn_mask is not None:
                    scores = scores + attn_mask
                if self._window_only:
                    scores[..., :P] = torch.finfo(scores.dtype).min
                probs = torch.softmax(scores, dim=-1)             # [B,h,1,P+<=W]
                normal = probs @ Vm                               # standard SWA read
                cb = probs[..., :P].unsqueeze(-1) * pV.unsqueeze(2)   # [B,h,1,P,hd] tiny
                tp = cb.amax(dim=3)                               # M_t [B,h,1,hd]
                nf = normal.transpose(1, 2).reshape(B, 1, self.read_dim)
                mf = tp.transpose(1, 2).reshape(B, 1, self.read_dim)
                mf = self._apply_gate(nf, mf, hidden_states)
                return nf + self._pool_alpha * mf
            ctx = F.scaled_dot_product_attention(Qn, Km, Vm, attn_mask=attn_mask)
            return ctx.transpose(1, 2).reshape(B, 1, self.read_dim)

        # ---- prefill (or non-cached full pass) over [prefix ; text] ----
        if self._seg is not None:
            seg_text, valid_text = self._seg.to(device), self._valid.to(device)
        else:
            seg_text = torch.full((B, L), SEG_CTX, dtype=torch.long, device=device)
            valid_text = torch.ones((B, L), dtype=torch.bool, device=device)
        pref_seg = torch.full((B, P), SEG_PREFIX, dtype=torch.long, device=device)
        pref_valid = torch.ones((B, P), dtype=torch.bool, device=device)
        seg_full = torch.cat([pref_seg, seg_text], dim=1)
        valid_full = torch.cat([pref_valid, valid_text.bool()], dim=1)

        def attend(prefix_h, wo=False):
            seq = torch.cat([prefix_h, hidden_states], dim=1)
            Qm, Km, Vm = self._project_mem(seq)
            mask = build_prefix_attention_mask(
                seg_full, valid_full, sliding_window_size=W,
                normal_attends_prefix=self.cfg.normal_attends_prefix,
                prefix_sees_query=self.cfg.prefix_sees_query, dtype=Qm.dtype)
            if wo:
                # window-only causal ablation: normal-token rows may NOT attend the prefix
                # columns (SWA window kept). Applies to the ordinary (nopool) SDPA read too,
                # not just the manual pool path. Only mask the NORMAL rows (P:), never the
                # prefix WRITE rows (:P), so the write is untouched.
                mask = mask.clone()
                mask[:, :, P:, :P] = torch.finfo(mask.dtype).min
            ctx = F.scaled_dot_product_attention(Qm, Km, Vm, attn_mask=mask.to(Qm.dtype))
            return ctx.transpose(1, 2).reshape(B, P + L, self.read_dim), Km, Vm

        prefix = self._prefix_embeds(B, dtype, device)
        if self.write_proj is not None and P > 0:
            if self._frozen_prefix is not None:
                # NO-CONTEXT read: reuse the memory written from the (now removed) context
                prefix = self._frozen_prefix.to(dtype=dtype, device=device)
                if prefix.shape[0] != B:
                    prefix = prefix.expand(B, -1, -1)
            else:
                # ---- WRITE: the prefix reads the context; residually update it so the memory
                # becomes input-dependent. (These are exactly the ctx[:, :P, :] rows that the
                # old single-stage code computed and threw away.)
                if self.cfg.write_ctx_only:
                    # STRICT document write: prefix queries attend the SEG_CTX hidden states
                    # ONLY -- not each other, not the query/answer. The written memory is then
                    # a pure function of the document, removing the last confounder (prefix
                    # self-attention, which let a slot copy the learned prefix instead of the doc).
                    # NOTE the mask: hidden_states is the WHOLE text sequence (ctx+query+answer
                    # in ctx-mode training), so "context only" must be enforced with _seg --
                    # an unmasked SDPA here would silently leak the query/answer into the write.
                    Qm, _, _ = self._project_mem(prefix)            # queries from the P prefix slots
                    _, Kc, Vc = self._project_mem(hidden_states)    # keys/values from the text seq
                    ctx_valid = (seg_text == SEG_CTX) & valid_text.bool()   # [B, L]
                    if (~ctx_valid.any(dim=1)).any():
                        # a row with NO ctx tokens would be all -inf -> NaN; fall back to valid
                        no_ctx = ~ctx_valid.any(dim=1)
                        ctx_valid = torch.where(no_ctx[:, None], valid_text.bool(), ctx_valid)
                    write_keep = self._prefix_write_keep(ctx_valid)
                    wm = torch.zeros(B, 1, P, L, dtype=Qm.dtype, device=device)
                    wm = wm.masked_fill(
                        ~write_keep[:, None, :, :], torch.finfo(Qm.dtype).min
                    )
                    cw = F.scaled_dot_product_attention(Qm, Kc, Vc, attn_mask=wm)  # [B, h, P, hd]
                    written = self.write_proj(cw.transpose(1, 2).reshape(B, P, self.read_dim)).to(dtype)
                else:
                    ctx_w, _, _ = attend(prefix)
                    written = self.write_proj(ctx_w[:, :P, :]).to(dtype)
                if self.cfg.memory_mode == "dynamic":
                    # memory CONTENT is only the written (document-dependent) part; the learned
                    # prefix is used ONLY as the write query, never added to the read K/V. This
                    # removes the static shortcut that let the optimizer ignore the document.
                    prefix = written
                else:
                    prefix = prefix + written
                if self._debug_write:   # capture from the REAL path (real mask, real segments)
                    self._dbg_static = self._prefix_embeds(B, dtype, device).detach()
                    self._dbg_written = prefix.detach()
                if self._freeze_write or self._write_only:
                    # KEEP the graph while training: two-forward (write -> drop context ->
                    # read) training backprops the READ loss through this tensor into
                    # prefix / write_proj / the WRITE-stage mem_q,k,v. detach() here would
                    # silently zero the writer's gradient and leave it at its random init.
                    self._frozen_prefix = prefix if self.training else prefix.detach()
        if self._write_only:
            # History write ends here: do not build the token READ (which is quadratic in
            # L for the pooled path), and do not prime a decode cache from history tokens.
            # forward() deliberately ignores this sentinel and runs the exact base q/k/v/o.
            return None
        # ---- READ
        if self.cfg.read_prefix_only and P > 0:
            # normal tokens attend to the PREFIX ONLY -- no normal-hidden bypass, so the
            # memory is the sole route for context information to reach the steer.
            if self._window_only:
                # read_prefix_only has NO window; removing the prefix leaves no read at all,
                # so window-only == zero read (== base). (This mode has no "window SWA" term.)
                return torch.zeros(B, L, self.read_dim, dtype=dtype, device=device)
            seq = torch.cat([prefix, hidden_states], dim=1)
            Qm, Km, Vm = self._project_mem(seq)
            Qn = Qm[:, :, P:, :]
            Kp, Vp = Km[:, :, :P, :], Vm[:, :, :P, :]
            ctxp = F.scaled_dot_product_attention(Qn, Kp, Vp)   # every prefix slot visible
            reads = ctxp.transpose(1, 2).reshape(B, L, self.read_dim)
            if self._mem_cache_on:
                self._mem_kv = (Kp, Vp, Kp[:, :, :0, :], Vp[:, :, :0, :])
            return reads
        # ---- READ (token-specific pool): ONE softmax over [prefix ; window] per token,
        # standard SWA read R_t plus the per-token max over WEIGHTED prefix contributions:
        #     reads_t = R_t + alpha * amax_p( alpha_{t,p} * v_p )
        # Manual attention (SDPA hides the probs); same scaling as SDPA (1/sqrt(hd)).
        if self.cfg.pool_reads and P > 0:
            seq = torch.cat([prefix, hidden_states], dim=1)
            Qm, Km, Vm = self._project_mem(seq)
            Qn = Qm[:, :, P:, :]                                  # normal-token queries
            mask = build_prefix_attention_mask(
                seg_full, valid_full, sliding_window_size=W,
                normal_attends_prefix=self.cfg.normal_attends_prefix,
                prefix_sees_query=self.cfg.prefix_sees_query, dtype=Qn.dtype)
            mrow = mask[:, :, P:, :]                              # normal query rows only
            Vp = Vm[:, :, :P, :]
            # dense [B,h,L,P+L] scores blow up at long context (LoCoMo ~50k). Chunk the QUERY
            # dim: softmax is per-row so this is EXACT; peak mem ~ [h, CHUNK, P+L], freed per
            # block. Debug/roll capture only in the small-L path (diagnostics use short seqs).
            CHUNK = 4096
            scale = Qn.shape[-1] ** -0.5
            if L > CHUNK and not self._debug_read and not self._pool_roll:
                nfs, mfs = [], []
                for cs in range(0, L, CHUNK):
                    ce = min(cs + CHUNK, L)
                    sc = (Qn[:, :, cs:ce, :] @ Km.transpose(-2, -1)) * scale
                    sc = sc + mrow[:, :, cs:ce, :].to(sc.dtype)
                    if self._window_only:
                        sc[..., :P] = torch.finfo(sc.dtype).min
                    pc = torch.softmax(sc, dim=-1)
                    nfs.append((pc @ Vm))
                    mfs.append(self._token_pool(pc[..., :P], Vp))
                    del sc, pc
                normal = torch.cat(nfs, dim=2); tp = torch.cat(mfs, dim=2)
            else:
                scores = (Qn @ Km.transpose(-2, -1)) * scale
                scores = scores + mrow.to(scores.dtype)
                if self._window_only:   # CAUSAL: drop prefix from read -> softmax over window
                    scores[..., :P] = torch.finfo(scores.dtype).min
                probs = torch.softmax(scores, dim=-1)             # [B,h,L,P+L]
                normal = probs @ Vm                               # R_t (standard SWA read)
                tp = self._token_pool(probs[..., :P], Vp)         # M_t (=0 if window_only)
                if self._pool_roll:     # shuffle intervention: token t gets token (t-roll)'s M
                    tp = tp.roll(self._pool_roll, dims=2)
            nf = normal.transpose(1, 2).reshape(B, L, self.read_dim)      # R_t
            mf = tp.transpose(1, 2).reshape(B, L, self.read_dim)          # M_t
            mf = self._apply_gate(nf, mf, hidden_states)                  # g_t([R_t;M_t]) * M_t
            if self._debug_read:
                self._dbg_RP = probs[..., :P]                     # prefix alphas [B,h,L,P]
                self._dbg_RH = nf                                 # R_t (full SWA read)
                self._dbg_pooled = mf                             # M_t (post-gate)
                self._dbg_Vp = Vm[:, :, :P, :]                    # prefix values [B,h,P,hd]
                # attention-WEIGHTED prefix read: sum_p alpha_{t,p} v_p (prefix's share of R_t)
                pr = (probs[..., :P] @ Vm[:, :, :P, :])           # [B,h,L,hd]
                self._dbg_pref_read = pr.transpose(1, 2).reshape(B, L, self.read_dim)
            reads = nf + self._pool_alpha * mf
            if self._mem_cache_on:  # prime decode: prefix KV + last W normal KV
                self._mem_kv = (Km[:, :, :P, :], Vm[:, :, :P, :],
                                Km[:, :, P:, :][:, :, -W:, :], Vm[:, :, P:, :][:, :, -W:, :])
            return reads

        # normal tokens attend to the (written) prefix + their SWA window
        ctx, Km, Vm = attend(prefix, wo=self._window_only)   # window_only masks prefix cols
        reads = ctx[:, P:, :]
        if self._mem_cache_on:  # prime the cache: (written) prefix KV + last W normal KV
            pK, pV = Km[:, :, :P, :], Vm[:, :, :P, :]
            nK, nV = Km[:, :, P:, :][:, :, -W:, :], Vm[:, :, P:, :][:, :, -W:, :]
            self._mem_kv = (pK, pV, nK, nV)
        return reads

    # -- forward (mirrors Qwen3Attention.forward + steer) ------------------
    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, **kwargs):
        base = self.base
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, base.head_dim)

        write_only = self._write_only
        steer = self._steer_enabled and not write_only
        # WRITE-only is independent of the read/steer toggle: even if a caller previously
        # disabled steering, explicitly entering write-only must still populate the memory.
        reads = self._memory_read(hidden_states) if (steer or write_only) else None
        if steer and self.read_proj is not None:
            reads = self.read_proj(reads)

        g = self.cfg.steer_gain
        q = base.q_proj(hidden_states)
        k = base.k_proj(hidden_states)
        v = base.v_proj(hidden_states)
        delta_o = None
        if steer and self.cfg.steer_mode == "deltamem":
            if "q" in self.delta_heads:
                q = q + g * self.delta_q(reads)
            if "k" in self.delta_heads:
                k = k + g * self.delta_k(reads)
            if "v" in self.delta_heads:
                v = v + g * self.delta_v(reads)
            if "o" in self.delta_heads:
                delta_o = self.delta_o(reads)

        query = base.q_norm(q.view(hidden_shape)).transpose(1, 2)
        key = base.k_norm(k.view(hidden_shape)).transpose(1, 2)
        value = v.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query, key = _q.apply_rotary_pos_emb(query, key, cos, sin)

        if past_key_values is not None:
            key, value = past_key_values.update(key, value, base.layer_idx)

        attention_interface = _q.ALL_ATTENTION_FUNCTIONS.get_interface(
            base.config._attn_implementation, _q.eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            base, query, key, value, attention_mask,
            dropout=0.0 if not self.training else base.attention_dropout,
            scaling=base.scaling, sliding_window=base.sliding_window, **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        out = base.o_proj(attn_output)

        if steer and self.cfg.steer_mode == "deltamem":
            if delta_o is not None:
                out = self._fuse_delta_o(out, delta_o)
        elif steer:
            out = out + g * self.res_gate * self.res_proj(reads)
        return out, attn_weights


def _parent(root, name):
    parts = name.split(".")
    p = root
    for part in parts[:-1]:
        p = getattr(p, part)
    return p, parts[-1]


def attach_prefix_steer(model, config: PrefixSteerConfig) -> list[str]:
    """Replace target self-attention modules with PrefixMemSteerAttention wrappers."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
    from dataclasses import replace as _replace
    text_cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    hidden = text_cfg.hidden_size
    pl = set(config.prefix_layers) if config.prefix_layers else None
    replaced = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, Qwen3Attention):
            continue
        if config.steer_layers and module.layer_idx not in config.steer_layers:
            continue
        # layers NOT in prefix_layers get P=0 (pure SWA steer: no prefix/write/pool)
        cfg_i = config
        if pl is not None and module.layer_idx not in pl:
            cfg_i = _replace(config, num_prefix_tokens=0, pool_reads=False,
                             prefix_write=False, pool_gate=False,
                             hybrid_read_mode="none")
        parent, attr = _parent(model, name)
        wrapped = PrefixMemSteerAttention(module, cfg_i, hidden).to(
            device=module.q_proj.weight.device, dtype=module.q_proj.weight.dtype
        )
        setattr(parent, attr, wrapped)
        replaced.append(name)
    if not replaced:
        raise RuntimeError("No attention modules matched for prefix-steer")
    return replaced


def iter_steer_modules(model):
    for m in model.modules():
        if isinstance(m, PrefixMemSteerAttention):
            yield m


def set_steer_segments(model, seg, valid):
    for m in iter_steer_modules(model):
        m.set_segments(seg, valid)


def set_steer_zero_prefix(model, flag):
    for m in iter_steer_modules(model):
        m.set_zero_prefix(flag)


def set_steer_enabled(model, flag):
    for m in iter_steer_modules(model):
        m.set_steer_enabled(flag)


def set_write_only(model, flag):
    """Toggle the long-history WRITE-only protocol on every attached layer.

    ``True`` clears stale frozen/cache state, writes a fresh graph-connected prefix on the
    next history forward, and keeps that forward on the pure frozen-backbone path (no memory
    READ or delta injection).  ``False`` stops WRITE-only mode but preserves the freshly
    frozen prefix for subsequent query forwards and their backward pass.

    The history forward must not be activation-checkpointed: checkpoint recomputation occurs
    after this mutable runtime flag has switched back to query/read mode, so it cannot replay
    the original stateful forward.  The frozen backbone itself builds no history autograd
    graph in WRITE-only mode; only each writer's graph is retained.
    """
    for m in iter_steer_modules(model):
        m.set_write_only(flag)


def set_gate_off(model, flag):
    """Force the pool gate to g==1 at inference (ablate the LEARNED gate, keep gated weights)."""
    for m in iter_steer_modules(model):
        m._gate_off = flag


def set_hybrid_prefix_off(model, flag):
    """Ablate only the additive prefix contribution in pooled_plus_prefix hybrids.

    The same module, written memories, pooled vector, steer heads, and learned weights stay
    active.  Therefore ``True`` is the exact pooled-steer branch of the trained hybrid.
    """
    for m in iter_steer_modules(model):
        m._hybrid_prefix_off = bool(flag)


def set_hybrid_pool_off(model, flag):
    """Ablate only the pooled contribution in pooled_plus_prefix hybrids.

    ``True`` makes the hybrid READ return ``gate * prefix_read``.  This is intended
    for branch-dropout training that forces the written prefix path to learn useful
    signal.  It does not change ``set_hybrid_prefix_off``: prefix-off remains the
    exact pooled-only same-weights ablation even if both runtime flags are set.
    """
    for m in iter_steer_modules(model):
        m._hybrid_pool_off = bool(flag)


def set_window_only(model, flag):
    """CAUSAL: remove the prefix from the READ (window-only SWA, M_t=0), keeping everything
    else.  For pooled_plus_prefix (which intentionally has no local window read), this is
    an alias for its pooled-only branch."""
    for m in iter_steer_modules(model):
        m._window_only = flag


def set_mem_cache(model, flag):
    """Enable/reset memory KV cache for incremental (KV-cached) decoding."""
    for m in iter_steer_modules(model):
        m.set_mem_cache(flag)


def set_write_freeze(model, flag):
    """Arm the NO-CONTEXT protocol: the next forward WRITES the memory from the context and
    freezes it; subsequent forwards reuse that memory even if the context is gone.

    set_write_freeze(False) only STOPS writing -- it KEEPS the frozen memory. (The old
    version cleared it on both True and False, which forced every caller to hand-roll
    `m._freeze_write = False` loops to avoid wiping the memory it had just written.)"""
    for m in iter_steer_modules(model):
        m._freeze_write = flag
        if flag:
            # arm a NEW write: also drop any decode KV cache derived from the PREVIOUS
            # document, so stale prefix/window K/V can never leak across documents.
            m._frozen_prefix = None
            m._frozen_history_pool = None
            m._mem_kv = None


def clear_frozen_memory(model):
    # clear BOTH the frozen written prefix AND the decode KV cache, so switching document /
    # eval condition can never reuse a previous doc's memory or stale prefix/window K/V.
    for m in iter_steer_modules(model):
        m._frozen_prefix = None
        m._frozen_history_pool = None
        m._mem_kv = None
        m._mem_norm_len = 0


def has_frozen_memory(model):
    present = []
    for m in iter_steer_modules(model):
        if m._hybrid_pool_prefix:
            value = (
                m._frozen_history_pool is not None
                and m._frozen_prefix is not None
            )
        elif m.cfg.history_pool_mode != "none":
            value = m._frozen_history_pool is not None
        else:
            value = m._frozen_prefix is not None
        present.append(value)
    return present


_STEER_MARKERS = (
    ".prefix", ".mem_q.", ".mem_k.", ".mem_v.", ".write_proj.",
    ".history_pool_query",
    ".hybrid_prefix_gate_logit",
    ".delta_q.", ".delta_k.", ".delta_v.", ".delta_o.",
    ".res_proj.", ".res_gate", ".share_gate", ".read_proj.", ".mgate.",
    ".fusion_lambda",
)

# The differential fusion's own parameters, separable from the memory path so a
# stage-1 experiment can freeze a trained sidecar and train ONLY the fusion.
_FUSION_MARKERS = (".fusion_lambda",)


def is_fusion_param_name(name: str) -> bool:
    return any(mk in name for mk in _FUSION_MARKERS)


def set_fusion_calibrating(model, flag: bool):
    """Update the variance_diff running means without training anything."""
    n = 0
    for m in iter_steer_modules(model):
        m._fusion_calibrating = bool(flag)
        n += 1
    return n


def set_collect_fusion_tensors(model, flag: bool):
    """Expose (Y, C) per layer from the next forward, with autograd intact."""
    n = 0
    for m in iter_steer_modules(model):
        m.collect_fusion_tensors = bool(flag)
        if not flag:
            m.last_fusion_tensors = None
        n += 1
    return n


def collect_fusion_tensors(model) -> list:
    """[(layer_index, Y, C), ...] from the last forward, in layer order."""
    out = []
    for i, m in enumerate(iter_steer_modules(model)):
        if m.last_fusion_tensors is not None:
            out.append((i, *m.last_fusion_tensors))
    return out


def collect_fusion_stats(model) -> dict:
    """Mean of the per-layer fusion diagnostics of the last forward."""
    rows = [m.last_fusion_stats for m in iter_steer_modules(model) if m.last_fusion_stats]
    if not rows:
        return {}
    keys = set(rows[0])
    for r in rows:
        keys &= set(r)
    out = {k: float(sum(r[k] for r in rows) / len(rows)) for k in sorted(keys)}
    out["n_layers"] = len(rows)
    return out


def freeze_steer_keep_fusion(model):
    """Stage-1 setup: frozen backbone, frozen memory path, trainable fusion only."""
    for name, p in model.named_parameters():
        p.requires_grad_(is_fusion_param_name(name))
    return [n for n, p in model.named_parameters() if p.requires_grad]


def is_steer_param_name(name: str) -> bool:
    return any(mk in name for mk in _STEER_MARKERS)


def freeze_backbone_keep_steer(model):
    """Freeze everything except the prefix/memory/steer projections."""
    for name, p in model.named_parameters():
        p.requires_grad_(is_steer_param_name(name))
