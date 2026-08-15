"""Differential Extension (DEX) and the sign / adapter / attention-tuning controls.

Reference: "Understanding Differential Transformer Unchains Pretrained
Self-Attentions", arXiv:2505.16333 (NeurIPS 2025), Sec. 3, Eq. (3)-(5), Fig. 9,
App. B.3/B.4/E.1.  No official code has been released, so this is a from-paper
implementation.

Paper method (Eq. 5), per attention head ``h`` of a layer::

    O   = softmax(Q K^T / sqrt(d)) V
    O'  = O - lambda(t) * 1[h in H] * f_D(O)

``O'`` is concatenated across heads and passed to ``W_O``.  ``f_D`` is a
learnable projection parameterised by ``W_D`` (App. E.3: ``W_D in R^{dv x dv}``)
and Fig. 9's pseudocode hands the *same* ``f_D`` to every head of a layer, so
``W_D`` is per layer and shared across that layer's selected heads.

lambda annealing (Eq. 4)::

    lambda(t) = (1 - a) * (t / T) * lambda_init + a * lambda_learn,
    a         = min(1, t / T)

with ``lambda_learn`` initialised around zero and ``lambda_init`` taken from the
Differential Transformer depth-aware schedule, applied with the official
0-based layer index: ``0.8 - 0.6 * exp(-0.3 * layer_idx)`` (equivalently
``0.8 - 0.6 exp(-0.3 (l - 1))`` for 1-based ``l``), which App. B.3 reports as the
best setting.  Layer 0 gets 0.2.

The module below implements every experimental variant behind one code path so
that only the configuration differs between runs:

===================  =========================================  ==============================
variant              forward                                    trainable
===================  =========================================  ==============================
``base``             ``O' = O``                                 nothing
``dex_minus``        ``O' = O - lambda(t) f_D(O)``              W_K, W_V, W_O, W_D, lambda
``dex_plus``         ``O' = O + lambda(t) f_D(O)``              W_K, W_V, W_O, W_D, lambda
``residual_adapter`` ``O' = O + g_D(O)``  (lambda == 1, fixed)  W_K, W_V, W_O, W_D
``attn_only``        ``O' = O``                                 W_K, W_V, W_O
``adapter_only``     ``O' = O - lambda(t) f_D(O)``              W_D, lambda
===================  =========================================  ==============================

Note the intended equivalence: ``dex_plus`` and ``residual_adapter`` differ only
in whether the correction is scaled by the annealed/learned ``lambda(t)`` or by a
constant 1.  If ``dex_plus`` is run with ``--lambda-anneal-steps 0`` and a fixed
non-learnable ``lambda = 1`` the two become numerically identical; they are kept
separate because the paper's DEX always carries the lambda schedule.

Insertion point.  ``attn_output.reshape(B, T, H * Dh)`` -- the tensor fed to
``o_proj`` -- *is* the concatenation of the per-head ``O_h`` in head order, so
applying a per-head-block projection to it is exactly Eq. (5).  Wrapping
``o_proj`` therefore hits the Fig. 9 position without re-implementing (and
risking divergence from) the HF attention kernel.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch
import torch.nn as nn


VARIANTS = (
    "base",
    "dex_minus",
    "dex_plus",
    "ungated_adapter",
    "residual_adapter",   # deprecated alias of ungated_adapter (older artefacts)
    "attn_only",
    "adapter_only",
    "swa_steer",
    "diff_split",
    "lora",
)

# ``ungated_adapter`` is NOT a capacity-matched control for DEX: it differs in
# the lambda treatment (constant 1 instead of the annealed/learned schedule), so
# it also perturbs the pretrained model at full strength from step 0.  The
# matched ordinary-adapter control is ``dex_plus``, which shares the schedule,
# the parameter count and the initialisation distribution and differs only in
# the sign -- and the sign is a reparameterisation (W_D -> -W_D).
#
# ``swa_steer`` is the parameter-efficient counterpart of ``attn_only``.  The
# main-table result is that DEX's entire gain is the 566M-parameter finetune of
# W_K/W_V/W_O that it performs alongside its 0.6M adapter (``attn_only`` alone
# reaches it).  ``swa_steer`` asks whether that finetune is *necessary*: the
# backbone stays fully frozen and a sliding-window memory sidecar
# (``deltamem.core.prefix_steer``) is trained instead, steering the frozen
# attention additively via ``out = fuse(o_proj(attn), delta_o(reads))``.  It
# therefore shares ``attn_only``'s forward-path target (the attention output)
# while touching none of the pretrained weights.  Attaching the sidecar is the
# trainer's job; this variant only declares that the steer parameters -- and
# nothing in the backbone -- are the trainable set.

HEAD_SELECTIONS = ("entropy_high", "entropy_low", "importance_low", "all")


@dataclass(frozen=True)
class DexConfig:
    """One configuration object per run; the variant name resolves the rest."""

    variant: str = "dex_minus"
    # --- adapter ---------------------------------------------------------
    head_selection: str = "entropy_high"
    heads_per_layer: int = -1          # k; -1 => half of the heads (paper default)
    head_plan_path: str = ""           # JSON from scripts/dex_select_heads.py
    fd_bias: bool = False
    fd_init: str = "linear_default"    # linear_default | zeros
    fd_init_gain: float = 1.0
    # --- lambda ----------------------------------------------------------
    lambda_init_mode: str = "diff_depth"   # diff_depth | fixed
    lambda_init: float = 0.8               # used when lambda_init_mode == fixed
    lambda_learn_init: float = 0.0
    lambda_learnable: bool = True
    lambda_anneal_steps: int = 0           # T; 0 disables the Eq. (4) schedule
    # --- what is trained -------------------------------------------------
    train_attn: bool = True                # W_K, W_V, W_O (paper trains these)
    allow_no_anneal: bool = False          # opt in to a DEX run without Eq. (4)
    layers: tuple[int, ...] = field(default_factory=tuple)  # empty => every layer

    # Resolved by ``resolve``; not meant to be set by hand.
    sign: float = -1.0
    adapter_enabled: bool = True
    # Train the attached prefix/SWA memory sidecar instead of the backbone.  The
    # sidecar itself is attached by the trainer (attach_prefix_steer); this flag
    # only routes ``set_trainable``.  Mutually exclusive with ``train_attn``:
    # the point of the variant is that no pretrained weight is updated.
    train_steer: bool = False

    def resolve(self) -> "DexConfig":
        """Turn ``variant`` into the concrete switches, leaving everything else.

        Validation runs on the *resolved* config: ``adapter_enabled`` is a
        resolved field whose pre-resolution default is ``True``, so checking it
        first would reject perfectly fine ``base``/``attn_only`` configs.
        """
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}; pick from {VARIANTS}")
        if self.head_selection not in HEAD_SELECTIONS:
            raise ValueError(f"unknown head_selection {self.head_selection!r}")

        if self.variant == "base":
            resolved = replace(self, adapter_enabled=False, train_attn=False, sign=0.0)
        elif self.variant == "attn_only":
            resolved = replace(self, adapter_enabled=False, train_attn=True, sign=0.0)
        elif self.variant == "dex_minus":
            resolved = replace(self, adapter_enabled=True, train_attn=True, sign=-1.0)
        elif self.variant == "dex_plus":
            resolved = replace(self, adapter_enabled=True, train_attn=True, sign=+1.0)
        elif self.variant in ("ungated_adapter", "residual_adapter"):
            # O' = O + g_D(O): same parameter shape/init, no lambda modulation.
            resolved = replace(
                self,
                adapter_enabled=True,
                train_attn=True,
                sign=+1.0,
                lambda_learnable=False,
                lambda_anneal_steps=0,
                lambda_init_mode="fixed",
                lambda_init=1.0,
                lambda_learn_init=1.0,
            )
        elif self.variant == "adapter_only":
            resolved = replace(self, adapter_enabled=True, train_attn=False, sign=-1.0)
        elif self.variant == "swa_steer":
            # frozen backbone + trained memory sidecar; no DEX adapter at all, so
            # the forward is bit-identical to ``base`` until the sidecar is attached.
            resolved = replace(self, adapter_enabled=False, train_attn=False,
                               sign=0.0, train_steer=True)
        elif self.variant == "diff_split":
            # frozen backbone + post-hoc function-preserving differential head split.
            # No DEX adapter and no prefix_steer sidecar: the split module is attached
            # separately by the training entry, so the forward is bit-identical to
            # ``base`` until delta_q leaves zero.
            resolved = replace(self, adapter_enabled=False, train_attn=False, sign=0.0)
        elif self.variant == "lora":
            # frozen backbone + standard PEFT LoRA on the same 12 layers; adapters
            # are attached by the training entry.  B is zero-init, so the forward is
            # bit-identical to ``base`` until the first update, exactly like the
            # split and the additive sidecar.
            resolved = replace(self, adapter_enabled=False, train_attn=False, sign=0.0)
        else:  # pragma: no cover - guarded above
            raise AssertionError("unreachable")

        if resolved.train_steer and resolved.train_attn:
            raise ValueError(
                "train_steer trains a sidecar *instead of* the backbone; enabling "
                "train_attn as well would reintroduce the 566M finetune the variant "
                "is designed to avoid"
            )

        paper_style = resolved.variant in ("dex_minus", "dex_plus", "adapter_only")
        if paper_style and resolved.lambda_anneal_steps <= 0 and not resolved.allow_no_anneal:
            raise ValueError(
                "paper-style DEX needs lambda_anneal_steps > 0 (Eq. 4). Pass "
                "allow_no_anneal=True to run the no-annealing ablation on purpose."
            )
        if (
            resolved.adapter_enabled
            and resolved.fd_init == "zeros"
            and resolved.lambda_learn_init == 0.0
            and resolved.lambda_anneal_steps <= 0
        ):
            raise ValueError(
                "W_D=0 with lambda=0 and no annealing is a dead configuration: "
                "both gradients are identically zero."
            )
        return resolved


def diff_lambda_init(layer_idx: int) -> float:
    """Depth-aware lambda_init from Differential Transformer.

    ``layer_idx`` is **0-based**, matching the official implementation
    (microsoft/unilm ``Diff-Transformer/multihead_flashdiff_1.py``:
    ``lambda_init_fn(depth)`` with ``depth = current layer index``).  The DIFF
    paper writes the same schedule 1-based as ``0.8 - 0.6 exp(-0.3 (l - 1))``.
    Layer 0 therefore gets 0.2, not 0.3555.
    """
    return 0.8 - 0.6 * math.exp(-0.3 * layer_idx)


class AttentionOutputAdapter(nn.Module):
    """f_D / g_D applied to the per-head attention output ``O`` (Eq. 3/5).

    ``forward`` takes ``[B, T, H, Dh]`` and returns the same shape.  Heads that
    are not selected are returned bit-identical: the correction is multiplied by
    a 0/1 head mask *before* the addition, so a non-selected head receives
    exactly ``+0``.
    """

    def __init__(
        self,
        *,
        head_dim: int,
        num_heads: int,
        selected_heads: tuple[int, ...],
        sign: float,
        lambda_init: float,
        lambda_learn_init: float,
        lambda_learnable: bool,
        lambda_anneal_steps: int,
        fd_bias: bool = False,
        fd_init: str = "linear_default",
        fd_init_gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.num_heads = int(num_heads)
        self.sign = float(sign)
        self.lambda_init = float(lambda_init)
        self.lambda_anneal_steps = int(lambda_anneal_steps)
        self.selected_heads = tuple(sorted(int(h) for h in selected_heads))
        if any(h < 0 or h >= self.num_heads for h in self.selected_heads):
            raise ValueError("selected head index out of range")

        self.proj = nn.Linear(self.head_dim, self.head_dim, bias=fd_bias)
        if fd_init == "zeros":
            nn.init.zeros_(self.proj.weight)
        elif fd_init == "linear_default":
            if fd_init_gain != 1.0:
                with torch.no_grad():
                    self.proj.weight.mul_(fd_init_gain)
        else:
            raise ValueError(f"unknown fd_init {fd_init!r}")
        if fd_bias:
            nn.init.zeros_(self.proj.bias)

        lam0 = torch.tensor(float(lambda_learn_init), dtype=torch.float32)
        if lambda_learnable:
            self.lambda_learn = nn.Parameter(lam0)
        else:
            self.register_buffer("lambda_learn", lam0)
        self.lambda_learnable = bool(lambda_learnable)

        mask = torch.zeros(self.num_heads, dtype=torch.float32)
        for h in self.selected_heads:
            mask[h] = 1.0
        self.register_buffer("head_mask", mask, persistent=True)
        self.register_buffer("step", torch.zeros((), dtype=torch.long), persistent=True)

        # Diagnostics filled in by ``forward`` when enabled (no autograd graph).
        self.collect_stats = False
        self.last_stats: dict[str, float] = {}

    # -- lambda schedule ---------------------------------------------------
    def current_lambda(self) -> torch.Tensor:
        """lambda(t): Eq. (4) when annealing is on.

        With ``T == 0`` and a non-learnable lambda the value is pinned to this
        layer's ``lambda_init`` -- the "always-on differential branch" condition
        (and, for ``residual_adapter``, lambda_init == 1).  With ``T == 0`` and a
        learnable lambda it is simply ``lambda_learn``.
        """
        lam_learn = self.lambda_learn
        T = self.lambda_anneal_steps
        if T <= 0:
            if not self.lambda_learnable:
                return torch.as_tensor(
                    self.lambda_init, dtype=lam_learn.dtype, device=lam_learn.device
                )
            return lam_learn
        # tensor-only: .item() here would force a GPU->CPU sync per layer per
        # forward and would break torch.compile / CUDA graphs.
        ratio = self.step.to(dtype=lam_learn.dtype) / float(T)
        alpha = ratio.clamp(max=1.0)
        return (1.0 - alpha) * ratio * self.lambda_init + alpha * lam_learn

    @torch.no_grad()
    def set_step(self, step: int) -> None:
        self.step.fill_(int(step))

    def forward(self, heads: torch.Tensor) -> torch.Tensor:  # [B, T, H, Dh]
        if heads.shape[-1] != self.head_dim or heads.shape[-2] != self.num_heads:
            raise ValueError(
                f"expected [..., {self.num_heads}, {self.head_dim}], got {tuple(heads.shape)}"
            )
        lam = self.current_lambda()
        corr = self.proj(heads)
        mask = self.head_mask.to(corr.dtype).view(*([1] * (corr.dim() - 2)), -1, 1)
        corr = corr * mask
        raw = lam.to(corr.dtype) * corr          # Delta = lambda f_D(O), paper's Fig. 11
        out = heads + self.sign * raw           # O' = O -/+ Delta
        if self.collect_stats:
            with torch.no_grad():
                raw_f = raw.float()             # Delta  (unsigned, as in Fig. 11)
                delta = (self.sign * raw).float()   # the signed update actually applied
                base = heads.float()
                dn = delta.norm().item()
                bn = base.norm().item()
                sel = self.head_mask.bool()
                dsel = delta[..., sel, :]
                bsel = base[..., sel, :]
                rsel = raw_f[..., sel, :]
                cos_fn = torch.nn.functional.cosine_similarity
                cos_update = cos_fn(
                    dsel.reshape(-1, self.head_dim), bsel.reshape(-1, self.head_dim), dim=-1
                )
                cos_raw = cos_fn(
                    rsel.reshape(-1, self.head_dim), bsel.reshape(-1, self.head_dim), dim=-1
                )
                self.last_stats = {
                    "lambda": float(lam.item()),
                    "corr_norm": dn,
                    "out_norm": bn,
                    "corr_ratio": dn / max(bn, 1e-12),
                    "corr_ratio_selected": dsel.norm().item() / max(bsel.norm().item(), 1e-12),
                    "corr_mean": float(delta.mean().item()),
                    "corr_std": float(delta.std().item()),
                    # cos(O, Delta) with Delta = lambda f_D(O): comparable to Fig. 11
                    "cos_raw_corr_o": float(cos_raw.mean().item()),
                    # cos(O, signed update) = -cos_raw_corr_o for dex_minus
                    "cos_delta_o": float(cos_update.mean().item()),
                    "n_selected_heads": int(sel.sum().item()),
                }
        return out


class DexOutputProjection(nn.Module):
    """``o_proj`` wrapper that applies the adapter to the concatenated heads.

    ``x`` arrives as ``[B, T, H * Dh]`` (the concat of the per-head ``O_h``); it
    is viewed as ``[B, T, H, Dh]``, adapted, flattened back and handed to the
    frozen/trainable base ``o_proj``.
    """

    def __init__(self, base: nn.Linear, adapter: AttentionOutputAdapter | None,
                 num_heads: int, head_dim: int) -> None:
        super().__init__()
        if base.in_features != int(num_heads) * int(head_dim):
            raise ValueError(
                f"o_proj expects in_features={base.in_features}, but "
                f"num_heads*head_dim={int(num_heads) * int(head_dim)}"
            )
        self.base = base
        self.adapter = adapter
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)

    # keep duck-typing with nn.Linear for callers that introspect o_proj
    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adapter is not None:
            shape = x.shape
            # reshape, not view: some attention backends return non-contiguous O
            heads = x.reshape(*shape[:-1], self.num_heads, self.head_dim)
            heads = self.adapter(heads)
            x = heads.reshape(*shape)
        return self.base(x)


def _text_config(model):
    cfg = model.config
    return cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg


def load_head_plan(path: str | Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def select_heads_for_layer(
    layer_idx: int,
    num_heads: int,
    cfg: DexConfig,
    plan: dict | None,
) -> tuple[int, ...]:
    """Resolve which heads of a layer receive the differential adaptation."""
    k = cfg.heads_per_layer if cfg.heads_per_layer > 0 else num_heads // 2
    k = max(0, min(k, num_heads))
    if cfg.head_selection == "all":
        return tuple(range(num_heads))
    if plan is None:
        raise ValueError(
            f"head_selection={cfg.head_selection!r} needs a head plan "
            "(scripts/dex_select_heads.py)"
        )
    want = "entropy" if cfg.head_selection.startswith("entropy") else "importance"
    got = plan.get("criterion")
    if got != want and want not in plan:
        raise ValueError(
            f"head plan has criterion {got!r} and no {want!r} block, but "
            f"head_selection={cfg.head_selection!r} needs {want!r} scores"
        )
    # prefer the explicitly named block so an entropy plan can never be read as
    # importance scores (and vice versa).  NB: ``plan.get(want, plan["scores"])``
    # would evaluate the default eagerly and raise KeyError on a plan that has
    # only the named block.
    if want in plan:
        table = plan[want]
    elif "scores" in plan:
        table = plan["scores"]
    else:
        raise KeyError(f"head plan contains neither {want!r} nor 'scores'")
    scores = table[str(layer_idx)]
    if len(scores) != num_heads:
        raise ValueError(
            f"head plan for layer {layer_idx} has {len(scores)} heads, model has {num_heads}"
        )
    order = sorted(range(num_heads), key=lambda h: scores[h])
    if cfg.head_selection in ("entropy_low", "importance_low"):
        chosen = order[:k]
    elif cfg.head_selection == "entropy_high":
        chosen = order[::-1][:k]
    else:  # pragma: no cover - guarded in resolve()
        raise ValueError(cfg.head_selection)
    return tuple(sorted(chosen))


def attach_dex(model, cfg: DexConfig, plan: dict | None = None) -> dict:
    """Wrap every targeted attention's ``o_proj``; returns an attachment report."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    cfg = cfg.resolve()
    if plan is None and cfg.head_plan_path:
        plan = load_head_plan(cfg.head_plan_path)
    text_cfg = _text_config(model)
    num_heads = text_cfg.num_attention_heads
    head_dim = getattr(text_cfg, "head_dim", None) or (
        text_cfg.hidden_size // num_heads
    )
    layers = set(cfg.layers) if cfg.layers else None

    report = {
        "variant": cfg.variant,
        "adapter_enabled": cfg.adapter_enabled,
        "sign": cfg.sign,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "layers": [],
        "selected_heads": {},
        "lambda_init": {},
    }
    for name, module in list(model.named_modules()):
        if not isinstance(module, Qwen3Attention):
            continue
        li = module.layer_idx
        if layers is not None and li not in layers:
            continue
        if isinstance(module.o_proj, DexOutputProjection):
            raise RuntimeError(f"DEX already attached to layer {li}")
        adapter = None
        if cfg.adapter_enabled:
            heads = select_heads_for_layer(li, num_heads, cfg, plan)
            lam_init = (
                diff_lambda_init(li)
                if cfg.lambda_init_mode == "diff_depth"
                else float(cfg.lambda_init)
            )
            adapter = AttentionOutputAdapter(
                head_dim=head_dim,
                num_heads=num_heads,
                selected_heads=heads,
                sign=cfg.sign,
                lambda_init=lam_init,
                lambda_learn_init=cfg.lambda_learn_init,
                lambda_learnable=cfg.lambda_learnable,
                lambda_anneal_steps=cfg.lambda_anneal_steps,
                fd_bias=cfg.fd_bias,
                fd_init=cfg.fd_init,
                fd_init_gain=cfg.fd_init_gain,
            ).to(device=module.o_proj.weight.device)
            # only W_D follows the backbone dtype; lambda stays fp32 (a bf16
            # scalar near 0.75 has eps ~ 4e-3 and would swallow small updates)
            adapter.proj.to(dtype=module.o_proj.weight.dtype)
            report["selected_heads"][str(li)] = list(heads)
            report["lambda_init"][str(li)] = lam_init
        module.o_proj = DexOutputProjection(module.o_proj, adapter, num_heads, head_dim)
        report["layers"].append(li)
    if not report["layers"]:
        raise RuntimeError("no attention modules matched for DEX")
    return report


def iter_dex_modules(model):
    for m in model.modules():
        if isinstance(m, DexOutputProjection):
            yield m


def iter_dex_adapters(model):
    for m in iter_dex_modules(model):
        if m.adapter is not None:
            yield m.adapter


def set_dex_step(model, step: int) -> None:
    for a in iter_dex_adapters(model):
        a.set_step(step)


def set_dex_stats(model, flag: bool) -> None:
    for a in iter_dex_adapters(model):
        a.collect_stats = bool(flag)


def collect_dex_stats(model) -> dict:
    """Aggregate the per-layer adapter diagnostics of the last forward."""
    keys = (
        "lambda", "corr_norm", "out_norm", "corr_ratio", "corr_ratio_selected",
        "corr_mean", "corr_std", "cos_delta_o", "cos_raw_corr_o",
    )
    rows = [a.last_stats for a in iter_dex_adapters(model) if a.last_stats]
    if not rows:
        return {}
    out = {k: float(sum(r[k] for r in rows) / len(rows)) for k in keys}
    out["n_layers"] = len(rows)
    return out


def is_dex_param_name(name: str) -> bool:
    return ".o_proj.adapter." in name


def is_attn_trainable_param_name(name: str) -> bool:
    """W_K, W_V, W_O -- the self-attention parameters the paper keeps training."""
    if ".self_attn." not in name:
        return False
    return (
        ".k_proj.weight" in name
        or ".k_proj.bias" in name
        or ".v_proj.weight" in name
        or ".v_proj.bias" in name
        or ".o_proj.base.weight" in name
        or ".o_proj.base.bias" in name
        or (".o_proj.weight" in name and ".o_proj.base." not in name)
    )


def set_trainable(model, cfg: DexConfig) -> list[str]:
    """Freeze everything, then re-enable exactly this variant's trainable set.

    Walks the attention modules instead of matching parameter names, so a layer
    subset (``cfg.layers``) restricts the unfrozen ``W_K/W_V/W_O`` to exactly the
    layers that carry an adapter.  Name matching alone would unfreeze every
    layer's attention and silently invalidate any layer ablation.
    """
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    cfg = cfg.resolve()
    target_layers = set(cfg.layers) if cfg.layers else None
    for p in model.parameters():
        p.requires_grad_(False)

    for module in model.modules():
        if not isinstance(module, Qwen3Attention):
            continue
        if target_layers is not None and module.layer_idx not in target_layers:
            continue
        o_proj = module.o_proj
        if cfg.train_attn:
            base_o = o_proj.base if isinstance(o_proj, DexOutputProjection) else o_proj
            for proj in (module.k_proj, module.v_proj, base_o):
                for p in proj.parameters():
                    p.requires_grad_(True)
        if cfg.adapter_enabled and isinstance(o_proj, DexOutputProjection) and o_proj.adapter is not None:
            for p in o_proj.adapter.parameters():
                p.requires_grad_(True)

    if cfg.train_steer:
        # The sidecar lives on the PrefixMemSteerAttention wrapper, not inside
        # Qwen3Attention, so it needs its own pass.  Which layers carry a sidecar
        # is decided at attach time by PrefixSteerConfig.steer_layers; every
        # parameter that exists here is by construction one we mean to train.
        from deltamem.core.prefix_steer import is_steer_param_name
        n_steer = 0
        for name, p in model.named_parameters():
            if is_steer_param_name(name):
                p.requires_grad_(True)
                n_steer += 1
        if n_steer == 0:
            raise RuntimeError(
                "train_steer is set but the model carries no steer parameters; "
                "attach_prefix_steer must run before set_trainable"
            )

    return [n for n, p in model.named_parameters() if p.requires_grad]


def trainable_report(model) -> dict:
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and is_dex_param_name(n)
    )
    try:
        from deltamem.core.prefix_steer import is_steer_param_name
    except Exception:                                   # sidecar module optional
        steer = 0
    else:
        steer = sum(
            p.numel() for n, p in model.named_parameters()
            if p.requires_grad and is_steer_param_name(n)
        )
    return {
        "trainable_param_count": int(count),
        "adapter_param_count": int(adapter),
        # backbone attention only: the sidecar is counted separately so a
        # swa_steer run does not report 11.8M of steer as "attn"
        "attn_param_count": int(count - adapter - steer),
        "steer_param_count": int(steer),
        "trainable_param_names_head": names[:8],
        "trainable_tensor_count": len(names),
    }
