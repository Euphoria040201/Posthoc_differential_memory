"""Strict, backwards-compatible prefix-steer checkpoint restoration for eval.

Checkpoint ``cfg`` dictionaries are architecture manifests, not a collection of
optional inference hints.  In particular, a non-parameterized switch can change
the computation without producing an ``unexpected`` state-dict key.  Restore
every known field explicitly and keep audited legacy defaults here so old
checkpoints never inherit a newer dataclass default by accident.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping

from deltamem.core.prefix_steer import (
    PrefixSteerConfig,
    is_steer_param_name,
)


_REQUIRED_LEGACY_FIELDS = frozenset(
    {
        "num_prefix_tokens",
        "sliding_window_size",
        "mem_num_heads",
        "mem_head_dim",
        "steer_mode",
        "normal_attends_prefix",
        "prefix_sees_query",
    }
)

# These are the semantics of a checkpoint created before each field existed.
# Some intentionally differ from today's PrefixSteerConfig defaults.
_LEGACY_DEFAULTS: dict[str, Any] = {
    "prefix_init_std": 0.02,
    "prefix_init_dist": "normal",
    "prefix_write": False,
    "read_prefix_only": False,
    "memory_mode": "residual",
    "write_ctx_only": False,
    "prefix_write_layout": "global",
    "prefix_write_overlap_tokens": 0,
    "pool_reads": False,
    "pool_gate": False,
    "pool_gate_input": "rm",
    "pool_gate_max": 1.0,
    "pool_gate_bias": 0.0,
    "steer_layers": (),
    "prefix_layers": (),
    "steer_gain": 1.0,
    "share_qkv": False,
    "memory_value_source": "trainable",
    "delta_heads": "qkvo",
    "delta_rank": 0,
    "read_proj_dim": 0,
    "output_fusion": "fixed",
    "output_fusion_eps": 1e-6,
    "output_fusion_scale_max": 10.0,
    "history_pool_mode": "none",
    "hybrid_read_mode": "none",
    "hybrid_prefix_gate_mode": "fixed",
    "hybrid_prefix_gate_init": 0.1,
    # Differential-fusion knobs.  Every checkpoint written before they existed is
    # an ADDITIVE run (output_fusion in {"fixed","rms_match","cosine"}), and these
    # three fields are read only by the differential branches, so no pre-existing
    # checkpoint's computation can depend on them.  They are audited here with the
    # current dataclass defaults purely to satisfy the sync guard.
    "fusion_lambda_init": 0.1,
    "fusion_lambda_max": 1.0,
    "fusion_ema_momentum": 0.99,
}


def restore_prefix_steer_config(
    raw_cfg: Mapping[str, Any],
    *,
    steer_gain_override: float | None = None,
    pool_reads_override: bool | None = None,
) -> PrefixSteerConfig:
    """Restore one exact architecture, with explicit old-checkpoint semantics."""

    if not isinstance(raw_cfg, Mapping):
        raise ValueError("checkpoint cfg must be a mapping")
    known_fields = {field.name for field in fields(PrefixSteerConfig)}
    audited_fields = _REQUIRED_LEGACY_FIELDS | set(_LEGACY_DEFAULTS)
    unaudited = sorted(known_fields - audited_fields)
    stale_defaults = sorted(audited_fields - known_fields)
    if unaudited or stale_defaults:
        raise RuntimeError(
            "eval checkpoint loader is out of sync with PrefixSteerConfig: "
            f"unaudited={unaudited}, removed={stale_defaults}"
        )

    unknown = sorted(set(raw_cfg) - known_fields)
    if unknown:
        raise ValueError(
            "checkpoint cfg contains fields unknown to this evaluator/core: "
            f"{unknown}"
        )
    missing = sorted(_REQUIRED_LEGACY_FIELDS - set(raw_cfg))
    if missing:
        raise ValueError(
            f"checkpoint cfg is missing required legacy fields: {missing}"
        )

    values = dict(_LEGACY_DEFAULTS)
    values.update({key: raw_cfg[key] for key in raw_cfg})
    for name in ("steer_layers", "prefix_layers"):
        values[name] = tuple(values.get(name) or ())
    if steer_gain_override is not None:
        values["steer_gain"] = steer_gain_override
    if pool_reads_override is not None:
        values["pool_reads"] = pool_reads_override
    return PrefixSteerConfig(**values)


def load_steer_state_strict(
    model: Any,
    state: Mapping[str, Any],
    *,
    label: str = "ours",
) -> None:
    """Load steer tensors and reject both dropped and randomly missing tensors."""

    if not isinstance(state, Mapping):
        raise ValueError("checkpoint state must be a tensor mapping")
    _, unexpected = model.load_state_dict(state, strict=False)
    steer_names = {
        name
        for name, _ in model.named_parameters()
        if is_steer_param_name(name)
    }
    missing_steer = sorted(steer_names - set(state))
    dropped = sorted(str(name) for name in unexpected)
    if missing_steer:
        raise RuntimeError(
            f"[{label}] {len(missing_steer)} steer params are NOT in the ckpt "
            f"and would stay at random init: {missing_steer[:4]}"
        )
    if dropped:
        raise RuntimeError(
            f"[{label}] {len(dropped)} ckpt tensors were NOT loaded: "
            f"{dropped[:4]} (cfg mismatch)"
        )


def has_persistent_history_writer(config: PrefixSteerConfig) -> bool:
    """Whether a separate context WRITE pass can create persistent memory."""

    return (
        config.history_pool_mode != "none"
        or (config.num_prefix_tokens > 0 and config.prefix_write)
    )
