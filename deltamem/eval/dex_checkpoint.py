"""Complete sparse checkpoints: restore every changed weight and algorithm buffer."""
from __future__ import annotations

from dataclasses import fields

from deltamem.core.dex import DexConfig, is_dex_param_name, set_trainable
from deltamem.core.prefix_steer import is_steer_state_name


def restore_dex_config(raw):
    known = {f.name for f in fields(DexConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown DEX config fields: {sorted(unknown)}")
    values = dict(raw)
    values["layers"] = tuple(values.get("layers", ()))
    return DexConfig(**values).resolve()


def required_dex_state_names(model, config):
    # set_trainable restores the declared training contract and respects layer subsets.
    names = set(set_trainable(model, config))
    names.update(n for n in model.state_dict()
                 if is_dex_param_name(n) or (config.train_steer and is_steer_state_name(n)))
    return names


def dex_state_dict(model, config):
    names = required_dex_state_names(model, config)
    return {n: value.detach().cpu().clone() for n, value in model.state_dict().items()
            if n in names}


def load_dex_state_strict(model, state, config):
    required = required_dex_state_names(model, config)
    missing = sorted(required - set(state))
    if missing:
        raise RuntimeError(f"incomplete DEX checkpoint: {len(missing)} required weights/buffers "
                           f"missing, including {missing[:5]}; adapter-only files cannot "
                           "restore an attention-finetuned model")
    params = dict(model.named_parameters())
    params.update(model.named_buffers())
    for name, value in state.items():
        if name in params and value.is_floating_point():
            params[name].data = params[name].data.to(dtype=value.dtype)
    _, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"DEX checkpoint/config mismatch: {unexpected[:5]}")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
